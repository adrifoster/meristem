"""Class for managing model ensembles"""

from __future__ import annotations

import os
import subprocess
import json
import shlex
from pathlib import Path
from dataclasses import replace
import pandas as pd

from meristem.commands import meristem_cmd
from meristem.ensemble.config import EnsembleConfig
from meristem.run.stages import Stage
from meristem.run.model_run import ModelRun

# state file name
_STATE_FILE = "ensemble_state.json"

# parameter file extensions
_PARAM_EXTENSIONS = (".nc", ".json")


class Ensemble:
    """Class for model ensemble management

    Use Ensemble.create() to initialize a new ensemble on disk, or
    Ensemble.load() to resume an existing one.

    Attributes
    ===========
    config: EnsembleConfig
        Ensemble config with all attributes
    param_files: dict[str, Path]
        dictionary mapping of member string to full path to parameter file
    """

    def __init__(
        self, config: EnsembleConfig, param_files: dict[str, Path] | None = None
    ):
        self.config = config
        self.param_files: dict[str, Path] = param_files or {}

    @property
    def root(self) -> Path:
        """Directory for ensemble and member state, PBS scripts and logs"""
        return self.config.root

    @property
    def base_case(self) -> Path:
        """Directory with built CIME case that every member is cloned from"""
        return self.config.base_case

    @classmethod
    def create(cls, config: EnsembleConfig | dict | Path | str) -> Ensemble:
        """Initialize a brand-new ensemble from a YAML file, config dict, or
        EnsembleConfig.

        Creates root on disk and writes the initial state file.
        Call this once; use load for all subsequent access.

        Args:
            config: path to an ensemble YAML file, a dict with the same schema,
                or an EnsembleConfig

        Returns:
            Ensemble: Ensemble instance
        Raises:
            ValueError: if required config keys are missing or invalid
            TypeError: if config is not an EnsembleConfig, dict, str, or Path
            FileExistsError: if a state file already exists at the target root,
            indicating that the ensemble has already been initialized
        """
        ensemble = cls(_to_config(config))

        state_path = ensemble.root / _STATE_FILE
        if state_path.exists():
            raise FileExistsError(
                f"A state file already exists at {state_path}. "
                "Use Ensemble.load() to resume an existing ensemble."
            )

        ensemble.param_files = ensemble.find_param_files()
        ensemble.check_base_case()
        ensemble.check_stage_scripts()
        member_runs = ensemble._member_runs()

        ensemble.root.mkdir(parents=True, exist_ok=True)
        for run in member_runs:
            run.root.mkdir(parents=True, exist_ok=True)
            run.save()
        ensemble.save()
        return ensemble

    @classmethod
    def load(cls, root: Path | str) -> Ensemble:
        """Load an existing ensemble from its state file.

        Args:
            root (Path | str): Path to root

        Returns:
            Ensemble: Ensemble instance
        Raises:
            FileNotFoundError: if no state file exists at root
        """
        state_path = Path(root) / _STATE_FILE
        if not state_path.exists():
            raise FileNotFoundError(
                f"No state file found at {state_path}. "
                "Use Ensemble.create() to initialise a new ensemble."
            )
        state = json.loads(state_path.read_text(encoding="utf-8"))
        param_files = {member: Path(path) for member, path in state["members"].items()}
        return cls(EnsembleConfig.from_dict(state["config"]), param_files)

    def save(self):
        """Atomically write current state to ``<root>/ensemble_state.json``.

        Uses a write-to-temp-then-rename pattern so the state file is
        never left in a partially written state if the process is killed.
        """
        state = {
            "config": self.config.to_dict(),
            "members": {member: str(path) for member, path in self.param_files.items()},
        }
        tmp = self.root / f"{_STATE_FILE}.tmp"
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.rename(self.root / _STATE_FILE)

    def find_param_members(self) -> list[str]:
        """Get the list of parameter members that will be used in this ensemble

        Raises:
            FileNotFoundError: Can't find the member list file
            ValueError: member list file is empty
            ValueError: member list file contains duplicates
            ValueError: one or more requested members not in member list file

        Returns:
            list[str]: list of requested members that exist in the member list file
        """

        # read in parameter file list
        list_file = self.config.paramgen_dir / f"{self.config.paramgen_prefix}.txt"
        if not list_file.is_file():
            raise FileNotFoundError(
                f"member list {list_file} not found. "
                f"Has meristem paramgen been run with prefix {self.config.paramgen_prefix!r}?"
            )

        # get all members
        all_members = [
            line.strip()
            for line in list_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not all_members:
            raise ValueError(f"member list {list_file} is empty.")

        # check for duplicates
        duplicates = sorted({m for m in all_members if all_members.count(m) > 1})
        if duplicates:
            raise ValueError(f"member list {list_file} has duplicates: {duplicates}")

        # subset to intended members
        if self.config.members is None:
            return all_members

        unknown = sorted(set(self.config.members) - set(all_members))
        if unknown:
            raise ValueError(f"requested members not in {list_file}: {unknown}")
        wanted = set(self.config.members)
        return [m for m in all_members if m in wanted]

    def find_param_files(self) -> dict[str, Path]:
        """Find parameter files associated with the member list

        Raises:
            FileNotFoundError: can't find the key file
            ValueError: key file has no ensemble column
            FileNotFoundError: can't find a parameter file or member not on key_file

        Returns:
            dict[str, Path]: dictionary with member name and path to parameter file
        """

        selected = self.find_param_members()

        # read in key file
        key_file = self.config.paramgen_dir / f"{self.config.paramgen_prefix}_key.csv"
        if not key_file.is_file():
            raise FileNotFoundError(
                f"key file {key_file} not found. "
                f"Has meristem paramgen been run with prefix {self.config.paramgen_prefix!r}?"
            )
        # read as str so all-digit member names (e.g. "0001") aren't parsed as ints
        key = pd.read_csv(key_file, dtype=str)
        if "ensemble" not in key.columns:
            raise ValueError(
                f"key file {key_file} has no 'ensemble' column "
                f"(columns: {list(key.columns)})."
            )
        key_members = set(key["ensemble"])

        # check to make sure parameter files exist
        param_files: dict[str, Path] = {}
        problems: list[str] = []
        for member in selected:
            if member not in key_members:
                problems.append(f"  {member}: {member} not in {key_file}")
                continue
            found = [
                self.config.paramgen_dir / f"{member}{ext}"
                for ext in _PARAM_EXTENSIONS
                if (self.config.paramgen_dir / f"{member}{ext}").is_file()
            ]
            if len(found) == 1:
                param_files[member] = found[0]
            elif not found:
                problems.append(f"  {member}: no {member}.nc or {member}.json")
            else:
                problems.append(f"  {member}: both .nc and .json exist")

        if problems:
            raise FileNotFoundError(
                f"parameter files missing or ambiguous in {self.config.paramgen_dir}:\n"
                + "\n".join(problems)
            )
        return param_files

    def check_base_case(self):
        """Check to make sure the supplied base case exists and can be cloned

        Raises:
            FileNotFoundError: Can't find the base case directory
            FileNotFoundError: Can't find the xmlquery script inside the base case dir
            PermissionError: xmlquery isn't executable
            RuntimeError: xmlquery exited with an error
            ValueError: BUILD_COMPLETE didn't work or returned FALSE
        """
        if not self.base_case.is_dir():
            raise FileNotFoundError(f"cannot find base case {self.base_case}")

        xmlquery = self.base_case / "xmlquery"
        if not xmlquery.is_file():
            raise FileNotFoundError(
                f"{self.base_case} has no xmlquery; is it a CIME case directory?"
            )
        if not os.access(xmlquery, os.X_OK):
            raise PermissionError(f"{xmlquery} is not executable.")

        result = subprocess.run(
            ["./xmlquery", "BUILD_COMPLETE", "--value"],
            cwd=self.base_case,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"xmlquery BUILD_COMPLETE failed in {self.base_case} "
                f"(exit {result.returncode}): "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
        if result.stdout.strip().upper() != "TRUE":
            raise ValueError(
                f"base case {self.base_case} is not built (BUILD_COMPLETE="
                f"{result.stdout.strip()!r}); run ./case.build first."
            )

    def check_stage_scripts(self):
        """Check that every member stage's script exists

        Raises:
            FileNotFoundError: if any stage's script is missing
        """
        missing = [
            f"  {stage.name}: {stage.script}"
            for stage in self.config.member_stages
            if stage.script is not None and not stage.script.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "member stage script(s) not found:\n" + "\n".join(missing)
            )

    def _member_stages(self, member: str) -> list[Stage]:
        """Build a member's stages: each member stage, in its own case under
        case_dir, running ``meristem ensemble prepare`` then its script (if any)

        Args:
            member (str): member name

        Returns:
            list[Stage]: the member's stages, in member_stages order
        """
        stages = []
        for stage_config in self.config.member_stages:
            command = shlex.join(
                [
                    *shlex.split(meristem_cmd()),
                    "ensemble",
                    "prepare",
                    "--root",
                    str(self.root),
                    "--member",
                    member,
                    "--stage",
                    stage_config.name,
                ]
            )
            case_root = (
                self.config.case_dir
                / f"{self.config.base_case.name}_{member}_{stage_config.name}"
            )
            # the job runs prepare (clone the base case, set the parameter
            # file), then the stage's own script, if any, on the clone
            stages.append(
                Stage(
                    config=replace(stage_config, command=command, case_root=case_root)
                )
            )
        return stages

    def _member_runs(self) -> list[ModelRun]:
        """Build and validate (without writing) a ModelRun for every member

        Raises:
            ValueError: if a member's run is invalid (e.g. its run_id has
            characters ModelRun doesn't allow, or case_dir is relative)

        Returns:
            list[ModelRun]: one run per member, in param_files order
        """
        runs = []
        for member in self.param_files:
            run = ModelRun(
                root=self.root / "members" / member,
                run_id=f"{self.config.ensemble_id}_{member}",
                stages=self._member_stages(member),
                email=self.config.email or None,
                project=self.config.project,
            )
            run._validate_stages()  # pylint: disable=protected-access
            runs.append(run)
        return runs


def _to_config(config: EnsembleConfig | dict | Path | str) -> EnsembleConfig:
    """Normalise *config* to an EnsembleConfig.

    Args:
        config: an EnsembleConfig, a dict, or a path to a YAML file.

    Returns:
        EnsembleConfig: the validated config

    Raises:
        TypeError: if *config* is not an EnsembleConfig, dict, str, or Path.
    """
    if isinstance(config, EnsembleConfig):
        return config
    if isinstance(config, dict):
        return EnsembleConfig.from_dict(config)
    if isinstance(config, (str, Path)):
        return EnsembleConfig.from_yaml(config)
    raise TypeError(
        "config must be an EnsembleConfig, dict, str, or Path, "
        f"got {type(config).__name__!r}."
    )
