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

    @property
    def case_dir(self) -> Path:
        """Directory where member clones are created"""
        return self.config.case_dir

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

    def submit(self, limit: int | None = None, dry_run: bool = False) -> dict[str, str]:
        """Submit the first stage of every pending member"""
        runs = {member: self.member_run(member) for member in self.param_files}

    def prepare(self, member: str, stage: str):
        """Clone and configure one member's case

        Args:
            member (str): member string
            stage (str): stage string name

        Raises:
            ValueError: unknown member stage, or base case not built
            FileNotFoundError: missing base case, parameter file, member run,
                or CIME script (xmlquery, create_clone, case.setup)
            PermissionError: a CIME script isn't executable
            FileExistsError: the stage's case_root exists but isn't its complete clone
            RuntimeError: a CIME command (create_clone, case.setup) failed
        """

        if stage not in [s.name for s in self.config.member_stages]:
            raise ValueError(
                f"unknown member stage {stage!r}; member stages are "
                f"{[s.name for s in self.config.member_stages]}."
            )

        self.check_base_case()
        self.check_param_files([member])
        run = self.member_run(member)
        case_root = self._case_root(member, stage)

        # a stage that's re-run (e.g. retried after a crash) reuses its clone,
        # which already has its PROJECT from create_clone
        if not _is_existing_case(case_root):
            self._clone(case_root, run.project)

        # safe to re-run: CIME skips setup that's already been done
        self._case_setup(case_root)

    def member_run(self, member: str) -> ModelRun:
        """Load one member's ModelRun"""
        return ModelRun.load(self.root / "members" / member)

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

        _check_case_script("xmlquery", self.base_case)

        result = _run(["./xmlquery", "BUILD_COMPLETE", "--value"], self.base_case)

        if result.upper() != "TRUE":
            raise ValueError(
                f"base case {self.base_case} is not built (BUILD_COMPLETE="
                f"{result!r}); run ./case.build first."
            )

    def check_param_files(self, members: list[str] | None = None):
        """Check that the stored parameter files still exist

        Args:
            members (list[str] | None, optional): members to check, or None for all.
                Defaults to None.

        Raises:
            FileNotFoundError: if any of the checked members' parameter files is missing
        """
        to_check = (
            self.param_files
            if members is None
            else {m: self.param_files[m] for m in members}
        )
        missing = [f"  {m}: {p}" for m, p in to_check.items() if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "parameter file(s) no longer exist:\n" + "\n".join(missing)
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
        case_dir, running ``meristem ensemble prepare`` then its script (if any),
        then ``meristem ensemble finalize``

        Args:
            member (str): member name

        Returns:
            list[Stage]: the member's stages, in member_stages order
        """
        stages = []
        for stage_config in self.config.member_stages:
            command = self._member_command("prepare", member, stage_config.name)
            post_command = self._member_command("finalize", member, stage_config.name)

            case_root = self._case_root(member, stage_config.name)
            # the job runs prepare (clone the base case, set the parameter
            # file), then the stage's own script, if any, on the clone, then finalize
            stages.append(
                Stage(
                    config=replace(
                        stage_config,
                        command=command,
                        post_command=post_command,
                        case_root=case_root,
                    )
                )
            )
        return stages

    def _member_command(self, action: str, member: str, stage: str) -> str:
        """Creates an ensemble member command

        Args:
            action (str): action (meristem command)
            member (str): member string name
            stage (str): member stage's string name

        Returns:
            str: command
        """
        command = shlex.join(
            [
                *shlex.split(meristem_cmd()),
                "ensemble",
                action,
                "--root",
                str(self.root),
                "--member",
                member,
                "--stage",
                stage,
            ]
        )
        return command

    def _case_root(self, member: str, stage: str) -> Path:
        """Case directory for one member stage's clone

        Args:
            member (str): member name string
            stage (str): stage string

        Returns:
            Path: path to case directory for member stage's clone
        """
        return self.config.case_dir / f"{self.config.base_case.name}_{member}_{stage}"

    def _case_setup(self, case_root: Path):
        """Run case.setup in a member case

        Args:
            case_root (Path): path to member's case root

        Raises:
            FileNotFoundError: case_root has no case.setup
            PermissionError: case.setup isn't executable
            RuntimeError: case.setup failed
        """
        _check_case_script("case.setup", case_root)
        try:
            _run(["./case.setup"], case_root)
        except RuntimeError as exc:
            raise RuntimeError(f"case.setup failed: {exc}") from exc

    def _clone(self, case_root: Path, project: str):
        """Clone a single case at case_root from base_case

        Args:
            case_root (Path): path to the member's case_root
            project (str): PROJECT to set on the clone (create_clone --project)

        Raises:
            FileNotFoundError: no executable create_clone found
            RuntimeError: case_root failed
            RuntimeError: case_root passed but something else went wrong and the case
            was not successfully created
        """

        cimeroot = Path(_run(["./xmlquery", "CIMEROOT", "--value"], self.base_case))
        create_clone = cimeroot / "scripts" / "create_clone"
        if not create_clone.is_file() or not os.access(create_clone, os.X_OK):
            raise FileNotFoundError(
                f"no executable create_clone at {create_clone} (CIMEROOT of base "
                f"case {self.base_case}); has CIME moved since the base case was built?"
            )

        cmd = [
            str(create_clone),
            "--case",
            str(case_root),
            "--clone",
            str(self.base_case),
            "--keepexe",
            "--project",
            project,
        ]
        if self.config.output_root:
            cmd.extend(["--cime-output-root", str(self.config.output_root)])

        self.case_dir.mkdir(parents=True, exist_ok=True)
        try:
            _run(cmd, self.base_case)
        except RuntimeError as exc:
            leftover = (
                f" It left a partial {case_root}; remove it before retrying."
                if case_root.exists()
                else ""
            )
            raise RuntimeError(
                f"cloning {self.base_case} failed: {exc}.{leftover}"
            ) from exc

        if not _is_existing_case(case_root):
            raise RuntimeError(
                f"create_clone exited successfully but {case_root} was not created."
            )

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


def _check_case_script(script: str, case_dir: Path) -> Path:
    """Check that a CIME case script (e.g. xmlquery, case.setup) exists in a
    case directory and is executable

    Args:
        script (str): script name
        case_dir (Path): path to case dir

    Raises:
        FileNotFoundError: can't find script
        PermissionError: isn't executable

    Returns:
        Path: path to the script
    """
    path = case_dir / script
    if not path.is_file():
        raise FileNotFoundError(
            f"{case_dir} has no {script}; is it a CIME case directory?"
        )
    if not os.access(path, os.X_OK):
        raise PermissionError(f"{path} is not executable.")

    return path


def _is_existing_case(case_root: Path) -> bool:
    """Check whether a member stage's case directory already holds its clone

    Args:
        case_root (Path): member stage case directory

    Raises:
        FileExistsError: if case_root exists but isn't a complete CIME case
        (e.g. a clone that died partway); it's never removed automatically

    Returns:
        bool: True if case_root is a complete CIME case, False if it doesn't exist
    """
    if not case_root.exists():
        return False

    hint = "remove it and retry the stage to re-clone it."
    xmlquery = case_root / "xmlquery"
    if not (case_root / "env_case.xml").is_file() or not os.access(xmlquery, os.X_OK):
        raise FileExistsError(
            f"{case_root} exists but is not a complete CIME case; {hint}"
        )
    try:
        case = _run(["./xmlquery", "CASE", "--value"], case_root)
    except RuntimeError as exc:
        raise FileExistsError(
            f"{case_root} exists but xmlquery CASE failed ({exc}); {hint}"
        ) from exc
    if case != case_root.name:
        raise FileExistsError(
            f"{case_root} exists but holds case {case!r}, not {case_root.name!r}; "
            f"{hint}"
        )
    return True


def _run(cmd: list[str] | str, cwd: Path) -> str:
    """Wrapper for subprocess calls

    Args:
        cmd (list[str] | str): input command
        cwd (Path): working directory for command

    Raises:
        RuntimeError: Command gave non-zero exit status

    Returns:
        str: output from command
    """

    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"command {cmd} failed in {cwd} "
            f"(exit {result.returncode}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.strip()
