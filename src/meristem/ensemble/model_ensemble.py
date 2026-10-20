"""Class for managing parameter ensembles"""

from __future__ import annotations

import os
import subprocess
import json
from pathlib import Path
import pandas as pd

from meristem.ensemble.config import EnsembleConfig

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
    """

    def __init__(self, config: EnsembleConfig):
        self.config = config

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

        param_files = ensemble.find_param_files()
        ensemble.check_base_case()

        ensemble.root.mkdir(parents=True, exist_ok=True)
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
        return cls(EnsembleConfig.from_dict(state["config"]))

    def save(self):
        """Atomically write current state to ``<root>/ensemble_state.json``.

        Uses a write-to-temp-then-rename pattern so the state file is
        never left in a partially written state if the process is killed.
        """
        state = {"config": self.config.to_dict()}
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
