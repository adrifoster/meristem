"""Dataclasses for reading and validating ensemble YAML"""

from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import os
from pathlib import Path
import yaml

from meristem.run.stages import StageConfig
from meristem.utils import check_id, to_absolute_path

_REQUIRED_KEYS = frozenset(
    {
        "ensemble_id",
        "root",
        "paramgen",
        "base_case",
        "paramfile_key",
        "member_stages",
    }
)
_OPTIONAL_KEYS = frozenset(
    {
        "output_root",
        "case_dir",
        "project",
        "email",
        "helper_queue",
    }
)
_REQUIRED_PARAMGEN_KEYS = frozenset({"dir", "prefix"})
_OPTIONAL_PARAMGEN_KEYS = frozenset({"members"})
_PARAMFILE_KEYS = frozenset({"paramfile", "fates_paramfile"})


def _validate_member_stage(stage: StageConfig) -> None:
    """Check a member stage can be run by the ensemble

    Raises:
        ValueError: if the stage sets no_cime (every member stage runs a CIME
        case), case_root (the ensemble puts each case under case_dir), command
        (the ensemble sets it to meristem ensemble prepare) or post_command
        (the ensemble sets it to meristem ensemble finalize)
    """
    if stage.no_cime:
        raise ValueError(
            f"member stage {stage.name!r}: no_cime is not allowed; "
            "every member stage runs a CIME case."
        )
    if stage.case_root is not None:
        raise ValueError(
            f"member stage {stage.name!r}: case_root is not allowed; each member's "
            "case is put under case_dir."
        )
    if stage.command:
        raise ValueError(
            f"member stage {stage.name!r}: command is not allowed; the ensemble "
            "runs meristem ensemble prepare for each stage, then the stage's "
            "script (if any)."
        )
    if stage.post_command:
        raise ValueError(
            f"member stage {stage.name!r}: post_command is not allowed; the "
            "ensemble runs meristem ensemble finalize after the stage's script."
        )


def _validate_top_level_keys(config) -> None:
    """Check that the config is a mapping with every required key (absent or
    null counts as missing) and no unknown keys

    Raises:
        ValueError: if any check fails
    """
    if config is None:
        raise ValueError("ensemble config is empty.")
    if not isinstance(config, dict):
        raise ValueError(
            "ensemble config must be a mapping of keys to values, "
            f"got {type(config).__name__}."
        )
    for key in _REQUIRED_KEYS:
        if config.get(key) is None:
            raise ValueError(f"missing required key {key} in ensemble config.")
    unknown = set(config) - _REQUIRED_KEYS - _OPTIONAL_KEYS
    if unknown:
        raise ValueError(f"unknown key(s) {sorted(unknown)} in ensemble config.")


def _validate_members(members) -> None:
    """Check that a members list is non-empty, all non-empty strings, and unique

    Raises:
        ValueError: if any check fails
    """
    if not members:
        raise ValueError("members must be 'all' or a non-empty list.")
    bad = [m for m in members if not isinstance(m, str) or not m]
    if bad:
        raise ValueError(
            f"members must be non-empty strings, got {bad}; "
            "quote numeric member names in the YAML."
        )
    duplicates = sorted(m for m, n in Counter(members).items() if n > 1)
    if duplicates:
        raise ValueError(f"members must be unique, got duplicates {duplicates}.")


@dataclass(frozen=True)
class EnsembleConfig:  # pylint: disable=too-many-instance-attributes
    """Immutable configuration for an ensemble, read from the ensemble YAML.

    Attributes
    -----------
    ensemble_id: short identifier, used in case and job names
    root: directory for ensemble and member state, PBS scripts and logs
    base_case: built CIME case that every member is cloned from
    case_dir: directory where member clones are created (default <root>/cases)
    paramgen_dir: meristem paramgen output directory
    paramgen_prefix: paramgen prefix (reads <prefix>.txt and <prefix>_key.csv)
    paramfile_key: user_nl_clm variable to set, "paramfile" or "fates_paramfile"
    member_stages: stages run once per member, in order
    members: member names to run, or None for all
    output_root: passed to create_clone --cime-output-root (None = CIME default)
    project: PBS project code (falls back to $PROJECT)
    email: email for job scheduler emails
    helper_queue: queue for members' small advance and fail jobs (e.g.
        "develop"), or None to use each member stage's queue
    """

    ensemble_id: str
    root: Path
    base_case: Path
    paramgen_dir: Path
    paramgen_prefix: str
    paramfile_key: str
    member_stages: tuple[StageConfig, ...]
    case_dir: Path
    members: tuple[str, ...] | None = None
    output_root: Path | None = None
    project: str = ""
    email: str | None = None
    helper_queue: str | None = None

    def __post_init__(self):
        if not isinstance(self.ensemble_id, str):
            raise ValueError(
                f"ensemble_id must be a string, got {type(self.ensemble_id).__name__} "
                f"{self.ensemble_id!r}; quote it in the YAML."
            )
        check_id("ensemble_id", self.ensemble_id)
        if not self.paramgen_prefix:
            raise ValueError("paramgen prefix must not be empty.")
        if self.paramfile_key not in _PARAMFILE_KEYS:
            raise ValueError(
                f"paramfile_key must be one of {sorted(_PARAMFILE_KEYS)}, "
                f"got {self.paramfile_key!r}."
            )

        if not self.member_stages:
            raise ValueError("member_stages must not be empty.")
        names = [stage.name for stage in self.member_stages]
        if len(names) != len(set(names)):
            raise ValueError(f"member stage names must be unique, got {names}.")
        for stage in self.member_stages:
            _validate_member_stage(stage)

        if self.members is not None:
            _validate_members(self.members)

        # paths must be absolute: they end up in PBS jobs (member commands, case
        # roots, saved parameter file paths), which don't run from the current
        # directory. __setattr__ is required because the dataclass is frozen
        for name in ("root", "base_case", "case_dir", "paramgen_dir", "output_root"):
            path = to_absolute_path(name, getattr(self, name))
            object.__setattr__(self, name, path)

        object.__setattr__(self, "member_stages", tuple(self.member_stages))

        if self.members is not None:
            object.__setattr__(self, "members", tuple(self.members))

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary in the schema from_dict reads, so
        from_dict(config.to_dict()) == config

        Returns:
            dict: JSON/YAML-serializable ensemble config
        """
        return {
            "ensemble_id": self.ensemble_id,
            "root": str(self.root),
            "paramgen": {
                "dir": str(self.paramgen_dir),
                "prefix": self.paramgen_prefix,
                # from_dict reads None as invalid, so "all members" is written "all"
                "members": "all" if self.members is None else list(self.members),
            },
            "base_case": str(self.base_case),
            "case_dir": str(self.case_dir),
            "paramfile_key": self.paramfile_key,
            "member_stages": [stage.to_dict() for stage in self.member_stages],
            "output_root": (
                str(self.output_root) if self.output_root is not None else None
            ),
            "project": self.project,
            "email": self.email,
            "helper_queue": self.helper_queue,
        }

    @classmethod
    def from_dict(cls, config: dict) -> EnsembleConfig:
        """Construct an EnsembleConfig from a config dictionary

        A required key counts as missing if it's absent or null; one that's
        present but empty (e.g. member_stages: []) is left to __post_init__,
        which says what's wrong with it

        Args:
            config (dict): parsed ensemble YAML

        Returns:
            EnsembleConfig: EnsembleConfig instance
        Raises:
            ValueError: if config isn't a mapping, or keys are missing, unknown,
                or malformed
        """
        _validate_top_level_keys(config)

        paramgen = config["paramgen"]
        if not isinstance(paramgen, dict):
            raise ValueError(
                "paramgen must be a mapping with keys "
                f"{sorted(_REQUIRED_PARAMGEN_KEYS)}, got {paramgen!r}."
            )
        for key in _REQUIRED_PARAMGEN_KEYS:
            if paramgen.get(key) is None:
                raise ValueError(
                    f"missing required key {key} in ensemble config subdict paramgen"
                )

        unknown = set(paramgen) - _REQUIRED_PARAMGEN_KEYS - _OPTIONAL_PARAMGEN_KEYS
        if unknown:
            raise ValueError(
                f"unknown key(s) {sorted(unknown)} in paramgen ensemble config."
            )

        members = paramgen.get("members", "all")
        if members != "all" and not isinstance(members, list):
            raise ValueError(
                "paramgen members must be 'all' or a list of member names."
            )

        if not isinstance(config["member_stages"], list):
            raise ValueError("member_stages must be a list.")
        for idx, stage in enumerate(config["member_stages"]):
            if not isinstance(stage, dict):
                raise ValueError(
                    f"member_stages[{idx}] must be a mapping of stage settings, "
                    f"got {stage!r}."
                )

        return cls(
            ensemble_id=config["ensemble_id"],
            root=config["root"],
            base_case=config["base_case"],
            case_dir=config.get("case_dir") or Path(config["root"]) / "cases",
            paramgen_dir=paramgen["dir"],
            paramgen_prefix=paramgen["prefix"],
            paramfile_key=config["paramfile_key"],
            member_stages=tuple(
                StageConfig.from_dict(s) for s in config["member_stages"]
            ),
            members=None if members == "all" else tuple(members),
            output_root=config.get("output_root"),
            # a blank "project:" in the YAML is None
            project=config.get("project") or "",
            email=config.get("email"),
            helper_queue=config.get("helper_queue"),
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> EnsembleConfig:
        """Read an ensemble YAML file, expanding $VARS from the environment

        Args:
            path (Path | str): path to the ensemble YAML

        Returns:
            EnsembleConfig: EnsembleConfig instance
        """
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(os.path.expandvars(f.read())))
