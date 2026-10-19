"""Dataclasses for reading and validating ensemble YAML"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re

from meristem.run.stages import StageConfig

_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")
_PARAMFILE_KEYS = frozenset({"paramfile", "fates_paramfile"})


@dataclass(frozen=True)
class EnsembleConfig:  # pylint: disable=too-many-instance-attributes
    """Immutable configuration for an ensemble, read from the ensemble YAML.

    Attributes
    -----------
    ensemble_id: short identifier, used in case and job names
    root: directory for ensemble and member state, PBS scripts and logs
    base_case: built CIME case that every member is cloned from
    case_dir: directory where member clones are created
    paramgen_dir: meristem paramgen output directory
    paramgen_prefix: paramgen prefix (reads <prefix>.txt and <prefix>_key.csv)
    paramfile_key: user_nl_clm variable to set, "paramfile" or "fates_paramfile"
    member_stages: stages run once per member, in order
    members: member names to run, or None for all
    output_root: passed to create_clone --cime-output-root (None = CIME default)
    project: PBS project code (falls back to $PROJECT)
    email: email for job scheduler emails 
    """

    ensemble_id: str
    root: Path
    base_case: Path
    case_dir: Path
    paramgen_dir: Path
    paramgen_prefix: str
    paramfile_key: str
    member_stages: tuple[StageConfig, ...]
    members: tuple[str, ...] | None = None
    output_root: Path | None = None
    project: str = ""
    user: str | None = None

    def __post_init__(self):
        if not _ID_RE.match(self.ensemble_id):
            raise ValueError(
                f"ensemble_id {self.ensemble_id!r} contains invalid characters. "
                "Only letters, digits, hyphens, and underscores are allowed."
            )
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
            if stage.no_cime:
                raise ValueError(
                    f"member stage {stage.name!r}: no_cime is not allowed; "
                    "every member stage runs a CIME case."
                )

        if self.members is not None and not self.members:
            raise ValueError("members must be 'all' or a non-empty list.")

        # __setattr__ is required because the dataclass is frozen
        for name in ("root", "base_case", "case_dir", "paramgen_dir"):
            object.__setattr__(self, name, Path(getattr(self, name)))

        if self.output_root is not None:
            object.__setattr__(self, name, Path(self.output_root))

        object.__setattr__(self, "member_stages", tuple(self.member_stages))

        if self.members is not None:
            object.__setattr__(self, "members", tuple(self.members))
