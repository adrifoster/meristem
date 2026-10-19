"""
Module for holding core classes for tethering:
StageStatus Enum, plus StageState, StageConfig, and Stage
"""

from __future__ import annotations

import re
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

_WALLTIME_RE = re.compile(r"^\d+(?::\d+)*$")
_MEMORY_RE = re.compile(r"^\d+(\.\d+)?(B|KB|MB|GB|TB)$", re.IGNORECASE)

# keys that belong to StageState in a flat Stage dict.
# must be kept in sync with StageState.__init__ parameters.
_RUNTIME_KEYS: frozenset[str] = frozenset(
    {
        "status",
        "job_id",
        "advance_job_id",
        "fail_job_ids",
        "submit_time",
        "end_time",
        "attempts",
    }
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class StageStatus(Enum):
    """Stage Status Enum"""

    PENDING = "pending"
    SUBMITTED = "submitted"
    DONE = "done"
    FAILED = "failed"

    def to_str(self) -> str:
        """Return value string

        Returns:
            str: output string
        """
        return self.value

    @classmethod
    def from_str(cls, raw: str) -> StageStatus:
        """Instantiate the class from an input raw string

        Args:
            raw (str): input string

        Raises:
            ValueError: invalid status supplied

        Returns:
            StageStatus: StageStatus instance
        """
        try:
            return cls(raw.lower().strip())
        except ValueError as e:
            raise ValueError(
                f"Invalid StageStatus {raw!r}. "
                f"Valid values: {[e.value for e in cls]}"
            ) from e


_ALLOWED_TRANSITIONS: dict[StageStatus, frozenset[StageStatus]] = {
    StageStatus.PENDING: frozenset(
        {StageStatus.SUBMITTED, StageStatus.FAILED, StageStatus.PENDING}
    ),
    StageStatus.SUBMITTED: frozenset({StageStatus.DONE, StageStatus.FAILED}),
    StageStatus.DONE: frozenset({StageStatus.PENDING, StageStatus.FAILED}),
    StageStatus.FAILED: frozenset({StageStatus.PENDING}),
}


# ---------------------------------------------------------------------------
# StageStage (mutable runtime state)
# ---------------------------------------------------------------------------


class StageState:
    """Mutable runtime state for one stage. Owned by `Stage`.

    Mutations for submit_time, end_time, attempts, and status gi through methods or properties
    that enforce invariants:
    - submit_time cannot be set after end_time is set
    - end_time requires submit_time to be set first
    - end_time cannot precede submit_time
    - attempts is incremented via increment_attempts(), never set directly
    - status must always be a class StageStatus instance

    Note:
    A StageState is intended to be owned by a single class Stage instance.
    Do not share the same StageState object between multiple Stage objects.
    Use Stage.status = StageState(...) to replace it.

    Attributes
    -----------
    status: StageStatus
        Current status of the stage
    job_id: str or None
        PBS job ID of actual job assigned after submission
    advance_job_id: str or None
        PBS job ID of advance job assigned after submission
    fail_job_ids: list of str
        PBS job IDs of fail jobs assigned after submission (one per CIME job
        they watch)
    submit_time: float or None
        Unix timestamp of the most recent submission
    end_time: float or None
        Unix timestamp of completion or failure
    attempts: int
        How many times this stage has been submitted (must be >=0)
    """

    def __init__(
        self,
        status: StageStatus = StageStatus.PENDING,
        job_id: str | None = None,
        advance_job_id: str | None = None,
        fail_job_ids: list[str] | None = None,
        submit_time: float | None = None,
        end_time: float | None = None,
        attempts: int = 0,
    ):
        self._status = status
        self.job_id = job_id
        self.advance_job_id = advance_job_id
        self.fail_job_ids = list(fail_job_ids or [])
        self._submit_time = submit_time
        self._end_time = end_time
        self._attempts = attempts
        self._validate()

    def _validate(self):
        if not isinstance(self._status, StageStatus):
            raise TypeError(
                f"status must be a StageStatus instance, got {type(self._status).__name__!r}"
            )
        if self._attempts < 0:
            raise ValueError(f"attempts must be >=0, got {self._attempts!r}")
        if self._end_time is not None:
            if self._submit_time is None:
                raise ValueError(
                    "end_time is set but submit_time is None; "
                    "a stage cannot finish before it starts."
                )
            if self._end_time < self._submit_time:
                raise ValueError(
                    f"end_time ({self._end_time}) cannot precede "
                    f"submit_time ({self._submit_time})."
                )

    @property
    def status(self) -> StageStatus:
        """Get the current StageStatus"""
        return self._status

    @status.setter
    def status(self, value: StageStatus):
        if not isinstance(value, StageStatus):
            raise TypeError(
                f"status must be a StageStatus instance, got {type(value).__name__!r}"
            )
        allowed = _ALLOWED_TRANSITIONS[self._status]
        if value not in allowed:
            raise ValueError(
                f"Invalid status transition: {self._status} to {value}. "
                f"Allowed: {allowed}"
            )
        self._status = value
        self._validate()

    @property
    def submit_time(self) -> float | None:
        """Get the submit time

        Returns:
            float | None: submit time
        """
        return self._submit_time

    @submit_time.setter
    def submit_time(self, value: float | None):
        """Set the submit time

        Args:
            value (float): submit_time to set
        """
        self._submit_time = value
        self._validate()

    @property
    def end_time(self) -> float | None:
        """Get the end time

        Returns:
            float | None: end time
        """
        return self._end_time

    @end_time.setter
    def end_time(self, value: float | None):
        """Set the end time

        Args:
            value (float): end_time to set
        """
        self._end_time = value
        self._validate()

    @property
    def attempts(self) -> int:
        """Get the number of attempts

        Returns:
            int: attempts
        """
        return self._attempts

    def increment_attempts(self):
        """Increment the number of attempts"""
        self._attempts += 1

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StageState):
            return NotImplemented
        return (
            self.status == other.status
            and self.job_id == other.job_id
            and self.advance_job_id == other.advance_job_id
            and self.fail_job_ids == other.fail_job_ids
            and self._submit_time == other._submit_time
            and self._end_time == other._end_time
            and self._attempts == other._attempts
        )

    def __repr__(self) -> str:
        return (
            f"StageState(status={self.status!r}, job_id={self.job_id!r}, "
            f"advance_job_id={self.advance_job_id!r}, fail_job_ids={self.fail_job_ids!r}, "
            f"submit_time={self._submit_time!r}, end_time={self._end_time!r}, "
            f"attempts={self._attempts!r})"
        )

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary

        Note that the ``status`` value is stored as its string representation so that
        the dict is directly JSON-serializable without a custom encoder.

        Returns:
            dict: output dictionary
        """
        return {
            "status": self.status.to_str(),
            "job_id": self.job_id,
            "advance_job_id": self.advance_job_id,
            "fail_job_ids": list(self.fail_job_ids),
            "submit_time": self._submit_time,
            "end_time": self._end_time,
            "attempts": self._attempts,
        }

    @classmethod
    def from_dict(cls, input_dict: dict) -> StageState:
        """Construct a StageState from an input dictionary

        Args:
            input_dict (dict): input dictionary, must contain at minimum a `status` key

        Returns:
            StageState: StageState instance
        Raises:
            ValueError: If "status" is absent or not a valid StageStatus value
        """
        input_dict = dict(input_dict)
        fail_job_ids = input_dict.get("fail_job_ids")
        if fail_job_ids is None:
            # older state files stored a single fail_job_id
            legacy = input_dict.get("fail_job_id")
            fail_job_ids = [legacy] if legacy else []
        return cls(
            status=StageStatus.from_str(input_dict.get("status", "")),
            job_id=input_dict.get("job_id"),
            advance_job_id=input_dict.get("advance_job_id"),
            fail_job_ids=fail_job_ids,
            submit_time=input_dict.get("submit_time"),
            end_time=input_dict.get("end_time"),
            attempts=input_dict.get("attempts", 0),
        )


# ---------------------------------------------------------------------------
# StageConfig (immutable configuration)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageConfig:  # pylint: disable=too-many-instance-attributes
    """Dataclass that holds immutable properties of a run.

    Frozen so that config cannot be accidentally mutated after construction.
    All attributes are set at creation and never change.

    Attributes
    -----------
    name: unique identifier, e.g., "spinup_ad"
    walltime: PBS walltime string, e.g., "06:00:00"
    queue: PBS queue name
    spinup_check: if True, advance() reads runs spinup_stability before proceeding
    ncpus: CPUs per node requested from PBS
    select: number of nodes
    memory: memory requested string, e.g. "1GB"
    extra_pbs: additional raw #PBS directives
    script: path to setup script; must either supply this or command;
        called as bash <script> <case_root> <previous_case_name> <previous_case_root>
    command: command to run (in place of script)
    case_root: absolute path to the CESM case directory. If None, the run uses
        <root>/<name>
    submit_args: extra submission arguments, optional
    """

    name: str
    queue: str
    walltime: str = "01:00:00"
    spinup_check: bool = False
    ncpus: int = 1
    select: int = 1
    memory: str = "10GB"
    no_cime: bool = False
    extra_pbs: tuple[str, ...] = ()
    script: Path | None = None
    command: str | None = None
    case_root: Path | None = None
    submit_args: str | None = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("name must not be empty.")

        if not self.script and not self.command:
            raise ValueError("either script or command must be supplied.")

        if self.script and self.command:
            raise ValueError(
                "both script and command cannot be supplied together; choose one"
            )

        # __setattr__ is required because the dataclass is frozen
        if self.script is not None:
            object.__setattr__(self, "script", Path(self.script))

        if self.case_root is not None:
            object.__setattr__(self, "case_root", Path(self.case_root))

        _validate_walltime(self.walltime)

        if not self.queue:
            raise ValueError("queue must not be empty.")
        if self.ncpus < 1:
            raise ValueError(f"ncpus must be >=1, got {self.ncpus!r}.")
        if self.select < 1:
            raise ValueError(f"select must be >=1, got {self.select!r}.")
        if not _MEMORY_RE.match(self.memory):
            raise ValueError(
                f"memory {self.memory!r} does not match expected format "
                "<number><unit> where unit is one of B, KB, MB, GB, TB. "
                "Example: '16GB'"
            )

    def validate(self):
        """Validate that this config is ready for submission.

        Separated from __post_init__ so StageConfig can be constructed and
        deserialized without requiring filesystem access

        Raises:
            ValueError: if the script path does not exist
        """
        if self.script and not self.script.exists():
            raise ValueError(f"cannot open script {self.script!r}.")

    def to_dict(self) -> dict:
        """Serialize configuration fields to a plain dictionary"""
        return {
            "name": self.name,
            "case_root": str(self.case_root) if self.case_root is not None else None,
            "submit_args": self.submit_args if self.submit_args is not None else None,
            "script": str(self.script) if self.script is not None else None,
            "command": self.command,
            "walltime": self.walltime,
            "queue": self.queue,
            "spinup_check": self.spinup_check,
            "ncpus": self.ncpus,
            "select": self.select,
            "memory": self.memory,
            "no_cime": self.no_cime,
            "extra_pbs": list(self.extra_pbs),
        }

    @classmethod
    def from_dict(cls, input_dict: dict) -> StageConfig:
        """Construct a class StageConfig from an input dictionary

        Args:
            input_dict (dict): Mapping of configuration field names to values

        Returns:
            StageConfig: StageConfig instance
        """
        in_dict = dict(input_dict)
        in_dict["extra_pbs"] = tuple(in_dict.get("extra_pbs", []))
        return cls(**in_dict)


# ---------------------------------------------------------------------------
# Stage  (composes config + runtime state)
# ---------------------------------------------------------------------------


@dataclass
class Stage:
    """Dataclass for one stage of a run. Holds both config (set at creation, never
    changes) and runtime state (updated as the stage progresses).

    Attributes
    -----------
    config: StageConfig attributes
    state: StageState instance; reset with stage.state = StageState()
    """

    config: StageConfig
    state: StageState = field(default_factory=StageState)

    @classmethod
    def from_dict(cls, input_dict: dict) -> Stage:
        """Reconstruct class Stage from a flat dictionary

        Args:
            input_dict (dict): input dictionary

        Returns:
            Stage: Stage instance
        """
        in_dict = dict(input_dict)
        runtime_dict = {
            k: in_dict.pop(k)
            for k in _RUNTIME_KEYS
            if k in in_dict
        }
        return cls(
            config=StageConfig.from_dict(in_dict),
            state=StageState.from_dict(runtime_dict) if runtime_dict else StageState(),
        )

    def to_dict(self) -> dict:
        """Flatten a class Stage to a single-level dictionary

        Returns:
            dict: output dictionary
        """
        return {**self.config.to_dict(), **self.state.to_dict()}


def _validate_walltime(walltime: str):
    """Validate walltime for a PBS submission

    Args:
        walltime (str): input string
    """
    if not _WALLTIME_RE.match(walltime):
        raise ValueError(
            f"walltime {walltime!r} must be a colon-separated time string, "
            "e.g. '6:00:00' or '06:00:00' or '24:00:00'. "
            "See PBS documentation for full format options."
        )
    parts = walltime.split(":")
    if len(parts) > 4:
        raise ValueError(
            f"walltime {walltime!r} has too many components (max 4: DD:HH:MM:SS)."
        )
    # only validate mm:ss ranges when we can identify them
    if len(parts) >= 2 and int(parts[-1]) > 59:
        raise ValueError(f"walltime seconds must be 00-59, got {parts[-1]!r}.")
    if len(parts) >= 3 and int(parts[-2]) > 59:
        raise ValueError(f"walltime minutes must be 00-59, got {parts[-2]!r}.")
    if all(int(part) == 0 for part in parts):
        raise ValueError("walltime must be non-zero.")
