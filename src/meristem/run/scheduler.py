"""Lightweight interface for job scheduler commands"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Protocol


class Scheduler(Protocol):
    """Interface protocol class for job schedulers

    name is persisted in the run state file so ModelRun.load can rebuild the scheduler
    """

    name: str

    def submit(self, job_file: Path) -> str:
        """Submit a job"""
        ...


class PbsScheduler:
    """Interface for PBS Job Scheduler"""

    name = "pbs"

    def submit(self, job_file: Path) -> str:
        """Submit a job with qsub

        Args:
            job_file (Path): path to job file

        Returns:
            str: output string from return of qsub command
        """
        result = subprocess.run(
            ["qsub", str(job_file)],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()


# scheduler name -> class, used to rebuild a scheduler from a config or state file
_SCHEDULERS: dict[str, type[Scheduler]] = {
    PbsScheduler.name: PbsScheduler,
}

DEFAULT_SCHEDULER = PbsScheduler.name


def get_scheduler(name: str) -> Scheduler:
    """Instantiate a scheduler from its name

    Args:
        name (str): scheduler name, e.g. "pbs"

    Raises:
        ValueError: if name is not a known scheduler

    Returns:
        Scheduler: scheduler instance
    """
    try:
        scheduler_cls = _SCHEDULERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown scheduler {name!r}. Known schedulers: {sorted(_SCHEDULERS)}"
        ) from None
    return scheduler_cls()
