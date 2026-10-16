"""Lightweight interface for job scheduler commands"""

from __future__ import annotations

import re
import subprocess
from enum import Enum
from pathlib import Path
from typing import Protocol


class JobState(Enum):
    """Scheduler-independent state of a submitted job"""

    QUEUED = "queued"
    RUNNING = "running"
    FINISHED = "finished"
    # the scheduler has no record of the job (e.g. aged out of its history)
    UNKNOWN = "unknown"


class Scheduler(Protocol):
    """Interface protocol class for job schedulers

    name is persisted in the run state file so ModelRun.load can rebuild the scheduler
    """

    name: str

    def submit(self, job_file: Path) -> str:
        """Submit a job"""
        ...

    def query(self, job_ids: list[str]) -> dict[str, JobState]:
        """Get the current state of each job, keyed by the job IDs passed in"""
        ...


# PBS job_state letter -> JobState. Q/H/R/F are the common cases; the rest are
# mapped to the closest state so a live job is never reported as gone
_PBS_STATES: dict[str, JobState] = {
    "Q": JobState.QUEUED,  # queued
    "H": JobState.QUEUED,  # held (e.g. waiting on a dependency)
    "W": JobState.QUEUED,  # waiting for its start time
    "R": JobState.RUNNING,  # running
    "S": JobState.RUNNING,  # suspended
    "E": JobState.RUNNING,  # exiting after running
    "F": JobState.FINISHED,  # finished
}

# qstat exit code when one or more job IDs are unknown (PBSE_UNKJOBID)
_QSTAT_UNKNOWN_JOB_EXIT = 153

_QSTAT_JOB_ID_RE = re.compile(r"^Job Id:\s*(\S+)")
_QSTAT_JOB_STATE_RE = re.compile(r"^\s+job_state\s*=\s*(\S+)")


def _job_number(job_id: str) -> str:
    """Get the numeric part of a PBS job ID, e.g. "12345" from "12345.desched1"

    qsub and qstat don't always write the server suffix the same way, so job IDs
    are matched on this part
    """
    return job_id.split(".", 1)[0]


def _parse_qstat_states(output: str) -> dict[str, str]:
    """Parse ``qstat -f`` output into {job number: job_state letter}

    Args:
        output (str): stdout of ``qstat -x -f``

    Returns:
        dict[str, str]: PBS job_state letter for each job in the output
    """
    states: dict[str, str] = {}
    current = None
    for line in output.splitlines():
        if match := _QSTAT_JOB_ID_RE.match(line):
            current = _job_number(match.group(1))
        elif current is not None and (match := _QSTAT_JOB_STATE_RE.match(line)):
            states[current] = match.group(1)
    return states


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

    def query(self, job_ids: list[str]) -> dict[str, JobState]:
        """Get the state of each job with one ``qstat -x -f`` call

        -x includes finished jobs from the PBS history. Jobs PBS has no record of
        (e.g. aged out of the history) are UNKNOWN

        Args:
            job_ids (list[str]): PBS job IDs

        Raises:
            RuntimeError: if qstat fails for any reason other than unknown job IDs,
                or reports a job_state this class doesn't recognize

        Returns:
            dict[str, JobState]: state of each job, keyed by the job IDs passed in
        """
        if not job_ids:
            return {}
        result = subprocess.run(
            ["qstat", "-x", "-f", *job_ids],
            capture_output=True,
            text=True,
            check=False,
        )
        # a non-zero exit is expected when any ID is unknown; anything else means
        # qstat itself failed, which must not be mistaken for every job being gone
        if result.returncode not in (0, _QSTAT_UNKNOWN_JOB_EXIT):
            raise RuntimeError(
                f"qstat failed (exit {result.returncode}): "
                f"{(result.stderr or result.stdout).strip()}"
            )

        letters = _parse_qstat_states(result.stdout)
        states = {}
        for job_id in job_ids:
            letter = letters.get(_job_number(job_id))
            if letter is None:
                states[job_id] = JobState.UNKNOWN
            elif letter in _PBS_STATES:
                states[job_id] = _PBS_STATES[letter]
            else:
                raise RuntimeError(
                    f"qstat reported unrecognized job_state {letter!r} for {job_id}"
                )
        return states


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
