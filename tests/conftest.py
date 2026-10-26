"""Fixtures shared across all meristem tests."""

from pathlib import Path
import pytest

from meristem.run.scheduler import JobState


class FakeScheduler:
    """In-memory Scheduler that records submitted job files instead of calling qsub

    Set job_states to control what query() reports; jobs not in it are UNKNOWN
    """

    name = "fake"

    def __init__(self):
        self.submitted: list[Path] = []
        self.job_states: dict[str, JobState] = {}
        self.queried: list[list[str]] = []

    def submit(self, job_file: Path) -> str:
        """Record job_file and return a fake job ID"""
        self.submitted.append(job_file)
        return f"{len(self.submitted)}.fake"

    def query(self, job_ids: list[str]) -> dict[str, JobState]:
        """Record the query and return the states a test set in job_states"""
        self.queried.append(list(job_ids))
        return {
            job_id: self.job_states.get(job_id, JobState.UNKNOWN) for job_id in job_ids
        }


@pytest.fixture
def fake_scheduler() -> FakeScheduler:
    """A fresh FakeScheduler"""
    return FakeScheduler()
