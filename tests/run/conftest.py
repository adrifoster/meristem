"""Fixtures shared across all meristem.run tests."""

from pathlib import Path
import pytest


class FakeScheduler:
    """In-memory Scheduler that records submitted job files instead of calling qsub"""

    name = "fake"

    def __init__(self):
        self.submitted: list[Path] = []

    def submit(self, job_file: Path) -> str:
        """Record job_file and return a fake job ID"""
        self.submitted.append(job_file)
        return f"{len(self.submitted)}.fake"


@pytest.fixture
def fake_scheduler() -> FakeScheduler:
    """A fresh FakeScheduler"""
    return FakeScheduler()
