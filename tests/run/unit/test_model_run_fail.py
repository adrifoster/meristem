"""Tests for ModelRun.fail behavior"""

from __future__ import annotations

import time
import pytest

from meristem.run.model_run import ModelRun
from meristem.run.stages import StageStatus
from meristem.run.scheduler import JobState


@pytest.fixture
def mid_run(created_run) -> ModelRun:
    """created_run whose stage is submitted (setup job 1.fake) with its CIME jobs
    (10, 11), advance job (12) and fail jobs (13, 14) recorded
    """
    created_run.submit()
    state = created_run.stages[0].state
    state.cime_job_ids = ["10.fake", "11.fake"]
    state.advance_job_id = "12.fake"
    state.fail_job_ids = ["13.fake", "14.fake"]
    created_run.save()
    return created_run


def test_fail_sets_stage_status_to_failed(created_run):
    """Test that fail() marks the stage as FAILED"""
    created_run.stages[0].state.status = StageStatus.SUBMITTED
    created_run.stages[0].state.submit_time = time.time() - 100
    created_run.fail("spinup_ad")
    assert created_run.stages[0].state.status is StageStatus.FAILED


def test_fail_on_done_stage_status_to_failed(created_run):
    """Test that fail() marks the stage as FAILED even if it was DONE"""
    created_run.stages[0].state.status = StageStatus.SUBMITTED
    created_run.stages[0].state.status = StageStatus.DONE
    created_run.stages[0].state.submit_time = time.time() - 100
    created_run.fail("spinup_ad")
    assert created_run.stages[0].state.status is StageStatus.FAILED


def test_fail_sets_end_time(created_run):
    """Test that fail() sets end_time on the stage"""
    created_run.stages[0].state.status = StageStatus.SUBMITTED
    created_run.stages[0].state.submit_time = time.time() - 100
    created_run.fail("spinup_ad")
    assert created_run.stages[0].state.end_time is not None


def test_fail_persists_to_disk(created_run):
    """Test that fail() saves the updated state to disk"""
    created_run.submit()
    created_run.fail("spinup_ad")
    loaded = ModelRun.load(created_run.root, scheduler=created_run.scheduler)
    assert loaded.stages[0].state.status is StageStatus.FAILED


def test_fail_preserves_job_id(created_run):
    """Test that fail() does not clobber the existing job ID"""
    created_run.submit()
    created_run.fail("spinup_ad")
    assert created_run.stages[0].state.job_id == "1.fake"


def test_fail_preserves_attempts(created_run):
    """Test that fail() does not reset the attempts counter"""
    created_run.submit()
    created_run.fail("spinup_ad")
    assert created_run.stages[0].state.attempts == 1


def test_fail_unknown_stage_raises(created_run):
    """Test that fail() raises for an unknown stage name"""
    with pytest.raises(ValueError, match="nonexistent_stage"):
        created_run.fail("nonexistent_stage")


@pytest.mark.parametrize(
    "cant_fail_status",
    [
        StageStatus.PENDING,
        StageStatus.FAILED,
    ],
)
def test_advance_non_pending_or_failed_rases(created_run, cant_fail_status):
    """Test that ModelRun.fail() raises a value error if status isn't SUBMITTED"""
    object.__setattr__(created_run.stages[0].state, "_status", cant_fail_status)
    with pytest.raises(ValueError, match="fail"):
        created_run.fail("spinup_ad")


def test_fail_cancels_only_active_cime_jobs(mid_run, fake_scheduler):
    """Test that fail() cancels the stage's active CIME jobs, and leaves the
    advance and fail jobs for PBS to delete and the calling setup job alone
    """
    fake_scheduler.job_states = {
        job: JobState.QUEUED
        for job in ("10.fake", "11.fake", "12.fake", "13.fake", "14.fake")
    }
    fake_scheduler.job_states["1.fake"] = (
        JobState.RUNNING
    )  # the setup job, calling fail
    mid_run.fail("spinup_ad")
    assert fake_scheduler.canceled == [["10.fake", "11.fake"]]
    assert mid_run.stages[0].state.status is StageStatus.FAILED


@pytest.mark.parametrize(
    "states, cancelled",
    [
        ({"10.fake": JobState.FINISHED, "11.fake": JobState.QUEUED}, [["11.fake"]]),
        ({"10.fake": JobState.FINISHED, "11.fake": JobState.FINISHED}, []),
        ({}, []),  # unknown to the scheduler
    ],
)
def test_fail_skips_finished_cime_jobs(mid_run, fake_scheduler, states, cancelled):
    """Test that fail() only cancels CIME jobs still queued or running"""
    fake_scheduler.job_states = states
    mid_run.fail("spinup_ad")
    assert fake_scheduler.canceled == cancelled


def test_fail_without_cime_jobs_cancels_nothing(created_run, fake_scheduler):
    """Test that fail() on a stage with no CIME jobs recorded cancels nothing"""
    created_run.submit()
    created_run.fail("spinup_ad")
    assert fake_scheduler.canceled == []
    assert created_run.stages[0].state.status is StageStatus.FAILED


@pytest.mark.parametrize("method", ["cancel", "query"])
def test_fail_cancel_error_warns_and_still_fails(
    mid_run, fake_scheduler, monkeypatch, capsys, method
):
    """Test that if qdel or qstat fails, the stage is still FAILED on disk and a
    warning says how to cancel the CIME jobs
    """
    fake_scheduler.job_states = {"10.fake": JobState.RUNNING}

    def _broken(*args):
        raise RuntimeError("cannot connect to server")

    monkeypatch.setattr(fake_scheduler, method, _broken)
    mid_run.fail("spinup_ad")

    loaded = ModelRun.load(mid_run.root, scheduler=fake_scheduler)
    assert loaded.stages[0].state.status is StageStatus.FAILED
    assert "meristem run cancel" in capsys.readouterr().out


def test_stale_fail_cancels_nothing(mid_run, fake_scheduler):
    """Test that a fail from an older attempt is ignored: the stage stays
    SUBMITTED and no CIME jobs are cancelled
    """
    fake_scheduler.job_states = {"10.fake": JobState.RUNNING}
    mid_run.fail("spinup_ad", attempt=0)  # the stage is on attempt 1
    assert mid_run.stages[0].state.status is StageStatus.SUBMITTED
    assert fake_scheduler.canceled == []
