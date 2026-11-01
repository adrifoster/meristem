"""Tests for ModelRun.cancel behavior"""

from __future__ import annotations

import pytest

from meristem.run.model_run import ModelRun
from meristem.run.stages import StageStatus
from meristem.run.scheduler import JobState


def test_cancel_cancels_running_job(created_run, fake_scheduler):
    """Test that ModelRun.cancel() marks the canceled stage as FAILED"""
    created_run.submit()
    created_run.scheduler.job_states = {"1.fake": JobState.RUNNING}
    canceled = created_run.cancel()

    assert canceled == ["1.fake"]
    assert fake_scheduler.canceled == [["1.fake"]]

    assert created_run.stages[0].state.status is StageStatus.FAILED
    loaded = ModelRun.load(created_run.root, scheduler=fake_scheduler)
    assert loaded.stages[0].state.status is StageStatus.FAILED


def test_cancel_cancels_cime_jobs(created_run, fake_scheduler):
    """Test that ModelRun.cancel() cancels cime jobs and marks stage as FAILED"""
    created_run.submit()
    state = created_run.stages[0].state
    state.cime_job_ids = ["10.fake", "11.fake"]  # case.run, case.st_archive
    state.advance_job_id = "12.fake"
    state.fail_job_ids = ["13.fake", "14.fake"]
    fake_scheduler.job_states = {
        "1.fake": JobState.FINISHED,  # the setup job is done
        "10.fake": JobState.RUNNING,
        "11.fake": JobState.QUEUED,
        "12.fake": JobState.QUEUED,
        "13.fake": JobState.QUEUED,
        "14.fake": JobState.QUEUED,
    }
    canceled = created_run.cancel()

    expected = ["12.fake", "13.fake", "14.fake", "10.fake", "11.fake"]
    assert canceled == expected
    assert fake_scheduler.canceled == [expected]
    assert created_run.stages[0].state.status is StageStatus.FAILED


def test_cancel_nothing_active_returns_empty(created_run, fake_scheduler):
    """Test that ModelRun.cancel() returns empty list when nothing is running"""
    preloaded = ModelRun.load(created_run.root, scheduler=fake_scheduler)

    canceled = created_run.cancel()

    assert canceled == []
    assert fake_scheduler.canceled == []
    postloaded = ModelRun.load(created_run.root, scheduler=fake_scheduler)
    assert preloaded.stages[0].state.status == postloaded.stages[0].state.status


def test_cancel_dry_run_changes_nothing(created_run, fake_scheduler):
    """Test that a dry run returns the jobs it would cancel, but cancels none
    and leaves the stage SUBMITTED
    """
    created_run.submit()
    fake_scheduler.job_states = {"1.fake": JobState.RUNNING}

    assert created_run.cancel(dry_run=True) == ["1.fake"]

    assert fake_scheduler.canceled == []
    loaded = ModelRun.load(created_run.root, scheduler=fake_scheduler)
    assert loaded.stages[0].state.status is StageStatus.SUBMITTED


def test_cancel_failed_stage_with_live_jobs(created_run, fake_scheduler):
    """Test that cancel() cancels the still-active jobs of a stage that's already
    FAILED (e.g. its setup job failed after case.submit), leaves it FAILED, and
    lets retry go ahead
    """
    created_run.submit()  # setup job 1.fake
    state = created_run.stages[0].state
    state.cime_job_ids = ["10.fake", "11.fake"]
    state.advance_job_id = "12.fake"
    created_run.fail("spinup_ad")  # the setup job's error trap
    fake_scheduler.job_states = {
        "1.fake": JobState.FINISHED,
        "10.fake": JobState.RUNNING,
        "11.fake": JobState.QUEUED,
        "12.fake": JobState.QUEUED,
    }

    assert created_run.cancel() == ["12.fake", "10.fake", "11.fake"]
    assert created_run.stages[0].state.status is StageStatus.FAILED

    created_run.retry("spinup_ad")
    assert created_run.stages[0].state.status is StageStatus.SUBMITTED
    assert created_run.stages[0].state.attempts == 2


def test_cancel_only_touches_stages_with_active_jobs(two_stage_run, fake_scheduler):
    """Test that cancelling a run mid-pipeline cancels the current stage's jobs
    and leaves the finished stage DONE
    """
    two_stage_run.submit()  # spinup_ad: 1.fake
    two_stage_run.advance("spinup_ad")  # spinup_ad DONE, spinup_sasu: 2.fake
    fake_scheduler.job_states = {"2.fake": JobState.RUNNING}

    assert two_stage_run.cancel() == ["2.fake"]

    ad, sasu = two_stage_run.stages
    assert ad.state.status is StageStatus.DONE
    assert sasu.state.status is StageStatus.FAILED


def test_cancel_qdel_failure_changes_nothing(created_run, fake_scheduler, monkeypatch):
    """Test that if qdel fails, the error propagates and the stage stays SUBMITTED"""
    created_run.submit()
    fake_scheduler.job_states = {"1.fake": JobState.RUNNING}

    def _fail(job_ids):
        raise RuntimeError("qdel failed (exit 1): cannot connect to server")

    monkeypatch.setattr(fake_scheduler, "cancel", _fail)
    with pytest.raises(RuntimeError, match="cannot connect"):
        created_run.cancel()
    loaded = ModelRun.load(created_run.root, scheduler=fake_scheduler)
    assert loaded.stages[0].state.status is StageStatus.SUBMITTED
