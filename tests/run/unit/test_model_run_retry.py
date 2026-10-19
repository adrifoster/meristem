"""Tests for ModelRun.retry behavior"""

from __future__ import annotations

import pytest

from meristem.run.model_run import ModelRun
from meristem.run.scheduler import JobState
from meristem.run.stages import StageStatus


def test_retry_calls_submit_stage(created_run, mocker):
    """Test that retry() delegates to _submit_stage"""
    created_run.stages[0].state.status = StageStatus.FAILED
    mock = mocker.patch.object(created_run, "_submit_stage", return_value="12345.pbs")
    created_run.retry("spinup_ad")
    mock.assert_called_once_with(
        created_run.stages[0], dry_run=False, skip_script=False
    )


def test_retry_resets_to_pending_before_submitting(created_run, mocker):
    """Test that retry() resets stage to PENDING before calling _submit_stage"""
    created_run.stages[0].state.status = StageStatus.FAILED
    created_run.stages[0].state.advance_job_id = "111.pbs"
    created_run.stages[0].state.fail_job_ids = ["222.pbs", "333.pbs"]

    def check_status(stage, dry_run=False, skip_script=False):
        # signature must match _submit_stage, which is called with these as keywords
        # pylint: disable=unused-argument
        assert stage.state.status is StageStatus.PENDING
        assert stage.state.end_time is None
        assert stage.state.submit_time is None
        assert stage.state.job_id is None
        assert stage.state.advance_job_id is None
        assert stage.state.fail_job_ids == []
        return "12345.pbs"

    mocker.patch.object(created_run, "_submit_stage", side_effect=check_status)
    created_run.retry("spinup_ad")


def test_retry_persists_pending_before_submitting(created_run, mocker):
    """Test that retry() saves PENDING state before calling _submit_stage"""
    created_run.stages[0].state.status = StageStatus.FAILED
    save_calls = []
    original_save = created_run.save

    def tracking_save():
        save_calls.append(created_run.stages[0].state.status)
        original_save()

    created_run.save = tracking_save
    mocker.patch.object(created_run, "_submit_stage", return_value="12345.pbs")
    created_run.retry("spinup_ad")
    assert StageStatus.PENDING in save_calls


def test_retry_failed_stage(created_run, mocker):
    """Test that retry() accepts a FAILED stage"""
    created_run.stages[0].state.status = StageStatus.FAILED
    mock = mocker.patch.object(created_run, "_submit_stage", return_value="12345.pbs")
    created_run.retry("spinup_ad")
    mock.assert_called_once()


def test_retry_pending_stage(created_run, mocker):
    """Test that retry() accepts a PENDING stage"""
    mock = mocker.patch.object(created_run, "_submit_stage", return_value="12345.pbs")
    created_run.retry("spinup_ad")
    mock.assert_called_once()


@pytest.mark.parametrize("status", [StageStatus.SUBMITTED, StageStatus.DONE])
def test_retry_invalid_status_raises(created_run, mocker, status):
    """Test that retry() raises for non-retryable stage statuses"""
    object.__setattr__(created_run.stages[0].state, "_status", status)
    mocker.patch.object(created_run, "_submit_stage")
    with pytest.raises(ValueError, match="spinup_ad"):
        created_run.retry("spinup_ad")


@pytest.mark.parametrize("status", [StageStatus.SUBMITTED, StageStatus.DONE])
def test_retry_invalid_status_does_not_call_submit_stage(created_run, mocker, status):
    """Test that retry() never calls _submit_stage for invalid statuses"""
    object.__setattr__(created_run.stages[0].state, "_status", status)
    mock = mocker.patch.object(created_run, "_submit_stage")
    with pytest.raises(ValueError):
        created_run.retry("spinup_ad")
    mock.assert_not_called()


def test_retry_uses_current_stage_when_no_name_given(two_stage_run, mocker):
    """Test that retry() falls back to current_stage when stage_name is None"""
    two_stage_run.stages[0].state.status = StageStatus.FAILED
    mock = mocker.patch.object(two_stage_run, "_submit_stage", return_value="12345.pbs")
    two_stage_run.retry(None)
    mock.assert_called_once_with(
        two_stage_run.stages[0], dry_run=False, skip_script=False
    )


def test_retry_returns_none_when_nothing_to_retry(two_stage_run, mocker):
    """Test that retry(None) returns None when all stages are DONE"""
    for stage in two_stage_run.stages:
        stage.state.status = StageStatus.SUBMITTED
        stage.state.status = StageStatus.DONE
    mocker.patch.object(two_stage_run, "_submit_stage")
    result = two_stage_run.retry(None)
    assert result is None


def test_retry_dry_run_passed_through(created_run, mocker):
    """Test that retry() passes dry_run through to _submit_stage"""
    mock = mocker.patch.object(created_run, "_submit_stage", return_value="DRY_run")
    created_run.retry("spinup_ad", dry_run=True)
    mock.assert_called_once_with(created_run.stages[0], dry_run=True, skip_script=False)


def test_retry_skip_script_passed_through(created_run, mocker):
    """Test that retry() passes skip_script through to _submit_stage"""
    mock = mocker.patch.object(created_run, "_submit_stage", return_value="DRY_run")
    created_run.retry("spinup_ad", skip_script=True)
    mock.assert_called_once_with(created_run.stages[0], dry_run=False, skip_script=True)


def test_retry_unknown_stage_raises(created_run, mocker):
    """Test that retry() raises for an unknown stage name"""
    mocker.patch.object(created_run, "_submit_stage")
    with pytest.raises(ValueError, match="nonexistent"):
        created_run.retry("nonexistent")


# ---------------------------------------------------------------------------
# retry(lost=True)
# ---------------------------------------------------------------------------


def _submit_with_jobs(run: ModelRun) -> None:
    """Submit the first stage and record advance/fail jobs, as the setup job would"""
    run.submit()
    run.submit_advance("spinup_ad", "55555.fake")
    run.submit_fail("spinup_ad", "55555.fake")


def test_retry_lost_resubmits_stage(created_run, fake_scheduler):
    """Test that retry(lost=True) resubmits a lost stage with fresh job IDs,
    and the new state is saved to disk
    """
    _submit_with_jobs(created_run)
    old_job_id = created_run.stages[0].state.job_id
    # every job is gone (UNKNOWN by default in FakeScheduler)

    new_job_id = created_run.retry("spinup_ad", lost=True)

    state = created_run.stages[0].state
    assert new_job_id != old_job_id
    assert state.status is StageStatus.SUBMITTED
    assert state.job_id == new_job_id
    assert state.advance_job_id is None
    assert state.fail_job_ids == []
    assert state.attempts == 2
    reloaded = ModelRun.load(created_run.root, scheduler=fake_scheduler)
    assert reloaded.stages[0].state == state


def test_retry_lost_goes_through_failed(created_run, mocker):
    """Test that retry(lost=True) marks the stage FAILED before resetting it to
    PENDING (SUBMITTED -> PENDING isn't an allowed transition)
    """
    _submit_with_jobs(created_run)
    seen = []

    def check_status(stage, dry_run=False, skip_script=False):
        # pylint: disable=unused-argument
        seen.append(stage.state.status)
        return "12345.pbs"

    original_setter = type(created_run.stages[0].state).status.fset

    def tracking_setter(state, value):
        seen.append(value)
        original_setter(state, value)

    mocker.patch.object(
        type(created_run.stages[0].state),
        "status",
        property(type(created_run.stages[0].state).status.fget, tracking_setter),
    )
    mocker.patch.object(created_run, "_submit_stage", side_effect=check_status)
    created_run.retry("spinup_ad", lost=True)
    assert seen == [StageStatus.FAILED, StageStatus.PENDING, StageStatus.PENDING]


@pytest.mark.parametrize("active", [JobState.QUEUED, JobState.RUNNING])
def test_retry_lost_not_lost_raises(created_run, fake_scheduler, active):
    """Test that retry(lost=True) refuses a SUBMITTED stage that still has an
    active job, and leaves it SUBMITTED without resubmitting
    """
    _submit_with_jobs(created_run)
    fake_scheduler.job_states = {created_run.stages[0].state.advance_job_id: active}
    submitted_before = len(fake_scheduler.submitted)

    with pytest.raises(ValueError, match="not lost"):
        created_run.retry("spinup_ad", lost=True)

    assert created_run.stages[0].state.status is StageStatus.SUBMITTED
    assert len(fake_scheduler.submitted) == submitted_before
    reloaded = ModelRun.load(created_run.root, scheduler=fake_scheduler)
    assert reloaded.stages[0].state.status is StageStatus.SUBMITTED


@pytest.mark.parametrize("status", [StageStatus.PENDING, StageStatus.FAILED])
def test_retry_lost_on_non_submitted_stage_raises(created_run, mocker, status):
    """Test that retry(lost=True) raises for a stage that isn't SUBMITTED"""
    created_run.stages[0].state.status = status
    mock = mocker.patch.object(created_run, "_submit_stage")
    with pytest.raises(ValueError, match="not lost"):
        created_run.retry("spinup_ad", lost=True)
    mock.assert_not_called()


def test_retry_submitted_without_lost_still_raises(created_run, mocker):
    """Test that a lost stage still can't be retried without lost=True"""
    _submit_with_jobs(created_run)
    mock = mocker.patch.object(created_run, "_submit_stage")
    with pytest.raises(ValueError, match="PENDING or FAILED"):
        created_run.retry("spinup_ad")
    mock.assert_not_called()


def test_retry_lost_uses_current_stage_when_no_name_given(two_stage_run):
    """Test that retry(None, lost=True) checks and resubmits the current stage"""
    two_stage_run.submit()
    new_job_id = two_stage_run.retry(None, lost=True)
    assert two_stage_run.stages[0].state.job_id == new_job_id
    assert two_stage_run.stages[0].state.attempts == 2


def test_retry_lost_query_error_propagates(created_run, fake_scheduler, mocker):
    """Test that a scheduler failure stops retry(lost=True) without resubmitting"""
    _submit_with_jobs(created_run)
    mocker.patch.object(
        fake_scheduler, "query", side_effect=RuntimeError("qstat failed")
    )
    mock = mocker.patch.object(created_run, "_submit_stage")
    with pytest.raises(RuntimeError, match="qstat failed"):
        created_run.retry("spinup_ad", lost=True)
    mock.assert_not_called()
    assert created_run.stages[0].state.status is StageStatus.SUBMITTED
