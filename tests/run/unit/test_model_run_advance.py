"""Tests for ModelRun.advance behavior"""

from __future__ import annotations

import copy

import pytest
import time

from meristem.run.model_run import ModelRun
from meristem.run.stages import StageStatus


def test_advance_marks_stage_done(two_stage_run):
    """Test that ModelRun.advance() marks the completed stage as DONE"""
    two_stage_run.submit()
    two_stage_run.advance("spinup_ad")
    assert two_stage_run.stages[0].state.status is StageStatus.DONE


def test_advance_sets_end_time(two_stage_run):
    """Test that ModelRun.advance() sets end_time on the completed stage"""
    two_stage_run.submit()
    two_stage_run.advance("spinup_ad")
    assert two_stage_run.stages[0].state.end_time is not None


def test_advance_persists_completed_stage_to_disk(two_stage_run):
    """Test that ModelRun.advance() saves the DONE state to disk"""
    two_stage_run.submit()
    two_stage_run.advance("spinup_ad")
    loaded = ModelRun.load(two_stage_run.root, scheduler=two_stage_run.scheduler)
    assert loaded.stages[0].state.status is StageStatus.DONE


def test_advance_submits_next_stage(two_stage_run):
    """Test that advance() submits the next PENDING stage"""
    two_stage_run.submit()
    two_stage_run.advance("spinup_ad")
    assert two_stage_run.stages[1].state.status is StageStatus.SUBMITTED


def test_advance_next_stage_job_id_set(two_stage_run):
    """Test that advance() sets a job ID on the next stage"""
    two_stage_run.submit()
    two_stage_run.advance("spinup_ad")
    assert two_stage_run.stages[1].state.job_id == "2.fake"


def test_advance_returns_next_job_id(two_stage_run):
    """Test that advance() returns the PBS job ID of the next stage"""
    two_stage_run.submit()
    result = two_stage_run.advance("spinup_ad")
    assert result == "2.fake"


def test_advance_persists_next_stage_submission(two_stage_run):
    """Test that advance() persists the next stage submission to disk"""
    two_stage_run.submit()
    two_stage_run.advance("spinup_ad")
    loaded = ModelRun.load(two_stage_run.root, scheduler=two_stage_run.scheduler)
    assert loaded.stages[1].state.status is StageStatus.SUBMITTED


def test_advance_last_stage_returns_none(two_stage_run):
    """Test that advance() returns None when the last stage completes"""
    two_stage_run.submit()
    two_stage_run.advance("spinup_ad")
    result = two_stage_run.advance("spinup_sasu")
    assert result is None


def test_advance_last_stage_does_not_call_qsub(two_stage_run, fake_scheduler):
    """Test that advance() does not call qsub when the last stage completes"""
    object.__setattr__(two_stage_run.stages[0].state, "_status", StageStatus.DONE)
    object.__setattr__(two_stage_run.stages[1].state, "_status", StageStatus.SUBMITTED)
    object.__setattr__(two_stage_run.stages[1].state, "_submit_time", time.time() - 100)
    two_stage_run.advance("spinup_sasu")
    assert not fake_scheduler.submitted


def test_advance_last_stage_still_marks_done(two_stage_run):
    """Test that advance() marks the last stage DONE even with no next stage"""
    object.__setattr__(two_stage_run.stages[0].state, "_status", StageStatus.DONE)
    object.__setattr__(two_stage_run.stages[1].state, "_status", StageStatus.SUBMITTED)
    object.__setattr__(two_stage_run.stages[1].state, "_submit_time", time.time() - 100)
    two_stage_run.advance("spinup_sasu")
    assert two_stage_run.stages[1].state.status is StageStatus.DONE


def test_advance_dry_run_does_not_call_qsub(two_stage_run, fake_scheduler):
    """Test that advance(dry_run=True) does not call qsub"""
    two_stage_run.stages[0].state.status = StageStatus.SUBMITTED
    two_stage_run.stages[0].state.submit_time = time.time() - 100
    two_stage_run.advance("spinup_ad", dry_run=True)
    assert not fake_scheduler.submitted


def test_advance_dry_run_does_not_change_state(two_stage_run):
    """Test that advance(dry_run=True) doesn't mark the stage DONE or submit the
    next one, in memory or on disk
    """
    two_stage_run.submit()
    states = copy.deepcopy([stage.state for stage in two_stage_run.stages])
    two_stage_run.advance("spinup_ad", dry_run=True)
    assert [stage.state for stage in two_stage_run.stages] == states
    reloaded = ModelRun.load(two_stage_run.root, scheduler=two_stage_run.scheduler)
    assert [stage.state for stage in reloaded.stages] == states


def test_advance_dry_run_writes_next_stage_script(two_stage_run):
    """Test that advance(dry_run=True) writes the next stage's job script"""
    two_stage_run.submit()
    result = two_stage_run.advance("spinup_ad", dry_run=True)
    assert result == f"DRY_{two_stage_run.run_id}_spinup_sasu"
    assert (two_stage_run.root / f"{two_stage_run.run_id}_spinup_sasu.pbs").exists()


def test_advance_dry_run_last_stage_returns_none(two_stage_run):
    """Test that advance(dry_run=True) on the last stage returns None"""
    object.__setattr__(two_stage_run.stages[0].state, "_status", StageStatus.DONE)
    object.__setattr__(two_stage_run.stages[1].state, "_status", StageStatus.SUBMITTED)
    assert two_stage_run.advance("spinup_sasu", dry_run=True) is None
    assert two_stage_run.stages[1].state.status is StageStatus.SUBMITTED


def test_advance_dry_run_non_submitted_raises(two_stage_run):
    """Test that advance(dry_run=True) still rejects a stage that isn't SUBMITTED"""
    with pytest.raises(ValueError, match="can only advance SUBMITTED"):
        two_stage_run.advance("spinup_ad", dry_run=True)


def test_advance_unknown_stage_raises(two_stage_run):
    """Test that advance() raises for an unknown stage name"""
    with pytest.raises(ValueError, match="nonexistent_stage"):
        two_stage_run.advance("nonexistent_stage")


def test_advance_does_not_affect_other_stages(two_stage_run):
    """Test that advance() only modifies the completed and next stages"""
    two_stage_run.submit()
    two_stage_run.advance("spinup_ad")
    # completed stage is DONE, next is SUBMITTED — no others should change
    assert two_stage_run.stages[0].state.status is StageStatus.DONE
    assert two_stage_run.stages[1].state.status is StageStatus.SUBMITTED


@pytest.mark.parametrize(
    "cant_advance_status",
    [
        StageStatus.FAILED,
        StageStatus.PENDING,
        StageStatus.DONE,
    ],
)
def test_advance_non_pending_or_failed_rases(created_run, cant_advance_status):
    """Test that ModelRun.advance() raises a value error if status isn't PENDING or FAILED"""
    object.__setattr__(created_run.stages[0].state, "_status", cant_advance_status)
    with pytest.raises(ValueError, match="advance"):
        created_run.advance("spinup_ad")
