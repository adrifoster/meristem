"""Tests that ModelRun ignores calls from an older attempt's jobs, and that job
scripts carry their attempt
"""

from __future__ import annotations

import pytest

from meristem.run.model_run import ModelRun
from meristem.run.stages import StageStatus


@pytest.fixture
def retried_run(two_stage_run) -> ModelRun:
    """two_stage_run whose first stage failed on attempt 1 and was retried, so
    it's SUBMITTED on attempt 2
    """
    two_stage_run.submit()
    two_stage_run.fail("spinup_ad", attempt=1)
    two_stage_run.retry("spinup_ad")
    assert two_stage_run.stages[0].state.attempts == 2
    return two_stage_run


def _first_stage_on_disk(run: ModelRun):
    """The first stage's state, as saved"""
    return ModelRun.load(run.root, scheduler=run.scheduler).stages[0].state


def test_stale_advance_is_ignored(retried_run, fake_scheduler, capsys):
    """Test that an advance from attempt 1 doesn't finish attempt 2 or start
    the next stage
    """
    submitted = len(fake_scheduler.submitted)
    assert retried_run.advance("spinup_ad", attempt=1) is None
    for state in (retried_run.stages[0].state, _first_stage_on_disk(retried_run)):
        assert state.status is StageStatus.SUBMITTED
    assert retried_run.stages[1].state.status is StageStatus.PENDING
    assert len(fake_scheduler.submitted) == submitted
    assert (
        "ignoring advance from attempt 1; the stage is on attempt 2"
        in capsys.readouterr().out
    )


@pytest.mark.parametrize("attempt", [2, None])
def test_current_or_unset_attempt_advances(retried_run, attempt):
    """Test that an advance from the current attempt, or from a job script
    written before attempts were recorded (None), advances as usual
    """
    retried_run.advance("spinup_ad", attempt=attempt)
    assert retried_run.stages[0].state.status is StageStatus.DONE
    assert retried_run.stages[1].state.status is StageStatus.SUBMITTED


def test_stale_fail_is_ignored(retried_run):
    """Test that a fail from attempt 1 (e.g. its fail job) doesn't fail attempt 2"""
    retried_run.fail("spinup_ad", attempt=1)
    for state in (retried_run.stages[0].state, _first_stage_on_disk(retried_run)):
        assert state.status is StageStatus.SUBMITTED


@pytest.mark.parametrize("attempt", [2, None])
def test_current_or_unset_attempt_fails(retried_run, attempt):
    """Test that a fail from the current attempt, or with no attempt, fails it"""
    retried_run.fail("spinup_ad", attempt=attempt)
    assert _first_stage_on_disk(retried_run).status is StageStatus.FAILED


def test_stale_submit_advance_is_ignored(retried_run, fake_scheduler):
    """Test that a submit-advance from attempt 1's setup job submits nothing and
    doesn't record an advance job on attempt 2
    """
    submitted = len(fake_scheduler.submitted)
    assert retried_run.submit_advance("spinup_ad", "99.pbs", attempt=1) is None
    assert len(fake_scheduler.submitted) == submitted
    assert _first_stage_on_disk(retried_run).advance_job_id is None


def test_stale_submit_fail_is_ignored(retried_run, fake_scheduler):
    """Test that a submit-fail from attempt 1's setup job submits nothing and
    doesn't record fail jobs on attempt 2
    """
    submitted = len(fake_scheduler.submitted)
    assert retried_run.submit_fail("spinup_ad", ["99.pbs"], attempt=1) == []
    assert len(fake_scheduler.submitted) == submitted
    assert _first_stage_on_disk(retried_run).fail_job_ids == []


def test_current_attempt_submits_advance_and_fail(retried_run):
    """Test that submit-advance and submit-fail from the current attempt record
    their jobs
    """
    advance = retried_run.submit_advance("spinup_ad", "99.pbs", attempt=2)
    fails = retried_run.submit_fail("spinup_ad", ["98.pbs", "99.pbs"], attempt=2)
    state = _first_stage_on_disk(retried_run)
    assert state.advance_job_id == advance
    assert state.fail_job_ids == fails


def test_job_scripts_carry_their_attempt(retried_run):
    """Test that the setup, advance and fail job scripts written for attempt 2
    pass --attempt 2 back to meristem
    """
    root, run_id = retried_run.root, retried_run.run_id
    setup = (root / f"{run_id}_spinup_ad.pbs").read_text()
    assert "ATTEMPT=2\n" in setup

    retried_run.submit_advance("spinup_ad", "99.pbs", attempt=2)
    retried_run.submit_fail("spinup_ad", ["99.pbs"], attempt=2)
    for suffix in ("_advance", "_fail"):
        script = (root / f"{run_id}_spinup_ad{suffix}.pbs").read_text()
        assert "--stage spinup_ad --attempt 2" in script, suffix
