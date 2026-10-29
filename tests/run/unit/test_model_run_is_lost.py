"""Tests for ModelRun.is_lost behavior"""

from __future__ import annotations

import time

import pytest

from meristem.run.model_run import _UNKNOWN_GRACE_SECONDS, ModelRun
from meristem.run.scheduler import JobState
from meristem.run.stages import StageStatus

SETUP, ADVANCE, FAIL, FAIL_2 = "1.fake", "2.fake", "3.fake", "4.fake"
CIME_RUN, CIME_ARCHIVE = "5.fake", "6.fake"


def _submit(
    run: ModelRun,
    job_id=SETUP,
    advance_job_id=None,
    fail_job_ids=(),
    age=2 * _UNKNOWN_GRACE_SECONDS,
    cime_job_ids=(),
):
    """Put the first stage in SUBMITTED with the given job IDs, saved to disk.
    By default submitted long enough ago that the UNKNOWN grace period is over
    """
    state = run.stages[0].state
    state.status = StageStatus.SUBMITTED
    state.submit_time = time.time() - age
    state.job_id = job_id
    state.advance_job_id = advance_job_id
    state.fail_job_ids = list(fail_job_ids)
    state.cime_job_ids = list(cime_job_ids)
    run.save()


@pytest.mark.parametrize("status", [StageStatus.PENDING, StageStatus.FAILED])
def test_is_lost_not_submitted_is_false(created_run, fake_scheduler, status):
    """Test that a stage that isn't SUBMITTED is never lost, and PBS isn't queried"""
    created_run.stages[0].state.status = status
    assert not created_run.is_lost("spinup_ad")
    assert not fake_scheduler.queried


def test_is_lost_done_is_false(created_run, fake_scheduler):
    """Test that a DONE stage is never lost, and PBS isn't queried"""
    _submit(created_run)
    created_run.stages[0].state.status = StageStatus.DONE
    assert not created_run.is_lost("spinup_ad")
    assert not fake_scheduler.queried


@pytest.mark.parametrize("setup_state", [JobState.QUEUED, JobState.RUNNING])
def test_is_lost_setup_job_active_is_false(created_run, fake_scheduler, setup_state):
    """Test that a stage whose setup job is still queued or running isn't lost"""
    _submit(created_run)
    fake_scheduler.job_states = {SETUP: setup_state}
    assert not created_run.is_lost("spinup_ad")


@pytest.mark.parametrize("setup_state", [JobState.FINISHED, JobState.UNKNOWN])
def test_is_lost_setup_job_gone_before_follow_up_jobs_is_true(
    created_run, fake_scheduler, setup_state
):
    """Test that a setup job that ended before submitting the advance and fail
    jobs (e.g. killed at walltime) leaves the stage lost
    """
    _submit(created_run)
    fake_scheduler.job_states = {SETUP: setup_state}
    assert created_run.is_lost("spinup_ad")


def test_is_lost_setup_job_unknown_within_grace_is_false(created_run, fake_scheduler):
    """Test that a just-submitted setup job missing from a lagging qstat cache
    (UNKNOWN) doesn't make the stage lost
    """
    _submit(created_run, age=60)
    fake_scheduler.job_states = {SETUP: JobState.UNKNOWN}
    assert not created_run.is_lost("spinup_ad")


def test_is_lost_setup_job_finished_within_grace_is_true(created_run, fake_scheduler):
    """Test that the grace period only covers UNKNOWN: a setup job PBS reports
    FINISHED before submitting the follow-up jobs leaves the stage lost
    """
    _submit(created_run, age=60)
    fake_scheduler.job_states = {SETUP: JobState.FINISHED}
    assert created_run.is_lost("spinup_ad")


def test_is_lost_follow_up_jobs_unknown_within_grace_is_true(
    created_run, fake_scheduler
):
    """Test that the grace period only covers the setup job: once the setup job
    has finished, UNKNOWN advance and fail jobs leave the stage lost
    """
    _submit(created_run, advance_job_id=ADVANCE, fail_job_ids=[FAIL], age=60)
    fake_scheduler.job_states = {
        SETUP: JobState.FINISHED,
        ADVANCE: JobState.UNKNOWN,
        FAIL: JobState.UNKNOWN,
    }
    assert created_run.is_lost("spinup_ad")


def test_is_lost_no_cime_stage_setup_job_gone_is_true(
    run_config_dict, no_cime_stage_config_dict, fake_scheduler
):
    """Test that a no_cime stage, which never has advance/fail jobs, is lost
    once its setup job is gone
    """
    run_config_dict["stages"] = [no_cime_stage_config_dict]
    run = ModelRun.create(run_config_dict, scheduler=fake_scheduler)
    _submit(run)
    fake_scheduler.job_states = {SETUP: JobState.FINISHED}
    assert run.is_lost("spinup_ad")


def test_is_lost_cime_running_is_false(created_run, fake_scheduler):
    """Test that a stage whose advance and fail jobs are waiting on the CIME run
    isn't lost, even though the setup job has finished
    """
    _submit(created_run, advance_job_id=ADVANCE, fail_job_ids=[FAIL])
    fake_scheduler.job_states = {
        SETUP: JobState.FINISHED,
        ADVANCE: JobState.QUEUED,
        FAIL: JobState.QUEUED,
    }
    assert not created_run.is_lost("spinup_ad")


@pytest.mark.parametrize("gone", [JobState.FINISHED, JobState.UNKNOWN])
def test_is_lost_all_jobs_gone_is_true(created_run, fake_scheduler, gone):
    """Test that a stage is lost once its setup, advance, and fail jobs are all
    finished or unknown to PBS
    """
    _submit(created_run, advance_job_id=ADVANCE, fail_job_ids=[FAIL])
    fake_scheduler.job_states = {SETUP: gone, ADVANCE: gone, FAIL: gone}
    assert created_run.is_lost("spinup_ad")


def test_is_lost_one_of_two_fail_jobs_queued_is_false(created_run, fake_scheduler):
    """Test that a stage isn't lost while any of its fail jobs is still waiting"""
    _submit(created_run, advance_job_id=ADVANCE, fail_job_ids=[FAIL, FAIL_2])
    fake_scheduler.job_states = {
        SETUP: JobState.FINISHED,
        ADVANCE: JobState.UNKNOWN,
        FAIL: JobState.UNKNOWN,
        FAIL_2: JobState.QUEUED,
    }
    assert not created_run.is_lost("spinup_ad")


def test_is_lost_two_fail_jobs_all_gone_is_true(created_run, fake_scheduler):
    """Test that a stage is lost once every job, including both fail jobs, is gone
    (e.g. an earlier segment of a resubmitting CIME run crashed)
    """
    _submit(created_run, advance_job_id=ADVANCE, fail_job_ids=[FAIL, FAIL_2])
    assert created_run.is_lost("spinup_ad")
    assert fake_scheduler.queried == [[SETUP, ADVANCE, FAIL, FAIL_2]]


def test_is_lost_only_advance_recorded_and_queued_is_false(created_run, fake_scheduler):
    """Test that a setup job killed between submitting the advance and fail jobs
    doesn't make the stage lost while the advance job is still waiting
    """
    _submit(created_run, advance_job_id=ADVANCE)
    fake_scheduler.job_states = {SETUP: JobState.FINISHED, ADVANCE: JobState.QUEUED}
    assert not created_run.is_lost("spinup_ad")


@pytest.mark.parametrize("running", [CIME_RUN, CIME_ARCHIVE])
def test_is_lost_setup_died_after_case_submit_cime_running_is_false(
    created_run, fake_scheduler, running
):
    """Test that a setup job killed after recording its CIME jobs, but before
    submitting the advance and fail jobs, doesn't make the stage lost while a
    CIME job is still going (retrying it would start a second CIME run)
    """
    _submit(created_run, cime_job_ids=[CIME_RUN, CIME_ARCHIVE])
    fake_scheduler.job_states = {SETUP: JobState.FINISHED, running: JobState.RUNNING}
    assert not created_run.is_lost("spinup_ad")


def test_is_lost_setup_died_after_case_submit_cime_finished_is_true(
    created_run, fake_scheduler
):
    """Test that the same stage is lost once its CIME jobs have finished, since
    nothing will mark it done or failed
    """
    _submit(created_run, cime_job_ids=[CIME_RUN, CIME_ARCHIVE])
    fake_scheduler.job_states = {
        job: JobState.FINISHED for job in (SETUP, CIME_RUN, CIME_ARCHIVE)
    }
    assert created_run.is_lost("spinup_ad")


def test_is_lost_queries_cime_jobs_too(created_run, fake_scheduler):
    """Test that is_lost's single query includes the CIME jobs, after the others"""
    _submit(
        created_run,
        advance_job_id=ADVANCE,
        fail_job_ids=[FAIL],
        cime_job_ids=[CIME_RUN, CIME_ARCHIVE],
    )
    created_run.is_lost("spinup_ad")
    assert fake_scheduler.queried == [[SETUP, ADVANCE, FAIL, CIME_RUN, CIME_ARCHIVE]]


def test_is_lost_queries_all_recorded_jobs_once(created_run, fake_scheduler):
    """Test that is_lost makes a single query for all recorded job IDs"""
    _submit(created_run, advance_job_id=ADVANCE, fail_job_ids=[FAIL])
    created_run.is_lost("spinup_ad")
    assert fake_scheduler.queried == [[SETUP, ADVANCE, FAIL]]


def test_is_lost_dry_run_is_false(created_run, fake_scheduler):
    """Test that a dry-run stage (only DRY_ IDs) isn't lost, and PBS isn't queried"""
    _submit(
        created_run,
        job_id="DRY_member_0001_spinup_ad",
        advance_job_id="DRY_advance_member_0001_spinup_ad",
        fail_job_ids=["DRY_fail_member_0001_spinup_ad"],
    )
    assert not created_run.is_lost("spinup_ad")
    assert not fake_scheduler.queried


def test_is_lost_query_error_propagates(created_run, fake_scheduler, mocker):
    """Test that a scheduler failure raises instead of reporting the stage lost"""
    _submit(created_run)
    mocker.patch.object(
        fake_scheduler, "query", side_effect=RuntimeError("qstat failed")
    )
    with pytest.raises(RuntimeError, match="qstat failed"):
        created_run.is_lost("spinup_ad")


def test_is_lost_stage_done_on_disk_after_query_is_false(
    created_run, fake_scheduler, mocker
):
    """Test that a stage the advance job marked DONE on disk after this run was
    loaded isn't lost, and the run picks up the DONE state
    """
    _submit(created_run, advance_job_id=ADVANCE, fail_job_ids=[FAIL])

    def advance_finishes(job_ids):
        # the advance job marks the stage DONE and exits before qstat runs
        other = ModelRun.load(created_run.root, scheduler=fake_scheduler)
        other.stages[0].state.status = StageStatus.DONE
        other.stages[0].state.end_time = time.time()
        other.save()
        return {job_id: JobState.FINISHED for job_id in job_ids}

    mocker.patch.object(fake_scheduler, "query", side_effect=advance_finishes)
    assert not created_run.is_lost("spinup_ad")
    assert created_run.stages[0].state.status is StageStatus.DONE


def test_is_lost_stage_retried_on_disk_after_query_is_false(
    created_run, fake_scheduler, mocker
):
    """Test that a stage retried elsewhere after this run was loaded (SUBMITTED
    again, with new job IDs) isn't reported lost on the old jobs' states
    """
    _submit(created_run, advance_job_id=ADVANCE, fail_job_ids=[FAIL])

    def retried_elsewhere(job_ids):
        other = ModelRun.load(created_run.root, scheduler=fake_scheduler)
        state = other.stages[0].state
        state.job_id, state.advance_job_id, state.fail_job_ids = "9.fake", None, []
        other.save()
        return {job_id: JobState.FINISHED for job_id in job_ids}

    mocker.patch.object(fake_scheduler, "query", side_effect=retried_elsewhere)
    assert not created_run.is_lost("spinup_ad")


def test_lost_stages_lists_lost_stages(two_stage_run, fake_scheduler):
    """Test that lost_stages() returns the lost stages and skips the rest"""
    two_stage_run.submit()  # first stage SUBMITTED; its job is UNKNOWN
    two_stage_run.stages[0].state.submit_time -= 2 * _UNKNOWN_GRACE_SECONDS
    two_stage_run.save()
    assert two_stage_run.lost_stages() == ["spinup_ad"]
    fake_scheduler.job_states = {two_stage_run.stages[0].state.job_id: JobState.RUNNING}
    assert not two_stage_run.lost_stages()


def test_lost_stages_does_not_change_state_file(created_run):
    """Test that lost_stages() (what --check runs) never writes the state file"""
    _submit(created_run)
    state_file = created_run.root / "run_state.json"
    before = state_file.read_text()
    assert created_run.lost_stages() == ["spinup_ad"]
    assert state_file.read_text() == before
    assert created_run.stages[0].state.status is StageStatus.SUBMITTED


def test_is_lost_unknown_stage_raises(created_run):
    """Test that is_lost raises for an unknown stage name"""
    with pytest.raises(ValueError, match="nonexistent_stage"):
        created_run.is_lost("nonexistent_stage")
