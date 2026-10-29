"""Tests for ModelRun.retry behavior"""

from __future__ import annotations

import copy

import pytest

from meristem.run.model_run import _UNKNOWN_GRACE_SECONDS, ModelRun
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


def test_retry_dry_run_writes_script_only(created_run, mocker):
    """Test that retry(dry_run=True) writes the job script without submitting"""
    created_run.stages[0].state.status = StageStatus.FAILED
    submit = mocker.patch.object(created_run, "_submit_stage")
    dry = mocker.spy(created_run, "_dry_run_stage")
    created_run.retry("spinup_ad", dry_run=True, skip_script=True)
    submit.assert_not_called()
    dry.assert_called_once_with(created_run.stages[0], skip_script=True)


def test_retry_dry_run_does_not_change_state(created_run):
    """Test that retry(dry_run=True) leaves a FAILED stage FAILED, in memory and on disk"""
    stage = created_run.stages[0]
    stage.state.status = StageStatus.SUBMITTED
    stage.state.job_id = "12345.pbs"
    stage.state.status = StageStatus.FAILED
    created_run.save()
    state = copy.deepcopy(stage.state)
    created_run.retry("spinup_ad", dry_run=True)
    assert created_run.stages[0].state == state
    reloaded = ModelRun.load(created_run.root, scheduler=created_run.scheduler)
    assert reloaded.stages[0].state == state


def test_retry_dry_run_rejects_submitted_stage(created_run):
    """Test that retry(dry_run=True) still rejects a stage that can't be retried"""
    created_run.stages[0].state.status = StageStatus.SUBMITTED
    with pytest.raises(ValueError, match="can only retry PENDING or FAILED"):
        created_run.retry("spinup_ad", dry_run=True)


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


def _age_submission(run: ModelRun) -> None:
    """Backdate the first stage's submission past the UNKNOWN grace period"""
    run.stages[0].state.submit_time -= 2 * _UNKNOWN_GRACE_SECONDS
    run.save()


def _submit_with_jobs(run: ModelRun) -> None:
    """Submit the first stage and record advance/fail jobs, as the setup job would"""
    run.submit()
    run.submit_advance("spinup_ad", "55555.fake")
    run.submit_fail("spinup_ad", "55555.fake")
    _age_submission(run)


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


def test_retry_lost_dry_run_does_not_change_state(created_run, fake_scheduler):
    """Test that retry(lost=True, dry_run=True) leaves a lost stage SUBMITTED with
    its old job IDs, and submits nothing
    """
    _submit_with_jobs(created_run)
    n_submitted = len(fake_scheduler.submitted)
    state = copy.deepcopy(created_run.stages[0].state)
    result = created_run.retry("spinup_ad", lost=True, dry_run=True)
    assert result.startswith("DRY_")
    assert len(fake_scheduler.submitted) == n_submitted
    assert created_run.stages[0].state == state
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
    _age_submission(two_stage_run)
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


@pytest.fixture
def two_script_run(
    tmp_path, ad_stage_config_dict, sasu_stage_config_dict, fake_scheduler
) -> ModelRun:
    """A two-stage ModelRun whose stages have separate scripts (ad.sh, sasu.sh)"""
    stages = []
    for stage_dict in (ad_stage_config_dict, sasu_stage_config_dict):
        script = tmp_path / f"{stage_dict['name'].split('_')[-1]}.sh"
        script.touch()
        stages.append({**stage_dict, "script": str(script)})
    return ModelRun.create(
        {"root": str(tmp_path / "run"), "project": "PROJ123", "stages": stages},
        scheduler=fake_scheduler,
    )


def test_retry_missing_later_script_raises(two_script_run, tmp_path, fake_scheduler):
    """Test that retry checks later stages' scripts up front, so a bad later
    stage fails now rather than inside an advance job
    """
    two_script_run.stages[0].state.status = StageStatus.FAILED
    two_script_run.save()
    (tmp_path / "sasu.sh").unlink()
    with pytest.raises(ValueError, match=r"cannot open script.*sasu\.sh"):
        two_script_run.retry("spinup_ad")
    assert fake_scheduler.submitted == []


def test_retry_invalid_config_leaves_state_unchanged(two_script_run, tmp_path):
    """Test that a failed check doesn't reset or save the stage"""
    stage = two_script_run.stages[0]
    stage.state.status = StageStatus.FAILED
    stage.state.job_id = "999.pbs"
    two_script_run.save()
    (tmp_path / "sasu.sh").unlink()
    with pytest.raises(ValueError):
        two_script_run.retry("spinup_ad")
    assert stage.state.status == StageStatus.FAILED
    assert stage.state.job_id == "999.pbs"
    loaded = ModelRun.load(two_script_run.root, scheduler=two_script_run.scheduler)
    assert loaded.stages[0].state.status == StageStatus.FAILED
    assert loaded.stages[0].state.job_id == "999.pbs"


def test_retry_ignores_earlier_stage_scripts(two_script_run, tmp_path, mocker):
    """Test that retry doesn't check stages before the retried one (they won't
    rerun), so a cleaned-up earlier script doesn't block it
    """
    two_script_run.stages[0].state.status = StageStatus.SUBMITTED
    two_script_run.stages[0].state.status = StageStatus.DONE
    two_script_run.stages[1].state.status = StageStatus.FAILED
    (tmp_path / "ad.sh").unlink()
    mock = mocker.patch.object(two_script_run, "_submit_stage", return_value="1.pbs")
    two_script_run.retry("spinup_sasu")
    mock.assert_called_once()


# retry while the failed attempt's jobs are still active


@pytest.fixture
def failed_with_jobs(created_run) -> ModelRun:
    """created_run whose stage failed with its setup (1.fake), advance (2.fake),
    fail (3.fake, 4.fake) and CIME jobs (5.fake, 6.fake) recorded, e.g. the
    setup job failed after case.submit
    """
    created_run.submit()
    state = created_run.stages[0].state
    state.advance_job_id = "2.fake"
    state.fail_job_ids = ["3.fake", "4.fake"]
    state.cime_job_ids = ["5.fake", "6.fake"]
    created_run.save()
    created_run.fail("spinup_ad")
    return created_run


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("job_state", [JobState.QUEUED, JobState.RUNNING])
@pytest.mark.parametrize("job_id", ["1.fake", "2.fake", "4.fake", "5.fake", "6.fake"])
def test_retry_failed_with_active_jobs_raises(
    failed_with_jobs, fake_scheduler, job_id, job_state, dry_run
):
    """Test that retrying a FAILED stage whose setup, advance or fail job is
    still queued or running raises, naming it, and changes and submits nothing
    """
    fake_scheduler.job_states = {job_id: job_state}
    submitted = len(fake_scheduler.submitted)
    with pytest.raises(ValueError, match=rf"still queued or running: \['{job_id}'\]"):
        failed_with_jobs.retry("spinup_ad", dry_run=dry_run)
    assert len(fake_scheduler.submitted) == submitted
    state = (
        ModelRun.load(failed_with_jobs.root, scheduler=fake_scheduler).stages[0].state
    )
    assert state.status is StageStatus.FAILED
    assert state.attempts == 1
    assert (state.job_id, state.advance_job_id) == ("1.fake", "2.fake")


def test_retry_failed_lists_every_active_job(failed_with_jobs, fake_scheduler):
    """Test that the error lists all the attempt's active jobs, in order"""
    fake_scheduler.job_states = {
        "1.fake": JobState.FINISHED,
        "2.fake": JobState.QUEUED,
        "3.fake": JobState.QUEUED,
        "4.fake": JobState.RUNNING,
    }
    with pytest.raises(ValueError, match=r"\['2.fake', '3.fake', '4.fake'\]"):
        failed_with_jobs.retry("spinup_ad")


@pytest.mark.parametrize("job_state", [JobState.FINISHED, JobState.UNKNOWN])
def test_retry_failed_with_finished_jobs(failed_with_jobs, fake_scheduler, job_state):
    """Test that a FAILED stage whose jobs are all finished (or gone from the
    scheduler) is retried
    """
    fake_scheduler.job_states = {
        job: job_state
        for job in ("1.fake", "2.fake", "3.fake", "4.fake", "5.fake", "6.fake")
    }
    failed_with_jobs.retry("spinup_ad")
    state = failed_with_jobs.stages[0].state
    assert state.status is StageStatus.SUBMITTED
    assert state.attempts == 2
    assert fake_scheduler.queried == [
        ["1.fake", "2.fake", "3.fake", "4.fake", "5.fake", "6.fake"]
    ]
    assert state.cime_job_ids == []  # reset for the new attempt


def test_retry_failed_without_recorded_jobs_skips_query(created_run, fake_scheduler):
    """Test that a FAILED stage with no recorded jobs (e.g. it was never
    submitted) is retried without querying the scheduler
    """
    created_run.stages[0].state.status = StageStatus.FAILED
    created_run.retry("spinup_ad")
    assert not fake_scheduler.queried
    assert created_run.stages[0].state.status is StageStatus.SUBMITTED


def test_retry_failed_skips_dry_run_job_ids(created_run, fake_scheduler):
    """Test that dry-run placeholder job IDs aren't sent to the scheduler"""
    state = created_run.stages[0].state
    state.status = StageStatus.FAILED
    state.advance_job_id = "DRY_advance_member_0001_spinup_ad"
    created_run.retry("spinup_ad")
    assert not fake_scheduler.queried


def test_retry_failed_query_error_raises(failed_with_jobs, fake_scheduler, mocker):
    """Test that a failed scheduler query stops the retry, rather than being
    taken as 'no jobs active'
    """
    mocker.patch.object(fake_scheduler, "query", side_effect=RuntimeError("qstat down"))
    with pytest.raises(RuntimeError, match="qstat down"):
        failed_with_jobs.retry("spinup_ad")
    assert failed_with_jobs.stages[0].state.status is StageStatus.FAILED


# retry a named stage before the stages before it are done


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize(
    "earlier", [StageStatus.FAILED, StageStatus.PENDING, StageStatus.SUBMITTED]
)
def test_retry_later_stage_before_earlier_done_raises(
    two_stage_run, fake_scheduler, earlier, dry_run
):
    """Test that retrying a named later stage raises, naming the earlier stage
    that isn't done, and changes and submits nothing
    """
    object.__setattr__(two_stage_run.stages[0].state, "_status", earlier)
    two_stage_run.save()
    with pytest.raises(
        ValueError,
        match=rf"Can't retry 'spinup_sasu' before 'spinup_ad' is done "
        rf"\(it's {earlier.value}\); retry 'spinup_ad' first",
    ):
        two_stage_run.retry("spinup_sasu", dry_run=dry_run)
    assert not fake_scheduler.submitted
    loaded = ModelRun.load(two_stage_run.root, scheduler=fake_scheduler)
    assert [s.state.status for s in loaded.stages] == [earlier, StageStatus.PENDING]


@pytest.mark.parametrize("later", [StageStatus.PENDING, StageStatus.FAILED])
def test_retry_later_stage_after_earlier_done(two_stage_run, later):
    """Test that a named later stage is retried once every stage before it is done"""
    object.__setattr__(two_stage_run.stages[0].state, "_status", StageStatus.DONE)
    object.__setattr__(two_stage_run.stages[1].state, "_status", later)
    two_stage_run.retry("spinup_sasu")
    assert two_stage_run.stages[1].state.status is StageStatus.SUBMITTED


def test_retry_without_stage_picks_first_unfinished(two_stage_run):
    """Test that retry with no stage name retries the first stage that isn't
    done, so it never trips the earlier-stages check
    """
    object.__setattr__(two_stage_run.stages[0].state, "_status", StageStatus.FAILED)
    two_stage_run.retry(None)
    assert two_stage_run.stages[0].state.status is StageStatus.SUBMITTED
    assert two_stage_run.stages[1].state.status is StageStatus.PENDING


def test_unfinished_before(two_stage_run):
    """Test that unfinished_before returns the first earlier stage that isn't
    done, or None for the first stage or when all earlier stages are done
    """
    assert two_stage_run.unfinished_before("spinup_ad") is None
    assert two_stage_run.unfinished_before("spinup_sasu").config.name == "spinup_ad"
    object.__setattr__(two_stage_run.stages[0].state, "_status", StageStatus.DONE)
    assert two_stage_run.unfinished_before("spinup_sasu") is None
    with pytest.raises(ValueError, match="No stage named 'nope'"):
        two_stage_run.unfinished_before("nope")
