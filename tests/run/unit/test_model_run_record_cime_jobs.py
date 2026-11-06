"""Tests for ModelRun.record_cime_jobs behavior"""

from __future__ import annotations

import pytest

from meristem.run.model_run import ModelRun


def _saved_cime_job_ids(run: ModelRun) -> list[str]:
    """The first stage's CIME job IDs, as saved"""
    loaded = ModelRun.load(run.root, scheduler=run.scheduler)
    return loaded.stages[0].state.cime_job_ids


@pytest.mark.parametrize(
    "job_ids, expected",
    [
        (
            "case.run:100.desched1, case.st_archive:101.desched1",
            ["100.desched1", "101.desched1"],
        ),
        ("case.run:100.desched1", ["100.desched1"]),  # no archiver (DOUT_S FALSE)
        ("case.run:100,case.st_archive:101", ["100", "101"]),  # no space
        ("", []),  # case.submit reported nothing
    ],
)
def test_record_cime_jobs_saves_job_ids(created_run, job_ids, expected):
    """Test that record_cime_jobs saves every job ID in JOB_IDS, in order"""
    created_run.submit()
    assert created_run.record_cime_jobs("spinup_ad", job_ids, attempt=1) == expected
    assert _saved_cime_job_ids(created_run) == expected


def test_record_cime_jobs_replaces_previous(created_run):
    """Test that recording again replaces the stage's CIME job IDs"""
    created_run.submit()
    created_run.record_cime_jobs("spinup_ad", "case.run:100, case.st_archive:101")
    created_run.record_cime_jobs("spinup_ad", "case.run:200")
    assert _saved_cime_job_ids(created_run) == ["200"]


def test_record_cime_jobs_from_older_attempt_is_ignored(created_run):
    """Test that a record-cime-jobs from an older attempt's setup job changes
    nothing
    """
    created_run.submit()
    created_run.record_cime_jobs("spinup_ad", "case.run:100", attempt=1)
    assert created_run.record_cime_jobs("spinup_ad", "case.run:999", attempt=0) == []
    assert _saved_cime_job_ids(created_run) == ["100"]


def test_record_cime_jobs_unknown_stage_raises(created_run):
    """Test that record_cime_jobs raises for an unknown stage"""
    with pytest.raises(ValueError, match="No stage named"):
        created_run.record_cime_jobs("nope", "case.run:100")


_RESUBMIT_LOG = """\
Submitting job script ...
Submitted job id is 100.desched1
Submitted job id is 101.desched1
Submitted job id is 102.desched1
Submitted job id is 103.desched1
Submitted job id is 104.desched1
Submitted job id is 105.desched1
Submitted job case.run with id 104.desched1
Submitted job case.st_archive with id 105.desched1
"""
_LAST_SEGMENT = "case.run:104.desched1, case.st_archive:105.desched1"


def test_record_cime_jobs_reads_every_segment_from_submit_log(created_run, tmp_path):
    """Test that with RESUBMIT > 0 every segment's jobs are recorded, not just
    the last segment's (all JOB_IDS holds)
    """
    created_run.submit()
    log = tmp_path / "case_submit.log"
    log.write_text(_RESUBMIT_LOG)
    expected = [f"{n}.desched1" for n in range(100, 106)]
    assert (
        created_run.record_cime_jobs(
            "spinup_ad", _LAST_SEGMENT, attempt=1, submit_log=log
        )
        == expected
    )
    assert _saved_cime_job_ids(created_run) == expected


@pytest.mark.parametrize("log_text", ["no job lines here\n", None])
def test_record_cime_jobs_warns_when_log_lacks_job_ids(
    created_run, tmp_path, capsys, log_text
):
    """Test that a submit log that doesn't list JOB_IDS's jobs (or is missing)
    warns and still records JOB_IDS, rather than failing after jobs are queued
    """
    created_run.submit()
    log = tmp_path / "case_submit.log"
    if log_text is not None:
        log.write_text(log_text)
    recorded = created_run.record_cime_jobs(
        "spinup_ad", _LAST_SEGMENT, attempt=1, submit_log=log
    )
    assert recorded == ["104.desched1", "105.desched1"]
    assert "WARNING" in capsys.readouterr().out
