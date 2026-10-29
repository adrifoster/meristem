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
