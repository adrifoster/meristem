"""Tests for ModelRun.format_status behavior"""

from __future__ import annotations

import pytest

from meristem.run.stages import StageStatus


def test_format_status_contains_run_id(created_run):
    """Test that format_status() includes the run ID"""
    assert created_run.run_id in created_run.format_status()


def test_format_status_contains_root(created_run):
    """Test that format_status() includes the root path"""
    assert str(created_run.root) in created_run.format_status()


def test_format_status_contains_stage_name(created_run):
    """Test that format_status() includes the stage name"""
    assert created_run.stages[0].config.name in created_run.format_status()


def test_format_status_contains_status_value(created_run):
    """Test that format_status() includes the status string"""
    assert "pending" in created_run.format_status()


@pytest.mark.parametrize(
    "status,icon",
    [
        (StageStatus.PENDING, "·"),
        (StageStatus.SUBMITTED, "..."),
        (StageStatus.DONE, "✓"),
        (StageStatus.FAILED, "✗"),
    ],
)
def test_format_status_correct_icon(created_run, status, icon):
    """Test that format_status() uses the correct icon for each status"""
    object.__setattr__(created_run.stages[0].state, "_status", status)
    assert icon in created_run.format_status()


def test_format_status_job_id_shown_when_present(created_run):
    """Test that format_status() shows the job ID when set"""
    created_run.stages[0].state.job_id = "12345.pbs"
    assert "12345.pbs" in created_run.format_status()


def test_format_status_job_id_not_shown_when_absent(created_run):
    """Test that format_status() omits the job ID bracket when not set"""
    assert "[" not in created_run.format_status()


def test_format_status_attempts_not_shown_on_first_submission(created_run):
    """Test that format_status() omits attempts on the first submission"""
    created_run.stages[0].state.increment_attempts()
    assert "attempts=" not in created_run.format_status()


def test_format_status_attempts_shown_after_retry(created_run):
    """Test that format_status() shows attempts when greater than 1"""
    created_run.stages[0].state.increment_attempts()
    created_run.stages[0].state.increment_attempts()
    assert "attempts=2" in created_run.format_status()


def test_format_status_multiple_stages(two_stage_run):
    """Test that format_status() includes all stage names"""
    output = two_stage_run.format_status()
    for stage in two_stage_run.stages:
        assert stage.config.name in output


def test_format_status_returns_string(created_run):
    """Test that format_status() returns a string"""
    assert isinstance(created_run.format_status(), str)


def test_format_status_shows_lost_stage(two_stage_run):
    """Test that a stage passed in lost is shown as lost, and others are not"""
    two_stage_run.submit()
    lines = two_stage_run.format_status(lost=["spinup_ad"]).splitlines()
    ad_line = next(line for line in lines if "spinup_ad" in line)
    sasu_line = next(line for line in lines if "spinup_sasu" in line)
    assert ad_line.lstrip().startswith("?")
    assert "lost" in ad_line
    assert "submitted" not in ad_line
    assert "lost" not in sasu_line
    assert "pending" in sasu_line


def test_format_status_lost_still_shows_job_id(created_run):
    """Test that a lost stage still shows its job ID"""
    created_run.submit()
    job_id = created_run.stages[0].state.job_id
    assert f"[{job_id}]" in created_run.format_status(lost=["spinup_ad"])
