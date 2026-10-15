"""Tests for scheduler implementations"""

from __future__ import annotations

import subprocess

import pytest

from meristem.run.scheduler import PbsScheduler, get_scheduler


def test_pbs_submit_calls_qsub_with_correct_args(tmp_path, mocker):
    """Test that PbsScheduler.submit() calls qsub with the correct arguments"""
    mock_run = mocker.patch("meristem.run.scheduler.subprocess.run")
    mock_run.return_value.stdout = "12345.pbs\n"
    job_file = tmp_path / "job.pbs"

    PbsScheduler().submit(job_file)

    mock_run.assert_called_once_with(
        ["qsub", str(job_file)],
        capture_output=True,
        text=True,
        check=True,
    )


def test_pbs_submit_returns_stripped_job_id(tmp_path, mocker):
    """Test that PbsScheduler.submit() returns the job ID without trailing whitespace"""
    mock_run = mocker.patch("meristem.run.scheduler.subprocess.run")
    mock_run.return_value.stdout = "12345.pbs\n"
    assert PbsScheduler().submit(tmp_path / "job.pbs") == "12345.pbs"


def test_pbs_submit_failure_raises(tmp_path, mocker):
    """Test that a failing qsub raises CalledProcessError"""
    mocker.patch(
        "meristem.run.scheduler.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "qsub"),
    )
    with pytest.raises(subprocess.CalledProcessError):
        PbsScheduler().submit(tmp_path / "job.pbs")


def test_get_scheduler_pbs():
    """Test that get_scheduler("pbs") returns a PbsScheduler"""
    assert isinstance(get_scheduler("pbs"), PbsScheduler)


def test_get_scheduler_unknown_raises():
    """Test that get_scheduler raises for an unknown name and lists known ones"""
    with pytest.raises(ValueError, match="nonexistent.*pbs"):
        get_scheduler("nonexistent")
