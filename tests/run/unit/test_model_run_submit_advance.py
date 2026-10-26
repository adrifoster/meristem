"""Tests for ModelRun.submit_advance behavior"""

from __future__ import annotations

import subprocess
import time
import pytest

from meristem.run.model_run import ModelRun
from meristem.run.stages import StageStatus


def test_submit_advance_unknown_stage_raises(created_run):
    """Test that submit_advance() raises for an unknown stage name"""
    with pytest.raises(ValueError, match="nonexistent_stage"):
        created_run.submit_advance("nonexistent_stage", "5555.deched")


def test_submit_advance_writes_jobscript(created_run):
    """Test that ModelRun.submit_advance() actually writes a jobscript"""
    created_run.stages[0].state.status = StageStatus.SUBMITTED
    created_run.stages[0].state.submit_time = time.time() - 100
    created_run.submit_advance("spinup_ad", "55555.deched")
    job_file = created_run.root / f"{created_run.run_id}_spinup_ad_advance.pbs"
    assert job_file.exists()


@pytest.mark.parametrize(
    "field,expected",
    [
        ("PBS -N ", "job_name"),
        ("#PBS -q ", "queue"),
        ("#PBS -A ", "project"),
        ("#PBS -M ", "email"),
        ("#PBS -W depend=afterok:", "cime_job_id"),
    ],
)
def test_submit_advance_script_contains_pbs_directives(created_run, field, expected):
    """Test that ModelRun.submit_advance() creates a PBS file with the correct PBS directives"""

    stage = created_run.stages[0]
    stage.state.status = StageStatus.SUBMITTED
    stage.state.submit_time = time.time() - 100

    cime_job_id = "55555.deched"
    created_run.submit_advance("spinup_ad", cime_job_id)

    job_file = (
        created_run.root / f"{created_run.run_id}_spinup_ad_advance.pbs"
    ).read_text()
    values = {
        "job_name": f"{created_run.run_id}_{stage.config.name}",
        "queue": stage.config.queue,
        "project": created_run.project,
        "email": created_run.email,
        "cime_job_id": cime_job_id,
    }
    assert field + values[expected] in job_file


def test_submit_advance_script_references_run_root(created_run):
    """Test that the job script references the correct run root"""
    stage = created_run.stages[0]
    stage.state.status = StageStatus.SUBMITTED
    stage.state.submit_time = time.time() - 100

    created_run.submit_advance("spinup_ad", "55555.deched")
    content = (
        created_run.root / f"{created_run.run_id}_spinup_ad_advance.pbs"
    ).read_text()
    assert f"meristem run advance --root {created_run.root}" in content


def test_submit_advance_script_references_stage_name(created_run):
    """Test that the job script references the correct stage name"""
    stage = created_run.stages[0]
    stage.state.status = StageStatus.SUBMITTED
    stage.state.submit_time = time.time() - 100

    created_run.submit_advance("spinup_ad", "55555.deched")
    content = (
        created_run.root / f"{created_run.run_id}_spinup_ad_advance.pbs"
    ).read_text()
    assert f"--stage {created_run.stages[0].config.name}" in content


def test_submit_advance_returns_job_id(created_run):
    """Test that submit_advance() returns the PBS job ID"""
    stage = created_run.stages[0]
    stage.state.status = StageStatus.SUBMITTED
    stage.state.submit_time = time.time() - 100
    result = created_run.submit_advance("spinup_ad", "55555.pbs")
    assert result == "1.fake"


def test_submit_advance_calls_qsub(created_run, fake_scheduler):
    """Test that submit_advance() calls qsub"""
    stage = created_run.stages[0]
    stage.state.status = StageStatus.SUBMITTED
    stage.state.submit_time = time.time() - 100
    created_run.submit_advance("spinup_ad", "55555.pbs")
    assert len(fake_scheduler.submitted) == 1


def test_submit_advance_dry_run_does_not_call_qsub(created_run, fake_scheduler):
    """Test that submit_advance(dry_run=True) does not call qsub"""
    stage = created_run.stages[0]
    stage.state.status = StageStatus.SUBMITTED
    stage.state.submit_time = time.time() - 100
    created_run.submit_advance("spinup_ad", "55555.pbs", dry_run=True)
    assert not fake_scheduler.submitted


def test_submit_advance_dry_run_job_id_format(created_run):
    """Test that submit_advance(dry_run=True) produces a synthetic DRY_ job ID"""
    stage = created_run.stages[0]
    stage.state.status = StageStatus.SUBMITTED
    stage.state.submit_time = time.time() - 100
    result = created_run.submit_advance("spinup_ad", "55555.pbs", dry_run=True)
    assert result == f"DRY_advance_{created_run.run_id}_spinup_ad"


def test_submit_advance_dry_run_still_writes_script(created_run):
    """Test that submit_advance(dry_run=True) still writes the advance script"""
    stage = created_run.stages[0]
    stage.state.status = StageStatus.SUBMITTED
    stage.state.submit_time = time.time() - 100
    created_run.submit_advance("spinup_ad", "55555.pbs", dry_run=True)
    job_file = created_run.root / f"{created_run.run_id}_spinup_ad_advance.pbs"
    assert job_file.exists()


def test_submit_advance_qsub_failure_raises(created_run, mocker):
    """Test that a qsub failure propagates out of submit_advance()"""
    mocker.patch.object(
        created_run.scheduler,
        "submit",
        side_effect=subprocess.CalledProcessError(1, "qsub"),
    )
    with pytest.raises(subprocess.CalledProcessError):
        created_run.submit_advance("spinup_ad", "55555.pbs")


def test_submit_advance_does_not_update_stage_status_or_job_id(created_run):
    """submit_advance() does not update stage status or the setup job_id
    (it only records advance_job_id)
    """
    stage = created_run.stages[0]
    stage.state.status = StageStatus.SUBMITTED
    stage.state.submit_time = time.time() - 100
    created_run.submit_advance("spinup_ad", "55555.pbs")
    assert created_run.stages[0].state.status is StageStatus.SUBMITTED
    assert created_run.stages[0].state.job_id is None


def test_submit_advance_records_advance_job_id(created_run):
    """Test that submit_advance() saves the advance job ID on the stage and to disk"""
    stage = created_run.stages[0]
    stage.state.status = StageStatus.SUBMITTED
    stage.state.submit_time = time.time() - 100
    job_id = created_run.submit_advance("spinup_ad", "55555.pbs")
    assert created_run.stages[0].state.advance_job_id == job_id
    reloaded = ModelRun.load(created_run.root, scheduler=created_run.scheduler)
    assert reloaded.stages[0].state.advance_job_id == job_id


def test_submit_advance_dry_run_does_not_record_job_id(created_run):
    """Test that submit_advance(dry_run=True) doesn't record the DRY_ job ID"""
    stage = created_run.stages[0]
    stage.state.status = StageStatus.SUBMITTED
    stage.state.submit_time = time.time() - 100
    created_run.save()
    job_id = created_run.submit_advance("spinup_ad", "55555.pbs", dry_run=True)
    assert job_id.startswith("DRY_")
    assert created_run.stages[0].state.advance_job_id is None
    reloaded = ModelRun.load(created_run.root, scheduler=created_run.scheduler)
    assert reloaded.stages[0].state.advance_job_id is None
