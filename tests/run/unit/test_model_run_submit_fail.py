"""Tests for ModelRun.submit_fail behavior"""

from __future__ import annotations

import subprocess
import time
import pytest

from meristem.run.model_run import ModelRun
from meristem.run.stages import StageStatus


def test_submit_fail_unknown_stage_raises(created_run):
    """Test that submit_fail() raises for an unknown stage name"""
    with pytest.raises(ValueError, match="nonexistent_stage"):
        created_run.submit_fail("nonexistent_stage", "5555.deched")


def test_submit_fail_writes_jobscript(created_run):
    """Test that ModelRun.submit_fail() actually writes a jobscript"""
    created_run.stages[0].state.status = StageStatus.SUBMITTED
    created_run.stages[0].state.submit_time = time.time() - 100
    created_run.submit_fail("spinup_ad", "55555.deched")
    job_file = created_run.root / f"{created_run.run_id}_spinup_ad_fail.pbs"
    assert job_file.exists()


@pytest.mark.parametrize(
    "field,expected",
    [
        ("PBS -N ", "job_name"),
        ("#PBS -q ", "queue"),
        ("#PBS -A ", "project"),
        ("#PBS -W depend=afternotok:", "cime_job_id"),
    ],
)
def test_submit_fail_script_contains_pbs_directives(created_run, field, expected):
    """Test that ModelRun.submit_fail() creates a PBS file with the correct PBS directives"""

    stage = created_run.stages[0]
    stage.state.status = StageStatus.SUBMITTED
    stage.state.submit_time = time.time() - 100

    cime_job_id = "55555.deched"
    created_run.submit_fail("spinup_ad", cime_job_id)

    job_file = (
        created_run.root / f"{created_run.run_id}_spinup_ad_fail.pbs"
    ).read_text()
    values = {
        "job_name": f"{created_run.run_id}_{stage.config.name}",
        "queue": stage.config.queue,
        "project": created_run.project,
        "cime_job_id": cime_job_id,
    }
    assert field + values[expected] in job_file


def test_submit_fail_script_references_run_root(created_run):
    """Test that the job script references the correct run root"""
    stage = created_run.stages[0]
    stage.state.status = StageStatus.SUBMITTED
    stage.state.submit_time = time.time() - 100

    created_run.submit_fail("spinup_ad", "55555.deched")
    content = (
        created_run.root / f"{created_run.run_id}_spinup_ad_fail.pbs"
    ).read_text()
    assert f"meristem run fail --root {created_run.root}" in content


def test_submit_fail_script_references_stage_name(created_run):
    """Test that the job script references the correct stage name"""
    stage = created_run.stages[0]
    stage.state.status = StageStatus.SUBMITTED
    stage.state.submit_time = time.time() - 100

    created_run.submit_fail("spinup_ad", "55555.deched")
    content = (
        created_run.root / f"{created_run.run_id}_spinup_ad_fail.pbs"
    ).read_text()
    assert f"--stage {created_run.stages[0].config.name}" in content


def test_submit_fail_returns_job_id(created_run):
    """Test that submit_fail() returns the PBS job ID"""
    stage = created_run.stages[0]
    stage.state.status = StageStatus.SUBMITTED
    stage.state.submit_time = time.time() - 100
    result = created_run.submit_fail("spinup_ad", "55555.pbs")
    assert result == ["1.fake"]


def test_submit_fail_calls_qsub(created_run, fake_scheduler):
    """Test that submit_fail() calls qsub"""
    stage = created_run.stages[0]
    stage.state.status = StageStatus.SUBMITTED
    stage.state.submit_time = time.time() - 100
    created_run.submit_fail("spinup_ad", "55555.pbs")
    assert len(fake_scheduler.submitted) == 1


def test_submit_fail_dry_run_does_not_call_qsub(created_run, fake_scheduler):
    """Test that submit_fail(dry_run=True) does not call qsub"""
    stage = created_run.stages[0]
    stage.state.status = StageStatus.SUBMITTED
    stage.state.submit_time = time.time() - 100
    created_run.submit_fail("spinup_ad", "55555.pbs", dry_run=True)
    assert not fake_scheduler.submitted


def test_submit_fail_dry_run_job_id_format(created_run):
    """Test that submit_fail(dry_run=True) produces a synthetic DRY_ job ID"""
    stage = created_run.stages[0]
    stage.state.status = StageStatus.SUBMITTED
    stage.state.submit_time = time.time() - 100
    result = created_run.submit_fail("spinup_ad", "55555.pbs", dry_run=True)
    assert result == [f"DRY_fail_{created_run.run_id}_spinup_ad"]


def test_submit_fail_dry_run_still_writes_script(created_run):
    """Test that submit_fail(dry_run=True) still writes the advance script"""
    stage = created_run.stages[0]
    stage.state.status = StageStatus.SUBMITTED
    stage.state.submit_time = time.time() - 100
    created_run.submit_fail("spinup_ad", "55555.pbs", dry_run=True)
    job_file = created_run.root / f"{created_run.run_id}_spinup_ad_fail.pbs"
    assert job_file.exists()


def test_submit_fail_qsub_failure_raises(created_run, mocker):
    """Test that a qsub failure propagates out of submit_advance()"""
    mocker.patch.object(
        created_run.scheduler,
        "submit",
        side_effect=subprocess.CalledProcessError(1, "qsub"),
    )
    with pytest.raises(subprocess.CalledProcessError):
        created_run.submit_fail("spinup_ad", "55555.pbs")


def test_submit_fail_does_not_update_stage_status_or_job_id(created_run):
    """submit_fail() does not update stage status or the setup job_id
    (it only records fail_job_ids)
    """
    stage = created_run.stages[0]
    stage.state.status = StageStatus.SUBMITTED
    stage.state.submit_time = time.time() - 100
    created_run.submit_fail("spinup_ad", "55555.pbs")
    assert created_run.stages[0].state.status is StageStatus.SUBMITTED
    assert created_run.stages[0].state.job_id is None


def test_submit_fail_records_fail_job_ids(created_run):
    """Test that submit_fail() saves the fail job ID on the stage and to disk"""
    stage = created_run.stages[0]
    stage.state.status = StageStatus.SUBMITTED
    stage.state.submit_time = time.time() - 100
    job_ids = created_run.submit_fail("spinup_ad", "55555.pbs")
    assert created_run.stages[0].state.fail_job_ids == job_ids
    reloaded = ModelRun.load(created_run.root, scheduler=created_run.scheduler)
    assert reloaded.stages[0].state.fail_job_ids == job_ids


def test_submit_fail_dry_run_does_not_record_job_ids(created_run):
    """Test that submit_fail(dry_run=True) doesn't record the DRY_ job IDs"""
    stage = created_run.stages[0]
    stage.state.status = StageStatus.SUBMITTED
    stage.state.submit_time = time.time() - 100
    stage.state.fail_job_ids = ["111.pbs"]
    created_run.save()
    job_ids = created_run.submit_fail(
        "spinup_ad", ["55555.pbs", "55556.pbs"], dry_run=True
    )
    assert job_ids == [
        f"DRY_fail_1_{created_run.run_id}_spinup_ad",
        f"DRY_fail_2_{created_run.run_id}_spinup_ad",
    ]
    assert created_run.stages[0].state.fail_job_ids == ["111.pbs"]
    reloaded = ModelRun.load(created_run.root, scheduler=created_run.scheduler)
    assert reloaded.stages[0].state.fail_job_ids == ["111.pbs"]


def test_submit_fail_script_name_not_doubled(created_run):
    """Test that the fail job's PBS name has a single _fail suffix"""
    created_run.submit_fail("spinup_ad", "55555.pbs")
    content = (
        created_run.root / f"{created_run.run_id}_spinup_ad_fail.pbs"
    ).read_text()
    assert f"#PBS -N {created_run.run_id}_spinup_ad_fail\n" in content
    assert "_fail_fail" not in content


def test_submit_fail_script_sends_no_mail(created_run):
    """Test that the fail job doesn't mail when PBS aborts it after a good run"""
    created_run.submit_fail("spinup_ad", "55555.pbs")
    content = (
        created_run.root / f"{created_run.run_id}_spinup_ad_fail.pbs"
    ).read_text()
    assert "#PBS -m n\n" in content


# ---------------------------------------------------------------------------
# one fail job per CIME job (case.run and case.st_archive)
# ---------------------------------------------------------------------------

RUN_JOB, ARCHIVE_JOB = "100.desched1", "101.desched1"


def test_submit_fail_two_ids_submits_two_jobs(created_run, fake_scheduler):
    """Test that each CIME job ID gets its own fail job, all recorded"""
    job_ids = created_run.submit_fail("spinup_ad", [RUN_JOB, ARCHIVE_JOB])
    assert job_ids == ["1.fake", "2.fake"]
    assert len(fake_scheduler.submitted) == 2
    reloaded = ModelRun.load(created_run.root, scheduler=fake_scheduler)
    assert reloaded.stages[0].state.fail_job_ids == job_ids


def test_submit_fail_two_ids_each_script_depends_on_its_job(created_run):
    """Test that each fail script has its own name and afternotok dependency"""
    created_run.submit_fail("spinup_ad", [RUN_JOB, ARCHIVE_JOB])
    for idx, cime_job_id in enumerate([RUN_JOB, ARCHIVE_JOB], start=1):
        name = f"{created_run.run_id}_spinup_ad_fail_{idx}"
        content = (created_run.root / f"{name}.pbs").read_text()
        assert f"#PBS -N {name}\n" in content
        assert f"#PBS -W depend=afternotok:{cime_job_id}\n" in content


def test_submit_fail_duplicate_ids_submit_once(created_run, fake_scheduler):
    """Test that a repeated ID (no st_archive, so final job is case.run) gives
    one fail job with the plain _fail name
    """
    job_ids = created_run.submit_fail("spinup_ad", [RUN_JOB, RUN_JOB])
    assert job_ids == ["1.fake"]
    assert fake_scheduler.submitted == [
        created_run.root / f"{created_run.run_id}_spinup_ad_fail.pbs"
    ]


def test_submit_fail_no_ids_raises(created_run, fake_scheduler):
    """Test that submit_fail() needs at least one CIME job ID"""
    with pytest.raises(ValueError, match="at least one"):
        created_run.submit_fail("spinup_ad", [])
    assert not fake_scheduler.submitted


def test_submit_fail_second_qsub_failure_keeps_first_job(created_run, mocker):
    """Test that if the second qsub fails, the first fail job is still recorded
    so is_lost() can see it
    """
    mocker.patch.object(
        created_run.scheduler,
        "submit",
        side_effect=["1.fake", subprocess.CalledProcessError(1, "qsub")],
    )
    with pytest.raises(subprocess.CalledProcessError):
        created_run.submit_fail("spinup_ad", [RUN_JOB, ARCHIVE_JOB])
    reloaded = ModelRun.load(created_run.root, scheduler=created_run.scheduler)
    assert reloaded.stages[0].state.fail_job_ids == ["1.fake"]


def test_submit_fail_replaces_previous_fail_job_ids(created_run):
    """Test that submit_fail() records only the jobs it submitted this time"""
    created_run.stages[0].state.fail_job_ids = ["old.fake"]
    job_ids = created_run.submit_fail("spinup_ad", [RUN_JOB, ARCHIVE_JOB])
    assert created_run.stages[0].state.fail_job_ids == job_ids
