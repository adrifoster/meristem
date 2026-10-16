"""Test that generated scripts pass PBS validation"""

import subprocess
import pytest

from meristem.run.scheduler import JobState


@pytest.mark.pbs
def test_qsub_accepts_submit_script(integration_run):
    """Test that generated PBS submit script is accepted by PBS"""
    job_id = integration_run.submit()
    assert job_id, "expected a job ID from qsub"
    result = subprocess.run(
        ["qdel", job_id], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, f"qdel failed:\n{result.stderr}"


@pytest.mark.pbs
def test_qsub_accepts_advance_script(integration_run):
    """Submit setup job, use real job ID to test advance script submission"""
    # submit setup job — get a real PBS job ID
    setup_job_id = integration_run.submit()
    assert setup_job_id

    try:
        # now submit advance script with the real job ID
        advance_job_id = integration_run.submit_advance("spinup_ad", setup_job_id)
        assert advance_job_id

        # cancel both immediately
        subprocess.run(["qdel", advance_job_id], check=False)
    finally:
        # always cancel the setup job
        subprocess.run(["qdel", setup_job_id], check=False)


@pytest.mark.pbs
def test_qstat_query_reports_submitted_and_unknown_jobs(integration_run):
    """Test that PbsScheduler.query() parses real qstat output: a just-submitted
    job is active, and an ID PBS has never seen is UNKNOWN
    """
    job_id = integration_run.submit()
    assert job_id
    try:
        states = integration_run.scheduler.query([job_id, "1.desched1"])
        assert states[job_id] in (JobState.QUEUED, JobState.RUNNING)
        assert states["1.desched1"] is JobState.UNKNOWN
        assert not integration_run.is_lost("spinup_ad")
    finally:
        subprocess.run(["qdel", job_id], check=False)
