"""Tests for scheduler implementations"""

from __future__ import annotations

import subprocess

import pytest

from meristem.run.scheduler import JobState, PbsScheduler, get_scheduler


def _qstat_block(job_id: str, letter: str) -> str:
    """One job's block of ``qstat -x -f`` output, trimmed to the fields that matter"""
    return (
        f"Job Id: {job_id}\n"
        f"    Job_Name = member_0001_spinup_ad\n"
        f"    job_state = {letter}\n"
        f"    queue = cpudev\n"
        "\n"
    )


def _mock_qstat(mocker, stdout: str, returncode: int = 0, stderr: str = ""):
    """Patch subprocess.run to return the given qstat result"""
    mock_run = mocker.patch("meristem.run.scheduler.subprocess.run")
    mock_run.return_value.stdout = stdout
    mock_run.return_value.stderr = stderr
    mock_run.return_value.returncode = returncode
    return mock_run


def _mock_qdel(mocker, stdout: str = "", returncode: int = 0, stderr: str = ""):
    """Patch subprocess.run to return the given qdel result"""
    mock_run = mocker.patch("meristem.run.scheduler.subprocess.run")
    mock_run.return_value.stdout = stdout
    mock_run.return_value.stderr = stderr
    mock_run.return_value.returncode = returncode
    return mock_run


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


def test_pbs_query_calls_qstat_once_with_all_ids(mocker):
    """Test that PbsScheduler.query() makes a single qstat -x -f call for all IDs"""
    mock_run = _mock_qstat(
        mocker, _qstat_block("1.desched1", "Q") + _qstat_block("2.desched1", "R")
    )
    PbsScheduler().query(["1.desched1", "2.desched1"])
    mock_run.assert_called_once_with(
        ["qstat", "-x", "-f", "1.desched1", "2.desched1"],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "letter,expected",
    [
        ("Q", JobState.QUEUED),
        ("H", JobState.QUEUED),
        ("R", JobState.RUNNING),
        ("F", JobState.FINISHED),
        # less common states map to the closest of the four
        ("W", JobState.QUEUED),
        ("S", JobState.RUNNING),
        ("E", JobState.RUNNING),
        # routed from a routing queue (e.g. develop) to its execution queue
        ("T", JobState.QUEUED),
    ],
)
def test_pbs_query_maps_job_state(mocker, letter, expected):
    """Test that PbsScheduler.query() maps each PBS job_state letter to a JobState"""
    _mock_qstat(mocker, _qstat_block("1.desched1", letter))
    assert PbsScheduler().query(["1.desched1"]) == {"1.desched1": expected}


def test_pbs_query_unknown_job_is_unknown(mocker):
    """Test that a job qstat has no record of is UNKNOWN. NCAR's qstat prints the
    'Unknown Job Id' message on stdout and exits 153
    """
    _mock_qstat(mocker, "qstat: Unknown Job Id 9.desched1\n", returncode=153)
    assert PbsScheduler().query(["9.desched1"]) == {"9.desched1": JobState.UNKNOWN}


def test_pbs_query_mixed_known_and_unknown(mocker):
    """Test that known jobs are still reported when others are unknown (exit 153)"""
    _mock_qstat(
        mocker,
        _qstat_block("1.desched1", "F") + "qstat: Unknown Job Id 9.desched1\n",
        returncode=153,
    )
    assert PbsScheduler().query(["1.desched1", "9.desched1"]) == {
        "1.desched1": JobState.FINISHED,
        "9.desched1": JobState.UNKNOWN,
    }


def test_pbs_query_matches_on_job_number(mocker):
    """Test that a job ID without the server suffix matches qstat's full ID,
    and results are keyed by the ID passed in
    """
    _mock_qstat(mocker, _qstat_block("1.desched1.hsn.de.hpc.ucar.edu", "R"))
    assert PbsScheduler().query(["1"]) == {"1": JobState.RUNNING}


def test_pbs_query_qstat_failure_raises(mocker):
    """Test that a qstat failure other than unknown IDs raises instead of
    reporting every job as UNKNOWN
    """
    _mock_qstat(mocker, "", returncode=1, stderr="cannot connect to server")
    with pytest.raises(RuntimeError, match="cannot connect to server"):
        PbsScheduler().query(["1.desched1"])


# B and X are job-array states and M a job moved to another server, none of
# which meristem's jobs get
@pytest.mark.parametrize("letter", ["Z", "B", "X", "M"])
def test_pbs_query_unrecognized_state_raises(mocker, letter):
    """Test that an unrecognized job_state raises instead of guessing"""
    _mock_qstat(mocker, _qstat_block("1.desched1", letter))
    with pytest.raises(RuntimeError, match=f"'{letter}'"):
        PbsScheduler().query(["1.desched1"])


def test_pbs_query_empty_list_does_not_call_qstat(mocker):
    """Test that query([]) returns {} without running qstat"""
    mock_run = mocker.patch("meristem.run.scheduler.subprocess.run")
    assert not PbsScheduler().query([])
    mock_run.assert_not_called()


def test_fake_scheduler_query_returns_set_states(fake_scheduler):
    """Test that FakeScheduler.query() returns the states a test sets, and
    UNKNOWN for anything else
    """
    fake_scheduler.job_states = {"1.fake": JobState.RUNNING}
    assert fake_scheduler.query(["1.fake", "2.fake"]) == {
        "1.fake": JobState.RUNNING,
        "2.fake": JobState.UNKNOWN,
    }
    assert fake_scheduler.queried == [["1.fake", "2.fake"]]


def test_get_scheduler_pbs():
    """Test that get_scheduler("pbs") returns a PbsScheduler"""
    assert isinstance(get_scheduler("pbs"), PbsScheduler)


def test_get_scheduler_unknown_raises():
    """Test that get_scheduler raises for an unknown name and lists known ones"""
    with pytest.raises(ValueError, match="nonexistent.*pbs"):
        get_scheduler("nonexistent")


def test_pbs_cancel_calls_qdel_once_with_all_ids(mocker):
    """Test that PbsScheduler.cancel() makes a single qdel call for all IDs"""
    mock_run = _mock_qdel(mocker)
    PbsScheduler().cancel(["1.desched1", "2.desched1"])
    mock_run.assert_called_once_with(
        ["qdel", "1.desched1", "2.desched1"],
        capture_output=True,
        text=True,
        check=False,
    )


def test_pbs_cancel_empty_list_does_not_call_qdel(mocker):
    """Test that cancel([]) returns None without running qdel"""
    mock_run = mocker.patch("meristem.run.scheduler.subprocess.run")
    assert not PbsScheduler().cancel([])
    mock_run.assert_not_called()


def test_pbs_cancel_ignores_unknown_job(mocker):
    """Test that a job qdel ignores an unkown job. NCAR's qdel prints the
    'Unknown Job Id' message on stdout and exits 153
    """
    _mock_qdel(mocker, "qdel: Unknown Job Id 9.desched1\n", returncode=153)
    assert not PbsScheduler().cancel(["9.desched1"])


def test_pbs_cancel_ignores_finished_job(mocker):
    """Test that a job qdel ignores a finished job. NCAR's qdel prints the
    'Job has finished' message on stdout and exits 35
    """
    _mock_qdel(mocker, "Job has finished 8015997.desched1\n", returncode=35)
    assert not PbsScheduler().cancel(["9.desched1"])


def test_pbs_cancel_qdel_failure_raises(mocker):
    """Test that a qdel failure other than unknown IDs raises"""
    _mock_qdel(mocker, returncode=1, stderr="cannot connect to server")
    with pytest.raises(RuntimeError, match="cannot connect to server"):
        PbsScheduler().cancel(["1.desched1"])


def test_fake_scheduler_cancel_cancelled_holds_ids_and_cancels(fake_scheduler):
    """Test that FakeScheduler.cancel() updates the cancelled listt and sets the
    job_states correctly
    """
    fake_scheduler.job_states = {"1.fake": JobState.RUNNING}
    fake_scheduler.cancel(["1.fake"])
    assert fake_scheduler.canceled == [["1.fake"]]
    assert fake_scheduler.query(["1.fake"]) == {
        "1.fake": JobState.FINISHED,
    }
