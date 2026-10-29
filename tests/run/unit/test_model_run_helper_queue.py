"""Tests for ModelRun.helper_queue: the queue for the advance and fail jobs"""

from __future__ import annotations

import json

import pytest

from meristem.run.model_run import _STATE_FILE, ModelRun


def _queue_line(path) -> str:
    """The #PBS -q line of a job script"""
    return next(l for l in path.read_text().splitlines() if l.startswith("#PBS -q"))


def _job_scripts(run: ModelRun) -> dict[str, str]:
    """Submit the run's first stage, then write its advance and fail jobs, and
    return each job script's #PBS -q line
    """
    run.submit()
    run.submit_advance("spinup_ad", "100.pbs")
    run.submit_fail("spinup_ad", ["100.pbs"])
    prefix = f"{run.run_id}_spinup_ad"
    return {
        job: _queue_line(run.root / f"{prefix}{suffix}.pbs")
        for job, suffix in (("setup", ""), ("advance", "_advance"), ("fail", "_fail"))
    }


def test_helper_queue_used_for_advance_and_fail_jobs(run_config_dict, fake_scheduler):
    """Test that the advance and fail jobs go to helper_queue, and the stage's
    own setup job still goes to the stage's queue
    """
    run_config_dict["helper_queue"] = "develop"
    run = ModelRun.create(run_config_dict, scheduler=fake_scheduler)
    assert _job_scripts(run) == {
        "setup": "#PBS -q regular",
        "advance": "#PBS -q develop",
        "fail": "#PBS -q develop",
    }


@pytest.mark.parametrize("helper_queue", [None, ""])
def test_no_helper_queue_uses_stage_queue(
    run_config_dict, fake_scheduler, helper_queue
):
    """Test that without a helper_queue (unset, or blank in the YAML) every job
    goes to the stage's queue
    """
    if helper_queue is not None:
        run_config_dict["helper_queue"] = helper_queue
    run = ModelRun.create(run_config_dict, scheduler=fake_scheduler)
    assert run.helper_queue is None
    assert set(_job_scripts(run).values()) == {"#PBS -q regular"}


def test_helper_queue_saved_and_loaded(run_config_dict, fake_scheduler):
    """Test that helper_queue is kept in the state file, so jobs written by a
    later command (e.g. submit-advance inside the setup job) still use it
    """
    run_config_dict["helper_queue"] = "develop"
    run = ModelRun.create(run_config_dict, scheduler=fake_scheduler)
    state = json.loads((run.root / _STATE_FILE).read_text(encoding="utf-8"))
    assert state["helper_queue"] == "develop"
    assert ModelRun.load(run.root, scheduler=fake_scheduler).helper_queue == "develop"


def test_state_file_without_helper_queue_loads(created_run, fake_scheduler):
    """Test that a state file written before helper_queue existed loads with
    none, so the stage queue is used
    """
    path = created_run.root / _STATE_FILE
    state = json.loads(path.read_text(encoding="utf-8"))
    del state["helper_queue"]
    path.write_text(json.dumps(state), encoding="utf-8")
    assert (
        ModelRun.load(created_run.root, scheduler=fake_scheduler).helper_queue is None
    )


@pytest.mark.parametrize("helper_queue", ["develop\ntouch PWNED", "dev elop", "a;b"])
def test_unsafe_helper_queue_raises(run_config_dict, fake_scheduler, helper_queue):
    """Test that a helper_queue that could break out of the #PBS -q line is
    rejected at create
    """
    run_config_dict["helper_queue"] = helper_queue
    with pytest.raises(ValueError, match="helper_queue .* contains invalid characters"):
        ModelRun.create(run_config_dict, scheduler=fake_scheduler)
