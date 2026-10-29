"""Tests for ModelRun.__init__ behavior"""

from __future__ import annotations

from pathlib import Path
import pytest

from meristem.run.model_run import ModelRun
from meristem.run.stages import Stage


def test_model_run_empty_run_id_raises(tmp_path, ad_stage):
    """Test that ModelRun initialized with an empty run_id raises a ValueError"""
    with pytest.raises(ValueError, match="run_id"):
        ModelRun(
            root=tmp_path,
            run_id="",
            stages=[ad_stage],
            email="name@ucar.edu",
            project="PROJ",
        )


@pytest.mark.parametrize("bad_id", ["has space", "has/slash", "has@symbol"])
def test_model_run_invalid_run_id_characters_raises(tmp_path, ad_stage, bad_id):
    """Test that ModelRun initialized with run_id with bad characters raises a ValueError"""
    with pytest.raises(ValueError, match="run_id"):
        ModelRun(
            root=tmp_path,
            run_id=bad_id,
            stages=[ad_stage],
            email="name@ucar.edu",
            project="PROJ",
        )


@pytest.mark.parametrize("good_id", ["run", "member_0042", "my-spinup", "RUN01"])
def test_model_run_valid_run_id_accepted(tmp_path, ad_stage, good_id):
    """Test that ModelRun initialized with run_id with allowed characters works"""
    run = ModelRun(
        root=tmp_path,
        run_id=good_id,
        stages=[ad_stage],
        email="name@ucar.edu",
        project="PROJ",
    )
    assert run.run_id == good_id


def test_model_run_env_project_accempted(tmp_path, ad_stage, monkeypatch):
    """Test that ModelRun initialized without a project code uses the environment variable"""
    monkeypatch.setenv("PROJECT", "PROJ")
    run = ModelRun(
        root=tmp_path,
        run_id="run",
        stages=[ad_stage],
        email="name@ucar.edu",
        project="",
    )
    assert run.project == "PROJ"
    monkeypatch.delenv("PROJECT", raising=False)


def test_model_run_empty_project_raises(tmp_path, ad_stage, monkeypatch):
    """Test that ModelRun initialized without a project code raises a ValueError"""
    monkeypatch.setenv("PROJECT", "")
    with pytest.raises(ValueError, match="project"):
        ModelRun(
            root=tmp_path,
            run_id="run",
            stages=[ad_stage],
            email="name@ucar.edu",
            project="",
        )
    monkeypatch.delenv("PROJECT", raising=False)


def test_model_run_empty_stages_raises(tmp_path):
    """Test that ModelRun initialized with an empty stages list raises a ValueError"""
    with pytest.raises(ValueError, match="stages"):
        ModelRun(
            root=tmp_path,
            run_id="run",
            email="name@ucar.edu",
            stages=[],
            project="PROJ",
        )


def test_model_run_missing_script_does_not_raise(tmp_path, ad_stage_config_dict):
    """Test that ModelRun.__init__ doesn't check scripts exist (that's done at
    create and submit), so a run can still be loaded after a script moves
    """
    ad_stage_config_dict["script"] = str(tmp_path / "nope.sh")
    ModelRun(
        root=tmp_path,
        run_id="run",
        stages=[Stage.from_dict(ad_stage_config_dict)],
        email="name@ucar.edu",
        project="PROJ",
    )


def test_model_run_stages_stored_as_tuple(tmp_path, ad_stage):
    """Test that ModelRun initializes stages as a tuple"""
    run = ModelRun(
        root=tmp_path,
        run_id="run",
        stages=[ad_stage],
        email="name@ucar.edu",
        project="PROJ",
    )
    assert isinstance(run.stages, tuple)


def test_model_run_root_coerced_to_path(tmp_path, ad_stage):
    """Test that ModelRun initializes run as a Path"""
    run = ModelRun(
        root=str(tmp_path),
        run_id="run",
        stages=[ad_stage],
        email="name@ucar.edu",
        project="PROJ",
    )
    assert isinstance(run.root, Path)


@pytest.mark.parametrize("relative", [".", "runs/myrun", "./runs/../runs/myrun"])
def test_model_run_relative_root_resolved(tmp_path, ad_stage, monkeypatch, relative):
    """Test that a relative root (e.g. --root . on the command line) is made
    absolute against the current directory, since it's written into job scripts
    that PBS doesn't run from here
    """
    monkeypatch.chdir(tmp_path)
    run = ModelRun(root=relative, run_id="run", stages=[ad_stage], project="PROJ")
    assert run.root.is_absolute()
    assert run.root == (tmp_path / relative).resolve()


def test_model_run_load_relative_root_writes_absolute_job_script(
    created_run, fake_scheduler, monkeypatch
):
    """Test that a run loaded with a relative root (as `meristem run submit
    --root .` does) writes the absolute root into its job script
    """
    monkeypatch.chdir(created_run.root)
    run = ModelRun.load(".", scheduler=fake_scheduler)
    assert run.root == created_run.root.resolve()
    run.submit()
    script = (run.root / f"{run.run_id}_spinup_ad.pbs").read_text()
    root_line = next(
        line for line in script.splitlines() if line.startswith("RUN_ROOT=")
    )
    assert root_line == f"RUN_ROOT={run.root}"
    assert Path(root_line.split("=", 1)[1]).is_absolute()
