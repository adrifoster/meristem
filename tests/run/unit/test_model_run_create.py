"""Tests for ModelRun.create behavior"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest
import yaml

from meristem.run.model_run import ModelRun, _STATE_FILE
from meristem.run.scheduler import PbsScheduler
from meristem.run.stages import Stage


def test_model_run_create_valid_config_passes(run_config_dict):
    """Test ModelRun.create with a valid config passes"""
    ModelRun.create(run_config_dict)


def test_model_run_create_dict_returned_unchanged(run_config_dict):
    """Test ModelRun.create does not modify the input dict"""
    original = copy.deepcopy(run_config_dict)
    ModelRun.create(run_config_dict)
    assert run_config_dict == original


def test_model_run_create_yaml_passes(tmp_path, run_config_dict):
    """Test that ModelRun.create with a yaml file passes"""
    yaml_path = tmp_path / "config.yml"
    yaml_path.write_text(yaml.dump(run_config_dict), encoding="utf-8")
    ModelRun.create(yaml_path)


@pytest.mark.parametrize(
    "text, match",
    [
        ("", r"^run config is empty\.$"),
        ("# only a comment\n", r"^run config is empty\.$"),
        ("- a\n- b\n", r"must be a mapping of keys to values, got list\.$"),
        ("hello\n", r"must be a mapping of keys to values, got str\.$"),
    ],
)
def test_model_run_create_yaml_not_a_mapping_raises(tmp_path, text, match):
    """Test that an empty or non-mapping run.yaml raises a clear ValueError
    (it used to be an AttributeError)
    """
    path = tmp_path / "run.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        ModelRun.create(path)


def test_model_run_create_yaml_path_as_string_passes(tmp_path, run_config_dict):
    """Test that ModelRun.create with a yaml file as a string passes"""
    yaml_path = tmp_path / "config.yml"
    yaml_path.write_text(yaml.dump(run_config_dict), encoding="utf-8")
    ModelRun.create(str(yaml_path))


def test_model_run_create_invalid_type_raises():
    """Test that ModelRun.create with an invalid type raises a TypeError"""
    with pytest.raises(TypeError, match="config must be"):
        ModelRun.create(42)


@pytest.mark.parametrize("missing_key", ["root", "stages"])
def test_model_run_create_missing_required_key_raises(run_config_dict, missing_key):
    """Test that ModelRun.create with missing keys raises a ValueError"""
    del run_config_dict[missing_key]
    with pytest.raises(ValueError, match=missing_key):
        ModelRun.create(run_config_dict)


def test_model_run_create_empty_stages_list_raises(run_config_dict):
    """Test that ModelRun.create with a dict with empty stages list raises a ValueError"""
    run_config_dict["stages"] = []
    with pytest.raises(ValueError, match="stages"):
        ModelRun.create(run_config_dict)


def test_model_run_create_stages_not_a_list_raises(run_config_dict):
    """Test that ModelRun.create with a dict with stages not a list raises a ValueError"""
    run_config_dict["stages"] = "not_a_list"
    with pytest.raises(ValueError, match="stages"):
        ModelRun.create(run_config_dict)


def test_model_run_create_creates_root_directory(run_config_dict):
    """Test that ModelRun.create creates the root directory"""
    run = ModelRun.create(run_config_dict)
    assert run.root.is_dir()


def test_model_run_create_writes_state_file(run_config_dict):
    """Test that ModelRun.create writes the state file"""
    run = ModelRun.create(run_config_dict)
    assert (run.root / _STATE_FILE).exists()


def test_model_run_create_fields_set_correctly(run_config_dict):
    """Test that ModelRun.create sets the fields correctly"""
    run = ModelRun.create(run_config_dict)
    assert run.run_id == "member_0001"
    assert run.project == "PROJ123"
    assert len(run.stages) == 1


def test_model_run_create_default_run_id(run_config_dict):
    """Test that ModelRun.create sets the default run_id if left blank"""
    del run_config_dict["run_id"]
    run = ModelRun.create(run_config_dict)
    assert run.run_id == "run"


def test_model_run_raises_if_state_file_already_exists(run_config_dict):
    """Test that ModelRun.create raises a ValueError if the state file already exists"""
    ModelRun.create(run_config_dict)
    with pytest.raises(FileExistsError, match="already exists"):
        ModelRun.create(run_config_dict)


def test_model_run_creates_nested_root(tmp_path, run_config_dict):
    """Test that ModelRun.create can create a nested root file"""
    run_config_dict["root"] = str(tmp_path / "a" / "b" / "c")
    run = ModelRun.create(run_config_dict)
    assert run.root.is_dir()


def test_model_run_create_test_invalid_config_raises_before_touching_filesystem(
    run_config_dict,
):
    """Test that ModelRun.create with an invalid input dict fails before it creates the root"""
    run_config_dict["stages"] = []
    with pytest.raises(ValueError):
        ModelRun.create(run_config_dict)
    # root should not have been created
    assert not Path(run_config_dict["root"]).exists()


def test_model_run_create_from_yaml(tmp_path, run_config_dict):
    """Test that ModelRun.create can be created from a yaml"""
    yaml_path = tmp_path / "config.yml"
    yaml_path.write_text(yaml.dump(run_config_dict), encoding="utf-8")
    run = ModelRun.create(yaml_path)
    assert run.project == "PROJ123"


def test_model_run_create_default_scheduler_is_pbs(run_config_dict):
    """Test that ModelRun.create uses PBS when the config has no scheduler key"""
    run = ModelRun.create(run_config_dict)
    assert isinstance(run.scheduler, PbsScheduler)


def test_model_run_create_scheduler_from_config(run_config_dict):
    """Test that ModelRun.create builds the scheduler named in the config"""
    run_config_dict["scheduler"] = "pbs"
    run = ModelRun.create(run_config_dict)
    assert isinstance(run.scheduler, PbsScheduler)


def test_model_run_create_unknown_scheduler_raises_before_touching_filesystem(
    run_config_dict,
):
    """Test that an unknown scheduler name fails before the root is created"""
    run_config_dict["scheduler"] = "nonexistent"
    with pytest.raises(ValueError, match="nonexistent"):
        ModelRun.create(run_config_dict)
    assert not Path(run_config_dict["root"]).exists()


def test_model_run_create_scheduler_arg_overrides_config(
    run_config_dict, fake_scheduler
):
    """Test that a scheduler passed to ModelRun.create wins over the config key"""
    run_config_dict["scheduler"] = "pbs"
    run = ModelRun.create(run_config_dict, scheduler=fake_scheduler)
    assert run.scheduler is fake_scheduler


def test_model_run_create_root_with_whitespace_raises(tmp_path, run_config_dict):
    """Test that a root containing whitespace is rejected before it is created"""
    run_config_dict["root"] = str(tmp_path / "my run")
    with pytest.raises(ValueError, match="whitespace"):
        ModelRun.create(run_config_dict)
    assert not Path(run_config_dict["root"]).exists()


def test_model_run_create_missing_script_raises(tmp_path, run_config_dict):
    """Test that a stage script that doesn't exist is rejected before root is
    created
    """
    run_config_dict["stages"][0]["script"] = str(tmp_path / "nope.sh")
    with pytest.raises(ValueError, match=r"cannot open script.*nope\.sh"):
        ModelRun.create(run_config_dict)
    assert not Path(run_config_dict["root"]).exists()


def test_model_run_create_missing_later_script_raises(tmp_path, run_config_dict):
    """Test that every stage's script is checked at create, not just the first"""
    later = {**run_config_dict["stages"][0], "name": "later"}
    later["script"] = str(tmp_path / "later.sh")
    run_config_dict["stages"].append(later)
    with pytest.raises(ValueError, match=r"later\.sh"):
        ModelRun.create(run_config_dict)
    assert not Path(run_config_dict["root"]).exists()


@pytest.mark.parametrize(
    "root, hint",
    [
        ("runs/myrun", ""),
        ("$SCRATCH/myrun", r" \(is an environment variable unset\?\)"),
    ],
)
def test_model_run_create_relative_root_raises(
    run_config_dict, tmp_path, monkeypatch, root, hint
):
    """Test that a relative root in a run config is rejected (like every other
    config path), hinting at an unset environment variable, and writes nothing
    """
    monkeypatch.chdir(tmp_path)
    run_config_dict["root"] = root
    with pytest.raises(
        ValueError, match=rf"root '{re.escape(root)}' must be an absolute path{hint}"
    ):
        ModelRun.create(run_config_dict)
    assert not (tmp_path / root).exists()


def test_model_run_create_relative_case_root_raises(run_config_dict):
    """Test that a relative case_root on any stage is rejected at create"""
    later = {**run_config_dict["stages"][0], "name": "later", "case_root": "cases/x"}
    run_config_dict["stages"].append(later)
    with pytest.raises(ValueError, match="case_root 'cases/x' for stage 'later'"):
        ModelRun.create(run_config_dict)
    assert not Path(run_config_dict["root"]).exists()


def test_model_run_create_duplicate_stage_names_raises(run_config_dict):
    """Test that two stages with the same name are rejected, naming it, and
    nothing is written (they'd share a job script and case root, and only the
    first could be found by name)
    """
    stage = run_config_dict["stages"][0]
    run_config_dict["stages"] = [stage, {**stage, "queue": "develop"}]
    with pytest.raises(
        ValueError, match=r"stage names must be unique, got duplicates \['spinup_ad'\]"
    ):
        ModelRun.create(run_config_dict)
    assert not Path(run_config_dict["root"]).exists()


@pytest.mark.parametrize("with_script", [False, True])
def test_model_run_create_command_raises(run_config_dict, with_script):
    """Test that a run config can't set a stage command (with or without a
    script), naming the stage, and writes nothing
    """
    second = {**run_config_dict["stages"][0], "name": "later", "command": "echo hi"}
    if not with_script:
        del second["script"]
    run_config_dict["stages"].append(second)
    with pytest.raises(ValueError, match=r"stage\(s\) \['later'\] set 'command'"):
        ModelRun.create(run_config_dict)
    assert not Path(run_config_dict["root"]).exists()


def test_model_run_create_null_command_raises(run_config_dict):
    """Test that even an empty command key is rejected (it's not a user key)"""
    run_config_dict["stages"][0]["command"] = None
    with pytest.raises(ValueError, match="set 'command'"):
        ModelRun.create(run_config_dict)


@pytest.mark.parametrize("post_command", ["echo hi", None])
def test_model_run_create_post_command_raises(run_config_dict, post_command):
    """Test that a run config can't set a stage post_command (even an empty
    one), naming the stage, and writes nothing
    """
    second = {
        **run_config_dict["stages"][0],
        "name": "later",
        "post_command": post_command,
    }
    run_config_dict["stages"].append(second)
    with pytest.raises(ValueError, match=r"stage\(s\) \['later'\] set 'post_command'"):
        ModelRun.create(run_config_dict)
    assert not Path(run_config_dict["root"]).exists()


def test_model_run_create_no_script_raises(run_config_dict):
    """Test that a stage with no script is rejected at create, writing nothing"""
    del run_config_dict["stages"][0]["script"]
    with pytest.raises(ValueError, match="stage 'spinup_ad' has no setup script"):
        ModelRun.create(run_config_dict)
    assert not Path(run_config_dict["root"]).exists()


def test_model_run_load_keeps_command(tmp_path, command_stage_config_dict):
    """Test that load accepts a stage command meristem wrote to the state file
    (only run configs reject it)
    """
    run = ModelRun(
        root=tmp_path / "run",
        run_id="run",
        stages=[Stage.from_dict(command_stage_config_dict)],
        project="PROJ",
    )
    run.root.mkdir()
    run.save()
    loaded = ModelRun.load(run.root)
    assert loaded.stages[0].config.command == command_stage_config_dict["command"]


def test_model_run_load_after_script_removed(created_run, script_file):
    """Test that load doesn't recheck scripts, so status etc. still work on a
    run whose script has since moved
    """
    script_file.unlink()
    loaded = ModelRun.load(created_run.root, scheduler=created_run.scheduler)
    assert loaded.stage_names == created_run.stage_names
