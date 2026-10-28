"""Tests for the Ensemble class"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import yaml

from meristem.commands import meristem_cmd
from meristem.ensemble.config import EnsembleConfig
from meristem.ensemble.model_ensemble import (
    Ensemble,
    _archive_dir,
    _is_existing_case,
    _preview,
)
from meristem.run import model_run
from meristem.run.model_run import _STATE_FILE as _RUN_STATE_FILE, ModelRun
from meristem.run.stages import StageStatus

from tests.ensemble.conftest import MEMBERS

_STATE_FILE = "ensemble_state.json"


def _ensemble(config_dict: dict) -> Ensemble:
    """An Ensemble built in memory (nothing checked or written to disk)"""
    return Ensemble(EnsembleConfig.from_dict(config_dict))


def _with_members(config_dict: dict, members: list[str]) -> Ensemble:
    """An in-memory Ensemble that requests only the given members"""
    config_dict["paramgen"]["members"] = members
    return _ensemble(config_dict)


@pytest.fixture
def yaml_path(tmp_path, ready_config_dict) -> Path:
    """ready_config_dict written to a YAML file"""
    path = tmp_path / "ensemble.yaml"
    path.write_text(yaml.safe_dump(ready_config_dict), encoding="utf-8")
    return path


@pytest.fixture
def created_ensemble(ready_config_dict) -> Ensemble:
    """An Ensemble created (on disk) from ready_config_dict"""
    return Ensemble.create(ready_config_dict)


# ---------------------------------------------------------------------------
# properties
# ---------------------------------------------------------------------------


def test_param_files_default_empty(ensemble_config_dict):
    """Test that param_files defaults to an empty dict"""
    assert _ensemble(ensemble_config_dict).param_files == {}


def test_param_files_from_init(ensemble_config_dict):
    """Test that param_files passed to __init__ are stored"""
    param_files = {"ens_0000": Path("/p/ens_0000.nc")}
    ensemble = Ensemble(EnsembleConfig.from_dict(ensemble_config_dict), param_files)
    assert ensemble.param_files == param_files


def test_root_property(ensemble_config_dict):
    """Test that root is the config's root"""
    ensemble = _ensemble(ensemble_config_dict)
    assert ensemble.root == Path(ensemble_config_dict["root"])
    assert ensemble.root == ensemble.config.root


def test_base_case_property(ensemble_config_dict):
    """Test that base_case is the config's base_case"""
    ensemble = _ensemble(ensemble_config_dict)
    assert ensemble.base_case == Path(ensemble_config_dict["base_case"])
    assert ensemble.base_case == ensemble.config.base_case


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


def test_create_from_dict(ready_config_dict):
    """Test that create accepts a config dict"""
    ensemble = Ensemble.create(ready_config_dict)
    assert ensemble.config == EnsembleConfig.from_dict(ready_config_dict)


@pytest.mark.parametrize("as_str", [False, True])
def test_create_from_yaml(yaml_path, ready_config_dict, as_str):
    """Test that create accepts a YAML path as a Path or str"""
    ensemble = Ensemble.create(str(yaml_path) if as_str else yaml_path)
    assert ensemble.config == EnsembleConfig.from_dict(ready_config_dict)


def test_create_from_ensemble_config(ready_config_dict):
    """Test that create accepts an EnsembleConfig and uses it as is"""
    config = EnsembleConfig.from_dict(ready_config_dict)
    assert Ensemble.create(config).config is config


@pytest.mark.parametrize("config", [None, 42, ["root"]])
def test_create_bad_type_raises(config):
    """Test that create raises TypeError for anything but a config, dict, or path"""
    with pytest.raises(TypeError, match="config must be an EnsembleConfig"):
        Ensemble.create(config)  # type: ignore[arg-type]


def test_create_invalid_config_raises(ready_config_dict):
    """Test that create raises the config's ValueError and writes nothing"""
    del ready_config_dict["base_case"]
    with pytest.raises(ValueError, match="missing required key base_case"):
        Ensemble.create(ready_config_dict)
    assert not Path(ready_config_dict["root"]).exists()


def test_create_makes_root_and_state_file(ready_config_dict):
    """Test that create makes root (including parents) and writes the state file"""
    root = Path(ready_config_dict["root"]) / "nested" / "dir"
    ready_config_dict["root"] = str(root)
    Ensemble.create(ready_config_dict)
    assert root.is_dir()
    assert (root / _STATE_FILE).is_file()


def test_create_existing_state_file_raises(created_ensemble, ready_config_dict):
    """Test that create refuses to overwrite an existing ensemble"""
    state_path = created_ensemble.root / _STATE_FILE
    before = state_path.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="Ensemble.load"):
        Ensemble.create(ready_config_dict)
    assert state_path.read_text(encoding="utf-8") == before


def test_create_existing_root_without_state_file(ready_config_dict):
    """Test that create works in a root directory that exists but has no state"""
    Path(ready_config_dict["root"]).mkdir(parents=True)
    ensemble = Ensemble.create(ready_config_dict)
    assert (ensemble.root / _STATE_FILE).is_file()


def test_create_missing_paramgen_output_raises(ensemble_config_dict, write_base_case):
    """Test that create checks the paramgen output and writes nothing if it's
    missing
    """
    write_base_case()
    with pytest.raises(FileNotFoundError, match="member list"):
        Ensemble.create(ensemble_config_dict)
    assert not Path(ensemble_config_dict["root"]).exists()


def test_create_missing_param_file_raises(
    ensemble_config_dict, write_paramgen, write_base_case
):
    """Test that create checks every member has a parameter file"""
    write_paramgen(param_files=MEMBERS[:-1])
    write_base_case()
    with pytest.raises(FileNotFoundError, match=MEMBERS[-1]):
        Ensemble.create(ensemble_config_dict)
    assert not Path(ensemble_config_dict["root"]).exists()


def test_create_unbuilt_base_case_raises(
    ensemble_config_dict, write_paramgen, write_base_case
):
    """Test that create checks the base case is built and writes nothing if not"""
    write_paramgen()
    write_base_case(build_complete="FALSE")
    with pytest.raises(ValueError, match="not built"):
        Ensemble.create(ensemble_config_dict)
    assert not Path(ensemble_config_dict["root"]).exists()


def test_create_sets_param_files(ready_config_dict):
    """Test that create finds and stores every member's parameter file"""
    directory = Path(ready_config_dict["paramgen"]["dir"])
    ensemble = Ensemble.create(ready_config_dict)
    assert ensemble.param_files == {m: directory / f"{m}.nc" for m in MEMBERS}


def test_create_with_existing_stage_script(ready_config_dict, tmp_path):
    """Test that create passes when a member stage's script exists"""
    script = tmp_path / "setup.sh"
    script.touch()
    ready_config_dict["member_stages"] = [
        {"name": "setup", "script": str(script), "queue": "regular"}
    ]
    Ensemble.create(ready_config_dict)


def test_create_missing_stage_script_raises(ready_config_dict, tmp_path):
    """Test that create checks member stage scripts and writes nothing if one
    is missing
    """
    ready_config_dict["member_stages"] = [
        {"name": "setup", "script": str(tmp_path / "nope.sh"), "queue": "regular"}
    ]
    with pytest.raises(FileNotFoundError, match=r"setup: .*nope\.sh"):
        Ensemble.create(ready_config_dict)
    assert not Path(ready_config_dict["root"]).exists()


# ---------------------------------------------------------------------------
# create: member runs
# ---------------------------------------------------------------------------


def _member_run(ensemble: Ensemble, member: str) -> ModelRun:
    """Load a member's ModelRun from disk"""
    return ModelRun.load(ensemble.root / "members" / member)


def test_create_writes_member_runs(created_ensemble):
    """Test that create writes a loadable ModelRun for every member, and no others"""
    members_dir = created_ensemble.root / "members"
    assert sorted(p.name for p in members_dir.iterdir()) == MEMBERS
    for member in MEMBERS:
        assert (members_dir / member / _RUN_STATE_FILE).is_file()
        run = _member_run(created_ensemble, member)
        assert run.run_id == f"my_ens-1_{member}"


def test_prepare_unknown_stage_raises(created_ensemble):
    """Test that prepare raises for a stage that isn't a member stage"""
    with pytest.raises(ValueError, match="unknown member stage 'nope'"):
        created_ensemble.prepare(MEMBERS[0], "nope")


def test_prepare_unknown_member_raises(created_ensemble):
    """Test that prepare raises a ValueError for a member not in the ensemble"""
    with pytest.raises(ValueError, match=r"unknown member\(s\) \['nope'\]"):
        created_ensemble.prepare("nope", "spinup_ad")


def test_create_member_runs_only_requested_members(ready_config_dict):
    """Test that only requested members get a run"""
    ready_config_dict["paramgen"]["members"] = [MEMBERS[1]]
    ensemble = Ensemble.create(ready_config_dict)
    assert [p.name for p in (ensemble.root / "members").iterdir()] == [MEMBERS[1]]


def test_create_member_run_project_and_email(ready_config_dict):
    """Test that the config's project and email are passed to member runs"""
    ready_config_dict["email"] = "me@x.org"
    run = _member_run(Ensemble.create(ready_config_dict), MEMBERS[0])
    assert run.project == "PROJ123"
    assert run.email == "me@x.org"


def test_create_member_run_project_from_env(ready_config_dict, monkeypatch):
    """Test that member runs fall back to $PROJECT when the config has none"""
    monkeypatch.setenv("PROJECT", "ENVPROJ")
    ready_config_dict["project"] = ""
    run = _member_run(Ensemble.create(ready_config_dict), MEMBERS[0])
    assert run.project == "ENVPROJ"


def test_create_no_project_raises(ready_config_dict, monkeypatch):
    """Test that create raises, writing nothing, with no project or $PROJECT"""
    monkeypatch.delenv("PROJECT", raising=False)
    ready_config_dict["project"] = ""
    with pytest.raises(ValueError, match="project"):
        Ensemble.create(ready_config_dict)
    assert not Path(ready_config_dict["root"]).exists()


def test_case_root_differs_between_ensembles(ready_config_dict, tmp_path):
    """Test that two ensembles sharing a base case, parameter members and
    case_dir give each member stage a different case
    """
    first = Ensemble.create(ready_config_dict)
    ready_config_dict["ensemble_id"] = "my_ens-2"
    ready_config_dict["root"] = str(tmp_path / "ens2")
    second = Ensemble.create(ready_config_dict)
    first_root = _member_run(first, MEMBERS[0]).stages[0].config.case_root
    second_root = _member_run(second, MEMBERS[0]).stages[0].config.case_root
    assert first_root.parent == second_root.parent
    assert first_root != second_root


def test_create_member_stages_run_prepare(ready_config_dict, tmp_path):
    """Test that each member stage, in order, runs meristem ensemble prepare in
    its own case under case_dir, keeping the stage's other settings
    """
    ready_config_dict["member_stages"] = [
        {"name": "ad", "queue": "regular"},
        {"name": "post", "queue": "develop", "ncpus": 4},
    ]
    ensemble = Ensemble.create(ready_config_dict)
    member = MEMBERS[1]
    run = _member_run(ensemble, member)
    assert run.stage_names == ["ad", "post"]
    for stage in run.stages:
        name = stage.config.name
        assert stage.config.script is None
        assert shlex.split(stage.config.command) == [
            *shlex.split(meristem_cmd()),
            "ensemble",
            "prepare",
            "--root",
            str(ensemble.root),
            "--member",
            member,
            "--stage",
            name,
        ]
        case = f"base_case_{ensemble.config.ensemble_id}_{member}_{name}"
        assert stage.config.case_root == tmp_path / "cases" / case
        assert stage.state.status == StageStatus.PENDING
    assert run.stages[1].config.queue == "develop"
    assert run.stages[1].config.ncpus == 4


def test_create_member_stage_keeps_script(ready_config_dict, tmp_path):
    """Test that a member stage with a script keeps it and gets the prepare
    command too (the job runs prepare, then the script)
    """
    script = tmp_path / "setup.sh"
    script.touch()
    ready_config_dict["member_stages"] = [
        {"name": "setup", "script": str(script), "queue": "regular"}
    ]
    stage = _member_run(Ensemble.create(ready_config_dict), MEMBERS[0]).stages[0]
    assert stage.config.script == script
    assert "prepare" in shlex.split(stage.config.command)


def test_create_member_stage_without_script_runs_only_prepare(created_ensemble):
    """Test that a member stage with no script runs only prepare"""
    stage = _member_run(created_ensemble, MEMBERS[0]).stages[0]
    assert stage.config.script is None
    assert "prepare" in shlex.split(stage.config.command)


def test_create_member_stage_command_quotes_root(ready_config_dict, tmp_path):
    """Test that a root with shell special characters stays one argument in the
    command
    """
    root = tmp_path / "it's_root"
    ready_config_dict["root"] = str(root)
    stage = _member_run(Ensemble.create(ready_config_dict), MEMBERS[0]).stages[0]
    args = shlex.split(stage.config.command)
    assert args[args.index("--root") + 1] == str(root)


def test_create_invalid_member_run_id_raises(
    ensemble_config_dict, write_paramgen, write_base_case
):
    """Test that a member name ModelRun can't use as a run_id raises before
    anything is written (not after some members' runs are)
    """
    write_paramgen(members=["ens_0000", "ens.0001"])
    write_base_case()
    ensemble_config_dict["project"] = "PROJ123"
    with pytest.raises(ValueError, match=r"run_id 'my_ens-1_ens\.0001'"):
        Ensemble.create(ensemble_config_dict)
    assert not Path(ensemble_config_dict["root"]).exists()


def test_create_relative_case_dir_raises(ready_config_dict):
    """Test that a relative case_dir (which would give relative member
    case_roots) raises before anything is written
    """
    ready_config_dict["case_dir"] = "cases"
    with pytest.raises(ValueError, match="case_dir 'cases' must be an absolute path"):
        Ensemble.create(ready_config_dict)
    assert not Path(ready_config_dict["root"]).exists()


# ---------------------------------------------------------------------------
# load / save
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("as_str", [False, True])
def test_load_round_trip(created_ensemble, as_str):
    """Test that load (given a Path or str) restores the created config"""
    root = created_ensemble.root
    loaded = Ensemble.load(str(root) if as_str else root)
    assert loaded.config == created_ensemble.config


def test_load_round_trip_with_members_and_optional_keys(ready_config_dict, tmp_path):
    """Test that a members list and optional keys survive create -> load"""
    ready_config_dict["paramgen"]["members"] = [MEMBERS[0], MEMBERS[2]]
    ready_config_dict.update(
        output_root=str(tmp_path / "out"), project="P12345678", email="me@x.org"
    )
    created = Ensemble.create(ready_config_dict)
    assert Ensemble.load(created.root).config == created.config


def test_load_does_not_recheck_inputs(created_ensemble, ready_config_dict):
    """Test that load works after the paramgen output has moved away (it only
    reads the state file)
    """
    for path in Path(ready_config_dict["paramgen"]["dir"]).iterdir():
        path.unlink()
    assert Ensemble.load(created_ensemble.root).config == created_ensemble.config


def test_load_restores_param_files(created_ensemble):
    """Test that load restores param_files as Paths, in member order"""
    loaded = Ensemble.load(created_ensemble.root)
    assert loaded.param_files == created_ensemble.param_files
    assert list(loaded.param_files) == MEMBERS
    assert all(isinstance(path, Path) for path in loaded.param_files.values())


def test_load_restores_member_subset(ready_config_dict):
    """Test that load restores only the requested members' parameter files"""
    ready_config_dict["paramgen"]["members"] = [MEMBERS[2], MEMBERS[0]]
    created = Ensemble.create(ready_config_dict)
    assert list(Ensemble.load(created.root).param_files) == [MEMBERS[0], MEMBERS[2]]


def test_load_missing_state_file_raises(tmp_path):
    """Test that load raises FileNotFoundError when there's no state file"""
    with pytest.raises(FileNotFoundError, match="Ensemble.create"):
        Ensemble.load(tmp_path)


def test_load_moved_ensemble_raises(created_ensemble, tmp_path):
    """Test that loading an ensemble from somewhere other than where it was
    created raises, naming both locations
    """
    moved = created_ensemble.root.rename(tmp_path / "moved")
    with pytest.raises(ValueError) as excinfo:
        Ensemble.load(moved)
    assert str(moved) in str(excinfo.value)
    assert f"created at {created_ensemble.root}" in str(excinfo.value)


def test_load_through_symlink(created_ensemble, tmp_path):
    """Test that loading through a symlink to the ensemble's root works"""
    link = tmp_path / "link"
    link.symlink_to(created_ensemble.root)
    assert Ensemble.load(link).root == created_ensemble.root


def test_load_relative_root(created_ensemble, monkeypatch):
    """Test that loading with a relative path to the ensemble's root works"""
    monkeypatch.chdir(created_ensemble.root.parent)
    assert Ensemble.load(created_ensemble.root.name).root == created_ensemble.root


def test_save_writes_config_and_members(created_ensemble):
    """Test that the state file holds the config's to_dict under "config" and
    the member -> parameter file paths (as str) under "members"
    """
    state = json.loads((created_ensemble.root / _STATE_FILE).read_text("utf-8"))
    assert state == {
        "config": created_ensemble.config.to_dict(),
        "members": {m: str(p) for m, p in created_ensemble.param_files.items()},
    }


def test_save_without_param_files_writes_empty_members(ensemble_config_dict):
    """Test that an in-memory Ensemble with no param_files saves empty members"""
    ensemble = _ensemble(ensemble_config_dict)
    ensemble.root.mkdir(parents=True)
    ensemble.save()
    state = json.loads((ensemble.root / _STATE_FILE).read_text("utf-8"))
    assert state["members"] == {}


def test_save_writes_current_param_files(created_ensemble):
    """Test that save writes param_files as they are now, not as at create"""
    created_ensemble.param_files = {"ens_0001": Path("/new/ens_0001.nc")}
    created_ensemble.save()
    loaded = Ensemble.load(created_ensemble.root)
    assert loaded.param_files == {"ens_0001": Path("/new/ens_0001.nc")}


def test_save_leaves_no_tmp_file(created_ensemble):
    """Test that the atomic write doesn't leave its temp file behind"""
    created_ensemble.save()
    assert not (created_ensemble.root / f"{_STATE_FILE}.tmp").exists()


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


@pytest.fixture
def submit_ensemble(created_ensemble, fake_scheduler, monkeypatch) -> Ensemble:
    """created_ensemble whose member runs load with fake_scheduler, so submit
    records job files instead of calling qsub
    """
    monkeypatch.setattr(model_run, "get_scheduler", lambda name: fake_scheduler)
    return created_ensemble


def _first_status(ensemble: Ensemble, member: str) -> StageStatus:
    """Status of a member's first stage, as saved on disk"""
    return _member_run(ensemble, member).stages[0].state.status


def _set_first_status(ensemble: Ensemble, member: str, status: StageStatus):
    """Save a member's first stage with the given status, skipping the
    transition checks
    """
    run = _member_run(ensemble, member)
    object.__setattr__(run.stages[0].state, "_status", status)
    run.save()


def test_submit_all_pending_members(submit_ensemble, fake_scheduler):
    """Test that submit submits every member's first stage and returns its job ID"""
    result = submit_ensemble.submit()
    assert list(result) == MEMBERS
    assert len(fake_scheduler.submitted) == len(MEMBERS)
    for member, job_id in result.items():
        state = _member_run(submit_ensemble, member).stages[0].state
        assert state.status is StageStatus.SUBMITTED
        assert state.job_id == job_id


@pytest.mark.parametrize(
    "status", [StageStatus.SUBMITTED, StageStatus.DONE, StageStatus.FAILED]
)
def test_submit_skips_members_not_pending(submit_ensemble, fake_scheduler, status):
    """Test that a member whose first stage isn't PENDING is left alone"""
    _set_first_status(submit_ensemble, MEMBERS[0], status)
    result = submit_ensemble.submit()
    assert list(result) == MEMBERS[1:]
    assert len(fake_scheduler.submitted) == len(MEMBERS) - 1
    assert _first_status(submit_ensemble, MEMBERS[0]) is status


def test_submit_limit(submit_ensemble, fake_scheduler):
    """Test that limit submits the first pending members, in member order"""
    result = submit_ensemble.submit(limit=2)
    assert list(result) == MEMBERS[:2]
    assert len(fake_scheduler.submitted) == 2
    assert _first_status(submit_ensemble, MEMBERS[2]) is StageStatus.PENDING


def test_submit_limit_then_rest(submit_ensemble, fake_scheduler):
    """Test that a second submit picks up the members a limit left out"""
    submit_ensemble.submit(limit=2)
    result = submit_ensemble.submit()
    assert list(result) == MEMBERS[2:]
    assert len(fake_scheduler.submitted) == len(MEMBERS)


def test_submit_limit_counts_only_pending(submit_ensemble):
    """Test that limit skips over members that aren't pending"""
    _set_first_status(submit_ensemble, MEMBERS[0], StageStatus.SUBMITTED)
    result = submit_ensemble.submit(limit=1)
    assert list(result) == [MEMBERS[1]]


@pytest.mark.parametrize("limit", [0, -1])
def test_submit_limit_below_one_raises(submit_ensemble, fake_scheduler, limit):
    """Test that a limit below 1 raises and submits nothing"""
    with pytest.raises(ValueError, match=f"limit must be greater than 0, got {limit}"):
        submit_ensemble.submit(limit=limit)
    assert not fake_scheduler.submitted
    for member in MEMBERS:
        assert _first_status(submit_ensemble, member) is StageStatus.PENDING


def test_submit_no_pending_members(submit_ensemble, fake_scheduler, capsys):
    """Test that submit with nothing pending returns {} and submits nothing"""
    submit_ensemble.submit()
    capsys.readouterr()
    assert submit_ensemble.submit() == {}
    assert len(fake_scheduler.submitted) == len(MEMBERS)
    assert "my_ens-1: no pending members" in capsys.readouterr().out


def test_submit_missing_param_file_submits_nothing(submit_ensemble, fake_scheduler):
    """Test that a missing parameter file raises before any member is submitted"""
    submit_ensemble.param_files[MEMBERS[-1]].unlink()
    with pytest.raises(FileNotFoundError):
        submit_ensemble.submit()
    assert not fake_scheduler.submitted
    for member in MEMBERS:
        assert _first_status(submit_ensemble, member) is StageStatus.PENDING


def test_submit_checks_only_selected_param_files(submit_ensemble):
    """Test that a missing parameter file for a member limit leaves out
    doesn't stop the others being submitted
    """
    submit_ensemble.param_files[MEMBERS[-1]].unlink()
    result = submit_ensemble.submit(limit=2)
    assert list(result) == MEMBERS[:2]


def test_submit_dry_run(submit_ensemble, fake_scheduler):
    """Test that a dry run writes every member's job script but submits nothing
    and leaves every member pending
    """
    result = submit_ensemble.submit(dry_run=True)
    assert list(result) == MEMBERS
    assert not fake_scheduler.submitted
    stage = submit_ensemble.config.member_stages[0].name
    for member, job_id in result.items():
        run = _member_run(submit_ensemble, member)
        assert job_id.startswith("DRY_")
        assert (run.root / f"{run.run_id}_{stage}.pbs").is_file()
        assert run.stages[0].state.status is StageStatus.PENDING


def test_submit_after_dry_run(submit_ensemble, fake_scheduler):
    """Test that a real submit after a dry run submits every member"""
    submit_ensemble.submit(dry_run=True)
    result = submit_ensemble.submit()
    assert list(result) == MEMBERS
    assert len(fake_scheduler.submitted) == len(MEMBERS)


# ---------------------------------------------------------------------------
# retry
# ---------------------------------------------------------------------------


def _set_status(ensemble: Ensemble, member: str, status: StageStatus, index: int = 0):
    """Save a member's stage (by index) with the given status, skipping the
    transition checks
    """
    run = _member_run(ensemble, member)
    object.__setattr__(run.stages[index].state, "_status", status)
    run.save()


@pytest.fixture
def retry_calls(monkeypatch) -> list[tuple]:
    """Record ModelRun.retry calls as (member, stage, dry_run, skip_script, lost)
    instead of resubmitting; each call returns "job-<member>"
    """
    calls = []

    def _retry(self, stage_name, dry_run=False, skip_script=False, lost=False):
        calls.append((self.root.name, stage_name, dry_run, skip_script, lost))
        return f"job-{self.root.name}"

    monkeypatch.setattr(ModelRun, "retry", _retry)
    return calls


@pytest.fixture
def lost(monkeypatch):
    """Factory that makes ModelRun.lost_stages return {member: [stage, ...]};
    with None, lost_stages raises so a test can check it's never queried
    """

    def _set(lost_by_member: dict[str, list[str]] | None):
        def _lost_stages(self):
            if lost_by_member is None:
                raise AssertionError("lost_stages should not be called")
            return lost_by_member.get(self.root.name, [])

        monkeypatch.setattr(ModelRun, "lost_stages", _lost_stages)

    return _set


@pytest.fixture
def two_stage_ensemble(ready_config_dict) -> Ensemble:
    """An Ensemble created with two member stages, spinup_ad then transient"""
    ready_config_dict["member_stages"].append({"name": "transient", "queue": "regular"})
    return Ensemble.create(ready_config_dict)


def test_retry_nothing_selected_raises(created_ensemble, retry_calls):
    """Test that retry with no flags, members, or stage raises"""
    with pytest.raises(ValueError, match="nothing selected to retry"):
        created_ensemble.retry()
    assert not retry_calls


def test_retry_unknown_stage_raises(created_ensemble, retry_calls):
    """Test that an unknown stage name raises and lists the member stages"""
    with pytest.raises(ValueError, match="unknown member stage 'nope'"):
        created_ensemble.retry(failed=True, stage="nope")
    assert not retry_calls


def test_retry_missing_param_file_raises(created_ensemble, retry_calls):
    """Test that a missing parameter file raises before anything is retried"""
    _set_status(created_ensemble, MEMBERS[0], StageStatus.FAILED)
    created_ensemble.param_files[MEMBERS[-1]].unlink()
    with pytest.raises(FileNotFoundError):
        created_ensemble.retry(failed=True)
    assert not retry_calls


def test_retry_checks_only_selected_param_files(created_ensemble, retry_calls):
    """Test that a missing parameter file for an unselected member doesn't stop
    the selected ones being retried
    """
    _set_status(created_ensemble, MEMBERS[0], StageStatus.FAILED)
    created_ensemble.param_files[MEMBERS[-1]].unlink()
    created_ensemble.retry(failed=True, members=[MEMBERS[0]])
    assert retry_calls == [(MEMBERS[0], "spinup_ad", False, False, False)]


def test_retry_nothing_to_retry(created_ensemble, retry_calls, capsys):
    """Test that retry with nothing failed returns {} and retries nothing"""
    assert created_ensemble.retry(failed=True) == {}
    assert not retry_calls
    assert "my_ens-1: nothing to retry" in capsys.readouterr().out


def test_retry_failed_selects_failed_members(created_ensemble, retry_calls):
    """Test that failed=True retries only the failed members, not as lost"""
    _set_status(created_ensemble, MEMBERS[1], StageStatus.FAILED)
    result = created_ensemble.retry(failed=True)
    assert list(result) == [MEMBERS[1]]
    assert retry_calls == [(MEMBERS[1], "spinup_ad", False, False, False)]


def test_retry_failed_with_members_subset(created_ensemble, retry_calls):
    """Test that a failed member outside members= is left alone"""
    for member in MEMBERS[:2]:
        _set_status(created_ensemble, member, StageStatus.FAILED)
    created_ensemble.retry(failed=True, members=[MEMBERS[1]])
    assert retry_calls == [(MEMBERS[1], "spinup_ad", False, False, False)]


def test_retry_failed_with_stage_filter(two_stage_ensemble, retry_calls):
    """Test that stage= only retries failures of that stage"""
    _set_status(two_stage_ensemble, MEMBERS[0], StageStatus.FAILED, index=0)
    _set_status(two_stage_ensemble, MEMBERS[1], StageStatus.DONE, index=0)
    _set_status(two_stage_ensemble, MEMBERS[1], StageStatus.FAILED, index=1)
    result = two_stage_ensemble.retry(failed=True, stage="transient")
    assert list(result) == [MEMBERS[1]]
    assert retry_calls == [(MEMBERS[1], "transient", False, False, False)]


def test_retry_lost_selects_lost_stage(created_ensemble, retry_calls, lost):
    """Test that lost=True retries the lost stages, marked as lost"""
    _set_status(created_ensemble, MEMBERS[2], StageStatus.SUBMITTED)
    lost({MEMBERS[2]: ["spinup_ad"]})
    result = created_ensemble.retry(lost=True)
    assert list(result) == [MEMBERS[2]]
    assert retry_calls == [(MEMBERS[2], "spinup_ad", False, False, True)]


def test_retry_lost_not_queried_without_flag(created_ensemble, retry_calls, lost):
    """Test that the scheduler isn't queried for lost stages unless lost=True"""
    _set_status(created_ensemble, MEMBERS[0], StageStatus.FAILED)
    lost(None)
    created_ensemble.retry(failed=True)
    assert retry_calls == [(MEMBERS[0], "spinup_ad", False, False, False)]


def test_retry_failed_and_lost(created_ensemble, retry_calls, lost):
    """Test that failed=True and lost=True together retry both kinds"""
    _set_status(created_ensemble, MEMBERS[0], StageStatus.FAILED)
    _set_status(created_ensemble, MEMBERS[2], StageStatus.SUBMITTED)
    lost({MEMBERS[2]: ["spinup_ad"]})
    created_ensemble.retry(failed=True, lost=True)
    assert retry_calls == [
        (MEMBERS[0], "spinup_ad", False, False, False),
        (MEMBERS[2], "spinup_ad", False, False, True),
    ]


def test_retry_member_retries_current_stage(two_stage_ensemble, retry_calls):
    """Test that naming only a member resubmits its current (first not done) stage"""
    _set_status(two_stage_ensemble, MEMBERS[0], StageStatus.DONE, index=0)
    _set_status(two_stage_ensemble, MEMBERS[0], StageStatus.FAILED, index=1)
    two_stage_ensemble.retry(members=[MEMBERS[0]])
    assert retry_calls == [(MEMBERS[0], "transient", False, False, False)]


def test_retry_member_pending_is_resubmitted(created_ensemble, retry_calls):
    """Test that a named member's PENDING current stage can be resubmitted"""
    result = created_ensemble.retry(members=[MEMBERS[0]])
    assert list(result) == [MEMBERS[0]]
    assert retry_calls == [(MEMBERS[0], "spinup_ad", False, False, False)]


def test_retry_explicit_stage_all_members(created_ensemble, retry_calls):
    """Test that naming only a stage resubmits it for every member"""
    _set_status(created_ensemble, MEMBERS[1], StageStatus.FAILED)
    result = created_ensemble.retry(stage="spinup_ad")
    assert list(result) == MEMBERS
    assert [call[1] for call in retry_calls] == ["spinup_ad"] * len(MEMBERS)


def test_retry_explicit_done_member_skipped(created_ensemble, retry_calls, capsys):
    """Test that a named member with every stage done is skipped"""
    _set_status(created_ensemble, MEMBERS[0], StageStatus.DONE)
    assert created_ensemble.retry(members=[MEMBERS[0]]) == {}
    assert not retry_calls
    assert "my_ens-1: nothing to retry" in capsys.readouterr().out


@pytest.mark.parametrize("status", [StageStatus.SUBMITTED, StageStatus.DONE])
def test_retry_explicit_refuses_active_stage_lists_all(
    created_ensemble, retry_calls, status
):
    """Test that explicitly resubmitting a submitted or done stage raises,
    listing every refused member, and retries nothing
    """
    for member in MEMBERS[:2]:
        _set_status(created_ensemble, member, status)
    with pytest.raises(ValueError, match="can only resubmit PENDING or FAILED") as err:
        created_ensemble.retry(stage="spinup_ad")
    for member in MEMBERS[:2]:
        assert f"{member}: spinup_ad is {status.value}" in str(err.value)
    assert not retry_calls


def test_retry_forwards_dry_run_and_skip_script(created_ensemble, retry_calls):
    """Test that dry_run and skip_script are passed to each member's retry"""
    _set_status(created_ensemble, MEMBERS[0], StageStatus.FAILED)
    created_ensemble.retry(failed=True, dry_run=True, skip_script=True)
    assert retry_calls == [(MEMBERS[0], "spinup_ad", True, True, False)]


def test_retry_returns_member_to_job_id(created_ensemble, retry_calls):
    """Test that retry returns member -> its retry's job ID, in member order"""
    for member in MEMBERS:
        _set_status(created_ensemble, member, StageStatus.FAILED)
    result = created_ensemble.retry(failed=True)
    assert result == {member: f"job-{member}" for member in MEMBERS}
    assert [call[0] for call in retry_calls] == MEMBERS


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

_STATUS_COLUMNS = [
    "member",
    "stage",
    "status",
    "job_id",
    "attempts",
    "case_root",
    "submit_time",
    "end_time",
]


def test_status_columns(created_ensemble):
    """Test that status returns the documented columns, in order"""
    assert list(created_ensemble.status().columns) == _STATUS_COLUMNS


def test_status_one_row_per_member_and_stage(two_stage_ensemble):
    """Test that status has a row per member and stage, member by member
    in stage order
    """
    status = two_stage_ensemble.status()
    assert list(zip(status["member"], status["stage"])) == [
        (member, stage) for member in MEMBERS for stage in ("spinup_ad", "transient")
    ]


def test_status_new_ensemble_all_pending(created_ensemble):
    """Test that a new ensemble's stages are pending, unsubmitted, and point at
    each member's stage case
    """
    status = created_ensemble.status()
    assert (status["status"] == "pending").all()
    assert status["job_id"].isna().all()
    assert (status["attempts"] == 0).all()
    assert status["submit_time"].isna().all()
    assert status["end_time"].isna().all()
    assert list(status["case_root"]) == [
        str(created_ensemble.case_dir / f"base_case_my_ens-1_{member}_spinup_ad")
        for member in MEMBERS
    ]


def test_status_after_submit(submit_ensemble):
    """Test that a submitted member shows its job ID, attempt, and submit time"""
    job_ids = submit_ensemble.submit(limit=1)
    row = submit_ensemble.status().iloc[0]
    assert row["status"] == "submitted"
    assert row["job_id"] == job_ids[MEMBERS[0]]
    assert row["attempts"] == 1
    assert row["submit_time"] > 0
    assert row["end_time"] is None


@pytest.mark.parametrize("stage_status", list(StageStatus))
def test_status_reports_stage_status(created_ensemble, stage_status):
    """Test that each stage status is reported by its string value"""
    _set_status(created_ensemble, MEMBERS[1], stage_status)
    status = created_ensemble.status()
    assert status.set_index("member").loc[MEMBERS[1], "status"] == stage_status.value


def test_status_members_subset(created_ensemble):
    """Test that members= limits status to those members, in the order given"""
    status = created_ensemble.status(members=[MEMBERS[2], MEMBERS[0]])
    assert list(status["member"]) == [MEMBERS[2], MEMBERS[0]]


def test_status_no_members(created_ensemble):
    """Test that members=[] returns an empty frame that still has every column"""
    status = created_ensemble.status(members=[])
    assert status.empty
    assert list(status.columns) == _STATUS_COLUMNS


def test_status_unknown_member_raises(created_ensemble):
    """Test that an unknown member raises"""
    with pytest.raises(ValueError):
        created_ensemble.status(members=["nope"])


def test_status_missing_param_file_raises(created_ensemble):
    """Test that a missing parameter file for a selected member raises"""
    created_ensemble.param_files[MEMBERS[0]].unlink()
    with pytest.raises(FileNotFoundError):
        created_ensemble.status()


def test_status_doesnt_check_queue_by_default(created_ensemble, lost):
    """Test that the scheduler isn't queried and nothing is lost by default"""
    lost(None)
    status = created_ensemble.status()
    assert "lost" not in set(status["status"])


def test_status_check_queue_marks_lost(two_stage_ensemble, lost):
    """Test that check_queue reports only the lost stages as lost"""
    _set_status(two_stage_ensemble, MEMBERS[1], StageStatus.SUBMITTED, index=1)
    lost({MEMBERS[1]: ["transient"]})
    status = two_stage_ensemble.status(check_queue=True)
    lost_rows = status[status["status"] == "lost"]
    assert list(zip(lost_rows["member"], lost_rows["stage"])) == [
        (MEMBERS[1], "transient")
    ]


def test_status_without_case_root(created_ensemble):
    """Test that a stage with no case_root reports a missing value"""
    run = _member_run(created_ensemble, MEMBERS[0])
    run.stages[0].config = replace(run.stages[0].config, case_root=None)
    run.save()
    status = created_ensemble.status()
    assert list(status["case_root"].isna()) == [True, False, False]


# ---------------------------------------------------------------------------
# format_status
# ---------------------------------------------------------------------------


def _counts(text: str) -> dict[str, dict[str, int]]:
    """Parse the counts table out of format_status text as
    {stage: {status: count}}, keeping the column order
    """
    lines = text.splitlines()
    header = lines[3].split()
    counts = {}
    for line in lines[4:]:
        if not line.strip():
            break
        stage, *values = line.split()
        counts[stage] = dict(zip(header, map(int, values)))
    return counts


def _fake_status(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """A status() frame from (member, stage, status) rows"""
    return pd.DataFrame(
        [
            {
                "member": member,
                "stage": stage,
                "status": status,
                "job_id": None,
                "attempts": 0,
                "case_root": None,
                "submit_time": None,
                "end_time": None,
            }
            for member, stage, status in rows
        ],
        columns=_STATUS_COLUMNS,
    )


def test_format_status_header(created_ensemble):
    """Test that the text starts with the ensemble ID, root, and member count"""
    lines = created_ensemble.format_status(created_ensemble.status()).splitlines()
    assert lines[:3] == [
        f"Ensemble: my_ens-1 ({created_ensemble.root})",
        f"{len(MEMBERS)} members",
        "",
    ]


def test_format_status_counts_all_pending(created_ensemble):
    """Test that a new ensemble counts every member as pending, with a column
    for every status but lost
    """
    text = created_ensemble.format_status(created_ensemble.status())
    assert _counts(text) == {
        "spinup_ad": {"pending": 3, "submitted": 0, "done": 0, "failed": 0}
    }


def test_format_status_counts_per_stage(two_stage_ensemble):
    """Test that counts are per stage, with stages in config order"""
    _set_status(two_stage_ensemble, MEMBERS[0], StageStatus.DONE, index=0)
    _set_status(two_stage_ensemble, MEMBERS[0], StageStatus.FAILED, index=1)
    _set_status(two_stage_ensemble, MEMBERS[1], StageStatus.SUBMITTED, index=0)
    text = two_stage_ensemble.format_status(two_stage_ensemble.status())
    assert _counts(text) == {
        "spinup_ad": {"pending": 1, "submitted": 1, "done": 1, "failed": 0},
        "transient": {"pending": 2, "submitted": 0, "done": 0, "failed": 1},
    }


def test_format_status_no_rows(created_ensemble):
    """Test that an empty status still shows every stage with zero counts"""
    text = created_ensemble.format_status(created_ensemble.status(members=[]))
    assert _counts(text) == {
        "spinup_ad": {"pending": 0, "submitted": 0, "done": 0, "failed": 0}
    }


def test_format_status_lost_column(created_ensemble, lost):
    """Test that a lost column is added (last) only when a stage is lost"""
    _set_status(created_ensemble, MEMBERS[2], StageStatus.SUBMITTED)
    lost({MEMBERS[2]: ["spinup_ad"]})
    status = created_ensemble.status(check_queue=True)
    counts = _counts(created_ensemble.format_status(status))
    assert list(counts["spinup_ad"]) == [
        "pending",
        "submitted",
        "done",
        "failed",
        "lost",
    ]
    assert counts["spinup_ad"]["lost"] == 1
    assert counts["spinup_ad"]["submitted"] == 0


def test_format_status_nothing_failed_or_lost(created_ensemble):
    """Test that there's no failed or lost section or retry hint when nothing
    failed or was lost
    """
    text = created_ensemble.format_status(created_ensemble.status())
    assert "Failed" not in text
    assert "Lost" not in text
    assert "retry with" not in text


@pytest.mark.parametrize(
    "state, stage_status", [("failed", StageStatus.FAILED), ("lost", None)]
)
def test_format_status_lists_failed_or_lost(
    created_ensemble, lost, state, stage_status
):
    """Test that failed and lost members are listed with a retry hint"""
    if stage_status is not None:
        _set_status(created_ensemble, MEMBERS[1], stage_status)
    lost({MEMBERS[1]: ["spinup_ad"]} if state == "lost" else {})
    status = created_ensemble.status(check_queue=True)
    lines = created_ensemble.format_status(status).splitlines()
    heading = f"{state.capitalize()} (1): ['{MEMBERS[1]}:spinup_ad']"
    hint = f"  retry with: meristem ensemble retry --root {created_ensemble.root} --{state}"
    assert lines[lines.index(heading) + 1] == hint


def test_format_status_failed_and_lost(created_ensemble):
    """Test that the failed section comes before the lost section"""
    status = _fake_status(
        [
            (MEMBERS[0], "spinup_ad", "lost"),
            (MEMBERS[1], "spinup_ad", "failed"),
            (MEMBERS[2], "spinup_ad", "done"),
        ]
    )
    text = created_ensemble.format_status(status)
    assert text.index("Failed (1)") < text.index("Lost (1)")


def test_format_status_previews_at_most_ten(created_ensemble):
    """Test that the failed list shows ten members and how many more there are"""
    names = [f"m{i:02d}" for i in range(12)]
    status = _fake_status([(name, "spinup_ad", "failed") for name in names])
    text = created_ensemble.format_status(status)
    shown = [f"{name}:spinup_ad" for name in names[:10]]
    assert f"Failed (12): {shown} and 2 more" in text


def test_format_status_hides_members_by_default(created_ensemble):
    """Test that members aren't listed unless show_members is set"""
    text = created_ensemble.format_status(created_ensemble.status())
    for member in MEMBERS:
        assert member not in text


def test_format_status_show_members(created_ensemble):
    """Test that show_members appends a member table without the case root"""
    _set_status(created_ensemble, MEMBERS[1], StageStatus.FAILED)
    status = created_ensemble.status()
    text = created_ensemble.format_status(status, show_members=True)
    table = text.split("\n\n")[-1].splitlines()
    assert table[0].split() == ["member", "stage", "status", "job_id", "attempts"]
    assert [line.split()[:3] for line in table[1:]] == [
        [MEMBERS[0], "spinup_ad", "pending"],
        [MEMBERS[1], "spinup_ad", "failed"],
        [MEMBERS[2], "spinup_ad", "pending"],
    ]
    assert "case_root" not in text


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

_STAGE_COLUMNS = [
    "status",
    "job_id",
    "attempts",
    "case_root",
    "submit_time",
    "end_time",
    "archive_dir",
]


def _write_archive_case(case_root: Path, archive: str, exit_code: int = 0) -> Path:
    """Make a fake case dir whose xmlquery prints archive (its DOUT_S_ROOT)"""
    case_root.mkdir(parents=True, exist_ok=True)
    xmlquery = case_root / "xmlquery"
    xmlquery.write_text(
        f"#!/bin/sh\necho {archive}\nexit {exit_code}\n", encoding="utf-8"
    )
    xmlquery.chmod(0o755)
    return case_root


def test_manifest_columns(two_stage_ensemble):
    """Test that the manifest has the key columns, then member, parameter file,
    and each stage's columns prefixed by its name, in stage order
    """
    manifest = two_stage_ensemble.manifest()
    assert list(manifest.columns) == [
        "member",
        "fates_leaf_slatop",
        "param_file",
        *[f"spinup_ad_{c}" for c in _STAGE_COLUMNS],
        *[f"transient_{c}" for c in _STAGE_COLUMNS],
    ]


def test_manifest_one_row_per_member(created_ensemble):
    """Test that the manifest has one row per member, in member order, with its
    parameter file
    """
    manifest = created_ensemble.manifest()
    assert list(manifest["member"]) == MEMBERS
    assert list(manifest["param_file"]) == [
        str(created_ensemble.param_files[member]) for member in MEMBERS
    ]


def test_manifest_stage_values_match_status(two_stage_ensemble):
    """Test that each stage's columns hold that member's status() values"""
    _set_status(two_stage_ensemble, MEMBERS[1], StageStatus.DONE, index=0)
    _set_status(two_stage_ensemble, MEMBERS[1], StageStatus.FAILED, index=1)
    manifest = two_stage_ensemble.manifest().set_index("member")
    assert list(manifest["spinup_ad_status"]) == ["pending", "done", "pending"]
    assert list(manifest["transient_status"]) == ["pending", "failed", "pending"]
    status = two_stage_ensemble.status().set_index(["member", "stage"])
    for member in MEMBERS:
        for stage in ("spinup_ad", "transient"):
            assert (
                manifest.loc[member, f"{stage}_case_root"]
                == status.loc[(member, stage), "case_root"]
            )


def test_manifest_key_values(created_ensemble):
    """Test that each member's paramgen key values are merged in, as strings"""
    manifest = created_ensemble.manifest()
    assert list(manifest["fates_leaf_slatop"]) == ["0.0", "0.1", "0.2"]
    assert "ensemble" not in manifest.columns


def test_manifest_key_only_selected_members(ready_config_dict):
    """Test that key rows for members outside the ensemble are left out"""
    ready_config_dict["paramgen"]["members"] = [MEMBERS[1]]
    ensemble = Ensemble.create(ready_config_dict)
    manifest = ensemble.manifest()
    assert list(manifest["member"]) == [MEMBERS[1]]
    assert list(manifest["fates_leaf_slatop"]) == ["0.1"]


def test_manifest_without_key_file(created_ensemble):
    """Test that the manifest is built without key columns when the key file
    is gone
    """
    key_file = created_ensemble.config.paramgen_dir / "ens_key.csv"
    key_file.unlink()
    manifest = created_ensemble.manifest()
    assert list(manifest.columns[:2]) == ["member", "param_file"]
    assert list(manifest["member"]) == MEMBERS


def test_manifest_no_archive_dir_before_cases_exist(created_ensemble):
    """Test that archive_dir is empty for cases that haven't been cloned yet"""
    manifest = created_ensemble.manifest()
    assert manifest["spinup_ad_archive_dir"].isna().all()


def test_manifest_archive_dir_from_case(created_ensemble, tmp_path):
    """Test that archive_dir is the case's DOUT_S_ROOT once the case exists"""
    archive = tmp_path / "archive" / MEMBERS[0]
    _write_archive_case(_stage_case_root(created_ensemble), str(archive))
    manifest = created_ensemble.manifest().set_index("member")
    assert manifest.loc[MEMBERS[0], "spinup_ad_archive_dir"] == str(archive)
    assert manifest.loc[MEMBERS[1:], "spinup_ad_archive_dir"].isna().all()


def test_manifest_without_case_root(created_ensemble):
    """Test that a stage with no case_root has no archive_dir"""
    run = _member_run(created_ensemble, MEMBERS[0])
    run.stages[0].config = replace(run.stages[0].config, case_root=None)
    run.save()
    manifest = created_ensemble.manifest()
    assert manifest["spinup_ad_archive_dir"].isna().all()


@pytest.mark.parametrize("case_root", [None, ""])
def test_archive_dir_no_case_root(case_root):
    """Test that there's no archive dir without a case root"""
    assert _archive_dir(case_root) is None


def test_archive_dir_no_xmlquery(tmp_path):
    """Test that there's no archive dir for a directory that isn't a case"""
    assert _archive_dir(str(tmp_path)) is None


def test_archive_dir_xmlquery_fails(tmp_path):
    """Test that there's no archive dir when xmlquery fails"""
    case_root = _write_archive_case(tmp_path / "case", "boom", exit_code=1)
    assert _archive_dir(str(case_root)) is None


def test_archive_dir_xmlquery_not_executable(tmp_path):
    """Test that there's no archive dir when xmlquery can't be run"""
    case_root = _write_archive_case(tmp_path / "case", "/scratch/archive/case")
    (case_root / "xmlquery").chmod(0o644)
    assert _archive_dir(str(case_root)) is None


def test_archive_dir_value(tmp_path):
    """Test that the archive dir is xmlquery's DOUT_S_ROOT, stripped"""
    case_root = _write_archive_case(tmp_path / "case", "/scratch/archive/case")
    assert _archive_dir(str(case_root)) == "/scratch/archive/case"


# ---------------------------------------------------------------------------
# find_param_members
# ---------------------------------------------------------------------------


def test_find_param_members_all(ensemble_config_dict, write_paramgen):
    """Test that members: all returns every member, in list-file order"""
    write_paramgen(members=["ens_0002", "ens_0000", "ens_0001"])
    assert _ensemble(ensemble_config_dict).find_param_members() == [
        "ens_0002",
        "ens_0000",
        "ens_0001",
    ]


def test_find_param_members_subset_in_list_file_order(
    ensemble_config_dict, write_paramgen
):
    """Test that requested members are returned in list-file order, not the
    order they were requested in
    """
    write_paramgen()
    ensemble = _with_members(ensemble_config_dict, ["ens_0002", "ens_0000"])
    assert ensemble.find_param_members() == ["ens_0000", "ens_0002"]


def test_find_param_members_ignores_blank_lines_and_whitespace(
    ensemble_config_dict, write_paramgen
):
    """Test that blank lines and surrounding whitespace in the list are ignored"""
    directory = write_paramgen()
    (directory / "ens.txt").write_text(
        "\n  ens_0000  \n\n\tens_0001\nens_0002\n\n", encoding="utf-8"
    )
    assert _ensemble(ensemble_config_dict).find_param_members() == MEMBERS


def test_find_param_members_missing_list_file_raises(ensemble_config_dict):
    """Test that a missing <prefix>.txt raises, naming the prefix"""
    with pytest.raises(FileNotFoundError, match=r"ens\.txt not found.*'ens'"):
        _ensemble(ensemble_config_dict).find_param_members()


@pytest.mark.parametrize("contents", ["", "\n\n  \n"])
def test_find_param_members_empty_list_file_raises(
    ensemble_config_dict, write_paramgen, contents
):
    """Test that an empty (or whitespace-only) list file raises"""
    directory = write_paramgen()
    (directory / "ens.txt").write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match="is empty"):
        _ensemble(ensemble_config_dict).find_param_members()


def test_find_param_members_duplicates_raise(ensemble_config_dict, write_paramgen):
    """Test that duplicates in the list file raise, each listed once"""
    write_paramgen(members=["b", "a", "c", "a", "b", "b"])
    with pytest.raises(ValueError, match=r"has duplicates: \['a', 'b'\]"):
        _ensemble(ensemble_config_dict).find_param_members()


def test_find_param_members_unknown_requested_raises(
    ensemble_config_dict, write_paramgen
):
    """Test that requesting members not in the list file raises, listing them"""
    write_paramgen()
    ensemble = _with_members(ensemble_config_dict, ["ens_0001", "zz", "aa"])
    with pytest.raises(ValueError, match=r"requested members not in .*\['aa', 'zz'\]"):
        ensemble.find_param_members()


def test_find_param_members_uses_prefix(ensemble_config_dict, write_paramgen):
    """Test that the list file is chosen by the paramgen prefix"""
    write_paramgen()
    ensemble_config_dict["paramgen"]["prefix"] = "other"
    with pytest.raises(FileNotFoundError, match=r"other\.txt"):
        _ensemble(ensemble_config_dict).find_param_members()


# ---------------------------------------------------------------------------
# find_param_files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ext", [".nc", ".json"])
def test_find_param_files_all(ensemble_config_dict, write_paramgen, ext):
    """Test that each member maps to its .nc or .json parameter file, in order"""
    directory = write_paramgen(ext=ext)
    param_files = _ensemble(ensemble_config_dict).find_param_files()
    assert param_files == {m: directory / f"{m}{ext}" for m in MEMBERS}
    assert list(param_files) == MEMBERS


def test_find_param_files_mixed_extensions(ensemble_config_dict, write_paramgen):
    """Test that members can have different (but single) file extensions"""
    directory = write_paramgen(param_files=MEMBERS[:2])
    (directory / f"{MEMBERS[2]}.json").touch()
    param_files = _ensemble(ensemble_config_dict).find_param_files()
    assert param_files[MEMBERS[2]] == directory / f"{MEMBERS[2]}.json"


def test_find_param_files_subset(ensemble_config_dict, write_paramgen):
    """Test that only requested members are returned, and unrequested members
    don't need parameter files
    """
    directory = write_paramgen(param_files=["ens_0001"])
    ensemble = _with_members(ensemble_config_dict, ["ens_0001"])
    assert ensemble.find_param_files() == {"ens_0001": directory / "ens_0001.nc"}


def test_find_param_files_ignores_other_files(ensemble_config_dict, write_paramgen):
    """Test that unrelated files in the paramgen dir don't affect the result"""
    directory = write_paramgen()
    (directory / "ens_0000.txt").touch()
    (directory / "ens_0000.nc.bak").touch()
    (directory / "ens_config.yaml").touch()
    param_files = _ensemble(ensemble_config_dict).find_param_files()
    assert param_files["ens_0000"] == directory / "ens_0000.nc"


def test_find_param_files_member_list_error_propagates(ensemble_config_dict):
    """Test that find_param_files raises find_param_members' errors"""
    with pytest.raises(FileNotFoundError, match="member list"):
        _ensemble(ensemble_config_dict).find_param_files()


def test_find_param_files_missing_key_file_raises(ensemble_config_dict, write_paramgen):
    """Test that a missing <prefix>_key.csv raises, naming the prefix"""
    directory = write_paramgen()
    (directory / "ens_key.csv").unlink()
    with pytest.raises(FileNotFoundError, match=r"ens_key\.csv not found.*'ens'"):
        _ensemble(ensemble_config_dict).find_param_files()


def test_find_param_files_member_not_in_key_raises(
    ensemble_config_dict, write_paramgen
):
    """Test that a member missing from the key csv is reported"""
    write_paramgen(key_members=MEMBERS[:-1])
    with pytest.raises(
        FileNotFoundError, match=rf"{MEMBERS[-1]}: {MEMBERS[-1]} not in .*ens_key"
    ):
        _ensemble(ensemble_config_dict).find_param_files()


def test_find_param_files_no_ensemble_column_raises(
    ensemble_config_dict, write_paramgen
):
    """Test that a key csv without an ensemble column raises a ValueError
    naming the columns it does have
    """
    directory = write_paramgen()
    (directory / "ens_key.csv").write_text(
        "member,fates_leaf_slatop\nens_0000,0.1\n", encoding="utf-8"
    )
    with pytest.raises(
        ValueError, match=r"no 'ensemble' column \(columns: \['member', 'fates"
    ):
        _ensemble(ensemble_config_dict).find_param_files()


def test_find_param_files_numeric_member_names(ensemble_config_dict, write_paramgen):
    """Test that all-digit member names match the key csv (pandas reads them as
    ints)
    """
    directory = write_paramgen(members=["0001", "0002"])
    param_files = _ensemble(ensemble_config_dict).find_param_files()
    assert param_files == {m: directory / f"{m}.nc" for m in ["0001", "0002"]}


def test_find_param_files_no_param_file_raises(ensemble_config_dict, write_paramgen):
    """Test that a member with neither a .nc nor a .json file is reported"""
    write_paramgen(param_files=MEMBERS[1:])
    with pytest.raises(
        FileNotFoundError, match=r"ens_0000: no ens_0000\.nc or ens_0000\.json"
    ):
        _ensemble(ensemble_config_dict).find_param_files()


def test_find_param_files_both_extensions_raises(ensemble_config_dict, write_paramgen):
    """Test that a member with both a .nc and a .json file is ambiguous"""
    directory = write_paramgen()
    (directory / "ens_0001.json").touch()
    with pytest.raises(FileNotFoundError, match="ens_0001: both .nc and .json"):
        _ensemble(ensemble_config_dict).find_param_files()


def test_find_param_files_reports_all_problems(ensemble_config_dict, write_paramgen):
    """Test that every problem member is reported in one error, not just the first"""
    directory = write_paramgen(
        key_members=["ens_0001", "ens_0002"], param_files=["ens_0001"]
    )
    (directory / "ens_0001.json").touch()
    with pytest.raises(FileNotFoundError) as excinfo:
        _ensemble(ensemble_config_dict).find_param_files()
    lines = str(excinfo.value).splitlines()[1:]
    assert lines[0].startswith("  ens_0000: ens_0000 not in")
    assert lines[1] == "  ens_0001: both .nc and .json exist"
    assert lines[2] == "  ens_0002: no ens_0002.nc or ens_0002.json"


def test_find_param_files_problem_outside_selection_ignored(
    ensemble_config_dict, write_paramgen
):
    """Test that problems with unrequested members don't raise"""
    directory = write_paramgen(key_members=["ens_0000"], param_files=["ens_0000"])
    ensemble = _with_members(ensemble_config_dict, ["ens_0000"])
    assert ensemble.find_param_files() == {"ens_0000": directory / "ens_0000.nc"}


# ---------------------------------------------------------------------------
# check_base_case
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["TRUE", "true", "True"])
def test_check_base_case_built_passes(ensemble_config_dict, write_base_case, value):
    """Test that a base case with BUILD_COMPLETE TRUE (any case) passes"""
    write_base_case(build_complete=value)
    _ensemble(ensemble_config_dict).check_base_case()


def test_check_base_case_missing_dir_raises(ensemble_config_dict):
    """Test that a nonexistent base case raises"""
    with pytest.raises(FileNotFoundError, match="cannot find base case"):
        _ensemble(ensemble_config_dict).check_base_case()


def test_check_base_case_is_file_raises(ensemble_config_dict):
    """Test that a base case path that's a file, not a dir, raises"""
    Path(ensemble_config_dict["base_case"]).touch()
    with pytest.raises(FileNotFoundError, match="cannot find base case"):
        _ensemble(ensemble_config_dict).check_base_case()


def test_check_base_case_no_xmlquery_raises(ensemble_config_dict):
    """Test that a dir without xmlquery (not a CIME case) raises"""
    Path(ensemble_config_dict["base_case"]).mkdir()
    with pytest.raises(FileNotFoundError, match="no xmlquery"):
        _ensemble(ensemble_config_dict).check_base_case()


@pytest.mark.parametrize("value", ["FALSE", "''"])
def test_check_base_case_not_built_raises(ensemble_config_dict, write_base_case, value):
    """Test that a base case whose BUILD_COMPLETE isn't TRUE raises"""
    write_base_case(build_complete=value)
    with pytest.raises(ValueError, match="is not built.*case.build"):
        _ensemble(ensemble_config_dict).check_base_case()


def test_check_base_case_xmlquery_fails_raises(ensemble_config_dict, write_base_case):
    """Test that a failing xmlquery raises RuntimeError with its exit code and
    stderr, even if it printed TRUE
    """
    write_base_case(exit_code=3, stderr="ERROR: env_build.xml is corrupt")
    with pytest.raises(RuntimeError, match=r"exit 3\): ERROR: env_build\.xml"):
        _ensemble(ensemble_config_dict).check_base_case()


def test_check_base_case_xmlquery_fails_without_stderr_reports_stdout(
    ensemble_config_dict, write_base_case
):
    """Test that a failing xmlquery with nothing on stderr reports its stdout"""
    write_base_case(build_complete="oops", exit_code=1)
    with pytest.raises(RuntimeError, match=r"exit 1\): oops"):
        _ensemble(ensemble_config_dict).check_base_case()


def test_check_base_case_xmlquery_not_executable_raises(
    ensemble_config_dict, write_base_case
):
    """Test that an xmlquery without execute permission raises PermissionError"""
    base_case = write_base_case()
    (base_case / "xmlquery").chmod(0o644)
    with pytest.raises(PermissionError, match="xmlquery is not executable"):
        _ensemble(ensemble_config_dict).check_base_case()


def test_check_base_case_runs_in_base_case_dir(
    ensemble_config_dict, write_base_case, tmp_path, monkeypatch
):
    """Test that xmlquery runs from the base case, not the current directory"""
    write_base_case()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    _ensemble(ensemble_config_dict).check_base_case()


# ---------------------------------------------------------------------------
# check_param_files
# ---------------------------------------------------------------------------


def test_check_param_files_all_present(created_ensemble):
    """Test that check_param_files passes when every parameter file exists"""
    created_ensemble.check_param_files()
    created_ensemble.check_param_files([MEMBERS[0]])


def test_check_param_files_missing_raises(created_ensemble):
    """Test that a missing parameter file raises, naming the member and file"""
    path = created_ensemble.param_files[MEMBERS[1]]
    path.unlink()
    with pytest.raises(
        FileNotFoundError, match=f"{MEMBERS[1]}: {re.escape(str(path))}"
    ):
        created_ensemble.check_param_files()


def test_check_param_files_only_checks_given_members(created_ensemble):
    """Test that a missing file for a member not asked about doesn't raise"""
    created_ensemble.param_files[MEMBERS[1]].unlink()
    created_ensemble.check_param_files([MEMBERS[0]])


def test_check_param_files_unknown_member_raises(created_ensemble):
    """Test that an unknown member raises a ValueError showing at most three
    unknown names, the number of members, and the first three members
    """
    unknown = ["x1", "x2", "x3", "x4", "x5"]
    with pytest.raises(ValueError) as excinfo:
        created_ensemble.check_param_files([MEMBERS[0], *unknown])
    assert str(excinfo.value) == (
        "unknown member(s) ['x1', 'x2', 'x3'] and 2 more; the ensemble has "
        f"{len(MEMBERS)} members, e.g. {MEMBERS[:3]}."
    )


@pytest.mark.parametrize(
    "names, expected",
    [
        ([], "[]"),
        (["a", "b", "c"], "['a', 'b', 'c']"),
        (["a", "b", "c", "d"], "['a', 'b', 'c'] and 1 more"),
    ],
)
def test_preview(names, expected):
    """Test that _preview shows at most three names and counts the rest"""
    assert _preview(names) == expected


# ---------------------------------------------------------------------------
# check_stage_scripts
# ---------------------------------------------------------------------------


def _stage(name: str, script: Path | None = None) -> dict:
    """A member stage dict running script, or no script if None"""
    if script is None:
        return {"name": name, "queue": "regular"}
    return {"name": name, "script": str(script), "queue": "regular"}


def test_check_stage_scripts_no_script_stages_pass(ensemble_config_dict):
    """Test that stages without a script need no file"""
    _ensemble(ensemble_config_dict).check_stage_scripts()


def test_check_stage_scripts_existing_scripts_pass(ensemble_config_dict, tmp_path):
    """Test that stages whose scripts exist pass, alongside stages without one"""
    script = tmp_path / "setup.sh"
    script.touch()
    ensemble_config_dict["member_stages"] = [_stage("setup", script), _stage("run")]
    _ensemble(ensemble_config_dict).check_stage_scripts()


def test_check_stage_scripts_directory_raises(ensemble_config_dict, tmp_path):
    """Test that a script path that's a directory, not a file, raises"""
    script = tmp_path / "scripts"
    script.mkdir()
    ensemble_config_dict["member_stages"] = [_stage("setup", script)]
    with pytest.raises(FileNotFoundError, match="setup"):
        _ensemble(ensemble_config_dict).check_stage_scripts()


def test_check_stage_scripts_reports_all_missing(ensemble_config_dict, tmp_path):
    """Test that every missing script is reported in one error, in stage order,
    and stages with existing scripts or no script aren't
    """
    present = tmp_path / "present.sh"
    present.touch()
    ensemble_config_dict["member_stages"] = [
        _stage("first", tmp_path / "a.sh"),
        _stage("ok", present),
        _stage("no_script"),
        _stage("last", tmp_path / "b.sh"),
    ]
    with pytest.raises(FileNotFoundError) as excinfo:
        _ensemble(ensemble_config_dict).check_stage_scripts()
    lines = str(excinfo.value).splitlines()
    assert lines[0] == "member stage script(s) not found:"
    assert lines[1:] == [
        f"  first: {tmp_path / 'a.sh'}",
        f"  last: {tmp_path / 'b.sh'}",
    ]


# ---------------------------------------------------------------------------
# _is_existing_case
# ---------------------------------------------------------------------------


def _write_case(case_root: Path, case: str | None = None, exit_code: int = 0) -> Path:
    """Make a fake CIME case dir whose xmlquery prints case (default: its name)"""
    case_root.mkdir(parents=True)
    (case_root / "env_case.xml").touch()
    xmlquery = case_root / "xmlquery"
    xmlquery.write_text(
        f"#!/bin/sh\necho {case or case_root.name}\nexit {exit_code}\n",
        encoding="utf-8",
    )
    xmlquery.chmod(0o755)
    return case_root


def test_is_existing_case_missing(tmp_path):
    """Test that a case_root that doesn't exist isn't reused"""
    assert _is_existing_case(tmp_path / "base_ens_0000_spinup") is False


def test_is_existing_case_complete(tmp_path):
    """Test that a complete case with the right name is reused"""
    assert _is_existing_case(_write_case(tmp_path / "base_ens_0000_spinup")) is True


def test_is_existing_case_partial_raises(tmp_path):
    """Test that a directory that isn't a complete case raises and is kept"""
    case_root = tmp_path / "base_ens_0000_spinup"
    case_root.mkdir()
    with pytest.raises(FileExistsError, match="not a complete CIME case"):
        _is_existing_case(case_root)
    assert case_root.is_dir()


def test_is_existing_case_xmlquery_fails_raises(tmp_path):
    """Test that a case whose xmlquery fails raises"""
    case_root = _write_case(tmp_path / "base_ens_0000_spinup", exit_code=1)
    with pytest.raises(FileExistsError, match="xmlquery CASE failed"):
        _is_existing_case(case_root)


def test_is_existing_case_wrong_case_raises(tmp_path):
    """Test that a case holding a different case name raises"""
    case_root = _write_case(tmp_path / "base_ens_0000_spinup", case="other")
    with pytest.raises(FileExistsError, match="holds case 'other'"):
        _is_existing_case(case_root)


def _stage_case_root(ensemble: Ensemble) -> Path | None:
    """The first member's first stage case_root, as stored on its ModelRun"""
    return _member_run(ensemble, MEMBERS[0]).stages[0].config.case_root


def _prepare(ensemble: Ensemble):
    """Prepare the first member's first stage"""
    ensemble.prepare(MEMBERS[0], ensemble.config.member_stages[0].name)


def test_prepare_partial_case_root_raises(created_ensemble):
    """Test that prepare raises if the stage's case_root is a partial case"""
    _stage_case_root(created_ensemble).mkdir(parents=True)
    with pytest.raises(FileExistsError, match="not a complete CIME case"):
        _prepare(created_ensemble)


# ---------------------------------------------------------------------------
# prepare: clone
# ---------------------------------------------------------------------------

# what the fake create_clone does once it has parsed --case into $case_root
_CREATE_CLONE_BODIES = {
    # a complete clone: what _is_existing_case accepts, plus a case.setup
    # that logs each run to case.setup.log
    "ok": (
        'mkdir "$case_root"\n'
        'touch "$case_root/env_case.xml"\n'
        'printf \'#!/bin/sh\\necho %s\\n\' "$(basename "$case_root")" '
        '> "$case_root/xmlquery"\n'
        'chmod 755 "$case_root/xmlquery"\n'
        "printf '#!/bin/sh\\necho ran >> case.setup.log\\n' "
        '> "$case_root/case.setup"\n'
        'chmod 755 "$case_root/case.setup"\n'
    ),
    "fail": "echo 'clone exploded' >&2\nexit 1\n",
    "fail_partial": "mkdir \"$case_root\"\necho 'clone exploded' >&2\nexit 1\n",
    "noop": "exit 0\n",
}


@pytest.fixture
def fake_cime(ensemble_config_dict, tmp_path):
    """Factory that points the base case's CIMEROOT at a fake CIME whose
    scripts/create_clone logs its arguments (one per line) and behaves
    as mode (a _CREATE_CLONE_BODIES key)

    Args (of the returned function):
        mode: create_clone behavior
        executable: whether create_clone is executable
        present: whether create_clone exists at all

    Returns (of the returned function):
        Path: the create_clone argument log
    """
    base_case = Path(ensemble_config_dict["base_case"])
    cimeroot = tmp_path / "cime"
    log = tmp_path / "create_clone.log"

    def _write(mode="ok", executable=True, present=True) -> Path:
        base_case.mkdir(parents=True, exist_ok=True)
        xmlquery = base_case / "xmlquery"
        xmlquery.write_text(
            "#!/bin/sh\n"
            f'case "$1" in\n  CIMEROOT) echo {cimeroot} ;;\n  *) echo TRUE ;;\nesac\n',
            encoding="utf-8",
        )
        xmlquery.chmod(0o755)
        if present:
            create_clone = cimeroot / "scripts" / "create_clone"
            create_clone.parent.mkdir(parents=True, exist_ok=True)
            create_clone.write_text(
                "#!/bin/sh\n"
                f'printf "%s\\n" "$@" >> {log}\n'
                "while [ $# -gt 0 ]; do\n"
                '  [ "$1" = "--case" ] && case_root=$2\n'
                "  shift\n"
                "done\n" + _CREATE_CLONE_BODIES[mode],
                encoding="utf-8",
            )
            create_clone.chmod(0o755 if executable else 0o644)
        return log

    return _write


def test_prepare_clones_case(created_ensemble, fake_cime):
    """Test that prepare runs create_clone --keepexe from the base case, into
    the member run's case_root, with the member run's project
    """
    log = fake_cime()
    _prepare(created_ensemble)
    case_root = _stage_case_root(created_ensemble)
    assert _is_existing_case(case_root)
    args = log.read_text(encoding="utf-8").splitlines()
    assert args[args.index("--case") + 1] == str(case_root)
    assert args[args.index("--clone") + 1] == str(created_ensemble.base_case)
    assert "--keepexe" in args
    assert args[args.index("--project") + 1] == "PROJ123"
    assert "--cime-output-root" not in args


def test_prepare_creates_case_dir(created_ensemble, fake_cime):
    """Test that prepare makes case_dir if it doesn't exist yet"""
    fake_cime()
    assert not created_ensemble.case_dir.exists()
    _prepare(created_ensemble)
    assert created_ensemble.case_dir.is_dir()


def test_prepare_passes_output_root(ready_config_dict, fake_cime, tmp_path):
    """Test that output_root is passed as --cime-output-root (two arguments)"""
    ready_config_dict["output_root"] = str(tmp_path / "output")
    ensemble = Ensemble.create(ready_config_dict)
    log = fake_cime()
    _prepare(ensemble)
    args = log.read_text(encoding="utf-8").splitlines()
    assert args[args.index("--cime-output-root") + 1] == str(tmp_path / "output")


@pytest.mark.parametrize(
    "kwargs", [{"present": False}, {"executable": False}], ids=["missing", "not_exec"]
)
def test_prepare_no_create_clone_raises(created_ensemble, fake_cime, kwargs):
    """Test that a missing or non-executable create_clone raises, naming it"""
    fake_cime(**kwargs)
    with pytest.raises(FileNotFoundError, match="no executable create_clone at"):
        _prepare(created_ensemble)


def test_prepare_clone_failure_raises(created_ensemble, fake_cime):
    """Test that a failed create_clone raises with its stderr, and no
    leftover hint when it left nothing behind
    """
    fake_cime(mode="fail")
    with pytest.raises(RuntimeError, match="clone exploded") as excinfo:
        _prepare(created_ensemble)
    assert f"cloning {created_ensemble.base_case} failed" in str(excinfo.value)
    assert str(_stage_case_root(created_ensemble)) in str(excinfo.value)
    assert "partial" not in str(excinfo.value)


def test_prepare_clone_failure_partial_raises(created_ensemble, fake_cime):
    """Test that a failed create_clone that left a partial case says to remove
    it, and leaves it in place
    """
    fake_cime(mode="fail_partial")
    with pytest.raises(RuntimeError, match="remove it before retrying"):
        _prepare(created_ensemble)
    assert _stage_case_root(created_ensemble).is_dir()


def test_prepare_clone_success_without_case_raises(created_ensemble, fake_cime):
    """Test that create_clone exiting 0 without making the case raises"""
    fake_cime(mode="noop")
    with pytest.raises(RuntimeError, match="was not created"):
        _prepare(created_ensemble)


def test_prepare_clones_then_reuses_case(created_ensemble, fake_cime):
    """Test that prepare clones the stage's case, and a re-run reuses it"""
    log = fake_cime()
    _prepare(created_ensemble)
    _prepare(created_ensemble)
    assert _is_existing_case(_stage_case_root(created_ensemble))
    assert log.read_text(encoding="utf-8").splitlines().count("--keepexe") == 1


def test_prepare_runs_case_setup_every_time(created_ensemble, fake_cime):
    """Test that prepare runs case.setup on a new clone, and again on a re-run"""
    fake_cime()
    _prepare(created_ensemble)
    _prepare(created_ensemble)
    setup_log = _stage_case_root(created_ensemble) / "case.setup.log"
    assert setup_log.read_text(encoding="utf-8").splitlines() == ["ran", "ran"]


# ---------------------------------------------------------------------------
# prepare: case.setup
# ---------------------------------------------------------------------------


def _write_case_setup(ensemble: Ensemble, body: str, mode: int = 0o755) -> Path:
    """Make the first member's first stage an existing case (so prepare
    doesn't clone it) with a fake case.setup running body

    Returns:
        Path: the case_root
    """
    case_root = _write_case(_stage_case_root(ensemble))
    case_setup = case_root / "case.setup"
    case_setup.write_text(f"#!/bin/sh\n{body}", encoding="utf-8")
    case_setup.chmod(mode)
    return case_root


def test_prepare_runs_case_setup_in_case_dir(created_ensemble):
    """Test that prepare runs case.setup from the case directory"""
    case_root = _write_case_setup(created_ensemble, "pwd > ran_in\n")
    _prepare(created_ensemble)
    assert (case_root / "ran_in").read_text(encoding="utf-8").strip() == str(case_root)


def test_prepare_case_setup_missing_raises(created_ensemble):
    """Test that a case with no case.setup raises"""
    case_root = _write_case_setup(created_ensemble, "exit 0\n")
    (case_root / "case.setup").unlink()
    with pytest.raises(FileNotFoundError, match="has no case.setup"):
        _prepare(created_ensemble)


def test_prepare_case_setup_not_executable_raises(created_ensemble):
    """Test that a non-executable case.setup raises"""
    _write_case_setup(created_ensemble, "exit 0\n", mode=0o644)
    with pytest.raises(PermissionError, match="case.setup is not executable"):
        _prepare(created_ensemble)


def test_prepare_case_setup_failure_raises(created_ensemble):
    """Test that a failed case.setup raises with its stderr"""
    _write_case_setup(created_ensemble, "echo 'setup exploded' >&2\nexit 1\n")
    with pytest.raises(RuntimeError, match="case.setup failed: .*setup exploded"):
        _prepare(created_ensemble)


# ---------------------------------------------------------------------------
# finalize
# ---------------------------------------------------------------------------

# a fake preview_namelists that "generates" lnd_in from user_nl_clm
_COPY_TO_LND_IN = "cp user_nl_clm CaseDocs/lnd_in\n"


def _write_namelist_case(
    ensemble: Ensemble,
    user_nl_clm: str | None = "",
    lnd_in: str | None = None,
    body: str = _COPY_TO_LND_IN,
    mode: int = 0o755,
) -> Path:
    """Make the first member's first stage an existing case with user_nl_clm
    and CaseDocs/lnd_in (each none if None) and a fake preview_namelists
    running body

    Returns:
        Path: the case_root
    """
    case_root = _write_case(_stage_case_root(ensemble))
    if user_nl_clm is not None:
        (case_root / "user_nl_clm").write_text(user_nl_clm, encoding="utf-8")
    (case_root / "CaseDocs").mkdir()
    if lnd_in is not None:
        (case_root / "CaseDocs" / "lnd_in").write_text(lnd_in, encoding="utf-8")
    preview_namelists = case_root / "preview_namelists"
    preview_namelists.write_text(f"#!/bin/sh\n{body}", encoding="utf-8")
    preview_namelists.chmod(mode)
    return case_root


def _finalize(ensemble: Ensemble):
    """Finalize the first member's first stage"""
    ensemble.finalize(MEMBERS[0], ensemble.config.member_stages[0].name)


def _param_file(ensemble: Ensemble) -> str:
    """The first member's parameter file"""
    return str(ensemble.param_files[MEMBERS[0]])


def _assignment(ensemble: Ensemble, key: str = "fates_paramfile") -> str:
    """The user_nl_clm line finalize writes for the first member"""
    return f"{key} = '{_param_file(ensemble)}'\n"


def test_finalize_unknown_stage_raises(created_ensemble):
    """Test that finalize raises for a stage that isn't a member stage"""
    with pytest.raises(ValueError, match="unknown member stage 'nope'"):
        created_ensemble.finalize(MEMBERS[0], "nope")


def test_finalize_unknown_member_raises(created_ensemble):
    """Test that finalize raises a ValueError for a member not in the ensemble"""
    with pytest.raises(ValueError, match=r"unknown member\(s\) \['nope'\]"):
        created_ensemble.finalize("nope", "spinup_ad")


def test_finalize_unprepared_case_raises(created_ensemble):
    """Test that finalize raises if the stage's case hasn't been made"""
    with pytest.raises(FileNotFoundError, match="run prepare first"):
        _finalize(created_ensemble)


def test_finalize_sets_then_checks_paramfile(created_ensemble):
    """Test that finalize sets the key in user_nl_clm, then checks the lnd_in
    preview_namelists regenerates (not the stale one already there)
    """
    case_root = _write_namelist_case(
        created_ensemble, "a = 1\n", lnd_in="fates_paramfile = '/stale.nc'\n"
    )
    _finalize(created_ensemble)
    expected = "a = 1\n" + _assignment(created_ensemble)
    assert (case_root / "user_nl_clm").read_text(encoding="utf-8") == expected


# ---------------------------------------------------------------------------
# finalize: set parameter file
# ---------------------------------------------------------------------------


def _finalized_user_nl_clm(ensemble: Ensemble, text: str) -> str:
    """Write text as the first member stage's user_nl_clm, finalize, and
    return the resulting user_nl_clm
    """
    case_root = _write_namelist_case(ensemble, text)
    _finalize(ensemble)
    return (case_root / "user_nl_clm").read_text(encoding="utf-8")


def test_finalize_appends_to_empty_user_nl_clm(created_ensemble):
    """Test that the key is added to an empty user_nl_clm"""
    result = _finalized_user_nl_clm(created_ensemble, "")
    assert result == _assignment(created_ensemble)


def test_finalize_keeps_other_user_nl_clm_lines(created_ensemble):
    """Test that unrelated lines are kept, in order, before the new key"""
    text = "hist_nhtfrq = 0\n! a comment\nuse_fates = .true.\n"
    result = _finalized_user_nl_clm(created_ensemble, text)
    assert result == text + _assignment(created_ensemble)


@pytest.mark.parametrize(
    "line",
    [
        "fates_paramfile = '/old.nc'",
        "FATES_PARAMFILE = '/old.nc'",
        "  Fates_ParamFile='/old.nc'",
        "\tfates_paramfile\t=\t'/old.nc'",
    ],
)
def test_finalize_replaces_existing_paramfile(created_ensemble, line):
    """Test that an existing assignment is replaced regardless of case or
    whitespace
    """
    result = _finalized_user_nl_clm(created_ensemble, f"a = 1\n{line}\nb = 2\n")
    assert result == "a = 1\nb = 2\n" + _assignment(created_ensemble)


def test_finalize_removes_duplicate_paramfiles(created_ensemble):
    """Test that every existing assignment is removed, leaving exactly one"""
    text = "fates_paramfile = '/x.nc'\nfates_paramfile = '/y.nc'\n"
    result = _finalized_user_nl_clm(created_ensemble, text)
    assert result == _assignment(created_ensemble)


def test_finalize_is_idempotent(created_ensemble):
    """Test that finalizing twice gives the same user_nl_clm as once"""
    user_nl_clm = _write_namelist_case(created_ensemble, "a = 1\n") / "user_nl_clm"
    _finalize(created_ensemble)
    once = user_nl_clm.read_text(encoding="utf-8")
    _finalize(created_ensemble)
    assert user_nl_clm.read_text(encoding="utf-8") == once


def test_finalize_set_key_not_matched_by_suffix(ready_config_dict):
    """Test that paramfile_key paramfile leaves fates_paramfile alone"""
    ready_config_dict["paramfile_key"] = "paramfile"
    ensemble = Ensemble.create(ready_config_dict)
    text = "fates_paramfile = '/f.nc'\nparamfile = '/old.nc'\n"
    result = _finalized_user_nl_clm(ensemble, text)
    assert result == "fates_paramfile = '/f.nc'\n" + _assignment(ensemble, "paramfile")


def test_finalize_keeps_key_in_comment_or_value(created_ensemble):
    """Test that lines only mentioning the key (not assigning it) are kept"""
    text = "! fates_paramfile = '/c.nc'\nfoo = 'fates_paramfile'\n"
    result = _finalized_user_nl_clm(created_ensemble, text)
    assert result == text + _assignment(created_ensemble)


@pytest.mark.parametrize(
    "text, kept",
    [
        ("a = 1", "a = 1\n"),
        ("a = 1\nb = 2", "a = 1\nb = 2\n"),
        ("a = 1\nfates_paramfile = '/old.nc'", "a = 1\n"),
    ],
    ids=["one_line", "two_lines", "key_last"],
)
def test_finalize_user_nl_clm_no_trailing_newline(created_ensemble, text, kept):
    """Test that the new line isn't glued onto a last line with no newline"""
    result = _finalized_user_nl_clm(created_ensemble, text)
    assert result == kept + _assignment(created_ensemble)


def test_finalize_missing_user_nl_clm_raises(created_ensemble):
    """Test that a case with no user_nl_clm raises"""
    _write_namelist_case(created_ensemble, user_nl_clm=None)
    with pytest.raises(FileNotFoundError, match="user_nl_clm"):
        _finalize(created_ensemble)


def test_finalize_leaves_no_user_nl_clm_tmp_file(created_ensemble):
    """Test that the atomic write doesn't leave its temp file behind"""
    case_root = _write_namelist_case(created_ensemble, "a = 1\n")
    _finalize(created_ensemble)
    assert not (case_root / "user_nl_clm.tmp").exists()


# ---------------------------------------------------------------------------
# finalize: check parameter file
# ---------------------------------------------------------------------------

# a fake preview_namelists that leaves CaseDocs/lnd_in as written
_KEEP_LND_IN = "exit 0\n"


@pytest.mark.parametrize(
    "line",
    [
        " fates_paramfile = '{}'",
        " FATES_PARAMFILE='{}'",
        ' fates_paramfile = "{}"',
    ],
)
def test_finalize_check_passes(created_ensemble, line):
    """Test that lnd_in setting the key to the parameter file passes,
    regardless of case, whitespace, or quote style
    """
    lnd_in = f" a = 1\n{line.format(_param_file(created_ensemble))}\n b = 2\n"
    _write_namelist_case(created_ensemble, lnd_in=lnd_in, body=_KEEP_LND_IN)
    _finalize(created_ensemble)


def test_finalize_check_passes_through_symlink(created_ensemble, tmp_path):
    """Test that lnd_in naming the parameter file through a symlinked
    directory passes, since it's the same file
    """
    param_file = Path(_param_file(created_ensemble))
    link = tmp_path / "link"
    link.symlink_to(param_file.parent)
    _write_namelist_case(
        created_ensemble,
        lnd_in=f" fates_paramfile = '{link / param_file.name}'\n",
        body=_KEEP_LND_IN,
    )
    _finalize(created_ensemble)


@pytest.mark.parametrize("spelling", ["{dir}//{name}", "{dir}/./{name}"])
def test_finalize_check_passes_other_spelling(created_ensemble, spelling):
    """Test that lnd_in naming the parameter file with a different but
    equivalent spelling passes
    """
    param_file = Path(_param_file(created_ensemble))
    value = spelling.format(dir=param_file.parent, name=param_file.name)
    _write_namelist_case(
        created_ensemble, lnd_in=f" fates_paramfile = '{value}'\n", body=_KEEP_LND_IN
    )
    _finalize(created_ensemble)


def test_finalize_check_wrong_value_raises(created_ensemble):
    """Test that lnd_in setting the key to a different file raises, naming both"""
    _write_namelist_case(
        created_ensemble, lnd_in=" fates_paramfile = '/p/b.nc'\n", body=_KEEP_LND_IN
    )
    expected = f"['/p/b.nc'], expected '{_param_file(created_ensemble)}'"
    with pytest.raises(RuntimeError, match=re.escape(expected)):
        _finalize(created_ensemble)


@pytest.mark.parametrize(
    "lnd_in",
    [
        "",
        " paramfile = '{}'\n",
        "! fates_paramfile = '{}'\n",
        " foo = '{}'\n",
    ],
    ids=["empty", "other_key", "comment", "path_only"],
)
def test_finalize_check_key_missing_raises(created_ensemble, lnd_in):
    """Test that the path appearing anywhere but as the key's value raises"""
    _write_namelist_case(
        created_ensemble,
        lnd_in=lnd_in.format(_param_file(created_ensemble)),
        body=_KEEP_LND_IN,
    )
    with pytest.raises(RuntimeError, match="fates_paramfile is not set in"):
        _finalize(created_ensemble)


def test_finalize_check_key_not_matched_by_suffix(ready_config_dict):
    """Test that paramfile_key paramfile isn't satisfied by fates_paramfile"""
    ready_config_dict["paramfile_key"] = "paramfile"
    ensemble = Ensemble.create(ready_config_dict)
    _write_namelist_case(
        ensemble,
        lnd_in=f" fates_paramfile = '{_param_file(ensemble)}'\n",
        body=_KEEP_LND_IN,
    )
    with pytest.raises(RuntimeError, match="paramfile is not set in"):
        _finalize(ensemble)


def test_finalize_check_duplicates_raise(created_ensemble):
    """Test that more than one assignment in lnd_in raises, even if one is right"""
    param_file = _param_file(created_ensemble)
    _write_namelist_case(
        created_ensemble,
        lnd_in=f" fates_paramfile = '{param_file}'\n fates_paramfile = '/p/b.nc'\n",
        body=_KEEP_LND_IN,
    )
    with pytest.raises(RuntimeError, match=re.escape(f"['{param_file}', '/p/b.nc']")):
        _finalize(created_ensemble)


def test_finalize_missing_lnd_in_raises(created_ensemble):
    """Test that a case with no CaseDocs/lnd_in raises"""
    _write_namelist_case(created_ensemble, body=_KEEP_LND_IN)
    with pytest.raises(FileNotFoundError, match="lnd_in"):
        _finalize(created_ensemble)


def test_finalize_no_preview_namelists_raises(created_ensemble):
    """Test that a case with no preview_namelists raises"""
    case_root = _write_namelist_case(created_ensemble)
    (case_root / "preview_namelists").unlink()
    with pytest.raises(FileNotFoundError, match="has no preview_namelists"):
        _finalize(created_ensemble)


def test_finalize_preview_namelists_not_executable_raises(created_ensemble):
    """Test that a non-executable preview_namelists raises"""
    _write_namelist_case(created_ensemble, mode=0o644)
    with pytest.raises(PermissionError, match="preview_namelists is not executable"):
        _finalize(created_ensemble)


def test_finalize_preview_namelists_failure_raises(created_ensemble):
    """Test that a failed preview_namelists raises with its stderr"""
    _write_namelist_case(
        created_ensemble, body="echo 'namelist exploded' >&2\nexit 1\n"
    )
    with pytest.raises(
        RuntimeError, match="preview_namelists failed: .*namelist exploded"
    ):
        _finalize(created_ensemble)
