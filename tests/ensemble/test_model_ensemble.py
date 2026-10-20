"""Tests for the Ensemble class"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest
import yaml

from meristem.commands import meristem_cmd
from meristem.ensemble.config import EnsembleConfig
from meristem.ensemble.model_ensemble import Ensemble
from meristem.run.model_run import _STATE_FILE as _RUN_STATE_FILE, ModelRun
from meristem.run.stages import StageStatus

from .conftest import MEMBERS

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
    del ready_config_dict["case_dir"]
    with pytest.raises(ValueError, match="missing required key case_dir"):
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
        assert (
            stage.config.case_root == tmp_path / "cases" / f"base_case_{member}_{name}"
        )
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
