"""Tests for the Ensemble class"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from meristem.ensemble.config import EnsembleConfig
from meristem.ensemble.model_ensemble import Ensemble

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


def test_load_missing_state_file_raises(tmp_path):
    """Test that load raises FileNotFoundError when there's no state file"""
    with pytest.raises(FileNotFoundError, match="Ensemble.create"):
        Ensemble.load(tmp_path)


def test_save_writes_config_under_config_key(created_ensemble):
    """Test that the state file holds the config's to_dict under "config" """
    state = json.loads((created_ensemble.root / _STATE_FILE).read_text("utf-8"))
    assert state == {"config": created_ensemble.config.to_dict()}


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
