"""Tests for EnsembleConfig"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import yaml

from meristem.ensemble.config import EnsembleConfig
from meristem.run.stages import StageConfig


def _stage(name: str = "spinup_ad", **kwargs) -> StageConfig:
    """A minimal member StageConfig"""
    return StageConfig.from_dict({"name": name, "queue": "regular", **kwargs})


def _config(tmp_path, **overrides) -> EnsembleConfig:
    """EnsembleConfig built directly, with overrides for any field"""
    fields = {
        "ensemble_id": "ens",
        "root": tmp_path / "root",
        "base_case": tmp_path / "base_case",
        "case_dir": tmp_path / "cases",
        "paramgen_dir": tmp_path / "paramgen",
        "paramgen_prefix": "ens",
        "paramfile_key": "paramfile",
        "member_stages": (_stage(),),
    }
    fields.update(overrides)
    return EnsembleConfig(**fields)


# ---------------------------------------------------------------------------
# __post_init__
# ---------------------------------------------------------------------------


def test_init_defaults(tmp_path):
    """Test that optional fields default to None / empty"""
    config = _config(tmp_path)
    assert config.members is None
    assert config.output_root is None
    assert config.project == ""
    assert config.email is None


def test_from_dict_case_dir_optional(ensemble_config_dict):
    """Test that case_dir may be omitted from the config"""
    del ensemble_config_dict["case_dir"]
    config = EnsembleConfig.from_dict(ensemble_config_dict)
    assert config.case_dir == Path(ensemble_config_dict["root"]) / "cases"


def test_from_dict_blank_case_dir_defaults_to_root_cases(ensemble_config_dict):
    """Test that a blank "case_dir:" in the YAML (None) defaults to <root>/cases"""
    ensemble_config_dict["case_dir"] = None
    config = EnsembleConfig.from_dict(ensemble_config_dict)
    assert config.case_dir == Path(ensemble_config_dict["root"]) / "cases"


def test_init_converts_paths(tmp_path):
    """Test that path fields given as strings are converted to Path"""
    config = _config(
        tmp_path,
        root=str(tmp_path / "root"),
        base_case=str(tmp_path / "base_case"),
        case_dir=str(tmp_path / "cases"),
        paramgen_dir=str(tmp_path / "paramgen"),
        output_root=str(tmp_path / "out"),
    )
    assert config.root == tmp_path / "root"
    assert config.base_case == tmp_path / "base_case"
    assert config.case_dir == tmp_path / "cases"
    assert config.paramgen_dir == tmp_path / "paramgen"
    assert config.output_root == tmp_path / "out"
    for name in ("root", "base_case", "case_dir", "paramgen_dir", "output_root"):
        assert isinstance(getattr(config, name), Path)


@pytest.mark.parametrize(
    "name", ["root", "base_case", "case_dir", "paramgen_dir", "output_root"]
)
def test_init_relative_path_raises(tmp_path, name):
    """Test that every path field must be absolute (they're used in PBS jobs)"""
    with pytest.raises(
        ValueError, match=rf"^{name} 'relative/dir' must be an absolute path\.$"
    ):
        _config(tmp_path, **{name: "relative/dir"})


def test_from_dict_relative_member_stage_script_raises(ensemble_config_dict):
    """Test that a relative member stage script is rejected when the config is read"""
    ensemble_config_dict["member_stages"][0]["script"] = "rel_script.sh"
    with pytest.raises(
        ValueError, match="script 'rel_script.sh' for stage .* must be an absolute path"
    ):
        EnsembleConfig.from_dict(ensemble_config_dict)


def test_init_output_root_none_allowed(tmp_path):
    """Test that an unset output_root isn't checked for being absolute"""
    assert _config(tmp_path, output_root=None).output_root is None


def test_init_converts_sequences_to_tuples(tmp_path):
    """Test that member_stages and members given as lists are stored as tuples"""
    config = _config(tmp_path, member_stages=[_stage()], members=["m1", "m2"])
    assert config.member_stages == (_stage(),)
    assert config.members == ("m1", "m2")


def test_init_is_frozen(tmp_path):
    """Test that EnsembleConfig can't be modified after construction"""
    config = _config(tmp_path)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.ensemble_id = "other"  # type: ignore[misc]


@pytest.mark.parametrize("ensemble_id", ["ens", "ENS_01", "my-ens", "a"])
def test_init_valid_ensemble_id(tmp_path, ensemble_id):
    """Test that ensemble_ids of letters, digits, hyphens, underscores are allowed"""
    assert _config(tmp_path, ensemble_id=ensemble_id).ensemble_id == ensemble_id


@pytest.mark.parametrize("ensemble_id", ["my ens", "ens/1", "ens.1", "ens$"])
def test_init_invalid_ensemble_id_raises(tmp_path, ensemble_id):
    """Test that an ensemble_id with other characters raises"""
    with pytest.raises(ValueError, match="invalid characters"):
        _config(tmp_path, ensemble_id=ensemble_id)


def test_init_empty_ensemble_id_raises(tmp_path):
    """Test that an empty ensemble_id says it's empty"""
    with pytest.raises(ValueError, match=r"^ensemble_id must not be empty\.$"):
        _config(tmp_path, ensemble_id="")


@pytest.mark.parametrize("ensemble_id", [42, 1.5, None, ["ens"]])
def test_init_non_string_ensemble_id_raises(tmp_path, ensemble_id):
    """Test that a non-string ensemble_id (e.g. an unquoted number in the YAML)
    raises a ValueError instead of a TypeError from the regex
    """
    with pytest.raises(ValueError, match="ensemble_id must be a string"):
        _config(tmp_path, ensemble_id=ensemble_id)


def test_init_empty_paramgen_prefix_raises(tmp_path):
    """Test that an empty paramgen prefix raises"""
    with pytest.raises(ValueError, match="prefix must not be empty"):
        _config(tmp_path, paramgen_prefix="")


@pytest.mark.parametrize("key", ["paramfile", "fates_paramfile"])
def test_init_valid_paramfile_key(tmp_path, key):
    """Test that both supported paramfile keys are accepted"""
    assert _config(tmp_path, paramfile_key=key).paramfile_key == key


def test_init_invalid_paramfile_key_raises(tmp_path):
    """Test that any other paramfile_key raises"""
    with pytest.raises(ValueError, match="paramfile_key must be one of"):
        _config(tmp_path, paramfile_key="clm_paramfile")


def test_init_empty_member_stages_raises(tmp_path):
    """Test that an ensemble with no member stages raises"""
    with pytest.raises(ValueError, match="member_stages must not be empty"):
        _config(tmp_path, member_stages=())


def test_init_duplicate_stage_names_raises(tmp_path):
    """Test that two member stages with the same name raise"""
    with pytest.raises(ValueError, match="must be unique"):
        _config(tmp_path, member_stages=(_stage("ad"), _stage("ad")))


def test_init_multiple_stages_keep_order(tmp_path):
    """Test that multiple uniquely named member stages are kept in order"""
    config = _config(tmp_path, member_stages=[_stage("ad"), _stage("postad")])
    assert [stage.name for stage in config.member_stages] == ["ad", "postad"]


@pytest.mark.parametrize("script", [None, "/path/to/setup.sh"])
def test_init_command_stage_raises(tmp_path, script):
    """Test that a member stage can't set a command (with or without a script),
    since the ensemble sets it
    """
    stage = _stage("ad", command="echo hi", script=script)
    with pytest.raises(ValueError, match="member stage 'ad': command is not allowed"):
        _config(tmp_path, member_stages=(stage,))


def test_init_no_cime_stage_raises(tmp_path):
    """Test that a no_cime member stage raises"""
    with pytest.raises(ValueError, match="no_cime is not allowed"):
        _config(tmp_path, member_stages=(_stage(no_cime=True),))


def test_init_case_root_stage_raises(tmp_path):
    """Test that a member stage can't set case_root, since the ensemble puts
    each member's case under case_dir
    """
    stage = _stage("ad", case_root="/path/to/case")
    with pytest.raises(
        ValueError, match="member stage 'ad': case_root is not allowed; .* case_dir"
    ):
        _config(tmp_path, member_stages=(stage,))


def test_init_post_command_stage_raises(tmp_path):
    """Test that a member stage can't set a post_command, since the ensemble sets it"""
    stage = _stage("ad", post_command="echo hi")
    with pytest.raises(
        ValueError,
        match="member stage 'ad': post_command is not allowed; .* finalize",
    ):
        _config(tmp_path, member_stages=(stage,))


def test_init_empty_members_raises(tmp_path):
    """Test that an empty members list raises (None means all members)"""
    with pytest.raises(ValueError, match="non-empty list"):
        _config(tmp_path, members=())


@pytest.mark.parametrize(
    "members", [(1, 2), ("m1", 2), ("m1", None), ("m1", ""), (["m1"],)]
)
def test_init_non_string_members_raises(tmp_path, members):
    """Test that member names that aren't non-empty strings raise"""
    with pytest.raises(ValueError, match="members must be non-empty strings"):
        _config(tmp_path, members=members)


def test_init_duplicate_members_raises(tmp_path):
    """Test that duplicate member names raise, listing each duplicate once"""
    with pytest.raises(ValueError, match=r"duplicates \['a', 'b'\]"):
        _config(tmp_path, members=("b", "a", "c", "a", "b"))


# ---------------------------------------------------------------------------
# from_dict
# ---------------------------------------------------------------------------


def test_from_dict_minimal(ensemble_config_dict, member_stage_dict, tmp_path):
    """Test that a dict with only the required keys builds the expected config"""
    config = EnsembleConfig.from_dict(ensemble_config_dict)
    assert config == EnsembleConfig(
        ensemble_id="my_ens-1",
        root=tmp_path / "root",
        base_case=tmp_path / "base_case",
        case_dir=tmp_path / "cases",
        paramgen_dir=tmp_path / "paramgen",
        paramgen_prefix="ens",
        paramfile_key="fates_paramfile",
        member_stages=(StageConfig.from_dict(member_stage_dict),),
    )


def test_from_dict_optional_keys(ensemble_config_dict, tmp_path):
    """Test that the optional top-level keys are read"""
    ensemble_config_dict.update(
        output_root=str(tmp_path / "out"),
        project="P12345678",
        email="me@example.com",
    )
    config = EnsembleConfig.from_dict(ensemble_config_dict)
    assert config.output_root == tmp_path / "out"
    assert config.project == "P12345678"
    assert config.email == "me@example.com"


def test_from_dict_blank_project_is_empty_string(ensemble_config_dict):
    """Test that a blank "project:" in the YAML (None) becomes "" """
    ensemble_config_dict["project"] = None
    assert EnsembleConfig.from_dict(ensemble_config_dict).project == ""


def test_from_dict_non_string_ensemble_id_raises(ensemble_config_dict):
    """Test that an unquoted numeric ensemble_id raises a ValueError"""
    ensemble_config_dict["ensemble_id"] = 42
    with pytest.raises(ValueError, match="ensemble_id must be a string"):
        EnsembleConfig.from_dict(ensemble_config_dict)


def test_from_dict_does_not_modify_input(ensemble_config_dict):
    """Test that from_dict leaves the input dict unchanged"""
    before = yaml.safe_dump(ensemble_config_dict)
    EnsembleConfig.from_dict(ensemble_config_dict)
    assert yaml.safe_dump(ensemble_config_dict) == before


@pytest.mark.parametrize(
    "key",
    [
        "ensemble_id",
        "root",
        "paramgen",
        "base_case",
        "paramfile_key",
        "member_stages",
    ],
)
def test_from_dict_missing_required_key_raises(ensemble_config_dict, key):
    """Test that each required key is required"""
    del ensemble_config_dict[key]
    with pytest.raises(ValueError, match=f"missing required key {key}"):
        EnsembleConfig.from_dict(ensemble_config_dict)


@pytest.mark.parametrize("key", ["ensemble_id", "root", "member_stages", "paramgen"])
def test_from_dict_null_required_key_raises(ensemble_config_dict, key):
    """Test that a required key that's null (e.g. a blank "root:") counts as missing"""
    ensemble_config_dict[key] = None
    with pytest.raises(ValueError, match=f"missing required key {key}"):
        EnsembleConfig.from_dict(ensemble_config_dict)


@pytest.mark.parametrize(
    "key, empty, match",
    [
        ("ensemble_id", "", r"^ensemble_id must not be empty\.$"),
        ("ensemble_id", [], "ensemble_id must be a string"),
        ("root", "", r"^root must not be empty\.$"),
        ("root", [], r"^root must be a path, got list \[\]\.$"),
        ("root", {}, r"^root must be a path, got dict \{\}\.$"),
        ("member_stages", [], r"^member_stages must not be empty\.$"),
        ("member_stages", "", r"^member_stages must be a list\.$"),
        ("member_stages", {}, r"^member_stages must be a list\.$"),
        ("paramgen", "", "paramgen must be a mapping"),
        ("paramgen", [], "paramgen must be a mapping"),
        ("paramgen", {}, "missing required key (dir|prefix) .*paramgen"),
    ],
)
def test_from_dict_empty_required_key_raises(ensemble_config_dict, key, empty, match):
    """Test that a required key present but empty says what's wrong with it,
    rather than that it's missing
    """
    ensemble_config_dict[key] = empty
    with pytest.raises(ValueError, match=match):
        EnsembleConfig.from_dict(ensemble_config_dict)


def test_from_dict_unknown_key_raises(ensemble_config_dict):
    """Test that unknown top-level keys raise, listing them sorted"""
    ensemble_config_dict.update(zeta=1, alpha=2)
    with pytest.raises(ValueError, match=r"unknown key\(s\) \['alpha', 'zeta'\]"):
        EnsembleConfig.from_dict(ensemble_config_dict)


@pytest.mark.parametrize("paramgen", ["some/dir", ["dir", "prefix"], 3])
def test_from_dict_paramgen_not_mapping_raises(ensemble_config_dict, paramgen):
    """Test that paramgen given as something other than a mapping raises"""
    ensemble_config_dict["paramgen"] = paramgen
    with pytest.raises(ValueError, match="paramgen must be a mapping"):
        EnsembleConfig.from_dict(ensemble_config_dict)


@pytest.mark.parametrize("key", ["dir", "prefix"])
def test_from_dict_missing_paramgen_key_raises(ensemble_config_dict, key):
    """Test that each required paramgen key is required"""
    del ensemble_config_dict["paramgen"][key]
    with pytest.raises(ValueError, match=f"missing required key {key}.*paramgen"):
        EnsembleConfig.from_dict(ensemble_config_dict)


@pytest.mark.parametrize("key", ["dir", "prefix"])
def test_from_dict_null_paramgen_key_raises(ensemble_config_dict, key):
    """Test that a required paramgen key that's null counts as missing"""
    ensemble_config_dict["paramgen"][key] = None
    with pytest.raises(ValueError, match=f"missing required key {key}.*paramgen"):
        EnsembleConfig.from_dict(ensemble_config_dict)


@pytest.mark.parametrize(
    "key, match",
    [
        ("dir", r"^paramgen_dir must not be empty\.$"),
        ("prefix", r"^paramgen prefix must not be empty\.$"),
    ],
)
def test_from_dict_empty_paramgen_key_raises(ensemble_config_dict, key, match):
    """Test that a required paramgen key present but empty says what's wrong"""
    ensemble_config_dict["paramgen"][key] = ""
    with pytest.raises(ValueError, match=match):
        EnsembleConfig.from_dict(ensemble_config_dict)


@pytest.mark.parametrize(
    "config, match",
    [
        (None, r"^ensemble config is empty\.$"),
        (["a", "b"], r"^ensemble config must be a mapping .*, got list\.$"),
        ("text", r"^ensemble config must be a mapping .*, got str\.$"),
    ],
)
def test_from_dict_not_a_mapping_raises(config, match):
    """Test that a config that isn't a mapping raises a ValueError"""
    with pytest.raises(ValueError, match=match):
        EnsembleConfig.from_dict(config)


def test_from_dict_unknown_paramgen_key_raises(ensemble_config_dict):
    """Test that unknown paramgen keys raise"""
    ensemble_config_dict["paramgen"]["seed"] = 42
    with pytest.raises(ValueError, match=r"unknown key\(s\) \['seed'\] in paramgen"):
        EnsembleConfig.from_dict(ensemble_config_dict)


def test_from_dict_members_default_is_all(ensemble_config_dict):
    """Test that leaving out paramgen members means all members (None)"""
    assert EnsembleConfig.from_dict(ensemble_config_dict).members is None


def test_from_dict_members_all(ensemble_config_dict):
    """Test that paramgen members: all means all members (None)"""
    ensemble_config_dict["paramgen"]["members"] = "all"
    assert EnsembleConfig.from_dict(ensemble_config_dict).members is None


def test_from_dict_members_list(ensemble_config_dict):
    """Test that a paramgen members list is stored as a tuple, in order"""
    ensemble_config_dict["paramgen"]["members"] = ["member_0002", "member_0001"]
    config = EnsembleConfig.from_dict(ensemble_config_dict)
    assert config.members == ("member_0002", "member_0001")


@pytest.mark.parametrize("members", ["some", "member_0001", 3, {"a": 1}])
def test_from_dict_members_not_all_or_list_raises(ensemble_config_dict, members):
    """Test that paramgen members other than 'all' or a list raise"""
    ensemble_config_dict["paramgen"]["members"] = members
    with pytest.raises(ValueError, match="'all' or a list"):
        EnsembleConfig.from_dict(ensemble_config_dict)


def test_from_dict_members_empty_list_raises(ensemble_config_dict):
    """Test that an empty paramgen members list raises"""
    ensemble_config_dict["paramgen"]["members"] = []
    with pytest.raises(ValueError, match="non-empty list"):
        EnsembleConfig.from_dict(ensemble_config_dict)


def test_from_dict_members_numeric_raises(ensemble_config_dict):
    """Test that unquoted numeric member names in the YAML raise"""
    ensemble_config_dict["paramgen"]["members"] = [1, 2]
    with pytest.raises(ValueError, match="members must be non-empty strings"):
        EnsembleConfig.from_dict(ensemble_config_dict)


def test_from_dict_members_duplicates_raise(ensemble_config_dict):
    """Test that duplicate paramgen member names raise"""
    ensemble_config_dict["paramgen"]["members"] = ["member_0001", "member_0001"]
    with pytest.raises(ValueError, match="members must be unique"):
        EnsembleConfig.from_dict(ensemble_config_dict)


def test_from_dict_member_stages_not_list_raises(
    ensemble_config_dict, member_stage_dict
):
    """Test that member_stages given as a single mapping (not a list) raises"""
    ensemble_config_dict["member_stages"] = member_stage_dict
    with pytest.raises(ValueError, match="member_stages must be a list"):
        EnsembleConfig.from_dict(ensemble_config_dict)


@pytest.mark.parametrize("entry", ["spinup_ad", ["spinup_ad"], None, 3])
def test_from_dict_member_stage_not_mapping_raises(
    ensemble_config_dict, member_stage_dict, entry
):
    """Test that a member_stages entry that isn't a mapping raises, naming its
    index
    """
    ensemble_config_dict["member_stages"] = [member_stage_dict, entry]
    with pytest.raises(ValueError, match=r"member_stages\[1\] must be a mapping"):
        EnsembleConfig.from_dict(ensemble_config_dict)


def test_from_dict_builds_stage_configs(ensemble_config_dict, member_stage_dict):
    """Test that each member stage dict becomes a StageConfig, in order"""
    second = {**member_stage_dict, "name": "postad", "walltime": "06:00:00"}
    ensemble_config_dict["member_stages"] = [member_stage_dict, second]
    config = EnsembleConfig.from_dict(ensemble_config_dict)
    assert config.member_stages == (
        StageConfig.from_dict(member_stage_dict),
        StageConfig.from_dict(second),
    )


def test_from_dict_invalid_stage_raises(ensemble_config_dict, member_stage_dict):
    """Test that a StageConfig validation error propagates"""
    member_stage_dict["ncpus"] = 0
    with pytest.raises(ValueError, match="ncpus must be >=1"):
        EnsembleConfig.from_dict(ensemble_config_dict)


def test_from_dict_unsafe_member_stage_name_raises(
    ensemble_config_dict, member_stage_dict
):
    """Test that a member stage name that isn't safe in job and case names is
    rejected
    """
    member_stage_dict["name"] = "spin up"
    with pytest.raises(ValueError, match="stage name 'spin up' contains invalid"):
        EnsembleConfig.from_dict(ensemble_config_dict)


def test_from_dict_stage_with_script(ensemble_config_dict, member_stage_dict):
    """Test that a member stage can have a script (it runs after prepare)"""
    member_stage_dict["script"] = "/path/to/setup.sh"
    config = EnsembleConfig.from_dict(ensemble_config_dict)
    assert config.member_stages[0].script == Path("/path/to/setup.sh")


def test_from_dict_runs_post_init_validation(ensemble_config_dict):
    """Test that __post_init__ validation still applies via from_dict"""
    ensemble_config_dict["paramfile_key"] = "bad"
    with pytest.raises(ValueError, match="paramfile_key must be one of"):
        EnsembleConfig.from_dict(ensemble_config_dict)


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


def test_to_dict_round_trip_minimal(ensemble_config_dict):
    """Test that from_dict(to_dict()) gives back an equal config (required keys)"""
    config = EnsembleConfig.from_dict(ensemble_config_dict)
    assert EnsembleConfig.from_dict(config.to_dict()) == config


def test_to_dict_round_trip_all_keys(ensemble_config_dict, member_stage_dict, tmp_path):
    """Test that from_dict(to_dict()) gives back an equal config (all keys set)"""
    ensemble_config_dict["paramgen"]["members"] = ["member_0002", "member_0001"]
    ensemble_config_dict["member_stages"].append(
        {**member_stage_dict, "name": "postad", "walltime": "06:00:00"}
    )
    ensemble_config_dict.update(
        output_root=str(tmp_path / "out"),
        project="P12345678",
        email="me@example.com",
    )
    config = EnsembleConfig.from_dict(ensemble_config_dict)
    assert EnsembleConfig.from_dict(config.to_dict()) == config


def test_to_dict_all_members_written_as_all(ensemble_config_dict):
    """Test that members=None (all members) is written as "all", not null"""
    config = EnsembleConfig.from_dict(ensemble_config_dict)
    assert config.to_dict()["paramgen"]["members"] == "all"


def test_to_dict_is_json_serializable(ensemble_config_dict, tmp_path):
    """Test that to_dict output (paths included) can be written as JSON"""
    ensemble_config_dict["output_root"] = str(tmp_path / "out")
    config = EnsembleConfig.from_dict(ensemble_config_dict)
    assert json.loads(json.dumps(config.to_dict())) == config.to_dict()


# ---------------------------------------------------------------------------
# from_yaml
# ---------------------------------------------------------------------------


def _write_yaml(path: Path, config: dict) -> Path:
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@pytest.mark.parametrize("as_str", [False, True])
def test_from_yaml_matches_from_dict(tmp_path, ensemble_config_dict, as_str):
    """Test that from_yaml (given a Path or str) gives the same config as from_dict"""
    path = _write_yaml(tmp_path / "ensemble.yaml", ensemble_config_dict)
    config = EnsembleConfig.from_yaml(str(path) if as_str else path)
    assert config == EnsembleConfig.from_dict(ensemble_config_dict)


def test_from_yaml_expands_env_vars(tmp_path, ensemble_config_dict, monkeypatch):
    """Test that $VARS and ${VARS} in the YAML are expanded from the environment"""
    monkeypatch.setenv("ENS_SCRATCH", str(tmp_path / "scratch"))
    monkeypatch.setenv("ENS_PROJECT", "P12345678")
    ensemble_config_dict["root"] = "$ENS_SCRATCH/root"
    ensemble_config_dict["project"] = "${ENS_PROJECT}"
    path = _write_yaml(tmp_path / "ensemble.yaml", ensemble_config_dict)

    config = EnsembleConfig.from_yaml(path)
    assert config.root == tmp_path / "scratch" / "root"
    assert config.project == "P12345678"


def test_from_yaml_unset_env_var_raises(tmp_path, ensemble_config_dict, monkeypatch):
    """Test that an unset $VAR is left as written (os.path.expandvars behavior),
    so the path is relative and raises, hinting at the unset variable
    """
    monkeypatch.delenv("ENS_UNSET_VAR", raising=False)
    ensemble_config_dict["root"] = "$ENS_UNSET_VAR/root"
    path = _write_yaml(tmp_path / "ensemble.yaml", ensemble_config_dict)
    with pytest.raises(
        ValueError,
        match=r"root '\$ENS_UNSET_VAR/root' must be an absolute path "
        r"\(is an environment variable unset\?\)",
    ):
        EnsembleConfig.from_yaml(path)


def test_from_yaml_validation_error_propagates(tmp_path, ensemble_config_dict):
    """Test that an invalid YAML config raises the from_dict error"""
    del ensemble_config_dict["base_case"]
    path = _write_yaml(tmp_path / "ensemble.yaml", ensemble_config_dict)
    with pytest.raises(ValueError, match="missing required key base_case"):
        EnsembleConfig.from_yaml(path)


@pytest.mark.parametrize("text", ["", "# only a comment\n"])
def test_from_yaml_empty_file_raises(tmp_path, text):
    """Test that an empty YAML file (or one with only comments) says it's empty"""
    path = tmp_path / "ensemble.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=r"^ensemble config is empty\.$"):
        EnsembleConfig.from_yaml(path)


def test_from_yaml_missing_file_raises(tmp_path):
    """Test that a nonexistent YAML path raises FileNotFoundError"""
    with pytest.raises(FileNotFoundError):
        EnsembleConfig.from_yaml(tmp_path / "missing.yaml")
