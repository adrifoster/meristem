"""Fixtures shared across meristem.ensemble tests."""

from __future__ import annotations

from pathlib import Path

import pytest

# members written by write_paramgen by default, as paramgen names them
MEMBERS = ["ens_0000", "ens_0001", "ens_0002"]

@pytest.fixture
def members():
    """ensemble members fixture"""
    return MEMBERS

@pytest.fixture
def member_stage_dict() -> dict:
    """Minimal member stage dict (no script, so the member stage only runs
    meristem ensemble prepare, and no script file is needed)
    """
    return {"name": "spinup_ad", "queue": "regular"}


@pytest.fixture
def ensemble_config_dict(tmp_path, member_stage_dict) -> dict:
    """Smallest valid ensemble config dict (all required keys, no optional ones)"""
    return {
        "ensemble_id": "my_ens-1",
        "root": str(tmp_path / "root"),
        "paramgen": {"dir": str(tmp_path / "paramgen"), "prefix": "ens"},
        "base_case": str(tmp_path / "base_case"),
        "case_dir": str(tmp_path / "cases"),
        "paramfile_key": "fates_paramfile",
        "member_stages": [member_stage_dict],
    }


@pytest.fixture
def write_paramgen(ensemble_config_dict):
    """Factory that writes paramgen output (member list, key csv, parameter
    files) into the config's paramgen dir, the way meristem paramgen lays it out

    Args (of the returned function):
        members: names written to <prefix>.txt
        key_members: names in the key csv's ensemble column (default: members)
        param_files: names to write a parameter file for (default: members)
        ext: parameter file extension
    """
    paramgen = ensemble_config_dict["paramgen"]
    directory, prefix = Path(paramgen["dir"]), paramgen["prefix"]

    def _write(
        members=tuple(MEMBERS), key_members=None, param_files=None, ext=".nc"
    ) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{prefix}.txt").write_text(
            "".join(f"{m}\n" for m in members), encoding="utf-8"
        )
        key_rows = [
            f"{m},{0.1 * i:.1f}"
            for i, m in enumerate(members if key_members is None else key_members)
        ]
        (directory / f"{prefix}_key.csv").write_text(
            "ensemble,fates_leaf_slatop\n" + "\n".join(key_rows) + "\n",
            encoding="utf-8",
        )
        for member in members if param_files is None else param_files:
            (directory / f"{member}{ext}").touch()
        return directory

    return _write


@pytest.fixture
def write_base_case(ensemble_config_dict):
    """Factory that makes the config's base case dir with an executable
    xmlquery script that prints build_complete (and stderr, to stderr) and exits
    with exit_code
    """
    base_case = Path(ensemble_config_dict["base_case"])

    def _write(build_complete="TRUE", exit_code=0, stderr="") -> Path:
        base_case.mkdir(parents=True, exist_ok=True)
        xmlquery = base_case / "xmlquery"
        xmlquery.write_text(
            f"#!/bin/sh\necho {build_complete}\necho '{stderr}' >&2\nexit {exit_code}\n",
            encoding="utf-8",
        )
        xmlquery.chmod(0o755)
        return base_case

    return _write


@pytest.fixture
def ready_config_dict(ensemble_config_dict, write_paramgen, write_base_case) -> dict:
    """ensemble_config_dict with paramgen output and a built base case on disk,
    so Ensemble.create succeeds

    Sets project, since the member ModelRuns need one and $PROJECT may be unset
    """
    write_paramgen()
    write_base_case()
    ensemble_config_dict["project"] = "PROJ123"
    return ensemble_config_dict
