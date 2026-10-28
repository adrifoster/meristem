"""Fixtures for ensemble e2e tests"""

from __future__ import annotations

import subprocess
from pathlib import Path
import yaml
import json

import pytest

from tests.fakes.cime import CASE_DICT_FILE


@pytest.fixture
def fake_base_case(fake, ensemble_config_dict) -> Path:
    """Runs fake_create_newcase with env=fake.env, check=True"""

    base_case = Path(ensemble_config_dict["base_case"])
    subprocess.run(
        ["fake_create_newcase", str(base_case)],
        env=fake.env,
        check=True,
    )
    case_json_file = base_case / CASE_DICT_FILE
    case_json = json.loads(case_json_file.read_text(encoding="utf-8"))
    case_json["BUILD_COMPLETE"] = "TRUE"
    case_json["CIMEROOT"] = str(fake.cimeroot)
    case_json_file.write_text(json.dumps(case_json, indent=2), encoding="utf-8")
    return base_case


@pytest.fixture
def make_ensemble(
    tmp_path: Path, ensemble_config_dict, write_paramgen, fake_base_case, meristem
):
    """Factory that writes paramgen output and an ensemble YAML, runs
    `meristem ensemble create`, and returns the ensemble root

    Args:
        member_stages: stage dicts (default: ensemble_config_dict's)
        members: members written by write_paramgen (default: all of MEMBERS)
    """

    def _make_ensemble(member_stages=None, members=None, **cfg) -> Path:

        if members is None:
            write_paramgen()
        else:
            write_paramgen(members=members)

        config = {
            **ensemble_config_dict,
            "base_case": str(fake_base_case),
            "project": "FAKEPROJ",
        }
        if member_stages is not None:
            config["member_stages"] = member_stages
        config.update(cfg)

        file = tmp_path / "ensemble.yaml"
        file.write_text(yaml.safe_dump(config), encoding="utf-8")

        result = meristem("ensemble", "create", "--config", str(file))
        assert result.returncode == 0, result.stderr
        return Path(config["root"])

    return _make_ensemble


@pytest.fixture
def run_ensemble(fake, meristem, make_ensemble):
    """Fixture that runs a factory ensemble"""

    def _run_ensemble(member_stages=None, members=None) -> Path:
        root = make_ensemble(member_stages=member_stages, members=members)
        result = meristem("ensemble", "submit", "--root", str(root))
        assert result.returncode == 0, result.stderr
        fake.run_until_idle()

        return root

    return _run_ensemble
