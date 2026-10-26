"""Fixtures for meristem run e2e tests"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml
import pytest

from tests.fakes.fake_pbs import FakePbs


@pytest.fixture
def fake(tmp_path):
    """Fake PBS fixture"""
    return FakePbs(tmp_path / "pbs")


@pytest.fixture
def meristem(fake):
    """Fake meristem CLI"""

    def _meristem(*args):
        """Runs the meristem CLI a way a user would, but with the fake environment"""
        return subprocess.run(
            [sys.executable, "-m", "meristem", *args],
            env=fake.env,
            capture_output=True,
            text=True,
            check=False,
        )

    return _meristem


@pytest.fixture
def stage_script(tmp_path):
    """stage_script factory"""

    def _stage_script(body: str = 'fake_create_newcase "$1"', name="stage") -> Path:
        """Create a fake script for a stage

        Args:
            body (str, optional): body of script. Defaults to 'fake_create_newcase "_summary_"'.
            name (str, optional): name of file. Defaults to "stage".

        Returns:
            Path: path to script
        """
        script = tmp_path / f"{name}.sh"
        script.write_text(f"#!/bin/bash\n{body}\n")
        return script

    return _stage_script


@pytest.fixture
def make_run(tmp_path: Path, stage_script, meristem):
    """Makes a ModelRun instance

    Args:
        tmp_path (Path): root dir
        stage_script: generates path to script
        meristem: meristem CLI
    """

    def _make_run(stages, **cfg) -> Path:
        config = {
            "root": str(tmp_path / "run_root"),
            "run_id": "e2e_test",
            "project": "FAKEPROJ",
            "stages": [],
        }
        for stage in stages:
            stage_dict = {"name": stage["name"], "queue": "develop"}
            script = stage.get("script")
            if script:
                stage_dict["script"] = str(script)
            else:
                stage_dict["script"] = str(stage_script(name=stage["name"]))
            if stage.get("no_cime"):
                stage_dict["no_cime"] = True

            config["stages"].append(stage_dict)

        config.update(cfg)

        file = tmp_path / "config.yaml"
        file.write_text(yaml.safe_dump(config), encoding="utf-8")

        result = meristem("run", "create", "--config", str(file))

        assert result.returncode == 0, result.stderr
        return Path(config["root"])

    return _make_run
