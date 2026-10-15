"""Test that generated scripts are valid bash scripts"""

import shlex
import subprocess
import pytest
import time
from pathlib import Path

from meristem.run.model_run import ModelRun
from meristem.run.stages import StageStatus

SCRIPTS = Path(__file__).parent.parent / "smoke" / "scripts"


@pytest.mark.parametrize("script", sorted(SCRIPTS.glob("*.sh")))
def test_smoke_script_syntax(script):
    """Test that smoke test scripts have valid bash syntax"""
    result = subprocess.run(
        ["bash", "-n", str(script)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, f"Syntax error in {script.name}:\n{result.stderr}"


def test_submit_script_passes_bash_validation(created_run):
    """Test that generated PBS submit script passes bash -n validation"""
    created_run.submit()
    job_file = created_run.root / f"{created_run.run_id}_spinup_ad.pbs"
    result = subprocess.run(
        ["bash", "-n", str(job_file)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, f"Syntax error in {job_file.name}:\n{result.stderr}"


def test_submit_advance_script_passes_bash_validation(created_run):
    """Test that ModelRun.submit_advance() script passes bash -n validation"""
    created_run.stages[0].state.status = StageStatus.SUBMITTED
    created_run.stages[0].state.submit_time = time.time() - 100
    created_run.submit_advance("spinup_ad", "55555.deched")
    job_file = created_run.root / f"{created_run.run_id}_spinup_ad_advance.pbs"
    result = subprocess.run(
        ["bash", "-n", str(job_file)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, f"Syntax error in {job_file.name}:\n{result.stderr}"


@pytest.fixture
def awkward_run(tmp_path, ad_stage_config_dict, fake_scheduler):
    """A two-stage run whose second-stage script and case paths need shell quoting.
    The second stage's script records the arguments it receives in args.txt
    """
    awkward_dir = tmp_path / "it's a $dir; with spaces"
    awkward_dir.mkdir()
    args_file = tmp_path / "args.txt"
    script = awkward_dir / "record args.sh"
    script.write_text(f"printf '%s\\n' \"$@\" > {shlex.quote(str(args_file))}\n")
    config = {
        "root": str(tmp_path / "myrun"),
        "run_id": "member_0001",
        "project": "PROJ123",
        "user": "name@ucar.edu",
        "stages": [
            {**ad_stage_config_dict, "case_root": str(awkward_dir / "ad case")},
            {
                **ad_stage_config_dict,
                "name": "spinup_sasu",
                "script": str(script),
                "case_root": str(awkward_dir / "sasu case"),
            },
        ],
    }
    run = ModelRun.create(config, scheduler=fake_scheduler)
    run.submit()
    run.advance("spinup_ad")
    run.submit_advance("spinup_sasu", "55555.deched")
    run.submit_fail("spinup_sasu", "55555.deched")
    return run


@pytest.mark.parametrize("suffix", ["", "_advance", "_fail"])
def test_awkward_path_scripts_pass_bash_validation(awkward_run, suffix):
    """Test that job scripts with paths needing quoting pass bash -n validation"""
    job_file = awkward_run.root / f"{awkward_run.run_id}_spinup_sasu{suffix}.pbs"
    result = subprocess.run(
        ["bash", "-n", str(job_file)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, f"Syntax error in {job_file.name}:\n{result.stderr}"


def test_awkward_path_setup_script_receives_intact_args(awkward_run, tmp_path):
    """Run the generated setup script call and check the arguments it receives"""
    job_file = awkward_run.root / f"{awkward_run.run_id}_spinup_sasu.pbs"
    setup_line = next(
        line for line in job_file.read_text().splitlines() if line.startswith("bash -c")
    )
    subprocess.run(["bash", "-c", setup_line], check=True)

    ad_case_root, sasu_case_root = (
        stage.config.case_root for stage in awkward_run.stages
    )
    received = (tmp_path / "args.txt").read_text().splitlines()
    assert received == [str(sasu_case_root), ad_case_root.name, str(ad_case_root)]
