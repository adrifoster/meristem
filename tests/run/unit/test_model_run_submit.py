"""Tests for ModelRun.submit behavior"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import pytest

from meristem.commands import meristem_cmd
from meristem.run import model_run
from meristem.run.model_run import ModelRun
from meristem.run.stages import Stage, StageStatus
from jinja2 import TemplateNotFound


def test_submit_writes_jobscript(created_run):
    """Test that ModelRun.submit() actually writes a jobscript"""
    created_run.submit()
    job_file = created_run.root / f"{created_run.run_id}_spinup_ad.pbs"
    assert job_file.exists()


def test_submit_raises_if_template_missing(created_run, mocker):
    """Test that submit() raises if the PBS template file is missing"""
    mocker.patch(
        "meristem.run.model_run._ENV.get_template",
        side_effect=TemplateNotFound("setup_script_template.txt"),
    )
    with pytest.raises(TemplateNotFound):
        created_run.submit()


@pytest.mark.parametrize(
    "field,expected",
    [
        ("PBS -N ", "job_name"),
        ("#PBS -q ", "queue"),
        ("#PBS -l select=", "select"),
        ("ncpus=", "ncpus"),
        ("mem=", "memory"),
        ("#PBS -l walltime=", "walltime"),
        ("#PBS -A ", "project"),
        ("#PBS -M ", "email"),
    ],
)
def test_submit_script_contains_pbs_directives(created_run, field, expected):
    """Test that ModelRun.submit() creates a PBS file with the correct PBS directives"""
    created_run.submit()
    job_file = (created_run.root / f"{created_run.run_id}_spinup_ad.pbs").read_text()
    stage = created_run.stages[0]
    values = {
        "job_name": f"{created_run.run_id}_{stage.config.name}",
        "queue": stage.config.queue,
        "select": str(stage.config.select),
        "ncpus": str(stage.config.ncpus),
        "memory": str(stage.config.memory),
        "walltime": stage.config.walltime,
        "project": created_run.project,
        "email": created_run.email,
    }
    assert field + values[expected] in job_file


def test_submit_script_no_depend_line(created_run):
    """Test that the depend line is not written for the first ModelRun.submit"""
    created_run.submit()
    job_file = (created_run.root / f"{created_run.run_id}_spinup_ad.pbs").read_text()
    assert "#PBS -W depend=" not in job_file


def test_submit_script_trap_references_run_root(created_run):
    """Test that the ERR trap in the job script references the correct run root"""
    created_run.submit()
    content = (created_run.root / f"{created_run.run_id}_spinup_ad.pbs").read_text()
    assert f"RUN_ROOT={created_run.root}\n" in content
    assert 'trap \'meristem run --root "$RUN_ROOT"' in content


def test_submit_script_trap_references_stage_name(created_run):
    """Test that the ERR trap in the job script references the correct stage name"""
    created_run.submit()
    content = (created_run.root / f"{created_run.run_id}_spinup_ad.pbs").read_text()
    assert f"STAGE={created_run.stages[0].config.name}\n" in content
    assert '--fail --stage "$STAGE"\' ERR' in content


def test_submit_script_correct_script(created_run):
    """Test that ModelRun.submit() creates a PBS file with the script"""
    created_run.submit()
    job_file = (created_run.root / f"{created_run.run_id}_spinup_ad.pbs").read_text()
    assert str(created_run.stages[0].config.script) in job_file


def test_submit_script_cd_into_case_root(created_run):
    """Test that ModelRun.submit() contains a cd into the correct case_root"""
    created_run.submit()
    job_file = (created_run.root / f"{created_run.run_id}_spinup_ad.pbs").read_text()
    case_root = created_run.root / created_run.stages[0].config.name
    assert f"cd {case_root}" in job_file


def test_submit_script_uses_configured_case_root(
    tmp_path, run_config_dict, fake_scheduler
):
    """Test that a stage's case_root overrides the default <root>/<name>"""
    case_root = tmp_path / "elsewhere" / "my_case"
    run_config_dict["stages"][0]["case_root"] = str(case_root)
    run = ModelRun.create(run_config_dict, scheduler=fake_scheduler)
    run.submit()
    job_file = (run.root / f"{run.run_id}_spinup_ad.pbs").read_text()
    assert f"{run.stages[0].config.script} {case_root}" in job_file
    assert f"cd {case_root}" in job_file


def test_shquote_filter_rejects_none():
    """Test that the shquote template filter raises on None instead of
    rendering the literal word 'None'
    """
    # pylint: disable=protected-access
    with pytest.raises(ValueError, match="None"):
        model_run._ENV.filters["shquote"](None)


@pytest.mark.parametrize(
    "value, expected",
    [("plain", "plain"), ("has space", "'has space'"), (Path("/a b"), "'/a b'")],
)
def test_shquote_filter_quotes_values(value, expected):
    """Test that the shquote template filter quotes strings and paths"""
    # pylint: disable=protected-access
    assert model_run._ENV.filters["shquote"](value) == expected


def _run_with_stages(root: Path, stage_dicts: list[dict], scheduler) -> ModelRun:
    """A ModelRun built directly (as Ensemble does) and saved, since run configs
    can't set a stage command
    """
    run = ModelRun(
        root=root,
        run_id="member_0001",
        stages=[Stage.from_dict(stage) for stage in stage_dicts],
        project="PROJ123",
        scheduler=scheduler,
    )
    run.root.mkdir(parents=True)
    run.save()
    return run


@pytest.fixture
def command_run(tmp_path, command_stage_config_dict, fake_scheduler) -> ModelRun:
    """A ModelRun whose single stage runs a command instead of a script"""
    return _run_with_stages(
        tmp_path / "myrun", [command_stage_config_dict], fake_scheduler
    )


def _command_call(job_file: Path) -> list[str]:
    """The bash -c invocation of a stage's command in a generated job script"""
    for line in job_file.read_text().splitlines():
        if line.startswith("bash -c"):
            return shlex.split(line)
    raise AssertionError(f"no command call in {job_file}")


def test_submit_command_stage_runs_command_in_strict_mode(command_run):
    """Test that a command stage runs its command under bash strict mode as a
    single bash -c argument
    """
    command_run.submit()
    job_file = command_run.root / f"{command_run.run_id}_spinup_ad.pbs"
    assert _command_call(job_file) == [
        "bash",
        "-c",
        "set -euo pipefail; meristem ensemble prepare --member 1",
    ]


def test_submit_command_stage_has_no_setup_script_call(command_run):
    """Test that a command stage neither sources a setup script nor renders 'None'"""
    command_run.submit()
    job_file = (command_run.root / f"{command_run.run_id}_spinup_ad.pbs").read_text()
    assert 'source "$0"' not in job_file
    assert "None" not in job_file


def test_submit_command_stage_survives_reload(command_run, fake_scheduler):
    """Test that a command stage still renders its command after state is
    saved to disk and reloaded (e.g. when a later stage is advanced)
    """
    reloaded = ModelRun.load(command_run.root, scheduler=fake_scheduler)
    assert reloaded.stages[0].config.script is None
    reloaded.submit()
    job_file = reloaded.root / f"{reloaded.run_id}_spinup_ad.pbs"
    assert _command_call(job_file)[-1].endswith("meristem ensemble prepare --member 1")
    assert 'source "$0"' not in job_file.read_text()


def _job_scripts(run: ModelRun) -> dict[str, str]:
    """Render the submit, advance, and fail job scripts for a run's first stage"""
    # pylint: disable=protected-access
    stage = run.stages[0]
    return {
        "submit": run._write_job_script(stage).read_text(),
        "advance": run._write_advance_script(stage, "5555.fake").read_text(),
        "fail": run._write_fail_script(stage, "5555.fake").read_text(),
    }


@pytest.mark.parametrize("kind", ["submit", "advance", "fail"])
def test_job_scripts_call_this_meristem_install(created_run, kind):
    """Test that every job script calls meristem through the interpreter that
    wrote it, rather than activating a conda environment
    """
    content = _job_scripts(created_run)[kind]
    assert meristem_cmd() in content
    assert "conda activate" not in content


def test_submit_command_stage_can_call_meristem_without_conda(
    tmp_path, command_stage_config_dict, fake_scheduler
):
    """Test that a command stage's bash -c child shell can call `meristem` with
    only system directories on PATH (no activated environment)
    """
    run = _run_with_stages(
        tmp_path / "myrun",
        [{**command_stage_config_dict, "command": "meristem --version"}],
        fake_scheduler,
    )
    run.submit()
    job_file = run.root / f"{run.run_id}_spinup_ad.pbs"

    # the function definition, its export, and the command call, in order
    snippet = "\n".join(
        line
        for line in job_file.read_text().splitlines()
        if line.startswith(("meristem() {", "export -f meristem", "bash -c"))
    )
    result = subprocess.run(
        ["/bin/bash", "-c", "set -euo pipefail\n" + snippet],
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "meristem" in result.stdout


def test_submit_command_stage_strict_mode_stops_on_failure(
    tmp_path, command_stage_config_dict, fake_scheduler
):
    """Test that the rendered command line really stops at the first failing
    command and exits non-zero
    """
    run = _run_with_stages(
        tmp_path / "myrun",
        [{**command_stage_config_dict, "command": "false; echo after"}],
        fake_scheduler,
    )
    run.submit()
    job_file = run.root / f"{run.run_id}_spinup_ad.pbs"
    result = subprocess.run(
        _command_call(job_file), capture_output=True, text=True, check=False
    )
    assert result.returncode != 0
    assert "after" not in result.stdout


def test_submit_skip_script_omits_command(command_run):
    """Test that skip_script also skips a stage's command"""
    # pylint: disable=protected-access
    job_file = command_run._write_job_script(
        command_run.stages[0], skip_script=True
    ).read_text()
    assert "meristem ensemble prepare" not in job_file


@pytest.fixture
def command_and_script_run(
    tmp_path, minimal_stage_config_dict, fake_scheduler
) -> ModelRun:
    """A ModelRun whose single stage runs a command and then a script"""
    stage = {**minimal_stage_config_dict, "command": "echo prepare"}
    return _run_with_stages(tmp_path / "myrun", [stage], fake_scheduler)


def _setup_calls(job_file: Path) -> list[list[str]]:
    """Every bash -c invocation (command, then script) in a job script, in order"""
    return [
        shlex.split(line)
        for line in job_file.read_text().splitlines()
        if line.startswith("bash -c")
    ]


def test_submit_command_and_script_runs_command_then_script(
    command_and_script_run, script_file
):
    """Test that a stage with both runs its command, then its script with the
    case root, each in strict mode
    """
    run = command_and_script_run
    run.submit()
    calls = _setup_calls(run.root / f"{run.run_id}_spinup_ad.pbs")
    assert calls == [
        ["bash", "-c", "set -euo pipefail; echo prepare"],
        [
            "bash",
            "-c",
            'set -euo pipefail; source "$0" "$@"',
            str(script_file),
            str(run.root / "spinup_ad"),
        ],
    ]


def test_submit_command_and_script_stops_if_command_fails(
    tmp_path, minimal_stage_config_dict, fake_scheduler
):
    """Test that the script doesn't run if the command before it fails (the job
    runs under set -e)
    """
    script = tmp_path / "setup.sh"
    script.write_text("echo script-ran\n", encoding="utf-8")
    run = _run_with_stages(
        tmp_path / "myrun",
        [{**minimal_stage_config_dict, "script": str(script), "command": "false"}],
        fake_scheduler,
    )
    run.submit()
    calls = _setup_calls(run.root / f"{run.run_id}_spinup_ad.pbs")
    result = subprocess.run(
        ["/bin/bash", "-c", "set -e\n" + "\n".join(shlex.join(c) for c in calls)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "script-ran" not in result.stdout


def test_submit_skip_script_omits_command_and_script(command_and_script_run):
    """Test that skip_script skips both a stage's command and its script"""
    # pylint: disable=protected-access
    job_file = command_and_script_run._write_job_script(
        command_and_script_run.stages[0], skip_script=True
    )
    assert _setup_calls(job_file) == []


@pytest.mark.parametrize("command", ["--submit-advance", "--submit-fail"])
def test_submit_script_correct_clm_run_submit_commands(created_run, command):
    """
    Test that ModelRun.submit() creates a PBS file with the correct meristem run commands
    for advancing and failing
    """
    created_run.submit()
    job_file = (created_run.root / f"{created_run.run_id}_spinup_ad.pbs").read_text()
    assert f'meristem run --root "$RUN_ROOT" {command} --stage "$STAGE"' in job_file


def _run_cime_block(
    run: ModelRun, job_ids: str
) -> tuple[subprocess.CompletedProcess, Path]:
    """Run the setup script's CIME block (cd case root through the meristem
    calls) against stub case.submit and xmlquery, with meristem recording its
    arguments instead of running

    Returns:
        tuple: the finished process, and the stub case root
    """
    case_root = run.root / run.stages[0].config.name
    case_root.mkdir(parents=True, exist_ok=True)
    (case_root / "case.submit").write_text(
        '#!/bin/bash\nprintf "%s\\n" "$@" > submit_args.txt\n'
    )
    (case_root / "xmlquery").write_text(
        '#!/bin/bash\n[[ "$*" == "--value JOB_IDS" ]] && printf "%s" "$FAKE_JOB_IDS"\n'
    )
    for stub in ("case.submit", "xmlquery"):
        (case_root / stub).chmod(0o755)

    lines = (run.root / f"{run.run_id}_{run.stages[0].config.name}.pbs").read_text()
    lines = lines.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("cd "))
    end = next(i for i, line in enumerate(lines) if "Setup complete" in line)
    calls = run.root / "meristem_calls.txt"
    script = "\n".join(
        [
            "set -euo pipefail",
            "RUN_ROOT=root STAGE=stage",
            f'meristem() {{ echo "$*" >> {shlex.quote(str(calls))}; }}',
            *lines[start:end],
        ]
    )
    result = subprocess.run(
        ["/bin/bash", "-c", script],
        env={"PATH": "/usr/bin:/bin", "FAKE_JOB_IDS": job_ids},
        capture_output=True,
        text=True,
        check=False,
    )
    return result, case_root


@pytest.mark.parametrize(
    "job_ids,advance_on,fail_on",
    [
        (
            "case.run:100.desched1, case.st_archive:101.desched1",
            "101.desched1",
            "100.desched1 --cime-job-id 101.desched1",
        ),
        # DOUT_S=FALSE: no archiver, so everything waits on case.run
        (
            "case.run:100.desched1",
            "100.desched1",
            "100.desched1 --cime-job-id 100.desched1",
        ),
    ],
)
def test_submit_script_cime_block_uses_case_run_and_archive_ids(
    created_run, job_ids, advance_on, fail_on
):
    """Test that the setup script calls case.submit with only its own arguments,
    advances after the final (archive) job, and watches case.run and the final
    job with fail jobs
    """
    created_run.submit()
    result, case_root = _run_cime_block(created_run, job_ids)
    assert result.returncode == 0, result.stderr

    # case.submit and JOB_IDS= on separate lines (trim_blocks once joined them)
    submit_args = (case_root / "submit_args.txt").read_text().splitlines()
    assert submit_args == ["--resubmit-immediate"]

    calls = (created_run.root / "meristem_calls.txt").read_text().splitlines()
    assert calls == [
        f"run --root root --submit-advance --stage stage --cime-job-id {advance_on}",
        f"run --root root --submit-fail --stage stage --cime-job-id {fail_on}",
    ]


def test_submit_script_cime_block_passes_submit_args(
    run_config_dict, ad_stage_config_dict, fake_scheduler
):
    """Test that submit_args reach case.submit as one -a argument, on its own line"""
    run_config_dict["stages"] = [
        {**ad_stage_config_dict, "submit_args": "-l job_priority=premium"}
    ]
    run = ModelRun.create(run_config_dict, scheduler=fake_scheduler)
    run.submit()
    result, case_root = _run_cime_block(
        run, "case.run:100.desched1, case.st_archive:101.desched1"
    )
    assert result.returncode == 0, result.stderr
    submit_args = (case_root / "submit_args.txt").read_text().splitlines()
    assert submit_args == ["--resubmit-immediate", "-a", "-l job_priority=premium"]


def test_submit_script_cime_block_without_case_run_id_fails(created_run):
    """Test that the setup script stops before submitting advance/fail jobs when
    JOB_IDS has no case.run ID
    """
    created_run.submit()
    result, _ = _run_cime_block(created_run, "")
    assert result.returncode != 0
    assert "No case.run job ID" in result.stderr
    assert not (created_run.root / "meristem_calls.txt").exists()


def test_submit_stage_status_set_to_submitted(created_run):
    """Test that ModelRun.submit() sets the first stage status to SUBMITTED"""
    created_run.submit()
    assert created_run.stages[0].state.status is StageStatus.SUBMITTED


def test_submit_stage_job_id_set(created_run):
    """Test that ModelRun.submit() stores the job ID returned by qsub on the stage"""
    created_run.submit()
    assert created_run.stages[0].state.job_id == "1.fake"


def test_submit_stage_attempts_incremented(created_run):
    """Test that ModelRun.submit() increments attempts on the stage"""
    created_run.submit()
    assert created_run.stages[0].state.attempts == 1


def test_submit_stage_submit_time_set(created_run):
    """Test that ModelRun.submit() sets submit_time on the stage"""
    created_run.submit()
    assert created_run.stages[0].state.submit_time is not None


def test_submit_state_persisted_to_disk(created_run):
    """Test that ModelRun.submit() persists updated stage state to disk"""
    created_run.submit()
    loaded = ModelRun.load(created_run.root, scheduler=created_run.scheduler)
    assert loaded.stages[0].state.job_id == "1.fake"
    assert loaded.stages[0].state.status is StageStatus.SUBMITTED
    assert loaded.stages[0].state.attempts == 1


def test_submit_returns_job_id(created_run):
    """Test that ModelRun.submit() returns the PBS job ID"""
    assert created_run.submit() == "1.fake"


@pytest.mark.parametrize(
    "non_pending_status",
    [
        StageStatus.SUBMITTED,
        StageStatus.DONE,
        StageStatus.FAILED,
    ],
)
def test_submit_skips_if_first_stage_not_pending(
    created_run, fake_scheduler, non_pending_status
):
    """Test that ModelRun.submit() is a no-op if the first stage is not PENDING"""
    object.__setattr__(created_run.stages[0].state, "_status", non_pending_status)
    created_run.submit()
    assert not fake_scheduler.submitted


def test_submit_passes_job_file_to_scheduler(created_run, fake_scheduler):
    """Test that ModelRun.submit() hands the generated job script to the scheduler"""
    created_run.submit()
    job_file = created_run.root / f"{created_run.run_id}_spinup_ad.pbs"
    assert fake_scheduler.submitted == [job_file]


def test_submit_returns_existing_job_id_when_skipping(created_run):
    """Test that ModelRun.submit() returns the existing job ID when skipping"""
    created_run.stages[0].state.status = StageStatus.SUBMITTED
    created_run.stages[0].state.job_id = "existing.pbs"
    assert created_run.submit() == "existing.pbs"


def test_submit_returns_none_when_skipping_with_no_job_id(created_run):
    """Test that ModelRun.submit() returns empty string when skipping a stage with no job ID"""
    created_run.stages[0].state.status = StageStatus.SUBMITTED
    assert created_run.submit() is None


def test_submit_does_not_mutate_stage_when_skipping(created_run):
    """Test that ModelRun.submit() does not change stage state when skipping"""
    created_run.stages[0].state.status = StageStatus.SUBMITTED
    created_run.stages[0].state.increment_attempts()
    created_run.stages[0].state.increment_attempts()
    created_run.stages[0].state.increment_attempts()
    created_run.submit()
    assert created_run.stages[0].state.attempts == 3


def test_submit_dry_run_does_not_call_qsub(created_run, fake_scheduler):
    """Test that submit(dry_run=True) does not call qsub"""
    created_run.submit(dry_run=True)
    assert not fake_scheduler.submitted


def test_submit_dry_run_job_id_format(created_run):
    """Test that submit(dry_run=True) produces a synthetic DRY_ job ID"""
    created_run.submit(dry_run=True)
    expected = f"DRY_{created_run.run_id}_{created_run.stages[0].config.name}"
    assert created_run.stages[0].state.job_id == expected


def test_submit_dry_run_still_writes_job_script(created_run):
    """Test that submit(dry_run=True) still generates the PBS script"""
    created_run.submit(dry_run=True)
    job_file = created_run.root / f"{created_run.run_id}_spinup_ad.pbs"
    assert job_file.exists()


def test_submit_dry_run_still_updates_stage_state(created_run):
    """Test that submit(dry_run=True) still marks the stage as SUBMITTED"""
    created_run.submit(dry_run=True)
    assert created_run.stages[0].state.status is StageStatus.SUBMITTED


def test_submit_qsub_failure_leaves_stage_pending(created_run, mocker):
    """Test that a qsub failure does not mutate stage state"""
    mocker.patch.object(
        created_run.scheduler,
        "submit",
        side_effect=subprocess.CalledProcessError(1, "qsub"),
    )
    with pytest.raises(subprocess.CalledProcessError):
        created_run.submit()
    assert created_run.stages[0].state.status is StageStatus.PENDING


def test_submit_qsub_failure_does_not_persist(created_run, mocker):
    """Test that a qsub failure does not write SUBMITTED state to disk"""
    mocker.patch.object(
        created_run.scheduler,
        "submit",
        side_effect=subprocess.CalledProcessError(1, "qsub"),
    )
    with pytest.raises(subprocess.CalledProcessError):
        created_run.submit()
    loaded = ModelRun.load(created_run.root, scheduler=created_run.scheduler)
    assert loaded.stages[0].state.status is StageStatus.PENDING


def test_submit_qsub_failure_does_not_increment_attempts(created_run, mocker):
    """Test that a qsub failure does not increment attempts"""
    mocker.patch.object(
        created_run.scheduler,
        "submit",
        side_effect=subprocess.CalledProcessError(1, "qsub"),
    )
    with pytest.raises(subprocess.CalledProcessError):
        created_run.submit()
    assert created_run.stages[0].state.attempts == 0


def _setup_script_args(job_file: Path) -> list[str]:
    """Arguments passed to the user's setup script in a generated job script"""
    for line in job_file.read_text().splitlines():
        if line.startswith("bash -c"):
            # split the way bash would, so quoted paths stay one argument
            return shlex.split(line.split('"$@"\' ', 1)[1])[1:]
    raise AssertionError(f"no setup script call in {job_file}")


def test_submit_first_stage_passes_no_previous_case(created_run):
    """Test that the first stage's setup script gets only its case_root"""
    created_run.submit()
    job_file = created_run.root / f"{created_run.run_id}_spinup_ad.pbs"
    assert _setup_script_args(job_file) == [str(created_run.root / "spinup_ad")]


def test_advance_passes_previous_case_name_and_root(two_stage_run):
    """Test that the next stage's setup script gets the previous case name and root"""
    two_stage_run.submit()
    two_stage_run.advance("spinup_ad")
    job_file = two_stage_run.root / f"{two_stage_run.run_id}_spinup_sasu.pbs"
    assert _setup_script_args(job_file) == [
        str(two_stage_run.root / "spinup_sasu"),
        "spinup_ad",
        str(two_stage_run.root / "spinup_ad"),
    ]


def test_advance_passes_configured_previous_case_root(
    tmp_path, ad_stage_config_dict, sasu_stage_config_dict, fake_scheduler
):
    """Test that a custom case_root on the previous stage sets both the previous
    case name (its basename) and the previous case root
    """
    ad_case_root = tmp_path / "elsewhere" / "my_ad_case"
    config = {
        "root": str(tmp_path / "two_stage"),
        "run_id": "member_0001",
        "project": "PROJ123",
        "email": "name@ucar.edu",
        "stages": [
            {**ad_stage_config_dict, "case_root": str(ad_case_root)},
            sasu_stage_config_dict,
        ],
    }
    run = ModelRun.create(config, scheduler=fake_scheduler)
    run.submit()
    run.advance("spinup_ad")
    job_file = run.root / f"{run.run_id}_spinup_sasu.pbs"
    assert _setup_script_args(job_file)[1:] == ["my_ad_case", str(ad_case_root)]


@pytest.mark.parametrize(
    "awkward_name",
    ["has space", "has $dollar", "has 'quote", "has;semicolon"],
)
def test_submit_script_quotes_awkward_paths(
    tmp_path, ad_stage_config_dict, sasu_stage_config_dict, fake_scheduler, awkward_name
):
    """Test that script and case paths with shell special characters reach the
    setup script as single, unmodified arguments
    """
    script = tmp_path / f"{awkward_name}.sh"
    script.touch()
    ad_case_root = tmp_path / awkward_name / "ad case"
    sasu_case_root = tmp_path / awkward_name / "sasu case"
    config = {
        "root": str(tmp_path / "two_stage"),
        "run_id": "member_0001",
        "project": "PROJ123",
        "email": "name@ucar.edu",
        "stages": [
            {**ad_stage_config_dict, "case_root": str(ad_case_root)},
            {
                **sasu_stage_config_dict,
                "script": str(script),
                "case_root": str(sasu_case_root),
            },
        ],
    }
    run = ModelRun.create(config, scheduler=fake_scheduler)
    run.submit()
    run.advance("spinup_ad")
    job_file = run.root / f"{run.run_id}_spinup_sasu.pbs"
    content = job_file.read_text()

    assert f"' {shlex.quote(str(script))} " in content
    assert _setup_script_args(job_file) == [
        str(sasu_case_root),
        "ad case",
        str(ad_case_root),
    ]
    assert f"cd {shlex.quote(str(sasu_case_root))}\n" in content


def test_submit_missing_later_script_raises(
    tmp_path, ad_stage_config_dict, sasu_stage_config_dict, fake_scheduler
):
    """Test that submit checks every stage's script up front, so a bad later
    stage fails now rather than inside an advance job, and nothing is submitted
    """
    sasu_script = tmp_path / "sasu.sh"
    sasu_script.touch()
    run = ModelRun.create(
        {
            "root": str(tmp_path / "run"),
            "project": "PROJ123",
            "stages": [
                ad_stage_config_dict,
                {**sasu_stage_config_dict, "script": str(sasu_script)},
            ],
        },
        scheduler=fake_scheduler,
    )
    sasu_script.unlink()
    with pytest.raises(ValueError, match=r"cannot open script.*sasu\.sh"):
        run.submit()
    assert fake_scheduler.submitted == []
    assert run.stages[0].state.status == StageStatus.PENDING


def test_submit_skipped_does_not_validate(created_run, script_file):
    """Test that submit doesn't validate when the first stage isn't PENDING
    (nothing is submitted, so a moved script shouldn't block it)
    """
    created_run.submit()
    script_file.unlink()
    assert created_run.submit() == created_run.stages[0].state.job_id
