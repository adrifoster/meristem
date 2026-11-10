"""Tests for meristem.run.cli"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, call

import pytest

from meristem.run.cli import (
    EXIT_LOST,
    build_parser,
    cmd_advance,
    cmd_create,
    cmd_fail,
    cmd_retry,
    cmd_status,
    cmd_submit,
    cmd_submit_advance,
    cmd_submit_fail,
    cmd_cancel,
    cmd_clean,
    main,
)


@pytest.fixture
def mock_run(mocker):
    """A mock ModelRun instance with all methods stubbed."""
    run = mocker.MagicMock()
    run.run_id = "member_0001"
    run.root = "/tmp/run"
    run.submit.return_value = "12345.pbs"
    run.submit_advance.return_value = "12345.pbs"
    run.advance.return_value = "12345.pbs"
    run.retry.return_value = "12345.pbs"
    run.format_status.return_value = "Run: member_0001"
    run.cancel.return_value = ["12345.pbs"]
    run.lost_stages.return_value = []
    return run


@pytest.fixture
def mock_load(mocker, mock_run):
    """Patch ModelRun.load to return mock_run."""
    return mocker.patch("meristem.run.cli.ModelRun.load", return_value=mock_run)


@pytest.fixture
def mock_create(mocker, mock_run):
    """Patch ModelRun.create to return mock_run."""
    return mocker.patch("meristem.run.cli.ModelRun.create", return_value=mock_run)


SUBMIT_ADVANCE_ARGV = [
    "submit-advance",
    "--root",
    "/tmp/run",
    "--stage",
    "spinup_ad",
    "--cime-job-id",
    "55555.pbs",
]

SUBMIT_FAIL_ARGV = [
    "submit-fail",
    "--root",
    "/tmp/run",
    "--stage",
    "spinup_ad",
    "--cime-job-id",
    "55555.pbs",
]


# parser


@pytest.mark.parametrize(
    "argv,handler",
    [
        (["create", "--config", "run.yaml"], cmd_create),
        (["submit", "--root", "/tmp/run"], cmd_submit),
        (SUBMIT_ADVANCE_ARGV, cmd_submit_advance),
        (SUBMIT_FAIL_ARGV, cmd_submit_fail),
        (["advance", "--root", "/tmp/run", "--stage", "spinup_ad"], cmd_advance),
        (["fail", "--root", "/tmp/run", "--stage", "spinup_ad"], cmd_fail),
        (["retry", "--root", "/tmp/run"], cmd_retry),
        (["status", "--root", "/tmp/run"], cmd_status),
        (["cancel", "--root", "/tmp/run"], cmd_cancel),
        (["clean", "--root", "/tmp/run"], cmd_clean),
    ],
)
def test_parser_subcommand_sets_handler(argv, handler):
    """Test that each subcommand parses and routes to its handler"""
    args = build_parser().parse_args(argv)
    assert args.handler is handler


def test_parser_retry_accepts_optional_args():
    """Test that retry accepts --stage, --skip-script and --lost"""
    args = build_parser().parse_args(
        [
            "retry",
            "--root",
            "/tmp/run",
            "--stage",
            "spinup_ad",
            "--skip-script",
            "--lost",
        ]
    )
    assert args.stage == "spinup_ad"
    assert args.skip_script
    assert args.lost


def test_parser_submit_fail_collects_cime_job_ids():
    """Test that a repeated --cime-job-id with submit-fail is collected as a list"""
    args = build_parser().parse_args(SUBMIT_FAIL_ARGV + ["--cime-job-id", "55556.pbs"])
    assert args.cime_job_id == ["55555.pbs", "55556.pbs"]


def test_parser_submit_advance_cime_job_id_is_a_string():
    """Test that submit-advance takes a single --cime-job-id as a string"""
    args = build_parser().parse_args(SUBMIT_ADVANCE_ARGV)
    assert args.cime_job_id == "55555.pbs"


@pytest.mark.parametrize(
    "argv",
    [
        [],  # no subcommand
        ["bogus", "--root", "/tmp/run"],  # unknown subcommand
        ["create"],  # missing --config
        [
            "create",
            "--config",
            "run.yaml",
            "--root",
            "/tmp/run",
        ],  # create has no --root
        ["submit"],  # missing --root
        ["status"],  # missing --root
        ["retry"],  # missing --root
        ["cancel"],  # missing --root
        ["clean"],  # missing --root
        ["advance", "--root", "/tmp/run"],  # missing --stage
        ["fail", "--root", "/tmp/run"],  # missing --stage
        SUBMIT_ADVANCE_ARGV[:-2],  # missing --cime-job-id
        SUBMIT_FAIL_ARGV[:-2],  # missing --cime-job-id
        [
            "submit-advance",
            "--root",
            "/tmp/run",
            "--cime-job-id",
            "1.pbs",
        ],  # no --stage
        ["submit-fail", "--root", "/tmp/run", "--cime-job-id", "1.pbs"],  # no --stage
    ],
)
def test_main_argument_errors_exit_with_code_2(argv):
    """Test that missing or invalid arguments error with code 2"""
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2


@pytest.mark.parametrize("flag", ["--skip-script", "--lost"])
@pytest.mark.parametrize(
    "argv",
    [
        ["submit", "--root", "/tmp/run"],
        ["advance", "--root", "/tmp/run", "--stage", "spinup_ad"],
        ["fail", "--root", "/tmp/run", "--stage", "spinup_ad"],
        ["status", "--root", "/tmp/run"],
        ["cancel", "--root", "/tmp/run"],
        SUBMIT_ADVANCE_ARGV,
        SUBMIT_FAIL_ARGV,
    ],
)
def test_retry_only_flags_rejected_elsewhere(argv, flag):
    """Test that --skip-script and --lost are only accepted by retry"""
    with pytest.raises(SystemExit) as exc:
        main(argv + [flag])
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["fail", "--root", "/tmp/run", "--stage", "spinup_ad"],
        ["status", "--root", "/tmp/run"],
    ],
)
def test_dry_run_rejected_where_unsupported(argv):
    """Test that --dry-run is rejected by subcommands that never submit"""
    with pytest.raises(SystemExit) as exc:
        main(argv + ["--dry-run"])
    assert exc.value.code == 2


# create


def test_create_calls_model_run_create(mock_create, capsys):
    """Test that create calls ModelRun.create and prints the next step"""
    assert main(["create", "--config", "run.yaml"]) == 0
    mock_create.assert_called_once_with("run.yaml")
    assert "meristem run submit --root /tmp/run" in capsys.readouterr().out


# submit


@pytest.mark.usefixtures("mock_load")
@pytest.mark.parametrize("dry_run", [False, True])
def test_submit_calls_submit(mock_run, capsys, dry_run):
    """Test that submit calls ModelRun.submit, passing --dry-run through"""
    argv = ["submit", "--root", "/tmp/run"] + (["--dry-run"] if dry_run else [])
    assert main(argv) == 0
    mock_run.submit.assert_called_once_with(dry_run=dry_run)
    assert "Submitted: 12345.pbs" in capsys.readouterr().out


def test_submit_loads_run_from_root(mock_load):
    """Test that the run is loaded from --root"""
    main(["submit", "--root", "/tmp/run"])
    assert str(mock_load.call_args.args[0]) == "/tmp/run"


@pytest.mark.usefixtures("mock_load")
def test_submit_prints_nothing_without_job_id(mock_run, capsys):
    """Test that no job ID line is printed when nothing was submitted"""
    mock_run.submit.return_value = None
    main(["submit", "--root", "/tmp/run"])
    assert "Submitted" not in capsys.readouterr().out


# submit-advance


@pytest.mark.usefixtures("mock_load")
@pytest.mark.parametrize("dry_run", [False, True])
def test_submit_advance_calls_submit_advance(mock_run, dry_run):
    """Test that submit-advance calls ModelRun.submit_advance, passing --dry-run through"""
    argv = SUBMIT_ADVANCE_ARGV + (["--dry-run"] if dry_run else [])
    assert main(argv) == 0
    mock_run.submit_advance.assert_called_once_with(
        "spinup_ad", "55555.pbs", dry_run=dry_run, attempt=None
    )


# submit-fail


@pytest.mark.usefixtures("mock_load")
@pytest.mark.parametrize("dry_run", [False, True])
def test_submit_fail_calls_submit_fail(mock_run, dry_run):
    """Test that submit-fail calls ModelRun.submit_fail, passing --dry-run through"""
    argv = SUBMIT_FAIL_ARGV + (["--dry-run"] if dry_run else [])
    assert main(argv) == 0
    mock_run.submit_fail.assert_called_once_with(
        "spinup_ad", ["55555.pbs"], dry_run=dry_run, attempt=None
    )


@pytest.mark.usefixtures("mock_load")
def test_submit_fail_passes_every_cime_job_id(mock_run, capsys):
    """Test that a repeated --cime-job-id reaches submit_fail as a list, and each
    fail job ID is printed
    """
    mock_run.submit_fail.return_value = ["1.pbs", "2.pbs"]
    main(SUBMIT_FAIL_ARGV[:-1] + ["100.pbs", "--cime-job-id", "101.pbs"])
    mock_run.submit_fail.assert_called_once_with(
        "spinup_ad", ["100.pbs", "101.pbs"], dry_run=False, attempt=None
    )
    out = capsys.readouterr().out
    assert "Fail job submitted: 1.pbs" in out
    assert "Fail job submitted: 2.pbs" in out


# advance


@pytest.mark.usefixtures("mock_load")
@pytest.mark.parametrize("dry_run", [False, True])
def test_advance_calls_advance(mock_run, dry_run):
    """Test that advance calls ModelRun.advance, passing --dry-run through"""
    argv = ["advance", "--root", "/tmp/run", "--stage", "spinup_ad"]
    assert main(argv + (["--dry-run"] if dry_run else [])) == 0
    mock_run.advance.assert_called_once_with("spinup_ad", dry_run=dry_run, attempt=None)


# fail


@pytest.mark.usefixtures("mock_load")
def test_fail_calls_fail(mock_run):
    """Test that fail calls ModelRun.fail"""
    assert main(["fail", "--root", "/tmp/run", "--stage", "spinup_ad"]) == 0
    mock_run.fail.assert_called_once_with("spinup_ad", attempt=None)


@pytest.mark.usefixtures("mock_load")
@pytest.mark.parametrize(
    "argv,method",
    [
        (SUBMIT_ADVANCE_ARGV, "submit_advance"),
        (SUBMIT_FAIL_ARGV, "submit_fail"),
        (["advance", "--root", "/tmp/run", "--stage", "spinup_ad"], "advance"),
        (["fail", "--root", "/tmp/run", "--stage", "spinup_ad"], "fail"),
    ],
)
def test_job_callbacks_pass_attempt(mock_run, argv, method):
    """Test that --attempt (set by meristem's job scripts) reaches each method
    jobs call back into, as an int
    """
    assert main(argv + ["--attempt", "3"]) == 0
    assert getattr(mock_run, method).call_args.kwargs["attempt"] == 3


@pytest.mark.usefixtures("mock_load")
@pytest.mark.parametrize(
    "extra,attempt,submit_log",
    [
        ([], None, None),
        (["--attempt", "2"], 2, None),
        (
            ["--submit-log", "/tmp/run/case_submit.log"],
            None,
            Path("/tmp/run/case_submit.log"),
        ),
    ],
)
def test_record_cime_jobs_calls_record_cime_jobs(mock_run, extra, attempt, submit_log):
    """Test that record-cime-jobs passes the stage, JOB_IDS and attempt through"""
    argv = ["record-cime-jobs", "--root", "/tmp/run", "--stage", "spinup_ad"]
    argv += ["--job-ids", "case.run:100, case.st_archive:101", *extra]
    assert main(argv) == 0
    mock_run.record_cime_jobs.assert_called_once_with(
        "spinup_ad",
        "case.run:100, case.st_archive:101",
        attempt=attempt,
        submit_log=submit_log,
    )


# retry


@pytest.mark.usefixtures("mock_load")
@pytest.mark.parametrize(
    "extra,expected_args,expected_kwargs",
    [
        ([], (None,), {}),
        (["--stage", "spinup_ad"], ("spinup_ad",), {}),
        (["--dry-run"], (None,), {"dry_run": True}),
        (["--skip-script"], (None,), {"skip_script": True}),
        (["--stage", "spinup_ad", "--lost"], ("spinup_ad",), {"lost": True}),
    ],
)
def test_retry_calls_retry(mock_run, extra, expected_args, expected_kwargs):
    """Test that retry calls ModelRun.retry, passing optional args through"""
    assert main(["retry", "--root", "/tmp/run"] + extra) == 0
    kwargs = {"dry_run": False, "skip_script": False, "lost": False}
    kwargs.update(expected_kwargs)
    mock_run.retry.assert_called_once_with(*expected_args, **kwargs)


# status


@pytest.mark.usefixtures("mock_load")
def test_status_nothing_lost_returns_zero(mock_run, capsys):
    """Test that status prints the status and returns 0 when nothing is lost"""
    assert main(["status", "--root", "/tmp/run"]) == 0
    mock_run.format_status.assert_called_once_with(lost=[])
    out = capsys.readouterr().out
    assert "Run: member_0001" in out
    assert "--lost" not in out


@pytest.mark.usefixtures("mock_load")
def test_status_lost_returns_exit_lost(mock_run, capsys):
    """Test that status returns EXIT_LOST, shows the lost stage, and prints
    the retry command
    """
    mock_run.lost_stages.return_value = ["spinup_ad"]
    assert main(["status", "--root", "/tmp/run"]) == EXIT_LOST
    mock_run.format_status.assert_called_once_with(lost=["spinup_ad"])
    assert (
        "meristem run retry --root /tmp/run --stage spinup_ad --lost"
        in capsys.readouterr().out
    )


@pytest.mark.usefixtures("mock_load")
def test_status_does_not_retry(mock_run):
    """Test that status never resubmits anything"""
    mock_run.lost_stages.return_value = ["spinup_ad"]
    main(["status", "--root", "/tmp/run"])
    mock_run.retry.assert_not_called()
    mock_run.submit.assert_not_called()


# error handling


def test_create_empty_config_exits_one(tmp_path, capsys):
    """Test that `run create` with an empty YAML file reports it and exits 1"""
    path = tmp_path / "run.yaml"
    path.write_text("", encoding="utf-8")
    assert main(["create", "--config", str(path)]) == 1
    assert "ERROR: run config is empty." in capsys.readouterr().err


def test_main_load_missing_root_exits_one(tmp_path, capsys):
    """Test that a missing root gives exit status 1 and an error on stderr"""
    result = main(["status", "--root", str(tmp_path / "nonexistent")])
    assert result == 1
    assert "ERROR" in capsys.readouterr().err


def test_main_keyboard_interrupt_exits_130(mocker):
    """Test that main with a keyboard interrupt exits with status 130"""
    mocker.patch("meristem.run.cli.ModelRun.load", side_effect=KeyboardInterrupt)
    assert main(["status", "--root", "/tmp/run"]) == 130


def test_main_unexpected_exception_exits_one(mocker, capsys):
    """Test that an unexpected exception exits with status 1 and prints to stderr"""
    mocker.patch("meristem.run.cli.ModelRun.load", side_effect=RuntimeError("boom"))
    assert main(["status", "--root", "/tmp/run"]) == 1
    assert "ERROR: boom" in capsys.readouterr().err


@pytest.mark.usefixtures("mock_load")
def test_cancel_error_exits_one(mock_run, capsys):
    """Test that a qdel failure exits with status 1 and prints the error"""
    mock_run.cancel.side_effect = RuntimeError("qdel failed (exit 1): no server")
    assert main(["cancel", "--root", "/tmp/run"]) == 1
    assert "ERROR: qdel failed" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["create", "--config", "run.yaml"],
        ["status", "--root", "/tmp/run"],
    ],
)
def test_main_debug_flag_reraises_exception(mocker, argv):
    """Test that --debug causes exceptions to propagate rather than being caught"""
    mocker.patch("meristem.run.cli.ModelRun.create", side_effect=RuntimeError("boom"))
    mocker.patch("meristem.run.cli.ModelRun.load", side_effect=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        main(argv + ["--debug"])


@pytest.mark.usefixtures("mock_load")
@pytest.mark.parametrize("dry_run", [False, True])
def test_cancel_calls_cancel(mock_run, capsys, dry_run):
    """Test that cancel calls ModelRun.cancel, passing --dry-run through"""
    argv = ["cancel", "--root", "/tmp/run"] + (["--dry-run"] if dry_run else [])
    assert main(argv) == 0
    mock_run.cancel.assert_called_once_with(dry_run=dry_run)
    out = capsys.readouterr().out
    if dry_run:
        assert "Canceled:" not in out
    else:
        assert "Canceled: 12345.pbs" in out


# clean

CLEAN_ARGV = ["clean", "--root", "/tmp/run"]
CLEAN_TARGETS = [Path("/tmp/run/spinup_ad"), Path("/tmp/run")]


@pytest.fixture
def mock_clean_run(mock_run):
    """mock_run whose clean returns CLEAN_TARGETS"""
    mock_run.clean.return_value = CLEAN_TARGETS
    return mock_run


def _no_input(_prompt=""):
    """Stand-in for input() in tests where the prompt must not be shown"""
    raise AssertionError("input() should not be called")


def _clean_call(dry_run, stages=None, include_archive=False, cancel=False):
    """The call cmd_clean should make to ModelRun.clean"""
    return call(
        stages=stages, include_archive=include_archive, cancel=cancel, dry_run=dry_run
    )


def test_parser_clean_defaults():
    """Test that clean's options default to all stages and nothing extra"""
    args = build_parser().parse_args(CLEAN_ARGV)
    assert args.stages is None
    assert not args.cancel
    assert not args.include_archive
    assert not args.yes
    assert not args.dry_run


def test_parser_clean_accepts_optional_args():
    """Test that clean accepts --stage(s), --cancel, --include-archive and --yes"""
    args = build_parser().parse_args(
        CLEAN_ARGV
        + ["--stage", "a", "b", "--stages", "c"]
        + ["--cancel", "--include-archive", "--yes"]
    )
    assert args.stages == ["a", "b", "c"]
    assert args.cancel
    assert args.include_archive
    assert args.yes


@pytest.mark.usefixtures("mock_load")
def test_clean_dry_run_only_previews(mock_clean_run, monkeypatch, capsys):
    """Test that --dry-run lists the paths without asking or deleting"""
    monkeypatch.setattr("builtins.input", _no_input)
    assert main(CLEAN_ARGV + ["--dry-run"]) == 0
    assert mock_clean_run.clean.call_args_list == [_clean_call(dry_run=True)]
    out = capsys.readouterr().out
    assert all(str(path) in out for path in CLEAN_TARGETS)


@pytest.mark.usefixtures("mock_load")
@pytest.mark.parametrize("answer", ["n", "", "no", "maybe"])
def test_clean_declined_deletes_nothing(mock_clean_run, monkeypatch, capsys, answer):
    """Test that anything but y/yes at the prompt deletes nothing"""
    monkeypatch.setattr("builtins.input", lambda _prompt="": answer)
    assert main(CLEAN_ARGV) == 0
    assert mock_clean_run.clean.call_args_list == [_clean_call(dry_run=True)]
    assert "Nothing deleted" in capsys.readouterr().out


@pytest.mark.usefixtures("mock_load")
@pytest.mark.parametrize("answer", ["y", "yes", " Y "])
def test_clean_confirmed_deletes(mock_clean_run, monkeypatch, capsys, answer):
    """Test that y/yes at the prompt previews, then cleans for real"""
    monkeypatch.setattr("builtins.input", lambda _prompt="": answer)
    assert main(CLEAN_ARGV) == 0
    assert mock_clean_run.clean.call_args_list == [
        _clean_call(dry_run=True),
        _clean_call(dry_run=False),
    ]
    assert "Cleaned 2" in capsys.readouterr().out


@pytest.mark.usefixtures("mock_load")
def test_clean_yes_skips_prompt(mock_clean_run, monkeypatch):
    """Test that --yes cleans without asking, passing the options through"""
    monkeypatch.setattr("builtins.input", _no_input)
    argv = CLEAN_ARGV + ["--stage", "spinup_ad", "--include-archive", "--yes"]
    assert main(argv) == 0
    options = {"stages": ["spinup_ad"], "include_archive": True}
    assert mock_clean_run.clean.call_args_list == [
        _clean_call(dry_run=True, **options),
        _clean_call(dry_run=False, **options),
    ]


@pytest.mark.usefixtures("mock_load")
def test_clean_whole_run_says_root_goes_only_if_empty(mock_clean_run, capsys):
    """Test that a whole clean's preview says root is removed only if empty"""
    assert main(CLEAN_ARGV + ["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert f"then {mock_clean_run.root}, if nothing else is left in it" in out


@pytest.mark.usefixtures("mock_load")
def test_clean_one_stage_does_not_mention_root(capsys):
    """Test that cleaning some stages doesn't say root goes"""
    assert main(CLEAN_ARGV + ["--stage", "spinup_ad", "--dry-run"]) == 0
    assert "if nothing else is left" not in capsys.readouterr().out


@pytest.mark.usefixtures("mock_load")
def test_clean_cancel_says_jobs_will_be_canceled(mock_clean_run, capsys):
    """Test that --cancel is passed through and the preview says so"""
    assert main(CLEAN_ARGV + ["--cancel", "--dry-run"]) == 0
    assert mock_clean_run.clean.call_args_list == [
        _clean_call(dry_run=True, cancel=True)
    ]
    assert "canceled first" in capsys.readouterr().out


@pytest.mark.usefixtures("mock_load")
def test_clean_nothing_to_clean(mock_run, monkeypatch, capsys):
    """Test that an empty preview says so and neither asks nor cleans"""
    mock_run.clean.return_value = []
    monkeypatch.setattr("builtins.input", _no_input)
    assert main(CLEAN_ARGV) == 0
    assert mock_run.clean.call_count == 1
    assert "Nothing to clean" in capsys.readouterr().out


@pytest.mark.usefixtures("mock_load")
def test_clean_error_exits_one(mock_run, capsys):
    """Test that a refused clean (e.g. jobs still active) exits 1 with the error"""
    mock_run.clean.side_effect = ValueError("jobs still queued or running")
    assert main(CLEAN_ARGV) == 1
    assert "ERROR: jobs still queued or running" in capsys.readouterr().err


@pytest.mark.usefixtures("mock_load")
def test_clean_without_terminal_says_use_yes(mock_clean_run, monkeypatch, capsys):
    """Test that clean with no terminal to answer the prompt deletes nothing
    and says to use --yes in the error message
    """
    # input() raises EOFError when no terminal
    monkeypatch.setattr("builtins.input", Mock(side_effect=EOFError))
    assert main(CLEAN_ARGV) == 1
    assert mock_clean_run.clean.call_args_list == [_clean_call(dry_run=True)]
    assert "--yes" in capsys.readouterr().err
