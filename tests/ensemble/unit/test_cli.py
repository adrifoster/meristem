"""Tests for meristem.ensemble.cli"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import call

import pandas as pd
import pytest

from meristem.ensemble.cli import (
    EXIT_LOST,
    build_parser,
    cmd_cancel,
    cmd_clean,
    cmd_create,
    cmd_finalize,
    cmd_manifest,
    cmd_prepare,
    cmd_retry,
    cmd_status,
    cmd_submit,
    main,
)

ROOT = "/tmp/ens"
MEMBER_STAGE = ["--member", "ens_0000", "--stage", "spinup_ad"]


def _status(statuses: list[str]) -> pd.DataFrame:
    """A minimal status() frame with the given statuses"""
    return pd.DataFrame(
        {
            "member": [f"ens_{i:04d}" for i in range(len(statuses))],
            "stage": "spinup_ad",
            "status": statuses,
        }
    )


@pytest.fixture
def mock_ensemble(mocker, tmp_path):
    """A mock Ensemble instance with all methods stubbed"""
    ensemble = mocker.MagicMock()
    ensemble.config.ensemble_id = "my_ens-1"
    ensemble.root = tmp_path
    ensemble.status.return_value = _status(["pending", "done"])
    ensemble.format_status.return_value = "Ensemble: my_ens-1"
    ensemble.manifest.return_value = pd.DataFrame({"member": ["ens_0000"]})
    return ensemble


@pytest.fixture
def mock_load(mocker, mock_ensemble):
    """Patch Ensemble.load to return mock_ensemble"""
    return mocker.patch(
        "meristem.ensemble.cli.Ensemble.load", return_value=mock_ensemble
    )


@pytest.fixture
def mock_create(mocker, mock_ensemble):
    """Patch Ensemble.create to return mock_ensemble"""
    return mocker.patch(
        "meristem.ensemble.cli.Ensemble.create", return_value=mock_ensemble
    )


# parser


@pytest.mark.parametrize(
    "argv,handler",
    [
        (["create", "--config", "ens.yaml"], cmd_create),
        (["submit", "--root", ROOT], cmd_submit),
        (["prepare", "--root", ROOT, *MEMBER_STAGE], cmd_prepare),
        (["finalize", "--root", ROOT, *MEMBER_STAGE], cmd_finalize),
        (["retry", "--root", ROOT], cmd_retry),
        (["status", "--root", ROOT], cmd_status),
        (["manifest", "--root", ROOT], cmd_manifest),
        (["cancel", "--root", ROOT], cmd_cancel),
        (["clean", "--root", ROOT], cmd_clean),
    ],
)
def test_parser_subcommand_sets_handler(argv, handler):
    """Test that each subcommand parses and routes to its handler"""
    args = build_parser().parse_args(argv)
    assert args.handler is handler


def test_parser_submit_limit_is_int():
    """Test that submit --limit is parsed as an int"""
    args = build_parser().parse_args(["submit", "--root", ROOT, "--limit", "2"])
    assert args.limit == 2


@pytest.mark.parametrize("command", ["retry", "status", "cancel", "clean"])
@pytest.mark.parametrize(
    "flags",
    [
        ["--members", "a", "b"],
        ["--member", "a", "b"],
        ["--member", "a", "--members", "b"],
    ],
)
def test_parser_members_collects_list(command, flags):
    """Test that --member/--members take several names and extend each other"""
    args = build_parser().parse_args([command, "--root", ROOT, *flags])
    assert args.members == ["a", "b"]


@pytest.mark.parametrize("command", ["retry", "status", "cancel", "clean"])
def test_parser_members_default_none(command):
    """Test that members defaults to None (all members)"""
    args = build_parser().parse_args([command, "--root", ROOT])
    assert args.members is None


@pytest.mark.parametrize(
    "argv",
    [
        [],  # no subcommand
        ["bogus", "--root", ROOT],  # unknown subcommand
        ["create"],  # missing --config
        ["create", "--config", "ens.yaml", "--root", ROOT],  # create has no --root
        ["submit"],  # missing --root
        ["submit", "--root", ROOT, "--limit", "two"],  # --limit not an int
        ["prepare", "--root", ROOT, "--member", "ens_0000"],  # missing --stage
        ["prepare", "--root", ROOT, "--stage", "spinup_ad"],  # missing --member
        ["finalize", "--root", ROOT, "--member", "ens_0000"],  # missing --stage
        ["finalize", "--root", ROOT, "--stage", "spinup_ad"],  # missing --member
        ["retry"],  # missing --root
        ["retry", "--root", ROOT, "--members"],  # --members with no names
        ["status"],  # missing --root
        ["manifest"],  # missing --root
        ["cancel"],  # missing --root
        ["cancel", "--root", ROOT, "--members"],  # --members with no names
        ["clean"],  # missing --root
        ["clean", "--root", ROOT, "--stages"],  # --stages with no names
    ],
)
def test_main_argument_errors_exit_with_code_2(argv):
    """Test that missing or invalid arguments error with code 2"""
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["prepare", "--root", ROOT, *MEMBER_STAGE],
        ["finalize", "--root", ROOT, *MEMBER_STAGE],
        ["status", "--root", ROOT],
        ["manifest", "--root", ROOT],
    ],
)
def test_dry_run_rejected_where_unsupported(argv):
    """Test that --dry-run is rejected by subcommands that never submit"""
    with pytest.raises(SystemExit) as exc:
        main(argv + ["--dry-run"])
    assert exc.value.code == 2


@pytest.mark.parametrize("flag", ["--failed", "--lost", "--skip-script"])
@pytest.mark.parametrize(
    "argv",
    [
        ["submit", "--root", ROOT],
        ["status", "--root", ROOT],
        ["prepare", "--root", ROOT, *MEMBER_STAGE],
        ["cancel", "--root", ROOT],
        ["clean", "--root", ROOT],
    ],
)
def test_retry_only_flags_rejected_elsewhere(argv, flag):
    """Test that --failed, --lost and --skip-script are only accepted by retry"""
    with pytest.raises(SystemExit) as exc:
        main(argv + [flag])
    assert exc.value.code == 2


# create


def test_create_calls_ensemble_create(mock_create, mock_ensemble, capsys):
    """Test that create calls Ensemble.create and prints the next step"""
    assert main(["create", "--config", "ens.yaml"]) == 0
    mock_create.assert_called_once_with("ens.yaml")
    out = capsys.readouterr().out
    assert f"Initialized ensemble 'my_ens-1' at {mock_ensemble.root}" in out
    assert f"meristem ensemble submit --root {mock_ensemble.root}" in out


# submit


@pytest.mark.usefixtures("mock_load")
@pytest.mark.parametrize(
    "extra,limit,dry_run",
    [
        ([], None, False),
        (["--limit", "2"], 2, False),
        (["--dry-run"], None, True),
    ],
)
def test_submit_calls_submit(mock_ensemble, capsys, extra, limit, dry_run):
    """Test that submit calls Ensemble.submit, passing --limit and --dry-run"""
    assert main(["submit", "--root", ROOT, *extra]) == 0
    mock_ensemble.submit.assert_called_once_with(limit=limit, dry_run=dry_run)
    assert "Submitted ensemble 'my_ens-1'" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    [
        ["submit", "--root", ROOT],
        ["prepare", "--root", ROOT, *MEMBER_STAGE],
        ["finalize", "--root", ROOT, *MEMBER_STAGE],
        ["retry", "--root", ROOT, "--failed"],
        ["status", "--root", ROOT],
        ["manifest", "--root", ROOT],
    ],
)
def test_loads_ensemble_from_root(mock_load, argv):
    """Test that every --root command loads the ensemble from --root"""
    main(argv)
    mock_load.assert_called_once_with(Path(ROOT))


# prepare / finalize


@pytest.mark.usefixtures("mock_load")
@pytest.mark.parametrize("command", ["prepare", "finalize"])
def test_prepare_finalize_call_ensemble(mock_ensemble, command):
    """Test that prepare and finalize call the Ensemble method of the same
    name with --member and --stage
    """
    assert main([command, "--root", ROOT, *MEMBER_STAGE]) == 0
    getattr(mock_ensemble, command).assert_called_once_with(
        member="ens_0000", stage="spinup_ad"
    )


# retry


@pytest.mark.usefixtures("mock_load")
@pytest.mark.parametrize(
    "extra,expected",
    [
        (["--failed"], {"failed": True}),
        (["--lost"], {"lost": True}),
        (["--failed", "--lost"], {"failed": True, "lost": True}),
        (["--stage", "spinup_ad"], {"stage": "spinup_ad"}),
        (["--members", "a", "b"], {"members": ["a", "b"]}),
        (["--failed", "--skip-script"], {"failed": True, "skip_script": True}),
        (["--failed", "--dry-run"], {"failed": True, "dry_run": True}),
    ],
)
def test_retry_calls_retry(mock_ensemble, extra, expected):
    """Test that retry calls Ensemble.retry, passing every option through"""
    assert main(["retry", "--root", ROOT, *extra]) == 0
    kwargs = {
        "failed": False,
        "lost": False,
        "members": None,
        "stage": None,
        "skip_script": False,
        "dry_run": False,
    }
    kwargs.update(expected)
    mock_ensemble.retry.assert_called_once_with(**kwargs)


def test_retry_nothing_selected_exits_one(mock_load, mock_ensemble, capsys):
    """Test that retry's own 'nothing selected' error exits 1 with the message"""
    mock_ensemble.retry.side_effect = ValueError("nothing selected to retry")
    assert main(["retry", "--root", ROOT]) == 1
    assert mock_load.called
    assert "ERROR: nothing selected to retry" in capsys.readouterr().err


# cancel


@pytest.mark.usefixtures("mock_load")
@pytest.mark.parametrize(
    "extra,expected",
    [
        ([], {}),
        (["--dry-run"], {"dry_run": True}),
        (["--members", "a", "b"], {"members": ["a", "b"]}),
        (["--member", "a", "--dry-run"], {"members": ["a"], "dry_run": True}),
    ],
)
def test_cancel_calls_cancel(mock_ensemble, extra, expected):
    """Test that cancel calls Ensemble.cancel, passing every option through"""
    assert main(["cancel", "--root", ROOT, *extra]) == 0
    kwargs = {"members": None, "dry_run": False}
    kwargs.update(expected)
    mock_ensemble.cancel.assert_called_once_with(**kwargs)


def test_cancel_error_exits_one(mock_load, mock_ensemble, capsys):
    """Test that a cancel failure (e.g. qdel) exits 1 with the message"""
    mock_ensemble.cancel.side_effect = RuntimeError("qdel failed (exit 1): no server")
    assert main(["cancel", "--root", ROOT]) == 1
    assert mock_load.called
    assert "ERROR: qdel failed" in capsys.readouterr().err


# clean

CLEAN_ARGV = ["clean", "--root", ROOT]


def _clean_targets(n: int) -> list[Path]:
    """n fake clone paths, then the ensemble root, as a whole clean lists them"""
    return [Path(f"/cases/clone_{i}") for i in range(n)] + [Path(ROOT)]


def _no_input(_prompt=""):
    """Stand-in for input() in tests where the prompt must not be shown"""
    raise AssertionError("input() should not be called")


def _clean_call(
    dry_run, members=None, stages=None, include_archive=False, cancel=False
):
    """The call cmd_clean should make to Ensemble.clean"""
    return call(
        members=members,
        stages=stages,
        include_archive=include_archive,
        cancel=cancel,
        dry_run=dry_run,
    )


@pytest.fixture
def mock_clean(mock_ensemble):
    """mock_ensemble whose clean returns 8 paths (7 clones and the root)"""
    mock_ensemble.clean.return_value = _clean_targets(7)
    return mock_ensemble


def test_parser_clean_defaults():
    """Test that clean's options default to everything and nothing extra"""
    args = build_parser().parse_args(CLEAN_ARGV)
    assert args.members is None
    assert args.stages is None
    assert not args.include_archive
    assert not args.cancel
    assert not args.yes
    assert not args.dry_run


def test_parser_clean_accepts_optional_args():
    """Test that clean accepts --stage(s), --include-archive, --cancel and --yes"""
    args = build_parser().parse_args(
        CLEAN_ARGV
        + ["--stage", "a", "b", "--stages", "c"]
        + ["--include-archive", "--cancel", "--yes"]
    )
    assert args.stages == ["a", "b", "c"]
    assert args.include_archive
    assert args.cancel
    assert args.yes


@pytest.mark.usefixtures("mock_load")
def test_clean_dry_run_lists_every_path(mock_clean, monkeypatch, capsys):
    """Test that --dry-run prints every path without asking or deleting"""
    monkeypatch.setattr("builtins.input", _no_input)
    assert main(CLEAN_ARGV + ["--dry-run"]) == 0
    assert mock_clean.clean.call_args_list == [_clean_call(dry_run=True)]
    out = capsys.readouterr().out
    assert all(str(path) in out for path in _clean_targets(7))


@pytest.mark.usefixtures("mock_load")
def test_clean_prompt_shows_summary(mock_clean, monkeypatch, capsys):
    """Test that the prompt shows a count, a few paths and the root, not
    every path
    """
    monkeypatch.setattr("builtins.input", lambda _prompt="": "n")
    main(CLEAN_ARGV)
    out = capsys.readouterr().out
    assert "Will delete 8 paths" in out
    assert "/cases/clone_0" in out
    assert "/cases/clone_6" not in out
    assert f"ensemble root {mock_clean.root}" in out
    assert "--dry-run" in out


@pytest.mark.usefixtures("mock_load")
@pytest.mark.parametrize(
    "extra", [["--members", "ens_0000"], ["--stages", "spinup_ad"]]
)
def test_clean_partial_does_not_mention_root(mock_ensemble, monkeypatch, capsys, extra):
    """Test that cleaning some members or stages doesn't say the root goes"""
    mock_ensemble.clean.return_value = [Path("/cases/clone_0")]
    monkeypatch.setattr("builtins.input", lambda _prompt="": "n")
    main(CLEAN_ARGV + extra)
    assert "ensemble root" not in capsys.readouterr().out


@pytest.mark.usefixtures("mock_load")
@pytest.mark.parametrize("answer", ["n", "", "no", "maybe"])
def test_clean_declined_deletes_nothing(mock_clean, monkeypatch, capsys, answer):
    """Test that anything but y/yes at the prompt deletes nothing"""
    monkeypatch.setattr("builtins.input", lambda _prompt="": answer)
    assert main(CLEAN_ARGV) == 0
    assert mock_clean.clean.call_args_list == [_clean_call(dry_run=True)]
    assert "Nothing deleted" in capsys.readouterr().out


@pytest.mark.usefixtures("mock_load")
@pytest.mark.parametrize("answer", ["y", "yes", " Y "])
def test_clean_confirmed_deletes(mock_clean, monkeypatch, answer):
    """Test that y/yes at the prompt previews, then cleans for real"""
    monkeypatch.setattr("builtins.input", lambda _prompt="": answer)
    assert main(CLEAN_ARGV) == 0
    assert mock_clean.clean.call_args_list == [
        _clean_call(dry_run=True),
        _clean_call(dry_run=False),
    ]


@pytest.mark.usefixtures("mock_load")
def test_clean_yes_skips_prompt(mock_clean, monkeypatch):
    """Test that --yes cleans without asking, passing every option through"""
    monkeypatch.setattr("builtins.input", _no_input)
    argv = CLEAN_ARGV + [
        "--members",
        "ens_0000",
        "--stages",
        "spinup_ad",
        "--include-archive",
        "--yes",
    ]
    assert main(argv) == 0
    options = {
        "members": ["ens_0000"],
        "stages": ["spinup_ad"],
        "include_archive": True,
    }
    assert mock_clean.clean.call_args_list == [
        _clean_call(dry_run=True, **options),
        _clean_call(dry_run=False, **options),
    ]


@pytest.mark.usefixtures("mock_load")
def test_clean_cancel_says_jobs_will_be_canceled(mock_clean, monkeypatch, capsys):
    """Test that --cancel is passed through and the summary says so"""
    monkeypatch.setattr("builtins.input", lambda _prompt="": "n")
    main(CLEAN_ARGV + ["--cancel"])
    assert mock_clean.clean.call_args_list == [_clean_call(dry_run=True, cancel=True)]
    assert "canceled first" in capsys.readouterr().out


@pytest.mark.usefixtures("mock_load")
def test_clean_nothing_to_clean(mock_ensemble, monkeypatch, capsys):
    """Test that an empty preview says so and neither asks nor cleans"""
    mock_ensemble.clean.return_value = []
    monkeypatch.setattr("builtins.input", _no_input)
    assert main(CLEAN_ARGV) == 0
    assert mock_ensemble.clean.call_count == 1
    assert "Nothing to clean" in capsys.readouterr().out


def test_clean_error_exits_one(mock_load, mock_ensemble, capsys):
    """Test that a refused clean (e.g. jobs still active) exits 1 with the error"""
    mock_ensemble.clean.side_effect = ValueError("jobs still queued or running")
    assert main(CLEAN_ARGV) == 1
    assert mock_load.called
    assert "ERROR: jobs still queued or running" in capsys.readouterr().err


# status


@pytest.mark.usefixtures("mock_load")
def test_status_prints_and_returns_zero(mock_ensemble, capsys):
    """Test that status prints format_status and returns 0 when nothing is lost"""
    assert main(["status", "--root", ROOT]) == 0
    mock_ensemble.status.assert_called_once_with(members=None, check_queue=False)
    mock_ensemble.format_status.assert_called_once_with(
        mock_ensemble.status.return_value, show_members=False
    )
    assert "Ensemble: my_ens-1" in capsys.readouterr().out


@pytest.mark.usefixtures("mock_load")
def test_status_passes_options(mock_ensemble):
    """Test that --members, --check-queue and --print-members are passed through"""
    main(
        [
            "status",
            "--root",
            ROOT,
            "--members",
            "a",
            "b",
            "--check-queue",
            "--print-members",
        ]
    )
    mock_ensemble.status.assert_called_once_with(members=["a", "b"], check_queue=True)
    assert mock_ensemble.format_status.call_args.kwargs == {"show_members": True}


@pytest.mark.usefixtures("mock_load")
@pytest.mark.parametrize(
    "statuses,code",
    [
        (["pending", "done"], 0),
        (["failed", "done"], 0),
        (["lost", "done"], EXIT_LOST),
        ([], 0),
    ],
)
def test_status_exit_code(mock_ensemble, statuses, code):
    """Test that status returns EXIT_LOST only when a stage is lost"""
    mock_ensemble.status.return_value = _status(statuses)
    assert main(["status", "--root", ROOT, "--check-queue"]) == code


@pytest.mark.usefixtures("mock_load")
def test_status_does_not_retry(mock_ensemble):
    """Test that status never resubmits anything"""
    mock_ensemble.status.return_value = _status(["lost"])
    main(["status", "--root", ROOT, "--check-queue"])
    mock_ensemble.retry.assert_not_called()
    mock_ensemble.submit.assert_not_called()


# manifest


@pytest.mark.usefixtures("mock_load")
def test_manifest_default_output(mock_ensemble, capsys):
    """Test that manifest writes <root>/manifest.csv by default"""
    assert main(["manifest", "--root", ROOT]) == 0
    out = mock_ensemble.root / "manifest.csv"
    assert pd.read_csv(out, dtype=str).equals(mock_ensemble.manifest.return_value)
    assert f"Wrote manifest: {out}" in capsys.readouterr().out


@pytest.mark.usefixtures("mock_load")
@pytest.mark.parametrize("flag", ["-o", "--output"])
def test_manifest_output_option(mock_ensemble, tmp_path, flag):
    """Test that -o/--output sets where the manifest is written"""
    out = tmp_path / "elsewhere.csv"
    main(["manifest", "--root", ROOT, flag, str(out)])
    assert out.is_file()
    assert not (mock_ensemble.root / "manifest.csv").exists()


@pytest.mark.usefixtures("mock_load")
def test_manifest_writes_no_index(mock_ensemble):
    """Test that the manifest CSV has no pandas index column"""
    main(["manifest", "--root", ROOT])
    header = (mock_ensemble.root / "manifest.csv").read_text().splitlines()[0]
    assert header == "member"


# error handling


def test_main_load_missing_root_exits_one(tmp_path, capsys):
    """Test that a missing root gives exit status 1 and an error on stderr"""
    assert main(["status", "--root", str(tmp_path / "nonexistent")]) == 1
    assert "ERROR" in capsys.readouterr().err


def test_main_keyboard_interrupt_exits_130(mocker, capsys):
    """Test that a keyboard interrupt exits with status 130"""
    mocker.patch("meristem.ensemble.cli.Ensemble.load", side_effect=KeyboardInterrupt)
    assert main(["status", "--root", ROOT]) == 130
    assert "Interrupted" in capsys.readouterr().err


def test_main_unexpected_exception_exits_one(mocker, capsys):
    """Test that an unexpected exception exits with status 1 and prints to stderr"""
    mocker.patch(
        "meristem.ensemble.cli.Ensemble.load", side_effect=RuntimeError("boom")
    )
    assert main(["status", "--root", ROOT]) == 1
    assert "ERROR: boom" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["create", "--config", "ens.yaml"],
        ["status", "--root", ROOT],
    ],
)
def test_main_debug_flag_reraises_exception(mocker, argv):
    """Test that --debug lets exceptions propagate rather than being caught"""
    mocker.patch(
        "meristem.ensemble.cli.Ensemble.create", side_effect=RuntimeError("boom")
    )
    mocker.patch(
        "meristem.ensemble.cli.Ensemble.load", side_effect=RuntimeError("boom")
    )
    with pytest.raises(RuntimeError, match="boom"):
        main(argv + ["--debug"])
