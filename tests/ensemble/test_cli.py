"""Tests for meristem.ensemble.cli"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from meristem.ensemble.cli import (
    EXIT_LOST,
    build_parser,
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


@pytest.mark.parametrize("command", ["retry", "status"])
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


@pytest.mark.parametrize("command", ["retry", "status"])
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
