"""Tests for the top-level meristem command (meristem.cli)"""

import importlib
import sys

import pytest

import meristem
from meristem.cli import main


def test_no_command_exits_with_usage_error(capsys):
    """Test that meristem with no command exits with code 2 and lists the commands"""
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
    assert "COMMAND" in capsys.readouterr().err


def test_unknown_command_exits_with_usage_error():
    """Test that an unknown command exits with code 2"""
    with pytest.raises(SystemExit) as exc:
        main(["notacommand"])
    assert exc.value.code == 2


def test_help_lists_commands(capsys):
    """Test that meristem --help lists both commands"""
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "paramgen" in out
    assert "run" in out


def test_version(capsys):
    """Test that meristem --version prints the package version"""
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert meristem.__version__ in capsys.readouterr().out


@pytest.mark.parametrize(
    "command,module,rest",
    [
        ("paramgen", "meristem.paramgen.cli", ["config.yaml"]),
        ("run", "meristem.run.cli", ["status", "--root", "/tmp/run"]),
    ],
)
def test_dispatches_remaining_args_to_subcommand(mocker, command, module, rest):
    """Test that everything after the command is passed to that command's main()"""
    importlib.import_module(module)
    sub_main = mocker.patch(f"{module}.main", return_value=0)
    assert main([command, *rest]) == 0
    sub_main.assert_called_once_with(rest)


def test_returns_subcommand_exit_code(mocker):
    """Test that the subcommand's exit code is returned unchanged"""
    importlib.import_module("meristem.run.cli")
    mocker.patch("meristem.run.cli.main", return_value=1)
    assert main(["run", "status", "--root", "/tmp/run"]) == 1


def test_subcommand_help_reaches_subcommand_parser(capsys):
    """Test that `meristem run --help` shows the run command's own options"""
    with pytest.raises(SystemExit) as exc:
        main(["run", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "usage: meristem run" in out
    assert "create" in out
    assert "submit-advance" in out


def test_run_does_not_import_paramgen(mocker):
    """Test that `meristem run` doesn't import paramgen (keeps PBS job calls fast)"""
    mocker.patch.dict(sys.modules)  # restore sys.modules after the test
    for name in [m for m in sys.modules if m.startswith("meristem.paramgen")]:
        del sys.modules[name]
    importlib.import_module("meristem.run.cli")
    mocker.patch("meristem.run.cli.main", return_value=0)
    main(["run", "status", "--root", "/tmp/run"])
    assert not any(m.startswith("meristem.paramgen") for m in sys.modules)
