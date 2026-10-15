"""Tests for meristem.commands and the `python -m meristem` entry point"""

import shlex
import subprocess
import sys

import meristem
from meristem.commands import meristem_cmd


def test_meristem_cmd_uses_running_interpreter():
    """Test that meristem_cmd() runs this install through the current Python"""
    assert shlex.split(meristem_cmd()) == [sys.executable, "-m", "meristem"]


def test_meristem_cmd_quotes_awkward_interpreter_path(monkeypatch):
    """Test that an interpreter path with a space stays a single shell word"""
    monkeypatch.setattr(sys, "executable", "/opt/my envs/bin/python")
    assert meristem_cmd() == "'/opt/my envs/bin/python' -m meristem"
    assert shlex.split(meristem_cmd())[0] == "/opt/my envs/bin/python"


def test_python_dash_m_meristem_runs():
    """Test that `python -m meristem --version` runs the meristem CLI"""
    result = subprocess.run(
        [sys.executable, "-m", "meristem", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert meristem.__version__ in result.stdout
