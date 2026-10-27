"""Test for __main___"""

import runpy
import pytest

def test_python_m_meristem_calls_main(monkeypatch):
    """Simple test that CLI works"""
    monkeypatch.setattr("meristem.cli.main", lambda: 3)

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("meristem", run_name="__main__")

    assert exc.value.code == 3