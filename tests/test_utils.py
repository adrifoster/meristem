"""Tests for meristem.utils"""

from __future__ import annotations

import json
import os
import stat
import threading
from pathlib import Path

import pytest

from meristem.utils import (
    check_email,
    check_id,
    to_absolute_path,
    write_atomically,
)


def test_to_absolute_path_none_passes():
    """Test that None (an unset optional field) is returned as None"""
    assert to_absolute_path("output_root", None) is None


@pytest.mark.parametrize("path", ["/abs/dir", Path("/abs/dir")])
def test_to_absolute_path_returns_path(path):
    """Test that an absolute str or Path comes back as an equal Path"""
    result = to_absolute_path("root", path)
    assert isinstance(result, Path)
    assert result == Path("/abs/dir")


@pytest.mark.parametrize("path", ["relative/dir", Path("relative/dir")])
def test_to_absolute_path_relative_raises(path):
    """Test that a relative str or Path raises, naming the field and path,
    with no environment variable hint
    """
    with pytest.raises(
        ValueError, match=r"^root 'relative/dir' must be an absolute path\.$"
    ):
        to_absolute_path("root", path)


def test_to_absolute_path_empty_raises():
    """Test that an empty string raises, rather than becoming "." and being
    reported as relative
    """
    with pytest.raises(ValueError, match=r"^root must not be empty\.$"):
        to_absolute_path("root", "")


@pytest.mark.parametrize("path", [[], {}, 3])
def test_to_absolute_path_not_a_path_raises(path):
    """Test that a value that isn't a str or Path raises a ValueError"""
    with pytest.raises(ValueError, match=r"^root must be a path, got "):
        to_absolute_path("root", path)


def test_to_absolute_path_where_in_message():
    """Test that where is added after the path in the error message"""
    with pytest.raises(
        ValueError,
        match=r"^script 'setup\.sh' for stage 'ad' must be an absolute path\.$",
    ):
        to_absolute_path("script", "setup.sh", where=" for stage 'ad'")


def test_to_absolute_path_env_var_hint():
    """Test that a relative path still holding a $ hints at an unset variable"""
    with pytest.raises(
        ValueError,
        match=r"^root '\$UNSET/dir' must be an absolute path "
        r"\(is an environment variable unset\?\)\.$",
    ):
        to_absolute_path("root", "$UNSET/dir")


@pytest.mark.parametrize("value", ["run", "member_0042", "my-spinup", "AD-1", "x"])
def test_check_id_valid_passes(value):
    """Test that letters, digits, hyphens and underscores are allowed"""
    check_id("stage name", value)


@pytest.mark.parametrize(
    "value",
    [
        'x"; touch PWNED; echo "',  # shell code in a job script
        "$(whoami)",
        "../escaped",  # path outside the run
        "a/b",
        "a b",  # breaks #PBS -N
        "a.b",
        "a\tb",
        "a\nb",
    ],
)
def test_check_id_invalid_characters_raises(value):
    """Test that anything else is rejected, naming the field and value"""
    with pytest.raises(ValueError) as err:
        check_id("stage name", value, where=" in run 'r'")
    assert str(err.value) == (
        f"stage name {value!r} in run 'r' contains invalid characters. "
        "Only letters, digits, hyphens, and underscores are allowed."
    )


def test_check_id_empty_raises():
    """Test that an empty identifier is rejected"""
    with pytest.raises(ValueError, match=r"^run_id must not be empty\.$"):
        check_id("run_id", "")


@pytest.mark.parametrize("value", [None, 1, ["a"]])
def test_check_id_non_string_raises(value):
    """Test that a non-string (e.g. an unquoted number in YAML) is rejected"""
    with pytest.raises(ValueError, match="stage name must be a string"):
        check_id("stage name", value)


@pytest.mark.parametrize(
    "value",
    ["name@ucar.edu", "first.last+pbs@ncar.ucar.edu", "a@b.org,c@d.org"],
)
def test_check_email_valid_passes(value):
    """Test that one address, or a comma-separated list, is allowed"""
    check_email("email", value)


@pytest.mark.parametrize(
    "value, bad",
    [
        ("name@ucar.edu -l select=100", ["name@ucar.edu -l select=100"]),
        ("name@ucar.edu\ntouch PWNED", ["name@ucar.edu\ntouch PWNED"]),
        ('"name"@ucar.edu', ['"name"@ucar.edu']),
        ("name", ["name"]),
        ("a@b.org, c@d.org", [" c@d.org"]),  # PBS lists have no spaces
        ("a@b.org,", [""]),
        ("", [""]),
    ],
)
def test_check_email_invalid_raises(value, bad):
    """Test that anything that isn't a name@host list is rejected, naming the
    bad entries
    """
    with pytest.raises(ValueError) as err:
        check_email("email", value)
    assert str(err.value).endswith(f"not valid: {bad}.")


def test_check_email_non_string_raises():
    """Test that a non-string email is rejected"""
    with pytest.raises(ValueError, match="email must be a string"):
        check_email("email", ["name@ucar.edu"])


def test_write_atomically_writes_and_replaces(tmp_path):
    """Test that write_atomically creates the file, then replaces its contents"""
    path = tmp_path / "state.json"
    write_atomically(path, "first")
    assert path.read_text(encoding="utf-8") == "first"
    write_atomically(path, "second")
    assert path.read_text(encoding="utf-8") == "second"
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_write_atomically_uses_umask_permissions(tmp_path):
    """Test that the file gets the usual permissions (not the temp file's 0600),
    so others who can read the directory can read the state
    """
    old = os.umask(0o022)
    try:
        write_atomically(tmp_path / "state.json", "x")
    finally:
        os.umask(old)
    assert stat.S_IMODE((tmp_path / "state.json").stat().st_mode) == 0o644


def test_write_atomically_failure_keeps_original(tmp_path, monkeypatch):
    """Test that a failure partway leaves the original file untouched and no
    temp file behind
    """
    path = tmp_path / "state.json"
    path.write_text("original", encoding="utf-8")

    def _fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", _fail)
    with pytest.raises(OSError, match="disk full"):
        write_atomically(path, "new")
    assert path.read_text(encoding="utf-8") == "original"
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_write_atomically_concurrent_writers(tmp_path):
    """Test that many writers saving the same file at once never fail and never
    leave it corrupt: it's always one writer's complete contents
    """
    path = tmp_path / "state.json"
    contents = [json.dumps({"writer": i, "pad": "x" * 50_000}) for i in range(8)]
    errors = []

    def _writer(text):
        try:
            for _ in range(50):
                write_atomically(path, text)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            errors.append(exc)

    threads = [threading.Thread(target=_writer, args=(text,)) for text in contents]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert path.read_text(encoding="utf-8") in contents
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]
