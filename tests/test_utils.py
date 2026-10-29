"""Tests for meristem.utils"""

from __future__ import annotations

from pathlib import Path

import pytest

from meristem.utils import check_email, check_id, to_absolute_path


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
