"""Tests for meristem.utils"""

from __future__ import annotations

from pathlib import Path

import pytest

from meristem.utils import to_absolute_path


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
