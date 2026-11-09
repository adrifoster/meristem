"""Tests for utility functions"""

from pathlib import Path

import pytest
import pandas as pd
from meristem.paramgen.utils import read_param_list, read_posterior_sources


def test_read_param_list_returns_main(param_dir):
    """Test that read_param_list actually returns the main list"""
    main, _ = read_param_list(param_dir)
    assert isinstance(main, pd.DataFrame)
    assert "parameter_name" in main.columns


def test_read_param_list_returns_pft_sheets(param_dir):
    """Test that read_param_list actually returns the pft_sheets"""
    _, pft_sheets = read_param_list(param_dir)
    assert "fates_leaf_slatop" in pft_sheets
    assert isinstance(pft_sheets["fates_leaf_slatop"], pd.DataFrame)


def test_read_param_list_no_pft_sheets(tmp_path):
    """Test that read_param_list returns an empty dict if there are no pft sheets"""
    pd.DataFrame({"parameter_name": ["x"], "param_type": ["default"]}).to_csv(
        tmp_path / "main.csv", index=False
    )
    _, pft_sheets = read_param_list(tmp_path)
    assert not pft_sheets


def test_read_param_list_missing_dir():
    """Test that read_param_list raises if it can't find the param_dir"""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        read_param_list(Path("/nonexistent/path"))


def test_read_param_list_not_a_directory(tmp_path):
    """Test that read_param_list raises if param_dir isn't a directory"""
    f = tmp_path / "not_a_dir.csv"
    f.write_text("x")
    with pytest.raises(FileNotFoundError, match="not a directory"):
        read_param_list(f)


def test_read_param_list_missing_main(tmp_path):
    """Test that read_param_list raises if it can't find main.csv"""
    with pytest.raises(FileNotFoundError, match="main.csv"):
        read_param_list(tmp_path)


def test_read_param_list_empty_main(tmp_path):
    """Test that read_param_list raises if main.csv is empty"""
    (tmp_path / "main.csv").write_text("")
    with pytest.raises(ValueError, match="empty"):
        read_param_list(tmp_path)


def test_read_param_list_headers_only_main(tmp_path):
    """Test read_param_list raises if main is just columns"""
    (tmp_path / "main.csv").write_text("parameter_name,param_type\n")
    with pytest.raises(ValueError, match="empty"):
        read_param_list(tmp_path)


def test_read_posterior_sources_paths_relative_to_yaml(tmp_path, monkeypatch):
    """Test that relative posterior file paths are relative to the posterior
    sources YAML's folder, not the current directory; absolute files are kept
    """
    sources_dir = tmp_path / "posteriors"
    sources_dir.mkdir()
    yaml_file = sources_dir / "posterior_sources.yaml"
    yaml_file.write_text(
        "fates_leafn_vert_scaler:\n"
        "  parameters: [coeff1, coeff2]\n"
        "  files:\n"
        "    - path: samples.txt\n"
        "      array_indices: all\n"
        "    - path: /abs/samples.txt\n"
        "      array_indices: [0]\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    files = read_posterior_sources(yaml_file)["fates_leafn_vert_scaler"]["files"]
    assert Path(files[0]["path"]) == sources_dir / "samples.txt"
    assert Path(files[1]["path"]) == Path("/abs/samples.txt")
