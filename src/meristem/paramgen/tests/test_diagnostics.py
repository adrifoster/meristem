"""Tests for param_ens_gen.diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
import pandas as pd

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from param_ens_gen.diagnostics import normalize_defaults, plot_param_bounds


def test_plot_param_bounds_returns_figure(ensemble_param_dir, default_param_file):
    """plot_param_bounds returns a matplotlib Figure."""
    df = normalize_defaults(
        ensemble_param_dir,
        default_param_file,
        param_list=["fates_leaf_slatop", "fates_canopy_closure_thresh"],
    )
    fig = plot_param_bounds(df)
    assert fig is not None
    plt.close(fig)


def test_plot_param_bounds_category_divider_drawn(
    ensemble_param_dir, default_param_file
):
    """plot_param_bounds draws a divider line between categories."""
    df = normalize_defaults(
        ensemble_param_dir,
        default_param_file,
        param_list=["fates_canopy_closure_thresh", "fates_leaf_slatop"],
    )
    # put biogeochemistry first so stomatal comes after — triggers category change
    df = df.sort_values("category").reset_index(drop=True)
    fig = plot_param_bounds(df)
    assert fig is not None
    plt.close(fig)


def test_normalize_defaults_returns_dataframe(
    ensemble_param_dir, default_param_file, posterior_config_file
):
    """normalize_defaults returns a DataFrame with correct columns."""
    df = normalize_defaults(
        ensemble_param_dir, default_param_file, posterior_config_file
    )
    assert isinstance(df, pd.DataFrame)
    assert "parameter" in df.columns
    assert "normalized_value" in df.columns


def test_normalize_defaults_has_row_per_parameter_index(
    ensemble_param_dir, default_param_file, posterior_config_file
):
    """normalize_defaults returns one row per parameter per index."""
    df = normalize_defaults(
        ensemble_param_dir, default_param_file, posterior_config_file
    )
    assert len(df) == 19


def test_normalize_defaults_param_list_subsets(ensemble_param_dir, default_param_file):
    """normalize_defaults respects param_list."""
    df = normalize_defaults(
        ensemble_param_dir,
        default_param_file,
        param_list=["fates_leaf_slatop", "fates_canopy_closure_thresh"],
    )
    # fates_leaf_slatop has 3 PFTs + 1 scalar = 4 rows
    assert len(df) == 4
    assert "fates_canopy_closure_thresh" in df["parameter"].values
    assert "fates_leaf_slatop_0" in df["parameter"].values


def test_normalize_defaults_unknown_param_raises(
    ensemble_param_dir, default_param_file
):
    """normalize_defaults raises ValueError for unknown parameter in param_list."""
    with pytest.raises(ValueError, match="not found in main.csv"):
        normalize_defaults(
            ensemble_param_dir,
            default_param_file,
            param_list=["not_a_real_param"],
        )


def test_normalize_defaults_missing_posterior_sources_raises(
    ensemble_param_dir, default_param_file, tmp_path
):
    """normalize_defaults raises FileNotFoundError for missing posterior sources."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        normalize_defaults(
            ensemble_param_dir,
            default_param_file,
            posterior_sources=tmp_path / "nonexistent.yaml",
        )


def test_normalize_defaults_json(
    ensemble_param_dir, json_param_file, posterior_config_file
):
    """normalize_defaults works with JSON parameter files."""
    df = normalize_defaults(ensemble_param_dir, json_param_file, posterior_config_file)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 19


def test_normalize_defaults_scalar_param_has_one_row(
    ensemble_param_dir, default_param_file
):
    """normalize_defaults returns one row for a scalar parameter."""
    df = normalize_defaults(
        ensemble_param_dir,
        default_param_file,
        param_list=["fates_canopy_closure_thresh"],
    )
    assert len(df) == 1
    assert df["parameter"].iloc[0] == "fates_canopy_closure_thresh"
    
def test_normalize_defaults_pft_index_returns_single_row(
    ensemble_param_dir, default_param_file
):
    """normalize_defaults with pft_index returns one row per parameter."""
    df = normalize_defaults(
        ensemble_param_dir,
        default_param_file,
        param_list=["fates_leaf_slatop"],
        pft_index=1,
    )
    assert len(df) == 1
    assert df["parameter"].iloc[0] == "fates_leaf_slatop"


def test_normalize_defaults_pft_param_has_one_row_per_pft(
    ensemble_param_dir, default_param_file
):
    """normalize_defaults returns one row per PFT for a PFT-dimensioned parameter."""
    df = normalize_defaults(
        ensemble_param_dir,
        default_param_file,
        param_list=["fates_leaf_slatop"],
    )
    assert len(df) == 3
    assert set(df["parameter"]) == {
        "fates_leaf_slatop_0",
        "fates_leaf_slatop_1",
        "fates_leaf_slatop_2",
    }


def test_normalize_defaults_uniform_values_in_unit_interval(
    ensemble_param_dir, default_param_file
):
    """normalize_defaults returns values in [0, 1] for uniform parameters."""
    df = normalize_defaults(
        ensemble_param_dir,
        default_param_file,
        param_list=["fates_leaf_slatop", "fates_canopy_closure_thresh"],
    )
    assert np.all(df["normalized_value"].values >= 0.0)
    assert np.all(df["normalized_value"].values <= 1.0)


def test_normalize_defaults_joint_param_has_row_per_base_param_per_index(
    ensemble_param_dir, default_param_file, posterior_config_file
):
    """normalize_defaults returns one row per base_param per index for joint parameters."""
    df = normalize_defaults(
        ensemble_param_dir,
        default_param_file,
        posterior_config_file,
        param_list=["fates_leafn_vert_scaler"],
    )
    assert len(df) == 6
    assert any(df["parameter"].str.startswith("fates_leafn_vert_scaler_coeff1"))
    assert any(df["parameter"].str.startswith("fates_leafn_vert_scaler_coeff2"))

def test_normalize_defaults_clips_out_of_bounds_defaults(
    tmp_path, default_param_file
):
    """normalize_defaults clips defaults outside bounds to 0.0 or 1.0."""
    # create a param where default is below min
    pd.DataFrame([{
        "parameter_name": "fates_leaf_slatop",
        "long_name": "SLA",
        "category": "stomatal",
        "subcategory": "photosynthesis",
        "units": "m^2/gC",
        "coord": "['fates_pft']",
        "param_type": "default",
        "strategy": "uniform",
        "param_min": "0.025",  # min higher than default value of 0.010
        "param_max": "0.05",
        "slice_dim": None,
        "slice_index": None,
        "root_param": None,
        "base_params": "",
    }]).to_csv(tmp_path / "main.csv", index=False)

    df = normalize_defaults(tmp_path, default_param_file)
    # default 0.010 is below min 0.025 so should be clipped to 0.0
    assert df["normalized_value"].iloc[0] == pytest.approx(0.0)
    assert df["normalized_value"].iloc[1] == pytest.approx(0.0)
    assert df["normalized_value"].iloc[2] == pytest.approx(0.2)


def test_normalize_defaults_filters_nan_values(
    tmp_path, default_param_file
):
    """normalize_defaults filters out nan default values."""
    pd.DataFrame([{
        "parameter_name": "fates_leaf_slatop",
        "long_name": "SLA",
        "category": "stomatal",
        "subcategory": "photosynthesis",
        "units": "m^2/gC",
        "coord": "['fates_pft']",
        "param_type": "default",
        "strategy": "uniform",
        "param_min": "0.005",
        "param_max": "0.05",
        "slice_dim": None,
        "slice_index": None,
        "root_param": None,
        "base_params": "",
    }]).to_csv(tmp_path / "main.csv", index=False)

    # default_ds has 3 PFTs, all non-nan, so all 3 rows should appear
    df = normalize_defaults(tmp_path, default_param_file)
    assert len(df) == 3
    assert not df["normalized_value"].isna().any()