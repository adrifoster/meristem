"""Tests for meristem.paramgen.diagnostics."""

from __future__ import annotations

import json
import numpy as np
import pytest
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

from meristem.paramgen.diagnostics import (
    _expand_normalized,
    normalize_defaults,
    plot_param_bounds,
    read_ensemble_values,
)
from meristem.paramgen.parameter import Parameter
from meristem.paramgen.parameter_dataset import (
    NetCDFParameterDataset,
    ParameterDataset,
)


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


def test_normalize_defaults_clips_out_of_bounds_defaults(tmp_path, default_param_file):
    """normalize_defaults clips defaults outside bounds to 0.0 or 1.0."""
    # create a param where default is below min
    pd.DataFrame(
        [
            {
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
            }
        ]
    ).to_csv(tmp_path / "main.csv", index=False)

    df = normalize_defaults(tmp_path, default_param_file)
    # default 0.010 is below min 0.025 so should be clipped to 0.0
    assert df["normalized_value"].iloc[0] == pytest.approx(0.0)
    assert df["normalized_value"].iloc[1] == pytest.approx(0.0)
    assert df["normalized_value"].iloc[2] == pytest.approx(0.2)


def test_normalize_defaults_filters_nan_values(tmp_path, default_param_file):
    """normalize_defaults filters out nan default values."""
    pd.DataFrame(
        [
            {
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
            }
        ]
    ).to_csv(tmp_path / "main.csv", index=False)

    # default_ds has 3 PFTs, all non-nan, so all 3 rows should appear
    df = normalize_defaults(tmp_path, default_param_file)
    assert len(df) == 3
    assert not df["normalized_value"].isna().any()


def _two_dim_param(dims: list[str], pft_dim: str | None = None) -> Parameter:
    """A (2, 3) default parameter 'v' on the given dimensions."""
    ds = NetCDFParameterDataset(xr.Dataset({"v": (dims, np.ones((2, 3)))}))
    row = pd.Series(
        {
            "parameter_name": "v",
            "subcategory": "photosynthesis",
            "coord": str(dims),
            "param_type": "default",
            "strategy": "uniform",
            "param_min": "0",
            "param_max": "1",
            "pft_dim": pft_dim,
        }
    )
    return Parameter.from_row(row, ds)


def test_expand_normalized_labels_2d_by_dimension_index():
    """2-D values are labelled by their index along each dimension."""
    param = _two_dim_param(["leafage", "fates_pft"])
    rows = _expand_normalized(param, np.arange(6.0).reshape(2, 3))
    assert [(name, val) for name, val, _ in rows] == [
        ("v_0_0", 0.0),
        ("v_0_1", 1.0),
        ("v_0_2", 2.0),
        ("v_1_0", 3.0),
        ("v_1_1", 4.0),
        ("v_1_2", 5.0),
    ]


@pytest.mark.parametrize(
    "dims, pft_dim",
    [(["leafage", "fates_pft"], None), (["leafage", "plant_type"], "plant_type")],
)
def test_expand_normalized_pft_index_filters_along_pft_dim(dims, pft_dim):
    """pft_index selects along the PFT dimension, not by flat position."""
    param = _two_dim_param(dims, pft_dim=pft_dim)
    rows = _expand_normalized(param, np.arange(6.0).reshape(2, 3), pft_index=2)
    assert [(name, val) for name, val, _ in rows] == [("v_0", 2.0), ("v_1", 5.0)]


def test_expand_normalized_pft_index_keeps_non_pft_param_whole():
    """pft_index leaves a parameter without a PFT dimension unfiltered."""
    param = _two_dim_param(["leafage", "plant_type"])
    rows = _expand_normalized(param, np.arange(6.0).reshape(2, 3), pft_index=0)
    assert len(rows) == 6


def test_read_ensemble_values_matches_member_files(lh_ensemble):
    """read_ensemble_values returns each member's values as written to its file,
    keeping only the values that vary across members
    """
    lh_ensemble.create_ensemble()
    ens_dir = lh_ensemble.ensemble_dir
    df = read_ensemble_values(ens_dir, "test")

    assert list(df.index) == [f"test_{i:04d}" for i in range(5)]
    assert "fates_leaf_slatop_0" in df.columns
    assert (df.nunique() > 1).all()
    ext = lh_ensemble.default_ds.file_extension
    for member in df.index:
        ds = ParameterDataset.from_path(ens_dir / f"{member}{ext}")
        np.testing.assert_allclose(
            df.loc[member, ["fates_leaf_slatop_0", "fates_leaf_slatop_1"]],
            ds["fates_leaf_slatop"].values[:2],
        )
        ds.close()


def test_read_ensemble_values_skips_non_numeric(json_param_file, tmp_path):
    """Non-numeric variables are skipped when variables is None, but are read
    if requested explicitly.
    """
    with open(json_param_file, encoding="utf-8") as f:
        data = json.load(f)

    data["parameters"]["fates_pftname"] = {
        "dtype": "string",  # use whatever dtype ParameterDataset expects for text
        "dims": ["fates_pft"],
        "long_name": "PFT name",
        "units": "unitless",
        "data": ["pft_a", "pft_b", "pft_c"],
    }

    ens_dir = tmp_path / "string_ensemble"
    ens_dir.mkdir()
    with open(ens_dir / "member_0000.json", "w", encoding="utf-8") as f:
        json.dump(data, f)
    (ens_dir / "test.txt").write_text("member_0000\n")

    # variables=None: the string variable is skipped, numeric ones are kept
    df = read_ensemble_values(ens_dir, "test", drop_constant=False)
    assert not any(col.startswith("fates_pftname") for col in df.columns)
    assert "fates_leaf_slatop_0" in df.columns

    # explicit request bypasses the skip
    df_explicit = read_ensemble_values(
        ens_dir, "test", variables=["fates_pftname"], drop_constant=False
    )
    assert "fates_pftname_0" in df_explicit.columns


def test_read_ensemble_values_variables_and_keep_constant(lh_ensemble):
    """read_ensemble_values reads only the listed variables and, with
    drop_constant=False, keeps every element
    """
    lh_ensemble.create_ensemble()
    df = read_ensemble_values(
        lh_ensemble.ensemble_dir,
        "test",
        variables=["fates_leaf_slatop"],
        drop_constant=False,
    )
    assert list(df.columns) == [
        "fates_leaf_slatop_0",
        "fates_leaf_slatop_1",
        "fates_leaf_slatop_2",
    ]


def test_read_ensemble_values_missing_member_file_raises(tmp_path):
    """read_ensemble_values raises when a listed member has no parameter file."""
    (tmp_path / "test.txt").write_text("test_0000\n")
    with pytest.raises(FileNotFoundError, match="test_0000"):
        read_ensemble_values(tmp_path, "test")
