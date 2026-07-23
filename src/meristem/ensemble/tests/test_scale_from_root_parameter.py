"""Tests for ScaleFromRootParam class"""

from __future__ import annotations

import numpy as np
import pytest

from param_ens_gen.parameter import DimIndex, Parameter


def test_validate_specs_passes_for_scale(scale_from_root_row, param_dataset):
    """_validate_specs() does not raise for a correctly configured ScaleFromRootParameter."""
    Parameter.from_row(scale_from_root_row, param_dataset)


def test_scale_from_root_missing_root_param_raises(default_row, param_dataset):
    """ScaleFromRootParam raises if missing the root parameter"""
    default_row["parameter_name"] = "smpsc_delta"
    default_row["coord"] = "['fates_pft']"
    default_row["param_type"] = "scale_from_root"
    default_row["param_min"] = "-600000"
    default_row["param_max"] = "-20000"
    default_row["root_param"] = ""
    default_row["base_params"] = "fates_nonhydro_smpsc"

    with pytest.raises(ValueError, match="root_param"):
        Parameter.from_row(default_row, param_dataset)


def test_scale_from_root_missing_base_param_raises(default_row, param_dataset):
    """ScaleFromRootParam raises if missing the base parameter"""
    default_row["parameter_name"] = "smpsc_delta"
    default_row["coord"] = "['fates_pft']"
    default_row["param_type"] = "scale_from_root"
    default_row["param_min"] = "-600000"
    default_row["param_max"] = "-20000"
    default_row["root_param"] = "fates_nonhydro_smpso"
    default_row["base_params"] = ""

    with pytest.raises(ValueError, match="base_param"):
        Parameter.from_row(default_row, param_dataset)


def test_scale_from_root_validate_multiple_base_params_raises(
    default_row, param_dataset
):
    """ScaleFromRootParam raises for multiple base_params"""

    default_row["parameter_name"] = "fates_leaf_vcmax25top"
    default_row["parameter_name"] = "['fates_leafage_class', 'fates_pft']"
    default_row["param_type"] = "scale_from_root"
    default_row["param_min"] = "20.0"
    default_row["param_max"] = "90.0"
    default_row["root_param"] = "fates_nonhydro_smpso"
    default_row["base_params"] = "['fates_leaf_vcmax25top', 'fates_leaf_slatop']"

    with pytest.raises(ValueError, match="Too many base_params"):
        Parameter.from_row(default_row, param_dataset)


def test_scale_from_root_get_default(scale_from_root_row, param_dataset):
    """ScaleFromRootParameter.get_default returns root parameter values."""
    param = Parameter.from_row(scale_from_root_row, param_dataset)
    result = param.get_default(param_dataset)
    np.testing.assert_allclose(result, [-100000.0, -110000.0, -120000.0])


def test_scale_from_root_set_value_no_fixed(
    scale_from_root_row, param_dataset, working_param_dataset
):
    """ScaleFromRootParameter.set_value adds delta to root values for all indices."""
    param = Parameter.from_row(scale_from_root_row, param_dataset)
    delta = -10000.0
    param.set_value(working_param_dataset, param_dataset, value=delta)

    expected = np.array([-50000.0, -60000.0, -70000.0]) + delta
    np.testing.assert_allclose(
        working_param_dataset["fates_nonhydro_smpsc"].values, expected
    )


def test_scale_from_root_set_value_with_fixed(
    scale_from_root_row, param_dataset, working_param_dataset
):
    """ScaleFromRootParameter.set_value leaves fixed indices at their default values."""
    param = Parameter.from_row(scale_from_root_row, param_dataset)
    param.set_value(
        working_param_dataset,
        param_dataset,
        value=-10000.0,
        fixed_indices={"fates_pft": [0]},
    )
    arr = working_param_dataset["fates_nonhydro_smpsc"].values
    assert arr[0] == pytest.approx(-100000.0)  # fixed — default value
    assert arr[1] == pytest.approx(-70000.0)  # root + delta
    assert arr[2] == pytest.approx(-80000.0)  # root + delta


def test_scale_from_root_set_value_with_active_index(
    scale_from_root_row, param_dataset, working_param_dataset
):
    """ScaleFromRootParameter.set_value writes delta only at active_index."""
    param = Parameter.from_row(scale_from_root_row, param_dataset)
    param.active_index = DimIndex(dim="fates_pft", index=0)
    param.set_value(working_param_dataset, param_dataset, value=-5000.0)

    arr = working_param_dataset["fates_nonhydro_smpsc"].values
    assert arr[0] == pytest.approx(-55000.0)  # root[0] + delta
    assert arr[1] == pytest.approx(-110000.0)  # unchanged
    assert arr[2] == pytest.approx(-120000.0)  # unchanged


def test_scale_from_root_set_value_adds_delta_to_root(
    scale_param, working_param_dataset, param_dataset
):
    """ScaleFromRootParameter.set_value correctly adds delta to the root param"""
    # simulate driver having written root already
    working_param_dataset["fates_nonhydro_smpso"].values[:] = np.array(
        [-55000.0, -65000.0, -75000.0]
    )
    scale_param.active_index = DimIndex("fates_pft", 1)
    scale_param.set_value(working_param_dataset, param_dataset, -50000.0)
    # index 1: root[1] + delta = -65000 + -50000 = -115000
    assert working_param_dataset["fates_nonhydro_smpsc"].values[1] == pytest.approx(
        -115000.0
    )


def test_scale_from_root_set_value_other_slots_unchanged(
    scale_param, working_param_dataset, param_dataset
):
    """ScaleFromRootParameter.set_value correctly writes only to the active index"""
    working_param_dataset["fates_nonhydro_smpso"].values[:] = np.array(
        [-55000.0, -65000.0, -75000.0]
    )
    scale_param.active_index = DimIndex("fates_pft", 1)
    scale_param.set_value(working_param_dataset, param_dataset, -50000.0)
    assert working_param_dataset["fates_nonhydro_smpsc"].values[0] == pytest.approx(
        param_dataset["fates_nonhydro_smpsc"].values[0]
    )
    assert working_param_dataset["fates_nonhydro_smpsc"].values[2] == pytest.approx(
        param_dataset["fates_nonhydro_smpsc"].values[2]
    )


def test_scale_from_root_set_value_reads_written_root_not_default(
    scale_param, working_param_dataset, param_dataset
):
    """Root must be read from working_param_dataset, not param_dataset."""
    working_param_dataset["fates_nonhydro_smpso"].values[:] = -999000.0
    scale_param.active_index = DimIndex("fates_pft", 1)
    scale_param.set_value(working_param_dataset, param_dataset, -1000.0)
    assert working_param_dataset["fates_nonhydro_smpsc"].values[1] == pytest.approx(
        -1000000.0
    )


def test_scale_from_root_set_value_scalar_delta(
    scale_param, working_param_dataset, param_dataset
):
    """ScaleFromRootParameter.set_value can set a scalar delta"""
    working_param_dataset["fates_nonhydro_smpso"].values[:] = np.array(
        [-55000.0, -65000.0, -75000.0]
    )
    scale_param.set_value(working_param_dataset, param_dataset, -50000.0, None)
    np.testing.assert_allclose(
        working_param_dataset["fates_nonhydro_smpsc"].values,
        [-105000.0, -115000.0, -125000.0],
    )


def test_write_full_array_delta(scale_param, working_param_dataset, param_dataset):
    """ScaleFromRoot parameter.set_value can set a whole array"""
    working_param_dataset["fates_nonhydro_smpso"].values[:] = np.array(
        [-55000.0, -65000.0, -75000.0]
    )
    delta = np.array([-10000.0, -20000.0, -30000.0])
    scale_param.set_value(working_param_dataset, param_dataset, delta, None)
    np.testing.assert_allclose(
        working_param_dataset["fates_nonhydro_smpsc"].values,
        [-65000.0, -85000.0, -105000.0],
    )


def test_write_full_fixed_indices_held_at_default(
    scale_param, working_param_dataset, param_dataset
):
    """ScaleFromRoot.set_value can set an array with fixed indices held"""
    working_param_dataset["fates_nonhydro_smpso"].values[:] = np.array(
        [-55000.0, -65000.0, -75000.0]
    )
    scale_param.set_value(
        working_param_dataset, param_dataset, -50000.0, {"fates_pft": [1]}
    )
    assert working_param_dataset["fates_nonhydro_smpsc"].values[0] == pytest.approx(
        -105000.0
    )
    assert working_param_dataset["fates_nonhydro_smpsc"].values[1] == pytest.approx(
        param_dataset["fates_nonhydro_smpsc"].values[1]
    )
    assert working_param_dataset["fates_nonhydro_smpsc"].values[2] == pytest.approx(
        -125000.0
    )


def test_write_full_reads_written_root_not_default(
    scale_param, working_param_dataset, param_dataset
):
    """Root must be read from working_param_dataset, not param_dataset."""
    working_param_dataset["fates_nonhydro_smpso"].values[:] = -999000.0
    scale_param.set_value(working_param_dataset, param_dataset, -1000.0, None)
    np.testing.assert_allclose(
        working_param_dataset["fates_nonhydro_smpsc"].values,
        [-1000000.0, -1000000.0, -1000000.0],
    )


def test_param_dataset_not_mutated(scale_param, working_param_dataset, param_dataset):
    """ScaleFromRootParameter.set_value does not mutate param_dataset"""
    original_smpso = param_dataset["fates_nonhydro_smpso"].values.copy()
    original_smpsc = param_dataset["fates_nonhydro_smpsc"].values.copy()
    working_param_dataset["fates_nonhydro_smpso"].values[:] = np.array(
        [-55000.0, -65000.0, -75000.0]
    )
    scale_param.set_value(
        working_param_dataset, param_dataset, -50000.0, {"fates_pft": [0]}
    )
    np.testing.assert_array_equal(
        param_dataset["fates_nonhydro_smpso"].values, original_smpso
    )
    np.testing.assert_array_equal(
        param_dataset["fates_nonhydro_smpsc"].values, original_smpsc
    )


def test_scale_from_root_get_default_returns_scalar_when_expanded(
    scale_from_root_row, param_dataset
):
    """get_default returns a scalar when active_index is set."""
    param = Parameter.from_row(scale_from_root_row, param_dataset)
    param.active_index = DimIndex("fates_pft", 1)
    result = param.get_default(param_dataset)
    assert np.ndim(result) == 0
    assert float(result) == pytest.approx(-110000.0)


def test_scale_from_root_get_default_returns_array_when_not_expanded(
    scale_from_root_row, param_dataset
):
    """get_default returns full array when active_index is None."""
    param = Parameter.from_row(scale_from_root_row, param_dataset)
    result = param.get_default(param_dataset)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
