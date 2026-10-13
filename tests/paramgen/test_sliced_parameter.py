"""Tests for SlicedParameter class"""

from __future__ import annotations

import numpy as np
import pytest

from param_ens_gen.parameter import DimIndex, Parameter


def test_sliced_param_validate_specs_passes(sliced_row, param_dataset):
    """_validate_specs() does not raise for a correctly configured SlicedParameter."""
    Parameter.from_row(sliced_row, param_dataset)


def test_sliced_param_validate_specs_missing_slice_dim_raises(
    default_row, param_dataset
):
    """_validate_specs() raises if slice_dim is missing"""

    default_row["parameter_name"] = "fates_leaf_vcmax25top"
    default_row["parameter_name"] = "['fates_leafage_class', 'fates_pft']"
    default_row["param_type"] = "sliced"
    default_row["param_min"] = "20.0"
    default_row["param_max"] = "90.0"
    default_row["slice_dim"] = ""
    default_row["slice_index"] = "0"
    default_row["base_params"] = "fates_leaf_vcmax25top"

    with pytest.raises(ValueError, match="slice_dim"):
        Parameter.from_row(default_row, param_dataset)


def test_sliced_param_validate_specs_missing_slice_index_raises(
    default_row, param_dataset
):
    """_validate_specs() raises if slice_index is missing"""

    default_row["parameter_name"] = "fates_leaf_vcmax25top"
    default_row["parameter_name"] = "['fates_leafage_class', 'fates_pft']"
    default_row["param_type"] = "sliced"
    default_row["param_min"] = "20.0"
    default_row["param_max"] = "90.0"
    default_row["slice_dim"] = "fates_leafage_class"
    default_row["slice_index"] = None
    default_row["base_params"] = "fates_leaf_vcmax25top"

    with pytest.raises(ValueError, match="slice_index"):
        Parameter.from_row(default_row, param_dataset)


def test_sliced_param_validate_specs_missing_base_params_raises(
    default_row, param_dataset
):
    """_validate_specs() raises if base_params is missing"""

    default_row["parameter_name"] = "fates_leaf_vcmax25top"
    default_row["parameter_name"] = "['fates_leafage_class', 'fates_pft']"
    default_row["param_type"] = "sliced"
    default_row["param_min"] = "20.0"
    default_row["param_max"] = "90.0"
    default_row["slice_dim"] = "fates_leafage_class"
    default_row["slice_index"] = "0"
    default_row["base_params"] = None

    with pytest.raises(ValueError, match="base_params"):
        Parameter.from_row(default_row, param_dataset)


def test_sliced_param_validate_multiple_base_params_raises(default_row, param_dataset):
    """_validate_specs() raises for multiple base_params"""

    default_row["parameter_name"] = "fates_leaf_vcmax25top"
    default_row["parameter_name"] = "['fates_leafage_class', 'fates_pft']"
    default_row["param_type"] = "sliced"
    default_row["param_min"] = "20.0"
    default_row["param_max"] = "90.0"
    default_row["slice_dim"] = "fates_leafage_class"
    default_row["slice_index"] = "0"
    default_row["base_params"] = "['fates_leaf_vcmax25top', 'fates_leaf_slatop']"

    with pytest.raises(ValueError, match="Too many base_params"):
        Parameter.from_row(default_row, param_dataset)


def test_sliced_get_default(sliced_row, param_dataset):
    """SlicedParameter.get_default returns the slice at slice_index."""
    param = Parameter.from_row(sliced_row, param_dataset)
    result = param.get_default(param_dataset)
    np.testing.assert_allclose(result, [50.0, 60.0, 70.0])


def test_sliced_get_default_returns_scalar_when_expanded(sliced_row, param_dataset):
    """get_default returns a scalar when active_index is set."""
    param = Parameter.from_row(sliced_row, param_dataset)
    param.active_index = DimIndex("fates_pft", 1)
    result = param.get_default(param_dataset)
    assert np.ndim(result) == 0
    assert float(result) == pytest.approx(60.0)


def test_sliced_get_default_returns_array_when_not_expanded(sliced_row, param_dataset):
    """get_default returns full slice array when active_index is None."""
    param = Parameter.from_row(sliced_row, param_dataset)
    result = param.get_default(param_dataset)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)


def test_sliced_param_set_value_active_index_sets_correct_element(
    sliced_param, working_param_dataset, param_dataset
):
    """SlicedParameter.set_value with an active index sets the correct element"""
    sliced_param.active_index = DimIndex("fates_pft", 1)
    sliced_param.set_value(working_param_dataset, param_dataset, 99.0, None)
    result = working_param_dataset["fates_leaf_vcmax25top"].values
    assert result[0, 1] == pytest.approx(99.0)


def test_sliced_param_set_value_active_index_leaves_other_indices_unchanged(
    sliced_param, working_param_dataset, param_dataset
):
    """SlicedParameter.set_value with an active index doesn't modify other indices"""
    sliced_param.active_index = DimIndex("fates_pft", 1)
    sliced_param.set_value(working_param_dataset, param_dataset, 99.0, None)
    result = working_param_dataset["fates_leaf_vcmax25top"].values
    assert result[0, 0] == pytest.approx(
        param_dataset["fates_leaf_vcmax25top"].values[0, 0]
    )
    assert result[0, 2] == pytest.approx(
        param_dataset["fates_leaf_vcmax25top"].values[0, 2]
    )
    # other leafage_class row entirely untouched
    np.testing.assert_array_equal(
        result[1, :], param_dataset["fates_leaf_vcmax25top"].values[1, :]
    )


def test_sliced_param_set_value_only_modifies_target_slice(
    sliced_param, working_param_dataset, param_dataset
):
    """SlicedParameter.set_value without active index modifies the correct slice"""
    sliced_param.set_value(working_param_dataset, param_dataset, 75.0, None)
    result = working_param_dataset["fates_leaf_vcmax25top"].values
    np.testing.assert_array_equal(result[0, :], [75.0, 75.0, 75.0])
    # leafage_class row 1 untouched
    np.testing.assert_array_equal(
        result[1, :], param_dataset["fates_leaf_vcmax25top"].values[1, :]
    )


def test_sliced_param_set_value_fixed_indices_respected(
    sliced_param, working_param_dataset, param_dataset
):
    """SlicedParameter.set_value correctly sets with fixed indices"""
    sliced_param.set_value(
        working_param_dataset, param_dataset, 75.0, {"fates_pft": [1]}
    )
    result = working_param_dataset["fates_leaf_vcmax25top"].values
    assert result[0, 0] == pytest.approx(75.0)
    assert result[0, 1] == pytest.approx(
        param_dataset["fates_leaf_vcmax25top"].values[0, 1]
    )
    assert result[0, 2] == pytest.approx(75.0)


def test_sliced_param_set_value_array_value(
    sliced_param, working_param_dataset, param_dataset
):
    """SlicedParameter.set_value can set a whole array"""
    value = np.array([21.0, 32.0, 43.0])
    sliced_param.set_value(working_param_dataset, param_dataset, value, None)
    np.testing.assert_array_equal(
        working_param_dataset["fates_leaf_vcmax25top"].values[0, :], value
    )


def test_sliced_param_set_value_ds_not_mutated(
    sliced_param, working_param_dataset, param_dataset
):
    """SlicedParameter.set_value doesn't modify the default ds"""
    original = param_dataset["fates_leaf_vcmax25top"].values.copy()
    sliced_param.set_value(
        working_param_dataset, param_dataset, 75.0, {"fates_pft": [0]}
    )
    np.testing.assert_array_equal(
        param_dataset["fates_leaf_vcmax25top"].values, original
    )


def test_sliced_set_value_no_fixed(sliced_row, param_dataset, working_param_dataset):
    """SlicedParameter.set_value writes value across all indices at the slice."""
    param = Parameter.from_row(sliced_row, param_dataset)
    param.set_value(working_param_dataset, param_dataset, value=80.0)

    arr = working_param_dataset["fates_leaf_vcmax25top"].values
    # slice_index=0 (leafage_class dim) should be updated
    np.testing.assert_allclose(arr[0, :], [80.0, 80.0, 80.0])
    # other leafage_class rows unchanged
    np.testing.assert_allclose(arr[1, :], [40.0, 50.0, 60.0])


def test_sliced_set_value_with_fixed_indices(
    sliced_row, param_dataset, working_param_dataset
):
    """SlicedParameter.set_value leaves fixed PFTs at their default values."""
    param = Parameter.from_row(sliced_row, param_dataset)
    param.set_value(
        working_param_dataset,
        param_dataset,
        value=80.0,
        fixed_indices={"fates_pft": [0]},
    )
    arr = working_param_dataset["fates_leaf_vcmax25top"].values
    assert arr[0, 0] == pytest.approx(50.0)  # fixed — unchanged
    assert arr[0, 1] == pytest.approx(80.0)  # written
    assert arr[0, 2] == pytest.approx(80.0)  # written


def test_sliced_set_value_with_active_index(
    sliced_row, param_dataset, working_param_dataset
):
    """SlicedParameter.set_value writes only to (slice_index, active_index) cell."""
    param = Parameter.from_row(sliced_row, param_dataset)
    param.active_index = DimIndex(dim="fates_pft", index=1)
    param.set_value(working_param_dataset, param_dataset, value=99.0)

    arr = working_param_dataset["fates_leaf_vcmax25top"].values
    assert arr[0, 0] == pytest.approx(50.0)  # unchanged
    assert arr[0, 1] == pytest.approx(99.0)  # written
    assert arr[0, 2] == pytest.approx(70.0)  # unchanged
    np.testing.assert_allclose(arr[1, :], [40.0, 50.0, 60.0])  # other slice unchanged
