"""Tests for DefaultParameter class"""

from __future__ import annotations

import numpy as np
import pytest

from meristem.paramgen.parameter import DimIndex, Parameter


def test_default_validate_specs_passes(default_row, param_dataset):
    """_validate_specs() does not raise for a correctly configured DefaultParameter."""
    Parameter.from_row(default_row, param_dataset)


def test_default_missing_variable_raises(default_row, param_dataset):
    """DefaultParameter with a missing parameter raises"""
    default_row["parameter_name"] = "nonexistent_var"
    with pytest.raises(ValueError, match="not found"):
        Parameter.from_row(default_row, param_dataset)


def test_dim_mismatch_raises(default_row, param_dataset):
    """DefaultParameter with a dimension mismatch raises"""
    default_row["parameter_name"] = "fates_leaf_slatop"
    default_row["coord"] = "[]"
    with pytest.raises(ValueError, match="dims"):
        Parameter.from_row(default_row, param_dataset)


def test_default_get_default_pft(default_row, param_dataset):
    """DefaultParameter.get_default returns the full PFT array."""
    param = Parameter.from_row(default_row, param_dataset)
    result = param.get_default(param_dataset)
    np.testing.assert_allclose(result, [0.010, 0.020, 0.030])


def test_default_get_default_scalar(scalar_row, param_dataset):
    """DefaultParameter.get_default returns a scalar for scalar parameters."""
    param = Parameter.from_row(scalar_row, param_dataset)
    result = param.get_default(param_dataset)
    assert float(result) == pytest.approx(0.5)


def test_default_get_default_returns_scalar_when_expanded(default_row, param_dataset):
    """get_default returns a scalar when active_index is set."""
    param = Parameter.from_row(default_row, param_dataset)
    param.active_index = DimIndex("fates_pft", 1)
    result = param.get_default(param_dataset)
    assert np.ndim(result) == 0
    assert float(result) == pytest.approx(0.020)


def test_default_get_default_returns_array_when_not_expanded(
    default_row, param_dataset
):
    """get_default returns full array when active_index is None."""
    param = Parameter.from_row(default_row, param_dataset)
    result = param.get_default(param_dataset)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)


def test_default_set_value_uniform_no_fixed(
    default_row, param_dataset, working_param_dataset
):
    """DefaultParameter.set_value writes value to all array positions with a scalar."""
    param = Parameter.from_row(default_row, param_dataset)
    param.set_value(working_param_dataset, param_dataset, value=0.025)
    np.testing.assert_allclose(
        working_param_dataset["fates_leaf_slatop"].values,
        [0.025, 0.025, 0.025],
    )


def test_default_set_value_with_fixed_indices(
    default_row, param_dataset, working_param_dataset
):
    """DefaultParameter.set_value leaves fixed PFTs at their default values."""
    param = Parameter.from_row(default_row, param_dataset)
    param.set_value(
        working_param_dataset,
        param_dataset,
        value=0.025,
        fixed_indices={"fates_pft": [2]},  # fix PFT index 2 (0-based)
    )
    assert working_param_dataset["fates_leaf_slatop"].values[0] == pytest.approx(0.025)
    assert working_param_dataset["fates_leaf_slatop"].values[1] == pytest.approx(0.025)
    assert working_param_dataset["fates_leaf_slatop"].values[2] == pytest.approx(
        0.030
    )  # unchanged


def test_default_set_value_sets_single_slot(
    default_param, working_param_dataset, param_dataset
):
    """DefaultParameter.set_value writes to active_index position"""
    default_param.active_index = DimIndex(dim="fates_pft", index=1)
    default_param.set_value(working_param_dataset, param_dataset, 0.5)
    assert working_param_dataset["fates_leaf_slatop"].values[1] == pytest.approx(0.5)


def test_default_set_value_with_active_index(
    default_row, param_dataset, working_param_dataset
):
    """DefaultParameter.set_value writes only to active_index position."""
    param = Parameter.from_row(default_row, param_dataset)
    param.active_index = DimIndex(dim="fates_pft", index=1)
    param.set_value(working_param_dataset, param_dataset, 0.099)

    assert working_param_dataset["fates_leaf_slatop"].values[0] == pytest.approx(
        0.010
    )  # unchanged
    assert working_param_dataset["fates_leaf_slatop"].values[1] == pytest.approx(
        0.099
    )  # written
    assert working_param_dataset["fates_leaf_slatop"].values[2] == pytest.approx(
        0.030
    )  # unchanged


def test_default_set_value_write_at_index_other_slots_unchanged(
    default_param, working_param_dataset, param_dataset
):
    """DefaultParameter.set_value doesn't write to other positions when active_index is on"""
    default_param.active_index = DimIndex(dim="fates_pft", index=1)
    default_param.set_value(working_param_dataset, param_dataset, 0.5)
    assert working_param_dataset["fates_leaf_slatop"].values[0] == pytest.approx(
        param_dataset["fates_leaf_slatop"].values[0]
    )
    assert working_param_dataset["fates_leaf_slatop"].values[2] == pytest.approx(
        param_dataset["fates_leaf_slatop"].values[2]
    )


def test_default_set_value_write_at_index_rejects_array_value(
    default_param, working_param_dataset, param_dataset
):
    """Test that set_value with an active_index raises with an array"""
    default_param.active_index = DimIndex(dim="fates_pft", index=1)
    with pytest.raises(ValueError, match="scalar"):
        default_param.set_value(
            working_param_dataset, param_dataset, np.array([0.01, 0.02])
        )


def test_default_set_value_scalar_value_broadcasts(
    default_param, working_param_dataset, param_dataset
):
    """Test that set_value with a scalar broadcasts to the whole array"""
    default_param.set_value(working_param_dataset, param_dataset, 0.025, None)
    np.testing.assert_array_equal(
        working_param_dataset["fates_leaf_slatop"].values, [0.025, 0.025, 0.025]
    )


def test_default_set_value_array_value(
    default_param, working_param_dataset, param_dataset
):
    """DefaultParameter.set_value can set a whole array"""
    value = np.array([0.011, 0.022, 0.033])
    default_param.set_value(working_param_dataset, param_dataset, value, None)
    np.testing.assert_array_equal(
        working_param_dataset["fates_leaf_slatop"].values, value
    )


def test_default_set_value_fixed_indices_held_at_default(
    default_param, working_param_dataset, param_dataset
):
    """DefaultValue.set_value holds fixed_indices at default"""
    default_param.set_value(
        working_param_dataset, param_dataset, 0.025, {"fates_pft": [1]}
    )
    result = working_param_dataset["fates_leaf_slatop"].values
    assert result[0] == pytest.approx(0.025)
    assert result[1] == pytest.approx(param_dataset["fates_leaf_slatop"].values[1])
    assert result[2] == pytest.approx(0.025)


def test_default_set_value_scalar_param(
    scalar_param, working_param_dataset, param_dataset
):
    """DefaultValue.set_value correctly writes a scalar parameter"""
    scalar_param.set_value(working_param_dataset, param_dataset, 0.8, None)
    assert float(
        working_param_dataset["fates_canopy_closure_thresh"].values
    ) == pytest.approx(0.8)


def test_default_set_value_ds_not_mutated(
    default_param, working_param_dataset, param_dataset
):
    """DefaultValue.set_value doesn't mutate the default dataset"""
    original = param_dataset["fates_leaf_slatop"].values.copy()
    default_param.set_value(
        working_param_dataset, param_dataset, 0.025, {"fates_pft": [0]}
    )
    np.testing.assert_array_equal(param_dataset["fates_leaf_slatop"].values, original)


def test_default_set_value_scalar_fills_all_positions(
    default_row, param_dataset, working_param_dataset
):
    """DefaultParameter.set_value writes value to all array positions with a scalar."""
    param = Parameter.from_row(default_row, param_dataset)
    working_param_dataset["fates_leaf_slatop"].values = np.array([1.0, 2.0, 3.0])
    param.set_value(working_param_dataset, param_dataset, value=9.0)
    np.testing.assert_equal(
        working_param_dataset["fates_leaf_slatop"].values,
        [9.0, 9.0, 9.0],
    )


def test_default_set_value_array_passthrough(
    default_row, param_dataset, working_param_dataset
):
    """DefaultParameter.set_value writes can pass an array"""
    param = Parameter.from_row(default_row, param_dataset)
    working_param_dataset["fates_leaf_slatop"].values = np.array([1.0, 2.0, 3.0])
    param.set_value(
        working_param_dataset, param_dataset, value=np.array([4.0, 5.0, 6.0])
    )
    np.testing.assert_equal(
        working_param_dataset["fates_leaf_slatop"].values,
        [4.0, 5.0, 6.0],
    )


def test_default_set_value_scalar_fixed_positions_held(
    default_row, param_dataset, working_param_dataset
):
    """DefaultParameter.set_value correctly holds indices fixed with an input scalar value"""
    param = Parameter.from_row(default_row, param_dataset)
    working_param_dataset["fates_leaf_slatop"].values = np.array([1.0, 2.0, 3.0])
    param.set_value(working_param_dataset, param_dataset, 9.0, {"fates_pft": [1]})
    np.testing.assert_equal(
        working_param_dataset["fates_leaf_slatop"].values,
        [9.0, 2.0, 9.0],
    )


def test_default_set_value_array_fixed_positions_held(
    default_row, param_dataset, working_param_dataset
):
    """DefaultParameter.set_value correctly holds indices fixed with an input array value"""
    param = Parameter.from_row(default_row, param_dataset)
    working_param_dataset["fates_leaf_slatop"].values = np.array([1.0, 2.0, 3.0])
    param.set_value(
        working_param_dataset,
        param_dataset,
        np.array([7.0, 8.0, 9.0]),
        {"fates_pft": [0]},
    )
    np.testing.assert_equal(
        working_param_dataset["fates_leaf_slatop"].values,
        [1.0, 8.0, 9.0],
    )


def test_default_set_value_all_positions_fixed_returns_original(
    default_row, param_dataset, working_param_dataset
):
    """DefaultParameter.set_value correctly can hold all indices fixed"""
    param = Parameter.from_row(default_row, param_dataset)
    working_param_dataset["fates_leaf_slatop"].values = np.array([1.0, 2.0, 3.0])
    param.set_value(
        working_param_dataset,
        param_dataset,
        np.array([7.0, 8.0, 9.0]),
        {"fates_pft": [0, 1, 2]},
    )
    np.testing.assert_equal(
        working_param_dataset["fates_leaf_slatop"].values,
        np.array([1.0, 2.0, 3.0]),
    )


@pytest.mark.parametrize(
    "fixed_indices",
    [
        {"fates_leafage_class": [0]},  # a dimension this parameter doesn't have
        {"fates_pft": []},  # nothing fixed on its dimension
    ],
)
def test_default_set_value_ignored_fixed_indices_hold_nothing(
    default_row, param_dataset, working_param_dataset, fixed_indices
):
    """DefaultParameter.set_value writes every position when fixed_indices
    names another dimension or lists no indices
    """
    param = Parameter.from_row(default_row, param_dataset)
    working_param_dataset["fates_leaf_slatop"].values = np.array([1.0, 2.0, 3.0])
    param.set_value(working_param_dataset, param_dataset, 9.0, fixed_indices)
    np.testing.assert_equal(
        working_param_dataset["fates_leaf_slatop"].values,
        [9.0, 9.0, 9.0],
    )


def test_default_set_value_shape_mismatch_raises(
    default_row, param_dataset, working_param_dataset
):
    """DefaultParameter.set_value raises for a shape mismatch between value and param"""
    param = Parameter.from_row(default_row, param_dataset)
    working_param_dataset["fates_leaf_slatop"].values = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="shape"):
        param.set_value(working_param_dataset, param_dataset, np.ones(5), None)


def test_default_set_value_0d_array_input(
    scalar_row, param_dataset, working_param_dataset
):
    """DefaultParameter.set_value can take a 0d array"""
    param = Parameter.from_row(scalar_row, param_dataset)
    working_param_dataset["fates_canopy_closure_thresh"].values = 1.0
    param.set_value(working_param_dataset, param_dataset, np.float64(9.0), None)
    assert float(
        working_param_dataset["fates_canopy_closure_thresh"].values
    ) == pytest.approx(9.0)


def test_default_set_value_float_input(
    default_param, param_dataset, working_param_dataset
):
    """DefaultParameter.set_value can take a float input with an active_index"""
    default_param.active_index = DimIndex("fates_pft", 1)
    default_param.set_value(working_param_dataset, param_dataset, 3.14)
    assert working_param_dataset["fates_leaf_slatop"].values[1] == pytest.approx(3.14)


def test_default_set_value_0d_input(
    default_param, param_dataset, working_param_dataset
):
    """DefaultParameter.set_value can take a 0d input with an active_index"""
    default_param.active_index = DimIndex("fates_pft", 1)
    default_param.set_value(working_param_dataset, param_dataset, np.float64(7.0))
    assert working_param_dataset["fates_leaf_slatop"].values[1] == pytest.approx(7.0)


def test_default_set_value_single_element_array_input(
    default_param, param_dataset, working_param_dataset
):
    """DefaultParameter.set_value can take a single_element_array input with an active_index"""
    default_param.active_index = DimIndex("fates_pft", 1)
    default_param.set_value(working_param_dataset, param_dataset, np.array([2.5]))
    assert working_param_dataset["fates_leaf_slatop"].values[1] == pytest.approx(2.5)


def test_default_set_value_multi_element_array_raises(
    default_param, param_dataset, working_param_dataset
):
    """DefaultParameter.set_value with active_index and a multi-element array raises"""
    default_param.active_index = DimIndex("fates_pft", 1)
    with pytest.raises(ValueError, match="scalar"):
        default_param.set_value(
            working_param_dataset, param_dataset, np.array([1.0, 2.0])
        )


def test_default_normalize_at_min(default_row, param_dataset):
    """normalize() returns 0.0 for the minimum bound."""
    param = Parameter.from_row(default_row, param_dataset)
    assert param.normalize(0.005, param_dataset) == pytest.approx(0.0)


def test_default_normalize_at_max(default_row, param_dataset):
    """normalize() returns 1.0 for the maximum bound."""
    param = Parameter.from_row(default_row, param_dataset)
    assert param.normalize(0.05, param_dataset) == pytest.approx(1.0)


def test_default_normalize_at_midpoint(default_row, param_dataset):
    """normalize() returns 0.5 for the midpoint of min and max."""
    param = Parameter.from_row(default_row, param_dataset)
    assert param.normalize(0.0275, param_dataset) == pytest.approx(0.5)


# multi-dimensional parameters: parameter_a has dims (dim_1, dim_2) and values
# [[50, 60, 70], [40, 50, 60]], so dim_2 is the second axis


def test_default_get_default_expanded_along_second_axis(multi_dim_row, param_dataset):
    """get_default selects along the active dimension, not the first axis"""
    param = Parameter.from_row(multi_dim_row, param_dataset)
    param.active_index = DimIndex(dim="dim_2", index=1)
    np.testing.assert_allclose(param.get_default(param_dataset), [60.0, 50.0])


def test_default_set_value_expanded_along_second_axis(
    multi_dim_row, param_dataset, working_param_dataset
):
    """set_value with an active_index on the second axis writes that column only"""
    param = Parameter.from_row(multi_dim_row, param_dataset)
    param.active_index = DimIndex(dim="dim_2", index=1)
    param.set_value(working_param_dataset, param_dataset, 9.0)
    np.testing.assert_allclose(
        working_param_dataset["parameter_a"].values,
        [[50.0, 9.0, 70.0], [40.0, 9.0, 60.0]],
    )


def test_default_set_value_expanded_accepts_sub_array(
    multi_dim_row, param_dataset, working_param_dataset
):
    """set_value with an active_index accepts an array shaped like the sub-array"""
    param = Parameter.from_row(multi_dim_row, param_dataset)
    param.active_index = DimIndex(dim="dim_2", index=2)
    param.set_value(working_param_dataset, param_dataset, np.array([1.0, 2.0]))
    np.testing.assert_allclose(
        working_param_dataset["parameter_a"].values,
        [[50.0, 60.0, 1.0], [40.0, 50.0, 2.0]],
    )


def test_default_set_value_expanded_rejects_wrong_shape(
    multi_dim_row, param_dataset, working_param_dataset
):
    """set_value with an active_index rejects an array not shaped like the sub-array"""
    param = Parameter.from_row(multi_dim_row, param_dataset)
    param.active_index = DimIndex(dim="dim_2", index=1)
    with pytest.raises(ValueError, match="scalar"):
        param.set_value(working_param_dataset, param_dataset, np.array([1.0, 2.0, 3.0]))


def test_default_set_value_expanded_on_missing_dim_raises(
    multi_dim_row, param_dataset, working_param_dataset
):
    """set_value raises if the active dimension isn't one of the variable's dims"""
    param = Parameter.from_row(multi_dim_row, param_dataset)
    param.active_index = DimIndex(dim="fates_pft", index=1)
    with pytest.raises(ValueError, match="fates_pft"):
        param.set_value(working_param_dataset, param_dataset, 9.0)


@pytest.mark.parametrize(
    "fixed_indices,expected",
    [
        ({"dim_2": [0]}, [[50.0, 9.0, 9.0], [40.0, 9.0, 9.0]]),
        ({"dim_1": [1]}, [[9.0, 9.0, 9.0], [40.0, 50.0, 60.0]]),
        ({"dim_1": [0], "dim_2": [2]}, [[50.0, 60.0, 70.0], [9.0, 9.0, 60.0]]),
    ],
)
def test_default_set_value_multi_dim_fixed_indices_held(
    multi_dim_row, param_dataset, working_param_dataset, fixed_indices, expected
):
    """fixed_indices hold positions on whichever dimension they name"""
    param = Parameter.from_row(multi_dim_row, param_dataset)
    param.set_value(
        working_param_dataset, param_dataset, 9.0, fixed_indices=fixed_indices
    )
    np.testing.assert_allclose(working_param_dataset["parameter_a"].values, expected)


def test_default_set_value_expanded_fixed_indices_on_other_dim_held(
    multi_dim_row, param_dataset, working_param_dataset
):
    """an expanded parameter leaves fixed positions on its other dimensions alone"""
    param = Parameter.from_row(multi_dim_row, param_dataset)
    param.active_index = DimIndex(dim="dim_2", index=1)
    param.set_value(
        working_param_dataset, param_dataset, 9.0, fixed_indices={"dim_1": [1]}
    )
    np.testing.assert_allclose(
        working_param_dataset["parameter_a"].values,
        [[50.0, 9.0, 70.0], [40.0, 50.0, 60.0]],
    )


# per-PFT bounds on expanded parameters: pft_sheet gives
# param_min [0.005, 0.004, 0.008] and param_max [0.040, 0.035, 0.060]


@pytest.mark.parametrize(
    "index,expected_min,expected_max",
    [(0, 0.005, 0.040), (1, 0.004, 0.035), (2, 0.008, 0.060)],
)
def test_default_expanded_uses_own_pft_bounds(
    pft_row, pft_sheet, param_dataset, index, expected_min, expected_max
):
    """An expanded parameter with per-PFT bounds samples within its own PFT's bounds"""
    param = Parameter.from_row(pft_row, param_dataset, pft_sheet=pft_sheet)
    param.active_index = DimIndex(dim="fates_pft", index=index)
    assert param.sample(0.0, param_dataset) == pytest.approx(expected_min)
    assert param.sample(1.0, param_dataset) == pytest.approx(expected_max)
    assert param.sample(0.5, param_dataset) == pytest.approx(
        (expected_min + expected_max) / 2
    )


def test_default_expanded_normalize_uses_own_pft_bounds(
    pft_row, pft_sheet, param_dataset
):
    """normalize on an expanded parameter uses its own PFT's bounds"""
    param = Parameter.from_row(pft_row, param_dataset, pft_sheet=pft_sheet)
    param.active_index = DimIndex(dim="fates_pft", index=1)
    assert param.normalize(0.004, param_dataset) == pytest.approx(0.0)
    assert param.normalize(0.035, param_dataset) == pytest.approx(1.0)


def test_default_expanded_percent_bounds_use_own_default(default_row, param_dataset):
    """An expanded parameter with percent bounds resolves them from its own default"""
    default_row["param_min"] = "50percent"
    default_row["param_max"] = "50percent"
    param = Parameter.from_row(default_row, param_dataset)
    param.active_index = DimIndex(dim="fates_pft", index=2)
    # default at PFT 2 is 0.030
    assert param.sample(0.0, param_dataset) == pytest.approx(0.015)
    assert param.sample(1.0, param_dataset) == pytest.approx(0.045)


def test_default_expanded_pft_bounds_on_2d_param(
    multi_dim_row, pft_sheet, param_dataset
):
    """Per-PFT bounds on a 2-D parameter expanded along pft (its second axis)
    give that PFT's bounds down the whole column
    """
    multi_dim_row["parameter_name"] = "fates_leaf_vcmax25top"
    multi_dim_row["coord"] = "['fates_leafage_class', 'fates_pft']"
    multi_dim_row["param_min"] = "pft"
    multi_dim_row["param_max"] = "pft"
    param = Parameter.from_row(multi_dim_row, param_dataset, pft_sheet=pft_sheet)
    param.active_index = DimIndex(dim="fates_pft", index=1)
    np.testing.assert_allclose(param.sample(0.0, param_dataset), [0.004, 0.004])
    np.testing.assert_allclose(param.sample(1.0, param_dataset), [0.035, 0.035])
