"""Tests for JointParameter class"""

from __future__ import annotations

import numpy as np
import pytest

from param_ens_gen.parameter import DimIndex, Parameter


def test_validate_specs_passes_for_joint_parameter(
    joint_param_row, param_dataset, posterior_config
):
    """_validate_specs() does not raise for a correctly configured JointParameter."""
    Parameter.from_row(
        joint_param_row, param_dataset, posterior_config=posterior_config
    )


def test_joint_param_empty_base_params_raises(
    default_row, param_dataset, posterior_config
):
    """Test that a JointParameter with empty base_params raises"""
    default_row["parameter_name"] = "fates_leafn_vert_scaler"
    default_row["coord"] = "['fates_pft']"
    default_row["param_type"] = "joint"
    default_row["strategy"] = "posterior"
    default_row["param_min"] = ""
    default_row["param_max"] = ""
    default_row["base_params"] = "[]"
    with pytest.raises(ValueError, match="base_params"):
        Parameter.from_row(
            default_row, param_dataset, posterior_config=posterior_config
        )


def test_joint_get_default(joint_param_row, param_dataset, posterior_config):
    """JointParameter.get_default returns a list of arrays, one per base_param."""
    param = Parameter.from_row(
        joint_param_row, param_dataset, posterior_config=posterior_config
    )
    result = param.get_default(param_dataset)
    assert isinstance(result, list)
    assert len(result) == 2
    np.testing.assert_allclose(result[0], [0.012, 0.015, 0.005])
    np.testing.assert_allclose(result[1], [2.1, 2.5, 2.6])


def test_joint_get_default_returns_scalars_when_expanded(
    joint_param_row, param_dataset, posterior_config
):
    """get_default returns list of scalars when active_index is set."""
    param = Parameter.from_row(
        joint_param_row, param_dataset, posterior_config=posterior_config
    )
    param.active_index = DimIndex("fates_pft", 1)
    result = param.get_default(param_dataset)
    assert isinstance(result, list)
    assert len(result) == 2
    assert np.ndim(result[0]) == 0
    assert np.ndim(result[1]) == 0


def test_joint_get_default_returns_arrays_when_not_expanded(
    joint_param_row, param_dataset, posterior_config
):
    """get_default returns list of full arrays when active_index is None."""
    param = Parameter.from_row(
        joint_param_row, param_dataset, posterior_config=posterior_config
    )
    result = param.get_default(param_dataset)
    assert isinstance(result, list)
    assert all(isinstance(r, np.ndarray) for r in result)
    assert all(r.shape == (3,) for r in result)


def test_joint_set_value(
    joint_param_row, param_dataset, working_param_dataset, posterior_config
):
    """JointParameter.set_value writes each array to its corresponding variable."""
    param = Parameter.from_row(
        joint_param_row, param_dataset, posterior_config=posterior_config
    )
    new_values = [
        np.array([0.5, 0.6, 0.7]),
        np.array([5.0, 6.0, 7.0]),
    ]
    param.set_value(working_param_dataset, param_dataset, value=new_values)
    np.testing.assert_allclose(
        working_param_dataset["fates_leafn_vert_scaler_coeff1"].values, [0.5, 0.6, 0.7]
    )
    np.testing.assert_allclose(
        working_param_dataset["fates_leafn_vert_scaler_coeff2"].values, [5.0, 6.0, 7.0]
    )


def test_joint_set_value_wrong_length_raises(
    joint_param_row, param_dataset, working_param_dataset, posterior_config
):
    """JointParameter.set_value raises ValueError if value length mismatches base_params."""
    param = Parameter.from_row(
        joint_param_row, param_dataset, posterior_config=posterior_config
    )
    with pytest.raises(ValueError, match="expected 2 arrays"):
        param.set_value(
            working_param_dataset, param_dataset, value=[np.array([0.5, 0.6, 0.7])]
        )


def test_joint_set_value_with_index_sets_single_slot_in_all_params(
    joint_param, working_param_dataset, param_dataset
):
    """JointParameter.set_value with active_index just sets a single index"""
    joint_param.active_index = DimIndex("fates_pft", 1)
    joint_param.set_value(
        working_param_dataset,
        param_dataset,
        [0.99, 3.5],
        None,
    )
    assert working_param_dataset["fates_leafn_vert_scaler_coeff1"].values[
        1
    ] == pytest.approx(0.99)
    assert working_param_dataset["fates_leafn_vert_scaler_coeff2"].values[
        1
    ] == pytest.approx(3.5)
    # other indices untouched
    for i in [0, 2]:
        assert working_param_dataset["fates_leafn_vert_scaler_coeff1"].values[
            i
        ] == pytest.approx(param_dataset["fates_leafn_vert_scaler_coeff1"].values[i])
        assert working_param_dataset["fates_leafn_vert_scaler_coeff2"].values[
            i
        ] == pytest.approx(param_dataset["fates_leafn_vert_scaler_coeff2"].values[i])


def test_joint_set_value_broadcasts_arrays_to_all_params(
    joint_param, working_param_dataset, param_dataset
):
    """JointParameter.set_value can broadcast arrays to all indices"""
    coeff1 = np.array([0.010, 0.011, 0.012])
    coeff2 = np.array([2.2, 2.3, 2.4])
    joint_param.set_value(working_param_dataset, param_dataset, [coeff1, coeff2], None)
    np.testing.assert_array_equal(
        working_param_dataset["fates_leafn_vert_scaler_coeff1"].values, coeff1
    )
    np.testing.assert_array_equal(
        working_param_dataset["fates_leafn_vert_scaler_coeff2"].values, coeff2
    )


def test_joint_set_value_fixed_indices_respected_across_all_params(
    joint_param, working_param_dataset, param_dataset
):
    """JointParameter.set_value can hold fixed_indices at default"""
    coeff1 = np.array([0.010, 0.011, 0.012])
    coeff2 = np.array([2.2, 2.3, 2.4])
    joint_param.set_value(
        working_param_dataset, param_dataset, [coeff1, coeff2], {"fates_pft": [1]}
    )
    # index 1 held at default for both
    assert working_param_dataset["fates_leafn_vert_scaler_coeff1"].values[
        1
    ] == pytest.approx(param_dataset["fates_leafn_vert_scaler_coeff1"].values[1])
    assert working_param_dataset["fates_leafn_vert_scaler_coeff2"].values[
        1
    ] == pytest.approx(param_dataset["fates_leafn_vert_scaler_coeff2"].values[1])
    # other indices written
    assert working_param_dataset["fates_leafn_vert_scaler_coeff1"].values[
        0
    ] == pytest.approx(0.010)
    assert working_param_dataset["fates_leafn_vert_scaler_coeff2"].values[
        2
    ] == pytest.approx(2.4)


def test_joint_set_value_param_dataset_not_mutated(
    joint_param, working_param_dataset, param_dataset
):
    """JointParameter.set_value does not change the param_dataset"""
    original_c1 = param_dataset["fates_leafn_vert_scaler_coeff1"].values.copy()
    original_c2 = param_dataset["fates_leafn_vert_scaler_coeff2"].values.copy()
    joint_param.set_value(
        working_param_dataset,
        param_dataset,
        [np.ones(3), np.ones(3)],
        {"fates_pft": [0]},
    )
    np.testing.assert_array_equal(
        param_dataset["fates_leafn_vert_scaler_coeff1"].values, original_c1
    )
    np.testing.assert_array_equal(
        param_dataset["fates_leafn_vert_scaler_coeff2"].values, original_c2
    )


def test_joint_parameter_coerce_non_iterable_raises(
    joint_param, working_param_dataset, param_dataset
):
    """set_value raises TypeError if value is a non-iterable scalar."""
    with pytest.raises(TypeError, match="non-iterable"):
        joint_param.set_value(working_param_dataset, param_dataset, 5.0)

    with pytest.raises(TypeError, match="non-iterable"):
        joint_param.set_value(working_param_dataset, param_dataset, np.float64(5.0))


# =============================================================================
# Sample
# =============================================================================


def test_default_parameter_sample(default_param, param_dataset):
    """sample() returns a value with the expected value."""
    result = default_param.sample(0.5, param_dataset)
    assert result == pytest.approx(0.0275)


def test_default_parameter_sample_bounds(default_param, param_dataset):
    """sample() at 0.0 and 1.0 returns min and max."""
    assert default_param.sample(0.0, param_dataset) == pytest.approx(0.005)
    assert default_param.sample(1.0, param_dataset) == pytest.approx(0.05)


def test_scalar_parameter_sample(scalar_param, param_dataset):
    """sample() returns a value with the expected value."""
    result = scalar_param.sample(0.5, param_dataset)
    assert result == pytest.approx(0.8)


def test_scalar_parameter_sample_bounds(scalar_param, param_dataset):
    """sample() at 0.0 and 1.0 returns min and max."""
    assert scalar_param.sample(0.0, param_dataset) == pytest.approx(0.7)
    assert scalar_param.sample(1.0, param_dataset) == pytest.approx(0.9)


def test_percent_parameter_sample(percent_row, param_dataset):
    """sample() returns a value with the expected value."""
    param = Parameter.from_row(percent_row, param_dataset)
    result = param.sample(0.5, param_dataset)
    expected = param_dataset["fates_leaf_vcmax25top"].values
    np.testing.assert_allclose(result, expected)


def test_joint_parameter_sample(joint_param, param_dataset):
    """sample() returns a list of arrays, one per base_param."""
    result = joint_param.sample(0.5, param_dataset)

    # should be a list with one entry per base_param
    assert len(result) == 2

    # each entry should be an array with one value per PFT
    assert result[0].shape == (3,)
    assert result[1].shape == (3,)

    # values should be within the posterior range
    assert np.all(result[0] >= 0.0) and np.all(result[0] <= 0.95)  # coeff1 range
    assert np.all(result[1] >= 10.0) and np.all(result[1] <= 19.5)  # coeff2 range
