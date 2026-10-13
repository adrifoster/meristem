"""Tests for Parameter class"""

from __future__ import annotations

import numpy as np
import pytest


from param_ens_gen.parameter import (
    DimIndex,
    DefaultParameter,
    JointParameter,
    Parameter,
    ScaleFromRootParameter,
    SlicedParameter,
)
from param_ens_gen.sampler import (
    UniformSampler,
    PosteriorSampler,
)

from param_ens_gen.distribution_stat import (
    FixedStat,
    PercentStat,
    PFTStat,
)

# ===========================================================================
# DimIndex
# ===========================================================================


def test_dimindex_stores_dim_and_index():
    """DimIndex correctly stores dim name and 0-based index."""
    di = DimIndex(dim="fates_pft", index=2)
    assert di.dim == "fates_pft"
    assert di.index == 2


def test_dimindex_is_named_tuple():
    """DimIndex is a named tuple"""
    di = DimIndex("fates_pft", 0)
    assert di[0] == "fates_pft"
    assert di[1] == 0


# ===========================================================================
# Parameter registry / from_row
# ===========================================================================


def test_from_row_returns_default_parameter(default_row, param_dataset):
    """from_row returns a DefaultParameter for param_type='default'."""
    param = Parameter.from_row(default_row, param_dataset)
    assert isinstance(param, DefaultParameter)


def test_from_row_returns_sliced_parameter(sliced_row, param_dataset):
    """from_row returns a SlicedParameter for param_type='sliced'."""
    param = Parameter.from_row(sliced_row, param_dataset)
    assert isinstance(param, SlicedParameter)


def test_from_row_returns_scale_from_root_parameter(scale_from_root_row, param_dataset):
    """from_row returns a ScaleFromRootParameter for param_type='scale_from_root'."""
    param = Parameter.from_row(scale_from_root_row, param_dataset)
    assert isinstance(param, ScaleFromRootParameter)


def test_from_row_returns_joint_parameter(
    joint_param_row, param_dataset, posterior_config
):
    """from_row returns a JointParameter for param_type='joint'."""
    param = Parameter.from_row(
        joint_param_row, param_dataset, posterior_config=posterior_config
    )
    assert isinstance(param, JointParameter)


def test_from_row_raises_for_unknown_param_type(default_row, param_dataset):
    """from_row raises ValueError for an unregistered param_type."""
    default_row["param_type"] = "unknown_type"
    with pytest.raises(ValueError, match="Unknown param_type"):
        Parameter.from_row(default_row, param_dataset)


def test_from_row_empty_param_type_raises(default_row, param_dataset):
    """from_row raises ValueError for a missing param_type cell"""
    default_row["param_type"] = ""
    with pytest.raises(ValueError, match="Unknown param_type"):
        Parameter.from_row(default_row, param_dataset)


def test_from_row_whitespace_param_type_stripped(default_row, param_dataset):
    """from_row raises ValueError for a missing param_type cell"""
    default_row["param_type"] = "  default  "
    param = Parameter.from_row(default_row, param_dataset)
    assert isinstance(param, DefaultParameter)


# ===========================================================================
# Parameter from_row / __init__
# ===========================================================================


def test_from_row_default_grabs_paramspec(default_row, param_dataset):
    """from_row correctly constructs the param spec data"""
    param = Parameter.from_row(default_row, param_dataset)
    assert param.spec is not None
    spec = param.spec
    assert spec.name == "fates_leaf_slatop"
    assert spec.param_type == "default"
    assert spec.dims == ["fates_pft"]
    assert spec.slice_dim is None
    assert spec.slice_index is None
    assert spec.root_param is None
    assert not spec.base_params


def test_from_row_uniform_parameter_has_uniform_sampler(default_row, param_dataset):
    """A uniform strategy parameter has a UniformSampler with correct DistributionStat"""
    param = Parameter.from_row(default_row, param_dataset)
    assert param.sampler is not None
    assert isinstance(param.sampler, UniformSampler)
    assert isinstance(param.sampler.min_stat, FixedStat)
    assert isinstance(param.sampler.max_stat, FixedStat)


def test_from_row_posterior_parameter_has_posterior_sampler(
    joint_param_row, param_dataset, posterior_config
):
    """A posterior strategy parameter has a PosteriorSampler"""
    param = Parameter.from_row(
        joint_param_row, param_dataset, posterior_config=posterior_config
    )
    assert param.sampler is not None
    assert isinstance(param.sampler, PosteriorSampler)
    assert param.sampler.parameters == [
        "fates_leafn_vert_scaler_coeff1",
        "fates_leafn_vert_scaler_coeff2",
    ]
    assert len(param.sampler.sources) == 1


def test_from_row_percent_parameter_has_percent_stat(percent_row, param_dataset):
    """from_row correctly constructs a parameter with percent bounds."""
    param = Parameter.from_row(percent_row, param_dataset)
    assert param.sampler is not None
    assert isinstance(param.sampler, UniformSampler)
    assert isinstance(param.sampler.min_stat, PercentStat)
    assert isinstance(param.sampler.max_stat, PercentStat)


def test_from_row_pft_parameter_has_pft_stat(pft_row, param_dataset, pft_sheet):
    """from_row correctly constructs a parameter with pft bounds."""
    param = Parameter.from_row(pft_row, param_dataset, pft_sheet=pft_sheet)
    assert param.sampler is not None
    assert isinstance(param.sampler, UniformSampler)
    assert isinstance(param.sampler.min_stat, PFTStat)
    assert isinstance(param.sampler.max_stat, PFTStat)


def test_from_row_active_index_is_none_on_construction(default_row, param_dataset):
    """active_index is None on a freshly constructed Parameter."""
    param = Parameter.from_row(default_row, param_dataset)
    assert param.active_index is None


# ===========================================================================
# Parameter from_row / validate_params
# ===========================================================================


def test_from_row_raises_for_missing_variable(default_row, param_dataset):
    """from_row raises ValueError when the variable is missing from the dataset."""
    default_row["parameter_name"] = "nonexistent_param"
    with pytest.raises(ValueError, match="not found in default dataset"):
        Parameter.from_row(default_row, param_dataset)


def test_from_row_raises_for_wrong_dims(default_row, param_dataset):
    """from_row raises ValueError when spec.dims does not match the dataset dims."""
    # Give the spec the wrong dims — variable exists but dims won't match
    default_row["coord"] = "['fates_leafage_class', 'fates_pft']"
    with pytest.raises(ValueError, match="Dimensions must match exactly"):
        Parameter.from_row(default_row, param_dataset)


def test_from_row_raises_for_missing_base_param(sliced_row, param_dataset):
    """from_row raises ValueError when a base_param variable is missing."""
    sliced_row["base_params"] = "['nonexistent_param']"
    with pytest.raises(ValueError, match="not found in default dataset"):
        Parameter.from_row(sliced_row, param_dataset)


def test_from_row_raises_for_missing_root_param(scale_from_root_row, param_dataset):
    """from_row raises ValueError when root_param is missing from the dataset."""
    scale_from_root_row["root_param"] = "nonexistent_root"
    with pytest.raises(ValueError, match="not found in default dataset"):
        Parameter.from_row(scale_from_root_row, param_dataset)


# ===========================================================================
# Parameter from_row / validate_posterior
# ===========================================================================


def test_from_row_non_posterior_sampler_passes_through(default_row, param_dataset):
    """from_row with a uniform sampler must not trigger any posterior validation."""
    param = DefaultParameter(default_row, param_dataset)
    assert param.sampler is not None


def test_from_row_posterior_parameters_mismatch_raises(
    joint_param_row, param_dataset, posterior_file
):
    """from_row with posterior parameters that don't match the config raises a ValueError."""
    # joint param targets coeff1 and coeff2, but config names something else
    config = {
        "parameters": ["wrong_param_a", "wrong_param_b"],
        "files": [{"path": str(posterior_file), "array_indices": "all"}],
    }
    with pytest.raises(ValueError, match="mismatch"):
        Parameter.from_row(joint_param_row, param_dataset, posterior_config=config)


def test_from_row_posterior_partial_parameters_mismatch_raises(
    joint_param_row, param_dataset, posterior_file
):
    """from_row with posterior parameters that don't match the config raises a ValueError."""
    # joint param targets coeff1 and coeff2, but config names something else
    config = {
        "parameters": ["fates_leafn_vert_scaler_coeff1", "wrong_param_b"],
        "files": [{"path": str(posterior_file), "array_indices": "all"}],
    }
    with pytest.raises(ValueError, match="mismatch"):
        Parameter.from_row(joint_param_row, param_dataset, posterior_config=config)


def test_from_row_non_broadcast_multi_dim_raises(
    default_row, param_dataset, posterior_file
):
    """from_row with multi-dimensional posterior parameters without broadcast raises"""
    default_row["parameter_name"] = "fates_leaf_vcmax25top"
    default_row["coord"] = "['fates_leafage_class', 'fates_pft']"
    default_row["strategy"] = "posterior"
    default_row["param_min"] = ""
    default_row["param_max"] = ""
    default_row["base_params"] = "fates_leaf_vcmax25top"
    print(default_row)
    config = {
        "parameters": ["fates_leaf_vcmax25top"],
        "files": [
            # non-broadcast: specific indices only
            {"path": str(posterior_file), "array_indices": [0, 1]}
        ],
    }
    with pytest.raises(ValueError, match="broadcast mode"):
        Parameter.from_row(default_row, param_dataset, posterior_config=config)


def test_from_row_array_indices_out_of_bounds_raises(
    joint_param_row, param_dataset, posterior_file
):
    """from_row raises if the array_indices on the posterior config doesn't match the parameter"""
    config = {
        "parameters": [
            "fates_leafn_vert_scaler_coeff1",
            "fates_leafn_vert_scaler_coeff2",
        ],
        "files": [
            # n_indices=3 (N_PFTS); index 3 is out of bounds
            {"path": str(posterior_file), "array_indices": [0, 1, 3]}
        ],
    }
    with pytest.raises(ValueError, match="dimension mismatch"):
        Parameter.from_row(joint_param_row, param_dataset, posterior_config=config)


# ===========================================================================
# Parameter free dims
# ===========================================================================


def test_free_dims_is_none_on_scalar_param(scalar_param):
    """free_dims on a scalar param is None"""
    assert scalar_param.free_dims is None


def test_free_dims_on_1d_param(default_param):
    """free_dims for a 1-d param is correct"""
    assert default_param.free_dims == ["fates_pft"]


def test_free_dims_returns_list(default_param):
    """free_dims returns a list"""
    assert isinstance(default_param.free_dims, list)


def test_free_dims_excludes_slice_dim(sliced_param):
    """free_dims on a sliced parameter should exclude the slice"""
    assert sliced_param.free_dims == ["fates_pft"]


def test_free_dims_includes_all_dims(multi_dim_param):
    """free_dims on a multi-dimensional parameter should include all dimensions"""
    assert multi_dim_param.free_dims == ["dim_1", "dim_2"]


# ===========================================================================
# Parameter n_indices
# ===========================================================================


def test_n_indices_scalar(scalar_param):
    """n_indices of a scalar parameter is [1]"""
    assert scalar_param.n_indices == [1]


def test_n_indices_1d(default_param):
    """n_indices of a 1d parameter is correct"""
    assert default_param.n_indices == [3]


def test_n_indices_sliced(sliced_param):
    """n_indices of a sliced parameter is just the slice"""
    assert sliced_param.n_indices == [3]


def test_n_indices_multi_param(multi_dim_param):
    """n_indices of a multi-parameter has all dims"""
    assert multi_dim_param.n_indices == [2, 3]


# =============================================================================
# set_value dispatch
# =============================================================================


def test_set_value_dispatches_to_write_at_index_when_expanded(
    default_param, working_param_dataset, param_dataset, mocker
):
    """set_value dispatches to write_at_index when expanded dims are used"""
    default_param.active_index = DimIndex("fates_pft", 1)
    mock_at_index = mocker.patch.object(default_param, "_write_at_index")
    mock_full = mocker.patch.object(default_param, "_write_full")

    default_param.set_value(working_param_dataset, param_dataset, 9.0)

    mock_at_index.assert_called_once_with(
        working_param_dataset, DimIndex("fates_pft", 1), 9.0
    )
    mock_full.assert_not_called()


def test_set_value_dispatches_to_write_full_when_not_expanded(
    default_param, working_param_dataset, param_dataset, mocker
):
    """set_value dispatches to write_full when expanded dims are not being used"""
    assert default_param.active_index is None
    mock_at_index = mocker.patch.object(default_param, "_write_at_index")
    mock_full = mocker.patch.object(default_param, "_write_full")

    default_param.set_value(working_param_dataset, param_dataset, 9.0)

    mock_full.assert_called_once_with(working_param_dataset, param_dataset, 9.0, {})
    mock_at_index.assert_not_called()


def test_set_value_none_fixed_indices_becomes_empty_dict(
    default_param, working_param_dataset, param_dataset, monkeypatch
):
    """fixed is set to an empty dict when it is None"""
    received = {}

    def capture(
        ds, param_dataset, value, fixed_indices
    ):  # pylint: disable=unused-argument
        received["fixed_indices"] = fixed_indices

    monkeypatch.setattr(default_param, "_write_full", capture)
    default_param.set_value(
        working_param_dataset, param_dataset, 9.0, fixed_indices=None
    )
    assert received["fixed_indices"] == {}


def test_set_value_applies_precision(default_row, param_dataset, working_param_dataset):
    """set_value rounds the value to the specified precision before writing."""
    default_row["precision"] = ".2f"
    param = Parameter.from_row(default_row, param_dataset)
    param.set_value(working_param_dataset, param_dataset, 0.123456)
    result = working_param_dataset["fates_leaf_slatop"].values
    assert result[0] == pytest.approx(0.12)


def test_set_value_no_precision_unchanged(
    default_row, param_dataset, working_param_dataset
):
    """set_value does not round when precision is None."""
    param = Parameter.from_row(default_row, param_dataset)
    param.set_value(working_param_dataset, param_dataset, 0.123456)
    result = working_param_dataset["fates_leaf_slatop"].values
    assert result[0] == pytest.approx(0.123456)


def test_set_value_precision_applied_to_array(
    default_row, param_dataset, working_param_dataset
):
    """set_value rounds array values to the specified precision."""
    default_row["precision"] = ".3f"
    param = Parameter.from_row(default_row, param_dataset)
    param.set_value(
        working_param_dataset, param_dataset, np.array([0.12345, 0.23456, 0.34567])
    )
    np.testing.assert_allclose(
        working_param_dataset["fates_leaf_slatop"].values, [0.123, 0.235, 0.346]
    )


def test_set_value_precision_applied_with_active_index(
    default_row, param_dataset, working_param_dataset
):
    """set_value rounds correctly when active_index is set."""
    default_row["precision"] = ".2f"
    param = Parameter.from_row(default_row, param_dataset)
    param.active_index = DimIndex("fates_pft", 1)
    param.set_value(working_param_dataset, param_dataset, 0.123456)
    assert working_param_dataset["fates_leaf_slatop"].values[1] == pytest.approx(0.12)


def test_set_value_precision_applied_to_joint_list(
    joint_param_row, param_dataset, working_param_dataset, posterior_config
):
    """set_value rounds list values for joint parameters."""
    joint_param_row["precision"] = ".2f"
    param = Parameter.from_row(
        joint_param_row, param_dataset, posterior_config=posterior_config
    )
    param.set_value(
        working_param_dataset,
        param_dataset,
        [np.array([0.12345, 0.23456, 0.34567]), np.array([2.1234, 2.5678, 2.9999])],
    )
    np.testing.assert_allclose(
        working_param_dataset["fates_leafn_vert_scaler_coeff1"].values,
        [0.12, 0.23, 0.35],
    )
    np.testing.assert_allclose(
        working_param_dataset["fates_leafn_vert_scaler_coeff2"].values,
        [2.12, 2.57, 3.00],
    )


# =============================================================================
# Sample
# =============================================================================


def test_sample_passes_active_index_to_context(default_param, param_dataset, mocker):
    """sample() passes active_index through to the SampleContext."""
    default_param.active_index = DimIndex("fates_pft", 1)
    mock_sample = mocker.patch.object(default_param.sampler, "sample", return_value=0.5)
    default_param.sample(0.5, param_dataset)
    ctx = mock_sample.call_args[0][1]
    assert ctx.array_index == 1


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


def test_build_context_pft_axis_set_for_1d_pft_param(
    default_param, param_dataset, mocker
):
    """_build_context sets pft_axis=0 for a 1D PFT parameter."""
    mock_sample = mocker.patch.object(default_param.sampler, "sample", return_value=0.5)
    default_param.sample(0.5, param_dataset)
    ctx = mock_sample.call_args[0][1]
    assert ctx.pft_axis == 0


def test_build_context_pft_axis_none_for_scalar_param(
    scalar_param, param_dataset, mocker
):
    """_build_context sets pft_axis=None for a scalar parameter."""
    mock_sample = mocker.patch.object(scalar_param.sampler, "sample", return_value=0.5)
    scalar_param.sample(0.5, param_dataset)
    ctx = mock_sample.call_args[0][1]
    assert ctx.pft_axis is None


def test_build_context_pft_axis_correct_for_2d_param(
    multi_dim_row, param_dataset, mocker
):
    """_build_context sets pft_axis to the correct axis for a 2D parameter with pft dim."""
    # make a 2D param where pft is on axis 1: ['dim_1', 'fates_pft']
    multi_dim_row["parameter_name"] = "fates_leaf_vcmax25top"
    multi_dim_row["coord"] = "['fates_leafage_class', 'fates_pft']"
    multi_dim_row["param_min"] = "20.0"
    multi_dim_row["param_max"] = "90.0"
    param = Parameter.from_row(multi_dim_row, param_dataset)
    mock_sample = mocker.patch.object(param.sampler, "sample", return_value=0.5)
    param.sample(0.5, param_dataset)
    ctx = mock_sample.call_args[0][1]
    assert ctx.pft_axis == 1


def test_build_context_pft_axis_none_for_non_pft_param(
    multi_dim_param, param_dataset, mocker
):
    """_build_context sets pft_axis=None when no dimension contains 'pft'."""
    mock_sample = mocker.patch.object(
        multi_dim_param.sampler, "sample", return_value=0.5
    )
    multi_dim_param.sample(0.5, param_dataset)
    ctx = mock_sample.call_args[0][1]
    assert ctx.pft_axis is None


# =============================================================================
# Normalize
# =============================================================================


def test_normalize_passes_active_index_to_context(default_param, default_ds, mocker):
    """normalize() passes active_index through to the SampleContext."""
    default_param.active_index = DimIndex("fates_pft", 1)
    mock_normalize = mocker.patch.object(
        default_param.sampler, "normalize", return_value=0.5
    )
    default_param.normalize(0.02, default_ds)
    ctx = mock_normalize.call_args[0][1]
    assert ctx.array_index == 1


def test_normalize_is_inverse_of_sample(default_param, default_ds):
    """normalize(sample(v)) == v for any v in [0, 1]."""
    for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
        sampled = default_param.sample(v, default_ds)
        assert default_param.normalize(sampled, default_ds) == pytest.approx(v)


def test_default_parameter_normalize_bounds(default_param, param_dataset):
    """normalize() at 0.005 and 0.05 returns 0.0 and 1.0."""
    assert default_param.normalize(0.005, param_dataset) == pytest.approx(0.0)
    assert default_param.normalize(0.05, param_dataset) == pytest.approx(1.0)


def test_scalar_parameter_normalize(scalar_param, param_dataset):
    """normalize() returns a value with the expected value."""
    result = scalar_param.normalize(0.8, param_dataset)
    assert result == pytest.approx(0.5)


def test_scalar_parameter_normalize_bounds(scalar_param, param_dataset):
    """normalize() at min and max returns 0.0 and 1.0."""
    assert scalar_param.normalize(0.7, param_dataset) == pytest.approx(0.0)
    assert scalar_param.normalize(0.9, param_dataset) == pytest.approx(1.0)


def test_percent_parameter_normalize(percent_row, param_dataset):
    """normalize() returns a value with the expected value."""
    param = Parameter.from_row(percent_row, param_dataset)
    default = param_dataset["fates_leaf_vcmax25top"].values
    result = param.normalize(default, param_dataset)
    np.testing.assert_allclose(result, 0.5)


def test_joint_parameter_normalize(joint_param, param_dataset):
    """normalize() returns a list of arrays, one per base_param."""
    default_values = [
        np.array([0.012, 0.015, 0.005]),
        np.array([2.1, 2.5, 2.6]),
    ]
    result = joint_param.normalize(default_values, param_dataset)

    assert len(result) == 2
    assert result[0].shape == (3,)
    assert result[1].shape == (3,)

    # normalized values should be in [0, 1]
    assert np.all(result[0] >= 0.0) and np.all(result[0] <= 1.0)
    assert np.all(result[1] >= 0.0) and np.all(result[1] <= 1.0)


def test_normalize_clips_to_bounds(default_row, param_dataset, mocker):
    """Parameter.normalize passes clip_to_bounds to the sampler."""
    param = Parameter.from_row(default_row, param_dataset)
    mock_normalize = mocker.patch.object(
        param.sampler, "normalize", return_value=0.0
    )
    param.normalize(0.0, param_dataset, clip_to_bounds=True)
    assert mock_normalize.call_args[1].get("clip_to_bounds") or \
           mock_normalize.call_args[0][2] == True