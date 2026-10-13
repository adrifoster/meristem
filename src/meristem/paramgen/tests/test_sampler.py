"""Tests for param_ens_gen.sampler."""

import numpy as np
import pandas as pd
import pytest

from param_ens_gen.sampler import (
    Sampler,
    SampleContext,
    UniformSampler,
    PosteriorSampler,
)
from param_ens_gen.distribution_stat import (
    FixedStat,
    PercentStat,
    PFTStat,
)

# ===========================================================================
# Sampler.from_row_and_sheet: factory
# ===========================================================================


def test_from_row_and_sheet_returns_uniform_sampler(default_row):
    """from_row_and_sheet() returns a UniformSampler for strategy='uniform'."""
    sampler = Sampler.from_row_and_sheet(default_row)
    assert isinstance(sampler, UniformSampler)


def test_from_row_and_sheet_returns_posterior_sampler(
    joint_param_row, posterior_config
):
    """from_row_and_sheet() returns a PosteriorSampler for strategy='posterior'."""
    sampler = Sampler.from_row_and_sheet(
        joint_param_row, posterior_config=posterior_config
    )
    assert isinstance(sampler, PosteriorSampler)


def test_from_row_and_sheet_unknown_strategy_raises(default_row):
    """from_row_and_sheet() raises ValueError for an unknown strategy."""
    row = default_row.copy()
    row["strategy"] = "not_a_strategy"
    with pytest.raises(ValueError, match="Unknown strategy"):
        Sampler.from_row_and_sheet(row)


# ===========================================================================
# UniformSampler.__init__
# ===========================================================================


def test_uniform_sampler_fixed_bounds(default_row):
    """UniformSampler parses fixed min/max bounds correctly."""
    sampler = UniformSampler(default_row)
    assert isinstance(sampler.min_stat, FixedStat)
    assert isinstance(sampler.max_stat, FixedStat)
    assert sampler.min_stat.value == pytest.approx(0.005)
    assert sampler.max_stat.value == pytest.approx(0.05)


def test_uniform_sampler_percent_bounds(percent_row):
    """UniformSampler parses percent min/max bounds correctly."""
    sampler = UniformSampler(percent_row)
    assert isinstance(sampler.min_stat, PercentStat)
    assert isinstance(sampler.max_stat, PercentStat)


def test_uniform_sampler_pft_bounds(pft_uniform_row, pft_sheet):
    """UniformSampler parses PFT-specific bounds correctly."""
    sampler = UniformSampler(pft_uniform_row, pft_sheet=pft_sheet)
    assert isinstance(sampler.min_stat, PFTStat)
    assert isinstance(sampler.max_stat, PFTStat)


def test_uniform_sampler_mixed_pft_raises(default_row):
    """UniformSampler raises ValueError when only one of min/max is 'pft'."""
    row = default_row.copy()
    row["param_min"] = "pft"
    row["param_max"] = "0.05"
    with pytest.raises(ValueError, match="both be 'pft' or neither"):
        UniformSampler(row)


def test_resolve_bounds_pft_axis_1(param_dataset):
    """resolve_bounds() correctly resolves PFT bounds along axis 1 for 2D parameters."""
    row = pd.Series(
        {
            "parameter_name": "fates_leaf_vcmax25top",
            "strategy": "uniform",
            "param_min": "pft",
            "param_max": "pft",
        }
    )
    pft_sheet = pd.DataFrame(
        {
            "pft_index": [1, 2, 3],
            "pft_name": ["white_spruce", "black_spruce", "deciduous"],
            "param_min": [20.0, 25.0, 30.0],
            "param_max": [80.0, 85.0, 90.0],
        }
    )
    sampler = UniformSampler(row, pft_sheet=pft_sheet)
    default_value = param_dataset["fates_leaf_vcmax25top"].values  # shape (2, 3)
    min_val, max_val = sampler.resolve_bounds(default_value, pft_axis=1)
    assert min_val.shape == (2, 3)
    assert max_val.shape == (2, 3)
    # all rows along leafage_class axis should have the same PFT-specific values
    np.testing.assert_allclose(min_val[0, :], [20.0, 25.0, 30.0])
    np.testing.assert_allclose(min_val[1, :], [20.0, 25.0, 30.0])
    np.testing.assert_allclose(max_val[0, :], [80.0, 85.0, 90.0])
    np.testing.assert_allclose(max_val[1, :], [80.0, 85.0, 90.0])


# ===========================================================================
# UniformSampler.resolve_bounds
# ===========================================================================


def test_resolve_bounds_fixed(default_row):
    """resolve_bounds() returns the correct fixed min and max."""
    sampler = UniformSampler(default_row)
    min_val, max_val = sampler.resolve_bounds(None)
    assert min_val == pytest.approx(0.005)
    assert max_val == pytest.approx(0.05)


def test_resolve_bounds_percent(percent_row):
    """resolve_bounds() applies percent correctly to the default value."""
    sampler = UniformSampler(percent_row)
    default = np.array([1.0, 2.0, 4.0])
    min_val, max_val = sampler.resolve_bounds(default)
    np.testing.assert_allclose(min_val, [0.5, 1.0, 2.0])
    np.testing.assert_allclose(max_val, [1.5, 3.0, 6.0])


def test_resolve_bounds_pft(pft_uniform_row, pft_sheet, param_dataset):
    """resolve_bounds() returns per-PFT arrays for PFT-specific bounds."""
    sampler = UniformSampler(pft_uniform_row, pft_sheet=pft_sheet)
    default_value = param_dataset["fates_leaf_slatop"].values
    min_val, max_val = sampler.resolve_bounds(default_value, pft_axis=0)
    np.testing.assert_allclose(min_val, [0.005, 0.004, 0.008])
    np.testing.assert_allclose(max_val, [0.040, 0.035, 0.060])


def test_resolve_bounds_valid_passes():
    """resolve_bounds() does not raise when min < max"""
    row = pd.Series(
        {
            "parameter_name": "test_param",
            "strategy": "uniform",
            "param_min": "0.1",
            "param_max": "0.9",
        }
    )
    sampler = UniformSampler(row)
    min_val, max_val = sampler.resolve_bounds(None)
    assert min_val == pytest.approx(0.1)
    assert max_val == pytest.approx(0.9)


def test_resolve_bounds_min_greater_than_max_raises():
    """resolve_bounds() raises ValueError when resolved min > max."""
    row = pd.Series(
        {
            "parameter_name": "test_param",
            "strategy": "uniform",
            "param_min": "0.9",
            "param_max": "0.1",
        }
    )
    sampler = UniformSampler(row)
    with pytest.raises(ValueError, match="min > max"):
        sampler.resolve_bounds(None)


def test_resolve_bounds_array_any_violation_raises(param_dataset):
    """resolve_bounds() raises ValueError if any element has min > max."""
    row = pd.Series(
        {
            "parameter_name": "test_param",
            "strategy": "uniform",
            "param_min": "pft",
            "param_max": "pft",
        }
    )
    pft_sheet = pd.DataFrame(
        {
            "pft_index": [1, 2, 3],
            "pft_name": ["white_spruce", "black_spruce", "deciduous"],
            "param_min": [0.1, 0.5, 0.9],
            "param_max": [0.9, 0.5, 0.8],
        }
    )
    default_value = param_dataset["fates_leaf_slatop"].values
    sampler = UniformSampler(row, pft_sheet=pft_sheet)
    with pytest.raises(ValueError, match="min > max"):
        sampler.resolve_bounds(default_value, pft_axis=0)


# ===========================================================================
# UniformSampler.sample
# ===========================================================================


def test_sample_negative_raises(default_row):
    """sample(negative number) raises a ValueError."""
    sampler = UniformSampler(default_row)
    with pytest.raises(ValueError, match="normalized_value"):
        sampler.sample(-0.01, SampleContext())


def test_sample_greater_than_one_raises(default_row):
    """sample(>1.0) raises a ValueError."""
    sampler = UniformSampler(default_row)
    with pytest.raises(ValueError, match="normalized_value"):
        sampler.sample(1.1, SampleContext())


def test_sample_at_zero_returns_min(default_row):
    """sample(0.0) returns the minimum bound."""
    sampler = UniformSampler(default_row)
    assert sampler.sample(0.0, SampleContext()) == pytest.approx(0.005)


def test_sample_at_one_returns_max(default_row):
    """sample(1.0) returns the maximum bound."""
    sampler = UniformSampler(default_row)
    assert sampler.sample(1.0, SampleContext()) == pytest.approx(0.05)


def test_sample_at_half_returns_midpoint(default_row):
    """sample(0.5) returns the midpoint of min and max."""
    sampler = UniformSampler(default_row)
    assert sampler.sample(0.5, SampleContext()) == pytest.approx(0.0275)


def test_sample_monotonic_with_increasing_input(default_row):
    """sample() produces monotonically increasing values as input increases."""
    sampler = UniformSampler(default_row)
    ctx = SampleContext()
    values = [sampler.sample(v, ctx) for v in np.linspace(0, 1, 20)]
    assert all(values[i] <= values[i + 1] for i in range(len(values) - 1))


def test_sample_at_zero_pft_returns_min(pft_uniform_row, pft_sheet, param_dataset):
    """sample(0.0) returns minimum bounds for PFT-specific bounds."""
    sampler = UniformSampler(pft_uniform_row, pft_sheet=pft_sheet)
    default_value = param_dataset["fates_leaf_slatop"].values
    result = sampler.sample(0.0, SampleContext(default_value=default_value, pft_axis=0))
    np.testing.assert_allclose(result, [0.005, 0.004, 0.008])


def test_sample_at_one_pft_returns_max(pft_uniform_row, pft_sheet, param_dataset):
    """sample(1.0) returns maximum bounds for PFT-specific bounds."""
    sampler = UniformSampler(pft_uniform_row, pft_sheet=pft_sheet)
    default_value = param_dataset["fates_leaf_slatop"].values
    result = sampler.sample(1.0, SampleContext(default_value=default_value, pft_axis=0))
    np.testing.assert_allclose(result, [0.040, 0.035, 0.060])


def test_sample_at_half_pft_returns_midpoint(pft_uniform_row, pft_sheet, param_dataset):
    """sample(0.5) returns the midpoint of min and max for PFT-specific bounds."""
    sampler = UniformSampler(pft_uniform_row, pft_sheet=pft_sheet)
    default_value = param_dataset["fates_leaf_slatop"].values
    result = sampler.sample(0.5, SampleContext(default_value=default_value, pft_axis=0))
    np.testing.assert_allclose(result, [0.0225, 0.0195, 0.034])


def test_sample_at_zero_percent_returns_min(percent_row):
    """sample(0.0) returns minimum bounds for percent bounds."""
    sampler = UniformSampler(percent_row)
    assert sampler.sample(0.0, SampleContext(default_value=10.0)) == pytest.approx(5.0)


def test_sample_at_one_percent_returns_max(percent_row):
    """sample(1.0) returns maximum bounds for percent bounds."""
    sampler = UniformSampler(percent_row)
    assert sampler.sample(1.0, SampleContext(default_value=10.0)) == pytest.approx(15.0)


def test_sample_at_half_percent_returns_default(percent_row):
    """sample(0.5) returns midpoint between min and max bounds for percent bounds."""
    sampler = UniformSampler(percent_row)
    default_val = 10.0
    assert sampler.sample(
        0.5, SampleContext(default_value=default_val)
    ) == pytest.approx(default_val)


def test_sample_half_array_returns_array(percent_row):
    """sample(0.5) returns midpoint between min and bounds for percent bounds, and
    is still an array"""
    sampler = UniformSampler(percent_row)
    default_val = np.array([[50.0, 60.0, 70.0], [40.0, 50.0, 60.0]])
    assert sampler.sample(
        0.5, SampleContext(default_value=default_val)
    ) == pytest.approx(default_val)


# ===========================================================================
# UniformSampler.normalize
# ===========================================================================


def test_normalize_below_min_raises(default_row):
    """normalize() raises ValueErrro for below minimum bound without clip_to_bounds."""
    sampler = UniformSampler(default_row)
    with pytest.raises(ValueError, match="minimum"):
        sampler.normalize(0.0, SampleContext())


def test_normalize_above_max_raises(default_row):
    """normalize() raises ValueErrro for above maximum bound without clip_to_bounds."""
    sampler = UniformSampler(default_row)
    with pytest.raises(ValueError, match="maximum"):
        sampler.normalize(1.0, SampleContext())


def test_normalize_is_inverse_of_sample(default_row):
    """normalize(sample(v)) == v for any v in [0, 1]."""
    sampler = UniformSampler(default_row)
    ctx = SampleContext()
    for v in np.linspace(0, 1, 10):
        sampled = sampler.sample(v, ctx)
        assert sampler.normalize(sampled, ctx) == pytest.approx(v)


def test_normalize_at_min_returns_zero(default_row):
    """normalize() returns 0.0 for the minimum bound."""
    sampler = UniformSampler(default_row)
    assert sampler.normalize(0.005, SampleContext()) == pytest.approx(0.0)


def test_normalize_at_max_returns_one(default_row):
    """normalize() returns 1.0 for the maximum bound."""
    sampler = UniformSampler(default_row)
    assert sampler.normalize(0.05, SampleContext()) == pytest.approx(1.0)


def test_normalize_at_min_pft_returns_zero(pft_uniform_row, pft_sheet, param_dataset):
    """normalize() returns 0.0 for minimum bounds for PFT-specific bounds."""
    sampler = UniformSampler(pft_uniform_row, pft_sheet=pft_sheet)
    default_value = param_dataset["fates_leaf_slatop"].values
    result = sampler.normalize(
        [0.005, 0.004, 0.008], SampleContext(default_value=default_value, pft_axis=0)
    )
    np.testing.assert_allclose(result, [0.0, 0.0, 0.0])


def test_normalize_at_max_pft_returns_zero(pft_uniform_row, pft_sheet, param_dataset):
    """normalize() returns 1.0 for maximum bounds for PFT-specific bounds."""
    sampler = UniformSampler(pft_uniform_row, pft_sheet=pft_sheet)
    default_value = param_dataset["fates_leaf_slatop"].values
    result = sampler.normalize(
        [0.040, 0.035, 0.060], SampleContext(default_value=default_value, pft_axis=0)
    )
    np.testing.assert_allclose(result, [1.0, 1.0, 1.0])


def test_normalize_at_max_percent_returns_one(percent_row):
    """normalize() returns 1.0 for maximum bounds for percent bounds."""
    sampler = UniformSampler(percent_row)
    assert sampler.normalize(15.0, SampleContext(default_value=10.0)) == pytest.approx(
        1.0
    )


def test_normalize_at_min_percent_returns_zero(percent_row):
    """normalize() returns 0.0 for minimum bounds for percent bounds."""
    sampler = UniformSampler(percent_row)
    assert sampler.normalize(5.0, SampleContext(default_value=10.0)) == pytest.approx(
        0.0
    )


def test_normalize_at_midpoint_percent_returns_half(percent_row):
    """normalize() returns 0.5 for halfway between min and max bounds for percent bounds."""
    sampler = UniformSampler(percent_row)
    default_val = 10.0
    assert sampler.normalize(
        default_val, SampleContext(default_value=default_val)
    ) == pytest.approx(0.5)


def test_normalize_clips_below_min(default_row):
    """normalize() with clip_to_bounds=True clips values below min to 0.0."""
    sampler = UniformSampler(default_row)
    result = sampler.normalize(0.0, SampleContext(), clip_to_bounds=True)
    assert result == pytest.approx(0.0)
    
    
def test_normalize_clips_above_max(default_row):
    """normalize() with clip_to_bounds=True clips values above max to 1.0."""
    sampler = UniformSampler(default_row)
    result = sampler.normalize(0.1, SampleContext(), clip_to_bounds=True)
    assert result == pytest.approx(1.0)
    
    
# ===========================================================================
# PosteriorSampler.__init__
# ===========================================================================


def test_posterior_sampler_missing_config_raises(joint_param_row):
    """PosteriorSampler raises ValueError when posterior_config is None."""
    with pytest.raises(ValueError, match="posterior_sources"):
        PosteriorSampler(joint_param_row, posterior_config=None)


def test_posterior_sampler_constructs_sources(joint_param_row, posterior_config):
    """PosteriorSampler constructs one PosteriorSource per file entry."""
    sampler = PosteriorSampler(joint_param_row, posterior_config=posterior_config)
    assert len(sampler.sources) == 1
    assert sampler.parameters == [
        "fates_leafn_vert_scaler_coeff1",
        "fates_leafn_vert_scaler_coeff2",
    ]


# ===========================================================================
# PosteriorSampler.sample: broadcast
# ===========================================================================


def test_posterior_sampler_broadcast_returns_list_of_arrays(
    joint_param_row, posterior_config
):
    """sample() with array_index=None returns a list of arrays (one per parameter)."""
    sampler = PosteriorSampler(joint_param_row, posterior_config=posterior_config)
    ctx = SampleContext(n_indices=3)
    result = sampler.sample(0.5, ctx)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(r, np.ndarray) for r in result)
    assert all(len(r) == 3 for r in result)


def test_posterior_sampler_broadcast_fills_all_indices(
    joint_param_row, posterior_config
):
    """sample() broadcast mode fills all n_indices positions with the same value."""
    sampler = PosteriorSampler(joint_param_row, posterior_config=posterior_config)
    ctx = SampleContext(n_indices=3)
    result = sampler.sample(0.5, ctx)
    assert result[0][0] == pytest.approx(result[0][1])
    assert result[0][1] == pytest.approx(result[0][2])


def test_posterior_sampler_multi_source_fills_correct_indices(
    joint_param_row, multi_source_posterior_config
):
    """_draw_broadcast fills each array index from the correct source."""
    sampler = PosteriorSampler(
        joint_param_row, posterior_config=multi_source_posterior_config
    )
    ctx = SampleContext(n_indices=3)
    result = sampler.sample(0.5, ctx)

    # indices 0 and 1 come from posterior_file (fates_leafn_vert_scaler_coeff1 in [0, 0.95])
    assert 0.0 <= result[0][0] <= 0.95
    assert 0.0 <= result[0][1] <= 0.95

    # index 2 comes from posterior_file_2 (fates_leafn_vert_scaler_coeff1 in [1.0, 1.95])
    assert 1.0 <= result[0][2] <= 1.95


# ===========================================================================
# PosteriorSampler.sample
# ===========================================================================


def test_posterior_sampler_per_index_returns_list_of_single_element_arrays(
    joint_param_row, posterior_config
):
    """sample() with array_index set returns a list of single-element arrays."""
    sampler = PosteriorSampler(joint_param_row, posterior_config=posterior_config)
    ctx = SampleContext(array_index=99)
    result = sampler.sample(0.5, ctx)
    assert isinstance(result, list)
    assert all(len(r) == 1 for r in result)


def test_posterior_sampler_no_index_match_raises(joint_param_row, posterior_file):
    """sample() raises ValueError when no source covers the index."""
    config = {
        "parameters": [
            "fates_leafn_vert_scaler_coeff1",
            "fates_leafn_vert_scaler_coeff2",
        ],
        "files": [{"path": str(posterior_file), "array_indices": [0, 1]}],
    }
    sampler = PosteriorSampler(joint_param_row, posterior_config=config)
    with pytest.raises(ValueError, match="No source found"):
        sampler.sample(0.5, SampleContext(array_index=99))

# ===========================================================================
# PosteriorSampler.normalize
# ===========================================================================



def test_posterior_sampler_normalize_for_index(tmp_path):
    """normalize routes to _unscale_for_index when array_index is set."""
    f = tmp_path / "posterior.txt"
    f.write_text("leaf_cn\n10.0\n20.0\n30.0\n")

    posterior_config = {
        "parameters": ["leaf_cn"],
        "files": [{"path": str(f), "array_indices": "all"}],
    }
    row = pd.Series({"parameter_name": "leaf_cn", "strategy": "posterior"})
    sampler = PosteriorSampler(row, posterior_config=posterior_config)

    context = SampleContext(array_index=0)
    result = sampler.normalize(20.0, context)
    assert 0.0 <= result <= 1.0


def test_posterior_sampler_normalize_broadcast(tmp_path):
    """normalize routes to _unscale_broadcast when array_index is None."""
    f = tmp_path / "posterior.txt"
    f.write_text("leaf_cn\n10.0\n20.0\n30.0\n")

    posterior_config = {
        "parameters": ["leaf_cn"],
        "files": [{"path": str(f), "array_indices": "all"}],
    }
    row = pd.Series({"parameter_name": "leaf_cn", "strategy": "posterior"})
    sampler = PosteriorSampler(row, posterior_config=posterior_config)

    context = SampleContext(array_index=None, n_indices=3)
    result = sampler.normalize(20.0, context)
    assert len(result) == 1  # one entry per parameter
    assert len(result[0]) == 3  # one value per index


def test_unscale_broadcast_multiple_sources(tmp_path):
    """_unscale_broadcast else branch: multiple sources with specific array indices."""
    f0 = tmp_path / "posterior_0.txt"
    f1 = tmp_path / "posterior_1.txt"
    f0.write_text("leaf_cn\n10.0\n20.0\n30.0\n")
    f1.write_text("leaf_cn\n10.0\n20.0\n30.0\n")

    posterior_config = {
        "parameters": ["leaf_cn"],
        "files": [
            {"path": str(f0), "array_indices": [0, 1]},
            {"path": str(f1), "array_indices": [2]},
        ],
    }
    row = pd.Series({"parameter_name": "leaf_cn", "strategy": "posterior"})
    sampler = PosteriorSampler(row, posterior_config=posterior_config)

    context = SampleContext(array_index=None, n_indices=3)
    result = sampler.normalize(20.0, context)
    assert len(result) == 1  # one entry per parameter
    assert len(result[0]) == 3  # one value per index
    assert all(0.0 <= v <= 1.0 for v in result[0])


def test_posterior_sampler_normalize_broadcast_joint(joint_param_row, posterior_config):
    """normalize() with a list of arrays uses sort_param_index correctly."""
    sampler = PosteriorSampler(joint_param_row, posterior_config=posterior_config)
    default_values = [
        np.array([0.012, 0.015, 0.005]),  # coeff1 defaults
        np.array([2.1, 2.5, 2.6]),  # coeff2 defaults
    ]
    ctx = SampleContext(array_index=None, n_indices=3)
    result = sampler.normalize(default_values, ctx)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(r, np.ndarray) for r in result)
    assert all(len(r) == 3 for r in result)
    assert all(np.all(r >= 0.0) and np.all(r <= 1.0) for r in result)


def test_posterior_sampler_normalize_broadcast_joint_multi_source(
    joint_param_row, posterior_file, posterior_file_2
):
    """normalize() with list of arrays works for multi-source case."""
    config = {
        "parameters": [
            "fates_leafn_vert_scaler_coeff1",
            "fates_leafn_vert_scaler_coeff2",
        ],
        "files": [
            {"path": str(posterior_file), "array_indices": [0, 1]},
            {"path": str(posterior_file_2), "array_indices": [2]},
        ],
    }
    sampler = PosteriorSampler(joint_param_row, posterior_config=config)
    default_values = [
        np.array(
            [0.012, 0.015, 1.5]
        ),  # coeff1: indices 0,1 from file1 [0,0.95], index 2 from file2 [1.0,1.95]
        np.array(
            [12.0, 14.0, 22.0]
        ),  # coeff2: indices 0,1 from file1 [10,19.5], index 2 from file2 [20,29.5]
    ]
    ctx = SampleContext(array_index=None, n_indices=3)
    result = sampler.normalize(default_values, ctx)
    assert len(result) == 2
    assert all(np.all(r >= 0.0) and np.all(r <= 1.0) for r in result)
