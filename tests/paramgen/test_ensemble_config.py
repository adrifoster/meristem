"""Tests for EnsembleConfig class"""

from pathlib import Path
import numpy as np
import pytest
from meristem.paramgen.ensemble_config import (
    EnsembleConfig,
    LatinHypercubeConfig,
    MorrisEnsembleConfig,
    OneAtATimeConfig,
)


def test_ensemble_config_paths_coerced(base_paths):
    """__post_init__ converts string paths to Path objects."""
    config = EnsembleConfig(
        param_dir=str(base_paths["param_dir"]),
        ensemble_dir=str(base_paths["ensemble_dir"]),
        file_prefix="test_ensemble",
        default_param_file=str(base_paths["default_param_file"]),
    )
    assert isinstance(config.param_dir, Path)
    assert isinstance(config.ensemble_dir, Path)
    assert isinstance(config.default_param_file, Path)


def test_ensemble_config_posterior_sources_coerced(base_paths, tmp_path):
    """posterior_sources is coerced to Path when provided."""
    config = EnsembleConfig(
        **base_paths,
        posterior_sources=str(tmp_path / "posterior.yaml"),
    )
    assert isinstance(config.posterior_sources, Path)


def test_ensemble_config_posterior_sources_none(base_paths):
    """posterior_sources remains None when not provided."""
    config = EnsembleConfig(**base_paths)
    assert config.posterior_sources is None


def test_ensemble_config_optional_defaults(base_paths):
    """param_list and fixed_indices default to None."""
    config = EnsembleConfig(**base_paths)
    assert config.param_list is None
    assert config.fixed_indices is None


def test_latin_hypercube_config_defaults(base_paths):
    """LatinHypercubeConfig defaults to 100 members and no prebuilt matrix."""
    config = LatinHypercubeConfig(**base_paths)
    assert config.ensemble_members == 100
    assert config.prebuilt is None


def test_latin_hypercube_config_prebuilt(base_paths):
    """LatinHypercubeConfig accepts a prebuilt matrix."""
    matrix = np.random.default_rng(0).random((50, 3))
    config = LatinHypercubeConfig(**base_paths, ensemble_members=50, prebuilt=matrix)
    assert config.prebuilt.shape == (50, 3)


def test_one_at_a_time_config(base_paths):
    """OneAtATimeConfig constructs successfully."""
    config = OneAtATimeConfig(**base_paths)
    assert isinstance(config, OneAtATimeConfig)


def test_morris_config_defaults(base_paths):
    """MorrisEnsembleConfig defaults to 100 trajectories, 4 levels, no optimization."""
    config = MorrisEnsembleConfig(**base_paths)
    assert config.morris_n == 100
    assert config.levels == 4
    assert config.optimal_trajectories is None


def test_morris_config_custom_values(base_paths):
    """MorrisEnsembleConfig accepts custom Morris settings."""
    config = MorrisEnsembleConfig(
        **base_paths, morris_n=20, levels=6, optimal_trajectories=10
    )
    assert config.morris_n == 20
    assert config.levels == 6
    assert config.optimal_trajectories == 10


@pytest.mark.parametrize("levels", [0, 1, 3, 5, -2])
def test_morris_config_odd_or_nonpositive_levels_raises(base_paths, levels):
    """MorrisEnsembleConfig rejects levels that are not a positive even number."""
    with pytest.raises(ValueError, match="levels must be a positive even number"):
        MorrisEnsembleConfig(**base_paths, levels=levels)


@pytest.mark.parametrize("levels", [2, 4, 8])
def test_morris_config_even_levels_accepted(base_paths, levels):
    """MorrisEnsembleConfig accepts positive even levels."""
    config = MorrisEnsembleConfig(**base_paths, levels=levels)
    assert config.levels == levels


def test_morris_config_paths_coerced(base_paths):
    """MorrisEnsembleConfig inherits path coercion from EnsembleConfig."""
    config = MorrisEnsembleConfig(
        param_dir=str(base_paths["param_dir"]),
        ensemble_dir=str(base_paths["ensemble_dir"]),
        file_prefix="test_ensemble",
        default_param_file=str(base_paths["default_param_file"]),
    )
    assert isinstance(config.param_dir, Path)
    assert isinstance(config.ensemble_dir, Path)
    assert isinstance(config.default_param_file, Path)


# =============================================================================
# input validation
# =============================================================================


@pytest.mark.parametrize("file_prefix", ["", "a/b", None, 3])
def test_ensemble_config_bad_file_prefix_raises(base_paths, file_prefix):
    """file_prefix must be a non-empty string without '/'."""
    base_paths["file_prefix"] = file_prefix
    with pytest.raises(ValueError, match="file_prefix"):
        EnsembleConfig(**base_paths)


@pytest.mark.parametrize("param_list", ["fates_leaf_slatop", [], ["a", 1]])
def test_ensemble_config_bad_param_list_raises(base_paths, param_list):
    """param_list must be a non-empty list of names; a bare string is rejected."""
    with pytest.raises(ValueError, match="param_list"):
        EnsembleConfig(**base_paths, param_list=param_list)


@pytest.mark.parametrize(
    "fixed_indices",
    [[0, 1], {"fates_pft": [0, 1.5]}, {"fates_pft": "0"}, {"fates_pft": [True]}],
)
def test_ensemble_config_bad_fixed_indices_raises(base_paths, fixed_indices):
    """fixed_indices must map dimension names to lists of integers."""
    with pytest.raises(ValueError, match="fixed_indices"):
        EnsembleConfig(**base_paths, fixed_indices=fixed_indices)


@pytest.mark.parametrize("fixed_indices", [{"fates_pft": [0, 2]}, {"fates_pft": 1}])
def test_ensemble_config_fixed_indices_accepted(base_paths, fixed_indices):
    """fixed_indices accepts a list of integers or a single integer per dimension."""
    config = EnsembleConfig(**base_paths, fixed_indices=fixed_indices)
    assert config.fixed_indices == fixed_indices


@pytest.mark.parametrize("config_cls", [LatinHypercubeConfig, MorrisEnsembleConfig])
@pytest.mark.parametrize("seed", [-1, 1.5, "42", True])
def test_config_bad_seed_raises(base_paths, config_cls, seed):
    """seed must be a non-negative integer or None."""
    with pytest.raises(ValueError, match="seed"):
        config_cls(**base_paths, seed=seed)


@pytest.mark.parametrize("config_cls", [LatinHypercubeConfig, MorrisEnsembleConfig])
@pytest.mark.parametrize("seed", [None, 0, 42, np.int64(7)])
def test_config_seed_accepted(base_paths, config_cls, seed):
    """seed accepts None, zero, and Python or numpy integers."""
    assert config_cls(**base_paths, seed=seed).seed == seed


@pytest.mark.parametrize("ensemble_members", [0, -5, 10.0, "10", True])
def test_latin_hypercube_config_bad_ensemble_members_raises(
    base_paths, ensemble_members
):
    """ensemble_members must be a positive integer."""
    with pytest.raises(ValueError, match="ensemble_members"):
        LatinHypercubeConfig(**base_paths, ensemble_members=ensemble_members)


@pytest.mark.parametrize(
    "prebuilt, match",
    [
        (np.full(5, 0.5), "2-D"),
        (np.full((4, 3), 0.5), "rows"),
        (np.full((5, 3), 1.5), r"\[0, 1\]"),
        (np.full((5, 3), np.nan), r"\[0, 1\]"),
    ],
)
def test_latin_hypercube_config_bad_prebuilt_raises(base_paths, prebuilt, match):
    """prebuilt must be 2-D, have ensemble_members rows, and lie in [0, 1]."""
    with pytest.raises(ValueError, match=match):
        LatinHypercubeConfig(**base_paths, ensemble_members=5, prebuilt=prebuilt)


def test_latin_hypercube_config_prebuilt_list_converted(base_paths):
    """A prebuilt nested list is converted to a float array."""
    config = LatinHypercubeConfig(
        **base_paths, ensemble_members=2, prebuilt=[[0, 1], [1, 0]]
    )
    assert isinstance(config.prebuilt, np.ndarray)
    assert config.prebuilt.dtype == float


@pytest.mark.parametrize("morris_n", [0, -1, 4.0, True])
def test_morris_config_bad_morris_n_raises(base_paths, morris_n):
    """morris_n must be a positive integer."""
    with pytest.raises(ValueError, match="morris_n"):
        MorrisEnsembleConfig(**base_paths, morris_n=morris_n)


@pytest.mark.parametrize("levels", [4.0, "4", True])
def test_morris_config_non_integer_levels_raises(base_paths, levels):
    """levels must be an integer."""
    with pytest.raises(ValueError, match="levels must be a positive even number"):
        MorrisEnsembleConfig(**base_paths, levels=levels)


@pytest.mark.parametrize("optimal_trajectories", [1, 3.0, 10, 11])
def test_morris_config_bad_optimal_trajectories_raises(
    base_paths, optimal_trajectories
):
    """optimal_trajectories must be an integer, at least 2 and less than morris_n."""
    with pytest.raises(ValueError, match="optimal_trajectories"):
        MorrisEnsembleConfig(
            **base_paths, morris_n=10, optimal_trajectories=optimal_trajectories
        )
