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
