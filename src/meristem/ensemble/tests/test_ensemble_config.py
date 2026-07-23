"""Tests for EnsembleConfig class"""

from pathlib import Path
import numpy as np
from param_ens_gen.ensemble_config import (
    EnsembleConfig,
    LatinHypercubeConfig,
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
