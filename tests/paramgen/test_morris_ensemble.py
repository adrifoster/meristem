"""Tests for MorrisEnsemble"""

import pytest
import numpy as np

from meristem.paramgen.param_ensemble import ParamEnsemble, MorrisEnsemble
from meristem.paramgen.ensemble_config import MorrisEnsembleConfig


def _rows_per_trajectory(ensemble: MorrisEnsemble) -> int:
    """A Morris trajectory has one row per sampling unit, plus a starting point"""
    return ensemble.num_sampling_units + 1


def test_from_dict_valid_morris(
    ensemble_param_dir, default_param_file, tmp_path, posterior_config_file
):
    """Test that from_dict can create a MorrisEnsemble class from a valid dict"""
    ensemble = ParamEnsemble.from_dict(
        {
            "ensemble_type": "Morris",
            "param_dir": str(ensemble_param_dir),
            "ensemble_dir": str(tmp_path / "ensemble"),
            "file_prefix": "test",
            "default_param_file": str(default_param_file),
            "posterior_sources": str(posterior_config_file),
            "morris_n": 4,
            "levels": 6,
            "optimal_trajectories": 2,
        }
    )
    assert isinstance(ensemble, MorrisEnsemble)
    assert ensemble.morris_n == 4
    assert ensemble.levels == 6
    assert ensemble.optimal_trajectories == 2


def test_from_dict_invalid_morris(
    ensemble_param_dir, default_param_file, tmp_path, posterior_config_file
):
    """Test that from_dict raises if you have ensemble_members in a Morris dict"""
    with pytest.raises(TypeError, match="Invalid config key for MorrisEnsemble"):
        ParamEnsemble.from_dict(
            {
                "ensemble_type": "Morris",
                "param_dir": str(ensemble_param_dir),
                "ensemble_dir": str(tmp_path / "ensemble"),
                "file_prefix": "test",
                "default_param_file": str(default_param_file),
                "ensemble_members": 5,
                "posterior_sources": posterior_config_file,
            }
        )


def test_from_dict_odd_levels_raises(
    ensemble_param_dir, default_param_file, tmp_path, posterior_config_file
):
    """Test that from_dict surfaces the even-levels check from MorrisEnsembleConfig"""
    with pytest.raises(ValueError, match="levels must be a positive even number"):
        ParamEnsemble.from_dict(
            {
                "ensemble_type": "Morris",
                "param_dir": str(ensemble_param_dir),
                "ensemble_dir": str(tmp_path / "ensemble"),
                "file_prefix": "test",
                "default_param_file": str(default_param_file),
                "posterior_sources": posterior_config_file,
                "levels": 5,
            }
        )


def test_morris_sets_attributes_from_config(morris_ensemble):
    """Test that the Morris settings are copied from the config"""
    assert morris_ensemble.morris_n == 4
    assert morris_ensemble.levels == 4
    assert morris_ensemble.optimal_trajectories is None


def test_build_morris_samples_shape(morris_ensemble):
    """Test that the Morris matrix has morris_n trajectories of (D + 1) rows each"""
    x = morris_ensemble.build_morris_samples()
    assert x.shape == (
        morris_ensemble.morris_n * _rows_per_trajectory(morris_ensemble),
        morris_ensemble.num_sampling_units,
    )


def test_build_morris_samples_on_grid(morris_ensemble):
    """Test that every Morris value sits on one of the `levels` grid points in [0, 1]"""
    x = morris_ensemble.build_morris_samples()
    grid = np.linspace(0.0, 1.0, morris_ensemble.levels)
    assert np.all(np.isclose(x[..., None], grid).any(axis=-1))


def test_build_morris_samples_one_change_per_step(morris_ensemble):
    """Test that each step within a trajectory changes exactly one sampling unit"""
    x = morris_ensemble.build_morris_samples()
    rows = _rows_per_trajectory(morris_ensemble)
    for start in range(0, len(x), rows):
        steps = np.diff(x[start : start + rows], axis=0)
        assert np.all(np.count_nonzero(steps, axis=1) == 1)


def test_build_morris_samples_optimal_trajectories(
    ensemble_param_dir, default_netcdf_file, tmp_path, posterior_config_file
):
    """Test that optimal_trajectories selects that many trajectories from the pool"""
    config = MorrisEnsembleConfig(
        param_dir=ensemble_param_dir,
        ensemble_dir=tmp_path / "ensemble",
        file_prefix="test",
        default_param_file=default_netcdf_file,
        posterior_sources=posterior_config_file,
        morris_n=6,
        levels=4,
        optimal_trajectories=3,
    )
    ensemble = MorrisEnsemble(config)
    x = ensemble.build_morris_samples()
    assert x.shape[0] == 3 * _rows_per_trajectory(ensemble)


def test_build_morris_samples_bad_optimal_trajectories_raises(
    ensemble_param_dir, default_netcdf_file, tmp_path, posterior_config_file
):
    """Test that optimal_trajectories larger than morris_n raises"""
    config = MorrisEnsembleConfig(
        param_dir=ensemble_param_dir,
        ensemble_dir=tmp_path / "ensemble",
        file_prefix="test",
        default_param_file=default_netcdf_file,
        posterior_sources=posterior_config_file,
        morris_n=4,
        optimal_trajectories=5,
    )
    ensemble = MorrisEnsemble(config)
    with pytest.raises(ValueError, match="optimal_trajectories"):
        ensemble.build_morris_samples()


def test_build_morris_samples_zero_params(morris_ensemble):
    """Test that build_morris_samples with zero sampling units returns an empty matrix"""
    morris_ensemble.sampling_units = []
    x = morris_ensemble.build_morris_samples()
    assert x.shape == (0, 0)
    assert not morris_ensemble.create_samples()


def test_morris_create_samples_length(morris_ensemble):
    """Test that create_samples returns one sample per row of the Morris matrix"""
    samples = morris_ensemble.create_samples()
    assert len(samples) == morris_ensemble.morris_n * _rows_per_trajectory(
        morris_ensemble
    )


def test_morris_create_samples_structure(morris_ensemble):
    """Test that create_samples returns one value per sampling unit in [0, 1]"""
    samples = morris_ensemble.create_samples()
    for sample in samples:
        assert len(sample) == morris_ensemble.num_sampling_units
        for ps in sample:
            assert 0.0 <= ps.normalized_value <= 1.0


def test_morris_create_samples_maps_columns_to_groups(morris_ensemble, mocker):
    """Test that column j of the Morris matrix goes to sampling unit j"""
    n_units = morris_ensemble.num_sampling_units
    matrix = np.arange(2 * n_units, dtype=float).reshape(2, n_units) / (2 * n_units)
    mocker.patch.object(morris_ensemble, "build_morris_samples", return_value=matrix)
    samples = morris_ensemble.create_samples()
    for i, sample in enumerate(samples):
        for j, ps in enumerate(sample):
            assert ps.group is morris_ensemble.sampling_units[j]
            assert ps.normalized_value == matrix[i, j]


def test_morris_create_ensemble_member(morris_ensemble):
    """Test that create_ensemble_member works for a Morris ensemble"""
    samples = morris_ensemble.create_samples()
    ds = morris_ensemble.create_ensemble_member(samples[0])
    assert "fates_leaf_slatop" in ds


def test_morris_create_ensemble_key(morris_ensemble):
    """Test that the ensemble key has one row per member and one column per group"""
    samples = morris_ensemble.create_samples()
    key = morris_ensemble.create_ensemble_key(samples)
    group_names = [g.name for g in morris_ensemble.sampling_units]
    assert list(key.columns) == ["ensemble", *group_names]
    assert len(key) == len(samples)


def test_morris_create_ensemble_key_values_match_samples(morris_ensemble):
    """Test that the ensemble key rows line up with the samples, in member order"""
    samples = morris_ensemble.create_samples()
    key = morris_ensemble.create_ensemble_key(samples)
    assert key["ensemble"].iloc[0] == "test_0000"
    for i, sample in enumerate(samples):
        for ps in sample:
            assert key[ps.group.name].iloc[i] == ps.normalized_value


def test_morris_create_ensemble_writes_files(morris_ensemble):
    """Test that create_ensemble writes one file per Morris sample, plus key and list"""
    morris_ensemble.create_ensemble()
    ext = morris_ensemble.default_ds.file_extension
    output_files = list(morris_ensemble.ensemble_dir.glob(f"test_*{ext}"))
    assert len(output_files) == morris_ensemble.morris_n * _rows_per_trajectory(
        morris_ensemble
    )
    assert (morris_ensemble.ensemble_dir / "test_key.csv").exists()
    assert (morris_ensemble.ensemble_dir / "test.txt").exists()


def test_grouped_morris_trajectory_length_uses_sampling_units(
    grouped_param_dir, default_param_file, tmp_path
):
    """Morris trajectories have (sampling units + 1) rows, not (parameters + 1)."""
    config = MorrisEnsembleConfig(
        param_dir=grouped_param_dir,
        ensemble_dir=tmp_path / "ensemble",
        file_prefix="test",
        default_param_file=default_param_file,
        morris_n=4,
    )
    ensemble = MorrisEnsemble(config)
    samples = ensemble.create_samples()
    assert len(samples) == 4 * (ensemble.num_sampling_units + 1)
    assert len(samples) < 4 * (ensemble.num_params + 1)
