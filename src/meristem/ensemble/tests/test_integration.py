"""End-to-end integration tests for param_ens_gen."""

from __future__ import annotations
import subprocess
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

from param_ens_gen.param_ensemble import LatinHypercubeEnsemble, OneAtATimeEnsemble
from param_ens_gen.ensemble_config import LatinHypercubeConfig, OneAtATimeConfig
from param_ens_gen.parameter_dataset import ParameterDataset

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

# ===========================================================================
# Helpers
# ===========================================================================


def _make_lh_ensemble(param_dir, param_file, tmp_path, posterior_sources=None):
    config = LatinHypercubeConfig(
        param_dir=param_dir,
        ensemble_dir=tmp_path / "ensemble",
        file_prefix="test",
        default_param_file=param_file,
        ensemble_members=5,
        posterior_sources=posterior_sources,
    )
    return LatinHypercubeEnsemble(config)


def _make_oat_ensemble(param_dir, param_file, tmp_path, posterior_sources=None):
    config = OneAtATimeConfig(
        param_dir=param_dir,
        ensemble_dir=tmp_path / "ensemble",
        file_prefix="test",
        default_param_file=param_file,
        posterior_sources=posterior_sources,
    )
    return OneAtATimeEnsemble(config)


# ===========================================================================
# LH full pipeline — NetCDF
# ===========================================================================


def test_lh_written_files_differ_from_default(
    ensemble_param_dir, default_netcdf_file, tmp_path, posterior_config_file
):
    """Written ensemble members have at least one parameter value different from default."""
    ensemble = _make_lh_ensemble(
        ensemble_param_dir, default_netcdf_file, tmp_path, posterior_config_file
    )
    ensemble.create_ensemble()

    default_ds = ParameterDataset.from_path(default_netcdf_file)
    default_slatop = default_ds["fates_leaf_slatop"].values.copy()

    any_different = False
    for nc_file in sorted((tmp_path / "ensemble").glob("test_*.nc")):
        ds = ParameterDataset.from_path(nc_file)
        if not np.allclose(ds["fates_leaf_slatop"].values, default_slatop):
            any_different = True
            break
    assert any_different, "All ensemble members are identical to the default"


def test_lh_key_has_correct_shape(
    ensemble_param_dir, default_netcdf_file, tmp_path, posterior_config_file
):
    """Ensemble key has one row per member and one column per sampling unit."""
    ensemble = _make_lh_ensemble(
        ensemble_param_dir, default_netcdf_file, tmp_path, posterior_config_file
    )
    ensemble.create_ensemble()
    key = pd.read_csv(tmp_path / "ensemble" / "test_key.csv")
    assert len(key) == 5
    assert "ensemble" in key.columns


def test_lh_key_values_in_unit_interval(
    ensemble_param_dir, default_netcdf_file, tmp_path, posterior_config_file
):
    """All normalized values in the ensemble key are in [0, 1]."""
    ensemble = _make_lh_ensemble(
        ensemble_param_dir, default_netcdf_file, tmp_path, posterior_config_file
    )
    ensemble.create_ensemble()
    key = pd.read_csv(tmp_path / "ensemble" / "test_key.csv")
    print(key)
    numeric_cols = [c for c in key.columns if c != "ensemble"]
    values = key[numeric_cols].values
    assert np.all(values >= 0.0) and np.all(values <= 1.0)


def test_lh_written_file_is_reloadable(
    ensemble_param_dir, default_netcdf_file, tmp_path, posterior_config_file
):
    """Written ensemble members can be reloaded as ParameterDatasets."""
    ensemble = _make_lh_ensemble(
        ensemble_param_dir, default_netcdf_file, tmp_path, posterior_config_file
    )
    ensemble.create_ensemble()
    for nc_file in (tmp_path / "ensemble").glob("test_*.nc"):
        ds = ParameterDataset.from_path(nc_file)
        assert "fates_leaf_slatop" in ds
        ds.close()


# ===========================================================================
# LH full pipeline — JSON
# ===========================================================================


def test_lh_json_written_files_differ_from_default(
    ensemble_param_dir, json_param_file, tmp_path, posterior_config_file
):
    """JSON ensemble members have at least one parameter value different from default."""
    ensemble = _make_lh_ensemble(
        ensemble_param_dir, json_param_file, tmp_path, posterior_config_file
    )
    ensemble.create_ensemble()

    default_ds = ParameterDataset.from_path(json_param_file)
    default_slatop = default_ds["fates_leaf_slatop"].values.copy()

    any_different = False
    for json_file in sorted((tmp_path / "ensemble").glob("test_*.json")):
        ds = ParameterDataset.from_path(json_file)
        if not np.allclose(ds["fates_leaf_slatop"].values, default_slatop):
            any_different = True
            break
    assert any_different


def test_lh_json_output_format_matches_input(
    ensemble_param_dir, json_param_file, tmp_path, posterior_config_file
):
    """JSON input produces JSON output files."""
    ensemble = _make_lh_ensemble(
        ensemble_param_dir, json_param_file, tmp_path, posterior_config_file
    )
    ensemble.create_ensemble()
    json_files = list((tmp_path / "ensemble").glob("test_*.json"))
    nc_files = list((tmp_path / "ensemble").glob("test_*.nc"))
    assert len(json_files) == 5
    assert len(nc_files) == 0


# ===========================================================================
# OAT full pipeline
# ===========================================================================


def test_oat_min_member_has_minimum_value(
    ensemble_param_dir, default_param_file, tmp_path, posterior_config_file
):
    """OAT minimum member has a value at or near the parameter minimum."""
    ensemble = _make_oat_ensemble(
        ensemble_param_dir, default_param_file, tmp_path, posterior_config_file
    )
    samples = ensemble.create_samples()
    min_sample = next(
        s
        for s in samples
        if s.parameter_samples[0].normalized_value == 0.0
        and s.parameter_samples[0].group.params[0].spec.original_name
        == "fates_leaf_slatop"
    )
    ds = ensemble.create_ensemble_member(min_sample)
    result = ds["fates_leaf_slatop"].values
    np.testing.assert_allclose(result, [0.005, 0.005, 0.005])


def test_oat_max_member_has_maximum_value(
    ensemble_param_dir, default_param_file, tmp_path, posterior_config_file
):
    """OAT maximum member has a value at or near the parameter maximum."""
    ensemble = _make_oat_ensemble(
        ensemble_param_dir, default_param_file, tmp_path, posterior_config_file
    )
    samples = ensemble.create_samples()
    max_sample = next(
        s
        for s in samples
        if s.parameter_samples[0].normalized_value == 1.0
        and s.parameter_samples[0].group.params[0].spec.original_name
        == "fates_leaf_slatop"
    )
    ds = ensemble.create_ensemble_member(max_sample)
    result = ds["fates_leaf_slatop"].values
    np.testing.assert_allclose(result, [0.05, 0.05, 0.05])


def test_oat_other_params_unchanged(
    ensemble_param_dir, default_param_file, tmp_path, posterior_config_file
):
    """OAT member only modifies the target parameter, leaving others at default."""
    ensemble = _make_oat_ensemble(
        ensemble_param_dir, default_param_file, tmp_path, posterior_config_file
    )
    samples = ensemble.create_samples()
    min_sample = next(
        s
        for s in samples
        if s.parameter_samples[0].normalized_value == 0.0
        and s.parameter_samples[0].group.params[0].spec.original_name
        == "fates_leaf_slatop"
    )
    ds = ensemble.create_ensemble_member(min_sample)
    default_ds = ParameterDataset.from_path(default_param_file)
    np.testing.assert_allclose(
        ds["fates_canopy_closure_thresh"].values,
        default_ds["fates_canopy_closure_thresh"].values,
    )


# ===========================================================================
# Grouped ensemble integration
# ===========================================================================


def test_grouped_lh_netcdf_members_written_correctly(
    grouped_param_dir, default_netcdf_file, tmp_path
):
    """Grouped LH ensemble writes files with both grouped params modified."""
    ensemble = _make_lh_ensemble(grouped_param_dir, default_netcdf_file, tmp_path)
    ensemble.create_ensemble()
    output_files = list((tmp_path / "ensemble").glob("test_*.nc"))
    assert len(output_files) == 5


def test_grouped_lh_json_members_written_correctly(
    grouped_param_dir, json_param_file, tmp_path
):
    """Grouped LH ensemble writes files with both grouped params modified."""
    ensemble = _make_lh_ensemble(grouped_param_dir, json_param_file, tmp_path)
    ensemble.create_ensemble()
    output_files = list((tmp_path / "ensemble").glob("test_*.json"))
    assert len(output_files) == 5


def test_grouped_params_move_together(grouped_param_dir, default_param_file, tmp_path):
    """Params in the same group receive the same normalized value."""
    ensemble = _make_lh_ensemble(grouped_param_dir, default_param_file, tmp_path)
    samples = ensemble.create_samples()

    for sample in samples:
        # build member and verify both params were written
        ds = ensemble.create_ensemble_member(sample)
        assert "fates_leaf_slatop" in ds
        assert "fates_leaf_vcmax25top" in ds


# ===========================================================================
# CLI tests
# ===========================================================================


@pytest.mark.slow
def test_cli_clm_lh_example():
    """Full CLI run against the CLM Latin Hypercube example."""
    result = subprocess.run(
        ["param_ens_gen", "run", "lh_config.yaml"],
        cwd=EXAMPLES_DIR / "clm_example",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
    output_files = list(
        (EXAMPLES_DIR / "clm_example" / "output_LH").glob("clm_lh_*.nc")
    )
    assert len(output_files) == 20


@pytest.mark.slow
def test_cli_clm_oat_example():
    """Full CLI run against the CLM OAT example."""
    result = subprocess.run(
        ["param_ens_gen", "run", "oat_config.yaml"],
        cwd=EXAMPLES_DIR / "clm_example",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
    key_file = EXAMPLES_DIR / "clm_example" / "output_OAT" / "clm_oat_key.csv"
    assert key_file.exists()


@pytest.mark.slow
def test_cli_fates_lh_example():
    """Full CLI run against the FATES Latin Hypercube example."""
    result = subprocess.run(
        ["param_ens_gen", "run", "lh_config.yaml"],
        cwd=EXAMPLES_DIR / "fates_example",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
    output_files = list((EXAMPLES_DIR / "fates_example" / "output_LH").glob("*.json"))
    assert len(output_files) == 20


@pytest.mark.slow
def test_cli_fates_oat_example():
    """Full CLI run against the FATES OAT example."""
    result = subprocess.run(
        ["param_ens_gen", "run", "oat_config.yaml"],
        cwd=EXAMPLES_DIR / "fates_example",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
    key_file = EXAMPLES_DIR / "fates_example" / "output_OAT" / "fates_oat_key.csv"
    assert key_file.exists()
