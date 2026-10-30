"""End-to-end integration tests for meristem.paramgen."""

from __future__ import annotations
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd
import yaml

from meristem.paramgen.param_ensemble import LatinHypercubeEnsemble, OneAtATimeEnsemble
from meristem.paramgen.ensemble_config import LatinHypercubeConfig, OneAtATimeConfig
from meristem.paramgen.parameter_dataset import ParameterDataset
from meristem.paramgen.sampler import PosteriorSampler

EXAMPLES_DIR = Path(__file__).parents[2] / "examples" / "paramgen"

# run the CLI through this interpreter (as job scripts do), so the tests don't
# depend on the meristem console script being on PATH
MERISTEM = [sys.executable, "-m", "meristem"]

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
# Posterior parameter calibrated for only some PFTs
# ===========================================================================


def test_lh_posterior_uncovered_pfts_keep_defaults(
    ensemble_param_dir, default_param_file, tmp_path, posterior_file
):
    """A posterior parameter whose files cover only PFTs 0 and 1 is drawn for
    those PFTs and left at its default for PFT 2 in every written member
    """
    sources = tmp_path / "partial_posterior.yaml"
    sources.write_text(
        yaml.dump(
            {
                "fates_leafn_vert_scaler": {
                    "parameters": [
                        "fates_leafn_vert_scaler_coeff1",
                        "fates_leafn_vert_scaler_coeff2",
                    ],
                    "files": [{"path": str(posterior_file), "array_indices": [0, 1]}],
                }
            }
        ),
        encoding="utf-8",
    )
    ensemble = _make_lh_ensemble(
        ensemble_param_dir, default_param_file, tmp_path, sources
    )
    ensemble.create_ensemble()

    default_ds = ParameterDataset.from_path(default_param_file)
    members = sorted((tmp_path / "ensemble").glob(f"test_*{default_param_file.suffix}"))
    assert len(members) == 5
    for path in members:
        ds = ParameterDataset.from_path(path)
        for var in ("fates_leafn_vert_scaler_coeff1", "fates_leafn_vert_scaler_coeff2"):
            default = default_ds[var].values
            values = ds[var].values
            assert values[2] == pytest.approx(default[2]), (path.name, var)
            assert not np.allclose(values[:2], default[:2]), (path.name, var)


# ===========================================================================
# Posterior strategy on a parameter that writes one variable
# ===========================================================================


def test_lh_posterior_default_param_writes_draws(tmp_path, default_param_file):
    """A default (non-joint) parameter drawn from a posterior writes one
    posterior draw, for every PFT, into each member
    """
    param_dir = tmp_path / "params"
    param_dir.mkdir()
    pd.DataFrame(
        [
            {
                "parameter_name": "fates_leaf_slatop",
                "long_name": "SLA",
                "category": "stomatal",
                "subcategory": "photosynthesis",
                "units": "m^2/gC",
                "coord": "['fates_pft']",
                "param_type": "default",
                "strategy": "posterior",
                "param_min": "",
                "param_max": "",
                "slice_dim": None,
                "slice_index": None,
                "root_param": None,
                "base_params": "",
            }
        ]
    ).to_csv(param_dir / "main.csv", index=False)
    draws = np.linspace(0.005, 0.05, 21)
    posterior = tmp_path / "slatop_posterior.txt"
    posterior.write_text("fates_leaf_slatop\n" + "\n".join(map(str, draws)) + "\n")
    sources = tmp_path / "posterior_sources.yaml"
    sources.write_text(
        yaml.dump(
            {
                "fates_leaf_slatop": {
                    "parameters": ["fates_leaf_slatop"],
                    "files": [{"path": str(posterior), "array_indices": "all"}],
                }
            }
        ),
        encoding="utf-8",
    )

    _make_lh_ensemble(
        param_dir, default_param_file, tmp_path, sources
    ).create_ensemble()

    members = sorted((tmp_path / "ensemble").glob(f"test_*{default_param_file.suffix}"))
    assert len(members) == 5
    for path in members:
        values = ParameterDataset.from_path(path)["fates_leaf_slatop"].values
        assert np.allclose(values, values[0]), path.name  # one draw for every PFT
        assert np.isclose(draws, values[0]).any(), path.name  # it's a posterior draw


# draws in the fixtures' posterior files: a posterior value is a whole draw, so
# its normalized value is only known to within one draw's step
_POSTERIOR_DRAWS = 20


def _normalized_written(param, member: ParameterDataset, default_ds) -> np.ndarray:
    """The normalized value(s) of what a member file holds for a parameter: its
    own sampled value (get_default on the member, e.g. base - root for
    scale_from_root), normalized against the default file
    """
    normalized = param.normalize(param.get_default(member), default_ds)
    parts = normalized if isinstance(normalized, list) else [normalized]
    return np.concatenate([np.ravel(part) for part in parts])


def _tolerance(param) -> float:
    """How closely a parameter's written value can be normalized back"""
    if isinstance(param.sampler, PosteriorSampler):
        return 1.0 / (_POSTERIOR_DRAWS - 1)
    return 1e-9


def _assert_members_match_key(ensemble) -> int:
    """Check that every member file holds, for every parameter, the value its
    key row says was sampled (written and key agree, nothing is missing,
    truncated, or written in the wrong place)

    Returns:
        int: how many (member, parameter) values were checked
    """
    key = pd.read_csv(ensemble.ensemble_dir / f"{ensemble.file_prefix}_key.csv")
    extension = ensemble.default_ds.file_extension
    checked = 0
    for row in key.to_dict("records"):
        member = ParameterDataset.from_path(
            ensemble.ensemble_dir / f"{row['ensemble']}{extension}"
        )
        for group in ensemble.sampling_units:
            for param in group.params:
                normalized = _normalized_written(param, member, ensemble.default_ds)
                np.testing.assert_allclose(
                    normalized,
                    row[group.name],
                    atol=_tolerance(param),
                    err_msg=f"{row['ensemble']}: {param.spec.name}",
                )
                checked += 1
    return checked


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


def test_lh_members_match_key(
    ensemble_param_dir, default_param_file, tmp_path, posterior_config_file
):
    """Every LH member file holds, for every parameter (uniform, percent,
    scale_from_root, joint posterior), the value its key row says was sampled
    """
    ensemble = _make_lh_ensemble(
        ensemble_param_dir, default_param_file, tmp_path, posterior_config_file
    )
    ensemble.create_ensemble()
    assert _assert_members_match_key(ensemble) == 5 * len(ensemble.params)


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


def _variables(param) -> set[str]:
    """The variables a parameter writes"""
    return set(param.spec.base_params or [param.spec.original_name])


def test_oat_members_change_only_their_parameter(
    ensemble_param_dir, default_param_file, tmp_path, posterior_config_file
):
    """Every written OAT member has its perturbed parameter (or group) at its
    minimum or maximum, as its key row says, and every other variable exactly at
    its default
    """
    ensemble = _make_oat_ensemble(
        ensemble_param_dir, default_param_file, tmp_path, posterior_config_file
    )
    ensemble.create_ensemble()
    key = pd.read_csv(tmp_path / "ensemble" / "test_key.csv")
    assert len(key) == 2 * len(ensemble.sampling_units)
    groups = {group.name: group for group in ensemble.sampling_units}
    default = ensemble.default_ds
    extension = default.file_extension

    for row in key.to_dict("records"):
        member = ParameterDataset.from_path(
            tmp_path / "ensemble" / f"{row['ensemble']}{extension}"
        )
        _assert_oat_member(row, member, groups[row["parameter"]], default)


def _assert_oat_member(row: dict, member, group, default) -> None:
    """Check one OAT member: its perturbed parameters at the key's direction,
    every other numeric variable exactly at its default
    """
    expected = {"minimum": 0.0, "maximum": 1.0}[row["direction"]]
    for param in group.params:
        np.testing.assert_allclose(
            _normalized_written(param, member, default),
            expected,
            atol=_tolerance(param),
            err_msg=f"{row['ensemble']}: {param.spec.name}",
        )
    perturbed = set().union(*(_variables(param) for param in group.params))
    for var in default.data_vars:
        if var in perturbed or default[var].values.dtype.kind not in "fiu":
            continue
        np.testing.assert_array_equal(
            member[var].values,
            default[var].values,
            err_msg=f"{row['ensemble']} ({row['parameter']}): {var} changed",
        )


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


def test_grouped_lh_members_match_key(grouped_param_dir, default_param_file, tmp_path):
    """A grouped LH ensemble writes 5 members, each holding the values its key
    row says were sampled, for every parameter
    """
    ensemble = _make_lh_ensemble(grouped_param_dir, default_param_file, tmp_path)
    ensemble.create_ensemble()
    members = list((tmp_path / "ensemble").glob(f"test_*{default_param_file.suffix}"))
    assert len(members) == 5
    assert _assert_members_match_key(ensemble) == 5 * len(ensemble.params)


def test_grouped_params_move_together(grouped_param_dir, default_param_file, tmp_path):
    """In every written member, parameters in the same group (and, for an
    expanded group, the same index) sit at the same place in their ranges
    """
    ensemble = _make_lh_ensemble(grouped_param_dir, default_param_file, tmp_path)
    ensemble.create_ensemble()
    extension = ensemble.default_ds.file_extension
    multi_param_groups = [g for g in ensemble.sampling_units if len(g.params) > 1]
    assert multi_param_groups  # the fixture has groups to check

    for path in sorted(ensemble.ensemble_dir.glob(f"test_*{extension}")):
        member = ParameterDataset.from_path(path)
        for group in multi_param_groups:
            values = [
                _normalized_written(param, member, ensemble.default_ds)
                for param in group.params
            ]
            for param, normalized in zip(group.params[1:], values[1:]):
                np.testing.assert_allclose(
                    normalized,
                    values[0][0],
                    err_msg=f"{path.name}: {group.name}: {param.spec.name}",
                )


# ===========================================================================
# CLI tests
# ===========================================================================


def _copy_example(name: str, tmp_path: Path) -> Path:
    """Copy an example into tmp_path, without any earlier outputs, so the CLI
    runs from a clean directory and doesn't write into the repo."""
    dest = tmp_path / name
    shutil.copytree(
        EXAMPLES_DIR / name, dest, ignore=shutil.ignore_patterns("output_*")
    )
    return dest


@pytest.mark.slow
def test_cli_clm_lh_example(tmp_path):
    """Full CLI run against the CLM Latin Hypercube example."""
    example = _copy_example("clm_example", tmp_path)
    result = subprocess.run(
        [*MERISTEM, "paramgen", "lh_config.yaml"],
        cwd=example,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
    output_files = list((example / "output_LH").glob("clm_lh_*.nc"))
    assert len(output_files) == 20


@pytest.mark.slow
def test_cli_clm_oat_example(tmp_path):
    """Full CLI run against the CLM OAT example."""
    example = _copy_example("clm_example", tmp_path)
    result = subprocess.run(
        [*MERISTEM, "paramgen", "oat_config.yaml"],
        cwd=example,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
    assert (example / "output_OAT" / "clm_oat_key.csv").exists()


@pytest.mark.slow
def test_cli_fates_lh_example(tmp_path):
    """Full CLI run against the FATES Latin Hypercube example."""
    example = _copy_example("fates_example", tmp_path)
    result = subprocess.run(
        [*MERISTEM, "paramgen", "lh_config.yaml"],
        cwd=example,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
    output_files = list((example / "output_LH").glob("fates_lh_*.json"))
    assert len(output_files) == 20


@pytest.mark.slow
def test_cli_fates_oat_example(tmp_path):
    """Full CLI run against the FATES OAT example."""
    example = _copy_example("fates_example", tmp_path)
    result = subprocess.run(
        [*MERISTEM, "paramgen", "oat_config.yaml"],
        cwd=example,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
    assert (example / "output_OAT" / "fates_oat_key.csv").exists()
