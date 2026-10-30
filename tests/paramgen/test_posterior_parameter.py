"""Tests for posterior sampling on parameters that write one variable (default
and sliced), which the posterior sampler (built for joint parameters) handles as
a one-item list
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from meristem.paramgen.diagnostics import normalize_defaults
from meristem.paramgen.parameter import DimIndex, Parameter


def _posterior(tmp_path: Path, variable: str, low: float, high: float) -> dict:
    """A one-variable posterior config whose 21 sorted draws run low..high"""
    path = tmp_path / f"{variable}_posterior.txt"
    draws = np.linspace(low, high, 21)
    path.write_text(f"{variable}\n" + "\n".join(map(str, draws)) + "\n")
    return {
        "parameters": [variable],
        "files": [{"path": str(path), "array_indices": "all"}],
    }


@pytest.fixture
def posterior_default(default_row, param_dataset, tmp_path) -> Parameter:
    """fates_leaf_slatop (per PFT, default [0.01, 0.02, 0.03]) drawn from a
    posterior from 0.005 to 0.05, not expanded
    """
    default_row["strategy"] = "posterior"
    default_row["param_min"] = ""
    default_row["param_max"] = ""
    config = _posterior(tmp_path, "fates_leaf_slatop", 0.005, 0.05)
    return Parameter.from_row(default_row, param_dataset, posterior_config=config)


def test_posterior_default_samples_array(posterior_default, param_dataset):
    """Test that sample() gives a plain array of the parameter's values (one
    draw for every PFT), not a one-item list
    """
    value = posterior_default.sample(0.5, param_dataset)
    assert isinstance(value, np.ndarray)
    np.testing.assert_allclose(value, [0.0275] * 3)


@pytest.mark.parametrize("u, expected", [(0.0, 0.005), (1.0, 0.05)])
def test_posterior_default_set_value_writes_draw(
    posterior_default, param_dataset, working_param_dataset, u, expected
):
    """Test that a sampled value can be written (it used to fail with a (1, 3)
    vs (3,) shape error)
    """
    value = posterior_default.sample(u, param_dataset)
    posterior_default.set_value(working_param_dataset, param_dataset, value)
    np.testing.assert_allclose(
        working_param_dataset["fates_leaf_slatop"].values, [expected] * 3
    )


def test_posterior_default_normalize(posterior_default, param_dataset):
    """Test that normalize() takes the parameter's plain array and gives back
    a plain array of normalized values
    """
    normalized = posterior_default.normalize(np.array([0.0275] * 3), param_dataset)
    assert isinstance(normalized, np.ndarray)
    np.testing.assert_allclose(normalized, [0.5] * 3)


def test_posterior_default_expanded_still_works(
    posterior_default, param_dataset, working_param_dataset
):
    """Test that the expanded (one PFT at a time) case samples a scalar, writes
    only its PFT, and normalizes as before
    """
    posterior_default.active_index = DimIndex("fates_pft", 1)
    value = posterior_default.sample(1.0, param_dataset)
    assert np.size(value) == 1
    posterior_default.set_value(working_param_dataset, param_dataset, value)
    np.testing.assert_allclose(
        working_param_dataset["fates_leaf_slatop"].values, [0.01, 0.05, 0.03]
    )
    assert posterior_default.normalize(0.05, param_dataset) == pytest.approx(1.0)


def test_posterior_sliced_samples_and_writes(
    sliced_row, param_dataset, working_param_dataset, tmp_path
):
    """Test that a sliced parameter drawn from a posterior samples and writes
    its slice (leafage class 0) and leaves the other slice alone
    """
    sliced_row["strategy"] = "posterior"
    sliced_row["param_min"] = ""
    sliced_row["param_max"] = ""
    config = _posterior(tmp_path, "fates_leaf_vcmax25top", 20.0, 90.0)
    param = Parameter.from_row(sliced_row, param_dataset, posterior_config=config)
    before = working_param_dataset["fates_leaf_vcmax25top"].values.copy()

    value = param.sample(1.0, param_dataset)
    param.set_value(working_param_dataset, param_dataset, value)

    after = working_param_dataset["fates_leaf_vcmax25top"].values
    np.testing.assert_allclose(after[0], [90.0] * 3)
    np.testing.assert_allclose(after[1], before[1])


def test_posterior_joint_unchanged(joint_param, param_dataset):
    """Test that a joint parameter still samples and normalizes one array per
    variable
    """
    value = joint_param.sample(0.5, param_dataset)
    assert isinstance(value, list) and len(value) == 2
    normalized = joint_param.normalize(value, param_dataset)
    assert isinstance(normalized, list) and len(normalized) == 2


def test_normalize_defaults_posterior_default_param(tmp_path, default_param_file):
    """Test that diagnostics can place a posterior default parameter's defaults
    (it used to fail inside the posterior source)
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
    sources = tmp_path / "posterior_sources.yaml"
    config = _posterior(tmp_path, "fates_leaf_slatop", 0.005, 0.05)
    sources.write_text(
        "fates_leaf_slatop:\n"
        "  parameters: [fates_leaf_slatop]\n"
        f"  files: [{{path: {config['files'][0]['path']}, array_indices: all}}]\n"
    )

    df = normalize_defaults(param_dir, default_param_file, posterior_sources=sources)
    assert len(df) == 3
    # each PFT's default (0.01, 0.02, 0.03) at its own place among the draws
    draws = np.linspace(0.005, 0.05, 21)
    expected = np.searchsorted(draws, [0.01, 0.02, 0.03]) / (len(draws) - 1)
    np.testing.assert_allclose(df["normalized_value"], expected)
    assert df["normalized_value"].is_monotonic_increasing
