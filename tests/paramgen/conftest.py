"""Fixtures shared across param_ens_gen tests."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import xarray as xr
import pytest
import yaml


from param_ens_gen.posterior_source import PosteriorSource
from param_ens_gen.parameter import (
    DefaultParameter,
    JointParameter,
    ScaleFromRootParameter,
    SlicedParameter,
)
from param_ens_gen.ensemble_config import LatinHypercubeConfig, OneAtATimeConfig
from param_ens_gen.param_ensemble import (
    LatinHypercubeEnsemble,
    OneAtATimeEnsemble,
)
from param_ens_gen.parameter_dataset import (
    NetCDFParameterDataset,
    FATESJSONParameterDataset,
)

N_PFTS = 3
N_LEAFAGE = 2
N_ORGANS = 4

# ========================================================================================
# ParamSpec fixtures
# ========================================================================================


def _base_row(**kwargs) -> pd.Series:
    """Build a minimal valid main-sheet row, with overrides."""
    defaults = {
        "parameter_name": "fates_leaf_slatop",
        "long_name": "Specific Leaf Area (SLA) at top of canopy, projected area basis",
        "category": "stomatal",
        "subcategory": "photosynthesis",
        "units": "m^2/gC",
        "coord": "['fates_pft']",
        "param_type": "default",
        "strategy": "uniform",
        "param_min": "0.005",
        "param_max": "0.05",
        "slice_dim": None,
        "slice_index": None,
        "root_param": None,
        "base_params": "",
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


@pytest.fixture
def default_row() -> pd.Series:
    """A valid row for a default PFT parameter.

    Returns:
        pd.Series: spreadsheet row
    """
    return _base_row()


@pytest.fixture
def scalar_row() -> pd.Series:
    """A valid row for a scalar (non-PFT) default parameter.

    Returns:
        pd.Series: spreadsheet row
    """
    return _base_row(
        parameter_name="fates_canopy_closure_thresh",
        long_name="canopy coverage when crown area allometry changes from savanna to forest",
        category="biogeochemistry",
        subcategory="vegetation dynamics",
        units="1/yr",
        coord="[]",
        param_min="0.7",
        param_max="0.9",
    )


@pytest.fixture
def multi_dim_row() -> pd.Series:
    """A valid row for a multi-dimensional default parameter.

    Returns:
        pd.Series: spreadsheet row
    """
    return _base_row(
        parameter_name="parameter_a",
        coord="['dim_1', 'dim_2']",
    )


@pytest.fixture
def sliced_row() -> pd.Series:
    """A valid row for a sliced parameter (fates_leafage_class x fates_pft).

    Returns:
        pd.Series: spreadsheet row
    """
    return _base_row(
        parameter_name="fates_leaf_vcmax25top",
        long_name="maximum carboxylation rate of Rub. at 25C, canopy top",
        category="stomatal",
        subcategory="photosynthesis",
        units="umol CO2/m^2/s",
        coord="['fates_leafage_class', 'fates_pft']",
        param_type="sliced",
        param_min="20.0",
        param_max="90.0",
        slice_dim="fates_leafage_class",
        slice_index=0,
        base_params="fates_leaf_vcmax25top",
    )


@pytest.fixture
def scale_from_root_row() -> pd.Series:
    """A valid row for a scale_from_root parameter.

    Returns:
        pd.Series: spreadsheet row
    """
    return _base_row(
        parameter_name="smpsc_delta",
        long_name="Soil water potential at full stomatal closing (delta)",
        category="stomatal",
        subcategory="vegetation water",
        units="mm",
        coord="['fates_pft']",
        param_type="scale_from_root",
        param_min="-600000",
        param_max="-20000",
        root_param="fates_nonhydro_smpso",
        base_params="fates_nonhydro_smpsc",
    )


@pytest.fixture
def root_row() -> pd.Series:
    """A valid row for a root parameter.

    Returns:
        pd.Series: spreadsheet row
    """
    return _base_row(
        parameter_name="fates_nonhydro_smpso",
        coord="['fates_pft']",
        param_type="default",
        param_min="-200000",
        param_max="-10000",
    )


@pytest.fixture
def mutually_dependent_rows() -> tuple[pd.Series, pd.Series]:
    """Two mutually-dependent scale_from_root parameters"""

    row_a = _base_row(
        parameter_name="param_a",
        coord="['fates_pft']",
        param_type="scale_from_root",
        param_min="-600000",
        param_max="-20000",
        root_param="fates_nonhydro_smpso",
        base_params="fates_nonhydro_smpsc",
    )
    row_b = _base_row(
        parameter_name="param_b",
        coord="['fates_pft']",
        param_type="scale_from_root",
        param_min="-600000",
        param_max="-20000",
        root_param="fates_nonhydro_smpsc",
        base_params="fates_nonhydro_smpso",
    )

    return row_a, row_b


@pytest.fixture
def joint_param_row() -> pd.Series:
    """A valid row for a joint parameter.

    Returns:
        pd.Series: spreadsheet row
    """
    return _base_row(
        parameter_name="fates_leafn_vert_scaler",
        long_name="Coefficient one for decrease in leaf nitrogen through the canopy",
        category="stomatal",
        subcategory="photosynthesis",
        units="unitless",
        coord="['fates_pft']",
        param_type="joint",
        strategy="posterior",
        param_min="",
        param_max="",
        base_params="['fates_leafn_vert_scaler_coeff1', 'fates_leafn_vert_scaler_coeff2']",
    )


# ========================================================================================
# PosteriorSource fixtures
# ========================================================================================


@pytest.fixture
def posterior_file(tmp_path) -> Path:
    """A minimal posterior text file with two parameters and 20 rows.

    Columns: param_a, param_b
    Values are shuffled so that _load's sort behaviour is actually tested.

    Returns:
        Path: path to the temporary file
    """
    rng = np.random.default_rng(42)
    n = 20
    data = pd.DataFrame(
        {
            "fates_leafn_vert_scaler_coeff1": rng.permutation(
                np.linspace(0.0, 0.95, n)
            ),
            "fates_leafn_vert_scaler_coeff2": rng.permutation(
                np.linspace(10.0, 19.5, n)
            ),
        }
    )
    path = tmp_path / "posterior.txt"
    data.to_csv(path, sep=" ", index=False)
    return path


@pytest.fixture
def posterior_file_2(tmp_path) -> Path:
    """A second minimal posterior file for multi-source tests."""
    rng = np.random.default_rng(99)
    n = 20
    data = pd.DataFrame(
        {
            "fates_leafn_vert_scaler_coeff1": rng.permutation(
                np.linspace(1.0, 1.95, n)
            ),
            "fates_leafn_vert_scaler_coeff2": rng.permutation(
                np.linspace(20.0, 29.5, n)
            ),
        }
    )
    path = tmp_path / "posterior_2.txt"
    data.to_csv(path, sep=" ", index=False)
    return path


@pytest.fixture
def empty_posterior_file(tmp_path) -> Path:
    """An empty posterior text file with two parameters.

    Columns: fates_leafn_vert_scaler_coeff1, fates_leafn_vert_scaler_coeff2

    Returns:
        Path: path to the temporary file
    """
    data = pd.DataFrame(
        {
            "fates_leafn_vert_scaler_coeff1": [],
            "fates_leafn_vert_scaler_coeff2": [],
        }
    )
    path = tmp_path / "empty_posterior.txt"
    data.to_csv(path, sep=" ", index=False)
    return path


@pytest.fixture
def single_row_posterior(tmp_path):
    """A single-row posterior source"""
    file = tmp_path / "posterior.txt"
    file.write_text("leaf_cn\n25.0\n")
    return PosteriorSource(
        path=file,
        array_indices="all",
        parameters=["leaf_cn"],
    )


@pytest.fixture
def posterior_source(posterior_file) -> PosteriorSource:
    """A PosteriorSource pointing at posterior_file, not yet loaded.

    Returns:
        PosteriorSource: unloaded source
    """
    return PosteriorSource(
        path=posterior_file,
        array_indices="all",
        parameters=["fates_leafn_vert_scaler_coeff1", "fates_leafn_vert_scaler_coeff2"],
    )


@pytest.fixture
def empty_posterior_source(empty_posterior_file) -> PosteriorSource:
    """A PosteriorSource pointing at empty_posterior_file, not yet loaded.

    Returns:
        PosteriorSource: unloaded source
    """
    return PosteriorSource(
        path=empty_posterior_file,
        array_indices="all",
        parameters=["fates_leafn_vert_scaler_coeff1", "fates_leafn_vert_scaler_coeff2"],
    )


# ===========================================================================
# DistributionStat fixtures
# ===========================================================================


@pytest.fixture
def percent_row() -> pd.Series:
    """A valid row for a parameter with percent-based bounds.

    Returns:
        pd.Series: spreadsheet row
    """
    return _base_row(
        parameter_name="fates_leaf_vcmax25top",
        coord="['fates_leafage_class', 'fates_pft']",
        param_min="50percent",
        param_max="50percent",
    )


@pytest.fixture
def pft_sheet() -> pd.DataFrame:
    """A minimal per-parameter PFT bounds sheet with 3 PFTs.

    Returns:
        pd.DataFrame: PFT bounds sheet
    """
    return pd.DataFrame(
        {
            "pft_index": [1, 2, 3],
            "pft_name": ["white_spruce", "black_spruce", "deciduous"],
            "param_min": [0.005, 0.004, 0.008],
            "param_max": [0.040, 0.035, 0.060],
        }
    )


@pytest.fixture
def pft_row() -> pd.Series:
    """A valid row for a parameter with PFT-specific bounds.

    Returns:
        pd.Series: spreadsheet row
    """
    return _base_row(
        parameter_name="fates_leaf_slatop",
        param_min="pft",
        param_max="pft",
    )


# ===========================================================================
# Sampler fixtures
# ===========================================================================


@pytest.fixture
def posterior_config(posterior_file) -> dict:
    """A minimal posterior_config dict pointing at posterior_file.

    Covers a single broadcast source (array_indices='all') for two parameters.

    Returns:
        dict: posterior config
    """
    return {
        "parameters": [
            "fates_leafn_vert_scaler_coeff1",
            "fates_leafn_vert_scaler_coeff2",
        ],
        "files": [
            {
                "path": str(posterior_file),
                "array_indices": "all",
            }
        ],
    }


@pytest.fixture
def multi_source_posterior_config(posterior_file, posterior_file_2) -> dict:
    """A posterior_config dict pointing at two non-broadcast sources covering different indices.

    Returns:
        dict: posterior config
    """
    return {
        "parameters": [
            "fates_leafn_vert_scaler_coeff1",
            "fates_leafn_vert_scaler_coeff2",
        ],
        "files": [
            {
                "path": str(posterior_file),
                "array_indices": [0, 1],
            },
            {"path": str(posterior_file_2), "array_indices": [2]},
        ],
    }


@pytest.fixture
def pft_uniform_row() -> pd.Series:
    """A valid row for a uniform parameter with PFT-specific bounds."""
    return _base_row(
        parameter_name="fates_leaf_slatop",
        strategy="uniform",
        param_min="pft",
        param_max="pft",
    )


# ===========================================================================
# Parameter fixtures
# ===========================================================================


@pytest.fixture
def default_ds() -> xr.Dataset:
    """A minimal default FATES parameter dataset covering all param types.

    Returns:
        xr.Dataset: default dataset
    """
    return xr.Dataset(
        {
            # default PFT parameter
            "fates_leaf_slatop": (
                ["fates_pft"],
                np.array([0.010, 0.020, 0.030]),
            ),
            # scalar parameter
            "fates_canopy_closure_thresh": (
                [],
                np.float64(0.5),
            ),
            # sliced parameter (leafage_class x pft)
            "fates_leaf_vcmax25top": (
                ["fates_leafage_class", "fates_pft"],
                np.array([[50.0, 60.0, 70.0], [40.0, 50.0, 60.0]]),
            ),
            # scale_from_root — root and dependent
            "fates_nonhydro_smpso": (
                ["fates_pft"],
                np.array([-50000.0, -60000.0, -70000.0]),
            ),
            "fates_nonhydro_smpsc": (
                ["fates_pft"],
                np.array([-100000.0, -110000.0, -120000.0]),
            ),
            # joint targets
            "fates_leafn_vert_scaler_coeff1": (
                ["fates_pft"],
                np.array([0.012, 0.015, 0.005]),
            ),
            "fates_leafn_vert_scaler_coeff2": (
                ["fates_pft"],
                np.array([2.1, 2.5, 2.6]),
            ),
            "parameter_a": (
                ["dim_1", "dim_2"],
                np.array([[50.0, 60.0, 70.0], [40.0, 50.0, 60.0]]),
            ),
        }
    )


@pytest.fixture
def working_ds(default_ds) -> xr.Dataset:
    """A deep copy of default_ds to use as the working dataset.

    Args:
        default_ds (xr.Dataset): fixture

    Returns:
        xr.Dataset: working copy
    """
    return default_ds.copy(deep=True)


# =============================================================================
# Parameter fixtures
# =============================================================================


@pytest.fixture(params=["netcdf", "json"])
def default_param(request, default_row, default_ds, json_dataset):
    """DefaultParameter for fates_leaf_slatop, parametrized over backends."""
    ds = (
        NetCDFParameterDataset(default_ds)
        if request.param == "netcdf"
        else json_dataset
    )
    return DefaultParameter(default_row, ds)


@pytest.fixture(params=["netcdf", "json"])
def multi_dim_param(request, multi_dim_row, default_ds, json_dataset):
    """DefaultParameter for a multi-dimensional parameter, parametrized over backends."""
    ds = (
        NetCDFParameterDataset(default_ds)
        if request.param == "netcdf"
        else json_dataset
    )
    return DefaultParameter(multi_dim_row, ds)


@pytest.fixture(params=["netcdf", "json"])
def scalar_param(request, scalar_row, default_ds, json_dataset):
    """DefaultParameter for scalar fates_canopy_closure_thresh, parametrized over backends."""
    ds = (
        NetCDFParameterDataset(default_ds)
        if request.param == "netcdf"
        else json_dataset
    )
    return DefaultParameter(scalar_row, ds)


@pytest.fixture(params=["netcdf", "json"])
def sliced_param(request, sliced_row, default_ds, json_dataset):
    """SlicedParameter for fates_leaf_vcmax25top, parametrized over backends."""
    ds = (
        NetCDFParameterDataset(default_ds)
        if request.param == "netcdf"
        else json_dataset
    )
    return SlicedParameter(sliced_row, ds)


@pytest.fixture(params=["netcdf", "json"])
def scale_param(request, scale_from_root_row, default_ds, json_dataset):
    """ScaleFromRootParameter, parametrized over backends."""
    ds = (
        NetCDFParameterDataset(default_ds)
        if request.param == "netcdf"
        else json_dataset
    )
    return ScaleFromRootParameter(scale_from_root_row, ds)


@pytest.fixture(params=["netcdf", "json"])
def joint_param(request, joint_param_row, default_ds, json_dataset, posterior_config):
    """JointParameter, parametrized over backends."""
    ds = (
        NetCDFParameterDataset(default_ds)
        if request.param == "netcdf"
        else json_dataset
    )
    return JointParameter(joint_param_row, ds, posterior_config=posterior_config)


# =============================================================================
# EnsembleConfig fixtures
# =============================================================================


@pytest.fixture
def base_paths(tmp_path):
    """Base paths for EnsembleConfig"""
    return {
        "param_dir": tmp_path,
        "ensemble_dir": tmp_path / "ensemble",
        "file_prefix": "test_ensemble",
        "default_param_file": tmp_path / "default.nc",
    }


# =============================================================================
# param_dir fixtures
# =============================================================================


@pytest.fixture
def param_dir(tmp_path) -> Path:
    """A parameter dir with main and pft sheets"""
    main = pd.DataFrame(
        {
            "parameter_name": ["fates_leaf_slatop"],
            "param_type": ["default"],
        }
    )
    main.to_csv(tmp_path / "main.csv", index=False)

    pft = pd.DataFrame(
        {
            "pft_index": [1, 2, 3],
            "param_min": [0.005, 0.004, 0.008],
            "param_max": [0.040, 0.035, 0.060],
        }
    )
    pft.to_csv(tmp_path / "fates_leaf_slatop.csv", index=False)
    return tmp_path


@pytest.fixture
def ensemble_param_dir(tmp_path) -> Path:
    """A param_dir with a complete valid main.csv for ensemble tests."""
    pd.DataFrame(
        [
            {
                "parameter_name": "fates_leaf_slatop",
                "long_name": "SLA at top of canopy",
                "category": "stomatal",
                "subcategory": "photosynthesis",
                "units": "m^2/gC",
                "coord": "['fates_pft']",
                "param_type": "default",
                "strategy": "uniform",
                "param_min": "0.005",
                "param_max": "0.05",
                "slice_dim": None,
                "slice_index": None,
                "root_param": None,
                "base_params": "",
            },
            {
                "parameter_name": "fates_leaf_vcmax25top",
                "long_name": "maximum carboxylation rate of Rub. at 25C, canopy top",
                "category": "stomatal",
                "subcategory": "photosynthesis",
                "units": "umol CO2/m^2/s",
                "coord": "['fates_leafage_class', 'fates_pft']",
                "param_type": "sliced",
                "strategy": "uniform",
                "param_min": "20.0",
                "param_max": "90.0",
                "slice_dim": "fates_leafage_class",
                "slice_index": 0,
                "root_param": None,
                "base_params": "fates_leaf_vcmax25top",
            },
            {
                "parameter_name": "fates_canopy_closure_thresh",
                "long_name": "canopy coverage at which crown area allometry changes",
                "category": "biogeochemistry",
                "subcategory": "vegetation dynamics",
                "units": "1/yr",
                "coord": "[]",
                "param_type": "default",
                "strategy": "uniform",
                "param_min": "0.1",
                "param_max": "0.7",
                "slice_dim": None,
                "slice_index": None,
                "root_param": None,
                "base_params": None,
            },
            {
                "parameter_name": "smpsc_delta",
                "long_name": "Soil water potential at full stomatal closing (delta)",
                "category": "stomatal",
                "subcategory": "vegetation water",
                "units": "mm",
                "coord": "['fates_pft']",
                "param_type": "scale_from_root",
                "strategy": "uniform",
                "param_min": "-600000",
                "param_max": "-20000",
                "slice_dim": None,
                "slice_index": None,
                "root_param": "fates_nonhydro_smpso",
                "base_params": "fates_nonhydro_smpsc",
            },
            {
                "parameter_name": "fates_nonhydro_smpso",
                "long_name": "Soil water potential at full stomatal opening",
                "category": "stomatal",
                "subcategory": "vegetation water",
                "units": "mm",
                "coord": "['fates_pft']",
                "param_type": "default",
                "strategy": "uniform",
                "param_min": "20percent",
                "param_max": "20percent",
                "slice_dim": None,
                "slice_index": None,
                "root_param": None,
                "base_params": None,
            },
            {
                "parameter_name": "fates_leafn_vert_scaler",
                "long_name": "Coefficient for decrease in leaf nitrogen through the canopy",
                "category": "stomatal",
                "subcategory": "photosynthesis",
                "units": "unitless",
                "coord": "['fates_pft']",
                "param_type": "joint",
                "strategy": "posterior",
                "param_min": "",
                "param_max": "",
                "slice_dim": None,
                "slice_index": None,
                "root_param": None,
                "base_params": (
                    "['fates_leafn_vert_scaler_coeff1',"
                    " 'fates_leafn_vert_scaler_coeff2']"
                ),
            },
        ]
    ).to_csv(tmp_path / "main.csv", index=False)
    return tmp_path


@pytest.fixture
def default_netcdf_file(tmp_path, default_ds) -> Path:
    """Write default_ds to a temp netCDF file and return the path."""
    path = tmp_path / "default.nc"
    default_ds.to_netcdf(path)
    return path


@pytest.fixture
def posterior_config_file(tmp_path, posterior_file) -> Path:
    """A minimal posterior config YAML file pointing at posterior_file."""
    config = {
        "fates_leafn_vert_scaler": {
            "parameters": [
                "fates_leafn_vert_scaler_coeff1",
                "fates_leafn_vert_scaler_coeff2",
            ],
            "files": [
                {
                    "path": str(posterior_file),
                    "array_indices": "all",
                }
            ],
        }
    }
    path = tmp_path / "posterior_sources.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    return path


@pytest.fixture
def grouped_param_dir(tmp_path) -> Path:
    """A param_dir with grouped and ungrouped parameters."""
    pd.DataFrame(
        [
            {
                "parameter_name": "fates_leaf_slatop",
                "long_name": "SLA at top of canopy",
                "category": "stomatal",
                "subcategory": "photosynthesis",
                "units": "m^2/gC",
                "coord": "['fates_pft']",
                "param_type": "default",
                "strategy": "uniform",
                "param_min": "0.005",
                "param_max": "0.05",
                "slice_dim": None,
                "slice_index": None,
                "root_param": None,
                "base_params": "",
                "group_name": "photosynthesis",
            },
            {
                "parameter_name": "fates_leaf_vcmax25top",
                "long_name": "maximum carboxylation rate of Rub. at 25C, canopy top",
                "category": "stomatal",
                "subcategory": "photosynthesis",
                "units": "umol CO2/m^2/s",
                "coord": "['fates_leafage_class', 'fates_pft']",
                "param_type": "sliced",
                "strategy": "uniform",
                "param_min": "20.0",
                "param_max": "90.0",
                "slice_dim": "fates_leafage_class",
                "slice_index": 0,
                "root_param": None,
                "base_params": "fates_leaf_vcmax25top",
                "group_name": "photosynthesis",
            },
            {
                "parameter_name": "fates_nonhydro_smpso",
                "long_name": "Soil water potential at full stomatal opening",
                "category": "stomatal",
                "subcategory": "vegetation water",
                "units": "mm",
                "coord": "['fates_pft']",
                "param_type": "default",
                "strategy": "uniform",
                "param_min": "20percent",
                "param_max": "20percent",
                "slice_dim": None,
                "slice_index": None,
                "root_param": None,
                "base_params": None,
                "group_name": "water",
                "expand_dim": "fates_pft",
            },
            {
                "parameter_name": "smpsc_delta",
                "long_name": "Soil water potential at full stomatal closing (delta)",
                "category": "stomatal",
                "subcategory": "vegetation water",
                "units": "mm",
                "coord": "['fates_pft']",
                "param_type": "scale_from_root",
                "strategy": "uniform",
                "param_min": "-600000",
                "param_max": "-20000",
                "slice_dim": None,
                "slice_index": None,
                "root_param": "fates_nonhydro_smpso",
                "base_params": "fates_nonhydro_smpsc",
                "group_name": "water",
                "expand_dim": "fates_pft",
            },
            {
                "parameter_name": "fates_canopy_closure_thresh",
                "long_name": "canopy coverage at which crown area allometry changes",
                "category": "biogeochemistry",
                "subcategory": "vegetation dynamics",
                "units": "1/yr",
                "coord": "[]",
                "param_type": "default",
                "strategy": "uniform",
                "param_min": "0.1",
                "param_max": "0.7",
                "slice_dim": None,
                "slice_index": None,
                "root_param": None,
                "base_params": None,
                "group_name": None,
            },
        ]
    ).to_csv(tmp_path / "main.csv", index=False)
    return tmp_path


@pytest.fixture
def mixed_expand_dim_group_dir(tmp_path) -> Path:
    """A param_dir with a group where parameters have mixed expand_dim values."""
    pd.DataFrame(
        [
            {
                "parameter_name": "fates_leaf_slatop",
                "long_name": "SLA at top of canopy",
                "category": "stomatal",
                "subcategory": "photosynthesis",
                "units": "m^2/gC",
                "coord": "['fates_pft']",
                "param_type": "default",
                "strategy": "uniform",
                "param_min": "0.005",
                "param_max": "0.05",
                "slice_dim": None,
                "slice_index": None,
                "root_param": None,
                "base_params": "",
                "group_name": "bad_group",
                "expand_dim": "fates_pft",
            },
            {
                "parameter_name": "fates_canopy_closure_thresh",
                "long_name": "canopy coverage at which crown area allometry changes",
                "category": "biogeochemistry",
                "subcategory": "vegetation dynamics",
                "units": "1/yr",
                "coord": "[]",
                "param_type": "default",
                "strategy": "uniform",
                "param_min": "0.1",
                "param_max": "0.7",
                "slice_dim": None,
                "slice_index": None,
                "root_param": None,
                "base_params": None,
                "group_name": "bad_group",
                "expand_dim": None,
            },
        ]
    ).to_csv(tmp_path / "main.csv", index=False)
    return tmp_path


@pytest.fixture
def scale_from_root_different_group_dir(tmp_path) -> Path:
    """A param_dir where scale_from_root and its root are in different groups."""
    pd.DataFrame(
        [
            {
                "parameter_name": "fates_nonhydro_smpso",
                "long_name": "Soil water potential at full stomatal opening",
                "category": "stomatal",
                "subcategory": "vegetation water",
                "units": "mm",
                "coord": "['fates_pft']",
                "param_type": "default",
                "strategy": "uniform",
                "param_min": "20percent",
                "param_max": "20percent",
                "slice_dim": None,
                "slice_index": None,
                "root_param": None,
                "base_params": None,
                "group_name": "water",
            },
            {
                "parameter_name": "smpsc_delta",
                "long_name": "Soil water potential at full stomatal closing (delta)",
                "category": "stomatal",
                "subcategory": "vegetation water",
                "units": "mm",
                "coord": "['fates_pft']",
                "param_type": "scale_from_root",
                "strategy": "uniform",
                "param_min": "-600000",
                "param_max": "-20000",
                "slice_dim": None,
                "slice_index": None,
                "root_param": "fates_nonhydro_smpso",
                "base_params": "fates_nonhydro_smpsc",
                "group_name": "soil",
            },
        ]
    ).to_csv(tmp_path / "main.csv", index=False)
    return tmp_path


# =============================================================================
# ParamEnsemble fixtures
# =============================================================================


@pytest.fixture(params=["netcdf", "json"])
def lh_ensemble(
    request,
    ensemble_param_dir,
    default_netcdf_file,
    json_param_file,
    tmp_path,
    posterior_config_file,
):
    """A LatinHypercube ensemble parametrized over NetCDF and JSON backends."""
    param_file = default_netcdf_file if request.param == "netcdf" else json_param_file
    config = LatinHypercubeConfig(
        param_dir=ensemble_param_dir,
        ensemble_dir=tmp_path / "ensemble",
        file_prefix="test",
        default_param_file=param_file,
        posterior_sources=posterior_config_file,
        ensemble_members=5,
    )
    return LatinHypercubeEnsemble(config)


@pytest.fixture(params=["netcdf", "json"])
def oat_ensemble(
    request,
    ensemble_param_dir,
    default_netcdf_file,
    json_param_file,
    tmp_path,
    posterior_config_file,
):
    """A OneAtATime ensemble parametrized over NetCDF and JSON backends."""
    param_file = default_netcdf_file if request.param == "netcdf" else json_param_file
    config = OneAtATimeConfig(
        param_dir=ensemble_param_dir,
        ensemble_dir=tmp_path / "ensemble",
        file_prefix="test",
        default_param_file=param_file,
        posterior_sources=posterior_config_file,
    )
    return OneAtATimeEnsemble(config)


@pytest.fixture
def expand_param_dir(tmp_path) -> Path:
    """A param_dir with one expandable and one non-expandable parameter."""
    pd.DataFrame(
        [
            {
                "parameter_name": "fates_leaf_slatop",
                "long_name": "SLA at top of canopy",
                "category": "stomatal",
                "subcategory": "photosynthesis",
                "units": "m^2/gC",
                "coord": "['fates_pft']",
                "param_type": "default",
                "strategy": "uniform",
                "param_min": "0.005",
                "param_max": "0.05",
                "slice_dim": None,
                "slice_index": None,
                "root_param": None,
                "base_params": "",
                "expand_dim": "fates_pft",
            },
            {
                "parameter_name": "fates_canopy_closure_thresh",
                "long_name": "canopy closure threshold",
                "category": "biogeochemistry",
                "subcategory": "vegetation dynamics",
                "units": "1/yr",
                "coord": "[]",
                "param_type": "default",
                "strategy": "uniform",
                "param_min": "0.1",
                "param_max": "0.7",
                "slice_dim": None,
                "slice_index": None,
                "root_param": None,
                "base_params": None,
                "expand_dim": None,
            },
        ]
    ).to_csv(tmp_path / "main.csv", index=False)
    return tmp_path


@pytest.fixture(params=["netcdf", "json"])
def lh_expand_ensemble(
    request, expand_param_dir, default_param_file, json_param_file, tmp_path
):
    """A LH ensemble with expanded dims, parametrized over NetCDF and JSON backends."""
    param_file = default_param_file if request.param == "netcdf" else json_param_file
    config = LatinHypercubeConfig(
        param_dir=expand_param_dir,
        ensemble_dir=tmp_path / "ensemble",
        file_prefix="test",
        default_param_file=param_file,
        ensemble_members=5,
    )
    return LatinHypercubeEnsemble(config)


@pytest.fixture
def netcdf_dataset(default_ds) -> NetCDFParameterDataset:
    """A NetCDFParameterDataset wrapping default_ds."""
    return NetCDFParameterDataset(default_ds)


@pytest.fixture
def json_param_file(tmp_path) -> Path:
    """A minimal FATES-style JSON parameter file mirroring default_ds."""
    data = {
        "attributes": {"history": "test fixture"},
        "dimensions": {
            "fates_pft": 3,
            "fates_leafage_class": 2,
            "dim_1": 2,
            "dim_2": 3,
        },
        "parameters": {
            "fates_leaf_slatop": {
                "dtype": "float",
                "dims": ["fates_pft"],
                "long_name": "SLA at top of canopy",
                "units": "m^2/gC",
                "data": [0.010, 0.020, 0.030],
            },
            "fates_canopy_closure_thresh": {
                "dtype": "float",
                "dims": ["scalar"],
                "long_name": "canopy closure threshold",
                "units": "1/yr",
                "data": [0.5],
            },
            "fates_leaf_vcmax25top": {
                "dtype": "float",
                "dims": ["fates_leafage_class", "fates_pft"],
                "long_name": "maximum carboxylation rate",
                "units": "umol CO2/m^2/s",
                "data": [[50.0, 60.0, 70.0], [40.0, 50.0, 60.0]],
            },
            "fates_nonhydro_smpso": {
                "dtype": "float",
                "dims": ["fates_pft"],
                "long_name": "soil water potential at full stomatal opening",
                "units": "mm",
                "data": [-50000.0, -60000.0, -70000.0],
            },
            "fates_nonhydro_smpsc": {
                "dtype": "float",
                "dims": ["fates_pft"],
                "long_name": "soil water potential at full stomatal closing",
                "units": "mm",
                "data": [-100000.0, -110000.0, -120000.0],
            },
            "fates_leafn_vert_scaler_coeff1": {
                "dtype": "float",
                "dims": ["fates_pft"],
                "long_name": "leaf nitrogen scaler coeff1",
                "units": "unitless",
                "data": [0.012, 0.015, 0.005],
            },
            "fates_leafn_vert_scaler_coeff2": {
                "dtype": "float",
                "dims": ["fates_pft"],
                "long_name": "leaf nitrogen scaler coeff2",
                "units": "unitless",
                "data": [2.1, 2.5, 2.6],
            },
            "parameter_a": {
                "dtype": "float",
                "dims": ["dim_1", "dim_2"],
                "long_name": "test multi-dim parameter",
                "units": "unitless",
                "data": [[50.0, 60.0, 70.0], [40.0, 50.0, 60.0]],
            },
        },
    }
    path = tmp_path / "default.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


@pytest.fixture
def json_dataset(json_param_file) -> FATESJSONParameterDataset:
    """A FATESJSONParameterDataset loaded from json_param_file."""
    return FATESJSONParameterDataset.load(json_param_file)


@pytest.fixture(params=["netcdf", "json"])
def param_dataset(request, default_ds, json_dataset):
    """A ParameterDataset fixture parametrized over NetCDF and JSON backends."""
    if request.param == "netcdf":
        return NetCDFParameterDataset(default_ds)
    return json_dataset


@pytest.fixture(params=["netcdf", "json"])
def working_param_dataset(request, default_ds, json_dataset):
    """A working copy ParameterDataset fixture parametrized over NetCDF and JSON backends."""
    if request.param == "netcdf":
        return NetCDFParameterDataset(default_ds.copy(deep=True))
    return json_dataset.copy()


@pytest.fixture(params=["netcdf", "json"])
def default_param_file(request, default_netcdf_file, json_param_file):
    """default parameter file, parametrized over backends."""
    ds = default_netcdf_file if request.param == "netcdf" else json_param_file
    return ds
