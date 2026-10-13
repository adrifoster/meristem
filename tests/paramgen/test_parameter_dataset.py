"""Tests for ParameterDataset and ParameterVariable implementations."""

from __future__ import annotations
import json

import numpy as np
import pytest

from param_ens_gen.parameter_dataset import (
    ParameterDataset,
    NetCDFParameterDataset,
    FATESJSONParameterDataset,
    NetCDFParameterVariable,
    FATESJSONParameterVariable,
)

# ===========================================================================
# ParameterDataset.from_path
# ===========================================================================


def test_from_path_loads_netcdf(default_netcdf_file):
    """from_path returns a NetCDFParameterDataset for .nc files."""
    ds = ParameterDataset.from_path(default_netcdf_file)
    assert isinstance(ds, NetCDFParameterDataset)


def test_from_path_loads_json(json_param_file):
    """from_path returns a JSONParameterDataset for .json files."""
    ds = ParameterDataset.from_path(json_param_file)
    assert isinstance(ds, FATESJSONParameterDataset)


def test_from_path_unsupported_extension_raises(tmp_path):
    """from_path raises ValueError for unsupported file extensions."""
    bad_file = tmp_path / "params.txt"
    bad_file.write_text("not a param file")
    with pytest.raises(ValueError, match="Unsupported parameter file extension"):
        ParameterDataset.from_path(bad_file)


# ===========================================================================
# NetCDFParameterDataset
# ===========================================================================


def test_netcdf_sizes(netcdf_dataset):
    """NetCDFParameterDataset.sizes returns correct dimension sizes."""
    assert netcdf_dataset.sizes["fates_pft"] == 3
    assert netcdf_dataset.sizes["fates_leafage_class"] == 2


def test_netcdf_dims(netcdf_dataset):
    """NetCDFParameterDataset.dims returns all dimension names."""
    assert "fates_pft" in netcdf_dataset.dims
    assert "fates_leafage_class" in netcdf_dataset.dims


def test_netcdf_data_vars(netcdf_dataset):
    """NetCDFParameterDataset.data_vars returns all variable names."""
    assert "fates_leaf_slatop" in netcdf_dataset.data_vars
    assert "fates_canopy_closure_thresh" in netcdf_dataset.data_vars


def test_netcdf_contains(netcdf_dataset):
    """NetCDFParameterDataset.__contains__ returns True for existing variables."""
    assert "fates_leaf_slatop" in netcdf_dataset
    assert "nonexistent" not in netcdf_dataset


def test_netcdf_getitem_returns_variable(netcdf_dataset):
    """NetCDFParameterDataset.__getitem__ returns a ParameterVariable."""
    var = netcdf_dataset["fates_leaf_slatop"]
    assert isinstance(var, NetCDFParameterVariable)


def test_netcdf_getitem_missing_raises(netcdf_dataset):
    """NetCDFParameterDataset.__getitem__ raises KeyError for missing variables."""
    with pytest.raises(KeyError, match="nonexistent"):
        print(netcdf_dataset["nonexistent"])


def test_netcdf_variable_values(netcdf_dataset):
    """NetCDFParameterVariable.values returns correct array."""
    np.testing.assert_allclose(
        netcdf_dataset["fates_leaf_slatop"].values, [0.010, 0.020, 0.030]
    )


def test_netcdf_variable_dims(netcdf_dataset):
    """NetCDFParameterVariable.dims returns correct dimension names."""
    assert netcdf_dataset["fates_leaf_slatop"].dims == ["fates_pft"]


def test_netcdf_variable_isel(netcdf_dataset):
    """NetCDFParameterVariable.isel returns correct slice."""
    result = netcdf_dataset["fates_leaf_vcmax25top"].isel({"fates_leafage_class": 0})
    np.testing.assert_allclose(result, [50.0, 60.0, 70.0])


def test_netcdf_variable_values_setter(netcdf_dataset):
    """NetCDFParameterVariable.values setter updates the array."""
    new_values = np.array([1.0, 2.0, 3.0])
    netcdf_dataset["fates_leaf_slatop"].values = new_values
    np.testing.assert_allclose(netcdf_dataset["fates_leaf_slatop"].values, new_values)


def test_netcdf_copy_is_independent(netcdf_dataset):
    """NetCDFParameterDataset.copy() produces an independent working copy."""
    copy = netcdf_dataset.copy()
    copy["fates_leaf_slatop"].values = np.array([99.0, 99.0, 99.0])
    np.testing.assert_allclose(
        netcdf_dataset["fates_leaf_slatop"].values, [0.010, 0.020, 0.030]
    )


def test_netcdf_save_and_reload(netcdf_dataset, tmp_path):
    """NetCDFParameterDataset.save() writes a file that can be reloaded."""
    path = tmp_path / "output.nc"
    netcdf_dataset.save(path)
    assert path.exists()
    reloaded = NetCDFParameterDataset.load(path)
    np.testing.assert_allclose(
        reloaded["fates_leaf_slatop"].values, [0.010, 0.020, 0.030]
    )


def test_netcdf_scalar_variable(netcdf_dataset):
    """NetCDFParameterVariable handles scalar parameters correctly."""
    result = netcdf_dataset["fates_canopy_closure_thresh"].values
    assert float(result) == pytest.approx(0.5)


# ===========================================================================
# JSONParameterDataset
# ===========================================================================


def test_json_sizes(json_dataset):
    """FATESJSONParameterDataset.sizes returns correct dimension sizes."""
    assert json_dataset.sizes["fates_pft"] == 3
    assert json_dataset.sizes["fates_leafage_class"] == 2


def test_json_dims(json_dataset):
    """FATESJSONParameterDataset.dims returns all dimension names."""
    assert "fates_pft" in json_dataset.dims
    assert "fates_leafage_class" in json_dataset.dims


def test_json_data_vars(json_dataset):
    """FATESJSONParameterDataset.data_vars returns all variable names."""
    assert "fates_leaf_slatop" in json_dataset.data_vars
    assert "fates_canopy_closure_thresh" in json_dataset.data_vars


def test_json_contains(json_dataset):
    """JSONParameterDataset.__contains__ returns True for existing variables."""
    assert "fates_leaf_slatop" in json_dataset
    assert "nonexistent" not in json_dataset


def test_json_getitem_returns_variable(json_dataset):
    """FATESFATESJSONParameterDataset.__getitem__ returns a ParameterVariable."""
    var = json_dataset["fates_leaf_slatop"]
    assert isinstance(var, FATESJSONParameterVariable)


def test_json_getitem_missing_raises(json_dataset):
    """FATESJSONParameterDataset.__getitem__ raises KeyError for missing variables."""
    with pytest.raises(KeyError, match="nonexistent"):
        print(json_dataset["nonexistent"])


def test_json_variable_values(json_dataset):
    """FATESJSONParameterVariable.values returns correct array."""
    np.testing.assert_allclose(
        json_dataset["fates_leaf_slatop"].values, [0.010, 0.020, 0.030]
    )


def test_json_variable_dims(json_dataset):
    """FATESJSONParameterVariable.dims returns correct dimension names."""
    assert json_dataset["fates_leaf_slatop"].dims == ["fates_pft"]


def test_json_variable_isel(json_dataset):
    """JSONParameterVariable.isel returns correct slice."""
    result = json_dataset["fates_leaf_vcmax25top"].isel({"fates_leafage_class": 0})
    np.testing.assert_allclose(result, [50.0, 60.0, 70.0])


def test_json_variable_values_setter(json_dataset):
    """JSONParameterVariable.values setter updates the array."""
    new_values = np.array([1.0, 2.0, 3.0])
    json_dataset["fates_leaf_slatop"].values = new_values
    np.testing.assert_allclose(json_dataset["fates_leaf_slatop"].values, new_values)


def test_json_copy_is_independent(json_dataset):
    """FATESJSONParameterDataset.copy() produces an independent working copy."""
    copy = json_dataset.copy()
    copy["fates_leaf_slatop"].values = np.array([99.0, 99.0, 99.0])
    np.testing.assert_allclose(
        json_dataset["fates_leaf_slatop"].values, [0.010, 0.020, 0.030]
    )


def test_json_save_and_reload(json_dataset, tmp_path):
    """FATESJSONParameterDataset.save() writes a file that can be reloaded."""
    path = tmp_path / "output.json"
    json_dataset.save(path)
    assert path.exists()
    reloaded = FATESJSONParameterDataset.load(path)
    np.testing.assert_allclose(
        reloaded["fates_leaf_slatop"].values, [0.010, 0.020, 0.030]
    )


def test_json_save_preserves_modified_values(json_dataset, tmp_path):
    """FATESJSONParameterDataset.save() writes modified values correctly."""
    json_dataset["fates_leaf_slatop"].values = np.array([1.0, 2.0, 3.0])
    path = tmp_path / "output.json"
    json_dataset.save(path)
    reloaded = FATESJSONParameterDataset.load(path)
    np.testing.assert_allclose(reloaded["fates_leaf_slatop"].values, [1.0, 2.0, 3.0])


def test_json_scalar_variable(json_dataset):
    """FATESJSONParameterDataset handles scalar parameters correctly."""
    result = json_dataset["fates_canopy_closure_thresh"].values
    assert float(result) == pytest.approx(0.5)


def test_json_multidim_variable_shape(json_dataset):
    """FATESJSONParameterDataset handles multi-dimensional parameters correctly."""
    result = json_dataset["fates_leaf_vcmax25top"].values
    assert result.shape == (2, 3)
    np.testing.assert_allclose(result[0, :], [50.0, 60.0, 70.0])
    np.testing.assert_allclose(result[1, :], [40.0, 50.0, 60.0])


def test_json_close_is_noop(json_dataset):
    """FATESJSONParameterDataset.close() does not raise."""
    json_dataset.close()


def test_json_scalar_dim_loads_correctly(json_dataset):
    """JSONParameterVariable handles FATES-style ['scalar'] dim correctly."""
    result = json_dataset["fates_canopy_closure_thresh"].values
    assert result.ndim == 0
    assert float(result) == pytest.approx(0.5)


def test_json_scalar_dim_roundtrips(json_dataset, tmp_path):
    """JSONParameterVariable with ['scalar'] dim writes and reloads correctly."""
    json_dataset["fates_canopy_closure_thresh"].values = np.float64(0.9)
    path = tmp_path / "output.json"
    json_dataset.save(path)
    reloaded = FATESJSONParameterDataset.load(path)
    assert float(reloaded["fates_canopy_closure_thresh"].values) == pytest.approx(0.9)
    # verify the format is preserved as ['scalar'] not []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["parameters"]["fates_canopy_closure_thresh"]["dims"] == ["scalar"]
    assert isinstance(data["parameters"]["fates_canopy_closure_thresh"]["data"], list)


def test_json_empty_dims_raises(tmp_path):
    """FATESJSONParameterVariable raises for empty dims list."""
    data = {
        "attributes": {},
        "dimensions": {},
        "parameters": {
            "my_scalar": {
                "dtype": "float",
                "dims": [],
                "long_name": "test",
                "units": "1",
                "data": 0.5,
            }
        },
    }
    path = tmp_path / "test.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    ds = FATESJSONParameterDataset.load(path)
    with pytest.raises(ValueError, match="empty dims"):
        ds["my_scalar"].values = np.float64(0.9)


def test_json_save_empty_dims_raises(tmp_path):
    """FATESJSONParameterDataset.save raises for empty dims list."""
    data = {
        "attributes": {},
        "dimensions": {},
        "parameters": {
            "my_scalar": {
                "dtype": "float",
                "dims": [],
                "long_name": "test",
                "units": "1",
                "data": 0.5,
            }
        },
    }
    path = tmp_path / "test.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    ds = FATESJSONParameterDataset.load(path)
    # access the variable to put it in the cache
    _ = ds["my_scalar"]
    with pytest.raises(ValueError, match="empty dims"):
        ds.save(path)
