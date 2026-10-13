"""
Parameter dataset - classes for dealing with the parameter dataset
This supports format-agnostic parameter datasets.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import copy
import json

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# ParameterVariable
# ---------------------------------------------------------------------------


class ParameterVariable(ABC):
    """Abstract interface for a single variable within a ParameterDataset.

    Attributes
    ----------
    values : np.ndarray
        The raw numpy array for this variable. Settable in-place.
    dims : list[str]
        The dimension names for this variable, in order.
    """

    @property
    @abstractmethod
    def values(self) -> np.ndarray:
        """Return the raw numpy array for this variable."""

    @values.setter
    @abstractmethod
    def values(self, arr: np.ndarray) -> None:
        """Set the raw numpy array for this variable in-place."""

    @property
    @abstractmethod
    def dims(self) -> list[str]:
        """Return the dimension names for this variable."""

    @abstractmethod
    def isel(self, indexers: dict[str, int]) -> np.ndarray:
        """Return a numpy array sliced at the given dimension indices.

        Args:
            indexers (dict[str, int]): Mapping of dimension name to index.

        Returns:
            np.ndarray: Sliced array.
        """


# ---------------------------------------------------------------------------
# ParameterDataset
# ---------------------------------------------------------------------------


class ParameterDataset(ABC):
    """Abstract interface for a parameter dataset.

    Wraps either a NetCDF or JSON parameter file and exposes a common
    set of operations used throughout param_ens_gen.
    """

    @property
    @abstractmethod
    def file_extension(self) -> np.ndarray:
        """Return the correct file extension"""

    @property
    @abstractmethod
    def sizes(self) -> dict[str, int]:
        """Return a mapping of dimension name to size."""

    @property
    @abstractmethod
    def dims(self) -> list[str]:
        """Return the list of dimension names."""

    @property
    @abstractmethod
    def data_vars(self) -> list[str]:
        """Return the list of variable names."""

    @abstractmethod
    def __contains__(self, var: str) -> bool:
        """Return True if the variable exists in this dataset."""

    @abstractmethod
    def __getitem__(self, var: str) -> ParameterVariable:
        """Return the ParameterVariable for the given variable name."""

    @abstractmethod
    def copy(self) -> ParameterDataset:
        """Return a working copy of this dataset.

        The copy is independent — modifications to it do not affect the
        original. Used to create per-ensemble-member working datasets.
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """Write this dataset to a file.

        Args:
            path (Path): Output file path. The format is determined by
                the concrete implementation.
        """

    @abstractmethod
    def close(self) -> None:
        """Release any file handles held by this dataset."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> ParameterDataset:
        """Load a dataset from a file.

        Args:
            path (Path): Path to the parameter file.

        Returns:
            ParameterDataset: Loaded dataset.
        """

    @classmethod
    def from_path(cls, path: Path) -> ParameterDataset:
        """Load a dataset from a file, detecting format from the extension.

        Supported extensions: .nc (NetCDF), .json (JSON).

        Args:
            path (Path): Path to the parameter file.

        Raises:
            ValueError: If the file extension is not supported.

        Returns:
            ParameterDataset: Loaded dataset.
        """
        suffix = path.suffix.lower()
        if suffix == ".nc":
            return NetCDFParameterDataset.load(path)
        if suffix == ".json":
            return FATESJSONParameterDataset.load(path)
        raise ValueError(
            f"Unsupported parameter file extension '{suffix}'. "
            "Supported formats: .nc (NetCDF), .json (JSON)."
        )


# ---------------------------------------------------------------------------
# NetCDF implementation
# ---------------------------------------------------------------------------


class NetCDFParameterVariable(ParameterVariable):
    """ParameterVariable backed by an xarray DataArray."""

    def __init__(self, da: xr.DataArray) -> None:
        self._da = da

    @property
    def values(self) -> np.ndarray:
        return self._da.values

    @values.setter
    def values(self, arr: np.ndarray) -> None:
        self._da.values = arr

    @property
    def dims(self) -> list[str]:
        return list(self._da.dims)

    def isel(self, indexers: dict[str, int]) -> np.ndarray:
        return self._da.isel(indexers).values


class NetCDFParameterDataset(ParameterDataset):
    """ParameterDataset backed by an xarray Dataset."""

    def __init__(self, ds: xr.Dataset) -> None:
        self._ds = ds

    @property
    def file_extension(self) -> str:
        return ".nc"

    @property
    def sizes(self) -> dict[str, int]:
        return dict(self._ds.sizes)

    @property
    def dims(self) -> list[str]:
        return list(self._ds.dims)

    @property
    def data_vars(self) -> list[str]:
        return list(self._ds.data_vars)

    def __contains__(self, var: str) -> bool:
        return var in self._ds

    def __getitem__(self, var: str) -> NetCDFParameterVariable:
        if var not in self._ds:
            raise KeyError(
                f"Variable '{var}' not found in dataset. "
                f"Available variables: {sorted(self._ds.data_vars)}"
            )
        return NetCDFParameterVariable(self._ds[var])

    def copy(self) -> NetCDFParameterDataset:
        return NetCDFParameterDataset(self._ds.copy(deep=False))

    def save(self, path: Path) -> None:
        self._ds.to_netcdf(path, engine="netcdf4")

    def close(self) -> None:
        self._ds.close()

    @classmethod
    def load(cls, path: Path) -> NetCDFParameterDataset:
        return cls(xr.open_dataset(path))


# ---------------------------------------------------------------------------
# JSON implementation
# ---------------------------------------------------------------------------


class FATESJSONParameterVariable(ParameterVariable):
    """ParameterVariable backed by a dict entry in a JSON parameter file.

    The variable's data is stored as a numpy array and written back to
    the parent dataset's dict when set.
    """

    def __init__(
        self,
        name: str,
        entry: dict,
        dim_sizes: dict[str, int],
    ) -> None:
        self._name = name
        self._entry = entry
        self._dim_sizes = dim_sizes
        self._values: np.ndarray = self._load_values()

    def _load_values(self) -> np.ndarray:
        data = self._entry["data"]
        dims = self._entry.get("dims", [])
        if dims == ["scalar"]:
            # single-element list — extract scalar value
            val = data[0] if isinstance(data, list) else data
            return np.asarray(val if val is not None else np.nan)
        arr = np.array(data)
        if dims:
            expected_shape = tuple(self._dim_sizes[d] for d in dims)
            arr = arr.reshape(expected_shape)
        return arr

    @property
    def values(self) -> np.ndarray:
        return self._values

    @values.setter
    def values(self, arr: np.ndarray) -> None:
        self._values = np.asarray(arr)
        dims = self._entry.get("dims", [])
        if dims == ["scalar"]:
            self._entry["data"] = [float(self._values)]
        elif dims:
            self._entry["data"] = self._values.tolist()
        else:
            raise ValueError(
                "Unexpected empty dims for FATES JSON parameter. "
                "Scalar parameters should use dims=['scalar'] in FATES JSON format."
            )

    @property
    def dims(self) -> list[str]:
        dims = self._entry.get("dims", [])
        return [] if dims == ["scalar"] else dims

    def isel(self, indexers: dict[str, int]) -> np.ndarray:
        dims = self.dims
        arr = self._values
        # apply each indexer in order, adjusting axis as earlier axes collapse
        offset = 0
        for dim, idx in indexers.items():
            axis = dims.index(dim) - offset
            arr = np.take(arr, idx, axis=axis)
            offset += 1
        return arr


class FATESJSONParameterDataset(ParameterDataset):
    """ParameterDataset backed by a FATES JSON parameter file.

    The JSON format has three top-level keys:
        - attributes: global metadata dict
        - dimensions: mapping of dimension name to size
        - parameters: mapping of parameter name to parameter dict

    Each parameter dict has:
        - dtype: string type hint ('float', 'integer', 'string')
        - dims: list of dimension names (empty list for scalars)
        - long_name: human-readable description
        - units: units string
        - data: the parameter values (nested list for multi-dim, scalar otherwise)
    """

    def __init__(self, data: dict) -> None:
        self._data = data
        self._dim_sizes: dict[str, int] = dict(data.get("dimensions", {}))
        self._params: dict[str, dict] = data.get("parameters", {})
        # cache of ParameterVariable objects keyed by name
        self._cache: dict[str, FATESJSONParameterVariable] = {}

    @property
    def file_extension(self) -> str:
        return ".json"

    @property
    def sizes(self) -> dict[str, int]:
        return dict(self._dim_sizes)

    @property
    def dims(self) -> list[str]:
        return list(self._dim_sizes.keys())

    @property
    def data_vars(self) -> list[str]:
        return list(self._params.keys())

    def __contains__(self, var: str) -> bool:
        return var in self._params

    def __getitem__(self, var: str) -> FATESJSONParameterVariable:
        if var not in self._params:
            raise KeyError(
                f"Variable '{var}' not found in dataset. "
                f"Available variables: {sorted(self._params.keys())}"
            )
        if var not in self._cache:
            self._cache[var] = FATESJSONParameterVariable(
                name=var,
                entry=self._params[var],
                dim_sizes=self._dim_sizes,
            )
        return self._cache[var]

    def copy(self) -> FATESJSONParameterDataset:
        """Return a deep copy of this dataset for use as a working copy."""
        new_data = copy.deepcopy(self._data)
        return FATESJSONParameterDataset(new_data)

    def save(self, path: Path) -> None:
        for name, var in self._cache.items():
            dims = self._params[name].get("dims", [])
            if dims == ["scalar"]:
                self._params[name]["data"] = [float(var.values)]
            elif var.values.ndim > 0:
                self._params[name]["data"] = var.values.tolist()
            else:
                raise ValueError(
                    f"Parameter '{name}' has unexpected empty dims in FATES JSON format. "
                    "Scalar parameters should use dims=['scalar']."
                )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    def close(self) -> None:
        """No-op — JSON datasets hold no file handles."""

    @classmethod
    def load(cls, path: Path) -> FATESJSONParameterDataset:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)
