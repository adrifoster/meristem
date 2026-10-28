"""
Parameter - classes for parameter logic
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import NamedTuple, Optional
import copy
import pandas as pd
import numpy as np

from .param_spec import ParamSpec
from .distribution_stat import PFTStat
from .sampler import Sampler, SampleContext, PosteriorSampler, UniformSampler
from .parameter_dataset import ParameterDataset, ParameterVariable


class DimIndex(NamedTuple):
    """A pinned position in a single dimension.

    Used on expanded Parameter objects to record which dimension and index

    Attributes
    ----------
    dim : str
        The dimension name, e.g. 'fates_pft'.
    index : int
        The 0-based index along that dimension.
    """

    dim: str
    index: int


class Parameter(ABC):
    """Abstract base for parameter logic.

    Parameters
    ----------
    spec : ParamSpec
        All metadata for this parameter
    sampler : Sampler
        Class for parameter sampling
    active_index: DimIndex | None
        Set by the expansion step on expanded Parameters. Records which dimension
        and index this Parameter is responsible for. None on unexpanded Parameters.
    """

    _registry: dict[str, type[Parameter]] = {}

    def __init_subclass__(cls, param_type: str, **kwargs):
        super().__init_subclass__(**kwargs)
        Parameter._registry[param_type] = cls

    def __init__(
        self,
        row: pd.Series,
        default_ds: ParameterDataset,
        pft_sheet: pd.DataFrame | None = None,
        posterior_config: dict | None = None,
    ):
        self.spec = ParamSpec.from_row(row)
        self.sampler = Sampler.from_row_and_sheet(row, pft_sheet, posterior_config)
        self.active_index: Optional[DimIndex] = None

        # store only sizes so we don't hold a reference to the full dataset.
        # this is used by the n_indices property without ordering constraints.
        self._dim_sizes: dict[str, int] = dict(default_ds.sizes)

        self._validate_specs()
        self._validate_params(default_ds)
        self._validate_posterior()
        self._validate_pft_dim()

    @classmethod
    def from_row(
        cls,
        row: pd.Series,
        default_ds: ParameterDataset,
        pft_sheet: pd.DataFrame | None = None,
        posterior_config: dict | None = None,
    ) -> Parameter:
        """Construct the correct Parameter subclass from a spreadsheet row.

        Args:
            row (pd.Series): A row from the 'main' sheet DataFrame.
            default_ds (ParameterDataset): The default parameter dataset.
            pft_sheet (pd.DataFrame | None, optional): Optional per-PFT bounds sheet.
                Defaults to None.
            posterior_config (dict | None, optional): Optional posterior sampling
                configuration. Defaults to None.

        Raises:
            ValueError: If param_type is not registered.

        Returns:
            Parameter: An instance of the appropriate Parameter subclass.
        """
        param_type = str(row.get("param_type", "")).strip()
        subclass = cls._registry.get(param_type)
        if subclass is None:
            raise ValueError(
                f"Unknown param_type '{param_type}'. "
                f"Valid types: {sorted(cls._registry)}"
            )
        return subclass(row, default_ds, pft_sheet, posterior_config)

    @property
    def free_dims(self) -> str | list[str] | None:
        """The free dimensions for this parameter, or None for scalars.

        For sliced parameters the slice dimension is excluded, leaving at most
        one free dimension. For scalar parameters (no dims) this returns None

        Returns:
            str | list[str] | None: Dimension name(s), or None when this parameter is scalar.

        """
        if self.spec.slice_dim is None:
            free = self.spec.dims
        else:
            free = [d for d in self.spec.dims if d != self.spec.slice_dim]
        return free if free else None

    @property
    def n_indices(self) -> list[int]:
        """Number of positions along the free dimension (1 for scalars)."""
        return (
            [self._dim_sizes.get(dim, 1) for dim in self.free_dims]
            if self.free_dims
            else [1]
        )

    @property
    def pft_axis(self) -> int | None:
        """Axis of the PFT dimension within free_dims, or None if there isn't one.

        Uses spec.pft_dim if set. Otherwise uses the one free dimension whose name
        contains 'pft'; if none or several do, returns None.
        """
        dims = self.free_dims or []
        if self.spec.pft_dim is not None:
            return dims.index(self.spec.pft_dim)
        candidates = [i for i, dim in enumerate(dims) if "pft" in dim.lower()]
        return candidates[0] if len(candidates) == 1 else None

    def _validate_specs(self) -> None:
        """Validate type-specific required fields on self.spec.

        Called from __init__ after self.spec is set. The base implementation
        is a no-op; subclasses override to assert the fields they require.
        This is intentionally not abstract — types with no extra required
        fields (e.g. DefaultParameter) do not need to override.
        """

    def _validate_params(self, default_ds: ParameterDataset) -> None:
        """Check that all variables this parameter touches exist in default_ds
        with the correct dimensions.

        For sliced parameters, spec.dims includes the slice dimension because
        it reflects the full variable shape in the dataset. The slice is an
        access pattern, not a dataset shape change.

        Args:
            default_ds (ParameterDataset): The default parameter dataset.

        Raises:
            ValueError: If a variable is missing or has unexpected dimensions.
        """
        for var in self._variables_to_validate():
            if var not in default_ds:
                raise ValueError(
                    f"Parameter '{self.spec.name}': variable '{var}' not found "
                    f"in default dataset. Available variables: "
                    f"{sorted(default_ds.data_vars)}"
                )
            actual_dims = list(default_ds[var].dims)
            if actual_dims != self.spec.dims:
                raise ValueError(
                    f"Parameter '{self.spec.name}': variable '{var}' has dims "
                    f"{actual_dims} in default dataset but spec.dims is "
                    f"{self.spec.dims}. Dimensions must match exactly."
                )

    def _validate_pft_dim(self) -> None:
        """Check that the PFT dimension can be identified where it's needed.

        Raises:
            ValueError: If spec.pft_dim is not a free dimension, or if a
                non-scalar parameter has per-PFT bounds but no single PFT
                dimension can be identified.
        """
        dims = self.free_dims or []
        if self.spec.pft_dim is not None and self.spec.pft_dim not in dims:
            raise ValueError(
                f"Parameter '{self.spec.name}': pft_dim '{self.spec.pft_dim}' is not "
                f"one of its free dimensions {dims}."
            )
        has_pft_bounds = isinstance(self.sampler, UniformSampler) and (
            isinstance(self.sampler.min_stat, PFTStat)
            or isinstance(self.sampler.max_stat, PFTStat)
        )
        if has_pft_bounds and dims and self.pft_axis is None:
            raise ValueError(
                f"Parameter '{self.spec.name}': has per-PFT bounds but the PFT "
                f"dimension can't be identified from its dimensions {dims}. "
                "Set 'pft_dim' in main.csv to name it."
            )

    def _validate_posterior(self):
        if not isinstance(self.sampler, PosteriorSampler):
            return
        for source in self.sampler.sources:
            if not np.array_equal(source.parameters, self._variables_to_validate()):
                raise ValueError(
                    f"Parameter '{self.spec.name}': mismatch in `base_params` "
                    "and `parameters` on PosteriorSource "
                    f"PosteriorSource parameters: {source.parameters} "
                    f"`base_params`: {self._variables_to_validate()}"
                )
            if not source.is_broadcast:
                if len(self.n_indices) > 1:
                    raise ValueError(
                        f"Parameter '{self.spec.name}': has dimensions {self.n_indices} "
                        "And has a PosteriorSampler not using broadcast mode. "
                        "Currently this is not supported. "
                        "PosteriorSampler parameters must have only one dimension or "
                        "have array_indices = 'all'"
                    )
                # check that the input array_indices will fit in this dim
                if np.any(np.array(source.array_indices) >= self.n_indices[0]):
                    raise ValueError(
                        f"Parameter '{self.spec.name}': dimension mismatch between "
                        "default dataset and posterior sampler array_indices. "
                        f"array_indices: {source.array_indices} "
                        f"parameter indices: {self.n_indices[0]}."
                    )

    @abstractmethod
    def _variables_to_validate(self) -> list[str]:
        """Return the variable names in the dataset that this parameter touches.

        Each subclass must implement this explicitly so that _validate_dataset
        checks exactly the right variables. Abstract rather than a shared
        default to prevent new subclasses from silently inheriting incorrect
        behaviour.

        Returns:
            List of variable name strings.
        """

    def sample(
        self,
        normalized_value: float,
        default_ds: ParameterDataset,
    ) -> float | np.ndarray:
        """Sample a parameter given an input normalized value

        Builds a SampleContext from the dataset and fixed_indices, then
        delegates to self.sampler. Subclasses override _build_context if
        they need different context behaviour (e.g. JointParameter passes
        mask=None).

        Args:
            normalized_value (float): normalized value [0-1] used to sample
            default_ds (ParameterDataset): default parameter dataset. used for validating
                dimensions and indices
            fixed_indices (dict[str, list[int]]): 0-based indices to hold at default.

        Returns:
            float: Sampled parameter value.
        """
        context = self._build_context(self._full_default(default_ds))
        return self.sampler.sample(normalized_value, context)

    def normalize(
        self,
        value: float | np.ndarray,
        default_ds: ParameterDataset,
        clip_to_bounds: bool = False,
    ) -> float | np.ndarray:
        """Normalize a parameter to 0-1 given an input value

        Builds a SampleContext from the dataset and fixed_indices, then
        delegates to self.sampler. Subclasses override _build_context if
        they need different context behaviour (e.g. JointParameter passes
        mask=None).

        Args:
            value (float): value to normalize
            default_ds (ParameterDataset): default parameter dataset. used for validating
                dimensions and indices
            fixed_indices (dict[str, list[int]]): 0-based indices to hold at default.
            clip_to_bounds (bool, optional): optionally clip to 0-1 if value is outside
            min/max range. Defaults to False.

        Returns:
            float: normalized [0-1] parameter value.
        """
        context = self._build_context(self._full_default(default_ds))
        return self.sampler.normalize(value, context, clip_to_bounds=clip_to_bounds)

    def _build_context(
        self,
        default_value: float | np.ndarray | list[np.ndarray],
    ) -> SampleContext:
        """Build a SampleContext for this parameter.

        For an expanded parameter, default_value is still the full default array, so
        the sampler can resolve bounds over every index (e.g. per-PFT bounds) before
        selecting this parameter's index along index_axis.

        Args:
            default_value: Full default value(s) for this parameter.
        Returns:
            SampleContext populated for this parameter.
        """
        dims = self.free_dims or []
        index_axis = None
        if self.active_index is not None:
            if self.active_index.dim not in dims:
                raise ValueError(
                    f"Parameter '{self.spec.name}': expanded dimension "
                    f"'{self.active_index.dim}' is not one of its dimensions {dims}."
                )
            index_axis = dims.index(self.active_index.dim)
        return SampleContext(
            default_value=default_value,
            array_index=(
                self.active_index.index if self.active_index is not None else None
            ),
            index_axis=index_axis,
            n_indices=self.n_indices,
            pft_axis=self.pft_axis,
        )

    def get_default(
        self,
        default_ds: ParameterDataset,
    ) -> float | np.ndarray | list[np.ndarray]:
        """Extract the relevant default value(s) from a parameter dataset.

        For an expanded parameter, only the value(s) at active_index.

        Args:
            default_ds (ParameterDataset): The default parameter dataset

        Returns:
            float | np.ndarray | list[np.ndarray]: default value
        """
        full = self._full_default(default_ds)
        if self.active_index is None:
            return full
        dims = self.free_dims or []
        if isinstance(full, list):
            return [_take(arr, dims, self.active_index) for arr in full]
        return _take(full, dims, self.active_index)

    @abstractmethod
    def _full_default(
        self,
        default_ds: ParameterDataset,
    ) -> float | np.ndarray | list[np.ndarray]:
        """Extract the full default value(s), ignoring any active_index.

        The returned array's dimensions are free_dims.

        Args:
            default_ds (ParameterDataset): The default parameter dataset

        Returns:
            float | np.ndarray | list[np.ndarray]: default value
        """

    def set_value(
        self,
        ds: ParameterDataset,
        default_ds: ParameterDataset,
        value: float | np.ndarray | list[np.ndarray],
        fixed_indices: dict[str, list[int]] | None = None,
    ) -> None:
        """Write a value into the working dataset.

        Args:
            ds (ParameterDataset): Working copy of the parameter dataset. Modified in place.
            default_ds (ParameterDataset): Unchanging default dataset. Used to restore fixed
                positions.
            value (float | np.ndarray | list[np.ndarray]): Value to write
            fixed_indices (dict[str, list[int]] | None): Run-level mapping of dimension to
                0-based indices to hold at default. None means no indices are fixed
        """
        value = self._apply_precision(value)
        if self.active_index is not None:
            self._write_at_index(ds, self.active_index, value, fixed_indices or {})
        else:
            self._write_full(ds, default_ds, value, fixed_indices or {})

    @abstractmethod
    def _write_at_index(
        self,
        ds: ParameterDataset,
        index: DimIndex,
        value: float | np.ndarray | list[np.ndarray],
        fixed_indices: dict[str, list[int]],
    ) -> None:
        """Write a value at one position along the expanded dimension.

        Called when this parameter is expanded (``active_index`` is set). For a
        1-D variable that position is a single element; for a multi-dimensional
        variable it is the sub-array at ``index`` along ``index.dim``. ``value``
        must be a scalar (broadcast across the sub-array) or match the sub-array's
        shape; subclasses should enforce this via ``_fit_to_shape``.

        Args:
            ds (ParameterDataset): Working copy of the parameter dataset. Modified in place.
            index (DimIndex): The dimension and position to write.
            value (float | np.ndarray | list[np.ndarray]): Value to write.
            fixed_indices (dict[str, list[int]]): Dimension-to-indices mapping for
                positions on the variable's other dimensions to leave unchanged.
        """

    @abstractmethod
    def _write_full(
        self,
        ds: ParameterDataset,
        default_ds: ParameterDataset,
        value: float | np.ndarray | list[np.ndarray],
        fixed_indices: dict[str, list[int]],
    ) -> None:
        """Broadcast a value across all non-fixed positions in the dataset.

        Called when this parameter is not expanded (active_index is None).
        value is either a scalar (broadcast to all free positions) or a
        full-dimension array (the sampler always sees the full dimension).
        fixed_indices is applied as a post-processing mask: those positions
        are restored from default_ds after writing.

        Args:
            ds (ParameterDataset): Working copy of the parameter dataset. Modified in place.
            default_ds (ParameterDataset): Unchanging default dataset. Used to restore
                fixed positions.
            value (float | np.ndarray | list[np.ndarray]): Scalar or full-dimension
                array value to write.
            fixed_indices (dict[str, list[int]]): Dimension-to-indices mapping for
                positions to hold at default. Empty dict means no positions are fixed.
        """

    def for_index(self, dim: str, index: int) -> Parameter:
        """Return a copy of this Parameter bound to a specific dim/index.

        sampler and _dim_sizes are intentionally shared.
        active_index and spec.name should be the only fields that differ between
        expanded copies.
        """
        clone = copy.copy(self)
        clone.active_index = DimIndex(dim=dim, index=index)
        clone.spec = copy.copy(self.spec)
        clone.spec.name = f"{self.spec.name}_{index}"
        return clone

    def _apply_precision(
        self, value: float | np.ndarray | list[np.ndarray]
    ) -> float | np.ndarray | list[np.ndarray]:
        """Round value to the precision specified in spec.precision.

        Args:
            value: Sampled value to round.

        Returns:
            Rounded value, or value unchanged if precision is None.
        """
        if self.spec.precision is None:
            return value
        if isinstance(value, list):
            return [np.round(v, self.spec.precision) for v in value]
        return np.round(value, self.spec.precision)


# ----------------------------------------------------------------------------------------
# Concrete Parameter classes
# ----------------------------------------------------------------------------------------


class DefaultParameter(Parameter, param_type="default"):
    """Standard parameter: written directly to ds[name]."""

    def _variables_to_validate(self) -> list[str]:
        return [self.spec.name]

    def _full_default(
        self, default_ds: ParameterDataset
    ) -> float | np.ndarray | list[np.ndarray]:
        return default_ds[self.spec.original_name].values

    def _write_at_index(
        self,
        ds: ParameterDataset,
        index: DimIndex,
        value: float | np.ndarray | list[np.ndarray],
        fixed_indices: dict[str, list[int]],
    ):
        _write_at(
            ds[self.spec.original_name],
            index,
            value,
            fixed_indices,
            self.spec.original_name,
        )

    def _write_full(
        self,
        ds: ParameterDataset,
        default_ds: ParameterDataset,
        value: float | np.ndarray | list[np.ndarray],
        fixed_indices: dict[str, list[int]],
    ):
        var = ds[self.spec.original_name]
        var.values = _broadcast_to_array(
            var.values.copy(), value, var.dims, fixed_indices, self.spec.original_name
        )


class SlicedParameter(Parameter, param_type="sliced"):
    """Parameter that targets one slice of a dimension."""

    def _validate_specs(self) -> None:
        """Require slice_dim, slice_index, and base_params to all be set.
        Also len(base_params) == 1.
        """
        missing = []
        if self.spec.slice_dim is None:
            missing.append("slice_dim")
        if self.spec.slice_index is None:
            missing.append("slice_index")
        if not self.spec.base_params:
            missing.append("base_params")
        if missing:
            raise ValueError(
                f"Parameter '{self.spec.name}' has param_type 'sliced' but the "
                f"following required fields are not set: {missing}."
            )
        if len(self.spec.base_params) > 1:
            raise ValueError(
                f"Too many base_params for parameter {self.spec.name}. "
                "Shouldn't have more than one base_param for SlicedParameters. "
                f"Got {len(self.spec.base_params)}: {self.spec.base_params}"
            )

    def _variables_to_validate(self) -> list[str]:
        return self.spec.base_params

    def _full_default(self, default_ds: ParameterDataset) -> np.ndarray:
        return np.asarray(
            default_ds[self.spec.base_params[0]].isel(
                {self.spec.slice_dim: self.spec.slice_index}
            )
        )

    def _write_at_index(
        self,
        ds: ParameterDataset,
        index: DimIndex,
        value: float | np.ndarray | list[np.ndarray],
        fixed_indices: dict[str, list[int]],
    ):
        base_param_name = self.spec.base_params[0]
        arr = ds[base_param_name].values.copy()
        idx = self._slice_index_tuple(arr, ds[base_param_name].dims)
        arr[idx] = _write_at_array(
            arr[idx],
            self._slice_dims(ds),
            index,
            value,
            fixed_indices,
            base_param_name,
        )
        ds[base_param_name].values = arr

    def _slice_index_tuple(self, arr: np.ndarray, da_dims: list[str]) -> tuple:
        """Build an index tuple pointing at the configured slice."""
        return _position(da_dims, arr.ndim, self.spec.slice_dim, self.spec.slice_index)

    def _slice_dims(self, ds: ParameterDataset) -> list[str]:
        """Dimensions of the configured slice: the variable's dims minus slice_dim."""
        return [
            d for d in ds[self.spec.base_params[0]].dims if d != self.spec.slice_dim
        ]

    def _write_full(
        self,
        ds: ParameterDataset,
        default_ds: ParameterDataset,
        value: float | np.ndarray | list[np.ndarray],
        fixed_indices: dict[str, list[int]],
    ):
        base_param_name = self.spec.base_params[0]
        arr = ds[base_param_name].values.copy()
        idx = self._slice_index_tuple(arr, ds[base_param_name].dims)
        arr[idx] = _broadcast_to_array(
            arr[idx].copy(),
            value,
            self._slice_dims(ds),
            fixed_indices,
            base_param_name,
        )
        ds[base_param_name].values = arr


class ScaleFromRootParameter(Parameter, param_type="scale_from_root"):
    """Parameter whose value is root + delta.

    WARNING:
        **Write-order dependency.** set_value reads the current value of
        ds[root_param] and expects it to have already been written during
        this sampling step. Callers (e.g. the ensemble driver) **must** write
        root parameters before any ScaleFromRootParameter that depends on
        them. Violating this order produces silently incorrect values rather
        than an error.
    """

    def _validate_specs(self) -> None:
        """Require root_param and base_params to both be set, len(base_params) == 1,
        and root_param not to be the base param (a parameter can't scale from itself).
        """
        missing = []
        if self.spec.root_param is None:
            missing.append("root_param")
        if not self.spec.base_params:
            missing.append("base_params")
        if missing:
            raise ValueError(
                f"Parameter '{self.spec.name}' has param_type 'scale_from_root' "
                f"but the following required fields are not set: {missing}."
            )
        if len(self.spec.base_params) > 1:
            raise ValueError(
                f"Too many base_params for parameter {self.spec.name}. "
                "Shouldn't have more than one base_param for SlicedParameters. "
                f"Got {len(self.spec.base_params)}: {self.spec.base_params}"
            )
        if self.spec.root_param in self.spec.base_params:
            raise ValueError(
                f"Parameter '{self.spec.name}' has root_param "
                f"'{self.spec.root_param}' as its base_param; a scale_from_root "
                "parameter can't scale from itself."
            )

    def _variables_to_validate(self) -> list[str]:
        """Include both base_params and root_param in dataset validation."""
        return [*self.spec.base_params, self.spec.root_param]

    def _full_default(self, default_ds: ParameterDataset) -> np.ndarray:
        return default_ds[self.spec.base_params[0]].values

    def _write_at_index(
        self,
        ds: ParameterDataset,
        index: DimIndex,
        value: float | np.ndarray | list[np.ndarray],
        fixed_indices: dict[str, list[int]],
    ):
        # root_param must already be written by the caller — see class docstring.
        # root_param has the same dims as base_param (checked by _validate_params)
        base_param_name = self.spec.base_params[0]
        var = ds[base_param_name]
        root_at_index = _take(ds[self.spec.root_param].values, var.dims, index)
        delta = _fit_to_shape(value, np.shape(root_at_index), base_param_name)
        _write_at(var, index, root_at_index + delta, fixed_indices, base_param_name)

    def _write_full(
        self,
        ds: ParameterDataset,
        default_ds: ParameterDataset,
        value: float | np.ndarray | list[np.ndarray],
        fixed_indices: dict[str, list[int]],
    ):
        # root_param must already be written by the caller — see class docstring.
        base_param_name = self.spec.base_params[0]
        var = ds[base_param_name]
        root_arr = ds[self.spec.root_param].values
        new_arr = np.array(root_arr + np.asarray(value), dtype=float)
        _hold_fixed(
            new_arr, default_ds[base_param_name].values, var.dims, fixed_indices
        )
        var.values = new_arr


class JointParameter(Parameter, param_type="joint"):
    """Parameter which stands for multiple connected parameters (e.g. posterior draws).

    The value passed to set_value and returned by sample is a
    sequence of arrays — one per entry in spec.base_params.  The type
    annotations on the base class use float | np.ndarray | list[np.ndarray]
    for generality, but for this subclass only list[np.ndarray] (or any
    sequence of array-likes of the same length as base_params) is valid.

    """

    def _validate_specs(self):
        """Require base_params to be non-empty."""
        if not self.spec.base_params:
            raise ValueError(
                f"Parameter '{self.spec.name}' has param_type 'joint' but "
                "base_params is not set."
            )

    def _variables_to_validate(self) -> list[str]:
        return self.spec.base_params

    def _full_default(self, default_ds: ParameterDataset) -> list[np.ndarray]:
        return [default_ds[p].values for p in self.spec.base_params]

    def _coerce_value_seq(
        self, value: float | np.ndarray | list[np.ndarray]
    ) -> list[np.ndarray]:
        """Validate and return value as a list aligned with base_params

        Args:
            value (float | np.ndarray | list[np.ndarray]): input value

        Returns:
            list[np.ndarray]: output list
        """
        try:
            value_seq = list(value)
        except TypeError as te:
            raise TypeError(
                f"Parameter '{self.spec.name}' (joint): expected a sequence of "
                f"{len(self.spec.base_params)} arrays (one per base_param) but "
                f"got a non-iterable value of type {type(value).__name__}."
            ) from te
        if len(value_seq) != len(self.spec.base_params):
            raise ValueError(
                f"Parameter '{self.spec.name}' (joint): expected "
                f"{len(self.spec.base_params)} arrays (one per base_param: "
                f"{self.spec.base_params}) but got {len(value_seq)}."
            )
        return value_seq

    def _write_at_index(
        self,
        ds: ParameterDataset,
        index: DimIndex,
        value: float | np.ndarray | list[np.ndarray],
        fixed_indices: dict[str, list[int]],
    ):
        """ "Write a list of arrays into the dataset, one per base_param.

        Args:
            ds (ParameterDataset): Working copy of the parameter dataset. Modified in place.
            index (DimIndex): The dimension and position to write.
            value (float | np.ndarray | list[np.ndarray]): One array per entry in
                spec.base_params. Must have the same length as spec.base_params.
            fixed_indices (dict[str, list[int]]): Positions on other dimensions to
                leave unchanged.

        Raises:
            TypeError: If value is not iterable (e.g. a bare float was passed).
            ValueError: If len(value) does not match len(spec.base_params).
        """
        for parameter, val in zip(self.spec.base_params, self._coerce_value_seq(value)):
            _write_at(ds[parameter], index, val, fixed_indices, parameter)

    def _write_full(
        self,
        ds: ParameterDataset,
        default_ds: ParameterDataset,
        value: float | np.ndarray | list[np.ndarray],
        fixed_indices: dict[str, list[int]],
    ):
        for parameter, val in zip(self.spec.base_params, self._coerce_value_seq(value)):
            var = ds[parameter]
            var.values = _broadcast_to_array(
                var.values.copy(), val, var.dims, fixed_indices, parameter
            )


# ----------------------------------------------------------------------------------------
# Private helpers
# ----------------------------------------------------------------------------------------


def _position(dims: list[str], ndim: int, dim: str, index: int | list[int]) -> tuple:
    """Build an index tuple selecting index along dim and everything along other dims.

    Args:
        dims (list[str]): dimension names of the array, in order
        ndim (int): number of dimensions of the array
        dim (str): dimension to index along
        index (int | list[int]): position(s) along dim

    Raises:
        ValueError: dim is not one of dims

    Returns:
        tuple: index tuple
    """
    dims = list(dims)
    if dim not in dims:
        raise ValueError(
            f"Dimension '{dim}' is not one of the variable's dimensions {dims}."
        )
    idx: list = [slice(None)] * ndim
    idx[dims.index(dim)] = index
    return tuple(idx)


def _take(arr: np.ndarray, dims: list[str], index: DimIndex) -> float | np.ndarray:
    """Return the value(s) of arr at index along index.dim.

    Args:
        arr (np.ndarray): input array
        dims (list[str]): dimension names of arr, in order
        index (DimIndex): dimension and position to select

    Returns:
        float | np.ndarray: scalar for a 1-D array, otherwise the sub-array
    """
    return arr[_position(dims, np.ndim(arr), index.dim, index.index)]


def _hold_fixed(
    new: np.ndarray,
    held: np.ndarray,
    dims: list[str],
    fixed_indices: dict[str, list[int]],
) -> None:
    """Copy the fixed positions of held into new, in place.

    A position is fixed if its index along any dimension in fixed_indices is listed.

    Args:
        new (np.ndarray): array being written. Modified in place.
        held (np.ndarray): array holding the values to keep, same shape as new
        dims (list[str]): dimension names of both arrays, in order
        fixed_indices (dict[str, list[int]]): dimension-to-indices mapping of
            positions to keep. Dimensions the arrays don't have are ignored.
    """
    for dim, indices in fixed_indices.items():
        if dim in dims and len(indices) > 0:
            pos = _position(dims, new.ndim, dim, list(indices))
            new[pos] = held[pos]


def _write_at_array(
    arr: np.ndarray,
    dims: list[str],
    index: DimIndex,
    value: float | np.ndarray,
    fixed_indices: dict[str, list[int]],
    name: str,
) -> np.ndarray:
    """Return a copy of arr with value written at index along index.dim.

    Fixed positions on the other dimensions keep their values from arr.

    Args:
        arr (np.ndarray): array to write into
        dims (list[str]): dimension names of arr, in order
        index (DimIndex): dimension and position to write
        value (float | np.ndarray): scalar, or array shaped like the sub-array at index
        fixed_indices (dict[str, list[int]]): positions to leave unchanged
        name (str): parameter name, for error messages

    Returns:
        np.ndarray: output array
    """
    result = arr.copy()
    pos = _position(dims, result.ndim, index.dim, index.index)
    result[pos] = _fit_to_shape(value, result[pos].shape, name)
    _hold_fixed(result, arr, dims, fixed_indices)
    return result


def _write_at(
    variable: ParameterVariable,
    index: DimIndex,
    value: float | np.ndarray,
    fixed_indices: dict[str, list[int]],
    name: str,
) -> None:
    """Write value into a dataset variable at index along index.dim.

    Args:
        variable (ParameterVariable): variable to write. Modified in place.
        index (DimIndex): dimension and position to write
        value (float | np.ndarray): scalar, or array shaped like the sub-array at index
        fixed_indices (dict[str, list[int]]): positions to leave unchanged
        name (str): parameter name, for error messages
    """
    variable.values = _write_at_array(
        variable.values, variable.dims, index, value, fixed_indices, name
    )


def _broadcast_to_array(
    arr: np.ndarray,
    value: float | np.ndarray,
    dims: list[str],
    fixed_indices: dict[str, list[int]],
    name: str,
) -> np.ndarray:
    """Write value into arr at all non-fixed positions.

    Scalar value: broadcast to every non-fixed position.
    Array value: must match arr shape; fixed positions are skipped.

    Args:
        arr (np.ndarray): current array from dataset
        value (float | np.ndarray): input value
        dims (list[str]): dimension names of arr, in order
        fixed_indices (dict[str, list[int]]): dimension-to-indices mapping of
            positions to keep at their current value
        name (str): parameter name

    Raises:
        ValueError: incorrect shape

    Returns:
        np.ndarray: output array
    """
    value_arr = np.asarray(value)
    if value_arr.ndim > 0 and value_arr.shape != arr.shape:
        raise ValueError(
            f"Parameter '{name}': value shape {value_arr.shape} does not "
            f"match target array shape {arr.shape}."
        )

    if arr.ndim == 0:
        return np.asarray(float(value_arr))

    result = arr.copy()
    result[...] = value_arr
    _hold_fixed(result, arr, dims, fixed_indices)
    return result


def _fit_to_shape(
    value: float | np.ndarray, shape: tuple[int, ...], name: str
) -> float | np.ndarray:
    """Return value ready to write into a target of the given shape.

    Args:
        value (float | np.ndarray): input value
        shape (tuple[int, ...]): shape of the target
        name (str): parameter name

    Raises:
        ValueError: value is neither a scalar nor shaped like the target

    Returns:
        float | np.ndarray: a Python float for scalar (or single-element) input,
        otherwise the array itself
    """
    arr = np.asarray(value)
    if arr.size == 1:
        return float(arr.item())
    if arr.shape == tuple(shape):
        return arr
    raise ValueError(
        f"Parameter '{name}': expected a scalar value or an array of shape "
        f"{tuple(shape)} but got an array of shape {arr.shape}. "
        "Pass a scalar or expand the spec."
    )
