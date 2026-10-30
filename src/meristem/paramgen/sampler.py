"""
Sampler class

This class in in charge of sampling a parameter given an input normalized_value, and
normalizing a concrete parameter to a normalized value

"""

from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import numpy as np

from .distribution_stat import DistributionStat
from .posterior_source import PosteriorSource, _DEFAULT_SORT_INDEX
from .utils import validate_normalized_value


@dataclass
class SampleContext:
    """Sampling context passed to Sampler.

    Each Sampler subclass uses only the fields relevant to it and ignores
    the rest. This avoids a mixed-concern signature on the abstract interface
    where some arguments are only meaningful to one subclass.

    Attributes
    ----------
    default_value : float | np.ndarray | list[np.ndarray] | None
        Default value(s) for this parameter from the dataset.
        Required by UniformSampler to resolve percent bounds.
        Used by PosteriorSampler in broadcast mode as the starting values, so
        positions no source covers keep their defaults.
    array_index : int | None
        The active array index when sampling an expanded (per-index) parameter.
        None when sampling all free indices together (broadcast mode).
        Used by PosteriorSampler to select the correct PosteriorSource, and by
        UniformSampler to select this index from bounds resolved over the full array.
    index_axis : int | None
        Axis of default_value that array_index selects along. None when
        array_index is None.
    n_indices : list[int] | None
        Number of positions along each free dimension ([1] for scalars), e.g.
        [3] for a per-PFT parameter. Used by PosteriorSampler in broadcast mode
        as the shape of the output arrays.
        Ignored by UniformSampler.
    pft_axis: int | None
        Axis along which the pft-dimension exists
    """

    default_value: float | np.ndarray | list[np.ndarray] | None = None
    array_index: int | None = None
    index_axis: int | None = None
    n_indices: list[int] | None = None
    pft_axis: int | None = None


class Sampler(ABC):
    """Class for parameter sampling.

    Subclasses must implement:
        - __init__(self, row, pft_sheet, posterior_config): parse all
          sampling configuration from the spreadsheet row at construction time.
        - sample(self, normalized_value, context): draw a value given a normalised input.
        - normalize(self, value, context, clip_to_bounds): convert a concrete value back
            to normalised [0, 1].
    """

    _registry: dict[str, type[Sampler]] = {}

    def __init_subclass__(cls, sampler_type: str, **kwargs):
        super().__init_subclass__(**kwargs)
        Sampler._registry[sampler_type] = cls

    @classmethod
    def from_row_and_sheet(
        cls,
        row: pd.Series,
        pft_sheet: pd.DataFrame | None = None,
        posterior_config: dict | None = None,
    ) -> Sampler:
        """Construct Sampler from a main sheet row, pft_sheet, or posterior_config

        Args:
            row (pd.Series): A row from the main sheet.
            pft_sheet (pd.DataFrame | None, optional): The per-parameter PFT sheet,
                required when a relevant column is 'pft'. Ignored otherwise.
                Defaults to None.

        Raises:
            Unknown strategy

        Returns:
            Sampler: Sampler
        """
        param_strategy = str(row.get("strategy", "")).strip()
        subclass = cls._registry.get(param_strategy)
        if subclass is None:
            raise ValueError(
                f"Unknown strategy '{param_strategy}'. "
                f"Valid types: {sorted(cls._registry)}"
            )
        return subclass(
            row, pft_sheet, posterior_config
        )  # pylint: disable=not-callable

    @abstractmethod
    def sample(
        self,
        normalized_value: float,
        context: SampleContext,
    ) -> float | np.ndarray:
        """Generate a sample for a parameter

        Args:
            normalized_value (float): normalized [0-1] input value
            context: Sampling context. Each subclass uses the fields relevant to its
                strategy and ignores the rest.
        Returns:
            float | np.ndarray: sampled parameter value.
        """

    @abstractmethod
    def normalize(
        self,
        value: float | np.ndarray,
        context: SampleContext,
        clip_to_bounds: bool = False,
    ) -> float | np.ndarray:
        """Convert a concrete parameter value into a normalized [0, 1] value.

        Args:
            value (float | np.ndarray): concrete parameter to normalize
            context: Sampling context. Each subclass uses the fields relevant to its
                strategy and ignores the rest.
            clip_to_bounds (bool, optional): optionally clip to [0-1] if outside range,
                otherwise method will raise a ValueError. Defaults to None

        Returns:
            float | np.ndarray: normalized value(s) in [0 to 1]
        """


class UniformSampler(Sampler, sampler_type="uniform"):
    """Uniform Sampler - scales between a minimum and a maximum given an input [0-1] value

    Attributes
    ===========
    min_stat: DistributionStat
        Minimum parameter bound
    max_stat: DistributionStat
        Maximum parameter bound
    """

    def __init__(
        # pylint: disable=unused-argument
        self,
        row: pd.Series,
        pft_sheet: pd.DataFrame | None = None,
        posterior_config: dict | None = None,
    ):

        min_raw = str(row.get("param_min", "")).strip().lower()
        max_raw = str(row.get("param_max", "")).strip().lower()

        if (min_raw == "pft") != (max_raw == "pft"):
            raise ValueError(
                f"Parameter '{row.get('parameter_name')}': param_min and param_max "
                "must both be 'pft' or neither — mixing is not supported."
            )

        self.min_stat = DistributionStat.parse(row, "min", pft_sheet)
        self.max_stat = DistributionStat.parse(row, "max", pft_sheet)

    def resolve_bounds(
        self,
        default_value: float | np.ndarray | None = None,
        pft_axis: int | None = None,
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        """Resolve min and max bounds and return (min_val, max_val).

        Args:
            default_value (float | np.ndarray | None, optional): default value from
            parameter file. Required if either bound is a PercentBound. Defaults to None.
            pft_axis (int | None, optional) Axis along which pfts are dimensioned

        Returns:
            tuple[float | np.ndarray, float | np.ndarray]: (min_val, max_val)
        """

        min_val = self.min_stat.resolve(default_value, pft_axis=pft_axis)
        max_val = self.max_stat.resolve(default_value, pft_axis=pft_axis)

        _validate_bounds(min_val, max_val)

        return min_val, max_val

    def sample(
        self,
        normalized_value: float,
        context: SampleContext,
    ) -> float | np.ndarray:
        """Generate a sample for a parameter"""
        validate_normalized_value(normalized_value)
        min_val, max_val = self.resolve_bounds(context.default_value, context.pft_axis)
        return _at_array_index(
            min_val + normalized_value * (max_val - min_val), context
        )

    def normalize(
        self,
        value: float | np.ndarray,
        context: SampleContext,
        clip_to_bounds: bool = False,
    ) -> float | np.ndarray:
        """Convert a concrete parameter value into a normalized [0, 1] value.

        Positions where min == max have no range to normalize against — the
        parameter is effectively fixed there — so they return 0.5 rather than
        dividing by zero.
        """

        min_val, max_val = (
            _at_array_index(bound, context)
            for bound in self.resolve_bounds(
                context.default_value, pft_axis=context.pft_axis
            )
        )

        if clip_to_bounds:
            value = np.clip(value, min_val, max_val)
        if np.any(value > max_val):
            raise ValueError(
                f"value: {value} exceeds maximum value(s). "
                f"maximum value is {max_val}"
            )
        if np.any(value < min_val):
            raise ValueError(
                f"value: {value} is below minimum value(s). "
                f"minimum value is {min_val}"
            )

        span = np.asarray(max_val, dtype=float) - np.asarray(min_val, dtype=float)
        degenerate = span == 0.0
        normalized = np.where(
            degenerate,
            0.5,
            (np.asarray(value, dtype=float) - min_val)
            / np.where(degenerate, 1.0, span),
        )
        return float(normalized) if normalized.ndim == 0 else normalized


class PosteriorSampler(Sampler, sampler_type="posterior"):
    """Posterior Sampler - pulls from a posterior distribution

    Attributes
    ===========
    parameters: list[str]
        list of column/parameter names in each PosteriorSource
    sources: list[PosteriorSource]
        list of PosteriorSources that can be used to draw from a distribution
    """

    def __init__(
        # pylint: disable=unused-argument
        self,
        row: pd.Series,
        pft_sheet: pd.DataFrame | None = None,
        posterior_config: dict | None = None,
    ):
        if posterior_config is None:
            raise ValueError(
                f"Parameter '{row.get('parameter_name')}' has "
                "strategy='posterior' but no posterior_sources yaml was supplied."
            )

        self.parameters = posterior_config["parameters"]
        sources = [
            PosteriorSource(
                path=Path(file_entry["path"]),
                array_indices=file_entry["array_indices"],
                parameters=self.parameters,
                sort_param_index=posterior_config.get(
                    "sort_index", _DEFAULT_SORT_INDEX
                ),
            )
            for file_entry in posterior_config["files"]
        ]
        _check_sources(row.get("parameter_name"), sources)
        self.sources = sources

    def covers(self, array_index: int) -> bool:
        """Whether some source has draws for an array index

        Args:
            array_index (int): array index

        Returns:
            bool: True if a source covers array_index
        """
        return any(
            source.is_broadcast or array_index in source.array_indices
            for source in self.sources
        )

    def sample(
        self,
        normalized_value: float,
        context: SampleContext,
    ) -> float | np.ndarray:
        """Generate a sample for a parameter"""
        if context.array_index is not None:
            return self._draw_for_index(normalized_value, context.array_index)
        return self._draw_broadcast(normalized_value, context)

    def normalize(
        self,
        value: float | np.ndarray,
        context: SampleContext,
        clip_to_bounds: bool = False,
    ) -> float | np.ndarray:
        """Convert a concrete parameter value into a normalized [0, 1] value."""
        if context.array_index is not None:
            return self._unscale_for_index(
                value, context.array_index, clip_to_bounds=clip_to_bounds
            )
        return self._unscale_broadcast(
            value, context.n_indices, clip_to_bounds=clip_to_bounds
        )

    def _draw_for_index(
        self, normalized_value: float, array_index: int
    ) -> list[np.ndarray]:
        """draws from a PosteriorSource given an input array_index

        Args:
            normalized_value (float): input normalized [0-1] value to use as quantile
            array_index (int): array index we are pulling for

        Returns:
            list[np.ndarray]: output parameter value(s)
        """
        source = self._source_for_index(array_index)
        row = source.draw_row(normalized_value)
        return [np.array([row[v]]) for v in self.parameters]

    def _unscale_for_index(
        self, value: float, array_index: int, clip_to_bounds: bool = False
    ) -> float:
        """convert a concrete parameter value into a normalized [0-1] value for a
        specific array index

        Args:
            value (float): concrete parameter value
            array_index (int): array index to use
            clip_to_bounds (bool, optional): optionally clip to [0-1] if outside bounds
            Defaults to False.

        Returns:
            float: normalized value
        """
        source = self._source_for_index(array_index)
        return source.unscale(value, clip_to_bounds=clip_to_bounds)

    def _source_for_index(self, array_index: int) -> PosteriorSource:
        """Return the correct PosteriorSource given an input index

        Args:
            array_index (int): input array_index for PosteriorSource

        Raises:
            ValueError: No source found for the input array index

        Returns:
            PosteriorSource: Correct PosteriorSource
        """
        for source in self.sources:
            if source.is_broadcast or array_index in source.array_indices:
                return source
        raise ValueError(
            f"No source found for array index {array_index}"
            f"Check your posterior_sources.yaml."
        )

    def _unscale_broadcast(
        self,
        value: float | list[np.ndarray],
        n_indices: list[int] | None,
        clip_to_bounds: bool = False,
    ) -> list[np.ndarray]:
        if len(self.sources) == 1 and self.sources[0].is_broadcast:
            result = [np.zeros(n_indices) for _ in self.parameters]
            sort_idx = self.sources[0].sort_param_index
            sort_values = np.broadcast_to(
                np.asarray(
                    value[sort_idx] if isinstance(value, list) else value, dtype=float
                ),
                tuple(n_indices),
            )
            # one draw applies to every index, but each index's value sits at
            # its own quantile of the draws (e.g. per-PFT defaults)
            for idx in np.ndindex(*n_indices):
                unscaled = self.sources[0].unscale(
                    float(sort_values[idx]), clip_to_bounds=clip_to_bounds
                )
                for k, _ in enumerate(self.parameters):
                    result[k][idx] = unscaled
        else:
            # index-specific sources (_check_sources: none is broadcast). A position
            # no source covers isn't sampled, so it has no normalized value
            result = [np.full(n_indices, np.nan) for _ in self.parameters]
            for source in self.sources:
                sort_idx = source.sort_param_index
                for array_idx in source.array_indices:
                    if isinstance(value, list):
                        sort_value = float(value[sort_idx][array_idx])
                    else:
                        sort_value = float(value)
                    row = source.unscale(sort_value, clip_to_bounds=clip_to_bounds)
                    for k, _ in enumerate(self.parameters):
                        result[k][array_idx] = row
        return result

    def _draw_broadcast(
        self, normalized_value: float, context: SampleContext
    ) -> list[np.ndarray]:
        if len(self.sources) == 1 and self.sources[0].is_broadcast:
            result = [np.zeros(context.n_indices) for _ in self.parameters]
            row = self.sources[0].draw_row(normalized_value)
            for k, var in enumerate(self.parameters):
                result[k][:] = row[var]
        else:
            # index-specific sources (_check_sources: none is broadcast). Start
            # from the defaults so a position no source covers keeps its default
            result = self._starting_values(context)
            for source in self.sources:
                row = source.draw_row(normalized_value)
                for array_idx in source.array_indices:
                    for k, var in enumerate(self.parameters):
                        result[k][array_idx] = row[var]

        return result

    def _starting_values(self, context: SampleContext) -> list[np.ndarray]:
        """Each parameter's default values as a float array, or nan where there
        are no defaults, to fill in from the sources

        Args:
            context (SampleContext): sampling context

        Returns:
            list[np.ndarray]: one array per parameter, shaped n_indices
        """
        if context.default_value is None:
            return [np.full(context.n_indices, np.nan) for _ in self.parameters]
        defaults = context.default_value
        if not isinstance(defaults, list):
            defaults = [defaults]
        return [
            np.array(default, dtype=float).reshape(context.n_indices)
            for default in defaults
        ]


def _check_sources(name: str | None, sources: list[PosteriorSource]) -> None:
    """Check a parameter's posterior sources say unambiguously where each index's
    draws come from

    Args:
        name (str | None): parameter name, for errors
        sources (list[PosteriorSource]): the parameter's posterior sources

    Raises:
        ValueError: a broadcast source (array_indices 'all') isn't the only source
        ValueError: an array index is listed by more than one source
    """
    if len(sources) > 1 and any(source.is_broadcast for source in sources):
        raise ValueError(
            f"Parameter '{name}': a posterior file with array_indices 'all' must "
            f"be its only file, but it has {len(sources)}. List the indices each "
            "file covers instead."
        )
    counts = Counter(
        i for source in sources if not source.is_broadcast for i in source.array_indices
    )
    repeated = sorted(i for i, count in counts.items() if count > 1)
    if repeated:
        raise ValueError(
            f"Parameter '{name}': array indices {repeated} are listed by more than "
            "one posterior file; each index can come from only one file."
        )


def _at_array_index(
    value: float | np.ndarray, context: SampleContext
) -> float | np.ndarray:
    """Select an expanded parameter's position from a value resolved over the full array.

    Bounds are resolved against the full default array, so per-index bounds (e.g.
    per-PFT bounds) land in the right place; an expanded parameter then takes only
    its own index. Scalars and unexpanded parameters pass through unchanged.

    Args:
        value (float | np.ndarray): value over the full array, or a scalar
        context (SampleContext): sampling context

    Returns:
        float | np.ndarray: value at context.array_index along context.index_axis
    """
    if context.array_index is None or np.ndim(value) == 0:
        return value
    axis = 0 if context.index_axis is None else context.index_axis
    return np.take(value, context.array_index, axis=axis)


def _validate_bounds(
    min_val: float | np.ndarray,
    max_val: float | np.ndarray,
) -> None:
    """Raise error if any min > max after resolution

    Args:
        min_val (float | np.ndarray): minimum value
        max_val (float | np.ndarray): maximum value

    Raises:
        ValueError: Parameter min > max
    """
    if min_val is None or max_val is None:  # pragma: no cover
        raise ValueError(
            f"Parameter min or max is None  - cannot scale "
            f"(min={min_val}, max={max_val}). Check inputs"
        )

    min_arr = np.asarray(min_val)
    max_arr = np.asarray(max_val)
    if np.any(min_arr > max_arr):
        raise ValueError(
            f"Parameter has min > max " f"(min={min_val}, max={max_val}). Check inputs"
        )
