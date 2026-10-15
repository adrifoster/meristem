"""
Ensemble configuration dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _is_int(value) -> bool:
    """True for Python and numpy integers, but not bools."""
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _check_int(name: str, value, minimum: int, allow_none: bool = False) -> None:
    """Raise ValueError unless value is an integer >= minimum (or None if allowed)."""
    if allow_none and value is None:
        return
    if not _is_int(value) or value < minimum:
        expected = f"an integer >= {minimum}" + (" or None" if allow_none else "")
        raise ValueError(f"{name} must be {expected}, got {value!r}.")


@dataclass
class EnsembleConfig:
    """Base configuration shared by all ensemble types.

    Attributes
    ----------
    param_dir : Path
        Path to the directory with the parameter metadata datafiles
    ensemble_dir : Path
        Directory where ensemble member files will be written.
    file_prefix : str
        Prefix for output filenames, e.g. 'my_ensemble' produces
        'my_ensemble_0001.nc', 'my_ensemble_0002.nc', etc.
    default_data_file : Path
        Path to default parameter dataset. Used as the base for all ensemble
        members and for parameter validation at construction time.
    param_list : list[str] | None
        Optional subset of parameter names to use. If None, all
        parameters in the spreadsheet are used.
    fixed_indices : dict[str, list[int]] | None
        Run-level mapping of dimension name to 0-based indices to hold at
        their default values across all ensemble members. For example,
        ``{'fates_pft': [7, 8, 9]}`` fixes PFTs 8, 9, and 10 (0-based).
        If None, all indices are free.
    posterior_sources : Path | None
        Path to a YAML file defining posterior distribution sources for
        parameters with strategy='posterior'. If None, no posterior
        parameters are expected.
    """

    param_dir: Path
    ensemble_dir: Path
    file_prefix: str
    default_param_file: Path
    param_list: list[str] | None = None
    fixed_indices: dict[str, list[int]] | None = None
    posterior_sources: Path | None = None

    def __post_init__(self):
        self.param_dir = Path(self.param_dir)
        self.ensemble_dir = Path(self.ensemble_dir)
        self.default_param_file = Path(self.default_param_file)
        if self.posterior_sources is not None:
            self.posterior_sources = Path(self.posterior_sources)

        if (
            not isinstance(self.file_prefix, str)
            or not self.file_prefix
            or "/" in self.file_prefix
        ):
            raise ValueError(
                "file_prefix must be a non-empty string without '/', "
                f"got {self.file_prefix!r}."
            )

        if self.param_list is not None and (
            isinstance(self.param_list, str)
            or not self.param_list
            or not all(isinstance(name, str) for name in self.param_list)
        ):
            raise ValueError(
                "param_list must be a non-empty list of parameter names or None, "
                f"got {self.param_list!r}."
            )

        if self.fixed_indices is not None:
            if not isinstance(self.fixed_indices, dict):
                raise ValueError(
                    "fixed_indices must be a dict of dimension name to a list of "
                    f"indices, got {self.fixed_indices!r}."
                )
            for dim, indices in self.fixed_indices.items():
                if isinstance(indices, (str, bytes)) or not all(
                    _is_int(i) for i in np.atleast_1d(indices)
                ):
                    raise ValueError(
                        f"fixed_indices['{dim}'] must be a list of integer indices, "
                        f"got {indices!r}."
                    )


@dataclass
class LatinHypercubeConfig(EnsembleConfig):
    """Configuration for a Latin Hypercube ensemble.

    Attributes
    ----------
    ensemble_members : int
        Number of ensemble members to generate. Each member corresponds
        to one row of the Latin Hypercube sample matrix.
    prebuilt : np.ndarray | None
        Optional pre-built Latin Hypercube sample matrix of shape
        (n_samples, n_params). If supplied, this matrix is used directly
        instead of generating a new one. Useful for reproducibility or
        when the LH matrix was generated externally.
    seed : int | None
        optional seed for reproducibility
    """

    ensemble_members: int = 100
    prebuilt: np.ndarray | None = None
    seed: int | None = None

    def __post_init__(self):
        super().__post_init__()

        _check_int("ensemble_members", self.ensemble_members, minimum=1)
        _check_int("seed", self.seed, minimum=0, allow_none=True)

        if self.prebuilt is not None:
            self.prebuilt = np.asarray(self.prebuilt, dtype=float)
            if self.prebuilt.ndim != 2:
                raise ValueError(
                    "prebuilt must be a 2-D array of shape "
                    f"(ensemble_members, n_params), got {self.prebuilt.ndim}-D."
                )
            if self.prebuilt.shape[0] != self.ensemble_members:
                raise ValueError(
                    f"prebuilt has {self.prebuilt.shape[0]} rows but "
                    f"ensemble_members is {self.ensemble_members}."
                )
            if not np.all((self.prebuilt >= 0.0) & (self.prebuilt <= 1.0)):
                raise ValueError("prebuilt values must all be in [0, 1].")


@dataclass
class OneAtATimeConfig(EnsembleConfig):
    """Configuration for a One-At-A-Time (OAT) sensitivity ensemble."""


@dataclass
class MorrisEnsembleConfig(EnsembleConfig):
    """Configuration for a Morris sensitivity ensemble.

    Attributes
    ----------
    morris_n : int
        Number of trajectories to generate.
    levels : int
        Number of grid levels for partitioning parameter space. Must be a
        positive even number.
    optimal_trajectories : int | None
        If set, the number of optimal trajectories to select from the larger
        pool of morris_n (at least 2 and less than morris_n). If None, all
        morris_n trajectories are used.
    seed : int | None
        optional seed for reproducibility
    """

    morris_n: int = 100
    levels: int = 4
    optimal_trajectories: int | None = None
    seed: int | None = None

    def __post_init__(self):
        super().__post_init__()
        _check_int("morris_n", self.morris_n, minimum=1)
        if not _is_int(self.levels) or self.levels < 2 or self.levels % 2 != 0:
            raise ValueError(
                f"levels must be a positive even number, got {self.levels!r}."
            )
        _check_int(
            "optimal_trajectories",
            self.optimal_trajectories,
            minimum=2,
            allow_none=True,
        )
        if (
            self.optimal_trajectories is not None
            and self.optimal_trajectories >= self.morris_n
        ):
            raise ValueError(
                f"optimal_trajectories ({self.optimal_trajectories}) must be less "
                f"than morris_n ({self.morris_n})."
            )
        _check_int("seed", self.seed, minimum=0, allow_none=True)
