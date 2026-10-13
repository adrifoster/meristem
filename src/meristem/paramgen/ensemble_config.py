"""
Ensemble configuration dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


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

    param_dir: str | Path
    ensemble_dir: str | Path
    file_prefix: str
    default_param_file: str | Path
    param_list: Optional[list[str]] = None
    fixed_indices: Optional[dict[str, list[int]]] = None
    posterior_sources: Optional[Path] = None

    def __post_init__(self):
        self.param_dir = Path(self.param_dir)
        self.ensemble_dir = Path(self.ensemble_dir)
        self.default_param_file = Path(self.default_param_file)
        if self.posterior_sources is not None:
            self.posterior_sources = Path(self.posterior_sources)


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
    """

    ensemble_members: int = 100
    prebuilt: Optional[np.ndarray] = None


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
        pool of morris_n (between 2 and morris_n). If None, all morris_n
        trajectories are used.
    """

    morris_n: int = 100
    levels: int = 4
    optimal_trajectories: int | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.levels < 2 or self.levels % 2 != 0:
            raise ValueError(
                f"levels must be a positive even number, got {self.levels}."
            )
