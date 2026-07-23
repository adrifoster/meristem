"""
ParamEnsemble class - responsible for generating the entire ensemble
"""

from __future__ import annotations
from dataclasses import dataclass, fields
from pathlib import Path
from abc import ABC, abstractmethod

from tqdm import tqdm
import yaml
import pandas as pd
import numpy as np
from scipy.stats import qmc

from .parameter_dataset import ParameterDataset
from .parameter import Parameter
from .ensemble_config import EnsembleConfig, LatinHypercubeConfig, OneAtATimeConfig
from .sort_params import sort_params
from .utils import read_param_list


@dataclass
class ParameterSample:
    """A single parameter paired with its normalized sample value.

    Attributes
    ----------
    group: ParamGroup
        The group of parameters being sampled together
    normalized_value: float
        A value in [0, 1]. For uniform parameters this is passed to a scaler to produce
        an actual value. For posterior parameters this is used as a quantile index into
        the posterior distribution.
    """

    group: ParamGroup
    normalized_value: float


@dataclass
class ParamGroup:
    """A group of parameters that share the same normalized sample value.

    For grouped parameters, all parameters in the group get the same normalized
    value. For ungrouped parameters, the group contains a single parameter.

    Attributes
    ----------
    name : str
        Group name, or parameter spec name for ungrouped parameters.
    params : list[Parameter]
        Parameters in this group.
    """

    name: str
    params: list[Parameter]


@dataclass
class EnsembleMemberSample:
    """All parameter samples for one ensemble member

    Attributes
    ----------
    parameter_samples : list[ParameterSample]
        One ParameterSample per parameter being varied in this ensemble member.
    """

    parameter_samples: list[ParameterSample]

    def __iter__(self):
        """Iterate over the ParameterSamples in this member"""
        return iter(self.parameter_samples)

    def __len__(self):
        """Return the number of ParameterSamples in this member"""
        return len(self.parameter_samples)


class ParamEnsemble(ABC):
    """Abstract base class for the parameter ensemble class

    Attributes
    ----------
    ensemble_dir : Path
        Path to where parameter files will be written out
    file_prefix: str
        Prefix for output filenames, e.g. 'my_ensemble' produces 'my_ensemble_0001.nc', etc.
    default_ds: ParameterDataset
        Default parameter dataset. Used as the base for all ensemble
        members and for parameter validation at construction time.
    params: list[Parameter]
        list of Parameter objects to sample
    num_params: int
        number of parameters
    fixed_indices: dict[str, list[int]] | None
        Mapping of dimension name to 0-based indices to hold at
        their default values across all ensemble members. For example,
        {'fates_pft': [7, 8, 9]} fixes PFTs 8, 9, and 10 (0-based).
        If None, all indices are free.
    """

    _registry: dict[str, type[ParamEnsemble]] = {}

    def __init_subclass__(cls, ensemble_type: str, **kwargs):
        super().__init_subclass__(**kwargs)
        ParamEnsemble._registry[ensemble_type] = cls

    def __init__(
        self,
        config: EnsembleConfig,
    ):

        # read in the parameter metadata
        main, pft_sheets = read_param_list(config.param_dir)

        # subset to only a list of parameters if supplied
        if config.param_list is not None:
            missing = set(config.param_list) - set(main.parameter_name)
            if missing:
                raise ValueError(
                    f"param_list contains parameters not found in main.csv: "
                    f"{sorted(missing)}. "
                    f"Available parameters: {sorted(main.parameter_name)}"
                )
            main = main[main.parameter_name.isin(config.param_list)].copy()

        # create output directory if it doesn't exit yet
        self.ensemble_dir = Path(config.ensemble_dir)
        self.ensemble_dir.mkdir(parents=True, exist_ok=True)

        # set attributes
        self.file_prefix = config.file_prefix

        if not config.default_param_file.exists():
            raise FileNotFoundError(
                f"Default parameter file '{config.default_param_file}' does not exist."
            )
        self.default_ds = ParameterDataset.from_path(config.default_param_file)

        if config.posterior_sources:
            if not config.posterior_sources.exists():
                raise FileNotFoundError(
                    f"Posterior sources file '{config.posterior_sources}' does not exist."
                )
            with open(config.posterior_sources, "r", encoding="utf-8") as f:
                posterior_config = yaml.safe_load(f)

        self.fixed_indices: dict[str, list[int]] = config.fixed_indices or {}
        if self.fixed_indices:
            _validate_fixed_indices(self.fixed_indices, self.default_ds)

        # create sorted list of parameter objects
        # this automatically checks and sorts order to make sure anything
        # that depends on another parameter is either not being modified or
        # gets written first
        self.params = sort_params(
            [
                Parameter.from_row(
                    row,
                    pft_sheet=pft_sheets.get(row["parameter_name"]),
                    default_ds=self.default_ds,
                    posterior_config=(
                        posterior_config.get(row["parameter_name"])
                        if config.posterior_sources
                        else None
                    ),
                )
                for _, row in main.iterrows()
            ]
        )

        # if we have expand_dims on a parameter, expand them to be one per expanded index
        self._expand_params()

        # validate grouped parameters - we can't mix expanded dims within a group
        self._validate_groups()

        # build the groups
        self.sampling_units: list[ParamGroup] = []
        self._build_sampling_units()

        self.num_sampling_units = len(self.sampling_units)
        self.num_params = len(self.params)

    @classmethod
    def from_dict(
        cls,
        config: dict,
    ) -> ParamEnsemble:
        """Construct the correct ParamEnsemble subclass from an input configuration dict.

        The dict must contain an 'ensemble_type' key whose value matches
        a registered subclass (e.g. 'LatinHypercube'). All other keys
        are passed to the corresponding config dataclass.

        Args:
            config (dict): Configuration dictionary. Example::

            {
                'ensemble_type': 'LatinHypercube',
                'param_dir': 'param_dir/',
                'ensemble_dir': 'output/',
                'file_prefix': 'my_run',
                'default_ds': ds,
                'n_samples': 200,
                'fixed_indices': {'fates_pft': [7, 8, 9]},
                'posterior_sources': 'posteriors.yaml',

            }
        Raises:
            ValueError
                If 'ensemble_type' is missing or not a registered type.
            TypeError
                If the config dict contains keys not recognised by the config
                dataclass (misspelled or unsupported options).
        Returns:
            ParamEnsemble: A fully constructed ensemble subclass instance.
        """
        config = config.copy()

        ensemble_type = config.pop("ensemble_type", None)
        if ensemble_type is None:
            raise ValueError(
                "'ensemble_type' is required in the config dict. "
                f"Valid types: {sorted(cls._registry)}"
            )

        subclass = cls._registry.get(ensemble_type)
        if subclass is None:
            raise ValueError(
                f"Unknown ensemble_type '{ensemble_type}'. "
                f"Valid types: {sorted(cls._registry)}"
            )
        return subclass.from_config(config)

    def create_ensemble(self):
        """Create and write out all ensemble parameter files

        Args:
            ensemble parameter files.
        """
        samples = self.create_samples()
        for i, sample in tqdm(enumerate(samples), total=len(samples), unit="member"):
            ds = self.create_ensemble_member(sample)
            file_name = f"{self.file_prefix}_{_generate_suffix(i)}{self.default_ds.file_extension}"
            ds.save(self.ensemble_dir / file_name)
            ds.close()

        ensemble_key = self.create_ensemble_key(samples)
        ensemble_key.to_csv(
            self.ensemble_dir / f"{self.file_prefix}_key.csv", index=False
        )

        _write_ensemble_list(
            self.ensemble_dir,
            self.file_prefix,
            list(ensemble_key.ensemble.values),
        )

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict) -> ParamEnsemble:
        """Construct this subclass from a config dict (ensemble_type already removed).

        Implementations should construct the appropriate config dataclass
        from the dict, then pass it to the constructor:

        cfg = LatinHypercubeConfig(**config)
        return cls(cfg)

        Args:
            config (dict): Config dict with 'ensemble_type' already popped.

        Returns:
            ParamEnsemble: A fully constructed ensemble subclass instance.
        """

    @abstractmethod
    def create_samples(self) -> list[EnsembleMemberSample]:
        """Create samples from the list of parameters

        Returns:
            list[EnsembleMemberSample]: One EnsembleMemberSample per ensemble member,
            each containing one ParameterSample per parameter.
        """

    @abstractmethod
    def create_ensemble_member(self, sample: EnsembleMemberSample) -> ParameterDataset:
        """Create one member of the ensemble

        Args:
            sample (EnsembleMemberSample): Parameter samples for this member.
                Contains one ParameterSample per parameter.
        Returns:
            ParameterDataset: one member of the ensemble with updated values from default
        """

    @abstractmethod
    def create_ensemble_key(self, samples: list[EnsembleMemberSample]) -> pd.DataFrame:
        """Create the ensemble key that goes with this ensemble

        Args:
            samples (list[EnsembleMemberSample]): One EnsembleMemberSample per ensemble
                member, each containing one ParameterSample per parameter.

        Returns:
            pd.DataFrame: output data frame that serves as ensemble key
        """

    def _expand_params(self):
        """Expand params list with expand_dim set; pass others through unchanged."""

        # map of all dimension names to all valid 0-based indices
        full_index_map = {
            dim: list(range(self.default_ds.sizes[dim])) for dim in self.default_ds.dims
        }
        result = []
        for param in self.params:
            if param.spec.expand_dim is not None:
                result.extend(
                    _expand_one_param(param, self.fixed_indices, full_index_map)
                )
            else:
                result.append(param)
        self.params = result

    def _validate_groups(self):
        """Validate group constraints.

        Raises:
            ValueError: If a group contains parameters with mixed expand_dim values.
            ValueError: If a scale_from_root param and its root are in different groups.
        """
        # validate scale_from_root params and their roots are in the same group
        for param in self.params:
            if param.spec.root_param is None:
                continue
            root = next(
                (
                    p
                    for p in self.params
                    if p.spec.name == param.spec.root_param
                    or (
                        param.active_index is not None
                        and p.spec.name
                        == f"{param.spec.root_param}_{param.active_index.index}"
                    )
                ),
                None,
            )
            if root is None:
                continue
            if param.spec.group_name != root.spec.group_name:
                raise ValueError(
                    f"Parameter '{param.spec.name}' (scale_from_root) is in group "
                    f"'{param.spec.group_name}' but its root '{param.spec.root_param}' "
                    f"is in group '{root.spec.group_name}'. "
                    "A scale_from_root parameter and its root must be in the same group "
                    "or both ungrouped."
                )

        # validate all params in a group have the same expand_dim
        groups: dict[str, list[str | None]] = {}
        for param in self.params:
            group = param.spec.group_name
            if group is None:
                continue
            if group not in groups:
                groups[group] = []
            groups[group].append(param.spec.expand_dim)

        for group, expand_dims in groups.items():
            unique = set(expand_dims)
            if len(unique) > 1:
                raise ValueError(
                    f"Group '{group}' contains parameters with mixed expand_dim values: "
                    f"{sorted(str(d) for d in unique)}. "
                    "All parameters in a group must have the same expand_dim, or all None."
                )

    def _build_sampling_units(self):
        """Build sampling units from params list.

        Each group becomes one sampling unit. Ungrouped parameters each become their
        own singleton sampling unit
        """
        group_lookup: dict[str, ParamGroup] = {}
        self.sampling_units: list[ParamGroup] = []

        for param in self.params:
            group_name = param.spec.group_name
            if group_name is None:
                # ungrouped - one sampling unit per parameter
                self.sampling_units.append(
                    ParamGroup(name=param.spec.name, params=[param])
                )
            else:
                if group_name not in group_lookup:
                    new_group = ParamGroup(name=group_name, params=[])
                    group_lookup[group_name] = new_group
                    self.sampling_units.append(new_group)
                group_lookup[group_name].params.append(param)


class LatinHypercubeEnsemble(ParamEnsemble, ensemble_type="LatinHypercube"):
    """Concrete class for the Latin Hypercube ensemble class

    Parameters
    ----------
    n_samples : int
        number of ensemble members to generate. Each member corresponds
        to one row of the Latin Hypercube sample matrix.
    prebuilt: np.ndarray | None
        Optional pre-built Latin Hypercube sample matrix of shape
        (n_samples, n_params). If supplied, this matrix is used directly
        instead of generating a new one. Useful for reproducibility or
        when the LH matrix was generated externally.
    posterior_configs: dict[str, PosteriorConfig]
        PosteriorConfig objects for pulling from a distribution

    """

    def __init__(
        self,
        config: LatinHypercubeConfig,
    ):
        super().__init__(config)
        self.n_samples = config.ensemble_members
        self.prebuilt = config.prebuilt

    @classmethod
    def from_config(cls, config: dict) -> LatinHypercubeEnsemble:
        """Construct from a plain dict (ensemble_type already removed)

        Args:
            config (dict): Must contain all required LatinHypercubeConfig fields.
            Unrecognised keys raise TypeError.

        Raises:
            TypeError: Uncrecogized or missing keys

        Returns:
            LatinHypercubeEnsemble: A fully constructed ensemble LatinHypercubeEnsemble
            instance.
        """
        try:
            cfg = LatinHypercubeConfig(**config)
        except TypeError as e:
            raise TypeError(
                f"Invalid config key for LatinHypercubeEnsemble: {e}. "
                f"Valid keys: {[f.name for f in fields(LatinHypercubeConfig)]}"
            ) from e
        return cls(cfg)

    def create_samples(self) -> list[EnsembleMemberSample]:
        """Create samples from the list of parameters

        Returns:
            list[EnsembleMemberSample]: One EnsembleMemberSample per ensemble member,
            each containing one ParameterSample per parameter.
        """

        # build latin hypercube
        latin_hypercube = self.build_lh(self.prebuilt)

        # draw samples
        samples = []
        for i in range(self.n_samples):
            ensemble_member = EnsembleMemberSample(
                parameter_samples=[
                    ParameterSample(
                        group=group,
                        normalized_value=float(latin_hypercube[i, j]),
                    )
                    for j, group in enumerate(self.sampling_units)
                ]
            )
            samples.append(ensemble_member)

        return samples

    def create_ensemble_member(self, sample: EnsembleMemberSample) -> ParameterDataset:
        """Create one member of the ensemble

        Args:
            sample (EnsembleMemberSample): Parameter samples for this member.
                Contains one ParameterSample per parameter.
        Returns:
            ParameterDataset: one member of the ensemble with updated values from default
        """

        ds = self.default_ds.copy()
        for param_sample in sample:
            for param in param_sample.group.params:
                value = param.sample(param_sample.normalized_value, self.default_ds)
                param.set_value(
                    ds, self.default_ds, value, fixed_indices=self.fixed_indices
                )
        return ds

    def create_ensemble_key(self, samples: list[EnsembleMemberSample]) -> pd.DataFrame:
        """Create the ensemble key that goes with this ensemble

        Args:
            samples (list[EnsembleMemberSample]): One EnsembleMemberSample per ensemble
                member, each containing one ParameterSample per parameter.

        Returns:
            pd.DataFrame: output data frame that serves as ensemble key
        """

        param_dfs = []
        for i, sample in enumerate(samples):
            group_names = []
            sample_values = []
            for param_sample in sample:
                group_names.append(param_sample.group.name)
                sample_values.append(param_sample.normalized_value)
            df = pd.DataFrame({"parameter": group_names, "value": sample_values})
            df["ensemble"] = f"{self.file_prefix}_{_generate_suffix(i)}"
            param_dfs.append(df)
        param_df = pd.concat(param_dfs, ignore_index=True)
        return (
            param_df.pivot(index="ensemble", columns="parameter", values="value")
            .reset_index()
            .rename_axis(None, axis=1)
        )

    def build_lh(self, prebuilt: np.ndarray | None = None) -> np.ndarray:
        """Create a Latin Hypercube, or validate a pre-built one

        Args:
            prebuilt (np.ndarray | None, optional): Optional pre-built hypercube.
            Defaults to None.

        Raises:
            ValueError: Supplied pre-built Latin Hypercube dimensions do not match setup

        Returns:
            np.ndarray: output Latin Hypercube array
        """
        if self.num_sampling_units == 0:
            return np.empty((self.n_samples, 0))

        # validate pre-built LH
        if prebuilt is not None:
            if prebuilt.shape != (self.n_samples, self.num_sampling_units):
                raise ValueError(
                    f"Pre-built LH sample has shape {prebuilt.shape}, "
                    f"expected ({self.n_samples}, {self.num_sampling_units})."
                )
            return prebuilt

        # otherwise generate one
        return qmc.LatinHypercube(d=self.num_sampling_units).random(n=self.n_samples)


class OneAtATimeEnsemble(ParamEnsemble, ensemble_type="OAT"):
    """Concrete class for the One-at-a-time (OAT) ensemble class"""

    def __init__(
        self,
        config: OneAtATimeConfig,
    ):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: dict) -> OneAtATimeEnsemble:
        """Construct from a plain dict (ensemble_type already removed)

        Args:
            config (dict): Must contain all required OneAtATimeConfig fields.
            Unrecognised keys raise TypeError.

        Raises:
            TypeError: Uncrecogized or missing keys

        Returns:
            OneAtATimeEnsemble: A fully constructed ensemble OneAtATimeEnsemble
            instance.
        """
        try:
            cfg = OneAtATimeConfig(**config)
        except TypeError as e:
            raise TypeError(
                f"Invalid config key for OneAtATimeConfig: {e}. "
                f"Valid keys: {[f.name for f in fields(OneAtATimeConfig)]}"
            ) from e
        return cls(cfg)

    def create_samples(self) -> list[EnsembleMemberSample]:
        """Create samples from the list of parameters

        Returns:
            list[EnsembleMemberSample]: One EnsembleMemberSample per ensemble member,
            each containing one ParameterSample per parameter.
        """

        samples = []
        for group in self.sampling_units:
            samples.append(
                EnsembleMemberSample([ParameterSample(group, 0.0)])  # minimum
            )
            samples.append(
                EnsembleMemberSample([ParameterSample(group, 1.0)])  # maximum
            )
        return samples

    def create_ensemble_member(self, sample: EnsembleMemberSample) -> ParameterDataset:
        """Create one member of the ensemble

        Args:
            sample (EnsembleMemberSample): Parameter samples for this member.
                Contains one ParameterSample per parameter.
        Returns:
            ParameterDataset: one member of the ensemble with updated values from default
        """
        if len(sample) != 1:
            raise ValueError(
                f"OneAtATimeEnsemble expects exactly one ParameterSample per member "
                f"but got {len(sample)}."
            )

        param_sample = sample.parameter_samples[0]
        ds = self.default_ds.copy()
        for param in param_sample.group.params:
            value = param.sample(param_sample.normalized_value, self.default_ds)
            param.set_value(
                ds, self.default_ds, value, fixed_indices=self.fixed_indices
            )

        return ds

    def create_ensemble_key(self, samples: list[EnsembleMemberSample]) -> pd.DataFrame:
        """Create the ensemble key that goes with this ensemble

        Args:
            samples (list[EnsembleMemberSample]): One EnsembleMemberSample per ensemble
                member, each containing one ParameterSample per parameter.

        Returns:
            pd.DataFrame: output data frame that serves as ensemble key
        """
        group_names = []
        ensembles = []
        values = []
        for i, sample in enumerate(samples):
            if len(sample) != 1:
                raise ValueError(
                    f"OneAtATimeEnsemble expects exactly one ParameterSample per member "
                    f"but got {len(sample)}."
                )

            param_sample = sample.parameter_samples[0]
            group_names.append(param_sample.group.name)
            values.append(param_sample.normalized_value)
            ensembles.append(f"{self.file_prefix}_{_generate_suffix(i)}")

        param_df = pd.DataFrame(
            {
                "ensemble": ensembles,
                "parameter": group_names,
                "normalized_value": values,
            }
        )

        param_df["direction"] = param_df.normalized_value.map(
            {0.0: "minimum", 1.0: "maximum"}
        )
        nan_mask = param_df["direction"].isna()
        if nan_mask.any():
            raise ValueError(
                f"OneAtATimeEnsemble expects only 0.0 or 1.0 normalized values but found "
                f"{param_df[nan_mask]}"
                f"In the ensemble key"
            )
        return param_df.drop(columns=["normalized_value"])


def _generate_suffix(ensemble_number: int, pad_length: int = 3) -> str:
    """Generate a suffix for an ensemble member

    Args:
        ensemble_number (int): ensemble number
        pad_length (int, optional): pad length. Defaults to 3.

    Returns:
        str: output string
    """
    return str(ensemble_number).zfill(pad_length)


def _write_ensemble_list(out_dir: Path, file_prefix: str, ensembles: list[str]):
    """Writes out a list of ensemble members to supply to the run_ens script

    Args:
        out_dir (Path): output directory to write file to
        file_prefix (str): ensemble list file prefix
        ensembles (list[str]): list of ensembles
    """
    with open(out_dir / f"{file_prefix}.txt", "w", encoding="utf-8") as f:
        for ens in ensembles:
            f.write(f"{ens}\n")


def _validate_fixed_indices(
    fixed_indices: dict[str, list[int]],
    default_ds: ParameterDataset,
) -> None:
    """Check to make sure supplied fixed_indices dict is compatible with input
    default parameter dataset

    Args:
        fixed_indices (dict[str, list[int]]):  Mapping of dimension
            name to 0-based indices to hold at default.
        default_ds (ParameterDataset): Default parameter dataset.

    Raises:
        ValueError: fixed_indices dimension not found in dataset
        ValueError: fixed_indices has out-of-range indices
    """
    for dim, indices in fixed_indices.items():
        if dim not in default_ds.sizes:
            raise ValueError(
                f"fixed_indices dimension '{dim}' not found in default dataset. "
                f"Available dimensions: {sorted(default_ds.sizes)}"
            )
        dim_size = default_ds.sizes[dim]
        bad = [i for i in indices if i < 0 or i >= dim_size]
        if bad:
            raise ValueError(
                f"fixed_indices['{dim}'] contains out-of-range indices {bad}. "
                f"Dimension '{dim}' has size {dim_size} (valid: 0–{dim_size - 1})."
            )


def _expand_one_param(
    param: Parameter,
    fixed_indices: dict[str, list[int]],
    full_index_map: dict[str, list[int]],
) -> list[Parameter]:

    expand_dim = param.spec.expand_dim
    if expand_dim not in full_index_map:
        raise ValueError(
            f"Parameter '{param.spec.name}': expand_dim '{expand_dim}' "
            f"not found in default_ds. Available: {sorted(full_index_map)}"
        )
    fixed_for_dim = fixed_indices.get(expand_dim, [])
    active = [i for i in full_index_map[expand_dim] if i not in fixed_for_dim]
    return [param.for_index(expand_dim, idx) for idx in active]
