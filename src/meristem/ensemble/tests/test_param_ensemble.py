"""Tests for ParamEnsemble"""

from pathlib import Path

import pytest
import numpy as np
import pandas as pd

from param_ens_gen.param_ensemble import (
    ParamEnsemble,
    LatinHypercubeEnsemble,
    OneAtATimeEnsemble,
    EnsembleMemberSample,
    ParameterSample,
    ParamGroup,
)

from param_ens_gen.ensemble_config import LatinHypercubeConfig, OneAtATimeConfig
from param_ens_gen.parameter import DimIndex, Parameter
from param_ens_gen.parameter_dataset import ParameterDataset


def test_from_dict_missing_ensemble_type(
    ensemble_param_dir, default_param_file, tmp_path, posterior_config_file
):
    """Test that from_dict raises if the ensemble dictionary is missing the ensemble type"""
    with pytest.raises(ValueError, match="ensemble_type.*required"):
        ParamEnsemble.from_dict(
            {
                "param_dir": str(ensemble_param_dir),
                "ensemble_dir": str(tmp_path / "ensemble"),
                "file_prefix": "test",
                "default_param_file": str(default_param_file),
                "posterior_sources": posterior_config_file,
            }
        )


def test_from_dict_unknown_ensemble_type(
    ensemble_param_dir, default_param_file, tmp_path, posterior_config_file
):
    """Test that from_dict raises if the ensemble dictionary has an unknown ensemble type"""
    with pytest.raises(ValueError, match="Unknown ensemble_type"):
        ParamEnsemble.from_dict(
            {
                "ensemble_type": "BadType",
                "param_dir": str(ensemble_param_dir),
                "ensemble_dir": str(tmp_path / "ensemble"),
                "file_prefix": "test",
                "default_param_file": str(default_param_file),
                "posterior_sources": posterior_config_file,
            }
        )


def test_from_dict_bad_key(
    ensemble_param_dir, default_param_file, tmp_path, posterior_config_file
):
    """Test that from_dict raises if the ensemble dictionary has an invalid key"""
    with pytest.raises(TypeError, match="Invalid config key"):
        ParamEnsemble.from_dict(
            {
                "ensemble_type": "LatinHypercube",
                "param_dir": str(ensemble_param_dir),
                "ensemble_dir": str(tmp_path / "ensemble"),
                "file_prefix": "test",
                "default_param_file": str(default_param_file),
                "not_a_real_key": 42,
                "posterior_sources": posterior_config_file,
            }
        )


def test_from_dict_valid_latin_hypercube(
    ensemble_param_dir, default_param_file, tmp_path, posterior_config_file
):
    """Test that from_dict cab create a LatinHypercubeEnsemble class from a valid dict"""
    ensemble = ParamEnsemble.from_dict(
        {
            "ensemble_type": "LatinHypercube",
            "param_dir": str(ensemble_param_dir),
            "ensemble_dir": str(tmp_path / "ensemble"),
            "file_prefix": "test",
            "default_param_file": str(default_param_file),
            "ensemble_members": 5,
            "posterior_sources": str(posterior_config_file),
        }
    )
    assert isinstance(ensemble, LatinHypercubeEnsemble)


def test_from_dict_valid_oat(
    ensemble_param_dir, default_param_file, tmp_path, posterior_config_file
):
    """Test that from_dict cab create a OneAtATime class from a valid dict"""
    ensemble = ParamEnsemble.from_dict(
        {
            "ensemble_type": "OAT",
            "param_dir": str(ensemble_param_dir),
            "ensemble_dir": str(tmp_path / "ensemble"),
            "file_prefix": "test",
            "default_param_file": str(default_param_file),
            "posterior_sources": posterior_config_file,
        }
    )
    assert isinstance(ensemble, OneAtATimeEnsemble)


def test_from_dict_invalid_oat(
    ensemble_param_dir, default_param_file, tmp_path, posterior_config_file
):
    """Test that from_dict raises if you have ensemble_members in a OAT dict"""
    with pytest.raises(TypeError, match="Invalid config key"):
        ParamEnsemble.from_dict(
            {
                "ensemble_type": "OAT",
                "param_dir": str(ensemble_param_dir),
                "ensemble_dir": str(tmp_path / "ensemble"),
                "file_prefix": "test",
                "default_param_file": str(default_param_file),
                "ensemble_members": 5,
                "posterior_sources": posterior_config_file,
            }
        )


def test_from_dict_valid_latin_hypercube_creates_ensemble_dir(
    ensemble_param_dir, default_param_file, tmp_path, posterior_config_file
):
    """Test that from_dict creates the ensemble_dir"""
    ensemble = ParamEnsemble.from_dict(
        {
            "ensemble_type": "LatinHypercube",
            "param_dir": str(ensemble_param_dir),
            "ensemble_dir": str(tmp_path / "ensemble"),
            "file_prefix": "test",
            "default_param_file": str(default_param_file),
            "ensemble_members": 5,
            "posterior_sources": posterior_config_file,
        }
    )
    assert isinstance(ensemble.ensemble_dir, Path)
    assert ensemble.ensemble_dir.exists()


def test_from_dict_valid_latin_hypercube_correctly_sets_attributes(
    ensemble_param_dir, default_param_file, tmp_path, posterior_config_file
):
    """Test that from_dict correctly sets attributes"""
    ensemble = ParamEnsemble.from_dict(
        {
            "ensemble_type": "LatinHypercube",
            "param_dir": str(ensemble_param_dir),
            "ensemble_dir": str(tmp_path / "ensemble"),
            "file_prefix": "test",
            "default_param_file": str(default_param_file),
            "ensemble_members": 5,
            "posterior_sources": posterior_config_file,
            "fixed_indices": {"fates_pft": [1]},
        }
    )
    assert ensemble.file_prefix == "test"
    assert isinstance(ensemble.default_ds, ParameterDataset)
    assert ensemble.fixed_indices == {"fates_pft": [1]}


def test_from_dict_valid_latin_hypercube_fixed_indices_empty_if_unset(
    ensemble_param_dir, default_param_file, tmp_path, posterior_config_file
):
    """Test that from_dict sets fixed_indices to an empty dict if it is unset"""
    ensemble = ParamEnsemble.from_dict(
        {
            "ensemble_type": "LatinHypercube",
            "param_dir": str(ensemble_param_dir),
            "ensemble_dir": str(tmp_path / "ensemble"),
            "file_prefix": "test",
            "default_param_file": str(default_param_file),
            "ensemble_members": 5,
            "posterior_sources": posterior_config_file,
        }
    )
    assert ensemble.fixed_indices == {}


def test_param_list_unknown_parameter_raises(
    ensemble_param_dir, default_param_file, tmp_path, posterior_config_file
):
    """Test that from_dict raises if there is a parameter in param_list not in main.csv"""
    config = LatinHypercubeConfig(
        param_dir=ensemble_param_dir,
        ensemble_dir=tmp_path / "ensemble",
        file_prefix="test",
        default_param_file=default_param_file,
        posterior_sources=posterior_config_file,
        param_list=["fates_leaf_slatop", "not_a_real_param"],
    )
    with pytest.raises(ValueError, match="not found in main.csv"):
        LatinHypercubeEnsemble(config)


def test_missing_netcdf_default_param_file_raises(
    ensemble_param_dir, tmp_path, posterior_config_file
):
    """Test that from_dict raises if it can't find the default parameter file"""
    config = LatinHypercubeConfig(
        param_dir=ensemble_param_dir,
        ensemble_dir=tmp_path / "ensemble",
        file_prefix="test",
        posterior_sources=posterior_config_file,
        default_param_file=tmp_path / "nonexistent.nc",
    )
    with pytest.raises(FileNotFoundError, match="does not exist"):
        LatinHypercubeEnsemble(config)


def test_missing_posterior_sources_file_raises(
    ensemble_param_dir, tmp_path, default_param_file
):
    """Test that from_dict raises if it can't find the posterior sources file"""
    config = LatinHypercubeConfig(
        param_dir=ensemble_param_dir,
        ensemble_dir=tmp_path / "ensemble",
        file_prefix="test",
        posterior_sources=tmp_path / "nonexistant.yaml",
        default_param_file=default_param_file,
    )
    with pytest.raises(FileNotFoundError, match="does not exist"):
        LatinHypercubeEnsemble(config)


def test_from_dict_params_correct(lh_ensemble):
    """Test that the ensemble correctly sets the list of parameters"""
    assert len(lh_ensemble.params) == 6


def test_from_dict_param_list_subsets_correctly(
    ensemble_param_dir, default_param_file, tmp_path, posterior_config_file
):
    """Test that from_dict correctly subsets the parameters with param_list"""
    param_list = ["fates_leaf_slatop", "fates_leaf_vcmax25top"]
    ensemble = ParamEnsemble.from_dict(
        {
            "ensemble_type": "LatinHypercube",
            "param_dir": str(ensemble_param_dir),
            "ensemble_dir": str(tmp_path / "ensemble"),
            "file_prefix": "test",
            "default_param_file": str(default_param_file),
            "ensemble_members": 5,
            "posterior_sources": posterior_config_file,
            "param_list": param_list,
        }
    )
    names = [p.spec.name for p in ensemble.params]
    assert len(ensemble.params) == 2
    assert set(param_list) == set(names)


def test_ensemble_params_types(lh_ensemble):
    """Test that all params are Parameter instances."""
    assert all(isinstance(p, Parameter) for p in lh_ensemble.params)


def test_ensemble_params_names(lh_ensemble, tmp_path):
    """Test that parameter names match what's in main.csv."""
    param_data = pd.read_csv(tmp_path / "main.csv")
    param_names = param_data.parameter_name.unique()
    names = [p.spec.name for p in lh_ensemble.params]
    assert set(param_names) == set(names)


def test_sort_params_root_before_dependent(
    ensemble_param_dir, default_param_file, posterior_config_file, tmp_path
):
    """Test that sort_params ensures root parameter is written before scale_from_root dependent."""
    config = LatinHypercubeConfig(
        param_dir=ensemble_param_dir,
        ensemble_dir=tmp_path / "ensemble",
        file_prefix="test",
        posterior_sources=posterior_config_file,
        default_param_file=default_param_file,
    )
    ensemble = LatinHypercubeEnsemble(config)
    names = [p.spec.name for p in ensemble.params]
    assert names.index("fates_nonhydro_smpso") < names.index("smpsc_delta")


def test_fixed_indices_invalid_dimension_raises(
    ensemble_param_dir, default_param_file, posterior_config_file, tmp_path
):
    """Test that fixed_indices with an unknown dimension raises ValueError."""
    config = LatinHypercubeConfig(
        param_dir=ensemble_param_dir,
        ensemble_dir=tmp_path / "ensemble",
        file_prefix="test",
        posterior_sources=posterior_config_file,
        default_param_file=default_param_file,
        fixed_indices={"not_a_dim": [0, 1]},
    )
    with pytest.raises(ValueError, match="not found in default dataset"):
        LatinHypercubeEnsemble(config)


def test_fixed_indices_out_of_range_raises(
    ensemble_param_dir, default_param_file, posterior_config_file, tmp_path
):
    """Test that fixed_indices with out-of-range indices raises ValueError."""
    config = LatinHypercubeConfig(
        param_dir=ensemble_param_dir,
        ensemble_dir=tmp_path / "ensemble",
        file_prefix="test",
        posterior_sources=posterior_config_file,
        default_param_file=default_param_file,
        fixed_indices={"fates_pft": [99]},  # default_ds has N_PFTS=3
    )
    with pytest.raises(ValueError, match="out-of-range indices"):
        LatinHypercubeEnsemble(config)


def test_fixed_indices_valid(
    ensemble_param_dir, default_param_file, posterior_config_file, tmp_path
):
    """Test that valid fixed_indices constructs successfully."""
    config = LatinHypercubeConfig(
        param_dir=ensemble_param_dir,
        ensemble_dir=tmp_path / "ensemble",
        file_prefix="test",
        posterior_sources=posterior_config_file,
        default_param_file=default_param_file,
        fixed_indices={"fates_pft": [0, 1]},
    )
    ensemble = LatinHypercubeEnsemble(config)
    assert ensemble.fixed_indices == {"fates_pft": [0, 1]}


def test_fixed_indices_negative_index_raises(
    ensemble_param_dir, default_param_file, posterior_config_file, tmp_path
):
    """Test that fixed_indices with out of range indices raises"""
    config = LatinHypercubeConfig(
        param_dir=ensemble_param_dir,
        ensemble_dir=tmp_path / "ensemble",
        file_prefix="test",
        posterior_sources=posterior_config_file,
        default_param_file=default_param_file,
        fixed_indices={"fates_pft": [-1]},
    )
    with pytest.raises(ValueError, match="out-of-range indices"):
        LatinHypercubeEnsemble(config)


def test_grouped_params_constructs_successfully(
    grouped_param_dir, default_param_file, tmp_path
):
    """ParamEnsemble constructs without error when groups are valid."""
    config = LatinHypercubeConfig(
        param_dir=grouped_param_dir,
        ensemble_dir=tmp_path / "ensemble",
        file_prefix="test",
        default_param_file=default_param_file,
        ensemble_members=5,
    )
    ensemble = LatinHypercubeEnsemble(config)
    assert ensemble is not None


def test_scale_from_root_different_group_raises(
    scale_from_root_different_group_dir, default_param_file, tmp_path
):
    """ParamEnsemble raises ValueError when scale_from_root and root are in different groups."""
    config = LatinHypercubeConfig(
        param_dir=scale_from_root_different_group_dir,
        ensemble_dir=tmp_path / "ensemble",
        file_prefix="test",
        default_param_file=default_param_file,
        ensemble_members=5,
    )
    with pytest.raises(ValueError, match="same group"):
        LatinHypercubeEnsemble(config)


def test_scale_from_root_root_not_varied_is_valid(tmp_path, default_param_file):
    """ParamEnsemble constructs when scale_from_root root is not being varied."""
    pd.DataFrame(
        [
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

    config = LatinHypercubeConfig(
        param_dir=tmp_path,
        ensemble_dir=tmp_path / "ensemble",
        file_prefix="test",
        default_param_file=default_param_file,
        ensemble_members=5,
    )
    ensemble = LatinHypercubeEnsemble(config)
    assert ensemble is not None


def test_mixed_expand_dim_group_raises(
    mixed_expand_dim_group_dir, default_param_file, tmp_path
):
    """ParamEnsemble raises ValueError when a group has mixed expand_dim values."""
    config = LatinHypercubeConfig(
        param_dir=mixed_expand_dim_group_dir,
        ensemble_dir=tmp_path / "ensemble",
        file_prefix="test",
        default_param_file=default_param_file,
        ensemble_members=5,
    )
    with pytest.raises(ValueError, match="mixed expand_dim"):
        LatinHypercubeEnsemble(config)


def test_build_lh_shape(lh_ensemble):
    """Test that build_lh builds a Latin Hypercube with the correct shape"""
    lh = lh_ensemble.build_lh()
    assert lh.shape == (5, lh_ensemble.num_sampling_units)
    assert np.all(lh >= 0.0) and np.all(lh <= 1.0)


def test_build_lh_prebuilt(lh_ensemble):
    """Test that build_lh with a correct pre-built LH returns that pre-built array"""
    prebuilt = np.random.default_rng(0).random((5, lh_ensemble.num_sampling_units))
    lh = lh_ensemble.build_lh(prebuilt=prebuilt)
    np.testing.assert_array_equal(lh, prebuilt)


def test_build_lh_prebuilt_wrong_samples(lh_ensemble):
    """Test that build_lh given an incorrect (wrong num samples) pre-built LH raises"""
    bad = np.random.default_rng(0).random((10, lh_ensemble.num_sampling_units))
    with pytest.raises(ValueError, match="shape"):
        lh_ensemble.build_lh(prebuilt=bad)


def test_build_lh_prebuilt_wrong_params(lh_ensemble):
    """Test that build_lh given an incorrect (wrong num parameters) pre-built LH raises"""
    bad = np.random.default_rng(0).random((5, 10))
    with pytest.raises(ValueError, match="shape"):
        lh_ensemble.build_lh(prebuilt=bad)


def test_build_lh_zero_params(lh_ensemble):
    """Test that build_lh with zero params returns something"""
    lh_ensemble.num_sampling_units = 0
    lh = lh_ensemble.build_lh()
    assert lh.shape == (5, 0)


def test_lh_create_samples_length(lh_ensemble):
    """Test that create_samples returns the correct number of samples"""
    samples = lh_ensemble.create_samples()
    assert len(samples) == 5


def test_lh_create_samples_structure(lh_ensemble):
    """Test that create_samples returns the correct structure"""
    samples = lh_ensemble.create_samples()
    for sample in samples:
        assert len(sample) == lh_ensemble.num_sampling_units
        for ps in sample:
            assert 0.0 <= ps.normalized_value <= 1.0


def test_oat_create_samples_length(oat_ensemble):
    """Test that OAT produces 2 samples per parameter (min and max)."""
    samples = oat_ensemble.create_samples()
    assert len(samples) == 2 * oat_ensemble.num_sampling_units


def test_oat_create_samples_values(oat_ensemble):
    """Test that OAT samples are always 0.0 or 1.0."""
    samples = oat_ensemble.create_samples()
    values = [s.parameter_samples[0].normalized_value for s in samples]
    assert set(values) == {0.0, 1.0}


def test_lh_create_ensemble_member(lh_ensemble):
    """Test that create_ensemble_member works for a LH ensemble"""
    samples = lh_ensemble.create_samples()
    ds = lh_ensemble.create_ensemble_member(samples[0])
    assert "fates_leaf_slatop" in ds


def test_oat_create_ensemble_member(oat_ensemble):
    """Test that create_ensemble_member works for a OAT ensemble"""
    samples = oat_ensemble.create_samples()
    ds = oat_ensemble.create_ensemble_member(samples[0])
    assert "fates_leaf_slatop" in ds


def test_oat_create_ensemble_member_wrong_length(oat_ensemble, lh_ensemble):
    """OAT raises if sample has more than one ParameterSample."""
    lh_samples = lh_ensemble.create_samples()
    with pytest.raises(ValueError, match="exactly one"):
        oat_ensemble.create_ensemble_member(lh_samples[0])


def test_lh_create_ensemble_key(lh_ensemble):
    """Test that create_ensemble_key works for an LH ensemble"""
    samples = lh_ensemble.create_samples()
    key = lh_ensemble.create_ensemble_key(samples)
    assert "ensemble" in key.columns
    assert len(key) == 5


def test_oat_create_ensemble_key(oat_ensemble):
    """Test that create_ensemble_key works for an OAT ensemble"""
    samples = oat_ensemble.create_samples()
    key = oat_ensemble.create_ensemble_key(samples)
    assert "ensemble" in key.columns
    assert "direction" in key.columns
    assert set(key["direction"]) == {"minimum", "maximum"}


def test_oat_create_ensemble_key_wrong_length_raises(oat_ensemble, lh_ensemble):
    """Test that create_ensemble_key raises if any sample has more than one ParameterSample."""
    lh_samples = lh_ensemble.create_samples()
    with pytest.raises(ValueError, match="exactly one"):
        oat_ensemble.create_ensemble_key(lh_samples)


def test_oat_create_ensemble_key_bad_direction_raises(oat_ensemble):
    """Test that create_ensemble_key raises if normalized_value is not 0.0 or 1.0."""
    param = oat_ensemble.params[0]
    bad_sample = EnsembleMemberSample(
        [ParameterSample(ParamGroup(name=param.spec.name, params=[param]), 0.5)]
    )
    with pytest.raises(ValueError, match="expects only 0.0 or 1.0"):
        oat_ensemble.create_ensemble_key([bad_sample])


def test_create_ensemble_writes_files(lh_ensemble):
    """Test that create_ensemble writes all the files we require"""
    lh_ensemble.create_ensemble()
    ext = lh_ensemble.default_ds.file_extension
    output_files = list(lh_ensemble.ensemble_dir.glob(f"test_*{ext}"))
    assert len(output_files) == 5
    key_file = lh_ensemble.ensemble_dir / "test_key.csv"
    assert key_file.exists()
    list_file = lh_ensemble.ensemble_dir / "test.txt"
    assert list_file.exists()


def test_expand_params_count(lh_expand_ensemble):
    """Test that expandable param over N_PFTS=3 produces 3 clones; scalar passes through."""
    # 3 expanded copies of fates_leaf_slatop + 1 scalar = 4 total
    assert len(lh_expand_ensemble.params) == 4


def test_expand_params_names(lh_expand_ensemble):
    """Test that expanded params are named with index suffix; non-expanded is unchanged."""
    names = [p.spec.name for p in lh_expand_ensemble.params]
    assert "fates_leaf_slatop_0" in names
    assert "fates_leaf_slatop_1" in names
    assert "fates_leaf_slatop_2" in names
    assert "fates_canopy_closure_thresh" in names


def test_expand_params_active_index_set(lh_expand_ensemble):
    """Test that expanded params have active_index set to the correct dim and index."""
    expanded = [p for p in lh_expand_ensemble.params if p.active_index is not None]
    assert len(expanded) == 3
    indices = {p.active_index for p in expanded}
    assert indices == {
        DimIndex("fates_pft", 0),
        DimIndex("fates_pft", 1),
        DimIndex("fates_pft", 2),
    }


def test_expand_params_non_expanded_has_no_active_index(lh_expand_ensemble):
    """Test that non-expanded params have active_index=None."""
    scalar = next(
        p
        for p in lh_expand_ensemble.params
        if p.spec.name == "fates_canopy_closure_thresh"
    )
    assert scalar.active_index is None


def test_expand_params_spec_not_shared(lh_expand_ensemble):
    """Test that each expanded clone has its own spec instance."""
    expanded = [p for p in lh_expand_ensemble.params if p.active_index is not None]
    specs = [id(p.spec) for p in expanded]
    assert len(specs) == len(set(specs))


def test_expand_params_fixed_indices_excluded(
    expand_param_dir, default_param_file, tmp_path
):
    """Test that indices listed in fixed_indices are not expanded into."""
    config = LatinHypercubeConfig(
        param_dir=expand_param_dir,
        ensemble_dir=tmp_path / "ensemble",
        file_prefix="test",
        default_param_file=default_param_file,
        ensemble_members=5,
        fixed_indices={"fates_pft": [2]},
    )
    ensemble = LatinHypercubeEnsemble(config)
    expanded = [p for p in ensemble.params if p.active_index is not None]
    active_indices = [p.active_index.index for p in expanded]
    assert 2 not in active_indices
    assert len(expanded) == 2


def test_expand_params_unknown_expand_dim_raises(tmp_path, default_param_file):
    """Test that expand_dim referencing a dimension not in default_ds raises ValueError."""
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
                "strategy": "uniform",
                "param_min": "0.005",
                "param_max": "0.05",
                "slice_dim": None,
                "slice_index": None,
                "root_param": None,
                "base_params": "",
                "expand_dim": "not_a_real_dim",
            }
        ]
    ).to_csv(tmp_path / "main.csv", index=False)

    config = LatinHypercubeConfig(
        param_dir=tmp_path,
        ensemble_dir=tmp_path / "ensemble",
        file_prefix="test",
        default_param_file=default_param_file,
        ensemble_members=5,
    )
    with pytest.raises(ValueError, match="not found in default_ds"):
        LatinHypercubeEnsemble(config)


def test_grouped_params_num_sampling_units(
    grouped_param_dir, default_param_file, tmp_path
):
    """num_sampling_units is less than num_params when groups exist."""
    config = LatinHypercubeConfig(
        param_dir=grouped_param_dir,
        ensemble_dir=tmp_path / "ensemble",
        file_prefix="test",
        default_param_file=default_param_file,
        ensemble_members=5,
    )
    ensemble = LatinHypercubeEnsemble(config)
    # 2 params in "photosynthesis" group + 6 expanded params in "water" group + 1 ungrouped
    # = 9 params, 3 sampling units
    assert ensemble.num_params == 9
    assert ensemble.num_sampling_units == 3


def test_grouped_params_share_normalized_value(
    grouped_param_dir, default_param_file, tmp_path
):
    """Params in the same group get the same normalized value in create_samples."""
    config = LatinHypercubeConfig(
        param_dir=grouped_param_dir,
        ensemble_dir=tmp_path / "ensemble",
        file_prefix="test",
        default_param_file=default_param_file,
        ensemble_members=5,
    )
    ensemble = LatinHypercubeEnsemble(config)
    samples = ensemble.create_samples()
    for sample in samples:
        photo_sample = next(ps for ps in sample if ps.group.name == "photosynthesis")
        # all params in the group share the same normalized value
        assert len(photo_sample.group.params) == 2


def test_grouped_oat_produces_correct_number_of_members(
    grouped_param_dir, default_param_file, tmp_path
):
    """OAT produces 2 members per sampling unit, not per parameter."""

    config = OneAtATimeConfig(
        param_dir=grouped_param_dir,
        ensemble_dir=tmp_path / "ensemble",
        file_prefix="test",
        default_param_file=default_param_file,
    )
    ensemble = OneAtATimeEnsemble(config)
    samples = ensemble.create_samples()
    assert len(samples) == 2 * ensemble.num_sampling_units
    assert len(samples) < 2 * ensemble.num_params


def test_grouped_ensemble_key_uses_group_names(
    grouped_param_dir, default_param_file, tmp_path
):
    """Ensemble key columns use group names, not individual parameter names."""
    config = LatinHypercubeConfig(
        param_dir=grouped_param_dir,
        ensemble_dir=tmp_path / "ensemble",
        file_prefix="test",
        default_param_file=default_param_file,
        ensemble_members=5,
    )
    ensemble = LatinHypercubeEnsemble(config)
    samples = ensemble.create_samples()
    key = ensemble.create_ensemble_key(samples)
    assert "photosynthesis" in key.columns
    assert "fates_canopy_closure_thresh" in key.columns
    # individual param names should not appear as columns
    assert "fates_leaf_slatop" not in key.columns
    assert "fates_nonhydro_smpso" not in key.columns


def test_grouped_ensemble_member_writes_all_params(
    grouped_param_dir, default_param_file, tmp_path
):
    """create_ensemble_member writes values for all params in a group."""
    config = LatinHypercubeConfig(
        param_dir=grouped_param_dir,
        ensemble_dir=tmp_path / "ensemble",
        file_prefix="test",
        default_param_file=default_param_file,
        ensemble_members=5,
    )
    ensemble = LatinHypercubeEnsemble(config)
    samples = ensemble.create_samples()
    ds = ensemble.create_ensemble_member(samples[0])
    assert "fates_leaf_slatop" in ds
    assert "fates_canopy_closure_thresh" in ds
