"""Tests for param_ens_gen.distribution_stat.

Scope
-----
- DistributionStat.parse: all accepted formats, all error cases
- DistributionStat.from_row: fixed, percent, pft, and error cases
- FixedStat.resolve
- PercentStat.resolve: min, max, sd, scalar and array default values
- PFTStat.resolve and PFTStat.from_sheet
"""

import numpy as np
import pandas as pd
import pytest

from param_ens_gen.distribution_stat import (
    DistributionStat,
    FixedStat,
    PercentStat,
    PFTStat,
)


def _row(param_min="0.005", param_max="0.05", **kwargs) -> pd.Series:
    """Build a minimal valid row for parse() calls."""
    defaults = {
        "parameter_name": "fates_leaf_slatop",
        "param_min": param_min,
        "param_max": param_max,
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


# ===========================================================================
# DistributionStat.parse: successful construction
# ===========================================================================


def test_parse_plain_number_returns_fixed_stat():
    """parse() returns a FixedStat for a plain numeric string."""
    stat = DistributionStat.parse(_row(param_max="0.9"), stat_type="max")
    assert isinstance(stat, FixedStat)
    assert stat.value == pytest.approx(0.9)


def test_parse_negative_number_returns_fixed_stat():
    """parse() correctly handles negative numeric values."""
    stat = DistributionStat.parse(_row(param_min="-50000"), stat_type="min")
    assert isinstance(stat, FixedStat)
    assert stat.value == pytest.approx(-50000.0)


def test_parse_integer_string_returns_fixed_stat():
    """parse() accepts an integer string."""
    stat = DistributionStat.parse(_row(param_min="1"), stat_type="min")
    assert isinstance(stat, FixedStat)
    assert stat.value == pytest.approx(1.0)


def test_parse_percent_word_returns_percent_stat():
    """parse() returns a PercentStat for '50percent' syntax."""
    stat = DistributionStat.parse(_row(param_min="50percent"), stat_type="min")
    assert isinstance(stat, PercentStat)
    assert stat.percent == pytest.approx(50.0)


def test_parse_percent_symbol_returns_percent_stat():
    """parse() returns a PercentStat for '50%' syntax."""
    stat = DistributionStat.parse(_row(param_max="50%"), stat_type="max")
    assert isinstance(stat, PercentStat)
    assert stat.percent == pytest.approx(50.0)


def test_parse_percent_with_spaces_returns_percent_stat():
    """parse() returns a PercentStat for '50 percent' syntax."""
    stat = DistributionStat.parse(_row(param_min="50 percent"), stat_type="min")
    assert isinstance(stat, PercentStat)
    assert stat.percent == pytest.approx(50.0)


def test_parse_percent_stat_stores_stat_type():
    """parse() stores the stat_type on the returned PercentStat."""
    stat = DistributionStat.parse(_row(param_min="30percent"), stat_type="min")
    assert stat.stat_type == "min"


def test_parse_percent_stat_stores_stat_type_sd():
    """parse() stores the stat_type 'sd' on the returned PercentStat."""
    stat = DistributionStat.parse(_row(param_sd="30percent"), stat_type="sd")
    assert stat.stat_type == "sd"


def test_parse_pft_returns_pft_stat(pft_row, pft_sheet):
    """parse() returns a PFTStat when param_min is 'pft' and sheet is supplied."""
    stat = DistributionStat.parse(pft_row, stat_type="min", pft_sheet=pft_sheet)
    assert isinstance(stat, PFTStat)


# ===========================================================================
# DistributionStat.parse: error cases
# ===========================================================================


def test_parse_unknown_stat_type_raises():
    """parse() raises ValueError for an unrecognised stat_type."""
    with pytest.raises(ValueError, match="Unknown stat_type"):
        DistributionStat.parse(_row(), stat_type="unknown")


def test_parse_empty_cell_raises():
    """parse() raises ValueError when the cell is empty/NaN."""
    with pytest.raises(ValueError, match="empty"):
        DistributionStat.parse(_row(param_min=float("nan")), stat_type="min")


def test_parse_pft_without_sheet_raises(pft_row):
    """parse() raises ValueError when param is 'pft' but no sheet is supplied."""
    with pytest.raises(ValueError, match="pft_sheet"):
        DistributionStat.parse(pft_row, stat_type="min", pft_sheet=None)


def test_parse_percent_for_mean_raises():
    """parse() raises ValueError for percent syntax with stat_type='mean'."""
    with pytest.raises(ValueError, match="mean"):
        DistributionStat.parse(_row(param_mean="50percent"), stat_type="mean")


def test_parse_zero_percent_raises():
    """parse() raises ValueError for 0 percent (would make min == max == default)."""
    with pytest.raises(ValueError, match="0"):
        DistributionStat.parse(_row(param_min="0percent"), stat_type="min")


def test_parse_negative_percent_raises():
    """parse() raises ValueError for negative percentages."""
    with pytest.raises(ValueError, match="-50"):
        DistributionStat.parse(_row(param_min="-50percent"), stat_type="min")


def test_parse_non_numeric_string_raises():
    """parse() raises ValueError for an unrecognisable string."""
    with pytest.raises(ValueError, match="Could not parse"):
        DistributionStat.parse(_row(param_min="not_a_number"), stat_type="min")


def test_parse_malformed_percent_raises():
    """parse() raises ValueError when percent prefix is not a number."""
    with pytest.raises(ValueError, match="Could not parse percent"):
        DistributionStat.parse(_row(param_min="abcpercent"), stat_type="min")


# ===========================================================================
# FixedStat.resolve
# ===========================================================================


def test_fixed_stat_resolve_returns_value():
    """FixedStat.resolve() returns its stored value regardless of default_value."""
    stat = FixedStat(value=0.9)
    assert stat.resolve() == pytest.approx(0.9)


def test_fixed_stat_resolve_ignores_default_value():
    """FixedStat.resolve() ignores any default_value passed to it."""
    stat = FixedStat(value=0.9)
    assert stat.resolve(default_value=np.array([1.0, 2.0, 3.0])) == pytest.approx(0.9)


# ===========================================================================
# PercentStat.resolve
# ===========================================================================


def test_percent_stat_resolve_min_scalar():
    """PercentStat.resolve() for min subtracts the delta from the default."""
    stat = PercentStat(percent=50.0, stat_type="min")
    result = stat.resolve(default_value=1.0)
    assert result == pytest.approx(0.5)


def test_percent_stat_resolve_max_scalar():
    """PercentStat.resolve() for max adds the delta to the default."""
    stat = PercentStat(percent=50.0, stat_type="max")
    result = stat.resolve(default_value=1.0)
    assert result == pytest.approx(1.5)


def test_percent_stat_resolve_sd_scalar():
    """PercentStat.resolve() for sd adds the delta to the default."""
    stat = PercentStat(percent=25.0, stat_type="sd")
    result = stat.resolve(default_value=2.0)
    assert result == pytest.approx(2.5)


def test_percent_stat_resolve_min_array():
    """PercentStat.resolve() broadcasts correctly over a numpy array."""
    stat = PercentStat(percent=50.0, stat_type="min")
    default = np.array([1.0, 2.0, 4.0])
    result = stat.resolve(default_value=default)
    np.testing.assert_allclose(result, [0.5, 1.0, 2.0])


def test_percent_stat_resolve_uses_abs_for_negative_default():
    """PercentStat.resolve() uses abs(default) so the delta is always positive."""
    stat = PercentStat(percent=50.0, stat_type="min")
    # min of -100 should be -100 - 50 = -150, not -100 + 50 = -50
    result = stat.resolve(default_value=-100.0)
    assert result == pytest.approx(-150.0)


def test_percent_stat_resolve_none_default_raises():
    """PercentStat.resolve() raises ValueError when default_value is None."""
    stat = PercentStat(percent=50.0, stat_type="min")
    with pytest.raises(ValueError, match="default_value"):
        stat.resolve(default_value=None)


def test_percent_stat_resolve_unknown_stat_raises():
    """PercentStat.resolve() raises AssertionError with an unknown."""
    stat = PercentStat(percent=50.0, stat_type="min")
    stat.stat_type = "unknown_stat"
    with pytest.raises(AssertionError, match="unknown_stat"):
        stat.resolve(default_value=5.0)


def test_percent_stat_resolve_mean_raises():
    """PercentStat.resolve() raises AssertionError with an unknown."""
    stat = PercentStat(percent=50.0, stat_type="min")
    stat.stat_type = "mean"
    with pytest.raises(AssertionError, match="mean"):
        stat.resolve(default_value=5.0)


# ===========================================================================
# PFTStat.from_sheet and resolve
# ===========================================================================


def test_pft_stat_from_sheet_returns_correct_values(pft_sheet):
    """PFTStat.from_sheet() loads the correct min values from the sheet."""
    stat = PFTStat.from_sheet(pft_sheet, "param_min")
    np.testing.assert_allclose(stat.values, [0.005, 0.004, 0.008])


def test_pft_stat_from_sheet_max_returns_correct_values(pft_sheet):
    """PFTStat.from_sheet() loads the correct max values from the sheet."""
    stat = PFTStat.from_sheet(pft_sheet, "param_max")
    np.testing.assert_allclose(stat.values, [0.040, 0.035, 0.060])


def test_pft_stat_from_sheet_values_are_float(pft_sheet):
    """PFTStat.from_sheet() stores values as float dtype."""
    stat = PFTStat.from_sheet(pft_sheet, "param_min")
    assert stat.values.dtype == float


def test_pft_stat_from_sheet_non_numeric_raises():
    """PFTStat.from_sheet() raises ValueError for non-numeric cell values."""
    bad_sheet = pd.DataFrame(
        {
            "pft_index": [1, 2, 3],
            "pft_name": ["white_spruce", "black_spruce", "deciduous"],
            "param_min": ["50percent", 0.004, 0.008],
            "param_max": [0.040, 0.035, 0.060],
        }
    )
    with pytest.raises(ValueError, match="fixed numbers"):
        PFTStat.from_sheet(bad_sheet, "param_min")


def test_pft_stat_from_sheet_missing_index_raises():
    """PFTStat.from_sheet() raises ValueError for missing pft_index."""
    bad_sheet = pd.DataFrame(
        {
            "pft_name": ["white_spruce", "black_spruce", "deciduous"],
            "param_min": [0.001, 0.004, 0.008],
            "param_max": [0.040, 0.035, 0.060],
        }
    )
    with pytest.raises(ValueError, match="PFT sheet"):
        PFTStat.from_sheet(bad_sheet, "param_min")


def test_pft_stat_from_sheet_missing_column_raises():
    """PFTStat.from_sheet() raises ValueError for missing column."""
    bad_sheet = pd.DataFrame(
        {
            "pft_index": [1, 2, 3],
            "pft_name": ["white_spruce", "black_spruce", "deciduous"],
            "param_max": [0.040, 0.035, 0.060],
        }
    )
    with pytest.raises(ValueError, match="PFT sheet"):
        PFTStat.from_sheet(bad_sheet, "param_min")


def test_pft_stat_from_sheet_indices_are_zero_based(pft_sheet):
    """PFTStat.from_sheet() stores 0-based indices."""
    stat = PFTStat.from_sheet(pft_sheet, "param_min")
    np.testing.assert_array_equal(stat.indices, [0, 1, 2])


def test_pft_stat_from_sheet_non_contiguous_indices():
    """PFTStat.from_sheet() correctly stores non-contiguous indices."""
    partial_sheet = pd.DataFrame(
        {
            "pft_index": [1, 3],
            "pft_name": ["white_spruce", "deciduous"],
            "param_min": [0.005, 0.008],
            "param_max": [0.040, 0.060],
        }
    )
    stat = PFTStat.from_sheet(partial_sheet, "param_min")
    np.testing.assert_array_equal(stat.indices, [0, 2])


def test_pft_stat_from_sheet_non_integer_pft_index_raises():
    """PFTStat.from_sheet() raises ValueError for non-integer pft_index values."""
    bad_sheet = pd.DataFrame(
        {
            "pft_index": ["a", "b", "c"],
            "pft_name": ["white_spruce", "black_spruce", "deciduous"],
            "param_min": [0.005, 0.004, 0.008],
            "param_max": [0.040, 0.035, 0.060],
        }
    )
    with pytest.raises(ValueError, match="non-integer"):
        PFTStat.from_sheet(bad_sheet, "param_min")


def test_pft_stat_from_sheet_sorts_by_pft_index():
    """PFTStat.from_sheet() sorts rows by pft_index before storing."""
    unsorted_sheet = pd.DataFrame(
        {
            "pft_index": [3, 1, 2],
            "pft_name": ["deciduous", "white_spruce", "black_spruce"],
            "param_min": [0.008, 0.005, 0.004],
            "param_max": [0.060, 0.040, 0.035],
        }
    )
    stat = PFTStat.from_sheet(unsorted_sheet, "param_min")
    np.testing.assert_allclose(stat.values, [0.005, 0.004, 0.008])
    np.testing.assert_array_equal(stat.indices, [0, 1, 2])


def test_pft_stat_resolve_returns_values(pft_sheet, param_dataset):
    """PFTStat.resolve() returns the stored values array."""
    stat = PFTStat.from_sheet(pft_sheet, "param_min")
    default_value = param_dataset["fates_leaf_slatop"].values
    result = stat.resolve(default_value)
    np.testing.assert_allclose(result, stat.values)


def test_pft_stat_resolve_no_default_raises(pft_sheet):
    """PFTStat.resolve() raises ValueError when no default_value is provided."""
    stat = PFTStat.from_sheet(pft_sheet, "param_min")
    with pytest.raises(ValueError, match="requires a default_value array"):
        stat.resolve()


def test_pft_stat_resolve_float_default_raises(pft_sheet):
    """PFTStat.resolve() raises ValueError when default_value is a float."""
    stat = PFTStat.from_sheet(pft_sheet, "param_min")
    with pytest.raises(ValueError, match="requires a default_value array"):
        stat.resolve(0.5)


def test_pft_stat_resolve_none_default_raises(pft_sheet):
    """PFTStat.resolve() raises ValueError when default_value is None."""
    stat = PFTStat.from_sheet(pft_sheet, "param_min")
    with pytest.raises(ValueError, match="requires a default_value array"):
        stat.resolve(None)


def test_pft_stat_resolve_partial_sheet_fills_from_default(param_dataset):
    """PFTStat.resolve() fills missing PFT indices from default_value."""
    partial_sheet = pd.DataFrame(
        {
            "pft_index": [1, 3],  # only PFTs 1 and 3 (1-based)
            "pft_name": ["white_spruce", "deciduous"],
            "param_min": [0.005, 0.008],
            "param_max": [0.040, 0.060],
        }
    )
    stat = PFTStat.from_sheet(partial_sheet, "param_min")
    default_value = param_dataset["fates_leaf_slatop"].values  # [0.010, 0.020, 0.030]
    result = stat.resolve(default_value)
    # index 0 (pft_index=1): sheet value
    # index 1 (pft_index=2): default
    # index 2 (pft_index=3): sheet value
    np.testing.assert_allclose(result, [0.005, 0.020, 0.008])


def test_pft_stat_resolve_2d_pft_axis_0(pft_sheet):
    """PFTStat.resolve() correctly indexes along axis 0 for 2D arrays."""
    # shape (3, 2): pft along axis 0
    default_value = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    stat = PFTStat.from_sheet(pft_sheet, "param_min")
    result = stat.resolve(default_value, pft_axis=0)
    # all 3 pft rows replaced
    assert result.shape == (3, 2)
    np.testing.assert_allclose(result[0, :], [0.005, 0.005])
    np.testing.assert_allclose(result[1, :], [0.004, 0.004])
    np.testing.assert_allclose(result[2, :], [0.008, 0.008])


def test_pft_stat_resolve_2d_pft_axis_1(pft_sheet):
    """PFTStat.resolve() correctly indexes along axis 1 for 2D arrays."""
    # shape (2, 3): pft along axis 1
    default_value = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    stat = PFTStat.from_sheet(pft_sheet, "param_min")
    result = stat.resolve(default_value, pft_axis=1)
    assert result.shape == (2, 3)
    np.testing.assert_allclose(result[:, 0], [0.005, 0.005])
    np.testing.assert_allclose(result[:, 1], [0.004, 0.004])
    np.testing.assert_allclose(result[:, 2], [0.008, 0.008])


def test_pft_stat_resolve_2d_partial_sheet_fills_from_default():
    """PFTStat.resolve() fills missing PFT indices from default for 2D arrays."""
    partial_sheet = pd.DataFrame(
        {
            "pft_index": [1, 3],
            "pft_name": ["white_spruce", "deciduous"],
            "param_min": [0.005, 0.008],
            "param_max": [0.040, 0.060],
        }
    )
    stat = PFTStat.from_sheet(partial_sheet, "param_min")
    default_value = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = stat.resolve(default_value, pft_axis=1)
    # pft index 0 (pft_index=1): sheet value
    # pft index 1 (pft_index=2): default
    # pft index 2 (pft_index=3): sheet value
    np.testing.assert_allclose(result[:, 0], [0.005, 0.005])
    np.testing.assert_allclose(result[:, 1], [2.0, 5.0])  # default
    np.testing.assert_allclose(result[:, 2], [0.008, 0.008])


def test_pft_stat_resolve_2d_missing_pft_axis_raises():
    """PFTStat.resolve() raises ValueError for 2D array without pft_axis."""
    stat = PFTStat(
        values=np.array([0.005, 0.004, 0.008]),
        indices=np.array([0, 1, 2]),
    )
    default_value = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    with pytest.raises(ValueError, match="pft_axis"):
        stat.resolve(default_value, pft_axis=None)
