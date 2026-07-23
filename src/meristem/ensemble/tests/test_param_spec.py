"""Tests for param_ens_gen.param_spec: ParamSpec and its parsing helpers."""

import pytest

from param_ens_gen.param_spec import ParamSpec, _REQUIRED_COLUMNS, _parse_dims

# _parse_dims is tested directly because its defensive branches (non-string,
# whitespace) are unreachable through from_row but protect against future callers.

# ===========================================================================
# ParamSpec.from_row: successful construction
# ===========================================================================


def test_from_row_default(default_row):
    """from_row correctly constructs a default parameter."""
    spec = ParamSpec.from_row(default_row)
    assert spec.name == "fates_leaf_slatop"
    assert spec.param_type == "default"
    assert spec.dims == ["fates_pft"]
    assert spec.slice_dim is None
    assert spec.slice_index is None
    assert spec.root_param is None
    assert not spec.base_params


def test_from_row_scalar(scalar_row):
    """from_row correctly constructs a scalar parameter."""
    spec = ParamSpec.from_row(scalar_row)
    assert not spec.dims


def test_from_row_sliced(sliced_row):
    """from_row correctly constructs a sliced parameter."""
    spec = ParamSpec.from_row(sliced_row)
    assert spec.param_type == "sliced"
    assert spec.slice_dim == "fates_leafage_class"
    assert spec.slice_index == 0
    assert spec.base_params == ["fates_leaf_vcmax25top"]


def test_from_row_scale_from_root(scale_from_root_row):
    """from_row correctly constructs a scale_from_root parameter."""
    spec = ParamSpec.from_row(scale_from_root_row)
    assert spec.param_type == "scale_from_root"
    assert spec.root_param == "fates_nonhydro_smpso"
    assert spec.base_params == ["fates_nonhydro_smpsc"]


def test_from_row_joint_param(joint_param_row):
    """from_row correctly constructs a joint_param parameter."""
    spec = ParamSpec.from_row(joint_param_row)
    assert spec.param_type == "joint"
    assert spec.base_params == [
        "fates_leafn_vert_scaler_coeff1",
        "fates_leafn_vert_scaler_coeff2",
    ]


def test_from_row_metadata_fields(default_row):
    """from_row correctly reads metadata fields (category, long_name, units)."""
    spec = ParamSpec.from_row(default_row)
    assert (
        spec.long_name
        == "Specific Leaf Area (SLA) at top of canopy, projected area basis"
    )
    assert spec.category == "stomatal"
    assert spec.subcategory == "photosynthesis"
    assert spec.units == "m^2/gC"


@pytest.mark.parametrize("value", [None, float("nan")])
def test_from_row_missing_coord_raises(default_row, value):
    """from_row raises ValueError when the coord cell is NaN (blank in spreadsheet).

    NaN coord is a data error — the spreadsheet author forgot to fill it in.
    An explicit '[]' should be used for scalar parameters instead.
    """
    row = default_row.copy()
    row["coord"] = value
    with pytest.raises(ValueError, match="coord cell is missing"):
        ParamSpec.from_row(row)


@pytest.mark.parametrize(
    "missing_cols",
    list(_REQUIRED_COLUMNS),
)
def test_from_row_missing_column_raises(default_row, missing_cols):
    """from_row raises ValueError if there is a missing column."""
    row = default_row.copy()
    row = row.drop(missing_cols)
    with pytest.raises(KeyError, match="Row is missing "):
        ParamSpec.from_row(row)


@pytest.mark.parametrize("blank_col", list(_REQUIRED_COLUMNS))
@pytest.mark.parametrize("blank_value", ["", "   ", "\t"])
def test_from_row_blank_column_raises(default_row, blank_col, blank_value):
    """from_row raises ValueError if a required column is blank."""
    row = default_row.copy()
    row[blank_col] = blank_value
    with pytest.raises(ValueError, match="Row has blank values in required columns"):
        ParamSpec.from_row(row)


def test_from_row_explicit_empty_coord_is_scalar(default_row):
    """from_row produces an empty dims list when coord is explicitly '[]'."""
    row = default_row.copy()
    row["coord"] = "[]"
    spec = ParamSpec.from_row(row)
    assert not spec.dims


def test_from_row_base_params_parsed_from_list_literal(joint_param_row):
    """from_row parses a Python list literal in the base_params cell."""
    spec = ParamSpec.from_row(joint_param_row)
    assert len(spec.base_params) == 2


def test_from_row_base_params_parsed_from_plain_string(scale_from_root_row):
    """from_row parses a plain (non-bracketed) string in the base_params cell."""
    spec = ParamSpec.from_row(scale_from_root_row)
    assert spec.base_params == ["fates_nonhydro_smpsc"]


@pytest.mark.parametrize(
    "value, expected_slice_index",
    [
        (None, None),
        (float("nan"), None),  # blank numeric cell from pandas
        ("3", 3),
        ("0", 0),
        ("", None),
    ],
)
def test_from_row_slice_index_valid(sliced_row, value, expected_slice_index):
    """from_row with a slice index is valid"""
    row = sliced_row.copy()
    row["slice_index"] = value
    result = ParamSpec.from_row(row)
    assert result.slice_index == expected_slice_index


@pytest.mark.parametrize("value", ["abc", "3.7"])
def test_from_row_slice_index_raises(sliced_row, value):
    """from_row raises for invalid values for slice_index"""
    row = sliced_row.copy()
    row["slice_index"] = value
    with pytest.raises(ValueError, match="slice_index"):
        ParamSpec.from_row(row)


@pytest.mark.parametrize(
    "value, match",
    [
        (None, "coord cell is missing"),
        (float("nan"), "coord cell is missing"),
        (42, "coord cell has unexpected type"),
        ([], "coord cell has unexpected type"),
        ("   ", "coord cell is blank"),
    ],
)
def test_parse_dims_raises(value, match):
    """_parse_dims raises for invalid values"""
    with pytest.raises(ValueError, match=match):
        _parse_dims(value)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("[]", []),
        ("['fates_pft']", ["fates_pft"]),
        ("'fates_pft'", ["fates_pft"]),
        ("fates_pft", ["fates_pft"]),
        ("'fates_pft', 'pft'", ["fates_pft", "pft"]),
        ("fates_pft, pft", ["fates_pft", "pft"]),
        ("['fates_pft', 'pft']", ["fates_pft", "pft"]),
    ],
)
def test_parse_dims_valid(value, expected):
    """_parse_dims works for valid values"""
    assert _parse_dims(value) == expected


@pytest.mark.parametrize(
    "value, expected_base_params",
    [
        (None, []),
        (float("nan"), []),  # blank numeric cell from pandas
        ("[]", []),
        ("['a', 'b']", ["a", "b"]),
        ("'a'", ["a"]),
        ("'a', 'b'", ["a", "b"]),
        ("a, b, c", ["a", "b", "c"]),
        ("fates_pft", ["fates_pft"]),
    ],
)
def test_from_row_parse_list(sliced_row, value, expected_base_params):
    """from_row works for valid values for base_params"""
    row = sliced_row.copy()
    row["base_params"] = value
    result = ParamSpec.from_row(row)
    assert result.base_params == expected_base_params


@pytest.mark.parametrize("value", [42, [], True])
def test_parse_list_bad_type_raises(sliced_row, value):
    """from_row raises for bad values for base_params"""
    row = sliced_row.copy()
    row["base_params"] = value
    with pytest.raises(ValueError, match="unexpected type"):
        ParamSpec.from_row(row)


def test_from_row_precision_parsed(default_row):
    """from_row correctly parses a precision string."""
    default_row["precision"] = ".4f"
    spec = ParamSpec.from_row(default_row)
    assert spec.precision == ".4f"


def test_from_row_precision_none_when_missing(default_row):
    """from_row sets precision to None when the cell is missing."""
    spec = ParamSpec.from_row(default_row)
    assert spec.precision is None


@pytest.mark.parametrize("value", [None, float("nan"), ""])
def test_from_row_precision_none_for_blank(default_row, value):
    """from_row sets precision to None for blank/null values."""
    default_row["precision"] = value
    spec = ParamSpec.from_row(default_row)
    assert spec.precision is None


def test_from_row_precision_normalizes_leading_zero(default_row):
    """from_row normalizes '0.2f' to '.2f'."""
    default_row["precision"] = "0.2f"
    spec = ParamSpec.from_row(default_row)
    assert spec.precision == ".2f"


def test_from_row_group_name_parsed(default_row):
    """from_row correctly parses a group_name string."""
    default_row["group_name"] = "photosynthesis"
    spec = ParamSpec.from_row(default_row)
    assert spec.group_name == "photosynthesis"


def test_from_row_group_name_none_when_missing(default_row):
    """from_row sets group_name to None when the cell is missing."""
    spec = ParamSpec.from_row(default_row)
    assert spec.group_name is None


@pytest.mark.parametrize("value", [None, float("nan"), ""])
def test_from_row_group_name_none_for_blank(default_row, value):
    """from_row sets group_name to None for blank/null values."""
    default_row["group_name"] = value
    spec = ParamSpec.from_row(default_row)
    assert spec.group_name is None


# ===========================================================================
# ParamSpec.__post_init__: field-level invariants
#
# These invariants hold regardless of param_type: putting a slice field on a
# non-sliced type, or root_param on a non-scale_from_root type, is always
# wrong at the field level. The Parameter subclass validates type-specific
# required fields (e.g. SlicedParameter checks that slice_dim IS present).
# ===========================================================================


def test_slice_dim_on_non_sliced_raises(default_row):
    """ValueError is raised when slice_dim is set on a non-sliced param."""
    row = default_row.copy()
    row["slice_dim"] = "fates_leafage_class"
    with pytest.raises(ValueError, match="slice_dim set but param_type"):
        ParamSpec.from_row(row)


def test_slice_index_on_non_sliced_raises(default_row):
    """ValueError is raised when slice_index is set on a non-sliced param.

    Uses "0" (string) because the default_row Series is string-typed,
    matching how pandas reads numeric cells from Excel.
    """
    row = default_row.copy()
    row["slice_index"] = "0"
    with pytest.raises(ValueError, match="slice_index set but param_type"):
        ParamSpec.from_row(row)


def test_root_param_on_non_scale_raises(default_row):
    """ValueError is raised when root_param is set on a non-scale_from_root param."""
    row = default_row.copy()
    row["root_param"] = "fates_nonhydro_smpso"
    with pytest.raises(ValueError, match="root_param set but param_type"):
        ParamSpec.from_row(row)
