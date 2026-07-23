"""Tests for sort_params.py"""

import pytest

from param_ens_gen.sort_params import sort_params
from param_ens_gen.parameter import DefaultParameter, ScaleFromRootParameter


def test_sort_params_no_dependencies(default_param, scalar_param):
    """Parameters with no dependencies preserve relative order."""
    params = [default_param, scalar_param]
    result = sort_params(params)
    assert [p.spec.name for p in result] == [
        default_param.spec.name,
        scalar_param.spec.name,
    ]


def test_sort_params_root_before_dependent(
    param_dataset, scale_from_root_row, root_row
):
    """ScaleFromRootParameter is sorted after its root."""

    root = DefaultParameter(root_row, param_dataset)
    dependent = ScaleFromRootParameter(scale_from_root_row, param_dataset)

    # pass dependent first to prove sorting does work
    result = sort_params([dependent, root])
    names = [p.spec.name for p in result]
    assert names.index("fates_nonhydro_smpso") < names.index("smpsc_delta")


def test_sort_params_root_not_in_list(scale_param):
    """Root param absent from list — no constraint added, no error raised."""
    result = sort_params([scale_param])
    assert len(result) == 1
    assert result[0].spec.name == scale_param.spec.name


def test_sort_params_cycle_raises(param_dataset, mutually_dependent_rows):
    """Cycle among ScaleFromRootParameters raises ValueError."""

    row_a, row_b = mutually_dependent_rows
    param_a = ScaleFromRootParameter(row_a, param_dataset)
    param_b = ScaleFromRootParameter(row_b, param_dataset)

    with pytest.raises(ValueError, match="Cycle detected"):
        sort_params([param_a, param_b])


def test_sort_params_empty():
    """Empty list returns empty list."""
    assert sort_params([]) == []
