"""Diagnostic utility functions for param_ens_gen

Functions for analyzing and understanding parameter ranges and default values,
useful for plotting and ensemble design.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .parameter import Parameter
from .parameter_dataset import ParameterDataset
from .utils import read_param_list

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover
    HAS_MATPLOTLIB = False

_CATEGORY_COLORS = {
    "acclimation": "#5BA8A0",
    "allometry": "#1D9E75",
    "allocation": "#7F77DD",
    "biogeochemistry": "#7DBB9E",
    "canopy aerodynamics": "#90D5FF",
    "canopy evaporation": "#5DADE2",
    "decomposition": "#639922",
    "fire": "#b22222",
    "FUN": "#341539",
    "LUNA": "#F7C59F",
    "mortality": "#D4537E",
    "nitrogen": "#8B9D2B",
    "phenology": "#EF9F27",
    "photosynthesis": "#D85A30",
    "radiation": "#0F6E56",
    "recruitment": "#993556",
    "respiration": "#BA7517",
    "snow": "#B0C4DE",
    "soil water": "#13265C",
    "stomatal": "#C97DB0",
    "turbulence": "#85B7EB",
    "vegetation water": "#378ADD",
    "vegetation dynamics": "#F0997B",
}
_DEFAULT_COLOR = "#2C2C2A"


def _expand_normalized(
    param: Parameter,
    normalized: float | np.ndarray | list,
    pft_index: int | None = None,
) -> list[tuple[str, float, str]]:
    """Expand a normalized value into (name, value, category) tuples."""
    param_names = (
        param.spec.base_params if param.spec.base_params else [param.spec.name]
    )
    normalized_list = normalized if isinstance(normalized, list) else [normalized]
    subcategory = param.spec.subcategory

    rows = []
    for pname, norm_val in zip(param_names, normalized_list):
        arr = np.asarray(norm_val)
        if arr.ndim == 0:
            rows.append((pname, float(arr), subcategory))
        else:
            for i, v in enumerate(arr.flat):
                if pft_index is None or i == pft_index:
                    rows.append(
                        (
                            f"{pname}_{i}" if pft_index is None else pname,
                            float(v),
                            subcategory,
                        )
                    )
    return rows


def normalize_defaults(
    param_dir: Path,
    default_param_file: Path,
    posterior_sources: Path | None = None,
    param_list: list[str] | None = None,
    pft_index: int | None = None,
) -> pd.DataFrame:
    """Return normalized default values for all parameters in param_dir.

    For each parameter, computes where the default value sits in [0, 1]
    relative to the parameter's prior bounds. Useful for understanding
    whether defaults are centered, skewed toward min or max, or outside
    the specified range.

    Args:
        param_dir (Path): Path to the parameter metadata directory containing
            main.csv and optional per-PFT bound files.
        default_param_file (Path): Path to the default parameter file
            (NetCDF or JSON).
        posterior_sources (Path | None, optional): Path to a posterior sources
            YAML file. Required for parameters with strategy='posterior'.
            Defaults to None.
        param_list (list[str] | None, optional): Subset of parameters to
            normalize. If None, all parameters in main.csv are used.
            Defaults to None.
        pft_index (int | None, optional): If set, only returns this PFT axis. Defaults to
        None.

    Returns:
        pd.DataFrame: One row per parameter with columns:
            - 'parameter': parameter name
            - 'normalized_value': where the default sits in [0, 1]
    """
    param_dir = Path(param_dir)
    default_param_file = Path(default_param_file)

    main, pft_sheets = read_param_list(param_dir)

    if param_list is not None:
        missing = set(param_list) - set(main.parameter_name)
        if missing:
            raise ValueError(
                f"param_list contains parameters not found in main.csv: "
                f"{sorted(missing)}. "
                f"Available parameters: {sorted(main.parameter_name)}"
            )
        main = main[main.parameter_name.isin(param_list)].copy()

    posterior_config: dict = {}
    if posterior_sources is not None:
        posterior_sources = Path(posterior_sources)
        if not posterior_sources.exists():
            raise FileNotFoundError(
                f"Posterior sources file '{posterior_sources}' does not exist."
            )
        with open(posterior_sources, "r", encoding="utf-8") as f:
            posterior_config = yaml.safe_load(f)

    default_ds = ParameterDataset.from_path(default_param_file)

    params = [
        Parameter.from_row(
            row,
            pft_sheet=pft_sheets.get(row["parameter_name"]),
            default_ds=default_ds,
            posterior_config=(
                posterior_config.get(row["parameter_name"])
                if posterior_config
                else None
            ),
        )
        for _, row in main.iterrows()
    ]

    rows = []
    for param in params:
        default_value = param.get_default(default_ds)
        normalized = param.normalize(default_value, default_ds, clip_to_bounds=True)
        rows.extend(
            (name, val, cat)
            for name, val, cat in _expand_normalized(
                param, normalized, pft_index=pft_index
            )
            if not (isinstance(val, float) and np.isnan(val))
        )
    return pd.DataFrame(rows, columns=["parameter", "normalized_value", "category"])


def _build_legend(present_cats: list[str], category_colors: dict, default_color: str):
    """Build legend elements for plot_param_bounds."""
    elements = [
        mpatches.Patch(
            facecolor=category_colors.get(cat, "#888780"), label=cat, alpha=0.8
        )
        for cat in sorted(present_cats)
    ]
    elements.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=default_color,
            markersize=5,
            label="Default value",
        )
    )
    return elements


def _draw_param_row(ax, y: int, x_def: float, color: str, default_color: str):
    """Draw a single parameter row on the axes."""
    ax.plot(
        [0.0, 1.0],
        [y, y],
        color=color,
        linewidth=3.5,
        solid_capstyle="round",
        alpha=0.75,
        zorder=2,
    )
    ax.plot(
        [0.0, 1.0],
        [y, y],
        marker="|",
        color=color,
        markersize=7,
        markeredgewidth=1.5,
        linewidth=0,
        zorder=3,
    )
    ax.plot(
        x_def, y, "o", color=default_color, markersize=4.5, zorder=4, markeredgewidth=0
    )


def plot_param_bounds(df: pd.DataFrame):
    """Plot normalized parameter ranges with default values marked.

    Each parameter is shown as a horizontal bar spanning [0, 1] with a dot
    marking where the default value sits. Parameters are grouped by category
    and color-coded accordingly.

    Args:
        df (pd.DataFrame): Output of normalize_defaults(), with columns
            'parameter', 'normalized_value', and 'category'.

    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: conda install matplotlib"
        )  # pragma: no cover

    df = df.sort_values(by="category")
    n_params = len(df)
    fig_height = max(6, n_params * 0.32 + 2)
    fig, ax = plt.subplots(figsize=(9, fig_height))

    yticks, ylabels = [], []
    prev_cat = None
    y = 0

    for _, row in df.iterrows():
        color = _CATEGORY_COLORS.get(row.get("category", ""), "#888780")
        x_def = float(np.clip(row["normalized_value"], 0.0, 1.0))

        if y % 2 == 0:
            ax.axhspan(y - 0.4, y + 0.4, color="#f7f7f5", zorder=0, linewidth=0)

        if row.get("category") != prev_cat and prev_cat is not None:
            ax.axhline(
                y - 0.5, color="#cccccc", linewidth=0.8, linestyle="--", zorder=1
            )
        prev_cat = row.get("category")

        _draw_param_row(ax, y, x_def, color, _DEFAULT_COLOR)

        yticks.append(y)
        ylabels.append(row["parameter"])
        y += 1

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7.5, fontfamily="monospace")
    ax.invert_yaxis()
    ax.set_xlim(-0.05, 1.05)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["min", "25%", "50%", "75%", "max"], fontsize=8)
    ax.set_xlabel("Normalized prior range  [min to max]", fontsize=9)
    ax.axvline(0.5, color="#cccccc", linewidth=0.8, linestyle=":", zorder=0)
    ax.set_ylim(n_params - 0.5, -0.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(axis="x", color="#e0e0e0", linewidth=0.6, zorder=0)

    present_cats = df["category"].unique() if "category" in df.columns else []

    ax.legend(
        handles=_build_legend(present_cats, _CATEGORY_COLORS, _DEFAULT_COLOR),
        fontsize=7.5,
        framealpha=0.9,
        edgecolor="#cccccc",
        bbox_to_anchor=[1.1, 0.5],
        loc="center",
    )
    ax.set_title(
        "Parameter ranges by functional group",
        fontsize=11,
        fontweight="500",
        pad=12,
    )
    
    return fig
