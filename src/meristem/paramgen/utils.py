"""Utility functions"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import yaml


def read_param_list(
    param_dir: Path,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Read in the directory of CSVs that sets up all the parameters.

    Expects a directory containing at least a 'main.csv' file. Any other *csv files in
    the directory are read as PFT-specific parameter sheets, keyed by their filename
    without extension.

    Args:
        param_data_file (Path): path to directory containing parameter CSVs.

    Returns:
        tuple[pd.DataFrame, dict[str, pd.DataFrame]]: main dataframe, dictionary of
        sheets with pft-specific parameter values
    """
    param_dir = Path(param_dir)

    if not param_dir.exists():
        raise FileNotFoundError(f"Parameter directory '{param_dir}' does not exist.")
    if not param_dir.is_dir():
        raise FileNotFoundError(f"'{param_dir}' is not a directory.")

    main_path = param_dir / "main.csv"
    if not main_path.exists():
        raise FileNotFoundError(f"'main.csv' not found in '{param_dir}'.")

    try:
        main = pd.read_csv(main_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"'main.csv' in '{param_dir}' is empty.") from exc

    if main.empty:
        raise ValueError(f"'main.csv' in '{param_dir}' is empty.")

    pft_sheets = {}
    for csv_file in sorted(param_dir.glob("*.csv")):
        if csv_file.stem != "main":
            pft_sheets[csv_file.stem] = pd.read_csv(csv_file)

    return main, pft_sheets


def read_posterior_sources(posterior_sources: Path | None) -> dict:
    """Read the posterior sources YAML file that maps parameters to posterior draws.

    Args:
        posterior_sources (Path | None): path to the posterior sources YAML file.

    Returns:
        dict: posterior config keyed by parameter name, or an empty dict if
        posterior_sources is None.
    """
    if posterior_sources is None:
        return {}

    posterior_sources = Path(posterior_sources)
    if not posterior_sources.exists():
        raise FileNotFoundError(
            f"Posterior sources file '{posterior_sources}' does not exist."
        )
    with open(posterior_sources, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def validate_normalized_value(normalized_value: float) -> None:
    """Raise ValueError if normalized_value is outside [0, 1].

    Args:
        normalized_value (float): Value to validate.
    """
    if normalized_value > 1.0:
        raise ValueError(
            f"normalized_value={normalized_value}. "
            "Cannot use a normalized_value greater than 1.0"
        )
    if normalized_value < 0.0:
        raise ValueError(
            f"normalized_value={normalized_value}. "
            "Cannot use a normalized_value less than 0.0"
        )
