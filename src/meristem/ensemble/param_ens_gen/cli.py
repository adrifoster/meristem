"""Command-line interface for param_ens_gen"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from param_ens_gen import ParamEnsemble


def _load_config(config_path: Path) -> dict:
    """Load and return a YAML config file.

    Args:
        config_path (Path): Path to the YAML config file.

    Returns:
        dict: Parsed config dictionary.
    """
    if not config_path.exists():
        print(f"Error: config file '{config_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        print("Error: config file must be a YAML mapping.", file=sys.stderr)
        sys.exit(1)

    return config


def _run(args: argparse.Namespace):

    config = _load_config(Path(args.config))

    try:
        ensemble = ParamEnsemble.from_dict(config)
    except (ValueError, TypeError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    ensemble_type = config.get("ensemble_type", "")
    print(
        f"Running {ensemble_type} ensemble: "
        f"{ensemble.num_params} parameters, "
        f"output directory: {ensemble.ensemble_dir}"
    )

    ensemble.create_ensemble()
    print(f"Ensemble complete. Output written to {ensemble.ensemble_dir}/")


def main():
    """Entry point for the param_ens_gen CLI."""
    parser = argparse.ArgumentParser(
        prog="param_ens_gen",
        description="Generate CLM/FATES parameter ensembles.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run subcommand
    run_parser = subparsers.add_parser(
        "run",
        help="Generate an ensemble from a YAML config file.",
        description=(
            "Generate a parameter ensemble from a YAML config file. "
            "The config file should contain the same keys as the Python "
            "ParamEnsemble.from_dict() interface."
        ),
    )
    run_parser.add_argument(
        "config",
        metavar="CONFIG",
        help="Path to the YAML configuration file.",
    )
    run_parser.set_defaults(func=_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
