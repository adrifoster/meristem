"""Command-line interface for meristem.paramgen"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from .param_ensemble import ParamEnsemble


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
    """Generate the ensemble described by the config file in args.config

    Args:
        args (argparse.Namespace): parsed namespace from argparse
    """

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

    try:
        ensemble.create_ensemble(overwrite=args.overwrite)
    except FileExistsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"Ensemble complete. Output written to {ensemble.ensemble_dir}/")


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser.

    Returns:
        argparse.ArgumentParser: argument parser
    """
    parser = argparse.ArgumentParser(
        prog="meristem paramgen",
        description=(
            "Generate a CLM/FATES parameter ensemble from a YAML config file. "
            "The config file should contain the same keys as the Python "
            "ParamEnsemble.from_dict() interface."
        ),
    )
    parser.add_argument(
        "config",
        metavar="CONFIG",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing ensemble with the same file_prefix in "
            "ensemble_dir (its files are deleted first)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the meristem paramgen command.

    Args:
        argv (list[str] | None): argument list (defaults to sys.argv if None).

    Returns:
        int: exit code, 0 on success (errors exit via sys.exit)
    """
    args = build_parser().parse_args(argv)
    _run(args)
    return 0
