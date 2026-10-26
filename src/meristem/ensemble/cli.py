#!/usr/bin/env python3
"""
CLI for a ensemble run (the `meristem ensemble` command).
"""

import argparse
from pathlib import Path

from .model_ensemble import Ensemble


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="meristem ensemble",
        description="Drive an ensemble of model runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # debug command, used by all subcommands
    debug = argparse.ArgumentParser(add_help=False)
    debug.add_argument(
        "--debug", action="store_true", help="Show the full traceback on errors."
    )

    # create subparsers
    subparsers = parser.add_subparsers(
        dest="command", metavar="COMMAND", help="Available subcommands.", required=True
    )

    # meristem ensemble create --config CONFIG [--debug]
    create = subparsers.add_parser(
        "create",
        parents=[debug],
        help="Initialize a new ensemble from a YAML config",
    )
    create.add_argument("--config", required=True, help="Path to run.yaml")
    create.set_defaults(handler=cmd_create)

    return parser


def cmd_create(args):
    ensemble = Ensemble.create(args.config)
    # print(f"Initialized run '{run.run_id}' at {run.root}")
    # print(f"Next step:  meristem run submit --root {run.root}")
