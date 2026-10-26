#!/usr/bin/env python3
"""
CLI for a ensemble run (the `meristem ensemble` command).
"""

import argparse
import sys
from pathlib import Path

from .model_ensemble import Ensemble


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser"""
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

    # all subcommands (except create) use this
    root = argparse.ArgumentParser(add_help=False)
    root.add_argument("--root", required=True, help="Ensemble root directory.")

    # dry-run command, used by some subcommands
    dry_run = argparse.ArgumentParser(add_help=False)
    dry_run.add_argument(
        "--dry-run", action="store_true", help="Write job scripts but don't submit them with qsub."
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

    # meristem ensemble submit --root R [--debug] [--dry-run] [--limit N]
    submit = subparsers.add_parser(
        "submit",
        parents=[debug, root, dry_run],
        help="Submit ensemble."
    )
    submit.set_defaults(handler=cmd_submit)
    submit.add_argument("--limit", help="Optional number of ensemble members to run as a test.")

    return parser


def cmd_create(args: argparse.Namespace) -> int:
    """Execute create and return an exit code"""
    ensemble = Ensemble.create(args.config)
    print(f"Initialized ensemble '{ensemble.config.ensemble_id}' at {ensemble.root}")
    print(f"Next step:  meristem ensemble submit --root {ensemble.root}")
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    """Execute submit and return an exit code"""
    ensemble = Ensemble.load(Path(args.root))
    ensemble.submit(limit=args.limit, dry_run=args.dry_run)
    print(f"Submitted ensemble '{ensemble.config.ensemble_id}' at {ensemble.root}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the meristem run command.

    Args:
        argv (list[str] | None): argument list (defaults to sys.argv if None). Defaults to None.

    Returns:
        int: exit code, 0 on success, 1 on runtime error, 2 on argument error,
            EXIT_LOST (3) if status found a lost stage
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return args.handler(args)
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 130
    # top-level catch-all: report any error cleanly unless --debug is set
    except Exception as e:  # pylint: disable=broad-exception-caught
        if args.debug:
            raise
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
