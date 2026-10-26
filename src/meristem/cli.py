"""
Top-level `meristem` command.

Dispatches to the subcommand CLIs, each of which parses its own arguments:
    meristem paramgen config.yaml
    meristem run create --config run.yaml
    meristem run status --root /path/to/run
"""

from __future__ import annotations

import argparse
import importlib
import sys

from . import __version__

# subcommand name -> (module with a main(argv) -> int function, one-line help)
# modules are imported only when their subcommand is used, so e.g. the
# `meristem run` calls inside PBS jobs don't import paramgen's dependencies
_COMMANDS = {
    "paramgen": ("meristem.paramgen.cli", "Generate a parameter ensemble"),
    "run": ("meristem.run.cli", "Create, submit and manage one (tethered) model run"),
    "ensemble": ("meristem.ensemble.cli", "Create, submit, and manage an ensemble of model runs"),
}


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser.

    Returns:
        argparse.ArgumentParser: argument parser
    """
    parser = argparse.ArgumentParser(
        prog="meristem",
        description="Parameter ensembles and tethered runs for CLM and CLM-FATES.",
        epilog="Run 'meristem COMMAND --help' for a command's options.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for name, (_, help_text) in _COMMANDS.items():
        # add_help=False so `meristem run --help` reaches the run CLI's own parser
        subparsers.add_parser(name, help=help_text, add_help=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the meristem command.

    Args:
        argv (list[str] | None): argument list (defaults to sys.argv if None).

    Returns:
        int: exit code from the subcommand
    """
    argv = sys.argv[1:] if argv is None else argv
    args, rest = build_parser().parse_known_args(argv)
    module = importlib.import_module(_COMMANDS[args.command][0])
    return module.main(rest)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
