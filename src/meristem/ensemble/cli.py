#!/usr/bin/env python3
"""
CLI for a ensemble run (the `meristem ensemble` command).
"""

import argparse
import sys
from pathlib import Path

from .model_ensemble import Ensemble

# exit code from status when a stage is lost (1 is a runtime error, 2 an argument error)
EXIT_LOST = 3


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
        "--dry-run",
        action="store_true",
        help="Write job scripts but don't submit them with qsub.",
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
        "submit", parents=[debug, root, dry_run], help="Submit ensemble."
    )
    submit.set_defaults(handler=cmd_submit)
    submit.add_argument(
        "--limit",
        type=int,
        help="Optional number of ensemble members to run as a test.",
    )

    # meristem ensemble prepare --root R --member M --stage S [--debug]
    prepare = subparsers.add_parser(
        "prepare",
        parents=[debug, root],
        help="Clone one member's case and run and run case.setup.",
    )
    prepare.add_argument("--stage", required=True, help="Stage to prepare.")
    prepare.add_argument("--member", required=True, help="Member to prepare.")
    prepare.set_defaults(handler=cmd_prepare)

    # meristem ensemble finalize --root R --member M --stage S [--debug]
    finalize = subparsers.add_parser(
        "finalize",
        parents=[debug, root],
        help="Set the member's parameter file in its case and verify it.",
    )
    finalize.add_argument("--stage", required=True, help="Stage to finalize.")
    finalize.add_argument("--member", required=True, help="Member to finalize.")
    finalize.set_defaults(handler=cmd_finalize)

    # meristem ensemble retry --root R [--failed] [--stage S] [--members M [M...]]  /
    # [--lost] [--debug] [--dry-run]
    retry = subparsers.add_parser(
        "retry",
        parents=[debug, root, dry_run],
        help="Retry the selected stages and/or members. "
        "If --failed retry all failed jobs. If --lost retry lost jobs.",
    )
    retry.add_argument("--stage", help="Stage to retry.")
    retry.add_argument(
        "--failed",
        action="store_true",
        help="Retry all failed jobs.",
    )
    retry.add_argument(
        "--lost",
        action="store_true",
        help="Retry all lost jobs.",
    )
    retry.add_argument(
        "--skip-script",
        action="store_true",
        help="Skip running any script and just resubmit job.",
    )
    retry.add_argument(
        "--member",
        "--members",
        dest="members",
        nargs="+",
        action="extend",
        help="Members to retry (e.g. --members 0001 0002). Defaults to all if not present.",
    )
    retry.set_defaults(handler=cmd_retry)

    # meristem ensemble status --root R [--members M [M...]] [--print-members] [--check-queue]
    status = subparsers.add_parser(
        "status",
        parents=[debug, root],
        help="Print the ensemble status.",
    )
    status.add_argument(
        "--member",
        "--members",
        dest="members",
        nargs="+",
        action="extend",
        help="Members to check (e.g. --members 0001 0002). If not present, defaults to all.",
    )
    status.add_argument(
        "--print-members", action="store_true", help="Also list every member."
    )
    status.add_argument(
        "--check-queue",
        action="store_true",
        help="Query job scheduler and report lost stages.",
    )
    status.set_defaults(handler=cmd_status)

    # meristem ensemble manifest --root R [-o OUT]
    manifest = subparsers.add_parser(
        "manifest",
        parents=[debug, root],
        help="Write an ensemble manifest csv.",
    )
    manifest.add_argument(
        "-o",
        "--output",
        help="File to write to. Default is <root>/manifest.csv",
    )
    manifest.set_defaults(handler=cmd_manifest)

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


def cmd_prepare(args: argparse.Namespace) -> int:
    """Execute prepare and return an exit code"""
    ensemble = Ensemble.load(Path(args.root))
    ensemble.prepare(member=args.member, stage=args.stage)
    return 0


def cmd_finalize(args: argparse.Namespace) -> int:
    """Execute finalize and return an exit code"""
    ensemble = Ensemble.load(Path(args.root))
    ensemble.finalize(member=args.member, stage=args.stage)
    return 0


def cmd_retry(args: argparse.Namespace) -> int:
    """Execute retry and return an exit code"""
    ensemble = Ensemble.load(Path(args.root))
    ensemble.retry(
        failed=args.failed,
        lost=args.lost,
        members=args.members,
        stage=args.stage,
        skip_script=args.skip_script,
        dry_run=args.dry_run,
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Print the ensemble status and return 0, or EXIT_LOST if any stage is lost"""
    ensemble = Ensemble.load(Path(args.root))
    status = ensemble.status(members=args.members, check_queue=args.check_queue)
    print(ensemble.format_status(status, show_members=args.print_members))
    return EXIT_LOST if (status["status"] == "lost").any() else 0


def cmd_manifest(args: argparse.Namespace) -> int:
    """Write the ensemble manifest to a CSV file"""
    ensemble = Ensemble.load(Path(args.root))
    out = Path(args.output) if args.output else ensemble.root / "manifest.csv"
    ensemble.manifest().to_csv(out, index=False)
    print(f"Wrote manifest: {out}")
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
