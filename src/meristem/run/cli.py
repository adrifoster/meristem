#!/usr/bin/env python3
"""
CLI for a single ModelRun (the `meristem run` command).

First-time setup: meristem run create --config run.yaml

All other commands (many called automatically by pipeline):
    meristem run submit --root /path/to/run [--dry-run] [--debug]
    meristem run submit-advance --root /path/to/run --stage spinup_ad --cime-job-id 2003.desched1
    meristem run submit-fail --root /path/to/run --stage spinup_ad \
        --cime-job-id 2002.desched1 --cime-job-id 2003.desched1
    meristem run advance --root /path/to/run --stage spinup_ad
    meristem run fail --root /path/to/run --stage spinup_ad
    meristem run retry --root /path/to/run [--stage spinup_ad] [--skip-script] [--lost]
    meristem run status --root /path/to/run
"""

import argparse
import sys
from pathlib import Path

from .model_run import ModelRun

# exit code from status when a stage is lost (1 is a runtime error, 2 an argument error)
EXIT_LOST = 3


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser.

    Returns:
        argparse.ArgumentParser: argument parser
    """
    parser = argparse.ArgumentParser(
        prog="meristem run",
        description="Drive a single ModelRun",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # parent parsers

    # debug command, used by all subcommands
    debug = argparse.ArgumentParser(add_help=False)
    debug.add_argument(
        "--debug", action="store_true", help="Show the full traceback on errors."
    )

    # all subcommands (except create) use this
    root = argparse.ArgumentParser(add_help=False)
    root.add_argument("--root", required=True, help="Run root directory.")

    # dry-run command, used by submit, submit-advance, submit-fail, advance, and retry
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

    # meristem run create CONFIG [--debug]
    create = subparsers.add_parser(
        "create",
        parents=[debug],
        help="Initialize a new run from a YAML config",
    )
    create.add_argument("--config", required=True, help="Path to run.yaml")
    create.set_defaults(handler=cmd_create)

    # meristem run submit --root R [--dry-run] [--debug]
    submit = subparsers.add_parser(
        "submit",
        parents=[debug, root, dry_run],
        help="Kick off the first stage.",
    )
    submit.set_defaults(handler=cmd_submit)

    # meristem run record-cime-jobs --root R --stage S --job-ids JOB_IDS [--debug]
    record_cime_jobs = subparsers.add_parser(
        "record-cime-jobs",
        parents=[debug, root],
        help="Record the CIME jobs case.submit queued for a stage.",
    )
    record_cime_jobs.add_argument("--stage", required=True, help="Stage name.")
    record_cime_jobs.add_argument(
        "--job-ids",
        required=True,
        help="The case's JOB_IDS, e.g. 'case.run:123, case.st_archive:124'.",
    )
    record_cime_jobs.add_argument(
        "--attempt",
        type=int,
        help="Attempt of the stage the calling job belongs to; a call from an "
        "older attempt is ignored. Set by meristem's job scripts.",
    )
    record_cime_jobs.set_defaults(handler=cmd_record_cime_jobs)

    # meristem run submit-advance --root R --stage S --cime-job-id ID [--debug] [--dry-run]
    submit_advance = subparsers.add_parser(
        "submit-advance",
        parents=[debug, root, dry_run],
        help="Write and submit the advance job.",
    )
    submit_advance.add_argument("--stage", required=True, help="Stage name.")
    submit_advance.add_argument(
        "--cime-job-id",
        required=True,
        help="CIME job scheduler ID.",
    )
    submit_advance.add_argument(
        "--attempt",
        type=int,
        help="Attempt of the stage the calling job belongs to; a call from an "
        "older attempt is ignored. Set by meristem's job scripts.",
    )
    submit_advance.set_defaults(handler=cmd_submit_advance)

    # meristem run submit-fail --root R --stage S --cime-job-id ID [--debug] [--dry-run]
    submit_fail = subparsers.add_parser(
        "submit-fail",
        parents=[debug, root, dry_run],
        help="Write and submit the fail job.",
    )
    submit_fail.add_argument("--stage", required=True, help="Stage name.")
    submit_fail.add_argument(
        "--cime-job-id",
        required=True,
        action="append",
        help="CIME job scheduler ID.",
    )
    submit_fail.add_argument(
        "--attempt",
        type=int,
        help="Attempt of the stage the calling job belongs to; a call from an "
        "older attempt is ignored. Set by meristem's job scripts.",
    )
    submit_fail.set_defaults(handler=cmd_submit_fail)

    # meristem run advance --root R --stage S [--debug] [--dry-run]
    advance = subparsers.add_parser(
        "advance",
        parents=[debug, root, dry_run],
        help="Mark stage done and submit the next stage.",
    )
    advance.add_argument("--stage", required=True, help="Stage name.")
    advance.add_argument(
        "--attempt",
        type=int,
        help="Attempt of the stage the calling job belongs to; a call from an "
        "older attempt is ignored. Set by meristem's job scripts.",
    )
    advance.set_defaults(handler=cmd_advance)

    # meristem run fail --root R --stage S [--debug]
    fail = subparsers.add_parser(
        "fail", parents=[debug, root], help="Mark stage as failed."
    )
    fail.add_argument("--stage", required=True, help="Stage name.")
    fail.add_argument(
        "--attempt",
        type=int,
        help="Attempt of the stage the calling job belongs to; a call from an "
        "older attempt is ignored. Set by meristem's job scripts.",
    )
    fail.set_defaults(handler=cmd_fail)

    # meristem run retry --root R [--stage S] [--debug] [--dry-run] [--skip-script] [--lost]
    retry = subparsers.add_parser(
        "retry",
        parents=[debug, root, dry_run],
        help="Reset and resubmit a pending or failed stage.",
    )
    retry.add_argument("--stage", required=False, help="Optional stage name.")
    retry.add_argument(
        "--skip-script",
        action="store_true",
        help="Resubmit without re-running the setup script",
    )
    retry.add_argument(
        "--lost",
        action="store_true",
        help=(
            "Retry a submitted stage that is lost (none of its jobs "
            "are still queued or running; see meristem run status)"
        ),
    )
    retry.set_defaults(handler=cmd_retry)

    # meristem run status --root R [--debug]
    status = subparsers.add_parser(
        "status",
        parents=[debug, root],
        help=(
            "Query the scheduler and print run status, showing lost stages "
            f"(changes nothing; exits {EXIT_LOST} if a stage is lost)"
        ),
    )
    status.set_defaults(handler=cmd_status)

    return parser


def _print_job_id(label: str, job_id: str | None) -> None:
    """Print a submitted job ID, if there is one

    Args:
        label (str): message to print before the job ID
        job_id (str | None): PBS job ID, or None if nothing was submitted
    """
    if job_id:
        print(f"{label}: {job_id}")


def cmd_create(args: argparse.Namespace) -> int:
    """Execute create action and return a successful exit code

    Args:
        args (argparse.Namespace): parsed namespace from argparse

    Returns:
        int: 0 on success
    """
    run = ModelRun.create(args.config)
    print(f"Initialised run '{run.run_id}' at {run.root}")
    print(f"Next step:  meristem run submit --root {run.root}")
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    """Execute submit action and return a successful exit code

    Args:
        args (argparse.Namespace): parsed namespace from argparse

    Returns:
        int: 0 on success
    """
    run = ModelRun.load(Path(args.root))
    _print_job_id("Submitted", run.submit(dry_run=args.dry_run))
    return 0


def cmd_record_cime_jobs(args: argparse.Namespace) -> int:
    """Execute record-cime-jobs action and return a successful exit code

    Args:
        args (argparse.Namespace): parsed namespace from argparse

    Returns:
        int: 0 on success
    """
    run = ModelRun.load(Path(args.root))
    run.record_cime_jobs(args.stage, args.job_ids, attempt=args.attempt)
    return 0


def cmd_submit_advance(args: argparse.Namespace) -> int:
    """Execute submit-advance action and return a successful exit code

    Args:
        args (argparse.Namespace): parsed namespace from argparse

    Returns:
        int: 0 on success
    """
    run = ModelRun.load(Path(args.root))
    _print_job_id(
        "Advance job submitted",
        run.submit_advance(
            args.stage, args.cime_job_id, dry_run=args.dry_run, attempt=args.attempt
        ),
    )
    return 0


def cmd_submit_fail(args: argparse.Namespace) -> int:
    """Execute submit-fail action and return a successful exit code

    Args:
        args (argparse.Namespace): parsed namespace from argparse

    Returns:
        int: 0 on success
    """
    run = ModelRun.load(Path(args.root))
    job_ids = run.submit_fail(
        args.stage, args.cime_job_id, dry_run=args.dry_run, attempt=args.attempt
    )
    print("\n".join(f"Fail job submitted: {job_id}" for job_id in job_ids))
    return 0


def cmd_advance(args: argparse.Namespace) -> int:
    """Execute advance action and return a successful exit code

    Args:
        args (argparse.Namespace): parsed namespace from argparse

    Returns:
        int: 0 on success
    """
    run = ModelRun.load(Path(args.root))
    _print_job_id(
        "Next stage submitted",
        run.advance(args.stage, dry_run=args.dry_run, attempt=args.attempt),
    )
    return 0


def cmd_fail(args: argparse.Namespace) -> int:
    """Execute fail action and return a successful exit code

    Args:
        args (argparse.Namespace): parsed namespace from argparse

    Returns:
        int: 0 on success
    """
    run = ModelRun.load(Path(args.root))
    run.fail(args.stage, attempt=args.attempt)
    return 0


def cmd_retry(args: argparse.Namespace) -> int:
    """Execute retry action and return a successful exit code

    Args:
        args (argparse.Namespace): parsed namespace from argparse

    Returns:
        int: 0 on success
    """
    run = ModelRun.load(Path(args.root))
    _print_job_id(
        "Resubmitted",
        run.retry(
            args.stage,
            dry_run=args.dry_run,
            skip_script=args.skip_script,
            lost=args.lost,
        ),
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Execute retry action and return an exit code

    Args:
        args (argparse.Namespace): parsed namespace from argparse

    Returns:
        int: 0 on success, EXIT_LOST if any lost stages
    """
    run = ModelRun.load(Path(args.root))
    lost = run.lost_stages()
    print(run.format_status(lost=lost))
    for name in lost:
        print(
            f"Stage {name!r} is lost. Retry with: "
            f"meristem run retry --root {run.root} --stage {name} --lost"
        )
    if lost:
        return EXIT_LOST
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
