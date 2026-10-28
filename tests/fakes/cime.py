"""Fake CIME calls"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import os
import sys
import subprocess

CASE_DICT_FILE = "case.json"
FAULTS_DICT_FILE = "fake_cime.json"
RUN_LOG_FILE = "runs.log"
ACCEPTED_KEYS = (
    "CASE",
    "BUILD_COMPLETE",
    "CIMEROOT",
    "JOB_IDS",
    "PROJECT",
    "DOUT_S_ROOT",
)


def create_newcase(case_root: Path) -> int:
    """Fake create_newcase
    Creates the case_root and writes exectuables that are wrappers
    bake to fake/cime.py. Like CIME, refuses a case_root that already exists

    Args:
        case_root (Path): path to case_root

    Returns:
        int: return code
    """
    if case_root.exists():
        print(f"create_newcase: {case_root} already exists", file=sys.stderr)
        return 1
    case_root.mkdir(parents=True)

    faults = json.loads(os.environ.get("FAKE_CIME", "{}"))
    (case_root / FAULTS_DICT_FILE).write_text(
        json.dumps(faults, indent=2), encoding="utf-8"
    )

    for file in (
        "case.submit",
        "xmlquery",
        "xmlchange",
        "case.setup",
        "preview_namelists",
    ):
        _write_case_wrapper(case_root, file)

    env_case_file = case_root / "env_case.xml"
    env_case_file.write_text("env_case placeholder\n", encoding="utf-8")

    user_nl_clm_file = case_root / "user_nl_clm"
    user_nl_clm_file.write_text("! user_nl_clm\n", encoding="utf-8")

    case_dict = {
        "CASE": case_root.name,
        "BUILD_COMPLETE": False,
        "CIMEROOT": "",
        "JOB_IDS": None,
        "DOUT_S_ROOT": str(case_root / "archive"),
    }
    (case_root / CASE_DICT_FILE).write_text(
        json.dumps(case_dict, indent=2), encoding="utf-8"
    )
    return 0


def create_clone(args: list[str]) -> int:
    """Fake create_clone: make a new case from an existing one

    Args:
        args (list[str]): arguments as passed by Ensemble._clone

    Returns:
        int: return code
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--clone", required=True)
    parser.add_argument("--keepexe", action="store_true")
    parser.add_argument("--project")
    parser.add_argument("--cime-output-root")
    parsed = parser.parse_args(args)

    if json.loads(os.environ.get("FAKE_CIME", "{}")).get("create_clone") == "fail":
        print("create_clone: injected failure", file=sys.stderr)
        return 1

    case_root = Path(parsed.case)
    clone_root = Path(parsed.clone)
    if case_root.exists():
        print(f"create_clone: {case_root} already exists", file=sys.stderr)
        return 1

    create_newcase(case_root)
    (case_root / "user_nl_clm").write_text(
        (clone_root / "user_nl_clm").read_text(encoding="utf-8"), encoding="utf-8"
    )
    base = json.loads((clone_root / CASE_DICT_FILE).read_text(encoding="utf-8"))
    case_dict = json.loads((case_root / CASE_DICT_FILE).read_text(encoding="utf-8"))
    case_dict["BUILD_COMPLETE"] = base["BUILD_COMPLETE"]
    case_dict["CIMEROOT"] = base["CIMEROOT"]
    case_dict["PROJECT"] = parsed.project
    (case_root / CASE_DICT_FILE).write_text(
        json.dumps(case_dict, indent=2), encoding="utf-8"
    )
    return 0


def xmlquery(case_root: Path, args: list[str]) -> int:
    """Fake xmlquery

    Args:
        case_root (Path): path to case_root
        args (list[str]): args to query, can be 'key' or '--value key'

    Returns:
        int: exit code
    """

    faults = _faults(case_root)

    parser = argparse.ArgumentParser(description="Internal XML Query Parser")
    parser.add_argument("--value", action="store_true")
    parser.add_argument("key")
    parsed = parser.parse_args(args)

    key = parsed.key

    if key not in ACCEPTED_KEYS:
        print(f"Unknown xmlquery variable {key}")
        print(f"Accepted keys: {ACCEPTED_KEYS}")
        return 1

    if faults.get("job_ids") == "" and key == "JOB_IDS":
        print("")
        return 0

    if not (case_root / CASE_DICT_FILE).is_file():
        print(
            f"Cannot find {CASE_DICT_FILE} in {case_root} " "Run create_newcase first"
        )
        return 1

    case_dict = json.loads((case_root / CASE_DICT_FILE).read_text(encoding="utf-8"))
    value = case_dict.get(key)
    if value is None:
        print(f"xml variable {key} not in {case_root} / case.json")
        return 1
    print(value)
    return 0


def case_submit(case_root: Path, args: list[str]) -> int:
    """Fake case_submit

    Args:
        case_root (Path): path to case root
        args (list[str]): input arguments (ignored here)

    Returns:
        int: return code
    """

    faults = _faults(case_root)

    if faults.get("case.submit") == "fail":
        print(f"case_submit: injected failure for {case_root}", file=sys.stderr)
        return 1

    parser = argparse.ArgumentParser(description="Internal case_submit parser")
    _, _ = parser.parse_known_args(args)

    if not (case_root / CASE_DICT_FILE).is_file():
        print(
            f"Cannot find {CASE_DICT_FILE} in {case_root} " "Run create_newcase first"
        )
        return 1

    run_log = case_root / RUN_LOG_FILE

    run_fail = faults.get("case.run") == "fail"
    run_script = _write_job_script(
        case_root, "case.run", run_log, exit_nonzero=run_fail
    )

    job_ids = [f"case.run:{_qsub(run_script)}"]
    if not faults.get("no_archive"):
        run_id = job_ids[0].split(":", 1)[1]

        archive_fail = faults.get("st_archive") == "fail"
        archive_script = _write_job_script(
            case_root, "case.st_archive", run_log, exit_nonzero=archive_fail
        )
        job_ids.append(f"case.st_archive:{_qsub(archive_script, depend=run_id)}")

    case_dict = json.loads((case_root / CASE_DICT_FILE).read_text(encoding="utf-8"))
    case_dict["JOB_IDS"] = ", ".join(job_ids)
    (case_root / CASE_DICT_FILE).write_text(
        json.dumps(case_dict, indent=2), encoding="utf-8"
    )
    return 0


def _qsub(script: Path, depend: str | None = None) -> str:
    """Submit script with the fake qsub and return its job ID."""
    cmd = ["qsub"]
    if depend:
        cmd += ["-W", f"depend=afterok:{depend}"]
    cmd.append(str(script))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"qsub failed for {script}: {result.stderr.strip()}")
    return result.stdout.strip()


def _write_job_script(
    case_root: Path, job: str, run_log: Path, exit_nonzero: bool = False
) -> Path:
    """Write a fake case.run / case.st_archive job script."""
    name = case_root.name

    file = case_root / job
    job_name = job.replace("case", name)

    lines = [
        "#!/bin/bash",
        f"#PBS -N {job_name}",
        f"#PBS -o {case_root}/{name}.{job}.log",
        f"echo {job} {name} >> {run_log}",
    ]
    if exit_nonzero:
        lines.append("exit 1")
    file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return file


def preview_namelists(case_root: Path) -> int:
    """Fake preview_namelists: write CaseDocs/lnd_in from user_nl_clm"""
    docs = case_root / "CaseDocs"
    docs.mkdir(exist_ok=True)
    text = (case_root / "user_nl_clm").read_text(encoding="utf-8")
    (docs / "lnd_in").write_text(text, encoding="utf-8")
    return 0


def xmlchange(case_root: Path, args: list[str]) -> int:
    """Fake xmlchange. Just a pass through for now.

    Args:
        case_root (Path): path to case root
        args (list[str]): input args

    Returns:
        int: return code
    """
    # pylint: disable=unused-argument
    return 0


def case_setup(case_root: Path) -> int:  # pylint: disable=unused-argument
    """Fake case_setup. Just a pass through for now.

    Args:
        case_root (Path): path to case root

    Returns:
        int: return code
    """
    return 0


def _write_case_wrapper(case_root: Path, command: str) -> None:
    """Write an executable in case_root that calls back into this file."""
    body = (
        "#!/bin/sh\n"
        'CASE_DIR="$(cd "$(dirname "$0")" && pwd)"\n'
        f'exec {sys.executable} {Path(__file__).resolve()} {command} "$CASE_DIR" "$@"\n'
    )
    path = case_root / command
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _faults(case_root: Path):
    """Return faults file dict

    Returns:
        dict: faults dict
    """
    faults_file = case_root / FAULTS_DICT_FILE
    if not faults_file.is_file():
        return {}
    return json.loads(faults_file.read_text(encoding="utf-8"))


COMMANDS = {
    "create_newcase": create_newcase,
    "create_clone": create_clone,
    "xmlquery": xmlquery,
    "xmlchange": xmlchange,
    "case.setup": case_setup,
    "case.submit": case_submit,
    "preview_namelists": preview_namelists,
}


def main():
    """Main call using the fake cime"""

    name = sys.argv[1]
    case_root = Path(sys.argv[2])
    if name not in COMMANDS:
        print(f"fake cime: unknown command {name!r}", file=sys.stderr)
        sys.exit(2)

    if name == "create_clone":
        sys.exit(create_clone(sys.argv[2:]))

    if name in ("create_newcase", "case.setup", "preview_namelists"):
        code = COMMANDS[name](case_root)
    else:
        code = COMMANDS[name](case_root, sys.argv[3:])
    sys.exit(code)


if __name__ == "__main__":  # pragma: no cover
    main()
