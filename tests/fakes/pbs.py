"""A fake PBS scheduler"""

from __future__ import annotations

import re
import os
import json
import sys
from pathlib import Path


def _spool() -> Path:
    """Returns a path from input env var

    Returns:
        Path: path
    """
    return Path(os.environ["FAKE_PBS_SPOOL"])


def _load() -> list:
    """Returns the list in json_file, or [] if the file doesn't exist yet.

    Returns:
        list: list in json_file
    """
    json_file = _spool() / "jobs.json"
    if not json_file.is_file():
        return []
    return json.loads(json_file.read_text(encoding="utf-8"))


def _save(jobs: list):
    """Writes the job_list to json

    Args:
        jobs (list): job list

    """
    json_file = _spool() / "jobs.json"
    with open(json_file, "w", encoding="utf-8") as file:
        json.dump(jobs, file, indent=2)


def _control() -> dict:
    """Return control file dict

    Returns:
        dict: control dict
    """
    control_file = _spool() / "control.json"
    if not control_file.is_file():
        return {}
    return json.loads(control_file.read_text(encoding="utf-8"))


def _get_name(text: str) -> str | None:
    """Parses text to get a PBS job name

    Args:
        text (str): input text

    Returns:
        str | None: PBS job name
    """
    m = re.search(r"^#PBS -N (\S+)", text, re.MULTILINE)
    return m.group(1) if m else None


def _get_log(text: str) -> str | None:
    """Parses text to get a PBS log file

    Args:
        text (str): input text

    Returns:
        str | None: log file
    """
    m = re.search(r"^#PBS -o (\S+)", text, re.MULTILINE)
    return m.group(1) if m else None


def _get_depend(text: str) -> tuple[str, str] | None:
    """Parses text to get a depends tuple

    Args:
        text (str): input text

    Returns:
        tuple[str, str] | None: [depend_type, job_id]
    """
    m = re.search(r"depend=(afterok|afternotok):(\S+)", text, re.MULTILINE)
    if m:
        return m.group(1), m.group(2)
    return None


def qsub(argv: list[str]) -> int:
    """Mock qsub

    Args:
        argv (list[str]): input argument

    Returns:
        int: 0 on success; 1 on injected failure or bad depend
    """

    jobs = _load()
    script = argv[-1]
    pattern = _control().get("qsub_fail")
    if pattern and re.search(pattern, script):
        print(f"qsub: injected failure for {script}", file=sys.stderr)
        return 1

    text = Path(script).read_text(encoding="utf-8")
    job_id = f"{len(jobs) + 1}.fakepbs"

    depend = _get_depend(" ".join(argv[:-1])) or _get_depend(text)

    everything = " ".join(argv[:-1]) + "\n" + text
    if "depend=" in everything and depend is None:
        print(f"qsub: could not parse dependency in {script}", file=sys.stderr)
        return 1

    jobs.append(
        {
            "id": job_id,
            "script": script,
            "name": _get_name(text),
            "log": _get_log(text),
            "depend": depend,
            "state": "Q",
            "exit": None,
            "deleted": False,
        }
    )
    _save(jobs)
    print(job_id)
    return 0


def qstat(argv: list[str]) -> int:
    """Mock qstat

    Args:
        argv (list[str]): input argument

    Returns:
        int: return code: 0 for success; 153 for missing; 2 for injected failure
    """
    if _control().get("qstat_broken"):
        print("qstat: injected failure", file=sys.stderr)
        return 2

    ids = [a for a in argv if not a.startswith("-")]
    by_number = {job["id"].split(".", 1)[0]: job for job in _load()}

    missing = False
    for requested in ids:
        number = requested.split(".", 1)[0]
        job = by_number.get(number)
        if job is None:
            missing = True
            continue

        print(f"Job Id: {job['id']}")
        print(f"    job_state = {job['state']}")
    return 153 if missing else 0


def qdel(argv: list[str]) -> int:
    """Mock qdel

    Args:
        argv (list[str]): input argument

    Returns:
        int: return code: 0 for success; 153 for missing
    """
    ids = [a for a in argv if not a.startswith("-")]

    jobs = _load()
    by_number = {job["id"].split(".", 1)[0]: job for job in jobs}

    missing = False
    for requested in ids:
        number = requested.split(".", 1)[0]
        job = by_number.get(number)
        if job is None:
            missing = True
            continue
        job["state"] = "F"
        job["deleted"] = True
        job["exit"] = 271
    _save(jobs)
    return 153 if missing else 0


COMMANDS = {"qsub": qsub, "qstat": qstat, "qdel": qdel}


def main():
    """Main call using the fake pbs"""

    name = sys.argv[1]
    if name not in COMMANDS:
        print(f"fake pbs: unknown command {name!r}", file=sys.stderr)
        sys.exit(2)
    sys.exit(COMMANDS[name](sys.argv[2:]))


if __name__ == "__main__":
    main()
