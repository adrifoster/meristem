"""A fake PBS class"""

from __future__ import annotations

import os
import json
import subprocess
import sys
from pathlib import Path


class FakePbs:
    """Fake PBS class"""

    def __init__(self, root: Path):
        """Initialize Fakepbs

        Args:
            root (Path): root directory
        """

        self.root = root
        self.bin_dir = self.root / "bin_dir"
        self.spool = self.root / "spool"

        self.env = dict(os.environ)
        self.env["PATH"] = f"{self.bin_dir}{os.pathsep}{self.env['PATH']}"
        self.env["FAKE_PBS_SPOOL"] = str(self.spool)

        self.root.mkdir(parents=True, exist_ok=True)
        for directory in (self.bin_dir, self.spool):
            directory.mkdir(parents=True, exist_ok=True)

        pbs_path = Path(__file__).parent / "pbs.py"
        for command in ("qsub", "qstat", "qdel"):
            _create_file(self.root / "bin_dir", pbs_path, command)

        cime_path = Path(__file__).parent / "cime.py"
        self.cimeroot = self.root / "cimeroot"
        (self.cimeroot / "scripts").mkdir(parents=True, exist_ok=True)
        _create_file(self.cimeroot / "scripts", cime_path, "create_clone")
        _create_file(
            self.root / "bin_dir",
            cime_path,
            "create_newcase",
            file_name="fake_create_newcase",
        )

    @property
    def control_file(self) -> Path:
        """path to control file"""
        return self.spool / "control.json"

    @property
    def json_file(self) -> Path:
        """path to jobs.json file"""
        return self.spool / "jobs.json"

    def fail_qsub(self, pattern: str):
        """Inject a qsub failure into the control file

        Args:
            pattern (str): script name pattern
        """
        control = self.control()
        control["qsub_fail"] = pattern
        self.control_file.write_text(json.dumps(control, indent=2), encoding="utf-8")

    def break_qstat(self):
        """Inject a qstat failure into the control file"""
        control = self.control()
        control["qstat_broken"] = True
        self.control_file.write_text(json.dumps(control, indent=2), encoding="utf-8")

    def jobs(self) -> list:
        """Return a loaded jobs list

        Returns:
            list: list of job dicts
        """
        path = self.json_file
        if not path.is_file():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def control(self) -> dict:
        """Loads the control file

        Returns:
            dict: control file dict
        """
        path = self.control_file
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def logs(self) -> str:
        """Dumps all logs on the spool

        Returns:
            str: dump of all logs
        """
        jobs = self.jobs()

        lines = []
        for job in jobs:
            log_path = self._log_path(job)
            if not log_path.is_file():
                log_text = "(no log)"
            else:
                log_text = log_path.read_text(encoding="utf-8")

            lines.append(
                f"=== {job['id']} ({job['name']}) state={job['state']} exit={job['exit']} ==="
            )
            lines.append(log_text)
        return "\n".join(lines)

    def step(self) -> bool:
        """Steps through one iteration of running jobs on the spool

        Returns:
            bool: True if something was done; False if no action was taken
        """
        job, action = self._next_action(self.jobs())
        if job is None:
            return False
        if action == "run":
            self._run_job(job)
        elif action == "delete":
            self._update_job(_number(job["id"]), state="F", deleted=True)
        return True

    def run_until_idle(self, max_jobs=200):
        """Runs step() until max_jobs exceeded or no jobs left

        Args:
            max_jobs (int, optional): maximum number of jobs to run. Defaults to 200.

        Raises:
            RuntimeError: max_jobs exceeded
        """

        num_jobs = 0
        while self.step():
            num_jobs += 1
            if num_jobs > max_jobs:
                raise RuntimeError(
                    f"more than {max_jobs} jobs run or deleted; the chain may be looping"
                )

    def kill(self, job_id: str):
        """Kill job_id

        Args:
            job_id (str): id of job to kill
        """
        self._update_job(_number(job_id), state="F", exit=271)

    def _update_job(self, number: str, **fields):
        """Reload the spool, set fields on one job, and save

        Args:
            number (str): job number
        """
        jobs = self.jobs()
        job = next(j for j in jobs if _number(j["id"]) == number)
        job.update(fields)
        self.json_file.write_text(json.dumps(jobs, indent=2), encoding="utf-8")

    def _log_path(self, job: dict) -> Path:
        """Get path to the log file of the job

        Args:
            job (dict): job dict

        Returns:
            Path: path to log file
        """
        return (
            Path(job["log"]) if job["log"] else self.spool / f"{_number(job["id"])}.log"
        )

    def _run_job(self, job: dict):
        """Run the job

        Args:
            job (dict): job dict
        """
        number = _number(job["id"])
        log_path = self._log_path(job)
        self._update_job(number, state="R")

        cwd = self.root / "scratch" / number
        cwd.mkdir(parents=True, exist_ok=True)
        with open(log_path, mode="w", encoding="utf-8") as f:
            result = subprocess.run(
                ["bash", job["script"]],
                cwd=cwd,
                env=self.env,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=False,
            )

        self._update_job(number, exit=result.returncode, state="F")

    def _next_action(self, jobs: list[dict]) -> tuple[dict, str] | tuple[None, None]:
        """Return (job, action) for the first queued job that can act, else (None, None).

        Args:
            jobs (list[dict]): list of job dicts

        Returns:
            tuple[dict, str] | tuple[None, None]: job, action
        """

        by_number = {_number(j["id"]): j for j in jobs}
        for job in jobs:
            if job["state"] != "Q":
                continue
            dep = None
            if job["depend"] is not None:
                dep = by_number.get(job["depend"][1].split(".", 1)[0])
            action = _decide(job, dep)
            if action != "wait":
                return job, action
        return None, None


def _decide(job: dict, dep: dict | None) -> str:
    """Check the job state and dependency state and decide
    what to do with the job

    Args:
        job (dict): job
        dep (dict | None): dependency job

    Raises:
        ValueError: job depends on a job that doesn't exist
        ValueError: unexpected job dependency state

    Returns:
        str: _description_
    """
    if job["depend"] is None:
        return "run"
    if dep is None:
        raise ValueError(
            f"{job['id']} depends on {job['depend']} but that job doesn't exist"
        )
    if dep["deleted"]:
        return "delete"
    if dep["state"] in ["Q", "R"]:
        return "wait"
    if dep["state"] == "F":
        dep_type, _ = job["depend"]
        succeeded = dep["exit"] == 0
        wanted_success = dep_type == "afterok"
        return "run" if succeeded == wanted_success else "delete"
    raise ValueError(f"unexpected dependency state {dep['state']!r}")


def _number(job_id: str) -> str:
    """Returns job number substring from full job id

    Args:
        job_id (str): full job_id

    Returns:
        str: job number
    """
    return job_id.split(".", 1)[0]


def _create_file(
    root_dir: Path, fake_path: Path, command: str, file_name: str | None = None
):
    """Create an executable file

    Args:
        root_dir (Path): path to root
        fake_path (Path): path to where fake scripts are
        command (str): command
        file_name (optional, str): optional file name. Defaults to command
    """
    file = root_dir / command if not file_name else root_dir / file_name

    lines = ["#!/bin/sh", f'exec {sys.executable} {str(fake_path)} {command} "$@"']

    file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    file.chmod(0o755)
