"""Class for managing model ensembles"""

from __future__ import annotations

import os
import re
import json
import shlex
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from dataclasses import replace
import io
from contextlib import redirect_stdout
from typing import TypeVar

import pandas as pd

from meristem.commands import meristem_cmd
from meristem.ensemble.config import EnsembleConfig
from meristem.run.stages import Stage, StageStatus
from meristem.run.model_run import _STATE_FILE as _RUN_STATE_FILE, ModelRun
from meristem.utils import (
    check_root_new,
    remove_root_if_empty,
    write_atomically,
    run_command,
)

T = TypeVar("T")

# state file name
_STATE_FILE = "ensemble_state.json"

# parameter file extensions each paramfile_key can read: CLM only reads
# netCDF, FATES reads JSON or (older versions) netCDF
_PARAM_EXTENSIONS = {
    "paramfile": (".nc",),
    "fates_paramfile": (".nc", ".json"),
}
_ALL_PARAM_EXTENSIONS = (".nc", ".json")

# status order for printing
_STATUS_ORDER = [s.value for s in StageStatus] + ["lost"]


class Ensemble:
    """Class for model ensemble management

    Use Ensemble.create() to initialize a new ensemble on disk, or
    Ensemble.load() to resume an existing one.

    Attributes
    ===========
    config: EnsembleConfig
        Ensemble config with all attributes
    param_files: dict[str, Path]
        dictionary mapping of member string to full path to parameter file
    """

    def __init__(
        self, config: EnsembleConfig, param_files: dict[str, Path] | None = None
    ):
        self.config = config
        self.param_files: dict[str, Path] = param_files or {}

    @property
    def root(self) -> Path:
        """Directory for ensemble and member state, PBS scripts and logs"""
        return self.config.root

    @property
    def base_case(self) -> Path:
        """Directory with built CIME case that every member is cloned from"""
        return self.config.base_case

    @property
    def case_dir(self) -> Path:
        """Directory where member clones are created"""
        return self.config.case_dir

    @property
    def stage_names(self) -> list[str]:
        """Names of the member stages, in order"""
        return [s.name for s in self.config.member_stages]

    @classmethod
    def create(cls, config: EnsembleConfig | dict | Path | str) -> Ensemble:
        """Initialize a brand-new ensemble from a YAML file, config dict, or
        EnsembleConfig.

        Creates root on disk and writes the initial state file.
        Call this once; use load for all subsequent access.

        Args:
            config: path to an ensemble YAML file, a dict with the same schema,
                or an EnsembleConfig

        Returns:
            Ensemble: Ensemble instance
        Raises:
            ValueError: if required config keys are missing or invalid
            TypeError: if config is not an EnsembleConfig, dict, str, or Path
            FileExistsError: if a state file already exists at the target root,
            indicating that the ensemble has already been initialized
            FileExistsError: if root already exists (even empty), since a whole
            clean deletes root
            FileExistsError: if any member stage's case directory already exists
            (e.g. left in a shared case_dir by an ensemble with the same
            ensemble_id), since prepare would reuse it as this member's clone
        """
        ensemble = cls(_to_config(config))

        state_path = ensemble.root / _STATE_FILE
        if state_path.exists():
            raise FileExistsError(
                f"A state file already exists at {state_path}. "
                "Use Ensemble.load() to resume an existing ensemble."
            )
        # a whole clean deletes root, so root must be one meristem makes
        check_root_new(ensemble.root)

        ensemble.param_files = ensemble._find_param_files()
        ensemble._check_base_case()
        ensemble._check_stage_scripts()
        member_runs = ensemble._member_runs()
        ensemble._check_case_roots_free()

        ensemble.root.mkdir(parents=True)
        for run in member_runs:
            run.root.mkdir(parents=True, exist_ok=True)
            run.save()
        ensemble.save()
        return ensemble

    @classmethod
    def load(cls, root: Path | str) -> Ensemble:
        """Load an existing ensemble from its state file.

        Args:
            root (Path | str): Path to root

        Returns:
            Ensemble: Ensemble instance
        Raises:
            FileNotFoundError: if no state file exists at root
            ValueError: if the ensemble was created somewhere else (i.e. moved)
        """
        state_path = Path(root) / _STATE_FILE
        if not state_path.exists():
            raise FileNotFoundError(
                f"No state file found at {state_path}. "
                "Use Ensemble.create() to initialise a new ensemble."
            )
        state = json.loads(state_path.read_text(encoding="utf-8"))
        config = EnsembleConfig.from_dict(state["config"])

        # absolute paths under the original root are saved in this state file,
        # every member's run state, the job scripts, and the cases, so a moved
        # ensemble would keep using the old location
        if Path(root).resolve() != config.root.resolve():
            raise ValueError(
                f"ensemble at {root} was created at {config.root}; ensembles "
                "can't be moved, so create it again at the new location."
            )

        param_files = {member: Path(path) for member, path in state["members"].items()}
        return cls(config, param_files)

    def submit(
        self, limit: int | None = None, dry_run: bool = False
    ) -> dict[str, str | None]:
        """Submit the first stage of every pending member

        A member is pending if its first stage has never been submitted.

        Args:
            limit (int | None, optional): submit at most this many pending members.
                Defaults to None (all).
            dry_run (bool, optional): write job scripts but don't submit.
                Defaults to False.

        Raises:
            ValueError: if limit is less than 1
            FileNotFoundError: if any selected member's parameter file is missing
            RuntimeError: if submitting any member failed (e.g. qsub refused it),
                naming each; every other member is still submitted

        Returns:
            dict[str, str]: member name -> job ID of its first stage's setup job
        """
        if limit is not None and limit < 1:
            raise ValueError(f"limit must be greater than 0, got {limit}")

        runs = {member: self.member_run(member) for member in self.param_files}
        pending = [
            member
            for member, run in runs.items()
            if run.stages[0].state.status == StageStatus.PENDING
        ]

        if limit is not None:
            pending = pending[:limit]

        if not pending:
            print(f"{self.config.ensemble_id}: no pending members")
            return {}

        self._check_param_files(pending)

        with redirect_stdout(io.StringIO()):
            job_ids, errors = _for_each_member(
                pending, lambda member: runs[member].submit(dry_run=dry_run)
            )
        if dry_run:
            print(
                f"{self.config.ensemble_id}: dry run, wrote job scripts for "
                f"{len(job_ids)} members in {self.root / 'members'}"
            )
        else:
            print(f"{self.config.ensemble_id}: submitted {len(job_ids)} members")
        _raise_member_errors(self.config.ensemble_id, "submit", errors, len(pending))

        return job_ids

    def retry(
        self,
        failed: bool = False,
        lost: bool = False,
        members: list[str] | None = None,
        stage: str | None = None,
        skip_script: bool = False,
        dry_run: bool = False,
    ) -> dict[str, str | None]:
        """Retry the selected stages

        Args:
            failed (bool, optional): Retry all failed members. Defaults to False.
            lost (bool, optional): Retry lost members. Defaults to False.
            members (list[str] | None, optional): list of members to retry. Defaults to None.
            stage (str | None, optional): stage name to retry. Defaults to None.
            skip_script (bool, optional): skip the script call and just try resubmitting
            dry_run (bool, optional): only write pbs scripts and do not submit.
            Defaults to False.

        Raises:
            ValueError: unknown member stage
            ValueError: nothing selected to retry
            RuntimeError: if retrying any member failed (e.g. its jobs are still
                queued, or qsub refused it), naming each; every other member is
                still retried

        Returns:
            dict[str, str | None]: each resubmitted member -> job ID of its retried
            stage's setup job (a DRY_ placeholder for a dry run); empty if nothing
            needed retrying
        """

        explicit = not failed and not lost
        if explicit and members is None and stage is None:
            raise ValueError(
                "nothing selected to retry; pass failed=True and/or lost=True "
                "(--failed / --lost), or name what to resubmit with members=... "
                "and/or stage=... (--member / --stage)."
            )
        if stage is not None:
            self._check_stages([stage])

        if members is not None:
            self._check_members(members)
        selected = list(self.param_files) if members is None else members

        runs = {member: self.member_run(member) for member in selected}
        to_retry = self._find_retries(runs, stage, lost, failed)
        if not to_retry:
            print(f"{self.config.ensemble_id}: nothing to retry")
            return {}
        self._check_param_files(list(to_retry))

        def _retry(member: str) -> str | None:
            name, was_lost = to_retry[member]
            return runs[member].retry(
                name, dry_run=dry_run, skip_script=skip_script, lost=was_lost
            )

        job_ids, errors = _for_each_member(list(to_retry), _retry)
        _raise_member_errors(self.config.ensemble_id, "retry", errors, len(to_retry))
        return job_ids

    def cancel(
        self, members: list[str] | None = None, dry_run: bool = False
    ) -> dict[str, list[str]]:
        """Cancel all (or some) ensemble members' current stages/jobs.
        Submitted stages are marked failed, so ensemble retry --failed resubmits them

        Args:
            members (list[str] | None, optional): Optional members to cancel.
                Defaults to None. If None, defaults to canceling all members.
            dry_run (bool, optional): Do not actually cancel, just print what
                job_ids would be canceled. Defaults to False.

        Raises:
            ValueError: unknown members
            RuntimeError: if canceling any member failed (e.g. qstat/qdel failed),
                naming each; every other member is still canceled

        Returns:
            dict[str, list[str]]: dictionary with member: job_ids
        """
        selected = self._select_members(members)
        with redirect_stdout(io.StringIO()):
            results, errors = _for_each_member(
                selected,
                lambda member: self.member_run(member).cancel(dry_run=dry_run),
            )
        job_ids = {member: ids for member, ids in results.items() if ids}

        if job_ids:
            n_jobs = sum(len(ids) for ids in job_ids.values())
            verb = "would cancel" if dry_run else "canceled"
            print(
                f"{self.config.ensemble_id}: {verb} {n_jobs} jobs in "
                f"{len(job_ids)} members ({_preview(list(job_ids))})"
            )
        elif not errors:
            print(f"{self.config.ensemble_id}: no queued or running jobs to cancel")
        _raise_member_errors(self.config.ensemble_id, "cancel", errors, len(selected))

        return job_ids

    def clean(
        self,
        members: list[str] | None = None,
        stages: list[str] | None = None,
        include_archive: bool = False,
        cancel: bool = False,
        dry_run: bool = False,
    ) -> list[Path]:
        """Clean (delete) directories and paths associated with input members and stages
        (or all if None supplied)

        Cleaning only some members or stages keeps each member's state, while a whole-ensemble
        clean deletes every member, then <root>/manifest.csv and the ensemble state
        file, then removes <root> only if nothing else is left in it (otherwise
        it's kept and what's left printed). Members already cleaned by a whole
        clean that failed partway are skipped, so rerunning it finishes. With cancel, the
        selected members' active jobs are canceled first and their submitted stages
        marked failed (as in cancel), so retry --failed can rerun them.

        Args:
            members (list[str] | None, optional): Optional members to clean.
                Defaults to None. If None, cleans all members.
            stages (list[str] | None, optional): Optional stages to clean.
                Defaults to None. If None, cleans all stages.
            include_archive (bool, optional): Also clean the archive (DOUT_S_ROOT)
                directories associated with these stages. Defaults to False.
                The run directory is always cleaned.
            cancel (bool, optional): If jobs are queued or running, cancel them first
                (marking submitted stages failed) instead of raising. Defaults to False.
            dry_run (bool, optional): Do not actually delete anything, just list what would
                be deleted. Defaults to False.

        Raises:
            ValueError: Unknown member stages
            ValueError: Some members still have jobs running or queued
            FileNotFoundError: a member's run directory is missing
            RuntimeError: qstat/qdel/xmlquery failed

        Returns:
            list[Path]: list of cleaned files/directories
        """
        selected = self._select_members(members)

        if stages is not None:
            self._check_stages(stages)

        whole = members is None and stages is None
        if whole:
            run_stages = None
            # a member whose state file is gone was finished by a whole clean that
            # then failed on a later member; skip it so rerunning clean can finish
            selected = [
                member
                for member in selected
                if (self.root / "members" / member / _RUN_STATE_FILE).is_file()
            ]
        else:
            run_stages = stages or self.stage_names

        targets: list[Path] = []
        blocked: list[str] = []
        with redirect_stdout(io.StringIO()):
            for member in selected:
                try:
                    targets += self.member_run(member).clean(
                        stages=run_stages,
                        include_archive=include_archive,
                        cancel=cancel,
                        dry_run=True,
                    )
                except ValueError as exc:
                    blocked.append(f"{member}: {exc}")

        if blocked:
            raise ValueError(
                _preview(blocked),
            )

        # the ensemble's own files; its state file goes last, so a clean that fails
        # partway can be rerun. root itself is removed only if it's then empty
        own = [self.root / "manifest.csv", self.root / _STATE_FILE] if whole else []
        own = [path for path in own if path.is_file()]
        targets += own

        if dry_run:
            return targets

        targets = []
        with redirect_stdout(io.StringIO()):
            for member in selected:
                targets += self.member_run(member).clean(
                    stages=run_stages,
                    include_archive=include_archive,
                    cancel=cancel,
                    dry_run=False,
                )
        for path in own:
            path.unlink(missing_ok=True)
        targets += own
        if whole:
            remove_root_if_empty(self.root)

        print(
            f"{self.config.ensemble_id}: cleaned {len(targets)} paths "
            f"in {len(selected)} members"
        )
        return targets

    def status(
        self, members: list[str] | None = None, check_queue: bool = False
    ) -> pd.DataFrame:
        """Get the status of every member stage

        Args:
            members (list[str] | None, optional): List of members to check.
                Defaults to None (all).
            check_queue (bool, optional): also query job scheduler and report lost stages.
                as 'lost'. Defaults to False.

        Raises:
            ValueError: if any member isn't in this ensemble

        Returns:
            pd.DataFrame: one row per member and stage, with columns member, stage,
            status, job_id, and attempts
        """

        rows = []
        if members is not None:
            self._check_members(members)
        selected = list(self.param_files) if members is None else members

        for member in selected:
            run = self.member_run(member)
            lost = set(run.lost_stages()) if check_queue else set()
            for run_stage in run.stages:
                name = run_stage.config.name
                rows.append(
                    {
                        "member": member,
                        "stage": name,
                        "status": (
                            "lost" if name in lost else run_stage.state.status.to_str()
                        ),
                        "job_id": run_stage.state.job_id,
                        "attempts": run_stage.state.attempts,
                        "case_root": (
                            str(run_stage.config.case_root)
                            if run_stage.config.case_root
                            else None
                        ),
                        "submit_time": run_stage.state.submit_time,
                        "end_time": run_stage.state.end_time,
                    }
                )
        return pd.DataFrame(
            rows,
            columns=[
                "member",
                "stage",
                "status",
                "job_id",
                "attempts",
                "case_root",
                "submit_time",
                "end_time",
            ],
        )

    def format_status(self, status: pd.DataFrame, show_members: bool = False) -> str:
        """Format status() for printing: counts per stage, then the failed and
        lost members

        Args:
            status (pd.DataFrame): output of status()
            show_members (bool, optional): print every member and stage too.
                Defaults to False

        Returns:
            str: status text
        """
        columns = [
            c for c in _STATUS_ORDER if c != "lost" or "lost" in set(status["status"])
        ]
        counts = (
            status.groupby(["stage", "status"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=self.stage_names, columns=columns, fill_value=0)
            .rename_axis(index=None, columns=None)
        )
        lines = [
            f"Ensemble: {self.config.ensemble_id} ({self.root})",
            f"{len(self.param_files)} members",
            "",
            counts.to_string(),
        ]
        for state in ("failed", "lost"):
            rows = status[status["status"] == state]
            if not rows.empty:
                names = [f"{r.member}:{r.stage}" for r in rows.itertuples()]
                lines += [
                    "",
                    f"{state.capitalize()} ({len(names)}): {_preview(names, 10)}",
                ]
                lines.append(
                    f"  retry with: meristem ensemble retry --root {self.root} --{state}"
                )
        if show_members:
            cols = ["member", "stage", "status", "job_id", "attempts"]
            lines += ["", status[cols].to_string(index=False)]
        return "\n".join(lines)

    def manifest(self) -> pd.DataFrame:
        """One row per member: paramgen key values, parameter file, and for each
        stage its status, job ID, attempts, case root, archive directory and times

        Returns:
            pd.DataFrame: the manifest
        """
        status = self.status()
        wide = pd.DataFrame({"member": list(self.param_files)})
        wide["param_file"] = [str(p) for p in self.param_files.values()]
        for stage in self.stage_names:
            part = status[status["stage"] == stage].drop(columns="stage")
            part = part.set_index("member")
            part.columns = [f"{stage}_{c}" for c in part.columns]
            part[f"{stage}_archive_dir"] = part[f"{stage}_case_root"].map(_archive_dir)
            wide = wide.merge(part, left_on="member", right_index=True, how="left")

        key_file = self.config.paramgen_dir / f"{self.config.paramgen_prefix}_key.csv"
        if key_file.is_file():
            key = pd.read_csv(key_file, dtype=str).rename(
                columns={"ensemble": "member"}
            )
            wide = key.merge(wide, on="member", how="right")
        return wide

    def _select_members(self, members: list[str] | None = None) -> list[str]:
        """Return select members or all members, checking for errors

        Args:
            members (list[str]): input member list

        Returns:
            list[str]: verified member list
        """
        if members is not None:
            self._check_members(members)
            return members
        return list(self.param_files)

    def _check_stages(self, stages: list[str]):
        """Check that input stages are actually in the ensemble

        Args:
            stages (list[str]): input list of stage names

        Raises:
            ValueError: Unknown member stages
        """
        unknown = [s for s in stages if s not in self.stage_names]
        if unknown:
            names = ", ".join(repr(s) for s in unknown)
            raise ValueError(
                f"unknown member stage {names}; member stages are {self.stage_names}."
            )

    def _find_retries(
        self, runs: dict[str, ModelRun], stage: str | None, lost: bool, failed: bool
    ) -> dict[str, tuple[str, bool]]:
        """Pick the stage to retry for each member

        With neither failed nor lost, the named member and/or stage is resubmitted
        as it is (its current stage if only a member is named). Otherwise stages
        are selected by status.

        Args:
            runs (dict[str, ModelRun]): member name: run
            stage (str | None): only this member stage
            lost (bool): select lost stages
            failed (bool): select failed stages

        Raises:
            ValueError: an explicitly named stage is running or done, or a stage
                before it isn't done (lists every member affected)

        Returns:
            dict[str, tuple[str, bool]]: member : (stage name, whether it was lost)
        """
        explicit = not failed and not lost
        to_retry: dict[str, tuple[str, bool]] = {}
        refused: list[str] = []
        blocked: list[str] = []

        for member, run in runs.items():
            if explicit:
                name = _explicit_retry(member, run, stage, refused, blocked)
                if name is not None:
                    to_retry[member] = (name, False)
                continue

            lost_names = set(run.lost_stages()) if lost else set()
            for run_stage in run.stages:
                name = run_stage.config.name
                if stage is not None and name != stage:
                    continue
                if name in lost_names:
                    to_retry[member] = (name, True)
                elif failed and run_stage.state.status == StageStatus.FAILED:
                    to_retry[member] = (name, False)
        problems = []
        if refused:
            problems.append(
                "can only resubmit PENDING or FAILED stages (use --lost for a lost "
                "one):\n" + "\n".join(refused)
            )
        if blocked:
            problems.append(
                "can only resubmit a stage once every stage before it is done "
                "(retry those first):\n" + "\n".join(blocked)
            )
        if problems:
            raise ValueError("\n".join(problems))
        return to_retry

    def prepare(self, member: str, stage: str):
        """Clone and configure one member's case

        Args:
            member (str): member string
            stage (str): stage string name

        Raises:
            ValueError: unknown member or member stage, or base case not built
            FileNotFoundError: missing base case, parameter file, member run,
                or CIME script (xmlquery, create_clone, case.setup)
            PermissionError: a CIME script isn't executable
            FileExistsError: the stage's case_root exists but isn't its complete clone
            RuntimeError: a CIME command (create_clone, case.setup) failed
        """
        self._check_member_stage(member, stage)
        self._check_base_case()
        run = self.member_run(member)
        case_root = self._case_root(member, stage)

        # a stage that's re-run (e.g. retried after a crash) reuses its clone,
        # which already has its PROJECT from create_clone
        if not _is_existing_case(case_root):
            self._clone(case_root, run.project)

        # safe to re-run: CIME skips setup that's already been done
        self._case_setup(case_root)

    def finalize(self, member: str, stage: str):
        """Post-script hook for each member stage which set's the parameter file

        Args:
            member (str): member string
            stage (str): stage string name
        """
        self._check_stage(member, stage)
        case_root = self._case_root(member, stage)
        self._set_paramfile(case_root, self.param_files[member])
        self._check_paramfile(case_root, self.param_files[member])

    def member_run(self, member: str) -> ModelRun:
        """Load one member's ModelRun"""
        return ModelRun.load(self.root / "members" / member)

    def save(self):
        """Atomically write current state to ``<root>/ensemble_state.json``.

        Written via a uniquely named temporary file and a rename
        (write_atomically), so the state file is never left partially written,
        even if the process is killed or another process saves at the same time.
        """
        state = {
            "config": self.config.to_dict(),
            "members": {member: str(path) for member, path in self.param_files.items()},
        }
        write_atomically(self.root / _STATE_FILE, json.dumps(state, indent=2))

    def _find_param_members(self) -> list[str]:
        """Get the list of parameter members that will be used in this ensemble

        Raises:
            FileNotFoundError: Can't find the member list file
            ValueError: member list file is empty
            ValueError: member list file contains duplicates
            ValueError: one or more requested members not in member list file

        Returns:
            list[str]: list of requested members that exist in the member list file
        """

        # read in parameter file list
        list_file = self.config.paramgen_dir / f"{self.config.paramgen_prefix}.txt"
        if not list_file.is_file():
            raise FileNotFoundError(
                f"member list {list_file} not found. "
                f"Has meristem paramgen been run with prefix {self.config.paramgen_prefix!r}?"
            )

        # get all members
        all_members = [
            line.strip()
            for line in list_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not all_members:
            raise ValueError(f"member list {list_file} is empty.")

        # check for duplicates
        duplicates = sorted(m for m, n in Counter(all_members).items() if n > 1)
        if duplicates:
            raise ValueError(f"member list {list_file} has duplicates: {duplicates}")

        # subset to intended members
        if self.config.members is None:
            return all_members

        unknown = sorted(set(self.config.members) - set(all_members))
        if unknown:
            raise ValueError(f"requested members not in {list_file}: {unknown}")
        wanted = set(self.config.members)
        return [m for m in all_members if m in wanted]

    def _find_param_files(self) -> dict[str, Path]:
        """Find parameter files associated with the member list

        Raises:
            FileNotFoundError: can't find the key file
            ValueError: key file has no ensemble column
            FileNotFoundError: can't find a parameter file or member not on key_file

        Returns:
            dict[str, Path]: dictionary with member name and path to parameter file
        """

        selected = self._find_param_members()

        # read in key file
        key_file = self.config.paramgen_dir / f"{self.config.paramgen_prefix}_key.csv"
        if not key_file.is_file():
            raise FileNotFoundError(
                f"key file {key_file} not found. "
                f"Has meristem paramgen been run with prefix {self.config.paramgen_prefix!r}?"
            )
        # read as str so all-digit member names (e.g. "0001") aren't parsed as ints
        key = pd.read_csv(key_file, dtype=str)
        if "ensemble" not in key.columns:
            raise ValueError(
                f"key file {key_file} has no 'ensemble' column "
                f"(columns: {list(key.columns)})."
            )
        key_members = set(key["ensemble"])

        # check to make sure parameter files exist, with an extension the
        # paramfile_key can read
        param_key = self.config.paramfile_key
        extensions = _PARAM_EXTENSIONS[param_key]
        param_files: dict[str, Path] = {}
        problems: list[str] = []
        for member in selected:
            if member not in key_members:
                problems.append(f"  {member}: {member} not in {key_file}")
                continue
            # check every extension, so a stray file of another format is
            # ambiguous even when the key can't read it
            found = [
                self.config.paramgen_dir / f"{member}{ext}"
                for ext in _ALL_PARAM_EXTENSIONS
                if (self.config.paramgen_dir / f"{member}{ext}").is_file()
            ]
            if len(found) > 1:
                problems.append(f"  {member}: both .nc and .json exist")
            elif not found:
                problems.append(
                    f"  {member}: no "
                    + " or ".join(f"{member}{ext}" for ext in extensions)
                )
            elif found[0].suffix not in extensions:
                problems.append(
                    f"  {member}: {found[0].name} can't be used for {param_key}, "
                    f"which needs {' or '.join(extensions)}"
                )
            else:
                param_files[member] = found[0]

        if problems:
            raise FileNotFoundError(
                f"parameter files missing or ambiguous in {self.config.paramgen_dir}:\n"
                + "\n".join(problems)
            )
        return param_files

    def _check_base_case(self):
        """Check to make sure the supplied base case exists and can be cloned

        Raises:
            FileNotFoundError: Can't find the base case directory
            FileNotFoundError: Can't find the xmlquery script inside the base case dir
            PermissionError: xmlquery isn't executable
            RuntimeError: xmlquery exited with an error
            ValueError: BUILD_COMPLETE didn't work or returned FALSE
        """
        if not self.base_case.is_dir():
            raise FileNotFoundError(f"cannot find base case {self.base_case}")

        _check_case_script("xmlquery", self.base_case)

        result = run_command(
            ["./xmlquery", "BUILD_COMPLETE", "--value"], self.base_case
        )

        if result.upper() != "TRUE":
            raise ValueError(
                f"base case {self.base_case} is not built (BUILD_COMPLETE="
                f"{result!r}); run ./case.build first."
            )

    def _check_param_files(self, members: list[str] | None = None):
        """Check that the stored parameter files still exist

        Args:
            members (list[str] | None, optional): members to check, or None for all.
                Defaults to None.

        Raises:
            ValueError: if any of members isn't in this ensemble
            FileNotFoundError: if any of the checked members' parameter files is missing
        """
        if members is not None:
            self._check_members(members)

        to_check = (
            self.param_files
            if members is None
            else {m: self.param_files[m] for m in members}
        )
        missing = [f"  {m}: {p}" for m, p in to_check.items() if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "parameter file(s) no longer exist:\n" + "\n".join(missing)
            )

    def _check_case_roots_free(self):
        """Check no member stage's case directory exists yet

        prepare reuses an existing case with the expected name as the member's
        own clone (so a retried stage keeps its clone), so a leftover case from
        another ensemble with the same name would be reused silently. Checked at
        create only: later, an existing case is this ensemble's own.

        Raises:
            FileExistsError: if any of them exists
        """
        existing = [
            str(case_root)
            for member in self.param_files
            for stage in self.config.member_stages
            if (case_root := self._case_root(member, stage.name)).exists()
        ]
        if existing:
            raise FileExistsError(
                f"{len(existing)} member case directories already exist, e.g. "
                f"{_preview(existing)}. They may be left from another ensemble with "
                f"ensemble_id {self.config.ensemble_id!r}; remove them, or use a "
                "different case_dir or ensemble_id."
            )

    def _check_stage_scripts(self):
        """Check that every member stage's script exists

        Raises:
            FileNotFoundError: if any stage's script is missing
        """
        missing = [
            f"  {stage.name}: {stage.script}"
            for stage in self.config.member_stages
            if stage.script is not None and not stage.script.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "member stage script(s) not found:\n" + "\n".join(missing)
            )

    def _check_members(self, members: list[str]):
        """Check that member keys actually point to real members

        Args:
            members (list[str]): Input member key list

        Raises:
            ValueError: Unknown members
        """
        unknown = sorted(set(members) - set(self.param_files))
        if unknown:
            raise ValueError(
                f"unknown member(s) {_preview(unknown)}; the ensemble has "
                f"{len(self.param_files)} members, e.g. "
                f"{_preview(list(self.param_files))}."
            )

    def _member_stages(self, member: str) -> list[Stage]:
        """Build a member's stages: each member stage, in its own case under
        case_dir, running ``meristem ensemble prepare`` then its script (if any),
        then ``meristem ensemble finalize``

        Args:
            member (str): member name

        Returns:
            list[Stage]: the member's stages, in member_stages order
        """
        stages = []
        for stage_config in self.config.member_stages:
            command = self._member_command("prepare", member, stage_config.name)
            post_command = self._member_command("finalize", member, stage_config.name)

            case_root = self._case_root(member, stage_config.name)
            # the job runs prepare (clone the base case, set the parameter
            # file), then the stage's own script, if any, on the clone, then finalize
            stages.append(
                Stage(
                    config=replace(
                        stage_config,
                        command=command,
                        post_command=post_command,
                        case_root=case_root,
                    )
                )
            )
        return stages

    def _member_command(self, action: str, member: str, stage: str) -> str:
        """Creates an ensemble member command

        Args:
            action (str): action (meristem command)
            member (str): member string name
            stage (str): member stage's string name

        Returns:
            str: command
        """
        command = shlex.join(
            [
                *shlex.split(meristem_cmd()),
                "ensemble",
                action,
                "--root",
                str(self.root),
                "--member",
                member,
                "--stage",
                stage,
            ]
        )
        return command

    def _set_paramfile(self, case_root: Path, param_file: Path):
        """Point a case's user_nl_clm at a member's parameter file

        Removes any existing assignment of ``config.paramfile_key`` and appends a new one.
        Written via temp-file-then-rename

        Args:
            case_root (Path): case directory
            param_file (Path): parameter file to use
        Raises:
            FileNotFoundError: user_nl_clm doesn't exist (broken case?)
        """
        user_nl_clm = case_root / "user_nl_clm"
        if not user_nl_clm.is_file():
            raise FileNotFoundError(
                f"can't find {user_nl_clm}; case {case_root} looks broken"
            )

        key = self.config.paramfile_key

        # namelists are case-insensitive; the re anchor keeps "paramfile" from
        # matching "fates_paramfile"
        pattern = re.compile(rf"^\s*{re.escape(key)}\s*=", re.IGNORECASE)

        text = user_nl_clm.read_text(encoding="utf-8")
        lines = [
            line for line in text.splitlines(keepends=True) if not pattern.match(line)
        ]
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(f"{key} = '{param_file}'\n")

        tmp = case_root / "user_nl_clm.tmp"
        tmp.write_text("".join(lines), encoding="utf-8")
        tmp.rename(user_nl_clm)

    def _check_paramfile(self, case_root: Path, param_file: Path):
        """Check that a case's generated lnd_in sets ``config.paramfile_key`` to
        a member's parameter file

        Runs preview_namelists, then checks the key's value in CaseDocs/lnd_in
        (not just that the path appears somewhere, so a comment or a different
        key can't pass)

        Args:
            case_root (Path): case directory
            param_file (Path): parameter file the case should use

        Raises:
            RuntimeError: preview_namelists failed, or the key is missing or
                set to something else
            FileNotFoundError: CaseDocs/lnd_in doesn't exist (broken case?)
        """
        _check_case_script("preview_namelists", case_root)
        try:
            run_command(["./preview_namelists"], case_root)
        except RuntimeError as exc:
            raise RuntimeError(f"preview_namelists failed: {exc}") from exc

        lnd_in = case_root / "CaseDocs" / "lnd_in"
        if not lnd_in.is_file():
            raise FileNotFoundError(
                f"can't find {lnd_in}; case {case_root} looks broken"
            )

        key = self.config.paramfile_key
        # same anchor as _set_paramfile, capturing the quoted value
        pattern = re.compile(
            rf"^\s*{re.escape(key)}\s*=\s*(['\"])(.*?)\1", re.IGNORECASE | re.MULTILINE
        )
        text = lnd_in.read_text(encoding="utf-8")
        values = [match.group(2) for match in pattern.finditer(text)]
        if not values:
            raise RuntimeError(f"{key} is not set in {lnd_in}")
        # compared resolved, so build-namelist writing the same file another
        # way (through a symlink, or with e.g. // or ./) still passes
        if len(values) != 1 or Path(values[0]).resolve() != param_file.resolve():
            raise RuntimeError(
                f"{key} in {lnd_in} is {values}, expected {str(param_file)!r}"
            )

    def _check_member_stage(self, member: str, stage: str):
        """Check that member and stage are in this ensemble, and the member's
        parameter file still exists

        Args:
            member (str): member name
            stage (str): member stage name

        Raises:
            ValueError: unknown member stage or member
            FileNotFoundError: the member's parameter file is missing
        """
        self._check_stages([stage])
        self._check_param_files([member])

    def _check_stage(self, member: str, stage: str):
        """Checks to see if stage is ready for submission

        Args:
            member (str): member string
            stage (str): stage name string

        Raises:
            ValueError: unknown member or member stage
            FileNotFoundError: missing parameter file, or can't find stage's
                case or it isn't complete
        """
        self._check_member_stage(member, stage)

        case_root = self._case_root(member, stage)
        if not _is_existing_case(case_root):
            raise FileNotFoundError(
                f"can't find case {case_root} or isn't a complete case. "
                "run prepare first"
            )

    def _case_root(self, member: str, stage: str) -> Path:
        """Case directory for one member stage's clone

        The name includes ensemble_id, so ensembles that reuse the same base case
        and parameter members don't share cases or CIME output directories.

        Args:
            member (str): member name string
            stage (str): stage string

        Returns:
            Path: path to case directory for member stage's clone
        """
        case = (
            f"{self.config.base_case.name}_{self.config.ensemble_id}_{member}_{stage}"
        )
        return self.config.case_dir / case

    def _case_setup(self, case_root: Path):
        """Run case.setup in a member case

        Args:
            case_root (Path): path to member's case root

        Raises:
            FileNotFoundError: case_root has no case.setup
            PermissionError: case.setup isn't executable
            RuntimeError: case.setup failed
        """
        _check_case_script("case.setup", case_root)
        try:
            run_command(["./case.setup"], case_root)
        except RuntimeError as exc:
            raise RuntimeError(f"case.setup failed: {exc}") from exc

    def _clone(self, case_root: Path, project: str):
        """Clone a single case at case_root from base_case

        Args:
            case_root (Path): path to the member's case_root
            project (str): PROJECT to set on the clone (create_clone --project)

        Raises:
            FileNotFoundError: no executable create_clone found
            RuntimeError: case_root failed
            RuntimeError: case_root passed but something else went wrong and the case
            was not successfully created
        """

        cimeroot = Path(
            run_command(["./xmlquery", "CIMEROOT", "--value"], self.base_case)
        )
        create_clone = cimeroot / "scripts" / "create_clone"
        if not create_clone.is_file() or not os.access(create_clone, os.X_OK):
            raise FileNotFoundError(
                f"no executable create_clone at {create_clone} (CIMEROOT of base "
                f"case {self.base_case}); has CIME moved since the base case was built?"
            )

        cmd = [
            str(create_clone),
            "--case",
            str(case_root),
            "--clone",
            str(self.base_case),
            "--keepexe",
            "--project",
            project,
        ]
        if self.config.output_root:
            cmd.extend(["--cime-output-root", str(self.config.output_root)])

        self.case_dir.mkdir(parents=True, exist_ok=True)
        try:
            run_command(cmd, self.base_case)
        except RuntimeError as exc:
            leftover = (
                f" It left a partial {case_root}; remove it before retrying."
                if case_root.exists()
                else ""
            )
            raise RuntimeError(
                f"cloning {self.base_case} failed: {exc}.{leftover}"
            ) from exc

        if not _is_existing_case(case_root):
            raise RuntimeError(
                f"create_clone exited successfully but {case_root} was not created."
            )

    def _member_runs(self) -> list[ModelRun]:
        """Build and validate (without writing) a ModelRun for every member

        Raises:
            ValueError: if a member's run is invalid (e.g. its run_id has
            characters ModelRun doesn't allow, or case_dir is relative)

        Returns:
            list[ModelRun]: one run per member, in param_files order
        """
        runs = []
        for member in self.param_files:
            run = ModelRun(
                root=self.root / "members" / member,
                run_id=f"{self.config.ensemble_id}_{member}",
                stages=self._member_stages(member),
                email=self.config.email or None,
                project=self.config.project,
                helper_queue=self.config.helper_queue,
            )
            run._validate_stages()  # pylint: disable=protected-access
            runs.append(run)
        return runs


def _explicit_retry(
    member: str,
    run: ModelRun,
    stage: str | None,
    refused: list[str],
    blocked: list[str],
) -> str | None:
    """The stage to resubmit for a member when what to retry is named rather
    than selected by status: the named stage, or the member's current stage

    Args:
        member (str): member name
        run (ModelRun): the member's run
        stage (str | None): the named stage, or None for the current stage
        refused (list[str]): appended to if the stage is running or done
        blocked (list[str]): appended to if a stage before it isn't done

    Returns:
        str | None: the stage name, or None if there's nothing to retry or it
        can't be (then refused or blocked says why)
    """
    target = (
        next((s for s in run.stages if s.config.name == stage), None)
        if stage
        else run.current_stage
    )
    if target is None:
        return None
    status = target.state.status
    if status not in (StageStatus.PENDING, StageStatus.FAILED):
        refused.append(f" {member}: {target.config.name} is {status.value}")
        return None
    # a named stage can be a later one: it mustn't start on unfinished input
    blocking = run.unfinished_before(target.config.name)
    if blocking is not None:
        blocked.append(
            f" {member}: {target.config.name} (before it, "
            f"{blocking.config.name} is {blocking.state.status.value})"
        )
        return None
    return target.config.name


# what one member's submit, retry or cancel can raise (qsub/qstat/qdel failures,
# a refused retry, a missing file) without the other members being affected
_MEMBER_ERRORS = (RuntimeError, ValueError, OSError)


def _for_each_member(
    members: list[str], action: Callable[[str], T]
) -> tuple[dict[str, T], dict[str, Exception]]:
    """Run action on each member, carrying on past a member that fails

    Args:
        members (list[str]): members, in order
        action (Callable[[str], T]): what to do for one member

    Returns:
        tuple[dict[str, T], dict[str, Exception]]: each successful member's
        result, and each failed member's error
    """
    results: dict[str, T] = {}
    errors: dict[str, Exception] = {}
    for member in members:
        try:
            results[member] = action(member)
        except _MEMBER_ERRORS as exc:
            errors[member] = exc
    return results, errors


def _raise_member_errors(
    ensemble_id: str, action: str, errors: dict[str, Exception], n_members: int
) -> None:
    """Raise one error naming every member action failed for, if any

    Args:
        ensemble_id (str): ensemble ID, for the message
        action (str): what was being done, e.g. "submit"
        errors (dict[str, Exception]): failed member -> its error
        n_members (int): how many members action was run on

    Raises:
        RuntimeError: if errors isn't empty, chained from the first error
    """
    if not errors:
        return
    n_shown = 10
    lines = [
        f"  {member}: " + str(exc).replace("\n", "\n    ")
        for member, exc in list(errors.items())[:n_shown]
    ]
    if len(errors) > n_shown:
        lines.append(f"  ...and {len(errors) - n_shown} more")
    raise RuntimeError(
        f"{ensemble_id}: {action} failed for {len(errors)} of {n_members} members "
        f"(the other {n_members - len(errors)} went through):\n" + "\n".join(lines)
    ) from next(iter(errors.values()))


def _preview(names: list[str], n: int = 3) -> str:
    """Show at most n names, saying how many more there are

    Args:
        names (list[str]): names to show
        n (int, optional): most names to show. Defaults to 3.

    Returns:
        str: e.g. "['a', 'b', 'c'] and 2 more"
    """
    more = f" and {len(names) - n} more" if len(names) > n else ""
    return f"{names[:n]}{more}"


def _archive_dir(case_root: str | None) -> str | None:
    """DOUT_S_ROOT of a case, or None if the case doesn't exist yet"""
    if (
        pd.isna(case_root)
        or not case_root
        or not (Path(case_root) / "xmlquery").is_file()
    ):
        return None
    try:
        return run_command(["./xmlquery", "DOUT_S_ROOT", "--value"], Path(case_root))
    except (RuntimeError, OSError):
        return None


def _to_config(config: EnsembleConfig | dict | Path | str) -> EnsembleConfig:
    """Normalise *config* to an EnsembleConfig.

    Args:
        config: an EnsembleConfig, a dict, or a path to a YAML file.

    Returns:
        EnsembleConfig: the validated config

    Raises:
        TypeError: if *config* is not an EnsembleConfig, dict, str, or Path.
    """
    if isinstance(config, EnsembleConfig):
        return config
    if isinstance(config, dict):
        return EnsembleConfig.from_dict(config)
    if isinstance(config, (str, Path)):
        return EnsembleConfig.from_yaml(config)
    raise TypeError(
        "config must be an EnsembleConfig, dict, str, or Path, "
        f"got {type(config).__name__!r}."
    )


def _check_case_script(script: str, case_dir: Path) -> Path:
    """Check that a CIME case script (e.g. xmlquery, case.setup) exists in a
    case directory and is executable

    Args:
        script (str): script name
        case_dir (Path): path to case dir

    Raises:
        FileNotFoundError: can't find script
        PermissionError: isn't executable

    Returns:
        Path: path to the script
    """
    path = case_dir / script
    if not path.is_file():
        raise FileNotFoundError(
            f"{case_dir} has no {script}; is it a CIME case directory?"
        )
    if not os.access(path, os.X_OK):
        raise PermissionError(f"{path} is not executable.")

    return path


def _is_existing_case(case_root: Path) -> bool:
    """Check whether a member stage's case directory already holds its clone

    Args:
        case_root (Path): member stage case directory

    Raises:
        FileExistsError: if case_root exists but isn't a complete CIME case
        (e.g. a clone that died partway); it's never removed automatically

    Returns:
        bool: True if case_root is a complete CIME case, False if it doesn't exist
    """
    if not case_root.exists():
        return False

    hint = "remove it and retry the stage to re-clone it."
    xmlquery = case_root / "xmlquery"
    if not (case_root / "env_case.xml").is_file() or not os.access(xmlquery, os.X_OK):
        raise FileExistsError(
            f"{case_root} exists but is not a complete CIME case; {hint}"
        )
    try:
        case = run_command(["./xmlquery", "CASE", "--value"], case_root)
    except RuntimeError as exc:
        raise FileExistsError(
            f"{case_root} exists but xmlquery CASE failed ({exc}); {hint}"
        ) from exc
    if case != case_root.name:
        raise FileExistsError(
            f"{case_root} exists but holds case {case!r}, not {case_root.name!r}; "
            f"{hint}"
        )
    return True
