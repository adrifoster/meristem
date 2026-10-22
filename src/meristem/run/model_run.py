"""ModelRun class - manages full lifecycle of a multi-stage run"""

from __future__ import annotations

import os
import time
import json
from pathlib import Path
import re
import shlex
from collections.abc import Collection

from jinja2 import Environment, FileSystemLoader
import yaml

from meristem.commands import meristem_cmd
from .scheduler import DEFAULT_SCHEDULER, JobState, Scheduler, get_scheduler
from .stages import Stage, StageStatus

# templates dir
_TEMPLATES = Path(__file__).parent / "templates"

_ENV = Environment(
    loader=FileSystemLoader(_TEMPLATES),
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
)


def _shquote(value) -> str:
    """Quote a value for safe use as a single shell word in job scripts

    Raises:
        ValueError: if value is None, rather than silently rendering 'None'
    """
    if value is None:
        raise ValueError("cannot shell-quote None in a job script template")
    return shlex.quote(str(value))


_ENV.filters["shquote"] = _shquote

# state file name
_STATE_FILE = "run_state.json"

# characters that are safe in a run_id (used as a directory component).
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

# keys that must be present in a config dict.
_REQUIRED_CONFIG_KEYS = frozenset({"root", "stages"})

# seconds after submission that an UNKNOWN setup job still counts as active.
# NCAR's qstat wrapper (qstat-cache) serves a snapshot that lags by minutes, so
# a just-submitted job isn't in it yet
_UNKNOWN_GRACE_SECONDS = 600


class ModelRun:
    """
    Manages the full lifecycle of a multi-stage CESM run.

    Use ModelRun.create() to initialise a new run on disk, or
    ModelRun.load() to resume an existing one.

    Attributes
    ===========
    root: working directory; all CIME cases live under here
    run_id: short identifier, e.g. "member_0042" or "my_spinup"
    stages: ordered tuple of Stage instances
    email: email for the machine - can be None
    project: project code to use for job submission. can use os.environ variable
    scheduler: job scheduler used to submit job scripts. defaults to PBS
    """

    def __init__(
        self,
        root: Path | str,
        run_id: str,
        stages: tuple[Stage, ...] | list[Stage],
        project: str,
        email: str | None = None,
        scheduler: Scheduler | None = None,
    ):
        self.root = Path(root)
        self.run_id = run_id
        self.stages: tuple[Stage, ...] = tuple(stages)
        self.email = email
        self.project = project if project else os.getenv("PROJECT", "")
        self.scheduler = scheduler or get_scheduler(DEFAULT_SCHEDULER)
        self._validate()

    def _validate(self):
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        if not _RUN_ID_RE.match(self.run_id):
            raise ValueError(
                f"run_id {self.run_id!r} contains invalid characters. "
                "Only letters, digits, hyphens, and underscores are allowed."
            )
        if not self.project:
            raise ValueError(
                "project must not be empty and $PROJECT environment variable not set."
            )
        if not self.stages:
            raise ValueError("stages must not be empty")
        # the PBS log file (#PBS -o) lives under root, and PBS directives can't
        # reliably hold paths with whitespace
        if any(char.isspace() for char in str(self.root)):
            raise ValueError(
                f"root {str(self.root)!r} must not contain whitespace "
                "(PBS can't write job logs to such paths)."
            )

    def _validate_stages(self, start: str | None = None):
        """Validate stage configs (script exists, case_root absolute)

        Not part of _validate so that load still works (e.g. for status) if a
        script has since moved; called at create, submit, and retry instead, so
        a bad later stage fails up front rather than inside an advance job.

        Args:
            start (str | None, optional): validate only this stage and those
            after it (the ones still to run). Defaults to None (all stages).
        """
        idx = self.stage_names.index(start) if start else 0
        for stage in self.stages[idx:]:
            stage.config.validate()

    @classmethod
    def create(
        cls,
        config: Path | str | dict,
        scheduler: Scheduler | None = None,
    ) -> ModelRun:
        """Initialise a brand-new run from a YAML file or config dict.

        Creates root on disk and writes the initial state file.
        Call this once; use load for all subsequent access.

        Args:
            config: path to a YAML file, or a dict with the same schema
            scheduler: scheduler to use. If None, built from the config's
                "scheduler" key (default "pbs")

        Returns:
            ModelRun: ModelRun instance
        Raises:
            ValueError: if required config keys are missing or invalid
            FileExistsError: if a state file already exists at the target root,
            indicating that the run has already been initialized
        """
        cfg = _load_config(config)
        _validate_config(cfg)

        root = Path(cfg["root"])
        state_path = root / _STATE_FILE
        if state_path.exists():
            raise FileExistsError(
                f"A state file already exists at {state_path}. "
                "Use ModelRun.load() to resume an existing run."
            )

        stages = [Stage.from_dict(stage) for stage in cfg["stages"]]
        run = cls(
            root=root,
            run_id=cfg.get("run_id", "run"),
            stages=stages,
            email=cfg.get("email"),
            project=cfg.get("project", ""),
            scheduler=scheduler
            or get_scheduler(cfg.get("scheduler", DEFAULT_SCHEDULER)),
        )
        run._validate_stages()
        run.root.mkdir(parents=True, exist_ok=True)
        run.save()
        return run

    @classmethod
    def load(cls, root: Path | str, scheduler: Scheduler | None = None) -> ModelRun:
        """Load an existing run from its state file.

        Args:
            root (Path | str): Path to root
            scheduler (Scheduler | None, optional): scheduler to use. If None, built
                from the state file's "scheduler" key (default "pbs")

        Returns:
            ModelRun: ModelRun instance
        Raises:
            FileNotFoundError: if no state file exists at root
        """
        root = Path(root)
        state_path = root / _STATE_FILE
        if not state_path.exists():
            raise FileNotFoundError(
                f"No state file found at {state_path}. "
                "Has this run been initialised with ModelRun.create()?"
            )

        state = json.loads(state_path.read_text(encoding="utf-8"))

        return cls(
            root=root,
            run_id=state["run_id"],
            email=state.get("email"),
            stages=[Stage.from_dict(stage) for stage in state["stages"]],
            project=state.get("project", ""),
            # state files written before the scheduler key existed are PBS runs
            scheduler=scheduler
            or get_scheduler(state.get("scheduler", DEFAULT_SCHEDULER)),
        )

    def save(self):
        """Atomically write current state to ``<root>/run_state.json``.

        Uses a write-to-temp-then-rename pattern so the state file is
        never left in a partially written state if the process is killed.
        """
        state = {
            "run_id": self.run_id,
            "email": self.email,
            "project": self.project,
            "scheduler": self.scheduler.name,
            "stages": [stage.to_dict() for stage in self.stages],
        }
        tmp = self.root / f"{_STATE_FILE}.tmp"
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.rename(self.root / _STATE_FILE)

    @property
    def stage_names(self) -> list[str]:
        """Get list of stage names"""
        return [stage.config.name for stage in self.stages]

    @property
    def current_stage(self) -> Stage | None:
        """Get the first non-DONE stage

        Returns:
            Stage | None: Stage, or None if all complete
        """
        for stage in self.stages:
            if stage.state.status != StageStatus.DONE:
                return stage
        return None

    def is_lost(self, stage_name: str) -> bool:
        """Check whether a submitted stage has no jobs left that will finish it.

        A stage is lost when it is SUBMITTED but none of its recorded jobs
        (setup, and advance and fail once the setup job has submitted them) are
        still queued or running, so nothing in PBS will ever mark it DONE or
        FAILED. This covers a setup job killed before it submitted the advance
        and fail jobs (including every no_cime stage), and advance/fail jobs
        that ended or were deleted without updating the stage.

        A setup job PBS reports as UNKNOWN within _UNKNOWN_GRACE_SECONDS of
        submission counts as active, since qstat's cache may not have it yet.

        PBS is queried before the state file is re-read: a job saves the state
        before it exits, so any job PBS reports as finished has already written
        its result. The stage is only lost if it is still SUBMITTED on disk with
        the same job IDs; this run's stages are updated to match the disk.

        This is also how a crash in an earlier segment of a resubmitting CIME run
        (RESUBMIT > 0) shows up: CIME's JOB_IDS only holds the last segment's
        jobs, so the fail jobs can't watch earlier segments, and PBS deletes
        every later job (including the advance and fail jobs) along with them.

        Known gaps (the stage is reported lost while its CIME run is going):
        a setup job killed between case.submit and --submit-advance, and stages
        submitted before advance/fail job IDs were recorded (old state files).

        Args:
            stage_name (str): name of the stage to check

        Raises:
            RuntimeError: if the scheduler query fails, so an outage is never
                mistaken for every job being gone

        Returns:
            bool: True if nothing in PBS will finish the stage
        """
        stage = self._stage(stage_name)
        if stage.state.status != StageStatus.SUBMITTED:
            return False

        recorded = (
            stage.state.job_id,
            stage.state.advance_job_id,
            *stage.state.fail_job_ids,
        )
        job_ids = [
            job_id for job_id in recorded if job_id and not job_id.startswith("DRY_")
        ]
        if not job_ids:
            # dry run: nothing was really submitted
            return False

        states = self.scheduler.query(job_ids)
        active = {JobState.QUEUED, JobState.RUNNING}
        if any(states.get(job_id, JobState.UNKNOWN) in active for job_id in job_ids):
            return False
        # a lagging qstat cache may not have the setup job yet. Only the setup
        # job needs this: it submits the advance and fail jobs before exiting, so
        # a snapshot that shows it finished also has them
        setup_job_id, submit_time = stage.state.job_id, stage.state.submit_time
        if (
            setup_job_id is not None
            and states.get(setup_job_id, JobState.UNKNOWN) is JobState.UNKNOWN
            and submit_time is not None
            and time.time() - submit_time < _UNKNOWN_GRACE_SECONDS
        ):
            return False

        # re-read the state now that PBS has been queried (see docstring)
        self.stages = ModelRun.load(self.root, scheduler=self.scheduler).stages
        fresh = self._stage(stage_name).state
        return fresh.status == StageStatus.SUBMITTED and recorded == (
            fresh.job_id,
            fresh.advance_job_id,
            *fresh.fail_job_ids,
        )

    def _stage(self, name: str) -> Stage:
        """Get the Stage given an input name

        Args:
            name (str): stage.config.name

        Returns:
            Stage: stage
        """
        try:
            idx = self.stage_names.index(name)
        except ValueError as exc:
            raise ValueError(
                f"No stage named {name!r}. "
                f"Known stages: {[stage.config.name for stage in self.stages]}"
            ) from exc
        return self.stages[idx]

    def _next_stage(self, after: str) -> Stage | None:
        """Get the next stage given an input name

        Args:
            after (str): stage before the stage we want

        Raises:
            ValueError: Can't find supplied stage

        Returns:
            Stage | None: next Stage, or None if no more stages
        """
        try:
            idx = self.stage_names.index(after)
        except ValueError as exc:
            # this branch is theoretically unreachable via the public API since callers
            # first go through _stage() which raises the same error so this is defensive
            # in case we ever call this without _stage first
            raise ValueError(
                f"No stage named {after!r}. " f"Known stages: {self.stage_names}"
            ) from exc
        return self.stages[idx + 1] if idx + 1 < len(self.stages) else None

    def _previous_stage(self, before: str) -> Stage | None:
        """Get the previous stage given an input name

        Args:
            before (str): stage before the stage we want

        Raises:
            ValueError: Can't find supplied stage

        Returns:
            Stage | None: previous Stage, or None if first stage
        """
        try:
            idx = self.stage_names.index(before)
        except ValueError as exc:
            raise ValueError(
                f"No stage named {before!r}. " f"Known stages: {self.stage_names}"
            ) from exc
        return self.stages[idx - 1] if idx > 0 else None

    def _case_root_for(self, stage: Stage) -> Path:
        """Generate the case root name for this stage

        Args:
            stage (Stage): input Stage

        Returns:
            Path: path to stage's case root
        """
        if stage.config.case_root is not None:
            return stage.config.case_root
        return self.root / stage.config.name

    def lost_stages(self) -> list[str]:
        """Get the names of all lost stages (see is_lost). Queries the scheduler
        but doesn't change the state file

        Returns:
            list[str]: names of lost stages, in stage order
        """
        return [name for name in self.stage_names if self.is_lost(name)]

    def format_status(self, lost: Collection[str] = ()) -> str:
        """Format the status of the run that can be printed to the screen

        Args:
            lost (Collection[str], optional): names of stages to show as lost
                instead of their recorded status, e.g. from lost_stages().
                Defaults to none

        Returns:
            str: status text
        """
        icons = {
            StageStatus.PENDING: "·",
            StageStatus.SUBMITTED: "...",
            StageStatus.DONE: "✓",
            StageStatus.FAILED: "✗",
        }
        lines = [f"Run: {self.run_id} ({self.root})"]
        for stage in self.stages:
            is_lost = stage.config.name in lost
            icon = "?" if is_lost else icons[stage.state.status]
            status = "lost" if is_lost else stage.state.status.value
            job = f" [{stage.state.job_id}]" if stage.state.job_id else ""
            # only show attempts if the stage has been retried at least once
            attempts = (
                f" attempts={stage.state.attempts}" if stage.state.attempts > 1 else ""
            )
            lines.append(
                f" {icon} {stage.config.name:<20}  {status:<10}{job}{attempts}"
            )
        return "\n".join(lines)

    def submit(self, dry_run: bool = False) -> str | None:
        """Submit the first pending stage.

        Args:
            dry_run (bool, optional): If True, only print text to screen and don't
            submit anything. Defaults to False.

        Returns:
            str: PBS job ID of the submit script
        """
        first = self.stages[0]
        if first.state.status != StageStatus.PENDING:
            print(
                f"  {self.run_id}: skipping submit (first stage is {first.state.status.value})"
            )
            return first.state.job_id or None
        self._validate_stages()
        return self._submit_stage(first, dry_run=dry_run)

    def fail(self, stage_name: str):
        """Mark a stage as failed (called from the ERR trap).

        Args:
            stage_name (str): stage name
        """
        stage = self._stage(stage_name)
        if stage.state.status not in (StageStatus.SUBMITTED, StageStatus.DONE):
            raise ValueError(
                f"Stage {stage.config.name!r} is {stage.state.status.value} "
                "- can only fail SUBMITTED or DONE stages."
            )
        stage.state.status = StageStatus.FAILED
        stage.state.end_time = time.time()
        self.save()
        print(f"  {self.run_id} / {stage_name}: FAILED")

    def submit_advance(
        self, stage_name: str, cime_job_id: str, dry_run: bool = False
    ) -> str:
        """Write the advance script with the CIME job ID and submit it
        Called from the setup job script after case.submit completes

        Args:
            stage_name (str): name of stage
            cime_job_id (str): CIME job ID
            dry_run (bool, optional): If True, only print text to screen and don't
            submit anything. Defaults to False.

        Returns:
            str: PBS job ID of the advance script
        """
        stage = self._stage(stage_name)
        job_file = self._write_advance_script(stage, cime_job_id)
        job_id = (
            f"DRY_advance_{self.run_id}_{stage_name}"
            if dry_run
            else self.scheduler.submit(job_file)
        )
        stage.state.advance_job_id = job_id
        self.save()
        print(
            f"  {self.run_id} / {stage_name}: advance job submitted {job_id} "
            f"(afterok:{cime_job_id})"
        )
        return job_id

    def submit_fail(
        self, stage_name: str, cime_job_ids: str | list[str], dry_run: bool = False
    ) -> list[str]:
        """Write and submit one fail script per CIME job ID
        This is so that if our CIME job crashes we can fail the stage
        Called from the setup job script after case.submit completes

        One fail job per CIME job is needed because PBS deletes (rather than
        runs) an afternotok job whose dependency was itself deleted: if
        case.run crashes, st_archive is deleted along with anything waiting on
        it. At most one of these fail jobs ever runs, since a failed job takes
        every job after it in the chain down with it.

        Args:
            stage_name (str): name of stage
            cime_job_ids (str | list[str]): CIME job ID(s), e.g. case.run and
                case.st_archive. Duplicates are submitted once
            dry_run (bool, optional): If True, only print text to screen and don't
            submit anything. Defaults to False.

        Returns:
            list[str]: PBS job IDs of the fail scripts
        """
        stage = self._stage(stage_name)
        if isinstance(cime_job_ids, str):
            cime_job_ids = [cime_job_ids]
        cime_job_ids = list(dict.fromkeys(cime_job_ids))
        if not cime_job_ids:
            raise ValueError("submit_fail needs at least one CIME job ID")

        stage.state.fail_job_ids = []
        for idx, cime_job_id in enumerate(cime_job_ids, start=1):
            # a single fail job keeps the plain _fail name
            suffix = f"_fail_{idx}" if len(cime_job_ids) > 1 else "_fail"
            job_file = self._write_fail_script(stage, cime_job_id, suffix)
            job_id = (
                f"DRY{suffix}_{self.run_id}_{stage_name}"
                if dry_run
                else self.scheduler.submit(job_file)
            )
            # save after each one so a later qsub failure keeps what was submitted
            stage.state.fail_job_ids.append(job_id)
            self.save()
            print(
                f"  {self.run_id} / {stage_name}: fail job submitted {job_id} "
                f"(afternotok:{cime_job_id})"
            )
        return list(stage.state.fail_job_ids)

    def advance(self, completed_stage_name: str, dry_run: bool = False) -> str | None:
        """Advance the pipeline and submit the next stage

        Args:
            completed_stage_name (str): completed stage name
            dry_run (bool, optional): If True, only print text to screen and don't
            submit anything. Defaults to False.

        Returns:
            str | None: PBS job ID of the advance script, or None if all stages complete
        """
        stage = self._stage(completed_stage_name)
        if stage.state.status != StageStatus.SUBMITTED:
            raise ValueError(
                f"Stage {stage.config.name!r} is {stage.state.status.value} "
                "— can only advance SUBMITTED stages."
            )
        stage.state.status = StageStatus.DONE
        stage.state.end_time = time.time()
        self.save()
        print(f"  {self.run_id} / {completed_stage_name}: DONE")

        next_stage = self._next_stage(completed_stage_name)
        if next_stage is None:
            print(f" {self.run_id}: all stages complete!")
            return None
        return self._submit_stage(next_stage, dry_run=dry_run)

    def retry(
        self,
        stage_name: str | None,
        dry_run: bool = False,
        skip_script: bool = False,
        lost: bool = False,
    ) -> str | None:
        """Reset a failed or stuck stage and resubmit it.

        Args:
            stage_name (str | None): stage name
            dry_run (bool, optional): If True, only print text to screen and don't
            submit anything. Defaults to False.
            skip_script (bool, optional): If True, just submit and don't run the script.
            Defaults to False
            lost (bool, optional): If True, retry a SUBMITTED stage that is lost
            (see is_lost), marking it FAILED first. Defaults to False

        Raises:
            ValueError: if the stage can't be retried, or lost is True and the
            stage isn't lost (so a running stage is never resubmitted)
            ValueError: if this or a later stage's config is invalid (e.g. its
            script is missing); the stage is left as it was

        Returns:
            str | None: PBS job ID of the advance script, or None if all stages complete
        """
        stage = self._stage(stage_name) if stage_name else self.current_stage
        if stage is None:
            print(f"  {self.run_id}: nothing to retry")
            return None
        if lost:
            name = stage.config.name
            if not self.is_lost(name):
                raise ValueError(
                    f"Stage {name!r} is not lost — can only retry with lost=True "
                    "a SUBMITTED stage with no jobs still queued or running."
                )
            # is_lost re-read the state file, so look the stage up again
            stage = self._stage(name)
            # go through FAILED: SUBMITTED -> PENDING isn't an allowed transition
            stage.state.status = StageStatus.FAILED
            print(f"  {self.run_id} / {name}: lost, marked FAILED")
        if stage.state.status not in (StageStatus.PENDING, StageStatus.FAILED):
            raise ValueError(
                f"Stage {stage.config.name!r} is {stage.state.status.value} "
                "— can only retry PENDING or FAILED stages."
            )
        self._validate_stages(start=stage.config.name)
        stage.state.status = StageStatus.PENDING
        stage.state.end_time = None
        stage.state.submit_time = None
        stage.state.job_id = None
        stage.state.advance_job_id = None
        stage.state.fail_job_ids = []
        self.save()
        return self._submit_stage(stage, dry_run=dry_run, skip_script=skip_script)

    def _submit_stage(
        self,
        stage: Stage,
        depend_job_id: str | None = None,
        dry_run: bool = False,
        skip_script: bool = False,
    ) -> str:
        """Actually submit a stage

        Args:
            stage (Stage): stage
            depend_job_id (str | None, optional): CIME job id to add to afterok
                dependency. Defaults to None.
            dry_run (bool, optional): If True, only print text to screen and don't
            submit anything. Defaults to False.
            skip_script (bool, optional): If True, just submit and don't run the script.
                Defaults to False

        Raises:
            ValueError: Can only submit PENDING or FAILED stages

        Returns:
            str: _description_
        """

        # validation and submission
        stage.config.validate()
        if stage.state.status not in (StageStatus.PENDING, StageStatus.FAILED):
            raise ValueError(
                f"Stage {stage.config.name!r} is {stage.state.status.value} "
                "- can only submit PENDING or FAILED stages."
            )
        job_file = self._write_job_script(stage, depend_job_id, skip_script)
        job_id = (
            f"DRY_{self.run_id}_{stage.config.name}"
            if dry_run
            else self.scheduler.submit(job_file)
        )

        # record successful submission
        stage.state.job_id = job_id
        stage.state.status = StageStatus.SUBMITTED
        stage.state.submit_time = time.time()
        stage.state.increment_attempts()
        self.save()
        print(f"{self.run_id} / {stage.config.name}: submitted {job_id}")
        return job_id

    def _write_job_script(
        self, stage: Stage, depend_job_id: str | None = None, skip_script: bool = False
    ) -> Path:
        """Write the setup job script for one stage.

        The script:
        1. Runs the user's setup script (creates/configures/builds the CIME case)
        2. Submits the CIME case, captures its final PBS job ID
        3. Calls meristem run --submit-advance, which writes and submits the
             advance script in Python with the CIME job ID already baked in

        Args:
            stage (Stage): input Stage
            depend_job_id (str | None, optional): Depend Job ID. Defaults to None.
            skip_script (bool, optional): If True, skips the script and just submits

        Returns:
            Path: path to job script
        """
        case_root = self._case_root_for(stage)
        previous_stage = self._previous_stage(stage.config.name)
        previous_case_root = (
            self._case_root_for(previous_stage) if previous_stage else None
        )
        # CIME names a case after the basename of its case directory
        previous_case = previous_case_root.name if previous_case_root else ""

        job_name = f"{self.run_id}_{stage.config.name}"
        job_file = self.root / f"{job_name}.pbs"
        extra = "\n".join(f"#PBS {d}" for d in stage.config.extra_pbs)

        log_file = self.root / f"{job_name}.log"

        template = _ENV.get_template("submit_script_template.txt")
        template = template.render(
            meristem=meristem_cmd(),
            job_name=job_name,
            queue=stage.config.queue,
            select=stage.config.select,
            ncpus=stage.config.ncpus,
            memory=stage.config.memory,
            walltime=stage.config.walltime,
            project=self.project,
            log_file=log_file,
            email=self.email,
            depend_job_id=depend_job_id,
            extra=extra,
            root=self.root,
            name=stage.config.name,
            previous_case=previous_case,
            previous_case_root=previous_case_root or "",
            run_id=self.run_id,
            script=stage.config.script,
            case_root=case_root,
            cime=not stage.config.no_cime,
            skip_script=skip_script,
            submit_args=stage.config.submit_args,
            command=stage.config.command,
            post_command=stage.config.post_command,
        )
        job_file.write_text(template, encoding="utf-8")
        return job_file

    def _write_advance_script(self, stage: Stage, cime_job_id: str) -> Path:
        """Write the advance job script with the real CIME job ID already in
        the afterok dependency line

        Args:
            stage (Stage): Stage to advance
            cime_job_id (str): CIME job ID

        Returns:
            Path: path to script
        """
        job_name = f"{self.run_id}_{stage.config.name}_advance"
        job_file = self.root / f"{job_name}.pbs"

        template = _ENV.get_template("advance_script_template.txt")
        template = template.render(
            meristem=meristem_cmd(),
            job_name=job_name,
            queue=stage.config.queue,
            project=self.project,
            email=self.email,
            cime_job_id=cime_job_id,
            root=self.root,
            stage_name=stage.config.name,
        )
        job_file.write_text(template, encoding="utf-8")
        return job_file

    def _write_fail_script(
        self, stage: Stage, cime_job_id: str, suffix: str = "_fail"
    ) -> Path:
        """Write the fail job script with the real CIME job ID already in
        the afternotok dependency line

        Args:
            stage (Stage): Stage to fail
            cime_job_id (str): CIME job ID
            suffix (str, optional): appended to the job and file name, so each
                fail job of a stage gets its own script. Defaults to "_fail".

        Returns:
            Path: path to script
        """
        job_name = f"{self.run_id}_{stage.config.name}{suffix}"
        job_file = self.root / f"{job_name}.pbs"

        template = _ENV.get_template("fail_script_template.txt")
        template = template.render(
            meristem=meristem_cmd(),
            job_name=job_name,
            queue=stage.config.queue,
            project=self.project,
            email=self.email,
            cime_job_id=cime_job_id,
            root=self.root,
            stage_name=stage.config.name,
        )
        job_file.write_text(template, encoding="utf-8")
        return job_file


def _validate_config(config: dict) -> None:
    """Raise a descriptive ValueError for any missing or invalid config keys.

    Args:
        config: raw config dictionary (from YAML or passed directly)

    Raises:
        ValueError: if any required key is absent or malformed
    """
    missing = _REQUIRED_CONFIG_KEYS - config.keys()
    if missing:
        raise ValueError(
            f"Config is missing required key(s): {sorted(missing)}. "
            f"Required keys are: {sorted(_REQUIRED_CONFIG_KEYS)}"
        )
    if not isinstance(config["stages"], list):
        raise ValueError(
            f"Config key 'stages' must be a list, got {type(config['stages']).__name__!r}."
        )
    if len(config["stages"]) == 0:
        raise ValueError("Config key 'stages' must not be empty.")
    # command is for meristem's own setup steps (e.g. ensemble prepare); a
    # user's setup belongs in a script
    with_command = [
        stage.get("name", idx)
        for idx, stage in enumerate(config["stages"])
        if isinstance(stage, dict) and "command" in stage
    ]
    if with_command:
        raise ValueError(
            f"stage(s) {with_command} set 'command', which is reserved for "
            "meristem's own setup steps; use a setup script instead."
        )

    with_post_command = [
        stage.get("name", idx)
        for idx, stage in enumerate(config["stages"])
        if isinstance(stage, dict) and "post_command" in stage
    ]
    if with_post_command:
        raise ValueError(
            f"stage(s) {with_post_command} set 'post_command', which is reserved for "
            "meristem's own finalize scripts."
        )


def _load_config(config: Path | str | dict) -> dict:
    """Normalise *config* to a plain dict.

    Args:
        config: a dict, or a path to a YAML file.

    Returns:
        A plain dict with the config contents.

    Raises:
        TypeError: if *config* is not a dict, str, or Path.
    """
    if isinstance(config, dict):
        return config
    if isinstance(config, (str, Path)):
        with open(config, encoding="utf-8") as f:
            raw = os.path.expandvars(f.read())
            return yaml.safe_load(raw)
    raise TypeError(
        f"config must be a dict, str, or Path, got {type(config).__name__!r}."
    )
