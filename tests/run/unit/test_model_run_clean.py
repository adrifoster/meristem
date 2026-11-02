"""Tests for the ModelRun.clean method"""

from __future__ import annotations

from pathlib import Path

import pytest

from meristem.run.model_run import ModelRun
from meristem.run.scheduler import JobState

# pylint: disable=protected-access


def _job_files(run: ModelRun, stage: str) -> list:
    """Create a stage's setup, advance and fail job scripts and logs in the run root"""
    paths = []
    for suffix in ("", "_advance", "_fail", "_fail_1"):
        for ext in ("pbs", "log"):
            path = run.root / f"{run.run_id}_{stage}{suffix}.{ext}"
            path.touch()
            paths.append(path)
    return paths


def _fake_case(case_root: Path, rundir: Path, archive: Path) -> None:
    """Make a case dir with a stub xmlquery that reports the given RUNDIR and DOUT_S_ROOT"""
    case_root.mkdir(parents=True)
    xmlquery = case_root / "xmlquery"
    xmlquery.write_text(
        "#!/bin/bash\n"
        f'case "$1" in RUNDIR) echo {rundir} ;; DOUT_S_ROOT) echo {archive} ;; *) exit 1 ;; esac\n'
    )
    xmlquery.chmod(0o755)


def test_clean_targets_all_stages_ends_with_root(two_stage_run):
    """Test that _clean_targets() includes all the required items and that the last item
    is the run.root"""

    expected = []
    for stage in two_stage_run.stages:
        case_root = two_stage_run._case_root_for(stage)
        case_root.mkdir(parents=True)
        expected.append(case_root)
        expected.extend(_job_files(two_stage_run, stage.config.name))

    targets = two_stage_run._clean_targets()
    assert set(expected) <= set(targets)
    assert targets[-1] == two_stage_run.root


def test_clean_targets_one_stage(two_stage_run):
    """Test that _clean_targets appropriately targets only stage's files and dirs"""
    expected = []
    not_expected = []
    for stage in two_stage_run.stages:
        case_root = two_stage_run._case_root_for(stage)
        case_root.mkdir(parents=True)
        job_files = _job_files(two_stage_run, stage.config.name)
        if stage.config.name == "spinup_ad":
            expected.append(case_root)
            expected.extend(job_files)
        else:
            not_expected.append(case_root)
            not_expected.extend(job_files)

    targets = two_stage_run._clean_targets(["spinup_ad"])
    assert set(expected) <= set(targets)
    assert not set(not_expected) <= set(targets)
    assert two_stage_run.root not in targets


def test_clean_targets_skips_similar_stage_name(created_run):
    """Test that _clean_targets skips things with a similar stage name"""

    for stage in created_run.stages:
        case_root = created_run._case_root_for(stage)
        case_root.mkdir(parents=True)
        _ = _job_files(created_run, stage.config.name)

    path = created_run.root / "member_0001_spinup_ad_x.pbs"
    path.touch()

    targets = created_run._clean_targets(["spinup_ad"])
    assert path not in targets


def test_clean_targets_skips_missing_case_dir(created_run):
    """Test that _clean_targets skips a case_dir if it doesn't exist"""
    for stage in created_run.stages:
        _ = _job_files(created_run, stage.config.name)

    targets = created_run._clean_targets(["spinup_ad"], include_archive=True)
    assert created_run._case_root_for(created_run.stages[0]) not in targets


def test_clean_targets_case_root_outside_run(tmp_path, run_config_dict, fake_scheduler):
    """Test that _clean_targets works with a case_root outside the run root"""

    outside = tmp_path / "elsewhere" / "case"
    run_config_dict["stages"][0]["case_root"] = str(outside)
    run = ModelRun.create(run_config_dict, scheduler=fake_scheduler)

    for stage in run.stages:
        case_root = run._case_root_for(stage)
        case_root.mkdir(parents=True)
        _ = _job_files(run, stage.config.name)

    targets = run._clean_targets()
    assert outside in targets


def test_clean_targets_unknown_stage_raises(created_run):
    """Test that _clean_targets raises a ValueError for an unknown stage"""

    for stage in created_run.stages:
        case_root = created_run._case_root_for(stage)
        case_root.mkdir(parents=True)
        _ = _job_files(created_run, stage.config.name)

    with pytest.raises(ValueError):
        created_run._clean_targets(["nope"])


def test_clean_targets_includes_rundir(tmp_path, created_run):
    """Test that a stage's CIME run dir is cleaned with its case, but not its
    archive unless asked
    """
    case_root = created_run.root / "spinup_ad"
    rundir = tmp_path / "scratch" / "spinup_ad" / "run"
    archive = tmp_path / "archive" / "spinup_ad"
    _fake_case(case_root, rundir, archive)
    rundir.mkdir(parents=True)
    archive.mkdir(parents=True)

    targets = created_run._clean_targets()

    assert rundir in targets
    assert archive not in targets


def test_clean_targets_with_output_includes_outdir(tmp_path, created_run):
    """Test that a stage's CIME run dir is cleaned with its case along with run_dir
    and archive_dir when asked
    """
    case_root = created_run.root / "spinup_ad"
    rundir = tmp_path / "scratch" / "spinup_ad" / "run"
    archive = tmp_path / "archive" / "spinup_ad"
    _fake_case(case_root, rundir, archive)
    rundir.mkdir(parents=True)
    archive.mkdir(parents=True)

    targets = created_run._clean_targets(include_archive=True)

    assert rundir in targets
    assert archive in targets


def test_clean_targets_skips_output_without_case_name(tmp_path, created_run):
    """Test that a RUNDIR or archive that isn't named for the case (e.g. a
    misconfigured case pointing at a shared scratch dir) is never cleaned
    """
    case_root = created_run.root / "spinup_ad"
    shared_scratch = tmp_path / "scratch"  # no "spinup_ad" in this path
    shared_archive = tmp_path / "archive"
    _fake_case(case_root, shared_scratch, shared_archive)
    shared_scratch.mkdir()
    shared_archive.mkdir()

    targets = created_run._clean_targets(include_archive=True)

    assert shared_scratch not in targets
    assert shared_archive not in targets


def _make_stage_files(run: ModelRun) -> None:
    """Create every stage's (default) case dir and job files"""
    for stage in run.stages:
        (run.root / stage.config.name).mkdir(parents=True)
        _job_files(run, stage.config.name)


def _make_active(run: ModelRun, fake_scheduler) -> str:
    """Give the run's first stage a setup job the scheduler reports as running"""
    run.stages[0].state.job_id = "1.fake"
    fake_scheduler.job_states = {"1.fake": JobState.RUNNING}
    return "1.fake"


def test_clean_deletes_everything(tmp_path, created_run):
    """Test that cleaning the whole run deletes the run root and the run dir"""
    _make_stage_files(created_run)
    rundir = tmp_path / "scratch" / "spinup_ad" / "run"
    rundir.mkdir(parents=True)
    (created_run.root / "spinup_ad" / "xmlquery").write_text(
        f"#!/bin/bash\necho {rundir}\n"
    )
    (created_run.root / "spinup_ad" / "xmlquery").chmod(0o755)

    targets = created_run.clean()

    assert targets[-1] == created_run.root
    assert not created_run.root.exists()
    assert not rundir.exists()


def test_clean_one_stage_keeps_others(two_stage_run, fake_scheduler):
    """Test that cleaning one stage deletes only its case dir and job files,
    and leaves the run's state alone
    """
    _make_stage_files(two_stage_run)
    statuses = [stage.state.status for stage in two_stage_run.stages]

    targets = two_stage_run.clean(["spinup_ad"])

    assert targets
    assert not any(path.exists() for path in targets)
    assert (two_stage_run.root / "spinup_sasu").is_dir()
    assert len(list(two_stage_run.root.glob("*_spinup_sasu*"))) == 8
    reloaded = ModelRun.load(two_stage_run.root, scheduler=fake_scheduler)
    assert [stage.state.status for stage in reloaded.stages] == statuses


def test_clean_dry_run_deletes_nothing(created_run):
    """Test that a dry run returns the paths but deletes none of them"""
    _make_stage_files(created_run)

    targets = created_run.clean(dry_run=True)

    assert targets
    assert all(path.exists() for path in targets)


@pytest.mark.parametrize("dry_run", [False, True])
def test_clean_active_jobs_raises(created_run, fake_scheduler, dry_run):
    """Test that clean (even a dry run) refuses while jobs are active, naming
    them, and neither cancels nor deletes anything
    """
    _make_stage_files(created_run)
    job_id = _make_active(created_run, fake_scheduler)

    with pytest.raises(ValueError, match=job_id):
        created_run.clean(dry_run=dry_run)

    assert not fake_scheduler.canceled
    assert (created_run.root / "spinup_ad").is_dir()


def test_clean_cancel(created_run, fake_scheduler):
    """Test that clean with cancel cancels the active jobs, then deletes"""
    _make_stage_files(created_run)
    job_id = _make_active(created_run, fake_scheduler)

    created_run.clean(cancel=True)

    assert fake_scheduler.canceled == [[job_id]]
    assert not created_run.root.exists()


def test_clean_cancel_dry_run_does_not_cancel(created_run, fake_scheduler):
    """Test that a dry run with cancel neither cancels nor deletes"""
    _make_stage_files(created_run)
    _make_active(created_run, fake_scheduler)

    targets = created_run.clean(cancel=True, dry_run=True)

    assert not fake_scheduler.canceled
    assert all(path.exists() for path in targets)


def test_clean_skips_missing_paths(created_run, monkeypatch):
    """Test that a target already gone when its turn comes (e.g. deleted by
    hand, or inside a dir deleted earlier) is skipped
    """
    _make_stage_files(created_run)
    gone = created_run.root / "already_gone.log"
    real_targets = created_run._clean_targets
    monkeypatch.setattr(
        created_run,
        "_clean_targets",
        lambda *args: [gone, *real_targets(*args)],
    )

    targets = created_run.clean()

    assert targets[0] == gone
    assert not created_run.root.exists()


def test_clean_does_not_follow_symlink(tmp_path, created_run):
    """Test that a symlink inside a cleaned case dir is removed without
    deleting what it points to
    """
    _make_stage_files(created_run)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").touch()
    (created_run.root / "spinup_ad" / "link").symlink_to(outside)

    created_run.clean(["spinup_ad"])

    assert (outside / "keep.txt").exists()
    assert not (created_run.root / "spinup_ad").exists()


def test_clean_symlink_target_is_unlinked(tmp_path, created_run):
    """Test that a target that is itself a symlink (e.g. a case_root pointing
    elsewhere) is unlinked rather than followed
    """
    outside = tmp_path / "real_case"
    outside.mkdir()
    (outside / "keep.txt").touch()
    (created_run.root / "spinup_ad").symlink_to(outside)

    created_run.clean(["spinup_ad"])

    assert not (created_run.root / "spinup_ad").is_symlink()
    assert (outside / "keep.txt").exists()
