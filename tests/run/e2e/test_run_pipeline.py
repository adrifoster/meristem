"""E2E tests for mermistem run"""

# pylint: disable=protected-access

from __future__ import annotations

import json
from pathlib import Path

import subprocess
import pytest

from tests.run.e2e.conftest import CREATE_CASE

pytestmark = pytest.mark.e2e


def test_dry_run_submits_nothing(make_run, fake, meristem):
    """Test submit with --dry-run does nothing"""
    root = make_run([{"name": "stage1"}])
    result = meristem("run", "submit", "--root", str(root), "--dry-run")
    assert result.returncode == 0, result.stderr
    assert fake.jobs() == []


def test_single_stage_runs_to_done(make_run, fake, meristem):
    """Test that a single stage pipeline works"""
    root = make_run([{"name": "stage1"}])

    result = meristem("run", "submit", "--root", str(root))
    assert result.returncode == 0, result.stderr
    fake.run_until_idle()

    jobs = fake.jobs()
    assert [j["name"] for j in jobs] == [
        "e2e_test_stage1",
        "stage1.run",
        "stage1.st_archive",
        "e2e_test_stage1_advance",
        "e2e_test_stage1_fail_1",
        "e2e_test_stage1_fail_2",
    ]

    result = meristem("run", "status", "--root", str(root))
    assert result.returncode == 0, result.stdout + result.stderr
    stage_line = next(l for l in result.stdout.splitlines() if "stage1" in l)
    assert "✓" in stage_line, result.stdout

    jobs = fake.jobs()
    assert all(j["state"] == "F" for j in jobs), fake.logs()
    assert all(j["exit"] in (0, None) for j in jobs), fake.logs()
    deleted = [j for j in jobs if j["deleted"]]
    assert deleted, "expected the fail jobs to have been deleted"
    for job in deleted:
        assert not fake._log_path(job).exists(), fake.logs()


def test_two_stages_tether(make_run, fake, meristem):
    """Test that a two-stage pipeline"""
    root = make_run([{"name": "stage1"}, {"name": "stage2"}])

    assert meristem("run", "submit", "--root", str(root)).returncode == 0
    fake.run_until_idle()

    result = meristem("run", "status", "--root", str(root))
    lines = result.stdout.splitlines()
    for name in ("stage1", "stage2"):
        line = next(l for l in lines if name in l)
        assert "✓" in line, result.stdout + fake.logs()


def test_three_stages_plus_final(
    make_run, stage_script, fake, meristem, tmp_path
):  # pylint: disable=too-many-locals
    """Test that a thre-stage pipeline"""
    record = tmp_path / "args.txt"
    # $2/$3 don't exist for the first stage, so ${2:-} avoids a set -u error
    log_args = f'echo "$1|${{2:-}}|${{3:-}}" >> {record}'
    cime_body = f"{log_args}\n{CREATE_CASE}"

    root = make_run(
        [
            {"name": "stage1", "script": stage_script(cime_body, name="stage1")},
            {"name": "stage2", "script": stage_script(cime_body, name="stage2")},
            {"name": "stage3", "script": stage_script(cime_body, name="stage3")},
            {
                "name": "final",
                "script": stage_script(log_args, name="final"),
                "no_cime": True,
            },
        ]
    )

    assert meristem("run", "submit", "--root", str(root)).returncode == 0
    fake.run_until_idle()

    # every stage is done
    result = meristem("run", "status", "--root", str(root))
    assert result.returncode == 0, result.stdout + result.stderr
    stage_lines = result.stdout.splitlines()[1:]
    for name in ("stage1", "stage2", "stage3", "final"):
        line = next(l for l in stage_lines if name in l)
        assert "✓" in line, result.stdout + fake.logs()
    # each stage was handed the previous stage's case name and root
    rows = [l.split("|") for l in record.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4, rows
    assert rows[0][1:] == ["", ""]
    for prev, cur in zip(rows, rows[1:]):
        assert cur[2] == prev[0], rows
        assert cur[1] == Path(prev[0]).name, rows

    # fail jobs were deleted, never run
    fail_jobs = [j for j in fake.jobs() if "_fail_" in j["name"]]
    assert fail_jobs
    for job in fail_jobs:
        assert job["deleted"] is True
        assert not fake._log_path(job).exists()


def test_no_archive(make_run, fake, meristem):
    """Test that a single-stage pipeline with no archive"""
    fake.env["FAKE_CIME"] = json.dumps({"no_archive": True})
    root = make_run([{"name": "stage1"}])

    assert meristem("run", "submit", "--root", str(root)).returncode == 0
    fake.run_until_idle()

    result = meristem("run", "status", "--root", str(root))
    line = next(l for l in result.stdout.splitlines()[1:] if "stage1" in l)
    assert "✓" in line, result.stdout + fake.logs()

    jobs = {j["name"]: j for j in fake.jobs()}
    assert sorted(jobs) == [
        "e2e_test_stage1",
        "e2e_test_stage1_advance",
        "e2e_test_stage1_fail",
        "stage1.run",
    ], fake.logs()

    run_id = jobs["stage1.run"]["id"]
    assert jobs["e2e_test_stage1_advance"]["depend"] == ["afterok", run_id]
    assert jobs["e2e_test_stage1_fail"]["depend"] == ["afternotok", run_id]
    assert jobs["e2e_test_stage1_fail"]["deleted"] is True


def clear_faults(fake, root, stage_script):
    """Clear injected faults"""
    fake.env.pop("FAKE_CIME", None)
    (root / "stage1" / "fake_cime.json").unlink(missing_ok=True)
    (fake.spool / "control.json").unlink(missing_ok=True)
    stage_script(name="stage1")


def test_setup_script_fails(make_run, stage_script, fake, meristem):
    """Test that an injected script failure is logged"""
    bad = stage_script("exit 3", name="stage1")
    root = make_run([{"name": "stage1", "script": bad}, {"name": "stage2"}])
    meristem("run", "submit", "--root", str(root))
    fake.run_until_idle()

    result = meristem("run", "status", "--root", str(root))
    lines = result.stdout.splitlines()[1:]
    assert "✗" in next(l for l in lines if "stage1" in l), result.stdout + fake.logs()
    assert "·" in next(l for l in lines if "stage2" in l), result.stdout + fake.logs()
    assert all(j["state"] == "F" for j in fake.jobs()), fake.logs()


def test_setup_script_fails_then_retry(make_run, stage_script, fake, meristem):
    """Test that an injected script failure is logged and then retry works"""
    bad = stage_script("exit 3", name="stage1")
    root = make_run([{"name": "stage1", "script": bad}, {"name": "stage2"}])
    meristem("run", "submit", "--root", str(root))
    fake.run_until_idle()

    clear_faults(fake, root, stage_script)
    result = meristem("run", "retry", "--root", str(root))
    assert result.returncode == 0, result.stdout + result.stderr
    fake.run_until_idle()

    result = meristem("run", "status", "--root", str(root))
    lines = result.stdout.splitlines()[1:]
    assert "✓" in next(l for l in lines if "stage1" in l), result.stdout + fake.logs()
    assert "✓" in next(l for l in lines if "stage2" in l), result.stdout + fake.logs()
    assert "attempts=2" in next(l for l in lines if "stage1" in l), result.stdout


def test_case_submit_fails(make_run, fake, meristem):
    """Test that an injected submit failure is logged"""
    fake.env["FAKE_CIME"] = json.dumps({"case.submit": "fail"})
    root = make_run([{"name": "stage1"}, {"name": "stage2"}])
    meristem("run", "submit", "--root", str(root))
    fake.run_until_idle()

    result = meristem("run", "status", "--root", str(root))
    lines = result.stdout.splitlines()[1:]
    assert "✗" in next(l for l in lines if "stage1" in l), result.stdout + fake.logs()
    assert "·" in next(l for l in lines if "stage2" in l), result.stdout + fake.logs()
    assert all(j["state"] == "F" for j in fake.jobs()), fake.logs()


def test_case_submit_fails_then_retry(make_run, stage_script, fake, meristem):
    """Test that an injected submit failure is logged and then retry works"""
    fake.env["FAKE_CIME"] = json.dumps({"case.submit": "fail"})
    root = make_run([{"name": "stage1"}, {"name": "stage2"}])
    meristem("run", "submit", "--root", str(root))
    fake.run_until_idle()

    clear_faults(fake, root, stage_script)
    result = meristem("run", "retry", "--root", str(root))
    assert result.returncode == 0, result.stdout + result.stderr
    fake.run_until_idle()

    result = meristem("run", "status", "--root", str(root))
    lines = result.stdout.splitlines()[1:]
    assert "✓" in next(l for l in lines if "stage1" in l), result.stdout + fake.logs()
    assert "✓" in next(l for l in lines if "stage2" in l), result.stdout + fake.logs()
    assert "attempts=2" in next(l for l in lines if "stage1" in l), result.stdout


def test_blank_job_ids(make_run, fake, meristem):
    """Test that blank job_ids results in a failure"""
    fake.env["FAKE_CIME"] = json.dumps({"job_ids": ""})
    root = make_run([{"name": "stage1"}, {"name": "stage2"}])
    meristem("run", "submit", "--root", str(root))
    fake.run_until_idle()

    result = meristem("run", "status", "--root", str(root))
    lines = result.stdout.splitlines()[1:]
    assert "✗" in next(l for l in lines if "stage1" in l), result.stdout + fake.logs()
    assert "·" in next(l for l in lines if "stage2" in l), result.stdout + fake.logs()
    assert all(j["state"] == "F" for j in fake.jobs()), fake.logs()


def test_blank_job_ids_then_retry(make_run, stage_script, fake, meristem):
    """Test that blank job_ids results in a failure and then retry works"""
    fake.env["FAKE_CIME"] = json.dumps({"job_ids": ""})
    root = make_run([{"name": "stage1"}, {"name": "stage2"}])
    meristem("run", "submit", "--root", str(root))
    fake.run_until_idle()

    clear_faults(fake, root, stage_script)
    result = meristem("run", "retry", "--root", str(root))
    assert result.returncode == 0, result.stdout + result.stderr
    fake.run_until_idle()

    result = meristem("run", "status", "--root", str(root))
    lines = result.stdout.splitlines()[1:]
    assert "✓" in next(l for l in lines if "stage1" in l), result.stdout + fake.logs()
    assert "✓" in next(l for l in lines if "stage2" in l), result.stdout + fake.logs()
    assert "attempts=2" in next(l for l in lines if "stage1" in l), result.stdout


def test_case_run_fails(make_run, fake, meristem):
    """Test that an injected failure in case.run is logged"""
    fake.env["FAKE_CIME"] = json.dumps({"case.run": "fail"})
    root = make_run([{"name": "stage1"}, {"name": "stage2"}])
    meristem("run", "submit", "--root", str(root))
    fake.run_until_idle()

    result = meristem("run", "status", "--root", str(root))
    lines = result.stdout.splitlines()[1:]
    assert "✗" in next(l for l in lines if "stage1" in l), result.stdout + fake.logs()
    assert "·" in next(l for l in lines if "stage2" in l), result.stdout + fake.logs()
    assert all(j["state"] == "F" for j in fake.jobs()), fake.logs()

    fail_1 = next(j for j in fake.jobs() if j["name"].endswith("stage1_fail_1"))
    assert fail_1["deleted"] is False and fail_1["exit"] == 0, fake.logs()


def test_case_run_fails_then_retry(make_run, stage_script, fake, meristem):
    """Test that an injected failure in case.run is logged and then retry works"""
    fake.env["FAKE_CIME"] = json.dumps({"case.run": "fail"})
    root = make_run([{"name": "stage1"}, {"name": "stage2"}])
    meristem("run", "submit", "--root", str(root))
    fake.run_until_idle()

    clear_faults(fake, root, stage_script)
    result = meristem("run", "retry", "--root", str(root))
    assert result.returncode == 0, result.stdout + result.stderr
    fake.run_until_idle()

    result = meristem("run", "status", "--root", str(root))
    lines = result.stdout.splitlines()[1:]
    assert "✓" in next(l for l in lines if "stage1" in l), result.stdout + fake.logs()
    assert "✓" in next(l for l in lines if "stage2" in l), result.stdout + fake.logs()
    assert "attempts=2" in next(l for l in lines if "stage1" in l), result.stdout


def test_st_archive_fails(make_run, fake, meristem):
    """Test that an injected failure in st_archive is logged"""
    fake.env["FAKE_CIME"] = json.dumps({"st_archive": "fail"})
    root = make_run([{"name": "stage1"}, {"name": "stage2"}])
    meristem("run", "submit", "--root", str(root))
    fake.run_until_idle()

    result = meristem("run", "status", "--root", str(root))
    lines = result.stdout.splitlines()[1:]
    assert "✗" in next(l for l in lines if "stage1" in l), result.stdout + fake.logs()
    assert "·" in next(l for l in lines if "stage2" in l), result.stdout + fake.logs()
    assert all(j["state"] == "F" for j in fake.jobs()), fake.logs()

    fail_2 = next(j for j in fake.jobs() if j["name"].endswith("stage1_fail_2"))
    assert fail_2["deleted"] is False and fail_2["exit"] == 0, fake.logs()


def test_st_archive_fails_then_retry(make_run, stage_script, fake, meristem):
    """Test that an injected failure in st_archive is logged and retry works"""
    fake.env["FAKE_CIME"] = json.dumps({"st_archive": "fail"})
    root = make_run([{"name": "stage1"}, {"name": "stage2"}])
    meristem("run", "submit", "--root", str(root))
    fake.run_until_idle()

    clear_faults(fake, root, stage_script)
    result = meristem("run", "retry", "--root", str(root))
    assert result.returncode == 0, result.stdout + result.stderr
    fake.run_until_idle()

    result = meristem("run", "status", "--root", str(root))
    lines = result.stdout.splitlines()[1:]
    assert "✓" in next(l for l in lines if "stage1" in l), result.stdout + fake.logs()
    assert "✓" in next(l for l in lines if "stage2" in l), result.stdout + fake.logs()
    assert "attempts=2" in next(l for l in lines if "stage1" in l), result.stdout


def test_advance_qsub_fails(make_run, fake, meristem):
    """Test that an injected failure in qsub is logged"""
    fake.fail_qsub(r"_advance\.pbs$")
    root = make_run([{"name": "stage1"}, {"name": "stage2"}])
    meristem("run", "submit", "--root", str(root))
    fake.run_until_idle()

    result = meristem("run", "status", "--root", str(root))
    lines = result.stdout.splitlines()[1:]
    assert "✗" in next(l for l in lines if "stage1" in l), result.stdout + fake.logs()
    assert "·" in next(l for l in lines if "stage2" in l), result.stdout + fake.logs()
    assert all(j["state"] == "F" for j in fake.jobs()), fake.logs()


def test_advance_qsub_fails_then_retry(make_run, stage_script, fake, meristem):
    """Test that an injected failure in qsub is logged and retry works"""
    fake.fail_qsub(r"_advance\.pbs$")
    root = make_run([{"name": "stage1"}, {"name": "stage2"}])
    meristem("run", "submit", "--root", str(root))
    fake.run_until_idle()

    clear_faults(fake, root, stage_script)
    result = meristem("run", "retry", "--root", str(root))
    assert result.returncode == 0, result.stdout + result.stderr
    fake.run_until_idle()

    result = meristem("run", "status", "--root", str(root))
    lines = result.stdout.splitlines()[1:]
    assert "✓" in next(l for l in lines if "stage1" in l), result.stdout + fake.logs()
    assert "✓" in next(l for l in lines if "stage2" in l), result.stdout + fake.logs()
    assert "attempts=2" in next(l for l in lines if "stage1" in l), result.stdout


def test_next_stage_script_missing_fails_it_then_retry(
    make_run, fake, meristem, tmp_path
):
    """Test that when the advance job can't submit the next stage (its script
    was removed), the finished stage is done and the next stage failed, and a
    plain retry resubmits only the next stage
    """
    root = make_run([{"name": "stage1"}, {"name": "stage2"}])
    assert meristem("run", "submit", "--root", str(root)).returncode == 0
    script = tmp_path / "stage2.sh"
    body = script.read_text(encoding="utf-8")
    script.unlink()
    fake.run_until_idle()

    result = meristem("run", "status", "--root", str(root))
    lines = result.stdout.splitlines()[1:]
    assert "✓" in next(l for l in lines if "stage1" in l), result.stdout + fake.logs()
    assert "✗" in next(l for l in lines if "stage2" in l), result.stdout + fake.logs()
    advance = next(j for j in fake.jobs() if j["name"] == "e2e_test_stage1_advance")
    assert advance["exit"] != 0, fake.logs()
    assert "cannot open script" in fake.logs()

    script.write_text(body, encoding="utf-8")
    result = meristem("run", "retry", "--root", str(root))
    assert result.returncode == 0, result.stdout + result.stderr
    fake.run_until_idle()

    result = meristem("run", "status", "--root", str(root))
    lines = result.stdout.splitlines()[1:]
    for name in ("stage1", "stage2"):
        line = next(l for l in lines if name in l)
        # status only shows attempts once a stage has been submitted twice
        assert "✓" in line and "attempts=" not in line, result.stdout + fake.logs()
    state = json.loads((root / "run_state.json").read_text(encoding="utf-8"))
    assert [s["attempts"] for s in state["stages"]] == [1, 1], state


def _stage_states(root: Path) -> dict[str, tuple[str, int]]:
    """Each stage's (status, attempts), as saved"""
    state = json.loads((root / "run_state.json").read_text(encoding="utf-8"))
    return {s["name"]: (s["status"], s["attempts"]) for s in state["stages"]}


def test_failed_attempt_with_live_jobs_then_retry(make_run, fake, meristem):
    """Test a setup job that fails after case.submit, leaving its CIME and
    advance jobs queued: retry is refused while they're active, and once retried
    a leftover advance job from the old attempt is ignored rather than
    finishing the new attempt
    """
    root = make_run([{"name": "stage1"}, {"name": "stage2"}])
    fake.fail_qsub(r"_fail(_\d+)?\.pbs$")  # fails the setup job after case.submit
    assert meristem("run", "submit", "--root", str(root)).returncode == 0
    fake.step()  # stage1's setup job
    assert _stage_states(root)["stage1"] == ("failed", 1), fake.logs()
    advance = next(j for j in fake.jobs() if j["name"] == "e2e_test_stage1_advance")
    assert advance["state"] == "Q"

    # attempt 1's case.run, st_archive and advance jobs are still queued
    result = meristem("run", "retry", "--root", str(root))
    assert result.returncode == 1
    assert "still queued or running" in result.stderr
    assert advance["id"] in result.stderr
    assert _stage_states(root)["stage1"] == ("failed", 1)

    # attempt 1's CIME jobs finish; its advance job is gone (killed here)
    (fake.spool / "control.json").unlink()
    fake.step()  # case.run
    fake.step()  # st_archive
    fake.kill(advance["id"])
    result = meristem("run", "retry", "--root", str(root))
    assert result.returncode == 0, result.stdout + result.stderr
    assert _stage_states(root)["stage1"] == ("submitted", 2)

    # attempt 1's advance script runs anyway (e.g. the scheduler lost track of
    # it): it's ignored instead of marking attempt 2 done and starting stage2
    stale = subprocess.run(
        ["bash", advance["script"]],
        env=fake.env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert stale.returncode == 0, stale.stdout + stale.stderr
    assert "ignoring advance from attempt 1" in stale.stdout
    assert _stage_states(root) == {
        "stage1": ("submitted", 2),
        "stage2": ("pending", 0),
    }

    fake.run_until_idle()
    assert _stage_states(root) == {
        "stage1": ("done", 2),
        "stage2": ("done", 1),
    }, fake.logs()


def test_setup_fails_before_advance_job_cime_still_tracked(make_run, fake, meristem):
    """Test a setup job that fails after case.submit but before submitting any
    advance or fail job: its CIME jobs are still recorded, so retry is refused
    while they run (instead of starting a second CIME run on the same case),
    and allowed once they finish
    """
    root = make_run([{"name": "stage1"}])
    fake.fail_qsub(r"_advance\.pbs$")  # only the advance job's qsub fails
    assert meristem("run", "submit", "--root", str(root)).returncode == 0
    fake.step()  # the setup job
    jobs = {j["name"]: j for j in fake.jobs()}
    assert sorted(jobs) == ["e2e_test_stage1", "stage1.run", "stage1.st_archive"]
    state = json.loads((root / "run_state.json").read_text(encoding="utf-8"))
    stage = state["stages"][0]
    assert stage["status"] == "failed", fake.logs()
    assert stage["advance_job_id"] is None and stage["fail_job_ids"] == []
    run_id, archive_id = jobs["stage1.run"]["id"], jobs["stage1.st_archive"]["id"]
    assert stage["cime_job_ids"] == [run_id, archive_id]

    result = meristem("run", "retry", "--root", str(root))
    assert result.returncode == 1
    assert "still queued or running" in result.stderr
    assert run_id in result.stderr and archive_id in result.stderr

    (fake.spool / "control.json").unlink()
    fake.run_until_idle()  # the CIME jobs finish; nothing else was queued
    result = meristem("run", "retry", "--root", str(root))
    assert result.returncode == 0, result.stdout + result.stderr
    fake.run_until_idle()
    assert _stage_states(root) == {"stage1": ("done", 2)}, fake.logs()


def test_killed_setup_job_is_lost_then_retry(make_run, fake, meristem):
    """Test that a killed setup job is lost and retry works"""
    root = make_run([{"name": "stage1"}])

    assert meristem("run", "submit", "--root", str(root)).returncode == 0
    fake.kill(fake.jobs()[0]["id"])

    result = meristem("run", "status", "--root", str(root))
    assert result.returncode == 3, result.stdout + result.stderr
    line = next(l for l in result.stdout.splitlines()[1:] if "stage1" in l)
    assert "lost" in line, result.stdout

    result = meristem("run", "retry", "--root", str(root), "--lost")
    assert result.returncode == 0, result.stdout + result.stderr
    fake.run_until_idle()

    result = meristem("run", "status", "--root", str(root))
    assert result.returncode == 0, result.stdout + fake.logs()
    line = next(l for l in result.stdout.splitlines()[1:] if "stage1" in l)
    assert "✓" in line, result.stdout + fake.logs()


def test_case_run_killed_with_fail_jobs_deleted_is_lost(make_run, fake, meristem):
    """Test that a killed with fail jobs is deleted and lost"""
    root = make_run([{"name": "stage1"}])
    meristem("run", "submit", "--root", str(root))

    # runs only the setup job, which queues case.run, st_archive, advance and the fail jobs
    fake.step()

    jobs = {j["name"]: j for j in fake.jobs()}
    fake.kill(jobs["stage1.run"]["id"])
    for name in ("e2e_test_stage1_fail_1", "e2e_test_stage1_fail_2"):
        subprocess.run(["qdel", jobs[name]["id"]], env=fake.env, check=True)

    fake.run_until_idle()  # st_archive and advance get deleted

    result = meristem("run", "status", "--root", str(root))
    assert result.returncode == 3, result.stdout + result.stderr + fake.logs()
    line = next(l for l in result.stdout.splitlines()[1:] if "stage1" in l)
    assert "lost" in line, result.stdout + fake.logs()


def test_broken_qstat_is_not_reported_as_lost(make_run, fake, meristem):
    """Test that a broken qstat is not reported as lost"""
    root = make_run([{"name": "stage1"}])
    meristem("run", "submit", "--root", str(root))

    fake.break_qstat()

    result = meristem("run", "status", "--root", str(root))
    output = result.stdout + result.stderr
    assert result.returncode not in (0, 3), output
    assert "qstat failed" in output, output
    assert "lost" not in output, output

    result = meristem("run", "retry", "--root", str(root), "--lost")
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "qstat failed" in output, output
    assert "lost" not in output.replace("--lost", ""), output
