"""End-to-end tests for meristem ensemble"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from meristem.ensemble.cli import EXIT_LOST
from meristem.ensemble.model_ensemble import Ensemble
from tests.fakes.cime import CASE_DICT_FILE

pytestmark = pytest.mark.e2e

_TWO_STAGES = [
    {"name": "spinup_ad", "queue": "develop"},
    {"name": "transient", "queue": "develop"},
]


def _assert_case_configured(case_root: Path, param_file: Path, fake):
    """Check a member stage's case was cloned from the fake base case, has the
    member's parameter file in user_nl_clm and lnd_in, and was run and archived
    """
    assert case_root.is_dir(), fake.logs()
    case = json.loads((case_root / CASE_DICT_FILE).read_text(encoding="utf-8"))
    assert case["CASE"] == case_root.name
    assert case["PROJECT"] == "FAKEPROJ"
    assert case["BUILD_COMPLETE"] == "TRUE"
    assert case["CIMEROOT"] == str(fake.cimeroot)

    expected = f"fates_paramfile = '{param_file}'"
    assert _paramfile_lines(case_root) == [expected]
    lnd_in = (case_root / "CaseDocs" / "lnd_in").read_text(encoding="utf-8")
    assert expected in lnd_in.splitlines()

    runs = (case_root / "runs.log").read_text(encoding="utf-8").splitlines()
    assert runs == [
        f"case.run {case_root.name}",
        f"case.st_archive {case_root.name}",
    ]


def _paramfile_lines(case_root: Path) -> list[str]:
    """The fates_paramfile lines in a case's user_nl_clm"""
    text = (case_root / "user_nl_clm").read_text(encoding="utf-8")
    return [line for line in text.splitlines() if line.startswith("fates_paramfile")]


def _assert_jobs_finished_cleanly(fake):
    """Check every job finished with exit 0 (or never ran), the fail jobs were
    deleted without running, and nothing else was deleted
    """
    jobs = fake.jobs()
    assert all(j["state"] == "F" for j in jobs), fake.logs()
    assert all(j["exit"] in (0, None) for j in jobs), fake.logs()
    fail_jobs = [j for j in jobs if "_fail_" in j["name"]]
    assert fail_jobs
    assert all(j["deleted"] and j["log"] is None for j in fail_jobs)
    assert all(not j["deleted"] for j in jobs if "_fail_" not in j["name"])


def test_dry_run_submits_nothing(make_ensemble, fake, meristem):
    """Test submit with --dry-run does nothing"""
    root = make_ensemble()
    result = meristem("ensemble", "submit", "--root", str(root), "--dry-run")
    assert result.returncode == 0, result.stderr
    assert fake.jobs() == []

    result = meristem("ensemble", "status", "--root", str(root))
    assert result.returncode == 0, result.stderr

    out = result.stdout
    assert "3 members" in out
    row = next(l for l in out.splitlines() if l.startswith("spinup_ad"))
    assert row.split() == ["spinup_ad", "3", "0", "0", "0"]


def test_single_stage_runs_to_done(run_ensemble, fake, meristem):
    """Test that an ensemble with a single stage pipeline works"""
    root = run_ensemble()

    result = meristem("ensemble", "status", "--root", str(root))
    assert result.returncode == 0, result.stderr

    out = result.stdout
    assert "3 members" in out
    row = next(l for l in out.splitlines() if l.startswith("spinup_ad"))
    assert row.split() == ["spinup_ad", "0", "0", "3", "0"]

    _assert_jobs_finished_cleanly(fake)


def test_each_member_case_is_cloned_and_configured(run_ensemble, fake, members):
    """Test that each member case is correctly cloned and configured"""
    root = run_ensemble()
    ensemble = Ensemble.load(root)
    status = ensemble.status()
    assert len(status) == len(members), fake.logs()

    for row in status.itertuples():
        _assert_case_configured(
            Path(str(row.case_root)), ensemble.param_files[str(row.member)], fake
        )


def test_members_get_distinct_cases_and_paramfiles(run_ensemble, fake, members):
    """Test that every member has its own case, set to its own parameter file"""
    root = run_ensemble()
    status = Ensemble.load(root).status()
    case_roots = [Path(str(c)) for c in status["case_root"]]
    assert len(set(case_roots)) == len(members), fake.logs()
    paramfile_lines = [tuple(_paramfile_lines(c)) for c in case_roots]
    assert len(set(paramfile_lines)) == len(members)


def test_two_stages_runs_to_done(run_ensemble, fake, members):
    """Test that a two-stage ensemble runs both stages of every member, each
    in its own configured case, with each member's first stage starting its second
    """
    members = members[:2]  # two stages double the jobs; two members are enough
    root = run_ensemble(member_stages=_TWO_STAGES, members=members)
    ensemble = Ensemble.load(root)
    status = ensemble.status()
    assert len(status) == len(members) * 2, fake.logs()

    assert (status["status"] == "done").all(), fake.logs()
    assert list(zip(status["member"], status["stage"])) == [
        (m, s) for m in members for s in ("spinup_ad", "transient")
    ]
    assert status["case_root"].nunique() == len(members) * 2
    for row in status.itertuples():
        _assert_case_configured(
            Path(str(row.case_root)), ensemble.param_files[str(row.member)], fake
        )

    # transient isn't a PBS dependency of spinup_ad: spinup_ad's advance job
    # submits it, so it's numbered after that member's advance job
    number = {j["name"]: int(j["id"].split(".")[0]) for j in fake.jobs()}
    for member in members:
        assert (
            number[f"my_ens-1_{member}_transient"]
            > number[f"my_ens-1_{member}_spinup_ad_advance"]
        )
    _assert_jobs_finished_cleanly(fake)


def test_transient_gets_its_own_spinup_case(run_ensemble, fake, members, tmp_path):
    """Test that each member's transient script is handed that member's own
    spinup_ad case as its previous case
    """
    members = members[:2]
    record = tmp_path / "args.txt"
    script = tmp_path / "transient.sh"
    # the job sources the script under bash, so it needs no shebang
    script.write_text(f'echo "$1|$2|$3" >> {record}\n', encoding="utf-8")
    stages = [_TWO_STAGES[0], {**_TWO_STAGES[1], "script": str(script)}]

    root = run_ensemble(member_stages=stages, members=members)
    status = Ensemble.load(root).status().set_index(["member", "stage"])

    rows = [l.split("|") for l in record.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(members), fake.logs()
    for member in members:
        transient = status.loc[(member, "transient"), "case_root"]
        spinup = status.loc[(member, "spinup_ad"), "case_root"]
        assert [transient, Path(spinup).name, spinup] in rows


def test_status_cli_after_run(run_ensemble, meristem, members):
    """Test that status after a clean run checks the queue, finds nothing lost,
    and lists every member as done
    """
    root = run_ensemble()
    result = meristem(
        "ensemble", "status", "--root", str(root), "--check-queue", "--print-members"
    )
    assert result.returncode == 0, result.stdout + result.stderr

    lines = result.stdout.splitlines()
    assert "Failed" not in result.stdout
    assert "Lost" not in result.stdout
    for member in members:
        line = next(l for l in lines if l.startswith(member))
        assert line.split()[:3] == [member, "spinup_ad", "done"]


def test_manifest_after_run(run_ensemble, meristem, members):
    """Test that the manifest has a row per member with its key values, its
    done stage, and the stage case's archive directory
    """
    root = run_ensemble()
    result = meristem("ensemble", "manifest", "--root", str(root))
    assert result.returncode == 0, result.stderr
    out = root / "manifest.csv"
    assert f"Wrote manifest: {out}" in result.stdout

    manifest = pd.read_csv(out, dtype=str)
    assert list(manifest["member"]) == members
    assert list(manifest["fates_leaf_slatop"]) == ["0.0", "0.1", "0.2"]
    assert (manifest["spinup_ad_status"] == "done").all()
    assert manifest["spinup_ad_job_id"].str.endswith(".fakepbs").all()
    for row in manifest.itertuples():
        case_root = Path(str(row.spinup_ad_case_root))
        assert row.spinup_ad_archive_dir == str(case_root / "archive")


def test_case_run_fails_then_retry_failed(run_ensemble, fake, meristem, tmp_path):
    """Test that one member's failed case.run is reported as failed while the
    others finish, and that retry --failed reruns it to done
    """
    failing = "ens_0001"
    marker = tmp_path / "failed_once"
    script = tmp_path / "inject.sh"
    # fail case.run for one member on its first attempt only, by writing the
    # fake CIME fault file into that member's freshly prepared case
    script.write_text(
        f'if [[ "$1" == *_{failing}_* && ! -e {marker} ]]; then\n'
        f"  touch {marker}\n"
        '  echo \'{"case.run": "fail"}\' > "$1/fake_cime.json"\n'
        "else\n"
        "  echo '{}' > \"$1/fake_cime.json\"\n"
        "fi\n",
        encoding="utf-8",
    )
    stage = {"name": "spinup_ad", "queue": "develop", "script": str(script)}
    root = run_ensemble(member_stages=[stage])

    status = Ensemble.load(root).status().set_index("member")["status"]
    assert status[failing] == "failed", fake.logs()
    assert (status.drop(failing) == "done").all(), fake.logs()

    result = meristem("ensemble", "status", "--root", str(root))
    assert result.returncode == 0, result.stderr
    assert f"Failed (1): ['{failing}:spinup_ad']" in result.stdout
    assert f"meristem ensemble retry --root {root} --failed" in result.stdout

    result = meristem("ensemble", "retry", "--root", str(root), "--failed")
    assert result.returncode == 0, result.stderr
    fake.run_until_idle()

    status = Ensemble.load(root).status().set_index("member")
    assert (status["status"] == "done").all(), fake.logs()
    assert status.loc[failing, "attempts"] == 2
    assert (status.drop(failing)["attempts"] == 1).all()


def test_retry_skip_script_still_sets_paramfile(run_ensemble, fake, meristem, tmp_path):
    """Test that retrying a member with --skip-script still runs finalize, so
    its case gets its own parameter file rather than keeping the base case's.
    Here the member's script fails before finalize ever ran, then is skipped
    """
    failing = "ens_0001"
    script = tmp_path / "fails.sh"
    # fails for one member whenever it runs, so the retry only succeeds if the
    # script really is skipped
    script.write_text(f'if [[ "$1" == *_{failing}_* ]]; then exit 3; fi\n')
    stage = {"name": "spinup_ad", "queue": "develop", "script": str(script)}
    root = run_ensemble(member_stages=[stage])

    ensemble = Ensemble.load(root)
    failing_case = Path(ensemble.status().set_index("member").loc[failing, "case_root"])
    assert ensemble.status().set_index("member").loc[failing, "status"] == "failed"
    # finalize never ran, so the clone still has only the base case's settings
    assert _paramfile_lines(failing_case) == [], fake.logs()

    result = meristem(
        "ensemble", "retry", "--root", str(root), "--failed", "--skip-script"
    )
    assert result.returncode == 0, result.stderr
    fake.run_until_idle()

    status = Ensemble.load(root).status()
    assert (status["status"] == "done").all(), fake.logs()
    for row in status.itertuples():
        _assert_case_configured(
            Path(str(row.case_root)), ensemble.param_files[str(row.member)], fake
        )


def test_killed_setup_job_is_lost_then_retry_lost(make_ensemble, fake, meristem):
    """Test that a member whose setup job was killed is reported as lost (exit
    code 3) while the others finish, and that retry --lost reruns it to done
    """
    lost = "ens_0001"
    root = make_ensemble()
    assert meristem("ensemble", "submit", "--root", str(root)).returncode == 0
    job = next(j for j in fake.jobs() if j["name"] == f"my_ens-1_{lost}_spinup_ad")
    fake.kill(job["id"])
    fake.run_until_idle()

    result = meristem("ensemble", "status", "--root", str(root), "--check-queue")
    assert result.returncode == EXIT_LOST, result.stdout + result.stderr
    assert f"Lost (1): ['{lost}:spinup_ad']" in result.stdout

    result = meristem("ensemble", "retry", "--root", str(root), "--lost")
    assert result.returncode == 0, result.stderr
    fake.run_until_idle()

    result = meristem("ensemble", "status", "--root", str(root), "--check-queue")
    assert result.returncode == 0, result.stdout + result.stderr
    status = Ensemble.load(root).status()
    assert (status["status"] == "done").all(), fake.logs()
