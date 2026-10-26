"""Tests for fake PBS"""

from __future__ import annotations

import subprocess
import os

import pytest
import json

from meristem.run.scheduler import _parse_qstat_states
from .pbs import qsub, qstat, qdel, _load, _save
from .fake_pbs import FakePbs, _decide


def test_qsub_parses_depend(tmp_path, monkeypatch):
    """Test qsub"""
    monkeypatch.setenv("FAKE_PBS_SPOOL", str(tmp_path))

    script = tmp_path / "job.sh"
    script.write_text(
        "#!/bin/bash\n#PBS -N mystage\n#PBS -W depend=afterok:5.fakepbs\necho hi\n"
    )

    qsub([str(script)])

    jobs = _load()
    assert jobs[0]["name"] == "mystage"
    assert jobs[0]["depend"] == ["afterok", "5.fakepbs"]

    qsub(["-W depend=afternotok:6.fakepbs", str(script)])
    jobs = _load()
    assert jobs[1]["depend"] == ["afternotok", "6.fakepbs"]


def test_qstat_output_parses_with_real_parser(tmp_path, monkeypatch, capsys):
    """Test qstat with good input"""
    monkeypatch.setenv("FAKE_PBS_SPOOL", str(tmp_path))
    _save([{"id": "1.fakepbs", "state": "Q"}])

    code = qstat(["-x", "-f", "1.fakepbs"])

    assert code == 0
    assert _parse_qstat_states(capsys.readouterr().out) == {"1": "Q"}


def test_qstat_output_parses_return_153_with_missing(tmp_path, monkeypatch, capsys):
    """Test qstat with bad input"""
    monkeypatch.setenv("FAKE_PBS_SPOOL", str(tmp_path))
    _save(
        [
            {"id": "1.fakepbs", "state": "Q"},
            {"id": "3.fakepbs", "state": "R"},
        ]
    )

    code = qstat(["-x", "-f", "1", "99", "3"])

    assert code == 153

    parsed = _parse_qstat_states(capsys.readouterr().out)
    assert parsed == {"1": "Q", "3": "R"}


def test_qdel_marks_job_finished_and_deleted(tmp_path, monkeypatch):
    """Test qdel with good input"""
    monkeypatch.setenv("FAKE_PBS_SPOOL", str(tmp_path))
    _save([{"id": "1.fakepbs", "state": "Q", "deleted": False}])

    assert qdel(["1.fakepbs"]) == 0

    job = _load()[0]
    assert job["state"] == "F"
    assert job["deleted"] is True


def test_qdel_on_missing_id_errors(tmp_path, monkeypatch):
    """Test qdel with bad input"""
    monkeypatch.setenv("FAKE_PBS_SPOOL", str(tmp_path))
    _save([{"id": "1.fakepbs", "state": "Q", "deleted": False}])

    assert qdel(["5.fakepbs"]) == 153

    job = _load()[0]
    assert job["state"] == "Q"
    assert job["deleted"] is False


def test_wrapper_qsub_runs_via_path(tmp_path):
    """Test that the qsub runs via path correctly"""
    fake = FakePbs(tmp_path / "pbs")

    script = tmp_path / "job.sh"
    script.write_text("#!/bin/bash\n#PBS -N mystage\necho hi\n")

    result = subprocess.run(
        ["qsub", str(script)],
        env=fake.env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "1.fakepbs"


def test_wrapper_qsub_fail_works_correctly(tmp_path):
    """Test that the fakepbs qsub_fail works"""
    fake = FakePbs(tmp_path / "pbs")

    script = tmp_path / "advance.sh"
    script.write_text("#!/bin/bash\n#PBS -N mystage\necho hi\n")

    script_pass = tmp_path / "another.sh"
    script_pass.write_text("#!/bin/bash\n#PBS -N mystage\necho hi\n")

    fake.fail_qsub("advance")

    result = subprocess.run(
        ["qsub", str(script)],
        env=fake.env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "injected" in result.stderr
    jobs = fake.jobs()
    assert jobs == []

    result = subprocess.run(
        ["qsub", str(script_pass)],
        env=fake.env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert len(fake.jobs()) == 1


def test_wrapper_break_qstat_works_correctly(tmp_path):
    """Test that the fakepbs break_qstat works"""
    fake = FakePbs(tmp_path / "pbs")
    fake.break_qstat()

    result = subprocess.run(
        ["qstat", "-x", "-f", "1"],
        env=fake.env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "injected" in result.stderr


def test_wrapper_qstat_passes_with_failed_script(tmp_path):
    """Test that the wrapper qstat passes with a failed qsub"""
    fake = FakePbs(tmp_path / "pbs")
    fake.fail_qsub("advance")

    script = tmp_path / "advance.sh"
    script.write_text("#!/bin/bash\n#PBS -N mystage\necho hi\n")

    script_pass = tmp_path / "another.sh"
    script_pass.write_text("#!/bin/bash\n#PBS -N mystage\necho hi\n")

    subprocess.run(
        ["qsub", str(script)],
        env=fake.env,
        capture_output=True,
        text=True,
        check=False,
    )
    subprocess.run(
        ["qsub", str(script_pass)],
        env=fake.env,
        capture_output=True,
        text=True,
        check=False,
    )

    result = subprocess.run(
        ["qstat", "-x", "-f", "1"],
        env=fake.env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0


@pytest.mark.parametrize(
    "depend, dep, expected",
    [
        (None, None, "run"),
        (
            ["afterok", "1.fakepbs"],
            {"state": "Q", "exit": None, "deleted": False},
            "wait",
        ),
        (
            ["afterok", "1.fakepbs"],
            {"state": "F", "exit": None, "deleted": True},
            "delete",
        ),
        (["afterok", "1.fakepbs"], {"state": "F", "exit": 0, "deleted": False}, "run"),
        (
            ["afterok", "1.fakepbs"],
            {"state": "F", "exit": 1, "deleted": False},
            "delete",
        ),
        (
            ["afternotok", "1.fakepbs"],
            {"state": "F", "exit": 1, "deleted": False},
            "run",
        ),
        (
            ["afternotok", "1.fakepbs"],
            {"state": "F", "exit": 0, "deleted": False},
            "delete",
        ),
    ],
)
def test_decide(depend, dep, expected):
    """Test the _decide function"""
    assert _decide({"id": "2.fakepbs", "depend": depend}, dep) == expected


def test_run_job_runs_script(tmp_path):
    """Test that _run_job runs a script correctly"""
    fake = FakePbs(tmp_path / "pbs")

    script = tmp_path / "my_script.sh"
    script.write_text("#!/bin/bash\n#PBS -N mystage\necho hello; exit 3\n")

    subprocess.run(["qsub", str(script)], env=fake.env, check=True)
    fake._run_job(fake.jobs()[0])

    job = fake.jobs()[0]
    assert job["exit"] == 3
    assert job["state"] == "F"
    log = fake.spool / "1.log"
    assert "hello" in log.read_text(encoding="utf-8")


def test_run_job_runs_qsub(tmp_path):
    """Test that _run_job runs a script correctly when it calls qsub"""
    fake = FakePbs(tmp_path / "pbs")

    script = tmp_path / "my_script.sh"
    child = tmp_path / "child.sh"
    child.write_text("#!/bin/bash\necho child\n")
    script.write_text(f"#!/bin/bash\nqsub {child}\n")

    subprocess.run(["qsub", str(script)], env=fake.env, check=True)
    fake._run_job(fake.jobs()[0])

    assert len(fake.jobs()) == 2
    job = fake.jobs()[0]
    assert job["exit"] == 0
    assert job["state"] == "F"

    child_job = fake.jobs()[1]
    assert child_job["script"] == str(child)
    assert child_job["state"] == "Q"


def test_run_until_idle_runs_two_jobs(tmp_path):
    """Test that run_until_idle runs two jobs"""
    fake = FakePbs(tmp_path / "pbs")

    script = tmp_path / "my_script.sh"
    script.write_text("#!/bin/bash\n#PBS -N mystage\necho hello; exit 0\n")

    script_dep = tmp_path / "my_second_script.sh"
    script_dep.write_text(
        "#!/bin/bash\n#PBS -N mystage\n#PBS -W depend=afterok:1.fakepbs\necho hello; exit 0\n"
    )

    subprocess.run(["qsub", str(script)], env=fake.env, check=True)
    subprocess.run(["qsub", str(script_dep)], env=fake.env, check=True)

    fake.run_until_idle()
    jobs = fake.jobs()
    assert len(jobs) == 2
    assert jobs[0]["exit"] == 0
    assert jobs[1]["exit"] == 0
    assert jobs[0]["state"] == "F"
    assert jobs[1]["state"] == "F"


def test_run_until_idle_runs_one_job(tmp_path):
    """Test that run_until_idle runs only one job if the first exits 1 with a depend"""
    fake = FakePbs(tmp_path / "pbs")

    script = tmp_path / "my_script.sh"
    script.write_text("#!/bin/bash\n#PBS -N mystage\necho hello; exit 1\n")

    script_dep = tmp_path / "my_second_script.sh"
    script_dep.write_text(
        "#!/bin/bash\n#PBS -N mystage\n#PBS -W depend=afterok:1.fakepbs\necho hello; exit 0\n"
    )

    subprocess.run(["qsub", str(script)], env=fake.env, check=True)
    subprocess.run(["qsub", str(script_dep)], env=fake.env, check=True)

    fake.run_until_idle()
    jobs = fake.jobs()
    assert len(jobs) == 2
    assert jobs[0]["exit"] == 1
    assert jobs[1]["exit"] is None
    assert jobs[1]["deleted"] is True
    assert jobs[0]["state"] == "F"


def test_run_until_idle_runs_dependencies(tmp_path):
    """Test that run_until_idle accurately parses dependencies"""
    fake = FakePbs(tmp_path / "pbs")

    script = tmp_path / "my_script.sh"
    script.write_text("#!/bin/bash\n#PBS -N mystage\necho hello; exit 1\n")

    script_ok = tmp_path / "my_script_ok.sh"
    script_ok.write_text(
        "#!/bin/bash\n#PBS -N mystage\n#PBS -W depend=afterok:1.fakepbs\necho hello; exit 0\n"
    )

    script_notok = tmp_path / "my_script_notok.sh"
    script_notok.write_text(
        "#!/bin/bash\n#PBS -N mystage\n#PBS -W depend=afternotok:1.fakepbs\necho hello; exit 0\n"
    )

    subprocess.run(["qsub", str(script)], env=fake.env, check=True)
    subprocess.run(["qsub", str(script_ok)], env=fake.env, check=True)
    subprocess.run(["qsub", str(script_notok)], env=fake.env, check=True)

    fake.run_until_idle()
    jobs = fake.jobs()
    assert len(jobs) == 3
    assert jobs[0]["exit"] == 1
    assert jobs[1]["exit"] is None
    assert jobs[2]["exit"] == 0

    assert jobs[0]["state"] == "F"
    assert jobs[1]["deleted"] is True
    assert jobs[2]["state"] == "F"


def test_run_until_idle_cascade(tmp_path):
    """Test that run_until_idle accurately parses a dependency cascade"""
    fake = FakePbs(tmp_path / "pbs")

    script = tmp_path / "my_script.sh"
    script.write_text("#!/bin/bash\n#PBS -N mystage\necho hello; exit 1\n")

    script_ok1 = tmp_path / "my_script_ok1.sh"
    script_ok1.write_text(
        "#!/bin/bash\n#PBS -N mystage\n#PBS -W depend=afterok:1.fakepbs\necho hello; exit 0\n"
    )

    script_ok2 = tmp_path / "my_script_ok2.sh"
    script_ok2.write_text(
        "#!/bin/bash\n#PBS -N mystage\n#PBS -W depend=afterok:2.fakepbs\necho hello; exit 0\n"
    )

    subprocess.run(["qsub", str(script)], env=fake.env, check=True)
    subprocess.run(["qsub", str(script_ok1)], env=fake.env, check=True)
    subprocess.run(["qsub", str(script_ok2)], env=fake.env, check=True)

    fake.run_until_idle()
    jobs = fake.jobs()
    assert len(jobs) == 3
    assert jobs[0]["exit"] == 1
    assert jobs[1]["exit"] is None
    assert jobs[2]["exit"] is None

    assert jobs[0]["state"] == "F"
    assert jobs[1]["deleted"] is True
    assert jobs[2]["deleted"] is True


def test_qsub_catch_bad_depend(tmp_path):
    """Test that qsub catches a bad depend string in a script"""
    fake = FakePbs(tmp_path / "pbs")

    script_bad = tmp_path / "my_script_bad.sh"
    script_bad.write_text(
        "#!/bin/bash\n#PBS -N mystage\n#PBS -W depend=afternok:1.fakepbs\necho hello; exit 0\n"
    )

    result = subprocess.run(
        ["qsub", str(script_bad)],
        env=fake.env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "dependency" in result.stderr


def test_qsub_catch_bad_depend_in_call(tmp_path):
    """Test that qsub catches a bad depend string in the call"""
    fake = FakePbs(tmp_path / "pbs")

    script = tmp_path / "my_script_bad.sh"
    script.write_text("#!/bin/bash\n#PBS -N mystage\necho hello; exit 0\n")

    result = subprocess.run(
        ["qsub", "-W depend=afternok:1.fakebs", str(script)],
        env=fake.env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "dependency" in result.stderr


def test_logs(tmp_path):
    """Test the logs method"""
    fake = FakePbs(tmp_path / "pbs")

    script1 = tmp_path / "my_script1.sh"
    script1.write_text("#!/bin/bash\n#PBS -N mystage1\necho hello; exit 0\n")

    script2 = tmp_path / "my_script2.sh"
    script2.write_text("#!/bin/bash\n#PBS -N mystage2\necho hello again; exit 0\n")

    subprocess.run(["qsub", str(script1)], env=fake.env, check=True)
    subprocess.run(["qsub", str(script2)], env=fake.env, check=True)

    fake._run_job(fake.jobs()[0])
    out = fake.logs()
    assert "1.fakepbs" in out and "2.fakepbs" in out
    assert "hello" in out
    assert "(no log)" in out
    assert "hello again" not in out


def test_kill(tmp_path):
    """Test the kill method"""
    fake = FakePbs(tmp_path / "pbs")

    script1 = tmp_path / "my_script1.sh"
    script1.write_text("#!/bin/bash\n#PBS -N mystage1\necho hello; exit 0\n")

    script2 = tmp_path / "my_script2.sh"
    script2.write_text(
        "#!/bin/bash\n#PBS -N mystage2\n#PBS -W depend=afterok:1.fakepbs\necho hello; exit 0\n"
    )

    script3 = tmp_path / "my_script3.sh"
    script3.write_text(
        "#!/bin/bash\n#PBS -N mystage3\n#PBS -W depend=afternotok:1.fakepbs\necho hello; exit 0\n"
    )

    subprocess.run(["qsub", str(script1)], env=fake.env, check=True)
    subprocess.run(["qsub", str(script2)], env=fake.env, check=True)
    subprocess.run(["qsub", str(script3)], env=fake.env, check=True)

    fake.kill("1.fakepbs")
    fake.run_until_idle()

    jobs = fake.jobs()
    assert jobs[0]["exit"] == 271
    assert not fake._log_path(jobs[0]).is_file()

    assert jobs[1]["deleted"] is True
    assert not fake._log_path(jobs[1]).is_file()

    assert jobs[2]["state"] == "F"
    assert jobs[2]["exit"] == 0


def test_fake_create_newcase(tmp_path):
    """Test that the fake_create_newcase runs via path correctly"""
    fake = FakePbs(tmp_path / "pbs")

    subprocess.run(
        ["fake_create_newcase", str(tmp_path / "mycase")],
        env=fake.env,
        check=True,
    )

    case_dir = tmp_path / "mycase"
    assert case_dir.is_dir()

    files = [f.name for f in case_dir.iterdir() if f.is_file()]
    expected_files = [
        "case.submit",
        "xmlquery",
        "xmlchange",
        "case.setup",
        "preview_namelists",
        "env_case.xml",
        "user_nl_clm",
    ]
    for file_name in expected_files:
        assert file_name in files

    for name in (
        "case.submit",
        "xmlquery",
        "xmlchange",
        "case.setup",
        "preview_namelists",
    ):
        assert os.access(case_dir / name, os.X_OK)

    result = subprocess.run(
        ["./xmlquery", "--value", "CASE"],
        cwd=case_dir,
        env=fake.env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.stdout.strip() == "mycase"

    result = subprocess.run(
        ["./xmlquery", "--value", "JOB_IDS"],
        cwd=case_dir,
        env=fake.env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1

    result = subprocess.run(
        [
            "./case.setup",
        ],
        cwd=case_dir,
        env=fake.env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0


def test_case_submit(tmp_path):
    """Test that the case_submit runs via path correctly"""
    fake = FakePbs(tmp_path / "pbs")
    case_dir = tmp_path / "mycase"

    subprocess.run(
        ["fake_create_newcase", str(case_dir)],
        env=fake.env,
        check=True,
    )

    subprocess.run(
        ["./case.submit", "--resubmit-immediate"],
        cwd=case_dir,
        env=fake.env,
        check=True,
    )

    result = subprocess.run(
        ["./xmlquery", "--value", "JOB_IDS"],
        cwd=case_dir,
        env=fake.env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "case.run:1.fakepbs, case.st_archive:2.fakepbs"

    jobs = fake.jobs()
    assert len(jobs) == 2
    assert jobs[1]["depend"] == ["afterok", "1.fakepbs"]

    fake.run_until_idle()
    jobs = fake.jobs()
    assert jobs[0]['state'] == 'F'
    assert jobs[1]['state'] == 'F'
    assert jobs[0]['exit'] == 0
    assert jobs[1]['exit'] == 0

    log_file = case_dir / 'runs.log'
    log_content = log_file.read_text(encoding="utf-8", errors="ignore")
    assert log_content.splitlines() == ["case.run mycase", "case.st_archive mycase"]

    subprocess.run(
        ["./case.submit", "--resubmit-immediate"],
        cwd=case_dir,
        env=fake.env,
        check=True,
    )

    result = subprocess.run(
        ["./xmlquery", "--value", "JOB_IDS"],
        cwd=case_dir,
        env=fake.env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "case.run:3.fakepbs, case.st_archive:4.fakepbs"
    
    
def test_case_submit_fail(tmp_path):
    """Test that the injected failure into case_submit works"""
    fake = FakePbs(tmp_path / "pbs")
    case_dir = tmp_path / "mycase"
    
    fake.env["FAKE_CIME"] = json.dumps({"case.submit": "fail"})

    subprocess.run(
        ["fake_create_newcase", str(case_dir)],
        env=fake.env,
        check=True,
    )
    
    result = subprocess.run(
        ["./case.submit", "--resubmit-immediate"],
        cwd=case_dir,
        env=fake.env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert 'injected' in result.stderr.strip()
    assert fake.jobs() == []


def test_case_run_fail(tmp_path):
    """Test that the injected failure into case.run works"""
    fake = FakePbs(tmp_path / "pbs")
    case_dir = tmp_path / "mycase"
    
    fake.env["FAKE_CIME"] = json.dumps({"case.run": "fail"})

    subprocess.run(
        ["fake_create_newcase", str(case_dir)],
        env=fake.env,
        check=True,
    )
    
    result = subprocess.run(
        ["./case.submit", "--resubmit-immediate"],
        cwd=case_dir,
        env=fake.env,
        check=False,
    )
    assert result.returncode == 0
    assert len(fake.jobs()) == 2

    fake.run_until_idle()
    jobs = fake.jobs()
    assert jobs[0]['state'] == 'F'
    assert jobs[1]['state'] == 'F'
    assert jobs[0]['exit'] == 1
    assert jobs[1]['exit'] is None
    assert jobs[1]['deleted'] is True

    log_file = case_dir / 'runs.log'
    log_content = log_file.read_text(encoding="utf-8", errors="ignore")
    assert log_content.splitlines() == ["case.run mycase"]


def test_case_no_archive(tmp_path):
    """Test that the injected no_archive works"""
    fake = FakePbs(tmp_path / "pbs")
    case_dir = tmp_path / "mycase"
    
    fake.env["FAKE_CIME"] = json.dumps({"no_archive": True})

    subprocess.run(
        ["fake_create_newcase", str(case_dir)],
        env=fake.env,
        check=True,
    )
    
    result = subprocess.run(
        ["./case.submit", "--resubmit-immediate"],
        cwd=case_dir,
        env=fake.env,
        check=False,
    )
    assert result.returncode == 0
    assert len(fake.jobs()) == 1

    fake.run_until_idle()
    jobs = fake.jobs()
    assert jobs[0]['state'] == 'F'
    assert jobs[0]['exit'] == 0

    result = subprocess.run(
        ["./xmlquery", "--value", "JOB_IDS"],
        cwd=case_dir,
        env=fake.env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.stdout.strip() == "case.run:1.fakepbs"



def test_case_archive_fail(tmp_path):
    """Test that blank JOB_IDS: submit succeeds, xmlquery returns an empty string"""
    fake = FakePbs(tmp_path / "pbs")
    case_dir = tmp_path / "mycase"
    
    fake.env["FAKE_CIME"] = json.dumps({"st_archive": "fail"})

    subprocess.run(
        ["fake_create_newcase", str(case_dir)],
        env=fake.env,
        check=True,
    )
    
    result = subprocess.run(
        ["./case.submit", "--resubmit-immediate"],
        cwd=case_dir,
        env=fake.env,
        check=False,
    )
    assert result.returncode == 0
    assert len(fake.jobs()) == 2

    fake.run_until_idle()
    jobs = fake.jobs()
    assert jobs[0]['state'] == 'F'
    assert jobs[1]['state'] == 'F'
    assert jobs[0]['exit'] == 0
    assert jobs[1]['exit'] == 1

    log_file = case_dir / 'runs.log'
    log_content = log_file.read_text(encoding="utf-8", errors="ignore")
    assert log_content.splitlines() == ["case.run mycase", "case.st_archive mycase"]
    
    
def test_xmlquery_blank_job_ids(tmp_path):
    """Test that the job_ids blank inject works"""
    fake = FakePbs(tmp_path / "pbs")
    case_dir = tmp_path / "mycase"

    fake.env["FAKE_CIME"] = json.dumps({"job_ids": ""})

    subprocess.run(
        ["fake_create_newcase", str(case_dir)],
        env=fake.env,
        check=True,
    )

    subprocess.run(
        ["./case.submit", "--resubmit-immediate"],
        cwd=case_dir,
        env=fake.env,
        check=True,
    )

    result = subprocess.run(
        ["./xmlquery", "--value", "JOB_IDS"],
        cwd=case_dir,
        env=fake.env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""
    assert len(fake.jobs()) == 2
