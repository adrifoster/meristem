"""Tests for fake CIME methods"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import json

from . import cime
from .cime import (
    create_newcase,
    create_clone,
    xmlquery,
    xmlchange,
    case_submit,
    case_setup,
    CASE_DICT_FILE,
    preview_namelists,
    _qsub,
)


@pytest.fixture
def fake_pbs_env(fake, monkeypatch):
    """Point PATH and FAKE_PBS_SPOOL at the fake PBS, so qsub is the fake one"""
    monkeypatch.setenv("PATH", fake.env["PATH"])
    monkeypatch.setenv("FAKE_PBS_SPOOL", fake.env["FAKE_PBS_SPOOL"])
    return fake


@pytest.fixture
def case_root(tmp_path, monkeypatch):
    """Case root fixture"""
    monkeypatch.delenv("FAKE_CIME", raising=False)
    root = tmp_path / "case"
    create_newcase(root)
    return root


@pytest.fixture
def clone_source(tmp_path, monkeypatch):
    """Source case fixture"""
    monkeypatch.delenv("FAKE_CIME", raising=False)
    src = tmp_path / "src"
    create_newcase(src)

    (src / "user_nl_clm").write_text("hist_nhtfrq = -24\n", encoding="utf-8")
    case_dict = json.loads((src / CASE_DICT_FILE).read_text())
    case_dict["BUILD_COMPLETE"] = True
    case_dict["CIMEROOT"] = "/path/to/cime"
    (src / CASE_DICT_FILE).write_text(json.dumps(case_dict), encoding="utf-8")
    return src


@pytest.fixture
def make_case(tmp_path, monkeypatch, fake_pbs_env):  # pylint: disable=unused-argument
    """Make case fixture; case_submit on the case submits to the fake PBS"""

    def _make(faults=None):
        if faults is None:
            monkeypatch.delenv("FAKE_CIME", raising=False)
        else:
            monkeypatch.setenv("FAKE_CIME", json.dumps(faults))
        root = tmp_path / "case"
        create_newcase(root)
        return root

    return _make


def job_ids(root):
    """Get JOB IDs from root"""
    value = json.loads((root / CASE_DICT_FILE).read_text())["JOB_IDS"]
    return dict(item.split(":", 1) for item in value.split(", "))


def _run_main(monkeypatch, *argv) -> int:
    """Run cime.main() with argv as the command line and return its exit code"""
    monkeypatch.setattr(sys, "argv", ["cime.py", *argv])
    with pytest.raises(SystemExit) as exc:
        cime.main()
    return exc.value.code


def test_create_newcase_makes_case_files(tmp_path, monkeypatch):
    """Test create_newcase basic behavior"""
    monkeypatch.delenv("FAKE_CIME", raising=False)
    case_root = tmp_path / "case"

    assert create_newcase(case_root) == 0
    assert (case_root / "env_case.xml").read_text() == "env_case placeholder\n"
    assert (case_root / "user_nl_clm").read_text() == "! user_nl_clm\n"

    case_dict = json.loads((case_root / CASE_DICT_FILE).read_text())
    assert case_dict["CASE"] == "case"
    assert case_dict["BUILD_COMPLETE"] is False
    assert case_dict["DOUT_S_ROOT"] == str(case_root / "archive")


def test_create_newcase_writes_faults_from_env(tmp_path, monkeypatch):
    """Test that create_newcase bakes FAKE_CIME into the case's faults file"""
    monkeypatch.setenv("FAKE_CIME", json.dumps({"st_archive": "fail"}))
    case_root = tmp_path / "case"

    create_newcase(case_root)
    faults = json.loads((case_root / cime.FAULTS_DICT_FILE).read_text())
    assert faults == {"st_archive": "fail"}


def test_create_newcase_refuses_existing_case(tmp_path, monkeypatch, capsys):
    """Test that, like CIME, create_newcase refuses a case_root that already
    exists, and leaves it as it was
    """
    monkeypatch.delenv("FAKE_CIME", raising=False)
    case_root = tmp_path / "case"
    case_root.mkdir()
    (case_root / "keep.txt").write_text("mine\n")

    assert create_newcase(case_root) == 1
    assert "already exists" in capsys.readouterr().err
    assert sorted(p.name for p in case_root.iterdir()) == ["keep.txt"]


WRAPPER_FILES = (
    "case.submit",
    "xmlquery",
    "xmlchange",
    "case.setup",
    "preview_namelists",
)


@pytest.mark.parametrize("name", WRAPPER_FILES)
def test_create_newcase_writes_executable_wrappers(tmp_path, monkeypatch, name):
    """Test that create_newcase correctly writes the executables"""
    monkeypatch.delenv("FAKE_CIME", raising=False)
    case_root = tmp_path / "case"

    create_newcase(case_root)

    wrapper = case_root / name
    assert wrapper.is_file()
    assert os.access(wrapper, os.X_OK)

    lines = wrapper.read_text().splitlines()
    assert lines[0] == "#!/bin/sh"

    expected_exec = (
        f'exec {sys.executable} {Path(cime.__file__).resolve()} {name} "$CASE_DIR" "$@"'
    )
    assert lines[-1] == expected_exec


def test_create_clone_copies_from_source(tmp_path, clone_source):
    """Test that create_clone copies from the source"""
    new = tmp_path / "new"

    rc = create_clone(
        ["--case", str(new), "--clone", str(clone_source), "--project", "P123"]
    )

    assert rc == 0
    assert (new / "user_nl_clm").read_text() == "hist_nhtfrq = -24\n"

    case_dict = json.loads((new / CASE_DICT_FILE).read_text())
    assert case_dict["BUILD_COMPLETE"] is True
    assert case_dict["CIMEROOT"] == "/path/to/cime"
    assert case_dict["PROJECT"] == "P123"
    assert case_dict["CASE"] == "new"


def test_create_clone_injected_failure(tmp_path, clone_source, monkeypatch, capsys):
    """Test the injected failure in create_clone"""
    monkeypatch.setenv("FAKE_CIME", json.dumps({"create_clone": "fail"}))
    new = tmp_path / "new"

    rc = create_clone(["--case", str(new), "--clone", str(clone_source)])

    assert rc == 1
    assert "injected failure" in capsys.readouterr().err
    assert not new.exists()


def test_create_clone_refuses_existing_case(tmp_path, clone_source, capsys):
    """Test that create_clone fails on an existing case"""
    existing = tmp_path / "existing"
    existing.mkdir()

    rc = create_clone(["--case", str(existing), "--clone", str(clone_source)])

    assert rc == 1
    assert "already exists" in capsys.readouterr().err


def test_create_clone_project_defaults_to_none(tmp_path, clone_source):
    """Test that create_clone project defaults to None"""
    new = tmp_path / "new"
    create_clone(["--case", str(new), "--clone", str(clone_source)])

    assert json.loads((new / CASE_DICT_FILE).read_text())["PROJECT"] is None


@pytest.mark.parametrize("missing", ["--case", "--clone"])
def test_create_clone_requires_case_and_clone(tmp_path, clone_source, missing):
    """Test that create_clone fails on missing args"""
    args = {"--case": str(tmp_path / "new"), "--clone": str(clone_source)}
    del args[missing]
    argv = [x for kv in args.items() for x in kv]

    with pytest.raises(SystemExit):
        create_clone(argv)


@pytest.mark.parametrize("args", [["CASE"], ["--value", "CASE"]])
def test_xmlquery_returns_value(case_root, capsys, args):
    """xmlquery returns value for good arguments"""
    rc = xmlquery(case_root, args)

    assert rc == 0
    assert capsys.readouterr().out.strip() == "case"


def test_xmlquery_prints_false_value(case_root, capsys):
    """A falsey value (None) returns False"""
    rc = xmlquery(case_root, ["BUILD_COMPLETE"])

    assert rc == 0
    assert capsys.readouterr().out.strip() == "False"


def test_xmlchange_passes(case_root):
    """Make sure xmlchange returns 0"""
    rc = xmlchange(case_root, ["args"])
    assert rc == 0


def test_case_setup_passes(case_root):
    """Make sure case_setup returns 0"""
    rc = case_setup(case_root)
    assert rc == 0


def test_xmlquery_unknown_key(case_root, capsys):
    """Test that xmlquery fails with an unknown key"""
    rc = xmlquery(case_root, ["NOT_A_KEY"])

    assert rc == 1
    assert "Unknown xmlquery variable NOT_A_KEY" in capsys.readouterr().out


def test_xmlquery_missing_case_json(case_root, capsys):
    """Test that xmlquery with a missing case json fails"""
    (case_root / CASE_DICT_FILE).unlink()

    rc = xmlquery(case_root, ["CASE"])

    assert rc == 1
    assert "Run create_newcase first" in capsys.readouterr().out


def test_xmlquery_key_with_none_value(case_root, capsys):
    """Test that xmlquery fail with a None in a case_dict"""
    # JOB_IDS is None in the default case dict
    rc = xmlquery(case_root, ["JOB_IDS"])

    assert rc == 1
    assert "not in" in capsys.readouterr().out


def test_xmlquery_job_ids_fault_prints_empty(tmp_path, monkeypatch, capsys):
    """Test that xmlquery with an injected JOB IDs fault returns blank"""
    monkeypatch.setenv("FAKE_CIME", json.dumps({"job_ids": ""}))
    root = tmp_path / "case"
    create_newcase(root)  # bakes FAKE_CIME into the faults file

    rc = xmlquery(root, ["JOB_IDS"])

    assert rc == 0
    assert capsys.readouterr().out == "\n"


def test_xmlquery_requires_key(case_root):
    """Test that xmlquery called with no key errors"""
    with pytest.raises(SystemExit):
        xmlquery(case_root, [])


def test_case_submit_records_run_and_archive(make_case, fake_pbs_env):
    """Test that case_submit queues case.run then st_archive (after case.run
    succeeds) and records both job IDs
    """
    root = make_case()

    rc = case_submit(root, [])

    assert rc == 0
    ids = job_ids(root)
    assert list(ids) == ["case.run", "case.st_archive"]
    jobs = {j["id"]: j for j in fake_pbs_env.jobs()}
    assert sorted(jobs) == sorted(ids.values())
    assert jobs[ids["case.run"]]["depend"] is None
    assert jobs[ids["case.st_archive"]]["depend"] == ["afterok", ids["case.run"]]


def test_case_submit_no_archive(make_case):
    """Test that case_submit with no_archive only does the one job"""
    root = make_case({"no_archive": True})

    assert case_submit(root, []) == 0
    assert list(job_ids(root)) == ["case.run"]


def test_case_submit_injected_failure(make_case, capsys):
    """Test that case_submit with an injected failure actually fails"""
    root = make_case({"case.submit": "fail"})

    rc = case_submit(root, [])

    assert rc == 1
    assert "injected failure" in capsys.readouterr().err
    case_dict = json.loads((root / CASE_DICT_FILE).read_text())
    assert case_dict["JOB_IDS"] is None  # returned before anything was submitted


def test_case_submit_without_faults_file_has_no_faults(make_case):
    """Test that deleting a case's faults file clears its faults (how the e2e
    tests clear an injected failure before a retry)
    """
    root = make_case({"case.submit": "fail"})
    (root / cime.FAULTS_DICT_FILE).unlink()

    assert case_submit(root, []) == 0
    assert "case.run" in job_ids(root)


def test_case_submit_missing_case_json(make_case, capsys):
    """Test that case_submit with a missing case json fails"""
    root = make_case()
    (root / CASE_DICT_FILE).unlink()

    rc = case_submit(root, [])

    assert rc == 1
    assert "Run create_newcase first" in capsys.readouterr().out


def test_case_submit_ignores_unknown_args(make_case):
    """Test that case_submit ignores unknown arguments"""
    root = make_case()

    assert case_submit(root, ["--no-batch", "--bogus", "value"]) == 0


@pytest.mark.parametrize("fault", ["case.run", "st_archive"])
def test_case_submit_job_faults_do_not_fail_submit(make_case, fault):
    """Test cases_submit with injected fails in case.run and st_archive"""
    root = make_case({fault: "fail"})

    assert case_submit(root, []) == 0
    assert "JOB_IDS" in json.loads((root / CASE_DICT_FILE).read_text())


def _job_script(tmp_path: Path, name: str = "job") -> Path:
    """Write a minimal PBS job script"""
    script = tmp_path / f"{name}.sh"
    script.write_text(f"#!/bin/bash\n#PBS -N {name}\necho hi\n", encoding="utf-8")
    return script


def test_qsub_submits_to_fake_pbs(fake_pbs_env, tmp_path):
    """Test that _qsub submits a script with the fake qsub and returns its job ID"""
    job_id = _qsub(_job_script(tmp_path))
    [job] = fake_pbs_env.jobs()
    assert job["id"] == job_id
    assert job["depend"] is None


def test_qsub_with_depend(fake_pbs_env, tmp_path):
    """Test that _qsub passes depend as an afterok dependency"""
    first = _qsub(_job_script(tmp_path, "first"))
    second = _qsub(_job_script(tmp_path, "second"), depend=first)
    job = next(j for j in fake_pbs_env.jobs() if j["id"] == second)
    assert job["depend"] == ["afterok", first]


def test_qsub_failure_raises(fake_pbs_env, tmp_path):
    """Test that a failed qsub raises with qsub's error"""
    fake_pbs_env.fail_qsub(r"job\.sh$")
    with pytest.raises(RuntimeError, match="qsub failed for .*injected failure"):
        _qsub(_job_script(tmp_path))
    assert not fake_pbs_env.jobs()


def test_preview_namelists_writes_lnd_in(case_root):
    """Test that preview_namelists writes the lnd_in file"""
    (case_root / "user_nl_clm").write_text("hist_nhtfrq = -24\n", encoding="utf-8")

    rc = preview_namelists(case_root)

    assert rc == 0
    assert (case_root / "CaseDocs" / "lnd_in").read_text() == "hist_nhtfrq = -24\n"


def test_preview_namelists_overwrites_existing(case_root):
    """Test that preview_namelists overwrites whatever was existing"""
    preview_namelists(case_root)

    (case_root / "user_nl_clm").write_text("hist_nhtfrq = -24\n", encoding="utf-8")
    rc = preview_namelists(case_root)

    assert rc == 0
    assert (case_root / "CaseDocs" / "lnd_in").read_text() == "hist_nhtfrq = -24\n"


def test_preview_namelists_requires_user_nl_clm(case_root):
    """Test that preview_namelists fails with missing user_nl_clm"""
    (case_root / "user_nl_clm").unlink()

    with pytest.raises(FileNotFoundError):
        preview_namelists(case_root)


def test_main_unknown_command_exits_2(monkeypatch, tmp_path, capsys):
    """Test that main rejects an unknown command"""
    assert _run_main(monkeypatch, "bogus", str(tmp_path)) == 2
    assert "unknown command 'bogus'" in capsys.readouterr().err


def test_main_create_newcase_gets_case_root(monkeypatch, tmp_path):
    """Test that main passes create_newcase just the case root"""
    monkeypatch.delenv("FAKE_CIME", raising=False)
    case_root = tmp_path / "case"
    assert _run_main(monkeypatch, "create_newcase", str(case_root)) == 0
    assert (case_root / CASE_DICT_FILE).is_file()


def test_main_xmlquery_gets_case_root_and_args(monkeypatch, case_root, capsys):
    """Test that main passes other commands the case root and the rest of argv"""
    assert _run_main(monkeypatch, "xmlquery", str(case_root), "--value", "CASE") == 0
    assert capsys.readouterr().out.strip() == "case"


def test_main_create_clone_gets_its_flags(monkeypatch, tmp_path, clone_source):
    """Test that main passes create_clone its --case/--clone flags"""
    case = tmp_path / "clone"
    code = _run_main(
        monkeypatch, "create_clone", "--case", str(case), "--clone", str(clone_source)
    )
    assert code == 0
    assert (case / CASE_DICT_FILE).is_file()
