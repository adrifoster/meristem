"""Tests for fake PBS methods"""

from __future__ import annotations

import json
import sys

import pytest

from .pbs import main, qsub, qstat, qdel, _control, _load, _save
from meristem.run.scheduler import _parse_qstat_states


def test_control_reads_control_file(tmp_path, monkeypatch):
    """Test that _control returns the injected faults in the spool's control file"""
    monkeypatch.setenv("FAKE_PBS_SPOOL", str(tmp_path))
    faults = {"qsub_fail": r"_advance\.pbs$", "qstat_broken": True}
    (tmp_path / "control.json").write_text(json.dumps(faults), encoding="utf-8")

    assert _control() == faults


def test_control_without_control_file_is_empty(tmp_path, monkeypatch):
    """Test that _control returns no faults when there's no control file"""
    monkeypatch.setenv("FAKE_PBS_SPOOL", str(tmp_path))

    assert _control() == {}


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


def _write_control(spool, faults: dict):
    """Write injected faults to the spool's control file"""
    (spool / "control.json").write_text(json.dumps(faults), encoding="utf-8")


def test_qsub_injected_failure(tmp_path, monkeypatch, capsys):
    """Test that qsub fails for a script matching the qsub_fail pattern, without
    queueing it, and still queues scripts that don't match
    """
    monkeypatch.setenv("FAKE_PBS_SPOOL", str(tmp_path))
    _write_control(tmp_path, {"qsub_fail": r"_advance\.pbs$"})
    advance = tmp_path / "run_advance.pbs"
    advance.write_text("#!/bin/bash\n#PBS -N advance\necho hi\n")
    other = tmp_path / "run_stage.pbs"
    other.write_text("#!/bin/bash\n#PBS -N stage\necho hi\n")

    assert qsub([str(advance)]) == 1
    assert f"injected failure for {advance}" in capsys.readouterr().err
    assert not _load()

    assert qsub([str(other)]) == 0
    assert [j["name"] for j in _load()] == ["stage"]


@pytest.mark.parametrize("in_script", [True, False])
def test_qsub_unparseable_depend_fails(tmp_path, monkeypatch, capsys, in_script):
    """Test that qsub refuses a dependency it can't parse (in the script or on
    the command line) instead of queueing the job without it
    """
    monkeypatch.setenv("FAKE_PBS_SPOOL", str(tmp_path))
    script = tmp_path / "job.sh"
    depend = "-W depend=afterany:1.fakepbs"  # afterany isn't supported
    if in_script:
        script.write_text(f"#!/bin/bash\n#PBS -N job\n#PBS {depend}\necho hi\n")
        argv = [str(script)]
    else:
        script.write_text("#!/bin/bash\n#PBS -N job\necho hi\n")
        argv = [depend, str(script)]

    assert qsub(argv) == 1
    assert "could not parse dependency" in capsys.readouterr().err
    assert not _load()


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


def test_qstat_injected_failure(tmp_path, monkeypatch, capsys):
    """Test that a broken qstat exits 2 with an error and reports no jobs"""
    monkeypatch.setenv("FAKE_PBS_SPOOL", str(tmp_path))
    _save([{"id": "1.fakepbs", "state": "Q"}])
    _write_control(tmp_path, {"qstat_broken": True})

    assert qstat(["-x", "-f", "1.fakepbs"]) == 2
    out, err = capsys.readouterr()
    assert "qstat: injected failure" in err
    assert out == ""


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


def _run_main(monkeypatch, *argv) -> int:
    """Run pbs.main() with argv as the command line and return its exit code"""
    monkeypatch.setattr(sys, "argv", ["pbs.py", *argv])
    with pytest.raises(SystemExit) as exc:
        main()
    return exc.value.code


def test_main_unknown_command_exits_2(monkeypatch, capsys):
    """Test that main rejects an unknown command"""
    assert _run_main(monkeypatch, "qbogus") == 2
    assert "unknown command 'qbogus'" in capsys.readouterr().err


def test_main_passes_args_and_exit_code(tmp_path, monkeypatch, capsys):
    """Test that main passes the rest of argv to the command and exits with
    the command's return code
    """
    monkeypatch.setenv("FAKE_PBS_SPOOL", str(tmp_path))
    _save([{"id": "1.fakepbs", "state": "Q", "deleted": False}])

    assert _run_main(monkeypatch, "qstat", "-x", "-f", "1.fakepbs") == 0
    assert _parse_qstat_states(capsys.readouterr().out) == {"1": "Q"}

    assert _run_main(monkeypatch, "qdel", "5.fakepbs") == 153
