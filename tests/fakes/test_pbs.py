"""Tests for fake PBS methods"""

from __future__ import annotations

from .pbs import qsub, qstat, qdel, _load, _save
from meristem.run.scheduler import _parse_qstat_states


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
