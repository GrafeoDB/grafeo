"""Automatic checkpoints by interval and WAL size, and their status (#426)."""

import time

import pytest
from grafeo import GrafeoDB


def wait_until(condition, what):
    deadline = time.monotonic() + 10
    while not condition():
        assert time.monotonic() < deadline, f"never: {what}"
        time.sleep(0.02)


def test_triggers_set_on_an_open_database_checkpoint_and_read_back(tmp_path):
    db = GrafeoDB(str(tmp_path / "amsterdam.grafeo"))
    status = db.checkpoint_status()
    assert (status["interval_secs"], status["wal_bytes"]) == (None, None)
    assert status["failures"] == 0
    assert status["stopped"] is False

    db.execute("INSERT (:Person {name: 'Alix'})")
    assert db.wal_status()["size_bytes"] > 0
    db.set_checkpoint_triggers(interval_secs=0.3, wal_bytes=64 * 1024 * 1024)
    status = db.checkpoint_status()
    assert status["interval_secs"] == pytest.approx(0.3)
    assert status["wal_bytes"] == 64 * 1024 * 1024
    wait_until(lambda: db.wal_status()["size_bytes"] == 0, "the checkpoint")
    assert db.checkpoint_status()["last_checkpoint"] is not None

    db.set_checkpoint_triggers()
    status = db.checkpoint_status()
    assert (status["interval_secs"], status["wal_bytes"]) == (None, None)
    db.close()


def test_triggers_are_refused_where_nothing_checkpoints():
    db = GrafeoDB()
    with pytest.raises(Exception, match="nothing to checkpoint"):
        db.set_checkpoint_triggers(wal_bytes=1024)


def test_a_zero_trigger_is_refused(tmp_path):
    db = GrafeoDB(str(tmp_path / "berlin.grafeo"))
    with pytest.raises(Exception, match="zero"):
        db.set_checkpoint_triggers(wal_bytes=0)
    with pytest.raises(Exception, match="zero"):
        db.set_checkpoint_triggers(interval_secs=0)
    db.close()
