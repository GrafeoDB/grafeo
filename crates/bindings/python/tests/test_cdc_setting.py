"""The CDC setting is stored with the database (#679): `enable_cdc()` and `disable_cdc()` persist
it, so `cdc_enabled` after a close and a reopen is what it was, and they raise when it cannot be
stored (a read-only open). An in-memory database toggles as before.
"""

import pathlib

import pytest

grafeo = pytest.importorskip("grafeo")

pytestmark = pytest.mark.requires("cdc")


def test_the_setting_survives_a_reopen_both_ways(tmp_path):
    path = str(tmp_path / "amsterdam.grafeo")
    db = grafeo.GrafeoDB(path)
    assert db.cdc_enabled is False
    db.enable_cdc()
    db.close()

    db = grafeo.GrafeoDB(path)
    assert db.cdc_enabled is True
    db.disable_cdc()
    db.close()

    db = grafeo.GrafeoDB(path)
    assert db.cdc_enabled is False
    db.close()


def test_a_read_only_open_raises_and_changes_nothing(tmp_path):
    path = str(tmp_path / "berlin.grafeo")
    db = grafeo.GrafeoDB(path)
    db.execute("INSERT (:Person {name: 'Alix'})")
    db.close()

    reader = grafeo.GrafeoDB.open_read_only(path)
    with pytest.raises(grafeo.GrafeoError):
        reader.enable_cdc()
    assert reader.cdc_enabled is False
    reader.close()


def test_an_in_memory_database_toggles():
    db = grafeo.GrafeoDB()
    db.enable_cdc()
    assert db.cdc_enabled is True
    db.disable_cdc()
    assert db.cdc_enabled is False


def _kept_behind(path):
    """The bytes of every WAL segment but the newest: what checkpoints kept behind."""
    segments = sorted(pathlib.Path(path + ".wal").glob("wal_*.log"))
    return sum(segment.stat().st_size for segment in segments[:-1])


def test_turning_cdc_off_removes_the_cursor_and_the_next_checkpoint_trims_the_wal(tmp_path):
    """The stored setting keeps the WAL from the change feed's cursor; turning it off removes
    the cursor (the file in the WAL directory), and the next checkpoint releases the WAL the
    cursor kept (the segments behind the newest)."""
    path = str(tmp_path / "lyon.grafeo")
    cursor = pathlib.Path(path + ".wal") / "cdc_cursor_default.json"
    note = "x" * 300
    db = grafeo.GrafeoDB(path)
    db.enable_cdc()
    assert cursor.exists(), "on writes the cursor"
    for round_ in range(4):
        db.execute(
            f"UNWIND range({round_ * 100}, {round_ * 100 + 99}) AS i "
            f"INSERT (:Person {{i: i, note: '{note}'}})"
        )
        db.wal_checkpoint()
    assert _kept_behind(path) > 0, "the cursor keeps the WAL through checkpoints"

    db.disable_cdc()
    assert not cursor.exists(), "off removes the cursor"
    db.execute(f"INSERT (:Person {{i: 10000, note: '{note}'}})")
    db.wal_checkpoint()
    assert _kept_behind(path) == 0, "the next checkpoint releases what the cursor kept"
    db.close()
