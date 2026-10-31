"""Persona: Ingrid runs Grafeo in production and upgrades it from 0.5.x.

She opens the released 0.5.44 file (read-only first, then for real, which
migrates it and keeps the old file), checks it, takes a full and an
incremental backup, restores each epoch to a new path, has a backup killed
halfway, and makes the operator's mistakes: restoring over an existing
database, from a folder with no backup, to an epoch no backup covers, an
incremental backup before any full one, a backup of an in-memory database.
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

try:
    import grafeo

    GRAFEO_AVAILABLE = True
except ImportError:
    GRAFEO_AVAILABLE = False

pytestmark = pytest.mark.skipif(not GRAFEO_AVAILABLE, reason="Grafeo Python bindings not installed")

RELEASED = (
    Path(__file__).resolve().parents[4]
    / "crates"
    / "grafeo-engine"
    / "tests"
    / "fixtures"
    / "released"
    / "0.5.44"
    / "closed.grafeo"
)


def names(directory):
    return sorted(path.name for path in directory.iterdir())


def count(db, query="MATCH (n) RETURN count(n) AS c"):
    return db.execute(query).to_list()[0]["c"]


@pytest.fixture
def upgraded(tmp_path):
    """The 0.5.44 file opened read-write once (migrated), and closed."""
    path = tmp_path / "prod.grafeo"
    shutil.copy(RELEASED, path)
    grafeo.GrafeoDB.open(str(path)).close()
    return path


# ---------------------------------------------------------------------------
# The upgrade
# ---------------------------------------------------------------------------


def test_persona_ingrid_looks_at_the_old_file_read_only_first(tmp_path):
    path = tmp_path / "prod.grafeo"
    shutil.copy(RELEASED, path)
    before = path.read_bytes()
    db = grafeo.GrafeoDB.open_read_only(str(path))
    assert count(db) == 9
    db.close()
    assert names(tmp_path) == ["prod.grafeo"]
    assert path.read_bytes() == before


def test_persona_ingrid_upgrades_and_keeps_the_old_file(tmp_path):
    path = tmp_path / "prod.grafeo"
    shutil.copy(RELEASED, path)
    old = grafeo.GrafeoDB.open_read_only(str(path))
    want = {
        "nodes": count(old),
        "edges": count(old, "MATCH ()-[r]->() RETURN count(r) AS c"),
        "graphs": sorted(old.list_graphs()),
        "labels": {entry["name"]: entry["count"] for entry in old.schema()["labels"]},
    }
    old.close()
    db = grafeo.GrafeoDB.open(str(path))
    assert "prod.grafeo.pre-0.6" in names(tmp_path)
    assert (tmp_path / "prod.grafeo.pre-0.6").read_bytes() == RELEASED.read_bytes()
    got = {
        "nodes": count(db),
        "edges": count(db, "MATCH ()-[r]->() RETURN count(r) AS c"),
        "graphs": sorted(db.list_graphs()),
        "labels": {entry["name"]: entry["count"] for entry in db.schema()["labels"]},
    }
    assert got == want
    assert db.validate() == []
    db.execute("INSERT (:Person {name: 'Ingrid'})")
    db.close()
    again = grafeo.GrafeoDB.open(str(path))
    assert count(again, "MATCH (p:Person {name: 'Ingrid'}) RETURN count(p) AS c") == 1
    again.close()


# ---------------------------------------------------------------------------
# Backups and restores
# ---------------------------------------------------------------------------


def test_persona_ingrid_restores_each_epoch_of_a_backup_chain(upgraded, tmp_path):
    backups = tmp_path / "backups"
    db = grafeo.GrafeoDB.open(str(upgraded))
    db.execute("INSERT (:Person {name: 'Ingrid'})")
    db.backup_full(str(backups))
    first = db.current_epoch()
    db.execute("INSERT (:Person {name: 'Harm'})")
    db.backup_incremental(str(backups))
    second = db.current_epoch()
    db.close()
    query = "MATCH (p:Person) WHERE p.name IN ['Ingrid', 'Harm'] RETURN p.name AS n ORDER BY n"
    for epoch, people in [(first, ["Ingrid"]), (second, ["Harm", "Ingrid"])]:
        out = tmp_path / f"restored-{epoch}.grafeo"
        grafeo.GrafeoDB.restore_to_epoch(str(backups), epoch, str(out))
        restored = grafeo.GrafeoDB.open_read_only(str(out))
        assert [row["n"] for row in restored.execute(query).to_list()] == people
        assert count(restored) == 9 + len(people)
        restored.close()


def test_persona_ingrid_will_not_restore_over_a_database(upgraded, tmp_path):
    backups = tmp_path / "backups"
    db = grafeo.GrafeoDB.open(str(upgraded))
    db.backup_full(str(backups))
    epoch = db.current_epoch()
    db.close()
    with pytest.raises(grafeo.GrafeoError, match="restore output path already exists"):
        grafeo.GrafeoDB.restore_to_epoch(str(backups), epoch, str(upgraded))
    reopened = grafeo.GrafeoDB.open_read_only(str(upgraded))
    assert count(reopened) == 9
    reopened.close()


def open_after_a_crash(path):
    deadline = time.monotonic() + 10
    while True:
        try:
            return grafeo.GrafeoDB.open(str(path))
        except grafeo.GrafeoError as error:
            if "locked" not in str(error) or time.monotonic() > deadline:
                raise
            time.sleep(0.05)


def test_persona_ingrid_has_a_backup_killed_halfway(tmp_path):
    path = tmp_path / "big.grafeo"
    db = grafeo.GrafeoDB(str(path))
    db.batch_create_nodes_with_props(["Item"], [{"i": i, "text": "x" * 200} for i in range(50_000)])
    db.close()
    backups = tmp_path / "backups"
    child = (
        "import grafeo\n"
        f"db = grafeo.GrafeoDB.open({str(path)!r})\n"
        "print('start', flush=True)\n"
        f"db.backup_full({str(backups)!r})\n"
    )
    for delay in (0.01, 0.1):
        process = subprocess.Popen([sys.executable, "-c", child], stdout=subprocess.PIPE, text=True)
        process.stdout.readline()
        time.sleep(delay)
        process.kill()
        process.wait()
        # The next backup works and its chain restores. Windows releases a
        # killed process's file lock a moment after it exits: retry briefly.
        db = open_after_a_crash(path)
        assert count(db) == 50_000 + (0 if delay == 0.01 else 1)
        db.backup_full(str(backups))
        db.execute("INSERT (:Item {i: -1})")
        db.backup_incremental(str(backups))
        epoch = db.current_epoch()
        db.close()
        assert not [name for name in names(backups) if name.endswith(".partial")]
        out = tmp_path / f"restored-{delay}.grafeo"
        grafeo.GrafeoDB.restore_to_epoch(str(backups), epoch, str(out))
        restored = grafeo.GrafeoDB.open_read_only(str(out))
        assert count(restored) == count(grafeo.GrafeoDB.open_read_only(str(path)))
        restored.close()


# ---------------------------------------------------------------------------
# Operator mistakes: a clear error, never an internal one
# ---------------------------------------------------------------------------


def test_persona_ingrid_backs_up_an_in_memory_database(tmp_path):
    with pytest.raises(grafeo.GrafeoError, match="a backup needs a persistent database"):
        grafeo.GrafeoDB().backup_full(str(tmp_path / "backups"))


@pytest.mark.parametrize(
    "mistake",
    ["a folder with no backup", "an incremental before any full", "an epoch no backup covers"],
)
def test_persona_ingrid_gets_a_clear_error_for_an_operator_mistake(mistake, upgraded, tmp_path):
    backups = tmp_path / "backups"
    with pytest.raises(grafeo.GrafeoError) as caught:
        if mistake == "a folder with no backup":
            grafeo.GrafeoDB.restore_to_epoch(str(tmp_path / "nothing"), 1, str(tmp_path / "out"))
        elif mistake == "an incremental before any full":
            db = grafeo.GrafeoDB.open(str(upgraded))
            try:
                db.backup_incremental(str(backups))
            finally:
                db.close()
        else:
            db = grafeo.GrafeoDB.open(str(upgraded))
            db.execute("INSERT (:Person {name: 'Ingrid'})")
            db.backup_full(str(backups))
            db.close()
            grafeo.GrafeoDB.restore_to_epoch(str(backups), 0, str(tmp_path / "out"))
    message = str(caught.value)
    assert "Internal error" not in message, message
    assert "backup" in message, message
