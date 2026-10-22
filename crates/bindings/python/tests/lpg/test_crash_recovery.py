"""A process that exits without close() leaves its commits, once (#417).

The reporter's server crashed after wal_checkpoint(): on the reopen, the
WAL replay added the edges the checkpoint already held a second time. Each
test here writes in a child process that ends with os._exit(0), so nothing
closes or checkpoints, and then reopens the database.
"""

import subprocess
import sys

from grafeo import GrafeoDB


def crash_after(path, body):
    """Runs `body` with `db` open on `path` in a child process that exits without close()."""
    script = (
        f"import os\nfrom grafeo import GrafeoDB\ndb = GrafeoDB({path!r})\n{body}\nos._exit(0)\n"
    )
    child = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert child.returncode == 0, child.stderr


def knows(db):
    rows = db.execute("MATCH (a)-[:KNOWS]->(b) RETURN a.name AS a, b.name AS b ORDER BY a")
    return [(row["a"], row["b"]) for row in rows]


def test_a_crash_after_wal_checkpoint_keeps_each_edge_once(tmp_path):
    path = str(tmp_path / "server.grafeo")
    crash_after(
        path,
        "db.execute(\"INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})\")\n"
        "db.wal_checkpoint()\n"
        "db.execute(\"INSERT (:Person {name: 'Vincent'})\")",
    )
    # Two more processes reopen, replay the same WAL, and crash again.
    for _ in range(2):
        crash_after(path, "pass")

    db = GrafeoDB.open_read_only(path)
    assert knows(db) == [("Alix", "Gus")]
    assert (db.node_count, db.edge_count) == (3, 1)
    db.close()
    db = GrafeoDB(path)
    assert knows(db) == [("Alix", "Gus")]
    assert (db.node_count, db.edge_count) == (3, 1)
    db.close()


def test_an_edge_deleted_after_the_checkpoint_stays_deleted(tmp_path):
    path = str(tmp_path / "server.grafeo")
    crash_after(
        path,
        "db.execute(\"INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})\")\n"
        "db.wal_checkpoint()\n"
        "db.execute(\"MATCH (:Person {name: 'Alix'})-[k:KNOWS]->() DELETE k\")",
    )
    crash_after(path, "pass")
    db = GrafeoDB(path)
    assert knows(db) == []
    assert (db.node_count, db.edge_count) == (2, 0)
    db.close()
