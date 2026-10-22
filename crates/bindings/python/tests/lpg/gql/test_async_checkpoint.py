"""Tests for WAL checkpoint and save persistence through the Python binding layer.

Covers checkpoint-based durability, property round-trips, empty DB edge cases,
delete-then-checkpoint scenarios, multiple checkpoints, and in-memory save.
"""

import tempfile
from pathlib import Path

import pytest
from grafeo import GrafeoDB


def _has_grafeo_file_support():
    """Check if the grafeo-file format is supported by trying to create one."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "probe.grafeo")
        db = GrafeoDB(path=db_path)
        db.execute("INSERT (:Probe {x: 1})")
        db.close()
        return Path(db_path).is_file()


_skip_no_grafeo_file = pytest.mark.skipif(
    not _has_grafeo_file_support(),
    reason="grafeo-file feature not enabled (need storage feature)",
)


@_skip_no_grafeo_file
class TestAsyncCheckpoint:
    """Tests for WAL checkpoint and save persistence."""

    def test_checkpoint_preserves_nodes_and_edges(self):
        """Checkpoint persists Person/Company nodes and a WORKS_AT edge."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "ckpt.grafeo")

            db = GrafeoDB(path=db_path)
            db.execute("INSERT (:Person {name: 'Alix', age: 30})")
            db.execute("INSERT (:Person {name: 'Gus', age: 25})")
            db.execute("INSERT (:Company {name: 'Acme Corp', founded: 2010})")
            db.execute(
                "MATCH (a:Person {name: 'Alix'}), (c:Company {name: 'Acme Corp'}) "
                "INSERT (a)-[:WORKS_AT {role: 'Engineer'}]->(c)"
            )

            db.wal_checkpoint()
            db.close()

            db2 = GrafeoDB.open(db_path)
            info = db2.info()
            assert info["node_count"] == 3
            assert info["edge_count"] == 1

            result = db2.execute("MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN p.name, c.name")
            assert len(result) == 1
            row = result[0]
            assert row["p.name"] == "Alix"
            assert row["c.name"] == "Acme Corp"

            db2.close()

    def test_checkpoint_preserves_properties(self):
        """Multiple property types round-trip through checkpoint and reopen."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "props.grafeo")

            db = GrafeoDB(path=db_path)
            db.execute(
                "INSERT (:Thing {"
                "  str_val: 'hello', "
                "  int_val: 42, "
                "  float_val: 3.14, "
                "  bool_val: true, "
                "  list_val: [1, 2, 3]"
                "})"
            )

            db.wal_checkpoint()
            db.close()

            db2 = GrafeoDB.open(db_path)
            result = db2.execute(
                "MATCH (t:Thing) RETURN t.str_val, t.int_val, t.float_val, t.bool_val, t.list_val"
            )
            assert len(result) == 1
            row = result[0]
            assert row["t.str_val"] == "hello"
            assert row["t.int_val"] == 42
            assert abs(row["t.float_val"] - 3.14) < 0.001
            assert row["t.bool_val"] is True
            assert row["t.list_val"] == [1, 2, 3]

            db2.close()

    def test_checkpoint_empty_db(self):
        """Checkpointing an empty persistent DB should not error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "empty.grafeo")

            db = GrafeoDB(path=db_path)
            db.wal_checkpoint()
            db.close()

            db2 = GrafeoDB.open(db_path)
            info = db2.info()
            assert info["node_count"] == 0
            assert info["edge_count"] == 0
            db2.close()

    def test_checkpoint_after_deletes(self):
        """After deleting a node and checkpointing, the count should be 2."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "deletes.grafeo")

            db = GrafeoDB(path=db_path)
            db.execute("INSERT (:Person {name: 'Alix'})")
            db.execute("INSERT (:Person {name: 'Gus'})")
            db.execute("INSERT (:Person {name: 'Vincent'})")

            # Delete one
            db.execute("MATCH (p:Person {name: 'Vincent'}) DELETE p")

            db.wal_checkpoint()
            db.close()

            db2 = GrafeoDB.open(db_path)
            info = db2.info()
            assert info["node_count"] == 2

            result = db2.execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
            names = sorted(row["p.name"] for row in result)
            assert names == ["Alix", "Gus"]

            db2.close()

    def test_multiple_checkpoints(self):
        """Data from two separate insert-then-checkpoint phases both persist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "multi.grafeo")

            db = GrafeoDB(path=db_path)

            # Phase 1
            db.execute("INSERT (:Person {name: 'Alix'})")
            db.wal_checkpoint()

            # Phase 2
            db.execute("INSERT (:Person {name: 'Gus'})")
            db.execute("INSERT (:Person {name: 'Vincent'})")
            db.wal_checkpoint()

            db.close()

            db2 = GrafeoDB.open(db_path)
            info = db2.info()
            assert info["node_count"] == 3

            result = db2.execute("MATCH (p:Person) RETURN p.name")
            names = sorted(row["p.name"] for row in result)
            assert names == ["Alix", "Gus", "Vincent"]

            db2.close()

    def test_save_then_open(self):
        """An in-memory DB saved to .grafeo can be reopened with all data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "saved.grafeo")

            db = GrafeoDB()
            db.execute("INSERT (:Person {name: 'Alix', age: 30})")
            db.execute("INSERT (:Person {name: 'Gus', age: 25})")
            db.execute(
                "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) "
                "INSERT (a)-[:KNOWS {since: 2020}]->(b)"
            )
            db.save(db_path)

            db2 = GrafeoDB.open(db_path)
            info = db2.info()
            assert info["node_count"] == 2
            assert info["edge_count"] == 1

            result = db2.execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
            names = sorted(row["p.name"] for row in result)
            assert names == ["Alix", "Gus"]

            result_edge = db2.execute("MATCH ()-[e:KNOWS]->() RETURN e.since")
            assert len(result_edge) == 1
            assert result_edge[0]["e.since"] == 2020

            db2.close()
