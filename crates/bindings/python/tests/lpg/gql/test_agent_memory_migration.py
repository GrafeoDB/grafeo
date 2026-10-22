"""Agent memory migration tests (Discussion #155).

Verifies: persistence with vectors, BYOV 384-dim, bulk import,
concurrent reads, Python lifecycle, hybrid search, and storage size.

Run:
    pytest tests/lpg/gql/test_agent_memory_migration.py -v
"""

import math
import tempfile
import threading
import time
from pathlib import Path

import pytest

try:
    from grafeo import GrafeoDB

    GRAFEO_AVAILABLE = True
except ImportError:
    GRAFEO_AVAILABLE = False

pytestmark = pytest.mark.skipif(not GRAFEO_AVAILABLE, reason="Grafeo Python bindings not installed")


def random_384d_vector(seed: int) -> list[float]:
    """Deterministic pseudo-random 384-dim vector (LCG + L2-normalize)."""
    state = (seed * 6_364_136_223_846_793_005 + 1) & 0xFFFF_FFFF_FFFF_FFFF
    raw = []
    for _ in range(384):
        state = (state * 6_364_136_223_846_793_005 + 1) & 0xFFFF_FFFF_FFFF_FFFF
        raw.append(((state >> 33) / 0xFFFF_FFFF) * 2.0 - 1.0)
    norm = math.sqrt(sum(x * x for x in raw))
    if norm > 0:
        raw = [x / norm for x in raw]
    return raw


ENTITY_LABELS = ["Person", "Concept", "Event", "Document"]
EDGE_TYPES = ["KNOWS", "RELATED_TO", "MENTIONS", "OCCURRED_AT", "AUTHORED"]


# ============================================================================
# Q1 + Q3: Persistence and Python lifecycle
# ============================================================================


class TestPersistence:
    def test_create_persist_reopen(self):
        """Persistent DB survives close/reopen with all data intact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "agent.grafeo")

            db = GrafeoDB(path=db_path)
            for i in range(50):
                label = ENTITY_LABELS[i % len(ENTITY_LABELS)]
                db.create_node(
                    [label],
                    {
                        "text": f"fact_{i}",
                        "confidence": 0.5 + i * 0.01,
                    },
                )
            info = db.info()
            assert info["node_count"] == 50
            db.close()

            # Reopen and verify
            db2 = GrafeoDB.open(db_path)
            info2 = db2.info()
            assert info2["node_count"] == 50, "nodes should survive close/reopen"
            db2.close()

    def test_vector_index_after_reopen(self):
        """Vectors persist, but index must be recreated after reopen."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "vectors.grafeo")

            db = GrafeoDB(path=db_path)
            for i in range(20):
                db.create_node(
                    ["Memory"],
                    {"embedding": random_384d_vector(i), "text": f"fact_{i}"},
                )
            db.create_vector_index("Memory", "embedding", dimensions=384, metric="cosine")

            # Search works before close
            results = db.vector_search("Memory", "embedding", random_384d_vector(0), k=5)
            assert len(results) == 5
            db.close()

            # Reopen: data persists; index metadata is in snapshot v4 format
            # but WAL-based persistence requires manual recreation.
            db2 = GrafeoDB.open(db_path)
            assert db2.info()["node_count"] == 20

            # Recreate vector index after reopen
            db2.create_vector_index("Memory", "embedding", dimensions=384, metric="cosine")
            results2 = db2.vector_search("Memory", "embedding", random_384d_vector(0), k=5)
            assert len(results2) == 5, "search works after reopen + index recreation"
            db2.close()

    def test_context_manager(self):
        """with-statement lifecycle works correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "ctx.grafeo")

            with GrafeoDB(path=db_path) as db:
                db.execute("INSERT (:Memory {text: 'context manager test'})")
                info = db.info()
                assert info["node_count"] == 1

            # After exit: database path should exist (directory for WAL format)
            assert Path(db_path).exists(), "database should exist after context exit"

            # Data persists on reopen
            db2 = GrafeoDB.open(db_path)
            assert db2.info()["node_count"] == 1
            db2.close()

    def test_error_propagation(self):
        """Invalid GQL raises RuntimeError."""
        db = GrafeoDB()
        with pytest.raises(RuntimeError):
            db.execute("THIS IS NOT VALID GQL")

    def test_close_idempotent(self):
        """Calling close() twice should not error."""
        db = GrafeoDB()
        db.execute("INSERT (:Memory {text: 'test'})")
        db.close()
        db.close()  # second close should be fine


# ============================================================================
# Q4: Storage size
# ============================================================================


class TestStorageSize:
    def test_measure_disk_size(self):
        """Measure disk footprint for 500 entities with 384-dim vectors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "size.grafeo")

            db = GrafeoDB(path=db_path)
            for i in range(500):
                label = ENTITY_LABELS[i % len(ENTITY_LABELS)]
                db.create_node(
                    [label],
                    {
                        "text": f"Entity {i}: agent memory content for testing storage",
                        "embedding": random_384d_vector(i),
                        "confidence": 0.5 + i * 0.001,
                    },
                )
            db.close()

            # Measure total disk usage (may be single file or WAL directory)
            total_size = 0
            db_root = Path(db_path)
            if db_root.is_file():
                total_size = db_root.stat().st_size
            elif db_root.is_dir():
                for f in db_root.rglob("*"):
                    if f.is_file():
                        total_size += f.stat().st_size
            # Also check sidecar WAL if separate
            wal_path = Path(db_path + ".wal")
            if wal_path.exists():
                for f in wal_path.rglob("*"):
                    if f.is_file():
                        total_size += f.stat().st_size

            size_mb = total_size / (1024 * 1024)
            print(f"\nStorage: {size_mb:.2f} MB for 500 entities + 384-dim vectors")
            print(f"Per entity: {total_size / 500:.0f} bytes")
            assert total_size > 0, "database should have non-zero disk footprint"


# ============================================================================
# Q5: Bring Your Own Vectors (BYOV) 384-dim
# ============================================================================


class TestBYOV:
    def test_384_dim_vectors(self):
        """Insert 384-dim vectors as list[float], create index, search."""
        db = GrafeoDB()
        for i in range(30):
            db.create_node(
                ["Memory"],
                {"embedding": random_384d_vector(i)},
            )
        db.create_vector_index("Memory", "embedding", dimensions=384, metric="cosine")

        # Search with the same vector as node 0
        results = db.vector_search("Memory", "embedding", random_384d_vector(0), k=5)
        assert len(results) == 5

        # First result should be very close (same vector)
        node_id, distance = results[0]
        assert distance < 0.1, f"closest should have distance < 0.1, got {distance}"

        # Results ordered by distance
        for j in range(len(results) - 1):
            assert results[j][1] <= results[j + 1][1], "results should be sorted"

    def test_batch_create_vectors(self):
        """batch_create_nodes with 100 384-dim vectors."""
        db = GrafeoDB()
        vectors = [random_384d_vector(i) for i in range(100)]
        ids = db.batch_create_nodes("Memory", "embedding", vectors)
        assert len(ids) == 100

        db.create_vector_index("Memory", "embedding", dimensions=384, metric="cosine")
        results = db.vector_search("Memory", "embedding", random_384d_vector(0), k=10)
        assert len(results) == 10

    def test_hybrid_search(self):
        """Combined text + vector search for agent memory retrieval."""
        db = GrafeoDB()
        facts = [
            ("Alix works at the Amsterdam office", 0),
            ("Gus is a data scientist in Berlin", 1),
            ("Vincent leads the Paris team", 2),
            ("Jules manages graph databases", 3),
            ("Mia researches neural networks", 4),
            ("Butch develops in Rust and Python", 5),
        ]
        for text, seed in facts:
            db.create_node(
                ["Memory"],
                {"content": text, "embedding": random_384d_vector(seed)},
            )

        db.create_text_index("Memory", "content")
        db.create_vector_index("Memory", "embedding", dimensions=384, metric="cosine")

        results = db.hybrid_search(
            "Memory",
            text_property="content",
            vector_property="embedding",
            query_text="graph database",
            query_vector=random_384d_vector(3),  # bias toward Jules
            k=3,
        )
        assert len(results) > 0, "hybrid search should return results"


# ============================================================================
# Q6: Bulk import
# ============================================================================


class TestBulkImport:
    def test_gql_loop_import(self):
        """Import 500 entities via execute() loop in a transaction."""
        db = GrafeoDB()
        start = time.time()

        with db.begin_transaction() as tx:
            for i in range(500):
                label = ENTITY_LABELS[i % len(ENTITY_LABELS)]
                tx.execute(
                    f"INSERT (:{label} {{name: 'entity_{i}', confidence: {0.5 + i * 0.001}}})"
                )

        elapsed = time.time() - start
        info = db.info()
        assert info["node_count"] == 500
        print(f"\nGQL loop import: 500 entities in {elapsed:.3f}s ({500 / elapsed:.0f} ops/sec)")

    def test_api_import(self):
        """Import 500 entities via create_node() API."""
        db = GrafeoDB()
        start = time.time()

        for i in range(500):
            label = ENTITY_LABELS[i % len(ENTITY_LABELS)]
            db.create_node(
                [label],
                {"name": f"entity_{i}", "confidence": 0.5 + i * 0.001},
            )

        elapsed = time.time() - start
        info = db.info()
        assert info["node_count"] == 500
        print(f"\nAPI import: 500 entities in {elapsed:.3f}s ({500 / elapsed:.0f} ops/sec)")


# ============================================================================
# Q2: Concurrent reads
# ============================================================================


class TestConcurrency:
    def test_threaded_vector_search(self):
        """4 threads doing vector_search simultaneously."""
        db = GrafeoDB()
        # Seed data
        for i in range(200):
            db.create_node(
                ["Memory"],
                {"embedding": random_384d_vector(i)},
            )
        db.create_vector_index("Memory", "embedding", dimensions=384, metric="cosine")

        errors = []
        results_count = []
        barrier = threading.Barrier(4)

        def reader(thread_id):
            try:
                barrier.wait()
                for q in range(20):
                    query = random_384d_vector(10_000 + thread_id * 100 + q)
                    res = db.vector_search("Memory", "embedding", query, k=5)
                    results_count.append(len(res))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=reader, args=(tid,)) for tid in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"reader threads errored: {errors}"
        assert all(r == 5 for r in results_count), (
            f"all searches should return 5 results: {results_count}"
        )
