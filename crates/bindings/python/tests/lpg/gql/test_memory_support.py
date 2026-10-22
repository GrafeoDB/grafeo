"""Python integration tests for grafeo-memory engine support.

Tests the engine features that grafeo-memory depends on:
- batch_create_nodes_with_props
- Property indexes
- Vector search with operator filters ($gte, $lt, etc.)
- Temporal property versioning (when feature enabled)
- Property batch operations

These tests run against the Python binding to catch regressions
before they propagate to grafeo-memory.
"""

import pytest

# =============================================================================
# batch_create_nodes_with_props
# =============================================================================


class TestBatchCreateNodesWithProps:
    def test_creates_nodes_with_mixed_properties(self, db):
        ids = db.batch_create_nodes_with_props(
            "Memory",
            [
                {"text": "hello", "user_id": "u1", "score": 0.95},
                {"text": "world", "user_id": "u1", "score": 0.80},
            ],
        )
        assert len(ids) == 2

        node = db.get_node(ids[0])
        props = node.properties if hasattr(node, "properties") else {}
        if callable(props):
            props = props()
        assert props["text"] == "hello"
        assert props["user_id"] == "u1"

    def test_creates_nodes_with_vectors(self, db):
        ids = db.batch_create_nodes_with_props(
            "Doc",
            [
                {"text": "doc1", "embedding": [1.0, 0.0, 0.0]},
                {"text": "doc2", "embedding": [0.0, 1.0, 0.0]},
            ],
        )
        assert len(ids) == 2

        node = db.get_node(ids[0])
        props = node.properties if hasattr(node, "properties") else {}
        if callable(props):
            props = props()
        assert "embedding" in props
        assert len(props["embedding"]) == 3

    def test_empty_list_returns_empty(self, db):
        ids = db.batch_create_nodes_with_props("Memory", [])
        assert ids == []

    def test_nodes_with_different_property_sets(self, db):
        ids = db.batch_create_nodes_with_props(
            "Item",
            [
                {"text": "short"},
                {
                    "text": "detailed",
                    "user_id": "u1",
                    "priority": 1,
                    "archived": False,
                    "score": 0.5,
                },
            ],
        )
        assert len(ids) == 2

    def test_vector_auto_indexed(self, db):
        """Vectors in batch-created nodes are auto-inserted into matching indexes."""
        db.create_vector_index("Doc", "emb", dimensions=3, metric="cosine")

        db.batch_create_nodes_with_props(
            "Doc",
            [
                {"emb": [1.0, 0.0, 0.0]},
                {"emb": [0.0, 1.0, 0.0]},
            ],
        )

        results = db.vector_search("Doc", "emb", [1.0, 0.0, 0.0], k=10)
        assert len(results) == 2

    def test_preserves_node_count(self, db):
        assert db.node_count == 0
        db.batch_create_nodes_with_props(
            "Memory",
            [
                {"text": "one"},
                {"text": "two"},
                {"text": "three"},
            ],
        )
        assert db.node_count == 3


# =============================================================================
# Property indexes
# =============================================================================


class TestPropertyIndexes:
    def test_create_and_check_index(self, db):
        db.create_property_index("user_id")
        assert db.has_property_index("user_id")

    def test_find_nodes_by_property(self, db):
        db.create_property_index("user_id")
        db.batch_create_nodes_with_props(
            "Memory",
            [
                {"text": "a", "user_id": "u1"},
                {"text": "b", "user_id": "u2"},
                {"text": "c", "user_id": "u1"},
            ],
        )
        nodes = db.find_nodes_by_property("user_id", "u1")
        assert len(nodes) == 2

    def test_drop_index(self, db):
        db.create_property_index("temp_field")
        assert db.has_property_index("temp_field")
        db.drop_property_index("temp_field")
        assert not db.has_property_index("temp_field")

    def test_index_does_not_exist(self, db):
        assert not db.has_property_index("nonexistent_field")


# =============================================================================
# Vector search with operator filters
# =============================================================================


class TestVectorSearchOperatorFilters:
    @pytest.fixture
    def memory_db(self, db):
        """Set up a DB with indexed Memory nodes for filter tests."""
        db.create_vector_index("Memory", "embedding", dimensions=3, metric="cosine")
        db.batch_create_nodes_with_props(
            "Memory",
            [
                {
                    "text": "old u1",
                    "user_id": "u1",
                    "created_at": 1000,
                    "embedding": [1.0, 0.0, 0.0],
                },
                {
                    "text": "new u1",
                    "user_id": "u1",
                    "created_at": 2000,
                    "embedding": [0.0, 1.0, 0.0],
                },
                {
                    "text": "newest u1",
                    "user_id": "u1",
                    "created_at": 3000,
                    "embedding": [0.0, 0.0, 1.0],
                },
                {
                    "text": "new u2",
                    "user_id": "u2",
                    "created_at": 2000,
                    "embedding": [1.0, 1.0, 0.0],
                },
            ],
        )
        return db

    def test_gte_filter(self, memory_db):
        results = memory_db.vector_search(
            "Memory",
            "embedding",
            [1.0, 0.0, 0.0],
            k=10,
            filters={"created_at": {"$gte": 2000}},
        )
        assert len(results) == 3, "3 nodes have created_at >= 2000"

    def test_lt_filter(self, memory_db):
        results = memory_db.vector_search(
            "Memory",
            "embedding",
            [1.0, 0.0, 0.0],
            k=10,
            filters={"created_at": {"$lt": 2000}},
        )
        assert len(results) == 1, "1 node has created_at < 2000"

    def test_equality_filter(self, memory_db):
        results = memory_db.vector_search(
            "Memory",
            "embedding",
            [1.0, 0.0, 0.0],
            k=10,
            filters={"user_id": "u1"},
        )
        assert len(results) == 3, "3 nodes belong to u1"

    def test_combined_equality_and_operator(self, memory_db):
        results = memory_db.vector_search(
            "Memory",
            "embedding",
            [1.0, 0.0, 0.0],
            k=10,
            filters={"user_id": "u1", "created_at": {"$gte": 2000}},
        )
        assert len(results) == 2, "2 u1 nodes have created_at >= 2000"

    def test_no_filter_returns_all(self, memory_db):
        results = memory_db.vector_search(
            "Memory",
            "embedding",
            [1.0, 0.0, 0.0],
            k=10,
        )
        assert len(results) == 4

    def test_impossible_filter_returns_empty(self, memory_db):
        results = memory_db.vector_search(
            "Memory",
            "embedding",
            [1.0, 0.0, 0.0],
            k=10,
            filters={"created_at": {"$gt": 99999}},
        )
        assert len(results) == 0


# =============================================================================
# Temporal property versioning (requires temporal feature)
# =============================================================================


class TestTemporalPropertyVersioning:
    """Tests for get_node_property_at_epoch and get_node_property_history.

    These require the `temporal` feature to be enabled at build time.
    Tests are skipped if the methods are not available.
    """

    def test_property_history_available(self, db):
        if not hasattr(db, "get_node_property_history"):
            pytest.skip("temporal feature not enabled")
        node = db.create_node(["Person"], {"name": "Alix"})
        node_id = node.id if hasattr(node, "id") else node
        db.set_node_property(node_id, "name", "Alicia")
        history = db.get_node_property_history(node_id, "name")
        assert len(history) == 2
        assert history[0][1] == "Alix"
        assert history[1][1] == "Alicia"

    def test_property_at_epoch_available(self, db):
        if not hasattr(db, "get_node_property_at_epoch"):
            pytest.skip("temporal feature not enabled")
        # Use transactions to get distinct epochs
        with db.begin_transaction() as tx:
            tx.execute("INSERT (:Person {name: 'Gus'})")
            tx.commit()
        epoch_after_create = db.current_epoch()

        node_id = db.execute("MATCH (p:Person {name: 'Gus'}) RETURN id(p)").scalar()

        with db.begin_transaction() as tx:
            tx.execute("MATCH (p:Person {name: 'Gus'}) SET p.name = 'Gustav'")
            tx.commit()

        val = db.get_node_property_at_epoch(node_id, "name", epoch_after_create)
        assert val == "Gus"

    def test_all_property_history_available(self, db):
        if not hasattr(db, "get_all_node_property_history"):
            pytest.skip("temporal feature not enabled")
        node = db.create_node(["Person"], {"name": "Alix", "age": 30})
        node_id = node.id if hasattr(node, "id") else node
        db.set_node_property(node_id, "age", 31)

        all_hist = db.get_all_node_property_history(node_id)
        assert "name" in all_hist
        assert "age" in all_hist
        assert len(all_hist["age"]) == 2

    def test_property_history_empty_for_nonexistent(self, db):
        if not hasattr(db, "get_node_property_history"):
            pytest.skip("temporal feature not enabled")
        history = db.get_node_property_history(99999, "name")
        assert history == []


# =============================================================================
# Transactions
# =============================================================================


class TestTransactions:
    def test_begin_and_commit(self, db):
        with db.begin_transaction() as tx:
            tx.execute("INSERT (:Person {name: 'Alix'})")
            tx.commit()
        result = db.execute("MATCH (p:Person) RETURN p.name")
        assert len(list(result)) == 1

    def test_rollback_discards_changes(self, db):
        with db.begin_transaction() as tx:
            tx.execute("INSERT (:Person {name: 'Ghost'})")
            tx.rollback()
        result = db.execute("MATCH (p:Person) RETURN p.name")
        assert len(list(result)) == 0

    def test_isolation_levels(self, db):
        for level in ["read_committed", "snapshot", "serializable"]:
            with db.begin_transaction(level) as tx:
                tx.execute("MATCH (n) RETURN n LIMIT 1")
                tx.commit()

    def test_context_manager_auto_commits(self, db):
        """Clean exit from context manager auto-commits."""
        with db.begin_transaction() as tx:
            tx.execute("INSERT (:AutoCommit {val: 1})")
        # If auto-commit works, the node should exist
        result = db.execute("MATCH (n:AutoCommit) RETURN n.val")
        rows = list(result)
        assert len(rows) == 1


# =============================================================================
# CDC / node_history
# =============================================================================


class TestCDCNodeHistory:
    def test_node_history_after_create(self, db):
        if not hasattr(db, "node_history"):
            pytest.skip("CDC not available")
        db.enable_cdc()
        node = db.create_node(["Memory"], {"text": "hello"})
        node_id = node.id if hasattr(node, "id") else node
        events = db.node_history(node_id)
        assert len(events) == 1

    def test_node_history_after_property_change(self, db):
        if not hasattr(db, "node_history"):
            pytest.skip("CDC not available")
        db.enable_cdc()
        node = db.create_node(["Memory"], {"text": "v1"})
        node_id = node.id if hasattr(node, "id") else node
        db.set_node_property(node_id, "text", "v2")
        events = db.node_history(node_id)
        assert len(events) == 2

    def test_node_history_empty_for_nonexistent(self, db):
        if not hasattr(db, "node_history"):
            pytest.skip("CDC not available")
        events = db.node_history(99999)
        assert events == []
