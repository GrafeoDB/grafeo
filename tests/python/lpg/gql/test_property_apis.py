"""Tests for direct property, label, and index APIs.

Covers: set_node_property, set_edge_property, remove_node_property,
remove_edge_property, add_node_label, remove_node_label, get_node_labels,
create_property_index, drop_property_index, has_property_index,
find_nodes_by_property, get_nodes_by_label, get_property_batch.
"""

import pytest
from datetime import datetime, timezone

try:
    import grafeo

    GRAFEO_AVAILABLE = True
except ImportError:
    GRAFEO_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not GRAFEO_AVAILABLE, reason="Grafeo Python bindings not installed"
)


@pytest.fixture
def db():
    """Create a fresh in-memory database."""
    return grafeo.GrafeoDB()


@pytest.fixture
def populated_db(db):
    """Database with a few nodes and edges pre-created."""
    alice = db.create_node(["Person"], {"name": "Alice", "age": 30})
    bob = db.create_node(["Person"], {"name": "Bob", "age": 25})
    charlie = db.create_node(["Person", "Employee"], {"name": "Charlie", "age": 35})
    edge = db.create_edge(alice.id, bob.id, "KNOWS", {"since": 2020})
    return {
        "db": db,
        "alice": alice,
        "bob": bob,
        "charlie": charlie,
        "edge": edge,
    }


# ── Node Property Manipulation ──────────────────────────────────────


class TestSetNodeProperty:
    """Test set_node_property() method."""

    def test_set_new_property(self, populated_db):
        db = populated_db["db"]
        alice = populated_db["alice"]
        db.set_node_property(alice.id, "city", "NYC")
        node = db.get_node(alice.id)
        assert node is not None
        result = db.execute(
            f"MATCH (n:Person) WHERE id(n) = {alice.id} RETURN n.city AS c"
        )
        rows = list(result)
        assert rows[0]["c"] == "NYC"

    def test_overwrite_existing_property(self, populated_db):
        db = populated_db["db"]
        alice = populated_db["alice"]
        db.set_node_property(alice.id, "name", "Alicia")
        result = db.execute(
            f"MATCH (n:Person) WHERE id(n) = {alice.id} RETURN n.name AS name"
        )
        rows = list(result)
        assert rows[0]["name"] == "Alicia"

    def test_set_property_various_types(self, populated_db):
        db = populated_db["db"]
        nid = populated_db["alice"].id

        db.set_node_property(nid, "active", True)
        db.set_node_property(nid, "score", 3.14)
        db.set_node_property(nid, "count", 42)
        db.set_node_property(nid, "bio", "Hello world")
        db.set_node_property(nid, "empty", None)

        node = db.get_node(nid)
        assert node is not None

    def test_set_property_list_value(self, populated_db):
        db = populated_db["db"]
        nid = populated_db["alice"].id
        db.set_node_property(nid, "tags", ["a", "b", "c"])
        result = db.execute(f"MATCH (n) WHERE id(n) = {nid} RETURN n.tags AS t")
        rows = list(result)
        assert len(rows[0]["t"]) == 3

    def test_set_property_map_value(self, populated_db):
        db = populated_db["db"]
        nid = populated_db["alice"].id
        db.set_node_property(nid, "meta", {"x": 1, "y": "two"})
        result = db.execute(f"MATCH (n) WHERE id(n) = {nid} RETURN n.meta AS m")
        rows = list(result)
        m = rows[0]["m"]
        assert m["x"] == 1
        assert m["y"] == "two"


class TestRemoveNodeProperty:
    """Test remove_node_property() method."""

    def test_remove_existing_property(self, populated_db):
        db = populated_db["db"]
        nid = populated_db["alice"].id
        removed = db.remove_node_property(nid, "age")
        assert removed is True
        result = db.execute(f"MATCH (n) WHERE id(n) = {nid} RETURN n.age AS a")
        rows = list(result)
        assert rows[0]["a"] is None

    def test_remove_nonexistent_property(self, populated_db):
        db = populated_db["db"]
        nid = populated_db["alice"].id
        removed = db.remove_node_property(nid, "nonexistent")
        assert removed is False


# ── Edge Property Manipulation ──────────────────────────────────────


class TestSetEdgeProperty:
    """Test set_edge_property() method."""

    def test_set_new_edge_property(self, populated_db):
        db = populated_db["db"]
        eid = populated_db["edge"].id
        db.set_edge_property(eid, "weight", 0.75)
        edge = db.get_edge(eid)
        assert edge is not None

    def test_overwrite_edge_property(self, populated_db):
        db = populated_db["db"]
        eid = populated_db["edge"].id
        db.set_edge_property(eid, "since", 2021)
        edge = db.get_edge(eid)
        assert edge is not None


class TestRemoveEdgeProperty:
    """Test remove_edge_property() method."""

    def test_remove_existing_edge_property(self, populated_db):
        db = populated_db["db"]
        eid = populated_db["edge"].id
        removed = db.remove_edge_property(eid, "since")
        assert removed is True

    def test_remove_nonexistent_edge_property(self, populated_db):
        db = populated_db["db"]
        eid = populated_db["edge"].id
        removed = db.remove_edge_property(eid, "nonexistent")
        assert removed is False


# ── Label Management ────────────────────────────────────────────────


class TestLabelManagement:
    """Test add_node_label, remove_node_label, get_node_labels."""

    def test_add_label(self, populated_db):
        db = populated_db["db"]
        nid = populated_db["alice"].id
        added = db.add_node_label(nid, "Manager")
        assert added is True
        labels = db.get_node_labels(nid)
        assert "Manager" in labels
        assert "Person" in labels

    def test_add_duplicate_label(self, populated_db):
        db = populated_db["db"]
        nid = populated_db["alice"].id
        added = db.add_node_label(nid, "Person")
        # Adding existing label returns False
        assert added is False

    def test_remove_label(self, populated_db):
        db = populated_db["db"]
        nid = populated_db["charlie"].id
        # Charlie has Person and Employee
        removed = db.remove_node_label(nid, "Employee")
        assert removed is True
        labels = db.get_node_labels(nid)
        assert "Employee" not in labels
        assert "Person" in labels

    def test_remove_nonexistent_label(self, populated_db):
        db = populated_db["db"]
        nid = populated_db["alice"].id
        removed = db.remove_node_label(nid, "NonExistent")
        assert removed is False

    def test_get_labels(self, populated_db):
        db = populated_db["db"]
        nid = populated_db["charlie"].id
        labels = db.get_node_labels(nid)
        assert "Person" in labels
        assert "Employee" in labels

    def test_get_labels_nonexistent_node(self, populated_db):
        db = populated_db["db"]
        labels = db.get_node_labels(999999)
        assert labels is None


# ── Property Index ──────────────────────────────────────────────────


class TestPropertyIndex:
    """Test property index CRUD and lookup."""

    def test_create_and_has_index(self, populated_db):
        db = populated_db["db"]
        db.create_property_index("name")
        assert db.has_property_index("name") is True

    def test_has_index_nonexistent(self, populated_db):
        db = populated_db["db"]
        assert db.has_property_index("nonexistent") is False

    def test_drop_index(self, populated_db):
        db = populated_db["db"]
        db.create_property_index("name")
        dropped = db.drop_property_index("name")
        assert dropped is True
        assert db.has_property_index("name") is False

    def test_drop_nonexistent_index(self, populated_db):
        db = populated_db["db"]
        dropped = db.drop_property_index("nonexistent")
        assert dropped is False

    def test_find_nodes_by_property(self, populated_db):
        db = populated_db["db"]
        db.create_property_index("name")
        ids = db.find_nodes_by_property("name", "Alice")
        assert len(ids) == 1
        node = db.get_node(ids[0])
        assert node is not None

    def test_find_nodes_by_property_no_match(self, populated_db):
        db = populated_db["db"]
        db.create_property_index("name")
        ids = db.find_nodes_by_property("name", "NonExistent")
        assert len(ids) == 0

    def test_find_nodes_by_int_property(self, populated_db):
        db = populated_db["db"]
        db.create_property_index("age")
        ids = db.find_nodes_by_property("age", 30)
        assert len(ids) == 1


# ── Batch Operations ────────────────────────────────────────────────


class TestGetNodesByLabel:
    """Test get_nodes_by_label() with pagination."""

    def test_basic_retrieval(self, populated_db):
        db = populated_db["db"]
        nodes = db.get_nodes_by_label("Person")
        assert len(nodes) == 3

    def test_with_limit(self, populated_db):
        db = populated_db["db"]
        nodes = db.get_nodes_by_label("Person", limit=2)
        assert len(nodes) == 2

    def test_with_offset(self, populated_db):
        db = populated_db["db"]
        all_nodes = db.get_nodes_by_label("Person")
        offset_nodes = db.get_nodes_by_label("Person", offset=1)
        assert len(offset_nodes) == len(all_nodes) - 1

    def test_with_limit_and_offset(self, populated_db):
        db = populated_db["db"]
        nodes = db.get_nodes_by_label("Person", limit=1, offset=1)
        assert len(nodes) == 1

    def test_nonexistent_label(self, populated_db):
        db = populated_db["db"]
        nodes = db.get_nodes_by_label("NonExistent")
        assert len(nodes) == 0


class TestGetPropertyBatch:
    """Test get_property_batch() method."""

    def test_basic_batch(self, populated_db):
        db = populated_db["db"]
        ids = [
            populated_db["alice"].id,
            populated_db["bob"].id,
            populated_db["charlie"].id,
        ]
        values = db.get_property_batch(ids, "name")
        assert len(values) == 3
        names = [v for v in values if v is not None]
        assert "Alice" in names
        assert "Bob" in names
        assert "Charlie" in names

    def test_batch_with_missing_property(self, populated_db):
        db = populated_db["db"]
        ids = [populated_db["alice"].id, populated_db["bob"].id]
        values = db.get_property_batch(ids, "nonexistent")
        assert all(v is None for v in values)

    def test_empty_batch(self, populated_db):
        db = populated_db["db"]
        values = db.get_property_batch([], "name")
        assert len(values) == 0


# ── Edge CRUD ───────────────────────────────────────────────────────


class TestEdgeCrud:
    """Test edge get/delete operations."""

    def test_get_edge(self, populated_db):
        db = populated_db["db"]
        edge = db.get_edge(populated_db["edge"].id)
        assert edge is not None
        assert edge.edge_type == "KNOWS"
        assert edge.source_id == populated_db["alice"].id
        assert edge.target_id == populated_db["bob"].id

    def test_get_nonexistent_edge(self, populated_db):
        db = populated_db["db"]
        edge = db.get_edge(999999)
        assert edge is None

    def test_delete_edge(self, populated_db):
        db = populated_db["db"]
        eid = populated_db["edge"].id
        deleted = db.delete_edge(eid)
        assert deleted is True
        assert db.get_edge(eid) is None

    def test_delete_nonexistent_edge(self, populated_db):
        db = populated_db["db"]
        deleted = db.delete_edge(999999)
        assert deleted is False

    def test_delete_node(self, populated_db):
        db = populated_db["db"]
        nid = populated_db["bob"].id
        deleted = db.delete_node(nid)
        assert deleted is True
        assert db.get_node(nid) is None

    def test_delete_nonexistent_node(self, populated_db):
        db = populated_db["db"]
        deleted = db.delete_node(999999)
        assert deleted is False


# ── Type Roundtrips ─────────────────────────────────────────────────


class TestTypeRoundtrips:
    """Test all Value type conversions through set/get property."""

    def test_bool_roundtrip(self, db):
        node = db.create_node(["T"], {"val": True})
        result = db.execute(f"MATCH (n) WHERE id(n) = {node.id} RETURN n.val AS v")
        assert list(result)[0]["v"] is True

    def test_int_roundtrip(self, db):
        node = db.create_node(["T"], {"val": 42})
        result = db.execute(f"MATCH (n) WHERE id(n) = {node.id} RETURN n.val AS v")
        assert list(result)[0]["v"] == 42

    def test_float_roundtrip(self, db):
        node = db.create_node(["T"], {"val": 3.14})
        result = db.execute(f"MATCH (n) WHERE id(n) = {node.id} RETURN n.val AS v")
        assert abs(list(result)[0]["v"] - 3.14) < 0.001

    def test_string_roundtrip(self, db):
        node = db.create_node(["T"], {"val": "hello"})
        result = db.execute(f"MATCH (n) WHERE id(n) = {node.id} RETURN n.val AS v")
        assert list(result)[0]["v"] == "hello"

    def test_null_roundtrip(self, db):
        node = db.create_node(["T"], {"val": None})
        result = db.execute(f"MATCH (n) WHERE id(n) = {node.id} RETURN n.val AS v")
        assert list(result)[0]["v"] is None

    def test_bytes_property_stored(self, db):
        data = b"\x00\x01\x02\xff"
        node = db.create_node(["T"])
        db.set_node_property(node.id, "val", data)
        result = db.execute(f"MATCH (n) WHERE id(n) = {node.id} RETURN n.val AS v")
        v = list(result)[0]["v"]
        # bytes is stored; may come back as bytes or list depending on pipeline
        assert v is not None

    def test_datetime_roundtrip(self, db):
        dt = datetime(2024, 6, 15, 12, 30, 0, tzinfo=timezone.utc)
        node = db.create_node(["T"])
        db.set_node_property(node.id, "val", dt)
        result = db.execute(f"MATCH (n) WHERE id(n) = {node.id} RETURN n.val AS v")
        v = list(result)[0]["v"]
        assert isinstance(v, datetime)

    def test_list_roundtrip(self, db):
        node = db.create_node(["T"], {"val": [1, "two", True]})
        result = db.execute(f"MATCH (n) WHERE id(n) = {node.id} RETURN n.val AS v")
        v = list(result)[0]["v"]
        assert len(v) == 3

    def test_map_roundtrip(self, db):
        node = db.create_node(["T"], {"val": {"a": 1, "b": "two"}})
        result = db.execute(f"MATCH (n) WHERE id(n) = {node.id} RETURN n.val AS v")
        v = list(result)[0]["v"]
        assert v["a"] == 1
        assert v["b"] == "two"

    def test_vector_roundtrip(self, db):
        node = db.create_node(["T"], {"val": [1.0, 2.0, 3.0]})
        result = db.execute(f"MATCH (n) WHERE id(n) = {node.id} RETURN n.val AS v")
        v = list(result)[0]["v"]
        assert len(v) == 3
        assert abs(v[0] - 1.0) < 0.01

    def test_empty_string(self, db):
        node = db.create_node(["T"], {"val": ""})
        result = db.execute(f"MATCH (n) WHERE id(n) = {node.id} RETURN n.val AS v")
        assert list(result)[0]["v"] == ""

    def test_large_int(self, db):
        big = 2**53 - 1  # max safe integer
        node = db.create_node(["T"], {"val": big})
        result = db.execute(f"MATCH (n) WHERE id(n) = {node.id} RETURN n.val AS v")
        assert list(result)[0]["v"] == big

    def test_negative_float(self, db):
        node = db.create_node(["T"], {"val": -1.5e10})
        result = db.execute(f"MATCH (n) WHERE id(n) = {node.id} RETURN n.val AS v")
        assert abs(list(result)[0]["v"] - (-1.5e10)) < 1.0


# ── Error Handling ──────────────────────────────────────────────────


class TestErrorHandling:
    """Test error conditions raise appropriate exceptions."""

    def test_invalid_query_syntax(self, db):
        with pytest.raises(Exception):
            db.execute("THIS IS NOT VALID GQL")

    def test_execute_on_closed_in_memory_db(self, db):
        db.close()
        # In-memory DB close is a no-op; operations still work
        result = db.execute("MATCH (n) RETURN n")
        assert list(result) == []

    def test_get_node_nonexistent(self, db):
        node = db.get_node(999999)
        assert node is None

    def test_set_property_nonexistent_node_silent(self, db):
        # Setting property on nonexistent node succeeds silently
        db.set_node_property(999999, "key", "value")

    def test_set_property_nonexistent_edge_silent(self, db):
        # Setting property on nonexistent edge succeeds silently
        db.set_edge_property(999999, "key", "value")

    def test_double_close(self, db):
        db.close()
        # Second close should not raise
        db.close()
