"""GQL change data capture (CDC) integration tests."""

import pytest

try:
    from grafeo import GrafeoDB

    GRAFEO_AVAILABLE = True
except ImportError:
    GRAFEO_AVAILABLE = False


@pytest.fixture
def db():
    if not GRAFEO_AVAILABLE:
        pytest.skip("grafeo not installed")
    return GrafeoDB(cdc=True)


class TestCDC:
    def test_node_history_after_create(self, db):
        node = db.create_node(["Person"], {"name": "Alix"})
        history = db.node_history(node.id)
        assert len(history) >= 1

    def test_node_history_after_update(self, db):
        node = db.create_node(["Person"], {"name": "Alix"})
        db.set_node_property(node.id, "age", 30)
        history = db.node_history(node.id)
        assert len(history) >= 2

    def test_edge_history_after_create(self, db):
        a = db.create_node(["N"])
        b = db.create_node(["N"])
        edge = db.create_edge(a.id, b.id, "R")
        history = db.edge_history(edge.id)
        assert len(history) >= 1

    def test_changes_between_epochs(self, db):
        db.create_node(["Person"], {"name": "Alix"})
        db.create_node(["Person"], {"name": "Gus"})
        # Get all changes from epoch 0 to a large number
        changes = db.changes_between(0, 1000)
        assert len(changes) >= 2

    def test_empty_history(self, db):
        # Node ID that doesn't exist
        history = db.node_history(9999)
        assert len(history) == 0

    def test_transaction_with_cdc_enabled(self, db):
        """Per-transaction CDC override tracks changes when enabled."""
        with db.begin_transaction_with_cdc(True) as tx:
            tx.execute("INSERT (:Person {name: 'Gus'})")
            tx.commit()
        result = db.execute("MATCH (n:Person {name: 'Gus'}) RETURN n")
        rows = list(result)
        assert len(rows) == 1
        node_id = rows[0]["n"]["_id"]
        history = db.node_history(node_id)
        assert len(history) >= 1

    def test_transaction_with_cdc_disabled(self, db):
        """Per-transaction CDC override suppresses tracking when disabled."""
        # Insert with CDC explicitly disabled for this transaction
        with db.begin_transaction_with_cdc(False) as tx:
            tx.execute("INSERT (:Person {name: 'Vincent'})")
            tx.commit()
        result = db.execute("MATCH (n:Person {name: 'Vincent'}) RETURN n")
        rows = list(result)
        assert len(rows) == 1
        node_id = rows[0]["n"]["_id"]
        history = db.node_history(node_id)
        assert len(history) == 0


class TestCDCEventContext:
    """Events say what was created, changed or deleted."""

    def test_events_describe_nodes_and_edges(self, db):
        alix = db.create_node(["Graph", "File"], {"id": "a"})
        gus = db.create_node(["Graph", "Concept"], {"id": "b"})
        edge = db.create_edge(alix.id, gus.id, "REFERENCES", {"id": "e1"})
        db.remove_node_label(alix.id, "File")
        db.delete_edge(edge.id)
        db.delete_node(gus.id)

        events = db.changes_between(0, db.current_epoch() + 1)

        def find(entity_type, kind, entity_id):
            return next(
                e
                for e in events
                if e["entity_type"] == entity_type
                and e["kind"] == kind
                and e["entity_id"] == entity_id
            )

        endpoints = ("REFERENCES", alix.id, gus.id)
        created = find("node", "create", gus.id)
        assert sorted(created["labels"]) == ["Concept", "Graph"]
        assert created["edge_type"] is None and created["src_id"] is None

        edge_created = find("edge", "create", edge.id)
        assert (
            edge_created["edge_type"],
            edge_created["src_id"],
            edge_created["dst_id"],
        ) == endpoints
        assert edge_created["labels"] is None

        relabeled = find("node", "update", alix.id)
        assert sorted(relabeled["before_labels"]) == ["File", "Graph"]
        assert relabeled["labels"] == ["Graph"]

        edge_deleted = find("edge", "delete", edge.id)
        assert (
            edge_deleted["edge_type"],
            edge_deleted["src_id"],
            edge_deleted["dst_id"],
        ) == endpoints
        assert edge_deleted["before"] == {"id": "e1"}

        deleted = find("node", "delete", gus.id)
        assert sorted(deleted["labels"]) == ["Concept", "Graph"]
        assert deleted["before"] == {"id": "b"}
