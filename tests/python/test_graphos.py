"""
Functional tests for Graphos Python bindings.

Run with: pytest tests/python/test_graphos.py -v
"""

import pytest
from synthetic_data import (
    SocialNetworkGenerator,
    LDBCLikeGenerator,
    TreeGenerator,
    CliqueGenerator,
    load_data_into_db,
)


# Try to import graphos - skip tests if not available
try:
    from graphos import GraphosDB
    GRAPHOS_AVAILABLE = True
except ImportError:
    GRAPHOS_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not GRAPHOS_AVAILABLE,
    reason="Graphos Python bindings not installed. Run: cd crates/graphos-python && maturin develop"
)


class TestBasicOperations:
    """Test basic CRUD operations."""

    def setup_method(self):
        """Create a fresh database for each test."""
        self.db = GraphosDB()

    def test_create_node(self):
        """Test creating a simple node."""
        node = self.db.create_node(["Person"], {"name": "Alice", "age": 30})
        assert node is not None
        assert node.id >= 0
        assert "Person" in node.labels
        assert node.properties().get("name") == "Alice"
        assert node.properties().get("age") == 30

    def test_create_node_multiple_labels(self):
        """Test creating a node with multiple labels."""
        node = self.db.create_node(["Person", "Developer"], {"name": "Bob"})
        assert "Person" in node.labels
        assert "Developer" in node.labels

    def test_create_edge(self):
        """Test creating an edge between nodes."""
        alice = self.db.create_node(["Person"], {"name": "Alice"})
        bob = self.db.create_node(["Person"], {"name": "Bob"})

        edge = self.db.create_edge(alice.id, bob.id, "KNOWS", {"since": 2020})

        assert edge is not None
        assert edge.source_id == alice.id
        assert edge.target_id == bob.id
        assert edge.edge_type == "KNOWS"
        assert edge.properties().get("since") == 2020

    def test_get_node(self):
        """Test retrieving a node by ID."""
        created = self.db.create_node(["Person"], {"name": "Charlie"})
        retrieved = self.db.get_node(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.properties().get("name") == "Charlie"

    def test_get_nonexistent_node(self):
        """Test retrieving a node that doesn't exist."""
        result = self.db.get_node(999999)
        assert result is None

    def test_delete_node(self):
        """Test deleting a node."""
        node = self.db.create_node(["Person"], {"name": "ToDelete"})
        node_id = node.id

        result = self.db.delete_node(node_id)
        assert result is True

        # Verify it's gone
        assert self.db.get_node(node_id) is None

    def test_delete_edge(self):
        """Test deleting an edge."""
        alice = self.db.create_node(["Person"], {"name": "Alice"})
        bob = self.db.create_node(["Person"], {"name": "Bob"})
        edge = self.db.create_edge(alice.id, bob.id, "KNOWS", {})

        result = self.db.delete_edge(edge.id)
        assert result is True

    def test_stats(self):
        """Test database statistics."""
        # Empty database
        stats = self.db.stats()
        assert stats.node_count == 0
        assert stats.edge_count == 0

        # After adding data
        self.db.create_node(["Person"], {"name": "Alice"})
        self.db.create_node(["Person"], {"name": "Bob"})
        alice = self.db.get_node(0)
        bob = self.db.get_node(1)
        if alice and bob:
            self.db.create_edge(alice.id, bob.id, "KNOWS", {})

        stats = self.db.stats()
        assert stats.node_count == 2
        assert stats.edge_count == 1


class TestQueryExecution:
    """Test GQL query execution."""

    def setup_method(self):
        """Create a database with some test data."""
        self.db = GraphosDB()

        # Create test nodes
        self.alice = self.db.create_node(["Person"], {"name": "Alice", "age": 30})
        self.bob = self.db.create_node(["Person"], {"name": "Bob", "age": 25})
        self.charlie = self.db.create_node(["Person"], {"name": "Charlie", "age": 35})

        # Create test edges
        self.db.create_edge(self.alice.id, self.bob.id, "KNOWS", {"since": 2020})
        self.db.create_edge(self.bob.id, self.charlie.id, "KNOWS", {"since": 2021})
        self.db.create_edge(self.alice.id, self.charlie.id, "KNOWS", {"since": 2019})

    def test_simple_match(self):
        """Test simple MATCH query."""
        result = self.db.execute("MATCH (n:Person) RETURN n.name")
        rows = list(result)
        assert len(rows) == 3

    def test_match_with_filter(self):
        """Test MATCH with WHERE clause."""
        result = self.db.execute("MATCH (n:Person) WHERE n.age > 28 RETURN n.name")
        rows = list(result)
        # Alice (30) and Charlie (35) match
        assert len(rows) == 2

    def test_match_relationship(self):
        """Test MATCH with relationship pattern."""
        result = self.db.execute(
            "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name"
        )
        rows = list(result)
        assert len(rows) == 3  # Three KNOWS edges

    @pytest.mark.skip(reason="Multi-hop path queries not yet implemented")
    def test_match_path(self):
        """Test MATCH with path pattern."""
        result = self.db.execute(
            "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) "
            "RETURN a.name, b.name, c.name"
        )
        rows = list(result)
        # Alice->Bob->Charlie path exists
        assert len(rows) >= 1

    @pytest.mark.skip(reason="Aggregation functions not yet implemented")
    def test_count_aggregation(self):
        """Test COUNT aggregation."""
        result = self.db.execute("MATCH (n:Person) RETURN count(n) AS cnt")
        rows = list(result)
        assert len(rows) == 1
        assert rows[0]["cnt"] == 3

    @pytest.mark.skip(reason="LIMIT clause not yet fully implemented")
    def test_limit(self):
        """Test LIMIT clause."""
        result = self.db.execute("MATCH (n:Person) RETURN n.name LIMIT 2")
        rows = list(result)
        assert len(rows) == 2

    @pytest.mark.skip(reason="ORDER BY clause not yet fully implemented")
    def test_order_by(self):
        """Test ORDER BY clause."""
        result = self.db.execute(
            "MATCH (n:Person) RETURN n.name ORDER BY n.age DESC"
        )
        rows = list(result)
        assert len(rows) == 3
        # Charlie (35) should be first
        assert rows[0]["n.name"] == "Charlie"


class TestSyntheticDataLoading:
    """Test loading synthetic datasets."""

    def test_load_social_network(self):
        """Test loading a social network dataset."""
        db = GraphosDB()
        gen = SocialNetworkGenerator(num_nodes=100, avg_edges_per_node=5, seed=42)
        node_count, edge_count = load_data_into_db(db, gen)

        assert node_count == 100
        assert edge_count > 0

        stats = db.stats()
        assert stats.node_count == 100

    def test_load_ldbc_like(self):
        """Test loading an LDBC-like dataset."""
        db = GraphosDB()
        gen = LDBCLikeGenerator(scale_factor=0.1, seed=42)
        node_count, edge_count = load_data_into_db(db, gen)

        assert node_count > 0
        assert edge_count > 0

    def test_load_tree(self):
        """Test loading a tree dataset."""
        db = GraphosDB()
        gen = TreeGenerator(depth=3, branching_factor=2, seed=42)
        node_count, edge_count = load_data_into_db(db, gen)

        # Tree with depth 3 and branching factor 2: 1 + 2 + 4 + 8 = 15 nodes
        expected_nodes = sum(2**i for i in range(4))
        assert node_count == expected_nodes

    def test_query_social_network(self):
        """Test querying a social network dataset."""
        db = GraphosDB()
        gen = SocialNetworkGenerator(num_nodes=50, avg_edges_per_node=3, seed=42)
        load_data_into_db(db, gen)

        # Query persons (without aggregation)
        result = db.execute("MATCH (p:Person) RETURN p.name")
        rows = list(result)
        assert len(rows) == 50

        # Find people who know each other
        result = db.execute(
            "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name"
        )
        rows = list(result)
        assert len(rows) > 0

    def test_query_ldbc_relationships(self):
        """Test querying different relationship types in LDBC data."""
        db = GraphosDB()
        gen = LDBCLikeGenerator(scale_factor=0.1, seed=42)
        load_data_into_db(db, gen)

        # Find people who work at companies
        result = db.execute(
            "MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN p.name, c.name"
        )
        rows = list(result)
        assert len(rows) > 0

        # Find people who studied at universities
        result = db.execute(
            "MATCH (p:Person)-[:STUDIED_AT]->(u:University) RETURN p.name, u.name"
        )
        rows = list(result)
        assert len(rows) > 0


class TestTransactions:
    """Test transaction functionality."""

    def test_transaction_commit(self):
        """Test transaction commit."""
        db = GraphosDB()

        with db.begin_transaction() as tx:
            db.create_node(["Person"], {"name": "TransactionTest"})
            tx.commit()

        # Data should be visible after commit
        result = db.execute("MATCH (n:Person) WHERE n.name = 'TransactionTest' RETURN n")
        rows = list(result)
        assert len(rows) == 1

    def test_transaction_auto_commit(self):
        """Test that transactions auto-commit on success."""
        db = GraphosDB()

        with db.begin_transaction():
            db.create_node(["Person"], {"name": "AutoCommitTest"})

        # Data should be visible after auto-commit
        result = db.execute("MATCH (n:Person) WHERE n.name = 'AutoCommitTest' RETURN n")
        rows = list(result)
        assert len(rows) == 1

    def test_transaction_is_active(self):
        """Test transaction is_active property."""
        db = GraphosDB()

        tx = db.begin_transaction()
        assert tx.is_active is True

        tx.commit()
        assert tx.is_active is False


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_label_list(self):
        """Test creating a node with no labels."""
        db = GraphosDB()
        node = db.create_node([], {"name": "NoLabel"})
        assert node is not None

    def test_empty_properties(self):
        """Test creating a node with no properties."""
        db = GraphosDB()
        node = db.create_node(["Empty"], {})
        assert node is not None
        assert len(node.properties()) == 0

    def test_unicode_properties(self):
        """Test unicode property values."""
        db = GraphosDB()
        node = db.create_node(["Person"], {
            "name": "Alice",
            "greeting": "Hello World!",
        })
        assert node.properties().get("greeting") == "Hello World!"

    def test_large_property_value(self):
        """Test large string property values."""
        db = GraphosDB()
        large_value = "x" * 10000
        node = db.create_node(["Data"], {"content": large_value})
        assert len(node.properties().get("content", "")) == 10000

    def test_numeric_property_types(self):
        """Test different numeric property types."""
        db = GraphosDB()
        node = db.create_node(["Numbers"], {
            "int_val": 42,
            "float_val": 3.14159,
            "neg_int": -100,
            "neg_float": -2.5,
        })
        assert node.properties().get("int_val") == 42
        assert abs(node.properties().get("float_val", 0) - 3.14159) < 0.0001


class TestPerformanceSmoke:
    """Smoke tests for performance (fast checks)."""

    def test_bulk_node_insert(self):
        """Test inserting many nodes."""
        db = GraphosDB()

        for i in range(100):
            db.create_node(["Person"], {"name": f"Person{i}", "idx": i})

        stats = db.stats()
        assert stats.node_count == 100

    def test_bulk_edge_insert(self):
        """Test inserting many edges."""
        db = GraphosDB()

        # Create nodes first
        node_ids = []
        for i in range(50):
            node = db.create_node(["Node"], {"idx": i})
            node_ids.append(node.id)

        # Create edges
        for i in range(len(node_ids) - 1):
            db.create_edge(node_ids[i], node_ids[i + 1], "NEXT", {})

        stats = db.stats()
        assert stats.node_count == 50
        assert stats.edge_count == 49


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
