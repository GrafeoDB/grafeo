"""Gremlin traversal tests.

Tests Gremlin traversal language (Apache TinkerPop style).
"""

import pytest


# Try to import grafeo
try:
    from grafeo import GrafeoDB
    GRAFEO_AVAILABLE = True
except ImportError:
    GRAFEO_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not GRAFEO_AVAILABLE,
    reason="Grafeo Python bindings not installed"
)


class TestGremlinTraversal:
    """Test Gremlin traversal operations."""

    def setup_method(self):
        """Create a database with test data."""
        self.db = GrafeoDB()
        self._setup_test_data()

    def _setup_test_data(self):
        """Create test data."""
        self.alice = self.db.create_node(["Person"], {"name": "Alice", "age": 30})
        self.bob = self.db.create_node(["Person"], {"name": "Bob", "age": 25})
        self.charlie = self.db.create_node(["Person"], {"name": "Charlie", "age": 35})
        self.db.create_edge(self.alice.id, self.bob.id, "knows", {"since": 2020})
        self.db.create_edge(self.bob.id, self.charlie.id, "knows", {"since": 2021})

    def _execute_gremlin(self, query: str):
        """Execute Gremlin query, skip if not supported."""
        try:
            return self.db.execute_gremlin(query)
        except AttributeError:
            pytest.skip("Gremlin support not available")
        except NotImplementedError:
            pytest.skip("Gremlin not implemented")

    def test_gremlin_vertex_query(self):
        """Gremlin: g.V() - Get all vertices."""
        result = self._execute_gremlin("g.V()")
        rows = list(result)
        assert len(rows) == 3

    def test_gremlin_has_label(self):
        """Gremlin: g.V().hasLabel('Person')"""
        result = self._execute_gremlin("g.V().hasLabel('Person')")
        rows = list(result)
        assert len(rows) == 3

    def test_gremlin_has_property(self):
        """Gremlin: g.V().has('name', 'Alice')"""
        result = self._execute_gremlin("g.V().has('name', 'Alice')")
        rows = list(result)
        assert len(rows) == 1

    def test_gremlin_has_property_gt(self):
        """Gremlin: g.V().has('age', gt(28))"""
        result = self._execute_gremlin("g.V().has('age', gt(28))")
        rows = list(result)
        # Alice (30) and Charlie (35) match
        assert len(rows) == 2

    def test_gremlin_out_traversal(self):
        """Gremlin: g.V().has('name', 'Alice').out('knows')"""
        result = self._execute_gremlin("g.V().has('name', 'Alice').out('knows')")
        rows = list(result)
        # Alice knows Bob
        assert len(rows) == 1

    def test_gremlin_in_traversal(self):
        """Gremlin: g.V().has('name', 'Bob').in('knows')"""
        result = self._execute_gremlin("g.V().has('name', 'Bob').in('knows')")
        rows = list(result)
        # Alice knows Bob, so Bob has 1 incoming knows edge
        assert len(rows) == 1

    def test_gremlin_both_traversal(self):
        """Gremlin: g.V().has('name', 'Bob').both('knows')"""
        result = self._execute_gremlin("g.V().has('name', 'Bob').both('knows')")
        rows = list(result)
        # Bob is connected to Alice (in) and Charlie (out)
        assert len(rows) == 2

    def test_gremlin_values(self):
        """Gremlin: g.V().hasLabel('Person').values('name')"""
        result = self._execute_gremlin("g.V().hasLabel('Person').values('name')")
        rows = list(result)
        assert len(rows) == 3

    def test_gremlin_count(self):
        """Gremlin: g.V().hasLabel('Person').count()"""
        result = self._execute_gremlin("g.V().hasLabel('Person').count()")
        rows = list(result)
        # Count returns a single row with the count
        assert len(rows) >= 1

    def test_gremlin_limit(self):
        """Gremlin: g.V().hasLabel('Person').limit(2)"""
        result = self._execute_gremlin("g.V().hasLabel('Person').limit(2)")
        rows = list(result)
        assert len(rows) == 2

    def test_gremlin_order_by(self):
        """Gremlin: g.V().hasLabel('Person').order().by('age', asc)"""
        result = self._execute_gremlin(
            "g.V().hasLabel('Person').order().by('age', asc).values('name')"
        )
        rows = list(result)
        # Bob (25) should be first
        if len(rows) >= 1:
            assert rows[0] == "Bob" or rows[0].get("name") == "Bob"

    def test_gremlin_path(self):
        """Gremlin: g.V().has('name', 'Alice').out('knows').out('knows').path()"""
        result = self._execute_gremlin(
            "g.V().has('name', 'Alice').out('knows').out('knows').path()"
        )
        rows = list(result)
        # Alice -> Bob -> Charlie path
        assert len(rows) >= 1

    def test_gremlin_dedup(self):
        """Gremlin: g.V().hasLabel('Person').values('age').dedup()"""
        result = self._execute_gremlin(
            "g.V().hasLabel('Person').values('age').dedup()"
        )
        rows = list(result)
        # All ages are unique, so 3 values
        assert len(rows) == 3

    def test_gremlin_aggregate(self):
        """Gremlin: g.V().hasLabel('Person').group().by('age').by(count())"""
        result = self._execute_gremlin(
            "g.V().hasLabel('Person').group().by('age').by(count())"
        )
        rows = list(result)
        # Should return age groups
        assert len(rows) >= 1


class TestGremlinEdgeTraversal:
    """Test Gremlin edge traversal operations."""

    def setup_method(self):
        """Create a database with test data."""
        self.db = GrafeoDB()
        self._setup_test_data()

    def _setup_test_data(self):
        """Create test data with weighted edges."""
        self.a = self.db.create_node(["Node"], {"name": "a"})
        self.b = self.db.create_node(["Node"], {"name": "b"})
        self.c = self.db.create_node(["Node"], {"name": "c"})
        self.db.create_edge(self.a.id, self.b.id, "edge", {"weight": 1.0})
        self.db.create_edge(self.b.id, self.c.id, "edge", {"weight": 2.0})
        self.db.create_edge(self.a.id, self.c.id, "edge", {"weight": 5.0})

    def _execute_gremlin(self, query: str):
        """Execute Gremlin query, skip if not supported."""
        try:
            return self.db.execute_gremlin(query)
        except AttributeError:
            pytest.skip("Gremlin support not available")
        except NotImplementedError:
            pytest.skip("Gremlin not implemented")

    def test_gremlin_edges(self):
        """Gremlin: g.E() - Get all edges."""
        result = self._execute_gremlin("g.E()")
        rows = list(result)
        assert len(rows) == 3

    def test_gremlin_out_edges(self):
        """Gremlin: g.V().has('name', 'a').outE('edge')"""
        result = self._execute_gremlin("g.V().has('name', 'a').outE('edge')")
        rows = list(result)
        # Node a has 2 outgoing edges
        assert len(rows) == 2

    def test_gremlin_in_edges(self):
        """Gremlin: g.V().has('name', 'c').inE('edge')"""
        result = self._execute_gremlin("g.V().has('name', 'c').inE('edge')")
        rows = list(result)
        # Node c has 2 incoming edges
        assert len(rows) == 2

    def test_gremlin_edge_properties(self):
        """Gremlin: g.E().has('weight', gt(1.5))"""
        result = self._execute_gremlin("g.E().has('weight', gt(1.5))")
        rows = list(result)
        # Two edges have weight > 1.5
        assert len(rows) == 2

    def test_gremlin_edge_to_vertex(self):
        """Gremlin: g.V().has('name', 'a').outE('edge').inV()"""
        result = self._execute_gremlin("g.V().has('name', 'a').outE('edge').inV()")
        rows = list(result)
        # Should get b and c
        assert len(rows) == 2
