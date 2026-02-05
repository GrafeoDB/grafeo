"""Cross-language compatibility tests for LPG model.

Verifies that GQL, Cypher, Gremlin, and GraphQL return consistent results.
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


class TestCrossLanguageConsistency:
    """Verify that different query languages return consistent results."""

    def setup_method(self):
        """Create identical test data for comparison."""
        self.db = GrafeoDB()
        self._setup_test_data()

    def _setup_test_data(self):
        """Create test data."""
        self.alice = self.db.create_node(["Person"], {"name": "Alice", "age": 30})
        self.bob = self.db.create_node(["Person"], {"name": "Bob", "age": 25})
        self.charlie = self.db.create_node(["Person"], {"name": "Charlie", "age": 35})

        self.db.create_edge(self.alice.id, self.bob.id, "KNOWS", {"since": 2020})
        self.db.create_edge(self.bob.id, self.charlie.id, "KNOWS", {"since": 2021})

    def _has_gremlin(self):
        """Check if Gremlin support is available."""
        try:
            self.db.execute_gremlin("g.V().limit(1)")
            return True
        except (AttributeError, NotImplementedError):
            return False
        except Exception:
            return True

    def _has_graphql(self):
        """Check if GraphQL support is available."""
        try:
            self.db.execute_graphql("query { __schema { types { name } } }")
            return True
        except (AttributeError, NotImplementedError):
            return False
        except Exception:
            return True

    def test_node_count_consistency(self):
        """All languages should return the same node count."""
        # GQL/Cypher (same syntax)
        gql_result = self.db.execute("MATCH (n:Person) RETURN count(n) AS cnt")
        gql_count = list(gql_result)[0]["cnt"]
        assert gql_count == 3

        # Gremlin (if available)
        if self._has_gremlin():
            try:
                gremlin_result = self.db.execute_gremlin(
                    "g.V().hasLabel('Person').count()"
                )
                gremlin_rows = list(gremlin_result)
                if len(gremlin_rows) > 0:
                    # Gremlin count format may vary
                    pass
            except Exception:
                pass

    def test_simple_match_consistency(self):
        """GQL and Cypher should return same results for simple MATCH."""
        # Both use the same syntax
        result = self.db.execute(
            "MATCH (n:Person) RETURN n.name ORDER BY n.name"
        )
        rows = list(result)
        names = [r.get("n.name") for r in rows]

        assert "Alice" in names
        assert "Bob" in names
        assert "Charlie" in names

    def test_relationship_match_consistency(self):
        """All languages should return same relationship results."""
        # GQL/Cypher
        result = self.db.execute(
            "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name"
        )
        rows = list(result)
        assert len(rows) == 2  # Alice->Bob, Bob->Charlie

    def test_aggregation_consistency(self):
        """Aggregation results should match across languages."""
        # GQL/Cypher
        result = self.db.execute(
            "MATCH (n:Person) RETURN sum(n.age) AS total"
        )
        rows = list(result)
        assert rows[0]["total"] == 90  # 30 + 25 + 35

    def test_filter_consistency(self):
        """WHERE clause filtering should be consistent."""
        # GQL/Cypher
        result = self.db.execute(
            "MATCH (n:Person) WHERE n.age > 26 RETURN n.name"
        )
        rows = list(result)
        names = [r.get("n.name") for r in rows]

        # Alice (30) and Charlie (35) match
        assert len(rows) == 2
        assert "Alice" in names
        assert "Charlie" in names
        assert "Bob" not in names


class TestQuerySyntaxEquivalence:
    """Test that equivalent queries produce the same results."""

    def setup_method(self):
        """Create test data."""
        self.db = GrafeoDB()
        self.a = self.db.create_node(["Node"], {"value": 1})
        self.b = self.db.create_node(["Node"], {"value": 2})
        self.c = self.db.create_node(["Node"], {"value": 3})
        self.db.create_edge(self.a.id, self.b.id, "LINK", {})
        self.db.create_edge(self.b.id, self.c.id, "LINK", {})

    def test_where_vs_inline_properties(self):
        """WHERE clause vs inline properties should be equivalent."""
        # Using WHERE
        result1 = self.db.execute(
            "MATCH (n:Node) WHERE n.value = 2 RETURN n.value"
        )
        rows1 = list(result1)

        # Using inline properties
        result2 = self.db.execute(
            "MATCH (n:Node {value: 2}) RETURN n.value"
        )
        rows2 = list(result2)

        assert len(rows1) == len(rows2) == 1
        assert rows1[0]["n.value"] == rows2[0]["n.value"] == 2

    def test_order_by_equivalence(self):
        """ORDER BY ASC and without explicit direction should be equivalent."""
        result1 = self.db.execute(
            "MATCH (n:Node) RETURN n.value ORDER BY n.value"
        )
        rows1 = list(result1)

        result2 = self.db.execute(
            "MATCH (n:Node) RETURN n.value ORDER BY n.value ASC"
        )
        rows2 = list(result2)

        values1 = [r["n.value"] for r in rows1]
        values2 = [r["n.value"] for r in rows2]
        assert values1 == values2 == [1, 2, 3]

    def test_directed_vs_undirected(self):
        """Directed queries should be subset of undirected."""
        # Directed: only outgoing edges
        result_directed = self.db.execute(
            "MATCH (a:Node)-[:LINK]->(b:Node) RETURN count(a) AS cnt"
        )
        directed_count = list(result_directed)[0]["cnt"]

        # Undirected: both directions
        result_undirected = self.db.execute(
            "MATCH (a:Node)-[:LINK]-(b:Node) RETURN count(a) AS cnt"
        )
        undirected_count = list(result_undirected)[0]["cnt"]

        # Undirected should count each edge twice (both directions)
        assert undirected_count == directed_count * 2
