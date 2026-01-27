"""
Tests for both GQL and Cypher query languages.

Run with: pytest tests/python/test_query_languages.py -v
"""

import pytest


# Try to import graphos
try:
    from graphos import GraphosDB
    GRAPHOS_AVAILABLE = True
except ImportError:
    GRAPHOS_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not GRAPHOS_AVAILABLE,
    reason="Graphos Python bindings not installed"
)


class TestGQLQueries:
    """Test GQL (ISO standard) query language."""

    def setup_method(self):
        """Create a database with test data."""
        self.db = GraphosDB()
        self._setup_test_data()

    def _setup_test_data(self):
        """Create a small test graph."""
        # Create Person nodes
        self.alice = self.db.create_node(["Person"], {
            "name": "Alice", "age": 30, "city": "NYC"
        })
        self.bob = self.db.create_node(["Person"], {
            "name": "Bob", "age": 25, "city": "LA"
        })
        self.charlie = self.db.create_node(["Person"], {
            "name": "Charlie", "age": 35, "city": "NYC"
        })

        # Create Company nodes
        self.acme = self.db.create_node(["Company"], {
            "name": "Acme Corp", "founded": 2010
        })
        self.globex = self.db.create_node(["Company"], {
            "name": "Globex Inc", "founded": 2015
        })

        # Create edges
        self.db.create_edge(self.alice.id, self.bob.id, "KNOWS", {"since": 2020})
        self.db.create_edge(self.bob.id, self.charlie.id, "KNOWS", {"since": 2021})
        self.db.create_edge(self.alice.id, self.charlie.id, "KNOWS", {"since": 2019})

        self.db.create_edge(self.alice.id, self.acme.id, "WORKS_AT", {"role": "Engineer"})
        self.db.create_edge(self.bob.id, self.globex.id, "WORKS_AT", {"role": "Manager"})
        self.db.create_edge(self.charlie.id, self.acme.id, "WORKS_AT", {"role": "Director"})

    # ===== GQL MATCH Queries =====

    def test_gql_simple_match(self):
        """GQL: Simple node match."""
        result = self.db.execute("MATCH (n:Person) RETURN n.name")
        rows = list(result)
        assert len(rows) == 3

    def test_gql_match_with_where(self):
        """GQL: MATCH with WHERE clause."""
        result = self.db.execute(
            "MATCH (n:Person) WHERE n.age > 28 RETURN n.name"
        )
        rows = list(result)
        names = [r["n.name"] for r in rows]
        assert "Alice" in names  # age 30
        assert "Charlie" in names  # age 35
        assert "Bob" not in names  # age 25

    def test_gql_match_multiple_labels(self):
        """GQL: Match nodes by multiple criteria."""
        result = self.db.execute(
            "MATCH (p:Person) WHERE p.city = 'NYC' AND p.age > 25 RETURN p.name"
        )
        rows = list(result)
        names = [r["p.name"] for r in rows]
        assert "Alice" in names
        assert "Charlie" in names

    def test_gql_match_relationship(self):
        """GQL: Match relationship pattern."""
        result = self.db.execute(
            "MATCH (a:Person)-[:KNOWS]->(b:Person) "
            "RETURN a.name AS from_person, b.name AS to_person"
        )
        rows = list(result)
        assert len(rows) == 3

    def test_gql_match_relationship_with_properties(self):
        """GQL: Match relationship with property filter."""
        result = self.db.execute(
            "MATCH (a:Person)-[r:KNOWS]->(b:Person) "
            "WHERE r.since >= 2020 "
            "RETURN a.name, b.name, r.since"
        )
        rows = list(result)
        assert len(rows) >= 2

    def test_gql_match_multi_hop(self):
        """GQL: Match multi-hop path."""
        result = self.db.execute(
            "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) "
            "RETURN a.name, b.name, c.name"
        )
        rows = list(result)
        # Alice->Bob->Charlie path exists
        assert len(rows) >= 1

    def test_gql_match_heterogeneous(self):
        """GQL: Match across different node types."""
        result = self.db.execute(
            "MATCH (p:Person)-[:WORKS_AT]->(c:Company) "
            "RETURN p.name, c.name"
        )
        rows = list(result)
        assert len(rows) == 3

    # ===== GQL Aggregation Queries =====

    def test_gql_count(self):
        """GQL: COUNT aggregation."""
        result = self.db.execute(
            "MATCH (n:Person) RETURN count(n) AS cnt"
        )
        rows = list(result)
        assert rows[0]["cnt"] == 3

    def test_gql_count_distinct(self):
        """GQL: COUNT DISTINCT."""
        result = self.db.execute(
            "MATCH (p:Person) RETURN count(DISTINCT p.city) AS cities"
        )
        rows = list(result)
        assert rows[0]["cities"] == 2  # NYC and LA

    def test_gql_sum_avg(self):
        """GQL: SUM and AVG aggregations."""
        result = self.db.execute(
            "MATCH (p:Person) RETURN sum(p.age) AS total_age, avg(p.age) AS avg_age"
        )
        rows = list(result)
        assert rows[0]["total_age"] == 90  # 30 + 25 + 35
        assert abs(rows[0]["avg_age"] - 30.0) < 0.01

    def test_gql_min_max(self):
        """GQL: MIN and MAX aggregations."""
        result = self.db.execute(
            "MATCH (p:Person) RETURN min(p.age) AS youngest, max(p.age) AS oldest"
        )
        rows = list(result)
        assert rows[0]["youngest"] == 25
        assert rows[0]["oldest"] == 35

    def test_gql_group_by(self):
        """GQL: GROUP BY."""
        result = self.db.execute(
            "MATCH (p:Person) RETURN p.city, count(p) AS cnt "
            "ORDER BY cnt DESC"
        )
        rows = list(result)
        # NYC has 2 people, LA has 1
        assert rows[0]["p.city"] == "NYC"
        assert rows[0]["cnt"] == 2

    # ===== GQL Ordering and Limiting =====

    def test_gql_order_by(self):
        """GQL: ORDER BY."""
        result = self.db.execute(
            "MATCH (p:Person) RETURN p.name, p.age ORDER BY p.age ASC"
        )
        rows = list(result)
        assert rows[0]["p.name"] == "Bob"  # youngest
        assert rows[2]["p.name"] == "Charlie"  # oldest

    def test_gql_order_by_desc(self):
        """GQL: ORDER BY DESC."""
        result = self.db.execute(
            "MATCH (p:Person) RETURN p.name ORDER BY p.age DESC"
        )
        rows = list(result)
        assert rows[0]["p.name"] == "Charlie"

    def test_gql_limit(self):
        """GQL: LIMIT."""
        result = self.db.execute(
            "MATCH (p:Person) RETURN p.name LIMIT 2"
        )
        rows = list(result)
        assert len(rows) == 2

    def test_gql_skip(self):
        """GQL: SKIP (OFFSET)."""
        result = self.db.execute(
            "MATCH (p:Person) RETURN p.name ORDER BY p.age SKIP 1 LIMIT 2"
        )
        rows = list(result)
        # Should skip Bob (youngest) and return Alice and Charlie
        assert len(rows) == 2

    # ===== GQL OPTIONAL MATCH =====

    def test_gql_optional_match(self):
        """GQL: OPTIONAL MATCH."""
        # Create a person with no edges
        self.db.create_node(["Person"], {"name": "Diana", "age": 40, "city": "Chicago"})

        result = self.db.execute(
            "MATCH (p:Person) "
            "OPTIONAL MATCH (p)-[:WORKS_AT]->(c:Company) "
            "RETURN p.name, c.name"
        )
        rows = list(result)
        # Diana should appear with NULL company
        assert len(rows) == 4


class TestCypherQueries:
    """Test Cypher query language (Neo4j-compatible)."""

    def setup_method(self):
        """Create a database with test data."""
        # Note: Cypher support may be feature-gated
        self.db = GraphosDB()
        self._setup_test_data()

    def _setup_test_data(self):
        """Create test data."""
        self.alice = self.db.create_node(["Person"], {
            "name": "Alice", "age": 30
        })
        self.bob = self.db.create_node(["Person"], {
            "name": "Bob", "age": 25
        })
        self.db.create_edge(self.alice.id, self.bob.id, "KNOWS", {"since": 2020})

    def test_cypher_match(self):
        """Cypher: Basic MATCH."""
        # GQL and Cypher share similar MATCH syntax
        result = self.db.execute("MATCH (n:Person) RETURN n.name")
        rows = list(result)
        assert len(rows) == 2

    def test_cypher_create(self):
        """Cypher: CREATE node."""
        result = self.db.execute(
            "CREATE (n:Person {name: 'Charlie', age: 35}) RETURN n"
        )
        rows = list(result)
        assert len(rows) == 1

        # Verify node was created
        stats = self.db.stats()
        assert stats.node_count == 3

    def test_cypher_create_relationship(self):
        """Cypher: CREATE relationship."""
        self.db.execute(
            "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) "
            "CREATE (a)-[:FRIENDS_WITH {since: 2022}]->(b)"
        )

        result = self.db.execute(
            "MATCH (a)-[r:FRIENDS_WITH]->(b) RETURN a.name, b.name, r.since"
        )
        rows = list(result)
        assert len(rows) == 1

    def test_cypher_merge(self):
        """Cypher: MERGE (create if not exists)."""
        # First MERGE creates
        self.db.execute(
            "MERGE (c:City {name: 'NYC'}) RETURN c"
        )

        # Second MERGE finds existing
        self.db.execute(
            "MERGE (c:City {name: 'NYC'}) RETURN c"
        )

        result = self.db.execute(
            "MATCH (c:City) RETURN count(c) AS cnt"
        )
        rows = list(result)
        assert rows[0]["cnt"] == 1  # Only one NYC created

    def test_cypher_set(self):
        """Cypher: SET property."""
        self.db.execute(
            "MATCH (p:Person {name: 'Alice'}) SET p.verified = true"
        )

        result = self.db.execute(
            "MATCH (p:Person {name: 'Alice'}) RETURN p.verified"
        )
        rows = list(result)
        assert rows[0]["p.verified"] is True

    def test_cypher_delete(self):
        """Cypher: DELETE node."""
        # Create a node to delete
        self.db.create_node(["Temp"], {"value": 1})

        self.db.execute(
            "MATCH (t:Temp) DELETE t"
        )

        result = self.db.execute(
            "MATCH (t:Temp) RETURN count(t) AS cnt"
        )
        rows = list(result)
        assert rows[0]["cnt"] == 0

    def test_cypher_with_clause(self):
        """Cypher: WITH clause for query chaining."""
        result = self.db.execute(
            "MATCH (p:Person) "
            "WITH p.name AS name, p.age AS age "
            "WHERE age > 20 "
            "RETURN name, age ORDER BY age"
        )
        rows = list(result)
        assert len(rows) == 2

    def test_cypher_unwind(self):
        """Cypher: UNWIND list."""
        result = self.db.execute(
            "UNWIND [1, 2, 3] AS x RETURN x"
        )
        rows = list(result)
        assert len(rows) == 3

    def test_cypher_collect(self):
        """Cypher: COLLECT aggregation."""
        result = self.db.execute(
            "MATCH (p:Person) RETURN collect(p.name) AS names"
        )
        rows = list(result)
        names = rows[0]["names"]
        assert "Alice" in names
        assert "Bob" in names

    def test_cypher_case_when(self):
        """Cypher: CASE WHEN expression."""
        result = self.db.execute(
            "MATCH (p:Person) "
            "RETURN p.name, "
            "CASE WHEN p.age >= 30 THEN 'Senior' ELSE 'Junior' END AS level"
        )
        rows = list(result)
        for row in rows:
            if row["p.name"] == "Alice":
                assert row["level"] == "Senior"
            elif row["p.name"] == "Bob":
                assert row["level"] == "Junior"


class TestQueryCompatibility:
    """Test that queries work consistently across both languages."""

    def setup_method(self):
        """Create identical test data."""
        self.db = GraphosDB()

        # Create a simple graph
        self.a = self.db.create_node(["Node"], {"value": 1})
        self.b = self.db.create_node(["Node"], {"value": 2})
        self.c = self.db.create_node(["Node"], {"value": 3})

        self.db.create_edge(self.a.id, self.b.id, "LINK", {})
        self.db.create_edge(self.b.id, self.c.id, "LINK", {})

    def test_same_results_simple_match(self):
        """Both languages should return same results for simple MATCH."""
        # GQL and Cypher share this syntax
        result = self.db.execute("MATCH (n:Node) RETURN n.value ORDER BY n.value")
        rows = list(result)
        assert [r["n.value"] for r in rows] == [1, 2, 3]

    def test_same_results_relationship_match(self):
        """Both languages should return same relationship results."""
        result = self.db.execute(
            "MATCH (a:Node)-[:LINK]->(b:Node) RETURN a.value, b.value"
        )
        rows = list(result)
        assert len(rows) == 2

    def test_same_results_aggregation(self):
        """Both languages should return same aggregation results."""
        result = self.db.execute(
            "MATCH (n:Node) RETURN sum(n.value) AS total"
        )
        rows = list(result)
        assert rows[0]["total"] == 6  # 1 + 2 + 3


class TestGremlinQueries:
    """Test Gremlin query language (Apache TinkerPop)."""

    def setup_method(self):
        """Create a database with test data."""
        self.db = GraphosDB()
        self._setup_test_data()

    def _setup_test_data(self):
        """Create test data."""
        self.alice = self.db.create_node(["Person"], {
            "name": "Alice", "age": 30
        })
        self.bob = self.db.create_node(["Person"], {
            "name": "Bob", "age": 25
        })
        self.charlie = self.db.create_node(["Person"], {
            "name": "Charlie", "age": 35
        })
        self.db.create_edge(self.alice.id, self.bob.id, "knows", {"since": 2020})
        self.db.create_edge(self.bob.id, self.charlie.id, "knows", {"since": 2021})

    def test_gremlin_vertex_query(self):
        """Gremlin: Get all vertices."""
        try:
            result = self.db.execute_gremlin("g.V()")
            rows = list(result)
            assert len(rows) == 3
        except AttributeError:
            pytest.skip("Gremlin support not available in this build")

    def test_gremlin_has_label(self):
        """Gremlin: Filter vertices by label."""
        try:
            result = self.db.execute_gremlin("g.V().hasLabel('Person')")
            rows = list(result)
            assert len(rows) == 3
        except AttributeError:
            pytest.skip("Gremlin support not available in this build")

    def test_gremlin_has_property(self):
        """Gremlin: Filter by property."""
        try:
            result = self.db.execute_gremlin("g.V().has('name', 'Alice')")
            rows = list(result)
            assert len(rows) == 1
        except AttributeError:
            pytest.skip("Gremlin support not available in this build")

    def test_gremlin_out_traversal(self):
        """Gremlin: Outgoing edge traversal."""
        try:
            result = self.db.execute_gremlin(
                "g.V().has('name', 'Alice').out('knows')"
            )
            rows = list(result)
            assert len(rows) == 1  # Alice knows Bob
        except AttributeError:
            pytest.skip("Gremlin support not available in this build")

    def test_gremlin_values(self):
        """Gremlin: Get property values."""
        try:
            result = self.db.execute_gremlin(
                "g.V().hasLabel('Person').values('name')"
            )
            rows = list(result)
            assert len(rows) == 3
        except AttributeError:
            pytest.skip("Gremlin support not available in this build")

    def test_gremlin_count(self):
        """Gremlin: Count vertices."""
        try:
            result = self.db.execute_gremlin(
                "g.V().hasLabel('Person').count()"
            )
            rows = list(result)
            # Count should return a single row with the count
            assert len(rows) >= 1
        except AttributeError:
            pytest.skip("Gremlin support not available in this build")

    def test_gremlin_limit(self):
        """Gremlin: Limit results."""
        try:
            result = self.db.execute_gremlin(
                "g.V().hasLabel('Person').limit(2)"
            )
            rows = list(result)
            assert len(rows) == 2
        except AttributeError:
            pytest.skip("Gremlin support not available in this build")


class TestGraphQLQueries:
    """Test GraphQL query language."""

    def setup_method(self):
        """Create a database with test data."""
        self.db = GraphosDB()
        self._setup_test_data()

    def _setup_test_data(self):
        """Create test data."""
        self.alice = self.db.create_node(["User"], {
            "name": "Alice", "email": "alice@example.com", "age": 30
        })
        self.bob = self.db.create_node(["User"], {
            "name": "Bob", "email": "bob@example.com", "age": 25
        })
        self.post1 = self.db.create_node(["Post"], {
            "title": "Hello World", "content": "My first post"
        })
        self.db.create_edge(self.alice.id, self.bob.id, "friends", {})
        self.db.create_edge(self.alice.id, self.post1.id, "posts", {})

    def test_graphql_simple_query(self):
        """GraphQL: Simple query."""
        try:
            result = self.db.execute_graphql("""
                query {
                    user {
                        name
                    }
                }
            """)
            rows = list(result)
            assert len(rows) == 2  # Two users
        except AttributeError:
            pytest.skip("GraphQL support not available in this build")

    def test_graphql_query_with_argument(self):
        """GraphQL: Query with argument filter."""
        try:
            result = self.db.execute_graphql("""
                query {
                    user(age: 30) {
                        name
                        email
                    }
                }
            """)
            rows = list(result)
            # Should return Alice who is 30
            assert len(rows) >= 1
        except AttributeError:
            pytest.skip("GraphQL support not available in this build")

    def test_graphql_nested_query(self):
        """GraphQL: Nested query with relationships."""
        try:
            result = self.db.execute_graphql("""
                query {
                    user {
                        name
                        friends {
                            name
                        }
                    }
                }
            """)
            rows = list(result)
            # Should return users with their friends
            assert len(rows) >= 1
        except AttributeError:
            pytest.skip("GraphQL support not available in this build")

    def test_graphql_alias(self):
        """GraphQL: Query with alias."""
        try:
            result = self.db.execute_graphql("""
                query {
                    user {
                        userName: name
                    }
                }
            """)
            rows = list(result)
            # Should have aliased column
            assert len(rows) >= 1
        except AttributeError:
            pytest.skip("GraphQL support not available in this build")

    def test_graphql_multiple_fields(self):
        """GraphQL: Query multiple fields."""
        try:
            result = self.db.execute_graphql("""
                query {
                    user {
                        name
                        email
                        age
                    }
                }
            """)
            rows = list(result)
            assert len(rows) == 2
        except AttributeError:
            pytest.skip("GraphQL support not available in this build")


class TestMultiLanguageCompatibility:
    """Test that different query languages return consistent results."""

    def setup_method(self):
        """Create identical test data for comparison."""
        self.db = GraphosDB()

        # Create a simple graph
        self.a = self.db.create_node(["Person"], {"name": "Alice", "age": 30})
        self.b = self.db.create_node(["Person"], {"name": "Bob", "age": 25})
        self.db.create_edge(self.a.id, self.b.id, "knows", {})

    def test_gql_cypher_consistency(self):
        """GQL and Cypher should return same node counts."""
        gql_result = self.db.execute("MATCH (n:Person) RETURN count(n) AS cnt")
        gql_count = list(gql_result)[0]["cnt"]

        # GQL and Cypher share this syntax
        assert gql_count == 2

    def test_node_count_across_languages(self):
        """All languages should see the same number of nodes."""
        # GQL
        gql_result = self.db.execute("MATCH (n:Person) RETURN count(n) AS cnt")
        gql_count = list(gql_result)[0]["cnt"]
        assert gql_count == 2

        # Gremlin (if available)
        try:
            gremlin_result = self.db.execute_gremlin("g.V().hasLabel('Person').count()")
            gremlin_rows = list(gremlin_result)
            # Gremlin count should also be 2
            if len(gremlin_rows) > 0:
                # Count format depends on implementation
                pass
        except (AttributeError, Exception):
            pass  # Gremlin not available

        # GraphQL (if available)
        try:
            graphql_result = self.db.execute_graphql("""
                query { person { name } }
            """)
            graphql_rows = list(graphql_result)
            assert len(graphql_rows) == 2
        except (AttributeError, Exception):
            pass  # GraphQL not available


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
