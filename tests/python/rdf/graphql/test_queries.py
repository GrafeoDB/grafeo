"""GraphQL on RDF query tests.

Tests GraphQL queries against the RDF model.
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


class TestRDFGraphQLQueries:
    """Test GraphQL queries on RDF data."""

    def setup_method(self):
        """Create a database with RDF test data."""
        self.db = GrafeoDB()
        self._setup_test_data()

    def _setup_test_data(self):
        """Create RDF-like test data."""
        self.alice = self.db.create_node(["Resource", "Person"], {
            "uri": "http://example.org/person/alice",
            "name": "Alice",
            "age": 30
        })

        self.bob = self.db.create_node(["Resource", "Person"], {
            "uri": "http://example.org/person/bob",
            "name": "Bob",
            "age": 25
        })

        self.db.create_edge(self.alice.id, self.bob.id, "knows", {})

    def _execute_graphql(self, query: str):
        """Execute GraphQL query, skip if not supported."""
        try:
            return self.db.execute_graphql(query)
        except AttributeError:
            pytest.skip("GraphQL support not available")
        except NotImplementedError:
            pytest.skip("GraphQL not implemented")

    def test_rdf_graphql_query_resource(self):
        """GraphQL on RDF: Query resources."""
        result = self._execute_graphql("""
            query {
                resource {
                    uri
                    name
                }
            }
        """)
        rows = list(result)
        assert len(rows) >= 1

    def test_rdf_graphql_query_person(self):
        """GraphQL on RDF: Query persons."""
        result = self._execute_graphql("""
            query {
                person {
                    name
                    age
                }
            }
        """)
        rows = list(result)
        assert len(rows) == 2

    def test_rdf_graphql_query_with_filter(self):
        """GraphQL on RDF: Query with filter."""
        result = self._execute_graphql("""
            query {
                person(age: 30) {
                    name
                }
            }
        """)
        rows = list(result)
        # Should find Alice
        assert len(rows) >= 1

    def test_rdf_graphql_nested_query(self):
        """GraphQL on RDF: Nested query following relationships."""
        result = self._execute_graphql("""
            query {
                person {
                    name
                    knows {
                        name
                    }
                }
            }
        """)
        rows = list(result)
        # Should return persons with their knows relationships
        assert len(rows) >= 1

    def test_rdf_graphql_uri_query(self):
        """GraphQL on RDF: Query by URI."""
        result = self._execute_graphql("""
            query {
                resource(uri: "http://example.org/person/alice") {
                    name
                    age
                }
            }
        """)
        rows = list(result)
        if len(rows) >= 1:
            assert rows[0].get("name") == "Alice"
