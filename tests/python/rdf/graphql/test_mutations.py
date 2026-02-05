"""GraphQL on RDF mutation tests.

NOTE: GraphQL on RDF currently only supports Query operations, not mutations.
The graphql_rdf_translator in Rust explicitly rejects mutation operations.

For RDF data modifications, use SPARQL UPDATE operations:
- INSERT DATA for adding triples
- DELETE DATA for removing triples
- DELETE/INSERT WHERE for modifications
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


class TestRDFGraphQLMutations:
    """GraphQL on RDF mutation tests.

    Note: GraphQL on RDF does not currently support mutations.
    The graphql_rdf_translator only handles Query operations.
    Mutations on RDF data should use SPARQL UPDATE instead.

    These tests verify that SPARQL UPDATE operations work correctly
    for modifying RDF data.
    """

    def setup_method(self):
        """Create a fresh database."""
        self.db = GrafeoDB()

    def test_sparql_insert_data(self):
        """Test SPARQL INSERT DATA for adding triples."""
        # Insert RDF data using SPARQL
        self.db.execute_sparql("""
            INSERT DATA {
                <http://example.org/alice> <http://example.org/name> "Alice" .
                <http://example.org/alice> <http://example.org/age> 30 .
            }
        """)

        # Verify data was inserted
        result = list(self.db.execute_sparql("""
            SELECT ?name WHERE {
                <http://example.org/alice> <http://example.org/name> ?name .
            }
        """))
        assert len(result) == 1

    def test_sparql_delete_data(self):
        """Test SPARQL DELETE DATA for removing triples."""
        # Insert data first
        self.db.execute_sparql("""
            INSERT DATA {
                <http://example.org/bob> <http://example.org/name> "Bob" .
            }
        """)

        # Verify insertion
        result = list(self.db.execute_sparql("""
            SELECT ?name WHERE {
                <http://example.org/bob> <http://example.org/name> ?name .
            }
        """))
        assert len(result) == 1

        # Delete the triple
        self.db.execute_sparql("""
            DELETE DATA {
                <http://example.org/bob> <http://example.org/name> "Bob" .
            }
        """)

        # Verify deletion
        result = list(self.db.execute_sparql("""
            SELECT ?name WHERE {
                <http://example.org/bob> <http://example.org/name> ?name .
            }
        """))
        assert len(result) == 0

    def test_use_sparql_for_rdf_mutations(self):
        """Demonstrate that SPARQL should be used for RDF mutations.

        For RDF data modifications, use SPARQL UPDATE operations:
        - INSERT DATA for adding triples
        - DELETE DATA for removing triples
        - DELETE/INSERT WHERE for modifications
        """
        # Insert RDF data using SPARQL
        self.db.execute_sparql("""
            INSERT DATA {
                <http://example.org/alice> <http://example.org/name> "Alice" .
                <http://example.org/alice> <http://example.org/age> 30 .
            }
        """)

        # Verify data was inserted
        result = list(self.db.execute_sparql("""
            SELECT ?name WHERE {
                <http://example.org/alice> <http://example.org/name> ?name .
            }
        """))
        assert len(result) == 1

        # Delete using SPARQL
        self.db.execute_sparql("""
            DELETE DATA {
                <http://example.org/alice> <http://example.org/name> "Alice" .
            }
        """)

        # Verify deletion
        result = list(self.db.execute_sparql("""
            SELECT ?name WHERE {
                <http://example.org/alice> <http://example.org/name> ?name .
            }
        """))
        assert len(result) == 0
