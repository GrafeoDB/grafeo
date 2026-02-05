"""GraphQL on RDF transaction tests.

Tests transaction behavior with RDF data and GraphQL queries.
Since GraphQL on RDF only supports queries (not mutations),
we test read isolation within transactions using SPARQL for mutations.
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


class TestRDFGraphQLTransactions:
    """Test transaction behavior with RDF data.

    GraphQL on RDF only supports Query operations, so mutations
    must be performed with SPARQL. These tests verify that:
    1. SPARQL mutations work within transactions
    2. Transaction isolation works correctly
    3. Commit/rollback affects RDF data appropriately
    """

    def setup_method(self):
        """Create a fresh database."""
        self.db = GrafeoDB()

    def test_sparql_transaction_commit(self):
        """Test that committed SPARQL changes persist."""
        tx = self.db.begin_transaction()

        # Insert data in transaction
        tx.execute_sparql("""
            INSERT DATA {
                <http://example.org/tx/alice> <http://example.org/name> "Alice" .
            }
        """)

        tx.commit()

        # Verify data persisted after commit
        result = list(self.db.execute_sparql("""
            SELECT ?name WHERE {
                <http://example.org/tx/alice> <http://example.org/name> ?name .
            }
        """))
        assert len(result) == 1

    def test_sparql_transaction_rollback(self):
        """Test that rolled back SPARQL changes don't persist."""
        tx = self.db.begin_transaction()

        # Insert data in transaction
        tx.execute_sparql("""
            INSERT DATA {
                <http://example.org/rollback/bob> <http://example.org/name> "Bob" .
            }
        """)

        tx.rollback()

        # Verify data was not persisted
        result = list(self.db.execute_sparql("""
            SELECT ?name WHERE {
                <http://example.org/rollback/bob> <http://example.org/name> ?name .
            }
        """))
        assert len(result) == 0

    def test_multiple_sparql_operations_in_transaction(self):
        """Test multiple SPARQL operations in single transaction."""
        tx = self.db.begin_transaction()

        # Insert multiple triples
        tx.execute_sparql("""
            INSERT DATA {
                <http://example.org/multi/alice> <http://example.org/name> "Alice" .
                <http://example.org/multi/bob> <http://example.org/name> "Bob" .
            }
        """)

        # Delete one triple
        tx.execute_sparql("""
            DELETE DATA {
                <http://example.org/multi/alice> <http://example.org/name> "Alice" .
            }
        """)

        tx.commit()

        # Only Bob should remain
        result = list(self.db.execute_sparql("""
            SELECT ?s WHERE {
                ?s <http://example.org/name> ?name .
            }
        """))
        assert len(result) == 1
