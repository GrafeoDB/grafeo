"""SPARQL implementation of mutation tests.

Tests SPARQL Update operations (INSERT DATA, DELETE DATA, etc.).
Note: SPARQL mutations operate on RDF triples, not LPG nodes/edges.
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


class TestSPARQLMutations:
    """SPARQL Update mutation tests.

    Note: SPARQL mutations operate on RDF triples.
    The base mutation tests are for LPG, so we implement
    SPARQL-specific mutation tests here.
    """

    def test_insert_data_single_triple(self, db):
        """Test INSERT DATA with a single triple."""
        query = """
            INSERT DATA {
                <http://example.org/alice> <http://example.org/name> "Alice" .
            }
        """
        db.execute_sparql(query)

        # Verify the triple was inserted
        result = list(db.execute_sparql("""
            SELECT ?name WHERE {
                <http://example.org/alice> <http://example.org/name> ?name .
            }
        """))
        assert len(result) > 0

    def test_insert_data_multiple_triples(self, db):
        """Test INSERT DATA with multiple triples."""
        query = """
            INSERT DATA {
                <http://example.org/alice> <http://example.org/name> "Alice" .
                <http://example.org/alice> <http://example.org/age> 30 .
                <http://example.org/alice> a <http://example.org/Person> .
            }
        """
        db.execute_sparql(query)

        # Verify triples were inserted
        result = list(db.execute_sparql("""
            SELECT ?p ?o WHERE {
                <http://example.org/alice> ?p ?o .
            }
        """))
        assert len(result) >= 3

    def test_delete_data_single_triple(self, db):
        """Test DELETE DATA with a single triple."""
        # First insert
        db.execute_sparql("""
            INSERT DATA {
                <http://example.org/bob> <http://example.org/name> "Bob" .
            }
        """)

        # Then delete
        db.execute_sparql("""
            DELETE DATA {
                <http://example.org/bob> <http://example.org/name> "Bob" .
            }
        """)

        # Verify deletion
        result = list(db.execute_sparql("""
            SELECT ?name WHERE {
                <http://example.org/bob> <http://example.org/name> ?name .
            }
        """))
        assert len(result) == 0

    def test_delete_where(self, db):
        """Test DELETE WHERE pattern matching."""
        # Insert test data
        db.execute_sparql("""
            INSERT DATA {
                <http://example.org/temp1> <http://example.org/status> "temporary" .
                <http://example.org/temp2> <http://example.org/status> "temporary" .
                <http://example.org/keep> <http://example.org/status> "permanent" .
            }
        """)

        # Delete all temporary items
        db.execute_sparql("""
            DELETE WHERE {
                ?s <http://example.org/status> "temporary" .
            }
        """)

        # Verify only permanent item remains
        result = list(db.execute_sparql("""
            SELECT ?s WHERE {
                ?s <http://example.org/status> ?status .
            }
        """))
        assert len(result) == 1

    def test_modify_delete_insert(self, db):
        """Test DELETE/INSERT WHERE (modify operation)."""
        # Insert initial data
        db.execute_sparql("""
            INSERT DATA {
                <http://example.org/item> <http://example.org/version> 1 .
            }
        """)

        # Modify: delete old version, insert new version
        db.execute_sparql("""
            DELETE { ?s <http://example.org/version> ?old }
            INSERT { ?s <http://example.org/version> 2 }
            WHERE { ?s <http://example.org/version> ?old }
        """)

        # Verify version was updated
        result = list(db.execute_sparql("""
            SELECT ?v WHERE {
                <http://example.org/item> <http://example.org/version> ?v .
            }
        """))
        assert len(result) == 1
        # Version should be 2 now


class TestSPARQLGraphManagement:
    """Tests for SPARQL graph management operations."""

    def test_create_graph(self, db):
        """Test CREATE GRAPH."""
        db.execute_sparql("""
            CREATE GRAPH <http://example.org/newgraph>
        """)
        # Graph creation should succeed

    def test_drop_graph(self, db):
        """Test DROP GRAPH."""
        db.execute_sparql("""
            CREATE GRAPH <http://example.org/tempgraph>
        """)
        db.execute_sparql("""
            DROP GRAPH <http://example.org/tempgraph>
        """)
        # Graph should be dropped

    def test_clear_default(self, db):
        """Test CLEAR DEFAULT."""
        # Insert data
        db.execute_sparql("""
            INSERT DATA {
                <http://example.org/s> <http://example.org/p> "value" .
            }
        """)

        # Clear default graph
        db.execute_sparql("CLEAR DEFAULT")

        # Verify data is gone
        result = list(db.execute_sparql("""
            SELECT ?s WHERE { ?s ?p ?o }
        """))
        assert len(result) == 0
