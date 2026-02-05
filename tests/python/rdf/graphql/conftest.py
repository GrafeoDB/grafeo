"""GraphQL on RDF pytest fixtures and configuration."""

import pytest

# Try to import grafeo
try:
    from grafeo import GrafeoDB
    GRAFEO_AVAILABLE = True
except ImportError:
    GRAFEO_AVAILABLE = False


@pytest.fixture
def db():
    """Create a fresh in-memory GrafeoDB instance."""
    if not GRAFEO_AVAILABLE:
        pytest.skip("grafeo not installed")
    return GrafeoDB()


@pytest.fixture
def rdf_graphql_db(db):
    """Create a database with RDF data for GraphQL queries."""
    # Create resources with URIs
    alice = db.create_node(["Resource", "Person"], {
        "uri": "http://example.org/person/alice",
        "name": "Alice",
        "age": 30
    })

    bob = db.create_node(["Resource", "Person"], {
        "uri": "http://example.org/person/bob",
        "name": "Bob",
        "age": 25
    })

    # Create knows relationship
    db.create_edge(alice.id, bob.id, "knows", {})

    return db
