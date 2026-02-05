"""SPARQL-specific pytest fixtures and configuration."""

import pytest

# Try to import grafeo
try:
    from grafeo import GrafeoDB
    GRAFEO_AVAILABLE = True
except ImportError:
    GRAFEO_AVAILABLE = False


def has_sparql_support(db):
    """Check if SPARQL support is available."""
    try:
        db.execute_sparql("SELECT * WHERE { ?s ?p ?o } LIMIT 1")
        return True
    except (AttributeError, NotImplementedError):
        return False
    except Exception:
        return True  # Has method but might fail for other reasons


@pytest.fixture
def db():
    """Create a fresh in-memory GrafeoDB instance."""
    if not GRAFEO_AVAILABLE:
        pytest.skip("grafeo not installed")
    db = GrafeoDB()
    if not has_sparql_support(db):
        pytest.skip("SPARQL support not available in this build")
    return db


@pytest.fixture
def db_api():
    """Create a fresh in-memory GrafeoDB instance for Python API tests.
    This fixture does NOT require SPARQL support."""
    if not GRAFEO_AVAILABLE:
        pytest.skip("grafeo not installed")
    return GrafeoDB()


@pytest.fixture
def sparql_db(db):
    """Create a database with RDF test data for SPARQL queries."""
    # Create resources representing triples
    alice = db.create_node(["Resource"], {
        "uri": "http://example.org/person/alice",
        "rdf:type": "http://xmlns.com/foaf/0.1/Person",
        "foaf:name": "Alice",
        "foaf:age": 30
    })

    bob = db.create_node(["Resource"], {
        "uri": "http://example.org/person/bob",
        "rdf:type": "http://xmlns.com/foaf/0.1/Person",
        "foaf:name": "Bob",
        "foaf:age": 25
    })

    charlie = db.create_node(["Resource"], {
        "uri": "http://example.org/person/charlie",
        "rdf:type": "http://xmlns.com/foaf/0.1/Person",
        "foaf:name": "Charlie",
        "foaf:age": 35
    })

    # Create foaf:knows relationships
    db.create_edge(alice.id, bob.id, "foaf:knows", {})
    db.create_edge(bob.id, charlie.id, "foaf:knows", {})

    return db
