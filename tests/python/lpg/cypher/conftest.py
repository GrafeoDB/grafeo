"""Cypher-specific pytest fixtures and configuration."""

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
def pattern_db(db):
    """Create a database with pattern test data."""
    # Create Person nodes
    alice = db.create_node(["Person"], {
        "name": "Alice", "age": 30, "city": "NYC"
    })
    bob = db.create_node(["Person"], {
        "name": "Bob", "age": 25, "city": "LA"
    })
    charlie = db.create_node(["Person"], {
        "name": "Charlie", "age": 35, "city": "NYC"
    })

    # Create Company nodes
    acme = db.create_node(["Company"], {
        "name": "Acme Corp", "founded": 2010
    })
    globex = db.create_node(["Company"], {
        "name": "Globex Inc", "founded": 2015
    })

    # Create KNOWS edges
    db.create_edge(alice.id, bob.id, "KNOWS", {"since": 2020})
    db.create_edge(bob.id, charlie.id, "KNOWS", {"since": 2021})
    db.create_edge(alice.id, charlie.id, "KNOWS", {"since": 2019})

    # Create WORKS_AT edges
    db.create_edge(alice.id, acme.id, "WORKS_AT", {"role": "Engineer"})
    db.create_edge(bob.id, globex.id, "WORKS_AT", {"role": "Manager"})
    db.create_edge(charlie.id, acme.id, "WORKS_AT", {"role": "Director"})

    return db
