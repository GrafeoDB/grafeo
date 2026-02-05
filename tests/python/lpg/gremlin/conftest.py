"""Gremlin-specific pytest fixtures and configuration."""

import pytest

# Try to import grafeo
try:
    from grafeo import GrafeoDB
    GRAFEO_AVAILABLE = True
except ImportError:
    GRAFEO_AVAILABLE = False


def has_gremlin_support(db):
    """Check if Gremlin support is available."""
    try:
        db.execute_gremlin("g.V().limit(1)")
        return True
    except (AttributeError, NotImplementedError):
        return False
    except Exception:
        return True  # Has method but might fail for other reasons


@pytest.fixture
def db():
    """Create a fresh in-memory GrafeoDB instance.

    This fixture requires Gremlin support.
    For tests using only Python API, use db_api fixture instead.
    """
    if not GRAFEO_AVAILABLE:
        pytest.skip("grafeo not installed")
    database = GrafeoDB()
    if not has_gremlin_support(database):
        pytest.skip("Gremlin support not available in this build")
    return database


@pytest.fixture
def db_api():
    """Create a fresh in-memory GrafeoDB instance for Python API tests.

    This fixture does NOT require Gremlin support - use for tests
    that only use db.create_node(), db.algorithms, etc.
    """
    if not GRAFEO_AVAILABLE:
        pytest.skip("grafeo not installed")
    return GrafeoDB()


@pytest.fixture
def traversal_db(db):
    """Create a database with traversal test data."""
    # Create Person nodes
    alice = db.create_node(["Person"], {"name": "Alice", "age": 30})
    bob = db.create_node(["Person"], {"name": "Bob", "age": 25})
    charlie = db.create_node(["Person"], {"name": "Charlie", "age": 35})

    # Create knows edges (lowercase for Gremlin convention)
    db.create_edge(alice.id, bob.id, "knows", {"since": 2020})
    db.create_edge(bob.id, charlie.id, "knows", {"since": 2021})

    return db
