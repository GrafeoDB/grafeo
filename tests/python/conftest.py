"""Root pytest fixtures for Grafeo Python tests.

This module provides common fixtures available to all test modules.
"""

import sys
from pathlib import Path

# Add the project root to sys.path so imports like 'tests.python.bases' work
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
import random

# Try to import grafeo
try:
    import grafeo
    GRAFEO_AVAILABLE = True
except ImportError:
    GRAFEO_AVAILABLE = False


# Import fixtures from fixtures module
try:
    from tests.python.fixtures.generators import (
        SocialNetworkGenerator,
        LDBCLikeGenerator,
        TreeGenerator,
        CliqueGenerator,
        load_data_into_db,
    )
    from tests.python.fixtures.datasets import (
        create_social_graph,
        create_ldbc_graph,
        create_tree_graph,
        create_algorithm_test_graph,
        create_pattern_test_graph,
    )
except ImportError:
    # Fallback if relative imports don't work
    pass


@pytest.fixture
def db():
    """Create a fresh in-memory GrafeoDB instance."""
    if not GRAFEO_AVAILABLE:
        pytest.skip("grafeo not installed")
    return grafeo.GrafeoDB()


@pytest.fixture
def node_ids(db):
    """Create a test graph and return node IDs.

    Creates a random graph with 100 nodes and 300 edges for algorithm testing.
    """
    n_nodes = 100
    n_edges = 300

    node_ids = []
    for i in range(n_nodes):
        node = db.create_node(["Node"], {"index": i})
        node_ids.append(node.id)

    edges = set()
    while len(edges) < n_edges:
        src = random.choice(node_ids)
        dst = random.choice(node_ids)
        if src != dst and (src, dst) not in edges:
            db.create_edge(src, dst, "EDGE", {"weight": random.uniform(0.1, 10.0)})
            edges.add((src, dst))

    return node_ids


@pytest.fixture
def social_graph(db):
    """Create a social network graph.

    Creates 50 Person nodes with KNOWS edges.
    """
    if not GRAFEO_AVAILABLE:
        pytest.skip("grafeo not installed")

    try:
        return create_social_graph(db, size=50)
    except NameError:
        # Fallback implementation
        gen = SocialNetworkGenerator(num_nodes=50, avg_edges_per_node=5, seed=42)
        node_count, edge_count = load_data_into_db(db, gen)
        return {"node_count": node_count, "edge_count": edge_count}


@pytest.fixture
def pattern_graph(db):
    """Create a graph for pattern matching tests.

    Creates Person and Company nodes with various relationships.
    """
    if not GRAFEO_AVAILABLE:
        pytest.skip("grafeo not installed")

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

    return {
        "alice": alice,
        "bob": bob,
        "charlie": charlie,
        "acme": acme,
        "globex": globex,
    }


# Register markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as benchmarks"
    )
    config.addinivalue_line(
        "markers", "gql: marks tests requiring GQL support"
    )
    config.addinivalue_line(
        "markers", "cypher: marks tests requiring Cypher support"
    )
    config.addinivalue_line(
        "markers", "gremlin: marks tests requiring Gremlin support"
    )
    config.addinivalue_line(
        "markers", "graphql: marks tests requiring GraphQL support"
    )
    config.addinivalue_line(
        "markers", "sparql: marks tests requiring SPARQL support"
    )
