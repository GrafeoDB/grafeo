"""SolvOR comparison tests for the RDF GraphQL context.

NOTE: The SolvOR adapter in Grafeo works with the LPG (Labeled Property Graph)
store for flow network optimization problems. RDF triples are stored separately.

The "RDF GraphQL" context indicates the query language environment, but
SolvOR comparison tests use LPG data for graph structure.
"""

import random
import pytest
from tests.python.bases.test_solvor import (
    BaseSolvORComparisonTest,
    BaseSolvORBenchmarkTest,
)

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


@pytest.fixture
def db():
    """Create a fresh database instance."""
    return GrafeoDB()


class TestRDFGraphQLSolvORComparison(BaseSolvORComparisonTest):
    """GraphQL on RDF implementation of solvOR plugin comparison tests.

    Tests Grafeo's as_solvor() plugin against standalone solvOR library.
    Graph setup uses Python API with RDF-style nodes (GraphQL execution not required).
    """

    def create_db(self):
        """Create a fresh database instance."""
        return GrafeoDB()

    def setup_flow_network(self, db, n_nodes: int, n_edges: int,
                           seed: int = 42) -> dict:
        """Set up a flow network using Python API with RDF-style nodes."""
        rng = random.Random(seed)

        node_ids = []
        for i in range(n_nodes):
            node = db.create_node(["Resource", "Node"], {
                "uri": f"http://example.org/node/{i}",
                "index": i
            })
            node_ids.append(node.id)

        source = node_ids[0]
        sink = node_ids[-1]

        edges: list[tuple] = []
        edge_set: set[tuple] = set()

        # Create backbone path to ensure connectivity
        for i in range(min(5, n_nodes - 1)):
            src = node_ids[i]
            dst = node_ids[i + 1]
            if (src, dst) not in edge_set:
                capacity = rng.randint(10, 50)
                cost = rng.randint(1, 10)
                db.create_edge(src, dst, "FLOW", {"capacity": capacity, "cost": cost})
                edges.append((src, dst, capacity, cost))
                edge_set.add((src, dst))

        # Add remaining random edges
        attempts = 0
        while len(edges) < n_edges and attempts < n_edges * 3:
            attempts += 1
            src = rng.choice(node_ids)
            dst = rng.choice(node_ids)
            if src != dst and (src, dst) not in edge_set and (dst, src) not in edge_set:
                capacity = rng.randint(5, 30)
                cost = rng.randint(1, 15)
                db.create_edge(src, dst, "FLOW", {"capacity": capacity, "cost": cost})
                edges.append((src, dst, capacity, cost))
                edge_set.add((src, dst))

        return {"node_ids": node_ids, "source": source, "sink": sink, "edges": edges}


class TestRDFGraphQLSolvORBenchmark(BaseSolvORBenchmarkTest):
    """GraphQL on RDF implementation of solvOR plugin benchmark tests.

    Uses RDF-style nodes (with URIs).
    """

    def create_db(self):
        """Create a fresh database instance."""
        return GrafeoDB()

    def setup_flow_network(self, db, n_nodes: int, n_edges: int,
                           seed: int = 42) -> dict:
        """Set up a flow network using Python API with RDF-style nodes."""
        rng = random.Random(seed)

        node_ids = []
        for i in range(n_nodes):
            node = db.create_node(["Resource", "Node"], {
                "uri": f"http://example.org/node/{i}",
                "index": i
            })
            node_ids.append(node.id)

        source = node_ids[0]
        sink = node_ids[-1]

        edges: list[tuple] = []
        edge_set: set[tuple] = set()

        # Create backbone path
        for i in range(min(5, n_nodes - 1)):
            src = node_ids[i]
            dst = node_ids[i + 1]
            if (src, dst) not in edge_set:
                capacity = rng.randint(10, 50)
                cost = rng.randint(1, 10)
                db.create_edge(src, dst, "FLOW", {"capacity": capacity, "cost": cost})
                edges.append((src, dst, capacity, cost))
                edge_set.add((src, dst))

        # Add remaining random edges
        attempts = 0
        while len(edges) < n_edges and attempts < n_edges * 3:
            attempts += 1
            src = rng.choice(node_ids)
            dst = rng.choice(node_ids)
            if src != dst and (src, dst) not in edge_set and (dst, src) not in edge_set:
                capacity = rng.randint(5, 30)
                cost = rng.randint(1, 15)
                db.create_edge(src, dst, "FLOW", {"capacity": capacity, "cost": cost})
                edges.append((src, dst, capacity, cost))
                edge_set.add((src, dst))

        return {"node_ids": node_ids, "source": source, "sink": sink, "edges": edges}
