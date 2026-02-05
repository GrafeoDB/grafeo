"""SPARQL implementation of solvOR plugin comparison tests.

Compares Grafeo's as_solvor() plugin against standalone solvOR library.
Uses Python API for graph setup (SPARQL query execution not required).
"""

import random
import pytest
from grafeo import GrafeoDB
from tests.python.bases.test_solvor import (
    BaseSolvORComparisonTest,
    BaseSolvORBenchmarkTest,
)


@pytest.fixture
def db():
    """Create a fresh database instance."""
    return GrafeoDB()


class TestSPARQLSolvORComparison(BaseSolvORComparisonTest):
    """SPARQL implementation of solvOR plugin comparison tests.

    Tests Grafeo's as_solvor() plugin against standalone solvOR library.
    Graph setup uses Python API (SPARQL execution not required).
    """

    def create_db(self):
        """Create a fresh database instance."""
        return GrafeoDB()

    def setup_flow_network(self, db, n_nodes: int, n_edges: int,
                           seed: int = 42) -> dict:
        """Set up a flow network using Python API."""
        rng = random.Random(seed)

        node_ids = []
        for i in range(n_nodes):
            node = db.create_node(["Node"], {"index": i})
            node_ids.append(node.id)

        source = node_ids[0]
        sink = node_ids[-1]

        edges: list[tuple] = []
        edge_set: set[tuple] = set()
        while len(edges) < n_edges:
            src_idx = rng.randint(0, n_nodes - 2)
            dst_idx = rng.randint(src_idx + 1, n_nodes - 1)
            src = node_ids[src_idx]
            dst = node_ids[dst_idx]

            if (src, dst) not in edge_set:
                capacity = rng.randint(1, 100)
                cost = rng.randint(1, 50)
                db.create_edge(src, dst, "FLOW", {"capacity": capacity, "cost": cost})
                edges.append((src, dst, capacity, cost))
                edge_set.add((src, dst))

        return {"node_ids": node_ids, "source": source, "sink": sink, "edges": edges}


class TestSPARQLSolvORBenchmark(BaseSolvORBenchmarkTest):
    """SPARQL implementation of solvOR plugin benchmark tests."""

    def create_db(self):
        """Create a fresh database instance."""
        return GrafeoDB()

    def setup_flow_network(self, db, n_nodes: int, n_edges: int,
                           seed: int = 42) -> dict:
        """Set up a flow network using Python API."""
        rng = random.Random(seed)

        node_ids = []
        for i in range(n_nodes):
            node = db.create_node(["Node"], {"index": i})
            node_ids.append(node.id)

        source = node_ids[0]
        sink = node_ids[-1]

        edges: list[tuple] = []
        edge_set: set[tuple] = set()
        while len(edges) < n_edges:
            src_idx = rng.randint(0, n_nodes - 2)
            dst_idx = rng.randint(src_idx + 1, n_nodes - 1)
            src = node_ids[src_idx]
            dst = node_ids[dst_idx]

            if (src, dst) not in edge_set:
                capacity = rng.randint(1, 100)
                cost = rng.randint(1, 50)
                db.create_edge(src, dst, "FLOW", {"capacity": capacity, "cost": cost})
                edges.append((src, dst, capacity, cost))
                edge_set.add((src, dst))

        return {"node_ids": node_ids, "source": source, "sink": sink, "edges": edges}
