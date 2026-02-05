"""GraphQL implementation of solvOR plugin comparison tests.

Compares Grafeo's as_solvor() plugin against standalone solvOR library.
Uses Python API for graph setup (GraphQL execution not required).
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


class TestGraphQLSolvORComparison(BaseSolvORComparisonTest):
    """GraphQL implementation of solvOR plugin comparison tests.

    Tests Grafeo's as_solvor() plugin against standalone solvOR library.
    Graph setup uses Python API (GraphQL execution not required).
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


class TestGraphQLSolvORBenchmark(BaseSolvORBenchmarkTest):
    """GraphQL implementation of solvOR plugin benchmark tests."""

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
