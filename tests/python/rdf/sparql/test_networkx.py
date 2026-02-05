"""SPARQL implementation of NetworkX comparison tests.

Compares Grafeo algorithm results against NetworkX to verify correctness.
Note: These tests use Python API only, they don't require SPARQL support.
"""

import random
import time
import pytest
from grafeo import GrafeoDB
from tests.python.bases.test_networkx import (
    BaseNetworkXComparisonTest,
    BaseNetworkXBenchmarkTest,
)


@pytest.fixture
def db():
    """Create a fresh database instance (no SPARQL required)."""
    return GrafeoDB()


class TestSPARQLNetworkXComparison(BaseNetworkXComparisonTest):
    """SPARQL implementation of NetworkX comparison tests.

    Note: Algorithms are accessed via db.algorithms.*, not via SPARQL queries.
    Uses Python API for graph setup (faster).
    """

    def create_db(self):
        """Create a fresh database instance."""
        return GrafeoDB()

    def setup_random_graph(self, db, n_nodes: int, n_edges: int,
                           weighted: bool = True, seed: int = 42) -> dict:
        """Set up a random graph for testing."""
        rng = random.Random(seed)

        node_ids = []
        for i in range(n_nodes):
            node = db.create_node(["Node"], {"index": i})
            node_ids.append(node.id)

        edges = []
        edge_set = set()
        while len(edges) < n_edges:
            src = rng.choice(node_ids)
            dst = rng.choice(node_ids)
            if src != dst and (src, dst) not in edge_set:
                weight = rng.uniform(0.1, 10.0) if weighted else 1.0
                props = {"weight": weight} if weighted else {}
                db.create_edge(src, dst, "EDGE", props)
                edges.append((src, dst, weight))
                edge_set.add((src, dst))

        return {"node_ids": node_ids, "edges": edges}

    def run_bfs(self, db, start_node) -> set:
        """Run BFS and return visited nodes as a set."""
        result = db.algorithms.bfs(start_node)
        return set(result)

    def run_dfs(self, db, start_node) -> set:
        """Run DFS and return visited nodes as a set."""
        result = db.algorithms.dfs(start_node)
        return set(result)

    def run_dijkstra(self, db, source) -> dict:
        """Run Dijkstra and return {node: distance} dict."""
        return db.algorithms.dijkstra(source, weight="weight")

    def run_connected_components(self, db) -> int:
        """Run connected components and return count."""
        return db.algorithms.connected_component_count()

    def run_pagerank(self, db, damping: float = 0.85) -> dict:
        """Run PageRank and return {node: score} dict."""
        return db.algorithms.pagerank(damping=damping)

    def run_degree_centrality(self, db) -> dict:
        """Run degree centrality and return {node: centrality} dict."""
        return db.algorithms.degree_centrality(normalized=True)


class TestSPARQLNetworkXBenchmark(BaseNetworkXBenchmarkTest):
    """SPARQL implementation of NetworkX performance comparison tests.

    Note: Algorithms are accessed via db.algorithms.*, not via SPARQL queries.
    """

    def create_db(self):
        """Create a fresh database instance."""
        return GrafeoDB()

    def setup_random_graph(self, db, n_nodes: int, n_edges: int,
                           weighted: bool = True, seed: int = 42) -> dict:
        """Set up a random graph for benchmarking."""
        rng = random.Random(seed)

        node_ids = []
        for i in range(n_nodes):
            node = db.create_node(["Node"], {"index": i})
            node_ids.append(node.id)

        edges = []
        edge_set = set()
        while len(edges) < n_edges:
            src = rng.choice(node_ids)
            dst = rng.choice(node_ids)
            if src != dst and (src, dst) not in edge_set:
                weight = rng.uniform(0.1, 10.0) if weighted else 1.0
                props = {"weight": weight} if weighted else {}
                db.create_edge(src, dst, "EDGE", props)
                edges.append((src, dst, weight))
                edge_set.add((src, dst))

        return {"node_ids": node_ids, "edges": edges}

    def run_grafeo_pagerank(self, db) -> float:
        """Run Grafeo PageRank and return execution time in ms."""
        start = time.perf_counter()
        db.algorithms.pagerank(damping=0.85)
        return (time.perf_counter() - start) * 1000

    def run_grafeo_dijkstra(self, db, source) -> float:
        """Run Grafeo Dijkstra and return execution time in ms."""
        start = time.perf_counter()
        db.algorithms.dijkstra(source, weight="weight")
        return (time.perf_counter() - start) * 1000

    def run_grafeo_bfs(self, db, start) -> float:
        """Run Grafeo BFS and return execution time in ms."""
        start_time = time.perf_counter()
        db.algorithms.bfs(start)
        return (time.perf_counter() - start_time) * 1000
