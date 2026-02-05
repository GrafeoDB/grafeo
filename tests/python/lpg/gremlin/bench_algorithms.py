"""Gremlin implementation of algorithm benchmarks.

Benchmarks graph algorithms using db.algorithms.* API.
Gremlin is used for setup only.
"""

import random
from tests.python.bases.bench_algorithms import BaseBenchAlgorithms


class BenchGremlinAlgorithms(BaseBenchAlgorithms):
    """Gremlin implementation of algorithm benchmarks.

    Note: Algorithms are accessed via db.algorithms.*, not via Gremlin queries.
    Uses Python API for graph setup (faster for benchmarks).
    """

    def setup_random_graph(self, db, n_nodes: int, n_edges: int,
                           weighted: bool = True) -> dict:
        """Set up a random graph for benchmarking."""
        rng = random.Random(42)

        node_ids = []
        for i in range(n_nodes):
            node = db.create_node(["Node"], {"index": i})
            node_ids.append(node.id)

        edges = set()
        while len(edges) < n_edges:
            src = rng.choice(node_ids)
            dst = rng.choice(node_ids)
            if src != dst and (src, dst) not in edges:
                props = {"weight": rng.uniform(0.1, 10.0)} if weighted else {}
                db.create_edge(src, dst, "edge", props)
                edges.add((src, dst))

        return {"node_ids": node_ids, "edge_count": len(edges)}

    def run_bfs(self, db, start_node) -> list:
        """Run BFS from start node."""
        return db.algorithms.bfs(start_node)

    def run_dfs(self, db, start_node) -> list:
        """Run DFS from start node."""
        return db.algorithms.dfs(start_node)

    def run_dijkstra(self, db, source, target=None, weight_prop: str = "weight"):
        """Run Dijkstra's algorithm."""
        if target is None:
            return db.algorithms.dijkstra(source, weight=weight_prop)
        return db.algorithms.dijkstra(source, target, weight_prop)

    def run_bellman_ford(self, db, source, weight_prop: str = "weight"):
        """Run Bellman-Ford algorithm."""
        return db.algorithms.bellman_ford(source, weight_prop)

    def run_connected_components(self, db) -> dict:
        """Run connected components."""
        return db.algorithms.connected_components()

    def run_strongly_connected_components(self, db) -> list:
        """Run strongly connected components."""
        return db.algorithms.strongly_connected_components()

    def run_pagerank(self, db, damping: float = 0.85, iterations: int = 20) -> dict:
        """Run PageRank algorithm."""
        return db.algorithms.pagerank(damping=damping, max_iterations=iterations)

    def run_degree_centrality(self, db, normalized: bool = True) -> dict:
        """Run degree centrality."""
        return db.algorithms.degree_centrality(normalized=normalized)

    def run_betweenness_centrality(self, db) -> dict:
        """Run betweenness centrality."""
        return db.algorithms.betweenness_centrality()

    def run_closeness_centrality(self, db) -> dict:
        """Run closeness centrality."""
        return db.algorithms.closeness_centrality()

    def run_label_propagation(self, db) -> dict:
        """Run label propagation community detection."""
        return db.algorithms.label_propagation()

    def run_louvain(self, db) -> dict:
        """Run Louvain community detection."""
        return db.algorithms.louvain()

    def run_kruskal(self, db, weight_prop: str = "weight") -> dict:
        """Run Kruskal's MST algorithm."""
        return db.algorithms.kruskal(weight_prop)

    def run_prim(self, db, weight_prop: str = "weight") -> dict:
        """Run Prim's MST algorithm."""
        return db.algorithms.prim(weight_prop)
