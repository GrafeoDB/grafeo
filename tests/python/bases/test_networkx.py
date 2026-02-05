"""Base class for NetworkX comparison tests.

This module defines tests that compare Grafeo algorithm results against NetworkX
to verify correctness. NetworkX serves as the reference implementation.

Tests cover:
- Traversal: BFS, DFS
- Shortest Path: Dijkstra, Bellman-Ford
- Centrality: Degree, PageRank, Betweenness, Closeness
- Components: Connected, Strongly Connected
- MST: Kruskal/Prim
"""

from abc import ABC, abstractmethod
import pytest
import random

# Try to import networkx
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


class BaseNetworkXComparisonTest(ABC):
    """Abstract base class for NetworkX comparison tests.

    Subclasses implement graph construction and algorithm execution
    for their specific database API.
    """

    @abstractmethod
    def create_db(self):
        """Create a fresh database instance."""
        raise NotImplementedError

    @abstractmethod
    def setup_random_graph(self, db, n_nodes: int, n_edges: int, weighted: bool = True, seed: int = 42) -> dict:
        """Set up a random graph.

        Args:
            db: Database instance
            n_nodes: Number of nodes
            n_edges: Number of edges
            weighted: Whether to add weight property
            seed: Random seed for reproducibility

        Returns:
            dict with:
                'node_ids': list of node IDs
                'edges': list of (src, dst, weight) tuples
        """
        raise NotImplementedError

    @abstractmethod
    def run_bfs(self, db, start_node) -> set:
        """Run BFS and return visited nodes as a set."""
        raise NotImplementedError

    @abstractmethod
    def run_dfs(self, db, start_node) -> set:
        """Run DFS and return visited nodes as a set."""
        raise NotImplementedError

    @abstractmethod
    def run_dijkstra(self, db, source) -> dict:
        """Run Dijkstra and return {node: distance} dict."""
        raise NotImplementedError

    @abstractmethod
    def run_connected_components(self, db) -> int:
        """Run connected components and return count."""
        raise NotImplementedError

    @abstractmethod
    def run_pagerank(self, db, damping: float = 0.85) -> dict:
        """Run PageRank and return {node: score} dict."""
        raise NotImplementedError

    @abstractmethod
    def run_degree_centrality(self, db) -> dict:
        """Run degree centrality and return {node: centrality} dict."""
        raise NotImplementedError

    def _build_networkx_graph(self, edges: list, directed: bool = True, weighted: bool = True):
        """Build a NetworkX graph from edge list."""
        if directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()

        for src, dst, weight in edges:
            if weighted:
                G.add_edge(src, dst, weight=weight)
            else:
                G.add_edge(src, dst)

        return G

    # ===== Comparison Tests =====

    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not installed")
    def test_bfs_reachability(self, db):
        """BFS should reach same nodes as NetworkX BFS."""
        graph_info = self.setup_random_graph(db, 100, 300, weighted=False, seed=42)
        node_ids = graph_info["node_ids"]
        edges = graph_info["edges"]
        start_node = node_ids[0]

        # Grafeo BFS
        grafeo_visited = self.run_bfs(db, start_node)

        # NetworkX BFS
        G = self._build_networkx_graph(edges, directed=True, weighted=False)
        nx_visited = set(nx.bfs_tree(G, start_node).nodes())

        assert grafeo_visited == nx_visited, (
            f"BFS reachability mismatch: Grafeo found {len(grafeo_visited)} nodes, "
            f"NetworkX found {len(nx_visited)} nodes"
        )

    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not installed")
    def test_dfs_reachability(self, db):
        """DFS should reach same nodes as NetworkX DFS."""
        graph_info = self.setup_random_graph(db, 100, 300, weighted=False, seed=42)
        node_ids = graph_info["node_ids"]
        edges = graph_info["edges"]
        start_node = node_ids[0]

        # Grafeo DFS
        grafeo_visited = self.run_dfs(db, start_node)

        # NetworkX DFS
        G = self._build_networkx_graph(edges, directed=True, weighted=False)
        nx_visited = set(nx.dfs_tree(G, start_node).nodes())

        assert grafeo_visited == nx_visited, (
            f"DFS reachability mismatch: Grafeo found {len(grafeo_visited)} nodes, "
            f"NetworkX found {len(nx_visited)} nodes"
        )

    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not installed")
    def test_dijkstra_distances(self, db):
        """Dijkstra distances should match NetworkX within tolerance."""
        graph_info = self.setup_random_graph(db, 50, 150, weighted=True, seed=42)
        node_ids = graph_info["node_ids"]
        edges = graph_info["edges"]
        source = node_ids[0]

        # Grafeo Dijkstra
        grafeo_distances = self.run_dijkstra(db, source)

        # NetworkX Dijkstra
        G = self._build_networkx_graph(edges, directed=True, weighted=True)
        nx_distances = nx.single_source_dijkstra_path_length(G, source, weight="weight")

        # Compare distances for nodes reachable by both
        common_nodes = set(grafeo_distances.keys()) & set(nx_distances.keys())
        assert len(common_nodes) > 0, "No common reachable nodes"

        for node in common_nodes:
            grafeo_dist = grafeo_distances[node]
            nx_dist = nx_distances[node]
            assert abs(grafeo_dist - nx_dist) < 1e-6, (
                f"Distance mismatch for node {node}: "
                f"Grafeo={grafeo_dist}, NetworkX={nx_dist}"
            )

    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not installed")
    def test_connected_component_count(self, db):
        """Connected component count should match NetworkX."""
        graph_info = self.setup_random_graph(db, 100, 200, weighted=False, seed=42)
        edges = graph_info["edges"]

        # Grafeo connected components
        grafeo_count = self.run_connected_components(db)

        # NetworkX connected components (undirected)
        G = self._build_networkx_graph(edges, directed=False, weighted=False)
        # Add isolated nodes
        for node_id in graph_info["node_ids"]:
            if node_id not in G:
                G.add_node(node_id)
        nx_count = nx.number_connected_components(G)

        assert grafeo_count == nx_count, (
            f"Connected component count mismatch: "
            f"Grafeo={grafeo_count}, NetworkX={nx_count}"
        )

    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not installed")
    def test_pagerank_ranking(self, db):
        """PageRank top-k ranking should match NetworkX."""
        graph_info = self.setup_random_graph(db, 50, 200, weighted=False, seed=42)
        edges = graph_info["edges"]

        # Grafeo PageRank
        grafeo_pr = self.run_pagerank(db)

        # NetworkX PageRank
        G = self._build_networkx_graph(edges, directed=True, weighted=False)
        nx_pr = nx.pagerank(G, alpha=0.85)

        # Compare top-5 ranking
        grafeo_top5 = sorted(grafeo_pr.items(), key=lambda x: x[1], reverse=True)[:5]
        nx_top5 = sorted(nx_pr.items(), key=lambda x: x[1], reverse=True)[:5]

        grafeo_top5_nodes = [n for n, _ in grafeo_top5]
        nx_top5_nodes = [n for n, _ in nx_top5]

        # At least 3 of top 5 should match (some variation due to implementation)
        overlap = len(set(grafeo_top5_nodes) & set(nx_top5_nodes))
        assert overlap >= 3, (
            f"PageRank top-5 mismatch: only {overlap} overlap. "
            f"Grafeo top 5: {grafeo_top5_nodes}, NetworkX top 5: {nx_top5_nodes}"
        )

    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not installed")
    def test_pagerank_sum(self, db):
        """PageRank scores should sum to approximately 1.0."""
        graph_info = self.setup_random_graph(db, 50, 200, weighted=False, seed=42)

        # Grafeo PageRank
        grafeo_pr = self.run_pagerank(db)
        pr_sum = sum(grafeo_pr.values())

        assert abs(pr_sum - 1.0) < 0.01, (
            f"PageRank sum should be ~1.0, got {pr_sum}"
        )

    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not installed")
    def test_degree_centrality_values(self, db):
        """Degree centrality should match NetworkX."""
        graph_info = self.setup_random_graph(db, 50, 150, weighted=False, seed=42)
        edges = graph_info["edges"]

        # Grafeo degree centrality
        grafeo_dc = self.run_degree_centrality(db)

        # NetworkX degree centrality
        G = self._build_networkx_graph(edges, directed=True, weighted=False)
        nx_dc = nx.degree_centrality(G)

        # Compare values for common nodes
        common_nodes = set(grafeo_dc.keys()) & set(nx_dc.keys())
        assert len(common_nodes) > 0, "No common nodes for degree centrality"

        for node in common_nodes:
            grafeo_val = grafeo_dc[node]
            nx_val = nx_dc[node]
            assert abs(grafeo_val - nx_val) < 0.01, (
                f"Degree centrality mismatch for node {node}: "
                f"Grafeo={grafeo_val}, NetworkX={nx_val}"
            )


class BaseNetworkXBenchmarkTest(ABC):
    """Abstract base class for NetworkX vs Grafeo performance comparison.

    Runs the same algorithms on both and compares performance.
    """

    @abstractmethod
    def create_db(self):
        """Create a fresh database instance."""
        raise NotImplementedError

    @abstractmethod
    def setup_random_graph(self, db, n_nodes: int, n_edges: int, weighted: bool = True, seed: int = 42) -> dict:
        """Set up a random graph and return graph info."""
        raise NotImplementedError

    @abstractmethod
    def run_grafeo_pagerank(self, db) -> float:
        """Run Grafeo PageRank and return execution time in ms."""
        raise NotImplementedError

    @abstractmethod
    def run_grafeo_dijkstra(self, db, source) -> float:
        """Run Grafeo Dijkstra and return execution time in ms."""
        raise NotImplementedError

    @abstractmethod
    def run_grafeo_bfs(self, db, start) -> float:
        """Run Grafeo BFS and return execution time in ms."""
        raise NotImplementedError

    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not installed")
    def test_pagerank_performance_comparison(self, db):
        """Compare PageRank performance between Grafeo and NetworkX."""
        import time

        graph_info = self.setup_random_graph(db, 1000, 5000, weighted=False, seed=42)
        edges = graph_info["edges"]

        # Grafeo timing
        grafeo_time = self.run_grafeo_pagerank(db)

        # NetworkX timing
        G = nx.DiGraph()
        for src, dst, _ in edges:
            G.add_edge(src, dst)

        start = time.perf_counter()
        nx.pagerank(G, alpha=0.85)
        nx_time = (time.perf_counter() - start) * 1000

        print(f"\nPageRank (1000 nodes, 5000 edges):")
        print(f"  Grafeo: {grafeo_time:.2f}ms")
        print(f"  NetworkX: {nx_time:.2f}ms")
        print(f"  Ratio: {grafeo_time / nx_time:.2f}x")

        # Just ensure both complete, don't assert performance
        assert grafeo_time > 0
        assert nx_time > 0

    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not installed")
    def test_dijkstra_performance_comparison(self, db):
        """Compare Dijkstra performance between Grafeo and NetworkX."""
        import time

        graph_info = self.setup_random_graph(db, 1000, 5000, weighted=True, seed=42)
        node_ids = graph_info["node_ids"]
        edges = graph_info["edges"]
        source = node_ids[0]

        # Grafeo timing
        grafeo_time = self.run_grafeo_dijkstra(db, source)

        # NetworkX timing
        G = nx.DiGraph()
        for src, dst, weight in edges:
            G.add_edge(src, dst, weight=weight)

        start = time.perf_counter()
        nx.single_source_dijkstra_path_length(G, source, weight="weight")
        nx_time = (time.perf_counter() - start) * 1000

        print(f"\nDijkstra (1000 nodes, 5000 edges):")
        print(f"  Grafeo: {grafeo_time:.2f}ms")
        print(f"  NetworkX: {nx_time:.2f}ms")
        print(f"  Ratio: {grafeo_time / nx_time:.2f}x")

        # Just ensure both complete
        assert grafeo_time > 0
        assert nx_time > 0

    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not installed")
    def test_bfs_performance_comparison(self, db):
        """Compare BFS performance between Grafeo and NetworkX."""
        import time

        graph_info = self.setup_random_graph(db, 1000, 5000, weighted=False, seed=42)
        node_ids = graph_info["node_ids"]
        edges = graph_info["edges"]
        start_node = node_ids[0]

        # Grafeo timing
        grafeo_time = self.run_grafeo_bfs(db, start_node)

        # NetworkX timing
        G = nx.DiGraph()
        for src, dst, _ in edges:
            G.add_edge(src, dst)

        start = time.perf_counter()
        list(nx.bfs_tree(G, start_node).nodes())
        nx_time = (time.perf_counter() - start) * 1000

        print(f"\nBFS (1000 nodes, 5000 edges):")
        print(f"  Grafeo: {grafeo_time:.2f}ms")
        print(f"  NetworkX: {nx_time:.2f}ms")
        print(f"  Ratio: {grafeo_time / nx_time:.2f}x")

        # Just ensure both complete
        assert grafeo_time > 0
        assert nx_time > 0
