"""Base class for graph algorithm tests.

This module defines test logic for graph algorithms:
- Traversal: BFS, DFS
- Shortest Path: Dijkstra, A*, Bellman-Ford
- Centrality: Degree, PageRank, Betweenness
- Components: Connected, Strongly Connected
- Community: Label Propagation, Louvain
- MST: Kruskal, Prim
"""

from abc import ABC, abstractmethod
import pytest


class BaseAlgorithmsTest(ABC):
    """Abstract base class for algorithm tests."""

    @abstractmethod
    def setup_algorithm_graph(self, db, n_nodes: int = 100, n_edges: int = 300):
        """Set up a random graph for algorithm testing.

        Args:
            db: Database instance
            n_nodes: Number of nodes
            n_edges: Number of edges

        Returns:
            dict with 'node_ids' list and optional metadata
        """
        raise NotImplementedError

    # ===== Traversal Tests =====

    def test_bfs(self, db):
        """Test BFS traversal."""
        graph_info = self.setup_algorithm_graph(db)
        node_ids = graph_info["node_ids"]
        start_node = node_ids[0]

        result = db.algorithms.bfs(start_node)
        assert len(result) > 0, "BFS should visit at least the start node"
        assert start_node in result, "BFS should include the start node"

    def test_bfs_layers(self, db):
        """Test BFS with layer information."""
        graph_info = self.setup_algorithm_graph(db)
        node_ids = graph_info["node_ids"]
        start_node = node_ids[0]

        layers = db.algorithms.bfs_layers(start_node)
        assert len(layers) > 0, "BFS layers should have at least one layer"
        assert start_node in layers[0], "Start node should be in first layer"

    def test_dfs(self, db):
        """Test DFS traversal."""
        graph_info = self.setup_algorithm_graph(db)
        node_ids = graph_info["node_ids"]
        start_node = node_ids[0]

        result = db.algorithms.dfs(start_node)
        assert len(result) > 0, "DFS should visit at least the start node"

    def test_dfs_all(self, db):
        """Test DFS that visits all nodes."""
        graph_info = self.setup_algorithm_graph(db)
        node_ids = graph_info["node_ids"]

        result = db.algorithms.dfs_all()
        unique_nodes = len(set(result))
        assert unique_nodes <= len(node_ids), "DFS all should not visit more unique nodes than exist"

    # ===== Component Tests =====

    def test_connected_components(self, db):
        """Test connected components."""
        graph_info = self.setup_algorithm_graph(db)
        node_ids = graph_info["node_ids"]

        components = db.algorithms.connected_components()
        assert len(components) == len(node_ids), "All nodes should have a component"

    def test_connected_component_count(self, db):
        """Test counting connected components."""
        graph_info = self.setup_algorithm_graph(db)

        count = db.algorithms.connected_component_count()
        assert count >= 1, "Should have at least one component"

    def test_strongly_connected_components(self, db):
        """Test strongly connected components."""
        graph_info = self.setup_algorithm_graph(db)

        scc = db.algorithms.strongly_connected_components()
        assert len(scc) >= 1, "Should have at least one SCC"

    def test_is_dag(self, db):
        """Test DAG detection."""
        graph_info = self.setup_algorithm_graph(db)

        is_dag = db.algorithms.is_dag()
        assert isinstance(is_dag, bool)

    def test_topological_sort(self, db):
        """Test topological sort."""
        graph_info = self.setup_algorithm_graph(db)
        node_ids = graph_info["node_ids"]

        topo = db.algorithms.topological_sort()
        # May return None if graph has cycle
        if topo is not None:
            assert len(topo) == len(node_ids)

    # ===== Shortest Path Tests =====

    def test_dijkstra(self, db):
        """Test Dijkstra's algorithm."""
        graph_info = self.setup_algorithm_graph(db)
        node_ids = graph_info["node_ids"]
        source = node_ids[0]

        distances = db.algorithms.dijkstra(source)
        assert len(distances) > 0, "Dijkstra should find distances to at least one node"
        assert source in distances, "Dijkstra should include source"
        assert distances[source] == 0, "Distance to source should be 0"

    def test_dijkstra_with_target(self, db):
        """Test Dijkstra with specific target."""
        graph_info = self.setup_algorithm_graph(db)
        node_ids = graph_info["node_ids"]
        source = node_ids[0]
        target = node_ids[min(10, len(node_ids) - 1)]

        result = db.algorithms.dijkstra(source, target, "weight")
        # May return None if no path exists
        if result is not None:
            dist, path = result
            assert dist >= 0, "Distance should be non-negative"
            assert len(path) >= 2, "Path should have at least source and target"

    def test_bellman_ford(self, db):
        """Test Bellman-Ford algorithm."""
        graph_info = self.setup_algorithm_graph(db)
        node_ids = graph_info["node_ids"]
        source = node_ids[0]

        result = db.algorithms.bellman_ford(source, "weight")
        assert "distances" in result
        assert "has_negative_cycle" in result
        assert isinstance(result["has_negative_cycle"], bool)

    # ===== Centrality Tests =====

    def test_degree_centrality(self, db):
        """Test degree centrality."""
        graph_info = self.setup_algorithm_graph(db)
        node_ids = graph_info["node_ids"]

        degree = db.algorithms.degree_centrality()
        assert len(degree) == len(node_ids), "Should compute centrality for all nodes"

    def test_degree_centrality_normalized(self, db):
        """Test normalized degree centrality."""
        graph_info = self.setup_algorithm_graph(db)
        node_ids = graph_info["node_ids"]

        degree_norm = db.algorithms.degree_centrality(normalized=True)
        assert len(degree_norm) == len(node_ids)
        # Normalized values should be between 0 and 1
        for v in degree_norm.values():
            assert 0 <= v <= 1, "Normalized centrality should be in [0, 1]"

    def test_pagerank(self, db):
        """Test PageRank algorithm."""
        graph_info = self.setup_algorithm_graph(db)
        node_ids = graph_info["node_ids"]

        pr = db.algorithms.pagerank()
        assert len(pr) == len(node_ids)
        pr_sum = sum(pr.values())
        assert abs(pr_sum - 1.0) < 0.01, "PageRank should sum to ~1.0"

    def test_betweenness_centrality(self, db):
        """Test betweenness centrality."""
        graph_info = self.setup_algorithm_graph(db)
        node_ids = graph_info["node_ids"]

        bc = db.algorithms.betweenness_centrality()
        assert len(bc) == len(node_ids)

    def test_closeness_centrality(self, db):
        """Test closeness centrality."""
        graph_info = self.setup_algorithm_graph(db)
        node_ids = graph_info["node_ids"]

        cc = db.algorithms.closeness_centrality()
        assert len(cc) == len(node_ids)

    # ===== Community Detection Tests =====

    def test_label_propagation(self, db):
        """Test label propagation community detection."""
        graph_info = self.setup_algorithm_graph(db)
        node_ids = graph_info["node_ids"]

        lp = db.algorithms.label_propagation()
        assert len(lp) == len(node_ids)
        n_communities = len(set(lp.values()))
        assert n_communities >= 1, "Should detect at least one community"

    def test_louvain(self, db):
        """Test Louvain community detection."""
        graph_info = self.setup_algorithm_graph(db)

        louvain = db.algorithms.louvain()
        assert "num_communities" in louvain
        assert "modularity" in louvain
        assert louvain["num_communities"] >= 1

    # ===== MST Tests =====

    def test_kruskal(self, db):
        """Test Kruskal's MST algorithm."""
        graph_info = self.setup_algorithm_graph(db)

        kruskal = db.algorithms.kruskal("weight")
        assert "edges" in kruskal
        assert "total_weight" in kruskal

    def test_prim(self, db):
        """Test Prim's MST algorithm."""
        graph_info = self.setup_algorithm_graph(db)

        prim = db.algorithms.prim("weight")
        assert "edges" in prim
        assert "total_weight" in prim

    # ===== Structure Analysis Tests =====

    def test_articulation_points(self, db):
        """Test articulation points detection."""
        graph_info = self.setup_algorithm_graph(db)

        ap = db.algorithms.articulation_points()
        assert isinstance(ap, (list, set))

    def test_bridges(self, db):
        """Test bridge detection."""
        graph_info = self.setup_algorithm_graph(db)

        bridges = db.algorithms.bridges()
        assert isinstance(bridges, (list, set))

    def test_kcore(self, db):
        """Test k-core decomposition."""
        graph_info = self.setup_algorithm_graph(db)

        kcore = db.algorithms.kcore()
        assert "max_core" in kcore or isinstance(kcore, dict)
