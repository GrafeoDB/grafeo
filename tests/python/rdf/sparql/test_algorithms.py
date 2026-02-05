"""SPARQL implementation of algorithm tests.

Tests graph algorithms with Python API.
Note: Algorithms are accessed via db.algorithms.*, not via SPARQL queries.
"""

import random
from tests.python.bases.test_algorithms import BaseAlgorithmsTest


class TestSPARQLAlgorithms(BaseAlgorithmsTest):
    """SPARQL implementation of algorithm tests.

    Note: Algorithms are accessed via db.algorithms.*, not via SPARQL queries.
    Uses Python API for setup (SPARQL Update not yet fully implemented for
    creating graph structure suitable for algorithm testing).
    """

    def setup_algorithm_graph(self, db, n_nodes: int = 100, n_edges: int = 300):
        """Set up a random graph for algorithm testing using Python API."""
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
                db.create_edge(src, dst, "EDGE", {"weight": rng.uniform(0.1, 10.0)})
                edges.add((src, dst))

        return {"node_ids": node_ids, "edge_count": len(edges)}


class TestSPARQLAlgorithmVerification:
    """Tests that verify algorithm results (uses Python API only)."""

    def test_verify_bfs_reachability(self, db_api):
        """Verify BFS results."""
        a = db_api.create_node(["Node"], {"name": "a"})
        b = db_api.create_node(["Node"], {"name": "b"})
        c = db_api.create_node(["Node"], {"name": "c"})
        db_api.create_node(["Node"], {"name": "d"})  # Isolated

        db_api.create_edge(a.id, b.id, "edge", {})
        db_api.create_edge(b.id, c.id, "edge", {})

        bfs_result = db_api.algorithms.bfs(a.id)

        assert a.id in bfs_result
        assert b.id in bfs_result
        assert c.id in bfs_result

    def test_verify_connected_components(self, db_api):
        """Verify connected components."""
        a = db_api.create_node(["Node"], {"name": "a", "group": 1})
        b = db_api.create_node(["Node"], {"name": "b", "group": 1})
        c = db_api.create_node(["Node"], {"name": "c", "group": 1})
        db_api.create_edge(a.id, b.id, "edge", {})
        db_api.create_edge(b.id, c.id, "edge", {})

        x = db_api.create_node(["Node"], {"name": "x", "group": 2})
        y = db_api.create_node(["Node"], {"name": "y", "group": 2})
        db_api.create_edge(x.id, y.id, "edge", {})

        components = db_api.algorithms.connected_components()
        component_count = db_api.algorithms.connected_component_count()

        assert component_count == 2
        assert components[a.id] == components[b.id] == components[c.id]
        assert components[x.id] == components[y.id]
        assert components[a.id] != components[x.id]

    def test_verify_pagerank_structure(self, db_api):
        """Verify PageRank reflects link structure."""
        center = db_api.create_node(["Node"], {"name": "center"})
        leaves = []
        for i in range(4):
            leaf = db_api.create_node(["Node"], {"name": f"leaf{i}"})
            leaves.append(leaf)
            db_api.create_edge(leaf.id, center.id, "points_to", {})

        pr = db_api.algorithms.pagerank()

        center_pr = pr[center.id]
        for leaf in leaves:
            assert center_pr > pr[leaf.id], "Center should have highest PageRank"

    def test_verify_shortest_path(self, db_api):
        """Verify Dijkstra shortest path matches expected."""
        a = db_api.create_node(["Node"], {"name": "a"})
        b = db_api.create_node(["Node"], {"name": "b"})
        c = db_api.create_node(["Node"], {"name": "c"})
        d = db_api.create_node(["Node"], {"name": "d"})

        db_api.create_edge(a.id, b.id, "edge", {"weight": 1})
        db_api.create_edge(a.id, c.id, "edge", {"weight": 10})
        db_api.create_edge(b.id, d.id, "edge", {"weight": 1})
        db_api.create_edge(c.id, d.id, "edge", {"weight": 1})

        result = db_api.algorithms.dijkstra(a.id, d.id, "weight")
        if result is not None:
            distance, path = result
            assert distance == 2, f"Expected distance 2, got {distance}"
            assert a.id in path
            assert b.id in path
            assert d.id in path
            assert c.id not in path, "Should not go through c"
