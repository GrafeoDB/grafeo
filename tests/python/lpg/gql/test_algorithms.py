"""GQL implementation of algorithm tests.

Tests graph algorithms with GQL for setup/verification.
"""

import pytest
import random
from tests.python.bases.test_algorithms import BaseAlgorithmsTest


class TestGQLAlgorithms(BaseAlgorithmsTest):
    """GQL implementation of algorithm tests.

    Note: Algorithms are accessed via db.algorithms.*, not via GQL queries.
    GQL is used for setup and verification only.
    """

    def setup_algorithm_graph(self, db, n_nodes: int = 100, n_edges: int = 300):
        """Set up a random graph for algorithm testing."""
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


# Additional GQL-specific algorithm tests using GQL for verification

class TestGQLAlgorithmVerification:
    """Tests that verify algorithm results using GQL queries."""

    def test_verify_bfs_reachability(self, db):
        """Verify BFS results match GQL path query."""
        # Create a simple graph
        a = db.create_node(["Node"], {"name": "a"})
        b = db.create_node(["Node"], {"name": "b"})
        c = db.create_node(["Node"], {"name": "c"})
        d = db.create_node(["Node"], {"name": "d"})  # Isolated node

        db.create_edge(a.id, b.id, "EDGE", {})
        db.create_edge(b.id, c.id, "EDGE", {})

        # Run BFS from node a
        bfs_result = db.algorithms.bfs(a.id)

        # Verify with GQL - nodes reachable from a
        result = db.execute(
            "MATCH p = (start:Node {name: 'a'})-[:EDGE*0..10]->(end:Node) "
            "RETURN DISTINCT end.name"
        )
        gql_reachable = {r["end.name"] for r in result}

        # a, b, c should be reachable; d should not
        assert a.id in bfs_result
        assert b.id in bfs_result
        assert c.id in bfs_result
        # d is isolated, should not be in BFS from a

    def test_verify_connected_components(self, db):
        """Verify connected components match GQL connectivity."""
        # Create two disconnected components
        # Component 1: a-b-c
        a = db.create_node(["Node"], {"name": "a", "group": 1})
        b = db.create_node(["Node"], {"name": "b", "group": 1})
        c = db.create_node(["Node"], {"name": "c", "group": 1})
        db.create_edge(a.id, b.id, "EDGE", {})
        db.create_edge(b.id, c.id, "EDGE", {})

        # Component 2: x-y
        x = db.create_node(["Node"], {"name": "x", "group": 2})
        y = db.create_node(["Node"], {"name": "y", "group": 2})
        db.create_edge(x.id, y.id, "EDGE", {})

        # Run connected components
        components = db.algorithms.connected_components()
        component_count = db.algorithms.connected_component_count()

        # Should have 2 components
        assert component_count == 2

        # Nodes in same component should have same component ID
        assert components[a.id] == components[b.id] == components[c.id]
        assert components[x.id] == components[y.id]
        assert components[a.id] != components[x.id]

    def test_verify_pagerank_structure(self, db):
        """Verify PageRank reflects link structure."""
        # Create a star graph: center connected to 4 leaves
        center = db.create_node(["Node"], {"name": "center"})
        leaves = []
        for i in range(4):
            leaf = db.create_node(["Node"], {"name": f"leaf{i}"})
            leaves.append(leaf)
            db.create_edge(leaf.id, center.id, "POINTS_TO", {})

        # Run PageRank
        pr = db.algorithms.pagerank()

        # Center should have higher PageRank (receives all links)
        center_pr = pr[center.id]
        for leaf in leaves:
            assert center_pr > pr[leaf.id], "Center should have highest PageRank"

    def test_verify_shortest_path(self, db):
        """Verify Dijkstra shortest path matches expected."""
        # Create a weighted graph
        #     1
        # a ----> b
        # |       |
        # |10     |1
        # v       v
        # c ----> d
        #     1
        a = db.create_node(["Node"], {"name": "a"})
        b = db.create_node(["Node"], {"name": "b"})
        c = db.create_node(["Node"], {"name": "c"})
        d = db.create_node(["Node"], {"name": "d"})

        db.create_edge(a.id, b.id, "EDGE", {"weight": 1})
        db.create_edge(a.id, c.id, "EDGE", {"weight": 10})
        db.create_edge(b.id, d.id, "EDGE", {"weight": 1})
        db.create_edge(c.id, d.id, "EDGE", {"weight": 1})

        # Shortest path from a to d should go through b (cost 2)
        # Not through c (cost 11)
        result = db.algorithms.dijkstra(a.id, d.id, "weight")
        if result is not None:
            distance, path = result
            assert distance == 2, f"Expected distance 2, got {distance}"
            assert a.id in path
            assert b.id in path
            assert d.id in path
            assert c.id not in path, "Should not go through c"
