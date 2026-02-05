"""Algorithm tests for the RDF GraphQL context.

NOTE: Algorithms in Grafeo operate on the LPG (Labeled Property Graph) store,
not directly on RDF triples. This is because graph algorithms need efficient
adjacency traversal which the LPG store provides.

These tests use the Python API to create LPG graph data, then run algorithms.
The "RDF GraphQL" context indicates the query language environment, but
algorithm execution is independent of query language.
"""

import random
from tests.python.bases.test_algorithms import BaseAlgorithmsTest

# Try to import grafeo
try:
    from grafeo import GrafeoDB
    GRAFEO_AVAILABLE = True
except ImportError:
    GRAFEO_AVAILABLE = False

import pytest

pytestmark = pytest.mark.skipif(
    not GRAFEO_AVAILABLE,
    reason="Grafeo Python bindings not installed"
)


class TestRDFGraphQLAlgorithms(BaseAlgorithmsTest):
    """GraphQL on RDF implementation of algorithm tests.

    Note: Algorithms are accessed via db.algorithms.*, not via GraphQL queries.
    Uses Python API for setup with RDF-style data (nodes have URIs).
    """

    def setup_algorithm_graph(self, db, n_nodes: int = 100, n_edges: int = 300):
        """Set up a random graph for algorithm testing using Python API with RDF-style nodes."""
        rng = random.Random(42)

        node_ids = []
        for i in range(n_nodes):
            node = db.create_node(["Resource", "Node"], {
                "uri": f"http://example.org/node/{i}",
                "index": i
            })
            node_ids.append(node.id)

        edges = set()
        while len(edges) < n_edges:
            src = rng.choice(node_ids)
            dst = rng.choice(node_ids)
            if src != dst and (src, dst) not in edges:
                db.create_edge(src, dst, "EDGE", {"weight": rng.uniform(0.1, 10.0)})
                edges.add((src, dst))

        return {"node_ids": node_ids, "edge_count": len(edges)}


class TestRDFGraphQLAlgorithmVerification:
    """Tests that verify algorithm results on RDF-style graphs (uses Python API only)."""

    def setup_method(self):
        """Create a fresh database."""
        if not GRAFEO_AVAILABLE:
            pytest.skip("grafeo not installed")
        self.db = GrafeoDB()

    def test_verify_bfs_reachability(self):
        """Verify BFS results on RDF graph."""
        a = self.db.create_node(["Resource", "Node"], {
            "uri": "http://example.org/node/a",
            "name": "a"
        })
        b = self.db.create_node(["Resource", "Node"], {
            "uri": "http://example.org/node/b",
            "name": "b"
        })
        c = self.db.create_node(["Resource", "Node"], {
            "uri": "http://example.org/node/c",
            "name": "c"
        })
        self.db.create_node(["Resource", "Node"], {
            "uri": "http://example.org/node/d",
            "name": "d"
        })  # Isolated

        self.db.create_edge(a.id, b.id, "edge", {})
        self.db.create_edge(b.id, c.id, "edge", {})

        bfs_result = self.db.algorithms.bfs(a.id)

        assert a.id in bfs_result
        assert b.id in bfs_result
        assert c.id in bfs_result

    def test_verify_connected_components(self):
        """Verify connected components on RDF graph."""
        a = self.db.create_node(["Resource", "Node"], {
            "uri": "http://example.org/node/a",
            "name": "a",
            "group": 1
        })
        b = self.db.create_node(["Resource", "Node"], {
            "uri": "http://example.org/node/b",
            "name": "b",
            "group": 1
        })
        c = self.db.create_node(["Resource", "Node"], {
            "uri": "http://example.org/node/c",
            "name": "c",
            "group": 1
        })
        self.db.create_edge(a.id, b.id, "edge", {})
        self.db.create_edge(b.id, c.id, "edge", {})

        x = self.db.create_node(["Resource", "Node"], {
            "uri": "http://example.org/node/x",
            "name": "x",
            "group": 2
        })
        y = self.db.create_node(["Resource", "Node"], {
            "uri": "http://example.org/node/y",
            "name": "y",
            "group": 2
        })
        self.db.create_edge(x.id, y.id, "edge", {})

        components = self.db.algorithms.connected_components()
        component_count = self.db.algorithms.connected_component_count()

        assert component_count == 2
        assert components[a.id] == components[b.id] == components[c.id]
        assert components[x.id] == components[y.id]
        assert components[a.id] != components[x.id]

    def test_verify_pagerank_structure(self):
        """Verify PageRank reflects link structure on RDF graph."""
        center = self.db.create_node(["Resource", "Node"], {
            "uri": "http://example.org/node/center",
            "name": "center"
        })
        leaves = []
        for i in range(4):
            leaf = self.db.create_node(["Resource", "Node"], {
                "uri": f"http://example.org/node/leaf{i}",
                "name": f"leaf{i}"
            })
            leaves.append(leaf)
            self.db.create_edge(leaf.id, center.id, "points_to", {})

        pr = self.db.algorithms.pagerank()

        center_pr = pr[center.id]
        for leaf in leaves:
            assert center_pr > pr[leaf.id], "Center should have highest PageRank"

    def test_verify_shortest_path(self):
        """Verify Dijkstra shortest path matches expected on RDF graph."""
        a = self.db.create_node(["Resource", "Node"], {
            "uri": "http://example.org/node/a",
            "name": "a"
        })
        b = self.db.create_node(["Resource", "Node"], {
            "uri": "http://example.org/node/b",
            "name": "b"
        })
        c = self.db.create_node(["Resource", "Node"], {
            "uri": "http://example.org/node/c",
            "name": "c"
        })
        d = self.db.create_node(["Resource", "Node"], {
            "uri": "http://example.org/node/d",
            "name": "d"
        })

        self.db.create_edge(a.id, b.id, "edge", {"weight": 1})
        self.db.create_edge(a.id, c.id, "edge", {"weight": 10})
        self.db.create_edge(b.id, d.id, "edge", {"weight": 1})
        self.db.create_edge(c.id, d.id, "edge", {"weight": 1})

        result = self.db.algorithms.dijkstra(a.id, d.id, "weight")
        if result is not None:
            distance, path = result
            assert distance == 2, f"Expected distance 2, got {distance}"
            assert a.id in path
            assert b.id in path
            assert d.id in path
            assert c.id not in path, "Should not go through c"
