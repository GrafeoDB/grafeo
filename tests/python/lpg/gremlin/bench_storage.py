"""Gremlin implementation of storage benchmarks.

Benchmarks storage operations using Gremlin syntax.
"""

import random
import pytest
from tests.python.bases.bench_storage import BaseBenchStorage


class BenchGremlinStorage(BaseBenchStorage):
    """Gremlin implementation of storage benchmarks.

    Note: Uses Python API for node/edge creation (faster for benchmarks).
    Gremlin queries for read operations.
    """

    def create_single_node(self, db, labels: list[str], props: dict):
        """Create a single node using Python API."""
        return db.create_node(labels, props)

    def create_edge(self, db, source_id, target_id, rel_type: str, props: dict):
        """Create a single edge using Python API."""
        return db.create_edge(source_id, target_id, rel_type, props)

    def execute_query(self, db, query: str) -> list:
        """Execute a Gremlin query and return results."""
        try:
            return list(db.execute_gremlin(query))
        except AttributeError:
            pytest.skip("Gremlin support not available")

    def full_scan_query(self, label: str, limit: int = None) -> str:
        """Gremlin full scan query."""
        query = f"g.V().hasLabel('{label}')"
        if limit:
            query += f".limit({limit})"
        return query

    def count_query(self, label: str) -> str:
        """Gremlin count query."""
        return f"g.V().hasLabel('{label}').count()"

    def filter_query(self, label: str, prop: str, op: str, value) -> str:
        """Gremlin filter query."""
        op_map = {">": "gt", "<": "lt", ">=": "gte", "<=": "lte", "=": "eq"}
        gremlin_op = op_map.get(op, "eq")
        val = f"'{value}'" if isinstance(value, str) else value
        return f"g.V().hasLabel('{label}').has('{prop}', {gremlin_op}({val}))"

    def point_lookup_query(self, label: str, prop: str, value) -> str:
        """Gremlin point lookup query."""
        val = f"'{value}'" if isinstance(value, str) else value
        return f"g.V().hasLabel('{label}').has('{prop}', {val})"

    def one_hop_query(self, from_label: str, rel_type: str, to_label: str,
                      limit: int = None) -> str:
        """Gremlin 1-hop traversal query."""
        query = f"g.V().hasLabel('{from_label}').out('{rel_type}').hasLabel('{to_label}')"
        if limit:
            query += f".limit({limit})"
        return query

    def two_hop_query(self, label: str, rel_type: str, limit: int = None) -> str:
        """Gremlin 2-hop traversal query."""
        return (
            f"g.V().hasLabel('{label}')"
            f".out('{rel_type}')"
            f".out('{rel_type}')"
            f".count()"
        )

    def aggregation_query(self, label: str, group_prop: str, agg_prop: str) -> str:
        """Gremlin aggregation query."""
        return (
            f"g.V().hasLabel('{label}')"
            f".group().by('{group_prop}')"
            f".by(values('{agg_prop}').mean())"
        )

    def sort_query(self, label: str, sort_prop: str, desc: bool = False,
                   limit: int = 100) -> str:
        """Gremlin sort query."""
        order = "desc" if desc else "asc"
        return (
            f"g.V().hasLabel('{label}')"
            f".order().by('{sort_prop}', {order})"
            f".limit({limit})"
        )

    def triangle_query(self, label: str, rel_type: str) -> str:
        """Gremlin triangle pattern query."""
        return (
            f"g.V().hasLabel('{label}').as('a')"
            f".out('{rel_type}').hasLabel('{label}').as('b')"
            f".out('{rel_type}').hasLabel('{label}').as('c')"
            f".out('{rel_type}').where(eq('a'))"
            f".count()"
        )

    def setup_social_network(self, db, num_nodes: int, avg_edges: int):
        """Set up social network graph using Python API."""
        rng = random.Random(42)
        cities = ["NYC", "LA", "Chicago", "Houston", "Phoenix"]

        node_ids = []
        for i in range(num_nodes):
            node = db.create_node(["Person"], {
                "name": f"Person{i}",
                "age": 20 + rng.randint(0, 50),
                "city": rng.choice(cities),
                "email": f"user{i}@example.com",
            })
            node_ids.append(node.id)

        target_edges = num_nodes * avg_edges
        edge_count = 0
        edge_set = set()
        attempts = 0
        max_attempts = target_edges * 3

        while edge_count < target_edges and attempts < max_attempts:
            attempts += 1
            src = rng.choice(node_ids)
            dst = rng.choice(node_ids)
            if src != dst and (src, dst) not in edge_set:
                db.create_edge(src, dst, "knows", {"since": 2000 + rng.randint(0, 24)})
                edge_set.add((src, dst))
                edge_count += 1

    def setup_clique_graph(self, db, num_cliques: int, clique_size: int):
        """Set up clique graph for triangle testing using Python API."""
        for c in range(num_cliques):
            node_ids = []
            for i in range(clique_size):
                node = db.create_node(["Node"], {"clique": c, "idx": i})
                node_ids.append(node.id)

            for i, src in enumerate(node_ids):
                for dst in node_ids[i + 1:]:
                    db.create_edge(src, dst, "connected", {})
                    db.create_edge(dst, src, "connected", {})
