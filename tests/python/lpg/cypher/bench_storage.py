"""Cypher implementation of storage benchmarks.

Benchmarks storage operations using Cypher syntax.
"""

import random
from tests.python.bases.bench_storage import BaseBenchStorage


class BenchCypherStorage(BaseBenchStorage):
    """Cypher implementation of storage benchmarks."""

    def create_single_node(self, db, labels: list[str], props: dict):
        """Create a single node using Python API."""
        return db.create_node(labels, props)

    def create_edge(self, db, source_id, target_id, rel_type: str, props: dict):
        """Create a single edge using Python API."""
        return db.create_edge(source_id, target_id, rel_type, props)

    def execute_query(self, db, query: str) -> list:
        """Execute a Cypher query and return results."""
        return list(db.execute(query))

    def full_scan_query(self, label: str, limit: int = None) -> str:
        """Cypher full scan query."""
        query = f"MATCH (n:{label}) RETURN n"
        if limit:
            query += f" LIMIT {limit}"
        return query

    def count_query(self, label: str) -> str:
        """Cypher count query."""
        return f"MATCH (n:{label}) RETURN count(n) AS cnt"

    def filter_query(self, label: str, prop: str, op: str, value) -> str:
        """Cypher filter query."""
        val = f"'{value}'" if isinstance(value, str) else value
        return f"MATCH (n:{label}) WHERE n.{prop} {op} {val} RETURN n"

    def point_lookup_query(self, label: str, prop: str, value) -> str:
        """Cypher point lookup query."""
        val = f"'{value}'" if isinstance(value, str) else value
        return f"MATCH (n:{label}) WHERE n.{prop} = {val} RETURN n"

    def one_hop_query(self, from_label: str, rel_type: str, to_label: str,
                      limit: int = None) -> str:
        """Cypher 1-hop traversal query."""
        query = f"MATCH (a:{from_label})-[:{rel_type}]->(b:{to_label}) RETURN a, b"
        if limit:
            query += f" LIMIT {limit}"
        return query

    def two_hop_query(self, label: str, rel_type: str, limit: int = None) -> str:
        """Cypher 2-hop traversal query."""
        return (
            f"MATCH (a:{label})-[:{rel_type}]->(b)-[:{rel_type}]->(c) "
            f"RETURN count(c) AS cnt"
        )

    def aggregation_query(self, label: str, group_prop: str, agg_prop: str) -> str:
        """Cypher aggregation query."""
        return (
            f"MATCH (n:{label}) "
            f"RETURN n.{group_prop}, count(n) AS cnt, avg(n.{agg_prop}) AS avg_val"
        )

    def sort_query(self, label: str, sort_prop: str, desc: bool = False,
                   limit: int = 100) -> str:
        """Cypher sort query."""
        order = "DESC" if desc else "ASC"
        return (
            f"MATCH (n:{label}) "
            f"RETURN n ORDER BY n.{sort_prop} {order} LIMIT {limit}"
        )

    def triangle_query(self, label: str, rel_type: str) -> str:
        """Cypher triangle pattern query."""
        return (
            f"MATCH (a:{label})-[:{rel_type}]->(b:{label})-[:{rel_type}]->(c:{label})"
            f"-[:{rel_type}]->(a) "
            f"RETURN count(a) AS cnt"
        )

    def setup_social_network(self, db, num_nodes: int, avg_edges: int):
        """Set up social network graph."""
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
                db.create_edge(src, dst, "KNOWS", {"since": 2000 + rng.randint(0, 24)})
                edge_set.add((src, dst))
                edge_count += 1

    def setup_clique_graph(self, db, num_cliques: int, clique_size: int):
        """Set up clique graph for triangle testing."""
        for c in range(num_cliques):
            node_ids = []
            for i in range(clique_size):
                node = db.create_node(["Node"], {"clique": c, "idx": i})
                node_ids.append(node.id)

            for i, src in enumerate(node_ids):
                for dst in node_ids[i + 1:]:
                    db.create_edge(src, dst, "CONNECTED", {})
                    db.create_edge(dst, src, "CONNECTED", {})
