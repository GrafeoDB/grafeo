"""GraphQL implementation of storage benchmarks.

Benchmarks storage operations using GraphQL syntax.
"""

import random
import pytest
from tests.python.bases.bench_storage import BaseBenchStorage


class BenchGraphQLStorage(BaseBenchStorage):
    """GraphQL implementation of storage benchmarks.

    Note: Uses Python API for node/edge creation (faster for benchmarks).
    GraphQL queries for read operations.
    """

    def create_single_node(self, db, labels: list[str], props: dict):
        """Create a single node using Python API."""
        return db.create_node(labels, props)

    def create_edge(self, db, source_id, target_id, rel_type: str, props: dict):
        """Create a single edge using Python API."""
        return db.create_edge(source_id, target_id, rel_type, props)

    def execute_query(self, db, query: str) -> list:
        """Execute a GraphQL query and return results."""
        try:
            return list(db.execute_graphql(query))
        except AttributeError:
            pytest.skip("GraphQL support not available")

    def full_scan_query(self, label: str, limit: int = None) -> str:
        """GraphQL full scan query."""
        limit_arg = f"(limit: {limit})" if limit else ""
        return f"""
            query {{
                {label.lower()}{limit_arg} {{
                    id
                    name
                    age
                }}
            }}
        """

    def count_query(self, label: str) -> str:
        """GraphQL count query."""
        return f"""
            query {{
                {label.lower()}Count
            }}
        """

    def filter_query(self, label: str, prop: str, op: str, value) -> str:
        """GraphQL filter query."""
        val = f'"{value}"' if isinstance(value, str) else value
        # GraphQL uses comparison operators via arguments
        op_name = {"=": "eq", ">": "gt", "<": "lt", ">=": "gte", "<=": "lte"}.get(op, "eq")
        return f"""
            query {{
                {label.lower()}(filter: {{ {prop}: {{ {op_name}: {val} }} }}) {{
                    id
                }}
            }}
        """

    def point_lookup_query(self, label: str, prop: str, value) -> str:
        """GraphQL point lookup query."""
        val = f'"{value}"' if isinstance(value, str) else value
        return f"""
            query {{
                {label.lower()}({prop}: {val}) {{
                    id
                }}
            }}
        """

    def one_hop_query(self, from_label: str, rel_type: str, to_label: str,
                      limit: int = None) -> str:
        """GraphQL 1-hop traversal query."""
        limit_arg = f"(limit: {limit})" if limit else ""
        return f"""
            query {{
                {from_label.lower()}{limit_arg} {{
                    id
                    {rel_type} {{
                        id
                    }}
                }}
            }}
        """

    def two_hop_query(self, label: str, rel_type: str, limit: int = None) -> str:
        """GraphQL 2-hop traversal query."""
        return f"""
            query {{
                {label.lower()} {{
                    {rel_type} {{
                        {rel_type} {{
                            id
                        }}
                    }}
                }}
            }}
        """

    def aggregation_query(self, label: str, group_prop: str, agg_prop: str) -> str:
        """GraphQL aggregation query."""
        return f"""
            query {{
                {label.lower()}Aggregate(groupBy: "{group_prop}") {{
                    {group_prop}
                    count
                    avg_{agg_prop}
                }}
            }}
        """

    def sort_query(self, label: str, sort_prop: str, desc: bool = False,
                   limit: int = 100) -> str:
        """GraphQL sort query."""
        order = "DESC" if desc else "ASC"
        return f"""
            query {{
                {label.lower()}(orderBy: {{ {sort_prop}: {order} }}, limit: {limit}) {{
                    id
                    {sort_prop}
                }}
            }}
        """

    def triangle_query(self, label: str, rel_type: str) -> str:
        """GraphQL triangle pattern query (limited support)."""
        return f"""
            query {{
                {label.lower()} {{
                    {rel_type} {{
                        {rel_type} {{
                            {rel_type} {{
                                id
                            }}
                        }}
                    }}
                }}
            }}
        """

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
                db.create_edge(src, dst, "KNOWS", {"since": 2000 + rng.randint(0, 24)})
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
                    db.create_edge(src, dst, "CONNECTED", {})
                    db.create_edge(dst, src, "CONNECTED", {})
