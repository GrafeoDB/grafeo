"""SPARQL implementation of storage benchmarks.

Benchmarks storage operations using SPARQL syntax.
Note: Uses Python API for node/edge creation (faster for benchmarks).
"""

import random
import pytest
from tests.python.bases.bench_storage import BaseBenchStorage


class BenchSPARQLStorage(BaseBenchStorage):
    """SPARQL implementation of storage benchmarks.

    Note: Uses Python API for node/edge creation (faster for benchmarks).
    SPARQL queries for read operations.
    """

    def create_single_node(self, db, labels: list[str], props: dict):
        """Create a single node using Python API."""
        return db.create_node(labels, props)

    def create_edge(self, db, source_id, target_id, rel_type: str, props: dict):
        """Create a single edge using Python API."""
        return db.create_edge(source_id, target_id, rel_type, props)

    def execute_query(self, db, query: str) -> list:
        """Execute a SPARQL query and return results."""
        try:
            return list(db.execute_sparql(query))
        except AttributeError:
            pytest.skip("SPARQL support not available")

    def full_scan_query(self, label: str, limit: int = None) -> str:
        """SPARQL full scan query using rdf:type."""
        limit_clause = f"LIMIT {limit}" if limit else ""
        return f"""
            SELECT ?s ?name ?age WHERE {{
                ?s a <http://example.org/{label}> .
                OPTIONAL {{ ?s <http://example.org/name> ?name }}
                OPTIONAL {{ ?s <http://example.org/age> ?age }}
            }} {limit_clause}
        """

    def count_query(self, label: str) -> str:
        """SPARQL count query."""
        return f"""
            SELECT (COUNT(?s) AS ?cnt) WHERE {{
                ?s a <http://example.org/{label}> .
            }}
        """

    def filter_query(self, label: str, prop: str, op: str, value) -> str:
        """SPARQL filter query."""
        sparql_op = {"=": "=", ">": ">", "<": "<", ">=": ">=", "<=": "<="}.get(op, "=")
        val = f'"{value}"' if isinstance(value, str) else value
        return f"""
            SELECT ?s WHERE {{
                ?s a <http://example.org/{label}> .
                ?s <http://example.org/{prop}> ?val .
                FILTER(?val {sparql_op} {val})
            }}
        """

    def point_lookup_query(self, label: str, prop: str, value) -> str:
        """SPARQL point lookup query."""
        val = f'"{value}"' if isinstance(value, str) else value
        return f"""
            SELECT ?s WHERE {{
                ?s a <http://example.org/{label}> .
                ?s <http://example.org/{prop}> {val} .
            }}
        """

    def one_hop_query(self, from_label: str, rel_type: str, to_label: str,
                      limit: int = None) -> str:
        """SPARQL 1-hop traversal query using property paths."""
        limit_clause = f"LIMIT {limit}" if limit else ""
        return f"""
            SELECT ?s ?t WHERE {{
                ?s a <http://example.org/{from_label}> .
                ?s <http://example.org/{rel_type}> ?t .
                ?t a <http://example.org/{to_label}> .
            }} {limit_clause}
        """

    def two_hop_query(self, label: str, rel_type: str, limit: int = None) -> str:
        """SPARQL 2-hop traversal query."""
        return f"""
            SELECT ?s ?hop2 WHERE {{
                ?s a <http://example.org/{label}> .
                ?s <http://example.org/{rel_type}> ?hop1 .
                ?hop1 <http://example.org/{rel_type}> ?hop2 .
            }}
        """

    def aggregation_query(self, label: str, group_prop: str, agg_prop: str) -> str:
        """SPARQL aggregation query."""
        return f"""
            SELECT ?{group_prop} (COUNT(?s) AS ?count) (AVG(?{agg_prop}Val) AS ?avg_{agg_prop})
            WHERE {{
                ?s a <http://example.org/{label}> .
                ?s <http://example.org/{group_prop}> ?{group_prop} .
                ?s <http://example.org/{agg_prop}> ?{agg_prop}Val .
            }}
            GROUP BY ?{group_prop}
        """

    def sort_query(self, label: str, sort_prop: str, desc: bool = False,
                   limit: int = 100) -> str:
        """SPARQL sort query."""
        order = "DESC" if desc else "ASC"
        return f"""
            SELECT ?s ?{sort_prop} WHERE {{
                ?s a <http://example.org/{label}> .
                ?s <http://example.org/{sort_prop}> ?{sort_prop} .
            }}
            ORDER BY {order}(?{sort_prop})
            LIMIT {limit}
        """

    def triangle_query(self, label: str, rel_type: str) -> str:
        """SPARQL triangle pattern query."""
        return f"""
            SELECT ?a ?b ?c WHERE {{
                ?a a <http://example.org/{label}> .
                ?b a <http://example.org/{label}> .
                ?c a <http://example.org/{label}> .
                ?a <http://example.org/{rel_type}> ?b .
                ?b <http://example.org/{rel_type}> ?c .
                ?c <http://example.org/{rel_type}> ?a .
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
