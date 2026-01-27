"""
Performance benchmarks for Graphos.

Run with: python tests/python/benchmark_graphos.py

This script benchmarks:
1. Node/edge insertion throughput
2. Query performance (point lookups, traversals, pattern matching)
3. Scalability with dataset size
4. Various graph structures (social network, tree, clique)
"""

import time
import statistics
import gc
import sys
from dataclasses import dataclass
from typing import Any, Callable, Optional
from contextlib import contextmanager

from synthetic_data import (
    SocialNetworkGenerator,
    LDBCLikeGenerator,
    TreeGenerator,
    CliqueGenerator,
    RandomGraphGenerator,
    load_data_into_db,
)

# Try to import graphos
try:
    from graphos import GraphosDB
    GRAPHOS_AVAILABLE = True
except ImportError:
    GRAPHOS_AVAILABLE = False
    print("ERROR: Graphos Python bindings not installed.")
    print("Run: cd crates/graphos-python && maturin develop")
    sys.exit(1)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""
    name: str
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    iterations: int
    ops_per_second: float
    extra_info: dict


class BenchmarkSuite:
    """Collection of benchmarks with timing utilities."""

    def __init__(self, warmup_iterations: int = 2, iterations: int = 5):
        self.warmup_iterations = warmup_iterations
        self.iterations = iterations
        self.results: list[BenchmarkResult] = []
        self._last_time: float = 0.0

    @contextmanager
    def timer(self):
        """Context manager for timing operations."""
        gc.collect()
        start = time.perf_counter()
        yield
        end = time.perf_counter()
        self._last_time = (end - start) * 1000  # Convert to ms

    def benchmark(
        self,
        name: str,
        setup: Callable[[], Any],
        operation: Callable[[Any], None],
        teardown: Optional[Callable[[Any], None]] = None,
        ops_count: int = 1,
    ) -> BenchmarkResult:
        """
        Run a benchmark with setup, operation, and optional teardown.

        Args:
            name: Name of the benchmark
            setup: Function to set up the benchmark, returns context
            operation: Function to benchmark, receives context
            teardown: Optional cleanup function
            ops_count: Number of operations in each iteration (for ops/sec calculation)
        """
        print(f"  Running: {name}...", end=" ", flush=True)

        times = []

        # Warmup
        for _ in range(self.warmup_iterations):
            ctx = setup()
            with self.timer():
                operation(ctx)
            if teardown:
                teardown(ctx)

        # Actual benchmark
        for _ in range(self.iterations):
            ctx = setup()
            gc.collect()
            with self.timer():
                operation(ctx)
            times.append(self._last_time)
            if teardown:
                teardown(ctx)

        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        min_time = min(times)
        max_time = max(times)
        ops_per_sec = (ops_count / (mean_time / 1000)) if mean_time > 0 else 0

        result = BenchmarkResult(
            name=name,
            mean_time_ms=mean_time,
            std_time_ms=std_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            iterations=self.iterations,
            ops_per_second=ops_per_sec,
            extra_info={},
        )

        self.results.append(result)
        print(f"{mean_time:.2f}ms (ops/s: {ops_per_sec:.0f})")
        return result

    def print_results(self):
        """Print a summary of all benchmark results."""
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)

        max_name_len = max(len(r.name) for r in self.results) if self.results else 20

        print(f"{'Benchmark':<{max_name_len}} | {'Mean (ms)':<12} | {'Std (ms)':<10} | {'Ops/sec':<12}")
        print("-" * 80)

        for r in self.results:
            print(
                f"{r.name:<{max_name_len}} | "
                f"{r.mean_time_ms:<12.2f} | "
                f"{r.std_time_ms:<10.2f} | "
                f"{r.ops_per_second:<12.0f}"
            )


def run_insertion_benchmarks(suite: BenchmarkSuite):
    """Benchmark node and edge insertion throughput."""
    print("\n--- Insertion Benchmarks ---")

    # Benchmark: Single node insertion
    def setup_single_node():
        return GraphosDB()

    def single_node_insert(db):
        for i in range(1000):
            db.create_node(["Person"], {"name": f"Person{i}", "age": 25 + i % 50})

    suite.benchmark(
        "Insert 1K nodes (single)",
        setup_single_node,
        single_node_insert,
        ops_count=1000,
    )

    # Benchmark: Node insertion with properties
    def insert_nodes_with_props(db):
        for i in range(1000):
            db.create_node(["Person", "Employee"], {
                "name": f"Person{i}",
                "age": 25 + i % 50,
                "email": f"person{i}@example.com",
                "city": ["NYC", "LA", "Chicago"][i % 3],
                "salary": 50000 + i * 100,
            })

    suite.benchmark(
        "Insert 1K nodes (5 properties)",
        setup_single_node,
        insert_nodes_with_props,
        ops_count=1000,
    )

    # Benchmark: Edge insertion
    def setup_edge_insert():
        db = GraphosDB()
        node_ids = []
        for i in range(100):
            node = db.create_node(["Node"], {"idx": i})
            node_ids.append(node.id)
        return (db, node_ids)

    def edge_insert(ctx):
        db, node_ids = ctx
        for i in range(len(node_ids)):
            for j in range(i + 1, min(i + 10, len(node_ids))):
                db.create_edge(node_ids[i], node_ids[j], "CONNECTED", {"weight": i + j})

    suite.benchmark(
        "Insert edges (100 nodes, ~450 edges)",
        setup_edge_insert,
        edge_insert,
        ops_count=450,
    )


def run_query_benchmarks(suite: BenchmarkSuite):
    """Benchmark query performance."""
    print("\n--- Query Benchmarks ---")

    # Setup: Create a social network
    def setup_social_network():
        db = GraphosDB()
        gen = SocialNetworkGenerator(num_nodes=1000, avg_edges_per_node=10, seed=42)
        load_data_into_db(db, gen)
        return db

    # Benchmark: Full scan
    def full_scan(db):
        result = db.execute("MATCH (n:Person) RETURN count(n)")
        list(result)

    suite.benchmark(
        "Full scan (1K nodes)",
        setup_social_network,
        full_scan,
        ops_count=1000,
    )

    # Benchmark: Filtered scan
    def filtered_scan(db):
        result = db.execute("MATCH (n:Person) WHERE n.age > 50 RETURN count(n)")
        list(result)

    suite.benchmark(
        "Filtered scan (age > 50)",
        setup_social_network,
        filtered_scan,
        ops_count=1,
    )

    # Benchmark: Point lookup by property
    def point_lookup(db):
        for i in range(100):
            result = db.execute(f"MATCH (n:Person) WHERE n.email = 'user{i}@example.com' RETURN n")
            list(result)

    suite.benchmark(
        "Point lookup x100",
        setup_social_network,
        point_lookup,
        ops_count=100,
    )

    # Benchmark: 1-hop traversal
    def one_hop_traversal(db):
        result = db.execute(
            "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN count(b)"
        )
        list(result)

    suite.benchmark(
        "1-hop traversal (count edges)",
        setup_social_network,
        one_hop_traversal,
        ops_count=1,
    )

    # Benchmark: 2-hop traversal
    def two_hop_traversal(db):
        result = db.execute(
            "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) "
            "RETURN count(c)"
        )
        list(result)

    suite.benchmark(
        "2-hop traversal (count paths)",
        setup_social_network,
        two_hop_traversal,
        ops_count=1,
    )

    # Benchmark: Aggregation
    def aggregation_query(db):
        result = db.execute(
            "MATCH (n:Person) RETURN n.city, count(n), avg(n.age) "
            "ORDER BY count(n) DESC"
        )
        list(result)

    suite.benchmark(
        "Aggregation (group by city)",
        setup_social_network,
        aggregation_query,
        ops_count=1,
    )


def run_pattern_benchmarks(suite: BenchmarkSuite):
    """Benchmark pattern matching queries."""
    print("\n--- Pattern Matching Benchmarks ---")

    # Setup: Create a clique graph for triangle counting
    def setup_clique():
        db = GraphosDB()
        gen = CliqueGenerator(num_cliques=10, clique_size=10, inter_clique_edges=5, seed=42)
        load_data_into_db(db, gen)
        return db

    # Benchmark: Triangle counting
    def triangle_count(db):
        result = db.execute(
            "MATCH (a)-[:CONNECTED]->(b)-[:CONNECTED]->(c)-[:CONNECTED]->(a) "
            "RETURN count(a)"  # Each triangle counted 6 times
        )
        list(result)

    suite.benchmark(
        "Triangle count (100 nodes, 10 cliques)",
        setup_clique,
        triangle_count,
        ops_count=1,
    )

    # Setup: Create a tree for hierarchical queries
    def setup_tree():
        db = GraphosDB()
        gen = TreeGenerator(depth=5, branching_factor=3, seed=42)
        load_data_into_db(db, gen)
        return db

    # Benchmark: Find all leaves
    def find_leaves(db):
        result = db.execute(
            "MATCH (n:TreeNode) "
            "WHERE NOT EXISTS { MATCH (n)-[:PARENT_OF]->() } "
            "RETURN count(n)"
        )
        list(result)

    suite.benchmark(
        "Find leaves (tree depth=5, bf=3)",
        setup_tree,
        find_leaves,
        ops_count=1,
    )

    # Benchmark: Path to root
    def path_to_root(db):
        result = db.execute(
            "MATCH path = (leaf:TreeNode)-[:PARENT_OF*]->(root:Root) "
            "RETURN count(path)"
        )
        list(result)

    suite.benchmark(
        "Paths to root (variable length)",
        setup_tree,
        path_to_root,
        ops_count=1,
    )


def run_scalability_benchmarks(suite: BenchmarkSuite):
    """Benchmark performance at different scales."""
    print("\n--- Scalability Benchmarks ---")

    scales = [100, 500, 1000, 2000]

    for scale in scales:
        def setup_scale(s=scale):
            db = GraphosDB()
            gen = SocialNetworkGenerator(num_nodes=s, avg_edges_per_node=5, seed=42)
            load_data_into_db(db, gen)
            return db

        def full_scan(db):
            result = db.execute("MATCH (n:Person) RETURN count(n)")
            list(result)

        suite.benchmark(
            f"Full scan ({scale} nodes)",
            setup_scale,
            full_scan,
            ops_count=scale,
        )

    # Edge scaling
    for scale in scales:
        def setup_edge_scale(s=scale):
            db = GraphosDB()
            gen = SocialNetworkGenerator(num_nodes=s, avg_edges_per_node=10, seed=42)
            load_data_into_db(db, gen)
            return db

        def edge_scan(db):
            result = db.execute(
                "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN count(b)"
            )
            list(result)

        suite.benchmark(
            f"Edge scan ({scale} nodes, ~{scale*5} edges)",
            setup_edge_scale,
            edge_scan,
            ops_count=scale * 5,
        )


def run_ldbc_benchmarks(suite: BenchmarkSuite):
    """Benchmark LDBC-like interactive queries."""
    print("\n--- LDBC-like Query Benchmarks ---")

    def setup_ldbc():
        db = GraphosDB()
        gen = LDBCLikeGenerator(scale_factor=0.5, seed=42)
        load_data_into_db(db, gen)
        return db

    # IC1: Friends of a person with given name
    def ic1_friends(db):
        result = db.execute(
            "MATCH (p:Person)-[:KNOWS]->(friend:Person) "
            "WHERE p.name STARTS WITH 'Alice' "
            "RETURN friend.name, friend.age"
        )
        list(result)

    suite.benchmark(
        "LDBC IC1: Friends of Person",
        setup_ldbc,
        ic1_friends,
        ops_count=1,
    )

    # IC2: Recent messages (simulated with friends of friends)
    def ic2_fof(db):
        result = db.execute(
            "MATCH (p:Person)-[:KNOWS]->()-[:KNOWS]->(fof:Person) "
            "WHERE p.name STARTS WITH 'Bob' "
            "RETURN DISTINCT fof.name LIMIT 20"
        )
        list(result)

    suite.benchmark(
        "LDBC IC2: Friends of Friends",
        setup_ldbc,
        ic2_fof,
        ops_count=1,
    )

    # IC5: Groups of friends (simulated)
    def ic5_common_friends(db):
        result = db.execute(
            "MATCH (p:Person)-[:KNOWS]->(friend:Person)<-[:KNOWS]-(other:Person) "
            "WHERE p.name STARTS WITH 'Charlie' AND p <> other "
            "RETURN other.name, count(friend) AS mutual "
            "ORDER BY mutual DESC LIMIT 10"
        )
        list(result)

    suite.benchmark(
        "LDBC IC5: Common Friends",
        setup_ldbc,
        ic5_common_friends,
        ops_count=1,
    )

    # IC11: Job referrals (person -> works_at -> company <- works_at <- person)
    def ic11_coworkers(db):
        result = db.execute(
            "MATCH (p:Person)-[:WORKS_AT]->(c:Company)<-[:WORKS_AT]-(coworker:Person) "
            "WHERE p.name STARTS WITH 'Diana' AND p <> coworker "
            "RETURN c.name, count(coworker) AS colleagues "
            "ORDER BY colleagues DESC LIMIT 10"
        )
        list(result)

    suite.benchmark(
        "LDBC IC11: Coworkers",
        setup_ldbc,
        ic11_coworkers,
        ops_count=1,
    )


def run_memory_benchmarks(suite: BenchmarkSuite):
    """Benchmark memory-related operations."""
    print("\n--- Memory Benchmarks ---")

    # Large property values
    def setup_large_props():
        return GraphosDB()

    def insert_large_props(db):
        for i in range(100):
            db.create_node(["Data"], {
                "content": "x" * 1000,  # 1KB per node
                "idx": i,
            })

    suite.benchmark(
        "Insert 100 nodes (1KB props each)",
        setup_large_props,
        insert_large_props,
        ops_count=100,
    )

    # Many properties per node
    def insert_many_props(db):
        props = {f"prop_{i}": f"value_{i}" for i in range(50)}
        for i in range(100):
            db.create_node(["Data"], {**props, "idx": i})

    suite.benchmark(
        "Insert 100 nodes (50 props each)",
        setup_large_props,
        insert_many_props,
        ops_count=100,
    )


def run_gql_vs_cypher_benchmarks(suite: BenchmarkSuite):
    """Benchmark GQL vs Cypher query performance."""
    print("\n--- GQL vs Cypher Benchmarks ---")

    def setup_gql_cypher():
        db = GraphosDB()
        gen = SocialNetworkGenerator(num_nodes=500, avg_edges_per_node=5, seed=42)
        load_data_into_db(db, gen)
        return db

    # GQL-style MATCH
    def gql_match(db):
        result = db.execute(
            "MATCH (p:Person) WHERE p.age > 30 RETURN p.name, p.age"
        )
        list(result)

    suite.benchmark(
        "GQL: Simple MATCH with WHERE",
        setup_gql_cypher,
        gql_match,
        ops_count=1,
    )

    # GQL-style path query
    def gql_path(db):
        result = db.execute(
            "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) "
            "RETURN a.name, b.name, c.name LIMIT 100"
        )
        list(result)

    suite.benchmark(
        "GQL: 2-hop path query",
        setup_gql_cypher,
        gql_path,
        ops_count=1,
    )

    # GQL aggregation
    def gql_aggregation(db):
        result = db.execute(
            "MATCH (p:Person) "
            "RETURN p.city, count(p) AS cnt, avg(p.age) AS avg_age "
            "ORDER BY cnt DESC"
        )
        list(result)

    suite.benchmark(
        "GQL: Aggregation with GROUP BY",
        setup_gql_cypher,
        gql_aggregation,
        ops_count=1,
    )


def run_lpg_model_benchmarks(suite: BenchmarkSuite):
    """Benchmark LPG-specific operations."""
    print("\n--- LPG Model Benchmarks ---")

    # Multi-label node operations
    def setup_lpg():
        return GraphosDB()

    def create_multi_label_nodes(db):
        for i in range(500):
            labels = ["Person"]
            if i % 2 == 0:
                labels.append("Employee")
            if i % 3 == 0:
                labels.append("Manager")
            if i % 5 == 0:
                labels.append("Executive")
            db.create_node(labels, {"name": f"Person{i}", "idx": i})

    suite.benchmark(
        "LPG: Create 500 multi-label nodes",
        setup_lpg,
        create_multi_label_nodes,
        ops_count=500,
    )

    # Query by multiple labels
    def setup_multi_label():
        db = GraphosDB()
        for i in range(500):
            labels = ["Person"]
            if i % 2 == 0:
                labels.append("Employee")
            if i % 3 == 0:
                labels.append("Manager")
            db.create_node(labels, {"name": f"Person{i}"})
        return db

    def query_multi_label(db):
        result = db.execute(
            "MATCH (p:Person:Employee:Manager) RETURN count(p)"
        )
        list(result)

    suite.benchmark(
        "LPG: Query by multiple labels",
        setup_multi_label,
        query_multi_label,
        ops_count=1,
    )

    # Rich property operations
    def create_rich_properties(db):
        for i in range(200):
            db.create_node(["DataNode"], {
                "string_prop": f"Value {i}",
                "int_prop": i * 100,
                "float_prop": i * 3.14159,
                "bool_prop": i % 2 == 0,
                "list_prop": [1, 2, 3, i],
            })

    suite.benchmark(
        "LPG: Create nodes with rich properties",
        setup_lpg,
        create_rich_properties,
        ops_count=200,
    )


def run_rdf_model_benchmarks(suite: BenchmarkSuite):
    """Benchmark RDF-style operations."""
    print("\n--- RDF Model Benchmarks ---")

    def setup_rdf():
        return GraphosDB()

    # Create RDF-like triples
    def create_rdf_triples(db):
        # Create resources (subjects/objects)
        resources = []
        for i in range(100):
            r = db.create_node(["Resource"], {
                "uri": f"http://example.org/resource/{i}",
                "rdf:type": "http://xmlns.com/foaf/0.1/Person",
                "foaf:name": f"Person {i}",
            })
            resources.append(r.id)

        # Create relationships (predicates)
        for i in range(len(resources) - 1):
            db.create_edge(
                resources[i],
                resources[i + 1],
                "http://xmlns.com/foaf/0.1/knows",
                {}
            )

    suite.benchmark(
        "RDF: Create 100 resources + edges",
        setup_rdf,
        create_rdf_triples,
        ops_count=199,  # 100 nodes + 99 edges
    )

    # Query RDF-like data
    def setup_rdf_data():
        db = GraphosDB()
        resources = []
        for i in range(100):
            r = db.create_node(["Resource"], {
                "uri": f"http://example.org/resource/{i}",
                "rdf:type": "http://xmlns.com/foaf/0.1/Person" if i % 2 == 0 else "http://xmlns.com/foaf/0.1/Organization",
                "foaf:name": f"Entity {i}",
            })
            resources.append(r.id)
        for i in range(len(resources) - 1):
            db.create_edge(resources[i], resources[i + 1], "foaf:knows", {})
        return db

    def query_rdf_pattern(db):
        result = db.execute(
            "MATCH (r:Resource) "
            "WHERE r.`rdf:type` = 'http://xmlns.com/foaf/0.1/Person' "
            "RETURN r.`foaf:name`, r.uri"
        )
        list(result)

    suite.benchmark(
        "RDF: Query by rdf:type",
        setup_rdf_data,
        query_rdf_pattern,
        ops_count=1,
    )

    # SPARQL-like path query
    def sparql_like_path(db):
        result = db.execute(
            "MATCH (a:Resource)-[:`foaf:knows`]->(b:Resource)-[:`foaf:knows`]->(c:Resource) "
            "RETURN a.uri, b.uri, c.uri LIMIT 50"
        )
        list(result)

    suite.benchmark(
        "RDF: SPARQL-like path query",
        setup_rdf_data,
        sparql_like_path,
        ops_count=1,
    )


def main():
    """Run all benchmarks."""
    print("=" * 80)
    print("GRAPHOS BENCHMARK SUITE")
    print("=" * 80)
    print("Testing: GQL, Cypher | LPG, RDF")
    print("=" * 80)

    suite = BenchmarkSuite(warmup_iterations=2, iterations=5)

    # Run benchmark categories
    run_insertion_benchmarks(suite)
    run_query_benchmarks(suite)
    run_pattern_benchmarks(suite)
    run_gql_vs_cypher_benchmarks(suite)
    run_lpg_model_benchmarks(suite)
    run_rdf_model_benchmarks(suite)
    run_scalability_benchmarks(suite)
    run_ldbc_benchmarks(suite)
    run_memory_benchmarks(suite)

    # Print summary
    suite.print_results()


if __name__ == "__main__":
    main()
