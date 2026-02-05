"""Base class for storage benchmarks.

This module defines the benchmark infrastructure for all storage operations:

Write Operations:
- Node insertion (single, bulk, with properties)
- Edge insertion
- Multi-label nodes
- Large/many properties

Read Operations:
- Full scans
- Filtered scans (selectivity testing)
- Point lookups
- Traversals (1-hop, 2-hop)
- Aggregations
- Sorting
- Pattern matching (triangles)
"""

import time
import gc
import statistics
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional
from contextlib import contextmanager


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


class BaseBenchStorage(ABC):
    """Abstract base class for storage benchmarks.

    Subclasses implement query/operation methods for their specific language.
    """

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

    def measure(self, func: Callable, iterations: int = 10) -> tuple[float, float]:
        """Run function multiple times and return (mean, std) in milliseconds."""
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0

    def benchmark(
        self,
        name: str,
        setup: Callable[[], Any],
        operation: Callable[[Any], None],
        teardown: Optional[Callable[[Any], None]] = None,
        ops_count: int = 1,
    ) -> BenchmarkResult:
        """Run a benchmark with setup, operation, and optional teardown."""
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
        print("STORAGE BENCHMARK RESULTS")
        print("=" * 80)

        if not self.results:
            print("No results to display")
            return

        max_name_len = max(len(r.name) for r in self.results)

        print(f"{'Benchmark':<{max_name_len}} | {'Mean (ms)':<12} | {'Std (ms)':<10} | {'Ops/sec':<12}")
        print("-" * 80)

        for r in self.results:
            print(
                f"{r.name:<{max_name_len}} | "
                f"{r.mean_time_ms:<12.2f} | "
                f"{r.std_time_ms:<10.2f} | "
                f"{r.ops_per_second:<12.0f}"
            )

    # ===== Abstract Methods: Write Operations =====

    @abstractmethod
    def create_single_node(self, db, labels: list[str], props: dict):
        """Create a single node."""
        raise NotImplementedError

    @abstractmethod
    def create_edge(self, db, source_id, target_id, rel_type: str, props: dict):
        """Create a single edge."""
        raise NotImplementedError

    # ===== Abstract Methods: Read Operations =====

    @abstractmethod
    def execute_query(self, db, query: str) -> list:
        """Execute a query and return results as a list."""
        raise NotImplementedError

    @abstractmethod
    def full_scan_query(self, label: str, limit: int = None) -> str:
        """Query: MATCH (n:Label) RETURN n [LIMIT x]"""
        raise NotImplementedError

    @abstractmethod
    def count_query(self, label: str) -> str:
        """Query: MATCH (n:Label) RETURN count(n)"""
        raise NotImplementedError

    @abstractmethod
    def filter_query(self, label: str, prop: str, op: str, value) -> str:
        """Query: MATCH (n:Label) WHERE n.prop op value RETURN n"""
        raise NotImplementedError

    @abstractmethod
    def point_lookup_query(self, label: str, prop: str, value) -> str:
        """Query: MATCH (n:Label) WHERE n.prop = value RETURN n"""
        raise NotImplementedError

    @abstractmethod
    def one_hop_query(self, from_label: str, rel_type: str, to_label: str, limit: int = None) -> str:
        """Query: MATCH (a:Label)-[:REL]->(b:Label) RETURN a, b"""
        raise NotImplementedError

    @abstractmethod
    def two_hop_query(self, label: str, rel_type: str, limit: int = None) -> str:
        """Query: MATCH (a)-[:REL]->(b)-[:REL]->(c) RETURN count(c)"""
        raise NotImplementedError

    @abstractmethod
    def aggregation_query(self, label: str, group_prop: str, agg_prop: str) -> str:
        """Query: MATCH (n:Label) RETURN n.group_prop, count(n), avg(n.agg_prop)"""
        raise NotImplementedError

    @abstractmethod
    def sort_query(self, label: str, sort_prop: str, desc: bool = False, limit: int = 100) -> str:
        """Query: MATCH (n:Label) RETURN n ORDER BY n.prop [DESC] LIMIT x"""
        raise NotImplementedError

    @abstractmethod
    def triangle_query(self, label: str, rel_type: str) -> str:
        """Query: MATCH (a)-[:REL]->(b)-[:REL]->(c)-[:REL]->(a) RETURN count(a)"""
        raise NotImplementedError

    # ===== Abstract Methods: Graph Setup =====

    @abstractmethod
    def setup_social_network(self, db, num_nodes: int, avg_edges: int):
        """Set up a social network graph for benchmarking."""
        raise NotImplementedError

    @abstractmethod
    def setup_clique_graph(self, db, num_cliques: int, clique_size: int):
        """Set up a clique graph for triangle benchmarking."""
        raise NotImplementedError

    # ===== Write Benchmarks =====

    def bench_single_node_insert(self, db_factory, count: int = 1000):
        """Benchmark single node insertion."""
        def setup():
            return db_factory()

        def operation(db):
            for i in range(count):
                self.create_single_node(db, ["Person"], {"name": f"Person{i}", "age": 25 + i % 50})

        return self.benchmark(
            f"Insert {count} nodes",
            setup,
            operation,
            ops_count=count,
        )

    def bench_node_with_properties(self, db_factory, count: int = 1000):
        """Benchmark node insertion with multiple properties."""
        def setup():
            return db_factory()

        def operation(db):
            for i in range(count):
                self.create_single_node(db, ["Person", "Employee"], {
                    "name": f"Person{i}",
                    "age": 25 + i % 50,
                    "email": f"person{i}@example.com",
                    "city": ["NYC", "LA", "Chicago"][i % 3],
                    "salary": 50000 + i * 100,
                })

        return self.benchmark(
            f"Insert {count} nodes (5 props)",
            setup,
            operation,
            ops_count=count,
        )

    def bench_edge_insert(self, db_factory, node_count: int = 100):
        """Benchmark edge insertion."""
        def setup():
            db = db_factory()
            node_ids = []
            for i in range(node_count):
                node = self.create_single_node(db, ["Node"], {"idx": i})
                node_ids.append(node.id)
            return (db, node_ids)

        def operation(ctx):
            db, node_ids = ctx
            for i in range(len(node_ids)):
                for j in range(i + 1, min(i + 10, len(node_ids))):
                    self.create_edge(db, node_ids[i], node_ids[j], "CONNECTED", {"weight": i + j})

        return self.benchmark(
            f"Insert edges ({node_count} nodes)",
            setup,
            operation,
            ops_count=node_count * 4,
        )

    def bench_large_properties(self, db_factory, count: int = 100):
        """Benchmark nodes with large property values."""
        def setup():
            return db_factory()

        def operation(db):
            for i in range(count):
                self.create_single_node(db, ["Data"], {
                    "content": "x" * 1000,
                    "idx": i,
                })

        return self.benchmark(
            f"Insert {count} nodes (1KB props)",
            setup,
            operation,
            ops_count=count,
        )

    def bench_many_properties(self, db_factory, count: int = 100, prop_count: int = 50):
        """Benchmark nodes with many properties."""
        props = {f"prop_{i}": f"value_{i}" for i in range(prop_count)}

        def setup():
            return db_factory()

        def operation(db):
            for i in range(count):
                self.create_single_node(db, ["Data"], {**props, "idx": i})

        return self.benchmark(
            f"Insert {count} nodes ({prop_count} props)",
            setup,
            operation,
            ops_count=count,
        )

    def bench_multi_label_nodes(self, db_factory, count: int = 500):
        """Benchmark multi-label node creation."""
        def setup():
            return db_factory()

        def operation(db):
            for i in range(count):
                labels = ["Person"]
                if i % 2 == 0:
                    labels.append("Employee")
                if i % 3 == 0:
                    labels.append("Manager")
                if i % 5 == 0:
                    labels.append("Executive")
                self.create_single_node(db, labels, {"name": f"Person{i}", "idx": i})

        return self.benchmark(
            f"Insert {count} multi-label nodes",
            setup,
            operation,
            ops_count=count,
        )

    # ===== Read Benchmarks =====

    def bench_full_scan(self, db_factory, setup_func, node_count: int = 1000):
        """Benchmark full scan with LIMIT."""
        def setup():
            db = db_factory()
            setup_func(db)
            return db

        def operation(db):
            query = self.full_scan_query("Person", limit=1000)
            list(self.execute_query(db, query))

        return self.benchmark(
            f"Full scan LIMIT 1000",
            setup,
            operation,
            ops_count=1000,
        )

    def bench_count_nodes(self, db_factory, setup_func):
        """Benchmark COUNT(*) all nodes."""
        def setup():
            db = db_factory()
            setup_func(db)
            return db

        def operation(db):
            query = self.count_query("Person")
            list(self.execute_query(db, query))

        return self.benchmark(
            "COUNT(*) all nodes",
            setup,
            operation,
            ops_count=1,
        )

    def bench_high_selectivity_filter(self, db_factory, setup_func):
        """Benchmark high selectivity filter (should skip most data)."""
        def setup():
            db = db_factory()
            setup_func(db)
            return db

        def operation(db):
            query = self.filter_query("Person", "age", ">", 75)
            list(self.execute_query(db, query))

        return self.benchmark(
            "Filter age > 75 (high selectivity)",
            setup,
            operation,
            ops_count=1,
        )

    def bench_low_selectivity_filter(self, db_factory, setup_func):
        """Benchmark low selectivity filter (scans most data)."""
        def setup():
            db = db_factory()
            setup_func(db)
            return db

        def operation(db):
            query = self.filter_query("Person", "age", ">", 25)
            list(self.execute_query(db, query))

        return self.benchmark(
            "Filter age > 25 (low selectivity)",
            setup,
            operation,
            ops_count=1,
        )

    def bench_point_lookup(self, db_factory, setup_func, lookup_count: int = 100):
        """Benchmark point lookups."""
        def setup():
            db = db_factory()
            setup_func(db)
            return db

        def operation(db):
            for i in range(lookup_count):
                query = self.point_lookup_query("Person", "email", f"user{i}@example.com")
                list(self.execute_query(db, query))

        return self.benchmark(
            f"Point lookup x{lookup_count}",
            setup,
            operation,
            ops_count=lookup_count,
        )

    def bench_one_hop_traversal(self, db_factory, setup_func):
        """Benchmark 1-hop traversal."""
        def setup():
            db = db_factory()
            setup_func(db)
            return db

        def operation(db):
            query = self.one_hop_query("Person", "KNOWS", "Person", limit=1000)
            list(self.execute_query(db, query))

        return self.benchmark(
            "1-hop traversal LIMIT 1000",
            setup,
            operation,
            ops_count=1,
        )

    def bench_two_hop_traversal(self, db_factory, setup_func):
        """Benchmark 2-hop traversal."""
        def setup():
            db = db_factory()
            setup_func(db)
            return db

        def operation(db):
            query = self.two_hop_query("Person", "KNOWS")
            list(self.execute_query(db, query))

        return self.benchmark(
            "2-hop traversal (count)",
            setup,
            operation,
            ops_count=1,
        )

    def bench_aggregation(self, db_factory, setup_func):
        """Benchmark aggregation with GROUP BY."""
        def setup():
            db = db_factory()
            setup_func(db)
            return db

        def operation(db):
            query = self.aggregation_query("Person", "city", "age")
            list(self.execute_query(db, query))

        return self.benchmark(
            "Aggregation (group by city)",
            setup,
            operation,
            ops_count=1,
        )

    def bench_sort(self, db_factory, setup_func, desc: bool = False):
        """Benchmark sorting."""
        def setup():
            db = db_factory()
            setup_func(db)
            return db

        def operation(db):
            query = self.sort_query("Person", "age", desc=desc, limit=100)
            list(self.execute_query(db, query))

        order = "DESC" if desc else "ASC"
        return self.benchmark(
            f"Sort by age {order} LIMIT 100",
            setup,
            operation,
            ops_count=1,
        )

    def bench_triangle_count(self, db_factory, num_cliques: int = 10, clique_size: int = 10):
        """Benchmark triangle counting."""
        def setup():
            db = db_factory()
            self.setup_clique_graph(db, num_cliques, clique_size)
            return db

        def operation(db):
            query = self.triangle_query("Node", "CONNECTED")
            list(self.execute_query(db, query))

        return self.benchmark(
            f"Triangle count ({num_cliques}x{clique_size} cliques)",
            setup,
            operation,
            ops_count=1,
        )

    # ===== Benchmark Suites =====

    def run_write_benchmarks(self, db_factory):
        """Run all write benchmarks."""
        print("\n--- Write Benchmarks ---")
        self.bench_single_node_insert(db_factory, count=500)
        self.bench_node_with_properties(db_factory, count=500)
        self.bench_edge_insert(db_factory, node_count=50)
        self.bench_large_properties(db_factory, count=50)
        self.bench_many_properties(db_factory, count=50, prop_count=30)
        self.bench_multi_label_nodes(db_factory, count=200)

    def run_read_benchmarks(self, db_factory, node_count: int = 500, avg_edges: int = 5):
        """Run all read benchmarks."""
        def setup_func(db):
            self.setup_social_network(db, node_count, avg_edges)

        print("\n--- Scan Benchmarks ---")
        self.bench_full_scan(db_factory, setup_func, node_count)
        self.bench_count_nodes(db_factory, setup_func)

        print("\n--- Filter Benchmarks ---")
        self.bench_high_selectivity_filter(db_factory, setup_func)
        self.bench_low_selectivity_filter(db_factory, setup_func)
        self.bench_point_lookup(db_factory, setup_func, lookup_count=50)

        print("\n--- Traversal Benchmarks ---")
        self.bench_one_hop_traversal(db_factory, setup_func)
        self.bench_two_hop_traversal(db_factory, setup_func)

        print("\n--- Aggregation & Sort Benchmarks ---")
        self.bench_aggregation(db_factory, setup_func)
        self.bench_sort(db_factory, setup_func, desc=False)
        self.bench_sort(db_factory, setup_func, desc=True)

        print("\n--- Pattern Benchmarks ---")
        self.bench_triangle_count(db_factory, num_cliques=5, clique_size=5)

    def run_all(self, db_factory, node_count: int = 500, avg_edges: int = 5):
        """Run all storage benchmarks."""
        self.run_write_benchmarks(db_factory)
        self.run_read_benchmarks(db_factory, node_count, avg_edges)
        self.print_results()
        return self.results
