"""Base class for algorithm benchmarks.

This module defines the benchmark infrastructure for graph algorithms:
- Traversal: BFS, DFS
- Shortest Path: Dijkstra, Bellman-Ford
- Centrality: Degree, PageRank, Betweenness
- Components: Connected, Strongly Connected
- Community: Label Propagation, Louvain
- MST: Kruskal, Prim
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


class BaseBenchAlgorithms(ABC):
    """Abstract base class for algorithm benchmarks.

    Subclasses implement algorithm execution for their specific language/API.
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
        print("ALGORITHM BENCHMARK RESULTS")
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

    # ===== Abstract Methods =====

    @abstractmethod
    def setup_random_graph(self, db, n_nodes: int, n_edges: int, weighted: bool = True) -> dict:
        """Set up a random graph for benchmarking.

        Args:
            db: Database instance
            n_nodes: Number of nodes
            n_edges: Number of edges
            weighted: Whether to add weight property to edges

        Returns:
            dict with 'node_ids' list and optional metadata
        """
        raise NotImplementedError

    @abstractmethod
    def run_bfs(self, db, start_node) -> list:
        """Run BFS from start node."""
        raise NotImplementedError

    @abstractmethod
    def run_dfs(self, db, start_node) -> list:
        """Run DFS from start node."""
        raise NotImplementedError

    @abstractmethod
    def run_dijkstra(self, db, source, target=None, weight_prop: str = "weight"):
        """Run Dijkstra's algorithm."""
        raise NotImplementedError

    @abstractmethod
    def run_bellman_ford(self, db, source, weight_prop: str = "weight"):
        """Run Bellman-Ford algorithm."""
        raise NotImplementedError

    @abstractmethod
    def run_connected_components(self, db) -> dict:
        """Run connected components."""
        raise NotImplementedError

    @abstractmethod
    def run_strongly_connected_components(self, db) -> list:
        """Run strongly connected components."""
        raise NotImplementedError

    @abstractmethod
    def run_pagerank(self, db, damping: float = 0.85, iterations: int = 20) -> dict:
        """Run PageRank algorithm."""
        raise NotImplementedError

    @abstractmethod
    def run_degree_centrality(self, db, normalized: bool = True) -> dict:
        """Run degree centrality."""
        raise NotImplementedError

    @abstractmethod
    def run_betweenness_centrality(self, db) -> dict:
        """Run betweenness centrality."""
        raise NotImplementedError

    @abstractmethod
    def run_closeness_centrality(self, db) -> dict:
        """Run closeness centrality."""
        raise NotImplementedError

    @abstractmethod
    def run_label_propagation(self, db) -> dict:
        """Run label propagation community detection."""
        raise NotImplementedError

    @abstractmethod
    def run_louvain(self, db) -> dict:
        """Run Louvain community detection."""
        raise NotImplementedError

    @abstractmethod
    def run_kruskal(self, db, weight_prop: str = "weight") -> dict:
        """Run Kruskal's MST algorithm."""
        raise NotImplementedError

    @abstractmethod
    def run_prim(self, db, weight_prop: str = "weight") -> dict:
        """Run Prim's MST algorithm."""
        raise NotImplementedError

    # ===== Benchmark Tests =====

    def bench_bfs(self, db_factory, n_nodes: int = 1000, n_edges: int = 5000):
        """Benchmark BFS traversal."""
        def setup():
            db = db_factory()
            graph_info = self.setup_random_graph(db, n_nodes, n_edges, weighted=False)
            return (db, graph_info["node_ids"][0])

        def operation(ctx):
            db, start_node = ctx
            self.run_bfs(db, start_node)

        return self.benchmark(
            f"BFS ({n_nodes} nodes, {n_edges} edges)",
            setup,
            operation,
            ops_count=1,
        )

    def bench_dfs(self, db_factory, n_nodes: int = 1000, n_edges: int = 5000):
        """Benchmark DFS traversal."""
        def setup():
            db = db_factory()
            graph_info = self.setup_random_graph(db, n_nodes, n_edges, weighted=False)
            return (db, graph_info["node_ids"][0])

        def operation(ctx):
            db, start_node = ctx
            self.run_dfs(db, start_node)

        return self.benchmark(
            f"DFS ({n_nodes} nodes, {n_edges} edges)",
            setup,
            operation,
            ops_count=1,
        )

    def bench_dijkstra(self, db_factory, n_nodes: int = 1000, n_edges: int = 5000):
        """Benchmark Dijkstra's algorithm."""
        def setup():
            db = db_factory()
            graph_info = self.setup_random_graph(db, n_nodes, n_edges, weighted=True)
            return (db, graph_info["node_ids"][0])

        def operation(ctx):
            db, source = ctx
            self.run_dijkstra(db, source)

        return self.benchmark(
            f"Dijkstra ({n_nodes} nodes, {n_edges} edges)",
            setup,
            operation,
            ops_count=1,
        )

    def bench_bellman_ford(self, db_factory, n_nodes: int = 500, n_edges: int = 2000):
        """Benchmark Bellman-Ford algorithm."""
        def setup():
            db = db_factory()
            graph_info = self.setup_random_graph(db, n_nodes, n_edges, weighted=True)
            return (db, graph_info["node_ids"][0])

        def operation(ctx):
            db, source = ctx
            self.run_bellman_ford(db, source)

        return self.benchmark(
            f"Bellman-Ford ({n_nodes} nodes, {n_edges} edges)",
            setup,
            operation,
            ops_count=1,
        )

    def bench_connected_components(self, db_factory, n_nodes: int = 1000, n_edges: int = 3000):
        """Benchmark connected components."""
        def setup():
            db = db_factory()
            self.setup_random_graph(db, n_nodes, n_edges, weighted=False)
            return db

        def operation(db):
            self.run_connected_components(db)

        return self.benchmark(
            f"Connected Components ({n_nodes} nodes)",
            setup,
            operation,
            ops_count=1,
        )

    def bench_strongly_connected_components(self, db_factory, n_nodes: int = 1000, n_edges: int = 5000):
        """Benchmark strongly connected components."""
        def setup():
            db = db_factory()
            self.setup_random_graph(db, n_nodes, n_edges, weighted=False)
            return db

        def operation(db):
            self.run_strongly_connected_components(db)

        return self.benchmark(
            f"Strongly Connected Components ({n_nodes} nodes)",
            setup,
            operation,
            ops_count=1,
        )

    def bench_pagerank(self, db_factory, n_nodes: int = 1000, n_edges: int = 5000):
        """Benchmark PageRank algorithm."""
        def setup():
            db = db_factory()
            self.setup_random_graph(db, n_nodes, n_edges, weighted=False)
            return db

        def operation(db):
            self.run_pagerank(db)

        return self.benchmark(
            f"PageRank ({n_nodes} nodes)",
            setup,
            operation,
            ops_count=1,
        )

    def bench_degree_centrality(self, db_factory, n_nodes: int = 1000, n_edges: int = 5000):
        """Benchmark degree centrality."""
        def setup():
            db = db_factory()
            self.setup_random_graph(db, n_nodes, n_edges, weighted=False)
            return db

        def operation(db):
            self.run_degree_centrality(db)

        return self.benchmark(
            f"Degree Centrality ({n_nodes} nodes)",
            setup,
            operation,
            ops_count=1,
        )

    def bench_betweenness_centrality(self, db_factory, n_nodes: int = 200, n_edges: int = 1000):
        """Benchmark betweenness centrality (O(V*E) complexity)."""
        def setup():
            db = db_factory()
            self.setup_random_graph(db, n_nodes, n_edges, weighted=False)
            return db

        def operation(db):
            self.run_betweenness_centrality(db)

        return self.benchmark(
            f"Betweenness Centrality ({n_nodes} nodes)",
            setup,
            operation,
            ops_count=1,
        )

    def bench_closeness_centrality(self, db_factory, n_nodes: int = 500, n_edges: int = 2000):
        """Benchmark closeness centrality."""
        def setup():
            db = db_factory()
            self.setup_random_graph(db, n_nodes, n_edges, weighted=False)
            return db

        def operation(db):
            self.run_closeness_centrality(db)

        return self.benchmark(
            f"Closeness Centrality ({n_nodes} nodes)",
            setup,
            operation,
            ops_count=1,
        )

    def bench_label_propagation(self, db_factory, n_nodes: int = 1000, n_edges: int = 5000):
        """Benchmark label propagation community detection."""
        def setup():
            db = db_factory()
            self.setup_random_graph(db, n_nodes, n_edges, weighted=False)
            return db

        def operation(db):
            self.run_label_propagation(db)

        return self.benchmark(
            f"Label Propagation ({n_nodes} nodes)",
            setup,
            operation,
            ops_count=1,
        )

    def bench_louvain(self, db_factory, n_nodes: int = 1000, n_edges: int = 5000):
        """Benchmark Louvain community detection."""
        def setup():
            db = db_factory()
            self.setup_random_graph(db, n_nodes, n_edges, weighted=False)
            return db

        def operation(db):
            self.run_louvain(db)

        return self.benchmark(
            f"Louvain ({n_nodes} nodes)",
            setup,
            operation,
            ops_count=1,
        )

    def bench_kruskal(self, db_factory, n_nodes: int = 1000, n_edges: int = 5000):
        """Benchmark Kruskal's MST."""
        def setup():
            db = db_factory()
            self.setup_random_graph(db, n_nodes, n_edges, weighted=True)
            return db

        def operation(db):
            self.run_kruskal(db)

        return self.benchmark(
            f"Kruskal MST ({n_nodes} nodes)",
            setup,
            operation,
            ops_count=1,
        )

    def bench_prim(self, db_factory, n_nodes: int = 1000, n_edges: int = 5000):
        """Benchmark Prim's MST."""
        def setup():
            db = db_factory()
            self.setup_random_graph(db, n_nodes, n_edges, weighted=True)
            return db

        def operation(db):
            self.run_prim(db)

        return self.benchmark(
            f"Prim MST ({n_nodes} nodes)",
            setup,
            operation,
            ops_count=1,
        )

    def run_traversal_benchmarks(self, db_factory, n_nodes: int = 1000, n_edges: int = 5000):
        """Run traversal algorithm benchmarks."""
        print("\n--- Traversal Benchmarks ---")
        self.bench_bfs(db_factory, n_nodes, n_edges)
        self.bench_dfs(db_factory, n_nodes, n_edges)

    def run_shortest_path_benchmarks(self, db_factory, n_nodes: int = 1000, n_edges: int = 5000):
        """Run shortest path algorithm benchmarks."""
        print("\n--- Shortest Path Benchmarks ---")
        self.bench_dijkstra(db_factory, n_nodes, n_edges)
        self.bench_bellman_ford(db_factory, n_nodes // 2, n_edges // 2)  # Smaller for BF

    def run_component_benchmarks(self, db_factory, n_nodes: int = 1000, n_edges: int = 3000):
        """Run component algorithm benchmarks."""
        print("\n--- Component Benchmarks ---")
        self.bench_connected_components(db_factory, n_nodes, n_edges)
        self.bench_strongly_connected_components(db_factory, n_nodes, n_edges * 2)

    def run_centrality_benchmarks(self, db_factory, n_nodes: int = 500, n_edges: int = 2000):
        """Run centrality algorithm benchmarks."""
        print("\n--- Centrality Benchmarks ---")
        self.bench_pagerank(db_factory, n_nodes, n_edges)
        self.bench_degree_centrality(db_factory, n_nodes, n_edges)
        self.bench_betweenness_centrality(db_factory, n_nodes // 2, n_edges // 2)  # Smaller for BC
        self.bench_closeness_centrality(db_factory, n_nodes, n_edges)

    def run_community_benchmarks(self, db_factory, n_nodes: int = 1000, n_edges: int = 5000):
        """Run community detection algorithm benchmarks."""
        print("\n--- Community Detection Benchmarks ---")
        self.bench_label_propagation(db_factory, n_nodes, n_edges)
        self.bench_louvain(db_factory, n_nodes, n_edges)

    def run_mst_benchmarks(self, db_factory, n_nodes: int = 1000, n_edges: int = 5000):
        """Run MST algorithm benchmarks."""
        print("\n--- Minimum Spanning Tree Benchmarks ---")
        self.bench_kruskal(db_factory, n_nodes, n_edges)
        self.bench_prim(db_factory, n_nodes, n_edges)

    def run_all(self, db_factory, n_nodes: int = 1000, n_edges: int = 5000):
        """Run all algorithm benchmarks."""
        self.run_traversal_benchmarks(db_factory, n_nodes, n_edges)
        self.run_shortest_path_benchmarks(db_factory, n_nodes, n_edges)
        self.run_component_benchmarks(db_factory, n_nodes, n_edges)
        self.run_centrality_benchmarks(db_factory, n_nodes // 2, n_edges // 2)
        self.run_community_benchmarks(db_factory, n_nodes, n_edges)
        self.run_mst_benchmarks(db_factory, n_nodes, n_edges)

        self.print_results()
        return self.results
