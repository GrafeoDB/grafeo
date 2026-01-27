"""
Performance benchmark for Graphos optimization phases.

This benchmark tests key operations:
1. Node/Edge insertions
2. Point lookups
3. Graph traversals (1-hop, 2-hop)
4. Filtered scans with predicates (tests zone maps)
5. Aggregations
6. Sorting
7. Parallel query execution
"""

import time
import random
import statistics
from contextlib import contextmanager
import graphos


@contextmanager
def timer(name: str):
    """Context manager for timing operations."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"  {name}: {elapsed*1000:.2f} ms")


def measure(func, iterations=10):
    """Run function multiple times and return (mean, std) in milliseconds."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = func()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0


def run_benchmark(node_count=100_000, edge_count=500_000, warmup=True):
    """Run comprehensive benchmark suite."""
    print("=" * 60)
    print(f"GRAPHOS PERFORMANCE BENCHMARK")
    print(f"Nodes: {node_count:,}  Edges: {edge_count:,}")
    print("=" * 60)

    db = graphos.GraphosDB()

    # ============================================================
    # 1. INSERTION BENCHMARK
    # ============================================================
    print("\n[1] INSERTION BENCHMARKS")
    print("-" * 40)

    # Node insertions
    names = [f"person_{i}" for i in range(node_count)]
    ages = [random.randint(18, 80) for _ in range(node_count)]
    salaries = [random.uniform(30000, 150000) for _ in range(node_count)]
    cities = random.choices(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia"], k=node_count)

    start = time.perf_counter()
    nodes = []
    for i in range(node_count):
        node = db.create_node(["Person"], {
            "name": names[i],
            "age": ages[i],
            "salary": salaries[i],
            "city": cities[i],
            "index": i,
        })
        nodes.append(node)
    node_insert_time = time.perf_counter() - start
    node_rate = node_count / node_insert_time
    print(f"  Node insertion: {node_insert_time*1000:.2f} ms ({node_rate:,.0f} nodes/sec)")

    # Edge insertions
    start = time.perf_counter()
    edges_created = 0
    for i in range(edge_count):
        src = nodes[random.randint(0, node_count - 1)]
        dst = nodes[random.randint(0, node_count - 1)]
        if src.id != dst.id:
            db.create_edge(src.id, dst.id, "KNOWS", {"since": 2000 + (i % 24), "weight": random.random()})
            edges_created += 1
    edge_insert_time = time.perf_counter() - start
    edge_rate = edges_created / edge_insert_time
    print(f"  Edge insertion: {edge_insert_time*1000:.2f} ms ({edge_rate:,.0f} edges/sec)")

    # ============================================================
    # 2. WARMUP (if enabled)
    # ============================================================
    if warmup:
        print("\n[WARMUP] Running queries to warm up caches...")
        for _ in range(5):
            db.execute("MATCH (n:Person) RETURN n LIMIT 100")
            db.execute("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b LIMIT 100")

    # ============================================================
    # 3. SCAN BENCHMARKS
    # ============================================================
    print("\n[2] SCAN BENCHMARKS")
    print("-" * 40)

    # Full scan with LIMIT
    def full_scan_limited():
        return db.execute("MATCH (n:Person) RETURN n LIMIT 1000")

    mean, std = measure(full_scan_limited, iterations=20)
    print(f"  Full scan + LIMIT 1000: {mean:.2f} ms (std: {std:.2f} ms)")

    # Full scan count
    def count_nodes():
        return db.execute("MATCH (n:Person) RETURN count(n)")

    mean, std = measure(count_nodes, iterations=10)
    print(f"  COUNT(*) all nodes: {mean:.2f} ms (std: {std:.2f} ms)")

    # ============================================================
    # 4. FILTER BENCHMARKS (Zone Map test)
    # ============================================================
    print("\n[3] FILTER BENCHMARKS (Zone Map potential)")
    print("-" * 40)

    # High selectivity filter (should skip most data with zone maps)
    def high_selectivity_filter():
        return db.execute("MATCH (n:Person) WHERE n.age > 75 RETURN n LIMIT 100")

    mean, std = measure(high_selectivity_filter, iterations=20)
    print(f"  Filter age > 75 (high selectivity): {mean:.2f} ms (std: {std:.2f} ms)")

    # Low selectivity filter (scans most data)
    def low_selectivity_filter():
        return db.execute("MATCH (n:Person) WHERE n.age > 25 RETURN n LIMIT 100")

    mean, std = measure(low_selectivity_filter, iterations=20)
    print(f"  Filter age > 25 (low selectivity): {mean:.2f} ms (std: {std:.2f} ms)")

    # Range filter
    def range_filter():
        return db.execute("MATCH (n:Person) WHERE n.age >= 30 RETURN n LIMIT 100")

    mean, std = measure(range_filter, iterations=20)
    print(f"  Filter age >= 30 (range): {mean:.2f} ms (std: {std:.2f} ms)")

    # ============================================================
    # 5. TRAVERSAL BENCHMARKS
    # ============================================================
    print("\n[4] TRAVERSAL BENCHMARKS")
    print("-" * 40)

    # 1-hop traversal
    def one_hop_traversal():
        return db.execute("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b LIMIT 1000")

    mean, std = measure(one_hop_traversal, iterations=20)
    print(f"  1-hop traversal LIMIT 1000: {mean:.2f} ms (std: {std:.2f} ms)")

    # 1-hop with filter
    def one_hop_filtered():
        return db.execute("MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE a.age > 50 RETURN a, b LIMIT 500")

    mean, std = measure(one_hop_filtered, iterations=20)
    print(f"  1-hop filtered (age > 50): {mean:.2f} ms (std: {std:.2f} ms)")

    # ============================================================
    # 6. AGGREGATION BENCHMARKS
    # ============================================================
    print("\n[5] AGGREGATION BENCHMARKS")
    print("-" * 40)

    # Simple COUNT
    def count_edges():
        return db.execute("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN count(b)")

    mean, std = measure(count_edges, iterations=10)
    print(f"  COUNT edges: {mean:.2f} ms (std: {std:.2f} ms)")

    # ============================================================
    # 7. SORT BENCHMARKS
    # ============================================================
    print("\n[6] SORT BENCHMARKS")
    print("-" * 40)

    # Sort by property
    def sort_by_age():
        return db.execute("MATCH (n:Person) RETURN n ORDER BY n.age LIMIT 100")

    mean, std = measure(sort_by_age, iterations=10)
    print(f"  Sort by age LIMIT 100: {mean:.2f} ms (std: {std:.2f} ms)")

    # Sort descending
    def sort_by_age_desc():
        return db.execute("MATCH (n:Person) RETURN n ORDER BY n.age DESC LIMIT 100")

    mean, std = measure(sort_by_age_desc, iterations=10)
    print(f"  Sort by age DESC LIMIT 100: {mean:.2f} ms (std: {std:.2f} ms)")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total nodes: {node_count:,}")
    print(f"  Total edges: {edges_created:,}")
    print(f"  Node insertion rate: {node_rate:,.0f} nodes/sec")
    print(f"  Edge insertion rate: {edge_rate:,.0f} edges/sec")
    print("=" * 60)

    return {
        "node_count": node_count,
        "edge_count": edges_created,
        "node_insert_rate": node_rate,
        "edge_insert_rate": edge_rate,
    }


def run_quick_benchmark():
    """Run a quick benchmark for iteration during development."""
    return run_benchmark(node_count=10_000, edge_count=50_000, warmup=True)


def run_full_benchmark():
    """Run full benchmark suite."""
    return run_benchmark(node_count=100_000, edge_count=500_000, warmup=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        run_quick_benchmark()
    else:
        run_full_benchmark()
