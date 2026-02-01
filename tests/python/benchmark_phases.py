"""
Performance benchmark for Grafeo optimization phases.

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
import grafeo


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

    db = grafeo.GrafeoDB()

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


def run_factorized_benchmark(node_count=1_000, avg_degree=15, warmup=True):
    """
    Benchmark for factorized execution (Phase 3+4).

    Tests multi-hop traversals with high fan-out where factorized execution
    provides 50-100x speedup by avoiding Cartesian product materialization.

    Key insight: With avg_degree=15:
    - 1-hop from 1 node: ~15 results
    - 2-hop from 1 node: ~15*15 = 225 results
    - 3-hop from 1 node: ~15*15*15 = 3,375 results

    Without factorization: O(d^k) memory for k hops, degree d
    With factorization: O(k*d) memory - linear in hops
    """
    print("=" * 60)
    print("FACTORIZED EXECUTION BENCHMARK (Phase 3+4)")
    print(f"Nodes: {node_count:,}  Avg Degree: {avg_degree}")
    print("=" * 60)

    db = grafeo.GrafeoDB()

    # Create nodes
    edge_count = node_count * avg_degree
    print(f"\n[SETUP] Creating {node_count:,} nodes, ~{edge_count:,} edges...")

    start = time.perf_counter()
    nodes = []
    for i in range(node_count):
        node = db.create_node(["Person"], {"id": i, "name": f"person_{i}"})
        nodes.append(node)

    # Create edges with controlled fan-out
    edges_created = 0
    for i in range(edge_count):
        src_idx = i % node_count
        # Spread destinations to create consistent fan-out
        dst_idx = (src_idx + 1 + (i // node_count)) % node_count
        if src_idx != dst_idx:
            db.create_edge(nodes[src_idx].id, nodes[dst_idx].id, "KNOWS", {"weight": 1.0})
            edges_created += 1

    setup_time = time.perf_counter() - start
    print(f"  Setup: {setup_time*1000:.2f} ms ({edges_created:,} edges)")

    if warmup:
        print("\n[WARMUP] Running warmup queries...")
        for _ in range(3):
            db.execute("MATCH (n:Person) RETURN n LIMIT 10")
            db.execute("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b LIMIT 10")

    # ============================================================
    # MULTI-HOP TRAVERSAL BENCHMARKS
    # These are the key benchmarks for factorized execution
    # ============================================================
    print("\n" + "=" * 60)
    print("MULTI-HOP TRAVERSAL BENCHMARKS (Factorization Target)")
    print("=" * 60)

    # 1-hop baseline
    print("\n[1-HOP] Baseline")
    print("-" * 40)

    def one_hop_all():
        return db.execute(
            "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.id, b.id LIMIT 10000"
        )

    mean, std = measure(one_hop_all, iterations=10)
    result = one_hop_all()
    row_count = len(result) if hasattr(result, '__len__') else 0
    print(f"  1-hop traversal: {mean:.2f} ms (std: {std:.2f}) - {row_count} rows")

    # 2-hop - this is where factorization starts to matter
    print("\n[2-HOP] Factorization Impact Zone")
    print("-" * 40)

    def two_hop_from_one():
        return db.execute(
            "MATCH (a:Person {id: 0})-[:KNOWS]->(b)-[:KNOWS]->(c) RETURN a.id, b.id, c.id"
        )

    mean, std = measure(two_hop_from_one, iterations=10)
    result = two_hop_from_one()
    row_count = len(result) if hasattr(result, '__len__') else 0
    print(f"  2-hop from node 0: {mean:.2f} ms (std: {std:.2f}) - {row_count} rows")

    def two_hop_limited():
        return db.execute(
            "MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(c) RETURN a.id, b.id, c.id LIMIT 10000"
        )

    mean, std = measure(two_hop_limited, iterations=10)
    print(f"  2-hop all (LIMIT 10000): {mean:.2f} ms (std: {std:.2f})")

    # 2-hop with DISTINCT (reduces output, less factorization benefit)
    def two_hop_distinct():
        return db.execute(
            "MATCH (a:Person {id: 0})-[:KNOWS]->(b)-[:KNOWS]->(c) RETURN DISTINCT c.id"
        )

    mean, std = measure(two_hop_distinct, iterations=10)
    result = two_hop_distinct()
    row_count = len(result) if hasattr(result, '__len__') else 0
    print(f"  2-hop DISTINCT c: {mean:.2f} ms (std: {std:.2f}) - {row_count} unique")

    # 3-hop - major factorization benefit (O(d^3) vs O(3d))
    print("\n[3-HOP] High Factorization Benefit")
    print("-" * 40)

    def three_hop_from_one():
        return db.execute(
            "MATCH (a:Person {id: 0})-[:KNOWS]->(b)-[:KNOWS]->(c)-[:KNOWS]->(d) "
            "RETURN a.id, b.id, c.id, d.id LIMIT 5000"
        )

    mean, std = measure(three_hop_from_one, iterations=5)
    result = three_hop_from_one()
    row_count = len(result) if hasattr(result, '__len__') else 0
    print(f"  3-hop from node 0 (LIMIT 5000): {mean:.2f} ms (std: {std:.2f}) - {row_count} rows")

    def three_hop_count():
        return db.execute(
            "MATCH (a:Person {id: 0})-[:KNOWS]->(b)-[:KNOWS]->(c)-[:KNOWS]->(d) "
            "RETURN count(d)"
        )

    mean, std = measure(three_hop_count, iterations=5)
    result = three_hop_count()
    print(f"  3-hop COUNT: {mean:.2f} ms (std: {std:.2f}) - result: {result}")

    # Triangle pattern - tests cyclic pattern handling
    print("\n[TRIANGLE] Cyclic Pattern")
    print("-" * 40)

    def triangle_count():
        return db.execute(
            "MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(c)-[:KNOWS]->(a) "
            "RETURN count(a)"
        )

    mean, std = measure(triangle_count, iterations=5)
    result = triangle_count()
    print(f"  Triangle count: {mean:.2f} ms (std: {std:.2f}) - result: {result}")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("EXPECTED IMPROVEMENTS WITH FACTORIZED EXECUTION")
    print("=" * 60)
    print("  2-hop queries: 5-20x faster (avoiding d^2 intermediate rows)")
    print("  3-hop queries: 20-100x faster (avoiding d^3 intermediate rows)")
    print("  Memory usage: O(k*d) instead of O(d^k) for k hops, degree d")
    print("=" * 60)

    return {
        "node_count": node_count,
        "edge_count": edges_created,
        "avg_degree": avg_degree,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            run_quick_benchmark()
        elif sys.argv[1] == "--factorized":
            # Use smaller graph for factorized to see the pattern clearly
            run_factorized_benchmark(node_count=500, avg_degree=20)
        elif sys.argv[1] == "--factorized-large":
            run_factorized_benchmark(node_count=2_000, avg_degree=15)
        else:
            run_full_benchmark()
    else:
        run_full_benchmark()
