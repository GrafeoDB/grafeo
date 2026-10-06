---
title: Performance Baselines
description: Measured performance characteristics of Grafeo operations.
tags:
  - performance
  - benchmarks
  - optimization
---

# Performance Baselines

Measured performance characteristics showing what to expect from Grafeo.

!!! note "Benchmark Environment"
    Unless otherwise noted, benchmarks were run on:

    - **CPU**: AMD Ryzen 9 5900X (12 cores, 24 threads)
    - **RAM**: 64GB DDR4-3600
    - **Storage**: NVMe SSD (Samsung 980 Pro)
    - **OS**: Ubuntu 22.04 LTS
    - **Rust**: 1.91.1 (release build with LTO)

---

## Insert Performance

### Node Insertion

| Operation | Throughput | Latency (p50) | Latency (p99) |
|-----------|------------|---------------|---------------|
| Single node (no properties) | ~2M ops/s | 0.5 μs | 2 μs |
| Single node (5 properties) | ~1M ops/s | 1 μs | 4 μs |
| Batch insert (1K nodes) | ~1.5M nodes/s | N/A | N/A |
| Batch insert (100K nodes) | ~1.2M nodes/s | N/A | N/A |

### Edge Insertion

| Operation | Throughput | Latency (p50) | Latency (p99) |
|-----------|------------|---------------|---------------|
| Single edge (no properties) | ~1M ops/s | 1 μs | 4 μs |
| Single edge (3 properties) | ~500K ops/s | 2 μs | 8 μs |
| Batch insert (1K edges) | ~800K edges/s | N/A | N/A |

---

## Query Performance

### Point Lookups

| Operation | Latency (cold) | Latency (warm) |
|-----------|----------------|----------------|
| Get node by ID | <1 μs | <0.5 μs |
| Get edge by ID | <1 μs | <0.5 μs |
| Get node properties | 1-2 μs | <1 μs |
| Property index lookup | <1 μs | <0.5 μs |

### Pattern Matching

Tested on a social network graph with 1M nodes and 10M edges:

| Query Pattern | Latency | Notes |
|--------------|---------|-------|
| `MATCH (n:Person) RETURN n LIMIT 100` | 0.1 ms | Label scan with limit |
| `MATCH (n:Person {name: $name})` (indexed) | 0.01 ms | Hash index lookup |
| `MATCH (n:Person {name: $name})` (no index) | 50 ms | Full scan |
| `MATCH (a)-[:KNOWS]->(b) LIMIT 100` | 0.2 ms | Simple traversal |
| `MATCH (a)-[:KNOWS*1..3]->(b) LIMIT 100` | 2-20 ms | Variable-length path |
| `MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(c)` | 5-50 ms | Two-hop traversal |

### Aggregations

Tested on 1M nodes with `age` property:

| Query | Latency |
|-------|---------|
| `MATCH (n:Person) RETURN count(n)` | 5 ms |
| `MATCH (n:Person) RETURN avg(n.age)` | 15 ms |
| `MATCH (n:Person) RETURN n.city, count(*)` | 30 ms |
| `MATCH (n:Person) RETURN max(n.age), min(n.age)` | 12 ms |

---

## Graph Algorithm Performance

All benchmarks on graphs with 1M nodes unless noted:

| Algorithm | Time | Memory |
|-----------|------|--------|
| BFS (single source) | 50-200 ms | O(V) |
| DFS (single source) | 50-200 ms | O(V) |
| Connected components | 200-500 ms | O(V) |
| Strongly connected (1M directed) | 500-1000 ms | O(V) |
| PageRank (20 iterations) | 800-1200 ms | O(V) |
| Dijkstra (single source) | 300-800 ms | O(V + E) |
| Triangle counting | 2-10 s | O(E) |
| Louvain community | 2-5 s | O(V + E) |

### Scaling Behavior

PageRank on varying graph sizes:

| Nodes | Edges | Time |
|-------|-------|------|
| 100K | 1M | 80 ms |
| 500K | 5M | 400 ms |
| 1M | 10M | 900 ms |
| 5M | 50M | 5.5 s |
| 10M | 100M | 12 s |

---

## Vector Search Performance

Using HNSW index with 128-dimensional vectors:

| Operation | 100K vectors | 1M vectors | 10M vectors |
|-----------|-------------|------------|-------------|
| Build index | 2 s | 25 s | 5 min |
| k-NN query (k=10) | 0.2 ms | 0.5 ms | 1 ms |
| k-NN query (k=100) | 0.5 ms | 1 ms | 2 ms |
| Hybrid query (vector + filter) | 1-5 ms | 5-20 ms | 20-100 ms |

### Recall vs Speed Tradeoff

| ef_search | Recall@10 | Latency |
|-----------|-----------|---------|
| 16 | 85% | 0.1 ms |
| 64 | 95% | 0.3 ms |
| 128 | 98% | 0.6 ms |
| 256 | 99% | 1.2 ms |

---

## Memory Usage

### Per-Entity Overhead

| Component | Bytes |
|-----------|-------|
| Node (no properties) | 40-56 |
| Node (with 5 string properties) | 200-400 |
| Edge (no properties) | 32-48 |
| Edge (with 3 properties) | 150-250 |
| Adjacency entry | 8-16 |

### Index Memory

| Index Type | Overhead |
|------------|----------|
| Hash index | 16-24 bytes/entry |
| B-tree index | 24-32 bytes/entry |
| HNSW (128-dim, M=16) | 1.5-2 KB/vector |

### Working Set Sizes

Approximate memory for graph operations:

| Graph Size | Cold (disk) | Warm (in memory) |
|------------|-------------|------------------|
| 100K nodes, 1M edges | 50 MB | 150 MB |
| 1M nodes, 10M edges | 500 MB | 1.5 GB |
| 10M nodes, 100M edges | 5 GB | 15 GB |

---

## Transaction Performance

### Single-Threaded

| Operation | Throughput |
|-----------|------------|
| Read-only tx (1 query) | 100K tx/s |
| Read-write tx (1 insert) | 50K tx/s |
| Read-write tx (10 inserts) | 20K tx/s |

### Concurrent Transactions

4 threads, read-heavy workload (90% reads):

| Contention | Throughput | Abort Rate |
|------------|------------|------------|
| Low (random access) | 200K tx/s | <1% |
| Medium (hot spots) | 100K tx/s | 5-10% |
| High (same nodes) | 20K tx/s | 30-50% |

### Block-STM (Batch Mode)

When processing batches of similar transactions:

| Conflict Rate | Speedup (4 cores) |
|---------------|-------------------|
| 0% | 3.8x |
| 5% | 3.2x |
| 10% | 2.5x |
| 20% | 1.8x |

---

## Compression Effectiveness

On typical graph data:

| Data Type | Compression | Ratio |
|-----------|-------------|-------|
| Node labels | Dictionary | 10-50x |
| String properties | Dictionary | 2-10x |
| Integer IDs | Delta + BitPack | 4-8x |
| Boolean properties | BitVector | 8x |
| Timestamps | Delta | 6-10x |

Overall storage reduction: **40-60%** compared to uncompressed.

---

## Benchmark Comparisons (LDBC-Inspired)

Results from [graph-bench](graph-bench.md) (SF0.1), which includes workloads inspired by the [LDBC Social Network Benchmark](https://ldbcouncil.org/benchmarks/snb/). Methodology: 3 warmup runs, 10 measured runs, median reported.

!!! warning "Not Official LDBC Results"
    These are **not official LDBC Benchmark results**. The benchmarks have not been audited by the LDBC Council, use synthetic datasets (not the official LDBC Datagen), and run at reduced scale factors. See the [LDBC disclaimer](https://github.com/GrafeoDB/graph-bench#ldbc-disclaimer) for full details. LDBC specifications are used under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).

### Embedded (in-process)

| Database | SNB Interactive | Memory | Graph Analytics | Memory | ACID | Memory |
|----------|---------------:|-------:|----------------:|-------:|-----:|-------:|
| **Grafeo** | **2,904 ms** | 136 MB | **0.4 ms** | 43 MB | **40 ms** | 67 MB |
| LadybugDB | 5,333 ms | 4,890 MB | 225 ms | 250 MB | 128 ms | 4,914 MB |
| FalkorDB Lite | 7,454 ms | 156 MB | 89 ms | 88 MB | 72 ms | 144 MB |

### Server (over network)

| Database | SNB Interactive | Graph Analytics | ACID |
|----------|---------------:|----------------:|-----:|
| **Grafeo Server** | **730 ms** | **15 ms** | 198 ms |
| Memgraph | 4,113 ms | 19 ms | **107 ms** |
| Neo4j | 6,788 ms | 253 ms | 369 ms |
| ArangoDB | 40,043 ms | 22,739 ms | 2,110 ms |

Full results: [embedded](https://github.com/GrafeoDB/graph-bench/blob/main/RESULTS_EMBEDDED.md) | [server](https://github.com/GrafeoDB/graph-bench/blob/main/RESULTS_SERVER.md)

---

## Arrow Export Performance

The Arrow bulk export path (`nodes_to_arrow()`, `edges_to_arrow()` and their Polars/pandas variants) builds a single RecordBatch in Rust and serializes it as IPC bytes. Python receives the buffer in one call with no per-element PyO3 crossings.

| Method | 1K nodes | 100K nodes | 1M nodes |
|--------|----------|------------|----------|
| `nodes_df()` (fallback) | 5 ms | 500 ms | 6 s |
| `nodes_to_arrow()` | 1 ms | 10 ms | 80 ms |
| **Speedup** | 5x | 50x | 75x |

The existing `nodes_df()`/`edges_df()` auto-detect pyarrow at runtime and use the Arrow fast path when available. Installing pyarrow speeds up all DataFrame exports without code changes.

---

## Ring Index Performance

The Ring Index (`ring-index` feature) stores RDF triples in a wavelet-tree structure that supports all six triple-pattern access orders (SPO, SOP, PSO, POS, OSP, OPS) in a single compact index.

| Operation | Ring Index | Hash-based indexes |
|-----------|-----------|-------------------|
| Single pattern lookup | O(log sigma) | O(1) amortized |
| Star join (3 patterns) | O(k log sigma) via WCOJ | O(n) hash probe per pattern |
| COUNT(\*) (bound pattern) | O(log sigma), no materialization | O(n) full scan |
| COUNT(\*) (unbound) | O(1) via `store.len()` | O(1) via `store.len()` |
| Memory (1M triples) | ~12 MB | ~48 MB (6 hash maps) |

The SPARQL planner automatically selects Ring Index when available. Leapfrog worst-case optimal join (WCOJ) handles multi-way star joins without intermediate materialization. For patterns requiring LANG or DATATYPE columns (not stored in the ring), the planner falls back to hash join.

---

## Optimizing Performance

### Index Strategy

Create indexes for frequently-queried properties:

```python
# Before: O(n) scan
db.find_nodes_by_property("email", "alix@example.com")  # 50ms on 1M nodes

# Create index
db.create_property_index("email")

# After: O(1) lookup
db.find_nodes_by_property("email", "alix@example.com")  # 0.01ms
```

### Batch Operations

Use batch APIs for bulk operations:

```python
# Slow: Individual calls
for node_id in node_ids:
    props = db.get_node(node_id).properties  # 1000 calls = 1ms each = 1s total

# Fast: Batch call
results = db.get_nodes_by_label("Person", limit=1000)  # Single call = 10ms
```

### Query Hints

Add LIMIT to exploratory queries:

```python
# Potentially slow
db.execute("MATCH (n:Person)-[:KNOWS*1..5]->(m) RETURN n, m")

# Bounded
db.execute("MATCH (n:Person)-[:KNOWS*1..5]->(m) RETURN n, m LIMIT 1000")
```

### Memory Configuration

For large graphs, configure memory limits via GQL:

```python
from grafeo import GrafeoDB

db = GrafeoDB()

# Set memory limit via database configuration
db.execute("SET DATABASE OPTION memory_limit = 8589934592")   # 8GB
db.execute("SET DATABASE OPTION spill_path = '/tmp/grafeo_spill'")
```

---

## Workload Profiling

Use built-in statistics:

```python
# Check database stats
stats = db.detailed_stats()
print(f"Memory usage: {stats['memory_bytes'] / 1024**2:.1f} MB")

# Check if indexes exist
print(f"Has email index: {db.has_property_index('email')}")

# Schema overview
schema = db.schema()
for label in schema['labels']:
    print(f"{label['name']}: {label['count']} nodes")
```

For detailed profiling, use Python's `cProfile`:

```python
import cProfile
import pstats

with cProfile.Profile() as pr:
    result = db.execute("MATCH (n:Person)-[:KNOWS]->(m) RETURN n, m LIMIT 1000")

stats = pstats.Stats(pr)
stats.sort_stats('cumulative')
stats.print_stats(10)
```
