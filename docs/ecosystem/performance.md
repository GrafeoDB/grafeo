---
title: Performance Baselines
description: Measured performance characteristics of Grafeo operations.
tags:
  - performance
  - benchmarks
  - optimization
---

# Performance Baselines

Measured performance characteristics to help you understand what to expect from Grafeo.

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

## Comparison with Other Systems

!!! warning "Methodology Note"
    These comparisons use similar hardware and configurations where possible, but exact comparisons depend heavily on workload and tuning. Use as rough guidance only.

### Insert Throughput (1M nodes)

| System | Throughput |
|--------|------------|
| Grafeo (in-memory) | 1.2M/s |
| Neo4j Community (embedded) | 200K/s |
| NetworkX | 500K/s |
| DuckDB (relational) | 2M/s |

### Query Latency (point lookup)

| System | Latency |
|--------|---------|
| Grafeo | <1 μs |
| Neo4j | 0.1-0.5 ms |
| NetworkX | 1-5 μs |

### PageRank (1M nodes, 10M edges)

| System | Time |
|--------|------|
| Grafeo | 0.9 s |
| Neo4j GDS | 4-6 s |
| NetworkX | 8-12 s |
| graph-tool | 0.5 s |

---

## Optimizing Performance

### Index Strategy

Create indexes for frequently-queried properties:

```python
# Before: O(n) scan
db.find_nodes_by_property("email", "alice@example.com")  # 50ms on 1M nodes

# Create index
db.create_property_index("email")

# After: O(1) lookup
db.find_nodes_by_property("email", "alice@example.com")  # 0.01ms
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

For large graphs, configure memory limits:

```python
from grafeo import Config, GrafeoDB

config = Config.in_memory()
config.memory_limit = 8 * 1024 * 1024 * 1024  # 8GB
config.spill_path = "/tmp/grafeo_spill"

db = GrafeoDB.with_config(config)
```

---

## Profiling Your Workload

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
