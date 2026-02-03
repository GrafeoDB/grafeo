# graph-bench

Comprehensive benchmark suite for comparing graph database performance.

[:octicons-mark-github-16: GitHub](https://github.com/GrafeoDB/graph-bench){ .md-button }

## Overview

graph-bench provides standardized, reproducible performance testing across multiple graph database engines. It measures 25 benchmarks organized into 7 categories.

## Supported Databases

| Type | Databases |
|------|-----------|
| **Embedded** | Grafeo (GQL), LadybugDB (Cypher), DuckDB (SQL/PGQ) |
| **Server** | Neo4j, Memgraph, ArangoDB, FalkorDB, NebulaGraph |

## Benchmark Categories

### Storage (4 benchmarks)
- `node_insertion`: Bulk node creation
- `edge_insertion`: Bulk edge creation
- `single_read`: Individual node lookup
- `batch_read`: Batch node retrieval

### Traversal (5 benchmarks)
- `hop_1` / `hop_2`: N-hop neighbor traversal
- `bfs` / `dfs`: Breadth/depth-first search
- `shortest_path`: Dijkstra shortest path

### Algorithms (4 benchmarks)
- `pagerank`: PageRank centrality
- `community_detection`: Louvain community detection
- `betweenness_centrality`: Betweenness centrality
- `closeness_centrality`: Closeness centrality

### Query (3 benchmarks)
- `aggregation_count`: Count aggregations
- `filter_equality`: Equality filtering
- `filter_range`: Range filtering

### Pattern Matching (2 benchmarks)
- `triangle_count`: Triangle enumeration
- `common_neighbors`: Common neighbor queries

### Graph Structure (4 benchmarks)
- `connected_components`: Component detection
- `degree_distribution`: Degree statistics
- `graph_density`: Density calculation
- `reachability`: Path existence queries

### Write Operations (3 benchmarks)
- `property_update`: Property modifications
- `edge_add_existing`: Edge creation between existing nodes
- `mixed_workload`: Combined read/write operations

## Test Scales

| Scale | Nodes | Edges | Warmup | Iterations |
|-------|-------|-------|--------|------------|
| small | 10K | 50K | 2 | 5 |
| medium | 100K | 500K | 3 | 10 |
| large | 1M | 5M | 5 | 10 |

## Installation

```bash
uv add graph-bench
```

## Usage

### CLI

```bash
# Run all benchmarks for Grafeo at small scale
uv run graph-bench run -d grafeo -s small --verbose

# Run specific benchmark
uv run graph-bench run -d grafeo -b traversal.bfs -s medium

# Compare multiple databases
uv run graph-bench run -d grafeo,neo4j,duckdb -s small
```

### Python API

```python
from graph_bench import BenchmarkRunner
from graph_bench.adapters import GrafeoAdapter

adapter = GrafeoAdapter()
runner = BenchmarkRunner(adapter, scale="small")
results = runner.run_all()

for result in results:
    print(f"{result.name}: {result.mean_ms:.2f}ms")
```

## Server Setup

For server-based databases, use Docker Compose:

```bash
cd graph-bench
docker compose up -d
```

## Metrics Collected

Each benchmark captures:

- **Timing**: mean, median, min, max (nanoseconds)
- **Throughput**: operations per second
- **Percentiles**: 95th, 99th percentile latencies
- **Stability**: standard deviation, coefficient of variation

## Requirements

- Python 3.12+
- Docker (for server databases)
- 8GB+ RAM recommended for large scale

## License

Apache-2.0
