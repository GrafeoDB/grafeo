---
title: Path Finding
description: Path finding algorithms.
tags:
  - algorithms
  - paths
---

# Path Finding

Algorithms for finding paths between nodes.

## Shortest Path

Find the shortest path between two nodes.

```python
import grafeo

db = grafeo.GrafeoDB()
algs = db.algorithms

# (distance, path), or None when 100 is unreachable from 1.
# Without a weight, the distance is the number of hops.
found = algs.dijkstra(1, target=100)
if found is None:
    print("No path")
else:
    distance, path = found
    print(f"Distance: {distance}")
    for node_id in path:
        print(f"  -> {node_id}")
```

## Dijkstra's Algorithm

Weighted shortest path using Dijkstra's algorithm.

```python
algs = db.algorithms
found = algs.dijkstra(source=1, target=100, weight="cost")
```

## Breadth-First Search

Traverse the graph level by level.

```python
algs = db.algorithms

# BFS from a starting node
visited = algs.bfs(start=1)

# BFS with distance layers
layers = algs.bfs_layers(start=1)
for distance, nodes in enumerate(layers):
    print(f"Distance {distance}: {len(nodes)} nodes")
```

## Depth-First Search

Traverse the graph depth-first.

```python
algs = db.algorithms

# DFS from a starting node
visited = algs.dfs(start=1)

# DFS visiting all nodes
all_visited = algs.dfs_all()
```

## All Pairs Shortest Distances

Precompute the shortest distance between every pair of nodes with
Floyd-Warshall: distances, not paths, in O(n^3) time and O(n^2) memory.

```python
algs = db.algorithms
distances = algs.floyd_warshall()
```

## Single-Source Shortest Paths (SSSP)

Compute shortest-path distances from a single source to all reachable nodes,
weighted by an edge property. Compatible with LDBC Graphanalytics.

### Python API

```python
algs = db.algorithms
results = algs.sssp(source="Alix", weight_attr="cost")
for node_id, distance in results.items():
    print(f"Node {node_id}: distance {distance}")
```

### GQL / Cypher / SQL/PGQ

```sql
CALL grafeo.sssp('Alix', 'cost') YIELD node_id, distance
```

## Algorithm Complexity

| Algorithm | Time Complexity | Space |
|-----------|-----------------|-------|
| Shortest Path (BFS) | O(V + E) | O(V) |
| Shortest Path (Dijkstra) | O((V + E) log V) | O(V) |
| SSSP | O((V + E) log V) | O(V) |
| All Pairs (Floyd-Warshall) | O(V^3) | O(V^2) |
