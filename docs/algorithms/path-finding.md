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
from graphos.algorithms import shortest_path

path = shortest_path(db,
    source=node_a,
    target=node_b
)

print(f"Path length: {len(path.nodes) - 1}")
for node in path.nodes:
    print(f"  -> {node}")
```

### Weighted Shortest Path

```python
path = shortest_path(db,
    source=node_a,
    target=node_b,
    weight_property='distance'
)

print(f"Total weight: {path.total_weight}")
```

## All Shortest Paths

Find all paths of minimum length.

```python
from graphos.algorithms import all_shortest_paths

paths = all_shortest_paths(db,
    source=node_a,
    target=node_b
)

print(f"Found {len(paths)} shortest paths")
```

## Single Source Shortest Paths

Find shortest paths from one node to all others.

```python
from graphos.algorithms import sssp

distances = sssp(db, source=node_a)

for node_id, distance in distances.items():
    print(f"Node {node_id}: distance {distance}")
```

## All Pairs Shortest Paths

Precompute all pairwise distances.

```python
from graphos.algorithms import apsp

distances = apsp(db)

# Query distance between any two nodes
d = distances.get(node_a, node_b)
```

## K-Shortest Paths

Find k shortest paths (may not be disjoint).

```python
from graphos.algorithms import k_shortest_paths

paths = k_shortest_paths(db,
    source=node_a,
    target=node_b,
    k=5
)
```

## Algorithm Complexity

| Algorithm | Time Complexity | Space |
|-----------|-----------------|-------|
| Shortest Path (BFS) | O(V + E) | O(V) |
| Shortest Path (Dijkstra) | O((V + E) log V) | O(V) |
| SSSP | O((V + E) log V) | O(V) |
| APSP (Floyd-Warshall) | O(V^3) | O(V^2) |
| K-Shortest | O(K * V * (E + V log V)) | O(K * V) |
