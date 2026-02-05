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
algs = db.algorithms()

path = algs.shortest_path(source=1, target=100)

print(f"Path length: {len(path)}")
for node_id in path:
    print(f"  -> {node_id}")
```

## Dijkstra's Algorithm

Weighted shortest path using Dijkstra's algorithm.

```python
algs = db.algorithms()
path = algs.dijkstra(source=1, target=100)
```

## Breadth-First Search

Traverse the graph level by level.

```python
algs = db.algorithms()

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
algs = db.algorithms()

# DFS from a starting node
visited = algs.dfs(start=1)

# DFS visiting all nodes
all_visited = algs.dfs_all()
```

## All Pairs Shortest Paths

Precompute all pairwise distances.

```python
algs = db.algorithms()
distances = algs.all_pairs_shortest_path()
```

## Algorithm Complexity

| Algorithm | Time Complexity | Space |
|-----------|-----------------|-------|
| Shortest Path (BFS) | O(V + E) | O(V) |
| Shortest Path (Dijkstra) | O((V + E) log V) | O(V) |
| All Pairs (Floyd-Warshall) | O(V^3) | O(V^2) |
