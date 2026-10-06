---
title: Graph Metrics
description: Compute graph-level statistics.
tags:
  - algorithms
  - metrics
---

# Graph Metrics

Compute statistics that describe the overall graph structure.

## Basic Metrics

```python
import grafeo

db = grafeo.GrafeoDB()

# Basic counts via database methods
print(f"Nodes: {db.node_count}")
print(f"Edges: {db.edge_count}")

# Additional metrics via algorithms
algs = db.algorithms()
```

## Transitivity (Clustering Coefficient)

Global measure of how clustered the graph is.

```python
algs = db.algorithms()
transitivity = algs.transitivity()
print(f"Transitivity: {transitivity:.4f}")
```

## Triangle Count

Count triangles for clustering analysis.

```python
algs = db.algorithms()
triangles = algs.triangles()
print(f"Total triangles: {triangles}")

# Parallel version for large graphs (degree-ordered, merge intersection)
triangles = algs.triangles(parallel=True)
```

## K-Truss Decomposition

Find dense subgraphs where every edge is supported by at least k-2 triangles. The truss number of an edge is the maximum k for which it belongs to the k-truss.

```python
algs = db.algorithms()

# Full decomposition: truss number for every edge
result = algs.ktruss()
print(f"Max truss: {result['max_truss']}")

# Extract edges in the 4-truss
edges = algs.ktruss(k=4)
print(f"Edges in 4-truss: {len(edges)}")
```

## Subgraph Isomorphism

Count occurrences of a pattern subgraph within the graph. Uses VF2 backtracking with degree pruning.

```python
algs = db.algorithms()

# Count triangles via subgraph isomorphism (should match triangles() * 6)
count = algs.subgraph_isomorphism_count(
    pattern_edges=[(0, 1), (1, 2), (2, 0)],
    pattern_nodes=3
)
```

## Degree Distribution

Use the NetworkX adapter for degree statistics:

```python
nx_adapter = db.as_networkx(directed=True)
dist = nx_adapter.degree_distribution()

for degree, count in sorted(dist.items()):
    print(f"Degree {degree}: {count} nodes")
```

## Summary Table

| Metric | Range | Interpretation |
|--------|-------|----------------|
| Density | 0-1 | Higher = more connected |
| Transitivity | 0-1 | Higher = more clustered |
| Avg Degree | 0-n | Higher = more edges per node |
