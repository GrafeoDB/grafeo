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
print(f"Nodes: {db.node_count()}")
print(f"Edges: {db.edge_count()}")

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
