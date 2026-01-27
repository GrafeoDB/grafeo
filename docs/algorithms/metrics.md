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
from graphos.algorithms import graph_metrics

metrics = graph_metrics(db)

print(f"Nodes: {metrics.node_count}")
print(f"Edges: {metrics.edge_count}")
print(f"Density: {metrics.density:.4f}")
print(f"Avg Degree: {metrics.average_degree:.2f}")
```

## Density

Ratio of actual edges to possible edges.

```python
from graphos.algorithms import density

d = density(db)
# 0 = no edges, 1 = complete graph
```

## Diameter

Longest shortest path in the graph.

```python
from graphos.algorithms import diameter

d = diameter(db)
print(f"Graph diameter: {d}")
```

## Radius

Minimum eccentricity (shortest max distance from any node).

```python
from graphos.algorithms import radius

r = radius(db)
```

## Clustering Coefficient

Measure of how clustered the graph is.

```python
from graphos.algorithms import clustering_coefficient

# Global clustering coefficient
global_cc = clustering_coefficient(db, method='global')

# Local clustering coefficients
local_cc = clustering_coefficient(db, method='local')

# Average local
avg_cc = clustering_coefficient(db, method='average')
```

## Degree Distribution

```python
from graphos.algorithms import degree_distribution

dist = degree_distribution(db)

for degree, count in sorted(dist.items()):
    print(f"Degree {degree}: {count} nodes")
```

## Summary Table

| Metric | Range | Interpretation |
|--------|-------|----------------|
| Density | 0-1 | Higher = more connected |
| Diameter | 1-n | Lower = more compact |
| Clustering | 0-1 | Higher = more clustered |
| Avg Degree | 0-n | Higher = more edges per node |
