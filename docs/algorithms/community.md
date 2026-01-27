---
title: Community Detection
description: Community detection algorithms.
tags:
  - algorithms
  - community
---

# Community Detection

Find clusters and communities within graphs.

!!! note "Coming Soon"
    These algorithms are planned for upcoming releases.

## Louvain Algorithm

Fast modularity-based community detection.

```python
from graphos.algorithms import louvain

communities = louvain(db,
    resolution=1.0
)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `resolution` | 1.0 | Higher = smaller communities |

## Label Propagation

Semi-supervised community detection.

```python
from graphos.algorithms import label_propagation

communities = label_propagation(db,
    max_iterations=100
)
```

## Connected Components

Find disconnected subgraphs.

```python
from graphos.algorithms import connected_components

components = connected_components(db)

print(f"Found {len(components)} components")
for i, comp in enumerate(components):
    print(f"Component {i}: {len(comp)} nodes")
```

## Strongly Connected Components

For directed graphs.

```python
from graphos.algorithms import strongly_connected_components

sccs = strongly_connected_components(db)
```

## Triangle Count

Count triangles for clustering analysis.

```python
from graphos.algorithms import triangle_count

triangles = triangle_count(db)
print(f"Total triangles: {triangles['total']}")
```

## Use Cases

| Algorithm | Best For |
|-----------|----------|
| Louvain | Large graphs, quality clusters |
| Label Propagation | Fast, scalable |
| Connected Components | Graph structure analysis |
| Triangle Count | Clustering coefficient |
