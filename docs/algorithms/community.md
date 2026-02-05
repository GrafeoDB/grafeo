---
title: Community Detection
description: Community detection algorithms.
tags:
  - algorithms
  - community
---

# Community Detection

Find clusters and communities within graphs.

## Louvain Algorithm

Fast modularity-based community detection.

```python
import grafeo

db = grafeo.GrafeoDB()
algs = db.algorithms()

communities = algs.louvain()
for community_id, members in communities.items():
    print(f"Community {community_id}: {len(members)} members")
```

## Label Propagation

Semi-supervised community detection.

```python
algs = db.algorithms()
communities = algs.label_propagation()
```

## Connected Components

Find disconnected subgraphs.

```python
algs = db.algorithms()
components = algs.connected_components()

print(f"Found {len(components)} components")
for i, comp in enumerate(components):
    print(f"Component {i}: {len(comp)} nodes")
```

## Strongly Connected Components

For directed graphs.

```python
algs = db.algorithms()
sccs = algs.strongly_connected_components()
```

## Weakly Connected Components

For directed graphs, ignoring edge direction.

```python
algs = db.algorithms()
wccs = algs.weakly_connected_components()
```

## Triangle Count

Count triangles for clustering analysis.

```python
algs = db.algorithms()
triangles = algs.triangles()
print(f"Total triangles: {triangles}")
```

## Use Cases

| Algorithm | Best For |
|-----------|----------|
| Louvain | Large graphs, quality clusters |
| Label Propagation | Fast, scalable |
| Connected Components | Graph structure analysis |
| Triangle Count | Clustering coefficient |
