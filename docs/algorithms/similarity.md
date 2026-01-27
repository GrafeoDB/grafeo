---
title: Similarity Algorithms
description: Node and graph similarity measures.
tags:
  - algorithms
  - similarity
---

# Similarity Algorithms

Measure similarity between nodes or graphs.

!!! note "Coming Soon"
    These algorithms are planned for upcoming releases.

## Node Similarity

### Cosine Similarity

Based on shared neighbors.

```python
from graphos.algorithms import node_similarity

similar_nodes = node_similarity(db,
    node_id=1,
    method='cosine',
    limit=10
)
```

### Jaccard Similarity

Based on neighbor overlap.

```python
similar_nodes = node_similarity(db,
    node_id=1,
    method='jaccard',
    limit=10
)
```

### Overlap Coefficient

Minimum set comparison.

```python
similar_nodes = node_similarity(db,
    node_id=1,
    method='overlap',
    limit=10
)
```

## Pairwise Similarity

Compare two specific nodes.

```python
from graphos.algorithms import pairwise_similarity

score = pairwise_similarity(db,
    node_a=1,
    node_b=2,
    method='cosine'
)
```

## Graph Similarity

Compare entire graphs or subgraphs.

```python
from graphos.algorithms import graph_similarity

score = graph_similarity(
    graph_a,
    graph_b,
    method='edit_distance'
)
```

## Use Cases

| Algorithm | Use Case |
|-----------|----------|
| Cosine | Feature-based comparison |
| Jaccard | Set-based comparison |
| Edit Distance | Structural comparison |
