---
title: Similarity Algorithms
description: Node and graph similarity measures.
tags:
  - algorithms
  - similarity
---

# Similarity Algorithms

Measure similarity between nodes or graphs.

!!! note "Advanced Feature"
    Some similarity algorithms are available through the NetworkX adapter for extended functionality.

## Using the NetworkX Adapter

For advanced similarity measures, use the NetworkX integration:

```python
import grafeo

db = grafeo.GrafeoDB()

# Convert to NetworkX graph
nx_adapter = db.as_networkx(directed=False)
nx_graph = nx_adapter.to_networkx()

# Use NetworkX similarity functions
import networkx as nx

# Jaccard coefficient for potential edges
preds = nx.jaccard_coefficient(nx_graph, [(1, 2), (1, 3)])
for u, v, score in preds:
    print(f"Jaccard({u}, {v}) = {score:.4f}")
```

## Node Similarity via Algorithms API

Basic similarity through the algorithms interface:

```python
algs = db.algorithms()

# Use clustering coefficient as a similarity proxy
transitivity = algs.transitivity()
```

## Use Cases

| Algorithm | Use Case |
|-----------|----------|
| Jaccard | Set-based comparison |
| Cosine | Feature-based comparison |
| Common Neighbors | Link prediction |
