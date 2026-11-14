---
title: Link Prediction
description: Predict missing or future relationships.
tags:
  - algorithms
  - link-prediction
---

# Link Prediction

Algorithms for predicting missing or future edges.

!!! note "NetworkX Integration"
    Link prediction algorithms are available through the NetworkX adapter.

## Using the NetworkX Adapter

```python
import grafeo
import networkx as nx

db = grafeo.GrafeoDB()

# Convert to NetworkX graph
nx_adapter = db.as_networkx(directed=False)
nx_graph = nx_adapter.to_networkx()

# Get non-edges for prediction
non_edges = list(nx.non_edges(nx_graph))[:100]  # Sample
```

## Common Neighbors

Predict links based on shared connections.

```python
preds = nx.common_neighbor_centrality(nx_graph, non_edges)
for u, v, score in preds:
    print(f"({u}, {v}): {score:.4f}")
```

## Jaccard Coefficient

Normalized common neighbors.

```python
preds = nx.jaccard_coefficient(nx_graph, non_edges)
for u, v, score in preds:
    print(f"Jaccard({u}, {v}) = {score:.4f}")
```

## Adamic-Adar Index

Weighted common neighbors (rare neighbors count more).

```python
preds = nx.adamic_adar_index(nx_graph, non_edges)
for u, v, score in preds:
    print(f"Adamic-Adar({u}, {v}) = {score:.4f}")
```

## Preferential Attachment

Product of node degrees.

```python
preds = nx.preferential_attachment(nx_graph, non_edges)
for u, v, score in preds:
    print(f"PA({u}, {v}) = {score}")
```

## Resource Allocation

Similar to Adamic-Adar but with different weighting.

```python
preds = nx.resource_allocation_index(nx_graph, non_edges)
for u, v, score in preds:
    print(f"RA({u}, {v}) = {score:.4f}")
```

## Use Cases

- Friend suggestions in social networks
- Product recommendations
- Knowledge graph completion
- Collaboration prediction
