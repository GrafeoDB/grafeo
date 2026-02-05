---
title: Centrality Algorithms
description: Centrality measures for graph analysis.
tags:
  - algorithms
  - centrality
---

# Centrality Algorithms

Centrality algorithms identify the most important nodes in a graph.

## PageRank

Measures node importance based on incoming links.

```python
import grafeo

db = grafeo.GrafeoDB()
algs = db.algorithms()

scores = algs.pagerank()

for node_id, score in scores.items():
    print(f"Node {node_id}: {score:.4f}")
```

### Use Cases

- Search engine ranking
- Social influence analysis
- Citation importance

## Betweenness Centrality

Measures how often a node lies on shortest paths.

```python
algs = db.algorithms()
scores = algs.betweenness_centrality()
```

### Use Cases

- Identifying bridges/brokers
- Network vulnerability analysis
- Information flow bottlenecks

## Closeness Centrality

Measures average distance to all other nodes.

```python
algs = db.algorithms()
scores = algs.closeness_centrality()
```

### Use Cases

- Identifying well-connected nodes
- Optimal placement problems
- Influence spread analysis

## Degree Centrality

Simple count of connections.

```python
algs = db.algorithms()
scores = algs.degree_centrality()
```

### Use Cases

- Quick importance estimate
- Hub identification
- Activity analysis

## Eigenvector Centrality

Importance based on neighbor importance.

```python
algs = db.algorithms()
scores = algs.eigenvector_centrality()
```

### Use Cases

- Social influence
- Similar to PageRank but undirected
- Prestige measurement
