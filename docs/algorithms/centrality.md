---
title: Centrality Algorithms
description: Centrality measures for graph analysis.
tags:
  - algorithms
  - centrality
---

# Centrality Algorithms

Centrality algorithms identify the most important nodes in a graph.

!!! note "Coming Soon"
    These algorithms are planned for upcoming releases.

## PageRank

Measures node importance based on incoming links.

```python
from graphos.algorithms import pagerank

scores = pagerank(db,
    damping=0.85,
    iterations=20
)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `damping` | 0.85 | Probability of following a link |
| `iterations` | 20 | Maximum iterations |
| `tolerance` | 1e-6 | Convergence threshold |

### Use Cases

- Search engine ranking
- Social influence analysis
- Citation importance

## Betweenness Centrality

Measures how often a node lies on shortest paths.

```python
from graphos.algorithms import betweenness_centrality

scores = betweenness_centrality(db)
```

### Use Cases

- Identifying bridges/brokers
- Network vulnerability analysis
- Information flow bottlenecks

## Closeness Centrality

Measures average distance to all other nodes.

```python
from graphos.algorithms import closeness_centrality

scores = closeness_centrality(db)
```

### Use Cases

- Identifying well-connected nodes
- Optimal placement problems
- Influence spread analysis

## Degree Centrality

Simple count of connections.

```python
from graphos.algorithms import degree_centrality

scores = degree_centrality(db,
    direction='both'  # 'in', 'out', or 'both'
)
```

### Use Cases

- Quick importance estimate
- Hub identification
- Activity analysis

## Eigenvector Centrality

Importance based on neighbor importance.

```python
from graphos.algorithms import eigenvector_centrality

scores = eigenvector_centrality(db)
```

### Use Cases

- Social influence
- Similar to PageRank but undirected
- Prestige measurement
