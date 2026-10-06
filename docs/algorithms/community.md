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

Fast modularity-based community detection. Louvain repeats two phases until a
round changes nothing: each node moves to the neighbouring community that raises
modularity the most, then every community becomes a single node of a smaller
graph. A higher `resolution` gives smaller communities.

```python
import grafeo

db = grafeo.GrafeoDB()
# ... create nodes and edges ...

result = db.algorithms.louvain()  # resolution=1.0
print(f"{result['num_communities']} communities, modularity {result['modularity']:.3f}")
for node_id, community_id in result["communities"].items():
    print(f"Node {node_id}: community {community_id}")
```

The same graph always gives the same communities, numbered 0, 1, 2, ... in order
of their smallest node id. Edge direction is ignored.

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

## Stochastic Block Partition

Bayesian community detection using the degree-corrected stochastic block model. Minimizes the description length (MDL) of the graph under the generative model, making it more principled than modularity-based methods for graphs with heterogeneous degree distributions.

```python
algs = db.algorithms()

# Auto-detect optimal number of blocks
result = algs.stochastic_block_partition()
print(f"Found {result['num_blocks']} blocks (DL: {result['description_length']:.2f})")

# Target a specific number of blocks
result = algs.stochastic_block_partition(num_blocks=4)

# Incremental update after adding edges (warm start from prior partition)
result2 = algs.stochastic_block_partition(prior_partition=result['partition'])
```

Part of the [GraphChallenge](https://graphchallenge.mit.edu/) streaming benchmark suite.

## Partition Quality Metrics

Compare two community assignments (e.g., predicted vs ground truth):

```python
from grafeo.algorithms import metrics

ri  = metrics.rand_index(partition_a, partition_b)
ari = metrics.adjusted_rand_index(partition_a, partition_b)
nmi = metrics.normalized_mutual_information(partition_a, partition_b)

prec = metrics.pairwise_precision(predicted, truth)
rec  = metrics.pairwise_recall(predicted, truth)
```

| Metric | Range | 1.0 means |
|--------|-------|-----------|
| Rand Index | [0, 1] | Identical partitions |
| Adjusted Rand Index | [-1, 1] | Identical (0 = random) |
| Normalized Mutual Information | [0, 1] | Identical partitions |
| Pairwise Precision | [0, 1] | Every predicted co-cluster is true |
| Pairwise Recall | [0, 1] | Every true co-cluster is predicted |

## Use Cases

| Algorithm | Best For |
|-----------|----------|
| Louvain | Large graphs, quality clusters |
| Label Propagation | Fast, scalable |
| Stochastic Block Partition | Heterogeneous graphs, Bayesian inference |
| Connected Components | Graph structure analysis |
| Triangle Count | Clustering coefficient |
