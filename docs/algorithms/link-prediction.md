---
title: Link Prediction
description: Predict missing or future relationships.
tags:
  - algorithms
  - link-prediction
---

# Link Prediction

Algorithms for predicting missing or future edges.

!!! note "Coming Soon"
    These algorithms are planned for upcoming releases.

## Common Neighbors

Predict links based on shared connections.

```python
from graphos.algorithms import common_neighbors

predictions = common_neighbors(db,
    node_a=1,
    node_b=2
)

print(f"Common neighbor score: {predictions.score}")
```

## Jaccard Coefficient

Normalized common neighbors.

```python
from graphos.algorithms import jaccard_coefficient

score = jaccard_coefficient(db, node_a, node_b)
```

## Adamic-Adar Index

Weighted common neighbors (rare neighbors count more).

```python
from graphos.algorithms import adamic_adar

score = adamic_adar(db, node_a, node_b)
```

## Preferential Attachment

Product of node degrees.

```python
from graphos.algorithms import preferential_attachment

score = preferential_attachment(db, node_a, node_b)
```

## Resource Allocation

Similar to Adamic-Adar but with different weighting.

```python
from graphos.algorithms import resource_allocation

score = resource_allocation(db, node_a, node_b)
```

## Batch Predictions

Generate predictions for many node pairs.

```python
from graphos.algorithms import predict_links

predictions = predict_links(db,
    method='adamic_adar',
    limit=100
)

for source, target, score in predictions:
    print(f"{source} -> {target}: {score:.4f}")
```

## Use Cases

- Friend suggestions in social networks
- Product recommendations
- Knowledge graph completion
- Collaboration prediction
