---
title: Query Optimization
description: Cost-based query optimization.
---

# Query Optimization

Graphos uses cost-based optimization to select efficient query plans.

## Optimizer Pipeline

```mermaid
graph LR
    LP[Logical Plan] --> RW[Rewrite Rules]
    RW --> CE[Cardinality Estimation]
    CE --> CBO[Cost-Based Selection]
    CBO --> PP[Physical Plan]
```

## Sections

- [Cost Model](cost.md)
- [Cardinality Estimation](cardinality.md)
- [Join Ordering](joins.md)
