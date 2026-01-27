---
title: Join Ordering
description: DPccp algorithm for join ordering.
tags:
  - architecture
  - optimization
---

# Join Ordering

Graphos uses DPccp (Dynamic Programming connected complement pairs) for join ordering.

## The Problem

For n tables, there are (2n-2)!/((n-1)!) possible join orders.

| Tables | Possible Orders |
|--------|----------------|
| 3 | 12 |
| 5 | 1,680 |
| 10 | ~17 billion |

## DPccp Algorithm

1. Enumerate all connected subgraphs
2. For each subgraph, find optimal join
3. Build up larger plans from smaller optimal plans
4. Prune dominated plans

## Join Selection

| Join Type | Best When |
|-----------|-----------|
| Hash Join | Large inputs, equality predicates |
| Nested Loop | Small inner, indexed |
| Merge Join | Sorted inputs |
