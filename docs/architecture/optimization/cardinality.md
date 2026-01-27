---
title: Cardinality Estimation
description: Estimating result set sizes.
tags:
  - architecture
  - optimization
---

# Cardinality Estimation

Accurate cardinality estimation is crucial for plan selection.

## Statistics Collected

| Statistic | Purpose |
|-----------|---------|
| Row count | Base cardinality |
| Distinct count | Join estimation |
| Histograms | Range selectivity |
| Null fraction | Null handling |

## Selectivity Estimation

```
// Equality predicate
selectivity = 1 / distinct_count

// Range predicate
selectivity = (high - low) / (max - min)

// Join
output_rows = (rows_a * rows_b) / max(distinct_a, distinct_b)
```

## ANALYZE Command

```sql
ANALYZE  -- Collect statistics for all tables
```
