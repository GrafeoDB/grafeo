---
title: Zone Maps
description: Statistics for predicate pushdown and data skipping.
tags:
  - architecture
  - storage
---

# Zone Maps

Zone maps store statistics about data chunks to enable data skipping.

## What Zone Maps Store

| Statistic | Purpose |
|-----------|---------|
| Min value | Skip chunks where max < filter value |
| Max value | Skip chunks where min > filter value |
| Null count | Skip chunks with no nulls for IS NULL |
| Distinct estimate | Cardinality estimation |
| Bloom filter | Point lookups |

## Example

```
Query: WHERE age > 50

Chunk 0: min=20, max=45  -> SKIP (max < 50)
Chunk 1: min=30, max=60  -> SCAN (range overlaps)
Chunk 2: min=55, max=80  -> SCAN (range overlaps)
Chunk 3: min=18, max=35  -> SKIP (max < 50)
```

## Predicate Support

| Predicate | Zone Map Check |
|-----------|----------------|
| `x = v` | min <= v <= max |
| `x > v` | max > v |
| `x < v` | min < v |
| `x >= v` | max >= v |
| `x <= v` | min <= v |
| `x IS NULL` | null_count > 0 |
| `x IN (...)` | bloom filter check |
