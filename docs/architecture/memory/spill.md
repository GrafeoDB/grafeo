---
title: Spill to Disk
description: Transparent spilling for out-of-core operations.
tags:
  - architecture
  - memory
---

# Spill to Disk

Large operations can spill to disk when memory is exhausted.

## Spill-Capable Operators

| Operator | Spill Strategy |
|----------|---------------|
| Hash Join | Partition both sides |
| Aggregate | Partition by group key |
| Sort | External merge sort |

## Hash Join Spilling

```
When build side exceeds memory:
1. Partition data by hash
2. Spill partitions that don't fit
3. Process in-memory partitions
4. Load and process spilled partitions
```

## Sort Spilling

```
1. Sort chunks in memory
2. Write sorted runs to disk
3. Merge sorted runs
```

## Configuration

```python
db = grafeo.GrafeoDB(
    memory_limit=4 * 1024 * 1024 * 1024,  # 4 GB
    spill_directory="/tmp/grafeo_spill"
)
```
