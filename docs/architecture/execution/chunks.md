---
title: Adaptive Chunks
description: Adaptive chunk sizing.
tags:
  - architecture
  - execution
---

# Adaptive Chunks

Chunk sizes adapt based on workload characteristics.

## Chunk Size Selection

| Factor | Smaller Chunks | Larger Chunks |
|--------|---------------|---------------|
| Selectivity | High (few pass filter) | Low (many pass) |
| LIMIT | Small limit | No limit |
| Memory | Limited memory | Ample memory |

## Adaptive Strategy

1. Start with default chunk size (1024-2048 rows)
2. Monitor selectivity and throughput
3. Adjust chunk size for subsequent operations
4. Balance cache efficiency vs overhead

## Benefits

- **LIMIT optimization** - Small chunks for early termination
- **Memory efficiency** - Smaller chunks for memory-constrained ops
- **Throughput** - Larger chunks for full scans
