---
title: Buffer Manager
description: Unified buffer management.
tags:
  - architecture
  - memory
---

# Buffer Manager

The buffer manager controls memory allocation across all operations.

## Responsibilities

- Track total memory usage
- Allocate memory to operators
- Evict when under pressure
- Coordinate with spill manager

## Memory Pools

```
Total Memory Budget: 4 GB
├── Query Execution: 2 GB
├── Buffer Pool: 1 GB
├── Indexes: 512 MB
└── Overhead: 512 MB
```

## Eviction Policy

LRU (Least Recently Used) with priority hints:

1. Evict cached results first
2. Then intermediate results
3. Finally, active operator state
