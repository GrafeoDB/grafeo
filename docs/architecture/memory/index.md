---
title: Memory Management
description: Memory allocation and management.
tags:
  - architecture
  - memory
---

# Memory Management

Grafeo manages memory through a unified buffer manager, arena allocators for query execution and transparent spill-to-disk for large operations.

## Buffer Manager

The buffer manager controls a single memory budget shared across all operations. By default it claims 75% of system RAM (falling back to 1 GB if detection fails).

Memory is tracked across four regions that share the unified budget:

| Region | Contents |
| ------ | -------- |
| GraphStorage | Nodes, edges, properties, adjacency lists |
| IndexBuffers | B-tree, hash, trie indexes |
| ExecutionBuffers | DataChunks, hash tables, sort buffers |
| SpillStaging | Staging area for operators under memory pressure |

### Pressure Levels

The buffer manager responds to memory pressure in stages:

| Level | Threshold | Action |
| ----- | --------- | ------ |
| Normal | < 70% | No action |
| Moderate | 70-85% | Proactive eviction of cold data |
| High | 85-95% | Aggressive eviction, trigger spilling |
| Critical | > 95% | Block new allocations |

Eviction follows LRU (Least Recently Used) with priority hints: cached results are evicted first, then intermediate results, then active operator state.

## Arena Allocators

Query execution allocates many short-lived objects (intermediate tuples, hash table entries, sort buffers) that all share the same lifetime: the query. Arenas let these allocate with a simple pointer bump and deallocate in one bulk reset at query end, avoiding per-object free overhead and heap fragmentation.

```text
Arena:
├── Chunk 1 (1MB): [allocated][allocated][free...]
├── Chunk 2 (1MB): [allocated][free.............]
└── Chunk 3 (1MB): [free......................]

Allocation: Bump pointer in current chunk
Deallocation: Reset entire arena at once
```

```rust
use grafeo_common::memory::Arena;
use grafeo_common::types::EpochId;

let arena = Arena::new(EpochId::INITIAL).unwrap();

// Allocate within arena
let data = arena.alloc_value(node).unwrap();
let more = arena.alloc_slice(&values).unwrap();
```

## Spill to Disk

Large operations can spill to disk when memory is exhausted. Spilling is triggered automatically when the buffer manager reaches High or Critical pressure levels.

| Operator | Spill Strategy |
| -------- | -------------- |
| Hash Join | Partition both sides by hash, spill partitions that don't fit, process spilled partitions later |
| Aggregate | Partition by group key |
| Sort | External merge sort: sort chunks in memory, write sorted runs to disk, merge runs |

### Configuration

```rust
use grafeo_engine::Config;

let config = Config::default()
    .with_memory_limit(4 * 1024 * 1024 * 1024)  // 4 GB explicit limit
    .with_spill_path("/tmp/grafeo-spill");       // Directory for spill files

let db = GrafeoDB::with_config(config);
```
