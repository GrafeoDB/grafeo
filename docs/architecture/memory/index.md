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

The buffer manager controls a single memory budget shared across all operations. By default it claims 75% of the memory the process may use: the smaller of the machine's memory and any limit set on the process from outside (a cgroup level on Linux, at the process's own cgroup or any parent; a job object on Windows). When neither can be read the figure is 1 GiB, a 768 MiB budget, and the open logs a warning. In the browser (WebAssembly) the figure is a fixed 1 GiB by design, a 768 MiB budget, with no warning: a tab's real ceiling depends on the browser and the device. `Config::with_memory_fraction` takes another share of the same figure, detected at each open and never stored in the file. `Config::with_memory_limit` sets bytes: a database file stores the limit given when it is created, or later through `set_stored_config`; a limit given at a later open applies to that open only.

### What the limit counts

The memory limit bounds what the database's own structures hold: the store's tables, columns and versions, its indexes, the RDF store, the change data capture log, open transactions' change sets and the queries' operator buffers. Each counts the capacity it allocates, as it grows. The process uses more than that: the allocator's own bookkeeping and the free space it keeps, whole pages where the structures use part of one, and memory the database does not account for (thread stacks, the plan and query caches, the language runtime of a binding). Measured on Linux against the process's private memory (`Private_Clean + Private_Dirty` in `/proc/self/smaps_rollup`), the counted bytes were 72% of the growth for 100,000 nodes with eight integer properties, and 76% for 10,000 nodes with a 384-dimension vector and its index. So a process holding a database at a limit L grows to about 1.4 times L (1 / 0.72; 1.3 times in the vector case), roughly 40% more than the limit: leave that margin when choosing it. To keep the process under 4 GiB, for example, set a limit of about 2.5 GiB, which keeps some room beside that margin.

At the limit, a statement that needs more memory than the database can free (by evicting or spilling what it can) fails with `GRAFEO-S001` and changes nothing; reads keep working. A database that already holds more than the limit when it opens (a file written under a higher limit, or a limit lowered since) still opens: `open_warnings()` reports `MEMORY_LIMIT_PASSED`, and writes that need memory fail until the limit is raised.

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
| High | 85-<100% | Aggressive eviction, trigger spilling |
| Critical | 100% | Refuse what does not fit after eviction (`GRAFEO-S001`) |

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

Under memory pressure, embedding columns spill by row group, and each graph's newest row group keeps its column in memory, since new rows go there. After an explicit spill (`TierOverride::ForceDisk` through `Config::with_section_tier`, or a spill asked for by name) the newest group is on disk too, and a write into it reads its column back into memory first, so at the memory limit it can be refused until a later release, which writes such rows without reading the spilled part back ([#713](https://github.com/GrafeoDB/grafeo/issues/713)).

### Configuration

```rust
use grafeo_engine::Config;

let config = Config::default()
    .with_memory_limit(4 * 1024 * 1024 * 1024)  // 4 GB explicit limit
    .with_spill_path("/tmp/grafeo-spill");       // Directory for spill files

let db = GrafeoDB::with_config(config);
```
