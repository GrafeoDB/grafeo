---
title: Execution Engine
description: Query execution architecture.
tags:
  - architecture
  - execution
---

# Execution Engine

Grafeo uses a push-based, vectorized execution engine.

## Why Push-Based?

In a pull (Volcano) model, each row crosses every operator boundary through a virtual function call. For graph queries that touch many small results (traversals, pattern matches), this per-row overhead dominates. Push-based execution amortizes that cost: a source produces a batch, and operators process the entire batch before passing it downstream. This also maps naturally to morsel-driven parallelism, where independent workers each drive their own pipeline on different data partitions.

## Pipeline Structure

Data flows from source to sink through a chain of operators:

```mermaid
graph LR
    SCAN[Source] --> FILTER[Filter]
    FILTER --> PROJECT[Project]
    PROJECT --> AGG[Aggregate]
    AGG --> SINK[Sink]
```

The engine is built on three core traits:

- **Source** - Produces data chunks via `next_chunk()`
- **PushOperator** - Receives chunks via `push()`, forwards results to a downstream sink
- **Sink** - Consumes output via `consume()`

Operators are either **streaming** (Filter, Project, Limit) or **pipeline breakers** that must materialize their input before producing output (Sort, Aggregate, Distinct).

## Vectorized Processing

Operations process batches of rows (DataChunks) instead of single rows. The default chunk size is 2048 rows.

**Why batch-at-a-time?** Graph queries often combine traversals (pointer-chasing, hard to vectorize) with analytical operations (filtering, aggregation, sorting). Batch processing gives the analytical parts good cache behavior and lets the compiler auto-vectorize tight loops, while the traversal parts still benefit from reduced per-row interpretation overhead.

```rust
struct DataChunk {
    columns: Vec<ValueVector>,
    selection: Option<SelectionVector>,
    count: usize,
}
```

Explicit SIMD intrinsics (AVX2, SSE, NEON) are used for vector index distance computations (dot product, Euclidean, cosine, Manhattan), not for general query execution.

## Adaptive Chunk Sizing

Chunk sizes adapt based on memory pressure:

| Pressure Level | Chunk Size |
| ---------------- | ------------ |
| Normal | 2048 rows |
| Moderate (>70% memory) | 1024 rows |
| High (>85% memory) | 512 rows |
| Critical (>95% memory) | 256 rows |

Operators can also request specific sizes via chunk size hints (Small, Default, Large, Exact, AtMost).

## Morsel-Driven Parallelism

Work is divided into morsels (64K rows by default), which are larger than DataChunks to amortize scheduling overhead. Each worker thread runs its own operator chain on independent morsels:

```text
Thread 1: [Morsel 0] -> [Morsel 4] -> [Morsel 8]
Thread 2: [Morsel 1] -> [Morsel 5] -> [Morsel 9]
Thread 3: [Morsel 2] -> [Morsel 6] -> ...
Thread 4: [Morsel 3] -> [Morsel 7] -> ...
```

The scheduler uses per-worker local queues with NUMA-aware work stealing.
