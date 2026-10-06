---
title: System Overview
description: High-level system architecture.
tags:
  - architecture
---

# System Overview

Grafeo is designed as a high-performance, embeddable graph database.

## Design Goals

| Goal | Approach |
|------|----------|
| **Performance** | Batch-at-a-time vectorized execution, columnar storage |
| **Embeddability** | No required C dependencies, single library |
| **Safety** | Pure Rust, memory-safe by design |
| **Flexibility** | Plugin architecture, multiple storage backends |

## Query Flow

```mermaid
sequenceDiagram
    participant Client
    participant Session
    participant Parser
    participant Planner
    participant Optimizer
    participant Executor
    participant Storage

    Client->>Session: execute(query)
    Session->>Parser: parse(query)
    Parser-->>Session: AST
    Session->>Planner: plan(AST)
    Planner-->>Session: Logical Plan
    Session->>Optimizer: optimize(plan)
    Optimizer-->>Session: Physical Plan
    Session->>Executor: execute(plan)
    Executor->>Storage: scan/lookup
    Storage-->>Executor: data
    Executor-->>Session: results
    Session-->>Client: QueryResult
```

## Key Components

### Query Processing

1. **Parser** - GQL/Cypher/SPARQL/Gremlin/GraphQL/SQL-PGQ to AST
2. **Binder** - Semantic analysis and type checking
3. **Planner** - AST to logical plan
4. **Optimizer** - Cost-based optimization
5. **Executor** - Push-based execution

### Storage

1. **LPG Store** - Node and edge storage
2. **Property Store** - Columnar property storage
3. **Indexes** - Hash, adjacency, trie, vector (HNSW), text (BM25), [ring](storage/ring-index.md) (wavelet-tree compact RDF triple index)
4. **WAL** - Durability and recovery

### Memory

1. **Buffer Manager** - Unified memory budget (75% of system RAM by default)
2. **Arena Allocator** - Epoch-based bulk allocation for query execution
3. **Spill Manager** - Disk spilling for large operations

## Threading Model

- **Main Thread** - Coordinates query execution
- **Worker Threads** - Parallel query processing (morsel-driven)
- **Background Thread** - Checkpointing, compaction
