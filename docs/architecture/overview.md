---
title: System Overview
description: High-level system architecture.
tags:
  - architecture
---

# System Overview

Graphos is designed as a high-performance, embeddable graph database.

## Design Goals

| Goal | Approach |
|------|----------|
| **Performance** | Vectorized execution, SIMD, columnar storage |
| **Embeddability** | No external dependencies, single library |
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

1. **Parser** - GQL/Cypher to AST
2. **Binder** - Semantic analysis and type checking
3. **Planner** - AST to logical plan
4. **Optimizer** - Cost-based optimization
5. **Executor** - Push-based execution

### Storage

1. **LPG Store** - Node and edge storage
2. **Property Store** - Columnar property storage
3. **Indexes** - Hash, B-tree, adjacency
4. **WAL** - Durability and recovery

### Memory

1. **Buffer Manager** - Memory allocation
2. **Arena Allocator** - Epoch-based allocation
3. **Spill Manager** - Disk spilling for large operations

## Threading Model

- **Main Thread** - Coordinates query execution
- **Worker Threads** - Parallel query processing (morsel-driven)
- **Background Thread** - Checkpointing, compaction
