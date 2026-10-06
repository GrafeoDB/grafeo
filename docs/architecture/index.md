---
title: Architecture
description: Grafeo system architecture and internals.
---

# Architecture

Understand how Grafeo is designed and implemented.

## Overview

Grafeo is built as a modular system with clear separation of concerns:

```mermaid
graph TB
    subgraph "User Interface"
        PY[Python API]
        RS[Rust API]
        JS[Node.js API]
        WA[WASM API]
        CC[C/Go/C#/Dart FFI]
    end

    subgraph "grafeo-engine"
        DB[Database]
        SESS[Session Manager]
        QP[Query Processor]
        TXN[Transaction Manager]
    end

    subgraph "grafeo-adapters"
        GQL[GQL Parser]
        WAL[WAL Storage]
        PLUG[Plugins]
    end

    subgraph "grafeo-core"
        LPG[LPG Store]
        IDX[Indexes]
        EXEC[Execution Engine]
    end

    subgraph "grafeo-common"
        TYPES[Types]
        MEM[Memory]
        UTIL[Utilities]
    end

    PY --> DB
    RS --> DB
    JS --> DB
    WA --> DB
    CC --> DB
    DB --> SESS
    SESS --> QP
    SESS --> TXN
    QP --> GQL
    QP --> EXEC
    TXN --> LPG
    EXEC --> LPG
    EXEC --> IDX
    LPG --> TYPES
    IDX --> MEM
```

## Sections

<div class="grid cards" markdown>

-   **[System Overview](overview.md)**

    ---

    High-level architecture, design principles and query flow.

-   **[Crate Structure](crates.md)**

    ---

    The crates and their responsibilities.

-   **[Storage Model](storage/index.md)**

    ---

    Columnar properties, chunked adjacency lists, compression.

-   **[Execution Engine](execution/index.md)**

    ---

    Push-based vectorized execution and parallelism.

-   **[Query Optimization](optimization/index.md)**

    ---

    Cost-based optimization, join ordering, cardinality estimation.

-   **[Memory Management](memory/index.md)**

    ---

    Buffer manager, arena allocators, spill-to-disk.

-   **[Transactions](transactions/index.md)**

    ---

    MVCC, snapshot isolation, conflict detection.

</div>

## Design Principles

1. **Performance First** - Batch-at-a-time vectorized execution, columnar storage, morsel-driven parallelism
2. **Embeddable** - No required C dependencies, single library
3. **Safe** - Written in safe Rust, memory-safe by design
4. **Modular** - Clear crate boundaries, strict layering
5. **Extensible** - Plugin architecture, multiple storage backends
