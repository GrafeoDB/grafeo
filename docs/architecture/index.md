---
title: Architecture
description: Graphos system architecture and internals.
---

# Architecture

Understand how Graphos is designed and implemented.

## Overview

Graphos is built as a modular system with clear separation of concerns:

```mermaid
graph TB
    subgraph "User Interface"
        PY[Python API]
        RS[Rust API]
    end

    subgraph "graphos-engine"
        DB[Database]
        SESS[Session Manager]
        QP[Query Processor]
        TXN[Transaction Manager]
    end

    subgraph "graphos-adapters"
        GQL[GQL Parser]
        WAL[WAL Storage]
        PLUG[Plugins]
    end

    subgraph "graphos-core"
        LPG[LPG Store]
        IDX[Indexes]
        EXEC[Execution Engine]
    end

    subgraph "graphos-common"
        TYPES[Types]
        MEM[Memory]
        UTIL[Utilities]
    end

    PY --> DB
    RS --> DB
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

    High-level architecture and design principles.

-   **[Crate Structure](crates.md)**

    ---

    The five crates and their responsibilities.

-   **[Storage Model](storage/index.md)**

    ---

    How data is stored and organized.

-   **[Execution Engine](execution/index.md)**

    ---

    Query execution and optimization.

-   **[Query Optimization](optimization/index.md)**

    ---

    Cost-based optimization strategies.

-   **[Memory Management](memory/index.md)**

    ---

    Memory allocation and management.

-   **[Transactions](transactions/index.md)**

    ---

    MVCC and isolation levels.

</div>

## Design Principles

1. **Performance First** - Optimized for graph workloads
2. **Embeddable** - Zero external dependencies
3. **Safe** - Written in safe Rust
4. **Modular** - Clear crate boundaries
5. **Extensible** - Plugin architecture
