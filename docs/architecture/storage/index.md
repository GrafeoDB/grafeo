---
title: Storage Model
description: How Graphos stores graph data.
---

# Storage Model

Graphos uses a hybrid storage model optimized for graph workloads.

## Overview

```mermaid
graph TB
    subgraph "Logical Layer"
        NODES[Nodes]
        EDGES[Edges]
        PROPS[Properties]
    end

    subgraph "Physical Layer"
        COLS[Columnar Storage]
        ADJ[Adjacency Lists]
        IDX[Indexes]
    end

    subgraph "Persistence Layer"
        WAL[Write-Ahead Log]
        DATA[Data Files]
    end

    NODES --> COLS
    EDGES --> ADJ
    PROPS --> COLS
    COLS --> DATA
    ADJ --> DATA
    IDX --> DATA
    WAL --> DATA
```

## Sections

<div class="grid cards" markdown>

-   **[Columnar Properties](columnar.md)**

    ---

    Type-specific columnar storage for properties.

-   **[Adjacency Lists](adjacency.md)**

    ---

    Chunked adjacency lists for edge traversal.

-   **[Zone Maps](zone-maps.md)**

    ---

    Statistics for data skipping.

-   **[Compression](compression.md)**

    ---

    Type-specific compression strategies.

</div>
