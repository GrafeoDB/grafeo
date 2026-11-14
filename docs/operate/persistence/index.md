---
title: Persistence
description: Storage modes and data durability in Grafeo.
---

# Persistence

Grafeo supports multiple storage modes for different use cases.

## Storage Modes

| Mode | Durability | Performance | Use Case |
|------|------------|-------------|----------|
| In-Memory | None | Fastest | Testing, temporary data |
| Persistent | Full | Fast | Production workloads |

## Sections

<div class="grid cards" markdown>

-   **[In-Memory Mode](in-memory.md)**

    ---

    Fast, temporary storage for testing and caching.

-   **[Persistent Storage](persistent.md)**

    ---

    Durable storage with WAL and checkpointing.

-   **[WAL Recovery](wal.md)**

    ---

    Understanding crash recovery and durability guarantees.

</div>
