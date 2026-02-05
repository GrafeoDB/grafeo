---
title: WAL Recovery
description: Understanding write-ahead logging and crash recovery.
tags:
  - persistence
  - wal
  - recovery
---

# WAL Recovery

Grafeo uses Write-Ahead Logging (WAL) to ensure durability and enable crash recovery.

## How WAL Works

1. **Log First** - All changes are written to the WAL before being applied
2. **Apply Changes** - Changes are applied to the main data files
3. **Checkpoint** - Periodically, WAL is merged into data files
4. **Truncate** - Old WAL entries are removed after checkpointing

## Crash Recovery

When opening a database after a crash:

1. Grafeo detects incomplete transactions in the WAL
2. Committed transactions are replayed
3. Uncommitted transactions are rolled back
4. Database is restored to a consistent state

```python
# Recovery happens automatically on open
db = grafeo.GrafeoDB(path="my_graph.db")
# Database is now in a consistent state
```

## Checkpointing

Checkpoints merge WAL changes into the main data files:

```python
# Manual checkpoint
db.checkpoint()

# Automatic checkpointing happens periodically
```

## Durability Levels

| Level | Description |
|-------|-------------|
| **Transaction Durability** | Each committed transaction is durable |
| **Session Durability** | All session changes persist after close |
| **Database Durability** | Data survives process crash |

## Best Practices

1. **Regular Checkpoints** - Reduce recovery time
2. **Sufficient Disk Space** - WAL can grow between checkpoints
3. **Proper Shutdown** - Use `close()` when possible
