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
2. **Apply Changes** - Changes are applied to the database in memory
3. **Checkpoint** - Periodically, the WAL is merged into the database file
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

## Damage in the WAL

The end of a write that a crash cut off (a torn tail) is expected: every open cuts it and keeps the commits before it. Damage anywhere else, a record whose checksum or authentication fails with whole records after it, is refused: replaying the commits after it on top of the missing ones would build a state that never existed. The open fails with `GRAFEO-S002`, naming the file and the offset of the damage, and changes nothing.

To open such a database with the commits before the damage, salvage it at that open:

```python
db = grafeo.GrafeoDB("my_graph.grafeo", wal_salvage=True)
```

```javascript
const db = GrafeoDB.create("my_graph.grafeo", { walSalvage: true });
```

```rust
let db = GrafeoDB::with_config(Config::persistent("my_graph.grafeo").with_wal_salvage())?;
```

Nothing is deleted: what the open did not replay is kept in `<file>.wal.damaged` (`<file>.wal.damaged-2` and so on for later salvages). The setting applies to that open only; the database file does not store it.

## Checkpointing

Checkpoints merge WAL changes into the database file:

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
