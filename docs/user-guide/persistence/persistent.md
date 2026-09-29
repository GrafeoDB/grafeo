---
title: Persistent Storage
description: Using Grafeo with durable storage.
tags:
  - persistence
  - storage
---

# Persistent Storage

Persistent mode stores data durably on disk.

## Creating a Persistent Database

=== "Python"

    ```python
    import grafeo

    db = grafeo.GrafeoDB(path="my_graph.db")
    ```

=== "Rust"

    ```rust
    use grafeo::GrafeoDB;

    let db = GrafeoDB::open("my_graph.db")?;
    ```

## File Structure

A path without the `.grafeo` extension is a directory database:

```text
my_graph.db/
├── wal/            # Write-ahead log: the database's data
└── LOCK            # Held while the database is open for writing
```

A directory database keeps its data in the write-ahead log and replays it when it opens. A [single-file database](#single-file-format-grafeo) keeps its state in the `.grafeo` file and the changes since its last checkpoint in a `my_graph.grafeo.wal/` directory next to it.

## Durability Guarantees

- **Write-Ahead Logging (WAL)**: a transaction's changes are written to the WAL when it commits
- **Checkpointing**: a `.grafeo` file is brought up to date periodically and on `close()`, after which the WAL keeps only newer changes
- **Crash Recovery**: the WAL is replayed automatically when the database opens

## Sync Modes

The sync mode decides when the WAL is flushed to disk (`fsync`). Every mode hands each commit to the operating system right away, so a crash of the process loses no committed data; the modes differ in what a power loss or an operating system crash can lose.

| Mode | When the WAL is synced | Lost on power loss |
|------|------------------------|--------------------|
| `DurabilityMode::Sync` | at every commit | nothing |
| `DurabilityMode::Batch` (default: 100 ms, 1,000 records) | at a commit once the delay has passed or that many records were written | commits since the last sync |
| `DurabilityMode::Adaptive` | by a background thread, at a steady interval | commits since the last sync |
| `DurabilityMode::NoSync` | never; the operating system writes it back | commits not yet written back |

`Batch` checks its limits when a commit is written. The last commit before an idle period is therefore synced by the next write or by `close()`, not when the delay passes.

The sync mode is set through the Rust `Config` builder; the Python constructor uses the default.

```rust
use grafeo::{Config, DurabilityMode, GrafeoDB};

let config = Config::persistent("my_graph.db").with_wal_durability(DurabilityMode::Sync);
let db = GrafeoDB::with_config(config)?;
```

## Single-File Format (`.grafeo`)

Since 0.5.21, Grafeo supports a single-file database format. The entire database is stored in one `.grafeo` file with a sidecar WAL directory for crash safety.

=== "Python"

    ```python
    db = grafeo.GrafeoDB(path="my_graph.grafeo")
    ```

=== "Rust"

    ```rust
    let db = GrafeoDB::open("my_graph.grafeo")?;
    ```

Features:

- Dual-header crash safety with CRC32 checksums
- Checkpoints write the new state to `my_graph.grafeo.checkpoint` first and copy it over the database file only when it is complete, so a checkpoint that fails (for example on a full disk) leaves the last good state readable. A checkpoint needs free disk space for a second copy of the file while it runs.
- Automatic format detection: `.grafeo` extension uses single-file mode, directory paths use multi-file mode
- Exclusive file locking prevents multiple processes from opening the same file simultaneously

## Read-Only Mode

Open a database in read-only mode to allow multiple processes to read the same `.grafeo` file concurrently. Mutations are rejected at the session level.

=== "Python"

    ```python
    db = grafeo.GrafeoDB.open_read_only("my_graph.grafeo")
    ```

=== "Rust"

    ```rust
    let db = GrafeoDB::open_read_only("my_graph.grafeo")?;
    ```

Read-only mode uses a shared file lock instead of an exclusive lock, so multiple readers can coexist.

## One Writer at a Time

A persistent database can be open for writing by one `GrafeoDB` instance at a time, in one process. Opening it again, from the same process or another one, fails with a "locked by another process" error (`database file is locked by another process` for a `.grafeo` file, `database is locked by another process` for a directory) until the first instance calls `close()` or is dropped. A directory database is locked through its `LOCK` file, which it takes only when the WAL is enabled (the default).

To share a database between processes, run it behind [Grafeo Server](https://github.com/GrafeoDB/grafeo-server), or open `.grafeo` files in [read-only mode](#read-only-mode) from the readers.

## Reopening a Database

```python
# First session
db = grafeo.GrafeoDB(path="my_graph.db")
db.execute("INSERT (:Person {name: 'Alix'})")
db.close()  # releases the lock

# Later session: data persists
db = grafeo.GrafeoDB(path="my_graph.db")
result = db.execute("MATCH (p:Person) RETURN p.name")
# Returns 'Alix'
```

## Snapshots

Save and restore database snapshots for backup or migration:

=== "Python"

    ```python
    # Export snapshot
    data = db.snapshot()

    # Import snapshot (atomic, with pre-validation)
    db.restore_snapshot(data)

    # Save to file
    db.save("backup.grafeo")
    ```

=== "Rust"

    ```rust
    // Export
    let data = db.snapshot()?;

    // Restore (validates before applying)
    db.restore_snapshot(&data)?;
    ```

Snapshots include all nodes, edges, properties, labels, schema definitions, index metadata and named graph data. The current format is v4, which also preserves temporal version history.
