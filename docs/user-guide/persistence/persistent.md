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

- Two alternating database headers, and a CRC-32 checksum on every header and every piece of data
- Checkpoints are copy-on-write: a checkpoint writes the new state into space in the file that the last good state does not use, and switches the database header to it only once it is on disk. A checkpoint that fails or is cut off by a crash (for example on a full disk) leaves the last good state readable. A checkpoint needs free disk space for a second copy of the data while it runs; the next checkpoint reuses the space of the older copy.
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

Read-only mode uses a shared file lock instead of an exclusive lock, so multiple readers can coexist. A file written by 0.5.x is read into memory once instead, and is not migrated (see [Upgrading from 0.5](#upgrading-from-05)).

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

## Upgrading from 0.5

Grafeo 0.6.0 writes `.grafeo` files in a new format. A `.grafeo` file written by 0.5.x is migrated the first time 0.6 opens it for writing: with `GrafeoDB(path=...)` in Python, `GrafeoDB::open` in Rust, a read-write open in another binding, or a command of the `grafeo` command line tool.

The migration reads the old database, including the changes in its WAL, and writes it to a new file (named `my_graph.grafeo.migrating` while it is written, so the migration needs free disk space for a copy of the database). It then renames the old files and gives the new file the database's name. The old files are kept, byte for byte:

| Written by 0.5.x | Kept as |
|------------------|---------|
| `my_graph.grafeo` | `my_graph.grafeo.pre-0.6` |
| `my_graph.grafeo.wal/` | `my_graph.grafeo.pre-0.6.wal/` |
| `my_graph.grafeo.checkpoint` (a checkpoint 0.5.44 left pending) | `my_graph.grafeo.pre-0.6.checkpoint` |

A migration never replaces a kept copy: while one of these names is taken, the open fails before it writes anything. If a migration fails or is cut off by a crash, the next read-write open finishes it or starts it again; the old files are never changed. A read-write open in another process waits up to five seconds for a running migration, then fails with "database locked: a migration is running".

0.7.0 will no longer read 0.5.x files: open each 0.5.x database once with 0.6, for writing, before you upgrade to 0.7.

### Before the First Open

Stop every 0.5.x process that uses the database. 0.5.x cannot open the migrated file, and while a 0.5.x process has the file open for writing, the migration fails and changes nothing.

### Read-Only Opens

A read-only open (`GrafeoDB.open_read_only()` in Python, `GrafeoDB::open_read_only` or `Config::read_only` in Rust) and `open_in_memory()` read a 0.5.x file, with its WAL, without migrating or changing it. A read-only open loads such a file into memory once and then holds no lock on it. If a migration was cut off after the old file was renamed, read-only opens and `open_in_memory()` fail until a read-write open has finished it.

### Encrypted Databases

A read-write open with a key (`Config::encryption`) migrates a 0.5.x file into an encrypted file. The kept files are not encrypted, as 0.5.x never encrypted its files: `my_graph.grafeo.pre-0.6`, `my_graph.grafeo.pre-0.6.wal/` and, if present, `my_graph.grafeo.pre-0.6.checkpoint`. Remove all of them once you no longer need to go back to 0.5.x. See [Encryption at Rest](../../getting-started/security.md#encryption-at-rest).

### Going Back to 0.5.x

The kept copy holds the database as it was before the migration: what was written with 0.6 since then is not in it, only in the 0.6 file. To return to 0.5.x:

1. Close the database in every 0.6 process, read-only opens included.
2. Move the 0.6 file `my_graph.grafeo` aside rather than deleting it, for example to `my_graph-0.6.grafeo`, so the writes made since the migration are not lost. If its WAL `my_graph.grafeo.wal/` exists, move it along as `my_graph-0.6.grafeo.wal/`: 0.5.x would otherwise replay the 0.6 WAL.
3. Rename the kept files back: `my_graph.grafeo.pre-0.6` to `my_graph.grafeo` and, if they exist, `my_graph.grafeo.pre-0.6.wal/` to `my_graph.grafeo.wal/` and `my_graph.grafeo.pre-0.6.checkpoint` to `my_graph.grafeo.checkpoint`.

Once you no longer need to go back to 0.5.x, you can delete the kept files.
