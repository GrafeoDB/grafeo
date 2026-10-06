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

A persistent database is a single file at the path you give, whatever its extension (`.grafeo`, `.db` or none). While it is open, the changes since its last checkpoint are kept in a write-ahead log next to it:

```text
my_graph.db         # The database: its state as of the last checkpoint
my_graph.db.wal/    # Write-ahead log, while the database is open
```

A path with a trailing separator (`./data/`) names the same database as `./data`, and `path()` (`db.path` in Python) returns the path as you gave it.

The database also writes next to its path: its WAL (`<path>.wal/`), spill files under memory pressure (`<path>.spill/`, while open for writing; a read-only open spills to the system temp directory) and, while the database is created, short-lived files (`<path>.creating`, `<path>.migrate.lock`). The directory that holds the path must therefore be writable, not only the database file.

By default, Grafeo 0.5.x stored a database at a path without the `.grafeo` extension as a WAL directory (a directory holding `wal/`). 0.6 no longer creates WAL directories: it migrates an existing one to a single file at the same path the first time it opens it for writing (see [Upgrading from 0.5](#upgrading-from-05)). A directory that is not a 0.5.x database, such as an empty directory created beforehand, is refused as not a database and left unchanged: give the database a path where nothing exists yet, for example a file inside that directory.

## Durability Guarantees

- **Write-Ahead Logging (WAL)**: a transaction's changes are written to the WAL when it commits
- **Checkpointing**: the database file is brought up to date periodically and on `close()`, after which the WAL keeps only newer changes
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

The entire database is stored in one file in the `.grafeo` format, with a sidecar WAL directory for crash safety. Grafeo has had single-file databases since 0.5.21; 0.6.0 writes a new version of the format and uses it for every database, whatever the extension of its path. `.grafeo` is the usual one:

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
- Exclusive file locking prevents multiple processes from opening the same file simultaneously

### Storage Format Setting (Rust)

`Config::storage_format` decides only what a new path becomes. `StorageFormat::Auto` (the default) creates a single file there, and an existing path opens as what it holds, whatever the setting. `StorageFormat::SingleFile` and `StorageFormat::WalDirectory` are deprecated and removed in 0.7.0: `SingleFile` does the same as `Auto`, and `WalDirectory` opens an existing 0.5.x WAL directory by migrating it, as the other settings do, and fails at a new path with "WAL directories are no longer created". Use `StorageFormat::Auto` (or leave the setting out) instead.

## Read-Only Mode

Open a database in read-only mode to allow multiple processes to read the same database file concurrently. Mutations are rejected at the session level.

=== "Python"

    ```python
    db = grafeo.GrafeoDB.open_read_only("my_graph.grafeo")
    ```

=== "Rust"

    ```rust
    let db = GrafeoDB::open_read_only("my_graph.grafeo")?;
    ```

Read-only mode uses a shared file lock instead of an exclusive lock, so multiple readers can coexist. It sees every commit, also those a writer that exited without `close()` left in the WAL (`<path>.wal/`), which it reads without changing anything. A database written by 0.5.x (a file or a WAL directory) is read into memory once instead, and is not migrated (see [Upgrading from 0.5](#upgrading-from-05)).

## One Writer at a Time

A persistent database can be open for writing by one `GrafeoDB` instance at a time, in one process. Opening it again, from the same process or another one, fails with `database file is locked by another process` until the first instance calls `close()` or is dropped.

To share a database between processes, run it behind [Grafeo Server](https://github.com/GrafeoDB/grafeo-server), or open it in [read-only mode](#read-only-mode) from the readers.

## After `close()`

Once `close()` of a persistent database starts, the handle takes no more writes: commits and statements that write (in every query language, SPARQL updates included), schema statements and graph commands, the direct calls that write (nodes, edges, properties and labels, named graphs, property, vector and text indexes, imports, `batch_insert_rdf` and `restore_snapshot`), and the calls that persist (`wal_checkpoint()`, `save()`, the backups, `compact()`) fail with the database-closed error (`GRAFEO-T007`, Python `DatabaseClosedError`). A write already in progress, `restore_snapshot` included, completes first and is saved; so does an import or `batch_insert_rdf` that is already writing, while one still reading its input is refused. Reads still work. Open the database again to write. An in-memory database has nothing to persist and keeps working.

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

`save()` writes a copy of the database to a new single file, whatever the extension of the path, and fails if the path already exists. The copy is a database of its own: open it like any other.

## Upgrading from 0.5

Grafeo 0.6.0 writes databases in a new file format, and every database is a single file. A database written by 0.5.x is migrated the first time 0.6 opens it for writing: with `GrafeoDB(path=...)` in Python, `GrafeoDB::open` in Rust, a read-write open in another binding, or a command of the `grafeo` command line tool. Both kinds of 0.5.x database are migrated: a single file (usually `.grafeo`, at any extension), and a WAL directory (a directory holding `wal/`, which 0.5.x created by default for a path without the `.grafeo` extension).

The migration reads the old database, including the changes in its WAL, and writes it to a new file (named `my_graph.grafeo.migrating` while it is written, so the migration needs free disk space for a copy of the database). It then renames the old files and gives the new file the database's name. The old files are kept, byte for byte:

| Written by 0.5.x | Kept as |
|------------------|---------|
| `my_graph.grafeo` | `my_graph.grafeo.pre-0.6` |
| `my_graph.grafeo.wal/` | `my_graph.grafeo.pre-0.6.wal/` |
| `my_graph.grafeo.checkpoint` (a checkpoint 0.5.44 left pending) | `my_graph.grafeo.pre-0.6.checkpoint` |
| `my_graph.grafeo.spill/` (see [Spilled Embeddings](#spilled-embeddings)) | `my_graph.grafeo.pre-0.6.spill/` |
| `my_graph.db/` (a WAL directory) | `my_graph.db.pre-0.6/`, the whole directory, and `my_graph.db.spill/` as `my_graph.db.pre-0.6.spill/` |

A WAL directory becomes a file at the same path: after the migration, `my_graph.db` is the 0.6 database file, and the path you used before opens it, also when it ends with a separator (`my_graph.db/`). The whole directory moves to `my_graph.db.pre-0.6/`, also files in it that are not part of the database: 0.5.x created `wal/` inside any directory it was given, so an application may keep other files there. Move them out before the first 0.6 open, or take them from the kept directory afterwards.

A migration never replaces a kept copy: while one of these names is taken, the open fails before it writes anything. A kept copy is never migrated itself: a read-write open of `my_graph.grafeo.pre-0.6` fails; open it read-only, or rename it back first (see [Going Back to 0.5.x](#going-back-to-05x)). If a migration fails or is cut off by a crash, the next read-write open finishes it or starts it again. The old files are never changed, with one exception: a crash while migrating a WAL directory written by 0.5.43 (which has no `LOCK` file) can leave an empty `LOCK` file in the kept directory, as the migration creates one to lock the directory. A read-write open in another process waits up to five seconds for a running migration, then fails with "database locked: a migration is running".

0.7.0 will no longer read 0.5.x databases: open each 0.5.x database once with 0.6, for writing, before you upgrade to 0.7.

### Before the First Open

Stop every 0.5.x process that uses the database. 0.5.x cannot open the migrated file. While a 0.5.x process has a database file open for writing, or a 0.5.44 process has a WAL directory open, the migration fails with a "locked by another process" error and changes nothing. 0.5.43 and older take no lock on a WAL directory, so the migration cannot tell that they use it: make sure they are stopped.

### Builds Without the `wal` Feature

A build without the `wal` feature cannot replay a WAL, so it refuses to open (or migrate) a 0.5.x WAL directory, whose WAL holds all of its data, and a 0.5.x file whose sidecar WAL holds files, and changes nothing. The `grafeo` Rust crate's default profile (`embedded`) is such a build. Open these databases read-write once with a build that has `wal`: the `grafeo` crate with the `lpg` or `storage` feature, the Python, Node.js or C bindings, or the `grafeo` command line tool. A 0.5.x file that was closed cleanly (its sidecar WAL is gone or empty) migrates in every build. For the same reason, a read-only and a read-write open in such a build refuse a 0.6 file whose WAL holds commits a writer left when it exited without `close()`, and change nothing: open it with a build that has `wal`, which replays them.

### Read-Only Opens

A read-only open (`GrafeoDB.open_read_only()` in Python, `GrafeoDB::open_read_only` or `Config::read_only` in Rust) and `open_in_memory()` read a 0.5.x database, a file with its WAL or a WAL directory, without migrating or changing it. A read-only open loads such a database into memory once and then holds no lock on it. If a migration was cut off after the old files were renamed, read-only opens and `open_in_memory()` fail until a read-write open has finished it.

### Encrypted Databases

A read-write open with a key (`Config::encryption`) migrates a 0.5.x database, a WAL directory included, into an encrypted file. The kept files are not encrypted, as 0.5.x never encrypted its files: `my_graph.grafeo.pre-0.6`, `my_graph.grafeo.pre-0.6.wal/` and, if present, `my_graph.grafeo.pre-0.6.checkpoint` and `my_graph.grafeo.pre-0.6.spill/`, or the directory `my_graph.db.pre-0.6/` (and `my_graph.db.pre-0.6.spill/`). Remove all of them once you no longer need to go back to 0.5.x. A database that spilled embeddings to a configured spill path needs those files moved into `my_graph.grafeo.spill/` before the first open with the key (see [Spilled Embeddings](#spilled-embeddings)). See [Encryption at Rest](../../getting-started/security.md#encryption-at-rest).

### Spilled Embeddings

Before 0.6, spilling a vector index (under memory pressure, or with `TierOverride::ForceDisk`) moved its embeddings out of the database into files named `vectors_<label>%3A<property>.bin` in the spill directory (`<path>.spill/`, or the configured spill path), so a database closed while spilled held them only there. An open in 0.6 reads the files in `<path>.spill/` back into the database: a read-write open writes them into the database file and then deletes them, and a read-only open keeps them in memory and changes nothing. The migration of a 0.5.x database reads them into the new file and keeps `<path>.spill/` with the other old files, as `<path>.pre-0.6.spill/`. Files in a configured spill path are read only with a 0.5.x database (see below). Going back to 0.5.x keeps them in both places.

Only the files of the database's own vector indexes are read, and only for embeddings the database does not hold, so an embedding changed while spilled keeps its newer value. Each file is read in batches of at most 8 MiB, so the open needs memory for the embeddings it reads back plus 8 MiB, not for the whole file. The vector index of each file found (read or not) is rebuilt from the database's embeddings, as the index saved while its embeddings were spilled could not see an embedding set then; a read-write open writes the rebuilt index into the database file. The old files record no removals and no history, which leaves two limits:

- An embedding removed while spilled comes back, as it did when 0.5.x reloaded the spill files: remove it again.
- Each file is read once (`<path>.spill/` at the first open, a configured spill path by the migration), and only into nodes of the vector index's label. A node of that label that 0.5.x created after another one was deleted while spilled can have taken the deleted node's id, and with it its embedding: check the embeddings of such nodes created since the database last spilled.

Some files stay where they are, and the log names each one:

- a file of a vector index the database does not have (a crash can lose an index created since the last checkpoint): create the index again and reopen the database, which then reads the file;
- a file whose embeddings have another number of dimensions than the index (a damaged file, or another database's in a shared spill path): its embeddings are not read;
- a file that cannot be read: its embeddings are not read, or, when the read stops partway, those read before stay in the database (the log says how many);
- every file in a configured spill path (see below).

Files in a configured spill path are read only when a 0.5.x database is read: by its migration, which writes their embeddings into the new 0.6 file, and by a read-only open of a 0.5.x database or of its kept `.pre-0.6` copy. They are never deleted, as other databases may share the path and the kept copy needs them to go back to 0.5.x; each such read logs the files it read and that they stay. A 0.6 database never reads them again, so an embedding you remove after the upgrade stays removed, and no node created later gets an old embedding. Remove the files once every database that shares the path has been upgraded and you no longer need 0.5.x.

An encrypted database has no spill path (`Config::validate` refuses one with a key), so a migration with a key does not read old files in a configured spill path, and the encrypted file lacks their embeddings. Before the first open with the key, move the files of the database's own vector indexes from the configured spill path into `<path>.spill/`, which the migration reads.

A database written by a 0.6 development build (before the 0.6.0 release) that spilled to a configured spill path does not get those embeddings back: only its own `<path>.spill/` is read.

0.5.x databases that shared one spill path wrote over each other's spill files (one file per label and property, from whichever database spilled last). An open reads the file of an index it has, whichever database wrote it, so check the embeddings of such databases after the upgrade.

### Special Cases

- **A directory whose parent is not writable**: the migration writes next to the database path (`<path>.migrate.lock`, the image `<path>.migrating`, the kept `<path>.pre-0.6/`), and the 0.6 database keeps its WAL (`<path>.wal/`) and spill files (`<path>.spill/`) there too. The process therefore needs write access to the directory that holds the database path, not only to the database directory, which a 0.5.x WAL directory did not need. Without it the open fails and changes nothing: grant it, or move the database into a directory of its own, as for a mount point.
- **A symbolic link or junction**: when the database path is a link, the migrated file is created where the link is, and the link itself is renamed to `<path>.pre-0.6`. The 0.5.x data stays where the link points, and the new file (with its image while it is written, and later its WAL) is on the link's file system, which needs the free space. To keep the database on the link's target, open the target path instead.
- **A mount point**: a database directory that is a mount point (a container volume, for example) cannot be renamed, so it cannot be migrated in place. For a database at the mount point `/data`, copy `/data/wal/` to `/data/graph/wal/` and open `/data/graph` from then on. Or open `/data` read-only and `save()` it to a new file inside the volume, such as `/data/graph.grafeo`, and open that file from then on.
- **Open files on Windows**: on Windows a directory cannot be renamed while any process has a file inside it open. The migration of a WAL directory then fails, and changes nothing, until they are closed. A 0.5.x process that still has a database open with spilled embeddings keeps a file in its spill directory open: the migration then stops after moving the database file, the error names the spill directory, and every open fails until that process is stopped; the next read-write open finishes the migration.
- **The current directory**: a read-write open of a 0.5.x WAL directory from a process whose current directory is inside it (with `.` or by its name) is refused, as a process's own working directory cannot be moved (a read-only open works). Open it read-write from a process whose current directory is outside it.

### Going Back to 0.5.x

The kept copy holds the database as it was before the migration: what was written with 0.6 since then is not in it, only in the 0.6 file. To return to 0.5.x:

1. Close the database in every 0.6 process, read-only opens included.
2. Move the 0.6 file `my_graph.grafeo` aside rather than deleting it, for example to `my_graph-0.6.grafeo`, so the writes made since the migration are not lost. If its WAL `my_graph.grafeo.wal/` exists, move it along as `my_graph-0.6.grafeo.wal/`: 0.5.x would otherwise replay the 0.6 WAL.
3. Remove `my_graph.grafeo.spill/` if 0.6 created one: it is a cache, and 0.5.x never reads it. Then rename the kept files back: `my_graph.grafeo.pre-0.6` to `my_graph.grafeo` and, if they exist, `my_graph.grafeo.pre-0.6.wal/` to `my_graph.grafeo.wal/`, `my_graph.grafeo.pre-0.6.checkpoint` to `my_graph.grafeo.checkpoint` and `my_graph.grafeo.pre-0.6.spill/` to `my_graph.grafeo.spill/`. Old spill files in a configured spill path stayed where they were.

For a WAL directory the steps are the same: move the 0.6 file `my_graph.db` and its WAL `my_graph.db.wal/` aside, remove `my_graph.db.spill/` if 0.6 created one, then rename the kept directory `my_graph.db.pre-0.6/` back to `my_graph.db/` and, if it exists, `my_graph.db.pre-0.6.spill/` to `my_graph.db.spill/`.

Once you no longer need to go back to 0.5.x, you can delete the kept files.
