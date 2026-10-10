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

The database also writes next to its path: its WAL (`<path>.wal/`), spill files under memory pressure (`<path>.spill/`, while open for writing; a read-only open spills to a directory of its own in the system temp directory). A configured spill path (`Config::spill_path` in Rust) replaces both: a read-write and a read-only open then spill there and, while the database is created, short-lived files (`<path>.creating`, `<path>.migrate.lock`). The directory that holds the path must therefore be writable, not only the database file.

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

The sync mode is set through the Rust `Config` builder; the Python constructor uses the default. Build the modes with settings with `DurabilityMode::batch(max_delay, max_records)` and `DurabilityMode::adaptive(interval)`, in whole milliseconds.

```rust
use std::time::Duration;

use grafeo::{Config, DurabilityMode, GrafeoDB};

let config = Config::persistent("my_graph.db").with_wal_durability(DurabilityMode::Sync);
let db = GrafeoDB::with_config(config)?;

let batched = Config::persistent("my_other_graph.db")
    .with_wal_durability(DurabilityMode::batch(Duration::from_millis(19), 88));
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
- A checkpoint writes the data in chunks of at most 1 MiB as it goes, without first encoding whole sections in memory. While it writes a vector or text index, changes to that index (inserting, updating or removing a node's vector or text) wait until the index is written, and so do searches of that index that arrive after a waiting change.
- A checkpoint writes the committed state, also while transactions are open: what an open transaction deleted or changed is written as it was committed, and nothing it created is written. A checkpoint taken while an open transaction has changed the default graph leaves out the vector and text indexes, and the next open builds them from the data. `save()`, `to_memory()`, `export_snapshot()` and the backups copy the committed state in the same way.
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

A transaction still open when `close()` runs is left out of the file: the final checkpoint writes the committed state (see above). Its commit then fails with the database-closed error; it can only roll back.

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

### Builds Without the `compact-store` Feature

A database on which [`compact()`](../compact-store.md) ran in 0.5.44 or older keeps its compacted base in its file, which only a build with the `compact-store` feature can read: it folds the base into the store as it opens the file. A build without it refuses such a file on a read-write open (which would migrate it), a read-only open and `open_in_memory()`, and changes nothing. The `grafeo` Rust crate (unless you add the `compact-store` feature) and the `grafeo` command line tool are such builds; the Python, Node.js and C bindings have the feature. Open these databases with a build that has it.

### Builds Without the `triple-store`, `vector-index` or `text-index` Feature

RDF triples, in the database file, in its WAL or in a 0.5.x WAL directory, can only be read by a build with the `triple-store` feature. Vector and text indexes can only be kept by a build with `vector-index` and `text-index`: a build without them cannot build such an index, and would drop its definition (label, property, dimensions, metric) from the file. A build without one of these features refuses a database that holds such data, a 0.6 or a 0.5.x one, on a read-write open (which would migrate a 0.5.x database), a read-only open and `open_in_memory()`, with an error that names the data and the feature, and changes nothing. It refuses a snapshot that holds such data the same way, where it used to leave the triples or the index definitions out: `import_snapshot()` (`importSnapshot()` in WebAssembly builds without these features) creates no database, and `restore_snapshot()` leaves the database as it was. The `grafeo` Rust crate's default profile lacks `triple-store` (add the `triple-store` or `rdf` feature), its `lpg` and `rdf` profiles lack `vector-index` and `text-index` (add the `ai` feature), and the `grafeo` command line tool lacks all three; the Python, Node.js and C bindings have them. Before 0.6.0 such a build opened these databases without the triples or the indexes, and its next checkpoint lost them for good.

### Read-Only Opens

A read-only open (`GrafeoDB.open_read_only()` in Python, `GrafeoDB::open_read_only` or `Config::read_only` in Rust) and `open_in_memory()` read a 0.5.x database, a file with its WAL or a WAL directory, without migrating or changing it. A read-only open loads such a database into memory once and then holds no lock on it. If a migration was cut off after the old files were renamed, read-only opens and `open_in_memory()` fail until a read-write open has finished it.

### Encrypted Databases

A read-write open with a key (`Config::encryption`) migrates a 0.5.x database, a WAL directory included, into an encrypted file. The kept files are not encrypted, as 0.5.x never encrypted its files: `my_graph.grafeo.pre-0.6`, `my_graph.grafeo.pre-0.6.wal/` and, if present, `my_graph.grafeo.pre-0.6.checkpoint` and `my_graph.grafeo.pre-0.6.spill/`, or the directory `my_graph.db.pre-0.6/` (and `my_graph.db.pre-0.6.spill/`). Remove all of them once you no longer need to go back to 0.5.x. A database that spilled embeddings to a configured spill path needs copies of those files in `my_graph.grafeo.spill/` before the first open with the key: copy them, do not move them (see [Spilled Embeddings](#spilled-embeddings)). See [Encryption at Rest](../../getting-started/security.md#encryption-at-rest).

### Spilled Embeddings

Before 0.6, spilling a vector index (under memory pressure, or with `TierOverride::ForceDisk`) moved its embeddings out of the database into files named `vectors_<label>%3A<property>.bin` in the spill directory (`<path>.spill/`, or the configured spill path), so a database closed while spilled held them only there. An open in 0.6 reads the files in `<path>.spill/` back into the database: a read-write open writes them into the database file and then deletes them (or moves one it could not take in completely to `<path>.spill/kept/`, see below), and a read-only open keeps them in memory and changes nothing. The migration of a 0.5.x database reads them into the new file and keeps `<path>.spill/` with the other old files, as `<path>.pre-0.6.spill/`. Files in a configured spill path are read only with a 0.5.x database (see below). Going back to 0.5.x keeps them in both places.

Only the files of the database's own vector indexes are read, and only for embeddings the database does not hold, so an embedding changed while spilled keeps its newer value. Each file is read in batches of at most 8 MiB, so the open needs memory for the embeddings it reads back plus 8 MiB, not for the whole file. The vector index of each file found (read or not) is rebuilt from the database's embeddings, as the index saved while its embeddings were spilled could not see an embedding set then; a read-write open writes the rebuilt index into the database file. A 0.5.x database gets every vector index rebuilt when it is migrated or opened read-only, whether old files are found or not: one that reloaded its spilled embeddings before it closed has no file left, and its saved index can still miss the embeddings set while they were spilled. The migration writes the rebuilt indexes into the new file. The old files record no removals and no history, which leaves two limits:

- An embedding removed while spilled comes back, as it did when 0.5.x reloaded the spill files: remove it again.
- Each file is taken in once, and only into nodes of the vector index's label: the files in `<path>.spill/` by the first read-write open, which then deletes them, those in a configured spill path by the migration. A read-only open changes nothing, so it reads the files again at every open until a read-write open takes them in. A node of that label that 0.5.x created after another one was deleted while spilled can have taken the deleted node's id, and with it its embedding: check the embeddings of such nodes created since the database last spilled. A node that lost the label while its embeddings were spilled kept its embedding only in the file: it does not get it back, and the file moves to `kept/` (or, for a 0.5.x database, stays in `<path>.pre-0.6.spill/`; see below).

Some files stay where they are, and the log names each one:

- a file of a vector index the database does not have (a crash can lose an index created since the last checkpoint): create the index again and reopen the database, which then reads the file;
- a file that an I/O error keeps from being read (for example one without read permission, or one another process holds open): its embeddings are not read, or, when the read stops partway, those read before stay in the database (the log says how many). Until the file can be read or is removed, every open rebuilds the file's vector index, and every read-write open also writes a checkpoint, so each open takes as long as building that index: make the file readable, or remove it;
- every file in a configured spill path (see below).

A read-write open moves the files it cannot take in completely, for a reason the next open would find again, to `<path>.spill/kept/`, once its checkpoint holds the rest, and the log names each one and why:

- a file whose embeddings have another number of dimensions than the index (a damaged file, or another database's in a shared spill path): none of its embeddings are read;
- a file that is not a spill file, or whose length is not what its header says (shorter, or with records past the count it gives): none of its embeddings are read;
- a file that holds embeddings of nodes without the vector index's label: the other embeddings are read. In 0.5.x a node that lost the label kept its embedding, and while it was spilled only the file had it. The log gives how many there are and the first node ids.

So the index is rebuilt and a checkpoint written once, not at every open. A file never replaces one already in `kept/`: a taken name gets a numeric suffix (`.1`, `.2`, ...). No open reads `kept/` again, and 0.5.x does not read it either when you go back (it reads only the files at the top level of the spill directory). Check the files there, for example to set an embedding on a node again, and remove them once you no longer need them. A read-only open moves nothing: it leaves these files where they are. Nor does a migration: the files of a 0.5.x database it cannot take in completely stay in `<path>.pre-0.6.spill/` (or the configured spill path) with the other old files, and the log names them, so check them there before you remove the kept 0.5.x files.

Files in a configured spill path are read only when a 0.5.x database is read: by its migration, which writes their embeddings into the new 0.6 file, and by a read-only open of a 0.5.x database or of its kept `.pre-0.6` copy. They are never deleted, as other databases may share the path and the kept copy needs them to go back to 0.5.x; each such read logs the files it read and that they stay. A 0.6 database never reads them again, so an embedding you remove after the upgrade stays removed, and no node created later gets an old embedding. Remove the files once every database that shares the path has been upgraded and you no longer need 0.5.x.

An encrypted database has no spill path (`Config::validate` refuses one with a key), so a migration with a key does not read old files in a configured spill path, and the encrypted file lacks their embeddings. Before the first open with the key, copy the files of the database's own vector indexes from the configured spill path into `<path>.spill/`, which the migration reads. Copy them, do not move them: other 0.5.x databases that share the path still need their files, and going back to 0.5.x reads them from the configured spill path, while the migration keeps the copies in `<path>.pre-0.6.spill/`.

A database written by a 0.6 development build (before the 0.6.0 release) that spilled to a configured spill path does not get those embeddings back: only its own `<path>.spill/` is read. Nor is its vector index rebuilt when it reloaded its spilled embeddings before it closed, which left no old file: the index can miss embeddings set while they were spilled. Call `rebuild_vector_index()` once for each of its vector indexes.

0.5.x databases that shared one spill path wrote over each other's spill files (one file per label and property, from whichever database spilled last). An open reads the file of an index it has, whichever database wrote it, so check the embeddings of such databases after the upgrade.

### Special Cases

- **A directory whose parent is not writable**: the migration writes next to the database path (`<path>.migrate.lock`, the image `<path>.migrating`, the kept `<path>.pre-0.6/`), and the 0.6 database keeps its WAL (`<path>.wal/`) and spill files (`<path>.spill/`) there too. The process therefore needs write access to the directory that holds the database path, not only to the database directory, which a 0.5.x WAL directory did not need. Without it the open fails and changes nothing: grant it, or move the database into a directory of its own, as for a mount point.
- **A symbolic link or junction**: when the database path is a link, the migrated file is created where the link is, and the link itself is renamed to `<path>.pre-0.6`. The 0.5.x data stays where the link points, and the new file (with its image while it is written, and later its WAL) is on the link's file system, which needs the free space. To keep the database on the link's target, open the target path instead.
- **A mount point**: a database directory that is a mount point (a container volume, for example) cannot be renamed, so it cannot be migrated in place. For a database at the mount point `/data`, copy `/data/wal/` to `/data/graph/wal/` and open `/data/graph` from then on. Or open `/data` read-only and `save()` it to a new file inside the volume, such as `/data/graph.grafeo`, and open that file from then on.
- **Open files on Windows**: on Windows a directory cannot be renamed while any process has a file inside it open. The migration of a WAL directory then fails, and changes nothing, until they are closed. A 0.5.x process that still has a database open with spilled embeddings keeps a file in its spill directory open: the migration then stops after moving the database file, the error names the spill directory, and every open fails until that process is stopped; the next read-write open finishes the migration.
- **The current directory**: a read-write open of a 0.5.x WAL directory from a process whose current directory is inside it (with `.` or by its name) is refused, as a process's own working directory cannot be moved (a read-only open works). Open it read-write from a process whose current directory is outside it.
- **A value nested deeper than 128 levels**: 0.6 stores property values (lists, maps and paths) nested at most 128 levels deep. The migration of a database that holds a deeper one fails, names the node or edge and the property, and changes nothing: change that value with 0.5.x first.
- **Properties stored as null**: 0.5.x could store a property whose value is null, and wrote `GCounter` and `OnCounter` values to its file as null. In 0.6 a property with a null value does not exist, so after the migration such a property is gone: `keys()` and `properties()` no longer list it.

### Going Back to 0.5.x

The kept copy holds the database as it was before the migration: what was written with 0.6 since then is not in it, only in the 0.6 file. To return to 0.5.x:

1. Close the database in every 0.6 process, read-only opens included.
2. Move the 0.6 file `my_graph.grafeo` aside rather than deleting it, for example to `my_graph-0.6.grafeo`, so the writes made since the migration are not lost. If its WAL `my_graph.grafeo.wal/` exists, move it along as `my_graph-0.6.grafeo.wal/`: 0.5.x would otherwise replay the 0.6 WAL.
3. Remove `my_graph.grafeo.spill/` if 0.6 created one, once you have checked the old files in its `kept/` (see [Spilled Embeddings](#spilled-embeddings)): the rest is a cache, and 0.5.x reads neither. Then rename the kept files back: `my_graph.grafeo.pre-0.6` to `my_graph.grafeo` and, if they exist, `my_graph.grafeo.pre-0.6.wal/` to `my_graph.grafeo.wal/`, `my_graph.grafeo.pre-0.6.checkpoint` to `my_graph.grafeo.checkpoint` and `my_graph.grafeo.pre-0.6.spill/` to `my_graph.grafeo.spill/`. Old spill files in a configured spill path stayed where they were.

For a WAL directory the steps are the same: move the 0.6 file `my_graph.db` and its WAL `my_graph.db.wal/` aside, remove `my_graph.db.spill/` if 0.6 created one (after checking its `kept/`), then rename the kept directory `my_graph.db.pre-0.6/` back to `my_graph.db/` and, if it exists, `my_graph.db.pre-0.6.spill/` to `my_graph.db.spill/`.

Once you no longer need to go back to 0.5.x, you can delete the kept files.
