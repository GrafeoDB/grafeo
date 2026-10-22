---
title: Storage Backends
description: Running a database's queries and transactions on a store of your own, through the store contract.
tags:
  - extending
  - storage
---

# Storage Backends

A database keeps its graph in the built-in row-group store. It can run on a store of your own instead: the
queries, the transactions and the write-ahead log are the database's, and the rows are the store's. The store
implements the **store contract**, two traits in `grafeo::store`:

- **`GraphStore`**, the read side: every read names the view it reads at (a transaction at its snapshot, or a
  committed reader at an epoch) and returns a `Result`.
- **`GraphBackend`**, the write side: the store runs the database's transactions in it.

```rust
use std::sync::Arc;
use grafeo::{Config, GrafeoDB};
use grafeo::store::GraphBackend;

let backend: Arc<dyn GraphBackend> = Arc::new(MyStore::open("trips.rows")?);
let db = GrafeoDB::with_store(backend, Config::persistent("trips"))?;
db.execute("INSERT (:Person {name: 'Alix'})")?;
```

`GrafeoDB::with_read_store(Arc<dyn GraphStore>, config)` builds a read-only database on a store that implements
the read side only.

!!! note "Rust-only and experimental"
    A store of your own is a Rust extension point; the bindings do not expose it. Calling the traits is stable,
    but implementing them, `with_store` and `with_read_store` are
    [experimental](../../stability.md#experimental-apis) through 0.6.x: a release may add a required method.
    Pin the exact version when you implement them.

## The Transaction

The engine owns the transaction: it checks constraints, reserves the ids, and records each write with what it
replaced, in one list that feeds the write-ahead log, change data capture and the backend. The backend applies
and keeps the writes:

| Method | What the backend does |
| --- | --- |
| `reserve_node_ids`, `reserve_edge_ids` | Hands out ids that are never given out again, also after a rollback |
| `begin(transaction, snapshot)` | Opens a transaction before its first write and returns its handle |
| `apply(op, writer)` | Applies one write: in a transaction (seen only by it until it commits), as a replayed one, or as one that commits at once |
| `commit(handle, changes, epoch)` | Makes the transaction's writes visible to readers at `epoch` and later |
| `rollback(handle, changes)` | Drops the transaction's writes |
| `rollback_to(handle, savepoint, changes)` | Drops the writes after a savepoint |
| `collect(horizon)` | Drops the versions no reader at `horizon` or later needs |
| `durable_epoch()`, `flush(epoch)` | Say what the backend keeps across a restart (see below) |

`commit` and the rollbacks receive the recorded changes, read-only, so a backend needs no undo log of its own.

The engine also keeps every edge's ends alive: an edge's create claims its two ends and a node's delete claims the
node, so of an edge's create and a concurrent delete of one of its ends only the first writer commits (the other
gets a retryable conflict, also at commit), and a delete of a node with edges is refused unless it detaches them.
So no committed edge ever ends at a deleted node, and a backend's reads need not check an edge's other end.

## Durability

A database on a store of your own with a path (`Config::persistent(path)`) logs every commit to its own
write-ahead log, `<path>.wal`, which one process holds at a time. An open after a crash replays it: the catalog,
and the commits after the epoch the backend reports durable.

- A backend that keeps nothing across a restart returns `EpochId::INITIAL` from `durable_epoch()`: every open
  replays the whole log, which is never truncated.
- A durable backend makes every commit up to `epoch` survive a restart in `flush(epoch)`, and from then on reports
  that epoch from `durable_epoch()`. A checkpoint (`wal_checkpoint()`, `close()`) calls `flush`, then starts the
  log again with the catalog, dropping the commits the backend holds.

Without a path the database is in memory, and the backend's own durability is its own.

## What Works on a Store of Your Own

| Works | Is the unsupported error (`GRAFEO-Q004`) |
| --- | --- |
| Queries in every language, transactions, savepoints, failed statements undone | Named graphs and schemas with a graph of their own (`CREATE GRAPH`, `set_current_graph` to another graph) |
| The direct API: `create_node`, `set_node_property`, `get_node`, `find_nodes_by_property` and the rest | Property, vector and text indexes, and every vector, text and hybrid search |
| Node and edge types (`CREATE NODE TYPE`), constraints | Property history: `get_node_property_history`, `get_node_property_at_epoch` |
| Change data capture, metrics, the session settings | The database file and its copies: `save`, `to_memory`, `export_snapshot`, `restore_snapshot`, `iter_nodes`, `iter_edges`, `store()` |
| The write-ahead log and crash recovery (with a path) | Bulk imports (`import_tsv`, `import_mmio`) and backups (`backup_full`, `backup_incremental`) |

An unsupported call changes nothing. `memory_usage()` reports what the engine holds; the backend's rows are its
own.

## Checking a Backend

The `conformance` feature publishes the suite the built-in store runs. It writes a store through `GraphBackend`
as the engine does and checks what every read path sees, at every view: a transaction's own writes, what other
readers do not see until the commit, what an earlier reader keeps, rollbacks and savepoints, replay, ids,
immediate writes and the counts. `conformance_suite!` gives one test per case:

```toml
[dev-dependencies]
grafeo = { version = "0.6", features = ["conformance"] }
```

```rust
use grafeo::store::{KnownGap, conformance_suite};

/// The cases this store is known to fail, each with its reason.
const GAPS: &[KnownGap] = &[];

fn store() -> MyStore {
    MyStore::new_in_memory()
}

conformance_suite!(MyStore, store, GAPS);
```

A case listed as a known gap must keep failing: once it passes, its test fails too and asks you to drop the gap,
so the list always says where the store stands.

## An Example

The tests of the `grafeo` crate hold a complete backend, `crates/grafeo/tests/example_backend.rs`. It keeps its
rows in a `RowGroupStore` and writes them to a file of its own at each flush, so it is durable; an in-memory
variant reports `EpochId::INITIAL`. Both run the conformance suite, a crash and the recovery from the log, a
rollback and a reopen.
