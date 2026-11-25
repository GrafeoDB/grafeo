# Change Data Capture (CDC)

Change Data Capture tracks mutations (creates, updates, deletes) as they
happen, giving you a replayable event log per entity. Use it for undo/redo,
audit trails, syncing changes to a remote server, or replication.

## Enabling CDC

CDC is **opt-in** to avoid overhead on the mutation hot path.
There are two levels of control:

1. **Database-wide default**: set at construction time or toggled at runtime.
2. **Per-session override**: each session can opt in or out regardless of the
   database default.

Turning CDC on or off at runtime stores the setting with the database file, so
a reopen has it as it was (an in-memory database keeps it while it is open).
The call checkpoints, and fails on a read-only open or when the checkpoint
fails, leaving the setting as it was. A database on a store of its own
(`GrafeoDB::with_store`) has no file to store it in, so the call is refused
there. While the setting is stored on, the database keeps its write-ahead log
from a cursor (see below), so turning CDC on for an open without the WAL is
refused, and while it is stored on, a writable open without the WAL is refused
too (open it with the WAL, open it read-only, or turn CDC off from an open with
the WAL). Both refusals are configuration errors
(`GRAFEO-V001`). `Config::with_cdc()` (Python `GrafeoDB(cdc=True)`) turns CDC
on for that open only and stores nothing; the open logs a warning when the
file's stored setting is off.

With the setting stored on, the database keeps a cursor in its write-ahead log
directory (`cdc_cursor_default.json`), and every checkpoint, and `close()`,
keeps the log from that cursor on, as they keep it for a backup cursor: at most
`Config::with_wal_retention_limit` beyond the last checkpoint (1 GiB by
default). Past the limit a checkpoint removes the oldest part and logs a warning
once. So while the setting is stored on, `<path>.wal` stays after `close()`.
Turning the setting on while it is on leaves the cursor where it is (and writes
it if it is missing). Turning the setting off removes the cursor, and the next
checkpoint releases the log it kept. When the setting is stored off but
the cursor cannot be removed, the call fails with
[`GRAFEO-S007`](error-codes.md#storage-s): the cursor then keeps no log, and
the next open, or turning the setting off again, removes it.

### Rust

```rust
use grafeo_engine::{Config, GrafeoDB};

// Enable CDC for all sessions via config
let db = GrafeoDB::with_config(Config::in_memory().with_cdc())?;

// Or toggle at runtime (affects future sessions only), stored with the file
db.set_cdc(true)?;

// Per-session override
let tracked   = db.session_with_cdc(true);   // CDC on for this session
let untracked = db.session_with_cdc(false);  // CDC off for this session
let default   = db.session();                // follows database default
```

### Python

```python
from grafeo import GrafeoDB

# Enable at construction
db = GrafeoDB(cdc=True)

# Or toggle at runtime
db.enable_cdc()
db.disable_cdc()

# Check current state
print(db.cdc_enabled)  # True / False
```

### Node.js / TypeScript

```typescript
import { GrafeoDB } from '@grafeo-db/js';

const db = GrafeoDB.create();
db.enableCdc();
console.log(db.isCdcEnabled); // true
db.disableCdc();
```

### C

```c
#include "grafeo.h"

GrafeoDatabase* db = grafeo_open_memory();
GrafeoStatus status = grafeo_set_cdc(db, true);
bool enabled = grafeo_is_cdc_enabled(db);
```

## Querying change history

Once CDC is enabled, every mutation records a `ChangeEvent` with:

| Field           | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `entity_id`     | Node or edge ID; for an RDF triple, a 64-bit hash of its terms and its graph (see below) |
| `graph`         | The graph the entity is in: its name (`schema/name` inside a schema, and `schema/__default__` for a schema's default graph); `None` for the default graph |
| `kind`          | `Create`, `Update`, or `Delete`                                             |
| `epoch`         | Commit epoch (monotonically increasing)                                     |
| `timestamp`     | HLC timestamp (hybrid logical clock): rising across commits and across reopens, with or without a WAL left to replay |
| `before`        | Properties before the change; on a delete, the properties it had at the last commit (an empty map for none). `None` when the write's before-image is too large to log with it (past a WAL record's 1 GiB, or a list of very many tiny properties): the delete or update still succeeds, and only the properties are left out, so a delete's event keeps its labels, or its edge type and endpoints. Only when those alone pass the limits (labels of about 1 GiB together) does a delete's event have neither |
| `after`         | Properties after the change                                                 |
| `labels`        | Nodes: the labels on create and delete, the labels after a label change     |
| `before_labels` | Nodes: the labels before a label change                                     |
| `edge_type`     | Edges: the type, on create and delete                                       |
| `src_id`        | Edges: the source node, on create and delete                                |
| `dst_id`        | Edges: the destination node, on create and delete                           |
| `deleted_with`  | Edges: on a delete by `DETACH DELETE` or `detach_delete_node`, the node deleted with it |

A node deleted with its edges is one change: its commit reports a Delete event for each edge, naming the node in `deleted_with`, followed by the node's own Delete event. An edge deleted on its own has no `deleted_with`. `changes_between` returns events in commit order: by epoch, and within a commit in the order its writes were applied.

A field that does not apply is `None` (Rust, Python) or `null` (Node.js). Adding
or removing a label is an `Update` event with `labels` and `before_labels`.
Events gain fields only: a patch release can add a field (a key in Python and
Node.js), never remove or rename one.

A transaction reports its net changes:

- A node or edge created in a transaction has one `Create` event that shows it
  as the transaction left it: property and label changes made later in the same
  transaction are part of that event rather than events of their own.
- A node, edge or RDF triple created and deleted in the same transaction reports
  no events, nor do the edges a `DETACH DELETE` removes with such a node.
- A node or edge that existed before the transaction, changed and then deleted
  in it, reports its `Delete` alone, with the properties and labels it had
  before the transaction.
- An RDF triple deleted and inserted again in the same transaction reports no
  events.
- Updates to a node or edge that exists before and after the transaction are not
  folded: each write reports its own `Update`.

A batch call (`batch_create_nodes`, `batch_create_nodes_with_props`,
`batch_create_nodes_with_labels`, `batch_create_edges`) reports one `Create`
event per node or edge it creates. The bulk loads report none: the imports
(`import_tsv`, `import_tsv_str`, `import_mmio`, `import_tsv_rdf`) and
`batch_insert_rdf` write their rows to the store and the WAL as they go,
without events.

Node and edge IDs repeat across named graphs, so an ID names an entity only
together with its graph. `history` on the database reads the default graph, and
on a session the session's current graph; `changes_between` returns the events
of every graph, each with its `graph`.

An RDF triple's ID is the first 8 bytes of SHA-256 over its terms and its graph, the
same in every process and release. It is unique in practice but not guaranteed: a
consumer that must never take two triples for one keys on the terms the event carries
(`triple_subject`, `triple_predicate`, `triple_object`, `triple_graph`).

The Rust helpers `CdcLog::record_triple_insert` and `record_triple_delete` take
N-Triples terms and a graph IRI without angle brackets (`None` for the default
graph), and record the event a commit of that triple records: the same ID and the
normalised terms. Terms that are not one triple (a blank argument included), or a
build without the triple store, record nothing. Such a call logs a warning in a
build with the `tracing` feature, and is always counted in `CdcLog::dropped_lines()`.

### Per-entity history

```rust
// Full history for a node
let events = db.history(node_id)?;

// History since a specific epoch
let recent = db.history_since(node_id, EpochId::new(42))?;
```

### Range queries

```rust
// All changes across all entities in an epoch range
let changes = db.changes_between(EpochId::new(10), EpochId::new(50))?;
```

### Python

```python
history = db.node_history(node_id)
history = db.edge_history(edge_id)
history = db.node_history_since(node_id, since_epoch=42)
changes = db.changes_between(start_epoch=10, end_epoch=50)
```

### Node.js

```typescript
const history = await db.nodeHistory(nodeId);
const history = await db.edgeHistory(edgeId);
const history = await db.nodeHistorySince(nodeId, 42);
const changes = await db.changesBetween(10, 50);
```

## Transaction semantics

CDC events are **buffered** during a transaction:

- On **commit**, buffered events are flushed to the CDC log with the commit
  epoch assigned.
- On **rollback**, buffered events are discarded.
- On **rollback to savepoint**, events after the savepoint are truncated.

This means the CDC log only contains events from successfully committed
transactions.

## Performance considerations

When CDC is disabled (the default), there is **zero overhead** on the mutation
path. No HLC timestamps are generated, no events are buffered, and no store
wrapping occurs.

When enabled, each mutation incurs:

- One HLC timestamp reservation per commit (`SystemTime::now()` syscall + atomic CAS):
  a commit's events carry consecutive timestamps, in the order its writes were applied
- One event buffer push (mutex-protected `Vec`)
- Property snapshot allocations for before/after state

For bulk data loading or benchmarks, disable CDC or use `session_with_cdc(false)`
to avoid this overhead.

### Memory bound

The in-memory log counts the bytes its events hold and keeps them under a bound.
When the log passes the bound, its oldest whole commits are dropped until it is
at or below three quarters of the bound; the newest commit is always kept, even
when it alone is larger than the bound. The bound is
`Config::with_cdc_memory_limit` (Python `cdc_memory_limit=`, Node.js
`cdcMemoryLimit`, C `grafeo_options_set_cdc_memory_limit`, C#
`SetCdcMemoryLimit`, Go `Options.CdcMemoryLimit`, Dart `cdcMemoryLimit:`);
unset, it is a sixteenth of the memory limit (with no memory limit set, of the
buffer manager's default budget, 75% of the memory the process may use), and it
is never above the memory limit. Zero is refused at the open (`GRAFEO-V001`). The epoch and event bounds
of `CdcRetentionConfig` apply too; `CdcRetentionConfig::unlimited()` lifts those
two, not the byte bound, and a retention without a byte bound of its own keeps
the one `with_cdc_memory_limit` set. `memory_usage()` reports the bound in force
as `cdc.limit_bytes`.

Under memory pressure the eviction cycle may prune the log further, the newest
commit included. Pruning the in-memory log never touches the log kept on disk
for the change feed's cursor.

## Feature flag

CDC requires the `cdc` feature flag at compile time. The `lpg`, `ai` and `enterprise`
profiles include it. The `edge` (WASM) profile does not.

```toml
[dependencies]
grafeo = { version = "0.6", features = ["ai"] }  # includes cdc
```
