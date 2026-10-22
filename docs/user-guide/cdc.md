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

### Rust

```rust
use grafeo_engine::{Config, GrafeoDB};

// Enable CDC for all sessions via config
let db = GrafeoDB::with_config(Config::in_memory().with_cdc())?;

// Or toggle at runtime (affects future sessions only)
db.set_cdc_enabled(true);

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
import { GrafeoDB } from '@grafeo-db/node';

const db = GrafeoDB.create();
db.enableCdc();
console.log(db.isCdcEnabled); // true
db.disableCdc();
```

### C

```c
#include "grafeo.h"

GrafeoDatabase* db = grafeo_open_memory();
grafeo_set_cdc_enabled(db, true);
bool enabled = grafeo_is_cdc_enabled(db);
```

## Querying change history

Once CDC is enabled, every mutation records a `ChangeEvent` with:

| Field           | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `entity_id`     | Node or edge ID                                                             |
| `graph`         | The graph the entity is in: its name (`schema/name` inside a schema, and `schema/__default__` for a schema's default graph); `None` for the default graph |
| `kind`          | `Create`, `Update`, or `Delete`                                             |
| `epoch`         | Commit epoch (monotonically increasing)                                     |
| `timestamp`     | HLC timestamp (hybrid logical clock)                                        |
| `before`        | Properties before the change; on a delete, the last properties              |
| `after`         | Properties after the change                                                 |
| `labels`        | Nodes: the labels on create and delete, the labels after a label change     |
| `before_labels` | Nodes: the labels before a label change                                     |
| `edge_type`     | Edges: the type, on create and delete                                       |
| `src_id`        | Edges: the source node, on create and delete                                |
| `dst_id`        | Edges: the destination node, on create and delete                           |

A field that does not apply is `None` (Rust, Python) or `null` (Node.js). Adding
or removing a label is an `Update` event with `labels` and `before_labels`.
Events gain fields only: a patch release can add a field (a key in Python and
Node.js), never remove or rename one.

A node or edge created in a transaction has one `Create` event that shows it as
the transaction left it: property and label changes made later in the same
transaction are part of that event rather than events of their own.

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

- One HLC timestamp generation (`SystemTime::now()` syscall + atomic CAS)
- One event buffer push (mutex-protected `Vec`)
- Property snapshot allocations for before/after state

For bulk data loading or benchmarks, disable CDC or use `session_with_cdc(false)`
to avoid this overhead.

## Feature flag

CDC requires the `cdc` feature flag at compile time. The `lpg`, `ai` and `enterprise`
profiles include it. The `edge` (WASM) profile does not.

```toml
[dependencies]
grafeo = { version = "0.5", features = ["ai"] }  # includes cdc
```
