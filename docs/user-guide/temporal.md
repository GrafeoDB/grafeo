---
title: Temporal Properties & Time-Travel
description: Version history, point-in-time queries and epoch-based time-travel in Grafeo.
tags:
  - temporal
  - time-travel
  - versioning
---

# Temporal Properties & Time-Travel

Grafeo has two complementary temporal features:

1. **Temporal types**: Date, Time, Duration and zoned variants for storing temporal data. They are always available.
2. **History and time travel**: a database can keep the committed states of a retention window, and any query, view
   or direct read can then read the database as it was at an earlier epoch or time. History is off by default and
   costs nothing while it is off.

!!! note "The `temporal` feature flag"
    Since 0.6.0 the `temporal` Cargo feature is deprecated and enables nothing: history is a per-database setting
    (see [History retention](#history-retention)), and the history APIs are in every build. It is removed in 0.7.0.

## Temporal Types

Temporal types are first-class values that can be stored as properties, used in expressions and returned from queries. They are available in all query languages and all bindings.

### Supported Types

| Type | Example | Description |
|------|---------|-------------|
| `DATE` | `2024-01-15` | Calendar date (year, month, day) |
| `TIME` | `14:30:00` | Local time (hour, minute, second) |
| `DATETIME` | `2024-01-15T14:30:00` | Local date and time |
| `DURATION` | `P1Y2M3D` | ISO 8601 duration |
| `ZONED DATETIME` | `2024-01-15T14:30:00+01:00` | Datetime with UTC offset |
| `ZONED TIME` | `14:30:00+01:00` | Time with UTC offset |

### Typed Literals

Inline temporal values use the typed literal syntax:

```sql
RETURN DATE '2024-01-15'
RETURN TIME '14:30:00'
RETURN DATETIME '2024-01-15T14:30:00Z'
RETURN DURATION 'P1Y2M3D'
RETURN ZONED DATETIME '2024-01-15T14:30:00+01:00'
RETURN ZONED TIME '14:30:00+01:00'
```

### String Constructors

Parse temporal values from ISO 8601 strings:

```sql
RETURN date('2024-01-15')
RETURN time('14:30:00')
RETURN datetime('2024-01-15T14:30:00')
RETURN duration('P1Y2M')
```

### Map Constructors

Build temporal values from named components:

```sql
RETURN date({year: 2024, month: 3})
RETURN time({hour: 14, minute: 30, second: 45})
RETURN datetime({year: 2024, month: 3, day: 15, hour: 14, minute: 30})
RETURN duration({years: 1, months: 2, days: 3})
```

Omitted components default to their zero value (1 for month/day, 0 for time components).

### Arithmetic

Durations can be added to or subtracted from dates and datetimes:

```sql
-- 30 days after a date
RETURN DATE '2024-01-15' + DURATION 'P30D'
-- Result: 2024-02-14

-- Find events in the last 7 days
MATCH (e:Event)
WHERE e.created_at > now() - DURATION 'P7D'
RETURN e.title, e.created_at
```

### Component Extraction

Extract individual components from temporal values:

```sql
WITH DATE '2024-06-15' AS d
RETURN year(d), month(d), day(d)
-- 2024, 6, 15

WITH TIME '14:30:45' AS t
RETURN hour(t), minute(t), second(t)
-- 14, 30, 45
```

For the full list of temporal functions, see the [Temporal Functions](gql/functions-temporal.md) reference.

---

## History Retention

A database keeps the versions that open transactions still read, and with a retention window also the versions of
every commit within that window. Retention is a setting of the database, stored in its file, so a reopen keeps it:

| Setting | What the database keeps |
|---------|-------------------------|
| `HistoryRetention::Off` (default) | only what open transactions read; no extra cost |
| `HistoryRetention::Duration(d)` | every committed state of the last `d` |
| `HistoryRetention::Forever` | every committed state |

```rust
use std::time::Duration;
use grafeo::{Config, GrafeoDB, HistoryRetention};

let db = GrafeoDB::with_config(
    Config::persistent("travels.grafeo")
        .with_history_retention(HistoryRetention::Duration(Duration::from_secs(7 * 24 * 3600))),
)?;
```

### How It Works

1. Every commit advances the database epoch and records its time.
2. Garbage collection keeps every version a reader at an epoch inside the window still needs: the horizon is the
   older of the oldest open snapshot and the start of the window.
3. A checkpoint writes the kept versions with the data, and a crash or a reopen keeps them: the history of a window
   reads the same after a restart.
4. A read before what the database keeps fails with "history before epoch ... is not retained": it never answers
   with versions garbage collection may already have dropped.

Changing the retention takes effect from then on: what was collected before stays gone.

---

## Time-Travel Queries

*Available since 0.5.13.*

Time-travel lets you run any query against a historical database snapshot. The results reflect the state of nodes, edges, properties and labels as of the specified epoch.

### `execute_at_epoch`

The simplest way to query a past state: pass an epoch number alongside the query.

=== "Python"

    ```python
    result = db.execute_at_epoch(
        "MATCH (p:Person) RETURN p.name, p.city",
        epoch=5,
    )
    for row in result:
        print(row['p.name'], row['p.city'])
    ```

=== "Rust"

    ```rust
    let result = session.execute_at_epoch(
        "MATCH (p:Person) RETURN p.name, p.city",
        EpochId::new(5),
    )?;
    for row in result.rows() {
        println!("{:?}", row);
    }
    ```

### `as_of`: a view of the database at a point in time

`GrafeoDB::as_of` gives a read-only view of the database at a point in time, which you query like the database
itself. The view holds its epoch while it lives, so its reads stay the same. A point in time is an epoch, a time, a
GQL `ZONED DATETIME` value, or a duration ago. The names follow SQL:2011's `FOR SYSTEM_TIME AS OF`.

```rust
use std::time::Duration;
use grafeo::TimePoint;

// The database as it was five minutes ago.
let view = db.as_of(TimePoint::ago(Duration::from_secs(300)))?;
let people = view.execute("MATCH (p:Person) RETURN p.name")?;

// One query at an epoch or a time, without keeping a view.
let then = db.execute_as_of("MATCH (p:Person) RETURN count(p)", then_epoch)?;

// The epoch a point in time reads at.
let epoch = db.epoch_as_of(TimePoint::ago(Duration::from_secs(300)))?;
```

A write through a view fails: a view of the past is read-only.

### Session-Level Viewing Epoch

For multiple queries against the same historical snapshot, set a viewing epoch on the session. All subsequent queries will see the database as of that epoch until the override is cleared.

=== "Python"

    ```python
    # Not yet exposed in the Python binding; use execute_at_epoch instead
    ```

=== "Rust"

    ```rust
    session.set_viewing_epoch(EpochId::new(5));

    // Both queries see epoch 5
    let r1 = session.execute("MATCH (p:Person) RETURN p.name")?;
    let r2 = session.execute("MATCH (p:Person) RETURN p.city")?;

    session.clear_viewing_epoch();
    ```

=== "GQL"

    ```sql
    SESSION SET PARAMETER viewing_epoch = 5

    -- All subsequent queries in this session see epoch 5
    MATCH (p:Person)
    RETURN p.name, p.city
    ```

---

## Version History APIs

These APIs return the full version history of a node or edge, including creation and deletion epochs.

### `get_node_at_epoch` / `get_edge_at_epoch`

Retrieve a single node or edge as it existed at a specific epoch. It is `None` if the entity did not exist at that epoch; an epoch older than the history the database keeps is an error ("history before epoch ... is not retained"), never an answer.

=== "Python"

    ```python
    # Get node state at epoch 3
    node = db.get_node_at_epoch(node_id, epoch=3)
    if node is not None:
        print(node.labels, node.properties)

    # Get edge state at epoch 3
    edge = db.get_edge_at_epoch(edge_id, epoch=3)
    if edge is not None:
        print(edge.edge_type, edge.properties)
    ```

=== "Rust"

    ```rust
    if let Some(node) = db.get_node_at_epoch(node_id, EpochId::new(3))? {
        println!("Labels: {:?}", node.labels);
        println!("Properties: {:?}", node.properties);
    }

    if let Some(edge) = db.get_edge_at_epoch(edge_id, EpochId::new(3))? {
        println!("Type: {:?}", edge.edge_type);
        println!("Properties: {:?}", edge.properties);
    }
    ```

### `get_node_history` / `get_edge_history`

Return every version of a node or edge as a list of `(created_epoch, deleted_epoch, entity)` tuples. The `deleted_epoch` is `None` if the entity still exists.

=== "Python"

    ```python
    history = db.get_node_history(node_id)
    for created, deleted, node in history:
        status = f"deleted at {deleted}" if deleted else "active"
        print(f"Epoch {created} ({status}): {node.properties}")
    ```

=== "Rust"

    ```rust
    let history = db.get_node_history(node_id);
    for (created, deleted, node) in &history {
        let status = match deleted {
            Some(d) => format!("deleted at {d}"),
            None => "active".to_string(),
        };
        println!("Epoch {created} ({status}): {:?}", node.properties);
    }
    ```

### Practical Example: Audit Trail

Combine temporal properties with history APIs to build an audit trail:

```python
from grafeo import GrafeoDB

db = GrafeoDB()

# Create and update a person over several transactions
with db.begin_transaction() as tx:
    tx.execute("INSERT (:Person {name: 'Gus', role: 'engineer'})")
    tx.commit()

with db.begin_transaction() as tx:
    tx.execute("MATCH (p:Person {name: 'Gus'}) SET p.role = 'senior engineer'")
    tx.commit()

with db.begin_transaction() as tx:
    tx.execute("MATCH (p:Person {name: 'Gus'}) SET p.role = 'staff engineer'")
    tx.commit()

# Retrieve Gus's node ID
result = db.execute("MATCH (p:Person {name: 'Gus'}) RETURN id(p) AS nid")
gus_id = result[0]['nid']

# Walk through the full history
history = db.get_node_history(gus_id)
for created, deleted, node in history:
    print(f"Epoch {created}: role = {node.properties.get('role')}")
# Epoch 1: role = engineer
# Epoch 2: role = senior engineer
# Epoch 3: role = staff engineer
```

---

## Performance Considerations

| Retention | Cost |
|-----------|------|
| Off (default) | none: writes, reads and memory are the same as without history |
| A window or `Forever` | each kept version takes about 72 bytes in memory and about 25 bytes in a checkpoint; reads at the current epoch are unchanged |

### Tips

- **Keep the window as short as your use case allows**: memory and file size grow with the number of versions in it.
- **Epochs advance on commit**: each committed transaction or auto-commit statement is one epoch.
- **History survives restarts**: checkpoints, crashes and reopens keep the window's versions.
