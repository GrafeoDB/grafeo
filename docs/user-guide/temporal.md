---
title: Temporal Properties & Time-Travel
description: Version history, point-in-time queries and epoch-based time-travel in Grafeo.
tags:
  - temporal
  - time-travel
  - versioning
---

# Temporal Properties & Time-Travel

!!! warning "Feature flag required for versioned properties"
    Temporal *types* (Date, Time, Duration, etc.) are always available. Temporal *properties* (versioned history, time-travel queries) require the `temporal` feature flag, which is **not** included in any named profile (`embedded`, `browser`, `server`, `full`). Enable it explicitly:

    ```toml
    [dependencies]
    grafeo = { version = "0.5", features = ["temporal"] }
    ```

    Or combine it with a profile: `features = ["embedded", "temporal"]`.

Grafeo supports two complementary temporal features:

1. **Temporal types**: Date, Time, Duration and zoned variants for storing temporal data. These are always available with no feature flags required.
2. **Temporal properties**: opt-in versioned property and label history with time-travel queries. This requires the `temporal` feature flag (available since 0.5.24).

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

## Temporal Properties

*Requires the `temporal` feature flag. Available since 0.5.24.*

When enabled, Grafeo records an append-only version history for every property and label change. Each mutation creates a new version tagged with the current epoch, so you can later reconstruct the exact state of any node or edge at any point in time.

### How It Works

1. Every committed transaction advances the database epoch
2. When a property is set or a label is added/removed, the change is recorded in an append-only `VersionLog`
3. The version log stores the epoch and the new value for each change
4. Rollbacks undo version creation, so aborted transactions leave no trace

### Example: Tracking Property Changes

```python
from grafeo import GrafeoDB

db = GrafeoDB()

# Create a person (epoch advances on commit)
with db.begin_transaction() as tx:
    tx.execute("INSERT (:Person {name: 'Alix', city: 'Amsterdam'})")
    tx.commit()

epoch_v1 = db.current_epoch()

# Update city
with db.begin_transaction() as tx:
    tx.execute("MATCH (p:Person {name: 'Alix'}) SET p.city = 'Berlin'")
    tx.commit()

epoch_v2 = db.current_epoch()

# Update again
with db.begin_transaction() as tx:
    tx.execute("MATCH (p:Person {name: 'Alix'}) SET p.city = 'Paris'")
    tx.commit()

# Current state: Paris
result = db.execute("MATCH (p:Person {name: 'Alix'}) RETURN p.city")
# => 'Paris'

# Historical state at epoch_v1: Amsterdam
result = db.execute_at_epoch(
    "MATCH (p:Person {name: 'Alix'}) RETURN p.city",
    epoch=epoch_v1,
)
# => 'Amsterdam'
```

### Snapshot Persistence

Temporal version history survives snapshot save/load cycles. Snapshot format v4 (0.5.24+) serializes the full `VersionLog` for each property and label, so restoring a snapshot preserves the complete history.

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

Retrieve a single node or edge as it existed at a specific epoch. Returns `None` if the entity did not exist at that epoch.

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
    if let Some(node) = db.get_node_at_epoch(node_id, EpochId::new(3)) {
        println!("Labels: {:?}", node.labels);
        println!("Properties: {:?}", node.properties);
    }

    if let Some(edge) = db.get_edge_at_epoch(edge_id, EpochId::new(3)) {
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

Temporal versioning adds overhead to write operations because each mutation appends to the `VersionLog`. Read performance is optimized with several strategies:

| Scenario | Cost | Details |
|----------|------|---------|
| Current-epoch reads | O(1) | Fast path: the latest entry is checked first, skipping the binary search |
| Historical reads | O(log N) | Binary search over the version log for the target epoch |
| Write overhead | Append-only | Each property/label change appends one entry to the version log |

### Overhead Benchmarks

| Version | Read overhead (vs. non-temporal) |
|---------|----------------------------------|
| 0.5.24 | ~16% |
| 0.5.25 | ~6% (optimized `VersionLog::at()` fast path, eliminated double HashMap lookup) |

### Tips

- **Enable only when needed**: the `temporal` feature is opt-in. If you do not need property versioning, leave it disabled for zero overhead.
- **Epoch advances on commit**: epochs only increment when explicit transactions commit. Auto-commit queries each produce one epoch.
- **Snapshots preserve history**: saving and loading snapshots retains the full version log, so you can time-travel across restarts.
