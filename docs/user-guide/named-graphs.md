---
title: Named Graphs
description: Working with multiple named graphs in a single Grafeo database.
tags:
  - graphs
  - multi-graph
---

# Named Graphs

Grafeo supports multiple named graphs within a single database instance. Each graph has its own nodes, edges, labels, indexes and MVCC versioning. This lets you partition data into logical units while sharing a single database connection and benefiting from cross-graph transactional guarantees.

Every database starts with an implicit **default graph**. All queries target the default graph until you explicitly create and switch to a named graph.

## Creating Named Graphs

Use `CREATE GRAPH` to add a new named graph. The basic form creates an open (schema-free) graph:

```sql
CREATE GRAPH friends
```

### Variants

| Syntax | Description |
|--------|-------------|
| `CREATE GRAPH g` | Create an open (schema-free) graph |
| `CREATE GRAPH g ANY` | Explicitly open: any node/edge types allowed |
| `CREATE GRAPH g OPEN` | Synonym for `ANY` |
| `CREATE GRAPH g IF NOT EXISTS` | No-op if the graph already exists |
| `CREATE GRAPH g TYPED my_type` | Bind the graph to a declared graph type |
| `CREATE GRAPH g2 LIKE g1` | Copy the schema (graph type) of `g1`, without data |
| `CREATE GRAPH g3 AS COPY OF g1` | Copy both schema and data from `g1` |

### Examples

```sql
-- Open graph (no type constraints)
CREATE GRAPH friends

-- Idempotent creation
CREATE GRAPH friends IF NOT EXISTS

-- Copy structure only
CREATE GRAPH friends_staging LIKE friends

-- Full clone with data
CREATE GRAPH friends_backup AS COPY OF friends

-- Typed graph (requires a previously created graph type)
CREATE GRAPH TYPE SocialNetwork (
    NODE TYPE Person (name STRING NOT NULL),
    EDGE TYPE KNOWS (since INT64)
)
CREATE GRAPH social TYPED SocialNetwork
```

### Python API

```python
import grafeo

db = grafeo.GrafeoDB()
db.create_graph("friends")
```

### Rust API

```rust
use grafeo::GrafeoDB;

let db = GrafeoDB::new_in_memory();
db.create_graph("friends").unwrap();
```

## Switching Graphs

Once a named graph exists, switch to it with `USE GRAPH` or `SESSION SET GRAPH`. Both set the active graph for the current session; `SESSION SET GRAPH` follows the ISO/IEC 39075 session-state grammar.

```sql
-- Switch to the friends graph
USE GRAPH friends

-- ISO session syntax (equivalent)
SESSION SET GRAPH friends

-- Return to the default graph
USE GRAPH DEFAULT
```

After switching, all subsequent queries in that session read from and write to the selected graph.

### Python API

```python
db.set_graph("friends")

# Check which graph is active
print(db.current_graph())  # "friends"
```

### Rust API

```rust
let session = db.session();
session.execute("CREATE GRAPH friends")?;
session.execute("USE GRAPH friends")?;

assert_eq!(session.current_graph(), Some("friends".to_string()));
```

## Working in Several Graphs at Once

`db.set_graph()` selects one graph for the whole database object, so code that
works in two graphs would have to switch back and forth, and threads would race
on the switch. A graph handle works in one graph without switching: it has the
query methods, the direct API and transactions of the database, and every call
goes to its graph whatever `set_graph()` selects. One handle can be used from
several threads at once. A call through a handle whose graph no longer exists
raises an error; it never falls back to another graph.

### Python API

```python
db.create_graph("extraction")
db.create_graph("model")
extraction, model = db.graph("extraction"), db.graph("model")

source = extraction.create_node(["File"], {"path": "billing.py"})
component = model.create_node(["Component"], {"name": "Billing"})
model.execute("MATCH (c:Component) RETURN c.name")

with model.begin_transaction() as tx:
    tx.execute("INSERT (:Component {name: 'Ledger'})")
    tx.commit()
```

### Rust API

```rust
let model = db.graph("model")?;
model.execute("INSERT (:Component {name: 'Billing'})")?;

// Direct writes and transactions go through a session in the graph
let session = model.session()?;
session.create_node_with_props(&["Component"], [("name", Value::from("Ledger"))])?;
```

Writes to different graphs never conflict with each other: every graph numbers
its nodes and edges on its own, so node 0 of one graph and node 0 of another are
different nodes.

## Listing and Dropping Graphs

### Listing

`SHOW GRAPHS` returns all named graphs visible in the current schema context:

```sql
SHOW GRAPHS
```

In Python:

```python
names = db.list_graphs()
print(names)  # ["friends", "colleagues"]
```

In Rust:

```rust
let names = db.list_graphs();
println!("{names:?}");
```

### Dropping

Remove a named graph with `DROP GRAPH`. All data in the graph is deleted. If the session is currently using that graph, the session resets to the default graph automatically.

```sql
DROP GRAPH friends

-- No error if the graph does not exist
DROP GRAPH IF EXISTS friends
```

In Python:

```python
db.drop_graph("friends")
```

In Rust:

```rust
db.drop_graph("friends")?;
```

## Cross-Graph Transactions

`USE GRAPH` works within an active transaction. You can switch between graphs mid-transaction, and the commit or rollback applies atomically across all graphs that were touched.

```sql
START TRANSACTION

USE GRAPH friends
INSERT (:Person {name: 'Alix'})

USE GRAPH colleagues
INSERT (:Person {name: 'Gus'})

COMMIT
-- Both inserts succeed or neither does
```

If a conflict is detected in any graph during commit, the entire transaction is rolled back. Savepoints also span graph boundaries: rolling back to a savepoint restores the active graph that was set when the savepoint was created.

## Data Isolation

Each named graph has completely separate storage. Queries only see data in the active graph, and mutations in one graph never affect another.

```sql
CREATE GRAPH alpha
CREATE GRAPH beta

USE GRAPH alpha
INSERT (:Person {name: 'Alix'})

USE GRAPH beta
INSERT (:Person {name: 'Gus'})

-- Only Gus is visible here
MATCH (p:Person) RETURN p.name
```

Internally, the query plan cache keys include the active graph name. This means switching graphs does not cause stale cache hits: plans compiled for `alpha` are never reused when the session is using `beta`.

## Persistence

Named graphs are fully persisted when the database uses a persistent storage backend. `CREATE GRAPH` and `DROP GRAPH` operations are WAL-logged, so they survive crashes and are recovered on restart.

```python
import grafeo

# Persistent database
db = grafeo.GrafeoDB("my_data.grafeo")
db.execute("CREATE GRAPH friends")
db.execute("USE GRAPH friends")
db.execute("INSERT (:Person {name: 'Alix'})")

# After restart, the friends graph and its data are still present
db2 = grafeo.GrafeoDB("my_data.grafeo")
db2.execute("USE GRAPH friends")
result = db2.execute("MATCH (p:Person) RETURN p.name")
# Returns Alix
```

Snapshots also include named graph state. Both incremental and full snapshots capture all named graphs along with the default graph.

## Schema-Scoped Graphs

!!! warning "Experimental"
    Schema-scoped graph resolution is experimental and incomplete. The session schema is parsed and stored, but full catalog-level schema isolation (e.g., preventing cross-schema graph access, schema-qualified names in queries) is not yet implemented. Use this feature for forward-compatible session state only.

Grafeo supports ISO/IEC 39075 session schemas. When a schema is set, graph names are resolved within that schema context. `CREATE GRAPH` creates the graph under the active schema, and `SHOW GRAPHS` only returns graphs belonging to that schema.

```sql
-- Set the session schema
SESSION SET SCHEMA analytics

-- This graph is scoped to "analytics"
CREATE GRAPH daily_metrics

-- Only shows graphs in the analytics schema
SHOW GRAPHS

-- Switch schema
SESSION SET SCHEMA reporting
CREATE GRAPH monthly_summary
```

Schema and graph are independent session settings (per ISO/IEC 39075 Section 7.1). Changing the schema does not reset the active graph, and changing the graph does not reset the schema.

```sql
-- Reset schema without affecting the active graph
SESSION RESET SCHEMA

-- Reset graph without affecting the schema
SESSION RESET GRAPH

-- Reset everything
SESSION RESET
```
