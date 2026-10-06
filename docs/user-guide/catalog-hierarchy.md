# Catalog, Schemas, and Graphs

Grafeo implements the three-level namespace defined by **ISO/IEC 39075 (GQL)**:

```
Catalog (one per database)
+-- default schema
|   +-- default graph
|   +-- named graph "analytics"
+-- schema "reporting"
|   +-- default graph (auto-created)
|   +-- named graph "quarterly"
+-- schema "social"
    +-- default graph
    +-- named graph "friends"
```

A **catalog** is implicit: every `GrafeoDB` instance is its own catalog. A
**schema** is a named namespace inside the catalog. A **graph** is a named
property graph inside a schema. Each schema gets a default graph automatically,
so callers who never issue `CREATE GRAPH` still have somewhere to write.

## Session state

Two independent session fields drive graph resolution:

| Field            | Set by                    | Reset by                  |
| ---------------- | ------------------------- | ------------------------- |
| `current_schema` | `SESSION SET SCHEMA s`    | `SESSION RESET SCHEMA`    |
| `current_graph`  | `SESSION SET GRAPH g`     | `SESSION RESET GRAPH`     |

Setting one does not reset the other. Queries resolve to the storage key
`schema/graph` (for example, `reporting/quarterly`), falling back to
`<schema>/__default__` or the global default when a field is unset.

## Commands

```sql
-- Create a schema and its auto-default graph
CREATE SCHEMA reporting;

-- Create a named graph inside the current schema
SESSION SET SCHEMA reporting;
CREATE GRAPH quarterly;
SESSION SET GRAPH quarterly;

-- Typed graphs bind to graph types in the same schema, or to qualified ones
CREATE GRAPH TYPE social_network (
  NODE TYPE Person (name STRING NOT NULL),
  EDGE TYPE KNOWS
);
CREATE GRAPH friends TYPED social_network;           -- current schema
CREATE GRAPH shared  TYPED social.social_network;     -- qualified reference

-- Inspect
SHOW SCHEMAS;
SHOW GRAPHS;  -- filtered to the current schema; hides the auto-default

-- Drop (must be empty per Section 12.3)
DROP SCHEMA reporting;
```

## Isolation guarantees

- A query issued under `SESSION SET SCHEMA a` cannot see nodes, edges, labels,
  or types created under `SESSION SET SCHEMA b`.
- `CREATE NODE TYPE`, `CREATE EDGE TYPE`, and `CREATE GRAPH TYPE` are scoped
  to the current schema. `SHOW NODE TYPES` lists only the current schema's
  types.
- `DROP SCHEMA` fails unless the schema is empty (no user-created graphs and
  no types). The auto-created default graph is exempt.
- Schemas and graphs round-trip through snapshot export / import and WAL
  replay. Data isolation is preserved across reopen.

## Naming rules

Schema and graph identifiers must not contain `/`. Grafeo uses `/` internally
to build the compound storage key `schema/graph`, so `CREATE SCHEMA foo/bar`
and `CREATE GRAPH a/b` are rejected at parse time.

## Transactions across schemas

A single transaction can write to graphs in multiple schemas. Both COMMIT
and ROLLBACK are atomic across all schemas touched by the transaction,
including `SESSION SET SCHEMA` switches mid-transaction.
