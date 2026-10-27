---
title: Transactions and Sessions
description: GQL transaction control and session configuration in Grafeo.
tags:
  - gql
  - transactions
  - sessions
---

# Transactions and Sessions

GQL provides transaction control and session configuration commands.

## Transaction Control

### Starting Transactions

```sql
-- Start a read-write transaction (default)
START TRANSACTION

-- Explicit read-write
START TRANSACTION READ WRITE

-- Read-only transaction (mutations will be rejected)
START TRANSACTION READ ONLY
```

### Isolation Levels

Specify the isolation level when starting a transaction:

```sql
-- Read committed
START TRANSACTION ISOLATION LEVEL READ COMMITTED

-- Snapshot isolation (default; repeatable reads)
START TRANSACTION ISOLATION LEVEL SNAPSHOT ISOLATION

-- SERIALIZABLE is recognized but refused with GRAFEO-Q004.

-- Combine with access mode
START TRANSACTION READ ONLY ISOLATION LEVEL SNAPSHOT ISOLATION
```

Serializable requests, including `READ ONLY`, fail with `GRAFEO-Q004` until [query read tracking (#400)](https://github.com/GrafeoDB/grafeo/issues/400) is implemented. Use snapshot isolation.

### Committing and Rolling Back

```sql
-- Commit the current transaction
COMMIT

-- Roll back the current transaction
ROLLBACK
```

Schema statements (creating, altering and dropping types, constraints, indexes and procedures) and graph commands (`CREATE GRAPH`, `DROP GRAPH`) take effect immediately, also inside a transaction, and a rollback does not undo them.

### Savepoints

Create named savepoints within a transaction for partial rollback:

```sql
-- Create a savepoint
SAVEPOINT sp1

-- Roll back to savepoint (undo changes made after it)
ROLLBACK TO SAVEPOINT sp1

-- Release a savepoint (discard it, keep changes)
RELEASE SAVEPOINT sp1
```

### Example: Savepoint Workflow

```sql
START TRANSACTION READ WRITE

-- First update
MATCH (a:Account {id: 'A001'})
SET a.balance = a.balance + 500

SAVEPOINT before_bonus

-- Tentative bonus
MATCH (a:Account {id: 'A001'})
SET a.bonus = 100

-- Changed our mind, undo the bonus
ROLLBACK TO SAVEPOINT before_bonus

-- The +500 balance change is preserved
COMMIT
```

### Nested Transactions

Starting a transaction inside an existing transaction creates an implicit savepoint:

```sql
START TRANSACTION READ WRITE
MATCH (n:Counter) SET n.value = 1

-- Creates implicit savepoint (inner transaction)
START TRANSACTION
MATCH (n:Counter) SET n.value = 99
ROLLBACK  -- Rolls back to the implicit savepoint

-- n.value is still 1
COMMIT
```

### Example: Transaction Workflow

```sql
-- Transfer between accounts
START TRANSACTION READ WRITE

MATCH (src:Account {id: 'A001'})
SET src.balance = src.balance - 100

MATCH (dst:Account {id: 'A002'})
SET dst.balance = dst.balance + 100

COMMIT
```

```sql
-- Read-only reporting query
START TRANSACTION READ ONLY

MATCH (p:Person)-[:WORKS_AT]->(c:Company)
RETURN c.name, count(p) AS employees
ORDER BY employees DESC

COMMIT
```

## Session Commands

### Switching Graphs

```sql
-- Set the active graph for the session
USE GRAPH my_graph

-- SESSION SET equivalent
SESSION SET GRAPH my_graph
```

### Time Zone

```sql
-- Set the session time zone: a UTC offset
SESSION SET TIME ZONE '+01:00'
SESSION SET TIME ZONE '-05:30'
SESSION SET TIME ZONE 'UTC'
```

The time zone is a time zone displacement, as ISO GQL defines it: a UTC offset of at most 14 hours, `Z` or
`UTC`. Any other text, such as a named zone (`Europe/Amsterdam`), fails with GQLSTATUS 22009 (invalid time zone
displacement value). Named zones may come later ([#651](https://github.com/GrafeoDB/grafeo/issues/651)).

GQL and SQL/PGQ compare a local datetime or time with a zoned one by reading the local value at the session
time zone, as SQL does: with `'+01:00'`, `local_datetime('2024-03-15T10:00:00')` equals
`zoned_datetime('2024-03-15T10:00:00+01:00')`; without a session time zone the local value is read at UTC.
Cypher keeps the two apart, as openCypher defines it: a zoned and a local value are never equal, and `<` or
`>` between them is null. The time zone carries over to the next call of a database's `execute`, as the graph
and the schema do, and `SESSION RESET TIME ZONE` clears it.

### Schema

```sql
-- Set the default schema for the session
SESSION SET SCHEMA analytics
```

### Session Parameters

Set named parameters that persist for the session:

```sql
-- Set a session parameter
SESSION SET PARAMETER $threshold = 0.5

-- Use it in subsequent queries
MATCH (p:Person)
WHERE p.score > $threshold
RETURN p.name
```

### Reset and Close

```sql
-- Reset all session settings to defaults
SESSION RESET

-- Reset all settings
SESSION RESET ALL

-- Close the session
SESSION CLOSE
```

## Programmatic API

For transaction control from host languages (Python, Node.js, Rust), see [Transactions](../transactions.md).
