---
title: Mutations
description: Creating, updating and deleting graph data in GQL.
tags:
  - gql
  - mutations
---

# Mutations

GQL supports mutations for creating, updating and deleting graph data.

## Creating Nodes

```sql
-- Create a node
INSERT (:Person {name: 'Alix', age: 30})

-- Create multiple nodes
INSERT (:Person {name: 'Alix'})
INSERT (:Person {name: 'Gus'})

-- Create with multiple labels
INSERT (:Person:Employee {name: 'Harm'})
```

## Creating Edges

```sql
-- Create an edge between existing nodes
MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'})
INSERT (a)-[:KNOWS]->(b)

-- Create edge with properties
MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'})
INSERT (a)-[:KNOWS {since: 2020, strength: 'close'}]->(b)
```

## Updating Properties

```sql
-- Set a property
MATCH (p:Person {name: 'Alix'})
SET p.age = 31

-- Set multiple properties
MATCH (p:Person {name: 'Alix'})
SET p.age = 31, p.city = 'New York'

-- Set from another property
MATCH (p:Person)
SET p.displayName = p.firstName + ' ' + p.lastName
```

## Removing Properties

```sql
-- Remove a property
MATCH (p:Person {name: 'Alix'})
REMOVE p.temporaryField

-- Set to null (equivalent)
MATCH (p:Person {name: 'Alix'})
SET p.temporaryField = null
```

## Map Property Operations

### Replace All Properties

`SET n = {map}` replaces all existing properties with the map contents:

```sql
-- Replace all properties on a node
MATCH (p:Person {name: 'Alix'})
SET p = {name: 'Alix', age: 31, city: 'NYC'}
-- Any properties not in the map are removed
```

### Merge Properties

`SET n += {map}` merges the map into existing properties, keeping properties not in the map:

```sql
-- Add or update properties, keep existing ones
MATCH (p:Person {name: 'Alix'})
SET p += {city: 'NYC', role: 'engineer'}
-- Existing properties like name and age are preserved
```

## Label Operations

```sql
-- Add labels to a node
MATCH (p:Person {name: 'Alix'})
SET p:Employee:Manager

-- Remove a label
MATCH (p:Person {name: 'Alix'})
REMOVE p:Manager
```

## Deleting Nodes

GQL supports two delete modes for nodes:

```sql
-- DELETE (or NODETACH DELETE): errors if the node has edges
MATCH (p:Person {name: 'Alix'})
DELETE p

-- Explicit NODETACH (same behavior as bare DELETE)
MATCH (p:Person {name: 'Alix'})
NODETACH DELETE p

-- DETACH DELETE: delete the node and all its connected edges
MATCH (p:Person {name: 'Alix'})
DETACH DELETE p
```

!!! tip
    Use `DELETE` (without DETACH) when you want to ensure no edges are accidentally removed. The query will fail if the node still has connections, giving you a chance to handle them explicitly.

## Deleting Edges

```sql
-- Delete specific edge
MATCH (a:Person {name: 'Alix'})-[r:KNOWS]->(b:Person {name: 'Gus'})
DELETE r

-- Delete all edges of a type from a node
MATCH (p:Person {name: 'Alix'})-[r:KNOWS]->()
DELETE r
```

## DELETE with Expressions

DELETE targets can be general expressions, not just plain variables (GD04):

```sql
-- Delete using a property access expression
MATCH (n:Person)-[:MANAGES]->(dept:Department)
DELETE dept

-- Delete with a function expression
MATCH (n:Person)-[:HAS]->(item)
WHERE item.expired = true
DELETE item
```

## UNWIND (List Expansion)

Expand a list into individual rows. Useful for batch operations.

```sql
-- Unwind a literal list
UNWIND [1, 2, 3] AS x
RETURN x

-- Unwind with parameters (Python: db.execute(query, {'names': ['Alix', 'Gus']}))
UNWIND $names AS name
RETURN name

-- Batch create edges from a parameter list
UNWIND $edges AS e
MATCH (a:Person {name: e.from}), (b:Person {name: e.to})
INSERT (a)-[:KNOWS]->(b)
```

## FOR (GQL Standard List Iteration)

The GQL standard equivalent of UNWIND (ISO/IEC 39075 section 14.8).

```sql
-- Iterate over a list
FOR x IN [4, 5, 6]
RETURN x

-- Batch insert nodes
FOR person IN $people
INSERT (:Person {name: person.name, age: person.age})
```

### FOR WITH ORDINALITY / OFFSET

Track the position of each element during iteration:

```sql
-- 1-based index (ORDINALITY)
FOR x IN ['a', 'b', 'c'] WITH ORDINALITY i
RETURN x, i
-- Returns: ('a', 1), ('b', 2), ('c', 3)

-- 0-based index (OFFSET)
FOR x IN ['a', 'b', 'c'] WITH OFFSET i
RETURN x, i
-- Returns: ('a', 0), ('b', 1), ('c', 2)
```

## Merge (Upsert)

```sql
-- Create if not exists, match if exists
MERGE (p:Person {email: 'alix@example.com'})
SET p.lastSeen = timestamp()
RETURN p

-- Merge with different actions
MERGE (p:Person {email: 'alix@example.com'})
ON CREATE SET p.created = timestamp()
ON MATCH SET p.lastSeen = timestamp()
RETURN p

-- Merge relationships
MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'})
MERGE (a)-[r:KNOWS]->(b)
ON CREATE SET r.since = 2024
RETURN r

-- Merge with UNWIND for batch upserts
UNWIND $people AS person
MERGE (p:Person {email: person.email})
ON CREATE SET p.name = person.name, p.created = timestamp()
ON MATCH SET p.lastSeen = timestamp()
```

When the pattern matches more than one existing node or relationship, `MERGE` binds each of them, one row per
match, and `ON MATCH` and a later `SET` apply to all of them.

## LOAD DATA (Multi-Format Import)

Import data from external files directly in GQL:

```sql
-- CSV import
LOAD DATA FROM 'data/people.csv' FORMAT CSV WITH HEADERS AS row
INSERT (:Person {name: row.name, age: CAST(row.age AS INT)})

-- JSONL import (requires jsonl-import feature)
LOAD DATA FROM 'data/events.jsonl' FORMAT JSONL AS event
INSERT (:Event {type: event.type, timestamp: event.ts})

-- Parquet import (requires parquet-import feature)
LOAD DATA FROM 'data/records.parquet' FORMAT PARQUET AS rec
INSERT (:Record {id: rec.id, value: rec.value})
```

## EXPLAIN and PROFILE

Inspect query plans without or with execution:

```sql
-- Show optimized logical plan (no execution)
EXPLAIN MATCH (p:Person)-[:KNOWS]->(f:Person)
WHERE p.name = 'Alix'
RETURN f.name

-- Execute and return per-operator metrics (rows, time, calls)
PROFILE MATCH (p:Person)-[:KNOWS]->(f:Person)
WHERE p.name = 'Alix'
RETURN f.name
```
