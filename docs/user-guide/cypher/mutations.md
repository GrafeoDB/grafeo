---
title: Mutations
description: Creating, updating and deleting graph data in Cypher.
tags:
  - cypher
  - mutations
---

# Mutations

Cypher supports mutations for creating, updating and deleting graph data. A query that ends with an update
(`CREATE`, `MERGE`, `SET`, `REMOVE`, `DELETE`, `FOREACH` or a `CALL` subquery without `RETURN`) has no result, as
in openCypher: no rows and no columns. End it with `RETURN` to get values back; the result's write counters say
what it changed either way.

## Creating Nodes

```cypher
-- Create a node
CREATE (p:Person {name: 'Alix', age: 30})
RETURN p

-- Create multiple nodes
CREATE (a:Person {name: 'Alix'})
CREATE (b:Person {name: 'Gus'})

-- Create with multiple labels
CREATE (e:Person:Employee {name: 'Harm'})
```

## Creating Relationships

```cypher
-- Create a relationship between existing nodes
MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'})
CREATE (a)-[:KNOWS]->(b)

-- Create relationship with properties
MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'})
CREATE (a)-[:KNOWS {since: 2020, strength: 'close'}]->(b)

-- Create nodes and relationships together
CREATE (a:Person {name: 'Alix'})-[:KNOWS]->(b:Person {name: 'Gus'})
RETURN a, b
```

## Updating Properties

```cypher
-- Set a property
MATCH (p:Person {name: 'Alix'})
SET p.age = 31

-- Set multiple properties
MATCH (p:Person {name: 'Alix'})
SET p.age = 31, p.city = 'New York'

-- Set from another property
MATCH (p:Person)
SET p.displayName = p.firstName + ' ' + p.lastName

-- Replace all properties
MATCH (p:Person {name: 'Alix'})
SET p = {name: 'Alix', age: 31, city: 'NYC'}

-- Add to existing properties
MATCH (p:Person {name: 'Alix'})
SET p += {city: 'NYC', active: true}
```

## Removing Properties

```cypher
-- Remove a property
MATCH (p:Person {name: 'Alix'})
REMOVE p.temporaryField

-- Set to null (equivalent)
MATCH (p:Person {name: 'Alix'})
SET p.temporaryField = null
```

## Deleting Nodes

```cypher
-- Delete a node (must have no relationships)
MATCH (p:Person {name: 'Alix'})
DELETE p

-- Delete node and all its relationships
MATCH (p:Person {name: 'Alix'})
DETACH DELETE p
```

## Deleting Relationships

```cypher
-- Delete specific relationship
MATCH (a:Person {name: 'Alix'})-[r:KNOWS]->(b:Person {name: 'Gus'})
DELETE r

-- Delete all relationships of a type from a node
MATCH (p:Person {name: 'Alix'})-[r:KNOWS]->()
DELETE r
```

## UNWIND (List Expansion)

Expand a list into individual rows. Useful for batch operations.

```cypher
-- Unwind a literal list
UNWIND [1, 2, 3] AS x
RETURN x

-- Unwind with parameters (Python: db.execute_cypher(query, {'names': ['Alix', 'Gus']}))
UNWIND $names AS name
RETURN name

-- Batch create relationships from a parameter list
UNWIND $edges AS e
MATCH (a:Person {name: e.from}), (b:Person {name: e.to})
CREATE (a)-[:KNOWS]->(b)
```

## Merge (Upsert)

```cypher
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
MERGE (a)-[:KNOWS]->(b)

-- Merge with inline relationship SET
MERGE (a:Person {id: 1})-[r:KNOWS]->(b:Person {id: 2})
SET r.weight = 0.5
```

When the pattern matches more than one existing node or relationship, `MERGE` binds each of them, one row per
match, and `ON MATCH` and a later `SET` apply to all of them.

## FOREACH

Iterate over a list and execute mutations for each element. `FOREACH` passes each row on once, unchanged; its
variable and what its updates bind are not visible after it:

```cypher
-- Create nodes from a list
FOREACH (name IN ['Alix', 'Gus', 'Vincent'] |
    CREATE (:Person {name: name})
)

-- Update nodes from a list
MATCH (p:Person)
WITH collect(p) AS people
FOREACH (person IN people |
    SET person.updated = true
)
```

## CALL Subqueries

Run a subquery for each input row. The subquery sees the outer variables its importing `WITH` lists, or the ones
a variable scope clause names (`CALL (p) { ... }`, `CALL (*) { ... }` for all of them, `CALL () { ... }` for
none). An importing `WITH` only lists variables: a `WHERE`, `DISTINCT`, alias or expression in it, or an
`ORDER BY`, `SKIP` or `LIMIT` right after it, is an error (a second `WITH` can do those). A subquery returns new
names only, so rename an imported variable to return it (`RETURN p AS person`). A subquery without a final
`RETURN` (a unit subquery) runs for its writes and passes each row on once, as it came in, as `FOREACH` does:
nothing it binds is visible after it.

```cypher
-- Per-person friend count via subquery
MATCH (p:Person)
CALL {
    WITH p
    MATCH (p)-[:KNOWS]->(friend)
    RETURN count(friend) AS friend_count
}
RETURN p.name, friend_count

-- Subquery with mutations
MATCH (p:Person)
CALL {
    WITH p
    MATCH (p)-[:OWNS]->(item:Item)
    WHERE item.expired = true
    DETACH DELETE item
    RETURN count(*) AS deleted
}
RETURN p.name, deleted

-- Each person's oldest friend, with a scope clause (a person who knows nobody is left out)
MATCH (p:Person)
CALL (p) {
    MATCH (p)-[:KNOWS]->(friend)
    RETURN friend.name AS oldest_friend ORDER BY friend.age DESC LIMIT 1
}
RETURN p.name, oldest_friend

-- Parts joined by UNION, each with its own importing WITH
MATCH (p:Person)
CALL {
    WITH p MATCH (p)-[:KNOWS]->(other) RETURN other.name AS contact
    UNION
    WITH p MATCH (other)-[:KNOWS]->(p) RETURN other.name AS contact
}
RETURN p.name, contact
```

## UNION

Combine results from multiple queries:

```cypher
-- UNION (deduplicates rows)
MATCH (p:Person) WHERE p.city = 'Amsterdam'
RETURN p.name AS name
UNION
MATCH (p:Person) WHERE p.city = 'Berlin'
RETURN p.name AS name

-- UNION ALL (keeps duplicates)
MATCH (p:Person)-[:LIVES_IN]->(c:City)
RETURN c.name AS city
UNION ALL
MATCH (p:Person)-[:WORKS_IN]->(c:City)
RETURN c.name AS city
```

## LOAD CSV

Import data from CSV files:

```cypher
-- With headers (access fields by name)
LOAD CSV WITH HEADERS FROM 'data/people.csv' AS row
CREATE (:Person {name: row.name, age: toInteger(row.age)})

-- Without headers (access fields by index)
LOAD CSV FROM 'data/people.csv' AS row
CREATE (:Person {name: row[0], age: toInteger(row[1])})

-- Custom field terminator
LOAD CSV WITH HEADERS FROM 'data/people.tsv' AS row FIELDTERMINATOR '\t'
CREATE (:Person {name: row.name})

-- File URI
LOAD CSV WITH HEADERS FROM 'file:///data/people.csv' AS row
RETURN row.name
```

!!! note "File path restrictions"
    LOAD CSV reads from the local filesystem only. You can specify a relative path (e.g., `'data/people.csv'`) or a `file://` URI (e.g., `'file:///data/people.csv'`). Remote URLs such as `http://` or `https://` are not supported. The `file://` prefix is stripped before opening the file, so both `'data/file.csv'` and `'file:///data/file.csv'` resolve to the same local path.

## Schema DDL

Create and manage indexes and constraints:

```cypher
-- Create index
CREATE INDEX FOR (p:Person) ON (p.name)

-- Create vector index
CREATE INDEX FOR (n:Document) ON (n.embedding) USING VECTOR

-- Create text index
CREATE INDEX FOR (n:Article) ON (n.content) USING TEXT

-- Create constraint
CREATE CONSTRAINT FOR (p:Person) REQUIRE p.email IS UNIQUE

-- List indexes and constraints
SHOW INDEXES
SHOW CONSTRAINTS

-- Drop
DROP INDEX index_name
DROP CONSTRAINT constraint_name
```

## EXPLAIN and PROFILE

```cypher
-- Show query plan without executing
EXPLAIN MATCH (p:Person)-[:KNOWS]->(f:Person)
RETURN f.name

-- Execute and return per-operator metrics
PROFILE MATCH (p:Person)-[:KNOWS]->(f:Person)
RETURN f.name
```
