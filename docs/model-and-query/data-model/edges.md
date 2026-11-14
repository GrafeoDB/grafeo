---
title: Edges and Types
description: Working with edges and relationship types in Grafeo.
tags:
  - data-model
  - edges
---

# Edges and Types

Edges represent relationships between nodes. Each edge has a type, direction and can have properties.

## Creating Edges

```sql
-- Create an edge between existing nodes
MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'})
INSERT (a)-[:KNOWS]->(b)

-- Create an edge with properties
MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'})
INSERT (a)-[:KNOWS {since: 2020, strength: 'close'}]->(b)
```

## Edge Direction

Edges have a direction (source -> target):

```sql
-- Outgoing edges from Alix
MATCH (a:Person {name: 'Alix'})-[:KNOWS]->(friend)
RETURN friend.name

-- Incoming edges to Gus
MATCH (person)-[:KNOWS]->(b:Person {name: 'Gus'})
RETURN person.name

-- Either direction
MATCH (a:Person {name: 'Alix'})-[:KNOWS]-(connected)
RETURN connected.name
```

## Edge Types

Edge types categorize relationships:

```sql
-- Different relationship types
INSERT (alix)-[:KNOWS]->(gus)
INSERT (alix)-[:WORKS_WITH]->(harm)
INSERT (alix)-[:MANAGES]->(dave)

-- Query specific types
MATCH (a:Person)-[:MANAGES]->(employee)
RETURN a.name AS manager, employee.name AS employee
```

## Updating Edges

```sql
-- Update edge properties
MATCH (a:Person {name: 'Alix'})-[r:KNOWS]->(b:Person {name: 'Gus'})
SET r.strength = 'best friend'
```

## Deleting Edges

```sql
-- Delete a specific edge
MATCH (a:Person {name: 'Alix'})-[r:KNOWS]->(b:Person {name: 'Gus'})
DELETE r

-- Delete all edges of a type
MATCH ()-[r:KNOWS]->()
DELETE r
```
