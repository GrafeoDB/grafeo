---
title: Filtering
description: Filtering results with WHERE clauses in Cypher.
tags:
  - cypher
  - filtering
---

# Filtering

The `WHERE` clause filters results based on conditions.

## Comparison Operators

```cypher
-- Equality
WHERE p.name = 'Alix'

-- Inequality
WHERE p.age <> 30

-- Greater/less than
WHERE p.age > 25
WHERE p.age < 40
WHERE p.age >= 25
WHERE p.age <= 40
```

## Boolean Logic

```cypher
-- AND
WHERE p.age > 25 AND p.active = true

-- OR
WHERE p.city = 'NYC' OR p.city = 'LA'

-- NOT
WHERE NOT p.archived

-- Combined
WHERE (p.age > 25 AND p.active) OR p.role = 'admin'
```

## String Operations

```cypher
-- Starts with
WHERE p.name STARTS WITH 'Al'

-- Ends with
WHERE p.email ENDS WITH '@company.com'

-- Contains
WHERE p.bio CONTAINS 'engineer'

-- Regular expression
WHERE p.email =~ '.*@gmail\\.com'
```

## List Operations

```cypher
-- IN list
WHERE p.status IN ['active', 'pending']

-- Element in property list
WHERE 'admin' IN p.roles
```

## Null Checks

```cypher
-- Is null
WHERE p.email IS NULL

-- Is not null
WHERE p.email IS NOT NULL
```

## Property Existence

```cypher
-- Check if property exists
WHERE exists(p.email)

-- Combined with value check
WHERE p.age IS NOT NULL AND p.age > 18
```

## Relationship WHERE

Inline predicates directly on relationship patterns:

```cypher
-- Inline predicate on relationship
MATCH (a)-[r:KNOWS WHERE r.since > 2020]->(b)
RETURN a.name, b.name

-- Equivalent to separate WHERE
MATCH (a)-[r:KNOWS]->(b)
WHERE r.since > 2020
RETURN a.name, b.name
```

## EXISTS Subqueries

Check for the existence of a pattern:

```cypher
-- People who manage at least one team
MATCH (p:Person)
WHERE EXISTS {
    MATCH (p)-[:MANAGES]->(:Team)
}
RETURN p.name

-- NOT EXISTS
MATCH (p:Person)
WHERE NOT EXISTS {
    MATCH (p)-[:ORDERED]->(:Order)
}
RETURN p.name AS no_orders
```

## List Predicate Functions

```cypher
-- any(): at least one element matches
MATCH (p:Person)
WHERE any(role IN p.roles WHERE role = 'admin')
RETURN p.name

-- all(): every element matches
MATCH path = (a)-[:KNOWS*]->(b)
WHERE all(n IN nodes(path) WHERE n.active = true)
RETURN path

-- none(): no element matches
MATCH (p:Person)
WHERE none(tag IN p.tags WHERE tag = 'inactive')
RETURN p.name

-- single(): exactly one element matches
MATCH (p:Person)
WHERE single(role IN p.roles WHERE role = 'lead')
RETURN p.name
```

## reduce()

Fold a list into a single value:

```cypher
-- Sum a list
RETURN reduce(acc = 0, x IN [1, 2, 3, 4, 5] | acc + x) AS total

-- String concatenation
MATCH (p:Person)-[:KNOWS]->(f:Person)
WITH p, collect(f.name) AS friends
RETURN p.name, reduce(s = '', name IN friends | s + name + ', ') AS friend_list
```

## Element Functions

```cypher
-- Get internal element ID
MATCH (p:Person)
RETURN elementId(p), p.name

-- Get labels
MATCH (n)
RETURN labels(n)

-- Get relationship type
MATCH ()-[r]->()
RETURN type(r)
```
