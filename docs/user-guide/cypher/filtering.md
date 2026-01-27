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
WHERE p.name = 'Alice'

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

## Path Filtering

```cypher
-- Filter based on relationship properties
MATCH (a)-[r:KNOWS]->(b)
WHERE r.since > 2020
RETURN a.name, b.name
```
