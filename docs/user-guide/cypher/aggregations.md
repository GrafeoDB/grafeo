---
title: Aggregations
description: Aggregation functions and grouping in Cypher.
tags:
  - cypher
  - aggregations
---

# Aggregations

Cypher provides aggregation functions for computing summaries over query results.

## Aggregation Functions

| Function | Description |
|----------|-------------|
| `count()` | Count items |
| `sum()` | Sum of values |
| `avg()` | Average of values |
| `min()` | Minimum value |
| `max()` | Maximum value |
| `collect()` | Collect into list |

## Count

```cypher
-- Count all nodes
MATCH (n)
RETURN count(n)

-- Count with label
MATCH (p:Person)
RETURN count(p)

-- Count distinct
MATCH (p:Person)-[:LIVES_IN]->(c:City)
RETURN count(DISTINCT c)
```

## Sum, Avg, Min, Max

```cypher
-- Sum
MATCH (o:Order)
RETURN sum(o.total)

-- Average
MATCH (p:Person)
RETURN avg(p.age)

-- Min and Max
MATCH (p:Product)
RETURN min(p.price), max(p.price)
```

## Collect

```cypher
-- Collect into list
MATCH (p:Person)
RETURN collect(p.name)

-- Collect with limit
MATCH (p:Person)
RETURN collect(p.name)[0..5]
```

## Grouping

```cypher
-- Group by property
MATCH (p:Person)
RETURN p.city, count(p) AS population
ORDER BY population DESC

-- Group by relationship target
MATCH (p:Person)-[:WORKS_AT]->(c:Company)
RETURN c.name, count(p) AS employees

-- Multiple aggregations
MATCH (o:Order)
RETURN
    o.status,
    count(o) AS order_count,
    sum(o.total) AS total_revenue,
    avg(o.total) AS avg_order_value
```

## WITH Clause

The `WITH` clause allows intermediate aggregations:

```cypher
-- Find people with more than 5 friends
MATCH (p:Person)-[:KNOWS]->(friend)
WITH p, count(friend) AS friend_count
WHERE friend_count > 5
RETURN p.name, friend_count
```

## UNWIND

The `UNWIND` clause expands a list into rows:

```cypher
-- Iterate over a list
UNWIND [1, 2, 3] AS x
RETURN x

-- Create nodes from list
UNWIND ['Alice', 'Bob', 'Carol'] AS name
CREATE (p:Person {name: name})
RETURN p
```
