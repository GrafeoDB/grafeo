---
title: Basic Queries
description: Learn basic GQL queries with MATCH and RETURN.
tags:
  - gql
  - queries
---

# Basic Queries

This guide covers the fundamentals of querying graphs with GQL.

## MATCH Clause

The `MATCH` clause finds patterns in the graph:

```sql
-- Match all nodes
MATCH (n)
RETURN n

-- Match nodes with a label
MATCH (p:Person)
RETURN p

-- Match nodes with properties
MATCH (p:Person {name: 'Alix'})
RETURN p
```

## RETURN Clause

The `RETURN` clause specifies what to return:

```sql
-- Return entire nodes
MATCH (p:Person)
RETURN p

-- Return specific properties
MATCH (p:Person)
RETURN p.name, p.age

-- Return with aliases
MATCH (p:Person)
RETURN p.name AS name, p.age AS age
```

## Combining MATCH and RETURN

```sql
-- Find all people and return their names
MATCH (p:Person)
RETURN p.name

-- Find people over 30
MATCH (p:Person)
WHERE p.age > 30
RETURN p.name, p.age

-- Find Alix's friends
MATCH (a:Person {name: 'Alix'})-[:KNOWS]->(friend)
RETURN friend.name
```

## Ordering Results

Without `ORDER BY`, rows come in no particular order. The order can change between runs, builds and versions (parallel execution, compaction and planner changes all affect it), and so can which rows `LIMIT` keeps. When the order matters, say so with `ORDER BY`. To find code that relies on the order anyway, open the database with the `shuffle_unordered` option in tests (Python: `GrafeoDB(shuffle_unordered=True)`, Node.js: `GrafeoDB.create(path, { shuffleUnordered: true })`, Rust: `Config::with_shuffle_unordered(true)`): every result without `ORDER BY` then comes back in random order.

```sql
-- Order by property
MATCH (p:Person)
RETURN p.name, p.age
ORDER BY p.age

-- Descending order
MATCH (p:Person)
RETURN p.name, p.age
ORDER BY p.age DESC

-- Multiple sort keys
MATCH (p:Person)
RETURN p.name, p.age
ORDER BY p.age DESC, p.name ASC

-- Control null placement (ISO GA03)
MATCH (p:Person)
RETURN p.name, p.age
ORDER BY p.age ASC NULLS FIRST

MATCH (p:Person)
RETURN p.name, p.age
ORDER BY p.age DESC NULLS LAST
```

## Limiting Results

```sql
-- Return first 10 results
MATCH (p:Person)
RETURN p.name
LIMIT 10

-- Skip and limit (pagination)
MATCH (p:Person)
RETURN p.name
ORDER BY p.name
SKIP 20 LIMIT 10
```

## DISTINCT Results

```sql
-- Remove duplicates
MATCH (p:Person)-[:LIVES_IN]->(c:City)
RETURN DISTINCT c.name
```

## OPTIONAL MATCH

`OPTIONAL MATCH` works like `MATCH`, but returns `null` for variables that have no match instead of filtering the row out entirely. This is similar to a SQL `LEFT JOIN`.

```sql
-- Find all people and optionally their pets
MATCH (p:Person)
OPTIONAL MATCH (p)-[:HAS_PET]->(pet:Animal)
RETURN p.name, pet.name
-- People without pets show null for pet.name

-- Chain optional patterns
MATCH (p:Person)
OPTIONAL MATCH (p)-[:WORKS_AT]->(c:Company)
OPTIONAL MATCH (c)-[:LOCATED_IN]->(city:City)
RETURN p.name, c.name, city.name
```

## SELECT (ISO Alternative to RETURN)

The ISO GQL standard uses `SELECT` as an alternative to `RETURN`. The semantics are identical.

```sql
-- These two queries are equivalent
MATCH (p:Person) WHERE p.age > 30
SELECT p.name, p.age

MATCH (p:Person) WHERE p.age > 30
RETURN p.name, p.age
```

## FINISH

`FINISH` consumes all input rows and returns an empty result. Use it for mutation-only queries where you do not need output.

```sql
-- Insert data without returning anything
MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'})
INSERT (a)-[:KNOWS]->(b)
FINISH
```

## Query Composition with NEXT

`NEXT` chains queries together: the output of the left query feeds into the right query as input. This enables multi-step transformations.

```sql
-- Find friends, then filter by age
MATCH (p:Person {name: 'Alix'})-[:KNOWS]->(friend)
RETURN friend
NEXT
MATCH (friend) WHERE friend.age > 25
RETURN friend.name, friend.age
```

## WITH Clause

The `WITH` clause creates an intermediate result that can be filtered or transformed before continuing the query.

```sql
-- Intermediate aggregation, then filter
MATCH (p:Person)-[:KNOWS]->(friend)
WITH p, count(friend) AS friend_count
WHERE friend_count > 5
RETURN p.name, friend_count

-- Pass all variables through with WITH *
MATCH (p:Person)-[:KNOWS]->(friend)
WITH *
WHERE friend.age > 25
RETURN p.name, friend.name
```

## LET (Variable Binding)

`LET` assigns computed values to variables for use in subsequent clauses.

```sql
-- Compute a derived value
MATCH (p:Person)
LET full_name = p.firstName + ' ' + p.lastName
RETURN full_name, p.age

-- Multiple bindings
MATCH (p:Person)
LET age_group = CASE WHEN p.age < 30 THEN 'young' ELSE 'senior' END,
    display = toUpper(p.name)
RETURN display, age_group
```

## Calling Procedures

### Named Procedure CALL

Use `CALL` to invoke a named procedure. `YIELD` selects which output fields to bind.

```sql
-- Call a built-in algorithm
CALL grafeo.pagerank() YIELD node, score
RETURN node.name, score
ORDER BY score DESC
LIMIT 10

-- Filter yielded results
CALL grafeo.pagerank() YIELD node, score
WHERE score > 0.5
RETURN node.name, score
```

### Inline Subquery CALL

`CALL { ... }` runs an inline subquery for each input row. Variables from the outer query are visible inside the block.

```sql
-- Per-person friend count via subquery
MATCH (p:Person)
CALL {
    MATCH (p)-[:KNOWS]->(friend)
    RETURN count(friend) AS friend_count
}
RETURN p.name, friend_count
```

### OPTIONAL CALL

`OPTIONAL CALL` returns `null` for output variables when the subquery produces no results, instead of filtering the row.

```sql
MATCH (p:Person)
OPTIONAL CALL {
    MATCH (p)-[:MANAGES]->(team:Team)
    RETURN team.name AS team_name
}
RETURN p.name, team_name
-- People who don't manage a team show null for team_name
```

## ISO Pagination Aliases

GQL supports ISO SQL-style pagination keywords as alternatives to `SKIP` and `LIMIT`:

```sql
-- OFFSET is a synonym for SKIP
MATCH (p:Person)
RETURN p.name
ORDER BY p.name
OFFSET 20 LIMIT 10

-- FETCH FIRST n ROWS is a synonym for LIMIT
MATCH (p:Person)
RETURN p.name
ORDER BY p.name
FETCH FIRST 10 ROWS
```
