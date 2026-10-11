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

Without `ORDER BY`, rows come in no particular order. The order can change between runs, builds and versions (parallel execution, compaction and planner changes all affect it), and so can which rows `LIMIT` keeps. When the order matters, say so with `ORDER BY`. To find code that relies on the order anyway, open the database with the `shuffle_unordered` option in tests (Python: `GrafeoDB(shuffle_unordered=True)`, Node.js: `GrafeoDB.create(path, { shuffleUnordered: true })`, Rust: `Config::with_shuffle_unordered(true)`): every result without `ORDER BY` then comes back in random order (a streamed result within each chunk, so the stream keeps its bounded memory).

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
ORDER BY p.age DESC NULLS FIRST
```

Nulls sort last in both directions, unless `NULLS FIRST` or `NULLS LAST` says otherwise. ISO GQL leaves this default to the implementation; Cypher queries keep openCypher's order, where nulls come last in ascending order and first in descending order.

Values of different types in one sort key, such as a property that holds a number on some nodes and a string on others, follow one fixed order, the one openCypher defines: maps, lists, paths, temporal values (zoned datetimes, datetimes, dates, zoned times, times, durations), strings, booleans, numbers, then null. Integers and floats compare as numbers, with NaN after infinity. Lists compare element by element with a prefix first, and maps by size, then keys, then values. Grafeo's own types fit in as follows: vectors after paths, bytes before strings and counters before numbers.

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

A condition on the optional part (a `WHERE` after it, or one inside its pattern) decides which matches count,
also when it reads a variable bound before: a row none of whose matches pass it keeps `null`.

```sql
-- Friends of Alix's friends who are older than Alix; a friend without one keeps null
MATCH (a:Person {name: 'Alix'})-[:KNOWS]->(b)
OPTIONAL MATCH (b)-[:KNOWS]->(c WHERE c.age > a.age)
RETURN b.name, c.name
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

`FINISH` runs the query to its end, its writes included, and returns no result: no rows and no columns. Use it for
mutation-only queries where you do not need output. A query that writes after a `MATCH` or `FOR` and ends without
`RETURN` has no result either.

```sql
-- Insert data without returning anything
MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'})
INSERT (a)-[:KNOWS]->(b)
FINISH
```

## Query Composition with NEXT

`NEXT` chains queries together: the rows the left query returns are the input of the right query, as the rows of
a `WITH` are for the clauses after it. Only the last query's `RETURN` is the result, and what a `RETURN` before
`NEXT` leaves out is not visible after it.

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

An expression in `WITH` needs a name (`WITH p.name AS name`), and a variable a `WITH` leaves out is not visible
after it.

## Statement Order

A query is a sequence of statements (`MATCH`, `OPTIONAL MATCH`, `FILTER`, `LET`, `FOR`, `CALL`, `WITH`, an
`ORDER BY` with `OFFSET` and `LIMIT`, and the writes `INSERT`, `SET`, `REMOVE` and `DELETE`) in any order, ending
in `RETURN`, `SELECT` or `FINISH`, or in nothing when it writes. Each statement reads the rows the ones before it
leave: a `WHERE` or `FILTER` filters the rows so far, and an `ORDER BY` and `LIMIT` before the end cut the rows the
statements after them see. A `WHERE` right after an `OPTIONAL MATCH` belongs to it, so a row without a match keeps
`null`; a `FILTER` there filters every row, those without a match included.

A `WHERE` or `FILTER` right after `INSERT`, `CREATE`, `MERGE` or `DELETE` is an error, because before 0.6.0 it
filtered the rows before the write: put the condition before the write, or filter the rows after it with
`WITH ... WHERE ...`. After `SET` or `REMOVE`, `FILTER` filters the rows after the write.

```sql
-- The three oldest people, then where they live
MATCH (p:Person)
ORDER BY p.age DESC LIMIT 3
MATCH (p)-[:LIVES_IN]->(c:City)
RETURN p.name, c.name

-- A WHERE between two MATCH statements
MATCH (a:Person) WHERE a.age > 30
MATCH (a)-[:KNOWS]->(b)
RETURN a.name, b.name

-- Write after WITH, then read what was written
MATCH (p:Person)
WITH p WHERE p.age > 65
SET p.retired = true
RETURN p.name, p.retired
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

A variable scope clause limits what the subquery sees: `CALL (p) { ... }` sees only `p`, and `CALL () { ... }`
sees no outer variable. What it sees stays visible in the whole body, also after a `WITH` that leaves it out:
`CALL (p) { MATCH (p)-[:KNOWS]->(f) WITH count(f) AS n RETURN p.name AS name, n }` gives one row per person,
`0` for one who knows nobody. A subquery returns new names only: returning an outer variable is an error, so rename it
(`RETURN p AS person`). A subquery without a result (no `RETURN`, or `FINISH`) runs for its writes and passes
each row on once, as it came in: nothing it binds is visible after it. The body can order and cut its rows, for the top rows per input row, and combine queries
with `UNION`, `EXCEPT`, `INTERSECT` or `OTHERWISE`:

```sql
-- Each person's oldest friend (a person who knows nobody is left out; OPTIONAL CALL keeps them)
MATCH (p:Person)
CALL (p) {
    MATCH (p)-[:KNOWS]->(friend)
    RETURN friend.name AS oldest_friend ORDER BY friend.age DESC LIMIT 1
}
RETURN p.name, oldest_friend

-- Whom each person knows or is known by
MATCH (p:Person)
CALL (p) {
    MATCH (p)-[:KNOWS]->(other) RETURN other.name AS contact
    UNION
    MATCH (other)-[:KNOWS]->(p) RETURN other.name AS contact
}
RETURN p.name, contact
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
