---
title: Cypher vs GQL Syntax
description: Syntax differences between Cypher and GQL.
tags:
  - cypher
  - gql
---

# Cypher vs GQL Syntax

Both Cypher and GQL are supported in Graphos. This page documents the syntax differences for users who want to understand both or switch between them.

## Creating Nodes

=== "Cypher"

    ```cypher
    CREATE (n:Person {name: 'Alice', age: 30})
    RETURN n
    ```

=== "GQL"

    ```sql
    INSERT (:Person {name: 'Alice', age: 30})
    ```

!!! tip
    GQL's `INSERT` doesn't require a `RETURN` clause for simple creates.

## Creating Relationships

=== "Cypher"

    ```cypher
    MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
    CREATE (a)-[r:KNOWS {since: 2020}]->(b)
    RETURN r
    ```

=== "GQL"

    ```sql
    MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
    INSERT (a)-[:KNOWS {since: 2020}]->(b)
    ```

## Identical Syntax

Many features are identical in both languages:

```sql
-- Works in both Cypher and GQL
MATCH (p:Person)
WHERE p.age > 25
RETURN p.name, p.age
ORDER BY p.age DESC
LIMIT 10
```

## Pattern Matching

```sql
-- Same in both
MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person)
WHERE a <> c
RETURN a.name, c.name
```

## Variable-Length Paths

=== "Cypher"

    ```cypher
    MATCH path = (a:Person)-[:KNOWS*1..3]->(b:Person)
    RETURN path
    ```

=== "GQL"

    ```sql
    MATCH path = (a:Person)-[:KNOWS*1..3]->(b:Person)
    RETURN path
    ```

The syntax is identical for variable-length paths.

## Function Differences

| Cypher | GQL | Description |
|--------|-----|-------------|
| `id(n)` | `id(n)` | Internal node ID |
| `type(r)` | `type(r)` | Relationship type |
| `labels(n)` | `labels(n)` | Node labels |
| `toUpper(s)` | `upper(s)` | Uppercase string |
| `toLower(s)` | `lower(s)` | Lowercase string |
| `size(list)` | `length(list)` | List length |

## UNWIND vs FOR

=== "Cypher"

    ```cypher
    UNWIND [1, 2, 3] AS x
    RETURN x
    ```

=== "GQL"

    ```sql
    FOR x IN [1, 2, 3]
    RETURN x
    ```

## Summary

| Feature | Cypher | GQL |
|---------|--------|-----|
| Create nodes | `CREATE` | `INSERT` |
| Pattern matching | `MATCH` | `MATCH` |
| Filtering | `WHERE` | `WHERE` |
| Iteration | `UNWIND` | `FOR` |
| Everything else | Mostly identical | Mostly identical |
