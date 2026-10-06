---
title: SQL/PGQ Query Language
description: Query graphs using standard SQL:2023 GRAPH_TABLE syntax.
---

# SQL/PGQ Query Language

SQL/PGQ (SQL:2023, ISO/IEC 9075-16) brings graph pattern matching to standard SQL. Write `SELECT ... FROM GRAPH_TABLE (MATCH ...)` and query the graph without leaving SQL.

## Overview

SQL/PGQ lets SQL developers query graphs using familiar syntax. The inner `MATCH` clause uses GQL pattern syntax, and `COLUMNS` maps graph results to SQL columns.

```sql
SELECT *
FROM GRAPH_TABLE (
    MATCH (a:Person)-[e:KNOWS]->(b:Person)
    COLUMNS (a.name AS person, e.since AS year, b.name AS friend)
) result
WHERE result.person = 'Alix'
ORDER BY result.year DESC
LIMIT 10;
```

## Basic Syntax

### GRAPH_TABLE

The `GRAPH_TABLE` function wraps a graph pattern match inside a SQL `FROM` clause:

```sql
SELECT columns
FROM GRAPH_TABLE (
    MATCH pattern
    COLUMNS (column_list)
) alias
```

### Pattern Matching

The `MATCH` clause uses GQL-style patterns:

```sql
-- Node pattern
MATCH (p:Person)

-- Edge pattern
MATCH (a:Person)-[:KNOWS]->(b:Person)

-- Multi-hop
MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(c)
```

### COLUMNS Clause

Map graph properties to SQL columns:

```sql
COLUMNS (
    a.name AS person_name,
    b.name AS friend_name,
    e.since AS year
)
```

## Examples

### Find friends of a person

```sql
SELECT *
FROM GRAPH_TABLE (
    MATCH (p:Person {name: 'Alix'})-[:KNOWS]->(f:Person)
    COLUMNS (f.name AS friend, f.age AS age)
);
```

### Friends of friends

```sql
SELECT DISTINCT result.fof_name
FROM GRAPH_TABLE (
    MATCH (me:Person {name: 'Alix'})-[:KNOWS]->()-[:KNOWS]->(fof:Person)
    COLUMNS (fof.name AS fof_name)
) result
WHERE result.fof_name <> 'Alix';
```

### Path functions

```sql
SELECT *
FROM GRAPH_TABLE (
    MATCH path = (a:Person)-[:KNOWS*1..3]->(b:Person)
    COLUMNS (
        a.name AS source,
        b.name AS target,
        LENGTH(path) AS hops
    )
);
```

## Using SQL/PGQ

=== "Python"

    ```python
    result = db.execute_sql("""
        SELECT * FROM GRAPH_TABLE (
            MATCH (p:Person)-[:KNOWS]->(f:Person)
            COLUMNS (p.name AS person, f.name AS friend)
        )
    """)
    ```

=== "Node.js"

    ```javascript
    const result = await db.executeSql(`
        SELECT * FROM GRAPH_TABLE (
            MATCH (p:Person)-[:KNOWS]->(f:Person)
            COLUMNS (p.name AS person, f.name AS friend)
        )
    `);
    ```

=== "Rust"

    ```rust
    let result = session.execute_sql(r#"
        SELECT * FROM GRAPH_TABLE (
            MATCH (p:Person)-[:KNOWS]->(f:Person)
            COLUMNS (p.name AS person, f.name AS friend)
        )
    "#)?;
    ```

=== "Go"

    ```go
    result, err := db.ExecuteSQL(`
        SELECT * FROM GRAPH_TABLE (
            MATCH (p:Person)-[:KNOWS]->(f:Person)
            COLUMNS (p.name AS person, f.name AS friend)
        )
    `)
    ```

## LEFT OUTER JOIN MATCH

Use `LEFT [OUTER] JOIN MATCH` or `OPTIONAL MATCH` inside `GRAPH_TABLE` to preserve rows that have no matching pattern. Unmatched columns are filled with NULL:

```sql
SELECT *
FROM GRAPH_TABLE (
    MATCH (p:Person)
    LEFT JOIN MATCH (p)-[:MANAGES]->(t:Team)
    COLUMNS (p.name AS person, t.name AS team)
);
```

This is equivalent to `OPTIONAL MATCH` in GQL/Cypher: people who do not manage a team still appear in the results with `team = NULL`.

## Calling Procedures

SQL/PGQ also supports `CALL` statements for invoking built-in graph algorithms:

```sql
-- Run PageRank
CALL grafeo.pagerank()

-- With parameters and column selection
CALL grafeo.pagerank({damping: 0.85}) YIELD node_id, score AS rank

-- List all available procedures
CALL grafeo.procedures()
```

See [Algorithms](../../algorithms/index.md) for the full list of available procedures.

## When to Use SQL/PGQ

**Use SQL/PGQ when:**

- The team already knows SQL
- Graph queries need to integrate into existing SQL workflows
- Standard SQL features (WHERE, ORDER BY, LIMIT, GROUP BY) are needed around graph patterns

**Use GQL directly when:**

- The full power of GQL is needed (INSERT, SET, DELETE, REMOVE)
- The work is graph-only without SQL wrapping
