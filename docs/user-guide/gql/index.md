---
title: GQL Query Language
description: Learn the GQL query language for Grafeo.
---

# GQL Query Language

GQL (Graph Query Language) is the ISO standard for querying property graphs (ISO/IEC 39075). Grafeo implements GQL as its primary query language.

## Overview

GQL uses pattern matching to query and manipulate graph data. The syntax is very similar to Cypher.

## Quick Reference

| Operation | Syntax |
|-----------|--------|
| Match nodes | `MATCH (n:Label)` |
| Match edges | `MATCH (a)-[:TYPE]->(b)` |
| Filter | `WHERE n.property > value` |
| Return | `RETURN n.property` |
| Create | `INSERT (:Label {prop: value})` |
| Update | `SET n.property = value` |
| Delete | `DELETE n` / `DETACH DELETE n` |
| Conditional | `CASE WHEN ... THEN ... END` |
| Cast | `CAST(expr AS type)` |
| Unwind list | `UNWIND [1, 2, 3] AS x` |
| For loop | `FOR x IN $items` |
| Combine queries | `... UNION ALL ...` |
| Call procedure | `CALL name(args) YIELD field` |
| Call subquery | `CALL { MATCH ... RETURN ... }` |
| Create type | `CREATE NODE TYPE Name (...)` |
| Create index | `CREATE INDEX FOR (n:Label) ON (n.prop)` |
| Transaction | `START TRANSACTION READ ONLY` |

## Learn More

<div class="grid cards" markdown>

-   **[Basic Queries](basic-queries.md)**

    ---

    MATCH, RETURN, OPTIONAL MATCH, WITH, LET, CALL and query composition.

-   **[Pattern Matching](patterns.md)**

    ---

    Node and edge patterns, label expressions, path quantifiers, search prefixes and path modes.

-   **[Filtering](filtering.md)**

    ---

    WHERE clauses, comparison operators, LIKE, type checking and graph predicates.

-   **[Expressions](expressions.md)**

    ---

    CASE, CAST, COALESCE, list comprehensions, reduce and subquery expressions.

-   **[Aggregations](aggregations.md)**

    ---

    COUNT, SUM, AVG, STDEV, percentiles, GROUP BY and HAVING.

-   **[Path Queries](paths.md)**

    ---

    Variable-length paths, shortest paths, path modes and path predicates.

-   **[Mutations](mutations.md)**

    ---

    INSERT, SET, DELETE, MERGE and label operations.

-   **[Composite Queries](composite-queries.md)**

    ---

    UNION, EXCEPT, INTERSECT and OTHERWISE for combining query results.

-   **[String Functions](functions-string.md)**

    ---

    String manipulation: toUpper, replace, split, substring and more.

-   **[Numeric Functions](functions-numeric.md)**

    ---

    Math functions: abs, sqrt, log, trigonometry and constants.

-   **[Temporal Functions](functions-temporal.md)**

    ---

    Date, time, datetime, duration, zoned temporals and arithmetic.

-   **[Element & Path Functions](functions-element.md)**

    ---

    Identity, labels, properties, path decomposition and list utilities.

-   **[Type System](types.md)**

    ---

    Data types, typed literals, type checking, CAST and three-valued logic.

-   **[Schema & DDL](schema.md)**

    ---

    Graph management, type definitions, indexes, constraints and stored procedures.

-   **[Transactions & Sessions](transactions.md)**

    ---

    Transaction control, isolation levels and session configuration.

</div>
