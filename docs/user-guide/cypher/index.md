---
title: Cypher Query Language
description: Learn the Cypher query language for Graphos.
---

# Cypher Query Language

Cypher is a declarative graph query language originally developed by Neo4j. Graphos fully supports Cypher alongside GQL.

## Overview

Cypher uses ASCII-art style pattern matching to query and manipulate graph data. It's designed to be intuitive and readable.

## Quick Reference

| Operation | Syntax |
|-----------|--------|
| Match nodes | `MATCH (n:Label)` |
| Match edges | `MATCH (a)-[:TYPE]->(b)` |
| Filter | `WHERE n.property > value` |
| Return | `RETURN n.property` |
| Create | `CREATE (:Label {prop: value})` |
| Update | `SET n.property = value` |
| Delete | `DELETE n` |

## Learn More

<div class="grid cards" markdown>

-   **[Basic Queries](basic-queries.md)**

    ---

    MATCH, RETURN, and basic pattern matching.

-   **[Pattern Matching](patterns.md)**

    ---

    Node and edge patterns in detail.

-   **[Filtering](filtering.md)**

    ---

    WHERE clauses and conditions.

-   **[Aggregations](aggregations.md)**

    ---

    COUNT, SUM, AVG, and grouping.

-   **[Path Queries](paths.md)**

    ---

    Variable-length paths and shortest paths.

-   **[Mutations](mutations.md)**

    ---

    CREATE, SET, DELETE operations.

</div>
