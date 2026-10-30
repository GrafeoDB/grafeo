---
title: Coming from Cypher
description: What changes when you move a query from Cypher to GQL in Grafeo.
tags:
  - gql
  - cypher
  - migration
---

# Coming from Cypher

Grafeo runs openCypher and ISO GQL on the same database, and most Cypher queries read much the same in GQL. This
page lists the places where the same query gives other rows in GQL, or where GQL writes it another way.

## Path Mode

A variable-length pattern in GQL matches walks by default (`WALK`): a walk may use an edge more than once, so it can
go back along the edge it came in on. openCypher uses each relationship at most once in a pattern, as a trail. On a
graph with one edge, Alix KNOWS Gus:

=== "Cypher"

    ```cypher
    MATCH (a:Person {name: 'Alix'})-[:KNOWS*1..2]-(b)
    RETURN b.name
    // Gus
    ```

=== "GQL"

    ```sql
    MATCH (a:Person {name: 'Alix'})-[:KNOWS]-{1,2}(b)
    RETURN b.name
    -- Gus, and Alix (the walk Alix, Gus, Alix)

    -- The rows Cypher gives: no edge twice
    MATCH TRAIL (a:Person {name: 'Alix'})-[:KNOWS]-{1,2}(b)
    RETURN b.name
    -- Gus
    ```

Write `MATCH TRAIL ...` to get Cypher's rows. See [Path Modes](paths.md#path-modes) for `ACYCLIC` and `SIMPLE`.

## Numbers With a Point

In GQL a number with a point is an exact `DECIMAL`; in Cypher it is a float:

=== "Cypher"

    ```cypher
    RETURN 0.1 + 0.2 = 0.3 AS exact
    // false
    ```

=== "GQL"

    ```sql
    RETURN 0.1 + 0.2 = 0.3 AS exact
    -- true
    ```

A decimal compares with a float by value, so `WHERE n.score > 0.77` keeps the same rows of a float property in both
languages. For a float in GQL, write an exponent (`0.77e0`) or the suffix `F` (`0.77F`), or pass the value as a
parameter. See [Exact and Approximate Numbers](types.md#exact-and-approximate-numbers).

## Quantifiers

ISO GQL writes the repetition after the edge: `-[:KNOWS]->{1,3}` for Cypher's `-[:KNOWS*1..3]->`, and
`-[:KNOWS]->{1,}` for `-[:KNOWS*]->`.

=== "Cypher"

    ```cypher
    MATCH (a:Person {name: 'Alix'})-[:KNOWS*1..3]->(b)
    RETURN DISTINCT b.name
    ```

=== "GQL"

    ```sql
    MATCH (a:Person {name: 'Alix'})-[:KNOWS]->{1,3}(b)
    RETURN DISTINCT b.name
    ```

Grafeo's GQL also takes the Cypher form `-[:KNOWS*1..3]->`, but only the ISO form runs in other GQL engines.

## WITH, UNWIND and LET

ISO GQL has no `WITH`: bind a value with `LET`, and keep rows with `FILTER` or a `WHERE` after a `MATCH`. Cypher's
`UNWIND list AS x` is `FOR x IN list`.

=== "Cypher"

    ```cypher
    MATCH (n:Person)
    WITH n, n.age * 2 AS doubled
    RETURN n.name, doubled

    MATCH (p:Person)
    UNWIND p.hobbies AS h
    RETURN p.name, h
    ```

=== "GQL"

    ```sql
    MATCH (n:Person)
    LET doubled = n.age * 2
    RETURN n.name, doubled

    MATCH (p:Person)
    FOR h IN p.hobbies
    RETURN p.name, h
    ```

Grafeo's GQL also takes `WITH`, `UNWIND`, `MERGE` and list comprehensions (`[x IN list WHERE x > 1 | x * 2]`), so a
Cypher query that uses them often runs as it is. These are not ISO GQL, and other GQL engines may refuse them.
`EXISTS { ... }` is ISO GQL.

A GQL query cannot start with `WITH`: write a leading `WITH 'Alix' AS name` as `LET name = 'Alix'`.

## Map Projection

Cypher's map projection `n {.name, .age}` is a map literal in GQL, `{name: n.name, age: n.age}`. A missing property
is a null value in both.

=== "Cypher"

    ```cypher
    MATCH (n:Person)
    RETURN n {.name, .age} AS person
    ```

=== "GQL"

    ```sql
    MATCH (n:Person)
    RETURN {name: n.name, age: n.age} AS person
    ```

## Comments

In Grafeo's GQL a line comment starts with `--` and a space (`--` without one is an undirected edge), and a block
comment is `/* ... */`. openCypher writes a line comment with `//`, which Grafeo's GQL does not read as a comment:
change it to `--` and a space when you move a query.

=== "Cypher"

    ```cypher
    // People in Amsterdam
    MATCH (p:Person {city: 'Amsterdam'}) /* any age */
    RETURN p.name
    ```

=== "GQL"

    ```sql
    -- People in Amsterdam
    MATCH (p:Person {city: 'Amsterdam'}) /* any age */
    RETURN p.name
    ```

## Reserved Words as Names

GQL reserves more words than openCypher, among them `LIKE` and `EXISTS`, so they cannot stand as a variable or an
alias as they are. `LIKE` is a common one: a Cypher variable `like` must be renamed or quoted as `` `like` `` in
GQL. Rename such a name, or quote it in backticks:

=== "Cypher"

    ```cypher
    MATCH (:Message)<-[like:LIKES]-(p:Person)
    RETURN p.name, like.creationDate
    ```

=== "GQL"

    ```sql
    MATCH (:Message)<-[lk:LIKES]-(p:Person)
    RETURN p.name, lk.creationDate

    -- The same name, quoted
    MATCH (:Message)<-[`like`:LIKES]-(p:Person)
    RETURN p.name, `like`.creationDate

    -- A reserved word as an alias, in backticks
    MATCH (p:Person)
    RETURN count(p) AS `exists`
    ```

## `sum()` of No Values

`sum()` over no rows, or over only nulls, is 0 in Cypher, as openCypher defines it, and `NULL` in GQL, as in SQL:

=== "Cypher"

    ```cypher
    UNWIND [] AS x
    RETURN sum(x) AS total
    // 0
    ```

=== "GQL"

    ```sql
    UNWIND [] AS x
    RETURN sum(x) AS total
    -- NULL
    ```

## Zoned and Local Datetimes

GQL compares a zoned datetime with a local one by reading the local value at the session time zone (UTC when the
session sets none, see `SESSION SET TIME ZONE`). openCypher keeps them apart: `=` between them is false and `<` is
null. Two zoned datetimes compare by their instant in both languages.

=== "Cypher"

    ```cypher
    RETURN datetime('2024-03-15T10:00:00+01:00') = localdatetime('2024-03-15T09:00:00') AS eq
    // false
    ```

=== "GQL"

    ```sql
    RETURN zoned_datetime('2024-03-15T10:00:00+01:00') = local_datetime('2024-03-15T09:00:00') AS eq
    -- true: the local value is read at UTC
    ```
