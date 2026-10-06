---
title: Expressions
description: GQL expressions, conditional logic, subqueries and list operations.
tags:
  - gql
  - expressions
---

# Expressions

GQL supports a rich expression language for conditional logic, type conversion, subqueries and list operations.

## CASE Expressions

### Searched CASE

Evaluate conditions in order and return the first matching result:

```sql
MATCH (p:Person)
RETURN p.name,
    CASE
        WHEN p.age < 18 THEN 'minor'
        WHEN p.age < 65 THEN 'adult'
        ELSE 'senior'
    END AS category
```

### Simple CASE

Compare an expression against specific values:

```sql
MATCH (p:Person)
RETURN p.name,
    CASE p.status
        WHEN 'A' THEN 'Active'
        WHEN 'I' THEN 'Inactive'
        WHEN 'P' THEN 'Pending'
        ELSE 'Unknown'
    END AS status_label
```

## NULLIF

Returns `null` if both arguments are equal, otherwise returns the first argument. Useful for avoiding division by zero:

```sql
MATCH (d:Department)
RETURN d.name,
    d.budget / NULLIF(d.headcount, 0) AS per_capita
```

## COALESCE

Returns the first non-null value from the argument list:

```sql
-- Display name fallback chain
MATCH (p:Person)
RETURN COALESCE(p.nickname, p.firstName, p.email) AS display_name
```

## CAST

Convert a value to a different type with `CAST(expr AS type)`:

```sql
-- Numeric conversions
RETURN CAST('42' AS INT)        -- 42
RETURN CAST(42 AS FLOAT)        -- 42.0
RETURN CAST(42 AS STRING)       -- '42'
RETURN CAST('true' AS BOOLEAN)  -- true

-- Temporal conversions
RETURN CAST('2024-01-15' AS DATE)
RETURN CAST('14:30:00' AS TIME)
RETURN CAST('2024-01-15T14:30:00Z' AS DATETIME)
RETURN CAST('P1Y2M' AS DURATION)

-- Zoned temporal conversions
RETURN CAST('2024-01-15T14:30:00+05:30' AS ZONED DATETIME)
RETURN CAST('14:30:00+05:30' AS ZONED TIME)

-- Wrap a scalar in a list
RETURN CAST(42 AS LIST)         -- [42]
```

Supported target types: `INT`/`INTEGER`, `FLOAT`, `STRING`, `BOOLEAN`, `DATE`, `TIME`, `DATETIME`, `DURATION`, `ZONED DATETIME`, `ZONED TIME`, `LIST`.

## Dynamic Parameters

Use `$param` placeholders in queries and provide values from the host language:

```sql
-- Parameterized query
MATCH (p:Person)
WHERE p.name = $name AND p.age > $min_age
RETURN p
```

```python
# Python
result = db.execute(
    "MATCH (p:Person) WHERE p.name = $name RETURN p",
    {"name": "Alix"}
)
```

```javascript
// Node.js
const result = await db.execute(
    "MATCH (p:Person) WHERE p.name = $name RETURN p",
    { name: "Alix" }
);
```

## LET ... IN ... END

Bind local variables within an expression. The variables are scoped to the expression only.

```sql
MATCH (p:Person)
RETURN p.name,
    LET base = p.salary,
        bonus = base * 0.1
    IN base + bonus
    END AS total_compensation
```

## List Comprehensions

Transform and filter lists in a single expression:

```sql
-- Filter: keep only even numbers
RETURN [x IN [1, 2, 3, 4, 5] WHERE x % 2 = 0] AS evens
-- [2, 4]

-- Transform: double each value
RETURN [x IN [1, 2, 3] | x * 2] AS doubled
-- [2, 4, 6]

-- Filter and transform
RETURN [x IN [1, 2, 3, 4, 5] WHERE x > 2 | x * 10] AS result
-- [30, 40, 50]

-- With graph data
MATCH (p:Person)-[:KNOWS]->(friend)
RETURN p.name,
    [f IN collect(friend) WHERE f.age > 30 | f.name] AS older_friends
```

## List Predicates

Test whether list elements satisfy a condition:

```sql
-- all(): every element must match
RETURN all(x IN [2, 4, 6] WHERE x % 2 = 0)  -- true

-- any(): at least one element must match
RETURN any(x IN [1, 2, 3] WHERE x > 2)  -- true

-- none(): no element must match
RETURN none(x IN [1, 2, 3] WHERE x > 5)  -- true

-- single(): exactly one element must match
RETURN single(x IN [1, 2, 3] WHERE x = 2)  -- true

-- Practical example: check path properties
MATCH path = (a:Person)-[:KNOWS*]->(b:Person)
WHERE a.name = 'Alix'
    AND all(n IN nodes(path) WHERE n.active = true)
RETURN path
```

## reduce()

Fold a list into a single value using an accumulator:

```sql
-- Sum a list
RETURN reduce(acc = 0, x IN [1, 2, 3, 4, 5] | acc + x) AS total
-- 15

-- Build a string
RETURN reduce(s = '', name IN ['Alix', 'Gus', 'Harm'] | s + name + ', ') AS names

-- Compute from graph data
MATCH (p:Person)
WITH collect(p.salary) AS salaries
RETURN reduce(total = 0, s IN salaries | total + s) AS payroll
```

## Subquery Expressions

### EXISTS

Check whether a pattern or subquery has any results:

```sql
-- Pattern exists
MATCH (p:Person)
WHERE EXISTS { MATCH (p)-[:MANAGES]->(:Team) }
RETURN p.name AS manager

-- Subquery with filtering
MATCH (p:Person)
WHERE EXISTS {
    MATCH (p)-[:ORDERED]->(o:Order)
    WHERE o.total > 1000
}
RETURN p.name AS high_value_customer
```

### COUNT

Count the number of results from a subquery:

```sql
MATCH (p:Person)
WHERE COUNT { MATCH (p)-[:KNOWS]->() } > 5
RETURN p.name AS popular_person
```

### VALUE

Return a single scalar value from a subquery:

```sql
MATCH (p:Person)
RETURN p.name,
    VALUE {
        MATCH (p)-[:KNOWS]->(friend)
        RETURN count(friend)
    } AS friend_count
```

## Index Access

Access list elements by index (0-based):

```sql
RETURN [10, 20, 30, 40][0]   -- 10
RETURN [10, 20, 30, 40][2]   -- 30

-- With graph data
MATCH (p:Person)
RETURN p.tags[0] AS first_tag
```

## Map Literals

Create and access map (dictionary) values:

```sql
-- Map literal
RETURN {name: 'Alix', age: 30, active: true} AS person

-- Access map properties
WITH {name: 'Alix', age: 30} AS person
RETURN person.name, person.age

-- Maps in mutations
MATCH (p:Person {name: 'Alix'})
SET p += {city: 'Amsterdam', role: 'engineer'}
```

A property can hold a map. Dotted access reads its keys, also nested, the same as a subscript; a missing key, or a value that is not a map, gives null. A column without an alias is named as written:

```sql
INSERT (:Doc {id: 'd1', meta: {route: 'directory', score: 0.5, source: {kind: 'file'}}})

MATCH (d:Doc)
WHERE d.meta.route = 'directory'
RETURN d.meta.score, d.meta.source.kind, d.meta['route']
-- columns: d.meta.score, d.meta.source.kind, d.meta['route']
```

Dotted access needs a map value: a property, a variable, a parameter, a map literal or `properties(n)`. On a node or edge that an expression returns, such as `startNode(r).name`, it is an error; match the node with a variable and read its property instead.

## SESSION_USER

Returns the current session user name. In embedded databases, this returns `'default'`.

```sql
RETURN SESSION_USER
```

## Introspection Functions

Query session state and database metadata:

```sql
-- Current schema name
RETURN CURRENT_SCHEMA

-- Current graph name
RETURN CURRENT_GRAPH

-- Database metadata (version, node count, edge count)
RETURN info()

-- Schema information (node types, edge types, indexes, constraints)
RETURN schema()
```

## String Concatenation

The `||` operator concatenates strings:

```sql
MATCH (p:Person)
RETURN p.firstName || ' ' || p.lastName AS fullName
```

## LIKE Pattern Matching

Match strings against patterns using `%` (any characters) and `_` (single character):

```sql
MATCH (p:Person)
WHERE p.name LIKE 'A%'
RETURN p.name

MATCH (p:Person)
WHERE p.email LIKE '%@example.com'
RETURN p.name, p.email
```

## IN Operator

Test membership in a list:

```sql
MATCH (p:Person)
WHERE p.city IN ['Amsterdam', 'Berlin', 'Paris']
RETURN p.name, p.city
```
