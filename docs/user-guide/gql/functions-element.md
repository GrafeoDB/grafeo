---
title: Element and Path Functions
description: GQL functions for inspecting graph elements, paths and lists in Grafeo.
tags:
  - gql
  - functions
  - elements
  - paths
---

# Element, Path and List Functions

## Element Functions

### Identity

```sql
-- Internal numeric ID
MATCH (p:Person {name: 'Alix'})
RETURN id(p)              -- e.g., 42

-- ISO element identity (string format)
MATCH (p:Person {name: 'Alix'})
RETURN element_id(p)      -- e.g., 'n:42'
```

### Labels and Types

```sql
-- Node labels (returns a list)
MATCH (n)
RETURN n.name, labels(n)
-- e.g., 'Alix', ['Person', 'Employee']

-- Edge type (returns a string)
MATCH ()-[r]->()
RETURN type(r)
-- e.g., 'KNOWS'

-- Check if a node has a specific label
MATCH (n)
WHERE hasLabel(n, 'Person')
RETURN n.name
```

### Properties

```sql
-- List all property keys
MATCH (p:Person {name: 'Alix', age: 30})
RETURN keys(p)
-- ['name', 'age']

-- Get all properties as a map
MATCH (p:Person {name: 'Alix'})
RETURN properties(p)
-- {name: 'Alix', age: 30}

-- Practical: inspect unknown data
MATCH (n)
RETURN labels(n), keys(n)
LIMIT 10
```

## Path Functions

### Path Decomposition

```sql
-- Capture a path variable
MATCH path = (a:Person)-[:KNOWS*]->(b:Person)
WHERE a.name = 'Alix' AND b.name = 'Dave'

-- Number of edges in the path
RETURN length(path)       -- e.g., 3

-- List of nodes in the path
RETURN nodes(path)        -- [Alix, Gus, Harm, Dave]

-- List of edges in the path
RETURN edges(path)
RETURN relationships(path) -- alias for edges()
```

### Path Predicates

Test structural properties of a path:

```sql
MATCH path = (a:Person)-[:KNOWS*]->(b:Person)
WHERE a.name = 'Alix'

-- No repeated nodes at all
RETURN isAcyclic(path)

-- No repeated nodes except first = last
RETURN isSimple(path)

-- No repeated edges
RETURN isTrail(path)
```

```sql
-- Practical: find only acyclic paths
MATCH path = (a:Person)-[:KNOWS*]->(b:Person)
WHERE a.name = 'Alix' AND isAcyclic(path)
RETURN b.name, length(path)
```

## List Functions

### Access

```sql
-- First element
RETURN head([1, 2, 3])      -- 1

-- All elements except the first
RETURN tail([1, 2, 3])      -- [2, 3]

-- Last element
RETURN last([1, 2, 3])      -- 3

-- Index access (0-based)
RETURN [10, 20, 30][1]      -- 20
```

### Size and Length

```sql
-- Number of elements
RETURN size([1, 2, 3])       -- 3
RETURN length([1, 2, 3])     -- 3 (alias)
```

### Transformation

```sql
-- Reverse a list
RETURN reverse([1, 2, 3])   -- [3, 2, 1]
```

### Range

Generate a list of integers:

```sql
-- range(start, end) -- inclusive on both ends
RETURN range(1, 5)           -- [1, 2, 3, 4, 5]

-- range(start, end, step)
RETURN range(0, 10, 2)       -- [0, 2, 4, 6, 8, 10]
RETURN range(10, 0, -2)      -- [10, 8, 6, 4, 2, 0]
```

```sql
-- Practical: generate test data
UNWIND range(1, 100) AS i
INSERT (:TestNode {index: i, value: rand()})
```

## Type Conversion Functions

| Function | Description |
|----------|-------------|
| `toInteger(expr)` / `toInt(expr)` | Convert to integer |
| `toFloat(expr)` | Convert to float |
| `toString(expr)` | Convert to string |
| `toBoolean(expr)` | Convert to boolean |
| `toList(expr)` | Wrap scalar in a list |

```sql
RETURN toInteger('42')      -- 42
RETURN toFloat('3.14')      -- 3.14
RETURN toString(42)         -- '42'
RETURN toBoolean('true')    -- true
RETURN toList(42)           -- [42]
RETURN toList([1, 2])       -- [1, 2] (already a list)
```

See also [Temporal Functions](functions-temporal.md) for `toDate()`, `toTime()`, `toDatetime()`, `toDuration()`, `toZonedDatetime()` and `toZonedTime()`.
