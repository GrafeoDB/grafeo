---
title: Basic Traversals
description: Learn basic Gremlin traversals with g.V(), filtering and vertex selection.
tags:
  - gremlin
  - traversals
---

# Basic Traversals

This guide covers the fundamentals of traversing graphs with Gremlin in Grafeo.

## Starting a Traversal

All Gremlin traversals start with the graph reference `g`:

```gremlin
// All vertices
g.V()

// All edges
g.E()

// Vertex by ID
g.V(1)

// Multiple vertices by ID
g.V(1, 2, 3)
```

## Filtering by Label

Use `hasLabel()` to filter vertices or edges by their label:

```gremlin
// All Person vertices
g.V().hasLabel('Person')

// Multiple labels
g.V().hasLabel('Person', 'Company')
```

## Filtering by Property

Use `has()` to filter by property values:

```gremlin
// Property equals value
g.V().has('name', 'Alix')

// Property exists (any value)
g.V().has('email')

// Label + property shorthand
g.V().has('Person', 'name', 'Alix')
```

## Predicate Filters

Use `P.*` predicates for comparison operators:

```gremlin
// Greater than
g.V().has('age', P.gt(25))

// Greater than or equal
g.V().has('age', P.gte(25))

// Less than
g.V().has('age', P.lt(40))

// Less than or equal
g.V().has('age', P.lte(40))

// Not equal
g.V().has('status', P.neq('inactive'))

// Between (inclusive start, exclusive end)
g.V().has('age', P.between(25, 40))

// Within a set of values
g.V().has('status', P.within('active', 'pending'))

// Not within a set
g.V().has('role', P.without('admin', 'superuser'))
```

## String Predicates

```gremlin
// Contains substring
g.V().has('name', P.containing('Ali'))

// Starts with
g.V().has('name', P.startingWith('A'))

// Ends with
g.V().has('email', P.endingWith('@company.com'))
```

## Negation Filters

```gremlin
// Property does NOT exist
g.V().hasNot('deleted')

// Filter by ID
g.V().hasId(42)

// Filter by multiple IDs
g.V().hasId(1, 2, 3)
```

## Deduplication

```gremlin
// Remove duplicate results
g.V().out('KNOWS').dedup()
```

## Limiting and Skipping

```gremlin
// First 10 results
g.V().limit(10)

// Skip first 5
g.V().skip(5)

// Range (skip 5, take 10)
g.V().range(5, 15)
```

## Python Example

```python
import grafeo

db = grafeo.GrafeoDB()

# Create data
db.execute("INSERT (:Person {name: 'Alix', age: 30})")
db.execute("INSERT (:Person {name: 'Gus', age: 25})")

# Basic traversal
result = db.execute_gremlin("g.V().hasLabel('Person')")
for row in result:
    print(row)

# Filter by property
result = db.execute_gremlin("g.V().has('name', 'Alix')")
```

## Supported Predicates

| Predicate | Description |
|-----------|-------------|
| `P.eq(value)` | Equal to |
| `P.neq(value)` | Not equal to |
| `P.gt(value)` | Greater than |
| `P.gte(value)` | Greater than or equal |
| `P.lt(value)` | Less than |
| `P.lte(value)` | Less than or equal |
| `P.within(values...)` | In a set of values |
| `P.without(values...)` | Not in a set of values |
| `P.between(start, end)` | Between (inclusive start, exclusive end) |
| `P.containing(str)` | Contains substring |
| `P.startingWith(str)` | Starts with prefix |
| `P.endingWith(str)` | Ends with suffix |
