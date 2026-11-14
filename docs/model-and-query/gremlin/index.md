---
title: Gremlin Query Language
description: Learn the Gremlin traversal language for Grafeo.
---

# Gremlin Query Language

Gremlin is a graph traversal language developed by Apache TinkerPop. Grafeo supports Gremlin as an optional query language via feature flag.

## Overview

Gremlin uses a functional, step-based approach to traverse and manipulate graph data. It's designed around the concept of traversing vertices and edges through a series of chained steps.

## Enabling Gremlin

Gremlin support is optional and requires a feature flag:

=== "Rust"

    ```bash
    cargo add grafeo-engine --features gremlin
    ```

=== "Python"

    ```bash
    uv add grafeo[gremlin]
    ```

## Quick Reference

| Operation | Syntax |
|-----------|--------|
| All vertices | `g.V()` |
| Vertex by ID | `g.V(id)` |
| Filter by label | `g.V().hasLabel('Person')` |
| Filter by property | `g.V().has('name', 'Alix')` |
| Outgoing edges | `g.V().out('KNOWS')` |
| Incoming edges | `g.V().in('KNOWS')` |
| Both directions | `g.V().both('KNOWS')` |
| Get properties | `g.V().values('name')` |
| Count results | `g.V().count()` |
| Limit results | `g.V().limit(10)` |

## Basic Examples

### Finding Vertices

```gremlin
// All vertices
g.V()

// Vertices with a specific label
g.V().hasLabel('Person')

// Vertex with specific property
g.V().has('name', 'Alix')

// Multiple conditions
g.V().hasLabel('Person').has('age', gt(25))
```

### Traversing Edges

```gremlin
// Friends of Alix
g.V().has('name', 'Alix').out('KNOWS')

// People who know Gus
g.V().has('name', 'Gus').in('KNOWS')

// Two-hop traversal
g.V().has('name', 'Alix').out('KNOWS').out('KNOWS')
```

### Repeated Traversals

```gremlin
// Fixed-depth traversal: exactly 3 hops out
g.V().has('name', 'Alix').repeat(out('KNOWS')).times(3)

// All-depths traversal: emit at every depth
g.V().has('name', 'Alix').repeat(out('KNOWS')).emit()
```

`repeat().times(n)` maps to variable-length expansion with a fixed depth. `repeat().emit()` returns intermediate results at every depth. Both work inside `union()`, `coalesce()` and other nested traversals.

!!! warning "Silently ignored steps"
    `path()`, `simplePath()`, and `loops()` are parsed but silently ignored at execution time: the query will succeed without error, but these steps will not produce correct results. The `until()` termination modifier is accepted in `repeat().until(predicate)` syntax, but the predicate itself is ignored; the traversal falls back to a default maximum depth of 32 hops. Until these steps are fully implemented, avoid relying on them for correctness.

### Getting Properties

```gremlin
// Get names of all people
g.V().hasLabel('Person').values('name')

// Get multiple properties
g.V().hasLabel('Person').valueMap('name', 'age')
```

### Aggregations

```gremlin
// Count all people
g.V().hasLabel('Person').count()

// Count friends
g.V().has('name', 'Alix').out('KNOWS').count()
```

## Python Usage

```python
import grafeo

db = grafeo.GrafeoDB()

# Create some data
db.execute("INSERT (:Person {name: 'Alix', age: 30})")
db.execute("INSERT (:Person {name: 'Gus', age: 25})")
db.execute("""
    MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'})
    INSERT (a)-[:KNOWS {since: 2020}]->(b)
""")

# Query with Gremlin
result = db.execute_gremlin("g.V().hasLabel('Person').values('name')")
for row in result:
    print(row)

# Traverse relationships
friends = db.execute_gremlin("g.V().has('name', 'Alix').out('KNOWS').values('name')")
```

## Rust Usage

```rust
use grafeo_engine::GrafeoDB;

let db = GrafeoDB::new_in_memory();

// Create data with GQL
db.execute("INSERT (:Person {name: 'Alix'})").unwrap();

// Query with Gremlin
let result = db.execute_gremlin("g.V().hasLabel('Person')").unwrap();
```

## Supported Steps

### Source Steps
- `g.V()` - Start traversal from vertices
- `g.V(id)` - Start from specific vertex

### Filter Steps
- `hasLabel(label)` - Filter by vertex label
- `has(key, value)` - Filter by property equality
- `has(key, predicate)` - Filter by property predicate

### Traversal Steps
- `out(label?)` - Traverse outgoing edges
- `in(label?)` - Traverse incoming edges
- `both(label?)` - Traverse both directions
- `outE(label?)` - Get outgoing edges
- `inE(label?)` - Get incoming edges

### Repetition Steps
- `repeat(traversal)` - Execute a traversal repeatedly
- `.times(n)` - Repeat exactly n times (fixed-depth expansion)
- `.emit()` - Emit results at every depth (all-depths expansion)

### Property Steps
- `values(key)` - Get property values
- `valueMap(keys...)` - Get multiple properties as map

### Terminal Steps
- `count()` - Count elements
- `limit(n)` - Limit results

## Learn More

<div class="grid cards" markdown>

-   **[Basic Traversals](basic-traversals.md)**

    ---

    g.V(), hasLabel, has and filtering.

-   **[Edge Traversals](edge-traversals.md)**

    ---

    out, in, both and relationship patterns.

-   **[Properties](properties.md)**

    ---

    values, valueMap and property access.

-   **[Aggregations](aggregations.md)**

    ---

    count, sum and grouping.

</div>
