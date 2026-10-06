---
title: Edge Traversals
description: Traverse edges and relationships with out, in, both and edge steps.
tags:
  - gremlin
  - edges
  - traversals
---

# Edge Traversals

This guide covers navigating relationships between vertices using Gremlin traversal steps.

## Vertex-to-Vertex Steps

These steps traverse edges and return the adjacent vertices:

```gremlin
// Outgoing neighbors (follow outgoing edges)
g.V().has('name', 'Alix').out('KNOWS')

// Incoming neighbors (follow incoming edges)
g.V().has('name', 'Gus').in('KNOWS')

// Both directions
g.V().has('name', 'Alix').both('KNOWS')
```

### Without Edge Label

Omit the label to traverse all edge types:

```gremlin
// All outgoing neighbors regardless of edge type
g.V().has('name', 'Alix').out()

// All incoming neighbors
g.V().has('name', 'Alix').in()

// All neighbors in both directions
g.V().has('name', 'Alix').both()
```

## Vertex-to-Edge Steps

These steps return the edge elements themselves, not the adjacent vertices:

```gremlin
// Outgoing edges
g.V().has('name', 'Alix').outE('KNOWS')

// Incoming edges
g.V().has('name', 'Gus').inE('KNOWS')

// Edges in both directions
g.V().has('name', 'Alix').bothE('KNOWS')
```

### Accessing Edge Properties

Once edges are obtained, their properties can be accessed:

```gremlin
// Get the 'since' property of outgoing KNOWS edges
g.V().has('name', 'Alix').outE('KNOWS').values('since')
```

## Edge-to-Vertex Steps

When traversing edges, use these steps to reach the connected vertices:

```gremlin
// Source vertex of an edge
g.V().has('name', 'Alix').outE('KNOWS').outV()

// Target vertex of an edge
g.V().has('name', 'Alix').outE('KNOWS').inV()

// Both endpoints
g.V().has('name', 'Alix').outE('KNOWS').bothV()

// The "other" vertex (not the one you came from)
g.V().has('name', 'Alix').outE('KNOWS').otherV()
```

## Chaining Traversals

Chain multiple steps for multi-hop traversals:

```gremlin
// Friends of friends
g.V().has('name', 'Alix').out('KNOWS').out('KNOWS')

// Friends of friends (unique)
g.V().has('name', 'Alix').out('KNOWS').out('KNOWS').dedup()

// People who work at companies Alix's friends work at
g.V().has('name', 'Alix')
    .out('KNOWS')
    .out('WORKS_AT')
    .in('WORKS_AT')
    .dedup()
```

## Filtering During Traversal

Combine traversal with filter steps:

```gremlin
// Friends of Alix who are over 30
g.V().has('name', 'Alix').out('KNOWS').has('age', P.gt(30))

// Outgoing KNOWS edges created after 2020
g.V().has('name', 'Alix').outE('KNOWS').has('since', P.gt(2020)).inV()
```

## Python Example

```python
import grafeo

db = grafeo.GrafeoDB()

# Create a social graph
db.execute("INSERT (:Person {name: 'Alix', age: 30})")
db.execute("INSERT (:Person {name: 'Gus', age: 25})")
db.execute("INSERT (:Person {name: 'Vincent', age: 35})")
db.execute("""
    MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'})
    INSERT (a)-[:KNOWS {since: 2020}]->(b)
""")
db.execute("""
    MATCH (b:Person {name: 'Gus'}), (c:Person {name: 'Vincent'})
    INSERT (b)-[:KNOWS {since: 2021}]->(c)
""")

# Direct friends
friends = db.execute_gremlin("g.V().has('name', 'Alix').out('KNOWS').values('name')")
for row in friends:
    print(row)  # Gus

# Friends of friends
fof = db.execute_gremlin(
    "g.V().has('name', 'Alix').out('KNOWS').out('KNOWS').values('name')"
)
for row in fof:
    print(row)  # Vincent
```

## Step Reference

| Step | From | Returns | Description |
|------|------|---------|-------------|
| `out(label?)` | Vertex | Vertices | Outgoing adjacent vertices |
| `in(label?)` | Vertex | Vertices | Incoming adjacent vertices |
| `both(label?)` | Vertex | Vertices | Adjacent vertices in both directions |
| `outE(label?)` | Vertex | Edges | Outgoing edges |
| `inE(label?)` | Vertex | Edges | Incoming edges |
| `bothE(label?)` | Vertex | Edges | Edges in both directions |
| `outV()` | Edge | Vertex | Source vertex of edge |
| `inV()` | Edge | Vertex | Target vertex of edge |
| `bothV()` | Edge | Vertices | Both endpoints of edge |
| `otherV()` | Edge | Vertex | The opposite endpoint |
