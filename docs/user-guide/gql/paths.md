---
title: Path Queries
description: Variable-length paths and shortest path queries in GQL.
tags:
  - gql
  - paths
---

# Path Queries

GQL supports variable-length paths for traversing the graph.

## Variable-Length Patterns

```sql
-- Any number of hops
MATCH (a:Person)-[:KNOWS*]->(b:Person)
RETURN a.name, b.name

-- Exactly 2 hops
MATCH (a:Person)-[:KNOWS*2]->(b:Person)
RETURN a.name, b.name

-- 1 to 3 hops
MATCH (a:Person)-[:KNOWS*1..3]->(b:Person)
RETURN a.name, b.name

-- Up to 5 hops
MATCH (a:Person)-[:KNOWS*..5]->(b:Person)
RETURN a.name, b.name

-- At least 2 hops
MATCH (a:Person)-[:KNOWS*2..]->(b:Person)
RETURN a.name, b.name
```

## Path Variables

```sql
-- Capture the path
MATCH path = (a:Person)-[:KNOWS*]->(b:Person)
WHERE a.name = 'Alix' AND b.name = 'Dave'
RETURN path

-- Path length
MATCH path = (a:Person)-[:KNOWS*]->(b:Person)
WHERE a.name = 'Alix'
RETURN b.name, length(path) AS distance
ORDER BY distance
```

## Shortest Path

```sql
-- Find shortest path
MATCH path = shortestPath((a:Person)-[:KNOWS*]-(b:Person))
WHERE a.name = 'Alix' AND b.name = 'Dave'
RETURN path, length(path)

-- All shortest paths
MATCH path = allShortestPaths((a:Person)-[:KNOWS*]-(b:Person))
WHERE a.name = 'Alix' AND b.name = 'Dave'
RETURN path
```

## Path Filtering

```sql
-- Filter paths by node properties
MATCH path = (a:Person)-[:KNOWS*]->(b:Person)
WHERE a.name = 'Alix'
  AND all(n IN nodes(path) WHERE n.active = true)
RETURN path

-- Filter by relationship properties
MATCH path = (a:Person)-[:KNOWS*]->(b:Person)
WHERE a.name = 'Alix'
  AND all(r IN relationships(path) WHERE r.strength > 0.5)
RETURN path
```

## Path Functions

| Function | Description |
|----------|-------------|
| `nodes(path)` | List of nodes in path |
| `relationships(path)` / `edges(path)` | List of edges in path |
| `length(path)` | Number of edges in path |
| `isAcyclic(path)` | True if no node appears more than once |
| `isSimple(path)` | True if no node repeats except first = last |
| `isTrail(path)` | True if no edge repeats |

## ISO Path Quantifiers

The ISO standard uses curly-brace syntax as an alternative to `*m..n`. See [Pattern Matching](patterns.md#iso-path-quantifiers) for the full comparison.

```sql
-- ISO: 2 to 4 hops
MATCH (a:Person)-[:KNOWS]{2,4}(b:Person)
RETURN a.name, b.name

-- ISO: exactly 3 hops
MATCH (a:Person)-[:KNOWS]{3}(b:Person)
RETURN a.name, b.name
```

## Path Modes

Path modes restrict which traversals are valid. Place the mode keyword before the pattern.

```sql
-- WALK (default): repeated nodes and edges allowed
MATCH path = WALK (a:Person)-[:KNOWS*]->(b:Person)
WHERE a.name = 'Alix'
RETURN b.name, length(path)

-- TRAIL: no edge can be visited more than once
MATCH path = TRAIL (a:Person)-[:KNOWS*]->(b:Person)
WHERE a.name = 'Alix'
RETURN b.name, length(path)

-- SIMPLE: no node visited more than once (except start = end)
MATCH path = SIMPLE (a:Person)-[:KNOWS*]->(b:Person)
WHERE a.name = 'Alix'
RETURN b.name, length(path)

-- ACYCLIC: strictly no repeated nodes
MATCH path = ACYCLIC (a:Person)-[:KNOWS*]->(b:Person)
WHERE a.name = 'Alix'
RETURN b.name, length(path)
```

Use path modes to control traversal behavior in cyclic graphs. `ACYCLIC` prevents infinite loops, while `TRAIL` allows revisiting nodes but not edges.

### Path Mode Inside Parenthesized Patterns (G049)

Path modes can also be placed inside a parenthesized quantified pattern, overriding any outer mode:

```sql
-- TRAIL mode inside quantified pattern
MATCH (TRAIL (a)-[:KNOWS]->(b)){1,3}
RETURN DISTINCT b.name

-- ACYCLIC inside, with outer WALK
MATCH WALK (ACYCLIC (a)-[:KNOWS]->(b)){2,5}
RETURN a.name, b.name
```

### WHERE Inside Parenthesized Patterns (G050)

A `WHERE` clause can filter within a parenthesized quantified pattern:

```sql
-- Only follow edges where the target meets a condition
MATCH ((a:Person)-[:KNOWS]->(b:Person) WHERE b.age > 30){1,3}
RETURN DISTINCT b.name

-- Combined with path mode
MATCH (TRAIL (a)-[e:KNOWS]->(b) WHERE e.since > 2020){1,4}
RETURN a.name, b.name
```

## Path Search Prefixes

Search prefixes control how many matching paths are returned. See [Pattern Matching](patterns.md#path-search-prefixes) for the complete list.

```sql
-- ANY SHORTEST: any one shortest path
MATCH path = ANY SHORTEST (a:Person)-[:KNOWS*]->(b:Person)
WHERE a.name = 'Alix' AND b.name = 'Dave'
RETURN path, length(path)

-- ALL SHORTEST: all paths of minimum length
MATCH path = ALL SHORTEST (a:Person)-[:KNOWS*]->(b:Person)
WHERE a.name = 'Alix' AND b.name = 'Dave'
RETURN path, length(path)

-- SHORTEST 3: the 3 shortest paths
MATCH path = SHORTEST 3 (a:Person)-[:KNOWS*]->(b:Person)
WHERE a.name = 'Alix' AND b.name = 'Dave'
RETURN path, length(path)
```

Shortest-path searches (these prefixes and `shortestPath`) only return paths that fit the edge's quantifier: `->+` or `-[:KNOWS*]->` is one or more hops, `->{1,3}` at most three, `->*` also pairs a node with itself (length 0), and an edge without a quantifier is a single hop. A pair of nodes without such a path has no row; use `OPTIONAL MATCH` to keep it with a null path.

## Path Predicate Functions

Test structural properties of a captured path:

```sql
MATCH path = (a:Person)-[:KNOWS*]->(b:Person)
WHERE a.name = 'Alix'
RETURN
    b.name,
    length(path) AS hops,
    isAcyclic(path) AS acyclic,
    isSimple(path) AS simple,
    isTrail(path) AS trail

-- Filter: only acyclic paths
MATCH path = (a:Person)-[:KNOWS*]->(b:Person)
WHERE a.name = 'Alix' AND isAcyclic(path)
RETURN b.name, length(path)
```

## Paths as Values

Paths are first-class values in GQL (GV55). A path variable can be returned, compared and passed to functions:

```sql
-- Return the path itself
MATCH p = (a:Person {name: 'Alix'})-[:KNOWS]->(b:Person)
RETURN p

-- Path equality: two paths are equal if they traverse the same nodes and edges
MATCH p1 = (a:Person)-[:KNOWS]->(b:Person),
      p2 = (c:Person)-[:KNOWS]->(d:Person)
WHERE a = c AND b = d
RETURN p1 = p2 AS same_path
```
