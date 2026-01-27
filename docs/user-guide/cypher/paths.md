---
title: Path Queries
description: Variable-length paths and shortest path queries in Cypher.
tags:
  - cypher
  - paths
---

# Path Queries

Cypher supports variable-length paths for traversing the graph.

## Variable-Length Patterns

```cypher
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

```cypher
-- Capture the path
MATCH path = (a:Person)-[:KNOWS*]->(b:Person)
WHERE a.name = 'Alice' AND b.name = 'Dave'
RETURN path

-- Path length
MATCH path = (a:Person)-[:KNOWS*]->(b:Person)
WHERE a.name = 'Alice'
RETURN b.name, length(path) AS distance
ORDER BY distance
```

## Shortest Path

```cypher
-- Find shortest path
MATCH path = shortestPath((a:Person)-[:KNOWS*]-(b:Person))
WHERE a.name = 'Alice' AND b.name = 'Dave'
RETURN path, length(path)

-- All shortest paths
MATCH path = allShortestPaths((a:Person)-[:KNOWS*]-(b:Person))
WHERE a.name = 'Alice' AND b.name = 'Dave'
RETURN path
```

## Path Filtering

```cypher
-- Filter paths by node properties
MATCH path = (a:Person)-[:KNOWS*]->(b:Person)
WHERE a.name = 'Alice'
  AND all(n IN nodes(path) WHERE n.active = true)
RETURN path

-- Filter by relationship properties
MATCH path = (a:Person)-[:KNOWS*]->(b:Person)
WHERE a.name = 'Alice'
  AND all(r IN relationships(path) WHERE r.strength > 0.5)
RETURN path
```

## Path Functions

| Function | Description |
|----------|-------------|
| `nodes(path)` | List of nodes in path |
| `relationships(path)` | List of relationships in path |
| `length(path)` | Number of relationships in path |
