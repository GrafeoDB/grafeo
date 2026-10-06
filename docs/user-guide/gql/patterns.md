---
title: Pattern Matching
description: Advanced pattern matching in GQL.
tags:
  - gql
  - patterns
---

# Pattern Matching

Pattern matching is the core of GQL. This guide covers node and edge patterns in detail.

## Node Patterns

```sql
-- Any node
(n)

-- Node with label
(p:Person)

-- Node with multiple labels
(e:Person:Employee)

-- Node with properties
(p:Person {name: 'Alix'})

-- Anonymous node (no variable)
(:Person)
```

## Edge Patterns

```sql
-- Outgoing edge
(a)-[:KNOWS]->(b)

-- Incoming edge
(a)<-[:KNOWS]-(b)

-- Either direction
(a)-[:KNOWS]-(b)

-- Any edge type
(a)-[r]->(b)

-- Edge with properties
(a)-[:KNOWS {since: 2020}]->(b)
```

## Complex Patterns

```sql
-- Chain of relationships
MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(c)
RETURN a.name, b.name, c.name

-- Multiple patterns
MATCH (a:Person)-[:KNOWS]->(b), (a)-[:WORKS_AT]->(c)
RETURN a.name, b.name, c.name

-- Triangle pattern
MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(c)-[:KNOWS]->(a)
RETURN a.name, b.name, c.name
```

## Path Alternation

Use `|` for set alternation (dedup) or `|+|` for multiset alternation (preserves duplicates):

```sql
-- Set alternation: match KNOWS or WORKS_WITH edges (dedup)
MATCH ((a)-[:KNOWS]->(b) | (a)-[:WORKS_WITH]->(b))
RETURN a.name, b.name

-- Multiset alternation: preserve duplicates across alternatives
MATCH ((a)-[:KNOWS]->(b) |+| (a)-[:KNOWS]->(b))
RETURN a.name, b.name
```

## Multiple Relationship Types

```sql
-- Match any of multiple types
MATCH (a)-[:KNOWS|:WORKS_WITH]->(b)
RETURN a.name, b.name

-- Match with type variable
MATCH (a)-[r:KNOWS|:WORKS_WITH]->(b)
RETURN a.name, type(r), b.name
```

## Optional Patterns

```sql
-- Return results even if pattern doesn't match
MATCH (p:Person)
OPTIONAL MATCH (p)-[:HAS_PET]->(pet)
RETURN p.name, pet.name
```

## Element WHERE Clauses

Filter directly inside a pattern element, without a separate `WHERE` clause:

```sql
-- Node-level WHERE
MATCH (p:Person WHERE p.age > 30)
RETURN p.name

-- Edge-level WHERE
MATCH (a)-[r:KNOWS WHERE r.since >= 2020]->(b)
RETURN a.name, b.name, r.since
```

## Label Expressions

GQL supports rich label expressions using the `IS` keyword with boolean operators.

```sql
-- Single label
MATCH (n IS Person) RETURN n.name

-- Disjunction: match Person OR Company
MATCH (n IS Person | Company) RETURN n.name

-- Conjunction: match nodes with BOTH labels
MATCH (n IS Person & Employee) RETURN n.name

-- Negation: match anything except Inactive
MATCH (n IS !Inactive) RETURN n.name

-- Wildcard: match any label
MATCH (n IS %) RETURN n.name

-- Parenthesized combinations
MATCH (n IS (Person | Company) & !Inactive) RETURN n.name
```

## Simplified Path Patterns

The ISO standard provides a shorthand syntax using `/` delimiters instead of brackets:

```sql
-- Outgoing: -/:Label/-> is equivalent to -[:Label]->
MATCH (a:Person)-/:KNOWS/->(b:Person)
RETURN b.name

-- Incoming: <-/:Label/- is equivalent to <-[:Label]-
MATCH (b:Person)<-/:KNOWS/-(a:Person)
RETURN a.name

-- Undirected: -/:Label/- is equivalent to -[:Label]-
MATCH (a:Person)-/:KNOWS/-(b:Person)
RETURN b.name

-- Multiple label alternatives: -/:L1|L2/->
MATCH (a:Person)-/:KNOWS|WORKS_WITH/->(b)
RETURN b.name

-- Tilde form: ~/:Label/~
MATCH (a:Person)~/:KNOWS/~(b:Person)
RETURN b.name
```

## ISO Tilde Syntax (Undirected Edges)

The ISO standard uses tilde (`~`) for undirected edges, as an alternative to the Cypher-style `-[]-`:

```sql
-- ISO undirected edge
MATCH (a)~[r:KNOWS]~(b)
RETURN a.name, b.name

-- Equivalent Cypher-style
MATCH (a)-[r:KNOWS]-(b)
RETURN a.name, b.name
```

## ISO Path Quantifiers

The ISO standard uses curly-brace syntax `{m,n}` for path quantifiers, as an alternative to the Cypher-style `*m..n`:

| ISO Syntax | Cypher Syntax | Meaning |
|------------|---------------|---------|
| `{2,5}` | `*2..5` | 2 to 5 hops |
| `{3}` | `*3` | Exactly 3 hops |
| `{2,}` | `*2..` | At least 2 hops |
| `{,5}` | `*..5` | At most 5 hops |

```sql
-- ISO: exactly 2 hops
MATCH (a:Person)-[:KNOWS]{2}(b:Person)
RETURN a.name, b.name

-- ISO: 1 to 3 hops
MATCH (a:Person)-[:KNOWS]{1,3}(b:Person)
RETURN a.name, b.name

-- Cypher equivalent: 1 to 3 hops
MATCH (a:Person)-[:KNOWS*1..3]->(b:Person)
RETURN a.name, b.name
```

## Questioned Edge

The questioned edge `->?` matches 0 or 1 hops, making the relationship optional. If the edge exists, it is matched; if not, the target node is `null`.

```sql
-- Optional relationship (0 or 1 hops)
MATCH (p:Person)-[:MANAGES]->?(team:Team)
RETURN p.name, team.name
```

## Path Search Prefixes

Path search prefixes control how many matching paths are returned.

```sql
-- ANY: return any single matching path
MATCH ANY (a:Person)-[:KNOWS*]->(b:Person)
WHERE a.name = 'Alix' AND b.name = 'Dave'
RETURN a, b

-- ANY k: return up to k paths
MATCH ANY 3 (a:Person)-[:KNOWS*]->(b:Person)
WHERE a.name = 'Alix'
RETURN b.name

-- ALL SHORTEST: all paths of minimum length
MATCH ALL SHORTEST (a:Person)-[:KNOWS*]->(b:Person)
WHERE a.name = 'Alix' AND b.name = 'Dave'
RETURN a, b

-- ANY SHORTEST: any one shortest path
MATCH ANY SHORTEST (a:Person)-[:KNOWS*]->(b:Person)
WHERE a.name = 'Alix' AND b.name = 'Dave'
RETURN a, b

-- SHORTEST k: the k shortest paths
MATCH SHORTEST 3 (a:Person)-[:KNOWS*]->(b:Person)
WHERE a.name = 'Alix' AND b.name = 'Dave'
RETURN a, b

-- SHORTEST k GROUPS: k groups of equal-length shortest paths
MATCH SHORTEST 2 GROUPS (a:Person)-[:KNOWS*]->(b:Person)
WHERE a.name = 'Alix' AND b.name = 'Dave'
RETURN a, b
```

## Path Modes

Path modes restrict which paths are valid during traversal. Place the mode keyword before the pattern.

| Mode | Rule |
|------|------|
| `WALK` | Default. Repeated nodes and edges allowed |
| `TRAIL` | No repeated edges |
| `SIMPLE` | No repeated nodes (except start = end) |
| `ACYCLIC` | No repeated nodes at all |

```sql
-- WALK (default): allow cycles
MATCH WALK (a:Person)-[:KNOWS*]->(b:Person)
WHERE a.name = 'Alix'
RETURN b.name

-- TRAIL: each edge visited at most once
MATCH TRAIL (a:Person)-[:KNOWS*]->(b:Person)
WHERE a.name = 'Alix'
RETURN b.name

-- SIMPLE: each node visited at most once (except endpoints)
MATCH SIMPLE (a:Person)-[:KNOWS*]->(b:Person)
WHERE a.name = 'Alix'
RETURN b.name

-- ACYCLIC: strictly no repeated nodes
MATCH ACYCLIC (a:Person)-[:KNOWS*]->(b:Person)
WHERE a.name = 'Alix'
RETURN b.name
```

## Match Modes

Match modes control uniqueness across multiple patterns in the same `MATCH`.

```sql
-- DIFFERENT EDGES: no edge can appear in more than one pattern binding
MATCH DIFFERENT EDGES
    (a)-[r1:KNOWS]->(b),
    (c)-[r2:KNOWS]->(d)
RETURN a.name, b.name, c.name, d.name

-- REPEATABLE ELEMENTS: relax the default uniqueness constraint
MATCH REPEATABLE ELEMENTS
    (a)-[:KNOWS]->(b),
    (a)-[:WORKS_WITH]->(b)
RETURN a.name, b.name
```
