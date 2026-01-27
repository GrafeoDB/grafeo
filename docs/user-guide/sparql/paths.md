---
title: Property Paths
description: Learn about SPARQL property paths for traversing relationships.
tags:
  - sparql
  - paths
  - traversal
---

# Property Paths

Property paths allow you to match paths of arbitrary length through the graph using a concise syntax.

## Path Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `/` | Sequence | `foaf:knows/foaf:name` |
| `\|` | Alternative | `foaf:name\|foaf:nick` |
| `*` | Zero or more | `foaf:knows*` |
| `+` | One or more | `foaf:knows+` |
| `?` | Zero or one | `foaf:knows?` |
| `^` | Inverse | `^foaf:knows` |
| `!` | Negation | `!rdf:type` |
| `()` | Grouping | `(foaf:knows/foaf:name)` |

## Sequence Paths

Match a sequence of predicates:

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

# Friend's names (two-hop path)
SELECT ?name
WHERE {
    <http://example.org/alice> foaf:knows/foaf:name ?name
}

# Equivalent to:
SELECT ?name
WHERE {
    <http://example.org/alice> foaf:knows ?friend .
    ?friend foaf:name ?name
}
```

## Alternative Paths

Match any of several predicates:

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

# Name or nickname
SELECT ?name
WHERE {
    ?person foaf:name|foaf:nick ?name
}

# Multiple alternatives
SELECT ?contact
WHERE {
    ?person foaf:mbox|foaf:phone|foaf:homepage ?contact
}
```

## Zero or More (`*`)

Match a path of any length (including zero):

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

# All people reachable through knows (including self)
SELECT ?person
WHERE {
    <http://example.org/alice> foaf:knows* ?person
}
```

## One or More (`+`)

Match a path of at least one edge:

```sparql
PREFIX ex: <http://example.org/>

# All ancestors (at least one parent step)
SELECT ?ancestor
WHERE {
    ?person ex:parent+ ?ancestor
}

# All descendants
SELECT ?descendant
WHERE {
    ?ancestor ex:parent+ ?descendant
}
```

## Zero or One (`?`)

Match an optional single edge:

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

# Person or their direct friend
SELECT ?target
WHERE {
    <http://example.org/alice> foaf:knows? ?target
}
```

## Inverse Paths

Traverse edges in reverse direction:

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

# Who knows Alice?
SELECT ?person
WHERE {
    <http://example.org/alice> ^foaf:knows ?person
}

# Equivalent to:
SELECT ?person
WHERE {
    ?person foaf:knows <http://example.org/alice>
}
```

## Negated Property Sets

Match any predicate except specified ones:

```sparql
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

# All predicates except rdf:type
SELECT ?p ?o
WHERE {
    <http://example.org/alice> !rdf:type ?o
    BIND(?p AS !rdf:type)
}

# Multiple exclusions
SELECT ?p ?o
WHERE {
    ?s !(rdf:type|rdfs:label) ?o
}
```

## Combining Path Operators

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX ex: <http://example.org/>

# Friends or family of any distance
SELECT ?related
WHERE {
    ?person (foaf:knows|ex:family)+ ?related
}

# All names in the friend network
SELECT DISTINCT ?name
WHERE {
    <http://example.org/alice> foaf:knows+/foaf:name ?name
}

# Bidirectional knows relationship
SELECT ?person
WHERE {
    <http://example.org/alice> (foaf:knows|^foaf:knows)+ ?person
}
```

## Path Length Constraints

Use subqueries or FILTER for length constraints:

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

# Friends of friends (exactly 2 hops)
SELECT ?fof
WHERE {
    <http://example.org/alice> foaf:knows/foaf:knows ?fof
    FILTER(?fof != <http://example.org/alice>)
}
```

## Shortest Path

SPARQL 1.1 doesn't have built-in shortest path, but you can simulate it:

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

# Find if path exists (any length)
ASK {
    <http://example.org/alice> foaf:knows+ <http://example.org/charlie>
}
```

## RDF Type Shorthand

The `a` keyword is shorthand for `rdf:type`:

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

# Using 'a' for rdf:type
SELECT ?person
WHERE {
    ?person a foaf:Person
}

# Equivalent to:
SELECT ?person
WHERE {
    ?person rdf:type foaf:Person
}
```

## Practical Examples

### Social Network Reachability

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

# Find all people within 3 hops
SELECT DISTINCT ?person ?name
WHERE {
    <http://example.org/alice> foaf:knows{1,3} ?person .
    ?person foaf:name ?name
}
```

### Hierarchical Data

```sparql
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

# All broader concepts (category hierarchy)
SELECT ?broader
WHERE {
    <http://example.org/concept123> skos:broader+ ?broader
}

# All narrower concepts
SELECT ?narrower
WHERE {
    <http://example.org/concept123> skos:narrower+ ?narrower
}
```

### Ontology Traversal

```sparql
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

# All superclasses
SELECT ?superclass
WHERE {
    <http://example.org/Dog> rdfs:subClassOf+ ?superclass
}

# All instances of a class and its subclasses
SELECT ?instance
WHERE {
    ?class rdfs:subClassOf* <http://example.org/Animal> .
    ?instance a ?class
}
```
