---
title: Triple Patterns
description: Learn about SPARQL triple patterns for matching RDF data.
tags:
  - sparql
  - patterns
  - rdf
---

# Triple Patterns

Triple patterns are the fundamental building blocks of SPARQL queries. They match subject-predicate-object triples in RDF data.

## Basic Triple Pattern

A triple pattern consists of three components:

```sparql
?subject ?predicate ?object
```

Each component can be:
- A **variable** (starts with `?` or `$`)
- An **IRI** (in angle brackets or prefixed)
- A **literal** (for objects only)

## Variables

Variables bind to values in matched triples:

```sparql
# All three positions as variables
SELECT ?s ?p ?o
WHERE { ?s ?p ?o }

# Mix of variables and constants
SELECT ?name
WHERE { <http://example.org/alice> <http://xmlns.com/foaf/0.1/name> ?name }
```

## IRIs (Resources)

IRIs identify resources:

```sparql
# Full IRI
SELECT ?name
WHERE { <http://example.org/alice> <http://xmlns.com/foaf/0.1/name> ?name }

# Prefixed IRI
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX ex: <http://example.org/>

SELECT ?name
WHERE { ex:alice foaf:name ?name }
```

## Literals

Literals represent data values:

```sparql
# Plain literal
SELECT ?person
WHERE { ?person foaf:name "Alice" }

# Language-tagged literal
SELECT ?label
WHERE { ?x rdfs:label ?label FILTER(LANG(?label) = "en") }

# Typed literal
SELECT ?person
WHERE { ?person foaf:age "30"^^xsd:integer }
```

## OPTIONAL Patterns

Match patterns that may or may not exist:

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

SELECT ?name ?email
WHERE {
    ?person foaf:name ?name .
    OPTIONAL { ?person foaf:mbox ?email }
}
```

Results include people without email addresses (with `?email` unbound).

## UNION Patterns

Match alternative patterns:

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

SELECT ?name
WHERE {
    { ?person foaf:name ?name }
    UNION
    { ?person foaf:nick ?name }
}
```

## MINUS Patterns

Exclude matching patterns:

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

# Find people who don't know Bob
SELECT ?name
WHERE {
    ?person foaf:name ?name
    MINUS {
        ?person foaf:knows <http://example.org/bob>
    }
}
```

## Graph Patterns

Query specific named graphs:

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

# Query a specific graph
SELECT ?name
WHERE {
    GRAPH <http://example.org/friends> {
        ?person foaf:name ?name
    }
}

# Query with graph variable
SELECT ?g ?name
WHERE {
    GRAPH ?g {
        ?person foaf:name ?name
    }
}
```

## Nested Patterns

Combine patterns with grouping:

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

SELECT ?name ?friend
WHERE {
    ?person foaf:name ?name .
    {
        ?person foaf:knows ?f .
        ?f foaf:name ?friend
    }
}
```

## BIND

Create new bindings from expressions:

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

SELECT ?name ?upperName
WHERE {
    ?person foaf:name ?name
    BIND(UCASE(?name) AS ?upperName)
}
```

## VALUES

Provide inline data:

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

SELECT ?name ?email
WHERE {
    VALUES ?person { <http://example.org/alice> <http://example.org/bob> }
    ?person foaf:name ?name .
    OPTIONAL { ?person foaf:mbox ?email }
}
```

## Subqueries

Nest queries within patterns:

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

SELECT ?name ?maxAge
WHERE {
    ?person foaf:name ?name .
    {
        SELECT (MAX(?age) AS ?maxAge)
        WHERE {
            ?p foaf:age ?age
        }
    }
}
```

## Pattern Summary

| Pattern | Description |
|---------|-------------|
| `?s ?p ?o` | Basic triple pattern |
| `OPTIONAL { }` | Optional match |
| `{ } UNION { }` | Alternative patterns |
| `MINUS { }` | Negation |
| `GRAPH ?g { }` | Named graph |
| `BIND(expr AS ?var)` | Variable binding |
| `VALUES ?var { }` | Inline data |
| `{ SELECT ... }` | Subquery |
