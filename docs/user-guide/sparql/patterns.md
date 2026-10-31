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
WHERE { <http://example.org/alix> <http://xmlns.com/foaf/0.1/name> ?name }
```

## IRIs (Resources)

IRIs identify resources:

```sparql
# Full IRI
SELECT ?name
WHERE { <http://example.org/alix> <http://xmlns.com/foaf/0.1/name> ?name }

# Prefixed IRI
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX ex: <http://example.org/>

SELECT ?name
WHERE { ex:alix foaf:name ?name }
```

## Literals

Literals represent data values:

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

# Plain literal
SELECT ?person
WHERE { ?person foaf:name "Alix" }

# Language-tagged literal
SELECT ?label
WHERE { ?x rdfs:label ?label FILTER(LANG(?label) = "en") }

# Typed literal
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
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

A `FILTER` at the top of the `OPTIONAL` group decides which matches extend a row, and reads the row and the match
together, as SPARQL 1.1 defines it: in `OPTIONAL { ?person ex:age ?age FILTER(?age > ?min) }`, `?min` can come from
before the `OPTIONAL`. A `FILTER` in a group nested inside the `OPTIONAL` reads that group alone. A group that opens
with `OPTIONAL` keeps one empty row when nothing matches.

A variable that an `OPTIONAL`, or one branch of a `UNION`, leaves unbound joins with any value of a later pattern
(SPARQL 1.1 compatible solutions): in `?s :name ?n OPTIONAL { ?s :city ?c } ?b :label ?c`, a person without a city
is joined with every label.

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

# Find people who don't know Gus
SELECT ?name
WHERE {
    ?person foaf:name ?name
    MINUS {
        ?person foaf:knows <http://example.org/gus>
    }
}
```

A group that holds only a `MINUS`, `{ MINUS { ... } }`, is one empty solution, as SPARQL 1.1 defines it, so a
`DELETE ... WHERE` over it deletes nothing.

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
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT ?g ?name
WHERE {
    GRAPH ?g {
        ?person foaf:name ?name
    }
}
```

`GRAPH ?g { ... }` evaluates its pattern once per named graph (the ones `FROM NAMED` lists, if any), as SPARQL 1.1
does, so `SELECT ?g WHERE { GRAPH ?g { } }` lists the named graphs. Inside it, a `MINUS` that shares only `?g` with
the rows removes nothing.

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

`BIND` to a variable already in scope in its group, such as `{ ?s :p ?o BIND(1 AS ?o) }`, is a syntax error that
names the variable, as SPARQL 1.1 requires, also in the `WHERE` of an update.

## VALUES

Provide inline data:

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

SELECT ?name ?email
WHERE {
    VALUES ?person { <http://example.org/alix> <http://example.org/gus> }
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
