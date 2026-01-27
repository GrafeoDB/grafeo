---
title: Basic Queries
description: Learn basic SPARQL queries with SELECT and WHERE.
tags:
  - sparql
  - queries
  - rdf
---

# Basic Queries

This guide covers the fundamentals of querying RDF data with SPARQL.

## SELECT Clause

The `SELECT` clause specifies which variables to return:

```sparql
# Select all variables
SELECT *
WHERE { ?s ?p ?o }

# Select specific variables
SELECT ?name ?age
WHERE { ?person foaf:name ?name . ?person foaf:age ?age }

# Select with DISTINCT
SELECT DISTINCT ?type
WHERE { ?x rdf:type ?type }
```

## WHERE Clause

The `WHERE` clause contains triple patterns to match:

```sparql
# Match all triples
SELECT ?s ?p ?o
WHERE { ?s ?p ?o }

# Match triples with a specific predicate
SELECT ?person ?name
WHERE { ?person <http://xmlns.com/foaf/0.1/name> ?name }

# Match triples with a specific object
SELECT ?person
WHERE { ?person rdf:type <http://xmlns.com/foaf/0.1/Person> }
```

## Prefixes

Use `PREFIX` to abbreviate IRIs:

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT ?name
WHERE {
    ?person rdf:type foaf:Person .
    ?person foaf:name ?name
}
```

## Multiple Triple Patterns

Chain triple patterns with `.` (period):

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

SELECT ?name ?email
WHERE {
    ?person foaf:name ?name .
    ?person foaf:mbox ?email .
    ?person foaf:age ?age
}
```

## Ordering Results

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

# Order by a variable
SELECT ?name ?age
WHERE {
    ?person foaf:name ?name .
    ?person foaf:age ?age
}
ORDER BY ?age

# Descending order
SELECT ?name ?age
WHERE {
    ?person foaf:name ?name .
    ?person foaf:age ?age
}
ORDER BY DESC(?age)

# Multiple sort keys
SELECT ?name ?age
WHERE {
    ?person foaf:name ?name .
    ?person foaf:age ?age
}
ORDER BY DESC(?age) ?name
```

## Limiting Results

```sparql
# Return first 10 results
SELECT ?name
WHERE { ?person foaf:name ?name }
LIMIT 10

# Skip and limit (pagination)
SELECT ?name
WHERE { ?person foaf:name ?name }
ORDER BY ?name
OFFSET 20 LIMIT 10
```

## ASK Queries

Check if a pattern exists (returns true/false):

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

ASK {
    ?person foaf:name "Alice" .
    ?person foaf:knows ?friend
}
```

## CONSTRUCT Queries

Build new RDF triples from query results:

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

CONSTRUCT {
    ?person foaf:fullName ?name
}
WHERE {
    ?person foaf:firstName ?first .
    ?person foaf:lastName ?last
    BIND(CONCAT(?first, " ", ?last) AS ?name)
}
```

## DESCRIBE Queries

Get information about a resource:

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

DESCRIBE ?person
WHERE {
    ?person foaf:name "Alice"
}
```

## Blank Nodes

Match anonymous nodes:

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

SELECT ?name ?street
WHERE {
    ?person foaf:name ?name .
    ?person foaf:address [
        foaf:street ?street
    ]
}
```

## Complete Example

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

# Find all people and their friends' names
SELECT ?personName ?friendName
WHERE {
    ?person rdf:type foaf:Person .
    ?person foaf:name ?personName .
    ?person foaf:knows ?friend .
    ?friend foaf:name ?friendName
}
ORDER BY ?personName ?friendName
LIMIT 100
```
