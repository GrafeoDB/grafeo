---
title: Filtering
description: Learn about SPARQL FILTER expressions and conditions.
tags:
  - sparql
  - filter
  - conditions
---

# Filtering

The `FILTER` clause restricts results based on conditions. Filters can use comparisons, logical operators, and built-in functions.

## Basic FILTER

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

SELECT ?name ?age
WHERE {
    ?person foaf:name ?name .
    ?person foaf:age ?age
    FILTER(?age > 30)
}
```

## Comparison Operators

| Operator | Description |
|----------|-------------|
| `=` | Equal |
| `!=` | Not equal |
| `<` | Less than |
| `>` | Greater than |
| `<=` | Less than or equal |
| `>=` | Greater than or equal |

```sparql
SELECT ?name ?age
WHERE {
    ?person foaf:name ?name .
    ?person foaf:age ?age
    FILTER(?age >= 18 && ?age <= 65)
}
```

## Logical Operators

| Operator | Description |
|----------|-------------|
| `&&` | Logical AND |
| `\|\|` | Logical OR |
| `!` | Logical NOT |

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

SELECT ?name
WHERE {
    ?person foaf:name ?name .
    ?person foaf:age ?age
    FILTER(?age > 18 && ?age < 65)
}

SELECT ?name
WHERE {
    ?person foaf:name ?name
    FILTER(!BOUND(?age) || ?age > 0)
}
```

## String Matching

### REGEX

```sparql
SELECT ?name
WHERE {
    ?person foaf:name ?name
    FILTER(REGEX(?name, "^A", "i"))  # Names starting with A (case-insensitive)
}
```

### CONTAINS, STRSTARTS, STRENDS

```sparql
SELECT ?name
WHERE {
    ?person foaf:name ?name
    FILTER(CONTAINS(?name, "son"))
}

SELECT ?name
WHERE {
    ?person foaf:name ?name
    FILTER(STRSTARTS(?name, "John"))
}

SELECT ?email
WHERE {
    ?person foaf:mbox ?email
    FILTER(STRENDS(STR(?email), ".org"))
}
```

## Type Checking Functions

| Function | Description |
|----------|-------------|
| `ISIRI(?x)` | True if IRI |
| `ISBLANK(?x)` | True if blank node |
| `ISLITERAL(?x)` | True if literal |
| `ISNUMERIC(?x)` | True if numeric |
| `BOUND(?x)` | True if variable is bound |

```sparql
SELECT ?value
WHERE {
    ?s ?p ?value
    FILTER(ISLITERAL(?value) && ISNUMERIC(?value))
}
```

## Language Filtering

```sparql
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

# Filter by language tag
SELECT ?label
WHERE {
    ?x rdfs:label ?label
    FILTER(LANG(?label) = "en")
}

# Filter for any English variant
SELECT ?label
WHERE {
    ?x rdfs:label ?label
    FILTER(LANGMATCHES(LANG(?label), "en"))
}
```

## Numeric Filtering

```sparql
PREFIX ex: <http://example.org/>

SELECT ?product ?price
WHERE {
    ?product ex:price ?price
    FILTER(?price > 10 && ?price < 100)
}

# Using numeric functions
SELECT ?value
WHERE {
    ?s ex:measurement ?value
    FILTER(ABS(?value) < 0.5)
}
```

## Date/Time Filtering

```sparql
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX ex: <http://example.org/>

SELECT ?event ?date
WHERE {
    ?event ex:date ?date
    FILTER(?date > "2024-01-01"^^xsd:date)
}

# Filter by year
SELECT ?event ?date
WHERE {
    ?event ex:date ?date
    FILTER(YEAR(?date) = 2024)
}
```

## EXISTS and NOT EXISTS

### EXISTS

True if the pattern has at least one match:

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

# People who know someone
SELECT ?name
WHERE {
    ?person foaf:name ?name
    FILTER EXISTS { ?person foaf:knows ?someone }
}
```

### NOT EXISTS

True if the pattern has no matches:

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

# People who don't know anyone
SELECT ?name
WHERE {
    ?person foaf:name ?name
    FILTER NOT EXISTS { ?person foaf:knows ?someone }
}
```

## IN and NOT IN

```sparql
PREFIX ex: <http://example.org/>

# Filter by list of values
SELECT ?person ?status
WHERE {
    ?person ex:status ?status
    FILTER(?status IN ("active", "pending", "approved"))
}

# Exclude values
SELECT ?person ?country
WHERE {
    ?person ex:country ?country
    FILTER(?country NOT IN ("Unknown", "N/A"))
}
```

## Combining Filters

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX ex: <http://example.org/>

SELECT ?name ?age ?country
WHERE {
    ?person foaf:name ?name .
    ?person foaf:age ?age .
    ?person ex:country ?country
    FILTER(
        ?age >= 18 &&
        ?age <= 65 &&
        ?country IN ("USA", "Canada", "UK") &&
        REGEX(?name, "^[A-Z]")
    )
}
```

## Filter Placement

Filters are typically placed after the patterns they reference:

```sparql
# Good: Filter after relevant patterns
SELECT ?name ?age
WHERE {
    ?person foaf:name ?name .
    ?person foaf:age ?age
    FILTER(?age > 30)
}

# Also valid: Multiple filters
SELECT ?name ?age
WHERE {
    ?person foaf:name ?name
    FILTER(STRLEN(?name) > 3)
    ?person foaf:age ?age
    FILTER(?age > 30)
}
```
