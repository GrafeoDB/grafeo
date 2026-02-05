---
title: Built-in Functions
description: Learn about SPARQL built-in functions for strings, numbers, dates, and more.
tags:
  - sparql
  - functions
  - expressions
---

# Built-in Functions

SPARQL provides a rich set of built-in functions for manipulating and testing values.

## String Functions

### Length and Substring

```sparql
# String length
SELECT ?name (STRLEN(?name) AS ?length)
WHERE { ?person foaf:name ?name }

# Substring (1-indexed)
SELECT (SUBSTR("Hello World", 1, 5) AS ?result)  # "Hello"
SELECT (SUBSTR("Hello World", 7) AS ?result)     # "World"
```

### Case Conversion

```sparql
SELECT ?name
       (UCASE(?name) AS ?upper)
       (LCASE(?name) AS ?lower)
WHERE { ?person foaf:name ?name }
```

### Search and Replace

```sparql
# Contains
SELECT ?name
WHERE {
    ?person foaf:name ?name
    FILTER(CONTAINS(?name, "John"))
}

# Starts/ends with
SELECT ?email
WHERE {
    ?person foaf:mbox ?email
    FILTER(STRSTARTS(STR(?email), "mailto:"))
    FILTER(STRENDS(STR(?email), ".org"))
}

# Replace
SELECT (REPLACE("Hello World", "World", "SPARQL") AS ?result)
# Result: "Hello SPARQL"

# Replace with regex
SELECT (REPLACE("a1b2c3", "[0-9]", "X") AS ?result)
# Result: "aXbXcX"
```

### String Construction

```sparql
# Concatenation
SELECT (CONCAT(?first, " ", ?last) AS ?fullName)
WHERE {
    ?person foaf:firstName ?first .
    ?person foaf:lastName ?last
}

# String encoding
SELECT (ENCODE_FOR_URI("Hello World") AS ?encoded)
# Result: "Hello%20World"
```

### String Testing

```sparql
# Regular expression
SELECT ?name
WHERE {
    ?person foaf:name ?name
    FILTER(REGEX(?name, "^[A-Z]", "i"))
}

# String comparison
SELECT ?a ?b
WHERE {
    ?x foaf:name ?a .
    ?y foaf:name ?b
    FILTER(STRBEFORE(?a, " ") = STRBEFORE(?b, " "))
}
```

## Numeric Functions

```sparql
# Absolute value
SELECT (ABS(-42) AS ?result)  # 42

# Rounding
SELECT (ROUND(3.7) AS ?rounded)    # 4
SELECT (FLOOR(3.7) AS ?floor)      # 3
SELECT (CEIL(3.2) AS ?ceiling)     # 4

# Random (0 to 1)
SELECT (RAND() AS ?random)
```

## Date/Time Functions

### Current Date/Time

```sparql
SELECT (NOW() AS ?currentTime)
```

### Extracting Components

```sparql
SELECT ?event ?date
       (YEAR(?date) AS ?year)
       (MONTH(?date) AS ?month)
       (DAY(?date) AS ?day)
       (HOURS(?date) AS ?hour)
       (MINUTES(?date) AS ?minute)
       (SECONDS(?date) AS ?second)
WHERE {
    ?event ex:date ?date
}

# Timezone
SELECT (TIMEZONE(?datetime) AS ?tz)
SELECT (TZ(?datetime) AS ?tzString)
```

## Type Functions

### Type Checking

```sparql
# Check types
SELECT ?value
WHERE {
    ?s ?p ?value
    FILTER(ISIRI(?value))      # Is IRI/URI?
    FILTER(ISBLANK(?value))    # Is blank node?
    FILTER(ISLITERAL(?value))  # Is literal?
    FILTER(ISNUMERIC(?value))  # Is numeric?
}

# Check if bound
SELECT ?name ?email
WHERE {
    ?person foaf:name ?name
    OPTIONAL { ?person foaf:mbox ?email }
    FILTER(BOUND(?email))
}
```

### Type Conversion

```sparql
# To string
SELECT (STR(<http://example.org/>) AS ?str)

# To IRI
SELECT (IRI("http://example.org/") AS ?iri)
SELECT (URI("http://example.org/") AS ?uri)

# Get datatype
SELECT ?value (DATATYPE(?value) AS ?type)
WHERE { ?s ex:prop ?value }

# Get language tag
SELECT ?label (LANG(?label) AS ?language)
WHERE { ?s rdfs:label ?label }
```

### Constructing Literals

```sparql
# String with language tag
SELECT (STRLANG("Hello", "en") AS ?greeting)

# Typed literal
SELECT (STRDT("42", xsd:integer) AS ?number)
```

## Hash Functions

```sparql
SELECT (MD5("hello") AS ?md5)
SELECT (SHA1("hello") AS ?sha1)
SELECT (SHA256("hello") AS ?sha256)
SELECT (SHA384("hello") AS ?sha384)
SELECT (SHA512("hello") AS ?sha512)
```

## Conditional Functions

### IF

```sparql
SELECT ?name (IF(?age >= 18, "Adult", "Minor") AS ?status)
WHERE {
    ?person foaf:name ?name .
    ?person foaf:age ?age
}
```

### COALESCE

Return first non-null value:

```sparql
SELECT ?person (COALESCE(?name, ?nick, "Unknown") AS ?displayName)
WHERE {
    ?person a foaf:Person
    OPTIONAL { ?person foaf:name ?name }
    OPTIONAL { ?person foaf:nick ?nick }
}
```

### BNODE

Create a blank node:

```sparql
SELECT (BNODE() AS ?newNode)
SELECT (BNODE("label") AS ?labeledNode)
```

## Node Functions

```sparql
# Get IRI as string
SELECT (STR(?person) AS ?iri)
WHERE { ?person a foaf:Person }

# Check same term
SELECT ?a ?b
WHERE {
    ?a foaf:knows ?b
    FILTER(SAMETERM(?a, ?b) = false)
}

# UUID generation
SELECT (UUID() AS ?uuid)
SELECT (STRUUID() AS ?uuidString)
```

## Function Summary

| Category | Functions |
|----------|-----------|
| **String** | `STRLEN`, `SUBSTR`, `UCASE`, `LCASE`, `CONTAINS`, `STRSTARTS`, `STRENDS`, `REPLACE`, `CONCAT`, `ENCODE_FOR_URI`, `REGEX` |
| **Numeric** | `ABS`, `ROUND`, `FLOOR`, `CEIL`, `RAND` |
| **Date/Time** | `NOW`, `YEAR`, `MONTH`, `DAY`, `HOURS`, `MINUTES`, `SECONDS`, `TIMEZONE`, `TZ` |
| **Type** | `ISIRI`, `ISBLANK`, `ISLITERAL`, `ISNUMERIC`, `BOUND`, `STR`, `IRI`, `DATATYPE`, `LANG` |
| **Hash** | `MD5`, `SHA1`, `SHA256`, `SHA384`, `SHA512` |
| **Conditional** | `IF`, `COALESCE` |
| **Node** | `BNODE`, `UUID`, `STRUUID`, `SAMETERM` |

## Custom Functions

Grafeo supports custom functions via plugins:

```sparql
PREFIX gfn: <http://grafeo.dev/functions/>

SELECT ?result
WHERE {
    BIND(gfn:customFunction(?input) AS ?result)
}
```

See [Custom Functions](../extending/custom-functions.md) for more information.
