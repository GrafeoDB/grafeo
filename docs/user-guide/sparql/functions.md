---
title: Built-in Functions
description: Learn about SPARQL built-in functions for strings, numbers, dates and more.
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
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

# String length
SELECT ?name (STRLEN(?name) AS ?length)
WHERE { ?person foaf:name ?name }

# Substring (1-indexed)
SELECT (SUBSTR("Hello World", 1, 5) AS ?result) WHERE {}  # "Hello"
SELECT (SUBSTR("Hello World", 7) AS ?result) WHERE {}  # "World"
```

### Case Conversion

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

SELECT ?name
       (UCASE(?name) AS ?upper)
       (LCASE(?name) AS ?lower)
WHERE { ?person foaf:name ?name }
```

### Search and Replace

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

# Contains
SELECT ?name
WHERE {
    ?person foaf:name ?name
    FILTER(CONTAINS(?name, "John"))
}

# Starts/ends with
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT ?email
WHERE {
    ?person foaf:mbox ?email
    FILTER(STRSTARTS(STR(?email), "mailto:"))
    FILTER(STRENDS(STR(?email), ".org"))
}

# Replace
SELECT (REPLACE("Hello World", "World", "SPARQL") AS ?result) WHERE {}
# Result: "Hello SPARQL"

# Replace with regex
SELECT (REPLACE("a1b2c3", "[0-9]", "X") AS ?result) WHERE {}
# Result: "aXbXcX"
```

### String Construction

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

# Concatenation
SELECT (CONCAT(?first, " ", ?last) AS ?fullName)
WHERE {
    ?person foaf:firstName ?first .
    ?person foaf:lastName ?last
}

# String encoding
SELECT (ENCODE_FOR_URI("Hello World") AS ?encoded) WHERE {}
# Result: "Hello%20World"
```

### String Testing

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

# Regular expression
SELECT ?name
WHERE {
    ?person foaf:name ?name
    FILTER(REGEX(?name, "^[A-Z]", "i"))
}

# String comparison
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT ?a ?b
WHERE {
    ?x foaf:name ?a .
    ?y foaf:name ?b
    FILTER(STRBEFORE(?a, " ") = STRBEFORE(?b, " "))
}
```

## Numeric Functions

Numbers follow the type promotion of SPARQL 1.1: `1.5` is an exact
`xsd:decimal` and `1.5e0` an `xsd:double`. An integer with an integer gives an
integer, except a division, which gives a decimal (`5 / 2` is `2.5`); a
decimal with an integer or a decimal gives a decimal; an `xsd:float` with a
float, an integer or a decimal gives an `xsd:float` (as does
`xsd:float("1.5")`); a double makes a double. Integer arithmetic past the 64-bit range goes on as decimals. A stored numeric
literal reads as a number of its type (`"2.50"^^xsd:decimal` as the decimal
`2.50`), so its arithmetic is exact too.

```sparql
SELECT (0.1 + 0.2 AS ?exact) WHERE {}  # 0.3, an xsd:decimal
SELECT (5 / 2 AS ?half) WHERE {}  # 2.5, an xsd:decimal

# Absolute value
SELECT (ABS(-42) AS ?result) WHERE {}  # 42

# Rounding
SELECT (ROUND(3.7) AS ?rounded) WHERE {}  # 4
SELECT (FLOOR(3.7) AS ?floor) WHERE {}  # 3
SELECT (CEIL(3.2) AS ?ceiling) WHERE {}  # 4

# Random (0 to 1)
SELECT (RAND() AS ?random) WHERE {}
```

## Date/Time Functions

### Current Date/Time

```sparql
SELECT (NOW() AS ?currentTime) WHERE {}
```

### Extracting Components

```sparql
PREFIX ex: <http://example.org/>

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
PREFIX ex: <http://example.org/>
SELECT (TIMEZONE(?datetime) AS ?tz) WHERE { ?event ex:date ?datetime }
PREFIX ex: <http://example.org/>
SELECT (TZ(?datetime) AS ?tzString) WHERE { ?event ex:date ?datetime }
```

## Type Functions

### Type Checking

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

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
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT ?name ?email
WHERE {
    ?person foaf:name ?name
    OPTIONAL { ?person foaf:mbox ?email }
    FILTER(BOUND(?email))
}
```

### Type Conversion

```sparql
PREFIX ex: <http://example.org/>

# To string
SELECT (STR(<http://example.org/>) AS ?str) WHERE {}

# To IRI
SELECT (IRI("http://example.org/") AS ?iri) WHERE {}
SELECT (URI("http://example.org/") AS ?uri) WHERE {}

# Get datatype
PREFIX ex: <http://example.org/>
SELECT ?value (DATATYPE(?value) AS ?type)
WHERE { ?s ex:prop ?value }

# Get language tag
SELECT ?label (LANG(?label) AS ?language)
WHERE { ?s rdfs:label ?label }

# XPath casts, by the type's IRI
SELECT (xsd:decimal("2.50") AS ?d) WHERE {}  # 2.50
SELECT (xsd:integer(2.9) AS ?i) WHERE {}  # 2, truncated
SELECT (xsd:double("1.5") AS ?f) WHERE {}  # 1.5
SELECT (xsd:string(1.50) AS ?s) WHERE {}  # "1.50"
SELECT (xsd:boolean("1") AS ?b) WHERE {}  # true
```

The casts are `xsd:decimal`, `xsd:double`, `xsd:float`, `xsd:integer` (and
`xsd:int`, `xsd:long`), `xsd:string` and `xsd:boolean`; a value that does not
cast gives no value.

### Constructing Literals

```sparql
# String with language tag
SELECT (STRLANG("Hello", "en") AS ?greeting) WHERE {}

# Typed literal
SELECT (STRDT("42", xsd:integer) AS ?number) WHERE {}
SELECT (STRDT("19.99", xsd:decimal) AS ?price) WHERE {}  # an exact decimal
```

## Hash Functions

```sparql
SELECT (MD5("hello") AS ?md5) WHERE {}
SELECT (SHA1("hello") AS ?sha1) WHERE {}
SELECT (SHA256("hello") AS ?sha256) WHERE {}
SELECT (SHA384("hello") AS ?sha384) WHERE {}
SELECT (SHA512("hello") AS ?sha512) WHERE {}
```

## Conditional Functions

### IF

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

SELECT ?name (IF(?age >= 18, "Adult", "Minor") AS ?status)
WHERE {
    ?person foaf:name ?name .
    ?person foaf:age ?age
}
```

### COALESCE

Return first non-null value:

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

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
SELECT (BNODE() AS ?newNode) WHERE {}
SELECT (BNODE("label") AS ?labeledNode) WHERE {}
```

## Node Functions

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

# Get IRI as string
SELECT (STR(?person) AS ?iri)
WHERE { ?person a foaf:Person }

# Check same term
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT ?a ?b
WHERE {
    ?a foaf:knows ?b
    FILTER(SAMETERM(?a, ?b) = false)
}

# UUID generation
SELECT (UUID() AS ?uuid) WHERE {}
SELECT (STRUUID() AS ?uuidString) WHERE {}
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

## EXPLAIN

View the optimized logical plan for a SPARQL query without executing it:

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

EXPLAIN SELECT ?name ?age
WHERE {
    ?person a foaf:Person .
    ?person foaf:name ?name .
    ?person foaf:age ?age
    FILTER(?age > 30)
}
```

## Data Mutations

SPARQL supports data modification via INSERT, DELETE and graph management:

### INSERT DATA

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX ex: <http://example.org/>

INSERT DATA {
    ex:alix a foaf:Person ;
        foaf:name "Alix" ;
        foaf:age 30 .
}

# Insert into a named graph
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX ex: <http://example.org/>
INSERT DATA {
    GRAPH <http://example.org/friends> {
        ex:alix foaf:knows ex:gus .
    }
}
```

### DELETE DATA

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX ex: <http://example.org/>

DELETE DATA {
    ex:alix foaf:age 30 .
}
```

### DELETE/INSERT (Conditional)

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX ex: <http://example.org/>

# Update age
DELETE { ?person foaf:age ?oldAge }
INSERT { ?person foaf:age 31 }
WHERE {
    ?person foaf:name "Alix" .
    ?person foaf:age ?oldAge
}
```

### CLEAR and DROP

```sparql
# Clear all triples from default graph
CLEAR DEFAULT

# A named graph to work on
INSERT DATA {
    GRAPH <http://example.org/friends> {
        <http://example.org/alix> <http://xmlns.com/foaf/0.1/knows> <http://example.org/gus> .
    }
}

# Clear a named graph
CLEAR GRAPH <http://example.org/friends>

# Drop a named graph (removes graph itself)
DROP GRAPH <http://example.org/friends>

# A graph that does not exist fails; SILENT makes the drop succeed anyway
DROP SILENT GRAPH <http://example.org/nonexistent>
```

### COPY, MOVE and ADD

```sparql
# The source graph must exist: give it a triple first
INSERT DATA {
    GRAPH <http://example.org/source> {
        <http://example.org/alix> <http://xmlns.com/foaf/0.1/name> "Alix" .
    }
}

# Copy all triples from one graph to another (replaces target)
COPY <http://example.org/source> TO <http://example.org/target>

# Add triples from source to target (keeps both)
ADD <http://example.org/source> TO <http://example.org/target>

# Move all triples (removes source)
MOVE <http://example.org/source> TO <http://example.org/target>
```

A `COPY`, `MOVE` or `ADD` whose source graph does not exist fails ("Source graph <...> does not exist"), as SPARQL
1.1 Update specifies; `SILENT` (`COPY SILENT ...`) skips it without an error.

All mutation operations are WAL-logged and recovered on restart.

See [Custom Functions](../extending/custom-functions.md) for more information.
