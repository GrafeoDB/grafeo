---
title: Aggregations
description: Learn about SPARQL aggregate functions and grouping.
tags:
  - sparql
  - aggregations
  - groupby
---

# Aggregations

SPARQL provides aggregate functions for computing summary statistics over groups of results.

## Aggregate Functions

| Function | Description |
|----------|-------------|
| `COUNT(*)` | Count all results |
| `COUNT(?x)` | Count bound values |
| `SUM(?x)` | Sum of numeric values |
| `AVG(?x)` | Average of numeric values |
| `MIN(?x)` | Minimum value |
| `MAX(?x)` | Maximum value |
| `SAMPLE(?x)` | Arbitrary value from group |
| `GROUP_CONCAT(?x)` | Concatenate values |

## COUNT

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

# Count all people
SELECT (COUNT(*) AS ?total)
WHERE {
    ?person a foaf:Person
}

# Count distinct values
SELECT (COUNT(DISTINCT ?country) AS ?countries)
WHERE {
    ?person foaf:based_near ?country
}
```

## SUM and AVG

```sparql
PREFIX ex: <http://example.org/>

# Total price
SELECT (SUM(?price) AS ?total)
WHERE {
    ?order ex:item ?item .
    ?item ex:price ?price
}

# Average age
SELECT (AVG(?age) AS ?avgAge)
WHERE {
    ?person foaf:age ?age
}
```

## MIN and MAX

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX ex: <http://example.org/>

# Youngest and oldest
SELECT (MIN(?age) AS ?youngest) (MAX(?age) AS ?oldest)
WHERE {
    ?person foaf:age ?age
}

# Price range
SELECT (MIN(?price) AS ?minPrice) (MAX(?price) AS ?maxPrice)
WHERE {
    ?product ex:price ?price
}
```

## GROUP BY

Group results by one or more variables:

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX ex: <http://example.org/>

# Count people per country
SELECT ?country (COUNT(?person) AS ?population)
WHERE {
    ?person a foaf:Person .
    ?person ex:country ?country
}
GROUP BY ?country

# Multiple grouping keys
SELECT ?country ?city (COUNT(?person) AS ?count)
WHERE {
    ?person ex:country ?country .
    ?person ex:city ?city
}
GROUP BY ?country ?city
```

## HAVING

Filter groups after aggregation:

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX ex: <http://example.org/>

# Countries with more than 100 people
SELECT ?country (COUNT(?person) AS ?population)
WHERE {
    ?person a foaf:Person .
    ?person ex:country ?country
}
GROUP BY ?country
HAVING (COUNT(?person) > 100)

# Categories with average price over 50
SELECT ?category (AVG(?price) AS ?avgPrice)
WHERE {
    ?product ex:category ?category .
    ?product ex:price ?price
}
GROUP BY ?category
HAVING (AVG(?price) > 50)
ORDER BY DESC(?avgPrice)
```

## GROUP_CONCAT

Concatenate values within a group:

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

# List all friends as a comma-separated string
SELECT ?person (GROUP_CONCAT(?friendName; separator=", ") AS ?friends)
WHERE {
    ?person foaf:name ?name .
    ?person foaf:knows ?friend .
    ?friend foaf:name ?friendName
}
GROUP BY ?person

# With DISTINCT
SELECT ?category (GROUP_CONCAT(DISTINCT ?tag; separator="; ") AS ?tags)
WHERE {
    ?product ex:category ?category .
    ?product ex:tag ?tag
}
GROUP BY ?category
```

## SAMPLE

Get an arbitrary value from each group:

```sparql
PREFIX ex: <http://example.org/>

# One example product per category
SELECT ?category (SAMPLE(?product) AS ?example)
WHERE {
    ?product ex:category ?category
}
GROUP BY ?category
```

## Combining Aggregates

```sparql
PREFIX ex: <http://example.org/>

SELECT ?category
       (COUNT(?product) AS ?productCount)
       (AVG(?price) AS ?avgPrice)
       (MIN(?price) AS ?minPrice)
       (MAX(?price) AS ?maxPrice)
       (SUM(?quantity) AS ?totalStock)
WHERE {
    ?product ex:category ?category .
    ?product ex:price ?price .
    ?product ex:quantity ?quantity
}
GROUP BY ?category
HAVING (COUNT(?product) >= 5)
ORDER BY DESC(?productCount)
```

## Aggregates with ORDER BY and LIMIT

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX ex: <http://example.org/>

# Top 10 most connected people
SELECT ?name (COUNT(?friend) AS ?friendCount)
WHERE {
    ?person foaf:name ?name .
    ?person foaf:knows ?friend
}
GROUP BY ?person ?name
ORDER BY DESC(?friendCount)
LIMIT 10
```

## Nested Aggregates with Subqueries

```sparql
PREFIX ex: <http://example.org/>

# Find categories with above-average product count
SELECT ?category ?productCount
WHERE {
    {
        SELECT ?category (COUNT(?product) AS ?productCount)
        WHERE {
            ?product ex:category ?category
        }
        GROUP BY ?category
    }
    {
        SELECT (AVG(?cnt) AS ?avgCount)
        WHERE {
            SELECT ?cat (COUNT(?p) AS ?cnt)
            WHERE { ?p ex:category ?cat }
            GROUP BY ?cat
        }
    }
    FILTER(?productCount > ?avgCount)
}
ORDER BY DESC(?productCount)
```

## DISTINCT with Aggregates

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

# Count unique countries per person
SELECT ?person (COUNT(DISTINCT ?country) AS ?countriesVisited)
WHERE {
    ?person foaf:visited ?country
}
GROUP BY ?person
```
