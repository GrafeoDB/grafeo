---
title: GQL vs SPARQL
description: Comparing GQL and SPARQL query languages.
tags:
  - gql
  - sparql
  - comparison
---

# GQL vs SPARQL

Comparing the ISO standard GQL with W3C standard SPARQL.

!!! info "Graphos Query Language"
    Graphos uses **GQL** as its query language. **SPARQL is not supported**. This guide is for users familiar with SPARQL/RDF who want to understand how GQL differs and how to accomplish similar tasks in Graphos.

## Overview

| Aspect | GQL | SPARQL |
|--------|-----|--------|
| **Standard** | ISO/IEC 39075 | W3C Recommendation |
| **Data Model** | LPG | RDF |
| **Pattern Style** | ASCII art | Triple patterns |
| **Verbosity** | Concise | More verbose |
| **Learning Curve** | Lower | Higher |

## Query Comparison

### Basic Query

=== "GQL"

    ```sql
    MATCH (p:Person)
    WHERE p.age > 25
    RETURN p.name, p.age
    ```

=== "SPARQL"

    ```sparql
    PREFIX : <http://example.org/>
    SELECT ?name ?age
    WHERE {
        ?p a :Person ;
           :name ?name ;
           :age ?age .
        FILTER (?age > 25)
    }
    ```

### Relationship Query

=== "GQL"

    ```sql
    MATCH (a:Person)-[:KNOWS]->(b:Person)
    RETURN a.name, b.name
    ```

=== "SPARQL"

    ```sparql
    PREFIX : <http://example.org/>
    SELECT ?aName ?bName
    WHERE {
        ?a a :Person ;
           :name ?aName ;
           :knows ?b .
        ?b a :Person ;
           :name ?bName .
    }
    ```

### Path Queries

=== "GQL"

    ```sql
    MATCH path = (a:Person)-[:KNOWS*1..3]->(b:Person)
    WHERE a.name = 'Alice'
    RETURN path
    ```

=== "SPARQL"

    ```sparql
    PREFIX : <http://example.org/>
    SELECT ?path
    WHERE {
        :Alice :knows+ ?b .
        ?b a :Person .
    }
    ```

### Aggregation

=== "GQL"

    ```sql
    MATCH (p:Person)-[:WORKS_AT]->(c:Company)
    RETURN c.name, count(p) AS employees
    ORDER BY employees DESC
    ```

=== "SPARQL"

    ```sparql
    PREFIX : <http://example.org/>
    SELECT ?companyName (COUNT(?p) AS ?employees)
    WHERE {
        ?p a :Person ;
           :worksAt ?c .
        ?c a :Company ;
           :name ?companyName .
    }
    GROUP BY ?companyName
    ORDER BY DESC(?employees)
    ```

### Creating Data

=== "GQL"

    ```sql
    INSERT (:Person {name: 'Alice', age: 30})
    ```

=== "SPARQL"

    ```sparql
    PREFIX : <http://example.org/>
    INSERT DATA {
        :Alice a :Person ;
               :name "Alice" ;
               :age 30 .
    }
    ```

## Key Differences

### 1. Pattern Syntax

GQL uses ASCII-art patterns that visually represent the graph:

```sql
-- GQL: Visual pattern
(a)-[:KNOWS]->(b)-[:WORKS_AT]->(c)
```

SPARQL uses triple patterns:

```sparql
-- SPARQL: Triple patterns
?a :knows ?b .
?b :worksAt ?c .
```

### 2. Properties on Relationships

GQL supports properties on relationships natively:

```sql
-- GQL
MATCH (a)-[r:KNOWS {since: 2020}]->(b)
RETURN r.strength
```

SPARQL requires reification:

```sparql
-- SPARQL (RDF-star)
<< ?a :knows ?b >> :since 2020 ;
                   :strength ?strength .
```

### 3. Schema

- **GQL**: Schema-optional, validation at application level
- **SPARQL**: Uses RDFS/OWL ontologies for schema and reasoning

### 4. Identity

- **GQL**: Internal IDs, application-managed
- **SPARQL**: URIs enable global identity and linking

## Migration Considerations

### From SPARQL to GQL

1. Map RDF types to LPG labels
2. Map predicates to edge types or properties
3. Handle reification explicitly
4. Adapt property path syntax

### From GQL to SPARQL

1. Define URI scheme for nodes
2. Create vocabulary for edge types
3. Handle edge properties with reification
4. Adapt pattern matching syntax

## When to Use Each

| Use Case | Recommended |
|----------|-------------|
| Application database | GQL |
| Knowledge graph | Either |
| Linked Data | SPARQL |
| Data integration | SPARQL |
| Performance-critical | GQL |
| Developer productivity | GQL |
