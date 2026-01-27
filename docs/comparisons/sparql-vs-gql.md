---
title: SPARQL vs GQL
description: Compare SPARQL and GQL query languages for graph databases.
tags:
  - sparql
  - gql
  - comparison
---

# SPARQL vs GQL

This guide compares SPARQL (W3C standard for RDF) with GQL (ISO standard for property graphs). Both are powerful graph query languages, but they're designed for different data models.

## Data Model Differences

| Aspect | SPARQL (RDF) | GQL (Property Graph) |
|--------|--------------|---------------------|
| **Basic Unit** | Triple (subject-predicate-object) | Nodes and edges with properties |
| **Schema** | Schema-less, uses ontologies | Labels and property types |
| **Identity** | IRIs (URIs) | Internal IDs |
| **Properties** | Reified as triples | First-class attributes |
| **Multi-values** | Multiple triples | Arrays/lists |

## Query Syntax Comparison

### Finding Entities

=== "SPARQL"

    ```sparql
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>

    SELECT ?name
    WHERE {
        ?person a foaf:Person .
        ?person foaf:name ?name
    }
    ```

=== "GQL"

    ```sql
    MATCH (person:Person)
    RETURN person.name
    ```

### Finding Relationships

=== "SPARQL"

    ```sparql
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>

    SELECT ?name ?friendName
    WHERE {
        ?person foaf:name ?name .
        ?person foaf:knows ?friend .
        ?friend foaf:name ?friendName
    }
    ```

=== "GQL"

    ```sql
    MATCH (person:Person)-[:KNOWS]->(friend:Person)
    RETURN person.name, friend.name
    ```

### Filtering

=== "SPARQL"

    ```sparql
    SELECT ?name ?age
    WHERE {
        ?person foaf:name ?name .
        ?person foaf:age ?age
        FILTER(?age > 30)
    }
    ```

=== "GQL"

    ```sql
    MATCH (person:Person)
    WHERE person.age > 30
    RETURN person.name, person.age
    ```

### Optional Matches

=== "SPARQL"

    ```sparql
    SELECT ?name ?email
    WHERE {
        ?person foaf:name ?name
        OPTIONAL { ?person foaf:mbox ?email }
    }
    ```

=== "GQL"

    ```sql
    MATCH (person:Person)
    OPTIONAL MATCH (person)-[:HAS_EMAIL]->(email:Email)
    RETURN person.name, email.address
    ```

### Aggregations

=== "SPARQL"

    ```sparql
    SELECT ?country (COUNT(?person) AS ?count)
    WHERE {
        ?person foaf:based_near ?country
    }
    GROUP BY ?country
    HAVING (COUNT(?person) > 10)
    ORDER BY DESC(?count)
    ```

=== "GQL"

    ```sql
    MATCH (person:Person)-[:LIVES_IN]->(country:Country)
    WITH country, COUNT(person) AS count
    WHERE count > 10
    RETURN country.name, count
    ORDER BY count DESC
    ```

### Path Traversal

=== "SPARQL"

    ```sparql
    # All ancestors (transitive closure)
    SELECT ?ancestor
    WHERE {
        ?person ex:parent+ ?ancestor
    }
    ```

=== "GQL"

    ```sql
    MATCH (person:Person)-[:PARENT*1..]->(ancestor:Person)
    RETURN ancestor
    ```

### Union Queries

=== "SPARQL"

    ```sparql
    SELECT ?contact
    WHERE {
        { ?person foaf:mbox ?contact }
        UNION
        { ?person foaf:phone ?contact }
    }
    ```

=== "GQL"

    ```sql
    MATCH (person:Person)-[:HAS_EMAIL]->(e:Email)
    RETURN e.address AS contact
    UNION
    MATCH (person:Person)-[:HAS_PHONE]->(p:Phone)
    RETURN p.number AS contact
    ```

## Feature Comparison

| Feature | SPARQL | GQL |
|---------|--------|-----|
| **Pattern Matching** | Triple patterns | Node/edge patterns |
| **Path Expressions** | Property paths (`+`, `*`, `?`) | Variable-length patterns (`*1..5`) |
| **Negation** | `MINUS`, `NOT EXISTS` | `NOT`, `WHERE NOT` |
| **Subqueries** | Full support | Full support |
| **Aggregation** | `GROUP BY`, `HAVING` | `WITH`, `HAVING` |
| **Updates** | SPARQL Update (INSERT/DELETE) | `INSERT`, `SET`, `DELETE` |
| **Federated Queries** | `SERVICE` keyword | Not standard |
| **Named Graphs** | `GRAPH` keyword | Limited support |

## When to Use Each

### Use SPARQL When:

- Working with **RDF/Linked Data**
- Need **semantic reasoning** (RDFS/OWL inference)
- Querying **knowledge graphs** with ontologies
- **Federated queries** across multiple endpoints
- Data follows **W3C standards** (Dublin Core, FOAF, Schema.org)

### Use GQL When:

- Working with **property graph** data
- Need **intuitive pattern matching** syntax
- Building **application databases**
- Relationships have **properties** (weights, timestamps)
- Prefer **ASCII-art style** query patterns

## Graphos Support

Graphos supports both query languages:

```rust
// Enable both features
[dependencies]
graphos-engine = { version = "0.1", features = ["gql", "sparql"] }
```

=== "SPARQL Query"

    ```python
    import graphos

    db = graphos.Database()

    # SPARQL query
    result = db.sparql("""
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        SELECT ?name
        WHERE { ?person foaf:name ?name }
    """)
    ```

=== "GQL Query"

    ```python
    import graphos

    db = graphos.Database()

    # GQL query
    result = db.query("""
        MATCH (person:Person)
        RETURN person.name
    """)
    ```

## Performance Considerations

| Aspect | SPARQL | GQL |
|--------|--------|-----|
| **Index Usage** | SPO, POS, OSP indexes | Node/edge/property indexes |
| **Join Strategy** | Hash/merge joins on variables | Pattern-based joins |
| **Path Queries** | Optimized for transitive closure | Optimized for bounded paths |
| **Cardinality** | Triple-based estimation | Node/edge-based estimation |

## Migration Tips

### SPARQL to GQL

1. Replace triple patterns with node/edge patterns
2. Convert `FILTER` to `WHERE` clauses
3. Replace property paths with variable-length patterns
4. Map IRIs to node labels and properties

### GQL to SPARQL

1. Define appropriate prefixes for your domain
2. Model node properties as separate triples
3. Convert edge patterns to predicate URIs
4. Use `OPTIONAL` for optional relationships

## Further Reading

- [SPARQL 1.1 Specification](https://www.w3.org/TR/sparql11-query/)
- [GQL Standard (ISO/IEC 39075)](https://www.iso.org/standard/76120.html)
- [LPG vs RDF Data Models](../user-guide/data-model/lpg-vs-rdf.md)
- [GQL vs SPARQL Concepts](../user-guide/data-model/gql-vs-sparql.md)
