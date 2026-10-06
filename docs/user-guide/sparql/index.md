---
title: SPARQL Query Language
description: Learn the SPARQL query language for RDF data in Grafeo.
---

# SPARQL Query Language

SPARQL (SPARQL Protocol and RDF Query Language) is the W3C standard query language for RDF (Resource Description Framework) data. Grafeo implements SPARQL 1.1 for querying RDF graphs.

## Overview

SPARQL uses triple patterns to match RDF data. It's designed for querying semantic web data and knowledge graphs.

## Quick Reference

| Operation | Syntax |
|-----------|--------|
| Select variables | `SELECT ?x ?y` |
| Match triples | `?s ?p ?o` |
| Filter results | `FILTER(?x > value)` |
| Optional patterns | `OPTIONAL { ?s ?p ?o }` |
| Union patterns | `{ ... } UNION { ... }` |
| Aggregate | `COUNT(?x)`, `SUM(?x)` |
| Order results | `ORDER BY ?x` |
| Limit results | `LIMIT 10` |
| Insert triples | `INSERT DATA { ... }` |
| Delete triples | `DELETE DATA { ... }` |
| Explain plan | `EXPLAIN SELECT ...` |
| Validate shapes | `db.validate_shacl(graph)` |

## RDF Data Model

Unlike property graphs (LPG), RDF uses triples:

```
Subject --Predicate--> Object
```

Example triples:
```
<http://example.org/alix> <http://xmlns.com/foaf/0.1/name> "Alix" .
<http://example.org/alix> <http://xmlns.com/foaf/0.1/knows> <http://example.org/gus> .
```

## Enabling SPARQL

SPARQL requires the `sparql` feature flag. The default Grafeo features (`lpg`, `gql`, `parallel`) do not include SPARQL.

=== "Rust"

    ```bash
    cargo add grafeo --features sparql
    ```

=== "Python"

    The published `grafeo` Python package includes SPARQL support by default:

    ```bash
    uv add grafeo
    ```

=== "Node.js"

    The published `@grafeo-db/node` package includes SPARQL support by default:

    ```bash
    npm install @grafeo-db/node
    ```

## Using SPARQL

=== "Python"

    ```python
    import grafeo

    db = grafeo.GrafeoDB()

    # Insert RDF triples
    db.execute_sparql("""
        INSERT DATA {
            <http://example.org/alix> <http://xmlns.com/foaf/0.1/name> "Alix" .
            <http://example.org/alix> <http://xmlns.com/foaf/0.1/knows> <http://example.org/gus> .
        }
    """)

    # Query triples
    result = db.execute_sparql("""
        SELECT ?name WHERE {
            <http://example.org/alix> <http://xmlns.com/foaf/0.1/name> ?name .
        }
    """)
    for row in result:
        print(row)
    ```

=== "Rust"

    ```rust
    use grafeo::GrafeoDB;

    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session.execute_sparql(r#"
        SELECT ?s ?p ?o WHERE { ?s ?p ?o }
    "#)?;
    ```

## Learn More

<div class="grid cards" markdown>

-   **[Basic Queries](basic-queries.md)**

    ---

    SELECT, WHERE and basic triple patterns.

-   **[Triple Patterns](patterns.md)**

    ---

    Matching subjects, predicates and objects.

-   **[Filtering](filtering.md)**

    ---

    FILTER expressions and conditions.

-   **[Aggregations](aggregations.md)**

    ---

    COUNT, SUM, AVG, GROUP BY and HAVING.

-   **[Property Paths](paths.md)**

    ---

    Path expressions for traversing relationships.

-   **[Built-in Functions](functions.md)**

    ---

    String, numeric and date/time functions.

-   **[Loading RDF Data](loading.md)**

    ---

    Turtle, N-Triples, N-Quads import and streaming load.

-   **[SPARQL Update](update.md)**

    ---

    INSERT DATA, DELETE DATA, pattern-based updates, and named graph operations.

-   **[EXPLAIN and EXPLAIN ANALYZE](explain.md)**

    ---

    Inspect query execution plans and profile operator performance.

-   **[SHACL Validation](shacl-validation.md)**

    ---

    Validate RDF data against SHACL shapes with SHACL Core constraint types.

</div>
