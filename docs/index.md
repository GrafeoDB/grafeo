---
title: Grafeo - High-Performance Graph Database
description: A pure-Rust, embeddable graph database with Python, Node.js, Go, and WebAssembly bindings using GQL (ISO standard) query language.
hide:
  - navigation
  - toc
---

<style>
.md-typeset h1 {
  display: none;
}
</style>

<div class="hero" markdown>

# **Grafeo**

### A pure-Rust, high-performance, embeddable graph database

[Get Started](getting-started/index.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/GrafeoDB/grafeo){ .md-button }

</div>

---

## Why Grafeo?

<div class="grid cards" markdown>

-   :material-lightning-bolt:{ .lg .middle } **High Performance**

    ---

    Built from the ground up in Rust for maximum performance with vectorized execution, adaptive chunking, and SIMD-optimized operations.

-   :material-database-search:{ .lg .middle } **Multi-Language Queries**

    ---

    GQL, Cypher, Gremlin, GraphQL, and SPARQL. Choose the query language that fits your needs and expertise.

-   :material-graph:{ .lg .middle } **LPG & RDF Support**

    ---

    Dual data model support for both Labeled Property Graphs and RDF triples. Choose the model that fits your domain.

-   :material-vector-line:{ .lg .middle } **Vector Search**

    ---

    HNSW-based similarity search with quantization (Scalar, Binary, Product). Combine graph traversal with semantic similarity.

-   :material-memory:{ .lg .middle } **Embeddable**

    ---

    Embed directly into your application with zero external dependencies. Perfect for edge computing, desktop apps, and serverless environments.

-   :fontawesome-brands-rust:{ .lg .middle } **Pure Rust**

    ---

    Written entirely in safe Rust with no C dependencies. Memory-safe by design with fearless concurrency.

-   :material-shield-check:{ .lg .middle } **ACID Transactions**

    ---

    Full ACID compliance with MVCC-based snapshot isolation. Reliable transactions for production workloads.

-   :material-language-python:{ .lg .middle } **Multi-Language Bindings**

    ---

    Python (PyO3), Node.js/TypeScript (napi-rs), Go (CGO), and WebAssembly (wasm-bindgen). Use Grafeo from your language of choice.

-   :material-notebook:{ .lg .middle } **Notebook Widgets**

    ---

    Interactive graph and vector visualizations for Marimo, Jupyter, and other notebooks. Explore your data visually.

</div>

---

## Quick Start

=== "Python"

    ```bash
    uv add grafeo
    ```

    ```python
    import grafeo

    # Create an in-memory database
    db = grafeo.GrafeoDB()

    # Create nodes and edges
    db.execute("""
        INSERT (:Person {name: 'Alice', age: 30})
        INSERT (:Person {name: 'Bob', age: 25})
    """)

    db.execute("""
        MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
        INSERT (a)-[:KNOWS {since: 2024}]->(b)
    """)

    # Query the graph
    result = db.execute("""
        MATCH (p:Person)-[:KNOWS]->(friend)
        RETURN p.name, friend.name
    """)

    for row in result:
        print(f"{row['p.name']} knows {row['friend.name']}")
    ```

=== "Rust"

    ```bash
    cargo add grafeo
    ```

    ```rust
    use grafeo::GrafeoDB;

    fn main() -> Result<(), grafeo_common::utils::error::Error> {
        // Create an in-memory database
        let db = GrafeoDB::new_in_memory();

        // Create a session and execute queries
        let mut session = db.session();

        session.execute(r#"
            INSERT (:Person {name: 'Alice', age: 30})
            INSERT (:Person {name: 'Bob', age: 25})
        "#)?;

        session.execute(r#"
            MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
            INSERT (a)-[:KNOWS {since: 2024}]->(b)
        "#)?;

        // Query the graph
        let result = session.execute(r#"
            MATCH (p:Person)-[:KNOWS]->(friend)
            RETURN p.name, friend.name
        "#)?;

        for row in result.rows {
            println!("{:?}", row);
        }

        Ok(())
    }
    ```

---

## Features

### Dual Data Model Support

Grafeo supports both major graph data models with optimized storage for each:

=== "LPG (Labeled Property Graph)"

    - **Nodes** with labels and properties
    - **Edges** with types and properties
    - **Properties** supporting rich data types
    - Ideal for social networks, knowledge graphs, application data

=== "RDF (Resource Description Framework)"

    - **Triples**: subject-predicate-object statements
    - **SPO/POS/OSP indexes** for efficient querying
    - W3C standard compliance
    - Ideal for semantic web, linked data, ontologies

### Query Languages

Choose the query language that fits your needs:

| Language | Data Model | Style |
|----------|------------|-------|
| **GQL** (default) | LPG | ISO standard, declarative pattern matching |
| **Cypher** | LPG | Neo4j-compatible, ASCII-art patterns |
| **Gremlin** | LPG | Apache TinkerPop, traversal-based |
| **GraphQL** | LPG, RDF | Schema-driven, familiar to web developers |
| **SPARQL** | RDF | W3C standard for RDF queries |

=== "GQL"

    ```sql
    MATCH (me:Person {name: 'Alice'})-[:KNOWS]->(friend)-[:KNOWS]->(fof)
    WHERE fof <> me
    RETURN DISTINCT fof.name
    ```

=== "Cypher"

    ```cypher
    MATCH (me:Person {name: 'Alice'})-[:KNOWS]->(friend)-[:KNOWS]->(fof)
    WHERE fof <> me
    RETURN DISTINCT fof.name
    ```

=== "Gremlin"

    ```gremlin
    g.V().has('name', 'Alice').out('KNOWS').out('KNOWS').
      where(neq('me')).values('name').dedup()
    ```

=== "GraphQL"

    ```graphql
    {
      Person(name: "Alice") {
        friends { friends { name } }
      }
    }
    ```

=== "SPARQL"

    ```sparql
    SELECT DISTINCT ?fofName WHERE {
      ?me foaf:name "Alice" .
      ?me foaf:knows ?friend .
      ?friend foaf:knows ?fof .
      ?fof foaf:name ?fofName .
      FILTER(?fof != ?me)
    }
    ```

### Architecture Highlights

- **Push-based execution engine** with morsel-driven parallelism
- **Columnar storage** with type-specific compression
- **Cost-based query optimizer** with cardinality estimation
- **MVCC transactions** with snapshot isolation
- **Zone maps** for intelligent data skipping

---

## Installation

=== "Python"

    ```bash
    uv add grafeo
    ```

=== "Node.js"

    ```bash
    npm install @grafeo-db/js
    ```

=== "Go"

    ```bash
    go get github.com/GrafeoDB/grafeo/crates/bindings/go
    ```

=== "Rust"

    ```bash
    cargo add grafeo
    ```

=== "WASM"

    ```bash
    npm install @grafeo-db/wasm
    ```

---

## License

Grafeo is licensed under the [Apache-2.0 License](https://github.com/GrafeoDB/grafeo/blob/main/LICENSE).
