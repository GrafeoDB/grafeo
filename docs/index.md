---
title: Grafeo - High-Performance Graph Database
description: A high-performance, embeddable graph database with a Rust core and no required C dependencies. Python, Node.js, Go, C, C#, Dart and WebAssembly bindings. GQL (ISO standard) query language.
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

### A fast, lean, embeddable graph database built in Rust

[Get Started](getting-started/index.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/GrafeoDB/grafeo){ .md-button }
[Join Discord](https://discord.gg/jrgMD2Zj3){ .md-button }

</div>

[![Grafeo Playground](assets/playground.png)](https://grafeo.ai)

---

## Why Grafeo?

<div class="grid cards" markdown>

-   :material-lightning-bolt:{ .lg .middle } **High Performance**

    ---

    Fastest graph database in our [graph-bench](ecosystem/graph-bench.md) suite (includes [LDBC-inspired](https://ldbcouncil.org/benchmarks/snb/) workloads), both embedded and as a server, with a lower memory footprint than other in-memory databases. Built in Rust with vectorized execution, adaptive chunking and SIMD-optimized operations.

-   :material-database-search:{ .lg .middle } **Multi-Language Queries**

    ---

    GQL, Cypher, Gremlin, GraphQL, SPARQL and SQL/PGQ. Choose the query language that fits the project and expertise level.

-   :material-graph:{ .lg .middle } **LPG & RDF Support**

    ---

    Dual data model support for both Labeled Property Graphs and RDF triples. Choose the model that fits the domain.

-   :material-vector-line:{ .lg .middle } **Vector Search**

    ---

    HNSW-based similarity search with quantization (Scalar, Binary, Product). Combine graph traversal with semantic similarity.

-   :material-memory:{ .lg .middle } **Embedded or Standalone**

    ---

    Embed directly into applications with zero external dependencies, or run as a standalone server with REST API and web UI. From edge devices to production clusters.

-   :fontawesome-brands-rust:{ .lg .middle } **Rust Core**

    ---

    Core database engine written in Rust with no required C dependencies. Optional allocators (jemalloc/mimalloc) and TLS use C libraries for performance. Memory-safe by design with fearless concurrency.

-   :material-shield-check:{ .lg .middle } **ACID Transactions**

    ---

    Full ACID compliance with MVCC-based snapshot isolation. Reliable transactions for production workloads.

-   :material-language-python:{ .lg .middle } **Multi-Language Bindings**

    ---

    Python (PyO3), Node.js/TypeScript (napi-rs), Go (CGO), C (FFI), C# (.NET 8 P/Invoke), Dart (dart:ffi) and WebAssembly (wasm-bindgen). Use Grafeo from the language of choice.

-   :material-puzzle:{ .lg .middle } **Ecosystem**

    ---

    AI integrations (LangChain, LlamaIndex, MCP), interactive notebook widgets, browser-based graphs via WebAssembly, standalone server with web UI and benchmarking tools.

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
        INSERT (:Person {name: 'Alix', age: 30})
        INSERT (:Person {name: 'Gus', age: 25})
    """)

    db.execute("""
        MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'})
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

    fn main() -> Result<(), grafeo::Error> {
        // Create an in-memory database
        let db = GrafeoDB::new_in_memory();

        // Create a session and execute queries
        let mut session = db.session();

        session.execute(r#"
            INSERT (:Person {name: 'Alix', age: 30})
            INSERT (:Person {name: 'Gus', age: 25})
        "#)?;

        session.execute(r#"
            MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'})
            INSERT (a)-[:KNOWS {since: 2024}]->(b)
        "#)?;

        // Query the graph
        let result = session.execute(r#"
            MATCH (p:Person)-[:KNOWS]->(friend)
            RETURN p.name, friend.name
        "#)?;

        for row in result.rows() {
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

Choose the query language that fits the project:

| Language | Data Model | Style |
|----------|------------|-------|
| **GQL** (default) | LPG | ISO standard, declarative pattern matching |
| **Cypher** | LPG | Neo4j-compatible, ASCII-art patterns |
| **Gremlin** | LPG | Apache TinkerPop, traversal-based |
| **GraphQL** | LPG, RDF | Schema-driven, familiar to web developers |
| **SPARQL** | RDF | W3C standard for RDF queries |
| **SQL/PGQ** | LPG | SQL:2023 GRAPH_TABLE for SQL-native graph queries |

=== "GQL"

    ```sql
    MATCH (me:Person {name: 'Alix'})-[:KNOWS]->(friend)-[:KNOWS]->(fof)
    WHERE fof <> me
    RETURN DISTINCT fof.name
    ```

=== "Cypher"

    ```cypher
    MATCH (me:Person {name: 'Alix'})-[:KNOWS]->(friend)-[:KNOWS]->(fof)
    WHERE fof <> me
    RETURN DISTINCT fof.name
    ```

=== "Gremlin"

    ```gremlin
    g.V().has('name', 'Alix').out('KNOWS').out('KNOWS').
      where(neq('me')).values('name').dedup()
    ```

=== "GraphQL"

    ```graphql
    {
      Person(name: "Alix") {
        friends { friends { name } }
      }
    }
    ```

=== "SPARQL"

    ```sparql
    SELECT DISTINCT ?fofName WHERE {
      ?me foaf:name "Alix" .
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

=== "C#"

    ```bash
    dotnet add package GrafeoDB
    ```

=== "Dart"

    ```yaml
    # pubspec.yaml
    dependencies:
      grafeo: ^0.5.43
    ```

=== "WASM"

    ```bash
    npm install @grafeo-db/wasm
    ```

---

## License

Grafeo is licensed under the [Apache-2.0 License](https://github.com/GrafeoDB/grafeo/blob/main/LICENSE).
