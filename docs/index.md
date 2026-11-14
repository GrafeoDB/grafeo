---
title: Grafeo documentation
description: Documentation for Grafeo, an embeddable graph database with a Rust core. Install, query, operate and extend it.
hide:
  - navigation
  - toc
---

# Grafeo documentation

Grafeo is an embeddable graph database with a Rust core. It stores labeled property graphs and RDF triples, and answers queries in GQL, Cypher, Gremlin, GraphQL, SPARQL and SQL/PGQ.

## Install

```bash
uv add grafeo
npm install @grafeo-db/js
cargo add grafeo
```

Other languages are covered under [Installation](get-started/install.md).

## A first query

```python
import grafeo

db = grafeo.GrafeoDB()

db.execute("INSERT (:Person {name: 'Alix'}), (:Person {name: 'Gus'})")
db.execute("""
    MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'})
    INSERT (a)-[:KNOWS {since: 2024}]->(b)
""")

result = db.execute("MATCH (p:Person)-[:KNOWS]->(f) RETURN p.name, f.name")
for row in result:
    print(f"{row['p.name']} knows {row['f.name']}")
```

The [quick start](get-started/quick-start.md) walks through the same steps in each language. Benchmark numbers are published with their method in [graph-bench](https://github.com/GrafeoDB/graph-bench).

## Sections

<div class="grid cards" markdown>

-   **[Get started](get-started/index.md)**

    Install, first graph, configuration, tutorials and upgrade notes.

-   **[Model & query](model-and-query/index.md)**

    The data model, the six query languages, vector and text search, algorithms.

-   **[Languages](languages/index.md)**

    Guides and API references for Python, Node.js, Rust, Go, WebAssembly, C, C# and Dart.

-   **[Operate](operate/persistence/index.md)**

    Persistence, transactions, deployment, observability and security.

-   **[Extend](extend/index.md)**

    Plugins, storage backends, custom functions and Rust extensions.

-   **[Internals](internals/index.md)**

    Architecture: crates, storage, execution, optimization, memory and transactions.

-   **[Project](project/ecosystem/index.md)**

    Ecosystem, contributing, versioning, API stability, roadmap and changelog.

</div>

Grafeo is licensed under the [Apache-2.0 License](https://github.com/GrafeoDB/grafeo/blob/main/LICENSE).
