A high-performance, embeddable graph database with a Rust core and no required C dependencies, supporting both **Labeled Property Graph (LPG)** and **RDF** data models. Optional allocators (jemalloc/mimalloc) and TLS use C libraries for performance.

## Features

- **Dual data model support**: LPG and RDF with optimized storage for each
- **Multi-language queries**: GQL, Cypher, Gremlin, GraphQL, SPARQL and SQL/PGQ
- Embeddable with zero external dependencies
- Multi-language bindings: Python (PyO3), Node.js (napi-rs), Go (CGO), C (FFI), C# (.NET P/Invoke), Dart (dart:ffi), WebAssembly (wasm-bindgen)
- In-memory and persistent storage modes
- MVCC transactions with snapshot isolation

## Query Language & Data Model Support

| Query Language | LPG | RDF | Feature |
|----------------|-----|-----|---------|
| GQL (ISO/IEC 39075) | ✅ | - | `gql`: in `lpg`, `rdf`, `edge` and the default |
| Cypher (openCypher 9.0) | ✅ | - | `cypher`: in `lpg` |
| Gremlin (Apache TinkerPop) | ✅ | - | `gremlin`: in `lpg` |
| GraphQL | ✅ | ✅ | `graphql`: in `rdf` |
| SPARQL (W3C 1.1) | - | ✅ | `sparql`: in `rdf` |
| SQL/PGQ (SQL:2023) | ✅ | - | `sql-pgq`: in `lpg` |

The `languages` group enables all six.

Grafeo uses a modular translator architecture where query languages are parsed into ASTs, then translated to a unified logical plan that executes against the appropriate storage backend (LPG or RDF).

### Data Models

- **LPG (Labeled Property Graph)**: Nodes with labels and properties, edges with types and properties. Ideal for social networks, knowledge graphs and application data.
- **RDF (Resource Description Framework)**: Triple-based storage (subject-predicate-object) with SPO/POS/OSP indexes. Ideal for semantic web, linked data and ontology-based applications.

## Installation

```bash
cargo add grafeo
```

Features come in persona profiles that describe use cases. Compose them freely:

| Profile | Contents | Use case |
|---------|----------|----------|
| `lpg` | GQL, Cypher, Gremlin, SQL/PGQ, storage (write-ahead log, database file, spill, mmap) | Graph applications |
| `rdf` | RDF triple store, GQL, SPARQL, GraphQL, SHACL, storage | Knowledge graphs, linked data |
| `analytics` | Graph algorithms, vector, text and hybrid search, JSON Lines and Parquet import | Graph analytics pipelines |
| `ai` | Vector, text and hybrid search, change data capture | RAG, semantic search, agent memory |
| `edge` | GQL, lightweight regex | WebAssembly, resource-constrained |
| `enterprise` | Metrics, tracing, async storage (deprecated) | Platform operators, observability |

```bash
cargo add grafeo --no-default-features --features lpg,ai   # LPG application with search
cargo add grafeo --no-default-features --features rdf      # RDF, SPARQL and SHACL
cargo add grafeo --no-default-features --features lpg,rdf,analytics,ai  # Every language and model
cargo add grafeo --no-default-features --features edge     # Minimal, for WebAssembly
cargo add grafeo --features embed                          # ONNX embeddings (opt-in, about 17 MB)
```

The default, `cargo add grafeo` alone, is the deprecated `embedded` profile (GQL, vector, text and hybrid search, change data capture, algorithms, parallel execution, the `storage` group (database file, write-ahead log, disk spilling, memory-mapped reads) and Arrow export), removed in 0.7.0. It includes `storage` since 0.6.4, so a commit survives a crash and the query memory limit holds.

See [Installation](https://docs.grafeo.dev/get-started/install/) and [API Stability](https://docs.grafeo.dev/project/stability/).

## Quick Start

```rust
use grafeo::GrafeoDB;

fn main() -> Result<(), grafeo::Error> {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    // Create nodes
    session.execute("INSERT (:Person {name: 'Alix', age: 30})")?;
    session.execute("INSERT (:Person {name: 'Gus', age: 25})")?;

    // Query
    let result = session.execute("MATCH (p:Person) RETURN p.name, p.age")?;
    for row in result.rows() {
        println!("{:?}", row);
    }

    Ok(())
}
```

## License

Apache-2.0
