# Grafeo

[![CI](https://github.com/StevenBtw/grafeo/actions/workflows/ci.yml/badge.svg)](https://github.com/StevenBtw/grafeo/actions/workflows/ci.yml)
[![Docs](https://github.com/StevenBtw/grafeo/actions/workflows/docs.yml/badge.svg)](https://github.com/StevenBtw/grafeo/actions/workflows/docs.yml)
[![codecov](https://codecov.io/gh/StevenBtw/grafeo/graph/badge.svg)](https://codecov.io/gh/StevenBtw/grafeo)
[![Crates.io](https://img.shields.io/crates/v/grafeo.svg)](https://crates.io/crates/grafeo)
[![PyPI](https://img.shields.io/pypi/v/grafeo.svg)](https://pypi.org/project/grafeo/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-grafeo.tech-blue)](https://grafeo.tech)
[![MSRV](https://img.shields.io/badge/MSRV-1.91.1-blue)](https://www.rust-lang.org)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org)

A pure-Rust, high-performance, embeddable graph database supporting both **Labeled Property Graph (LPG)** and **Resource Description Framework (RDF)** graph data models.

## Features

- **Dual data model support**: LPG and RDF with optimized storage for each
- **Multi-language queries**: GQL, Cypher, Gremlin, GraphQL, and SPARQL
- **GQL** (ISO/IEC 39075) - enabled by default
- **Cypher** (openCypher 9.0) - via feature flag
- **Gremlin** (Apache TinkerPop) - via feature flag
- **GraphQL** - via feature flag, supports both LPG and RDF
- **SPARQL** (W3C 1.1) - via feature flag for RDF queries
- Embeddable with zero external dependencies
- Python bindings via PyO3
- In-memory and persistent storage modes
- MVCC transactions with snapshot isolation

## Query Language & Data Model Support

| Query Language | LPG | RDF | Status |
|----------------|-----|-----|--------|
| GQL (ISO/IEC 39075) | ✅ | — | Default |
| Cypher (openCypher 9.0) | ✅ | — | Feature flag |
| Gremlin (Apache TinkerPop) | ✅ | — | Feature flag |
| GraphQL | ✅ | ✅ | Feature flag |
| SPARQL (W3C 1.1) | — | ✅ | Feature flag |

Grafeo uses a modular translator architecture where query languages are parsed into ASTs, then translated to a unified logical plan that executes against the appropriate storage backend (LPG or RDF).

### Data Models

- **LPG (Labeled Property Graph)**: Nodes with labels and properties, edges with types and properties. Ideal for social networks, knowledge graphs, and application data.
- **RDF (Resource Description Framework)**: Triple-based storage (subject-predicate-object) with SPO/POS/OSP indexes. Ideal for semantic web, linked data, and ontology-based applications.

## Installation

### Rust

```bash
cargo add grafeo
```

With additional query languages:

```bash
cargo add grafeo --features cypher   # Add Cypher support
cargo add grafeo --features gremlin  # Add Gremlin support
cargo add grafeo --features graphql  # Add GraphQL support
cargo add grafeo --features full     # All query languages
```

### Python

```bash
uv add grafeo
```

## Quick Start

### Python

```python
import grafeo

# Create an in-memory database
db = grafeo.GrafeoDB()

# Or open/create a persistent database
# db = grafeo.GrafeoDB("/path/to/database")

# Create nodes using GQL
db.execute("INSERT (:Person {name: 'Alice', age: 30})")
db.execute("INSERT (:Person {name: 'Bob', age: 25})")

# Create a relationship
db.execute("""
    MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
    INSERT (a)-[:KNOWS {since: 2020}]->(b)
""")

# Query the graph
result = db.execute("""
    MATCH (p:Person)-[:KNOWS]->(friend)
    RETURN p.name, friend.name
""")

for row in result:
    print(row)

# Or use the direct API
node = db.create_node(["Person"], {"name": "Carol"})
print(f"Created node with ID: {node.id}")
```

### Rust

```rust
use grafeo::GrafeoDB;

fn main() {
    // Create an in-memory database
    let db = GrafeoDB::new_in_memory();

    // Or open a persistent database
    // let db = GrafeoDB::open("./my_database").unwrap();

    // Execute GQL queries
    db.execute("INSERT (:Person {name: 'Alice'})").unwrap();

    let result = db.execute("MATCH (p:Person) RETURN p.name").unwrap();
    for row in result.rows {
        println!("{:?}", row);
    }
}
```

## Documentation

Full documentation is available at [grafeo.dev](https://grafeo.dev).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

Apache-2.0
