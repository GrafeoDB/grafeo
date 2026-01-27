# Graphos

[![Crates.io](https://img.shields.io/crates/v/graphos.svg)](https://crates.io/crates/graphos)
[![PyPI](https://img.shields.io/pypi/v/pygraphos.svg)](https://pypi.org/project/pygraphos/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-graphos.tech-blue)](https://graphos.tech)

A pure-Rust, high-performance, embeddable graph database.

## Features

- **Labeled Property Graph (LPG)** data model
- **GQL** query language (ISO/IEC 39075) - enabled by default
- **Cypher** query language support (via feature flag)
- **SPARQL** query language support (via feature flag)
- Embeddable with zero external dependencies
- Python bindings via PyO3
- In-memory and persistent storage modes
- MVCC transactions with snapshot isolation

## Query Language & Data Model Support

| Query Language | Data Model | Status |
|----------------|------------|--------|
| GQL (ISO/IEC 39075) | LPG | ✅ Default |
| Cypher (openCypher 9.0) | LPG | ✅ Feature flag |
| SPARQL (W3C 1.1) | RDF | 🚧 Planned |
| Gremlin (TinkerPop) | LPG | 🚧 Planned |
| GraphQL | LPG, RDF | 🚧 Planned |

Graphos uses a modular architecture where query languages translate to a unified logical plan, then execute against the appropriate storage backend.

## Installation

### Rust

```bash
cargo add graphos-engine
```

With additional query languages:

```bash
cargo add graphos-engine --features cypher
cargo add graphos-engine --features full  # GQL + Cypher + SPARQL
```

### Python

```bash
pip install pygraphos
```

## Quick Start

### Python

```python
import graphos

# Create an in-memory database
db = graphos.GraphosDB()

# Or open/create a persistent database
# db = graphos.GraphosDB("/path/to/database")

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
use graphos_engine::GraphosDB;

fn main() {
    // Create an in-memory database
    let db = GraphosDB::new_in_memory();

    // Or open a persistent database
    // let db = GraphosDB::open("./my_database").unwrap();

    // Execute GQL queries
    db.execute("INSERT (:Person {name: 'Alice'})").unwrap();

    let result = db.execute("MATCH (p:Person) RETURN p.name").unwrap();
    for row in result.rows {
        println!("{:?}", row);
    }
}
```

## Documentation

Full documentation is available at [graphos.tech](https://graphos.tech).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

Apache-2.0
