[![CI](https://github.com/GrafeoDB/grafeo/actions/workflows/ci.yml/badge.svg)](https://github.com/GrafeoDB/grafeo/actions/workflows/ci.yml)
[![Docs](https://github.com/GrafeoDB/grafeo/actions/workflows/docs.yml/badge.svg)](https://github.com/GrafeoDB/grafeo/actions/workflows/docs.yml)
[![codecov](https://codecov.io/gh/GrafeoDB/grafeo/graph/badge.svg)](https://codecov.io/gh/GrafeoDB/grafeo)
[![Crates.io](https://img.shields.io/crates/v/grafeo.svg)](https://crates.io/crates/grafeo)
[![PyPI](https://img.shields.io/pypi/v/grafeo.svg)](https://pypi.org/project/grafeo/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-grafeo.dev-blue)](https://grafeo.dev)
[![MSRV](https://img.shields.io/badge/MSRV-1.91.1-blue)](https://www.rust-lang.org)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org)

# Grafeo

Grafeo is a pure-Rust, high-performance graph database that can be embedded as a library or run as a standalone database, with optional in-memory or persistent storage. Grafeo supports both **Labeled Property Graph (LPG)** and **Resource Description Framework (RDF)** graph data models and all major query languages. 

## Features

### Core Capabilities

- **Dual data model support**: LPG and RDF with optimized storage for each
- **Multi-language queries**: GQL, Cypher, Gremlin, GraphQL, and SPARQL
- Embeddable with zero external dependencies
- Python bindings via PyO3
- In-memory and persistent storage modes
- MVCC transactions with snapshot isolation

### Query Languages

- **GQL** (ISO/IEC 39075)
- **Cypher** (openCypher 9.0)
- **Gremlin** (Apache TinkerPop)
- **GraphQL** (September 2025)
- **SPARQL** (W3C 1.1)

### Performance Features

- **Push-based vectorized execution** with adaptive chunk sizing
- **Morsel-driven parallelism** with auto-detected thread count
- **Columnar storage** with dictionary, delta, and RLE compression
- **Cost-based optimizer** with DPccp join ordering and histograms
- **Zone maps** for intelligent data skipping
- **Adaptive query execution** with runtime re-optimization
- **Transparent spilling** for out-of-core processing
- **Bloom filters** for efficient membership tests

## Query Language & Data Model Support

| Query Language | LPG | RDF |
|----------------|-----|-----|
| GQL | ✅ | — |
| Cypher | ✅ | — |
| Gremlin | ✅ | — |
| GraphQL | ✅ | ✅ | 
| SPARQL | — | ✅ |

Grafeo uses a modular translator architecture where query languages are parsed into ASTs, then translated to a unified logical plan that executes against the appropriate storage backend (LPG or RDF).

### Data Models

- **LPG (Labeled Property Graph)**: Nodes with labels and properties, edges with types and properties. Ideal for social networks, knowledge graphs, and application data.
- **RDF (Resource Description Framework)**: Triple-based storage (subject-predicate-object) with SPO/POS/OSP indexes. Ideal for semantic web, linked data, and ontology-based applications.

## Installation

### Rust

```bash
cargo add grafeo
```

All query languages (GQL, Cypher, Gremlin, GraphQL, SPARQL) are enabled by default. To disable specific languages:

```bash
cargo add grafeo --no-default-features --features gql,cypher
```

### Python

```bash
uv add grafeo
```

With CLI support:

```bash
uv add grafeo[cli]
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

# Manage labels
db.add_node_label(node.id, "Employee")     # Add a label
db.remove_node_label(node.id, "Contractor") # Remove a label
labels = db.get_node_labels(node.id)        # Get all labels
```

### Admin APIs (Python)

```python
# Database inspection
db.info()           # Overview: mode, counts, persistence
db.detailed_stats() # Memory usage, index counts
db.schema()         # Labels, edge types, property keys
db.validate()       # Integrity check

# Persistence control
db.save("/path/to/backup")    # Save to disk
db.to_memory()                # Create in-memory copy
GrafeoDB.open_in_memory(path) # Load as in-memory

# WAL management
db.wal_status()      # WAL info
db.wal_checkpoint()  # Force checkpoint
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

## Command-Line Interface

Optional admin CLI for operators and DevOps:

```bash
# Install with CLI support
uv add grafeo[cli]

# Inspection
grafeo info ./mydb              # Overview: counts, size, mode
grafeo stats ./mydb             # Detailed statistics
grafeo schema ./mydb            # Labels, edge types, property keys
grafeo validate ./mydb          # Integrity check

# Backup & restore
grafeo backup create ./mydb -o backup
grafeo backup restore backup ./copy --force

# WAL management
grafeo wal status ./mydb
grafeo wal checkpoint ./mydb

# Output formats
grafeo info ./mydb --format json  # Machine-readable JSON
grafeo info ./mydb --format table # Human-readable table (default)
```

## Documentation

Full documentation is available at [grafeo.dev](https://grafeo.dev).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## Acknowledgments

Grafeo's executiong engine draws inspiration from:

- [DuckDB](https://duckdb.org/), vectorized push-based execution, morsel-driven parallelism
- [LadybugDB](https://github.com/LadybugDB/ladybug), CSR-based adjacency indexing, factorized query processing

## License

Apache-2.0
