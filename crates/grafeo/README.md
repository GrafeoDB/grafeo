A pure-Rust, high-performance, embeddable graph database supporting both **Labeled Property Graph (LPG)** and **RDF** data models.

## Features

- **Dual data model support**: LPG and RDF with optimized storage for each
- **Multi-language queries**: GQL, Cypher, Gremlin, GraphQL, and SPARQL (all enabled by default)
- Embeddable with zero external dependencies
- Python bindings via PyO3
- In-memory and persistent storage modes
- MVCC transactions with snapshot isolation

## Query Language & Data Model Support

| Query Language | LPG | RDF | Status |
|----------------|-----|-----|--------|
| GQL (ISO/IEC 39075) | ✅ | — | Default |
| Cypher (openCypher 9.0) | ✅ | — | Default |
| Gremlin (Apache TinkerPop) | ✅ | — | Default |
| GraphQL | ✅ | ✅ | Default |
| SPARQL (W3C 1.1) | — | ✅ | Default |

Grafeo uses a modular translator architecture where query languages are parsed into ASTs, then translated to a unified logical plan that executes against the appropriate storage backend (LPG or RDF).

### Data Models

- **LPG (Labeled Property Graph)**: Nodes with labels and properties, edges with types and properties. Ideal for social networks, knowledge graphs, and application data.
- **RDF (Resource Description Framework)**: Triple-based storage (subject-predicate-object) with SPO/POS/OSP indexes. Ideal for semantic web, linked data, and ontology-based applications.

## Installation

```bash
cargo add grafeo
```

All query languages are enabled by default. For a minimal build with specific languages:

```bash
cargo add grafeo --no-default-features --features gql  # GQL only
```

## Quick Start

```rust
use grafeo::GrafeoDB;

fn main() -> Result<(), grafeo_common::utils::error::Error> {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    // Create nodes
    session.execute("INSERT (:Person {name: 'Alice', age: 30})")?;
    session.execute("INSERT (:Person {name: 'Bob', age: 25})")?;

    // Query
    let result = session.execute("MATCH (p:Person) RETURN p.name, p.age")?;
    for row in result.rows {
        println!("{:?}", row);
    }

    Ok(())
}
```

## License

Apache-2.0
