A pure-Rust, high-performance, embeddable graph database supporting both **Labeled Property Graph (LPG)** and **RDF** data models.

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
