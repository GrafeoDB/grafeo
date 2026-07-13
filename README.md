[![CI](https://github.com/GrafeoDB/grafeo/actions/workflows/ci.yml/badge.svg)](https://github.com/GrafeoDB/grafeo/actions/workflows/ci.yml)
[![grafeo.dev](https://github.com/GrafeoDB/grafeo/actions/workflows/docs.yml/badge.svg)](https://grafeo.dev)
[![codecov](https://codecov.io/gh/GrafeoDB/grafeo/graph/badge.svg)](https://codecov.io/gh/GrafeoDB/grafeo)
[![CodSpeed Badge](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/GrafeoDB/grafeo?utm_source=badge)
[![Crates.io](https://img.shields.io/crates/v/grafeo.svg?color=00ADD8)](https://crates.io/crates/grafeo)
[![PyPI](https://img.shields.io/pypi/v/grafeo.svg?color=00ADD8)](https://pypi.org/project/grafeo/)
[![npm](https://img.shields.io/npm/v/@grafeo-db/js.svg?color=00ADD8)](https://www.npmjs.com/package/@grafeo-db/js)
[![wasm](https://img.shields.io/npm/v/@grafeo-db/wasm.svg?label=wasm&color=00ADD8)](https://www.npmjs.com/package/@grafeo-db/wasm)
[![NuGet](https://img.shields.io/nuget/v/Grafeo.svg?color=00ADD8)](https://www.nuget.org/packages/Grafeo)
[![pub.dev](https://img.shields.io/pub/v/grafeo.svg?color=00ADD8)](https://pub.dev/packages/grafeo)
[![Go](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fproxy.golang.org%2Fgithub.com%2F!grafe!o!d!b%2Fgrafeo%2Fcrates%2Fbindings%2Fgo%2F%40latest&query=%24.Version&label=go&color=00ADD8&logo=go&logoColor=white)](https://pkg.go.dev/github.com/GrafeoDB/grafeo/crates/bindings/go)
[![Web](https://img.shields.io/npm/v/@grafeo-db/web.svg?label=web&color=7c4dff)](https://www.npmjs.com/package/@grafeo-db/web)
[![Server](https://img.shields.io/github/v/release/GrafeoDB/grafeo-server?label=server&color=7c4dff)](https://github.com/GrafeoDB/grafeo-server)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![MSRV](https://img.shields.io/badge/MSRV-1.91.1-blue)](https://www.rust-lang.org)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org)
[![Discord](https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white)](https://discord.gg/jrgMD2Zj3)

# Grafeo

Grafeo is a graph database built in Rust from the ground up for speed and low memory use. It runs embedded as a library or as a standalone server, with in-memory or persistent storage and full ACID transactions.

In our [graph-bench](https://github.com/GrafeoDB/graph-bench) suite (which includes workloads inspired by the [LDBC Social Network Benchmark](https://ldbcouncil.org/benchmarks/snb/)), Grafeo is the fastest tested graph database in both embedded and server configurations, while using a fraction of the memory of some of the alternatives.

[![Grafeo Playground](docs/assets/playground.png)](https://playground.grafeo.dev)

Grafeo supports both **Labeled Property Graph (LPG)** and **Resource Description Framework (RDF)** data models and all major query languages.

<details>
<summary><strong>Features</strong></summary>

### Core Capabilities

- **Dual data model support**: LPG and RDF with optimized storage for each
- **Multi-language queries**: GQL, Cypher, Gremlin, GraphQL, SPARQL and SQL/PGQ
- Embeddable with zero external dependencies: no JVM, no Docker, no external processes
- **Multi-language bindings**: Python (PyO3), Node.js/TypeScript (napi-rs), Go (CGO), C (FFI), C# (.NET 8 P/Invoke), Dart (dart:ffi), WebAssembly (wasm-bindgen)
- In-memory and persistent storage modes
- MVCC transactions with snapshot isolation

### Query Languages

- **GQL** (ISO/IEC 39075) with Unicode identifiers per spec
- **Cypher** (openCypher 9.0)
- **Gremlin** (Apache TinkerPop)
- **GraphQL**
- **SPARQL** (W3C 1.1) with SHACL validation and Ring Index WCOJ planner
- **SQL/PGQ** (SQL:2023)
- **EXPLAIN / EXPLAIN ANALYZE** across all six languages

### Vector Search & AI

- **Vector as a first-class type**: `Value::Vector(Arc<[f32]>)` stored alongside graph data
- **HNSW index**: O(log n) approximate nearest neighbor search with tunable recall
- **Distance functions**: Cosine, Euclidean, Dot Product, Manhattan (SIMD-accelerated: AVX2, SSE, NEON)
- **Vector quantization**: Scalar (f32 → u8), Binary (1-bit) and Product Quantization (8-32x compression)
- **BM25 text search**: Full-text inverted index with Unicode tokenizer and stop word removal
- **Hybrid search**: Combined text + vector search with Reciprocal Rank Fusion (RRF) or weighted fusion
- **Change data capture**: Before/after property snapshots for audit trails and history tracking
- **Hybrid graph+vector queries**: Combine graph traversals with vector similarity in GQL and SPARQL
- **Memory-mapped storage**: Disk-backed vectors with LRU cache for large datasets
- **Batch operations**: Parallel multi-query search via rayon

### Performance Features

- **Push-based vectorized execution** with adaptive chunk sizing
- **Morsel-driven parallelism** with auto-detected thread count
- **Block-STM conflict partitioning** for parallel transaction re-execution
- **Columnar storage** with dictionary, delta and RLE compression
- **Cost-based optimizer** with DPccp join ordering and histograms
- **Zone maps** for intelligent data skipping (including vector zone maps)
- **Adaptive query execution** with runtime re-optimization
- **Transparent spilling** for out-of-core processing
- **Streaming execution** for large result sets without buffering
- **Bloom filters** for efficient membership tests
- **Writable layered compact store**: columnar base with mutable overlay, `recompact()` to merge

### Security

- **Encryption at rest** (`encryption` feature): AES-256-GCM for WAL records and `.grafeo` sections, password-based (Argon2id) or raw-key setup
- **Role-based access control**: `Admin`, `ReadWrite`, `ReadOnly` roles enforced across all six query languages
- **Per-graph access grants**: scope an identity's access to specific named graphs
- **SHACL validation** (`shacl` feature): W3C Shapes Constraint Language with all 28 Core constraint types and SHACL-SPARQL
- **Resource limits**: query timeouts, property size caps, HNSW `max_elements` bound

### Operations & Observability

- **Incremental backup and point-in-time recovery**: `backup_full`, `backup_incremental`, `restore_to_epoch`
- **Prometheus metrics** (`metrics` feature): query, transaction, session, cache, and GC counters with text export
- **Change data capture**: before/after property snapshots with epoch-bounded retention
- **Async storage** (`async-storage` feature): non-blocking WAL and snapshot I/O via tokio
- **Tracing** (`tracing` feature): opt-in observability spans and events
- **Bulk export**: Arrow IPC, Polars, pandas, GEXF (Gephi), GraphML (Cytoscape, yEd, NetworkX)
- **Bulk import**: CSV, JSONL, TSV, Matrix Market (MMIO), Turtle, N-Triples with streaming loaders

</details>

<details>
<summary><strong>Benchmarks</strong></summary>

Tested with [graph-bench](https://github.com/GrafeoDB/graph-bench), which includes workloads inspired by the [LDBC Social Network Benchmark](https://ldbcouncil.org/benchmarks/snb/). These are not official LDBC Benchmark results (see [disclaimer](https://github.com/GrafeoDB/graph-bench#ldbc-disclaimer)).

**Embedded** (SF0.1, in-process):

| Database | SNB Interactive | Memory | Graph Analytics | Memory |
|----------|---------------:|-------:|----------------:|-------:|
| **Grafeo** | **2,904 ms** | 136 MB | **0.4 ms** | 43 MB |
| LadybugDB(Kuzu) | 5,333 ms | 4,890 MB | 225 ms | 250 MB |
| FalkorDB Lite | 7,454 ms | 156 MB | 89 ms | 88 MB |

**Server** (SF0.1, over network):

| Database | SNB Interactive | Graph Analytics |
|----------|---------------:|----------------:|
| **Grafeo Server** | **730 ms** | **15 ms** |
| Memgraph | 4,113 ms | 19 ms |
| Neo4j | 6,788 ms | 253 ms |
| ArangoDB | 40,043 ms | 22,739 ms |

Full results: [embedded](https://github.com/GrafeoDB/graph-bench/blob/main/RESULTS_EMBEDDED.md) | [server](https://github.com/GrafeoDB/graph-bench/blob/main/RESULTS_SERVER.md)

</details>

## Query Language & Data Model Support

| Query Language | LPG | RDF |
|----------------|-----|-----|
| GQL | ✅ | - |
| Cypher | ✅ | - |
| GraphQL | ✅ | ✅ | 
| Gremlin | ✅ | - |
| SPARQL | - | ✅ |
| SQL/PGQ | ✅ | - |

Grafeo uses a modular translator architecture where query languages are parsed into ASTs, then translated to a unified logical plan that executes against the appropriate storage backend (LPG or RDF).

### Data Models

- **LPG (Labeled Property Graph)**: Nodes with labels and properties, edges with types and properties. Ideal for social networks, knowledge graphs and application data.
- **RDF (Resource Description Framework)**: Triple-based storage (subject-predicate-object) with SPO/POS/OSP indexes. Ideal for semantic web, linked data and ontology-based applications.

## Installation

### Rust

```bash
cargo add grafeo
```

Grafeo uses persona-based feature profiles that describe use cases. Compose them freely:

```bash
# Default: LPG with GQL, AI, algorithms, parallel execution
cargo add grafeo

# Compose profiles for your use case
cargo add grafeo --features rdf          # Add RDF/SPARQL support
cargo add grafeo --features analytics    # Add graph algorithms
cargo add grafeo --features ai           # Add vector/text/hybrid search
cargo add grafeo --features enterprise   # Full feature set

# Or use individual flags
cargo add grafeo --no-default-features --features gql       # Minimal: GQL only
cargo add grafeo --no-default-features --features languages  # All query languages
cargo add grafeo --features embed                            # ONNX embeddings (opt-in, ~17MB)
```

| Profile | Contents | Use case |
|---------|----------|----------|
| `lpg` | GQL, AI, algorithms, parallel | Default for libraries and apps |
| `rdf` | SPARQL, triple-store, ring-index | Knowledge graphs, linked data |
| `analytics` | Algorithms, parallel | Graph analytics pipelines |
| `ai` | Vector, text, hybrid search, CDC | RAG, semantic search |
| `edge` | GQL, compact, regex-lite | WASM, resource-constrained |
| `enterprise` | Metrics, tracing, async I/O | Platform operators, observability |

### Node.js / TypeScript

```bash
npm install @grafeo-db/js
```

### Go

```bash
go get github.com/GrafeoDB/grafeo/crates/bindings/go
```

### WebAssembly

```bash
npm install @grafeo-db/wasm
```

### C# / .NET

```bash
dotnet add package Grafeo
```

### Dart

```yaml
# pubspec.yaml
dependencies:
  grafeo: ^0.5.42
```

### Python

```bash
pip install grafeo
# or with uv
uv add grafeo
```

With CLI support:

```bash
pip install grafeo[cli]
# or with uv
uv add grafeo[cli]
```

## Quick Start

### Node.js / TypeScript

```js
const { GrafeoDB } = require('@grafeo-db/js');

// Create an in-memory database
const db = await GrafeoDB.create();

// Or open a persistent database
// const db = await GrafeoDB.create({ path: './my-graph.db' });

// Create nodes and relationships
await db.execute("INSERT (:Person {name: 'Alix', age: 30})");
await db.execute("INSERT (:Person {name: 'Gus', age: 25})");
await db.execute(`
    MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'})
    INSERT (a)-[:KNOWS {since: 2020}]->(b)
`);

// Query the graph
const result = await db.execute(`
    MATCH (p:Person)-[:KNOWS]->(friend)
    RETURN p.name, friend.name
`);
console.log(result.toArray());

await db.close();
```

### Python

```python
import grafeo

# Create an in-memory database
db = grafeo.GrafeoDB()

# Or open/create a persistent database
# db = grafeo.GrafeoDB("/path/to/database")

# Create nodes using GQL
db.execute("INSERT (:Person {name: 'Alix', age: 30})")
db.execute("INSERT (:Person {name: 'Gus', age: 25})")

# Create a relationship
db.execute("""
    MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'})
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
node = db.create_node(["Person"], {"name": "Harm"})
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

# Named graphs and schemas
db.create_graph("social")
db.set_graph("social")
db.list_graphs()               # ['social']
db.set_schema("v1")
db.current_schema()            # 'v1'

# Graph projections (filtered virtual views)
db.create_projection("people", node_labels=["Person"], edge_types=["KNOWS"])
db.list_projections()          # ['people']
db.drop_projection("people")

# Data import
db.import_csv("users.csv", "Person", headers=True)
db.import_jsonl("events.jsonl", "Event")

# Backup and restore
db.backup_full("/backups/full")
db.backup_incremental("/backups/incr")
GrafeoDB.restore_to_epoch("/backups/full", epoch=100, output_path="./restored")

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
    db.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    let result = db.execute("MATCH (p:Person) RETURN p.name").unwrap();
    for row in result.rows() {
        println!("{:?}", row);
    }
}
```

### Vector Search

```python
import grafeo

db = grafeo.GrafeoDB()

# Store documents with embeddings
db.execute("""INSERT (:Document {
    title: 'Graph Databases',
    embedding: vector([0.1, 0.8, 0.3, 0.5])
})""")
db.execute("""INSERT (:Document {
    title: 'Vector Search',
    embedding: vector([0.2, 0.7, 0.4, 0.6])
})""")
db.execute("""INSERT (:Document {
    title: 'Cooking Recipes',
    embedding: vector([0.9, 0.1, 0.2, 0.1])
})""")

# Create an HNSW index for fast approximate search
db.execute("""
    CREATE VECTOR INDEX doc_idx ON :Document(embedding)
    DIMENSION 4 METRIC 'cosine'
""")

# Find similar documents using cosine similarity
query = [0.15, 0.75, 0.35, 0.55]
result = db.execute(f"""
    MATCH (d:Document)
    WHERE cosine_similarity(d.embedding, vector({query})) > 0.9
    RETURN d.title, cosine_similarity(d.embedding, vector({query})) AS score
    ORDER BY score DESC
""")
for row in result:
    print(row)  # Graph Databases, Vector Search (Cooking Recipes filtered out)
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

# Data import
grafeo import csv data.csv --path ./mydb --label Person
grafeo import jsonl events.jsonl --path ./mydb --label Event

# Backup & restore
grafeo backup create ./mydb -o backup
grafeo backup full ./mydb -o /backups/full
grafeo backup incremental ./mydb -o /backups/incr
grafeo backup restore-to-epoch /backups/full --epoch 100 -o ./restored
grafeo backup status /backups/full

# Data export
grafeo data dump ./mydb -o ./export/
grafeo data dump ./mydb -o graph.gexf --export-format gexf
grafeo data dump ./mydb -o graph.graphml --export-format graphml

# WAL management
grafeo wal status ./mydb
grafeo wal checkpoint ./mydb

# Output formats
grafeo info ./mydb --format json  # Machine-readable JSON
grafeo info ./mydb --format table # Human-readable table (default)
```

## Ecosystem

| Project | Description |
|---------|-------------|
| [**grafeo-server**](https://github.com/GrafeoDB/grafeo-server) | HTTP server & web UI: REST API, transactions, single binary (~40MB Docker image) |
| [**grafeo-web**](https://github.com/GrafeoDB/grafeo-web) | Browser-based Grafeo via WebAssembly with IndexedDB persistence |
| [**grafeo-loro**](https://github.com/OndeHQ/grafeo-loro) | Local-first graph database with invisible consensus: Loro CRDT sync + Grafeo execution in one process, offline-first with mathematically impossible conflicts |
| [**gwp**](https://github.com/GrafeoDB/gql-wire-protocol) | GQL Wire Protocol: gRPC wire protocol for GQL (ISO/IEC 39075) with client bindings in 5 languages |
| [**boltr**](https://github.com/GrafeoDB/boltr) | Bolt Wire Protocol: pure Rust Bolt v5.x implementation for Neo4j driver compatibility |
| [**grafeo-langchain**](https://github.com/GrafeoDB/grafeo-langchain) | LangChain integration: graph store, vector store, Graph RAG retrieval |
| [**grafeo-llamaindex**](https://github.com/GrafeoDB/grafeo-llamaindex) | LlamaIndex integration: PropertyGraphStore, vector search, knowledge graphs |
| [**grafeo-mcp**](https://github.com/GrafeoDB/grafeo-mcp) | Model Context Protocol server: expose Grafeo as tools for LLM agents |
| [**grafeo-memory**](https://github.com/GrafeoDB/grafeo-memory) | AI memory layer for LLM applications: fact extraction, deduplication, semantic search |
| [**anywidget-graph**](https://github.com/GrafeoDB/anywidget-graph) | Interactive graph visualization for Python notebooks (Marimo, Jupyter, VS Code, Colab) |
| [**anywidget-vector**](https://github.com/GrafeoDB/anywidget-vector) | 3D vector/embedding visualization for Python notebooks |
| [**playground**](https://grafeo.ai) | Interactive browser playground: query in 6 languages, visualize graphs, explore schemas |
| [**graph-bench**](https://github.com/GrafeoDB/graph-bench) | Benchmark suite comparing graph databases across 65 LDBC-inspired and custom benchmarks |
| [**ann-benchmarks**](https://github.com/GrafeoDB/ann-benchmarks) | Fork of ann-benchmarks with a Grafeo HNSW adapter for vector search benchmarking |

## Documentation

Full documentation is available at [grafeo.dev](https://grafeo.dev).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## Sponsors

Thank you to the people and organizations supporting Grafeo's development:

- [**Thibaut Mélen**](https://github.com/ThibautMelen) from [Supernovae](https://github.com/supernovae-st)

## Acknowledgments

Grafeo's execution engine draws inspiration from:

- [DuckDB](https://duckdb.org/), vectorized push-based execution, morsel-driven parallelism
- [Kuzu](https://github.com/kuzudb/kuzu), CSR-based adjacency indexing, factorized query processing

## License

Apache-2.0
