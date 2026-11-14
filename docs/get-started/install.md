---
title: Installation
description: Install Grafeo for Python, Node.js, Go, C#, Dart, Rust or WebAssembly.
---

# Installation

Grafeo supports Python, Node.js/TypeScript, Go, C#, Dart, Rust and WebAssembly. Choose the installation method for the preferred language.

## Python

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer:

```bash
uv add grafeo
```

### Using pip (alternative)

```bash
pip install grafeo  # If uv is not available
```

### Verify Installation

```python
import grafeo

# Print version
print(grafeo.__version__)

# Create an in-memory test database
db = grafeo.GrafeoDB()
print("Grafeo installed successfully!")

# Or create a persistent database on disk
db = grafeo.GrafeoDB("/path/to/my_graph.db")
```

### Platform Support

| Platform | Architecture | Support |
|----------|--------------|---------|
| Linux    | x86_64       | :material-check: Full |
| Linux    | aarch64      | :material-check: Full |
| macOS    | x86_64       | :material-check: Full |
| macOS    | arm64 (M1/M2)| :material-check: Full |
| Windows  | x86_64       | :material-check: Full |

## Node.js / TypeScript

```bash
npm install @grafeo-db/js
```

### Verify Installation

```js
import { GrafeoDB } from '@grafeo-db/js';

const db = GrafeoDB.create();
console.log('Grafeo installed successfully!');
db.close();
```

## Go

```bash
go get github.com/GrafeoDB/grafeo/crates/bindings/go
```

### Verify Installation

```go
package main

import (
    "fmt"
    grafeo "github.com/GrafeoDB/grafeo/crates/bindings/go"
)

func main() {
    db, _ := grafeo.OpenInMemory()
    defer db.Close()
    fmt.Println("Grafeo installed successfully!")
}
```

## WebAssembly

```bash
npm install @grafeo-db/wasm
```

### Verify Installation

```js
import init, { Database } from '@grafeo-db/wasm';

await init();
const db = new Database();
console.log('Grafeo WASM installed successfully!');
```

## C# / .NET

```bash
dotnet add package Grafeo
```

### Verify Installation

```csharp
using Grafeo;

using var db = GrafeoDB.Memory();
Console.WriteLine("Grafeo installed successfully!");
```

## Dart

Add to `pubspec.yaml`:

```yaml
dependencies:
  grafeo: ^0.6.2
```

### Verify Installation

```dart
import 'package:grafeo/grafeo.dart';

void main() {
  final db = GrafeoDB.memory();
  print('Grafeo installed successfully!');
  db.close();
}
```

## Rust

### Using Cargo

Add Grafeo to the project:

```bash
cargo add grafeo
```

Or add it manually to `Cargo.toml`:

```toml
[dependencies]
grafeo = "0.6"
```

### Feature Flags

Grafeo uses **persona-based feature profiles** that describe use cases rather than deployment targets. Compose them freely to match your needs:

```toml
[dependencies]
# Default: the deprecated `embedded` profile (GQL, search, algorithms, parallel
# execution, the database file and Arrow export), without a write-ahead log
grafeo = "0.6"

# LPG application with search
grafeo = { version = "0.6", default-features = false, features = ["lpg", "ai"] }

# Add RDF, SPARQL and SHACL
grafeo = { version = "0.6", features = ["rdf"] }

# Add graph algorithms, search and bulk import
grafeo = { version = "0.6", features = ["analytics"] }

# Every language and model
grafeo = { version = "0.6", default-features = false, features = ["lpg", "rdf", "analytics", "ai"] }

# Minimal: GQL only
grafeo = { version = "0.6", default-features = false, features = ["gql"] }

# With ONNX embedding generation (opt-in, ~17MB)
grafeo = { version = "0.6", features = ["embed"] }
```

!!! warning "The default features have no write-ahead log"
    `GrafeoDB::open(path)` works with the default features, but they have no write-ahead log: commits reach the file only when the database checkpoints (`close()`, dropping it, or `wal_checkpoint()`). **A process that crashes loses every commit since the last checkpoint.** For commits that survive a crash, add the `wal` feature (also in the `storage` group and the `lpg` and `rdf` profiles):

    ```toml
    grafeo = { version = "0.6", features = ["wal"] }
    ```

    This is the Rust crate only: the Python, Node.js, C, C#, Dart and Go packages include the write-ahead log. The Rust default includes it from 0.7.0 ([#468](https://github.com/GrafeoDB/grafeo/issues/468)).

#### Persona Profiles

| Profile | Contents | Use case |
|---------|----------|----------|
| `lpg` | GQL, Cypher, Gremlin, SQL/PGQ, storage (write-ahead log, database file, spill, mmap) | Graph applications |
| `rdf` | RDF triple store, GQL, SPARQL, GraphQL, SHACL, storage | Knowledge graphs, linked data |
| `analytics` | Graph algorithms, vector, text and hybrid search, JSON Lines and Parquet import | Graph analytics pipelines |
| `ai` | Vector, text and hybrid search, change data capture | RAG, semantic search, agent memory |
| `edge` | GQL, lightweight regex | WebAssembly, resource-constrained |
| `enterprise` | Metrics, tracing, async storage (deprecated) | Platform operators, observability |

`rdf` leaves out the compact RDF index: add the `ring-index` feature for it.

!!! note "Deprecated profiles"
    The old deployment-target profiles (`embedded`, `browser`, `server`, `full`) still work as aliases but are deprecated and scheduled for removal in 0.7.0. `embedded` is the default until then. Migrate to persona profiles when convenient: `full` is `lpg`, `rdf`, `analytics` and `ai`.

#### Convenience Groups

| Group | Contents | Description |
|-------|----------|-------------|
| `languages` | gql, cypher, sparql, gremlin, graphql, sql-pgq | All query language parsers |
| `ai` | vector-index, text-index, hybrid-search, cdc | AI/RAG search + change tracking |
| `storage` | wal, spill, mmap, grafeo-file | Persistence backends |
| `algos` | graph algorithms | SSSP, PageRank, centrality, community detection |
| `embed` | ort, tokenizers | ONNX embedding generation (opt-in, ~17MB) |

#### Individual Language Flags

| Feature | Description |
|---------|-------------|
| `gql` | GQL (ISO/IEC 39075): default query language |
| `cypher` | Cypher (openCypher 9.0) |
| `sparql` | SPARQL (W3C 1.1) + RDF support |
| `gremlin` | Gremlin (Apache TinkerPop) |
| `graphql` | GraphQL |
| `sql-pgq` | SQL/PGQ (SQL:2023 GRAPH_TABLE) |

#### Individual AI Flags

| Feature | Description |
|---------|-------------|
| `vector-index` | HNSW approximate nearest neighbor index |
| `text-index` | BM25 inverted index for full-text search |
| `hybrid-search` | Combined text + vector search with score fusion |
| `cdc` | Change data capture (before/after property snapshots) |

### Verify Installation

```rust
use grafeo::GrafeoDB;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db = GrafeoDB::new_in_memory();
    println!("Grafeo installed successfully!");
    Ok(())
}
```

## Grafeo Server (Docker)

For a standalone database server accessible via REST API, use [grafeo-server](../project/ecosystem/grafeo-server.md):

```bash
# Standard: all query languages over HTTP, web UI
docker run -p 7474:7474 -e GRAFEO_AUTH_TOKEN=my-secret-token grafeo/grafeo-server
```

The images listen on `0.0.0.0`, so they refuse to start without credentials: set `GRAFEO_AUTH_TOKEN` (or `GRAFEO_AUTH_USER` and `GRAFEO_AUTH_PASSWORD`), or `GRAFEO_ALLOW_UNAUTHENTICATED=true` on a trusted network only.

Four image variants are available:

| Variant | Tag | Description |
|---------|-----|-------------|
| **gwp** | `grafeo-server:gwp` | GQL over the GQL Wire Protocol (gRPC, port 7688), no UI |
| **bolt** | `grafeo-server:bolt` | Cypher over Bolt v5 (port 7687), no UI |
| **standard** | `grafeo-server:latest` | All six languages over HTTP (port 7474), web UI |
| **full** | `grafeo-server:full` | HTTP, GQL Wire Protocol and Bolt, AI and search with ONNX embeddings, TLS, web UI |

```bash
# GQL Wire Protocol: GQL over gRPC
docker run -p 7688:7688 -e GRAFEO_AUTH_TOKEN=my-secret-token grafeo/grafeo-server:gwp

# Full: every transport, AI and search
docker run -p 7474:7474 -p 7687:7687 -p 7688:7688 -e GRAFEO_AUTH_TOKEN=my-secret-token grafeo/grafeo-server:full
```

Server at `http://localhost:7474`. Web UI (standard/full) at `http://localhost:7474/studio/`.

See the [grafeo-server documentation](../project/ecosystem/grafeo-server.md) for full API reference and configuration.

## Building from Source

### Clone the Repository

```bash
git clone https://github.com/GrafeoDB/grafeo.git
cd grafeo
```

### Build Rust Crates

```bash
cargo build --workspace --release
```

### Build Python Package

```bash
cd crates/bindings/python
uv add maturin
maturin develop --release
```

This build has the same features as the published wheel: maturin reads them from `[tool.maturin] features` in `crates/bindings/python/pyproject.toml`. A `--features` option replaces that list instead of adding to it, so `maturin develop --features pyo3/extension-module` builds only the default set, without Cypher, SPARQL, Gremlin, GraphQL and SQL/PGQ.

To test another project against an unreleased version, build a wheel the way the release does (from the repository root) and install it into that project's environment:

```bash
maturin build --release --out dist -m crates/bindings/python/Cargo.toml
uv pip install --python /path/to/project/.venv dist/grafeo-*.whl
```

Check what a build contains with `grafeo.features()` and `grafeo.build_info()`:

```python
import grafeo

print(grafeo.features())    # ['gql', 'cypher', 'sparql', ...]
print(grafeo.build_info())  # version, git commit, dirty, features, profile
```

### Build Node.js Package

```bash
cd crates/bindings/node
npm install
npm run build
```

### Build WASM Package

```bash
./scripts/build-wasm.sh --features full
```

This builds the variant `@grafeo-db/wasm` publishes into `crates/bindings/wasm/pkg`. Without `--features`, the build has the `edge` profile (GQL only), as `@grafeo-db/wasm-lite` does.

## Next Steps

With Grafeo installed, continue to the [Quick Start](quick-start.md) guide.
