---
title: Installation
description: Install Grafeo for Python, Node.js, Go, Rust, or WebAssembly.
---

# Installation

Grafeo supports Python, Node.js/TypeScript, Go, Rust, and WebAssembly. Choose the installation method for your preferred language.

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

# Create a test database
db = grafeo.GrafeoDB()
print("Grafeo installed successfully!")
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
const { GrafeoDB } = require('@grafeo-db/js');

const db = await GrafeoDB.create();
console.log('Grafeo installed successfully!');
await db.close();
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

## Rust

### Using Cargo

Add Grafeo to your project:

```bash
cargo add grafeo
```

Or add it manually to your `Cargo.toml`:

```toml
[dependencies]
grafeo = "0.4"
```

### Feature Flags

All query languages are enabled by default. To use a minimal build:

```toml
[dependencies]
# Default: all query languages enabled
grafeo = "0.4"

# Minimal: only specific languages
grafeo = { version = "0.4", default-features = false, features = ["gql"] }
```

| Feature | Description |
|---------|-------------|
| `default` | All query languages (GQL, Cypher, Gremlin, GraphQL, SPARQL) |
| `gql` | GQL only |
| `cypher` | Cypher only |
| `sparql` | SPARQL and RDF support |
| `gremlin` | Gremlin only |
| `graphql` | GraphQL only |

### Verify Installation

```rust
use grafeo::Database;

fn main() -> Result<(), grafeo::Error> {
    let db = Database::open_in_memory()?;
    println!("Grafeo installed successfully!");
    Ok(())
}
```

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

### Build Node.js Package

```bash
cd crates/bindings/node
npm install
npm run build
```

### Build WASM Package

```bash
wasm-pack build crates/bindings/wasm --target web --release
```

## Next Steps

Now that you have Grafeo installed, continue to the [Quick Start](quickstart.md) guide.
