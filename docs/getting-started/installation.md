---
title: Installation
description: Install Grafeo for Python or Rust.
---

# Installation

Grafeo can be used from both Python and Rust. Choose the installation method for your preferred language.

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

## Rust

### Using Cargo

Add Grafeo to your project:

```bash
cargo add grafeo
```

Or add it manually to your `Cargo.toml`:

```toml
[dependencies]
grafeo = "0.1"
```

### Feature Flags

All query languages are enabled by default. To use a minimal build:

```toml
[dependencies]
# Default: all query languages enabled
grafeo = "0.1"

# Minimal: only specific languages
grafeo = { version = "0.1", default-features = false, features = ["gql"] }
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
git clone https://github.com/StevenBtw/grafeo.git
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

## Next Steps

Now that you have Grafeo installed, continue to the [Quick Start](quickstart.md) guide.
