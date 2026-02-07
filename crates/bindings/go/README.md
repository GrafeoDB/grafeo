# grafeo

Go bindings for [Grafeo](https://grafeo.dev) - a high-performance, pure-Rust, embeddable graph database.

## Requirements

- Go 1.22+
- CGO enabled (`CGO_ENABLED=1`)
- The `grafeo-c` shared library (`libgrafeo_c.so` / `libgrafeo_c.dylib` / `grafeo_c.dll`)

## Installation

```bash
go get github.com/GrafeoDB/grafeo/crates/bindings/go
```

## Quick Start

```go
package main

import (
    "fmt"
    "log"

    grafeo "github.com/GrafeoDB/grafeo/crates/bindings/go"
)

func main() {
    db, err := grafeo.OpenInMemory()
    if err != nil {
        log.Fatal(err)
    }
    defer db.Close()

    // Create nodes
    db.CreateNode([]string{"Person"}, map[string]any{"name": "Alice", "age": 30})
    db.CreateNode([]string{"Person"}, map[string]any{"name": "Bob", "age": 25})

    // Query with GQL
    result, err := db.Execute("MATCH (p:Person) WHERE p.age > 20 RETURN p.name, p.age")
    if err != nil {
        log.Fatal(err)
    }
    for _, row := range result.Rows {
        fmt.Printf("Name: %v, Age: %v\n", row["p.name"], row["p.age"])
    }
}
```

## Features

- GQL, Cypher, SPARQL, Gremlin, and GraphQL query languages
- Full node/edge CRUD with property management
- ACID transactions with configurable isolation levels
- HNSW vector similarity search
- Property indexes for fast lookups
- Thread-safe for concurrent use

## Building the Shared Library

```bash
# From the Grafeo repository root:
cargo build --release -p grafeo-c --features full

# The library is at:
#   target/release/libgrafeo_c.so      (Linux)
#   target/release/libgrafeo_c.dylib   (macOS)
#   target/release/grafeo_c.dll        (Windows)
```

## Links

- [Documentation](https://grafeo.dev)
- [GitHub](https://github.com/GrafeoDB/grafeo)
- [Python Package](https://pypi.org/project/grafeo/)
- [npm Package](https://www.npmjs.com/package/@grafeo-db/js)
