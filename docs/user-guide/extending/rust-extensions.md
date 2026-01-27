---
title: Rust Extensions
description: Extending Graphos with Rust code.
tags:
  - extending
  - rust
---

# Rust Extensions

Extend Graphos with custom Rust code for maximum performance.

## Why Rust Extensions?

- **Performance** - Native speed for computationally intensive operations
- **Type Safety** - Compile-time guarantees
- **Direct Access** - Access to internal Graphos APIs
- **Integration** - Use any Rust crate

## Creating an Extension Crate

### Cargo.toml

```toml
[package]
name = "my-graphos-extension"
version = "0.1.0"
edition = "2024"

[dependencies]
graphos-core = "0.1"
graphos-engine = "0.1"
```

### Extension Code

```rust
use graphos_core::graph::LpgStore;
use graphos_engine::{Database, Session};

/// Custom graph analysis function
pub fn analyze_connectivity(db: &Database) -> ConnectivityReport {
    let session = db.session().unwrap();

    // Access the underlying graph store
    let store = session.graph_store();

    let node_count = store.node_count();
    let edge_count = store.edge_count();
    let avg_degree = (edge_count as f64 * 2.0) / node_count as f64;

    ConnectivityReport {
        nodes: node_count,
        edges: edge_count,
        average_degree: avg_degree,
    }
}

pub struct ConnectivityReport {
    pub nodes: usize,
    pub edges: usize,
    pub average_degree: f64,
}
```

## Accessing Internal APIs

### Graph Store

```rust
use graphos_core::graph::lpg::LpgStore;

fn process_graph(store: &LpgStore) {
    // Iterate over nodes
    for node in store.nodes() {
        println!("Node: {:?}", node.id());
        for label in node.labels() {
            println!("  Label: {}", label);
        }
    }

    // Iterate over edges
    for edge in store.edges() {
        println!("Edge: {} -[{}]-> {}",
            edge.source(),
            edge.edge_type(),
            edge.target()
        );
    }
}
```

### Index Access

```rust
use graphos_core::index::BTreeIndex;

fn query_index(index: &BTreeIndex<String, NodeId>) {
    // Range query
    let results = index.range("A".."B");

    for (key, node_id) in results {
        println!("{}: {:?}", key, node_id);
    }
}
```

### Execution Engine

```rust
use graphos_core::execution::{DataChunk, Pipeline};

fn custom_operator(chunk: DataChunk) -> DataChunk {
    // Process data chunk
    chunk.filter(|row| row.get("age").as_int() > 30)
}
```

## Building and Using

### Build

```bash
cargo build --release
```

### Use in Application

```rust
use my_graphos_extension::analyze_connectivity;

let db = Database::open("my_graph.db")?;
let report = analyze_connectivity(&db);

println!("Nodes: {}", report.nodes);
println!("Edges: {}", report.edges);
println!("Avg Degree: {:.2}", report.average_degree);
```

## Exposing to Python

Use PyO3 to expose Rust extensions to Python:

```rust
use pyo3::prelude::*;

#[pyfunction]
fn analyze_connectivity_py(db: &PyGraphosDB) -> PyResult<PyDict> {
    let report = analyze_connectivity(db.inner());

    let dict = PyDict::new(py);
    dict.set_item("nodes", report.nodes)?;
    dict.set_item("edges", report.edges)?;
    dict.set_item("average_degree", report.average_degree)?;

    Ok(dict)
}

#[pymodule]
fn my_extension(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(analyze_connectivity_py, m)?)?;
    Ok(())
}
```
