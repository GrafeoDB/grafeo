---
title: Rust Extensions
description: Extending Grafeo with Rust code.
tags:
  - extending
  - rust
---

# Rust Extensions

Extend Grafeo with custom Rust code for maximum performance.

## Why Rust Extensions?

- **Performance** - Native speed for computationally intensive operations
- **Type Safety** - Compile-time guarantees
- **Direct Access** - Access to internal Grafeo APIs
- **Integration** - Use any Rust crate

## Creating an Extension Crate

### Cargo.toml

```toml
[package]
name = "my-grafeo-extension"
version = "0.1.0"
edition = "2024"

[dependencies]
grafeo-core = "0.6"
grafeo-engine = "0.6"
```

### Extension Code

```rust
use grafeo_common::types::ReadView;
use grafeo_common::utils::error::Result;
use grafeo_engine::GrafeoDB;

/// Custom graph analysis function
pub fn analyze_connectivity(db: &GrafeoDB) -> Result<ConnectivityReport> {
    // Access the underlying graph store (read-only), as a committed reader
    // at the current epoch sees it. Every read can fail (an external store's
    // own error), so it returns a Result.
    let store = db.graph_store();
    let view = ReadView::committed(db.current_epoch());

    let node_count = store.node_count(view)?;
    let edge_count = store.edge_count(view)?;
    let avg_degree = (edge_count as f64 * 2.0) / node_count as f64;

    Ok(ConnectivityReport {
        nodes: node_count,
        edges: edge_count,
        average_degree: avg_degree,
    })
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
use grafeo_common::types::ReadView;
use grafeo_common::utils::error::Result;
use grafeo_core::graph::{Direction, GraphStore};

/// Every read names who reads (`view`): the nodes and edges that reader sees.
fn process_graph(store: &dyn GraphStore, view: ReadView) -> Result<()> {
    for id in store.node_ids(view)? {
        let Some(node) = store.node(id, view)? else { continue };
        println!("Node: {:?}", node.id);
        for label in &node.labels {
            println!("  Label: {label}");
        }
        // Its outgoing edges of any type (`&[]`): those the view sees, whose
        // other end it sees
        for (target, edge) in store.edges(id, Direction::Outgoing, &[], view)? {
            if let Some(edge_type) = store.edge_type(edge, view)? {
                println!("Edge: {:?} -[{edge_type}]-> {:?}", node.id, target);
            }
        }
    }
    Ok(())
}
```

### Index Access

```rust
use grafeo_core::index::HashIndex;

fn query_index(index: &HashIndex<String, NodeId>) {
    // Point lookup
    if let Some(node_ids) = index.get("Alix") {
        for node_id in node_ids {
            println!("Found: {:?}", node_id);
        }
    }
}
```

### Execution Engine

```rust
use grafeo_core::execution::{DataChunk, Pipeline};

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
use my_grafeo_extension::analyze_connectivity;

let db = GrafeoDB::open("my_graph.db")?;
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
fn analyze_connectivity_py(db: &PyGrafeoDB) -> PyResult<PyDict> {
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
