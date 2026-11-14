---
title: grafeo-core
description: Core data structures crate.
tags:
  - api
  - rust
---

# grafeo-core

Core graph storage and execution engine.

## Graph Storage

```rust
use grafeo_core::graph::rowgroup::RowGroupStore;

// A raw write commits at once: for tests, loaders and seeding. A database
// writes through the store contract (`GraphBackend`), in transactions.
let store = RowGroupStore::new();
let node_id = store.create_node(&["Person"]);
```

## Indexes

```rust
use grafeo_core::index::HashIndex;

let index: HashIndex<String, NodeId> = HashIndex::new();
index.insert("Alix".into(), node_id);
```

## Execution

```rust
use grafeo_core::execution::{DataChunk, ValueVector, SelectionVector};

let chunk = DataChunk::empty();
```

## Note

This is an internal crate. The API may change between minor versions.
