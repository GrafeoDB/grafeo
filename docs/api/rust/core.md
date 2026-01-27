---
title: graphos-core
description: Core data structures crate.
tags:
  - api
  - rust
---

# graphos-core

Core graph storage and execution engine.

## Graph Storage

```rust
use graphos_core::graph::lpg::{LpgStore, NodeRecord, EdgeRecord};

let store = LpgStore::new();
let node_id = store.create_node(&["Person"], props);
```

## Indexes

```rust
use graphos_core::index::{HashIndex, BTreeIndex};

let index: HashIndex<String, NodeId> = HashIndex::new();
index.insert("Alice".into(), node_id);
```

## Execution

```rust
use graphos_core::execution::{DataChunk, Vector, SelectionVector};

let chunk = DataChunk::new(columns, 1024);
```

## Note

This is an internal crate. The API may change between minor versions.
