---
title: graphos-common
description: Foundation types crate.
tags:
  - api
  - rust
---

# graphos-common

Foundation types, memory allocators, and utilities.

## Types

```rust
use graphos_common::types::{NodeId, EdgeId, Value, LogicalType};
```

### NodeId / EdgeId

```rust
let node_id = NodeId(42);
let edge_id = EdgeId(100);
```

### Value

```rust
let v = Value::Int64(42);
let v = Value::String("hello".into());
let v = Value::List(vec![Value::Int64(1), Value::Int64(2)]);
```

### LogicalType

```rust
let t = LogicalType::Int64;
let t = LogicalType::String;
let t = LogicalType::List(Box::new(LogicalType::Int64));
```

## Memory

```rust
use graphos_common::memory::{Arena, Pool};

let arena = Arena::new();
let data = arena.alloc(MyStruct::new());
```

## Utilities

```rust
use graphos_common::utils::{FxHashMap, FxHashSet};

let map: FxHashMap<String, i64> = FxHashMap::default();
```
