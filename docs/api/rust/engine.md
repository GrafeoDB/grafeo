---
title: grafeo-engine
description: Database engine crate.
tags:
  - api
  - rust
---

# grafeo-engine

Main database facade and coordination.

## GrafeoDB

```rust
use grafeo_engine::{GrafeoDB, Config};

// In-memory
let db = GrafeoDB::new_in_memory();

// Persistent
let db = GrafeoDB::open("path/to/db")?;

// With config
let config = Config::in_memory()
    .with_memory_limit(4 * 1024 * 1024 * 1024)
    .with_threads(8);
let db = GrafeoDB::with_config(config)?;
```

## Session

```rust
let mut session = db.session();

session.execute("INSERT (:Person {name: 'Alix'})")?;

let result = session.execute("MATCH (p:Person) RETURN p.name")?;
for row in result.rows() {
    println!("{:?}", row);
}
```

## Transactions

```rust
let mut session = db.session();
session.begin_transaction()?;
session.execute("...")?;
session.commit()?;
// or
session.rollback()?;
```
