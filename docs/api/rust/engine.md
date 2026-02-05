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
let config = Config::builder()
    .memory_limit(4 * 1024 * 1024 * 1024)
    .threads(8)
    .build()?;
let db = GrafeoDB::with_config(config);
```

## Session

```rust
let mut session = db.session();

session.execute("INSERT (:Person {name: 'Alice'})")?;

let result = session.execute("MATCH (p:Person) RETURN p.name")?;
for row in result.rows {
    println!("{:?}", row);
}
```

## Transactions

```rust
let mut session = db.session();
session.begin()?;
session.execute("...")?;
session.commit()?;
// or
session.rollback()?;
```
