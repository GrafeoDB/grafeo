---
title: graphos-engine
description: Database engine crate.
tags:
  - api
  - rust
---

# graphos-engine

Main database facade and coordination.

## Database

```rust
use graphos_engine::{Database, Config};

// In-memory
let db = Database::open_in_memory()?;

// Persistent
let db = Database::open("path/to/db")?;

// With config
let config = Config::builder()
    .memory_limit(4 * 1024 * 1024 * 1024)
    .threads(8)
    .build()?;
let db = Database::open_with_config("path", config)?;
```

## Session

```rust
let session = db.session()?;

session.execute("INSERT (:Person {name: 'Alice'})")?;

let result = session.execute("MATCH (p:Person) RETURN p.name")?;
for row in result {
    println!("{}", row.get::<String>("p.name")?);
}
```

## Transactions

```rust
session.begin()?;
session.execute("...")?;
session.commit()?;
// or
session.rollback()?;
```
