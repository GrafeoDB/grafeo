---
title: Database Setup
description: Creating and configuring databases in Rust.
tags:
  - rust
  - database
---

# Database Setup

Learn how to create and configure Grafeo databases in Rust.

## Creating a Database

```rust
use grafeo::GrafeoDB;

// In-memory database
let db = GrafeoDB::new_in_memory();

// Persistent database
let db = GrafeoDB::open("my_graph.db")?;
```

## Configuration

```rust
use grafeo::{GrafeoDB, Config};

let config = Config::builder()
    .memory_limit(4 * 1024 * 1024 * 1024)  // 4 GB
    .threads(8)
    .build()?;

let db = GrafeoDB::with_config(config);
```

## Database Lifecycle

```rust
use grafeo::GrafeoDB;

fn main() -> Result<(), grafeo_common::utils::error::Error> {
    // Create database
    let db = GrafeoDB::open("my_graph.db")?;

    // Use the database
    let mut session = db.session();
    session.execute("INSERT (:Person {name: 'Alice'})")?;

    // Database is dropped and closed when it goes out of scope
    Ok(())
}
```

## Thread Safety

`GrafeoDB` is `Send` and `Sync`, so it can be shared across threads:

```rust
use grafeo::GrafeoDB;
use std::sync::Arc;
use std::thread;

let db = Arc::new(GrafeoDB::new_in_memory());

let handles: Vec<_> = (0..4).map(|i| {
    let db = Arc::clone(&db);
    thread::spawn(move || {
        let mut session = db.session();
        session.execute(&format!(
            "INSERT (:Person {{id: {}}})", i
        )).unwrap();
    })
}).collect();

for handle in handles {
    handle.join().unwrap();
}
```
