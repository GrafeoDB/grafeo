---
title: Rust API
description: Using Grafeo from Rust.
---

# Rust API

Grafeo is written in Rust and provides a native Rust API.

## Quick Start

```rust
use grafeo::GrafeoDB;

fn main() -> Result<(), grafeo_common::utils::error::Error> {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session.execute("INSERT (:Person {name: 'Alice'})")?;

    let result = session.execute("MATCH (p:Person) RETURN p.name")?;
    for row in result.rows {
        println!("{:?}", row);
    }

    Ok(())
}
```

## Sections

<div class="grid cards" markdown>

-   **[Database Setup](database.md)**

    ---

    Creating and configuring databases.

-   **[Graph Operations](operations.md)**

    ---

    CRUD operations on nodes and edges.

-   **[Sessions](sessions.md)**

    ---

    Session management and transactions.

</div>
