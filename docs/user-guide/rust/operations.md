---
title: Graph Operations
description: CRUD operations in Rust.
tags:
  - rust
  - operations
---

# Graph Operations

Learn how to perform CRUD operations using the Rust API.

## Creating Data

```rust
let session = db.session();

// Create nodes
session.execute(r#"
    INSERT (:Person {name: 'Alix', age: 30})
    INSERT (:Person {name: 'Gus', age: 25})
"#)?;

// Create edges
session.execute(r#"
    MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'})
    INSERT (a)-[:KNOWS {since: 2020}]->(b)
"#)?;
```

## Reading Data

```rust
let session = db.session();

// Query with results
let result = session.execute(r#"
    MATCH (p:Person)
    RETURN p.name, p.age
"#)?;

for row in result.rows() {
    let name = row[0].as_str().unwrap_or("?");
    let age = row[1].as_i64().unwrap_or(0);
    println!("{}: {}", name, age);
}
```

## Updating Data

```rust
let session = db.session();

session.execute(r#"
    MATCH (p:Person {name: 'Alix'})
    SET p.age = 31
"#)?;
```

## Deleting Data

```rust
let session = db.session();

// Delete edges
session.execute(r#"
    MATCH (a:Person)-[r:KNOWS]->(b:Person)
    DELETE r
"#)?;

// Delete nodes
session.execute(r#"
    MATCH (p:Person {name: 'Alix'})
    DETACH DELETE p
"#)?;
```

## Parameterized Queries

```rust
use std::collections::HashMap;
use grafeo::Value;

let session = db.session();

let mut params = HashMap::new();
params.insert("name".to_string(), Value::from("Alix"));

let result = session.execute_with_params(
    "MATCH (p:Person {name: $name}) RETURN p",
    params,
)?;
```
