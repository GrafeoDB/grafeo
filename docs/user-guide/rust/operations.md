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

## Loading CSV Files

`load_nodes_csv` and `load_edges_csv` load a typed CSV file in one bulk transaction (on `GrafeoDB`, and on a
`Session` in its open transaction):

```rust
use grafeo::loader::{Column, ColumnKind, EdgeLoad, EndKey, NodeLoad};

let people = NodeLoad {
    labels: vec!["Person".into()],
    key: Some("id".into()),
    columns: vec![
        Column { name: "id".into(), kind: ColumnKind::Int64 },
        Column { name: "born".into(), kind: ColumnKind::Date },
    ],
    ..NodeLoad::default()
};
let report = db.load_nodes_csv("people.csv", &people)?;

let knows = EdgeLoad {
    edge_type: "KNOWS".into(),
    source: EndKey { label: "Person".into(), property: "id".into(), column: "src".into() },
    target: EndKey { label: "Person".into(), property: "id".into(), column: "dst".into() },
    columns: vec![
        Column { name: "src".into(), kind: ColumnKind::Int64 },
        Column { name: "dst".into(), kind: ColumnKind::Int64 },
    ],
    ..EdgeLoad::default()
};
db.load_edges_csv("knows.csv", &knows)?;
```

A column the list does not name is a string; `ColumnKind::Skip` leaves one out. `LoadReport` gives the records
read, created and skipped (with `MissingEnd::Skip`). See the Python guide's
[Loading typed CSV files](../python/database.md#loading-typed-csv-files) for the rules, which are the same. New in
0.6.1 and [experimental](../../stability.md).

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
