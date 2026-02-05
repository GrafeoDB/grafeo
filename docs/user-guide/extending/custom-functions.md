---
title: Custom Functions
description: Adding user-defined functions to GQL.
tags:
  - extending
  - functions
---

# Custom Functions

Add user-defined functions (UDFs) to extend GQL.

## Registering Functions

### In Rust

```rust
use grafeo::{Database, Value};

let db = Database::open_in_memory()?;

// Register a scalar function
db.register_function("double", |args| {
    match &args[0] {
        Value::Int64(n) => Ok(Value::Int64(n * 2)),
        _ => Err("Expected integer".into())
    }
})?;
```

### In Python

```python
import grafeo

db = grafeo.GrafeoDB()

# Register a Python function
@db.register_function("greet")
def greet(name: str) -> str:
    return f"Hello, {name}!"
```

## Using Custom Functions

```sql
-- Use the custom function in queries
MATCH (p:Person)
RETURN p.name, double(p.age) AS doubled_age

MATCH (p:Person)
RETURN greet(p.name) AS greeting
```

## Function Types

### Scalar Functions

Return a single value:

```rust
db.register_function("uppercase", |args| {
    match &args[0] {
        Value::String(s) => Ok(Value::String(s.to_uppercase())),
        _ => Err("Expected string".into())
    }
})?;
```

### Aggregate Functions

Aggregate over multiple values:

```rust
db.register_aggregate("product", AggregateFunction {
    init: || Value::Int64(1),
    step: |acc, val| {
        match (acc, val) {
            (Value::Int64(a), Value::Int64(v)) => Value::Int64(a * v),
            _ => acc
        }
    },
    finalize: |acc| acc,
})?;
```

### Table Functions

Return multiple rows:

```rust
db.register_table_function("generate_series", |args| {
    let start = args[0].as_int()?;
    let end = args[1].as_int()?;
    Ok((start..=end).map(|i| vec![Value::Int64(i)]).collect())
})?;
```

```sql
-- Use table function
SELECT * FROM generate_series(1, 10)
```

## Function Signatures

```rust
// Define function with explicit signature
db.register_function_with_signature(
    "add",
    FunctionSignature {
        args: vec![DataType::Int64, DataType::Int64],
        returns: DataType::Int64,
    },
    |args| {
        let a = args[0].as_int()?;
        let b = args[1].as_int()?;
        Ok(Value::Int64(a + b))
    }
)?;
```

## Best Practices

1. **Validate Input** - Check argument types and counts
2. **Handle Nulls** - Decide null handling behavior
3. **Document** - Provide clear function documentation
4. **Test** - Unit test all custom functions
5. **Performance** - Keep functions efficient for use in queries
