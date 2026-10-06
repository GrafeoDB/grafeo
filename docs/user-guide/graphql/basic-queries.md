---
title: Basic Queries
description: Learn basic GraphQL queries for selecting types and fields.
tags:
  - graphql
  - queries
---

# Basic Queries

This guide covers the fundamentals of querying graph data with GraphQL in Grafeo.

## Anonymous Queries

The simplest query uses an anonymous operation with a selection set:

```graphql
{
  Person {
    name
  }
}
```

This finds all nodes with the `Person` label and returns their `name` property.

## Named Queries

Name queries for clarity:

```graphql
query GetPeople {
  Person {
    name
    age
  }
}
```

## Field Selection

Select the specific properties to return:

```graphql
# Single field
{
  Person {
    name
  }
}

# Multiple fields
{
  Person {
    name
    age
    email
    city
  }
}
```

## Type-to-Label Mapping

The root field name maps to a node label in the graph. Grafeo automatically capitalizes the first letter:

| GraphQL Root Field | Node Label |
|-------------------|------------|
| `Person { ... }` | `:Person` |
| `user { ... }` | `:User` |
| `company { ... }` | `:Company` |

## Filtering with Direct Arguments

Pass arguments to filter by property values:

```graphql
# Filter by name
{
  Person(name: "Alix") {
    name
    age
  }
}

# Filter by multiple properties (AND)
{
  Person(age: 30, city: "Utrecht") {
    name
    email
  }
}
```

## Pagination

Use `first` and `skip` to paginate results:

```graphql
# First 10 people
{
  Person(first: 10) {
    name
  }
}

# Skip 20, take next 10
{
  Person(first: 10, skip: 20) {
    name
  }
}
```

`limit` and `offset` are also supported as aliases:

```graphql
{
  Person(limit: 10, offset: 20) {
    name
  }
}
```

## Ordering

Sort results with `orderBy`:

```graphql
# Ascending by name
{
  Person(orderBy: { name: ASC }) {
    name
    age
  }
}

# Descending by age
{
  Person(orderBy: { age: DESC }) {
    name
    age
  }
}

# Multiple sort keys
{
  Person(orderBy: { age: DESC, name: ASC }) {
    name
    age
  }
}
```

## Combining Features

Combine filtering, ordering and pagination:

```graphql
{
  Person(
    name: "Alix"
    orderBy: { age: DESC }
    first: 5
  ) {
    name
    age
    city
  }
}
```

## Python Example

```python
import grafeo

db = grafeo.GrafeoDB()

# Create data
db.execute("INSERT (:Person {name: 'Alix', age: 30})")
db.execute("INSERT (:Person {name: 'Gus', age: 25})")

# Simple query
result = db.execute_graphql("""
{
  Person {
    name
    age
  }
}
""")
for row in result:
    print(row)

# With filter and pagination
result = db.execute_graphql("""
{
  Person(first: 10, orderBy: { name: ASC }) {
    name
  }
}
""")
```

## Rust Example

```rust
use grafeo_engine::GrafeoDB;

let db = GrafeoDB::new_in_memory();
db.execute("INSERT (:Person {name: 'Alix', age: 30})").unwrap();

let result = db.execute_graphql(r#"
{
  Person {
    name
    age
  }
}
"#).unwrap();
```

## Mutations

GraphQL mutations (`create<Type>`, `update<Type>`, `delete<Type>`) are limited to node property CRUD operations. Relationship creation, update, or deletion is not supported in GraphQL mutation payloads. Use GQL or Cypher for relationship manipulation. See the [GraphQL overview](index.md#mutations) for mutation syntax and examples.
