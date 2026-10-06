---
title: GraphQL Query Language
description: Query graph data using GraphQL syntax.
---

# GraphQL Query Language

GraphQL is a query language for APIs developed by Facebook. Grafeo supports GraphQL as an optional query language, enabling queries against graph data using familiar GraphQL syntax.

## Overview

GraphQL provides a schema-driven approach to querying data. In Grafeo, node labels map to GraphQL types, and relationships map to nested fields.

## Enabling GraphQL

GraphQL support is optional and requires a feature flag:

=== "Rust"

    ```bash
    cargo add grafeo-engine --features graphql
    ```

=== "Python"

    ```bash
    uv add grafeo[graphql]
    ```

## Quick Reference

| Operation | Syntax |
|-----------|--------|
| Query type | `{ Person { ... } }` |
| Get fields | `{ Person { name age } }` |
| Filter | `{ Person(name: "Alix") { ... } }` |
| Where clause | `{ Person(where: { age_gt: 30 }) { ... } }` |
| Pagination | `{ Person(first: 10, skip: 5) { ... } }` |
| Ordering | `{ Person(orderBy: { name: ASC }) { ... } }` |
| Nested relations | `{ Person { friends { name } } }` |
| Aliases | `{ alix: Person(name: "Alix") { ... } }` |
| Create mutation | `mutation { createPerson(name: "Alix") { id } }` |
| Delete mutation | `mutation { deletePerson(id: 123) }` |

## Basic Examples

### Simple Queries

```graphql
# Get all people with their names
{
  Person {
    name
  }
}

# Get multiple fields
{
  Person {
    name
    age
    email
  }
}
```

### Filtering with Arguments

```graphql
# Find person by name
{
  Person(name: "Alix") {
    name
    age
  }
}

# Filter by multiple properties
{
  Person(age: 30, city: "Utrecht") {
    name
    email
  }
}
```

### Where Clause with Operators

Use the `where` argument for advanced filtering with comparison operators:

```graphql
# Greater than
{
  Person(where: { age_gt: 30 }) {
    name
    age
  }
}

# Multiple conditions (AND)
{
  Person(where: { age_gte: 25, age_lte: 40 }) {
    name
    age
  }
}

# String operations
{
  Person(where: { name_contains: "Ali", email_ends_with: "@example.com" }) {
    name
    email
  }
}
```

**Supported operators:**

| Suffix | Operator | Example |
|--------|----------|---------|
| (none) | Equals | `name: "Alix"` |
| `_gt` | Greater than | `age_gt: 30` |
| `_gte` | Greater than or equal | `age_gte: 30` |
| `_lt` | Less than | `age_lt: 50` |
| `_lte` | Less than or equal | `age_lte: 50` |
| `_ne` | Not equal | `status_ne: "inactive"` |
| `_contains` | Contains substring | `name_contains: "Ali"` |
| `_starts_with` | Starts with | `name_starts_with: "A"` |
| `_ends_with` | Ends with | `email_ends_with: ".com"` |
| `_in` | In list | `status_in: ["active", "pending"]` |

### Pagination

Use `first` and `skip` to paginate results:

```graphql
# Get first 10 people
{
  Person(first: 10) {
    name
  }
}

# Skip first 20, get next 10
{
  Person(first: 10, skip: 20) {
    name
  }
}

# With filtering
{
  Person(where: { age_gt: 25 }, first: 5) {
    name
    age
  }
}
```

### Ordering

Use `orderBy` to sort results:

```graphql
# Sort by name ascending
{
  Person(orderBy: { name: ASC }) {
    name
    age
  }
}

# Sort by age descending
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

# Combined with pagination
{
  Person(orderBy: { age: DESC }, first: 10) {
    name
    age
  }
}
```

### Nested Relationships

```graphql
# Get person and their friends
{
  Person(name: "Alix") {
    name
    friends {
      name
      age
    }
  }
}

# Multiple levels deep
{
  Person(name: "Alix") {
    name
    friends {
      name
      friends {
        name
      }
    }
  }
}
```

### Using Aliases

```graphql
# Query multiple people with aliases
{
  alix: Person(name: "Alix") {
    name
    age
  }
  gus: Person(name: "Gus") {
    name
    age
  }
}
```

## Mutations

GraphQL mutations support creating and deleting nodes in the graph.

### Creating Nodes

Use `create<Type>` to create new nodes:

```graphql
# Create a new person
mutation {
  createPerson(name: "Alix", age: 30) {
    id
    name
  }
}

# Create with multiple properties
mutation {
  createUser(name: "Gus", email: "gus@example.com", active: true) {
    id
    name
    email
  }
}
```

The mutation field name follows the pattern `create<NodeLabel>`. Arguments become node properties.

### Deleting Nodes

Use `delete<Type>` to delete nodes:

```graphql
# Delete by ID
mutation {
  deletePerson(id: 123)
}

# Delete by property
mutation {
  deleteUser(email: "gus@example.com")
}
```

### Mutation Examples in Python

```python
import grafeo

db = grafeo.GrafeoDB()

# Create a node with GraphQL
result = db.execute_graphql('''
mutation {
  createPerson(name: "Alix", age: 30) {
    id
    name
  }
}
''')

# Query the created node
result = db.execute_graphql('''
{
  Person(name: "Alix") {
    name
    age
  }
}
''')

# Delete the node
db.execute_graphql('''
mutation {
  deletePerson(name: "Alix")
}
''')
```

## Python Usage

```python
import grafeo

db = grafeo.GrafeoDB()

# Create some data using GQL
db.execute("INSERT (:Person {name: 'Alix', age: 30})")
db.execute("INSERT (:Person {name: 'Gus', age: 25})")
db.execute("""
    MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'})
    INSERT (a)-[:friends]->(b)
""")

# Query with GraphQL
result = db.execute_graphql('''
{
  Person(name: "Alix") {
    name
    age
    friends {
      name
    }
  }
}
''')

for row in result:
    print(row)
```

## Rust Usage

```rust
use grafeo_engine::GrafeoDB;

let db = GrafeoDB::new_in_memory();

// Create data
db.execute("INSERT (:Person {name: 'Alix'})").unwrap();

// Query with GraphQL
let result = db.execute_graphql(r#"
{
  Person {
    name
  }
}
"#).unwrap();
```

## How It Maps to the Graph

GraphQL queries are translated to graph traversals:

| GraphQL | Graph Operation |
|---------|-----------------|
| Type name (e.g., `Person`) | Node label filter |
| Arguments | Property filters |
| Nested field | Edge traversal |
| Field name | Property access |

### Example Translation

```graphql
{
  Person(name: "Alix") {
    age
    friends {
      name
    }
  }
}
```

Translates to:

1. Find nodes with label `Person` where `name = "Alix"`
2. Return the `age` property
3. Traverse `friends` edges
4. Return `name` property of connected nodes

## Supported Features

### Query Operations
- Root type queries (label-based node selection)
- Field selection (property access)
- Arguments (property filtering)
- Where clause with comparison operators (`_gt`, `_lt`, `_contains`, etc.)
- Pagination (`first`, `skip`)
- Ordering (`orderBy` with `ASC`/`DESC`)
- Nested selections (relationship traversal)
- Aliases (multiple queries with names)
- Fragments (named and inline)

### Mutation Operations

- `create<Type>` - Create new nodes with properties
- `update<Type>` - Update node properties by filter match
- `delete<Type>` - Delete nodes by property match

Mutations operate on node properties only. Relationship creation, update, or deletion is not supported through GraphQL mutations. Use GQL or Cypher for relationship manipulation.

### Subscriptions

GraphQL subscriptions are **not supported**. The `subscription` keyword is parsed, but rejected at execution time with an error.

### Type Mapping
- GraphQL types map to node labels
- GraphQL fields map to properties or relationships
- Scalar fields return property values
- Object fields traverse relationships

## Learn More

<div class="grid cards" markdown>

-   **[Basic Queries](basic-queries.md)**

    ---

    Types, fields and simple selections.

-   **[Arguments](arguments.md)**

    ---

    Filtering with query arguments.

-   **[Relationships](relationships.md)**

    ---

    Nested queries and traversals.

-   **[Aliases & Fragments](aliases.md)**

    ---

    Advanced query composition.

</div>
