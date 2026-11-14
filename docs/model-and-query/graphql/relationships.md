---
title: Relationships
description: Query graph relationships using nested GraphQL selections.
tags:
  - graphql
  - relationships
  - traversals
---

# Relationships

This guide covers querying graph relationships using nested field selections in GraphQL.

## Nested Selections

In Grafeo, nested fields in a GraphQL query map to edge traversals. The field name corresponds to the edge type:

```graphql
{
  Person(name: "Alix") {
    name
    friends {
      name
      age
    }
  }
}
```

This query:

1. Finds nodes with label `Person` where `name = "Alix"`
2. Returns the `name` property
3. Traverses outgoing `friends` edges
4. Returns `name` and `age` of connected nodes

## How Nesting Maps to the Graph

| GraphQL | Graph Operation |
|---------|-----------------|
| Root field name (`Person`) | Node label filter |
| Scalar field (`name`) | Property access |
| Object field (`friends { ... }`) | Edge traversal (outgoing) |
| Arguments on nested field | Filter on target nodes |

## Multi-Level Nesting

Query multiple levels of relationships:

```graphql
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

This traverses two hops: Alix's friends, then their friends.

## Filtering Nested Results

Apply arguments to nested fields to filter related nodes:

```graphql
{
  Person(name: "Alix") {
    name
    friends(age: 30) {
      name
      age
    }
  }
}
```

## Multiple Relationship Types

Query different relationship types in the same query:

```graphql
{
  Person(name: "Alix") {
    name
    friends {
      name
    }
    colleagues {
      name
      company
    }
  }
}
```

## Python Example

```python
import grafeo

db = grafeo.GrafeoDB()

# Create a social graph
db.execute("INSERT (:Person {name: 'Alix', age: 30})")
db.execute("INSERT (:Person {name: 'Gus', age: 25})")
db.execute("INSERT (:Person {name: 'Vincent', age: 35})")
db.execute("""
    MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'})
    INSERT (a)-[:friends]->(b)
""")
db.execute("""
    MATCH (b:Person {name: 'Gus'}), (c:Person {name: 'Vincent'})
    INSERT (b)-[:friends]->(c)
""")

# Query relationships
result = db.execute_graphql("""
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
""")
for row in result:
    print(row)
```

## Rust Example

```rust
use grafeo_engine::GrafeoDB;

let db = GrafeoDB::new_in_memory();

// Create data
db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
db.execute("INSERT (:Person {name: 'Gus'})").unwrap();
db.execute(
    "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) INSERT (a)-[:friends]->(b)"
).unwrap();

// Query with nested relationships
let result = db.execute_graphql(r#"
{
  Person(name: "Alix") {
    name
    friends {
      name
    }
  }
}
"#).unwrap();
```
