---
title: Arguments
description: Filter graph data using GraphQL arguments and the where clause.
tags:
  - graphql
  - filtering
  - arguments
---

# Arguments

This guide covers filtering with GraphQL arguments, the `where` clause and comparison operators.

## Direct Arguments

The simplest filter uses direct property equality:

```graphql
# Exact match
{
  Person(name: "Alix") {
    name
    age
  }
}

# Multiple conditions (AND)
{
  Person(age: 30, city: "Utrecht") {
    name
    email
  }
}
```

## Where Clause

Use the `where` argument for advanced filtering with comparison operators:

```graphql
# Greater than
{
  Person(where: { age_gt: 30 }) {
    name
    age
  }
}

# Less than or equal
{
  Person(where: { age_lte: 25 }) {
    name
    age
  }
}

# Not equal
{
  Person(where: { status_ne: "inactive" }) {
    name
  }
}
```

## Supported Operators

| Suffix | Operator | Example |
|--------|----------|---------|
| *(none)* | Equals | `name: "Alix"` |
| `_gt` | Greater than | `age_gt: 30` |
| `_gte` | Greater than or equal | `age_gte: 30` |
| `_lt` | Less than | `age_lt: 50` |
| `_lte` | Less than or equal | `age_lte: 50` |
| `_ne` | Not equal | `status_ne: "inactive"` |
| `_contains` | Contains substring | `name_contains: "Ali"` |
| `_starts_with` | Starts with | `name_starts_with: "A"` |
| `_ends_with` | Ends with | `email_ends_with: ".com"` |
| `_in` | In list | `status_in: ["active", "pending"]` |

## Multiple Conditions

Multiple conditions in the `where` object are combined with AND:

```graphql
# Age between 25 and 40
{
  Person(where: { age_gte: 25, age_lte: 40 }) {
    name
    age
  }
}

# String conditions
{
  Person(where: { name_starts_with: "A", email_ends_with: "@example.com" }) {
    name
    email
  }
}
```

## Combining with Pagination and Ordering

Arguments can be used together:

```graphql
{
  Person(
    where: { age_gt: 25 }
    orderBy: { age: DESC }
    first: 10
    skip: 0
  ) {
    name
    age
  }
}
```

## Variable Definitions

Use variables for dynamic argument values:

```graphql
query FindPerson($name: String!) {
  Person(name: $name) {
    name
    age
    email
  }
}
```

## Python Example

```python
import grafeo

db = grafeo.GrafeoDB()

# Create data
db.execute("INSERT (:Person {name: 'Alix', age: 30, city: 'Utrecht'})")
db.execute("INSERT (:Person {name: 'Gus', age: 25, city: 'Portland'})")
db.execute("INSERT (:Person {name: 'Vincent', age: 35, city: 'Utrecht'})")

# Where clause with operators
result = db.execute_graphql("""
{
  Person(where: { age_gt: 25, city: "Utrecht" }) {
    name
    age
  }
}
""")
for row in result:
    print(row)  # Alix (30), Vincent (35)

# String filtering
result = db.execute_graphql("""
{
  Person(where: { name_starts_with: "A" }) {
    name
  }
}
""")
```
