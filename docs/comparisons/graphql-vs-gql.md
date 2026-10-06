---
title: GraphQL vs GQL
description: Compare GraphQL API queries with GQL graph queries.
---

# GraphQL vs GQL

This guide compares GraphQL (the API query language) with GQL (ISO/IEC 39075 graph query language). Despite similar names, they serve different purposes.

## Key Differences

| Aspect | GraphQL | GQL |
|--------|---------|-----|
| **Purpose** | API query language | Graph database query language |
| **Origin** | Facebook (2015) | ISO standard 39075 (2024) |
| **Focus** | Schema-driven API access | Pattern matching on graphs |
| **Scope** | Client-server communication | Direct database queries |

## Syntax Comparison

### Querying Data

=== "GraphQL"

    ```graphql
    {
      Person(name: "Alix") {
        name
        age
      }
    }
    ```

=== "GQL"

    ```sql
    MATCH (p:Person {name: 'Alix'})
    RETURN p.name, p.age
    ```

### Traversing Relationships

=== "GraphQL"

    ```graphql
    {
      Person(name: "Alix") {
        name
        friends {
          name
        }
      }
    }
    ```

=== "GQL"

    ```sql
    MATCH (a:Person {name: 'Alix'})-[:friends]->(f)
    RETURN a.name, f.name
    ```

### Multiple Levels

=== "GraphQL"

    ```graphql
    {
      Person(name: "Alix") {
        friends {
          friends {
            name
          }
        }
      }
    }
    ```

=== "GQL"

    ```sql
    MATCH (a:Person {name: 'Alix'})-[:friends]->()-[:friends]->(fof)
    RETURN fof.name
    ```

### Filtering with Operators

=== "GraphQL"

    ```graphql
    {
      Person(where: { age_gt: 30, name_contains: "Ali" }) {
        name
        age
      }
    }
    ```

=== "GQL"

    ```sql
    MATCH (p:Person)
    WHERE p.age > 30 AND p.name CONTAINS 'Ali'
    RETURN p.name, p.age
    ```

### Pagination and Ordering

=== "GraphQL"

    ```graphql
    {
      Person(orderBy: { age: DESC }, first: 10, skip: 5) {
        name
        age
      }
    }
    ```

=== "GQL"

    ```sql
    MATCH (p:Person)
    RETURN p.name, p.age
    ORDER BY p.age DESC
    LIMIT 10 OFFSET 5
    ```

### Aggregations

=== "GraphQL"

    Not natively supported (requires custom resolvers)

=== "GQL"

    ```sql
    MATCH (p:Person)
    RETURN COUNT(p)
    ```

### Mutations

=== "GraphQL"

    ```graphql
    mutation {
      createPerson(name: "Alix", age: 30) {
        id
        name
      }
    }
    ```

=== "GQL"

    ```sql
    INSERT (:Person {name: 'Alix', age: 30})
    ```

## When to Use Each

### Choose GraphQL When

- Building a client-facing API
- A schema-driven interface is needed
- Working with frontend developers familiar with GraphQL
- Integrating with existing GraphQL tooling
- Self-documenting API queries are desired

### Choose GQL When

- Direct database queries
- Complex pattern matching across relationships
- Graph algorithms and analytics
- Path finding and traversals
- Aggregations and analytics
- Full control over query semantics

## Feature Comparison

| Feature | GraphQL | GQL |
|---------|---------|-----|
| Schema required | Yes | No |
| Pattern matching | Via nesting | Native MATCH |
| Variable-length paths | No | Yes (`*1..5`) |
| Filtering | Where clause (`_gt`, `_lt`, etc.) | WHERE clause |
| Pagination | `first`/`skip` | LIMIT/OFFSET |
| Ordering | `orderBy` | ORDER BY |
| Aggregations | Custom resolvers | Native support |
| Mutations | `createX`/`deleteX` | INSERT, SET, DELETE |
| Joins | Nesting only | Arbitrary patterns |

## Using Both in Grafeo

Grafeo supports both languages, allowing different use cases:

```python
import grafeo

db = grafeo.GrafeoDB()

# Create data with GQL (full control)
db.execute("INSERT (:Person {name: 'Alix', age: 30})")
db.execute("INSERT (:Person {name: 'Gus', age: 25})")
db.execute("""
    MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'})
    INSERT (a)-[:friends]->(b)
""")

# Or create with GraphQL mutations
db.execute_graphql('''
mutation {
  createPerson(name: "Vincent", age: 35) {
    id
    name
  }
}
''')

# Query with GraphQL (familiar syntax, filtering, pagination)
result = db.execute_graphql('''
{
  Person(where: { age_gte: 25 }, orderBy: { age: DESC }, first: 10) {
    name
    age
    friends {
      name
    }
  }
}
''')

# Complex queries with GQL (full power)
result = db.execute("""
    MATCH (a:Person)-[:friends*1..3]->(distant)
    WHERE a.name = 'Alix' AND distant.age > 20
    RETURN DISTINCT distant.name, COUNT(*) as paths
""")
```

## Recommendation

| Use Case | Recommended |
|----------|-------------|
| API layer | GraphQL |
| Database queries | GQL |
| Complex traversals | GQL |
| Frontend integration | GraphQL |
| Analytics | GQL |
| Simple lookups | Either |

For most graph database operations, **GQL** provides more power and flexibility. Use **GraphQL** when a familiar, schema-driven interface is needed for application development or when working with teams experienced in GraphQL APIs.
