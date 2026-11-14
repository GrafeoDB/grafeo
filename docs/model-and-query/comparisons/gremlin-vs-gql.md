---
title: Gremlin vs GQL
description: Compare Gremlin traversal language with GQL query language.
---

# Gremlin vs GQL

This guide compares Gremlin (Apache TinkerPop) with GQL (ISO/IEC 39075) to help in choosing the right query language.

## Philosophy

| Aspect | Gremlin | GQL |
|--------|---------|-----|
| **Style** | Imperative traversal | Declarative pattern matching |
| **Origin** | Apache TinkerPop | ISO standard (39075) |
| **Focus** | Step-by-step navigation | What to find, not how |

## Syntax Comparison

### Finding Nodes

=== "Gremlin"

    ```gremlin
    g.V().hasLabel('Person').has('name', 'Alix')
    ```

=== "GQL"

    ```sql
    MATCH (p:Person {name: 'Alix'})
    RETURN p
    ```

### Traversing Relationships

=== "Gremlin"

    ```gremlin
    g.V().has('name', 'Alix').out('KNOWS').values('name')
    ```

=== "GQL"

    ```sql
    MATCH (a:Person {name: 'Alix'})-[:KNOWS]->(friend)
    RETURN friend.name
    ```

### Multiple Hops

=== "Gremlin"

    ```gremlin
    g.V().has('name', 'Alix').out('KNOWS').out('KNOWS').values('name')
    ```

=== "GQL"

    ```sql
    MATCH (a:Person {name: 'Alix'})-[:KNOWS]->()-[:KNOWS]->(fof)
    RETURN fof.name
    ```

### Counting

=== "Gremlin"

    ```gremlin
    g.V().hasLabel('Person').count()
    ```

=== "GQL"

    ```sql
    MATCH (p:Person)
    RETURN COUNT(p)
    ```

### Filtering

=== "Gremlin"

    ```gremlin
    g.V().hasLabel('Person').has('age', gt(25))
    ```

=== "GQL"

    ```sql
    MATCH (p:Person)
    WHERE p.age > 25
    RETURN p
    ```

## When to Use Each

### Choose Gremlin When

- Imperative, step-by-step traversal logic is preferred
- The team is familiar with functional programming patterns
- Fine-grained control over traversal order is needed
- Porting from another TinkerPop-compatible database

### Choose GQL When

- Declarative pattern matching is preferred
- The team is familiar with SQL-like syntax
- ISO-standard compatibility is desired
- Complex pattern matching is a priority
- Clear, readable queries are needed

## Feature Comparison

| Feature | Gremlin | GQL |
|---------|---------|-----|
| Pattern matching | Via chained steps | Native MATCH clause |
| Aggregations | `count()`, `sum()`, etc. | `COUNT()`, `SUM()`, etc. |
| Path queries | `repeat().until()` | Variable-length patterns |
| Subqueries | Lambda steps | EXISTS, subqueries |
| Mutations | `addV()`, `addE()` | INSERT, SET, DELETE |
| Readability | Method chaining | SQL-like syntax |

## Mixing Languages

Grafeo allows both languages in the same database:

```python
import grafeo

db = grafeo.GrafeoDB()

# Create data with GQL
db.execute("INSERT (:Person {name: 'Alix'})")
db.execute("INSERT (:Person {name: 'Gus'})")
db.execute("""
    MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'})
    INSERT (a)-[:KNOWS]->(b)
""")

# Query with Gremlin
result = db.execute_gremlin("g.V().hasLabel('Person').values('name')")

# Or query with GQL
result = db.execute("MATCH (p:Person) RETURN p.name")
```

## Recommendation

For most use cases, **GQL** is the recommended primary query language due to its:

- ISO standardization
- Readable, declarative syntax
- Powerful pattern matching
- Familiar SQL-like structure

Use **Gremlin** when imperative traversal control is needed or when migrating from a TinkerPop-based system.
