---
title: Properties
description: Property types and values in Grafeo.
tags:
  - data-model
  - properties
---

# Properties

Properties are key-value pairs stored on nodes and edges. Grafeo supports a rich set of property types.

## Supported Types

| Type | Example | Description |
|------|---------|-------------|
| `Boolean` | `true`, `false` | True/false values |
| `Int64` | `42`, `-100` | 64-bit signed integers |
| `Float64` | `3.14`, `-0.5` | 64-bit floating point |
| `String` | `'hello'` | UTF-8 text |
| `Vector` | `[0.1, 0.2, 0.3]` | f32 array for embeddings |
| `List` | `[1, 2, 3]` | Ordered collection |
| `Map` | `{key: 'value'}` | Key-value collection |
| `Date` | `'2024-01-15'` | Calendar date |
| `DateTime` | `'2024-01-15T10:30:00Z'` | Date and time |
| `Null` | `null` | Absence of value |

## Using Properties

### Setting Properties

```sql
INSERT (:Product {
    name: 'Widget',
    price: 29.99,
    in_stock: true,
    tags: ['electronics', 'sale'],
    metadata: {category: 'gadgets', sku: 'WDG-001'}
})
```

### Querying Properties

```sql
-- Simple property access
MATCH (p:Product)
RETURN p.name, p.price

-- Property comparisons
MATCH (p:Product)
WHERE p.price < 50 AND p.in_stock = true
RETURN p.name

-- List operations
MATCH (p:Product)
WHERE 'sale' IN p.tags
RETURN p.name
```

### Updating Properties

```sql
-- Set a property
MATCH (p:Product {name: 'Widget'})
SET p.price = 24.99

-- Set multiple properties
MATCH (p:Product {name: 'Widget'})
SET p.price = 24.99, p.on_sale = true

-- Remove a property
MATCH (p:Product {name: 'Widget'})
REMOVE p.on_sale
```

## Null Handling

```sql
-- Check for null
MATCH (p:Person)
WHERE p.email IS NULL
RETURN p.name

-- Check for not null
MATCH (p:Person)
WHERE p.email IS NOT NULL
RETURN p.name, p.email

-- Coalesce null values
MATCH (p:Person)
RETURN p.name, coalesce(p.email, 'no email') AS email
```

## Vector Properties

Vectors store dense embeddings for similarity search. See the [Vector Search Guide](../vector-search/index.md) for comprehensive documentation.

### Storing Vectors

```sql
-- Store embeddings on nodes
INSERT (:Document {
    title: 'Introduction to Graphs',
    embedding: [0.1, 0.2, 0.3, -0.1, 0.5]
})

-- Store with specific dimensions (384-dimensional embedding)
INSERT (:Product {
    name: 'Widget',
    description_embedding: $embedding  -- Passed as parameter
})
```

### Querying Vectors

```sql
-- Find similar documents using cosine similarity
MATCH (d:Document)
WHERE cosine_similarity(d.embedding, $query_embedding) > 0.8
RETURN d.title

-- Find k-nearest neighbors
MATCH (d:Document)
RETURN d.title, cosine_distance(d.embedding, $query) AS distance
ORDER BY distance
LIMIT 10
```

### Distance Functions

| Function                    | Description                                        |
| --------------------------- | -------------------------------------------------- |
| `cosine_similarity(a, b)`   | Cosine similarity (1 = identical, 0 = orthogonal)  |
| `cosine_distance(a, b)`     | 1 - cosine_similarity                              |
| `euclidean_distance(a, b)`  | L2 distance                                        |
| `dot_product(a, b)`         | Inner product                                      |
| `manhattan_distance(a, b)`  | L1 distance                                        |
