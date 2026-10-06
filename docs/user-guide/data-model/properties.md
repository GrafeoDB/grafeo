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
| `Date` | `'2024-01-15'` | Calendar date (ISO 8601) |
| `Time` | `'14:30:00'` | Time of day with optional timezone |
| `Duration` | `'P1Y2M3D'` | ISO 8601 duration (months, days, nanoseconds) |
| `DateTime` | `'2024-01-15T10:30:00Z'` | Date and time (microsecond precision) |
| `Null` | `null` | Absence of value |
| `GCounter` | `{"$gcounter": {...}}` | Grow-only CRDT counter |
| `OnCounter` | `{"$pncounter": {...}}` | Positive-negative CRDT counter |

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

## Temporal Properties

Grafeo supports temporal types for dates, times and durations with ISO 8601 formatting.

### Creating Temporal Properties

**GQL typed literal syntax:**

```sql
MATCH (p:Person {name: 'Alix'})
SET p.birthday = DATE '1990-06-15'

MATCH (e:Event {name: 'Meeting'})
SET e.start_time = TIME '14:30:00'

MATCH (t:Task {name: 'Sprint'})
SET t.length = DURATION 'P2W3D'
```

**Cypher function syntax:**

```sql
MATCH (p:Person {name: 'Alix'})
SET p.birthday = date('1990-06-15')

MATCH (e:Event {name: 'Meeting'})
SET e.length = duration('PT2H30M')
```

### Temporal Comparisons

```sql
-- GQL
MATCH (p:Person)
WHERE p.birthday > DATE '2000-01-01'
RETURN p.name

-- Cypher
MATCH (p:Person)
WHERE p.birthday > date('2000-01-01')
RETURN p.name
```

### Temporal Arithmetic

```sql
-- Date + Duration = Date
RETURN DATE '2024-01-15' + DURATION 'P1M' AS next_month
-- 2024-02-15

-- Date - Date = Duration
RETURN DATE '2024-03-15' - DATE '2024-01-01' AS diff
-- P74D

-- Duration + Duration = Duration
RETURN duration('P1Y') + duration('P6M') AS combined
-- P1Y6M
```

### Component Extraction

```sql
-- Cypher extraction functions
RETURN year(date('2024-03-15')) AS y,
       month(date('2024-03-15')) AS m,
       day(date('2024-03-15')) AS d

RETURN hour(time('14:30:45')) AS h,
       minute(time('14:30:45')) AS m,
       second(time('14:30:45')) AS s
```

### Duration Format

Durations use ISO 8601 format: `OnYnMnDTnHnMnS`

| Component | Meaning | Example |
|-----------|---------|---------|
| `P` | Period marker (required) | `P1Y` |
| `nY` | Years | `P2Y` = 2 years |
| `nM` | Months (before T) | `P3M` = 3 months |
| `nD` | Days | `P10D` = 10 days |
| `T` | Time separator | `PT5H` = 5 hours |
| `nH` | Hours | `PT12H` = 12 hours |
| `nM` | Minutes (after T) | `PT30M` = 30 minutes |
| `nS` | Seconds | `PT45S` = 45 seconds |

Combined example: `P1Y2M3DT4H5M6S` = 1 year, 2 months, 3 days, 4 hours, 5 minutes, 6 seconds.

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

## CRDT Counters

Grafeo supports two conflict-free replicated counter types for distributed
counting without coordination. These are first-class property values, stored
and queried like any other type.

### GCounter (grow-only)

A grow-only counter backed by per-replica tallies. Each replica increments its
own slot; the logical value is the sum of all slots. Merge takes the per-replica
maximum, making it commutative and idempotent.

In bindings the value is returned as a structured object:

```json
{
  "$gcounter": { "replica-a": 5, "replica-b": 3 },
  "$value": 8
}
```

`$value` is the computed total (sum of all replica slots).

### OnCounter (positive-negative)

Two grow-only maps track increments and decrements independently. The logical
value is `sum(pos) - sum(neg)`. Merge applies per-replica max to each map.

```json
{
  "$pncounter": { "pos": { "replica-a": 10 }, "neg": { "replica-a": 3 } },
  "$value": 7
}
```

Both counter types survive snapshot export/import and JSON round-trips without
loss. In aggregation contexts (GROUP BY, ORDER BY) they hash and compare by
their logical value.
