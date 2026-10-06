---
title: Type System
description: GQL data types, typed literals, type checking and type conversion.
tags:
  - gql
  - types
---

# Type System

GQL supports a rich type system with scalar, temporal, collection and graph element types.

## Scalar Types

| Type | Aliases | Example Literals |
|------|---------|-----------------|
| `INTEGER` | `INT`, `INT64` | `42`, `0xFF`, `0o77`, `0b1010` |
| `FLOAT` | `FLOAT64` | `3.14`, `1.5e10` |
| `STRING` | | `'hello'`, `"world"` |
| `BOOLEAN` | `BOOL` | `TRUE`, `FALSE` |
| `NULL` | | `NULL` |

```sql
-- Integer literals
RETURN 42          -- decimal
RETURN 0xFF        -- hexadecimal (255)
RETURN 0o77        -- octal (63)
RETURN 0b1010      -- binary (10)

-- Float literals
RETURN 3.14        -- decimal
RETURN 1.5e10      -- scientific notation

-- String literals (single or double quotes)
RETURN 'hello'
RETURN "world"

-- Boolean
RETURN TRUE, FALSE

-- Null
RETURN NULL
```

## Temporal Types

| Type | Literal Syntax | Example |
|------|---------------|---------|
| `DATE` | `DATE 'YYYY-MM-DD'` | `DATE '2024-01-15'` |
| `TIME` | `TIME 'HH:MM:SS'` | `TIME '14:30:00'` |
| `DATETIME` | `DATETIME 'ISO8601'` | `DATETIME '2024-01-15T14:30:00Z'` |
| `DURATION` | `DURATION 'ISO8601'` | `DURATION 'P1Y2M3D'` |
| `ZONED DATETIME` | `ZONED DATETIME 'ISO8601+offset'` | `ZONED DATETIME '2024-01-15T14:30:00+05:30'` |
| `ZONED TIME` | `ZONED TIME 'HH:MM:SS+offset'` | `ZONED TIME '14:30:00+05:30'` |

```sql
-- Date
RETURN DATE '2024-01-15'

-- Time
RETURN TIME '14:30:00'

-- Datetime (with or without timezone designator)
RETURN DATETIME '2024-01-15T14:30:00Z'
RETURN DATETIME '2024-01-15T14:30:00'

-- Duration (ISO 8601)
RETURN DURATION 'P1Y2M3D'     -- 1 year, 2 months, 3 days
RETURN DURATION 'PT12H30M'    -- 12 hours, 30 minutes

-- Zoned datetime (fixed UTC offset)
RETURN ZONED DATETIME '2024-01-15T14:30:00+05:30'
RETURN ZONED DATETIME '2024-01-15T14:30:00Z'

-- Zoned time
RETURN ZONED TIME '14:30:00-04:00'
```

See [Temporal Functions](functions-temporal.md) for constructors, extraction and arithmetic.

## Collection Types

### Lists

Ordered collections of values. Lists can contain mixed types.

```sql
-- List literal
RETURN [1, 2, 3]
RETURN ['Alix', 'Gus', 'Harm']
RETURN [1, 'mixed', true, null]

-- Index access (0-based)
RETURN [10, 20, 30][0]        -- 10

-- List functions
RETURN size([1, 2, 3])        -- 3
RETURN head([1, 2, 3])        -- 1
RETURN tail([1, 2, 3])        -- [2, 3]
RETURN last([1, 2, 3])        -- 3
```

### Maps

Key-value pairs. Keys are strings, values can be any type.

```sql
-- Map literal
RETURN {name: 'Alix', age: 30, active: true}

-- Property access
WITH {name: 'Alix', age: 30} AS person
RETURN person.name, person.age
```

## Type Checking

### IS TYPED / IS NOT TYPED

Check the runtime type of a value in `WHERE` clauses:

```sql
-- Check type
MATCH (p:Person)
WHERE p.age IS TYPED INTEGER
RETURN p.name

-- Negative check
MATCH (p:Person)
WHERE p.score IS NOT TYPED FLOAT
RETURN p.name
```

### CAST

Convert between types explicitly:

```sql
RETURN CAST('42' AS INTEGER)       -- 42
RETURN CAST(42 AS FLOAT)           -- 42.0
RETURN CAST(42 AS STRING)          -- '42'
RETURN CAST('true' AS BOOLEAN)     -- true
RETURN CAST('2024-01-15' AS DATE)
RETURN CAST(42 AS LIST)            -- [42]
```

See [Expressions](expressions.md#cast) for the complete list of supported CAST targets.

## Type Conversion Functions

| Function | Target Type |
|----------|-------------|
| `toInteger(expr)` / `toInt(expr)` | INTEGER |
| `toFloat(expr)` | FLOAT |
| `toString(expr)` | STRING |
| `toBoolean(expr)` | BOOLEAN |
| `toDate(expr)` | DATE |
| `toTime(expr)` | TIME |
| `toDatetime(expr)` | DATETIME |
| `toDuration(expr)` | DURATION |
| `toZonedDatetime(expr)` | ZONED DATETIME |
| `toZonedTime(expr)` | ZONED TIME |
| `toList(expr)` | LIST |

See [Element & Path Functions](functions-element.md#type-conversion-functions) and [Temporal Functions](functions-temporal.md#type-conversion) for details.

## Three-Valued Logic

GQL uses three-valued logic: `TRUE`, `FALSE` and `NULL` (unknown). `NULL` propagates through most operations:

```sql
-- NULL comparisons
RETURN NULL = NULL          -- NULL (not TRUE)
RETURN NULL <> 1            -- NULL
RETURN 1 + NULL             -- NULL

-- Use IS NULL / IS NOT NULL to test for null
MATCH (p:Person)
WHERE p.email IS NOT NULL
RETURN p.name

-- COALESCE to provide defaults
MATCH (p:Person)
RETURN COALESCE(p.nickname, p.name) AS display_name
```
