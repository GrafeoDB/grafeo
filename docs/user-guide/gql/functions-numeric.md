---
title: Numeric Functions
description: GQL numeric and mathematical functions in Grafeo.
tags:
  - gql
  - functions
  - math
---

# Numeric Functions

## Summary

| Function | Description |
|----------|-------------|
| `abs(n)` | Absolute value |
| `ceil(n)` / `ceiling(n)` | Round up |
| `floor(n)` | Round down |
| `round(n)` | Round to nearest |
| `sign(n)` | Sign (-1, 0 or 1) |
| `sqrt(n)` | Square root |
| `log(n)` / `ln(n)` | Natural logarithm |
| `log10(n)` | Base-10 logarithm |
| `power(b, x)` / `pow(b, x)` | b raised to the power x |
| `exp(n)` | e raised to the power n |
| `log2(n)` | Base-2 logarithm |
| `sin(n)`, `cos(n)`, `tan(n)` | Trigonometric functions |
| `asin(n)`, `acos(n)`, `atan(n)` | Inverse trigonometric |
| `atan2(y, x)` | Two-argument arctangent |
| `degrees(n)` | Radians to degrees |
| `radians(n)` | Degrees to radians |
| `pi()` | Pi constant |
| `e()` | Euler's number |
| `rand()` / `random()` | Random float [0, 1) |
| `cardinality(list)` | Number of elements in a list (alias for `size`) |

## Rounding

```sql
RETURN abs(-42)          -- 42
RETURN ceil(3.2)         -- 4
RETURN ceiling(3.2)      -- 4 (alias)
RETURN floor(3.8)        -- 3
RETURN round(3.5)        -- 4
RETURN sign(-42)         -- -1
RETURN sign(0)           -- 0
RETURN sign(42)          -- 1
```

## Power

```sql
RETURN power(2, 10)      -- 1024.0
RETURN pow(9, 0.5)       -- 3.0 (square root)
```

## Logarithmic and Exponential

```sql
RETURN sqrt(16)          -- 4.0
RETURN log(e())          -- 1.0
RETURN ln(e())           -- 1.0 (alias)
RETURN log10(1000)       -- 3.0
RETURN log2(8)           -- 3.0
RETURN exp(1)            -- 2.718281828...
```

## Trigonometric

All trigonometric functions work in radians.

```sql
RETURN sin(pi() / 2)    -- 1.0
RETURN cos(0)            -- 1.0
RETURN tan(pi() / 4)     -- ~1.0

-- Inverse
RETURN asin(1)           -- ~1.5708 (pi/2)
RETURN acos(1)           -- 0.0
RETURN atan(1)           -- ~0.7854 (pi/4)
RETURN atan2(1, 1)       -- ~0.7854 (pi/4)

-- Convert between degrees and radians
RETURN degrees(pi())     -- 180.0
RETURN radians(180)      -- ~3.14159
```

## Constants

```sql
RETURN pi()              -- 3.14159265...
RETURN e()               -- 2.71828182...
```

## Random

Returns a random float between 0 (inclusive) and 1 (exclusive):

```sql
-- Random value
RETURN rand()

-- Random sampling: get ~10% of nodes
MATCH (n:Person)
WHERE rand() < 0.1
RETURN n.name

-- Random ordering
MATCH (p:Person)
RETURN p.name
ORDER BY rand()
LIMIT 5
```

## Cardinality

`cardinality(list)` returns the number of elements in a list. It is an alias for `size()`.

```sql
MATCH (p:Person)
RETURN p.name, cardinality(p.hobbies) AS num_hobbies
```

## Arithmetic Operators

GQL supports standard arithmetic operators on numeric values:

| Operator | Description |
|----------|-------------|
| `+` | Addition |
| `-` | Subtraction |
| `*` | Multiplication |
| `/` | Division |
| `%` | Modulo (remainder) |
| `-n` | Unary negation |
| `+n` | Unary identity |

```sql
MATCH (p:Person)
RETURN p.name,
    p.salary * 12 AS annual,
    p.salary * 12 * 0.7 AS after_tax,
    p.bonus % 1000 AS remainder
```
