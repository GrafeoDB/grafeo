---
title: Vectorized Operations
description: Vectorized query execution.
tags:
  - architecture
  - execution
---

# Vectorized Operations

Operations process vectors of values instead of single values.

## Vectorization Benefits

- **SIMD** - Single instruction, multiple data
- **Cache** - Better cache utilization
- **Overhead** - Amortized interpretation overhead

## Example: Filter Operation

```
// Scalar (row-at-a-time)
for row in rows:
    if row.age > 30:
        output.append(row)

// Vectorized
ages = column["age"]
mask = ages > 30  // SIMD comparison
result = rows[mask]
```

## Data Chunk

Operations work on DataChunks:

```rust
struct DataChunk {
    columns: Vec<Vector>,
    selection: Option<SelectionVector>,
    count: usize,
}
```
