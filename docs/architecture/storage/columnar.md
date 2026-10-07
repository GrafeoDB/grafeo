---
title: Columnar Properties
description: Columnar storage for node and edge properties.
tags:
  - architecture
  - storage
---

# Columnar Properties

Properties are stored in a columnar format for efficient access and compression.

## Why Columnar?

| Benefit | Description |
|---------|-------------|
| **Compression** | Same-type values compress better |
| **Cache efficiency** | Sequential access patterns |
| **Vectorization** | SIMD-friendly operations |
| **Selective reads** | Only read needed columns |

## Storage Layout

```
Property Store for "Person" nodes:
┌─────────────────────────────────────────────┐
│ Column: "name" (String)                     │
├─────────────────────────────────────────────┤
│ ["Alix", "Gus", "Harm", "Dave", ...]      │
├─────────────────────────────────────────────┤
│ Column: "age" (Int64)                       │
├─────────────────────────────────────────────┤
│ [30, 25, 35, 28, ...]                       │
├─────────────────────────────────────────────┤
│ Column: "active" (Bool)                     │
├─────────────────────────────────────────────┤
│ [true, true, false, true, ...]              │
└─────────────────────────────────────────────┘
```

## Type-Specific Storage

| Type | Storage Format |
|------|----------------|
| Bool | Bit-packed array |
| Int64 | Native array or delta-encoded |
| Float64 | Native array |
| String | Dictionary + offsets |
| List | Nested columnar |

## Null Handling

Nulls are tracked with a validity bitmap:

```
Values:   [30, 25, _, 28, _, 35]
Validity: [1,  1,  0, 1,  0, 1 ]
```

## In the Database File

The layout above is the one in memory. A checkpoint writes each property
column as column chunks of the `LPG_STORE` section: per group of 65,536 node
or edge ids, the column's values in chunks of at most 1 MiB. A chunk stores
only the rows that have a value, behind a presence bitmap (one bit per row,
left out when every row has a value), so a missing property costs one bit
and is never stored as an empty value. Each chunk picks its codec from its
values:

| Values in the chunk | Stored as |
|---------------------|-----------|
| Non-negative `Int64` | Bit-packed to the width of the largest |
| `Int64`, some negative | 8 bytes each |
| `Float64` | 8 bytes each, by their bits |
| `Bool` | 1 bit each |
| `String` | The chunk's distinct strings once, and a 4-byte code per value |
| Vectors of one dimension count | 4 bytes per component |
| Any other kind, or several kinds | Each value in a lossless tagged encoding |

A typed codec is used only when every value of the chunk has its kind, so
every value reads back exactly as it was written. Each chunk of numbers,
booleans or short strings also stores a [zone map](zone-maps.md). See
[Column Chunks](container-format.md#column-chunks) for the byte layout.
