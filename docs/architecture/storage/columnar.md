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
│ ["Alice", "Bob", "Carol", "Dave", ...]      │
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
