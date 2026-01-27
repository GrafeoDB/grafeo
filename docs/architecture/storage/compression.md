---
title: Compression
description: Type-specific compression strategies.
tags:
  - architecture
  - storage
---

# Compression

Graphos uses type-specific compression for efficient storage.

## Compression Strategies

| Type | Strategy | Description |
|------|----------|-------------|
| Bool | Bit-packing | 8 bools per byte |
| Int64 | Delta + BitPack | Store differences, pack bits |
| Float64 | None / Gorilla | Raw or XOR-based |
| String | Dictionary | Common values stored once |

## Dictionary Encoding

For string columns with repeated values:

```
Original:  ["apple", "banana", "apple", "apple", "banana"]

Dictionary: {0: "apple", 1: "banana"}
Encoded:    [0, 1, 0, 0, 1]
```

## Delta Encoding

For sorted or sequential integers:

```
Original: [100, 102, 105, 107, 112]

Base: 100
Deltas: [0, 2, 3, 2, 5]
```

## Bit-Packing

Pack small integers into minimal bits:

```
Values: [3, 1, 4, 1, 5, 9] (max = 9, needs 4 bits)

Packed: 4 bits per value instead of 64
```

## Compression Selection

Graphos automatically selects compression based on data characteristics:

1. Analyze sample of data
2. Estimate compression ratio for each strategy
3. Select best strategy per column
