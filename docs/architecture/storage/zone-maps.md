---
title: Zone Maps
description: Statistics for predicate pushdown and data skipping.
tags:
  - architecture
  - storage
---

# Zone Maps

Zone maps store statistics about data chunks to enable data skipping.

## Why Zone Maps in a Graph Database?

Graph databases still need property filtering (`WHERE n.age > 30`). When properties are stored in columnar chunks, zone maps let the engine skip entire chunks whose min/max range doesn't overlap the filter predicate. This is especially effective for sorted or clustered properties.

## What Zone Maps Store

| Statistic | Purpose |
| --------- | ------- |
| Min value | Skip chunks where max < filter value |
| Max value | Skip chunks where min > filter value |
| Null count | Skip chunks with no nulls for IS NULL |
| Distinct estimate | Cardinality estimation |
| Bloom filter | Point lookups |

## Example

```text
Query: WHERE age > 50

Chunk 0: min=20, max=45  -> SKIP (max < 50)
Chunk 1: min=30, max=60  -> SCAN (range overlaps)
Chunk 2: min=55, max=80  -> SCAN (range overlaps)
Chunk 3: min=18, max=35  -> SKIP (max < 50)
```

## Predicate Support

| Predicate | Zone Map Check |
| --------- | -------------- |
| `x = v` | min <= v <= max |
| `x > v` | max > v |
| `x < v` | min < v |
| `x >= v` | max >= v |
| `x <= v` | min <= v |
| `x IS NULL` | null_count > 0 |
| `x IN (...)` | bloom filter check |

## In the Database File

Since 0.6.0, the column chunks of the database file store zone maps: a chunk
whose values are all `Int64`, all `Float64` (NaN left out), all `Bool` or all
`String` stores their minimum and maximum (for strings, only when both are at
most 64 bytes). A chunk covers at most 65,536 rows and 1 MiB, so a property
column gets one zone map per range of node or edge ids. A reader refuses a zone map
that differs from the chunk's values, so it can be trusted. The file stores no
null count, as the chunk's presence bitmap and value count give the rows
without a value, and no bloom filter.

An open still decodes every chunk into memory, so nothing skips chunks in the
file yet: queries use the zone maps that the property columns in memory keep.
Skipping chunks in the file comes with the compact-core store
([#432](https://github.com/GrafeoDB/grafeo/issues/432)), whose cold chunks are
these chunks. See [Column Chunks](container-format.md#column-chunks) for the
byte layout.
