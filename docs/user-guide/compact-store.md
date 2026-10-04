---
title: Compact Store
description: Convert a database to a layered columnar format for faster queries and lower memory usage; remains writable through an overlay.
tags:
  - performance
  - storage
  - compact-store
  - wasm
---

# Compact Store

CompactStore is a columnar graph format that trades some write performance for large
memory and query wins. After ingesting data, call `compact()` to switch the database
to a columnar layout with CSR adjacency. From 0.5.39, `compact()` is **non-destructive
and writable**: it produces a layered store with an immutable columnar base plus a
mutable overlay. Inserts and property updates after `compact()` land in the overlay;
compacting again merges the overlay back into a fresh base (`recompact()` in Rust,
`compact()` again in Python and Node.js).

Queries keep working across all supported languages, indexes (vector, text, hybrid)
can be created and searched post-compact, and named graphs are preserved across
`compact()` / `recompact()`.

**When to use it:** workloads that ingest once and query many times, or read-heavy
workloads with occasional updates. Code analysis tools, static knowledge graphs,
pre-built datasets for WASM or edge deployments.

## Performance

Measured on the same data, CompactStore vs the standard mutable LpgStore:

| Metric | LpgStore | CompactStore | Improvement |
|--------|----------|--------------|-------------|
| Memory per node (degree 5) | ~3,200 bytes | ~51 bytes | **63x** |
| Edge traversal (10K lookups) | 619 us | 5.3 us | **116x** |
| Property random access (10K) | 123 us | 10 us | **12x** |

The gains come from eliminating MVCC version chains, read locks, hash lookups, and
chunk decompression. CompactStore replaces those with array indexing and contiguous
memory reads.

## Quick Start

=== "Python"

    ```python
    import grafeo

    db = grafeo.GrafeoDB()

    # Ingest data (read-write phase)
    db.execute("INSERT (:Person {name: 'Alix', age: 30})")
    db.execute("INSERT (:Person {name: 'Gus', age: 25})")
    db.execute("INSERT (:City {name: 'Amsterdam'})")
    db.execute("""
        MATCH (p:Person {name: 'Alix'}), (c:City {name: 'Amsterdam'})
        INSERT (p)-[:LIVES_IN]->(c)
    """)

    # Switch to compact mode (subsequent writes go to a mutable overlay)
    db.compact()

    # Queries work as before, but faster
    result = db.execute("MATCH (p:Person)-[:LIVES_IN]->(c:City) RETURN p.name, c.name")
    ```

=== "Node.js"

    ```typescript
    import { GrafeoDB } from '@grafeo-db/node';

    const db = GrafeoDB.create();

    await db.execute("INSERT (:Person {name: 'Alix', age: 30})");
    await db.execute("INSERT (:City {name: 'Amsterdam'})");
    await db.execute(`
        MATCH (p:Person {name: 'Alix'}), (c:City {name: 'Amsterdam'})
        INSERT (p)-[:LIVES_IN]->(c)
    `);

    db.compact();

    const result = await db.execute(
        "MATCH (p:Person)-[:LIVES_IN]->(c:City) RETURN p.name, c.name"
    );
    ```

=== "WASM"

    ```javascript
    import init, { Database } from '@grafeo-db/wasm';
    await init();

    const db = new Database();
    db.execute("INSERT (:Person {name: 'Alix', age: 30})");
    db.execute("INSERT (:City {name: 'Amsterdam'})");

    db.compact();

    const result = db.execute(
        "MATCH (p:Person)-[:LIVES_IN]->(c:City) RETURN p.name, c.name"
    );
    ```

=== "C"

    ```c
    #include "grafeo.h"

    GrafeoDatabase *db = grafeo_open_memory();

    grafeo_execute(db, "INSERT (:Person {name: 'Alix', age: 30})");
    grafeo_execute(db, "INSERT (:City {name: 'Amsterdam'})");

    grafeo_compact(db);

    GrafeoResult *r = grafeo_execute(db,
        "MATCH (p:Person) RETURN p.name");
    ```

=== "Rust"

    ```rust
    use grafeo::GrafeoDB;

    let mut db = GrafeoDB::new_in_memory();

    db.execute("INSERT (:Person {name: 'Alix', age: 30})")?;
    db.execute("INSERT (:City {name: 'Amsterdam'})")?;

    db.compact()?;

    let result = db.execute(
        "MATCH (p:Person)-[:LIVES_IN]->(c:City) RETURN p.name, c.name"
    )?;
    ```

## How It Works

`compact()` performs four steps:

1. **Scans** all nodes from the current store, grouped by label
2. **Infers** column types from property values and builds per-label columnar tables
3. **Builds** forward and backward CSR adjacency for each edge type
4. **Swaps** the database to a layered store: the new columnar tables become the
   immutable base and a mutable overlay is attached on top to absorb subsequent
   writes. Compacting again later folds the overlay back into a fresh base.

The result is a `CompactStore` backed by:

- **Per-label columnar tables** with typed codecs (bit-packed integers, dictionary-encoded
  strings, boolean bitmaps)
- **Double-indexed CSR** (Compressed Sparse Row) for O(degree) forward and backward traversal
- **Zone maps** (min/max statistics per column) for predicate pushdown

## Type Mapping

Property values are automatically mapped to the most efficient columnar codec:

| Value type | Codec | Notes |
|------------|-------|-------|
| `Int64` (non-negative) | BitPacked | Auto-determined bit width |
| `Bool` | Bitmap | 1 bit per value |
| `String` | Dictionary | Deduplicated string table |
| `Float64` | Float64 (native) | 8 bytes per value, since 0.5.40 |
| `Vector` (f32) | Float32Vector (native) | Contiguous float32 storage, since 0.5.40 |
| Mixed `Int64 + Float64` | Float64 (native) | Columns coalesce to `Float64` when both types appear |
| Negative `Int64` | Dictionary | Serialized as string |
| `List`, `Map`, `Timestamp`, etc. | Dictionary | Serialized as string |

!!! note
    Before 0.5.40, `Float64` and `Vector` columns fell back to dictionary encoding,
    which preserved data but lost typed semantics for range scans. Native codecs
    now retain those semantics without a dictionary round-trip. Dictionary fallback
    still applies to negative integers and complex values (`List`, `Map`, etc.).

## Writes After `compact()`

Since 0.5.39, `compact()` returns a layered store: an immutable columnar base plus a
mutable overlay. New inserts and property updates land in the overlay and are visible
to subsequent queries (`get_node`, property reads, pattern matching, `list_graphs`).

Compact again to merge the overlay back into a fresh base (`recompact()` in Rust):

    db.compact()
    db.execute("INSERT (:Person {name: 'Mia'})")   # lands in overlay
    db.compact()                                    # merges overlay into new base

Indexes (`create_vector_index`, `create_text_index`, hybrid search) work on layered
stores: vector/text scan and search now fall through both layers.

## Limitations

- **Overlay write path**: writes go through the overlay, which is less optimized than
  `LpgStore`'s full MVCC path. Sustained write-heavy workloads should stay on `LpgStore`
  or compact again periodically.
- **Multi-label nodes**: nodes with multiple labels are stored under a compound key
  (e.g., `"Actor|Person"`, sorted alphabetically). A query like `MATCH (n:Person)` will
  not match nodes stored under `"Actor|Person"`. Workarounds:
    - **Preferred:** use a single label per node before compacting.
    - **Alternative:** query the compound label explicitly, e.g., `MATCH (n:Actor:Person)` (labels in alphabetical order).
    - **Alternative:** assign a canonical "primary" label and store additional labels as a list property instead.
- **No disk serialization**: `compact()` operates in memory. To persist a compacted database,
  use snapshot export (WASM) or save before compacting.
- **Databases compacted by 0.5.44 or older**: those versions stored a property that a node or
  edge lacked as the column's empty value (`''`, `0`, `0.0` or `false`), and such files keep those
  values. Since 0.6.0 a missing property stays missing after `compact()`
  ([#542](https://github.com/GrafeoDB/grafeo/issues/542)).

## Feature Flag

CompactStore requires the `compact-store` feature flag. It is **not** included in the engine-level named profiles (`embedded`, `browser`, `server`, `full`), but it is included in the binding-level defaults:

| Binding | Profile | Includes `compact-store` |
|---------|---------|--------------------------|
| Python (`grafeo-python`) | `embedded` | Yes |
| Node.js (`grafeo-node`) | `embedded` | Yes |
| C (`grafeo-c`) | `embedded` | Yes |
| WASM (`grafeo-wasm`) | `edge` | Yes |

For custom Rust builds: `cargo build --features compact-store`.
