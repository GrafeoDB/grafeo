---
title: Python API
description: Python API reference.
---

# Python API Reference

Complete reference for the `grafeo` Python package.

## Installation

```bash
uv add grafeo
```

## Quick Start

```python
import grafeo

db = grafeo.GrafeoDB()
db.execute("INSERT (:Person {name: 'Alix'})")
```

## Classes

| Class | Description |
|-------|-------------|
| [Database](database.md) | Database connection and management |
| [Node](node.md) | Graph node representation |
| [Edge](edge.md) | Graph edge representation |
| [QueryResult](result.md) | Query result iteration |
| [Transaction](transaction.md) | Transaction management |

## Module Functions

| Function | Description |
|----------|-------------|
| `grafeo.features()` | Features compiled into this build, by Cargo feature name: the query languages (`gql`, `cypher`, `sparql`, `gremlin`, `graphql`, `sql-pgq`) and optional capabilities such as `algos`, `vector-index` or `triple-store` |
| `grafeo.build_info()` | Dict describing the build: `version`, `commit` (git commit, or `None` outside a git checkout), `dirty` (uncommitted changes at build time), `features` and `profile` (`"release"` or `"debug"`) |
| `grafeo.simd_support()` | SIMD instruction set used for vector operations: `"avx2"`, `"sse"`, `"neon"` or `"scalar"` |
| `grafeo.vector(values)` | Builds a vector value from a list of floats |

Check at startup that a build has what your application needs:

```python
import grafeo

missing = {"cypher", "algos"} - set(grafeo.features())
if missing:
    raise RuntimeError(f"grafeo build lacks: {sorted(missing)}")
```
