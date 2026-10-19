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

## Decimals

An exact decimal (GQL and SQL/PGQ `1.50`, `CAST(x AS DECIMAL(10, 2))`, SPARQL
`xsd:decimal`) comes back as a `decimal.Decimal` with its scale, and a
`decimal.Decimal` parameter or property goes in exactly:

```python
from decimal import Decimal

db.execute("INSERT (:Item {price: 19.99})")
db.execute("MATCH (i:Item) RETURN i.price")[0]["i.price"]  # Decimal('19.99')
db.execute("RETURN $p * 3 AS total", {"p": Decimal("0.10")})  # Decimal('0.30')
```

A float stays a `float`. NaN, the infinities and a `Decimal` of more than 38
digits are refused with a `TypeError`. `to_pandas()` and `to_polars()` keep
the `Decimal` objects.

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
