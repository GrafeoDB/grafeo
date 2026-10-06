---
title: Arrow Export
description: Bulk export graph data to PyArrow, Polars and pandas for analytics workflows.
tags:
  - example
  - arrow
  - polars
  - pandas
  - analytics
---

# Arrow Export

Export graph data as columnar Arrow tables for fast analytics with PyArrow, Polars, DuckDB and pandas.

!!! tip "When to use Arrow export vs `nodes_df()`/`edges_df()`"

    | Method | Best for |
    |--------|----------|
    | `nodes_to_arrow()` / `edges_to_arrow()` | Large graphs, DuckDB integration, zero-copy sharing |
    | `nodes_to_polars()` / `edges_to_polars()` | Polars-native workflows, lazy evaluation |
    | `nodes_to_pandas()` / `edges_to_pandas()` | pandas users who want the Arrow fast path explicitly |
    | `nodes_df()` / `edges_df()` | Quick exploration (auto-uses Arrow when pyarrow is installed) |

    At scale (100K+ nodes), the Arrow path is **10-100x faster** than element-by-element export because the RecordBatch is built in Rust and serialized as a single IPC buffer.

## Setup

```bash
uv add grafeo pyarrow polars pandas duckdb
```

## Build a Sample Graph

```python
from grafeo import GrafeoDB

db = GrafeoDB()

db.execute("""
    INSERT (:Person {name: 'Alix', age: 30, city: 'Amsterdam'})
    INSERT (:Person {name: 'Gus', age: 28, city: 'Berlin'})
    INSERT (:Person {name: 'Vincent', age: 35, city: 'Paris'})
    INSERT (:Person {name: 'Jules', age: 32, city: 'Amsterdam'})
    INSERT (:Person {name: 'Mia', age: 27, city: 'Berlin'})
    INSERT (:Company {name: 'Acme', founded: 2015})
    INSERT (:Company {name: 'Globex', founded: 2020})
""")

db.execute("""
    MATCH (a:Person {name: 'Alix'}), (g:Person {name: 'Gus'})
    INSERT (a)-[:KNOWS {since: 2019}]->(g)
""")
db.execute("""
    MATCH (a:Person {name: 'Alix'}), (c:Company {name: 'Acme'})
    INSERT (a)-[:WORKS_AT {role: 'Engineer'}]->(c)
""")
db.execute("""
    MATCH (g:Person {name: 'Gus'}), (c:Company {name: 'Globex'})
    INSERT (g)-[:WORKS_AT {role: 'Designer'}]->(c)
""")
db.execute("""
    MATCH (v:Person {name: 'Vincent'}), (j:Person {name: 'Jules'})
    INSERT (v)-[:KNOWS {since: 2021}]->(j)
""")
```

## PyArrow: Filtering and DuckDB Integration

`nodes_to_arrow()` returns a `pyarrow.Table` with columns: `id` (uint64), `labels` (list\<utf8\>), plus one column per property key.

```python
import pyarrow.compute as pc

table = db.nodes_to_arrow()
print(table.schema)
print(f"{table.num_rows} nodes exported")
```

```title="Output"
id: uint64
labels: list<item: string>
name: string
age: int64
city: string
founded: int64
7 nodes exported
```

Filter directly on the Arrow table:

```python
# People in Amsterdam
mask = pc.equal(table.column("city"), "Amsterdam")
amsterdam = table.filter(mask)
print(amsterdam.to_pandas()[["name", "city"]])
```

```title="Output"
    name       city
0   Alix  Amsterdam
1  Jules  Amsterdam
```

Query with DuckDB (zero-copy, no data movement):

```python
import duckdb

result = duckdb.sql("""
    SELECT name, age, city
    FROM table
    WHERE age >= 30
    ORDER BY age DESC
""")
print(result.fetchdf())
```

```title="Output"
      name  age       city
0  Vincent   35      Paris
1    Jules   32  Amsterdam
2     Alix   30  Amsterdam
```

## Polars: Lazy Evaluation and Filtering

`nodes_to_polars()` returns a `polars.DataFrame` directly, without requiring pyarrow.

```python
import polars as pl

df = db.nodes_to_polars()
print(df)
```

```title="Output"
shape: (7, 5)
+-----+-----------+---------+------+-----------+
| id  | labels    | name    | age  | city      |
| u64 | list[str] | str     | i64  | str       |
+-----+-----------+---------+------+-----------+
| 1   | [Person]  | Alix    | 30   | Amsterdam |
| 2   | [Person]  | Gus     | 28   | Berlin    |
| ...                                          |
+-----+-----------+---------+------+-----------+
```

Polars lazy evaluation for efficient multi-step pipelines:

```python
young_berliners = (
    df.lazy()
    .filter(pl.col("city") == "Berlin")
    .filter(pl.col("age") < 30)
    .select("name", "age")
    .collect()
)
print(young_berliners)
```

```title="Output"
shape: (2, 2)
+------+-----+
| name | age |
| str  | i64 |
+------+-----+
| Gus  | 28  |
| Mia  | 27  |
+------+-----+
```

## pandas: Direct DataFrame Access

`nodes_to_pandas()` builds the Arrow table in Rust and converts to pandas in one step:

```python
df = db.nodes_to_pandas()
print(df.groupby("city")["age"].mean())
```

```title="Output"
city
Amsterdam    31.0
Berlin       27.5
Paris        35.0
Name: age, dtype: float64
```

## Edge Export

`edges_to_arrow()` returns columns: `id` (uint64), `type` (utf8), `source` (uint64), `target` (uint64), plus one column per property key.

```python
edges_table = db.edges_to_arrow()
print(edges_table.schema)
```

```title="Output"
id: uint64
type: string
source: uint64
target: uint64
since: int64
role: string
```

With Polars:

```python
edges_df = db.edges_to_polars()

# All KNOWS relationships
knows = edges_df.filter(pl.col("type") == "KNOWS")
print(knows.select("source", "target", "since"))
```

With DuckDB (join nodes and edges for a full view):

```python
nodes = db.nodes_to_arrow()
edges = db.edges_to_arrow()

result = duckdb.sql("""
    SELECT n1.name AS from_person, n2.name AS to_person, e.since
    FROM edges e
    JOIN nodes n1 ON e.source = n1.id
    JOIN nodes n2 ON e.target = n2.id
    WHERE e.type = 'KNOWS'
""")
print(result.fetchdf())
```

```title="Output"
  from_person to_person  since
0        Alix       Gus   2019
1     Vincent     Jules   2021
```

## Performance Note

The Arrow export path builds a single `RecordBatch` in Rust and serializes it as IPC bytes. Python receives the buffer and deserializes it in one call with no per-element PyO3 crossings. On graphs with 100K+ entities, this is typically **10-100x faster** than the row-by-row `nodes_df()`/`edges_df()` fallback.

The existing `nodes_df()`/`edges_df()` methods auto-detect pyarrow at runtime and use the Arrow fast path when available. Installing pyarrow speeds up all DataFrame exports without code changes.

## Next Steps

- [Performance Baselines](../../ecosystem/performance.md) for detailed throughput numbers
- [Graph Visualization example](graph-visualization.md) for interactive exploration
- [NetworkX Integration](networkx-integration.md) for graph algorithm workflows
