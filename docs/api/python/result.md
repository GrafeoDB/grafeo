---
title: grafeo.QueryResult
description: QueryResult class reference.
tags:
  - api
  - python
---

# grafeo.QueryResult

Query result iterator.

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `columns` | `List[str]` | Column names |
| `execution_time_ms` | `float` | Query execution time in milliseconds |
| `rows_scanned` | `int` | Rows the query read from storage before any filter: nodes its scans and index lookups read, edges its expands read, search hits and RDF triples |
| `counters` | `dict` | What the query's writes changed: `nodes_created`, `nodes_deleted`, `edges_created`, `edges_deleted`, `properties_set`, `labels_added`, `labels_removed` |
| `profile` | `dict \| None` | For a `PROFILE` statement, its operators as a tree: `total_time_ms` and `root`, each operator with `operator`, `detail`, `rows`, `calls`, `self_time_ms`, `time_ms` and `children` (experimental); `None` otherwise |

## Methods

### __iter__()

Iterate over rows.

```python
for row in result:
    print(row['column_name'])
```

### __getitem__()

Access a row by index.

```python
row = result[0]
```

### __len__()

Get the number of rows in the result.

```python
count = len(result)
```

### to_list()

Convert to list of dicts.

```python
def to_list(self) -> List[Dict[str, Any]]
```

### scalar()

Return the first column of the first row as a scalar value.

```python
def scalar(self) -> Any
```

### nodes()

Return all nodes from the result.

```python
def nodes(self) -> List[Node]
```

### edges()

Return all edges from the result.

```python
def edges(self) -> List[Edge]
```

## Example

```python
result = db.execute("MATCH (p:Person) RETURN p.name, p.age")

# Iterate
for row in result:
    print(row['p.name'])

# Length
print(f"Found {len(result)} rows")

# Index access
first_row = result[0]

# Convert to list
rows = result.to_list()

# Scalar value
count = db.execute("MATCH (n) RETURN count(n)").scalar()

# Get nodes
nodes = db.execute("MATCH (n:Person) RETURN n").nodes()
```
