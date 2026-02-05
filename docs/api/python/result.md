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

## Methods

### __iter__()

Iterate over rows.

```python
for row in result:
    print(row['column_name'])
```

### to_list()

Convert to list of dicts.

```python
def to_list(self) -> List[Dict[str, Any]]
```

### fetchone()

Fetch one row.

```python
def fetchone(self) -> Optional[Dict[str, Any]]
```

### fetchall()

Fetch all rows.

```python
def fetchall(self) -> List[Dict[str, Any]]
```

## Example

```python
result = db.execute("MATCH (p:Person) RETURN p.name, p.age")

# Iterate
for row in result:
    print(row['p.name'])

# Or convert to list
rows = list(result)
```
