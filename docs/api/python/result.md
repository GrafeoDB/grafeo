---
title: graphos.QueryResult
description: QueryResult class reference.
tags:
  - api
  - python
---

# graphos.QueryResult

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

### scalar()

Get single scalar value.

```python
def scalar(self) -> Any
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
with db.session() as session:
    result = session.execute("MATCH (p:Person) RETURN p.name, p.age")

    # Get column names
    print(result.columns)  # ['p.name', 'p.age']

    # Iterate
    for row in result:
        print(row['p.name'])

    # Or convert to list
    rows = result.to_list()
```
