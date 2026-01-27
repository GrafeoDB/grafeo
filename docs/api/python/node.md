---
title: graphos.Node
description: Node class reference.
tags:
  - api
  - python
---

# graphos.Node

Represents a graph node.

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `id` | `int` | Internal node ID |
| `labels` | `List[str]` | Node labels |

## Methods

### get()

Get a property value.

```python
def get(self, key: str, default: Any = None) -> Any
```

### keys()

Get all property keys.

```python
def keys(self) -> List[str]
```

### items()

Get all property items.

```python
def items(self) -> List[Tuple[str, Any]]
```

## Example

```python
with db.session() as session:
    result = session.execute("MATCH (n:Person) RETURN n LIMIT 1")
    row = next(iter(result))
    node = row['n']

    print(f"ID: {node.id}")
    print(f"Labels: {node.labels}")
    print(f"Name: {node.get('name')}")
```
