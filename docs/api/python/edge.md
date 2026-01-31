---
title: grafeo.Edge
description: Edge class reference.
tags:
  - api
  - python
---

# grafeo.Edge

Represents a graph edge.

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `id` | `int` | Internal edge ID |
| `type` | `str` | Edge type |
| `source` | `int` | Source node ID |
| `target` | `int` | Target node ID |

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

## Example

```python
result = db.execute("""
    MATCH (a)-[r:KNOWS]->(b)
    RETURN r LIMIT 1
""")
row = next(iter(result))
edge = row['r']

print(f"Type: {edge.type}")
print(f"From: {edge.source} To: {edge.target}")
print(f"Since: {edge.get('since')}")
```
