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
| `edge_type` | `str` | Edge type |
| `source_id` | `int` | Source node ID |
| `target_id` | `int` | Target node ID |

## Methods

### get()

Get a property value by key. Returns `None` if the property does not exist.

```python
def get(self, key: str) -> Optional[Any]
```

### properties()

Get all properties as a dictionary.

```python
def properties(self) -> Dict[str, Any]
```

## Operators

### `edge["key"]`

Access a property by key. Raises `KeyError` if the property does not exist.

### `"key" in edge`

Check whether the edge has a property with the given key.

## Example

```python
result = db.execute("""
    MATCH (a)-[r:KNOWS]->(b)
    RETURN r LIMIT 1
""")
row = next(iter(result))
edge = row['r']

print(f"Type: {edge.edge_type}")
print(f"From: {edge.source_id} To: {edge.target_id}")
print(f"Since: {edge.get('since')}")
print(f"All properties: {edge.properties()}")
```
