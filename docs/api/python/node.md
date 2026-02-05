---
title: grafeo.Node
description: Node class reference.
tags:
  - api
  - python
---

# grafeo.Node

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
result = db.execute("MATCH (n:Person) RETURN n LIMIT 1")
row = next(iter(result))
node = row['n']

print(f"ID: {node.id}")
print(f"Labels: {node.labels}")
print(f"Name: {node.get('name')}")
```

## Direct Node Creation

```python
# Create node with direct API
node = db.create_node(["Person"], {"name": "Alice", "age": 30})
print(f"Created node with ID: {node.id}")

# Manage labels
db.add_node_label(node.id, "Employee")
db.remove_node_label(node.id, "Contractor")
labels = db.get_node_labels(node.id)
```
