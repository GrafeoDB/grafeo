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

Get a property value by key. Returns `None` if the property does not exist.

```python
def get(self, key: str) -> Optional[Any]
```

### properties()

Get all properties as a dictionary.

```python
def properties(self) -> Dict[str, Any]
```

### has_label()

Check whether the node has a specific label.

```python
def has_label(self, label: str) -> bool
```

### keys()

Return a list of all property key names on this node.

```python
def keys(self) -> list[str]
```

### items()

Return a list of `(key, value)` pairs for all properties.

```python
def items(self) -> list[tuple[str, Any]]
```

## Operators

### `node["key"]`

Access a property by key. Raises `KeyError` if the property does not exist.

### `"key" in node`

Check whether the node has a property with the given key.

## Example

```python
result = db.execute("MATCH (n:Person) RETURN n LIMIT 1")
row = next(iter(result))
node = row['n']

print(f"ID: {node.id}")
print(f"Labels: {node.labels}")
print(f"Name: {node.get('name')}")
print(f"Is a Person: {node.has_label('Person')}")
print(f"All properties: {node.properties()}")
```

## Direct Node Creation

```python
# Create node with direct API
node = db.create_node(["Person"], {"name": "Alix", "age": 30})
print(f"Created node with ID: {node.id}")

# Manage labels
db.add_node_label(node.id, "Employee")
db.remove_node_label(node.id, "Contractor")
labels = db.get_node_labels(node.id)
```
