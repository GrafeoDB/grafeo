---
title: Working with Nodes
description: Node operations in Python.
tags:
  - python
  - nodes
---

# Working with Nodes

Learn how to create, read, update, and delete nodes using the Python API.

## Creating Nodes

```python
# Create a single node
db.execute("""
    INSERT (:Person {name: 'Alice', age: 30})
""")

# Create multiple nodes
db.execute("""
    INSERT (:Person {name: 'Bob', age: 25})
    INSERT (:Person {name: 'Carol', age: 35})
""")

# Create with multiple labels
db.execute("""
    INSERT (:Person:Employee {name: 'Dave', department: 'Engineering'})
""")
```

## Creating Nodes with Direct API

```python
# Create a node using the direct API
node = db.create_node(["Person"], {"name": "Alice", "age": 30})
print(f"Created node with ID: {node.id}")

# Add labels
db.add_node_label(node.id, "Employee")

# Get labels
labels = db.get_node_labels(node.id)
print(labels)  # ['Person', 'Employee']
```

## Reading Nodes

```python
# Find all nodes with label
result = db.execute("""
    MATCH (p:Person)
    RETURN p.name, p.age
""")

for row in result:
    print(f"{row['p.name']} is {row['p.age']} years old")

# Find specific node
result = db.execute("""
    MATCH (p:Person {name: 'Alice'})
    RETURN p
""")
```

## Updating Nodes

```python
# Update properties
db.execute("""
    MATCH (p:Person {name: 'Alice'})
    SET p.age = 31, p.city = 'New York'
""")

# Remove a property
db.execute("""
    MATCH (p:Person {name: 'Alice'})
    REMOVE p.temporary_field
""")

# Remove a label
db.remove_node_label(node.id, "Contractor")
```

## Deleting Nodes

```python
# Delete a node (must have no edges)
db.execute("""
    MATCH (p:Person {name: 'Alice'})
    DELETE p
""")

# Delete node and its edges
db.execute("""
    MATCH (p:Person {name: 'Bob'})
    DETACH DELETE p
""")
```
