---
title: Working with Edges
description: Edge operations in Python.
tags:
  - python
  - edges
---

# Working with Edges

Learn how to create and manage relationships between nodes.

## Creating Edges

```python
# First create nodes
db.execute("""
    INSERT (:Person {name: 'Alice'})
    INSERT (:Person {name: 'Bob'})
""")

# Create an edge
db.execute("""
    MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
    INSERT (a)-[:KNOWS]->(b)
""")

# Create edge with properties
db.execute("""
    MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
    INSERT (a)-[:WORKS_WITH {since: 2020, project: 'Alpha'}]->(b)
""")
```

## Reading Edges

```python
# Find edges
result = db.execute("""
    MATCH (a:Person)-[r:KNOWS]->(b:Person)
    RETURN a.name AS from, b.name AS to, r.since
""")

for row in result:
    print(f"{row['from']} knows {row['to']} since {row['r.since']}")

# Get edge type
result = db.execute("""
    MATCH (a:Person {name: 'Alice'})-[r]->(b)
    RETURN type(r) AS relationship_type, b.name
""")
```

## Updating Edges

```python
# Update edge properties
db.execute("""
    MATCH (a:Person {name: 'Alice'})-[r:KNOWS]->(b:Person {name: 'Bob'})
    SET r.strength = 'close', r.updated = true
""")
```

## Deleting Edges

```python
# Delete specific edge
db.execute("""
    MATCH (a:Person {name: 'Alice'})-[r:KNOWS]->(b:Person {name: 'Bob'})
    DELETE r
""")

# Delete all edges of a type
db.execute("""
    MATCH ()-[r:TEMPORARY]->()
    DELETE r
""")
```
