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
    INSERT (:Person {name: 'Alix'})
    INSERT (:Person {name: 'Gus'})
""")

# Create an edge
db.execute("""
    MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'})
    INSERT (a)-[:KNOWS]->(b)
""")

# Create edge with properties
db.execute("""
    MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'})
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
    MATCH (a:Person {name: 'Alix'})-[r]->(b)
    RETURN type(r) AS relationship_type, b.name
""")
```

## Updating Edges

```python
# Update edge properties
db.execute("""
    MATCH (a:Person {name: 'Alix'})-[r:KNOWS]->(b:Person {name: 'Gus'})
    SET r.strength = 'close', r.updated = true
""")
```

## Deleting Edges

```python
# Delete specific edge
db.execute("""
    MATCH (a:Person {name: 'Alix'})-[r:KNOWS]->(b:Person {name: 'Gus'})
    DELETE r
""")

# Delete all edges of a type
db.execute("""
    MATCH ()-[r:TEMPORARY]->()
    DELETE r
""")
```
