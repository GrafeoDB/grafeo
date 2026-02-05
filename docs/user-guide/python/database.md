---
title: Database Operations
description: Creating and managing databases in Python.
tags:
  - python
  - database
---

# Database Operations

Learn how to create and configure Grafeo databases in Python.

## Creating a Database

```python
import grafeo

# In-memory database
db = grafeo.GrafeoDB()

# Persistent database
db = grafeo.GrafeoDB("my_graph.db")
```

## Executing Queries

```python
# Execute a query directly on the database
db.execute("INSERT (:Person {name: 'Alice', age: 30})")

# Query and iterate results
result = db.execute("MATCH (p:Person) RETURN p.name, p.age")
for row in result:
    print(row)
```

## Query Languages

Grafeo supports multiple query languages:

```python
# GQL (default)
db.execute("MATCH (p:Person) RETURN p")

# Cypher
db.execute_cypher("MATCH (p:Person) RETURN p")

# Gremlin
db.execute_gremlin("g.V().hasLabel('Person')")

# GraphQL
db.execute_graphql("{ Person { name age } }")

# SPARQL (for RDF data)
db.execute_sparql("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")
```

## Transactions

Use transactions for atomic operations:

```python
# Begin a transaction
with db.begin_transaction() as tx:
    tx.execute("INSERT (:Person {name: 'Alice'})")
    tx.execute("INSERT (:Person {name: 'Bob'})")
    tx.commit()  # Both inserts committed atomically

# Rollback on error
with db.begin_transaction() as tx:
    tx.execute("INSERT (:Person {name: 'Charlie'})")
    tx.rollback()  # Changes discarded
```

## Direct Node/Edge API

Create nodes and edges programmatically:

```python
# Create a node
node = db.create_node(["Person"], {"name": "Alice", "age": 30})
print(f"Created node with ID: {node.id}")

# Create an edge
db.execute("""
    MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
    INSERT (a)-[:KNOWS {since: 2024}]->(b)
""")
```

## Label Management

Manage node labels directly:

```python
# Add a label to a node
db.add_node_label(node.id, "Employee")

# Remove a label
db.remove_node_label(node.id, "Contractor")

# Get all labels for a node
labels = db.get_node_labels(node.id)
print(labels)  # ['Person', 'Employee']
```

## Admin APIs

Inspect and manage the database:

```python
# Database info
info = db.info()
print(f"Nodes: {info['node_count']}, Edges: {info['edge_count']}")

# Detailed statistics
stats = db.detailed_stats()

# Schema information
schema = db.schema()

# Validate database integrity
db.validate()
```

## Persistence

```python
# Save database to disk
db.save("backup.db")

# Create an in-memory copy
memory_db = db.to_memory()

# Load database as in-memory
db = grafeo.GrafeoDB.open_in_memory("my_graph.db")
```
