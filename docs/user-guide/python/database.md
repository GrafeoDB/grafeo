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
db.execute("INSERT (:Person {name: 'Alix', age: 30})")

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
    tx.execute("INSERT (:Person {name: 'Alix'})")
    tx.execute("INSERT (:Person {name: 'Gus'})")
    tx.commit()  # Both inserts committed atomically

# Rollback on error
with db.begin_transaction() as tx:
    tx.execute("INSERT (:Person {name: 'Vincent'})")
    tx.rollback()  # Changes discarded
```

## Direct Node/Edge API

Create nodes and edges programmatically:

```python
# Create a node
node = db.create_node(["Person"], {"name": "Alix", "age": 30})
print(f"Created node with ID: {node.id}")

# Create an edge
db.execute("""
    MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'})
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

## Named Graphs and Schemas

Manage multiple named graphs and schemas within a single database:

```python
# Create and switch between named graphs
db.create_graph("social")
db.create_graph("work")
db.set_graph("social")
print(db.current_graph())  # 'social'
print(db.list_graphs())    # ['social', 'work']
db.reset_graph()           # Back to default graph
social = db.graph("social")  # A handle: works in 'social' without switching
social.execute("MATCH (n) RETURN count(n)")
db.drop_graph("work")

# Schema management
db.set_schema("v1")
print(db.current_schema())  # 'v1'
db.reset_schema()           # Back to default schema
```

## Graph Projections

Create filtered virtual views of your graph for algorithms or scoped queries.
A projection views the graph `set_graph()` selects when it is created (the
default graph when none is selected) and keeps reading that graph:

```python
# Create a projection with specific labels and edge types
db.create_projection("people",
    node_labels=["Person"],
    edge_types=["KNOWS", "WORKS_WITH"]
)

# Run graph algorithms on the projection only
scores = db.algorithms.pagerank(projection="people", directed=False)
communities = db.algorithms.louvain(projection="people")

print(db.list_projections())  # ['people']
db.drop_projection("people")
```

Without `projection=`, `db.algorithms` reads the graph `set_graph()` selects,
like `execute()` and `CALL grafeo.pagerank()`; `db.graph("name").algorithms`
reads one named graph.

`key=` keys a result by a node property instead of the node id, and makes
PageRank, Louvain and label propagation independent of the order the graph was
loaded in (see [Determinism](../../algorithms/index.md#determinism)):

```python
scores = db.algorithms.pagerank(projection="people", directed=False, key="id")
```

## Data Import

Import CSV and JSON Lines files directly as graph nodes:

```python
# CSV import (each row becomes a node with the given label)
count = db.import_csv("users.csv", "Person", headers=True)
print(f"Imported {count} nodes")

# JSON Lines import (each line is a JSON object)
count = db.import_jsonl("events.jsonl", "Event")
```

## Change Data Capture

Control CDC on a per-transaction basis:

```python
# Enable CDC for a specific transaction
with db.begin_transaction_with_cdc(True) as tx:
    tx.execute("INSERT (:Person {name: 'Alix'})")
    tx.commit()
    # CDC events are captured for this transaction

# Disable CDC for bulk operations
with db.begin_transaction_with_cdc(False) as tx:
    tx.execute("INSERT (:Temp {data: 'scratch'})")
    tx.commit()
    # No CDC events generated
```

## Backup and Restore

```python
# Full backup
db.backup_full("/backups/full")

# Incremental backup (WAL records since last backup)
db.backup_incremental("/backups/incr")

# Restore to a specific epoch
grafeo.GrafeoDB.restore_to_epoch("/backups/full", epoch=100, output_path="./restored")
```

## Persistence

```python
# Save database to disk
db.save("backup.db")

# Create an in-memory copy (with its indexes and constraints)
memory_db = db.to_memory()

# Load database as in-memory
db = grafeo.GrafeoDB.open_in_memory("my_graph.db")
```
