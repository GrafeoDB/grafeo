# grafeo

Python bindings for [Grafeo](https://grafeo.dev), a high-performance, embeddable graph database with a Rust core.

## Installation

```bash
uv add grafeo
# or: pip install grafeo
```

## Quick Start

```python
from grafeo import GrafeoDB

# In-memory database
db = GrafeoDB()

# Or persistent
# db = GrafeoDB("./my-graph")

# Create nodes
db.execute("INSERT (:Person {name: 'Alix', age: 30})")
db.execute("INSERT (:Person {name: 'Gus', age: 25})")
db.execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")

# Query the graph
result = db.execute("MATCH (p:Person) WHERE p.age > 20 RETURN p.name, p.age")
for row in result:
    print(row)
```

## API Overview

### Database

```python
db = GrafeoDB()              # in-memory
db = GrafeoDB("./path")      # persistent
db = GrafeoDB.open("./path") # open existing

db.node_count   # number of nodes
db.edge_count   # number of edges
```

### Query Languages

```python
result = db.execute(gql)                        # GQL (ISO standard)
result = db.execute(gql, {"name": "Alix"})     # GQL with parameters
result = db.execute_cypher(query)               # Cypher
result = db.execute_sparql(query)               # SPARQL
result = db.execute_gremlin(query)              # Gremlin
result = db.execute_graphql(query)              # GraphQL
result = db.execute_sql(query)                  # SQL/PGQ (SQL:2023)
```

### Node & Edge CRUD

```python
node = db.create_node(["Person"], {"name": "Alix", "age": 30})
edge = db.create_edge(source_id, target_id, "KNOWS", {"since": 2024})

n = db.get_node(node_id)   # Node or None
e = db.get_edge(edge_id)   # Edge or None

db.set_node_property(node_id, "key", "value")
db.set_edge_property(edge_id, "key", "value")

db.delete_node(node_id)
db.delete_edge(edge_id)
```

### Transactions

```python
# Context manager (auto-rollback on exception)
with db.begin_transaction() as tx:
    tx.execute("INSERT (:Person {name: 'Harm'})")
    tx.commit()

# With isolation levels (string or enum)
from grafeo import IsolationLevel
with db.begin_transaction(IsolationLevel.SERIALIZABLE) as tx:
    tx.execute("MATCH (n:Person) SET n.verified = true")
    tx.commit()

# Per-transaction CDC override
with db.begin_transaction_with_cdc(True) as tx:
    tx.execute("INSERT (:AuditedEvent {action: 'login'})")
    tx.commit()
```

### Named Graphs and Schemas

```python
db.create_graph("social")
db.set_graph("social")
print(db.list_graphs())       # ['social']
print(db.current_graph())     # 'social'
db.reset_graph()
db.drop_graph("social")

db.set_schema("v1")
print(db.current_schema())    # 'v1'
db.reset_schema()
```

### Graph Projections

```python
db.create_projection("people", node_labels=["Person"], edge_types=["KNOWS"])
print(db.list_projections())  # ['people']
db.drop_projection("people")
```

### Data Import

```python
count = db.import_csv("users.csv", "Person", headers=True)
count = db.import_jsonl("events.jsonl", "Event")
```

### Backup and Restore

```python
db.backup_full("/backups/full")
db.backup_incremental("/backups/incr")
GrafeoDB.restore_to_epoch("/backups/full", epoch=100, output_path="./restored")
```

### QueryResult

```python
result = db.execute("MATCH (n:Person) RETURN n.name, n.age")

result.columns          # column names
len(result)             # row count
result.execution_time_ms  # execution time (milliseconds)

for row in result:      # iterate rows
    print(row)

result[0]               # access by index
result.scalar()         # first column of first row
```

### Vector Search

```python
# Create an HNSW index
db.create_vector_index("Document", "embedding", dimensions=384)

# Insert vectors
node = db.create_node(["Document"], {"embedding": [0.1, 0.2, ...]})

# Search
results = db.vector_search("Document", "embedding", query_vector, k=10)
```

## Features

- GQL, Cypher, SPARQL, Gremlin, GraphQL, and SQL/PGQ query languages
- Full node/edge CRUD with native Python types
- ACID transactions with configurable isolation levels
- HNSW vector similarity search
- Property indexes for fast lookups
- Named graph and schema management
- Graph projections (filtered virtual views)
- CSV and JSON Lines import
- Incremental backup and restore
- Per-transaction CDC control
- Async support via `asyncio`
- Type stubs included

## Links

- [Documentation](https://grafeo.dev)
- [GitHub](https://github.com/GrafeoDB/grafeo)
- [npm Package](https://www.npmjs.com/package/@grafeo-db/js)
- [WASM Package](https://www.npmjs.com/package/@grafeo-db/wasm)

## License

Apache-2.0
