---
title: grafeo.GrafeoDB
description: Database class reference.
tags:
  - api
  - python
---

# grafeo.GrafeoDB

The main database class.

## Constructor

```python
GrafeoDB(
    path: Optional[str] = None
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | `None` | Database file path (None for in-memory) |

### Examples

```python
# In-memory database
db = grafeo.GrafeoDB()

# Persistent database
db = grafeo.GrafeoDB("my_graph.db")
```

## Query Methods

### execute()

Execute a GQL query.

```python
def execute(self, query: str) -> QueryResult
```

### execute_cypher()

Execute a Cypher query.

```python
def execute_cypher(self, query: str) -> QueryResult
```

### execute_gremlin()

Execute a Gremlin query.

```python
def execute_gremlin(self, query: str) -> QueryResult
```

### execute_graphql()

Execute a GraphQL query.

```python
def execute_graphql(self, query: str) -> QueryResult
```

### execute_sparql()

Execute a SPARQL query.

```python
def execute_sparql(self, query: str) -> QueryResult
```

## Node Operations

### create_node()

Create a node with labels and properties.

```python
def create_node(self, labels: List[str], properties: Dict[str, Any]) -> Node
```

### add_node_label()

Add a label to an existing node.

```python
def add_node_label(self, node_id: int, label: str) -> None
```

### remove_node_label()

Remove a label from a node.

```python
def remove_node_label(self, node_id: int, label: str) -> None
```

### get_node_labels()

Get all labels for a node.

```python
def get_node_labels(self, node_id: int) -> List[str]
```

## Transaction Methods

### begin_transaction()

Start a new transaction.

```python
def begin_transaction(self) -> Transaction
```

## Admin Methods

### info()

Get database information.

```python
def info(self) -> Dict[str, Any]
```

### detailed_stats()

Get detailed statistics.

```python
def detailed_stats(self) -> Dict[str, Any]
```

### schema()

Get schema information.

```python
def schema(self) -> Dict[str, Any]
```

### validate()

Validate database integrity.

```python
def validate(self) -> bool
```

## Example

```python
import grafeo

db = grafeo.GrafeoDB()

# Execute queries
db.execute("INSERT (:Person {name: 'Alice', age: 30})")

result = db.execute("MATCH (p:Person) RETURN p.name")
for row in result:
    print(row['p.name'])

# Use transactions
with db.begin_transaction() as tx:
    tx.execute("INSERT (:Person {name: 'Bob'})")
    tx.commit()
```
