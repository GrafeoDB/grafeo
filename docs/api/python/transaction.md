---
title: grafeo.Transaction
description: Transaction class reference.
tags:
  - api
  - python
---

# grafeo.Transaction

Transaction management.

## Methods

### execute()

Execute a query within the transaction.

```python
def execute(self, query: str) -> QueryResult
```

### execute_sparql()

Execute a SPARQL query within the transaction.

```python
def execute_sparql(self, query: str) -> QueryResult
```

### commit()

Commit the transaction.

```python
def commit(self) -> None
```

### rollback()

Rollback the transaction.

```python
def rollback(self) -> None
```

## Context Manager

```python
with db.begin_transaction() as tx:
    tx.execute("INSERT (:Node)")
    tx.commit()
```

## Example

```python
# Using context manager
with db.begin_transaction() as tx:
    tx.execute("INSERT (:Person {name: 'Alice'})")
    tx.execute("INSERT (:Person {name: 'Bob'})")
    tx.commit()  # Both inserts committed atomically

# Rollback on error
with db.begin_transaction() as tx:
    tx.execute("INSERT (:Person {name: 'Carol'})")
    tx.rollback()  # Changes discarded

# SPARQL transactions
with db.begin_transaction() as tx:
    tx.execute_sparql("""
        INSERT DATA {
            <http://example.org/alice> <http://xmlns.com/foaf/0.1/name> "Alice" .
        }
    """)
    tx.commit()
```
