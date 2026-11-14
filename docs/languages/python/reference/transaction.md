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
def execute(self, query: str, params: Optional[Dict] = None) -> QueryResult
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

## Properties

### isolation_level

The isolation level of this transaction.

```python
@property
def isolation_level(self) -> str
```

### is_active

Whether the transaction is still active (not yet committed or rolled back).

```python
@property
def is_active(self) -> bool
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
    tx.execute("INSERT (:Person {name: 'Alix'})")
    tx.execute("INSERT (:Person {name: 'Gus'})")
    tx.commit()  # Both inserts committed atomically

# Rollback on error
with db.begin_transaction() as tx:
    tx.execute("INSERT (:Person {name: 'Harm'})")
    tx.rollback()  # Changes discarded

# SPARQL transactions
with db.begin_transaction() as tx:
    tx.execute_sparql("""
        INSERT DATA {
            <http://example.org/alix> <http://xmlns.com/foaf/0.1/name> "Alix" .
        }
    """)
    tx.commit()
```
