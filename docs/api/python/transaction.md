---
title: graphos.Transaction
description: Transaction class reference.
tags:
  - api
  - python
---

# graphos.Transaction

Transaction management.

## Methods

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
with session.transaction() as tx:
    session.execute("INSERT (:Node)")
    # Auto-commit on success, rollback on exception
```

## Example

```python
with db.session() as session:
    # Manual transaction
    session.begin()
    try:
        session.execute("INSERT (:Person {name: 'Alice'})")
        session.execute("INSERT (:Person {name: 'Bob'})")
        session.commit()
    except Exception:
        session.rollback()
        raise

    # Or use context manager
    with session.transaction():
        session.execute("INSERT (:Person {name: 'Carol'})")
```
