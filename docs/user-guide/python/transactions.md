---
title: Transactions
description: Transaction management in Python.
tags:
  - python
  - transactions
---

# Transactions

Grafeo supports ACID transactions with snapshot isolation.

## Auto-Commit Mode

By default, each query runs in auto-commit mode:

```python
# Each execute is automatically committed
db.execute("INSERT (:Person {name: 'Alice'})")
db.execute("INSERT (:Person {name: 'Bob'})")
```

## Explicit Transactions

For multiple operations in a single atomic transaction, use `begin_transaction()`:

```python
# Start an explicit transaction
with db.begin_transaction() as tx:
    tx.execute("INSERT (:Person {name: 'Alice'})")
    tx.execute("INSERT (:Person {name: 'Bob'})")
    tx.commit()  # Both inserts committed atomically
```

## Transaction Context Manager

The transaction context manager provides automatic commit on success:

```python
with db.begin_transaction() as tx:
    tx.execute("INSERT (:Person {name: 'Alice'})")
    tx.execute("INSERT (:Person {name: 'Bob'})")
    tx.commit()
# If no exception, transaction is committed
```

## Rollback

Discard all changes in a transaction:

```python
with db.begin_transaction() as tx:
    tx.execute("INSERT (:Person {name: 'Alice'})")

    # Decide to rollback
    tx.rollback()
    # Alice was not created
```

## Rollback on Error

When an exception occurs, call `rollback()` to discard changes:

```python
with db.begin_transaction() as tx:
    try:
        tx.execute("INSERT (:Person {name: 'Alice'})")
        tx.execute("INSERT (:Person {name: 'Bob'})")
        raise ValueError("Something went wrong")
        tx.commit()
    except ValueError:
        tx.rollback()  # Both inserts discarded
```

## SPARQL Transactions

SPARQL operations also support transactions:

```python
with db.begin_transaction() as tx:
    tx.execute_sparql("""
        INSERT DATA {
            <http://example.org/alice> <http://xmlns.com/foaf/0.1/name> "Alice" .
        }
    """)
    tx.commit()

# Or rollback SPARQL changes
with db.begin_transaction() as tx:
    tx.execute_sparql("""
        INSERT DATA {
            <http://example.org/bob> <http://xmlns.com/foaf/0.1/name> "Bob" .
        }
    """)
    tx.rollback()  # Bob was not created
```

## Snapshot Isolation

Grafeo uses snapshot isolation by default. Each transaction sees a consistent snapshot of the database at the time it started:

```python
# Transaction 1 begins
tx1 = db.begin_transaction()

# Transaction 2 makes changes and commits
with db.begin_transaction() as tx2:
    tx2.execute("INSERT (:Person {name: 'NewPerson'})")
    tx2.commit()

# Transaction 1 still sees old data (snapshot isolation)
result = tx1.execute("MATCH (p:Person) RETURN count(p)")
# Does not include NewPerson

tx1.commit()
```

## Best Practices

1. **Keep transactions short** - Long transactions hold resources and increase conflict potential
2. **Use auto-commit for single operations** - No need for explicit transactions
3. **Always commit or rollback** - Don't leave transactions hanging
4. **Handle errors properly** - Call `rollback()` when operations fail
