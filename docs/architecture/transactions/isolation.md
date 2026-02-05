---
title: Isolation Levels
description: Transaction isolation levels.
tags:
  - architecture
  - transactions
---

# Isolation Levels

Grafeo supports Snapshot Isolation by default.

## Snapshot Isolation

- Each transaction sees a consistent snapshot
- Reads never block writes
- Writes never block reads
- Write conflicts detected at commit

## Phenomena Prevented

| Phenomenon | Prevented? |
|------------|------------|
| Dirty Read | Yes |
| Non-Repeatable Read | Yes |
| Phantom Read | Yes |
| Write Skew | Partially |

## Example

```python
# Transaction 1 sees a consistent snapshot
with db.begin_transaction() as tx1:
    # Sees snapshot at transaction begin time

    # Meanwhile, another transaction commits changes
    db.execute("MATCH (n {id: 'x'}) SET n.value = 100")

    # tx1 still sees the old value (snapshot isolation)
    result = tx1.execute("MATCH (n {id: 'x'}) RETURN n.value")
    tx1.commit()
```

## Conflict Detection

Write-write conflicts are detected:

```python
# T1 and T2 both try to update same row
# Second to commit will fail with conflict error
```
