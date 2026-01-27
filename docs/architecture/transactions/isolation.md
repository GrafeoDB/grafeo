---
title: Isolation Levels
description: Transaction isolation levels.
tags:
  - architecture
  - transactions
---

# Isolation Levels

Graphos supports Snapshot Isolation by default.

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
# Transaction 1
with db.session() as s1:
    s1.begin()
    # Sees snapshot at begin time

    # Transaction 2 commits changes
    with db.session() as s2:
        s2.execute("SET x = 100")

    # s1 still sees old value
    result = s1.execute("SELECT x")  # Old value
    s1.commit()
```

## Conflict Detection

Write-write conflicts are detected:

```python
# T1 and T2 both try to update same row
# Second to commit will fail with conflict error
```
