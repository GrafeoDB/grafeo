---
title: graphos.Database
description: Database class reference.
tags:
  - api
  - python
---

# graphos.Database

The main database class.

## Constructor

```python
Database(
    path: Optional[str] = None,
    memory_limit: Optional[int] = None,
    threads: Optional[int] = None,
    read_only: bool = False
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | `None` | Database file path (None for in-memory) |
| `memory_limit` | `int` | System RAM | Maximum memory in bytes |
| `threads` | `int` | CPU cores | Worker thread count |
| `read_only` | `bool` | `False` | Open in read-only mode |

## Methods

### session()

Create a new session.

```python
def session(self) -> Session
```

### close()

Close the database.

```python
def close(self) -> None
```

### checkpoint()

Force a checkpoint.

```python
def checkpoint(self) -> None
```

## Context Manager

```python
with graphos.Database(path="db.graphos") as db:
    with db.session() as session:
        session.execute("...")
```
