---
title: Python API
description: Python API reference.
---

# Python API Reference

Complete reference for the `pygraphos` Python package.

## Installation

```bash
pip install pygraphos
```

## Quick Start

```python
import graphos

db = graphos.Database()
with db.session() as session:
    session.execute("INSERT (:Person {name: 'Alice'})")
```

## Classes

| Class | Description |
|-------|-------------|
| [Database](database.md) | Database connection and management |
| [Node](node.md) | Graph node representation |
| [Edge](edge.md) | Graph edge representation |
| [QueryResult](result.md) | Query result iteration |
| [Transaction](transaction.md) | Transaction management |
