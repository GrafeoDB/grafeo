---
title: Python API
description: Python API reference.
---

# Python API Reference

Complete reference for the `grafeo` Python package.

## Installation

```bash
uv add grafeo
```

## Quick Start

```python
import grafeo

db = grafeo.GrafeoDB()
db.execute("INSERT (:Person {name: 'Alice'})")
```

## Classes

| Class | Description |
|-------|-------------|
| [Database](database.md) | Database connection and management |
| [Node](node.md) | Graph node representation |
| [Edge](edge.md) | Graph edge representation |
| [QueryResult](result.md) | Query result iteration |
| [Transaction](transaction.md) | Transaction management |
