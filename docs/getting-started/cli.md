---
title: Command-Line Interface
description: Admin CLI for Grafeo database management.
tags:
  - getting-started
  - cli
  - admin
---

# Command-Line Interface

Grafeo provides command-line tools for database administration. The CLI is designed for operators and DevOps — use the Python API for application logic, CLI for inspection and maintenance.

## Two CLI Options

Grafeo offers two CLI implementations:

| CLI | Installation | Features |
|-----|--------------|----------|
| **Rust CLI** (`grafeo-cli`) | `cargo install grafeo-cli` | Full-featured, native performance |
| **Python CLI** (`grafeo[cli]`) | `uv add grafeo[cli]` | Core commands, Python ecosystem |

### Feature Comparison

| Command | Rust CLI | Python CLI |
|---------|----------|------------|
| `info` | ✅ | ✅ |
| `stats` | ✅ | ✅ |
| `schema` | ✅ | ✅ |
| `validate` | ✅ | ✅ |
| `backup create/restore` | ✅ | ✅ |
| `wal status/checkpoint` | ✅ | ✅ |
| `index list/stats` | ✅ | — |
| `data dump/load` | ✅ | — |
| `compact` | ✅ | — |

## Installation

### Rust CLI (Recommended)

For full functionality and native performance:

```bash
cargo install grafeo-cli
```

### Python CLI

For environments where Rust is not available:

```bash
uv add grafeo[cli]
# or
pip install grafeo[cli]
```

The Python CLI requires the `click` package (installed automatically with `[cli]` extra).

## Commands

### Database Inspection

```bash
# Overview: counts, size, mode
grafeo info ./mydb

# Detailed statistics
grafeo stats ./mydb

# Schema: labels, edge types, property keys
grafeo schema ./mydb

# Integrity check
grafeo validate ./mydb
```

### Index Management (Rust CLI only)

```bash
# List all indexes
grafeo index list ./mydb

# Index statistics
grafeo index stats ./mydb
```

### Backup & Restore

```bash
# Create a native backup
grafeo backup create ./mydb -o backup.grafeo

# Restore from backup
grafeo backup restore backup.grafeo ./restored --force
```

### Data Export & Import (Rust CLI only)

```bash
# Export to portable format (Parquet for LPG, Turtle for RDF)
grafeo data dump ./mydb -o ./export/

# Import from dump
grafeo data load ./export/ ./newdb
```

### WAL Management

```bash
# Show WAL status
grafeo wal status ./mydb

# Force checkpoint
grafeo wal checkpoint ./mydb
```

### Compaction (Rust CLI only)

```bash
# Compact the database
grafeo compact ./mydb

# Dry run (show what would be done)
grafeo compact ./mydb --dry-run
```

## Output Formats

All commands support multiple output formats:

```bash
# Human-readable table (default)
grafeo info ./mydb --format table

# Machine-readable JSON
grafeo info ./mydb --format json
```

## Global Options

| Option | Description |
|--------|-------------|
| `--format` | Output format: `table` (default) or `json` |
| `--quiet`, `-q` | Suppress progress messages |
| `--verbose`, `-v` | Enable debug logging |
| `--help` | Show help |
| `--version` | Show version |

## Examples

### Check database health

```bash
$ grafeo info ./production.db
Property      | Value
--------------+-----------------
Mode          | lpg
Nodes         | 1,234,567
Edges         | 5,432,100
Persistent    | true
Path          | ./production.db
WAL Enabled   | true
Version       | 0.1.4
```

### Export to JSON for scripting

```bash
$ grafeo info ./mydb --format json
{
  "mode": "lpg",
  "node_count": 1234567,
  "edge_count": 5432100,
  "is_persistent": true,
  "path": "./production.db",
  "wal_enabled": true,
  "version": "0.1.4"
}
```

### Validate before deployment

```bash
$ grafeo validate ./mydb
✓ Database is valid

Errors: 0, Warnings: 0
```

## Python API Equivalents

The Python API provides the same functionality programmatically:

```python
import grafeo

db = grafeo.GrafeoDB.open("./mydb")

# Equivalent to: grafeo info ./mydb
print(db.info())

# Equivalent to: grafeo stats ./mydb
print(db.detailed_stats())

# Equivalent to: grafeo schema ./mydb
print(db.schema())

# Equivalent to: grafeo validate ./mydb
print(db.validate())

# Equivalent to: grafeo wal status ./mydb
print(db.wal_status())

# Equivalent to: grafeo wal checkpoint ./mydb
db.wal_checkpoint()

# Equivalent to: grafeo backup create ./mydb -o backup
db.save("./backup")
```
