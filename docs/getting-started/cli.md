---
title: Command-Line Interface
description: Grafeo CLI for querying, inspecting and maintaining graph databases.
tags:
  - getting-started
  - cli
  - admin
---

# Command-Line Interface

Grafeo ships a single Rust CLI binary (`grafeo`) for querying, inspecting and maintaining databases. It includes an interactive REPL, admin commands and multiple output formats.

## Installation

Install via any of these methods: they all provide the same CLI:

=== "Cargo (Recommended)"

    ```bash
    cargo install grafeo-cli
    ```

=== "pip / uv"

    ```bash
    uv add grafeo-cli
    # or
    pip install grafeo-cli
    ```

=== "npm"

    ```bash
    npm install -g @grafeo-db/cli
    # or one-shot:
    npx @grafeo-db/cli --version
    ```

=== "Download"

    Pre-built binaries for all platforms are attached to every
    [GitHub release](https://github.com/GrafeoDB/grafeo/releases).

Verify the installation:

```bash
grafeo version
```

## Quick Start

```bash
# Create a new database
grafeo init ./mydb

# Run a query
grafeo query ./mydb "INSERT (:Person {name: 'Alix', age: 30})"
grafeo query ./mydb "MATCH (n:Person) RETURN n.name, n.age"

# Launch the interactive shell
grafeo shell ./mydb
```

## Commands

### Query Execution

```bash
# Inline query
grafeo query ./mydb "MATCH (n) RETURN n LIMIT 10"

# From a file
grafeo query ./mydb --file query.gql

# From stdin
echo "MATCH (n) RETURN count(n)" | grafeo query ./mydb --stdin

# With parameters
grafeo query ./mydb "MATCH (n {name: \$name}) RETURN n" -p name=Alix

# Choose query language (default: gql)
grafeo query ./mydb "MATCH (n) RETURN n" --lang cypher
grafeo query ./mydb "SELECT * FROM GRAPH_TABLE ..." --lang sql

# Show execution time
grafeo query ./mydb "MATCH (n) RETURN n" --timing

# Truncate wide columns
grafeo query ./mydb "MATCH (n) RETURN n" --max-width 40
```

### Interactive Shell (REPL)

```bash
grafeo shell ./mydb
```

```
Grafeo 0.5.43 - Lpg mode, 42 nodes, 87 edges
Type :help for commands, :quit to exit.

grafeo> MATCH (n:Person) RETURN n.name, n.age
+----------+-------+
| n.name   | n.age |
+----------+-------+
| "Alix"   | 30    |
| "Gus"    | 25    |
+----------+-------+
2 rows (0.8ms)

grafeo> :begin
Transaction started.
grafeo[tx]> INSERT (:Person {name: 'Harm', age: 45})
grafeo[tx]> :commit
Transaction committed.
```

**Meta-commands:**

| Command | Description |
|---------|-------------|
| `:help` | Show available commands |
| `:quit` / Ctrl-D | Exit the shell |
| `:schema` | Show labels, edge types, property keys |
| `:info` | Show database info |
| `:stats` | Show detailed statistics |
| `:format <f>` | Set output format (`table`, `json`, `csv`) |
| `:timing` | Toggle query timing display |
| `:begin` | Start a transaction |
| `:commit` | Commit the current transaction |
| `:rollback` | Roll back the current transaction |

Transaction keywords (`BEGIN`, `COMMIT`, `ROLLBACK`) also work as plain text.

### Database Creation

```bash
# Create an LPG database (default)
grafeo init ./mydb

# Create an RDF database
grafeo init ./mydb --mode rdf
```

### Inspection

```bash
# Overview: counts, size, mode
grafeo info ./mydb

# Detailed statistics
grafeo stats ./mydb

# Schema: labels, edge types, property keys
grafeo schema ./mydb

# Integrity check (exit code 2 on failure)
grafeo validate ./mydb
```

### Index Management

```bash
grafeo index list ./mydb
grafeo index stats ./mydb
```

### Backup & Restore

```bash
# Snapshot backup (single file)
grafeo backup create ./mydb -o backup.grafeo
grafeo backup restore backup.grafeo ./restored --force

# Incremental backup (WAL-based, for production use)
grafeo backup full ./mydb -o /backups/full
grafeo backup incremental ./mydb -o /backups/incr
grafeo backup status /backups/full
grafeo backup restore-to-epoch /backups/full --epoch 100 -o ./restored
```

### Data Import

```bash
# Import CSV as graph nodes
grafeo import csv data.csv --path ./mydb --label Person
grafeo import csv data.csv --path ./mydb --label Person --headers false --separator ';'

# Import JSON Lines as graph nodes
grafeo import jsonl events.jsonl --path ./mydb --label Event
```

### Data Export

```bash
# Native dump/load
grafeo data dump ./mydb -o ./export/
grafeo data load ./export/ ./newdb

# Graph interchange formats
grafeo data dump ./mydb -o graph.gexf --export-format gexf
grafeo data dump ./mydb -o graph.graphml --export-format graphml
```

### WAL Management

```bash
grafeo wal status ./mydb
grafeo wal checkpoint ./mydb
```

### Compaction

```bash
grafeo compact ./mydb
grafeo compact ./mydb --dry-run
```

### Shell Completions

```bash
# Generate completions for your shell
grafeo completions bash > ~/.local/share/bash-completion/completions/grafeo
grafeo completions zsh > ~/.zfunc/_grafeo
grafeo completions fish > ~/.config/fish/completions/grafeo.fish
grafeo completions powershell >> $PROFILE
```

### Version & Build Info

```bash
$ grafeo version
grafeo 0.5.43

Build:
  rustc:    1.91.1
  target:   x86_64
  os:       linux
  features: gql, cypher, sparql, sql-pgq

Paths:
  config:   /home/user/.config/grafeo
  history:  /home/user/.config/grafeo/history
```

## Output Formats

All commands support multiple output formats:

```bash
# Auto-detect: table on TTY, JSON when piped
grafeo info ./mydb

# Explicit format
grafeo info ./mydb --format table
grafeo info ./mydb --format json
grafeo info ./mydb --format csv
```

## Global Options

| Option | Description |
|--------|-------------|
| `--format <auto\|table\|json\|csv>` | Output format (default: `auto`) |
| `--quiet`, `-q` | Suppress progress messages |
| `--verbose`, `-v` | Enable debug logging |
| `--no-color` | Disable colored output (also respects `NO_COLOR` env var) |
| `--color` | Force colored output even when piped |
| `--help` | Show help |
| `--version` | Show version |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error (runtime, I/O, query) |
| 2 | Validation failed (`grafeo validate` found errors) |

## Python API Equivalents

The Python API provides the same functionality programmatically:

```python
import grafeo

db = grafeo.GrafeoDB("./mydb")

# Equivalent to: grafeo info ./mydb
print(db.info())

# Equivalent to: grafeo stats ./mydb
print(db.detailed_stats())

# Equivalent to: grafeo schema ./mydb
print(db.schema())

# Equivalent to: grafeo validate ./mydb
print(db.validate())

# Equivalent to: grafeo query ./mydb "MATCH (n) RETURN n"
result = db.execute("MATCH (n) RETURN n")
```

## Migrating from the Python CLI

!!! note "Python CLI removed in 0.4.4"
    The `grafeo[cli]` Python CLI (Click-based) has been removed. Install the Rust binary
    instead via `cargo install grafeo-cli`, `pip install grafeo-cli`, or
    `npm install -g @grafeo-db/cli`. All previous commands are available with the same
    syntax, plus new features: `query`, `shell`, `init`, CSV output and shell completions.
