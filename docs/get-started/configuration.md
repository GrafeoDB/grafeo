---
title: Configuration
description: Configure Grafeo for different use cases.
---

# Configuration

Grafeo can be configured for different use cases, from small embedded applications to high-performance server deployments.

## Database Modes

### In-Memory Mode

For temporary data or maximum performance:

=== "Python"

    ```python
    import grafeo

    # In-memory database (default)
    db = grafeo.GrafeoDB()
    ```

=== "Rust"

    ```rust
    use grafeo::GrafeoDB;

    let db = GrafeoDB::new_in_memory();
    ```

!!! note "Data Persistence"
    In-memory databases do not persist data. All data is lost when the database is closed.

### Persistent Mode

For durable storage:

=== "Python"

    ```python
    import grafeo

    # Persistent database
    db = grafeo.GrafeoDB(path="my_graph.db")
    ```

=== "Rust"

    ```rust
    use grafeo::GrafeoDB;

    let db = GrafeoDB::open("my_graph.db")?;
    ```

## Configuration Options

!!! note "Python and Node.js constructors"
    The Python and Node.js constructors accept only `path` and `cdc` parameters.
    Advanced configuration options (`memory_limit`, `read_only`, etc.)
    are only available in the Rust API via the `Config` builder.

In Rust, start from `Config::in_memory()`, `Config::persistent(path)` or `Config::read_only(path)` and change settings with the `with_*` and `without_*` methods. Later releases add settings, so a `Config` cannot be built as a struct literal.

### Memory Limit (Rust only)

Control the maximum memory usage:

```rust
use grafeo::{GrafeoDB, Config};

let config = Config::in_memory()
    .with_memory_limit(4 * 1024 * 1024 * 1024);  // 4 GB

let db = GrafeoDB::with_config(config)?;
```

### Deprecated settings

`Config::with_threads`, `with_query_logging`, `with_schema_constraints` and
`without_schema_constraints`, with their getters, are deprecated since 0.6.1 and
removed in 0.7.0. They never had an effect: nothing sizes a thread pool from
`threads`, no query is logged, and the catalog enforces declared constraints
either way.

## Configuration Reference

| Option | Type | Default | Availability | Description |
|--------|------|---------|--------------|-------------|
| `path` | `string` | `None` | All (Rust, Python, Node.js) | Database file path (None for in-memory) |
| `cdc` | `bool` | `false` | All (Rust, Python, Node.js) | Enable change data capture |
| `memory_limit` | `int` | System RAM | Rust only (`Config` builder) | Maximum memory usage in bytes |
| `read_only` | `bool` | `false` | Rust only (`Config` builder) | Open database in read-only mode |

## Environment Variables

Grafeo can also be configured via environment variables:

| Variable | Description |
|----------|-------------|
| `GRAFEO_MEMORY_LIMIT` | Maximum memory in bytes |
| `GRAFEO_LOG_LEVEL` | Logging level (error, warn, info, debug, trace) |

## Performance Tuning (Rust)

Advanced tuning is available via the Rust `Config` builder. Python and Node.js
users get sensible defaults automatically.

### For High-Throughput Workloads

```rust
use grafeo::{GrafeoDB, Config};

let config = Config::persistent("high_throughput.db")
    .with_memory_limit(8 * 1024 * 1024 * 1024);  // 8 GB

let db = GrafeoDB::with_config(config)?;
```

### For Low-Memory Environments

```rust
use grafeo::{GrafeoDB, Config};

let config = Config::persistent("embedded.db")
    .with_memory_limit(256 * 1024 * 1024);  // 256 MB

let db = GrafeoDB::with_config(config)?;
```

### For Read-Heavy Workloads

```rust
use grafeo::{GrafeoDB, Config};

// Several processes can open the same file read-only
let config = Config::read_only("replica.db");

let db = GrafeoDB::with_config(config)?;
```

## Next Steps

- [User Guide](../model-and-query/index.md) - Learn more about using Grafeo
- [Architecture](../internals/index.md) - Understand how Grafeo works
