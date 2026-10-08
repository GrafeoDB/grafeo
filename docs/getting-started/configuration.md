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
    Advanced configuration options (`memory_limit`, `threads`, `read_only`, etc.)
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

### Thread Pool Size (Rust only)

Configure parallelism:

```rust
use grafeo::{GrafeoDB, Config};

let config = Config::in_memory().with_threads(8);

let db = GrafeoDB::with_config(config)?;
```

!!! tip "Default Thread Count"
    By default, Grafeo uses the number of available CPU cores.

## Configuration Reference

| Option | Type | Default | Availability | Description |
|--------|------|---------|--------------|-------------|
| `path` | `string` | `None` | All (Rust, Python, Node.js) | Database file path (None for in-memory) |
| `cdc` | `bool` | `false` | All (Rust, Python, Node.js) | Enable change data capture |
| `memory_limit` | `int` | System RAM | Rust only (`Config` builder) | Maximum memory usage in bytes |
| `threads` | `int` | CPU cores | Rust only (`Config` builder) | Number of worker threads |
| `read_only` | `bool` | `false` | Rust only (`Config` builder) | Open database in read-only mode |

## Environment Variables

Grafeo can also be configured via environment variables:

| Variable | Description |
|----------|-------------|
| `GRAFEO_MEMORY_LIMIT` | Maximum memory in bytes |
| `GRAFEO_THREADS` | Number of worker threads |
| `GRAFEO_LOG_LEVEL` | Logging level (error, warn, info, debug, trace) |

## Performance Tuning (Rust)

Advanced tuning is available via the Rust `Config` builder. Python and Node.js
users get sensible defaults automatically.

### For High-Throughput Workloads

```rust
use grafeo::{GrafeoDB, Config};

let config = Config::persistent("high_throughput.db")
    .with_memory_limit(8 * 1024 * 1024 * 1024)  // 8 GB
    .with_threads(16);

let db = GrafeoDB::with_config(config)?;
```

### For Low-Memory Environments

```rust
use grafeo::{GrafeoDB, Config};

let config = Config::persistent("embedded.db")
    .with_memory_limit(256 * 1024 * 1024)  // 256 MB
    .with_threads(2);

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

- [User Guide](../user-guide/index.md) - Learn more about using Grafeo
- [Architecture](../architecture/index.md) - Understand how Grafeo works
