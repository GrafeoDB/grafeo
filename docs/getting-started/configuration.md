---
title: Configuration
description: Configure Grafeo for your use case.
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
    use grafeo::Database;

    let db = Database::open_in_memory()?;
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
    use grafeo::Database;

    let db = Database::open("my_graph.db")?;
    ```

## Configuration Options

### Memory Limit

Control the maximum memory usage:

=== "Python"

    ```python
    db = grafeo.GrafeoDB(
        path="my_graph.db",
        memory_limit=4 * 1024 * 1024 * 1024  # 4 GB
    )
    ```

=== "Rust"

    ```rust
    use grafeo::{Database, Config};

    let config = Config::builder()
        .memory_limit(4 * 1024 * 1024 * 1024)  // 4 GB
        .build()?;

    let db = Database::open_with_config("my_graph.db", config)?;
    ```

### Thread Pool Size

Configure parallelism:

=== "Python"

    ```python
    db = grafeo.GrafeoDB(
        path="my_graph.db",
        threads=8
    )
    ```

=== "Rust"

    ```rust
    use grafeo::{Database, Config};

    let config = Config::builder()
        .threads(8)
        .build()?;

    let db = Database::open_with_config("my_graph.db", config)?;
    ```

!!! tip "Default Thread Count"
    By default, Grafeo uses the number of available CPU cores.

## Configuration Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `path` | `string` | `None` | Database file path (None for in-memory) |
| `memory_limit` | `int` | System RAM | Maximum memory usage in bytes |
| `threads` | `int` | CPU cores | Number of worker threads |
| `read_only` | `bool` | `false` | Open database in read-only mode |

## Environment Variables

Grafeo can also be configured via environment variables:

| Variable | Description |
|----------|-------------|
| `GRAFEO_MEMORY_LIMIT` | Maximum memory in bytes |
| `GRAFEO_THREADS` | Number of worker threads |
| `GRAFEO_LOG_LEVEL` | Logging level (error, warn, info, debug, trace) |

## Performance Tuning

### For High-Throughput Workloads

```python
db = grafeo.GrafeoDB(
    path="high_throughput.db",
    memory_limit=8 * 1024 * 1024 * 1024,  # 8 GB
    threads=16
)
```

### For Low-Memory Environments

```python
db = grafeo.GrafeoDB(
    path="embedded.db",
    memory_limit=256 * 1024 * 1024,  # 256 MB
    threads=2
)
```

### For Read-Heavy Workloads

```python
# Multiple read replicas can be opened read-only
db = grafeo.GrafeoDB(
    path="replica.db",
    read_only=True
)
```

## Next Steps

- [User Guide](../user-guide/index.md) - Learn more about using Grafeo
- [Architecture](../architecture/index.md) - Understand how Grafeo works
