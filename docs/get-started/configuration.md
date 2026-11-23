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

!!! note "Settings in the bindings"
    The bindings take the memory settings and the [query limits](#query-limits) when a
    database opens: as keywords of the Python `GrafeoDB(...)` constructor (`memory_limit`,
    `max_query_length`, ...) and as options of the Node.js `GrafeoDB.create(path, options)`
    (`memoryLimit`, `maxQueryLength`, ...). A read-only database opens with
    `GrafeoDB.open_read_only(path)` (Node.js `openReadOnly`). The other settings on this page
    are set in Rust, with the `Config` builder.

In Rust, start from `Config::in_memory()`, `Config::persistent(path)` or `Config::read_only(path)` and change settings with the `with_*` and `without_*` methods. Later releases add settings, so a `Config` cannot be built as a struct literal.

### Memory Limit

Control the maximum memory usage:

```rust
use grafeo::{GrafeoDB, Config};

let config = Config::in_memory()
    .with_memory_limit(4 * 1024 * 1024 * 1024);  // 4 GB

let db = GrafeoDB::with_config(config)?;
```

### Query limits

Four settings bound what one statement may cost. Each is set per open (never stored in the
database file), and each refuses 0:

| Setting | Default | Past it |
|---------|---------|---------|
| `Config::with_max_query_length(bytes)` | 16 MiB | the statement fails before it is read, `GRAFEO-Q002` (GQLSTATUS `42000`) |
| `Config::with_max_identifier_length(bytes)` | 16 KiB | the statement fails, `GRAFEO-Q002` (GQLSTATUS `42000`), naming the name's place |
| `Config::with_path_expansion_limit(expansions)` | 100 million | the statement fails, `GRAFEO-Q009` (GQLSTATUS `22000`) |
| `Config::with_path_search_budget(bytes)` | a quarter of the memory limit, at most 256 MiB | the path search fails, `GRAFEO-S001` |

- The query length counts the statement's UTF-8 bytes as given, comments and whitespace
  included; values passed as parameters do not count, so pass large values as parameters.
- A name is counted in bytes as written between its delimiters: identifiers, delimited
  identifiers and parameter names, SPARQL variables, prefixed names and blank node labels,
  Gremlin identifiers and GraphQL names. String literals and IRIs are not names.
- A path expansion is one edge a variable-length or shortest-path search reads to extend a path,
  plus one per node and edge it copies out of a path into a row (a path variable, an edge list).
  All the path searches of one statement share the limit. Searches for `DISTINCT` targets and
  RDF property paths are not counted.
- The path search budget bounds the memory one search holds at once, never above the memory
  limit.

Every error names its setting. The bindings take the same settings: `path_search_budget`,
`path_expansion_limit`, `max_query_length` and `max_identifier_length` in Python;
`pathSearchBudget`, `pathExpansionLimit`, `maxQueryLength` and `maxIdentifierLength` in Node.js,
WebAssembly and Dart; `grafeo_options_set_path_search_budget` and the like in C; the `Set...`
methods of `GrafeoOptions` in C# and the `Options` fields in Go.

```rust
use grafeo::{GrafeoDB, Config};

let config = Config::in_memory()
    .with_max_query_length(1 << 20)            // 1 MiB
    .with_path_expansion_limit(10_000_000);

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
| `memory_limit` | `int` | System RAM | All (Python `memory_limit`, Node.js `memoryLimit`) | Maximum memory usage in bytes |
| `read_only` | `bool` | `false` | All (Rust `Config::read_only`, Python `open_read_only`, Node.js `openReadOnly`) | Open database in read-only mode |
| `max_query_length` | `int` | 16 MiB | All (see [Query limits](#query-limits)) | Most bytes of a statement's text |
| `max_identifier_length` | `int` | 16 KiB | All (see [Query limits](#query-limits)) | Most bytes of one name in a statement |
| `path_expansion_limit` | `int` | 100 million | All (see [Query limits](#query-limits)) | Most path expansions of one statement |
| `path_search_budget` | `int` | a quarter of the memory limit, at most 256 MiB | All (see [Query limits](#query-limits)) | Most bytes one path search holds |

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
