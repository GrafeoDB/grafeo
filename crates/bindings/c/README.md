# grafeo-c

C FFI bindings for [Grafeo](https://grafeo.dev), a high-performance, embeddable graph database with a Rust core.

## Building

```bash
# From the Grafeo repository root:
cargo build --release -p grafeo-c --features full

# Output:
#   target/release/libgrafeo_c.so      (Linux)
#   target/release/libgrafeo_c.dylib   (macOS)
#   target/release/grafeo_c.dll        (Windows)
```

The header file is at `crates/bindings/c/grafeo.h`.

## Quick Start

```c
#include "grafeo.h"
#include <stdio.h>

int main(void) {
    /* Open an in-memory database (returns NULL on error) */
    GrafeoDatabase *db = grafeo_open_memory();
    if (!db) {
        fprintf(stderr, "Error: %s\n", grafeo_last_error());
        return 1;
    }

    /* Create nodes with labels (JSON array) and properties (JSON object) */
    uint64_t alix = grafeo_create_node(db, "[\"Person\"]", "{\"name\":\"Alix\",\"age\":30}");
    uint64_t gus  = grafeo_create_node(db, "[\"Person\"]", "{\"name\":\"Gus\",\"age\":25}");

    /* Create an edge */
    grafeo_create_edge(db, alix, gus, "KNOWS", "{\"since\":2020}");

    /* Query with GQL */
    GrafeoResult *r = grafeo_execute(db, "MATCH (p:Person) RETURN p.name, p.age");
    if (r) {
        printf("Rows: %zu\n", grafeo_result_row_count(r));
        printf("JSON: %s\n", grafeo_result_json(r));
        grafeo_free_result(r);
    } else {
        fprintf(stderr, "Query error: %s\n", grafeo_last_error());
    }

    /* Cleanup */
    grafeo_close(db);
    grafeo_free_database(db);
    return 0;
}
```

Compile with:

```bash
gcc -o example example.c -lgrafeo_c -L/path/to/target/release
```

## API Overview

### Lifecycle

```c
GrafeoDatabase* grafeo_open_memory(void);                   /* in-memory */
GrafeoDatabase* grafeo_open(const char* path);              /* persistent (a single file) */
GrafeoDatabase* grafeo_open_read_only(const char* path);    /* read-only */
GrafeoDatabase* grafeo_open_single_file(const char* path);  /* deprecated: same as grafeo_open, removed in 0.7.0 */
GrafeoStatus    grafeo_close(GrafeoDatabase* db);           /* flush and close */
void            grafeo_free_database(GrafeoDatabase* db);   /* free handle */
const char*     grafeo_version(void);                       /* library version */
```

### Query Execution

All query functions return `GrafeoResult*`, or `NULL` on error.

```c
GrafeoResult* grafeo_execute(db, query);                          /* GQL */
GrafeoResult* grafeo_execute_with_params(db, query, params_json); /* GQL + params */
GrafeoResult* grafeo_execute_cypher(db, query);                   /* Cypher */
GrafeoResult* grafeo_execute_gremlin(db, query);                  /* Gremlin */
GrafeoResult* grafeo_execute_graphql(db, query);                  /* GraphQL */
GrafeoResult* grafeo_execute_sparql(db, query);                   /* SPARQL */
GrafeoResult* grafeo_execute_sql(db, query);                      /* SQL/PGQ */
GrafeoResult* grafeo_execute_language(db, language, query, params_json);  /* any language */
```

Each language also has a `_with_params` variant (e.g. `grafeo_execute_cypher_with_params`).

### Results

```c
const char* grafeo_result_json(const GrafeoResult* r);              /* JSON rows */
size_t      grafeo_result_row_count(const GrafeoResult* r);         /* row count */
double      grafeo_result_execution_time_ms(const GrafeoResult* r); /* timing */
uint64_t    grafeo_result_rows_scanned(const GrafeoResult* r);      /* rows scanned */
const char* grafeo_result_nodes_json(const GrafeoResult* r);        /* extracted nodes */
const char* grafeo_result_edges_json(const GrafeoResult* r);        /* extracted edges */
void        grafeo_free_result(GrafeoResult* r);
```

All `const char*` pointers are valid until the parent `GrafeoResult` is freed.

### Node & Edge CRUD

```c
uint64_t     grafeo_create_node(db, labels_json, properties_json);
uint64_t     grafeo_create_edge(db, source_id, target_id, edge_type, properties_json);
GrafeoStatus grafeo_get_node(db, id, &node);
GrafeoStatus grafeo_get_edge(db, id, &edge);
int32_t      grafeo_delete_node(db, id);
int32_t      grafeo_delete_edge(db, id);
GrafeoStatus grafeo_set_node_property(db, id, key, value_json);
GrafeoStatus grafeo_set_edge_property(db, id, key, value_json);
int32_t      grafeo_remove_node_property(db, id, key);
int32_t      grafeo_remove_edge_property(db, id, key);
int32_t      grafeo_add_node_label(db, id, label);
int32_t      grafeo_remove_node_label(db, id, label);
char*        grafeo_get_node_labels(db, id);  /* free with grafeo_free_string */
```

### Transactions

```c
GrafeoTransaction* tx = grafeo_begin_transaction(db);
GrafeoResult*      r  = grafeo_transaction_execute(tx, "INSERT (:Person {name: 'Alix'})");
grafeo_free_result(r);
grafeo_commit(tx);          /* or grafeo_rollback(tx) */
grafeo_free_transaction(tx);
```

Also available: `grafeo_begin_transaction_with_isolation`, `grafeo_transaction_execute_with_params`, and `grafeo_transaction_execute_language`.

### Vector Search

```c
grafeo_create_vector_index(db, "Document", "embedding", 384, "cosine", 16, 200);

uint64_t *ids = NULL;
float *distances = NULL;
size_t count = 0;
grafeo_vector_search(db, "Document", "embedding", query_vec, 384, 5, -1, &ids, &distances, &count);
grafeo_free_vector_results(ids, distances, count);
```

Also available: `grafeo_mmr_search`, `grafeo_batch_create_nodes`, `grafeo_drop_vector_index`, `grafeo_rebuild_vector_index`.

### Error Handling

Functions that return pointers use `NULL` for errors. Functions that return `GrafeoStatus` use `GRAFEO_OK` (0) for success. In both cases, call `grafeo_last_error()` for details:

```c
GrafeoResult *r = grafeo_execute(db, query);
if (!r) {
    fprintf(stderr, "Error: %s\n", grafeo_last_error());
    grafeo_clear_error();
}
```

### Memory Management

- Opaque pointers (`GrafeoDatabase*`, `GrafeoResult*`, etc.) must be freed with their `grafeo_free_*` function
- `const char*` from accessor functions (e.g. `grafeo_result_json`, `grafeo_edge_type`) are valid until the parent is freed: do NOT free them
- `char*` from functions like `grafeo_info` and `grafeo_get_node_labels` are caller-owned: free with `grafeo_free_string`

## Features

- GQL, Cypher, SPARQL, Gremlin, GraphQL, and SQL/PGQ query languages
- Full node/edge CRUD with JSON property serialization
- ACID transactions with configurable isolation levels
- HNSW vector similarity search with batch operations and MMR
- Property indexes for fast lookups
- Schema context for multi-tenant graphs
- Change data capture (CDC)
- Thread-safe for concurrent use

## Links

- [Documentation](https://grafeo.dev)
- [GitHub](https://github.com/GrafeoDB/grafeo)
- [Go Bindings](https://github.com/GrafeoDB/grafeo/tree/main/crates/bindings/go) (uses this library via CGO)
- [C# Bindings](https://github.com/GrafeoDB/grafeo/tree/main/crates/bindings/csharp) (uses this library via P/Invoke)
- [Dart Bindings](https://github.com/GrafeoDB/grafeo/tree/main/crates/bindings/dart) (uses this library via dart:ffi)

## License

Apache-2.0
