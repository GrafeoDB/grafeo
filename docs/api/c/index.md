---
title: C API
description: API reference for the grafeo-c FFI bindings.
---

# C API

C-compatible FFI layer for embedding Grafeo in any language. Used by the Go bindings via CGO, the C# bindings via P/Invoke, and the Dart bindings via dart:ffi.

## Building

```bash
cargo build --release -p grafeo-c --features full
```

Output:

- `target/release/libgrafeo_c.so` (Linux)
- `target/release/libgrafeo_c.dylib` (macOS)
- `target/release/grafeo_c.dll` (Windows)

Header: `crates/bindings/c/grafeo.h`

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

    /* Create an edge between them */
    grafeo_create_edge(db, alix, gus, "KNOWS", "{\"since\":2020}");

    /* Execute a GQL query */
    GrafeoResult *r = grafeo_execute(db, "MATCH (p:Person) RETURN p.name, p.age");
    if (r) {
        printf("Rows: %zu\n", grafeo_result_row_count(r));
        printf("JSON: %s\n", grafeo_result_json(r));
        grafeo_free_result(r);
    } else {
        fprintf(stderr, "Query error: %s\n", grafeo_last_error());
    }

    grafeo_close(db);
    grafeo_free_database(db);
    return 0;
}
```

Compile:

```bash
gcc -o example example.c -lgrafeo_c -L/path/to/target/release
```

## Error Handling

Functions that return pointers (`GrafeoDatabase*`, `GrafeoResult*`, `GrafeoTransaction*`) return `NULL` on error. Functions that return `GrafeoStatus` return `GRAFEO_OK` (0) on success. In both cases, call `grafeo_last_error()` on failure for a human-readable message:

```c
GrafeoResult *r = grafeo_execute(db, query);
if (!r) {
    fprintf(stderr, "Error: %s\n", grafeo_last_error());
}
```

### Status Codes

```c
typedef enum {
    GRAFEO_OK                  = 0,
    GRAFEO_ERROR_DATABASE      = 1,
    GRAFEO_ERROR_QUERY         = 2,
    GRAFEO_ERROR_TRANSACTION   = 3,
    GRAFEO_ERROR_STORAGE       = 4,
    GRAFEO_ERROR_IO            = 5,
    GRAFEO_ERROR_SERIALIZATION = 6,
    GRAFEO_ERROR_INTERNAL      = 7,
    GRAFEO_ERROR_NULL_POINTER  = 8,
    GRAFEO_ERROR_INVALID_UTF8  = 9
} GrafeoStatus;
```

### Error Functions

```c
const char* grafeo_last_error(void);   /* thread-local error message, or NULL */
void        grafeo_clear_error(void);  /* clear the last error */
```

The pointer from `grafeo_last_error()` is valid until the next FFI call on the same thread. Do NOT free it.

## Lifecycle

```c
GrafeoDatabase* grafeo_open_memory(void);                    /* in-memory */
GrafeoDatabase* grafeo_open(const char* path);               /* persistent (a single file) */
GrafeoDatabase* grafeo_open_read_only(const char* path);     /* read-only (shared lock) */
GrafeoDatabase* grafeo_open_single_file(const char* path);   /* deprecated: same as grafeo_open, removed in 0.7.0 */
GrafeoStatus    grafeo_close(GrafeoDatabase* db);            /* flush and close */
void            grafeo_free_database(GrafeoDatabase* db);    /* free handle */
const char*     grafeo_version(void);                        /* library version (static) */
```

A persistent database is a single file at rest, whatever the extension of its path. While it is open for writing it also has a WAL, the `<path>.wal/` directory next to it, which a successful `grafeo_close` writes into the file and removes. A close that fails (a checkpoint error, or after a commit that did not complete) keeps the WAL, and the next open recovers from it. A database written by 0.5.x is migrated on its first read-write open (see [Upgrading from 0.5](../../user-guide/persistence/persistent.md#upgrading-from-05)).

All `grafeo_open*` functions return `NULL` on error. Always call `grafeo_close` before `grafeo_free_database` to flush pending writes.

## Query Execution

All query functions return a `GrafeoResult*` pointer, or `NULL` on error.

### GQL (default)

```c
GrafeoResult* grafeo_execute(GrafeoDatabase* db, const char* query);
GrafeoResult* grafeo_execute_with_params(GrafeoDatabase* db, const char* query, const char* params_json);
```

### Other Languages

Each language has a base function and a `_with_params` variant. Parameters are passed as a JSON object string.

```c
/* Cypher (requires cypher feature) */
GrafeoResult* grafeo_execute_cypher(GrafeoDatabase* db, const char* query);
GrafeoResult* grafeo_execute_cypher_with_params(GrafeoDatabase* db, const char* query, const char* params_json);

/* Gremlin (requires gremlin feature) */
GrafeoResult* grafeo_execute_gremlin(GrafeoDatabase* db, const char* query);
GrafeoResult* grafeo_execute_gremlin_with_params(GrafeoDatabase* db, const char* query, const char* params_json);

/* GraphQL (requires graphql feature) */
GrafeoResult* grafeo_execute_graphql(GrafeoDatabase* db, const char* query);
GrafeoResult* grafeo_execute_graphql_with_params(GrafeoDatabase* db, const char* query, const char* params_json);

/* SPARQL (requires sparql feature) */
GrafeoResult* grafeo_execute_sparql(GrafeoDatabase* db, const char* query);
GrafeoResult* grafeo_execute_sparql_with_params(GrafeoDatabase* db, const char* query, const char* params_json);

/* SQL/PGQ (requires sql-pgq feature) */
GrafeoResult* grafeo_execute_sql(GrafeoDatabase* db, const char* query);
GrafeoResult* grafeo_execute_sql_with_params(GrafeoDatabase* db, const char* query, const char* params_json);
```

### Unified Language Dispatcher

Execute a query in any supported language. `params_json` may be `NULL`.

```c
GrafeoResult* grafeo_execute_language(GrafeoDatabase* db, const char* language, const char* query, const char* params_json);
```

The `language` parameter accepts: `"gql"`, `"cypher"`, `"gremlin"`, `"graphql"`, `"sparql"`, or `"sql"`.

## Result Access

```c
const char* grafeo_result_json(const GrafeoResult* r);              /* JSON array of row objects */
size_t      grafeo_result_row_count(const GrafeoResult* r);         /* number of rows */
double      grafeo_result_execution_time_ms(const GrafeoResult* r); /* execution time in ms */
uint64_t    grafeo_result_rows_scanned(const GrafeoResult* r);      /* estimated rows scanned */
const char* grafeo_result_nodes_json(const GrafeoResult* r);        /* JSON array of node objects */
const char* grafeo_result_edges_json(const GrafeoResult* r);        /* JSON array of edge objects */
void        grafeo_free_result(GrafeoResult* r);                    /* free the result */
```

All `const char*` pointers from result accessors are valid until `grafeo_free_result` is called. Do NOT free them separately.

The `grafeo_result_nodes_json` and `grafeo_result_edges_json` functions return deduplicated, typed entities extracted from the result. Node objects have the shape `{"element_type": "node", "id": ..., "labels": [...], "properties": {...}}`. Edge objects have the shape `{"element_type": "edge", "id": ..., "type": "...", "source_id": ..., "target_id": ..., "properties": {...}}`.

## Node CRUD

### Create and Delete Nodes

```c
/* Returns new node ID, or UINT64_MAX on error */
uint64_t grafeo_create_node(GrafeoDatabase* db, const char* labels_json, const char* properties_json);

/* Returns 1 if deleted, 0 if not found, -1 on null pointer */
int32_t grafeo_delete_node(GrafeoDatabase* db, uint64_t id);
```

`labels_json` is a JSON array of strings (e.g. `"[\"Person\",\"Employee\"]"`). `properties_json` is a JSON object (e.g. `"{\"name\":\"Alix\"}"`), or `NULL` for no properties.

### Get Node

```c
GrafeoStatus grafeo_get_node(GrafeoDatabase* db, uint64_t id, GrafeoNode** out);
```

On success, `*out` must be freed with `grafeo_free_node`. Access fields with:

```c
uint64_t    grafeo_node_id(const GrafeoNode* node);
const char* grafeo_node_labels_json(const GrafeoNode* node);
const char* grafeo_node_properties_json(const GrafeoNode* node);
void        grafeo_free_node(GrafeoNode* node);
```

### Node Properties and Labels

```c
GrafeoStatus grafeo_set_node_property(GrafeoDatabase* db, uint64_t id, const char* key, const char* value_json);
int32_t      grafeo_remove_node_property(GrafeoDatabase* db, uint64_t id, const char* key);
int32_t      grafeo_add_node_label(GrafeoDatabase* db, uint64_t id, const char* label);
int32_t      grafeo_remove_node_label(GrafeoDatabase* db, uint64_t id, const char* label);
char*        grafeo_get_node_labels(GrafeoDatabase* db, uint64_t id);  /* free with grafeo_free_string */
```

## Edge CRUD

### Create and Delete Edges

```c
/* Returns new edge ID, or UINT64_MAX on error */
uint64_t grafeo_create_edge(GrafeoDatabase* db, uint64_t source_id, uint64_t target_id, const char* edge_type, const char* properties_json);

/* Returns 1 if deleted, 0 if not found, -1 on null pointer */
int32_t grafeo_delete_edge(GrafeoDatabase* db, uint64_t id);
```

### Get Edge

```c
GrafeoStatus grafeo_get_edge(GrafeoDatabase* db, uint64_t id, GrafeoEdge** out);
```

On success, `*out` must be freed with `grafeo_free_edge`. Access fields with:

```c
uint64_t    grafeo_edge_id(const GrafeoEdge* edge);
uint64_t    grafeo_edge_source_id(const GrafeoEdge* edge);
uint64_t    grafeo_edge_target_id(const GrafeoEdge* edge);
const char* grafeo_edge_type(const GrafeoEdge* edge);
const char* grafeo_edge_properties_json(const GrafeoEdge* edge);
void        grafeo_free_edge(GrafeoEdge* edge);
```

### Edge Properties

```c
GrafeoStatus grafeo_set_edge_property(GrafeoDatabase* db, uint64_t id, const char* key, const char* value_json);
int32_t      grafeo_remove_edge_property(GrafeoDatabase* db, uint64_t id, const char* key);
```

## Transactions

Transactions use snapshot isolation by default. A transaction that is neither committed nor rolled back is automatically rolled back when freed.

```c
GrafeoTransaction* grafeo_begin_transaction(GrafeoDatabase* db);
GrafeoTransaction* grafeo_begin_transaction_with_isolation(GrafeoDatabase* db, GrafeoIsolationLevel isolation);
```

Isolation levels: `GRAFEO_ISOLATION_READ_COMMITTED` (0), `GRAFEO_ISOLATION_SNAPSHOT` (1), `GRAFEO_ISOLATION_SERIALIZABLE` (2).

### Execute Within a Transaction

```c
GrafeoResult* grafeo_transaction_execute(GrafeoTransaction* tx, const char* query);
GrafeoResult* grafeo_transaction_execute_with_params(GrafeoTransaction* tx, const char* query, const char* params_json);
GrafeoResult* grafeo_transaction_execute_language(GrafeoTransaction* tx, const char* language, const char* query, const char* params_json);
```

### Commit, Rollback, and Free

```c
GrafeoStatus grafeo_commit(GrafeoTransaction* tx);
GrafeoStatus grafeo_rollback(GrafeoTransaction* tx);
void         grafeo_free_transaction(GrafeoTransaction* tx);
```

### Transaction Example

```c
GrafeoTransaction *tx = grafeo_begin_transaction(db);
if (!tx) {
    fprintf(stderr, "Error: %s\n", grafeo_last_error());
    return;
}

GrafeoResult *r = grafeo_transaction_execute(tx, "INSERT (:Person {name: 'Alix'})");
if (r) grafeo_free_result(r);

if (grafeo_commit(tx) != GRAFEO_OK) {
    fprintf(stderr, "Commit failed: %s\n", grafeo_last_error());
}
grafeo_free_transaction(tx);
```

## Schema Context

```c
GrafeoStatus grafeo_set_schema(GrafeoDatabase* db, const char* name);
GrafeoStatus grafeo_reset_schema(GrafeoDatabase* db);
const char*  grafeo_current_schema(const GrafeoDatabase* db);  /* NULL if no schema set */
```

The pointer from `grafeo_current_schema` is valid until the next call to `grafeo_current_schema`, `grafeo_set_schema`, or `grafeo_reset_schema` on the same thread.

## Property Indexes

```c
GrafeoStatus grafeo_create_property_index(GrafeoDatabase* db, const char* property);
int32_t      grafeo_drop_property_index(GrafeoDatabase* db, const char* property);
int32_t      grafeo_has_property_index(GrafeoDatabase* db, const char* property);
```

`grafeo_drop_property_index` returns 1 if the index was dropped, 0 if there was none, and -1 on an error (see `grafeo_last_error`), such as a database whose `grafeo_close` ran.

### Find Nodes by Property

```c
GrafeoStatus grafeo_find_nodes_by_property(GrafeoDatabase* db, const char* property, const char* value_json, uint64_t** out_ids, size_t* out_count);
void         grafeo_free_node_ids(uint64_t* ids, size_t count);
```

Example:

```c
uint64_t *ids = NULL;
size_t count = 0;
if (grafeo_find_nodes_by_property(db, "name", "\"Alix\"", &ids, &count) == GRAFEO_OK) {
    for (size_t i = 0; i < count; i++) {
        printf("Node ID: %llu\n", (unsigned long long)ids[i]);
    }
    grafeo_free_node_ids(ids, count);
}
```

## Vector Search

Requires the `vector-index` feature.

### Index Management

```c
GrafeoStatus grafeo_create_vector_index(GrafeoDatabase* db, const char* label, const char* property, int32_t dimensions, const char* metric, int32_t m, int32_t ef_construction);
int32_t      grafeo_drop_vector_index(GrafeoDatabase* db, const char* label, const char* property);
GrafeoStatus grafeo_rebuild_vector_index(GrafeoDatabase* db, const char* label, const char* property);
```

Pass `-1` for `dimensions`, `m`, or `ef_construction` to use defaults. Pass `NULL` for `metric` to default to cosine similarity. `grafeo_drop_vector_index` returns 1 if the index was dropped, 0 if there was none, and -1 on an error (see `grafeo_last_error`): a null pointer, a name that is not UTF-8, or a database error such as a closed database.

### Nearest Neighbor Search

```c
GrafeoStatus grafeo_vector_search(GrafeoDatabase* db, const char* label, const char* property, const float* query, size_t query_len, size_t k, int32_t ef, uint64_t** out_ids, float** out_distances, size_t* out_count);
```

### MMR Search (Maximal Marginal Relevance)

```c
GrafeoStatus grafeo_mmr_search(GrafeoDatabase* db, const char* label, const char* property, const float* query, size_t query_len, size_t k, int32_t fetch_k, float lambda, int32_t ef, uint64_t** out_ids, float** out_distances, size_t* out_count);
```

### Batch Node Creation

```c
GrafeoStatus grafeo_batch_create_nodes(GrafeoDatabase* db, const char* label, const char* property, const float* vectors, size_t vector_count, size_t dimensions, uint64_t** out_ids, size_t* out_count);
```

### Freeing Vector Results

```c
void grafeo_free_vector_results(uint64_t* ids, float* distances, size_t count);
void grafeo_free_node_ids(uint64_t* ids, size_t count);  /* for batch_create_nodes */
```

### Vector Search Example

```c
/* Create a vector index */
grafeo_create_vector_index(db, "Document", "embedding", 384, "cosine", 16, 200);

/* Search for 5 nearest neighbors */
float query_vec[384] = { /* ... */ };
uint64_t *ids = NULL;
float *distances = NULL;
size_t count = 0;

if (grafeo_vector_search(db, "Document", "embedding", query_vec, 384, 5, -1, &ids, &distances, &count) == GRAFEO_OK) {
    for (size_t i = 0; i < count; i++) {
        printf("Node %llu, distance: %f\n", (unsigned long long)ids[i], distances[i]);
    }
    grafeo_free_vector_results(ids, distances, count);
}
```

## CDC (Change Data Capture)

Requires the `cdc` feature.

```c
void grafeo_set_cdc_enabled(GrafeoDatabase* db, bool enabled);
bool grafeo_is_cdc_enabled(GrafeoDatabase* db);
```

## Statistics

```c
size_t grafeo_node_count(GrafeoDatabase* db);
size_t grafeo_edge_count(GrafeoDatabase* db);
```

## Admin

```c
char*        grafeo_info(GrafeoDatabase* db);             /* JSON string, free with grafeo_free_string */
GrafeoStatus grafeo_save(GrafeoDatabase* db, const char* path);
GrafeoStatus grafeo_wal_checkpoint(GrafeoDatabase* db);
```

### Compact Store

Convert to a layered columnar store for faster queries. Requires the `compact-store` feature.

```c
GrafeoStatus grafeo_compact(GrafeoDatabase* db);
```

After this call, queries run with lower memory usage and faster traversal, and writes go to a mutable overlay that the next `grafeo_compact` merges into the base. A database opened with `grafeo_open_read_only` stays read-only: writes still return an error. Fails while a transaction is open.

## Memory Management

- Opaque pointers (`GrafeoDatabase*`, `GrafeoResult*`, `GrafeoTransaction*`, `GrafeoNode*`, `GrafeoEdge*`) must be freed with their `grafeo_free_*` function
- `const char*` pointers from accessor functions (e.g. `grafeo_result_json`, `grafeo_node_labels_json`) are valid until the parent object is freed: do NOT free them
- `char*` pointers from functions like `grafeo_info` and `grafeo_get_node_labels` are caller-owned: free with `grafeo_free_string`
- Array results from vector search and property lookups must be freed with `grafeo_free_vector_results` or `grafeo_free_node_ids`

```c
void grafeo_free_database(GrafeoDatabase* db);
void grafeo_free_result(GrafeoResult* r);
void grafeo_free_transaction(GrafeoTransaction* tx);
void grafeo_free_node(GrafeoNode* node);
void grafeo_free_edge(GrafeoEdge* edge);
void grafeo_free_string(char* s);
void grafeo_free_node_ids(uint64_t* ids, size_t count);
void grafeo_free_vector_results(uint64_t* ids, float* distances, size_t count);
```

## Links

- [GitHub](https://github.com/GrafeoDB/grafeo/tree/main/crates/bindings/c)
- [Go bindings](https://github.com/GrafeoDB/grafeo/tree/main/crates/bindings/go) (built on this library)
- [C# bindings](https://github.com/GrafeoDB/grafeo/tree/main/crates/bindings/csharp) (uses this library via P/Invoke)
- [Dart bindings](https://github.com/GrafeoDB/grafeo/tree/main/crates/bindings/dart) (uses this library via dart:ffi)
