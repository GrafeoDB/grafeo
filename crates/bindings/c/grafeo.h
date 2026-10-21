/* Grafeo C API
 *
 * Link against libgrafeo_c.so (Linux), libgrafeo_c.dylib (macOS),
 * or grafeo_c.dll (Windows).
 *
 * Memory management:
 *   - Opaque pointers must be freed with their grafeo_free_* function.
 *   - Strings documented as "free with grafeo_free_string" are caller-owned.
 *   - Pointers documented as "valid until free" must NOT be freed separately.
 *
 * Error handling:
 *   - Functions return GrafeoStatus (0 = success).
 *   - On error, call grafeo_last_error() for a human-readable message.
 */

#ifndef GRAFEO_H
#define GRAFEO_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Status codes -------------------------------------------------------- */

/* GRAFEO_ERROR_TRANSACTION covers errors a retry of the transaction can
   fix (conflicts) and invalid transaction states. A write to a closed
   database (GRAFEO-T007 in the message), and after a commit that did not
   complete every later commit and every call that persists or copies the
   store (GRAFEO-T008: reopen the database; transactions still begin and
   reads still work), return GRAFEO_ERROR_DATABASE: no retry fixes them. */

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

/* ---- Transaction isolation levels ---------------------------------------- */

typedef enum {
    GRAFEO_ISOLATION_READ_COMMITTED = 0,
    GRAFEO_ISOLATION_SNAPSHOT       = 1,
    GRAFEO_ISOLATION_SERIALIZABLE   = 2
} GrafeoIsolationLevel;

typedef enum {
    GRAFEO_DIRECTION_OUT  = 0,
    GRAFEO_DIRECTION_IN   = 1,
    GRAFEO_DIRECTION_BOTH = 2
} GrafeoDirection;

/* ---- Opaque types -------------------------------------------------------- */

typedef struct GrafeoDatabase    GrafeoDatabase;
typedef struct GrafeoTransaction GrafeoTransaction;
typedef struct GrafeoResult      GrafeoResult;
typedef struct GrafeoGraph       GrafeoGraph;
typedef struct GrafeoNode        GrafeoNode;
typedef struct GrafeoEdge        GrafeoEdge;
typedef struct GrafeoStream      GrafeoStream;
typedef struct GrafeoAsOf       GrafeoAsOf;

/* ---- Error handling ------------------------------------------------------ */

const char* grafeo_last_error(void);
/* The GQLSTATUS of the last error, five characters such as "22012" (division
 * by zero) or "42001" (syntax error); NULL if no error. Valid until the next
 * call on this thread; do not free. */
const char* grafeo_last_error_gql_status(void);
void        grafeo_clear_error(void);

/* Marks a function that a later release removes; compilers warn on its use. */
#if defined(__GNUC__) || defined(__clang__)
#  define GRAFEO_DEPRECATED(message) __attribute__((deprecated(message)))
#elif defined(_MSC_VER)
#  define GRAFEO_DEPRECATED(message) __declspec(deprecated(message))
#else
#  define GRAFEO_DEPRECATED(message)
#endif

/* ---- Lifecycle ----------------------------------------------------------- */

GrafeoDatabase* grafeo_open_memory(void);
GrafeoDatabase* grafeo_open(const char* path);
GrafeoDatabase* grafeo_open_read_only(const char* path);
GRAFEO_DEPRECATED("same as grafeo_open since 0.6.0; removed in 0.7.0")
GrafeoDatabase* grafeo_open_single_file(const char* path);
GrafeoStatus    grafeo_close(GrafeoDatabase* db);
void            grafeo_free_database(GrafeoDatabase* db);
const char*     grafeo_version(void);

/* ---- Change Data Capture ------------------------------------------------- */

void grafeo_set_cdc_enabled(GrafeoDatabase* db, bool enabled);
bool grafeo_is_cdc_enabled(GrafeoDatabase* db);

/* ---- Query execution ----------------------------------------------------- */

GrafeoResult* grafeo_execute(GrafeoDatabase* db, const char* query);
GrafeoResult* grafeo_execute_with_params(GrafeoDatabase* db, const char* query, const char* params_json);
GrafeoResult* grafeo_execute_cypher(GrafeoDatabase* db, const char* query);
GrafeoResult* grafeo_execute_cypher_with_params(GrafeoDatabase* db, const char* query, const char* params_json);
GrafeoResult* grafeo_execute_gremlin(GrafeoDatabase* db, const char* query);
GrafeoResult* grafeo_execute_gremlin_with_params(GrafeoDatabase* db, const char* query, const char* params_json);
GrafeoResult* grafeo_execute_graphql(GrafeoDatabase* db, const char* query);
GrafeoResult* grafeo_execute_graphql_with_params(GrafeoDatabase* db, const char* query, const char* params_json);
GrafeoResult* grafeo_execute_sparql(GrafeoDatabase* db, const char* query);
GrafeoResult* grafeo_execute_sparql_with_params(GrafeoDatabase* db, const char* query, const char* params_json);

/* Requires sql-pgq feature. */
GrafeoResult* grafeo_execute_sql(GrafeoDatabase* db, const char* query);
GrafeoResult* grafeo_execute_sql_with_params(GrafeoDatabase* db, const char* query, const char* params_json);

/* Unified language dispatcher: language is "gql", "cypher", "gremlin", "graphql", "sparql", or "sql".
 * params_json may be NULL. */
GrafeoResult* grafeo_execute_language(GrafeoDatabase* db, const char* language, const char* query, const char* params_json);

/* ---- Result access ------------------------------------------------------- */

const char* grafeo_result_json(const GrafeoResult* result);
size_t      grafeo_result_row_count(const GrafeoResult* result);
double      grafeo_result_execution_time_ms(const GrafeoResult* result);
uint64_t    grafeo_result_rows_scanned(const GrafeoResult* result);
/* What the statement wrote; 0 for a read and for NULL. */
uint64_t    grafeo_result_nodes_created(const GrafeoResult* result);
uint64_t    grafeo_result_nodes_deleted(const GrafeoResult* result);
uint64_t    grafeo_result_edges_created(const GrafeoResult* result);
uint64_t    grafeo_result_edges_deleted(const GrafeoResult* result);
uint64_t    grafeo_result_properties_set(const GrafeoResult* result);
uint64_t    grafeo_result_labels_added(const GrafeoResult* result);
uint64_t    grafeo_result_labels_removed(const GrafeoResult* result);
const char* grafeo_result_nodes_json(const GrafeoResult* result);
const char* grafeo_result_edges_json(const GrafeoResult* result);
void        grafeo_free_result(GrafeoResult* result);

/* ---- Streaming (experimental, 0.5.40+) ---------------------------------- */
/* Lazy cursor-based results: bounded memory regardless of result size.
 * Read-only, pull-only: rejects mutations, ORDER BY, aggregate, DISTINCT,
 * EXPLAIN, PROFILE, and session/schema commands. Use grafeo_execute for
 * those. */

/* Opens a streaming query. Returns NULL on error (check grafeo_last_error). */
GrafeoStream* grafeo_stream_open(GrafeoDatabase* db, const char* query);

/* Returns the column names as a JSON array. Caller frees with grafeo_free_string. */
char* grafeo_stream_columns_json(const GrafeoStream* stream);

/* Pulls the next row into *out_json as a JSON object.
 * On success (GRAFEO_OK) with non-NULL *out_json: caller frees the
 * string with grafeo_free_string. On success with NULL *out_json: stream
 * exhausted, stop iterating. On any other status, grafeo_last_error has
 * details. */
GrafeoStatus grafeo_stream_next_row_json(GrafeoStream* stream, char** out_json);

/* Frees a stream handle. Safe to call on NULL. */
void grafeo_stream_free(GrafeoStream* stream);

/* ---- Schema context ------------------------------------------------------ */

GrafeoStatus    grafeo_set_schema(GrafeoDatabase* db, const char* name);
GrafeoStatus    grafeo_reset_schema(GrafeoDatabase* db);
const char*     grafeo_current_schema(const GrafeoDatabase* db);

/* ---- Node CRUD ----------------------------------------------------------- */

uint64_t     grafeo_create_node(GrafeoDatabase* db, const char* labels_json, const char* properties_json);
GrafeoStatus grafeo_get_node(GrafeoDatabase* db, uint64_t id, GrafeoNode** out);
int32_t      grafeo_delete_node(GrafeoDatabase* db, uint64_t id);
int32_t      grafeo_detach_delete_node(GrafeoDatabase* db, uint64_t id);
GrafeoStatus grafeo_set_node_property(GrafeoDatabase* db, uint64_t id, const char* key, const char* value_json);
int32_t      grafeo_remove_node_property(GrafeoDatabase* db, uint64_t id, const char* key);
int32_t      grafeo_add_node_label(GrafeoDatabase* db, uint64_t id, const char* label);
int32_t      grafeo_remove_node_label(GrafeoDatabase* db, uint64_t id, const char* label);
char*        grafeo_get_node_labels(GrafeoDatabase* db, uint64_t id);

uint64_t    grafeo_node_id(const GrafeoNode* node);
const char* grafeo_node_labels_json(const GrafeoNode* node);
const char* grafeo_node_properties_json(const GrafeoNode* node);
void        grafeo_free_node(GrafeoNode* node);

/* ---- Edge CRUD ----------------------------------------------------------- */

uint64_t     grafeo_create_edge(GrafeoDatabase* db, uint64_t source_id, uint64_t target_id, const char* edge_type, const char* properties_json);
GrafeoStatus grafeo_get_edge(GrafeoDatabase* db, uint64_t id, GrafeoEdge** out);
int32_t      grafeo_delete_edge(GrafeoDatabase* db, uint64_t id);
GrafeoStatus grafeo_set_edge_property(GrafeoDatabase* db, uint64_t id, const char* key, const char* value_json);
int32_t      grafeo_remove_edge_property(GrafeoDatabase* db, uint64_t id, const char* key);

uint64_t    grafeo_edge_id(const GrafeoEdge* edge);
uint64_t    grafeo_edge_source_id(const GrafeoEdge* edge);
uint64_t    grafeo_edge_target_id(const GrafeoEdge* edge);
const char* grafeo_edge_type(const GrafeoEdge* edge);
const char* grafeo_edge_properties_json(const GrafeoEdge* edge);
void        grafeo_free_edge(GrafeoEdge* edge);

/* A node's edges without a query: neighbor i is reached through edge i. edge_types_json is NULL for every type,
 * else a JSON array of strings; GRAFEO_DIRECTION_BOTH lists a self-loop once. *out_neighbors and *out_edges hold
 * *out_count ids each (NULL when none); free each with grafeo_free_node_ids(ptr, *out_count). */
GrafeoStatus grafeo_get_node_edges(GrafeoDatabase* db, uint64_t node_id, GrafeoDirection direction,
                                   const char* edge_types_json, uint64_t** out_neighbors, uint64_t** out_edges,
                                   size_t* out_count);

/* ---- Property indexes ---------------------------------------------------- */

GrafeoStatus grafeo_create_property_index(GrafeoDatabase* db, const char* property);
int32_t      grafeo_drop_property_index(GrafeoDatabase* db, const char* property);
int32_t      grafeo_has_property_index(GrafeoDatabase* db, const char* property);
GrafeoStatus grafeo_find_nodes_by_property(GrafeoDatabase* db, const char* property, const char* value_json, uint64_t** out_ids, size_t* out_count);

/* ---- Graph projections ---------------------------------------------------- */

/* Return 1 (created / dropped), 0 (name taken / no such projection), -1 on error (grafeo_last_error). */
int32_t      grafeo_create_projection(GrafeoDatabase* db, const char* name, const char* const* node_labels, size_t num_labels, const char* const* edge_types, size_t num_types);
int32_t      grafeo_drop_projection(GrafeoDatabase* db, const char* name);
char*        grafeo_list_projections(GrafeoDatabase* db);   /* JSON array of names, NULL on error; free with grafeo_free_string */
void         grafeo_free_node_ids(uint64_t* ids, size_t count);

/* ---- Vector operations --------------------------------------------------- */

GrafeoStatus grafeo_create_vector_index(GrafeoDatabase* db, const char* label, const char* property, int32_t dimensions, const char* metric, int32_t m, int32_t ef_construction);
int32_t      grafeo_drop_vector_index(GrafeoDatabase* db, const char* label, const char* property);
GrafeoStatus grafeo_rebuild_vector_index(GrafeoDatabase* db, const char* label, const char* property);
GrafeoStatus grafeo_vector_search(GrafeoDatabase* db, const char* label, const char* property, const float* query, size_t query_len, size_t k, int32_t ef, uint64_t** out_ids, float** out_distances, size_t* out_count);
GrafeoStatus grafeo_mmr_search(GrafeoDatabase* db, const char* label, const char* property, const float* query, size_t query_len, size_t k, int32_t fetch_k, float lambda, int32_t ef, uint64_t** out_ids, float** out_distances, size_t* out_count);
/* Requires vector-index feature. *out_ids has *out_count entries; free with grafeo_free_node_ids(*out_ids, *out_count). */
GrafeoStatus grafeo_batch_create_nodes(GrafeoDatabase* db, const char* label, const char* property, const float* vectors, size_t vector_count, size_t dimensions, uint64_t** out_ids, size_t* out_count);
void         grafeo_free_vector_results(uint64_t* ids, float* distances, size_t count);

/* ---- Batch writes (one transaction each) ---- */

/* One node per object of rows_json (a JSON array of property objects), each with the labels of labels_json (a
 * JSON array of strings). Node ids in row order; free *out_ids with grafeo_free_node_ids(*out_ids, *out_count). */
GrafeoStatus grafeo_batch_create_nodes_with_props(GrafeoDatabase* db, const char* labels_json, const char* rows_json,
                                                  uint64_t** out_ids, size_t* out_count);
/* edges_json: JSON array of {"src": id, "dst": id, "type": "KNOWS", "properties": {...}} ("properties" optional).
 * Edge ids in input order; free *out_ids with grafeo_free_node_ids(*out_ids, *out_count). */
GrafeoStatus grafeo_batch_create_edges(GrafeoDatabase* db, const char* edges_json,
                                       uint64_t** out_ids, size_t* out_count);
/* Requires vector-index. queries: query_count * dimensions floats; ef 0 for the index default; filters_json NULL or
 * a JSON object of property equalities. *out_json: JSON array with one array per query of {"id", "distance"}
 * objects, nearest first; free with grafeo_free_string. */
GrafeoStatus grafeo_batch_vector_search(GrafeoDatabase* db, const char* label, const char* property,
                                        const float* queries, size_t query_count, size_t dimensions,
                                        size_t k, size_t ef, const char* filters_json, char** out_json);

/* ---- Upserts (one statement each; *out_json: {"created", "updated", "skipped", "skipped_rows"}, free with
 * grafeo_free_string) ----
 * Nodes: rows_json is a JSON array of property objects matched by the key and all labels of labels_json;
 * options_json NULL or {"key": "id", "replace": false}. Edges: rows name their endpoints in the source and target
 * fields; options_json NULL or {"key", "endpoint_key", "endpoint_labels", "src_field", "dst_field", "replace"}. */
GrafeoStatus grafeo_upsert_nodes(GrafeoDatabase* db, const char* labels_json, const char* rows_json,
                                 const char* options_json, char** out_json);
GrafeoStatus grafeo_upsert_edges(GrafeoDatabase* db, const char* edge_type, const char* rows_json,
                                 const char* options_json, char** out_json);

/* ---- Graph handles: one named graph without switching the current graph ----
 * grafeo_graph returns NULL when the graph does not exist; each call fails once it was dropped. The functions
 * work as their database counterparts. */
GrafeoGraph*  grafeo_graph(GrafeoDatabase* db, const char* name);
void          grafeo_free_graph(GrafeoGraph* graph);
GrafeoResult* grafeo_graph_execute(GrafeoGraph* graph, const char* query);
GrafeoResult* grafeo_graph_execute_with_params(GrafeoGraph* graph, const char* query, const char* params_json);
GrafeoResult* grafeo_graph_execute_language(GrafeoGraph* graph, const char* language, const char* query, const char* params_json);
uint64_t      grafeo_graph_create_node(GrafeoGraph* graph, const char* labels_json, const char* properties_json);
uint64_t      grafeo_graph_create_edge(GrafeoGraph* graph, uint64_t source_id, uint64_t target_id, const char* edge_type, const char* properties_json);
GrafeoStatus  grafeo_graph_get_node(GrafeoGraph* graph, uint64_t id, GrafeoNode** out);
GrafeoStatus  grafeo_graph_get_edge(GrafeoGraph* graph, uint64_t id, GrafeoEdge** out);
GrafeoStatus  grafeo_graph_get_node_edges(GrafeoGraph* graph, uint64_t node_id, GrafeoDirection direction, const char* edge_types_json, uint64_t** out_neighbors, uint64_t** out_edges, size_t* out_count);
GrafeoStatus  grafeo_graph_set_node_property(GrafeoGraph* graph, uint64_t id, const char* key, const char* value_json);
GrafeoStatus  grafeo_graph_set_edge_property(GrafeoGraph* graph, uint64_t id, const char* key, const char* value_json);
int32_t       grafeo_graph_remove_node_property(GrafeoGraph* graph, uint64_t id, const char* key);
int32_t       grafeo_graph_remove_edge_property(GrafeoGraph* graph, uint64_t id, const char* key);
int32_t       grafeo_graph_add_node_label(GrafeoGraph* graph, uint64_t id, const char* label);
int32_t       grafeo_graph_remove_node_label(GrafeoGraph* graph, uint64_t id, const char* label);
int32_t       grafeo_graph_delete_node(GrafeoGraph* graph, uint64_t id);
int32_t       grafeo_graph_detach_delete_node(GrafeoGraph* graph, uint64_t id);
int32_t       grafeo_graph_delete_edge(GrafeoGraph* graph, uint64_t id);
GrafeoStatus  grafeo_graph_batch_create_nodes_with_props(GrafeoGraph* graph, const char* labels_json, const char* rows_json, uint64_t** out_ids, size_t* out_count);
GrafeoStatus  grafeo_graph_batch_create_edges(GrafeoGraph* graph, const char* edges_json, uint64_t** out_ids, size_t* out_count);
GrafeoStatus  grafeo_graph_upsert_nodes(GrafeoGraph* graph, const char* labels_json, const char* rows_json, const char* options_json, char** out_json);
GrafeoStatus  grafeo_graph_upsert_edges(GrafeoGraph* graph, const char* edge_type, const char* rows_json, const char* options_json, char** out_json);

/* ---- Time travel: reads of the database as it was at an earlier epoch or time ----
 * A point is a kind and a value: GRAFEO_POINT_EPOCH with an epoch, GRAFEO_POINT_TIME with milliseconds since 1970
 * (UTC), GRAFEO_POINT_AGO with milliseconds before now. A point before the history the database keeps fails ("not
 * retained"); grafeo_open_with_history sets how much it keeps (GRAFEO_HISTORY_DURATION: duration_ms before now; a
 * NULL path opens an in-memory database; a database file stores the setting). A view keeps the history it reads
 * until grafeo_free_as_of; a query that writes through it is refused. grafeo_as_of_get_node and _get_edge give
 * the node or edge as it was (GRAFEO_ERROR_DATABASE when it did not exist then), and _get_node_property a value as
 * JSON (NULL when there was none; free with grafeo_free_string). */
#define GRAFEO_POINT_EPOCH 0
#define GRAFEO_POINT_TIME  1
#define GRAFEO_POINT_AGO   2
#define GRAFEO_HISTORY_OFF      0
#define GRAFEO_HISTORY_FOREVER  1
#define GRAFEO_HISTORY_DURATION 2
GrafeoDatabase* grafeo_open_with_history(const char* path, int32_t history_kind, uint64_t duration_ms);
uint64_t        grafeo_current_epoch(GrafeoDatabase* db);
GrafeoStatus    grafeo_epoch_as_of(GrafeoDatabase* db, int32_t point_kind, int64_t point_value, uint64_t* epoch_out);
GrafeoResult*   grafeo_execute_as_of(GrafeoDatabase* db, const char* query, const char* params_json, int32_t point_kind, int64_t point_value);
GrafeoAsOf*     grafeo_as_of(GrafeoDatabase* db, int32_t point_kind, int64_t point_value);
uint64_t        grafeo_as_of_epoch(const GrafeoAsOf* view);
GrafeoResult*   grafeo_as_of_execute(const GrafeoAsOf* view, const char* query, const char* params_json);
GrafeoStatus    grafeo_as_of_get_node(const GrafeoAsOf* view, uint64_t id, GrafeoNode** out);
GrafeoStatus    grafeo_as_of_get_edge(const GrafeoAsOf* view, uint64_t id, GrafeoEdge** out);
char*           grafeo_as_of_get_node_property(const GrafeoAsOf* view, uint64_t id, const char* key);
void            grafeo_free_as_of(GrafeoAsOf* view);

/* ---- Change data capture (requires cdc; JSON arrays of events, NULL on error, free with grafeo_free_string) ----
 * Each event: entity_id, entity_type ("node"/"edge"), graph, kind ("create"/"update"/"delete"), epoch, timestamp,
 * before, after, labels, before_labels, edge_type, src_id, dst_id, deleted_with. */
char*        grafeo_node_history(GrafeoDatabase* db, uint64_t id);
char*        grafeo_edge_history(GrafeoDatabase* db, uint64_t id);
char*        grafeo_node_history_since(GrafeoDatabase* db, uint64_t id, uint64_t since_epoch);
char*        grafeo_changes_between(GrafeoDatabase* db, uint64_t start_epoch, uint64_t end_epoch);

/* ---- Statistics ---------------------------------------------------------- */

size_t grafeo_node_count(GrafeoDatabase* db);
size_t grafeo_edge_count(GrafeoDatabase* db);

/* ---- Transactions -------------------------------------------------------- */

GrafeoTransaction* grafeo_begin_transaction(GrafeoDatabase* db);
GrafeoTransaction* grafeo_begin_transaction_with_isolation(GrafeoDatabase* db, GrafeoIsolationLevel isolation);
GrafeoResult*      grafeo_transaction_execute(GrafeoTransaction* tx, const char* query);
GrafeoResult*      grafeo_transaction_execute_with_params(GrafeoTransaction* tx, const char* query, const char* params_json);
/* Execute in a specific language within a transaction; params_json may be NULL. */
GrafeoResult*      grafeo_transaction_execute_language(GrafeoTransaction* tx, const char* language, const char* query, const char* params_json);
GrafeoStatus       grafeo_commit(GrafeoTransaction* tx);
GrafeoStatus       grafeo_rollback(GrafeoTransaction* tx);
void               grafeo_free_transaction(GrafeoTransaction* tx);

/* ---- Admin --------------------------------------------------------------- */

char*        grafeo_info(GrafeoDatabase* db);
GrafeoStatus grafeo_save(GrafeoDatabase* db, const char* path);
GrafeoStatus grafeo_wal_checkpoint(GrafeoDatabase* db);
/* Automatic checkpoints: every interval_ms milliseconds and once the WAL holds
 * wal_bytes bytes since the last checkpoint; 0 turns a trigger off. */
GrafeoStatus grafeo_set_checkpoint_triggers(GrafeoDatabase* db, uint64_t interval_ms, uint64_t wal_bytes);
/* How the automatic checkpoints fare, as JSON. Free with grafeo_free_string. */
char*        grafeo_checkpoint_status(GrafeoDatabase* db);
GrafeoStatus grafeo_compact(GrafeoDatabase* db);

/* ---- Memory management --------------------------------------------------- */

void grafeo_free_string(char* s);

#ifdef __cplusplus
}
#endif

#endif /* GRAFEO_H */
