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

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Status codes -------------------------------------------------------- */

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

/* ---- Opaque types -------------------------------------------------------- */

typedef struct GrafeoDatabase    GrafeoDatabase;
typedef struct GrafeoTransaction GrafeoTransaction;
typedef struct GrafeoResult      GrafeoResult;
typedef struct GrafeoNode        GrafeoNode;
typedef struct GrafeoEdge        GrafeoEdge;

/* ---- Error handling ------------------------------------------------------ */

const char* grafeo_last_error(void);
void        grafeo_clear_error(void);

/* ---- Lifecycle ----------------------------------------------------------- */

GrafeoDatabase* grafeo_open_memory(void);
GrafeoDatabase* grafeo_open(const char* path);
GrafeoStatus    grafeo_close(GrafeoDatabase* db);
void            grafeo_free_database(GrafeoDatabase* db);
const char*     grafeo_version(void);

/* ---- Query execution ----------------------------------------------------- */

GrafeoResult* grafeo_execute(GrafeoDatabase* db, const char* query);
GrafeoResult* grafeo_execute_with_params(GrafeoDatabase* db, const char* query, const char* params_json);
GrafeoResult* grafeo_execute_cypher(GrafeoDatabase* db, const char* query);
GrafeoResult* grafeo_execute_gremlin(GrafeoDatabase* db, const char* query);
GrafeoResult* grafeo_execute_graphql(GrafeoDatabase* db, const char* query);
GrafeoResult* grafeo_execute_sparql(GrafeoDatabase* db, const char* query);

/* ---- Result access ------------------------------------------------------- */

const char* grafeo_result_json(const GrafeoResult* result);
size_t      grafeo_result_row_count(const GrafeoResult* result);
double      grafeo_result_execution_time_ms(const GrafeoResult* result);
uint64_t    grafeo_result_rows_scanned(const GrafeoResult* result);
void        grafeo_free_result(GrafeoResult* result);

/* ---- Node CRUD ----------------------------------------------------------- */

uint64_t     grafeo_create_node(GrafeoDatabase* db, const char* labels_json, const char* properties_json);
GrafeoStatus grafeo_get_node(GrafeoDatabase* db, uint64_t id, GrafeoNode** out);
int32_t      grafeo_delete_node(GrafeoDatabase* db, uint64_t id);
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

/* ---- Property indexes ---------------------------------------------------- */

GrafeoStatus grafeo_create_property_index(GrafeoDatabase* db, const char* property);
int32_t      grafeo_drop_property_index(GrafeoDatabase* db, const char* property);
int32_t      grafeo_has_property_index(GrafeoDatabase* db, const char* property);
GrafeoStatus grafeo_find_nodes_by_property(GrafeoDatabase* db, const char* property, const char* value_json, uint64_t** out_ids, size_t* out_count);
void         grafeo_free_node_ids(uint64_t* ids, size_t count);

/* ---- Vector operations --------------------------------------------------- */

GrafeoStatus grafeo_create_vector_index(GrafeoDatabase* db, const char* label, const char* property, int32_t dimensions, const char* metric, int32_t m, int32_t ef_construction);
GrafeoStatus grafeo_vector_search(GrafeoDatabase* db, const char* label, const char* property, const float* query, size_t query_len, size_t k, int32_t ef, uint64_t** out_ids, float** out_distances, size_t* out_count);
GrafeoStatus grafeo_batch_create_nodes(GrafeoDatabase* db, const char* label, const char* property, const float* vectors, size_t vector_count, size_t dimensions, uint64_t** out_ids);
void         grafeo_free_vector_results(uint64_t* ids, float* distances, size_t count);

/* ---- Statistics ---------------------------------------------------------- */

size_t grafeo_node_count(GrafeoDatabase* db);
size_t grafeo_edge_count(GrafeoDatabase* db);

/* ---- Transactions -------------------------------------------------------- */

GrafeoTransaction* grafeo_begin_tx(GrafeoDatabase* db);
GrafeoTransaction* grafeo_begin_tx_with_isolation(GrafeoDatabase* db, int32_t isolation);
GrafeoResult*      grafeo_tx_execute(GrafeoTransaction* tx, const char* query);
GrafeoResult*      grafeo_tx_execute_with_params(GrafeoTransaction* tx, const char* query, const char* params_json);
GrafeoStatus       grafeo_commit(GrafeoTransaction* tx);
GrafeoStatus       grafeo_rollback(GrafeoTransaction* tx);
void               grafeo_free_transaction(GrafeoTransaction* tx);

/* ---- Admin --------------------------------------------------------------- */

char*        grafeo_info(GrafeoDatabase* db);
GrafeoStatus grafeo_save(GrafeoDatabase* db, const char* path);
GrafeoStatus grafeo_wal_checkpoint(GrafeoDatabase* db);

/* ---- Memory management --------------------------------------------------- */

void grafeo_free_string(char* s);

#ifdef __cplusplus
}
#endif

#endif /* GRAFEO_H */
