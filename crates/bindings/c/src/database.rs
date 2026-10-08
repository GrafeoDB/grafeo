//! All `#[no_mangle] extern "C"` functions exposed by the Grafeo C API.

use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::c_char;
use std::sync::Arc;

use parking_lot::RwLock;

use grafeo_common::types::{EdgeId, NodeId};
use grafeo_engine::config::Config;
use grafeo_engine::database::GrafeoDB;

use crate::error::{GrafeoStatus, set_error, set_last_error, str_from_ptr};
use crate::types::{GrafeoDatabase, GrafeoEdge, GrafeoNode, GrafeoResult, GrafeoTransaction};

// ===== Thread-local storage =====

thread_local! {
    /// Stores the most recent schema name returned by [`grafeo_current_schema`].
    /// The pointer is valid until the next call to `grafeo_current_schema`,
    /// `grafeo_set_schema`, or `grafeo_reset_schema` on this thread.
    static LAST_SCHEMA: RefCell<Option<CString>> = const { RefCell::new(None) };
}

// ===== Helpers =====

/// Dereference an opaque database pointer, returning an error status on null.
macro_rules! db_ref {
    ($ptr:expr) => {{
        if $ptr.is_null() {
            set_last_error("Null database pointer");
            return GrafeoStatus::ErrorNullPointer;
        }
        // SAFETY: Caller guarantees ptr from grafeo_open* and not yet freed.
        unsafe { &*$ptr }
    }};
}

/// Same as `db_ref!` but returns a null pointer on error (for functions that
/// return pointers).
macro_rules! db_ref_or_null {
    ($ptr:expr) => {{
        if $ptr.is_null() {
            set_last_error("Null database pointer");
            return std::ptr::null_mut();
        }
        // SAFETY: Caller guarantees ptr from grafeo_open* and not yet freed.
        unsafe { &*$ptr }
    }};
}

/// Serialize a `QueryResult` into a `GrafeoResult`.
fn build_result(result: &grafeo_engine::database::QueryResult) -> *mut GrafeoResult {
    let json_rows: Vec<serde_json::Value> = result
        .rows()
        .iter()
        .map(|row| {
            let obj: serde_json::Map<String, serde_json::Value> = result
                .columns
                .iter()
                .zip(row.iter())
                .map(|(col, val)| (col.clone(), crate::types::value_to_json(val)))
                .collect();
            serde_json::Value::Object(obj)
        })
        .collect();

    let json_str = serde_json::to_string(&json_rows).unwrap_or_default();
    let c_json = CString::new(json_str).unwrap_or_default();

    // Extract typed entities (nodes and edges) from the result.
    let (raw_nodes, raw_edges) = grafeo_bindings_common::entity::extract_entities(result);

    let nodes_json_val: Vec<serde_json::Value> = raw_nodes
        .iter()
        .map(|n| {
            let mut obj = serde_json::Map::new();
            obj.insert(
                "element_type".to_string(),
                serde_json::Value::String("node".to_string()),
            );
            obj.insert("id".to_string(), serde_json::json!(n.id.as_u64()));
            obj.insert(
                "labels".to_string(),
                serde_json::Value::Array(
                    n.labels
                        .iter()
                        .map(|l| serde_json::Value::String(l.clone()))
                        .collect(),
                ),
            );
            let props: serde_json::Map<String, serde_json::Value> = n
                .properties
                .iter()
                .map(|(k, v)| (k.as_str().to_string(), crate::types::value_to_json(v)))
                .collect();
            obj.insert("properties".to_string(), serde_json::Value::Object(props));
            serde_json::Value::Object(obj)
        })
        .collect();

    let edges_json_val: Vec<serde_json::Value> = raw_edges
        .iter()
        .map(|e| {
            let mut obj = serde_json::Map::new();
            obj.insert(
                "element_type".to_string(),
                serde_json::Value::String("edge".to_string()),
            );
            obj.insert("id".to_string(), serde_json::json!(e.id.as_u64()));
            obj.insert(
                "type".to_string(),
                serde_json::Value::String(e.edge_type.clone()),
            );
            obj.insert(
                "source_id".to_string(),
                serde_json::json!(e.source_id.as_u64()),
            );
            obj.insert(
                "target_id".to_string(),
                serde_json::json!(e.target_id.as_u64()),
            );
            let props: serde_json::Map<String, serde_json::Value> = e
                .properties
                .iter()
                .map(|(k, v)| (k.as_str().to_string(), crate::types::value_to_json(v)))
                .collect();
            obj.insert("properties".to_string(), serde_json::Value::Object(props));
            serde_json::Value::Object(obj)
        })
        .collect();

    let nodes_str = serde_json::to_string(&nodes_json_val).unwrap_or_default();
    let edges_str = serde_json::to_string(&edges_json_val).unwrap_or_default();

    Box::into_raw(Box::new(GrafeoResult {
        json: c_json,
        row_count: result.rows().len(),
        execution_time_ms: result.execution_time_ms.unwrap_or(0.0),
        rows_scanned: result.rows_scanned.unwrap_or(0),
        nodes_json: CString::new(nodes_str).unwrap_or_default(),
        edges_json: CString::new(edges_str).unwrap_or_default(),
    }))
}

// =========================================================================
// Lifecycle
// =========================================================================

/// The id of a created node or edge, or `u64::MAX` with the error recorded
/// for `grafeo_last_error`.
fn created_id(result: grafeo_common::utils::error::Result<u64>) -> u64 {
    result.unwrap_or_else(|e| {
        set_last_error(&e.to_string());
        u64::MAX
    })
}

/// `1` or `0` for a yes-or-no result, or `-1` with the error recorded for
/// `grafeo_last_error`.
fn flag(result: grafeo_common::utils::error::Result<bool>) -> i32 {
    match result {
        Ok(done) => i32::from(done),
        Err(e) => {
            set_last_error(&e.to_string());
            -1
        }
    }
}

/// Create a new in-memory database.
///
/// Returns an opaque pointer, or null on error (check `grafeo_last_error()`).
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_open_memory() -> *mut GrafeoDatabase {
    let db = GrafeoDB::new_in_memory();
    Box::into_raw(Box::new(GrafeoDatabase {
        inner: Arc::new(RwLock::new(db)),
    }))
}

/// Open or create a persistent database at `path`.
///
/// The database is a single file, whatever the extension of `path`; a
/// database written by 0.5.x (a single file, usually `.grafeo`, or a WAL
/// directory) is migrated to the 0.6 format when it opens.
///
/// Returns an opaque pointer, or null on error.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_open(path: *const c_char) -> *mut GrafeoDatabase {
    let Ok(path_str) = str_from_ptr(path) else {
        return std::ptr::null_mut();
    };
    match GrafeoDB::with_config(Config::persistent(path_str)) {
        Ok(db) => Box::into_raw(Box::new(GrafeoDatabase {
            inner: Arc::new(RwLock::new(db)),
        })),
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// Open an existing database in read-only mode.
///
/// Uses a shared file lock, so multiple processes can read the same
/// database file concurrently. Mutations will return an error. A database
/// written by 0.5.x (a file, or a WAL directory) is read in place and not
/// migrated.
///
/// Returns an opaque pointer, or null on error.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_open_read_only(path: *const c_char) -> *mut GrafeoDatabase {
    let Ok(path_str) = str_from_ptr(path) else {
        return std::ptr::null_mut();
    };
    match GrafeoDB::with_config(Config::read_only(path_str)) {
        Ok(db) => Box::into_raw(Box::new(GrafeoDatabase {
            inner: Arc::new(RwLock::new(db)),
        })),
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// Open or create a persistent database at `path` using single-file format.
///
/// Deprecated since 0.6.0 and removed in 0.7.0: use [`grafeo_open`]. Since
/// 0.6 every database is a single file, whatever the extension of `path`, so
/// this is the same as [`grafeo_open`]. At rest only the file exists; a
/// sidecar `<path>.wal/` directory is used during operation and removed
/// automatically on close.
///
/// Returns an opaque pointer, or null on error (check `grafeo_last_error()`).
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_open_single_file(path: *const c_char) -> *mut GrafeoDatabase {
    let Ok(path_str) = str_from_ptr(path) else {
        return std::ptr::null_mut();
    };
    match GrafeoDB::with_config(Config::persistent(path_str)) {
        Ok(db) => Box::into_raw(Box::new(GrafeoDatabase {
            inner: Arc::new(RwLock::new(db)),
        })),
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// Close the database, flushing pending writes.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_close(db: *mut GrafeoDatabase) -> GrafeoStatus {
    if db.is_null() {
        return GrafeoStatus::Ok;
    }
    // SAFETY: Caller guarantees valid pointer from grafeo_open*.
    let db = unsafe { &*db };
    match db.inner.read().close() {
        Ok(()) => GrafeoStatus::Ok,
        Err(e) => set_error(&e),
    }
}

/// Free a database handle. Must be called after `grafeo_close`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_free_database(db: *mut GrafeoDatabase) {
    if !db.is_null() {
        // SAFETY: We take ownership back and drop it.
        unsafe { drop(Box::from_raw(db)) };
    }
}

/// Returns the library version string. The pointer is static and must NOT be freed.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_version() -> *const c_char {
    // Include a trailing NUL in the byte literal.
    static VERSION: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();
    VERSION.as_ptr().cast::<c_char>()
}

// =========================================================================
// Change Data Capture
// =========================================================================

/// Enable or disable CDC for all future sessions.
///
/// Does not affect sessions that were already created.
#[cfg(feature = "cdc")]
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_set_cdc_enabled(db: *mut GrafeoDatabase, enabled: bool) {
    if let Some(db) = unsafe { db.as_ref() } {
        db.inner.read().set_cdc_enabled(enabled);
    }
}

/// Returns whether CDC is currently enabled for new sessions.
#[cfg(feature = "cdc")]
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_is_cdc_enabled(db: *mut GrafeoDatabase) -> bool {
    unsafe { db.as_ref() }.map_or(false, |db| db.inner.read().is_cdc_enabled())
}

// =========================================================================
// Query Execution
// =========================================================================

/// Execute a GQL query. Returns a result pointer, or null on error.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_execute(
    db: *mut GrafeoDatabase,
    query: *const c_char,
) -> *mut GrafeoResult {
    let db = db_ref_or_null!(db);
    let Ok(query_str) = str_from_ptr(query) else {
        return std::ptr::null_mut();
    };
    let guard = db.inner.read();
    match guard.execute_language(query_str, "gql", None) {
        Ok(result) => build_result(&result),
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// Execute a GQL query with JSON-encoded parameters.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_execute_with_params(
    db: *mut GrafeoDatabase,
    query: *const c_char,
    params_json: *const c_char,
) -> *mut GrafeoResult {
    let db = db_ref_or_null!(db);
    let Ok(query_str) = str_from_ptr(query) else {
        return std::ptr::null_mut();
    };
    let guard = db.inner.read();
    let params = crate::types::parse_params(params_json);
    match guard.execute_language(query_str, "gql", params) {
        Ok(r) => build_result(&r),
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// Execute a Cypher query.
#[cfg(feature = "cypher")]
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_execute_cypher(
    db: *mut GrafeoDatabase,
    query: *const c_char,
) -> *mut GrafeoResult {
    let db = db_ref_or_null!(db);
    let Ok(query_str) = str_from_ptr(query) else {
        return std::ptr::null_mut();
    };
    match db.inner.read().execute_language(query_str, "cypher", None) {
        Ok(r) => build_result(&r),
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// Execute a Gremlin query.
#[cfg(feature = "gremlin")]
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_execute_gremlin(
    db: *mut GrafeoDatabase,
    query: *const c_char,
) -> *mut GrafeoResult {
    let db = db_ref_or_null!(db);
    let Ok(query_str) = str_from_ptr(query) else {
        return std::ptr::null_mut();
    };
    match db.inner.read().execute_language(query_str, "gremlin", None) {
        Ok(r) => build_result(&r),
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// Execute a GraphQL query.
#[cfg(feature = "graphql")]
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_execute_graphql(
    db: *mut GrafeoDatabase,
    query: *const c_char,
) -> *mut GrafeoResult {
    let db = db_ref_or_null!(db);
    let Ok(query_str) = str_from_ptr(query) else {
        return std::ptr::null_mut();
    };
    match db.inner.read().execute_language(query_str, "graphql", None) {
        Ok(r) => build_result(&r),
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// Execute a SPARQL query.
#[cfg(feature = "sparql")]
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_execute_sparql(
    db: *mut GrafeoDatabase,
    query: *const c_char,
) -> *mut GrafeoResult {
    let db = db_ref_or_null!(db);
    let Ok(query_str) = str_from_ptr(query) else {
        return std::ptr::null_mut();
    };
    match db.inner.read().execute_language(query_str, "sparql", None) {
        Ok(r) => build_result(&r),
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// Execute a SQL/PGQ query (SQL:2023 GRAPH_TABLE).
#[cfg(feature = "sql-pgq")]
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_execute_sql(
    db: *mut GrafeoDatabase,
    query: *const c_char,
) -> *mut GrafeoResult {
    let db = db_ref_or_null!(db);
    let Ok(query_str) = str_from_ptr(query) else {
        return std::ptr::null_mut();
    };
    match db.inner.read().execute_language(query_str, "sql", None) {
        Ok(r) => build_result(&r),
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

// =========================================================================
// Language-specific _with_params variants
// =========================================================================

/// Execute a Cypher query with named parameters (JSON object).
#[cfg(feature = "cypher")]
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_execute_cypher_with_params(
    db: *mut GrafeoDatabase,
    query: *const c_char,
    params_json: *const c_char,
) -> *mut GrafeoResult {
    let db = db_ref_or_null!(db);
    let Ok(query_str) = str_from_ptr(query) else {
        return std::ptr::null_mut();
    };
    let params = crate::types::parse_params(params_json);
    match db
        .inner
        .read()
        .execute_language(query_str, "cypher", params)
    {
        Ok(r) => build_result(&r),
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// Execute a Gremlin query with named parameters (JSON object).
#[cfg(feature = "gremlin")]
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_execute_gremlin_with_params(
    db: *mut GrafeoDatabase,
    query: *const c_char,
    params_json: *const c_char,
) -> *mut GrafeoResult {
    let db = db_ref_or_null!(db);
    let Ok(query_str) = str_from_ptr(query) else {
        return std::ptr::null_mut();
    };
    let params = crate::types::parse_params(params_json);
    match db
        .inner
        .read()
        .execute_language(query_str, "gremlin", params)
    {
        Ok(r) => build_result(&r),
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// Execute a GraphQL query with named parameters (JSON object).
#[cfg(feature = "graphql")]
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_execute_graphql_with_params(
    db: *mut GrafeoDatabase,
    query: *const c_char,
    params_json: *const c_char,
) -> *mut GrafeoResult {
    let db = db_ref_or_null!(db);
    let Ok(query_str) = str_from_ptr(query) else {
        return std::ptr::null_mut();
    };
    let params = crate::types::parse_params(params_json);
    match db
        .inner
        .read()
        .execute_language(query_str, "graphql", params)
    {
        Ok(r) => build_result(&r),
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// Execute a SPARQL query with named parameters (JSON object).
#[cfg(feature = "sparql")]
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_execute_sparql_with_params(
    db: *mut GrafeoDatabase,
    query: *const c_char,
    params_json: *const c_char,
) -> *mut GrafeoResult {
    let db = db_ref_or_null!(db);
    let Ok(query_str) = str_from_ptr(query) else {
        return std::ptr::null_mut();
    };
    let params = crate::types::parse_params(params_json);
    match db
        .inner
        .read()
        .execute_language(query_str, "sparql", params)
    {
        Ok(r) => build_result(&r),
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// Execute a SQL/PGQ query with named parameters (JSON object).
#[cfg(feature = "sql-pgq")]
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_execute_sql_with_params(
    db: *mut GrafeoDatabase,
    query: *const c_char,
    params_json: *const c_char,
) -> *mut GrafeoResult {
    let db = db_ref_or_null!(db);
    let Ok(query_str) = str_from_ptr(query) else {
        return std::ptr::null_mut();
    };
    let params = crate::types::parse_params(params_json);
    match db.inner.read().execute_language(query_str, "sql", params) {
        Ok(r) => build_result(&r),
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

// =========================================================================
// Unified language dispatcher
// =========================================================================

/// Execute a query in the given language with optional parameters.
///
/// `language` is one of: `"gql"`, `"cypher"`, `"gremlin"`, `"graphql"`,
/// `"sparql"`, `"sql"`. `params_json` may be null (no parameters).
///
/// Returns null on error; call `grafeo_last_error()` for details.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_execute_language(
    db: *mut GrafeoDatabase,
    language: *const c_char,
    query: *const c_char,
    params_json: *const c_char,
) -> *mut GrafeoResult {
    let db = db_ref_or_null!(db);
    let Ok(lang_str) = str_from_ptr(language) else {
        return std::ptr::null_mut();
    };
    let Ok(query_str) = str_from_ptr(query) else {
        return std::ptr::null_mut();
    };
    let params = crate::types::parse_params(params_json);
    match db
        .inner
        .read()
        .execute_language(query_str, lang_str, params)
    {
        Ok(r) => build_result(&r),
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

// =========================================================================
// Result Access
// =========================================================================

/// Get the JSON-encoded result rows. Returns null if result is null.
/// The pointer is valid until `grafeo_free_result` is called.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_result_json(result: *const GrafeoResult) -> *const c_char {
    if result.is_null() {
        return std::ptr::null();
    }
    // SAFETY: Caller guarantees valid pointer from grafeo_execute*.
    unsafe { &*result }.json.as_ptr()
}

/// Get the number of rows in the result.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_result_row_count(result: *const GrafeoResult) -> usize {
    if result.is_null() {
        return 0;
    }
    // SAFETY: Caller guarantees valid pointer.
    unsafe { &*result }.row_count
}

/// Get execution time in milliseconds.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_result_execution_time_ms(result: *const GrafeoResult) -> f64 {
    if result.is_null() {
        return 0.0;
    }
    // SAFETY: Caller guarantees valid pointer.
    unsafe { &*result }.execution_time_ms
}

/// Get estimated rows scanned.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_result_rows_scanned(result: *const GrafeoResult) -> u64 {
    if result.is_null() {
        return 0;
    }
    // SAFETY: Caller guarantees valid pointer.
    unsafe { &*result }.rows_scanned
}

/// Get a JSON array of typed node objects extracted from the result.
///
/// Each node object has the structure:
/// `{"element_type": "node", "id": <u64>, "labels": [...], "properties": {...}}`
///
/// Returns an empty array `"[]"` when the result contains no node entities.
/// The returned pointer is valid until `grafeo_free_result` is called.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_result_nodes_json(result: *const GrafeoResult) -> *const c_char {
    if result.is_null() {
        return std::ptr::null();
    }
    // SAFETY: Caller guarantees valid pointer from grafeo_execute*.
    unsafe { &*result }.nodes_json.as_ptr()
}

/// Get a JSON array of typed edge objects extracted from the result.
///
/// Each edge object has the structure:
/// `{"element_type": "edge", "id": <u64>, "type": "...", "source_id": <u64>, "target_id": <u64>, "properties": {...}}`
///
/// Returns an empty array `"[]"` when the result contains no edge entities.
/// The returned pointer is valid until `grafeo_free_result` is called.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_result_edges_json(result: *const GrafeoResult) -> *const c_char {
    if result.is_null() {
        return std::ptr::null();
    }
    // SAFETY: Caller guarantees valid pointer from grafeo_execute*.
    unsafe { &*result }.edges_json.as_ptr()
}

/// Free a query result.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_free_result(result: *mut GrafeoResult) {
    if !result.is_null() {
        // SAFETY: We take ownership back and drop it.
        unsafe { drop(Box::from_raw(result)) };
    }
}

// =========================================================================
// Schema context
// =========================================================================

/// Sets the current schema for subsequent execute calls.
///
/// Equivalent to `SESSION SET SCHEMA <name>` but persists across calls.
/// Call `grafeo_reset_schema` to clear it.
///
/// # Safety
/// `db` must be a valid pointer returned by `grafeo_open*`. `name` must be a
/// valid null-terminated UTF-8 string.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_set_schema(db: *mut GrafeoDatabase, name: *const c_char) -> GrafeoStatus {
    if db.is_null() || name.is_null() {
        set_last_error("Null pointer argument");
        return GrafeoStatus::ErrorNullPointer;
    }
    // SAFETY: Caller guarantees valid pointers.
    let db = unsafe { &*db };
    let Ok(name_str) = (unsafe { std::ffi::CStr::from_ptr(name) }).to_str() else {
        set_last_error("Invalid UTF-8 in schema name");
        return GrafeoStatus::ErrorInvalidUtf8;
    };
    if let Err(e) = db.inner.read().set_current_schema(Some(name_str)) {
        set_last_error(&e.to_string());
        return GrafeoStatus::ErrorQuery;
    }
    GrafeoStatus::Ok
}

/// Clears the current schema context.
///
/// Subsequent execute calls will use the default (no-schema) namespace.
///
/// # Safety
/// `db` must be a valid pointer returned by `grafeo_open*`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_reset_schema(db: *mut GrafeoDatabase) -> GrafeoStatus {
    if db.is_null() {
        set_last_error("Null database pointer");
        return GrafeoStatus::ErrorNullPointer;
    }
    // SAFETY: Caller guarantees valid pointer.
    let db = unsafe { &*db };
    if let Err(e) = db.inner.read().set_current_schema(None) {
        set_last_error(&e.to_string());
        return GrafeoStatus::ErrorQuery;
    }
    GrafeoStatus::Ok
}

/// Returns the current schema name, or NULL if no schema is set.
///
/// The returned string is valid until the next call to `grafeo_current_schema`,
/// `grafeo_set_schema`, or `grafeo_reset_schema` on this thread.
/// The caller must NOT free this pointer.
///
/// # Safety
/// `db` must be a valid pointer returned by `grafeo_open*`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_current_schema(db: *const GrafeoDatabase) -> *const c_char {
    if db.is_null() {
        return std::ptr::null();
    }
    // SAFETY: Caller guarantees valid pointer.
    let db = unsafe { &*db };
    match db.inner.read().current_schema() {
        Some(name) => LAST_SCHEMA.with(|cell| {
            *cell.borrow_mut() = CString::new(name).ok();
            cell.borrow()
                .as_ref()
                .map_or(std::ptr::null(), |s| s.as_ptr())
        }),
        None => {
            LAST_SCHEMA.with(|cell| *cell.borrow_mut() = None);
            std::ptr::null()
        }
    }
}

// =========================================================================
// Node CRUD
// =========================================================================

/// Create a node with labels (JSON array) and optional properties (JSON object).
/// Returns the new node ID, or `u64::MAX` on error.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_create_node(
    db: *mut GrafeoDatabase,
    labels_json: *const c_char,
    properties_json: *const c_char,
) -> u64 {
    if db.is_null() {
        set_last_error("Null database pointer");
        return u64::MAX;
    }
    // SAFETY: Caller guarantees valid pointer.
    let db = unsafe { &*db };
    let labels = crate::types::parse_labels(labels_json).unwrap_or_default();
    let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();

    let guard = db.inner.read();
    created_id(
        if let Some(props) = crate::types::parse_properties(properties_json) {
            guard.create_node_with_props(&label_refs, props)
        } else {
            guard.create_node(&label_refs)
        }
        .map(|id| id.as_u64()),
    )
}

/// Get a node by ID. Writes into `out`. Returns `Ok` or an error status.
/// On success, `out` must be freed with `grafeo_free_node`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_get_node(
    db: *mut GrafeoDatabase,
    id: u64,
    out: *mut *mut GrafeoNode,
) -> GrafeoStatus {
    let db = db_ref!(db);
    if out.is_null() {
        set_last_error("Null output pointer");
        return GrafeoStatus::ErrorNullPointer;
    }
    let guard = db.inner.read();
    match guard.get_node(NodeId::new(id)) {
        Some(node) => {
            let labels: Vec<serde_json::Value> = node
                .labels
                .iter()
                .map(|l| serde_json::Value::String(l.to_string()))
                .collect();
            let labels_json = CString::new(serde_json::to_string(&labels).unwrap_or_default())
                .unwrap_or_default();
            let properties_json = crate::types::properties_to_json(&node.properties);

            let gnode = Box::new(GrafeoNode {
                id: node.id.as_u64(),
                labels_json,
                properties_json,
            });
            // SAFETY: We checked out is not null above.
            unsafe { *out = Box::into_raw(gnode) };
            GrafeoStatus::Ok
        }
        None => {
            set_last_error(&format!("Node not found: {id}"));
            // SAFETY: We checked out is not null above.
            unsafe { *out = std::ptr::null_mut() };
            GrafeoStatus::ErrorDatabase
        }
    }
}

/// Delete a node by ID. Returns 1 if deleted, 0 if not found.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_delete_node(db: *mut GrafeoDatabase, id: u64) -> i32 {
    if db.is_null() {
        set_last_error("Null database pointer");
        return -1;
    }
    // SAFETY: Caller guarantees valid pointer.
    let db = unsafe { &*db };
    flag(db.inner.read().delete_node(NodeId::new(id)))
}

/// Set a property on a node. `value_json` is a JSON-encoded value.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_set_node_property(
    db: *mut GrafeoDatabase,
    id: u64,
    key: *const c_char,
    value_json: *const c_char,
) -> GrafeoStatus {
    let db = db_ref!(db);
    let key_str = match str_from_ptr(key) {
        Ok(s) => s,
        Err(e) => return e,
    };
    let Some(value) = crate::types::parse_value(value_json) else {
        set_last_error("Invalid JSON value");
        return GrafeoStatus::ErrorSerialization;
    };
    match db
        .inner
        .read()
        .set_node_property(NodeId::new(id), key_str, value)
    {
        Ok(()) => GrafeoStatus::Ok,
        Err(e) => set_error(&e),
    }
}

/// Remove a property from a node. Returns 1 if removed, 0 if not found.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_remove_node_property(
    db: *mut GrafeoDatabase,
    id: u64,
    key: *const c_char,
) -> i32 {
    if db.is_null() {
        set_last_error("Null database pointer");
        return -1;
    }
    // SAFETY: Caller guarantees valid pointer.
    let db = unsafe { &*db };
    let Ok(key_str) = str_from_ptr(key) else {
        return -1;
    };
    flag(
        db.inner
            .read()
            .remove_node_property(NodeId::new(id), key_str),
    )
}

/// Add a label to a node. Returns 1 if added, 0 if already present.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_add_node_label(
    db: *mut GrafeoDatabase,
    id: u64,
    label: *const c_char,
) -> i32 {
    if db.is_null() {
        set_last_error("Null database pointer");
        return -1;
    }
    // SAFETY: Caller guarantees valid pointer.
    let db = unsafe { &*db };
    let Ok(label_str) = str_from_ptr(label) else {
        return -1;
    };
    flag(db.inner.read().add_node_label(NodeId::new(id), label_str))
}

/// Remove a label from a node. Returns 1 if removed, 0 if not present.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_remove_node_label(
    db: *mut GrafeoDatabase,
    id: u64,
    label: *const c_char,
) -> i32 {
    if db.is_null() {
        set_last_error("Null database pointer");
        return -1;
    }
    // SAFETY: Caller guarantees valid pointer.
    let db = unsafe { &*db };
    let Ok(label_str) = str_from_ptr(label) else {
        return -1;
    };
    flag(
        db.inner
            .read()
            .remove_node_label(NodeId::new(id), label_str),
    )
}

/// Get labels for a node as a JSON array string.
/// Returns null if node not found. Caller must free with `grafeo_free_string`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_get_node_labels(db: *mut GrafeoDatabase, id: u64) -> *mut c_char {
    if db.is_null() {
        set_last_error("Null database pointer");
        return std::ptr::null_mut();
    }
    // SAFETY: Caller guarantees valid pointer.
    let db = unsafe { &*db };
    match db.inner.read().get_node_labels(NodeId::new(id)) {
        Some(labels) => {
            let json = serde_json::to_string(&labels).unwrap_or_default();
            CString::new(json).map_or(std::ptr::null_mut(), CString::into_raw)
        }
        None => std::ptr::null_mut(),
    }
}

/// Free a `GrafeoNode` returned by `grafeo_get_node`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_free_node(node: *mut GrafeoNode) {
    if !node.is_null() {
        // SAFETY: We take ownership back.
        unsafe { drop(Box::from_raw(node)) };
    }
}

/// Access the node ID from a `GrafeoNode`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_node_id(node: *const GrafeoNode) -> u64 {
    if node.is_null() {
        return u64::MAX;
    }
    // SAFETY: Caller guarantees valid pointer.
    unsafe { &*node }.id
}

/// Access labels JSON from a `GrafeoNode`. Valid until `grafeo_free_node`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_node_labels_json(node: *const GrafeoNode) -> *const c_char {
    if node.is_null() {
        return std::ptr::null();
    }
    // SAFETY: Caller guarantees valid pointer.
    unsafe { &*node }.labels_json.as_ptr()
}

/// Access properties JSON from a `GrafeoNode`. Valid until `grafeo_free_node`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_node_properties_json(node: *const GrafeoNode) -> *const c_char {
    if node.is_null() {
        return std::ptr::null();
    }
    // SAFETY: Caller guarantees valid pointer.
    unsafe { &*node }.properties_json.as_ptr()
}

// =========================================================================
// Edge CRUD
// =========================================================================

/// Create an edge. `properties_json` may be null for no properties.
/// Returns the new edge ID, or `u64::MAX` on error.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_create_edge(
    db: *mut GrafeoDatabase,
    source_id: u64,
    target_id: u64,
    edge_type: *const c_char,
    properties_json: *const c_char,
) -> u64 {
    if db.is_null() {
        set_last_error("Null database pointer");
        return u64::MAX;
    }
    // SAFETY: Caller guarantees valid pointer.
    let db = unsafe { &*db };
    let Ok(type_str) = str_from_ptr(edge_type) else {
        return u64::MAX;
    };
    let src = NodeId::new(source_id);
    let dst = NodeId::new(target_id);
    let guard = db.inner.read();

    created_id(
        if let Some(props) = crate::types::parse_properties(properties_json) {
            guard.create_edge_with_props(src, dst, type_str, props)
        } else {
            guard.create_edge(src, dst, type_str)
        }
        .map(|id| id.as_u64()),
    )
}

/// Get an edge by ID. Writes into `out`. Returns `Ok` or error status.
/// On success, `out` must be freed with `grafeo_free_edge`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_get_edge(
    db: *mut GrafeoDatabase,
    id: u64,
    out: *mut *mut GrafeoEdge,
) -> GrafeoStatus {
    let db = db_ref!(db);
    if out.is_null() {
        set_last_error("Null output pointer");
        return GrafeoStatus::ErrorNullPointer;
    }
    let guard = db.inner.read();
    match guard.get_edge(EdgeId(id)) {
        Some(edge) => {
            let edge_type = CString::new(edge.edge_type.to_string()).unwrap_or_default();
            let properties_json = crate::types::properties_to_json(&edge.properties);

            let gedge = Box::new(GrafeoEdge {
                id: edge.id.as_u64(),
                source_id: edge.src.as_u64(),
                target_id: edge.dst.as_u64(),
                edge_type,
                properties_json,
            });
            // SAFETY: We checked out is not null above.
            unsafe { *out = Box::into_raw(gedge) };
            GrafeoStatus::Ok
        }
        None => {
            set_last_error(&format!("Edge not found: {id}"));
            // SAFETY: We checked out is not null above.
            unsafe { *out = std::ptr::null_mut() };
            GrafeoStatus::ErrorDatabase
        }
    }
}

/// Delete an edge by ID. Returns 1 if deleted, 0 if not found.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_delete_edge(db: *mut GrafeoDatabase, id: u64) -> i32 {
    if db.is_null() {
        set_last_error("Null database pointer");
        return -1;
    }
    // SAFETY: Caller guarantees valid pointer.
    let db = unsafe { &*db };
    flag(db.inner.read().delete_edge(EdgeId(id)))
}

/// Set a property on an edge.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_set_edge_property(
    db: *mut GrafeoDatabase,
    id: u64,
    key: *const c_char,
    value_json: *const c_char,
) -> GrafeoStatus {
    let db = db_ref!(db);
    let key_str = match str_from_ptr(key) {
        Ok(s) => s,
        Err(e) => return e,
    };
    let Some(value) = crate::types::parse_value(value_json) else {
        set_last_error("Invalid JSON value");
        return GrafeoStatus::ErrorSerialization;
    };
    match db
        .inner
        .read()
        .set_edge_property(EdgeId(id), key_str, value)
    {
        Ok(()) => GrafeoStatus::Ok,
        Err(e) => set_error(&e),
    }
}

/// Remove a property from an edge. Returns 1 if removed, 0 if not found.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_remove_edge_property(
    db: *mut GrafeoDatabase,
    id: u64,
    key: *const c_char,
) -> i32 {
    if db.is_null() {
        set_last_error("Null database pointer");
        return -1;
    }
    // SAFETY: Caller guarantees valid pointer.
    let db = unsafe { &*db };
    let Ok(key_str) = str_from_ptr(key) else {
        return -1;
    };
    flag(db.inner.read().remove_edge_property(EdgeId(id), key_str))
}

/// Free a `GrafeoEdge` returned by `grafeo_get_edge`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_free_edge(edge: *mut GrafeoEdge) {
    if !edge.is_null() {
        // SAFETY: We take ownership back.
        unsafe { drop(Box::from_raw(edge)) };
    }
}

/// Access the edge ID.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_edge_id(edge: *const GrafeoEdge) -> u64 {
    if edge.is_null() {
        return u64::MAX;
    }
    // SAFETY: Caller guarantees valid pointer.
    unsafe { &*edge }.id
}

/// Access source node ID.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_edge_source_id(edge: *const GrafeoEdge) -> u64 {
    if edge.is_null() {
        return u64::MAX;
    }
    // SAFETY: Caller guarantees valid pointer.
    unsafe { &*edge }.source_id
}

/// Access target node ID.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_edge_target_id(edge: *const GrafeoEdge) -> u64 {
    if edge.is_null() {
        return u64::MAX;
    }
    // SAFETY: Caller guarantees valid pointer.
    unsafe { &*edge }.target_id
}

/// Access edge type string. Valid until `grafeo_free_edge`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_edge_type(edge: *const GrafeoEdge) -> *const c_char {
    if edge.is_null() {
        return std::ptr::null();
    }
    // SAFETY: Caller guarantees valid pointer.
    unsafe { &*edge }.edge_type.as_ptr()
}

/// Access edge properties JSON. Valid until `grafeo_free_edge`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_edge_properties_json(edge: *const GrafeoEdge) -> *const c_char {
    if edge.is_null() {
        return std::ptr::null();
    }
    // SAFETY: Caller guarantees valid pointer.
    unsafe { &*edge }.properties_json.as_ptr()
}

// =========================================================================
// Property Indexes
// =========================================================================

/// Create a property index.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_create_property_index(
    db: *mut GrafeoDatabase,
    property: *const c_char,
) -> GrafeoStatus {
    let db = db_ref!(db);
    let prop_str = match str_from_ptr(property) {
        Ok(s) => s,
        Err(e) => return e,
    };
    match db.inner.read().create_property_index(prop_str) {
        Ok(()) => GrafeoStatus::Ok,
        Err(e) => set_error(&e),
    }
}

/// Drop a property index. Returns 1 if dropped, 0 if not found, -1 on error
/// (check `grafeo_last_error()`): a null pointer, a name that is not UTF-8, or
/// a database error such as a closed database.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_drop_property_index(
    db: *mut GrafeoDatabase,
    property: *const c_char,
) -> i32 {
    if db.is_null() {
        set_last_error("Null database pointer");
        return -1;
    }
    // SAFETY: Caller guarantees valid pointer.
    let db = unsafe { &*db };
    let Ok(prop_str) = str_from_ptr(property) else {
        return -1;
    };
    match db.inner.read().drop_property_index(prop_str) {
        Ok(dropped) => i32::from(dropped),
        Err(e) => {
            set_error(&e);
            -1
        }
    }
}

/// Check if a property index exists. Returns 1 if exists, 0 otherwise.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_has_property_index(
    db: *mut GrafeoDatabase,
    property: *const c_char,
) -> i32 {
    if db.is_null() {
        return 0;
    }
    // SAFETY: Caller guarantees valid pointer.
    let db = unsafe { &*db };
    let Ok(prop_str) = str_from_ptr(property) else {
        return 0;
    };
    i32::from(db.inner.read().has_property_index(prop_str))
}

/// Find nodes by property value. Writes node IDs into `out_ids` and count
/// into `out_count`. Caller must free `*out_ids` with `grafeo_free_node_ids`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_find_nodes_by_property(
    db: *mut GrafeoDatabase,
    property: *const c_char,
    value_json: *const c_char,
    out_ids: *mut *mut u64,
    out_count: *mut usize,
) -> GrafeoStatus {
    let db = db_ref!(db);
    if out_ids.is_null() || out_count.is_null() {
        set_last_error("Null output pointer");
        return GrafeoStatus::ErrorNullPointer;
    }
    // Defensively zero so error paths never leave outputs uninitialized.
    unsafe {
        *out_count = 0;
        *out_ids = std::ptr::null_mut();
    }
    let prop_str = match str_from_ptr(property) {
        Ok(s) => s,
        Err(e) => return e,
    };
    let Some(value) = crate::types::parse_value(value_json) else {
        set_last_error("Invalid JSON value");
        return GrafeoStatus::ErrorSerialization;
    };
    let ids = db.inner.read().find_nodes_by_property(prop_str, &value);
    let count = ids.len();
    let mut raw_ids: Vec<u64> = ids.iter().map(|id| id.as_u64()).collect();
    raw_ids.shrink_to_fit();
    let ptr = raw_ids.as_mut_ptr();
    std::mem::forget(raw_ids);

    // SAFETY: We checked these are non-null above.
    unsafe {
        *out_ids = ptr;
        *out_count = count;
    }
    GrafeoStatus::Ok
}

/// Free a node ID array from `grafeo_find_nodes_by_property`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_free_node_ids(ids: *mut u64, count: usize) {
    if !ids.is_null() && count > 0 {
        // SAFETY: Reconstructs the Vec that was forgotten in find_nodes_by_property.
        unsafe {
            let slice = std::ptr::slice_from_raw_parts_mut(ids, count);
            drop(Box::from_raw(slice));
        }
    }
}

// =========================================================================
// Vector Operations
// =========================================================================

/// Create a vector similarity index on a node property.
/// `dimensions`, `m`, and `ef_construction` use -1 for default.
/// `metric` may be null for default (cosine).
#[cfg(feature = "vector-index")]
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_create_vector_index(
    db: *mut GrafeoDatabase,
    label: *const c_char,
    property: *const c_char,
    dimensions: i32,
    metric: *const c_char,
    m: i32,
    ef_construction: i32,
) -> GrafeoStatus {
    let db = db_ref!(db);
    let label_str = match str_from_ptr(label) {
        Ok(s) => s,
        Err(e) => return e,
    };
    let prop_str = match str_from_ptr(property) {
        Ok(s) => s,
        Err(e) => return e,
    };
    // reason: C FFI parameters are i32; the > 0 checks guarantee non-negative before widening to usize
    #[allow(clippy::cast_sign_loss)]
    let dims = if dimensions > 0 {
        Some(dimensions as usize)
    } else {
        None
    };
    let metric_str = if metric.is_null() {
        None
    } else {
        str_from_ptr(metric).ok()
    };
    // reason: value is non-negative by preceding guard check
    #[allow(clippy::cast_sign_loss)]
    let m_val = if m > 0 { Some(m as usize) } else { None };
    // reason: value is non-negative by preceding validation
    #[allow(clippy::cast_sign_loss)]
    let ef_val = if ef_construction > 0 {
        Some(ef_construction as usize)
    } else {
        None
    };

    match db
        .inner
        .read()
        .create_vector_index(label_str, prop_str, dims, metric_str, m_val, ef_val, None)
    {
        Ok(()) => GrafeoStatus::Ok,
        Err(e) => set_error(&e),
    }
}

/// Drop a vector index for the given label and property.
/// Returns 1 if removed, 0 if not found, -1 on error (check
/// `grafeo_last_error()`): a null pointer, a name that is not UTF-8, or a
/// database error such as a closed database.
#[cfg(feature = "vector-index")]
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_drop_vector_index(
    db: *mut GrafeoDatabase,
    label: *const c_char,
    property: *const c_char,
) -> i32 {
    if db.is_null() {
        set_last_error("Null database pointer");
        return -1;
    }
    // SAFETY: Caller guarantees valid pointer from grafeo_open*.
    let db = unsafe { &*db };
    let Ok(label_str) = str_from_ptr(label) else {
        return -1;
    };
    let Ok(prop_str) = str_from_ptr(property) else {
        return -1;
    };
    match db.inner.read().drop_vector_index(label_str, prop_str) {
        Ok(removed) => i32::from(removed),
        Err(e) => {
            set_error(&e);
            -1
        }
    }
}

/// Rebuild a vector index by rescanning all matching nodes.
/// Preserves original configuration.
#[cfg(feature = "vector-index")]
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_rebuild_vector_index(
    db: *mut GrafeoDatabase,
    label: *const c_char,
    property: *const c_char,
) -> GrafeoStatus {
    let db = db_ref!(db);
    let label_str = match str_from_ptr(label) {
        Ok(s) => s,
        Err(e) => return e,
    };
    let prop_str = match str_from_ptr(property) {
        Ok(s) => s,
        Err(e) => return e,
    };
    match db.inner.read().rebuild_vector_index(label_str, prop_str) {
        Ok(()) => GrafeoStatus::Ok,
        Err(e) => set_error(&e),
    }
}

/// Search for k nearest neighbors of a query vector.
/// Results written to `out_ids` and `out_distances` arrays of length `*out_count`.
/// Caller must free with `grafeo_free_vector_results`.
/// `ef` uses -1 for default.
#[cfg(feature = "vector-index")]
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_vector_search(
    db: *mut GrafeoDatabase,
    label: *const c_char,
    property: *const c_char,
    query: *const f32,
    query_len: usize,
    k: usize,
    ef: i32,
    out_ids: *mut *mut u64,
    out_distances: *mut *mut f32,
    out_count: *mut usize,
) -> GrafeoStatus {
    let db = db_ref!(db);
    if query.is_null() || out_ids.is_null() || out_distances.is_null() || out_count.is_null() {
        set_last_error("Null pointer argument");
        return GrafeoStatus::ErrorNullPointer;
    }
    // Defensively zero so error paths never leave outputs uninitialized.
    unsafe {
        *out_count = 0;
        *out_ids = std::ptr::null_mut();
        *out_distances = std::ptr::null_mut();
    }
    let label_str = match str_from_ptr(label) {
        Ok(s) => s,
        Err(e) => return e,
    };
    let prop_str = match str_from_ptr(property) {
        Ok(s) => s,
        Err(e) => return e,
    };
    // SAFETY: Caller guarantees query points to query_len f32s.
    let query_slice = unsafe { std::slice::from_raw_parts(query, query_len) };
    // reason: C FFI: > 0 check guarantees non-negative before widening
    #[allow(clippy::cast_sign_loss)]
    let ef_val = if ef > 0 { Some(ef as usize) } else { None };

    match db
        .inner
        .read()
        .vector_search(label_str, prop_str, query_slice, k, ef_val, None)
    {
        Ok(results) => {
            let count = results.len();
            let mut ids: Vec<u64> = results.iter().map(|(id, _)| id.as_u64()).collect();
            let mut dists: Vec<f32> = results.iter().map(|(_, d)| *d).collect();
            ids.shrink_to_fit();
            dists.shrink_to_fit();
            let ids_ptr = ids.as_mut_ptr();
            let dists_ptr = dists.as_mut_ptr();
            std::mem::forget(ids);
            std::mem::forget(dists);

            // SAFETY: We checked these are non-null above.
            unsafe {
                *out_ids = ids_ptr;
                *out_distances = dists_ptr;
                *out_count = count;
            }
            GrafeoStatus::Ok
        }
        Err(e) => set_error(&e),
    }
}

/// Search for diverse nearest neighbors using Maximal Marginal Relevance (MMR).
/// Results written to `out_ids` and `out_distances` arrays of length `*out_count`.
/// Caller must free with `grafeo_free_vector_results`.
/// `fetch_k` and `ef` use -1 for defaults. `lambda` uses negative for default (0.5).
#[cfg(feature = "vector-index")]
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_mmr_search(
    db: *mut GrafeoDatabase,
    label: *const c_char,
    property: *const c_char,
    query: *const f32,
    query_len: usize,
    k: usize,
    fetch_k: i32,
    lambda: f32,
    ef: i32,
    out_ids: *mut *mut u64,
    out_distances: *mut *mut f32,
    out_count: *mut usize,
) -> GrafeoStatus {
    let db = db_ref!(db);
    if query.is_null() || out_ids.is_null() || out_distances.is_null() || out_count.is_null() {
        set_last_error("Null pointer argument");
        return GrafeoStatus::ErrorNullPointer;
    }
    // Defensively zero so error paths never leave outputs uninitialized.
    unsafe {
        *out_count = 0;
        *out_ids = std::ptr::null_mut();
        *out_distances = std::ptr::null_mut();
    }
    let label_str = match str_from_ptr(label) {
        Ok(s) => s,
        Err(e) => return e,
    };
    let prop_str = match str_from_ptr(property) {
        Ok(s) => s,
        Err(e) => return e,
    };
    // SAFETY: Caller guarantees query points to query_len f32s.
    let query_slice = unsafe { std::slice::from_raw_parts(query, query_len) };
    // reason: C FFI: > 0 check guarantees non-negative before widening
    #[allow(clippy::cast_sign_loss)]
    let fetch_k_val = if fetch_k > 0 {
        Some(fetch_k as usize)
    } else {
        None
    };
    let lambda_val = if lambda >= 0.0 { Some(lambda) } else { None };
    // reason: value is non-negative by preceding guard check
    #[allow(clippy::cast_sign_loss)]
    let ef_val = if ef > 0 { Some(ef as usize) } else { None };

    match db.inner.read().mmr_search(
        label_str,
        prop_str,
        query_slice,
        k,
        fetch_k_val,
        lambda_val,
        ef_val,
        None,
    ) {
        Ok(results) => {
            let count = results.len();
            let mut ids: Vec<u64> = results.iter().map(|(id, _)| id.as_u64()).collect();
            let mut dists: Vec<f32> = results.iter().map(|(_, d)| *d).collect();
            ids.shrink_to_fit();
            dists.shrink_to_fit();
            let ids_ptr = ids.as_mut_ptr();
            let dists_ptr = dists.as_mut_ptr();
            std::mem::forget(ids);
            std::mem::forget(dists);

            // SAFETY: We checked these are non-null above.
            unsafe {
                *out_ids = ids_ptr;
                *out_distances = dists_ptr;
                *out_count = count;
            }
            GrafeoStatus::Ok
        }
        Err(e) => set_error(&e),
    }
}

/// Bulk-insert nodes with vector properties. Returns node IDs in `out_ids`
/// and the number of created nodes in `out_count`.
/// `vectors` is a flat array of `vector_count * dimensions` f32 values.
/// Caller must free `*out_ids` with `grafeo_free_node_ids(*out_ids, *out_count)`.
#[cfg(feature = "vector-index")]
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_batch_create_nodes(
    db: *mut GrafeoDatabase,
    label: *const c_char,
    property: *const c_char,
    vectors: *const f32,
    vector_count: usize,
    dimensions: usize,
    out_ids: *mut *mut u64,
    out_count: *mut usize,
) -> GrafeoStatus {
    let db = db_ref!(db);
    if vectors.is_null() || out_ids.is_null() || out_count.is_null() {
        set_last_error("Null pointer argument");
        return GrafeoStatus::ErrorNullPointer;
    }
    // Defensively zero so error paths never leave outputs uninitialized.
    unsafe {
        *out_count = 0;
        *out_ids = std::ptr::null_mut();
    }
    let label_str = match str_from_ptr(label) {
        Ok(s) => s,
        Err(e) => return e,
    };
    let prop_str = match str_from_ptr(property) {
        Ok(s) => s,
        Err(e) => return e,
    };
    // SAFETY: Caller guarantees vectors points to vector_count * dimensions f32s.
    let flat = unsafe { std::slice::from_raw_parts(vectors, vector_count * dimensions) };
    let vecs: Vec<Vec<f32>> = flat.chunks(dimensions).map(|c| c.to_vec()).collect();

    let node_ids = match db
        .inner
        .read()
        .batch_create_nodes(label_str, prop_str, vecs)
    {
        Ok(ids) => ids,
        Err(e) => return set_error(&e),
    };
    let mut raw_ids: Vec<u64> = node_ids.iter().map(|id| id.as_u64()).collect();
    raw_ids.shrink_to_fit();
    let count = raw_ids.len();
    let ptr = raw_ids.as_mut_ptr();
    std::mem::forget(raw_ids);

    // SAFETY: We checked out_ids and out_count are not null.
    unsafe {
        *out_ids = ptr;
        *out_count = count;
    }
    GrafeoStatus::Ok
}

/// Free vector search result arrays.
#[cfg(feature = "vector-index")]
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_free_vector_results(ids: *mut u64, distances: *mut f32, count: usize) {
    if !ids.is_null() && count > 0 {
        // SAFETY: Reconstructs the Vec that was forgotten in vector_search.
        unsafe {
            let slice = std::ptr::slice_from_raw_parts_mut(ids, count);
            drop(Box::from_raw(slice));
        }
    }
    if !distances.is_null() && count > 0 {
        // SAFETY: Reconstructs the Vec that was forgotten in vector_search.
        unsafe {
            let slice = std::ptr::slice_from_raw_parts_mut(distances, count);
            drop(Box::from_raw(slice));
        }
    }
}

// =========================================================================
// Statistics
// =========================================================================

/// Get the number of nodes.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_node_count(db: *mut GrafeoDatabase) -> usize {
    if db.is_null() {
        return 0;
    }
    // SAFETY: Caller guarantees valid pointer.
    unsafe { &*db }.inner.read().node_count()
}

/// Get the number of edges.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_edge_count(db: *mut GrafeoDatabase) -> usize {
    if db.is_null() {
        return 0;
    }
    // SAFETY: Caller guarantees valid pointer.
    unsafe { &*db }.inner.read().edge_count()
}

// =========================================================================
// Transactions
// =========================================================================

/// Begin a transaction with default isolation (snapshot isolation).
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_begin_transaction(db: *mut GrafeoDatabase) -> *mut GrafeoTransaction {
    if db.is_null() {
        set_last_error("Null database pointer");
        return std::ptr::null_mut();
    }
    // SAFETY: Caller guarantees valid pointer.
    let db = unsafe { &*db };
    let mut session = db.inner.read().session();
    match session.begin_transaction() {
        Ok(()) => Box::into_raw(Box::new(GrafeoTransaction {
            session: parking_lot::Mutex::new(Some(session)),
            committed: false,
            rolled_back: false,
        })),
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// Begin a transaction with a specific isolation level.
/// Levels: 0 = ReadCommitted, 1 = SnapshotIsolation, 2 = Serializable.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_begin_transaction_with_isolation(
    db: *mut GrafeoDatabase,
    isolation: i32,
) -> *mut GrafeoTransaction {
    if db.is_null() {
        set_last_error("Null database pointer");
        return std::ptr::null_mut();
    }
    // SAFETY: Caller guarantees valid pointer.
    let db = unsafe { &*db };

    let level = match isolation {
        0 => grafeo_engine::transaction::IsolationLevel::ReadCommitted,
        1 => grafeo_engine::transaction::IsolationLevel::SnapshotIsolation,
        2 => grafeo_engine::transaction::IsolationLevel::Serializable,
        _ => {
            set_last_error("Invalid isolation level (expected 0, 1, or 2)");
            return std::ptr::null_mut();
        }
    };

    let mut session = db.inner.read().session();
    match session.begin_transaction_with_isolation(level) {
        Ok(()) => Box::into_raw(Box::new(GrafeoTransaction {
            session: parking_lot::Mutex::new(Some(session)),
            committed: false,
            rolled_back: false,
        })),
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// Execute a query within a transaction.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_transaction_execute(
    tx: *mut GrafeoTransaction,
    query: *const c_char,
) -> *mut GrafeoResult {
    if tx.is_null() {
        set_last_error("Null transaction pointer");
        return std::ptr::null_mut();
    }
    // SAFETY: Caller guarantees valid pointer.
    let tx = unsafe { &*tx };
    if tx.committed || tx.rolled_back {
        set_last_error("Transaction is no longer active");
        return std::ptr::null_mut();
    }
    let Ok(query_str) = str_from_ptr(query) else {
        return std::ptr::null_mut();
    };
    let guard = tx.session.lock();
    let Some(session) = guard.as_ref() else {
        set_last_error("Transaction is no longer active");
        return std::ptr::null_mut();
    };
    match session.execute(query_str) {
        Ok(r) => build_result(&r),
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// Execute a query with params within a transaction.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_transaction_execute_with_params(
    tx: *mut GrafeoTransaction,
    query: *const c_char,
    params_json: *const c_char,
) -> *mut GrafeoResult {
    if tx.is_null() {
        set_last_error("Null transaction pointer");
        return std::ptr::null_mut();
    }
    // SAFETY: Caller guarantees valid pointer.
    let tx = unsafe { &*tx };
    if tx.committed || tx.rolled_back {
        set_last_error("Transaction is no longer active");
        return std::ptr::null_mut();
    }
    let Ok(query_str) = str_from_ptr(query) else {
        return std::ptr::null_mut();
    };
    let guard = tx.session.lock();
    let Some(session) = guard.as_ref() else {
        set_last_error("Transaction is no longer active");
        return std::ptr::null_mut();
    };
    let result = if let Some(params) = crate::types::parse_params(params_json) {
        session.execute_with_params(query_str, params)
    } else {
        session.execute(query_str)
    };
    match result {
        Ok(r) => build_result(&r),
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// Execute a query in the given language within a transaction.
///
/// `language` is one of: `"gql"`, `"cypher"`, `"gremlin"`, `"graphql"`,
/// `"sparql"`, `"sql"`. `params_json` may be null (no parameters).
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_transaction_execute_language(
    tx: *mut GrafeoTransaction,
    language: *const c_char,
    query: *const c_char,
    params_json: *const c_char,
) -> *mut GrafeoResult {
    if tx.is_null() {
        set_last_error("Null transaction pointer");
        return std::ptr::null_mut();
    }
    // SAFETY: Caller guarantees valid pointer.
    let tx = unsafe { &*tx };
    if tx.committed || tx.rolled_back {
        set_last_error("Transaction is no longer active");
        return std::ptr::null_mut();
    }
    let Ok(lang_str) = str_from_ptr(language) else {
        return std::ptr::null_mut();
    };
    let Ok(query_str) = str_from_ptr(query) else {
        return std::ptr::null_mut();
    };
    let guard = tx.session.lock();
    let Some(session) = guard.as_ref() else {
        set_last_error("Transaction is no longer active");
        return std::ptr::null_mut();
    };
    let params = crate::types::parse_params(params_json);
    match session.execute_language(query_str, lang_str, params) {
        Ok(r) => build_result(&r),
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// Commit a transaction.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_commit(tx: *mut GrafeoTransaction) -> GrafeoStatus {
    if tx.is_null() {
        set_last_error("Null transaction pointer");
        return GrafeoStatus::ErrorNullPointer;
    }
    // SAFETY: Caller guarantees valid pointer.
    let tx = unsafe { &mut *tx };
    if tx.committed {
        set_last_error("Transaction already committed");
        return GrafeoStatus::ErrorTransaction;
    }
    if tx.rolled_back {
        set_last_error("Transaction already rolled back");
        return GrafeoStatus::ErrorTransaction;
    }
    let mut guard = tx.session.lock();
    if let Some(ref mut session) = *guard {
        match session.commit() {
            Ok(()) => {
                tx.committed = true;
                GrafeoStatus::Ok
            }
            Err(e) => set_error(&e),
        }
    } else {
        set_last_error("Transaction is no longer active");
        GrafeoStatus::ErrorTransaction
    }
}

/// Rollback a transaction.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_rollback(tx: *mut GrafeoTransaction) -> GrafeoStatus {
    if tx.is_null() {
        set_last_error("Null transaction pointer");
        return GrafeoStatus::ErrorNullPointer;
    }
    // SAFETY: Caller guarantees valid pointer.
    let tx = unsafe { &mut *tx };
    if tx.committed {
        set_last_error("Transaction already committed");
        return GrafeoStatus::ErrorTransaction;
    }
    if tx.rolled_back {
        set_last_error("Transaction already rolled back");
        return GrafeoStatus::ErrorTransaction;
    }
    let mut guard = tx.session.lock();
    if let Some(ref mut session) = *guard {
        match session.rollback() {
            Ok(()) => {
                tx.rolled_back = true;
                GrafeoStatus::Ok
            }
            Err(e) => set_error(&e),
        }
    } else {
        set_last_error("Transaction is no longer active");
        GrafeoStatus::ErrorTransaction
    }
}

/// Free a transaction handle.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_free_transaction(tx: *mut GrafeoTransaction) {
    if !tx.is_null() {
        // SAFETY: We take ownership back. Drop impl handles auto-rollback.
        unsafe { drop(Box::from_raw(tx)) };
    }
}

// =========================================================================
// Admin
// =========================================================================

/// Clear all cached query plans.
///
/// Forces re-parsing and re-optimization on next execution.
/// Called automatically after DDL operations, but can be invoked manually.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_clear_plan_cache(db: *mut GrafeoDatabase) -> GrafeoStatus {
    let db = db_ref!(db);
    db.inner.read().clear_plan_cache();
    GrafeoStatus::Ok
}

/// Get database info as a JSON string. Caller must free with `grafeo_free_string`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_info(db: *mut GrafeoDatabase) -> *mut c_char {
    if db.is_null() {
        set_last_error("Null database pointer");
        return std::ptr::null_mut();
    }
    // SAFETY: Caller guarantees valid pointer.
    let db = unsafe { &*db };
    let info = db.inner.read().info();
    let json = serde_json::json!({
        "node_count": info.node_count,
        "edge_count": info.edge_count,
        "is_persistent": info.is_persistent,
        "path": info.path.as_ref().map(|p| p.to_string_lossy().into_owned()),
        "wal_enabled": info.wal_enabled,
        "version": info.version,
        "features": info.features,
    });
    let s = serde_json::to_string(&json).unwrap_or_default();
    CString::new(s).map_or(std::ptr::null_mut(), CString::into_raw)
}

/// Save database to a path. Caller provides a path string.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_save(db: *mut GrafeoDatabase, path: *const c_char) -> GrafeoStatus {
    let db = db_ref!(db);
    let path_str = match str_from_ptr(path) {
        Ok(s) => s,
        Err(e) => return e,
    };
    match db.inner.read().save(path_str) {
        Ok(()) => GrafeoStatus::Ok,
        Err(e) => set_error(&e),
    }
}

/// Create a full backup of the database.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_backup_full(
    db: *mut GrafeoDatabase,
    backup_dir: *const c_char,
) -> GrafeoStatus {
    let db = db_ref!(db);
    let dir = match str_from_ptr(backup_dir) {
        Ok(s) => s,
        Err(e) => return e,
    };
    match db.inner.read().backup_full(std::path::Path::new(dir)) {
        Ok(_) => GrafeoStatus::Ok,
        Err(e) => set_error(&e),
    }
}

/// Create an incremental backup (WAL records since last backup).
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_backup_incremental(
    db: *mut GrafeoDatabase,
    backup_dir: *const c_char,
) -> GrafeoStatus {
    let db = db_ref!(db);
    let dir = match str_from_ptr(backup_dir) {
        Ok(s) => s,
        Err(e) => return e,
    };
    match db
        .inner
        .read()
        .backup_incremental(std::path::Path::new(dir))
    {
        Ok(_) => GrafeoStatus::Ok,
        Err(e) => set_error(&e),
    }
}

/// Restore a database to a specific epoch from a backup chain.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_restore_to_epoch(
    backup_dir: *const c_char,
    epoch: u64,
    output_path: *const c_char,
) -> GrafeoStatus {
    let dir = match str_from_ptr(backup_dir) {
        Ok(s) => s,
        Err(e) => return e,
    };
    let out = match str_from_ptr(output_path) {
        Ok(s) => s,
        Err(e) => return e,
    };
    match grafeo_engine::GrafeoDB::restore_to_epoch(
        std::path::Path::new(dir),
        grafeo_common::types::EpochId::new(epoch),
        std::path::Path::new(out),
    ) {
        Ok(()) => GrafeoStatus::Ok,
        Err(e) => set_error(&e),
    }
}

/// Trigger a WAL checkpoint.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_wal_checkpoint(db: *mut GrafeoDatabase) -> GrafeoStatus {
    let db = db_ref!(db);
    match db.inner.read().wal_checkpoint() {
        Ok(()) => GrafeoStatus::Ok,
        Err(e) => set_error(&e),
    }
}

// =========================================================================
// CompactStore
// =========================================================================

/// Converts the default graph to a columnar CompactStore for faster queries.
///
/// Builds a columnar store with CSR adjacency from all nodes and edges. The
/// database stays writable: later writes go to an overlay on top of the
/// columnar base, which is merged into the base under memory pressure or by
/// calling `grafeo_compact` again. Named graphs and indexes stay. Fails (an
/// error status) while a transaction is open, or if the conversion fails.
#[cfg(feature = "compact-store")]
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_compact(db: *mut GrafeoDatabase) -> GrafeoStatus {
    let db = db_ref!(db);
    let mut guard = db.inner.write();
    match guard.compact() {
        Ok(()) => GrafeoStatus::Ok,
        Err(e) => set_error(&e),
    }
}

// =========================================================================
// Graph Projections
// =========================================================================

/// Creates a named graph projection over the graph selected now (the default
/// graph when none is selected). Returns 1 if created, 0 if a projection with
/// that name already exists, -1 on error (check `grafeo_last_error()`), such as
/// a selected graph that no longer exists or an invalid argument.
///
/// `node_labels` and `edge_types` are arrays of null-terminated UTF-8 strings.
/// Pass null with a count of 0 to include all nodes/edges.
///
/// # Safety
/// `db` must be a valid pointer returned by `grafeo_open*`. `name` must be a
/// valid null-terminated UTF-8 string. `node_labels` (if non-null) must point
/// to `num_labels` valid null-terminated UTF-8 string pointers. Same for
/// `edge_types` / `num_types`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_create_projection(
    db: *mut GrafeoDatabase,
    name: *const c_char,
    node_labels: *const *const c_char,
    num_labels: usize,
    edge_types: *const *const c_char,
    num_types: usize,
) -> i32 {
    use grafeo_core::graph::ProjectionSpec;

    if db.is_null() || name.is_null() {
        set_last_error("Null pointer argument");
        return -1;
    }
    // SAFETY: Caller guarantees valid pointers.
    let db = unsafe { &*db };
    let Ok(name_str) = str_from_ptr(name) else {
        return -1;
    };

    let mut spec = ProjectionSpec::new();

    // Reject null pointer with non-zero count (caller error)
    if node_labels.is_null() && num_labels > 0 {
        set_last_error("Null node_labels with a non-zero count");
        return -1;
    }
    if edge_types.is_null() && num_types > 0 {
        set_last_error("Null edge_types with a non-zero count");
        return -1;
    }

    if !node_labels.is_null() && num_labels > 0 {
        let mut labels = Vec::with_capacity(num_labels);
        for i in 0..num_labels {
            // SAFETY: Caller guarantees node_labels[0..num_labels] are valid.
            let ptr = unsafe { *node_labels.add(i) };
            let Ok(s) = str_from_ptr(ptr) else {
                return -1;
            };
            labels.push(s.to_owned());
        }
        spec = spec.with_node_labels(labels);
    }

    if !edge_types.is_null() && num_types > 0 {
        let mut types = Vec::with_capacity(num_types);
        for i in 0..num_types {
            // SAFETY: Caller guarantees edge_types[0..num_types] are valid.
            let ptr = unsafe { *edge_types.add(i) };
            let Ok(s) = str_from_ptr(ptr) else {
                return -1;
            };
            types.push(s.to_owned());
        }
        spec = spec.with_edge_types(types);
    }

    match db.inner.read().create_projection(name_str, spec) {
        Ok(created) => i32::from(created),
        Err(e) => {
            set_error(&e);
            -1
        }
    }
}

/// Drops a named graph projection. Returns 1 if it existed, 0 if not, -1 on
/// an invalid argument (check `grafeo_last_error()`).
///
/// # Safety
/// `db` must be a valid pointer returned by `grafeo_open*`. `name` must be a
/// valid null-terminated UTF-8 string.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_drop_projection(db: *mut GrafeoDatabase, name: *const c_char) -> i32 {
    if db.is_null() || name.is_null() {
        set_last_error("Null pointer argument");
        return -1;
    }
    // SAFETY: Caller guarantees valid pointers.
    let db = unsafe { &*db };
    let Ok(name_str) = str_from_ptr(name) else {
        return -1;
    };
    i32::from(db.inner.read().drop_projection(name_str))
}

/// Returns the names of all graph projections as a JSON array string.
/// Caller must free with `grafeo_free_string`.
///
/// # Safety
/// `db` must be a valid pointer returned by `grafeo_open*`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_list_projections(db: *mut GrafeoDatabase) -> *mut c_char {
    if db.is_null() {
        set_last_error("Null database pointer");
        return std::ptr::null_mut();
    }
    // SAFETY: Caller guarantees valid pointer.
    let db = unsafe { &*db };
    let names = db.inner.read().list_projections();
    let json = serde_json::to_string(&names).unwrap_or_default();
    CString::new(json).map_or(std::ptr::null_mut(), CString::into_raw)
}

// =========================================================================
// Memory Management
// =========================================================================

/// Free a string returned by any `grafeo_*` function that documents
/// "caller must free with `grafeo_free_string`".
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_free_string(s: *mut c_char) {
    if !s.is_null() {
        // SAFETY: Reconstructs the CString that was turned into a raw pointer.
        unsafe { drop(CString::from_raw(s)) };
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    /// After `grafeo_close` the handle lives on until `grafeo_free_database`:
    /// the index calls on it fail, and a drop returns -1 (not 0, "nothing to
    /// drop"), which the Go, Dart and C# wrappers read as an error.
    #[test]
    fn index_calls_after_close_return_an_error() {
        let dir = std::env::temp_dir().join(format!("grafeo-c-closed-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = CString::new(dir.join("amsterdam.grafeo").to_str().unwrap()).unwrap();
        let db = grafeo_open(path.as_ptr());
        assert!(!db.is_null());
        assert_eq!(grafeo_close(db), GrafeoStatus::Ok);

        let last_error = || {
            // SAFETY: the pointer is valid until the next call on this thread.
            unsafe { std::ffi::CStr::from_ptr(crate::error::grafeo_last_error()) }
                .to_string_lossy()
                .into_owned()
        };
        let property = CString::new("name").unwrap();
        assert_eq!(
            grafeo_create_property_index(db, property.as_ptr()),
            GrafeoStatus::ErrorDatabase
        );
        assert!(last_error().contains("GRAFEO-T007"), "{}", last_error());
        assert_eq!(grafeo_drop_property_index(db, property.as_ptr()), -1);
        assert!(last_error().contains("GRAFEO-T007"), "{}", last_error());
        #[cfg(feature = "vector-index")]
        {
            let label = CString::new("Person").unwrap();
            assert_eq!(
                grafeo_drop_vector_index(db, label.as_ptr(), property.as_ptr()),
                -1
            );
            assert!(last_error().contains("GRAFEO-T007"), "{}", last_error());
        }
        grafeo_free_database(db);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A drop with a null pointer or a name that is not UTF-8 returns -1, as
    /// every other error does, and sets `grafeo_last_error`: 0 means only
    /// "there was nothing to drop".
    #[test]
    fn a_drop_with_a_bad_argument_returns_an_error() {
        let db = grafeo_open_memory();
        let name = CString::new("name").unwrap();
        let not_utf8 = CString::new(vec![b'a', 0xff]).unwrap();

        let mut failures = Vec::new();
        let mut check = |call: &str, drop: &dyn Fn() -> i32| {
            crate::error::grafeo_clear_error();
            let result = drop();
            if result != -1 {
                failures.push(format!("{call}: returned {result}"));
            } else if crate::error::grafeo_last_error().is_null() {
                failures.push(format!("{call}: no last error"));
            }
        };
        check("property: null database", &|| {
            grafeo_drop_property_index(std::ptr::null_mut(), name.as_ptr())
        });
        check("property: null name", &|| {
            grafeo_drop_property_index(db, std::ptr::null())
        });
        check("property: not UTF-8", &|| {
            grafeo_drop_property_index(db, not_utf8.as_ptr())
        });
        #[cfg(feature = "vector-index")]
        {
            let label = CString::new("Person").unwrap();
            for (call, label, property) in [
                ("vector: null label", std::ptr::null(), name.as_ptr()),
                ("vector: null property", label.as_ptr(), std::ptr::null()),
                ("vector: label not UTF-8", not_utf8.as_ptr(), name.as_ptr()),
                (
                    "vector: property not UTF-8",
                    label.as_ptr(),
                    not_utf8.as_ptr(),
                ),
            ] {
                check(call, &|| grafeo_drop_vector_index(db, label, property));
            }
            check("vector: null database", &|| {
                grafeo_drop_vector_index(std::ptr::null_mut(), label.as_ptr(), name.as_ptr())
            });
            assert_eq!(
                grafeo_drop_vector_index(db, label.as_ptr(), name.as_ptr()),
                0,
                "no such index: 0"
            );
        }
        assert!(failures.is_empty(), "{failures:#?}");
        grafeo_close(db);
        grafeo_free_database(db);
    }

    /// `grafeo_create_projection` returns 1 when created, 0 when the name is
    /// taken, -1 with a last error on an invalid argument; `grafeo_drop_projection`
    /// returns 1 when it dropped one, 0 when there was none, -1 on an invalid
    /// argument.
    #[test]
    fn create_projection_returns_a_status() {
        let db = grafeo_open_memory();
        let name = CString::new("people").unwrap();
        let label = CString::new("Person").unwrap();
        let labels = [label.as_ptr()];
        let create = |labels: *const *const c_char, count: usize| {
            grafeo_create_projection(db, name.as_ptr(), labels, count, std::ptr::null(), 0)
        };
        assert_eq!(create(labels.as_ptr(), 1), 1);
        assert_eq!(create(labels.as_ptr(), 1), 0);
        assert_eq!(create(std::ptr::null(), 2), -1);
        // SAFETY: the pointer is valid until the next call on this thread.
        let error = unsafe { std::ffi::CStr::from_ptr(crate::error::grafeo_last_error()) }
            .to_string_lossy()
            .into_owned();
        assert!(error.contains("node_labels"), "{error}");
        assert_eq!(grafeo_drop_projection(db, name.as_ptr()), 1);
        assert_eq!(grafeo_drop_projection(db, name.as_ptr()), 0);
        assert_eq!(grafeo_drop_projection(db, std::ptr::null()), -1);
        grafeo_close(db);
        grafeo_free_database(db);
    }

    #[test]
    fn test_open_memory_and_close() {
        let db = grafeo_open_memory();
        assert!(!db.is_null());
        assert_eq!(grafeo_node_count(db), 0);
        assert_eq!(grafeo_edge_count(db), 0);
        let status = grafeo_close(db);
        assert_eq!(status, GrafeoStatus::Ok);
        grafeo_free_database(db);
    }

    #[test]
    fn test_create_node_and_get() {
        let db = grafeo_open_memory();
        let labels = CString::new(r#"["Person"]"#).unwrap();
        let props = CString::new(r#"{"name":"Alix","age":30}"#).unwrap();
        let id = grafeo_create_node(db, labels.as_ptr(), props.as_ptr());
        assert_ne!(id, u64::MAX);
        assert_eq!(grafeo_node_count(db), 1);

        let mut node_ptr: *mut GrafeoNode = std::ptr::null_mut();
        let status = grafeo_get_node(db, id, &raw mut node_ptr);
        assert_eq!(status, GrafeoStatus::Ok);
        assert!(!node_ptr.is_null());

        // SAFETY: We just verified it's not null.
        let node = unsafe { &*node_ptr };
        assert_eq!(node.id, id);

        grafeo_free_node(node_ptr);
        grafeo_close(db);
        grafeo_free_database(db);
    }

    #[test]
    fn test_create_edge_and_get() {
        let db = grafeo_open_memory();
        let labels_a = CString::new(r#"["Person"]"#).unwrap();
        let labels_b = CString::new(r#"["Person"]"#).unwrap();
        let id_a = grafeo_create_node(db, labels_a.as_ptr(), std::ptr::null());
        let id_b = grafeo_create_node(db, labels_b.as_ptr(), std::ptr::null());

        let edge_type = CString::new("KNOWS").unwrap();
        let edge_id = grafeo_create_edge(db, id_a, id_b, edge_type.as_ptr(), std::ptr::null());
        assert_ne!(edge_id, u64::MAX);
        assert_eq!(grafeo_edge_count(db), 1);

        let mut edge_ptr: *mut GrafeoEdge = std::ptr::null_mut();
        let status = grafeo_get_edge(db, edge_id, &raw mut edge_ptr);
        assert_eq!(status, GrafeoStatus::Ok);
        assert!(!edge_ptr.is_null());

        // SAFETY: We just verified it's not null.
        let edge = unsafe { &*edge_ptr };
        assert_eq!(edge.source_id, id_a);
        assert_eq!(edge.target_id, id_b);

        grafeo_free_edge(edge_ptr);
        grafeo_close(db);
        grafeo_free_database(db);
    }

    #[test]
    fn test_execute_query() {
        let db = grafeo_open_memory();
        let create = CString::new("CREATE (:Person {name: 'Alix', age: 30})").unwrap();
        let result = grafeo_execute(db, create.as_ptr());
        // CREATE returns a result (possibly empty rows).
        if !result.is_null() {
            grafeo_free_result(result);
        }

        let query = CString::new("MATCH (p:Person) RETURN p.name, p.age").unwrap();
        let result = grafeo_execute(db, query.as_ptr());
        assert!(!result.is_null());
        assert_eq!(grafeo_result_row_count(result), 1);

        let json_ptr = grafeo_result_json(result);
        assert!(!json_ptr.is_null());
        // SAFETY: Valid C string from our API.
        let json_str = unsafe { std::ffi::CStr::from_ptr(json_ptr) }
            .to_str()
            .unwrap();
        assert!(json_str.contains("Alix"));

        grafeo_free_result(result);
        grafeo_close(db);
        grafeo_free_database(db);
    }

    #[test]
    fn test_null_pointer_safety() {
        // All functions should handle null gracefully.
        let result = grafeo_execute(std::ptr::null_mut(), std::ptr::null());
        assert!(result.is_null());
        let err = crate::error::grafeo_last_error();
        assert!(!err.is_null());

        assert_eq!(grafeo_node_count(std::ptr::null_mut()), 0);
        assert_eq!(grafeo_edge_count(std::ptr::null_mut()), 0);
        assert_eq!(grafeo_delete_node(std::ptr::null_mut(), 0), -1);
        assert_eq!(grafeo_delete_edge(std::ptr::null_mut(), 0), -1);
    }

    #[test]
    fn test_transaction_commit() {
        let db = grafeo_open_memory();
        let tx = grafeo_begin_transaction(db);
        assert!(!tx.is_null());

        let query = CString::new("CREATE (:Tx {val: 1})").unwrap();
        let result = grafeo_transaction_execute(tx, query.as_ptr());
        if !result.is_null() {
            grafeo_free_result(result);
        }

        let status = grafeo_commit(tx);
        assert_eq!(status, GrafeoStatus::Ok);
        grafeo_free_transaction(tx);

        // Verify committed data is visible.
        assert_eq!(grafeo_node_count(db), 1);

        grafeo_close(db);
        grafeo_free_database(db);
    }

    #[test]
    fn test_transaction_rollback() {
        let db = grafeo_open_memory();

        // Create a baseline node.
        let labels = CString::new(r#"["Base"]"#).unwrap();
        grafeo_create_node(db, labels.as_ptr(), std::ptr::null());
        assert_eq!(grafeo_node_count(db), 1);

        let tx = grafeo_begin_transaction(db);
        let query = CString::new("CREATE (:Rolled {val: 2})").unwrap();
        let result = grafeo_transaction_execute(tx, query.as_ptr());
        if !result.is_null() {
            grafeo_free_result(result);
        }

        let status = grafeo_rollback(tx);
        assert_eq!(status, GrafeoStatus::Ok);
        grafeo_free_transaction(tx);

        // Rolled-back node should not be visible.
        assert_eq!(grafeo_node_count(db), 1);

        grafeo_close(db);
        grafeo_free_database(db);
    }

    #[test]
    fn test_node_property_crud() {
        let db = grafeo_open_memory();
        let labels = CString::new(r#"["Person"]"#).unwrap();
        let id = grafeo_create_node(db, labels.as_ptr(), std::ptr::null());

        let key = CString::new("city").unwrap();
        let value = CString::new(r#""Berlin""#).unwrap();
        let status = grafeo_set_node_property(db, id, key.as_ptr(), value.as_ptr());
        assert_eq!(status, GrafeoStatus::Ok);

        let removed = grafeo_remove_node_property(db, id, key.as_ptr());
        assert_eq!(removed, 1);

        grafeo_close(db);
        grafeo_free_database(db);
    }

    #[test]
    fn test_version() {
        let v = grafeo_version();
        assert!(!v.is_null());
        // SAFETY: Static string, always valid.
        let version_str = unsafe { std::ffi::CStr::from_ptr(v) }.to_str().unwrap();
        assert!(!version_str.is_empty(), "version_str is empty");
    }

    // ── Edge property CRUD ──

    #[test]
    fn test_edge_property_crud() {
        let db = grafeo_open_memory();
        let labels = CString::new(r#"["N"]"#).unwrap();
        let id_a = grafeo_create_node(db, labels.as_ptr(), std::ptr::null());
        let id_b = grafeo_create_node(db, labels.as_ptr(), std::ptr::null());

        let edge_type = CString::new("R").unwrap();
        let eid = grafeo_create_edge(db, id_a, id_b, edge_type.as_ptr(), std::ptr::null());
        assert_ne!(eid, u64::MAX);

        // Set edge property
        let key = CString::new("weight").unwrap();
        let value = CString::new("1.5").unwrap();
        let status = grafeo_set_edge_property(db, eid, key.as_ptr(), value.as_ptr());
        assert_eq!(status, GrafeoStatus::Ok);

        // Remove edge property
        let removed = grafeo_remove_edge_property(db, eid, key.as_ptr());
        assert_eq!(removed, 1);

        // Removing again returns 0
        let removed2 = grafeo_remove_edge_property(db, eid, key.as_ptr());
        assert_eq!(removed2, 0);

        grafeo_close(db);
        grafeo_free_database(db);
    }

    // ── Delete operations ──

    #[test]
    fn test_delete_node_and_edge() {
        let db = grafeo_open_memory();
        let labels = CString::new(r#"["N"]"#).unwrap();
        let id_a = grafeo_create_node(db, labels.as_ptr(), std::ptr::null());
        let id_b = grafeo_create_node(db, labels.as_ptr(), std::ptr::null());

        let edge_type = CString::new("R").unwrap();
        let eid = grafeo_create_edge(db, id_a, id_b, edge_type.as_ptr(), std::ptr::null());

        assert_eq!(grafeo_node_count(db), 2);
        assert_eq!(grafeo_edge_count(db), 1);

        // Delete edge
        assert_eq!(grafeo_delete_edge(db, eid), 1);
        assert_eq!(grafeo_edge_count(db), 0);
        // Second delete returns 0
        assert_eq!(grafeo_delete_edge(db, eid), 0);

        // Delete node
        assert_eq!(grafeo_delete_node(db, id_a), 1);
        assert_eq!(grafeo_node_count(db), 1);
        assert_eq!(grafeo_delete_node(db, id_a), 0);

        grafeo_close(db);
        grafeo_free_database(db);
    }

    // ── Label operations ──

    #[test]
    fn test_label_operations() {
        let db = grafeo_open_memory();
        let labels = CString::new(r#"["Person"]"#).unwrap();
        let id = grafeo_create_node(db, labels.as_ptr(), std::ptr::null());

        // Add label
        let label = CString::new("Employee").unwrap();
        assert_eq!(grafeo_add_node_label(db, id, label.as_ptr()), 1);
        // Adding same label returns 0
        assert_eq!(grafeo_add_node_label(db, id, label.as_ptr()), 0);

        // Get labels
        let labels_json = grafeo_get_node_labels(db, id);
        assert!(!labels_json.is_null());
        // SAFETY: We just verified the pointer is not null.
        let labels_str = unsafe { std::ffi::CStr::from_ptr(labels_json) }
            .to_str()
            .unwrap();
        assert!(labels_str.contains("Person"));
        assert!(labels_str.contains("Employee"));
        grafeo_free_string(labels_json);

        // Remove label
        assert_eq!(grafeo_remove_node_label(db, id, label.as_ptr()), 1);
        assert_eq!(grafeo_remove_node_label(db, id, label.as_ptr()), 0);

        grafeo_close(db);
        grafeo_free_database(db);
    }

    // ── Execute with parameters ──

    #[test]
    fn test_execute_with_params() {
        let db = grafeo_open_memory();

        // Create a node first
        let create = CString::new("CREATE (:Person {name: 'Alix', age: 30})").unwrap();
        let result = grafeo_execute(db, create.as_ptr());
        if !result.is_null() {
            grafeo_free_result(result);
        }

        // Query with parameters
        let query = CString::new("MATCH (n:Person) WHERE n.name = $name RETURN n.age").unwrap();
        let params = CString::new(r#"{"name":"Alix"}"#).unwrap();
        let result = grafeo_execute_with_params(db, query.as_ptr(), params.as_ptr());
        assert!(!result.is_null());
        assert_eq!(grafeo_result_row_count(result), 1);
        grafeo_free_result(result);

        grafeo_close(db);
        grafeo_free_database(db);
    }

    // ── Property index operations ──

    #[test]
    fn test_property_index_lifecycle() {
        let db = grafeo_open_memory();
        let prop = CString::new("name").unwrap();

        // No index initially
        assert_eq!(grafeo_has_property_index(db, prop.as_ptr()), 0);

        // Create index
        let status = grafeo_create_property_index(db, prop.as_ptr());
        assert_eq!(status, GrafeoStatus::Ok);
        assert_eq!(grafeo_has_property_index(db, prop.as_ptr()), 1);

        // Create node with the property
        let labels = CString::new(r#"["Person"]"#).unwrap();
        let props = CString::new(r#"{"name":"Alix"}"#).unwrap();
        grafeo_create_node(db, labels.as_ptr(), props.as_ptr());

        // Find by property
        let value = CString::new(r#""Alix""#).unwrap();
        let mut out_ids: *mut u64 = std::ptr::null_mut();
        let mut out_count: usize = 0;
        let status = grafeo_find_nodes_by_property(
            db,
            prop.as_ptr(),
            value.as_ptr(),
            &raw mut out_ids,
            &raw mut out_count,
        );
        assert_eq!(status, GrafeoStatus::Ok);
        assert_eq!(out_count, 1);
        if !out_ids.is_null() {
            grafeo_free_node_ids(out_ids, out_count);
        }

        // Drop index
        assert_eq!(grafeo_drop_property_index(db, prop.as_ptr()), 1);
        assert_eq!(grafeo_has_property_index(db, prop.as_ptr()), 0);
        assert_eq!(grafeo_drop_property_index(db, prop.as_ptr()), 0);

        grafeo_close(db);
        grafeo_free_database(db);
    }

    // ── Database info ──

    #[test]
    fn test_database_info() {
        let db = grafeo_open_memory();
        let labels = CString::new(r#"["Person"]"#).unwrap();
        let props = CString::new(r#"{"name":"Alix"}"#).unwrap();
        grafeo_create_node(db, labels.as_ptr(), props.as_ptr());

        let info = grafeo_info(db);
        assert!(!info.is_null());
        // SAFETY: We just verified the pointer is not null.
        let info_str = unsafe { std::ffi::CStr::from_ptr(info) }.to_str().unwrap();
        assert!(info_str.contains("node_count"));
        grafeo_free_string(info);

        grafeo_close(db);
        grafeo_free_database(db);
    }

    // ── Result metadata ──

    #[test]
    fn test_result_metadata() {
        let db = grafeo_open_memory();
        let create = CString::new("CREATE (:N {x: 1}), (:N {x: 2}), (:N {x: 3})").unwrap();
        let result = grafeo_execute(db, create.as_ptr());
        if !result.is_null() {
            grafeo_free_result(result);
        }

        let query = CString::new("MATCH (n:N) RETURN n.x").unwrap();
        let result = grafeo_execute(db, query.as_ptr());
        assert!(!result.is_null());

        assert_eq!(grafeo_result_row_count(result), 3);
        let time = grafeo_result_execution_time_ms(result);
        assert!(time >= 0.0);

        grafeo_free_result(result);
        grafeo_close(db);
        grafeo_free_database(db);
    }

    // ── Edge accessor functions ──

    #[test]
    fn test_edge_accessors() {
        let db = grafeo_open_memory();
        let labels = CString::new(r#"["N"]"#).unwrap();
        let id_a = grafeo_create_node(db, labels.as_ptr(), std::ptr::null());
        let id_b = grafeo_create_node(db, labels.as_ptr(), std::ptr::null());

        let edge_type = CString::new("KNOWS").unwrap();
        let edge_props = CString::new(r#"{"since":2020}"#).unwrap();
        let eid = grafeo_create_edge(db, id_a, id_b, edge_type.as_ptr(), edge_props.as_ptr());

        let mut edge_ptr: *mut GrafeoEdge = std::ptr::null_mut();
        let status = grafeo_get_edge(db, eid, &raw mut edge_ptr);
        assert_eq!(status, GrafeoStatus::Ok);
        assert!(!edge_ptr.is_null());

        // Test all accessor functions
        assert_eq!(grafeo_edge_id(edge_ptr), eid);
        assert_eq!(grafeo_edge_source_id(edge_ptr), id_a);
        assert_eq!(grafeo_edge_target_id(edge_ptr), id_b);

        let type_ptr = grafeo_edge_type(edge_ptr);
        assert!(!type_ptr.is_null());
        // SAFETY: We just verified the pointer is not null.
        let type_str = unsafe { std::ffi::CStr::from_ptr(type_ptr) }
            .to_str()
            .unwrap();
        assert_eq!(type_str, "KNOWS");

        let props_ptr = grafeo_edge_properties_json(edge_ptr);
        assert!(!props_ptr.is_null());
        // SAFETY: We just verified the pointer is not null.
        let props_str = unsafe { std::ffi::CStr::from_ptr(props_ptr) }
            .to_str()
            .unwrap();
        assert!(props_str.contains("2020"));

        grafeo_free_edge(edge_ptr);
        grafeo_close(db);
        grafeo_free_database(db);
    }
}
