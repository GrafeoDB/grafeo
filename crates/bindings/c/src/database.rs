//! All `#[no_mangle] extern "C"` functions exposed by the Grafeo C API.

use std::ffi::CString;
use std::os::raw::c_char;
use std::sync::Arc;

use parking_lot::RwLock;

use grafeo_common::types::{EdgeId, NodeId};
use grafeo_engine::config::Config;
use grafeo_engine::database::GrafeoDB;

use crate::error::{GrafeoStatus, set_error, set_last_error, str_from_ptr};
use crate::types::{GrafeoDatabase, GrafeoEdge, GrafeoNode, GrafeoResult, GrafeoTransaction};

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
        .rows
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

    Box::into_raw(Box::new(GrafeoResult {
        json: c_json,
        row_count: result.rows.len(),
        execution_time_ms: result.execution_time_ms.unwrap_or(0.0),
        rows_scanned: result.rows_scanned.unwrap_or(0),
    }))
}

// =========================================================================
// Lifecycle
// =========================================================================

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
    match guard.execute(query_str) {
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
    let result = if let Some(params) = crate::types::parse_params(params_json) {
        guard.execute_with_params(query_str, params)
    } else {
        guard.execute(query_str)
    };
    match result {
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
    match db.inner.read().execute_cypher(query_str) {
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
    match db.inner.read().execute_gremlin(query_str) {
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
    match db.inner.read().execute_graphql(query_str) {
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
    match db.inner.read().execute_sparql(query_str) {
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

/// Free a query result.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_free_result(result: *mut GrafeoResult) {
    if !result.is_null() {
        // SAFETY: We take ownership back and drop it.
        unsafe { drop(Box::from_raw(result)) };
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
    let id = if let Some(props) = crate::types::parse_properties(properties_json) {
        guard.create_node_with_props(&label_refs, props)
    } else {
        guard.create_node(&label_refs)
    };
    id.as_u64()
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
    i32::from(db.inner.read().delete_node(NodeId::new(id)))
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
    db.inner
        .read()
        .set_node_property(NodeId::new(id), key_str, value);
    GrafeoStatus::Ok
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
    i32::from(
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
    i32::from(db.inner.read().add_node_label(NodeId::new(id), label_str))
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
    i32::from(
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

    let id = if let Some(props) = crate::types::parse_properties(properties_json) {
        guard.create_edge_with_props(src, dst, type_str, props)
    } else {
        guard.create_edge(src, dst, type_str)
    };
    id.as_u64()
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
    i32::from(db.inner.read().delete_edge(EdgeId(id)))
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
    db.inner
        .read()
        .set_edge_property(EdgeId(id), key_str, value);
    GrafeoStatus::Ok
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
    i32::from(db.inner.read().remove_edge_property(EdgeId(id), key_str))
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
    db.inner.read().create_property_index(prop_str);
    GrafeoStatus::Ok
}

/// Drop a property index. Returns 1 if dropped, 0 if not found.
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
    i32::from(db.inner.read().drop_property_index(prop_str))
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
        unsafe { drop(Vec::from_raw_parts(ids, count, count)) };
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
    let m_val = if m > 0 { Some(m as usize) } else { None };
    let ef_val = if ef_construction > 0 {
        Some(ef_construction as usize)
    } else {
        None
    };

    match db
        .inner
        .read()
        .create_vector_index(label_str, prop_str, dims, metric_str, m_val, ef_val)
    {
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
    let ef_val = if ef > 0 { Some(ef as usize) } else { None };

    match db
        .inner
        .read()
        .vector_search(label_str, prop_str, query_slice, k, ef_val)
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

/// Bulk-insert nodes with vector properties. Returns node IDs in `out_ids`.
/// `vectors` is a flat array of `vector_count * dimensions` f32 values.
/// Caller must free `*out_ids` with `grafeo_free_node_ids`.
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
) -> GrafeoStatus {
    let db = db_ref!(db);
    if vectors.is_null() || out_ids.is_null() {
        set_last_error("Null pointer argument");
        return GrafeoStatus::ErrorNullPointer;
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

    let node_ids = db
        .inner
        .read()
        .batch_create_nodes(label_str, prop_str, vecs);
    let mut raw_ids: Vec<u64> = node_ids.iter().map(|id| id.as_u64()).collect();
    raw_ids.shrink_to_fit();
    let ptr = raw_ids.as_mut_ptr();
    let _count = raw_ids.len();
    std::mem::forget(raw_ids);

    // SAFETY: We checked out_ids is not null.
    unsafe { *out_ids = ptr };
    GrafeoStatus::Ok
}

/// Free vector search result arrays.
#[cfg(feature = "vector-index")]
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_free_vector_results(ids: *mut u64, distances: *mut f32, count: usize) {
    if !ids.is_null() && count > 0 {
        // SAFETY: Reconstructs the Vec that was forgotten in vector_search.
        unsafe { drop(Vec::from_raw_parts(ids, count, count)) };
    }
    if !distances.is_null() && count > 0 {
        // SAFETY: Reconstructs the Vec that was forgotten in vector_search.
        unsafe { drop(Vec::from_raw_parts(distances, count, count)) };
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
pub extern "C" fn grafeo_begin_tx(db: *mut GrafeoDatabase) -> *mut GrafeoTransaction {
    if db.is_null() {
        set_last_error("Null database pointer");
        return std::ptr::null_mut();
    }
    // SAFETY: Caller guarantees valid pointer.
    let db = unsafe { &*db };
    let mut session = db.inner.read().session();
    match session.begin_tx() {
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
pub extern "C" fn grafeo_begin_tx_with_isolation(
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
    match session.begin_tx_with_isolation(level) {
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
pub extern "C" fn grafeo_tx_execute(
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
pub extern "C" fn grafeo_tx_execute_with_params(
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
        let props = CString::new(r#"{"name":"Alice","age":30}"#).unwrap();
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
        let create = CString::new("CREATE (:Person {name: 'Alice', age: 30})").unwrap();
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
        assert!(json_str.contains("Alice"));

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
        let tx = grafeo_begin_tx(db);
        assert!(!tx.is_null());

        let query = CString::new("CREATE (:Tx {val: 1})").unwrap();
        let result = grafeo_tx_execute(tx, query.as_ptr());
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

        let tx = grafeo_begin_tx(db);
        let query = CString::new("CREATE (:Rolled {val: 2})").unwrap();
        let result = grafeo_tx_execute(tx, query.as_ptr());
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
        assert!(!version_str.is_empty());
    }
}
