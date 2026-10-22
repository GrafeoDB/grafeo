//! Opaque handle types and value conversion helpers for the C FFI layer.

use std::ffi::CString;
use std::os::raw::c_char;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::error::GrafeoStatus;
use grafeo_common::types::{PropertyKey, PropertyMap, Value};
use grafeo_engine::database::GrafeoDB;

// ---------------------------------------------------------------------------
// Opaque handle types
// ---------------------------------------------------------------------------

/// Opaque database handle. Created by `grafeo_open*`, freed by `grafeo_free_database`.
pub struct GrafeoDatabase {
    pub(crate) inner: Arc<RwLock<GrafeoDB>>,
}

/// Opaque handle on a named graph of a database (`grafeo_graph`, freed by
/// `grafeo_free_graph`): queries and direct calls in that graph, without
/// switching the database's current graph. Each call looks the graph up, so
/// one dropped meanwhile fails the call.
pub struct GrafeoGraph {
    pub(crate) inner: Arc<RwLock<GrafeoDB>>,
    pub(crate) schema: Option<String>,
    pub(crate) name: String,
}

/// Opaque transaction handle. Created by `grafeo_begin_transaction*`, freed by `grafeo_free_transaction`.
pub struct GrafeoTransaction {
    pub(crate) session: parking_lot::Mutex<Option<grafeo_engine::session::Session>>,
    pub(crate) committed: bool,
    pub(crate) rolled_back: bool,
}

impl Drop for GrafeoTransaction {
    fn drop(&mut self) {
        // Auto-rollback if not explicitly committed or rolled back.
        if !self.committed && !self.rolled_back {
            let mut guard = self.session.lock();
            if let Some(ref mut session) = *guard {
                let _ = session.rollback();
            }
        }
    }
}

/// Query result. Holds JSON-serialized rows and metadata.
pub struct GrafeoResult {
    pub(crate) json: CString,
    pub(crate) row_count: usize,
    pub(crate) execution_time_ms: f64,
    pub(crate) rows_scanned: u64,
    /// JSON array of extracted node objects (deduplicated, metadata stripped).
    pub(crate) nodes_json: CString,
    /// JSON array of extracted edge objects (deduplicated, metadata stripped).
    pub(crate) edges_json: CString,
    /// What the statement wrote (see `grafeo_result_nodes_created` and the
    /// other counter getters).
    pub(crate) counters: grafeo_engine::database::WriteCounters,
}

/// Structured node returned by CRUD operations.
pub struct GrafeoNode {
    pub(crate) id: u64,
    pub(crate) labels_json: CString,
    pub(crate) properties_json: CString,
}

/// Structured edge returned by CRUD operations.
pub struct GrafeoEdge {
    pub(crate) id: u64,
    pub(crate) source_id: u64,
    pub(crate) target_id: u64,
    pub(crate) edge_type: CString,
    pub(crate) properties_json: CString,
}

// ---------------------------------------------------------------------------
// Value ↔ JSON conversion
// ---------------------------------------------------------------------------

/// Convert a Grafeo `Value` to a `serde_json::Value`.
pub fn value_to_json(v: &Value) -> serde_json::Value {
    grafeo_bindings_common::json::value_to_json(v)
}

/// Convert a `serde_json::Value` to a Grafeo `Value`; `what` names it in
/// the error (`property 'age'`), which is set as the last error.
pub fn json_to_value(v: &serde_json::Value, what: &str) -> Result<Value, GrafeoStatus> {
    grafeo_bindings_common::json::json_to_value(v)
        .map_err(|error| crate::batch::invalid(&format!("{what}: {error}")))
}

/// The JSON a C string argument holds; `None` for a null pointer or JSON
/// `null`. `what` names the argument in the error.
fn json_or_none(
    json_ptr: *const c_char,
    what: &str,
) -> Result<Option<serde_json::Value>, GrafeoStatus> {
    if json_ptr.is_null() {
        return Ok(None);
    }
    match crate::batch::json_arg(json_ptr, what)? {
        serde_json::Value::Null => Ok(None),
        parsed => Ok(Some(parsed)),
    }
}

/// Serialize a [`PropertyMap`] to a JSON `CString`.
pub fn properties_to_json(props: &PropertyMap) -> CString {
    let obj: serde_json::Map<std::string::String, serde_json::Value> = props
        .iter()
        .map(|(k, v)| (k.as_str().to_string(), value_to_json(v)))
        .collect();
    let json_str = serde_json::to_string(&serde_json::Value::Object(obj)).unwrap_or_default();
    CString::new(json_str).unwrap_or_default()
}

/// Parse a JSON C-string into a `Vec<(PropertyKey, Value)>` for node/edge
/// creation; `None` for a null pointer or JSON `null`.
///
/// # Errors
///
/// Invalid JSON, JSON that is not an object, or a property that has no value
/// (an integer past 38 digits); the error names it and is set as the last
/// error.
pub fn parse_properties(
    json_ptr: *const c_char,
) -> Result<Option<Vec<(PropertyKey, Value)>>, GrafeoStatus> {
    let Some(parsed) = json_or_none(json_ptr, "properties")? else {
        return Ok(None);
    };
    let Some(obj) = parsed.as_object() else {
        return Err(crate::batch::invalid(&format!(
            "properties: not an object: {parsed}"
        )));
    };
    obj.iter()
        .map(|(k, v)| {
            let value = json_to_value(v, &format!("property '{k}'"))?;
            Ok((PropertyKey::new(k.clone()), value))
        })
        .collect::<Result<_, _>>()
        .map(Some)
}

/// Parse a JSON C-string into a `Vec<String>` (for labels).
pub fn parse_labels(json_ptr: *const c_char) -> Option<Vec<String>> {
    if json_ptr.is_null() {
        return None;
    }
    // SAFETY: Caller guarantees valid null-terminated C string.
    let s = unsafe { std::ffi::CStr::from_ptr(json_ptr) }
        .to_str()
        .ok()?;
    let parsed: serde_json::Value = serde_json::from_str(s).ok()?;
    let arr = parsed.as_array()?;
    Some(
        arr.iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect(),
    )
}

/// Parse a JSON C-string into a single `Value`.
///
/// # Errors
///
/// A null pointer, invalid JSON, or a value that has none (an integer past
/// 38 digits); the error is set as the last error.
pub fn parse_value(json_ptr: *const c_char) -> Result<Value, GrafeoStatus> {
    if json_ptr.is_null() {
        return Err(crate::batch::invalid("value: null pointer"));
    }
    json_to_value(&crate::batch::json_arg(json_ptr, "value")?, "value")
}

/// Parse a JSON C-string into a `HashMap<String, Value>` for query params;
/// `None` for a null pointer or JSON `null`.
///
/// # Errors
///
/// Invalid JSON, JSON that is not an object, or a parameter that has no
/// value (an integer past 38 digits); the error names it and is set as the
/// last error.
pub fn parse_params(
    json_ptr: *const c_char,
) -> Result<Option<std::collections::HashMap<String, Value>>, GrafeoStatus> {
    let Some(parsed) = json_or_none(json_ptr, "params")? else {
        return Ok(None);
    };
    grafeo_bindings_common::json::json_params_to_map(Some(&parsed))
        .map_err(|error| crate::batch::invalid(&error))
}
