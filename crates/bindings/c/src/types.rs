//! Opaque handle types and value conversion helpers for the C FFI layer.

use std::collections::BTreeMap;
use std::ffi::CString;
use std::os::raw::c_char;
use std::sync::Arc;

use parking_lot::RwLock;

use grafeo_common::types::{PropertyKey, Value};
use grafeo_engine::database::GrafeoDB;

// ---------------------------------------------------------------------------
// Opaque handle types
// ---------------------------------------------------------------------------

/// Opaque database handle. Created by `grafeo_open*`, freed by `grafeo_free_database`.
pub struct GrafeoDatabase {
    pub(crate) inner: Arc<RwLock<GrafeoDB>>,
}

/// Opaque transaction handle. Created by `grafeo_begin_tx*`, freed by `grafeo_free_transaction`.
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
    match v {
        Value::Null => serde_json::Value::Null,
        Value::Bool(b) => serde_json::Value::Bool(*b),
        Value::Int64(i) => serde_json::json!(*i),
        Value::Float64(f) => serde_json::json!(*f),
        Value::String(s) => serde_json::Value::String(s.to_string()),
        Value::Bytes(b) => {
            // Encode as array of integers for lossless roundtrip.
            let arr: Vec<serde_json::Value> =
                b.iter().map(|&byte| serde_json::json!(byte)).collect();
            serde_json::Value::Array(arr)
        }
        Value::Timestamp(ts) => serde_json::json!({ "$timestamp_us": ts.as_micros() }),
        Value::List(items) => serde_json::Value::Array(items.iter().map(value_to_json).collect()),
        Value::Map(map) => {
            let obj: serde_json::Map<std::string::String, serde_json::Value> = map
                .iter()
                .map(|(k, v)| (k.as_str().to_string(), value_to_json(v)))
                .collect();
            serde_json::Value::Object(obj)
        }
        Value::Vector(vec) => {
            serde_json::Value::Array(vec.iter().map(|&f| serde_json::json!(f)).collect())
        }
    }
}

/// Convert a `serde_json::Value` to a Grafeo `Value`.
pub fn json_to_value(v: &serde_json::Value) -> Value {
    match v {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int64(i)
            } else if let Some(f) = n.as_f64() {
                Value::Float64(f)
            } else {
                Value::Null
            }
        }
        serde_json::Value::String(s) => Value::String(s.as_str().into()),
        serde_json::Value::Array(arr) => {
            let items: Vec<Value> = arr.iter().map(json_to_value).collect();
            Value::List(items.into())
        }
        serde_json::Value::Object(obj) => {
            // Check for special $timestamp_us encoding.
            if let Some(ts) = obj.get("$timestamp_us").and_then(serde_json::Value::as_i64) {
                return Value::Timestamp(grafeo_common::types::Timestamp::from_micros(ts));
            }
            let mut map = BTreeMap::new();
            for (k, v) in obj {
                map.insert(PropertyKey::new(k.clone()), json_to_value(v));
            }
            Value::Map(Arc::new(map))
        }
    }
}

/// Serialize properties `BTreeMap` to a JSON `CString`.
pub fn properties_to_json(props: &BTreeMap<PropertyKey, Value>) -> CString {
    let obj: serde_json::Map<std::string::String, serde_json::Value> = props
        .iter()
        .map(|(k, v)| (k.as_str().to_string(), value_to_json(v)))
        .collect();
    let json_str = serde_json::to_string(&serde_json::Value::Object(obj)).unwrap_or_default();
    CString::new(json_str).unwrap_or_default()
}

/// Parse a JSON C-string into a `Vec<(PropertyKey, Value)>` for node/edge creation.
pub fn parse_properties(json_ptr: *const c_char) -> Option<Vec<(PropertyKey, Value)>> {
    if json_ptr.is_null() {
        return None;
    }
    // SAFETY: Caller guarantees valid null-terminated C string.
    let s = unsafe { std::ffi::CStr::from_ptr(json_ptr) }
        .to_str()
        .ok()?;
    let parsed: serde_json::Value = serde_json::from_str(s).ok()?;
    let obj = parsed.as_object()?;
    let props: Vec<(PropertyKey, Value)> = obj
        .iter()
        .map(|(k, v)| (PropertyKey::new(k.clone()), json_to_value(v)))
        .collect();
    Some(props)
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
pub fn parse_value(json_ptr: *const c_char) -> Option<Value> {
    if json_ptr.is_null() {
        return None;
    }
    // SAFETY: Caller guarantees valid null-terminated C string.
    let s = unsafe { std::ffi::CStr::from_ptr(json_ptr) }
        .to_str()
        .ok()?;
    let parsed: serde_json::Value = serde_json::from_str(s).ok()?;
    Some(json_to_value(&parsed))
}

/// Parse a JSON C-string into a `HashMap<String, Value>` for query params.
pub fn parse_params(json_ptr: *const c_char) -> Option<std::collections::HashMap<String, Value>> {
    if json_ptr.is_null() {
        return None;
    }
    // SAFETY: Caller guarantees valid null-terminated C string.
    let s = unsafe { std::ffi::CStr::from_ptr(json_ptr) }
        .to_str()
        .ok()?;
    let parsed: serde_json::Value = serde_json::from_str(s).ok()?;
    let obj = parsed.as_object()?;
    let map: std::collections::HashMap<String, Value> = obj
        .iter()
        .map(|(k, v)| (k.clone(), json_to_value(v)))
        .collect();
    Some(map)
}
