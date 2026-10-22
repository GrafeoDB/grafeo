//! Time travel: reads of the database as it was at an earlier epoch or time.
//!
//! A point in time is two arguments, a kind and a value, so no struct
//! crosses the ABI: `GRAFEO_POINT_EPOCH` with an epoch, `GRAFEO_POINT_TIME`
//! with milliseconds since 1970 (UTC), `GRAFEO_POINT_AGO` with milliseconds
//! before now. A database keeps the history these reads reach only with a
//! history retention (`grafeo_open_with_history`).

use std::ffi::{CString, c_char};
use std::time::Duration;

use grafeo_common::types::{EdgeId, EpochId, NodeId, Timestamp};
use grafeo_engine::{AsOfView, Config, GrafeoDB, HistoryRetention, TimePoint};
use parking_lot::RwLock;

use crate::database::{build_result, edge_out, node_out};
use crate::error::{GrafeoStatus, set_error, set_last_error, str_from_ptr};
use crate::types::{
    GrafeoDatabase, GrafeoEdge, GrafeoNode, GrafeoResult, parse_params, value_to_json,
};

/// A point that is an epoch.
pub const GRAFEO_POINT_EPOCH: i32 = 0;
/// A point that is a time, in milliseconds since 1970 (UTC).
pub const GRAFEO_POINT_TIME: i32 = 1;
/// A point that is a duration before now, in milliseconds.
pub const GRAFEO_POINT_AGO: i32 = 2;

/// No history: only what open snapshots read is kept.
pub const GRAFEO_HISTORY_OFF: i32 = 0;
/// Every state from when history was turned on.
pub const GRAFEO_HISTORY_FOREVER: i32 = 1;
/// The states committed within a duration before now.
pub const GRAFEO_HISTORY_DURATION: i32 = 2;

/// Opaque read-only view of the database as of a past epoch. Created by
/// `grafeo_as_of`, freed by `grafeo_free_as_of`.
pub struct GrafeoAsOf {
    /// Keeps the database open while the view lives.
    _db: std::sync::Arc<RwLock<GrafeoDB>>,
    view: AsOfView,
}

/// The point `kind` and `value` name, or `None` (with the error set).
fn time_point(kind: i32, value: i64) -> Option<TimePoint> {
    let millis = || {
        u64::try_from(value)
            .ok()
            .map(Duration::from_millis)
            .ok_or_else(|| format!("a duration ago is not negative: {value}"))
    };
    let point = match kind {
        GRAFEO_POINT_EPOCH => u64::try_from(value)
            .map(|epoch| TimePoint::Epoch(EpochId::new(epoch)))
            .map_err(|_| format!("an epoch is not negative: {value}")),
        GRAFEO_POINT_TIME => Ok(TimePoint::Timestamp(Timestamp::from_millis(value))),
        GRAFEO_POINT_AGO => millis().map(TimePoint::Ago),
        _ => Err(format!(
            "point kind {kind} is none of GRAFEO_POINT_EPOCH, GRAFEO_POINT_TIME and \
             GRAFEO_POINT_AGO"
        )),
    };
    point.map_err(|message| set_last_error(&message)).ok()
}

/// The history retention `kind` (and `duration_ms` for a duration) names,
/// or `None` (with the error set).
fn history_retention(kind: i32, duration_ms: u64) -> Option<HistoryRetention> {
    match kind {
        GRAFEO_HISTORY_OFF => Some(HistoryRetention::Off),
        GRAFEO_HISTORY_FOREVER => Some(HistoryRetention::Forever),
        GRAFEO_HISTORY_DURATION => Some(HistoryRetention::Duration(Duration::from_millis(
            duration_ms,
        ))),
        _ => {
            set_last_error(&format!(
                "history kind {kind} is none of GRAFEO_HISTORY_OFF, GRAFEO_HISTORY_FOREVER and \
                 GRAFEO_HISTORY_DURATION"
            ));
            None
        }
    }
}

/// Opens (or creates) the database at `path`, or an in-memory one when
/// `path` is NULL, keeping its history as `history_kind` says
/// (`GRAFEO_HISTORY_OFF`, `GRAFEO_HISTORY_FOREVER`, or
/// `GRAFEO_HISTORY_DURATION` for `duration_ms` before now). A database file
/// stores the setting. NULL on error. Free with `grafeo_free_database`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_open_with_history(
    path: *const c_char,
    history_kind: i32,
    duration_ms: u64,
) -> *mut GrafeoDatabase {
    let Some(retention) = history_retention(history_kind, duration_ms) else {
        return std::ptr::null_mut();
    };
    let config = if path.is_null() {
        Config::in_memory()
    } else {
        let Ok(path) = str_from_ptr(path) else {
            return std::ptr::null_mut();
        };
        Config::persistent(path)
    };
    match GrafeoDB::with_config(config.with_history_retention(retention)) {
        Ok(db) => Box::into_raw(Box::new(GrafeoDatabase {
            inner: std::sync::Arc::new(RwLock::new(db)),
        })),
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// The epoch of the last commit (0 for a NULL database).
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_current_epoch(db: *mut GrafeoDatabase) -> u64 {
    if db.is_null() {
        set_last_error("Null database pointer");
        return 0;
    }
    // SAFETY: Caller guarantees ptr from grafeo_open* and not yet freed.
    let db = unsafe { &*db };
    db.inner.read().current_epoch().as_u64()
}

/// Writes to `epoch_out` the epoch a read as of the point (`point_kind`,
/// `point_value`) reads at: the epoch itself, or the last one committed at
/// or before a time.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_epoch_as_of(
    db: *mut GrafeoDatabase,
    point_kind: i32,
    point_value: i64,
    epoch_out: *mut u64,
) -> GrafeoStatus {
    if db.is_null() || epoch_out.is_null() {
        set_last_error("Null pointer");
        return GrafeoStatus::ErrorNullPointer;
    }
    let Some(point) = time_point(point_kind, point_value) else {
        return GrafeoStatus::ErrorDatabase;
    };
    // SAFETY: Caller guarantees ptr from grafeo_open* and not yet freed.
    let db = unsafe { &*db };
    match db.inner.read().epoch_as_of(point) {
        Ok(epoch) => {
            // SAFETY: Caller guarantees `epoch_out` points to a u64.
            unsafe { *epoch_out = epoch.as_u64() };
            GrafeoStatus::Ok
        }
        Err(e) => set_error(&e),
    }
}

/// Runs a GQL query with `params_json` (a JSON object, or NULL) as the
/// database was at the point (`point_kind`, `point_value`). NULL on error.
/// Free with `grafeo_free_result`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_execute_as_of(
    db: *mut GrafeoDatabase,
    query: *const c_char,
    params_json: *const c_char,
    point_kind: i32,
    point_value: i64,
) -> *mut GrafeoResult {
    if db.is_null() {
        set_last_error("Null database pointer");
        return std::ptr::null_mut();
    }
    let Ok(query) = str_from_ptr(query) else {
        return std::ptr::null_mut();
    };
    let Some(point) = time_point(point_kind, point_value) else {
        return std::ptr::null_mut();
    };
    // SAFETY: Caller guarantees ptr from grafeo_open* and not yet freed.
    let db = unsafe { &*db };
    let guard = db.inner.read();
    let result = guard.epoch_as_of(point).and_then(|epoch| {
        guard
            .session()
            .execute_at_epoch_with_params(query, epoch, parse_params(params_json))
    });
    match result {
        Ok(result) => build_result(&result),
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// A read-only view of the database as it was at the point (`point_kind`,
/// `point_value`), which keeps the history it reads until freed. NULL on
/// error (a point before the history the database keeps). Free with
/// `grafeo_free_as_of`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_as_of(
    db: *mut GrafeoDatabase,
    point_kind: i32,
    point_value: i64,
) -> *mut GrafeoAsOf {
    if db.is_null() {
        set_last_error("Null database pointer");
        return std::ptr::null_mut();
    }
    let Some(point) = time_point(point_kind, point_value) else {
        return std::ptr::null_mut();
    };
    // SAFETY: Caller guarantees ptr from grafeo_open* and not yet freed.
    let db = unsafe { &*db };
    let view = db.inner.read().as_of(point);
    match view {
        Ok(view) => Box::into_raw(Box::new(GrafeoAsOf {
            _db: std::sync::Arc::clone(&db.inner),
            view,
        })),
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// The epoch a view reads at (0 for a NULL view).
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_as_of_epoch(view: *const GrafeoAsOf) -> u64 {
    if view.is_null() {
        set_last_error("Null view pointer");
        return 0;
    }
    // SAFETY: Caller guarantees ptr from grafeo_as_of and not yet freed.
    unsafe { &*view }.view.epoch().as_u64()
}

/// Runs a GQL query with `params_json` (a JSON object, or NULL) at the
/// view's epoch; a query that writes is refused. NULL on error. Free with
/// `grafeo_free_result`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_as_of_execute(
    view: *const GrafeoAsOf,
    query: *const c_char,
    params_json: *const c_char,
) -> *mut GrafeoResult {
    if view.is_null() {
        set_last_error("Null view pointer");
        return std::ptr::null_mut();
    }
    let Ok(query) = str_from_ptr(query) else {
        return std::ptr::null_mut();
    };
    // SAFETY: Caller guarantees ptr from grafeo_as_of and not yet freed.
    let view = &unsafe { &*view }.view;
    let result = match parse_params(params_json) {
        Some(params) => view.execute_with_params(query, params),
        None => view.execute(query),
    };
    match result {
        Ok(result) => build_result(&result),
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// Writes to `out` the node `id` as it was at the view's epoch: a
/// `GrafeoNode` to free with `grafeo_free_node`, or NULL with `GRAFEO_OK`
/// when it did not exist then (as `grafeo_get_node`).
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_as_of_get_node(
    view: *const GrafeoAsOf,
    id: u64,
    out: *mut *mut GrafeoNode,
) -> GrafeoStatus {
    if view.is_null() || out.is_null() {
        set_last_error("Null pointer");
        return GrafeoStatus::ErrorNullPointer;
    }
    // SAFETY: Caller guarantees ptr from grafeo_as_of and not yet freed.
    let view = &unsafe { &*view }.view;
    match view.get_node(NodeId::new(id)) {
        Ok(node) => node_out(node, out),
        Err(err) => set_error(&err),
    }
}

/// Writes to `out` the edge `id` as it was at the view's epoch: a
/// `GrafeoEdge` to free with `grafeo_free_edge`, or NULL with `GRAFEO_OK`
/// when it did not exist then (as `grafeo_get_edge`).
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_as_of_get_edge(
    view: *const GrafeoAsOf,
    id: u64,
    out: *mut *mut GrafeoEdge,
) -> GrafeoStatus {
    if view.is_null() || out.is_null() {
        set_last_error("Null pointer");
        return GrafeoStatus::ErrorNullPointer;
    }
    // SAFETY: Caller guarantees ptr from grafeo_as_of and not yet freed.
    let view = &unsafe { &*view }.view;
    match view.get_edge(EdgeId(id)) {
        Ok(edge) => edge_out(edge, out),
        Err(err) => set_error(&err),
    }
}

/// Writes to `out_json` the value of node `id`'s property `key` as it was at
/// the view's epoch, as JSON (in the shape of `grafeo_result_json`'s values),
/// to free with `grafeo_free_string`; NULL when the node or the property did
/// not exist then. Returns `Ok`, or the error status of a read the store
/// cannot do (with `out_json` NULL).
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_as_of_get_node_property(
    view: *const GrafeoAsOf,
    id: u64,
    key: *const c_char,
    out_json: *mut *mut c_char,
) -> GrafeoStatus {
    if view.is_null() || out_json.is_null() {
        set_last_error("Null pointer");
        return GrafeoStatus::ErrorNullPointer;
    }
    // SAFETY: checked not null above.
    unsafe { *out_json = std::ptr::null_mut() };
    let key = match str_from_ptr(key) {
        Ok(key) => key,
        Err(status) => return status,
    };
    // SAFETY: Caller guarantees ptr from grafeo_as_of and not yet freed.
    let view = &unsafe { &*view }.view;
    match view.get_node_property(NodeId::new(id), key) {
        Ok(Some(value)) => {
            let raw = CString::new(value_to_json(&value).to_string())
                .map_or(std::ptr::null_mut(), CString::into_raw);
            // SAFETY: checked not null above.
            unsafe { *out_json = raw };
            GrafeoStatus::Ok
        }
        Ok(None) => GrafeoStatus::Ok,
        Err(err) => set_error(&err),
    }
}

/// Frees a view and the history it keeps. Safe to call on NULL.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_free_as_of(view: *mut GrafeoAsOf) {
    if !view.is_null() {
        // SAFETY: Caller guarantees ptr from grafeo_as_of, freed once.
        drop(unsafe { Box::from_raw(view) });
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::{CStr, CString};

    use super::*;
    use crate::database::{
        grafeo_execute, grafeo_free_database, grafeo_free_result, grafeo_result_json,
    };
    use crate::error::grafeo_last_error;

    /// The rows of `result` as JSON text; frees it.
    fn json(result: *mut GrafeoResult) -> String {
        assert!(!result.is_null(), "{}", last_error());
        // SAFETY: the result is valid until freed below.
        let text = unsafe { CStr::from_ptr(grafeo_result_json(result)) }
            .to_string_lossy()
            .into_owned();
        grafeo_free_result(result);
        text
    }

    fn last_error() -> String {
        let error = grafeo_last_error();
        if error.is_null() {
            return String::new();
        }
        // SAFETY: the message lives until the next call on this thread.
        unsafe { CStr::from_ptr(error) }
            .to_string_lossy()
            .into_owned()
    }

    fn run(db: *mut GrafeoDatabase, query: &str) {
        let query = CString::new(query).unwrap();
        json(grafeo_execute(db, query.as_ptr()));
    }

    #[test]
    fn a_view_reads_the_state_of_an_epoch() {
        let db = grafeo_open_with_history(std::ptr::null(), GRAFEO_HISTORY_FOREVER, 0);
        assert!(!db.is_null(), "{}", last_error());
        run(db, "INSERT (:Person {name: 'Alix'})");
        let then = grafeo_current_epoch(db);
        run(db, "INSERT (:Person {name: 'Gus'})");

        let count = CString::new("MATCH (p:Person) RETURN count(p) AS n").unwrap();
        let view = grafeo_as_of(db, GRAFEO_POINT_EPOCH, i64::try_from(then).unwrap());
        assert!(!view.is_null(), "{}", last_error());
        assert_eq!(grafeo_as_of_epoch(view), then);
        assert_eq!(
            json(grafeo_as_of_execute(view, count.as_ptr(), std::ptr::null())),
            r#"[{"n":1}]"#
        );
        let write = CString::new("INSERT (:Person {name: 'Mia'})").unwrap();
        assert!(grafeo_as_of_execute(view, write.as_ptr(), std::ptr::null()).is_null());
        grafeo_free_as_of(view);
        grafeo_free_as_of(std::ptr::null_mut());

        let params = CString::new(r#"{"name": "Gus"}"#).unwrap();
        let gus = CString::new("MATCH (p:Person {name: $name}) RETURN count(p) AS n").unwrap();
        assert_eq!(
            json(grafeo_execute_as_of(
                db,
                gus.as_ptr(),
                params.as_ptr(),
                GRAFEO_POINT_EPOCH,
                i64::try_from(then).unwrap()
            )),
            r#"[{"n":0}]"#
        );
        let mut epoch = 0;
        assert_eq!(
            grafeo_epoch_as_of(db, GRAFEO_POINT_AGO, 0, &raw mut epoch),
            GrafeoStatus::Ok
        );
        assert_eq!(epoch, grafeo_current_epoch(db));
        grafeo_free_database(db);
    }

    #[test]
    fn a_view_reads_nodes_edges_and_properties_as_they_were() {
        let db = grafeo_open_with_history(std::ptr::null(), GRAFEO_HISTORY_FOREVER, 0);
        run(
            db,
            "INSERT (:Person {name: 'Alix'})-[:KNOWS {since: 2020}]->(:Person {name: 'Gus'})",
        );
        let ids = json(grafeo_execute(
            db,
            CString::new(
                "MATCH (a:Person {name: 'Alix'})-[k:KNOWS]->() RETURN id(a) AS a, id(k) AS k",
            )
            .unwrap()
            .as_ptr(),
        ));
        let ids: serde_json::Value = serde_json::from_str(&ids).unwrap();
        let (alix, knows) = (ids[0]["a"].as_u64().unwrap(), ids[0]["k"].as_u64().unwrap());
        let then = i64::try_from(grafeo_current_epoch(db)).unwrap();
        run(db, "MATCH (p:Person {name: 'Alix'}) SET p.name = 'Alix B.'");
        run(db, "MATCH ()-[k:KNOWS]->() DELETE k");

        let view = grafeo_as_of(db, GRAFEO_POINT_EPOCH, then);

        let name = CString::new("name").unwrap();
        let mut value = std::ptr::null_mut();
        assert_eq!(
            grafeo_as_of_get_node_property(view, alix, name.as_ptr(), &raw mut value),
            GrafeoStatus::Ok
        );
        // SAFETY: a string the call just handed out, freed below.
        assert_eq!(
            unsafe { CStr::from_ptr(value) }.to_str().unwrap(),
            r#""Alix""#
        );
        crate::database::grafeo_free_string(value);
        let missing = CString::new("city").unwrap();
        let mut absent = std::ptr::null_mut();
        assert_eq!(
            grafeo_as_of_get_node_property(view, alix, missing.as_ptr(), &raw mut absent),
            GrafeoStatus::Ok
        );
        assert!(absent.is_null(), "no city then");

        let mut node = std::ptr::null_mut();
        assert_eq!(
            grafeo_as_of_get_node(view, alix, &raw mut node),
            GrafeoStatus::Ok
        );
        // SAFETY: the node the call just handed out, freed below.
        let properties = unsafe { CStr::from_ptr((*node).properties_json.as_ptr()) };
        assert!(
            properties.to_str().unwrap().contains("\"Alix\""),
            "{properties:?}"
        );
        crate::database::grafeo_free_node(node);
        assert_eq!(
            grafeo_as_of_get_node(view, alix + 1000, &raw mut node),
            GrafeoStatus::Ok,
            "an absent node is no failure"
        );
        assert!(node.is_null());

        // The edge was there at `then`, and is gone now.
        let mut edge = std::ptr::null_mut();
        assert_eq!(
            grafeo_as_of_get_edge(view, knows, &raw mut edge),
            GrafeoStatus::Ok
        );
        crate::database::grafeo_free_edge(edge);
        let now = grafeo_as_of(db, GRAFEO_POINT_AGO, 0);
        assert_eq!(
            grafeo_as_of_get_edge(now, knows, &raw mut edge),
            GrafeoStatus::Ok
        );
        assert!(edge.is_null(), "the edge is gone now");
        grafeo_free_as_of(now);
        grafeo_free_as_of(view);
        grafeo_free_database(db);
    }

    #[test]
    fn a_time_without_history_and_a_bad_point_are_errors() {
        let db = grafeo_open_with_history(std::ptr::null(), GRAFEO_HISTORY_OFF, 0);
        let mut epoch = 0;
        assert_ne!(
            grafeo_epoch_as_of(db, GRAFEO_POINT_TIME, 0, &raw mut epoch),
            GrafeoStatus::Ok
        );
        assert!(last_error().contains("needs history"), "{}", last_error());
        assert!(grafeo_as_of(db, 7, 0).is_null());
        assert!(last_error().contains("point kind 7"), "{}", last_error());
        assert!(grafeo_as_of(db, GRAFEO_POINT_EPOCH, -1).is_null());
        assert!(grafeo_as_of(db, GRAFEO_POINT_AGO, -1).is_null());
        assert!(grafeo_open_with_history(std::ptr::null(), 9, 0).is_null());
        grafeo_free_database(db);
    }
}
