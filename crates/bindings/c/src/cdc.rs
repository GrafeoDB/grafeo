//! Change data capture history (#441), as JSON arrays of events in the shape
//! every binding returns (`grafeo_bindings_common::cdc::event_to_json`).

use std::ffi::{CString, c_char};

use grafeo_common::types::{EdgeId, EpochId, NodeId};
use grafeo_common::utils::error::Result;
use grafeo_engine::cdc::ChangeEvent;

use crate::error::{set_error, set_last_error};
use crate::types::GrafeoDatabase;

/// The events `read` returns as a JSON array (free with
/// `grafeo_free_string`), or NULL with the error set.
fn events(
    db: *mut GrafeoDatabase,
    read: impl FnOnce(&grafeo_engine::GrafeoDB) -> Result<Vec<ChangeEvent>>,
) -> *mut c_char {
    if db.is_null() {
        set_last_error("Null database pointer");
        return std::ptr::null_mut();
    }
    // SAFETY: Caller guarantees ptr from grafeo_open* and not yet freed.
    let db = unsafe { &*db };
    match read(&db.inner.read()) {
        Ok(events) => {
            let json = grafeo_bindings_common::cdc::events_to_json(&events);
            CString::new(json.to_string()).map_or(std::ptr::null_mut(), CString::into_raw)
        }
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// Every recorded change of node `id`, oldest first, as a JSON array of
/// events; NULL on error. Free with `grafeo_free_string`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_node_history(db: *mut GrafeoDatabase, id: u64) -> *mut c_char {
    events(db, |db| db.history(NodeId::new(id)))
}

/// Every recorded change of edge `id`, oldest first, as a JSON array of
/// events; NULL on error. Free with `grafeo_free_string`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_edge_history(db: *mut GrafeoDatabase, id: u64) -> *mut c_char {
    events(db, |db| db.history(EdgeId::new(id)))
}

/// The changes of node `id` at or after epoch `since`, as a JSON array of
/// events; NULL on error. Free with `grafeo_free_string`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_node_history_since(
    db: *mut GrafeoDatabase,
    id: u64,
    since: u64,
) -> *mut c_char {
    events(db, |db| {
        db.history_since(NodeId::new(id), EpochId::new(since))
    })
}

/// Every change of every entity in epochs `start` to `end` (both included),
/// in commit order, as a JSON array of events; NULL on error. Free with
/// `grafeo_free_string`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_changes_between(
    db: *mut GrafeoDatabase,
    start: u64,
    end: u64,
) -> *mut c_char {
    events(db, |db| {
        db.changes_between(EpochId::new(start), EpochId::new(end))
    })
}

#[cfg(test)]
mod tests {
    use std::ffi::{CStr, CString};

    use super::*;
    use crate::database::{
        grafeo_close, grafeo_execute, grafeo_free_database, grafeo_free_result, grafeo_free_string,
        grafeo_open_memory, grafeo_set_cdc_enabled,
    };

    fn events_of(json: *mut c_char) -> Vec<serde_json::Value> {
        assert!(!json.is_null());
        // SAFETY: a string the call handed out.
        let parsed =
            serde_json::from_str(unsafe { CStr::from_ptr(json) }.to_str().unwrap()).unwrap();
        grafeo_free_string(json);
        match parsed {
            serde_json::Value::Array(events) => events,
            other => panic!("not an array: {other}"),
        }
    }

    fn run(db: *mut GrafeoDatabase, query: &str) {
        let query = CString::new(query).unwrap();
        let result = grafeo_execute(db, query.as_ptr());
        assert!(!result.is_null());
        grafeo_free_result(result);
    }

    #[test]
    fn the_history_of_a_node_and_its_edges_in_the_shared_shape() {
        let db = grafeo_open_memory();
        grafeo_set_cdc_enabled(db, true);
        run(
            db,
            "INSERT (:Person {name: 'Butch'})-[:KNOWS]->(:Person {name: 'Hans'})",
        );
        run(db, "MATCH (p:Person {name: 'Butch'}) SET p.city = 'Berlin'");
        run(db, "MATCH (p:Person {name: 'Butch'}) DETACH DELETE p");

        // Butch is node 0, the edge is edge 0.
        let node = events_of(grafeo_node_history(db, 0));
        let kinds: Vec<&str> = node.iter().map(|e| e["kind"].as_str().unwrap()).collect();
        assert_eq!(kinds, ["create", "update", "delete"]);
        assert_eq!(node[0]["entity_type"], "node");
        assert_eq!(node[0]["labels"], serde_json::json!(["Person"]));
        assert_eq!(node[1]["after"], serde_json::json!({"city": "Berlin"}));

        let edge = events_of(grafeo_edge_history(db, 0));
        assert_eq!(edge.last().unwrap()["kind"], "delete");
        assert_eq!(
            edge.last().unwrap()["deleted_with"],
            0,
            "removed with Butch"
        );
        assert_eq!(edge[0]["edge_type"], "KNOWS");

        let since = events_of(grafeo_node_history_since(
            db,
            0,
            node[1]["epoch"].as_u64().unwrap(),
        ));
        assert_eq!(since.len(), 2);
        let all = events_of(grafeo_changes_between(db, 0, u64::MAX));
        assert!(all.len() >= 6, "{all:?}");
        let last: Vec<&str> = all[all.len() - 2..]
            .iter()
            .map(|e| e["entity_type"].as_str().unwrap())
            .collect();
        assert_eq!(
            last,
            ["edge", "node"],
            "the edge, then the node it was deleted with"
        );
        grafeo_close(db);
        grafeo_free_database(db);
    }

    #[test]
    fn a_null_database_returns_null() {
        assert!(grafeo_node_history(std::ptr::null_mut(), 0).is_null());
        assert!(grafeo_changes_between(std::ptr::null_mut(), 0, 1).is_null());
    }
}
