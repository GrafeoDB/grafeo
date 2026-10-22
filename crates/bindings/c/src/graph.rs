//! Graph handles (#441): queries and direct calls in one named graph without
//! switching the database's current graph, as `db.graph(name)` in Python and
//! Rust. Each call looks the graph up again, so one dropped since fails it.

use std::ffi::c_char;

use grafeo_common::types::{EdgeId, NodeId};

use crate::batch::{batch_edges, clear_ids, json_arg, labels, parse_node_rows, write_ids};
use crate::database::{build_result, created_id, edge_out, flag, node_out};
use crate::edges::{edges_args, write_edges};
use crate::error::{GrafeoStatus, set_error, set_last_error, str_from_ptr};
use crate::types::{GrafeoDatabase, GrafeoEdge, GrafeoGraph, GrafeoNode, GrafeoResult};
use crate::upsert::{clear_json, edge_options, node_options, rows, write_summary};

/// Runs `$body` with `$handle`, the graph `$graph` names; returns `$err`
/// (with the error set) for a null handle or a graph that no longer exists.
macro_rules! with_graph {
    ($graph:expr, $err:expr, |$handle:ident| $body:expr) => {{
        if $graph.is_null() {
            set_last_error("Null graph pointer");
            return $err;
        }
        // SAFETY: Caller guarantees ptr from grafeo_graph and not yet freed.
        let graph = unsafe { &*$graph };
        let db = graph.inner.read();
        let $handle = match db.graph_in(graph.schema.as_deref(), &graph.name) {
            Ok(handle) => handle,
            Err(e) => {
                set_error(&e);
                return $err;
            }
        };
        $body
    }};
}

/// A handle on graph `name` of the database's current schema, or NULL (with
/// the error set) when it does not exist. Free with `grafeo_free_graph`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_graph(db: *mut GrafeoDatabase, name: *const c_char) -> *mut GrafeoGraph {
    let db = db_ref_or_null!(db);
    let Ok(name) = str_from_ptr(name) else {
        return std::ptr::null_mut();
    };
    let schema = {
        let guard = db.inner.read();
        let schema = guard.current_schema();
        if let Err(e) = guard.graph_in(schema.as_deref(), name) {
            set_error(&e);
            return std::ptr::null_mut();
        }
        schema
    };
    Box::into_raw(Box::new(GrafeoGraph {
        inner: std::sync::Arc::clone(&db.inner),
        schema,
        name: name.to_string(),
    }))
}

/// Frees a graph handle. Safe to call on NULL.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_free_graph(graph: *mut GrafeoGraph) {
    if !graph.is_null() {
        // SAFETY: Caller guarantees ptr from grafeo_graph, freed once.
        drop(unsafe { Box::from_raw(graph) });
    }
}

/// Runs a GQL query in the graph; NULL on error. Free with
/// `grafeo_free_result`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_graph_execute(
    graph: *mut GrafeoGraph,
    query: *const c_char,
) -> *mut GrafeoResult {
    grafeo_graph_execute_language(graph, std::ptr::null(), query, std::ptr::null())
}

/// Runs a GQL query with `params_json` (a JSON object, or NULL) in the
/// graph; NULL on error.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_graph_execute_with_params(
    graph: *mut GrafeoGraph,
    query: *const c_char,
    params_json: *const c_char,
) -> *mut GrafeoResult {
    grafeo_graph_execute_language(graph, std::ptr::null(), query, params_json)
}

/// Runs a query in `language` (NULL for GQL; as `grafeo_execute_language`)
/// with `params_json` (or NULL) in the graph; NULL on error.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_graph_execute_language(
    graph: *mut GrafeoGraph,
    language: *const c_char,
    query: *const c_char,
    params_json: *const c_char,
) -> *mut GrafeoResult {
    let Ok(query) = str_from_ptr(query) else {
        return std::ptr::null_mut();
    };
    let language = if language.is_null() {
        "gql"
    } else {
        let Ok(language) = str_from_ptr(language) else {
            return std::ptr::null_mut();
        };
        language
    };
    let Ok(params) = crate::types::parse_params(params_json) else {
        return std::ptr::null_mut();
    };
    with_graph!(graph, std::ptr::null_mut(), |handle| {
        match handle.execute_language(query, language, params) {
            Ok(result) => build_result(&result),
            Err(e) => {
                set_error(&e);
                std::ptr::null_mut()
            }
        }
    })
}

/// Creates a node with `labels_json` (a JSON array) and `properties_json`
/// (a JSON object, or NULL) in the graph; its id, or `UINT64_MAX` on error.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_graph_create_node(
    graph: *mut GrafeoGraph,
    labels_json: *const c_char,
    properties_json: *const c_char,
) -> u64 {
    let labels = crate::types::parse_labels(labels_json).unwrap_or_default();
    let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
    let Ok(properties) = crate::types::parse_properties(properties_json) else {
        return u64::MAX;
    };
    with_graph!(graph, u64::MAX, |handle| {
        created_id(
            match properties {
                Some(properties) => handle.create_node_with_props(&label_refs, properties),
                None => handle.create_node(&label_refs),
            }
            .map(|id| id.as_u64()),
        )
    })
}

/// Creates an edge of `edge_type` with `properties_json` (or NULL) in the
/// graph; its id, or `UINT64_MAX` on error.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_graph_create_edge(
    graph: *mut GrafeoGraph,
    source_id: u64,
    target_id: u64,
    edge_type: *const c_char,
    properties_json: *const c_char,
) -> u64 {
    let Ok(edge_type) = str_from_ptr(edge_type) else {
        return u64::MAX;
    };
    let Ok(properties) = crate::types::parse_properties(properties_json) else {
        return u64::MAX;
    };
    let (src, dst) = (NodeId::new(source_id), NodeId::new(target_id));
    with_graph!(graph, u64::MAX, |handle| {
        created_id(
            match properties {
                Some(properties) => handle.create_edge_with_props(src, dst, edge_type, properties),
                None => handle.create_edge(src, dst, edge_type),
            }
            .map(|id| id.as_u64()),
        )
    })
}

/// Gets a node of the graph into `out` (free with `grafeo_free_node`; NULL
/// with `Ok` when it does not exist).
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_graph_get_node(
    graph: *mut GrafeoGraph,
    id: u64,
    out: *mut *mut GrafeoNode,
) -> GrafeoStatus {
    if out.is_null() {
        set_last_error("Null output pointer");
        return GrafeoStatus::ErrorNullPointer;
    }
    with_graph!(graph, GrafeoStatus::ErrorDatabase, |handle| {
        match handle.get_node(NodeId::new(id)) {
            Ok(node) => node_out(node, out),
            Err(e) => set_error(&e),
        }
    })
}

/// The edges of a node of the graph, as `grafeo_get_node_edges`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_graph_get_node_edges(
    graph: *mut GrafeoGraph,
    node_id: u64,
    direction: i32,
    edge_types_json: *const c_char,
    out_neighbors: *mut *mut u64,
    out_edges: *mut *mut u64,
    out_count: *mut usize,
) -> GrafeoStatus {
    let (direction, edge_types) = match edges_args(
        direction,
        edge_types_json,
        out_neighbors,
        out_edges,
        out_count,
    ) {
        Ok(args) => args,
        Err(status) => return status,
    };
    let edge_types: Vec<&str> = edge_types.iter().map(String::as_str).collect();
    with_graph!(graph, GrafeoStatus::ErrorDatabase, |handle| {
        match handle.get_node_edges(NodeId::new(node_id), direction, &edge_types) {
            Ok(edges) => write_edges(&edges, out_neighbors, out_edges, out_count),
            Err(e) => set_error(&e),
        }
    })
}

/// Gets an edge of the graph into `out` (free with `grafeo_free_edge`; NULL
/// with `Ok` when it does not exist).
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_graph_get_edge(
    graph: *mut GrafeoGraph,
    id: u64,
    out: *mut *mut GrafeoEdge,
) -> GrafeoStatus {
    if out.is_null() {
        set_last_error("Null output pointer");
        return GrafeoStatus::ErrorNullPointer;
    }
    with_graph!(graph, GrafeoStatus::ErrorDatabase, |handle| {
        match handle.get_edge(EdgeId::new(id)) {
            Ok(edge) => edge_out(edge, out),
            Err(e) => set_error(&e),
        }
    })
}

/// Sets a property of a node of the graph to `value_json`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_graph_set_node_property(
    graph: *mut GrafeoGraph,
    id: u64,
    key: *const c_char,
    value_json: *const c_char,
) -> GrafeoStatus {
    set_property(graph, key, value_json, |handle, key, value| {
        handle.set_node_property(NodeId::new(id), key, value)
    })
}

/// Sets a property of an edge of the graph to `value_json`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_graph_set_edge_property(
    graph: *mut GrafeoGraph,
    id: u64,
    key: *const c_char,
    value_json: *const c_char,
) -> GrafeoStatus {
    set_property(graph, key, value_json, |handle, key, value| {
        handle.set_edge_property(EdgeId::new(id), key, value)
    })
}

fn set_property(
    graph: *mut GrafeoGraph,
    key: *const c_char,
    value_json: *const c_char,
    set: impl FnOnce(
        &grafeo_engine::database::GraphHandle<'_>,
        &str,
        grafeo_common::types::Value,
    ) -> grafeo_common::utils::error::Result<()>,
) -> GrafeoStatus {
    let key = match str_from_ptr(key) {
        Ok(key) => key,
        Err(status) => return status,
    };
    let value = match crate::types::parse_value(value_json) {
        Ok(value) => value,
        Err(status) => return status,
    };
    with_graph!(graph, GrafeoStatus::ErrorDatabase, |handle| {
        match set(&handle, key, value) {
            Ok(()) => GrafeoStatus::Ok,
            Err(e) => set_error(&e),
        }
    })
}

/// A yes-or-no call on the graph with one string argument: 1 or 0, -1 on
/// error.
fn named_flag(
    graph: *mut GrafeoGraph,
    name: *const c_char,
    call: impl FnOnce(
        &grafeo_engine::database::GraphHandle<'_>,
        &str,
    ) -> grafeo_common::utils::error::Result<bool>,
) -> i32 {
    let Ok(name) = str_from_ptr(name) else {
        return -1;
    };
    with_graph!(graph, -1, |handle| flag(call(&handle, name)))
}

/// Removes a property of a node of the graph: 1 removed, 0 not there, -1 on
/// error.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_graph_remove_node_property(
    graph: *mut GrafeoGraph,
    id: u64,
    key: *const c_char,
) -> i32 {
    named_flag(graph, key, |handle, key| {
        handle.remove_node_property(NodeId::new(id), key)
    })
}

/// Removes a property of an edge of the graph: 1 removed, 0 not there, -1 on
/// error.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_graph_remove_edge_property(
    graph: *mut GrafeoGraph,
    id: u64,
    key: *const c_char,
) -> i32 {
    named_flag(graph, key, |handle, key| {
        handle.remove_edge_property(EdgeId::new(id), key)
    })
}

/// Adds a label to a node of the graph: 1 added, 0 already there, -1 on
/// error.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_graph_add_node_label(
    graph: *mut GrafeoGraph,
    id: u64,
    label: *const c_char,
) -> i32 {
    named_flag(graph, label, |handle, label| {
        handle.add_node_label(NodeId::new(id), label)
    })
}

/// Removes a label from a node of the graph: 1 removed, 0 not there, -1 on
/// error.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_graph_remove_node_label(
    graph: *mut GrafeoGraph,
    id: u64,
    label: *const c_char,
) -> i32 {
    named_flag(graph, label, |handle, label| {
        handle.remove_node_label(NodeId::new(id), label)
    })
}

/// Deletes a node of the graph (one with edges fails): 1 deleted, 0 not
/// found, -1 on error.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_graph_delete_node(graph: *mut GrafeoGraph, id: u64) -> i32 {
    with_graph!(graph, -1, |handle| flag(
        handle.delete_node(NodeId::new(id))
    ))
}

/// Deletes a node of the graph with all its edges, as `DETACH DELETE` does:
/// 1 deleted, 0 not found, -1 on error.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_graph_detach_delete_node(graph: *mut GrafeoGraph, id: u64) -> i32 {
    with_graph!(graph, -1, |handle| flag(
        handle.detach_delete_node(NodeId::new(id))
    ))
}

/// Deletes an edge of the graph: 1 deleted, 0 not found, -1 on error.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_graph_delete_edge(graph: *mut GrafeoGraph, id: u64) -> i32 {
    with_graph!(graph, -1, |handle| flag(
        handle.delete_edge(EdgeId::new(id))
    ))
}

/// `grafeo_batch_create_nodes_with_props` in the graph.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_graph_batch_create_nodes_with_props(
    graph: *mut GrafeoGraph,
    labels_json: *const c_char,
    rows_json: *const c_char,
    out_ids: *mut *mut u64,
    out_count: *mut usize,
) -> GrafeoStatus {
    if let Err(status) = clear_ids(out_ids, out_count) {
        return status;
    }
    let (labels, rows) = match parse_node_rows(labels_json, rows_json) {
        Ok(parsed) => parsed,
        Err(status) => return status,
    };
    let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
    with_graph!(graph, GrafeoStatus::ErrorDatabase, |handle| {
        match handle.batch_create_nodes_with_labels(&label_refs, rows) {
            Ok(ids) => write_ids(
                ids.iter().map(|id| id.as_u64()).collect(),
                out_ids,
                out_count,
            ),
            Err(e) => set_error(&e),
        }
    })
}

/// `grafeo_batch_create_edges` in the graph.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_graph_batch_create_edges(
    graph: *mut GrafeoGraph,
    edges_json: *const c_char,
    out_ids: *mut *mut u64,
    out_count: *mut usize,
) -> GrafeoStatus {
    if let Err(status) = clear_ids(out_ids, out_count) {
        return status;
    }
    let edges = match json_arg(edges_json, "edges_json").and_then(|json| batch_edges(&json)) {
        Ok(edges) => edges,
        Err(status) => return status,
    };
    with_graph!(graph, GrafeoStatus::ErrorDatabase, |handle| {
        match handle.batch_create_edges(edges) {
            Ok(ids) => write_ids(
                ids.iter().map(|id| id.as_u64()).collect(),
                out_ids,
                out_count,
            ),
            Err(e) => set_error(&e),
        }
    })
}

/// `grafeo_upsert_nodes` in the graph.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_graph_upsert_nodes(
    graph: *mut GrafeoGraph,
    labels_json: *const c_char,
    rows_json: *const c_char,
    options_json: *const c_char,
    out_json: *mut *mut c_char,
) -> GrafeoStatus {
    if let Err(status) = clear_json(out_json) {
        return status;
    }
    let parsed = json_arg(labels_json, "labels_json")
        .and_then(|json| labels(&json, "labels_json"))
        .and_then(|labels| Ok((labels, rows(rows_json)?, node_options(options_json)?)));
    let (labels, rows, (key, replace)) = match parsed {
        Ok(parsed) => parsed,
        Err(status) => return status,
    };
    let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
    with_graph!(graph, GrafeoStatus::ErrorDatabase, |handle| {
        match handle.upsert_nodes(&label_refs, &key, rows, replace) {
            Ok(summary) => write_summary(&summary, out_json),
            Err(e) => set_error(&e),
        }
    })
}

/// `grafeo_upsert_edges` in the graph.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_graph_upsert_edges(
    graph: *mut GrafeoGraph,
    edge_type: *const c_char,
    rows_json: *const c_char,
    options_json: *const c_char,
    out_json: *mut *mut c_char,
) -> GrafeoStatus {
    if let Err(status) = clear_json(out_json) {
        return status;
    }
    let edge_type = match str_from_ptr(edge_type) {
        Ok(edge_type) => edge_type,
        Err(status) => return status,
    };
    let (rows, options) =
        match rows(rows_json).and_then(|rows| Ok((rows, edge_options(options_json)?))) {
            Ok(parsed) => parsed,
            Err(status) => return status,
        };
    with_graph!(graph, GrafeoStatus::ErrorDatabase, |handle| {
        match handle.upsert_edges(edge_type, rows, &options) {
            Ok(summary) => write_summary(&summary, out_json),
            Err(e) => set_error(&e),
        }
    })
}

#[cfg(test)]
mod tests {
    use std::ffi::{CStr, CString};

    use super::*;
    use crate::database::{
        grafeo_close, grafeo_execute, grafeo_free_database, grafeo_free_node, grafeo_free_node_ids,
        grafeo_free_result, grafeo_free_string, grafeo_node_count, grafeo_node_labels_json,
        grafeo_node_properties_json, grafeo_open_memory, grafeo_result_json,
        grafeo_result_nodes_created,
    };
    use crate::error::grafeo_last_error;

    fn c(text: &str) -> CString {
        CString::new(text).unwrap()
    }

    fn last_error() -> String {
        // SAFETY: the error string lives until the next call on this thread.
        unsafe { CStr::from_ptr(grafeo_last_error()) }
            .to_string_lossy()
            .into_owned()
    }

    fn result_rows(result: *mut GrafeoResult) -> serde_json::Value {
        assert!(!result.is_null(), "{}", last_error());
        // SAFETY: valid until the result is freed.
        let json = unsafe { CStr::from_ptr(grafeo_result_json(result)) }
            .to_str()
            .unwrap()
            .to_string();
        grafeo_free_result(result);
        serde_json::from_str(&json).unwrap()
    }

    fn take_json(out: *mut c_char) -> serde_json::Value {
        assert!(!out.is_null(), "{}", last_error());
        // SAFETY: a string the call handed out.
        let json = serde_json::from_str(unsafe { CStr::from_ptr(out) }.to_str().unwrap()).unwrap();
        grafeo_free_string(out);
        json
    }

    /// A database with the graph `trips`, and a handle on it.
    fn with_trips() -> (*mut GrafeoDatabase, *mut GrafeoGraph) {
        let db = grafeo_open_memory();
        let create = c("CREATE GRAPH trips");
        grafeo_free_result(grafeo_execute(db, create.as_ptr()));
        let name = c("trips");
        let graph = grafeo_graph(db, name.as_ptr());
        assert!(!graph.is_null(), "{}", last_error());
        (db, graph)
    }

    fn close(db: *mut GrafeoDatabase, graph: *mut GrafeoGraph) {
        grafeo_free_graph(graph);
        grafeo_close(db);
        grafeo_free_database(db);
    }

    /// An absent node or edge of a graph is `Ok` with NULL; once the graph is
    /// dropped, a read fails with an error status (not NULL with `Ok`).
    #[test]
    fn a_graph_handle_tells_an_absent_node_from_a_failed_read() {
        let (db, graph) = with_trips();
        let (labels, props) = (c(r#"["City"]"#), c(r#"{"name": "Berlin"}"#));
        let berlin = grafeo_graph_create_node(graph, labels.as_ptr(), props.as_ptr());
        let mut node: *mut GrafeoNode = std::ptr::null_mut();
        assert_eq!(
            grafeo_graph_get_node(graph, 424_242, &raw mut node),
            GrafeoStatus::Ok
        );
        assert!(node.is_null(), "absent");
        let mut edge: *mut GrafeoEdge = std::ptr::null_mut();
        assert_eq!(
            grafeo_graph_get_edge(graph, 424_242, &raw mut edge),
            GrafeoStatus::Ok
        );
        assert!(edge.is_null(), "absent");

        let drop = c("DROP GRAPH trips");
        grafeo_free_result(grafeo_execute(db, drop.as_ptr()));
        assert_ne!(
            grafeo_graph_get_node(graph, berlin, &raw mut node),
            GrafeoStatus::Ok,
            "a read of a dropped graph fails"
        );
        assert!(node.is_null());
        assert!(last_error().contains("does not exist"), "{}", last_error());
        close(db, graph);
    }

    #[test]
    fn a_graph_handle_writes_and_reads_in_its_graph_only() {
        let (db, graph) = with_trips();
        let (labels, props) = (c(r#"["City"]"#), c(r#"{"name": "Prague"}"#));
        let prague = grafeo_graph_create_node(graph, labels.as_ptr(), props.as_ptr());
        let props = c(r#"{"name": "Berlin"}"#);
        let berlin = grafeo_graph_create_node(graph, labels.as_ptr(), props.as_ptr());
        let route = c("ROUTE");
        let edge =
            grafeo_graph_create_edge(graph, prague, berlin, route.as_ptr(), std::ptr::null());
        assert_ne!(edge, u64::MAX, "{}", last_error());
        assert_eq!(
            crate::database::counted(grafeo_node_count, db),
            0,
            "the default graph is untouched"
        );

        let query = c("MATCH (a:City)-[:ROUTE]->(b:City) RETURN a.name, b.name");
        assert_eq!(
            result_rows(grafeo_graph_execute(graph, query.as_ptr())),
            serde_json::json!([{"a.name": "Prague", "b.name": "Berlin"}])
        );
        let in_default = grafeo_execute(db, query.as_ptr());
        assert_eq!(result_rows(in_default), serde_json::json!([]));

        let (key, value) = (c("country"), c(r#""CZ""#));
        assert_eq!(
            grafeo_graph_set_node_property(graph, prague, key.as_ptr(), value.as_ptr()),
            GrafeoStatus::Ok
        );
        let capital = c("Capital");
        assert_eq!(
            grafeo_graph_add_node_label(graph, prague, capital.as_ptr()),
            1
        );
        let mut node = std::ptr::null_mut();
        assert_eq!(
            grafeo_graph_get_node(graph, prague, &raw mut node),
            GrafeoStatus::Ok
        );
        // SAFETY: valid until the node is freed.
        let (labels, props) = unsafe {
            (
                CStr::from_ptr(grafeo_node_labels_json(node))
                    .to_str()
                    .unwrap()
                    .to_string(),
                CStr::from_ptr(grafeo_node_properties_json(node))
                    .to_str()
                    .unwrap()
                    .to_string(),
            )
        };
        grafeo_free_node(node);
        let labels: serde_json::Value = serde_json::from_str(&labels).unwrap();
        assert!(
            labels
                .as_array()
                .unwrap()
                .contains(&serde_json::json!("Capital"))
        );
        let props: serde_json::Value = serde_json::from_str(&props).unwrap();
        assert_eq!(props["country"], "CZ");

        let params = c(r#"{"name": "Prague"}"#);
        let lookup = c("MATCH (c:City {name: $name}) RETURN c.country");
        assert_eq!(
            result_rows(grafeo_graph_execute_with_params(
                graph,
                lookup.as_ptr(),
                params.as_ptr()
            )),
            serde_json::json!([{"c.country": "CZ"}])
        );
        assert_eq!(
            grafeo_graph_delete_node(graph, prague),
            -1,
            "Prague has an edge"
        );
        assert_eq!(grafeo_graph_detach_delete_node(graph, prague), 1);
        assert_eq!(grafeo_graph_delete_edge(graph, edge), 0, "gone with Prague");
        close(db, graph);
    }

    #[test]
    fn a_graph_handle_batches_and_upserts_in_its_graph() {
        let (db, graph) = with_trips();
        let (labels, rows) = (
            c(r#"["City"]"#),
            c(r#"[{"id": 3, "name": "Paris"}, {"id": 19, "name": "Amsterdam"}]"#),
        );
        let (mut ids, mut count) = (std::ptr::null_mut(), 0);
        let status = grafeo_graph_batch_create_nodes_with_props(
            graph,
            labels.as_ptr(),
            rows.as_ptr(),
            &raw mut ids,
            &raw mut count,
        );
        assert_eq!(status, GrafeoStatus::Ok, "{}", last_error());
        // SAFETY: the call returned `count` ids.
        let cities = unsafe { std::slice::from_raw_parts(ids, count) }.to_vec();
        grafeo_free_node_ids(ids, count);
        let edges = c(&format!(
            r#"[{{"src": {}, "dst": {}, "type": "ROUTE"}}]"#,
            cities[0], cities[1]
        ));
        let status =
            grafeo_graph_batch_create_edges(graph, edges.as_ptr(), &raw mut ids, &raw mut count);
        assert_eq!(status, GrafeoStatus::Ok, "{}", last_error());
        grafeo_free_node_ids(ids, count);

        let rows = c(
            r#"[{"id": 3, "name": "Paris", "country": "FR"}, {"id": 88, "name": "Barcelona"}, {"name": "no key"}]"#,
        );
        let mut out = std::ptr::null_mut();
        let status = grafeo_graph_upsert_nodes(
            graph,
            labels.as_ptr(),
            rows.as_ptr(),
            std::ptr::null(),
            &raw mut out,
        );
        assert_eq!(status, GrafeoStatus::Ok, "{}", last_error());
        assert_eq!(
            take_json(out),
            serde_json::json!({"created": 1, "updated": 1, "skipped": 1, "skipped_rows": [2]})
        );
        let route = c("ROUTE");
        let rows = c(r#"[{"id": 1, "src": 19, "dst": 88, "km": 1500}]"#);
        let status = grafeo_graph_upsert_edges(
            graph,
            route.as_ptr(),
            rows.as_ptr(),
            std::ptr::null(),
            &raw mut out,
        );
        assert_eq!(status, GrafeoStatus::Ok, "{}", last_error());
        assert_eq!(take_json(out)["created"], 1);
        let query = c("MATCH (a:City)-[r:ROUTE]->(b:City) RETURN a.name, b.name ORDER BY a.name");
        assert_eq!(
            result_rows(grafeo_graph_execute(graph, query.as_ptr())),
            serde_json::json!([
                {"a.name": "Amsterdam", "b.name": "Barcelona"},
                {"a.name": "Paris", "b.name": "Amsterdam"}
            ])
        );
        assert_eq!(
            crate::database::counted(grafeo_node_count, db),
            0,
            "the default graph is untouched"
        );
        close(db, graph);
    }

    /// A handle on a graph that does not exist is NULL; one whose graph is
    /// dropped later fails each call.
    #[test]
    fn a_missing_or_dropped_graph_fails_cleanly() {
        let db = grafeo_open_memory();
        let name = c("nowhere");
        assert!(grafeo_graph(db, name.as_ptr()).is_null());
        assert!(last_error().contains("nowhere"), "{}", last_error());
        grafeo_close(db);
        grafeo_free_database(db);

        let (db, graph) = with_trips();
        let drop = c("DROP GRAPH trips");
        grafeo_free_result(grafeo_execute(db, drop.as_ptr()));
        let labels = c(r#"["City"]"#);
        assert_eq!(
            grafeo_graph_create_node(graph, labels.as_ptr(), std::ptr::null()),
            u64::MAX
        );
        let query = c("RETURN 1");
        assert!(grafeo_graph_execute(graph, query.as_ptr()).is_null());
        assert_eq!(grafeo_graph_delete_node(graph, 0), -1);
        close(db, graph);
        grafeo_free_graph(std::ptr::null_mut());
        assert!(grafeo_graph_execute(std::ptr::null_mut(), query.as_ptr()).is_null());
    }

    #[test]
    fn a_query_through_the_handle_counts_its_writes() {
        let (db, graph) = with_trips();
        let insert = c("INSERT (:City {name: 'Berlin'})");
        let result = grafeo_graph_execute(graph, insert.as_ptr());
        assert_eq!(grafeo_result_nodes_created(result), 1);
        grafeo_free_result(result);
        close(db, graph);
    }

    #[test]
    fn a_handle_lists_the_edges_of_its_graph() {
        let (db, graph) = with_trips();
        let labels = c(r#"["City"]"#);
        let prague = grafeo_graph_create_node(graph, labels.as_ptr(), std::ptr::null());
        let berlin = grafeo_graph_create_node(graph, labels.as_ptr(), std::ptr::null());
        let route_type = c("ROUTE");
        let route =
            grafeo_graph_create_edge(graph, prague, berlin, route_type.as_ptr(), std::ptr::null());
        let mut neighbors = std::ptr::null_mut();
        let mut edges = std::ptr::null_mut();
        let mut count = 0;
        let only_routes = c(r#"["ROUTE"]"#);
        let status = grafeo_graph_get_node_edges(
            graph,
            berlin,
            1,
            only_routes.as_ptr(),
            &raw mut neighbors,
            &raw mut edges,
            &raw mut count,
        );
        assert_eq!(status, GrafeoStatus::Ok, "{}", last_error());
        assert_eq!(count, 1);
        // SAFETY: both arrays hold `count` ids.
        assert_eq!(unsafe { (*neighbors, *edges) }, (prague, route));
        grafeo_free_node_ids(neighbors, count);
        grafeo_free_node_ids(edges, count);

        let drop = c("DROP GRAPH trips");
        grafeo_free_result(grafeo_execute(db, drop.as_ptr()));
        let status = grafeo_graph_get_node_edges(
            graph,
            berlin,
            1,
            std::ptr::null(),
            &raw mut neighbors,
            &raw mut edges,
            &raw mut count,
        );
        assert_ne!(status, GrafeoStatus::Ok);
        assert!(neighbors.is_null() && edges.is_null() && count == 0);
        close(db, graph);
    }
}
