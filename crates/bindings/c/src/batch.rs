//! Batch writes and batch vector search (#441): rows in as JSON, ids out as
//! arrays (freed with `grafeo_free_node_ids`), search results out as JSON.

use std::collections::HashMap;
use std::ffi::{CString, c_char};

use grafeo_bindings_common::json::json_to_value;
use grafeo_common::types::{NodeId, PropertyKey, Value};
use grafeo_engine::database::BatchEdge;

use crate::error::{GrafeoStatus, set_error, set_last_error, str_from_ptr};
use crate::types::GrafeoDatabase;

/// The JSON a C string argument holds; `what` names the argument in the
/// error.
pub(crate) fn json_arg(ptr: *const c_char, what: &str) -> Result<serde_json::Value, GrafeoStatus> {
    let text = str_from_ptr(ptr)?;
    serde_json::from_str(text).map_err(|e| invalid(&format!("{what}: invalid JSON: {e}")))
}

/// Sets `message` as the last error, the status of an argument that does not
/// hold what the function takes.
pub(crate) fn invalid(message: &str) -> GrafeoStatus {
    set_last_error(message);
    GrafeoStatus::ErrorSerialization
}

/// The properties of a JSON object (null for none).
pub(crate) fn properties(
    value: Option<&serde_json::Value>,
    what: &str,
) -> Result<HashMap<PropertyKey, Value>, GrafeoStatus> {
    match value {
        None | Some(serde_json::Value::Null) => Ok(HashMap::new()),
        Some(serde_json::Value::Object(map)) => Ok(map
            .iter()
            .map(|(key, value)| (PropertyKey::new(key.as_str()), json_to_value(value)))
            .collect()),
        Some(other) => Err(invalid(&format!("{what}: not an object: {other}"))),
    }
}

/// The labels of a JSON array of strings.
pub(crate) fn labels(value: &serde_json::Value, what: &str) -> Result<Vec<String>, GrafeoStatus> {
    let Some(items) = value.as_array() else {
        return Err(invalid(&format!(
            "{what}: not an array of strings: {value}"
        )));
    };
    items
        .iter()
        .map(|item| {
            item.as_str()
                .map(String::from)
                .ok_or_else(|| invalid(&format!("{what}: not a string: {item}")))
        })
        .collect()
}

/// The edges of a JSON array of `{"src", "dst", "type", "properties"}`
/// objects (`properties` optional).
pub(crate) fn batch_edges(value: &serde_json::Value) -> Result<Vec<BatchEdge>, GrafeoStatus> {
    let Some(items) = value.as_array() else {
        return Err(invalid(&format!("edges_json: not an array: {value}")));
    };
    items
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let id = |key: &str| {
                item.get(key)
                    .and_then(serde_json::Value::as_u64)
                    .map(NodeId::new)
                    .ok_or_else(|| invalid(&format!("edges_json[{i}]: no node id in \"{key}\"")))
            };
            let edge_type = item
                .get("type")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| invalid(&format!("edges_json[{i}]: no string \"type\"")))?;
            let mut edge = BatchEdge::new(id("src")?, id("dst")?, edge_type);
            edge.properties = properties(
                item.get("properties"),
                &format!("edges_json[{i}].properties"),
            )?;
            Ok(edge)
        })
        .collect()
}

/// Hands `ids` to the caller through `out_ids` and `out_count`, which the
/// caller checked are not null.
pub(crate) fn write_ids(
    ids: Vec<u64>,
    out_ids: *mut *mut u64,
    out_count: *mut usize,
) -> GrafeoStatus {
    let mut ids = ids;
    ids.shrink_to_fit();
    let count = ids.len();
    let ptr = ids.as_mut_ptr();
    std::mem::forget(ids);
    // SAFETY: the caller checked that both are not null.
    unsafe {
        *out_ids = ptr;
        *out_count = count;
    }
    GrafeoStatus::Ok
}

/// Zeroes the outputs of a function that returns ids, so no error path leaves
/// them uninitialized; an error if either is null.
pub(crate) fn clear_ids(out_ids: *mut *mut u64, out_count: *mut usize) -> Result<(), GrafeoStatus> {
    if out_ids.is_null() || out_count.is_null() {
        set_last_error("Null pointer argument");
        return Err(GrafeoStatus::ErrorNullPointer);
    }
    // SAFETY: both checked not null above.
    unsafe {
        *out_ids = std::ptr::null_mut();
        *out_count = 0;
    }
    Ok(())
}

/// Hands `json` to the caller through `out_json` (free with
/// `grafeo_free_string`), which the caller checked is not null.
pub(crate) fn write_json(json: &serde_json::Value, out_json: *mut *mut c_char) -> GrafeoStatus {
    let text = serde_json::to_string(json).unwrap_or_default();
    let ptr = CString::new(text).map_or(std::ptr::null_mut(), CString::into_raw);
    // SAFETY: the caller checked it is not null.
    unsafe { *out_json = ptr };
    GrafeoStatus::Ok
}

/// Creates one node with the labels of `labels_json` (a JSON array of
/// strings) per object of `rows_json` (a JSON array of property objects), in
/// one transaction: one row the schema refuses fails the call and creates
/// none. The node ids, in row order, go to `*out_ids` (free with
/// `grafeo_free_node_ids(*out_ids, *out_count)`).
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_batch_create_nodes_with_props(
    db: *mut GrafeoDatabase,
    labels_json: *const c_char,
    rows_json: *const c_char,
    out_ids: *mut *mut u64,
    out_count: *mut usize,
) -> GrafeoStatus {
    let db = db_ref!(db);
    if let Err(status) = clear_ids(out_ids, out_count) {
        return status;
    }
    let (labels, rows) = match parse_node_rows(labels_json, rows_json) {
        Ok(parsed) => parsed,
        Err(status) => return status,
    };
    let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
    match db
        .inner
        .read()
        .batch_create_nodes_with_labels(&label_refs, rows)
    {
        Ok(ids) => write_ids(
            ids.iter().map(|id| id.as_u64()).collect(),
            out_ids,
            out_count,
        ),
        Err(e) => set_error(&e),
    }
}

/// The labels and property rows of a batch node create.
pub(crate) fn parse_node_rows(
    labels_json: *const c_char,
    rows_json: *const c_char,
) -> Result<(Vec<String>, Vec<HashMap<PropertyKey, Value>>), GrafeoStatus> {
    let labels = labels(&json_arg(labels_json, "labels_json")?, "labels_json")?;
    let rows = json_arg(rows_json, "rows_json")?;
    let Some(rows) = rows.as_array() else {
        return Err(invalid(&format!("rows_json: not an array: {rows}")));
    };
    let rows = rows
        .iter()
        .enumerate()
        .map(|(i, row)| properties(Some(row), &format!("rows_json[{i}]")))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((labels, rows))
}

/// Creates the edges of `edges_json`, a JSON array of
/// `{"src": id, "dst": id, "type": "KNOWS", "properties": {...}}` objects
/// (`properties` optional), in one transaction: one edge the schema refuses
/// or naming a node that does not exist fails the call and creates none.
/// The edge ids, in input order, go to `*out_ids` (free with
/// `grafeo_free_node_ids(*out_ids, *out_count)`).
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_batch_create_edges(
    db: *mut GrafeoDatabase,
    edges_json: *const c_char,
    out_ids: *mut *mut u64,
    out_count: *mut usize,
) -> GrafeoStatus {
    let db = db_ref!(db);
    if let Err(status) = clear_ids(out_ids, out_count) {
        return status;
    }
    let edges = match json_arg(edges_json, "edges_json").and_then(|json| batch_edges(&json)) {
        Ok(edges) => edges,
        Err(status) => return status,
    };
    match db.inner.read().batch_create_edges(edges) {
        Ok(ids) => write_ids(
            ids.iter().map(|id| id.as_u64()).collect(),
            out_ids,
            out_count,
        ),
        Err(e) => set_error(&e),
    }
}

/// Searches the `k` nearest neighbours of each of `query_count` query vectors
/// (a flat array of `query_count * dimensions` floats) in the vector index on
/// `label`.`property`. `ef` is the search beam width (0 for the index's
/// default); `filters_json` (or NULL) a JSON object of property equalities the
/// results must match. `*out_json` receives a JSON array with one array per
/// query of `{"id": node id, "distance": d}` objects, nearest first (free with
/// `grafeo_free_string`).
#[cfg(feature = "vector-index")]
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_batch_vector_search(
    db: *mut GrafeoDatabase,
    label: *const c_char,
    property: *const c_char,
    queries: *const f32,
    query_count: usize,
    dimensions: usize,
    k: usize,
    ef: usize,
    filters_json: *const c_char,
    out_json: *mut *mut c_char,
) -> GrafeoStatus {
    let db = db_ref!(db);
    if out_json.is_null() || (queries.is_null() && query_count > 0) {
        set_last_error("Null pointer argument");
        return GrafeoStatus::ErrorNullPointer;
    }
    // SAFETY: checked not null above.
    unsafe { *out_json = std::ptr::null_mut() };
    let (label, property) = match (str_from_ptr(label), str_from_ptr(property)) {
        (Ok(label), Ok(property)) => (label, property),
        (Err(status), _) | (_, Err(status)) => return status,
    };
    let Some(total) = query_count.checked_mul(dimensions) else {
        set_last_error("query_count * dimensions overflows");
        return GrafeoStatus::ErrorInternal;
    };
    let vectors: Vec<Vec<f32>> = if total == 0 {
        Vec::new()
    } else {
        // SAFETY: the caller passes query_count * dimensions floats.
        let flat = unsafe { std::slice::from_raw_parts(queries, total) };
        flat.chunks(dimensions.max(1))
            .map(<[f32]>::to_vec)
            .collect()
    };
    let filters = if filters_json.is_null() {
        None
    } else {
        match json_arg(filters_json, "filters_json")
            .and_then(|json| properties(Some(&json), "filters_json"))
        {
            Ok(map) => Some(
                map.into_iter()
                    .map(|(key, value)| (key.as_str().to_string(), value))
                    .collect::<HashMap<String, Value>>(),
            ),
            Err(status) => return status,
        }
    };
    let ef = (ef > 0).then_some(ef);
    match db
        .inner
        .read()
        .batch_vector_search(label, property, &vectors, k, ef, filters.as_ref())
    {
        Ok(results) => {
            let json = serde_json::Value::Array(
                results
                    .into_iter()
                    .map(|hits| {
                        serde_json::Value::Array(
                            hits.into_iter()
                                .map(|(id, distance)| {
                                    serde_json::json!({"id": id.as_u64(), "distance": distance})
                                })
                                .collect(),
                        )
                    })
                    .collect(),
            );
            write_json(&json, out_json)
        }
        Err(e) => set_error(&e),
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::{CStr, CString};

    use super::*;
    use crate::database::{
        grafeo_close, grafeo_edge_count, grafeo_execute, grafeo_free_database,
        grafeo_free_node_ids, grafeo_free_result, grafeo_free_string, grafeo_node_count,
        grafeo_open_memory, grafeo_result_edges_created, grafeo_result_json,
        grafeo_result_labels_added, grafeo_result_nodes_created, grafeo_result_nodes_deleted,
        grafeo_result_properties_set,
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

    /// The ids a batch call handed out, freed again.
    fn take_ids(ids: *mut u64, count: usize) -> Vec<u64> {
        // SAFETY: the call returned `count` ids at `ids`.
        let copy = unsafe { std::slice::from_raw_parts(ids, count) }.to_vec();
        grafeo_free_node_ids(ids, count);
        copy
    }

    fn rows(db: *mut GrafeoDatabase, query: &str) -> serde_json::Value {
        let query = c(query);
        let result = grafeo_execute(db, query.as_ptr());
        assert!(!result.is_null(), "{}", last_error());
        // SAFETY: valid until the result is freed.
        let json = unsafe { CStr::from_ptr(grafeo_result_json(result)) }
            .to_str()
            .unwrap()
            .to_string();
        grafeo_free_result(result);
        serde_json::from_str(&json).unwrap()
    }

    #[test]
    fn a_batch_creates_nodes_with_labels_and_properties_then_edges() {
        let db = grafeo_open_memory();
        let (labels, nodes) = (
            c(r#"["Person", "Employee"]"#),
            c(r#"[{"name": "Alix", "age": 30}, {"name": "Gus"}, {}]"#),
        );
        let (mut ids, mut count) = (std::ptr::null_mut(), 0);
        let status = grafeo_batch_create_nodes_with_props(
            db,
            labels.as_ptr(),
            nodes.as_ptr(),
            &raw mut ids,
            &raw mut count,
        );
        assert_eq!(status, GrafeoStatus::Ok, "{}", last_error());
        let people = take_ids(ids, count);
        assert_eq!(people.len(), 3);
        assert_eq!(
            rows(
                db,
                "MATCH (p:Person:Employee) RETURN p.name ORDER BY p.name"
            ),
            serde_json::json!([{"p.name": "Alix"}, {"p.name": "Gus"}, {"p.name": null}])
        );

        let edges = c(&format!(
            r#"[{{"src": {a}, "dst": {g}, "type": "KNOWS", "properties": {{"since": 2020}}}},
                {{"src": {g}, "dst": {a}, "type": "KNOWS"}}]"#,
            a = people[0],
            g = people[1]
        ));
        let status = grafeo_batch_create_edges(db, edges.as_ptr(), &raw mut ids, &raw mut count);
        assert_eq!(status, GrafeoStatus::Ok, "{}", last_error());
        assert_eq!(take_ids(ids, count).len(), 2);
        assert_eq!(
            rows(
                db,
                "MATCH (a)-[k:KNOWS]->(b) RETURN a.name, b.name, k.since ORDER BY a.name"
            ),
            serde_json::json!([
                {"a.name": "Alix", "b.name": "Gus", "k.since": 2020},
                {"a.name": "Gus", "b.name": "Alix", "k.since": null}
            ])
        );
        grafeo_close(db);
        grafeo_free_database(db);
    }

    /// One edge naming a node that does not exist fails the whole batch.
    #[test]
    fn a_batch_with_a_bad_edge_creates_none() {
        let db = grafeo_open_memory();
        let (labels, nodes) = (c(r#"["Person"]"#), c(r#"[{"name": "Vincent"}]"#));
        let (mut ids, mut count) = (std::ptr::null_mut(), 0);
        grafeo_batch_create_nodes_with_props(
            db,
            labels.as_ptr(),
            nodes.as_ptr(),
            &raw mut ids,
            &raw mut count,
        );
        let vincent = take_ids(ids, count)[0];
        let edges = c(&format!(
            r#"[{{"src": {vincent}, "dst": {vincent}, "type": "SELF"}},
                {{"src": {vincent}, "dst": 88888, "type": "KNOWS"}}]"#
        ));
        let status = grafeo_batch_create_edges(db, edges.as_ptr(), &raw mut ids, &raw mut count);
        assert_ne!(status, GrafeoStatus::Ok);
        assert!(ids.is_null() && count == 0, "no ids on an error");
        assert_eq!(
            crate::database::counted(grafeo_edge_count, db),
            0,
            "none of the batch"
        );
        grafeo_close(db);
        grafeo_free_database(db);
    }

    /// Arguments that are not what the call takes fail with the
    /// serialization status and a message naming them.
    #[test]
    fn bad_json_arguments_fail_with_a_message() {
        let db = grafeo_open_memory();
        let (mut ids, mut count) = (std::ptr::null_mut(), 0);
        for (labels, nodes, named) in [
            ("[\"Person\"", "[]", "labels_json"),
            ("[3]", "[]", "labels_json"),
            ("[\"Person\"]", "{\"name\": \"Mia\"}", "rows_json"),
            ("[\"Person\"]", "[\"Mia\"]", "rows_json[0]"),
        ] {
            let (labels, nodes) = (c(labels), c(nodes));
            let status = grafeo_batch_create_nodes_with_props(
                db,
                labels.as_ptr(),
                nodes.as_ptr(),
                &raw mut ids,
                &raw mut count,
            );
            assert_eq!(status, GrafeoStatus::ErrorSerialization, "{named}");
            assert!(last_error().contains(named), "{named}: {}", last_error());
        }
        let edges = c(r#"[{"src": 1, "type": "KNOWS"}]"#);
        let status = grafeo_batch_create_edges(db, edges.as_ptr(), &raw mut ids, &raw mut count);
        assert_eq!(status, GrafeoStatus::ErrorSerialization);
        assert!(last_error().contains("\"dst\""), "{}", last_error());
        assert_eq!(crate::database::counted(grafeo_node_count, db), 0);
        grafeo_close(db);
        grafeo_free_database(db);
    }

    #[test]
    fn a_result_counts_what_the_statement_wrote() {
        let db = grafeo_open_memory();
        let query =
            c("INSERT (:Person {name: 'Jules', city: 'Paris'})-[:KNOWS]->(:Person {name: 'Mia'})");
        let result = grafeo_execute(db, query.as_ptr());
        assert_eq!(grafeo_result_nodes_created(result), 2);
        assert_eq!(grafeo_result_edges_created(result), 1);
        assert_eq!(grafeo_result_properties_set(result), 3);
        assert_eq!(grafeo_result_labels_added(result), 2);
        assert_eq!(grafeo_result_nodes_deleted(result), 0);
        grafeo_free_result(result);
        let read = c("MATCH (p:Person) RETURN p.name");
        let result = grafeo_execute(db, read.as_ptr());
        assert_eq!(
            grafeo_result_nodes_created(result),
            0,
            "a read writes nothing"
        );
        grafeo_free_result(result);
        assert_eq!(grafeo_result_nodes_created(std::ptr::null()), 0);
        grafeo_close(db);
        grafeo_free_database(db);
    }

    #[cfg(feature = "vector-index")]
    #[test]
    fn a_batch_vector_search_answers_each_query_nearest_first() {
        use crate::database::grafeo_create_vector_index;

        let db = grafeo_open_memory();
        let (labels, nodes) = (
            c(r#"["Doc"]"#),
            c(r#"[{"topic": "a"}, {"topic": "b"}, {"topic": "a"}]"#),
        );
        let (mut ids, mut count) = (std::ptr::null_mut(), 0);
        grafeo_batch_create_nodes_with_props(
            db,
            labels.as_ptr(),
            nodes.as_ptr(),
            &raw mut ids,
            &raw mut count,
        );
        let docs = take_ids(ids, count);
        for (id, vector) in docs.iter().zip(["[1.0, 0.0]", "[0.0, 1.0]", "[0.9, 0.1]"]) {
            rows(
                db,
                &format!("MATCH (d:Doc) WHERE id(d) = {id} SET d.embedding = vector({vector})"),
            );
        }
        let (label, property, metric) = (c("Doc"), c("embedding"), c("cosine"));
        let status = grafeo_create_vector_index(
            db,
            label.as_ptr(),
            property.as_ptr(),
            2,
            metric.as_ptr(),
            0,
            0,
        );
        assert_eq!(status, GrafeoStatus::Ok, "{}", last_error());

        let queries = [1.0_f32, 0.0, 0.0, 1.0];
        let mut out = std::ptr::null_mut();
        let status = grafeo_batch_vector_search(
            db,
            label.as_ptr(),
            property.as_ptr(),
            queries.as_ptr(),
            2,
            2,
            1,
            0,
            std::ptr::null(),
            &raw mut out,
        );
        assert_eq!(status, GrafeoStatus::Ok, "{}", last_error());
        // SAFETY: a string the call handed out.
        let json: serde_json::Value =
            serde_json::from_str(unsafe { CStr::from_ptr(out) }.to_str().unwrap()).unwrap();
        grafeo_free_string(out);
        assert_eq!(json[0][0]["id"], docs[0]);
        assert_eq!(json[1][0]["id"], docs[1]);

        let filters = c(r#"{"topic": "a"}"#);
        let status = grafeo_batch_vector_search(
            db,
            label.as_ptr(),
            property.as_ptr(),
            queries[2..].as_ptr(),
            1,
            2,
            1,
            0,
            filters.as_ptr(),
            &raw mut out,
        );
        assert_eq!(status, GrafeoStatus::Ok, "{}", last_error());
        // SAFETY: a string the call handed out.
        let json: serde_json::Value =
            serde_json::from_str(unsafe { CStr::from_ptr(out) }.to_str().unwrap()).unwrap();
        grafeo_free_string(out);
        assert_ne!(json[0][0]["id"], docs[1], "the filter leaves topic b out");
        grafeo_close(db);
        grafeo_free_database(db);
    }
}
