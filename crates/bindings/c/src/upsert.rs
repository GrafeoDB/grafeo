//! Upserts (#441): create or update nodes and edges by a key property, rows
//! and options in as JSON, the summary out as JSON.

use std::ffi::c_char;

use grafeo_engine::database::{EdgeUpsertOptions, UpsertSummary};

use crate::batch::{invalid, json_arg, labels, properties, write_json};
use crate::error::{GrafeoStatus, set_error, set_last_error, str_from_ptr};
use crate::types::GrafeoDatabase;

/// The rows of an upsert: a JSON array of property objects.
pub(crate) fn rows(
    rows_json: *const c_char,
) -> Result<
    Vec<std::collections::HashMap<grafeo_common::types::PropertyKey, grafeo_common::types::Value>>,
    GrafeoStatus,
> {
    let rows = json_arg(rows_json, "rows_json")?;
    let Some(items) = rows.as_array() else {
        return Err(invalid(&format!("rows_json: not an array: {rows}")));
    };
    items
        .iter()
        .enumerate()
        .map(|(i, row)| match row {
            serde_json::Value::Object(_) => properties(Some(row), &format!("rows_json[{i}]")),
            other => Err(invalid(&format!("rows_json[{i}]: not an object: {other}"))),
        })
        .collect()
}

/// The options object of an upsert (NULL for the defaults).
fn options(
    options_json: *const c_char,
) -> Result<serde_json::Map<String, serde_json::Value>, GrafeoStatus> {
    if options_json.is_null() {
        return Ok(serde_json::Map::new());
    }
    match json_arg(options_json, "options_json")? {
        serde_json::Value::Object(map) => Ok(map),
        serde_json::Value::Null => Ok(serde_json::Map::new()),
        other => Err(invalid(&format!("options_json: not an object: {other}"))),
    }
}

/// A string option, `None` when not given.
fn text(
    map: &serde_json::Map<String, serde_json::Value>,
    name: &str,
) -> Result<Option<String>, GrafeoStatus> {
    match map.get(name) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(value)) => Ok(Some(value.clone())),
        Some(other) => Err(invalid(&format!(
            "options_json.{name}: not a string: {other}"
        ))),
    }
}

/// A boolean option, `false` when not given.
fn flag(
    map: &serde_json::Map<String, serde_json::Value>,
    name: &str,
) -> Result<bool, GrafeoStatus> {
    match map.get(name) {
        None | Some(serde_json::Value::Null) => Ok(false),
        Some(serde_json::Value::Bool(value)) => Ok(*value),
        Some(other) => Err(invalid(&format!(
            "options_json.{name}: not a boolean: {other}"
        ))),
    }
}

/// The key and `replace` of a node upsert: `{"key": "id", "replace": false}`.
pub(crate) fn node_options(options_json: *const c_char) -> Result<(String, bool), GrafeoStatus> {
    let map = options(options_json)?;
    Ok((
        text(&map, "key")?.unwrap_or_else(|| "id".to_string()),
        flag(&map, "replace")?,
    ))
}

/// The options of an edge upsert: `key`, `endpoint_key`, `endpoint_labels`,
/// `src_field`, `dst_field` and `replace`, each optional.
pub(crate) fn edge_options(options_json: *const c_char) -> Result<EdgeUpsertOptions, GrafeoStatus> {
    let map = options(options_json)?;
    let mut options = EdgeUpsertOptions::new().with_replace(flag(&map, "replace")?);
    if let Some(key) = text(&map, "key")? {
        options = options.with_key(key);
    }
    if let Some(endpoint_key) = text(&map, "endpoint_key")? {
        options = options.with_endpoint_key(endpoint_key);
    }
    if let Some(endpoint_labels) = map.get("endpoint_labels").filter(|value| !value.is_null()) {
        options =
            options.with_endpoint_labels(labels(endpoint_labels, "options_json.endpoint_labels")?);
    }
    if let Some(src_field) = text(&map, "src_field")? {
        options = options.with_src_field(src_field);
    }
    if let Some(dst_field) = text(&map, "dst_field")? {
        options = options.with_dst_field(dst_field);
    }
    Ok(options)
}

/// Hands an upsert's summary to the caller as
/// `{"created", "updated", "skipped", "skipped_rows"}`.
pub(crate) fn write_summary(summary: &UpsertSummary, out_json: *mut *mut c_char) -> GrafeoStatus {
    write_json(
        &serde_json::json!({
            "created": summary.created,
            "updated": summary.updated,
            "skipped": summary.skipped,
            "skipped_rows": summary.skipped_rows,
        }),
        out_json,
    )
}

/// Zeroes `out_json`, an error if it is null.
pub(crate) fn clear_json(out_json: *mut *mut c_char) -> Result<(), GrafeoStatus> {
    if out_json.is_null() {
        set_last_error("Null pointer argument");
        return Err(GrafeoStatus::ErrorNullPointer);
    }
    // SAFETY: checked not null above.
    unsafe { *out_json = std::ptr::null_mut() };
    Ok(())
}

/// Creates or updates one node per object of `rows_json` (a JSON array of
/// property objects), matched by the key property and all labels of
/// `labels_json`, in one statement: a row the schema refuses fails the call
/// and writes nothing. `options_json` is NULL or `{"key": "id", "replace":
/// false}`: with `replace` a node's properties become exactly the row's,
/// without the row's are merged in; a row without the key is skipped.
/// `*out_json` receives `{"created", "updated", "skipped", "skipped_rows"}`
/// (free with `grafeo_free_string`).
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_upsert_nodes(
    db: *mut GrafeoDatabase,
    labels_json: *const c_char,
    rows_json: *const c_char,
    options_json: *const c_char,
    out_json: *mut *mut c_char,
) -> GrafeoStatus {
    let db = db_ref!(db);
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
    match db
        .inner
        .read()
        .upsert_nodes(&label_refs, &key, rows, replace)
    {
        Ok(summary) => write_summary(&summary, out_json),
        Err(e) => set_error(&e),
    }
}

/// Creates or updates one edge of `edge_type` per object of `rows_json`
/// between existing nodes, in one statement. Each row names its endpoints in
/// the source and target fields (`src` and `dst` by default) and holds the
/// edge key; every other field is an edge property. `options_json` is NULL
/// or an object of `key` (default `id`), `endpoint_key` (the node property
/// the endpoint fields hold, default `id`), `endpoint_labels`, `src_field`,
/// `dst_field` and `replace`. A row without the key or an endpoint field, or
/// whose endpoint key matches no node or several, is skipped. `*out_json`
/// receives `{"created", "updated", "skipped", "skipped_rows"}` (free with
/// `grafeo_free_string`).
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_upsert_edges(
    db: *mut GrafeoDatabase,
    edge_type: *const c_char,
    rows_json: *const c_char,
    options_json: *const c_char,
    out_json: *mut *mut c_char,
) -> GrafeoStatus {
    let db = db_ref!(db);
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
    match db.inner.read().upsert_edges(edge_type, rows, &options) {
        Ok(summary) => write_summary(&summary, out_json),
        Err(e) => set_error(&e),
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::{CStr, CString};

    use super::*;
    use crate::database::{
        grafeo_close, grafeo_execute, grafeo_free_database, grafeo_free_result, grafeo_free_string,
        grafeo_open_memory, grafeo_result_json,
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

    fn take_json(out: *mut c_char) -> serde_json::Value {
        assert!(!out.is_null(), "{}", last_error());
        // SAFETY: a string the call handed out.
        let json = serde_json::from_str(unsafe { CStr::from_ptr(out) }.to_str().unwrap()).unwrap();
        grafeo_free_string(out);
        json
    }

    fn rows_of(db: *mut GrafeoDatabase, query: &str) -> serde_json::Value {
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
    fn an_upsert_creates_updates_and_skips_nodes() {
        let db = grafeo_open_memory();
        let labels = c(r#"["File"]"#);
        let rows = c(r#"[{"path": "a.rs", "size": 3}, {"path": "b.rs", "size": 19}]"#);
        let options = c(r#"{"key": "path"}"#);
        let mut out = std::ptr::null_mut();
        let status = grafeo_upsert_nodes(
            db,
            labels.as_ptr(),
            rows.as_ptr(),
            options.as_ptr(),
            &raw mut out,
        );
        assert_eq!(status, GrafeoStatus::Ok, "{}", last_error());
        assert_eq!(
            take_json(out),
            serde_json::json!({"created": 2, "updated": 0, "skipped": 0, "skipped_rows": []})
        );

        // Merge (the default) keeps the other properties; replace drops them.
        let rows = c(r#"[{"path": "a.rs", "lines": 88}, {"size": 1}]"#);
        let status = grafeo_upsert_nodes(
            db,
            labels.as_ptr(),
            rows.as_ptr(),
            options.as_ptr(),
            &raw mut out,
        );
        assert_eq!(status, GrafeoStatus::Ok, "{}", last_error());
        assert_eq!(
            take_json(out),
            serde_json::json!({"created": 0, "updated": 1, "skipped": 1, "skipped_rows": [1]})
        );
        let replace = c(r#"{"key": "path", "replace": true}"#);
        let rows = c(r#"[{"path": "b.rs"}]"#);
        let status = grafeo_upsert_nodes(
            db,
            labels.as_ptr(),
            rows.as_ptr(),
            replace.as_ptr(),
            &raw mut out,
        );
        assert_eq!(status, GrafeoStatus::Ok, "{}", last_error());
        grafeo_free_string(out);
        assert_eq!(
            rows_of(
                db,
                "MATCH (f:File) RETURN f.path, f.size, f.lines ORDER BY f.path"
            ),
            serde_json::json!([
                {"f.path": "a.rs", "f.size": 3, "f.lines": 88},
                {"f.path": "b.rs", "f.size": null, "f.lines": null}
            ])
        );
        grafeo_close(db);
        grafeo_free_database(db);
    }

    #[test]
    fn an_upsert_links_existing_nodes_by_their_keys() {
        let db = grafeo_open_memory();
        rows_of(db, "INSERT (:File {path: 'a.rs'}), (:File {path: 'b.rs'})");
        let imports = c("IMPORTS");
        let rows = c(
            r#"[{"id": "a-b", "from": "a.rs", "to": "b.rs"}, {"id": "a-x", "from": "a.rs", "to": "x.rs"}]"#,
        );
        let options = c(
            r#"{"endpoint_key": "path", "endpoint_labels": ["File"], "src_field": "from", "dst_field": "to"}"#,
        );
        let mut out = std::ptr::null_mut();
        let status = grafeo_upsert_edges(
            db,
            imports.as_ptr(),
            rows.as_ptr(),
            options.as_ptr(),
            &raw mut out,
        );
        assert_eq!(status, GrafeoStatus::Ok, "{}", last_error());
        assert_eq!(
            take_json(out),
            serde_json::json!({"created": 1, "updated": 0, "skipped": 1, "skipped_rows": [1]})
        );
        assert_eq!(
            rows_of(db, "MATCH (a)-[:IMPORTS]->(b) RETURN a.path, b.path"),
            serde_json::json!([{"a.path": "a.rs", "b.path": "b.rs"}])
        );
        grafeo_close(db);
        grafeo_free_database(db);
    }

    #[test]
    fn bad_upsert_arguments_fail_with_a_message() {
        let db = grafeo_open_memory();
        let labels = c(r#"["File"]"#);
        let mut out = std::ptr::null_mut();
        for (rows, options, named) in [
            ("[3]", "null", "rows_json[0]"),
            ("[]", r#"{"key": 3}"#, "options_json.key"),
            ("[]", r#"{"replace": "yes"}"#, "options_json.replace"),
            ("[]", "[]", "options_json"),
        ] {
            let (rows, options) = (c(rows), c(options));
            let status = grafeo_upsert_nodes(
                db,
                labels.as_ptr(),
                rows.as_ptr(),
                options.as_ptr(),
                &raw mut out,
            );
            assert_eq!(status, GrafeoStatus::ErrorSerialization, "{named}");
            assert!(last_error().contains(named), "{named}: {}", last_error());
            assert!(out.is_null());
        }
        let imports = c("IMPORTS");
        let (rows, options) = (c("[]"), c(r#"{"endpoint_labels": "File"}"#));
        let status = grafeo_upsert_edges(
            db,
            imports.as_ptr(),
            rows.as_ptr(),
            options.as_ptr(),
            &raw mut out,
        );
        assert_eq!(status, GrafeoStatus::ErrorSerialization);
        assert!(last_error().contains("endpoint_labels"), "{}", last_error());
        grafeo_close(db);
        grafeo_free_database(db);
    }
}
