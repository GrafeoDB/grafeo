//! After `grafeo_close`, every call on the database and on what it handed out fails with
//! `GRAFEO-T007`, through its own error channel: a status, NULL, -1 or `u64::MAX`, with the
//! last error set. The calls without an error return clear the last error when they start
//! and return their neutral value with `GRAFEO-T007` set on a closed database, so a caller
//! can tell "closed" from a neutral answer.

use std::ffi::{CStr, CString, c_char};

use crate::batch::{grafeo_batch_create_edges, grafeo_batch_create_nodes_with_props};
use crate::database::{
    grafeo_add_node_label, grafeo_begin_transaction, grafeo_begin_transaction_with_isolation,
    grafeo_checkpoint_status, grafeo_clear_plan_cache, grafeo_close, grafeo_commit, grafeo_compact,
    grafeo_create_edge, grafeo_create_node, grafeo_create_projection, grafeo_create_property_index,
    grafeo_current_schema, grafeo_delete_edge, grafeo_delete_node, grafeo_detach_delete_node,
    grafeo_drop_projection, grafeo_drop_property_index, grafeo_edge_count, grafeo_execute,
    grafeo_execute_language, grafeo_execute_with_params, grafeo_find_nodes_by_property,
    grafeo_free_database, grafeo_free_transaction, grafeo_get_edge, grafeo_get_node,
    grafeo_get_node_labels, grafeo_has_property_index, grafeo_info, grafeo_list_projections,
    grafeo_memory_usage_json, grafeo_node_count, grafeo_open, grafeo_open_warnings,
    grafeo_remove_edge_property, grafeo_remove_node_label, grafeo_remove_node_property,
    grafeo_reset_schema, grafeo_set_checkpoint_triggers, grafeo_set_edge_property,
    grafeo_set_node_property, grafeo_set_schema, grafeo_transaction_execute, grafeo_wal_checkpoint,
};
use crate::edges::grafeo_get_node_edges;
use crate::error::{GrafeoStatus, grafeo_clear_error, grafeo_last_error};
use crate::graph::{grafeo_free_graph, grafeo_graph, grafeo_graph_execute};
use crate::session::{grafeo_free_session, grafeo_session_execute, grafeo_session_open};
use crate::stream::{grafeo_stream_free, grafeo_stream_next_row_json, grafeo_stream_open};
use crate::time_travel::{
    grafeo_as_of, grafeo_as_of_epoch, grafeo_as_of_execute, grafeo_current_epoch,
    grafeo_epoch_as_of, grafeo_execute_as_of, grafeo_free_as_of,
};
use crate::types::GrafeoDatabase;
use crate::upsert::{grafeo_upsert_edges, grafeo_upsert_nodes};
use crate::warning::{
    grafeo_stream_warning_code, grafeo_stream_warning_count, grafeo_stream_warning_kind,
    grafeo_stream_warning_message,
};

/// The last error on this thread, empty for none.
fn last_error() -> String {
    let pointer = grafeo_last_error();
    if pointer.is_null() {
        return String::new();
    }
    // SAFETY: the pointer is valid until the next call on this thread.
    unsafe { CStr::from_ptr(pointer) }
        .to_string_lossy()
        .into_owned()
}

fn text(value: &str) -> CString {
    CString::new(value).unwrap()
}

/// A persistent database at a fresh path, with two people and an edge between them.
fn open_database(name: &str) -> (*mut GrafeoDatabase, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!("grafeo-c-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = text(dir.join("amsterdam.grafeo").to_str().unwrap());
    let db = grafeo_open(path.as_ptr());
    assert!(!db.is_null(), "{}", last_error());
    let insert = text("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})");
    let result = grafeo_execute(db, insert.as_ptr());
    assert!(!result.is_null(), "{}", last_error());
    crate::database::grafeo_free_result(result);
    (db, dir)
}

/// The names of the rows whose call did not fail, or failed with another error.
fn not_closed(rows: Vec<(&'static str, bool, String)>) -> Vec<String> {
    rows.into_iter()
        .filter(|(_, failed, error)| !failed || !error.contains("GRAFEO-T007"))
        .map(|(name, failed, error)| format!("{name}: failed {failed}, error {error:?}"))
        .collect()
}

/// Runs `call`, which says whether the call reported a failure, and takes the last error.
fn row(name: &'static str, call: impl FnOnce() -> bool) -> (&'static str, bool, String) {
    grafeo_clear_error();
    let failed = call();
    (name, failed, last_error())
}

fn status(status: GrafeoStatus) -> bool {
    status != GrafeoStatus::Ok
}

/// The exports of `grafeo.h` whose first argument is a `handle` pointer (`GrafeoDatabase`,
/// `GrafeoGraph`, ...), by name: comments stripped, each the identifier before its `(`.
fn exports_on(handle: &str) -> Vec<String> {
    let mut header = include_str!("../grafeo.h").to_string();
    while let Some(open) = header.find("/*") {
        let close = header[open..]
            .find("*/")
            .map_or(header.len(), |end| open + end + 2);
        header.replace_range(open..close, " ");
    }
    let mut names = Vec::new();
    for declaration in header.split(';') {
        let Some(paren) = declaration.find('(') else {
            continue;
        };
        let name = declaration[..paren]
            .trim_end()
            .rsplit(|c: char| c.is_whitespace() || c == '*')
            .next()
            .unwrap_or("");
        let arguments = declaration[paren + 1..].trim_start();
        let first = arguments.strip_prefix("const ").unwrap_or(arguments);
        if name.starts_with("grafeo_")
            && first
                .strip_prefix(handle)
                .is_some_and(|rest| rest.trim_start().starts_with('*'))
        {
            names.push(name.to_string());
        }
    }
    names
}

#[test]
fn every_call_on_a_closed_database_fails_with_t007() {
    let (db, dir) = open_database("closed-calls");
    assert_eq!(grafeo_close(db), GrafeoStatus::Ok);

    let query = text("MATCH (p:Person) RETURN p.name");
    let gql = text("gql");
    let labels = text("[\"Person\"]");
    let properties = text("{\"name\": \"Vincent\"}");
    let key = text("name");
    let value = text("\"Amsterdam\"");
    let label = text("Director");
    let edge_type = text("KNOWS");
    let schema = text("berlin");
    let rows_json = text("[{\"id\": 1, \"name\": \"Jules\"}]");
    let edges_json = text("[{\"src\": 0, \"dst\": 1, \"type\": \"KNOWS\"}]");
    let mut node = std::ptr::null_mut();
    let mut edge = std::ptr::null_mut();
    let mut json: *mut c_char = std::ptr::null_mut();
    let mut ids: *mut u64 = std::ptr::null_mut();
    let mut neighbors: *mut u64 = std::ptr::null_mut();
    let mut count = 0usize;
    let mut epoch = 0u64;

    let mut rows = vec![
        row("grafeo_execute", || {
            grafeo_execute(db, query.as_ptr()).is_null()
        }),
        row("grafeo_execute_with_params", || {
            grafeo_execute_with_params(db, query.as_ptr(), std::ptr::null()).is_null()
        }),
        row("grafeo_execute_language", || {
            grafeo_execute_language(db, gql.as_ptr(), query.as_ptr(), std::ptr::null()).is_null()
        }),
        row("grafeo_stream_open", || {
            grafeo_stream_open(db, query.as_ptr()).is_null()
        }),
        row("grafeo_set_schema", || {
            status(grafeo_set_schema(db, schema.as_ptr()))
        }),
        row("grafeo_reset_schema", || status(grafeo_reset_schema(db))),
        row("grafeo_current_schema", || {
            grafeo_current_schema(db).is_null()
        }),
        row("grafeo_create_node", || {
            grafeo_create_node(db, labels.as_ptr(), properties.as_ptr()) == u64::MAX
        }),
        row("grafeo_get_node", || {
            status(grafeo_get_node(db, 0, &raw mut node))
        }),
        row("grafeo_delete_node", || grafeo_delete_node(db, 0) == -1),
        row("grafeo_detach_delete_node", || {
            grafeo_detach_delete_node(db, 0) == -1
        }),
        row("grafeo_set_node_property", || {
            status(grafeo_set_node_property(
                db,
                0,
                key.as_ptr(),
                value.as_ptr(),
            ))
        }),
        row("grafeo_remove_node_property", || {
            grafeo_remove_node_property(db, 0, key.as_ptr()) == -1
        }),
        row("grafeo_add_node_label", || {
            grafeo_add_node_label(db, 0, label.as_ptr()) == -1
        }),
        row("grafeo_remove_node_label", || {
            grafeo_remove_node_label(db, 0, label.as_ptr()) == -1
        }),
        row("grafeo_get_node_labels", || {
            status(grafeo_get_node_labels(db, 0, &raw mut json))
        }),
        row("grafeo_create_edge", || {
            grafeo_create_edge(db, 0, 1, edge_type.as_ptr(), std::ptr::null()) == u64::MAX
        }),
        row("grafeo_get_edge", || {
            status(grafeo_get_edge(db, 0, &raw mut edge))
        }),
        row("grafeo_delete_edge", || grafeo_delete_edge(db, 0) == -1),
        row("grafeo_set_edge_property", || {
            status(grafeo_set_edge_property(
                db,
                0,
                key.as_ptr(),
                value.as_ptr(),
            ))
        }),
        row("grafeo_remove_edge_property", || {
            grafeo_remove_edge_property(db, 0, key.as_ptr()) == -1
        }),
        row("grafeo_get_node_edges", || {
            status(grafeo_get_node_edges(
                db,
                0,
                0,
                std::ptr::null(),
                &raw mut neighbors,
                &raw mut ids,
                &raw mut count,
            ))
        }),
        row("grafeo_create_property_index", || {
            status(grafeo_create_property_index(db, key.as_ptr()))
        }),
        row("grafeo_drop_property_index", || {
            grafeo_drop_property_index(db, key.as_ptr()) == -1
        }),
        row("grafeo_has_property_index", || {
            grafeo_has_property_index(db, key.as_ptr()) == 0
        }),
        row("grafeo_find_nodes_by_property", || {
            status(grafeo_find_nodes_by_property(
                db,
                key.as_ptr(),
                value.as_ptr(),
                &raw mut ids,
                &raw mut count,
            ))
        }),
        row("grafeo_create_projection", || {
            grafeo_create_projection(
                db,
                schema.as_ptr(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
            ) == -1
        }),
        row("grafeo_drop_projection", || {
            grafeo_drop_projection(db, schema.as_ptr()) == -1
        }),
        row("grafeo_list_projections", || {
            grafeo_list_projections(db).is_null()
        }),
        row("grafeo_batch_create_nodes_with_props", || {
            status(grafeo_batch_create_nodes_with_props(
                db,
                labels.as_ptr(),
                rows_json.as_ptr(),
                &raw mut ids,
                &raw mut count,
            ))
        }),
        row("grafeo_batch_create_edges", || {
            status(grafeo_batch_create_edges(
                db,
                edges_json.as_ptr(),
                &raw mut ids,
                &raw mut count,
            ))
        }),
        row("grafeo_upsert_nodes", || {
            status(grafeo_upsert_nodes(
                db,
                labels.as_ptr(),
                rows_json.as_ptr(),
                std::ptr::null(),
                &raw mut json,
            ))
        }),
        row("grafeo_upsert_edges", || {
            status(grafeo_upsert_edges(
                db,
                edge_type.as_ptr(),
                edges_json.as_ptr(),
                std::ptr::null(),
                &raw mut json,
            ))
        }),
        row("grafeo_graph", || {
            grafeo_graph(db, schema.as_ptr()).is_null()
        }),
        row("grafeo_current_epoch", || grafeo_current_epoch(db) == 0),
        row("grafeo_epoch_as_of", || {
            status(grafeo_epoch_as_of(db, 0, 0, &raw mut epoch))
        }),
        row("grafeo_execute_as_of", || {
            grafeo_execute_as_of(db, query.as_ptr(), std::ptr::null(), 0, 0).is_null()
        }),
        row("grafeo_as_of", || grafeo_as_of(db, 0, 0).is_null()),
        row("grafeo_node_count", || {
            status(grafeo_node_count(db, &raw mut count))
        }),
        row("grafeo_edge_count", || {
            status(grafeo_edge_count(db, &raw mut count))
        }),
        row("grafeo_session_open", || {
            grafeo_session_open(db, 0).is_null()
        }),
        row("grafeo_begin_transaction", || {
            grafeo_begin_transaction(db).is_null()
        }),
        row("grafeo_begin_transaction_with_isolation", || {
            grafeo_begin_transaction_with_isolation(db, 0).is_null()
        }),
        row("grafeo_info", || grafeo_info(db).is_null()),
        row("grafeo_memory_usage_json", || {
            grafeo_memory_usage_json(db).is_null()
        }),
        row("grafeo_open_warnings", || {
            grafeo_open_warnings(db).is_null()
        }),
        row("grafeo_wal_checkpoint", || {
            status(grafeo_wal_checkpoint(db))
        }),
        row("grafeo_set_checkpoint_triggers", || {
            status(grafeo_set_checkpoint_triggers(db, 60_000, 0))
        }),
        row("grafeo_checkpoint_status", || {
            grafeo_checkpoint_status(db).is_null()
        }),
        row("grafeo_compact", || status(grafeo_compact(db))),
        row("grafeo_clear_plan_cache", || {
            status(grafeo_clear_plan_cache(db))
        }),
    ];
    #[cfg(feature = "cdc")]
    rows.extend([
        row("grafeo_set_cdc_enabled", || {
            crate::database::grafeo_set_cdc_enabled(db, true);
            true
        }),
        row("grafeo_is_cdc_enabled", || {
            !crate::database::grafeo_is_cdc_enabled(db)
        }),
        row("grafeo_node_history", || {
            crate::cdc::grafeo_node_history(db, 0).is_null()
        }),
        row("grafeo_edge_history", || {
            crate::cdc::grafeo_edge_history(db, 0).is_null()
        }),
        row("grafeo_node_history_since", || {
            crate::cdc::grafeo_node_history_since(db, 0, 0).is_null()
        }),
        row("grafeo_changes_between", || {
            crate::cdc::grafeo_changes_between(db, 0, 1).is_null()
        }),
    ]);
    #[cfg(feature = "storage")]
    {
        let save_path = text(dir.join("copy.grafeo").to_str().unwrap());
        let backups = text(dir.join("backups").to_str().unwrap());
        let mut released = 0i32;
        rows.extend([
            row("grafeo_save", || {
                status(crate::database::grafeo_save(db, save_path.as_ptr()))
            }),
            row("grafeo_backup_full", || {
                status(crate::database::grafeo_backup_full(db, backups.as_ptr()))
            }),
            row("grafeo_backup_incremental", || {
                status(crate::database::grafeo_backup_incremental(
                    db,
                    backups.as_ptr(),
                ))
            }),
            row("grafeo_release_backup_cursor", || {
                status(crate::database::grafeo_release_backup_cursor(
                    db,
                    &raw mut released,
                ))
            }),
        ]);
    }
    #[cfg(feature = "vector-index")]
    {
        let vector = [0.1_f32, 0.2, 0.3, 0.4];
        let item = text("Item");
        let embedding = text("embedding");
        let mut distances: *mut f32 = std::ptr::null_mut();
        rows.extend([
            row("grafeo_create_vector_index", || {
                status(crate::database::grafeo_create_vector_index(
                    db,
                    item.as_ptr(),
                    embedding.as_ptr(),
                    4,
                    std::ptr::null(),
                    -1,
                    -1,
                ))
            }),
            row("grafeo_drop_vector_index", || {
                crate::database::grafeo_drop_vector_index(db, item.as_ptr(), embedding.as_ptr())
                    == -1
            }),
            row("grafeo_rebuild_vector_index", || {
                status(crate::database::grafeo_rebuild_vector_index(
                    db,
                    item.as_ptr(),
                    embedding.as_ptr(),
                ))
            }),
            row("grafeo_vector_search", || {
                status(crate::database::grafeo_vector_search(
                    db,
                    item.as_ptr(),
                    embedding.as_ptr(),
                    vector.as_ptr(),
                    4,
                    1,
                    -1,
                    &raw mut ids,
                    &raw mut distances,
                    &raw mut count,
                ))
            }),
            row("grafeo_mmr_search", || {
                status(crate::database::grafeo_mmr_search(
                    db,
                    item.as_ptr(),
                    embedding.as_ptr(),
                    vector.as_ptr(),
                    4,
                    1,
                    -1,
                    0.5,
                    -1,
                    &raw mut ids,
                    &raw mut distances,
                    &raw mut count,
                ))
            }),
            row("grafeo_batch_create_nodes", || {
                status(crate::database::grafeo_batch_create_nodes(
                    db,
                    item.as_ptr(),
                    embedding.as_ptr(),
                    vector.as_ptr(),
                    1,
                    4,
                    &raw mut ids,
                    &raw mut count,
                ))
            }),
            row("grafeo_batch_vector_search", || {
                status(crate::batch::grafeo_batch_vector_search(
                    db,
                    item.as_ptr(),
                    embedding.as_ptr(),
                    vector.as_ptr(),
                    1,
                    4,
                    1,
                    0,
                    std::ptr::null(),
                    &raw mut json,
                ))
            }),
        ]);
    }
    #[cfg(feature = "cypher")]
    rows.extend([
        row("grafeo_execute_cypher", || {
            crate::database::grafeo_execute_cypher(db, query.as_ptr()).is_null()
        }),
        row("grafeo_execute_cypher_with_params", || {
            crate::database::grafeo_execute_cypher_with_params(db, query.as_ptr(), std::ptr::null())
                .is_null()
        }),
    ]);
    #[cfg(feature = "gremlin")]
    {
        let traversal = text("g.V()");
        rows.extend([
            row("grafeo_execute_gremlin", || {
                crate::database::grafeo_execute_gremlin(db, traversal.as_ptr()).is_null()
            }),
            row("grafeo_execute_gremlin_with_params", || {
                crate::database::grafeo_execute_gremlin_with_params(
                    db,
                    traversal.as_ptr(),
                    std::ptr::null(),
                )
                .is_null()
            }),
        ]);
    }
    #[cfg(feature = "graphql")]
    {
        let graphql = text("{ Person { name } }");
        rows.extend([
            row("grafeo_execute_graphql", || {
                crate::database::grafeo_execute_graphql(db, graphql.as_ptr()).is_null()
            }),
            row("grafeo_execute_graphql_with_params", || {
                crate::database::grafeo_execute_graphql_with_params(
                    db,
                    graphql.as_ptr(),
                    std::ptr::null(),
                )
                .is_null()
            }),
        ]);
    }
    #[cfg(feature = "sparql")]
    {
        let sparql = text("SELECT ?s WHERE { ?s ?p ?o }");
        rows.extend([
            row("grafeo_execute_sparql", || {
                crate::database::grafeo_execute_sparql(db, sparql.as_ptr()).is_null()
            }),
            row("grafeo_execute_sparql_with_params", || {
                crate::database::grafeo_execute_sparql_with_params(
                    db,
                    sparql.as_ptr(),
                    std::ptr::null(),
                )
                .is_null()
            }),
        ]);
    }
    #[cfg(feature = "sql-pgq")]
    {
        let sql = text("SELECT * FROM GRAPH_TABLE (MATCH (p:Person) COLUMNS (p.name AS name))");
        rows.extend([
            row("grafeo_execute_sql", || {
                crate::database::grafeo_execute_sql(db, sql.as_ptr()).is_null()
            }),
            row("grafeo_execute_sql_with_params", || {
                crate::database::grafeo_execute_sql_with_params(db, sql.as_ptr(), std::ptr::null())
                    .is_null()
            }),
        ]);
    }

    // The table covers every export on a database (all of them in a build with every feature).
    #[cfg(feature = "full")]
    {
        let listed: Vec<&str> = rows.iter().map(|(name, _, _)| *name).collect();
        let unlisted: Vec<String> = exports_on("GrafeoDatabase")
            .into_iter()
            .filter(|name| {
                !listed.contains(&name.as_str())
                    && name != "grafeo_close"
                    && name != "grafeo_free_database"
            })
            .collect();
        assert_eq!(
            unlisted,
            Vec::<String>::new(),
            "exports without a row in this table"
        );
    }
    assert_eq!(not_closed(rows), Vec::<String>::new());

    assert_eq!(
        grafeo_close(db),
        GrafeoStatus::Ok,
        "a second close does nothing"
    );
    grafeo_free_database(db);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The calls without an error return clear the last error when they start: called right
/// after a failing call on an open database, they leave no error behind.
#[test]
fn calls_without_an_error_return_clear_the_last_error() {
    let (db, dir) = open_database("neutral-calls");
    let query = text("MATCH (p:Person) RETURN p.name");
    let stream = grafeo_stream_open(db, query.as_ptr());
    assert!(!stream.is_null(), "{}", last_error());
    let view = grafeo_as_of(db, 0, i64::try_from(grafeo_current_epoch(db)).unwrap());
    assert!(!view.is_null(), "{}", last_error());
    let broken = text("MATCH (");
    let name = text("name");

    let after_a_failure = |call: &dyn Fn()| {
        assert!(grafeo_execute(db, broken.as_ptr()).is_null());
        assert!(!last_error().is_empty(), "the failing call set an error");
        call();
        last_error()
    };
    let mut left: Vec<(&str, String)> = vec![
        (
            "grafeo_current_epoch",
            after_a_failure(&|| {
                let _ = grafeo_current_epoch(db);
            }),
        ),
        (
            "grafeo_current_schema",
            after_a_failure(&|| {
                let _ = grafeo_current_schema(db);
            }),
        ),
        (
            "grafeo_as_of_epoch",
            after_a_failure(&|| {
                let _ = grafeo_as_of_epoch(view);
            }),
        ),
        (
            "grafeo_has_property_index",
            after_a_failure(&|| {
                let _ = grafeo_has_property_index(db, name.as_ptr());
            }),
        ),
        (
            "grafeo_stream_warning_count",
            after_a_failure(&|| {
                let _ = grafeo_stream_warning_count(stream);
            }),
        ),
        (
            "grafeo_stream_warning_code",
            after_a_failure(&|| {
                let _ = grafeo_stream_warning_code(stream, 0);
            }),
        ),
        (
            "grafeo_stream_warning_kind",
            after_a_failure(&|| {
                let _ = grafeo_stream_warning_kind(stream, 0);
            }),
        ),
        (
            "grafeo_stream_warning_message",
            after_a_failure(&|| {
                let _ = grafeo_stream_warning_message(stream, 0);
            }),
        ),
    ];
    #[cfg(feature = "cdc")]
    left.extend([
        (
            "grafeo_set_cdc_enabled",
            after_a_failure(&|| {
                crate::database::grafeo_set_cdc_enabled(db, false);
            }),
        ),
        (
            "grafeo_is_cdc_enabled",
            after_a_failure(&|| {
                let _ = crate::database::grafeo_is_cdc_enabled(db);
            }),
        ),
    ]);
    left.retain(|(_, error)| !error.is_empty());
    assert_eq!(
        left,
        Vec::<(&str, String)>::new(),
        "calls that left an error behind"
    );

    // After close each returns its neutral value with GRAFEO-T007 set.
    assert_eq!(grafeo_close(db), GrafeoStatus::Ok);
    let mut neutral = vec![
        row("grafeo_current_epoch", || grafeo_current_epoch(db) == 0),
        row("grafeo_current_schema", || {
            grafeo_current_schema(db).is_null()
        }),
        row("grafeo_as_of_epoch", || grafeo_as_of_epoch(view) == 0),
        row("grafeo_has_property_index", || {
            grafeo_has_property_index(db, name.as_ptr()) == 0
        }),
        row("grafeo_stream_warning_count", || {
            grafeo_stream_warning_count(stream) == 0
        }),
        row("grafeo_stream_warning_code", || {
            grafeo_stream_warning_code(stream, 0).is_null()
        }),
        row("grafeo_stream_warning_kind", || {
            grafeo_stream_warning_kind(stream, 0).is_null()
        }),
        row("grafeo_stream_warning_message", || {
            grafeo_stream_warning_message(stream, 0).is_null()
        }),
    ];
    #[cfg(feature = "cdc")]
    neutral.extend([
        row("grafeo_set_cdc_enabled", || {
            crate::database::grafeo_set_cdc_enabled(db, true);
            true
        }),
        row("grafeo_is_cdc_enabled", || {
            !crate::database::grafeo_is_cdc_enabled(db)
        }),
    ]);
    assert_eq!(not_closed(neutral), Vec::<String>::new());

    grafeo_free_as_of(view);
    grafeo_stream_free(stream);
    grafeo_free_database(db);
    let _ = std::fs::remove_dir_all(&dir);
}

/// What the database handed out before `grafeo_close` fails after it with `GRAFEO-T007`: every
/// export on a graph, session, transaction, stream or view handle (all of `grafeo.h`'s, but the
/// frees and `grafeo_session_close`, a no-op), each through its own error channel; the ones
/// without an error return give their neutral value with `GRAFEO-T007` set.
#[test]
fn what_the_database_handed_out_fails_after_close() {
    use crate::graph as g;
    use crate::session as s;
    use crate::time_travel as t;

    let (db, dir) = open_database("closed-handles");
    let query = text("MATCH (p:Person) RETURN p.name");
    let gql = text("gql");
    let create = text("CREATE GRAPH berlin");
    let created = grafeo_execute(db, create.as_ptr());
    assert!(!created.is_null(), "{}", last_error());
    crate::database::grafeo_free_result(created);
    let berlin = text("berlin");
    let graph = grafeo_graph(db, berlin.as_ptr());
    let session = grafeo_session_open(db, 0);
    let transaction = grafeo_begin_transaction(db);
    let stream = grafeo_stream_open(db, query.as_ptr());
    let view = grafeo_as_of(db, 0, i64::try_from(grafeo_current_epoch(db)).unwrap());
    for (name, handle_is_null) in [
        ("graph", graph.is_null()),
        ("session", session.is_null()),
        ("transaction", transaction.is_null()),
        ("stream", stream.is_null()),
        ("view", view.is_null()),
    ] {
        assert!(!handle_is_null, "{name}: {}", last_error());
    }
    let insert = text("INSERT (:Person {name: 'Mia'})");
    let inserted = grafeo_transaction_execute(transaction, insert.as_ptr());
    assert!(!inserted.is_null(), "{}", last_error());
    crate::database::grafeo_free_result(inserted);
    assert_eq!(grafeo_close(db), GrafeoStatus::Ok);

    let labels = text("[\"Person\"]");
    let key = text("name");
    let value = text("\"Berlin\"");
    let label = text("Director");
    let edge_type = text("KNOWS");
    let rows_json = text("[{\"id\": 1, \"name\": \"Jules\"}]");
    let edges_json = text("[{\"src\": 0, \"dst\": 1, \"type\": \"KNOWS\"}]");
    let mut node = std::ptr::null_mut();
    let mut edge = std::ptr::null_mut();
    let mut json: *mut c_char = std::ptr::null_mut();
    let mut ids: *mut u64 = std::ptr::null_mut();
    let mut neighbors: *mut u64 = std::ptr::null_mut();
    let mut count = 0usize;
    let null = std::ptr::null;
    let mut rows = vec![
        // The graph.
        row("grafeo_graph_execute", || {
            grafeo_graph_execute(graph, query.as_ptr()).is_null()
        }),
        row("grafeo_graph_execute_with_params", || {
            g::grafeo_graph_execute_with_params(graph, query.as_ptr(), null()).is_null()
        }),
        row("grafeo_graph_execute_language", || {
            g::grafeo_graph_execute_language(graph, gql.as_ptr(), query.as_ptr(), null()).is_null()
        }),
        row("grafeo_graph_create_node", || {
            g::grafeo_graph_create_node(graph, labels.as_ptr(), null()) == u64::MAX
        }),
        row("grafeo_graph_create_edge", || {
            g::grafeo_graph_create_edge(graph, 0, 1, edge_type.as_ptr(), null()) == u64::MAX
        }),
        row("grafeo_graph_get_node", || {
            status(g::grafeo_graph_get_node(graph, 0, &raw mut node))
        }),
        row("grafeo_graph_get_edge", || {
            status(g::grafeo_graph_get_edge(graph, 0, &raw mut edge))
        }),
        row("grafeo_graph_get_node_edges", || {
            status(g::grafeo_graph_get_node_edges(
                graph,
                0,
                0,
                null(),
                &raw mut neighbors,
                &raw mut ids,
                &raw mut count,
            ))
        }),
        row("grafeo_graph_set_node_property", || {
            status(g::grafeo_graph_set_node_property(
                graph,
                0,
                key.as_ptr(),
                value.as_ptr(),
            ))
        }),
        row("grafeo_graph_set_edge_property", || {
            status(g::grafeo_graph_set_edge_property(
                graph,
                0,
                key.as_ptr(),
                value.as_ptr(),
            ))
        }),
        row("grafeo_graph_remove_node_property", || {
            g::grafeo_graph_remove_node_property(graph, 0, key.as_ptr()) == -1
        }),
        row("grafeo_graph_remove_edge_property", || {
            g::grafeo_graph_remove_edge_property(graph, 0, key.as_ptr()) == -1
        }),
        row("grafeo_graph_add_node_label", || {
            g::grafeo_graph_add_node_label(graph, 0, label.as_ptr()) == -1
        }),
        row("grafeo_graph_remove_node_label", || {
            g::grafeo_graph_remove_node_label(graph, 0, label.as_ptr()) == -1
        }),
        row("grafeo_graph_delete_node", || {
            g::grafeo_graph_delete_node(graph, 0) == -1
        }),
        row("grafeo_graph_detach_delete_node", || {
            g::grafeo_graph_detach_delete_node(graph, 0) == -1
        }),
        row("grafeo_graph_delete_edge", || {
            g::grafeo_graph_delete_edge(graph, 0) == -1
        }),
        row("grafeo_graph_batch_create_nodes_with_props", || {
            status(g::grafeo_graph_batch_create_nodes_with_props(
                graph,
                labels.as_ptr(),
                rows_json.as_ptr(),
                &raw mut ids,
                &raw mut count,
            ))
        }),
        row("grafeo_graph_batch_create_edges", || {
            status(g::grafeo_graph_batch_create_edges(
                graph,
                edges_json.as_ptr(),
                &raw mut ids,
                &raw mut count,
            ))
        }),
        row("grafeo_graph_upsert_nodes", || {
            status(g::grafeo_graph_upsert_nodes(
                graph,
                labels.as_ptr(),
                rows_json.as_ptr(),
                null(),
                &raw mut json,
            ))
        }),
        row("grafeo_graph_upsert_edges", || {
            status(g::grafeo_graph_upsert_edges(
                graph,
                edge_type.as_ptr(),
                edges_json.as_ptr(),
                null(),
                &raw mut json,
            ))
        }),
        // The session.
        row("grafeo_session_execute", || {
            grafeo_session_execute(session, query.as_ptr()).is_null()
        }),
        row("grafeo_session_execute_with_params", || {
            s::grafeo_session_execute_with_params(session, query.as_ptr(), null()).is_null()
        }),
        row("grafeo_session_execute_language", || {
            s::grafeo_session_execute_language(session, gql.as_ptr(), query.as_ptr(), null())
                .is_null()
        }),
        row("grafeo_session_begin_transaction", || {
            s::grafeo_session_begin_transaction(session).is_null()
        }),
        // The transaction.
        row("grafeo_transaction_execute", || {
            grafeo_transaction_execute(transaction, query.as_ptr()).is_null()
        }),
        row("grafeo_transaction_execute_with_params", || {
            crate::database::grafeo_transaction_execute_with_params(
                transaction,
                query.as_ptr(),
                null(),
            )
            .is_null()
        }),
        row("grafeo_transaction_execute_language", || {
            crate::database::grafeo_transaction_execute_language(
                transaction,
                gql.as_ptr(),
                query.as_ptr(),
                null(),
            )
            .is_null()
        }),
        // The rollback first: after a commit that failed, a rollback has nothing to do and
        // answers, as it does on an open database.
        row("grafeo_rollback", || {
            status(crate::database::grafeo_rollback(transaction))
        }),
        row("grafeo_commit", || status(grafeo_commit(transaction))),
        // The stream (the warning accessors have no error return: neutral values).
        row("grafeo_stream_next_row_json", || {
            status(grafeo_stream_next_row_json(stream, &raw mut json))
        }),
        row("grafeo_stream_columns_json", || {
            crate::stream::grafeo_stream_columns_json(stream).is_null()
        }),
        row("grafeo_stream_warning_count", || {
            grafeo_stream_warning_count(stream) == 0
        }),
        row("grafeo_stream_warning_code", || {
            grafeo_stream_warning_code(stream, 0).is_null()
        }),
        row("grafeo_stream_warning_kind", || {
            grafeo_stream_warning_kind(stream, 0).is_null()
        }),
        row("grafeo_stream_warning_message", || {
            grafeo_stream_warning_message(stream, 0).is_null()
        }),
        // The view (its epoch has no error return: 0).
        row("grafeo_as_of_epoch", || grafeo_as_of_epoch(view) == 0),
        row("grafeo_as_of_execute", || {
            grafeo_as_of_execute(view, query.as_ptr(), null()).is_null()
        }),
        row("grafeo_as_of_get_node", || {
            status(t::grafeo_as_of_get_node(view, 0, &raw mut node))
        }),
        row("grafeo_as_of_get_edge", || {
            status(t::grafeo_as_of_get_edge(view, 0, &raw mut edge))
        }),
        row("grafeo_as_of_get_node_property", || {
            status(t::grafeo_as_of_get_node_property(
                view,
                0,
                key.as_ptr(),
                &raw mut json,
            ))
        }),
    ];

    #[cfg(feature = "gql")]
    rows.push(row("grafeo_session_stream_open", || {
        s::grafeo_session_stream_open(session, query.as_ptr()).is_null()
    }));

    // The table covers every export on these handles but the frees and the session's close.
    let listed: Vec<&str> = rows.iter().map(|(name, _, _)| *name).collect();
    let unlisted: Vec<String> = [
        "GrafeoGraph",
        "GrafeoSession",
        "GrafeoTransaction",
        "GrafeoStream",
        "GrafeoAsOf",
    ]
    .into_iter()
    .flat_map(exports_on)
    .filter(|name| {
        !listed.contains(&name.as_str())
            && !name.starts_with("grafeo_free_")
            && name != "grafeo_stream_free"
            && name != "grafeo_session_close"
            // A build without GQL has no session streams.
            && (cfg!(feature = "gql") || name != "grafeo_session_stream_open")
    })
    .collect();
    assert_eq!(
        unlisted,
        Vec::<String>::new(),
        "exports without a row in this table"
    );
    assert_eq!(not_closed(rows), Vec::<String>::new());
    assert_eq!(
        s::grafeo_session_close(session),
        GrafeoStatus::Ok,
        "a session's close after the database's is a no-op"
    );

    grafeo_free_as_of(view);
    grafeo_stream_free(stream);
    grafeo_free_transaction(transaction);
    grafeo_free_session(session);
    grafeo_free_graph(graph);
    grafeo_free_database(db);

    // The transaction open across close() left nothing in the file.
    let path = text(dir.join("amsterdam.grafeo").to_str().unwrap());
    let reopened = grafeo_open(path.as_ptr());
    assert!(!reopened.is_null(), "{}", last_error());
    let mut count = 0usize;
    assert_eq!(
        grafeo_node_count(reopened, &raw mut count),
        GrafeoStatus::Ok
    );
    assert_eq!(count, 2);
    grafeo_free_database(reopened);
    let _ = std::fs::remove_dir_all(&dir);
}
