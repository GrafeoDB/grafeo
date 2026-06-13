//! Integration tests for `execute_streaming()` / `OwnedResultStream`.
//!
//! Focus of this first pass:
//! - Equivalence: a streamed result collected back matches the materialized one.
//! - Early drop does not leak (counter returns to 0, subsequent commit is fine).
//! - Non-streamable queries (mutations, EXPLAIN, ORDER BY, aggregate, session
//!   commands) are rejected with a clear error.

use grafeo_common::types::{LogicalType, Value};
use grafeo_engine::GrafeoDB;

/// Seeds a small Person/KNOWS graph. Test data names follow the repo
/// convention (Alix, Gus, Tarantino characters) from CODE_STYLE.md.
fn seed_people(db: &GrafeoDB) {
    let alix = db.create_node(&["Person"]).unwrap();
    let gus = db.create_node(&["Person"]).unwrap();
    let vincent = db.create_node(&["Person"]).unwrap();
    let jules = db.create_node(&["Person"]).unwrap();
    let mia = db.create_node(&["Person"]).unwrap();

    db.set_node_property(alix, "name", Value::String("Alix".into()))
        .unwrap();
    db.set_node_property(alix, "age", Value::Int64(32)).unwrap();
    db.set_node_property(gus, "name", Value::String("Gus".into()))
        .unwrap();
    db.set_node_property(gus, "age", Value::Int64(28)).unwrap();
    db.set_node_property(vincent, "name", Value::String("Vincent".into()))
        .unwrap();
    db.set_node_property(vincent, "age", Value::Int64(45))
        .unwrap();
    db.set_node_property(jules, "name", Value::String("Jules".into()))
        .unwrap();
    db.set_node_property(jules, "age", Value::Int64(40))
        .unwrap();
    db.set_node_property(mia, "name", Value::String("Mia".into()))
        .unwrap();
    db.set_node_property(mia, "age", Value::Int64(24)).unwrap();

    db.create_edge(alix, gus, "KNOWS");
    db.create_edge(vincent, jules, "KNOWS");
    db.create_edge(jules, mia, "KNOWS");
}

#[test]
fn streaming_matches_materialized_scan() {
    let db = GrafeoDB::new_in_memory();
    seed_people(&db);

    let query = "MATCH (p:Person) RETURN p.name AS name, p.age AS age";

    let materialized = db.execute(query).expect("execute");
    let streamed = db
        .execute_streaming(query)
        .expect("execute_streaming")
        .collect()
        .expect("collect");

    // Streaming does not guarantee a specific row order vs materialized in the
    // absence of ORDER BY, but it must yield the same multiset of rows.
    let mut mat_sorted = materialized.rows().to_vec();
    let mut str_sorted = streamed.rows().to_vec();
    mat_sorted.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
    str_sorted.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));

    assert_eq!(
        mat_sorted, str_sorted,
        "streaming must produce the same rows as materialized execution"
    );
    assert_eq!(
        streamed.columns,
        vec!["name".to_string(), "age".to_string()]
    );
}

#[test]
fn streaming_matches_materialized_filter() {
    let db = GrafeoDB::new_in_memory();
    seed_people(&db);

    let query = "MATCH (p:Person) WHERE p.age > 30 RETURN p.name AS name";

    let materialized = db.execute(query).expect("execute");
    let streamed = db
        .execute_streaming(query)
        .expect("execute_streaming")
        .collect()
        .expect("collect");

    assert_eq!(materialized.rows().len(), streamed.rows().len());

    let mut mat_sorted = materialized.rows().to_vec();
    let mut str_sorted = streamed.rows().to_vec();
    mat_sorted.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
    str_sorted.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
    assert_eq!(mat_sorted, str_sorted);
}

#[test]
fn streaming_row_iter_yields_every_row() {
    let db = GrafeoDB::new_in_memory();
    seed_people(&db);

    let stream = db
        .execute_streaming("MATCH (p:Person) RETURN p.name")
        .expect("execute_streaming");
    let cols = stream.columns().to_vec();
    assert_eq!(cols, vec!["p.name".to_string()]);

    let rows: Vec<_> = stream
        .into_row_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("row iter");
    assert_eq!(rows.len(), 5);
}

#[test]
fn streaming_early_drop_releases_counter() {
    let db = GrafeoDB::new_in_memory();
    seed_people(&db);

    // Build a stream, pull one chunk, then drop the stream without exhausting.
    {
        let mut stream = db
            .execute_streaming("MATCH (p:Person) RETURN p.name")
            .expect("execute_streaming");
        let _first = stream.next_chunk().expect("first chunk");
        // stream drops here mid-iteration
    }

    // A subsequent full execute must still work and return all five rows.
    let result = db
        .execute("MATCH (p:Person) RETURN p.name")
        .expect("execute");
    assert_eq!(result.rows().len(), 5);
}

#[test]
fn streaming_rejects_mutation() {
    let db = GrafeoDB::new_in_memory();
    seed_people(&db);

    let err = db
        .execute_streaming("INSERT (:Person {name: 'Butch'})")
        .expect_err("should reject mutations");
    let msg = err.to_string();
    assert!(
        msg.to_lowercase().contains("mutat")
            || msg.to_lowercase().contains("execute() instead")
            || msg.to_lowercase().contains("cannot be streamed"),
        "expected rejection message, got: {msg}"
    );
}

#[test]
fn streaming_rejects_order_by() {
    let db = GrafeoDB::new_in_memory();
    seed_people(&db);

    // ORDER BY compiles to a push-based pipeline (Sort is a pipeline breaker).
    let err = db
        .execute_streaming("MATCH (p:Person) RETURN p.name AS n ORDER BY n")
        .expect_err("should reject push pipelines");
    assert!(
        err.to_string().to_lowercase().contains("push")
            || err
                .to_string()
                .to_lowercase()
                .contains("cannot be streamed"),
        "expected push-pipeline rejection, got: {err}"
    );
}

#[test]
fn streaming_rejects_session_command() {
    let db = GrafeoDB::new_in_memory();
    let err = db
        .execute_streaming("SESSION SET GRAPH analytics")
        .expect_err("should reject session commands");
    assert!(err.to_string().to_lowercase().contains("session"));
}

#[test]
fn streaming_rejects_explain() {
    let db = GrafeoDB::new_in_memory();
    seed_people(&db);
    let err = db
        .execute_streaming("EXPLAIN MATCH (p:Person) RETURN p.name")
        .expect_err("should reject EXPLAIN");
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("explain") || msg.contains("cannot be streamed"),
        "expected EXPLAIN rejection, got: {msg}"
    );
}

#[test]
fn streaming_empty_result_yields_no_rows() {
    let db = GrafeoDB::new_in_memory();
    seed_people(&db);

    let rows: Vec<_> = db
        .execute_streaming("MATCH (p:Person) WHERE p.age > 999 RETURN p.name")
        .expect("execute_streaming")
        .into_row_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("row iter");
    assert!(rows.is_empty());
}

// -------- Session-scoped streaming (ResultStream<'s> / RowIterator<'s>) ----

#[test]
fn session_streaming_yields_expected_rows() {
    let db = GrafeoDB::new_in_memory();
    seed_people(&db);
    let session = db.session();

    let stream = session
        .execute_streaming("MATCH (p:Person) RETURN p.name")
        .expect("session execute_streaming");
    assert_eq!(stream.columns(), &["p.name".to_string()]);

    let rows: Vec<_> = stream
        .into_row_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("rows");
    assert_eq!(rows.len(), 5);
}

#[test]
fn session_streaming_row_iterator_exposes_columns() {
    let db = GrafeoDB::new_in_memory();
    seed_people(&db);
    let session = db.session();

    let iter = session
        .execute_streaming("MATCH (p:Person) RETURN p.name AS n, p.age AS a")
        .expect("execute_streaming")
        .into_row_iter();
    assert_eq!(iter.columns(), &["n".to_string(), "a".to_string()]);
}

#[test]
fn session_streaming_next_chunk_then_exhaustion_returns_none() {
    let db = GrafeoDB::new_in_memory();
    seed_people(&db);
    let session = db.session();

    let mut stream = session
        .execute_streaming("MATCH (p:Person) RETURN p.name")
        .expect("execute_streaming");

    let first = stream.next_chunk().expect("first chunk");
    assert!(first.is_some(), "first chunk should yield rows");

    while stream.next_chunk().expect("chunk").is_some() {}
    // Exhaustion is idempotent: further next_chunk calls stay at None.
    assert!(stream.next_chunk().expect("post-exhaustion").is_none());
}

#[test]
fn session_streaming_collect_matches_execute() {
    let db = GrafeoDB::new_in_memory();
    seed_people(&db);
    let session = db.session();

    let materialized = session
        .execute("MATCH (p:Person) RETURN p.name")
        .expect("execute");
    let streamed = session
        .execute_streaming("MATCH (p:Person) RETURN p.name")
        .expect("execute_streaming")
        .collect()
        .expect("collect");
    assert_eq!(materialized.rows().len(), streamed.rows().len());
}

#[test]
fn session_streaming_rejects_schema_command() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    // ResultStream doesn't impl Debug, so we can't use expect_err here.
    let Err(err) = session.execute_streaming("CREATE GRAPH analytics") else {
        panic!("schema DDL must not be streamable");
    };
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("schema") || msg.contains("cannot be streamed"),
        "expected schema DDL rejection, got: {msg}"
    );
}

#[test]
fn session_streaming_second_call_hits_plan_cache() {
    let db = GrafeoDB::new_in_memory();
    seed_people(&db);
    let session = db.session();

    let query = "MATCH (p:Person) WHERE p.age > 30 RETURN p.name";
    let first: Vec<_> = session
        .execute_streaming(query)
        .expect("first")
        .into_row_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("first rows");
    // Second call goes through the cache-hit branch in build_streaming_plan.
    let second: Vec<_> = session
        .execute_streaming(query)
        .expect("second")
        .into_row_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("second rows");
    assert_eq!(first.len(), second.len());
}

// -------- OwnedResultStream / OwnedRowIterator specifics ------------------

#[test]
fn owned_stream_column_types_start_as_any() {
    let db = GrafeoDB::new_in_memory();
    seed_people(&db);
    let stream = db
        .execute_streaming("MATCH (p:Person) RETURN p.name AS name, p.age AS age")
        .expect("execute_streaming");
    let types = stream.column_types();
    assert_eq!(types.len(), 2);
    assert!(types.iter().all(|t| matches!(t, LogicalType::Any)));
}

#[test]
fn owned_stream_column_types_refine_after_first_chunk() {
    // refine_column_types runs after every non-empty chunk. Property accesses
    // like `p.name` stay as `Any` because GQL values are dynamically typed per
    // row, so we only assert that the slot count stays consistent and the
    // refinement path is actually exercised.
    let db = GrafeoDB::new_in_memory();
    seed_people(&db);
    let mut stream = db
        .execute_streaming("MATCH (p:Person) RETURN p.name AS name, p.age AS age")
        .expect("execute_streaming");
    let _ = stream.next_chunk().expect("chunk");
    assert_eq!(stream.column_types().len(), 2);
}

#[test]
fn owned_stream_debug_lists_columns() {
    let db = GrafeoDB::new_in_memory();
    seed_people(&db);
    let stream = db
        .execute_streaming("MATCH (p:Person) RETURN p.name AS name")
        .expect("execute_streaming");
    let dbg = format!("{stream:?}");
    assert!(dbg.contains("OwnedResultStream"));
    assert!(dbg.contains("name"));
}

#[test]
fn owned_row_iterator_exposes_columns() {
    let db = GrafeoDB::new_in_memory();
    seed_people(&db);
    let iter = db
        .execute_streaming("MATCH (p:Person) RETURN p.name AS n")
        .expect("execute_streaming")
        .into_row_iter();
    assert_eq!(iter.columns(), &["n".to_string()]);
}

#[test]
fn owned_stream_next_chunk_exhaustion_is_idempotent() {
    let db = GrafeoDB::new_in_memory();
    seed_people(&db);
    let mut stream = db
        .execute_streaming("MATCH (p:Person) RETURN p.name")
        .expect("execute_streaming");
    while stream.next_chunk().expect("chunk").is_some() {}
    assert!(stream.next_chunk().expect("after exhaustion").is_none());
    assert!(
        stream
            .next_chunk()
            .expect("after exhaustion again")
            .is_none()
    );
}
