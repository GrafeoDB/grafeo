//! Integration tests for `CALL grafeo.search.vector|mmr|text` procedures.
//!
//! Verifies that the procedure registry entries introduced in 0.5.41 reach
//! the HNSW and BM25 indexes owned by the LPG store and return the same
//! rows as the direct `db.vector_search` / `db.mmr_search` / `db.text_search`
//! APIs.

#![cfg(all(feature = "algos", feature = "vector-index", feature = "lpg"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

fn vec3(x: f32, y: f32, z: f32) -> Value {
    Value::Vector(vec![x, y, z].into())
}

fn setup_vector_graph() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let n1 = db.create_node(&["Doc"]).unwrap();
    db.set_node_property(n1, "emb", vec3(1.0, 0.0, 0.0))
        .unwrap();
    db.set_node_property(n1, "title", Value::from("A")).unwrap();
    let n2 = db.create_node(&["Doc"]).unwrap();
    db.set_node_property(n2, "emb", vec3(0.9, 0.1, 0.0))
        .unwrap();
    db.set_node_property(n2, "title", Value::from("B")).unwrap();
    let n3 = db.create_node(&["Doc"]).unwrap();
    db.set_node_property(n3, "emb", vec3(0.0, 1.0, 0.0))
        .unwrap();
    db.set_node_property(n3, "title", Value::from("C")).unwrap();

    db.create_vector_index("Doc", "emb", Some(3), Some("cosine"), None, None, None)
        .expect("create vector index");
    db
}

#[test]
fn test_call_search_vector_returns_rows() {
    let db = setup_vector_graph();
    let session = db.session();
    let result = session
        .execute("CALL grafeo.search.vector('Doc', 'emb', [1.0, 0.0, 0.0], 2)")
        .expect("CALL grafeo.search.vector should succeed");

    assert_eq!(
        result.columns,
        vec!["node_id".to_string(), "distance".to_string()]
    );
    assert_eq!(result.row_count(), 2);

    for row in result.rows() {
        assert!(matches!(row[0], Value::Int64(_)));
        assert!(matches!(row[1], Value::Float64(_)));
    }
}

#[test]
fn test_call_search_vector_matches_direct_api() {
    let db = setup_vector_graph();
    let direct = db
        .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 3, None, None)
        .expect("direct vector_search");

    let session = db.session();
    let result = session
        .execute("CALL grafeo.search.vector('Doc', 'emb', [1.0, 0.0, 0.0], 3)")
        .unwrap();

    assert_eq!(result.row_count(), direct.len());
    for (row, (node_id, distance)) in result.rows().iter().zip(direct.iter()) {
        let Value::Int64(proc_node_id) = row[0] else {
            panic!("expected Int64 node_id, got {:?}", row[0]);
        };
        let Value::Float64(proc_distance) = row[1] else {
            panic!("expected Float64 distance, got {:?}", row[1]);
        };
        assert_eq!(proc_node_id.cast_unsigned(), node_id.as_u64());
        assert!((proc_distance - f64::from(*distance)).abs() < 1e-6);
    }
}

#[test]
fn test_call_search_vector_with_yield_and_limit() {
    let db = setup_vector_graph();
    let session = db.session();
    let result = session
        .execute(
            "CALL grafeo.search.vector('Doc', 'emb', [1.0, 0.0, 0.0], 3) \
             YIELD node_id, distance \
             RETURN node_id, distance \
             ORDER BY distance ASC \
             LIMIT 2",
        )
        .unwrap();

    assert_eq!(result.row_count(), 2);
    let first_distance = match &result.rows()[0][1] {
        Value::Float64(f) => *f,
        _ => panic!("expected Float64 distance"),
    };
    let second_distance = match &result.rows()[1][1] {
        Value::Float64(f) => *f,
        _ => panic!("expected Float64 distance"),
    };
    assert!(first_distance <= second_distance);
}

#[test]
fn test_call_search_vector_missing_index_errors() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let err = session
        .execute("CALL grafeo.search.vector('Missing', 'emb', [1.0, 0.0, 0.0], 3)")
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("No vector index") || msg.contains("vector index"),
        "expected missing-index error, got: {msg}"
    );
}

#[test]
fn test_call_search_mmr_returns_rows() {
    let db = setup_vector_graph();
    let session = db.session();
    let result = session
        .execute("CALL grafeo.search.mmr('Doc', 'emb', [1.0, 0.0, 0.0], 2, 3, 0.5)")
        .expect("CALL grafeo.search.mmr should succeed");

    assert_eq!(
        result.columns,
        vec!["node_id".to_string(), "distance".to_string()]
    );
    assert!(result.row_count() <= 2);
}

#[test]
fn test_call_search_mmr_default_fetch_k_and_lambda() {
    let db = setup_vector_graph();
    let session = db.session();
    let result = session
        .execute("CALL grafeo.search.mmr('Doc', 'emb', [1.0, 0.0, 0.0], 2)")
        .expect("CALL grafeo.search.mmr with defaults should succeed");
    assert!(result.row_count() > 0);
}

#[test]
fn test_grafeo_procedures_list_includes_search_vector() {
    let db = setup_vector_graph();
    let session = db.session();
    let result = session.execute("CALL grafeo.procedures()").unwrap();

    let names: Vec<String> = result
        .rows()
        .iter()
        .filter_map(|row| match &row[0] {
            Value::String(s) => Some(s.to_string()),
            _ => None,
        })
        .collect();
    assert!(
        names.iter().any(|n| n == "grafeo.search.vector"),
        "grafeo.search.vector missing from registry. Registered: {names:?}"
    );
    assert!(
        names.iter().any(|n| n == "grafeo.search.mmr"),
        "grafeo.search.mmr missing from registry"
    );
}

#[cfg(feature = "text-index")]
mod text {
    use super::*;

    fn setup_text_graph() -> GrafeoDB {
        let db = GrafeoDB::new_in_memory();
        let n1 = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(n1, "body", Value::from("graph databases are great"))
            .unwrap();
        let n2 = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(n2, "body", Value::from("vector search for retrieval"))
            .unwrap();
        let n3 = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(n3, "body", Value::from("graph neural networks for nlp"))
            .unwrap();

        db.create_text_index("Doc", "body")
            .expect("create text index");
        db
    }

    #[test]
    fn test_call_search_text_returns_rows() {
        let db = setup_text_graph();
        let session = db.session();
        let result = session
            .execute("CALL grafeo.search.text('Doc', 'body', 'graph', 5)")
            .expect("CALL grafeo.search.text should succeed");

        assert_eq!(
            result.columns,
            vec!["node_id".to_string(), "score".to_string()]
        );
        assert!(
            result.row_count() >= 2,
            "both graph-mentioning docs should match"
        );
    }

    #[test]
    fn test_call_search_text_matches_direct_api() {
        let db = setup_text_graph();
        let direct = db
            .text_search("Doc", "body", "graph", 5)
            .expect("direct text_search");

        let session = db.session();
        let result = session
            .execute("CALL grafeo.search.text('Doc', 'body', 'graph', 5)")
            .unwrap();

        assert_eq!(result.row_count(), direct.len());
    }
}
