//! A query vector reaches a vector index only when the index can measure it
//! (#593): one of another size, or with a NaN or an infinite value, is an
//! error that names the index and both sizes (or the value), from every
//! entry point, where it used to panic (a crash) or return NaN distances.
//! The database answers as before afterwards. Writes and index creation
//! check their vectors the same way.

#![cfg(all(feature = "vector-index", feature = "lpg", feature = "gql"))]

use std::collections::HashMap;

use grafeo_common::types::{PropertyKey, Value};
use grafeo_common::utils::error::{Error, ErrorCode};
use grafeo_engine::GrafeoDB;

/// Four documents with 3-value embeddings and a cosine index on `:Doc(emb)`,
/// and a text index on `:Doc(text)` when text search is built in.
fn documents() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    for (vector, text) in [
        ([1.0, 0.0, 0.0], "graph notes from Amsterdam"),
        ([0.0, 1.0, 0.0], "graph notes from Berlin"),
        ([0.0, 0.0, 1.0], "graph notes from Paris"),
        ([0.5, 0.5, 0.0], "graph notes from Prague"),
    ] {
        let mut properties = HashMap::new();
        properties.insert(
            PropertyKey::new("emb"),
            Value::Vector(vector.to_vec().into()),
        );
        properties.insert(PropertyKey::new("text"), Value::from(text));
        db.create_node_with_props(&["Doc"], properties).unwrap();
    }
    db.create_vector_index("Doc", "emb", Some(3), Some("cosine"), None, None, None)
        .unwrap();
    #[cfg(feature = "text-index")]
    db.create_text_index("Doc", "text").unwrap();
    db
}

/// The query vectors of another size, with the error each one gives.
fn wrong_sizes() -> Vec<(Vec<f32>, &'static str)> {
    vec![
        (
            vec![0.3, 0.19],
            "the query vector has 2 dimensions; the index on :Doc(emb) expects 3",
        ),
        (
            vec![0.3, 0.19, 0.88, 0.3],
            "the query vector has 4 dimensions; the index on :Doc(emb) expects 3",
        ),
        (
            Vec::new(),
            "the query vector has 0 dimensions; the index on :Doc(emb) expects 3",
        ),
    ]
}

/// The query vectors with a value no distance can be measured to, with the
/// error each one gives.
fn non_finite() -> Vec<(Vec<f32>, &'static str)> {
    vec![
        (
            vec![0.9, f32::NAN, 0.0],
            "the query vector has NaN at position 1",
        ),
        (
            vec![0.9, f32::INFINITY, 0.0],
            "the query vector has inf at position 1",
        ),
        (
            vec![f32::NEG_INFINITY, 0.0, 0.0],
            "the query vector has -inf at position 0",
        ),
    ]
}

/// Asserts that `result` is the invalid-value error with `message`.
fn assert_refused<T: std::fmt::Debug>(result: Result<T, Error>, message: &str, what: &str) {
    let err = result.expect_err(&format!("{what}: expected an error ({message})"));
    assert_eq!(
        err.error_code(),
        ErrorCode::InvalidInput,
        "{what}: a user mistake, not an internal error: {err}"
    );
    assert!(err.to_string().contains(message), "{what}: {err}");
}

/// The database still answers a search with a query vector it can measure.
fn assert_still_answers(db: &GrafeoDB) {
    let hits = db
        .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 1, None, None)
        .expect("a search after a refused one");
    assert_eq!(hits.len(), 1);
    assert!(hits[0].1.abs() < 1e-6, "{hits:?}");
}

#[test]
fn every_search_call_refuses_a_query_vector_the_index_cannot_measure() {
    let db = documents();
    for (query, message) in wrong_sizes().into_iter().chain(non_finite()) {
        assert_refused(
            db.vector_search("Doc", "emb", &query, 2, None, None),
            message,
            "vector_search",
        );
        assert_refused(
            db.vector_search("Doc", "emb", &query, 2, Some(64), None),
            message,
            "vector_search with ef",
        );
        let mut filters = HashMap::new();
        filters.insert("text".to_string(), Value::from("graph notes from Paris"));
        assert_refused(
            db.vector_search("Doc", "emb", &query, 2, None, Some(&filters)),
            message,
            "vector_search with filters",
        );
        // One query of the batch is enough to refuse the batch.
        assert_refused(
            db.batch_vector_search(
                "Doc",
                "emb",
                &[vec![1.0, 0.0, 0.0], query.clone()],
                2,
                None,
                None,
            ),
            message,
            "batch_vector_search",
        );
        assert_refused(
            db.mmr_search("Doc", "emb", &query, 2, None, None, None, None),
            message,
            "mmr_search",
        );
        #[cfg(feature = "hybrid-search")]
        assert_refused(
            db.hybrid_search("Doc", "text", "emb", "graph", Some(&query), 2, None, None),
            message,
            "hybrid_search",
        );
        assert_still_answers(&db);
    }
}

#[test]
fn search_procedures_refuse_a_query_vector_of_another_size() {
    let db = documents();
    for (literal, message) in [
        (
            "[0.3, 0.19]",
            "the query vector has 2 dimensions; the index on :Doc(emb) expects 3",
        ),
        (
            "[0.3, 0.19, 0.88, 0.3]",
            "the query vector has 4 dimensions; the index on :Doc(emb) expects 3",
        ),
    ] {
        for query in [
            format!("CALL grafeo.search.vector('Doc', 'emb', {literal}, 2)"),
            format!("CALL grafeo.search.mmr('Doc', 'emb', {literal}, 2, 3, 0.5)"),
        ] {
            assert_refused(db.execute(&query), message, &query);
            #[cfg(feature = "cypher")]
            {
                let cypher = format!("{query} YIELD node_id RETURN node_id");
                assert_refused(db.execute_cypher(&cypher), message, &cypher);
            }
        }
        assert_still_answers(&db);
    }
}

/// A vector predicate in `WHERE` that a search of the index answers checks
/// its query vector against the index: in a literal or a parameter, and also
/// when the predicate's metric is not the index's, so that the search scans
/// the index's vectors instead. (A predicate evaluated per node, without an
/// index, gives NULL for vectors of different sizes.)
#[test]
fn vector_predicates_on_an_indexed_property_refuse_what_the_index_cannot_measure() {
    let db = documents();
    let queries = [
        "MATCH (d:Doc) WHERE cosine_similarity(d.emb, $q) > 0.1 RETURN d.text",
        "MATCH (d:Doc) WHERE euclidean_distance(d.emb, $q) < 3.0 RETURN d.text",
    ];
    for (vector, message) in wrong_sizes().into_iter().chain(non_finite()) {
        for query in queries {
            let mut params = HashMap::new();
            params.insert("q".to_string(), Value::Vector(vector.clone().into()));
            assert_refused(
                db.execute_with_params(query, params.clone()),
                message,
                query,
            );
            #[cfg(feature = "cypher")]
            assert_refused(
                db.execute_cypher_with_params(query, params),
                message,
                &format!("Cypher {query}"),
            );
        }
        assert_still_answers(&db);
    }
    assert_refused(
        db.execute("MATCH (d:Doc) WHERE cosine_similarity(d.emb, [0.3, 0.19]) > 0.1 RETURN d"),
        "the query vector has 2 dimensions; the index on :Doc(emb) expects 3",
        "a literal query vector",
    );
}

/// An indexed property takes only vectors the index can measure, from
/// every write path; a property without an index stores any vector.
#[test]
fn an_indexed_property_refuses_a_vector_with_nan_or_infinity() {
    let db = documents();
    for (vector, position, value) in [
        (vec![0.9, f32::NAN, 0.0], 1, "NaN"),
        (vec![f32::INFINITY, 0.0, 0.0], 0, "inf"),
    ] {
        let message = format!(
            "property 'emb' on :Doc has a vector index, which cannot measure {value} \
             (at position {position})"
        );
        let mut properties = HashMap::new();
        properties.insert(
            PropertyKey::new("emb"),
            Value::Vector(vector.clone().into()),
        );
        assert_refused(
            db.create_node_with_props(&["Doc"], properties.clone()),
            &message,
            "create_node_with_props",
        );
        assert_refused(
            db.batch_create_nodes("Doc", "emb", vec![vector.clone()]),
            &message,
            "batch_create_nodes",
        );
        let existing = db.create_node(&["Doc"]).unwrap();
        assert_refused(
            db.set_node_property(existing, "emb", Value::Vector(vector.clone().into())),
            &message,
            "set_node_property",
        );
        db.delete_node(existing).unwrap();
        let mut params = HashMap::new();
        params.insert("v".to_string(), Value::Vector(vector.clone().into()));
        assert_refused(
            db.execute_with_params("INSERT (:Doc {emb: $v})", params.clone()),
            &message,
            "INSERT",
        );
        // Without an index, the vector is a value like any other.
        db.execute_with_params("INSERT (:Draft {emb: $v})", params)
            .expect("an unindexed property stores any vector");
    }
    let hits = db
        .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 19, None, None)
        .unwrap();
    assert_eq!(hits.len(), 4, "only the four documents: {hits:?}");
    assert!(
        hits.iter().all(|(_, distance)| distance.is_finite()),
        "{hits:?}"
    );
}

/// `create_vector_index` reports a mistake in its arguments, or a vector the
/// index could not measure, as an invalid value naming what is wrong, not as
/// an internal error.
#[test]
fn creating_a_vector_index_reports_invalid_arguments() {
    let db = documents();
    assert_refused(
        db.create_vector_index("Doc", "other", Some(0), None, None, None, None),
        "a vector index needs at least 1 dimension",
        "dimensions 0",
    );
    assert_refused(
        db.create_vector_index("Doc", "other", Some(3), Some("hamming"), None, None, None),
        "Unknown distance metric 'hamming'. Use: cosine, euclidean, dot_product, manhattan",
        "an unknown metric",
    );
    assert_refused(
        db.create_vector_index("Doc", "emb", Some(4), None, None, None, None),
        "expected 4, found 3",
        "vectors of another size",
    );
    let draft = db.create_node(&["Draft"]).unwrap();
    db.set_node_property(
        draft,
        "emb",
        Value::Vector(vec![0.3, f32::NAN, 0.88].into()),
    )
    .unwrap();
    assert_refused(
        db.create_vector_index("Draft", "emb", None, None, None, None, None),
        "has NaN at position 1",
        "a vector with NaN",
    );
    // The same checks for CREATE VECTOR INDEX.
    for (statement, message) in [
        (
            "CREATE VECTOR INDEX zero ON :Doc(other) DIMENSION 0",
            "a vector index needs at least 1 dimension",
        ),
        (
            "CREATE VECTOR INDEX hamming ON :Doc(other) DIMENSION 3 METRIC 'hamming'",
            "Unknown distance metric 'hamming'",
        ),
        (
            "CREATE VECTOR INDEX wider ON :Doc(emb) DIMENSION 4",
            "expected 4, found 3",
        ),
        (
            "CREATE VECTOR INDEX drafts ON :Draft(emb)",
            "has NaN at position 1",
        ),
    ] {
        assert_refused(db.execute(statement), message, statement);
    }
    assert_still_answers(&db);
}
