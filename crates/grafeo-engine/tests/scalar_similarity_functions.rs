//! Integration tests for GQL scalar similarity functions (Piece 2a).
//!
//! Verifies that `cosine_similarity`, `cosine_distance`, `euclidean_distance`,
//! `manhattan_distance`, and `dot_product` are all reachable from WHERE /
//! ORDER BY / projection contexts and produce correct values.

#![cfg(all(feature = "lpg", feature = "gql"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::QueryResult;

fn setup() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["Doc"]).unwrap();
    db.set_node_property(a, "name", Value::from("A")).unwrap();
    db.set_node_property(a, "emb", Value::Vector(vec![1.0f32, 0.0, 0.0].into()))
        .unwrap();
    let b = db.create_node(&["Doc"]).unwrap();
    db.set_node_property(b, "name", Value::from("B")).unwrap();
    db.set_node_property(b, "emb", Value::Vector(vec![0.0f32, 1.0, 0.0].into()))
        .unwrap();
    let c = db.create_node(&["Doc"]).unwrap();
    db.set_node_property(c, "name", Value::from("C")).unwrap();
    db.set_node_property(c, "emb", Value::Vector(vec![1.0f32, 0.0, 0.0].into()))
        .unwrap(); // identical to A
    db
}

fn single_float(result: &QueryResult) -> f64 {
    assert_eq!(result.row_count(), 1, "expected single row");
    match result.rows()[0][0] {
        Value::Float64(f) => f,
        ref other => panic!("expected Float64, got {other:?}"),
    }
}

#[test]
fn test_cosine_similarity_projection() {
    let db = setup();
    let s = db.session();
    let result = s
        .execute(
            "MATCH (n:Doc {name: 'A'}) \
             RETURN cosine_similarity(n.emb, [1.0, 0.0, 0.0])",
        )
        .unwrap();
    let v = single_float(&result);
    assert!((v - 1.0).abs() < 1e-6, "A parallel to query: got {v}");
}

#[test]
fn test_cosine_distance_projection() {
    let db = setup();
    let s = db.session();

    // Same direction should have distance 0
    let r_a = s
        .execute(
            "MATCH (n:Doc {name: 'A'}) \
             RETURN cosine_distance(n.emb, [1.0, 0.0, 0.0])",
        )
        .unwrap();
    let d_a = single_float(&r_a);
    assert!(
        d_a.abs() < 1e-6,
        "A == query: cosine_distance should be 0, got {d_a}"
    );

    // Orthogonal should have distance 1
    let r_b = s
        .execute(
            "MATCH (n:Doc {name: 'B'}) \
             RETURN cosine_distance(n.emb, [1.0, 0.0, 0.0])",
        )
        .unwrap();
    let d_b = single_float(&r_b);
    assert!(
        (d_b - 1.0).abs() < 1e-6,
        "B orthogonal to query: cosine_distance should be 1, got {d_b}"
    );
}

#[test]
fn test_cosine_distance_and_similarity_sum_to_one() {
    let db = setup();
    let s = db.session();
    let result = s
        .execute(
            "MATCH (n:Doc {name: 'B'}) \
             RETURN cosine_similarity(n.emb, [1.0, 1.0, 0.0]) + cosine_distance(n.emb, [1.0, 1.0, 0.0])",
        )
        .unwrap();
    let sum = single_float(&result);
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "sim + dist should equal 1, got {sum}"
    );
}

#[test]
fn test_cosine_distance_in_where_clause() {
    let db = setup();
    let s = db.session();
    let result = s
        .execute(
            "MATCH (n:Doc) \
             WHERE cosine_distance(n.emb, [1.0, 0.0, 0.0]) < 0.1 \
             RETURN n.name ORDER BY n.name",
        )
        .unwrap();
    assert_eq!(result.row_count(), 2, "A and C should be close to query");
    let names: Vec<String> = result
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::String(s) => s.to_string(),
            _ => String::new(),
        })
        .collect();
    assert_eq!(names, vec!["A", "C"]);
}

#[test]
fn test_cosine_distance_in_order_by() {
    let db = setup();
    let s = db.session();
    let result = s
        .execute(
            "MATCH (n:Doc) \
             RETURN n.name \
             ORDER BY cosine_distance(n.emb, [1.0, 0.0, 0.0]) ASC, n.name ASC \
             LIMIT 1",
        )
        .unwrap();
    assert_eq!(result.row_count(), 1);
    let name = match &result.rows()[0][0] {
        Value::String(s) => s.to_string(),
        _ => String::new(),
    };
    // A and C are both distance 0; ORDER BY n.name ASC picks A
    assert_eq!(name, "A");
}

#[test]
fn test_euclidean_distance_projection() {
    let db = setup();
    let s = db.session();
    let result = s
        .execute(
            "MATCH (n:Doc {name: 'B'}) \
             RETURN euclidean_distance(n.emb, [0.0, 0.0, 0.0])",
        )
        .unwrap();
    let v = single_float(&result);
    assert!((v - 1.0).abs() < 1e-6, "|[0,1,0]| = 1, got {v}");
}

#[test]
fn test_manhattan_distance_projection() {
    let db = setup();
    let s = db.session();
    let result = s
        .execute(
            "MATCH (n:Doc {name: 'B'}) \
             RETURN manhattan_distance(n.emb, [1.0, 0.0, 0.0])",
        )
        .unwrap();
    let v = single_float(&result);
    assert!((v - 2.0).abs() < 1e-6, "L1([1,0,0], [0,1,0]) = 2, got {v}");
}

#[test]
fn test_dot_product_projection() {
    let db = setup();
    let s = db.session();
    let result = s
        .execute(
            "MATCH (n:Doc {name: 'A'}) \
             RETURN dot_product(n.emb, [2.0, 3.0, 4.0])",
        )
        .unwrap();
    let v = single_float(&result);
    assert!((v - 2.0).abs() < 1e-6, "[1,0,0]·[2,3,4] = 2, got {v}");
}
