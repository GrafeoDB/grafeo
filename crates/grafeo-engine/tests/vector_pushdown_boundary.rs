//! Boundary-row regression tests for vector predicate pushdown.
//!
//! Strict operators (`>` / `<`) on vector similarity / distance functions
//! push into a `VectorScanOperator` whose internal `apply_filters` uses
//! inclusive comparisons (`>=` / `<=`). Without a residual filter above the
//! scan, rows at exactly the threshold leak through.
//!
//! Each test seeds data with two rows whose distance equals the threshold
//! exactly in f32 (axis-aligned vectors against the origin) and asserts:
//!
//!   * strict ops exclude those boundary rows,
//!   * inclusive ops include them.
//!
//! Note on cosine: the `cosine_distance_scalar` implementation adds
//! `f32::EPSILON` to its denominator (see `simd.rs:257`), so the reported
//! similarity is always slightly less than the mathematical value. That makes
//! a "seed cos=T exactly" test unstable on the cosine metric. Euclidean and
//! manhattan are free of this bias and cover the same planner path via the
//! shared `ExtractedVectorPredicate::strict` constructor.
//!
//! ```bash
//! cargo test -p grafeo-engine --features lpg,gql,wal,vector-index \
//!     --test vector_pushdown_boundary
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "vector-index"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Euclidean fixture: axis-aligned vectors against `[0.0, 0.0, 0.0]` so the
/// distance equals the first component exactly. Threshold at `0.5` with two
/// boundary rows makes the leak a detectable row-count delta (2 vs 4).
fn setup_euclidean_fixture() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    for (label, d) in [
        ("d03", 0.3f32),
        ("d05a", 0.5),
        ("d05b", 0.5),
        ("d07", 0.7),
        ("d09", 0.9),
    ] {
        let n = db.create_node(&["Item"]).unwrap();
        db.set_node_property(n, "label", Value::String(label.into()))
            .unwrap();
        db.set_node_property(n, "vec", Value::Vector(vec![d, 0.0, 0.0].into()))
            .unwrap();
    }
    db.create_vector_index("Item", "vec", Some(3), Some("euclidean"), None, None, None)
        .expect("create vector index");
    db
}

#[test]
fn euclidean_distance_strict_lt_excludes_boundary() {
    let db = setup_euclidean_fixture();
    let s = db.session();

    let result = s
        .execute(
            "MATCH (n:Item) WHERE euclidean_distance(n.vec, [0.0, 0.0, 0.0]) < 0.5 RETURN n.label",
        )
        .expect("query should succeed");
    assert_eq!(
        result.row_count(),
        1,
        "strict < 0.5 must exclude both boundary rows; got {:?}",
        result.rows()
    );
}

#[test]
fn euclidean_distance_inclusive_le_includes_boundary() {
    let db = setup_euclidean_fixture();
    let s = db.session();

    let result = s
        .execute(
            "MATCH (n:Item) WHERE euclidean_distance(n.vec, [0.0, 0.0, 0.0]) <= 0.5 RETURN n.label",
        )
        .expect("query should succeed");
    assert_eq!(
        result.row_count(),
        3,
        "inclusive <= 0.5 must include both boundary rows; got {:?}",
        result.rows()
    );
}

/// Manhattan fixture: same axis-aligned shape; L1 distance equals `|x|`.
fn setup_manhattan_fixture() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    for (label, d) in [
        ("d03", 0.3f32),
        ("d05a", 0.5),
        ("d05b", 0.5),
        ("d07", 0.7),
        ("d09", 0.9),
    ] {
        let n = db.create_node(&["Item"]).unwrap();
        db.set_node_property(n, "label", Value::String(label.into()))
            .unwrap();
        db.set_node_property(n, "vec", Value::Vector(vec![d, 0.0, 0.0].into()))
            .unwrap();
    }
    db.create_vector_index("Item", "vec", Some(3), Some("manhattan"), None, None, None)
        .expect("create vector index");
    db
}

#[test]
fn manhattan_distance_strict_lt_excludes_boundary() {
    let db = setup_manhattan_fixture();
    let s = db.session();

    let result = s
        .execute(
            "MATCH (n:Item) WHERE manhattan_distance(n.vec, [0.0, 0.0, 0.0]) < 0.5 RETURN n.label",
        )
        .expect("query should succeed");
    assert_eq!(
        result.row_count(),
        1,
        "strict < 0.5 must exclude both boundary rows; got {:?}",
        result.rows()
    );
}

#[test]
fn manhattan_distance_inclusive_le_includes_boundary() {
    let db = setup_manhattan_fixture();
    let s = db.session();

    let result = s
        .execute(
            "MATCH (n:Item) WHERE manhattan_distance(n.vec, [0.0, 0.0, 0.0]) <= 0.5 RETURN n.label",
        )
        .expect("query should succeed");
    assert_eq!(
        result.row_count(),
        3,
        "inclusive <= 0.5 must include both boundary rows; got {:?}",
        result.rows()
    );
}
