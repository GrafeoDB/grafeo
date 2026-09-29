//! Regression tests for `Int64` round-trip through `GrafeoDB::compact()`.
//!
//! Pre-fix behaviour: any `Value::Int64` column containing at least one
//! negative value was routed by `infer_type_from_values` to
//! `InferredType::Dict`, which stringified each value for storage and
//! decoded it back as `Value::String`. `WHERE n.num = 100` never matched
//! post-compact (comparing `Int64` to `String`), and `sum(n.num)`
//! returned `Float64` because the aggregate operator's `SumInt` state
//! parsed the strings and transitioned to `SumFloat`.
//!
//! The `ColumnCodec::RawI64` variant introduced in this PR stores
//! signed integers natively, preserving both the GQL type and signed
//! ordering semantics across compaction.
//!
//! Tracked upstream as GrafeoDB/grafeo#301.
//!
//! ```bash
//! cargo test -p grafeo-engine --features "compact-store lpg gql" \
//!     --test post_compact_numeric_content
//! ```

#![cfg(all(feature = "compact-store", feature = "lpg", feature = "gql"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

fn scalar(db: &GrafeoDB, q: &str) -> Value {
    let s = db.session();
    s.execute(q).unwrap().rows()[0][0].clone()
}

fn row_count(db: &GrafeoDB, q: &str) -> usize {
    let s = db.session();
    s.execute(q).unwrap().rows().len()
}

// ── Type round-trip ─────────────────────────────────────────────

#[test]
fn positive_only_sum_returns_int64() {
    // Unchanged from pre-fix behaviour: all-non-negative columns already
    // went through BitPacked and round-tripped correctly. Guards against
    // regressions from the RawI64 wiring.
    let mut db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["A"]).unwrap();
    let b = db.create_node(&["A"]).unwrap();
    db.set_node_property(a, "num", Value::Int64(100)).unwrap();
    db.set_node_property(b, "num", Value::Int64(200)).unwrap();

    let pre = scalar(&db, "MATCH (n) RETURN sum(n.num)");
    db.compact().unwrap();
    let post = scalar(&db, "MATCH (n) RETURN sum(n.num)");

    assert_eq!(pre, Value::Int64(300));
    assert_eq!(post, Value::Int64(300), "pos-only column must stay Int64");
    assert_eq!(pre, post);
}

#[test]
fn mixed_signs_sum_returns_int64() {
    // The diagnostic case from GrafeoDB/grafeo#301. Pre-fix this returned
    // Float64(50.0) post-compact.
    let mut db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["A"]).unwrap();
    let b = db.create_node(&["A"]).unwrap();
    db.set_node_property(a, "num", Value::Int64(100)).unwrap();
    db.set_node_property(b, "num", Value::Int64(-50)).unwrap();

    let pre = scalar(&db, "MATCH (n) RETURN sum(n.num)");
    db.compact().unwrap();
    let post = scalar(&db, "MATCH (n) RETURN sum(n.num)");

    assert_eq!(pre, Value::Int64(50));
    assert_eq!(post, Value::Int64(50), "mixed-sign column must stay Int64");
    assert_eq!(pre, post);
}

#[test]
fn i64_extremes_roundtrip() {
    let mut db = GrafeoDB::new_in_memory();
    for v in [i64::MIN + 1, -1, 0, 1, i64::MAX] {
        let n = db.create_node(&["A"]).unwrap();
        db.set_node_property(n, "val", Value::Int64(v)).unwrap();
    }
    db.compact().unwrap();

    // Each value survives as Int64; WHERE matches on specific values.
    for v in [i64::MIN + 1, -1, 0, 1, i64::MAX] {
        let q = format!("MATCH (n:A) WHERE n.val = {v} RETURN n.val");
        let r = db.session().execute(&q).unwrap();
        assert_eq!(r.rows().len(), 1, "WHERE n.val = {v}: expected 1 row");
        assert_eq!(
            r.rows()[0][0],
            Value::Int64(v),
            "decoded value for {v} should be Int64"
        );
    }
}

// ── WHERE filter ────────────────────────────────────────────────

#[test]
fn where_matches_negative_int() {
    let mut db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["A"]).unwrap();
    let b = db.create_node(&["A"]).unwrap();
    db.set_node_property(a, "num", Value::Int64(42)).unwrap();
    db.set_node_property(b, "num", Value::Int64(-17)).unwrap();
    db.compact().unwrap();

    assert_eq!(row_count(&db, "MATCH (n:A) WHERE n.num = 42 RETURN n"), 1);
    assert_eq!(row_count(&db, "MATCH (n:A) WHERE n.num = -17 RETURN n"), 1);
    assert_eq!(row_count(&db, "MATCH (n:A) WHERE n.num = 99 RETURN n"), 0);
}

#[test]
fn where_range_matches_across_zero() {
    let mut db = GrafeoDB::new_in_memory();
    for v in [-10i64, -5, 0, 5, 10, -100, 100] {
        let n = db.create_node(&["A"]).unwrap();
        db.set_node_property(n, "val", Value::Int64(v)).unwrap();
    }
    db.compact().unwrap();

    // Signed range semantics — -100 ≤ val < 0 must include exactly -100, -10, -5.
    assert_eq!(
        row_count(
            &db,
            "MATCH (n:A) WHERE n.val >= -100 AND n.val < 0 RETURN n"
        ),
        3
    );
    // -5 ≤ val ≤ 5 must include -5, 0, 5.
    assert_eq!(
        row_count(&db, "MATCH (n:A) WHERE n.val >= -5 AND n.val <= 5 RETURN n"),
        3
    );
}

// ── Per-label content parity ────────────────────────────────────

#[test]
fn per_label_sum_preserves_int64_across_compact() {
    let mut db = GrafeoDB::new_in_memory();
    for v in [10, -20, 30] {
        let n = db.create_node(&["A"]).unwrap();
        db.set_node_property(n, "num", Value::Int64(v)).unwrap();
    }
    for v in [1, 2] {
        let n = db.create_node(&["B"]).unwrap();
        db.set_node_property(n, "num", Value::Int64(v)).unwrap();
    }

    let pre_a = scalar(&db, "MATCH (n:A) RETURN sum(n.num)");
    let pre_b = scalar(&db, "MATCH (n:B) RETURN sum(n.num)");
    db.compact().unwrap();
    let post_a = scalar(&db, "MATCH (n:A) RETURN sum(n.num)");
    let post_b = scalar(&db, "MATCH (n:B) RETURN sum(n.num)");

    assert_eq!(pre_a, Value::Int64(20));
    assert_eq!(
        post_a,
        Value::Int64(20),
        "A column has mixed signs (RawI64)"
    );
    assert_eq!(pre_b, Value::Int64(3));
    assert_eq!(
        post_b,
        Value::Int64(3),
        "B column is non-negative (BitPacked)"
    );
}
