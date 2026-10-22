//! Tests for NULL handling and type coercion in query execution.
//!
//! T2-03: NULL handling gaps (comparisons, aggregates, boolean logic, CASE, DISTINCT)
//! T2-04: Type coercion (Int64 vs Float64, mixed aggregates)
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test null_and_coercion
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

fn setup_with_nulls() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .create_node_with_props(
            &["Item"],
            [
                ("name", Value::String("alpha".into())),
                ("val", Value::Int64(10)),
                ("score", Value::Float64(1.5)),
            ],
        )
        .unwrap();
    session
        .create_node_with_props(
            &["Item"],
            [
                ("name", Value::String("beta".into())),
                ("val", Value::Null),
                ("score", Value::Float64(2.5)),
            ],
        )
        .unwrap();
    session
        .create_node_with_props(
            &["Item"],
            [
                ("name", Value::String("gamma".into())),
                ("val", Value::Int64(30)),
                ("score", Value::Null),
            ],
        )
        .unwrap();
    db
}

// ===========================================================================
// T2-03: NULL in comparisons
// ===========================================================================

#[test]
fn test_null_equality_filters_out() {
    let db = setup_with_nulls();
    let session = db.session();
    // WHERE val = NULL: three-valued logic says NULL = NULL is UNKNOWN, so no rows match
    let r = session
        .execute("MATCH (i:Item) WHERE i.val = NULL RETURN i.name AS name ORDER BY name")
        .unwrap();
    assert_eq!(
        r.rows().len(),
        0,
        "NULL = NULL is UNKNOWN: no rows should match, got {} rows",
        r.rows().len()
    );
}

#[test]
fn test_null_ne_null_is_unknown() {
    let db = setup_with_nulls();
    let session = db.session();
    // WHERE val <> NULL: also UNKNOWN, no rows match
    let r = session
        .execute("MATCH (i:Item) WHERE i.val <> NULL RETURN i.name AS name ORDER BY name")
        .unwrap();
    assert_eq!(
        r.rows().len(),
        0,
        "NULL <> NULL is UNKNOWN: no rows should match"
    );
}

#[test]
fn test_case_when_null_eq_null_is_unknown() {
    let db = setup_with_nulls();
    let session = db.session();
    let r = session
        .execute(
            "MATCH (i:Item) WHERE i.name = 'beta' \
             RETURN CASE WHEN i.val = NULL THEN 'hit' ELSE 'miss' END AS result",
        )
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(
        r.rows()[0][0].as_str(),
        Some("miss"),
        "CASE WHEN val = NULL should be UNKNOWN, falling through to ELSE"
    );
}

#[test]
fn test_simple_case_null_when_null() {
    let db = setup_with_nulls();
    let session = db.session();
    let r = session
        .execute(
            "MATCH (i:Item) WHERE i.name = 'beta' \
             RETURN CASE i.val WHEN NULL THEN 'hit' ELSE 'miss' END AS result",
        )
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(
        r.rows()[0][0].as_str(),
        Some("miss"),
        "Simple CASE: NULL WHEN NULL should not match"
    );
}

#[test]
fn test_or_with_null_unknown() {
    let db = setup_with_nulls();
    let session = db.session();
    // (val = NULL) is UNKNOWN; OR with a TRUE condition should still return the TRUE row
    let r = session
        .execute(
            "MATCH (i:Item) WHERE i.val = NULL OR i.name = 'alpha' \
             RETURN i.name AS name ORDER BY name",
        )
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(r.rows()[0][0].as_str(), Some("alpha"));
}

#[test]
fn test_and_with_null_unknown() {
    let db = setup_with_nulls();
    let session = db.session();
    // (val = NULL) is UNKNOWN; AND with anything is at most UNKNOWN, never TRUE
    let r = session
        .execute(
            "MATCH (i:Item) WHERE i.val = NULL AND i.name = 'beta' \
             RETURN i.name AS name",
        )
        .unwrap();
    assert_eq!(
        r.rows().len(),
        0,
        "UNKNOWN AND TRUE = UNKNOWN, should match nothing"
    );
}

#[test]
fn test_nullif_both_null() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let r = session
        .execute("RETURN NULLIF(NULL, NULL) AS result")
        .unwrap();
    assert!(
        r.rows()[0][0].is_null(),
        "NULLIF(NULL, NULL) should return NULL"
    );
}

#[test]
fn test_nullif_value_null() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let r = session
        .execute("RETURN NULLIF(42, NULL) AS result")
        .unwrap();
    assert_eq!(
        r.rows()[0][0].as_int64(),
        Some(42),
        "NULLIF(42, NULL) should return 42"
    );
}

#[test]
fn test_null_comparison_gt_filters_out() {
    let db = setup_with_nulls();
    let session = db.session();
    // WHERE val > NULL: three-valued logic says this is unknown, should filter out
    let r = session
        .execute("MATCH (i:Item) WHERE i.val > NULL RETURN i.name")
        .unwrap();
    assert_eq!(
        r.rows().len(),
        0,
        "Comparison with NULL should yield unknown and filter out all rows"
    );
}

#[test]
fn test_missing_property_is_null() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .create_node_with_props(&["Thing"], [("name", Value::String("only_name".into()))])
        .unwrap();

    let r = session
        .execute("MATCH (t:Thing) RETURN t.nonexistent AS val")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(
        r.rows()[0][0],
        Value::Null,
        "Missing property should return NULL"
    );
}

// ===========================================================================
// T2-03: NULL in aggregates
// ===========================================================================

#[test]
fn test_sum_skips_nulls() {
    let db = setup_with_nulls();
    let session = db.session();
    let r = session
        .execute("MATCH (i:Item) RETURN sum(i.val) AS total")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    // sum(10, NULL, 30) = 40
    match &r.rows()[0][0] {
        Value::Int64(v) => assert_eq!(*v, 40),
        Value::Float64(v) => assert!((*v - 40.0).abs() < 0.01),
        other => panic!("expected numeric, got {other:?}"),
    }
}

#[test]
fn test_avg_skips_nulls() {
    let db = setup_with_nulls();
    let session = db.session();
    let r = session
        .execute("MATCH (i:Item) RETURN avg(i.val) AS average")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    // avg(10, NULL, 30) = 20.0 (only 2 non-null values)
    if let Value::Float64(v) = r.rows()[0][0] {
        assert!(
            (v - 20.0).abs() < 0.01,
            "avg(10, NULL, 30) should be 20.0, got {v}"
        );
    } else {
        panic!("expected Float64, got {:?}", r.rows()[0][0]);
    }
}

#[test]
fn test_min_max_skip_nulls() {
    let db = setup_with_nulls();
    let session = db.session();
    let r = session
        .execute("MATCH (i:Item) RETURN min(i.val) AS lo, max(i.val) AS hi")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    // min(10, NULL, 30) = 10, max(10, NULL, 30) = 30
    match &r.rows()[0][0] {
        Value::Int64(v) => assert_eq!(*v, 10),
        Value::Float64(v) => assert!((*v - 10.0).abs() < 0.01),
        other => panic!("expected numeric for min, got {other:?}"),
    }
    match &r.rows()[0][1] {
        Value::Int64(v) => assert_eq!(*v, 30),
        Value::Float64(v) => assert!((*v - 30.0).abs() < 0.01),
        other => panic!("expected numeric for max, got {other:?}"),
    }
}

#[test]
fn test_count_excludes_nulls() {
    let db = setup_with_nulls();
    let session = db.session();
    // count(i.val) excludes NULLs; count(i) counts all rows
    let r = session
        .execute("MATCH (i:Item) RETURN count(i.val) AS cnt, count(i) AS total")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    // count(val) should exclude NULLs: 2
    assert_eq!(r.rows()[0][0], Value::Int64(2));
    // count(i) counts all rows: 3
    assert_eq!(r.rows()[0][1], Value::Int64(3));
}

// ===========================================================================
// T2-03: NULL in CASE WHEN
// ===========================================================================

#[test]
fn test_case_when_null_goes_to_else() {
    let db = setup_with_nulls();
    let session = db.session();
    let r = session
        .execute(
            "MATCH (i:Item) WHERE i.name = 'beta' \
             RETURN CASE WHEN i.val IS NOT NULL THEN 'has_val' ELSE 'no_val' END AS status",
        )
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(r.rows()[0][0], Value::String("no_val".into()));
}

#[test]
fn test_case_when_with_null_value() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .create_node_with_props(&["X"], [("v", Value::Int64(1))])
        .unwrap();
    let r = session
        .execute(
            "MATCH (x:X) \
             RETURN CASE x.v WHEN 1 THEN 'one' WHEN 2 THEN 'two' ELSE 'other' END AS label",
        )
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(r.rows()[0][0], Value::String("one".into()));
}

// ===========================================================================
// T2-03: NULL with IN operator
// ===========================================================================

#[test]
fn test_where_value_in_list_with_null() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .create_node_with_props(&["N"], [("v", Value::Int64(1))])
        .unwrap();
    session
        .create_node_with_props(&["N"], [("v", Value::Int64(2))])
        .unwrap();
    session
        .create_node_with_props(&["N"], [("v", Value::Int64(5))])
        .unwrap();

    let r = session
        .execute("MATCH (n:N) WHERE n.v IN [1, NULL, 5] RETURN n.v AS v ORDER BY v")
        .unwrap();
    // 1 and 5 match directly; NULL in the list should not cause issues
    assert!(
        r.rows().len() >= 2,
        "At least 1 and 5 should match, got {} rows",
        r.rows().len()
    );
}

// ===========================================================================
// T2-03: RETURN with NULL arithmetic
// ===========================================================================

#[test]
fn test_null_arithmetic_returns_null() {
    let db = setup_with_nulls();
    let session = db.session();
    // beta has val=NULL, so val + 1 should be NULL
    let r = session
        .execute(
            "MATCH (i:Item) WHERE i.name = 'beta' \
             RETURN i.val + 1 AS incremented",
        )
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(r.rows()[0][0], Value::Null, "NULL + 1 should be NULL");
}

// ===========================================================================
// T2-04: Type coercion
// ===========================================================================

/// Int64 property compared against Float64 literal: `WHERE n.v > 2.5` matches Int64(3).
#[test]
fn test_int_float_comparison_gt() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .create_node_with_props(&["Num"], [("v", Value::Int64(3))])
        .unwrap();

    let r = session
        .execute("MATCH (n:Num) WHERE n.v > 2.5 RETURN n.v AS v")
        .unwrap();
    assert_eq!(r.rows().len(), 1, "Int64(3) > Float64(2.5) should match");
}

/// Int64 vs Float64 with `<` operator.
#[test]
fn test_int_float_comparison_lt() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .create_node_with_props(&["Num"], [("v", Value::Int64(2))])
        .unwrap();

    let r = session
        .execute("MATCH (n:Num) WHERE n.v < 2.5 RETURN n.v AS v")
        .unwrap();
    assert_eq!(r.rows().len(), 1, "Int64(2) < Float64(2.5) should match");
}

#[test]
fn test_int_float_arithmetic() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .create_node_with_props(&["Num"], [("v", Value::Int64(3))])
        .unwrap();

    let r = session
        .execute("MATCH (n:Num) RETURN n.v + 0.5 AS result")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    // Int64(3) + Float64(0.5) should promote to Float64(3.5)
    if let Value::Float64(v) = r.rows()[0][0] {
        assert!((v - 3.5).abs() < 0.01, "3 + 0.5 should be 3.5, got {v}");
    } else {
        panic!("expected Float64, got {:?}", r.rows()[0][0]);
    }
}

/// Documents that SUM over mixed Int64/Float64 properties does not coerce
/// to Float64. The Float64 values are silently ignored by the integer sum path.
/// TODO: Once aggregate coercion is fixed, change to assert Float64(60.5).
#[test]
fn test_sum_mixed_int_float() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .create_node_with_props(&["Val"], [("n", Value::Int64(10))])
        .unwrap();
    session
        .create_node_with_props(&["Val"], [("n", Value::Float64(20.5))])
        .unwrap();
    session
        .create_node_with_props(&["Val"], [("n", Value::Int64(30))])
        .unwrap();

    let r = session
        .execute("MATCH (v:Val) RETURN sum(v.n) AS total")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    // SUM promotes to Float64 when it encounters mixed Int64/Float64 values.
    match &r.rows()[0][0] {
        Value::Float64(v) => {
            assert!(
                (*v - 60.5).abs() < 0.01,
                "sum(10, 20.5, 30) should be 60.5, got {v}"
            );
        }
        other => panic!("expected Float64(60.5), got {other:?}"),
    }
}

/// Int64 vs Float64 equality: `WHERE n.v = 5.0` matches Int64(5).
#[test]
fn test_int_equality_with_float() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .create_node_with_props(&["Num"], [("v", Value::Int64(5))])
        .unwrap();

    let r = session
        .execute("MATCH (n:Num) WHERE n.v = 5.0 RETURN n.v AS v")
        .unwrap();
    assert_eq!(r.rows().len(), 1, "Int64(5) = Float64(5.0) should match");
}

/// Same-type comparison works correctly.
#[test]
fn test_same_type_int_comparison() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .create_node_with_props(&["Num"], [("v", Value::Int64(3))])
        .unwrap();

    let r = session
        .execute("MATCH (n:Num) WHERE n.v > 2 RETURN n.v AS v")
        .unwrap();
    assert_eq!(r.rows().len(), 1, "Same-type Int64 comparison should work");
    assert_eq!(r.rows()[0][0], Value::Int64(3));
}

/// Same-type Float64 comparison works.
#[test]
fn test_same_type_float_comparison() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .create_node_with_props(&["Num"], [("v", Value::Float64(3.5))])
        .unwrap();

    let r = session
        .execute("MATCH (n:Num) WHERE n.v > 2.0 RETURN n.v AS v")
        .unwrap();
    assert_eq!(
        r.rows().len(),
        1,
        "Same-type Float64 comparison should work"
    );
}

// ===========================================================================
// T2-03: DISTINCT with NULLs
// ===========================================================================

#[test]
fn test_distinct_with_nulls() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .create_node_with_props(&["D"], [("v", Value::Int64(1))])
        .unwrap();
    session
        .create_node_with_props(&["D"], [("v", Value::Int64(1))])
        .unwrap();
    session
        .create_node_with_props(&["D"], [("v", Value::Null)])
        .unwrap();
    session
        .create_node_with_props(&["D"], [("v", Value::Null)])
        .unwrap();
    session
        .create_node_with_props(&["D"], [("v", Value::Int64(2))])
        .unwrap();

    let r = session
        .execute("MATCH (d:D) RETURN DISTINCT d.v AS v ORDER BY v")
        .unwrap();
    // Should have 3 distinct values: NULL, 1, 2
    assert_eq!(
        r.rows().len(),
        3,
        "DISTINCT should deduplicate NULLs: expected 3 distinct values, got {}",
        r.rows().len()
    );
}

// ===========================================================================
// T2-03: GROUP BY with NULL keys
// ===========================================================================

#[test]
fn test_group_by_null_key() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .create_node_with_props(
            &["Sale"],
            [
                ("region", Value::String("North".into())),
                ("amount", Value::Int64(100)),
            ],
        )
        .unwrap();
    session
        .create_node_with_props(
            &["Sale"],
            [("region", Value::Null), ("amount", Value::Int64(50))],
        )
        .unwrap();
    session
        .create_node_with_props(
            &["Sale"],
            [
                ("region", Value::String("North".into())),
                ("amount", Value::Int64(200)),
            ],
        )
        .unwrap();
    session
        .create_node_with_props(
            &["Sale"],
            [("region", Value::Null), ("amount", Value::Int64(75))],
        )
        .unwrap();

    let r = session
        .execute(
            "MATCH (s:Sale) \
             RETURN s.region AS region, sum(s.amount) AS total \
             ORDER BY region",
        )
        .unwrap();
    // Should have 2 groups: NULL (50+75=125), North (100+200=300)
    assert_eq!(r.rows().len(), 2, "NULL keys should form their own group");
}

// ===========================================================================
// NULL operands of a grouped aggregate, the first of a group too
// ===========================================================================

/// The operands, a null first, as in a group whose first row has no value.
const OPERANDS: &str = "UNWIND [null, 19, 3, null, 88, 3] AS x";

/// The aggregates of `x` that read an operand.
const AGGREGATES_OF_X: &str = "min(x) AS lo, max(x) AS hi, sample(x) AS s, count(x) AS n, \
     count(DISTINCT x) AS d, sum(x) AS total, avg(x) AS mean, stdev(x) AS sd, \
     percentileCont(x, 0.5) AS p, collect(x) AS xs, collect(DISTINCT x) AS ds";

/// Operands keyed by group: group 1 and 2 start with a null, group 3 has only one.
const KEYED_OPERANDS: &str =
    "UNWIND [[1, null], [1, 19], [2, null], [2, 3], [2, 88], [3, null]] AS r";

fn list(values: &[i64]) -> Value {
    Value::List(
        values
            .iter()
            .map(|&value| Value::Int64(value))
            .collect::<Vec<_>>()
            .into(),
    )
}

/// Every aggregate skips the null operands of a group, the first one too, and
/// a group key does not change what an aggregate returns.
fn assert_grouped_aggregates_skip_nulls(execute: impl Fn(&str) -> Vec<Vec<Value>>) {
    let grouped = execute(&format!("{OPERANDS} RETURN 0 AS g, {AGGREGATES_OF_X}"));
    assert_eq!(grouped.len(), 1);
    let row = &grouped[0];
    assert_eq!(
        row[..8],
        [
            Value::Int64(0),
            Value::Int64(3),
            Value::Int64(88),
            Value::Int64(19),
            Value::Int64(4),
            Value::Int64(3),
            Value::Int64(113),
            Value::Float64(28.25),
        ]
    );
    assert!(matches!(row[8], Value::Float64(_)), "stdev: {:?}", row[8]);
    assert_eq!(
        row[9..],
        [
            Value::Float64(11.0),
            list(&[19, 3, 88, 3]),
            list(&[19, 3, 88])
        ]
    );
    let ungrouped = execute(&format!("{OPERANDS} RETURN {AGGREGATES_OF_X}"));
    assert_eq!(ungrouped.len(), 1);
    assert_eq!(row[1..], ungrouped[0][..]);

    let keyed = execute(&format!(
        "{KEYED_OPERANDS} RETURN r[0] AS g, min(r[1]) AS lo, max(r[1]) AS hi, \
         sample(r[1]) AS s, collect(r[1]) AS xs ORDER BY g"
    ));
    assert_eq!(
        keyed,
        [
            vec![
                Value::Int64(1),
                Value::Int64(19),
                Value::Int64(19),
                Value::Int64(19),
                list(&[19]),
            ],
            vec![
                Value::Int64(2),
                Value::Int64(3),
                Value::Int64(88),
                Value::Int64(3),
                list(&[3, 88]),
            ],
            vec![
                Value::Int64(3),
                Value::Null,
                Value::Null,
                Value::Null,
                list(&[]),
            ],
        ]
    );
}

#[test]
fn grouped_aggregates_skip_a_leading_null_in_gql() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    assert_grouped_aggregates_skip_nulls(|query| session.execute(query).unwrap().rows().to_vec());
}

#[cfg(feature = "cypher")]
#[test]
fn grouped_aggregates_skip_a_leading_null_in_cypher() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    assert_grouped_aggregates_skip_nulls(|query| {
        session.execute_cypher(query).unwrap().rows().to_vec()
    });
}
