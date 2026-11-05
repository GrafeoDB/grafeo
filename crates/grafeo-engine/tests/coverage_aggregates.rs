//! Tests for aggregate translator and execution coverage gaps.
//!
//! Targets: aggregate.rs (45.45%), common.rs (64.48%), expression.rs (82.32%)
// Test values are small known constants
#![allow(clippy::cast_possible_truncation)]
//!
//! ```bash
//! cargo test -p grafeo-engine --test main coverage_aggregates::
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Creates 5 Data nodes (Alix, Gus, Vincent, Jules, Mia) with x/y/score properties.
fn stats_graph() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for (name, x, y) in [
        ("Alix", 1.0, 2.0),
        ("Gus", 2.0, 4.0),
        ("Vincent", 3.0, 6.0),
        ("Jules", 4.0, 8.0),
        ("Mia", 5.0, 10.0),
    ] {
        session
            .create_node_with_props(
                &["Data"],
                [
                    ("name", Value::String(name.into())),
                    ("x", Value::Float64(x)),
                    ("y", Value::Float64(y)),
                    ("score", Value::Int64(x as i64 * 10)),
                ],
            )
            .unwrap();
    }
    db
}

// ---------------------------------------------------------------------------
// Wrapped aggregates: exercises extract_wrapped_aggregate (Binary branch)
// ---------------------------------------------------------------------------

#[test]
fn test_wrapped_aggregate_count_gt_zero() {
    let db = stats_graph();
    let s = db.session();
    let r = s
        .execute("MATCH (d:Data) RETURN count(d) > 0 AS has_data")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(r.rows()[0][0], Value::Bool(true));
}

#[test]
fn test_wrapped_aggregate_sum_minus_literal() {
    let db = stats_graph();
    let s = db.session();
    let r = s
        .execute("MATCH (d:Data) RETURN sum(d.score) - 10 AS adjusted")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    // sum(10+20+30+40+50) - 10 = 140
    assert_eq!(r.rows()[0][0], Value::Int64(140));
}

#[test]
fn test_wrapped_aggregate_not_count() {
    let db = stats_graph();
    let s = db.session();
    let r = s
        .execute("MATCH (d:Data) RETURN NOT (count(d) > 100) AS few")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(r.rows()[0][0], Value::Bool(true));
}

// ---------------------------------------------------------------------------
// GROUP_CONCAT / LISTAGG with separator
// ---------------------------------------------------------------------------

#[test]
fn test_group_concat_default_separator() {
    let db = stats_graph();
    let s = db.session();
    let r = s
        .execute("MATCH (d:Data) RETURN group_concat(d.name) AS names")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    if let Value::String(names) = &r.rows()[0][0] {
        assert_eq!(names.split(' ').count(), 5);
    } else {
        panic!("expected string, got {:?}", r.rows()[0][0]);
    }
}

#[test]
fn test_group_concat_custom_separator() {
    let db = stats_graph();
    let s = db.session();
    let r = s
        .execute("MATCH (d:Data) RETURN group_concat(d.name, ';') AS names")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    if let Value::String(names) = &r.rows()[0][0] {
        assert!(names.contains(';'), "expected semicolons: {names}");
        assert_eq!(names.split(';').count(), 5);
    } else {
        panic!("expected string, got {:?}", r.rows()[0][0]);
    }
}

#[test]
fn test_listagg_default_comma() {
    let db = stats_graph();
    let s = db.session();
    let r = s
        .execute("MATCH (d:Data) RETURN listagg(d.name) AS names")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    if let Value::String(names) = &r.rows()[0][0] {
        assert!(names.contains(','), "expected commas: {names}");
    } else {
        panic!("expected string, got {:?}", r.rows()[0][0]);
    }
}

// ---------------------------------------------------------------------------
// SAMPLE aggregate
// ---------------------------------------------------------------------------

#[test]
fn test_sample_returns_one_value() {
    let db = stats_graph();
    let s = db.session();
    let r = s
        .execute("MATCH (d:Data) RETURN sample(d.name) AS picked")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    if let Value::String(name) = &r.rows()[0][0] {
        assert!(
            ["Alix", "Gus", "Vincent", "Jules", "Mia"].contains(&name.as_str()),
            "unexpected: {name}"
        );
    } else {
        panic!("expected string, got {:?}", r.rows()[0][0]);
    }
}

// ---------------------------------------------------------------------------
// COLLECT aggregate
// ---------------------------------------------------------------------------

#[test]
fn test_collect_to_list() {
    let db = stats_graph();
    let s = db.session();
    let r = s
        .execute("MATCH (d:Data) RETURN collect(d.score) AS scores")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    if let Value::List(items) = &r.rows()[0][0] {
        assert_eq!(items.len(), 5);
    } else {
        panic!("expected list, got {:?}", r.rows()[0][0]);
    }
}

// ---------------------------------------------------------------------------
// PERCENTILE with integer parameter (exercises int-to-float branch)
// ---------------------------------------------------------------------------

#[test]
fn test_percentile_disc_with_integer_param() {
    let db = stats_graph();
    let s = db.session();
    // percentile_disc(score, 1) should return max value
    let r = s
        .execute("MATCH (d:Data) RETURN percentile_disc(d.score, 1) AS p100")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    // May return Float64 or Int64 depending on implementation
    match &r.rows()[0][0] {
        Value::Int64(v) => assert_eq!(*v, 50),
        Value::Float64(v) => assert!((*v - 50.0).abs() < 0.01),
        other => panic!("expected numeric, got {other:?}"),
    }
}

#[test]
fn test_percentile_cont_zero() {
    let db = stats_graph();
    let s = db.session();
    let r = s
        .execute("MATCH (d:Data) RETURN percentile_cont(d.score, 0) AS p0")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    match &r.rows()[0][0] {
        Value::Float64(v) => assert!((*v - 10.0).abs() < 0.01),
        Value::Int64(v) => assert_eq!(*v, 10),
        other => panic!("expected numeric, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// STDEV / VARIANCE (sample and population)
// ---------------------------------------------------------------------------

#[test]
fn test_stdev_and_stdevp() {
    let db = stats_graph();
    let s = db.session();
    let r = s
        .execute("MATCH (d:Data) RETURN stdev(d.score) AS s, stdevp(d.score) AS sp")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    if let Value::Float64(s) = r.rows()[0][0] {
        assert!(s > 0.0);
    }
    if let Value::Float64(sp) = r.rows()[0][1] {
        assert!(sp > 0.0);
    }
}

#[test]
fn test_variance_and_variance_pop() {
    let db = stats_graph();
    let s = db.session();
    let r = s
        .execute("MATCH (d:Data) RETURN variance(d.score) AS v, var_pop(d.score) AS vp")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    if let Value::Float64(v) = r.rows()[0][0] {
        assert!(v > 0.0);
    }
}

// ---------------------------------------------------------------------------
// Multiple aggregates + GROUP BY in one query (uses WITH for grouping)
// ---------------------------------------------------------------------------

#[test]
fn test_mixed_aggregates_with_group_by() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for (name, city, score) in [
        ("Alix", "Amsterdam", 80),
        ("Gus", "Amsterdam", 90),
        ("Vincent", "Berlin", 70),
        ("Jules", "Berlin", 85),
    ] {
        session
            .create_node_with_props(
                &["Person"],
                [
                    ("name", Value::String(name.into())),
                    ("city", Value::String(city.into())),
                    ("score", Value::Int64(score)),
                ],
            )
            .unwrap();
    }
    // Use RETURN with aggregates directly: non-aggregated p.city acts as grouping key
    let r = session
        .execute(
            "MATCH (p:Person) \
             RETURN p.city AS city, count(p) AS cnt, min(p.score) AS lo, max(p.score) AS hi",
        )
        .unwrap();
    assert_eq!(r.rows().len(), 2);
    // Both groups should have count=2
    assert_eq!(r.rows()[0][1], Value::Int64(2));
    assert_eq!(r.rows()[1][1], Value::Int64(2));
}

// ---------------------------------------------------------------------------
// ORDER BY alias after aggregation (was: "Undefined variable 'city'")
// ---------------------------------------------------------------------------

#[test]
fn test_order_by_alias_after_aggregation() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for (name, city) in [
        ("Alix", "Berlin"),
        ("Gus", "Amsterdam"),
        ("Vincent", "Berlin"),
        ("Jules", "Amsterdam"),
        ("Mia", "Paris"),
    ] {
        session
            .create_node_with_props(
                &["Person"],
                [
                    ("name", Value::String(name.into())),
                    ("city", Value::String(city.into())),
                ],
            )
            .unwrap();
    }
    let r = session
        .execute("MATCH (p:Person) RETURN p.city AS city, count(p) AS cnt ORDER BY city")
        .unwrap();
    assert_eq!(r.rows().len(), 3);
    // Sorted ascending by city: Amsterdam, Berlin, Paris
    assert_eq!(r.rows()[0][0], Value::String("Amsterdam".into()));
    assert_eq!(r.rows()[1][0], Value::String("Berlin".into()));
    assert_eq!(r.rows()[2][0], Value::String("Paris".into()));
}

#[test]
fn test_order_by_aggregate_alias_desc() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for (name, city) in [
        ("Alix", "Berlin"),
        ("Gus", "Amsterdam"),
        ("Vincent", "Berlin"),
        ("Jules", "Amsterdam"),
        ("Mia", "Paris"),
    ] {
        session
            .create_node_with_props(
                &["Person"],
                [
                    ("name", Value::String(name.into())),
                    ("city", Value::String(city.into())),
                ],
            )
            .unwrap();
    }
    let r = session
        .execute("MATCH (p:Person) RETURN p.city AS city, count(p) AS cnt ORDER BY cnt DESC")
        .unwrap();
    assert_eq!(r.rows().len(), 3);
    // DESC by cnt: Amsterdam(2) and Berlin(2) first, Paris(1) last
    assert_eq!(r.rows()[0][1], Value::Int64(2));
    assert_eq!(r.rows()[1][1], Value::Int64(2));
    assert_eq!(r.rows()[2][1], Value::Int64(1));
    // Paris must be last
    assert_eq!(r.rows()[2][0], Value::String("Paris".into()));
}

// ---------------------------------------------------------------------------
// Expression coverage: NULLIF
// ---------------------------------------------------------------------------

#[test]
fn test_nullif_expression() {
    let db = stats_graph();
    let s = db.session();
    let r = s
        .execute("MATCH (d:Data) RETURN nullif(d.score, 10) AS v ORDER BY d.x")
        .unwrap();
    // First person (score=10) should get NULL
    assert_eq!(r.rows()[0][0], Value::Null);
    // Second person (score=20) should keep 20
    assert_eq!(r.rows()[1][0], Value::Int64(20));
}

// ---------------------------------------------------------------------------
// List predicates via Cypher-compatible syntax
// ---------------------------------------------------------------------------

// Tests all()/any() in WHERE clause (RETURN variant in gql_spec_compliance.rs)

#[test]
fn test_list_predicate_all() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();
    s.create_node_with_props(&["Flag"], [("v", Value::Int64(1))])
        .unwrap();
    let r = s
        .execute("MATCH (f:Flag) WHERE all(x IN [2, 4, 6] WHERE x % 2 = 0) RETURN f.v AS v")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
}

#[test]
fn test_list_predicate_any() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();
    s.create_node_with_props(&["Flag"], [("v", Value::Int64(1))])
        .unwrap();
    let r = s
        .execute("MATCH (f:Flag) WHERE any(x IN [1, 2, 3] WHERE x > 2) RETURN f.v AS v")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
}

// ---------------------------------------------------------------------------
// Wrapped aggregate: sum(x) * 2 (right-side aggregate in binary)
// ---------------------------------------------------------------------------

#[test]
fn test_wrapped_aggregate_literal_plus_count() {
    let db = stats_graph();
    let s = db.session();
    // 100 + count(d): aggregate on the right side of binary
    let r = s
        .execute("MATCH (d:Data) RETURN 100 + count(d) AS total")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(r.rows()[0][0], Value::Int64(105));
}

// ---------------------------------------------------------------------------
// COUNT(DISTINCT x) and COUNT(x) non-null
// ---------------------------------------------------------------------------

#[test]
fn test_count_distinct() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for city in ["Amsterdam", "Amsterdam", "Berlin", "Berlin", "Paris"] {
        session
            .create_node_with_props(&["City"], [("name", Value::String(city.into()))])
            .unwrap();
    }
    let r = session
        .execute("MATCH (c:City) RETURN count(DISTINCT c.name) AS unique_cities")
        .unwrap();
    assert_eq!(r.rows()[0][0], Value::Int64(3));
}

#[test]
fn test_count_expression_non_null() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .create_node_with_props(&["Item"], [("val", Value::Int64(1))])
        .unwrap();
    session
        .create_node_with_props(&["Item"], [("val", Value::Null)])
        .unwrap();
    session
        .create_node_with_props(&["Item"], [("val", Value::Int64(3))])
        .unwrap();

    let r = session
        .execute("MATCH (i:Item) RETURN count(i.val) AS cnt")
        .unwrap();
    assert_eq!(r.rows()[0][0], Value::Int64(2));
}

// ===========================================================================
// T2-01: HAVING clause tests
// ===========================================================================

#[test]
fn test_having_filters_groups() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for (name, city) in [
        ("Alix", "Amsterdam"),
        ("Gus", "Amsterdam"),
        ("Vincent", "Berlin"),
        ("Jules", "Berlin"),
        ("Mia", "Paris"),
    ] {
        session
            .create_node_with_props(
                &["Person"],
                [
                    ("name", Value::String(name.into())),
                    ("city", Value::String(city.into())),
                ],
            )
            .unwrap();
    }
    let r = session
        .execute(
            "MATCH (p:Person) \
             RETURN p.city AS city, count(p) AS cnt \
             ORDER BY city \
             HAVING cnt > 1",
        )
        .unwrap();
    // Only Amsterdam (2) and Berlin (2) qualify; Paris (1) is filtered out
    assert_eq!(
        r.rows().len(),
        2,
        "HAVING should filter groups with cnt <= 1"
    );
    assert_eq!(r.rows()[0][0], Value::String("Amsterdam".into()));
    assert_eq!(r.rows()[0][1], Value::Int64(2));
    assert_eq!(r.rows()[1][0], Value::String("Berlin".into()));
    assert_eq!(r.rows()[1][1], Value::Int64(2));
}

#[test]
fn test_having_no_matching_groups() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for city in ["Amsterdam", "Berlin", "Paris"] {
        session
            .create_node_with_props(&["City"], [("name", Value::String(city.into()))])
            .unwrap();
    }
    let r = session
        .execute(
            "MATCH (c:City) \
             RETURN c.name AS name, count(c) AS cnt \
             HAVING cnt > 10",
        )
        .unwrap();
    // All groups have cnt=1, none pass HAVING cnt > 10
    assert_eq!(r.rows().len(), 0, "No groups should pass HAVING cnt > 10");
}

#[test]
fn test_having_with_sum_aggregate() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for (name, dept, salary) in [
        ("Alix", "Engineering", 90),
        ("Gus", "Engineering", 80),
        ("Vincent", "Sales", 50),
        ("Jules", "Sales", 60),
        ("Mia", "Marketing", 70),
    ] {
        session
            .create_node_with_props(
                &["Employee"],
                [
                    ("name", Value::String(name.into())),
                    ("dept", Value::String(dept.into())),
                    ("salary", Value::Int64(salary)),
                ],
            )
            .unwrap();
    }
    let r = session
        .execute(
            "MATCH (e:Employee) \
             RETURN e.dept AS dept, sum(e.salary) AS total \
             ORDER BY dept \
             HAVING total > 100",
        )
        .unwrap();
    // Engineering: 170, Sales: 110, Marketing: 70
    // Only Engineering and Sales qualify
    assert_eq!(
        r.rows().len(),
        2,
        "Only depts with total > 100 should appear"
    );
    assert_eq!(r.rows()[0][0], Value::String("Engineering".into()));
    assert_eq!(r.rows()[1][0], Value::String("Sales".into()));
}

// ============================================================================
// Global aggregation on empty result set (pull-based path)
// ============================================================================

#[test]
fn count_on_empty_result_returns_zero() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let r = session
        .execute("MATCH (n:NonExistent) RETURN count(n) AS cnt")
        .unwrap();
    // Global COUNT on empty input should return one row with 0
    assert_eq!(
        r.rows().len(),
        1,
        "Global COUNT should always return one row"
    );
    assert_eq!(r.rows()[0][0], Value::Int64(0));
}

#[test]
fn sum_on_empty_result_returns_null() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let r = session
        .execute("MATCH (n:NonExistent) RETURN sum(n.x) AS total")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(r.rows()[0][0], Value::Null);
}

#[test]
fn avg_on_empty_result_returns_null() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let r = session
        .execute("MATCH (n:NonExistent) RETURN avg(n.x) AS average")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(r.rows()[0][0], Value::Null);
}

#[test]
fn min_max_on_empty_result_returns_null() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let r = session
        .execute("MATCH (n:NonExistent) RETURN min(n.x) AS lo, max(n.x) AS hi")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(r.rows()[0][0], Value::Null);
    assert_eq!(r.rows()[0][1], Value::Null);
}

// ===========================================================================
// T3: Wrapped aggregates with CASE expressions
// Exercises: extract_wrapped_aggregates CASE branch in gql/aggregate.rs
// ===========================================================================

#[test]
fn test_case_wrapping_aggregate() {
    let db = stats_graph();
    let s = db.session();
    // CASE WHEN count(d) > 3 THEN 'many' ELSE 'few' END
    let r = s
        .execute(
            "MATCH (d:Data) \
             RETURN CASE WHEN count(d) > 3 THEN 'many' ELSE 'few' END AS label",
        )
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(r.rows()[0][0], Value::String("many".into()));
}

#[test]
fn test_case_wrapping_aggregate_else_branch() {
    let db = stats_graph();
    let s = db.session();
    // CASE WHEN sum(d.score) > 9999 THEN 'huge' ELSE 'normal' END
    let r = s
        .execute(
            "MATCH (d:Data) \
             RETURN CASE WHEN sum(d.score) > 9999 THEN 'huge' ELSE 'normal' END AS label",
        )
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(r.rows()[0][0], Value::String("normal".into()));
}

#[test]
fn test_case_aggregate_in_then_branch() {
    let db = stats_graph();
    let s = db.session();
    // CASE WHEN true THEN count(d) ELSE 0 END
    let r = s
        .execute(
            "MATCH (d:Data) \
             RETURN CASE WHEN true THEN count(d) ELSE 0 END AS cnt",
        )
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(r.rows()[0][0], Value::Int64(5));
}

// ===========================================================================
// T3b: Cypher-specific CASE + aggregate (exercises extract_wrapped_aggregates
// CASE/Binary branches via the Cypher translator)
// ===========================================================================

/// Aggregate in WHEN condition — result depends on actual aggregate value.
/// Before the fix, sum() was not substituted and evaluated to null, so
/// null > 0 → null → ELSE taken incorrectly.
#[cfg(feature = "cypher")]
#[test]
fn test_cypher_case_when_aggregate_condition_true() {
    let db = stats_graph();
    let s = db.session();
    // sum(d.score) = 150, 150 > 0 is true → 'positive'
    let r = s
        .execute_cypher(
            "MATCH (d:Data) \
             RETURN CASE WHEN sum(d.score) > 0 THEN 'positive' ELSE 'zero' END AS label",
        )
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(r.rows()[0][0], Value::String("positive".into()));
}

/// Aggregate value in THEN branch — THEN must evaluate to the aggregate, not null.
#[cfg(feature = "cypher")]
#[test]
fn test_cypher_case_then_aggregate_value() {
    let db = stats_graph();
    let s = db.session();
    // CASE WHEN true THEN sum(d.score) ELSE -1 END → 150
    let r = s
        .execute_cypher(
            "MATCH (d:Data) \
             RETURN CASE WHEN true THEN sum(d.score) ELSE -1 END AS total",
        )
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(r.rows()[0][0], Value::Int64(150));
}

/// Multiple aggregates in ELSE branch (sum/count division).
/// Before the fix, the Binary handler only extracted one aggregate from
/// a two-aggregate expression, and the CASE handler discarded the substitute.
#[cfg(feature = "cypher")]
#[test]
fn test_cypher_case_else_aggregate_division() {
    let db = stats_graph();
    let s = db.session();
    // ELSE sum(d.score) / count(d) = 150 / 5 = 30
    let r = s
        .execute_cypher(
            "MATCH (d:Data) \
             RETURN CASE WHEN false THEN -1 ELSE sum(d.score) / count(d) END AS avg_score",
        )
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(r.rows()[0][0], Value::Int64(30));
}

// ===========================================================================
// T4: Non-aggregate function wrapping an aggregate argument
// Exercises: extract_wrapped_aggregates FunctionCall non-aggregate branch
// ===========================================================================

#[test]
fn test_non_aggregate_function_wrapping_aggregate() {
    let db = stats_graph();
    let s = db.session();
    // abs(sum(d.score)) wraps aggregate inside non-aggregate abs()
    let r = s
        .execute("MATCH (d:Data) RETURN abs(sum(d.score)) AS total")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    // sum(10+20+30+40+50) = 150
    assert_eq!(r.rows()[0][0], Value::Int64(150));
}

#[test]
fn test_tostring_wrapping_count() {
    let db = stats_graph();
    let s = db.session();
    let r = s
        .execute("MATCH (d:Data) RETURN toString(count(d)) AS cnt_str")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(r.rows()[0][0], Value::String("5".into()));
}

// ===========================================================================
// T5: Bivariate aggregates on empty result set (null finalize paths)
// Exercises: Bivariate finalize with count=0 in aggregate.rs
// ===========================================================================

#[test]
fn test_covar_samp_empty_returns_null() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();
    let r = s
        .execute("MATCH (p:NonExistent) RETURN COVAR_SAMP(p.y, p.x) AS cov")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(r.rows()[0][0], Value::Null);
}

#[test]
fn test_corr_empty_returns_null() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();
    let r = s
        .execute("MATCH (p:NonExistent) RETURN CORR(p.y, p.x) AS r")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(r.rows()[0][0], Value::Null);
}

#[test]
fn test_regr_slope_empty_returns_null() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();
    let r = s
        .execute("MATCH (p:NonExistent) RETURN REGR_SLOPE(p.y, p.x) AS slope")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(r.rows()[0][0], Value::Null);
}

#[test]
fn test_regr_intercept_empty_returns_null() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();
    let r = s
        .execute("MATCH (p:NonExistent) RETURN REGR_INTERCEPT(p.y, p.x) AS intercept")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(r.rows()[0][0], Value::Null);
}

#[test]
fn test_regr_r2_empty_returns_null() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();
    let r = s
        .execute("MATCH (p:NonExistent) RETURN REGR_R2(p.y, p.x) AS r2")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(r.rows()[0][0], Value::Null);
}

#[test]
fn test_regr_sxx_syy_sxy_empty_returns_null() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();
    let r = s
        .execute(
            "MATCH (p:NonExistent) \
             RETURN REGR_SXX(p.y, p.x) AS sxx, REGR_SYY(p.y, p.x) AS syy, REGR_SXY(p.y, p.x) AS sxy",
        )
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(r.rows()[0][0], Value::Null);
    assert_eq!(r.rows()[0][1], Value::Null);
    assert_eq!(r.rows()[0][2], Value::Null);
}

#[test]
fn test_regr_avgx_avgy_empty_returns_null() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();
    let r = s
        .execute(
            "MATCH (p:NonExistent) \
             RETURN REGR_AVGX(p.y, p.x) AS ax, REGR_AVGY(p.y, p.x) AS ay",
        )
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(r.rows()[0][0], Value::Null);
    assert_eq!(r.rows()[0][1], Value::Null);
}

#[test]
fn test_regr_count_empty_returns_zero() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();
    let r = s
        .execute("MATCH (p:NonExistent) RETURN REGR_COUNT(p.y, p.x) AS cnt")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    // REGR_COUNT returns the count of non-null pairs (0 for empty set)
    assert_eq!(r.rows()[0][0], Value::Int64(0));
}

// ===========================================================================
// T6: GROUP_CONCAT DISTINCT
// Exercises: GroupConcatDistinct update/finalize in aggregate.rs
// ===========================================================================

#[test]
fn test_group_concat_distinct() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for city in ["Amsterdam", "Berlin", "Amsterdam", "Berlin", "Paris"] {
        session
            .create_node_with_props(&["City"], [("name", Value::String(city.into()))])
            .unwrap();
    }
    let r = session
        .execute("MATCH (c:City) RETURN group_concat(DISTINCT c.name) AS names")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    if let Value::String(names) = &r.rows()[0][0] {
        // Should have exactly 3 unique cities
        assert_eq!(
            names.split(' ').count(),
            3,
            "expected 3 unique cities: {names}"
        );
    } else {
        panic!("expected string, got {:?}", r.rows()[0][0]);
    }
}

#[test]
fn test_group_concat_distinct_with_separator() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for city in ["Amsterdam", "Berlin", "Amsterdam", "Berlin", "Paris"] {
        session
            .create_node_with_props(&["City"], [("name", Value::String(city.into()))])
            .unwrap();
    }
    let r = session
        .execute("MATCH (c:City) RETURN group_concat(DISTINCT c.name, '|') AS names")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    if let Value::String(names) = &r.rows()[0][0] {
        assert_eq!(
            names.split('|').count(),
            3,
            "expected 3 unique cities: {names}"
        );
    } else {
        panic!("expected string, got {:?}", r.rows()[0][0]);
    }
}

// ===========================================================================
// T7: SUM DISTINCT with mixed int and float (SumIntDistinct -> SumFloatDistinct)
// Exercises: SumIntDistinct float conversion path in aggregate.rs
// ===========================================================================

#[test]
fn test_sum_distinct_mixed_int_float() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    // Insert nodes with Int64 and Float64 values; duplicates should be excluded
    session
        .create_node_with_props(&["Val"], [("v", Value::Int64(10))])
        .unwrap();
    session
        .create_node_with_props(&["Val"], [("v", Value::Int64(10))])
        .unwrap(); // duplicate
    session
        .create_node_with_props(&["Val"], [("v", Value::Float64(20.5))])
        .unwrap();
    session
        .create_node_with_props(&["Val"], [("v", Value::Float64(20.5))])
        .unwrap(); // duplicate

    let r = session
        .execute("MATCH (n:Val) RETURN sum(DISTINCT n.v) AS total")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    // 10 + 20.5 = 30.5 (converted to float when float encountered)
    match &r.rows()[0][0] {
        Value::Float64(f) => assert!((*f - 30.5).abs() < 0.01, "expected 30.5, got {f}"),
        other => panic!("expected Float64, got {other:?}"),
    }
}

// ===========================================================================
// T8: Wrapped aggregate with multiple aggregates in one expression
// Exercises: extract_wrapped_aggregates recursion with multiple aggregates
// ===========================================================================

#[test]
fn test_wrapped_two_aggregates_in_binary() {
    let db = stats_graph();
    let s = db.session();
    // sum(d.score) + count(d): both sides contain aggregates
    let r = s
        .execute("MATCH (d:Data) RETURN sum(d.score) + count(d) AS combo")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    // sum = 150, count = 5 -> 155
    assert_eq!(r.rows()[0][0], Value::Int64(155));
}

#[test]
fn test_wrapped_aggregate_multiply() {
    let db = stats_graph();
    let s = db.session();
    // avg(d.score) * 2: float multiplication
    let r = s
        .execute("MATCH (d:Data) RETURN avg(d.score) * 2 AS doubled")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    // avg(10,20,30,40,50) = 30.0, * 2 = 60.0
    match &r.rows()[0][0] {
        Value::Float64(f) => assert!((*f - 60.0).abs() < 0.01, "expected 60.0, got {f}"),
        Value::Int64(v) => assert_eq!(*v, 60),
        other => panic!("expected numeric, got {other:?}"),
    }
}

// ===========================================================================
// T9: Bivariate aggregates with insufficient data (single row)
// Exercises: Bivariate finalize with count=1 (null for sample-based stats)
// ===========================================================================

#[test]
fn test_covar_samp_single_row_returns_null() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();
    s.create_node_with_props(
        &["Point"],
        [("x", Value::Float64(1.0)), ("y", Value::Float64(2.0))],
    )
    .unwrap();
    let r = s
        .execute("MATCH (p:Point) RETURN COVAR_SAMP(p.y, p.x) AS cov")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    // Sample covariance needs n >= 2
    assert_eq!(r.rows()[0][0], Value::Null);
}

#[test]
fn test_covar_pop_single_row_returns_zero() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();
    s.create_node_with_props(
        &["Point"],
        [("x", Value::Float64(1.0)), ("y", Value::Float64(2.0))],
    )
    .unwrap();
    let r = s
        .execute("MATCH (p:Point) RETURN COVAR_POP(p.y, p.x) AS cov")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    // Population covariance with n=1: c_xy/1 = 0/1 = 0.0
    match &r.rows()[0][0] {
        Value::Float64(f) => assert!(f.abs() < 1e-10, "expected 0.0, got {f}"),
        other => panic!("expected Float64, got {other:?}"),
    }
}

// ===========================================================================
// T10: OPTIONAL MATCH with aggregation (project.rs Expression path)
// ===========================================================================

#[test]
fn test_optional_match_with_count_aggregate() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();
    s.create_node_with_props(&["Person"], [("name", Value::String("Alix".into()))])
        .unwrap();
    s.create_node_with_props(&["Person"], [("name", Value::String("Gus".into()))])
        .unwrap();
    // Alix has a friend, Gus does not
    s.execute(
        "MATCH (a:Person {name: 'Alix'}) MATCH (b:Person {name: 'Gus'}) INSERT (a)-[:KNOWS]->(b)",
    )
    .unwrap();

    let r = s
        .execute(
            "MATCH (p:Person) \
             OPTIONAL MATCH (p)-[:KNOWS]->(f:Person) \
             RETURN p.name AS name, count(f) AS friend_count \
             ORDER BY name",
        )
        .unwrap();
    assert_eq!(r.rows().len(), 2);
    // Alix knows Gus
    assert_eq!(r.rows()[0][0], Value::String("Alix".into()));
    assert_eq!(r.rows()[0][1], Value::Int64(1));
    // Gus knows nobody
    assert_eq!(r.rows()[1][0], Value::String("Gus".into()));
    assert_eq!(r.rows()[1][1], Value::Int64(0));
}

// ===========================================================================
// T11: Percentile with grouped data
// Exercises: percentile finalize interpolation path in aggregate.rs
// ===========================================================================

#[test]
fn test_percentile_cont_interpolation() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();
    // Create 4 values: 10, 20, 30, 40
    for v in [10, 20, 30, 40] {
        s.create_node_with_props(&["Item"], [("v", Value::Int64(v))])
            .unwrap();
    }
    // percentile_cont at 0.5 should interpolate between 20 and 30 = 25.0
    let r = s
        .execute("MATCH (i:Item) RETURN percentile_cont(i.v, 0.5) AS median")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    match &r.rows()[0][0] {
        Value::Float64(f) => assert!((*f - 25.0).abs() < 0.01, "expected 25.0, got {f}"),
        other => panic!("expected Float64, got {other:?}"),
    }
}

#[test]
fn test_percentile_cont_at_boundary() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();
    for v in [10, 20, 30, 40, 50] {
        s.create_node_with_props(&["Item"], [("v", Value::Int64(v))])
            .unwrap();
    }
    // percentile_cont at 0.25: rank = 0.25 * 4 = 1.0, so exact value at index 1 = 20.0
    let r = s
        .execute("MATCH (i:Item) RETURN percentile_cont(i.v, 0.25) AS p25")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    match &r.rows()[0][0] {
        Value::Float64(f) => assert!((*f - 20.0).abs() < 0.01, "expected 20.0, got {f}"),
        other => panic!("expected Float64, got {other:?}"),
    }
}

// ===========================================================================
// T12: Unary negation wrapping aggregate (Neg branch in extract_wrapped_aggregates)
// ===========================================================================

#[test]
fn test_wrapped_aggregate_unary_negation() {
    let db = stats_graph();
    let s = db.session();
    // -(count(d)): unary negation wrapping aggregate
    let r = s
        .execute("MATCH (d:Data) RETURN -(count(d)) AS neg")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(r.rows()[0][0], Value::Int64(-5));
}

// ===========================================================================
// T13: COLLECT DISTINCT
// Exercises: CollectDistinct update/finalize in aggregate.rs
// ===========================================================================

#[test]
fn test_collect_distinct() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();
    for city in ["Amsterdam", "Berlin", "Amsterdam", "Berlin", "Paris"] {
        s.create_node_with_props(&["City"], [("name", Value::String(city.into()))])
            .unwrap();
    }
    let r = s
        .execute("MATCH (c:City) RETURN collect(DISTINCT c.name) AS names")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    if let Value::List(items) = &r.rows()[0][0] {
        assert_eq!(
            items.len(),
            3,
            "expected 3 unique cities, got {}",
            items.len()
        );
    } else {
        panic!("expected list, got {:?}", r.rows()[0][0]);
    }
}

// ===========================================================================
// T14: AVG DISTINCT
// Exercises: AvgDistinct update/finalize in aggregate.rs
// ===========================================================================

#[test]
fn test_avg_distinct() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();
    // Values: 10, 10, 20, 20, 30 -> distinct: 10, 20, 30 -> avg = 20.0
    for v in [10, 10, 20, 20, 30] {
        s.create_node_with_props(&["Val"], [("v", Value::Int64(v))])
            .unwrap();
    }
    let r = s
        .execute("MATCH (n:Val) RETURN avg(DISTINCT n.v) AS a")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    match &r.rows()[0][0] {
        Value::Float64(f) => assert!((*f - 20.0).abs() < 0.01, "expected 20.0, got {f}"),
        other => panic!("expected Float64, got {other:?}"),
    }
}
