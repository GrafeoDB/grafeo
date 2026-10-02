//! Coverage tests targeted at `query/planner/lpg/project.rs`.
//!
//! These tests exercise branches in `plan_return`, `plan_return_projection`,
//! `plan_project`, `plan_sort`, `plan_limit`, and `plan_skip` that are not
//! already covered by `planner_coverage.rs` or `expression_and_projection.rs`.
//!
//! Focus areas:
//!   * RETURN arithmetic / comparison / aggregate (Binary branch)
//!   * RETURN of constants and functions that hit the generic expression arm
//!   * RETURN DISTINCT on computed expressions
//!   * ORDER BY NULLS FIRST / NULLS LAST
//!   * ORDER BY on a variable that is not present in the RETURN items
//!     (forces the augmented-Return path and the extra-column stripping)
//!   * ORDER BY alias
//!   * LIMIT 0 and SKIP combined with LIMIT
//!   * `length(p)`, `nodes(p)`, `edges(p)` on a path variable
//!   * `length(n.name)` on a non-path variable (falls through to expression)
//!
//! ```bash
//! cargo test -p grafeo-engine --test project_coverage --all-features
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// 5 Person nodes with Tarantino names in European cities.
/// Some nodes have `age=NULL` to exercise NULLS FIRST/LAST ordering.
fn people_graph() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // Alix: age 30, Amsterdam
    let _alix = session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Alix".into())),
                ("age", Value::Int64(30)),
                ("city", Value::String("Amsterdam".into())),
            ],
        )
        .unwrap();
    // Gus: age 25, Berlin
    let _gus = session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Gus".into())),
                ("age", Value::Int64(25)),
                ("city", Value::String("Berlin".into())),
            ],
        )
        .unwrap();
    // Vincent: age 40, Paris
    let _vincent = session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Vincent".into())),
                ("age", Value::Int64(40)),
                ("city", Value::String("Paris".into())),
            ],
        )
        .unwrap();
    // Jules: no age, Prague (exercises NULL ordering)
    let _jules = session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Jules".into())),
                ("city", Value::String("Prague".into())),
            ],
        )
        .unwrap();
    // Mia: no age, Amsterdam (duplicate city for DISTINCT)
    let _mia = session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Mia".into())),
                ("city", Value::String("Amsterdam".into())),
            ],
        )
        .unwrap();

    db
}

/// 5-node chain: A-B-C-D-E linked with KNOWS edges. Useful for `length(p)`,
/// `nodes(p)`, `edges(p)` tests on variable-length paths.
fn chain_graph() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let alix = session
        .create_node_with_props(&["Person"], [("name", Value::String("Alix".into()))])
        .unwrap();
    let gus = session
        .create_node_with_props(&["Person"], [("name", Value::String("Gus".into()))])
        .unwrap();
    let vincent = session
        .create_node_with_props(&["Person"], [("name", Value::String("Vincent".into()))])
        .unwrap();
    let jules = session
        .create_node_with_props(&["Person"], [("name", Value::String("Jules".into()))])
        .unwrap();
    let mia = session
        .create_node_with_props(&["Person"], [("name", Value::String("Mia".into()))])
        .unwrap();

    session.create_edge(alix, gus, "KNOWS").unwrap();
    session.create_edge(gus, vincent, "KNOWS").unwrap();
    session.create_edge(vincent, jules, "KNOWS").unwrap();
    session.create_edge(jules, mia, "KNOWS").unwrap();

    db
}

fn int_col(r: &[Value], idx: usize) -> Option<i64> {
    match &r[idx] {
        Value::Int64(i) => Some(*i),
        _ => None,
    }
}

fn string_col(r: &[Value], idx: usize) -> Option<String> {
    match &r[idx] {
        Value::String(s) => Some(s.to_string()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// RETURN arithmetic: hits the Binary catch-all arm in plan_return_projection
// ---------------------------------------------------------------------------

#[test]
fn return_arithmetic_multiple_ops() {
    // Table-driven over + - * / : all four land in the Binary arm, converted
    // to Expression via convert_expression.
    let db = people_graph();
    let session = db.session();

    let cases: &[(&str, i64)] = &[
        ("MATCH (n:Person {name:'Alix'}) RETURN n.age + 10 AS v", 40),
        ("MATCH (n:Person {name:'Alix'}) RETURN n.age - 5 AS v", 25),
        ("MATCH (n:Person {name:'Alix'}) RETURN n.age * 2 AS v", 60),
        ("MATCH (n:Person {name:'Alix'}) RETURN n.age / 3 AS v", 10),
    ];

    for (query, expected) in cases {
        let r = session.execute(query).unwrap();
        assert_eq!(r.rows().len(), 1, "query: {query}");
        assert_eq!(int_col(&r.rows()[0], 0), Some(*expected), "query: {query}");
    }
}

// ---------------------------------------------------------------------------
// RETURN comparison: exercises Binary with boolean result
// ---------------------------------------------------------------------------

#[test]
fn return_comparison_expressions() {
    let db = people_graph();
    let session = db.session();

    let r = session
        .execute(
            "MATCH (n:Person) WHERE n.age IS NOT NULL \
             RETURN n.name AS name, n.age > 30 AS is_senior ORDER BY name",
        )
        .unwrap();
    assert_eq!(r.rows().len(), 3);
    // Alix(30) -> false, Gus(25) -> false, Vincent(40) -> true
    assert_eq!(r.rows()[0][1], Value::Bool(false), "Alix");
    assert_eq!(r.rows()[1][1], Value::Bool(false), "Gus");
    assert_eq!(r.rows()[2][1], Value::Bool(true), "Vincent");
}

// ---------------------------------------------------------------------------
// RETURN with constants and literals: exercises the Literal arm
// ---------------------------------------------------------------------------

#[test]
fn return_constant_literal_alongside_property() {
    let db = people_graph();
    let session = db.session();

    let r = session
        .execute(
            "MATCH (n:Person {name:'Alix'}) \
             RETURN n.name AS name, 42 AS answer, 'hello' AS greeting",
        )
        .unwrap();

    assert_eq!(r.rows().len(), 1);
    assert_eq!(string_col(&r.rows()[0], 0).as_deref(), Some("Alix"));
    assert_eq!(r.rows()[0][1], Value::Int64(42));
    assert_eq!(r.rows()[0][2], Value::String("hello".into()));
}

// ---------------------------------------------------------------------------
// RETURN aggregates: count/sum/avg/min/max over Person.age
// ---------------------------------------------------------------------------

#[test]
fn return_aggregates_count_sum_avg_min_max() {
    let db = people_graph();
    let session = db.session();

    // 3 non-null ages: 30, 25, 40 (Alix, Gus, Vincent). Jules/Mia null.
    let r = session
        .execute(
            "MATCH (n:Person) RETURN \
             count(n) AS c, \
             sum(n.age) AS s, \
             min(n.age) AS lo, \
             max(n.age) AS hi",
        )
        .unwrap();

    assert_eq!(r.rows().len(), 1);
    let row = &r.rows()[0];
    assert_eq!(int_col(row, 0), Some(5), "count counts all 5 rows");
    assert_eq!(int_col(row, 1), Some(95), "sum of 30+25+40");
    assert_eq!(int_col(row, 2), Some(25), "min age");
    assert_eq!(int_col(row, 3), Some(40), "max age");

    // avg must return a float. Accepting Int64 here would hide an
    // integer-truncation bug (95/3 = 31.666..., so a broken avg could
    // truncate to 31 and still look plausible).
    let r = session
        .execute("MATCH (n:Person) RETURN avg(n.age) AS a")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    match &r.rows()[0][0] {
        Value::Float64(f) => assert!(
            (f - (95.0 / 3.0)).abs() < 1e-9,
            "avg(age) must be exactly 95/3, got {f}"
        ),
        other => {
            panic!("avg must return Float64 (integer truncation would be incorrect), got {other:?}")
        }
    }
}

// ---------------------------------------------------------------------------
// RETURN DISTINCT on a computed arithmetic expression
// Hits: needs_project=true path, then build_distinct
// ---------------------------------------------------------------------------

#[test]
fn return_distinct_on_arithmetic_expression() {
    let db = people_graph();
    let session = db.session();

    // Alix/Gus/Vincent have age 30/25/40; bucket = age / 10 -> 3, 2, 4 -> 3 distinct
    let r = session
        .execute(
            "MATCH (n:Person) WHERE n.age IS NOT NULL \
             RETURN DISTINCT n.age / 10 AS bucket ORDER BY bucket",
        )
        .unwrap();

    assert_eq!(r.rows().len(), 3, "expected 3 distinct buckets");
    assert_eq!(int_col(&r.rows()[0], 0), Some(2), "Gus/25 -> bucket 2");
    assert_eq!(int_col(&r.rows()[1], 0), Some(3), "Alix/30 -> bucket 3");
    assert_eq!(int_col(&r.rows()[2], 0), Some(4), "Vincent/40 -> bucket 4");
}

// ---------------------------------------------------------------------------
// ORDER BY NULLS FIRST / NULLS LAST
// Hits: NullOrder::NullsFirst / NullsLast branches in plan_sort
// ---------------------------------------------------------------------------

#[test]
fn order_by_nulls_first_puts_nulls_at_top() {
    let db = people_graph();
    let session = db.session();

    // Ages: 30,25,40,null,null. NULLS FIRST on ASC -> nulls first.
    let r = session
        .execute(
            "MATCH (n:Person) RETURN n.name AS name, n.age AS age ORDER BY age ASC NULLS FIRST",
        )
        .unwrap();

    assert_eq!(r.rows().len(), 5);
    // First two rows should have null age
    assert!(matches!(r.rows()[0][1], Value::Null), "row 0 age is null");
    assert!(matches!(r.rows()[1][1], Value::Null), "row 1 age is null");
    // Next rows must be ascending non-null
    assert_eq!(int_col(&r.rows()[2], 1), Some(25), "Gus next");
    assert_eq!(int_col(&r.rows()[3], 1), Some(30), "Alix next");
    assert_eq!(int_col(&r.rows()[4], 1), Some(40), "Vincent last");
}

#[test]
fn order_by_nulls_last_puts_nulls_at_bottom() {
    let db = people_graph();
    let session = db.session();

    let r = session
        .execute("MATCH (n:Person) RETURN n.name AS name, n.age AS age ORDER BY age ASC NULLS LAST")
        .unwrap();

    assert_eq!(r.rows().len(), 5);
    assert_eq!(int_col(&r.rows()[0], 1), Some(25), "Gus first");
    assert_eq!(int_col(&r.rows()[1], 1), Some(30), "Alix");
    assert_eq!(int_col(&r.rows()[2], 1), Some(40), "Vincent");
    assert!(matches!(r.rows()[3][1], Value::Null));
    assert!(matches!(r.rows()[4][1], Value::Null));
}

#[test]
fn order_by_desc_with_nulls_ordering() {
    // An explicit NULLS clause holds in either direction: DESC NULLS LAST puts
    // the nulls after the values, DESC NULLS FIRST before them. (The sort used
    // to reverse the null position along with the values for DESC, so both
    // came out the other way around.) Pin the full row ordering, not just the
    // non-null subsequence.
    let db = people_graph();
    let session = db.session();
    let ages = |order: &str| -> Vec<Value> {
        session
            .execute(&format!(
                "MATCH (n:Person) RETURN n.name AS name, n.age AS age ORDER BY age {order}"
            ))
            .unwrap()
            .rows()
            .iter()
            .map(|row| row[1].clone())
            .collect()
    };

    assert_eq!(
        ages("DESC NULLS LAST"),
        vec![
            Value::Int64(40),
            Value::Int64(30),
            Value::Int64(25),
            Value::Null,
            Value::Null,
        ],
    );
    assert_eq!(
        ages("DESC NULLS FIRST"),
        vec![
            Value::Null,
            Value::Null,
            Value::Int64(40),
            Value::Int64(30),
            Value::Int64(25),
        ],
    );
}

// ---------------------------------------------------------------------------
// ORDER BY alias: sort key is the alias defined in RETURN
// ---------------------------------------------------------------------------

#[test]
fn order_by_alias_desc() {
    let db = people_graph();
    let session = db.session();

    // RETURN ... AS a ORDER BY a DESC: the alias column lookup path.
    let r = session
        .execute(
            "MATCH (n:Person) WHERE n.age IS NOT NULL \
             RETURN n.age AS a ORDER BY a DESC",
        )
        .unwrap();

    assert_eq!(r.rows().len(), 3);
    assert_eq!(int_col(&r.rows()[0], 0), Some(40));
    assert_eq!(int_col(&r.rows()[1], 0), Some(30));
    assert_eq!(int_col(&r.rows()[2], 0), Some(25));
}

// ---------------------------------------------------------------------------
// ORDER BY on variable not in RETURN: augmented-Return path + extra-column strip
// ---------------------------------------------------------------------------

#[test]
fn order_by_property_not_in_return_strips_extra_column() {
    let db = people_graph();
    let session = db.session();

    // n.age is not projected in RETURN, so plan_sort must inject an augmented
    // Return item, sort on it, then strip the extra column after sorting.
    let r = session
        .execute(
            "MATCH (n:Person) WHERE n.age IS NOT NULL \
             RETURN n.name ORDER BY n.age ASC",
        )
        .unwrap();

    assert_eq!(r.rows().len(), 3);
    // Output columns must still be only 1 (the extra sort column was stripped).
    assert_eq!(
        r.rows()[0].len(),
        1,
        "extra sort column should have been stripped, got {} cols",
        r.rows()[0].len()
    );
    assert_eq!(
        string_col(&r.rows()[0], 0).as_deref(),
        Some("Gus"),
        "age 25"
    );
    assert_eq!(
        string_col(&r.rows()[1], 0).as_deref(),
        Some("Alix"),
        "age 30"
    );
    assert_eq!(
        string_col(&r.rows()[2], 0).as_deref(),
        Some("Vincent"),
        "age 40"
    );
}

// ---------------------------------------------------------------------------
// LIMIT 0: edge case where no rows are returned
// ---------------------------------------------------------------------------

#[test]
fn limit_zero_returns_no_rows() {
    let db = people_graph();
    let session = db.session();

    let r = session
        .execute("MATCH (n:Person) RETURN n.name LIMIT 0")
        .unwrap();
    assert_eq!(r.rows().len(), 0, "LIMIT 0 yields empty result");
}

// ---------------------------------------------------------------------------
// SKIP + LIMIT combined: plan_skip into plan_limit
// ---------------------------------------------------------------------------

#[test]
fn skip_then_limit_windows_result() {
    let db = people_graph();
    let session = db.session();

    // 5 people sorted by name: Alix, Gus, Jules, Mia, Vincent
    // SKIP 1 LIMIT 2 -> Gus, Jules
    let r = session
        .execute("MATCH (n:Person) RETURN n.name AS name ORDER BY name SKIP 1 LIMIT 2")
        .unwrap();

    assert_eq!(r.rows().len(), 2);
    assert_eq!(string_col(&r.rows()[0], 0).as_deref(), Some("Gus"));
    assert_eq!(string_col(&r.rows()[1], 0).as_deref(), Some("Jules"));
}

#[test]
fn skip_beyond_result_set_yields_empty() {
    let db = people_graph();
    let session = db.session();

    // Only 5 rows; SKIP 10 should be empty.
    let r = session
        .execute("MATCH (n:Person) RETURN n.name SKIP 10")
        .unwrap();
    assert!(r.rows().is_empty());
}

// ---------------------------------------------------------------------------
// length(p) on a path variable: the path-detail variable branch
// ---------------------------------------------------------------------------

#[test]
fn length_on_path_variable() {
    let db = chain_graph();
    let session = db.session();

    // Path with 3 hops: A-B-C-D. length(p) should be 3.
    let r = session
        .execute(
            "MATCH p = (a:Person {name:'Alix'})-[:KNOWS *3..3]->(d:Person) \
             RETURN length(p) AS len",
        )
        .unwrap();

    assert_eq!(r.rows().len(), 1);
    assert_eq!(int_col(&r.rows()[0], 0), Some(3));
}

// ---------------------------------------------------------------------------
// nodes(p) / edges(p) on a path variable: path-detail column branch
// ---------------------------------------------------------------------------

#[test]
fn nodes_and_edges_on_path_variable() {
    let db = chain_graph();
    let session = db.session();

    // 2-hop path A->B->C: nodes list has 3 entries, edges list has 2.
    let r = session
        .execute(
            "MATCH p = (a:Person {name:'Alix'})-[:KNOWS *2..2]->(c:Person) \
             RETURN nodes(p) AS ns, edges(p) AS es",
        )
        .unwrap();

    assert_eq!(r.rows().len(), 1);
    match &r.rows()[0][0] {
        Value::List(ns) => assert_eq!(ns.len(), 3, "2-hop path has 3 nodes"),
        other => panic!("expected list for nodes(p), got {other:?}"),
    }
    match &r.rows()[0][1] {
        Value::List(es) => assert_eq!(es.len(), 2, "2-hop path has 2 edges"),
        other => panic!("expected list for edges(p), got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// length() on a non-variable argument: falls through to expression evaluation
// Covers the `else` branch inside the "length" match arm (lines ~185-194).
// ---------------------------------------------------------------------------

#[test]
fn length_on_string_falls_through_to_expression() {
    let db = people_graph();
    let session = db.session();

    // length(n.name) -> string length. The first arg is a Property access,
    // not a bare Variable, so the planner falls through to convert_expression.
    let r = session
        .execute("MATCH (n:Person {name:'Vincent'}) RETURN length(n.name) AS len")
        .unwrap();

    assert_eq!(r.rows().len(), 1);
    assert_eq!(int_col(&r.rows()[0], 0), Some(7), "len('Vincent') == 7");
}

// ---------------------------------------------------------------------------
// DISTINCT across multiple projected columns
// ---------------------------------------------------------------------------

#[test]
fn distinct_across_multiple_columns() {
    let db = people_graph();
    let session = db.session();

    // city/age_group pairs from the 5 people (age / 10 bucket, null -> null):
    //   Alix/Amsterdam/3, Gus/Berlin/2, Vincent/Paris/4, Jules/Prague/null,
    //   Mia/Amsterdam/null
    // DISTINCT (city, bucket): 5 distinct pairs.
    let r = session
        .execute(
            "MATCH (n:Person) \
             RETURN DISTINCT n.city AS city, n.age / 10 AS bucket \
             ORDER BY city",
        )
        .unwrap();

    assert_eq!(r.rows().len(), 5, "expected 5 distinct (city,bucket) pairs");
}

// ---------------------------------------------------------------------------
// RETURN type(r) / labels(n) mixed with arithmetic: exercises EdgeType arm
// alongside the catch-all expression arm.
// ---------------------------------------------------------------------------

#[test]
fn return_type_function_with_arithmetic() {
    let db = chain_graph();
    let session = db.session();

    let r = session
        .execute(
            "MATCH (a:Person {name:'Alix'})-[r:KNOWS]->(b) \
             RETURN type(r) AS t, 1 + 1 AS two",
        )
        .unwrap();

    assert_eq!(r.rows().len(), 1);
    assert_eq!(string_col(&r.rows()[0], 0).as_deref(), Some("KNOWS"));
    assert_eq!(int_col(&r.rows()[0], 1), Some(2));
}
