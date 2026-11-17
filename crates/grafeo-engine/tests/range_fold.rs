//! An aggregate over a label's range is folded in one store call (`PlannerRule::RangeFold`): the
//! label's rows in range hand their values straight to the aggregate's states, in row order,
//! without an id list or a second read by id. The answer is the range scan's and the aggregate's
//! (the rule off), value for value and float bit for float bit, an error the same error:
//!
//! - every value kind, each aggregate the fold takes, on the filtered property and another one;
//! - hot, reopened (cold chunks) and thawed;
//! - a label absent from a group, and one spanning two full groups and a partial one;
//! - an old snapshot over a changed value, a deleted row, a label added and one removed;
//! - empty answers, 22003 on an integer sum past 64 bits, an average past 2^53;
//! - a timeout stops the fold.
//!
//! Each comparison runs on one session, the rule toggled between the two runs, so both read at
//! the same snapshot. The plan shows `RangeFold` where the rule fires and never where it must not.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main range_fold::
//! ```

#![cfg(feature = "gql")]

use std::collections::HashMap;

use grafeo_common::types::{Date, Decimal, Value};
use grafeo_engine::GrafeoDB;
use grafeo_engine::query::optimizer::{PlannerRule, PlannerRules};
use grafeo_engine::session::Session;

/// Rows of a row group.
const GROUP: i64 = 65_536;
/// 2^53: above it the f64 running sum of an average drops bits.
const EXACT: i64 = 9_007_199_254_740_992;

/// A value as text that tells floats apart by their bits (NaN, -0.0).
fn exact(value: &Value) -> String {
    match value {
        Value::Float64(float) => format!("f64 {:#018x}", float.to_bits()),
        other => format!("{other:?}"),
    }
}

/// The rows of `query` on `session` with the rules `rules`, or its error, as text.
fn answer_with(
    session: &mut Session,
    rules: PlannerRules,
    query: &str,
    params: &HashMap<String, Value>,
    sql: bool,
) -> String {
    session.set_planner_rules(rules);
    let result = if sql {
        session.execute_sql_with_params(query, params.clone())
    } else {
        session.execute_with_params(query, params.clone())
    };
    match result {
        Ok(result) => result
            .rows()
            .iter()
            .map(|row| row.iter().map(exact).collect::<Vec<_>>().join(", "))
            .collect::<Vec<_>>()
            .join(" | "),
        Err(error) => format!("error: {error}"),
    }
}

/// `query` with the fold and without it, on one session: the two answers, fold first.
fn both(
    session: &mut Session,
    query: &str,
    params: &HashMap<String, Value>,
    sql: bool,
) -> (String, String) {
    let all = PlannerRules::all();
    let folded = answer_with(session, all, query, params, sql);
    let plain = answer_with(
        session,
        all.with(PlannerRule::RangeFold, false),
        query,
        params,
        sql,
    );
    session.set_planner_rules(all);
    (folded, plain)
}

fn bound(value: &Value) -> HashMap<String, Value> {
    HashMap::from([("v".to_string(), value.clone())])
}

/// The aggregates the fold takes, alone and together, on `k` (the filtered property) and `q`.
const AGGREGATES: &[&str] = &[
    "count(*)",
    "count(n)",
    "count(n.k)",
    "count(n.q)",
    "sum(n.k)",
    "avg(n.k)",
    "min(n.k)",
    "max(n.k)",
    "sum(n.q)",
    "avg(n.q)",
    "min(n.q)",
    "max(n.q)",
    "count(*), sum(n.k), avg(n.k), min(n.q), max(n.q)",
];

/// The one-sided bounds of each direction.
const COMPARISONS: &[&str] = &[">", ">=", "<", "<="];

fn columns() -> Vec<(&'static str, Vec<Value>)> {
    let integers: Vec<Value> = [
        0,
        1,
        -1,
        7,
        EXACT - 1,
        EXACT,
        EXACT + 1,
        i64::MAX,
        i64::MIN,
        42,
    ]
    .map(Value::Int64)
    .to_vec();
    let floats: Vec<Value> = [
        0.0,
        -0.0,
        1.5,
        -2.25,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        1e300,
        0.1,
        9_007_199_254_740_993.0,
    ]
    .map(Value::Float64)
    .to_vec();
    let decimals: Vec<Value> = ["0", "1.5", "-2.25", "12345678901234567890.5", "0.1"]
        .map(|text| Value::Decimal(Decimal::parse(text).unwrap()))
        .to_vec();
    let strings: Vec<Value> = ["Alix", "Gus", "", "Vincent", "Zürich", "10", "9"]
        .map(Value::from)
        .to_vec();
    let dates: Vec<Value> = ["2020-01-01", "1999-12-31", "2026-11-17"]
        .map(|text| Value::Date(Date::parse(text).unwrap()))
        .to_vec();
    let booleans = vec![Value::Bool(true), Value::Bool(false)];
    let numbers = [integers.clone(), floats.clone(), decimals.clone()].concat();
    let mixed = [
        integers.clone(),
        floats.clone(),
        strings.clone(),
        dates.clone(),
    ]
    .concat();
    vec![
        ("integers", integers),
        ("floats", floats),
        ("decimals", decimals),
        ("strings", strings),
        ("dates", dates),
        ("booleans", booleans),
        ("numbers", numbers),
        ("mixed kinds", mixed),
    ]
}

/// `values` as `k` on nodes of `L`, each with the next value as `q`; the same on nodes of `Other`;
/// a node of `L` without `k`, and one with `k` but no `q`.
fn hold(db: &GrafeoDB, values: &[Value]) {
    for (row, value) in values.iter().enumerate() {
        let next = values[(row + 1) % values.len()].clone();
        db.create_node_with_props(&["L"], [("k", value.clone()), ("q", next.clone())])
            .unwrap();
        db.create_node_with_props(&["Other"], [("k", value.clone()), ("q", next)])
            .unwrap();
    }
    db.create_node_with_props(&["L"], [("q", values[0].clone())])
        .unwrap();
    db.create_node_with_props(&["L"], [("k", values[0].clone())])
        .unwrap();
}

/// Every aggregate against every bound of every comparison, with the fold and without: the
/// differences, named.
fn differences(db: &GrafeoDB, stage: &str, values: &[Value]) -> Vec<String> {
    let mut session = db.session();
    let mut bounds = values.to_vec();
    bounds.push(Value::Null);
    let mut wrong = Vec::new();
    for comparison in COMPARISONS {
        for aggregate in AGGREGATES {
            let query = format!("MATCH (n:L) WHERE n.k {comparison} $v RETURN {aggregate}");
            for value in &bounds {
                let (folded, plain) = both(&mut session, &query, &bound(value), false);
                if folded != plain {
                    wrong.push(format!(
                        "{stage}: {query} with {value:?}\n  fold  {folded}\n  plain {plain}"
                    ));
                }
            }
        }
    }
    wrong
}

/// `check` on `values` held hot, reopened (its columns cold), and thawed by a write.
fn on_each_stage(values: &[Value], check: &mut impl FnMut(&GrafeoDB, &str)) {
    let hot = GrafeoDB::new_in_memory();
    hold(&hot, values);
    check(&hot, "hot");
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fold.grafeo");
    {
        let db = GrafeoDB::open(&path).unwrap();
        hold(&db, values);
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    check(&db, "reopened");
    db.close().unwrap();
    let db = GrafeoDB::open(&path).unwrap();
    let node = db
        .create_node_with_props(&["L"], [("k", Value::Int64(7)), ("q", Value::from("Gus"))])
        .unwrap();
    check(&db, "thawed");
    assert!(db.delete_node(node).unwrap());
    db.close().unwrap();
}

fn report(wrong: &[String]) {
    let shown: Vec<&str> = wrong.iter().take(40).map(String::as_str).collect();
    assert!(
        wrong.is_empty(),
        "{} differ, the first of them:\n{}",
        wrong.len(),
        shown.join("\n")
    );
}

/// The fold answers as the range scan and the aggregate do, for every kind, aggregate, bound and
/// stage.
#[test]
fn every_kind_and_aggregate_folds_to_the_same_answer() {
    let mut wrong = Vec::new();
    for (name, values) in columns() {
        on_each_stage(&values, &mut |db, stage| {
            wrong.extend(differences(db, &format!("{name}, {stage}"), &values));
        });
    }
    report(&wrong);
}

/// The SQL/PGQ twin, whose projection only renames the properties, folds to the same answer.
#[test]
fn the_sql_twin_folds_to_the_same_answer() {
    let db = GrafeoDB::new_in_memory();
    let values: Vec<Value> = (0..300).map(|at| Value::Int64(at * 7 % 101)).collect();
    hold(&db, &values);
    let mut session = db.session();
    let query = "SELECT count(*) AS n, avg(len) AS mean, sum(other) AS total \
                 FROM GRAPH_TABLE (MATCH (n:L) WHERE n.k > $v COLUMNS (n.k AS len, n.q AS other))";
    for value in [0, 50, 100, 200].map(Value::Int64) {
        let (folded, plain) = both(&mut session, query, &bound(&value), true);
        assert_eq!(folded, plain, "{value:?}");
    }
}

/// A label only later groups hold, over two full groups and a partial one: the fold walks the
/// groups that hold it, and answers as the scan does.
#[test]
fn a_label_across_full_groups_and_absent_from_one() {
    let db = GrafeoDB::new_in_memory();
    db.execute(&format!(
        "UNWIND range(1, {GROUP}) AS i INSERT (:Other {{k: i}})"
    ))
    .unwrap();
    db.execute(&format!(
        "UNWIND range(1, {}) AS i INSERT (:L {{k: i % 1000, q: i}})",
        2 * GROUP + 5_000
    ))
    .unwrap();
    let mut session = db.session();
    for query in [
        "MATCH (n:L) WHERE n.k > $v RETURN count(*), sum(n.q), avg(n.k), min(n.q), max(n.q)",
        "MATCH (n:L) WHERE n.k <= $v RETURN count(n), avg(n.q)",
    ] {
        for value in [0, 500, 998, 999].map(Value::Int64) {
            let (folded, plain) = both(&mut session, query, &bound(&value), false);
            assert_eq!(folded, plain, "{query} with {value:?}");
        }
    }
}

/// An old snapshot reads its own view through the fold: a changed value, a deleted row, a label
/// removed and one added after it began, under it and after it.
#[test]
fn an_old_snapshot_folds_what_it_saw() {
    let db = GrafeoDB::new_in_memory();
    for (k, name) in [(10, "Alix"), (20, "Gus"), (30, "Mia"), (40, "Jules")] {
        db.create_node_with_props(
            &["L"],
            [("k", Value::Int64(k)), ("name", Value::from(name))],
        )
        .unwrap();
    }
    db.create_node_with_props(
        &["M"],
        [("k", Value::Int64(25)), ("name", Value::from("Hans"))],
    )
    .unwrap();
    let query =
        "MATCH (n:L) WHERE n.k > $v RETURN count(*), sum(n.k), avg(n.k), min(n.name), max(n.name)";
    let mut reader = db.session();
    reader.begin_transaction().unwrap();
    let before = both(&mut reader, query, &bound(&Value::Int64(15)), false);
    let writer = db.session();
    writer
        .execute("MATCH (n:L {name: 'Gus'}) SET n.k = 5")
        .unwrap();
    writer
        .execute("MATCH (n:L {name: 'Mia'}) DETACH DELETE n")
        .unwrap();
    writer
        .execute("MATCH (n:L {name: 'Jules'}) REMOVE n:L")
        .unwrap();
    writer
        .execute("MATCH (n:M {name: 'Hans'}) SET n:L")
        .unwrap();
    let under = both(&mut reader, query, &bound(&Value::Int64(15)), false);
    assert_eq!(under.0, under.1, "fold and scan under the old snapshot");
    assert_eq!(under, before, "the old snapshot reads what it saw");
    reader.rollback().unwrap();
    let after = both(&mut reader, query, &bound(&Value::Int64(15)), false);
    assert_eq!(after.0, after.1, "fold and scan after it");
    assert_ne!(after.0, before.0, "the changes show after the snapshot");
}

/// Nothing in range: a count of 0 and null sums, averages, minima and maxima.
#[test]
fn an_empty_range_folds_to_zero_and_nulls() {
    let db = GrafeoDB::new_in_memory();
    hold(&db, &[Value::Int64(1), Value::Int64(2)]);
    let mut session = db.session();
    let query =
        "MATCH (n:L) WHERE n.k > $v RETURN count(*), sum(n.k), avg(n.k), min(n.q), max(n.q)";
    let (folded, plain) = both(&mut session, query, &bound(&Value::Int64(100)), false);
    assert_eq!(folded, plain);
    assert_eq!(folded, "Int64(0), Null, Null, Null, Null");
}

/// An integer sum past 64 bits is out of range (22003), with the fold as without; an average over
/// values whose f64 running sum drops bits keeps today's bits.
#[test]
fn an_overflowing_sum_and_an_average_past_2_to_the_53() {
    let db = GrafeoDB::new_in_memory();
    for k in [i64::MAX, 1, 2] {
        db.create_node_with_props(&["L"], [("k", Value::Int64(k)), ("q", Value::Int64(EXACT))])
            .unwrap();
    }
    for k in 3..40 {
        db.create_node_with_props(&["L"], [("k", Value::Int64(k)), ("q", Value::Int64(1))])
            .unwrap();
    }
    let mut session = db.session();
    let (folded, plain) = both(
        &mut session,
        "MATCH (n:L) WHERE n.k > $v RETURN sum(n.k)",
        &bound(&Value::Int64(0)),
        false,
    );
    assert_eq!(folded, plain);
    assert!(folded.contains("numeric value out of range"), "{folded}");
    let (folded, plain) = both(
        &mut session,
        "MATCH (n:L) WHERE n.k > $v RETURN avg(n.q), sum(n.q)",
        &bound(&Value::Int64(0)),
        false,
    );
    assert_eq!(folded, plain);
}

/// The fold checks the query's deadline as it walks: under the probe it stops with the timeout.
#[cfg(feature = "testing-deadline-probe")]
#[test]
fn a_timeout_stops_the_fold() {
    use grafeo_common::utils::error::{Error, QueryErrorKind};
    use grafeo_core::execution::deadline;

    let db = GrafeoDB::new_in_memory();
    db.execute("UNWIND range(1, 5000) AS i INSERT (:L {k: i})")
        .unwrap();
    let query = "MATCH (n:L) WHERE n.k > 10 RETURN count(*), avg(n.k)";
    assert!(plan(&db, query, false).contains("RangeFold"));
    let _probe = deadline::expire_after_polls(0);
    let error = db.execute(query).unwrap_err();
    assert!(
        matches!(&error, Error::Query(query) if query.kind == QueryErrorKind::Timeout),
        "{error}"
    );
}

fn plan(db: &GrafeoDB, query: &str, sql: bool) -> String {
    let explain = format!("EXPLAIN {query}");
    let result = if sql {
        db.execute_sql(&explain)
    } else {
        db.execute(&explain)
    };
    result.unwrap().rows()[0][0].as_str().unwrap().to_string()
}

/// The rule fires on an aggregate over a label's range, in GQL and in the SQL/PGQ twin, and on no
/// other shape: DISTINCT, grouping, an expression to aggregate, a filter that is not a range, no
/// label, a projection that computes, a write before the read, or with the rule off.
#[test]
fn the_plan_folds_only_what_it_may() {
    let db = GrafeoDB::new_in_memory();
    hold(&db, &[Value::Int64(1), Value::Int64(2), Value::Int64(3)]);
    for (query, sql) in [
        ("MATCH (n:L) WHERE n.k > 1 RETURN count(*), avg(n.k)", false),
        (
            "MATCH (n:L) WHERE n.k <= 2 RETURN min(n.q), max(n.q), sum(n.q)",
            false,
        ),
        (
            "SELECT count(*) AS n, avg(len) AS mean FROM GRAPH_TABLE (MATCH (n:L) WHERE n.k > 1 COLUMNS (n.k AS len))",
            true,
        ),
    ] {
        let text = plan(&db, query, sql);
        assert!(
            text.contains("RangeFold"),
            "{query}
{text}"
        );
    }
    for (query, sql) in [
        (
            "MATCH (n:L) WHERE n.k > 1 RETURN count(DISTINCT n.k)",
            false,
        ),
        ("MATCH (n:L) WHERE n.k > 1 RETURN n.q, count(*)", false),
        ("MATCH (n:L) WHERE n.k > 1 RETURN sum(n.k * 2)", false),
        ("MATCH (n:L) WHERE n.k = 1 RETURN count(*)", false),
        ("MATCH (n:L) WHERE n.k + 1 > 2 RETURN count(*)", false),
        ("MATCH (n) WHERE n.k > 1 RETURN count(*)", false),
        (
            "SELECT avg(len) AS mean FROM GRAPH_TABLE (MATCH (n:L) WHERE n.k > 1 COLUMNS (n.k + 1 AS len))",
            true,
        ),
        (
            "INSERT (:L {k: 9}) WITH 1 AS one MATCH (n:L) WHERE n.k > 1 RETURN count(*)",
            false,
        ),
    ] {
        let text = plan(&db, query, sql);
        assert!(
            !text.contains("RangeFold"),
            "{query}
{text}"
        );
    }
    let mut session = db.session();
    session.set_planner_rules(PlannerRules::all().with(PlannerRule::RangeFold, false));
    let off = session
        .execute("EXPLAIN MATCH (n:L) WHERE n.k > 1 RETURN count(*)")
        .unwrap();
    assert!(!off.rows()[0][0].as_str().unwrap().contains("RangeFold"));
}

/// A transaction's own earlier statement is folded with it: statement 1 writes an `:L` node in
/// range, a node out of range and a label added, and statement 2's fold, before the commit, counts
/// what the transaction wrote as the scan does; a reader outside it does not see them.
#[test]
fn a_transaction_folds_its_own_earlier_writes() {
    let db = GrafeoDB::new_in_memory();
    hold(&db, &[Value::Int64(1), Value::Int64(5), Value::Int64(8)]);
    let query =
        "MATCH (n:L) WHERE n.k > $v RETURN count(*), sum(n.k), avg(n.q), min(n.q), max(n.k)";
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:L {k: 9, q: 100}), (:L {k: 2, q: -1})")
        .unwrap();
    session.execute("MATCH (n:Other {k: 8}) SET n:L").unwrap();
    // The fold runs inside the transaction, its writes pending: PROFILE shows the operator ran.
    let profile = session
        .execute_with_params(&format!("PROFILE {query}"), bound(&Value::Int64(4)))
        .unwrap();
    let profile = profile.rows()[0][0].as_str().unwrap().to_string();
    assert!(profile.contains("RangeFold"), "{profile}");
    let (folded, plain) = both(&mut session, query, &bound(&Value::Int64(4)), false);
    assert_eq!(folded, plain, "fold and scan inside the transaction");
    assert!(
        folded.starts_with("Int64(4)"),
        "5 and 8 held, 9 and the relabelled 8 written: {folded}"
    );
    let mut outside = db.session();
    let (other, other_plain) = both(&mut outside, query, &bound(&Value::Int64(4)), false);
    assert_eq!(other, other_plain);
    assert!(other.starts_with("Int64(2)"), "{other}");
    session.commit().unwrap();
    let (after, after_plain) = both(&mut outside, query, &bound(&Value::Int64(4)), false);
    assert_eq!(after, after_plain);
    assert_eq!(
        after, folded,
        "after the commit everyone folds what it wrote"
    );
}

/// Each kind of write in an earlier statement of the same transaction is folded as the scan
/// reads it, before the commit: a node created in range, the range's property set into the range
/// and out of it, the aggregated property set and set to null, a node deleted, the label removed
/// and the label added. Each runs in a transaction of its own, rolled back; PROFILE shows the fold
/// ran, its answer is the rule off's, and it is not the answer before the write.
#[test]
fn each_kind_of_earlier_write_is_folded() {
    let db = GrafeoDB::new_in_memory();
    hold(&db, &[Value::Int64(1), Value::Int64(5), Value::Int64(8)]);
    let query =
        "MATCH (n:L) WHERE n.k > $v RETURN count(*), sum(n.k), avg(n.q), min(n.q), max(n.k)";
    let range = bound(&Value::Int64(4));
    let mut session = db.session();
    let (unwritten, unwritten_plain) = both(&mut session, query, &range, false);
    assert_eq!(unwritten, unwritten_plain);
    for write in [
        "INSERT (:L {k: 9, q: 100})",
        "MATCH (n:L {k: 1}) SET n.k = 6",
        "MATCH (n:L {k: 8}) SET n.k = 2",
        "MATCH (n:L {k: 5}) SET n.q = 50",
        "MATCH (n:L {k: 5}) SET n.q = null",
        "MATCH (n:L {k: 8}) DETACH DELETE n",
        "MATCH (n:L {k: 5}) REMOVE n:L",
        "MATCH (n:Other {k: 8}) SET n:L",
    ] {
        session.begin_transaction().unwrap();
        session.execute(write).unwrap();
        let profile = session
            .execute_with_params(&format!("PROFILE {query}"), range.clone())
            .unwrap();
        let profile = profile.rows()[0][0].as_str().unwrap().to_string();
        assert!(profile.contains("RangeFold"), "{write}:\n{profile}");
        let (folded, plain) = both(&mut session, query, &range, false);
        assert_eq!(folded, plain, "{write}");
        assert_ne!(folded, unwritten, "{write} shows in the fold");
        session.rollback().unwrap();
    }
}

/// Another transaction's uncommitted writes are not folded: while a writer holds a created node,
/// a value set into the range and a label added, open, a transaction that folds reads what was
/// committed, as the scan does.
#[test]
fn another_transactions_uncommitted_writes_are_not_folded() {
    let db = GrafeoDB::new_in_memory();
    hold(&db, &[Value::Int64(1), Value::Int64(5), Value::Int64(8)]);
    let query =
        "MATCH (n:L) WHERE n.k > $v RETURN count(*), sum(n.k), avg(n.q), min(n.q), max(n.k)";
    let range = bound(&Value::Int64(4));
    let mut reader = db.session();
    let (committed, _) = both(&mut reader, query, &range, false);
    let mut writer = db.session();
    writer.begin_transaction().unwrap();
    writer.execute("INSERT (:L {k: 9, q: 100})").unwrap();
    writer.execute("MATCH (n:L {k: 1}) SET n.k = 6").unwrap();
    writer.execute("MATCH (n:Other {k: 8}) SET n:L").unwrap();
    reader.begin_transaction().unwrap();
    let profile = reader
        .execute_with_params(&format!("PROFILE {query}"), range.clone())
        .unwrap();
    let profile = profile.rows()[0][0].as_str().unwrap().to_string();
    assert!(profile.contains("RangeFold"), "{profile}");
    let (folded, plain) = both(&mut reader, query, &range, false);
    assert_eq!(folded, plain);
    assert_eq!(folded, committed, "the writer's open writes are not seen");
    reader.rollback().unwrap();
    writer.rollback().unwrap();
}

/// A write before the read is counted: the rule does not fire, and the answer counts the new node,
/// with the rule on as off (each run in a transaction of its own, rolled back).
#[test]
fn a_write_before_the_read_is_counted() {
    let db = GrafeoDB::new_in_memory();
    hold(&db, &[Value::Int64(1), Value::Int64(5)]);
    let query = "INSERT (:L {k: 9}) WITH 1 AS one MATCH (n:L) WHERE n.k > $v RETURN count(*)";
    let mut session = db.session();
    let mut answer = |rules| {
        session.begin_transaction().unwrap();
        let found = answer_with(&mut session, rules, query, &bound(&Value::Int64(4)), false);
        session.rollback().unwrap();
        found
    };
    let folded = answer(PlannerRules::all());
    let plain = answer(PlannerRules::all().with(PlannerRule::RangeFold, false));
    assert_eq!(folded, plain);
    assert_eq!(folded, "Int64(2)", "the node with 5 and the new one");
}
