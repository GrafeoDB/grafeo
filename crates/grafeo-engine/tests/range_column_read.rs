//! A range on a key reads the key's column and answers as the filter it stands for.
//!
//! The answer is the evaluator's `<`, `<=`, `>` and `>=` for every kind of value and of bound, hot,
//! reopened and thawed; a reader sees its own view through value chains and label chains; a limit
//! gives the first rows; and `rows_scanned` says what was read. `range_column_read.gtest` runs the
//! query cases on every binding.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main range_column_read::
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use grafeo_common::types::{Date, Decimal, Duration, Time, Timestamp, Value, ZonedDatetime};
use grafeo_engine::query::optimizer::{PlannerRule, PlannerRules};
use grafeo_engine::session::Session;
use grafeo_engine::{Config, GrafeoDB, HistoryRetention};

/// Rows of a row group: a column past this spans groups.
const GROUP: i64 = 65_536;
/// 2^53: above it an integer and the float nearest it may differ.
const EXACT: i64 = 9_007_199_254_740_992;

fn integers() -> Vec<Value> {
    [
        0,
        1,
        -1,
        5,
        7,
        EXACT - 1,
        EXACT,
        EXACT + 1,
        EXACT + 2,
        -EXACT - 1,
        i64::MAX,
        i64::MAX - 1,
        i64::MIN,
        i64::MIN + 1,
    ]
    .map(Value::Int64)
    .to_vec()
}

fn floats() -> Vec<Value> {
    [
        0.0,
        -0.0,
        1.0,
        1.5,
        5.0,
        7.5,
        -7.5,
        9_007_199_254_740_992.0,
        9_007_199_254_740_994.0,
        9_223_372_036_854_775_808.0,
        -9_223_372_036_854_775_808.0,
        1e300,
        -1e300,
        f64::MIN_POSITIVE,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ]
    .map(Value::Float64)
    .to_vec()
}

fn strings() -> Vec<Value> {
    [
        "",
        "0",
        "-0",
        "5",
        "7",
        "042",
        "42.0",
        "4.2e1",
        " 7",
        "NaN",
        "inf",
        "Infinity",
        "Alix",
        "alix",
        "9007199254740993",
        "9007199254740992.0",
    ]
    .map(Value::from)
    .to_vec()
}

fn booleans() -> Vec<Value> {
    vec![Value::Bool(true), Value::Bool(false)]
}

fn decimals() -> Vec<Value> {
    ["0", "1.5", "5", "7.50", "-7.5", "9007199254740993"]
        .map(|text| Value::Decimal(Decimal::parse(text).unwrap()))
        .to_vec()
}

fn temporals() -> Vec<Value> {
    let instant = Timestamp::from_micros(1_696_500_000_000_000);
    let later = Timestamp::from_micros(1_696_500_000_000_001);
    let noon = Time::from_hms(12, 0, 0).unwrap();
    vec![
        Value::Date(Date::from_days(0)),
        Value::Date(Date::from_days(19_635)),
        Value::Timestamp(instant),
        Value::Timestamp(later),
        Value::ZonedDatetime(ZonedDatetime::from_timestamp_offset(instant, 7_200)),
        Value::ZonedDatetime(ZonedDatetime::from_timestamp_offset(later, -3_600)),
        Value::Time(noon),
        Value::Time(noon.with_offset(3_600)),
        Value::Duration(Duration::new(1, 0, 0)),
        Value::Duration(Duration::new(0, 30, 0)),
    ]
}

fn others() -> Vec<Value> {
    vec![
        Value::List(vec![Value::Int64(1)].into()),
        Value::List(vec![Value::Int64(2), Value::from("Gus")].into()),
        Value::Vector(vec![1.0_f32, 2.0].into()),
    ]
}

/// Every value a column holds and a bound may be, null among the bounds.
fn every_value() -> Vec<Value> {
    let mut values = integers();
    values.extend(floats());
    values.extend(strings());
    values.extend(booleans());
    values.extend(decimals());
    values.extend(temporals());
    values.extend(others());
    values
}

/// The columns a range must read alike: each kind alone, so it is read typed, and mixed.
fn columns() -> Vec<(&'static str, Vec<Value>)> {
    let numbers = [integers(), floats()].concat();
    vec![
        ("integers", integers()),
        ("small integers", (0..40).map(Value::Int64).collect()),
        ("floats", floats()),
        (
            "whole floats",
            (0..40).map(|at| Value::Float64(f64::from(at))).collect(),
        ),
        ("strings", strings()),
        ("booleans", booleans()),
        ("decimals", decimals()),
        ("temporals", temporals()),
        ("numbers", numbers),
        ("all kinds", every_value()),
    ]
}

fn row_value(row: usize) -> Value {
    Value::Int64(i64::try_from(row).unwrap())
}

/// `values` on nodes of `L`, again on nodes of `Other`, and a node with no value of each label.
fn hold(db: &GrafeoDB, values: &[Value]) {
    for (row, value) in values.iter().enumerate() {
        db.create_node_with_props(&["L"], [("row", row_value(row)), ("k", value.clone())])
            .unwrap();
    }
    for value in values {
        db.create_node_with_props(
            &["Other"],
            [("row", Value::Int64(-1)), ("k", value.clone())],
        )
        .unwrap();
    }
    db.create_node_with_props(&["L"], [("row", Value::Int64(-2))])
        .unwrap();
    db.create_node_with_props(&["Other"], [("row", Value::Int64(-3))])
        .unwrap();
}

fn session_with(db: &GrafeoDB, rules: PlannerRules) -> Session {
    let mut session = db.session();
    session.set_planner_rules(rules);
    session
}

/// The language a form is written in: they read a zoned against a local datetime differently.
#[derive(Clone, Copy, Debug)]
enum Language {
    Gql,
    Cypher,
}

/// The rows in the order they came, or that the statement failed.
fn outcome(
    session: &Session,
    language: Language,
    query: &str,
    params: &HashMap<String, Value>,
) -> Result<Vec<String>, String> {
    let result = match language {
        Language::Gql => session.execute_with_params(query, params.clone()),
        Language::Cypher => session.execute_cypher_with_params(query, params.clone()),
    };
    result
        .map(|result| result.rows().iter().map(|row| format!("{row:?}")).collect())
        .map_err(|error| error.to_string())
}

fn plan_of(
    session: &Session,
    language: Language,
    query: &str,
    params: &HashMap<String, Value>,
) -> String {
    let explain = format!("EXPLAIN {query}");
    let result = match language {
        Language::Gql => session.execute_with_params(&explain, params.clone()),
        Language::Cypher => session.execute_cypher_with_params(&explain, params.clone()),
    };
    result.unwrap().rows()[0][0].as_str().unwrap().to_string()
}

fn one(value: &Value) -> HashMap<String, Value> {
    HashMap::from([("v".to_string(), value.clone())])
}

/// A bound of each side, where the range is one comparison. `n.row` tells the rows apart.
const ONE_BOUND: &[&str] = &[
    "MATCH (n:L) WHERE n.k > $v RETURN n.row",
    "MATCH (n:L) WHERE n.k >= $v RETURN n.row",
    "MATCH (n:L) WHERE n.k < $v RETURN n.row",
    "MATCH (n:L) WHERE n.k <= $v RETURN n.row",
    "MATCH (n:L) WHERE $v < n.k RETURN n.row",
    "MATCH (n:L) WHERE $v <= n.k RETURN n.row",
    "MATCH (n:L) WHERE $v > n.k RETURN n.row",
    "MATCH (n:L) WHERE $v >= n.k RETURN n.row",
    "MATCH (n) WHERE n.k > $v RETURN n.row",
    "MATCH (n) WHERE n.k <= $v RETURN n.row",
    "MATCH (n:Nobody) WHERE n.k > $v RETURN n.row",
    "MATCH (n:L) WHERE n.nothing > $v RETURN n.row",
    "MATCH (n:L) WHERE n.k > $v RETURN n.row LIMIT 3",
    "MATCH (n) WHERE n.k <= $v RETURN n.row SKIP 1 LIMIT 3",
    "MATCH (n:L) WHERE n.k > $v RETURN count(n)",
    "OPTIONAL MATCH (n:L) WHERE n.k > $v RETURN n.row",
];

/// What a filter around a range must keep: a read that took the whole predicate for a range would
/// lose a row here.
const AROUND: &[&str] = &[
    "MATCH (n:L) WHERE n.k > $v AND n.row % 2 = 0 RETURN n.row",
    "MATCH (n:L) WHERE n.row % 2 = 0 AND n.k <= $v RETURN n.row",
    "MATCH (n:L) WHERE n.k > $v OR n.row = 0 RETURN n.row",
    "MATCH (n:L) WHERE NOT n.k > $v RETURN n.row",
    "MATCH (n:L) WHERE NOT n.k <= $v RETURN n.row",
    "MATCH (n:L) WHERE (n.k > $v) IS NULL RETURN n.row",
];

/// The forms whose bound the range read takes.
const READS_THE_RANGE: &[&str] = &[
    "MATCH (n:L) WHERE n.k > $v RETURN n.row",
    "MATCH (n:L) WHERE n.k >= $v RETURN n.row",
    "MATCH (n:L) WHERE n.k < $v RETURN n.row",
    "MATCH (n:L) WHERE n.k <= $v RETURN n.row",
    "MATCH (n:L) WHERE $v < n.k RETURN n.row",
    "MATCH (n) WHERE n.k > $v RETURN n.row",
];

fn differs_from_the_filter(db: &GrafeoDB, stage: &str) -> Vec<String> {
    let read = db.session();
    let filter = session_with(db, PlannerRules::plain());
    let mut bounds = every_value();
    bounds.push(Value::Null);
    let mut wrong = Vec::new();
    for language in [Language::Gql, Language::Cypher] {
        for bound in &bounds {
            let params = one(bound);
            for form in ONE_BOUND.iter().chain(AROUND) {
                let found = outcome(&read, language, form, &params);
                let wanted = outcome(&filter, language, form, &params);
                if found != wanted {
                    wrong.push(format!(
                        "{stage}, {language:?}: {form} with {bound:?}\n  read   {found:?}\n  filter {wanted:?}"
                    ));
                }
            }
        }
    }
    wrong
}

/// Two bounds of one column: they meet at a value, cross, hold a fraction, and go past the reach of
/// a float, with a NaN on either side.
fn differs_between_two_bounds(db: &GrafeoDB, stage: &str) -> Vec<String> {
    let read = db.session();
    let filter = session_with(db, PlannerRules::plain());
    let bounds = [
        Value::Int64(1),
        Value::Int64(7),
        Value::Int64(EXACT),
        Value::Int64(EXACT + 1),
        Value::Int64(i64::MAX),
        Value::Float64(1.5),
        Value::Float64(-0.0),
        Value::Float64(0.0),
        Value::Float64(9_007_199_254_740_992.0),
        Value::Float64(f64::NAN),
        Value::Float64(f64::INFINITY),
        Value::from("5"),
        Value::from("Alix"),
        Value::Bool(false),
        Value::Decimal(Decimal::parse("1.5").unwrap()),
        Value::Null,
    ];
    let forms = [
        "MATCH (n:L) WHERE n.k > $v AND n.k < $w RETURN n.row",
        "MATCH (n:L) WHERE n.k >= $v AND n.k <= $w RETURN n.row",
        "MATCH (n:L) WHERE n.k > $v AND n.k <= $w RETURN n.row",
        "MATCH (n:L) WHERE n.k >= $v AND n.k < $w RETURN n.row",
        "MATCH (n:L) WHERE n.k < $w AND n.k > $v RETURN n.row",
        "MATCH (n) WHERE n.k >= $v AND n.k < $w RETURN n.row",
        "MATCH (n) WHERE $v <= n.k AND $w >= n.k RETURN n.row LIMIT 3",
    ];
    let mut wrong = Vec::new();
    for language in [Language::Gql, Language::Cypher] {
        for low in &bounds {
            for high in &bounds {
                let params = HashMap::from([
                    ("v".to_string(), low.clone()),
                    ("w".to_string(), high.clone()),
                ]);
                for form in forms {
                    let found = outcome(&read, language, form, &params);
                    let wanted = outcome(&filter, language, form, &params);
                    if found != wanted {
                        wrong.push(format!(
                            "{stage}, {language:?}: {form} with {low:?}, {high:?}\n  read   {found:?}\n  filter {wanted:?}"
                        ));
                    }
                }
            }
        }
    }
    wrong
}

fn on_each_stage(values: &[Value], check: impl Fn(&GrafeoDB, &str)) {
    let hot = GrafeoDB::new_in_memory();
    hold(&hot, values);
    check(&hot, "hot");
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kinds.grafeo");
    {
        let db = GrafeoDB::open(&path).unwrap();
        hold(&db, values);
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    check(&db, "reopened");
    db.close().unwrap();
    // A write thaws the column, and a value of another kind leaves its values right too.
    for written in [Value::Int64(7), Value::from("Gus"), Value::Float64(2.5)] {
        let db = GrafeoDB::open(&path).unwrap();
        let node = db
            .create_node_with_props(&["L"], [("row", Value::Int64(-4)), ("k", written.clone())])
            .unwrap();
        check(&db, &format!("thawed by {written:?}"));
        assert!(db.delete_node(node).unwrap());
        db.close().unwrap();
    }
}

/// Every difference of every column, so one wrong kind does not hide another.
fn report(wrong: &[String]) {
    let shown: Vec<&str> = wrong.iter().take(60).map(String::as_str).collect();
    assert!(
        wrong.is_empty(),
        "{} differ, the first of them:\n{}",
        wrong.len(),
        shown.join("\n")
    );
}

#[test]
fn every_kind_of_bound_finds_what_the_filter_finds() {
    let wrong = std::cell::RefCell::new(Vec::new());
    for (name, values) in columns() {
        on_each_stage(&values, |db, stage| {
            wrong
                .borrow_mut()
                .extend(differs_from_the_filter(db, &format!("{name}, {stage}")));
        });
    }
    report(&wrong.borrow());
}

#[test]
fn two_bounds_find_what_the_filter_finds() {
    let wrong = std::cell::RefCell::new(Vec::new());
    for (name, values) in columns() {
        on_each_stage(&values, |db, stage| {
            wrong
                .borrow_mut()
                .extend(differs_between_two_bounds(db, &format!("{name}, {stage}")));
        });
    }
    report(&wrong.borrow());
}

/// The forms above do read the range, so their agreement says something about it: a number, a
/// string, a boolean and a date bound, hot and reopened.
#[test]
fn a_range_shows_the_read_and_the_rule_takes_it_away() {
    let check = |db: &GrafeoDB, stage: &str| {
        let usual = db.session();
        let off = session_with(db, PlannerRules::all().with(PlannerRule::IndexRange, false));
        for bound in [
            Value::Int64(5),
            Value::Float64(1.5),
            Value::from("Alix"),
            Value::Bool(false),
            Value::Date(Date::from_days(1)),
        ] {
            let params = one(&bound);
            for form in READS_THE_RANGE {
                let plan = plan_of(&usual, Language::Gql, form, &params);
                assert!(
                    plan.contains("[range: k]"),
                    "{stage}: {form} with {bound:?}\n{plan}"
                );
                let plan = plan_of(&off, Language::Gql, form, &params);
                assert!(
                    !plan.contains("[range: k]"),
                    "{stage}, rule off: {form} with {bound:?}\n{plan}"
                );
            }
        }
    };
    let db = GrafeoDB::new_in_memory();
    hold(&db, &integers());
    check(&db, "hot");
    let (_dir, db) = reopened(|db| hold(db, &integers()));
    check(&db, "reopened");
    db.close().unwrap();
}

// ---------------------------------------------------------------------------
// The comparison written out: an agreement would hold of a wrong filter too
// ---------------------------------------------------------------------------

/// The rows of a column whose comparison with the bound holds, written from the rules: NaN is never
/// less or greater than a number, -0.0 equals 0.0, and an integer and a float compare exactly.
#[test]
fn numbers_compare_exactly() {
    let float = Value::Float64;
    let int = Value::Int64;
    let exact_float = EXACT as f64;
    let beyond = 9_223_372_036_854_775_808.0;
    // Column, form and bound, rows.
    type Case = (Vec<Value>, &'static str, Value, Vec<i64>);
    let cases: Vec<Case> = vec![
        // NaN against a number, either way round.
        (vec![float(f64::NAN), float(1.0)], ">", float(0.0), vec![1]),
        (vec![float(f64::NAN), float(1.0)], "<", float(2.0), vec![1]),
        (vec![float(f64::NAN), float(1.0)], ">=", int(0), vec![1]),
        (vec![float(f64::NAN), float(1.0)], "<=", int(2), vec![1]),
        (vec![float(1.0), int(2)], ">", float(f64::NAN), vec![]),
        (vec![float(1.0), int(2)], "<", float(f64::NAN), vec![]),
        (vec![float(1.0), int(2)], ">=", float(f64::NAN), vec![]),
        (vec![float(1.0), int(2)], "<=", float(f64::NAN), vec![]),
        (vec![float(f64::NAN)], ">=", float(f64::NAN), vec![]),
        (
            vec![float(f64::NAN)],
            "<=",
            float(f64::NEG_INFINITY),
            vec![],
        ),
        // -0.0 and 0.0 are one value, read from an integer too.
        (vec![float(-0.0), float(0.0)], ">=", float(0.0), vec![0, 1]),
        (vec![float(-0.0), float(0.0)], "<=", float(-0.0), vec![0, 1]),
        (vec![float(-0.0), float(0.0)], ">", float(-0.0), vec![]),
        (vec![float(-0.0), float(0.0)], "<", float(0.0), vec![]),
        (vec![float(-0.0), float(0.0)], ">=", int(0), vec![0, 1]),
        (vec![float(-0.0), float(0.0)], ">", int(0), vec![]),
        (vec![int(0)], ">=", float(-0.0), vec![0]),
        (vec![int(0)], "<", float(-0.0), vec![]),
        // 2^53 as an integer and as a float, and the integers a float cannot hold.
        (
            vec![int(EXACT), int(EXACT + 1), int(EXACT - 1)],
            ">",
            float(exact_float),
            vec![1],
        ),
        (
            vec![int(EXACT), int(EXACT + 1), int(EXACT - 1)],
            ">=",
            float(exact_float),
            vec![0, 1],
        ),
        (
            vec![int(EXACT), int(EXACT + 1), int(EXACT - 1)],
            "<",
            float(exact_float),
            vec![2],
        ),
        (
            vec![int(EXACT), int(EXACT + 1), int(EXACT - 1)],
            "<=",
            float(exact_float),
            vec![0, 2],
        ),
        (vec![float(exact_float)], "<", int(EXACT + 1), vec![0]),
        (vec![float(exact_float)], "<=", int(EXACT), vec![0]),
        (vec![float(exact_float)], ">", int(EXACT), vec![]),
        (vec![float(exact_float)], ">=", int(EXACT + 1), vec![]),
        (vec![float(exact_float + 2.0)], ">", int(EXACT + 1), vec![0]),
        // A fraction between two integers.
        (vec![int(1), int(2)], ">", float(1.5), vec![1]),
        (vec![int(1), int(2)], ">=", float(1.5), vec![1]),
        (vec![int(1), int(2)], "<", float(1.5), vec![0]),
        (vec![int(1), int(2)], "<=", float(1.5), vec![0]),
        (vec![int(-1), int(-2)], ">", float(-1.5), vec![0]),
        (vec![int(-1), int(-2)], "<", float(-1.5), vec![1]),
        // Past the reach of an integer.
        (
            vec![int(i64::MAX), int(i64::MIN)],
            "<",
            float(beyond),
            vec![0, 1],
        ),
        (
            vec![int(i64::MAX), int(i64::MIN)],
            ">=",
            float(beyond),
            vec![],
        ),
        (
            vec![int(i64::MAX), int(i64::MIN)],
            ">=",
            float(-beyond),
            vec![0, 1],
        ),
        (
            vec![int(i64::MAX), int(i64::MIN)],
            ">",
            float(-beyond),
            vec![0],
        ),
        (
            vec![int(i64::MAX), int(i64::MIN)],
            "<",
            float(f64::INFINITY),
            vec![0, 1],
        ),
        (
            vec![int(i64::MAX), int(i64::MIN)],
            ">",
            float(f64::NEG_INFINITY),
            vec![0, 1],
        ),
        (vec![float(beyond)], ">", int(i64::MAX), vec![0]),
        (vec![float(-beyond)], ">=", int(i64::MIN), vec![0]),
        (vec![float(-beyond)], ">", int(i64::MIN), vec![]),
        (vec![float(f64::INFINITY)], ">", int(i64::MAX), vec![0]),
        // A bound at the end of the integers takes nothing past it, and does not wrap.
        (vec![int(i64::MAX), int(0)], ">", int(i64::MAX), vec![]),
        (vec![int(i64::MIN), int(0)], "<", int(i64::MIN), vec![]),
        (vec![int(i64::MAX), int(0)], ">=", int(i64::MAX), vec![0]),
        (vec![int(i64::MIN), int(0)], "<=", int(i64::MIN), vec![0]),
    ];
    let check = |db: &GrafeoDB, stage: &str, form: &str, bound: &Value, wanted: &[i64]| {
        let query = format!("MATCH (n:L) WHERE n.k {form} $v RETURN n.row");
        for (name, session) in [
            ("read", db.session()),
            ("filter", session_with(db, PlannerRules::plain())),
        ] {
            let result = session.execute_with_params(&query, one(bound)).unwrap();
            let rows: Vec<i64> = result
                .rows()
                .iter()
                .map(|row| match row[0] {
                    Value::Int64(at) => at,
                    ref other => panic!("{other:?}"),
                })
                .collect();
            assert_eq!(rows, wanted, "{stage}, {name}: n.k {form} {bound:?}");
        }
    };
    for (values, form, bound, wanted) in &cases {
        let db = GrafeoDB::new_in_memory();
        keep(&db, values);
        check(&db, "hot", form, bound, wanted);
        let (_dir, db) = reopened(|db| keep(db, values));
        check(&db, "reopened", form, bound, wanted);
        db.close().unwrap();
    }
}

/// `values` as rows of `L`, in order.
fn keep(db: &GrafeoDB, values: &[Value]) {
    for (row, value) in values.iter().enumerate() {
        db.create_node_with_props(&["L"], [("row", row_value(row)), ("k", value.clone())])
            .unwrap();
    }
}

/// Over `'50'` and `50` and a node with no `k`, the table of GA04: a string and a number are two kinds,
/// so `=` is false; `<` orders them by kind in GQL, text before numbers, and is null in Cypher. The
/// range's answer to `n.k > 40` is the one a filter around it gives, in both languages.
#[test]
fn a_string_and_a_number_by_the_rule_of_each_language() {
    let fill = |db: &GrafeoDB| {
        keep(db, &[Value::from("50"), Value::Int64(50)]);
        db.create_node_with_props(&["L"], [("row", Value::Int64(2))])
            .unwrap();
    };
    // Predicate, GQL's rows, Cypher's rows.
    let cases: &[(&str, &[i64], &[i64])] = &[
        ("n.k > 40", &[1], &[1]),
        ("n.k >= 40", &[1], &[1]),
        ("n.k < 40", &[0], &[]),
        ("n.k <= 40", &[0], &[]),
        ("40 > n.k", &[0], &[]),
        ("n.k > 40 AND n.row >= 0", &[1], &[1]),
        ("n.k > 40 AND n.k < 60", &[1], &[1]),
        ("n.k < 40 AND n.row >= 0", &[0], &[]),
        ("n.k = 50", &[1], &[1]),
        ("n.k = '50'", &[0], &[0]),
        ("n.k <> 50", &[0], &[0]),
        ("n.k > '4'", &[0, 1], &[0]),
        ("n.k < '6'", &[0], &[0]),
        ("n.k > '4' AND n.k < '6'", &[0], &[0]),
        ("n.k >= '50' AND n.k <= '50'", &[0], &[0]),
        ("n.k > 40.5", &[1], &[1]),
        ("n.k < 40.5", &[0], &[]),
    ];
    let check = |db: &GrafeoDB, stage: &str| {
        for (predicate, gql, cypher) in cases {
            let query = format!("MATCH (n:L) WHERE {predicate} RETURN n.row ORDER BY n.row");
            for (name, session) in [
                ("read", db.session()),
                ("filter", session_with(db, PlannerRules::plain())),
            ] {
                for (language, wanted) in [(Language::Gql, gql), (Language::Cypher, cypher)] {
                    let found = outcome(&session, language, &query, &HashMap::new());
                    let wanted: Vec<String> = wanted
                        .iter()
                        .map(|row| format!("{:?}", vec![Value::Int64(*row)]))
                        .collect();
                    assert_eq!(
                        found,
                        Ok(wanted),
                        "{stage}, {name}, {language:?}: {predicate}"
                    );
                }
            }
        }
    };
    let db = GrafeoDB::new_in_memory();
    fill(&db);
    check(&db, "hot");
    let (_dir, db) = reopened(fill);
    check(&db, "reopened");
    db.close().unwrap();
}

/// One value of each kind, with its rank in the order across kinds and its family: two of one family
/// compare as their kind does, two of different families by the rank in GQL and not at all in Cypher.
fn one_of_each_kind() -> Vec<(&'static str, u8, Value)> {
    vec![
        ("number", 15, Value::Int64(5)),
        ("number", 15, Value::Float64(2.5)),
        ("number", 15, Value::Decimal(Decimal::parse("1.5").unwrap())),
        ("string", 11, Value::from("m")),
        ("boolean", 12, Value::Bool(true)),
        ("date", 6, Value::Date(Date::from_days(19_000))),
        (
            "datetime",
            5,
            Value::Timestamp(Timestamp::from_micros(1_696_500_000_000_000)),
        ),
        (
            "datetime",
            4,
            Value::ZonedDatetime(ZonedDatetime::from_timestamp_offset(
                Timestamp::from_micros(1_696_500_000_000_000),
                3_600,
            )),
        ),
        ("time", 8, Value::Time(Time::from_hms(10, 0, 0).unwrap())),
        (
            "time",
            7,
            Value::Time(Time::from_hms(10, 0, 0).unwrap().with_offset(3_600)),
        ),
        ("duration", 9, Value::Duration(Duration::new(0, 1, 0))),
        ("list", 1, Value::List(vec![Value::Int64(1)].into())),
        ("bytes", 10, Value::Bytes(vec![1_u8].into())),
        ("vector", 3, Value::Vector(vec![1.0_f32].into())),
    ]
}

/// A column of one value of each kind against a one-sided bound of each kind: a row of another family
/// is in the range by its kind's rank in GQL and never in Cypher; a row of the bound's family is in it
/// as the filter compares the two. Both the read and the filter answer so, hot and reopened.
#[test]
fn each_kind_against_each_one_sided_bound() {
    let kinds = one_of_each_kind();
    let fill = |db: &GrafeoDB| {
        let values: Vec<Value> = one_of_each_kind()
            .into_iter()
            .map(|(_, _, value)| value)
            .collect();
        keep(db, &values);
    };
    let check = |db: &GrafeoDB, stage: &str| {
        let read = db.session();
        let filter = session_with(db, PlannerRules::plain());
        let mut wrong = Vec::new();
        for (bound_family, bound_rank, bound) in &kinds {
            for op in ["<", "<=", ">", ">="] {
                let query = format!("MATCH (n:L) WHERE n.k {op} $v RETURN n.row ORDER BY n.row");
                for language in [Language::Gql, Language::Cypher] {
                    // Its own family, by the filter alone; on its own it answers as the kind does.
                    let same: Vec<i64> = kinds
                        .iter()
                        .enumerate()
                        .filter(|(_, (family, _, _))| family == bound_family)
                        .filter(|(row, _)| {
                            let alone = format!(
                                "MATCH (n:L {{row: {row}}}) WHERE n.k {op} $v RETURN n.row"
                            );
                            outcome(&filter, language, &alone, &one(bound))
                                .is_ok_and(|rows| !rows.is_empty())
                        })
                        .map(|(row, _)| i64::try_from(row).unwrap())
                        .collect();
                    let mut wanted: Vec<i64> = kinds
                        .iter()
                        .enumerate()
                        .filter(|(_, (family, rank, _))| {
                            family != bound_family
                                && matches!(language, Language::Gql)
                                && match op {
                                    "<" => rank < bound_rank,
                                    "<=" => rank <= bound_rank,
                                    ">" => rank > bound_rank,
                                    _ => rank >= bound_rank,
                                }
                        })
                        .map(|(row, _)| i64::try_from(row).unwrap())
                        .chain(same)
                        .collect();
                    wanted.sort_unstable();
                    let wanted: Vec<String> = wanted
                        .iter()
                        .map(|row| format!("{:?}", vec![Value::Int64(*row)]))
                        .collect();
                    for (name, session) in [("read", &read), ("filter", &filter)] {
                        let found = outcome(session, language, &query, &one(bound));
                        if found.as_ref() != Ok(&wanted) {
                            wrong.push(format!(
                                "{stage}, {name}, {language:?}: n.k {op} {bound:?}\n  found {found:?}\n  want  {wanted:?}"
                            ));
                        }
                    }
                }
            }
        }
        assert!(
            wrong.is_empty(),
            "{} wrong:\n{}",
            wrong.len(),
            wrong.join("\n")
        );
    };
    let db = GrafeoDB::new_in_memory();
    fill(&db);
    check(&db, "hot");
    let (_dir, db) = reopened(fill);
    check(&db, "reopened");
    db.close().unwrap();
}

/// Null as a bound finds nothing, and a row with no value is not found by any bound.
#[test]
fn null_bounds_and_rows_without_a_value() {
    let fill = |db: &GrafeoDB| {
        db.create_node_with_props(&["L"], [("row", Value::Int64(0)), ("k", Value::Int64(1))])
            .unwrap();
        db.create_node_with_props(&["L"], [("row", Value::Int64(1))])
            .unwrap();
        db.create_node_with_props(&["L"], [("row", Value::Int64(2)), ("k", Value::Null)])
            .unwrap();
    };
    let check = |db: &GrafeoDB, stage: &str| {
        for form in [
            "n.k > null",
            "n.k < null",
            "n.k >= null",
            "n.k <= null",
            "null < n.k",
        ] {
            let query = format!("MATCH (n:L) WHERE {form} RETURN n.row");
            for session in [db.session(), session_with(db, PlannerRules::plain())] {
                assert_eq!(
                    session.execute(&query).unwrap().row_count(),
                    0,
                    "{stage}: {query}"
                );
            }
        }
        for form in [
            "n.k < 2",
            "n.k > 0",
            "n.k >= -9223372036854775808",
            "n.k < 1.0e300",
        ] {
            let query = format!("MATCH (n:L) WHERE {form} RETURN n.row");
            for session in [db.session(), session_with(db, PlannerRules::plain())] {
                let result = session.execute(&query).unwrap();
                assert_eq!(result.rows().len(), 1, "{stage}: {query}");
                assert_eq!(result.rows()[0][0], Value::Int64(0), "{stage}: {query}");
            }
        }
    };
    let db = GrafeoDB::new_in_memory();
    fill(&db);
    check(&db, "hot");
    let (_dir, db) = reopened(fill);
    check(&db, "reopened");
    db.close().unwrap();
}

// ---------------------------------------------------------------------------
// Shapes of a column
// ---------------------------------------------------------------------------

fn numbered(db: &GrafeoDB, rows: i64, value: impl Fn(i64) -> Value) {
    for row in 0..rows {
        db.create_node_with_props(&["L"], [("row", Value::Int64(row)), ("k", value(row))])
            .unwrap();
    }
}

fn reopened(fill: impl Fn(&GrafeoDB)) -> (tempfile::TempDir, GrafeoDB) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("read.grafeo");
    {
        let db = GrafeoDB::open(&path).unwrap();
        fill(&db);
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    (dir, db)
}

/// Every bound here, as text, over a column read by the read and by the filter, hot and reopened.
fn shape_agrees(name: &str, fill: impl Fn(&GrafeoDB) + Copy, bounds: &[&str]) {
    let check = |db: &GrafeoDB, stage: &str| {
        let usual = db.session();
        let plain = session_with(db, PlannerRules::plain());
        for bound in bounds {
            for form in [
                format!("MATCH (n:L) WHERE n.k > {bound} RETURN n.row"),
                format!("MATCH (n:L) WHERE n.k >= {bound} RETURN n.row"),
                format!("MATCH (n:L) WHERE n.k < {bound} RETURN n.row"),
                format!("MATCH (n:L) WHERE n.k <= {bound} RETURN n.row"),
                format!("MATCH (n) WHERE n.k > {bound} RETURN n.row"),
                format!("MATCH (n:L) WHERE n.k > {bound} RETURN n.row LIMIT 7"),
                format!("MATCH (n:L) WHERE n.k >= {bound} AND n.k < 100000 RETURN n.row"),
            ] {
                let found = usual
                    .execute(&form)
                    .unwrap_or_else(|error| panic!("{name}, {stage}: {form}: {error}"));
                assert_eq!(
                    found.rows(),
                    plain.execute(&form).unwrap().rows(),
                    "{name}, {stage}: {form}"
                );
                let found = usual
                    .execute_cypher(&form)
                    .unwrap_or_else(|error| panic!("{name}, {stage}, Cypher: {form}: {error}"));
                assert_eq!(
                    found.rows(),
                    plain.execute_cypher(&form).unwrap().rows(),
                    "{name}, {stage}, Cypher: {form}"
                );
            }
        }
    };
    let db = GrafeoDB::new_in_memory();
    fill(&db);
    check(&db, "hot");
    let (_dir, db) = reopened(fill);
    check(&db, "reopened");
    db.close().unwrap();
}

const WIDE: &[&str] = &[
    "-1", "0", "5", "4999.5", "65536", "131071", "1.0e300", "-1.0e300",
];

#[test]
fn a_column_in_order_across_groups() {
    shape_agrees("in order", |db| numbered(db, GROUP * 2, Value::Int64), WIDE);
}

#[test]
fn a_shuffled_column_across_groups() {
    let rows = GROUP * 2;
    shape_agrees(
        "shuffled",
        move |db| numbered(db, rows, |row| Value::Int64(row * 7_919 % rows)),
        WIDE,
    );
}

/// One value throughout: a chunk of its own form.
#[test]
fn a_column_of_one_value() {
    shape_agrees(
        "one value",
        |db| numbered(db, 5_000, |_| Value::Int64(5)),
        WIDE,
    );
    shape_agrees(
        "one float",
        |db| numbered(db, 5_000, |_| Value::Float64(5.0)),
        WIDE,
    );
}

/// Small integers pack; a row in `presence` skips: the rows must not shift.
#[test]
fn a_column_with_rows_that_hold_nothing() {
    for every in [2, 3, 64, 65, 1_000] {
        let fill = move |db: &GrafeoDB| {
            for row in 0..5_000_i64 {
                if row % every == 0 {
                    db.create_node_with_props(
                        &["L"],
                        [("row", Value::Int64(row)), ("k", Value::Int64(row % 100))],
                    )
                    .unwrap();
                } else {
                    db.create_node_with_props(&["L"], [("row", Value::Int64(row))])
                        .unwrap();
                }
            }
        };
        shape_agrees(&format!("one row in {every}"), fill, WIDE);
        let floats = move |db: &GrafeoDB| {
            for row in 0..5_000_i64 {
                if row % every == 0 {
                    db.create_node_with_props(
                        &["L"],
                        [
                            ("row", Value::Int64(row)),
                            ("k", Value::Float64((row % 100) as f64 + 0.5)),
                        ],
                    )
                    .unwrap();
                } else {
                    db.create_node_with_props(&["L"], [("row", Value::Int64(row))])
                        .unwrap();
                }
            }
        };
        shape_agrees(&format!("one float in {every}"), floats, WIDE);
    }
}

/// A NaN in a chunk must not hide its other values from the chunk's bounds, nor be found by one.
#[test]
fn a_nan_among_floats_across_groups() {
    for at in [0, 1, 4_999, GROUP - 1, GROUP] {
        shape_agrees(
            &format!("a NaN at {at}"),
            move |db| {
                numbered(db, GROUP + 100, |row| {
                    if row == at {
                        Value::Float64(f64::NAN)
                    } else {
                        Value::Float64(row as f64)
                    }
                });
            },
            WIDE,
        );
    }
}

/// Integers in one group and floats in the next, then both in one chunk: bounds of two kinds.
#[test]
fn kinds_that_change_across_a_column() {
    shape_agrees(
        "by group",
        |db| {
            numbered(db, GROUP + 3_000, |row| {
                if row < GROUP {
                    Value::Int64(row)
                } else {
                    Value::Float64(row as f64 + 0.5)
                }
            });
        },
        WIDE,
    );
    shape_agrees(
        "by row",
        |db| {
            numbered(db, 5_000, |row| match row % 4 {
                0 => Value::Int64(row),
                1 => Value::Float64(row as f64 + 0.5),
                2 => Value::from(format!("{row}").as_str()),
                _ => Value::Bool(row % 8 == 3),
            });
        },
        WIDE,
    );
    shape_agrees(
        "strings then integers",
        |db| {
            numbered(db, GROUP + 3_000, |row| {
                if row < GROUP {
                    Value::from(format!("{row}").as_str())
                } else {
                    Value::Int64(row)
                }
            });
        },
        WIDE,
    );
}

/// Text order is not number order: `'10'` sorts before `'9'`, so a chunk's bounds as text must not
/// rule out a number bound. Hot, reopened, in a dictionary, and across groups.
#[test]
fn strings_that_spell_numbers_against_a_number_bound() {
    let bounds = [
        "5", "9", "9.5", "10", "99", "100", "1.0e3", "-1", "'9'", "'10'", "'Alix'",
    ];
    shape_agrees(
        "two digits",
        |db| {
            numbered(db, 5_000, |row| {
                Value::from(format!("{}", row % 90 + 10).as_str())
            });
        },
        &bounds,
    );
    shape_agrees(
        "one to three digits",
        |db| {
            numbered(db, 5_000, |row| {
                Value::from(format!("{}", row % 500).as_str())
            });
        },
        &bounds,
    );
    shape_agrees(
        "across groups",
        |db| {
            numbered(db, GROUP + 100, |row| {
                Value::from(format!("{}", row % 90 + 10).as_str())
            });
        },
        &bounds,
    );
    shape_agrees(
        "spellings",
        |db| {
            numbered(db, 5_000, |row| {
                Value::from(
                    ["42", "042", "42.0", "4.2e1", "+42", "NaN", "inf", "Alix"]
                        [usize::try_from(row % 8).unwrap()],
                )
            });
        },
        &[
            "41", "42", "42.0", "43", "1.0e300", "-1.0e300", "'42'", "'NaN'",
        ],
    );
}

/// Strings keep their order, read through a dictionary or not.
#[test]
fn strings_read_in_their_order() {
    let bounds = [
        "''",
        "'Amsterdam'",
        "'Berlin'",
        "'Paris'",
        "'Z'",
        "'city-0100'",
    ];
    shape_agrees(
        "few strings",
        |db| {
            numbered(db, 5_000, |row| {
                Value::from(
                    ["Amsterdam", "Berlin", "Paris", "Prague"][usize::try_from(row % 4).unwrap()],
                )
            });
        },
        &bounds,
    );
    shape_agrees(
        "many strings",
        |db| {
            numbered(db, 5_000, |row| {
                Value::from(format!("city-{row:04}").as_str())
            });
        },
        &bounds,
    );
    shape_agrees(
        "strings across groups",
        |db| {
            numbered(db, GROUP + 100, |row| {
                Value::from(format!("city-{:04}", row % 1_000).as_str())
            });
        },
        &bounds,
    );
}

/// A key only some nodes hold, and a label only some nodes hold, in groups of their own.
#[test]
fn a_key_and_a_label_few_nodes_hold() {
    let fill = |db: &GrafeoDB| {
        for row in 0..GROUP * 2 + 100 {
            let labels: &[&str] = if row % 10_000 == 0 { &["L"] } else { &["M"] };
            if row % 5_000 == 0 {
                db.create_node_with_props(
                    labels,
                    [("row", Value::Int64(row)), ("k", Value::Int64(row))],
                )
                .unwrap();
            } else {
                db.create_node_with_props(labels, [("row", Value::Int64(row))])
                    .unwrap();
            }
        }
    };
    shape_agrees("few", fill, WIDE);
    let check = |db: &GrafeoDB, stage: &str| {
        let rows = db
            .session()
            .execute("MATCH (n:L) WHERE n.k >= 20000 RETURN n.row")
            .unwrap();
        let rows: Vec<&Value> = rows.rows().iter().map(|row| &row[0]).collect();
        assert_eq!(
            rows,
            [
                20_000, 30_000, 40_000, 50_000, 60_000, 70_000, 80_000, 90_000, 100_000, 110_000,
                120_000, 130_000
            ]
            .map(Value::Int64)
            .iter()
            .collect::<Vec<_>>(),
            "{stage}"
        );
    };
    let db = GrafeoDB::new_in_memory();
    fill(&db);
    check(&db, "hot");
    let (_dir, db) = reopened(fill);
    check(&db, "reopened");
    db.close().unwrap();
}

// ---------------------------------------------------------------------------
// A reader's view
// ---------------------------------------------------------------------------

/// Values of each kind in their order: the first is below the second, and the third is what a change
/// moves a row to.
const KINDS: [(&str, &str, &str); 4] = [
    ("1", "2", "3"),
    ("1.5", "2.5", "3.5"),
    ("'Amsterdam'", "'Berlin'", "'Paris'"),
    ("false", "true", "true"),
];

fn names_of(session: &Session, query: &str) -> Vec<String> {
    let mut names: Vec<String> = session
        .execute(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"))
        .rows()
        .iter()
        .map(|row| row[0].as_str().unwrap_or("null").to_string())
        .collect();
    names.sort();
    names
}

/// `db` holding Alix, Gus and Mia at `first`, Jules at `second`, Hans of `M` at `first`.
fn people(path: Option<&std::path::Path>, first: &str, second: &str) -> GrafeoDB {
    let db = path.map_or_else(GrafeoDB::new_in_memory, |path| {
        GrafeoDB::open(path).unwrap()
    });
    let session = db.session();
    session
        .execute(&format!(
            "INSERT (:L {{name: 'Alix', k: {first}}}), (:L {{name: 'Gus', k: {first}}}), \
             (:L {{name: 'Mia', k: {first}}}), (:L {{name: 'Jules', k: {second}}}), (:M {{name: 'Hans', k: {first}}})"
        ))
        .unwrap();
    drop(session);
    if let Some(path) = path {
        db.close().unwrap();
        return GrafeoDB::open(path).unwrap();
    }
    db
}

fn each_stage(check: impl Fn(&GrafeoDB, (&str, &str, &str))) {
    for kind in KINDS {
        check(&people(None, kind.0, kind.1), kind);
        let dir = tempfile::tempdir().unwrap();
        let db = people(Some(&dir.path().join("view.grafeo")), kind.0, kind.1);
        check(&db, kind);
        db.close().unwrap();
    }
}

fn below(value: &str) -> String {
    format!("MATCH (n:L) WHERE n.k <= {value} RETURN n.name")
}

fn above(value: &str) -> String {
    format!("MATCH (n:L) WHERE n.k > {value} RETURN n.name")
}

#[test]
fn an_old_snapshot_reads_what_it_saw_after_a_change_of_value() {
    each_stage(|db, (first, second, _)| {
        let mut reader = db.session();
        reader.begin_transaction().unwrap();
        assert_eq!(names_of(&reader, &below(first)), ["Alix", "Gus", "Mia"]);
        assert_eq!(names_of(&reader, &above(first)), ["Jules"]);
        db.session()
            .execute(&format!("MATCH (n:L {{name: 'Alix'}}) SET n.k = {second}"))
            .unwrap();
        assert_eq!(
            names_of(&reader, &below(first)),
            ["Alix", "Gus", "Mia"],
            "{first}"
        );
        assert_eq!(names_of(&reader, &above(first)), ["Jules"], "{first}");
        reader.rollback().unwrap();
        assert_eq!(names_of(&reader, &below(first)), ["Gus", "Mia"], "{first}");
        assert_eq!(
            names_of(&reader, &above(first)),
            ["Alix", "Jules"],
            "{first}"
        );
    });
}

#[test]
fn an_old_snapshot_reads_what_it_saw_after_a_delete_and_an_insert() {
    each_stage(|db, (first, _, _)| {
        let query = below(first);
        let mut reader = db.session();
        reader.begin_transaction().unwrap();
        assert_eq!(names_of(&reader, &query), ["Alix", "Gus", "Mia"]);
        let writer = db.session();
        writer
            .execute("MATCH (n:L {name: 'Gus'}) DELETE n")
            .unwrap();
        writer
            .execute(&format!("INSERT (:L {{name: 'Vincent', k: {first}}})"))
            .unwrap();
        assert_eq!(names_of(&reader, &query), ["Alix", "Gus", "Mia"], "{first}");
        reader.rollback().unwrap();
        assert_eq!(
            names_of(&reader, &query),
            ["Alix", "Mia", "Vincent"],
            "{first}"
        );
    });
}

#[test]
fn an_old_snapshot_reads_what_it_saw_after_a_change_of_label() {
    each_stage(|db, (first, _, _)| {
        let on_l = below(first);
        let on_m = format!("MATCH (n:M) WHERE n.k <= {first} RETURN n.name");
        let mut reader = db.session();
        reader.begin_transaction().unwrap();
        let writer = db.session();
        writer
            .execute("MATCH (n:L {name: 'Alix'}) REMOVE n:L")
            .unwrap();
        writer
            .execute("MATCH (n:M {name: 'Hans'}) SET n:L")
            .unwrap();
        assert_eq!(names_of(&reader, &on_l), ["Alix", "Gus", "Mia"], "{first}");
        assert_eq!(names_of(&reader, &on_m), ["Hans"], "{first}");
        reader.rollback().unwrap();
        assert_eq!(names_of(&reader, &on_l), ["Gus", "Hans", "Mia"], "{first}");
    });
}

#[test]
fn an_old_snapshot_reads_a_value_removed_since() {
    each_stage(|db, (first, _, _)| {
        let query = below(first);
        let mut reader = db.session();
        reader.begin_transaction().unwrap();
        db.session()
            .execute("MATCH (n:L {name: 'Mia'}) REMOVE n.k")
            .unwrap();
        assert_eq!(names_of(&reader, &query), ["Alix", "Gus", "Mia"], "{first}");
        reader.rollback().unwrap();
        assert_eq!(names_of(&reader, &query), ["Alix", "Gus"], "{first}");
    });
}

/// A value of another kind: the column turns mixed under the old reader, whose rows stay.
#[test]
fn an_old_snapshot_reads_what_it_saw_after_a_change_of_kind() {
    each_stage(|db, (first, _, _)| {
        let mut reader = db.session();
        reader.begin_transaction().unwrap();
        let writer = db.session();
        writer
            .execute("MATCH (n:L {name: 'Alix'}) SET n.k = [1, 2]")
            .unwrap();
        writer
            .execute("MATCH (n:L {name: 'Gus'}) SET n.k = 'Gus'")
            .unwrap();
        assert_eq!(
            names_of(&reader, &below(first)),
            ["Alix", "Gus", "Mia"],
            "{first}"
        );
        reader.rollback().unwrap();
        let plain = session_with(db, PlannerRules::plain());
        assert_eq!(
            names_of(&reader, &below(first)),
            names_of(&plain, &below(first)),
            "{first}"
        );
        assert_eq!(
            names_of(&reader, &above(first)),
            names_of(&plain, &above(first)),
            "{first}"
        );
    });
}

#[test]
fn a_transaction_reads_its_own_changes_and_no_one_elses() {
    each_stage(|db, (first, second, third)| {
        let mut writer = db.session();
        writer.begin_transaction().unwrap();
        writer
            .execute(&format!("INSERT (:L {{name: 'Vincent', k: {first}}})"))
            .unwrap();
        writer
            .execute(&format!("MATCH (n:L {{name: 'Alix'}}) SET n.k = {second}"))
            .unwrap();
        writer
            .execute("MATCH (n:L {name: 'Gus'}) DELETE n")
            .unwrap();
        writer
            .execute(&format!("MATCH (n:L {{name: 'Jules'}}) SET n.k = {third}"))
            .unwrap();
        assert_eq!(
            names_of(&writer, &below(first)),
            ["Mia", "Vincent"],
            "own, {first}"
        );
        assert_eq!(
            names_of(&writer, &above(first)),
            ["Alix", "Jules"],
            "own, {first}"
        );
        let other = db.session();
        assert_eq!(
            names_of(&other, &below(first)),
            ["Alix", "Gus", "Mia"],
            "another's, {first}"
        );
        assert_eq!(
            names_of(&other, &above(first)),
            ["Jules"],
            "another's, {first}"
        );
        writer.rollback().unwrap();
        assert_eq!(
            names_of(&writer, &below(first)),
            ["Alix", "Gus", "Mia"],
            "after the rollback, {first}"
        );
        assert_eq!(
            names_of(&writer, &above(first)),
            ["Jules"],
            "after the rollback, {first}"
        );
    });
}

#[test]
fn a_past_epoch_reads_what_it_saw() {
    for (first, second, _) in KINDS {
        let db = GrafeoDB::with_config(
            Config::in_memory().with_history_retention(HistoryRetention::Forever),
        )
        .unwrap();
        let session = db.session();
        session
            .execute(&format!(
                "INSERT (:L {{name: 'Alix', k: {first}}}), (:L {{name: 'Gus', k: {first}}}), (:L {{name: 'Jules', k: {second}}})"
            ))
            .unwrap();
        let epoch = db.current_epoch();
        session
            .execute(&format!("MATCH (n:L {{name: 'Alix'}}) SET n.k = {second}"))
            .unwrap();
        session
            .execute("MATCH (n:L {name: 'Gus'}) DELETE n")
            .unwrap();
        let then = db.session();
        then.set_viewing_epoch(epoch);
        assert_eq!(names_of(&then, &below(first)), ["Alix", "Gus"], "{first}");
        assert_eq!(names_of(&then, &above(first)), ["Jules"], "{first}");
        then.clear_viewing_epoch();
        assert!(names_of(&then, &below(first)).is_empty(), "{first}");
        assert_eq!(names_of(&then, &above(first)), ["Alix", "Jules"], "{first}");
    }
}

/// A reader's rows through the read and through the filter at one snapshot.
struct Readers {
    read: Session,
    filter: Session,
}

impl Readers {
    fn begin(db: &GrafeoDB) -> Self {
        let mut read = db.session();
        let mut filter = session_with(db, PlannerRules::plain());
        read.begin_transaction().unwrap();
        filter.begin_transaction().unwrap();
        Self { read, filter }
    }

    fn agree(&self, query: &str, stage: &str) -> Vec<Vec<Value>> {
        let found = self.read.execute(query).unwrap().rows().to_vec();
        assert_eq!(
            found,
            self.filter.execute(query).unwrap().rows(),
            "{stage}: {query}"
        );
        found
    }

    fn end(mut self) {
        self.read.rollback().unwrap();
        self.filter.rollback().unwrap();
    }
}

fn rows_at(found: &[Vec<Value>]) -> Vec<i64> {
    found
        .iter()
        .map(|row| match row[0] {
            Value::Int64(at) => at,
            ref other => panic!("{other:?}"),
        })
        .collect()
}

/// A row moved past every bound of its group, and one moved in from past them: a reader of either
/// view finds what it saw, though the bounds of the column as it was written rule the value out, or
/// the value as it is now.
fn moves_out_of_the_bounds(
    fill: impl Fn(&GrafeoDB) + Copy,
    past: &str,
    below_all: &str,
    kind: &str,
) {
    let check = |db: &GrafeoDB, stage: &str| {
        let beyond = format!("MATCH (n:L) WHERE n.k > {past} RETURN n.row ORDER BY n.row");
        let under = format!("MATCH (n:L) WHERE n.k < {below_all} RETURN n.row ORDER BY n.row");
        let small = "MATCH (n:L) WHERE n.k < 10 RETURN n.row ORDER BY n.row";
        let reader = Readers::begin(db);
        assert!(
            rows_at(&reader.agree(&beyond, stage)).is_empty(),
            "{kind}, {stage}"
        );
        assert!(
            rows_at(&reader.agree(&under, stage)).is_empty(),
            "{kind}, {stage}"
        );
        assert_eq!(
            rows_at(&reader.agree(small, stage)).len(),
            10,
            "{kind}, {stage}"
        );
        let writer = db.session();
        writer
            .execute(&format!("MATCH (n:L {{row: 5}}) SET n.k = {past} + 1"))
            .unwrap();
        writer
            .execute(&format!(
                "MATCH (n:L {{row: {}}}) SET n.k = {below_all} - 1",
                GROUP + 5
            ))
            .unwrap();
        // Old: what it saw, the chain's values against bounds that hold them or not.
        assert!(
            rows_at(&reader.agree(&beyond, stage)).is_empty(),
            "{kind}, {stage}: old"
        );
        assert!(
            rows_at(&reader.agree(&under, stage)).is_empty(),
            "{kind}, {stage}: old"
        );
        assert_eq!(
            rows_at(&reader.agree(small, stage)).len(),
            10,
            "{kind}, {stage}: old"
        );
        reader.end();
        // New: the rows as they are, though the group's bounds did not hold the value.
        let now = Readers::begin(db);
        assert_eq!(
            rows_at(&now.agree(&beyond, stage)),
            [5],
            "{kind}, {stage}: new"
        );
        assert_eq!(
            rows_at(&now.agree(&under, stage)),
            [GROUP + 5],
            "{kind}, {stage}: new"
        );
        assert_eq!(
            rows_at(&now.agree(small, stage)),
            [0, 1, 2, 3, 4, 6, 7, 8, 9, GROUP + 5],
            "{kind}, {stage}: new"
        );
        now.end();
    };
    let db = GrafeoDB::new_in_memory();
    fill(&db);
    check(&db, "hot");
    let (_dir, db) = reopened(fill);
    check(&db, "reopened");
    db.close().unwrap();
}

#[test]
fn a_value_moved_past_a_groups_bounds_is_read_at_the_view() {
    moves_out_of_the_bounds(
        |db| {
            numbered(db, GROUP * 2 + 100, |row| {
                // Each group holds what its rows say, so a chunk's bounds rule the others out.
                Value::Int64(row)
            });
        },
        "9000000",
        "-9000000",
        "integers",
    );
    moves_out_of_the_bounds(
        |db| {
            numbered(db, GROUP * 2 + 100, |row| Value::Float64(row as f64));
        },
        "9000000.5",
        "-9000000.5",
        "floats",
    );
}

/// A row with a chain in a later slice is read as the view sees it, once, by an old reader and a
/// new one, in a column of any kinds.
#[test]
fn a_chained_row_in_a_later_slice_is_read_at_the_view() {
    const SLICED: i64 = 7_000;
    let sliced = |db: &GrafeoDB| {
        numbered(db, SLICED, |row| match row % 4 {
            0 => Value::Int64(row % 10),
            1 => Value::from(format!("{}", row % 10).as_str()),
            2 => Value::Float64((row % 10) as f64),
            _ => Value::Bool(row % 8 == 3),
        });
    };
    let check = |db: &GrafeoDB, stage: &str| {
        let on = |form: &str| format!("MATCH (n:L) WHERE n.k {form} RETURN n.row ORDER BY n.row");
        let forms = [on("<= 3"), on("> 50"), on(">= 3"), on("< 3.0"), on("> '3'")];
        let reader = Readers::begin(db);
        let before: Vec<_> = forms
            .iter()
            .map(|query| reader.agree(query, stage))
            .collect();
        for write in [
            "MATCH (n:L {row: 4800}) SET n.k = 99",
            "MATCH (n:L {row: 4803}) SET n.k = 99",
            "MATCH (n:L {row: 5001}) SET n.k = 3",
        ] {
            db.session().execute(write).unwrap();
        }
        let after: Vec<_> = forms
            .iter()
            .map(|query| reader.agree(query, stage))
            .collect();
        assert_eq!(after, before, "{stage}: the old reader's rows");
        reader.end();
        let now = Readers::begin(db);
        for query in &forms {
            now.agree(query, stage);
        }
        assert_eq!(
            rows_at(&now.agree(&on("> 50"), stage)),
            [4800, 4803],
            "{stage}"
        );
        now.end();
    };
    let db = GrafeoDB::new_in_memory();
    sliced(&db);
    check(&db, "hot");
    let (_dir, db) = reopened(sliced);
    check(&db, "reopened");
    db.close().unwrap();
}

/// A row whose newest labels lost a label its chain keeps is read for a reader of that view, and not
/// after, under the label's form, with the key's column hot and cold.
#[test]
fn a_row_that_lost_a_label_its_chain_keeps_is_read_at_the_view() {
    let fill = |db: &GrafeoDB| {
        let node = |labels: &[&str], row: i64, key: i64| {
            db.create_node_with_props(
                labels,
                [("row", Value::Int64(row)), ("k", Value::Int64(key))],
            )
            .unwrap();
        };
        for at in 0..GROUP + 8 {
            match at {
                3 => node(&["A"], at, 5),
                4 => node(&["B"], at, 5),
                5 => node(&["A", "B"], at, 5),
                // Far past the other rows of `A`, in a word of its own.
                40_000 => node(&["A", "Other"], at, 5),
                // The only row of `A` in the second group.
                _ if at == GROUP + 2 => node(&["A"], at, 5),
                _ if at == GROUP + 3 => node(&["B"], at, 6),
                _ => node(&["Other"], at, 6),
            }
        }
    };
    let forms = [
        "MATCH (n:A) WHERE n.k <= 5 RETURN n.row ORDER BY n.row",
        "MATCH (n:A) WHERE n.k >= 5 AND n.k < 6 RETURN n.row ORDER BY n.row",
        "MATCH (n:B) WHERE n.k < 7 RETURN n.row ORDER BY n.row",
    ];
    let check = |db: &GrafeoDB, stage: &str| {
        let reader = Readers::begin(db);
        let before: Vec<_> = forms
            .iter()
            .map(|query| reader.agree(query, stage))
            .collect();
        assert_eq!(rows_at(&before[0]), [3, 5, 40_000, GROUP + 2], "{stage}");
        for query in [
            "MATCH (n {row: 3}) REMOVE n:A".to_owned(),
            "MATCH (n {row: 40000}) REMOVE n:A".to_owned(),
            format!("MATCH (n {{row: {}}}) REMOVE n:A", GROUP + 2),
            format!("MATCH (n {{row: {}}}) SET n:A, n.k = 5", GROUP + 5),
        ] {
            db.session().execute(&query).unwrap();
        }
        let after: Vec<_> = forms
            .iter()
            .map(|query| reader.agree(query, stage))
            .collect();
        assert_eq!(after, before, "{stage}: the old reader's rows");
        reader.end();
        let now = Readers::begin(db);
        assert_eq!(
            rows_at(&now.agree(forms[0], stage)),
            [5, GROUP + 5],
            "{stage}: the new reader"
        );
        now.agree(forms[1], stage);
        now.agree(forms[2], stage);
        now.end();
    };
    let db = GrafeoDB::new_in_memory();
    fill(&db);
    check(&db, "hot");
    let (_dir, db) = reopened(fill);
    check(&db, "reopened");
    db.close().unwrap();
}

/// A kept plan reads the column at each run: its bounds are not those it saw when it was made.
#[test]
fn a_kept_plan_reads_the_column_again() {
    let db = GrafeoDB::new_in_memory();
    numbered(&db, 5, Value::Int64);
    let session = db.session();
    let literal = "MATCH (n:L) WHERE n.k >= 3 RETURN n.row";
    let parameter = "MATCH (n:L) WHERE n.k >= $v RETURN n.row";
    let count = |query: &str, bound: i64| {
        session
            .execute_with_params(query, one(&Value::Int64(bound)))
            .unwrap()
            .row_count()
    };
    assert_eq!(count(literal, 0), 2);
    assert_eq!(count(parameter, 3), 2);
    assert_eq!(count(parameter, 0), 5);
    session.execute("INSERT (:L {row: 5, k: 9})").unwrap();
    assert_eq!(count(literal, 0), 3);
    assert_eq!(count(parameter, 3), 3);
    session.execute("MATCH (n:L {row: 0}) SET n.k = 8").unwrap();
    assert_eq!(count(literal, 0), 4);
    assert_eq!(count(parameter, 8), 2);
    assert!(
        db.query_cache().stats().optimized_hits > 0,
        "the plan was kept, so this tests a kept plan"
    );
}

/// A clause after a write reads what the write left, not what the planner saw.
#[test]
fn a_range_after_a_write_of_its_statement_reads_what_the_write_left() {
    let cases = [
        (
            "MATCH (h:Hub) SET h.c = 4 WITH h OPTIONAL MATCH (t:Hub) WHERE t.c > 3 RETURN t.name",
            &["Alix"][..],
        ),
        (
            "MATCH (h:Hub) SET h.c = 4 WITH h MATCH (t:Hub) WHERE t.c >= 4 AND t.c < 5 RETURN t.name",
            &["Alix"][..],
        ),
        (
            "MATCH (h:Hub) SET h.c = 4 WITH h OPTIONAL MATCH (t) WHERE t.c > 3 RETURN t.name",
            &["Alix"][..],
        ),
        (
            "MATCH (h:Hub) SET h.c = 9 WITH h OPTIONAL MATCH (t:Hub) WHERE t.c < 2 RETURN t.name",
            &["null"][..],
        ),
        (
            "INSERT (:Hub {name: 'Gus', c: 8}) WITH 1 AS one OPTIONAL MATCH (t:Hub) WHERE t.c > 7 RETURN t.name",
            &["Gus"][..],
        ),
        (
            "MATCH (h:Hub) REMOVE h.c WITH h OPTIONAL MATCH (t:Hub) WHERE t.c < 2 RETURN t.name",
            &["null"][..],
        ),
    ];
    for (query, wanted) in cases {
        for rules in [PlannerRules::all(), PlannerRules::plain()] {
            let db = GrafeoDB::new_in_memory();
            let session = session_with(&db, rules);
            session
                .execute("INSERT (:Hub {name: 'Alix', c: 1})")
                .unwrap();
            assert_eq!(names_of(&session, query), wanted, "{query}");
        }
    }
}

/// A subquery's read runs from the start for each row, cut short by its limit or not.
#[test]
fn a_read_runs_again_for_each_row() {
    let rows = GROUP + 2_000;
    let fill = move |db: &GrafeoDB| numbered(db, rows, |row| Value::Int64(row % 7));
    let check = |db: &GrafeoDB, stage: &str| {
        let usual = db.session();
        let plain = session_with(db, PlannerRules::plain());
        for query in [
            "UNWIND [1, 2, 3] AS x CALL { MATCH (n:L) WHERE n.k > 3 RETURN n.row AS r LIMIT 3 } RETURN x, r",
            "UNWIND [1, 2, 3] AS x CALL { MATCH (n:L) WHERE n.k > 3 RETURN n.row AS r LIMIT 3000 } RETURN x, r",
            "UNWIND [1, 2] AS x CALL { MATCH (n:L) WHERE n.k > 3 RETURN n.row AS r SKIP 2000 LIMIT 3 } RETURN x, r",
            "UNWIND [1, 2] AS x OPTIONAL MATCH (n:L) WHERE n.k > 5 RETURN x, count(n)",
            "MATCH (a:L {row: 0}) MATCH (b:L) WHERE b.k > 5 RETURN count(*)",
        ] {
            assert_eq!(
                usual
                    .execute(query)
                    .unwrap_or_else(|error| panic!("{stage}: {query}: {error}"))
                    .rows(),
                plain.execute(query).unwrap().rows(),
                "{stage}: {query}"
            );
        }
    };
    let db = GrafeoDB::new_in_memory();
    fill(&db);
    check(&db, "hot");
    let (_dir, db) = reopened(fill);
    check(&db, "reopened");
    db.close().unwrap();
}

/// A reader's count is one snapshot: a writer moves rows across the bound two at a time, so the count
/// of each transaction stays what it was and never shows half a move.
#[test]
fn a_read_beside_a_writer_sees_one_snapshot() {
    let db = Arc::new(GrafeoDB::new_in_memory());
    numbered(&db, 2_000, |row| {
        Value::Int64(if row % 2 == 0 { 1 } else { 100 })
    });
    let stop = Arc::new(AtomicBool::new(false));
    let writer = {
        let db = Arc::clone(&db);
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            let mut session = db.session();
            let mut at = 0_i64;
            while !stop.load(Ordering::Relaxed) {
                let (down, up) = (at % 2_000, (at + 1) % 2_000);
                session.begin_transaction().unwrap();
                let moved = session
                    .execute(&format!("MATCH (n:L {{row: {down}}}) SET n.k = 100"))
                    .and_then(|_| {
                        session.execute(&format!("MATCH (n:L {{row: {up}}}) SET n.k = 1"))
                    });
                match moved {
                    Ok(_) => {
                        let _ = session.commit();
                    }
                    Err(_) => {
                        let _ = session.rollback();
                    }
                }
                at += 2;
            }
        })
    };
    let mut session = db.session();
    for _ in 0..200 {
        session.begin_transaction().unwrap();
        let low = session
            .execute("MATCH (n:L) WHERE n.k < 50 RETURN count(n)")
            .unwrap();
        let high = session
            .execute("MATCH (n:L) WHERE n.k >= 50 RETURN count(n)")
            .unwrap();
        assert_eq!(low.rows()[0][0], Value::Int64(1_000));
        assert_eq!(high.rows()[0][0], Value::Int64(1_000));
        session.rollback().unwrap();
    }
    stop.store(true, Ordering::Relaxed);
    writer.join().unwrap();
}

// ---------------------------------------------------------------------------
// Labels, limits and graphs
// ---------------------------------------------------------------------------

/// The first rows that pass, in the filter's order, though the first values in range are of another
/// label, and with a label only a later group holds.
#[test]
fn a_limit_gives_the_filters_rows() {
    let rows = GROUP * 2 + 100;
    let fill = move |db: &GrafeoDB| {
        for row in 0..rows {
            let label = if row < GROUP + 50 { "M" } else { "L" };
            db.create_node_with_props(
                &[label, "All"],
                [("row", Value::Int64(row)), ("k", Value::Int64(row % 11))],
            )
            .unwrap();
        }
    };
    let check = |db: &GrafeoDB, stage: &str| {
        let usual = db.session();
        let plain = session_with(db, PlannerRules::plain());
        for limit in [0, 1, 2, 10, 1_000, 70_000, 300_000] {
            for form in [
                format!("MATCH (n:L) WHERE n.k > 5 RETURN n.row LIMIT {limit}"),
                format!("MATCH (n:M) WHERE n.k > 5 RETURN n.row LIMIT {limit}"),
                format!("MATCH (n:All) WHERE n.k > 5 RETURN n.row LIMIT {limit}"),
                format!("MATCH (n) WHERE n.k > 5 RETURN n.row LIMIT {limit}"),
                format!("MATCH (n) WHERE n.k > 500 RETURN n.row LIMIT {limit}"),
                format!("MATCH (n:L) WHERE n.k > 5 RETURN n.row SKIP 3 LIMIT {limit}"),
                format!("MATCH (n:L) WHERE n.k > 5 RETURN n.row ORDER BY n.row DESC LIMIT {limit}"),
                format!("MATCH (n:L) WHERE n.k > 5 AND n.k < 8 RETURN n.row LIMIT {limit}"),
            ] {
                assert_eq!(
                    usual.execute(&form).unwrap().rows(),
                    plain.execute(&form).unwrap().rows(),
                    "{stage}: {form}"
                );
            }
        }
    };
    let db = GrafeoDB::new_in_memory();
    fill(&db);
    check(&db, "hot");
    let (_dir, db) = reopened(fill);
    check(&db, "reopened");
    db.close().unwrap();
}

/// A limit over a chain: the row an old reader sees in range comes in its place among the others.
#[test]
fn a_limit_over_a_chain_keeps_the_order() {
    let fill = |db: &GrafeoDB| numbered(db, GROUP + 100, |row| Value::Int64(row % 100));
    let check = |db: &GrafeoDB, stage: &str| {
        let query = "MATCH (n:L) WHERE n.k >= 50 RETURN n.row LIMIT 3";
        let reader = Readers::begin(db);
        assert_eq!(
            rows_at(&reader.agree(query, stage)),
            [50, 51, 52],
            "{stage}"
        );
        let writer = db.session();
        writer.execute("MATCH (n:L {row: 50}) SET n.k = 1").unwrap();
        writer.execute("MATCH (n:L {row: 3}) SET n.k = 70").unwrap();
        assert_eq!(
            rows_at(&reader.agree(query, stage)),
            [50, 51, 52],
            "{stage}: old"
        );
        reader.end();
        let now = Readers::begin(db);
        assert_eq!(
            rows_at(&now.agree(query, stage)),
            [3, 51, 52],
            "{stage}: new"
        );
        now.end();
    };
    let db = GrafeoDB::new_in_memory();
    fill(&db);
    check(&db, "hot");
    let (_dir, db) = reopened(fill);
    check(&db, "reopened");
    db.close().unwrap();
}

/// A node with several labels is one row under each, a label no node holds finds nothing and neither
/// does a key no node holds, with no error.
#[test]
fn labels_keys_and_several_labels() {
    let fill = |db: &GrafeoDB| {
        let session = db.session();
        session
            .execute(
                "INSERT (:A {row: 0, k: 5}), (:A:B {row: 1, k: 5}), (:B {row: 2, k: 5}), \
                 ({row: 3, k: 5}), (:A {row: 4, k: 1}), (:A {row: 5})",
            )
            .unwrap();
    };
    let cases: &[(&str, &[i64])] = &[
        ("MATCH (n:A) WHERE n.k > 1 RETURN n.row", &[0, 1]),
        ("MATCH (n:B) WHERE n.k > 1 RETURN n.row", &[1, 2]),
        ("MATCH (n) WHERE n.k > 1 RETURN n.row", &[0, 1, 2, 3]),
        ("MATCH (n:A:B) WHERE n.k > 1 RETURN n.row", &[1]),
        ("MATCH (n:A|B) WHERE n.k > 1 RETURN n.row", &[0, 1, 2]),
        ("MATCH (n:!A) WHERE n.k > 1 RETURN n.row", &[2, 3]),
        ("MATCH (n:Nobody) WHERE n.k > 1 RETURN n.row", &[]),
        ("MATCH (n:A) WHERE n.nothing > 1 RETURN n.row", &[]),
        ("MATCH (n) WHERE n.nothing < 1 RETURN n.row", &[]),
    ];
    let check = |db: &GrafeoDB, stage: &str| {
        for (query, wanted) in cases {
            for (name, session) in [
                ("read", db.session()),
                ("filter", session_with(db, PlannerRules::plain())),
            ] {
                let result = session
                    .execute(query)
                    .unwrap_or_else(|error| panic!("{stage}, {name}: {query}: {error}"));
                let mut found = rows_at(result.rows());
                found.sort_unstable();
                assert_eq!(found, *wanted, "{stage}, {name}: {query}");
            }
        }
    };
    let db = GrafeoDB::new_in_memory();
    fill(&db);
    check(&db, "hot");
    let (_dir, db) = reopened(fill);
    check(&db, "reopened");
    db.close().unwrap();
}

/// A graph's column is its own.
#[test]
fn a_graph_reads_its_own_column() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("INSERT (:L {name: 'Alix', k: 1})").unwrap();
    session.execute("CREATE GRAPH g").unwrap();
    session.execute("USE GRAPH g").unwrap();
    session
        .execute("INSERT (:L {name: 'Gus', k: 1}), (:L {name: 'Mia', k: 2})")
        .unwrap();
    let query = "MATCH (n:L) WHERE n.k >= 1 RETURN n.name";
    assert_eq!(names_of(&session, query), ["Gus", "Mia"]);
    assert_eq!(names_of(&db.session(), query), ["Alix"]);
}

/// A range on a hop's end and with an index on the key give the filter's rows.
#[test]
fn a_range_beside_a_hop_and_an_index() {
    let fill = |db: &GrafeoDB| {
        let session = db.session();
        session
            .execute(
                "UNWIND range(0, 999) AS i INSERT (a:Hub {row: i})-[:R]->(b:L {row: i, k: i % 10})",
            )
            .unwrap();
    };
    let check = |db: &GrafeoDB, stage: &str| {
        let usual = db.session();
        let plain = session_with(db, PlannerRules::plain());
        for query in [
            "MATCH (a:Hub {row: 3})-[:R]->(n:L) WHERE n.k > 2 RETURN n.row",
            "MATCH (a:Hub)-[:R]->(n:L) WHERE n.k > 7 RETURN a.row, n.row ORDER BY a.row",
            "MATCH (n:L)<-[:R]-(a:Hub) WHERE n.k >= 2 AND n.k < 4 RETURN a.row ORDER BY a.row",
            "MATCH (n:L) WHERE n.k > 8 MATCH (a:Hub)-[:R]->(n) RETURN a.row ORDER BY a.row",
        ] {
            assert_eq!(
                usual.execute(query).unwrap().rows(),
                plain.execute(query).unwrap().rows(),
                "{stage}: {query}"
            );
        }
    };
    let db = GrafeoDB::new_in_memory();
    fill(&db);
    check(&db, "hot");
    db.session()
        .execute("CREATE INDEX by_k FOR (n:L) ON (n.k)")
        .unwrap();
    check(&db, "hot, an index");
    let (_dir, db) = reopened(fill);
    check(&db, "reopened");
    db.session()
        .execute("CREATE INDEX by_k FOR (n:L) ON (n.k)")
        .unwrap();
    check(&db, "reopened, an index");
    db.close().unwrap();
}

/// `PROFILE` runs what the statement runs, and `EXPLAIN` shows the read for a plan `PROFILE` runs.
#[test]
fn explain_and_profile_agree() {
    let fill = |db: &GrafeoDB| numbered(db, 1_000, |row| Value::Int64(row % 10));
    let check = |db: &GrafeoDB, stage: &str| {
        let session = db.session();
        for query in [
            "MATCH (n:L) WHERE n.k > 7 RETURN n.row",
            "MATCH (n) WHERE n.k > 7 AND n.k <= 8 RETURN n.row LIMIT 5",
        ] {
            let plain = session.execute(query).unwrap();
            let explained = session.execute(&format!("EXPLAIN {query}")).unwrap();
            let plan = explained.rows()[0][0].as_str().unwrap();
            assert!(plan.contains("[range: k]"), "{stage}: {plan}");
            let profiled = session.execute(&format!("PROFILE {query}")).unwrap();
            let text = format!("{:?}", profiled.rows());
            assert!(text.contains("rows"), "{stage}: {text}");
            assert_eq!(
                session.execute(query).unwrap().rows(),
                plain.rows(),
                "{stage}: {query}"
            );
        }
    };
    let db = GrafeoDB::new_in_memory();
    fill(&db);
    check(&db, "hot");
    let (_dir, db) = reopened(fill);
    check(&db, "reopened");
    db.close().unwrap();
}
