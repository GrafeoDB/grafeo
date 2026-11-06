//! An equality on a key no index serves reads the key's column instead of filtering every node.
//!
//! The answer is the scan's and its filter's, for every kind of value and of key, hot, reopened and
//! thawed; a reader sees its own view; a clause after a write reads what the write left; and
//! `rows_scanned` says what was read. `key_column_read.gtest` runs the query cases on every binding,
//! and `key_holders` holds the label forms and the hops.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main key_column_read::
//! ```

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::query::optimizer::{PlannerRule, PlannerRules};
use grafeo_engine::session::Session;

/// Rows of a row group: a column past this spans groups.
const GROUP: i64 = 65_536;

fn row_value(row: usize) -> Value {
    Value::Int64(i64::try_from(row).unwrap())
}

fn every_value() -> Vec<Value> {
    let big = 9_007_199_254_740_993_i64;
    vec![
        Value::Int64(0),
        Value::Int64(1),
        Value::Int64(-1),
        Value::Int64(42),
        Value::Int64(big),
        Value::Int64(big - 1),
        Value::Int64(i64::MAX),
        Value::Int64(i64::MIN),
        Value::Float64(0.0),
        Value::Float64(-0.0),
        Value::Float64(1.0),
        Value::Float64(1.5),
        Value::Float64(42.0),
        Value::Float64(9_007_199_254_740_992.0),
        Value::Float64(9_223_372_036_854_775_808.0),
        Value::Float64(f64::NAN),
        Value::Float64(f64::INFINITY),
        Value::Float64(f64::NEG_INFINITY),
        Value::from("0"),
        Value::from("1"),
        Value::from("042"),
        Value::from("42"),
        Value::from("42.0"),
        Value::from("4.2e1"),
        Value::from("-0"),
        Value::from("NaN"),
        Value::from("Alix"),
        Value::from("alix"),
        Value::from(""),
        Value::Bool(true),
        Value::Bool(false),
        Value::List(vec![Value::Int64(1)].into()),
    ]
}

/// The columns a lookup must read alike: each kind alone, so it is read typed, and all in one.
fn columns() -> Vec<(&'static str, Vec<Value>)> {
    let values = every_value();
    let only = |test: fn(&Value) -> bool| {
        values
            .iter()
            .filter(|v| test(v))
            .cloned()
            .collect::<Vec<_>>()
    };
    vec![
        ("integers", only(|v| matches!(v, Value::Int64(_)))),
        ("floats", only(|v| matches!(v, Value::Float64(_)))),
        ("strings", only(|v| matches!(v, Value::String(_)))),
        ("booleans", only(|v| matches!(v, Value::Bool(_)))),
        ("all kinds", values.clone()),
    ]
}

fn hold(db: &GrafeoDB, values: &[Value]) {
    for (row, value) in values.iter().enumerate() {
        db.create_node_with_props(&["L"], [("row", row_value(row)), ("k", value.clone())])
            .unwrap();
    }
    // A node of another label holds each too: the label decides.
    for value in values {
        db.create_node_with_props(
            &["Other"],
            [("row", Value::Int64(-1)), ("k", value.clone())],
        )
        .unwrap();
    }
    // A node with no value at all.
    db.create_node_with_props(&["L"], [("row", Value::Int64(-2))])
        .unwrap();
}

fn session_with(db: &GrafeoDB, rules: PlannerRules) -> Session {
    let mut session = db.session();
    session.set_planner_rules(rules);
    session
}

/// The rows in the order they came.
fn rows_of(session: &Session, query: &str, key: &Value) -> Vec<String> {
    let params = HashMap::from([("v".to_string(), key.clone())]);
    let result = session
        .execute_with_params(query, params)
        .unwrap_or_else(|error| panic!("{query} with {key:?}: {error}"));
    result.rows().iter().map(|row| format!("{row:?}")).collect()
}

fn plan_of(session: &Session, query: &str, key: &Value) -> String {
    let params = HashMap::from([("v".to_string(), key.clone())]);
    let result = session
        .execute_with_params(&format!("EXPLAIN {query}"), params)
        .unwrap();
    result.rows()[0][0].as_str().unwrap().to_string()
}

/// Every key form over `db`: the column read's rows are the scan's, in the scan's order.
fn agrees_with_the_scan(db: &GrafeoDB, stage: &str) {
    let read = db.session();
    let scan = session_with(db, PlannerRules::plain());
    let mut keys = every_value();
    keys.push(Value::Null);
    for key in &keys {
        for query in [
            "MATCH (n:L {k: $v}) RETURN n.row",
            "MATCH (n:L) WHERE n.k = $v RETURN n.row",
            "MATCH (n {k: $v}) RETURN n.row",
            "MATCH (n:L) WHERE n.k IN [$v] RETURN n.row",
            "MATCH (n:L) WHERE n.k IN [$v, 'Gus'] RETURN n.row",
            "MATCH (n:L) WHERE n.k IN [$v, 42, 1.5] RETURN n.row",
            "MATCH (n:L {k: $v}) RETURN n.row LIMIT 1",
            "OPTIONAL MATCH (n:L {k: $v}) RETURN n.row",
        ] {
            assert_eq!(
                rows_of(&read, query, key),
                rows_of(&scan, query, key),
                "{stage}: {query} with {key:?}\n{}",
                plan_of(&read, query, key)
            );
        }
    }
}

#[test]
fn every_kind_of_key_finds_what_the_scan_finds_hot() {
    for (name, values) in columns() {
        let db = GrafeoDB::new_in_memory();
        hold(&db, &values);
        agrees_with_the_scan(&db, name);
    }
}

#[test]
fn every_kind_of_key_finds_what_the_scan_finds_reopened() {
    for (name, values) in columns() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("kinds.grafeo");
        {
            let db = GrafeoDB::open(&path).unwrap();
            hold(&db, &values);
            db.close().unwrap();
        }
        let db = GrafeoDB::open(&path).unwrap();
        agrees_with_the_scan(&db, name);
        db.close().unwrap();
    }
}

/// A write thaws the column it writes to: the values held before it read the same, and a write of
/// another kind leaves them right too.
#[test]
fn every_kind_of_key_finds_what_the_scan_finds_thawed() {
    for (name, values) in columns() {
        for written in [Value::Int64(7), Value::from("Gus"), Value::Float64(2.5)] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("thawed.grafeo");
            {
                let db = GrafeoDB::open(&path).unwrap();
                hold(&db, &values);
                db.close().unwrap();
            }
            let db = GrafeoDB::open(&path).unwrap();
            let node = db
                .create_node_with_props(&["L"], [("row", Value::Int64(-3)), ("k", written.clone())])
                .unwrap();
            agrees_with_the_scan(&db, &format!("{name}, {written:?}"));
            assert_eq!(
                rows_of(&db.session(), "MATCH (n:L {k: $v}) RETURN n.row", &written)
                    .last()
                    .map(String::as_str),
                Some("[Int64(-3)]"),
                "{name}: the new row {node:?} is found"
            );
            db.close().unwrap();
        }
    }
}

/// The values found by `=` across kinds, written out: the agreement above would hold of a wrong scan.
#[test]
fn a_key_finds_the_values_equal_to_it_in_each_kind() {
    let db = GrafeoDB::new_in_memory();
    hold(&db, &every_value());
    let session = db.session();
    let names = every_value();
    let found = |key: Value| -> Vec<String> {
        let mut rows = rows_of(&session, "MATCH (n:L {k: $v}) RETURN n.row", &key);
        rows.sort();
        rows.iter()
            .map(|row| {
                let row: usize = row
                    .trim_start_matches("[Int64(")
                    .trim_end_matches(")]")
                    .parse()
                    .unwrap();
                format!("{:?}", names[row])
            })
            .collect()
    };
    assert_eq!(
        found(Value::Int64(1)).len(),
        found(Value::Float64(1.0)).len()
    );
    let at_42 = found(Value::Int64(42));
    for value in ["Int64(42)", "Float64(42)"] {
        assert!(at_42.iter().any(|row| row == value), "{value} in {at_42:?}");
    }
    // A string equals no number, whatever it spells (GA04).
    for value in ["String(\"42\")", "String(\"042\")"] {
        assert!(
            !at_42.iter().any(|row| row == value),
            "{value} not in {at_42:?}"
        );
    }
    assert_eq!(found(Value::from("042")), ["String(\"042\")"]);
    assert!(
        found(Value::Float64(f64::NAN)).is_empty(),
        "NaN equals nothing"
    );
    let zero = found(Value::Float64(0.0));
    assert!(zero.iter().any(|row| row == "Float64(-0)"), "{zero:?}");
    assert!(zero.iter().any(|row| row == "Int64(0)"), "{zero:?}");
    assert_eq!(found(Value::from("Alix")), ["String(\"Alix\")"]);
    assert_eq!(found(Value::Bool(true)), ["Bool(true)"]);
    assert!(found(Value::Null).is_empty(), "NULL equals nothing");
}

// ---------------------------------------------------------------------------
// The plan
// ---------------------------------------------------------------------------

fn numbered(db: &GrafeoDB, rows: i64, value: impl Fn(i64) -> Value) {
    for row in 0..rows {
        db.create_node_with_props(&["L"], [("row", Value::Int64(row)), ("k", value(row))])
            .unwrap();
    }
}

fn plan(session: &Session, query: &str) -> String {
    session.execute(&format!("EXPLAIN {query}")).unwrap().rows()[0][0]
        .as_str()
        .unwrap()
        .to_string()
}

/// `IN` and `=`, a label or none, read the column; each way off the rule is the scan's filter.
#[test]
fn an_equality_shows_the_read_and_the_rule_takes_it_away() {
    let db = GrafeoDB::new_in_memory();
    numbered(&db, 10, Value::Int64);
    let usual = db.session();
    let off = session_with(
        &db,
        PlannerRules::all().with(PlannerRule::KeyColumnLookup, false),
    );
    let plain = session_with(&db, PlannerRules::plain());
    for query in [
        "MATCH (n:L {k: 5}) RETURN n.row",
        "MATCH (n:L) WHERE n.k = 5 RETURN n.row",
        "MATCH (n {k: 5}) RETURN n.row",
        "MATCH (n:L) WHERE n.k IN [5, 6] RETURN n.row",
        "MATCH (n:L) WHERE n.k = 5 AND n.row > 1 RETURN n.row",
    ] {
        let on = plan(&usual, query);
        assert!(on.contains("[key column: k]"), "{query}\n{on}");
        for session in [&off, &plain] {
            let without = plan(session, query);
            assert!(!without.contains("[key column"), "{query}\n{without}");
            assert_eq!(
                session.execute(query).unwrap().rows(),
                usual.execute(query).unwrap().rows(),
                "{query}"
            );
        }
    }
    for query in [
        "MATCH (n:L) WHERE n.k > 5 RETURN n.row",
        "MATCH (n:L) WHERE n.k <> 5 RETURN n.row",
        "MATCH (n:L) WHERE n.k = 5 OR n.k = 6 RETURN n.row",
        "MATCH (n:L) WHERE NOT n.k = 5 RETURN n.row",
        "MATCH (n:L) WHERE n.k = n.row RETURN n.row",
        "MATCH (n:L) RETURN n.row",
    ] {
        let shown = plan(&usual, query);
        assert!(!shown.contains("[key column"), "{query}\n{shown}");
    }
}

/// With an index the index is read, and an index that covers part of the nodes reads the rest.
#[test]
fn an_index_goes_first() {
    let db = GrafeoDB::new_in_memory();
    numbered(&db, 10, Value::Int64);
    let session = db.session();
    session
        .execute("CREATE INDEX on_l FOR (n:L) ON (n.k)")
        .unwrap();
    let shown = plan(&session, "MATCH (n:L {k: 5}) RETURN n.row");
    assert!(
        shown.contains("[index: k]") && !shown.contains("[key column"),
        "{shown}"
    );
    let shown = plan(&session, "MATCH (n {k: 5}) RETURN n.row");
    assert!(shown.contains("[indexes and remainder: k]"), "{shown}");
}

/// Of two equalities, the one with an index, whichever comes first, and without one the first
/// written; the other is checked on what it finds.
#[test]
fn of_two_keys_the_read_answers_with_both() {
    let db = GrafeoDB::new_in_memory();
    for row in 0..100_i64 {
        db.create_node_with_props(
            &["L"],
            [
                ("row", Value::Int64(row)),
                ("a", Value::Int64(row % 10)),
                ("b", Value::Int64(row % 7)),
            ],
        )
        .unwrap();
    }
    let session = db.session();
    let query = "MATCH (n:L {a: 3, b: 3}) RETURN n.row ORDER BY n.row";
    let wanted = [vec![Value::Int64(3)], vec![Value::Int64(73)]];
    assert_eq!(session.execute(query).unwrap().rows(), wanted);
    assert!(plan(&session, query).contains("[key column: a]"));
    assert!(plan(&session, "MATCH (n:L {b: 3, a: 3}) RETURN n.row").contains("[key column: b]"));
    session
        .execute("CREATE INDEX on_b FOR (n:L) ON (n.b)")
        .unwrap();
    let shown = plan(&session, query);
    assert!(shown.contains("[index: b]"), "{shown}");
    assert_eq!(session.execute(query).unwrap().row_count(), 2);
    // A key no node holds finds nothing, whichever comes first.
    assert_eq!(
        session
            .execute("MATCH (n:L {none: 3, a: 3}) RETURN n.row")
            .unwrap()
            .row_count(),
        0
    );
    assert_eq!(
        session
            .execute("MATCH (n:L {a: 3, none: 3}) RETURN n.row")
            .unwrap()
            .row_count(),
        0
    );
}

/// Each form of label narrows the read and the answer alike.
#[test]
fn every_label_form_reads_only_its_own_nodes() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "INSERT (:A {name: 'Alix', k: 1}), (:B {name: 'Gus', k: 1}), (:A:B {name: 'Vincent', k: 1}), \
             (:C {name: 'Jules', k: 1}), ({name: 'Mia', k: 1}), (:A {name: 'Butch', k: 2})",
        )
        .unwrap();
    let plain = session_with(&db, PlannerRules::plain());
    for (query, wanted) in [
        ("MATCH (n:A {k: 1}) RETURN n.name", 2),
        ("MATCH (n:A&B {k: 1}) RETURN n.name", 1),
        ("MATCH (n:A|B {k: 1}) RETURN n.name", 3),
        ("MATCH (n:!A {k: 1}) RETURN n.name", 3),
        ("MATCH (n:% {k: 1}) RETURN n.name", 4),
        ("MATCH (n {k: 1}) RETURN n.name", 5),
        ("MATCH (n:Absent {k: 1}) RETURN n.name", 0),
        ("MATCH (n:A|Absent {k: 1}) RETURN n.name", 2),
        ("MATCH (n:A&Absent {k: 1}) RETURN n.name", 0),
    ] {
        let read = session.execute(query).unwrap();
        assert_eq!(
            read.row_count(),
            wanted,
            "{query}\n{}",
            plan(&session, query)
        );
        assert_eq!(read.rows(), plain.execute(query).unwrap().rows(), "{query}");
    }
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
        .execute("INSERT (:L {name: 'Gus', k: 1}), (:L {name: 'Mia', k: 1})")
        .unwrap();
    assert_eq!(
        session
            .execute("MATCH (n:L {k: 1}) RETURN n.name")
            .unwrap()
            .row_count(),
        2
    );
    assert_eq!(
        db.session()
            .execute("MATCH (n:L {k: 1}) RETURN n.name")
            .unwrap()
            .row_count(),
        1
    );
}

// ---------------------------------------------------------------------------
// A reader's view
// ---------------------------------------------------------------------------

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

/// The values of each kind a view is checked over, written as GQL: first, second and a third no
/// node holds.
const KINDS: [(&str, &str, &str); 4] = [
    ("1", "2", "3"),
    ("1.5", "2.5", "3.5"),
    ("'Amsterdam'", "'Berlin'", "'Paris'"),
    ("true", "false", "'none'"),
];

/// `db` holding Alix, Gus and Mia at `first` and more rows of the kind, hot or reopened.
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

#[test]
fn an_old_snapshot_reads_what_it_saw_after_a_change_of_value() {
    each_stage(|db, (first, second, _)| {
        let on = |value: &str| format!("MATCH (n:L {{k: {value}}}) RETURN n.name");
        let mut reader = db.session();
        reader.begin_transaction().unwrap();
        assert_eq!(names_of(&reader, &on(first)), ["Alix", "Gus", "Mia"]);
        db.session()
            .execute(&format!("MATCH (n:L {{name: 'Alix'}}) SET n.k = {second}"))
            .unwrap();
        assert_eq!(
            names_of(&reader, &on(first)),
            ["Alix", "Gus", "Mia"],
            "{first}"
        );
        assert_eq!(names_of(&reader, &on(second)), ["Jules"], "{second}");
        reader.rollback().unwrap();
        assert_eq!(names_of(&reader, &on(first)), ["Gus", "Mia"], "{first}");
        assert_eq!(
            names_of(&reader, &on(second)),
            ["Alix", "Jules"],
            "{second}"
        );
    });
}

#[test]
fn an_old_snapshot_reads_what_it_saw_after_a_delete_and_an_insert() {
    each_stage(|db, (first, _, _)| {
        let query = format!("MATCH (n:L {{k: {first}}}) RETURN n.name");
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
        let on_l = format!("MATCH (n:L {{k: {first}}}) RETURN n.name");
        let on_m = format!("MATCH (n:M {{k: {first}}}) RETURN n.name");
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
        let query = format!("MATCH (n:L {{k: {first}}}) RETURN n.name");
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

#[test]
fn a_transaction_reads_its_own_changes_and_no_one_elses() {
    each_stage(|db, (first, second, third)| {
        let on = |value: &str| format!("MATCH (n:L {{k: {value}}}) RETURN n.name");
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
            names_of(&writer, &on(first)),
            ["Mia", "Vincent"],
            "own, {first}"
        );
        assert_eq!(names_of(&writer, &on(second)), ["Alix"], "own, {second}");
        assert_eq!(names_of(&writer, &on(third)), ["Jules"], "own, {third}");
        let other = db.session();
        assert_eq!(
            names_of(&other, &on(first)),
            ["Alix", "Gus", "Mia"],
            "other, {first}"
        );
        assert_eq!(names_of(&other, &on(second)), ["Jules"], "other, {second}");
        assert_eq!(
            names_of(&other, &on(third)),
            Vec::<String>::new(),
            "other, {third}"
        );
        writer.rollback().unwrap();
        assert_eq!(
            names_of(&writer, &on(first)),
            ["Alix", "Gus", "Mia"],
            "after, {first}"
        );
        assert_eq!(names_of(&writer, &on(second)), ["Jules"], "after, {second}");
        assert_eq!(
            names_of(&writer, &on(third)),
            Vec::<String>::new(),
            "after, {third}"
        );
    });
}

/// A plan kept for a statement holds no nodes: the next call reads the column as it then stands.
#[test]
fn a_kept_plan_reads_the_column_again() {
    let db = GrafeoDB::new_in_memory();
    numbered(&db, 5, |_| Value::Int64(1));
    let session = db.session();
    let literal = "MATCH (n:L {k: 1}) RETURN n.row";
    let parameter = "MATCH (n:L {k: $v}) RETURN n.row";
    assert_eq!(session.execute(literal).unwrap().row_count(), 5);
    assert_eq!(rows_of(&session, parameter, &Value::Int64(1)).len(), 5);
    session.execute("INSERT (:L {row: 5, k: 1})").unwrap();
    assert_eq!(session.execute(literal).unwrap().row_count(), 6);
    assert_eq!(rows_of(&session, parameter, &Value::Int64(1)).len(), 6);
    session.execute("MATCH (n:L {row: 0}) SET n.k = 2").unwrap();
    assert_eq!(session.execute(literal).unwrap().row_count(), 5);
    assert_eq!(rows_of(&session, parameter, &Value::Int64(2)).len(), 1);
    assert!(
        db.query_cache().stats().optimized_hits > 0,
        "the plan was kept, so this tests a kept plan"
    );
}

// ---------------------------------------------------------------------------
// After a write of the statement
// ---------------------------------------------------------------------------

/// A clause after a write reads what the write left, not what the planner saw: a lookup made while
/// planning finds nothing the statement writes. Every form: an equality and an `IN`, with no index,
/// with an index, and with an index for part of the nodes.
#[test]
fn a_lookup_after_a_write_of_its_statement_reads_what_the_write_left() {
    let setups: [(&str, &[&str]); 3] = [
        ("no index", &[]),
        ("an index", &["CREATE INDEX on_hub FOR (n:Hub) ON (n.c)"]),
        (
            "an index for one label",
            &["CREATE INDEX on_other FOR (n:Other) ON (n.c)"],
        ),
    ];
    let cases = [
        (
            "MATCH (h:Hub) SET h.c = 4 WITH h OPTIONAL MATCH (t:Hub {c: 4}) RETURN t.name",
            &["Alix"][..],
        ),
        (
            "MATCH (h:Hub) SET h.c = 4 WITH h OPTIONAL MATCH (t:Hub) WHERE t.c IN [4, 5] RETURN t.name",
            &["Alix"][..],
        ),
        (
            "MATCH (h:Hub) SET h.c = 4 WITH h OPTIONAL MATCH (t {c: 4}) RETURN t.name",
            &["Alix"][..],
        ),
        (
            "MATCH (h:Hub) SET h.c = 4 WITH h OPTIONAL MATCH (t) WHERE t.c IN [4, 5] RETURN t.name",
            &["Alix"][..],
        ),
        (
            "MATCH (h:Hub) SET h.c = 4 WITH h MATCH (t:Hub {c: 4}) RETURN t.name",
            &["Alix"][..],
        ),
        (
            "MATCH (h:Hub) SET h.c = 4 WITH h MATCH (t:Hub) WHERE t.c IN [4, 5] RETURN t.name",
            &["Alix"][..],
        ),
        (
            "MATCH (h:Hub) SET h.c = 9 WITH h OPTIONAL MATCH (t:Hub {c: 1}) RETURN t.name",
            &["null"][..],
        ),
        (
            "MATCH (h:Hub) SET h.c = 9 WITH h OPTIONAL MATCH (t:Hub) WHERE t.c IN [1, 2] RETURN t.name",
            &["null"][..],
        ),
        (
            "INSERT (:Hub {name: 'Gus', c: 8}) WITH 1 AS one OPTIONAL MATCH (t:Hub {c: 8}) RETURN t.name",
            &["Gus"][..],
        ),
        (
            "INSERT (:Hub {name: 'Gus', c: 8}) WITH 1 AS one OPTIONAL MATCH (t:Hub) WHERE t.c IN [8] RETURN t.name",
            &["Gus"][..],
        ),
    ];
    for (setup, statements) in setups {
        for (query, wanted) in cases {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();
            session
                .execute("INSERT (:Hub {name: 'Alix', c: 1})")
                .unwrap();
            for statement in statements {
                session.execute(statement).unwrap();
            }
            assert_eq!(names_of(&session, query), wanted, "{setup}: {query}");
        }
    }
}

// ---------------------------------------------------------------------------
// What `rows_scanned` says
// ---------------------------------------------------------------------------

fn scanned(session: &Session, query: &str) -> u64 {
    session.execute(query).unwrap().rows_scanned().unwrap()
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

/// Rows in order across three groups: a value in one group reads no other, and a value beyond them
/// all reads nothing; shuffled, the bounds of every group hold it and every row is read.
#[test]
fn a_group_the_bounds_rule_out_adds_nothing() {
    let rows = GROUP * 3;
    let in_order = |db: &GrafeoDB| numbered(db, rows, Value::Int64);
    let shuffled = |db: &GrafeoDB| numbered(db, rows, |row| Value::Int64(row * 7_919 % rows));

    let db = GrafeoDB::new_in_memory();
    in_order(&db);
    let session = db.session();
    assert_eq!(
        scanned(&session, "MATCH (n:L {k: 5}) RETURN n.row"),
        GROUP as u64,
        "hot, one group"
    );
    assert_eq!(
        scanned(&session, "MATCH (n:L {k: 5000000}) RETURN n.row"),
        0,
        "hot, beyond"
    );

    let (_dir, db) = reopened(in_order);
    let session = db.session();
    let one = scanned(&session, "MATCH (n:L {k: 5}) RETURN n.row");
    assert!(
        one > 0 && one <= GROUP as u64,
        "reopened, one chunk or group: {one}"
    );
    assert_eq!(
        scanned(&session, "MATCH (n:L {k: 5000000}) RETURN n.row"),
        0,
        "reopened, beyond"
    );
    assert_eq!(
        session
            .execute("MATCH (n:L {k: 5}) RETURN n.row")
            .unwrap()
            .row_count(),
        1
    );
    db.close().unwrap();

    let db = GrafeoDB::new_in_memory();
    shuffled(&db);
    let session = db.session();
    assert_eq!(
        scanned(&session, "MATCH (n:L {k: 5}) RETURN n.row"),
        rows.unsigned_abs(),
        "hot, shuffled"
    );
    assert_eq!(
        scanned(&session, "MATCH (n:L {k: 5000000}) RETURN n.row"),
        0,
        "hot, beyond shuffled"
    );

    let (_dir, db) = reopened(shuffled);
    let session = db.session();
    assert_eq!(
        scanned(&session, "MATCH (n:L {k: 5}) RETURN n.row"),
        rows.unsigned_abs(),
        "reopened, shuffled"
    );
    assert_eq!(
        scanned(&session, "MATCH (n:L {k: 5000000}) RETURN n.row"),
        0,
        "reopened, beyond shuffled"
    );
    db.close().unwrap();
}

/// Only rows with a value are read: a column on half the rows reads half, hot and reopened.
#[test]
fn a_row_without_a_value_is_not_read() {
    let fill = |db: &GrafeoDB| {
        for row in 0..2_000_i64 {
            if row % 2 == 0 {
                db.create_node_with_props(&["L"], [("k", Value::Int64(row * 7 % 1_999))])
                    .unwrap();
            } else {
                db.create_node_with_props(&["L"], [("other", Value::Int64(row))])
                    .unwrap();
            }
        }
    };
    let db = GrafeoDB::new_in_memory();
    fill(&db);
    assert_eq!(
        scanned(&db.session(), "MATCH (n:L {k: 7}) RETURN n.k"),
        1_000,
        "hot"
    );
    let (_dir, db) = reopened(fill);
    assert_eq!(
        scanned(&db.session(), "MATCH (n:L {k: 7}) RETURN n.k"),
        1_000,
        "reopened"
    );
    db.close().unwrap();
}

/// Rows of other labels hold the key too, and are not read.
#[test]
fn a_row_of_another_label_is_not_read() {
    let fill = |db: &GrafeoDB| {
        for row in 0..2_000_i64 {
            let label = if row % 4 == 0 { "L" } else { "M" };
            db.create_node_with_props(&[label], [("k", Value::Int64(row * 7 % 1_999))])
                .unwrap();
        }
    };
    let db = GrafeoDB::new_in_memory();
    fill(&db);
    assert_eq!(
        scanned(&db.session(), "MATCH (n:L {k: 7}) RETURN n.k"),
        500,
        "hot"
    );
    assert_eq!(
        scanned(&db.session(), "MATCH (n {k: 7}) RETURN n.k"),
        2_000,
        "hot, no label"
    );
    let (_dir, db) = reopened(fill);
    assert_eq!(
        scanned(&db.session(), "MATCH (n:L {k: 7}) RETURN n.k"),
        500,
        "reopened"
    );
    assert_eq!(
        scanned(&db.session(), "MATCH (n {k: 7}) RETURN n.k"),
        2_000,
        "reopened, no label"
    );
    db.close().unwrap();
}

/// A row whose node went still holds its value, and was read to learn that; a row with a chain is
/// read at the view, once.
#[test]
fn a_deleted_row_that_holds_a_value_is_read_and_not_found() {
    let db = GrafeoDB::new_in_memory();
    numbered(&db, 100, |row| Value::Int64(row % 2));
    let session = db.session();
    session
        .execute("MATCH (n:L) WHERE n.row < 10 DELETE n")
        .unwrap();
    let result = session.execute("MATCH (n:L {k: 1}) RETURN n.row").unwrap();
    assert_eq!(result.row_count(), 45);
    assert_eq!(
        result.rows_scanned(),
        Some(100),
        "held rows, the deleted ones among them"
    );
    session
        .execute("MATCH (n:L {row: 20}) SET n.k = 1")
        .unwrap();
    let result = session.execute("MATCH (n:L {k: 1}) RETURN n.row").unwrap();
    assert_eq!(result.row_count(), 46);
    assert_eq!(
        result.rows_scanned(),
        Some(100),
        "a changed row counts once"
    );
}

/// The read decodes into a temporary: reading a cold column leaves it cold and adds no memory.
#[test]
#[expect(
    deprecated,
    reason = "reads the store's figures, and writes nothing through it"
)]
fn a_read_leaves_the_column_cold() {
    let fill = |db: &GrafeoDB| {
        for row in 0..5_000_i64 {
            db.create_node_with_props(
                &["L"],
                [
                    ("k", Value::Int64(row * 7 % 4_999)),
                    ("s", Value::from(format!("name-{row}").as_str())),
                ],
            )
            .unwrap();
        }
    };
    let (_dir, db) = reopened(fill);
    let store = db.store().unwrap();
    let cold = store.cold_columns();
    assert!(cold >= 2, "{cold}");
    let session = db.session();
    assert_eq!(
        session
            .execute("MATCH (n:L {k: 7}) RETURN count(n)")
            .unwrap()
            .row_count(),
        1
    );
    assert_eq!(store.cold_columns(), cold, "an integer column");
    assert_eq!(
        session
            .execute("MATCH (n:L {s: 'nobody'}) RETURN count(n)")
            .unwrap()
            .row_count(),
        1
    );
    assert_eq!(
        store.cold_columns(),
        cold,
        "a string column, a key it does not hold"
    );
    db.close().unwrap();
}

// ---------------------------------------------------------------------------
// A lazy read: nodes and values the statement writes while it reads
// ---------------------------------------------------------------------------

/// Enough holders to cross a group, so a write of a row in a later group can reach the read.
const CROSSING: i64 = GROUP + 2_000;

/// A key every holder shares, rows crossing a group.
fn crossing(db: &GrafeoDB) {
    numbered(db, CROSSING, |_| Value::Int64(1));
}

fn single(session: &Session, query: &str) -> Value {
    let result = session
        .execute(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"));
    result.rows()[0][0].clone()
}

/// A node the statement creates holds the key the read looks for, and is not read: each row found
/// makes one, and the statement ends.
#[test]
fn a_node_the_statement_creates_is_not_read() {
    for (name, query) in [
        ("insert", "MATCH (n:L {k: 1}) INSERT (:L {k: 1, row: -1})"),
        (
            "insert of another label",
            "MATCH (n {k: 1}) INSERT (:X {k: 1, row: -1})",
        ),
        ("merge", "MATCH (n {k: 1}) MERGE (:X {k: 1, row: n.row})"),
        (
            "where",
            "MATCH (n:L) WHERE n.k = 1 INSERT (:L {k: 1, row: -1})",
        ),
        (
            "call per row",
            "MATCH (n:L {k: 1}) CALL { WITH n INSERT (:L {k: 1, row: -1}) } RETURN count(*)",
        ),
        (
            "in",
            "MATCH (n:L) WHERE n.k IN [1, 2] INSERT (:L {k: 1, row: -1})",
        ),
    ] {
        let db = GrafeoDB::new_in_memory();
        crossing(&db);
        let session = db.session();
        session
            .execute(query)
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(
            single(&session, "MATCH (n {k: 1}) RETURN count(n)"),
            Value::Int64(2 * CROSSING),
            "{name}: one node per row found, none read again"
        );
    }
}

/// The statement's writes on rows the read has not reached answer as a scan's do, whichever way
/// they go; a `plain` session runs the scan.
#[test]
fn a_write_ahead_of_the_read_answers_as_a_scan_does() {
    for query in [
        // Moves a later row out of the match.
        "MATCH (n:L {k: 1}) WHERE n.row = 5 MATCH (m:L {row: 65540}) SET m.k = 2",
        "MATCH (n:L {k: 1}) MATCH (m:L {row: n.row + 65536}) SET m.k = 2 RETURN count(*)",
        // Moves a later row into the match.
        "MATCH (n:L {k: 2}) MATCH (m:L {row: n.row + 65536}) SET m.k = 2 RETURN count(*)",
        // Takes a later row away.
        "MATCH (n:L {k: 1}) MATCH (m:L {row: n.row + 65536}) DELETE m RETURN count(*)",
        // Changes what it found.
        "MATCH (n:L {k: 1}) SET n.k = 2 RETURN count(*)",
        "MATCH (n:L {k: 1}) SET n.k = 'Gus' RETURN count(*)",
        "MATCH (n:L {k: 1}) REMOVE n.k RETURN count(*)",
        "MATCH (n:L {k: 1}) REMOVE n:L RETURN count(*)",
        "MATCH (n:L {k: 1}) DELETE n RETURN count(*)",
    ] {
        let answer = |rules: PlannerRules| {
            let db = GrafeoDB::new_in_memory();
            // Ten rows at 1 and ten at 2 in the first group; in the second, those a row of the
            // first reaches at `row + 65536`: at 1 for the first ten, and at 3 for the others.
            numbered(&db, CROSSING, |row| {
                Value::Int64(match row {
                    0..10 | 65_536..65_546 => 1,
                    10..20 => 2,
                    _ => 3,
                })
            });
            let session = session_with(&db, rules);
            let rows = session
                .execute(query)
                .unwrap_or_else(|error| panic!("{query}: {error}"))
                .rows()
                .to_vec();
            let left = db.session();
            (
                rows,
                single(&left, "MATCH (n) RETURN count(n)"),
                single(&left, "MATCH (n) WHERE n.k = 1 RETURN count(n)"),
                single(&left, "MATCH (n) WHERE n.k = 2 RETURN count(n)"),
            )
        };
        assert_eq!(
            answer(PlannerRules::all()),
            answer(PlannerRules::plain()),
            "{query}"
        );
    }
}

/// The lookup holds what it read across a re-run: a subquery reads it from the start for each row.
#[test]
fn a_read_runs_again_for_each_row_of_a_subquery() {
    let db = GrafeoDB::new_in_memory();
    crossing(&db);
    let session = db.session();
    for (query, wanted) in [
        (
            "UNWIND [1, 2, 3] AS x CALL { MATCH (n:L {k: 1}) RETURN n.row AS r } RETURN count(*)",
            3 * CROSSING,
        ),
        (
            "UNWIND [1, 2] AS x OPTIONAL MATCH (n:L {k: 1}) RETURN count(n)",
            2 * CROSSING,
        ),
        (
            "MATCH (a:L {row: 0}) MATCH (b:L {k: 1}) RETURN count(*)",
            CROSSING,
        ),
    ] {
        assert_eq!(single(&session, query), Value::Int64(wanted), "{query}");
    }
}

/// `LIMIT` stops the read: what `rows_scanned` says falls with it, and the rows are the first ones.
#[test]
fn a_limit_stops_the_read_at_its_group() {
    let db = GrafeoDB::new_in_memory();
    numbered(&db, GROUP * 3, |_| Value::Int64(1));
    let session = db.session();
    let whole = session.execute("MATCH (n:L {k: 1}) RETURN n.row").unwrap();
    assert_eq!(whole.rows_scanned(), Some(GROUP as u64 * 3));
    let limited = session
        .execute("MATCH (n:L {k: 1}) RETURN n.row LIMIT 10")
        .unwrap();
    assert_eq!(limited.row_count(), 10);
    assert_eq!(limited.rows_scanned(), Some(GROUP as u64), "one group");
    assert_eq!(
        limited.rows()[9][0],
        Value::Int64(9),
        "the first rows, in order"
    );
}

/// A read with an answer it need not check again still checks the rest of the predicate.
#[test]
fn an_exact_answer_keeps_the_other_checks() {
    let db = GrafeoDB::new_in_memory();
    for row in 0..100_i64 {
        db.create_node_with_props(
            &["L"],
            [("k", Value::Int64(row % 2)), ("j", Value::Int64(row % 3))],
        )
        .unwrap();
        db.create_node_with_props(
            &["M"],
            [
                ("k", Value::Float64((row % 2) as f64)),
                ("j", Value::Int64(row % 3)),
            ],
        )
        .unwrap();
    }
    let session = db.session();
    let plain = session_with(&db, PlannerRules::plain());
    for query in [
        "MATCH (n:L {k: 1}) WHERE n.j = 2 RETURN count(n)",
        "MATCH (n:L {k: 1, j: 2}) RETURN count(n)",
        "MATCH (n:L {k: 1.0}) WHERE n.j > 0 RETURN count(n)",
        "MATCH (n:L) WHERE n.k = 1 AND NOT n.j = 2 RETURN count(n)",
        "MATCH (n:M {k: 1}) WHERE n.j = 2 RETURN count(n)",
        "MATCH (n {k: 1}) WHERE n.j = 2 RETURN count(n)",
        "MATCH (n:L) WHERE n.k = 1 OR n.j = 2 RETURN count(n)",
        "MATCH (n:L) WHERE n.k IN [1] AND n.j = 2 RETURN count(n)",
        "MATCH (n:L&!M {k: 1}) RETURN count(n)",
    ] {
        assert_eq!(single(&session, query), single(&plain, query), "{query}");
    }
}

/// An index for part of the nodes and the column for the rest, with a found node in a group of each.
#[test]
fn indexed_and_column_nodes_merge_in_order_across_groups() {
    let db = GrafeoDB::new_in_memory();
    for row in 0..CROSSING {
        let label = if row % 2 == 0 { "A" } else { "B" };
        db.create_node_with_props(
            &[label],
            [("row", Value::Int64(row)), ("k", Value::Int64(row % 1_000))],
        )
        .unwrap();
    }
    let session = db.session();
    session
        .execute("CREATE INDEX on_a FOR (n:A) ON (n.k)")
        .unwrap();
    let query = "MATCH (n {k: 7}) RETURN n.row";
    assert!(plan(&session, query).contains("[indexes and remainder: k]"));
    let rows = session.execute(query).unwrap();
    let plain = session_with(&db, PlannerRules::plain());
    assert_eq!(rows.rows(), plain.execute(query).unwrap().rows());
    assert_eq!(
        i64::try_from(rows.row_count()).unwrap(),
        (CROSSING + 993) / 1_000
    );
}

// ---------------------------------------------------------------------------
// A spilled column, and rare and common labels over a cold one
// ---------------------------------------------------------------------------

/// A vector column spills under pressure and a lookup of it reads through the map as a loaded one
/// does, with the rule on and off; other kinds of column are in `spill::tests` at the store.
#[cfg(all(
    feature = "vector-index",
    feature = "spill",
    feature = "mmap",
    feature = "wal"
))]
#[test]
#[expect(
    deprecated,
    reason = "reads the store's figures, and writes nothing through it"
)]
fn a_spilled_column_reads_as_a_loaded_one() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("spilled.grafeo");
    let vector = |seed: i64| Value::Vector(vec![(seed % 40) as f32, 19.0, 88.0].into());
    let db = GrafeoDB::open(&path).unwrap();
    for row in 0..GROUP + 2_000 {
        db.create_node_with_props(
            &["L"],
            [("row", Value::Int64(row)), ("embedding", vector(row))],
        )
        .unwrap();
    }
    db.create_vector_index("L", "embedding", None, None, None, None, None)
        .unwrap();
    db.buffer_manager().spill_all();
    let store = db.store().unwrap();
    assert!(
        !store.spilled_node_property_columns().is_empty(),
        "something to read from a map"
    );
    let usual = db.session();
    let off = session_with(
        &db,
        PlannerRules::all().with(PlannerRule::KeyColumnLookup, false),
    );
    let plain = session_with(&db, PlannerRules::plain());
    for key in [vector(7), vector(1_000), Value::Int64(7), Value::from("7")] {
        for query in [
            "MATCH (n:L {embedding: $v}) RETURN n.row",
            "MATCH (n {embedding: $v}) RETURN n.row LIMIT 5",
        ] {
            let wanted = rows_of(&plain, query, &key);
            assert_eq!(rows_of(&usual, query, &key), wanted, "{query} with {key:?}");
            assert_eq!(
                rows_of(&off, query, &key),
                wanted,
                "rule off: {query} with {key:?}"
            );
        }
    }
    db.close().unwrap();
}

/// Through every path: few rows of a rare label read one at a time, a common label compares in
/// place, off a word's boundary and sparse.
#[test]
fn a_rare_and_a_common_label_read_a_cold_column_alike() {
    for (name, value) in [
        (
            "integers",
            (|row| Value::Int64(row % 100)) as fn(i64) -> Value,
        ),
        ("strings", |row| {
            Value::from(format!("k{}", row % 100).as_str())
        }),
        ("mixed", |row| {
            if row % 2 == 0 {
                Value::Int64(row % 100)
            } else {
                Value::from(format!("{}", row % 100).as_str())
            }
        }),
    ] {
        let (_dir, db) = reopened(|db: &GrafeoDB| {
            for row in 0..5_000_i64 {
                let label = if row % 97 == 0 { "Rare" } else { "Common" };
                if row % 5 == 3 {
                    db.create_node_with_props(&[label], [("row", Value::Int64(row))])
                        .unwrap();
                } else {
                    db.create_node_with_props(
                        &[label],
                        [("row", Value::Int64(row)), ("k", value(row))],
                    )
                    .unwrap();
                }
            }
        });
        let usual = db.session();
        let plain = session_with(&db, PlannerRules::plain());
        for label in ["Rare", "Common"] {
            for key in [
                Value::Int64(0),
                Value::Int64(7),
                Value::from("7"),
                Value::from("k7"),
                Value::Int64(5_000),
            ] {
                let query = format!("MATCH (n:{label} {{k: $v}}) RETURN n.row");
                assert_eq!(
                    rows_of(&usual, &query, &key),
                    rows_of(&plain, &query, &key),
                    "{name}: {query} with {key:?}"
                );
            }
        }
        db.close().unwrap();
    }
}

// ---------------------------------------------------------------------------
// Strings that read as numbers, and a limit over each kind
// ---------------------------------------------------------------------------

/// Strings of every spelling a number may have. No string equals a number, whatever it spells (ISO/IEC
/// 39075 19.3, openCypher): a number key finds none of them, and a string key only its own text.
const SPELLINGS: &[&str] = &[
    "42",
    "042",
    "0042",
    "+42",
    "-42",
    "-042",
    "42.0",
    "42.00",
    "+42.0",
    "4.2e1",
    "4.2E1",
    "420e-1",
    " 42",
    "42 ",
    "4 2",
    "0x2A",
    "1_000",
    "42a",
    "a42",
    "--42",
    "+-42",
    "4.2.1",
    "e1",
    "1e",
    ".",
    "+",
    "-",
    "0",
    "00",
    "-0",
    "+0",
    "0.0",
    "-0.0",
    ".0",
    "0.",
    ".5",
    "5.",
    "1000",
    "1e3",
    "1E3",
    "nan",
    "NaN",
    "NAN",
    "+nan",
    "-nan",
    "inf",
    "Inf",
    "INF",
    "+inf",
    "-inf",
    "infinity",
    "Infinity",
    "-Infinity",
    "infin",
    "1e400",
    "-1e400",
    "1e-400",
    "9007199254740993",
    "9007199254740992",
    "9007199254740992.0",
    "9223372036854775807",
    "9223372036854775808",
    "true",
    "",
    " ",
    "abc",
    "é",
    "42é",
];

/// `=` of a string and a key: a string equals a string of the same text, and never a number.
fn equal_by_the_rule(text: &str, key: &Value) -> bool {
    match key {
        Value::Int64(_) | Value::Float64(_) => false,
        Value::String(key) => text == key.as_str(),
        other => panic!("{other:?}"),
    }
}

fn spelling_keys() -> Vec<Value> {
    let mut keys = vec![
        Value::Int64(42),
        Value::Int64(-42),
        Value::Int64(0),
        Value::Int64(1_000),
        Value::Int64(9_007_199_254_740_993),
        Value::Int64(9_007_199_254_740_992),
        Value::Int64(i64::MAX),
        Value::Float64(42.0),
        Value::Float64(-42.0),
        Value::Float64(0.0),
        Value::Float64(-0.0),
        Value::Float64(0.5),
        Value::Float64(5.0),
        Value::Float64(1_000.0),
        Value::Float64(f64::NAN),
        Value::Float64(f64::INFINITY),
        Value::Float64(f64::NEG_INFINITY),
        Value::Float64(9_007_199_254_740_992.0),
        Value::Float64(9_223_372_036_854_775_808.0),
    ];
    keys.extend(SPELLINGS.iter().map(|text| Value::from(*text)));
    keys
}

/// Each row's `row`, for the rows of a query that returns it, sorted.
fn found_rows(session: &Session, query: &str, key: &Value) -> Vec<i64> {
    let params = HashMap::from([("v".to_string(), key.clone())]);
    let result = session
        .execute_with_params(query, params)
        .unwrap_or_else(|error| panic!("{query} with {key:?}: {error}"));
    let mut rows: Vec<i64> = result
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::Int64(row) => *row,
            other => panic!("{other:?}"),
        })
        .collect();
    rows.sort_unstable();
    rows
}

fn string_column(db: &GrafeoDB) {
    for (row, text) in SPELLINGS.iter().enumerate() {
        db.create_node_with_props(&["L"], [("row", row_value(row)), ("k", Value::from(*text))])
            .unwrap();
    }
    db.create_node_with_props(&["L"], [("row", Value::Int64(-2))])
        .unwrap();
}

fn answers_by_the_rule(db: &GrafeoDB, stage: &str) {
    let usual = db.session();
    let plain = session_with(db, PlannerRules::plain());
    for key in spelling_keys() {
        let wanted: Vec<i64> = SPELLINGS
            .iter()
            .enumerate()
            .filter(|(_, text)| equal_by_the_rule(text, &key))
            .map(|(row, _)| i64::try_from(row).unwrap())
            .collect();
        for query in [
            "MATCH (n:L {k: $v}) RETURN n.row",
            "MATCH (n:L) WHERE n.k = $v RETURN n.row",
            "MATCH (n {k: $v}) RETURN n.row",
            "MATCH (n:L) WHERE n.k IN [$v] RETURN n.row",
        ] {
            assert_eq!(
                found_rows(&usual, query, &key),
                wanted,
                "{stage}: {query} with {key:?}"
            );
            assert_eq!(
                found_rows(&plain, query, &key),
                wanted,
                "{stage}, scan: {query} with {key:?}"
            );
        }
    }
}

/// A string column against a number key and a string key, in every spelling, on each way a column
/// is held: a number key finds no string, and `'042'` is not `'42'` as two strings.
#[test]
fn a_string_column_answers_as_equals_reads_it() {
    let db = GrafeoDB::new_in_memory();
    string_column(&db);
    answers_by_the_rule(&db, "hot");

    let (_dir, db) = reopened(string_column);
    answers_by_the_rule(&db, "reopened");
    db.create_node_with_props(&["Other"], [("k", Value::from("Gus"))])
        .unwrap();
    db.session()
        .execute("MATCH (n:L {row: -2}) SET n.other = 1")
        .unwrap();
    answers_by_the_rule(&db, "reopened, written beside");
    db.close().unwrap();

    let (_dir, db) = reopened(string_column);
    db.session()
        .execute("MATCH (n:L {row: -2}) SET n.k = 'Gus'")
        .unwrap();
    answers_by_the_rule(&db, "thawed");
    db.close().unwrap();
}

/// The cases spelled out, as the agreement above would hold of a rule written wrongly: a number key
/// finds no string however it is spelled, and a string key only its own text.
#[test]
fn no_string_equals_a_number() {
    let db = GrafeoDB::new_in_memory();
    string_column(&db);
    let session = db.session();
    let texts = |key: Value| -> Vec<&str> {
        found_rows(&session, "MATCH (n:L {k: $v}) RETURN n.row", &key)
            .into_iter()
            .map(|row| SPELLINGS[usize::try_from(row).unwrap()])
            .collect()
    };
    let none: Vec<&str> = Vec::new();
    for key in [
        Value::Int64(42),
        Value::Float64(42.0),
        Value::Int64(0),
        Value::Float64(-0.0),
        Value::Int64(1_000),
        Value::Float64(f64::NAN),
        Value::Float64(f64::INFINITY),
        Value::Int64(9_007_199_254_740_993),
    ] {
        assert_eq!(texts(key.clone()), none, "{key:?}");
    }
    assert_eq!(texts(Value::from("042")), ["042"]);
    assert_eq!(texts(Value::from("42")), ["42"]);
    assert_eq!(texts(Value::from("")), [""]);
    assert_eq!(texts(Value::from("nan")), ["nan"]);
    assert_eq!(texts(Value::from("inf")), ["inf"]);
}

/// A limit over each kind gives the rows the scan gives, in its order, across a chunk and a group,
/// when the first found row comes late, and with an index for part of the nodes.
#[test]
fn a_limit_over_each_kind_gives_the_scans_rows() {
    let kinds: [(&str, fn(i64) -> Value, Value, Value); 8] = [
        (
            "integers",
            |row| Value::Int64(row % 2),
            Value::Int64(1),
            Value::Int64(9),
        ),
        (
            "floats",
            |row| Value::Float64((row % 2) as f64),
            Value::Float64(1.0),
            Value::Int64(1),
        ),
        (
            "booleans",
            |row| Value::Bool(row % 2 == 1),
            Value::Bool(true),
            Value::Bool(true),
        ),
        (
            "few strings",
            |row| Value::from(if row % 2 == 0 { "Berlin" } else { "Paris" }),
            Value::from("Paris"),
            Value::from("Oslo"),
        ),
        (
            "unique strings",
            |row| Value::from(format!("k{row}").as_str()),
            Value::from("k70000"),
            Value::from("k5"),
        ),
        (
            "number strings",
            |row| Value::from(if row % 2 == 0 { "042" } else { "42.0" }),
            Value::Int64(42),
            Value::Float64(42.0),
        ),
        (
            "empty strings",
            |row| Value::from(if row % 2 == 0 { "" } else { "Gus" }),
            Value::from(""),
            Value::from("Gus"),
        ),
        (
            "two kinds",
            |row| {
                if row % 2 == 0 {
                    Value::Int64(1)
                } else {
                    Value::from("1")
                }
            },
            Value::Int64(1),
            Value::Float64(1.0),
        ),
    ];
    for (name, value, first, second) in kinds {
        let fill = |db: &GrafeoDB| {
            for row in 0..GROUP + 8_000 {
                let label = if row % 16 == 0 { "A" } else { "L" };
                db.create_node_with_props(
                    &[label],
                    [("row", Value::Int64(row)), ("k", value(row))],
                )
                .unwrap();
            }
        };
        let hot = GrafeoDB::new_in_memory();
        fill(&hot);
        let (_dir, cold) = reopened(fill);
        let indexed = GrafeoDB::new_in_memory();
        fill(&indexed);
        indexed
            .session()
            .execute("CREATE INDEX on_a FOR (n:A) ON (n.k)")
            .unwrap();
        for (stage, db) in [
            ("hot", &hot),
            ("reopened", &cold),
            ("indexed in part", &indexed),
        ] {
            let usual = db.session();
            let plain = session_with(db, PlannerRules::plain());
            for key in [&first, &second] {
                for query in [
                    "MATCH (n:L {k: $v}) RETURN n.row LIMIT 1",
                    "MATCH (n:L {k: $v}) RETURN n.row LIMIT 10",
                    "MATCH (n:L {k: $v}) RETURN n.row LIMIT 3000",
                    "MATCH (n:L {k: $v}) RETURN n.row LIMIT 40000",
                    "MATCH (n:L {k: $v}) RETURN n.row SKIP 5 LIMIT 10",
                    "MATCH (n {k: $v}) RETURN n.row LIMIT 10",
                    "MATCH (n:L) WHERE n.k IN [$v] RETURN n.row LIMIT 10",
                ] {
                    assert_eq!(
                        rows_of(&usual, query, key),
                        rows_of(&plain, query, key),
                        "{name}, {stage}: {query} with {key:?}"
                    );
                }
            }
        }
        cold.close().unwrap();
    }
}

// ---------------------------------------------------------------------------
// A string key, by `=`
// ---------------------------------------------------------------------------

/// `values` as the `k` of `:L` nodes numbered by `row`, on each way a column is held, and `check`
/// run on each: hot, reopened, and thawed by a write of the same kind.
fn on_each_stage(values: &[Value], check: impl Fn(&GrafeoDB, &str)) {
    let fill = |db: &GrafeoDB| {
        for (row, value) in values.iter().enumerate() {
            db.create_node_with_props(&["L"], [("row", row_value(row)), ("k", value.clone())])
                .unwrap();
        }
    };
    let db = GrafeoDB::new_in_memory();
    fill(&db);
    check(&db, "hot");
    let (_dir, db) = reopened(fill);
    check(&db, "reopened");
    db.close().unwrap();
    let (_dir, db) = reopened(fill);
    let session = db.session();
    session
        .execute_with_params(
            "MATCH (n:L {row: 0}) SET n.k = $v",
            HashMap::from([("v".to_string(), values[0].clone())]),
        )
        .unwrap();
    check(&db, "thawed");
    db.close().unwrap();
}

/// Each query form with `key` finds `wanted`, the rows `=` finds with the rule off.
fn finds_rows(db: &GrafeoDB, stage: &str, key: &Value, wanted: &[i64]) {
    let usual = db.session();
    let plain = session_with(db, PlannerRules::plain());
    for query in [
        "MATCH (n:L {k: $v}) RETURN n.row",
        "MATCH (n:L) WHERE n.k = $v RETURN n.row",
        "MATCH (n {k: $v}) RETURN n.row",
        "MATCH (n:L) WHERE n.k IN [$v] RETURN n.row",
    ] {
        assert_eq!(
            found_rows(&plain, query, key),
            wanted,
            "{stage}, rule off: {query} with {key:?}"
        );
        assert_eq!(
            found_rows(&usual, query, key),
            wanted,
            "{stage}: {query} with {key:?}"
        );
    }
}

#[test]
fn a_string_key_over_an_integer_column() {
    let values = [Value::Int64(42), Value::Int64(7)];
    on_each_stage(&values, |db, stage| {
        // No string equals a number.
        for text in ["42", "042", "+42", "4.2e1", "42.0", "abc"] {
            finds_rows(db, stage, &Value::from(text), &[]);
        }
        finds_rows(db, stage, &Value::Int64(42), &[0]);
        finds_rows(db, stage, &Value::Float64(42.0), &[0]);
    });
}

#[test]
fn a_string_key_over_a_float_column() {
    let values = [Value::Float64(42.0), Value::Float64(7.5)];
    on_each_stage(&values, |db, stage| {
        for text in ["42", "042", "4.2e1", "42.0", "7.5", "abc"] {
            finds_rows(db, stage, &Value::from(text), &[]);
        }
        finds_rows(db, stage, &Value::Int64(42), &[0]);
        finds_rows(db, stage, &Value::Float64(7.5), &[1]);
    });
}

#[test]
fn a_key_over_a_column_of_strings_that_read_as_numbers() {
    let values = [Value::from("042"), Value::from("42"), Value::from("Gus")];
    on_each_stage(&values, |db, stage| {
        finds_rows(db, stage, &Value::from("42"), &[1]);
        finds_rows(db, stage, &Value::from("042"), &[0]);
        finds_rows(db, stage, &Value::Int64(42), &[]);
        finds_rows(db, stage, &Value::Float64(42.0), &[]);
        finds_rows(db, stage, &Value::from("Gus"), &[2]);
    });
}

/// A group of any values: a string key finds its own text, a number key the numbers, and neither
/// the other kind.
#[test]
fn a_key_over_a_column_of_any_values() {
    let values = [
        Value::from("042"),
        Value::from("42"),
        Value::Int64(42),
        Value::Bool(true),
    ];
    on_each_stage(&values, |db, stage| {
        finds_rows(db, stage, &Value::from("042"), &[0]);
        finds_rows(db, stage, &Value::from("42"), &[1]);
        finds_rows(db, stage, &Value::from("4.2e1"), &[]);
        finds_rows(db, stage, &Value::Int64(42), &[2]);
        finds_rows(db, stage, &Value::Float64(42.0), &[2]);
        finds_rows(db, stage, &Value::Bool(true), &[3]);
        finds_rows(db, stage, &Value::from("true"), &[]);
        finds_rows(db, stage, &Value::Int64(1), &[]);
    });
}

/// A list of keys of both kinds answers for each, with nothing found twice.
#[test]
fn a_list_of_string_and_number_keys() {
    let values = [
        Value::from("042"),
        Value::from("42"),
        Value::Int64(42),
        Value::Float64(7.0),
        Value::from("7"),
    ];
    on_each_stage(&values, |db, stage| {
        let usual = db.session();
        let plain = session_with(db, PlannerRules::plain());
        for (query, wanted) in [
            (
                "MATCH (n:L) WHERE n.k IN ['042', '42'] RETURN n.row",
                &[0, 1][..],
            ),
            (
                "MATCH (n:L) WHERE n.k IN ['042', 7] RETURN n.row",
                &[0, 3][..],
            ),
            (
                "MATCH (n:L) WHERE n.k IN [42, 42.0, '42'] RETURN n.row",
                &[1, 2][..],
            ),
            ("MATCH (n:L) WHERE n.k IN ['nobody'] RETURN n.row", &[][..]),
        ] {
            assert_eq!(
                found_rows(&plain, query, &Value::Null),
                wanted,
                "{stage}, rule off: {query}"
            );
            assert_eq!(
                found_rows(&usual, query, &Value::Null),
                wanted,
                "{stage}: {query}"
            );
        }
    });
}

// ---------------------------------------------------------------------------
// Slices of a read
// ---------------------------------------------------------------------------

/// Rows of a column read by slices, any kinds, over three slices of a group.
const SLICED: i64 = 7_000;

/// Each kind holds every digit, so `k: 3` finds integers and floats and `k: '3'` the strings (GA04).
fn sliced_value(row: i64) -> Value {
    let digit = (row / 4) % 10;
    match row % 4 {
        0 => Value::Int64(digit),
        1 => Value::from(format!("{digit}").as_str()),
        2 => Value::Float64(digit as f64),
        _ => Value::Bool(row % 8 == 3),
    }
}

fn sliced(db: &GrafeoDB) {
    for row in 0..SLICED {
        db.create_node_with_props(
            &["L"],
            [("row", Value::Int64(row)), ("k", sliced_value(row))],
        )
        .unwrap();
    }
}

/// A subquery's read runs from the start for each row, cut short by its limit or not: nothing of the
/// chunk it decoded or the slice it reached carries over.
#[test]
fn a_read_cut_short_starts_again_for_the_next_row() {
    let check = |db: &GrafeoDB, stage: &str| {
        let usual = db.session();
        let plain = session_with(db, PlannerRules::plain());
        for query in [
            "UNWIND [1, 2, 3] AS x CALL { MATCH (n:L {k: 3}) RETURN n.row AS r LIMIT 3 } RETURN x, r",
            "UNWIND [1, 2, 3] AS x CALL { MATCH (n:L {k: 3}) RETURN n.row AS r LIMIT 3000 } RETURN x, r",
            "UNWIND [1, 2] AS x CALL { MATCH (n:L {k: 3}) RETURN n.row AS r SKIP 2000 LIMIT 3 } RETURN x, r",
            "UNWIND [3, 5, 3] AS x OPTIONAL MATCH (n:L {k: 3}) RETURN x, count(n)",
        ] {
            assert_eq!(
                usual
                    .execute(query)
                    .unwrap_or_else(|error| panic!("{query}: {error}"))
                    .rows(),
                plain.execute(query).unwrap().rows(),
                "{stage}: {query}"
            );
        }
    };
    let db = GrafeoDB::new_in_memory();
    sliced(&db);
    check(&db, "hot");
    let (_dir, db) = reopened(sliced);
    check(&db, "reopened");
    db.close().unwrap();
}

/// A row with a chain in a later slice is read as the view sees it, once, by an old reader and a
/// new one, in a column read by slices.
#[test]
fn a_chained_row_in_a_later_slice_is_read_at_the_view() {
    let check = |db: &GrafeoDB, stage: &str| {
        let on = |key: &str| format!("MATCH (n:L {{k: {key}}}) RETURN n.row ORDER BY n.row");
        let mut reader = db.session();
        reader.begin_transaction().unwrap();
        let before: Vec<_> = [on("3"), on("99")]
            .iter()
            .map(|query| reader.execute(query).unwrap().rows().to_vec())
            .collect();
        assert!(!before[0].is_empty(), "{stage}: k: 3 finds rows");
        // Rows in the third slice: one leaves `k: 3`, one comes to `k: 99`, and one comes in.
        db.session()
            .execute("MATCH (n:L {row: 4812}) SET n.k = 99")
            .unwrap();
        db.session()
            .execute("MATCH (n:L {row: 4803}) SET n.k = 99")
            .unwrap();
        db.session()
            .execute("MATCH (n:L {row: 5001}) SET n.k = 3")
            .unwrap();
        let after: Vec<_> = [on("3"), on("99")]
            .iter()
            .map(|query| reader.execute(query).unwrap().rows().to_vec())
            .collect();
        assert_eq!(after, before, "{stage}: the old reader's rows");
        reader.rollback().unwrap();
        let plain = session_with(db, PlannerRules::plain());
        for query in [on("3"), on("99"), on("'3'"), on("3.0")] {
            assert_eq!(
                reader.execute(&query).unwrap().rows(),
                plain.execute(&query).unwrap().rows(),
                "{stage}: {query}"
            );
        }
        let new_rows = reader.execute(&on("99")).unwrap();
        assert_eq!(new_rows.row_count(), 2, "{stage}");
    };
    let db = GrafeoDB::new_in_memory();
    sliced(&db);
    check(&db, "hot");
    let (_dir, db) = reopened(sliced);
    check(&db, "reopened");
    db.close().unwrap();
}

/// Only the rows of a slice are read for a limit a slice meets, in a column of any values or strings.
#[test]
fn a_limit_reads_a_slice_of_a_column_it_compares_by_value() {
    let db = GrafeoDB::new_in_memory();
    sliced(&db);
    let usual = db.session();
    let limited = usual
        .execute("MATCH (n:L {k: 3}) RETURN n.row LIMIT 1")
        .unwrap();
    assert_eq!(limited.row_count(), 1);
    let read = limited.rows_scanned().unwrap();
    assert!(read > 0 && read <= 2_048, "a slice, read {read}");
    let whole = usual.execute("MATCH (n:L {k: 3}) RETURN n.row").unwrap();
    assert_eq!(whole.rows_scanned(), Some(SLICED as u64), "each row once");
    let (_dir, db) = reopened(sliced);
    let session = db.session();
    let limited = session
        .execute("MATCH (n:L {k: 3}) RETURN n.row LIMIT 1")
        .unwrap();
    let read = limited.rows_scanned().unwrap();
    assert!(read > 0 && read <= 2_048, "reopened, a slice, read {read}");
    let whole = session.execute("MATCH (n:L {k: 3}) RETURN n.row").unwrap();
    assert_eq!(
        whole.rows_scanned(),
        Some(SLICED as u64),
        "reopened, each row once"
    );
    db.close().unwrap();
}

// ---------------------------------------------------------------------------
// A number key over strings, and dictionary chunks
// ---------------------------------------------------------------------------

#[test]
fn a_number_key_finds_no_spelling_of_a_string() {
    let values = [
        Value::from("42"),
        Value::from("42.0"),
        Value::from("4.2e1"),
        Value::from("+42"),
    ];
    on_each_stage(&values, |db, stage| {
        // A number key finds no string, however it spells the number.
        finds_rows(db, stage, &Value::Int64(42), &[]);
        finds_rows(db, stage, &Value::Float64(42.0), &[]);
        finds_rows(db, stage, &Value::Float64(42.5), &[]);
        finds_rows(db, stage, &Value::Int64(43), &[]);
        finds_rows(db, stage, &Value::from("42.0"), &[1]);
    });
}

/// Past what a float holds as well: a number key finds no string, the string key its own text.
#[test]
fn a_number_key_past_a_floats_reach_finds_no_string() {
    let values = [
        Value::from("9007199254740993"),
        Value::from("9007199254740992"),
    ];
    on_each_stage(&values, |db, stage| {
        finds_rows(db, stage, &Value::Int64(9_007_199_254_740_993), &[]);
        finds_rows(db, stage, &Value::Int64(9_007_199_254_740_992), &[]);
        finds_rows(db, stage, &Value::Float64(9_007_199_254_740_992.0), &[]);
        finds_rows(db, stage, &Value::from("9007199254740993"), &[0]);
    });
}

/// Few strings, in chunks of a dictionary each: all with the rule off's answer, by a key in the
/// first chunk only, in a later one only, in both, and in neither.
fn by_group(db: &GrafeoDB) {
    for row in 0..GROUP + 5_000 {
        let city = match (row < GROUP, row % 3) {
            (true, 0) => "Amsterdam",
            (true, 1) => "Berlin",
            (true, _) => "42",
            (false, 0) => "Prague",
            (false, 1) => "Berlin",
            (false, _) => "042",
        };
        db.create_node_with_props(
            &["L"],
            [("row", Value::Int64(row)), ("k", Value::from(city))],
        )
        .unwrap();
    }
}

#[test]
fn a_key_in_a_later_dictionary_chunk_is_found() {
    let check = |db: &GrafeoDB, stage: &str| {
        let usual = db.session();
        let plain = session_with(db, PlannerRules::plain());
        for key in [
            Value::from("Prague"),
            Value::from("Amsterdam"),
            Value::from("Berlin"),
            Value::from("Oslo"),
            Value::from("042"),
            Value::Int64(42),
            Value::Float64(42.0),
        ] {
            for query in [
                "MATCH (n:L {k: $v}) RETURN n.row LIMIT 1",
                "MATCH (n:L {k: $v}) RETURN n.row LIMIT 10",
                "MATCH (n:L {k: $v}) RETURN n.row LIMIT 3000",
                "MATCH (n:L {k: $v}) RETURN n.row SKIP 2500 LIMIT 3",
                "MATCH (n:L {k: $v}) RETURN count(n)",
            ] {
                assert_eq!(
                    rows_of(&usual, query, &key),
                    rows_of(&plain, query, &key),
                    "{stage}: {query} with {key:?}"
                );
            }
        }
    };
    let db = GrafeoDB::new_in_memory();
    by_group(&db);
    check(&db, "hot");
    let (_dir, db) = reopened(by_group);
    check(&db, "reopened");
    let session = db.session();
    for key in [Value::Int64(42), Value::Float64(42.0)] {
        assert_eq!(
            rows_of(&session, "MATCH (n:L {k: $v}) RETURN count(n)", &key),
            ["[Int64(0)]"],
            "a number key over a dictionary of text: {key:?}"
        );
    }
    assert_eq!(
        rows_of(
            &session,
            "MATCH (n:L {k: $v}) RETURN count(n)",
            &Value::from("042")
        ),
        ["[Int64(1667)]"],
        "the text alone"
    );
    drop(session);
    db.close().unwrap();
}

/// A read part way through a dictionary chunk, by slices, starts over for the next row of a subquery.
#[test]
fn a_read_cut_short_in_a_dictionary_chunk_starts_again() {
    let (_dir, db) = reopened(by_group);
    let usual = db.session();
    let plain = session_with(&db, PlannerRules::plain());
    for query in [
        "UNWIND [1, 2, 3] AS x CALL { MATCH (n:L {k: 'Berlin'}) RETURN n.row AS r LIMIT 3 } RETURN x, r",
        "UNWIND [1, 2] AS x CALL { MATCH (n:L {k: 'Berlin'}) RETURN n.row AS r SKIP 1500 LIMIT 3 } RETURN x, r",
        "UNWIND [1, 2] AS x CALL { MATCH (n:L {k: 42}) RETURN n.row AS r SKIP 1500 LIMIT 3 } RETURN x, r",
        "UNWIND [1, 2] AS x CALL { MATCH (n:L {k: 'Prague'}) RETURN n.row AS r LIMIT 2 } RETURN x, r",
    ] {
        assert_eq!(
            usual
                .execute(query)
                .unwrap_or_else(|error| panic!("{query}: {error}"))
                .rows(),
            plain.execute(query).unwrap().rows(),
            "{query}"
        );
    }
    db.close().unwrap();
}

/// A row of each kind of scope, with `k: 5` until a writer moves it, among rows of `k` from 1,000 up,
/// in two groups: `moved` is read through its chain.
fn chained_scopes(db: &GrafeoDB) {
    let row = |row: i64| ("row", Value::Int64(row));
    for at in 0..GROUP + 200 {
        let key = if matches!(at, 7 | 9) || at == GROUP + 7 || at == GROUP + 9 {
            5
        } else {
            1_000 + at % 1_000
        };
        let labels: &[&str] = match at {
            9 => &[],
            _ if at == GROUP + 9 => &["M"],
            _ => &["L"],
        };
        db.create_node_with_props(labels, [row(at), ("k", Value::Int64(key))])
            .unwrap();
    }
}

const CHAINED_FORMS: &[&str] = &[
    "MATCH (n:L {k: @}) RETURN n.row ORDER BY n.row",
    "MATCH (n {k: @}) RETURN n.row ORDER BY n.row",
    "MATCH (n:L|M {k: @}) RETURN n.row ORDER BY n.row",
    "MATCH (n:M|Nobody {k: @}) RETURN n.row ORDER BY n.row",
    "MATCH (n) WHERE (n:L OR n:M) AND n.k = @ RETURN n.row ORDER BY n.row",
    "MATCH (n:L) WHERE n.k IN [@, 6] RETURN n.row ORDER BY n.row",
];

/// A key that only a chain holds now: an old reader finds the rows, with and without labels, and a
/// new one does not, against the plain plan at both views. A key past every value finds nothing at
/// either, though the chained rows are read for it.
#[test]
fn a_key_only_a_chain_holds_is_read_at_the_view() {
    let check = |db: &GrafeoDB, stage: &str| {
        let plain = |view: &mut Session| {
            view.begin_transaction().unwrap();
        };
        let mut reader = db.session();
        let mut plain_reader = session_with(db, PlannerRules::plain());
        plain(&mut reader);
        plain(&mut plain_reader);
        for moved in [7, 9, GROUP + 7, GROUP + 9] {
            db.session()
                .execute(&format!("MATCH (n {{row: {moved}}}) SET n.k = 1500"))
                .unwrap();
        }
        let plain_new = session_with(db, PlannerRules::plain());
        for form in CHAINED_FORMS {
            for key in ["5", "5.0e0"] {
                let query = form.replace('@', key);
                let old = reader.execute(&query).unwrap();
                assert_eq!(
                    old.rows(),
                    plain_reader.execute(&query).unwrap().rows(),
                    "{stage}: the old reader, {query}"
                );
                assert!(
                    old.row_count() > 0,
                    "{stage}: the old reader finds it: {query}"
                );
                let new = db.session().execute(&query).unwrap();
                assert_eq!(
                    new.row_count(),
                    0,
                    "{stage}: only a chain holds it: {query}"
                );
                assert_eq!(plain_new.execute(&query).unwrap().row_count(), 0, "{stage}");
            }
            // A string equals no number, whatever it spells (GA04).
            for key in ["0", "5000000", "'nobody'", "'5'", "'05'"] {
                let query = form.replace('@', key);
                assert_eq!(
                    reader.execute(&query).unwrap().row_count(),
                    0,
                    "{stage}: {query}"
                );
                assert_eq!(
                    db.session().execute(&query).unwrap().row_count(),
                    0,
                    "{stage}: {query}"
                );
            }
        }
        reader.rollback().unwrap();
        plain_reader.rollback().unwrap();
    };
    let db = GrafeoDB::new_in_memory();
    chained_scopes(&db);
    check(&db, "hot");
    let (_dir, db) = reopened(chained_scopes);
    check(&db, "reopened");
    db.close().unwrap();
}

/// One label's rows, and a row far past them that loses a label while a reader lives: under a union
/// and under a required label, where the group's other rows no longer reach that word, and in a
/// group where no row has the label any more.
fn label_chains(db: &GrafeoDB) {
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
            // The only row of `A` and `B` in the second group.
            _ if at == GROUP + 2 => node(&["A"], at, 5),
            _ if at == GROUP + 3 => node(&["B"], at, 6),
            _ => node(&["Other"], at, 6),
        }
    }
}

/// A row whose newest labels lost a label its chain keeps is read for a reader of that view, and not
/// after, in every form of scope, hot and with the key's column cold.
#[test]
fn a_row_that_lost_a_label_its_chain_keeps_is_read_under_a_union() {
    let forms: &[&str] = &[
        "MATCH (n:A {k: 5}) RETURN n.row ORDER BY n.row",
        "MATCH (n:A|Nobody {k: 5}) RETURN n.row ORDER BY n.row",
        "MATCH (n:A|B {k: 5}) RETURN n.row ORDER BY n.row",
        "MATCH (n:B|A {k: 5}) RETURN n.row ORDER BY n.row",
        "MATCH (n) WHERE (n:A OR n:B) AND n.k = 5 RETURN n.row ORDER BY n.row",
        "MATCH (n:A) WHERE n.k IN [5, 7] RETURN n.row ORDER BY n.row",
        "MATCH (n:A|B) WHERE n.k IN [5, 7] RETURN n.row ORDER BY n.row",
    ];
    let rows = |session: &Session, query: &str| -> Vec<i64> {
        session
            .execute(query)
            .unwrap()
            .rows()
            .iter()
            .map(|row| match row[0] {
                Value::Int64(at) => at,
                ref other => panic!("{other:?}"),
            })
            .collect()
    };
    let check = |db: &GrafeoDB, stage: &str| {
        let mut reader = db.session();
        let mut plain_reader = session_with(db, PlannerRules::plain());
        reader.begin_transaction().unwrap();
        plain_reader.begin_transaction().unwrap();
        let before: Vec<Vec<i64>> = forms.iter().map(|query| rows(&reader, query)).collect();
        assert_eq!(before[0], [3, 5, 40_000, GROUP + 2], "{stage}");
        assert_eq!(before[2], [3, 4, 5, 40_000, GROUP + 2], "{stage}");
        for query in [
            "MATCH (n {row: 3}) REMOVE n:A",
            "MATCH (n {row: 40000}) REMOVE n:A",
        ] {
            db.session().execute(query).unwrap();
        }
        db.session()
            .execute(&format!("MATCH (n {{row: {}}}) REMOVE n:A", GROUP + 2))
            .unwrap();
        db.session()
            .execute(&format!(
                "MATCH (n {{row: {}}}) SET n:A, n.k = 5",
                GROUP + 5
            ))
            .unwrap();
        let plain_new = session_with(db, PlannerRules::plain());
        for (query, before) in forms.iter().zip(&before) {
            assert_eq!(
                &rows(&reader, query),
                before,
                "{stage}: the old reader, {query}"
            );
            assert_eq!(
                &rows(&plain_reader, query),
                before,
                "{stage}: the plain plan, {query}"
            );
            let new = rows(&db.session(), query);
            assert_eq!(
                new,
                rows(&plain_new, query),
                "{stage}: the new reader, {query}"
            );
            assert!(
                !new.contains(&40_000) && new.contains(&(GROUP + 5)),
                "{stage}: {query}: {new:?}"
            );
        }
        reader.rollback().unwrap();
        plain_reader.rollback().unwrap();
    };
    let db = GrafeoDB::new_in_memory();
    label_chains(&db);
    check(&db, "hot");
    let (_dir, db) = reopened(label_chains);
    check(&db, "reopened");
    db.close().unwrap();
}

/// A list of `len` distinct keys held by the first rows, as GQL.
fn list_of(len: usize) -> String {
    let keys: Vec<String> = (0..len).map(|key| key.to_string()).collect();
    format!("[{}]", keys.join(", "))
}

/// Whether the plan reads the key's column, by `EXPLAIN` and by `PROFILE`, which must agree.
fn reads_the_column(session: &Session, query: &str) -> bool {
    let explained = plan(session, &format!("EXPLAIN {query}"));
    let profiled = session.execute(&format!("PROFILE {query}")).unwrap();
    let profiled = format!("{:?}", profiled.rows());
    let by_explain = explained.contains("key column");
    let by_profile = profiled.contains("KeyColumnScan");
    assert_eq!(
        by_explain, by_profile,
        "{query}\nEXPLAIN {explained}\nPROFILE {profiled}"
    );
    by_explain
}

/// Below a table's size a list reads the column, and from its size up it keeps the plain filter, as
/// the rarest required label, a union's sum and all nodes count it. `EXPLAIN` and `PROFILE` say the
/// same on each side, and each answer is the plain plan's.
#[test]
fn explain_and_profile_agree_around_a_long_list() {
    let db = GrafeoDB::new_in_memory();
    let node = |labels: &[&str], key: i64| {
        db.create_node_with_props(labels, [("k", Value::Int64(key))])
            .unwrap();
    };
    // Few: 5 rows. Pair: 3 more. Both: 2 of Few with Many too. Many: 50 in all. 60 nodes.
    for key in 0..3 {
        node(&["Few"], key);
    }
    for key in 3..5 {
        node(&["Few", "Many"], key);
    }
    for key in 5..8 {
        node(&["Pair"], key);
    }
    for key in 8..56 {
        node(&["Many"], key);
    }
    for key in 56..60 {
        node(&["Other"], key);
    }
    let session = db.session();
    let plain = session_with(&db, PlannerRules::plain());
    // Pattern, size of what a read looks at.
    let scopes: &[(&str, usize)] = &[
        ("MATCH (n:Few) WHERE n.k IN @ RETURN n.k ORDER BY n.k", 5),
        ("MATCH (n:Pair) WHERE n.k IN @ RETURN n.k ORDER BY n.k", 3),
        (
            "MATCH (n:Few&Many) WHERE n.k IN @ RETURN n.k ORDER BY n.k",
            5,
        ),
        (
            "MATCH (n:Many&Few) WHERE n.k IN @ RETURN n.k ORDER BY n.k",
            5,
        ),
        (
            "MATCH (n:Few|Pair) WHERE n.k IN @ RETURN n.k ORDER BY n.k",
            8,
        ),
        ("MATCH (n) WHERE n.k IN @ RETURN n.k ORDER BY n.k", 60),
    ];
    for (form, size) in scopes {
        for len in [1, size / 2, size - 1, *size, size + 1, size * 3] {
            if len == 0 {
                continue;
            }
            let query = form.replace('@', &list_of(len));
            assert_eq!(
                reads_the_column(&session, &query),
                len < *size,
                "{query}: a list of {len} over {size}"
            );
            assert_eq!(
                session.execute(&query).unwrap().rows(),
                plain.execute(&query).unwrap().rows(),
                "{query}"
            );
        }
    }

    // A list's length is what is written: a repeated key counts.
    let repeated = "MATCH (n:Pair) WHERE n.k IN [5, 5, 5] RETURN n.k";
    assert!(!reads_the_column(&session, repeated));
    assert_eq!(session.execute(repeated).unwrap().row_count(), 1);

    // A label no node has: nothing to read, either way.
    let nobody = "MATCH (n:Nobody) WHERE n.k IN [1] RETURN n.k";
    reads_the_column(&session, nobody);
    assert_eq!(session.execute(nobody).unwrap().row_count(), 0);

    // An equality is no list, and an index goes first whatever the list's length.
    assert!(reads_the_column(
        &session,
        "MATCH (n:Pair {k: 5}) RETURN n.k"
    ));
    let indexed = GrafeoDB::new_in_memory();
    for key in 0..3 {
        indexed
            .create_node_with_props(&["Few"], [("k", Value::Int64(key))])
            .unwrap();
    }
    let on_index = indexed.session();
    on_index
        .execute("CREATE INDEX few_k FOR (n:Few) ON (n.k)")
        .unwrap();
    let long = format!("MATCH (n:Few) WHERE n.k IN {} RETURN n.k", list_of(9));
    let shown = plan(&on_index, &long);
    assert!(
        shown.contains("[index: k]"),
        "the index, not the column: {shown}"
    );
    assert_eq!(on_index.execute(&long).unwrap().row_count(), 3);
}

/// A statement's own rows and a list over a table that grew or shrank: the count picks a route only,
/// so a reader's answer holds whichever side of the boundary the latest count is on.
#[test]
fn the_count_that_picks_the_route_does_not_change_an_answer() {
    let db = GrafeoDB::new_in_memory();
    for key in 0..4 {
        db.create_node_with_props(&["Few"], [("k", Value::Int64(key))])
            .unwrap();
    }
    let query = "MATCH (n:Few) WHERE n.k IN [0, 1, 2, 3] RETURN n.k ORDER BY n.k";
    let mut reader = db.session();
    reader.begin_transaction().unwrap();
    let before = reader.execute(query).unwrap().rows().to_vec();
    assert_eq!(before.len(), 4);
    // Growing the table moves the latest count over the list's length.
    let writer = db.session();
    for key in 0..4 {
        writer
            .execute(&format!("INSERT (:Few {{k: {key}}})"))
            .unwrap();
    }
    assert_eq!(
        reader.execute(query).unwrap().rows(),
        before,
        "the old view"
    );
    assert_eq!(
        writer.execute(query).unwrap().row_count(),
        8,
        "the new view"
    );
    // Shrinking it moves the count back under, for the old reader too.
    writer.execute("MATCH (n:Few) DETACH DELETE n").unwrap();
    assert_eq!(
        reader.execute(query).unwrap().rows(),
        before,
        "the old view"
    );
    assert_eq!(
        writer.execute(query).unwrap().row_count(),
        0,
        "the new view"
    );
    reader.rollback().unwrap();

    // Rows the transaction made itself are read on both sides.
    let mut own = db.session();
    own.begin_transaction().unwrap();
    own.execute("INSERT (:Few {k: 0}), (:Few {k: 1})").unwrap();
    assert_eq!(own.execute(query).unwrap().row_count(), 2);
    own.rollback().unwrap();
    assert_eq!(db.session().execute(query).unwrap().row_count(), 0);
}
