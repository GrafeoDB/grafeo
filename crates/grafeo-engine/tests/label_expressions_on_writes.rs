//! `INSERT` and single-node `MERGE` with a label expression (ISO/IEC 39075:2024 13.2 and 18).
//!
//! `IS A` and `A&B` set exactly those labels; `|`, `!`, `%` and parentheses are an error that names
//! the statement. They used to create a node with no label, silently. `MERGE` matches on the labels,
//! not on the property alone, and a `MERGE` of a path takes labels only.
//! `16.8d_label_expressions_on_writes.gtest` runs the same cases on every binding.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main label_expressions_on_writes::
//! ```

use grafeo_common::types::Value;
use grafeo_engine::session::Session;
use grafeo_engine::{Config, GrafeoDB};

/// Every node as its `k` and sorted labels, sorted.
fn nodes(session: &Session) -> Vec<String> {
    let result = session.execute("MATCH (x) RETURN x.k, labels(x)").unwrap();
    let mut rows: Vec<String> = result
        .rows()
        .iter()
        .map(|row| {
            let mut labels: Vec<String> = match &row[1] {
                Value::List(items) => items
                    .iter()
                    .map(|item| item.as_str().unwrap().to_string())
                    .collect(),
                other => panic!("labels: {other:?}"),
            };
            labels.sort();
            let k = row[0]
                .as_int64()
                .map_or("null".to_string(), |k| k.to_string());
            format!("{k}:{}", labels.join(","))
        })
        .collect();
    rows.sort();
    rows
}

fn fresh() -> (GrafeoDB, Session) {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    (db, session)
}

fn run_all(session: &Session, statements: &[&str]) {
    for statement in statements {
        session
            .execute(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }
}

// ---------------------------------------------------------------------------
// INSERT sets the labels
// ---------------------------------------------------------------------------

#[test]
fn insert_sets_exactly_the_labels() {
    for (statement, wanted) in [
        ("INSERT (n IS A {k: 1})", vec!["1:A"]),
        ("INSERT (n:A&B {k: 1})", vec!["1:A,B"]),
        ("INSERT (n IS A&B {k: 1})", vec!["1:A,B"]),
        ("INSERT (n:A&B&C {k: 1})", vec!["1:A,B,C"]),
        ("INSERT (n:A&A {k: 1})", vec!["1:A"]),
        ("INSERT (:A&B {k: 1})", vec!["1:A,B"]),
        ("INSERT (IS A {k: 1})", vec!["1:A"]),
        ("INSERT (n:A {k: 1})", vec!["1:A"]),
        ("INSERT (n:A:B {k: 1})", vec!["1:A,B"]),
        (
            "INSERT (n IS A {k: 1}), (m:B&C {k: 2})",
            vec!["1:A", "2:B,C"],
        ),
        (
            "UNWIND [1, 2] AS i INSERT (n IS A {k: i})",
            vec!["1:A", "2:A"],
        ),
    ] {
        let (_db, session) = fresh();
        session
            .execute(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
        assert_eq!(nodes(&session), wanted, "{statement}");
    }
}

/// Every node of a path takes its expression, and the path's edge stands between them.
#[test]
fn insert_sets_the_labels_on_every_node_of_a_path() {
    let (_db, session) = fresh();
    session
        .execute("INSERT (n IS A {k: 1})-[:R]->(m:A&B {k: 2})<-[:S]-(o IS C&D {k: 3})")
        .unwrap();
    assert_eq!(nodes(&session), vec!["1:A", "2:A,B", "3:C,D"]);
    let read = session
        .execute("MATCH (x:A)-[:R]->(y:A:B)<-[:S]-(z:C:D) RETURN x.k, y.k, z.k")
        .unwrap();
    assert_eq!(
        read.rows(),
        vec![vec![Value::Int64(1), Value::Int64(2), Value::Int64(3)]]
    );
}

#[test]
fn insert_after_a_match_sets_the_labels() {
    let (_db, session) = fresh();
    run_all(
        &session,
        &[
            "INSERT (:T {k: 0})",
            "MATCH (t:T) INSERT (t)-[:R]->(n:A&B {k: 1})",
            "MATCH (t:T) INSERT (n IS C {k: 2})-[:R]->(t)",
        ],
    );
    assert_eq!(nodes(&session), vec!["0:T", "1:A,B", "2:C"]);
}

#[test]
fn insert_returns_the_labels_it_set() {
    let (_db, session) = fresh();
    let result = session
        .execute("INSERT (n:A&B {k: 1}) RETURN n.k, size(labels(n))")
        .unwrap();
    assert_eq!(result.rows(), vec![vec![Value::Int64(1), Value::Int64(2)]]);
}

/// A node so labelled is what the label finds, through a scan and an index, and a rule on the label
/// holds for it.
#[test]
fn inserted_labels_reach_the_index_and_the_rules() {
    let (_db, session) = fresh();
    run_all(
        &session,
        &[
            "CREATE INDEX by_a FOR (n:A) ON (n.k)",
            "CREATE CONSTRAINT one_a FOR (n:B) ON (n.k) UNIQUE",
            "INSERT (n IS A {k: 1})",
            "INSERT (n:A&B {k: 2})",
        ],
    );
    let found = |query: &str| session.execute(query).unwrap().row_count();
    assert_eq!(found("MATCH (n:A {k: 1}) RETURN n"), 1);
    assert_eq!(found("MATCH (n:A {k: 2}) RETURN n"), 1);
    assert_eq!(found("MATCH (n:B {k: 2}) RETURN n"), 1);
    assert!(
        session.execute("INSERT (n:B&C {k: 2})").is_err(),
        "the rule on :B holds for a node labelled through an expression"
    );
    assert_eq!(nodes(&session), vec!["1:A", "2:A,B"]);
}

/// The labels survive a reopen, by the log and by a checkpoint.
#[test]
fn inserted_labels_survive_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("labels.grafeo");
    {
        let db = GrafeoDB::open(&path).unwrap();
        run_all(
            &db.session(),
            &["INSERT (n IS A {k: 1})", "MERGE (m:B&C {k: 2})"],
        );
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(nodes(&db.session()), vec!["1:A", "2:B,C"]);
    db.close().unwrap();
}

#[test]
fn a_rollback_takes_back_the_node_and_its_labels() {
    let (_db, mut session) = fresh();
    session.begin_transaction().unwrap();
    run_all(
        &session,
        &["INSERT (n IS A {k: 1})", "MERGE (m:B&C {k: 2})"],
    );
    assert_eq!(nodes(&session), vec!["1:A", "2:B,C"]);
    session.rollback().unwrap();
    assert_eq!(nodes(&session), Vec::<String>::new());
}

// ---------------------------------------------------------------------------
// The forms a statement cannot set
// ---------------------------------------------------------------------------

/// What no node could be given as labels: `|`, `!`, `%` and parentheses.
const REFUSED: [&str; 11] = [
    ":A|B", " IS A|B", ":!A", " IS !A", ":%", " IS %", ":(A)", " IS (A)", ":(A&B)", ":A&!B",
    ":A&(B|C)",
];

/// The statement's error names it, and nothing is written, also when another part of it could be.
#[test]
fn insert_refuses_what_no_node_could_be_given() {
    for form in REFUSED {
        let bare = form.trim();
        for statement in [
            format!("INSERT (n{form} {{k: 1}})"),
            format!("INSERT ({bare} {{k: 1}})"),
            format!("INSERT (n{form})-[:R]->(m:T)"),
            format!("INSERT (n:T)-[:R]->(m{form})"),
            format!("INSERT (o:T {{k: 2}}), (n{form} {{k: 1}})"),
            format!("UNWIND [1, 2] AS i INSERT (n{form} {{k: i}})"),
        ] {
            let (_db, session) = fresh();
            let error = session
                .execute(&statement)
                .expect_err(&statement)
                .to_string();
            assert!(error.contains("INSERT"), "{statement}: {error}");
            assert_eq!(nodes(&session), Vec::<String>::new(), "{statement}");
        }
    }
}

#[test]
fn merge_refuses_what_no_node_could_be_given() {
    for form in REFUSED {
        let statement = format!("MERGE (n{form} {{k: 1}})");
        let (_db, session) = fresh();
        session.execute("INSERT (:A {k: 1})").unwrap();
        let error = session
            .execute(&statement)
            .expect_err(&statement)
            .to_string();
        assert!(error.contains("MERGE"), "{statement}: {error}");
        assert_eq!(nodes(&session), vec!["1:A"], "{statement}");
    }
}

/// A variable the match bound is the node that exists: an expression on it would add labels, and is
/// refused as a label on it is, and nothing is written.
#[test]
fn insert_refuses_an_expression_on_a_bound_variable() {
    for statement in [
        "MATCH (t:T) INSERT (t IS C)-[:R]->(m:B {k: 2})",
        "MATCH (t:T) INSERT (t:A&B)-[:R]->(m:B {k: 2})",
        "MATCH (t:T) INSERT (m:B {k: 2})-[:R]->(t IS C)",
        "MATCH (t:T) INSERT (t:A)-[:R]->(m:B {k: 2})",
    ] {
        let (_db, session) = fresh();
        session.execute("INSERT (:T {k: 0})").unwrap();
        let error = session.execute(statement).expect_err(statement).to_string();
        assert!(error.contains("labels"), "{statement}: {error}");
        assert_eq!(nodes(&session), vec!["0:T"], "{statement}");
    }
}

/// A node the insert declares with an expression and no variable the match bound is a new node, also
/// when the path ends at a bound one.
#[test]
fn an_expression_declares_a_new_node_beside_a_bound_one() {
    for (statement, wanted) in [
        (
            "MATCH (t:T) INSERT (n IS C)-[:R]->(t)",
            vec!["0:T", "null:C"],
        ),
        (
            "MATCH (t:T) INSERT (t)-[:R]->(n IS C)",
            vec!["0:T", "null:C"],
        ),
        (
            "MATCH (t:T) INSERT (:C&D)-[:R]->(t)",
            vec!["0:T", "null:C,D"],
        ),
        ("MATCH (t:T) INSERT (IS C)-[:R]->(t)", vec!["0:T", "null:C"]),
    ] {
        let (_db, session) = fresh();
        session.execute("INSERT (:T {k: 0})").unwrap();
        session
            .execute(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
        assert_eq!(nodes(&session), wanted, "{statement}");
        let edges = session.execute("MATCH (:C)-[:R]-(:T) RETURN 1").unwrap();
        assert_eq!(edges.row_count(), 1, "{statement}");
    }
}

/// In a transaction, the refused write adds nothing to what the transaction has.
#[test]
fn a_refused_write_adds_nothing_to_its_transaction() {
    let (_db, mut session) = fresh();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:A {k: 1})").unwrap();
    assert!(session.execute("INSERT (n:A|B {k: 2})").is_err());
    assert!(session.execute("MERGE (n:!A {k: 3})").is_err());
    assert_eq!(nodes(&session), vec!["1:A"]);
    session.rollback().unwrap();
}

// ---------------------------------------------------------------------------
// MERGE
// ---------------------------------------------------------------------------

#[test]
fn merge_creates_with_exactly_the_labels() {
    for (statement, wanted) in [
        ("MERGE (n IS A {k: 1})", "1:A"),
        ("MERGE (n:A&B {k: 1})", "1:A,B"),
        ("MERGE (n IS A&B {k: 1})", "1:A,B"),
        ("MERGE (n:A&B&C {k: 1})", "1:A,B,C"),
        ("MERGE (n:A {k: 1})", "1:A"),
        ("MERGE (n:A:B {k: 1})", "1:A,B"),
    ] {
        let (_db, session) = fresh();
        session.execute(statement).unwrap();
        assert_eq!(nodes(&session), vec![wanted], "{statement}");
        session.execute(statement).unwrap();
        assert_eq!(nodes(&session), vec![wanted], "{statement}, a second time");
    }
}

/// What the graph holds before, the statement, and the graph after: whether it matched is whether the
/// node is new. The labels decide, not the property alone.
#[test]
fn merge_matches_on_the_labels() {
    for (before, statement, wanted) in [
        (
            "INSERT (:Other {k: 1})",
            "MERGE (n IS A {k: 1})",
            vec!["1:A", "1:Other"],
        ),
        (
            "INSERT ({k: 1})",
            "MERGE (n IS A {k: 1})",
            vec!["1:", "1:A"],
        ),
        (
            "INSERT (:A {k: 1})",
            "MERGE (n:A&B {k: 1})",
            vec!["1:A", "1:A,B"],
        ),
        (
            "INSERT (:B {k: 1})",
            "MERGE (n IS A&B {k: 1})",
            vec!["1:A,B", "1:B"],
        ),
        (
            "INSERT (:A:B {k: 1})",
            "MERGE (n IS A {k: 1})",
            vec!["1:A,B"],
        ),
        (
            "INSERT (:A:B:C {k: 1})",
            "MERGE (n:A&B {k: 1})",
            vec!["1:A,B,C"],
        ),
        (
            "INSERT (:A {k: 2})",
            "MERGE (n IS A {k: 1})",
            vec!["1:A", "2:A"],
        ),
        (
            "INSERT (:A {k: 1}), (:B {k: 2})",
            "MERGE (n IS A)",
            vec!["1:A", "2:B"],
        ),
        (
            "INSERT (:B {k: 2})",
            "MERGE (n IS A)",
            vec!["2:B", "null:A"],
        ),
        (
            "INSERT (:Other {k: 1})",
            "MERGE (n:A {k: 1})",
            vec!["1:A", "1:Other"],
        ),
        (
            "INSERT (:A {k: 1})",
            "MERGE (n:A:B {k: 1})",
            vec!["1:A", "1:A,B"],
        ),
    ] {
        let (_db, session) = fresh();
        session.execute(before).unwrap();
        session
            .execute(statement)
            .unwrap_or_else(|error| panic!("{before}; {statement}: {error}"));
        assert_eq!(nodes(&session), wanted, "{before}; {statement}");
    }
}

/// The same with an index on the key, where the lookup must still check the labels.
#[test]
fn merge_matches_on_the_labels_through_an_index() {
    for index in [
        "CREATE INDEX by_k FOR (n:Other) ON (n.k)",
        "CREATE INDEX by_k FOR (n:A) ON (n.k)",
        "CREATE INDEX by_k FOR (n:B) ON (n.k)",
    ] {
        let (_db, session) = fresh();
        run_all(
            &session,
            &[index, "INSERT (:Other {k: 1})", "INSERT (:A {k: 2})"],
        );
        run_all(
            &session,
            &[
                "MERGE (n IS A {k: 1})",
                "MERGE (n IS A {k: 1})",
                "MERGE (n:A&B {k: 2})",
                "MERGE (n:A&B {k: 2})",
            ],
        );
        assert_eq!(
            nodes(&session),
            vec!["1:A", "1:Other", "2:A", "2:A,B"],
            "{index}"
        );
    }
    let db = GrafeoDB::with_config(Config::in_memory()).unwrap();
    db.create_property_index("k").unwrap();
    let session = db.session();
    run_all(
        &session,
        &[
            "INSERT (:Other {k: 1})",
            "MERGE (n IS A {k: 1})",
            "MERGE (n IS A {k: 1})",
        ],
    );
    assert_eq!(nodes(&session), vec!["1:A", "1:Other"], "key-wide");
}

/// `ON CREATE` and `ON MATCH` follow whether the labels held.
#[test]
fn merge_runs_on_create_or_on_match_by_the_labels() {
    let on = "ON CREATE SET n.new = true ON MATCH SET n.seen = true";
    let read = |session: &Session, query: &str| session.execute(query).unwrap().row_count();
    let (_db, session) = fresh();
    session.execute("INSERT (:Other {k: 1})").unwrap();
    session
        .execute(&format!("MERGE (n IS A {{k: 1}}) {on}"))
        .unwrap();
    assert_eq!(read(&session, "MATCH (n:A {new: true}) RETURN n"), 1);
    assert_eq!(read(&session, "MATCH (n {seen: true}) RETURN n"), 0);
    assert_eq!(read(&session, "MATCH (n:Other {new: true}) RETURN n"), 0);
    let (_db, session) = fresh();
    session.execute("INSERT (:A:B {k: 1})").unwrap();
    session
        .execute(&format!("MERGE (n:A&B {{k: 1}}) {on}"))
        .unwrap();
    assert_eq!(read(&session, "MATCH (n:A:B {seen: true}) RETURN n"), 1);
    assert_eq!(read(&session, "MATCH (n {new: true}) RETURN n"), 0);
}

#[test]
fn merge_returns_the_node_it_found_or_made() {
    let (_db, session) = fresh();
    session
        .execute("INSERT (:A:B {k: 1, name: 'Alix'})")
        .unwrap();
    let found = session
        .execute("MERGE (n IS A {k: 1}) RETURN n.name, size(labels(n))")
        .unwrap();
    assert_eq!(
        found.rows(),
        vec![vec![Value::from("Alix"), Value::Int64(2)]]
    );
    let made = session
        .execute("MERGE (n:C&D {k: 2}) RETURN n.k, size(labels(n))")
        .unwrap();
    assert_eq!(made.rows(), vec![vec![Value::Int64(2), Value::Int64(2)]]);
}

/// A `MERGE` of a path takes labels only, as before, and writes nothing.
#[test]
fn a_merge_of_a_path_keeps_refusing_an_expression() {
    for statement in [
        "MERGE (n IS A)-[:R]->(m)",
        "MERGE (n:A&B)-[:R]->(m)",
        "MERGE (n:A)-[:R]->(m IS B)",
        "MERGE (n:A)-[:R]->(m:B&C)",
        "MERGE (n:A|B)-[:R]->(m)",
    ] {
        let (_db, session) = fresh();
        let error = session.execute(statement).expect_err(statement).to_string();
        assert!(error.contains("MERGE"), "{statement}: {error}");
        assert_eq!(nodes(&session), Vec::<String>::new(), "{statement}");
    }
    let (_db, session) = fresh();
    session.execute("MERGE (n:A)-[:R]->(m:B:C)").unwrap();
    assert_eq!(nodes(&session), vec!["null:A", "null:B,C"]);
}

// ---------------------------------------------------------------------------
// Cypher refuses them
// ---------------------------------------------------------------------------

#[test]
fn cypher_refuses_an_expression_on_a_write() {
    for statement in [
        "CREATE (n IS A {k: 1})",
        "CREATE (n:A&B {k: 1})",
        "CREATE (n:A|B {k: 1})",
        "MERGE (n IS A {k: 1})",
        "MERGE (n:A&B {k: 1})",
        "MERGE (n:A|B {k: 1})",
    ] {
        let (_db, session) = fresh();
        assert!(session.execute_cypher(statement).is_err(), "{statement}");
        assert_eq!(nodes(&session), Vec::<String>::new(), "{statement}");
    }
    let (_db, session) = fresh();
    session.execute_cypher("CREATE (n:A:B {k: 1})").unwrap();
    session.execute_cypher("MERGE (n:A:B {k: 1})").unwrap();
    assert_eq!(nodes(&session), vec!["1:A,B"]);
}
