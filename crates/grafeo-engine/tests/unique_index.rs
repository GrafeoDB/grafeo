//! A UNIQUE or NODE KEY rule builds its index, so an insert under it no longer scans the label.
//!
//! Tests first, from the design and the rulings: every statement form makes one index, on the first
//! key, named after a named constraint and with no name otherwise; it goes in every graph the rule
//! reaches and in each graph made later; it goes with a rollback; it stays through a reopen and a
//! crash, and after `DROP CONSTRAINT`, which says so. The index changes no verdict: a rule refuses
//! the same writes with it and without it.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main unique_index::
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use std::path::{Path, PathBuf};

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;

const RULE: &str = "CREATE CONSTRAINT c1 FOR (n:File) ON (n.id) UNIQUE";

fn run_all(session: &Session, statements: &[&str]) {
    for statement in statements {
        session
            .execute(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }
}

/// What `SHOW INDEXES` lists, sorted, `NULL` for a null cell.
fn shown(session: &Session) -> Vec<[String; 4]> {
    let mut rows: Vec<[String; 4]> = session
        .execute("SHOW INDEXES")
        .unwrap()
        .rows()
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| match value {
                    Value::String(value) => value.to_string(),
                    Value::Null => "NULL".to_string(),
                    other => panic!("SHOW INDEXES lists strings and nulls: {other:?}"),
                })
                .collect::<Vec<_>>()
                .try_into()
                .expect("four columns")
        })
        .collect();
    rows.sort();
    rows
}

fn rows(wanted: &[[&str; 4]]) -> Vec<[String; 4]> {
    let mut rows: Vec<_> = wanted.iter().map(|row| row.map(str::to_string)).collect();
    rows.sort();
    rows
}

/// Whether a key from the rows finds `label`'s nodes through an index on `key`, in the session's
/// graph: the physical index, whatever the names say. The graph needs a node of `label`.
fn reads_index(session: &Session, label: &str, key: &str) -> bool {
    let query = format!("UNWIND [1] AS x MATCH (n:{label}) WHERE n.{key} = x RETURN n.{key}");
    let plan = session.execute(&format!("EXPLAIN {query}")).unwrap().rows()[0][0]
        .as_str()
        .unwrap()
        .to_string();
    plan.contains(&format!("[index: {key}]"))
}

fn constraint_count(session: &Session) -> usize {
    session.execute("SHOW CONSTRAINTS").unwrap().row_count()
}

fn count(session: &Session, query: &str) -> i64 {
    match session.execute(query).unwrap().rows()[0][0] {
        Value::Int64(count) => count,
        ref other => panic!("{query}: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Every statement form
// ---------------------------------------------------------------------------

#[test]
fn each_form_of_the_rule_builds_one_index_on_its_first_key() {
    let named = [["c1", "Hash", "File", "id"]];
    let unnamed = [["NULL", "Hash", "File", "id"]];
    for (language, statement, wanted) in [
        ("gql", RULE, &named),
        (
            "gql",
            "CREATE CONSTRAINT FOR (n:File) ON (n.id) UNIQUE",
            &unnamed,
        ),
        (
            "gql",
            "CREATE CONSTRAINT c1 FOR (n:File) ON (n.id) NODE KEY",
            &named,
        ),
        (
            "gql",
            "CREATE CONSTRAINT FOR (n:File) ON (n.id) NODE KEY",
            &unnamed,
        ),
        (
            "gql",
            "CREATE CONSTRAINT c1 FOR (n:File) ON (n.id, n.part) NODE KEY",
            &named,
        ),
        (
            "gql",
            "CREATE CONSTRAINT c1 FOR (n:File) ON (n.id, n.part) UNIQUE",
            &named,
        ),
        (
            "cypher",
            "CREATE CONSTRAINT c1 FOR (n:File) REQUIRE n.id IS UNIQUE",
            &named,
        ),
        (
            "cypher",
            "CREATE CONSTRAINT FOR (n:File) REQUIRE n.id IS UNIQUE",
            &unnamed,
        ),
        (
            "cypher",
            "CREATE CONSTRAINT c1 FOR (n:File) REQUIRE n.id IS NODE KEY",
            &named,
        ),
    ] {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        match language {
            "gql" => session.execute(statement),
            _ => session.execute_cypher(statement),
        }
        .unwrap_or_else(|error| panic!("{statement}: {error}"));
        assert_eq!(shown(&session), rows(wanted), "{statement}");
        assert!(db.has_property_index("id"), "{statement}");
        assert!(
            !db.has_property_index("part"),
            "{statement}: only the first key"
        );
        run_all(&session, &["INSERT (:File {id: 1, part: 1})"]);
        assert!(reads_index(&session, "File", "id"), "{statement}");
    }
}

#[test]
fn the_rules_that_check_no_value_against_another_build_no_index() {
    for statement in [
        "CREATE CONSTRAINT c1 FOR (n:File) ON (n.id) NOT NULL",
        "CREATE CONSTRAINT c1 FOR (n:File) ON (n.id) EXISTS",
    ] {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        run_all(&session, &[statement]);
        assert_eq!(shown(&session), rows(&[]), "{statement}");
        assert!(!db.has_property_index("id"), "{statement}");
    }
}

/// The primary key of `CREATE PROPERTY GRAPH` is the inline form: its index goes in the new graph.
#[test]
fn a_primary_key_builds_its_index_in_the_new_property_graph() {
    let db = GrafeoDB::new_in_memory();
    db.execute_sql(
        "CREATE PROPERTY GRAPH Shop NODE TABLES (Item (id BIGINT PRIMARY KEY, name VARCHAR))",
    )
    .unwrap();
    let session = db.session();
    assert!(!db.has_property_index("id"), "not in the default graph");
    run_all(
        &session,
        &["USE GRAPH Shop", "INSERT (:Item {id: 1, name: 'lamp'})"],
    );
    assert!(reads_index(&session, "Item", "id"));
    assert!(!reads_index(&session, "Item", "name"), "only the key");
    assert!(
        session
            .execute("INSERT (:Item {id: 1, name: 'desk'})")
            .is_err()
    );
}

#[test]
fn an_index_already_there_gets_no_second_and_stays_after_the_rule() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &["CREATE INDEX by_id FOR (n:File) ON (n.id)", RULE],
    );
    // The statement built nothing, so it gave no name: `c1` is no index.
    assert_eq!(shown(&session), rows(&[["by_id", "Hash", "File", "id"]]));
    run_all(&session, &["DROP CONSTRAINT c1", "INSERT (:File {id: 1})"]);
    assert!(db.has_property_index("id"));
    assert!(reads_index(&session, "File", "id"));
    assert!(shown(&session).iter().any(|row| row[0] == "by_id"));
    assert_eq!(constraint_count(&session), 0);
}

/// One graph has an index on the key and another has none: the statement builds there, so it
/// gives its name, and the graph that had one keeps its own.
#[test]
fn a_rule_over_an_index_in_one_graph_builds_in_the_other_and_names_it() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            "CREATE GRAPH other",
            "CREATE INDEX by_id FOR (n:File) ON (n.id)",
            RULE,
        ],
    );
    let listed = shown(&session);
    for name in ["by_id", "c1"] {
        assert!(
            listed.iter().any(|row| row[0] == name),
            "{name}: {listed:?}"
        );
    }
    run_all(&session, &["INSERT (:File {id: 1})"]);
    assert!(
        reads_index(&session, "File", "id"),
        "the graph that had one"
    );
    run_all(&session, &["USE GRAPH other", "INSERT (:File {id: 1})"]);
    assert!(reads_index(&session, "File", "id"), "the graph it built in");
}

/// A rule the data breaks fails, and the index it would have built is not left behind.
#[test]
fn a_refused_rule_leaves_no_index() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(&session, &["INSERT (:File {id: 1}), (:File {id: 1})"]);
    assert!(session.execute(RULE).is_err());
    assert_eq!(shown(&session), rows(&[]));
    assert!(!db.has_property_index("id"));

    let mut open = db.session();
    open.begin_transaction().unwrap();
    if open.execute(RULE).is_ok() {
        assert!(open.commit().is_err());
    } else {
        open.rollback().unwrap();
    }
    assert_eq!(shown(&session), rows(&[]));
    assert!(!db.has_property_index("id"));
}

// ---------------------------------------------------------------------------
// The index holds the rows
// ---------------------------------------------------------------------------

/// Rows from before the rule, from the rule's own transaction and from after it are all found, and
/// a key that moves is found where it went.
#[test]
fn the_index_finds_what_was_written_before_and_after_the_rule() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    run_all(&session, &["INSERT (:File {id: 1, name: 'before'})"]);
    session.begin_transaction().unwrap();
    run_all(
        &session,
        &[
            "INSERT (:File {id: 2, name: 'ahead in the transaction'})",
            RULE,
            "INSERT (:File {id: 3, name: 'behind in the transaction'})",
        ],
    );
    session.commit().unwrap();
    run_all(&session, &["INSERT (:File {id: 4, name: 'after'})"]);
    assert!(reads_index(&session, "File", "id"));
    let found = "UNWIND [1, 2, 3, 4, 5, 6] AS x MATCH (n:File) WHERE n.id = x RETURN count(n)";
    assert_eq!(count(&session, found), 4);
    for id in 1..=4 {
        assert!(
            session
                .execute(&format!("INSERT (:File {{id: {id}}})"))
                .is_err(),
            "{id} is taken"
        );
    }
    run_all(
        &session,
        &[
            "MATCH (n:File {id: 1}) SET n.id = 5",
            "MATCH (n:File {id: 2}) DELETE n",
            "MATCH (n:File {id: 3}) REMOVE n.id",
        ],
    );
    assert_eq!(count(&session, found), 2);
    run_all(
        &session,
        &[
            "INSERT (:File {id: 1})",
            "INSERT (:File {id: 2})",
            "INSERT (:File {id: 3})",
        ],
    );
    assert!(session.execute("INSERT (:File {id: 5})").is_err());
    assert_eq!(count(&session, found), 5);
}

// ---------------------------------------------------------------------------
// A transaction
// ---------------------------------------------------------------------------

#[test]
fn a_rollback_takes_the_index_and_a_commit_keeps_it() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    run_all(&session, &["INSERT (:File {id: 1})"]);

    session.begin_transaction().unwrap();
    run_all(&session, &[RULE]);
    session.rollback().unwrap();
    assert_eq!(shown(&session), rows(&[]));
    assert!(!db.has_property_index("id"));
    assert!(!reads_index(&session, "File", "id"));
    run_all(&session, &["INSERT (:File {id: 2})"]);

    session.begin_transaction().unwrap();
    run_all(&session, &[RULE]);
    session.commit().unwrap();
    assert_eq!(shown(&session), rows(&[["c1", "Hash", "File", "id"]]));
    assert!(reads_index(&session, "File", "id"));
}

/// An index's claim is per graph, label and key: rules on one key and two labels both commit, each
/// with its own index, and a second rule on the same label conflicts at its statement.
#[test]
fn two_open_rules_conflict_only_on_one_label_and_key() {
    let other_label = "CREATE CONSTRAINT c2 FOR (n:Other) ON (n.id) UNIQUE";
    let same_label = "CREATE CONSTRAINT c3 FOR (n:File) ON (n.id, n.part) NODE KEY";
    let db = GrafeoDB::new_in_memory();
    let (mut first, mut second) = (db.session(), db.session());
    first.begin_transaction().unwrap();
    second.begin_transaction().unwrap();
    run_all(&first, &[RULE]);
    run_all(&second, &[other_label]);
    first.commit().unwrap();
    second.commit().unwrap();
    assert_eq!(
        shown(&db.session()),
        rows(&[["c1", "Hash", "File", "id"], ["c2", "Hash", "Other", "id"]])
    );

    let db = GrafeoDB::new_in_memory();
    let (mut first, mut second) = (db.session(), db.session());
    first.begin_transaction().unwrap();
    second.begin_transaction().unwrap();
    run_all(&first, &[RULE]);
    let error = second
        .execute(same_label)
        .expect_err("the label and key are claimed");
    assert!(error.to_string().contains("GRAFEO-T001"), "{error}");
    second.rollback().unwrap();
    first.commit().unwrap();
    run_all(&db.session(), &[same_label]);
    assert_eq!(constraint_count(&db.session()), 2);
    assert_eq!(shown(&db.session()), rows(&[["c1", "Hash", "File", "id"]]));
}

/// Rules on two keys take no claim of each other, and both commit.
#[test]
fn two_open_rules_on_two_keys_both_commit() {
    let db = GrafeoDB::new_in_memory();
    let (mut first, mut second) = (db.session(), db.session());
    first.begin_transaction().unwrap();
    second.begin_transaction().unwrap();
    run_all(&first, &[RULE]);
    run_all(
        &second,
        &["CREATE CONSTRAINT c2 FOR (n:File) ON (n.part) UNIQUE"],
    );
    first.commit().unwrap();
    second.commit().unwrap();
    assert!(db.has_property_index("id"));
    assert!(db.has_property_index("part"));
    assert_eq!(constraint_count(&db.session()), 2);
}

/// A savepoint rolled back past the rule takes the index, as a rollback does.
#[test]
fn a_rollback_to_a_savepoint_takes_the_index() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    run_all(&session, &["INSERT (:File {id: 1})"]);
    session.begin_transaction().unwrap();
    session.savepoint("before").unwrap();
    run_all(&session, &[RULE]);
    session.rollback_to_savepoint("before").unwrap();
    session.commit().unwrap();
    assert_eq!(shown(&session), rows(&[]));
    assert!(!db.has_property_index("id"));
    assert_eq!(constraint_count(&session), 0);
}

// ---------------------------------------------------------------------------
// `DROP CONSTRAINT` and names
// ---------------------------------------------------------------------------

#[test]
fn drop_constraint_leaves_the_index_and_says_so() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(&session, &[RULE, "INSERT (:File {id: 1})"]);
    let status = session
        .execute("DROP CONSTRAINT c1")
        .unwrap()
        .status_message
        .expect("a status");
    assert!(status.contains("index 'c1' stays"), "{status}");
    assert!(status.contains("DROP INDEX c1"), "{status}");
    assert_eq!(shown(&session), rows(&[["c1", "Hash", "File", "id"]]));
    assert!(reads_index(&session, "File", "id"));
    run_all(&session, &["INSERT (:File {id: 1})"]);

    run_all(&session, &["DROP INDEX c1"]);
    assert_eq!(shown(&session), rows(&[]));
    assert!(!db.has_property_index("id"));
    assert_eq!(
        count(&session, "MATCH (n:File) WHERE n.id = 1 RETURN count(n)"),
        2
    );
}

/// A constraint given no name gets a generated one, and its index none: the status names no index.
#[test]
fn a_constraint_with_no_name_leaves_an_index_with_none() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &["CREATE CONSTRAINT FOR (n:File) ON (n.id) UNIQUE"],
    );
    let generated = session.execute("SHOW CONSTRAINTS").unwrap();
    let name = generated.rows()[0][0].as_str().unwrap().to_string();
    let status = session
        .execute(&format!("DROP CONSTRAINT {name}"))
        .unwrap()
        .status_message
        .expect("a status");
    assert!(!status.contains("index '"), "{status}");
    assert!(
        status.contains("the index on property id stays"),
        "{status}"
    );
    assert_eq!(shown(&session), rows(&[["NULL", "Hash", "File", "id"]]));
    assert!(db.drop_property_index("id").unwrap());
}

/// A name that stands for another index stays that index's: the rule's index has no name.
#[test]
fn a_name_taken_by_another_index_is_not_given() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(&session, &["CREATE INDEX c1 FOR (n:Other) ON (n.x)", RULE]);
    assert_eq!(
        shown(&session),
        rows(&[["c1", "Hash", "Other", "x"], ["NULL", "Hash", "File", "id"]])
    );
    let status = session
        .execute("DROP CONSTRAINT c1")
        .unwrap()
        .status_message
        .expect("a status");
    assert!(!status.contains("DROP INDEX"), "{status}");
    assert!(
        status.contains("the index on property id stays"),
        "{status}"
    );
    assert!(db.has_property_index("id"));
    assert!(db.has_property_index("x"));
}

/// With its index dropped the rule still holds: it scans, as it did.
#[test]
fn a_rule_without_its_index_still_refuses() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(&session, &[RULE, "INSERT (:File {id: 1})", "DROP INDEX c1"]);
    assert!(!db.has_property_index("id"));
    assert_eq!(constraint_count(&session), 1);
    assert!(session.execute("INSERT (:File {id: 1})").is_err());
    run_all(&session, &["INSERT (:File {id: 2})"]);
}

// ---------------------------------------------------------------------------
// Every graph the rule reaches
// ---------------------------------------------------------------------------

#[test]
fn the_index_goes_in_every_graph_there_whichever_is_active() {
    for active in ["default", "other"] {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        run_all(&session, &["CREATE GRAPH other", "CREATE GRAPH third"]);
        run_all(&session, &[&format!("USE GRAPH {active}"), RULE]);
        for graph in ["default", "other", "third"] {
            run_all(
                &session,
                &[&format!("USE GRAPH {graph}"), "INSERT (:File {id: 1})"],
            );
            assert!(reads_index(&session, "File", "id"), "{active}: {graph}");
            assert!(
                session.execute("INSERT (:File {id: 1})").is_err(),
                "{active}: {graph}"
            );
        }
    }
}

#[test]
fn a_graph_made_after_the_rule_gets_the_index() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            RULE,
            "CREATE GRAPH later",
            "USE GRAPH later",
            "INSERT (:File {id: 1})",
        ],
    );
    assert!(reads_index(&session, "File", "id"));
    assert!(session.execute("INSERT (:File {id: 1})").is_err());
}

/// The API's way of making a graph is the same operation.
#[test]
fn a_graph_made_through_the_api_gets_the_index() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(&session, &[RULE]);
    assert!(db.create_graph("later").unwrap());
    run_all(&session, &["USE GRAPH later", "INSERT (:File {id: 1})"]);
    assert!(reads_index(&session, "File", "id"));
}

/// A rule reaches a bound graph whose type lists the label, and not one whose type does not.
#[test]
fn a_bound_graph_gets_the_index_when_its_type_lists_the_label() {
    for rule_first in [true, false] {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        run_all(
            &session,
            &[
                "CREATE NODE TYPE File (id INT64)",
                "CREATE NODE TYPE Town (id INT64)",
            ],
        );
        if rule_first {
            run_all(&session, &[RULE]);
        }
        run_all(
            &session,
            &[
                "CREATE GRAPH TYPE listed (NODE TYPE File)",
                "CREATE GRAPH TYPE unlisted (NODE TYPE Town)",
                "CREATE GRAPH with_file TYPED listed",
                "CREATE GRAPH with_town TYPED unlisted",
            ],
        );
        if !rule_first {
            run_all(&session, &[RULE]);
        }
        run_all(&session, &["USE GRAPH with_file", "INSERT (:File {id: 1})"]);
        assert!(
            reads_index(&session, "File", "id"),
            "rule first: {rule_first}"
        );
        run_all(&session, &["USE GRAPH with_town", "INSERT (:Town {id: 1})"]);
        assert!(
            !reads_index(&session, "Town", "id"),
            "rule first: {rule_first}"
        );
    }
}

#[test]
fn a_rollback_of_the_rule_takes_its_index_from_every_graph() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    run_all(&session, &["CREATE GRAPH other"]);
    session.begin_transaction().unwrap();
    run_all(&session, &[RULE]);
    session.rollback().unwrap();
    for graph in ["default", "other"] {
        run_all(
            &session,
            &[&format!("USE GRAPH {graph}"), "INSERT (:File {id: 1})"],
        );
        assert!(!reads_index(&session, "File", "id"), "{graph}");
    }
}

#[test]
fn a_rollback_of_a_new_graph_takes_its_index_and_a_new_one_has_it() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    run_all(&session, &[RULE]);
    session.begin_transaction().unwrap();
    run_all(&session, &["CREATE GRAPH later"]);
    session.rollback().unwrap();
    assert!(
        session.execute("USE GRAPH later").is_err(),
        "the graph went"
    );
    run_all(
        &session,
        &[
            "CREATE GRAPH later",
            "USE GRAPH later",
            "INSERT (:File {id: 1})",
        ],
    );
    assert!(reads_index(&session, "File", "id"));
}

/// One transaction, in both orders: the new graph has the index.
#[test]
fn a_rule_and_a_graph_in_one_transaction_meet_in_either_order() {
    for statements in [[RULE, "CREATE GRAPH later"], ["CREATE GRAPH later", RULE]] {
        let db = GrafeoDB::new_in_memory();
        let mut session = db.session();
        session.begin_transaction().unwrap();
        run_all(&session, &statements);
        session.commit().unwrap();
        run_all(&session, &["USE GRAPH later", "INSERT (:File {id: 1})"]);
        assert!(reads_index(&session, "File", "id"), "{statements:?}");
        run_all(&session, &["USE GRAPH default", "INSERT (:File {id: 1})"]);
        assert!(reads_index(&session, "File", "id"), "{statements:?}");
    }
}

/// `DROP INDEX` takes the name and the active graph's index; the others stay, with no name.
#[test]
fn drop_index_takes_the_active_graphs_index_and_the_others_stay() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(&session, &["CREATE GRAPH other", RULE, "DROP INDEX c1"]);
    assert!(!db.has_property_index("id"));
    run_all(&session, &["USE GRAPH other", "INSERT (:File {id: 1})"]);
    assert_eq!(shown(&session), rows(&[["NULL", "Hash", "File", "id"]]));
    assert!(reads_index(&session, "File", "id"));
}

// ---------------------------------------------------------------------------
// A reopen and a crash
// ---------------------------------------------------------------------------

fn run_child(child: &mut std::process::Command) {
    let status = grafeo_common::testing::child_process::run(child).unwrap();
    assert!(status.success(), "the child process failed");
}

/// What must hold of a database that had the rule and `{id: 1}` in `File`, in each graph named.
fn assert_rule_and_index(path: &Path, graphs: &[&str]) {
    let db = GrafeoDB::open(path).unwrap();
    let session = db.session();
    assert_eq!(shown(&session), rows(&[["c1", "Hash", "File", "id"]]));
    for graph in graphs {
        run_all(&session, &[&format!("USE GRAPH {graph}")]);
        assert!(reads_index(&session, "File", "id"), "{graph}");
        assert!(
            session.execute("INSERT (:File {id: 1})").is_err(),
            "{graph}"
        );
        run_all(&session, &["INSERT (:File {id: 2})"]);
        assert_eq!(
            count(
                &session,
                "UNWIND [1, 2] AS x MATCH (n:File) WHERE n.id = x RETURN count(n)"
            ),
            2,
            "{graph}"
        );
    }
    drop(session);
    db.close().unwrap();
}

#[test]
fn a_reopen_keeps_the_index_in_every_graph() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("closed.grafeo");
    let db = GrafeoDB::open(&path).unwrap();
    let session = db.session();
    run_all(
        &session,
        &[
            "CREATE GRAPH other",
            RULE,
            "CREATE GRAPH later",
            "INSERT (:File {id: 1})",
            "USE GRAPH other",
            "INSERT (:File {id: 1})",
            "USE GRAPH later",
            "INSERT (:File {id: 1})",
        ],
    );
    drop(session);
    db.close().unwrap();
    assert_rule_and_index(&path, &["default", "other", "later"]);
}

/// A dropped rule's index is still there after a reopen, and the rule is not.
#[test]
fn a_reopen_after_drop_constraint_keeps_the_index_alone() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dropped.grafeo");
    let db = GrafeoDB::open(&path).unwrap();
    let session = db.session();
    run_all(
        &session,
        &[RULE, "INSERT (:File {id: 1})", "DROP CONSTRAINT c1"],
    );
    drop(session);
    db.close().unwrap();
    let db = GrafeoDB::open(&path).unwrap();
    let session = db.session();
    assert_eq!(constraint_count(&session), 0);
    assert_eq!(shown(&session), rows(&[["c1", "Hash", "File", "id"]]));
    assert!(reads_index(&session, "File", "id"));
    run_all(&session, &["INSERT (:File {id: 1})"]);
    drop(session);
    db.close().unwrap();
}

const CRASH_PATH: &str = "GRAFEO_UNIQUE_INDEX_CRASH_PATH";
const CRASH_POINT: &str = "GRAFEO_UNIQUE_INDEX_CRASH_POINT";
#[cfg(feature = "testing-crash-injection")]
const CRASH_MARKER: &str = "GRAFEO_UNIQUE_INDEX_CRASH_MARKER";
#[cfg(feature = "testing-crash-injection")]
const CRASH_TRANSACTION: &str = "GRAFEO_UNIQUE_INDEX_CRASH_TRANSACTION";

/// Everything committed, then the process ends with no close: the log has it all.
#[test]
fn committed_rule_child() {
    let Some(path) = std::env::var_os(CRASH_PATH) else {
        return;
    };
    if std::env::var_os(CRASH_POINT).is_some() {
        return;
    }
    let db = GrafeoDB::open(PathBuf::from(path)).unwrap();
    let mut session = db.session();
    run_all(&session, &["CREATE GRAPH other"]);
    session.begin_transaction().unwrap();
    run_all(&session, &[RULE]);
    session.commit().unwrap();
    run_all(
        &session,
        &[
            "CREATE GRAPH later",
            "INSERT (:File {id: 1})",
            "USE GRAPH other",
            "INSERT (:File {id: 1})",
            "USE GRAPH later",
            "INSERT (:File {id: 1})",
        ],
    );
    std::process::exit(0);
}

#[test]
fn a_crash_after_the_commit_keeps_the_index_in_every_graph() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("crashed.grafeo");
    run_child(grafeo_common::child_test!("committed_rule_child").env(CRASH_PATH, &path));
    assert_rule_and_index(&path, &["default", "other", "later"]);
}

/// The rule's statement, or its commit, up to a crash at the point given. Reports through the
/// marker file when the statement finished before the point came.
#[cfg(feature = "testing-crash-injection")]
#[test]
fn rule_crash_child() {
    use grafeo_common::testing::crash::{CrashResult, with_crash_at};
    use std::panic::AssertUnwindSafe;

    let (Some(path), Some(point), Some(marker)) = (
        std::env::var_os(CRASH_PATH),
        std::env::var_os(CRASH_POINT),
        std::env::var_os(CRASH_MARKER),
    ) else {
        return;
    };
    let point: u64 = point.to_str().unwrap().parse().unwrap();
    let in_transaction = std::env::var_os(CRASH_TRANSACTION).is_some();
    let db = GrafeoDB::open(PathBuf::from(path)).unwrap();
    let mut session = db.session();
    run_all(&session, &["INSERT (:File {id: 1})"]);
    let outcome = if in_transaction {
        session.begin_transaction().unwrap();
        run_all(&session, &[RULE]);
        with_crash_at(point, AssertUnwindSafe(|| session.commit().unwrap()))
    } else {
        with_crash_at(
            point,
            AssertUnwindSafe(|| {
                session.execute(RULE).unwrap();
            }),
        )
    };
    if matches!(outcome, CrashResult::Completed(())) {
        std::fs::write(PathBuf::from(marker), b"done").unwrap();
    }
    std::process::exit(0);
}

#[cfg(feature = "testing-crash-injection")]
fn crash_sweep(in_transaction: bool) {
    for point in 1..=400_u64 {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sweep.grafeo");
        let marker = dir.path().join("completed");
        let mut child = grafeo_common::child_test!("rule_crash_child");
        child
            .env(CRASH_PATH, &path)
            .env(CRASH_POINT, point.to_string())
            .env(CRASH_MARKER, &marker);
        if in_transaction {
            child.env(CRASH_TRANSACTION, "1");
        }
        run_child(&mut child);

        let db = GrafeoDB::open(&path).unwrap();
        let session = db.session();
        let rule = constraint_count(&session) == 1;
        let index = db.has_property_index("id");
        assert_eq!(
            rule, index,
            "a crash at {point}: the rule {rule}, the index {index}"
        );
        assert_eq!(
            rule,
            reads_index(&session, "File", "id"),
            "a crash at {point}"
        );
        assert_eq!(
            session.execute("INSERT (:File {id: 1})").is_err(),
            rule,
            "a crash at {point}"
        );
        drop(session);
        db.close().unwrap();
        if marker.exists() {
            assert!(rule, "the statement finished at {point}");
            assert!(point > 1, "no point crashed before the statement finished");
            return;
        }
    }
    panic!("the statement never finished within the points tried");
}

#[cfg(feature = "testing-crash-injection")]
#[test]
fn a_crash_in_the_statement_leaves_both_or_neither() {
    crash_sweep(false);
}

#[cfg(feature = "testing-crash-injection")]
#[test]
fn a_crash_in_the_commit_leaves_both_or_neither() {
    crash_sweep(true);
}

// ---------------------------------------------------------------------------
// The index changes no verdict
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum Index {
    /// The one the rule builds.
    TheRules,
    /// The rule's own dropped: the rule scans.
    Dropped,
    /// One from the API before the rule. Today, the only way to have one.
    MadeFirst,
}

fn ruled(index: Index) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    if matches!(index, Index::MadeFirst) {
        db.create_property_index("id").unwrap();
    }
    db.execute(RULE).unwrap();
    if matches!(index, Index::Dropped) {
        db.execute("DROP INDEX c1").unwrap();
        assert!(!db.has_property_index("id"));
    }
    db
}

/// A value of each kind and spelling `=` treats specially, NaN and null beside them.
fn values() -> Vec<Value> {
    let db = GrafeoDB::new_in_memory();
    let mut values: Vec<Value> = [
        "1",
        "1.0",
        "'1'",
        "'1.0'",
        "42",
        "'042'",
        "'42.0'",
        "0.1e0 + 0.2e0",
        "0.3e0",
        "'0.3'",
        "0.0e0",
        "-0.0e0",
        "true",
        "'true'",
        "[1]",
        "[1.0]",
        "['1']",
        "{a: 1}",
        "{a: 1.0}",
        "9007199254740993",
        "9007199254740992.0e0",
    ]
    .iter()
    .map(|literal| {
        db.execute(&format!("RETURN {literal} AS v"))
            .unwrap_or_else(|error| panic!("{literal}: {error}"))
            .rows()[0][0]
            .clone()
    })
    .collect();
    values.push(Value::Float64(f64::NAN));
    values.push(Value::Null);
    values
}

/// What happens to each write of a scenario, in order: `Ok` or the error's text.
fn outcome<T>(result: grafeo_common::utils::error::Result<T>) -> String {
    match result {
        Ok(_) => "accepted".to_string(),
        Err(error) => error.to_string(),
    }
}

fn verdicts_on_value_pairs(index: Index) -> Vec<String> {
    let values = values();
    let insert = "INSERT (:File {id: $v})";
    let mut verdicts = Vec::new();
    for first in &values {
        for second in &values {
            let db = ruled(index);
            let session = db.session();
            let params =
                |value: &Value| std::collections::HashMap::from([("v".to_string(), value.clone())]);
            let one = outcome(session.execute_with_params(insert, params(first)));
            let two = outcome(session.execute_with_params(insert, params(second)));
            let held = count(&session, "MATCH (n:File) RETURN count(n)");
            verdicts.push(format!(
                "{first:?} then {second:?}: {one}; {two}; {held} held"
            ));
        }
    }
    verdicts
}

/// Each scenario as text: every write's outcome and what is held at the end.
fn verdicts_on_scenarios(index: Index) -> Vec<String> {
    let mut verdicts = Vec::new();

    // Another open transaction holds the value.
    let db = ruled(index);
    let (mut a, mut b) = (db.session(), db.session());
    a.begin_transaction().unwrap();
    b.begin_transaction().unwrap();
    let one = outcome(a.execute("INSERT (:File {id: 1})"));
    let two = outcome(b.execute("INSERT (:File {id: 1})"));
    let three = outcome(a.commit());
    let four = outcome(b.commit());
    let held = count(&db.session(), "MATCH (n:File) RETURN count(n)");
    verdicts.push(format!("two open: {one}; {two}; {three}; {four}; {held}"));

    // Another open transaction deletes the value.
    let db = ruled(index);
    run_all(&db.session(), &["INSERT (:File {id: 1})"]);
    let (mut a, mut b) = (db.session(), db.session());
    a.begin_transaction().unwrap();
    b.begin_transaction().unwrap();
    let one = outcome(a.execute("MATCH (n:File {id: 1}) DELETE n"));
    let two = outcome(b.execute("INSERT (:File {id: 1})"));
    let three = outcome(a.commit());
    let four = outcome(b.commit());
    let held = count(&db.session(), "MATCH (n:File) RETURN count(n)");
    verdicts.push(format!(
        "open delete: {one}; {two}; {three}; {four}; {held}"
    ));

    // An open transaction that rolls back holds it no more.
    let db = ruled(index);
    let (mut a, b) = (db.session(), db.session());
    a.begin_transaction().unwrap();
    let one = outcome(a.execute("INSERT (:File {id: 1})"));
    a.rollback().unwrap();
    let two = outcome(b.execute("INSERT (:File {id: 1})"));
    verdicts.push(format!("rolled back: {one}; {two}"));

    // A value committed after a snapshot.
    let db = ruled(index);
    let mut a = db.session();
    a.begin_transaction().unwrap();
    let one = outcome(a.execute("MATCH (n:File) RETURN n.id"));
    let two = outcome(db.session().execute("INSERT (:File {id: 1})"));
    let three = outcome(a.execute("INSERT (:File {id: 1})"));
    let four = outcome(a.commit());
    let held = count(&db.session(), "MATCH (n:File) RETURN count(n)");
    verdicts.push(format!(
        "after the snapshot: {one}; {two}; {three}; {four}; {held}"
    ));

    // A deleted node's value, and values that move.
    let db = ruled(index);
    let session = db.session();
    for (name, statement) in [
        ("first", "INSERT (:File {id: 1, name: 'a'})"),
        ("second", "INSERT (:File {id: 2, name: 'b'})"),
        (
            "set onto a value",
            "MATCH (n:File {name: 'b'}) SET n.id = 1",
        ),
        ("set away", "MATCH (n:File {name: 'a'}) SET n.id = 3"),
        ("the freed value", "INSERT (:File {id: 1, name: 'c'})"),
        ("delete", "MATCH (n:File {name: 'c'}) DELETE n"),
        ("the deleted value", "INSERT (:File {id: 1, name: 'd'})"),
        ("remove", "MATCH (n:File {name: 'd'}) REMOVE n.id"),
        ("the removed value", "INSERT (:File {id: 1, name: 'e'})"),
        (
            "two in a statement",
            "INSERT (:File {id: 7}), (:File {id: 7})",
        ),
    ] {
        verdicts.push(format!("{name}: {}", outcome(session.execute(statement))));
    }
    verdicts.push(format!(
        "held: {}",
        count(&session, "MATCH (n:File) RETURN count(n)")
    ));

    // An index covers the key across labels; the rule covers one label.
    let db = ruled(index);
    let session = db.session();
    for (name, statement) in [
        ("another label", "INSERT (:Other {id: 1})"),
        ("the rule's label", "INSERT (:File {id: 1})"),
        ("another label again", "INSERT (:Other {id: 1})"),
        ("both labels", "INSERT (:File:Other {id: 1})"),
        ("label gained", "MATCH (n:Other) SET n:File"),
        ("label lost", "MATCH (n:File) REMOVE n:File"),
        ("after the label went", "INSERT (:File {id: 1})"),
    ] {
        verdicts.push(format!("{name}: {}", outcome(session.execute(statement))));
    }

    // Another graph holds the value.
    let db = ruled(index);
    let session = db.session();
    for statement in [
        "CREATE GRAPH other",
        "USE GRAPH other",
        "INSERT (:File {id: 1})",
        "USE GRAPH default",
        "INSERT (:File {id: 1})",
        "INSERT (:File {id: 1})",
    ] {
        verdicts.push(format!(
            "{statement}: {}",
            outcome(session.execute(statement))
        ));
    }
    verdicts
}

fn assert_same(with: Vec<String>, without: Vec<String>) {
    assert_eq!(with.len(), without.len());
    let differ: Vec<_> = with
        .iter()
        .zip(&without)
        .filter(|(with, without)| with != without)
        .collect();
    assert!(
        differ.is_empty(),
        "with an index against without, {} differ: {:#?}",
        differ.len(),
        &differ[..differ.len().min(40)]
    );
}

/// Today the rule's own has no index, and the other has one: the comparison the change must keep.
#[test]
fn an_index_made_first_changes_no_verdict() {
    assert_same(
        verdicts_on_value_pairs(Index::MadeFirst),
        verdicts_on_value_pairs(Index::TheRules),
    );
    assert_same(
        verdicts_on_scenarios(Index::MadeFirst),
        verdicts_on_scenarios(Index::TheRules),
    );
}

#[test]
fn a_rule_refuses_alike_with_its_index_and_without() {
    assert_same(
        verdicts_on_value_pairs(Index::TheRules),
        verdicts_on_value_pairs(Index::Dropped),
    );
    assert_same(
        verdicts_on_scenarios(Index::TheRules),
        verdicts_on_scenarios(Index::Dropped),
    );
}

/// A string and a number are two values to the rule, whatever the string spells; two numbers of
/// one value, and two lists of such numbers, are one.
#[test]
fn a_string_and_a_number_are_two_values_to_a_rule() {
    let literal = |text: &str| -> Value {
        GrafeoDB::new_in_memory()
            .execute(&format!("RETURN {text} AS v"))
            .unwrap()
            .rows()[0][0]
            .clone()
    };
    let mut wrong = Vec::new();
    for index in [Index::TheRules, Index::MadeFirst, Index::Dropped] {
        let verdicts = verdicts_on_value_pairs(index);
        let held = |first: &str, second: &str| -> String {
            let (first, second) = (literal(first), literal(second));
            let line = verdicts
                .iter()
                .find(|verdict| verdict.starts_with(&format!("{first:?} then {second:?}: ")))
                .unwrap_or_else(|| panic!("{first:?} then {second:?}"));
            line.rsplit("; ").next().unwrap().to_string()
        };
        for (first, second, count) in [
            ("'042'", "42", "2 held"),
            ("42", "'042'", "2 held"),
            ("'1'", "1", "2 held"),
            ("'1.0'", "1.0", "2 held"),
            ("'true'", "true", "2 held"),
            ("['1']", "[1]", "2 held"),
            ("1", "1.0", "1 held"),
            ("[1]", "[1.0]", "1 held"),
            ("0.0e0", "-0.0e0", "1 held"),
        ] {
            let found = held(first, second);
            if found != count {
                wrong.push(format!(
                    "{index:?}: {first} then {second}: {found}, want {count}"
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// What no reading of `=` can move, whatever the index.
#[test]
fn the_plain_verdicts_hold() {
    for index in [Index::TheRules, Index::MadeFirst] {
        let db = ruled(index);
        let session = db.session();
        run_all(
            &session,
            &["INSERT (:File {id: 1})", "INSERT (:File {id: 2})"],
        );
        assert!(
            session.execute("INSERT (:File {id: 1})").is_err(),
            "{index:?}"
        );
        run_all(
            &session,
            &["INSERT (:Other {id: 1})", "INSERT (:File {id: 3})"],
        );
        run_all(
            &session,
            &["MATCH (n:File {id: 3}) DELETE n", "INSERT (:File {id: 3})"],
        );
    }
}

// ---------------------------------------------------------------------------
// A common value: past the candidates an index lookup takes, the label is scanned
// ---------------------------------------------------------------------------

/// The most candidates a lookup reads, so the verdicts past it are worth pinning.
const CANDIDATE_CAP: i64 = 1_024;

/// `nodes` of `label` with `id` `value` and `part` 0, 1, 2 and on.
fn many_of_one_value(session: &Session, label: &str, nodes: i64, value: i64) {
    run_all(
        session,
        &[&format!(
            "UNWIND range(0, {}) AS part INSERT (:{label} {{id: {value}, part: part}})",
            nodes - 1
        )],
    );
}

/// A refused write under a rule reads the rule's label, not every label with the value: its cost
/// follows the small label however many nodes of another label hold the value. Measured against
/// the same database with none of them, so the machine's speed cancels out.
#[test]
fn a_common_value_on_another_label_costs_what_the_rules_label_costs() {
    const OTHERS: i64 = 30_000;
    const ATTEMPTS: u32 = 100;
    let attempts = |others: i64| {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        run_all(&session, &[RULE, "INSERT (:File {id: 7})"]);
        if others > 0 {
            many_of_one_value(&session, "Other", others, 7);
        }
        (0..3)
            .map(|_| {
                let start = std::time::Instant::now();
                for _ in 0..ATTEMPTS {
                    assert!(session.execute("INSERT (:File {id: 7})").is_err());
                }
                start.elapsed()
            })
            .min()
            .unwrap()
    };
    let alone = attempts(0);
    let beside = attempts(OTHERS);
    assert!(
        beside < alone * 10,
        "{beside:?} beside {OTHERS} nodes of another label, {alone:?} alone"
    );
}

/// Verdicts past the cap: on another label and on the rule's own label through a key's first
/// property, including a duplicate that sits past the cap in the index's list.
fn verdicts_past_the_cap(index: Index) -> Vec<String> {
    let nodes = CANDIDATE_CAP + 100;
    let mut verdicts = Vec::new();

    let db = ruled(index);
    let session = db.session();
    many_of_one_value(&session, "Other", nodes, 7);
    for (name, statement) in [
        ("first of the rule's label", "INSERT (:File {id: 7})"),
        ("a duplicate", "INSERT (:File {id: 7})"),
        ("another value", "INSERT (:File {id: 8})"),
        ("set onto the value", "MATCH (n:File {id: 8}) SET n.id = 7"),
        ("another label still", "INSERT (:Other {id: 7})"),
    ] {
        verdicts.push(format!("{name}: {}", outcome(session.execute(statement))));
    }
    verdicts.push(format!(
        "held: {}",
        count(&session, "MATCH (n:File) WHERE n.id = 7 RETURN count(n)")
    ));

    // A key of two properties: the index holds the first, so its list is long and only the second
    // tells them apart.
    let db = GrafeoDB::new_in_memory();
    if matches!(index, Index::MadeFirst) {
        db.create_property_index("id").unwrap();
    }
    let session = db.session();
    run_all(
        &session,
        &["CREATE CONSTRAINT c1 FOR (n:File) ON (n.id, n.part) NODE KEY"],
    );
    if matches!(index, Index::Dropped) {
        run_all(&session, &["DROP INDEX c1"]);
        assert!(!db.has_property_index("id"));
    }
    many_of_one_value(&session, "File", nodes, 7);
    let last = nodes - 1;
    for statement in [
        "INSERT (:File {id: 7, part: 0})".to_string(),
        format!("INSERT (:File {{id: 7, part: {}}})", CANDIDATE_CAP - 1),
        format!("INSERT (:File {{id: 7, part: {CANDIDATE_CAP}}})"),
        format!("INSERT (:File {{id: 7, part: {last}}})"),
        format!("INSERT (:File {{id: 7, part: {}}})", last + 1),
        format!(
            "MATCH (n:File {{id: 7, part: {}}}) SET n.part = {last}",
            last + 1
        ),
        format!("MATCH (n:File {{id: 7, part: {last}}}) DELETE n"),
        format!("INSERT (:File {{id: 7, part: {last}}})"),
    ] {
        verdicts.push(format!(
            "{statement}: {}",
            outcome(session.execute(&statement))
        ));
    }
    verdicts.push(format!(
        "held: {}",
        count(&session, "MATCH (n:File) WHERE n.id = 7 RETURN count(n)")
    ));
    verdicts
}

#[test]
fn past_the_cap_a_rule_refuses_alike_with_its_index_and_without() {
    let with = verdicts_past_the_cap(Index::TheRules);
    assert_same(with.clone(), verdicts_past_the_cap(Index::Dropped));
    assert_same(with.clone(), verdicts_past_the_cap(Index::MadeFirst));
    // What no reading can move.
    for (line, accepted) in [
        ("first of the rule's label", true),
        ("a duplicate", false),
        ("another value", true),
        ("set onto the value", false),
        ("another label still", true),
    ] {
        let verdict = with
            .iter()
            .find(|verdict| verdict.starts_with(line))
            .unwrap();
        assert_eq!(verdict.ends_with(": accepted"), accepted, "{verdict}");
    }
    let last = CANDIDATE_CAP + 99;
    for (part, accepted) in [
        (0, false),
        (CANDIDATE_CAP - 1, false),
        (CANDIDATE_CAP, false),
        (last, false),
        (last + 1, true),
    ] {
        let needle = format!("INSERT (:File {{id: 7, part: {part}}}): ");
        let verdict = with
            .iter()
            .find(|verdict| verdict.starts_with(&needle))
            .unwrap();
        assert_eq!(verdict.ends_with(": accepted"), accepted, "{verdict}");
    }
}

// ---------------------------------------------------------------------------
// Two rules on one key in one transaction
// ---------------------------------------------------------------------------

const RULE_A: &str = "CREATE CONSTRAINT ra FOR (n:A) ON (n.id) UNIQUE";
const RULE_B: &str = "CREATE CONSTRAINT rb FOR (n:B) ON (n.id) UNIQUE";

/// An index each, each rule's name, both rules: what must hold after the commit, a reopen and a
/// replay. Two rules on one key and two labels build two indexes.
fn assert_two_rules(db: &GrafeoDB) {
    let session = db.session();
    assert_eq!(constraint_count(&session), 2);
    assert!(db.has_property_index("id"));
    assert_eq!(
        shown(&session),
        rows(&[["ra", "Hash", "A", "id"], ["rb", "Hash", "B", "id"]])
    );
    run_all(&session, &["INSERT (:A {id: 1})", "INSERT (:B {id: 1})"]);
    assert!(
        session.execute("INSERT (:A {id: 1})").is_err(),
        "the rule on A"
    );
    assert!(
        session.execute("INSERT (:B {id: 1})").is_err(),
        "the rule on B"
    );
    assert!(reads_index(&session, "A", "id"));
    assert!(reads_index(&session, "B", "id"));
}

fn two_rules_in_a_transaction(db: &GrafeoDB) {
    let mut session = db.session();
    session.begin_transaction().unwrap();
    run_all(&session, &[RULE_A, RULE_B]);
    session.commit().unwrap();
}

#[test]
fn two_rules_on_one_key_and_two_labels_in_a_transaction_build_an_index_each() {
    let db = GrafeoDB::new_in_memory();
    two_rules_in_a_transaction(&db);
    assert_two_rules(&db);
}

/// On one label and key, whichever is written first builds the index and names it, in one
/// transaction or in two statements, as one way of writing them cannot list what the other does
/// not.
#[test]
fn the_first_of_two_rules_names_the_index_however_they_are_written() {
    const KEY_A: &str = "CREATE CONSTRAINT ka FOR (n:A) ON (n.id, n.x) NODE KEY";
    for in_a_transaction in [true, false] {
        for (first, second, name, label) in [(RULE_A, KEY_A, "ra", "A"), (KEY_A, RULE_A, "ka", "A")]
        {
            let db = GrafeoDB::new_in_memory();
            let mut session = db.session();
            if in_a_transaction {
                session.begin_transaction().unwrap();
            }
            run_all(&session, &[first, second]);
            if in_a_transaction {
                session.commit().unwrap();
            }
            assert_eq!(
                shown(&session),
                rows(&[[name, "Hash", label, "id"]]),
                "in a transaction: {in_a_transaction}, first: {first}"
            );
            assert_eq!(constraint_count(&session), 2);
        }
    }
}

/// A rule that built nothing has no index to name when it goes.
#[test]
fn dropping_the_rule_that_named_nothing_says_the_index_on_its_property_stays() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    run_all(
        &session,
        &[
            RULE_A,
            "CREATE CONSTRAINT rb FOR (n:A) ON (n.id, n.x) NODE KEY",
        ],
    );
    session.commit().unwrap();
    let status = session
        .execute("DROP CONSTRAINT rb")
        .unwrap()
        .status_message
        .expect("a status");
    assert!(
        status.contains("the index on property id stays"),
        "{status}"
    );
    assert!(!status.contains("DROP INDEX"), "{status}");
    assert_eq!(shown(&session), rows(&[["ra", "Hash", "A", "id"]]));
    let status = session
        .execute("DROP CONSTRAINT ra")
        .unwrap()
        .status_message
        .expect("a status");
    assert!(status.contains("index 'ra' stays"), "{status}");
    assert!(status.contains("DROP INDEX ra"), "{status}");
}

/// A second rule that reaches a graph the first did not builds there, and so takes its name.
#[test]
fn a_second_rule_that_builds_in_another_graph_takes_its_name() {
    for in_a_transaction in [true, false] {
        let db = GrafeoDB::new_in_memory();
        let mut session = db.session();
        run_all(
            &session,
            &[
                "CREATE NODE TYPE A (id INT64)",
                "CREATE NODE TYPE B (id INT64)",
                "CREATE GRAPH TYPE lists_a (NODE TYPE A)",
                "CREATE GRAPH TYPE lists_b (NODE TYPE B)",
                "CREATE GRAPH with_a TYPED lists_a",
                "CREATE GRAPH with_b TYPED lists_b",
            ],
        );
        if in_a_transaction {
            session.begin_transaction().unwrap();
        }
        run_all(&session, &[RULE_A, RULE_B]);
        if in_a_transaction {
            session.commit().unwrap();
        }
        assert_eq!(
            shown(&session),
            rows(&[["ra", "Hash", "A", "id"], ["rb", "Hash", "B", "id"]]),
            "in a transaction: {in_a_transaction}"
        );
        run_all(&session, &["USE GRAPH with_a", "INSERT (:A {id: 1})"]);
        assert!(reads_index(&session, "A", "id"), "{in_a_transaction}");
        run_all(&session, &["USE GRAPH with_b", "INSERT (:B {id: 1})"]);
        assert!(reads_index(&session, "B", "id"), "{in_a_transaction}");
    }
}

/// `DROP INDEX` then a rule on the key in one transaction: the rule builds, as the index is gone
/// by then, and it is there after the commit. A rollback leaves the index as it stood.
#[test]
fn a_rule_after_a_drop_of_the_index_in_one_transaction_builds_it() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    run_all(
        &session,
        &[
            "CREATE INDEX x FOR (n:File) ON (n.id)",
            "INSERT (:File {id: 1})",
        ],
    );

    session.begin_transaction().unwrap();
    run_all(&session, &["DROP INDEX x", RULE]);
    session.rollback().unwrap();
    assert_eq!(shown(&session), rows(&[["x", "Hash", "File", "id"]]));
    assert!(db.has_property_index("id"));
    assert_eq!(constraint_count(&session), 0);

    session.begin_transaction().unwrap();
    run_all(&session, &["DROP INDEX x", RULE]);
    session.commit().unwrap();
    assert_eq!(shown(&session), rows(&[["c1", "Hash", "File", "id"]]));
    assert!(db.has_property_index("id"));
    assert!(reads_index(&session, "File", "id"));
    assert!(session.execute("INSERT (:File {id: 1})").is_err());
    run_all(&session, &["INSERT (:File {id: 2})"]);
    assert_eq!(
        count(
            &session,
            "UNWIND [1, 2] AS x MATCH (n:File) WHERE n.id = x RETURN count(n)"
        ),
        2,
        "the index holds the rows from before and after"
    );
}

#[test]
fn a_rollback_of_two_rules_on_one_key_takes_everything() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    run_all(&session, &[RULE_A, RULE_B]);
    session.rollback().unwrap();
    assert_eq!(constraint_count(&session), 0);
    assert_eq!(shown(&session), rows(&[]));
    assert!(!db.has_property_index("id"));
    run_all(&session, &["INSERT (:A {id: 1})", "INSERT (:A {id: 1})"]);
}

#[test]
fn two_rules_on_one_key_survive_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("two.grafeo");
    let db = GrafeoDB::open(&path).unwrap();
    two_rules_in_a_transaction(&db);
    db.close().unwrap();
    let db = GrafeoDB::open(&path).unwrap();
    assert_two_rules(&db);
    db.close().unwrap();
    let db = GrafeoDB::open(&path).unwrap();
    assert_two_rules_held(&db);
    db.close().unwrap();
}

/// After `assert_two_rules` wrote its rows, with the index still there and holding them.
fn assert_two_rules_held(db: &GrafeoDB) {
    let session = db.session();
    assert_eq!(constraint_count(&session), 2);
    assert!(db.has_property_index("id"));
    assert!(session.execute("INSERT (:A {id: 1})").is_err());
    assert!(session.execute("INSERT (:B {id: 1})").is_err());
    run_all(&session, &["INSERT (:A {id: 2})"]);
}

const TWO_RULES_PATH: &str = "GRAFEO_UNIQUE_INDEX_TWO_RULES_PATH";
const TWO_RULES_POINT: &str = "GRAFEO_UNIQUE_INDEX_TWO_RULES_POINT";
const TWO_RULES_MARKER: &str = "GRAFEO_UNIQUE_INDEX_TWO_RULES_MARKER";

/// Commits two rules on one key up to a crash at the point given; with no point, the whole commit
/// and then an exit with no close.
#[test]
fn two_rules_crash_child() {
    use grafeo_common::testing::crash::{CrashResult, with_crash_at};
    use std::panic::AssertUnwindSafe;

    let Some(path) = std::env::var_os(TWO_RULES_PATH) else {
        return;
    };
    let db = GrafeoDB::open(PathBuf::from(path)).unwrap();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    run_all(&session, &[RULE_A, RULE_B]);
    if let (Some(point), Some(marker)) = (
        std::env::var_os(TWO_RULES_POINT),
        std::env::var_os(TWO_RULES_MARKER),
    ) {
        let point: u64 = point.to_str().unwrap().parse().unwrap();
        let outcome = with_crash_at(point, AssertUnwindSafe(|| session.commit().unwrap()));
        if matches!(outcome, CrashResult::Completed(())) {
            std::fs::write(PathBuf::from(marker), b"done").unwrap();
        }
    } else {
        session.commit().unwrap();
    }
    std::process::exit(0);
}

#[test]
fn a_crash_after_the_commit_of_two_rules_replays_both_indexes_and_names() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("two.grafeo");
    run_child(grafeo_common::child_test!("two_rules_crash_child").env(TWO_RULES_PATH, &path));
    let db = GrafeoDB::open(&path).unwrap();
    assert_two_rules(&db);
    db.close().unwrap();
}

#[cfg(feature = "testing-crash-injection")]
#[test]
fn a_crash_in_the_commit_of_two_rules_leaves_all_or_nothing() {
    for point in 1..=400_u64 {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sweep.grafeo");
        let marker = dir.path().join("completed");
        run_child(
            grafeo_common::child_test!("two_rules_crash_child")
                .env(TWO_RULES_PATH, &path)
                .env(TWO_RULES_POINT, point.to_string())
                .env(TWO_RULES_MARKER, &marker),
        );
        let db = GrafeoDB::open(&path).unwrap();
        let session = db.session();
        let rules = constraint_count(&session);
        assert!(
            rules == 0 || rules == 2,
            "a crash at {point}: {rules} rules"
        );
        assert_eq!(
            db.has_property_index("id"),
            rules == 2,
            "a crash at {point}"
        );
        drop(session);
        if rules == 2 {
            assert_two_rules(&db);
        }
        db.close().unwrap();
        if marker.exists() {
            assert_eq!(rules, 2, "the commit finished at {point}");
            assert!(point > 1, "no point crashed before the commit finished");
            return;
        }
    }
    panic!("the commit never finished within the points tried");
}

// ---------------------------------------------------------------------------
// The cap is the label's own count: reading more candidates than it has nodes costs more than
// reading the label
// ---------------------------------------------------------------------------

/// A value on fewer nodes than the old fixed cap, but on many more than the rule's label has: the
/// write reads the small label, not the candidates.
#[test]
fn a_value_held_past_a_small_label_costs_that_label() {
    const OTHERS: i64 = 900;
    const ATTEMPTS: u32 = 300;
    let attempts = |others: i64| {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        run_all(&session, &[RULE, "INSERT (:File {id: 7})"]);
        if others > 0 {
            many_of_one_value(&session, "Other", others, 7);
        }
        (0..3)
            .map(|_| {
                let start = std::time::Instant::now();
                for _ in 0..ATTEMPTS {
                    assert!(session.execute("INSERT (:File {id: 7})").is_err());
                }
                start.elapsed()
            })
            .min()
            .unwrap()
    };
    let alone = attempts(0);
    let beside = attempts(OTHERS);
    assert!(
        beside < alone * 5,
        "{beside:?} beside {OTHERS} nodes of another label, {alone:?} alone"
    );
}

/// Every side of the cap with a small label: the verdicts of a write, also of a transaction whose
/// view is not the latest count's, which can pick either route.
fn verdicts_around_a_small_count(index: Index) -> Vec<String> {
    let mut verdicts = Vec::new();
    for files in [0, 1, 3] {
        for others in [0, 1, 3, 5] {
            let shape = format!("{files} of the label, {others} of another");
            let fresh = || {
                let db = ruled(index);
                let session = db.session();
                for key in 0..files {
                    run_all(&session, &[&format!("INSERT (:File {{id: {}}})", 7 + key)]);
                }
                if others > 0 {
                    many_of_one_value(&session, "Other", others, 7);
                }
                db
            };

            let db = fresh();
            let session = db.session();
            let one = outcome(session.execute("INSERT (:File {id: 7})"));
            let two = outcome(session.execute("INSERT (:File {id: 7})"));
            let three = outcome(session.execute("INSERT (:File {id: 100})"));
            verdicts.push(format!("{shape}, plain: {one}; {two}; {three}"));

            // The transaction's own rows are not in the latest count.
            let db = fresh();
            let mut session = db.session();
            session.begin_transaction().unwrap();
            let one = outcome(session.execute("INSERT (:File {id: 50})"));
            let two = outcome(session.execute("INSERT (:File {id: 50})"));
            let three = outcome(session.execute("INSERT (:File {id: 7})"));
            let four = outcome(session.commit());
            let held = count(&db.session(), "MATCH (n:File) RETURN count(n)");
            verdicts.push(format!(
                "{shape}, own rows: {one}; {two}; {three}; {four}; {held}"
            ));

            // Its deletes are in the view and not yet in the count.
            let db = fresh();
            let mut session = db.session();
            session.begin_transaction().unwrap();
            let one = outcome(session.execute("MATCH (n:File {id: 7}) DELETE n"));
            let two = outcome(session.execute("INSERT (:File {id: 7})"));
            let three = outcome(session.execute("INSERT (:File {id: 7})"));
            let four = outcome(session.commit());
            let held = count(&db.session(), "MATCH (n:File) RETURN count(n)");
            verdicts.push(format!(
                "{shape}, own deletes: {one}; {two}; {three}; {four}; {held}"
            ));
        }
    }
    verdicts
}

#[test]
fn around_a_small_count_a_rule_refuses_alike_with_its_index_and_without() {
    let with = verdicts_around_a_small_count(Index::TheRules);
    assert_same(with.clone(), verdicts_around_a_small_count(Index::Dropped));
    assert_same(
        with.clone(),
        verdicts_around_a_small_count(Index::MadeFirst),
    );
    // A label's first rows, written by the transaction itself, with the value on another label: the
    // latest count is 0, and the second write of the value is still refused.
    let verdict = with
        .iter()
        .find(|verdict| verdict.starts_with("0 of the label, 5 of another, own rows:"))
        .unwrap();
    let parts: Vec<&str> = verdict.split(": ").nth(1).unwrap().split("; ").collect();
    assert_eq!(parts[0], "accepted", "{verdict}");
    assert_ne!(verdict.matches("accepted").count(), 4, "{verdict}");
    assert!(verdict.ends_with("; 2"), "{verdict}");
}
