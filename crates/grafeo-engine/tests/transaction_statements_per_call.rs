//! A transaction statement through a call that has no lasting session is refused.
//!
//! Tests first, from the user's decision for 0.6.2: `GrafeoDB::execute` and its siblings run each
//! call in a session of their own, so `START TRANSACTION` opened a transaction that ended with
//! the call, the next write committed at once, and a `ROLLBACK` found nothing to undo. The
//! statement is refused, with an error that points to `begin_transaction` or a session, and
//! `COMMIT` and `ROLLBACK` say the same. In a session they keep working.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main transaction_statements_per_call::
//! ```

#![cfg(feature = "gql")]

use std::collections::HashMap;

use grafeo_common::utils::error::{Error, Result};
use grafeo_common::utils::gqlstatus::GqlStatus;
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::QueryResult;

const STATEMENTS: [&str; 7] = [
    "START TRANSACTION",
    "START TRANSACTION READ ONLY",
    "START TRANSACTION READ WRITE",
    "start transaction",
    "COMMIT",
    "ROLLBACK",
    "commit",
];

/// Every per-call way into the database.
fn per_call(db: &GrafeoDB, statement: &str) -> Vec<(&'static str, Result<QueryResult>)> {
    vec![
        ("execute", db.execute(statement)),
        (
            "execute_with_params",
            db.execute_with_params(statement, HashMap::new()),
        ),
        (
            "execute_language",
            db.execute_language(statement, "gql", None),
        ),
    ]
}

fn says_where_to_go(error: &Error) -> bool {
    let text = error.to_string();
    text.contains("begin_transaction") || text.contains("session")
}

#[test]
fn a_transaction_statement_on_a_call_is_refused_and_says_where_to_go() {
    let db = GrafeoDB::new_in_memory();
    for statement in STATEMENTS {
        for (way, answer) in per_call(&db, statement) {
            let error = answer.expect_err(&format!("{way}: {statement} was accepted"));
            assert!(
                says_where_to_go(&error),
                "{way}: {statement}: the error does not point anywhere: {error}"
            );
            assert!(
                GqlStatus::from(&error).as_str().starts_with("25"),
                "{way}: {statement}: {} is no transaction state status: {error}",
                GqlStatus::from(&error).as_str()
            );
        }
    }
}

/// The status the gtests pin for `COMMIT` and `ROLLBACK` with no transaction stays.
#[test]
fn commit_and_rollback_on_a_call_keep_their_status() {
    let db = GrafeoDB::new_in_memory();
    for statement in ["COMMIT", "ROLLBACK"] {
        let error = db.execute(statement).expect_err(statement);
        assert_eq!(
            GqlStatus::from(&error).as_str(),
            "25000",
            "{statement}: {error}"
        );
        assert!(
            error.to_string().contains("No active transaction"),
            "{statement}: {error}"
        );
    }
}

/// The refused call writes nothing and holds nothing open: a write after it commits as any other,
/// and is there after a reopen. Refusal is no change to the writes around it.
#[test]
fn the_refused_call_leaves_nothing_behind() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("per-call.grafeo");
    let db = GrafeoDB::open(&path).unwrap();
    assert!(db.execute("START TRANSACTION").is_err());
    db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    assert!(db.execute("ROLLBACK").is_err());
    let names = |db: &GrafeoDB| {
        db.execute("MATCH (p:Person) RETURN p.name")
            .unwrap()
            .row_count()
    };
    assert_eq!(
        names(&db),
        1,
        "the write is not undone by the refused rollback"
    );
    assert!(db.execute("COMMIT").is_err());
    assert_eq!(names(&db), 1);
    db.close().unwrap();
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(names(&db), 1, "after a reopen");
    db.close().unwrap();
}

/// A session's transaction goes on working: its statements hold across calls.
#[test]
fn a_sessions_transaction_statements_keep_working() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let names = |query: &str| db.session().execute(query).unwrap().row_count();

    session.execute("START TRANSACTION").unwrap();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    assert_eq!(
        names("MATCH (p:Person) RETURN p.name"),
        0,
        "before the commit"
    );
    session.execute("ROLLBACK").unwrap();
    assert_eq!(names("MATCH (p:Person) RETURN p.name"), 0, "rolled back");

    session.execute("START TRANSACTION").unwrap();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    session.execute("COMMIT").unwrap();
    assert_eq!(names("MATCH (p:Person) RETURN p.name"), 1, "committed");

    session.execute("START TRANSACTION READ ONLY").unwrap();
    assert!(session.execute("INSERT (:Person {name: 'Mia'})").is_err());
    session.execute("ROLLBACK").unwrap();
    assert_eq!(names("MATCH (p:Person) RETURN p.name"), 1);

    // With no transaction open a session says so, as it always has, and has nothing to point at.
    let error = session.execute("COMMIT").expect_err("nothing to commit");
    assert_eq!(GqlStatus::from(&error).as_str(), "25000", "{error}");
}

/// A statement that only looks like one is not refused: a label, a property, a string.
#[test]
fn other_statements_on_a_call_are_untouched() {
    let db = GrafeoDB::new_in_memory();
    for query in [
        "INSERT (:COMMIT {ROLLBACK: 'START TRANSACTION'})",
        "MATCH (n:COMMIT) RETURN n.ROLLBACK",
        "RETURN 'COMMIT' AS word",
        "RETURN 'START TRANSACTION' AS word",
    ] {
        db.execute(query)
            .unwrap_or_else(|error| panic!("{query}: {error}"));
    }
}
