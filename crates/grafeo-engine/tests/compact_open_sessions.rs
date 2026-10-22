//! Sessions opened before `compact()` go on as before.
//!
//! A session opened before `compact()` reads and writes the same store as one
//! opened after it: its reads see every write made since, and its writes
//! reach the database and every other session, its named graphs included,
//! also after a reopen. A transaction open across `compact()` keeps its
//! changes to itself until it commits.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test compact_open_sessions
//! ```

#![cfg(all(feature = "lpg", feature = "gql"))]

use grafeo_common::types::Value;
use grafeo_engine::{GrafeoDB, Session};

/// The `name`s of the `Person` nodes, sorted.
fn names(result: &grafeo_engine::database::QueryResult) -> Vec<String> {
    result
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::String(name) => name.to_string(),
            other => panic!("not a name: {other:?}"),
        })
        .collect()
}

/// The people the database sees.
fn people(db: &GrafeoDB) -> Vec<String> {
    names(
        &db.execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
            .unwrap(),
    )
}

/// The people `session` sees.
fn people_in(session: &Session) -> Vec<String> {
    names(
        &session
            .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
            .unwrap(),
    )
}

/// The names, as owned strings.
fn list(names: &[&str]) -> Vec<String> {
    names.iter().map(ToString::to_string).collect()
}

/// A database holding Alix.
fn with_alix() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    db
}

/// A session opened before `compact()` writes the compacted store, with a
/// statement and with a direct call: the database sees both, and the
/// session sees what others wrote after `compact()`.
#[test]
fn a_session_opened_before_compact_writes_the_compacted_store() {
    let mut db = with_alix();
    let session = db.session();
    db.compact().unwrap();

    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    session
        .create_node_with_props(&["Person"], [("name", Value::from("Vincent"))])
        .unwrap();
    assert_eq!(
        people(&db),
        list(&["Alix", "Gus", "Vincent"]),
        "the database sees the session's writes"
    );

    db.execute("INSERT (:Person {name: 'Mia'})").unwrap();
    assert_eq!(
        people_in(&session),
        list(&["Alix", "Gus", "Mia", "Vincent"]),
        "the session sees the writes made after compact()"
    );
    db.compact().unwrap();
    assert_eq!(
        people_in(&session),
        list(&["Alix", "Gus", "Mia", "Vincent"]),
        "and follows a merge of the overlay too"
    );
}

/// A transaction that a session opened before `compact()` begins after it
/// commits into the compacted store, and its rollback leaves nothing.
#[test]
fn a_transaction_of_a_session_opened_before_compact_commits_in_the_compacted_store() {
    let mut db = with_alix();
    let mut session = db.session();
    db.compact().unwrap();

    session.begin_transaction().unwrap();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    session
        .execute("MATCH (a:Person {name: 'Alix'}) SET a.city = 'Amsterdam'")
        .unwrap();
    assert_eq!(people(&db), list(&["Alix"]), "not before the commit");
    session.commit().unwrap();

    session.begin_transaction().unwrap();
    session.execute("INSERT (:Person {name: 'Jules'})").unwrap();
    session.rollback().unwrap();

    assert_eq!(people(&db), list(&["Alix", "Gus"]));
    let city = db
        .execute("MATCH (a:Person {name: 'Alix'}) RETURN a.city")
        .unwrap();
    assert_eq!(city.rows(), [vec![Value::from("Amsterdam")]]);
}

/// `compact()` while a transaction is open leaves the transaction alone:
/// its changes stay its own until it commits, and then everyone sees them,
/// also after another `compact()`.
#[test]
fn compact_leaves_an_open_transaction_alone() {
    let mut db = with_alix();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();

    db.compact().unwrap();
    assert_eq!(
        people(&db),
        list(&["Alix"]),
        "Gus is the open transaction's own"
    );
    session.execute("INSERT (:Person {name: 'Mia'})").unwrap();
    session.commit().unwrap();
    assert_eq!(people(&db), list(&["Alix", "Gus", "Mia"]));

    db.compact().unwrap();
    assert_eq!(
        people(&db),
        list(&["Alix", "Gus", "Mia"]),
        "compacted with the transaction's changes"
    );
}

/// A session opened before `compact()` keeps its named graph: `compact()`
/// moves the named graphs into the overlay, and the session writes and reads
/// them there.
#[test]
fn a_session_opened_before_compact_keeps_its_named_graph() {
    let mut db = with_alix();
    db.execute("CREATE GRAPH paris").unwrap();
    let session = db.session();
    session.execute("USE GRAPH paris").unwrap();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    db.compact().unwrap();

    session.execute("INSERT (:Person {name: 'Mia'})").unwrap();
    assert_eq!(people_in(&session), list(&["Gus", "Mia"]), "the session");
    let other = db.session();
    other.execute("USE GRAPH paris").unwrap();
    assert_eq!(people_in(&other), list(&["Gus", "Mia"]), "another session");
    assert_eq!(people(&db), list(&["Alix"]), "the default graph");
}

/// The writes of a session opened before `compact()` are in the file after
/// a close and a reopen.
#[cfg(all(feature = "wal", feature = "grafeo-file"))]
#[test]
fn a_session_opened_before_compact_writes_survive_a_reopen() {
    use grafeo_engine::Config;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("people.grafeo");
    {
        let mut db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        let session = db.session();
        db.compact().unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        session
            .create_node_with_props(&["Person"], [("name", Value::from("Vincent"))])
            .unwrap();
        db.close().unwrap();
    }
    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    assert_eq!(people(&db), list(&["Alix", "Gus", "Vincent"]));
    db.close().unwrap();
}
