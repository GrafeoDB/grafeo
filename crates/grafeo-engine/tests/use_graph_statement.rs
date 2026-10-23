//! `USE g <statement>` (ISO/IEC 39075 use graph clause, web#55): a GQL
//! statement focused on a graph reads and writes that graph, inside a
//! transaction as outside, and leaves the session's current graph as it was,
//! also when it fails. The gtests in `lpg/gql/23.1_named_graphs.gtest` cover
//! the statement forms on every runner; these cover transactions, failures,
//! concurrent sessions, the database-level calls, streaming, graph types and
//! a reopen.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test use_graph_statement
//! ```

#![cfg(feature = "gql")]

use std::sync::Arc;

use grafeo_common::types::Value;
use grafeo_common::utils::GqlStatus;
use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;

/// The values of the first column of `query`, in order.
fn column(session: &Session, query: &str) -> Vec<Value> {
    session
        .execute(query)
        .unwrap()
        .rows()
        .iter()
        .map(|row| row[0].clone())
        .collect()
}

fn names(names: &[&str]) -> Vec<Value> {
    names.iter().map(|name| Value::from(*name)).collect()
}

/// The people in `graph`, sorted by name, read with a focused statement.
fn people_in(session: &Session, graph: &str) -> Vec<Value> {
    column(
        session,
        &format!("USE {graph} MATCH (p:Person) RETURN p.name ORDER BY p.name"),
    )
}

/// A database with the empty graphs alpha and beta.
fn two_graphs() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE GRAPH alpha").unwrap();
    db.execute("CREATE GRAPH beta").unwrap();
    db
}

#[test]
fn focused_writes_commit_with_their_transaction() {
    let db = two_graphs();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("USE alpha INSERT (:Person {name: 'Alix'})")
        .unwrap();
    session
        .execute("USE beta INSERT (:Person {name: 'Gus'})")
        .unwrap();
    session
        .execute("INSERT (:Person {name: 'Vincent'})")
        .unwrap();
    // The transaction reads its own focused writes.
    assert_eq!(people_in(&session, "alpha"), names(&["Alix"]));
    session.commit().unwrap();

    let reader = db.session();
    assert_eq!(people_in(&reader, "alpha"), names(&["Alix"]));
    assert_eq!(people_in(&reader, "beta"), names(&["Gus"]));
    assert_eq!(
        column(&reader, "MATCH (p:Person) RETURN p.name"),
        names(&["Vincent"])
    );
    assert_eq!(session.current_graph(), None);
}

#[test]
fn focused_writes_roll_back_with_their_transaction() {
    let db = two_graphs();
    let mut session = db.session();
    session.execute("USE GRAPH beta").unwrap();
    session.begin_transaction().unwrap();
    session
        .execute("USE alpha INSERT (:Person {name: 'Alix'})")
        .unwrap();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    session.rollback().unwrap();

    assert_eq!(people_in(&session, "alpha"), names(&[]));
    assert_eq!(people_in(&session, "beta"), names(&[]));
    assert_eq!(session.current_graph().as_deref(), Some("beta"));
}

#[test]
fn a_failed_focused_statement_writes_nothing_and_keeps_the_session_graph() {
    let db = two_graphs();
    let session = db.session();
    session.execute("USE GRAPH beta").unwrap();

    let missing = session
        .execute("USE nowhere INSERT (:Person {name: 'Alix'})")
        .unwrap_err();
    assert_eq!(GqlStatus::from(&missing).to_string(), "42002", "{missing}");
    assert!(
        missing
            .to_string()
            .contains("Graph 'nowhere' does not exist"),
        "{missing}"
    );
    assert_eq!(session.current_graph().as_deref(), Some("beta"));

    // The insert runs, then the RETURN fails: the statement leaves nothing.
    let failed = session
        .execute("USE alpha INSERT (p:Person {name: 'Alix'}) RETURN 1 / 0 AS boom")
        .unwrap_err();
    assert_eq!(GqlStatus::from(&failed).to_string(), "22012", "{failed}");
    assert_eq!(people_in(&session, "alpha"), names(&[]));
    assert_eq!(session.current_graph().as_deref(), Some("beta"));
    // The next statement reads the session graph again.
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    assert_eq!(people_in(&session, "beta"), names(&["Gus"]));
}

#[test]
fn different_graphs_in_one_statement_are_not_supported_yet() {
    let db = two_graphs();
    let session = db.session();
    for query in [
        "USE alpha MATCH (p) RETURN p.name AS name UNION USE beta MATCH (p) RETURN p.name AS name",
        "USE alpha MATCH (p) RETURN p.name AS name NEXT USE beta RETURN name",
        "USE alpha MATCH (p) USE beta MATCH (c) RETURN p, c",
        "MATCH (p) USE alpha MATCH (c) RETURN p, c",
        "USE alpha MATCH (p) CALL { USE beta MATCH (c) RETURN c } RETURN p, c",
    ] {
        let error = session.execute(query).unwrap_err().to_string();
        assert!(error.contains("not supported yet"), "{query}: {error}");
        assert!(error.contains("GRAFEO-Q004"), "{query}: {error}");
    }
}

#[test]
fn a_subquery_without_use_reads_the_focused_graph() {
    let db = two_graphs();
    let session = db.session();
    session
        .execute("USE alpha INSERT (:Person {name: 'Alix'}), (:City {name: 'Paris'})")
        .unwrap();
    session.execute("INSERT (:City {name: 'Berlin'})").unwrap();
    let rows = session
        .execute(
            "USE alpha MATCH (p:Person) CALL { MATCH (c:City) RETURN c.name AS city } \
             RETURN p.name, city",
        )
        .unwrap();
    assert_eq!(
        rows.rows(),
        &[vec![Value::from("Alix"), Value::from("Paris")]]
    );
}

#[test]
fn the_database_calls_keep_their_current_graph() {
    let db = two_graphs();
    db.execute("USE GRAPH beta").unwrap();
    db.execute("USE alpha INSERT (:Person {name: 'Alix'})")
        .unwrap();
    assert_eq!(db.current_graph().as_deref(), Some("beta"));
    db.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    let session = db.session();
    assert_eq!(people_in(&session, "alpha"), names(&["Alix"]));
    assert_eq!(people_in(&session, "beta"), names(&["Gus"]));
}

#[test]
fn a_cached_plan_keeps_its_graph() {
    let db = two_graphs();
    let session = db.session();
    session
        .execute("USE alpha INSERT (:Person {name: 'Alix'})")
        .unwrap();
    let query = "USE alpha MATCH (p:Person) RETURN p.name";
    // The same text from two session graphs, and with parameters.
    assert_eq!(column(&session, query), names(&["Alix"]));
    session.execute("USE GRAPH beta").unwrap();
    assert_eq!(column(&session, query), names(&["Alix"]));
    let params = std::collections::HashMap::from([("name".to_string(), Value::from("Alix"))]);
    for _ in 0..2 {
        let result = session
            .execute_with_params(
                "USE alpha MATCH (p:Person {name: $name}) RETURN p.name",
                params.clone(),
            )
            .unwrap();
        assert_eq!(result.rows(), &[vec![Value::from("Alix")]]);
    }
    assert_eq!(session.current_graph().as_deref(), Some("beta"));
}

#[test]
fn a_stream_reads_the_focused_graph() {
    let db = two_graphs();
    let session = db.session();
    session
        .execute("USE alpha INSERT (:Person {name: 'Alix'})")
        .unwrap();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    let streamed = session
        .execute_streaming("USE alpha MATCH (p:Person) RETURN p.name")
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(streamed.rows(), &[vec![Value::from("Alix")]]);
    assert_eq!(session.current_graph(), None);
}

#[test]
fn a_closed_graph_type_checks_the_focused_writes() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE NODE TYPE Person (name STRING NOT NULL)")
        .unwrap();
    db.execute("CREATE GRAPH TYPE people (NODE TYPE Person)")
        .unwrap();
    db.execute("CREATE GRAPH typed TYPED people").unwrap();
    let session = db.session();
    session
        .execute("USE typed INSERT (:Person {name: 'Alix'})")
        .unwrap();
    let refused = session
        .execute("USE typed INSERT (:Animal {name: 'Rex'})")
        .unwrap_err();
    assert!(refused.to_string().contains("Animal"), "{refused}");
    // The default graph has no type: the same insert works there.
    session.execute("INSERT (:Animal {name: 'Rex'})").unwrap();
    assert_eq!(people_in(&session, "typed"), names(&["Alix"]));
}

#[test]
fn concurrent_sessions_focus_on_their_own_graphs() {
    const WRITES: i64 = 200;

    let db = Arc::new(two_graphs());
    let writers: Vec<_> = ["alpha", "beta"]
        .into_iter()
        .map(|graph| {
            let db = Arc::clone(&db);
            std::thread::spawn(move || {
                let session = db.session();
                for i in 0..WRITES {
                    session
                        .execute(&format!("USE {graph} INSERT (:Visit {{i: {i}}})"))
                        .unwrap();
                }
                session.current_graph()
            })
        })
        .collect();
    for writer in writers {
        assert_eq!(writer.join().unwrap(), None, "a writer's session graph");
    }
    let session = db.session();
    for graph in ["alpha", "beta"] {
        // Every write of its own writer, each once: i runs over 0..WRITES.
        assert_eq!(
            session
                .execute(&format!(
                    "USE {graph} MATCH (v:Visit) RETURN count(v), sum(v.i)"
                ))
                .unwrap()
                .rows(),
            [[
                Value::Int64(WRITES),
                Value::Int64(WRITES * (WRITES - 1) / 2)
            ]],
            "{graph}"
        );
    }
    assert_eq!(
        column(&session, "MATCH (v:Visit) RETURN count(v)"),
        [Value::Int64(0)],
        "the default graph"
    );
}

#[cfg(all(feature = "wal", feature = "grafeo-file"))]
#[test]
fn focused_writes_are_there_after_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("floria.grafeo");
    let db = GrafeoDB::open(&path).unwrap();
    db.execute("CREATE GRAPH floria").unwrap();
    db.execute("USE floria INSERT (:Person {name: 'Alix'})")
        .unwrap();
    db.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    db.close().unwrap();

    for read_only in [false, true] {
        let db = if read_only {
            GrafeoDB::open_read_only(&path).unwrap()
        } else {
            GrafeoDB::open(&path).unwrap()
        };
        let session = db.session();
        assert_eq!(
            people_in(&session, "floria"),
            names(&["Alix"]),
            "{read_only}"
        );
        assert_eq!(
            column(&session, "MATCH (p:Person) RETURN p.name"),
            names(&["Gus"]),
            "{read_only}"
        );
        drop(session);
        db.close().unwrap();
    }
}
