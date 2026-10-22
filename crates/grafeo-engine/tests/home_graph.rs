//! HOME_SCHEMA and HOME_GRAPH name the schema and graph a session started in
//! (ISO/IEC 39075 Section 4.7.3): USE and SESSION SET move the current graph
//! and schema, never the home ones, and SESSION RESET brings them back home
//! (Section 7.2).
//!
//! ```bash
//! cargo test -p grafeo-engine --test home_graph
//! ```

use grafeo_common::types::Value;
use grafeo_engine::{GrafeoDB, Session};

const HOME_AND_CURRENT: &str = "RETURN HOME_SCHEMA AS home_schema, HOME_GRAPH AS home_graph, \
     CURRENT_SCHEMA AS current_schema, CURRENT_GRAPH AS current_graph";

/// `[home_schema, home_graph, current_schema, current_graph]` of one row.
fn names(result: grafeo_engine::database::QueryResult) -> Vec<String> {
    result.rows()[0]
        .iter()
        .map(|value| match value {
            Value::String(s) => s.to_string(),
            other => panic!("expected a name, got {other:?}"),
        })
        .collect()
}

fn in_session(session: &Session) -> Vec<String> {
    names(session.execute(HOME_AND_CURRENT).unwrap())
}

fn in_database(db: &GrafeoDB) -> Vec<String> {
    names(db.execute(HOME_AND_CURRENT).unwrap())
}

fn db() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE GRAPH reports").unwrap();
    db.execute("CREATE SCHEMA staging").unwrap();
    db
}

#[test]
fn a_new_session_is_home_in_the_default_graph() {
    let db = db();
    let session = db.session();
    assert_eq!(
        in_session(&session),
        ["default", "default", "default", "default"]
    );
}

#[test]
fn use_and_session_set_leave_the_home_alone() {
    let db = db();
    let session = db.session();

    session.execute("USE GRAPH reports").unwrap();
    assert_eq!(
        in_session(&session),
        ["default", "default", "default", "reports"]
    );

    // Graph names resolve in the current schema: leave `reports` first
    session.execute("SESSION SET GRAPH default").unwrap();
    session.execute("SESSION SET SCHEMA staging").unwrap();
    assert_eq!(
        in_session(&session),
        ["default", "default", "staging", "default"]
    );

    session.execute("SESSION RESET").unwrap();
    assert_eq!(
        in_session(&session),
        ["default", "default", "default", "default"]
    );
}

/// One-shot calls share the database's graph context like one long session:
/// a `USE GRAPH` carries over to the next call, the home stays the default.
#[test]
fn one_shot_calls_keep_the_default_home_across_use_graph() {
    let db = db();
    db.execute("USE GRAPH reports").unwrap();
    assert_eq!(
        in_database(&db),
        ["default", "default", "default", "reports"]
    );

    db.execute("SESSION SET GRAPH default").unwrap();
    db.execute("SESSION SET SCHEMA staging").unwrap();
    assert_eq!(
        in_database(&db),
        ["default", "default", "staging", "default"]
    );
}

/// A session opened while the database's context is on a graph starts there,
/// so that graph is its home, also after it moves on.
#[test]
fn a_session_opened_on_the_database_graph_is_home_there() {
    let db = db();
    db.execute("USE GRAPH reports").unwrap();

    let session = db.session();
    assert_eq!(
        in_session(&session),
        ["default", "reports", "default", "reports"]
    );

    session.execute("SESSION SET GRAPH default").unwrap();
    assert_eq!(
        in_session(&session),
        ["default", "reports", "default", "default"]
    );
}

#[test]
fn a_handle_session_is_home_in_its_graph() {
    let db = db();
    let reports = db.graph("reports").unwrap().session().unwrap();
    assert_eq!(
        in_session(&reports),
        ["default", "reports", "default", "reports"]
    );

    let staging = db
        .graph_in(Some("staging"), "default")
        .unwrap()
        .session()
        .unwrap();
    assert_eq!(
        in_session(&staging),
        ["staging", "default", "staging", "default"]
    );

    // Opening a handle session does not move the database's own home
    assert_eq!(
        in_database(&db),
        ["default", "default", "default", "default"]
    );
}

#[test]
fn session_reset_returns_a_handle_session_to_its_graph() {
    let db = db();
    let reports = db.graph("reports").unwrap().session().unwrap();
    reports.execute("SESSION SET GRAPH default").unwrap();
    reports.execute("SESSION SET SCHEMA staging").unwrap();
    assert_eq!(
        in_session(&reports),
        ["default", "reports", "staging", "default"]
    );

    reports.execute("SESSION RESET").unwrap();
    assert_eq!(
        in_session(&reports),
        ["default", "reports", "default", "reports"]
    );

    // The reset reads the home graph: its data, not the default graph's
    reports.execute("INSERT (:Report {id: 'r1'})").unwrap();
    let count = db.execute("MATCH (n) RETURN count(n) AS c").unwrap().rows()[0][0].clone();
    assert_eq!(count, Value::Int64(0));
}

#[test]
fn session_reset_graph_and_schema_return_separately() {
    let db = db();
    let staging = db
        .graph_in(Some("staging"), "default")
        .unwrap()
        .session()
        .unwrap();
    db.execute("CREATE SCHEMA archive").unwrap();
    staging.execute("SESSION SET SCHEMA archive").unwrap();
    assert_eq!(
        in_session(&staging),
        ["staging", "default", "archive", "default"]
    );

    staging.execute("SESSION RESET SCHEMA").unwrap();
    assert_eq!(
        in_session(&staging),
        ["staging", "default", "staging", "default"]
    );

    let reports = db.graph("reports").unwrap().session().unwrap();
    reports.execute("SESSION SET GRAPH default").unwrap();
    reports.execute("SESSION RESET GRAPH").unwrap();
    assert_eq!(
        in_session(&reports),
        ["default", "reports", "default", "reports"]
    );
}

/// A session opened after a db-level USE is home in that graph: SESSION
/// RESET brings it back there, while the one-shot context resets to default.
#[test]
fn session_reset_after_a_database_use_graph() {
    let db = db();
    db.execute("USE GRAPH reports").unwrap();

    let session = db.session();
    session.execute("SESSION SET GRAPH default").unwrap();
    session.execute("SESSION RESET").unwrap();
    assert_eq!(
        in_session(&session),
        ["default", "reports", "default", "reports"]
    );

    db.execute("SESSION RESET").unwrap();
    assert_eq!(
        in_database(&db),
        ["default", "default", "default", "default"]
    );
}
