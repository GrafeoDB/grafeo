//! `CREATE OR REPLACE GRAPH` (ISO/IEC 39075 12.4): an existing graph of the name is dropped in the
//! same change as the new one is created.
//!
//! Tests first, from the decided behaviour:
//! - an absent graph is created; a present one is replaced by an empty graph, its type binding
//!   gone with it, or by `AS COPY OF` another graph;
//! - a replace by a copy of itself is refused, and the graph stays;
//! - an open transaction's changes in the graph refuse the replace, as they refuse `DROP GRAPH`;
//! - a session using the graph keeps using it under its name;
//! - in an explicit transaction it behaves as `DROP GRAPH` there: applied at the commit, undone by
//!   a rollback;
//! - after a WAL replay, and after a crash at any point of the statement, the database holds the
//!   old graph or the new one, never neither.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main create_or_replace_graph::
//! ```

#![cfg(all(feature = "gql", feature = "grafeo-file", feature = "wal"))]

use std::path::PathBuf;

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;

/// The `Person` nodes of `graph`, by name, sorted.
fn people(session: &Session, graph: &str) -> Vec<String> {
    session.execute(&format!("USE GRAPH {graph}")).unwrap();
    let mut names: Vec<String> = session
        .execute("MATCH (p:Person) RETURN p.name")
        .unwrap()
        .rows()
        .iter()
        .filter_map(|row| match &row[0] {
            Value::String(name) => Some(name.to_string()),
            _ => None,
        })
        .collect();
    names.sort();
    names
}

/// The names `SHOW GRAPHS` lists.
fn graphs(db: &GrafeoDB) -> Vec<String> {
    db.execute("SHOW GRAPHS")
        .unwrap()
        .rows()
        .iter()
        .filter_map(|row| match &row[0] {
            Value::String(name) => Some(name.to_string()),
            _ => None,
        })
        .collect()
}

/// A database whose graph `g` holds Vincent and Jules.
fn with_g() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("CREATE GRAPH g").unwrap();
    session.execute("USE GRAPH g").unwrap();
    session
        .execute("INSERT (:Person {name: 'Vincent'}), (:Person {name: 'Jules'})")
        .unwrap();
    db
}

#[test]
fn an_absent_graph_is_created() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE OR REPLACE GRAPH g").unwrap();
    assert!(graphs(&db).contains(&"g".to_string()));
    assert_eq!(people(&db.session(), "g"), Vec::<String>::new());
}

#[test]
fn a_present_graph_is_replaced_by_an_empty_one() {
    let db = with_g();
    db.execute("CREATE OR REPLACE GRAPH g").unwrap();
    assert!(graphs(&db).contains(&"g".to_string()));
    assert_eq!(
        people(&db.session(), "g"),
        Vec::<String>::new(),
        "the old data is gone"
    );
    // The default graph is not touched.
    assert_eq!(people(&db.session(), "default"), Vec::<String>::new());
}

#[test]
fn a_plain_create_of_a_present_graph_still_fails() {
    let db = with_g();
    let error = db.execute("CREATE GRAPH g").unwrap_err().to_string();
    assert!(error.contains("already exists"), "{error}");
    assert_eq!(people(&db.session(), "g"), ["Jules", "Vincent"]);
}

#[test]
fn the_old_graph_type_goes_with_it() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for statement in [
        "CREATE NODE TYPE Person (name STRING)",
        "CREATE GRAPH TYPE people (NODE TYPE Person)",
        "CREATE GRAPH g TYPED people",
    ] {
        session.execute(statement).unwrap();
    }
    session.execute("USE GRAPH g").unwrap();
    assert!(
        session.execute("INSERT (:Robot {name: 'Fiona'})").is_err(),
        "the typed graph refuses a label its type lacks"
    );
    session.execute("CREATE OR REPLACE GRAPH g").unwrap();
    session.execute("USE GRAPH g").unwrap();
    session
        .execute("INSERT (:Robot {name: 'Fiona'})")
        .expect("the new graph has no type");
}

#[test]
fn a_replace_takes_the_type_its_own_statement_names() {
    let db = with_g();
    let session = db.session();
    session
        .execute("CREATE NODE TYPE Person (name STRING)")
        .unwrap();
    session
        .execute("CREATE GRAPH TYPE people (NODE TYPE Person)")
        .unwrap();
    session
        .execute("CREATE OR REPLACE GRAPH g TYPED people")
        .unwrap();
    session.execute("USE GRAPH g").unwrap();
    assert!(session.execute("INSERT (:Robot {name: 'Fiona'})").is_err());
}

#[test]
fn a_replace_as_a_copy_of_another_graph_holds_its_data() {
    let db = with_g();
    let session = db.session();
    session.execute("CREATE GRAPH h").unwrap();
    session.execute("USE GRAPH h").unwrap();
    session.execute("INSERT (:Person {name: 'Mia'})").unwrap();
    session
        .execute("CREATE OR REPLACE GRAPH g AS COPY OF h")
        .unwrap();
    assert_eq!(people(&session, "g"), ["Mia"]);
    assert_eq!(people(&session, "h"), ["Mia"], "the source stays");
}

#[test]
fn a_replace_by_a_copy_of_itself_is_refused() {
    let db = with_g();
    for statement in [
        "CREATE OR REPLACE GRAPH g AS COPY OF g",
        "CREATE OR REPLACE GRAPH g LIKE g",
    ] {
        let error = db.execute(statement).unwrap_err().to_string();
        assert!(error.contains("copy of itself"), "{statement}: {error}");
        assert_eq!(
            people(&db.session(), "g"),
            ["Jules", "Vincent"],
            "{statement}"
        );
    }
}

#[test]
fn open_changes_of_another_transaction_refuse_the_replace() {
    let db = with_g();
    let writer = db.session();
    writer.execute("USE GRAPH g").unwrap();
    writer.execute("START TRANSACTION").unwrap();
    writer.execute("INSERT (:Person {name: 'Butch'})").unwrap();

    let error = db
        .execute("CREATE OR REPLACE GRAPH g")
        .unwrap_err()
        .to_string();
    let drop_error = db.execute("DROP GRAPH g").unwrap_err().to_string();
    assert!(error.contains("open transaction"), "{error}");
    assert!(drop_error.contains("open transaction"), "{drop_error}");

    writer.execute("COMMIT").unwrap();
    assert_eq!(people(&db.session(), "g"), ["Butch", "Jules", "Vincent"]);
    db.execute("CREATE OR REPLACE GRAPH g")
        .expect("once the transaction ended, the replace goes through");
    assert_eq!(people(&db.session(), "g"), Vec::<String>::new());
}

#[test]
fn a_session_using_the_graph_keeps_using_it() {
    let db = with_g();
    let user = db.session();
    user.execute("USE GRAPH g").unwrap();
    db.execute("CREATE OR REPLACE GRAPH g").unwrap();
    user.execute("INSERT (:Person {name: 'Django'})").unwrap();
    assert_eq!(people(&db.session(), "g"), ["Django"]);
    assert_eq!(
        people(&db.session(), "default"),
        Vec::<String>::new(),
        "the write did not go to the default graph"
    );

    // The session that replaced the graph, too.
    let replacer = db.session();
    replacer.execute("USE GRAPH g").unwrap();
    replacer.execute("CREATE OR REPLACE GRAPH g").unwrap();
    replacer
        .execute("INSERT (:Person {name: 'Shosanna'})")
        .unwrap();
    assert_eq!(people(&db.session(), "g"), ["Shosanna"]);
}

#[test]
fn in_a_transaction_it_applies_at_the_commit() {
    let db = with_g();
    let session = db.session();
    session.execute("START TRANSACTION").unwrap();
    session.execute("CREATE OR REPLACE GRAPH g").unwrap();
    assert_eq!(
        people(&db.session(), "g"),
        ["Jules", "Vincent"],
        "another session sees the old graph until the commit"
    );
    session.execute("COMMIT").unwrap();
    assert_eq!(people(&db.session(), "g"), Vec::<String>::new());
}

#[test]
fn in_a_transaction_a_rollback_undoes_it_as_it_undoes_a_drop() {
    for statement in ["CREATE OR REPLACE GRAPH g", "DROP GRAPH g"] {
        let db = with_g();
        let session = db.session();
        session.execute("START TRANSACTION").unwrap();
        session.execute(statement).unwrap();
        session.execute("ROLLBACK").unwrap();
        assert_eq!(
            people(&db.session(), "g"),
            ["Jules", "Vincent"],
            "{statement}: the rollback keeps the old graph"
        );
    }
}

const REPLAY_PATH_VAR: &str = "GRAFEO_CREATE_OR_REPLACE_GRAPH_REPLAY_PATH";

/// Child of [`a_replace_survives_a_wal_replay`]: replaces `g` and exits with the database alive,
/// so the replace is only in the WAL.
#[test]
fn replay_child() {
    let Some(path) = std::env::var_os(REPLAY_PATH_VAR) else {
        return;
    };
    let db = GrafeoDB::open(PathBuf::from(path)).unwrap();
    let session = db.session();
    session.execute("CREATE OR REPLACE GRAPH g").unwrap();
    session.execute("USE GRAPH g").unwrap();
    session.execute("INSERT (:Person {name: 'Hans'})").unwrap();
    std::process::exit(0);
}

#[test]
fn a_replace_survives_a_wal_replay() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("replay.grafeo");
    {
        let db = GrafeoDB::open(&path).unwrap();
        let session = db.session();
        session.execute("CREATE GRAPH g").unwrap();
        session.execute("USE GRAPH g").unwrap();
        session
            .execute("INSERT (:Person {name: 'Vincent'})")
            .unwrap();
        db.close().unwrap();
    }
    let status = grafeo_common::testing::child_process::run(
        grafeo_common::child_test!("replay_child").env(REPLAY_PATH_VAR, &path),
    )
    .unwrap();
    assert!(status.success(), "the child process failed");
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(people(&db.session(), "g"), ["Hans"]);
}

/// A crash at each point of `CREATE OR REPLACE GRAPH g` (feature `testing-crash-injection`): the
/// reopened database holds the old `g` (Vincent) or the new, empty one, never no `g`. The replace
/// is one WAL group, so today it has a single crash point before that group is synced: the sweep
/// shows the old `g` after it, and `a_replace_survives_a_wal_replay` covers the new one.
#[cfg(feature = "testing-crash-injection")]
mod crash {
    use std::panic::AssertUnwindSafe;

    use grafeo_common::testing::child_process;
    use grafeo_common::testing::crash::{CrashResult, with_crash_at};

    use super::*;

    const POINT_VAR: &str = "GRAFEO_CREATE_OR_REPLACE_GRAPH_CRASH_POINT";
    const PATH_VAR: &str = "GRAFEO_CREATE_OR_REPLACE_GRAPH_CRASH_PATH";
    const CRASHED: i32 = 3;
    const REPLACED: i32 = 4;

    /// The child: replaces `g` with a crash at `point`, and exits with the database alive whether
    /// it crashed or not.
    #[test]
    fn replace_child() {
        let (Ok(point), Some(path)) = (std::env::var(POINT_VAR), std::env::var_os(PATH_VAR)) else {
            return;
        };
        let db = GrafeoDB::open(PathBuf::from(path)).unwrap();
        let borrowed = AssertUnwindSafe(&db);
        match with_crash_at(point.parse().unwrap(), move || {
            borrowed.execute("CREATE OR REPLACE GRAPH g")
        }) {
            CrashResult::Completed(Ok(_)) => std::process::exit(REPLACED),
            CrashResult::Completed(Err(error)) => panic!("the replace failed: {error}"),
            _ => std::process::exit(CRASHED),
        }
    }

    #[test]
    fn a_crash_during_a_replace_leaves_the_old_graph_or_the_new_one() {
        let mut completed = false;
        let mut crashed = 0_u32;
        for point in 1..=256_u64 {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("crash.grafeo");
            {
                let db = GrafeoDB::open(&path).unwrap();
                let session = db.session();
                session.execute("CREATE GRAPH g").unwrap();
                session.execute("USE GRAPH g").unwrap();
                session
                    .execute("INSERT (:Person {name: 'Vincent'})")
                    .unwrap();
                db.close().unwrap();
            }
            let status = child_process::run(
                grafeo_common::child_test!("replace_child")
                    .env(POINT_VAR, point.to_string())
                    .env(PATH_VAR, &path),
            )
            .unwrap();
            let db = GrafeoDB::open(&path).unwrap();
            assert!(
                graphs(&db).contains(&"g".to_string()),
                "point {point}: g is there"
            );
            let left = people(&db.session(), "g");
            assert!(
                left == ["Vincent"] || left.is_empty(),
                "point {point}: the old g or the new one, not {left:?}"
            );
            match status.code() {
                Some(REPLACED) => {
                    assert_eq!(
                        left,
                        Vec::<String>::new(),
                        "point {point}: the replace completed"
                    );
                    completed = true;
                    break;
                }
                Some(CRASHED) => crashed += 1,
                other => panic!("point {point}: the child exited with {other:?}"),
            }
        }
        assert!(
            completed,
            "the sweep reached the statement's end within 256 points"
        );
        assert!(
            crashed >= 1,
            "the replace has at least one crash point, or the sweep proves nothing"
        );
    }
}
