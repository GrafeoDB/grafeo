//! A standalone type that a graph type lists stays in force where it was (#764): the parts a spec
//! case cannot show. The statement-level rule is in `tests/spec/lpg/gql/listed_standalone_types.gtest`
//! and `graph_type_scope.gtest`.
//!
//! - **The mark:** the catalog marks a standalone type a graph type lists. The mark holds after a
//!   checkpoint and a reopen, and after a WAL replay of the statements that made it.
//! - **A crash during `CREATE GRAPH TYPE`** leaves the old state or the new one, and in both the
//!   type is in force in the default graph.
//! - **A 0.6.2 file** in which a graph type took a standalone type over keeps that meaning.
//!   `validate()` names the rows of the default graph that would break the type, with the steps to
//!   restore it (warning `LISTED_TYPE_ROWS`, next to `GRAPH_TYPE_NOT_BOUND`). After those steps the
//!   type is in force again, marked.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main listed_standalone_types::
//! ```

#![cfg(all(feature = "gql", feature = "grafeo-file", feature = "wal"))]

use std::path::{Path, PathBuf};

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

const TYPES: [&str; 4] = [
    "CREATE NODE TYPE A (h FLOAT64)",
    "CREATE EDGE TYPE ROAD (km INT64)",
    "CREATE GRAPH TYPE t (NODE TYPE A)",
    "CREATE GRAPH TYPE u (EDGE TYPE ROAD)",
];

fn run_all(db: &GrafeoDB, statements: &[&str]) {
    for statement in statements {
        db.execute(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }
}

/// `statement` fails, its message holding `part`.
fn refused(db: &GrafeoDB, statement: &str, part: &str) {
    let error = db
        .execute(statement)
        .expect_err(&format!("{statement} must be refused"));
    assert!(error.to_string().contains(part), "{statement}: {error}");
}

/// `A` and `ROAD` are in force in the default graph, and neither can be dropped while a graph type
/// lists it.
fn assert_in_force(db: &GrafeoDB, stage: &str) {
    refused(db, "INSERT (:A {k: 9, h: 'x'})", "expects FLOAT64");
    refused(
        db,
        "INSERT (:Town)-[:ROAD {km: 'far'}]->(:Town)",
        "expects INT64",
    );
    refused(db, "DROP NODE TYPE A", "graph type 't'");
    refused(db, "DROP EDGE TYPE ROAD", "graph type 'u'");
    db.execute("INSERT (:A {k: 9, h: 9.5e0})")
        .unwrap_or_else(|error| panic!("{stage}: a value of the type: {error}"));
}

/// The mark holds after a checkpoint: closed and reopened, the listed standalone types are in force
/// in the default graph; after `DROP GRAPH TYPE` and another reopen, still.
#[test]
fn the_mark_holds_after_a_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("types.grafeo");
    let db = GrafeoDB::open(&path).unwrap();
    run_all(&db, &TYPES);
    db.close().unwrap();

    let db = GrafeoDB::open(&path).unwrap();
    assert_in_force(&db, "reopened");
    run_all(&db, &["DROP GRAPH TYPE t", "DROP GRAPH TYPE u"]);
    db.close().unwrap();

    let db = GrafeoDB::open(&path).unwrap();
    refused(&db, "INSERT (:A {k: 1, h: 'x'})", "expects FLOAT64");
    refused(
        &db,
        "INSERT (:Town)-[:ROAD {km: 'far'}]->(:Town)",
        "expects INT64",
    );
    run_all(&db, &["DROP NODE TYPE A", "DROP EDGE TYPE ROAD"]);
}

const REPLAY_PATH_VAR: &str = "GRAFEO_LISTED_STANDALONE_REPLAY_PATH";

/// Child of [`the_mark_holds_after_a_wal_replay`]: makes the types and exits with the database
/// alive, so they are only in the WAL.
#[test]
fn replay_child() {
    let Some(path) = std::env::var_os(REPLAY_PATH_VAR) else {
        return;
    };
    let db = GrafeoDB::open(PathBuf::from(path)).unwrap();
    run_all(&db, &TYPES);
    std::process::exit(0);
}

/// The mark holds after a WAL replay: the statements that made it, replayed at open, leave the
/// types in force as the live statements did.
#[test]
fn the_mark_holds_after_a_wal_replay() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("replay.grafeo");
    let status = grafeo_common::testing::child_process::run(
        grafeo_common::child_test!("replay_child").env(REPLAY_PATH_VAR, &path),
    )
    .unwrap();
    assert!(status.success(), "the child process failed");
    let db = GrafeoDB::open(&path).unwrap();
    assert_in_force(&db, "replayed");
}

/// A 0.6.2 file in which `t` took `A` over (`scripts/listed_standalone_type_fixture.py`, the
/// reporter's steps): opened, it keeps its meaning (`A` governs only `t`'s graphs, so the default
/// graph's rows stay as written and a wrong-typed write there is accepted). `validate()` names the
/// two rows that would break `A`, with the steps to restore it; after those steps `A` is in force
/// in the default graph, marked.
#[test]
fn a_taken_over_type_in_a_0_6_2_file_keeps_its_meaning_and_is_restored() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/listed_standalone_type/0.6.2.grafeo");
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("0.6.2.grafeo");
    std::fs::copy(&fixture, &path).unwrap();
    let db = GrafeoDB::open(&path).unwrap();
    let rows = |db: &GrafeoDB| {
        db.execute("MATCH (n:A) RETURN n.k, n.h ORDER BY n.k")
            .unwrap()
            .rows()
            .to_vec()
    };
    assert_eq!(
        rows(&db),
        [
            vec![Value::Int64(1), Value::from("y")],
            vec![Value::Int64(3), Value::from("z")],
            vec![Value::Int64(4), Value::Float64(2.5)],
        ]
    );

    // The take-over stands in the old file.
    db.execute("INSERT (:A {k: 5, h: 'w'})").unwrap();
    db.execute("MATCH (n:A {k: 5}) DELETE n").unwrap();

    let breaking: Vec<i64> = [1, 3]
        .iter()
        .map(|k| {
            let rows = db
                .execute(&format!("MATCH (n:A {{k: {k}}}) RETURN id(n)"))
                .unwrap()
                .rows()
                .to_vec();
            match rows[0][0] {
                Value::Int64(id) => id,
                ref other => panic!("an id: {other:?}"),
            }
        })
        .collect();
    let result = db.validate().unwrap();
    assert!(result.errors.is_empty(), "{:?}", result.errors);
    let lapsed: Vec<_> = result
        .warnings
        .iter()
        .filter(|warning| warning.code == "LISTED_TYPE_ROWS")
        .collect();
    assert_eq!(
        lapsed.len(),
        1,
        "one warning for A in the default graph: {lapsed:?}"
    );
    let message = &lapsed[0].message;
    for part in [
        "'A'",
        "default",
        "2",
        "DROP GRAPH TYPE t",
        "CREATE NODE TYPE A",
    ] {
        assert!(message.contains(part), "{part} in {message}");
    }
    for id in &breaking {
        assert!(
            message.contains(&id.to_string()),
            "the id {id} in {message}"
        );
    }
    assert!(
        result
            .warnings
            .iter()
            .any(|warning| warning.code == "GRAPH_TYPE_NOT_BOUND"),
        "the not-bound warning stays"
    );

    // The restore steps: fix the rows, drop the graph type (which drops A, only t listed it),
    // declare A on its own, list it again.
    run_all(
        &db,
        &[
            "MATCH (n:A) WHERE n.k IN [1, 3] SET n.h = CAST(n.k AS FLOAT64)",
            "DROP GRAPH TYPE t",
            "CREATE NODE TYPE A (h FLOAT64)",
            "CREATE GRAPH TYPE t (NODE TYPE A)",
        ],
    );
    refused(&db, "INSERT (:A {k: 6, h: 'x'})", "expects FLOAT64");
    refused(&db, "DROP NODE TYPE A", "graph type 't'");
    let result = db.validate().unwrap();
    assert!(
        !result
            .warnings
            .iter()
            .any(|warning| warning.code == "LISTED_TYPE_ROWS"),
        "nothing breaks A now: {:?}",
        result.warnings
    );
    db.close().unwrap();
    let db = GrafeoDB::open(&path).unwrap();
    refused(&db, "INSERT (:A {k: 6, h: 'x'})", "expects FLOAT64");
}

/// A crash at each point of `CREATE GRAPH TYPE t (NODE TYPE A)` (feature
/// `testing-crash-injection`): the reopened database holds the old state (no `t`) or the new one
/// (`t` listing `A`, marked), and in both `A` is in force in the default graph.
#[cfg(feature = "testing-crash-injection")]
mod crash {
    use std::panic::AssertUnwindSafe;

    use grafeo_common::testing::child_process;
    use grafeo_common::testing::crash::{CrashResult, with_crash_at};

    use super::*;

    const POINT_VAR: &str = "GRAFEO_LISTED_STANDALONE_CRASH_POINT";
    const PATH_VAR: &str = "GRAFEO_LISTED_STANDALONE_CRASH_PATH";
    const CRASHED: i32 = 3;
    const LISTED: i32 = 4;

    /// The child: lists `A` in a new graph type with a crash at `point`, and exits with the
    /// database alive whether it crashed or not.
    #[test]
    fn create_graph_type_child() {
        let (Ok(point), Some(path)) = (std::env::var(POINT_VAR), std::env::var_os(PATH_VAR)) else {
            return;
        };
        let db = GrafeoDB::open(PathBuf::from(path)).unwrap();
        let borrowed = AssertUnwindSafe(&db);
        match with_crash_at(point.parse().unwrap(), move || {
            borrowed.execute("CREATE GRAPH TYPE t (NODE TYPE A)")
        }) {
            CrashResult::Completed(Ok(_)) => std::process::exit(LISTED),
            CrashResult::Completed(Err(error)) => panic!("CREATE GRAPH TYPE failed: {error}"),
            _ => std::process::exit(CRASHED),
        }
    }

    #[test]
    fn a_crash_during_create_graph_type_keeps_the_type_in_force() {
        let mut completed = false;
        for point in 1..=256_u64 {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("crash.grafeo");
            let db = GrafeoDB::open(&path).unwrap();
            run_all(&db, &["CREATE NODE TYPE A (h FLOAT64)"]);
            db.close().unwrap();

            let status = child_process::run(
                grafeo_common::child_test!("create_graph_type_child")
                    .env(POINT_VAR, point.to_string())
                    .env(PATH_VAR, &path),
            )
            .unwrap();
            let db = GrafeoDB::open(&path).unwrap();
            refused(&db, "INSERT (:A {k: 1, h: 'x'})", "expects FLOAT64");
            let listed = db
                .execute("SHOW GRAPH TYPES")
                .unwrap()
                .rows()
                .iter()
                .any(|row| row.iter().any(|value| value == &Value::from("t")));
            if listed {
                refused(&db, "DROP NODE TYPE A", "graph type 't'");
            }
            match status.code() {
                Some(LISTED) => {
                    assert!(
                        listed,
                        "point {point}: the statement completed, so t is there"
                    );
                    completed = true;
                    break;
                }
                Some(CRASHED) => {}
                other => panic!("point {point}: the child exited with {other:?}"),
            }
        }
        assert!(
            completed,
            "the sweep reached the statement's end within 256 points"
        );
    }
}
