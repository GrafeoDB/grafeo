//! Triples written with SPARQL to a database file survive a reopen in every
//! build with the triple store: after `close()`, which checkpoints them into
//! the file; after a crash before any checkpoint, from the sidecar WAL; after
//! a crash that follows a checkpoint, from the file and the WAL; and after a
//! crash at any point of `close()` (with `testing-crash-injection`). So do
//! whole-graph operations (`COPY`, `MOVE`, `ADD`, `CLEAR ALL`), and nothing
//! comes back of what a rollback to a savepoint undid (#414). A crash
//! runs in a child process that exits without `close()` and without
//! destructors, and each test checks the files a crash leaves (the sidecar
//! WAL still there) before it reopens them.
//!
//! An engine built with the triple store and without the LPG store loaded
//! nothing back, from the file or the WAL (#544): the `triple-store` feature
//! now enables `lpg`. CI runs these tests with the features of the feature
//! matrix's `rdf` engine, which name no `lpg`.
//!
//! What a reopen shows is compared with a database in memory that ran the same
//! statements: the reopened database is the one that was written.
//!
//! ```bash
//! cargo test -p grafeo-engine --no-default-features --features \
//!     gql,sparql,graphql,triple-store,shacl,wal,grafeo-file,spill,mmap,regex,testing-crash-injection \
//!     --test rdf_persistence
//! cargo test -p grafeo-engine --all-features --test rdf_persistence
//! ```

#![cfg(all(
    feature = "sparql",
    feature = "triple-store",
    feature = "wal",
    feature = "grafeo-file"
))]

use std::path::{Path, PathBuf};

use grafeo_common::testing::child_process;
use grafeo_engine::{Config, GrafeoDB};

/// Tells [`crash_child`] which scenario to run.
const SCENARIO_VAR: &str = "GRAFEO_RDF_PERSISTENCE_SCENARIO";
/// Tells [`crash_child`] where its database is.
const PATH_VAR: &str = "GRAFEO_RDF_PERSISTENCE_PATH";

/// The exit code of a child whose `close()` ran to its end: the crash point
/// lies past the last one `close()` reaches.
#[cfg(feature = "testing-crash-injection")]
const CLOSE_COMPLETED: i32 = 3;

/// Three triples in the default graph and one in the named graph
/// `http://ex.org/amsterdam`.
const FIRST: &str = r#"INSERT DATA {
    <http://ex.org/alix> <http://ex.org/knows> <http://ex.org/gus> .
    <http://ex.org/gus> <http://ex.org/knows> <http://ex.org/vincent> .
    <http://ex.org/alix> <http://ex.org/name> "Alix" .
    GRAPH <http://ex.org/amsterdam> {
        <http://ex.org/mia> <http://ex.org/livesIn> <http://ex.org/amsterdam> .
    }
}"#;

/// What [`THEN_CHANGE`] deletes after [`FIRST`].
const DELETE: &str = r"DELETE DATA {
    <http://ex.org/gus> <http://ex.org/knows> <http://ex.org/vincent> .
}";

/// What [`THEN_CHANGE`] inserts after [`DELETE`]: one triple in each graph.
const INSERT_MORE: &str = r#"INSERT DATA {
    <http://ex.org/jules> <http://ex.org/knows> <http://ex.org/mia> .
    GRAPH <http://ex.org/amsterdam> {
        <http://ex.org/jules> <http://ex.org/name> "Jules" .
    }
}"#;

/// The statements of the first write.
const FIRST_WRITE: &[&str] = &[FIRST];

/// Whole-graph operations after [`FIRST`]: Amsterdam copied to Berlin, the
/// default graph added to Paris, Amsterdam moved to Prague.
const GRAPH_OPS: &[&str] = &[
    "COPY <http://ex.org/amsterdam> TO <http://ex.org/berlin>",
    "ADD DEFAULT TO <http://ex.org/paris>",
    "MOVE <http://ex.org/amsterdam> TO <http://ex.org/prague>",
];

/// `CLEAR ALL` after [`FIRST`], then one triple.
const CLEAR_ALL: &[&str] = &[
    "CLEAR ALL",
    "INSERT DATA { <http://ex.org/butch> <http://ex.org/knows> <http://ex.org/mia> }",
];

/// Graph operations onto the graph they read, which change nothing.
const ONTO_ITSELF: &[&str] = &[
    "MOVE <http://ex.org/amsterdam> TO <http://ex.org/amsterdam>",
    "COPY DEFAULT TO DEFAULT",
    "ADD <http://ex.org/amsterdam> TO <http://ex.org/amsterdam>",
    "MOVE DEFAULT TO DEFAULT",
];

/// The graph operations each crash scenario runs after [`FIRST`], and the
/// triples of the default graph and of the named graphs they leave.
const GRAPH_SCENARIOS: &[(&str, &[&str], (usize, usize))] = &[
    ("graph_ops", GRAPH_OPS, (3, 5)),
    ("clear_all", CLEAR_ALL, (1, 0)),
    ("onto_itself", ONTO_ITSELF, (3, 1)),
    ("clear_named", &["CLEAR NAMED"], (3, 0)),
    ("drop_named", &["DROP NAMED"], (3, 0)),
    ("clear_default", &["CLEAR DEFAULT"], (0, 1)),
    ("drop_default", &["DROP DEFAULT"], (0, 1)),
];

/// The graph operations of the crash scenario `name`, if it is one.
fn graph_scenario(name: &str) -> Option<&'static [&'static str]> {
    GRAPH_SCENARIOS
        .iter()
        .find(|(scenario, _, _)| *scenario == name)
        .map(|(_, writes, _)| *writes)
}

/// The insert of a transaction before its savepoint, which it commits.
const KEPT: &str =
    "INSERT DATA { <http://ex.org/vincent> <http://ex.org/knows> <http://ex.org/jules> }";

/// The insert of a transaction after its savepoint, which a rollback to the
/// savepoint undoes.
const UNDONE: &str = r#"INSERT DATA {
    <http://ex.org/hans> <http://ex.org/knows> <http://ex.org/shosanna> .
    GRAPH <http://ex.org/berlin> { <http://ex.org/hans> <http://ex.org/name> "Hans" . }
}"#;

/// The statements of the second write, after the first one was checkpointed.
const THEN_CHANGE: &[&str] = &[DELETE, INSERT_MORE];

fn open(path: &Path) -> GrafeoDB {
    GrafeoDB::with_config(Config::persistent(path))
        .unwrap_or_else(|error| panic!("open {}: {error}", path.display()))
}

fn open_read_only(path: &Path) -> GrafeoDB {
    GrafeoDB::open_read_only(path)
        .unwrap_or_else(|error| panic!("read-only open {}: {error}", path.display()))
}

fn sidecar_wal(path: &Path) -> PathBuf {
    let mut sidecar = path.as_os_str().to_owned();
    sidecar.push(".wal");
    PathBuf::from(sidecar)
}

/// Runs `statements` on `db`.
fn run(db: &GrafeoDB, statements: &[&str]) {
    for statement in statements {
        db.execute_sparql(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }
}

/// Every triple of `db`: those of the default graph as `[s, p, o]`, those of
/// a named graph as `[g, s, p, o]`, sorted.
fn triples(db: &GrafeoDB) -> Vec<Vec<String>> {
    let mut found: Vec<Vec<String>> = Vec::new();
    for query in [
        "SELECT ?s ?p ?o WHERE { ?s ?p ?o }",
        "SELECT ?g ?s ?p ?o WHERE { GRAPH ?g { ?s ?p ?o } }",
    ] {
        let result = db
            .execute_sparql(query)
            .unwrap_or_else(|error| panic!("{query}: {error}"));
        found.extend(
            result
                .rows()
                .iter()
                .map(|row| row.iter().map(|value| format!("{value:?}")).collect()),
        );
    }
    found.sort();
    found
}

/// The triples of a database in memory after `writes`, each a list of
/// statements run in turn.
fn expected(writes: &[&[&str]]) -> Vec<Vec<String>> {
    let db = GrafeoDB::new_in_memory();
    for statements in writes {
        run(&db, statements);
    }
    triples(&db)
}

/// The number of triples in the default graph (3 elements) and in named
/// graphs (4 elements) of `triples`.
fn counts(triples: &[Vec<String>]) -> (usize, usize) {
    (
        triples.iter().filter(|triple| triple.len() == 3).count(),
        triples.iter().filter(|triple| triple.len() == 4).count(),
    )
}

/// The triples after [`FIRST_WRITE`]: three in the default graph, one in a
/// named graph.
fn after_first_write() -> Vec<Vec<String>> {
    let triples = expected(&[FIRST_WRITE]);
    assert_eq!(counts(&triples), (3, 1), "the first write: {triples:?}");
    triples
}

/// The triples after [`FIRST_WRITE`] and [`THEN_CHANGE`]: Gus no longer knows
/// Vincent, Jules knows Mia, and the named graph holds Jules's name too.
fn after_the_change() -> Vec<Vec<String>> {
    let triples = expected(&[FIRST_WRITE, THEN_CHANGE]);
    assert_eq!(counts(&triples), (3, 2), "after the change: {triples:?}");
    triples
}

/// Runs `scenario` of [`crash_child`] in a child process on the database at
/// `path`, and returns its exit code.
fn in_child(scenario: &str, path: &Path) -> i32 {
    let status = child_process::run(
        std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "crash_child", "--nocapture"])
            .env(SCENARIO_VAR, scenario)
            .env(PATH_VAR, path),
    )
    .unwrap();
    status
        .code()
        .unwrap_or_else(|| panic!("scenario {scenario}: the child was killed: {status}"))
}

/// Runs `scenario` in a child process that crashes: it exits without
/// `close()`, and without dropping the database.
fn crash_after(scenario: &str, path: &Path) {
    assert_eq!(in_child(scenario, path), 0, "scenario {scenario} failed");
}

/// Child-process entry for [`in_child`]; a no-op when run directly.
#[test]
fn crash_child() {
    let Ok(scenario) = std::env::var(SCENARIO_VAR) else {
        return;
    };
    let path = PathBuf::from(std::env::var_os(PATH_VAR).unwrap());
    let db = open(&path);
    run(&db, FIRST_WRITE);
    match scenario.as_str() {
        // A crash before any checkpoint: the WAL alone holds the triples.
        "first_write" => {}
        // A crash after whole-graph operations, which the WAL alone holds.
        graph if graph_scenario(graph).is_some() => run(&db, graph_scenario(graph).unwrap()),
        // A crash after a transaction that rolled back to a savepoint and
        // committed: neither the store nor the WAL holds what it undid.
        "savepoint" => {
            let mut session = db.session();
            session.begin_transaction().unwrap();
            session.execute_sparql(KEPT).unwrap();
            session.savepoint("before_hans").unwrap();
            session.execute_sparql(UNDONE).unwrap();
            session.rollback_to_savepoint("before_hans").unwrap();
            session.commit().unwrap();
            assert_eq!(
                triples(&db),
                expected(&[FIRST_WRITE, &[KEPT]]),
                "in memory, before the crash"
            );
        }
        // A crash after a checkpoint: the file holds the first write, the WAL
        // the change.
        "checkpoint_then_change" => {
            db.wal_checkpoint().unwrap();
            run(&db, THEN_CHANGE);
        }
        // The same after the checkpoint of `close()` and a reopen.
        "close_then_change" => {
            db.close().unwrap();
            drop(db);
            let db = open(&path);
            run(&db, THEN_CHANGE);
            // Crash: no close(), no destructors.
            std::process::exit(0);
        }
        // A crash at point N of the `close()` after the first write
        // (`close_first:N`) or after a change that follows a checkpoint
        // (`close_change:N`). The database stays outside the closure, so the
        // injected panic does not drop (and so close) it.
        #[cfg(feature = "testing-crash-injection")]
        other if other.starts_with("close_first:") || other.starts_with("close_change:") => {
            use grafeo_common::testing::crash::{CrashResult, with_crash_at};

            let (phase, point) = other.split_once(':').unwrap();
            let point: u64 = point.parse().unwrap();
            if phase == "close_change" {
                db.wal_checkpoint().unwrap();
                run(&db, THEN_CHANGE);
            }
            let borrowed = std::panic::AssertUnwindSafe(&db);
            let outcome = with_crash_at(point, move || borrowed.close());
            if let CrashResult::Completed(closed) = outcome {
                closed.unwrap();
                std::process::exit(CLOSE_COMPLETED);
            }
            // Crash: no destructors.
            std::process::exit(0);
        }
        other => panic!("unknown scenario {other}"),
    }
    // Crash: no close(), no destructors.
    std::process::exit(0);
}

/// Asserts that the crash left the database file and its sidecar WAL, which
/// only `close()` removes.
fn assert_crashed(path: &Path, what: &str) {
    assert!(path.is_file(), "{what}: the database file is there");
    assert!(
        sidecar_wal(path).is_dir(),
        "{what}: the crash left the sidecar WAL {}",
        sidecar_wal(path).display()
    );
}

/// Asserts that a read-only open shows `want`, and leaves the sidecar WAL
/// (a read-only open writes nothing); then that a read-write open shows it,
/// and a reopen after its `close()`, which checkpoints what it replayed.
fn assert_reopens_with(path: &Path, want: &[Vec<String>], what: &str) {
    let crashed = sidecar_wal(path).is_dir();
    let db = open_read_only(path);
    assert_eq!(triples(&db), want, "{what}: a read-only open");
    db.close().unwrap();
    drop(db);
    assert_eq!(
        sidecar_wal(path).is_dir(),
        crashed,
        "{what}: a read-only open leaves the sidecar WAL as it is"
    );

    let db = open(path);
    assert_eq!(triples(&db), want, "{what}: a read-write open");
    db.close().unwrap();
    drop(db);
    assert!(
        !sidecar_wal(path).exists(),
        "{what}: close() removes the sidecar WAL"
    );

    let db = open(path);
    assert_eq!(
        triples(&db),
        want,
        "{what}: a reopen after the close, from the file alone"
    );
    db.close().unwrap();
}

/// `close()` checkpoints the triples into the file, and a reopen reads them.
#[test]
fn triples_survive_a_close_and_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("amsterdam.grafeo");
    let db = open(&path);
    run(&db, FIRST_WRITE);
    db.close().unwrap();
    drop(db);
    assert!(
        !sidecar_wal(&path).exists(),
        "close() removes the sidecar WAL: the file alone holds the triples"
    );

    assert_reopens_with(&path, &after_first_write(), "after close()");
}

/// A change after a reopen reaches the file at the next `close()`, with the
/// triples the file held before.
#[test]
fn a_change_after_a_reopen_keeps_the_earlier_triples() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("berlin.grafeo");
    let db = open(&path);
    run(&db, FIRST_WRITE);
    db.close().unwrap();
    drop(db);

    let db = open(&path);
    run(&db, THEN_CHANGE);
    db.close().unwrap();
    drop(db);

    assert_reopens_with(&path, &after_the_change(), "after two closes");
}

/// A process that exits before any checkpoint leaves the triples in the
/// sidecar WAL alone: the reopen replays them.
#[test]
fn triples_survive_a_crash_before_any_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("paris.grafeo");
    crash_after("first_write", &path);
    assert_crashed(&path, "a crash before any checkpoint");

    assert_reopens_with(
        &path,
        &after_first_write(),
        "after a crash before any checkpoint",
    );
}

/// Whole-graph operations come back from the WAL as they ran: a copy, an
/// add and a move, a `CLEAR ALL`, which clears every graph (#414), `CLEAR
/// NAMED` and `DROP NAMED`, which keep the default graph, `CLEAR DEFAULT` and
/// `DROP DEFAULT`, which keep the named graphs, and operations onto the graph
/// they read, which change nothing.
#[test]
fn whole_graph_operations_survive_a_crash() {
    for &(scenario, writes, shape) in GRAPH_SCENARIOS {
        let want = expected(&[FIRST_WRITE, writes]);
        assert_eq!(counts(&want), shape, "{scenario}: {want:?}");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("amsterdam.grafeo");
        crash_after(scenario, &path);
        assert_crashed(&path, scenario);

        assert_reopens_with(&path, &want, scenario);
    }
}

/// A rollback to a savepoint drops the RDF changes after it from the
/// transaction, in memory and in the WAL: after a crash, the reopen shows
/// what the transaction kept, and nothing of what it undid (#414).
#[test]
fn a_rolled_back_savepoint_drops_rdf_changes_in_memory_and_the_wal() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("berlin.grafeo");
    crash_after("savepoint", &path);
    assert_crashed(&path, "a crash after the commit");

    let want = expected(&[FIRST_WRITE, &[KEPT]]);
    assert_eq!(counts(&want), (4, 1), "{want:?}");
    assert_reopens_with(&path, &want, "after a rollback to a savepoint");
}

/// A process that exits after a checkpoint (`wal_checkpoint()`, or the one of
/// `close()` before a reopen) leaves the earlier triples in the file and the
/// change (a delete and an insert) in the sidecar WAL: the reopen applies the
/// change to the file's triples.
#[test]
fn triples_survive_a_crash_after_a_checkpoint() {
    for scenario in ["checkpoint_then_change", "close_then_change"] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prague.grafeo");
        crash_after(scenario, &path);
        assert_crashed(&path, scenario);

        assert_reopens_with(&path, &after_the_change(), scenario);
    }
}

/// A crash at any point of `close()` (its checkpoint, then the removal of the
/// sidecar WAL) loses no triple and brings back none that was deleted, after
/// the first write (`close_first`) and after a change that follows a
/// checkpoint (`close_change`).
#[cfg(feature = "testing-crash-injection")]
#[test]
fn a_crash_at_any_point_of_close_loses_no_triple() {
    for (phase, want) in [
        ("close_first", after_first_write()),
        ("close_change", after_the_change()),
    ] {
        let mut crashes = 0;
        // Until a close() runs to its end: every point before it crashes.
        for point in 1.. {
            let what = format!("{phase}, crash point {point}");
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("barcelona.grafeo");
            match in_child(&format!("{phase}:{point}"), &path) {
                0 => {
                    crashes += 1;
                    assert_crashed(&path, &what);
                    assert_reopens_with(&path, &want, &what);
                }
                CLOSE_COMPLETED => {
                    assert!(
                        !sidecar_wal(&path).exists(),
                        "{what}: a completed close() removes the sidecar WAL"
                    );
                    assert_reopens_with(&path, &want, &what);
                    break;
                }
                code => panic!("{what}: the child failed with exit code {code}"),
            }
        }
        // The checkpoint's points (the WAL rotation, the image, the WAL
        // mark) and the removal of the WAL.
        assert!(
            crashes >= 9,
            "{phase}: close() reached only {crashes} crash points"
        );
    }
}
