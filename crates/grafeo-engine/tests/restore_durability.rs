//! A restore is durable when it returns (#677).
//!
//! `restore_snapshot` replaces the data of a persistent database. Once it
//! returns, the restored state must survive a crash: it is in the database
//! file or in the WAL before the call reports success. These tests restore a
//! snapshot of another state in a child process that then exits without
//! `close()` (`Drop for GrafeoDB` would close and checkpoint, which hides the
//! bug), and check that a reopen holds exactly the restored state, plus what
//! was committed after the restore, and nothing of the state before it.
//!
//! The state before the restore (S1) is Alix and Gus in Amsterdam and Berlin;
//! the snapshot (S2) is Vincent, Jules and Mia, built in an in-memory
//! database. A dump compares labels, properties and edges (by the nodes they
//! join), not raw ids.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test restore_durability
//! ```

#![cfg(all(feature = "gql", feature = "wal", feature = "grafeo-file"))]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use grafeo_common::testing::child_process;
use grafeo_common::types::EpochId;
use grafeo_engine::GrafeoDB;

const SCENARIO_VAR: &str = "GRAFEO_RESTORE_DURABILITY_SCENARIO";
const PATH_VAR: &str = "GRAFEO_RESTORE_DURABILITY_PATH";

/// What a child prints before it exits: the epoch it reached.
const EPOCH_PREFIX: &str = "reached epoch ";

/// The commits a child runs after the restore: an insert, an edge to the new
/// node and a property change on a restored node.
const COMMITS_AFTER_RESTORE: [&str; 3] = [
    "INSERT (:City {name: 'Amsterdam'})",
    "MATCH (v:Person {name: 'Vincent'}), (c:City {name: 'Amsterdam'}) \
     INSERT (v)-[:LIVES_IN {since: 2020}]->(c)",
    "MATCH (j:Person {name: 'Jules'}) SET j.city = 'Amsterdam'",
];

/// Writes S1 into `db`: Alix and Gus, an edge between them, each statement a
/// commit of its own.
fn write_s1(db: &GrafeoDB) {
    for statement in [
        "INSERT (:Person {name: 'Alix', city: 'Amsterdam', age: 31})",
        "INSERT (:Person:Pilot {name: 'Gus', city: 'Berlin', age: 44})",
        "MATCH (a:Person {name: 'Alix'}), (g:Person {name: 'Gus'}) \
         INSERT (a)-[:KNOWS {since: 2015}]->(g)",
        "MATCH (a:Person {name: 'Alix'}) SET a.age = 32",
    ] {
        db.execute(statement)
            .unwrap_or_else(|e| panic!("S1 statement {statement}: {e}"));
    }
}

/// An in-memory database holding S2: Vincent, Jules and Mia, with edges.
fn s2_database() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    for statement in [
        "INSERT (:Person {name: 'Vincent', city: 'Paris', age: 40})",
        "INSERT (:Person:Driver {name: 'Jules', city: 'Berlin', age: 38})",
        "INSERT (:Person {name: 'Mia', city: 'Paris', age: 27})",
        "MATCH (v:Person {name: 'Vincent'}), (j:Person {name: 'Jules'}) \
         INSERT (v)-[:WORKS_WITH {since: 1994}]->(j)",
        "MATCH (v:Person {name: 'Vincent'}), (m:Person {name: 'Mia'}) \
         INSERT (v)-[:KNOWS {since: 1994}]->(m)",
        "MATCH (m:Person {name: 'Mia'}) SET m.age = 28",
    ] {
        db.execute(statement)
            .unwrap_or_else(|e| panic!("S2 statement {statement}: {e}"));
    }
    db
}

/// The snapshot bytes of S2.
fn s2_snapshot() -> Vec<u8> {
    s2_database().export_snapshot().expect("export S2")
}

/// The dump a reopen must show: S2, then the commits after the restore when
/// `with_commits`, applied to an in-memory import of the snapshot.
fn expected_dump(with_commits: bool) -> Dump {
    let db = GrafeoDB::import_snapshot(&s2_snapshot()).expect("import S2");
    if with_commits {
        for statement in COMMITS_AFTER_RESTORE {
            db.execute(statement)
                .unwrap_or_else(|e| panic!("expected statement {statement}: {e}"));
        }
    }
    dump(&db)
}

/// Every node (sorted labels and properties) and every edge (its type,
/// properties and the nodes it joins), sorted.
#[derive(Debug, PartialEq, Eq)]
struct Dump {
    nodes: Vec<String>,
    edges: Vec<String>,
}

fn dump(db: &GrafeoDB) -> Dump {
    let nodes: Vec<_> = db.iter_nodes().expect("iterate the nodes").collect();
    let described: HashMap<_, _> = nodes
        .iter()
        .map(|node| {
            let mut labels: Vec<&str> = node.labels.iter().map(|label| label.as_str()).collect();
            labels.sort_unstable();
            (
                node.id,
                format!("(:{} {})", labels.join(":"), properties(&node.properties)),
            )
        })
        .collect();
    let mut node_lines: Vec<String> = described.values().cloned().collect();
    node_lines.sort();
    let mut edge_lines: Vec<String> = db
        .iter_edges()
        .expect("iterate the edges")
        .map(|edge| {
            let end = |id| {
                described
                    .get(&id)
                    .cloned()
                    .unwrap_or_else(|| format!("<missing node {id:?}>"))
            };
            format!(
                "{}-[:{} {}]->{}",
                end(edge.src),
                edge.edge_type.as_str(),
                properties(&edge.properties),
                end(edge.dst)
            )
        })
        .collect();
    edge_lines.sort();
    Dump {
        nodes: node_lines,
        edges: edge_lines,
    }
}

fn properties(map: &grafeo_common::types::PropertyMap) -> String {
    let mut entries: Vec<String> = map
        .iter()
        .map(|(key, value)| format!("{}: {value:?}", key.as_str()))
        .collect();
    entries.sort();
    format!("{{{}}}", entries.join(", "))
}

/// The names of every person in `db`, sorted.
fn names(db: &GrafeoDB) -> Vec<String> {
    db.execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
        .expect("read the names")
        .rows()
        .iter()
        .map(|row| match &row[0] {
            grafeo_common::types::Value::String(name) => name.to_string(),
            other => panic!("a name is a string, got {other:?}"),
        })
        .collect()
}

fn sidecar_wal(path: &Path) -> PathBuf {
    let mut sidecar = path.as_os_str().to_owned();
    sidecar.push(".wal");
    PathBuf::from(sidecar)
}

/// Runs `scenario` in a child process on the database at `path`; the child
/// exits without closing the database, like a crash. Returns the epoch the
/// child reached after its last commit.
fn crash_after(scenario: &str, path: &Path) -> EpochId {
    let output = child_process::output(
        std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "restore_crash_child", "--nocapture"])
            .env(SCENARIO_VAR, scenario)
            .env(PATH_VAR, path),
    )
    .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "scenario {scenario} failed: {stdout}{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let epoch = stdout
        .lines()
        .find_map(|line| line.strip_prefix(EPOCH_PREFIX))
        .unwrap_or_else(|| panic!("scenario {scenario} printed no epoch: {stdout}"));
    EpochId::new(epoch.trim().parse().unwrap())
}

/// Child-process entry for [`crash_after`]; a no-op when run directly.
#[test]
fn restore_crash_child() {
    let Ok(scenario) = std::env::var(SCENARIO_VAR) else {
        return;
    };
    let path = PathBuf::from(std::env::var_os(PATH_VAR).unwrap());
    let snapshot = s2_snapshot();
    let db = GrafeoDB::open(&path).unwrap();
    write_s1(&db);
    match scenario.as_str() {
        // S1 is only in the WAL; the restore, then the crash.
        "restore" => db.restore_snapshot(&snapshot).unwrap(),
        // S1 is only in the WAL; the restore, commits, then the crash.
        "restore_commits" => {
            db.restore_snapshot(&snapshot).unwrap();
            for statement in COMMITS_AFTER_RESTORE {
                db.execute(statement)
                    .unwrap_or_else(|e| panic!("after the restore, {statement}: {e}"));
            }
        }
        // S1 is in the file (a checkpoint); the restore, then the crash.
        "checkpoint_restore" => {
            db.wal_checkpoint().unwrap();
            db.restore_snapshot(&snapshot).unwrap();
        }
        // S1 is in the file; the restore, commits, then the crash.
        "checkpoint_restore_commits" => {
            db.wal_checkpoint().unwrap();
            db.restore_snapshot(&snapshot).unwrap();
            for statement in COMMITS_AFTER_RESTORE {
                db.execute(statement)
                    .unwrap_or_else(|e| panic!("after the restore, {statement}: {e}"));
            }
        }
        other => panic!("unknown scenario {other}"),
    }
    println!("{EPOCH_PREFIX}{}", db.current_epoch().as_u64());
    // Crash: no close(), no destructors (Drop would close and checkpoint).
    std::process::exit(0);
}

/// Crashes after `scenario`, reopens and checks the reopen holds exactly
/// `expected`, nothing of S1, at an epoch not below the child's last one.
fn check_crash(scenario: &str, file: &str, expected: &Dump) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(file);
    let reached = crash_after(scenario, &path);
    assert!(
        sidecar_wal(&path).exists(),
        "a crash leaves the sidecar WAL behind (the child did not close)"
    );

    let db = GrafeoDB::open(&path).unwrap_or_else(|e| panic!("reopen after {scenario}: {e}"));
    let reopened = dump(&db);
    let people = names(&db);
    assert!(
        !people.iter().any(|name| name == "Alix" || name == "Gus"),
        "after {scenario} and a crash, the reopen brings back S1 (the state \
         before the restore): people {people:?}, reopened {reopened:#?}"
    );
    assert_eq!(
        &reopened, expected,
        "after {scenario} and a crash, the reopen holds exactly the restored \
         state and what was committed after it"
    );
    assert!(
        db.current_epoch() >= reached,
        "after {scenario}, the reopen continues at or above epoch {}, got {}",
        reached.as_u64(),
        db.current_epoch().as_u64()
    );
    db.close().unwrap();
}

/// S1 only in the WAL, a restore of S2, a crash: the reopen holds S2.
///
/// Fails today: the restore writes nothing to the WAL, so the reopen replays
/// S1's commits from the WAL and holds Alix and Gus.
#[test]
fn a_restore_survives_a_crash() {
    check_crash("restore", "amsterdam.grafeo", &expected_dump(false));
}

/// S1 only in the WAL, a restore of S2, commits after it, a crash: the
/// reopen holds S2 plus those commits and nothing of S1.
///
/// Fails today: the WAL holds S1's commits and then the later ones, which
/// replay on top of S1 (the restored nodes they refer to are not there).
#[test]
fn commits_after_a_restore_survive_a_crash_on_top_of_it() {
    check_crash("restore_commits", "berlin.grafeo", &expected_dump(true));
}

/// S1 in the file (a checkpoint ran first), a restore of S2, a crash: the
/// reopen holds S2.
///
/// Fails today: the file holds S1 and the WAL after the checkpoint is empty,
/// so the reopen holds S1.
#[test]
fn a_restore_after_a_checkpoint_survives_a_crash() {
    check_crash("checkpoint_restore", "paris.grafeo", &expected_dump(false));
}

/// S1 in the file, a restore of S2, commits after it, a crash: the reopen
/// holds S2 plus those commits and nothing of S1.
///
/// Fails today: the file holds S1, and the commits after the restore replay
/// on top of it.
#[test]
fn commits_after_a_restore_after_a_checkpoint_survive_a_crash() {
    check_crash(
        "checkpoint_restore_commits",
        "prague.grafeo",
        &expected_dump(true),
    );
}

/// Control (passes today): a restore, then a clean `close()`, then a reopen
/// holds S2. The dump and the snapshot round trip are right, so the crash
/// tests above fail for the missing durability, not for the test.
#[test]
fn a_restore_then_close_survives_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("barcelona.grafeo");
    let expected = expected_dump(false);
    {
        let db = GrafeoDB::open(&path).unwrap();
        write_s1(&db);
        assert_ne!(dump(&db), expected, "S1 differs from S2");
        db.restore_snapshot(&s2_snapshot()).unwrap();
        assert_eq!(dump(&db), expected, "the restore holds S2 at once");
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(dump(&db), expected, "a clean close keeps the restore");
    assert_eq!(names(&db), ["Jules", "Mia", "Vincent"]);
    db.close().unwrap();
}
