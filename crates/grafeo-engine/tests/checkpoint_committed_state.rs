//! Checkpoints write the committed state (#412).
//!
//! A checkpoint while a transaction is open writes what was committed: the
//! nodes and edges the transaction deleted come back with their labels and
//! values, the values and labels it changed are written as they were before
//! it, and nothing it created is written. Each test runs a transaction in a
//! child process (DETACH DELETE, or SET, REMOVE and label changes), takes a
//! checkpoint, and ends the process without `close()`: with the transaction
//! open, after a rollback, or after a commit. The child exits while its
//! database is alive, so no clean close runs, and the parent checks that the
//! WAL is still there before it reopens the file, which reads the image and
//! replays the commits the WAL holds.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test checkpoint_committed_state
//! ```

#![cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "wal",
    feature = "grafeo-file"
))]

use std::path::{Path, PathBuf};
use std::process::Command;

use grafeo_common::testing::child_process;
use grafeo_common::types::Value;
use grafeo_engine::config::DurabilityMode;
use grafeo_engine::{Config, GrafeoDB};

/// The database path a child process works on.
const PATH_VAR: &str = "GRAFEO_CHECKPOINT_COMMITTED_STATE_PATH";
/// Which child: `<scenario>:<graph>`, the graph empty for the default graph.
const CHILD_VAR: &str = "GRAFEO_CHECKPOINT_COMMITTED_STATE_CHILD";
/// Exit code of a child that reached its end.
const EXITED: i32 = 19;
/// The named graph the tests also run in.
const TRIPS: &str = "trips";
/// The graphs each test runs in: the default graph and a named graph.
const GRAPHS: [&str; 2] = ["", TRIPS];

/// Alix -KNOWS-> Gus -KNOWS-> Mia, the committed state every transaction
/// below begins from.
const TRAVELLERS: [&str; 5] = [
    "INSERT (:Person {name: 'Alix', age: 19, city: 'Amsterdam'})",
    "INSERT (:Person:Employee {name: 'Gus', age: 3, city: 'Paris'})",
    "INSERT (:Person:Employee {name: 'Mia', age: 88})",
    "MATCH (a:Person {name: 'Alix'}), (g:Person {name: 'Gus'}) \
     INSERT (a)-[:KNOWS {since: 1988}]->(g)",
    "MATCH (g:Person {name: 'Gus'}), (m:Person {name: 'Mia'}) \
     INSERT (g)-[:KNOWS {since: 319}]->(m)",
];

/// The open transaction deletes Gus and both his edges.
const DETACH_DELETE: &str = "MATCH (g:Person {name: 'Gus'}) DETACH DELETE g";

/// The open transaction sets a value twice, adds one, removes one, adds and
/// removes labels and changes an edge value.
const WRITES: [&str; 7] = [
    "MATCH (a:Person {name: 'Alix'}) SET a.age = 88",
    "MATCH (a:Person {name: 'Alix'}) SET a.age = 3",
    "MATCH (a:Person {name: 'Alix'}) SET a.nickname = 'Al'",
    "MATCH (a:Person {name: 'Alix'}) REMOVE a.city",
    "MATCH (a:Person {name: 'Alix'}) SET a:Traveller",
    "MATCH (m:Person {name: 'Mia'}) REMOVE m:Employee",
    "MATCH (:Person {name: 'Alix'})-[k:KNOWS]->() SET k.since = 3",
];

fn open(path: &Path) -> GrafeoDB {
    GrafeoDB::with_config(Config::persistent(path).with_wal_durability(DurabilityMode::Sync))
        .unwrap()
}

/// The sidecar WAL directory of the database file at `path`.
fn sidecar_wal(path: &Path) -> PathBuf {
    let mut sidecar = path.as_os_str().to_owned();
    sidecar.push(".wal");
    PathBuf::from(sidecar)
}

/// A database at `path` holding [`TRAVELLERS`] in the default graph and in
/// graph "trips", closed, so the file holds them.
fn travellers(path: &Path) {
    let db = open(path);
    db.create_graph(TRIPS).unwrap();
    for graph in GRAPHS {
        let session = db.session();
        if !graph.is_empty() {
            session.use_graph(graph);
        }
        for statement in TRAVELLERS {
            session.execute(statement).unwrap();
        }
    }
    db.close().unwrap();
}

/// Runs the child `scenario` in `graph` on the database at `path`, then
/// checks it ended without closing the database: its WAL is still there.
fn run_child(scenario: &str, graph: &str, path: &Path) {
    let output = child_process::output(
        Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "committed_state_child", "--nocapture"])
            .env(CHILD_VAR, format!("{scenario}:{graph}"))
            .env(PATH_VAR, path),
    )
    .unwrap();
    assert_eq!(
        output.status.code(),
        Some(EXITED),
        "the child {scenario} in {graph:?} exited early:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        sidecar_wal(path).exists(),
        "the child {scenario} in {graph:?} closed its database: no WAL is left to replay"
    );
}

/// Child-process entry for [`run_child`]; a no-op when run directly.
///
/// Begins a transaction in the graph, runs the scenario's statements, takes
/// a checkpoint, then (for some scenarios) rolls back or commits, and exits
/// with the database still open.
#[test]
fn committed_state_child() {
    let (Ok(which), Some(path)) = (std::env::var(CHILD_VAR), std::env::var_os(PATH_VAR)) else {
        return;
    };
    let (scenario, graph) = which.split_once(':').unwrap();
    let db = open(Path::new(&path));
    let mut session = db.session();
    if !graph.is_empty() {
        session.use_graph(graph);
    }
    session.begin_transaction().unwrap();
    let statements: Vec<&str> = match scenario {
        "delete" => vec![DETACH_DELETE],
        "writes" => WRITES.to_vec(),
        "rollback" | "commit" => WRITES.into_iter().chain([DETACH_DELETE]).collect(),
        #[cfg(feature = "compact-store")]
        "compacted_writes" | "compacted_rollback" => COMPACTED_WRITES.to_vec(),
        other => panic!("unknown scenario {other}"),
    };
    for statement in statements {
        session.execute(statement).unwrap();
    }
    db.wal_checkpoint().unwrap();
    match scenario {
        "rollback" | "compacted_rollback" => session.rollback().unwrap(),
        "commit" => session.commit().unwrap(),
        _ => {}
    }
    std::process::exit(EXITED);
}

/// The sorted labels in a `labels(...)` value.
fn sorted_labels(value: &Value) -> Vec<String> {
    let Value::List(labels) = value else {
        panic!("labels() returned {value:?}");
    };
    let mut labels: Vec<String> = labels
        .iter()
        .map(|label| match label {
            Value::String(label) => label.to_string(),
            other => panic!("a label {other:?}"),
        })
        .collect();
    labels.sort();
    labels
}

/// One person: name, sorted labels, age, city and nickname.
type Person = (Value, Vec<String>, Value, Value, Value);

/// The people of `graph` in `db`, by name.
fn people(db: &GrafeoDB, graph: &str) -> Vec<Person> {
    let session = db.session();
    if !graph.is_empty() {
        session.use_graph(graph);
    }
    session
        .execute(
            "MATCH (p:Person) RETURN p.name AS name, labels(p) AS labels, p.age AS age, \
             p.city AS city, p.nickname AS nickname ORDER BY name",
        )
        .unwrap()
        .rows()
        .iter()
        .map(|row| {
            (
                row[0].clone(),
                sorted_labels(&row[1]),
                row[2].clone(),
                row[3].clone(),
                row[4].clone(),
            )
        })
        .collect()
}

/// The KNOWS edges of `graph` in `db`: source, target and `since`, by source.
fn knows(db: &GrafeoDB, graph: &str) -> Vec<Vec<Value>> {
    let session = db.session();
    if !graph.is_empty() {
        session.use_graph(graph);
    }
    session
        .execute(
            "MATCH (a)-[k:KNOWS]->(b) RETURN a.name AS source, b.name AS target, \
             k.since AS since ORDER BY source",
        )
        .unwrap()
        .rows()
        .to_vec()
}

fn person(
    name: &str,
    labels: &[&str],
    age: i64,
    city: Option<&str>,
    nickname: Option<&str>,
) -> Person {
    let text = |value: Option<&str>| value.map_or(Value::Null, Value::from);
    (
        Value::from(name),
        labels.iter().map(|label| (*label).to_string()).collect(),
        Value::Int64(age),
        text(city),
        text(nickname),
    )
}

/// Asserts that `graph` in the database at `path` holds [`TRAVELLERS`] as
/// committed: every person with their labels and values, both edges with
/// theirs.
fn assert_travellers(path: &Path, graph: &str, after: &str) {
    let db = open(path);
    assert_eq!(
        people(&db, graph),
        [
            person("Alix", &["Person"], 19, Some("Amsterdam"), None),
            person("Gus", &["Employee", "Person"], 3, Some("Paris"), None),
            person("Mia", &["Employee", "Person"], 88, None, None),
        ],
        "{after} in {graph:?}: the people as committed"
    );
    assert_eq!(
        knows(&db, graph),
        [
            vec![Value::from("Alix"), Value::from("Gus"), Value::Int64(1988)],
            vec![Value::from("Gus"), Value::from("Mia"), Value::Int64(319)],
        ],
        "{after} in {graph:?}: the edges as committed"
    );
    db.close().unwrap();
}

/// Guarantee 1: a checkpoint during an open DETACH DELETE, then a crash: the
/// file holds the node with its labels and values and both edges with
/// theirs.
#[test]
fn a_checkpoint_during_an_open_detach_delete_keeps_the_node_and_its_edges() {
    for graph in GRAPHS {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("amsterdam.grafeo");
        travellers(&path);
        run_child("delete", graph, &path);
        assert_travellers(&path, graph, "an open DETACH DELETE");
    }
}

/// Guarantee 2: a checkpoint during open SET, REMOVE and label changes, then
/// a crash: the file holds the committed values and labels.
#[test]
fn a_checkpoint_during_open_writes_keeps_the_committed_values_and_labels() {
    for graph in GRAPHS {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("berlin.grafeo");
        travellers(&path);
        run_child("writes", graph, &path);
        assert_travellers(&path, graph, "open writes");
    }
}

/// Guarantee 4: a checkpoint during an open transaction, which then rolls
/// back, then a crash: the file holds the committed state.
#[test]
fn a_checkpoint_then_a_rollback_leaves_the_committed_state() {
    for graph in GRAPHS {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("paris.grafeo");
        travellers(&path);
        run_child("rollback", graph, &path);
        assert_travellers(&path, graph, "a rollback after the checkpoint");
    }
}

/// Guarantee 5: a checkpoint during an open transaction, which then commits,
/// then a crash: the image holds the committed state from before it, and the
/// WAL replays the commit on it.
#[test]
fn a_checkpoint_then_a_commit_replays_the_commit_on_the_image() {
    for graph in GRAPHS {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prague.grafeo");
        travellers(&path);
        run_child("commit", graph, &path);
        let db = open(&path);
        assert_eq!(
            people(&db, graph),
            [
                person("Alix", &["Person", "Traveller"], 3, None, Some("Al")),
                person("Mia", &["Person"], 88, None, None),
            ],
            "in {graph:?}: the commit, replayed"
        );
        assert_eq!(
            knows(&db, graph),
            Vec::<Vec<Value>>::new(),
            "in {graph:?}: the commit deleted both edges"
        );
        db.close().unwrap();
    }
}

/// The open transaction changes Vincent, whom the database got after
/// `compact()`: a value set twice, one added, one removed, a label added and
/// one removed.
#[cfg(feature = "compact-store")]
const COMPACTED_WRITES: [&str; 6] = [
    "MATCH (v:Person {name: 'Vincent'}) SET v.age = 88",
    "MATCH (v:Person {name: 'Vincent'}) SET v.age = 3",
    "MATCH (v:Person {name: 'Vincent'}) SET v.nickname = 'Vin'",
    "MATCH (v:Person {name: 'Vincent'}) REMOVE v.city",
    "MATCH (v:Person {name: 'Vincent'}) SET v:Traveller",
    "MATCH (v:Person {name: 'Vincent'}) REMOVE v:Employee",
];

/// A compacted database at `path`: Alix in the compacted base and Vincent,
/// added after `compact()`, in the overlay; closed, so the file holds both.
#[cfg(feature = "compact-store")]
fn compacted(path: &Path) {
    let mut db = open(path);
    db.execute("INSERT (:Person {name: 'Alix', age: 19, city: 'Amsterdam'})")
        .unwrap();
    db.compact().unwrap();
    db.execute("INSERT (:Person:Employee {name: 'Vincent', age: 19, city: 'Prague'})")
        .unwrap();
    db.close().unwrap();
}

/// Asserts that the compacted database at `path` holds Alix and Vincent as
/// committed.
#[cfg(feature = "compact-store")]
fn assert_compacted(path: &Path, after: &str) {
    let db = open(path);
    assert_eq!(
        people(&db, ""),
        [
            person("Alix", &["Person"], 19, Some("Amsterdam"), None),
            person("Vincent", &["Employee", "Person"], 19, Some("Prague"), None),
        ],
        "{after}: the people as committed"
    );
    db.close().unwrap();
}

/// Guarantee 2 on a compacted database: a checkpoint during open writes to
/// a node added after `compact()`, then a crash: the overlay in the file
/// holds the committed values and labels.
#[cfg(feature = "compact-store")]
#[test]
fn after_compact_a_checkpoint_during_open_writes_keeps_the_committed_state() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("amsterdam.grafeo");
    compacted(&path);
    run_child("compacted_writes", "", &path);
    assert_compacted(&path, "open writes after compact()");
}

/// Guarantee 4 on a compacted database: a checkpoint during open writes to
/// a node added after `compact()`, a rollback, then a crash: the file holds
/// the committed state.
#[cfg(feature = "compact-store")]
#[test]
fn after_compact_a_checkpoint_then_a_rollback_leaves_the_committed_state() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("berlin.grafeo");
    compacted(&path);
    run_child("compacted_rollback", "", &path);
    assert_compacted(&path, "a rollback after compact()");
}
