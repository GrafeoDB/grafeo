//! Bulk writes: a batch call records the ids it reserved as one range per
//! table instead of an entry per row, and an import (or an RDF batch insert)
//! writes its rows to the WAL as it goes, closed by one commit marker. What
//! they guarantee: a batch is all or nothing, and its ids are never handed
//! out again after a rollback; in a transaction it follows the savepoints;
//! a crash in the middle of a batch or an import leaves nothing of it, and
//! one that returned survives a crash.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test bulk_writes
//! ```

#![cfg(all(feature = "lpg", feature = "gql"))]

use std::collections::HashMap;

use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, Value};
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::BatchEdge;

/// A person's properties.
fn person(name: &str) -> HashMap<PropertyKey, Value> {
    HashMap::from([(PropertyKey::new("name"), Value::from(name))])
}

/// The values of a single-column query, as text, sorted.
fn column(db: &GrafeoDB, query: &str) -> Vec<String> {
    let mut values: Vec<String> = db
        .execute(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"))
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::String(text) => text.to_string(),
            other => other.to_string(),
        })
        .collect();
    values.sort();
    values
}

/// A value nested deeper than a database stores: a row holding it is
/// refused.
fn too_deep() -> Value {
    let mut value = Value::from("Prague");
    for _ in 0..=grafeo_common::storage::value_codec::MAX_PROPERTY_VALUE_DEPTH {
        value = Value::List(std::sync::Arc::from(vec![value]));
    }
    value
}

/// A batch whose later row fails leaves nothing of the batch: not its
/// earlier rows, in the default graph or a named one, not in the counts,
/// and not in change data capture.
#[cfg(feature = "cdc")]
#[test]
fn a_failing_row_undoes_the_whole_batch() {
    use grafeo_engine::Config;

    let db = GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap();
    db.execute("CREATE GRAPH trips").unwrap();
    let alix = db.create_node(&["Person"]).unwrap();
    let gus = db.create_node(&["Person"]).unwrap();
    // The epochs after the creates above.
    let start = EpochId::new(db.current_epoch().as_u64() + 1);
    let (nodes, edges) = (db.node_count(), db.edge_count());

    let mut bad = person("Vincent");
    bad.insert(PropertyKey::new("trips"), too_deep());
    db.batch_create_nodes_with_props("Person", vec![person("Mia"), person("Jules"), bad.clone()])
        .unwrap_err();
    db.batch_create_edges(vec![
        BatchEdge::new(alix, gus, "KNOWS"),
        BatchEdge::new(gus, alix, "KNOWS"),
        BatchEdge::new(gus, NodeId::new(388), "KNOWS"),
    ])
    .unwrap_err();
    let trips = db.graph("trips").unwrap();
    trips
        .batch_create_nodes_with_props("City", vec![person("Paris"), bad])
        .unwrap_err();

    assert_eq!((db.node_count(), db.edge_count()), (nodes, edges));
    assert_eq!(column(&db, "MATCH (p:Person) RETURN count(p)"), ["2"]);
    assert_eq!(column(&db, "MATCH ()-[r]->() RETURN count(r)"), ["0"]);
    assert_eq!(
        trips.execute("MATCH (n) RETURN count(n)").unwrap().rows()[0][0],
        Value::Int64(0)
    );
    let events = db.changes_between(start, db.current_epoch()).unwrap();
    assert!(
        events.is_empty(),
        "a failed batch reports nothing: {events:?}"
    );

    // The batches go through once their rows are fine.
    let created = db
        .batch_create_nodes_with_props("Person", vec![person("Mia"), person("Jules")])
        .unwrap();
    assert_eq!(created.len(), 2);
    assert_eq!(
        db.changes_between(start, db.current_epoch()).unwrap().len(),
        2,
        "one create event per row"
    );
}

/// The ids a batch reserved are never handed out again once it is rolled
/// back: not by the next batch, not by a single create.
#[test]
fn a_batchs_ids_are_never_handed_out_again_after_a_rollback() {
    let db = GrafeoDB::new_in_memory();
    let alix = db.create_node(&["Person"]).unwrap();
    let gus = db.create_node(&["Person"]).unwrap();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    let nodes = session
        .batch_create_nodes_with_props("Person", vec![person("Mia"), person("Jules")])
        .unwrap();
    let edges = session
        .batch_create_edges(vec![
            BatchEdge::new(alix, gus, "KNOWS"),
            BatchEdge::new(gus, alix, "KNOWS"),
        ])
        .unwrap();
    session.rollback().unwrap();
    assert!(nodes.iter().all(|id| db.get_node(*id).is_none()));

    let mut later_nodes = db
        .batch_create_nodes_with_props("Person", vec![person("Vincent"), person("Butch")])
        .unwrap();
    later_nodes.push(db.create_node(&["Person"]).unwrap());
    let mut later_edges: Vec<EdgeId> = db
        .batch_create_edges(vec![BatchEdge::new(alix, gus, "KNOWS")])
        .unwrap();
    later_edges.push(db.create_edge(gus, alix, "KNOWS").unwrap());
    assert!(
        later_nodes.iter().all(|id| !nodes.contains(id)),
        "{later_nodes:?} reuses one of {nodes:?}"
    );
    assert!(
        later_edges.iter().all(|id| !edges.contains(id)),
        "{later_edges:?} reuses one of {edges:?}"
    );
}

/// A batch call inside a transaction is part of it: a rollback to a
/// savepoint before a batch takes the batch back, also from change data
/// capture, and the commit reports and logs the batches before it and
/// after it.
#[cfg(feature = "cdc")]
#[test]
fn a_batch_in_a_transaction_follows_its_savepoints() {
    use grafeo_engine::Config;
    use grafeo_engine::cdc::{ChangeKind, EntityId};

    let db = GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap();
    let start = EpochId::new(db.current_epoch().as_u64() + 1);
    let mut session = db.session();
    session.begin_transaction().unwrap();
    let kept = session
        .batch_create_nodes_with_props("Person", vec![person("Alix"), person("Gus")])
        .unwrap();
    session.savepoint("before_mia").unwrap();
    session
        .batch_create_nodes_with_props("Person", vec![person("Mia")])
        .unwrap();
    session
        .batch_create_edges(vec![BatchEdge::new(kept[0], kept[1], "KNOWS")])
        .unwrap();
    session.rollback_to_savepoint("before_mia").unwrap();
    let knows = session
        .batch_create_edges(vec![
            BatchEdge::new(kept[1], kept[0], "KNOWS").with_properties([("since", 2019_i64)]),
        ])
        .unwrap();
    session.commit().unwrap();

    assert_eq!(
        column(&db, "MATCH (p:Person) RETURN p.name"),
        ["Alix", "Gus"]
    );
    assert_eq!(
        column(
            &db,
            "MATCH (a)-[r:KNOWS]->(b) RETURN b.name + ' ' + toString(r.since)"
        ),
        ["Alix 2019"]
    );
    // The events of one commit, in no fixed order.
    let mut events: Vec<String> = db
        .changes_between(start, db.current_epoch())
        .unwrap()
        .into_iter()
        .map(|event| format!("{:?} {:?}", event.entity_id, event.kind))
        .collect();
    events.sort();
    let mut expected: Vec<String> = [
        EntityId::Node(kept[0]),
        EntityId::Node(kept[1]),
        EntityId::Edge(knows[0]),
    ]
    .iter()
    .map(|entity| format!("{entity:?} {:?}", ChangeKind::Create))
    .collect();
    expected.sort();
    assert_eq!(
        events, expected,
        "the creates of the batches the commit kept"
    );
}

/// Crashes: each child process writes to its database, then exits without
/// `close()`, so the reopen replays the WAL.
#[cfg(all(feature = "wal", feature = "grafeo-file"))]
mod crash {
    use std::path::{Path, PathBuf};

    use grafeo_common::testing::child_process;
    use grafeo_engine::GrafeoDB;
    use grafeo_engine::database::BatchEdge;

    use super::{column, person};

    /// Tells a child process where its database is.
    const PATH_VAR: &str = "GRAFEO_BULK_WRITES_CRASH_PATH";

    /// The TSV of a ring of `nodes` nodes: each line one edge.
    fn ring(nodes: u64) -> String {
        use std::fmt::Write;

        let mut tsv = String::new();
        for node in 0..nodes {
            writeln!(tsv, "{node}\t{}", (node + 1) % nodes).unwrap();
        }
        tsv
    }

    /// Runs the child test `name` on a new database, which it leaves
    /// crashed, and returns the database's path.
    fn crashed_by(name: &str, dir: &Path) -> PathBuf {
        let path = dir.join("prague.grafeo");
        let status = child_process::run(
            std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", name, "--nocapture"])
                .env(PATH_VAR, &path),
        )
        .unwrap();
        assert!(status.success(), "the child process {name} failed");
        let mut sidecar = path.as_os_str().to_owned();
        sidecar.push(".wal");
        assert!(
            PathBuf::from(sidecar).exists(),
            "the crash left the WAL for the reopen to replay"
        );
        path
    }

    /// The database of a child process, or `None` when the test runs on its
    /// own.
    fn child_database() -> Option<GrafeoDB> {
        let path = std::env::var_os(PATH_VAR)?;
        Some(GrafeoDB::open(PathBuf::from(path)).unwrap())
    }

    /// What the reopened database holds: the people, the cities and the
    /// number of edges.
    fn state(db: &GrafeoDB) -> (Vec<String>, Vec<String>, Vec<String>) {
        (
            column(db, "MATCH (p:Person) RETURN p.name"),
            column(db, "MATCH (n:_Imported) RETURN count(n)"),
            column(db, "MATCH ()-[r]->() RETURN count(r)"),
        )
    }

    /// Batches and an import that returned survive a crash: the WAL holds
    /// their rows, a batch's kept with its range until its commit, an
    /// import's written as it went.
    #[test]
    fn batches_and_imports_that_returned_survive_a_crash() {
        if let Some(db) = child_database() {
            let people = db
                .batch_create_nodes_with_props("Person", vec![person("Alix"), person("Gus")])
                .unwrap();
            db.batch_create_edges(vec![
                BatchEdge::new(people[0], people[1], "KNOWS").with_properties([("since", 3_i64)]),
            ])
            .unwrap();
            db.import_tsv_str(&ring(3_000), "NEXT", true).unwrap();
            // A batch a savepoint rollback took back is not logged.
            let mut session = db.session();
            session.begin_transaction().unwrap();
            session
                .batch_create_nodes_with_props("Person", vec![person("Vincent")])
                .unwrap();
            session.savepoint("before_mia").unwrap();
            session
                .batch_create_nodes_with_props("Person", vec![person("Mia")])
                .unwrap();
            session.rollback_to_savepoint("before_mia").unwrap();
            session.commit().unwrap();
            // Crash: no close(), no checkpoint, no destructors.
            std::process::exit(0);
        }
        let dir = tempfile::tempdir().unwrap();
        let path = crashed_by(
            "crash::batches_and_imports_that_returned_survive_a_crash",
            dir.path(),
        );
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(
            state(&db),
            (
                vec!["Alix".to_string(), "Gus".to_string(), "Vincent".to_string()],
                vec!["3000".to_string()],
                vec!["3001".to_string()],
            )
        );
        assert_eq!(
            column(&db, "MATCH ()-[r:KNOWS]->() RETURN r.since"),
            ["3"],
            "a batch's values replay too"
        );
        db.close().unwrap();
    }

    /// A crash in the middle of a batch leaves nothing of it: its rows are
    /// logged only by its commit. The batch before it is kept.
    #[cfg(feature = "testing-statement-injection")]
    #[test]
    fn a_crash_in_the_middle_of_a_batch_leaves_nothing_of_it() {
        if let Some(db) = child_database() {
            let people = db
                .batch_create_nodes_with_props("Person", vec![person("Alix"), person("Gus")])
                .unwrap();
            // The second batch's commit holds commits off, with every row
            // applied and nothing logged: the crash comes there.
            grafeo_common::testing::commit_hook::during_next_held_change(|| {
                std::process::exit(0);
            });
            let edges = (0..3_000)
                .map(|_| BatchEdge::new(people[0], people[1], "KNOWS"))
                .collect();
            let _ = db.batch_create_edges(edges);
            unreachable!("the child process exits in the batch's commit");
        }
        let dir = tempfile::tempdir().unwrap();
        let path = crashed_by(
            "crash::a_crash_in_the_middle_of_a_batch_leaves_nothing_of_it",
            dir.path(),
        );
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(
            state(&db),
            (
                vec!["Alix".to_string(), "Gus".to_string()],
                vec!["0".to_string()],
                vec!["0".to_string()],
            )
        );
        // Writes go on after the replay, the batch's ids unused.
        let vincent = db.create_node(&["Person"]).unwrap();
        assert!(db.get_node(vincent).is_some());
        db.close().unwrap();
    }

    /// An import is atomic across a crash: one in the middle of it, with
    /// part of its rows in the WAL and no commit marker after them, leaves
    /// nothing of it; the writes before it are kept, and so are the ones
    /// after the reopen.
    #[cfg(feature = "testing-crash-injection")]
    #[test]
    fn an_import_is_atomic_across_a_crash() {
        if let Some(db) = child_database() {
            db.batch_create_nodes_with_props("Person", vec![person("Alix")])
                .unwrap();
            // The crash exits at once, without unwinding into the database.
            std::panic::set_hook(Box::new(|info| {
                eprintln!("{info}");
                std::process::exit(0);
            }));
            // The second run of records written: the import is half logged.
            grafeo_common::testing::crash::enable_crash_at(2);
            let _ = db.import_tsv_str(&ring(5_000), "NEXT", true);
            unreachable!("the child process exits in the import");
        }
        let dir = tempfile::tempdir().unwrap();
        let path = crashed_by("crash::an_import_is_atomic_across_a_crash", dir.path());
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(
            state(&db),
            (
                vec!["Alix".to_string()],
                vec!["0".to_string()],
                vec!["0".to_string()]
            )
        );
        db.import_tsv_str(&ring(3), "NEXT", true).unwrap();
        assert_eq!(column(&db, "MATCH (n:_Imported) RETURN count(n)"), ["3"]);
        db.close().unwrap();
    }

    /// An import whose WAL write fails halfway leaves nothing: its rows are
    /// undone, and the records it wrote are closed by an abort marker, so the
    /// commit marker of a later write does not commit them for the replay
    /// after a crash.
    #[cfg(feature = "testing-crash-injection")]
    #[test]
    fn a_failed_import_leaves_nothing_also_after_later_commits_and_a_crash() {
        if let Some(db) = child_database() {
            let failed = grafeo_common::testing::crash::with_failure_at(2, || {
                db.import_tsv_str(&ring(5_000), "NEXT", true)
            });
            assert!(failed.is_err(), "the second run of records fails");
            assert_eq!(state(&db).1, ["0"], "nothing of it in memory");
            db.batch_create_nodes_with_props("Person", vec![person("Gus")])
                .unwrap();
            std::process::exit(0);
        }
        let dir = tempfile::tempdir().unwrap();
        let path = crashed_by(
            "crash::a_failed_import_leaves_nothing_also_after_later_commits_and_a_crash",
            dir.path(),
        );
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(
            state(&db),
            (
                vec!["Gus".to_string()],
                vec!["0".to_string()],
                vec!["0".to_string()]
            )
        );
        db.close().unwrap();
    }

    /// An RDF batch insert and an RDF import that returned survive a crash,
    /// and one interrupted halfway leaves nothing.
    #[cfg(all(feature = "triple-store", feature = "testing-crash-injection"))]
    #[test]
    fn rdf_batch_inserts_survive_a_crash_whole_or_not_at_all() {
        use grafeo_core::graph::rdf::{Term, Triple};

        let knows = |from: u64, to: u64| {
            Triple::new(
                Term::iri(format!("http://example.org/person/{from}")),
                Term::iri("http://example.org/knows"),
                Term::iri(format!("http://example.org/person/{to}")),
            )
        };
        if let Some(db) = child_database() {
            db.batch_insert_rdf((0..3).map(|n| knows(n, n + 1)))
                .unwrap();
            let dir =
                std::path::Path::new(&std::env::var_os(PATH_VAR).unwrap()).with_extension("tsv");
            std::fs::write(&dir, ring(19)).unwrap();
            db.import_tsv_rdf(&dir, "http://example.org/next", "http://example.org/n/")
                .unwrap();
            std::panic::set_hook(Box::new(|info| {
                eprintln!("{info}");
                std::process::exit(0);
            }));
            grafeo_common::testing::crash::enable_crash_at(2);
            let _ = db.batch_insert_rdf((100..10_000).map(|n| knows(n, n + 1)));
            unreachable!("the child process exits in the insert");
        }
        let dir = tempfile::tempdir().unwrap();
        let path = crashed_by(
            "crash::rdf_batch_inserts_survive_a_crash_whole_or_not_at_all",
            dir.path(),
        );
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(
            db.rdf_store().len(),
            3 + 19,
            "the two inserts that returned"
        );
        assert!(db.rdf_store().contains(&knows(2, 3)));
        assert!(!db.rdf_store().contains(&knows(100, 101)));
        db.close().unwrap();
    }
}
