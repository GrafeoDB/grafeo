//! A node deleted with its edges is one change (#465): `DETACH DELETE` and
//! `detach_delete_node` remove the node and all its edges together, they
//! roll back and replay together, and change data capture reports a Delete
//! event per edge naming the node in `deleted_with`, then the node's.
//! Concurrent writes to the node's edges conflict (`edge_endpoint_conflicts`).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test detach_delete
//! ```

#![cfg(all(feature = "lpg", feature = "gql"))]

use grafeo_common::types::{NodeId, Value};
use grafeo_engine::GrafeoDB;

/// Alix knows Gus and Vincent, Vincent knows Gus. Returns Alix, Gus,
/// Vincent.
fn people(db: &GrafeoDB) -> (NodeId, NodeId, NodeId) {
    let person = |name: &str| {
        db.create_node_with_props(&["Person"], [("name", Value::from(name))])
            .unwrap()
    };
    let (alix, gus, vincent) = (person("Alix"), person("Gus"), person("Vincent"));
    for (from, to) in [(alix, gus), (alix, vincent), (vincent, gus)] {
        db.create_edge(from, to, "KNOWS").unwrap();
    }
    (alix, gus, vincent)
}

/// The `(from, to)` names of every `KNOWS` edge, sorted.
fn knows(db: &GrafeoDB) -> Vec<(String, String)> {
    let mut edges: Vec<(String, String)> = db
        .execute("MATCH (a)-[:KNOWS]->(b) RETURN a.name, b.name")
        .unwrap()
        .rows()
        .iter()
        .map(|row| (name(&row[0]), name(&row[1])))
        .collect();
    edges.sort();
    edges
}

fn name(value: &Value) -> String {
    match value {
        Value::String(name) => name.to_string(),
        other => panic!("a name: {other:?}"),
    }
}

fn pair(from: &str, to: &str) -> (String, String) {
    (from.to_string(), to.to_string())
}

#[test]
fn detach_delete_node_removes_the_node_with_its_edges() {
    let db = GrafeoDB::new_in_memory();
    let (_, gus, _) = people(&db);
    assert!(db.delete_node(gus).is_err(), "delete_node refuses edges");

    assert!(db.detach_delete_node(gus).unwrap());
    assert_eq!(knows(&db), [pair("Alix", "Vincent")]);
    assert_eq!((db.node_count(), db.edge_count()), (2, 1));
    assert!(!db.detach_delete_node(gus).unwrap(), "gone already");
}

#[test]
fn a_detach_delete_node_rolls_back_with_its_edges() {
    let db = GrafeoDB::new_in_memory();
    let (_, gus, _) = people(&db);
    let mut session = db.session();
    session.begin_transaction().unwrap();
    assert!(session.detach_delete_node(gus).unwrap());
    session.rollback().unwrap();
    assert_eq!(
        knows(&db),
        [
            pair("Alix", "Gus"),
            pair("Alix", "Vincent"),
            pair("Vincent", "Gus")
        ]
    );
    assert_eq!((db.node_count(), db.edge_count()), (3, 3));
}

#[test]
fn a_graph_handle_detach_deletes_in_its_graph() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE GRAPH trips").unwrap();
    let graph = db.graph("trips").unwrap();
    let paris = graph
        .create_node_with_props(&["City"], [("name", Value::from("Paris"))])
        .unwrap();
    let prague = graph
        .create_node_with_props(&["City"], [("name", Value::from("Prague"))])
        .unwrap();
    graph.create_edge(paris, prague, "ROUTE").unwrap();
    assert!(graph.detach_delete_node(paris).unwrap());
    let session = db.session();
    session.use_graph("trips");
    let left = session
        .execute("MATCH (c:City) RETURN c.name")
        .unwrap()
        .rows()
        .to_vec();
    assert_eq!(left, vec![vec![Value::from("Prague")]]);
    assert_eq!(
        session
            .execute("MATCH ()-[r:ROUTE]->() RETURN count(r)")
            .unwrap()
            .rows()[0][0],
        Value::Int64(0)
    );
}

/// CDC names the node on each edge a detach delete removes, in the commit
/// of the node's own Delete event, and on no other edge delete.
#[cfg(feature = "cdc")]
mod change_data_capture {
    use super::*;
    use grafeo_common::types::EpochId;
    use grafeo_engine::Config;
    use grafeo_engine::cdc::{ChangeEvent, ChangeKind, EntityId};

    fn db() -> GrafeoDB {
        GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap()
    }

    /// The Delete events of the last commit: `(entity, deleted_with)`.
    fn deletes(db: &GrafeoDB) -> Vec<(EntityId, Option<u64>)> {
        let events: Vec<ChangeEvent> = db
            .changes_between(EpochId::new(0), EpochId::new(u64::MAX))
            .unwrap();
        let last = events.last().unwrap().epoch;
        events
            .into_iter()
            .filter(|event| event.epoch == last && event.kind == ChangeKind::Delete)
            .map(|event| (event.entity_id, event.deleted_with))
            .collect()
    }

    #[test]
    fn a_detach_delete_names_the_node_on_its_edge_events() {
        for (how, delete) in [
            (
                "DETACH DELETE",
                (|db: &GrafeoDB, _| {
                    db.execute("MATCH (g:Person {name: 'Gus'}) DETACH DELETE g")
                        .unwrap();
                }) as fn(&GrafeoDB, NodeId),
            ),
            ("detach_delete_node", |db: &GrafeoDB, gus| {
                db.detach_delete_node(gus).unwrap();
            }),
        ] {
            let db = db();
            let (_, gus, _) = people(&db);
            delete(&db, gus);
            let events = deletes(&db);
            assert_eq!(events.len(), 3, "{how}: {events:?}");
            let (edges, node) = events.split_at(2);
            assert!(
                edges
                    .iter()
                    .all(|(entity, with)| matches!(entity, EntityId::Edge(_))
                        && *with == Some(gus.as_u64())),
                "{how}: {edges:?}"
            );
            assert_eq!(node, [(EntityId::Node(gus), None)], "{how}");
        }
    }

    /// An edge deleted on its own, also in the statement that deletes its
    /// node, names no node.
    #[test]
    fn an_edge_deleted_on_its_own_names_no_node() {
        let db = db();
        people(&db);
        db.execute(
            "MATCH (:Person {name: 'Alix'})-[r:KNOWS]->(:Person {name: 'Vincent'}) DELETE r",
        )
        .unwrap();
        assert!(
            matches!(deletes(&db).as_slice(), [(EntityId::Edge(_), None)]),
            "{:?}",
            deletes(&db)
        );

        db.execute("MATCH (v:Person {name: 'Vincent'})-[r:KNOWS]->() DELETE r, v")
            .unwrap();
        assert!(
            deletes(&db).iter().all(|(_, with)| with.is_none()),
            "{:?}",
            deletes(&db)
        );
    }
}

/// A detach delete committed before a crash replays whole.
#[cfg(all(feature = "wal", feature = "grafeo-file", not(miri)))]
mod replay {
    use super::*;
    use grafeo_common::testing::child_process;
    use std::path::{Path, PathBuf};

    const PATH_VAR: &str = "GRAFEO_DETACH_DELETE_PATH";

    /// Child-process entry for the crash; a no-op when run directly.
    #[test]
    fn crash_child() {
        let Some(path) = std::env::var_os(PATH_VAR) else {
            return;
        };
        let db = GrafeoDB::open(Path::new(&path)).unwrap();
        let gus = db
            .execute("MATCH (g:Person {name: 'Gus'}) RETURN id(g)")
            .unwrap()
            .rows()[0][0]
            .as_int64()
            .unwrap();
        db.detach_delete_node(NodeId::new(u64::try_from(gus).unwrap()))
            .unwrap();
        // A crash: no close, no checkpoint, no destructors.
        std::process::exit(0);
    }

    #[test]
    fn a_committed_detach_delete_replays_with_its_edges() {
        let dir = tempfile::tempdir().unwrap();
        let path: PathBuf = dir.path().join("berlin.grafeo");
        let db = GrafeoDB::open(&path).unwrap();
        people(&db);
        db.close().unwrap();
        drop(db);
        let status = child_process::run(
            std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "replay::crash_child", "--nocapture"])
                .env(PATH_VAR, &path),
        )
        .unwrap();
        assert!(status.success(), "the child failed");

        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(knows(&db), [pair("Alix", "Vincent")]);
        assert_eq!((db.node_count(), db.edge_count()), (2, 1));
    }
}
