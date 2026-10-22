//! A [`Snapshot`] reads one pinned state of one graph: every read through it
//! sees the committed state of its epoch while it lives, across later
//! commits and garbage collection, from the database, a graph handle, a
//! projection and a view of an earlier epoch.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test snapshots
//! ```

use grafeo_common::types::{NodeId, Value};
use grafeo_engine::{Config, Direction, GrafeoDB, HistoryRetention, ProjectionSpec, Snapshot};

fn name(snapshot: &Snapshot, id: NodeId) -> Option<Value> {
    snapshot.node_property(id, "name", snapshot.view()).unwrap()
}

fn nodes(snapshot: &Snapshot) -> usize {
    snapshot.node_count(snapshot.view()).unwrap()
}

/// Alix knows Gus, in the default graph.
fn people() -> (GrafeoDB, NodeId, NodeId) {
    let db = GrafeoDB::new_in_memory();
    let alix = db
        .create_node_with_props(&["Person"], [("name", Value::from("Alix"))])
        .unwrap();
    let gus = db
        .create_node_with_props(&["Person"], [("name", Value::from("Gus"))])
        .unwrap();
    db.create_edge(alix, gus, "KNOWS").unwrap();
    (db, alix, gus)
}

#[test]
fn a_snapshot_reads_one_state_across_commits_and_collects() {
    let (db, alix, gus) = people();
    let snapshot = db.snapshot().unwrap();
    assert_eq!(snapshot.epoch(), db.current_epoch());

    // Later commits: a value set, a node and an edge added, a node deleted.
    db.set_node_property(alix, "name", Value::from("Alix B."))
        .unwrap();
    let vincent = db.create_node(&["Person"]).unwrap();
    db.create_edge(gus, vincent, "KNOWS").unwrap();
    db.execute("MATCH (p:Person {name: 'Alix B.'}) DETACH DELETE p")
        .unwrap();
    // A collect while the snapshot lives keeps every version it reads.
    db.gc();

    assert_eq!(name(&snapshot, alix), Some(Value::from("Alix")));
    assert_eq!(nodes(&snapshot), 2);
    assert_eq!(snapshot.edge_count(snapshot.view()).unwrap(), 1);
    assert_eq!(
        snapshot
            .edges(gus, Direction::Both, &[], snapshot.view())
            .unwrap()
            .len(),
        1,
        "Gus's edge from Alix, not the later one to Vincent"
    );
    assert!(!snapshot.is_node_visible(vincent, snapshot.view()).unwrap());

    // A new snapshot sees the commits.
    let now = db.snapshot().unwrap();
    assert!(now.epoch() > snapshot.epoch());
    assert_eq!(name(&now, alix), None, "Alix is deleted now");
    assert_eq!(nodes(&now), 2, "Gus and Vincent");
}

#[test]
fn a_snapshot_reads_the_selected_graph_and_a_handle_its_own() {
    let (db, _, _) = people();
    db.create_graph("trips").unwrap();
    let trips = db.graph("trips").unwrap();
    trips
        .create_node_with_props(&["City"], [("name", Value::from("Prague"))])
        .unwrap();

    assert_eq!(nodes(&db.snapshot().unwrap()), 2, "the default graph");
    assert_eq!(nodes(&trips.snapshot().unwrap()), 1);
    db.set_current_graph(Some("trips")).unwrap();
    assert_eq!(nodes(&db.snapshot().unwrap()), 1, "the selected graph");

    // A snapshot keeps its graph's state when the graph is dropped; a new
    // one of the dropped graph is an error.
    let kept = trips.snapshot().unwrap();
    db.session().execute("DROP GRAPH trips").unwrap();
    assert_eq!(nodes(&kept), 1);
    let error = trips.snapshot().unwrap_err().to_string();
    assert!(error.contains("does not exist"), "{error}");
    let error = db.snapshot().unwrap_err().to_string();
    assert!(error.contains("does not exist"), "{error}");
}

#[test]
fn a_projection_snapshot_reads_the_projection() {
    let (db, alix, _) = people();
    db.create_node_with_props(&["City"], [("name", Value::from("Berlin"))])
        .unwrap();
    db.create_projection("people", ProjectionSpec::new().with_node_labels(["Person"]))
        .unwrap();
    let snapshot = db.projection("people").expect("the projection");
    assert_eq!(nodes(&snapshot), 2, "the people, not the city");
    assert_eq!(name(&snapshot, alix), Some(Value::from("Alix")));
    assert!(db.projection("nowhere").is_none());
}

#[test]
fn a_snapshot_of_an_earlier_epoch_outlives_its_view() {
    let db = GrafeoDB::with_config(
        Config::in_memory().with_history_retention(HistoryRetention::Forever),
    )
    .unwrap();
    let alix = db
        .create_node_with_props(&["Person"], [("name", Value::from("Alix"))])
        .unwrap();
    let then = db.current_epoch();
    db.set_node_property(alix, "name", Value::from("Alix B."))
        .unwrap();

    let snapshot = db.as_of(then).unwrap().snapshot();
    // The view is dropped: the snapshot has its own pin.
    db.set_node_property(alix, "name", Value::from("Alix C."))
        .unwrap();
    db.gc();
    assert_eq!(snapshot.epoch(), then);
    assert_eq!(name(&snapshot, alix), Some(Value::from("Alix")));
    assert_eq!(
        name(&db.snapshot().unwrap(), alix),
        Some(Value::from("Alix C."))
    );
}

/// `validate()` of a database on an external store reads it through one
/// snapshot (it panicked on the built-in store it does not have), and says
/// it cannot check the store's edges for dangling ends.
#[test]
fn validate_of_an_external_store_says_what_it_does_not_check() {
    use grafeo_core::graph::GraphStoreMut as _;
    use grafeo_core::graph::rowgroup::RowGroupStore;
    use std::sync::Arc;

    let store = Arc::new(RowGroupStore::new());
    store.create_node(&["Person"]);
    let db = GrafeoDB::with_read_store(store, Config::in_memory()).unwrap();
    let report = db.validate().unwrap();
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    let codes: Vec<&str> = report.warnings.iter().map(|w| w.code.as_str()).collect();
    assert_eq!(codes, ["EXTERNAL_EDGES_UNCHECKED", "NO_EDGES"]);

    // The built-in store's edges are checked, and none dangles.
    let (db, _, _) = people();
    let report = db.validate().unwrap();
    assert!(report.errors.is_empty() && report.warnings.is_empty());
}
