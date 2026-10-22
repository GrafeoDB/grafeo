//! The direct read API (`get_node`, `get_edge`, `get_node_labels`, the
//! counts, `find_nodes_by_property`, graph handles) reads what queries read:
//! the compacted data after `compact()`, an external store, and nothing of
//! another graph once the selected graph is gone.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test direct_reads
//! ```

use std::sync::Arc;

use grafeo_common::types::{NodeId, Value};
use grafeo_core::graph::rowgroup::RowGroupStore;
use grafeo_core::graph::traits::GraphStoreMut;
use grafeo_engine::{Config, GrafeoDB};

fn name(db: &GrafeoDB, id: NodeId) -> Option<Value> {
    db.get_node(id)
        .unwrap()
        .and_then(|node| node.get_property("name").cloned())
}

fn person(db: &GrafeoDB, name: &str) -> NodeId {
    db.create_node_with_props(&["Person"], [("name", Value::from(name))])
        .unwrap()
}

#[test]
fn direct_reads_see_the_compacted_data() {
    let mut db = GrafeoDB::new_in_memory();
    let alix = person(&db, "Alix");
    let gus = person(&db, "Gus");
    let knows = db.create_edge(alix, gus, "KNOWS").unwrap();
    db.compact().unwrap();
    let vincent = person(&db, "Vincent");

    assert_eq!(name(&db, alix), Some(Value::from("Alix")));
    assert_eq!(name(&db, vincent), Some(Value::from("Vincent")));
    assert_eq!(
        db.get_edge(knows).unwrap().map(|edge| (edge.src, edge.dst)),
        Some((alix, gus))
    );
    assert_eq!(
        db.get_node_labels(gus).unwrap(),
        Some(vec!["Person".to_string()])
    );
    assert_eq!((db.node_count().unwrap(), db.edge_count().unwrap()), (3, 1));
    assert_eq!(
        db.find_nodes_by_property("name", &Value::from("Alix"))
            .unwrap(),
        vec![alix]
    );
    let handle = db.graph("default").unwrap();
    assert_eq!(
        handle.get_node(alix).unwrap().map(|node| node.id),
        Some(alix)
    );
    assert_eq!(
        handle.get_edge(knows).unwrap().map(|edge| edge.id),
        Some(knows)
    );

    // The admin views count the same, and an edge from a compacted node to a
    // new one is not dangling.
    db.create_edge(alix, vincent, "KNOWS").unwrap();
    assert_eq!(
        (db.info().unwrap().node_count, db.info().unwrap().edge_count),
        (3, 2)
    );
    let stats = db.detailed_stats().unwrap();
    assert_eq!((stats.node_count, stats.edge_count), (3, 2));
    let validation = db.validate().unwrap();
    assert!(validation.errors.is_empty(), "{:?}", validation.errors);
}

/// The schema views count the labels, edge types and property keys of the
/// compacted data next to those written since.
#[test]
fn schema_views_see_the_compacted_data() {
    use grafeo_engine::SchemaInfo;

    let mut db = GrafeoDB::new_in_memory();
    let alix = person(&db, "Alix");
    let amsterdam = db
        .create_node_with_props(&["City"], [("population", Value::Int64(921_000))])
        .unwrap();
    db.create_edge(alix, amsterdam, "LIVES_IN").unwrap();
    db.compact().unwrap();
    let gus = person(&db, "Gus");
    db.create_edge_with_props(alix, gus, "KNOWS", [("since", Value::Int64(2020))])
        .unwrap();

    // Person and City, LIVES_IN and KNOWS, name, population and since.
    assert_eq!(
        (
            db.label_count().unwrap(),
            db.edge_type_count().unwrap(),
            db.property_key_count().unwrap()
        ),
        (2, 2, 3)
    );
    let stats = db.detailed_stats().unwrap();
    assert_eq!(
        (
            stats.label_count,
            stats.edge_type_count,
            stats.property_key_count
        ),
        (2, 2, 3)
    );

    let SchemaInfo::Lpg(schema) = db.schema().unwrap() else {
        panic!("expected an LPG schema");
    };
    let mut labels: Vec<_> = schema
        .labels
        .iter()
        .map(|label| (label.name.as_str(), label.count))
        .collect();
    labels.sort_unstable();
    assert_eq!(labels, [("City", 1), ("Person", 2)]);
    let mut edge_types: Vec<_> = schema
        .edge_types
        .iter()
        .map(|edge_type| (edge_type.name.as_str(), edge_type.count))
        .collect();
    edge_types.sort_unstable();
    assert_eq!(edge_types, [("KNOWS", 1), ("LIVES_IN", 1)]);
    let mut keys = schema.property_keys;
    keys.sort_unstable();
    assert_eq!(keys, ["name", "population", "since"]);
}

/// The schema and the counts show committed data only: an open
/// transaction's node and edge appear once it commits, and never after it
/// rolls back.
#[test]
fn schema_counts_committed_data_only() {
    use grafeo_engine::SchemaInfo;

    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
        .unwrap();
    let counts = |db: &GrafeoDB| {
        let SchemaInfo::Lpg(schema) = db.schema().unwrap() else {
            panic!("expected an LPG schema");
        };
        let people = schema
            .labels
            .iter()
            .find(|label| label.name == "Person")
            .map_or(0, |label| label.count);
        let knows = schema
            .edge_types
            .iter()
            .find(|edge_type| edge_type.name == "KNOWS")
            .map_or(0, |edge_type| edge_type.count);
        let stats = db.detailed_stats().unwrap();
        (people, knows, stats.node_count, stats.edge_count)
    };
    let insert = "MATCH (a:Person {name: 'Alix'}) INSERT (a)-[:KNOWS]->(:Person {name: 'Django'})";

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute(insert).unwrap();
    assert_eq!(counts(&db), (2, 1, 2, 1), "while the transaction is open");
    session.rollback().unwrap();
    assert_eq!(counts(&db), (2, 1, 2, 1), "after a rollback");

    session.begin_transaction().unwrap();
    session.execute(insert).unwrap();
    session.commit().unwrap();
    assert_eq!(counts(&db), (3, 2, 3, 2), "after the commit");
}

#[test]
fn direct_reads_on_an_external_store() {
    let store = Arc::new(RowGroupStore::new());
    let alix = store.create_node(&["Person"]);
    store.set_node_property(alix, "name", Value::from("Alix"));
    let gus = store.create_node(&["Person"]);
    let knows = store.create_edge(alix, gus, "KNOWS");
    for _ in 0..3 {
        store.new_epoch();
    }
    let db = GrafeoDB::with_store(
        Arc::clone(&store) as Arc<dyn GraphStoreMut>,
        Config::default(),
    )
    .unwrap();

    assert_eq!(name(&db, alix), Some(Value::from("Alix")));
    assert_eq!(db.get_edge(knows).unwrap().map(|edge| edge.dst), Some(gus));
    assert_eq!(
        db.get_node_labels(alix).unwrap(),
        Some(vec!["Person".to_string()])
    );
    assert_eq!(
        (
            db.label_count().unwrap(),
            db.edge_type_count().unwrap(),
            db.property_key_count().unwrap()
        ),
        (1, 1, 1)
    );
    assert_eq!(
        db.graph("default")
            .unwrap()
            .get_node(alix)
            .unwrap()
            .map(|node| node.id),
        Some(alix)
    );
    // The store's epoch, not a fresh count from zero.
    let before = db.current_epoch();
    assert_eq!(before, store.current_epoch());
    db.execute("INSERT (:Person {name: 'Vincent'})").unwrap();
    assert!(db.current_epoch() > before);

    // Named graphs need the built-in store.
    assert!(db.create_graph("model").is_err());
    assert_eq!(db.list_graphs(), Vec::<String>::new());
}

/// A graph selected with `set_current_graph` and then dropped elsewhere:
/// reads fail with the graph's absence instead of reading the default
/// graph's node with the same id.
#[test]
fn reads_of_a_dropped_selected_graph_fail() {
    let db = GrafeoDB::new_in_memory();
    let alix = person(&db, "Alix");
    db.create_graph("model").unwrap();
    db.set_current_graph(Some("model")).unwrap();
    let component = db
        .create_node_with_props(&["Component"], [("name", Value::from("Alix"))])
        .unwrap();
    // Every graph numbers its nodes from 0.
    assert_eq!(component, alix);
    db.session().execute("DROP GRAPH model").unwrap();

    // Not the default graph's node with the same id, and not an empty
    // answer either: the graph the reads name is gone.
    let gone = |what: &str, error: grafeo_common::utils::error::Error| {
        assert!(
            error.to_string().contains("Graph 'model' does not exist"),
            "{what}: {error}"
        );
    };
    gone("get_node", db.get_node(component).unwrap_err());
    gone(
        "get_node_labels",
        db.get_node_labels(component).unwrap_err(),
    );
    gone(
        "find_nodes_by_property",
        db.find_nodes_by_property("name", &Value::from("Alix"))
            .unwrap_err(),
    );

    db.set_current_graph(None).unwrap();
    assert_eq!(name(&db, alix), Some(Value::from("Alix")));
}

/// A view as of the current epoch keeps the graph it reads while it lives,
/// in a database that keeps no history: reads of the store at its read view
/// see one state, across later commits and a collect. The bindings read
/// through it where they make several store calls for one answer.
#[test]
fn reads_at_a_view_of_now_see_one_state_while_it_lives() {
    use grafeo_core::graph::{Direction, GraphStore};

    let db = GrafeoDB::new_in_memory();
    let alix = person(&db, "Alix");
    let gus = person(&db, "Gus");
    db.create_edge(alix, gus, "KNOWS").unwrap();
    let store = db.store();

    let now = db.as_of(db.current_epoch()).unwrap();
    let view = now.read_view();
    db.execute("MATCH (p:Person {name: 'Gus'}) DETACH DELETE p")
        .unwrap();
    db.execute("MATCH (p:Person {name: 'Alix'}) SET p.name = 'Mia'")
        .unwrap();
    db.gc();
    assert_eq!(
        store.nodes_by_label("Person", view).unwrap(),
        vec![alix, gus]
    );
    assert_eq!(
        store.node_property(alix, "name", view).unwrap(),
        Some(Value::from("Alix")),
        "the pinned view keeps the value it read"
    );
    assert_eq!(
        store
            .edges(alix, Direction::Outgoing, &[], view)
            .unwrap()
            .len(),
        1,
        "and the edge the delete took"
    );

    drop(now);
    db.gc();
    let later = db.as_of(db.current_epoch()).unwrap();
    assert_eq!(
        store.nodes_by_label("Person", later.read_view()).unwrap(),
        vec![alix]
    );
    assert_eq!(
        store
            .node_property(alix, "name", later.read_view())
            .unwrap(),
        Some(Value::from("Mia"))
    );
    assert_eq!(
        store
            .edges(alix, Direction::Outgoing, &[], later.read_view())
            .unwrap(),
        Vec::<(NodeId, grafeo_common::types::EdgeId)>::new()
    );
}
