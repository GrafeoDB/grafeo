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
use grafeo_core::graph::lpg::LpgStore;
use grafeo_core::graph::traits::GraphStoreMut;
use grafeo_engine::{Config, GrafeoDB, SchemaInfo};

fn name(db: &GrafeoDB, id: NodeId) -> Option<Value> {
    db.get_node(id)
        .and_then(|node| node.get_property("name").cloned())
}

fn person(db: &GrafeoDB, name: &str) -> NodeId {
    db.create_node_with_props(&["Person"], [("name", Value::from(name))])
        .unwrap()
}

#[cfg(feature = "compact-store")]
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
        db.get_edge(knows).map(|edge| (edge.src, edge.dst)),
        Some((alix, gus))
    );
    assert_eq!(db.get_node_labels(gus), Some(vec!["Person".to_string()]));
    assert_eq!((db.node_count(), db.edge_count()), (3, 1));
    assert_eq!(
        db.find_nodes_by_property("name", &Value::from("Alix")),
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
    assert_eq!((db.info().node_count, db.info().edge_count), (3, 2));
    let stats = db.detailed_stats();
    assert_eq!((stats.node_count, stats.edge_count), (3, 2));
    let validation = db.validate();
    assert!(validation.errors.is_empty(), "{:?}", validation.errors);
}

/// The schema views count the labels, edge types and property keys of the
/// compacted data next to those written since.
#[cfg(feature = "compact-store")]
#[test]
fn schema_views_see_the_compacted_data() {
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
            db.label_count(),
            db.edge_type_count(),
            db.property_key_count()
        ),
        (2, 2, 3)
    );
    let stats = db.detailed_stats();
    assert_eq!(
        (
            stats.label_count,
            stats.edge_type_count,
            stats.property_key_count
        ),
        (2, 2, 3)
    );

    let SchemaInfo::Lpg(schema) = db.schema() else {
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

#[test]
fn direct_reads_on_an_external_store() {
    let store = Arc::new(LpgStore::new().unwrap());
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
    assert_eq!(db.get_edge(knows).map(|edge| edge.dst), Some(gus));
    assert_eq!(db.get_node_labels(alix), Some(vec!["Person".to_string()]));
    assert_eq!(
        (
            db.label_count(),
            db.edge_type_count(),
            db.property_key_count()
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
/// reads find nothing instead of the default graph's node with the same id.
#[test]
fn reads_of_a_dropped_selected_graph_find_nothing() {
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

    assert!(db.get_node(component).is_none());
    assert!(db.get_node_labels(component).is_none());
    assert!(
        db.find_nodes_by_property("name", &Value::from("Alix"))
            .is_empty()
    );

    db.set_current_graph(None).unwrap();
    assert_eq!(name(&db, alix), Some(Value::from("Alix")));
}
