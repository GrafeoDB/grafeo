//! `compact()` keeps every edge id on its own edge: reads by id, queries and
//! deletes after a compaction reach the edge the id was given to.
//!
//! ```bash
//! cargo test -p grafeo-engine --features "compact-store lpg gql" --test compact_edge_ids
//! ```

#![cfg(all(feature = "compact-store", feature = "lpg", feature = "gql"))]

use grafeo_common::types::{EdgeId, NodeId, PropertyKey, Value};
use grafeo_engine::GrafeoDB;

fn rows(db: &GrafeoDB, query: &str) -> Vec<Vec<Value>> {
    db.session().execute(query).unwrap().rows().to_vec()
}

fn person(db: &GrafeoDB, name: &str) -> NodeId {
    let id = db.create_node(&["Person"]).unwrap();
    db.set_node_property(id, "name", Value::from(name)).unwrap();
    id
}

fn knows(db: &GrafeoDB, from: NodeId, to: NodeId, w: i64) -> EdgeId {
    let id = db.create_edge(from, to, "KNOWS").unwrap();
    db.set_edge_property(id, "w", Value::Int64(w)).unwrap();
    id
}

/// The endpoints and `w` of the edge `id` names.
fn edge(db: &GrafeoDB, id: EdgeId) -> (NodeId, NodeId, Option<Value>) {
    let edge = db.get_edge(id).unwrap();
    let w = edge.properties.get(&PropertyKey::new("w")).cloned();
    (edge.src, edge.dst, w)
}

/// Alix's edge to Mia is created before her edge to Gus, while Gus was
/// created before Mia: the order of a node's edges differs from the order of
/// their targets. Each id keeps its edge, its property and its delete.
#[test]
fn compact_keeps_each_edge_id_on_its_own_edge() {
    let mut db = GrafeoDB::new_in_memory();
    let alix = person(&db, "Alix");
    let gus = person(&db, "Gus");
    let mia = person(&db, "Mia");
    let to_mia = knows(&db, alix, mia, 3);
    let to_gus = knows(&db, alix, gus, 19);
    db.compact().unwrap();

    assert_eq!(edge(&db, to_mia), (alix, mia, Some(Value::Int64(3))));
    assert_eq!(edge(&db, to_gus), (alix, gus, Some(Value::Int64(19))));
    let id = |edge: EdgeId| Value::Int64(i64::try_from(edge.as_u64()).unwrap());
    assert_eq!(
        rows(
            &db,
            "MATCH (:Person {name: 'Alix'})-[r:KNOWS]->(b) RETURN id(r), b.name, r.w ORDER BY r.w"
        ),
        [
            vec![id(to_mia), Value::from("Mia"), Value::Int64(3)],
            vec![id(to_gus), Value::from("Gus"), Value::Int64(19)],
        ]
    );

    assert!(db.delete_edge(to_mia).unwrap());
    assert_eq!(
        rows(
            &db,
            "MATCH (:Person {name: 'Alix'})-[r:KNOWS]->(b) RETURN b.name, r.w"
        ),
        [vec![Value::from("Gus"), Value::Int64(19)]],
        "deleting the edge to Mia deletes that edge only"
    );
}

/// Parallel edges keep their own ids when reached from their target.
#[test]
fn compact_keeps_parallel_edges_apart_on_incoming_traversal() {
    let mut db = GrafeoDB::new_in_memory();
    let vincent = person(&db, "Vincent");
    let jules = person(&db, "Jules");
    let first = knows(&db, vincent, jules, 3);
    let second = knows(&db, vincent, jules, 88);
    db.compact().unwrap();

    let id = |edge: EdgeId| Value::Int64(i64::try_from(edge.as_u64()).unwrap());
    assert_eq!(
        rows(
            &db,
            "MATCH (:Person {name: 'Jules'})<-[r:KNOWS]-(a) RETURN id(r), r.w ORDER BY r.w"
        ),
        [
            vec![id(first), Value::Int64(3)],
            vec![id(second), Value::Int64(88)],
        ]
    );
}
