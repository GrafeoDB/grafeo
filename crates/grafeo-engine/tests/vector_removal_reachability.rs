//! Taking a vector out of an index keeps every other vector findable
//! (#600): the nodes that linked to it are linked past it, whichever write
//! takes it out. Each removal path of the issue, then a search for more
//! nodes than the index holds returns every remaining node.

#![cfg(all(feature = "vector-index", feature = "lpg", feature = "gql"))]

use std::collections::HashMap;

use grafeo_common::types::{NodeId, PropertyKey, Value};
use grafeo_engine::GrafeoDB;

/// The issue's data: eight items on a line, which the neighbor heuristic
/// links as a chain, so each item is the only link between its two halves.
fn items_on_a_line() -> (GrafeoDB, Vec<NodeId>) {
    let db = GrafeoDB::new_in_memory();
    let ids = (0..8)
        .map(|i| {
            let mut properties = HashMap::new();
            properties.insert(PropertyKey::new("key"), Value::Int64(i));
            properties.insert(
                PropertyKey::new("embedding"),
                Value::Vector(vec![1.0, i as f32 / 100.0, 0.0, 0.0].into()),
            );
            db.create_node_with_props(&["Item"], properties).unwrap()
        })
        .collect();
    db.create_vector_index(
        "Item",
        "embedding",
        Some(4),
        Some("euclidean"),
        None,
        None,
        None,
    )
    .unwrap();
    (db, ids)
}

/// The nodes a search for more nodes than the index holds finds, from the
/// start of the line and from its end.
fn found(db: &GrafeoDB) -> Vec<Vec<NodeId>> {
    [[1.0, 0.0, 0.0, 0.0], [1.0, 0.07, 0.0, 0.0]]
        .iter()
        .map(|query| {
            let mut ids: Vec<NodeId> = db
                .vector_search("Item", "embedding", query, 19, None, None)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            ids.sort_unstable();
            ids
        })
        .collect()
}

/// Asserts that both searches find exactly `expected`.
fn assert_all_found(db: &GrafeoDB, expected: &[NodeId], path: &str) {
    for ids in found(db) {
        assert_eq!(ids, expected, "{path}: every remaining vector is found");
    }
}

/// The ids without the fourth item, which the paths take out.
fn without_fourth(ids: &[NodeId]) -> Vec<NodeId> {
    let mut rest: Vec<NodeId> = ids.iter().copied().filter(|id| *id != ids[3]).collect();
    rest.sort_unstable();
    rest
}

#[test]
fn deleting_a_node_keeps_the_other_vectors_findable() {
    let (db, ids) = items_on_a_line();
    assert!(db.delete_node(ids[3]).unwrap());
    assert_all_found(&db, &without_fourth(&ids), "delete_node");

    let (db, ids) = items_on_a_line();
    db.execute("MATCH (n:Item {key: 3}) DETACH DELETE n")
        .unwrap();
    assert_all_found(&db, &without_fourth(&ids), "DETACH DELETE");
}

#[test]
fn removing_the_vector_keeps_the_other_vectors_findable() {
    let (db, ids) = items_on_a_line();
    assert!(db.remove_node_property(ids[3], "embedding").unwrap());
    assert_all_found(&db, &without_fourth(&ids), "remove_node_property");

    let (db, ids) = items_on_a_line();
    db.execute("MATCH (n:Item {key: 3}) REMOVE n.embedding")
        .unwrap();
    assert_all_found(&db, &without_fourth(&ids), "REMOVE n.embedding");
}

#[test]
fn replacing_all_properties_keeps_the_other_vectors_findable() {
    let (db, ids) = items_on_a_line();
    db.execute("MATCH (n:Item {key: 3}) SET n = {key: 3, name: 'Alix'}")
        .unwrap();
    assert_all_found(&db, &without_fourth(&ids), "SET n = map");

    let (db, ids) = items_on_a_line();
    let mut row = HashMap::new();
    row.insert(PropertyKey::new("key"), Value::Int64(3));
    row.insert(PropertyKey::new("name"), Value::from("Gus"));
    db.upsert_nodes(&["Item"], "key", vec![row], true).unwrap();
    assert_all_found(&db, &without_fourth(&ids), "upsert_nodes with replace");
}

#[test]
fn a_rolled_back_removal_leaves_every_vector_findable() {
    let (db, ids) = items_on_a_line();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (n:Item {key: 3}) REMOVE n.embedding")
        .unwrap();
    session.rollback().unwrap();
    let mut all = ids.clone();
    all.sort_unstable();
    assert_all_found(&db, &all, "a rolled-back REMOVE");

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (n:Item {key: 3}) DETACH DELETE n")
        .unwrap();
    session.rollback().unwrap();
    assert_all_found(&db, &all, "a rolled-back DETACH DELETE");

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.savepoint("before").unwrap();
    session
        .execute("MATCH (n:Item {key: 3}) REMOVE n.embedding")
        .unwrap();
    session.rollback_to_savepoint("before").unwrap();
    session.commit().unwrap();
    assert_all_found(&db, &all, "a REMOVE rolled back to a savepoint");
}

/// Many removals, one after the other, through the database: every vector
/// left is found after each one.
#[test]
fn removing_most_vectors_one_by_one_keeps_the_rest_findable() {
    let db = GrafeoDB::new_in_memory();
    let mut ids: Vec<NodeId> = (0..88)
        .map(|i| {
            // Two lines and a cluster: chains that a lost link cuts.
            let vector = match i % 3 {
                0 => vec![i as f32 * 0.3, 0.0, 3.0],
                1 => vec![0.0, i as f32 * 0.3, 19.0],
                _ => vec![19.0 + (i % 7) as f32 * 0.01, 19.0, 0.0],
            };
            let mut properties = HashMap::new();
            properties.insert(PropertyKey::new("embedding"), Value::Vector(vector.into()));
            db.create_node_with_props(&["Item"], properties).unwrap()
        })
        .collect();
    db.create_vector_index(
        "Item",
        "embedding",
        Some(3),
        Some("euclidean"),
        Some(4),
        None,
        None,
    )
    .unwrap();
    // Every third one, then every other one of the rest, in a scrambled order.
    let mut order: Vec<usize> = (0..ids.len()).collect();
    order.sort_by_key(|i| (i * 19) % 88);
    for &position in order.iter().take(66) {
        let removed = ids[position];
        if position % 2 == 0 {
            db.delete_node(removed).unwrap();
        } else {
            db.remove_node_property(removed, "embedding").unwrap();
        }
        ids[position] = NodeId::new(u64::MAX);
        let mut expected: Vec<NodeId> = ids
            .iter()
            .copied()
            .filter(|id| *id != NodeId::new(u64::MAX))
            .collect();
        expected.sort_unstable();
        for query in [[0.0, 0.0, 3.0], [0.0, 19.0, 19.0], [19.0, 19.0, 0.0]] {
            let mut found: Vec<NodeId> = db
                .vector_search("Item", "embedding", &query, 88, None, None)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            found.sort_unstable();
            assert_eq!(found, expected, "after removing {removed:?}");
        }
    }
}
