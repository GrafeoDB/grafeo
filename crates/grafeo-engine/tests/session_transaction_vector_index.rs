//! Regression tests for transaction-buffered session vector index updates.

#![cfg(feature = "vector-index")]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

fn vec3(x: f32, y: f32, z: f32) -> Value {
    Value::Vector(vec![x, y, z].into())
}

fn create_empty_doc_index(db: &GrafeoDB) {
    db.create_vector_index("Doc", "emb", Some(3), Some("cosine"), None, None, None)
        .expect("create empty vector index");
}

fn result_ids(db: &GrafeoDB, query: &[f32], k: usize) -> Vec<u64> {
    db.vector_search("Doc", "emb", query, k, None, None)
        .expect("vector search")
        .into_iter()
        .map(|(id, _)| id.as_u64())
        .collect()
}

#[test]
fn rolled_back_create_node_with_vector_props_is_not_searchable() {
    let db = GrafeoDB::new_in_memory();
    create_empty_doc_index(&db);

    let mut session = db.session();
    session.begin_transaction().expect("begin");
    let rolled_back = session
        .create_node_with_props(&["Doc"], [("emb", vec3(1.0, 0.0, 0.0))])
        .expect("create vector node");
    session.rollback().expect("rollback");

    let ids = result_ids(&db, &[1.0, 0.0, 0.0], 10);
    assert!(
        !ids.contains(&rolled_back.as_u64()),
        "rolled-back vector node must not be indexed"
    );
    assert!(
        ids.is_empty(),
        "empty index should remain empty after rollback"
    );
}

#[test]
fn commit_applies_create_update_and_delete_vector_intents() {
    let db = GrafeoDB::new_in_memory();
    create_empty_doc_index(&db);

    let mut session = db.session();
    session.begin_transaction().expect("begin create");
    let node = session
        .create_node_with_props(&["Doc"], [("emb", vec3(1.0, 0.0, 0.0))])
        .expect("create vector node");
    session.commit().expect("commit create");
    assert!(result_ids(&db, &[1.0, 0.0, 0.0], 10).contains(&node.as_u64()));

    session.begin_transaction().expect("begin update");
    session
        .set_node_property(node, "emb", vec3(0.0, 1.0, 0.0))
        .expect("update vector property");
    session.commit().expect("commit update");
    let updated = db
        .vector_search("Doc", "emb", &[0.0, 1.0, 0.0], 1, None, None)
        .expect("search updated vector");
    assert_eq!(updated.first().map(|(id, _)| *id), Some(node));
    assert!(
        updated
            .first()
            .is_some_and(|(_, distance)| *distance < 0.001),
        "updated vector should be indexed at the committed value"
    );

    session.begin_transaction().expect("begin delete");
    assert!(
        session.delete_node(node),
        "delete should see committed node"
    );
    session.commit().expect("commit delete");
    assert!(
        result_ids(&db, &[0.0, 1.0, 0.0], 10).is_empty(),
        "committed delete should remove the node from the vector index"
    );
}

#[test]
fn rollback_to_savepoint_truncates_later_vector_intents() {
    let db = GrafeoDB::new_in_memory();
    create_empty_doc_index(&db);

    let mut session = db.session();
    session.begin_transaction().expect("begin");
    let kept = session
        .create_node_with_props(&["Doc"], [("emb", vec3(1.0, 0.0, 0.0))])
        .expect("create kept node");
    session.savepoint("after_kept").expect("savepoint");
    let discarded = session
        .create_node_with_props(&["Doc"], [("emb", vec3(0.0, 1.0, 0.0))])
        .expect("create discarded node");
    session
        .rollback_to_savepoint("after_kept")
        .expect("rollback to savepoint");
    session.commit().expect("commit");

    let ids = result_ids(&db, &[0.5, 0.5, 0.0], 10);
    assert!(
        ids.contains(&kept.as_u64()),
        "pre-savepoint node should be indexed"
    );
    assert!(
        !ids.contains(&discarded.as_u64()),
        "post-savepoint node intent should be discarded"
    );
    assert_eq!(ids.len(), 1);
}

#[test]
fn commit_conflict_loser_discards_vector_intents() {
    let db = GrafeoDB::new_in_memory();
    create_empty_doc_index(&db);

    let mut seed = db.session();
    seed.begin_transaction().expect("begin seed");
    let node = seed
        .create_node_with_props(&["Doc"], [("emb", vec3(1.0, 0.0, 0.0))])
        .expect("seed node");
    seed.commit().expect("commit seed");

    let mut winner = db.session();
    let mut loser = db.session();
    winner.begin_transaction().expect("begin winner");
    loser.begin_transaction().expect("begin loser");

    winner
        .set_node_property(node, "emb", vec3(0.0, 1.0, 0.0))
        .expect("winner update");
    winner.commit().expect("winner commit");

    loser
        .set_node_property(node, "emb", vec3(0.0, 0.0, 1.0))
        .expect("loser update before commit conflict");
    assert!(
        loser.commit().is_err(),
        "loser should fail at commit due to write-write conflict"
    );

    let current = db.get_node(node).expect("node remains visible");
    assert_eq!(
        current.properties.get(&"emb".into()),
        Some(&vec3(0.0, 1.0, 0.0)),
        "conflict rollback should restore the winner's graph value"
    );

    let results = db
        .vector_search("Doc", "emb", &[0.0, 1.0, 0.0], 1, None, None)
        .expect("search winner vector");
    assert_eq!(results.first().map(|(id, _)| *id), Some(node));
    assert!(
        results
            .first()
            .is_some_and(|(_, distance)| *distance < 0.001),
        "conflict loser vector intent should not replace the committed index value"
    );
}
