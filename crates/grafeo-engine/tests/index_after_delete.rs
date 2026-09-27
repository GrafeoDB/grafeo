//! Indexes after a node is deleted.
//!
//! A deleted node used to stay in the property index (`find_nodes_by_property`
//! kept returning it, and a re-created node with the same key came back twice)
//! and in vector indexes (a vector search could still return it). Rolling the
//! delete back must restore both.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test index_after_delete
//! ```

#![allow(missing_docs)]

use grafeo_common::types::{NodeId, Value};
use grafeo_engine::GrafeoDB;

fn find(db: &GrafeoDB, value: &str) -> Vec<NodeId> {
    let mut ids = db.find_nodes_by_property("id", &Value::from(value));
    ids.sort_unstable();
    ids
}

#[test]
fn deleted_node_leaves_the_property_index() {
    // The downstream repro: create, DETACH DELETE through a query, re-create.
    let db = GrafeoDB::new_in_memory();
    db.create_property_index("id");
    let session = db.session();
    session
        .execute_cypher("CREATE (:Graph:File {id: 'a'})")
        .unwrap();
    let first = find(&db, "a");
    assert_eq!(first.len(), 1);

    session
        .execute_cypher("MATCH (n:Graph) WHERE n.id = 'a' DETACH DELETE n")
        .unwrap();
    assert!(find(&db, "a").is_empty(), "deleted node must not be found");

    let second = db.create_node_with_props(&["Graph", "File"], [("id", Value::from("a"))]);
    assert_eq!(find(&db, "a"), vec![second]);
}

#[test]
fn rolled_back_delete_is_found_again() {
    let db = GrafeoDB::new_in_memory();
    db.create_property_index("id");
    let node = db.create_node_with_props(&["Graph"], [("id", Value::from("a"))]);

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute_cypher("MATCH (n:Graph {id: 'a'}) DETACH DELETE n")
        .unwrap();
    session.rollback().unwrap();

    assert_eq!(find(&db, "a"), vec![node]);
}

#[test]
fn uncommitted_nodes_are_not_returned_by_the_lookup_api() {
    let db = GrafeoDB::new_in_memory();
    db.create_property_index("id");
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute_cypher("CREATE (:Graph {id: 'a'})").unwrap();
    assert!(
        find(&db, "a").is_empty(),
        "another transaction's uncommitted node must not be visible"
    );
    session.commit().unwrap();
    assert_eq!(find(&db, "a").len(), 1);
}

#[cfg(feature = "vector-index")]
mod vectors {
    use super::*;

    fn vector_db(quantization: Option<&str>) -> (GrafeoDB, NodeId, NodeId) {
        let db = GrafeoDB::new_in_memory();
        let near = db.create_node_with_props(
            &["Doc"],
            [
                ("id", Value::from("near")),
                ("emb", Value::Vector(vec![1.0f32, 0.0].into())),
            ],
        );
        let far = db.create_node_with_props(
            &["Doc"],
            [
                ("id", Value::from("far")),
                ("emb", Value::Vector(vec![0.0f32, 1.0].into())),
            ],
        );
        db.create_vector_index(
            "Doc",
            "emb",
            Some(2),
            Some("euclidean"),
            None,
            None,
            quantization,
        )
        .unwrap();
        (db, near, far)
    }

    fn search(db: &GrafeoDB) -> Vec<NodeId> {
        db.vector_search("Doc", "emb", &[1.0, 0.0], 5, None, None)
            .unwrap()
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }

    #[test]
    fn deleted_node_is_not_returned_by_vector_search() {
        for quantization in [None, Some("scalar")] {
            let (db, near, far) = vector_db(quantization);
            db.session()
                .execute_cypher("MATCH (n:Doc {id: 'near'}) DETACH DELETE n")
                .unwrap();
            let found = search(&db);
            assert!(!found.contains(&near), "{quantization:?}: {found:?}");
            assert_eq!(found, vec![far], "{quantization:?}");
        }
    }

    #[test]
    fn rolled_back_delete_is_returned_by_vector_search_again() {
        let (db, near, far) = vector_db(None);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session
            .execute_cypher("MATCH (n:Doc {id: 'near'}) DETACH DELETE n")
            .unwrap();
        session.rollback().unwrap();
        assert_eq!(search(&db), vec![near, far]);
    }
}
