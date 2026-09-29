//! Vector and text indexes follow every write: a query that inserts or sets
//! an indexed property, or gives a node an indexed label, is found through
//! the index afterwards, and one that removes them is not.

use grafeo_common::types::{NodeId, Value};
use grafeo_engine::GrafeoDB;

/// The `id` properties of `nodes`, sorted.
fn ids(db: &GrafeoDB, nodes: impl IntoIterator<Item = NodeId>) -> Vec<i64> {
    let mut ids: Vec<i64> = nodes
        .into_iter()
        .map(|node| match db.get_node(node).unwrap().get_property("id") {
            Some(Value::Int64(id)) => *id,
            other => panic!("node {node:?} has id {other:?}"),
        })
        .collect();
    ids.sort_unstable();
    ids
}

#[cfg(feature = "vector-index")]
#[test]
fn query_writes_reach_the_vector_index() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Doc {id: 1, emb: vector([1.0, 0.0])})")
        .unwrap();
    db.execute("CREATE VECTOR INDEX doc_emb ON :Doc(emb)")
        .unwrap();
    let search = |query: &[f32]| {
        db.vector_search("Doc", "emb", query, 10, None, None)
            .unwrap()
    };
    let nearest = |query: &[f32]| ids(&db, search(query).first().map(|(node, _)| *node));
    let found = |query: &[f32]| ids(&db, search(query).into_iter().map(|(node, _)| node));

    // Written after the index was created: a new node, a node that gets the
    // label, and a changed vector.
    db.execute("INSERT (:Doc {id: 2, emb: vector([0.0, 1.0])})")
        .unwrap();
    db.execute("INSERT (:Draft {id: 3, emb: vector([0.6, 0.8])})")
        .unwrap();
    db.execute("MATCH (n:Draft) SET n:Doc").unwrap();
    db.execute("MATCH (n:Doc {id: 1}) SET n.emb = vector([-1.0, 0.0])")
        .unwrap();
    assert_eq!(found(&[0.0, 1.0]), [1, 2, 3]);
    assert_eq!(nearest(&[0.0, 1.0]), [2]);
    assert_eq!(nearest(&[-1.0, 0.0]), [1]);

    // A lost label and a removed property take the node out of the index.
    db.execute("MATCH (n:Doc {id: 2}) REMOVE n:Doc").unwrap();
    db.execute("MATCH (n:Doc {id: 3}) REMOVE n.emb").unwrap();
    assert_eq!(found(&[0.0, 1.0]), [1]);
}

#[cfg(feature = "text-index")]
#[test]
fn query_label_changes_reach_the_text_index() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Doc {id: 1, body: 'graph database'})")
        .unwrap();
    db.create_text_index("Doc", "body").unwrap();
    let found = || {
        ids(
            &db,
            db.text_search("Doc", "body", "graph", 10)
                .unwrap()
                .into_iter()
                .map(|(node, _)| node),
        )
    };

    db.execute("INSERT (:Draft {id: 2, body: 'graph notes'})")
        .unwrap();
    db.execute("MATCH (n:Draft) SET n:Doc").unwrap();
    assert_eq!(found(), [1, 2]);

    db.execute("MATCH (n:Doc {id: 1}) REMOVE n:Doc").unwrap();
    assert_eq!(found(), [2]);
}

/// A vector index fixes the size of the vectors it indexes: a write of
/// another size is rejected rather than stored unfindable.
#[cfg(feature = "vector-index")]
#[test]
fn a_vector_of_the_wrong_size_is_rejected() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Doc {id: 1, emb: vector([1.0, 0.0])})")
        .unwrap();
    db.execute("CREATE VECTOR INDEX doc_emb ON :Doc(emb)")
        .unwrap();
    db.execute("INSERT (:Draft {id: 2, emb: vector([1.0, 0.0, 0.0])})")
        .unwrap();

    for query in [
        "INSERT (:Doc {id: 3, emb: vector([1.0, 0.0, 0.0])})",
        "MATCH (n:Doc {id: 1}) SET n.emb = vector([1.0, 0.0, 0.0])",
        "MATCH (n:Draft) SET n:Doc",
    ] {
        let err = db.execute(query).unwrap_err().to_string();
        assert!(
            err.contains("has a vector index of 2 dimensions, got a vector of 3"),
            "{query}: {err}"
        );
    }
}
