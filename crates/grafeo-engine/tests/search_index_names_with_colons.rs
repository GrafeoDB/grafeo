//! Text and vector indexes are told apart by their label and property, also
//! when a backquoted name holds a `:`. The store keyed them by the joined
//! `label:property`, so the indexes (`a`, `b:emb`) and (`a:b`, `emb`) shared
//! one key and the second replaced the first, and the catalog split the key
//! on its first `:`, so a reopen gave the index of label `a:b` back as one of
//! label `a`.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main search_index_names_with_colons::
//! ```

#![cfg(all(
    feature = "gql",
    feature = "vector-index",
    feature = "text-index",
    feature = "grafeo-file"
))]

use std::path::Path;

use grafeo_common::types::{NodeId, Value};
use grafeo_engine::{Config, GrafeoDB};

/// Alix (label `a`, vector `b:emb`) and Mia (label `a`, text `b:bio`), Gus
/// (label `a:b`, vector `emb`) and Vincent (label `a:b`, text `bio`), and the
/// four indexes on them.
fn build(db: &GrafeoDB) {
    db.execute(
        "INSERT (:a {name: 'Alix', `b:emb`: vector([1.0, 0.0])}), \
         (:a {name: 'Mia', `b:bio`: 'the canals of Amsterdam'}), \
         (:`a:b` {name: 'Gus', emb: vector([0.0, 1.0])}), \
         (:`a:b` {name: 'Vincent', bio: 'the bridges of Paris'})",
    )
    .unwrap();
    db.create_vector_index("a", "b:emb", Some(2), Some("euclidean"), None, None, None)
        .unwrap();
    db.create_vector_index("a:b", "emb", Some(2), Some("euclidean"), None, None, None)
        .unwrap();
    db.create_text_index("a", "b:bio").unwrap();
    db.create_text_index("a:b", "bio").unwrap();
}

/// The name of node `id`.
fn name(db: &GrafeoDB, id: NodeId) -> String {
    match db.get_node(id).unwrap().unwrap().get_property("name") {
        Some(Value::String(name)) => name.to_string(),
        other => panic!("node {id:?} has no name: {other:?}"),
    }
}

/// The names a vector search through the index on `label` and `property`
/// finds, nearest first.
fn nearest(db: &GrafeoDB, label: &str, property: &str) -> Vec<String> {
    db.vector_search(label, property, &[1.0, 0.0], 10, None, None)
        .unwrap()
        .into_iter()
        .map(|(id, _)| name(db, id))
        .collect()
}

/// The names a text search for `query` through the index on `label` and
/// `property` finds, best first.
fn matching(db: &GrafeoDB, label: &str, property: &str, query: &str) -> Vec<String> {
    db.text_search(label, property, query, 10, None)
        .unwrap()
        .into_iter()
        .map(|(id, _)| name(db, id))
        .collect()
}

/// Each of the four indexes finds only its own node.
fn assert_separate(db: &GrafeoDB) {
    assert_eq!(nearest(db, "a", "b:emb"), vec!["Alix"]);
    assert_eq!(nearest(db, "a:b", "emb"), vec!["Gus"]);
    assert_eq!(matching(db, "a", "b:bio", "canals"), vec!["Mia"]);
    assert_eq!(matching(db, "a:b", "bio", "bridges"), vec!["Vincent"]);
    assert_eq!(
        matching(db, "a", "b:bio", "bridges"),
        Vec::<String>::new(),
        "the text index of (a, b:bio) holds no text of (a:b, bio)"
    );
}

/// The indexes (`a`, `b:emb`) and (`a:b`, `emb`) are two indexes, and so
/// are (`a`, `b:bio`) and (`a:b`, `bio`).
#[test]
fn indexes_whose_joined_names_are_equal_are_separate() {
    let db = GrafeoDB::new_in_memory();
    build(&db);
    assert_separate(&db);

    assert!(db.drop_vector_index("a", "b:emb").unwrap());
    assert!(db.drop_text_index("a", "b:bio").unwrap());
    assert_eq!(
        nearest(&db, "a:b", "emb"),
        vec!["Gus"],
        "dropping (a, b:emb) leaves (a:b, emb)"
    );
    assert_eq!(matching(&db, "a:b", "bio", "bridges"), vec!["Vincent"]);
}

/// Taking label `a` off a node leaves it in the indexes of label `a:b`.
#[test]
fn removing_label_a_leaves_the_indexes_of_label_a_b() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (:a:`a:b` {name: 'Jules', emb: vector([1.0, 0.0]), bio: 'the trams of Prague'})",
    )
    .unwrap();
    db.create_vector_index("a:b", "emb", Some(2), Some("euclidean"), None, None, None)
        .unwrap();
    db.create_text_index("a:b", "bio").unwrap();
    db.create_vector_index("a", "emb", Some(2), Some("euclidean"), None, None, None)
        .unwrap();

    db.execute("MATCH (n:a) REMOVE n:a").unwrap();
    assert_eq!(nearest(&db, "a:b", "emb"), vec!["Jules"]);
    assert_eq!(matching(&db, "a:b", "bio", "trams"), vec!["Jules"]);
    assert_eq!(
        nearest(&db, "a", "emb"),
        Vec::<String>::new(),
        "the index of label a lost Jules"
    );
}

/// A rolled-back transaction that took the labels and changed the values
/// leaves each index as it was.
#[test]
fn a_rollback_restores_each_index() {
    let db = GrafeoDB::new_in_memory();
    build(&db);

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("MATCH (n:`a:b`) REMOVE n:`a:b`").unwrap();
    session
        .execute(
            "MATCH (n:a) SET n.`b:emb` = vector([0.0, 1.0]), n.`b:bio` = 'the trams of Prague'",
        )
        .unwrap();
    session.rollback().unwrap();

    assert_separate(&db);
    assert_eq!(matching(&db, "a", "b:bio", "trams"), Vec::<String>::new());
}

/// A checkpoint and a reopen give each index back with its own label and
/// property.
#[test]
fn a_reopen_keeps_each_index() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("barcelona.grafeo");
    let open = |path: &Path| GrafeoDB::with_config(Config::persistent(path)).unwrap();

    let db = open(&path);
    build(&db);
    db.close().unwrap();

    let db = open(&path);
    assert_separate(&db);
    db.close().unwrap();
}
