//! Many identical vectors (#391 follow-up): more exact duplicates than an
//! HNSW neighbor list holds (33 at the default `m = 16`) crowded each other,
//! and every vector inserted after them, out of the graph, so a search
//! silently missed them. Duplicates now share one graph node: every vector is
//! found, the duplicates at their distance, counting toward k in id order, in
//! the direct search and the GQL procedure, after a reopen too.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test vector_duplicates
//! ```

#![cfg(feature = "vector-index")]

use grafeo_common::types::{NodeId, Value};
use grafeo_engine::GrafeoDB;

const DUPLICATES: usize = 40;
const DISTANT: usize = 5;

fn vector(values: [f32; 3]) -> Value {
    Value::Vector(values.to_vec().into())
}

/// `DUPLICATES` docs at `[1, 0, 0]`, then `DISTANT` docs at other angles,
/// and an index on them. Returns the ids of both, ascending.
fn docs(db: &GrafeoDB) -> (Vec<NodeId>, Vec<NodeId>) {
    let create = |values: [f32; 3]| {
        let id = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(id, "emb", vector(values)).unwrap();
        id
    };
    let duplicates: Vec<NodeId> = (0..DUPLICATES).map(|_| create([1.0, 0.0, 0.0])).collect();
    let distant: Vec<NodeId> = (0..DISTANT)
        .map(|i| create([0.0, 1.0, i as f32 * 0.1]))
        .collect();
    db.create_vector_index("Doc", "emb", Some(3), Some("cosine"), None, None, None)
        .expect("create index");
    (duplicates, distant)
}

fn search(db: &GrafeoDB, query: [f32; 3], k: usize) -> Vec<NodeId> {
    db.vector_search("Doc", "emb", &query, k, None, None)
        .unwrap()
        .into_iter()
        .map(|(id, _)| id)
        .collect()
}

/// The ids the GQL procedure returns for `k` from `query`, ascending.
fn procedure(db: &GrafeoDB, query: [f32; 3], k: usize) -> Vec<NodeId> {
    let result = db
        .execute(&format!(
            "CALL grafeo.search.vector('Doc', 'emb', [{}, {}, {}], {k}) YIELD node_id RETURN node_id",
            query[0], query[1], query[2]
        ))
        .unwrap();
    let mut ids: Vec<NodeId> = result
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::Int64(id) => NodeId::new(u64::try_from(*id).unwrap()),
            other => panic!("a node id: {other:?}"),
        })
        .collect();
    ids.sort_unstable();
    ids
}

fn assert_every_vector_found(
    db: &GrafeoDB,
    duplicates: &[NodeId],
    distant: &[NodeId],
    stage: &str,
) {
    let k = DUPLICATES + DISTANT;
    let found = search(db, [1.0, 0.0, 0.0], k);
    assert_eq!(
        found[..DUPLICATES],
        *duplicates,
        "{stage}: the duplicates first, by id"
    );
    let mut rest = found[DUPLICATES..].to_vec();
    rest.sort_unstable();
    assert_eq!(rest, distant, "{stage}: then every distant vector");
    assert_eq!(
        search(db, [1.0, 0.0, 0.0], 3),
        duplicates[..3],
        "{stage}: k counts duplicates"
    );
    assert_eq!(
        search(db, [0.0, 1.0, 0.0], 1),
        distant[..1],
        "{stage}: a distant vector"
    );
    let mut all: Vec<NodeId> = duplicates.iter().chain(distant).copied().collect();
    all.sort_unstable();
    assert_eq!(
        procedure(db, [1.0, 0.0, 0.0], k),
        all,
        "{stage}: the GQL procedure"
    );
}

#[test]
fn many_identical_vectors_are_all_found() {
    let db = GrafeoDB::new_in_memory();
    let (duplicates, distant) = docs(&db);
    assert_every_vector_found(&db, &duplicates, &distant, "in memory");
}

/// Deleting the first duplicate (the graph's node for all of them) keeps the
/// others found; changing a duplicate's vector moves it out of them.
#[test]
fn deleting_and_updating_duplicates_keeps_every_vector_found() {
    let db = GrafeoDB::new_in_memory();
    let (duplicates, distant) = docs(&db);
    db.delete_node(duplicates[0]).unwrap();
    db.set_node_property(duplicates[5], "emb", vector([0.0, 0.0, 1.0]))
        .unwrap();
    let k = DUPLICATES + DISTANT;
    let near: Vec<NodeId> = search(&db, [1.0, 0.0, 0.0], k);
    let expected: Vec<NodeId> = duplicates[1..]
        .iter()
        .copied()
        .filter(|id| *id != duplicates[5])
        .collect();
    assert_eq!(near[..expected.len()], *expected);
    let mut rest = near[expected.len()..].to_vec();
    rest.sort_unstable();
    let mut moved_and_distant = distant.clone();
    moved_and_distant.push(duplicates[5]);
    moved_and_distant.sort_unstable();
    assert_eq!(
        rest, moved_and_distant,
        "then the moved vector and every distant one, not the deleted one"
    );
    assert_eq!(
        search(&db, [0.0, 0.0, 1.0], 1),
        [duplicates[5]],
        "the moved vector"
    );
}

#[cfg(all(feature = "wal", feature = "grafeo-file"))]
#[test]
fn many_identical_vectors_are_all_found_after_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("docs.grafeo");
    let db = GrafeoDB::open(&path).unwrap();
    let (duplicates, distant) = docs(&db);
    assert_every_vector_found(&db, &duplicates, &distant, "before the close");
    db.close().unwrap();
    let db = GrafeoDB::open(&path).unwrap();
    assert_every_vector_found(&db, &duplicates, &distant, "after a reopen");
    db.close().unwrap();
    let db = GrafeoDB::open_read_only(&path).unwrap();
    assert_every_vector_found(&db, &duplicates, &distant, "read-only");
}
