//! A vector index of a compacted database links the vectors written after
//! `compact()` to the compacted ones.
//!
//! After `compact()` the vectors of the compacted nodes are in the columnar
//! base and the overlay takes every write. A vector written then (a new node,
//! or a new value of a compacted node) goes into the same HNSW graph as the
//! compacted ones (an index built over them), linked to its nearest neighbors
//! among both, so a search finds it beside them: through `vector_search` and
//! through a query, which scans the index, after a merge of the overlay into
//! the base, and after a close and reopen. A search without an index of its
//! metric scans the compacted nodes and the later ones.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test overlay_vector_index
//! ```

#![cfg(all(
    feature = "compact-store",
    feature = "lpg",
    feature = "gql",
    feature = "vector-index"
))]

use grafeo_common::types::{NodeId, Value};
use grafeo_engine::GrafeoDB;

/// The people compacted, with their embeddings.
const BASE: [(&str, [f32; 2]); 3] = [
    ("Alix", [3.0, 3.0]),
    ("Gus", [19.0, 19.0]),
    ("Vincent", [88.0, 88.0]),
];

/// The people written after `compact()`, with their embeddings.
const LATER: [(&str, [f32; 2]); 2] = [("Mia", [3.0, 19.0]), ("Jules", [88.0, 3.0])];

/// Everyone, nearest to the origin first (euclidean): Alix (4.2), Mia (19.2),
/// Gus (26.9), Jules (88.1), Vincent (124.5).
const BY_DISTANCE_TO_THE_ORIGIN: [&str; 5] = ["Alix", "Mia", "Gus", "Jules", "Vincent"];

/// Creates a `:Person` with `name` and `embedding`.
fn person(db: &GrafeoDB, name: &str, embedding: [f32; 2]) -> NodeId {
    db.create_node_with_props(
        &["Person"],
        [
            ("name", Value::from(name)),
            ("embedding", Value::Vector(embedding.to_vec().into())),
        ],
    )
    .unwrap()
}

/// The euclidean vector index on `:Person(embedding)`.
fn create_index(db: &GrafeoDB) {
    db.create_vector_index(
        "Person",
        "embedding",
        Some(2),
        Some("euclidean"),
        None,
        None,
        None,
    )
    .unwrap();
}

/// The names of the `k` people nearest to `query`, nearest first, as
/// `vector_search` finds them.
fn nearest(db: &GrafeoDB, query: [f32; 2], k: usize) -> Vec<String> {
    db.vector_search("Person", "embedding", &query, k, None, None)
        .unwrap()
        .into_iter()
        .map(|(id, _)| name_of(db, id))
        .collect()
}

/// The query for the people within `distance` of `query`, by name; with
/// the index, its filter is a vector scan of the index.
fn within_query(query: [f32; 2], distance: f32) -> String {
    let [x, y] = query;
    format!(
        "MATCH (p:Person) WHERE euclidean_distance(p.embedding, [{x:.1}, {y:.1}]) <= {distance:.1}          RETURN p.name AS name ORDER BY name"
    )
}

/// The names of the people within `distance` of `query`, sorted, as a query
/// finds them.
fn within(db: &GrafeoDB, query: [f32; 2], distance: f32) -> Vec<String> {
    db.execute(&within_query(query, distance))
        .unwrap()
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::String(name) => name.to_string(),
            other => format!("{other:?}"),
        })
        .collect()
}

fn name_of(db: &GrafeoDB, id: NodeId) -> String {
    db.get_node(id)
        .and_then(|node| node.get_property("name")?.as_str().map(str::to_string))
        .unwrap_or_else(|| format!("{id:?}"))
}

/// Each person is the nearest to their own embedding, and a search for all
/// of them finds everyone in order of distance: through `vector_search` and
/// through a query, which scans the index.
fn assert_everyone_is_found(db: &GrafeoDB, stage: &str) {
    for (name, embedding) in BASE.iter().chain(&LATER) {
        assert_eq!(
            nearest(db, *embedding, 1),
            [*name],
            "{stage}: vector_search nearest to {name}"
        );
        assert_eq!(
            within(db, *embedding, 1.0),
            [*name],
            "{stage}: the query for who is near {name}"
        );
    }
    assert_eq!(
        nearest(db, [0.0, 0.0], 5),
        BY_DISTANCE_TO_THE_ORIGIN,
        "{stage}: vector_search for everyone"
    );
    let mut everyone = BY_DISTANCE_TO_THE_ORIGIN.map(str::to_string).to_vec();
    everyone.sort();
    assert_eq!(
        within(db, [0.0, 0.0], 188.0),
        everyone,
        "{stage}: the query for everyone"
    );
    let profile = db
        .execute(&format!("PROFILE {}", within_query([0.0, 0.0], 188.0)))
        .unwrap();
    let plan = format!("{:?}", profile.rows());
    assert!(
        plan.contains("VectorScan"),
        "{stage}: the query scans the index: {plan}"
    );
}

/// The compacted people, and the vector index over them.
fn compacted_people() -> GrafeoDB {
    let mut db = GrafeoDB::new_in_memory();
    for (name, embedding) in BASE {
        person(&db, name, embedding);
    }
    db.compact().unwrap();
    create_index(&db);
    db
}

/// The vectors written after `compact()` are found beside the compacted
/// ones.
#[test]
fn vectors_written_after_compact_are_found_beside_the_compacted_ones() {
    let db = compacted_people();
    for (name, embedding) in LATER {
        person(&db, name, embedding);
    }
    assert_everyone_is_found(&db, "after compact()");
}

/// A new value of a compacted node's vector is found where it moved to, and
/// the other compacted vectors where they are.
#[test]
fn a_compacted_node_whose_vector_changes_is_found_at_its_new_value() {
    let db = compacted_people();
    let gus = db
        .vector_search("Person", "embedding", &[19.0, 19.0], 1, None, None)
        .unwrap()[0]
        .0;
    assert_eq!(name_of(&db, gus), "Gus");
    db.set_node_property(gus, "embedding", Value::Vector(vec![88.0, 19.0].into()))
        .unwrap();
    assert_eq!(nearest(&db, [88.0, 19.0], 1), ["Gus"], "at the new value");
    assert_eq!(
        nearest(&db, [0.0, 0.0], 3),
        ["Alix", "Gus", "Vincent"],
        "everyone, Gus now at (88, 19)"
    );
}

/// After a merge of the overlay into the base, by `recompact()`: the vectors
/// written before it are found, and so are those written after it.
#[test]
fn vectors_are_found_across_a_merge() {
    let mut db = compacted_people();
    let (mia, jules) = (LATER[0], LATER[1]);
    person(&db, mia.0, mia.1);
    db.recompact().unwrap();
    person(&db, jules.0, jules.1);
    assert_everyone_is_found(&db, "a vector before and one after recompact()");
}

/// After a close and reopen of a compacted file: the vectors written before
/// the close are found, and so are those written after the reopen.
#[cfg(all(feature = "grafeo-file", feature = "wal"))]
#[test]
fn vectors_are_found_after_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vectors.grafeo");
    let (mia, jules) = (LATER[0], LATER[1]);
    {
        let mut db = GrafeoDB::open(&path).unwrap();
        for (name, embedding) in BASE {
            person(&db, name, embedding);
        }
        db.compact().unwrap();
        create_index(&db);
        person(&db, mia.0, mia.1);
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    assert!(
        db.layered_store().is_some(),
        "the file holds a compacted base"
    );
    person(&db, jules.0, jules.1);
    assert_everyone_is_found(&db, "a vector before the close and one after the reopen");
    db.close().unwrap();
}

/// Without an index of their metric, the vector searches scan the compacted
/// nodes beside the later ones: a query whose metric is not the index's,
/// and a threshold search of the store.
#[test]
fn a_search_without_an_index_of_its_metric_scans_both_layers() {
    use grafeo_core::index::vector::DistanceMetric;

    let db = compacted_people();
    for (name, embedding) in LATER {
        person(&db, name, embedding);
    }
    // Manhattan distances to (3, 3): Alix 0, Mia 16, Gus 32, Jules 85,
    // Vincent 170. The index measures euclidean distances.
    let query = "MATCH (p:Person) WHERE manhattan_distance(p.embedding, [3.0, 3.0]) <= 19.0 \
                 RETURN p.name AS name ORDER BY name";
    let profile = db.execute(&format!("PROFILE {query}")).unwrap();
    let plan = format!("{:?}", profile.rows());
    assert!(plan.contains("VectorScan"), "a vector scan: {plan}");
    let names: Vec<Value> = db
        .execute(query)
        .unwrap()
        .rows()
        .iter()
        .map(|row| row[0].clone())
        .collect();
    assert_eq!(
        names,
        [Value::from("Alix"), Value::from("Mia")],
        "a compacted node and a later one"
    );

    let store = db.graph_store();
    let found: Vec<String> = store
        .vector_search_with_threshold(
            Some("Person"),
            "embedding",
            &[3.0, 3.0],
            19.0,
            DistanceMetric::Euclidean,
        )
        .into_iter()
        .map(|(id, _)| name_of(&db, id))
        .collect();
    assert_eq!(
        found,
        ["Alix", "Mia"],
        "the threshold search, nearest first"
    );
}
