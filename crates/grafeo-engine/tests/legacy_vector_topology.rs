//! A 0.5.x file whose vector topology links to a node outside the index.
//!
//! The `VectorStore` section of a 0.5.x file holds the HNSW topology of each
//! vector index in the 0.5.x formats (v1 bincode, v2 paged envelope), which
//! `VectorStoreSection` reads without checking its links (see
//! `index::vector::section` in grafeo-core, where the reader is tested). An
//! open of a database builds its vector indexes from the data and does not
//! read that section, so such a topology is never served: these tests hold
//! that for every open of a 0.5.x file (read-write, which migrates it,
//! read-only and `open_in_memory`), with a `VectorStore` section in which a
//! document lists Mia, a `:Person` with an embedding that the index on
//! `:Document(embedding)` does not hold.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test legacy_vector_topology
//! ```

#![cfg(all(
    feature = "gql",
    feature = "wal",
    feature = "grafeo-file",
    feature = "triple-store",
    feature = "vector-index",
    feature = "text-index",
    not(miri)
))]

#[path = "common/legacy_file.rs"]
mod legacy_file;

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use grafeo_common::storage::section::{Section, SectionType};
use grafeo_common::types::{NodeId, Value};
use grafeo_core::index::vector::{
    DistanceMetric, HnswConfig, HnswIndex, VectorIndexKind, VectorStoreSection,
};
use grafeo_engine::GrafeoDB;

/// The released 0.5.x file the test file is built from: it has a vector
/// index on `:Document(embedding)` and Mia, a `:Person` with an embedding.
const VERSION: &str = "0.5.44";

/// The node id of the one node `query` returns.
fn id_of(db: &GrafeoDB, query: &str) -> NodeId {
    let result = db.execute(query).unwrap();
    assert_eq!(result.rows().len(), 1, "one node for `{query}`");
    match &result.rows()[0][0] {
        Value::Int64(id) => NodeId::new(u64::try_from(*id).unwrap()),
        other => panic!("id() is an integer, got {other:?}"),
    }
}

/// The documents and Mia of the released file, by node id.
struct Ids {
    canals: NodeId,
    museums: NodeId,
    bridges: NodeId,
    mia: NodeId,
}

fn ids() -> Ids {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("closed.grafeo");
    std::fs::copy(legacy_file::released(VERSION, "closed.grafeo"), &path).unwrap();
    let db = GrafeoDB::open_read_only(&path).unwrap();
    let document = |title: &str| {
        id_of(
            &db,
            &format!("MATCH (d:Document {{title: '{title}'}}) RETURN id(d)"),
        )
    };
    let ids = Ids {
        canals: document("Canals"),
        museums: document("Museums"),
        bridges: document("Bridges"),
        mia: id_of(&db, "MATCH (p:Person {name: 'Mia'}) RETURN id(p)"),
    };
    db.close().unwrap();
    ids
}

/// A v2 `VectorStore` section (the paged envelope 0.5.x wrote) whose
/// topology of `Document:embedding` holds the three documents, Canals
/// listing Mia too.
fn section_with_a_link_to_mia(ids: &Ids) -> Vec<u8> {
    let index = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
        3,
        DistanceMetric::Cosine,
    ))));
    index.restore_topology(
        Some(ids.canals),
        0,
        vec![
            (ids.canals, vec![vec![ids.museums, ids.bridges, ids.mia]]),
            (ids.museums, vec![vec![ids.canals, ids.bridges]]),
            (ids.bridges, vec![vec![ids.canals, ids.museums]]),
        ],
    );
    VectorStoreSection::new(vec![("Document:embedding".to_string(), index)])
        .serialize()
        .unwrap()
}

/// Checks that the vector index of `db` never returns Mia: a search for her
/// vector returns the documents only, and a filtered one the document the
/// filter keeps.
fn assert_never_mia(db: &GrafeoDB, ids: &Ids, context: &str) {
    let found: Vec<NodeId> = db
        .vector_search("Document", "embedding", &[0.0, 0.0, 1.0], 5, None, None)
        .unwrap_or_else(|error| panic!("{context}: vector search: {error}"))
        .into_iter()
        .map(|(node, _)| node)
        .collect();
    assert!(
        !found.contains(&ids.mia),
        "{context}: the search returned Mia, who is not in the index: {found:?}"
    );
    let mut sorted = found.clone();
    sorted.sort_unstable();
    let mut documents = vec![ids.canals, ids.museums, ids.bridges];
    documents.sort_unstable();
    assert_eq!(sorted, documents, "{context}: the three documents");

    let filters: HashMap<String, Value> = [("title".to_string(), Value::from("Bridges"))].into();
    let filtered: Vec<NodeId> = db
        .vector_search(
            "Document",
            "embedding",
            &[0.0, 0.0, 1.0],
            5,
            None,
            Some(&filters),
        )
        .unwrap_or_else(|error| panic!("{context}: filtered vector search: {error}"))
        .into_iter()
        .map(|(node, _)| node)
        .collect();
    assert_eq!(filtered, [ids.bridges], "{context}: a filtered search");
}

/// An open of a database file.
type Open = fn(&Path) -> grafeo_common::utils::error::Result<GrafeoDB>;

/// Control: every open of a 0.5.x file whose `VectorStore` section links a
/// document to Mia (outside the index) builds the index from the data, and
/// a search never returns Mia, also after the migrated file is reopened.
#[test]
fn a_0_5_file_with_a_link_outside_the_index_never_serves_it() {
    let ids = ids();
    let section = section_with_a_link_to_mia(&ids);
    let opens: [(&str, Open); 3] = [
        ("read-write", |path| GrafeoDB::open(path)),
        ("read-only", |path| GrafeoDB::open_read_only(path)),
        ("in memory", |path| GrafeoDB::open_in_memory(path)),
    ];
    for (name, open) in opens {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        legacy_file::write_0_5_file(
            &path,
            VERSION,
            |section_type| section_type != SectionType::VectorStore,
            &[(SectionType::VectorStore, section.clone())],
        );
        let db = open(&path).unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_never_mia(&db, &ids, name);
        drop(db);
        if name == "read-write" {
            let db = GrafeoDB::open(&path).unwrap();
            assert_never_mia(&db, &ids, "the migrated file reopened");
            db.close().unwrap();
        }
    }
}
