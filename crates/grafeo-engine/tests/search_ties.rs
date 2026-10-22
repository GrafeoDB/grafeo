//! Tied search scores come back in one order (#619): the tie by ascending
//! node id, the cut at k keeping the lowest ids, the same on every call, in
//! every search, and after a reopen and a read-only open (the reporter's
//! store was a file, whose indexes a reopen builds again in another order).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test search_ties
//! ```

#![cfg(all(feature = "vector-index", feature = "text-index"))]

use grafeo_common::types::{NodeId, Value};
use grafeo_engine::GrafeoDB;

const DOCS: usize = 200;
const K: usize = 7;

/// `DOCS` nodes that tie in every search: the same text, and vectors on a
/// circle in the x-y plane, each at cosine distance exactly 1 from
/// `[0, 0, 1]` (their dot product with it is 0). The vectors differ, so the
/// index links them as distinct vectors. Returns the ids, ascending.
fn tied_docs(db: &GrafeoDB) -> Vec<NodeId> {
    let mut ids: Vec<NodeId> = (0..DOCS)
        .map(|i| {
            let angle = std::f32::consts::TAU * i as f32 / DOCS as f32;
            let id = db.create_node(&["Doc"]).unwrap();
            db.set_node_property(id, "body", Value::String("Amsterdam canals".into()))
                .unwrap();
            db.set_node_property(
                id,
                "emb",
                Value::Vector(vec![angle.cos(), angle.sin(), 0.0].into()),
            )
            .unwrap();
            id
        })
        .collect();
    db.create_text_index("Doc", "body").expect("text index");
    db.create_vector_index("Doc", "emb", Some(3), Some("cosine"), None, None, None)
        .expect("vector index");
    ids.sort_unstable();
    ids
}

fn ids_of(results: &[(NodeId, f64)]) -> Vec<NodeId> {
    results.iter().map(|(id, _)| *id).collect()
}

fn text(db: &GrafeoDB) -> Vec<NodeId> {
    ids_of(&db.text_search("Doc", "body", "canals", K, None).unwrap())
}

fn vector(db: &GrafeoDB) -> Vec<NodeId> {
    // A beam wider than the index, so the search sees every tied vector.
    db.vector_search("Doc", "emb", &[0.0, 0.0, 1.0], K, Some(4 * DOCS), None)
        .unwrap()
        .into_iter()
        .map(|(id, _)| id)
        .collect()
}

#[cfg(feature = "hybrid-search")]
fn hybrid(db: &GrafeoDB, fusion: Option<grafeo_core::index::text::FusionMethod>) -> Vec<NodeId> {
    ids_of(
        &db.hybrid_search(
            "Doc",
            "body",
            "emb",
            "canals",
            Some(&[0.0, 0.0, 1.0]),
            K,
            fusion,
            None,
        )
        .unwrap(),
    )
}

/// The results of every search over the tied docs: text, vector, and with
/// the `hybrid-search` feature hybrid with RRF and with weights.
fn searches(db: &GrafeoDB) -> Vec<(&'static str, Vec<NodeId>)> {
    #[allow(
        unused_mut,
        reason = "the hybrid searches are added with their feature only"
    )]
    let mut all = vec![("text", text(db)), ("vector", vector(db))];
    #[cfg(feature = "hybrid-search")]
    {
        use grafeo_core::index::text::FusionMethod;
        all.push(("hybrid (RRF)", hybrid(db, None)));
        let weighted = FusionMethod::Weighted {
            weights: vec![0.5, 0.5],
        };
        all.push(("hybrid (weighted)", hybrid(db, Some(weighted))));
    }
    all
}

/// Ten calls of every search give `expected`. The exact searches (text, and
/// vector with a beam wider than the index) give the `K` lowest ids: an
/// approximate one (a hybrid search's vector part) sees only the tied
/// vectors its beam reaches, the same ones on every call and after a rebuild.
fn assert_searches_give(db: &GrafeoDB, expected: &[(&str, Vec<NodeId>)], when: &str) {
    for call in 0..10 {
        assert_eq!(searches(db), expected, "{when}, call {call}");
    }
}

#[test]
fn vector_and_weighted_hybrid_ties_keep_the_lowest_node_ids() {
    let db = GrafeoDB::new_in_memory();
    let ids = tied_docs(&db);
    let first = searches(&db);
    assert_eq!(first[0], ("text", ids[..K].to_vec()));
    assert_eq!(first[1], ("vector", ids[..K].to_vec()));
    assert_searches_give(&db, &first, "in memory");
}

/// A reopen builds the indexes again from the data: every search answers as
/// before, ties included, read-write and read-only.
#[test]
fn ties_keep_the_lowest_node_ids_after_a_reopen_and_read_only() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ties.grafeo");
    let before = {
        let db = GrafeoDB::open(&path).unwrap();
        let ids = tied_docs(&db);
        let before = searches(&db);
        assert_eq!(before[0], ("text", ids[..K].to_vec()));
        assert_eq!(before[1], ("vector", ids[..K].to_vec()));
        db.close().unwrap();
        before
    };
    for reopen in 0..3 {
        let db = GrafeoDB::open(&path).unwrap();
        assert_searches_give(&db, &before, &format!("after reopen {reopen}"));
        db.close().unwrap();
    }
    let db = GrafeoDB::open_read_only(&path).unwrap();
    assert_searches_give(&db, &before, "read-only");
}
