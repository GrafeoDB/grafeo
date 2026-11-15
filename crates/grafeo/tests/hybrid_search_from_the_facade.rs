//! A hybrid search runs through the `grafeo` crate alone: `HybridSearchOptions` and the
//! `FusionMethod` it takes come from the facade, so a user sets a fusion without depending on
//! `grafeo-core` (grafeo-server#85). Without the re-export this file does not compile.
//!
//! ```bash
//! cargo test -p grafeo --features hybrid-search --test main hybrid_search_from_the_facade::
//! ```

#![cfg(feature = "hybrid-search")]

use grafeo::database::HybridSearchOptions;
use grafeo::{FusionMethod, GrafeoDB, NodeId, Value};

/// Three documents about canals and bridges, with a text and a vector index.
fn documents() -> (GrafeoDB, [NodeId; 3]) {
    let db = GrafeoDB::new_in_memory();
    let mut ids = [NodeId::new(0); 3];
    for (id, (text, vector)) in ids.iter_mut().zip([
        ("Amsterdam canals", [1.0_f32, 0.0]),
        ("Utrecht canals", [0.0, 1.0]),
        ("Berlin bridges", [0.7, 0.7]),
    ]) {
        *id = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(*id, "content", Value::String(text.into()))
            .unwrap();
        db.set_node_property(*id, "emb", Value::Vector(vector.to_vec().into()))
            .unwrap();
    }
    db.create_text_index("Doc", "content").unwrap();
    db.create_vector_index("Doc", "emb", Some(2), Some("cosine"), None, None, None)
        .unwrap();
    (db, ids)
}

/// A fusion from `grafeo::FusionMethod` is taken: RRF with k 60 is the default fusion, so it
/// returns what a search without one returns, and a weighted fusion is a fusion too.
#[test]
fn a_hybrid_search_takes_a_fusion_from_the_facade() {
    let (db, [amsterdam, ..]) = documents();
    let query = [1.0_f32, 0.0];
    let search = |options: &HybridSearchOptions<'_>| {
        db.hybrid_search_with("Doc", "content", "emb", "canals", 2, options)
            .unwrap()
    };
    let rrf = search(
        &HybridSearchOptions::new()
            .with_query_vector(&query)
            .with_fusion(FusionMethod::Rrf { k: 60 }),
    );
    let default = search(&HybridSearchOptions::new().with_query_vector(&query));
    assert_eq!(rrf, default);
    assert_eq!(rrf.len(), 2);
    // Amsterdam matches the text and is the nearest vector: first in both halves.
    assert_eq!(rrf[0].0, amsterdam);
    let weighted = search(
        &HybridSearchOptions::new()
            .with_query_vector(&query)
            .with_fusion(FusionMethod::Weighted {
                weights: vec![0.5, 0.5],
            }),
    );
    assert_eq!(weighted[0].0, amsterdam);
}
