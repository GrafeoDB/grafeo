//! Disk reopen tests for durable vector index quantization (V2-QUANT).
//!
//! Layer 2 acceptance:
//! create(mode) → insert vectors → checkpoint → close → reopen →
//! inspect mode matches + search returns the seeded top-1 neighbor.
//!
//! Product quant is schema-supported; reopen oracle deferred for cost
//! (schema field still round-trips via Catalog unit tests).

#![cfg(all(feature = "vector-index", feature = "grafeo-file", feature = "lpg"))]

use grafeo_common::types::Value;
use grafeo_core::index::vector::QuantizationType;
use grafeo_engine::{Config, GrafeoDB};

fn seeded_vector(seed: u64, dim: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    let mut raw: Vec<f32> = (0..dim)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect();
    let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut raw {
            *x /= norm;
        }
    }
    raw
}

fn reopen_mode_and_search(mode: Option<&str>, expected: QuantizationType) {
    let dir = tempfile::TempDir::new().expect("tempdir");
    // Prefer /data/tmp when TMPDIR is set by the host; tempfile honors TMPDIR.
    let path = dir
        .path()
        .join(format!("v2_quant_{}.grafeo", expected.name()));
    let dim = 16;
    let n = 40;
    let query = seeded_vector(0, dim);

    let (top1_before, heap_before) = {
        let db = GrafeoDB::with_config(Config::persistent(&path)).expect("create db");
        let mut first_id = None;
        for i in 0..n {
            let node = db.create_node(&["Doc"]).expect("node");
            if i == 0 {
                first_id = Some(node);
            }
            db.set_node_property(
                node,
                "emb",
                Value::Vector(seeded_vector(i as u64, dim).into()),
            )
            .expect("emb");
        }
        db.create_vector_index("Doc", "emb", Some(dim), Some("cosine"), None, None, mode)
            .expect("create vector index");

        assert!(db.has_vector_index("Doc", "emb"));
        assert_eq!(
            db.vector_index_quantization("Doc", "emb"),
            Some(expected),
            "inspect after create"
        );

        let results = db
            .vector_search("Doc", "emb", &query, 3, None, None)
            .expect("search before");
        assert!(!results.is_empty(), "search ready before close");
        let top1 = results[0].0;
        assert_eq!(top1, first_id.expect("first"), "query0 should prefer seed0");
        let heap = db
            .vector_index_heap_memory_bytes("Doc", "emb")
            .expect("heap");

        db.wal_checkpoint().expect("checkpoint");
        db.close().expect("close");
        (top1, heap)
    };

    let db = GrafeoDB::open(&path).expect("reopen");
    assert!(
        db.has_vector_index("Doc", "emb"),
        "index registered after reopen"
    );
    assert_eq!(
        db.vector_index_quantization("Doc", "emb"),
        Some(expected),
        "mode sticky after reopen (Catalog authority)"
    );

    let results = db
        .vector_search("Doc", "emb", &query, 3, None, None)
        .expect("search after reopen");
    assert!(!results.is_empty(), "search ready after reopen");
    assert_eq!(
        results[0].0, top1_before,
        "top-1 neighbor identity preserved for synthetic fixture"
    );

    let heap_after = db
        .vector_index_heap_memory_bytes("Doc", "emb")
        .expect("heap after");
    // Not a pass/fail RSS gate — record order-of-magnitude sanity only.
    assert!(
        heap_after > 0 && heap_before > 0,
        "heap helpers must report non-zero after rehydrate/create (before={heap_before} after={heap_after})"
    );

    db.close().ok();
}

#[test]
fn quant_reopen_none_mode() {
    reopen_mode_and_search(None, QuantizationType::None);
    reopen_mode_and_search(Some("none"), QuantizationType::None);
}

#[test]
fn quant_reopen_scalar_mode() {
    reopen_mode_and_search(Some("scalar"), QuantizationType::Scalar);
}

#[test]
fn quant_reopen_binary_mode() {
    reopen_mode_and_search(Some("binary"), QuantizationType::Binary);
}

#[test]
fn inspect_api_missing_vs_plain() {
    let db = GrafeoDB::new_in_memory();
    assert!(!db.has_vector_index("Doc", "emb"));
    assert_eq!(db.vector_index_quantization("Doc", "emb"), None);
    assert_eq!(db.vector_index_heap_memory_bytes("Doc", "emb"), None);

    let node = db.create_node(&["Doc"]).unwrap();
    db.set_node_property(node, "emb", Value::Vector(seeded_vector(1, 4).into()))
        .unwrap();
    db.create_vector_index("Doc", "emb", Some(4), Some("cosine"), None, None, None)
        .unwrap();
    assert!(db.has_vector_index("Doc", "emb"));
    assert_eq!(
        db.vector_index_quantization("Doc", "emb"),
        Some(QuantizationType::None)
    );
}
