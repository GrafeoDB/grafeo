//! Tests for index operations after compact().
//!
//! Validates that vector and text indexes work correctly when the database
//! is in layered mode (after compact()), including the full cycle of
//! insert → compact → create index → search.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test post_compact_index
//! ```

#![cfg(all(feature = "compact-store", feature = "lpg"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

fn vec3(x: f32, y: f32, z: f32) -> Value {
    Value::Vector(vec![x, y, z].into())
}

/// Helper: create a DB with vector data, compact it, then return it.
fn setup_compacted_vector_db() -> GrafeoDB {
    let mut db = GrafeoDB::new_in_memory();

    let n1 = db.create_node(&["Doc"]);
    db.set_node_property(n1, "embedding", vec3(1.0, 0.0, 0.0));
    db.set_node_property(n1, "title", Value::String("alpha".into()));

    let n2 = db.create_node(&["Doc"]);
    db.set_node_property(n2, "embedding", vec3(0.0, 1.0, 0.0));
    db.set_node_property(n2, "title", Value::String("beta".into()));

    let n3 = db.create_node(&["Doc"]);
    db.set_node_property(n3, "embedding", vec3(0.0, 0.0, 1.0));
    db.set_node_property(n3, "title", Value::String("gamma".into()));

    db.compact().expect("compact");
    db
}

// ── Vector index after compact ─────────────────────────────────────

#[test]
#[cfg(feature = "vector-index")]
fn vector_index_after_compact_returns_results() {
    let db = setup_compacted_vector_db();

    db.create_vector_index("Doc", "embedding", Some(3), Some("cosine"), None, None, None)
        .expect("create vector index after compact");

    let results = db
        .vector_search("Doc", "embedding", &[1.0, 0.0, 0.0], 3, None, None)
        .expect("vector search after compact");

    assert_eq!(results.len(), 3, "should find all 3 pre-compact nodes");
}

#[test]
#[cfg(feature = "vector-index")]
fn vector_index_after_compact_nearest_neighbor_is_correct() {
    let db = setup_compacted_vector_db();

    db.create_vector_index("Doc", "embedding", Some(3), Some("cosine"), None, None, None)
        .expect("create vector index");

    let results = db
        .vector_search("Doc", "embedding", &[1.0, 0.0, 0.0], 1, None, None)
        .expect("search");

    assert_eq!(results.len(), 1);
    // Nearest to [1,0,0] should be node 1 (the one with [1,0,0])
    let (nearest_id, distance) = results[0];
    assert!(
        distance < 0.01,
        "exact match should have near-zero distance, got {distance}"
    );

    // Verify it's the right node by checking its property
    let title = db.graph_store().get_node_property(
        nearest_id,
        &grafeo_common::types::PropertyKey::new("title"),
    );
    assert_eq!(title, Some(Value::String("alpha".into())));
}

#[test]
#[cfg(feature = "vector-index")]
fn rebuild_vector_index_after_compact() {
    let db = setup_compacted_vector_db();

    db.create_vector_index("Doc", "embedding", Some(3), Some("cosine"), None, None, None)
        .expect("create");

    db.rebuild_vector_index("Doc", "embedding")
        .expect("rebuild after compact");

    let results = db
        .vector_search("Doc", "embedding", &[0.0, 1.0, 0.0], 3, None, None)
        .expect("search after rebuild");

    assert_eq!(results.len(), 3);
}

#[test]
#[cfg(feature = "vector-index")]
fn vector_search_with_filter_after_compact() {
    let mut db = GrafeoDB::new_in_memory();

    let n1 = db.create_node(&["Doc"]);
    db.set_node_property(n1, "embedding", vec3(1.0, 0.0, 0.0));
    db.set_node_property(n1, "category", Value::String("science".into()));

    let n2 = db.create_node(&["Doc"]);
    db.set_node_property(n2, "embedding", vec3(0.0, 1.0, 0.0));
    db.set_node_property(n2, "category", Value::String("art".into()));

    let n3 = db.create_node(&["Doc"]);
    db.set_node_property(n3, "embedding", vec3(0.0, 0.0, 1.0));
    db.set_node_property(n3, "category", Value::String("science".into()));

    db.compact().expect("compact");
    db.create_property_index("category");
    db.create_vector_index("Doc", "embedding", Some(3), Some("cosine"), None, None, None)
        .expect("create vector index");

    let mut filters = std::collections::HashMap::new();
    filters.insert("category".to_string(), Value::String("science".into()));

    let results = db
        .vector_search("Doc", "embedding", &[1.0, 0.0, 0.0], 3, None, Some(&filters))
        .expect("filtered search");

    assert_eq!(results.len(), 2, "should only return science nodes");
}

// ── Text index after compact ───────────────────────────────────────

#[test]
#[cfg(feature = "text-index")]
fn text_index_after_compact() {
    let mut db = GrafeoDB::new_in_memory();

    let n1 = db.create_node(&["Article"]);
    db.set_node_property(n1, "body", Value::String("the quick brown fox jumps over the lazy dog".into()));

    let n2 = db.create_node(&["Article"]);
    db.set_node_property(n2, "body", Value::String("a fast brown fox leaps over a sleepy hound".into()));

    let n3 = db.create_node(&["Article"]);
    db.set_node_property(n3, "body", Value::String("the cat sat on the mat".into()));

    db.compact().expect("compact");

    db.create_text_index("Article", "body").expect("create text index after compact");

    let results = db.text_search("Article", "body", "fox", 10).expect("text search");
    assert_eq!(results.len(), 2, "should find both fox articles from pre-compact data");
}

// ── Snapshot round-trip ────────────────────────────────────────────

#[test]
#[cfg(feature = "vector-index")]
fn snapshot_compact_vector_index_round_trip() {
    // Phase 1: build DB, export snapshot
    let snapshot_bytes = {
        let db = GrafeoDB::new_in_memory();
        let n1 = db.create_node(&["Doc"]);
        db.set_node_property(n1, "embedding", vec3(1.0, 0.0, 0.0));
        let n2 = db.create_node(&["Doc"]);
        db.set_node_property(n2, "embedding", vec3(0.0, 1.0, 0.0));
        let n3 = db.create_node(&["Doc"]);
        db.set_node_property(n3, "embedding", vec3(0.0, 0.0, 1.0));
        db.export_snapshot().expect("export")
    };

    // Phase 2: import → compact → create index → search
    let mut db = GrafeoDB::import_snapshot(&snapshot_bytes).expect("import");
    db.compact().expect("compact after import");

    db.create_vector_index("Doc", "embedding", Some(3), Some("cosine"), None, None, None)
        .expect("create vector index after snapshot+compact");

    let results = db
        .vector_search("Doc", "embedding", &[1.0, 0.0, 0.0], 3, None, None)
        .expect("vector search after snapshot+compact");

    assert_eq!(results.len(), 3, "should find all nodes from snapshot");
}

// ── LayeredStore trait method forwarding ────────────────────────────

#[test]
#[cfg(feature = "vector-index")]
fn layered_store_has_vector_index_forwards_to_overlay() {

    let mut db = GrafeoDB::new_in_memory();

    let n = db.create_node(&["Doc"]);
    db.set_node_property(n, "embedding", vec3(1.0, 0.0, 0.0));

    db.compact().expect("compact");

    // No index yet — graph_store (LayeredStore) should report false
    let gs = db.graph_store();
    assert!(!gs.has_vector_index("Doc", "embedding"));

    // Create index on the overlay via the imperative API
    db.create_vector_index("Doc", "embedding", Some(3), Some("cosine"), None, None, None)
        .expect("create");

    // Now LayeredStore should forward to overlay and report true
    let gs = db.graph_store();
    assert!(gs.has_vector_index("Doc", "embedding"));
    assert!(gs.get_vector_index_handle("Doc", "embedding").is_some());
}

#[test]
#[cfg(feature = "text-index")]
fn layered_store_has_text_index_forwards_to_overlay() {

    let mut db = GrafeoDB::new_in_memory();

    let n = db.create_node(&["Article"]);
    db.set_node_property(n, "body", Value::String("hello world".into()));

    db.compact().expect("compact");

    let gs = db.graph_store();
    assert!(!gs.has_text_index("Article", "body"));

    db.create_text_index("Article", "body").expect("create");

    let gs = db.graph_store();
    assert!(gs.has_text_index("Article", "body"));
}
