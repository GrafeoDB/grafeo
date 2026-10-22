//! Integration tests for vector embedding spill to disk.
//!
//! Tests the full lifecycle: insert vectors, spill them into a cache file
//! their column reads through, search, and verify correctness (#594 has the
//! durability tests: `spilled_embeddings.rs`).
//!
//! Run with: cargo test -p grafeo-engine --features "embedded,async-storage" --test vector_spill

// When temporal feature is active, tests are cfg'd out and imports become unused.
#![allow(unused_imports, dead_code)]

use grafeo_common::storage::{SectionMemoryConfig, SectionType, TierOverride};
use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

fn make_embedding(seed: u64, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| ((seed * 7 + i as u64) % 100) as f32 / 100.0)
        .collect()
}

/// A persistent database whose vector section is forced to disk.
fn force_disk_db(path: &std::path::Path) -> GrafeoDB {
    GrafeoDB::with_config(Config::persistent(path).with_section_config(
        SectionType::VectorStore,
        SectionMemoryConfig {
            max_ram: None,
            tier: TierOverride::ForceDisk,
        },
    ))
    .unwrap()
}

/// Three indexed `:Item` nodes with embeddings close to the x axis.
fn indexed_items(db: &GrafeoDB) -> Vec<grafeo_common::types::NodeId> {
    let ids = (0..3u8)
        .map(|i| {
            let embedding = vec![1.0, f32::from(i) / 10.0, 0.0, 0.0];
            db.create_node_with_props(&["Item"], [("embedding", Value::Vector(embedding.into()))])
                .unwrap()
        })
        .collect();
    db.create_vector_index("Item", "embedding", Some(4), None, None, None, None)
        .unwrap();
    ids
}

fn embedding(db: &GrafeoDB, id: grafeo_common::types::NodeId) -> Option<Vec<f32>> {
    match db.get_node(id)?.get_property("embedding")? {
        Value::Vector(v) => Some(v.to_vec()),
        _ => None,
    }
}

/// An embedding changed while its column is spilled wins over the spilled
/// one when the column is reloaded.
#[test]
#[cfg(all(feature = "vector-index", feature = "mmap", not(feature = "temporal")))]
fn an_embedding_updated_while_spilled_wins() {
    let dir = tempfile::TempDir::new().unwrap();
    let db = force_disk_db(&dir.path().join("updated.grafeo"));
    let ids = indexed_items(&db);
    db.buffer_manager().spill_all();

    let moved = vec![0.0, 0.0, 0.0, 1.0];
    db.set_node_property(ids[0], "embedding", Value::Vector(moved.clone().into()))
        .unwrap();

    assert!(db.reload_eligible(1.0) > 0);
    assert_eq!(embedding(&db, ids[0]), Some(moved));
    assert_eq!(embedding(&db, ids[1]), Some(vec![1.0, 0.1, 0.0, 0.0]));
}

/// Closed while spilled: the database file holds every embedding (the spill
/// cache is gone), and one changed while spilled is the one it holds.
#[test]
#[cfg(all(
    feature = "vector-index",
    feature = "mmap",
    feature = "grafeo-file",
    not(feature = "temporal")
))]
fn an_embedding_updated_while_spilled_wins_after_a_reopen() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("closed.grafeo");
    let moved = vec![0.0, 0.0, 0.0, 1.0];
    let ids = {
        let db = force_disk_db(&path);
        let ids = indexed_items(&db);
        db.buffer_manager().spill_all();
        db.set_node_property(ids[0], "embedding", Value::Vector(moved.clone().into()))
            .unwrap();
        db.close().unwrap();
        ids
    };

    assert!(!dir.path().join("closed.grafeo.spill").exists());
    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let nearest = db
        .vector_search("Item", "embedding", &[1.0, 0.2, 0.0, 0.0], 1, None, None)
        .unwrap();
    assert_eq!(nearest[0].0, ids[2], "{nearest:?}");
    assert_eq!(embedding(&db, ids[0]), Some(moved));
    assert_eq!(embedding(&db, ids[2]), Some(vec![1.0, 0.2, 0.0, 0.0]));
}

#[test]
#[cfg(all(feature = "vector-index", feature = "mmap", not(feature = "temporal")))]
fn force_disk_spills_and_search_works() {
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("spill_test.grafeo");

    let config = Config::persistent(&db_path).with_section_config(
        SectionType::VectorStore,
        SectionMemoryConfig {
            max_ram: None,
            tier: TierOverride::ForceDisk,
        },
    );

    let db = GrafeoDB::with_config(config).unwrap();

    // Insert nodes with vector properties
    let dim = 8;
    let mut node_ids = Vec::new();
    for i in 1..=10 {
        let id = db.create_node(&["Item"]).unwrap();
        let embedding = make_embedding(i, dim);
        db.set_node_property(id, "name", Value::from(format!("item_{i}")))
            .unwrap();
        db.set_node_property(id, "embedding", Value::Vector(embedding.into()))
            .unwrap();
        node_ids.push(id);
    }

    // Create vector index
    db.create_vector_index("Item", "embedding", Some(dim), None, None, None, None)
        .unwrap();

    // Trigger spill (ForceDisk was triggered at startup but we inserted after)
    db.buffer_manager().spill_all();

    // Check spill directory was created
    let spill_dir = db
        .buffer_manager()
        .config()
        .spill_path
        .as_ref()
        .expect("spill_path should be set for persistent DB");
    assert!(
        spill_dir.exists(),
        "spill directory should exist: {}",
        spill_dir.display()
    );

    // Vector search reads the spilled vectors in place
    let query = make_embedding(1, dim);
    let results = db
        .vector_search("Item", "embedding", &query, 5, None, None)
        .unwrap();
    assert!(
        !results.is_empty(),
        "vector search should return results after spill"
    );
    // Closest should be node with same seed
    assert_eq!(results[0].0, node_ids[0]);
}

#[test]
#[cfg(all(feature = "vector-index", feature = "mmap", not(feature = "temporal")))]
fn spill_with_no_vectors_is_noop() {
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("noop_test.grafeo");

    let config = Config::persistent(&db_path).with_section_config(
        SectionType::VectorStore,
        SectionMemoryConfig {
            max_ram: None,
            tier: TierOverride::ForceDisk,
        },
    );

    let db = GrafeoDB::with_config(config).unwrap();

    // No vectors, no indexes: spill should be a no-op
    let freed = db.buffer_manager().spill_all();
    assert_eq!(freed, 0, "spilling with no vectors should free 0 bytes");
}

#[test]
#[cfg(all(
    feature = "vector-index",
    feature = "mmap",
    feature = "grafeo-file",
    not(feature = "temporal")
))]
fn checkpoint_after_spill_preserves_non_vector_data() {
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("checkpoint_spill.grafeo");

    // Create and populate
    {
        let config = Config::persistent(&db_path);
        let db = GrafeoDB::with_config(config).unwrap();

        let id = db.create_node(&["Item"]).unwrap();
        db.set_node_property(id, "name", Value::from("test"))
            .unwrap();
        db.set_node_property(
            id,
            "embedding",
            Value::Vector(vec![1.0, 2.0, 3.0, 4.0].into()),
        )
        .unwrap();
        db.create_vector_index("Item", "embedding", Some(4), None, None, None, None)
            .unwrap();

        // Spill, then close (which checkpoints)
        db.buffer_manager().spill_all();
        db.close().unwrap();
    }

    // Reopen and verify non-vector data survived
    {
        let config = Config::persistent(&db_path);
        let db = GrafeoDB::with_config(config).unwrap();

        // Non-vector properties should survive (they're in the LPG section)
        let session = db.session();
        let result = session.execute("MATCH (n:Item) RETURN n.name").unwrap();
        assert!(
            !result.rows().is_empty(),
            "should find the node after reopen"
        );
        assert_eq!(result.rows()[0][0], Value::from("test"));
    }
}
