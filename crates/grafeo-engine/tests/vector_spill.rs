//! Integration tests for vector embedding spill to disk.
//!
//! Tests the full lifecycle: insert vectors, spill to MmapStorage,
//! search (via SpillableVectorAccessor), and verify correctness.
//!
//! Run with: cargo test -p grafeo-engine --features "embedded,async-storage" --test vector_spill

// When temporal feature is active, tests are cfg'd out and imports become unused.
#![allow(unused_imports, dead_code)]

use grafeo_common::storage::{SectionMemoryConfig, SectionType, TierOverride};
use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

fn force_disk_config(db_path: &std::path::Path, spill_path: &std::path::Path) -> Config {
    Config::persistent(db_path)
        .with_spill_path(spill_path)
        .with_section_tier(SectionType::VectorStore, TierOverride::ForceDisk)
}

fn assert_exact_hit(db: &GrafeoDB, expected: grafeo_common::types::NodeId, query: &[f32]) {
    let results = db
        .vector_search("Item", "embedding", query, 2, None, None)
        .unwrap();
    assert_eq!(results.first().map(|hit| hit.0), Some(expected));
    assert!(
        results[0].1 < 1.0e-6,
        "expected exact vector distance, got {}",
        results[0].1
    );
}

fn make_embedding(seed: u64, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| ((seed * 7 + i as u64) % 100) as f32 / 100.0)
        .collect()
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

    // Vector search should still work (via SpillableVectorAccessor)
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

#[test]
#[cfg(all(
    feature = "vector-index",
    feature = "mmap",
    feature = "grafeo-file",
    not(feature = "temporal")
))]
fn force_disk_fresh_mutate_checkpoint_close_reopen_search() {
    use grafeo_common::memory::StorageTier;

    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("fresh_lifecycle.grafeo");
    let spill_path = dir.path().join("fresh_lifecycle.spill");
    let first = vec![1.0, 0.0, 0.0, 0.0];
    let updated = vec![0.0, 1.0, 0.0, 0.0];
    let node;

    {
        let db = GrafeoDB::with_config(force_disk_config(&db_path, &spill_path)).unwrap();
        node = db.create_node(&["Item"]).unwrap();
        db.set_node_property(node, "embedding", Value::Vector(first.clone().into()))
            .unwrap();
        db.create_vector_index("Item", "embedding", Some(4), None, None, None, None)
            .unwrap();

        db.wal_checkpoint().unwrap();
        assert_eq!(
            db.storage_tiers().get(&SectionType::VectorStore),
            Some(&StorageTier::OnDisk),
            "checkpoint must enforce ForceDisk for indexes created after open"
        );
        assert_exact_hit(&db, node, &first);

        db.set_node_property(node, "embedding", Value::Vector(updated.clone().into()))
            .unwrap();
        assert_exact_hit(&db, node, &updated);
        db.wal_checkpoint().unwrap();
        assert_eq!(
            db.storage_tiers().get(&SectionType::VectorStore),
            Some(&StorageTier::OnDisk)
        );
        db.close().unwrap();
    }

    {
        let db = GrafeoDB::with_config(force_disk_config(&db_path, &spill_path)).unwrap();
        assert_eq!(
            db.storage_tiers().get(&SectionType::VectorStore),
            Some(&StorageTier::OnDisk)
        );
        assert_exact_hit(&db, node, &updated);
        db.close().unwrap();
    }

    // The container remains authoritative: switching ForceDisk off must not
    // depend on a spill sidecar to recover the latest vector.
    {
        let db = GrafeoDB::with_config(Config::persistent(&db_path)).unwrap();
        assert_exact_hit(&db, node, &updated);
        db.close().unwrap();
    }
}

#[test]
#[cfg(all(
    feature = "vector-index",
    feature = "mmap",
    feature = "grafeo-file",
    not(feature = "temporal")
))]
fn existing_auto_graph_reopens_and_searches_under_force_disk() {
    use grafeo_common::memory::StorageTier;

    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("existing_lifecycle.grafeo");
    let spill_path = dir.path().join("existing_lifecycle.spill");
    let embedding = vec![0.0, 0.0, 1.0, 0.0];
    let node;

    {
        let db = GrafeoDB::with_config(Config::persistent(&db_path)).unwrap();
        node = db.create_node(&["Item"]).unwrap();
        db.set_node_property(node, "embedding", Value::Vector(embedding.clone().into()))
            .unwrap();
        db.create_vector_index("Item", "embedding", Some(4), None, None, None, None)
            .unwrap();
        db.wal_checkpoint().unwrap();
        db.close().unwrap();
    }

    {
        let db = GrafeoDB::with_config(force_disk_config(&db_path, &spill_path)).unwrap();
        assert_eq!(
            db.storage_tiers().get(&SectionType::VectorStore),
            Some(&StorageTier::OnDisk)
        );
        assert_exact_hit(&db, node, &embedding);
        db.close().unwrap();
    }
}
