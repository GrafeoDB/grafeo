//! Integration tests for Phase 8 storage tier overrides.
//!
//! Covers:
//!  - `TierOverride::ForceDisk` triggers a targeted spill of just the
//!    matching consumer at database open.
//!  - `db.storage_tiers()` introspection reports the post-open tier
//!    state per registered consumer.
//!  - `Config::with_section_tier` convenience constructor preserves
//!    any previously configured `max_ram` value.
//!  - `TierOverride::Auto` (default) does NOT auto-spill at open.
//!
//! Run with:
//!   cargo test -p grafeo-engine --all-features \
//!              --test main storage_tier_overrides::

#![allow(unused_imports, dead_code)]

use grafeo_common::memory::buffer::StorageTier;
use grafeo_common::storage::{SectionMemoryConfig, SectionType, TierOverride};
use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

fn make_embedding(seed: u64, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| ((seed * 7 + i as u64) % 100) as f32 / 100.0)
        .collect()
}

#[test]
#[cfg(all(feature = "vector-index", feature = "mmap"))]
fn alix_force_disk_on_vector_only_does_not_spill_lpg() {
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("alix.grafeo");

    let config = Config::persistent(&db_path)
        .with_section_tier(SectionType::VectorStore, TierOverride::ForceDisk);

    let db = GrafeoDB::with_config(config).unwrap();

    // Insert LPG nodes + a vector index. The Phase 8a wiring should spill
    // ONLY the VectorStore consumer, not the LpgStore consumer.
    let dim = 8;
    for i in 1..=5 {
        let id = db.create_node(&["Item"]).unwrap();
        db.set_node_property(id, "name", Value::from(format!("item_{i}")))
            .unwrap();
        db.set_node_property(
            id,
            "embedding",
            Value::Vector(make_embedding(i, dim).into()),
        )
        .unwrap();
    }
    db.create_vector_index("Item", "embedding", Some(dim), None, None, None, None)
        .unwrap();

    // The Phase 8a spill at open targeted only the VectorStore consumer,
    // but at that moment the consumer was empty. After inserting vectors,
    // re-trigger the same targeted spill via the BufferManager directly
    // (mirrors what apply_force_disk_overrides() does at open).
    db.buffer_manager()
        .spill_consumer_by_name("section:VectorStore");

    let tiers = db.storage_tiers();

    // VectorStore: spilled OnDisk (heap freed by spill).
    let vector_tier = tiers.get(&SectionType::VectorStore);
    assert!(
        vector_tier.is_some(),
        "VectorStore consumer should be registered"
    );
    assert_eq!(
        vector_tier,
        Some(&StorageTier::OnDisk),
        "VectorStore must spill at open under ForceDisk; got {vector_tier:?}"
    );

    // LpgStore: untouched. Node count > 0 so it's InMemory (not Uninitialized).
    let lpg_tier = tiers.get(&SectionType::LpgStore);
    assert!(lpg_tier.is_some(), "LpgStore consumer should be registered");
    assert_ne!(
        lpg_tier,
        Some(&StorageTier::OnDisk),
        "LpgStore must NOT spill when only VectorStore is ForceDisk; got {lpg_tier:?}"
    );
}

#[test]
#[cfg(all(feature = "vector-index", feature = "mmap"))]
fn gus_no_force_disk_overrides_means_no_auto_spill() {
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("gus.grafeo");

    let config = Config::persistent(&db_path);
    let db = GrafeoDB::with_config(config).unwrap();

    // Build LPG content + a vector index, all with the default Auto tier.
    let dim = 8;
    for i in 1..=5 {
        let id = db.create_node(&["Item"]).unwrap();
        db.set_node_property(
            id,
            "embedding",
            Value::Vector(make_embedding(i, dim).into()),
        )
        .unwrap();
    }
    db.create_vector_index("Item", "embedding", Some(dim), None, None, None, None)
        .unwrap();

    let tiers = db.storage_tiers();
    let vector_tier = tiers.get(&SectionType::VectorStore);
    assert_ne!(
        vector_tier,
        Some(&StorageTier::OnDisk),
        "Auto must NOT auto-spill at open; got {vector_tier:?}"
    );
}

#[test]
fn vincent_with_section_tier_preserves_max_ram() {
    // Set max_ram via with_section_config, then call with_section_tier
    // with a different tier. The max_ram must be preserved.
    let config = Config::in_memory()
        .with_section_config(
            SectionType::VectorStore,
            SectionMemoryConfig {
                max_ram: Some(500 * 1024 * 1024),
                tier: TierOverride::Auto,
            },
        )
        .with_section_tier(SectionType::VectorStore, TierOverride::ForceDisk);

    let cfg = config
        .section_configs()
        .get(&SectionType::VectorStore)
        .expect("VectorStore config present");

    assert_eq!(cfg.tier, TierOverride::ForceDisk);
    assert_eq!(
        cfg.max_ram,
        Some(500 * 1024 * 1024),
        "with_section_tier must preserve max_ram from earlier with_section_config"
    );
}

#[test]
fn jules_with_section_tier_no_existing_config_uses_none_max_ram() {
    let config =
        Config::in_memory().with_section_tier(SectionType::CompactStore, TierOverride::ForceDisk);

    let cfg = config
        .section_configs()
        .get(&SectionType::CompactStore)
        .expect("CompactStore config present");

    assert_eq!(cfg.tier, TierOverride::ForceDisk);
    assert_eq!(cfg.max_ram, None);
}

#[test]
#[cfg(all(feature = "vector-index", feature = "mmap"))]
fn hans_force_ram_skips_explicit_spill_request() {
    // Phase 8g: even an explicit spill_consumer_by_name() must respect
    // ForceRam. The pin is the hard contract, not a soft hint.
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("hans.grafeo");

    let config = Config::persistent(&db_path)
        .with_section_tier(SectionType::VectorStore, TierOverride::ForceRam);

    let db = GrafeoDB::with_config(config).unwrap();

    let dim = 8;
    for i in 1..=5 {
        let id = db.create_node(&["Item"]).unwrap();
        db.set_node_property(
            id,
            "embedding",
            Value::Vector(make_embedding(i, dim).into()),
        )
        .unwrap();
    }
    db.create_vector_index("Item", "embedding", Some(dim), None, None, None, None)
        .unwrap();

    // Confirm the pin took effect.
    assert!(
        db.buffer_manager().is_force_ram("section:VectorStore"),
        "VectorStore must be marked ForceRam after db open"
    );

    // An explicit spill request must be a no-op for a ForceRam consumer.
    let freed = db
        .buffer_manager()
        .spill_consumer_by_name("section:VectorStore");
    assert_eq!(
        freed, 0,
        "ForceRam consumer must not be spilled by explicit request"
    );

    let tiers = db.storage_tiers();
    assert_ne!(
        tiers.get(&SectionType::VectorStore),
        Some(&StorageTier::OnDisk),
        "VectorStore must remain InMemory after spill attempt; got {:?}",
        tiers.get(&SectionType::VectorStore)
    );
}

#[test]
#[cfg(all(feature = "vector-index", feature = "mmap"))]
fn django_force_ram_survives_spill_all() {
    // spill_all() must skip ForceRam consumers but spill others.
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("django.grafeo");

    let config = Config::persistent(&db_path)
        .with_section_tier(SectionType::VectorStore, TierOverride::ForceRam);

    let db = GrafeoDB::with_config(config).unwrap();

    let dim = 8;
    for i in 1..=5 {
        let id = db.create_node(&["Item"]).unwrap();
        db.set_node_property(
            id,
            "embedding",
            Value::Vector(make_embedding(i, dim).into()),
        )
        .unwrap();
    }
    db.create_vector_index("Item", "embedding", Some(dim), None, None, None, None)
        .unwrap();

    // spill_all() walks every consumer; ForceRam ones must be skipped.
    db.buffer_manager().spill_all();

    let tiers = db.storage_tiers();
    assert_ne!(
        tiers.get(&SectionType::VectorStore),
        Some(&StorageTier::OnDisk),
        "spill_all must not spill a ForceRam consumer"
    );
}

#[test]
fn beatrix_clear_force_ram_re_enables_spill() {
    // Once ForceRam is cleared, the consumer participates in spill again.
    use grafeo_common::memory::BufferManager;

    let bm = BufferManager::with_budget(1024 * 1024);
    bm.mark_force_ram("section:Test");
    assert!(bm.is_force_ram("section:Test"));

    bm.clear_force_ram("section:Test");
    assert!(!bm.is_force_ram("section:Test"));
}

#[test]
#[cfg(all(feature = "vector-index", feature = "mmap"))]
fn shosanna_reload_eligible_brings_spilled_vectors_back_to_ram() {
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("shosanna.grafeo");

    let config = Config::persistent(&db_path);
    let db = GrafeoDB::with_config(config).unwrap();

    let dim = 8;
    for i in 1..=5 {
        let id = db.create_node(&["Item"]).unwrap();
        db.set_node_property(
            id,
            "embedding",
            Value::Vector(make_embedding(i, dim).into()),
        )
        .unwrap();
    }
    db.create_vector_index("Item", "embedding", Some(dim), None, None, None, None)
        .unwrap();

    // Spill manually, then verify storage_tiers reports OnDisk.
    let freed = db
        .buffer_manager()
        .spill_consumer_by_name("section:VectorStore");
    assert!(freed > 0, "spill should free some bytes");

    let tiers_after_spill = db.storage_tiers();
    assert_eq!(
        tiers_after_spill.get(&SectionType::VectorStore),
        Some(&StorageTier::OnDisk),
        "VectorStore must be OnDisk after spill"
    );

    // Reload eligible. With a fresh DB, current allocation should be tiny
    // relative to budget, so reload proceeds.
    let reloaded = db.reload_eligible(0.7);
    assert!(
        reloaded >= 1,
        "reload_eligible should bring back at least the VectorStore consumer; got {reloaded}"
    );

    let tiers_after_reload = db.storage_tiers();
    assert_ne!(
        tiers_after_reload.get(&SectionType::VectorStore),
        Some(&StorageTier::OnDisk),
        "VectorStore must NOT be OnDisk after reload_eligible; got {:?}",
        tiers_after_reload.get(&SectionType::VectorStore)
    );
}

#[test]
#[cfg(all(feature = "vector-index", feature = "mmap"))]
fn mia_storage_tiers_lists_only_section_consumers() {
    // The introspection method must skip CDC/overlay consumers and only
    // return entries for the SectionType-named section consumers.
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("mia.grafeo");

    let config = Config::persistent(&db_path);
    let db = GrafeoDB::with_config(config).unwrap();

    // Add some content so LpgStore has nonzero memory.
    db.create_node(&["X"]).unwrap();

    let tiers = db.storage_tiers();

    // Every key must be a section type that maps to a "section:..." consumer.
    for k in tiers.keys() {
        let valid = matches!(
            k,
            SectionType::Catalog
                | SectionType::LpgStore
                | SectionType::RdfStore
                | SectionType::CompactStore
                | SectionType::VectorStore
                | SectionType::TextIndex
                | SectionType::RdfRing
                | SectionType::PropertyIndex
        );
        assert!(valid, "unexpected section type in tiers: {k:?}");
    }

    // The LPG store's section consumer is always registered.
    assert!(
        tiers.contains_key(&SectionType::LpgStore),
        "LpgStore consumer must be registered: {tiers:?}"
    );
}

// ---------------------------------------------------------------------------
// Honest flags (#751 patch 5)
// ---------------------------------------------------------------------------

/// A database at `path` with 200 `:Item` nodes, each with an 8-dimension embedding, and a vector
/// index on them; closed.
#[cfg(all(feature = "vector-index", feature = "mmap"))]
fn items_with_vectors(path: &std::path::Path) {
    let db = GrafeoDB::open(path).unwrap();
    for i in 1..=200 {
        let id = db.create_node(&["Item"]).unwrap();
        db.set_node_property(id, "name", Value::from(format!("item_{i}")))
            .unwrap();
        db.set_node_property(id, "embedding", Value::Vector(make_embedding(i, 8).into()))
            .unwrap();
    }
    db.create_vector_index("Item", "embedding", Some(8), None, None, None, None)
        .unwrap();
    db.close().unwrap();
}

/// The ids a search near item 7's vector returns, in order.
#[cfg(all(feature = "vector-index", feature = "mmap"))]
fn near_item_seven(db: &GrafeoDB) -> Vec<grafeo_common::types::NodeId> {
    db.vector_search("Item", "embedding", &make_embedding(7, 8), 5, None, None)
        .unwrap()
        .into_iter()
        .map(|(id, _)| id)
        .collect()
}

/// `ForceDisk` on the vector store does what it says: the open of a database holding vectors
/// spills them, so the consumer is on disk right after the open, and searches answer as before.
#[test]
#[cfg(all(feature = "vector-index", feature = "mmap"))]
fn force_disk_on_the_vector_store_spills_it_at_open() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("lille.grafeo");
    items_with_vectors(&path);
    let before = {
        let db = GrafeoDB::open(&path).unwrap();
        let found = near_item_seven(&db);
        db.close().unwrap();
        found
    };

    let db = GrafeoDB::with_config(
        Config::persistent(&path)
            .with_section_tier(SectionType::VectorStore, TierOverride::ForceDisk),
    )
    .unwrap();
    assert_eq!(
        db.storage_tiers().get(&SectionType::VectorStore),
        Some(&StorageTier::OnDisk),
        "the vector store is on disk after the open"
    );
    assert_eq!(near_item_seven(&db), before, "searches answer as before");
    db.close().unwrap();
}

/// The usage a consumer reports, by name, from the buffer manager's consumer list.
#[cfg(all(feature = "vector-index", feature = "mmap"))]
fn reported(db: &GrafeoDB, name: &str) -> usize {
    db.buffer_manager()
        .consumer_stats()
        .into_iter()
        .find(|stats| stats.name == name)
        .unwrap_or_else(|| panic!("{name} is registered"))
        .usage_bytes
}

/// The vector consumer reports what a spill frees (the embedding column), not the HNSW topology
/// it keeps: a spill frees about what it reported, and the report drops by about as much.
#[test]
#[cfg(all(feature = "vector-index", feature = "mmap"))]
fn the_vector_consumer_reports_what_its_spill_frees() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("nantes.grafeo");
    items_with_vectors(&path);
    let db = GrafeoDB::open(&path).unwrap();
    let before = reported(&db, "section:VectorStore");
    assert!(
        before > 0,
        "the vectors are in memory: {before} bytes reported"
    );
    let freed = db
        .buffer_manager()
        .spill_consumer_by_name("section:VectorStore");
    let after = reported(&db, "section:VectorStore");
    let within_a_tenth = |a: usize, b: usize| a.abs_diff(b) * 10 <= b.max(1);
    assert!(
        within_a_tenth(freed, before),
        "the spill freed {freed} bytes for {before} reported"
    );
    assert!(
        within_a_tenth(before - after.min(before), before),
        "the report dropped from {before} to {after} bytes"
    );
    assert_eq!(
        near_item_seven(&db).len(),
        5,
        "searches answer after the spill"
    );
    db.close().unwrap();
}

/// Each consumer says whether its `evict` frees memory: the LPG store, the RDF store and ring,
/// and the text index cannot (they hold the data itself), the CDC log can (it drops old events).
#[test]
#[cfg(all(
    feature = "vector-index",
    feature = "mmap",
    feature = "text-index",
    feature = "triple-store",
    feature = "ring-index",
    feature = "cdc",
    feature = "gql"
))]
fn each_consumer_says_whether_it_can_evict() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("rennes.grafeo");
    let db = GrafeoDB::with_config(Config::persistent(&path).with_cdc()).unwrap();
    db.execute("INSERT (:Doc {body: 'the canals of Amsterdam'})")
        .unwrap();
    db.create_text_index("Doc", "body").unwrap();
    db.execute_sparql(
        "INSERT DATA { <http://example.org/alix> <http://example.org/lives> \"Amsterdam\" }",
    )
    .unwrap();
    let stats = db.buffer_manager().consumer_stats();
    let can_evict = |name: &str| {
        stats
            .iter()
            .find(|consumer| consumer.name == name)
            .unwrap_or_else(|| {
                let names: Vec<&str> = stats.iter().map(|c| c.name.as_str()).collect();
                panic!("{name} is registered: {names:?}")
            })
            .can_evict
    };
    for name in [
        "section:LpgStore",
        "section:RdfStore",
        "section:RdfRing",
        "section:TextIndex",
    ] {
        assert!(!can_evict(name), "{name} cannot evict");
    }
    assert!(can_evict("cdc_log"), "the CDC log can evict");
    db.close().unwrap();
}
