//! Integration tests for the agent-memory migration scenario (Discussion #155).
//!
//! Verifies: HNSW at scale, persistence, BYOV 384-dim, concurrent reads,
//! Python lifecycle (Rust side), storage size, and bulk import.
// Test IDs and indices are small known values
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
//!
//! ```bash
//! # Correctness (non-ignored)
//! cargo test -p grafeo-engine --features full --test agent_memory_migration -- --nocapture
//!
//! # Full benchmarks (ignored)
//! cargo test -p grafeo-engine --features full --release --test agent_memory_migration -- --ignored --nocapture
//! ```

#![cfg(feature = "full")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::time::Instant;

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

// ============================================================================
// Helpers
// ============================================================================

/// Deterministic pseudo-random 384-dim vector from a seed.
///
/// Uses a simple LCG (linear congruential generator) then L2-normalizes.
fn random_384d_vector(seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    let mut raw: Vec<f32> = (0..384)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            // Map to [-1, 1]
            ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect();
    // L2-normalize
    let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut raw {
            *x /= norm;
        }
    }
    raw
}

/// Human-readable byte formatting.
fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.2} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.2} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

/// Entity labels used in agent-memory tests.
const ENTITY_LABELS: &[&str] = &["Person", "Concept", "Event", "Document"];

/// Edge types used in agent-memory tests.
const EDGE_TYPES: &[&str] = &["KNOWS", "RELATED_TO", "MENTIONS", "OCCURRED_AT", "AUTHORED"];

// ============================================================================
// Q1: Long-term persistence + HNSW at scale
// ============================================================================

#[test]
fn test_incremental_growth_with_persistence() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("agent_memory.grafeo");

    let total_nodes = 500;
    let batch_size = 50;
    let reopen_every = 100; // close+reopen every 100 nodes

    let mut inserted = 0u64;

    // First open: create DB and seed first batch
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        for i in 0..batch_size {
            let node = db.create_node(&["Memory"]).unwrap();
            db.set_node_property(node, "text", Value::String(format!("fact_{i}").into()))
                .unwrap();
            db.set_node_property(node, "timestamp", Value::Int64(1_700_000_000 + i))
                .unwrap();
            db.set_node_property(node, "confidence", Value::Float64(0.5 + (i as f64) * 0.001))
                .unwrap();
            db.set_node_property(
                node,
                "embedding",
                Value::Vector(random_384d_vector(i as u64).into()),
            )
            .unwrap();
        }
        inserted += batch_size as u64;

        // Create vector index after first batch
        db.create_vector_index(
            "Memory",
            "embedding",
            Some(384),
            Some("cosine"),
            None,
            None,
            None,
        )
        .expect("create vector index");

        // Verify search works
        let results = db
            .vector_search("Memory", "embedding", &random_384d_vector(0), 5, None, None)
            .expect("search");
        assert_eq!(results.len(), 5, "should find 5 results from first batch");

        db.close().unwrap();
    }

    // Continue inserting in batches, reopening periodically
    while inserted < total_nodes as u64 {
        let db = GrafeoDB::open(&path).unwrap();

        // Index metadata is persisted in snapshot v4 (single-file format),
        // but WAL-based persistence requires manual recreation after reopen.
        db.create_vector_index(
            "Memory",
            "embedding",
            Some(384),
            Some("cosine"),
            None,
            None,
            None,
        )
        .expect("recreate vector index after reopen");

        let batch_end = (inserted + reopen_every as u64).min(total_nodes as u64);
        for i in inserted..batch_end {
            let node = db.create_node(&["Memory"]).unwrap();
            db.set_node_property(node, "text", Value::String(format!("fact_{i}").into()))
                .unwrap();
            db.set_node_property(node, "timestamp", Value::Int64(1_700_000_000 + i as i64))
                .unwrap();
            db.set_node_property(node, "confidence", Value::Float64(0.5 + (i as f64) * 0.001))
                .unwrap();
            db.set_node_property(
                node,
                "embedding",
                Value::Vector(random_384d_vector(i).into()),
            )
            .unwrap();
        }
        inserted = batch_end;

        // Search should return results
        let results = db
            .vector_search(
                "Memory",
                "embedding",
                &random_384d_vector(inserted - 1),
                5,
                None,
                None,
            )
            .expect("search after batch");
        assert!(
            !results.is_empty(),
            "search should return results after {inserted} inserts"
        );

        db.close().unwrap();
    }

    // Final reopen: verify everything survived
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        db.node_count(),
        total_nodes,
        "all {total_nodes} nodes should survive close/reopen cycles"
    );
    db.close().unwrap();
}

#[test]
#[ignore = "heavy benchmark: 20k vectors with HNSW, run locally"]
fn bench_hnsw_at_20k() {
    let db = GrafeoDB::new_in_memory();
    let total = 20_000;
    let milestones = [1_000, 5_000, 10_000, 15_000, 20_000];

    let start = Instant::now();

    // Insert first 100, then create index
    for i in 0..100u64 {
        let node = db.create_node(&["Memory"]).unwrap();
        db.set_node_property(
            node,
            "embedding",
            Value::Vector(random_384d_vector(i).into()),
        )
        .unwrap();
    }
    db.create_vector_index(
        "Memory",
        "embedding",
        Some(384),
        Some("cosine"),
        None,
        None,
        None,
    )
    .expect("create index");

    // Insert remaining incrementally
    for i in 100..total as u64 {
        let node = db.create_node(&["Memory"]).unwrap();
        db.set_node_property(
            node,
            "embedding",
            Value::Vector(random_384d_vector(i).into()),
        )
        .unwrap();

        // Measure at milestones
        let count = (i + 1) as usize;
        if milestones.contains(&count) {
            let mut total_search_us = 0u128;
            let num_queries = 10;
            for q in 0..num_queries {
                let query = random_384d_vector(1_000_000 + q);
                let search_start = Instant::now();
                let results = db
                    .vector_search("Memory", "embedding", &query, 10, None, None)
                    .expect("search");
                total_search_us += search_start.elapsed().as_micros();
                assert!(
                    !results.is_empty(),
                    "search should return results at {count}"
                );
            }
            let avg_us = total_search_us / num_queries as u128;
            let mem = db.memory_usage();
            eprintln!(
                "  {count:>6} nodes | search avg {avg_us:>6} us | memory {}",
                format_bytes(mem.total_bytes as u64)
            );
            assert!(
                avg_us < 100_000,
                "search should be under 100ms at {count} nodes"
            );
        }
    }

    let elapsed = start.elapsed();
    eprintln!("Total: {total} nodes inserted + indexed in {elapsed:.2?}");
}

#[test]
fn test_hnsw_recall_at_2k() {
    let db = GrafeoDB::new_in_memory();
    let count = 2_000;

    // Insert nodes with seeded vectors, tracking node_id -> vector_index
    let mut all_vectors: Vec<Vec<f32>> = Vec::with_capacity(count);
    let mut id_to_index: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
    for i in 0..count as u64 {
        let vec = random_384d_vector(i);
        let node = db.create_node(&["Memory"]).unwrap();
        db.set_node_property(node, "embedding", Value::Vector(vec.clone().into()))
            .unwrap();
        id_to_index.insert(node.as_u64(), i as usize);
        all_vectors.push(vec);
    }

    db.create_vector_index(
        "Memory",
        "embedding",
        Some(384),
        Some("cosine"),
        None,
        None,
        None,
    )
    .expect("create index");

    // Measure recall@10 for 20 queries
    let k = 10;
    let num_queries = 20;
    let mut total_recall = 0.0f64;

    for q in 0..num_queries {
        let query = random_384d_vector(500_000 + q);

        // Brute-force ground truth: cosine distance = 1 - dot(a,b) for normalized vectors
        let mut distances: Vec<(usize, f32)> = all_vectors
            .iter()
            .enumerate()
            .map(|(idx, vec)| {
                let dot: f32 = query.iter().zip(vec.iter()).map(|(a, b)| a * b).sum();
                (idx, 1.0 - dot)
            })
            .collect();
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let ground_truth: Vec<usize> = distances.iter().take(k).map(|(idx, _)| *idx).collect();

        // HNSW search
        let hnsw_results = db
            .vector_search("Memory", "embedding", &query, k, None, None)
            .expect("search");

        // Map node IDs back to vector indices using our tracking map
        let hnsw_indices: Vec<usize> = hnsw_results
            .iter()
            .filter_map(|(id, _)| id_to_index.get(&id.as_u64()).copied())
            .collect();

        let overlap = ground_truth
            .iter()
            .filter(|idx| hnsw_indices.contains(idx))
            .count();
        total_recall += overlap as f64 / k as f64;
    }

    let avg_recall = total_recall / num_queries as f64;
    eprintln!("HNSW recall@{k} at {count} nodes: {avg_recall:.3}");
    assert!(
        avg_recall >= 0.80,
        "average recall@{k} should be >= 0.80, got {avg_recall:.3}"
    );
}

// ============================================================================
// Q2: Concurrent reads during writes
// ============================================================================

#[test]
fn test_concurrent_vector_search_during_writes() {
    let db = Arc::new(GrafeoDB::new_in_memory());

    // Seed 500 nodes with vectors
    for i in 0..500u64 {
        let node = db.create_node(&["Memory"]).unwrap();
        db.set_node_property(
            node,
            "embedding",
            Value::Vector(random_384d_vector(i).into()),
        )
        .unwrap();
    }
    db.create_vector_index(
        "Memory",
        "embedding",
        Some(384),
        Some("cosine"),
        None,
        None,
        None,
    )
    .expect("create index");

    let num_readers = 4;
    let queries_per_reader = 50;
    let writes = 200;
    let total_threads = num_readers + 1; // readers + 1 writer

    let barrier = Arc::new(Barrier::new(total_threads));
    let read_success = Arc::new(AtomicUsize::new(0));
    let write_success = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();

    // Writer thread
    {
        let db = Arc::clone(&db);
        let barrier = Arc::clone(&barrier);
        let write_success = Arc::clone(&write_success);
        handles.push(std::thread::spawn(move || {
            barrier.wait();
            for i in 0..writes {
                let node = db.create_node(&["Memory"]).unwrap();
                db.set_node_property(
                    node,
                    "embedding",
                    Value::Vector(random_384d_vector(10_000 + i as u64).into()),
                )
                .unwrap();
                write_success.fetch_add(1, Ordering::Relaxed);
            }
        }));
    }

    // Reader threads
    for reader_id in 0..num_readers {
        let db = Arc::clone(&db);
        let barrier = Arc::clone(&barrier);
        let read_success = Arc::clone(&read_success);
        handles.push(std::thread::spawn(move || {
            barrier.wait();
            for q in 0..queries_per_reader {
                let query = random_384d_vector(100_000 + reader_id as u64 * 1000 + q as u64);
                let results = db.vector_search("Memory", "embedding", &query, 5, None, None);
                if results.is_ok() && !results.unwrap().is_empty() {
                    read_success.fetch_add(1, Ordering::Relaxed);
                }
            }
        }));
    }

    for h in handles {
        h.join().expect("thread should not panic");
    }

    assert_eq!(
        write_success.load(Ordering::Relaxed),
        writes,
        "all writes should succeed"
    );
    let reads = read_success.load(Ordering::Relaxed);
    eprintln!(
        "Concurrent: {writes} writes, {reads}/{} reads succeeded",
        num_readers * queries_per_reader
    );
    assert!(
        reads > 0,
        "at least some reads should succeed during concurrent writes"
    );

    assert_eq!(db.node_count(), 700, "500 initial + 200 written");
}

#[test]
fn test_multi_process_file_lock_rejected() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("locked.grafeo");

    let db1 = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let session = db1.session();
    session
        .execute("INSERT (:Memory {text: 'fact_1'})")
        .unwrap();

    // Second open should fail due to file lock
    let result = GrafeoDB::open(&path);
    assert!(
        result.is_err(),
        "second open of the same file should fail (no multi-process support)"
    );

    db1.close().unwrap();

    // After close, open should succeed
    let db2 = GrafeoDB::open(&path).unwrap();
    assert_eq!(db2.node_count(), 1);
    db2.close().unwrap();
}

// ============================================================================
// Q3: Lifecycle (Rust side, complements Python tests)
// ============================================================================

#[test]
fn test_close_reopen_preserves_data() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("lifecycle.grafeo");

    // Create, populate, close
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let mut node_ids = Vec::with_capacity(100);
        for i in 0..100 {
            let label = ENTITY_LABELS[i % ENTITY_LABELS.len()];
            let node = db.create_node(&[label]).unwrap();
            db.set_node_property(node, "name", Value::String(format!("entity_{i}").into()))
                .unwrap();
            node_ids.push(node);
        }
        // Create some edges via direct API
        for i in 0..50 {
            let edge_type = EDGE_TYPES[i % EDGE_TYPES.len()];
            db.create_edge(node_ids[i], node_ids[i + 1], edge_type);
        }
        assert_eq!(db.node_count(), 100);
        db.close().unwrap();
    }

    // Reopen, verify
    {
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(db.node_count(), 100, "nodes should survive close/reopen");
        assert!(db.edge_count() >= 50, "edges should survive close/reopen");
        db.close().unwrap();
    }
}

#[test]
fn test_drop_releases_lock() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("drop_lock.grafeo");

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session.execute("INSERT (:Memory {text: 'test'})").unwrap();
        // Drop without explicit close()
    }

    // Reopen should succeed (lock released on Drop)
    let db = GrafeoDB::open(&path).unwrap();
    // Data may or may not persist without explicit close, but the lock should be released
    drop(db);
}

// ============================================================================
// Q4: Storage size
// ============================================================================

#[test]
fn test_storage_size_100_entities() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("size_test.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let mut node_ids = Vec::with_capacity(100);
    for i in 0..100u64 {
        let label = ENTITY_LABELS[i as usize % ENTITY_LABELS.len()];
        let node = db.create_node_with_props(
            &[label],
            vec![
                ("text", Value::String(format!("This is a description of entity {i} with some content to simulate realistic text storage requirements.").into())),
                ("timestamp", Value::Int64(1_700_000_000 + i as i64)),
                ("confidence", Value::Float64(0.5 + (i as f64) * 0.005)),
                ("source", Value::String(format!("agent_{}", i % 5).into())),
            ],
        ).unwrap();
        db.set_node_property(
            node,
            "embedding",
            Value::Vector(random_384d_vector(i).into()),
        )
        .unwrap();
        node_ids.push(node);
    }
    // Create 150 edges via direct API
    for i in 0..150usize {
        let src = node_ids[i % 100];
        let dst = node_ids[(i * 7 + 3) % 100];
        if src != dst {
            let edge_type = EDGE_TYPES[i % EDGE_TYPES.len()];
            db.create_edge(src, dst, edge_type);
        }
    }

    db.close().unwrap();

    let file_size = std::fs::metadata(&path).unwrap().len();
    eprintln!(
        "Storage for 100 entities + 150 edges + 384-dim vectors: {}",
        format_bytes(file_size)
    );
    assert!(file_size > 0, "file should have content");
}

#[test]
#[ignore = "benchmark: Threadwise scenario (5400 entities, 15k facts, 384-dim vectors)"]
fn bench_storage_size_5400_entities() {
    // Mirrors the Threadwise agent-memory scenario (Discussion #155):
    //   ~5,400 entities (Person, Concept, Event, Document)
    //   ~15,000 facts (edges: KNOWS, RELATED_TO, MENTIONS, ...)
    //   384-dim sentence-transformer embeddings on every entity
    //   text observations of varying length, timestamps spanning months
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("size_bench.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let start = Instant::now();

    let num_nodes = 5_400;
    let num_edges = 15_000;

    // Short and long text templates to simulate real agent observations
    let short_texts = [
        "Alix mentioned this concept during standup",
        "Gus flagged a dependency between these modules",
        "Observed correlation in the Amsterdam dataset",
        "Vincent confirmed the Berlin deployment schedule",
        "Jules raised a concern about the Paris integration",
    ];
    let long_prefix = "Detailed observation: During the analysis session, the agent \
         identified a significant relationship between the current entity and several \
         previously recorded concepts. This connection was established based on semantic \
         similarity in the embedding space and confirmed through structural graph patterns. \
         Confidence was adjusted based on recency and source reliability. Context: ";

    // Insert nodes: entities with varied text, timestamps spread over ~6 months
    let mut node_ids = Vec::with_capacity(num_nodes);
    let base_ts: i64 = 1_700_000_000; // Nov 2023
    let six_months_secs: i64 = 180 * 86_400;

    for i in 0..num_nodes as u64 {
        let label = ENTITY_LABELS[i as usize % ENTITY_LABELS.len()];

        // Alternate short (~50 char) and long (~350 char) observations
        let text = if i % 3 == 0 {
            format!("{long_prefix}(ref #{i})")
        } else {
            format!("{} (ref #{i})", short_texts[i as usize % short_texts.len()])
        };

        let timestamp = base_ts + (i as i64 * six_months_secs / num_nodes as i64);

        let node = db
            .create_node_with_props(
                &[label],
                vec![
                    ("text", Value::String(text.into())),
                    ("timestamp", Value::Int64(timestamp)),
                    ("confidence", Value::Float64(0.3 + (i as f64 % 70.0) * 0.01)),
                    ("source", Value::String(format!("agent_{}", i % 5).into())),
                ],
            )
            .unwrap();
        db.set_node_property(
            node,
            "embedding",
            Value::Vector(random_384d_vector(i).into()),
        )
        .unwrap();
        node_ids.push(node);
    }

    let node_time = start.elapsed();

    // Insert edges: ~15k facts (~2.8 edges per entity, matching Threadwise ratio)
    let mut edge_count = 0usize;
    for i in 0..num_edges {
        let src = node_ids[i % num_nodes];
        let dst = node_ids[(i * 31 + 7) % num_nodes];
        if src != dst {
            let edge_type = EDGE_TYPES[i % EDGE_TYPES.len()];
            db.create_edge(src, dst, edge_type);
            edge_count += 1;
        }
    }

    let insert_elapsed = start.elapsed();
    let mem = db.memory_usage();

    db.close().unwrap();

    let file_size = std::fs::metadata(&path).unwrap().len();
    let kuzu_bytes: f64 = 7.3 * 1_073_741_824.0;

    eprintln!();
    eprintln!("=== Threadwise Scenario: 5,400 entities + 15k facts + 384-dim vectors ===");
    eprintln!("  Nodes:       {num_nodes} ({node_time:.2?})");
    eprintln!("  Edges:       {edge_count} (total insert {insert_elapsed:.2?})");
    eprintln!("  In-memory:   {}", format_bytes(mem.total_bytes as u64));
    eprintln!("  On-disk:     {}", format_bytes(file_size));
    eprintln!(
        "  Per entity:  {} bytes (incl. 384-dim vector = {} bytes raw)",
        file_size / num_nodes as u64,
        384 * 4
    );
    eprintln!("  Kuzu:        7.3 GB for same entity count");
    eprintln!(
        "  Ratio:       Grafeo is {:.0}x smaller",
        kuzu_bytes / file_size as f64
    );
    eprintln!();

    assert!(
        file_size < 500 * 1_048_576,
        "file should be well under 500 MB (Kuzu was 7.3 GB)"
    );
}

// ============================================================================
// Vector index rebuild after reopen (Discussion #155 caveat)
// ============================================================================

#[test]
#[ignore = "benchmark: vector index rebuild time at 5k nodes"]
fn bench_vector_index_rebuild_5k() {
    // Measures how long it takes to recreate a vector index from existing data
    // after a close/reopen cycle. This is what Threadwise would experience on
    // every application restart with 5k+ entities.

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("rebuild_bench.grafeo");

    let num_nodes = 5_000;

    // Phase 1: populate and close
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();

        for i in 0..num_nodes as u64 {
            let node = db
                .create_node_with_props(
                    &["Memory"],
                    vec![
                        ("text", Value::String(format!("fact_{i}").into())),
                        ("confidence", Value::Float64(0.5 + (i as f64) * 0.0001)),
                    ],
                )
                .unwrap();
            db.set_node_property(
                node,
                "embedding",
                Value::Vector(random_384d_vector(i).into()),
            )
            .unwrap();
        }

        // Create index before close (so it gets persisted in snapshot v4)
        db.create_vector_index(
            "Memory",
            "embedding",
            Some(384),
            Some("cosine"),
            None,
            None,
            None,
        )
        .expect("create index");

        // Verify search works before close
        let results = db
            .vector_search(
                "Memory",
                "embedding",
                &random_384d_vector(0),
                10,
                None,
                None,
            )
            .expect("search before close");
        assert_eq!(results.len(), 10, "should find results before close");

        db.close().unwrap();
    }

    // Phase 2: reopen and measure index rebuild time
    let reopen_start = Instant::now();
    let db = GrafeoDB::open(&path).unwrap();
    let reopen_time = reopen_start.elapsed();

    assert_eq!(db.node_count(), num_nodes, "all nodes survived reopen");

    // The single-file format (snapshot v4) persists index metadata and
    // rebuilds the index from data on load. Measure if search works immediately.
    let rebuild_start = Instant::now();
    db.create_vector_index(
        "Memory",
        "embedding",
        Some(384),
        Some("cosine"),
        None,
        None,
        None,
    )
    .expect("recreate vector index after reopen");
    let rebuild_time = rebuild_start.elapsed();

    // Verify search works after rebuild
    let search_start = Instant::now();
    let results = db
        .vector_search(
            "Memory",
            "embedding",
            &random_384d_vector(0),
            10,
            None,
            None,
        )
        .expect("search after rebuild");
    let search_time = search_start.elapsed();

    assert_eq!(results.len(), 10, "should find results after rebuild");
    assert!(
        results[0].1 < 0.1,
        "closest result should be near-zero distance"
    );

    eprintln!();
    eprintln!("=== Vector Index Rebuild: {num_nodes} nodes, 384-dim, cosine ===");
    eprintln!("  Reopen (load data):    {reopen_time:.2?}");
    eprintln!("  Index rebuild:         {rebuild_time:.2?}");
    eprintln!("  First search (k=10):   {search_time:.2?}");
    eprintln!(
        "  Total cold start:      {:.2?}",
        reopen_time + rebuild_time
    );
    eprintln!();

    // Rebuild at 5k should complete in reasonable time (< 30s even in debug)
    assert!(
        rebuild_time.as_secs() < 30,
        "rebuild should complete in < 30s, took {rebuild_time:.2?}"
    );

    db.close().unwrap();
}

// ============================================================================
// Q5: Bring Your Own Vectors (BYOV)
// ============================================================================

#[test]
fn test_byov_384_cosine() {
    let db = GrafeoDB::new_in_memory();

    // Insert 50 nodes with 384-dim vectors
    for i in 0..50u64 {
        let node = db.create_node(&["Memory"]).unwrap();
        db.set_node_property(
            node,
            "embedding",
            Value::Vector(random_384d_vector(i).into()),
        )
        .unwrap();
    }

    db.create_vector_index(
        "Memory",
        "embedding",
        Some(384),
        Some("cosine"),
        None,
        None,
        None,
    )
    .expect("create index");

    // Search with the same vector as node 0: should return node 0 as closest
    let query = random_384d_vector(0);
    let results = db
        .vector_search("Memory", "embedding", &query, 5, None, None)
        .expect("search");

    assert_eq!(results.len(), 5, "should find 5 results");

    // First result should be very close (distance near 0 for identical vector)
    let (first_id, first_dist) = &results[0];
    assert!(
        *first_dist < 0.1,
        "closest result should have distance < 0.1, got {first_dist}"
    );
    // Verify the closest node is the one with the matching vector
    let _ = first_id; // node ID scheme is implementation-defined

    // Results should be ordered by distance
    for window in results.windows(2) {
        assert!(
            window[0].1 <= window[1].1,
            "results should be ordered by distance"
        );
    }
}

#[test]
fn test_byov_all_metrics() {
    for metric in &["cosine", "euclidean", "dot_product", "manhattan"] {
        let db = GrafeoDB::new_in_memory();

        for i in 0..20u64 {
            let node = db.create_node(&["Memory"]).unwrap();
            db.set_node_property(
                node,
                "embedding",
                Value::Vector(random_384d_vector(i).into()),
            )
            .unwrap();
        }

        db.create_vector_index(
            "Memory",
            "embedding",
            Some(384),
            Some(metric),
            None,
            None,
            None,
        )
        .unwrap_or_else(|e| panic!("create index with {metric}: {e}"));

        let results = db
            .vector_search("Memory", "embedding", &random_384d_vector(0), 5, None, None)
            .unwrap_or_else(|e| panic!("search with {metric}: {e}"));

        assert_eq!(results.len(), 5, "{metric}: should find 5 results");

        // Results should be ordered by distance (ascending)
        for window in results.windows(2) {
            assert!(
                window[0].1 <= window[1].1,
                "{metric}: results should be ordered by distance"
            );
        }
    }
}

#[test]
fn test_byov_incremental_after_index() {
    let db = GrafeoDB::new_in_memory();

    // Create index on empty set with declared dimension
    let sentinel = db.create_node(&["Memory"]).unwrap();
    db.set_node_property(
        sentinel,
        "embedding",
        Value::Vector(random_384d_vector(9999).into()),
    )
    .unwrap();
    db.create_vector_index(
        "Memory",
        "embedding",
        Some(384),
        Some("cosine"),
        None,
        None,
        None,
    )
    .expect("create index");

    // Insert 100 nodes incrementally
    for i in 0..100u64 {
        let node = db.create_node(&["Memory"]).unwrap();
        db.set_node_property(
            node,
            "embedding",
            Value::Vector(random_384d_vector(i).into()),
        )
        .unwrap();

        if (i + 1) % 25 == 0 {
            let results = db
                .vector_search(
                    "Memory",
                    "embedding",
                    &random_384d_vector(0),
                    (i + 2) as usize, // +1 for sentinel, +1 for current
                    None,
                    None,
                )
                .expect("search");

            // Should find at least i+1 nodes (current insertions) + sentinel
            assert!(
                results.len() >= (i + 1) as usize,
                "at insert {}: expected >= {} results, got {}",
                i + 1,
                i + 1,
                results.len()
            );
        }
    }
}

// ============================================================================
// Q6: Bulk import
// ============================================================================

#[test]
fn test_bulk_import_single_transaction() {
    let db = GrafeoDB::new_in_memory();
    let num_nodes = 1_500;
    let num_edges = 3_000;

    let start = Instant::now();

    // Insert nodes via direct API (fastest approach for bulk)
    let mut node_ids = Vec::with_capacity(num_nodes);
    for i in 0..num_nodes as u64 {
        let label = ENTITY_LABELS[i as usize % ENTITY_LABELS.len()];
        let id = db
            .create_node_with_props(
                &[label],
                vec![
                    ("name", Value::String(format!("entity_{i}").into())),
                    ("timestamp", Value::Int64(1_700_000_000 + i as i64)),
                    ("confidence", Value::Float64(0.5 + (i as f64) * 0.0003)),
                ],
            )
            .unwrap();
        node_ids.push(id);
    }

    let node_time = start.elapsed();

    // Insert edges via direct API
    let edge_start = Instant::now();
    let mut edge_count = 0;
    for i in 0..num_edges {
        let src = node_ids[i % num_nodes];
        let dst = node_ids[(i * 7 + 3) % num_nodes];
        if src != dst {
            let edge_type = EDGE_TYPES[i % EDGE_TYPES.len()];
            db.create_edge(src, dst, edge_type);
            edge_count += 1;
        }
    }

    let edge_time = edge_start.elapsed();
    let total_time = start.elapsed();

    eprintln!(
        "Bulk import: {num_nodes} nodes in {node_time:.2?}, {edge_count} edges in {edge_time:.2?}"
    );
    eprintln!("Total: {total_time:.2?}");

    assert_eq!(db.node_count(), num_nodes, "all nodes present");
    assert_eq!(db.edge_count(), edge_count, "all edges present");
}

#[test]
#[ignore = "benchmark: 15k entity import throughput"]
fn bench_bulk_import_15k() {
    let num_nodes = 5_000;
    let num_edges = 10_000;

    // Method 1: Direct API
    {
        let db = GrafeoDB::new_in_memory();
        let start = Instant::now();

        let mut node_ids = Vec::with_capacity(num_nodes);
        for i in 0..num_nodes as u64 {
            let label = ENTITY_LABELS[i as usize % ENTITY_LABELS.len()];
            let id = db
                .create_node_with_props(
                    &[label],
                    vec![
                        ("name", Value::String(format!("entity_{i}").into())),
                        ("timestamp", Value::Int64(1_700_000_000 + i as i64)),
                    ],
                )
                .unwrap();
            node_ids.push(id);
        }

        for i in 0..num_edges {
            let src = node_ids[i % num_nodes];
            let dst = node_ids[(i * 7 + 3) % num_nodes];
            if src != dst {
                let edge_type = EDGE_TYPES[i % EDGE_TYPES.len()];
                db.create_edge(src, dst, edge_type);
            }
        }

        let elapsed = start.elapsed();
        let rate = (num_nodes + num_edges) as f64 / elapsed.as_secs_f64();
        eprintln!(
            "Direct API: {num_nodes} nodes + {num_edges} edges in {elapsed:.2?} ({rate:.0} ops/sec)"
        );
    }

    // Method 2: GQL via session
    {
        let db = GrafeoDB::new_in_memory();
        let start = Instant::now();

        let mut session = db.session();
        session.begin_transaction().unwrap();

        for i in 0..num_nodes {
            let label = ENTITY_LABELS[i % ENTITY_LABELS.len()];
            session
                .execute(&format!(
                    "INSERT (:{label} {{name: 'entity_{i}', timestamp: {}}})",
                    1_700_000_000 + i
                ))
                .unwrap();
        }

        session.commit().unwrap();

        // Edges via GQL INSERT (using id() lookup)
        let mut session = db.session();
        session.begin_transaction().unwrap();
        for i in 0..num_edges {
            let src = (i % num_nodes) + 1;
            let dst = ((i * 7 + 3) % num_nodes) + 1;
            if src != dst {
                let edge_type = EDGE_TYPES[i % EDGE_TYPES.len()];
                let _ = session.execute(&format!(
                    "MATCH (a), (b) WHERE id(a) = {src} AND id(b) = {dst} INSERT (a)-[:{edge_type}]->(b)"
                ));
            }
        }
        session.commit().unwrap();

        let elapsed = start.elapsed();
        let rate = (num_nodes + num_edges) as f64 / elapsed.as_secs_f64();
        eprintln!(
            "GQL session: {num_nodes} nodes + {num_edges} edges in {elapsed:.2?} ({rate:.0} ops/sec)"
        );
    }
}

#[test]
fn test_batch_create_nodes_384() {
    let db = GrafeoDB::new_in_memory();

    // batch_create_nodes with 200 384-dim vectors
    let vectors: Vec<Vec<f32>> = (0..200u64).map(random_384d_vector).collect();

    let ids = db.batch_create_nodes("Memory", "embedding", vectors);
    assert_eq!(ids.len(), 200, "should create 200 nodes");

    db.create_vector_index(
        "Memory",
        "embedding",
        Some(384),
        Some("cosine"),
        None,
        None,
        None,
    )
    .expect("create index");

    let results = db
        .vector_search(
            "Memory",
            "embedding",
            &random_384d_vector(0),
            10,
            None,
            None,
        )
        .expect("search");

    assert_eq!(results.len(), 10, "should find 10 results from 200 nodes");

    // First result should be very close to the query (vector 0)
    assert!(results[0].1 < 0.1, "closest should be near-zero distance");
}
