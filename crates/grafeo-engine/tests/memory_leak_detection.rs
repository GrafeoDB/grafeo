//! Memory leak detection tests.
//!
//! Uses Grafeo's memory introspection API to verify that memory usage
//! converges back to stable levels after repeated create/update/delete/GC
//! cycles. Targets the highest-risk area in any graph database: MVCC
//! version chain accumulation, session lifecycle, and index cleanup.
//!
//! The long-running stress variants are marked `#[ignore]` for CI.
//!
//! ```bash
//! cargo test -p grafeo-engine --test main memory_leak_detection::
//! cargo test -p grafeo-engine --test main memory_leak_detection:: -- --ignored
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use grafeo_engine::GrafeoDB;

// ============================================================================
// MVCC Version Chain Leak Detection
// ============================================================================

/// After creating and deleting nodes in committed transactions, then running
/// GC, the MVCC overhead should not grow without bound. This catches version
/// chains that survive GC due to epoch tracking bugs.
#[test]
fn test_version_chains_stabilize_after_gc() {
    let db = GrafeoDB::new_in_memory();

    // Warm-up: seed some data so structural overhead is accounted for.
    {
        let s = db.session();
        s.execute("INSERT (:Warmup {val: 0})").unwrap();
        s.execute("MATCH (w:Warmup) DELETE w").unwrap();
    }
    db.gc();
    let baseline = db.memory_usage();

    // Run 50 create-update-delete cycles, each in its own committed transaction.
    for i in 0..50 {
        let mut s = db.session();
        s.begin_transaction().unwrap();
        s.execute(&format!("INSERT (:Ephemeral {{round: {i}}})"))
            .unwrap();
        s.commit().unwrap();

        let mut s2 = db.session();
        s2.begin_transaction().unwrap();
        s2.execute("MATCH (e:Ephemeral) SET e.touched = true")
            .unwrap();
        s2.commit().unwrap();

        let mut s3 = db.session();
        s3.begin_transaction().unwrap();
        s3.execute("MATCH (e:Ephemeral) DELETE e").unwrap();
        s3.commit().unwrap();
    }

    // Force GC with no active transactions, so min_epoch == current_epoch.
    db.gc();

    let after = db.memory_usage();

    // Version chain depth should be small: all entities were deleted and GC'd.
    assert!(
        after.mvcc.max_chain_depth <= 2,
        "max chain depth should converge after GC, got {}",
        after.mvcc.max_chain_depth
    );

    // MVCC overhead should not have grown more than 4x from the baseline.
    // (Some overhead from the string pool and structural maps is expected.)
    let mvcc_growth = after
        .mvcc
        .total_bytes
        .saturating_sub(baseline.mvcc.total_bytes);
    assert!(
        mvcc_growth < 64 * 1024,
        "MVCC memory grew by {mvcc_growth} bytes after 50 create/delete cycles + GC, expected < 64 KiB"
    );
}

/// Repeated property updates on the same node should not cause unbounded
/// version chain growth when GC runs between batches.
#[test]
fn test_property_update_chains_cleaned_by_gc() {
    let db = GrafeoDB::new_in_memory();
    let setup = db.session();
    setup.execute("INSERT (:Counter {value: 0})").unwrap();

    // Run 5 rounds of 20 updates each, with GC between rounds.
    for round in 0..5 {
        for i in 0..20 {
            let mut s = db.session();
            s.begin_transaction().unwrap();
            s.execute(&format!(
                "MATCH (c:Counter) SET c.value = {}",
                round * 20 + i + 1
            ))
            .unwrap();
            s.commit().unwrap();
        }
        db.gc();
    }

    let usage = db.memory_usage();

    // After 100 updates with periodic GC, chain depth should be bounded.
    assert!(
        usage.mvcc.max_chain_depth <= 3,
        "expected chain depth <= 3 after periodic GC, got {}",
        usage.mvcc.max_chain_depth
    );
}

// ============================================================================
// Session Lifecycle Leak Detection
// ============================================================================

/// Creating and dropping many sessions should not cause memory to grow
/// indefinitely. This catches leaked transaction manager entries,
/// session metadata, or query cache bloat.
#[test]
fn test_session_lifecycle_no_leak() {
    let db = GrafeoDB::new_in_memory();

    // Warm up the query cache with one session.
    {
        let s = db.session();
        s.execute("INSERT (:Anchor {id: 1})").unwrap();
        s.execute("MATCH (a:Anchor) RETURN a.id").unwrap();
    }
    db.gc();
    let baseline = db.memory_usage();

    // Create and drop 200 sessions, each executing a read query.
    for _ in 0..200 {
        let s = db.session();
        s.execute("MATCH (a:Anchor) RETURN a.id").unwrap();
        // session dropped here
    }
    db.gc();
    let after = db.memory_usage();

    // Total memory growth should be minimal: no new data was created, and the
    // query cache should have a bounded size.
    let growth = after.total_bytes.saturating_sub(baseline.total_bytes);
    assert!(
        growth < 128 * 1024,
        "memory grew by {growth} bytes after 200 read-only sessions, expected < 128 KiB"
    );
}

/// Sessions with uncommitted transactions (auto-rolled-back on drop) should
/// not leave version chain debris.
#[test]
fn test_abandoned_transaction_cleanup() {
    let db = GrafeoDB::new_in_memory();
    db.gc();
    let baseline = db.memory_usage();

    // Create 100 sessions that begin a transaction and write, but never commit.
    for i in 0..100 {
        let mut s = db.session();
        s.begin_transaction().unwrap();
        s.execute(&format!("INSERT (:Ghost {{id: {i}}})")).unwrap();
        // session dropped without commit: auto-rollback
    }
    db.gc();

    let after = db.memory_usage();

    // No nodes should exist: all were rolled back.
    let verify = db.session();
    let result = verify.execute("MATCH (g:Ghost) RETURN g").unwrap();
    assert_eq!(
        result.row_count(),
        0,
        "all Ghost nodes should have been rolled back"
    );

    // MVCC overhead should be close to baseline.
    let mvcc_growth = after
        .mvcc
        .total_bytes
        .saturating_sub(baseline.mvcc.total_bytes);
    assert!(
        mvcc_growth < 32 * 1024,
        "MVCC grew by {mvcc_growth} bytes after 100 abandoned transactions, expected < 32 KiB"
    );
}

// ============================================================================
// Index Cleanup Leak Detection
// ============================================================================

/// Adjacency list memory should not grow unboundedly when edges are
/// repeatedly created and deleted.
#[test]
fn test_adjacency_index_cleanup_after_edge_churn() {
    let db = GrafeoDB::new_in_memory();
    let setup = db.session();
    setup.execute("INSERT (:Hub {name: 'center'})").unwrap();
    setup.execute("INSERT (:Spoke {name: 'target'})").unwrap();
    db.gc();
    let baseline = db.memory_usage();

    // 30 rounds: create an edge, then delete it.
    for _ in 0..30 {
        let mut s = db.session();
        s.begin_transaction().unwrap();
        s.execute("MATCH (h:Hub), (s:Spoke) INSERT (h)-[:LINK]->(s)")
            .unwrap();
        s.commit().unwrap();

        let mut s2 = db.session();
        s2.begin_transaction().unwrap();
        s2.execute("MATCH ()-[l:LINK]->() DELETE l").unwrap();
        s2.commit().unwrap();
    }
    db.gc();

    let after = db.memory_usage();
    let adj_growth = after
        .indexes
        .forward_adjacency_bytes
        .saturating_sub(baseline.indexes.forward_adjacency_bytes);

    // Adjacency overhead should be bounded: deleted edges should not
    // accumulate stale entries after GC.
    assert!(
        adj_growth < 32 * 1024,
        "forward adjacency grew by {adj_growth} bytes after 30 edge create/delete cycles, expected < 32 KiB"
    );
}

/// Label index memory should not grow when nodes with labels are
/// repeatedly created and deleted.
#[test]
fn test_label_index_cleanup_after_node_churn() {
    let db = GrafeoDB::new_in_memory();
    db.gc();
    let baseline = db.memory_usage();

    for _ in 0..50 {
        let mut s = db.session();
        s.begin_transaction().unwrap();
        s.execute("INSERT (:Transient {val: 1})").unwrap();
        s.commit().unwrap();

        let mut s2 = db.session();
        s2.begin_transaction().unwrap();
        s2.execute("MATCH (t:Transient) DELETE t").unwrap();
        s2.commit().unwrap();
    }
    db.gc();

    let after = db.memory_usage();
    let label_growth = after
        .indexes
        .label_index_bytes
        .saturating_sub(baseline.indexes.label_index_bytes);

    assert!(
        label_growth < 32 * 1024,
        "label index grew by {label_growth} bytes after 50 node create/delete cycles, expected < 32 KiB"
    );
}

// ============================================================================
// Stress Tests (long-running, run with --ignored)
// ============================================================================

/// High-volume concurrent workload: many threads each running full
/// create/update/delete cycles. Memory should converge to a stable level.
#[test]
#[ignore = "long-running stress test: run locally before releases"]
fn test_stress_concurrent_workload_memory_convergence() {
    let db = Arc::new(GrafeoDB::new_in_memory());

    // Seed data.
    {
        let s = db.session();
        for i in 0..10 {
            s.execute(&format!("INSERT (:Seed {{id: {i}}})")).unwrap();
        }
    }
    db.gc();

    // Record memory after 3 rounds and check convergence.
    let mut snapshots = Vec::new();

    for round in 0..3 {
        let num_threads = 8;
        let barrier = Arc::new(Barrier::new(num_threads));
        let completed = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..num_threads)
            .map(|t| {
                let db = Arc::clone(&db);
                let barrier = Arc::clone(&barrier);
                let completed = Arc::clone(&completed);

                thread::spawn(move || {
                    barrier.wait();

                    for i in 0..50 {
                        let label = format!("R{round}T{t}");

                        let mut s = db.session();
                        s.begin_transaction().unwrap();
                        s.execute(&format!("INSERT (:{label} {{iter: {i}}})"))
                            .unwrap();
                        s.commit().unwrap();

                        let mut s2 = db.session();
                        s2.begin_transaction().unwrap();
                        s2.execute(&format!("MATCH (n:{label}) DELETE n")).unwrap();
                        s2.commit().unwrap();
                    }

                    completed.fetch_add(1, Ordering::Relaxed);
                })
            })
            .collect();

        for h in handles {
            h.join().expect("worker thread panicked");
        }

        db.gc();
        snapshots.push(db.memory_usage().total_bytes);
    }

    // Memory between round 2 and round 3 should not grow by more than 25%.
    // If it does, something is accumulating.
    let round2 = snapshots[1];
    let round3 = snapshots[2];
    let growth_pct = if round2 > 0 {
        // reason: Percentage growth is always small, fits i64
        #[allow(clippy::cast_possible_truncation)]
        {
            ((round3 as f64 - round2 as f64) / round2 as f64 * 100.0) as i64
        }
    } else {
        0
    };

    assert!(
        growth_pct < 25,
        "memory grew {growth_pct}% between stress rounds 2 and 3 (round2={round2}, round3={round3}), expected < 25%"
    );
}

/// Sustained single-threaded workload over many iterations to detect slow
/// leaks that only manifest at scale.
#[test]
#[ignore = "long-running stress test: run locally before releases"]
fn test_stress_sustained_workload_slow_leak_detection() {
    let db = GrafeoDB::new_in_memory();

    // Warm-up phase: 100 iterations.
    for i in 0..100 {
        let mut s = db.session();
        s.begin_transaction().unwrap();
        s.execute(&format!("INSERT (:Warmup {{id: {i}}})")).unwrap();
        s.commit().unwrap();

        let mut s2 = db.session();
        s2.begin_transaction().unwrap();
        s2.execute("MATCH (w:Warmup) DELETE w").unwrap();
        s2.commit().unwrap();
    }
    db.gc();
    let baseline = db.memory_usage().total_bytes;

    // Sustained phase: 500 more iterations with GC every 50.
    for i in 0..500 {
        let mut s = db.session();
        s.begin_transaction().unwrap();
        s.execute(&format!("INSERT (:Work {{id: {i}}})")).unwrap();
        s.commit().unwrap();

        let mut s2 = db.session();
        s2.begin_transaction().unwrap();
        s2.execute("MATCH (w:Work) SET w.touched = true").unwrap();
        s2.commit().unwrap();

        let mut s3 = db.session();
        s3.begin_transaction().unwrap();
        s3.execute("MATCH (w:Work) DELETE w").unwrap();
        s3.commit().unwrap();

        if (i + 1) % 50 == 0 {
            db.gc();
        }
    }
    db.gc();
    let final_usage = db.memory_usage().total_bytes;

    // After 500 full cycles with periodic GC, memory should not have grown
    // more than 256 KiB from the post-warmup baseline.
    let growth = final_usage.saturating_sub(baseline);
    assert!(
        growth < 256 * 1024,
        "memory grew by {growth} bytes over 500 sustained cycles, expected < 256 KiB"
    );
}

/// Graph-specific pattern: star topology churn. Creates a hub node and many
/// spokes, deletes everything, repeats. Catches adjacency list fragmentation
/// and edge index leaks.
#[test]
#[ignore = "long-running stress test: run locally before releases"]
fn test_stress_star_topology_churn() {
    let db = GrafeoDB::new_in_memory();

    // Warm up.
    {
        let s = db.session();
        s.execute("INSERT (:Setup)").unwrap();
    }
    db.gc();
    let baseline = db.memory_usage();

    for round in 0..20 {
        // Create a hub with 50 spokes.
        let mut s = db.session();
        s.begin_transaction().unwrap();
        s.execute(&format!("INSERT (:Hub {{round: {round}}})"))
            .unwrap();
        for spoke in 0..50 {
            s.execute(&format!(
                "MATCH (h:Hub {{round: {round}}}) INSERT (h)-[:SPOKE]->(:Leaf {{id: {spoke}}})"
            ))
            .unwrap();
        }
        s.commit().unwrap();

        // Delete the entire star.
        let mut s2 = db.session();
        s2.begin_transaction().unwrap();
        s2.execute(&format!(
            "MATCH (h:Hub {{round: {round}}})-[r:SPOKE]->(l:Leaf) DELETE r, l"
        ))
        .unwrap();
        s2.execute(&format!("MATCH (h:Hub {{round: {round}}}) DELETE h"))
            .unwrap();
        s2.commit().unwrap();

        // GC every 5 rounds.
        if (round + 1) % 5 == 0 {
            db.gc();
        }
    }
    db.gc();

    let after = db.memory_usage();

    // After creating and deleting 20 star topologies (1000+ nodes, 1000 edges),
    // memory should converge back near baseline. Some residual overhead is
    // expected: Rust HashMaps do not shrink bucket arrays after removal, so
    // the store maps retain their high-water-mark capacity.
    let growth = after.total_bytes.saturating_sub(baseline.total_bytes);
    assert!(
        growth < 2 * 1024 * 1024,
        "memory grew by {growth} bytes after 20 star topology cycles, expected < 2 MiB"
    );

    // Adjacency specifically should be bounded.
    let adj_growth = after
        .indexes
        .forward_adjacency_bytes
        .saturating_sub(baseline.indexes.forward_adjacency_bytes);
    assert!(
        adj_growth < 128 * 1024,
        "adjacency index grew by {adj_growth} bytes after star churn, expected < 128 KiB"
    );
}
