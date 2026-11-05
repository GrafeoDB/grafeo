//! Epoch monotonicity and CDC correctness tests.
//!
//! Validates the three invariants that replication depends on:
//! 1. Strict epoch ordering in `changes_between()` output
//! 2. No gaps: all committed events appear in range queries
//! 3. Range correctness: only events within bounds are returned
//!
//! These tests exercise the epoch counter (`AtomicU64` with `SeqCst`)
//! and the CDC log (`RwLock<HashMap>`) under concurrent workloads.
//!
//! ```bash
//! cargo test --features "full" -p grafeo-engine --test main epoch_monotonicity:: -- --nocapture
//! ```

#![cfg(all(feature = "cdc", feature = "gql"))]

use std::collections::HashSet;
use std::sync::{Arc, Barrier};
use std::thread;

use grafeo_common::types::EpochId;
use grafeo_engine::cdc::{ChangeKind, EntityId};
use grafeo_engine::{Config, GrafeoDB};

// ============================================================================
// Test 1: Concurrent commits produce strictly increasing epochs
// ============================================================================

#[test]
fn concurrent_commits_produce_strictly_increasing_epochs() {
    let db = Arc::new(GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap());
    let num_threads = 2;
    let ops_per_thread = 5;
    let barrier = Arc::new(Barrier::new(num_threads));

    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                for i in 0..ops_per_thread {
                    let session = db.session();
                    session
                        .execute(&format!("INSERT (:Worker {{tid: {tid}, seq: {i}}})"))
                        .unwrap();
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread panicked");
    }

    let changes = db
        .changes_between(EpochId::new(0), EpochId::new(u64::MAX))
        .unwrap();

    // Invariant 1: events are sorted by epoch
    for window in changes.windows(2) {
        assert!(
            window[0].epoch <= window[1].epoch,
            "Events out of order: epoch {} followed by epoch {}",
            window[0].epoch.as_u64(),
            window[1].epoch.as_u64(),
        );
    }

    // Invariant 2: each auto-commit transaction gets a unique epoch.
    // Multiple events per transaction (Create node + Update properties) share
    // the same epoch, so we check uniqueness per NODE Create event's entity ID.
    let create_node_events: Vec<_> = changes
        .iter()
        .filter(|e| e.kind == ChangeKind::Create && matches!(e.entity_id, EntityId::Node(_)))
        .collect();

    let expected = num_threads * ops_per_thread;
    assert_eq!(
        create_node_events.len(),
        expected,
        "Expected {expected} node Create events, got {}",
        create_node_events.len()
    );

    // Each node creation should have a unique entity ID
    let unique_ids: HashSet<u64> = create_node_events
        .iter()
        .map(|e| e.entity_id.as_u64())
        .collect();
    assert_eq!(
        unique_ids.len(),
        expected,
        "Expected {expected} unique node IDs, got {} (duplicates)",
        unique_ids.len()
    );

    // Each auto-commit INSERT is its own transaction with a unique commit epoch
    // assigned via fetch_add(1, SeqCst). All buffered CDC events get the commit
    // epoch at flush time, so each node Create should have a distinct epoch.
    let unique_epochs: HashSet<u64> = create_node_events
        .iter()
        .map(|e| e.epoch.as_u64())
        .collect();
    assert_eq!(
        unique_epochs.len(),
        expected,
        "Expected {expected} distinct epochs (one per auto-commit), got {}",
        unique_epochs.len()
    );
}

// ============================================================================
// Test 2: changes_between with overlapping ranges has no gaps
// ============================================================================

#[test]
fn changes_between_no_gaps() {
    let db = GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap();
    let session = db.session();

    // Insert 10 nodes, each in its own auto-commit statement
    let mut epochs = Vec::new();
    for i in 0..10 {
        session
            .execute(&format!("INSERT (:Item {{seq: {i}}})"))
            .unwrap();
        epochs.push(db.current_epoch());
    }

    // Full range
    let all = db
        .changes_between(EpochId::new(0), EpochId::new(u64::MAX))
        .unwrap();
    let all_count = all.iter().filter(|e| e.kind == ChangeKind::Create).count();
    assert_eq!(all_count, 10, "Full range should have 10 creates");

    // First half
    let first_half = db.changes_between(EpochId::new(0), epochs[4]).unwrap();

    // Second half
    let second_half = db
        .changes_between(EpochId::new(epochs[4].as_u64() + 1), EpochId::new(u64::MAX))
        .unwrap();

    // Union of halves should equal full range
    let first_ids: HashSet<u64> = first_half.iter().map(|e| e.entity_id.as_u64()).collect();
    let second_ids: HashSet<u64> = second_half.iter().map(|e| e.entity_id.as_u64()).collect();
    let all_ids: HashSet<u64> = all.iter().map(|e| e.entity_id.as_u64()).collect();

    let union: HashSet<u64> = first_ids.union(&second_ids).copied().collect();
    assert_eq!(
        union, all_ids,
        "Union of non-overlapping halves must equal full range"
    );
}

// ============================================================================
// Test 3: Range bounds are strict
// ============================================================================

#[test]
fn changes_between_range_bounds_strict() {
    let db = GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap();
    let session = db.session();

    // Insert 5 nodes, capture commit epoch of each
    let mut commit_epochs = Vec::new();
    for i in 0..5 {
        session
            .execute(&format!("INSERT (:Bound {{seq: {i}}})"))
            .unwrap();
        commit_epochs.push(db.current_epoch());
    }

    // Query middle range [epoch_1, epoch_3]
    let middle = db
        .changes_between(commit_epochs[1], commit_epochs[3])
        .unwrap();

    // All returned events must be within bounds
    for event in &middle {
        assert!(
            event.epoch >= commit_epochs[1] && event.epoch <= commit_epochs[3],
            "Event epoch {} outside range [{}, {}]",
            event.epoch.as_u64(),
            commit_epochs[1].as_u64(),
            commit_epochs[3].as_u64(),
        );
    }

    // Events at epoch_0 and epoch_4 must NOT be in the result
    let middle_epochs: HashSet<u64> = middle.iter().map(|e| e.epoch.as_u64()).collect();
    assert!(
        !middle_epochs.contains(&commit_epochs[0].as_u64()),
        "Epoch {} (before range) should be excluded",
        commit_epochs[0].as_u64()
    );
    assert!(
        !middle_epochs.contains(&commit_epochs[4].as_u64()),
        "Epoch {} (after range) should be excluded",
        commit_epochs[4].as_u64()
    );
}

// ============================================================================
// Test 4: Concurrent sessions CDC event count matches
// ============================================================================

#[test]
fn concurrent_sessions_cdc_event_count_matches() {
    let db = Arc::new(GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap());
    let num_threads = 2;
    let nodes_per_thread = 5;
    let barrier = Arc::new(Barrier::new(num_threads));

    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                for i in 0..nodes_per_thread {
                    let session = db.session();
                    session
                        .execute(&format!("INSERT (:Counter {{tid: {tid}, seq: {i}}})"))
                        .unwrap();
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread panicked");
    }

    let changes = db
        .changes_between(EpochId::new(0), EpochId::new(u64::MAX))
        .unwrap();

    let create_count = changes
        .iter()
        .filter(|e| e.kind == ChangeKind::Create)
        .count();
    let expected = num_threads * nodes_per_thread;
    assert_eq!(
        create_count, expected,
        "Expected {expected} Create events, got {create_count}"
    );

    // No duplicate entity IDs
    let entity_ids: HashSet<u64> = changes
        .iter()
        .filter(|e| e.kind == ChangeKind::Create)
        .map(|e| e.entity_id.as_u64())
        .collect();
    assert_eq!(
        entity_ids.len(),
        expected,
        "Expected {expected} unique entity IDs, got {} (duplicates)",
        entity_ids.len()
    );
}

// ============================================================================
// Test 5: Transaction rollback leaves no epoch holes
// ============================================================================

#[test]
fn transaction_rollback_leaves_no_epoch_holes() {
    let db = GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap();

    // Commit 3
    for i in 0..3 {
        let session = db.session();
        session
            .execute(&format!("INSERT (:Committed {{seq: {i}}})"))
            .unwrap();
    }

    // Rollback 1
    {
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session
            .execute("INSERT (:RolledBack {seq: 'rb1'})")
            .unwrap();
        session.rollback().unwrap();
    }

    // Commit 2 more
    for i in 3..5 {
        let session = db.session();
        session
            .execute(&format!("INSERT (:Committed {{seq: {i}}})"))
            .unwrap();
    }

    // Rollback another
    {
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session
            .execute("INSERT (:RolledBack {seq: 'rb2'})")
            .unwrap();
        session.rollback().unwrap();
    }

    let changes = db
        .changes_between(EpochId::new(0), EpochId::new(u64::MAX))
        .unwrap();

    // Only committed events
    let create_events: Vec<_> = changes
        .iter()
        .filter(|e| e.kind == ChangeKind::Create && matches!(e.entity_id, EntityId::Node(_)))
        .collect();
    assert_eq!(
        create_events.len(),
        5,
        "Should have exactly 5 committed Create events, got {}",
        create_events.len()
    );

    // Verify epoch ordering is monotonically increasing (no disorder from rollbacks)
    for window in create_events.windows(2) {
        assert!(
            window[0].epoch < window[1].epoch,
            "Committed events should have strictly increasing epochs: {} then {}",
            window[0].epoch.as_u64(),
            window[1].epoch.as_u64(),
        );
    }
}
