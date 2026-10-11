//! Concurrent Sessions Integration Tests
//!
//! Tests for multi-session concurrent access patterns:
//! - Multiple sessions executing queries simultaneously
//! - Thread-safe shared database access
//! - Transaction isolation across sessions
//! - Race condition handling

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use grafeo_engine::GrafeoDB;

// ============================================================================
// Concurrent Session Access Tests
// ============================================================================

#[test]
fn test_concurrent_read_sessions() {
    // Multiple sessions reading simultaneously should not block.
    // 2 threads to stay within nextest timeout on 2-core CI runners.
    let db = Arc::new(GrafeoDB::new_in_memory());

    {
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        session.execute("INSERT (:Person {name: 'Harm'})").unwrap();
    }

    let num_threads = 2;
    let barrier = Arc::new(Barrier::new(num_threads));
    let success_count = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            let success_count = Arc::clone(&success_count);

            thread::spawn(move || {
                barrier.wait();

                let session = db.session();
                let result = session.execute("MATCH (n:Person) RETURN n.name").unwrap();

                if result.row_count() == 3 {
                    success_count.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    assert_eq!(
        success_count.load(Ordering::Relaxed),
        num_threads,
        "All concurrent reads should succeed"
    );
}

#[test]
fn test_concurrent_write_sessions() {
    // Multiple sessions writing to different entities should succeed.
    // Kept at 2 threads to stay within the nextest 60s timeout on
    // resource-constrained 2-core CI runners.
    let db = Arc::new(GrafeoDB::new_in_memory());

    let num_threads = 2;
    let barrier = Arc::new(Barrier::new(num_threads));
    let success_count = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..num_threads)
        .map(|i| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            let success_count = Arc::clone(&success_count);

            thread::spawn(move || {
                barrier.wait();

                let session = db.session();
                let query = format!("INSERT (:Thread{} {{id: {}}})", i, i);
                if session.execute(&query).is_ok() {
                    success_count.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    assert_eq!(
        success_count.load(Ordering::Relaxed),
        num_threads,
        "All concurrent writes to different entities should succeed"
    );

    let session = db.session();
    for i in 0..num_threads {
        let query = format!("MATCH (n:Thread{}) RETURN n", i);
        let result = session.execute(&query).unwrap();
        assert_eq!(
            result.row_count(),
            1,
            "Node for thread {} should exist exactly once",
            i
        );
    }
}

#[test]
fn test_session_isolation_between_threads() {
    // Changes in one session's transaction should not be visible to other sessions
    // until committed
    let db = Arc::new(GrafeoDB::new_in_memory());

    // Writer thread creates data in a transaction
    let db_clone = Arc::clone(&db);
    let writer_started = Arc::new(Barrier::new(2));
    let reader_check = Arc::new(Barrier::new(2));
    let writer_done = Arc::new(Barrier::new(2));

    let writer_started_clone = Arc::clone(&writer_started);
    let reader_check_clone = Arc::clone(&reader_check);
    let writer_done_clone = Arc::clone(&writer_done);

    let writer_handle = thread::spawn(move || {
        let mut session = db_clone.session();
        session.begin_transaction().unwrap();

        // Create a node within the transaction
        session
            .execute("INSERT (:IsolatedNode {secret: 'hidden'})")
            .unwrap();

        // Signal that writer has created the node
        writer_started_clone.wait();

        // Wait for reader to check
        reader_check_clone.wait();

        // Now commit
        session.commit().unwrap();

        // Signal done
        writer_done_clone.wait();
    });

    // Reader thread checks visibility
    let reader_handle = thread::spawn(move || {
        let session = db.session();

        // Wait for writer to create node (but not commit)
        writer_started.wait();

        // Check if we can see the node (we shouldn't - transaction not committed)
        // Note: This test checks the expected behavior when MVCC is fully integrated
        let result = session.execute("MATCH (n:IsolatedNode) RETURN n").unwrap();
        let before_commit_count = result.row_count();

        // Signal that reader has checked
        reader_check.wait();

        // Wait for writer to commit
        writer_done.wait();

        // Now we should see the node (after commit)
        let result = session.execute("MATCH (n:IsolatedNode) RETURN n").unwrap();
        let after_commit_count = result.row_count();

        (before_commit_count, after_commit_count)
    });

    writer_handle.join().expect("Writer thread panicked");
    let (before, after) = reader_handle.join().expect("Reader thread panicked");

    // After commit, the node should be visible
    assert_eq!(after, 1, "Node should be visible after commit");

    // Uncommitted data uses PENDING epoch, invisible to other sessions.
    assert_eq!(
        before, 0,
        "Dirty read prevented: uncommitted data is invisible to other sessions"
    );
}

// ============================================================================
// Stress Tests
// ============================================================================

#[test]
fn test_many_sessions_rapid_creation() {
    // Creating many sessions rapidly should not cause issues.
    // 2 threads to stay within nextest timeout on 2-core CI runners.
    let db = Arc::new(GrafeoDB::new_in_memory());

    let num_threads = 2;
    let sessions_per_thread = 20;
    let barrier = Arc::new(Barrier::new(num_threads));
    let success_count = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            let success_count = Arc::clone(&success_count);

            thread::spawn(move || {
                barrier.wait();

                for _ in 0..sessions_per_thread {
                    let session = db.session();
                    // Just creating and dropping sessions
                    drop(session);
                }
                success_count.fetch_add(1, Ordering::Relaxed);
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    assert_eq!(
        success_count.load(Ordering::Relaxed),
        num_threads,
        "All threads should complete without panic"
    );
}

#[test]
fn test_interleaved_transactions() {
    // Multiple sessions with interleaved transaction operations.
    // Kept lightweight (2 threads, 3 iterations) to avoid lock-contention
    // slowdowns on resource-constrained CI runners.
    let db = Arc::new(GrafeoDB::new_in_memory());

    let completed = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..2)
        .map(|thread_id| {
            let db = Arc::clone(&db);
            let completed = Arc::clone(&completed);

            thread::spawn(move || {
                for i in 0..3 {
                    let mut session = db.session();

                    session.begin_transaction().unwrap();

                    let query =
                        format!("INSERT (:Work {{thread: {}, iteration: {}}})", thread_id, i);
                    let _ = session.execute(&query);

                    if i % 3 == 0 {
                        let _ = session.rollback();
                    } else {
                        let _ = session.commit();
                    }
                }

                completed.fetch_add(1, Ordering::Relaxed);
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    assert_eq!(
        completed.load(Ordering::Relaxed),
        2,
        "All threads should complete"
    );
}

// ============================================================================
// Session State Tests
// ============================================================================

#[test]
fn test_session_transaction_state_independence() {
    // Each session should maintain independent transaction state
    let db = GrafeoDB::new_in_memory();

    let mut session1 = db.session();
    let mut session2 = db.session();

    // Session 1 starts transaction
    session1.begin_transaction().unwrap();
    assert!(session1.in_transaction());
    assert!(!session2.in_transaction());

    // Session 2 starts its own transaction
    session2.begin_transaction().unwrap();
    assert!(session1.in_transaction());
    assert!(session2.in_transaction());

    // Session 1 commits
    session1.commit().unwrap();
    assert!(!session1.in_transaction());
    assert!(session2.in_transaction());

    // Session 2 rolls back
    session2.rollback().unwrap();
    assert!(!session1.in_transaction());
    assert!(!session2.in_transaction());
}

#[test]
#[expect(deprecated, reason = "the deprecated setting is what this tests")]
fn test_session_auto_commit_independence() {
    // Auto-commit setting should be independent per session
    let db = GrafeoDB::new_in_memory();

    let mut session1 = db.session();
    let session2 = db.session();

    assert!(session1.auto_commit());
    assert!(session2.auto_commit());

    session1.set_auto_commit(false);

    assert!(!session1.auto_commit());
    assert!(session2.auto_commit());
}

// ============================================================================
// Database Shared State Tests
// ============================================================================

#[test]
fn test_sessions_share_committed_data() {
    // Data committed by one session should be visible to others
    let db = GrafeoDB::new_in_memory();

    let session1 = db.session();
    let session2 = db.session();

    // Session 1 creates and commits data
    session1.execute("INSERT (:Shared {key: 'value'})").unwrap();

    // Session 2 should see the data
    let result = session2.execute("MATCH (n:Shared) RETURN n.key").unwrap();
    assert_eq!(
        result.row_count(),
        1,
        "Session 2 should see committed data from Session 1"
    );
}

#[test]
fn test_node_count_consistency() {
    // Node count should be consistent across sessions
    let db = GrafeoDB::new_in_memory();

    // Create nodes from multiple sessions
    for i in 0..10 {
        let session = db.session();
        let query = format!("INSERT (:CountTest{{id: {}}})", i);
        session.execute(&query).unwrap();
    }

    // Check count from a new session
    let session = db.session();
    let result = session.execute("MATCH (n:CountTest) RETURN n").unwrap();
    assert_eq!(result.row_count(), 10, "Should see all 10 nodes");
}

// ============================================================================
// Async Session Tests (using tokio)
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_async_concurrent_sessions() {
    // Kept lightweight (3 tasks) to avoid lock-contention slowdowns
    // on resource-constrained CI runners (2-core GitHub Actions).
    use tokio::task;

    let db = Arc::new(GrafeoDB::new_in_memory());
    let num_tasks = 3;

    // Spawn async tasks
    let mut handles = Vec::new();

    for i in 0..num_tasks {
        let db: Arc<GrafeoDB> = Arc::clone(&db);
        handles.push(task::spawn_blocking(move || {
            let session = db.session();
            let query = format!("INSERT (:AsyncNode {{id: {}}})", i);
            session.execute(&query).unwrap();
        }));
    }

    // Wait for all tasks
    for handle in handles {
        handle.await.expect("Task panicked");
    }

    // Verify results
    let session = db.session();
    let result = session.execute("MATCH (n:AsyncNode) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        num_tasks,
        "All async nodes should exist"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_async_transaction_isolation() {
    use std::sync::atomic::AtomicBool;
    use tokio::task;

    let db = Arc::new(GrafeoDB::new_in_memory());
    let writer_committed = Arc::new(AtomicBool::new(false));

    // Writer task
    let db_writer: Arc<GrafeoDB> = Arc::clone(&db);
    let committed_flag = Arc::clone(&writer_committed);

    let writer = task::spawn_blocking(move || {
        let mut session = db_writer.session();
        session.begin_transaction().unwrap();
        session
            .execute("INSERT (:AsyncIsolated {data: 'test'})")
            .unwrap();
        session.commit().unwrap();
        committed_flag.store(true, Ordering::Release);
    });

    // Reader task: waits for writer commit via atomic flag, no sleep
    let db_reader: Arc<GrafeoDB> = Arc::clone(&db);
    let reader_flag = Arc::clone(&writer_committed);

    let reader = task::spawn_blocking(move || {
        // Spin until writer has committed
        while !reader_flag.load(Ordering::Acquire) {
            std::hint::spin_loop();
        }

        let session = db_reader.session();
        let result = session.execute("MATCH (n:AsyncIsolated) RETURN n").unwrap();
        result.row_count()
    });

    writer.await.expect("Writer task panicked");
    let count = reader.await.expect("Reader task panicked");

    assert_eq!(count, 1, "Should see committed data after writer completes");
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_session_after_transaction_error() {
    // Session should be usable after a transaction error
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    // Try to commit without transaction (should error)
    let result = session.commit();
    assert!(result.is_err());

    // Session should still work
    session.begin_transaction().unwrap();
    session.execute("INSERT (:AfterError)").unwrap();
    session.commit().unwrap();

    let result = session.execute("MATCH (n:AfterError) RETURN n").unwrap();
    assert_eq!(result.row_count(), 1);
}

#[test]
fn test_multiple_sequential_transactions() {
    // Same session should handle multiple sequential transactions
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    for i in 0..5 {
        session.begin_transaction().unwrap();
        let query = format!("INSERT (:Sequential{{iteration: {}}})", i);
        session.execute(&query).unwrap();
        session.commit().unwrap();
    }

    let result = session.execute("MATCH (n:Sequential) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        5,
        "All 5 sequential transactions should have committed"
    );
}

// ============================================================================
// Concurrent Stress Tests
// ============================================================================

#[test]
#[ignore = "stress test: slow in CI, run locally with --ignored"]
fn test_stress_concurrent_writers() {
    // 8 threads each inserting 50 nodes simultaneously
    let db = Arc::new(GrafeoDB::new_in_memory());

    let num_threads = 8;
    let writes_per_thread = 50;
    let barrier = Arc::new(Barrier::new(num_threads));
    let success_count = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            let success_count = Arc::clone(&success_count);

            thread::spawn(move || {
                barrier.wait();
                for i in 0..writes_per_thread {
                    let session = db.session();
                    let query = format!("INSERT (:Stress {{thread: {tid}, seq: {i}}})");
                    session.execute(&query).unwrap();
                }
                success_count.fetch_add(1, Ordering::Relaxed);
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    assert_eq!(success_count.load(Ordering::Relaxed), num_threads);

    // Verify total node count
    let session = db.session();
    let result = session.execute("MATCH (n:Stress) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        num_threads * writes_per_thread,
        "All nodes should be created"
    );
}

#[test]
#[ignore = "stress test: slow in CI, run locally with --ignored"]
fn test_stress_concurrent_reads_during_writes() {
    // Mixed workload: 4 writers + 8 readers operating simultaneously
    let db = Arc::new(GrafeoDB::new_in_memory());

    // Seed initial data
    {
        let session = db.session();
        for i in 0..100 {
            session
                .execute(&format!("INSERT (:Item {{id: {i}}})"))
                .unwrap();
        }
    }

    let num_writers = 4;
    let num_readers = 8;
    let barrier = Arc::new(Barrier::new(num_writers + num_readers));
    let read_errors = Arc::new(AtomicUsize::new(0));
    let write_errors = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();

    // Writer threads
    for tid in 0..num_writers {
        let db = Arc::clone(&db);
        let barrier = Arc::clone(&barrier);
        let errors = Arc::clone(&write_errors);
        handles.push(thread::spawn(move || {
            barrier.wait();
            for i in 0..20 {
                let session = db.session();
                let id = 1000 + tid * 100 + i;
                if session
                    .execute(&format!("INSERT (:Written {{id: {id}}})"))
                    .is_err()
                {
                    errors.fetch_add(1, Ordering::Relaxed);
                }
            }
        }));
    }

    // Reader threads
    for _ in 0..num_readers {
        let db = Arc::clone(&db);
        let barrier = Arc::clone(&barrier);
        let errors = Arc::clone(&read_errors);
        handles.push(thread::spawn(move || {
            barrier.wait();
            for _ in 0..20 {
                let session = db.session();
                if session.execute("MATCH (n:Item) RETURN n.id").is_err() {
                    errors.fetch_add(1, Ordering::Relaxed);
                }
            }
        }));
    }

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    assert_eq!(
        read_errors.load(Ordering::Relaxed),
        0,
        "No read errors expected"
    );
    assert_eq!(
        write_errors.load(Ordering::Relaxed),
        0,
        "No write errors expected"
    );
}

#[test]
#[ignore = "stress test: slow in CI, run locally with --ignored"]
fn test_stress_transaction_conflicts() {
    // 4 threads with interleaved commit/rollback patterns
    let db = Arc::new(GrafeoDB::new_in_memory());

    let num_threads = 4;
    let iterations = 6;
    let barrier = Arc::new(Barrier::new(num_threads));
    let completed = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            let completed = Arc::clone(&completed);

            thread::spawn(move || {
                barrier.wait();
                for i in 0..iterations {
                    let mut session = db.session();
                    session.begin_transaction().unwrap();
                    let query = format!("INSERT (:TxNode {{thread: {tid}, iter: {i}}})");
                    let _ = session.execute(&query);

                    // Commit even iterations, rollback odd
                    if i % 2 == 0 {
                        let _ = session.commit();
                    } else {
                        let _ = session.rollback();
                    }
                }
                completed.fetch_add(1, Ordering::Relaxed);
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    assert_eq!(completed.load(Ordering::Relaxed), num_threads);

    // Only committed nodes (even iterations) should exist
    let session = db.session();
    let result = session.execute("MATCH (n:TxNode) RETURN n").unwrap();
    // Each thread commits 5 of 10 iterations (0, 2, 4, 6, 8)
    let expected = num_threads * (iterations / 2);
    assert_eq!(
        result.row_count(),
        expected,
        "Only committed transactions should be visible"
    );
}

#[test]
#[ignore = "stress test: slow in CI, run locally with --ignored"]
fn test_stress_concurrent_epoch_pressure() {
    // 4 threads each running 8 sequential transactions, creates many epochs
    let db = Arc::new(GrafeoDB::new_in_memory());

    let num_threads = 4;
    let txns_per_thread = 8;
    let barrier = Arc::new(Barrier::new(num_threads));
    let completed = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            let completed = Arc::clone(&completed);

            thread::spawn(move || {
                barrier.wait();
                for i in 0..txns_per_thread {
                    let mut session = db.session();
                    session.begin_transaction().unwrap();
                    session
                        .execute(&format!("INSERT (:Epoch {{thread: {tid}, txn: {i}}})"))
                        .unwrap();
                    session.commit().unwrap();
                }
                completed.fetch_add(1, Ordering::Relaxed);
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    assert_eq!(completed.load(Ordering::Relaxed), num_threads);

    // All nodes should be visible
    let session = db.session();
    let result = session.execute("MATCH (n:Epoch) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        num_threads * txns_per_thread,
        "All epoch nodes should exist"
    );
}

#[test]
#[ignore = "stress test"]
fn concurrent_merge_same_node() {
    // 8 threads each run 10 rounds of MERGE with their own key.
    // The contention comes from all threads issuing MERGE on the same
    // label, forcing concurrent read-then-write cycles on the index.
    // Each thread's MERGE should be idempotent: no matter how many
    // rounds, it should produce exactly one node per key.
    let db = Arc::new(GrafeoDB::new_in_memory());

    let num_threads = 8;
    let rounds = 10;
    let barrier = Arc::new(Barrier::new(num_threads));

    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);

            thread::spawn(move || {
                barrier.wait();

                for round in 0..rounds {
                    let session = db.session();
                    let query = format!(
                        "MERGE (n:Shared {{key: 'thread_{tid}'}}) \
                         ON CREATE SET n.thread_id = {tid} \
                         ON MATCH SET n.round = {round}"
                    );
                    session.execute(&query).unwrap();
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    // Each thread should have exactly one node (MERGE is idempotent per key)
    let session = db.session();
    let result = session.execute("MATCH (n:Shared) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        num_threads,
        "Each thread should produce exactly 1 node via MERGE"
    );

    // Verify each thread's node exists
    for tid in 0..num_threads {
        let query = format!("MATCH (n:Shared {{key: 'thread_{tid}'}}) RETURN n");
        let result = session.execute(&query).unwrap();
        assert_eq!(
            result.row_count(),
            1,
            "Thread {tid} should have exactly 1 node"
        );
    }
}

#[test]
#[ignore = "stress test"]
fn concurrent_mixed_read_write_high_contention() {
    // 12 threads: 6 writers inserting unique nodes, 6 readers counting
    let db = Arc::new(GrafeoDB::new_in_memory());

    let num_writers = 6;
    let num_readers = 6;
    let total_threads = num_writers + num_readers;
    let barrier = Arc::new(Barrier::new(total_threads));
    let write_success = Arc::new(AtomicUsize::new(0));
    let read_errors = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();

    // Writer threads: each inserts a uniquely-labeled node
    static NAMES: [&str; 6] = ["Alix", "Gus", "Vincent", "Jules", "Mia", "Butch"];
    for wid in 0..num_writers {
        let db = Arc::clone(&db);
        let barrier = Arc::clone(&barrier);
        let write_success = Arc::clone(&write_success);
        let name = NAMES[wid];

        handles.push(thread::spawn(move || {
            barrier.wait();

            let session = db.session();
            let query = format!("INSERT (:Contention {{name: '{name}', writer: {wid}}})");
            if session.execute(&query).is_ok() {
                write_success.fetch_add(1, Ordering::Relaxed);
            }
        }));
    }

    // Reader threads: each runs a MATCH + count query multiple times
    for _ in 0..num_readers {
        let db = Arc::clone(&db);
        let barrier = Arc::clone(&barrier);
        let read_errors = Arc::clone(&read_errors);

        handles.push(thread::spawn(move || {
            barrier.wait();

            for _ in 0..10 {
                let session = db.session();
                if session
                    .execute("MATCH (n:Contention) RETURN count(n)")
                    .is_err()
                {
                    read_errors.fetch_add(1, Ordering::Relaxed);
                }
            }
        }));
    }

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    assert_eq!(
        read_errors.load(Ordering::Relaxed),
        0,
        "No read errors expected during high contention"
    );

    let writes = write_success.load(Ordering::Relaxed);
    assert_eq!(writes, num_writers, "All writers should succeed");

    // Verify total node count matches the number of successful writes
    let session = db.session();
    let result = session.execute("MATCH (n:Contention) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        writes,
        "Total nodes should equal number of successful writes"
    );
}

#[test]
#[ignore = "stress test"]
fn concurrent_schema_mutation_with_queries() {
    // 4 threads running queries while 2 threads create/drop node types.
    // Verifies no panics or crashes under concurrent schema changes.
    let db = Arc::new(GrafeoDB::new_in_memory());

    let num_query_threads = 4;
    let num_schema_threads = 2;
    let total_threads = num_query_threads + num_schema_threads;
    let barrier = Arc::new(Barrier::new(total_threads));
    let completed = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();

    // Schema mutation threads: create and drop node types in a loop
    for sid in 0..num_schema_threads {
        let db = Arc::clone(&db);
        let barrier = Arc::clone(&barrier);
        let completed = Arc::clone(&completed);

        handles.push(thread::spawn(move || {
            barrier.wait();

            for i in 0..5 {
                let session = db.session();
                let type_name = format!("Temp{sid}_{i}");
                // Create a node type
                let _ = session.execute(&format!("CREATE NODE TYPE {type_name} (val INTEGER)"));
                // Insert a node of that type
                let _ = session.execute(&format!("INSERT (:{type_name} {{val: {i}}})"));
                // Drop the node type
                let _ = session.execute(&format!("DROP NODE TYPE {type_name}"));
            }
            completed.fetch_add(1, Ordering::Relaxed);
        }));
    }

    // Query threads: run read queries concurrently with schema changes
    static QUERY_NAMES: [&str; 4] = ["Django", "Shosanna", "Hans", "Beatrix"];
    for qid in 0..num_query_threads {
        let db = Arc::clone(&db);
        let barrier = Arc::clone(&barrier);
        let completed = Arc::clone(&completed);
        let name = QUERY_NAMES[qid];

        handles.push(thread::spawn(move || {
            barrier.wait();

            let session = db.session();
            // Insert a node unrelated to the schema mutations
            let _ = session.execute(&format!(
                "INSERT (:QueryNode {{name: '{name}', qid: {qid}}})"
            ));
            // Run several read queries
            for _ in 0..10 {
                let _ = session.execute("MATCH (n:QueryNode) RETURN n.name");
            }
            completed.fetch_add(1, Ordering::Relaxed);
        }));
    }

    for handle in handles {
        handle
            .join()
            .expect("Thread panicked during concurrent schema mutation");
    }

    assert_eq!(
        completed.load(Ordering::Relaxed),
        total_threads,
        "All threads should complete without panic"
    );
}

#[test]
#[ignore = "stress test: slow in CI, run locally with --ignored"]
fn test_stress_rapid_session_lifecycle() {
    // 16 threads rapidly creating, using, and dropping sessions
    let db = Arc::new(GrafeoDB::new_in_memory());

    let num_threads = 16;
    let cycles = 100;
    let barrier = Arc::new(Barrier::new(num_threads));
    let completed = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            let completed = Arc::clone(&completed);

            thread::spawn(move || {
                barrier.wait();
                for _ in 0..cycles {
                    let session = db.session();
                    let _ = session.execute("MATCH (n) RETURN n LIMIT 1");
                    drop(session);
                }
                completed.fetch_add(1, Ordering::Relaxed);
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    assert_eq!(
        completed.load(Ordering::Relaxed),
        num_threads,
        "All threads should complete"
    );
}

#[test]
#[ignore = "stress test: slow in CI, run locally with --ignored"]
fn test_stress_concurrent_edges_and_nodes() {
    // Create nodes and edges simultaneously from multiple threads
    let db = Arc::new(GrafeoDB::new_in_memory());

    // Seed some nodes first (needed for edge creation)
    let session = db.session();
    for i in 0..20 {
        session
            .execute(&format!("INSERT (:Hub {{id: {i}}})"))
            .unwrap();
    }
    drop(session);

    let num_threads = 4;
    let barrier = Arc::new(Barrier::new(num_threads));
    let completed = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            let completed = Arc::clone(&completed);

            thread::spawn(move || {
                barrier.wait();
                let session = db.session();
                for i in 0..10 {
                    // Create new nodes
                    session
                        .execute(&format!("INSERT (:Spoke {{thread: {tid}, id: {i}}})"))
                        .unwrap();
                    // Create edges between existing hub nodes
                    let src = (tid * 5 + i) % 20;
                    let dst = (tid * 5 + i + 1) % 20;
                    let _ = session.execute(&format!(
                        "MATCH (a:Hub {{id: {src}}}), (b:Hub {{id: {dst}}}) \
                         INSERT (a)-[:LINK {{thread: {tid}}}]->(b)"
                    ));
                }
                completed.fetch_add(1, Ordering::Relaxed);
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    assert_eq!(completed.load(Ordering::Relaxed), num_threads);

    // Verify spoke nodes were created
    let session = db.session();
    let result = session.execute("MATCH (n:Spoke) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        num_threads * 10,
        "All spoke nodes should exist"
    );
}

// ============================================================================
// Isolation Anomaly Tests (T1-03)
// ============================================================================

/// Documents current dirty-read behavior: uncommitted writes in one session's
/// Dirty read prevention: uncommitted data from one transaction is invisible
/// to other sessions. Versions use PENDING epoch until committed.
#[test]
fn test_dirty_read_prevented() {
    let db = GrafeoDB::new_in_memory();

    let mut writer = db.session();
    writer.begin_transaction().unwrap();
    writer
        .execute("INSERT (:DirtyRead {val: 'uncommitted'})")
        .unwrap();

    // Reader (auto-commit, no explicit transaction) must NOT see uncommitted data.
    // Uncommitted versions use PENDING epoch, invisible to epoch-based reads.
    let reader = db.session();
    let result = reader.execute("MATCH (n:DirtyRead) RETURN n").unwrap();

    assert_eq!(
        result.row_count(),
        0,
        "Dirty read prevented: uncommitted data is invisible to other sessions"
    );

    // After commit, the data becomes visible
    writer.commit().unwrap();

    let result2 = reader.execute("MATCH (n:DirtyRead) RETURN n").unwrap();
    assert_eq!(
        result2.row_count(),
        1,
        "Committed data should be visible to other sessions"
    );
}

/// Documents that after rollback, the rolled-back data is no longer visible.
#[test]
fn test_rollback_hides_data_from_other_sessions() {
    let db = GrafeoDB::new_in_memory();

    let mut writer = db.session();
    writer.begin_transaction().unwrap();
    writer
        .execute("INSERT (:RollbackTest {val: 'temp'})")
        .unwrap();
    writer.rollback().unwrap();

    let reader = db.session();
    let result = reader.execute("MATCH (n:RollbackTest) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        0,
        "Rolled-back data should not be visible"
    );
}

/// Non-repeatable read: reader sees different results for the same query
/// when another session commits between reads.
#[test]
fn test_non_repeatable_read() {
    let db = GrafeoDB::new_in_memory();

    let session1 = db.session();
    session1.execute("INSERT (:NRR {val: 'original'})").unwrap();

    // Reader sees val='original'
    let reader = db.session();
    let r1 = reader.execute("MATCH (n:NRR) RETURN n.val AS val").unwrap();
    assert_eq!(r1.rows().len(), 1);

    // Writer updates
    session1
        .execute("MATCH (n:NRR) SET n.val = 'updated'")
        .unwrap();

    // Reader sees val='updated' (non-repeatable read)
    let r2 = reader.execute("MATCH (n:NRR) RETURN n.val AS val").unwrap();
    assert_eq!(r2.rows().len(), 1);
    // Without snapshot isolation, the reader sees the updated value
    assert_eq!(
        r2.rows()[0][0],
        grafeo_common::types::Value::String("updated".into()),
        "Non-repeatable read: reader sees committed update"
    );
}

/// Phantom read: reader sees new rows that didn't exist during its first read,
/// because another session inserted and committed between reads.
#[test]
fn test_phantom_read() {
    let db = GrafeoDB::new_in_memory();

    let session1 = db.session();
    session1.execute("INSERT (:Phantom {id: 1})").unwrap();

    // Reader sees 1 row
    let reader = db.session();
    let r1 = reader.execute("MATCH (n:Phantom) RETURN n").unwrap();
    assert_eq!(r1.row_count(), 1);

    // Writer inserts another row
    session1.execute("INSERT (:Phantom {id: 2})").unwrap();

    // Reader sees 2 rows (phantom read)
    let r2 = reader.execute("MATCH (n:Phantom) RETURN n").unwrap();
    assert_eq!(
        r2.row_count(),
        2,
        "Phantom read: new rows from other sessions are visible"
    );
}

/// Session drop mid-transaction: verifies that rolled-back data is not visible.
/// (Related to T1-04: Session Drop should auto-rollback.)
#[test]
fn test_drop_session_mid_transaction() {
    let db = GrafeoDB::new_in_memory();

    {
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session
            .execute("INSERT (:DropTest {val: 'should_vanish'})")
            .unwrap();
        // Session drops here without commit or rollback
    }

    let reader = db.session();
    let result = reader.execute("MATCH (n:DropTest) RETURN n").unwrap();
    // Drop impl auto-rollbacks the active transaction, so uncommitted data is discarded.
    assert_eq!(
        result.row_count(),
        0,
        "Drop impl should auto-rollback, discarding uncommitted data"
    );
}

// ============================================================================
// Write-Write Conflict Tests (T1-07)
// ============================================================================

/// Tests first-writer-wins conflict detection: two concurrent sessions
/// modifying the same node property through `Session.execute()`.
///
/// The second session's SET fails immediately (not at commit time) because
/// the first session already recorded a write to the same entity.
#[test]
fn test_write_write_conflict_through_execute() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (:Account {name: 'shared', balance: 100})")
        .unwrap();

    // Session 1: begin tx, read and update
    let mut s1 = db.session();
    s1.begin_transaction().unwrap();
    s1.execute("MATCH (a:Account {name: 'shared'}) SET a.balance = 200")
        .unwrap();

    // Session 2: begin tx, attempt to update the same node
    let mut s2 = db.session();
    s2.begin_transaction().unwrap();
    let set_result = s2.execute("MATCH (a:Account {name: 'shared'}) SET a.balance = 300");
    assert!(
        set_result.is_err(),
        "Second SET should fail with write-write conflict: {set_result:?}"
    );

    // First commit succeeds
    let commit1 = s1.commit();
    assert!(commit1.is_ok(), "First commit should succeed: {commit1:?}");

    // s2 rollback (transaction is still active even though SET failed)
    let _ = s2.rollback();

    // Verify final state: first writer's value persists
    let result = session
        .execute("MATCH (a:Account {name: 'shared'}) RETURN a.balance AS b")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], grafeo_common::types::Value::Int64(200));
}

/// Tests that a rollback in one session doesn't affect another session's committed writes.
#[test]
fn test_concurrent_write_one_rollback() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (:Counter {name: 'hits', val: 0})")
        .unwrap();

    // Session 1: update and commit
    let mut s1 = db.session();
    s1.begin_transaction().unwrap();
    s1.execute("MATCH (c:Counter {name: 'hits'}) SET c.val = 10")
        .unwrap();
    s1.commit().unwrap();

    // Session 2: update and rollback
    let mut s2 = db.session();
    s2.begin_transaction().unwrap();
    s2.execute("MATCH (c:Counter {name: 'hits'}) SET c.val = 999")
        .unwrap();
    s2.rollback().unwrap();

    // Session 1's value (10) should be the final state
    let result = session
        .execute("MATCH (c:Counter {name: 'hits'}) RETURN c.val AS v")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(
        result.rows()[0][0],
        grafeo_common::types::Value::Int64(10),
        "Rolled-back write should not affect committed value"
    );
}

// ============================================================================
// T2-05: Edge creation/deletion rollback
// ============================================================================

/// Create an edge inside a transaction, rollback, verify edge is absent.
#[test]
fn test_edge_create_rollback() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute(
            "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) INSERT (a)-[:KNOWS]->(b)",
        )
        .unwrap();

    // Edge exists inside transaction
    let mid = session.execute("MATCH ()-[r:KNOWS]->() RETURN r").unwrap();
    assert_eq!(mid.row_count(), 1, "Edge should exist inside transaction");

    session.rollback().unwrap();

    // After rollback, edge should be gone
    let reader = db.session();
    let result = reader.execute("MATCH ()-[r:KNOWS]->() RETURN r").unwrap();
    assert_eq!(
        result.row_count(),
        0,
        "Edge should not exist after rollback"
    );
}

/// DELETE edge followed by rollback restores the edge.
///
/// The delete records the edge as it was (type, endpoints, properties) in the
/// transaction's change set, and the rollback restores it from there.
#[test]
fn test_edge_delete_rollback() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
        .unwrap();

    // Verify edge exists
    let before = session.execute("MATCH ()-[r:KNOWS]->() RETURN r").unwrap();
    assert_eq!(before.row_count(), 1);

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (:Person {name: 'Alix'})-[r:KNOWS]->(:Person {name: 'Gus'}) DELETE r")
        .unwrap();
    session.rollback().unwrap();

    let reader = db.session();
    let result = reader.execute("MATCH ()-[r:KNOWS]->() RETURN r").unwrap();
    assert_eq!(
        result.row_count(),
        1,
        "Edge should be restored after rollback"
    );
}

/// DELETE node followed by rollback restores the node with its labels and properties.
#[test]
fn test_node_delete_rollback() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (:Temp {name: 'ephemeral'})")
        .unwrap();

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (t:Temp {name: 'ephemeral'}) DELETE t")
        .unwrap();
    session.rollback().unwrap();

    let reader = db.session();
    let result = reader
        .execute("MATCH (t:Temp) RETURN t.name AS name")
        .unwrap();
    assert_eq!(
        result.row_count(),
        1,
        "Node should be restored after rollback"
    );
    assert_eq!(
        result.rows()[0][0],
        grafeo_common::types::Value::String("ephemeral".into())
    );
}

/// DETACH DELETE followed by rollback restores the node and its edges.
#[test]
fn test_detach_delete_rollback() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
        .unwrap();

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (a:Person {name: 'Alix'}) DETACH DELETE a")
        .unwrap();
    session.rollback().unwrap();

    let reader = db.session();
    let nodes = reader
        .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
        .unwrap();
    assert_eq!(
        nodes.row_count(),
        2,
        "Both nodes should be restored after rollback"
    );
    assert_eq!(
        nodes.rows()[0][0],
        grafeo_common::types::Value::String("Alix".into())
    );
    assert_eq!(
        nodes.rows()[1][0],
        grafeo_common::types::Value::String("Gus".into())
    );

    // Edge should also be restored
    let edges = reader.execute("MATCH ()-[r:KNOWS]->() RETURN r").unwrap();
    assert_eq!(
        edges.row_count(),
        1,
        "Edge should be restored after DETACH DELETE rollback"
    );
}

// ============================================================================
// T2-06: Cross-session commit visibility
// ============================================================================

/// After INSERT+COMMIT in an explicit transaction, a new session sees the data.
#[test]
fn test_cross_session_visibility_after_explicit_commit() {
    let db = GrafeoDB::new_in_memory();

    let mut writer = db.session();
    writer.begin_transaction().unwrap();
    writer
        .execute("INSERT (:Visible {key: 'committed'})")
        .unwrap();
    writer.commit().unwrap();

    // New session should see committed data
    let reader = db.session();
    let result = reader
        .execute("MATCH (v:Visible) RETURN v.key AS key")
        .unwrap();
    assert_eq!(
        result.row_count(),
        1,
        "New session should see committed data"
    );
    assert_eq!(
        result.rows()[0][0],
        grafeo_common::types::Value::String("committed".into())
    );
}

/// Multiple mutations across transactions, verified by a fresh session.
#[test]
fn test_cross_session_visibility_multiple_mutations() {
    let db = GrafeoDB::new_in_memory();

    // First session: insert + set property
    let session1 = db.session();
    session1
        .execute("INSERT (:Item {name: 'widget', price: 10})")
        .unwrap();
    session1
        .execute("MATCH (i:Item {name: 'widget'}) SET i.price = 25")
        .unwrap();

    // Second session sees the updated value
    let session2 = db.session();
    let result = session2
        .execute("MATCH (i:Item {name: 'widget'}) RETURN i.price AS price")
        .unwrap();
    assert_eq!(result.row_count(), 1);
    assert_eq!(
        result.rows()[0][0],
        grafeo_common::types::Value::Int64(25),
        "New session should see the updated price"
    );
}

/// Edge creation visibility: edges inserted in one session visible in another.
#[test]
fn test_cross_session_edge_visibility() {
    let db = GrafeoDB::new_in_memory();

    let session1 = db.session();
    session1
        .execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
        .unwrap();

    let session2 = db.session();
    let result = session2
        .execute("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name AS from, b.name AS to")
        .unwrap();
    assert_eq!(result.row_count(), 1);
    assert_eq!(
        result.rows()[0][0],
        grafeo_common::types::Value::String("Alix".into())
    );
    assert_eq!(
        result.rows()[0][1],
        grafeo_common::types::Value::String("Gus".into())
    );
}
