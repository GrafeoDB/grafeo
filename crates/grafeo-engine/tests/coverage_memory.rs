//! Tests for memory introspection coverage.
//!
//! Targets: memory.rs (0%), admin.rs memory_usage path
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test coverage_memory
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

#[test]
fn test_memory_usage_empty_database() {
    let db = GrafeoDB::new_in_memory();
    let usage = db.memory_usage();
    // Empty database still has structural overhead
    assert!(
        usage.total_bytes > 0,
        "total should include structural overhead"
    );
    // No nodes or edges, but store has allocated maps
    assert_eq!(usage.mvcc.max_chain_depth, 0);
}

#[test]
fn test_memory_usage_with_data() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    for i in 0..20 {
        let n = session
            .create_node_with_props(
                &["Person"],
                [
                    ("name", Value::String(format!("Person{i}").into())),
                    ("age", Value::Int64(20 + i)),
                ],
            )
            .unwrap();
        if i > 0 {
            let prev = session
                .create_node_with_props(&["Marker"], [("idx", Value::Int64(i))])
                .unwrap();
            session.create_edge(prev, n, "LINKS_TO").unwrap();
        }
    }

    let usage = db.memory_usage();

    // Store should report non-zero for nodes, edges, properties
    assert!(usage.store.nodes_bytes > 0, "nodes_bytes should be > 0");
    assert!(usage.store.edges_bytes > 0, "edges_bytes should be > 0");
    assert!(usage.store.node_properties_bytes > 0);
    // Edges have no explicit properties, so edge_properties_bytes should be 0
    assert_eq!(
        usage.store.edge_properties_bytes, 0,
        "no edge properties were set"
    );
    assert!(usage.store.property_column_count > 0);

    // Indexes should report non-zero (adjacency, labels)
    assert!(usage.indexes.forward_adjacency_bytes > 0);
    assert!(usage.indexes.label_index_bytes > 0);
    assert!(usage.indexes.node_labels_bytes > 0);

    // MVCC should report at least depth 1 for each chain
    assert!(usage.mvcc.total_bytes > 0);
    assert!(usage.mvcc.max_chain_depth >= 1);
    assert!(usage.mvcc.average_chain_depth > 0.0);

    // String pool should contain "Person", "Marker", "LINKS_TO"
    assert!(usage.string_pool.label_count >= 2);
    assert!(usage.string_pool.edge_type_count >= 1);
    assert!(usage.string_pool.total_bytes > 0);
}

#[test]
fn test_memory_usage_after_mutations() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let _n = session
        .create_node_with_props(&["Account"], [("balance", Value::Int64(100))])
        .unwrap();
    let before = db.memory_usage();

    // Mutate to create version chains with depth > 1
    let mut txn = db.session();
    txn.begin_transaction().unwrap();
    txn.execute("MATCH (a:Account) SET a.balance = 200")
        .unwrap();
    txn.commit().unwrap();

    let mut txn2 = db.session();
    txn2.begin_transaction().unwrap();
    txn2.execute("MATCH (a:Account) SET a.balance = 300")
        .unwrap();
    txn2.commit().unwrap();

    let after = db.memory_usage();
    // After mutations, total bytes should be at least the same (in-place updates may not grow)
    assert!(
        after.total_bytes >= before.total_bytes,
        "total bytes should not shrink after mutations: before={}, after={}",
        before.total_bytes,
        after.total_bytes
    );
    // The store should still report property bytes (properties still exist)
    assert!(
        after.store.node_properties_bytes > 0,
        "node properties should still be tracked after mutations"
    );
}

#[test]
fn test_memory_usage_caches() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .create_node_with_props(&["Tag"], [("name", Value::String("test".into()))])
        .unwrap();

    // Execute a query to populate query cache
    session.execute("MATCH (t:Tag) RETURN t.name").unwrap();
    session.execute("MATCH (t:Tag) RETURN t.name").unwrap(); // cache hit

    let usage = db.memory_usage();
    assert!(usage.caches.cached_plan_count >= 1);
}

#[test]
fn test_detailed_stats_memory_matches_memory_usage_total() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    for i in 0..10 {
        session
            .create_node_with_props(&["Item"], [("v", Value::Int64(i))])
            .unwrap();
    }

    let stats = db.detailed_stats();
    let usage = db.memory_usage();
    assert_eq!(
        stats.memory_bytes, usage.total_bytes,
        "detailed_stats.memory_bytes must equal memory_usage().total_bytes"
    );
    assert!(
        stats.memory_bytes > 0,
        "populated database should report non-zero memory_bytes"
    );
}

#[test]
fn test_memory_usage_thread_safe() {
    use std::sync::Arc;
    use std::thread;

    let db = Arc::new(GrafeoDB::new_in_memory());
    let session = db.session();
    for i in 0..50 {
        session
            .create_node_with_props(&["N"], [("v", Value::Int64(i))])
            .unwrap();
    }

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let db = Arc::clone(&db);
            thread::spawn(move || {
                for _ in 0..20 {
                    let u = db.memory_usage();
                    assert!(u.total_bytes > 0);
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
}
