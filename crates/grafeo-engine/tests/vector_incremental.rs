//! Integration tests for incremental vector index operations.
//!
//! Tests that nodes added/deleted after `create_vector_index` are
//! automatically indexed/removed, and that drop/rebuild work correctly.

#![cfg(feature = "vector-index")]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Helper: create a 3D vector value.
fn vec3(x: f32, y: f32, z: f32) -> Value {
    Value::Vector(vec![x, y, z].into())
}

#[test]
fn test_incremental_insert_via_set_property() {
    let db = GrafeoDB::new_in_memory();

    // Create initial nodes and build index
    let n1 = db.create_node(&["Doc"]);
    db.set_node_property(n1, "emb", vec3(1.0, 0.0, 0.0));
    let n2 = db.create_node(&["Doc"]);
    db.set_node_property(n2, "emb", vec3(0.0, 1.0, 0.0));

    db.create_vector_index("Doc", "emb", Some(3), Some("cosine"), None, None, None)
        .expect("create index");

    // Add a new node AFTER index creation
    let n3 = db.create_node(&["Doc"]);
    db.set_node_property(n3, "emb", vec3(0.9, 0.1, 0.0));

    // Search should find the new node (closest to [1, 0, 0])
    let results = db
        .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 3, None, None)
        .expect("search");

    assert_eq!(results.len(), 3, "should find all 3 nodes");
    // n1 and n3 should be closest to query [1, 0, 0]
    let ids: Vec<u64> = results.iter().map(|(id, _)| id.as_u64()).collect();
    assert!(ids.contains(&n3.as_u64()), "n3 should be in results");
}

#[test]
fn test_repeated_vector_update_preserves_hnsw_reachability() {
    const ITERATIONS: usize = 60;
    const NODE_COUNT: usize = 6;

    for iteration in 0..ITERATIONS {
        let db = GrafeoDB::new_in_memory();
        let mut nodes = Vec::with_capacity(NODE_COUNT);

        for offset in 0..NODE_COUNT {
            let node = db.create_node(&["Doc"]);
            db.set_node_property(node, "emb", vec3(0.5 + offset as f32 * 0.01, 0.0, 0.0));
            nodes.push(node);
        }

        db.create_vector_index("Doc", "emb", Some(3), Some("cosine"), None, None, None)
            .expect("create index");

        let target = nodes[NODE_COUNT / 2];
        for _ in 0..5 {
            db.set_node_property(target, "emb", vec3(0.53, 0.0, 0.0));
        }

        let results = db
            .vector_search("Doc", "emb", &[0.5, 0.0, 0.0], NODE_COUNT, None, None)
            .expect("search");

        assert_eq!(
            results.len(),
            NODE_COUNT,
            "same-ID vector replay must not disconnect HNSW nodes (iteration {iteration})"
        );
    }
}

#[test]
fn test_vector_replacement_moves_node_without_losing_neighbors() {
    let db = GrafeoDB::new_in_memory();

    let left = db.create_node(&["Doc"]);
    db.set_node_property(left, "emb", vec3(1.0, 0.0, 0.0));
    let moving = db.create_node(&["Doc"]);
    db.set_node_property(moving, "emb", vec3(0.9, 0.1, 0.0));
    let right = db.create_node(&["Doc"]);
    db.set_node_property(right, "emb", vec3(0.0, 1.0, 0.0));

    db.create_vector_index("Doc", "emb", Some(3), Some("cosine"), None, None, None)
        .expect("create index");

    db.set_node_property(moving, "emb", vec3(0.0, 0.99, 0.01));

    let results = db
        .vector_search("Doc", "emb", &[0.0, 1.0, 0.0], 3, None, None)
        .expect("search");
    let ids: Vec<u64> = results.iter().map(|(id, _)| id.as_u64()).collect();

    assert_eq!(
        results.len(),
        3,
        "replacement must keep all nodes reachable"
    );
    assert_eq!(
        ids.first(),
        Some(&right.as_u64()),
        "the unchanged nearest node should remain first"
    );
    assert!(
        ids.contains(&moving.as_u64()),
        "the replaced vector must remain indexed"
    );
    assert!(
        ids.contains(&left.as_u64()),
        "unrelated nodes must remain indexed"
    );
}

#[test]
fn test_incremental_batch_create_after_index() {
    let db = GrafeoDB::new_in_memory();

    // Create initial node and build index
    let n1 = db.create_node(&["Doc"]);
    db.set_node_property(n1, "emb", vec3(1.0, 0.0, 0.0));

    db.create_vector_index("Doc", "emb", Some(3), Some("euclidean"), None, None, None)
        .expect("create index");

    // Batch-create nodes AFTER index
    let new_ids =
        db.batch_create_nodes("Doc", "emb", vec![vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]]);

    assert_eq!(new_ids.len(), 2);

    // Search should find all 3 nodes
    let results = db
        .vector_search("Doc", "emb", &[0.5, 0.5, 0.5], 10, None, None)
        .expect("search");

    assert_eq!(results.len(), 3, "should find original + batch nodes");
}

#[test]
fn test_delete_removes_from_index() {
    let db = GrafeoDB::new_in_memory();

    let n1 = db.create_node(&["Doc"]);
    db.set_node_property(n1, "emb", vec3(1.0, 0.0, 0.0));
    let n2 = db.create_node(&["Doc"]);
    db.set_node_property(n2, "emb", vec3(0.0, 1.0, 0.0));
    let n3 = db.create_node(&["Doc"]);
    db.set_node_property(n3, "emb", vec3(0.0, 0.0, 1.0));

    db.create_vector_index("Doc", "emb", Some(3), Some("euclidean"), None, None, None)
        .expect("create index");

    // Delete n2
    assert!(db.delete_node(n2));

    // Search should NOT return n2
    let results = db
        .vector_search("Doc", "emb", &[0.0, 1.0, 0.0], 10, None, None)
        .expect("search");

    let ids: Vec<u64> = results.iter().map(|(id, _)| id.as_u64()).collect();
    assert!(
        !ids.contains(&n2.as_u64()),
        "deleted node should not appear"
    );
    assert_eq!(results.len(), 2, "should find only 2 remaining nodes");
}

#[test]
fn test_label_after_vector_triggers_index() {
    let db = GrafeoDB::new_in_memory();

    // Build index on "Doc:emb" with an initial node
    let n1 = db.create_node(&["Doc"]);
    db.set_node_property(n1, "emb", vec3(1.0, 0.0, 0.0));

    db.create_vector_index("Doc", "emb", Some(3), Some("cosine"), None, None, None)
        .expect("create index");

    // Create a node WITHOUT the "Doc" label, set vector, THEN add label
    let n2 = db.create_node(&["Other"]);
    db.set_node_property(n2, "emb", vec3(0.0, 1.0, 0.0));
    // At this point n2 has label "Other", not "Doc": no index match
    db.add_node_label(n2, "Doc");
    // Now n2 has "Doc" label, should trigger auto-insert

    let results = db
        .vector_search("Doc", "emb", &[0.0, 1.0, 0.0], 10, None, None)
        .expect("search");

    let ids: Vec<u64> = results.iter().map(|(id, _)| id.as_u64()).collect();
    assert!(
        ids.contains(&n2.as_u64()),
        "label-after-vector node should be found"
    );
}

#[test]
fn test_drop_vector_index() {
    let db = GrafeoDB::new_in_memory();

    let n1 = db.create_node(&["Doc"]);
    db.set_node_property(n1, "emb", vec3(1.0, 0.0, 0.0));

    db.create_vector_index("Doc", "emb", Some(3), Some("cosine"), None, None, None)
        .expect("create index");

    // Search works
    assert!(
        db.vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 1, None, None)
            .is_ok()
    );

    // Drop
    assert!(db.drop_vector_index("Doc", "emb"));
    assert!(!db.drop_vector_index("Doc", "emb")); // second drop returns false

    // Search now fails
    assert!(
        db.vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 1, None, None)
            .is_err()
    );
}

#[test]
fn test_rebuild_vector_index() {
    let db = GrafeoDB::new_in_memory();

    let n1 = db.create_node(&["Doc"]);
    db.set_node_property(n1, "emb", vec3(1.0, 0.0, 0.0));

    db.create_vector_index("Doc", "emb", Some(3), Some("cosine"), None, None, None)
        .expect("create index");

    // Add more nodes
    let n2 = db.create_node(&["Doc"]);
    db.set_node_property(n2, "emb", vec3(0.0, 1.0, 0.0));
    let n3 = db.create_node(&["Doc"]);
    db.set_node_property(n3, "emb", vec3(0.0, 0.0, 1.0));

    // Rebuild rescans all nodes
    db.rebuild_vector_index("Doc", "emb").expect("rebuild");

    let results = db
        .vector_search("Doc", "emb", &[0.5, 0.5, 0.5], 10, None, None)
        .expect("search");

    assert_eq!(results.len(), 3, "rebuild should include all nodes");
}

#[test]
fn test_rebuild_nonexistent_index_fails() {
    let db = GrafeoDB::new_in_memory();
    assert!(db.rebuild_vector_index("Doc", "emb").is_err());
}

#[test]
fn test_set_vector_without_index_is_noop() {
    let db = GrafeoDB::new_in_memory();

    // No vector index exists: setting a vector property should not crash
    let n1 = db.create_node(&["Doc"]);
    db.set_node_property(n1, "emb", vec3(1.0, 0.0, 0.0));

    // Node should exist with the property
    let node = db.get_node(n1);
    assert!(node.is_some());
}
