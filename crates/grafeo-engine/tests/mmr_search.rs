//! Integration tests for MMR (Maximal Marginal Relevance) search.
//!
//! Tests the `mmr_search()` database method with various parameters.

#![cfg(feature = "vector-index")]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

fn vec3(x: f32, y: f32, z: f32) -> Value {
    Value::Vector(vec![x, y, z].into())
}

fn setup_db() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();

    // Create 5 nodes with varied vectors
    let n1 = db.create_node(&["Doc"]).unwrap();
    db.set_node_property(n1, "emb", vec3(1.0, 0.0, 0.0))
        .unwrap();

    let n2 = db.create_node(&["Doc"]).unwrap();
    db.set_node_property(n2, "emb", vec3(0.95, 0.05, 0.0))
        .unwrap(); // very similar to n1

    let n3 = db.create_node(&["Doc"]).unwrap();
    db.set_node_property(n3, "emb", vec3(0.0, 1.0, 0.0))
        .unwrap(); // orthogonal

    let n4 = db.create_node(&["Doc"]).unwrap();
    db.set_node_property(n4, "emb", vec3(0.0, 0.0, 1.0))
        .unwrap(); // orthogonal

    let n5 = db.create_node(&["Doc"]).unwrap();
    db.set_node_property(n5, "emb", vec3(0.9, 0.1, 0.0))
        .unwrap(); // similar to n1

    db.create_vector_index("Doc", "emb", Some(3), Some("cosine"), None, None, None)
        .expect("create index");

    db
}

#[test]
fn test_mmr_basic_search() {
    let db = setup_db();

    let results = db
        .mmr_search("Doc", "emb", &[1.0, 0.0, 0.0], 3, None, None, None, None)
        .expect("mmr search");

    assert_eq!(results.len(), 3, "should return k=3 results");
}

#[test]
fn test_mmr_lambda_1_matches_knn() {
    let db = setup_db();

    // lambda=1.0 means pure relevance (same as kNN)
    let mmr_results = db
        .mmr_search(
            "Doc",
            "emb",
            &[1.0, 0.0, 0.0],
            3,
            Some(20),
            Some(1.0),
            None,
            None,
        )
        .expect("mmr search");

    let knn_results = db
        .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 3, None, None)
        .expect("knn search");

    // Same top-k results (same IDs, though order may differ slightly)
    let mmr_ids: Vec<u64> = mmr_results.iter().map(|(id, _)| id.as_u64()).collect();
    let knn_ids: Vec<u64> = knn_results.iter().map(|(id, _)| id.as_u64()).collect();

    // All kNN results should appear in MMR results with lambda=1
    for id in &knn_ids {
        assert!(
            mmr_ids.contains(id),
            "lambda=1.0 MMR should match kNN results"
        );
    }
}

#[test]
fn test_mmr_lambda_0_maximizes_diversity() {
    let db = setup_db();

    // lambda=0.0 means pure diversity
    let results = db
        .mmr_search(
            "Doc",
            "emb",
            &[1.0, 0.0, 0.0],
            3,
            Some(20),
            Some(0.0),
            None,
            None,
        )
        .expect("mmr search");

    assert_eq!(results.len(), 3);

    // With pure diversity, after picking the first (most relevant),
    // the next picks should maximize distance from already selected.
    // So we should get diverse vectors, not the 3 clustered near [1,0,0].
    let ids: Vec<u64> = results.iter().map(|(id, _)| id.as_u64()).collect();

    // At least one of the orthogonal vectors should appear
    // (n3=[0,1,0] or n4=[0,0,1] which have IDs 2 or 3)
    let has_diverse = ids.iter().any(|&id| id >= 2);
    assert!(has_diverse, "diversity mode should pick orthogonal vectors");
}

#[test]
fn test_mmr_custom_fetch_k() {
    let db = setup_db();

    // Small fetch_k limits candidate pool
    let results = db
        .mmr_search(
            "Doc",
            "emb",
            &[1.0, 0.0, 0.0],
            2,
            Some(3), // only fetch 3 candidates
            Some(0.5),
            None,
            None,
        )
        .expect("mmr search");

    assert_eq!(results.len(), 2);
}

#[test]
fn test_mmr_k_greater_than_nodes() {
    let db = setup_db();

    // Request more than available
    let results = db
        .mmr_search("Doc", "emb", &[1.0, 0.0, 0.0], 100, None, None, None, None)
        .expect("mmr search");

    assert_eq!(results.len(), 5, "should return all nodes when k > count");
}

#[test]
fn test_mmr_search_nonexistent_index_fails() {
    let db = GrafeoDB::new_in_memory();
    let result = db.mmr_search("Nope", "emb", &[1.0, 0.0, 0.0], 3, None, None, None, None);
    assert!(result.is_err());
}

#[test]
fn test_mmr_returns_distances() {
    let db = setup_db();

    let results = db
        .mmr_search(
            "Doc",
            "emb",
            &[1.0, 0.0, 0.0],
            3,
            None,
            Some(1.0),
            None,
            None,
        )
        .expect("mmr search");

    // All distances should be non-negative
    for (_id, dist) in &results {
        assert!(*dist >= 0.0, "distances should be non-negative");
    }
}

#[test]
fn test_mmr_distances_match_vector_search() {
    let db = setup_db();
    let query = &[1.0_f32, 0.0, 0.0];

    // Get distances from both methods with lambda=1.0 (pure relevance)
    let mmr_results = db
        .mmr_search("Doc", "emb", query, 5, Some(20), Some(1.0), None, None)
        .expect("mmr search");
    let knn_results = db
        .vector_search("Doc", "emb", query, 5, None, None)
        .expect("vector search");

    // For each node that appears in both result sets, distances should match
    for (mmr_id, mmr_dist) in &mmr_results {
        if let Some((_, knn_dist)) = knn_results.iter().find(|(id, _)| id == mmr_id) {
            assert!(
                (mmr_dist - knn_dist).abs() < 1e-5,
                "mmr_search distance for node {} should match vector_search: mmr={}, knn={}",
                mmr_id.as_u64(),
                mmr_dist,
                knn_dist,
            );
        }
    }
}
