//! Integration tests for property-filtered vector search.
//!
//! Tests that `vector_search`, `batch_vector_search`, and `mmr_search`
//! correctly restrict results when property equality filters are provided.

#![cfg(feature = "vector-index")]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Helper: create a 3D vector value.
fn vec3(x: f32, y: f32, z: f32) -> Value {
    Value::Vector(vec![x, y, z].into())
}

/// Sets up a database with 6 Doc nodes, each with a vector and a `user_id` property.
fn setup_db() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();

    // user_id=1: nodes near [1, 0, 0]
    let n1 = db.create_node(&["Doc"]).unwrap();
    db.set_node_property(n1, "emb", vec3(1.0, 0.0, 0.0))
        .unwrap();
    db.set_node_property(n1, "user_id", Value::Int64(1))
        .unwrap();

    let n2 = db.create_node(&["Doc"]).unwrap();
    db.set_node_property(n2, "emb", vec3(0.95, 0.05, 0.0))
        .unwrap();
    db.set_node_property(n2, "user_id", Value::Int64(1))
        .unwrap();

    // user_id=2: nodes near [0, 1, 0]
    let n3 = db.create_node(&["Doc"]).unwrap();
    db.set_node_property(n3, "emb", vec3(0.0, 1.0, 0.0))
        .unwrap();
    db.set_node_property(n3, "user_id", Value::Int64(2))
        .unwrap();

    let n4 = db.create_node(&["Doc"]).unwrap();
    db.set_node_property(n4, "emb", vec3(0.05, 0.95, 0.0))
        .unwrap();
    db.set_node_property(n4, "user_id", Value::Int64(2))
        .unwrap();

    // user_id=3: node near [0, 0, 1]
    let n5 = db.create_node(&["Doc"]).unwrap();
    db.set_node_property(n5, "emb", vec3(0.0, 0.0, 1.0))
        .unwrap();
    db.set_node_property(n5, "user_id", Value::Int64(3))
        .unwrap();

    // No user_id property
    let n6 = db.create_node(&["Doc"]).unwrap();
    db.set_node_property(n6, "emb", vec3(0.5, 0.5, 0.0))
        .unwrap();

    // Create property index for fast lookups
    db.create_property_index("user_id");

    db.create_vector_index("Doc", "emb", Some(3), Some("cosine"), None, None, None)
        .expect("create index");

    let _ = (n1, n2, n3, n4, n5, n6);
    db
}

#[test]
fn test_filtered_vector_search_by_user_id() {
    let db = setup_db();

    // Search for vectors near [1, 0, 0] but only among user_id=2 nodes
    let filters: HashMap<String, Value> = [("user_id".to_string(), Value::Int64(2))]
        .into_iter()
        .collect();

    let results = db
        .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 5, None, Some(&filters))
        .expect("filtered search");

    // Should only return user_id=2 nodes (n3, n4)
    assert!(!results.is_empty());
    assert!(results.len() <= 2);

    // Verify all results have user_id=2
    for (id, _) in &results {
        let node = db.get_node(*id).expect("node exists");
        let uid = node
            .properties
            .get(&grafeo_common::types::PropertyKey::new("user_id"))
            .expect("has user_id");
        assert_eq!(uid, &Value::Int64(2), "result should be user_id=2");
    }
}

#[test]
fn test_filtered_search_without_filters_returns_all() {
    let db = setup_db();

    // No filters: should return all matching nodes
    let results = db
        .vector_search("Doc", "emb", &[0.5, 0.5, 0.0], 10, None, None)
        .expect("unfiltered search");

    assert_eq!(results.len(), 6, "should find all 6 Doc nodes");
}

#[test]
fn test_filtered_search_empty_filters_returns_all() {
    let db = setup_db();

    // Empty filter map: should behave like no filters
    let filters: HashMap<String, Value> = HashMap::new();
    let results = db
        .vector_search("Doc", "emb", &[0.5, 0.5, 0.0], 10, None, Some(&filters))
        .expect("empty filter search");

    assert_eq!(results.len(), 6, "empty filters should return all nodes");
}

#[test]
fn test_filtered_search_no_matches() {
    let db = setup_db();

    // user_id=999 doesn't exist
    let filters: HashMap<String, Value> = [("user_id".to_string(), Value::Int64(999))]
        .into_iter()
        .collect();

    let results = db
        .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 5, None, Some(&filters))
        .expect("filtered search");

    assert!(results.is_empty(), "no matching nodes should return empty");
}

#[test]
fn test_batch_vector_search_with_filters() {
    let db = setup_db();

    let filters: HashMap<String, Value> = [("user_id".to_string(), Value::Int64(1))]
        .into_iter()
        .collect();

    let queries = vec![vec![1.0f32, 0.0, 0.0], vec![0.0, 1.0, 0.0]];

    let results = db
        .batch_vector_search("Doc", "emb", &queries, 5, None, Some(&filters))
        .expect("batch filtered search");

    assert_eq!(results.len(), 2);

    // All results in both queries should be user_id=1
    for query_results in &results {
        for (id, _) in query_results {
            let node = db.get_node(*id).expect("node exists");
            let uid = node
                .properties
                .get(&grafeo_common::types::PropertyKey::new("user_id"))
                .expect("has user_id");
            assert_eq!(uid, &Value::Int64(1));
        }
    }
}

#[test]
fn test_mmr_search_with_filters() {
    let db = setup_db();

    let filters: HashMap<String, Value> = [("user_id".to_string(), Value::Int64(2))]
        .into_iter()
        .collect();

    let results = db
        .mmr_search(
            "Doc",
            "emb",
            &[0.0, 1.0, 0.0],
            2,
            None,
            None,
            None,
            Some(&filters),
        )
        .expect("mmr filtered search");

    assert!(!results.is_empty());
    assert!(results.len() <= 2);

    for (id, _) in &results {
        let node = db.get_node(*id).expect("node exists");
        let uid = node
            .properties
            .get(&grafeo_common::types::PropertyKey::new("user_id"))
            .expect("has user_id");
        assert_eq!(uid, &Value::Int64(2));
    }
}

#[test]
fn test_filtered_search_non_indexed_property() {
    let db = setup_db();

    // Filter on a property that is NOT indexed (no property index for "user_id=3"
    // actually it IS indexed, but let's use a different property)
    // Add a "category" property without creating a property index
    // Get the first node and add a category
    let results_all = db
        .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 6, None, None)
        .expect("find all");

    // Set "category" on first 2 nodes only
    for (id, _) in results_all.iter().take(2) {
        db.set_node_property(*id, "category", Value::String("science".into()))
            .unwrap();
    }

    // No property index for "category": should still work (scan fallback)
    let filters: HashMap<String, Value> =
        [("category".to_string(), Value::String("science".into()))]
            .into_iter()
            .collect();

    let results = db
        .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 10, None, Some(&filters))
        .expect("filtered search on non-indexed property");

    assert!(results.len() <= 2, "at most 2 nodes have category=science");
}

/// Creating an index with no dimensions and no existing vectors should error.
#[test]
fn test_create_vector_index_no_dims_no_data_errors() {
    let db = GrafeoDB::new_in_memory();
    let result = db.create_vector_index("Doc", "emb", None, None, None, None, None);
    assert!(result.is_err(), "should error without dimensions or data");
}

/// Creating an index with explicit dimensions but no data should succeed (empty index).
#[test]
fn test_create_vector_index_with_dims_no_data_succeeds() {
    let db = GrafeoDB::new_in_memory();
    db.create_vector_index("Doc", "emb", Some(4), Some("cosine"), None, None, None)
        .expect("should create empty index with explicit dimensions");

    // Insert a node with vector after index creation, auto-insert should work
    let id = db.create_node(&["Doc"]).unwrap();
    db.set_node_property(id, "emb", Value::Vector(vec![1.0, 0.0, 0.0, 0.0].into()))
        .unwrap();

    let results = db
        .vector_search("Doc", "emb", &[1.0, 0.0, 0.0, 0.0], 5, None, None)
        .expect("search should work");
    assert_eq!(results.len(), 1, "should find the one auto-inserted node");
}

/// Mimics the grafeo-memory pattern: create_node_with_props, set vector
/// separately, String-valued filters, no property index.
#[test]
fn test_grafeo_memory_pattern() {
    let db = GrafeoDB::new_in_memory();

    // Create vector index first (grafeo-memory calls _ensure_indexes early)
    db.create_vector_index(
        "Memory",
        "embedding",
        Some(4),
        Some("cosine"),
        None,
        None,
        None,
    )
    .expect("create index");

    // Create nodes with properties at creation time (grafeo-memory pattern)
    for i in 0..10 {
        let user = if i < 5 { "alix" } else { "gus" };
        let id = db
            .create_node_with_props(
                &["Memory"],
                vec![
                    (
                        grafeo_common::types::PropertyKey::new("text"),
                        Value::String(format!("memory {i}").into()),
                    ),
                    (
                        grafeo_common::types::PropertyKey::new("user_id"),
                        Value::String(user.into()),
                    ),
                ],
            )
            .unwrap();

        // Set embedding separately (grafeo-memory calls set_node_property for vectors)
        let emb = vec![(i as f32) / 10.0, 1.0 - (i as f32) / 10.0, 0.1, 0.1];
        db.set_node_property(id, "embedding", Value::Vector(emb.into()))
            .unwrap();
    }

    // Search WITHOUT filters first: should work
    let all_results = db
        .vector_search("Memory", "embedding", &[0.5, 0.5, 0.1, 0.1], 10, None, None)
        .expect("unfiltered search");
    assert_eq!(all_results.len(), 10, "should find all 10 Memory nodes");

    // Search WITH String-valued filter, NO property index (scan fallback)
    let filters: HashMap<String, Value> = [("user_id".to_string(), Value::String("alix".into()))]
        .into_iter()
        .collect();

    let results = db
        .vector_search(
            "Memory",
            "embedding",
            &[0.5, 0.5, 0.1, 0.1],
            10,
            None,
            Some(&filters),
        )
        .expect("filtered search should not error");

    assert_eq!(results.len(), 5, "should find 5 alix nodes");

    // Verify all results have user_id="alix"
    for (id, _) in &results {
        let node = db.get_node(*id).expect("node exists");
        let uid = node
            .properties
            .get(&grafeo_common::types::PropertyKey::new("user_id"))
            .expect("has user_id");
        assert_eq!(uid, &Value::String("alix".into()));
    }
}

// === Advanced filter operator tests ===

/// Helper: setup a database with 10 nodes having numeric `score` and string `category` properties.
fn setup_operator_db() -> GrafeoDB {
    use grafeo_common::types::PropertyKey;
    let db = GrafeoDB::new_in_memory();
    db.create_vector_index("Item", "emb", Some(3), Some("cosine"), None, None, None)
        .expect("create index");

    for i in 0..10 {
        let category = match i % 3 {
            0 => "preference",
            1 => "fact",
            _ => "event",
        };
        let id = db
            .create_node_with_props(
                &["Item"],
                vec![
                    (PropertyKey::new("score"), Value::Float64((i as f64) * 0.1)),
                    (PropertyKey::new("rank"), Value::Int64(i)),
                    (PropertyKey::new("category"), Value::String(category.into())),
                    (
                        PropertyKey::new("text"),
                        Value::String(format!("item number {i} is great").into()),
                    ),
                ],
            )
            .unwrap();
        let emb = vec![(i as f32) / 10.0, 1.0 - (i as f32) / 10.0, 0.5];
        db.set_node_property(id, "emb", Value::Vector(emb.into()))
            .unwrap();
    }
    db
}

/// Helper: build an operator filter as a Value::Map.
fn op_filter(ops: Vec<(&str, Value)>) -> Value {
    let map: std::collections::BTreeMap<grafeo_common::types::PropertyKey, Value> = ops
        .into_iter()
        .map(|(k, v)| (grafeo_common::types::PropertyKey::new(k), v))
        .collect();
    Value::Map(std::sync::Arc::new(map))
}

#[test]
fn test_filter_gt_lt() {
    let db = setup_operator_db();

    // rank > 5 → items 6,7,8,9 → 4 results
    let filters: HashMap<String, Value> = [(
        "rank".to_string(),
        op_filter(vec![("$gt", Value::Int64(5))]),
    )]
    .into_iter()
    .collect();

    let results = db
        .vector_search("Item", "emb", &[0.5, 0.5, 0.5], 10, None, Some(&filters))
        .expect("gt filter");
    assert_eq!(results.len(), 4, "rank > 5 should match 4 nodes");

    // rank < 3 → items 0,1,2 → 3 results
    let filters: HashMap<String, Value> = [(
        "rank".to_string(),
        op_filter(vec![("$lt", Value::Int64(3))]),
    )]
    .into_iter()
    .collect();

    let results = db
        .vector_search("Item", "emb", &[0.5, 0.5, 0.5], 10, None, Some(&filters))
        .expect("lt filter");
    assert_eq!(results.len(), 3, "rank < 3 should match 3 nodes");
}

#[test]
fn test_filter_gte_lte() {
    let db = setup_operator_db();

    // rank >= 3 AND rank <= 6 → items 3,4,5,6 → 4 results
    let filters: HashMap<String, Value> = [(
        "rank".to_string(),
        op_filter(vec![("$gte", Value::Int64(3)), ("$lte", Value::Int64(6))]),
    )]
    .into_iter()
    .collect();

    let results = db
        .vector_search("Item", "emb", &[0.5, 0.5, 0.5], 10, None, Some(&filters))
        .expect("gte/lte filter");
    assert_eq!(results.len(), 4, "rank in [3, 6] should match 4 nodes");
}

#[test]
fn test_filter_in() {
    let db = setup_operator_db();

    // category IN ["preference", "fact"] → items 0,1,3,4,6,7,9 → 7 results
    let filters: HashMap<String, Value> = [(
        "category".to_string(),
        op_filter(vec![(
            "$in",
            Value::List(
                vec![
                    Value::String("preference".into()),
                    Value::String("fact".into()),
                ]
                .into(),
            ),
        )]),
    )]
    .into_iter()
    .collect();

    let results = db
        .vector_search("Item", "emb", &[0.5, 0.5, 0.5], 10, None, Some(&filters))
        .expect("in filter");
    assert_eq!(
        results.len(),
        7,
        "category in [preference, fact] should match 7 nodes"
    );
}

#[test]
fn test_filter_nin() {
    let db = setup_operator_db();

    // category NOT IN ["event"] → items without event → 7 results
    let filters: HashMap<String, Value> = [(
        "category".to_string(),
        op_filter(vec![(
            "$nin",
            Value::List(vec![Value::String("event".into())].into()),
        )]),
    )]
    .into_iter()
    .collect();

    let results = db
        .vector_search("Item", "emb", &[0.5, 0.5, 0.5], 10, None, Some(&filters))
        .expect("nin filter");
    assert_eq!(results.len(), 7, "category not in [event] should match 7");
}

#[test]
fn test_filter_contains() {
    let db = setup_operator_db();

    // text contains "number 5" → only item 5
    let filters: HashMap<String, Value> = [(
        "text".to_string(),
        op_filter(vec![("$contains", Value::String("number 5".into()))]),
    )]
    .into_iter()
    .collect();

    let results = db
        .vector_search("Item", "emb", &[0.5, 0.5, 0.5], 10, None, Some(&filters))
        .expect("contains filter");
    assert_eq!(results.len(), 1, "text contains 'number 5' should match 1");
}

#[test]
fn test_filter_ne() {
    let db = setup_operator_db();

    // category != "event" → 7 results
    let filters: HashMap<String, Value> = [(
        "category".to_string(),
        op_filter(vec![("$ne", Value::String("event".into()))]),
    )]
    .into_iter()
    .collect();

    let results = db
        .vector_search("Item", "emb", &[0.5, 0.5, 0.5], 10, None, Some(&filters))
        .expect("ne filter");
    assert_eq!(results.len(), 7, "category != event should match 7");
}

#[test]
fn test_filter_mixed_equality_and_operators() {
    let db = setup_operator_db();

    // category == "preference" AND rank > 3 → preference items with rank > 3 → items 6, 9
    let filters: HashMap<String, Value> = [
        ("category".to_string(), Value::String("preference".into())),
        (
            "rank".to_string(),
            op_filter(vec![("$gt", Value::Int64(3))]),
        ),
    ]
    .into_iter()
    .collect();

    let results = db
        .vector_search("Item", "emb", &[0.5, 0.5, 0.5], 10, None, Some(&filters))
        .expect("mixed filter");
    assert_eq!(
        results.len(),
        2,
        "preference AND rank > 3 should match items 6 and 9"
    );
}

#[test]
fn test_filter_cross_type_numeric_comparison() {
    let db = setup_operator_db();

    // score (Float64) > 0 (Int64): cross-type comparison
    let filters: HashMap<String, Value> = [(
        "score".to_string(),
        op_filter(vec![("$gt", Value::Int64(0))]),
    )]
    .into_iter()
    .collect();

    let results = db
        .vector_search("Item", "emb", &[0.5, 0.5, 0.5], 10, None, Some(&filters))
        .expect("cross-type filter");
    assert_eq!(
        results.len(),
        9,
        "score > 0 (cross-type) should match 9 nodes (all except score=0.0)"
    );
}

// ---------------------------------------------------------------------------
// Distance ordering verification (T3-06)
// ---------------------------------------------------------------------------

#[test]
fn test_vector_search_results_ordered_by_distance() {
    let db = setup_db();

    let results = db
        .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 6, None, None)
        .expect("unfiltered search");

    assert!(results.len() >= 2, "should return multiple results");

    // Verify results are ordered by increasing distance
    for window in results.windows(2) {
        let (_, dist_a) = window[0];
        let (_, dist_b) = window[1];
        assert!(
            dist_a <= dist_b,
            "results should be ordered by distance: {dist_a} <= {dist_b}"
        );
    }
}

#[test]
fn test_filtered_vector_search_results_ordered_by_distance() {
    let db = setup_db();

    let filters: HashMap<String, Value> = [("user_id".to_string(), Value::Int64(1))]
        .into_iter()
        .collect();

    let results = db
        .vector_search("Doc", "emb", &[0.5, 0.5, 0.0], 5, None, Some(&filters))
        .expect("filtered search");

    assert!(!results.is_empty());
    for window in results.windows(2) {
        let (_, dist_a) = window[0];
        let (_, dist_b) = window[1];
        assert!(
            dist_a <= dist_b,
            "filtered results should be ordered by distance: {dist_a} <= {dist_b}"
        );
    }
}

// ============================================================================
// NULL semantics: $ne and $nin exclude nodes missing the filtered property
// ============================================================================

/// Documents that nodes without the filtered property are excluded from
/// $ne/$nin results (SQL three-valued NULL semantics).
#[test]
fn test_filter_ne_excludes_nodes_missing_property() {
    let db = GrafeoDB::new_in_memory();

    // 3 nodes: two with color, one without
    let n1 = db.create_node(&["Item"]).unwrap();
    db.set_node_property(n1, "emb", vec3(1.0, 0.0, 0.0))
        .unwrap();
    db.set_node_property(n1, "color", Value::String("red".into()))
        .unwrap();
    db.create_vector_index("Item", "emb", Some(3), None, None, None, None)
        .unwrap();

    let n2 = db.create_node(&["Item"]).unwrap();
    db.set_node_property(n2, "emb", vec3(0.0, 1.0, 0.0))
        .unwrap();
    db.set_node_property(n2, "color", Value::String("blue".into()))
        .unwrap();

    let n3 = db.create_node(&["Item"]).unwrap();
    db.set_node_property(n3, "emb", vec3(0.0, 0.0, 1.0))
        .unwrap();
    // n3 has no "color" property

    // $ne: "red" should match n2 (blue) but NOT n3 (no color)
    let filters: HashMap<String, Value> = [(
        "color".to_string(),
        op_filter(vec![("$ne", Value::String("red".into()))]),
    )]
    .into_iter()
    .collect();

    let results = db
        .vector_search("Item", "emb", &[0.5, 0.5, 0.5], 10, None, Some(&filters))
        .expect("ne filter");

    // Only n2 matches: n1 is red, n3 lacks the property entirely
    assert_eq!(
        results.len(),
        1,
        "$ne excludes nodes missing the property (NULL != 'red' is unknown): got {}",
        results.len()
    );
}
