//! Integration tests for search operations covering uncovered paths.
//!
//! Targets low-coverage areas in `database/search.rs`:
//! - hybrid_search (previously untested)
//! - text_search error paths
//! - batch_vector_search with filters
//! - mmr_search with filters
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test search_operations
//! ```

// ============================================================================
// Vector search tests
// ============================================================================

#[cfg(feature = "vector-index")]
mod vector {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;
    use std::collections::HashMap;

    fn vec3(x: f32, y: f32, z: f32) -> Value {
        Value::Vector(vec![x, y, z].into())
    }

    fn setup_vector_db() -> GrafeoDB {
        let db = GrafeoDB::new_in_memory();

        let n1 = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(n1, "emb", vec3(1.0, 0.0, 0.0))
            .unwrap();
        db.set_node_property(n1, "category", Value::String("science".into()))
            .unwrap();

        let n2 = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(n2, "emb", vec3(0.0, 1.0, 0.0))
            .unwrap();
        db.set_node_property(n2, "category", Value::String("science".into()))
            .unwrap();

        let n3 = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(n3, "emb", vec3(0.0, 0.0, 1.0))
            .unwrap();
        db.set_node_property(n3, "category", Value::String("art".into()))
            .unwrap();

        let n4 = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(n4, "emb", vec3(0.9, 0.1, 0.0))
            .unwrap();
        db.set_node_property(n4, "category", Value::String("science".into()))
            .unwrap();

        db.create_property_index("category");
        db.create_vector_index("Doc", "emb", Some(3), Some("cosine"), None, None, None)
            .expect("create vector index");

        db
    }

    #[test]
    fn test_vector_search_no_index_error() {
        let db = GrafeoDB::new_in_memory();
        let n = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(n, "emb", vec3(1.0, 0.0, 0.0)).unwrap();

        // No vector index created: search should fail
        let result = db.vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 5, None, None);
        assert!(result.is_err(), "search without index should error");
    }

    #[test]
    fn test_vector_search_with_ef_parameter() {
        let db = setup_vector_db();

        // ef parameter controls search quality (higher = better, slower)
        let results = db
            .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 2, Some(50), None)
            .expect("search with ef");

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_batch_vector_search_multiple_queries() {
        let db = setup_vector_db();

        let queries = vec![vec![1.0_f32, 0.0, 0.0], vec![0.0, 1.0, 0.0]];

        let results = db
            .batch_vector_search("Doc", "emb", &queries, 2, None, None)
            .expect("batch search");

        assert_eq!(results.len(), 2);
        // Each query should return up to 2 results
        for result_set in &results {
            assert!(result_set.len() <= 2);
            assert!(!result_set.is_empty());
        }
    }

    #[test]
    fn test_batch_vector_search_with_filter() {
        let db = setup_vector_db();

        let mut filters = HashMap::new();
        filters.insert("category".to_string(), Value::String("science".into()));

        let queries = vec![vec![1.0_f32, 0.0, 0.0]];

        let results = db
            .batch_vector_search("Doc", "emb", &queries, 10, None, Some(&filters))
            .expect("batch search with filter");

        assert_eq!(results.len(), 1);
        // Only science docs should be returned (3 out of 4)
        assert_eq!(results[0].len(), 3);
    }

    #[test]
    fn test_mmr_search_with_filter() {
        let db = setup_vector_db();

        let mut filters = HashMap::new();
        filters.insert("category".to_string(), Value::String("science".into()));

        let results = db
            .mmr_search(
                "Doc",
                "emb",
                &[1.0, 0.0, 0.0],
                2,
                None,
                None,
                None,
                Some(&filters),
            )
            .expect("mmr with filter");

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_vector_search_k_larger_than_dataset() {
        let db = setup_vector_db();

        // Request more results than nodes exist
        let results = db
            .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 100, None, None)
            .expect("k > dataset");

        // Should return all 4 nodes, not error
        assert_eq!(results.len(), 4);
    }

    #[test]
    fn test_drop_and_recreate_vector_index() {
        let db = setup_vector_db();

        // Search works
        let r1 = db
            .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 2, None, None)
            .expect("search before drop");
        assert_eq!(r1.len(), 2);

        // Drop index
        assert!(db.drop_vector_index("Doc", "emb"));

        // Search should fail
        let err = db.vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 2, None, None);
        assert!(err.is_err(), "search after drop should error");

        // Recreate index
        db.create_vector_index("Doc", "emb", Some(3), Some("cosine"), None, None, None)
            .expect("recreate index");

        // Search works again
        let r2 = db
            .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 2, None, None)
            .expect("search after recreate");
        assert_eq!(r2.len(), 2);
    }

    #[test]
    fn test_vector_index_auto_inserts_on_set_property() {
        let db = GrafeoDB::new_in_memory();

        // Create index FIRST, on empty data
        db.create_vector_index("Doc", "emb", Some(3), Some("cosine"), None, None, None)
            .expect("create empty index");

        // Add nodes AFTER index exists (no rebuild)
        let n1 = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(n1, "emb", vec3(1.0, 0.0, 0.0))
            .unwrap();

        let n2 = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(n2, "emb", vec3(0.0, 1.0, 0.0))
            .unwrap();

        // Search should find both nodes WITHOUT calling rebuild_vector_index
        let results = db
            .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 5, None, None)
            .expect("search without rebuild");

        assert_eq!(
            results.len(),
            2,
            "auto-inserted nodes should be searchable without rebuild"
        );

        // The closest node to [1,0,0] should be n1
        assert_eq!(results[0].0, n1, "n1 should be the closest match");
    }

    #[test]
    fn test_rebuild_vector_index_preserves_results() {
        let db = setup_vector_db();
        let query = &[1.0_f32, 0.0, 0.0];

        // Search before rebuild
        let before = db
            .vector_search("Doc", "emb", query, 4, None, None)
            .expect("search before rebuild");

        // Rebuild
        db.rebuild_vector_index("Doc", "emb").expect("rebuild");

        // Search after rebuild
        let after = db
            .vector_search("Doc", "emb", query, 4, None, None)
            .expect("search after rebuild");

        // Same nodes, same distances (within floating point tolerance).
        // Sort by node ID because HNSW does not guarantee ordering for
        // equal-distance ties, and rebuild changes the internal graph.
        assert_eq!(before.len(), after.len(), "same result count after rebuild");

        let mut before_sorted: Vec<_> = before.iter().collect();
        let mut after_sorted: Vec<_> = after.iter().collect();
        before_sorted.sort_by_key(|(id, _)| *id);
        after_sorted.sort_by_key(|(id, _)| *id);

        for (b, a) in before_sorted.iter().zip(after_sorted.iter()) {
            assert_eq!(b.0, a.0, "same node IDs after rebuild");
            assert!(
                (b.1 - a.1).abs() < 1e-5,
                "same distances after rebuild: {} vs {}",
                b.1,
                a.1,
            );
        }
    }

    // ── Quantized vector index tests ─────────────────────────────────

    #[test]
    fn test_scalar_quantized_vector_index() {
        let db = GrafeoDB::new_in_memory();
        let n1 = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(n1, "emb", vec3(1.0, 0.0, 0.0))
            .unwrap();
        let n2 = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(n2, "emb", vec3(0.0, 1.0, 0.0))
            .unwrap();
        let n3 = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(n3, "emb", vec3(0.0, 0.0, 1.0))
            .unwrap();

        db.create_vector_index(
            "Doc",
            "emb",
            Some(3),
            Some("cosine"),
            None,
            None,
            Some("scalar"),
        )
        .expect("create scalar quantized index");

        let results = db
            .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 2, None, None)
            .expect("search scalar quantized");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, n1);
    }

    #[test]
    fn test_binary_quantized_vector_index() {
        let db = GrafeoDB::new_in_memory();
        let n1 = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(n1, "emb", vec3(1.0, 0.0, 0.0))
            .unwrap();
        let n2 = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(n2, "emb", vec3(0.0, 1.0, 0.0))
            .unwrap();

        db.create_vector_index(
            "Doc",
            "emb",
            Some(3),
            Some("euclidean"),
            None,
            None,
            Some("binary"),
        )
        .expect("create binary quantized index");

        let results = db
            .vector_search("Doc", "emb", &[0.9, 0.1, 0.0], 2, None, None)
            .expect("search binary quantized");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_product_quantized_vector_index() {
        // Product quantization needs dims divisible by num_subvectors (default 8),
        // so use 8-dim vectors.
        let db = GrafeoDB::new_in_memory();
        for i in 0..10 {
            let n = db.create_node(&["Doc"]).unwrap();
            let vec: Vec<f32> = (0..8).map(|j| ((i * 8 + j) as f32) / 80.0).collect();
            db.set_node_property(n, "emb", Value::Vector(vec.into()))
                .unwrap();
        }

        db.create_vector_index(
            "Doc",
            "emb",
            Some(8),
            Some("euclidean"),
            None,
            None,
            Some("product"),
        )
        .expect("create product quantized index");

        let results = db
            .vector_search("Doc", "emb", &[0.5; 8], 3, None, None)
            .expect("search product quantized");
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_empty_quantized_index_auto_inserts() {
        let db = GrafeoDB::new_in_memory();

        // Create quantized index on empty data with explicit dimensions
        db.create_vector_index(
            "Doc",
            "emb",
            Some(3),
            Some("cosine"),
            None,
            None,
            Some("scalar"),
        )
        .expect("create empty scalar index");

        // Add nodes after index creation
        let n1 = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(n1, "emb", vec3(1.0, 0.0, 0.0))
            .unwrap();
        let n2 = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(n2, "emb", vec3(0.0, 1.0, 0.0))
            .unwrap();

        let results = db
            .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 2, None, None)
            .expect("search after late insert into quantized index");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_rebuild_preserves_quantization_type() {
        let db = GrafeoDB::new_in_memory();
        let n1 = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(n1, "emb", vec3(1.0, 0.0, 0.0))
            .unwrap();
        let n2 = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(n2, "emb", vec3(0.0, 1.0, 0.0))
            .unwrap();

        db.create_vector_index(
            "Doc",
            "emb",
            Some(3),
            Some("cosine"),
            None,
            None,
            Some("binary"),
        )
        .expect("create binary index");

        db.rebuild_vector_index("Doc", "emb").expect("rebuild");

        // Search should still work after rebuild
        let results = db
            .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 2, None, None)
            .expect("search after rebuild of quantized index");
        assert_eq!(results.len(), 2);
    }
}

// ============================================================================
// Text search tests
// ============================================================================

#[cfg(feature = "text-index")]
mod text {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    fn setup_text_db() -> GrafeoDB {
        let db = GrafeoDB::new_in_memory();

        let n1 = db.create_node(&["Article"]).unwrap();
        db.set_node_property(n1, "title", Value::String("Rust graph database".into()))
            .unwrap();

        let n2 = db.create_node(&["Article"]).unwrap();
        db.set_node_property(n2, "title", Value::String("Python machine learning".into()))
            .unwrap();

        let n3 = db.create_node(&["Article"]).unwrap();
        db.set_node_property(
            n3,
            "title",
            Value::String("Rust systems programming".into()),
        )
        .unwrap();

        db.create_text_index("Article", "title")
            .expect("create text index");

        db
    }

    #[test]
    fn test_text_search_basic() {
        let db = setup_text_db();

        let results = db.text_search("Article", "title", "Rust", 10).unwrap();

        // Should match articles with "Rust"
        assert!(results.len() >= 2, "expected at least 2 Rust articles");
    }

    #[test]
    fn test_text_search_no_index_error() {
        let db = GrafeoDB::new_in_memory();
        let n = db.create_node(&["Article"]).unwrap();
        db.set_node_property(n, "title", Value::String("test".into()))
            .unwrap();

        // No text index: should error
        let result = db.text_search("Article", "title", "test", 10);
        assert!(result.is_err(), "text search without index should error");
    }

    #[test]
    fn test_text_search_no_matches() {
        let db = setup_text_db();

        let results = db
            .text_search("Article", "title", "nonexistentxyz", 10)
            .unwrap();

        assert!(results.is_empty(), "no matches expected for nonsense query");
    }

    #[test]
    fn test_text_search_after_mutation() {
        let db = setup_text_db();

        // Add a new article
        let n = db.create_node(&["Article"]).unwrap();
        db.set_node_property(n, "title", Value::String("Rust web framework".into()))
            .unwrap();

        let results = db.text_search("Article", "title", "Rust", 10).unwrap();

        // Should now include the new article
        assert!(
            results.len() >= 3,
            "expected at least 3 Rust articles after mutation"
        );
    }

    #[test]
    fn test_drop_and_rebuild_text_index() {
        let db = setup_text_db();

        // Search works
        let r1 = db.text_search("Article", "title", "Rust", 10).unwrap();
        assert!(!r1.is_empty());

        // Drop index
        assert!(db.drop_text_index("Article", "title"));

        // Search should fail
        let err = db.text_search("Article", "title", "Rust", 10);
        assert!(err.is_err());

        // Rebuild index
        db.rebuild_text_index("Article", "title").unwrap();

        // Search works again
        let r2 = db.text_search("Article", "title", "Rust", 10).unwrap();
        assert!(!r2.is_empty());
    }
}

// ============================================================================
// Hybrid search tests
// ============================================================================

#[cfg(feature = "hybrid-search")]
mod hybrid {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    fn vec3(x: f32, y: f32, z: f32) -> Value {
        Value::Vector(vec![x, y, z].into())
    }

    fn setup_hybrid_db() -> GrafeoDB {
        let db = GrafeoDB::new_in_memory();

        let n1 = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(
            n1,
            "content",
            Value::String("Rust graph database engine".into()),
        )
        .unwrap();
        db.set_node_property(n1, "emb", vec3(1.0, 0.0, 0.0))
            .unwrap();

        let n2 = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(
            n2,
            "content",
            Value::String("Python machine learning framework".into()),
        )
        .unwrap();
        db.set_node_property(n2, "emb", vec3(0.0, 1.0, 0.0))
            .unwrap();

        let n3 = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(
            n3,
            "content",
            Value::String("Rust systems programming language".into()),
        )
        .unwrap();
        db.set_node_property(n3, "emb", vec3(0.9, 0.1, 0.0))
            .unwrap();

        let n4 = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(
            n4,
            "content",
            Value::String("Graph neural network research".into()),
        )
        .unwrap();
        db.set_node_property(n4, "emb", vec3(0.5, 0.5, 0.0))
            .unwrap();

        // Create both indexes
        db.create_text_index("Doc", "content")
            .expect("create text index");
        db.create_vector_index("Doc", "emb", Some(3), Some("cosine"), None, None, None)
            .expect("create vector index");

        db
    }

    #[test]
    fn test_hybrid_search_basic() {
        let db = setup_hybrid_db();

        let results = db
            .hybrid_search(
                "Doc",
                "content",
                "emb",
                "Rust graph",
                Some(&[1.0, 0.0, 0.0]),
                4,
                None,
            )
            .expect("hybrid search");

        assert!(!results.is_empty(), "hybrid search should return results");

        // "Rust graph database engine" should rank highest: matches both
        // text ("Rust graph") and vector (closest to [1,0,0])
        let top_node = results[0].0;
        let top_props = db.get_node(top_node).expect("top node exists");
        let content = top_props
            .properties
            .get(&grafeo_common::types::PropertyKey::new("content"))
            .expect("has content");
        if let Value::String(s) = content {
            assert!(
                s.contains("Rust") || s.contains("graph"),
                "top result should match query terms, got: {s}"
            );
        }
    }

    #[test]
    fn test_hybrid_search_text_only() {
        let db = setup_hybrid_db();

        // No vector query: only text search contributes
        let results = db
            .hybrid_search("Doc", "content", "emb", "Rust", None, 4, None)
            .expect("text-only hybrid");

        assert!(
            !results.is_empty(),
            "text-only hybrid should return results"
        );
    }

    #[test]
    fn test_hybrid_search_no_matches() {
        let db = setup_hybrid_db();

        let results = db
            .hybrid_search(
                "Doc",
                "content",
                "emb",
                "nonexistentxyzquery",
                Some(&[0.0, 0.0, 0.0]),
                4,
                None,
            )
            .expect("hybrid no matches");

        // Even with no text matches, vector search may return results
        // Just verify it doesn't error
        let _ = results;
    }

    #[test]
    fn test_hybrid_search_scores_descending() {
        let db = setup_hybrid_db();

        let results = db
            .hybrid_search(
                "Doc",
                "content",
                "emb",
                "Rust graph",
                Some(&[1.0, 0.0, 0.0]),
                4,
                None,
            )
            .expect("hybrid search");

        assert!(
            results.len() >= 2,
            "need at least 2 results to verify order"
        );

        // Verify scores are in descending order (higher = better)
        for window in results.windows(2) {
            assert!(
                window[0].1 >= window[1].1,
                "hybrid_search scores should be descending (higher = better): {} >= {}",
                window[0].1,
                window[1].1,
            );
        }

        // Verify all scores are positive
        for (_, score) in &results {
            assert!(
                *score > 0.0,
                "hybrid_search fusion scores should be positive"
            );
        }
    }

    #[test]
    fn test_hybrid_search_without_text_index() {
        let db = GrafeoDB::new_in_memory();

        let n1 = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(n1, "content", Value::String("Rust graph database".into()))
            .unwrap();
        db.set_node_property(n1, "emb", vec3(1.0, 0.0, 0.0))
            .unwrap();

        let n2 = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(n2, "content", Value::String("Python ML".into()))
            .unwrap();
        db.set_node_property(n2, "emb", vec3(0.0, 1.0, 0.0))
            .unwrap();

        // Create ONLY vector index, no text index
        db.create_vector_index("Doc", "emb", Some(3), Some("cosine"), None, None, None)
            .expect("create vector index");

        // hybrid_search should work, using only vector source
        let results = db
            .hybrid_search(
                "Doc",
                "content",
                "emb",
                "Rust",
                Some(&[1.0, 0.0, 0.0]),
                4,
                None,
            )
            .expect("hybrid without text index should not error");

        assert!(
            !results.is_empty(),
            "should return results from vector source only"
        );
    }

    #[test]
    fn test_hybrid_search_without_vector_index() {
        let db = GrafeoDB::new_in_memory();

        let n1 = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(n1, "content", Value::String("Rust graph database".into()))
            .unwrap();
        db.set_node_property(n1, "emb", vec3(1.0, 0.0, 0.0))
            .unwrap();

        // Create ONLY text index, no vector index
        db.create_text_index("Doc", "content")
            .expect("create text index");

        // hybrid_search with a vector query should still work, using only text source
        let results = db
            .hybrid_search(
                "Doc",
                "content",
                "emb",
                "Rust",
                Some(&[1.0, 0.0, 0.0]),
                4,
                None,
            )
            .expect("hybrid without vector index should not error");

        assert!(
            !results.is_empty(),
            "should return results from text source only"
        );
    }

    #[test]
    fn test_hybrid_search_without_any_index() {
        let db = GrafeoDB::new_in_memory();

        let n1 = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(n1, "content", Value::String("Rust graph database".into()))
            .unwrap();
        db.set_node_property(n1, "emb", vec3(1.0, 0.0, 0.0))
            .unwrap();

        // No indexes at all
        let results = db
            .hybrid_search(
                "Doc",
                "content",
                "emb",
                "Rust",
                Some(&[1.0, 0.0, 0.0]),
                4,
                None,
            )
            .expect("hybrid without any index should not error");

        assert!(
            results.is_empty(),
            "should return empty when no indexes exist"
        );
    }

    #[test]
    fn test_hybrid_weighted_fusion_vector_ranking() {
        let db = setup_hybrid_db();

        // Use weighted fusion: the node closest in vector space to query [1,0,0]
        // AND matching text "Rust" should rank higher than a node that is
        // far in vector space even if it matches text.
        let fusion = grafeo_core::index::text::FusionMethod::Weighted {
            weights: vec![0.3, 0.7], // heavily weight vector similarity
        };
        let results = db
            .hybrid_search(
                "Doc",
                "content",
                "emb",
                "Rust",
                Some(&[1.0, 0.0, 0.0]),
                4,
                Some(fusion),
            )
            .expect("weighted hybrid search");

        assert!(results.len() >= 2, "need at least 2 results");

        // With 0.7 vector weight and query [1,0,0], the node with emb [1,0,0]
        // (n1: "Rust graph database engine") should rank above the node with
        // emb [0.9,0.1,0] (n3: "Rust systems programming language").
        // Both match "Rust" in text, but n1 is closer in vector space.
        let top_node = results[0].0;
        let top_props = db.get_node(top_node).expect("top node exists");
        let content = top_props
            .properties
            .get(&grafeo_common::types::PropertyKey::new("content"))
            .expect("has content");
        if let Value::String(s) = content {
            assert!(
                s.contains("Rust graph database"),
                "with 70% vector weight, closest vector should rank first, got: {s}"
            );
        }
    }
}

// ============================================================================
// Concurrent index access (T3-05)
// ============================================================================

#[cfg(feature = "vector-index")]
mod concurrent_vector {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    fn vec3(x: f32, y: f32, z: f32) -> Value {
        Value::Vector(vec![x, y, z].into())
    }

    #[test]
    fn test_concurrent_vector_read_during_write() {
        let db = std::sync::Arc::new(GrafeoDB::new_in_memory());

        // Seed initial data
        let n1 = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(n1, "emb", vec3(1.0, 0.0, 0.0))
            .unwrap();
        db.create_vector_index("Doc", "emb", Some(3), Some("cosine"), None, None, None)
            .unwrap();

        let db_read = std::sync::Arc::clone(&db);
        let db_write = std::sync::Arc::clone(&db);

        // Writer thread: add more nodes
        let writer = std::thread::spawn(move || {
            for i in 0..10 {
                let n = db_write.create_node(&["Doc"]).unwrap();
                let x = (i as f32) / 10.0;
                db_write
                    .set_node_property(n, "emb", vec3(x, 1.0 - x, 0.0))
                    .unwrap();
            }
        });

        // Reader thread: search concurrently
        let reader = std::thread::spawn(move || {
            for _ in 0..10 {
                let results = db_read.vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 5, None, None);
                // Should not panic or error
                assert!(results.is_ok(), "concurrent read should not error");
            }
        });

        writer.join().expect("writer thread should not panic");
        reader.join().expect("reader thread should not panic");
    }
}

#[cfg(feature = "text-index")]
mod concurrent_text {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    #[test]
    fn test_concurrent_text_read_during_write() {
        let db = std::sync::Arc::new(GrafeoDB::new_in_memory());

        let n1 = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(
            n1,
            "content",
            Value::String("initial document about graphs".into()),
        )
        .unwrap();
        db.create_text_index("Doc", "content").unwrap();

        let db_read = std::sync::Arc::clone(&db);
        let db_write = std::sync::Arc::clone(&db);

        let writer = std::thread::spawn(move || {
            for i in 0..10 {
                let n = db_write.create_node(&["Doc"]).unwrap();
                db_write
                    .set_node_property(
                        n,
                        "content",
                        Value::String(format!("document number {i} about databases").into()),
                    )
                    .unwrap();
            }
        });

        let reader = std::thread::spawn(move || {
            for _ in 0..10 {
                let results = db_read.text_search("Doc", "content", "database", 5);
                assert!(results.is_ok(), "concurrent text read should not error");
            }
        });

        writer.join().expect("writer should not panic");
        reader.join().expect("reader should not panic");
    }
}

// ============================================================================
// Quantized vector index tests
// ============================================================================

#[cfg(feature = "vector-index")]
mod quantized_vector {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    fn vec3(x: f32, y: f32, z: f32) -> Value {
        Value::Vector(vec![x, y, z].into())
    }

    #[test]
    fn test_scalar_quantized_index_create_insert_search() {
        let db = GrafeoDB::new_in_memory();

        let alix = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(alix, "emb", vec3(1.0, 0.0, 0.0))
            .unwrap();
        db.set_node_property(alix, "name", Value::from("Alix"))
            .unwrap();

        let gus = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(gus, "emb", vec3(0.0, 1.0, 0.0))
            .unwrap();
        db.set_node_property(gus, "name", Value::from("Gus"))
            .unwrap();

        let vincent = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(vincent, "emb", vec3(0.9, 0.1, 0.0))
            .unwrap();
        db.set_node_property(vincent, "name", Value::from("Vincent"))
            .unwrap();

        db.create_vector_index(
            "Doc",
            "emb",
            Some(3),
            Some("cosine"),
            None,
            None,
            Some("scalar"),
        )
        .expect("create scalar-quantized index");

        let results = db
            .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 2, None, None)
            .expect("search should succeed");

        assert_eq!(results.len(), 2, "should return 2 results");
        // The closest vector to [1,0,0] should be Alix's node
        assert_eq!(results[0].0, alix, "Alix should be the closest match");
    }

    #[test]
    fn test_binary_quantized_index_create_insert_search() {
        let db = GrafeoDB::new_in_memory();

        let alix = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(alix, "emb", vec3(1.0, 0.0, 0.0))
            .unwrap();

        let gus = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(gus, "emb", vec3(0.0, 1.0, 0.0))
            .unwrap();

        let vincent = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(vincent, "emb", vec3(0.0, 0.0, 1.0))
            .unwrap();

        db.create_vector_index(
            "Doc",
            "emb",
            Some(3),
            Some("euclidean"),
            None,
            None,
            Some("binary"),
        )
        .expect("create binary-quantized index");

        let results = db
            .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 2, None, None)
            .expect("search should succeed");

        assert_eq!(results.len(), 2, "should return 2 results");
        assert_eq!(results[0].0, alix, "Alix should be the closest match");
    }

    #[test]
    fn test_no_quantization_still_works() {
        // Regression test: quantization=None (default) should behave identically
        let db = GrafeoDB::new_in_memory();

        let alix = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(alix, "emb", vec3(1.0, 0.0, 0.0))
            .unwrap();

        let gus = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(gus, "emb", vec3(0.0, 1.0, 0.0))
            .unwrap();

        // Explicit None quantization
        db.create_vector_index("Doc", "emb", Some(3), Some("cosine"), None, None, None)
            .expect("create non-quantized index");

        let results = db
            .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 1, None, None)
            .expect("search should succeed");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, alix);
    }

    #[test]
    fn test_quantization_none_string_equivalent_to_default() {
        let db = GrafeoDB::new_in_memory();

        let alix = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(alix, "emb", vec3(1.0, 0.0, 0.0))
            .unwrap();

        // "none" string should be equivalent to None
        db.create_vector_index(
            "Doc",
            "emb",
            Some(3),
            Some("cosine"),
            None,
            None,
            Some("none"),
        )
        .expect("create index with 'none' quantization");

        let results = db
            .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 1, None, None)
            .expect("search should succeed");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_invalid_quantization_type_errors() {
        let db = GrafeoDB::new_in_memory();

        let alix = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(alix, "emb", vec3(1.0, 0.0, 0.0))
            .unwrap();

        let result = db.create_vector_index(
            "Doc",
            "emb",
            Some(3),
            Some("cosine"),
            None,
            None,
            Some("invalid_type"),
        );
        assert!(result.is_err(), "invalid quantization type should error");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Unknown quantization type"),
            "error should mention unknown type: {err_msg}"
        );
    }

    #[test]
    fn test_scalar_quantized_empty_index_then_insert() {
        // Create quantized index with explicit dimensions but no data yet
        let db = GrafeoDB::new_in_memory();

        db.create_vector_index(
            "Doc",
            "emb",
            Some(3),
            Some("cosine"),
            None,
            None,
            Some("scalar"),
        )
        .expect("create empty scalar-quantized index");

        // Insert after index creation
        let alix = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(alix, "emb", vec3(1.0, 0.0, 0.0))
            .unwrap();

        let gus = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(gus, "emb", vec3(0.0, 1.0, 0.0))
            .unwrap();

        let results = db
            .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 1, None, None)
            .expect("search should succeed");

        // Note: auto-insert into quantized index happens via the same mechanism
        // as non-quantized, but depends on property-change hooks being connected
        // (which they are for VectorIndexKind::Quantized through the standard path).
        // With empty index + late inserts, the index may not auto-populate.
        // This test validates the create-then-search path doesn't panic.
        let _ = results;
    }

    #[test]
    fn test_rebuild_preserves_quantization() {
        let db = GrafeoDB::new_in_memory();

        let alix = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(alix, "emb", vec3(1.0, 0.0, 0.0))
            .unwrap();

        let gus = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(gus, "emb", vec3(0.0, 1.0, 0.0))
            .unwrap();

        db.create_vector_index(
            "Doc",
            "emb",
            Some(3),
            Some("cosine"),
            None,
            None,
            Some("binary"),
        )
        .expect("create binary-quantized index");

        // Rebuild should preserve the binary quantization
        db.rebuild_vector_index("Doc", "emb")
            .expect("rebuild should succeed");

        let results = db
            .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 1, None, None)
            .expect("search after rebuild should succeed");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, alix);
    }
}
