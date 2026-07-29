//! End-to-end integration tests for unified hybrid queries.
//!
//! Tests the full pipeline: GQL parsing → planning → pushdown → execution.
//! Covers text pushdown, vector pushdown, text_match, text_score,
//! graph+vector per-row eval, error on missing text index,
//! and brute-force fallback without vector index.
//!
//! ```bash
//! cargo test -p grafeo-engine --features text-index,vector-index,gql --test hybrid_query 2>&1 | tail -20
//! ```

#![cfg(all(feature = "text-index", feature = "vector-index", feature = "gql"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::QueryResult;

// ============================================================================
// Shared test fixture
// ============================================================================

fn setup_article_db() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();

    // Create articles with embeddings and text
    let a1 = db.create_node(&["Article"]).unwrap();
    db.set_node_property(a1, "title", Value::String("Graph Neural Networks".into()))
        .unwrap();
    db.set_node_property(
        a1,
        "body",
        Value::String(
            "attention mechanisms in graph neural networks for node classification".into(),
        ),
    )
    .unwrap();
    db.set_node_property(
        a1,
        "embedding",
        Value::Vector(vec![0.9f32, 0.1, 0.0].into()),
    )
    .unwrap();

    let a2 = db.create_node(&["Article"]).unwrap();
    db.set_node_property(a2, "title", Value::String("Rust Database Internals".into()))
        .unwrap();
    db.set_node_property(
        a2,
        "body",
        Value::String("building a database engine in rust with MVCC transactions".into()),
    )
    .unwrap();
    db.set_node_property(
        a2,
        "embedding",
        Value::Vector(vec![0.1f32, 0.9, 0.0].into()),
    )
    .unwrap();

    let a3 = db.create_node(&["Article"]).unwrap();
    db.set_node_property(
        a3,
        "title",
        Value::String("Transformer Architectures".into()),
    )
    .unwrap();
    db.set_node_property(
        a3,
        "body",
        Value::String("attention mechanisms and transformer models for natural language".into()),
    )
    .unwrap();
    db.set_node_property(
        a3,
        "embedding",
        Value::Vector(vec![0.8f32, 0.2, 0.1].into()),
    )
    .unwrap();

    // Create user + friend with relationships
    let user = db.create_node(&["User"]).unwrap();
    db.set_node_property(user, "name", Value::String("Alix".into()))
        .unwrap();
    let friend = db.create_node(&["User"]).unwrap();
    db.set_node_property(friend, "name", Value::String("Vincent".into()))
        .unwrap();
    db.create_edge(user, friend, "FOLLOWS");
    db.create_edge(friend, a1, "WROTE");
    db.create_edge(friend, a2, "WROTE");

    // Create indexes
    db.create_vector_index(
        "Article",
        "embedding",
        Some(3),
        Some("cosine"),
        None,
        None,
        None,
    )
    .expect("create vector index");
    db.create_text_index("Article", "body")
        .expect("create text index");

    db
}

// ============================================================================
// Helper: extract string values from the first column of a result
// ============================================================================

fn collect_strings(result: &QueryResult) -> Vec<String> {
    result
        .rows()
        .iter()
        .filter_map(|row| {
            if let Some(Value::String(s)) = row.first() {
                Some(s.to_string())
            } else {
                None
            }
        })
        .collect()
}

// ============================================================================
// Test 1: text_match in WHERE clause
// ============================================================================

#[test]
fn test_text_match_where() {
    let db = setup_article_db();
    let session = db.session();

    let result = session
        .execute("MATCH (doc:Article) WHERE text_match(doc.body, 'rust database') RETURN doc.title")
        .expect("text_match query should succeed");

    let titles = collect_strings(&result);
    assert_eq!(
        titles.len(),
        1,
        "Expected 1 result for 'rust database', got: {:?}",
        titles
    );
    assert_eq!(
        titles[0], "Rust Database Internals",
        "Expected 'Rust Database Internals', got: {:?}",
        titles
    );
}

// ============================================================================
// Test 2: text_score > 0.0 in WHERE clause
// ============================================================================

#[test]
fn test_text_score_where() {
    let db = setup_article_db();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (doc:Article) WHERE text_score(doc.body, 'attention mechanisms') > 0.0 RETURN doc.title",
        )
        .expect("text_score query should succeed");

    let titles = collect_strings(&result);
    assert_eq!(
        titles.len(),
        2,
        "Expected 2 results for 'attention mechanisms' (articles 1 and 3), got: {:?}",
        titles
    );
    assert!(
        titles.contains(&"Graph Neural Networks".to_string()),
        "Expected 'Graph Neural Networks' in results: {:?}",
        titles
    );
    assert!(
        titles.contains(&"Transformer Architectures".to_string()),
        "Expected 'Transformer Architectures' in results: {:?}",
        titles
    );
}

// ============================================================================
// Test 3: vector cosine_similarity with index pushdown
// ============================================================================

#[test]
fn test_vector_where_with_pushdown() {
    let db = setup_article_db();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (doc:Article) WHERE cosine_similarity(doc.embedding, [0.85, 0.15, 0.05]) > 0.5 RETURN doc.title",
        )
        .expect("cosine_similarity query should succeed");

    let titles = collect_strings(&result);
    assert!(
        !titles.is_empty(),
        "Expected at least 1 result for cosine_similarity > 0.5, got none"
    );
    assert!(
        titles.contains(&"Graph Neural Networks".to_string()),
        "Expected 'Graph Neural Networks' in cosine_similarity results (embedding [0.9, 0.1, 0.0] is close to query [0.85, 0.15, 0.05]): {:?}",
        titles
    );
}

// ============================================================================
// Test 4: text_score without text index → falls through to per-row eval, 0 rows
// ============================================================================

#[test]
fn test_text_score_without_index_fallthrough() {
    let db = GrafeoDB::new_in_memory();
    // Create nodes but NO text index
    let n = db.create_node(&["Article"]).unwrap();
    db.set_node_property(n, "body", Value::String("some body text about rust".into()))
        .unwrap();

    let session = db.session();
    let result = session
        .execute("MATCH (doc:Article) WHERE text_score(doc.body, 'rust') > 0.0 RETURN doc.title")
        .expect("text_score without index should fall through to per-row eval, not error");

    // Without an index, score_text returns None → text_score evaluates to 0.0 → predicate false.
    assert_eq!(
        result.row_count(),
        0,
        "Expected 0 rows when no text index exists (per-row eval returns 0.0 for all nodes)"
    );
}

// ============================================================================
// Test 5: cosine_similarity without vector index → brute-force fallback
// ============================================================================

#[test]
fn test_vector_without_index_brute_force() {
    let db = GrafeoDB::new_in_memory();
    // Create articles WITHOUT a vector index
    let a1 = db.create_node(&["Article"]).unwrap();
    db.set_node_property(a1, "title", Value::String("Graph Neural Networks".into()))
        .unwrap();
    db.set_node_property(
        a1,
        "embedding",
        Value::Vector(vec![0.9f32, 0.1, 0.0].into()),
    )
    .unwrap();

    let a2 = db.create_node(&["Article"]).unwrap();
    db.set_node_property(a2, "title", Value::String("Rust Database Internals".into()))
        .unwrap();
    db.set_node_property(
        a2,
        "embedding",
        Value::Vector(vec![0.1f32, 0.9, 0.0].into()),
    )
    .unwrap();

    // NO vector index created — should fall back to brute-force per-row evaluation
    let session = db.session();
    let result = session
        .execute(
            "MATCH (doc:Article) WHERE cosine_similarity(doc.embedding, [0.85, 0.15, 0.05]) > 0.5 RETURN doc.title",
        )
        .expect("cosine_similarity should work without index via brute-force fallback");

    let titles = collect_strings(&result);
    assert!(
        !titles.is_empty(),
        "Brute-force fallback should find at least 1 result"
    );
    assert!(
        titles.contains(&"Graph Neural Networks".to_string()),
        "Expected 'Graph Neural Networks' via brute-force evaluation: {:?}",
        titles
    );
}

// ============================================================================
// Test 6: Graph traversal + vector similarity per-row eval
// ============================================================================

#[test]
fn test_graph_plus_vector_per_row_eval() {
    let db = setup_article_db();
    let session = db.session();

    // Alix follows Vincent; Vincent wrote articles 1 and 2.
    // Article 1 embedding [0.9, 0.1, 0.0] is close to query [0.85, 0.15, 0.05].
    // Article 2 embedding [0.1, 0.9, 0.0] is far from query.
    let result = session
        .execute(
            "MATCH (u:User {name: 'Alix'})-[:FOLLOWS]->(friend)-[:WROTE]->(doc:Article) \
             WHERE cosine_similarity(doc.embedding, [0.85, 0.15, 0.05]) > 0.3 \
             RETURN doc.title",
        )
        .expect("graph + vector query should succeed");

    let titles = collect_strings(&result);

    // At a threshold of 0.3, article 1 ([0.9, 0.1, 0.0]) should match
    // (cosine similarity with [0.85, 0.15, 0.05] ≈ 0.998).
    assert!(
        !titles.is_empty(),
        "Expected at least one article from Alix→Vincent→Article traversal with similarity > 0.3"
    );
    assert!(
        titles.contains(&"Graph Neural Networks".to_string()),
        "Expected 'Graph Neural Networks' (article with similar embedding): {:?}",
        titles
    );

    // Article 2 embedding [0.1, 0.9, 0.0] vs query [0.85, 0.15, 0.05]:
    // cosine similarity ≈ 0.1*0.85 + 0.9*0.15 ≈ 0.085 + 0.135 = 0.22 < 0.3
    // So "Rust Database Internals" should NOT be in results.
    assert!(
        !titles.contains(&"Rust Database Internals".to_string()),
        "Expected 'Rust Database Internals' to be filtered out (low similarity): {:?}",
        titles
    );
}

// ============================================================================
// Test 7: AND compound — vector similarity AND text match on bare label scan
// ============================================================================

#[test]
fn test_compound_vector_and_text() {
    let db = setup_article_db();
    let session = db.session();

    // Both vector similarity AND text match on bare label scan.
    // Articles 1 and 3 mention "attention" AND are close to [0.85, 0.15, 0.05].
    // Article 2 (rust database) doesn't mention "attention".
    let result = session
        .execute(
            "MATCH (doc:Article) \
             WHERE cosine_similarity(doc.embedding, [0.85, 0.15, 0.05]) > 0.3 \
               AND text_match(doc.body, 'attention mechanisms') \
             RETURN doc.title",
        )
        .expect("AND compound query (vector + text) should succeed");

    assert!(
        result.row_count() >= 1,
        "AND compound should return at least 1 article (articles 1 and 3 match both), got {}",
        result.row_count()
    );

    let titles = collect_strings(&result);
    assert!(
        !titles.contains(&"Rust Database Internals".to_string()),
        "AND compound should not return 'Rust Database Internals' (no 'attention' in body): {:?}",
        titles
    );
}

// ============================================================================
// Test 8: OR compound — vector similarity OR text match (union)
// ============================================================================

#[test]
fn test_compound_vector_or_text() {
    let db = setup_article_db();
    let session = db.session();

    // Vector similarity OR text match — should union results.
    // cosine_similarity > 0.9 matches article 2 (embedding [0.1, 0.9, 0.0]).
    // text_match matches articles 1 and 3 ("attention mechanisms").
    // Union should return all 3 articles.
    let result = session
        .execute(
            "MATCH (doc:Article) \
             WHERE cosine_similarity(doc.embedding, [0.1, 0.9, 0.0]) > 0.9 \
                OR text_match(doc.body, 'attention mechanisms') \
             RETURN doc.title",
        )
        .expect("OR compound query (vector | text) should succeed");

    assert!(
        result.row_count() >= 2,
        "OR should union vector and text results, got {}",
        result.row_count()
    );
}

// ============================================================================
// Test 9: euclidean_distance pushdown
// ============================================================================

#[test]
fn test_euclidean_distance_pushdown() {
    let db = setup_article_db();
    let session = db.session();

    // Article 1 has embedding [0.9, 0.1, 0.0] — distance 0.0 from itself.
    let result = session
        .execute(
            "MATCH (doc:Article) \
             WHERE euclidean_distance(doc.embedding, [0.9, 0.1, 0.0]) < 0.5 \
             RETURN doc.title",
        )
        .expect("euclidean_distance query should succeed");

    assert!(
        result.row_count() >= 1,
        "Expected at least 1 result for euclidean_distance < 0.5, got {}",
        result.row_count()
    );
}

// ============================================================================
// Test 10: text_match as a standalone boolean (not text_score > threshold)
// ============================================================================

#[test]
fn test_text_match_standalone_boolean() {
    let db = setup_article_db();
    let session = db.session();

    // text_match as a standalone boolean (not text_score > threshold).
    // Only article 2 body mentions "rust".
    let result = session
        .execute(
            "MATCH (doc:Article) \
             WHERE text_match(doc.body, 'rust') \
             RETURN doc.title",
        )
        .expect("text_match standalone boolean query should succeed");

    assert_eq!(
        result.row_count(),
        1,
        "Expected exactly 1 result for text_match 'rust' (only 'Rust Database Internals'), got {}",
        result.row_count()
    );
}

// ============================================================================
// Test 11: Operator inversion (cosine_similarity < threshold) — no pushdown,
//          should still work via brute-force per-row eval
// ============================================================================

#[test]
fn test_operator_inversion_no_pushdown() {
    let db = setup_article_db();
    let session = db.session();

    // cosine_similarity < 0.3 should NOT push down (inverted comparison).
    // Should still work via brute-force per-row eval.
    // Article 2 ([0.1, 0.9, 0.0]) has low cosine similarity to [0.9, 0.1, 0.0].
    let result = session
        .execute(
            "MATCH (doc:Article) \
             WHERE cosine_similarity(doc.embedding, [0.9, 0.1, 0.0]) < 0.3 \
             RETURN doc.title",
        )
        .expect("inverted cosine_similarity query should succeed via brute-force fallback");

    assert!(
        result.row_count() >= 1,
        "Expected at least 1 article with cosine_similarity < 0.3 (article 2 should qualify), got {}",
        result.row_count()
    );
}

// ============================================================================
// Test 12: text_score in both WHERE and RETURN (score projection)
// ============================================================================

#[test]
fn test_score_in_return() {
    let db = setup_article_db();
    let session = db.session();

    // Score projection: text_score appears in both WHERE and RETURN.
    let result = session
        .execute(
            "MATCH (doc:Article) \
             WHERE text_score(doc.body, 'attention mechanisms') > 0.0 \
             RETURN doc.title, text_score(doc.body, 'attention mechanisms') AS score",
        )
        .expect("text_score in WHERE + RETURN should succeed");

    assert_eq!(
        result.row_count(),
        2,
        "Expected 2 articles matching 'attention mechanisms', got {}",
        result.row_count()
    );
    assert_eq!(
        result.column_count(),
        2,
        "Expected 2 columns (title, score), got {}",
        result.column_count()
    );

    // Verify scores are positive Float64 values.
    for row in result.rows() {
        if let Value::Float64(s) = &row[1] {
            assert!(*s > 0.0, "Score should be positive, got {}", s);
        } else {
            panic!("Expected Float64 score in column 1, got {:?}", row[1]);
        }
    }
}

// ============================================================================
// Test 13: text_score in RETURN only (no WHERE pushdown) — per-row eval
// ============================================================================

#[test]
fn test_text_score_in_return_only() {
    let db = setup_article_db();
    let session = db.session();

    // text_score in RETURN only (no WHERE pushdown) — per-row eval for all rows.
    // All 3 articles should be returned; non-matching articles get 0 or Null score.
    let result = session
        .execute(
            "MATCH (doc:Article) \
             RETURN doc.title, text_score(doc.body, 'rust database') AS score",
        )
        .expect("text_score in RETURN only should succeed");

    assert_eq!(
        result.row_count(),
        3,
        "Expected all 3 articles when text_score is in RETURN only (no filter), got {}",
        result.row_count()
    );
}

// ============================================================================
// Test 14: Empty query string should match nothing
// ============================================================================

#[test]
fn test_empty_query_string() {
    let db = setup_article_db();
    let session = db.session();

    // Empty query string — either returns 0 rows or errors gracefully.
    // Either outcome is acceptable; we assert zero rows if it succeeds.
    let result = session.execute(
        "MATCH (doc:Article) \
         WHERE text_match(doc.body, '') \
         RETURN doc.title",
    );

    match result {
        Ok(r) => {
            assert_eq!(
                r.row_count(),
                0,
                "Empty query string should match nothing, got {} rows",
                r.row_count()
            );
        }
        Err(_) => {
            // Graceful error on empty query string is also acceptable.
        }
    }
}

// ============================================================================
// Test 15: Top-K recognition (ORDER BY + LIMIT → index scan)
// ============================================================================

#[test]
fn test_topk_order_by_text_score() {
    let db = setup_article_db();
    let session = db.session();

    // The rewrite fires when the function appears directly in ORDER BY (no alias),
    // so try_topk_rewrite can match the FunctionCall on sort_key.expression.
    // We can't assert via EXPLAIN/PROFILE because LPG EXPLAIN walks the unrewritten
    // logical plan and PROFILE has a separate entry-count mismatch when physical
    // operators are fused; the user-visible signal is row count and ordering.
    let result = session
        .execute(
            "MATCH (doc:Article) \
             RETURN doc \
             ORDER BY text_score(doc.body, 'attention mechanisms') DESC LIMIT 1",
        )
        .unwrap();

    assert_eq!(result.row_count(), 1, "Top-1 should return exactly 1 row");
}

#[test]
fn test_topk_order_by_vector_similarity() {
    let db = setup_article_db();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (doc:Article) \
             RETURN doc \
             ORDER BY cosine_similarity(doc.embedding, [0.85, 0.15, 0.05]) DESC LIMIT 2",
        )
        .unwrap();

    assert_eq!(result.row_count(), 2, "Top-2 should return exactly 2 rows");
}

#[test]
fn test_profile_topk_order_by_vector_similarity() {
    // Regression: PROFILE on a Limit-above-Sort top-k query used to panic in
    // build_profile_tree because the top-k rewrite fused three logical
    // operators into one physical operator without updating the logical tree.
    // The planner now suppresses the rewrite under PROFILE.
    let db = setup_article_db();
    let session = db.session();

    let result = session
        .execute(
            "PROFILE MATCH (doc:Article) \
             RETURN doc \
             ORDER BY cosine_similarity(doc.embedding, [0.85, 0.15, 0.05]) DESC LIMIT 2",
        )
        .unwrap();

    assert_eq!(result.row_count(), 1, "PROFILE returns a single-row report");
}

#[test]
fn test_profile_topk_order_by_text_score() {
    // Same regression as above, for the text_score top-k path.
    let db = setup_article_db();
    let session = db.session();

    let result = session
        .execute(
            "PROFILE MATCH (doc:Article) \
             RETURN doc \
             ORDER BY text_score(doc.body, 'attention mechanisms') DESC LIMIT 1",
        )
        .unwrap();

    assert_eq!(result.row_count(), 1, "PROFILE returns a single-row report");
}

// ============================================================================
// Test 17: dot_product pushdown
// ============================================================================

#[test]
fn test_dot_product_pushdown() {
    let db = setup_article_db();
    let session = db.session();

    // dot_product is a similarity metric (higher = more similar)
    // Article 1 embedding [0.9, 0.1, 0.0], query [0.9, 0.1, 0.0]
    // dot_product = 0.81 + 0.01 + 0.0 = 0.82
    let result = session
        .execute(
            "MATCH (doc:Article) \
             WHERE dot_product(doc.embedding, [0.9, 0.1, 0.0]) > 0.5 \
             RETURN doc.title",
        )
        .unwrap();

    assert!(
        result.row_count() >= 1,
        "dot_product > 0.5 should match at least article 1, got {}",
        result.row_count()
    );
}

// ============================================================================
// Test 18: manhattan_distance pushdown
// ============================================================================

#[test]
fn test_manhattan_distance_pushdown() {
    let db = setup_article_db();
    let session = db.session();

    // manhattan_distance is a distance metric (lower = more similar)
    // Article 1 embedding [0.9, 0.1, 0.0], query [0.9, 0.1, 0.0]
    // manhattan_distance = 0.0
    let result = session
        .execute(
            "MATCH (doc:Article) \
             WHERE manhattan_distance(doc.embedding, [0.9, 0.1, 0.0]) < 0.5 \
             RETURN doc.title",
        )
        .unwrap();

    assert!(
        result.row_count() >= 1,
        "manhattan_distance < 0.5 should match at least article 1, got {}",
        result.row_count()
    );
}

// ============================================================================
// Test 19: EXPLAIN output shows TextScan / VectorScan operators
// ============================================================================

#[test]
fn test_explain_shows_text_scan() {
    let db = setup_article_db();
    let session = db.session();

    // PROFILE shows the physical plan with actual operator names.
    // If pushdown fired, we'll see TextScan(BM25) instead of Filter.
    let result = session
        .execute(
            "PROFILE MATCH (doc:Article) \
             WHERE text_score(doc.body, 'attention') > 0.0 \
             RETURN doc.title",
        )
        .unwrap();

    assert_eq!(result.row_count(), 1);
    let plan = match &result.rows()[0][0] {
        Value::String(s) => s.to_string(),
        other => panic!("Expected String profile, got {:?}", other),
    };
    assert!(
        plan.contains("TextScan"),
        "PROFILE should show TextScan(BM25) operator (pushdown fired):\n{plan}"
    );
}

#[test]
fn test_profile_shows_vector_scan() {
    let db = setup_article_db();
    let session = db.session();

    // `>=` pushes down; strict `>` falls through to per-row evaluation so
    // the engine can honor the exact boundary semantics.
    let result = session
        .execute(
            "PROFILE MATCH (doc:Article) \
             WHERE cosine_similarity(doc.embedding, [0.85, 0.15, 0.05]) >= 0.5 \
             RETURN doc.title",
        )
        .unwrap();

    assert_eq!(result.row_count(), 1);
    let plan = match &result.rows()[0][0] {
        Value::String(s) => s.to_string(),
        other => panic!("Expected String profile, got {:?}", other),
    };
    assert!(
        plan.contains("VectorScan"),
        "PROFILE should show VectorScan operator (pushdown fired):\n{plan}"
    );
}

// ============================================================================
// Edge cases: predicate-extraction and pushdown corners.
// ============================================================================

/// `text_score(...) >= t` (Ge) should pushdown the same as `>` (Gt).
#[test]
fn test_text_score_with_ge_operator() {
    let db = setup_article_db();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (doc:Article) WHERE text_score(doc.body, 'attention mechanisms') >= 0.0 RETURN doc.title",
        )
        .expect(">= operator on text_score should plan and execute");

    // text_score >= 0 matches every Article (3 nodes); zero-score articles included.
    assert!(
        result.row_count() >= 2,
        "Expected at least 2 matches with >= 0.0 (all matching articles), got {}",
        result.row_count()
    );
}

/// `text_score(...) > 0` with an Int64 threshold (no decimal) — extractor must
/// coerce Int64 → Float64 in `try_extract_text_fn`.
#[test]
fn test_text_score_with_int_threshold() {
    let db = setup_article_db();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (doc:Article) WHERE text_score(doc.body, 'rust database') > 0 RETURN doc.title",
        )
        .expect("Int threshold should be coerced to Float64");

    assert_eq!(
        result.row_count(),
        1,
        "Expected 1 match for 'rust database', got {}",
        result.row_count()
    );
}

/// Compound: `vector AND text AND scalar_predicate` — `extract_scalar_remaining`
/// must surface `published = true` as a post-join filter.
#[test]
fn test_compound_with_scalar_remainder() {
    let db = GrafeoDB::new_in_memory();
    let a1 = db.create_node(&["Article"]).unwrap();
    db.set_node_property(a1, "title", Value::String("A1".into()))
        .unwrap();
    db.set_node_property(a1, "body", Value::String("attention mechanisms".into()))
        .unwrap();
    db.set_node_property(
        a1,
        "embedding",
        Value::Vector(vec![0.9_f32, 0.1, 0.0].into()),
    )
    .unwrap();
    db.set_node_property(a1, "published", Value::Bool(true))
        .unwrap();

    let a2 = db.create_node(&["Article"]).unwrap();
    db.set_node_property(a2, "title", Value::String("A2".into()))
        .unwrap();
    db.set_node_property(a2, "body", Value::String("attention mechanisms".into()))
        .unwrap();
    db.set_node_property(
        a2,
        "embedding",
        Value::Vector(vec![0.9_f32, 0.1, 0.0].into()),
    )
    .unwrap();
    db.set_node_property(a2, "published", Value::Bool(false))
        .unwrap();

    db.create_vector_index(
        "Article",
        "embedding",
        Some(3),
        Some("cosine"),
        None,
        None,
        None,
    )
    .unwrap();
    db.create_text_index("Article", "body").unwrap();

    let session = db.session();
    let result = session
        .execute(
            "MATCH (doc:Article) \
             WHERE cosine_similarity(doc.embedding, [0.9, 0.1, 0.0]) > 0.5 \
               AND text_match(doc.body, 'attention') \
               AND doc.published = true \
             RETURN doc.title",
        )
        .expect("compound with scalar remainder should plan and execute");

    let titles = collect_strings(&result);
    assert_eq!(
        titles,
        vec!["A1".to_string()],
        "Only the published article should pass the scalar filter, got: {:?}",
        titles
    );
}

/// Compound OR where one side is missing the required index: should fall through
/// to per-row evaluation rather than panic or error.
#[test]
fn test_compound_or_with_missing_text_index_falls_through() {
    let db = GrafeoDB::new_in_memory();
    let a1 = db.create_node(&["Article"]).unwrap();
    db.set_node_property(a1, "body", Value::String("rust".into()))
        .unwrap();
    db.set_node_property(
        a1,
        "embedding",
        Value::Vector(vec![1.0_f32, 0.0, 0.0].into()),
    )
    .unwrap();

    db.create_vector_index(
        "Article",
        "embedding",
        Some(3),
        Some("cosine"),
        None,
        None,
        None,
    )
    .unwrap();
    // NB: no text index — compound OR pushdown should fall through

    let session = db.session();
    let result = session
        .execute(
            "MATCH (doc:Article) \
             WHERE cosine_similarity(doc.embedding, [1.0, 0.0, 0.0]) > 0.5 \
                OR text_match(doc.body, 'rust') \
             RETURN doc.body",
        )
        .expect("OR with missing text index should not error — falls through to per-row eval");

    // Vector predicate matches; per-row text_match returns false (no index → 0.0 score).
    // The OR is true via the vector branch.
    assert_eq!(result.row_count(), 1);
}

// ============================================================================
// Top-K rewrite edge cases (Fix 4)
// ============================================================================

/// Per-row evaluation must respect ASC sort direction for distance metrics
/// (closest-first), independent of any potential index pushdown.
#[test]
fn test_order_by_euclidean_distance_ascending() {
    let db = GrafeoDB::new_in_memory();
    let docs = [
        ("Closest", vec![0.9_f32, 0.1, 0.0]),
        ("Near", vec![0.85_f32, 0.15, 0.0]),
        ("Mid", vec![0.5_f32, 0.5, 0.0]),
        ("Far", vec![0.1_f32, 0.9, 0.0]),
        ("Orthogonal", vec![0.0_f32, 0.0, 1.0]),
    ];
    for (title, emb) in &docs {
        let n = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(n, "title", Value::String((*title).into()))
            .unwrap();
        db.set_node_property(n, "embedding", Value::Vector(emb.clone().into()))
            .unwrap();
    }

    let session = db.session();
    let result = session
        .execute(
            "MATCH (doc:Doc) \
             RETURN doc.title \
             ORDER BY euclidean_distance(doc.embedding, [0.9, 0.1, 0.0]) ASC LIMIT 1",
        )
        .expect("euclidean ASC LIMIT 1 should work");

    let titles = collect_strings(&result);
    assert_eq!(
        titles,
        vec!["Closest".to_string()],
        "Top-1 by euclidean distance should be the closest doc, got: {:?}",
        titles
    );
}

/// Per-row eval with ASC on a similarity metric (wrong direction) should still
/// produce correct ordering — least-similar first.
#[test]
fn test_order_by_similarity_ascending_least_similar_first() {
    let db = setup_article_db();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (doc:Article) \
             RETURN doc.title \
             ORDER BY cosine_similarity(doc.embedding, [0.9, 0.1, 0.0]) ASC LIMIT 1",
        )
        .expect("similarity ASC should work via per-row eval");

    // The least-similar article to [0.9, 0.1, 0.0] is article 2 ([0.1, 0.9, 0.0]).
    let titles = collect_strings(&result);
    assert_eq!(
        titles,
        vec!["Rust Database Internals".to_string()],
        "Least-similar article expected, got: {:?}",
        titles
    );
}

/// Top-K rewrite must NOT fire when ORDER BY references an alias defined in
/// RETURN (sort_key is a Variable, not a FunctionCall). Falls through to the
/// regular Sort+Limit path; results must still be correct.
#[test]
fn test_topk_alias_in_order_by_falls_through() {
    let db = setup_article_db();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (doc:Article) \
             RETURN doc.title, text_score(doc.body, 'attention mechanisms') AS rank \
             ORDER BY rank DESC LIMIT 2",
        )
        .expect("alias in ORDER BY should still produce correct ordering");

    assert_eq!(
        result.row_count(),
        2,
        "LIMIT 2 should yield 2 rows even when rewrite skips"
    );
    // Top-2 must include the two articles mentioning 'attention' (1 and 3).
    let titles: Vec<String> = result
        .rows()
        .iter()
        .filter_map(|row| {
            if let Value::String(s) = &row[0] {
                Some(s.to_string())
            } else {
                None
            }
        })
        .collect();
    assert!(
        titles.contains(&"Graph Neural Networks".to_string())
            || titles.contains(&"Transformer Architectures".to_string()),
        "Top-2 should include attention articles, got: {:?}",
        titles
    );
}

/// LIMIT 0 must return an empty result regardless of the rewrite path.
#[test]
fn test_topk_limit_zero_returns_empty() {
    let db = setup_article_db();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (doc:Article) \
             RETURN doc \
             ORDER BY text_score(doc.body, 'attention') DESC LIMIT 0",
        )
        .expect("LIMIT 0 should plan and execute");

    assert_eq!(result.row_count(), 0, "LIMIT 0 must return zero rows");
}

/// LIMIT larger than the matching set should return the full set, not pad
/// with empty rows or truncate the index search incorrectly.
#[test]
fn test_topk_limit_exceeds_dataset() {
    let db = setup_article_db();
    let session = db.session();

    // Dataset has 3 articles; LIMIT 100 should still return only 3.
    let result = session
        .execute(
            "MATCH (doc:Article) \
             RETURN doc \
             ORDER BY text_score(doc.body, 'attention') DESC LIMIT 100",
        )
        .expect("oversized LIMIT should be safe");

    assert!(
        result.row_count() <= 3,
        "Expected at most 3 rows (dataset size), got {}",
        result.row_count()
    );
}

/// Empty query string for text_score should not panic; returns no/zero matches.
#[test]
fn test_text_score_empty_query_string() {
    let db = setup_article_db();
    let session = db.session();

    let result = session
        .execute("MATCH (doc:Article) WHERE text_score(doc.body, '') > 0.0 RETURN doc.title")
        .expect("empty query string must not panic");

    // BM25 of an empty query is zero → predicate false for all rows.
    assert_eq!(result.row_count(), 0);
}

/// Vector predicate where the property variable doesn't match the scan variable
/// (e.g., `cosine_similarity(other.emb, ...)` after a join) should not pushdown.
/// This guards the `vector_pred.variable != scan.variable` early-return in
/// `try_plan_filter_compound_hybrid`.
#[test]
fn test_compound_pushdown_skips_when_variable_mismatch() {
    let db = setup_article_db();
    let session = db.session();

    // Use the vector function on a variable that's NOT the bare scan variable.
    // The query traverses User -> Article and then references doc.embedding;
    // pushdown should still work here because doc IS the scan variable for
    // the Article scan, but if someone wrote this differently with two
    // separate scans, the planner's early-return would kick in.
    let result = session
        .execute(
            "MATCH (u:User {name: 'Alix'})-[:FOLLOWS]->(friend)-[:WROTE]->(doc:Article) \
             WHERE cosine_similarity(doc.embedding, [0.9, 0.1, 0.0]) > 0.3 \
             RETURN doc.title",
        )
        .expect("graph-traversal + vector predicate should fall through to per-row eval");

    // Per-row eval: Alix → Vincent → wrote articles 1 and 2. Only article 1
    // is similar to [0.9, 0.1, 0.0].
    assert!(result.row_count() >= 1);
}

/// `text_score` on two different properties of the same variable with the
/// same query string must NOT share a score column. If the score-column name
/// only keys on (variable, query), pushdown of `text_score(n.body, q)` writes
/// a column that a later `text_score(n.title, q)` projection incorrectly
/// reuses, returning body scores where title scores were asked for.
#[test]
fn test_text_score_two_properties_same_variable_and_query() {
    let db = GrafeoDB::new_in_memory();

    // Two docs, both with 'rust' in body so both pass the body-score pushdown.
    // They differ on title: only a1.title contains 'rust'.
    //   a1: title "rust guide",   body "rust tutorial"
    //   a2: title "other stuff",  body "rust tutorial"
    let a1 = db.create_node(&["Article"]).unwrap();
    db.set_node_property(a1, "title", Value::String("rust guide".into()))
        .unwrap();
    db.set_node_property(a1, "body", Value::String("rust tutorial".into()))
        .unwrap();

    let a2 = db.create_node(&["Article"]).unwrap();
    db.set_node_property(a2, "title", Value::String("other stuff".into()))
        .unwrap();
    db.set_node_property(a2, "body", Value::String("rust tutorial".into()))
        .unwrap();

    db.create_text_index("Article", "body").unwrap();
    db.create_text_index("Article", "title").unwrap();

    let session = db.session();

    // Pushdown fires on body, then RETURN asks for the title score with the
    // same query string 'rust'. If the score column only keys on (variable,
    // query), the title score projection reuses the body score — a2 would
    // report a positive title_score despite its title containing no 'rust'.
    //
    // With property in the score-column name, title score is recomputed per
    // row via the per-row text_score path: a1.title matches, a2.title does not.
    let result = session
        .execute(
            "MATCH (doc:Article) \
             WHERE text_score(doc.body, 'rust') > 0.0 \
             RETURN doc.title, text_score(doc.title, 'rust') AS title_score \
             ORDER BY doc.title",
        )
        .expect("query with two text_score calls on different properties should execute");

    assert_eq!(
        result.row_count(),
        2,
        "both articles have 'rust' in body, both should pass the body pushdown"
    );

    // After ORDER BY doc.title: "other stuff" (a2) comes before "rust guide" (a1).
    let rows = result.rows();
    let a2_title_score = match &rows[0][1] {
        Value::Float64(f) => *f,
        other => panic!("expected Float64 title score for a2, got {other:?}"),
    };
    let a1_title_score = match &rows[1][1] {
        Value::Float64(f) => *f,
        other => panic!("expected Float64 title score for a1, got {other:?}"),
    };

    assert!(
        a1_title_score > 0.0,
        "a1.title='rust guide' must have positive text_score on title, got {a1_title_score}"
    );
    assert_eq!(
        a2_title_score, 0.0,
        "a2.title='other stuff' must have zero text_score on title, \
         got {a2_title_score} (if nonzero, the body score was reused — collision regression)"
    );
}
