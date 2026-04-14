//! End-to-end integration tests for unified hybrid queries.
//!
//! Tests the full pipeline: Cypher parsing → planning → pushdown → execution.
//! Covers text pushdown, vector pushdown, text_match, text_score,
//! graph+vector per-row eval, error on missing text index,
//! and brute-force fallback without vector index.
//!
//! ```bash
//! cargo test -p grafeo-engine --features text-index,vector-index,gql --test hybrid_query 2>&1 | tail -20
//! ```

#![cfg(all(feature = "text-index", feature = "vector-index", feature = "gql"))]

use grafeo_common::types::Value;
use grafeo_engine::database::QueryResult;
use grafeo_engine::GrafeoDB;

// ============================================================================
// Shared test fixture
// ============================================================================

fn setup_article_db() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();

    // Create articles with embeddings and text
    let a1 = db.create_node(&["Article"]);
    db.set_node_property(a1, "title", Value::String("Graph Neural Networks".into()));
    db.set_node_property(
        a1,
        "body",
        Value::String(
            "attention mechanisms in graph neural networks for node classification".into(),
        ),
    );
    db.set_node_property(
        a1,
        "embedding",
        Value::Vector(vec![0.9f32, 0.1, 0.0].into()),
    );

    let a2 = db.create_node(&["Article"]);
    db.set_node_property(a2, "title", Value::String("Rust Database Internals".into()));
    db.set_node_property(
        a2,
        "body",
        Value::String("building a database engine in rust with MVCC transactions".into()),
    );
    db.set_node_property(
        a2,
        "embedding",
        Value::Vector(vec![0.1f32, 0.9, 0.0].into()),
    );

    let a3 = db.create_node(&["Article"]);
    db.set_node_property(
        a3,
        "title",
        Value::String("Transformer Architectures".into()),
    );
    db.set_node_property(
        a3,
        "body",
        Value::String(
            "attention mechanisms and transformer models for natural language".into(),
        ),
    );
    db.set_node_property(
        a3,
        "embedding",
        Value::Vector(vec![0.8f32, 0.2, 0.1].into()),
    );

    // Create user + friend with relationships
    let user = db.create_node(&["User"]);
    db.set_node_property(user, "name", Value::String("Alice".into()));
    let friend = db.create_node(&["User"]);
    db.set_node_property(friend, "name", Value::String("Bob".into()));
    db.create_edge(user, friend, "FOLLOWS");
    db.create_edge(friend, a1, "WROTE");
    db.create_edge(friend, a2, "WROTE");

    // Create indexes
    db.create_vector_index("Article", "embedding", Some(3), Some("cosine"), None, None, None)
        .expect("create vector index");
    db.create_text_index("Article", "body").expect("create text index");

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
// Test 4: text_score without text index → must error
// ============================================================================

#[test]
fn test_text_score_without_index_errors() {
    let db = GrafeoDB::new_in_memory();
    // Create nodes but NO text index
    let n = db.create_node(&["Article"]);
    db.set_node_property(
        n,
        "body",
        Value::String("some body text about rust".into()),
    );

    let session = db.session();
    let result = session.execute(
        "MATCH (doc:Article) WHERE text_score(doc.body, 'rust') > 0.0 RETURN doc.title",
    );

    assert!(
        result.is_err(),
        "Expected error when using text_score without a text index, but got success"
    );
}

// ============================================================================
// Test 5: cosine_similarity without vector index → brute-force fallback
// ============================================================================

#[test]
fn test_vector_without_index_brute_force() {
    let db = GrafeoDB::new_in_memory();
    // Create articles WITHOUT a vector index
    let a1 = db.create_node(&["Article"]);
    db.set_node_property(a1, "title", Value::String("Graph Neural Networks".into()));
    db.set_node_property(
        a1,
        "embedding",
        Value::Vector(vec![0.9f32, 0.1, 0.0].into()),
    );

    let a2 = db.create_node(&["Article"]);
    db.set_node_property(a2, "title", Value::String("Rust Database Internals".into()));
    db.set_node_property(
        a2,
        "embedding",
        Value::Vector(vec![0.1f32, 0.9, 0.0].into()),
    );

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

    // Alice follows Bob; Bob wrote articles 1 and 2.
    // Article 1 embedding [0.9, 0.1, 0.0] is close to query [0.85, 0.15, 0.05].
    // Article 2 embedding [0.1, 0.9, 0.0] is far from query.
    let result = session
        .execute(
            "MATCH (u:User {name: 'Alice'})-[:FOLLOWS]->(friend)-[:WROTE]->(doc:Article) \
             WHERE cosine_similarity(doc.embedding, [0.85, 0.15, 0.05]) > 0.3 \
             RETURN doc.title",
        )
        .expect("graph + vector query should succeed");

    let titles = collect_strings(&result);

    // At a threshold of 0.3, article 1 ([0.9, 0.1, 0.0]) should match
    // (cosine similarity with [0.85, 0.15, 0.05] ≈ 0.998).
    assert!(
        !titles.is_empty(),
        "Expected at least one article from Alice→Bob→Article traversal with similarity > 0.3"
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

    // ORDER BY text_score DESC LIMIT 1 — should rewrite to TextScan(k=1)
    let result = session
        .execute(
            "MATCH (doc:Article) \
             RETURN doc.title, text_score(doc.body, 'attention mechanisms') AS rank \
             ORDER BY rank DESC LIMIT 1",
        )
        .unwrap();

    assert_eq!(result.row_count(), 1, "Top-1 should return exactly 1 row");
}

#[test]
fn test_topk_order_by_vector_similarity() {
    let db = setup_article_db();
    let session = db.session();

    // ORDER BY cosine_similarity DESC LIMIT 2 — should rewrite to VectorScan(k=2)
    let result = session
        .execute(
            "MATCH (doc:Article) \
             RETURN doc.title, cosine_similarity(doc.embedding, [0.85, 0.15, 0.05]) AS sim \
             ORDER BY sim DESC LIMIT 2",
        )
        .unwrap();

    assert_eq!(result.row_count(), 2, "Top-2 should return exactly 2 rows");
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

    let result = session
        .execute(
            "PROFILE MATCH (doc:Article) \
             WHERE cosine_similarity(doc.embedding, [0.85, 0.15, 0.05]) > 0.5 \
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
