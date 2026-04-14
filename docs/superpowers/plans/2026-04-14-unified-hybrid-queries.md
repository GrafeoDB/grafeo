# Unified Hybrid Queries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make graph + vector + text compose in a single Cypher/GQL query with index-accelerated execution.

**Architecture:** Teach the query planner to recognize vector/text function calls in WHERE and ORDER BY, rewrite them into existing VectorScan/new TextScan physical operators when backed by indexes, project scores for downstream reuse, and handle compound predicates by running both indexes and joining results.

**Tech Stack:** Rust, grafeo-core (operators, indexes), grafeo-engine (planner, optimizer), feature flags: `vector-index`, `text-index`, `hybrid-search`

**Spec:** `docs/superpowers/specs/2026-04-14-unified-hybrid-queries-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `crates/grafeo-core/src/index/text/inverted_index.rs` | Modify | Add `score_document()`, `search_with_threshold()` |
| `crates/grafeo-core/src/execution/operators/scan_text.rs` | Create | `TextScanOperator` — BM25 index scan, outputs `(NodeId, Float64)` |
| `crates/grafeo-core/src/execution/operators/mod.rs` | Modify | Export `TextScanOperator` |
| `crates/grafeo-core/src/execution/operators/filter.rs` | Modify | Add `eval_text_fn` to function cascade |
| `crates/grafeo-engine/src/query/plan.rs` | Modify | Add `TextScan(TextScanOp)` to `LogicalOperator`, define `TextScanOp` struct |
| `crates/grafeo-engine/src/query/planner/lpg/filter.rs` | Modify | Add `try_plan_filter_with_vector_index()`, `try_plan_filter_with_text_index()`, `try_plan_filter_compound_hybrid()` |
| `crates/grafeo-engine/src/query/planner/lpg/mod.rs` | Modify | Add `plan_text_scan()` dispatch, top-K recognition in `plan_operator()` |
| `crates/grafeo-engine/src/query/planner/lpg/project.rs` | Modify | Score projection + expression rewriting in `plan_sort()` |
| `crates/grafeo-engine/src/query/optimizer/cardinality.rs` | Modify | Add `estimate_text_scan()` |
| `crates/grafeo-engine/src/query/optimizer/cost.rs` | Modify | Add TextScan cost formula |
| `crates/grafeo-engine/tests/hybrid_query.rs` | Create | End-to-end integration tests |

---

### Task 1: InvertedIndex API — `score_document()` and `search_with_threshold()`

**Files:**
- Modify: `crates/grafeo-core/src/index/text/inverted_index.rs`

These two methods are the foundation. `score_document()` enables per-row text scoring in the filter fallback path. `search_with_threshold()` enables TextScan when the predicate is `text_score(...) > threshold`.

- [ ] **Step 1: Write failing test for `score_document()`**

Add to the existing `#[cfg(test)] mod tests` block at line 300:

```rust
#[test]
fn test_score_document_matches_search() {
    let mut index = InvertedIndex::new(BM25Config::default());
    index.insert(NodeId::new(1), "the quick brown fox jumps over the lazy dog");
    index.insert(NodeId::new(2), "a fast red car drives on the highway");
    index.insert(NodeId::new(3), "the brown dog sleeps all day");

    // score_document should return the same score as search for each doc
    let search_results = index.search("brown dog", 10);
    for (node_id, expected_score) in &search_results {
        let single_score = index.score_document(*node_id, "brown dog");
        assert!(
            (single_score - expected_score).abs() < 1e-10,
            "score_document({:?}) = {}, search = {}",
            node_id, single_score, expected_score
        );
    }

    // Document not matching any terms should score 0
    assert_eq!(index.score_document(NodeId::new(2), "brown dog"), 0.0);

    // Non-existent document should score 0
    assert_eq!(index.score_document(NodeId::new(999), "brown dog"), 0.0);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p grafeo-core --features text-index -- test_score_document_matches_search`
Expected: FAIL — `score_document` method doesn't exist.

- [ ] **Step 3: Implement `score_document()`**

Add after the `search()` method (after line 193), before `contains()`:

```rust
/// Scores a single document against a query using BM25.
///
/// Looks up each query term in its posting list, finds the entry for `id`,
/// and computes BM25 with the corpus statistics in this index.
/// Returns 0.0 if the document has no matching terms or doesn't exist.
///
/// O(query_terms) per call — does not iterate the full corpus.
pub fn score_document(&self, id: NodeId, query: &str) -> f64 {
    let query_tokens = self.tokenizer.tokenize(query);
    if query_tokens.is_empty() || self.doc_lengths.is_empty() {
        return 0.0;
    }

    let Some(&doc_len) = self.doc_lengths.get(&id) else {
        return 0.0;
    };

    let n = self.doc_lengths.len() as f64;
    let avg_dl = self.total_length as f64 / n;
    let dl = f64::from(doc_len);
    let mut score = 0.0;

    for token in &query_tokens {
        let Some(posting_list) = self.postings.get(token.as_str()) else {
            continue;
        };

        let df = posting_list.postings.len() as f64;
        let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();

        // Find this document's term frequency in the posting list
        let tf = posting_list
            .postings
            .iter()
            .find(|p| p.node_id == id)
            .map_or(0.0, |p| f64::from(p.term_freq));

        if tf > 0.0 {
            let tf_component = (tf * (self.config.k1 + 1.0))
                / (tf + self.config.k1 * (1.0 - self.config.b + self.config.b * dl / avg_dl));
            score += idf * tf_component;
        }
    }

    score
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p grafeo-core --features text-index -- test_score_document_matches_search`
Expected: PASS

- [ ] **Step 5: Write failing test for `search_with_threshold()`**

```rust
#[test]
fn test_search_with_threshold() {
    let mut index = InvertedIndex::new(BM25Config::default());
    index.insert(NodeId::new(1), "rust graph database engine");
    index.insert(NodeId::new(2), "rust programming language systems web server framework database engine query optimizer");
    index.insert(NodeId::new(3), "python web framework");

    // Get scores from search to calibrate threshold
    let all_results = index.search("rust database", 10);
    assert_eq!(all_results.len(), 2); // nodes 1 and 2

    // Node 1 scores higher (shorter doc, both terms)
    let high_score = all_results[0].1;
    let low_score = all_results[1].1;
    assert!(high_score > low_score);

    // Threshold between the two scores: only node 1 should pass
    let mid_threshold = (high_score + low_score) / 2.0;
    let filtered = index.search_with_threshold("rust database", mid_threshold);
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].0, NodeId::new(1));
    assert!(filtered[0].1 >= mid_threshold);

    // Threshold of 0 returns all matching docs
    let all = index.search_with_threshold("rust database", 0.0);
    assert_eq!(all.len(), 2);

    // Very high threshold returns nothing
    let none = index.search_with_threshold("rust database", 999.0);
    assert!(none.is_empty());

    // Empty query returns nothing
    let empty = index.search_with_threshold("", 0.0);
    assert!(empty.is_empty());
}
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cargo test -p grafeo-core --features text-index -- test_search_with_threshold`
Expected: FAIL — `search_with_threshold` method doesn't exist.

- [ ] **Step 7: Implement `search_with_threshold()`**

Add after `score_document()`:

```rust
/// Returns all documents scoring above `threshold` for the given query.
///
/// Unlike `search()` which returns top-k, this returns every document
/// whose BM25 score exceeds the threshold. Results are sorted by score
/// descending.
pub fn search_with_threshold(&self, query: &str, threshold: f64) -> Vec<(NodeId, f64)> {
    let query_tokens = self.tokenizer.tokenize(query);
    if query_tokens.is_empty() || self.doc_lengths.is_empty() {
        return Vec::new();
    }

    let n = self.doc_lengths.len() as f64;
    let avg_dl = self.total_length as f64 / n;

    let mut scores: HashMap<NodeId, f64> = HashMap::new();

    for token in &query_tokens {
        let Some(posting_list) = self.postings.get(token.as_str()) else {
            continue;
        };

        let df = posting_list.postings.len() as f64;
        let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();

        for posting in &posting_list.postings {
            let tf = f64::from(posting.term_freq);
            let dl = f64::from(self.doc_lengths.get(&posting.node_id).copied().unwrap_or(0));

            let tf_component = (tf * (self.config.k1 + 1.0))
                / (tf + self.config.k1 * (1.0 - self.config.b + self.config.b * dl / avg_dl));

            *scores.entry(posting.node_id).or_insert(0.0) += idf * tf_component;
        }
    }

    let mut results: Vec<(NodeId, f64)> = scores
        .into_iter()
        .filter(|(_, score)| *score >= threshold)
        .collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}
```

- [ ] **Step 8: Run test to verify it passes**

Run: `cargo test -p grafeo-core --features text-index -- test_search_with_threshold`
Expected: PASS

- [ ] **Step 9: Run all existing inverted index tests to check for regressions**

Run: `cargo test -p grafeo-core --features text-index -- inverted_index::tests`
Expected: All existing tests PASS.

- [ ] **Step 10: Commit**

```bash
git add crates/grafeo-core/src/index/text/inverted_index.rs
git commit -m "feat(text): add score_document() and search_with_threshold() to InvertedIndex

score_document() scores a single document in O(query_terms) for per-row
evaluation. search_with_threshold() returns all documents above a BM25
threshold for TextScan WHERE predicates."
```

---

### Task 2: TextScanOperator

**Files:**
- Create: `crates/grafeo-core/src/execution/operators/scan_text.rs`
- Modify: `crates/grafeo-core/src/execution/operators/mod.rs`

Modeled directly on `VectorScanOperator` (`scan_vector.rs`). Same output schema: `(NodeId, Float64)`.

- [ ] **Step 1: Write failing test**

Create `crates/grafeo-core/src/execution/operators/scan_text.rs` with the test module first:

```rust
//! Full-text search scan operator.
//!
//! Performs BM25-scored search using an inverted index, returning
//! matching nodes with their relevance scores.

use super::{Operator, OperatorError, OperatorResult};
use crate::execution::DataChunk;
use crate::index::text::InvertedIndex;
use grafeo_common::types::{LogicalType, NodeId, Value};
use std::sync::{Arc, RwLock};

// Struct and impl will go here

#[cfg(all(test, feature = "text-index"))]
mod tests {
    use super::*;
    use crate::index::text::BM25Config;

    fn make_index() -> Arc<RwLock<InvertedIndex>> {
        let mut index = InvertedIndex::new(BM25Config::default());
        index.insert(NodeId::new(1), "rust graph database engine");
        index.insert(NodeId::new(2), "python web framework");
        index.insert(NodeId::new(3), "rust systems programming language");
        Arc::new(RwLock::new(index))
    }

    #[test]
    fn test_text_scan_top_k() {
        let index = make_index();
        let mut scan = TextScanOperator::top_k(Arc::clone(&index), "rust", 2);

        let chunk = scan.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 2);

        // Both nodes 1 and 3 mention "rust"
        let n1 = chunk.column(0).unwrap().get_node_id(0);
        let score1 = chunk.column(1).unwrap().get_float64(0);
        assert!(n1.is_some());
        assert!(score1.unwrap() > 0.0);

        // Should be exhausted
        assert!(scan.next().unwrap().is_none());
    }

    #[test]
    fn test_text_scan_with_threshold() {
        let index = make_index();
        // Get all results first to find a threshold
        let all = index.read().unwrap().search("rust database", 10);
        assert_eq!(all.len(), 2); // nodes 1 and 3

        // Use a threshold that only the top result passes
        let mid = (all[0].1 + all[1].1) / 2.0;
        let mut scan = TextScanOperator::with_threshold(Arc::clone(&index), "rust database", mid);

        let chunk = scan.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 1);
        assert_eq!(chunk.column(0).unwrap().get_node_id(0), Some(all[0].0));

        assert!(scan.next().unwrap().is_none());
    }

    #[test]
    fn test_text_scan_no_matches() {
        let index = make_index();
        let mut scan = TextScanOperator::top_k(Arc::clone(&index), "nonexistent", 10);
        assert!(scan.next().unwrap().is_none());
    }

    #[test]
    fn test_text_scan_reset() {
        let index = make_index();
        let mut scan = TextScanOperator::top_k(Arc::clone(&index), "rust", 10);

        let chunk1 = scan.next().unwrap().unwrap();
        assert_eq!(chunk1.row_count(), 2);
        assert!(scan.next().unwrap().is_none());

        scan.reset();
        let chunk2 = scan.next().unwrap().unwrap();
        assert_eq!(chunk2.row_count(), 2);
    }

    #[test]
    fn test_text_scan_name() {
        let index = make_index();
        let scan = TextScanOperator::top_k(Arc::clone(&index), "rust", 10);
        assert_eq!(scan.name(), "TextScan(BM25)");
    }

    #[test]
    fn test_text_scan_chunk_capacity() {
        let index = make_index();
        let mut scan = TextScanOperator::top_k(Arc::clone(&index), "rust", 10)
            .with_chunk_capacity(1);

        let chunk1 = scan.next().unwrap().unwrap();
        assert_eq!(chunk1.row_count(), 1);
        let chunk2 = scan.next().unwrap().unwrap();
        assert_eq!(chunk2.row_count(), 1);
        assert!(scan.next().unwrap().is_none());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p grafeo-core --features text-index,lpg -- scan_text::tests`
Expected: FAIL — `TextScanOperator` not defined.

- [ ] **Step 3: Implement `TextScanOperator`**

Add above the test module in `scan_text.rs`:

```rust
/// A scan operator that finds nodes by full-text search with BM25 scoring.
///
/// Operates in two modes:
/// - **Top-K**: Returns the k highest-scoring documents (for ORDER BY + LIMIT)
/// - **Threshold**: Returns all documents scoring above a threshold (for WHERE)
///
/// # Output Schema
///
/// Returns a DataChunk with two columns:
/// 1. `Node` — The matched node ID
/// 2. `Float64` — The BM25 relevance score
pub struct TextScanOperator {
    /// The text index to search.
    index: Arc<RwLock<InvertedIndex>>,
    /// The search query string.
    query: String,
    /// Top-k mode (None = threshold mode).
    k: Option<usize>,
    /// Threshold mode (None = top-k mode).
    threshold: Option<f64>,
    /// Cached results from search.
    results: Vec<(NodeId, f64)>,
    /// Current position in results.
    position: usize,
    /// Whether search has been executed.
    executed: bool,
    /// Chunk capacity.
    chunk_capacity: usize,
}

impl TextScanOperator {
    /// Creates a text scan that returns the top-k results by BM25 score.
    #[must_use]
    pub fn top_k(index: Arc<RwLock<InvertedIndex>>, query: impl Into<String>, k: usize) -> Self {
        Self {
            index,
            query: query.into(),
            k: Some(k),
            threshold: None,
            results: Vec::new(),
            position: 0,
            executed: false,
            chunk_capacity: 2048,
        }
    }

    /// Creates a text scan that returns all documents above a score threshold.
    #[must_use]
    pub fn with_threshold(
        index: Arc<RwLock<InvertedIndex>>,
        query: impl Into<String>,
        threshold: f64,
    ) -> Self {
        Self {
            index,
            query: query.into(),
            k: None,
            threshold: Some(threshold),
            results: Vec::new(),
            position: 0,
            executed: false,
            chunk_capacity: 2048,
        }
    }

    /// Sets the chunk capacity for output batches.
    #[must_use]
    pub fn with_chunk_capacity(mut self, capacity: usize) -> Self {
        self.chunk_capacity = capacity;
        self
    }

    /// Executes the text search (lazily on first next() call).
    fn execute_search(&mut self) {
        if self.executed {
            return;
        }
        self.executed = true;

        let index = self.index.read().unwrap();
        self.results = if let Some(k) = self.k {
            index.search(&self.query, k)
        } else if let Some(threshold) = self.threshold {
            index.search_with_threshold(&self.query, threshold)
        } else {
            Vec::new()
        };
    }
}

impl Operator for TextScanOperator {
    fn next(&mut self) -> OperatorResult {
        self.execute_search();

        if self.position >= self.results.len() {
            return Ok(None);
        }

        let schema = [LogicalType::Node, LogicalType::Float64];
        let mut chunk = DataChunk::with_capacity(&schema, self.chunk_capacity);

        let end = (self.position + self.chunk_capacity).min(self.results.len());
        let count = end - self.position;

        {
            let node_col = chunk
                .column_mut(0)
                .ok_or_else(|| OperatorError::ColumnNotFound("node column".into()))?;
            for i in self.position..end {
                node_col.push_node_id(self.results[i].0);
            }
        }

        {
            let score_col = chunk
                .column_mut(1)
                .ok_or_else(|| OperatorError::ColumnNotFound("score column".into()))?;
            for i in self.position..end {
                score_col.push_float64(self.results[i].1);
            }
        }

        chunk.set_count(count);
        self.position = end;

        Ok(Some(chunk))
    }

    fn reset(&mut self) {
        self.position = 0;
        self.results.clear();
        self.executed = false;
    }

    fn name(&self) -> &'static str {
        "TextScan(BM25)"
    }
}
```

- [ ] **Step 4: Add module and export to `mod.rs`**

In `crates/grafeo-core/src/execution/operators/mod.rs`, add after line 43 (`mod scan_vector;`):

```rust
#[cfg(feature = "text-index")]
mod scan_text;
```

And add the export after line 98 (`pub use scan_vector::VectorScanOperator;`):

```rust
#[cfg(feature = "text-index")]
pub use scan_text::TextScanOperator;
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p grafeo-core --features text-index,lpg -- scan_text::tests`
Expected: All 6 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/grafeo-core/src/execution/operators/scan_text.rs crates/grafeo-core/src/execution/operators/mod.rs
git commit -m "feat(operators): add TextScanOperator for BM25 index scans

Parallel to VectorScanOperator. Supports top-k and threshold modes.
Outputs (NodeId, Float64) DataChunks with BM25 relevance scores."
```

---

### Task 3: eval_text_fn — per-row text scoring in filter evaluation

**Files:**
- Modify: `crates/grafeo-core/src/execution/operators/filter.rs`

This adds `text_score()` and `text_match()` as functions callable in WHERE clauses. When no planner pushdown fires (e.g., graph traversal narrows candidates), this is the execution path. It calls `InvertedIndex.score_document()` for each candidate row.

- [ ] **Step 1: Write failing test**

Add to the existing test module in `filter.rs` (after line 3951). The filter operator's `eval_function` already dispatches through `eval_vector_fn` etc. We need the store to have a text index. For unit testing, we test `eval_text_fn` directly. But the method is private on `ExpressionPredicate`, so we test through a full filter evaluation with a mock:

```rust
#[cfg(all(test, feature = "text-index", feature = "lpg"))]
mod text_fn_tests {
    use super::*;
    use crate::graph::lpg::LpgStore;

    #[test]
    fn test_text_score_function() {
        let store = Arc::new(LpgStore::new().unwrap());

        // Create nodes with string properties
        let n1 = store.create_node(&["Article"]);
        store.set_node_property(n1, "body", Value::String("rust graph database engine".into()));
        let n2 = store.create_node(&["Article"]);
        store.set_node_property(n2, "body", Value::String("python web framework".into()));

        // Create and populate text index
        store.create_text_index("Article", "body").unwrap();

        // Build a filter expression: text_score(n.body, "rust database") > 0
        let filter_expr = FilterExpression::BinaryOp {
            left: Box::new(FilterExpression::FunctionCall {
                name: "text_score".to_string(),
                args: vec![
                    FilterExpression::PropertyAccess {
                        object: Box::new(FilterExpression::Variable("doc".to_string())),
                        property: "body".to_string(),
                    },
                    FilterExpression::Literal(Value::String("rust database".into())),
                ],
            }),
            op: BinaryFilterOp::Gt,
            right: Box::new(FilterExpression::Literal(Value::Float64(0.0))),
        };

        // Create a chunk with node IDs
        let schema = [LogicalType::Node];
        let mut chunk = DataChunk::with_capacity(&schema, 2);
        chunk.column_mut(0).unwrap().push_node_id(n1);
        chunk.column_mut(0).unwrap().push_node_id(n2);
        chunk.set_count(2);

        let variable_columns: HashMap<String, usize> = [("doc".to_string(), 0)].into_iter().collect();

        let predicate = ExpressionPredicate::new(
            filter_expr,
            variable_columns,
            Arc::clone(&store) as Arc<dyn GraphStore>,
        );

        // n1 has "rust" and "database" -> should match
        assert!(predicate.evaluate(&chunk, 0));
        // n2 has neither -> should not match
        assert!(!predicate.evaluate(&chunk, 1));
    }

    #[test]
    fn test_text_match_function() {
        let store = Arc::new(LpgStore::new().unwrap());

        let n1 = store.create_node(&["Article"]);
        store.set_node_property(n1, "body", Value::String("rust graph database engine".into()));
        let n2 = store.create_node(&["Article"]);
        store.set_node_property(n2, "body", Value::String("python web framework".into()));

        store.create_text_index("Article", "body").unwrap();

        // text_match(doc.body, "rust") -> should return Bool
        let filter_expr = FilterExpression::FunctionCall {
            name: "text_match".to_string(),
            args: vec![
                FilterExpression::PropertyAccess {
                    object: Box::new(FilterExpression::Variable("doc".to_string())),
                    property: "body".to_string(),
                },
                FilterExpression::Literal(Value::String("rust".into())),
            ],
        };

        let schema = [LogicalType::Node];
        let mut chunk = DataChunk::with_capacity(&schema, 2);
        chunk.column_mut(0).unwrap().push_node_id(n1);
        chunk.column_mut(0).unwrap().push_node_id(n2);
        chunk.set_count(2);

        let variable_columns: HashMap<String, usize> = [("doc".to_string(), 0)].into_iter().collect();

        let predicate = ExpressionPredicate::new(
            filter_expr,
            variable_columns,
            Arc::clone(&store) as Arc<dyn GraphStore>,
        );

        // text_match returns Bool, so we check directly
        // n1 has "rust" -> true, n2 doesn't -> false
        assert!(predicate.evaluate(&chunk, 0));
        assert!(!predicate.evaluate(&chunk, 1));
    }
}
```

**Note:** The exact `FilterExpression` variant names (`BinaryOp`, `PropertyAccess`, etc.) must match the actual enum in `filter.rs`. Read the `FilterExpression` enum definition (around line 459) before writing the test — adapt variant names and field names to match exactly. The test above uses placeholder names; the implementer must verify against the actual enum.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p grafeo-core --features text-index,lpg -- text_fn_tests`
Expected: FAIL — `text_score` and `text_match` not recognized by `eval_function`.

- [ ] **Step 3: Implement `eval_text_fn()`**

Add the method to the `ExpressionPredicate` impl block, after `eval_vector_fn` (after line 3331):

```rust
#[cfg(feature = "text-index")]
fn eval_text_fn(
    &self,
    name: &str,
    args: &[FilterExpression],
    chunk: &DataChunk,
    row: usize,
) -> Option<Value> {
    match name {
        "text_score" | "text_match" => {
            if args.len() != 2 {
                return None;
            }

            // First arg: property access -> need node_id and property name
            // Extract the node_id from the chunk and the property name from the expression
            let (node_id, property_name) = self.extract_node_and_property(&args[0], chunk, row)?;

            // Second arg: query string
            let query_val = self.eval_expr(&args[1], chunk, row)?;
            let query_str = match &query_val {
                Value::String(s) => s.as_ref(),
                _ => return None,
            };

            // Get the node's label(s) to look up the text index
            let labels = self.store.get_node_labels(node_id)?;
            let label = labels.first()?;

            // Look up the text index for (label, property)
            let index = self.store.get_text_index(label, &property_name)?;
            let index_guard = index.read().ok()?;
            let score = index_guard.score_document(node_id, query_str);

            if name == "text_match" {
                Some(Value::Bool(score > 0.0))
            } else {
                Some(Value::Float64(score))
            }
        }
        _ => None,
    }
}
```

**Note:** The `extract_node_and_property` helper may not exist. The implementer should check how `eval_vector_fn` extracts property values from `FilterExpression` args and follow the same pattern. The key steps are: (1) evaluate the property access expression to get the Value, (2) get the node_id from the chunk at the variable's column index, (3) get the property name from the expression AST. Adapt to the actual `FilterExpression` structure.

- [ ] **Step 4: Add `eval_text_fn` to the function cascade**

In the `eval_function` method (line 1558-1576), add after the `eval_vector_fn` call:

```rust
.or_else(|| self.eval_vector_fn(name, args, chunk, row))
#[cfg(feature = "text-index")]
.or_else(|| self.eval_text_fn(name, args, chunk, row))
.or_else(|| self.eval_session_fn(name, args, chunk, row))
```

**Note:** Inline `#[cfg]` on method chain calls may not compile. If not, use a conditional block:

```rust
let result = self.eval_vector_fn(name, args, chunk, row);
#[cfg(feature = "text-index")]
let result = result.or_else(|| self.eval_text_fn(name, args, chunk, row));
let result = result.or_else(|| self.eval_session_fn(name, args, chunk, row));
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p grafeo-core --features text-index,lpg -- text_fn_tests`
Expected: PASS

- [ ] **Step 6: Run all filter tests for regressions**

Run: `cargo test -p grafeo-core --features text-index,lpg -- filter::tests`
Expected: All existing tests PASS.

- [ ] **Step 7: Commit**

```bash
git add crates/grafeo-core/src/execution/operators/filter.rs
git commit -m "feat(filter): add text_score() and text_match() function evaluation

Per-row BM25 scoring via InvertedIndex.score_document(). Used as the
fallback path when planner pushdown doesn't fire (e.g., graph traversal
already narrowed candidates)."
```

---

### Task 4: TextScanOp logical operator + planner dispatch

**Files:**
- Modify: `crates/grafeo-engine/src/query/plan.rs` (add `TextScanOp` struct and `LogicalOperator::TextScan` variant)
- Modify: `crates/grafeo-engine/src/query/planner/lpg/mod.rs` (add `plan_text_scan()` method, dispatch in `plan_operator()`)

- [ ] **Step 1: Add `TextScanOp` to plan.rs**

After `VectorJoinOp` definition (around line 1951), add:

```rust
/// Text search scan using BM25 inverted index.
///
/// Produced by the planner when it recognizes text_score/text_match
/// predicates in WHERE or text_score in ORDER BY + LIMIT.
#[derive(Debug, Clone)]
pub struct TextScanOp {
    /// The variable name bound to matched nodes.
    pub variable: String,
    /// The property containing indexed text.
    pub property: String,
    /// The label to look up the text index.
    pub label: String,
    /// The search query (literal or parameter).
    pub query: LogicalExpression,
    /// Top-k limit (for ORDER BY + LIMIT mode).
    pub k: Option<usize>,
    /// Score threshold (for WHERE mode).
    pub threshold: Option<f64>,
    /// Synthetic column name for the projected score.
    pub score_column: Option<String>,
}
```

Add the variant to `LogicalOperator` enum, after `VectorJoin` (line 272):

```rust
/// Scan using full-text search with BM25 scoring.
TextScan(TextScanOp),
```

- [ ] **Step 2: Add `plan_text_scan()` to the planner**

In `crates/grafeo-engine/src/query/planner/lpg/mod.rs`, add a method to the `Planner` impl:

```rust
/// Plans a TextScan operator into a physical TextScanOperator.
#[cfg(feature = "text-index")]
fn plan_text_scan(
    &self,
    scan: &TextScanOp,
) -> Result<(Box<dyn Operator>, Vec<String>)> {
    use grafeo_core::execution::operators::TextScanOperator;

    // Resolve the query expression to a string value
    let query_string = match &scan.query {
        LogicalExpression::Literal(Value::String(s)) => s.to_string(),
        LogicalExpression::Parameter(name) => {
            // Parameter resolution would happen here at execution time
            // For now, return error if unresolved
            return Err(Error::Internal(format!(
                "TextScan query parameter ${} not resolved", name
            )));
        }
        _ => return Err(Error::Internal(
            "TextScan query must be a string literal or parameter".to_string(),
        )),
    };

    // Look up the text index
    let index = self.store.get_text_index(&scan.label, &scan.property)
        .ok_or_else(|| Error::Internal(format!(
            "No text index on ({}, {})", scan.label, scan.property
        )))?;

    // Create the physical operator
    let operator: Box<dyn Operator> = if let Some(k) = scan.k {
        Box::new(TextScanOperator::top_k(index, &query_string, k))
    } else if let Some(threshold) = scan.threshold {
        Box::new(TextScanOperator::with_threshold(index, &query_string, threshold))
    } else {
        Box::new(TextScanOperator::top_k(index, &query_string, 100))
    };

    // Output columns: variable (node), score column if named
    let mut columns = vec![scan.variable.clone()];
    if let Some(ref score_col) = scan.score_column {
        columns.push(score_col.clone());
    }

    Ok((operator, columns))
}
```

- [ ] **Step 3: Add dispatch in `plan_operator()`**

In `plan_operator()` (line 659-661), replace the VectorScan error and add TextScan handling:

```rust
#[cfg(feature = "vector-index")]
LogicalOperator::VectorScan(scan) => self.plan_vector_scan(scan),
#[cfg(not(feature = "vector-index"))]
LogicalOperator::VectorScan(_) => Err(Error::Internal(
    "VectorScan requires vector-index feature".to_string(),
)),
#[cfg(feature = "text-index")]
LogicalOperator::TextScan(scan) => self.plan_text_scan(scan),
#[cfg(not(feature = "text-index"))]
LogicalOperator::TextScan(_) => Err(Error::Internal(
    "TextScan requires text-index feature".to_string(),
)),
```

**Note:** The VectorScan dispatch may already have a feature-gated handler elsewhere. Check if `plan_vector_scan` exists in another file before adding it. If the current code only has the error stub (lines 659-661), replace it.

- [ ] **Step 4: Verify compilation**

Run: `cargo check -p grafeo-engine --features text-index,vector-index,lpg,gql`
Expected: Compiles without errors.

- [ ] **Step 5: Commit**

```bash
git add crates/grafeo-engine/src/query/plan.rs crates/grafeo-engine/src/query/planner/lpg/mod.rs
git commit -m "feat(planner): add TextScanOp logical operator and plan_text_scan dispatch

TextScanOp holds the query, label, property, optional k/threshold.
The planner resolves it to a physical TextScanOperator, looking up
the text index from the store."
```

---

### Task 5: Vector predicate pushdown in filter.rs

**Files:**
- Modify: `crates/grafeo-engine/src/query/planner/lpg/filter.rs`

This is the core planner rewrite. Pattern-match vector function calls in WHERE predicates, check for HNSW index, produce VectorScanOperator.

- [ ] **Step 1: Add `try_plan_filter_with_vector_index()` method**

Add to the `Planner` impl block in `filter.rs`, after `try_plan_filter_with_range_index()` (after line 1122):

```rust
/// Attempts to push a vector similarity predicate into an HNSW index scan.
///
/// Recognizes patterns like:
/// - `cosine_similarity(n.prop, $vec) > 0.7` → VectorScan with min_similarity
/// - `euclidean_distance(n.prop, $vec) < 2.0` → VectorScan with max_distance
///
/// Only fires when the input is a NodeScan (full label scan). If the input
/// is already narrowed by graph traversal, per-row eval is faster.
#[cfg(feature = "vector-index")]
pub(super) fn try_plan_filter_with_vector_index(
    &self,
    filter: &FilterOp,
) -> Result<Option<(Box<dyn Operator>, Vec<String>)>> {
    // Only push down when input is a full label scan
    let LogicalOperator::NodeScan(scan) = filter.input.as_ref() else {
        return Ok(None);
    };
    let Some(ref label) = scan.label else {
        return Ok(None); // No label → can't look up index
    };

    // Extract a vector predicate from the filter expression
    let Some(extracted) = self.extract_vector_predicate(&filter.predicate) else {
        return Ok(None);
    };

    // Check that a vector index exists for this (label, property)
    if self.store.get_vector_index(label, &extracted.property).is_none() {
        return Ok(None); // No index → fall through to brute-force eval
    }

    // Build VectorScanOp
    let vector_scan = VectorScanOp {
        variable: scan.variable.clone(),
        index_name: Some(format!("{}:{}", label, extracted.property)),
        property: extracted.property.clone(),
        label: Some(label.clone()),
        query_vector: extracted.query_vector.clone(),
        k: 0, // threshold mode, not top-k
        metric: Some(extracted.metric),
        min_similarity: extracted.min_similarity,
        max_distance: extracted.max_distance,
        input: None,
    };

    // Plan it through the existing VectorScan path
    let (scan_op, scan_columns) = self.plan_operator(
        &LogicalOperator::VectorScan(vector_scan)
    )?;

    // If there are remaining predicates (AND with non-vector conditions),
    // wrap in a FilterOperator
    if let Some(remaining) = &extracted.remaining {
        let variable_columns: HashMap<String, usize> = scan_columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();
        let filter_expr = self.convert_expression(remaining)?;
        let predicate = ExpressionPredicate::new(
            filter_expr,
            variable_columns,
            Arc::clone(&self.store) as Arc<dyn GraphStore>,
        )
        .with_transaction_context(self.viewing_epoch, self.transaction_id)
        .with_session_context(self.session_context.clone());
        let filter_op = Box::new(FilterOperator::new(scan_op, Box::new(predicate)));
        Ok(Some((filter_op, scan_columns)))
    } else {
        Ok(Some((scan_op, scan_columns)))
    }
}
```

- [ ] **Step 2: Implement the predicate extraction helper**

```rust
/// Result of extracting a vector predicate from a filter expression.
#[cfg(feature = "vector-index")]
struct ExtractedVectorPredicate {
    property: String,
    variable: String,
    query_vector: LogicalExpression,
    metric: VectorMetric,
    min_similarity: Option<f32>,
    max_distance: Option<f32>,
    remaining: Option<LogicalExpression>,
}

#[cfg(feature = "vector-index")]
impl super::Planner {
    /// Extracts a vector similarity/distance predicate from a logical expression.
    ///
    /// Matches patterns like:
    /// - `cosine_similarity(n.prop, expr) > threshold`
    /// - `euclidean_distance(n.prop, expr) < threshold`
    ///
    /// For AND expressions, extracts the vector predicate and returns the
    /// non-vector parts as `remaining`.
    fn extract_vector_predicate(
        &self,
        expr: &LogicalExpression,
    ) -> Option<ExtractedVectorPredicate> {
        match expr {
            // Direct: vector_fn(n.prop, vec) > threshold
            LogicalExpression::Binary { left, op, right } => {
                // Try left side as function call
                if let LogicalExpression::FunctionCall { name, args, .. } = left.as_ref() {
                    if let Some(extracted) = self.try_extract_vector_fn(name, args, op, right) {
                        return Some(extracted);
                    }
                }
                // AND: recurse into both sides
                if *op == BinaryOp::And {
                    // Try left as vector, right as remaining
                    if let Some(mut extracted) = self.extract_vector_predicate(left) {
                        extracted.remaining = Some(match extracted.remaining {
                            Some(prev) => LogicalExpression::Binary {
                                left: Box::new(prev),
                                op: BinaryOp::And,
                                right: right.clone(),
                            },
                            None => *right.clone(),
                        });
                        return Some(extracted);
                    }
                    // Try right as vector, left as remaining
                    if let Some(mut extracted) = self.extract_vector_predicate(right) {
                        extracted.remaining = Some(match extracted.remaining {
                            Some(prev) => LogicalExpression::Binary {
                                left: left.clone(),
                                op: BinaryOp::And,
                                right: Box::new(prev),
                            },
                            None => *left.clone(),
                        });
                        return Some(extracted);
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Tries to extract a vector function call + threshold comparison.
    fn try_extract_vector_fn(
        &self,
        name: &str,
        args: &[LogicalExpression],
        op: &BinaryOp,
        threshold: &LogicalExpression,
    ) -> Option<ExtractedVectorPredicate> {
        if args.len() != 2 {
            return None;
        }

        // Determine metric and comparison direction
        let (metric, is_similarity) = match name {
            "cosine_similarity" => (VectorMetric::Cosine, true),
            "dot_product" => (VectorMetric::DotProduct, true),
            "euclidean_distance" => (VectorMetric::Euclidean, false),
            "manhattan_distance" => (VectorMetric::Manhattan, false),
            _ => return None,
        };

        // Similarity uses >, distance uses <
        let valid_op = if is_similarity {
            matches!(op, BinaryOp::Gt | BinaryOp::Ge)
        } else {
            matches!(op, BinaryOp::Lt | BinaryOp::Le)
        };
        if !valid_op {
            return None; // Inverted comparison (e.g., cosine < 0.3) — no pushdown
        }

        // One arg must be a property access, the other a vector literal/parameter
        let (variable, property, query_vector) =
            if let LogicalExpression::Property { variable, property } = &args[0] {
                (variable.clone(), property.clone(), args[1].clone())
            } else if let LogicalExpression::Property { variable, property } = &args[1] {
                (variable.clone(), property.clone(), args[0].clone())
            } else {
                return None; // Both args are property accesses (node-to-node) — no pushdown
            };

        // Extract threshold value
        let threshold_val = match threshold {
            LogicalExpression::Literal(Value::Float64(v)) => *v as f32,
            LogicalExpression::Literal(Value::Int64(v)) => *v as f32,
            _ => return None, // Non-literal threshold — no pushdown
        };

        let (min_similarity, max_distance) = if is_similarity {
            (Some(threshold_val), None)
        } else {
            (None, Some(threshold_val))
        };

        Some(ExtractedVectorPredicate {
            property,
            variable,
            query_vector,
            metric,
            min_similarity,
            max_distance,
            remaining: None,
        })
    }
}
```

- [ ] **Step 3: Hook into `plan_filter()`**

In `plan_filter()`, after the range index attempt (line 73-75), add:

```rust
// Try to use vector index for similarity/distance predicates
#[cfg(feature = "vector-index")]
if let Some(result) = self.try_plan_filter_with_vector_index(filter)? {
    return Ok(result);
}
```

- [ ] **Step 4: Verify compilation**

Run: `cargo check -p grafeo-engine --features vector-index,lpg,gql`
Expected: Compiles. (There will be import warnings for `VectorScanOp`, `VectorMetric` — add them to the import block at the top of `filter.rs`.)

- [ ] **Step 5: Commit**

```bash
git add crates/grafeo-engine/src/query/planner/lpg/filter.rs
git commit -m "feat(planner): vector predicate pushdown into HNSW index scans

Recognizes cosine_similarity/euclidean_distance/dot_product/manhattan_distance
in WHERE clauses, checks for HNSW index, rewrites to VectorScan operator.
Only fires on NodeScan input (bare label scan). Compound AND predicates
extract the vector part and leave remaining as residual filter."
```

---

### Task 6: Text predicate pushdown in filter.rs

**Files:**
- Modify: `crates/grafeo-engine/src/query/planner/lpg/filter.rs`

Same pattern as Task 5 but for `text_score()` and `text_match()`. Errors if no text index exists (per design decision D2).

- [ ] **Step 1: Add `try_plan_filter_with_text_index()` method**

Follow the same pattern as `try_plan_filter_with_vector_index()`. Key differences:
- Matches `text_score(n.prop, "query") > threshold` and `text_match(n.prop, "query")`
- `text_match` is treated as `text_score > 0.0`
- Errors (not falls through) if no text index exists
- Produces `TextScanOp` instead of `VectorScanOp`

```rust
#[cfg(feature = "text-index")]
pub(super) fn try_plan_filter_with_text_index(
    &self,
    filter: &FilterOp,
) -> Result<Option<(Box<dyn Operator>, Vec<String>)>> {
    let LogicalOperator::NodeScan(scan) = filter.input.as_ref() else {
        return Ok(None);
    };
    let Some(ref label) = scan.label else {
        return Ok(None);
    };

    let Some(extracted) = self.extract_text_predicate(&filter.predicate) else {
        return Ok(None);
    };

    // Text functions REQUIRE a text index (design decision D2)
    if self.store.get_text_index(label, &extracted.property).is_none() {
        return Err(Error::Query(format!(
            "text_score/text_match requires a text index on ({}, {}). \
             Create one with: db.create_text_index(\"{}\", \"{}\")",
            label, extracted.property, label, extracted.property
        )));
    }

    let text_scan = TextScanOp {
        variable: scan.variable.clone(),
        property: extracted.property.clone(),
        label: label.clone(),
        query: extracted.query_expr.clone(),
        k: None,
        threshold: Some(extracted.threshold),
        score_column: None,
    };

    let (scan_op, scan_columns) = self.plan_operator(
        &LogicalOperator::TextScan(text_scan)
    )?;

    // Handle remaining predicates
    if let Some(remaining) = &extracted.remaining {
        let variable_columns: HashMap<String, usize> = scan_columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();
        let filter_expr = self.convert_expression(remaining)?;
        let predicate = ExpressionPredicate::new(
            filter_expr,
            variable_columns,
            Arc::clone(&self.store) as Arc<dyn GraphStore>,
        )
        .with_transaction_context(self.viewing_epoch, self.transaction_id)
        .with_session_context(self.session_context.clone());
        let filter_op = Box::new(FilterOperator::new(scan_op, Box::new(predicate)));
        Ok(Some((filter_op, scan_columns)))
    } else {
        Ok(Some((scan_op, scan_columns)))
    }
}
```

- [ ] **Step 2: Implement text predicate extraction helper**

```rust
#[cfg(feature = "text-index")]
struct ExtractedTextPredicate {
    property: String,
    query_expr: LogicalExpression,
    threshold: f64,
    remaining: Option<LogicalExpression>,
}

#[cfg(feature = "text-index")]
impl super::Planner {
    fn extract_text_predicate(
        &self,
        expr: &LogicalExpression,
    ) -> Option<ExtractedTextPredicate> {
        match expr {
            LogicalExpression::Binary { left, op, right } => {
                // text_score(n.prop, "query") > threshold
                if let LogicalExpression::FunctionCall { name, args, .. } = left.as_ref() {
                    if name == "text_score" && matches!(op, BinaryOp::Gt | BinaryOp::Ge) {
                        if let Some(extracted) = self.try_extract_text_fn(args, right) {
                            return Some(extracted);
                        }
                    }
                }
                // AND: recurse (same pattern as vector extraction)
                if *op == BinaryOp::And {
                    if let Some(mut extracted) = self.extract_text_predicate(left) {
                        extracted.remaining = Some(match extracted.remaining {
                            Some(prev) => LogicalExpression::Binary {
                                left: Box::new(prev),
                                op: BinaryOp::And,
                                right: right.clone(),
                            },
                            None => *right.clone(),
                        });
                        return Some(extracted);
                    }
                    if let Some(mut extracted) = self.extract_text_predicate(right) {
                        extracted.remaining = Some(match extracted.remaining {
                            Some(prev) => LogicalExpression::Binary {
                                left: left.clone(),
                                op: BinaryOp::And,
                                right: Box::new(prev),
                            },
                            None => *left.clone(),
                        });
                        return Some(extracted);
                    }
                }
                None
            }
            // text_match(n.prop, "query") — standalone boolean
            LogicalExpression::FunctionCall { name, args, .. } if name == "text_match" => {
                self.try_extract_text_fn(args, &LogicalExpression::Literal(Value::Float64(0.0)))
            }
            _ => None,
        }
    }

    fn try_extract_text_fn(
        &self,
        args: &[LogicalExpression],
        threshold_expr: &LogicalExpression,
    ) -> Option<ExtractedTextPredicate> {
        if args.len() != 2 {
            return None;
        }

        let LogicalExpression::Property { property, .. } = &args[0] else {
            return None;
        };

        let threshold = match threshold_expr {
            LogicalExpression::Literal(Value::Float64(v)) => *v,
            LogicalExpression::Literal(Value::Int64(v)) => *v as f64,
            _ => return None,
        };

        Some(ExtractedTextPredicate {
            property: property.clone(),
            query_expr: args[1].clone(),
            threshold,
            remaining: None,
        })
    }
}
```

- [ ] **Step 3: Hook into `plan_filter()`**

After the vector index attempt:

```rust
#[cfg(feature = "text-index")]
if let Some(result) = self.try_plan_filter_with_text_index(filter)? {
    return Ok(result);
}
```

- [ ] **Step 4: Verify compilation**

Run: `cargo check -p grafeo-engine --features text-index,vector-index,lpg,gql`
Expected: Compiles.

- [ ] **Step 5: Commit**

```bash
git add crates/grafeo-engine/src/query/planner/lpg/filter.rs
git commit -m "feat(planner): text predicate pushdown into BM25 index scans

Recognizes text_score()/text_match() in WHERE clauses, checks for text index
(errors if missing per D2), rewrites to TextScan operator. Only fires on
NodeScan input."
```

---

### Task 7: Compound hybrid scan (AND/OR with both vector and text)

**Files:**
- Modify: `crates/grafeo-engine/src/query/planner/lpg/filter.rs`

When both vector and text predicates exist in the same WHERE clause.

- [ ] **Step 1: Implement `try_plan_filter_compound_hybrid()`**

After both individual pushdown methods:

```rust
/// Handles compound predicates with both vector AND/OR text.
///
/// Runs both index scans independently and hash-joins the results:
/// - AND → intersect on NodeId
/// - OR → union on NodeId
#[cfg(all(feature = "vector-index", feature = "text-index"))]
pub(super) fn try_plan_filter_compound_hybrid(
    &self,
    filter: &FilterOp,
) -> Result<Option<(Box<dyn Operator>, Vec<String>)>> {
    let LogicalOperator::NodeScan(scan) = filter.input.as_ref() else {
        return Ok(None);
    };
    let Some(ref label) = scan.label else {
        return Ok(None);
    };

    // Try to extract both vector and text predicates
    let vector = self.extract_vector_predicate(&filter.predicate);
    let text = self.extract_text_predicate(&filter.predicate);

    // Only proceed if BOTH are present
    let (vector_pred, text_pred) = match (vector, text) {
        (Some(v), Some(t)) => (v, t),
        _ => return Ok(None),
    };

    // Check both indexes exist
    if self.store.get_vector_index(label, &vector_pred.property).is_none() {
        return Ok(None);
    }
    if self.store.get_text_index(label, &text_pred.property).is_none() {
        return Err(Error::Query(format!(
            "text_score/text_match requires a text index on ({}, {})",
            label, text_pred.property
        )));
    }

    // Build and plan both scan operators
    let vector_scan_op = LogicalOperator::VectorScan(VectorScanOp {
        variable: scan.variable.clone(),
        index_name: Some(format!("{}:{}", label, vector_pred.property)),
        property: vector_pred.property.clone(),
        label: Some(label.clone()),
        query_vector: vector_pred.query_vector.clone(),
        k: 0,
        metric: Some(vector_pred.metric),
        min_similarity: vector_pred.min_similarity,
        max_distance: vector_pred.max_distance,
        input: None,
    });

    let text_scan_op = LogicalOperator::TextScan(TextScanOp {
        variable: scan.variable.clone(),
        property: text_pred.property.clone(),
        label: label.clone(),
        query: text_pred.query_expr.clone(),
        k: None,
        threshold: Some(text_pred.threshold),
        score_column: None,
    });

    let (left_op, left_cols) = self.plan_operator(&vector_scan_op)?;
    let (right_op, right_cols) = self.plan_operator(&text_scan_op)?;

    // Hash-join on NodeId (column 0 in both)
    // Determine join type based on AND vs OR
    let is_or = self.is_or_compound(&filter.predicate);
    let join_type = if is_or {
        PhysicalJoinType::Full // Union semantics
    } else {
        PhysicalJoinType::Inner // Intersect semantics
    };

    let join_condition = JoinCondition::Equality(EqualityCondition {
        left_column: 0,
        right_column: 0,
    });

    let mut columns = left_cols;
    // Add right score column (right col 1 = text score)
    if right_cols.len() > 1 {
        columns.push(right_cols[1].clone());
    }

    let join_op = Box::new(HashJoinOperator::new(
        left_op,
        right_op,
        join_condition,
        join_type,
    ));

    // Apply any remaining scalar predicates
    // (predicates that are neither vector nor text)
    let scalar_remaining = self.extract_scalar_remaining(&filter.predicate);
    if let Some(remaining) = scalar_remaining {
        let variable_columns: HashMap<String, usize> = columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();
        let filter_expr = self.convert_expression(&remaining)?;
        let predicate = ExpressionPredicate::new(
            filter_expr,
            variable_columns,
            Arc::clone(&self.store) as Arc<dyn GraphStore>,
        )
        .with_transaction_context(self.viewing_epoch, self.transaction_id)
        .with_session_context(self.session_context.clone());
        Ok(Some((Box::new(FilterOperator::new(join_op, Box::new(predicate))), columns)))
    } else {
        Ok(Some((join_op, columns)))
    }
}
```

**Note:** The implementer needs to:
1. Verify `HashJoinOperator::new` signature and `PhysicalJoinType` variants exist as named. Check `join.rs` for the actual API.
2. Implement `is_or_compound()` — check if the top-level predicate is an OR containing both vector and text parts.
3. Implement `extract_scalar_remaining()` — extract parts of the predicate that are neither vector nor text functions.

- [ ] **Step 2: Hook into `plan_filter()` — compound check runs before individual checks**

Reorder the hooks in `plan_filter()`:

```rust
// Try compound hybrid first (both vector AND text in same predicate)
#[cfg(all(feature = "vector-index", feature = "text-index"))]
if let Some(result) = self.try_plan_filter_compound_hybrid(filter)? {
    return Ok(result);
}

// Try individual vector pushdown
#[cfg(feature = "vector-index")]
if let Some(result) = self.try_plan_filter_with_vector_index(filter)? {
    return Ok(result);
}

// Try individual text pushdown
#[cfg(feature = "text-index")]
if let Some(result) = self.try_plan_filter_with_text_index(filter)? {
    return Ok(result);
}
```

- [ ] **Step 3: Verify compilation**

Run: `cargo check -p grafeo-engine --features hybrid-search,lpg,gql`
Expected: Compiles.

- [ ] **Step 4: Commit**

```bash
git add crates/grafeo-engine/src/query/planner/lpg/filter.rs
git commit -m "feat(planner): compound hybrid scan for vector AND/OR text predicates

When both vector and text predicates are present, runs both index scans
and hash-joins the results (intersect for AND, union for OR). Both
scores are projected as columns for downstream use."
```

---

### Task 8: Top-K recognition (ORDER BY + LIMIT → index scan)

**Files:**
- Modify: `crates/grafeo-engine/src/query/planner/lpg/project.rs`

Recognize `Sort(vector_fn/text_fn) → Limit(k) → NodeScan` and rewrite to VectorScan(k)/TextScan(k), eliminating Sort and Limit.

- [ ] **Step 1: Add top-K detection in `plan_sort()`**

At the beginning of `plan_sort()` (line 505), before the existing logic, add an early-return check:

```rust
pub(super) fn plan_sort(&self, sort: &SortOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
    // Top-K optimization: Sort(vector_fn) → Limit → NodeScan
    // can be rewritten to a single VectorScan(k) or TextScan(k)
    if let Some(result) = self.try_topk_rewrite(sort)? {
        return Ok(result);
    }

    // ... existing plan_sort logic continues
```

- [ ] **Step 2: Implement `try_topk_rewrite()`**

Add to the `Planner` impl (in `project.rs` or a new helper file):

```rust
/// Attempts to rewrite Sort+Limit on a vector/text function into
/// a direct index scan that returns results in order.
fn try_topk_rewrite(
    &self,
    sort: &SortOp,
) -> Result<Option<(Box<dyn Operator>, Vec<String>)>> {
    // Must have exactly one sort key
    if sort.keys.len() != 1 {
        return Ok(None);
    }
    let sort_key = &sort.keys[0];

    // Input must be Limit → NodeScan (or Limit → Return → NodeScan)
    // Walk through to find the Limit and the underlying scan
    let (limit_count, inner_op) = match sort.input.as_ref() {
        LogicalOperator::Limit(limit) => {
            (limit.count.value(), limit.input.as_ref())
        }
        _ => return Ok(None),
    };
    let Some(k) = limit_count else {
        return Ok(None);
    };

    // Inner should be a NodeScan (possibly wrapped in Return/Project)
    let node_scan = self.find_node_scan(inner_op);
    let Some((scan_var, scan_label)) = node_scan else {
        return Ok(None);
    };
    let Some(ref label) = scan_label else {
        return Ok(None);
    };

    // Sort key must be a vector or text function call
    match &sort_key.expression {
        LogicalExpression::FunctionCall { name, args, .. } => {
            match name.as_str() {
                #[cfg(feature = "vector-index")]
                "cosine_similarity" | "euclidean_distance" | "dot_product" | "manhattan_distance" => {
                    self.try_vector_topk(name, args, k as usize, &scan_var, label, sort_key)
                }
                #[cfg(feature = "text-index")]
                "text_score" => {
                    self.try_text_topk(args, k as usize, &scan_var, label)
                }
                _ => Ok(None),
            }
        }
        _ => Ok(None),
    }
}

/// Walks through Return/Project operators to find the underlying NodeScan.
fn find_node_scan(&self, op: &LogicalOperator) -> Option<(String, Option<String>)> {
    match op {
        LogicalOperator::NodeScan(scan) => Some((scan.variable.clone(), scan.label.clone())),
        LogicalOperator::Return(ret) => self.find_node_scan(&ret.input),
        LogicalOperator::Project(proj) => self.find_node_scan(&proj.input),
        _ => None,
    }
}
```

The `try_vector_topk` and `try_text_topk` methods construct the respective scan operators with `k` set and no threshold, following the same extraction logic from Tasks 5/6 but applied to sort key expressions instead of WHERE predicates.

- [ ] **Step 3: Verify compilation**

Run: `cargo check -p grafeo-engine --features hybrid-search,lpg,gql`
Expected: Compiles.

- [ ] **Step 4: Commit**

```bash
git add crates/grafeo-engine/src/query/planner/lpg/project.rs
git commit -m "feat(planner): top-K recognition rewrites Sort+Limit to index scan

Detects ORDER BY cosine_similarity/text_score + LIMIT k above a NodeScan
and rewrites to VectorScan(k)/TextScan(k), eliminating both Sort and Limit
operators. HNSW and BM25 already return results in score order."
```

---

### Task 9: Score projection + expression rewriting

**Files:**
- Modify: `crates/grafeo-engine/src/query/planner/lpg/filter.rs` (assign score column names during pushdown)
- Modify: `crates/grafeo-engine/src/query/planner/lpg/project.rs` (rewrite matching expressions in RETURN/ORDER BY)

When VectorScan/TextScan produces a score column, downstream expressions that call the same function with the same arguments should read the projected column instead of recomputing.

- [ ] **Step 1: Assign score column names during pushdown**

In `try_plan_filter_with_vector_index()` (Task 5), after creating the VectorScanOp, set the score column:

```rust
// The VectorScan output is (NodeId, score). Name the score column.
let score_column = format!("_vscore_{}", scan.variable);
```

Add this `score_column` to the returned columns list so downstream operators know about it.

Same for `try_plan_filter_with_text_index()`:

```rust
let score_column = format!("_tscore_{}", scan.variable);
```

- [ ] **Step 2: Rewrite matching expressions in project/sort planning**

In `plan_sort()` and `plan_project()` (project.rs), before converting expressions to physical form, check if any expression matches a vector/text function whose score is already projected:

```rust
/// Checks if a logical expression matches a projected score column.
/// If so, returns the column name to reference instead of recomputing.
fn find_projected_score(
    &self,
    expr: &LogicalExpression,
    columns: &[String],
) -> Option<String> {
    if let LogicalExpression::FunctionCall { name, args, .. } = expr {
        let is_vector_fn = matches!(
            name.as_str(),
            "cosine_similarity" | "euclidean_distance" | "dot_product" | "manhattan_distance"
        );
        let is_text_fn = name == "text_score";

        if is_vector_fn || is_text_fn {
            // Check if a matching _vscore_* or _tscore_* column exists
            let prefix = if is_vector_fn { "_vscore_" } else { "_tscore_" };
            if let Some(LogicalExpression::Property { variable, .. }) = args.first() {
                let score_col = format!("{}{}", prefix, variable);
                if columns.contains(&score_col) {
                    return Some(score_col);
                }
            }
        }
    }
    None
}
```

Use this in expression conversion: if `find_projected_score` returns a column name, emit a column reference instead of a function call.

- [ ] **Step 3: Verify compilation and run existing tests**

Run: `cargo test -p grafeo-engine --features hybrid-search,lpg,gql`
Expected: All existing tests PASS.

- [ ] **Step 4: Commit**

```bash
git add crates/grafeo-engine/src/query/planner/lpg/filter.rs crates/grafeo-engine/src/query/planner/lpg/project.rs
git commit -m "feat(planner): score projection eliminates redundant vector/text computation

When VectorScan or TextScan produces a score column, downstream RETURN
and ORDER BY expressions that call the same function are rewritten to
reference the projected column. Avoids reloading embeddings and
recomputing distances."
```

---

### Task 10: Cost model + cardinality estimation

**Files:**
- Modify: `crates/grafeo-engine/src/query/optimizer/cardinality.rs`
- Modify: `crates/grafeo-engine/src/query/optimizer/cost.rs`

- [ ] **Step 1: Add `estimate_text_scan()` to cardinality.rs**

Find the existing `estimate_vector_scan()` method (around line 940) and add a parallel method after it:

```rust
fn estimate_text_scan(&self, scan: &TextScanOp) -> f64 {
    if let Some(k) = scan.k {
        // Top-k mode: at most k results
        return k as f64;
    }
    // Threshold mode: estimate 10% of indexed documents match
    let default_selectivity = 0.1;
    // Try to get index size from table stats
    let base = self.table_stats
        .get(&scan.label)
        .map_or(1000.0, |s| s.row_count as f64);
    (base * default_selectivity).max(1.0)
}
```

Wire it into the main `estimate()` dispatch method, matching `LogicalOperator::TextScan(scan) => self.estimate_text_scan(scan)`.

- [ ] **Step 2: Add TextScan cost formula to cost.rs**

Find the VectorScan cost estimation (around line 468) and add after it:

```rust
LogicalOperator::TextScan(scan) => {
    let cardinality = self.card_estimator.estimate(op);
    let index_size = cardinality * 10.0; // Approximate corpus size
    // BM25 scoring is ~5x a simple tuple comparison
    let cpu = index_size * self.cpu_tuple_cost * 5.0;
    let io = 0.0; // In-memory index
    Cost { cpu, io, memory: cardinality * self.avg_tuple_size, network: 0.0 }
}
```

- [ ] **Step 3: Verify compilation**

Run: `cargo check -p grafeo-engine --features hybrid-search,lpg,gql`
Expected: Compiles.

- [ ] **Step 4: Commit**

```bash
git add crates/grafeo-engine/src/query/optimizer/cardinality.rs crates/grafeo-engine/src/query/optimizer/cost.rs
git commit -m "feat(optimizer): add TextScan cardinality estimation and cost formula

estimate_text_scan returns k for top-k queries, 10% selectivity for
threshold. Cost uses 5x cpu_tuple_cost for BM25 scoring overhead."
```

---

### Task 11: End-to-end integration tests

**Files:**
- Create: `crates/grafeo-engine/tests/hybrid_query.rs`

These tests run actual Cypher/GQL queries through the full pipeline and verify correct results with index acceleration.

- [ ] **Step 1: Create integration test file**

```rust
//! Integration tests for unified hybrid queries (graph + vector + text).
//!
//! Tests the full pipeline: Cypher/GQL parsing → planning → pushdown → execution.

#![cfg(all(feature = "hybrid-search", feature = "gql"))]

use grafeo_engine::GrafeoDB;
use grafeo_common::types::Value;

fn setup_article_db() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // Create articles with embeddings and text
    let a1 = session.create_node_with_props(
        &["Article"],
        [
            ("title", Value::String("Graph Neural Networks".into())),
            ("body", Value::String("attention mechanisms in graph neural networks for node classification".into())),
            ("embedding", Value::Vector(vec![0.9, 0.1, 0.0].into())),
        ],
    );
    let a2 = session.create_node_with_props(
        &["Article"],
        [
            ("title", Value::String("Rust Database Internals".into())),
            ("body", Value::String("building a database engine in rust with MVCC transactions".into())),
            ("embedding", Value::Vector(vec![0.1, 0.9, 0.0].into())),
        ],
    );
    let a3 = session.create_node_with_props(
        &["Article"],
        [
            ("title", Value::String("Transformer Architectures".into())),
            ("body", Value::String("attention mechanisms and transformer models for natural language".into())),
            ("embedding", Value::Vector(vec![0.8, 0.2, 0.1].into())),
        ],
    );

    // Create user with follows/wrote relationships
    let user = session.create_node_with_props(
        &["User"],
        [("name", Value::String("Alice".into()))],
    );
    let friend = session.create_node_with_props(
        &["User"],
        [("name", Value::String("Bob".into()))],
    );
    session.create_edge(user, friend, "FOLLOWS");
    session.create_edge(friend, a1, "WROTE");
    session.create_edge(friend, a2, "WROTE");

    // Create indexes
    db.create_vector_index("Article", "embedding", Some(3), Some("cosine"), None, None, None)
        .expect("create vector index");
    db.create_text_index("Article", "body").expect("create text index");

    db
}

#[test]
fn test_vector_where_with_pushdown() {
    let db = setup_article_db();
    let session = db.session();

    // This should use VectorScan (bare label scan with vector predicate)
    let result = session.execute(
        "MATCH (doc:Article) \
         WHERE cosine_similarity(doc.embedding, [0.85, 0.15, 0.05]) > 0.5 \
         RETURN doc.title"
    ).unwrap();

    // Articles 1 and 3 have embeddings similar to [0.85, 0.15, 0.05]
    assert!(result.row_count() >= 1);
    let titles: Vec<String> = result.rows().iter()
        .filter_map(|r| r[0].as_string().map(|s| s.to_string()))
        .collect();
    assert!(titles.contains(&"Graph Neural Networks".to_string()));
}

#[test]
fn test_text_score_where() {
    let db = setup_article_db();
    let session = db.session();

    let result = session.execute(
        "MATCH (doc:Article) \
         WHERE text_score(doc.body, 'attention mechanisms') > 0.0 \
         RETURN doc.title, text_score(doc.body, 'attention mechanisms') AS score"
    ).unwrap();

    // Articles 1 and 3 mention "attention mechanisms"
    assert_eq!(result.row_count(), 2);
}

#[test]
fn test_text_match_where() {
    let db = setup_article_db();
    let session = db.session();

    let result = session.execute(
        "MATCH (doc:Article) \
         WHERE text_match(doc.body, 'rust database') \
         RETURN doc.title"
    ).unwrap();

    assert_eq!(result.row_count(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Rust Database Internals".into()));
}

#[test]
fn test_topk_order_by_vector() {
    let db = setup_article_db();
    let session = db.session();

    // ORDER BY + LIMIT should use VectorScan top-K (no full scan)
    let result = session.execute(
        "MATCH (doc:Article) \
         RETURN doc.title, cosine_similarity(doc.embedding, [0.85, 0.15, 0.05]) AS sim \
         ORDER BY sim DESC LIMIT 2"
    ).unwrap();

    assert_eq!(result.row_count(), 2);
    // First result should be the most similar article
}

#[test]
fn test_topk_order_by_text() {
    let db = setup_article_db();
    let session = db.session();

    let result = session.execute(
        "MATCH (doc:Article) \
         RETURN doc.title, text_score(doc.body, 'attention mechanisms') AS rank \
         ORDER BY rank DESC LIMIT 1"
    ).unwrap();

    assert_eq!(result.row_count(), 1);
}

#[test]
fn test_compound_vector_and_text() {
    let db = setup_article_db();
    let session = db.session();

    // Both vector similarity AND text match
    let result = session.execute(
        "MATCH (doc:Article) \
         WHERE cosine_similarity(doc.embedding, [0.85, 0.15, 0.05]) > 0.3 \
           AND text_match(doc.body, 'attention mechanisms') \
         RETURN doc.title"
    ).unwrap();

    // Only Article 1 and 3 mention "attention" AND are similar to the query vector
    // Article 2 (rust database) doesn't mention attention
    assert!(result.row_count() >= 1);
}

#[test]
fn test_graph_plus_vector_per_row_eval() {
    let db = setup_article_db();
    let session = db.session();

    // Graph traversal narrows candidates → per-row eval, not pushdown
    let result = session.execute(
        "MATCH (u:User {name: 'Alice'})-[:FOLLOWS]->(friend)-[:WROTE]->(doc:Article) \
         WHERE cosine_similarity(doc.embedding, [0.85, 0.15, 0.05]) > 0.3 \
         RETURN doc.title"
    ).unwrap();

    // Alice follows Bob, Bob wrote articles 1 and 2
    // Only article 1 is similar to the query vector
    assert!(result.row_count() >= 1);
}

#[test]
fn test_text_score_without_index_errors() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session.create_node_with_props(
        &["Article"],
        [("body", Value::String("hello world".into()))],
    );

    // No text index created → should error
    let result = session.execute(
        "MATCH (doc:Article) \
         WHERE text_score(doc.body, 'hello') > 0.0 \
         RETURN doc.title"
    );

    assert!(result.is_err());
}

#[test]
fn test_vector_without_index_brute_force() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session.create_node_with_props(
        &["Article"],
        [("embedding", Value::Vector(vec![1.0, 0.0, 0.0].into()))],
    );

    // No vector index → should still work (brute-force eval)
    let result = session.execute(
        "MATCH (doc:Article) \
         WHERE cosine_similarity(doc.embedding, [1.0, 0.0, 0.0]) > 0.9 \
         RETURN doc"
    ).unwrap();

    assert_eq!(result.row_count(), 1);
}

#[test]
fn test_score_projection_no_double_compute() {
    let db = setup_article_db();
    let session = db.session();

    // Same function in WHERE and RETURN — score should be computed once
    let result = session.execute(
        "MATCH (doc:Article) \
         WHERE cosine_similarity(doc.embedding, [0.85, 0.15, 0.05]) > 0.3 \
         RETURN doc.title, cosine_similarity(doc.embedding, [0.85, 0.15, 0.05]) AS sim"
    ).unwrap();

    // Verify scores are present and valid
    for row in result.rows() {
        let score = &row[1];
        assert!(matches!(score, Value::Float64(s) if *s > 0.3));
    }
}
```

**Note:** The exact API for `session.execute()`, `session.create_node_with_props()`, `db.create_vector_index()`, and result inspection depends on the actual Grafeo public API. The implementer must adapt to match the real API by checking existing integration tests in `crates/grafeo-engine/tests/query_correctness.rs` and `crates/grafeo-engine/tests/search_operations.rs`.

- [ ] **Step 2: Run integration tests**

Run: `cargo test -p grafeo-engine --features hybrid-search,gql --test hybrid_query`
Expected: All tests PASS. If any fail, debug by checking EXPLAIN output and verifying the planner produces the expected operators.

- [ ] **Step 3: Run the full test suite for regressions**

Run: `cargo test -p grafeo-engine --features full`
Run: `cargo test -p grafeo-core --features full`
Expected: No regressions in existing tests.

- [ ] **Step 4: Commit**

```bash
git add crates/grafeo-engine/tests/hybrid_query.rs
git commit -m "test(hybrid): end-to-end integration tests for unified hybrid queries

Tests vector pushdown, text pushdown, top-K recognition, compound AND,
graph+vector per-row eval, score projection, error on missing text index,
and brute-force fallback without vector index."
```

---

## Dependency Graph

```
Task 1 (InvertedIndex API)
  ├── Task 2 (TextScanOperator)
  │     └── Task 4 (TextScanOp + planner dispatch)
  │           ├── Task 6 (text pushdown)
  │           └── Task 8 (top-K)
  └── Task 3 (eval_text_fn)
        └── Task 6 (text pushdown)

Task 5 (vector pushdown) ← independent, can run in parallel with 2-4

Task 7 (compound hybrid) ← depends on 5 + 6
Task 9 (score projection) ← depends on 5 + 6 + 8
Task 10 (cost model) ← depends on 4
Task 11 (integration tests) ← depends on everything
```

## Parallel execution opportunities

Tasks 1-4 and Task 5 are independent branches. An agent could work on Task 5 (vector pushdown) while another works on Tasks 1-4 (text pipeline). They converge at Task 7 (compound hybrid).
