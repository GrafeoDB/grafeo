//! BM25 text scan operator.
//!
//! Performs full-text search using an inverted index with BM25 scoring.
//! Supports top-k and score-threshold modes.

use super::{Operator, OperatorError, OperatorResult};
use crate::execution::DataChunk;
use crate::graph::traits::GraphStore;
use grafeo_common::types::{LogicalType, NodeId};
use std::sync::Arc;

/// A scan operator that retrieves nodes by BM25 text relevance.
///
/// This operator executes a full-text search against a store's text index
/// and returns results as `(NodeId, Float64)` DataChunk batches, where the
/// second column holds the BM25 score.
///
/// # Modes
///
/// - **Top-k**: return the `k` highest-scoring documents.
/// - **Threshold**: return all documents scoring at or above a threshold.
///
/// # Output Schema
///
/// Returns a DataChunk with two columns:
/// 1. `Node`    — The matched node ID
/// 2. `Float64` — The BM25 relevance score
pub struct TextScanOperator {
    /// The graph store to search.
    store: Arc<dyn GraphStore>,
    /// Label to search within.
    label: String,
    /// Property holding the text to search.
    property: String,
    /// Search query string.
    query: String,
    /// Top-k limit (None = threshold mode).
    k: Option<usize>,
    /// Minimum score threshold (None = top-k mode).
    threshold: Option<f64>,
    /// Cached search results after first execution.
    results: Vec<(NodeId, f64)>,
    /// Current read position in `results`.
    position: usize,
    /// Whether the search has already been executed.
    executed: bool,
    /// Rows per output DataChunk (default 2048).
    chunk_capacity: usize,
}

impl TextScanOperator {
    /// Creates a top-k text scan operator.
    ///
    /// Returns the `k` highest-scoring nodes for the given query.
    #[must_use]
    pub fn top_k(
        store: Arc<dyn GraphStore>,
        label: impl Into<String>,
        property: impl Into<String>,
        query: impl Into<String>,
        k: usize,
    ) -> Self {
        Self {
            store,
            label: label.into(),
            property: property.into(),
            query: query.into(),
            k: Some(k),
            threshold: None,
            results: Vec::new(),
            position: 0,
            executed: false,
            chunk_capacity: 2048,
        }
    }

    /// Creates a threshold-based text scan operator.
    ///
    /// Returns all nodes with a BM25 score ≥ `threshold`.
    #[must_use]
    pub fn with_threshold(
        store: Arc<dyn GraphStore>,
        label: impl Into<String>,
        property: impl Into<String>,
        query: impl Into<String>,
        threshold: f64,
    ) -> Self {
        Self {
            store,
            label: label.into(),
            property: property.into(),
            query: query.into(),
            k: None,
            threshold: Some(threshold),
            results: Vec::new(),
            position: 0,
            executed: false,
            chunk_capacity: 2048,
        }
    }

    /// Sets the chunk capacity (rows per output batch).
    #[must_use]
    pub fn with_chunk_capacity(mut self, capacity: usize) -> Self {
        self.chunk_capacity = capacity;
        self
    }

    /// Executes the search on first call and caches the results.
    fn execute_search(&mut self) {
        if self.executed {
            return;
        }
        self.executed = true;

        self.results = if let Some(k) = self.k {
            self.store
                .text_search(&self.label, &self.property, &self.query, k)
        } else if let Some(threshold) = self.threshold {
            self.store
                .text_search_with_threshold(&self.label, &self.property, &self.query, threshold)
        } else {
            Vec::new()
        };
    }
}

impl Operator for TextScanOperator {
    fn next(&mut self) -> OperatorResult {
        // Lazy execution on first call.
        self.execute_search();

        if self.position >= self.results.len() {
            return Ok(None);
        }

        let schema = [LogicalType::Node, LogicalType::Float64];
        let mut chunk = DataChunk::with_capacity(&schema, self.chunk_capacity);

        let end = (self.position + self.chunk_capacity).min(self.results.len());
        let count = end - self.position;

        // Fill node ID column.
        {
            let node_col = chunk
                .column_mut(0)
                .ok_or_else(|| OperatorError::ColumnNotFound("node column".into()))?;

            for i in self.position..end {
                let (node_id, _) = self.results[i];
                node_col.push_node_id(node_id);
            }
        }

        // Fill BM25 score column.
        {
            let score_col = chunk
                .column_mut(1)
                .ok_or_else(|| OperatorError::ColumnNotFound("score column".into()))?;

            for i in self.position..end {
                let (_, score) = self.results[i];
                score_col.push_float64(score);
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

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

#[cfg(all(test, feature = "text-index", feature = "lpg"))]
mod tests {
    use super::*;
    use crate::graph::lpg::LpgStore;
    use crate::graph::traits::GraphStore;
    use crate::index::text::{BM25Config, InvertedIndex};
    use grafeo_common::types::Value;
    use parking_lot::RwLock;
    use std::sync::Arc;

    fn make_store() -> Arc<LpgStore> {
        let store = Arc::new(LpgStore::new().expect("arena allocation"));

        // Create nodes
        let n1 = store.create_node(&["Doc"]);
        store.set_node_property(n1, "body", Value::String("rust graph database engine".into()));
        let n2 = store.create_node(&["Doc"]);
        store.set_node_property(n2, "body", Value::String("python web framework".into()));
        let n3 = store.create_node(&["Doc"]);
        store.set_node_property(
            n3,
            "body",
            Value::String("rust systems programming language".into()),
        );

        // Build and add text index
        let mut index = InvertedIndex::new(BM25Config::default());
        index.insert(n1, "rust graph database engine");
        index.insert(n2, "python web framework");
        index.insert(n3, "rust systems programming language");
        store.add_text_index("Doc", "body", Arc::new(RwLock::new(index)));

        store
    }

    #[test]
    fn test_text_scan_top_k() {
        let store = make_store();
        let mut scan = TextScanOperator::top_k(
            store.clone() as Arc<dyn GraphStore>,
            "Doc",
            "body",
            "rust",
            2,
        );
        let chunk = scan.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 2);
        let n1 = chunk.column(0).unwrap().get_node_id(0);
        let score1 = chunk.column(1).unwrap().get_float64(0);
        assert!(n1.is_some());
        assert!(score1.unwrap() > 0.0);
        assert!(scan.next().unwrap().is_none());
    }

    #[test]
    fn test_text_scan_with_threshold() {
        let store = make_store();
        let all = store.text_search("Doc", "body", "rust database", 10);
        assert_eq!(all.len(), 2);
        let mid = (all[0].1 + all[1].1) / 2.0;
        let mut scan = TextScanOperator::with_threshold(
            store.clone() as Arc<dyn GraphStore>,
            "Doc",
            "body",
            "rust database",
            mid,
        );
        let chunk = scan.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 1);
        assert_eq!(chunk.column(0).unwrap().get_node_id(0), Some(all[0].0));
        assert!(scan.next().unwrap().is_none());
    }

    #[test]
    fn test_text_scan_no_matches() {
        let store = make_store();
        let mut scan = TextScanOperator::top_k(
            store.clone() as Arc<dyn GraphStore>,
            "Doc",
            "body",
            "nonexistent",
            10,
        );
        assert!(scan.next().unwrap().is_none());
    }

    #[test]
    fn test_text_scan_reset() {
        let store = make_store();
        let mut scan = TextScanOperator::top_k(
            store.clone() as Arc<dyn GraphStore>,
            "Doc",
            "body",
            "rust",
            10,
        );
        let chunk1 = scan.next().unwrap().unwrap();
        assert_eq!(chunk1.row_count(), 2);
        assert!(scan.next().unwrap().is_none());
        scan.reset();
        let chunk2 = scan.next().unwrap().unwrap();
        assert_eq!(chunk2.row_count(), 2);
    }

    #[test]
    fn test_text_scan_name() {
        let store = make_store();
        let scan = TextScanOperator::top_k(
            store.clone() as Arc<dyn GraphStore>,
            "Doc",
            "body",
            "rust",
            10,
        );
        assert_eq!(scan.name(), "TextScan(BM25)");
    }

    #[test]
    fn test_text_scan_chunk_capacity() {
        let store = make_store();
        let mut scan = TextScanOperator::top_k(
            store.clone() as Arc<dyn GraphStore>,
            "Doc",
            "body",
            "rust",
            10,
        )
        .with_chunk_capacity(1);
        let chunk1 = scan.next().unwrap().unwrap();
        assert_eq!(chunk1.row_count(), 1);
        let chunk2 = scan.next().unwrap().unwrap();
        assert_eq!(chunk2.row_count(), 1);
        assert!(scan.next().unwrap().is_none());
    }
}
