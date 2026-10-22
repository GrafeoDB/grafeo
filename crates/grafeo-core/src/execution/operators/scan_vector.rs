//! Vector similarity scan operator.
//!
//! Delegates to the store's [`GraphStoreSearch::vector_search`] implementation,
//! which routes to HNSW when a matching index is available and falls back to
//! brute-force scan otherwise. The operator itself owns only the search
//! parameters and post-filter thresholds.

use super::{Operator, OperatorError, OperatorResult};
use crate::execution::DataChunk;
use crate::graph::GraphStoreSearch;
use crate::index::vector::DistanceMetric;
use grafeo_common::types::{LogicalType, NodeId};
use std::sync::Arc;

/// A scan operator that finds nodes by vector similarity.
///
/// Calls [`GraphStoreSearch::vector_search`] on the first `next()`, caches
/// the results, and streams them as `DataChunk` batches. The store decides
/// between HNSW-accelerated and brute-force execution based on what's
/// registered for (label, property).
///
/// # Output schema
///
/// Returns a DataChunk with two columns:
/// 1. `Node` - the matched node ID
/// 2. `Float64` - the distance score (units depend on `metric`)
///
/// # Example
///
/// ```no_run
/// use grafeo_core::execution::operators::{Operator, VectorScanOperator};
/// use grafeo_core::index::vector::DistanceMetric;
/// use grafeo_core::graph::lpg::LpgStore;
/// use grafeo_core::graph::GraphStoreSearch;
/// use std::sync::Arc;
///
/// # fn example() -> Result<(), grafeo_core::execution::operators::OperatorError> {
/// let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
/// let query = vec![0.1f32, 0.2, 0.3];
/// let mut scan = VectorScanOperator::new(
///     store,
///     Some("Document".to_string()),
///     "embedding".to_string(),
///     query,
///     10,
///     DistanceMetric::Cosine,
/// );
///
/// while let Some(chunk) = scan.next()? {
///     for i in 0..chunk.row_count() {
///         let node_id = chunk.column(0).and_then(|c| c.get_node_id(i));
///         let distance = chunk.column(1).and_then(|c| c.get_float64(i));
///         println!("Node {:?} at distance {:?}", node_id, distance);
///     }
/// }
/// # Ok(())
/// # }
/// ```
pub struct VectorScanOperator {
    store: Arc<dyn GraphStoreSearch>,
    label: Option<String>,
    property: String,
    query: Vec<f32>,
    k: usize,
    metric: DistanceMetric,
    min_similarity: Option<f32>,
    max_distance: Option<f32>,
    results: Vec<(NodeId, f64)>,
    position: usize,
    executed: bool,
    chunk_capacity: usize,
}

impl VectorScanOperator {
    /// Creates a new vector similarity scan.
    ///
    /// `label` scopes the search to nodes carrying that label (use `None` to
    /// scan every node with the named property). The store decides between
    /// HNSW and brute force based on (label, property, metric).
    #[must_use]
    pub fn new(
        store: Arc<dyn GraphStoreSearch>,
        label: Option<String>,
        property: String,
        query: Vec<f32>,
        k: usize,
        metric: DistanceMetric,
    ) -> Self {
        Self {
            store,
            label,
            property,
            query,
            k,
            metric,
            min_similarity: None,
            max_distance: None,
            results: Vec::new(),
            position: 0,
            executed: false,
            chunk_capacity: 2048,
        }
    }

    /// Filters out results with similarity below this threshold.
    ///
    /// Similarity is computed as `1.0 - distance` for cosine metric; the
    /// filter has no effect for other metrics (use `with_max_distance` instead).
    #[must_use]
    pub fn with_min_similarity(mut self, threshold: f32) -> Self {
        self.min_similarity = Some(threshold);
        self
    }

    /// Filters out results whose distance exceeds this threshold.
    #[must_use]
    pub fn with_max_distance(mut self, threshold: f32) -> Self {
        self.max_distance = Some(threshold);
        self
    }

    /// Sets the chunk capacity for output batches. Clamped to at least 1.
    #[must_use]
    pub fn with_chunk_capacity(mut self, capacity: usize) -> Self {
        self.chunk_capacity = capacity.max(1);
        self
    }

    fn execute_search(&mut self) {
        if self.executed {
            return;
        }
        self.executed = true;

        self.results = self.store.vector_search(
            self.label.as_deref(),
            &self.property,
            &self.query,
            self.k,
            self.metric,
        );

        self.apply_filters();
    }

    fn apply_filters(&mut self) {
        if self.min_similarity.is_none() && self.max_distance.is_none() {
            return;
        }

        self.results.retain(|(_, distance)| {
            let passes_similarity = match self.min_similarity {
                Some(threshold) if self.metric == DistanceMetric::Cosine => {
                    let similarity = 1.0 - distance;
                    similarity >= f64::from(threshold)
                }
                Some(_) => true,
                None => true,
            };

            let passes_distance = match self.max_distance {
                Some(threshold) => *distance <= f64::from(threshold),
                None => true,
            };

            passes_similarity && passes_distance
        });
    }
}

impl Operator for VectorScanOperator {
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
            let dist_col = chunk
                .column_mut(1)
                .ok_or_else(|| OperatorError::ColumnNotFound("distance column".into()))?;
            for i in self.position..end {
                dist_col.push_float64(self.results[i].1);
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
        "VectorScan"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

#[cfg(all(test, feature = "lpg", feature = "vector-index"))]
mod tests {
    use super::*;
    use crate::graph::lpg::LpgStore;
    use grafeo_common::types::Value;

    fn store_with_vectors(docs: &[(&str, Vec<f32>)]) -> Arc<dyn GraphStoreSearch> {
        let store = Arc::new(LpgStore::new().unwrap());
        for (property, vector) in docs {
            let node = store.create_node(&["Document"]);
            store.set_node_property(node, property, Value::Vector(vector.clone().into()));
        }
        store
    }

    #[test]
    fn test_vector_scan_brute_force() {
        let store = Arc::new(LpgStore::new().unwrap());

        let n1 = store.create_node(&["Document"]);
        let n2 = store.create_node(&["Document"]);
        let n3 = store.create_node(&["Document"]);

        store.set_node_property(n1, "embedding", Value::Vector(vec![0.1, 0.2, 0.3].into()));
        store.set_node_property(n2, "embedding", Value::Vector(vec![0.5, 0.6, 0.7].into()));
        store.set_node_property(n3, "embedding", Value::Vector(vec![0.9, 0.8, 0.7].into()));

        let query = vec![0.1, 0.2, 0.35];

        let mut scan = VectorScanOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            Some("Document".to_string()),
            "embedding".to_string(),
            query,
            2,
            DistanceMetric::Euclidean,
        );

        let chunk = scan.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 2);

        let first_node = chunk.column(0).unwrap().get_node_id(0);
        assert_eq!(first_node, Some(n1));

        assert!(scan.next().unwrap().is_none());
    }

    #[test]
    fn test_vector_scan_reset() {
        let store = Arc::new(LpgStore::new().unwrap());
        let n1 = store.create_node(&["Doc"]);
        store.set_node_property(n1, "vec", Value::Vector(vec![0.1, 0.2].into()));

        let mut scan = VectorScanOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            None,
            "vec".to_string(),
            vec![0.1, 0.2],
            10,
            DistanceMetric::Cosine,
        );

        let chunk1 = scan.next().unwrap().unwrap();
        assert_eq!(chunk1.row_count(), 1);
        assert!(scan.next().unwrap().is_none());

        scan.reset();
        let chunk2 = scan.next().unwrap().unwrap();
        assert_eq!(chunk2.row_count(), 1);
    }

    #[test]
    fn test_vector_scan_with_max_distance() {
        let store = Arc::new(LpgStore::new().unwrap());
        let n1 = store.create_node(&["Doc"]);
        let _n2 = store.create_node(&["Doc"]);
        store.set_node_property(n1, "vec", Value::Vector(vec![0.1, 0.0].into()));
        store.set_node_property(_n2, "vec", Value::Vector(vec![10.0, 10.0].into()));

        let mut scan = VectorScanOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            Some("Doc".to_string()),
            "vec".to_string(),
            vec![0.0, 0.0],
            10,
            DistanceMetric::Euclidean,
        )
        .with_max_distance(1.0);

        let chunk = scan.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 1);
        assert_eq!(chunk.column(0).unwrap().get_node_id(0), Some(n1));
    }

    #[test]
    fn test_vector_scan_empty_results() {
        let store = Arc::new(LpgStore::new().unwrap());
        store.create_node(&["Doc"]);

        let mut scan = VectorScanOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            None,
            "embedding".to_string(),
            vec![0.1, 0.2],
            10,
            DistanceMetric::Cosine,
        );

        assert!(scan.next().unwrap().is_none());
    }

    #[test]
    fn test_vector_scan_with_min_similarity() {
        let store = Arc::new(LpgStore::new().unwrap());
        let n1 = store.create_node(&["Doc"]);
        let n2 = store.create_node(&["Doc"]);
        store.set_node_property(n1, "vec", Value::Vector(vec![1.0, 0.0].into()));
        store.set_node_property(n2, "vec", Value::Vector(vec![0.707, 0.707].into()));

        let mut scan = VectorScanOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            Some("Doc".to_string()),
            "vec".to_string(),
            vec![0.0, 1.0],
            10,
            DistanceMetric::Cosine,
        )
        .with_min_similarity(0.5);

        let chunk = scan.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 1);
        assert_eq!(chunk.column(0).unwrap().get_node_id(0), Some(n2));
    }

    #[test]
    fn test_vector_scan_with_chunk_capacity() {
        let store = Arc::new(LpgStore::new().unwrap());
        for i in 0..10 {
            let node = store.create_node(&["Doc"]);
            store.set_node_property(node, "vec", Value::Vector(vec![i as f32, 0.0].into()));
        }

        let mut scan = VectorScanOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            Some("Doc".to_string()),
            "vec".to_string(),
            vec![0.0, 0.0],
            10,
            DistanceMetric::Euclidean,
        )
        .with_chunk_capacity(3);

        let chunk1 = scan.next().unwrap().unwrap();
        assert_eq!(chunk1.row_count(), 3);
        let chunk2 = scan.next().unwrap().unwrap();
        assert_eq!(chunk2.row_count(), 3);
        let chunk3 = scan.next().unwrap().unwrap();
        assert_eq!(chunk3.row_count(), 3);
        let chunk4 = scan.next().unwrap().unwrap();
        assert_eq!(chunk4.row_count(), 1);
        assert!(scan.next().unwrap().is_none());
    }

    #[test]
    fn test_vector_scan_name() {
        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
        let scan = VectorScanOperator::new(
            store,
            None,
            "vec".to_string(),
            vec![0.1],
            10,
            DistanceMetric::Cosine,
        );
        assert_eq!(scan.name(), "VectorScan");
    }

    // Suppresses the "unused helper" warning on `store_with_vectors` when
    // tests that use it are selected individually.
    #[test]
    fn _use_helper() {
        let _ = store_with_vectors(&[]);
    }
}
