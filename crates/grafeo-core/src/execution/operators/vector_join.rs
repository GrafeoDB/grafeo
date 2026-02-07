//! Vector join operator for combining graph patterns with vector similarity.
//!
//! This operator performs vector-based joins between graph traversal results
//! and vector similarity search. It enables hybrid queries that combine
//! graph structure with semantic similarity.
//!
//! # Use Cases
//!
//! 1. **Graph + Vector Filtering**: Find graph neighbors similar to a query
//! 2. **Aggregated Embeddings**: Use AVG(embeddings) to find similar entities
//! 3. **Entity-to-Entity Similarity**: Join nodes based on embedding similarity
//!
//! # Output Schema
//!
//! The output includes all columns from the left input plus:
//! - Right entity column (NodeId)
//! - Distance/similarity score column (Float64)

use super::{Operator, OperatorError, OperatorResult};
use crate::execution::DataChunk;
use crate::graph::lpg::LpgStore;
use crate::index::vector::{DistanceMetric, brute_force_knn};
use grafeo_common::types::{LogicalType, NodeId, PropertyKey, Value};
use std::sync::Arc;

#[cfg(feature = "vector-index")]
use crate::index::vector::HnswIndex;

/// Vector join operator for hybrid graph + vector queries.
///
/// Takes entities from the left input and finds similar entities on the right
/// based on vector embeddings. This enables queries like:
///
/// ```gql
/// MATCH (u:User)-[:LIKED]->(liked:Movie)
/// WITH avg(liked.embedding) AS taste
/// VECTOR JOIN (m:Movie) ON m.embedding
/// WHERE cosine_similarity(m.embedding, taste) > 0.7
/// RETURN m.title
/// ```
///
/// # Output Schema
///
/// Output columns: [left columns..., right_node: Node, score: Float64]
pub struct VectorJoinOperator {
    /// Left input operator providing entities.
    left: Box<dyn Operator>,
    /// The store to search vectors from.
    store: Arc<LpgStore>,
    /// The HNSW index for right side (None = brute-force).
    #[cfg(feature = "vector-index")]
    index: Option<Arc<HnswIndex>>,
    /// Property containing left-side vectors (for entity-to-entity similarity).
    /// If None, uses `query_vector` directly.
    left_property: Option<String>,
    /// Column index in left input for node IDs (to fetch properties).
    left_node_column: Option<usize>,
    /// Static query vector (used if left_property is None).
    query_vector: Option<Vec<f32>>,
    /// Property containing right-side vector embeddings.
    right_property: String,
    /// Label filter for right-side entities.
    right_label: Option<String>,
    /// Number of nearest neighbors per left-side row.
    k: usize,
    /// Distance metric.
    metric: DistanceMetric,
    /// Minimum similarity threshold (cosine only).
    min_similarity: Option<f32>,
    /// Maximum distance threshold.
    max_distance: Option<f32>,
    /// Search ef parameter for HNSW.
    ef: usize,
    /// Current left chunk being processed.
    current_left_chunk: Option<DataChunk>,
    /// Current row in left chunk.
    current_left_row: usize,
    /// Cached results for current left row: (right_node, distance).
    current_results: Vec<(NodeId, f32)>,
    /// Current position in results.
    current_result_position: usize,
    /// Output chunk capacity.
    chunk_capacity: usize,
    /// Left exhausted flag.
    left_exhausted: bool,
    /// Uses index flag for name().
    uses_index: bool,
}

impl VectorJoinOperator {
    /// Creates a vector join with a static query vector.
    ///
    /// Use this when the query vector is known upfront (e.g., a parameter).
    ///
    /// # Arguments
    ///
    /// * `left` - Left input operator
    /// * `store` - LPG store for vector property lookup
    /// * `query_vector` - The query vector for similarity search
    /// * `right_property` - Property containing right-side embeddings
    /// * `k` - Number of nearest neighbors per left row
    /// * `metric` - Distance metric
    #[must_use]
    pub fn with_static_query(
        left: Box<dyn Operator>,
        store: Arc<LpgStore>,
        query_vector: Vec<f32>,
        right_property: impl Into<String>,
        k: usize,
        metric: DistanceMetric,
    ) -> Self {
        Self {
            left,
            store,
            #[cfg(feature = "vector-index")]
            index: None,
            left_property: None,
            left_node_column: None,
            query_vector: Some(query_vector),
            right_property: right_property.into(),
            right_label: None,
            k,
            metric,
            min_similarity: None,
            max_distance: None,
            ef: 64,
            current_left_chunk: None,
            current_left_row: 0,
            current_results: Vec::new(),
            current_result_position: 0,
            chunk_capacity: 1024,
            left_exhausted: false,
            uses_index: false,
        }
    }

    /// Creates a vector join for entity-to-entity similarity.
    ///
    /// For each entity on the left, fetches its embedding and finds similar
    /// entities on the right.
    ///
    /// # Arguments
    ///
    /// * `left` - Left input operator
    /// * `store` - LPG store for property lookup
    /// * `left_node_column` - Column index containing NodeId in left input
    /// * `left_property` - Property containing left-side embeddings
    /// * `right_property` - Property containing right-side embeddings
    /// * `k` - Number of nearest neighbors per left entity
    /// * `metric` - Distance metric
    #[must_use]
    pub fn entity_to_entity(
        left: Box<dyn Operator>,
        store: Arc<LpgStore>,
        left_node_column: usize,
        left_property: impl Into<String>,
        right_property: impl Into<String>,
        k: usize,
        metric: DistanceMetric,
    ) -> Self {
        Self {
            left,
            store,
            #[cfg(feature = "vector-index")]
            index: None,
            left_property: Some(left_property.into()),
            left_node_column: Some(left_node_column),
            query_vector: None,
            right_property: right_property.into(),
            right_label: None,
            k,
            metric,
            min_similarity: None,
            max_distance: None,
            ef: 64,
            current_left_chunk: None,
            current_left_row: 0,
            current_results: Vec::new(),
            current_result_position: 0,
            chunk_capacity: 1024,
            left_exhausted: false,
            uses_index: false,
        }
    }

    /// Sets an HNSW index for the right side.
    #[cfg(feature = "vector-index")]
    #[must_use]
    pub fn with_index(mut self, index: Arc<HnswIndex>) -> Self {
        self.index = Some(index);
        self.uses_index = true;
        self
    }

    /// Sets a label filter for right-side entities.
    #[must_use]
    pub fn with_right_label(mut self, label: impl Into<String>) -> Self {
        self.right_label = Some(label.into());
        self
    }

    /// Sets minimum similarity threshold (cosine metric only).
    #[must_use]
    pub fn with_min_similarity(mut self, threshold: f32) -> Self {
        self.min_similarity = Some(threshold);
        self
    }

    /// Sets maximum distance threshold.
    #[must_use]
    pub fn with_max_distance(mut self, threshold: f32) -> Self {
        self.max_distance = Some(threshold);
        self
    }

    /// Sets the HNSW search ef parameter.
    #[must_use]
    pub fn with_ef(mut self, ef: usize) -> Self {
        self.ef = ef;
        self
    }

    /// Sets the output chunk capacity.
    #[must_use]
    pub fn with_chunk_capacity(mut self, capacity: usize) -> Self {
        self.chunk_capacity = capacity;
        self
    }

    /// Gets the query vector for the current left row.
    fn get_query_vector(&self) -> Option<Vec<f32>> {
        // Static query vector (same for all left rows)
        if let Some(ref v) = self.query_vector {
            return Some(v.clone());
        }

        // Entity-to-entity: fetch from left entity's property
        if let (Some(chunk), Some(col_idx), Some(prop)) = (
            &self.current_left_chunk,
            self.left_node_column,
            &self.left_property,
        ) && let Some(col) = chunk.column(col_idx)
            && let Some(node_id) = col.get_node_id(self.current_left_row)
            && let Some(Value::Vector(vec)) = self
                .store
                .get_node_property(node_id, &PropertyKey::new(prop))
        {
            return Some(vec.to_vec());
        }

        None
    }

    /// Performs vector search for the current query.
    fn search_right_side(&self, query: &[f32]) -> Vec<(NodeId, f32)> {
        #[cfg(feature = "vector-index")]
        {
            if let Some(ref index) = self.index {
                return index.search_with_ef(query, self.k, self.ef);
            }
        }

        // Brute-force search
        self.brute_force_search(query)
    }

    /// Performs brute-force k-NN search.
    fn brute_force_search(&self, query: &[f32]) -> Vec<(NodeId, f32)> {
        // Get right-side nodes (optionally filtered by label)
        let node_ids = match &self.right_label {
            Some(label) => self.store.nodes_by_label(label),
            None => self.store.node_ids(),
        };

        // Collect vectors from node properties
        let vectors: Vec<(NodeId, Vec<f32>)> = node_ids
            .into_iter()
            .filter_map(|id| {
                self.store
                    .get_node_property(id, &PropertyKey::new(&self.right_property))
                    .and_then(|v| {
                        if let Value::Vector(vec) = v {
                            Some((id, vec.to_vec()))
                        } else {
                            None
                        }
                    })
            })
            .collect();

        // Run brute-force k-NN
        let iter = vectors.iter().map(|(id, v)| (*id, v.as_slice()));
        brute_force_knn(iter, query, self.k, self.metric)
    }

    /// Applies similarity/distance filters to results.
    fn apply_filters(&self, results: &mut Vec<(NodeId, f32)>) {
        if self.min_similarity.is_none() && self.max_distance.is_none() {
            return;
        }

        results.retain(|(_, distance)| {
            // For cosine metric, convert distance to similarity
            let passes_similarity = match self.min_similarity {
                Some(threshold) if self.metric == DistanceMetric::Cosine => {
                    let similarity = 1.0 - distance;
                    similarity >= threshold
                }
                Some(_) => true, // Similarity filter only applies to cosine
                None => true,
            };

            let passes_distance = match self.max_distance {
                Some(threshold) => *distance <= threshold,
                None => true,
            };

            passes_similarity && passes_distance
        });
    }

    /// Advances to the next left row that has results.
    fn advance_left(&mut self) -> Result<bool, OperatorError> {
        loop {
            // Need a new left chunk?
            if self.current_left_chunk.is_none()
                || self.current_left_row
                    >= self
                        .current_left_chunk
                        .as_ref()
                        .map_or(0, DataChunk::row_count)
            {
                match self.left.next()? {
                    Some(chunk) => {
                        self.current_left_chunk = Some(chunk);
                        self.current_left_row = 0;
                    }
                    None => {
                        self.left_exhausted = true;
                        return Ok(false);
                    }
                }
            }

            // Get query vector for current left row
            if let Some(query) = self.get_query_vector() {
                // Search right side
                let mut results = self.search_right_side(&query);
                self.apply_filters(&mut results);

                if !results.is_empty() {
                    self.current_results = results;
                    self.current_result_position = 0;
                    return Ok(true);
                }
            }

            // No results for this left row, try next
            self.current_left_row += 1;
        }
    }
}

impl Operator for VectorJoinOperator {
    fn next(&mut self) -> OperatorResult {
        if self.left_exhausted && self.current_result_position >= self.current_results.len() {
            return Ok(None);
        }

        // Ensure we have results to process (advances left if needed)
        if self.current_result_position >= self.current_results.len() && !self.advance_left()? {
            return Ok(None);
        }

        // Get left chunk schema for output (now guaranteed to have a chunk)
        let left_chunk = self
            .current_left_chunk
            .as_ref()
            .ok_or_else(|| OperatorError::Execution("No left chunk available".into()))?;

        let left_schema: Vec<LogicalType> = (0..left_chunk.column_count())
            .filter_map(|i| left_chunk.column(i).map(|col| col.data_type().clone()))
            .collect();

        // Build output schema: [left columns..., right_node, score]
        let mut output_schema = left_schema.clone();
        output_schema.push(LogicalType::Node);
        output_schema.push(LogicalType::Float64);

        let mut output = DataChunk::with_capacity(&output_schema, self.chunk_capacity);
        let mut row_count = 0;

        while row_count < self.chunk_capacity {
            // Need more results?
            if self.current_result_position >= self.current_results.len() {
                self.current_left_row += 1;
                if !self.advance_left()? {
                    break;
                }
            }

            // Get current left row values
            let left_chunk = self
                .current_left_chunk
                .as_ref()
                .ok_or_else(|| OperatorError::Execution("No left chunk available".into()))?;

            // Copy left columns
            for (col_idx, _) in left_schema.iter().enumerate() {
                if let Some(col) = left_chunk.column(col_idx)
                    && let Some(out_col) = output.column_mut(col_idx)
                    && let Some(val) = col.get_value(self.current_left_row)
                {
                    out_col.push_value(val);
                }
            }

            // Add right node and score
            let (right_node, distance) = self.current_results[self.current_result_position];

            if let Some(node_col) = output.column_mut(left_schema.len()) {
                node_col.push_node_id(right_node);
            }

            if let Some(score_col) = output.column_mut(left_schema.len() + 1) {
                score_col.push_float64(f64::from(distance));
            }

            self.current_result_position += 1;
            row_count += 1;
        }

        if row_count == 0 {
            return Ok(None);
        }

        output.set_count(row_count);
        Ok(Some(output))
    }

    fn reset(&mut self) {
        self.left.reset();
        self.current_left_chunk = None;
        self.current_left_row = 0;
        self.current_results.clear();
        self.current_result_position = 0;
        self.left_exhausted = false;
    }

    fn name(&self) -> &'static str {
        if self.uses_index {
            "VectorJoin(HNSW)"
        } else {
            "VectorJoin(BruteForce)"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::operators::single_row::NodeListOperator;
    use std::sync::Arc as StdArc;

    #[test]
    fn test_vector_join_static_query() {
        let store = StdArc::new(LpgStore::new());

        // Create nodes with vector embeddings
        let n1 = store.create_node(&["Item"]);
        let n2 = store.create_node(&["Item"]);
        let n3 = store.create_node(&["Item"]);

        store.set_node_property(n1, "embedding", Value::Vector(vec![1.0, 0.0, 0.0].into()));
        store.set_node_property(n2, "embedding", Value::Vector(vec![0.0, 1.0, 0.0].into()));
        store.set_node_property(n3, "embedding", Value::Vector(vec![0.9, 0.1, 0.0].into()));

        // Create a simple left operator that produces one row
        let left = Box::new(NodeListOperator::new(vec![n1], 1024));

        // Create vector join with query similar to n1
        let query = vec![1.0, 0.0, 0.0];
        let mut join = VectorJoinOperator::with_static_query(
            left,
            StdArc::clone(&store),
            query,
            "embedding",
            3,
            DistanceMetric::Euclidean,
        );

        // Should find all 3 items, sorted by distance
        let mut total_results = 0;
        while let Ok(Some(chunk)) = join.next() {
            total_results += chunk.row_count();
        }

        assert_eq!(total_results, 3);
    }

    #[test]
    fn test_vector_join_entity_to_entity() {
        let store = StdArc::new(LpgStore::new());

        // Create source nodes
        let src1 = store.create_node(&["Source"]);
        store.set_node_property(src1, "embedding", Value::Vector(vec![1.0, 0.0].into()));

        // Create target nodes
        let t1 = store.create_node(&["Target"]);
        let t2 = store.create_node(&["Target"]);
        store.set_node_property(t1, "embedding", Value::Vector(vec![0.9, 0.1].into()));
        store.set_node_property(t2, "embedding", Value::Vector(vec![0.0, 1.0].into()));

        // Left operator produces source node
        let left = Box::new(NodeListOperator::new(vec![src1], 1024));

        // Entity-to-entity join: find targets similar to source
        let mut join = VectorJoinOperator::entity_to_entity(
            left,
            StdArc::clone(&store),
            0, // node column
            "embedding",
            "embedding",
            2,
            DistanceMetric::Euclidean,
        )
        .with_right_label("Target");

        // Should find both targets
        let mut total_results = 0;
        while let Ok(Some(chunk)) = join.next() {
            total_results += chunk.row_count();
            // First result should be t1 (closer to src1)
            if total_results == 1 {
                let right_col = chunk.column(1).unwrap();
                let right_node = right_col.get_node_id(0).unwrap();
                assert_eq!(right_node, t1);
            }
        }

        assert_eq!(total_results, 2);
    }

    #[test]
    fn test_vector_join_with_distance_filter() {
        let store = StdArc::new(LpgStore::new());

        // Create nodes with embeddings
        let n1 = store.create_node(&["Item"]);
        let n2 = store.create_node(&["Item"]);
        store.set_node_property(n1, "vec", Value::Vector(vec![1.0, 0.0].into()));
        store.set_node_property(n2, "vec", Value::Vector(vec![0.0, 1.0].into())); // Far away

        let left = Box::new(NodeListOperator::new(vec![n1], 1024));
        let query = vec![1.0, 0.0];

        let mut join = VectorJoinOperator::with_static_query(
            left,
            StdArc::clone(&store),
            query,
            "vec",
            10,
            DistanceMetric::Euclidean,
        )
        .with_max_distance(0.5); // Only very close matches

        // Should find only n1 (distance 0.0)
        let mut results = Vec::new();
        while let Ok(Some(chunk)) = join.next() {
            for i in 0..chunk.row_count() {
                let node = chunk.column(1).unwrap().get_node_id(i).unwrap();
                results.push(node);
            }
        }

        assert_eq!(results.len(), 1);
        assert_eq!(results[0], n1);
    }

    #[test]
    fn test_vector_join_name() {
        let store = StdArc::new(LpgStore::new());
        let left = Box::new(NodeListOperator::new(vec![], 1024));

        let join = VectorJoinOperator::with_static_query(
            left,
            store,
            vec![1.0],
            "embedding",
            5,
            DistanceMetric::Cosine,
        );

        assert_eq!(join.name(), "VectorJoin(BruteForce)");
    }
}
