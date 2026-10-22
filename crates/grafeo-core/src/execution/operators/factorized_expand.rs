//! Factorized expand operator for relationship traversal without row duplication.
//!
//! Unlike the regular [`ExpandOperator`](super::ExpandOperator) which duplicates input
//! rows for each neighbor, this operator keeps input data factorized and adds expansion
//! results as a new level. This provides massive memory and performance improvements
//! for multi-hop queries with high fan-out.
//!
//! # Example
//!
//! For a 2-hop query with 10 source nodes, each having 10 neighbors:
//!
//! - **Regular Expand**: 10 * 10 = 100 rows, each with duplicated source data
//! - **Factorized Expand**: 10 sources + 100 neighbors = 110 values, no duplication

use std::sync::Arc;

use super::expand::visible_edges_from;
use super::{FactorizedOperator, Operator, OperatorError, OperatorResult};
use crate::execution::DataChunk;
use crate::execution::factorized_chunk::FactorizedChunk;
use crate::execution::vector::ValueVector;
use crate::graph::Direction;
use crate::graph::GraphStoreSearch;
use grafeo_common::types::{EdgeId, EpochId, LogicalType, NodeId, TransactionId};

/// Result type for factorized operations.
pub type FactorizedResult = Result<Option<FactorizedChunk>, OperatorError>;

/// An expand operator that produces factorized output.
///
/// Instead of duplicating input rows for each neighbor (Cartesian product),
/// this operator adds neighbors as a new factorization level. This avoids
/// exponential blowup in multi-hop queries.
///
/// # Memory Comparison
///
/// For a query `MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(c)` with:
/// - 100 source nodes
/// - Average 10 neighbors per hop
///
/// **Regular Expand (flat)**:
/// - After hop 1: 100 * 10 = 1,000 rows
/// - After hop 2: 1,000 * 10 = 10,000 rows
/// - Memory: ~10,000 * row_size
///
/// **Factorized Expand**:
/// - Level 0: 100 source nodes
/// - Level 1: 1,000 first-hop neighbors
/// - Level 2: 10,000 second-hop neighbors
/// - Memory: ~11,100 values (no duplication)
pub struct FactorizedExpandOperator {
    /// The store to traverse.
    store: Arc<dyn GraphStoreSearch>,
    /// Input operator providing source nodes.
    input: Box<dyn Operator>,
    /// Index of the source node column in input.
    source_column: usize,
    /// Direction of edge traversal.
    direction: Direction,
    /// Edge type filter (empty = match all types, multiple = match any).
    edge_types: Vec<String>,
    /// Transaction ID for MVCC visibility (None = use current epoch).
    transaction_id: Option<TransactionId>,
    /// Epoch for version visibility.
    viewing_epoch: Option<EpochId>,
    /// When true, skip versioned MVCC lookups (fast path for read-only queries).
    read_only: bool,
    /// Whether the operator is exhausted.
    exhausted: bool,
    /// Column names for the input (for tracking).
    input_column_names: Vec<String>,
}

impl FactorizedExpandOperator {
    /// Creates a new factorized expand operator.
    pub fn new(
        store: Arc<dyn GraphStoreSearch>,
        input: Box<dyn Operator>,
        source_column: usize,
        direction: Direction,
        edge_types: Vec<String>,
    ) -> Self {
        Self {
            store,
            input,
            source_column,
            direction,
            edge_types,
            transaction_id: None,
            viewing_epoch: None,
            read_only: false,
            exhausted: false,
            input_column_names: Vec::new(),
        }
    }

    /// Sets the transaction context for MVCC visibility.
    pub fn with_transaction_context(
        mut self,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Self {
        self.viewing_epoch = Some(epoch);
        self.transaction_id = transaction_id;
        self
    }

    /// Marks this expand as read-only, enabling fast-path lookups.
    pub fn with_read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }

    /// Sets the input column names for schema tracking.
    pub fn with_column_names(mut self, names: Vec<String>) -> Self {
        self.input_column_names = names;
        self
    }

    /// Gets neighbors for a source node with type and visibility filtering
    /// (see [`visible_edges_from`]).
    fn get_neighbors(&self, source_id: NodeId) -> Vec<(NodeId, EdgeId)> {
        visible_edges_from(
            self.store.as_ref(),
            source_id,
            self.direction,
            &self.edge_types,
            self.viewing_epoch,
            self.transaction_id,
            self.read_only,
        )
    }

    /// Processes an input chunk and produces a factorized chunk with expansion.
    fn process_chunk(&self, input: DataChunk) -> Result<FactorizedChunk, OperatorError> {
        let source_col = input.column(self.source_column).ok_or_else(|| {
            OperatorError::ColumnNotFound(format!("Column {} not found", self.source_column))
        })?;

        let row_count = input.row_count();

        // Collect all edges and build offsets
        let mut edge_ids = ValueVector::with_type(LogicalType::Edge);
        let mut target_ids = ValueVector::with_type(LogicalType::Node);
        let mut offsets: Vec<u32> = Vec::with_capacity(row_count + 1);
        offsets.push(0);

        for row_idx in 0..row_count {
            let source_id = source_col.get_node_id(row_idx).ok_or_else(|| {
                OperatorError::Internal("Expected node ID in source column".into())
            })?;

            let neighbors = self.get_neighbors(source_id);

            for (target_id, edge_id) in neighbors {
                edge_ids.push_edge_id(edge_id);
                target_ids.push_node_id(target_id);
            }

            // reason: factorized column lengths fit u32
            #[allow(clippy::cast_possible_truncation)]
            offsets.push(edge_ids.len() as u32);
        }

        // Build column names
        let mut column_names: Vec<String> = if self.input_column_names.is_empty() {
            (0..input.column_count())
                .map(|i| format!("col_{}", i))
                .collect()
        } else {
            self.input_column_names.clone()
        };

        // Create the factorized chunk starting from the flat input
        let mut chunk = FactorizedChunk::from_flat(&input, column_names.clone());

        // Add the expansion level if there are any edges
        if !edge_ids.is_empty() {
            column_names.push("_edge".to_string());
            column_names.push("_target".to_string());

            chunk.add_level(
                vec![edge_ids, target_ids],
                vec!["_edge".to_string(), "_target".to_string()],
                &offsets,
            );
        }

        Ok(chunk)
    }

    /// Gets the next factorized chunk.
    ///
    /// This is the main method for factorized execution. For compatibility
    /// with the regular `Operator` trait, use `next()` which flattens the result.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the input operator or graph traversal fails.
    pub fn next_factorized(&mut self) -> FactorizedResult {
        if self.exhausted {
            return Ok(None);
        }

        match self.input.next() {
            Ok(Some(input)) => {
                let result = self.process_chunk(input)?;
                Ok(Some(result))
            }
            Ok(None) => {
                self.exhausted = true;
                Ok(None)
            }
            Err(e) => Err(e),
        }
    }
}

impl Operator for FactorizedExpandOperator {
    fn next(&mut self) -> OperatorResult {
        // For compatibility, flatten the factorized result
        match self.next_factorized() {
            Ok(Some(factorized)) => Ok(Some(factorized.flatten())),
            Ok(None) => Ok(None),
            Err(e) => Err(e),
        }
    }

    fn reset(&mut self) {
        self.input.reset();
        self.exhausted = false;
    }

    fn name(&self) -> &'static str {
        "FactorizedExpand"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

impl FactorizedOperator for FactorizedExpandOperator {
    fn next_factorized(&mut self) -> FactorizedResult {
        FactorizedExpandOperator::next_factorized(self)
    }
}

/// Builder for chaining multiple factorized expansions.
///
/// This is useful for multi-hop queries where you want to keep data
/// factorized across multiple expansion steps.
pub struct FactorizedExpandChain {
    /// The store to traverse.
    store: Arc<dyn GraphStoreSearch>,
    /// The source operator for the first expansion.
    source: Option<Box<dyn Operator>>,
    /// Accumulated factorized result.
    current_result: Option<FactorizedChunk>,
    /// Transaction context.
    transaction_id: Option<TransactionId>,
    viewing_epoch: Option<EpochId>,
    /// When true, skip versioned MVCC lookups (fast path for read-only queries).
    read_only: bool,
}

impl FactorizedExpandChain {
    /// Creates a new chain starting from a source operator.
    pub fn new(store: Arc<dyn GraphStoreSearch>, source: Box<dyn Operator>) -> Self {
        Self {
            store,
            source: Some(source),
            current_result: None,
            transaction_id: None,
            viewing_epoch: None,
            read_only: false,
        }
    }

    /// Sets the transaction context.
    pub fn with_transaction_context(
        mut self,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Self {
        self.viewing_epoch = Some(epoch);
        self.transaction_id = transaction_id;
        self
    }

    /// Marks this chain as read-only, enabling fast-path lookups.
    pub fn with_read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }

    /// Adds an expansion step to the chain.
    ///
    /// Returns `self` for chaining.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the source operator or graph traversal fails.
    pub fn expand(
        mut self,
        source_column: usize,
        direction: Direction,
        edge_types: Vec<String>,
    ) -> Result<Self, OperatorError> {
        // Get or create the initial factorized chunk
        if self.current_result.is_none() {
            if let Some(mut source) = self.source.take() {
                // Collect ALL batches from the source operator
                // This is necessary because the source may produce multiple batches
                let merged_input = Self::collect_all_batches(&mut *source)?;

                if let Some(input) = merged_input {
                    let mut expand = FactorizedExpandOperator::new(
                        Arc::clone(&self.store),
                        Box::new(SingleChunkOperator::new(input)),
                        source_column,
                        direction,
                        edge_types,
                    )
                    .with_read_only(self.read_only);

                    if let Some(epoch) = self.viewing_epoch {
                        expand = expand.with_transaction_context(epoch, self.transaction_id);
                    }

                    if let Some(result) = expand.next_factorized()? {
                        self.current_result = Some(result);
                    }
                }
            }
        } else {
            // Expand the deepest level of the factorized result
            // This adds a new level without flattening - the key to memory savings
            if let Some(mut factorized) = self.current_result.take() {
                let level_count_before = factorized.level_count();
                self.expand_deepest_level(&mut factorized, source_column, direction, edge_types)?;
                if factorized.level_count() > level_count_before {
                    // New level was added: keep result
                    self.current_result = Some(factorized);
                }
                // Otherwise no edges were found at this level, so no valid
                // paths exist through the chain. Leave current_result as None
                // to signal an empty result set.
            }
        }

        Ok(self)
    }

    /// Collects all batches from an operator into a single merged chunk.
    fn collect_all_batches(source: &mut dyn Operator) -> Result<Option<DataChunk>, OperatorError> {
        let mut chunks: Vec<DataChunk> = Vec::new();

        while let Some(mut chunk) = source.next()? {
            // IMPORTANT: Flatten the chunk to materialize selection vectors.
            // FilterOperator returns chunks with selection vectors that logically filter rows,
            // but the underlying column data is unchanged. We need to physically copy only
            // the selected rows before using the data in factorized expansion.
            chunk.flatten();

            if chunk.row_count() > 0 {
                chunks.push(chunk);
            }
        }

        if chunks.is_empty() {
            return Ok(None);
        }

        if chunks.len() == 1 {
            return Ok(Some(chunks.remove(0)));
        }

        // Merge multiple chunks into one
        let first = &chunks[0];
        let total_rows: usize = chunks.iter().map(|c| c.row_count()).sum();
        let col_count = first.column_count();

        // Create merged vectors for each column
        let mut merged_cols: Vec<ValueVector> = (0..col_count)
            .map(|i| {
                let col_type = first
                    .column(i)
                    .map_or(&LogicalType::Any, |c| c.data_type())
                    .clone();
                ValueVector::with_type(col_type)
            })
            .collect();

        // Copy data from all chunks
        for chunk in &chunks {
            for col_idx in 0..col_count {
                if let Some(src_col) = chunk.column(col_idx) {
                    let dst_col = &mut merged_cols[col_idx];
                    for row_idx in 0..chunk.row_count() {
                        if let Some(value) = src_col.get_value(row_idx) {
                            dst_col.push_value(value);
                        }
                    }
                }
            }
        }

        let mut merged = DataChunk::new(merged_cols);
        merged.set_count(total_rows);

        Ok(Some(merged))
    }

    /// Expands the deepest level of a factorized chunk, adding a new level.
    ///
    /// This is the key method for multi-hop factorized execution. Instead of
    /// flattening and re-expanding, it directly processes the deepest level's
    /// target nodes and adds neighbors as a new level.
    fn expand_deepest_level(
        &self,
        chunk: &mut FactorizedChunk,
        source_column: usize,
        direction: Direction,
        edge_types: Vec<String>,
    ) -> Result<(), OperatorError> {
        // Get the deepest level to find source nodes
        let deepest_level = chunk.level_count() - 1;
        let level = chunk
            .level(deepest_level)
            .ok_or_else(|| OperatorError::Internal("No levels in factorized chunk".into()))?;

        // Check if the source column exists in this level
        // If not, it means the previous expansion produced no edges (no level 1 was added)
        // In that case, there's nothing to expand further
        let Some(source_col) = level.column(source_column) else {
            // No source column means previous expand had no results
            // This is valid - just means no paths exist through this expansion
            return Ok(());
        };

        // Collect all edges for all source nodes in the deepest level
        let mut edge_ids = ValueVector::with_type(LogicalType::Edge);
        let mut target_ids = ValueVector::with_type(LogicalType::Node);
        let source_len = source_col.physical_len();
        let mut offsets: Vec<u32> = Vec::with_capacity(source_len + 1);
        offsets.push(0);

        // Iterate through all physical values in the source column
        for idx in 0..source_len {
            let source_id = source_col.data().get_node_id(idx).ok_or_else(|| {
                OperatorError::Internal("Expected node ID in source column".into())
            })?;

            // Get neighbors with filtering
            let neighbors = visible_edges_from(
                self.store.as_ref(),
                source_id,
                direction,
                &edge_types,
                self.viewing_epoch,
                self.transaction_id,
                self.read_only,
            );

            for (target_id, edge_id) in neighbors {
                edge_ids.push_edge_id(edge_id);
                target_ids.push_node_id(target_id);
            }

            // reason: factorized column lengths fit u32
            #[allow(clippy::cast_possible_truncation)]
            offsets.push(edge_ids.len() as u32);
        }

        // Add the new level if there are any edges
        if !edge_ids.is_empty() {
            chunk.add_level(
                vec![edge_ids, target_ids],
                vec!["_edge".to_string(), "_target".to_string()],
                &offsets,
            );
        }

        Ok(())
    }

    /// Finishes the chain and returns the factorized result.
    pub fn finish(self) -> Option<FactorizedChunk> {
        self.current_result
    }

    /// Finishes the chain and returns a flattened DataChunk.
    pub fn finish_flat(self) -> Option<DataChunk> {
        self.current_result.map(|c| c.flatten())
    }
}

/// Helper operator that returns a single chunk once.
struct SingleChunkOperator {
    chunk: Option<DataChunk>,
}

impl SingleChunkOperator {
    fn new(chunk: DataChunk) -> Self {
        Self { chunk: Some(chunk) }
    }
}

impl Operator for SingleChunkOperator {
    fn next(&mut self) -> OperatorResult {
        Ok(self.chunk.take())
    }

    fn reset(&mut self) {
        // Cannot reset - chunk is consumed
    }

    fn name(&self) -> &'static str {
        "SingleChunk"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// Configuration for a single expand step in a lazy chain.
#[derive(Clone)]
pub struct ExpandStep {
    /// Source column index within the current level.
    pub source_column: usize,
    /// Direction of edge traversal.
    pub direction: Direction,
    /// Edge type filter (empty = match all types, multiple = match any).
    pub edge_types: Vec<String>,
}

/// A lazy operator that executes a factorized expand chain when next() is called.
///
/// Unlike `FactorizedExpandChain` which executes immediately during construction,
/// this operator defers execution until query runtime. This is critical for
/// correctness when filters are applied above the expand chain.
///
/// # Factorized Aggregation Support
///
/// This operator supports returning factorized results via `next_factorized()`.
/// When the downstream operator can handle factorized data (e.g., factorized
/// aggregation), this avoids flattening and provides massive speedups.
pub struct LazyFactorizedChainOperator {
    /// The graph store.
    store: Arc<dyn GraphStoreSearch>,
    /// The source operator (filter, scan, etc). It stays here, so a reset
    /// can run the chain again.
    source: Box<dyn Operator>,
    /// The expand steps to execute.
    steps: Vec<ExpandStep>,
    /// Transaction ID for MVCC visibility.
    transaction_id: Option<TransactionId>,
    /// Epoch for version visibility.
    viewing_epoch: Option<EpochId>,
    /// When true, skip versioned MVCC lookups (fast path for read-only queries).
    read_only: bool,
    /// Cached flat result after execution.
    result: Option<DataChunk>,
    /// Cached factorized result after execution.
    factorized_result: Option<FactorizedChunk>,
    /// Whether execution has completed.
    executed: bool,
}

impl LazyFactorizedChainOperator {
    /// Creates a new lazy factorized chain operator.
    pub fn new(
        store: Arc<dyn GraphStoreSearch>,
        source: Box<dyn Operator>,
        steps: Vec<ExpandStep>,
    ) -> Self {
        Self {
            store,
            source,
            steps,
            transaction_id: None,
            viewing_epoch: None,
            read_only: false,
            result: None,
            factorized_result: None,
            executed: false,
        }
    }

    /// Sets the transaction context for MVCC visibility.
    pub fn with_transaction_context(
        mut self,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Self {
        self.viewing_epoch = Some(epoch);
        self.transaction_id = transaction_id;
        self
    }

    /// Marks this operator as read-only, enabling fast-path lookups.
    pub fn with_read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }

    /// Executes the chain and returns the factorized result.
    ///
    /// This is the key method for factorized aggregation - it returns the
    /// factorized chunk without flattening, allowing O(n) aggregation instead
    /// of O(n²) or worse.
    fn execute_factorized(&mut self) -> Result<Option<FactorizedChunk>, OperatorError> {
        // The chain expands the source's rows as one chunk; reading them here
        // keeps the source, so a reset can run the chain again (a correlated
        // subquery runs it once per outer row).
        let Some(input) = FactorizedExpandChain::collect_all_batches(&mut *self.source)? else {
            return Ok(None);
        };

        // Build and execute the chain
        let mut chain = FactorizedExpandChain::new(
            Arc::clone(&self.store),
            Box::new(SingleChunkOperator::new(input)),
        )
        .with_read_only(self.read_only);

        if let Some(epoch) = self.viewing_epoch {
            chain = chain.with_transaction_context(epoch, self.transaction_id);
        }

        // Execute each expand step
        for step in &self.steps {
            chain = chain.expand(step.source_column, step.direction, step.edge_types.clone())?;
        }

        // Return the factorized result (not flattened)
        Ok(chain.finish())
    }

    /// Returns the factorized result without flattening.
    ///
    /// Use this when the next operator can handle factorized data (e.g.,
    /// factorized aggregation). This is the key to 10-100x speedups for
    /// aggregate queries on multi-hop traversals.
    ///
    /// # Returns
    ///
    /// The factorized chunk, or None if exhausted or no results.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the underlying chain execution fails.
    pub fn next_factorized(&mut self) -> FactorizedResult {
        if self.executed {
            return Ok(self.factorized_result.take());
        }

        self.executed = true;
        self.factorized_result = self.execute_factorized()?;
        Ok(self.factorized_result.clone())
    }

    /// Executes the chain and returns the flattened result.
    fn execute(&mut self) -> Result<Option<DataChunk>, OperatorError> {
        // Use the factorized execution and then flatten
        let factorized = self.execute_factorized()?;
        Ok(factorized.map(|c| c.flatten()))
    }
}

impl Operator for LazyFactorizedChainOperator {
    fn next(&mut self) -> OperatorResult {
        if self.executed {
            return Ok(self.result.take());
        }

        self.executed = true;
        self.result = self.execute()?;
        Ok(self.result.take())
    }

    fn reset(&mut self) {
        self.source.reset();
        self.result = None;
        self.factorized_result = None;
        self.executed = false;
    }

    fn name(&self) -> &'static str {
        "LazyFactorizedChain"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

#[cfg(all(test, feature = "lpg"))]
mod tests {
    use super::*;
    use crate::execution::operators::ScanOperator;
    use crate::graph::lpg::LpgStore;

    #[test]
    fn test_factorized_expand_basic() {
        let store = Arc::new(LpgStore::new().unwrap());

        // Create nodes
        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["Person"]);
        let vincent = store.create_node(&["Person"]);

        // Alix knows Gus and Vincent
        store.create_edge(alix, gus, "KNOWS");
        store.create_edge(alix, vincent, "KNOWS");

        let scan = Box::new(ScanOperator::with_label(store.clone(), "Person"));

        let mut expand = FactorizedExpandOperator::new(
            store.clone(),
            scan,
            0,
            Direction::Outgoing,
            vec!["KNOWS".to_string()],
        );

        // Get factorized result
        let result = expand.next_factorized().unwrap();
        assert!(result.is_some());

        let chunk = result.unwrap();

        // Should have 2 levels: sources and neighbors
        assert_eq!(chunk.level_count(), 2);

        // Level 0 has 3 sources (Alix, Gus, Vincent)
        assert_eq!(chunk.level(0).unwrap().column_count(), 1);

        // Level 1 has edges and targets
        // Only Alix has outgoing KNOWS edges (to Gus and Vincent)
        // So we should have 2 edges total
        assert_eq!(chunk.level(1).unwrap().column_count(), 2);
    }

    #[test]
    fn test_factorized_vs_flat_equivalence() {
        let store = Arc::new(LpgStore::new().unwrap());

        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["Person"]);
        let vincent = store.create_node(&["Person"]);

        store.create_edge(alix, gus, "KNOWS");
        store.create_edge(alix, vincent, "KNOWS");
        store.create_edge(gus, vincent, "KNOWS");

        // Run factorized expand
        let scan1 = Box::new(ScanOperator::with_label(store.clone(), "Person"));
        let mut factorized_expand =
            FactorizedExpandOperator::new(store.clone(), scan1, 0, Direction::Outgoing, vec![]);

        let factorized_result = factorized_expand.next_factorized().unwrap().unwrap();
        let flat_from_factorized = factorized_result.flatten();

        // Run regular expand (using the factorized operator's flat interface)
        let scan2 = Box::new(ScanOperator::with_label(store.clone(), "Person"));
        let mut regular_expand =
            FactorizedExpandOperator::new(store.clone(), scan2, 0, Direction::Outgoing, vec![]);

        let flat_result = regular_expand.next().unwrap().unwrap();

        // Both should have the same row count
        assert_eq!(
            flat_from_factorized.row_count(),
            flat_result.row_count(),
            "Factorized and flat should produce same row count"
        );
    }

    #[test]
    fn test_factorized_expand_no_edges() {
        let store = Arc::new(LpgStore::new().unwrap());

        // Create nodes with no edges
        store.create_node(&["Person"]);
        store.create_node(&["Person"]);

        let scan = Box::new(ScanOperator::with_label(store.clone(), "Person"));

        let mut expand =
            FactorizedExpandOperator::new(store.clone(), scan, 0, Direction::Outgoing, vec![]);

        let result = expand.next_factorized().unwrap();
        assert!(result.is_some());

        let chunk = result.unwrap();
        // Should only have the source level (no expansion level added when no edges)
        assert_eq!(chunk.level_count(), 1);
    }

    #[test]
    fn test_factorized_chain_two_hop() {
        let store = Arc::new(LpgStore::new().unwrap());

        // Create a 2-hop graph: a -> b1, b2 -> c1, c2, c3, c4
        let a = store.create_node(&["Person"]);
        let b1 = store.create_node(&["Person"]);
        let b2 = store.create_node(&["Person"]);
        let c1 = store.create_node(&["Person"]);
        let c2 = store.create_node(&["Person"]);
        let c3 = store.create_node(&["Person"]);
        let c4 = store.create_node(&["Person"]);

        // a knows b1 and b2
        store.create_edge(a, b1, "KNOWS");
        store.create_edge(a, b2, "KNOWS");

        // b1 knows c1 and c2
        store.create_edge(b1, c1, "KNOWS");
        store.create_edge(b1, c2, "KNOWS");

        // b2 knows c3 and c4
        store.create_edge(b2, c3, "KNOWS");
        store.create_edge(b2, c4, "KNOWS");

        // Create source chunk with just node 'a'
        let mut source_chunk = DataChunk::with_capacity(&[LogicalType::Node], 1);
        source_chunk.column_mut(0).unwrap().push_node_id(a);
        source_chunk.set_count(1);

        let source = Box::new(SingleChunkOperator::new(source_chunk));

        // Build 2-hop chain
        let chain = FactorizedExpandChain::new(store.clone(), source)
            .expand(0, Direction::Outgoing, vec!["KNOWS".to_string()])
            .unwrap()
            .expand(1, Direction::Outgoing, vec!["KNOWS".to_string()]) // column 1 is target from first expand
            .unwrap();

        let result = chain.finish().expect("Should have result");

        // Should have 3 levels: source (a), hop1 (b1,b2), hop2 (c1,c2,c3,c4)
        assert_eq!(result.level_count(), 3);

        // Physical size: 1 (source) + 2+2 (hop1 edges+targets) + 4+4 (hop2 edges+targets) = 13
        // vs flat which would be 4 rows * 5 columns = 20
        assert_eq!(result.physical_size(), 13);

        // Logical row count should be 4 (4 paths: a->b1->c1, a->b1->c2, a->b2->c3, a->b2->c4)
        assert_eq!(result.logical_row_count(), 4);

        // Flatten and verify
        let flat = result.flatten();
        assert_eq!(flat.row_count(), 4);
    }

    #[test]
    fn test_factorized_expand_multi_edge_type_filter() {
        let store = Arc::new(LpgStore::new().unwrap());

        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["Person"]);
        let vincent = store.create_node(&["City"]);

        // Mixed edge types
        store.create_edge(alix, gus, "KNOWS");
        store.create_edge(alix, vincent, "LIVES_IN");
        store.create_edge(gus, vincent, "WORKS_AT");

        let scan = Box::new(ScanOperator::with_label(store.clone(), "Person"));

        // Filter for KNOWS and LIVES_IN (case-insensitive)
        let mut expand = FactorizedExpandOperator::new(
            store.clone(),
            scan,
            0,
            Direction::Outgoing,
            vec!["knows".to_string(), "lives_in".to_string()],
        );

        let result = expand.next_factorized().unwrap().unwrap();
        let flat = result.flatten();

        // From Alix: KNOWS (to Gus) and LIVES_IN (to Vincent) = 2 rows
        // From Gus: WORKS_AT is filtered out = 0 rows
        assert_eq!(flat.row_count(), 2);
    }

    #[test]
    fn test_factorized_memory_savings() {
        let store = Arc::new(LpgStore::new().unwrap());

        // Create a star graph: center connected to 10 leaves
        let center = store.create_node(&["Center"]);
        let mut leaves = Vec::new();
        for _ in 0..10 {
            let leaf = store.create_node(&["Leaf"]);
            store.create_edge(center, leaf, "POINTS_TO");
            leaves.push(leaf);
        }

        // Scan just the center
        let mut source_chunk = DataChunk::with_capacity(&[LogicalType::Node], 1);
        source_chunk.column_mut(0).unwrap().push_node_id(center);
        source_chunk.set_count(1);

        let single = Box::new(SingleChunkOperator::new(source_chunk));

        let mut expand =
            FactorizedExpandOperator::new(store.clone(), single, 0, Direction::Outgoing, vec![]);

        let factorized = expand.next_factorized().unwrap().unwrap();

        // Physical size should be 1 (source) + 10 (edges) + 10 (targets) = 21 values
        // vs flat which would be 10 rows * 3 columns = 30 values
        assert_eq!(factorized.physical_size(), 21);

        // But logical row count should be 10
        assert_eq!(factorized.logical_row_count(), 10);

        // Flatten and verify correctness
        let flat = factorized.flatten();
        assert_eq!(flat.row_count(), 10);
    }

    #[test]
    fn test_factorized_expand_into_any() {
        let store = Arc::new(LpgStore::new().unwrap());
        let scan = Box::new(ScanOperator::with_label(store.clone(), "Person"));
        let op = FactorizedExpandOperator::new(store.clone(), scan, 0, Direction::Outgoing, vec![]);
        let any = Box::new(op).into_any();
        assert!(any.downcast::<FactorizedExpandOperator>().is_ok());
    }

    /// After a reset the chain runs again, as a correlated subquery runs it
    /// once per outer row: Alix knows Gus, who lives in Berlin, both times.
    #[test]
    fn lazy_chain_runs_again_after_a_reset() {
        let store = Arc::new(LpgStore::new().unwrap());
        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["Person"]);
        let berlin = store.create_node(&["City"]);
        store.create_edge(alix, gus, "KNOWS");
        store.create_edge(gus, berlin, "LIVES_IN");

        let scan = Box::new(ScanOperator::with_label(store.clone(), "Person"));
        let step = |edge_type: &str, source_column| ExpandStep {
            source_column,
            direction: Direction::Outgoing,
            edge_types: vec![edge_type.to_string()],
        };
        let mut chain = LazyFactorizedChainOperator::new(
            store.clone(),
            scan,
            vec![step("KNOWS", 0), step("LIVES_IN", 1)],
        );
        let rows = |chain: &mut LazyFactorizedChainOperator| {
            let mut rows = 0;
            while let Some(chunk) = chain.next().unwrap() {
                rows += chunk.row_count();
            }
            rows
        };

        assert_eq!(rows(&mut chain), 1);
        chain.reset();
        assert_eq!(rows(&mut chain), 1);
    }

    #[test]
    fn test_lazy_factorized_chain_into_any() {
        let store = Arc::new(LpgStore::new().unwrap());
        let scan = Box::new(ScanOperator::with_label(store.clone(), "Person"));
        let op = LazyFactorizedChainOperator::new(store.clone(), scan, vec![]);
        let any = Box::new(op).into_any();
        assert!(any.downcast::<LazyFactorizedChainOperator>().is_ok());
    }
}
