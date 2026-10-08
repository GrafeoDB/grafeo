//! Expand operator for relationship traversal.

use super::{Operator, OperatorError, OperatorResult};
use crate::execution::DataChunk;
use crate::graph::Direction;
use crate::graph::GraphStoreSearch;
use grafeo_common::types::{EdgeId, EpochId, LogicalType, NodeId, TransactionId};
use std::sync::Arc;

/// An expand operator that traverses edges from source nodes.
///
/// For each input row containing a source node, this operator produces
/// output rows for each neighbor connected via matching edges.
pub struct ExpandOperator {
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
    /// Chunk capacity.
    chunk_capacity: usize,
    /// Current input chunk being processed.
    current_input: Option<DataChunk>,
    /// Current row index in the input chunk.
    current_row: usize,
    /// Current edge iterator for the current row.
    current_edges: Vec<(NodeId, EdgeId)>,
    /// Current edge index.
    current_edge_idx: usize,
    /// Whether the operator is exhausted.
    exhausted: bool,
    /// Transaction ID for MVCC visibility (None = use current epoch).
    transaction_id: Option<TransactionId>,
    /// Epoch for version visibility.
    viewing_epoch: Option<EpochId>,
    /// When true, skip versioned (MVCC) lookups even if a transaction ID is
    /// present.  Safe for read-only queries where the transaction has no
    /// pending writes, avoiding the cost of walking version chains.
    read_only: bool,
}

impl ExpandOperator {
    /// Creates a new expand operator.
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
            chunk_capacity: 2048,
            current_input: None,
            current_row: 0,
            current_edges: Vec::with_capacity(16), // typical node degree
            current_edge_idx: 0,
            exhausted: false,
            transaction_id: None,
            viewing_epoch: None,
            read_only: false,
        }
    }

    /// Sets the chunk capacity.
    pub fn with_chunk_capacity(mut self, capacity: usize) -> Self {
        self.chunk_capacity = capacity;
        self
    }

    /// Sets the transaction context for MVCC visibility.
    ///
    /// When set, the expand will only traverse visible edges and nodes.
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
    ///
    /// When the query has no mutations, versioned MVCC lookups (which walk
    /// version chains to find PENDING writes) can be skipped in favour of
    /// cheaper epoch-only visibility checks.
    pub fn with_read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }

    /// Minimum chunk size for locality sort to be worthwhile.
    /// Below this threshold, the sort cost exceeds the cache savings.
    const LOCALITY_SORT_THRESHOLD: usize = 1024;

    /// Loads the next input chunk.
    fn load_next_input(&mut self) -> Result<bool, OperatorError> {
        match self.input.next() {
            Ok(Some(mut chunk)) => {
                // Flatten the chunk if it has a selection vector so we can use direct indexing
                chunk.flatten();
                // Sort by source node ID for cache locality during adjacency lookups.
                // Only worthwhile for larger chunks where the sort cost is amortized.
                if chunk.len() > Self::LOCALITY_SORT_THRESHOLD {
                    chunk = chunk.sort_by_column(self.source_column);
                }
                self.current_input = Some(chunk);
                self.current_row = 0;
                self.current_edges.clear();
                self.current_edge_idx = 0;
                Ok(true)
            }
            Ok(None) => {
                self.exhausted = true;
                Ok(false)
            }
            Err(e) => Err(e),
        }
    }

    /// Loads edges for the current row.
    fn load_edges_for_current_row(&mut self) -> Result<bool, OperatorError> {
        let Some(chunk) = &self.current_input else {
            return Ok(false);
        };

        if self.current_row >= chunk.row_count() {
            return Ok(false);
        }

        let col = chunk.column(self.source_column).ok_or_else(|| {
            OperatorError::ColumnNotFound(format!("Column {} not found", self.source_column))
        })?;

        let source_id = col
            .get_node_id(self.current_row)
            .ok_or_else(|| OperatorError::Execution("Expected node ID in source column".into()))?;

        self.current_edges = visible_edges_from(
            self.store.as_ref(),
            source_id,
            self.direction,
            &self.edge_types,
            self.viewing_epoch,
            self.transaction_id,
            self.read_only,
        );
        self.current_edge_idx = 0;
        Ok(true)
    }
}

/// The edges from `node` in `direction` that a query sees, each with the node
/// at its other end: those of one of `edge_types` (any type when empty) whose
/// edge and other end are visible at `epoch` (every edge without one). The
/// expand operators, variable-length expands, factorized chains and shortest
/// paths all walk edges with it.
///
/// A query that may have written (`read_only` false) in a transaction reads
/// the type and the visibility as of its transaction: they see the edges it
/// created and has not committed (the committed type, `edge_type`, has none
/// for them), and not those it deleted. A read-only query outside a
/// transaction has no writes pending, so the committed state at its epoch
/// answers both, without walking version chains.
pub(super) fn visible_edges_from(
    store: &dyn GraphStoreSearch,
    node: NodeId,
    direction: Direction,
    edge_types: &[String],
    epoch: Option<EpochId>,
    transaction_id: Option<TransactionId>,
    read_only: bool,
) -> Vec<(NodeId, EdgeId)> {
    let versioned = if read_only {
        None
    } else {
        epoch.zip(transaction_id)
    };
    let mut edges = store.edges_from(node, direction);
    edges.retain(|&(other, edge)| {
        if !edge_types.is_empty() {
            let actual = match versioned {
                Some((epoch, transaction_id)) => {
                    store.edge_type_versioned(edge, epoch, transaction_id)
                }
                None => store.edge_type(edge),
            };
            let type_matches = actual.is_some_and(|actual| {
                edge_types
                    .iter()
                    .any(|wanted| actual.as_str().eq_ignore_ascii_case(wanted))
            });
            if !type_matches {
                return false;
            }
        }
        match (versioned, epoch) {
            (Some((epoch, transaction_id)), _) => {
                store.is_edge_visible_versioned(edge, epoch, transaction_id)
                    && store.is_node_visible_versioned(other, epoch, transaction_id)
            }
            (None, Some(epoch)) => {
                store.is_edge_visible_at_epoch(edge, epoch)
                    && store.is_node_visible_at_epoch(other, epoch)
            }
            (None, None) => true,
        }
    });
    edges
}

impl Operator for ExpandOperator {
    fn next(&mut self) -> OperatorResult {
        if self.exhausted {
            return Ok(None);
        }

        // Build output schema: preserve all input columns + edge + target
        // We need to build this dynamically based on input schema
        if self.current_input.is_none() {
            if !self.load_next_input()? {
                return Ok(None);
            }
            self.load_edges_for_current_row()?;
        }
        let input_chunk = self.current_input.as_ref().expect("input loaded above");

        // Build schema: [input_columns..., edge, target]
        let input_col_count = input_chunk.column_count();
        let mut schema: Vec<LogicalType> = (0..input_col_count)
            .map(|i| {
                input_chunk
                    .column(i)
                    .map_or(LogicalType::Any, |c| c.data_type().clone())
            })
            .collect();
        schema.push(LogicalType::Edge);
        schema.push(LogicalType::Node);

        let mut chunk = DataChunk::with_capacity(&schema, self.chunk_capacity);
        let mut count = 0;

        while count < self.chunk_capacity {
            // If we need a new input chunk
            if self.current_input.is_none() {
                if !self.load_next_input()? {
                    break;
                }
                self.load_edges_for_current_row()?;
            }

            // If we've exhausted edges for current row, move to next row
            while self.current_edge_idx >= self.current_edges.len() {
                self.current_row += 1;

                // If we've exhausted the current input chunk, get next one
                if self.current_row >= self.current_input.as_ref().map_or(0, |c| c.row_count()) {
                    self.current_input = None;
                    if !self.load_next_input()? {
                        // No more input chunks
                        if count > 0 {
                            chunk.set_count(count);
                            return Ok(Some(chunk));
                        }
                        return Ok(None);
                    }
                }

                self.load_edges_for_current_row()?;
            }

            // Get the current edge
            let (target_id, edge_id) = self.current_edges[self.current_edge_idx];

            // Copy all input columns to output
            let input = self.current_input.as_ref().expect("input loaded above");
            for col_idx in 0..input_col_count {
                if let Some(input_col) = input.column(col_idx)
                    && let Some(output_col) = chunk.column_mut(col_idx)
                {
                    // Use copy_row_to which preserves NodeId/EdgeId types
                    input_col.copy_row_to(self.current_row, output_col);
                }
            }

            // Add edge column
            if let Some(col) = chunk.column_mut(input_col_count) {
                col.push_edge_id(edge_id);
            }

            // Add target node column
            if let Some(col) = chunk.column_mut(input_col_count + 1) {
                col.push_node_id(target_id);
            }

            count += 1;
            self.current_edge_idx += 1;
        }

        if count > 0 {
            chunk.set_count(count);
            Ok(Some(chunk))
        } else {
            Ok(None)
        }
    }

    fn reset(&mut self) {
        self.input.reset();
        self.current_input = None;
        self.current_row = 0;
        self.current_edges.clear();
        self.current_edge_idx = 0;
        self.exhausted = false;
    }

    fn name(&self) -> &'static str {
        "Expand"
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

    /// Creates a new `LpgStore` wrapped in an `Arc` and returns both the
    /// concrete handle (for mutation) and a trait-object handle (for operators).
    fn test_store() -> (Arc<LpgStore>, Arc<dyn GraphStoreSearch>) {
        let store = Arc::new(LpgStore::new().unwrap());
        let dyn_store: Arc<dyn GraphStoreSearch> = Arc::clone(&store) as Arc<dyn GraphStoreSearch>;
        (store, dyn_store)
    }

    #[test]
    fn test_expand_outgoing() {
        let (store, dyn_store) = test_store();

        // Create nodes
        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["Person"]);
        let vincent = store.create_node(&["Person"]);

        // Create edges: Alix -> Gus, Alix -> Vincent
        store.create_edge(alix, gus, "KNOWS");
        store.create_edge(alix, vincent, "KNOWS");

        // Scan Alix only
        let scan = Box::new(ScanOperator::with_label(Arc::clone(&dyn_store), "Person"));

        let mut expand = ExpandOperator::new(
            Arc::clone(&dyn_store),
            scan,
            0, // source column
            Direction::Outgoing,
            vec![],
        );

        // Collect all results
        let mut results = Vec::new();
        while let Ok(Some(chunk)) = expand.next() {
            for i in 0..chunk.row_count() {
                let src = chunk.column(0).unwrap().get_node_id(i).unwrap();
                let edge = chunk.column(1).unwrap().get_edge_id(i).unwrap();
                let dst = chunk.column(2).unwrap().get_node_id(i).unwrap();
                results.push((src, edge, dst));
            }
        }

        // Alix -> Gus, Alix -> Vincent
        assert_eq!(results.len(), 2);

        // All source nodes should be Alix
        for (src, _, _) in &results {
            assert_eq!(*src, alix);
        }

        // Target nodes should be Gus and Vincent
        let targets: Vec<NodeId> = results.iter().map(|(_, _, dst)| *dst).collect();
        assert!(targets.contains(&gus));
        assert!(targets.contains(&vincent));
    }

    #[test]
    fn test_expand_with_edge_type_filter() {
        let (store, dyn_store) = test_store();

        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["Person"]);
        let company = store.create_node(&["Company"]);

        store.create_edge(alix, gus, "KNOWS");
        store.create_edge(alix, company, "WORKS_AT");

        let scan = Box::new(ScanOperator::with_label(Arc::clone(&dyn_store), "Person"));

        let mut expand = ExpandOperator::new(
            Arc::clone(&dyn_store),
            scan,
            0,
            Direction::Outgoing,
            vec!["KNOWS".to_string()],
        );

        let mut results = Vec::new();
        while let Ok(Some(chunk)) = expand.next() {
            for i in 0..chunk.row_count() {
                let dst = chunk.column(2).unwrap().get_node_id(i).unwrap();
                results.push(dst);
            }
        }

        // Only KNOWS edges should be followed
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], gus);
    }

    #[test]
    fn test_expand_incoming() {
        let (store, dyn_store) = test_store();

        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["Person"]);

        store.create_edge(alix, gus, "KNOWS");

        // Scan Gus
        let scan = Box::new(ScanOperator::with_label(Arc::clone(&dyn_store), "Person"));

        let mut expand =
            ExpandOperator::new(Arc::clone(&dyn_store), scan, 0, Direction::Incoming, vec![]);

        let mut results = Vec::new();
        while let Ok(Some(chunk)) = expand.next() {
            for i in 0..chunk.row_count() {
                let src = chunk.column(0).unwrap().get_node_id(i).unwrap();
                let dst = chunk.column(2).unwrap().get_node_id(i).unwrap();
                results.push((src, dst));
            }
        }

        // Gus <- Alix (Gus's incoming edge from Alix)
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, gus); // source in the expand is Gus
        assert_eq!(results[0].1, alix); // target is Alix (who points to Gus)
    }

    #[test]
    fn test_expand_no_edges() {
        let (store, dyn_store) = test_store();

        store.create_node(&["Person"]);

        let scan = Box::new(ScanOperator::with_label(Arc::clone(&dyn_store), "Person"));

        let mut expand =
            ExpandOperator::new(Arc::clone(&dyn_store), scan, 0, Direction::Outgoing, vec![]);

        let result = expand.next().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_expand_reset() {
        let (store, dyn_store) = test_store();

        let a = store.create_node(&["Person"]);
        let b = store.create_node(&["Person"]);
        store.create_edge(a, b, "KNOWS");

        let scan = Box::new(ScanOperator::with_label(Arc::clone(&dyn_store), "Person"));
        let mut expand =
            ExpandOperator::new(Arc::clone(&dyn_store), scan, 0, Direction::Outgoing, vec![]);

        // First pass
        let mut count1 = 0;
        while let Ok(Some(chunk)) = expand.next() {
            count1 += chunk.row_count();
        }

        // Reset and run again
        expand.reset();
        let mut count2 = 0;
        while let Ok(Some(chunk)) = expand.next() {
            count2 += chunk.row_count();
        }

        assert_eq!(count1, count2);
        assert_eq!(count1, 1);
    }

    #[test]
    fn test_expand_name() {
        let (_store, dyn_store) = test_store();
        let scan = Box::new(ScanOperator::with_label(Arc::clone(&dyn_store), "Person"));
        let expand =
            ExpandOperator::new(Arc::clone(&dyn_store), scan, 0, Direction::Outgoing, vec![]);
        assert_eq!(expand.name(), "Expand");
    }

    #[test]
    fn test_expand_with_chunk_capacity() {
        let (store, dyn_store) = test_store();

        let a = store.create_node(&["Person"]);
        for _ in 0..5 {
            let b = store.create_node(&["Person"]);
            store.create_edge(a, b, "KNOWS");
        }

        let scan = Box::new(ScanOperator::with_label(Arc::clone(&dyn_store), "Person"));
        let mut expand =
            ExpandOperator::new(Arc::clone(&dyn_store), scan, 0, Direction::Outgoing, vec![])
                .with_chunk_capacity(2);

        // With capacity 2 and 5 edges from node a, we should get multiple chunks
        let mut total = 0;
        let mut chunk_count = 0;
        while let Ok(Some(chunk)) = expand.next() {
            chunk_count += 1;
            total += chunk.row_count();
        }

        assert_eq!(total, 5);
        assert!(
            chunk_count >= 2,
            "Expected multiple chunks with small capacity"
        );
    }

    #[test]
    fn test_expand_edge_type_case_insensitive() {
        let (store, dyn_store) = test_store();

        let a = store.create_node(&["Person"]);
        let b = store.create_node(&["Person"]);
        store.create_edge(a, b, "KNOWS");

        let scan = Box::new(ScanOperator::with_label(Arc::clone(&dyn_store), "Person"));
        let mut expand = ExpandOperator::new(
            Arc::clone(&dyn_store),
            scan,
            0,
            Direction::Outgoing,
            vec!["knows".to_string()], // lowercase
        );

        let mut count = 0;
        while let Ok(Some(chunk)) = expand.next() {
            count += chunk.row_count();
        }

        // Should match case-insensitively
        assert_eq!(count, 1);
    }

    #[test]
    fn test_expand_multiple_source_nodes() {
        let (store, dyn_store) = test_store();

        let a = store.create_node(&["Person"]);
        let b = store.create_node(&["Person"]);
        let c = store.create_node(&["Person"]);

        store.create_edge(a, c, "KNOWS");
        store.create_edge(b, c, "KNOWS");

        let scan = Box::new(ScanOperator::with_label(Arc::clone(&dyn_store), "Person"));
        let mut expand =
            ExpandOperator::new(Arc::clone(&dyn_store), scan, 0, Direction::Outgoing, vec![]);

        let mut results = Vec::new();
        while let Ok(Some(chunk)) = expand.next() {
            for i in 0..chunk.row_count() {
                let src = chunk.column(0).unwrap().get_node_id(i).unwrap();
                let dst = chunk.column(2).unwrap().get_node_id(i).unwrap();
                results.push((src, dst));
            }
        }

        // Both a->c and b->c
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_expand_empty_input() {
        let (_store, dyn_store) = test_store();

        // No nodes with this label
        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&dyn_store),
            "Nonexistent",
        ));
        let mut expand =
            ExpandOperator::new(Arc::clone(&dyn_store), scan, 0, Direction::Outgoing, vec![]);

        let result = expand.next().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_expand_into_any() {
        let (_store, dyn_store) = test_store();
        let scan = Box::new(ScanOperator::with_label(Arc::clone(&dyn_store), "Person"));
        let op = ExpandOperator::new(Arc::clone(&dyn_store), scan, 0, Direction::Outgoing, vec![]);
        let any = Box::new(op).into_any();
        assert!(any.downcast::<ExpandOperator>().is_ok());
    }
}
