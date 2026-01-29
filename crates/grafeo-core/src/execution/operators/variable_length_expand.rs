//! Variable-length expand operator for multi-hop path traversal.

use super::{Operator, OperatorError, OperatorResult};
use crate::execution::DataChunk;
use crate::graph::Direction;
use crate::graph::lpg::LpgStore;
use grafeo_common::types::{EdgeId, EpochId, LogicalType, NodeId, TxId};
use std::collections::VecDeque;
use std::sync::Arc;

/// An expand operator that handles variable-length path patterns like `*1..3`.
///
/// For each input row containing a source node, this operator produces
/// output rows for each neighbor reachable within the hop range.
pub struct VariableLengthExpandOperator {
    /// The store to traverse.
    store: Arc<LpgStore>,
    /// Input operator providing source nodes.
    input: Box<dyn Operator>,
    /// Index of the source node column in input.
    source_column: usize,
    /// Direction of edge traversal.
    direction: Direction,
    /// Optional edge type filter.
    edge_type: Option<String>,
    /// Minimum number of hops.
    min_hops: u32,
    /// Maximum number of hops.
    max_hops: u32,
    /// Chunk capacity.
    chunk_capacity: usize,
    /// Transaction ID for MVCC visibility.
    tx_id: Option<TxId>,
    /// Epoch for version visibility.
    viewing_epoch: Option<EpochId>,
    /// Materialized input rows.
    input_rows: Option<Vec<InputRow>>,
    /// Current input row index.
    current_input_idx: usize,
    /// Output buffer for pending results.
    output_buffer: Vec<OutputRow>,
    /// Whether the operator is exhausted.
    exhausted: bool,
}

/// A materialized input row.
struct InputRow {
    /// All column values from the input.
    columns: Vec<ColumnValue>,
    /// The source node ID for expansion.
    source_node: NodeId,
}

/// A column value that can be node ID, edge ID, or generic value.
#[derive(Clone)]
enum ColumnValue {
    NodeId(NodeId),
    EdgeId(EdgeId),
    Value(grafeo_common::types::Value),
}

/// A ready output row.
struct OutputRow {
    /// Index into input_rows for the source row.
    input_idx: usize,
    /// The final edge in the path.
    edge_id: EdgeId,
    /// The target node.
    target_id: NodeId,
}

impl VariableLengthExpandOperator {
    /// Creates a new variable-length expand operator.
    pub fn new(
        store: Arc<LpgStore>,
        input: Box<dyn Operator>,
        source_column: usize,
        direction: Direction,
        edge_type: Option<String>,
        min_hops: u32,
        max_hops: u32,
    ) -> Self {
        Self {
            store,
            input,
            source_column,
            direction,
            edge_type,
            min_hops,
            max_hops: max_hops.max(min_hops), // Ensure max >= min
            chunk_capacity: 2048,
            tx_id: None,
            viewing_epoch: None,
            input_rows: None,
            current_input_idx: 0,
            output_buffer: Vec::new(),
            exhausted: false,
        }
    }

    /// Sets the chunk capacity.
    pub fn with_chunk_capacity(mut self, capacity: usize) -> Self {
        self.chunk_capacity = capacity;
        self
    }

    /// Sets the transaction context for MVCC visibility.
    pub fn with_tx_context(mut self, epoch: EpochId, tx_id: Option<TxId>) -> Self {
        self.viewing_epoch = Some(epoch);
        self.tx_id = tx_id;
        self
    }

    /// Materializes all input rows.
    fn materialize_input(&mut self) -> Result<(), OperatorError> {
        let mut rows = Vec::new();

        while let Some(mut chunk) = self.input.next()? {
            // Flatten to handle selection vectors
            chunk.flatten();

            for row_idx in 0..chunk.row_count() {
                // Extract the source node ID
                let col = chunk.column(self.source_column).ok_or_else(|| {
                    OperatorError::ColumnNotFound(format!(
                        "Column {} not found",
                        self.source_column
                    ))
                })?;

                let source_node = col.get_node_id(row_idx).ok_or_else(|| {
                    OperatorError::Execution("Expected node ID in source column".into())
                })?;

                // Materialize all columns
                let mut columns = Vec::with_capacity(chunk.column_count());
                for col_idx in 0..chunk.column_count() {
                    let col = chunk.column(col_idx).unwrap();
                    let value = if let Some(node_id) = col.get_node_id(row_idx) {
                        ColumnValue::NodeId(node_id)
                    } else if let Some(edge_id) = col.get_edge_id(row_idx) {
                        ColumnValue::EdgeId(edge_id)
                    } else if let Some(val) = col.get_value(row_idx) {
                        ColumnValue::Value(val)
                    } else {
                        ColumnValue::Value(grafeo_common::types::Value::Null)
                    };
                    columns.push(value);
                }

                rows.push(InputRow {
                    columns,
                    source_node,
                });
            }
        }

        self.input_rows = Some(rows);
        Ok(())
    }

    /// Gets edges from a node, respecting filters and visibility.
    fn get_edges(&self, node_id: NodeId) -> Vec<(NodeId, EdgeId)> {
        let epoch = self.viewing_epoch;
        let tx = self.tx_id.unwrap_or(TxId::SYSTEM);

        self.store
            .edges_from(node_id, self.direction)
            .filter(|(target_id, edge_id)| {
                // Filter by edge type if specified
                let type_matches = if let Some(ref filter_type) = self.edge_type {
                    if let Some(edge_type) = self.store.edge_type(*edge_id) {
                        edge_type.as_ref() == filter_type.as_str()
                    } else {
                        false
                    }
                } else {
                    true
                };

                if !type_matches {
                    return false;
                }

                // Filter by visibility if we have tx context
                if let Some(epoch) = epoch {
                    let edge_visible = self.store.get_edge_versioned(*edge_id, epoch, tx).is_some();
                    let target_visible = self
                        .store
                        .get_node_versioned(*target_id, epoch, tx)
                        .is_some();
                    edge_visible && target_visible
                } else {
                    true
                }
            })
            .collect()
    }

    /// Process one input row, generating all reachable outputs.
    fn process_input_row(&self, input_idx: usize, source_node: NodeId) -> Vec<OutputRow> {
        let mut results = Vec::new();

        // BFS from source node
        let mut frontier: VecDeque<(NodeId, u32, EdgeId)> = VecDeque::new();

        // Initialize frontier with immediate neighbors
        for (target, edge_id) in self.get_edges(source_node) {
            frontier.push_back((target, 1, edge_id));
        }

        // Process frontier
        while let Some((current_node, depth, edge_id)) = frontier.pop_front() {
            // If within the hop range, add to results
            if depth >= self.min_hops && depth <= self.max_hops {
                results.push(OutputRow {
                    input_idx,
                    edge_id,
                    target_id: current_node,
                });
            }

            // If we haven't reached max depth, continue expanding
            if depth < self.max_hops {
                for (target, next_edge_id) in self.get_edges(current_node) {
                    frontier.push_back((target, depth + 1, next_edge_id));
                }
            }
        }

        results
    }

    /// Fill the output buffer with results from the next input row.
    fn fill_output_buffer(&mut self) {
        let input_rows = match &self.input_rows {
            Some(rows) => rows,
            None => return,
        };

        while self.output_buffer.is_empty() && self.current_input_idx < input_rows.len() {
            let source_node = input_rows[self.current_input_idx].source_node;
            let results = self.process_input_row(self.current_input_idx, source_node);
            self.output_buffer.extend(results);
            self.current_input_idx += 1;
        }
    }
}

impl Operator for VariableLengthExpandOperator {
    fn next(&mut self) -> OperatorResult {
        if self.exhausted {
            return Ok(None);
        }

        // Materialize input on first call
        if self.input_rows.is_none() {
            self.materialize_input()?;
            if self.input_rows.as_ref().map_or(true, Vec::is_empty) {
                self.exhausted = true;
                return Ok(None);
            }
        }

        // Fill output buffer if empty
        self.fill_output_buffer();

        if self.output_buffer.is_empty() {
            self.exhausted = true;
            return Ok(None);
        }

        let input_rows = self.input_rows.as_ref().unwrap();

        // Build output chunk from buffer
        let num_input_cols = input_rows.first().map_or(0, |r| r.columns.len());

        // Schema: [input_columns..., edge, target]
        let mut schema: Vec<LogicalType> = Vec::with_capacity(num_input_cols + 2);
        if let Some(first_row) = input_rows.first() {
            for col_val in &first_row.columns {
                let ty = match col_val {
                    ColumnValue::NodeId(_) => LogicalType::Node,
                    ColumnValue::EdgeId(_) => LogicalType::Edge,
                    ColumnValue::Value(_) => LogicalType::Any,
                };
                schema.push(ty);
            }
        }
        schema.push(LogicalType::Edge);
        schema.push(LogicalType::Node);

        let mut chunk = DataChunk::with_capacity(&schema, self.chunk_capacity);

        // Take up to chunk_capacity rows from buffer
        let take_count = self.output_buffer.len().min(self.chunk_capacity);
        let to_output: Vec<_> = self.output_buffer.drain(..take_count).collect();

        for out_row in &to_output {
            let input_row = &input_rows[out_row.input_idx];

            // Copy input columns
            for (col_idx, col_val) in input_row.columns.iter().enumerate() {
                if let Some(out_col) = chunk.column_mut(col_idx) {
                    match col_val {
                        ColumnValue::NodeId(id) => out_col.push_node_id(*id),
                        ColumnValue::EdgeId(id) => out_col.push_edge_id(*id),
                        ColumnValue::Value(v) => out_col.push_value(v.clone()),
                    }
                }
            }

            // Add edge column
            if let Some(col) = chunk.column_mut(num_input_cols) {
                col.push_edge_id(out_row.edge_id);
            }

            // Add target node column
            if let Some(col) = chunk.column_mut(num_input_cols + 1) {
                col.push_node_id(out_row.target_id);
            }
        }

        chunk.set_count(to_output.len());
        Ok(Some(chunk))
    }

    fn reset(&mut self) {
        self.input.reset();
        self.input_rows = None;
        self.current_input_idx = 0;
        self.output_buffer.clear();
        self.exhausted = false;
    }

    fn name(&self) -> &'static str {
        "VariableLengthExpand"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::operators::ScanOperator;

    #[test]
    fn test_variable_length_expand_chain() {
        let store = Arc::new(LpgStore::new());

        // Create chain: a -> b -> c -> d
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        let d = store.create_node(&["Node"]);

        store.set_node_property(a, "name", "a".into());
        store.set_node_property(b, "name", "b".into());
        store.set_node_property(c, "name", "c".into());
        store.set_node_property(d, "name", "d".into());

        store.create_edge(a, b, "NEXT");
        store.create_edge(b, c, "NEXT");
        store.create_edge(c, d, "NEXT");

        // Create scan for all nodes
        let scan = Box::new(ScanOperator::with_label(Arc::clone(&store), "Node"));

        // Expand 1-3 hops from all nodes
        let mut expand = VariableLengthExpandOperator::new(
            Arc::clone(&store),
            scan,
            0,
            Direction::Outgoing,
            Some("NEXT".to_string()),
            1,
            3,
        );

        let mut results = Vec::new();
        while let Ok(Some(chunk)) = expand.next() {
            for i in 0..chunk.row_count() {
                let src = chunk.column(0).unwrap().get_node_id(i).unwrap();
                let dst = chunk.column(2).unwrap().get_node_id(i).unwrap();
                results.push((src, dst));
            }
        }

        // From 'a', we should reach b (1 hop), c (2 hops), d (3 hops)
        let a_targets: Vec<NodeId> = results
            .iter()
            .filter(|(s, _)| *s == a)
            .map(|(_, t)| *t)
            .collect();
        assert!(a_targets.contains(&b), "a should reach b");
        assert!(a_targets.contains(&c), "a should reach c");
        assert!(a_targets.contains(&d), "a should reach d");
        assert_eq!(a_targets.len(), 3, "a should reach exactly 3 nodes");
    }

    #[test]
    fn test_variable_length_expand_min_hops() {
        let store = Arc::new(LpgStore::new());

        // Create chain: a -> b -> c
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);

        store.create_edge(a, b, "NEXT");
        store.create_edge(b, c, "NEXT");

        let scan = Box::new(ScanOperator::with_label(Arc::clone(&store), "Node"));

        // Expand 2-3 hops only (skip 1 hop)
        let mut expand = VariableLengthExpandOperator::new(
            Arc::clone(&store),
            scan,
            0,
            Direction::Outgoing,
            Some("NEXT".to_string()),
            2, // min 2 hops
            3, // max 3 hops
        );

        let mut results = Vec::new();
        while let Ok(Some(chunk)) = expand.next() {
            for i in 0..chunk.row_count() {
                let src = chunk.column(0).unwrap().get_node_id(i).unwrap();
                let dst = chunk.column(2).unwrap().get_node_id(i).unwrap();
                results.push((src, dst));
            }
        }

        // From 'a', we should reach c (2 hops) but NOT b (1 hop)
        let a_targets: Vec<NodeId> = results
            .iter()
            .filter(|(s, _)| *s == a)
            .map(|(_, t)| *t)
            .collect();
        assert!(
            !a_targets.contains(&b),
            "a should NOT reach b with min_hops=2"
        );
        assert!(a_targets.contains(&c), "a should reach c");
    }
}
