//! Expand operator for relationship traversal.

use super::{Operator, OperatorError, OperatorResult};
use crate::execution::DataChunk;
use crate::graph::lpg::LpgStore;
use crate::graph::Direction;
use graphos_common::types::{EdgeId, LogicalType, NodeId};
use std::sync::Arc;

/// An expand operator that traverses edges from source nodes.
///
/// For each input row containing a source node, this operator produces
/// output rows for each neighbor connected via matching edges.
pub struct ExpandOperator {
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
}

impl ExpandOperator {
    /// Creates a new expand operator.
    pub fn new(
        store: Arc<LpgStore>,
        input: Box<dyn Operator>,
        source_column: usize,
        direction: Direction,
        edge_type: Option<String>,
    ) -> Self {
        Self {
            store,
            input,
            source_column,
            direction,
            edge_type,
            chunk_capacity: 2048,
            current_input: None,
            current_row: 0,
            current_edges: Vec::new(),
            current_edge_idx: 0,
            exhausted: false,
        }
    }

    /// Sets the chunk capacity.
    pub fn with_chunk_capacity(mut self, capacity: usize) -> Self {
        self.chunk_capacity = capacity;
        self
    }

    /// Loads the next input chunk.
    fn load_next_input(&mut self) -> Result<bool, OperatorError> {
        match self.input.next() {
            Ok(Some(chunk)) => {
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
        let chunk = match &self.current_input {
            Some(c) => c,
            None => return Ok(false),
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

        // Get edges from this node
        let edges: Vec<(NodeId, EdgeId)> = self
            .store
            .edges_from(source_id, self.direction)
            .filter(|(_, edge_id)| {
                // Filter by edge type if specified
                if let Some(ref filter_type) = self.edge_type {
                    if let Some(edge_type) = self.store.edge_type(*edge_id) {
                        edge_type.as_ref() == filter_type.as_str()
                    } else {
                        false
                    }
                } else {
                    true
                }
            })
            .collect();

        self.current_edges = edges;
        self.current_edge_idx = 0;
        Ok(true)
    }
}

impl Operator for ExpandOperator {
    fn next(&mut self) -> OperatorResult {
        if self.exhausted {
            return Ok(None);
        }

        // Output schema: [source_node, edge, target_node]
        let schema = [LogicalType::Node, LogicalType::Edge, LogicalType::Node];
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

            // Get the source node ID
            let source_id = self
                .current_input
                .as_ref()
                .and_then(|c| c.column(self.source_column))
                .and_then(|col| col.get_node_id(self.current_row))
                .ok_or_else(|| {
                    OperatorError::Execution("Failed to get source node ID".into())
                })?;

            // Get the current edge
            let (target_id, edge_id) = self.current_edges[self.current_edge_idx];

            // Add to output chunk
            // Columns 0, 1, 2 guaranteed to exist: chunk created with 3-column schema on line 138
            {
                // Source node
                let col = chunk
                    .column_mut(0)
                    .expect("column 0 exists: chunk created with 3-column schema");
                col.push_node_id(source_id);
            }
            {
                // Edge
                let col = chunk
                    .column_mut(1)
                    .expect("column 1 exists: chunk created with 3-column schema");
                col.push_edge_id(edge_id);
            }
            {
                // Target node
                let col = chunk
                    .column_mut(2)
                    .expect("column 2 exists: chunk created with 3-column schema");
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::operators::ScanOperator;

    #[test]
    fn test_expand_outgoing() {
        let store = Arc::new(LpgStore::new());

        // Create nodes
        let alice = store.create_node(&["Person"]);
        let bob = store.create_node(&["Person"]);
        let charlie = store.create_node(&["Person"]);

        // Create edges: Alice -> Bob, Alice -> Charlie
        store.create_edge(alice, bob, "KNOWS");
        store.create_edge(alice, charlie, "KNOWS");

        // Scan Alice only
        let scan = Box::new(ScanOperator::with_label(Arc::clone(&store), "Person"));

        let mut expand = ExpandOperator::new(
            Arc::clone(&store),
            scan,
            0, // source column
            Direction::Outgoing,
            None,
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

        // Alice -> Bob, Alice -> Charlie
        assert_eq!(results.len(), 2);

        // All source nodes should be Alice
        for (src, _, _) in &results {
            assert_eq!(*src, alice);
        }

        // Target nodes should be Bob and Charlie
        let targets: Vec<NodeId> = results.iter().map(|(_, _, dst)| *dst).collect();
        assert!(targets.contains(&bob));
        assert!(targets.contains(&charlie));
    }

    #[test]
    fn test_expand_with_edge_type_filter() {
        let store = Arc::new(LpgStore::new());

        let alice = store.create_node(&["Person"]);
        let bob = store.create_node(&["Person"]);
        let company = store.create_node(&["Company"]);

        store.create_edge(alice, bob, "KNOWS");
        store.create_edge(alice, company, "WORKS_AT");

        let scan = Box::new(ScanOperator::with_label(Arc::clone(&store), "Person"));

        let mut expand = ExpandOperator::new(
            Arc::clone(&store),
            scan,
            0,
            Direction::Outgoing,
            Some("KNOWS".to_string()),
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
        assert_eq!(results[0], bob);
    }

    #[test]
    fn test_expand_incoming() {
        let store = Arc::new(LpgStore::new());

        let alice = store.create_node(&["Person"]);
        let bob = store.create_node(&["Person"]);

        store.create_edge(alice, bob, "KNOWS");

        // Scan Bob
        let scan = Box::new(ScanOperator::with_label(Arc::clone(&store), "Person"));

        let mut expand = ExpandOperator::new(
            Arc::clone(&store),
            scan,
            0,
            Direction::Incoming,
            None,
        );

        let mut results = Vec::new();
        while let Ok(Some(chunk)) = expand.next() {
            for i in 0..chunk.row_count() {
                let src = chunk.column(0).unwrap().get_node_id(i).unwrap();
                let dst = chunk.column(2).unwrap().get_node_id(i).unwrap();
                results.push((src, dst));
            }
        }

        // Bob <- Alice (Bob's incoming edge from Alice)
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, bob); // source in the expand is Bob
        assert_eq!(results[0].1, alice); // target is Alice (who points to Bob)
    }

    #[test]
    fn test_expand_no_edges() {
        let store = Arc::new(LpgStore::new());

        store.create_node(&["Person"]);

        let scan = Box::new(ScanOperator::with_label(Arc::clone(&store), "Person"));

        let mut expand = ExpandOperator::new(
            Arc::clone(&store),
            scan,
            0,
            Direction::Outgoing,
            None,
        );

        let result = expand.next().unwrap();
        assert!(result.is_none());
    }
}
