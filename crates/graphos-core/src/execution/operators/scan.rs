//! Scan operator for reading data from storage.

use super::{Operator, OperatorError, OperatorResult};
use crate::execution::{DataChunk, ValueVector};
use crate::graph::lpg::LpgStore;
use graphos_common::types::{LogicalType, NodeId};
use std::sync::Arc;

/// A scan operator that reads nodes from storage.
pub struct ScanOperator {
    /// The store to scan from.
    store: Arc<LpgStore>,
    /// Label filter (None = all nodes).
    label: Option<String>,
    /// Current position in the scan.
    position: usize,
    /// Batch of node IDs to scan.
    batch: Vec<NodeId>,
    /// Whether the scan is exhausted.
    exhausted: bool,
    /// Chunk capacity.
    chunk_capacity: usize,
}

impl ScanOperator {
    /// Creates a new scan operator for all nodes.
    pub fn new(store: Arc<LpgStore>) -> Self {
        Self {
            store,
            label: None,
            position: 0,
            batch: Vec::new(),
            exhausted: false,
            chunk_capacity: 2048,
        }
    }

    /// Creates a new scan operator for nodes with a specific label.
    pub fn with_label(store: Arc<LpgStore>, label: impl Into<String>) -> Self {
        Self {
            store,
            label: Some(label.into()),
            position: 0,
            batch: Vec::new(),
            exhausted: false,
            chunk_capacity: 2048,
        }
    }

    /// Sets the chunk capacity.
    pub fn with_chunk_capacity(mut self, capacity: usize) -> Self {
        self.chunk_capacity = capacity;
        self
    }

    fn load_batch(&mut self) {
        if !self.batch.is_empty() || self.exhausted {
            return;
        }

        self.batch = match &self.label {
            Some(label) => self.store.nodes_by_label(label),
            None => {
                // For full scan, we'd need to iterate all nodes
                // This is a simplified implementation
                Vec::new()
            }
        };

        if self.batch.is_empty() {
            self.exhausted = true;
        }
    }
}

impl Operator for ScanOperator {
    fn next(&mut self) -> OperatorResult {
        self.load_batch();

        if self.exhausted || self.position >= self.batch.len() {
            return Ok(None);
        }

        // Create output chunk with node IDs
        let schema = [LogicalType::Node];
        let mut chunk = DataChunk::with_capacity(&schema, self.chunk_capacity);

        let end = (self.position + self.chunk_capacity).min(self.batch.len());
        let count = end - self.position;

        {
            let col = chunk.column_mut(0).unwrap();
            for i in self.position..end {
                col.push_node_id(self.batch[i]);
            }
        }

        chunk.set_count(count);
        self.position = end;

        Ok(Some(chunk))
    }

    fn reset(&mut self) {
        self.position = 0;
        self.batch.clear();
        self.exhausted = false;
    }

    fn name(&self) -> &'static str {
        "Scan"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_by_label() {
        let store = Arc::new(LpgStore::new());

        store.create_node(&["Person"]);
        store.create_node(&["Person"]);
        store.create_node(&["Animal"]);

        let mut scan = ScanOperator::with_label(Arc::clone(&store), "Person");

        let chunk = scan.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 2);

        // Should be exhausted
        let next = scan.next().unwrap();
        assert!(next.is_none());
    }

    #[test]
    fn test_scan_reset() {
        let store = Arc::new(LpgStore::new());
        store.create_node(&["Person"]);

        let mut scan = ScanOperator::with_label(Arc::clone(&store), "Person");

        // First scan
        let chunk1 = scan.next().unwrap().unwrap();
        assert_eq!(chunk1.row_count(), 1);

        // Reset
        scan.reset();

        // Second scan should work
        let chunk2 = scan.next().unwrap().unwrap();
        assert_eq!(chunk2.row_count(), 1);
    }
}
