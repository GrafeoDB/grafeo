//! Scan operator for reading data from storage.

use super::{Operator, OperatorError, OperatorResult, ReadCounter};
use crate::execution::DataChunk;
use crate::graph::GraphStore;
use grafeo_common::types::{LogicalType, NodeId, ReadView};
use std::sync::Arc;

/// A scan operator that reads nodes from storage.
pub struct ScanOperator {
    /// The store to scan from.
    store: Arc<dyn GraphStore>,
    /// The labels a node needs one of (empty: every node).
    labels: Vec<String>,
    /// Current position in the scan.
    position: usize,
    /// Batch of node IDs to scan.
    batch: Vec<NodeId>,
    /// Whether the scan is exhausted.
    exhausted: bool,
    /// Chunk capacity.
    chunk_capacity: usize,
    /// Who reads: the scan lists the nodes this view sees.
    view: ReadView,
    /// Counts the node IDs the scan reads.
    reads: Option<Arc<ReadCounter>>,
}

impl ScanOperator {
    /// Creates a scan of every node `view` sees.
    pub fn new(store: Arc<dyn GraphStore>, view: ReadView) -> Self {
        Self::with_any_label(store, Vec::new(), view)
    }

    /// Creates a scan of the nodes with `label` that `view` sees.
    pub fn with_label(
        store: Arc<dyn GraphStore>,
        label: impl Into<String>,
        view: ReadView,
    ) -> Self {
        Self::with_any_label(store, vec![label.into()], view)
    }

    /// Creates a scan of the nodes with at least one of `labels` (every
    /// node for none) that `view` sees, each once, in ID order.
    pub fn with_any_label(store: Arc<dyn GraphStore>, labels: Vec<String>, view: ReadView) -> Self {
        Self {
            store,
            labels,
            position: 0,
            batch: Vec::new(),
            exhausted: false,
            chunk_capacity: 2048,
            view,
            reads: None,
        }
    }

    /// Sets the chunk capacity.
    pub fn with_chunk_capacity(mut self, capacity: usize) -> Self {
        self.chunk_capacity = capacity;
        self
    }

    /// Counts the node IDs the scan reads in `reads`.
    #[must_use]
    pub fn with_read_counter(mut self, reads: Arc<ReadCounter>) -> Self {
        self.reads = Some(reads);
        self
    }

    /// Lists the nodes the view sees, or those with the labels as the view
    /// sees them, once.
    fn load_batch(&mut self) -> Result<(), OperatorError> {
        if !self.batch.is_empty() || self.exhausted {
            return Ok(());
        }
        let labels: Vec<&str> = self.labels.iter().map(String::as_str).collect();
        self.batch = match labels.as_slice() {
            [] => self.store.node_ids(self.view)?,
            [label] => self.store.nodes_by_label(label, self.view)?,
            labels => self.store.nodes_with_any_label(labels, self.view)?,
        };

        super::read_counter::count(self.reads.as_deref(), self.batch.len());
        if self.batch.is_empty() {
            self.exhausted = true;
        }
        Ok(())
    }
}

impl Operator for ScanOperator {
    fn next(&mut self) -> OperatorResult {
        self.load_batch()?;

        if self.exhausted || self.position >= self.batch.len() {
            return Ok(None);
        }

        // Create output chunk with node IDs
        let schema = [LogicalType::Node];
        let mut chunk = DataChunk::with_capacity(&schema, self.chunk_capacity);

        let end = (self.position + self.chunk_capacity).min(self.batch.len());
        let count = end - self.position;

        {
            // Column 0 guaranteed to exist: chunk created with single-column schema above
            let col = chunk
                .column_mut(0)
                .expect("column 0 exists: chunk created with single-column schema");
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

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphStoreMut;
    use crate::graph::rowgroup::RowGroupStore;

    #[test]
    fn test_scan_by_label() {
        let store: Arc<dyn GraphStoreMut> = Arc::new(RowGroupStore::new());

        store.create_node(&["Person"]);
        store.create_node(&["Person"]);
        store.create_node(&["Animal"]);

        let mut scan = ScanOperator::with_label(
            store.clone() as Arc<dyn GraphStore>,
            "Person",
            ReadView::latest(),
        );

        let chunk = scan.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 2);

        // Should be exhausted
        let next = scan.next().unwrap();
        assert!(next.is_none());
    }

    /// A scan of several labels returns each node once, in ID order: a node
    /// with both labels once, a node with neither not at all.
    #[test]
    fn a_scan_of_several_labels_returns_each_node_once() {
        let store: Arc<dyn GraphStoreMut> = Arc::new(RowGroupStore::new());
        let director = store.create_node(&["Director"]);
        store.create_node(&["City"]);
        let both = store.create_node(&["Person", "Director"]);
        let person = store.create_node(&["Person"]);

        let mut scan = ScanOperator::with_any_label(
            store.clone() as Arc<dyn GraphStore>,
            vec![
                "Person".to_string(),
                "Director".to_string(),
                "Missing".to_string(),
            ],
            ReadView::latest(),
        );
        let chunk = scan.next().unwrap().unwrap();
        let column = chunk.column(0).unwrap();
        let found: Vec<NodeId> = (0..chunk.row_count())
            .map(|row| column.get_node_id(row).unwrap())
            .collect();
        assert_eq!(found, [director, both, person]);
        assert!(scan.next().unwrap().is_none());
    }

    #[test]
    fn test_scan_reset() {
        let store: Arc<dyn GraphStoreMut> = Arc::new(RowGroupStore::new());
        store.create_node(&["Person"]);

        let mut scan = ScanOperator::with_label(
            store.clone() as Arc<dyn GraphStore>,
            "Person",
            ReadView::latest(),
        );

        // First scan
        let chunk1 = scan.next().unwrap().unwrap();
        assert_eq!(chunk1.row_count(), 1);

        // Reset
        scan.reset();

        // Second scan should work
        let chunk2 = scan.next().unwrap().unwrap();
        assert_eq!(chunk2.row_count(), 1);
    }

    #[test]
    fn test_full_scan() {
        let store: Arc<dyn GraphStoreMut> = Arc::new(RowGroupStore::new());

        // Create nodes with different labels
        store.create_node(&["Person"]);
        store.create_node(&["Person"]);
        store.create_node(&["Animal"]);
        store.create_node(&["Place"]);

        // Full scan (no label filter) should return all nodes
        let mut scan = ScanOperator::new(store.clone() as Arc<dyn GraphStore>, ReadView::latest());

        let chunk = scan.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 4, "Full scan should return all 4 nodes");

        // Should be exhausted
        let next = scan.next().unwrap();
        assert!(next.is_none());
    }

    #[test]
    fn test_scan_with_mvcc_context() {
        let store = Arc::new(RowGroupStore::new());

        // Two nodes, then a third: each create commits at its own epoch.
        store.create_node(&["Person"]);
        store.create_node(&["Person"]);
        let early = store.current_epoch();
        store.create_node(&["Person"]);
        let late = store.current_epoch();

        // A scan at the earlier epoch sees the first two nodes.
        let mut scan = ScanOperator::with_label(
            store.clone() as Arc<dyn GraphStore>,
            "Person",
            ReadView::new(early, None),
        );

        let chunk = scan.next().unwrap().unwrap();
        assert_eq!(
            chunk.row_count(),
            2,
            "Should see 2 nodes at the earlier epoch"
        );

        // A scan at the later epoch sees all three.
        let mut scan_all = ScanOperator::with_label(
            store.clone() as Arc<dyn GraphStore>,
            "Person",
            ReadView::new(late, None),
        );

        let chunk_all = scan_all.next().unwrap().unwrap();
        assert_eq!(
            chunk_all.row_count(),
            3,
            "Should see 3 nodes at the later epoch"
        );
    }

    #[test]
    fn test_scan_into_any() {
        let store: Arc<dyn GraphStoreMut> = Arc::new(RowGroupStore::new());
        let op = ScanOperator::with_label(
            store.clone() as Arc<dyn GraphStore>,
            "Person",
            ReadView::latest(),
        );
        let any = Box::new(op).into_any();
        assert!(any.downcast::<ScanOperator>().is_ok());
    }
}
