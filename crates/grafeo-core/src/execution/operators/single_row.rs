//! Single row, empty, and node list operators.
//!
//! - `SingleRowOperator`: Produces exactly one empty row. Used for queries like
//!   `UNWIND [1,2,3] AS x RETURN x` that don't have a MATCH clause.
//! - `EmptyOperator`: Produces no rows. Used when zone map pre-filtering
//!   determines that a filter predicate cannot match any data.
//! - `NodeListOperator`: Produces rows from a pre-computed list of node IDs.
//!   Used when property index lookups return a specific set of matching nodes.

use super::{Operator, OperatorResult};
use crate::execution::DataChunk;
use grafeo_common::types::{LogicalType, NodeId};

/// An operator that produces exactly one empty row.
///
/// This is useful for UNWIND clauses that operate on literal lists
/// without a prior MATCH clause.
pub struct SingleRowOperator {
    /// Whether the single row has been produced.
    produced: bool,
}

impl SingleRowOperator {
    /// Creates a new single row operator.
    #[must_use]
    pub fn new() -> Self {
        Self { produced: false }
    }
}

impl Default for SingleRowOperator {
    fn default() -> Self {
        Self::new()
    }
}

impl Operator for SingleRowOperator {
    fn next(&mut self) -> OperatorResult {
        if self.produced {
            return Ok(None);
        }

        self.produced = true;

        // Create a single row with no columns
        let mut chunk = DataChunk::with_capacity(&[], 1);
        chunk.set_count(1);

        Ok(Some(chunk))
    }

    fn reset(&mut self) {
        self.produced = false;
    }

    fn name(&self) -> &'static str {
        "SingleRow"
    }
}

/// An operator that produces no rows.
///
/// This is used when zone map pre-filtering determines that a filter
/// predicate cannot possibly match any data, allowing the entire scan
/// to be skipped.
pub struct EmptyOperator {
    /// The schema of the empty result (kept for debugging/explain output).
    #[allow(dead_code)]
    schema: Vec<LogicalType>,
}

impl EmptyOperator {
    /// Creates a new empty operator with the given schema.
    #[must_use]
    pub fn new(schema: Vec<LogicalType>) -> Self {
        Self { schema }
    }
}

impl Operator for EmptyOperator {
    fn next(&mut self) -> OperatorResult {
        // Always return None - no rows to produce
        Ok(None)
    }

    fn reset(&mut self) {
        // Nothing to reset
    }

    fn name(&self) -> &'static str {
        "Empty"
    }
}

/// An operator that produces rows from a pre-computed list of node IDs.
///
/// This is used when a property index lookup returns a specific set of matching
/// nodes, allowing O(1) lookups instead of full scans.
///
/// # Example
///
/// ```
/// use grafeo_core::execution::operators::NodeListOperator;
/// use grafeo_common::types::NodeId;
///
/// // Simulate nodes returned from property index lookup
/// let matching_nodes = vec![NodeId::new(1), NodeId::new(5), NodeId::new(10)];
///
/// let mut op = NodeListOperator::new(matching_nodes, 1024);
///
/// // First call returns chunk with matching nodes
/// // Subsequent calls return None when exhausted
/// ```
pub struct NodeListOperator {
    /// The list of node IDs to produce.
    nodes: Vec<NodeId>,
    /// Current position in the node list.
    position: usize,
    /// Number of nodes to produce per chunk.
    chunk_size: usize,
}

impl NodeListOperator {
    /// Creates a new node list operator with the given node IDs.
    #[must_use]
    pub fn new(nodes: Vec<NodeId>, chunk_size: usize) -> Self {
        Self {
            nodes,
            position: 0,
            chunk_size,
        }
    }
}

impl Operator for NodeListOperator {
    fn next(&mut self) -> OperatorResult {
        if self.position >= self.nodes.len() {
            return Ok(None);
        }

        let end = (self.position + self.chunk_size).min(self.nodes.len());
        let count = end - self.position;

        let schema = [LogicalType::Node];
        let mut chunk = DataChunk::with_capacity(&schema, self.chunk_size);

        {
            let col = chunk
                .column_mut(0)
                .expect("column 0 exists: chunk created with single-column schema");
            for i in self.position..end {
                col.push_node_id(self.nodes[i]);
            }
        }

        chunk.set_count(count);
        self.position = end;

        Ok(Some(chunk))
    }

    fn reset(&mut self) {
        self.position = 0;
    }

    fn name(&self) -> &'static str {
        "NodeList"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_row_operator() {
        let mut op = SingleRowOperator::new();

        // First call produces one row
        let chunk = op.next().unwrap();
        assert!(chunk.is_some());
        let chunk = chunk.unwrap();
        assert_eq!(chunk.row_count(), 1);

        // Second call produces None
        let chunk = op.next().unwrap();
        assert!(chunk.is_none());

        // After reset, produces one row again
        op.reset();
        let chunk = op.next().unwrap();
        assert!(chunk.is_some());
    }

    #[test]
    fn test_empty_operator() {
        let mut op = EmptyOperator::new(vec![LogicalType::Int64]);

        // Always returns None
        let chunk = op.next().unwrap();
        assert!(chunk.is_none());

        // After reset, still returns None
        op.reset();
        let chunk = op.next().unwrap();
        assert!(chunk.is_none());
    }

    #[test]
    fn test_node_list_operator() {
        let nodes = vec![NodeId::new(1), NodeId::new(5), NodeId::new(10)];
        let mut op = NodeListOperator::new(nodes, 2);

        // First call produces first 2 nodes
        let chunk = op.next().unwrap();
        assert!(chunk.is_some());
        let chunk = chunk.unwrap();
        assert_eq!(chunk.row_count(), 2);

        // Second call produces remaining 1 node
        let chunk = op.next().unwrap();
        assert!(chunk.is_some());
        let chunk = chunk.unwrap();
        assert_eq!(chunk.row_count(), 1);

        // Third call produces None (exhausted)
        let chunk = op.next().unwrap();
        assert!(chunk.is_none());

        // After reset, starts over
        op.reset();
        let chunk = op.next().unwrap();
        assert!(chunk.is_some());
        let chunk = chunk.unwrap();
        assert_eq!(chunk.row_count(), 2);
    }

    #[test]
    fn test_node_list_operator_empty() {
        let mut op = NodeListOperator::new(vec![], 10);

        // Empty list returns None immediately
        let chunk = op.next().unwrap();
        assert!(chunk.is_none());
    }
}
