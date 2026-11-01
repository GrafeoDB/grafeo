//! The count of a label's nodes, read from the store without a scan.
//!
//! `MATCH (n:Graph) RETURN count(*)` needs no node: the store keeps the
//! count of each label's nodes, and answers it for the query's view (from
//! the kept count for a reader at the store's epoch, by a scan of the
//! label for any other view, see [`GraphStore::nodes_by_label_count`]).

use std::sync::Arc;

use grafeo_common::types::{LogicalType, ReadView};

use super::{Operator, OperatorResult, ReadCounter};
use crate::execution::DataChunk;
use crate::execution::vector::ValueVector;
use crate::graph::GraphStore;

/// One row of `columns` integer columns, each the number of nodes with
/// `label` the view sees: the counts of `count(*)` and `count(n)` over a
/// scan of the label. A count the store read the nodes for counts them as
/// read (see [`with_read_counter`](Self::with_read_counter)).
pub struct LabelCountOperator {
    store: Arc<dyn GraphStore>,
    label: String,
    view: ReadView,
    columns: usize,
    produced: bool,
    reads: Option<Arc<ReadCounter>>,
}

impl LabelCountOperator {
    /// The count of `label`'s nodes `view` sees, in each of `columns`
    /// columns of one row.
    #[must_use]
    pub fn new(store: Arc<dyn GraphStore>, label: String, view: ReadView, columns: usize) -> Self {
        Self {
            store,
            label,
            view,
            columns,
            produced: false,
            reads: None,
        }
    }

    /// Counts the nodes the store reads for the count in `reads`: none when
    /// it keeps the count, the label's nodes otherwise (a transaction's own
    /// writes, a past epoch).
    #[must_use]
    pub fn with_read_counter(mut self, reads: Arc<ReadCounter>) -> Self {
        self.reads = Some(reads);
        self
    }
}

impl Operator for LabelCountOperator {
    fn next(&mut self) -> OperatorResult {
        if self.produced {
            return Ok(None);
        }
        self.produced = true;
        let kept = self.store.label_count_is_kept(&self.label, self.view);
        let count = self.store.nodes_by_label_count(&self.label, self.view)?;
        if !kept {
            super::read_counter::count(self.reads.as_deref(), count);
        }
        let count = i64::try_from(count).unwrap_or(i64::MAX);
        let columns = (0..self.columns)
            .map(|_| {
                let mut column = ValueVector::with_capacity(LogicalType::Int64, 1);
                column.push_int64(count);
                column
            })
            .collect();
        Ok(Some(DataChunk::new(columns)))
    }

    fn reset(&mut self) {
        self.produced = false;
    }

    fn name(&self) -> &'static str {
        "LabelCount"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}
