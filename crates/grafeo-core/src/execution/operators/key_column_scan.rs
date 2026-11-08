//! A lookup that reads a key's column as it runs, one store read at a time.
//!
//! Each `next` reads until it has rows, and emits chunks of the usual size, so a `LIMIT` is bounded
//! by one read and the store's read lock is held per read: a row group, or a slice of one where its
//! values must be compared one by one. Indexes that hold part of the nodes are read once, on the
//! first `next`, and each read's ids merge with that read's remainder.

use std::sync::Arc;

use grafeo_common::types::{LogicalType, NodeId, ReadView, Value};

use super::{Operator, OperatorError, OperatorResult, Predicate, ReadCounter};
use crate::execution::{DataChunk, SelectionVector};
use crate::graph::GraphStore;
use crate::graph::rowgroup::KeyColumnQuery;

/// What decides the nodes a [`KeyColumnScanOperator`] reads, beside the view and the labels of the
/// scope the store has applied.
pub enum KeyColumnChecks {
    /// A predicate every node must meet.
    Always(Box<dyn Predicate>),
    /// `whole`, in a group whose answer is only maybe; in a group whose answer is exact, `rest`,
    /// which is `whole` without the `=` the lookup answers (`None`: nothing remains).
    UnlessExact {
        /// The whole predicate.
        whole: Box<dyn Predicate>,
        /// The predicate without the `=`.
        rest: Option<Box<dyn Predicate>>,
    },
}

/// The indexes that hold the nodes of the labels a [`KeyColumnQuery`] leaves out.
#[derive(Debug, Clone)]
pub struct KeyColumnIndexes {
    /// The labels whose indexes to read.
    pub labels: Vec<String>,
    /// The values to look up in each.
    pub values: Vec<Value>,
    /// The label every node needs, which an index read does not check.
    pub label: Option<String>,
}

/// See the module docs. Its end is named: #435 replaces this operator, and the store call stays.
pub struct KeyColumnScanOperator {
    store: Arc<dyn GraphStore>,
    query: KeyColumnQuery,
    view: ReadView,
    checks: KeyColumnChecks,
    indexes: Option<KeyColumnIndexes>,
    reads: Option<Arc<ReadCounter>>,
    chunk_size: usize,
    /// The id the next group's read starts from; `None` when none is left.
    from: Option<u64>,
    /// The ids the store had given out at the first read: a node the statement goes on to create
    /// is not read, as a scan lists its nodes once.
    limit: Option<u64>,
    /// What the indexes found, ascending, and how far it merged; `None` until read.
    indexed: Option<Vec<NodeId>>,
    merged: usize,
    /// The current group's nodes, how far they went out, and whether its answer is exact.
    nodes: Vec<NodeId>,
    at: usize,
    exact: bool,
}

impl KeyColumnScanOperator {
    /// A lookup of `query` at `view`, its nodes decided by `checks`.
    #[must_use]
    pub fn new(
        store: Arc<dyn GraphStore>,
        query: KeyColumnQuery,
        view: ReadView,
        checks: KeyColumnChecks,
    ) -> Self {
        Self {
            store,
            query,
            view,
            checks,
            indexes: None,
            reads: None,
            chunk_size: 2048,
            from: Some(0),
            limit: None,
            indexed: None,
            merged: 0,
            nodes: Vec::new(),
            at: 0,
            exact: false,
        }
    }

    /// Also reads `indexes`, which hold the nodes `query` leaves out.
    #[must_use]
    pub fn with_indexes(mut self, indexes: KeyColumnIndexes) -> Self {
        self.indexes = Some(indexes);
        self
    }

    /// Counts the rows it reads into `reads`.
    #[must_use]
    pub fn with_read_counter(mut self, reads: Arc<ReadCounter>) -> Self {
        self.reads = Some(reads);
        self
    }

    /// What the indexes hold for its values, ascending; an index gone since planning, or one that
    /// cannot answer as the statement compares (see [`super::find_indexed_as_compared`]), leaves
    /// its nodes to the column.
    fn read_indexes(&mut self) -> Result<Vec<NodeId>, OperatorError> {
        let Some(indexes) = &self.indexes else {
            return Ok(Vec::new());
        };
        let mut found = Vec::new();
        for label in &indexes.labels {
            for value in &indexes.values {
                let Some(nodes) = super::find_indexed_as_compared(
                    self.store.as_ref(),
                    &[label.as_str()],
                    self.query.property(),
                    value,
                    self.view,
                    self.query.equal_across_kinds(),
                )?
                else {
                    self.query.read_every_label();
                    return Ok(Vec::new());
                };
                super::read_counter::count(self.reads.as_deref(), nodes.len());
                found.extend(nodes);
            }
        }
        found.sort_unstable();
        found.dedup();
        Ok(found)
    }

    /// Whether the view sees `id`, an index's candidate, with the label every node needs.
    fn index_node_counts(&self, id: NodeId) -> Result<bool, OperatorError> {
        let label = self.indexes.as_ref().and_then(|i| i.label.as_deref());
        Ok(match label {
            Some(label) => self.store.node_has_label(id, label, self.view)? == Some(true),
            None => self.store.is_node_visible(id, self.view)?,
        })
    }

    /// Reads the next group into `nodes`; `false` when none is left.
    fn read_group(&mut self) -> Result<bool, OperatorError> {
        let Some(from) = self.from else {
            return Ok(false);
        };
        if self.indexed.is_none() {
            self.indexed = Some(self.read_indexes()?);
        }
        let group = self
            .store
            .key_column_group(&mut self.query, from, self.view)?;
        super::read_counter::count(self.reads.as_deref(), group.compared);
        let limit = *self.limit.get_or_insert(group.ids_given);
        let mut nodes = group.nodes;
        nodes.truncate(nodes.partition_point(|id| id.as_u64() < limit));
        self.exact = group.exact;
        self.from = group.end.filter(|end| *end < limit);

        // The index's candidates of this group, or all that are left after the last.
        let indexed = self.indexed.take().unwrap_or_default();
        let rest = &indexed[self.merged..];
        let count = self.from.map_or(rest.len(), |end| {
            rest.partition_point(|id| id.as_u64() < end)
        });
        let mut candidates = Vec::new();
        for id in &rest[..count] {
            if self.index_node_counts(*id)? {
                candidates.push(*id);
            }
        }
        self.merged += count;
        self.indexed = Some(indexed);
        if !candidates.is_empty() {
            nodes.extend(candidates);
            nodes.sort_unstable();
            nodes.dedup();
            self.exact = false;
        }
        self.nodes = nodes;
        self.at = 0;
        Ok(true)
    }
}

impl Operator for KeyColumnScanOperator {
    fn next(&mut self) -> OperatorResult {
        loop {
            if self.at >= self.nodes.len() {
                if !self.read_group()? {
                    return Ok(None);
                }
                continue;
            }
            let end = (self.at + self.chunk_size).min(self.nodes.len());
            let count = end - self.at;
            let mut chunk = DataChunk::with_capacity(&[LogicalType::Node], count);
            let column = chunk
                .column_mut(0)
                .expect("column 0 exists: chunk created with single-column schema");
            for id in &self.nodes[self.at..end] {
                column.push_node_id(*id);
            }
            chunk.set_count(count);
            self.at = end;

            let predicate = match &self.checks {
                KeyColumnChecks::Always(predicate) => Some(predicate),
                KeyColumnChecks::UnlessExact { whole, .. } if !self.exact => Some(whole),
                KeyColumnChecks::UnlessExact { rest, .. } => rest.as_ref(),
            };
            if let Some(predicate) = predicate {
                let selection = SelectionVector::try_from_predicate(count, |row| {
                    predicate.evaluate(&chunk, row)
                })?;
                if selection.is_empty() {
                    continue;
                }
                chunk.set_selection(selection);
            }
            return Ok(Some(chunk));
        }
    }

    fn reset(&mut self) {
        self.from = Some(0);
        self.limit = None;
        self.indexed = None;
        self.merged = 0;
        self.nodes.clear();
        self.at = 0;
        self.exact = false;
        self.query.forget_decoded();
    }

    fn name(&self) -> &'static str {
        "KeyColumnScan"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use grafeo_common::change::DataOp;
    use grafeo_common::types::{NodeId, ReadView, Value};

    use super::{KeyColumnChecks, KeyColumnScanOperator};
    use crate::execution::DataChunk;
    use crate::execution::operators::{Operator, OperatorError, Predicate};
    use crate::graph::apply::{GraphBackend, Writer};
    use crate::graph::conformance::{apply_all, labels, next_epoch, properties};
    use crate::graph::rowgroup::{KeyColumnQuery, RowGroupStore};

    struct Everything;

    impl Predicate for Everything {
        fn evaluate(&self, _chunk: &DataChunk, _row: usize) -> Result<bool, OperatorError> {
            Ok(true)
        }
    }

    /// Rows of any values, read by slices: `k` is 3, as an integer or a float, every other row; the
    /// string '3' between them is no 3 (GA04).
    fn store() -> Arc<RowGroupStore> {
        let store = RowGroupStore::new();
        let ops: Vec<DataOp> = (0..7_000_u64)
            .map(|id| {
                let k = match id % 4 {
                    0 => Value::Int64(3),
                    1 => Value::Float64(3.0),
                    2 => Value::from("3"),
                    _ => Value::Bool(true),
                };
                DataOp::CreateNode {
                    id: NodeId::new(id),
                    labels: labels(&["L"]),
                    properties: properties(&[("k", k)]),
                }
            })
            .collect();
        store.reserve_node_ids(7_000).unwrap();
        let epoch = next_epoch(&store);
        apply_all(&store, &ops, Writer::Replay { epoch }).unwrap();
        Arc::new(store)
    }

    fn scan(store: &Arc<RowGroupStore>) -> KeyColumnScanOperator {
        KeyColumnScanOperator::new(
            Arc::clone(store) as Arc<dyn crate::graph::GraphStore>,
            KeyColumnQuery::new(&["L"], &[], &[], "k", &[Value::Int64(3)]),
            ReadView::latest(),
            KeyColumnChecks::Always(Box::new(Everything)),
        )
    }

    fn drain(operator: &mut KeyColumnScanOperator) -> Vec<NodeId> {
        let mut nodes = Vec::new();
        while let Some(chunk) = operator.next().unwrap() {
            let column = chunk.column(0).unwrap();
            for row in chunk.selected_indices() {
                nodes.push(column.get_node_id(row).unwrap());
            }
        }
        nodes
    }

    /// `reset` in the middle of a read, or after its end, starts it over: same rows, same order.
    #[test]
    fn a_reset_mid_read_starts_over() {
        let store = store();
        let whole = drain(&mut scan(&store));
        assert_eq!(whole.len(), 3_500);
        let mut operator = scan(&store);
        assert!(operator.next().unwrap().is_some());
        operator.reset();
        assert_eq!(drain(&mut operator), whole, "after a first chunk");
        operator.reset();
        assert_eq!(drain(&mut operator), whole, "after the end");
        assert!(operator.next().unwrap().is_none(), "and it stays ended");
    }
}
