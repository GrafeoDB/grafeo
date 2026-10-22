//! Range scan operator for property-bounded node scans.
//!
//! `RangeScanOperator` consumes
//! [`GraphStore::find_nodes_in_range`](crate::graph::GraphStore::find_nodes_in_range)
//! and emits `DataChunk`s of node ids whose property value falls within
//! `[min, max]` (with configurable inclusivity).
//!
//! ## Why a dedicated operator?
//!
//! The existing [`NodeListOperator`](super::single_row::NodeListOperator)
//! also chunks `Vec<NodeId>` into `DataChunk`s, but it loses the planner
//! signal that this scan is range-bounded. A dedicated operator:
//!
//! 1. Surfaces "range scan with per-block zone-map pruning" in EXPLAIN
//!    output so users can see the optimization fired.
//! 2. Owns the LIMIT-pushdown path (Phase 4e): when the planner knows a
//!    downstream LIMIT bound, the operator stops decoding rows after `n`
//!    matches without walking the rest of the column.
//! 3. Provides a stable seam for future enhancements (factorized output,
//!    parallel block scan) without churning the planner.
//!
//! ## Materialization strategy
//!
//! The operator materializes the lookup into a `Vec<NodeId>` on the first
//! `next()` call, then chunks. The optional limit set via
//! [`with_limit`](RangeScanOperator::with_limit) bounds what it keeps after
//! the label filter. Streaming chunk-by-chunk materialization needs a
//! cursor-based API on the store.
//!
//! ## Label and MVCC filtering
//!
//! The operator absorbs both filters: the label through
//! [`with_label_filter`](RangeScanOperator::with_label_filter), visibility
//! through the view it reads at, so the store's range lookup returns only
//! the nodes the view sees.

use std::sync::Arc;

use grafeo_common::types::{LogicalType, NodeId, ReadView, Value};
use grafeo_common::utils::hash::FxHashSet;

use super::{Operator, OperatorError, OperatorResult, ReadCounter};
use crate::execution::DataChunk;
use crate::graph::GraphStore;

/// Pull-based operator that emits node ids whose property value falls
/// within a range. See the module docs for details.
pub struct RangeScanOperator {
    store: Arc<dyn GraphStore>,
    property: String,
    min: Option<Value>,
    max: Option<Value>,
    min_inclusive: bool,
    max_inclusive: bool,
    chunk_capacity: usize,
    /// Optional row-count cap for LIMIT pushdown (Phase 4e).
    limit: Option<usize>,
    /// Optional label filter (only nodes of this label survive).
    label_filter: Option<String>,
    /// Who reads: the scan finds the nodes this view sees.
    view: ReadView,

    /// Materialized result, lazily built on first `next()`.
    materialized: Option<Vec<NodeId>>,
    /// Current cursor into `materialized`.
    position: usize,
    /// Counts the node IDs the range index yields.
    reads: Option<Arc<ReadCounter>>,
}

impl RangeScanOperator {
    /// Creates a range scan over `store` for the given property and bounds,
    /// of the nodes `view` sees.
    ///
    /// `chunk_capacity` is the number of rows per emitted `DataChunk`;
    /// the standard default in this codebase is 2048.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the range's parts and the view it reads at; the options are builders"
    )]
    pub fn new(
        store: Arc<dyn GraphStore>,
        property: impl Into<String>,
        min: Option<Value>,
        max: Option<Value>,
        min_inclusive: bool,
        max_inclusive: bool,
        chunk_capacity: usize,
        view: ReadView,
    ) -> Self {
        Self {
            store,
            property: property.into(),
            min,
            max,
            min_inclusive,
            max_inclusive,
            chunk_capacity,
            limit: None,
            label_filter: None,
            view,
            materialized: None,
            position: 0,
            reads: None,
        }
    }

    /// Counts the node IDs the range index yields in `reads`, before the
    /// label filter.
    #[must_use]
    pub fn with_read_counter(mut self, reads: Arc<ReadCounter>) -> Self {
        self.reads = Some(reads);
        self
    }

    /// Sets a row-count cap that bounds the materialization step: the scan
    /// keeps at most `limit` nodes of those that pass the label filter.
    /// Wired by the planner when a downstream `LIMIT k` is known statically.
    #[must_use]
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Returns the row-count cap set by [`with_limit`](Self::with_limit),
    /// or `None` if no cap is in effect. Used by planner tests to verify
    /// LIMIT pushdown wired the cap.
    #[must_use]
    pub fn limit(&self) -> Option<usize> {
        self.limit
    }

    /// Restricts the result to nodes carrying `label`.
    ///
    /// Applied during materialization by intersecting the iterator's
    /// output with the nodes of the label (as the view sees them, with
    /// one).
    #[must_use]
    pub fn with_label_filter(mut self, label: impl Into<String>) -> Self {
        self.label_filter = Some(label.into());
        self
    }

    fn ensure_materialized(&mut self) -> Result<(), OperatorError> {
        if self.materialized.is_some() {
            return Ok(());
        }

        // Pre-compute the label set once (avoids repeated lookups in the
        // hot path).
        let label_set: Option<FxHashSet<NodeId>> = match self.label_filter.as_ref() {
            Some(label) => Some(
                self.store
                    .nodes_by_label(label, self.view)?
                    .into_iter()
                    .collect(),
            ),
            None => None,
        };

        let (min, max) = (self.min.as_ref(), self.max.as_ref());
        let iter = self.store.find_nodes_in_range(
            &self.property,
            min,
            max,
            self.min_inclusive,
            self.max_inclusive,
            self.view,
        )?;

        // Filter inline by label and stop only after `limit` matches
        // *survive* the filter. Applying limit before filtering would
        // under-return rows when early range hits are filtered out.
        let mut collected: Vec<NodeId> = Vec::new();
        let mut read = 0;
        for id in iter {
            read += 1;
            if let Some(set) = label_set.as_ref()
                && !set.contains(&id)
            {
                continue;
            }
            collected.push(id);
            if let Some(n) = self.limit
                && collected.len() >= n
            {
                break;
            }
        }

        super::read_counter::count(self.reads.as_deref(), read);
        self.materialized = Some(collected);
        Ok(())
    }
}

impl Operator for RangeScanOperator {
    fn next(&mut self) -> OperatorResult {
        self.ensure_materialized()?;
        let nodes = self
            .materialized
            .as_ref()
            .expect("ensure_materialized populates Some");

        if self.position >= nodes.len() {
            return Ok(None);
        }

        // Guard against `chunk_capacity == 0`: that would set `end == position`,
        // emit empty chunks, and never advance — an infinite loop for callers.
        let step = self.chunk_capacity.max(1);
        let end = (self.position + step).min(nodes.len());
        let count = end - self.position;

        let schema = [LogicalType::Node];
        let mut chunk = DataChunk::with_capacity(&schema, step);
        {
            let col = chunk
                .column_mut(0)
                .expect("column 0 exists: chunk created with single-column schema");
            for i in self.position..end {
                col.push_node_id(nodes[i]);
            }
        }
        chunk.set_count(count);
        self.position = end;

        Ok(Some(chunk))
    }

    fn reset(&mut self) {
        self.position = 0;
        self.materialized = None;
    }

    fn name(&self) -> &'static str {
        "RangeScan"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::rowgroup::RowGroupStore;

    /// A store holding, per `(label, property, values)`, one node with that
    /// label for each value of the property.
    fn store_with(tables: &[(&str, &str, &[i64])]) -> Arc<dyn GraphStore> {
        let store = RowGroupStore::new();
        for (label, property, values) in tables {
            for value in *values {
                let id = store.create_node(&[label]);
                store.set_node_property(id, property, Value::Int64(*value));
            }
        }
        Arc::new(store)
    }

    fn build_person_store() -> Arc<dyn GraphStore> {
        store_with(&[("Person", "age", &[25, 30, 35, 40, 45])])
    }

    #[test]
    fn alix_range_scan_emits_matching_nodes() {
        let store = build_person_store();
        let mut op = RangeScanOperator::new(
            store,
            "age",
            Some(Value::Int64(30)),
            Some(Value::Int64(40)),
            true,
            true,
            2048,
            ReadView::latest(),
        );

        let chunk = op.next().unwrap().expect("first chunk should be Some");
        assert_eq!(chunk.row_count(), 3, "ages 30, 35, 40 match");
        let none = op.next().unwrap();
        assert!(none.is_none(), "single chunk fits all matches");
    }

    #[test]
    fn gus_range_scan_chunks_in_capacity_sized_batches() {
        let values: Vec<i64> = (0..100).collect();
        let store = store_with(&[("Big", "v", &values)]);

        let mut op =
            RangeScanOperator::new(store, "v", None, None, true, true, 10, ReadView::latest());

        let mut total = 0usize;
        let mut chunk_count = 0usize;
        while let Some(chunk) = op.next().unwrap() {
            chunk_count += 1;
            total += chunk.row_count();
            assert!(chunk.row_count() <= 10);
        }
        assert_eq!(total, 100);
        assert_eq!(chunk_count, 10);
    }

    #[test]
    fn vincent_range_scan_with_limit_short_circuits() {
        let values: Vec<i64> = (0..1000).collect();
        let store = store_with(&[("Big", "v", &values)]);

        let mut op =
            RangeScanOperator::new(store, "v", None, None, true, true, 64, ReadView::latest())
                .with_limit(5);

        let mut total = 0usize;
        while let Some(chunk) = op.next().unwrap() {
            total += chunk.row_count();
        }
        assert_eq!(total, 5, "limit caps the row count");
    }

    #[test]
    fn jules_range_scan_disjoint_range_yields_nothing() {
        let store = build_person_store();
        let mut op = RangeScanOperator::new(
            store,
            "age",
            Some(Value::Int64(100)),
            Some(Value::Int64(200)),
            true,
            true,
            2048,
            ReadView::latest(),
        );
        assert!(op.next().unwrap().is_none());
    }

    #[test]
    fn mia_range_scan_reset_replays_chunks() {
        let store = build_person_store();
        let mut op = RangeScanOperator::new(
            store,
            "age",
            Some(Value::Int64(25)),
            Some(Value::Int64(45)),
            true,
            true,
            2,
            ReadView::latest(),
        );

        let first_pass: Vec<usize> = std::iter::from_fn(|| op.next().unwrap())
            .map(|c| c.row_count())
            .collect();

        op.reset();

        let second_pass: Vec<usize> = std::iter::from_fn(|| op.next().unwrap())
            .map(|c| c.row_count())
            .collect();

        assert_eq!(first_pass, second_pass);
    }

    #[test]
    fn butch_range_scan_into_any_downcasts() {
        let store = build_person_store();
        let op = RangeScanOperator::new(
            store,
            "age",
            None,
            None,
            true,
            true,
            2048,
            ReadView::latest(),
        );
        let any = Box::new(op).into_any();
        assert!(any.downcast::<RangeScanOperator>().is_ok());
    }

    #[test]
    fn shosanna_range_scan_name_is_stable() {
        let store = build_person_store();
        let op = RangeScanOperator::new(
            store,
            "age",
            None,
            None,
            true,
            true,
            2048,
            ReadView::latest(),
        );
        assert_eq!(op.name(), "RangeScan");
    }

    #[test]
    fn hans_range_scan_with_label_filter_intersects() {
        // Two labels carry the same property name; label filter must
        // restrict results to one label only.
        let store = store_with(&[("A", "v", &[1, 2, 3]), ("B", "v", &[1, 2, 3])]);

        let mut op = RangeScanOperator::new(
            Arc::clone(&store),
            "v",
            None,
            None,
            true,
            true,
            2048,
            ReadView::latest(),
        )
        .with_label_filter("A");

        let chunk = op.next().unwrap().expect("at least one chunk");
        // Only nodes from label A survive; A has 3 rows.
        assert_eq!(chunk.row_count(), 3);

        // Sanity: without the label filter, we'd see 6 rows.
        let mut op_no_label =
            RangeScanOperator::new(store, "v", None, None, true, true, 2048, ReadView::latest());
        let chunk2 = op_no_label.next().unwrap().expect("at least one chunk");
        assert_eq!(chunk2.row_count(), 6);
    }

    #[test]
    fn beatrix_range_scan_label_filter_with_disjoint_label_yields_nothing() {
        let store = store_with(&[("A", "v", &[1, 2, 3])]);

        let mut op =
            RangeScanOperator::new(store, "v", None, None, true, true, 2048, ReadView::latest())
                .with_label_filter("Z");

        // Label "Z" doesn't exist; intersection is empty.
        assert!(op.next().unwrap().is_none());
    }

    #[test]
    fn django_range_scan_through_a_dyn_store_finds_every_match() {
        // The store reached as `Arc<dyn GraphStore>`: the trait
        // dispatch of `find_nodes_in_range_iter` finds every match.
        let store = build_person_store();
        let mut op = RangeScanOperator::new(
            Arc::clone(&store),
            "age",
            Some(Value::Int64(25)),
            Some(Value::Int64(45)),
            true,
            true,
            2048,
            ReadView::latest(),
        );
        let chunk = op.next().unwrap().expect("at least one chunk");
        assert_eq!(chunk.row_count(), 5);
    }
}
