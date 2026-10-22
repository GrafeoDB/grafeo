//! Range scan operator for property-bounded node scans.
//!
//! `RangeScanOperator` consumes
//! [`GraphStoreSearch::find_nodes_in_range_iter`](crate::graph::GraphStoreSearch::find_nodes_in_range_iter)
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
//! Phase 4c materializes the iterator into a `Vec<NodeId>` on the first
//! `next()` call, then chunks. Block-level skip pruning still happens
//! during iterator construction, so the architectural value is intact.
//! The materialization step is bounded by the optional limit set via
//! [`with_limit`](Self::with_limit) (Phase 4e). Streaming chunk-by-chunk
//! materialization is a future pass; it requires either a self-referential
//! struct or a cursor-based API on the store, neither of which is free
//! in safe Rust today.
//!
//! ## Label and MVCC filtering
//!
//! The planner used to filter the eager `Vec<NodeId>` after the range
//! lookup. The operator absorbs both filters via
//! [`with_label_filter`](Self::with_label_filter) and
//! [`with_transaction_context`](Self::with_transaction_context), preserving
//! the existing semantics while keeping the `RangeScanOperator` as the
//! single entry point.

use std::sync::Arc;

use grafeo_common::types::{EpochId, LogicalType, NodeId, TransactionId, Value};
use grafeo_common::utils::hash::FxHashSet;

use super::{Operator, OperatorResult};
use crate::execution::DataChunk;
use crate::graph::GraphStoreSearch;

/// Pull-based operator that emits node ids whose property value falls
/// within a range. See the module docs for details.
pub struct RangeScanOperator {
    store: Arc<dyn GraphStoreSearch>,
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
    /// Optional MVCC transaction context (epoch + tx).
    transaction_context: Option<(EpochId, TransactionId)>,

    /// Materialized result, lazily built on first `next()`.
    materialized: Option<Vec<NodeId>>,
    /// Current cursor into `materialized`.
    position: usize,
}

impl RangeScanOperator {
    /// Creates a range scan over `store` for the given property and bounds.
    ///
    /// `chunk_capacity` is the number of rows per emitted `DataChunk`;
    /// the standard default in this codebase is 2048.
    #[must_use]
    pub fn new(
        store: Arc<dyn GraphStoreSearch>,
        property: impl Into<String>,
        min: Option<Value>,
        max: Option<Value>,
        min_inclusive: bool,
        max_inclusive: bool,
        chunk_capacity: usize,
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
            transaction_context: None,
            materialized: None,
            position: 0,
        }
    }

    /// Sets a row-count cap that bounds the materialization step.
    ///
    /// When set, the underlying iterator is consumed via `take(limit)`,
    /// so blocks past the cap are never decoded. Wired by the planner
    /// in Phase 4e when a downstream `LIMIT k` is known statically.
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
    /// output with `store.nodes_by_label(label)`. Mirrors the eager
    /// `find_nodes_in_range` + `nodes_by_label` retain pattern that the
    /// planner used pre-Phase-4d.
    #[must_use]
    pub fn with_label_filter(mut self, label: impl Into<String>) -> Self {
        self.label_filter = Some(label.into());
        self
    }

    /// Filters results by MVCC visibility at the given epoch and tx.
    ///
    /// Applied during materialization via `store.get_node_versioned`.
    /// Required for any planner-emitted scan: the existing `plan_range_filter`
    /// always applies this filter, and the operator preserves that.
    #[must_use]
    pub fn with_transaction_context(
        mut self,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Self {
        self.transaction_context = Some((epoch, transaction_id));
        self
    }

    fn ensure_materialized(&mut self) {
        if self.materialized.is_some() {
            return;
        }

        // Pre-compute the label set once (avoids repeated lookups in the
        // hot path).
        let label_set: Option<FxHashSet<NodeId>> = self
            .label_filter
            .as_ref()
            .map(|label| self.store.nodes_by_label(label).into_iter().collect());
        let tx_ctx = self.transaction_context;

        let iter = self.store.find_nodes_in_range_iter(
            &self.property,
            self.min.as_ref(),
            self.max.as_ref(),
            self.min_inclusive,
            self.max_inclusive,
        );

        // Filter inline (label + MVCC) and stop only after `limit` matches
        // *survive* the filters. Applying limit before filtering would
        // under-return rows when early range hits are filtered out.
        let mut collected: Vec<NodeId> = Vec::new();
        for id in iter {
            if let Some(set) = label_set.as_ref()
                && !set.contains(&id)
            {
                continue;
            }
            if let Some((epoch, tx)) = tx_ctx
                && self.store.get_node_versioned(id, epoch, tx).is_none()
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

        self.materialized = Some(collected);
    }
}

impl Operator for RangeScanOperator {
    fn next(&mut self) -> OperatorResult {
        self.ensure_materialized();
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

#[cfg(all(test, feature = "lpg"))]
mod tests {
    use super::*;
    use crate::graph::lpg::LpgStore;

    /// A store holding, per `(label, property, values)`, one node with that
    /// label for each value of the property.
    fn store_with(tables: &[(&str, &str, &[i64])]) -> Arc<dyn GraphStoreSearch> {
        let store = LpgStore::new().unwrap();
        for (label, property, values) in tables {
            for value in *values {
                let id = store.create_node(&[label]);
                store.set_node_property(id, property, Value::Int64(*value));
            }
        }
        Arc::new(store)
    }

    fn build_person_store() -> Arc<dyn GraphStoreSearch> {
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

        let mut op = RangeScanOperator::new(store, "v", None, None, true, true, 10);

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

        let mut op = RangeScanOperator::new(store, "v", None, None, true, true, 64).with_limit(5);

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
        let op = RangeScanOperator::new(store, "age", None, None, true, true, 2048);
        let any = Box::new(op).into_any();
        assert!(any.downcast::<RangeScanOperator>().is_ok());
    }

    #[test]
    fn shosanna_range_scan_name_is_stable() {
        let store = build_person_store();
        let op = RangeScanOperator::new(store, "age", None, None, true, true, 2048);
        assert_eq!(op.name(), "RangeScan");
    }

    #[test]
    fn hans_range_scan_with_label_filter_intersects() {
        // Two labels carry the same property name; label filter must
        // restrict results to one label only.
        let store = store_with(&[("A", "v", &[1, 2, 3]), ("B", "v", &[1, 2, 3])]);

        let mut op = RangeScanOperator::new(Arc::clone(&store), "v", None, None, true, true, 2048)
            .with_label_filter("A");

        let chunk = op.next().unwrap().expect("at least one chunk");
        // Only nodes from label A survive; A has 3 rows.
        assert_eq!(chunk.row_count(), 3);

        // Sanity: without the label filter, we'd see 6 rows.
        let mut op_no_label = RangeScanOperator::new(store, "v", None, None, true, true, 2048);
        let chunk2 = op_no_label.next().unwrap().expect("at least one chunk");
        assert_eq!(chunk2.row_count(), 6);
    }

    #[test]
    fn beatrix_range_scan_label_filter_with_disjoint_label_yields_nothing() {
        let store = store_with(&[("A", "v", &[1, 2, 3])]);

        let mut op =
            RangeScanOperator::new(store, "v", None, None, true, true, 2048).with_label_filter("Z");

        // Label "Z" doesn't exist; intersection is empty.
        assert!(op.next().unwrap().is_none());
    }

    #[test]
    fn django_range_scan_through_a_dyn_store_finds_every_match() {
        // The store reached as `Arc<dyn GraphStoreSearch>`: the trait
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
        );
        let chunk = op.next().unwrap().expect("at least one chunk");
        assert_eq!(chunk.row_count(), 5);
    }

    /// The lazy `find_nodes_in_range_iter` of an `Arc<dyn GraphStoreSearch>`
    /// yields the same set as the eager `find_nodes_in_range`.
    #[test]
    fn tarantino_dyn_dispatch_yields_same_results_as_eager() {
        let store = build_person_store();
        let min = Value::Int64(30);
        let max = Value::Int64(40);
        let lazy: Vec<NodeId> = store
            .find_nodes_in_range_iter("age", Some(&min), Some(&max), true, true)
            .collect();
        let mut eager = store.find_nodes_in_range("age", Some(&min), Some(&max), true, true);
        let mut lazy_sorted = lazy;
        lazy_sorted.sort_unstable();
        eager.sort_unstable();
        assert_eq!(lazy_sorted, eager);
    }
}
