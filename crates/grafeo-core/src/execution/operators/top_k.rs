//! Streaming bounded-heap top-K operator.
//!
//! Subsumes `Limit ← Sort` for the `LIMIT k ORDER BY ...` pattern: instead of
//! materializing every input row, sorting them all, and discarding all but the
//! first k, this operator maintains a max-heap of size k keyed by the user's
//! sort tuple (with the comparator inverted so `peek()` returns the worst-by-
//! user-order). For input cardinality N, memory is O(k) and comparisons are
//! O(N log k). The heap is drained in user-requested order via
//! `BinaryHeap::into_sorted_vec`; no separate sort step is needed.
//!
//! Stability matches `Vec::sort_by`: rows tied on every sort key are output in
//! input order, achieved with a monotonic insertion-id tiebreaker.
//!
//! See `plan_limit` in `grafeo-engine` for the dispatch point that builds this
//! operator. PROFILE-mode plans bypass the rewrite (entry-count parity with the
//! logical tree); `LimitOperator(SortOperator(...))` runs instead.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;

use grafeo_common::types::{LogicalType, Value};

use super::sort::SortKey;
use super::value_utils::compare_values_with_nulls;
use super::{Operator, OperatorResult};
use crate::execution::DataChunk;
use crate::execution::chunk::DataChunkBuilder;

/// Streaming bounded top-K operator. See module docs.
pub struct TopKOperator {
    child: Box<dyn Operator>,
    /// Shared with every HeapEntry via Arc so HeapEntry::Ord can compare
    /// without raw pointers (`unsafe_code = "deny"` workspace-wide). The Arc
    /// is allocated once in `new` and refcount-bumped per heap insertion;
    /// the marginal cost is negligible at k=50, N=1M.
    sort_keys: Arc<Vec<SortKey>>,
    limit: usize,
    output_schema: Vec<LogicalType>,
    state: TopKState,
    #[cfg(test)]
    materialized_rows: std::sync::atomic::AtomicUsize,
}

enum TopKState {
    Building {
        heap: BinaryHeap<HeapEntry>,
        next_insertion_id: u64,
    },
    Draining {
        rows: Vec<HeapEntry>,
        position: usize,
    },
    Done,
}

struct HeapEntry {
    sort_values: Vec<Option<Value>>,
    row_values: Vec<Option<Value>>,
    insertion_id: u64,
    /// Shared with the owning operator. Refcount-bumped per insertion.
    sort_keys: Arc<Vec<SortKey>>,
}

impl TopKOperator {
    /// Constructs a streaming bounded top-k operator that yields the first
    /// `limit` rows of `child` in `sort_keys` order, using O(limit) memory
    /// regardless of `child`'s cardinality.
    ///
    /// Equivalent in output to `LimitOperator(SortOperator(child, sort_keys), limit)`,
    /// including stability on ties.
    ///
    /// # Example
    ///
    /// ```
    /// use grafeo_core::execution::DataChunk;
    /// use grafeo_core::execution::chunk::DataChunkBuilder;
    /// use grafeo_core::execution::operators::{Operator, OperatorResult, SortKey, TopKOperator};
    /// use grafeo_common::types::LogicalType;
    ///
    /// struct Source { chunk: Option<DataChunk> }
    /// impl Operator for Source {
    ///     fn next(&mut self) -> OperatorResult { Ok(self.chunk.take()) }
    ///     fn reset(&mut self) {}
    ///     fn name(&self) -> &'static str { "Source" }
    ///     fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> { self }
    /// }
    ///
    /// let mut b = DataChunkBuilder::new(&[LogicalType::Int64]);
    /// for v in [10i64, 50, 30, 20, 40] {
    ///     b.column_mut(0).unwrap().push_int64(v);
    ///     b.advance_row();
    /// }
    /// let source = Source { chunk: Some(b.finish()) };
    ///
    /// let mut top_k = TopKOperator::new(
    ///     Box::new(source),
    ///     vec![SortKey::descending(0)],
    ///     3,
    ///     vec![LogicalType::Int64],
    /// );
    ///
    /// let chunk = top_k.next().unwrap().unwrap();
    /// let mut out = vec![];
    /// for row in chunk.selected_indices() {
    ///     out.push(chunk.column(0).unwrap().get_int64(row).unwrap());
    /// }
    /// assert_eq!(out, vec![50, 40, 30]);
    /// ```
    pub fn new(
        child: Box<dyn Operator>,
        sort_keys: Vec<SortKey>,
        limit: usize,
        output_schema: Vec<LogicalType>,
    ) -> Self {
        Self {
            child,
            sort_keys: Arc::new(sort_keys),
            limit,
            output_schema,
            state: TopKState::Building {
                heap: BinaryHeap::new(),
                next_insertion_id: 0,
            },
            #[cfg(test)]
            materialized_rows: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Decomposes this operator into its child, sort keys, and limit for
    /// future push-based pipeline conversion. Mirrors
    /// `SortOperator::into_parts` and `LimitOperator::into_parts`.
    ///
    /// `pipeline_convert.rs` does not currently call this — TopK has no
    /// push variant in Phase 1 — but exposing the decomposition keeps the
    /// API surface uniform for future work.
    pub fn into_parts(self) -> (Box<dyn Operator>, Vec<SortKey>, usize) {
        // Arc::try_unwrap succeeds when no HeapEntry holds a clone (which is
        // the case when the operator hasn't been pulled, or after
        // into_sorted_vec drained the heap into Done). Defensive fallback
        // clones the contents if any entry happens to outlive (it shouldn't).
        let sort_keys = Arc::try_unwrap(self.sort_keys)
            .unwrap_or_else(|arc| (*arc).clone());
        (self.child, sort_keys, self.limit)
    }
}

impl Operator for TopKOperator {
    fn next(&mut self) -> OperatorResult {
        // Phase 1: build the heap by draining the child once.
        if matches!(self.state, TopKState::Building { .. }) {
            let TopKState::Building { mut heap, mut next_insertion_id } =
                std::mem::replace(&mut self.state, TopKState::Done)
            else {
                unreachable!()
            };

            while let Some(chunk) = self.child.next()? {
                for row_idx in chunk.selected_indices() {
                    let new_sort_values =
                        extract_sort_values(&chunk, row_idx, self.sort_keys.as_slice());

                    let should_push = if heap.len() < self.limit {
                        true
                    } else if let Some(top) = heap.peek() {
                        row_beats_heap_top(&new_sort_values, top, self.sort_keys.as_slice())
                    } else {
                        // limit == 0 — heap stays empty, never push.
                        false
                    };

                    if !should_push {
                        continue;
                    }

                    let row_values =
                        extract_row_values(&chunk, row_idx, self.output_schema.len());
                    #[cfg(test)]
                    self.materialized_rows.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let entry = HeapEntry {
                        sort_values: new_sort_values,
                        row_values,
                        insertion_id: next_insertion_id,
                        sort_keys: Arc::clone(&self.sort_keys),
                    };
                    next_insertion_id += 1;
                    if heap.len() < self.limit {
                        heap.push(entry);
                    } else {
                        // Heap is full and the new entry beat the worst; replace
                        // the heap's max in place. One sift-down vs push+pop's
                        // two reheapifies — significant for large N.
                        // peek_mut is None only for empty heap; we know
                        // heap.len() == self.limit > 0 here (limit==0 is
                        // filtered above by should_push=false).
                        if let Some(mut top) = heap.peek_mut() {
                            *top = entry;
                        }
                    }
                }
            }

            let rows = heap.into_sorted_vec();
            self.state = TopKState::Draining { rows, position: 0 };
        }

        // Phase 2: drain — one DataChunk per call.
        if let TopKState::Draining { rows, position } = &mut self.state {
            if *position < rows.len() {
                let mut builder = DataChunkBuilder::with_capacity(&self.output_schema, 2048);
                while *position < rows.len() && !builder.is_full() {
                    let entry = &rows[*position];
                    for col_idx in 0..self.output_schema.len() {
                        if let Some(dst_col) = builder.column_mut(col_idx) {
                            let val = entry
                                .row_values
                                .get(col_idx)
                                .and_then(|v| v.clone())
                                .unwrap_or(Value::Null);
                            dst_col.push_value(val);
                        }
                    }
                    builder.advance_row();
                    *position += 1;
                }
                if builder.row_count() > 0 {
                    return Ok(Some(builder.finish()));
                }
            }
            // Drain exhausted (or rows was empty for k=0). Transition to Done.
            self.state = TopKState::Done;
        }

        // Phase 3: done.
        Ok(None)
    }

    fn reset(&mut self) {
        self.child.reset();
        self.state = TopKState::Building {
            heap: BinaryHeap::new(),
            next_insertion_id: 0,
        };
        #[cfg(test)]
        self.materialized_rows.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    fn name(&self) -> &'static str {
        "TopK"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

#[cfg(test)]
impl TopKOperator {
    pub(crate) fn materialized_rows(&self) -> usize {
        self.materialized_rows.load(std::sync::atomic::Ordering::Relaxed)
    }
}

fn extract_sort_values(
    chunk: &DataChunk,
    row_idx: usize,
    sort_keys: &[SortKey],
) -> Vec<Option<Value>> {
    sort_keys
        .iter()
        .map(|k| chunk.column(k.column).and_then(|c| c.get_value(row_idx)))
        .collect()
}

fn extract_row_values(chunk: &DataChunk, row_idx: usize, n_cols: usize) -> Vec<Option<Value>> {
    (0..n_cols)
        .map(|i| chunk.column(i).and_then(|c| c.get_value(row_idx)))
        .collect()
}

/// Strict better-than test: does `new` beat the current heap top per user-requested order?
/// Inserting a new row that ties on every key must NOT displace the existing top
/// (stability — the existing top arrived first and wins ties).
fn row_beats_heap_top(
    new: &[Option<Value>],
    top: &HeapEntry,
    keys: &[SortKey],
) -> bool {
    use super::sort::SortDirection;
    for (i, key) in keys.iter().enumerate() {
        let cmp = compare_values_with_nulls(&new[i], &top.sort_values[i], key.null_order);
        let user_cmp = match key.direction {
            SortDirection::Ascending => cmp,
            SortDirection::Descending => cmp.reverse(),
        };
        match user_cmp {
            Ordering::Less => return true,   // new is strictly better
            Ordering::Greater => return false, // new is strictly worse
            Ordering::Equal => continue,
        }
    }
    false // every key tied — keep existing
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.insertion_id == other.insertion_id
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        use super::sort::SortDirection;
        // Both entries share the same Arc<Vec<SortKey>> (one per
        // TopKOperator); use self's view.
        //
        // Goal: BinaryHeap is a max-heap; peek() must return the
        // worst-by-user-order so we evict it on overflow.
        //   - User ASC: worst = largest value. peek wants largest →
        //     Ord must say "larger is greater" → heap_cmp = cmp (no reverse).
        //   - User DESC: worst = smallest value. peek wants smallest →
        //     Ord must say "smaller is greater" → heap_cmp = cmp.reverse().
        for (i, key) in self.sort_keys.iter().enumerate() {
            let cmp = compare_values_with_nulls(
                &self.sort_values[i],
                &other.sort_values[i],
                key.null_order,
            );
            let heap_cmp = match key.direction {
                SortDirection::Ascending => cmp,
                SortDirection::Descending => cmp.reverse(),
            };
            if heap_cmp != Ordering::Equal {
                return heap_cmp;
            }
        }
        // Tiebreak: larger insertion_id is "greater" so newer ties bubble to
        // peek and pop() evicts them first. into_sorted_vec then yields
        // older-first = input order, preserving stability.
        self.insertion_id.cmp(&other.insertion_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::DataChunk;
    use crate::execution::chunk::DataChunkBuilder;

    struct MockOperator {
        chunks: Vec<DataChunk>,
        position: usize,
    }

    impl MockOperator {
        fn new(chunks: Vec<DataChunk>) -> Self {
            Self { chunks, position: 0 }
        }
    }

    impl Operator for MockOperator {
        fn next(&mut self) -> OperatorResult {
            if self.position < self.chunks.len() {
                let chunk = std::mem::replace(&mut self.chunks[self.position], DataChunk::empty());
                self.position += 1;
                Ok(Some(chunk))
            } else {
                Ok(None)
            }
        }

        fn reset(&mut self) {
            self.position = 0;
        }

        fn name(&self) -> &'static str {
            "Mock"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    fn chunk_int64(values: &[i64]) -> DataChunk {
        let mut b = DataChunkBuilder::new(&[LogicalType::Int64]);
        for &v in values {
            b.column_mut(0).unwrap().push_int64(v);
            b.advance_row();
        }
        b.finish()
    }

    fn collect_int64_col(op: &mut dyn Operator) -> Vec<i64> {
        let mut out = Vec::new();
        while let Some(chunk) = op.next().unwrap() {
            for row in chunk.selected_indices() {
                out.push(chunk.column(0).unwrap().get_int64(row).unwrap());
            }
        }
        out
    }

    #[test]
    fn top_k_returns_top_k_descending() {
        let mock = MockOperator::new(vec![chunk_int64(&[10, 50, 30, 20, 40])]);
        let mut top_k = TopKOperator::new(
            Box::new(mock),
            vec![SortKey::descending(0)],
            3,
            vec![LogicalType::Int64],
        );
        let out = collect_int64_col(&mut top_k);
        assert_eq!(out, vec![50, 40, 30]);
    }

    fn chunk_int_str(rows: &[(i64, &str)]) -> DataChunk {
        let mut b = DataChunkBuilder::new(&[LogicalType::Int64, LogicalType::String]);
        for (n, s) in rows {
            b.column_mut(0).unwrap().push_int64(*n);
            b.column_mut(1).unwrap().push_string(*s);
            b.advance_row();
        }
        b.finish()
    }

    fn collect_int_str(op: &mut dyn Operator) -> Vec<(i64, String)> {
        let mut out = Vec::new();
        while let Some(chunk) = op.next().unwrap() {
            for row in chunk.selected_indices() {
                let n = chunk.column(0).unwrap().get_int64(row).unwrap();
                let s = chunk.column(1).unwrap().get_string(row).unwrap().to_string();
                out.push((n, s));
            }
        }
        out
    }

    #[test]
    fn top_k_is_stable_on_ties_descending() {
        // Tied on key=2 across two inputs; stability says first arrival wins.
        let mock = MockOperator::new(vec![chunk_int_str(&[(1, "a"), (2, "b"), (1, "c"), (2, "d")])]);
        let mut top_k = TopKOperator::new(
            Box::new(mock),
            vec![SortKey::descending(0)],
            2,
            vec![LogicalType::Int64, LogicalType::String],
        );
        let out = collect_int_str(&mut top_k);
        assert_eq!(out, vec![(2, "b".into()), (2, "d".into())]);
    }

    #[test]
    fn top_k_is_stable_on_ties_ascending() {
        let mock = MockOperator::new(vec![chunk_int_str(&[(3, "a"), (1, "b"), (3, "c"), (1, "d")])]);
        let mut top_k = TopKOperator::new(
            Box::new(mock),
            vec![SortKey::ascending(0)],
            2,
            vec![LogicalType::Int64, LogicalType::String],
        );
        let out = collect_int_str(&mut top_k);
        assert_eq!(out, vec![(1, "b".into()), (1, "d".into())]);
    }

    #[test]
    fn top_k_skips_materialization_for_losers() {
        // 1000 inputs forming a permutation of 0..1000 via i*31 mod 1000
        // (gcd(31,1000)=1, so this is a true permutation, no duplicates).
        // k=5 ASC: after the heap fills with the first 5 inputs, each
        // subsequent winner causes one peek_mut replace (1 materialization).
        // Total materializations should be far below 1000.
        let values: Vec<i64> = (0..1000)
            .map(|i| ((i * 31 + 7) % 1000) as i64)
            .collect();
        let mock = MockOperator::new(vec![chunk_int64(&values)]);
        let mut top_k = TopKOperator::new(
            Box::new(mock),
            vec![SortKey::ascending(0)],
            5,
            vec![LogicalType::Int64],
        );

        // Drain output to drive the build phase to completion.
        let out = collect_int64_col(&mut top_k);
        assert_eq!(out.len(), 5);

        // Pessimistic upper bound: every distinct min seen along the way could
        // be a materialization. For an unbiased permutation, the expected
        // number of new minima in 1000 draws is H(1000) ≈ 7.5; allow 50 for
        // slack against the specific permutation above.
        let materialized = top_k.materialized_rows();
        assert!(
            materialized < 50,
            "expected < 50 materializations for k=5 over 1000 inputs, got {materialized}"
        );
    }

    #[test]
    fn top_k_multi_key_mixed_directions() {
        // ORDER BY x DESC, y ASC. With k=2, the top 2 by (x DESC, y ASC):
        // input (3,5), (3,2), (1,9), (3,5b) → top 2 are (3,2) then (3,5)
        // (the second (3,5b) is dropped — it's strictly worse than (3,5) on
        // ASC string order).
        let mock = MockOperator::new(vec![
            chunk_int_str(&[(3, "5"), (3, "2"), (1, "9"), (3, "5b")]),
        ]);
        let mut top_k = TopKOperator::new(
            Box::new(mock),
            vec![SortKey::descending(0), SortKey::ascending(1)],
            2,
            vec![LogicalType::Int64, LogicalType::String],
        );
        let out = collect_int_str(&mut top_k);
        // x=3 wins over x=1; among x=3: y="2" < y="5" by ASC string order.
        assert_eq!(out, vec![(3, "2".into()), (3, "5".into())]);
    }

    #[test]
    fn top_k_handles_nulls_first_ascending() {
        use super::super::sort::NullOrder;
        let mut b = DataChunkBuilder::new(&[LogicalType::Int64]);
        for v in [Some(2i64), None, Some(5), None, Some(1)] {
            match v {
                Some(n) => b.column_mut(0).unwrap().push_int64(n),
                None => b.column_mut(0).unwrap().push_value(Value::Null),
            }
            b.advance_row();
        }
        let chunk = b.finish();
        let mock = MockOperator::new(vec![chunk]);

        let mut top_k = TopKOperator::new(
            Box::new(mock),
            vec![SortKey::ascending(0).with_null_order(NullOrder::NullsFirst)],
            3,
            vec![LogicalType::Int64],
        );

        // ORDER BY x ASC NULLS FIRST → [Null, Null, 1, 2, 5]; LIMIT 3 = [Null, Null, 1].
        let mut out = Vec::new();
        while let Some(chunk) = top_k.next().unwrap() {
            for row in chunk.selected_indices() {
                out.push(chunk.column(0).unwrap().get_value(row));
            }
        }
        assert_eq!(out.len(), 3);
        assert!(matches!(out[0], Some(Value::Null)));
        assert!(matches!(out[1], Some(Value::Null)));
        assert_eq!(out[2], Some(Value::Int64(1)));
    }

    #[test]
    fn top_k_handles_nulls_last_ascending() {
        use super::super::sort::NullOrder;
        let mut b = DataChunkBuilder::new(&[LogicalType::Int64]);
        for v in [Some(2i64), None, Some(5), None, Some(1)] {
            match v {
                Some(n) => b.column_mut(0).unwrap().push_int64(n),
                None => b.column_mut(0).unwrap().push_value(Value::Null),
            }
            b.advance_row();
        }
        let chunk = b.finish();
        let mock = MockOperator::new(vec![chunk]);

        let mut top_k = TopKOperator::new(
            Box::new(mock),
            vec![SortKey::ascending(0).with_null_order(NullOrder::NullsLast)],
            3,
            vec![LogicalType::Int64],
        );

        // ORDER BY x ASC NULLS LAST → [1, 2, 5, Null, Null]; LIMIT 3 = [1, 2, 5].
        let mut out = Vec::new();
        while let Some(chunk) = top_k.next().unwrap() {
            for row in chunk.selected_indices() {
                out.push(chunk.column(0).unwrap().get_value(row));
            }
        }
        assert_eq!(
            out,
            vec![Some(Value::Int64(1)), Some(Value::Int64(2)), Some(Value::Int64(5))]
        );
    }

    #[test]
    fn top_k_empty_input() {
        let mock = MockOperator::new(vec![]);
        let mut top_k = TopKOperator::new(
            Box::new(mock),
            vec![SortKey::descending(0)],
            5,
            vec![LogicalType::Int64],
        );
        assert_eq!(collect_int64_col(&mut top_k), Vec::<i64>::new());
    }

    #[test]
    fn top_k_k_zero_returns_no_rows() {
        let mock = MockOperator::new(vec![chunk_int64(&[1, 2, 3])]);
        let mut top_k = TopKOperator::new(
            Box::new(mock),
            vec![SortKey::descending(0)],
            0,
            vec![LogicalType::Int64],
        );
        assert_eq!(collect_int64_col(&mut top_k), Vec::<i64>::new());
    }

    #[test]
    fn top_k_k_greater_than_n() {
        let mock = MockOperator::new(vec![chunk_int64(&[10, 20, 30])]);
        let mut top_k = TopKOperator::new(
            Box::new(mock),
            vec![SortKey::descending(0)],
            10,
            vec![LogicalType::Int64],
        );
        // All 3 rows in DESC order.
        assert_eq!(collect_int64_col(&mut top_k), vec![30, 20, 10]);
    }

    #[test]
    fn top_k_returns_top_k_ascending() {
        let mock = MockOperator::new(vec![chunk_int64(&[10, 50, 30, 20, 40])]);
        let mut top_k = TopKOperator::new(
            Box::new(mock),
            vec![SortKey::ascending(0)],
            3,
            vec![LogicalType::Int64],
        );
        assert_eq!(collect_int64_col(&mut top_k), vec![10, 20, 30]);
    }

    #[test]
    fn top_k_spans_multiple_input_chunks() {
        let mock = MockOperator::new(vec![
            chunk_int64(&[10, 50]),
            chunk_int64(&[30, 20]),
            chunk_int64(&[40, 60]),
        ]);
        let mut top_k = TopKOperator::new(
            Box::new(mock),
            vec![SortKey::descending(0)],
            3,
            vec![LogicalType::Int64],
        );
        assert_eq!(collect_int64_col(&mut top_k), vec![60, 50, 40]);
    }

    #[test]
    fn top_k_into_parts_round_trip() {
        let mock = MockOperator::new(vec![chunk_int64(&[1, 2, 3])]);
        let top_k = TopKOperator::new(
            Box::new(mock),
            vec![SortKey::descending(0)],
            5,
            vec![LogicalType::Int64],
        );
        let (mut child, sort_keys, limit) = top_k.into_parts();
        assert_eq!(sort_keys.len(), 1);
        assert_eq!(limit, 5);
        // Child should still be drainable.
        let chunk = child.next().unwrap().expect("mock yields one chunk");
        assert_eq!(chunk.row_count(), 3);
    }

    #[test]
    fn top_k_name() {
        let mock = MockOperator::new(vec![]);
        let top_k = TopKOperator::new(
            Box::new(mock),
            vec![SortKey::descending(0)],
            5,
            vec![LogicalType::Int64],
        );
        assert_eq!(top_k.name(), "TopK");
    }

    #[test]
    fn top_k_into_any_downcasts() {
        let mock = MockOperator::new(vec![]);
        let op: Box<dyn Operator> = Box::new(TopKOperator::new(
            Box::new(mock),
            vec![SortKey::descending(0)],
            5,
            vec![LogicalType::Int64],
        ));
        let any = op.into_any();
        assert!(any.downcast::<TopKOperator>().is_ok());
    }
}
