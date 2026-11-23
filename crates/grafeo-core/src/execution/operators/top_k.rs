//! Streaming bounded-heap top-K operator.
//!
//! Subsumes `Limit` over `Sort` for the `LIMIT k ORDER BY ...` pattern.
//! Instead of materializing every input row, sorting all of them, and
//! discarding all but the first k, this operator maintains a max-heap of
//! size k keyed by the user's sort tuple, with the comparator inverted so
//! `peek()` returns the worst row by user order. For input cardinality N,
//! memory is O(k) and comparisons are O(N log k). The heap drains in
//! user-requested order via `BinaryHeap::into_sorted_vec`, so no separate
//! sort step is needed.
//!
//! Stability matches `slice::sort`'s stable guarantee: rows tied on every
//! sort key are output in input order, achieved with a monotonic
//! insertion-id tiebreaker.
//!
//! See `plan_limit` in `grafeo-engine` for the dispatch point that builds
//! this operator. PROFILE-mode plans bypass the rewrite for entry-count
//! parity with the logical tree, and `LimitOperator` over `SortOperator`
//! runs instead.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;

use grafeo_common::types::{TemporalComparison, Value};

use super::sort::SortKey;
use super::value_utils::{compare_sort_values, normalized};
use super::{Operator, OperatorResult};
use crate::execution::DataChunk;
use crate::execution::chunk::{ColumnTypes, DataChunkBuilder};

/// Streaming bounded top-K operator.
pub struct TopKOperator {
    child: Box<dyn Operator>,
    /// Shared with every `HeapEntry` via `Arc` so `HeapEntry::Ord` can
    /// compare without raw pointers (`unsafe_code = "deny"` workspace-wide).
    /// Allocated once in `new` and refcount-bumped per heap insertion. The
    /// marginal cost is negligible at k=50, N=1M.
    sort_keys: Arc<Vec<SortKey>>,
    limit: usize,
    /// The column types of the input chunks: the output keeps them.
    column_types: ColumnTypes,
    /// How many leading columns the output keeps, when the trailing ones are
    /// sort keys added only to sort by (see [`with_output_width`](Self::with_output_width)).
    output_width: Option<usize>,
    state: TopKState,
    /// How the sort keys read a local datetime against a zoned one (see
    /// [`TemporalComparison::normalize`]); `TypesApart` reads every value
    /// as itself.
    temporal: TemporalComparison,
    /// What it holds under the statement's memory, captured at construction
    /// (see [`capture`](crate::execution::memory::capture)); it counts nothing
    /// outside a budgeted statement. Its heap of up to `limit` rows cannot spill: past the query
    /// memory limit or the memory limit it goes on in the sort of
    /// [`with_spill_fallback`](Self::with_spill_fallback), or without one fails with `GRAFEO-S001`.
    memory: crate::execution::memory::OperatorMemory,
    /// Makes the spilling sort the rows go on in once the heap outgrows the statement's memory;
    /// `None` fails there instead.
    spill: Option<SpillSort>,
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
    /// The first `limit` rows of the spilling sort the heap went on in, as its output chunks.
    Spilled(std::collections::VecDeque<DataChunk>),
    Done,
}

/// Makes the spilling sort a top-k goes on in: the sort of the top-k's keys over all its
/// columns, keeping its output width.
pub type SpillSort =
    Box<dyn FnOnce() -> Box<dyn crate::execution::pipeline::PushOperator> + Send + Sync>;

/// Whether `error` refuses at the query memory limit or the memory limit (`GRAFEO-S001`), which
/// a spilling sort answers by spilling.
fn is_memory_refusal(error: &super::OperatorError) -> bool {
    use grafeo_common::utils::error::LimitBound;
    matches!(error, super::OperatorError::Wrapped(inner)
    if inner.limit_refusal().is_some_and(|refusal| {
        matches!(refusal.bound(), LimitBound::QueryMemoryLimit | LimitBound::MemoryLimit)
    }))
}

/// A sink that keeps the first `remaining` rows it is given, then stops the sort.
struct FirstRows {
    remaining: usize,
    chunks: std::collections::VecDeque<DataChunk>,
}

impl crate::execution::pipeline::Sink for FirstRows {
    fn consume(&mut self, mut chunk: DataChunk) -> Result<bool, super::OperatorError> {
        if chunk.row_count() > self.remaining {
            let keep: Vec<usize> = chunk.selected_indices().take(self.remaining).collect();
            chunk.set_selection(
                crate::execution::selection::SelectionVector::from_predicate(chunk.len(), |row| {
                    keep.binary_search(&row).is_ok()
                }),
            );
        }
        self.remaining -= chunk.row_count();
        if chunk.row_count() > 0 {
            self.chunks.push_back(chunk);
        }
        Ok(self.remaining > 0)
    }

    fn finalize(&mut self) -> Result<(), super::OperatorError> {
        Ok(())
    }

    fn name(&self) -> &'static str {
        "TopKSpilledRows"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
        self
    }
}

/// What the values of a kept row hold on the heap, as
/// [`HeapColumns::row_bytes`](crate::execution::memory::HeapColumns::row_bytes)
/// counted them when the row came in.
fn row_heap_bytes(values: &[Option<Value>]) -> usize {
    values
        .iter()
        .flatten()
        .map(Value::estimated_size_bytes)
        .fold(0, usize::saturating_add)
}

struct HeapEntry {
    /// The row's sort key values as the operator's temporal reading reads
    /// them: they are only compared, the output comes from `row_values`.
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
    /// including stability on ties. The rows keep their columns' types and
    /// values.
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
    /// for v in [19i64, 88, 33, 8, 319] {
    ///     b.column_mut(0).unwrap().push_int64(v);
    ///     b.advance_row();
    /// }
    /// let source = Source { chunk: Some(b.finish()) };
    ///
    /// let mut top_k = TopKOperator::new(
    ///     Box::new(source),
    ///     vec![SortKey::descending(0)],
    ///     3);
    ///
    /// let chunk = top_k.next().unwrap().unwrap();
    /// let mut out = vec![];
    /// for row in chunk.selected_indices() {
    ///     out.push(chunk.column(0).unwrap().get_int64(row).unwrap());
    /// }
    /// assert_eq!(out, vec![319, 88, 33]);
    /// ```
    #[must_use]
    pub fn new(child: Box<dyn Operator>, sort_keys: Vec<SortKey>, limit: usize) -> Self {
        Self {
            child,
            sort_keys: Arc::new(sort_keys),
            limit,
            column_types: ColumnTypes::default(),
            output_width: None,
            state: TopKState::Building {
                heap: BinaryHeap::new(),
                next_insertion_id: 0,
            },
            temporal: TemporalComparison::TypesApart,
            memory: crate::execution::memory::capture(),
            spill: None,
            #[cfg(test)]
            materialized_rows: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// The operator going on in the sort `spill` makes once its heap outgrows the statement's
    /// memory: the rows it keeps, in the order they came, then the rest of its input, of which
    /// the sort's first `limit` rows are the output. A row the heap dropped was beaten by `limit`
    /// rows before it, so it is in no top `limit`.
    #[must_use]
    pub fn with_spill_fallback(mut self, spill: SpillSort) -> Self {
        self.spill = Some(spill);
        self
    }

    /// The operator with `temporal` as the reading of its sort keys (the
    /// statement's `<`). The rows keep their values.
    #[must_use]
    pub fn with_temporal(mut self, temporal: TemporalComparison) -> Self {
        self.temporal = temporal;
        self
    }

    /// The operator returning only the first `width` columns: the ones after
    /// them are sort keys added only to sort by, as for
    /// [`SortOperator::with_output_width`](super::SortOperator::with_output_width).
    #[must_use]
    pub fn with_output_width(mut self, width: usize) -> Self {
        self.output_width = Some(width);
        self
    }

    /// How many leading columns each output row keeps, when the top-k drops
    /// trailing sort-key columns.
    #[must_use]
    pub fn output_width(&self) -> Option<usize> {
        self.output_width
    }

    /// How the sort keys read temporal values (see [`Self::with_temporal`]).
    #[must_use]
    pub fn temporal(&self) -> TemporalComparison {
        self.temporal
    }

    /// The keys the rows are ordered by.
    #[must_use]
    pub fn sort_keys(&self) -> &[SortKey] {
        &self.sort_keys
    }

    /// Decomposes this operator into its child, sort keys, and limit.
    ///
    /// Mirrors `SortOperator::into_parts` and `LimitOperator::into_parts`
    /// so a future `TopKPushOperator` can drop in via `pipeline_convert.rs`
    /// without an API break. `Arc::try_unwrap` succeeds before the operator
    /// is first pulled or once it has reached `TopKState::Done`. Mid-drain
    /// the rows `Vec` still holds `Arc` clones, so the fallback clones the
    /// keys.
    #[must_use]
    pub fn into_parts(self) -> (Box<dyn Operator>, Vec<SortKey>, usize) {
        let sort_keys = Arc::try_unwrap(self.sort_keys).unwrap_or_else(|arc| (*arc).clone());
        (self.child, sort_keys, self.limit)
    }

    /// Goes on in the spilling sort: the rows `heap` keeps, in the order they came, then the
    /// rest of the input; the sort's first `limit` rows are what the operator returns.
    fn spill_the_rest(
        &mut self,
        heap: BinaryHeap<HeapEntry>,
    ) -> Result<std::collections::VecDeque<DataChunk>, super::OperatorError> {
        let mut sort = (self.spill.take().expect("checked by the caller"))();
        let mut kept = heap.into_vec();
        kept.sort_unstable_by_key(|entry| entry.insertion_id);
        let types = self.column_types.types().to_vec();
        let mut first = FirstRows {
            remaining: self.limit,
            chunks: std::collections::VecDeque::new(),
        };
        for rows in kept.chunks(2048) {
            let mut builder = DataChunkBuilder::with_capacity(&types, rows.len());
            for entry in rows {
                for (col_idx, value) in entry.row_values.iter().enumerate() {
                    if let Some(column) = builder.column_mut(col_idx) {
                        column.push_value(value.clone().unwrap_or(Value::Null));
                    }
                }
                builder.advance_row();
            }
            sort.push(builder.finish(), &mut first)?;
        }
        // The rows are the sort's now: the heap's grant goes.
        drop(kept);
        self.memory.clear();
        while let Some(chunk) = self.child.next()? {
            crate::execution::deadline::check()?;
            sort.push(chunk, &mut first)?;
        }
        sort.finalize(&mut first)?;
        Ok(first.chunks)
    }
}

impl Operator for TopKOperator {
    fn next(&mut self) -> OperatorResult {
        if matches!(self.state, TopKState::Building { .. }) {
            let TopKState::Building {
                mut heap,
                mut next_insertion_id,
            } = std::mem::replace(&mut self.state, TopKState::Done)
            else {
                unreachable!("matches! guard above")
            };

            // What the kept rows' values hold on the heap (strings, lists).
            let mut kept_heap: usize = 0;
            while let Some(chunk) = self.child.next()? {
                crate::execution::deadline::check()?;
                self.column_types.add(&chunk);
                let width = self.column_types.types().len();
                // The columns that hold heap memory, picked once per chunk.
                let heap_columns =
                    crate::execution::memory::HeapColumns::counted(&self.memory, &chunk, None);

                for row_idx in chunk.selected_indices() {
                    let new_sort_values = extract_sort_values(
                        &chunk,
                        row_idx,
                        self.sort_keys.as_slice(),
                        self.temporal,
                    );

                    let should_push = if heap.len() < self.limit {
                        true
                    } else if let Some(top) = heap.peek() {
                        row_beats_heap_top(&new_sort_values, top, self.sort_keys.as_slice())
                    } else {
                        // limit == 0: heap stays empty, never push.
                        false
                    };

                    if !should_push {
                        continue;
                    }

                    let row_values = extract_row_values(&chunk, row_idx, width);
                    #[cfg(test)]
                    self.materialized_rows
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let entry = HeapEntry {
                        sort_values: new_sort_values,
                        row_values,
                        insertion_id: next_insertion_id,
                        sort_keys: Arc::clone(&self.sort_keys),
                    };
                    next_insertion_id += 1;
                    if !heap_columns.is_empty() {
                        kept_heap = kept_heap.saturating_add(heap_columns.row_bytes(row_idx));
                    }
                    if heap.len() < self.limit {
                        heap.push(entry);
                    } else {
                        // Heap is full and the new entry beat the worst: replace
                        // the heap's max in place. One sift-down vs push+pop's
                        // two reheapifies, significant for large N.
                        let mut top = heap.peek_mut().expect("heap.len() == limit > 0");
                        if kept_heap > 0 {
                            kept_heap = kept_heap.saturating_sub(row_heap_bytes(&top.row_values));
                        }
                        *top = entry;
                    }
                }
                // The heap and what its rows' values hold, against the
                // statement's memory: a chunk at a time, a compare within the
                // grant (a literal LIMIT can be large).
                let counted = self.memory.set(
                    heap.capacity()
                        .saturating_mul(
                            std::mem::size_of::<HeapEntry>()
                                + (width + self.sort_keys.len())
                                    * std::mem::size_of::<Option<Value>>(),
                        )
                        .saturating_add(kept_heap),
                );
                match counted {
                    Ok(()) => {}
                    Err(error) if self.spill.is_some() && is_memory_refusal(&error) => {
                        let chunks = self.spill_the_rest(heap)?;
                        self.state = TopKState::Spilled(chunks);
                        return self.next();
                    }
                    Err(error) => return Err(error),
                }
            }

            let rows = heap.into_sorted_vec();
            self.state = TopKState::Draining { rows, position: 0 };
        }

        if let TopKState::Spilled(chunks) = &mut self.state {
            if let Some(chunk) = chunks.pop_front() {
                return Ok(Some(chunk));
            }
            self.state = TopKState::Done;
            return Ok(None);
        }

        if let TopKState::Draining { rows, position } = &mut self.state {
            if *position < rows.len() {
                let types = self.column_types.types();
                let width = self
                    .output_width
                    .map_or(types.len(), |width| width.min(types.len()));
                let mut builder = DataChunkBuilder::with_capacity(&types[..width], 2048);
                while *position < rows.len() && !builder.is_full() {
                    let entry = &rows[*position];
                    for col_idx in 0..width {
                        if let Some(dst_col) = builder.column_mut(col_idx) {
                            let val = entry.row_values[col_idx].clone().unwrap_or(Value::Null);
                            dst_col.push_value(val);
                        }
                    }
                    grafeo_common::work::count(
                        grafeo_common::work::Work::ColumnCopied,
                        width as u64,
                    );
                    builder.advance_row();
                    *position += 1;
                }
                if builder.row_count() > 0 {
                    return Ok(Some(builder.finish()));
                }
            }
            self.state = TopKState::Done;
        }

        Ok(None)
    }

    fn reset(&mut self) {
        self.child.reset();
        self.column_types = ColumnTypes::default();
        self.state = TopKState::Building {
            heap: BinaryHeap::new(),
            next_insertion_id: 0,
        };
        self.memory.clear();
        #[cfg(test)]
        self.materialized_rows
            .store(0, std::sync::atomic::Ordering::Relaxed);
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
        self.materialized_rows
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// The sort key values of a row, as `temporal` reads them (see
/// [`TemporalComparison::normalize`]).
fn extract_sort_values(
    chunk: &DataChunk,
    row_idx: usize,
    sort_keys: &[SortKey],
    temporal: TemporalComparison,
) -> Vec<Option<Value>> {
    sort_keys
        .iter()
        .map(|k| {
            chunk
                .column(k.column)
                .and_then(|c| c.get_value(row_idx))
                .map(|value| normalized(value, temporal))
        })
        .collect()
}

fn extract_row_values(chunk: &DataChunk, row_idx: usize, n_cols: usize) -> Vec<Option<Value>> {
    (0..n_cols)
        .map(|i| chunk.column(i).and_then(|c| c.get_value(row_idx)))
        .collect()
}

/// Strict better-than test: does `new` beat the current heap top per
/// user-requested order?
///
/// Inserting a new row that ties on every key must NOT displace the existing
/// top. The existing top arrived first and wins ties (stability).
fn row_beats_heap_top(new: &[Option<Value>], top: &HeapEntry, keys: &[SortKey]) -> bool {
    for (i, key) in keys.iter().enumerate() {
        let user_cmp = compare_sort_values(
            new[i].as_ref(),
            top.sort_values[i].as_ref(),
            key.direction,
            key.null_order,
        );
        match user_cmp {
            Ordering::Less => return true,
            Ordering::Greater => return false,
            Ordering::Equal => continue,
        }
    }
    false
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
        // Both entries share the same Arc<Vec<SortKey>> (one per
        // TopKOperator); use self's view.
        //
        // Goal: BinaryHeap is a max-heap. peek() must return the
        // worst-by-user-order so we can evict it on overflow, so Ord is the
        // user order itself: a row that comes later is greater.
        for (i, key) in self.sort_keys.iter().enumerate() {
            let heap_cmp = compare_sort_values(
                self.sort_values[i].as_ref(),
                other.sort_values[i].as_ref(),
                key.direction,
                key.null_order,
            );
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
    use grafeo_common::types::LogicalType;

    struct MockOperator {
        chunks: Vec<DataChunk>,
        position: usize,
    }

    impl MockOperator {
        fn new(chunks: Vec<DataChunk>) -> Self {
            Self {
                chunks,
                position: 0,
            }
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
        let mock = MockOperator::new(vec![chunk_int64(&[19, 88, 33, 8, 319])]);
        let mut top_k = TopKOperator::new(Box::new(mock), vec![SortKey::descending(0)], 3);
        let out = collect_int64_col(&mut top_k);
        assert_eq!(out, vec![319, 88, 33]);
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
                let s = chunk
                    .column(1)
                    .unwrap()
                    .get_string(row)
                    .unwrap()
                    .to_string();
                out.push((n, s));
            }
        }
        out
    }

    #[test]
    fn top_k_is_stable_on_ties_descending() {
        // Tied on key=88 across two rows; stability says the first arrival wins.
        let mock = MockOperator::new(vec![chunk_int_str(&[
            (3, "Vincent"),
            (88, "Jules"),
            (3, "Mia"),
            (88, "Butch"),
        ])]);
        let mut top_k = TopKOperator::new(Box::new(mock), vec![SortKey::descending(0)], 2);
        let out = collect_int_str(&mut top_k);
        assert_eq!(out, vec![(88, "Jules".into()), (88, "Butch".into())]);
    }

    #[test]
    fn top_k_is_stable_on_ties_ascending() {
        let mock = MockOperator::new(vec![chunk_int_str(&[
            (88, "Vincent"),
            (3, "Jules"),
            (88, "Mia"),
            (3, "Butch"),
        ])]);
        let mut top_k = TopKOperator::new(Box::new(mock), vec![SortKey::ascending(0)], 2);
        let out = collect_int_str(&mut top_k);
        assert_eq!(out, vec![(3, "Jules".into()), (3, "Butch".into())]);
    }

    #[test]
    fn top_k_skips_materialization_for_losers() {
        // 1000 inputs forming a permutation of 0..1000 via i*31 mod 1000
        // (gcd(31, 1000) = 1, so this is a true permutation, no duplicates).
        // k=5 ASC: after the heap fills with the first 5 inputs, each
        // subsequent winner causes one peek_mut replace (1 materialization).
        // Total materializations should be far below 1000.
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let values: Vec<i64> = (0..1000_i64).map(|i| (i * 31 + 7) % 1000).collect();
        let mock = MockOperator::new(vec![chunk_int64(&values)]);
        let mut top_k = TopKOperator::new(Box::new(mock), vec![SortKey::ascending(0)], 5);

        let out = collect_int64_col(&mut top_k);
        assert_eq!(out.len(), 5);

        // Pessimistic upper bound: every distinct minimum seen along the way
        // could be a materialization. For an unbiased permutation, the
        // expected number of new minima in 1000 draws is H(1000) ~ 7.5;
        // allow 50 for slack against the specific permutation above.
        let materialized = top_k.materialized_rows();
        assert!(
            materialized < 50,
            "expected < 50 materializations for k=5 over 1000 inputs, got {materialized}"
        );
    }

    #[test]
    fn top_k_multi_key_mixed_directions() {
        // ORDER BY x DESC, y ASC. With k=2, the top 2 by (x DESC, y ASC):
        // input (88, "5"), (88, "3"), (19, "8"), (88, "5b") gives top 2 of
        // (88, "3") then (88, "5"). The second (88, "5b") is dropped, strictly
        // worse than (88, "5") on ASC string order.
        let mock = MockOperator::new(vec![chunk_int_str(&[
            (88, "5"),
            (88, "3"),
            (19, "8"),
            (88, "5b"),
        ])]);
        let mut top_k = TopKOperator::new(
            Box::new(mock),
            vec![SortKey::descending(0), SortKey::ascending(1)],
            2,
        );
        let out = collect_int_str(&mut top_k);
        assert_eq!(out, vec![(88, "3".into()), (88, "5".into())]);
    }

    #[test]
    fn top_k_handles_nulls_first_ascending() {
        use super::super::sort::NullOrder;
        let mut b = DataChunkBuilder::new(&[LogicalType::Int64]);
        for v in [Some(19_i64), None, Some(88), None, Some(3)] {
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
        );

        // ORDER BY x ASC NULLS FIRST gives [Null, Null, 3, 19, 88]; LIMIT 3 = [Null, Null, 3].
        let mut out = Vec::new();
        while let Some(chunk) = top_k.next().unwrap() {
            for row in chunk.selected_indices() {
                out.push(chunk.column(0).unwrap().get_value(row));
            }
        }
        assert_eq!(out.len(), 3);
        assert!(matches!(out[0], Some(Value::Null)));
        assert!(matches!(out[1], Some(Value::Null)));
        assert_eq!(out[2], Some(Value::Int64(3)));
    }

    #[test]
    fn top_k_handles_nulls_last_ascending() {
        use super::super::sort::NullOrder;
        let mut b = DataChunkBuilder::new(&[LogicalType::Int64]);
        for v in [Some(19_i64), None, Some(88), None, Some(3)] {
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
        );

        // ORDER BY x ASC NULLS LAST gives [3, 19, 88, Null, Null]; LIMIT 3 = [3, 19, 88].
        let mut out = Vec::new();
        while let Some(chunk) = top_k.next().unwrap() {
            for row in chunk.selected_indices() {
                out.push(chunk.column(0).unwrap().get_value(row));
            }
        }
        assert_eq!(
            out,
            vec![
                Some(Value::Int64(3)),
                Some(Value::Int64(19)),
                Some(Value::Int64(88))
            ]
        );
    }

    /// `NULLS LAST` holds when descending too, and descending puts nulls
    /// first by default (null sorts as the largest value).
    #[test]
    fn top_k_places_nulls_as_the_key_says_when_descending() {
        use super::super::sort::NullOrder;
        let input = || {
            let mut b = DataChunkBuilder::new(&[LogicalType::Int64]);
            for v in [Some(19_i64), None, Some(88), None, Some(3)] {
                match v {
                    Some(n) => b.column_mut(0).unwrap().push_int64(n),
                    None => b.column_mut(0).unwrap().push_value(Value::Null),
                }
                b.advance_row();
            }
            MockOperator::new(vec![b.finish()])
        };
        let top = |key: SortKey| {
            let mut top_k = TopKOperator::new(Box::new(input()), vec![key], 3);
            let mut out = Vec::new();
            while let Some(chunk) = top_k.next().unwrap() {
                for row in chunk.selected_indices() {
                    out.push(chunk.column(0).unwrap().get_value(row).unwrap());
                }
            }
            out
        };

        assert_eq!(
            top(SortKey::descending(0).with_null_order(NullOrder::NullsLast)),
            [Value::Int64(88), Value::Int64(19), Value::Int64(3)]
        );
        assert_eq!(
            top(SortKey::descending(0)),
            [Value::Null, Value::Null, Value::Int64(88)]
        );
    }

    #[test]
    fn top_k_empty_input() {
        let mock = MockOperator::new(vec![]);
        let mut top_k = TopKOperator::new(Box::new(mock), vec![SortKey::descending(0)], 5);
        assert_eq!(collect_int64_col(&mut top_k), Vec::<i64>::new());
    }

    #[test]
    fn top_k_k_zero_returns_no_rows() {
        let mock = MockOperator::new(vec![chunk_int64(&[3, 19, 88])]);
        let mut top_k = TopKOperator::new(Box::new(mock), vec![SortKey::descending(0)], 0);
        assert_eq!(collect_int64_col(&mut top_k), Vec::<i64>::new());
    }

    #[test]
    fn top_k_k_greater_than_n() {
        let mock = MockOperator::new(vec![chunk_int64(&[19, 88, 3])]);
        let mut top_k = TopKOperator::new(Box::new(mock), vec![SortKey::descending(0)], 10);
        assert_eq!(collect_int64_col(&mut top_k), vec![88, 19, 3]);
    }

    #[test]
    fn top_k_returns_top_k_ascending() {
        let mock = MockOperator::new(vec![chunk_int64(&[19, 88, 33, 8, 319])]);
        let mut top_k = TopKOperator::new(Box::new(mock), vec![SortKey::ascending(0)], 3);
        assert_eq!(collect_int64_col(&mut top_k), vec![8, 19, 33]);
    }

    #[test]
    fn top_k_spans_multiple_input_chunks() {
        let mock = MockOperator::new(vec![
            chunk_int64(&[19, 88]),
            chunk_int64(&[33, 8]),
            chunk_int64(&[40, 319]),
        ]);
        let mut top_k = TopKOperator::new(Box::new(mock), vec![SortKey::descending(0)], 3);
        assert_eq!(collect_int64_col(&mut top_k), vec![319, 88, 40]);
    }

    #[test]
    fn top_k_into_parts_round_trip() {
        let mock = MockOperator::new(vec![chunk_int64(&[3, 19, 88])]);
        let top_k = TopKOperator::new(Box::new(mock), vec![SortKey::descending(0)], 5);
        let (mut child, sort_keys, limit) = top_k.into_parts();
        assert_eq!(sort_keys.len(), 1);
        assert_eq!(limit, 5);
        let chunk = child.next().unwrap().expect("mock yields one chunk");
        assert_eq!(chunk.row_count(), 3);
    }

    #[test]
    fn top_k_name() {
        let mock = MockOperator::new(vec![]);
        let top_k = TopKOperator::new(Box::new(mock), vec![SortKey::descending(0)], 5);
        assert_eq!(top_k.name(), "TopK");
    }

    #[test]
    fn top_k_into_any_downcasts() {
        let mock = MockOperator::new(vec![]);
        let op: Box<dyn Operator> = Box::new(TopKOperator::new(
            Box::new(mock),
            vec![SortKey::descending(0)],
            5,
        ));
        let any = op.into_any();
        assert!(any.downcast::<TopKOperator>().is_ok());
    }
}
