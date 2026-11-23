//! ORDER BY of a SPARQL query (SPARQL 1.1 section 15.1): each row's keys are
//! computed once, a literal's value parsed once, and the rows sorted by them.
//!
//! A variable's key is its id column, keyed by the term and the value the
//! dictionary keeps for it (so a literal is not parsed again); an
//! expression's key is a term the plan projected after the returned columns,
//! which are the only ones the sort returns.
//!
//! Under `ORDER BY ... LIMIT` the sort is a top-k: it keeps the first
//! `offset + limit` rows in order, their returned values and keys, in a
//! bounded heap, and no input chunk beyond the one it reads. Its rows are the
//! full sort's first `k`, ties in input order.
//!
//! Past the query memory limit, a sort whose statement has a spill path
//! spills: it writes what it holds as a sorted run, in input order, and the
//! runs merge at the end (see [`grafeo_core::execution::spill::RunMerge`]),
//! so it answers as in memory, ties included. A top-k that outgrows its grant
//! writes its kept rows as the first run and spills the rest of its input,
//! less the rows that come at or after its last kept row, which cannot be
//! among the first `k`. A run's row is the returned columns, then one value
//! per key (its term id or term), from which the merge makes the key again.
//! Without a spill path the sort refuses with `GRAFEO-S001`.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;

use grafeo_common::types::{LogicalType, SparqlOrderKey, Value};
use grafeo_core::execution::DataChunk;
use grafeo_core::execution::chunk::DataChunkBuilder;
use grafeo_core::execution::memory::{HeapColumns, chunk_bytes};
use grafeo_core::execution::operators::{Operator, OperatorError, OperatorResult};
#[cfg(feature = "spill")]
use grafeo_core::execution::{
    memory::{OperatorMemory, OperatorMemoryContext},
    spill::{RowOrder, RunMerge, SortedRuns},
};

use super::term_ids::{QueryTermIds, id_of};
use super::terms;

/// The rows a merged run hands out at a time: an output chunk's.
#[cfg(feature = "spill")]
const MERGE_BATCH_ROWS: usize = 2048;

/// One ORDER BY key: the column of its terms, its direction, and where
/// unbound goes.
#[derive(Debug, Clone, Copy)]
pub(super) struct OrderKey {
    pub(super) column: usize,
    pub(super) descending: bool,
    pub(super) nulls_first: bool,
}

/// The keys of one row, an unbound key as `None`.
type RowKeys = Vec<Option<SparqlOrderKey>>;

/// How the rows `a` and `b` order by `order`: the one comparison of the full
/// sort and the top-k. `Equal` is a tie, which input order breaks.
fn compare(
    order: &[OrderKey],
    a: &[Option<SparqlOrderKey>],
    b: &[Option<SparqlOrderKey>],
) -> Ordering {
    for (key, (x, y)) in order.iter().zip(a.iter().zip(b)) {
        let ordering = match (x, y) {
            (None, None) => Ordering::Equal,
            (None, Some(_)) if key.nulls_first => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (Some(_), None) if key.nulls_first => Ordering::Greater,
            (Some(_), None) => Ordering::Less,
            (Some(x), Some(y)) if key.descending => y.cmp(x),
            (Some(x), Some(y)) => x.cmp(y),
        };
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    Ordering::Equal
}

/// The key of a key column's `value`: a term id's, a term's, or `None` for
/// unbound.
fn key_of(value: Value, terms: &QueryTermIds) -> Option<SparqlOrderKey> {
    match value {
        Value::Null => None,
        Value::Int64(_) => {
            let id = id_of(&value)?;
            SparqlOrderKey::of_inline(id).or_else(|| {
                let entry = terms.entry(id)?;
                Some(SparqlOrderKey::with_native(
                    Arc::clone(entry.term()),
                    entry.native(),
                ))
            })
        }
        Value::RdfTerm(term) => Some(SparqlOrderKey::new(term)),
        other => Some(SparqlOrderKey::new(Arc::new(terms::to_term(&other)?))),
    }
}

/// The value of `row`'s column `column` in `chunk`, null when it has none.
fn value_at(chunk: &DataChunk, column: usize, row: usize) -> Value {
    chunk
        .column(column)
        .and_then(|c| c.get_value(row))
        .unwrap_or(Value::Null)
}

/// The keys of `row` of `chunk` by `order` (see the module docs).
fn row_keys(order: &[OrderKey], terms: &QueryTermIds, chunk: &DataChunk, row: usize) -> RowKeys {
    order
        .iter()
        .map(|key| key_of(chunk.column(key.column)?.get_value(row)?, terms))
        .collect()
}

/// How a spilled SPARQL sort's runs order: a run's row is the `width`
/// returned values, then one value per key, as [`RdfSortOperator`] writes it.
#[cfg(feature = "spill")]
struct SparqlRowOrder {
    order: Arc<[OrderKey]>,
    terms: Arc<QueryTermIds>,
    width: usize,
}

#[cfg(feature = "spill")]
impl RowOrder for SparqlRowOrder {
    type Key = RowKeys;

    fn key(&mut self, row: &[Value]) -> Result<RowKeys, OperatorError> {
        Ok(row[self.width..]
            .iter()
            .map(|value| key_of(value.clone(), &self.terms))
            .collect())
    }

    fn compare(&self, a: &RowKeys, b: &RowKeys) -> Ordering {
        compare(&self.order, a, b)
    }

    fn key_bytes(&self, key: &RowKeys) -> usize {
        key.len() * std::mem::size_of::<Option<SparqlOrderKey>>()
    }
}

/// A row of the full sort, by its chunk and position, with its keys.
struct Row {
    chunk: usize,
    row: usize,
    keys: RowKeys,
}

/// A row the top-k keeps: its returned values (then its key columns' values
/// when the sort can spill, so it can become a run's row), and its keys,
/// taken out of its chunk.
struct Kept {
    /// The order, shared by every kept row, so the heap compares by it.
    order: Arc<[OrderKey]>,
    keys: RowKeys,
    values: Vec<Value>,
    /// Its place in the input: the later of two tied rows is after.
    sequence: u64,
    /// What its values and keys hold on the heap, as counted when it came.
    heap_bytes: usize,
}

impl PartialEq for Kept {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Kept {}

impl PartialOrd for Kept {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Kept {
    /// The rows' order, ties by input order: the heap's top is the last row
    /// it keeps, the one a better row evicts.
    fn cmp(&self, other: &Self) -> Ordering {
        compare(&self.order, &self.keys, &other.keys).then(self.sequence.cmp(&other.sequence))
    }
}

/// The rows once sorted, in their order.
enum Sorted {
    /// The full sort's: every input chunk, and its rows by position.
    Chunks {
        chunks: Vec<DataChunk>,
        rows: Vec<Row>,
    },
    /// The top-k's: the rows it kept, their values taken as they go out.
    Kept(Vec<Kept>),
    /// A spilled sort's: its runs' merge, and the rows still to return (a
    /// top-k's `k`; every row for a full sort).
    #[cfg(feature = "spill")]
    Merged {
        merge: Box<RunMerge<SparqlRowOrder>>,
        left: usize,
    },
}

impl Sorted {
    fn len(&self) -> usize {
        match self {
            Self::Chunks { rows, .. } => rows.len(),
            Self::Kept(rows) => rows.len(),
            #[cfg(feature = "spill")]
            Self::Merged { left, .. } => *left,
        }
    }
}

/// Sorts its input by SPARQL order keys (see the module docs).
pub(super) struct RdfSortOperator {
    child: Box<dyn Operator>,
    order: Arc<[OrderKey]>,
    /// The types of the returned columns, which come first.
    output_types: Vec<LogicalType>,
    terms: Arc<QueryTermIds>,
    /// `Some(k)`: a top-k, which returns the first `k` rows in order only.
    top_k: Option<usize>,
    /// The rows in order, once the input is read.
    sorted: Option<Sorted>,
    position: usize,
    /// What it holds under the statement's memory, captured at construction
    /// (see [`capture`](grafeo_core::execution::memory::capture)); it counts
    /// nothing outside a budgeted statement. Past the query memory limit or
    /// the memory limit it spills when `spill` holds a context, and fails
    /// with `GRAFEO-S001` otherwise.
    memory: grafeo_core::execution::OperatorMemory,
    /// The statement's context when it has a spill path: where runs go, and
    /// what the merge's memory counts under.
    #[cfg(feature = "spill")]
    spill: Option<OperatorMemoryContext>,
}

impl RdfSortOperator {
    /// Sorts `child`'s rows by `keys`, returning the columns of
    /// `output_types`; with `top_k` of `Some(k)`, the first `k` of them only
    /// (`ORDER BY ... LIMIT`, `k` its offset plus its limit).
    pub(super) fn new(
        child: Box<dyn Operator>,
        keys: Vec<OrderKey>,
        output_types: Vec<LogicalType>,
        terms: Arc<QueryTermIds>,
        top_k: Option<usize>,
    ) -> Self {
        Self {
            child,
            order: keys.into(),
            output_types,
            terms,
            top_k,
            sorted: None,
            position: 0,
            memory: grafeo_core::execution::memory::capture(),
            #[cfg(feature = "spill")]
            spill: grafeo_core::execution::memory::current()
                .filter(|ctx| ctx.spill_manager().is_some()),
        }
    }

    /// A run's row of `row` of `chunk`: the returned values, then the key
    /// columns' values.
    fn run_row(&self, chunk: &DataChunk, row: usize) -> Vec<Value> {
        let width = self.output_types.len();
        let mut values = Vec::with_capacity(width + self.order.len());
        values.extend((0..width).map(|column| value_at(chunk, column, row)));
        values.extend(
            self.order
                .iter()
                .map(|key| value_at(chunk, key.column, row)),
        );
        values
    }

    /// The key columns' values a kept row carries: one per key when the sort
    /// can spill (its kept rows may become a run), none otherwise.
    fn kept_keys(&self) -> usize {
        #[cfg(feature = "spill")]
        if self.spill.is_some() {
            return self.order.len();
        }
        0
    }

    /// What the top-k keeps of `row` of `chunk`: a run's row when it can
    /// spill, the returned values otherwise.
    fn kept_values(&self, chunk: &DataChunk, row: usize) -> Vec<Value> {
        if self.kept_keys() > 0 {
            return self.run_row(chunk, row);
        }
        (0..self.output_types.len())
            .map(|column| value_at(chunk, column, row))
            .collect()
    }

    /// No runs yet, for a sort that spills under `ctx`.
    #[cfg(feature = "spill")]
    fn new_runs(&self, ctx: &OperatorMemoryContext) -> SortedRuns<SparqlRowOrder> {
        SortedRuns::new(
            Arc::clone(
                ctx.spill_manager()
                    .expect("a spilling sort has a spill path"),
            ),
            self.output_types.len() + self.order.len(),
            SparqlRowOrder {
                order: Arc::clone(&self.order),
                terms: Arc::clone(&self.terms),
                width: self.output_types.len(),
            },
        )
    }

    /// Writes the held rows as a run, in their order, and counts nothing
    /// held any more.
    #[cfg(feature = "spill")]
    fn spill_rows(
        &mut self,
        runs: &mut SortedRuns<SparqlRowOrder>,
        chunks: &mut Vec<DataChunk>,
        rows: &mut Vec<Row>,
    ) -> Result<(), OperatorError> {
        // A stable sort: ties keep input order.
        rows.sort_by(|a, b| compare(&self.order, &a.keys, &b.keys));
        runs.spill(
            rows.iter()
                .map(|row| self.run_row(&chunks[row.chunk], row.row)),
        )?;
        chunks.clear();
        rows.clear();
        self.memory.clear();
        Ok(())
    }

    /// The merge of `runs` and the rows still held, which become its last
    /// run, under the statement's memory. The held rows' buffers are freed
    /// and the sort's grant goes back before the merge's passes run.
    #[cfg(feature = "spill")]
    fn merged(
        &mut self,
        mut runs: SortedRuns<SparqlRowOrder>,
        mut chunks: Vec<DataChunk>,
        mut rows: Vec<Row>,
        ctx: &OperatorMemoryContext,
        left: usize,
    ) -> Result<Sorted, OperatorError> {
        self.spill_rows(&mut runs, &mut chunks, &mut rows)?;
        // Emptied, they keep their capacity, which nothing counts any more.
        drop(chunks);
        drop(rows);
        self.memory.release();
        let merge = runs.merge(OperatorMemory::new(ctx.clone()), MERGE_BATCH_ROWS)?;
        Ok(Sorted::Merged {
            merge: Box::new(merge),
            left,
        })
    }

    /// Every row, sorted.
    fn sort(&mut self) -> Result<Sorted, OperatorError> {
        let mut chunks = Vec::new();
        let mut rows = Vec::new();
        #[cfg(feature = "spill")]
        let mut runs: Option<SortedRuns<SparqlRowOrder>> = None;
        while let Some(chunk) = self.child.next()? {
            let index = chunks.len();
            rows.extend(chunk.selected_indices().map(|row| Row {
                chunk: index,
                row,
                keys: row_keys(&self.order, &self.terms, &chunk, row),
            }));
            // The chunk and its rows' keys, against the statement's memory: a
            // chunk at a time, a compare within the grant.
            let counted = self.memory.add_with(|| {
                chunk_bytes(&chunk)
                    + chunk.row_count()
                        * (std::mem::size_of::<Row>()
                            + self.order.len() * std::mem::size_of::<Option<SparqlOrderKey>>())
            });
            chunks.push(chunk);
            // Without spilling built in, past the grant is a refusal.
            #[cfg(not(feature = "spill"))]
            counted?;
            #[cfg(feature = "spill")]
            if let Err(refusal) = counted {
                if let Some(ctx) = self.spill.clone() {
                    let mut spilled = runs.take().unwrap_or_else(|| self.new_runs(&ctx));
                    self.spill_rows(&mut spilled, &mut chunks, &mut rows)?;
                    runs = Some(spilled);
                    continue;
                }
                return Err(refusal);
            }
        }
        #[cfg(feature = "spill")]
        if let Some(runs) = runs {
            let ctx = self
                .spill
                .clone()
                .expect("a sort that spilled has a spill path");
            return self.merged(runs, chunks, rows, &ctx, usize::MAX);
        }
        // A stable sort: ties keep input order.
        rows.sort_by(|a, b| compare(&self.order, &a.keys, &b.keys));
        Ok(Sorted::Chunks { chunks, rows })
    }

    /// The first `k` rows, sorted: a row is kept while fewer than `k` are,
    /// or in place of the last kept when it comes before it.
    fn top(&mut self, k: usize) -> Result<Sorted, OperatorError> {
        let width = self.output_types.len();
        // What a kept row holds besides what its values and keys hold on the
        // heap.
        let kept_row_bytes = std::mem::size_of::<Kept>()
            + self.order.len() * std::mem::size_of::<Option<SparqlOrderKey>>()
            + (width + self.kept_keys()) * std::mem::size_of::<Value>();
        let mut heap: BinaryHeap<Kept> = BinaryHeap::new();
        let mut sequence: u64 = 0;
        let mut kept_heap_bytes: usize = 0;
        while let Some(chunk) = self.child.next()? {
            // The columns that hold heap memory (values and keys), picked
            // once per chunk.
            let heap_columns = HeapColumns::counted(&self.memory, &chunk, None);
            for row in chunk.selected_indices() {
                let keys = row_keys(&self.order, &self.terms, &chunk, row);
                let full = heap.len() >= k;
                // A row tied with the last kept comes after it.
                if full
                    && heap.peek().is_none_or(|last| {
                        compare(&self.order, &keys, &last.keys) != Ordering::Less
                    })
                {
                    sequence += 1;
                    continue;
                }
                let heap_bytes = heap_columns.row_bytes(row);
                let kept = Kept {
                    order: Arc::clone(&self.order),
                    keys,
                    values: self.kept_values(&chunk, row),
                    sequence,
                    heap_bytes,
                };
                sequence += 1;
                kept_heap_bytes = kept_heap_bytes.saturating_add(heap_bytes);
                // Full, the heap holds a row (`k` is above 0): the new row
                // takes the last one's place.
                if full && let Some(mut last) = heap.peek_mut() {
                    kept_heap_bytes = kept_heap_bytes.saturating_sub(last.heap_bytes);
                    *last = kept;
                } else {
                    heap.push(kept);
                }
            }
            // The kept rows, against the statement's memory: a chunk at a
            // time, a compare within the grant.
            let counted = self.memory.set(
                heap.capacity()
                    .saturating_mul(kept_row_bytes)
                    .saturating_add(kept_heap_bytes),
            );
            // Without spilling built in, past the grant is a refusal.
            #[cfg(not(feature = "spill"))]
            counted?;
            #[cfg(feature = "spill")]
            if let Err(refusal) = counted {
                if let Some(ctx) = self.spill.clone() {
                    return self.top_spilled(k, heap, &ctx);
                }
                return Err(refusal);
            }
        }
        Ok(Sorted::Kept(heap.into_sorted_vec()))
    }

    /// The top-k past its grant: its kept rows, in order, become the first
    /// run, the rest of its input spills as a full sort's does, less the rows
    /// at or after the last kept row when it keeps `k` (they cannot be among
    /// the first `k`, and a tie goes to the earlier kept row), and the first
    /// `k` rows of the merge are its rows.
    #[cfg(feature = "spill")]
    fn top_spilled(
        &mut self,
        k: usize,
        heap: BinaryHeap<Kept>,
        ctx: &OperatorMemoryContext,
    ) -> Result<Sorted, OperatorError> {
        let last = (heap.len() >= k)
            .then(|| heap.peek().map(|last| last.keys.clone()))
            .flatten();
        let mut runs = self.new_runs(ctx);
        let kept = heap.into_sorted_vec();
        runs.spill(kept.into_iter().map(|kept| kept.values))?;
        self.memory.clear();
        let mut chunks = Vec::new();
        let mut rows = Vec::new();
        while let Some(chunk) = self.child.next()? {
            let index = chunks.len();
            let before = rows.len();
            rows.extend(chunk.selected_indices().filter_map(|row| {
                let keys = row_keys(&self.order, &self.terms, &chunk, row);
                let after_last = last
                    .as_ref()
                    .is_some_and(|last| compare(&self.order, &keys, last) != Ordering::Less);
                (!after_last).then_some(Row {
                    chunk: index,
                    row,
                    keys,
                })
            }));
            if rows.len() == before {
                continue;
            }
            let counted = self.memory.add_with(|| {
                chunk_bytes(&chunk)
                    + chunk.row_count()
                        * (std::mem::size_of::<Row>()
                            + self.order.len() * std::mem::size_of::<Option<SparqlOrderKey>>())
            });
            chunks.push(chunk);
            if counted.is_err() {
                self.spill_rows(&mut runs, &mut chunks, &mut rows)?;
            }
        }
        self.merged(runs, chunks, rows, ctx, k)
    }
}

impl Operator for RdfSortOperator {
    fn next(&mut self) -> OperatorResult {
        let sorted = match self.sorted.take() {
            Some(sorted) => sorted,
            None => match self.top_k {
                Some(k) => self.top(k)?,
                None => self.sort()?,
            },
        };
        let sorted = self.sorted.insert(sorted);
        let width = self.output_types.len();
        #[cfg(feature = "spill")]
        if let Sorted::Merged { merge, left } = sorted {
            if *left == 0 {
                return Ok(None);
            }
            let Some(mut batch) = merge.next_batch()? else {
                return Ok(None);
            };
            batch.truncate(*left);
            *left -= batch.len();
            let done = *left == 0;
            let mut builder = DataChunkBuilder::with_capacity(&self.output_types, batch.len());
            for row in batch {
                for (column, value) in row.into_iter().take(width).enumerate() {
                    if let Some(target) = builder.column_mut(column) {
                        target.push_value(value);
                    }
                }
                builder.advance_row();
            }
            // A top-k that returned its `k` rows ends its merge: its buffers,
            // grant and run files go now, not when the operator drops.
            if done {
                *sorted = Sorted::Kept(Vec::new());
            }
            return Ok(Some(builder.finish()));
        }
        if self.position >= sorted.len() {
            return Ok(None);
        }
        let mut builder = DataChunkBuilder::with_capacity(&self.output_types, 2048);
        while self.position < sorted.len() && !builder.is_full() {
            match sorted {
                Sorted::Chunks { chunks, rows } => {
                    let row = &rows[self.position];
                    let source = &chunks[row.chunk];
                    for column in 0..width {
                        let value = value_at(source, column, row.row);
                        if let Some(target) = builder.column_mut(column) {
                            target.push_value(value);
                        }
                    }
                }
                Sorted::Kept(rows) => {
                    let values = std::mem::take(&mut rows[self.position].values);
                    for (column, value) in values.into_iter().take(width).enumerate() {
                        if let Some(target) = builder.column_mut(column) {
                            target.push_value(value);
                        }
                    }
                }
                #[cfg(feature = "spill")]
                Sorted::Merged { .. } => unreachable!("a merge returns above"),
            }
            builder.advance_row();
            self.position += 1;
        }
        Ok(Some(builder.finish()))
    }

    fn reset(&mut self) {
        self.child.reset();
        self.sorted = None;
        self.position = 0;
        self.memory.clear();
    }

    fn name(&self) -> &'static str {
        "Sort"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use grafeo_common::memory::buffer::BufferManager;
    use grafeo_core::execution::memory::{
        self, OperatorMemoryContext, StatementMemory, chunk_heap_bytes,
    };
    use grafeo_core::execution::vector::ValueVector;
    use grafeo_core::graph::rdf::{Literal, Term};

    const CHUNK_ROWS: i64 = 1024;

    /// `chunks` chunks of `CHUNK_ROWS` rows: a returned column of the row's
    /// number, and a key column of an integer literal with ten values, so
    /// most rows tie.
    struct Numbered {
        next: i64,
        left: usize,
    }

    impl Operator for Numbered {
        fn next(&mut self) -> OperatorResult {
            if self.left == 0 {
                return Ok(None);
            }
            self.left -= 1;
            let numbers: Vec<i64> = (self.next..self.next + CHUNK_ROWS).collect();
            self.next += CHUNK_ROWS;
            let returned: Vec<Value> = numbers.iter().map(|&n| Value::Int64(n)).collect();
            let keys: Vec<Value> = numbers
                .iter()
                .map(|n| Value::RdfTerm(Arc::new(Term::Literal(Literal::integer(n % 10)))))
                .collect();
            Ok(Some(DataChunk::new(vec![
                ValueVector::from_values(&returned),
                ValueVector::from_values(&keys),
            ])))
        }

        fn reset(&mut self) {}

        fn name(&self) -> &'static str {
            "Numbered"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    /// A sort of `chunks` chunks by the key column descending, keeping the
    /// first `top_k`, built in a statement capped at `limit`.
    fn sort_of(chunks: usize, top_k: Option<usize>, limit: usize) -> RdfSortOperator {
        let ctx = OperatorMemoryContext::new(BufferManager::with_budget(256 << 20))
            .with_query_limit(limit);
        let _scope = memory::enter(StatementMemory::Budgeted(ctx));
        RdfSortOperator::new(
            Box::new(Numbered {
                next: 0,
                left: chunks,
            }),
            vec![OrderKey {
                column: 1,
                descending: true,
                nulls_first: false,
            }],
            vec![LogicalType::Int64],
            Arc::new(QueryTermIds::default()),
            top_k,
        )
    }

    fn numbers(operator: &mut RdfSortOperator) -> Result<Vec<i64>, OperatorError> {
        let mut out = Vec::new();
        while let Some(chunk) = operator.next()? {
            let column = chunk.column(0).expect("the returned column");
            out.extend(
                chunk
                    .selected_indices()
                    .map(|row| column.get_int64(row).expect("a number")),
            );
        }
        Ok(out)
    }

    /// Over 64 chunks the top-k keeps at most `k` rows: what it counts stays
    /// within `k` rows plus one chunk, under a cap a full sort of the same
    /// input passes, and its rows are the full sort's first `k`, the ties
    /// (every key value has 6,554 rows) in input order across chunk
    /// boundaries.
    #[test]
    fn a_top_k_keeps_k_rows_and_ties_in_input_order() {
        let chunks = 64;
        let k = 1_500;
        let mut full = sort_of(chunks, None, usize::MAX);
        let want: Vec<i64> = numbers(&mut full).unwrap().into_iter().take(k).collect();
        // Key 9 descending first, in input order: 9, 19, 29 and on, past the
        // first chunk.
        assert_eq!(want.first(), Some(&9));
        assert!(want.windows(2).all(|pair| pair[0] + 10 == pair[1]));
        assert!(*want.last().unwrap() > CHUNK_ROWS);

        let mut top = sort_of(chunks, Some(k), usize::MAX);
        assert_eq!(numbers(&mut top).unwrap(), want);

        // A kept row: its entry, its key's slot, the slots of its returned
        // value and its key column's value, and what its key's term holds on
        // the heap, as one chunk's rows hold it on average.
        let chunk = Numbered { next: 0, left: 1 }.next().unwrap().unwrap();
        let one_chunk = chunk_bytes(&chunk);
        let kept_row = std::mem::size_of::<Kept>()
            + std::mem::size_of::<Option<SparqlOrderKey>>()
            + 2 * std::mem::size_of::<Value>()
            + chunk_heap_bytes(&chunk).div_ceil(chunk.row_count());
        // The heap's capacity is at most twice its rows.
        let bound = 2 * k * kept_row + one_chunk;
        assert!(
            top.memory.bytes() <= bound,
            "{} counted, bound {bound}",
            top.memory.bytes()
        );
        assert!(full.memory.bytes() > 4 * bound, "the full sort holds more");

        // Under a cap the full sort is past, the top-k answers alike.
        let cap = full.memory.bytes() / 4;
        let error = numbers(&mut sort_of(chunks, None, cap)).expect_err("past the cap");
        assert!(error.to_string().contains("query memory limit"), "{error}");
        assert_eq!(numbers(&mut sort_of(chunks, Some(k), cap)).unwrap(), want);
    }

    /// A `k` of 0 keeps nothing; a `k` past the row count keeps every row,
    /// the full sort's order.
    #[test]
    fn a_top_k_of_none_or_of_every_row() {
        assert_eq!(
            numbers(&mut sort_of(3, Some(0), usize::MAX)).unwrap(),
            Vec::<i64>::new()
        );
        let full = numbers(&mut sort_of(3, None, usize::MAX)).unwrap();
        assert_eq!(
            numbers(&mut sort_of(3, Some(10_000), usize::MAX)).unwrap(),
            full
        );
    }

    /// The sorts that spill: only with spilling built in.
    #[cfg(feature = "spill")]
    mod spilled {
        use super::*;

        use grafeo_core::execution::spill::SpillManager;

        /// As [`sort_of`], in a statement whose spill path is `spill`'s.
        fn spilling(
            chunks: usize,
            top_k: Option<usize>,
            limit: usize,
            spill: Option<&Arc<SpillManager>>,
        ) -> RdfSortOperator {
            let ctx = OperatorMemoryContext::new(BufferManager::with_budget(256 << 20))
                .with_spill_manager(spill.cloned())
                .with_query_limit(limit);
            let _scope = memory::enter(StatementMemory::Budgeted(ctx));
            RdfSortOperator::new(
                Box::new(Numbered {
                    next: 0,
                    left: chunks,
                }),
                vec![OrderKey {
                    column: 1,
                    descending: true,
                    nulls_first: false,
                }],
                vec![LogicalType::Int64],
                Arc::new(QueryTermIds::default()),
                top_k,
            )
        }

        /// A spill directory and its manager.
        fn spill_dir() -> (tempfile::TempDir, Arc<SpillManager>) {
            let dir = tempfile::tempdir().unwrap();
            let manager = Arc::new(SpillManager::new(dir.path().join("spill")).unwrap());
            (dir, manager)
        }

        fn files_left(dir: &tempfile::TempDir) -> usize {
            std::fs::read_dir(dir.path().join("spill")).map_or(0, |entries| entries.count())
        }

        /// The merge of a sort that spilled, after its first `next`.
        fn merge_of(operator: &RdfSortOperator) -> &RunMerge<SparqlRowOrder> {
            match &operator.sorted {
                Some(Sorted::Merged { merge, .. }) => merge,
                _ => panic!("the sort did not spill"),
            }
        }

        /// Past a limit far below its input, with a spill path, the full sort
        /// spills more runs than one pass merges, and answers as in memory, ties
        /// (every key value has 6,554 rows) in input order. While it merges, its
        /// grant and the statement's memory stay within the limit; at the end no
        /// file is left.
        #[test]
        fn a_spilled_sort_answers_as_in_memory_within_the_limit() {
            let chunks = 64;
            let want = numbers(&mut sort_of(chunks, None, usize::MAX)).unwrap();
            let (dir, manager) = spill_dir();
            let limit = 192 << 10;
            let mut operator = spilling(chunks, None, limit, Some(&manager));
            let mut got = Vec::new();
            while let Some(chunk) = operator.next().unwrap() {
                let merge = merge_of(&operator);
                assert!(merge.passes() > 1, "{} passes", merge.passes());
                assert!(merge.memory().granted() <= limit);
                let used = operator.spill.as_ref().unwrap().query().used();
                assert!(used <= limit, "{used} of {limit} in use");
                let column = chunk.column(0).unwrap();
                got.extend(
                    chunk
                        .selected_indices()
                        .map(|row| column.get_int64(row).unwrap()),
                );
            }
            assert_eq!(got, want);
            assert_eq!(files_left(&dir), 0);
        }

        /// A spilled sort dropped while it merges (the query was cancelled, or
        /// failed downstream) leaves no file.
        #[test]
        fn a_spilled_sort_dropped_midway_leaves_no_file() {
            let (dir, manager) = spill_dir();
            let mut operator = spilling(64, None, 192 << 10, Some(&manager));
            assert!(operator.next().unwrap().is_some());
            assert!(files_left(&dir) > 0);
            drop(operator);
            assert_eq!(files_left(&dir), 0);
        }

        /// A top-k whose `k` outgrows the limit, with a spill path, answers as the
        /// full sort's first `k`: at `k` near the row count, at it and past it.
        /// The keys have ten values over 65,536 rows, so the switch to the spill
        /// falls inside a run of ties. A top-k that returned its `k` rows has
        /// ended its merge: no file is left while the operator lives on.
        #[test]
        fn a_top_k_past_its_grant_spills_and_answers_as_the_full_sort() {
            let chunks = 64;
            let rows = chunks * usize::try_from(CHUNK_ROWS).unwrap();
            let full = numbers(&mut sort_of(chunks, None, usize::MAX)).unwrap();
            for k in [rows - 1_000, rows, rows + 10] {
                let (dir, manager) = spill_dir();
                let mut operator = spilling(chunks, Some(k), 192 << 10, Some(&manager));
                let first = operator.next().unwrap().expect("rows");
                assert!(
                    matches!(operator.sorted, Some(Sorted::Merged { .. })),
                    "k {k}: spilled"
                );
                let column = first.column(0).unwrap();
                let mut got: Vec<i64> = first
                    .selected_indices()
                    .map(|row| column.get_int64(row).unwrap())
                    .collect();
                got.extend(numbers(&mut operator).unwrap());
                assert_eq!(got, full[..k.min(rows)], "k {k}");
                assert_eq!(files_left(&dir), 0, "k {k}: the merge ended");
            }
        }

        /// Returns its chunks, then nothing.
        struct Listed(std::collections::VecDeque<DataChunk>);

        impl Operator for Listed {
            fn next(&mut self) -> OperatorResult {
                Ok(self.0.pop_front())
            }

            fn reset(&mut self) {}

            fn name(&self) -> &'static str {
                "Listed"
            }

            fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
                self
            }
        }

        /// A row of the threshold test: its id, the length of its text, its key.
        type Spec = (i64, usize, Option<i64>);

        /// A chunk of rows: a returned id, a returned text of the given length,
        /// and a key column of an integer literal or unbound.
        fn chunk_of(rows: &[Spec]) -> DataChunk {
            let ids: Vec<Value> = rows.iter().map(|&(id, _, _)| Value::Int64(id)).collect();
            let texts: Vec<Value> = rows
                .iter()
                .map(|&(_, len, _)| Value::from("x".repeat(len)))
                .collect();
            let keys: Vec<Value> = rows
                .iter()
                .map(|&(_, _, key)| {
                    key.map_or(Value::Null, |n| {
                        Value::RdfTerm(Arc::new(Term::Literal(Literal::integer(n))))
                    })
                })
                .collect();
            DataChunk::new(vec![
                ValueVector::from_values(&ids),
                ValueVector::from_values(&texts),
                ValueVector::from_values(&keys),
            ])
        }

        /// A sort of `chunks` by `key`, keeping the first `top_k`, in a statement
        /// capped at `limit` whose spill path is `spill`'s.
        fn sort_listed(
            chunks: &[Vec<Spec>],
            key: OrderKey,
            top_k: Option<usize>,
            limit: usize,
            spill: &Arc<SpillManager>,
        ) -> RdfSortOperator {
            let ctx = OperatorMemoryContext::new(BufferManager::with_budget(256 << 20))
                .with_spill_manager(Some(Arc::clone(spill)))
                .with_query_limit(limit);
            let _scope = memory::enter(StatementMemory::Budgeted(ctx));
            RdfSortOperator::new(
                Box::new(Listed(chunks.iter().map(|rows| chunk_of(rows)).collect())),
                vec![key],
                vec![LogicalType::Int64, LogicalType::String],
                Arc::new(QueryTermIds::default()),
                top_k,
            )
        }

        /// The ids `operator` returns, in order.
        fn ids(operator: &mut RdfSortOperator) -> Vec<i64> {
            numbers(operator).unwrap()
        }

        /// Whether key `a` comes strictly before key `b` in `key`'s order.
        fn before(a: Option<i64>, b: Option<i64>, key: OrderKey) -> bool {
            match (a, b) {
                (None, None) => false,
                (None, Some(_)) => key.nulls_first,
                (Some(_), None) => !key.nulls_first,
                (Some(a), Some(b)) if key.descending => a > b,
                (Some(a), Some(b)) => a < b,
            }
        }

        /// `chunks` chunks of 1,024 rows, ids from `*next` on, each with a text
        /// of `len` bytes and the key `key_of` gives its id.
        fn rows_of(
            next: &mut i64,
            chunks: usize,
            len: usize,
            key_of: impl Fn(i64) -> Option<i64>,
        ) -> Vec<Vec<Spec>> {
            (0..chunks)
                .map(|_| {
                    (0..1024)
                        .map(|_| {
                            *next += 1;
                            (*next, len, key_of(*next))
                        })
                        .collect()
                })
                .collect()
        }

        /// The threshold of a top-k that passes its grant with its heap full: the
        /// rows of the rest of the input at or after its `k`-th kept row are not
        /// written, the ones before it are, and the answer is the full sort's
        /// first `k`. Ascending with unbound first and descending with unbound
        /// last. The input: three chunks of light rows (A), then one chunk of
        /// heavier rows that come first in the order (B), which replace kept rows
        /// in the full heap and take it past the limit right after B, then light
        /// rows (C) tied with the `k`-th kept row, before it, after it and
        /// unbound.
        #[test]
        fn a_full_top_k_drops_the_rows_at_or_after_its_last() {
            const K: usize = 2_500;
            for key in [
                OrderKey {
                    column: 2,
                    descending: false,
                    nulls_first: true,
                },
                OrderKey {
                    column: 2,
                    descending: true,
                    nulls_first: false,
                },
            ] {
                let mut next = 0;
                let better = if key.descending { 90 } else { 10 };
                let a = rows_of(&mut next, 3, 1, |id| Some(50 + id % 5));
                let b = rows_of(&mut next, 1, 2048, |id| {
                    (id % 7 != 0).then_some(better + id % 3)
                });
                let key_by_id: std::collections::HashMap<i64, Option<i64>> = a
                    .iter()
                    .chain(&b)
                    .flatten()
                    .map(|&(id, _, key)| (id, key))
                    .collect();
                let (_dir, manager) = spill_dir();
                let ab: Vec<Vec<Spec>> = a.iter().chain(&b).cloned().collect();
                let kth = ids(&mut sort_listed(&ab, key, None, usize::MAX, &manager))[K - 1];
                let last = key_by_id[&kth];
                let last_n = last.expect("the k-th kept row has a key");
                let step = if key.descending { 1 } else { -1 };
                let c = rows_of(&mut next, 2, 1, |id| match id % 8 {
                    0..=3 => last,
                    4 | 5 => Some(last_n + step),
                    6 => Some(last_n - step),
                    _ => None,
                });
                let all: Vec<Vec<Spec>> = ab.iter().chain(&c).cloned().collect();
                let want: Vec<i64> = ids(&mut sort_listed(&all, key, None, usize::MAX, &manager))
                    .into_iter()
                    .take(K)
                    .collect();
                let written_from_c = c
                    .iter()
                    .flatten()
                    .filter(|&&(_, _, row_key)| before(row_key, last, key))
                    .count();
                assert!(
                    written_from_c > 0 && written_from_c < 2048,
                    "{written_from_c}"
                );

                // The limit: past what the heap holds after A, short of what it
                // holds after B, so it passes its grant full, after B.
                let held = |input: &[Vec<Spec>]| {
                    let mut operator = sort_listed(input, key, Some(K), usize::MAX, &manager);
                    operator.next().unwrap();
                    operator.memory.bytes()
                };
                let (after_a, after_b) = (held(&a), held(&ab));
                // Half a MiB on either side of the limit, past any grant step.
                assert!(after_b > after_a + (1 << 20), "{after_a} then {after_b}");
                let limit = after_a + (after_b - after_a) / 2;

                let (dir, manager) = spill_dir();
                let mut operator = sort_listed(&all, key, Some(K), limit, &manager);
                let first = operator.next().unwrap().expect("rows");
                let Some(Sorted::Merged { merge, .. }) = &operator.sorted else {
                    panic!("{key:?}: the top-k did not spill");
                };
                assert_eq!(
                    merge.rows(),
                    u64::try_from(K + written_from_c).unwrap(),
                    "{key:?}: the kept rows and the rows of C before the last kept"
                );
                let column = first.column(0).unwrap();
                let mut got: Vec<i64> = first
                    .selected_indices()
                    .map(|row| column.get_int64(row).unwrap())
                    .collect();
                got.extend(ids(&mut operator));
                assert_eq!(got, want, "{key:?}");
                assert_eq!(files_left(&dir), 0, "{key:?}");
            }
        }

        /// A top-k whose statement has no spill path keeps only the returned
        /// values of a row (no key columns: its rows never become a run); with one
        /// it keeps the key columns' values too.
        #[test]
        fn a_kept_row_carries_its_keys_only_with_a_spill_path() {
            let (_dir, manager) = spill_dir();
            for (mut operator, width) in [
                (sort_of(1, Some(10), usize::MAX), 1),
                (spilling(1, Some(10), usize::MAX, Some(&manager)), 2),
            ] {
                let Sorted::Kept(rows) = operator.top(10).unwrap() else {
                    panic!("an in-memory top-k");
                };
                assert_eq!(rows.len(), 10);
                assert!(rows.iter().all(|row| row.values.len() == width), "{width}");
            }
        }
    }
}
