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

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;

use grafeo_common::types::{LogicalType, SparqlOrderKey, Value};
use grafeo_core::execution::DataChunk;
use grafeo_core::execution::chunk::DataChunkBuilder;
use grafeo_core::execution::memory::{HeapColumns, chunk_bytes};
use grafeo_core::execution::operators::{Operator, OperatorError, OperatorResult};

use super::term_ids::{QueryTermIds, id_of};
use super::terms;

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

/// The keys of `row` of `chunk` by `order` (see the module docs).
fn row_keys(order: &[OrderKey], terms: &QueryTermIds, chunk: &DataChunk, row: usize) -> RowKeys {
    order
        .iter()
        .map(|key| {
            let value = chunk.column(key.column)?.get_value(row)?;
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
        })
        .collect()
}

/// A row of the full sort, by its chunk and position, with its keys.
struct Row {
    chunk: usize,
    row: usize,
    keys: RowKeys,
}

/// A row the top-k keeps: its returned values and keys, taken out of its
/// chunk.
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
}

impl Sorted {
    fn len(&self) -> usize {
        match self {
            Self::Chunks { rows, .. } => rows.len(),
            Self::Kept(rows) => rows.len(),
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
    /// nothing outside a budgeted statement. The sort cannot spill: past the
    /// query memory limit or the memory limit it fails with `GRAFEO-S001`, a
    /// top-k too once it holds that much (a `k` near the row count).
    memory: grafeo_core::execution::OperatorMemory,
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
        }
    }

    /// Every row, sorted.
    fn sort(&mut self) -> Result<Sorted, OperatorError> {
        let mut chunks = Vec::new();
        let mut rows = Vec::new();
        while let Some(chunk) = self.child.next()? {
            let index = chunks.len();
            rows.extend(chunk.selected_indices().map(|row| Row {
                chunk: index,
                row,
                keys: row_keys(&self.order, &self.terms, &chunk, row),
            }));
            // The chunk and its rows' keys, against the statement's memory: a
            // chunk at a time, a compare within the grant.
            self.memory.add_with(|| {
                chunk_bytes(&chunk)
                    + chunk.row_count()
                        * (std::mem::size_of::<Row>()
                            + self.order.len() * std::mem::size_of::<Option<SparqlOrderKey>>())
            })?;
            chunks.push(chunk);
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
            + width * std::mem::size_of::<Value>();
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
                    values: (0..width)
                        .map(|column| {
                            chunk
                                .column(column)
                                .and_then(|c| c.get_value(row))
                                .unwrap_or(Value::Null)
                        })
                        .collect(),
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
            self.memory.set(
                heap.capacity()
                    .saturating_mul(kept_row_bytes)
                    .saturating_add(kept_heap_bytes),
            )?;
        }
        Ok(Sorted::Kept(heap.into_sorted_vec()))
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
        if self.position >= sorted.len() {
            return Ok(None);
        }
        let width = self.output_types.len();
        let mut builder = DataChunkBuilder::with_capacity(&self.output_types, 2048);
        while self.position < sorted.len() && !builder.is_full() {
            match sorted {
                Sorted::Chunks { chunks, rows } => {
                    let row = &rows[self.position];
                    let source = &chunks[row.chunk];
                    for column in 0..width {
                        let value = source
                            .column(column)
                            .and_then(|c| c.get_value(row.row))
                            .unwrap_or(Value::Null);
                        if let Some(target) = builder.column_mut(column) {
                            target.push_value(value);
                        }
                    }
                }
                Sorted::Kept(rows) => {
                    let values = std::mem::take(&mut rows[self.position].values);
                    for (column, value) in values.into_iter().enumerate() {
                        if let Some(target) = builder.column_mut(column) {
                            target.push_value(value);
                        }
                    }
                }
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

        // A kept row: its entry, key and value slots, and what its key's
        // term holds on the heap, as one chunk's rows hold it on average.
        let chunk = Numbered { next: 0, left: 1 }.next().unwrap().unwrap();
        let one_chunk = chunk_bytes(&chunk);
        let kept_row = std::mem::size_of::<Kept>()
            + std::mem::size_of::<Option<SparqlOrderKey>>()
            + std::mem::size_of::<Value>()
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
}
