//! The shared left input of a delim join (DuckDB's name): the rows of a
//! semi or anti join's left side, read once, and the distinct values of the
//! columns the right side shares with it.
//!
//! A decorrelated `EXISTS { ... }` runs its subquery once, starting from the
//! distinct values of the variables it shares with the outer rows, instead of
//! once per row or once over every node: the subquery never sees a value no
//! outer row holds. [`DelimValuesOperator`] gives those values to the
//! subquery's plan, [`DelimReplayOperator`] gives the left rows to the join.

use std::collections::{HashSet, VecDeque};
use std::sync::Arc;

use grafeo_common::types::{HashableValue, LogicalType};
use parking_lot::Mutex;

use super::{DataChunk, Operator, OperatorResult};
use crate::execution::chunk::DEFAULT_CHUNK_SIZE;
use crate::execution::vector::ValueVector;

/// The left input of a delim join, read the first time either side asks.
pub struct DelimBuffer {
    state: Mutex<DelimState>,
}

struct DelimState {
    /// The left input.
    source: Box<dyn Operator>,
    /// The positions of the shared columns in its rows.
    key_columns: Vec<usize>,
    /// Whether `source` has been read since the last reset.
    filled: bool,
    /// The left rows not yet given to the join.
    chunks: VecDeque<DataChunk>,
    /// The distinct values of the key columns, in chunks.
    distinct: Vec<DataChunk>,
    /// Counts the fills, so a reader of `distinct` sees a refill.
    generation: u64,
}

impl DelimBuffer {
    /// A buffer over `source`, whose columns `key_columns` the right side
    /// shares.
    #[must_use]
    pub fn new(source: Box<dyn Operator>, key_columns: Vec<usize>) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(DelimState {
                source,
                key_columns,
                filled: false,
                chunks: VecDeque::new(),
                distinct: Vec::new(),
                generation: 0,
            }),
        })
    }

    /// Reads the left input and collects its distinct key values, once.
    fn fill(state: &mut DelimState) -> Result<(), super::OperatorError> {
        if state.filled {
            return Ok(());
        }
        let mut seen: HashSet<Vec<HashableValue>> = HashSet::new();
        let mut types: Option<Vec<LogicalType>> = None;
        let mut builder: Option<Vec<ValueVector>> = None;
        let mut distinct = Vec::new();
        while let Some(chunk) = state.source.next()? {
            let types = types.get_or_insert_with(|| {
                state
                    .key_columns
                    .iter()
                    .map(|&column| {
                        chunk
                            .column(column)
                            .map_or(LogicalType::Any, |vector| vector.data_type().clone())
                    })
                    .collect()
            });
            for row in chunk.selected_indices() {
                let key: Vec<HashableValue> = state
                    .key_columns
                    .iter()
                    .map(|&column| {
                        HashableValue(
                            chunk
                                .column(column)
                                .and_then(|vector| vector.get_value(row))
                                .unwrap_or(grafeo_common::types::Value::Null),
                        )
                    })
                    .collect();
                if !seen.insert(key.clone()) {
                    continue;
                }
                let vectors = builder.get_or_insert_with(|| {
                    types
                        .iter()
                        .map(|data_type| {
                            ValueVector::with_capacity(data_type.clone(), DEFAULT_CHUNK_SIZE)
                        })
                        .collect()
                });
                for (vector, HashableValue(value)) in vectors.iter_mut().zip(key) {
                    vector.push_value(value);
                }
                if vectors.first().map_or(0, ValueVector::len) == DEFAULT_CHUNK_SIZE {
                    distinct.push(DataChunk::new(builder.take().unwrap_or_default()));
                }
            }
            state.chunks.push_back(chunk);
        }
        if let Some(vectors) = builder.take()
            && vectors.first().is_some_and(|vector| !vector.is_empty())
        {
            distinct.push(DataChunk::new(vectors));
        }
        state.distinct = distinct;
        state.filled = true;
        state.generation += 1;
        Ok(())
    }

    /// Forgets what was read, so the next read starts the left input again.
    fn reset(&self) {
        let mut state = self.state.lock();
        state.source.reset();
        state.filled = false;
        state.chunks.clear();
        state.distinct.clear();
    }
}

/// The left side of a delim join: the rows of the buffered left input.
pub struct DelimReplayOperator {
    buffer: Arc<DelimBuffer>,
}

impl DelimReplayOperator {
    /// The rows of `buffer`'s left input.
    #[must_use]
    pub fn new(buffer: Arc<DelimBuffer>) -> Self {
        Self { buffer }
    }
}

impl Operator for DelimReplayOperator {
    fn next(&mut self) -> OperatorResult {
        let mut state = self.buffer.state.lock();
        DelimBuffer::fill(&mut state)?;
        Ok(state.chunks.pop_front())
    }

    /// Starts the left input again: the join runs anew.
    fn reset(&mut self) {
        self.buffer.reset();
    }

    fn name(&self) -> &'static str {
        "DelimReplay"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// The leaf a decorrelated subquery starts from: the distinct values of the
/// shared columns of the buffered left input.
pub struct DelimValuesOperator {
    buffer: Arc<DelimBuffer>,
    next: usize,
    generation: u64,
}

impl DelimValuesOperator {
    /// The distinct key values of `buffer`.
    #[must_use]
    pub fn new(buffer: Arc<DelimBuffer>) -> Self {
        Self {
            buffer,
            next: 0,
            generation: 0,
        }
    }
}

impl Operator for DelimValuesOperator {
    fn next(&mut self) -> OperatorResult {
        let mut state = self.buffer.state.lock();
        DelimBuffer::fill(&mut state)?;
        if state.generation != self.generation {
            self.generation = state.generation;
            self.next = 0;
        }
        let chunk = state.distinct.get(self.next).cloned();
        self.next += 1;
        Ok(chunk)
    }

    /// Reads the values again; the left input is not read again (an operator
    /// above may scan this leaf more than once).
    fn reset(&mut self) {
        self.next = 0;
    }

    fn name(&self) -> &'static str {
        "DelimScan"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}
