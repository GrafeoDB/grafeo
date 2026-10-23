//! ORDER BY of a SPARQL query (SPARQL 1.1 section 15.1): each row's keys are
//! computed once, a literal's value parsed once, and the rows sorted by them.
//!
//! A variable's key is its id column, keyed by the term and the value the
//! dictionary keeps for it (so a literal is not parsed again); an
//! expression's key is a term the plan projected after the returned columns,
//! which are the only ones the sort returns.

use std::cmp::Ordering;
use std::sync::Arc;

use grafeo_common::types::{LogicalType, SparqlOrderKey, Value};
use grafeo_core::execution::DataChunk;
use grafeo_core::execution::chunk::DataChunkBuilder;
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

/// A row to return, by its chunk and position, with its keys.
struct Row {
    chunk: usize,
    row: usize,
    keys: Vec<Option<SparqlOrderKey>>,
}

/// Sorts its input by SPARQL order keys (see the module docs).
pub(super) struct RdfSortOperator {
    child: Box<dyn Operator>,
    keys: Vec<OrderKey>,
    /// The types of the returned columns, which come first.
    output_types: Vec<LogicalType>,
    terms: Arc<QueryTermIds>,
    chunks: Vec<DataChunk>,
    rows: Vec<Row>,
    sorted: bool,
    position: usize,
}

impl RdfSortOperator {
    /// Sorts `child`'s rows by `keys`, returning the columns of
    /// `output_types`.
    pub(super) fn new(
        child: Box<dyn Operator>,
        keys: Vec<OrderKey>,
        output_types: Vec<LogicalType>,
        terms: Arc<QueryTermIds>,
    ) -> Self {
        Self {
            child,
            keys,
            output_types,
            terms,
            chunks: Vec::new(),
            rows: Vec::new(),
            sorted: false,
            position: 0,
        }
    }

    fn sort(&mut self) -> Result<(), OperatorError> {
        while let Some(chunk) = self.child.next()? {
            let index = self.chunks.len();
            for row in chunk.selected_indices() {
                let keys = self
                    .keys
                    .iter()
                    .map(|key| {
                        let value = chunk.column(key.column)?.get_value(row)?;
                        match value {
                            Value::Null => None,
                            Value::Int64(_) => {
                                let id = id_of(&value)?;
                                SparqlOrderKey::of_inline(id).or_else(|| {
                                    let entry = self.terms.entry(id)?;
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
                    .collect();
                self.rows.push(Row {
                    chunk: index,
                    row,
                    keys,
                });
            }
            self.chunks.push(chunk);
        }
        let order = &self.keys;
        self.rows.sort_by(|a, b| {
            for (key, (x, y)) in order.iter().zip(a.keys.iter().zip(&b.keys)) {
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
        });
        self.sorted = true;
        Ok(())
    }
}

impl Operator for RdfSortOperator {
    fn next(&mut self) -> OperatorResult {
        if !self.sorted {
            self.sort()?;
        }
        if self.position >= self.rows.len() {
            return Ok(None);
        }
        let width = self.output_types.len();
        let mut builder = DataChunkBuilder::with_capacity(&self.output_types, 2048);
        while self.position < self.rows.len() && !builder.is_full() {
            let row = &self.rows[self.position];
            let source = &self.chunks[row.chunk];
            for column in 0..width {
                let value = source
                    .column(column)
                    .and_then(|c| c.get_value(row.row))
                    .unwrap_or(Value::Null);
                if let Some(target) = builder.column_mut(column) {
                    target.push_value(value);
                }
            }
            builder.advance_row();
            self.position += 1;
        }
        Ok(Some(builder.finish()))
    }

    fn reset(&mut self) {
        self.child.reset();
        self.chunks.clear();
        self.rows.clear();
        self.sorted = false;
        self.position = 0;
    }

    fn name(&self) -> &'static str {
        "Sort"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}
