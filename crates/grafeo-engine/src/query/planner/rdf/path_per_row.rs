//! A property path whose subject or object the rows before it bind
//! (`?a :id 1 . ?b :id 2 . ?a (:KNOWS|^:KNOWS)+ ?b`, LDBC IC13 as RDF): a
//! search from each row's term, instead of the closure over every term of
//! the graph joined with the rows after it.

use std::sync::Arc;

use grafeo_common::clock::Instant;
use grafeo_common::types::{LogicalType, Value};
use grafeo_core::execution::DataChunk;
use grafeo_core::execution::operators::{
    Operator, OperatorError, OperatorResult, RdfPathConfig, RdfPathOperator, RdfPathReadOverlay,
};
use grafeo_core::graph::rdf::{RdfStore, Term};

use super::intern_value;
use super::term_ids::QueryTermIds;

/// What every path search of a query reads with: the store, the memory
/// budget, the deadline and the transaction's pending changes.
#[derive(Clone)]
pub(super) struct PathSource {
    pub(super) store: Arc<RdfStore>,
    pub(super) budget: usize,
    pub(super) deadline: Option<Instant>,
    pub(super) overlay: Option<Arc<dyn RdfPathReadOverlay>>,
}

impl PathSource {
    /// The search `config` describes.
    pub(super) fn operator(&self, config: RdfPathConfig) -> RdfPathOperator {
        RdfPathOperator::new(Arc::clone(&self.store), config)
            .with_memory_budget(self.budget)
            .with_deadline(self.deadline)
            .with_read_overlay(self.overlay.clone())
    }
}

/// A join of a path with rows that bind an end of it, as a search per row:
/// the path with those ends left open, their columns in the rows, and the
/// columns the path adds.
pub(super) struct PathPerRowPlan {
    pub(super) config: RdfPathConfig,
    pub(super) subject_column: Option<usize>,
    pub(super) object_column: Option<usize>,
    pub(super) columns: Vec<String>,
}

/// The rows of `input`, each joined with the path's solutions that start (or
/// end) at the terms of its columns `subject_column` and `object_column`:
/// one search per row, from its bound end, which stops at the row's other end
/// when that is bound too. The output is the input's columns, then the path's
/// own (its free end, then its graph variable), as term ids.
pub(super) struct PathPerRowOperator {
    pub(super) input: Box<dyn Operator>,
    pub(super) source: PathSource,
    /// The path, with the ends the rows bind left open.
    pub(super) config: RdfPathConfig,
    pub(super) subject_column: Option<usize>,
    pub(super) object_column: Option<usize>,
    /// How many columns the path adds.
    pub(super) path_columns: usize,
    pub(super) terms: Arc<QueryTermIds>,
}

impl PathPerRowOperator {
    /// The term of `column` at `row`, `None` when it holds none (a row no
    /// path joins).
    fn term_at(&self, chunk: &DataChunk, column: usize, row: usize) -> Option<Term> {
        let value = chunk.column(column)?.get_value(row)?;
        let entry = self.terms.entry_of(&value)?;
        Some(Term::clone(entry.term()))
    }

    /// The ends of the search for `row`: each bound end's term, `None` for an
    /// end the rows leave open; `None` for a row without a term at a bound end.
    fn ends(&self, chunk: &DataChunk, row: usize) -> Option<[Option<Term>; 2]> {
        let mut ends = [None, None];
        for (end, column) in ends
            .iter_mut()
            .zip([self.subject_column, self.object_column])
        {
            if let Some(column) = column {
                *end = Some(self.term_at(chunk, column, row)?);
            }
        }
        Some(ends)
    }
}

impl Operator for PathPerRowOperator {
    fn next(&mut self) -> OperatorResult {
        loop {
            let Some(chunk) = self.input.next()? else {
                return Ok(None);
            };
            let mut joined: Vec<(usize, Vec<Value>)> = Vec::new();
            for row in chunk.selected_indices() {
                let Some([subject, object]) = self.ends(&chunk, row) else {
                    continue;
                };
                let mut config = self.config.clone();
                if subject.is_some() {
                    config.subject = subject;
                }
                if object.is_some() {
                    config.object = object;
                }
                let mut search = self.source.operator(config);
                while let Some(found) = search.next()? {
                    for at in found.selected_indices() {
                        let values = (0..self.path_columns)
                            .map(|column| {
                                intern_value(
                                    &self.terms,
                                    found.column(column).and_then(|c| c.get_value(at)),
                                )
                            })
                            .collect::<Result<Vec<_>, OperatorError>>()?;
                        joined.push((row, values));
                    }
                }
            }
            if joined.is_empty() {
                continue;
            }
            let mut schema: Vec<LogicalType> = (0..chunk.column_count())
                .map(|i| {
                    chunk
                        .column(i)
                        .map_or(LogicalType::Any, |c| c.data_type().clone())
                })
                .collect();
            schema.extend(std::iter::repeat_n(
                LogicalType::RdfTermId,
                self.path_columns,
            ));
            let mut out = DataChunk::with_capacity(&schema, joined.len());
            for column in 0..chunk.column_count() {
                let (Some(source), Some(target)) = (chunk.column(column), out.column_mut(column))
                else {
                    continue;
                };
                for (row, _) in &joined {
                    target.push_value(source.get_value(*row).unwrap_or(Value::Null));
                }
            }
            for column in 0..self.path_columns {
                let Some(target) = out.column_mut(chunk.column_count() + column) else {
                    continue;
                };
                for (_, values) in &joined {
                    target.push_value(values[column].clone());
                }
            }
            out.set_count(joined.len());
            return Ok(Some(out));
        }
    }

    fn reset(&mut self) {
        self.input.reset();
    }

    fn name(&self) -> &'static str {
        "RdfPropertyPathPerRow"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}
