//! The named graphs of the dataset as rows: the physical operator of
//! [`NamedGraphsOp`](crate::query::plan::NamedGraphsOp).

use std::sync::Arc;

use grafeo_common::types::LogicalType;
use grafeo_common::types::rdf::Term;
use grafeo_core::execution::DataChunk;
use grafeo_core::execution::operators::{Operator, OperatorResult};
use grafeo_core::graph::rdf::{RdfStore, TriplePattern};

use super::term_ids::{QueryTermIds, id_value};
use crate::query::plan::DatasetRestriction;
use crate::transaction::RdfWriter;

/// One row per named graph, its IRI as a term id. With FROM NAMED the
/// named graphs are the ones it lists; otherwise the store's, with the
/// graphs the transaction wrote to but has not committed.
pub(super) struct NamedGraphsOperator {
    store: Arc<RdfStore>,
    writer: Option<RdfWriter>,
    dataset: Option<DatasetRestriction>,
    terms: Arc<QueryTermIds>,
    done: bool,
}

/// The IRIs of the named graphs of the dataset the store, the transaction's writer and FROM NAMED
/// make, sorted and without duplicates: with FROM NAMED those it lists, held or not; otherwise the
/// store's, with the graphs the transaction wrote to but has not committed.
pub(super) fn named_graph_names(
    store: &RdfStore,
    writer: Option<&RdfWriter>,
    dataset: Option<&DatasetRestriction>,
) -> Vec<String> {
    let mut names = match dataset {
        Some(dataset) if !dataset.named_graphs.is_empty() => dataset.named_graphs.clone(),
        _ => {
            let mut names = store.graph_names();
            if let Some(writer) = writer.filter(|w| w.has_pending()) {
                let any = TriplePattern {
                    subject: None,
                    predicate: None,
                    object: None,
                };
                names.extend(
                    writer
                        .find(&any, Some(&[]))
                        .into_iter()
                        .filter_map(|(graph, _)| graph),
                );
            }
            names
        }
    };
    names.sort_unstable();
    names.dedup();
    names
}

impl NamedGraphsOperator {
    pub(super) fn new(
        store: Arc<RdfStore>,
        writer: Option<RdfWriter>,
        dataset: Option<DatasetRestriction>,
        terms: Arc<QueryTermIds>,
    ) -> Self {
        Self {
            store,
            writer,
            dataset,
            terms,
            done: false,
        }
    }
}

impl Operator for NamedGraphsOperator {
    fn next(&mut self) -> OperatorResult {
        if self.done {
            return Ok(None);
        }
        self.done = true;
        let names = named_graph_names(&self.store, self.writer.as_ref(), self.dataset.as_ref());
        if names.is_empty() {
            return Ok(None);
        }
        let mut chunk = DataChunk::with_capacity(&[LogicalType::RdfTermId], names.len());
        let column = chunk.column_mut(0).expect("the chunk has the graph column");
        for name in names {
            let id = self.terms.intern(&Term::iri(name))?;
            column.push_value(id_value(id));
        }
        let count = column.len();
        chunk.set_count(count);
        Ok(Some(chunk))
    }

    fn reset(&mut self) {
        self.done = false;
    }

    fn name(&self) -> &'static str {
        "NamedGraphs"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}
