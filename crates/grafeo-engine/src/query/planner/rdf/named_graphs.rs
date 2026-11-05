//! The named graphs of the dataset as rows: the physical operator of
//! [`NamedGraphsOp`](crate::query::plan::NamedGraphsOp).

use std::sync::Arc;

use grafeo_common::types::LogicalType;
use grafeo_common::types::rdf::Term;
use grafeo_core::execution::DataChunk;
use grafeo_core::execution::operators::{Operator, OperatorResult};
use grafeo_core::graph::rdf::{RdfStore, TriplePattern};

use super::term_ids::{QueryTermIds, id_value};
use crate::query::plan::{
    DatasetNamedGraphs, DatasetRestriction, LogicalExpression, LogicalOperator, NamedGraph,
    TripleComponent, graph_of_a_sub_select, map_sparql_expressions,
};
use crate::transaction::RdfWriter;

/// One row per named graph, its IRI as a term id. A dataset clause lists the
/// named graphs; otherwise they are the store's, with the graphs the
/// transaction wrote to but has not committed.
pub(super) struct NamedGraphsOperator {
    store: Arc<RdfStore>,
    writer: Option<RdfWriter>,
    dataset: Option<DatasetRestriction>,
    terms: Arc<QueryTermIds>,
    done: bool,
}

/// The IRIs of the named graphs of the dataset the store, the transaction's writer and the query's
/// clauses make, sorted and without duplicates: those `FROM NAMED` lists, held or not, none with
/// `FROM` alone; otherwise the store's, with the graphs the transaction wrote to but has not
/// committed.
pub(super) fn named_graph_names(
    store: &RdfStore,
    writer: Option<&RdfWriter>,
    dataset: Option<&DatasetRestriction>,
) -> Vec<String> {
    let mut names = match dataset.and_then(DatasetRestriction::listed_named_graphs) {
        Some(listed) => listed.to_vec(),
        None => {
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

/// Narrows `dataset`'s named graphs to the graph `name`.
pub(super) fn read_only_graph(dataset: &mut Option<DatasetRestriction>, name: &str) {
    dataset.get_or_insert_default().named_graphs =
        DatasetNamedGraphs::Listed(vec![name.to_string()]);
}

/// `plan` as it reads with `variable` the graph `name`: whatever reads that variable as its graph
/// lists that one graph, keeping the variable, which a filter beside it may read, and a sub-SELECT's
/// patterns read that graph as a constant, as the variable is not theirs. The pattern of an
/// `EXISTS` in it reads so too. A nested `GRAPH` of another graph stays as it is.
pub(super) fn in_one_graph(plan: LogicalOperator, variable: &str, name: &str) -> LogicalOperator {
    let own = graph_of_a_sub_select(variable);
    in_the_graph(plan, variable, &own, name)
}

fn in_the_graph(plan: LogicalOperator, variable: &str, own: &str, name: &str) -> LogicalOperator {
    let reads = |graph: &Option<TripleComponent>, reader: &str| {
        graph.as_ref().and_then(TripleComponent::as_variable) == Some(reader)
    };
    let plan = plan.map_children(|child| in_the_graph(child, variable, own, name));
    let mut plan = map_sparql_expressions(plan, &mut |expression| {
        in_the_graph_of_an_expression(expression, variable, own, name)
    });
    match &mut plan {
        LogicalOperator::TripleScan(scan) if reads(&scan.graph, own) => {
            scan.graph = Some(TripleComponent::Iri(name.to_string()));
        }
        LogicalOperator::PropertyPath(path) if reads(&path.graph, own) => {
            path.graph = Some(TripleComponent::Iri(name.to_string()));
        }
        LogicalOperator::TripleScan(scan) if reads(&scan.graph, variable) => {
            read_only_graph(&mut scan.dataset, name);
        }
        LogicalOperator::PropertyPath(path) if reads(&path.graph, variable) => {
            read_only_graph(&mut path.dataset, name);
        }
        LogicalOperator::NamedGraphs(graphs) if graphs.variable == variable => {
            read_only_graph(&mut graphs.dataset, name);
        }
        LogicalOperator::InNamedGraph(graph) if matches!(&graph.graph, NamedGraph::Variable(nested) if nested == variable) =>
        {
            read_only_graph(&mut graph.dataset, name);
        }
        _ => {}
    }
    plan
}

/// `expression` with the plan of each `EXISTS` in it as [`in_the_graph`] makes it.
fn in_the_graph_of_an_expression(
    expression: LogicalExpression,
    variable: &str,
    own: &str,
    name: &str,
) -> LogicalExpression {
    match expression {
        LogicalExpression::ExistsSubquery(plan) => {
            LogicalExpression::ExistsSubquery(Box::new(in_the_graph(*plan, variable, own, name)))
        }
        other => {
            other.map_children(|child| in_the_graph_of_an_expression(child, variable, own, name))
        }
    }
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
