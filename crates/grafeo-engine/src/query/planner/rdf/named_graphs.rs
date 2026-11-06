//! The named graphs of the dataset as rows: the physical operator of
//! [`NamedGraphsOp`](crate::query::plan::NamedGraphsOp).

use std::sync::Arc;

use grafeo_common::types::rdf::Term;
use grafeo_common::types::{LogicalType, Value};
use grafeo_core::execution::DataChunk;
use grafeo_core::execution::operators::{Operator, OperatorResult};
use grafeo_core::graph::rdf::RdfStore;

use super::term_ids::{QueryTermIds, id_value};
use crate::query::plan::{
    DatasetNamedGraphs, DatasetRestriction, GroupGraph, LogicalExpression, LogicalOperator,
    TripleComponent,
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
/// `FROM` alone; otherwise the store's, with the graphs the transaction inserted into and has not
/// committed, an emptied one too, as its commit creates them.
pub(super) fn named_graph_names(
    store: &RdfStore,
    writer: Option<&RdfWriter>,
    dataset: Option<&DatasetRestriction>,
) -> Vec<String> {
    let mut names = match dataset.and_then(DatasetRestriction::listed_named_graphs) {
        Some(listed) => listed.to_vec(),
        None => {
            let mut names = store.graph_names();
            if let Some(writer) = writer {
                names.extend(writer.graphs_inserted_into());
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

/// `plan` as it reads with `variable` the graph `name`: what reads the graph under the group's
/// hidden name (`read_as`: a sub-SELECT's patterns, whose names are their own, or all of the
/// pattern when it names the variable itself) reads that graph as a constant; otherwise what reads
/// the variable as its graph lists that one graph, keeping the variable, which a filter beside it
/// may read. The pattern of an `EXISTS` in it reads so too. Where the pattern names the variable,
/// a nested `GRAPH` of it stays free: the join with the graph's row decides (sec 18.6).
pub(super) fn in_one_graph(
    plan: LogicalOperator,
    variable: &str,
    read_as: Option<&GroupGraph>,
    name: &str,
) -> LogicalOperator {
    let reads = Reads {
        variable,
        own: read_as.map(|read_as| read_as.name.as_str()),
        whole_pattern: read_as.is_some_and(|read_as| read_as.whole_pattern),
    };
    in_the_graph(plan, &reads, name)
}

/// How the pattern of a `GRAPH` group reads its graph, for [`in_the_graph`].
struct Reads<'a> {
    /// The group's variable.
    variable: &'a str,
    /// The hidden name, if the group has one.
    own: Option<&'a str>,
    /// Whether all of the pattern reads the hidden name: a read of the variable is then a nested
    /// group's.
    whole_pattern: bool,
}

fn in_the_graph(plan: LogicalOperator, how: &Reads<'_>, name: &str) -> LogicalOperator {
    let reads = |graph: &Option<TripleComponent>, reader: Option<&str>| {
        reader.is_some() && graph.as_ref().and_then(TripleComponent::as_variable) == reader
    };
    let variable = (!how.whole_pattern).then_some(how.variable);
    let plan = plan.map_children(|child| in_the_graph(child, how, name));
    let mut plan = map_sparql_expressions(plan, &mut |expression| {
        in_the_graph_of_an_expression(expression, how, name)
    });
    match &mut plan {
        LogicalOperator::TripleScan(scan) if reads(&scan.graph, how.own) => {
            scan.graph = Some(TripleComponent::Iri(name.to_string()));
        }
        LogicalOperator::PropertyPath(path) if reads(&path.graph, how.own) => {
            path.graph = Some(TripleComponent::Iri(name.to_string()));
        }
        LogicalOperator::TripleScan(scan) if reads(&scan.graph, variable) => {
            read_only_graph(&mut scan.dataset, name);
        }
        LogicalOperator::PropertyPath(path) if reads(&path.graph, variable) => {
            read_only_graph(&mut path.dataset, name);
        }
        LogicalOperator::NamedGraphs(graphs)
            if Some(graphs.variable.as_str()) == how.own
                || Some(graphs.variable.as_str()) == variable =>
        {
            read_only_graph(&mut graphs.dataset, name);
        }
        _ => {}
    }
    plan
}

/// `plan` with `f` applied to each expression its own operator holds, of the operators a SPARQL
/// plan holds them in: a filter, a `BIND`, a projection, a sort, an aggregation and a left join's
/// condition: the list the SPARQL translator's `sparql_expressions` reads.
fn map_sparql_expressions(
    mut plan: LogicalOperator,
    f: &mut dyn FnMut(LogicalExpression) -> LogicalExpression,
) -> LogicalOperator {
    let mut apply = |expression: &mut LogicalExpression| {
        let taken = std::mem::replace(expression, LogicalExpression::Literal(Value::Null));
        *expression = f(taken);
    };
    match &mut plan {
        LogicalOperator::Filter(filter) => apply(&mut filter.predicate),
        LogicalOperator::Bind(bind) => apply(&mut bind.expression),
        LogicalOperator::Project(project) => {
            for projection in &mut project.projections {
                apply(&mut projection.expression);
            }
        }
        LogicalOperator::Sort(sort) => {
            for key in &mut sort.keys {
                apply(&mut key.expression);
            }
        }
        LogicalOperator::Aggregate(aggregate) => {
            for key in &mut aggregate.group_by {
                apply(key);
            }
            for function in &mut aggregate.aggregates {
                for argument in [&mut function.expression, &mut function.expression2]
                    .into_iter()
                    .flatten()
                {
                    apply(argument);
                }
            }
            if let Some(having) = &mut aggregate.having {
                apply(having);
            }
        }
        LogicalOperator::LeftJoin(join) => {
            if let Some(condition) = &mut join.condition {
                apply(condition);
            }
        }
        _ => {}
    }
    plan
}

/// `expression` with the plan of each `EXISTS` in it as [`in_the_graph`] makes it.
fn in_the_graph_of_an_expression(
    expression: LogicalExpression,
    how: &Reads<'_>,
    name: &str,
) -> LogicalExpression {
    match expression {
        LogicalExpression::ExistsSubquery(plan) => {
            LogicalExpression::ExistsSubquery(Box::new(in_the_graph(*plan, how, name)))
        }
        other => other.map_children(|child| in_the_graph_of_an_expression(child, how, name)),
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
