//! An `EXISTS` that no run without the row answers: planned and run for each distinct row, by the
//! SPARQL 1.2 draft's definition (17.4.1.4, 18.3.2.7), where each group starts from the row.
//!
//! The row reaches the pattern in two ways. Whatever makes a name the row binds, a scan, a path, a
//! `BIND` or an alias, is joined with the row's value ([`inject`]), so each operator keeps its
//! columns and everything above it stays valid. An expression reads a name it has no value for from
//! the row, through [`ExpressionContext`](super::evaluator::ExpressionContext).

use std::collections::HashSet;
use std::sync::Arc;

use grafeo_common::types::{LogicalType, Value};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use grafeo_core::execution::DataChunk;
use grafeo_core::execution::operators::{Operator, OperatorError, OperatorResult, StatementClock};
use grafeo_core::graph::rdf::RdfStore;

use super::exists_read::{RowNames, row_names_in};
use super::term_ids::QueryTermIds;
use super::{RdfPlanner, id_at};
use crate::query::plan::{
    JoinOp, JoinType, LogicalOperator, NamedGraph, RowValuesOp, TripleComponent, always_bound,
    collect_expression_variables,
};
use crate::transaction::RdfWriter;

/// The prefix of the name of a column that holds an `EXISTS` answer, which no query can write.
pub(super) const MARK_PREFIX: &str = "exists#";

/// The row an `EXISTS` is tested for: the values of the names its pattern has.
pub(super) struct RowBinding {
    /// The name's value, a term id as a column holds it. A name the row leaves unbound is not here.
    values: FxHashMap<String, Value>,
    /// Whether the row binds anything at all, read or not: a `MINUS` asks.
    binds_anything: bool,
}

impl RowBinding {
    /// The value of `name`, if the row binds it.
    pub(super) fn value(&self, name: &str) -> Option<&Value> {
        self.values.get(name)
    }

    /// Whether the row binds anything at all.
    pub(super) fn binds_anything(&self) -> bool {
        self.binds_anything
    }

    fn names(&self) -> impl Iterator<Item = &str> {
        self.values.keys().map(String::as_str)
    }
}

/// What it takes to plan a pattern for a row, kept by the operator that runs it.
#[derive(Clone)]
pub(super) struct RowPlanParts {
    pub(super) store: Arc<RdfStore>,
    pub(super) chunk_size: usize,
    pub(super) writer: Option<RdfWriter>,
    pub(super) terms: Arc<QueryTermIds>,
    pub(super) path_search_budget: usize,
    pub(super) deadline: Option<grafeo_common::clock::Instant>,
    pub(super) read_counter: Arc<grafeo_core::execution::operators::ReadCounter>,
    pub(super) warnings: Arc<grafeo_common::utils::warning::WarningSink>,
    pub(super) clock: StatementClock,
}

/// A name of the pattern that the row has: its column on each side of the row.
struct RowName {
    name: String,
    left: Option<usize>,
    right: Option<usize>,
}

/// One `EXISTS` tested per row: a row of the same values is answered from what was found.
pub(super) struct ExistsPerRow {
    parts: RowPlanParts,
    subquery: LogicalOperator,
    names: Vec<RowName>,
    /// Whether the pattern holds a `MINUS`, so the answer depends on whether the row binds anything.
    holds_minus: bool,
    /// The columns of the row a query can bind, on each side.
    left_columns: Vec<usize>,
    right_columns: Vec<usize>,
    /// The row around this one, when the test is inside another per-row pattern.
    outer: Option<Arc<RowBinding>>,
    /// The pattern with the row's values joined in, by which names the row binds.
    rewrites: FxHashMap<Vec<bool>, LogicalOperator>,
    /// The answer for the values of the names, and whether the row binds anything.
    answers: FxHashMap<(Vec<Option<i64>>, bool), bool>,
}

impl ExistsPerRow {
    /// Forgets the answers, for a plan run again: the rows around it may
    /// differ. The plans of the pattern stay, as they depend only on which
    /// names a row binds.
    pub(super) fn reset(&mut self) {
        self.answers.clear();
    }

    /// The test of `subquery` for rows of `left` and, in a join's pair, then `right`.
    pub(super) fn new(
        parts: RowPlanParts,
        subquery: LogicalOperator,
        left: &[String],
        right: &[String],
        outer: Option<Arc<RowBinding>>,
    ) -> Self {
        let mut row: HashSet<&str> = left.iter().chain(right).map(String::as_str).collect();
        if let Some(outer) = &outer {
            row.extend(outer.names());
        }
        let RowNames { names, holds_minus } = row_names_in(&subquery, &row);
        let names = names
            .into_iter()
            .map(|name| RowName {
                left: left.iter().position(|column| *column == name),
                right: right.iter().position(|column| *column == name),
                name,
            })
            .collect();
        // A mark and a hidden name (the graph a `GRAPH ?g` group reads) are no variables of the
        // row: binding one alone binds nothing, for a `MINUS`.
        let bindable = |columns: &[String]| {
            columns
                .iter()
                .enumerate()
                .filter(|(_, name)| {
                    !name.starts_with(MARK_PREFIX) && !crate::query::plan::is_hidden_variable(name)
                })
                .map(|(at, _)| at)
                .collect()
        };
        Self {
            parts,
            subquery,
            names,
            holds_minus,
            left_columns: bindable(left),
            right_columns: bindable(right),
            outer,
            rewrites: FxHashMap::default(),
            answers: FxHashMap::default(),
        }
    }

    /// Whether the row binds anything at all.
    fn binds_anything(
        &self,
        left_chunk: &DataChunk,
        left_row: usize,
        right_chunk: &DataChunk,
        right_row: usize,
    ) -> bool {
        let any = |chunk: &DataChunk, row: usize, columns: &[usize]| {
            columns.iter().any(|&column| {
                chunk
                    .column(column)
                    .and_then(|column| column.get_value(row))
                    .is_some_and(|value| !value.is_null())
            })
        };
        any(left_chunk, left_row, &self.left_columns)
            || any(right_chunk, right_row, &self.right_columns)
            || self
                .outer
                .as_ref()
                .is_some_and(|outer| outer.binds_anything)
    }

    /// Whether the pattern has a solution for the row: a left row and, in a join's pair, a right
    /// one, where a variable both have reads the left one unless it is unbound there.
    pub(super) fn holds(
        &mut self,
        left_chunk: &DataChunk,
        left_row: usize,
        right_chunk: &DataChunk,
        right_row: usize,
    ) -> Result<bool, OperatorError> {
        let key: Vec<Option<i64>> = self
            .names
            .iter()
            .map(|name| {
                name.left
                    .and_then(|column| id_at(left_chunk, column, left_row))
                    .or_else(|| {
                        name.right
                            .and_then(|column| id_at(right_chunk, column, right_row))
                    })
                    .or_else(|| {
                        let value = self.outer.as_ref()?.value(&name.name)?;
                        match value {
                            Value::Int64(id) => Some(*id),
                            _ => None,
                        }
                    })
            })
            .collect();
        let binds_anything =
            self.holds_minus && self.binds_anything(left_chunk, left_row, right_chunk, right_row);
        let asked = (key, binds_anything);
        if let Some(&answer) = self.answers.get(&asked) {
            return Ok(answer);
        }
        if self
            .parts
            .deadline
            .is_some_and(|deadline| grafeo_common::clock::Instant::now() >= deadline)
        {
            return Err(OperatorError::Timeout);
        }
        let answer = self.run(&asked.0, binds_anything)?;
        self.answers.insert(asked, answer);
        Ok(answer)
    }

    /// Plans the pattern for the row with `values`, out of line as a plan may be deep, and reads
    /// up to its first row.
    #[inline(never)]
    fn run(&mut self, values: &[Option<i64>], binds_anything: bool) -> Result<bool, OperatorError> {
        let shape: Vec<bool> = values.iter().map(Option::is_some).collect();
        let names = &self.names;
        let subquery = &self.subquery;
        let rewrite = self.rewrites.entry(shape).or_insert_with(|| {
            let bound: FxHashSet<&str> = names
                .iter()
                .zip(values)
                .filter(|(_, value)| value.is_some())
                .map(|(name, _)| name.name.as_str())
                .collect();
            inject(subquery.clone(), &bound)
        });
        let binding = RowBinding {
            values: names
                .iter()
                .zip(values)
                .filter_map(|(name, value)| Some((name.name.clone(), Value::Int64((*value)?))))
                .collect(),
            binds_anything,
        };
        let planner = RdfPlanner::for_row(&self.parts, Arc::new(binding));
        let (mut operator, _, _) = planner.plan_operator(rewrite)?;
        while let Some(chunk) = operator.next()? {
            if chunk.selected_indices().next().is_some() {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

/// `plan` with the row's values joined to whatever makes a name of `bound`. A leaf is given only the
/// names it has, so it keeps exactly its columns.
pub(super) fn inject(plan: LogicalOperator, bound: &FxHashSet<&str>) -> LogicalOperator {
    let mut plan = plan.map_children(|child| inject(child, bound));
    let mut names: Vec<String> = Vec::new();
    let mut add = |name: &str| {
        if bound.contains(name) && !names.iter().any(|known| known == name) {
            names.push(name.to_string());
        }
    };
    match &mut plan {
        LogicalOperator::TripleScan(scan) => {
            let graph = scan.graph.as_ref();
            for component in [&scan.subject, &scan.predicate, &scan.object]
                .into_iter()
                .chain(graph)
            {
                if let Some(name) = TripleComponent::as_variable(component) {
                    add(name);
                }
            }
        }
        LogicalOperator::PropertyPath(path) => {
            let graph = path.graph.as_ref();
            for component in [&path.subject, &path.object].into_iter().chain(graph) {
                if let Some(name) = TripleComponent::as_variable(component) {
                    add(name);
                }
            }
        }
        LogicalOperator::NamedGraphs(graphs) => add(&graphs.variable),
        LogicalOperator::InNamedGraph(graph) => {
            if let NamedGraph::Variable(variable) = &graph.graph {
                add(variable);
            }
        }
        LogicalOperator::Bind(bind) => add(&bind.variable),
        LogicalOperator::Project(project) => {
            for projection in &project.projections {
                if let Some(alias) = &projection.alias {
                    add(alias);
                }
            }
        }
        LogicalOperator::Aggregate(aggregate) => {
            for function in &aggregate.aggregates {
                if let Some(alias) = &function.alias {
                    add(alias);
                }
            }
            // A key or an argument reads the row's value where its input has none.
            let lacked = lacked_by_input(aggregate, bound);
            if !lacked.is_empty() {
                let input = std::mem::replace(&mut *aggregate.input, LogicalOperator::Empty);
                *aggregate.input = joined_with_row(lacked, input);
            }
        }
        _ => {}
    }
    if names.is_empty() {
        plan
    } else {
        joined_with_row(names, plan)
    }
}

/// The names of `bound` that a key or an argument of `aggregate` reads and its input may leave out.
fn lacked_by_input(
    aggregate: &crate::query::plan::AggregateOp,
    bound: &FxHashSet<&str>,
) -> Vec<String> {
    let mut read = HashSet::new();
    let arguments = aggregate
        .aggregates
        .iter()
        .flat_map(|function| [&function.expression, &function.expression2])
        .flatten();
    for expression in aggregate.group_by.iter().chain(arguments) {
        collect_expression_variables(expression, &mut read);
    }
    let has = always_bound(&aggregate.input).unwrap_or_default();
    let mut lacked: Vec<String> = read
        .into_iter()
        .filter(|name| bound.contains(name.as_str()) && !has.contains(name))
        .collect();
    lacked.sort_unstable();
    lacked
}

fn joined_with_row(names: Vec<String>, plan: LogicalOperator) -> LogicalOperator {
    LogicalOperator::Join(JoinOp {
        left: Box::new(LogicalOperator::RowValues(RowValuesOp { names })),
        right: Box::new(plan),
        join_type: JoinType::Inner,
        conditions: Vec::new(),
    })
}

/// Whether expressions over the rows of `input` read aggregates' results: the row is gone there,
/// and a name of it that is no key is unbound.
pub(super) fn over_an_aggregate(input: &LogicalOperator) -> bool {
    match input {
        LogicalOperator::Aggregate(_) => true,
        LogicalOperator::Filter(op) => over_an_aggregate(&op.input),
        LogicalOperator::Sort(op) => over_an_aggregate(&op.input),
        LogicalOperator::Bind(op) => over_an_aggregate(&op.input),
        LogicalOperator::Distinct(op) => over_an_aggregate(&op.input),
        LogicalOperator::Limit(op) => over_an_aggregate(&op.input),
        LogicalOperator::Skip(op) => over_an_aggregate(&op.input),
        _ => false,
    }
}

/// One row: the values of the names, read from the row being tested.
pub(super) struct RowValuesOperator {
    /// The values, until the row was given.
    values: Option<Vec<Value>>,
    /// The values, to give again after a reset.
    kept: Vec<Value>,
}

impl RowValuesOperator {
    pub(super) fn new(values: Vec<Value>) -> Self {
        Self {
            values: Some(values.clone()),
            kept: values,
        }
    }
}

impl Operator for RowValuesOperator {
    fn next(&mut self) -> OperatorResult {
        let Some(values) = self.values.take() else {
            return Ok(None);
        };
        let schema = vec![LogicalType::RdfTermId; values.len()];
        let mut chunk = DataChunk::with_capacity(&schema, 1);
        for (at, value) in values.into_iter().enumerate() {
            if let Some(column) = chunk.column_mut(at) {
                column.push_value(value);
            }
        }
        chunk.set_count(1);
        Ok(Some(chunk))
    }

    fn reset(&mut self) {
        self.values = Some(self.kept.clone());
    }

    fn name(&self) -> &'static str {
        "RowValues"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}
