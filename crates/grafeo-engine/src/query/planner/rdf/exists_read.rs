//! What an `EXISTS` reads of the row it tests, and so how it runs.
//!
//! By the SPARQL 1.2 draft's definition (17.4.1.4, 18.3.2.7) each group of the pattern starts from
//! the row. Running the pattern once and comparing is a shortcut of that, taken only where it gives
//! the same answer:
//!
//! - Every name of the row in the pattern is read where its input always binds it, or is what a
//!   `BIND` or alias makes: its rows, compared with the row on the variables both have, are the
//!   answer.
//! - As well as that, conjuncts of its top filter read a variable of the row the pattern does not
//!   bind: they are taken out, the rest runs once, and they are checked per row over the rows the
//!   comparison leaves.
//! - Anything else runs per row (see [`super::exists_per_row`]).

use std::borrow::Cow;
use std::collections::HashSet;

use crate::query::plan::subqueries::exists_plans;
use crate::query::plan::{
    FilterOp, LogicalExpression, LogicalOperator, NamedGraph, TripleComponent, always_bound,
    collect_expression_variables,
};

/// How to run an `EXISTS` once, over rows of some columns.
pub(super) struct ExistsRead<'a> {
    /// The sub-query to run once.
    pub(super) subquery: Cow<'a, LogicalOperator>,
    /// The conjuncts to check per row, over the merged row and sub-query's row.
    pub(super) row_conjuncts: Vec<LogicalExpression>,
    /// The variables it shares with the row.
    pub(super) shared: Vec<String>,
}

/// `expression` with each `EXISTS` in it, at any depth, replaced by what `test` makes of its plan.
pub(super) fn take_exists(
    expression: LogicalExpression,
    test: &mut dyn FnMut(LogicalOperator) -> LogicalExpression,
) -> LogicalExpression {
    match expression {
        LogicalExpression::ExistsSubquery(plan) => test(*plan),
        other => other.map_children(|child| take_exists(child, test)),
    }
}

/// What a plan names of a row.
#[derive(Default)]
pub(super) struct RowNames {
    /// The names, each once, in the order they were met.
    pub(super) names: Vec<String>,
    /// Whether the plan or an `EXISTS` in it holds a `MINUS`, which reads whether the row binds
    /// anything at all.
    pub(super) holds_minus: bool,
}

/// Every name of `row` anywhere in `plan`, what it binds as well as what it reads, in an `EXISTS` in
/// it too: the row reaches every group.
pub(super) fn row_names_in(plan: &LogicalOperator, row: &HashSet<&str>) -> RowNames {
    let mut found = RowNames::default();
    add_names(plan, row, &mut found);
    found
}

fn add_name(name: &str, row: &HashSet<&str>, found: &mut RowNames) {
    if row.contains(name) && !found.names.iter().any(|known| known == name) {
        found.names.push(name.to_string());
    }
}

fn add_component(component: &TripleComponent, row: &HashSet<&str>, found: &mut RowNames) {
    if let Some(name) = component.as_variable() {
        add_name(name, row, found);
    }
}

fn add_expression(expression: &LogicalExpression, row: &HashSet<&str>, found: &mut RowNames) {
    let mut variables = HashSet::new();
    collect_expression_variables(expression, &mut variables);
    for variable in &variables {
        add_name(variable, row, found);
    }
    for plan in exists_plans(expression) {
        add_names(&plan, row, found);
    }
}

fn add_names(plan: &LogicalOperator, row: &HashSet<&str>, found: &mut RowNames) {
    match plan {
        LogicalOperator::TripleScan(scan) => {
            add_component(&scan.subject, row, found);
            add_component(&scan.predicate, row, found);
            add_component(&scan.object, row, found);
            if let Some(graph) = &scan.graph {
                add_component(graph, row, found);
            }
        }
        LogicalOperator::PropertyPath(path) => {
            add_component(&path.subject, row, found);
            add_component(&path.object, row, found);
            if let Some(graph) = &path.graph {
                add_component(graph, row, found);
            }
        }
        LogicalOperator::NamedGraphs(graphs) => add_name(&graphs.variable, row, found),
        LogicalOperator::InNamedGraph(graph) => {
            if let NamedGraph::Variable(variable) = &graph.graph {
                add_name(variable, row, found);
            }
        }
        LogicalOperator::Filter(filter) => add_expression(&filter.predicate, row, found),
        LogicalOperator::Bind(bind) => {
            add_name(&bind.variable, row, found);
            add_expression(&bind.expression, row, found);
        }
        LogicalOperator::Project(project) => {
            for projection in &project.projections {
                if let Some(alias) = &projection.alias {
                    add_name(alias, row, found);
                }
                add_expression(&projection.expression, row, found);
            }
        }
        LogicalOperator::Sort(sort) => {
            for key in &sort.keys {
                add_expression(&key.expression, row, found);
            }
        }
        LogicalOperator::Aggregate(aggregate) => {
            for key in &aggregate.group_by {
                add_expression(key, row, found);
            }
            for function in &aggregate.aggregates {
                if let Some(alias) = &function.alias {
                    add_name(alias, row, found);
                }
                let arguments = [&function.expression, &function.expression2];
                for argument in arguments.into_iter().flatten() {
                    add_expression(argument, row, found);
                }
            }
            if let Some(having) = &aggregate.having {
                add_expression(having, row, found);
            }
        }
        LogicalOperator::LeftJoin(join) => {
            if let Some(condition) = &join.condition {
                add_expression(condition, row, found);
            }
        }
        LogicalOperator::AntiJoin(_) => found.holds_minus = true,
        _ => {}
    }
    for child in plan.children() {
        add_names(child, row, found);
    }
}

/// How to run `subquery` once for rows of the columns `row`, `None` when no run without the row
/// answers as the definition does.
pub(super) fn read_of_exists<'a>(
    subquery: &'a LogicalOperator,
    row: &[String],
) -> Option<ExistsRead<'a>> {
    let mut base = subquery;
    let mut conjuncts = Vec::new();
    while let LogicalOperator::Filter(filter) = base {
        super::conjuncts_of(&filter.predicate, &mut conjuncts);
        base = &filter.input;
    }
    let mut reads = RowReads {
        row: row.iter().map(String::as_str).collect(),
        shared: Vec::new(),
    };
    let bound = always_bound(base).unwrap_or_default();
    let (moved, kept): (Vec<&LogicalExpression>, Vec<&LogicalExpression>) = conjuncts
        .into_iter()
        .partition(|conjunct| reads.reads_unbound(conjunct, &bound));
    if moved
        .iter()
        .any(|conjunct| !exists_plans(conjunct).is_empty())
    {
        return None;
    }
    for conjunct in &kept {
        reads.expression(conjunct, base).ok()?;
    }
    reads.plan(base).ok()?;
    let shared = reads.shared;
    if moved.is_empty() {
        return Some(ExistsRead {
            subquery: Cow::Borrowed(subquery),
            row_conjuncts: Vec::new(),
            shared,
        });
    }
    let kept = LogicalExpression::conjunction(kept.into_iter().cloned());
    let run = match kept {
        Some(predicate) => LogicalOperator::Filter(FilterOp {
            predicate,
            input: Box::new(base.clone()),
            pushdown_hint: None,
        }),
        None => base.clone(),
    };
    Some(ExistsRead {
        subquery: Cow::Owned(run),
        row_conjuncts: moved.into_iter().cloned().collect(),
        shared,
    })
}

/// A run once does not answer as the definition does.
struct NotSafe;

/// A walk of a sub-query for whether a run without the row answers for it.
struct RowReads<'a> {
    /// The row's columns.
    row: HashSet<&'a str>,
    /// The variables of the row it names.
    shared: Vec<String>,
}

impl RowReads<'_> {
    /// Whether a conjunct of the top filter reads a variable of the row that `bound` leaves out.
    fn reads_unbound(&self, conjunct: &LogicalExpression, bound: &HashSet<String>) -> bool {
        let mut variables = HashSet::new();
        collect_expression_variables(conjunct, &mut variables);
        variables
            .iter()
            .any(|name| self.row.contains(name.as_str()) && !bound.contains(name))
    }

    /// Takes in a name of the sub-query that the comparison with the row covers.
    fn name(&mut self, variable: &str) {
        if self.row.contains(variable) && !self.shared.iter().any(|name| name == variable) {
            self.shared.push(variable.to_string());
        }
    }

    fn component(&mut self, component: &TripleComponent) {
        if let Some(variable) = component.as_variable() {
            self.name(variable);
        }
    }

    /// The names of the row in `plan`.
    fn names_in(&self, plan: &LogicalOperator) -> Vec<String> {
        row_names_in(plan, &self.row).names
    }

    /// An expression over the rows of `input`: each name of the row it reads, in an `EXISTS` in it
    /// too, must be always bound by `input`.
    fn expression(
        &mut self,
        expression: &LogicalExpression,
        input: &LogicalOperator,
    ) -> Result<(), NotSafe> {
        let bound = always_bound(input).unwrap_or_default();
        let mut variables = HashSet::new();
        collect_expression_variables(expression, &mut variables);
        for variable in &variables {
            if self.row.contains(variable.as_str()) {
                if !bound.contains(variable) {
                    return Err(NotSafe);
                }
                self.name(variable);
            }
        }
        for plan in exists_plans(expression) {
            if self
                .names_in(&plan)
                .iter()
                .any(|name| !bound.contains(name))
            {
                return Err(NotSafe);
            }
            self.plan(&plan)?;
        }
        Ok(())
    }

    /// Walks `plan`: any operator not named as safe here runs per row, as a new one must decide.
    fn plan(&mut self, plan: &LogicalOperator) -> Result<(), NotSafe> {
        match plan {
            LogicalOperator::TripleScan(scan) => {
                self.component(&scan.subject);
                self.component(&scan.predicate);
                self.component(&scan.object);
                if let Some(graph) = &scan.graph {
                    self.component(graph);
                }
            }
            LogicalOperator::PropertyPath(path) => {
                self.component(&path.subject);
                self.component(&path.object);
                if let Some(graph) = &path.graph {
                    self.component(graph);
                }
            }
            LogicalOperator::NamedGraphs(graphs) => self.name(&graphs.variable),
            LogicalOperator::InNamedGraph(graph) => {
                if let NamedGraph::Variable(variable) = &graph.graph {
                    self.name(variable);
                }
            }
            LogicalOperator::Filter(filter) => self.expression(&filter.predicate, &filter.input)?,
            LogicalOperator::Bind(bind) => {
                self.name(&bind.variable);
                self.expression(&bind.expression, &bind.input)?;
            }
            LogicalOperator::Sort(sort) => {
                for key in &sort.keys {
                    self.expression(&key.expression, &sort.input)?;
                }
            }
            LogicalOperator::Project(project) => {
                // A name below that it does not give is hidden from a comparison with the row,
                // which the definition's row still reaches.
                let given = |name: &String| {
                    project.projections.iter().any(|projection| {
                        match (&projection.alias, &projection.expression) {
                            (Some(alias), _) => alias == name,
                            (None, LogicalExpression::Variable(variable)) => variable == name,
                            _ => false,
                        }
                    })
                };
                if !self.names_in(&project.input).iter().all(given) {
                    return Err(NotSafe);
                }
                for projection in &project.projections {
                    if let Some(alias) = &projection.alias {
                        self.name(alias);
                    }
                    self.expression(&projection.expression, &project.input)?;
                }
            }
            LogicalOperator::Aggregate(aggregate) => {
                let is_key = |name: &String| {
                    aggregate.group_by.iter().any(
                        |key| matches!(key, LogicalExpression::Variable(variable) if variable == name),
                    )
                };
                if !self.names_in(&aggregate.input).iter().all(is_key) {
                    return Err(NotSafe);
                }
                for key in &aggregate.group_by {
                    self.expression(key, &aggregate.input)?;
                }
                for function in &aggregate.aggregates {
                    if let Some(alias) = &function.alias {
                        self.name(alias);
                    }
                    let arguments = [&function.expression, &function.expression2];
                    for argument in arguments.into_iter().flatten() {
                        self.expression(argument, &aggregate.input)?;
                    }
                }
                if let Some(having) = &aggregate.having {
                    self.expression(having, &aggregate.input)?;
                }
            }
            LogicalOperator::LeftJoin(join) => {
                let bound = always_bound(&join.left).unwrap_or_default();
                let mut names = self.names_in(&join.right);
                if let Some(condition) = &join.condition {
                    let mut in_condition = RowNames::default();
                    add_expression(condition, &self.row, &mut in_condition);
                    names.extend(in_condition.names);
                }
                if names.iter().any(|name| !bound.contains(name)) {
                    return Err(NotSafe);
                }
                if let Some(condition) = &join.condition {
                    for plan in exists_plans(condition) {
                        self.plan(&plan)?;
                    }
                }
            }
            LogicalOperator::AntiJoin(_) => {
                if !self.row.is_empty() {
                    return Err(NotSafe);
                }
            }
            LogicalOperator::Limit(limit) => {
                if !self.names_in(&limit.input).is_empty() {
                    return Err(NotSafe);
                }
            }
            LogicalOperator::Skip(skip) => {
                if !self.names_in(&skip.input).is_empty() {
                    return Err(NotSafe);
                }
            }
            LogicalOperator::Join(_)
            | LogicalOperator::MultiWayJoin(_)
            | LogicalOperator::Union(_)
            | LogicalOperator::Distinct(_)
            | LogicalOperator::Empty => {}
            LogicalOperator::RowValues(_)
            | LogicalOperator::NodeScan(_)
            | LogicalOperator::EdgeScan(_)
            | LogicalOperator::Expand(_)
            | LogicalOperator::CreateNode(_)
            | LogicalOperator::CreateEdge(_)
            | LogicalOperator::Create(_)
            | LogicalOperator::DeleteNode(_)
            | LogicalOperator::DeleteEdge(_)
            | LogicalOperator::SetProperty(_)
            | LogicalOperator::AddLabel(_)
            | LogicalOperator::RemoveLabel(_)
            | LogicalOperator::Return(_)
            | LogicalOperator::MarkJoin(_)
            | LogicalOperator::Construct(_)
            | LogicalOperator::Unwind(_)
            | LogicalOperator::MapCollect(_)
            | LogicalOperator::Merge(_)
            | LogicalOperator::MergeRelationship(_)
            | LogicalOperator::ShortestPath(_)
            | LogicalOperator::InsertTriple(_)
            | LogicalOperator::DeleteTriple(_)
            | LogicalOperator::Modify(_)
            | LogicalOperator::ClearGraph(_)
            | LogicalOperator::CreateGraph(_)
            | LogicalOperator::DropGraph(_)
            | LogicalOperator::LoadGraph(_)
            | LogicalOperator::CopyGraph(_)
            | LogicalOperator::MoveGraph(_)
            | LogicalOperator::AddGraph(_)
            | LogicalOperator::HorizontalAggregate(_)
            | LogicalOperator::VectorScan(_)
            | LogicalOperator::VectorJoin(_)
            | LogicalOperator::TextScan(_)
            | LogicalOperator::Except(_)
            | LogicalOperator::Intersect(_)
            | LogicalOperator::Otherwise(_)
            | LogicalOperator::Apply(_)
            | LogicalOperator::ParameterScan(_)
            | LogicalOperator::DelimScan(_)
            | LogicalOperator::CreatePropertyGraph(_)
            | LogicalOperator::DropPropertyGraph(_)
            | LogicalOperator::CallProcedure(_)
            | LogicalOperator::LoadData(_) => return Err(NotSafe),
        }
        for child in plan.children() {
            self.plan(child)?;
        }
        Ok(())
    }
}
