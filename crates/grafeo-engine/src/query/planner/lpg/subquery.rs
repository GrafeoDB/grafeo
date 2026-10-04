//! `EXISTS` and `COUNT` subqueries planned per row of their input.
//!
//! The edge check (`FilterExpression::ExistsSubquery` and `CountSubquery`)
//! answers a subquery that one edge from a node of the row decides. Every
//! other one runs as a correlated `Apply`: the subquery's plan starts from a
//! row of the outer variables it uses, so its patterns continue from the
//! outer nodes and edges and its expressions read the outer values, and an
//! aggregate counts its rows (for `EXISTS`, up to one). The count becomes a
//! column of the row that the expression reads in place of the subquery.

use std::collections::HashSet;

use super::{Arc, Error, LogicalExpression, LogicalOperator, Operator, Result, Value};
use crate::query::plan::{
    AggregateExpr, AggregateFunction, AggregateOp, BinaryOp, LimitOp, MapProjectionEntry,
    ParameterScanOp, Projection, ReturnItem,
};
use crate::query::planner::common::output_column_name;
use grafeo_common::types::LogicalType;
use grafeo_core::execution::operators::{
    ApplyOperator, JoinType, NestedLoopJoinOperator, ParameterState,
};

impl super::Planner {
    /// Whether `expression` has an `EXISTS` or `COUNT` subquery that the edge
    /// check cannot answer (outside a list comprehension, list predicate or
    /// `reduce`, whose subqueries read the item variable). With the row's
    /// `columns`, also one that shares no variable with the row (see
    /// [`Self::edge_check_answers`]).
    pub(super) fn has_subquery_to_lift(
        &self,
        expression: &LogicalExpression,
        columns: Option<&[String]>,
    ) -> bool {
        let mut expression = expression.clone();
        let mut found = false;
        let _ = visit_subqueries(&mut expression, &mut |subquery| {
            found |= !self.edge_check_answers(subquery, columns);
            Ok(())
        });
        found
    }

    /// Plans every `EXISTS` and `COUNT` subquery in `expression` that the edge
    /// check cannot answer as a correlated `Apply` over `input`, which adds a
    /// column with its count. Returns the expression with each such subquery
    /// replaced by a read of that column (`> 0` for `EXISTS`), and the input
    /// and columns with the new ones. `input_writes`: whether the input's
    /// plan writes (see [`Self::plan_counted_subquery`]).
    pub(super) fn lift_subqueries(
        &self,
        expression: &LogicalExpression,
        mut input: Box<dyn Operator>,
        mut columns: Vec<String>,
        input_writes: bool,
    ) -> Result<(LogicalExpression, Box<dyn Operator>, Vec<String>)> {
        let mut expression = expression.clone();
        visit_subqueries(&mut expression, &mut |subquery| {
            if self.edge_check_answers(subquery, Some(&columns)) {
                return Ok(());
            }
            let (subplan, exists) = match subquery {
                LogicalExpression::ExistsSubquery(subplan) => (subplan.as_ref().clone(), true),
                LogicalExpression::CountSubquery(subplan) => (subplan.as_ref().clone(), false),
                _ => return Ok(()),
            };
            let column = self.next_subquery_column(&columns);
            let outer = std::mem::replace(
                &mut input,
                Box::new(grafeo_core::execution::operators::EmptyOperator::new(
                    vec![],
                )),
            );
            input = self.plan_counted_subquery(
                outer,
                &columns,
                subplan,
                exists,
                &column,
                input_writes,
            )?;
            columns.push(column.clone());
            let count = LogicalExpression::Variable(column);
            *subquery = if exists {
                LogicalExpression::Binary {
                    left: Box::new(count),
                    op: BinaryOp::Gt,
                    right: Box::new(LogicalExpression::Literal(Value::Int64(0))),
                }
            } else {
                count
            };
            Ok(())
        })?;
        Ok((expression, input, columns))
    }

    /// Lifts the subqueries of `RETURN` items (see [`Self::lift_subqueries`]),
    /// each item keeping its column name. `None` when no item has one.
    pub(super) fn lift_return_items(
        &self,
        items: &[ReturnItem],
        mut input: Box<dyn Operator>,
        mut columns: Vec<String>,
        input_writes: bool,
    ) -> Result<(Option<Vec<ReturnItem>>, Box<dyn Operator>, Vec<String>)> {
        if !items
            .iter()
            .any(|item| self.has_subquery_to_lift(&item.expression, Some(&columns)))
        {
            return Ok((None, input, columns));
        }
        let mut lifted = Vec::with_capacity(items.len());
        for item in items {
            let alias = Some(output_column_name(item.alias.as_deref(), &item.expression));
            let (expression, lifted_input, lifted_columns) =
                self.lift_subqueries(&item.expression, input, columns, input_writes)?;
            input = lifted_input;
            columns = lifted_columns;
            lifted.push(ReturnItem { expression, alias });
        }
        Ok((Some(lifted), input, columns))
    }

    /// Lifts the subqueries of `WITH` and `LET` projections (see
    /// [`Self::lift_subqueries`]), each keeping its column name. `None` when
    /// no projection has one.
    pub(super) fn lift_projections(
        &self,
        projections: &[Projection],
        mut input: Box<dyn Operator>,
        mut columns: Vec<String>,
        input_writes: bool,
    ) -> Result<(Option<Vec<Projection>>, Box<dyn Operator>, Vec<String>)> {
        if !projections
            .iter()
            .any(|projection| self.has_subquery_to_lift(&projection.expression, Some(&columns)))
        {
            return Ok((None, input, columns));
        }
        let mut lifted = Vec::with_capacity(projections.len());
        for projection in projections {
            let alias = Some(output_column_name(
                projection.alias.as_deref(),
                &projection.expression,
            ));
            let (expression, lifted_input, lifted_columns) =
                self.lift_subqueries(&projection.expression, input, columns, input_writes)?;
            input = lifted_input;
            columns = lifted_columns;
            lifted.push(Projection { expression, alias });
        }
        Ok((Some(lifted), input, columns))
    }

    /// Whether the edge check answers this `EXISTS` or `COUNT` subquery: one
    /// edge (or, for `EXISTS`, one path of one edge type) from a node, which
    /// `extract_exists_pattern` recognizes. `COUNT` counts single edges only.
    /// With the row's `columns`, the pattern must also share a node or edge
    /// with the row: one that shares none has one answer for all rows, which
    /// the edge check would find again for each row by walking every edge.
    fn edge_check_answers(&self, subquery: &LogicalExpression, columns: Option<&[String]>) -> bool {
        let check = match subquery {
            LogicalExpression::ExistsSubquery(subplan) => self.extract_exists_pattern(subplan),
            LogicalExpression::CountSubquery(subplan) => {
                self.extract_exists_pattern(subplan).and_then(|check| {
                    if check.one_edge {
                        Ok(check)
                    } else {
                        Err(Error::Internal("COUNT over more than one edge".to_string()))
                    }
                })
            }
            _ => return true,
        };
        check.is_ok_and(|check| {
            columns.is_none_or(|columns| {
                columns.iter().any(|column| {
                    *column == check.start_var
                        || *column == check.end_var
                        || check.edge_var.as_ref() == Some(column)
                })
            })
        })
    }

    /// A name for the column of a lifted subquery, unique in the query and
    /// not one of the row's `columns` (a variable may be named so).
    fn next_subquery_column(&self, columns: &[String]) -> String {
        loop {
            let n = self.subquery_counter.get();
            self.subquery_counter.set(n + 1);
            let name = format!("__subquery_{n}");
            if !columns.contains(&name) {
                return name;
            }
        }
    }

    /// Plans `subplan` once per row of `outer`, counting its rows (up to one
    /// when `exists`), as a correlated `Apply` that adds the count as `column`.
    /// A subquery that shares nothing with the row is counted once and joined
    /// to every row; below a write (`input_writes`) after all the rows are read,
    /// so the count sees what they wrote.
    fn plan_counted_subquery(
        &self,
        outer: Box<dyn Operator>,
        outer_columns: &[String],
        subplan: LogicalOperator,
        exists: bool,
        column: &str,
        input_writes: bool,
    ) -> Result<Box<dyn Operator>> {
        // The outer variables the subquery uses: those it names, as far as
        // they are columns of the row; all of them when the subquery has an
        // operator whose names are not known here.
        let shared: Vec<String> = match subplan_variables(&subplan) {
            Some(names) => outer_columns
                .iter()
                .filter(|column| names.contains(*column))
                .cloned()
                .collect(),
            None => outer_columns
                .iter()
                .filter(|column| !column.starts_with("__"))
                .cloned()
                .collect(),
        };
        let mut seeded = subplan;
        if !shared.is_empty()
            && !seed_with_parameters(
                &mut seeded,
                LogicalOperator::ParameterScan(ParameterScanOp {
                    columns: shared.clone(),
                }),
            )
        {
            return Err(Error::Internal(
                "Unsupported subquery: no pattern to start from the outer row".to_string(),
            ));
        }
        // A pattern through a node or edge of the outer row matches that node
        // or edge, as a later MATCH does: with the parameters at its start,
        // the variables they bring are bound, and the cycle pass turns a
        // pattern variable bound again into a check that it is the same.
        let seeded = crate::query::optimizer::close_cycles(seeded);
        let counted_input = if exists {
            LogicalOperator::Limit(LimitOp {
                count: 1.into(),
                input: Box::new(seeded),
            })
        } else {
            seeded
        };
        let counted = LogicalOperator::Aggregate(AggregateOp {
            group_by: Vec::new(),
            aggregates: vec![AggregateExpr {
                function: AggregateFunction::Count,
                expression: None,
                expression2: None,
                distinct: false,
                alias: Some(column.to_string()),
                percentile: None,
                separator: None,
            }],
            input: Box::new(counted_input),
            having: None,
        });

        if shared.is_empty() {
            let (inner, _) = self.plan_operator(&counted)?;
            self.scalar_columns.borrow_mut().insert(column.to_string());
            let mut schema = self.derive_schema_from_columns(outer_columns);
            schema.push(LogicalType::Int64);
            let mut join = NestedLoopJoinOperator::new(outer, inner, None, JoinType::Cross, schema);
            if input_writes {
                join = join.with_left_first();
            }
            return Ok(Box::new(join));
        }

        let state = Arc::new(ParameterState::new(shared.clone()));
        let indices: Vec<usize> = shared
            .iter()
            .filter_map(|name| outer_columns.iter().position(|column| column == name))
            .collect();
        // A subquery inside this one sets its own state while it is planned;
        // the one before this subquery comes back afterwards.
        let previous = self
            .correlated_param_state
            .replace(Some(Arc::clone(&state)));
        let planned = self.plan_operator(&counted);
        *self.correlated_param_state.borrow_mut() = previous;
        let (inner, _) = planned?;
        self.scalar_columns.borrow_mut().insert(column.to_string());
        Ok(Box::new(ApplyOperator::new_correlated(
            outer, inner, state, indices,
        )))
    }
}

/// Calls `f` on every `EXISTS` and `COUNT` subquery in `expression`, outside
/// the bodies of list comprehensions, list predicates and `reduce`, which
/// read their item variable (not a column of the row).
fn visit_subqueries(
    expression: &mut LogicalExpression,
    f: &mut dyn FnMut(&mut LogicalExpression) -> Result<()>,
) -> Result<()> {
    match expression {
        LogicalExpression::ExistsSubquery(_) | LogicalExpression::CountSubquery(_) => f(expression),
        LogicalExpression::Binary { left, right, .. } => {
            visit_subqueries(left, f)?;
            visit_subqueries(right, f)
        }
        LogicalExpression::Unary { operand, .. } => visit_subqueries(operand, f),
        LogicalExpression::FunctionCall { args, .. } | LogicalExpression::List(args) => {
            args.iter_mut().try_for_each(|arg| visit_subqueries(arg, f))
        }
        LogicalExpression::Map(entries) => entries
            .iter_mut()
            .try_for_each(|(_, value)| visit_subqueries(value, f)),
        LogicalExpression::IndexAccess { base, index } => {
            visit_subqueries(base, f)?;
            visit_subqueries(index, f)
        }
        LogicalExpression::MapAccess { base, .. } => visit_subqueries(base, f),
        LogicalExpression::SliceAccess { base, start, end } => {
            visit_subqueries(base, f)?;
            if let Some(start) = start {
                visit_subqueries(start, f)?;
            }
            if let Some(end) = end {
                visit_subqueries(end, f)?;
            }
            Ok(())
        }
        LogicalExpression::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            if let Some(operand) = operand {
                visit_subqueries(operand, f)?;
            }
            for (condition, result) in when_clauses {
                visit_subqueries(condition, f)?;
                visit_subqueries(result, f)?;
            }
            if let Some(else_clause) = else_clause {
                visit_subqueries(else_clause, f)?;
            }
            Ok(())
        }
        LogicalExpression::ListComprehension { list_expr, .. }
        | LogicalExpression::ListPredicate { list_expr, .. } => visit_subqueries(list_expr, f),
        LogicalExpression::Reduce { initial, list, .. } => {
            visit_subqueries(initial, f)?;
            visit_subqueries(list, f)
        }
        LogicalExpression::MapProjection { entries, .. } => {
            entries.iter_mut().try_for_each(|entry| match entry {
                MapProjectionEntry::LiteralEntry(_, value) => visit_subqueries(value, f),
                MapProjectionEntry::PropertySelector(_) | MapProjectionEntry::AllProperties => {
                    Ok(())
                }
            })
        }
        LogicalExpression::Literal(_)
        | LogicalExpression::Variable(_)
        | LogicalExpression::Property { .. }
        | LogicalExpression::Parameter(_)
        | LogicalExpression::Labels(_)
        | LogicalExpression::Type(_)
        | LogicalExpression::Id(_)
        | LogicalExpression::ValueSubquery(_)
        | LogicalExpression::PatternComprehension { .. } => Ok(()),
    }
}

/// Puts `parameters` under the first pattern of `plan`, as the input of its
/// leftmost node or edge scan, so the scan continues from the outer row (a
/// scan of an outer variable reuses its value). A parameter scan the
/// translator joined with the pattern (a join without a condition, which would
/// pair every outer row with every match) gives way to the pattern, seeded the
/// same way; one on its own is replaced. Returns false when the plan has no
/// scan to start from.
fn seed_with_parameters(plan: &mut LogicalOperator, parameters: LogicalOperator) -> bool {
    if let LogicalOperator::Join(join) = plan
        && (matches!(*join.left, LogicalOperator::ParameterScan(_))
            || matches!(*join.right, LogicalOperator::ParameterScan(_)))
    {
        let pattern = if matches!(*join.left, LogicalOperator::ParameterScan(_)) {
            std::mem::replace(&mut *join.right, LogicalOperator::Empty)
        } else {
            std::mem::replace(&mut *join.left, LogicalOperator::Empty)
        };
        *plan = pattern;
        return seed_with_parameters(plan, parameters);
    }
    match plan {
        // The one empty row a subquery that starts with OPTIONAL MATCH
        // starts from becomes the outer row.
        LogicalOperator::ParameterScan(_) | LogicalOperator::Empty => {
            *plan = parameters;
            true
        }
        LogicalOperator::NodeScan(scan) => match &mut scan.input {
            Some(input) => seed_with_parameters(input, parameters),
            None => {
                scan.input = Some(Box::new(parameters));
                true
            }
        },
        LogicalOperator::EdgeScan(scan) => match &mut scan.input {
            Some(input) => seed_with_parameters(input, parameters),
            None => {
                scan.input = Some(Box::new(parameters));
                true
            }
        },
        LogicalOperator::Expand(expand) => seed_with_parameters(&mut expand.input, parameters),
        LogicalOperator::Filter(filter) => seed_with_parameters(&mut filter.input, parameters),
        LogicalOperator::Project(project) => seed_with_parameters(&mut project.input, parameters),
        LogicalOperator::Return(ret) => seed_with_parameters(&mut ret.input, parameters),
        LogicalOperator::Aggregate(aggregate) => {
            seed_with_parameters(&mut aggregate.input, parameters)
        }
        LogicalOperator::Limit(limit) => seed_with_parameters(&mut limit.input, parameters),
        LogicalOperator::Skip(skip) => seed_with_parameters(&mut skip.input, parameters),
        LogicalOperator::Sort(sort) => seed_with_parameters(&mut sort.input, parameters),
        LogicalOperator::Distinct(distinct) => {
            seed_with_parameters(&mut distinct.input, parameters)
        }
        LogicalOperator::Bind(bind) => seed_with_parameters(&mut bind.input, parameters),
        LogicalOperator::Join(join) => seed_with_parameters(&mut join.left, parameters),
        LogicalOperator::LeftJoin(join) => seed_with_parameters(&mut join.left, parameters),
        LogicalOperator::AntiJoin(join) => seed_with_parameters(&mut join.left, parameters),
        LogicalOperator::Unwind(unwind) => {
            if matches!(*unwind.input, LogicalOperator::Empty) {
                *unwind.input = parameters;
                true
            } else {
                seed_with_parameters(&mut unwind.input, parameters)
            }
        }
        _ => false,
    }
}

/// Whether a subquery reads a value of the outer row other than through a
/// node or edge its patterns share with it: a variable its expressions use
/// that its patterns do not bind (`{id: s.id}`, `WHERE x.id = s.id`, an
/// `UNWIND` variable, a parameter scan of outer variables). Such a subquery
/// runs per row; one tied to the row by shared pattern variables alone can be
/// a semi-join on them. One that starts with OPTIONAL MATCH is not tied by
/// its patterns (its row of nulls binds none of them), so it runs per row too.
pub(super) fn reads_outer_values(subplan: &LogicalOperator) -> bool {
    if starts_with_optional_match(subplan) {
        return true;
    }
    let Some(used) = subplan_variables(subplan) else {
        return true;
    };
    let mut bound = HashSet::new();
    bound_names(subplan, &mut bound);
    used.iter().any(|name| !bound.contains(name))
}

/// Whether `plan` starts with an OPTIONAL MATCH: a left join of one empty row.
fn starts_with_optional_match(plan: &LogicalOperator) -> bool {
    match plan {
        LogicalOperator::LeftJoin(join) => {
            matches!(join.left.as_ref(), LogicalOperator::Empty)
                || starts_with_optional_match(&join.left)
        }
        LogicalOperator::Join(join) => starts_with_optional_match(&join.left),
        LogicalOperator::Filter(op) => starts_with_optional_match(&op.input),
        LogicalOperator::Project(op) => starts_with_optional_match(&op.input),
        LogicalOperator::Return(op) => starts_with_optional_match(&op.input),
        LogicalOperator::Aggregate(op) => starts_with_optional_match(&op.input),
        LogicalOperator::Limit(op) => starts_with_optional_match(&op.input),
        LogicalOperator::Skip(op) => starts_with_optional_match(&op.input),
        LogicalOperator::Sort(op) => starts_with_optional_match(&op.input),
        LogicalOperator::Distinct(op) => starts_with_optional_match(&op.input),
        LogicalOperator::Expand(op) => starts_with_optional_match(&op.input),
        LogicalOperator::NodeScan(op) => {
            op.input.as_deref().is_some_and(starts_with_optional_match)
        }
        _ => false,
    }
}

/// The names `plan` binds itself: its pattern variables, projection and
/// aggregate aliases, and the variables of `UNWIND` and `LET`.
fn bound_names(plan: &LogicalOperator, names: &mut HashSet<String>) {
    match plan {
        LogicalOperator::NodeScan(scan) => {
            names.insert(scan.variable.clone());
        }
        LogicalOperator::EdgeScan(scan) => {
            names.insert(scan.variable.clone());
        }
        LogicalOperator::Expand(expand) => {
            names.insert(expand.from_variable.clone());
            names.insert(expand.to_variable.clone());
            names.extend(expand.edge_variable.iter().cloned());
            names.extend(expand.path_alias.iter().cloned());
        }
        LogicalOperator::Project(project) => {
            names.extend(project.projections.iter().filter_map(|p| p.alias.clone()));
        }
        LogicalOperator::Return(ret) => {
            names.extend(ret.items.iter().filter_map(|item| item.alias.clone()));
        }
        // A group key only reads a name: one the subquery binds is bound by
        // its pattern, and one from the outer row stays an outer value.
        LogicalOperator::Aggregate(aggregate) => {
            names.extend(aggregate.aggregates.iter().filter_map(|a| a.alias.clone()));
        }
        LogicalOperator::Unwind(unwind) => {
            names.insert(unwind.variable.clone());
            names.extend(unwind.ordinality_var.iter().cloned());
            names.extend(unwind.offset_var.iter().cloned());
        }
        LogicalOperator::Bind(bind) => {
            names.insert(bind.variable.clone());
        }
        _ => {}
    }
    for child in plan.children() {
        bound_names(child, names);
    }
}

/// Every variable name `plan` binds or reads, also in the subqueries of its
/// expressions; `None` when it has an operator this does not look into.
fn subplan_variables(plan: &LogicalOperator) -> Option<HashSet<String>> {
    let mut names = HashSet::new();
    plan_names(plan, &mut names).then_some(names)
}

fn plan_names(plan: &LogicalOperator, names: &mut HashSet<String>) -> bool {
    match plan {
        LogicalOperator::NodeScan(scan) => {
            names.insert(scan.variable.clone());
            scan.input
                .as_deref()
                .is_none_or(|input| plan_names(input, names))
        }
        LogicalOperator::EdgeScan(scan) => {
            names.insert(scan.variable.clone());
            scan.input
                .as_deref()
                .is_none_or(|input| plan_names(input, names))
        }
        LogicalOperator::Expand(expand) => {
            names.insert(expand.from_variable.clone());
            names.insert(expand.to_variable.clone());
            names.extend(expand.edge_variable.iter().cloned());
            names.extend(expand.path_alias.iter().cloned());
            plan_names(&expand.input, names)
        }
        LogicalOperator::Filter(filter) => {
            expression_names(&filter.predicate, names) && plan_names(&filter.input, names)
        }
        LogicalOperator::Project(project) => {
            project
                .projections
                .iter()
                .all(|projection| expression_names(&projection.expression, names))
                && plan_names(&project.input, names)
        }
        LogicalOperator::Return(ret) => {
            ret.items
                .iter()
                .all(|item| expression_names(&item.expression, names))
                && plan_names(&ret.input, names)
        }
        LogicalOperator::Aggregate(aggregate) => {
            aggregate
                .group_by
                .iter()
                .all(|key| expression_names(key, names))
                && aggregate.aggregates.iter().all(|aggregate| {
                    aggregate
                        .expression
                        .iter()
                        .chain(&aggregate.expression2)
                        .all(|expression| expression_names(expression, names))
                })
                && aggregate
                    .having
                    .as_ref()
                    .is_none_or(|having| expression_names(having, names))
                && plan_names(&aggregate.input, names)
        }
        LogicalOperator::Sort(sort) => {
            sort.keys
                .iter()
                .all(|key| expression_names(&key.expression, names))
                && plan_names(&sort.input, names)
        }
        LogicalOperator::Limit(limit) => plan_names(&limit.input, names),
        LogicalOperator::Skip(skip) => plan_names(&skip.input, names),
        LogicalOperator::Distinct(distinct) => plan_names(&distinct.input, names),
        LogicalOperator::Unwind(unwind) => {
            names.insert(unwind.variable.clone());
            expression_names(&unwind.expression, names) && plan_names(&unwind.input, names)
        }
        LogicalOperator::Bind(bind) => {
            names.insert(bind.variable.clone());
            expression_names(&bind.expression, names) && plan_names(&bind.input, names)
        }
        LogicalOperator::Join(join) => {
            join.conditions.iter().all(|condition| {
                expression_names(&condition.left, names)
                    && expression_names(&condition.right, names)
            }) && plan_names(&join.left, names)
                && plan_names(&join.right, names)
        }
        LogicalOperator::LeftJoin(join) => {
            join.condition
                .as_ref()
                .is_none_or(|condition| expression_names(condition, names))
                && plan_names(&join.left, names)
                && plan_names(&join.right, names)
        }
        LogicalOperator::AntiJoin(join) => {
            plan_names(&join.left, names) && plan_names(&join.right, names)
        }
        LogicalOperator::Apply(apply) => {
            names.extend(apply.shared_variables.iter().cloned());
            plan_names(&apply.input, names) && plan_names(&apply.subplan, names)
        }
        LogicalOperator::ParameterScan(scan) => {
            names.extend(scan.columns.iter().cloned());
            true
        }
        LogicalOperator::Union(union) => union.inputs.iter().all(|input| plan_names(input, names)),
        LogicalOperator::Empty => true,
        _ => false,
    }
}

fn expression_names(expression: &LogicalExpression, names: &mut HashSet<String>) -> bool {
    match expression {
        LogicalExpression::Variable(name)
        | LogicalExpression::Labels(name)
        | LogicalExpression::Type(name)
        | LogicalExpression::Id(name) => {
            names.insert(name.clone());
            true
        }
        LogicalExpression::Property { variable, .. } => {
            names.insert(variable.clone());
            true
        }
        LogicalExpression::Literal(_) | LogicalExpression::Parameter(_) => true,
        LogicalExpression::Binary { left, right, .. } => {
            expression_names(left, names) && expression_names(right, names)
        }
        LogicalExpression::Unary { operand, .. } => expression_names(operand, names),
        LogicalExpression::FunctionCall { args, .. } | LogicalExpression::List(args) => {
            args.iter().all(|arg| expression_names(arg, names))
        }
        LogicalExpression::Map(entries) => entries
            .iter()
            .all(|(_, value)| expression_names(value, names)),
        LogicalExpression::IndexAccess { base, index } => {
            expression_names(base, names) && expression_names(index, names)
        }
        LogicalExpression::MapAccess { base, .. } => expression_names(base, names),
        LogicalExpression::SliceAccess { base, start, end } => {
            expression_names(base, names)
                && start.as_deref().is_none_or(|e| expression_names(e, names))
                && end.as_deref().is_none_or(|e| expression_names(e, names))
        }
        LogicalExpression::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            operand
                .as_deref()
                .is_none_or(|e| expression_names(e, names))
                && when_clauses.iter().all(|(condition, result)| {
                    expression_names(condition, names) && expression_names(result, names)
                })
                && else_clause
                    .as_deref()
                    .is_none_or(|e| expression_names(e, names))
        }
        LogicalExpression::ListComprehension {
            list_expr,
            filter_expr,
            map_expr,
            ..
        } => {
            expression_names(list_expr, names)
                && filter_expr
                    .as_deref()
                    .is_none_or(|e| expression_names(e, names))
                && expression_names(map_expr, names)
        }
        LogicalExpression::ListPredicate {
            list_expr,
            predicate,
            ..
        } => expression_names(list_expr, names) && expression_names(predicate, names),
        LogicalExpression::Reduce {
            initial,
            list,
            expression,
            ..
        } => {
            expression_names(initial, names)
                && expression_names(list, names)
                && expression_names(expression, names)
        }
        LogicalExpression::MapProjection { base, entries } => {
            names.insert(base.clone());
            entries.iter().all(|entry| match entry {
                MapProjectionEntry::LiteralEntry(_, value) => expression_names(value, names),
                MapProjectionEntry::PropertySelector(_) | MapProjectionEntry::AllProperties => true,
            })
        }
        LogicalExpression::ExistsSubquery(subplan)
        | LogicalExpression::CountSubquery(subplan)
        | LogicalExpression::ValueSubquery(subplan) => plan_names(subplan, names),
        LogicalExpression::PatternComprehension {
            subplan,
            projection,
        } => plan_names(subplan, names) && expression_names(projection, names),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::plan::{AggregateExpr, AggregateFunction, AggregateOp, NodeScanOp};

    /// A subquery grouping by a name its patterns do not bind (here `k`, an
    /// outer `UNWIND` variable) reads that outer value, so it runs per row;
    /// grouping by its own pattern variable does not.
    #[test]
    fn a_group_key_from_the_outer_row_is_an_outer_value() {
        let grouped_by = |key: &str| {
            LogicalOperator::Aggregate(AggregateOp {
                group_by: vec![LogicalExpression::Variable(key.into())],
                aggregates: vec![AggregateExpr {
                    function: AggregateFunction::Count,
                    expression: Some(LogicalExpression::Variable("p".into())),
                    expression2: None,
                    distinct: false,
                    alias: Some("c".into()),
                    percentile: None,
                    separator: None,
                }],
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "p".into(),
                    label: None,
                    input: None,
                })),
                having: None,
            })
        };
        assert!(reads_outer_values(&grouped_by("k")));
        assert!(!reads_outer_values(&grouped_by("p")));
    }
}
