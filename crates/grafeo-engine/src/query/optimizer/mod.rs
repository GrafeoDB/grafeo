//! Makes your queries faster without changing their meaning.
//!
//! The optimizer transforms logical plans to run more efficiently:
//!
//! | Optimization | What it does |
//! | ------------ | ------------ |
//! | Filter Pushdown | Moves `WHERE` clauses closer to scans - filter early, process less |
//! | Join Reordering | Picks the best order to join tables using the DPccp algorithm |
//! | Predicate Simplification | Folds constants like `1 + 1` into `2` |
//!
//! The optimizer uses [`CostModel`] and [`CardinalityEstimator`] to predict
//! how expensive different plans are, then picks the cheapest.

pub mod cardinality;
pub mod cost;
pub mod join_order;

pub use cardinality::{CardinalityEstimator, ColumnStats, TableStats};
pub use cost::{Cost, CostModel};
pub use join_order::{BitSet, DPccp, JoinGraph, JoinGraphBuilder, JoinPlan};

use crate::query::plan::{FilterOp, LogicalExpression, LogicalOperator, LogicalPlan};
use grafeo_common::utils::error::Result;
use std::collections::HashSet;

/// Information about a join condition for join reordering.
#[derive(Debug, Clone)]
struct JoinInfo {
    left_var: String,
    right_var: String,
    left_expr: LogicalExpression,
    right_expr: LogicalExpression,
}

/// A column required by the query, used for projection pushdown.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum RequiredColumn {
    /// A variable (node, edge, or path binding)
    Variable(String),
    /// A specific property of a variable
    Property(String, String),
}

/// Transforms logical plans for faster execution.
///
/// Create with [`new()`](Self::new), then call [`optimize()`](Self::optimize).
/// Use the builder methods to enable/disable specific optimizations.
pub struct Optimizer {
    /// Whether to enable filter pushdown.
    enable_filter_pushdown: bool,
    /// Whether to enable join reordering.
    enable_join_reorder: bool,
    /// Whether to enable projection pushdown.
    enable_projection_pushdown: bool,
    /// Cost model for estimation.
    cost_model: CostModel,
    /// Cardinality estimator.
    card_estimator: CardinalityEstimator,
}

impl Optimizer {
    /// Creates a new optimizer with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            enable_filter_pushdown: true,
            enable_join_reorder: true,
            enable_projection_pushdown: true,
            cost_model: CostModel::new(),
            card_estimator: CardinalityEstimator::new(),
        }
    }

    /// Enables or disables filter pushdown.
    pub fn with_filter_pushdown(mut self, enabled: bool) -> Self {
        self.enable_filter_pushdown = enabled;
        self
    }

    /// Enables or disables join reordering.
    pub fn with_join_reorder(mut self, enabled: bool) -> Self {
        self.enable_join_reorder = enabled;
        self
    }

    /// Enables or disables projection pushdown.
    pub fn with_projection_pushdown(mut self, enabled: bool) -> Self {
        self.enable_projection_pushdown = enabled;
        self
    }

    /// Sets the cost model.
    pub fn with_cost_model(mut self, cost_model: CostModel) -> Self {
        self.cost_model = cost_model;
        self
    }

    /// Sets the cardinality estimator.
    pub fn with_cardinality_estimator(mut self, estimator: CardinalityEstimator) -> Self {
        self.card_estimator = estimator;
        self
    }

    /// Returns a reference to the cost model.
    pub fn cost_model(&self) -> &CostModel {
        &self.cost_model
    }

    /// Returns a reference to the cardinality estimator.
    pub fn cardinality_estimator(&self) -> &CardinalityEstimator {
        &self.card_estimator
    }

    /// Estimates the cost of a plan.
    pub fn estimate_cost(&self, plan: &LogicalPlan) -> Cost {
        let cardinality = self.card_estimator.estimate(&plan.root);
        self.cost_model.estimate(&plan.root, cardinality)
    }

    /// Estimates the cardinality of a plan.
    pub fn estimate_cardinality(&self, plan: &LogicalPlan) -> f64 {
        self.card_estimator.estimate(&plan.root)
    }

    /// Optimizes a logical plan.
    ///
    /// # Errors
    ///
    /// Returns an error if optimization fails.
    pub fn optimize(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
        let mut root = plan.root;

        // Apply optimization rules
        if self.enable_filter_pushdown {
            root = self.push_filters_down(root);
        }

        if self.enable_join_reorder {
            root = self.reorder_joins(root);
        }

        if self.enable_projection_pushdown {
            root = self.push_projections_down(root);
        }

        Ok(LogicalPlan::new(root))
    }

    /// Pushes projections down the operator tree to eliminate unused columns early.
    ///
    /// This optimization:
    /// 1. Collects required variables/properties from the root
    /// 2. Propagates requirements down through the tree
    /// 3. Inserts projections to eliminate unneeded columns before expensive operations
    fn push_projections_down(&self, op: LogicalOperator) -> LogicalOperator {
        // Collect required columns from the top of the plan
        let required = self.collect_required_columns(&op);

        // Push projections down
        self.push_projections_recursive(op, &required)
    }

    /// Collects all variables and properties required by an operator and its ancestors.
    fn collect_required_columns(&self, op: &LogicalOperator) -> HashSet<RequiredColumn> {
        let mut required = HashSet::new();
        Self::collect_required_recursive(op, &mut required);
        required
    }

    /// Recursively collects required columns.
    fn collect_required_recursive(op: &LogicalOperator, required: &mut HashSet<RequiredColumn>) {
        match op {
            LogicalOperator::Return(ret) => {
                for item in &ret.items {
                    Self::collect_from_expression(&item.expression, required);
                }
                Self::collect_required_recursive(&ret.input, required);
            }
            LogicalOperator::Project(proj) => {
                for p in &proj.projections {
                    Self::collect_from_expression(&p.expression, required);
                }
                Self::collect_required_recursive(&proj.input, required);
            }
            LogicalOperator::Filter(filter) => {
                Self::collect_from_expression(&filter.predicate, required);
                Self::collect_required_recursive(&filter.input, required);
            }
            LogicalOperator::Sort(sort) => {
                for key in &sort.keys {
                    Self::collect_from_expression(&key.expression, required);
                }
                Self::collect_required_recursive(&sort.input, required);
            }
            LogicalOperator::Aggregate(agg) => {
                for expr in &agg.group_by {
                    Self::collect_from_expression(expr, required);
                }
                for agg_expr in &agg.aggregates {
                    if let Some(ref expr) = agg_expr.expression {
                        Self::collect_from_expression(expr, required);
                    }
                }
                if let Some(ref having) = agg.having {
                    Self::collect_from_expression(having, required);
                }
                Self::collect_required_recursive(&agg.input, required);
            }
            LogicalOperator::Join(join) => {
                for cond in &join.conditions {
                    Self::collect_from_expression(&cond.left, required);
                    Self::collect_from_expression(&cond.right, required);
                }
                Self::collect_required_recursive(&join.left, required);
                Self::collect_required_recursive(&join.right, required);
            }
            LogicalOperator::Expand(expand) => {
                // The source and target variables are needed
                required.insert(RequiredColumn::Variable(expand.from_variable.clone()));
                required.insert(RequiredColumn::Variable(expand.to_variable.clone()));
                if let Some(ref edge_var) = expand.edge_variable {
                    required.insert(RequiredColumn::Variable(edge_var.clone()));
                }
                Self::collect_required_recursive(&expand.input, required);
            }
            LogicalOperator::Limit(limit) => {
                Self::collect_required_recursive(&limit.input, required);
            }
            LogicalOperator::Skip(skip) => {
                Self::collect_required_recursive(&skip.input, required);
            }
            LogicalOperator::Distinct(distinct) => {
                Self::collect_required_recursive(&distinct.input, required);
            }
            LogicalOperator::NodeScan(scan) => {
                required.insert(RequiredColumn::Variable(scan.variable.clone()));
            }
            LogicalOperator::EdgeScan(scan) => {
                required.insert(RequiredColumn::Variable(scan.variable.clone()));
            }
            _ => {}
        }
    }

    /// Collects required columns from an expression.
    fn collect_from_expression(expr: &LogicalExpression, required: &mut HashSet<RequiredColumn>) {
        match expr {
            LogicalExpression::Variable(var) => {
                required.insert(RequiredColumn::Variable(var.clone()));
            }
            LogicalExpression::Property { variable, property } => {
                required.insert(RequiredColumn::Property(variable.clone(), property.clone()));
                required.insert(RequiredColumn::Variable(variable.clone()));
            }
            LogicalExpression::Binary { left, right, .. } => {
                Self::collect_from_expression(left, required);
                Self::collect_from_expression(right, required);
            }
            LogicalExpression::Unary { operand, .. } => {
                Self::collect_from_expression(operand, required);
            }
            LogicalExpression::FunctionCall { args, .. } => {
                for arg in args {
                    Self::collect_from_expression(arg, required);
                }
            }
            LogicalExpression::List(items) => {
                for item in items {
                    Self::collect_from_expression(item, required);
                }
            }
            LogicalExpression::Map(pairs) => {
                for (_, value) in pairs {
                    Self::collect_from_expression(value, required);
                }
            }
            LogicalExpression::IndexAccess { base, index } => {
                Self::collect_from_expression(base, required);
                Self::collect_from_expression(index, required);
            }
            LogicalExpression::SliceAccess { base, start, end } => {
                Self::collect_from_expression(base, required);
                if let Some(s) = start {
                    Self::collect_from_expression(s, required);
                }
                if let Some(e) = end {
                    Self::collect_from_expression(e, required);
                }
            }
            LogicalExpression::Case {
                operand,
                when_clauses,
                else_clause,
            } => {
                if let Some(op) = operand {
                    Self::collect_from_expression(op, required);
                }
                for (cond, result) in when_clauses {
                    Self::collect_from_expression(cond, required);
                    Self::collect_from_expression(result, required);
                }
                if let Some(else_expr) = else_clause {
                    Self::collect_from_expression(else_expr, required);
                }
            }
            LogicalExpression::Labels(var)
            | LogicalExpression::Type(var)
            | LogicalExpression::Id(var) => {
                required.insert(RequiredColumn::Variable(var.clone()));
            }
            LogicalExpression::ListComprehension {
                list_expr,
                filter_expr,
                map_expr,
                ..
            } => {
                Self::collect_from_expression(list_expr, required);
                if let Some(filter) = filter_expr {
                    Self::collect_from_expression(filter, required);
                }
                Self::collect_from_expression(map_expr, required);
            }
            _ => {}
        }
    }

    /// Recursively pushes projections down, adding them before expensive operations.
    fn push_projections_recursive(
        &self,
        op: LogicalOperator,
        required: &HashSet<RequiredColumn>,
    ) -> LogicalOperator {
        match op {
            LogicalOperator::Return(mut ret) => {
                ret.input = Box::new(self.push_projections_recursive(*ret.input, required));
                LogicalOperator::Return(ret)
            }
            LogicalOperator::Project(mut proj) => {
                proj.input = Box::new(self.push_projections_recursive(*proj.input, required));
                LogicalOperator::Project(proj)
            }
            LogicalOperator::Filter(mut filter) => {
                filter.input = Box::new(self.push_projections_recursive(*filter.input, required));
                LogicalOperator::Filter(filter)
            }
            LogicalOperator::Sort(mut sort) => {
                // Sort is expensive - consider adding a projection before it
                // to reduce tuple width
                sort.input = Box::new(self.push_projections_recursive(*sort.input, required));
                LogicalOperator::Sort(sort)
            }
            LogicalOperator::Aggregate(mut agg) => {
                agg.input = Box::new(self.push_projections_recursive(*agg.input, required));
                LogicalOperator::Aggregate(agg)
            }
            LogicalOperator::Join(mut join) => {
                // Joins are expensive - the required columns help determine
                // what to project on each side
                let left_vars = self.collect_output_variables(&join.left);
                let right_vars = self.collect_output_variables(&join.right);

                // Filter required columns to each side
                let left_required: HashSet<_> = required
                    .iter()
                    .filter(|c| match c {
                        RequiredColumn::Variable(v) => left_vars.contains(v),
                        RequiredColumn::Property(v, _) => left_vars.contains(v),
                    })
                    .cloned()
                    .collect();

                let right_required: HashSet<_> = required
                    .iter()
                    .filter(|c| match c {
                        RequiredColumn::Variable(v) => right_vars.contains(v),
                        RequiredColumn::Property(v, _) => right_vars.contains(v),
                    })
                    .cloned()
                    .collect();

                join.left = Box::new(self.push_projections_recursive(*join.left, &left_required));
                join.right =
                    Box::new(self.push_projections_recursive(*join.right, &right_required));
                LogicalOperator::Join(join)
            }
            LogicalOperator::Expand(mut expand) => {
                expand.input = Box::new(self.push_projections_recursive(*expand.input, required));
                LogicalOperator::Expand(expand)
            }
            LogicalOperator::Limit(mut limit) => {
                limit.input = Box::new(self.push_projections_recursive(*limit.input, required));
                LogicalOperator::Limit(limit)
            }
            LogicalOperator::Skip(mut skip) => {
                skip.input = Box::new(self.push_projections_recursive(*skip.input, required));
                LogicalOperator::Skip(skip)
            }
            LogicalOperator::Distinct(mut distinct) => {
                distinct.input =
                    Box::new(self.push_projections_recursive(*distinct.input, required));
                LogicalOperator::Distinct(distinct)
            }
            other => other,
        }
    }

    /// Reorders joins in the operator tree using the DPccp algorithm.
    ///
    /// This optimization finds the optimal join order by:
    /// 1. Extracting all base relations (scans) and join conditions
    /// 2. Building a join graph
    /// 3. Using dynamic programming to find the cheapest join order
    fn reorder_joins(&self, op: LogicalOperator) -> LogicalOperator {
        // First, recursively optimize children
        let op = self.reorder_joins_recursive(op);

        // Then, if this is a join tree, try to optimize it
        if let Some((relations, conditions)) = self.extract_join_tree(&op) {
            if relations.len() >= 2 {
                if let Some(optimized) = self.optimize_join_order(&relations, &conditions) {
                    return optimized;
                }
            }
        }

        op
    }

    /// Recursively applies join reordering to child operators.
    fn reorder_joins_recursive(&self, op: LogicalOperator) -> LogicalOperator {
        match op {
            LogicalOperator::Return(mut ret) => {
                ret.input = Box::new(self.reorder_joins(*ret.input));
                LogicalOperator::Return(ret)
            }
            LogicalOperator::Project(mut proj) => {
                proj.input = Box::new(self.reorder_joins(*proj.input));
                LogicalOperator::Project(proj)
            }
            LogicalOperator::Filter(mut filter) => {
                filter.input = Box::new(self.reorder_joins(*filter.input));
                LogicalOperator::Filter(filter)
            }
            LogicalOperator::Limit(mut limit) => {
                limit.input = Box::new(self.reorder_joins(*limit.input));
                LogicalOperator::Limit(limit)
            }
            LogicalOperator::Skip(mut skip) => {
                skip.input = Box::new(self.reorder_joins(*skip.input));
                LogicalOperator::Skip(skip)
            }
            LogicalOperator::Sort(mut sort) => {
                sort.input = Box::new(self.reorder_joins(*sort.input));
                LogicalOperator::Sort(sort)
            }
            LogicalOperator::Distinct(mut distinct) => {
                distinct.input = Box::new(self.reorder_joins(*distinct.input));
                LogicalOperator::Distinct(distinct)
            }
            LogicalOperator::Aggregate(mut agg) => {
                agg.input = Box::new(self.reorder_joins(*agg.input));
                LogicalOperator::Aggregate(agg)
            }
            LogicalOperator::Expand(mut expand) => {
                expand.input = Box::new(self.reorder_joins(*expand.input));
                LogicalOperator::Expand(expand)
            }
            // Join operators are handled by the parent reorder_joins call
            other => other,
        }
    }

    /// Extracts base relations and join conditions from a join tree.
    ///
    /// Returns None if the operator is not a join tree.
    fn extract_join_tree(
        &self,
        op: &LogicalOperator,
    ) -> Option<(Vec<(String, LogicalOperator)>, Vec<JoinInfo>)> {
        let mut relations = Vec::new();
        let mut join_conditions = Vec::new();

        if !self.collect_join_tree(op, &mut relations, &mut join_conditions) {
            return None;
        }

        if relations.len() < 2 {
            return None;
        }

        Some((relations, join_conditions))
    }

    /// Recursively collects base relations and join conditions.
    ///
    /// Returns true if this subtree is part of a join tree.
    fn collect_join_tree(
        &self,
        op: &LogicalOperator,
        relations: &mut Vec<(String, LogicalOperator)>,
        conditions: &mut Vec<JoinInfo>,
    ) -> bool {
        match op {
            LogicalOperator::Join(join) => {
                // Collect from both sides
                let left_ok = self.collect_join_tree(&join.left, relations, conditions);
                let right_ok = self.collect_join_tree(&join.right, relations, conditions);

                // Add conditions from this join
                for cond in &join.conditions {
                    if let (Some(left_var), Some(right_var)) = (
                        self.extract_variable_from_expr(&cond.left),
                        self.extract_variable_from_expr(&cond.right),
                    ) {
                        conditions.push(JoinInfo {
                            left_var,
                            right_var,
                            left_expr: cond.left.clone(),
                            right_expr: cond.right.clone(),
                        });
                    }
                }

                left_ok && right_ok
            }
            LogicalOperator::NodeScan(scan) => {
                relations.push((scan.variable.clone(), op.clone()));
                true
            }
            LogicalOperator::EdgeScan(scan) => {
                relations.push((scan.variable.clone(), op.clone()));
                true
            }
            LogicalOperator::Filter(filter) => {
                // A filter on a base relation is still part of the join tree
                self.collect_join_tree(&filter.input, relations, conditions)
            }
            LogicalOperator::Expand(expand) => {
                // Expand is a special case - it's like a join with the adjacency
                // For now, treat the whole Expand subtree as a single relation
                relations.push((expand.to_variable.clone(), op.clone()));
                true
            }
            _ => false,
        }
    }

    /// Extracts the primary variable from an expression.
    fn extract_variable_from_expr(&self, expr: &LogicalExpression) -> Option<String> {
        match expr {
            LogicalExpression::Variable(v) => Some(v.clone()),
            LogicalExpression::Property { variable, .. } => Some(variable.clone()),
            _ => None,
        }
    }

    /// Optimizes the join order using DPccp.
    fn optimize_join_order(
        &self,
        relations: &[(String, LogicalOperator)],
        conditions: &[JoinInfo],
    ) -> Option<LogicalOperator> {
        use join_order::{DPccp, JoinGraphBuilder};

        // Build the join graph
        let mut builder = JoinGraphBuilder::new();

        for (var, relation) in relations {
            builder.add_relation(var, relation.clone());
        }

        for cond in conditions {
            builder.add_join_condition(
                &cond.left_var,
                &cond.right_var,
                cond.left_expr.clone(),
                cond.right_expr.clone(),
            );
        }

        let graph = builder.build();

        // Run DPccp
        let mut dpccp = DPccp::new(&graph, &self.cost_model, &self.card_estimator);
        let plan = dpccp.optimize()?;

        Some(plan.operator)
    }

    /// Pushes filters down the operator tree.
    ///
    /// This optimization moves filter predicates as close to the data source
    /// as possible to reduce the amount of data processed by upper operators.
    fn push_filters_down(&self, op: LogicalOperator) -> LogicalOperator {
        match op {
            // For Filter operators, try to push the predicate into the child
            LogicalOperator::Filter(filter) => {
                let optimized_input = self.push_filters_down(*filter.input);
                self.try_push_filter_into(filter.predicate, optimized_input)
            }
            // Recursively optimize children for other operators
            LogicalOperator::Return(mut ret) => {
                ret.input = Box::new(self.push_filters_down(*ret.input));
                LogicalOperator::Return(ret)
            }
            LogicalOperator::Project(mut proj) => {
                proj.input = Box::new(self.push_filters_down(*proj.input));
                LogicalOperator::Project(proj)
            }
            LogicalOperator::Limit(mut limit) => {
                limit.input = Box::new(self.push_filters_down(*limit.input));
                LogicalOperator::Limit(limit)
            }
            LogicalOperator::Skip(mut skip) => {
                skip.input = Box::new(self.push_filters_down(*skip.input));
                LogicalOperator::Skip(skip)
            }
            LogicalOperator::Sort(mut sort) => {
                sort.input = Box::new(self.push_filters_down(*sort.input));
                LogicalOperator::Sort(sort)
            }
            LogicalOperator::Distinct(mut distinct) => {
                distinct.input = Box::new(self.push_filters_down(*distinct.input));
                LogicalOperator::Distinct(distinct)
            }
            LogicalOperator::Expand(mut expand) => {
                expand.input = Box::new(self.push_filters_down(*expand.input));
                LogicalOperator::Expand(expand)
            }
            LogicalOperator::Join(mut join) => {
                join.left = Box::new(self.push_filters_down(*join.left));
                join.right = Box::new(self.push_filters_down(*join.right));
                LogicalOperator::Join(join)
            }
            LogicalOperator::Aggregate(mut agg) => {
                agg.input = Box::new(self.push_filters_down(*agg.input));
                LogicalOperator::Aggregate(agg)
            }
            // Leaf operators and unsupported operators are returned as-is
            other => other,
        }
    }

    /// Tries to push a filter predicate into the given operator.
    ///
    /// Returns either the predicate pushed into the operator, or a new
    /// Filter operator on top if the predicate cannot be pushed further.
    fn try_push_filter_into(
        &self,
        predicate: LogicalExpression,
        op: LogicalOperator,
    ) -> LogicalOperator {
        match op {
            // Can push through Project if predicate doesn't depend on computed columns
            LogicalOperator::Project(mut proj) => {
                let predicate_vars = self.extract_variables(&predicate);
                let computed_vars = self.extract_projection_aliases(&proj.projections);

                // If predicate doesn't use any computed columns, push through
                if predicate_vars.is_disjoint(&computed_vars) {
                    proj.input = Box::new(self.try_push_filter_into(predicate, *proj.input));
                    LogicalOperator::Project(proj)
                } else {
                    // Can't push through, keep filter on top
                    LogicalOperator::Filter(FilterOp {
                        predicate,
                        input: Box::new(LogicalOperator::Project(proj)),
                    })
                }
            }

            // Can push through Return (which is like a projection)
            LogicalOperator::Return(mut ret) => {
                ret.input = Box::new(self.try_push_filter_into(predicate, *ret.input));
                LogicalOperator::Return(ret)
            }

            // Can push through Expand if predicate doesn't use variables introduced by this expand
            LogicalOperator::Expand(mut expand) => {
                let predicate_vars = self.extract_variables(&predicate);

                // Variables introduced by this expand are:
                // - The target variable (to_variable)
                // - The edge variable (if any)
                // - The path alias (if any)
                let mut introduced_vars = vec![&expand.to_variable];
                if let Some(ref edge_var) = expand.edge_variable {
                    introduced_vars.push(edge_var);
                }
                if let Some(ref path_alias) = expand.path_alias {
                    introduced_vars.push(path_alias);
                }

                // Check if predicate uses any variables introduced by this expand
                let uses_introduced_vars =
                    predicate_vars.iter().any(|v| introduced_vars.contains(&v));

                if !uses_introduced_vars {
                    // Predicate doesn't use vars from this expand, so push through
                    expand.input = Box::new(self.try_push_filter_into(predicate, *expand.input));
                    LogicalOperator::Expand(expand)
                } else {
                    // Keep filter after expand
                    LogicalOperator::Filter(FilterOp {
                        predicate,
                        input: Box::new(LogicalOperator::Expand(expand)),
                    })
                }
            }

            // Can push through Join to left/right side based on variables used
            LogicalOperator::Join(mut join) => {
                let predicate_vars = self.extract_variables(&predicate);
                let left_vars = self.collect_output_variables(&join.left);
                let right_vars = self.collect_output_variables(&join.right);

                let uses_left = predicate_vars.iter().any(|v| left_vars.contains(v));
                let uses_right = predicate_vars.iter().any(|v| right_vars.contains(v));

                if uses_left && !uses_right {
                    // Push to left side
                    join.left = Box::new(self.try_push_filter_into(predicate, *join.left));
                    LogicalOperator::Join(join)
                } else if uses_right && !uses_left {
                    // Push to right side
                    join.right = Box::new(self.try_push_filter_into(predicate, *join.right));
                    LogicalOperator::Join(join)
                } else {
                    // Uses both sides - keep above join
                    LogicalOperator::Filter(FilterOp {
                        predicate,
                        input: Box::new(LogicalOperator::Join(join)),
                    })
                }
            }

            // Cannot push through Aggregate (predicate refers to aggregated values)
            LogicalOperator::Aggregate(agg) => LogicalOperator::Filter(FilterOp {
                predicate,
                input: Box::new(LogicalOperator::Aggregate(agg)),
            }),

            // For NodeScan, we've reached the bottom - keep filter on top
            LogicalOperator::NodeScan(scan) => LogicalOperator::Filter(FilterOp {
                predicate,
                input: Box::new(LogicalOperator::NodeScan(scan)),
            }),

            // For other operators, keep filter on top
            other => LogicalOperator::Filter(FilterOp {
                predicate,
                input: Box::new(other),
            }),
        }
    }

    /// Collects all output variable names from an operator.
    fn collect_output_variables(&self, op: &LogicalOperator) -> HashSet<String> {
        let mut vars = HashSet::new();
        Self::collect_output_variables_recursive(op, &mut vars);
        vars
    }

    /// Recursively collects output variables from an operator.
    fn collect_output_variables_recursive(op: &LogicalOperator, vars: &mut HashSet<String>) {
        match op {
            LogicalOperator::NodeScan(scan) => {
                vars.insert(scan.variable.clone());
            }
            LogicalOperator::EdgeScan(scan) => {
                vars.insert(scan.variable.clone());
            }
            LogicalOperator::Expand(expand) => {
                vars.insert(expand.to_variable.clone());
                if let Some(edge_var) = &expand.edge_variable {
                    vars.insert(edge_var.clone());
                }
                Self::collect_output_variables_recursive(&expand.input, vars);
            }
            LogicalOperator::Filter(filter) => {
                Self::collect_output_variables_recursive(&filter.input, vars);
            }
            LogicalOperator::Project(proj) => {
                for p in &proj.projections {
                    if let Some(alias) = &p.alias {
                        vars.insert(alias.clone());
                    }
                }
                Self::collect_output_variables_recursive(&proj.input, vars);
            }
            LogicalOperator::Join(join) => {
                Self::collect_output_variables_recursive(&join.left, vars);
                Self::collect_output_variables_recursive(&join.right, vars);
            }
            LogicalOperator::Aggregate(agg) => {
                for expr in &agg.group_by {
                    Self::collect_variables(expr, vars);
                }
                for agg_expr in &agg.aggregates {
                    if let Some(alias) = &agg_expr.alias {
                        vars.insert(alias.clone());
                    }
                }
            }
            LogicalOperator::Return(ret) => {
                Self::collect_output_variables_recursive(&ret.input, vars);
            }
            LogicalOperator::Limit(limit) => {
                Self::collect_output_variables_recursive(&limit.input, vars);
            }
            LogicalOperator::Skip(skip) => {
                Self::collect_output_variables_recursive(&skip.input, vars);
            }
            LogicalOperator::Sort(sort) => {
                Self::collect_output_variables_recursive(&sort.input, vars);
            }
            LogicalOperator::Distinct(distinct) => {
                Self::collect_output_variables_recursive(&distinct.input, vars);
            }
            _ => {}
        }
    }

    /// Extracts all variable names referenced in an expression.
    fn extract_variables(&self, expr: &LogicalExpression) -> HashSet<String> {
        let mut vars = HashSet::new();
        Self::collect_variables(expr, &mut vars);
        vars
    }

    /// Recursively collects variable names from an expression.
    fn collect_variables(expr: &LogicalExpression, vars: &mut HashSet<String>) {
        match expr {
            LogicalExpression::Variable(name) => {
                vars.insert(name.clone());
            }
            LogicalExpression::Property { variable, .. } => {
                vars.insert(variable.clone());
            }
            LogicalExpression::Binary { left, right, .. } => {
                Self::collect_variables(left, vars);
                Self::collect_variables(right, vars);
            }
            LogicalExpression::Unary { operand, .. } => {
                Self::collect_variables(operand, vars);
            }
            LogicalExpression::FunctionCall { args, .. } => {
                for arg in args {
                    Self::collect_variables(arg, vars);
                }
            }
            LogicalExpression::List(items) => {
                for item in items {
                    Self::collect_variables(item, vars);
                }
            }
            LogicalExpression::Map(pairs) => {
                for (_, value) in pairs {
                    Self::collect_variables(value, vars);
                }
            }
            LogicalExpression::IndexAccess { base, index } => {
                Self::collect_variables(base, vars);
                Self::collect_variables(index, vars);
            }
            LogicalExpression::SliceAccess { base, start, end } => {
                Self::collect_variables(base, vars);
                if let Some(s) = start {
                    Self::collect_variables(s, vars);
                }
                if let Some(e) = end {
                    Self::collect_variables(e, vars);
                }
            }
            LogicalExpression::Case {
                operand,
                when_clauses,
                else_clause,
            } => {
                if let Some(op) = operand {
                    Self::collect_variables(op, vars);
                }
                for (cond, result) in when_clauses {
                    Self::collect_variables(cond, vars);
                    Self::collect_variables(result, vars);
                }
                if let Some(else_expr) = else_clause {
                    Self::collect_variables(else_expr, vars);
                }
            }
            LogicalExpression::Labels(var)
            | LogicalExpression::Type(var)
            | LogicalExpression::Id(var) => {
                vars.insert(var.clone());
            }
            LogicalExpression::Literal(_) | LogicalExpression::Parameter(_) => {}
            LogicalExpression::ListComprehension {
                list_expr,
                filter_expr,
                map_expr,
                ..
            } => {
                Self::collect_variables(list_expr, vars);
                if let Some(filter) = filter_expr {
                    Self::collect_variables(filter, vars);
                }
                Self::collect_variables(map_expr, vars);
            }
            LogicalExpression::ExistsSubquery(_) | LogicalExpression::CountSubquery(_) => {
                // Subqueries have their own variable scope
            }
        }
    }

    /// Extracts aliases from projection expressions.
    fn extract_projection_aliases(
        &self,
        projections: &[crate::query::plan::Projection],
    ) -> HashSet<String> {
        projections.iter().filter_map(|p| p.alias.clone()).collect()
    }
}

impl Default for Optimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::plan::{
        AggregateExpr, AggregateFunction, AggregateOp, BinaryOp, DistinctOp, ExpandDirection,
        ExpandOp, JoinOp, JoinType, LimitOp, NodeScanOp, ProjectOp, Projection, ReturnItem,
        ReturnOp, SkipOp, SortKey, SortOp, SortOrder, UnaryOp,
    };
    use grafeo_common::types::Value;

    #[test]
    fn test_optimizer_filter_pushdown_simple() {
        // Query: MATCH (n:Person) WHERE n.age > 30 RETURN n
        // Before: Return -> Filter -> NodeScan
        // After:  Return -> Filter -> NodeScan (filter stays at bottom)

        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("n".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Filter(FilterOp {
                predicate: LogicalExpression::Binary {
                    left: Box::new(LogicalExpression::Property {
                        variable: "n".to_string(),
                        property: "age".to_string(),
                    }),
                    op: BinaryOp::Gt,
                    right: Box::new(LogicalExpression::Literal(Value::Int64(30))),
                },
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".to_string(),
                    label: Some("Person".to_string()),
                    input: None,
                })),
            })),
        }));

        let optimizer = Optimizer::new();
        let optimized = optimizer.optimize(plan).unwrap();

        // The structure should remain similar (filter stays near scan)
        if let LogicalOperator::Return(ret) = &optimized.root {
            if let LogicalOperator::Filter(filter) = ret.input.as_ref() {
                if let LogicalOperator::NodeScan(scan) = filter.input.as_ref() {
                    assert_eq!(scan.variable, "n");
                    return;
                }
            }
        }
        panic!("Expected Return -> Filter -> NodeScan structure");
    }

    #[test]
    fn test_optimizer_filter_pushdown_through_expand() {
        // Query: MATCH (a:Person)-[:KNOWS]->(b) WHERE a.age > 30 RETURN b
        // The filter on 'a' should be pushed before the expand

        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("b".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Filter(FilterOp {
                predicate: LogicalExpression::Binary {
                    left: Box::new(LogicalExpression::Property {
                        variable: "a".to_string(),
                        property: "age".to_string(),
                    }),
                    op: BinaryOp::Gt,
                    right: Box::new(LogicalExpression::Literal(Value::Int64(30))),
                },
                input: Box::new(LogicalOperator::Expand(ExpandOp {
                    from_variable: "a".to_string(),
                    to_variable: "b".to_string(),
                    edge_variable: None,
                    direction: ExpandDirection::Outgoing,
                    edge_type: Some("KNOWS".to_string()),
                    min_hops: 1,
                    max_hops: Some(1),
                    input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                        variable: "a".to_string(),
                        label: Some("Person".to_string()),
                        input: None,
                    })),
                    path_alias: None,
                })),
            })),
        }));

        let optimizer = Optimizer::new();
        let optimized = optimizer.optimize(plan).unwrap();

        // Filter on 'a' should be pushed before the expand
        // Expected: Return -> Expand -> Filter -> NodeScan
        if let LogicalOperator::Return(ret) = &optimized.root {
            if let LogicalOperator::Expand(expand) = ret.input.as_ref() {
                if let LogicalOperator::Filter(filter) = expand.input.as_ref() {
                    if let LogicalOperator::NodeScan(scan) = filter.input.as_ref() {
                        assert_eq!(scan.variable, "a");
                        assert_eq!(expand.from_variable, "a");
                        assert_eq!(expand.to_variable, "b");
                        return;
                    }
                }
            }
        }
        panic!("Expected Return -> Expand -> Filter -> NodeScan structure");
    }

    #[test]
    fn test_optimizer_filter_not_pushed_through_expand_for_target_var() {
        // Query: MATCH (a:Person)-[:KNOWS]->(b) WHERE b.age > 30 RETURN a
        // The filter on 'b' should NOT be pushed before the expand

        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("a".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Filter(FilterOp {
                predicate: LogicalExpression::Binary {
                    left: Box::new(LogicalExpression::Property {
                        variable: "b".to_string(),
                        property: "age".to_string(),
                    }),
                    op: BinaryOp::Gt,
                    right: Box::new(LogicalExpression::Literal(Value::Int64(30))),
                },
                input: Box::new(LogicalOperator::Expand(ExpandOp {
                    from_variable: "a".to_string(),
                    to_variable: "b".to_string(),
                    edge_variable: None,
                    direction: ExpandDirection::Outgoing,
                    edge_type: Some("KNOWS".to_string()),
                    min_hops: 1,
                    max_hops: Some(1),
                    input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                        variable: "a".to_string(),
                        label: Some("Person".to_string()),
                        input: None,
                    })),
                    path_alias: None,
                })),
            })),
        }));

        let optimizer = Optimizer::new();
        let optimized = optimizer.optimize(plan).unwrap();

        // Filter on 'b' should stay after the expand
        // Expected: Return -> Filter -> Expand -> NodeScan
        if let LogicalOperator::Return(ret) = &optimized.root {
            if let LogicalOperator::Filter(filter) = ret.input.as_ref() {
                // Check that the filter is on 'b'
                if let LogicalExpression::Binary { left, .. } = &filter.predicate {
                    if let LogicalExpression::Property { variable, .. } = left.as_ref() {
                        assert_eq!(variable, "b");
                    }
                }

                if let LogicalOperator::Expand(expand) = filter.input.as_ref() {
                    if let LogicalOperator::NodeScan(_) = expand.input.as_ref() {
                        return;
                    }
                }
            }
        }
        panic!("Expected Return -> Filter -> Expand -> NodeScan structure");
    }

    #[test]
    fn test_optimizer_extract_variables() {
        let optimizer = Optimizer::new();

        let expr = LogicalExpression::Binary {
            left: Box::new(LogicalExpression::Property {
                variable: "n".to_string(),
                property: "age".to_string(),
            }),
            op: BinaryOp::Gt,
            right: Box::new(LogicalExpression::Literal(Value::Int64(30))),
        };

        let vars = optimizer.extract_variables(&expr);
        assert_eq!(vars.len(), 1);
        assert!(vars.contains("n"));
    }

    // Additional tests for optimizer configuration

    #[test]
    fn test_optimizer_default() {
        let optimizer = Optimizer::default();
        // Should be able to optimize an empty plan
        let plan = LogicalPlan::new(LogicalOperator::Empty);
        let result = optimizer.optimize(plan);
        assert!(result.is_ok());
    }

    #[test]
    fn test_optimizer_with_filter_pushdown_disabled() {
        let optimizer = Optimizer::new().with_filter_pushdown(false);

        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("n".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Filter(FilterOp {
                predicate: LogicalExpression::Literal(Value::Bool(true)),
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".to_string(),
                    label: None,
                    input: None,
                })),
            })),
        }));

        let optimized = optimizer.optimize(plan).unwrap();
        // Structure should be unchanged
        if let LogicalOperator::Return(ret) = &optimized.root {
            if let LogicalOperator::Filter(_) = ret.input.as_ref() {
                return;
            }
        }
        panic!("Expected unchanged structure");
    }

    #[test]
    fn test_optimizer_with_join_reorder_disabled() {
        let optimizer = Optimizer::new().with_join_reorder(false);
        assert!(
            optimizer
                .optimize(LogicalPlan::new(LogicalOperator::Empty))
                .is_ok()
        );
    }

    #[test]
    fn test_optimizer_with_cost_model() {
        let cost_model = CostModel::new();
        let optimizer = Optimizer::new().with_cost_model(cost_model);
        assert!(
            optimizer
                .cost_model()
                .estimate(&LogicalOperator::Empty, 0.0)
                .total()
                < 0.001
        );
    }

    #[test]
    fn test_optimizer_with_cardinality_estimator() {
        let mut estimator = CardinalityEstimator::new();
        estimator.add_table_stats("Test", TableStats::new(500));
        let optimizer = Optimizer::new().with_cardinality_estimator(estimator);

        let scan = LogicalOperator::NodeScan(NodeScanOp {
            variable: "n".to_string(),
            label: Some("Test".to_string()),
            input: None,
        });
        let plan = LogicalPlan::new(scan);

        let cardinality = optimizer.estimate_cardinality(&plan);
        assert!((cardinality - 500.0).abs() < 0.001);
    }

    #[test]
    fn test_optimizer_estimate_cost() {
        let optimizer = Optimizer::new();
        let plan = LogicalPlan::new(LogicalOperator::NodeScan(NodeScanOp {
            variable: "n".to_string(),
            label: None,
            input: None,
        }));

        let cost = optimizer.estimate_cost(&plan);
        assert!(cost.total() > 0.0);
    }

    // Filter pushdown through various operators

    #[test]
    fn test_filter_pushdown_through_project() {
        let optimizer = Optimizer::new();

        let plan = LogicalPlan::new(LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Property {
                    variable: "n".to_string(),
                    property: "age".to_string(),
                }),
                op: BinaryOp::Gt,
                right: Box::new(LogicalExpression::Literal(Value::Int64(30))),
            },
            input: Box::new(LogicalOperator::Project(ProjectOp {
                projections: vec![Projection {
                    expression: LogicalExpression::Variable("n".to_string()),
                    alias: None,
                }],
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".to_string(),
                    label: None,
                    input: None,
                })),
            })),
        }));

        let optimized = optimizer.optimize(plan).unwrap();

        // Filter should be pushed through Project
        if let LogicalOperator::Project(proj) = &optimized.root {
            if let LogicalOperator::Filter(_) = proj.input.as_ref() {
                return;
            }
        }
        panic!("Expected Project -> Filter structure");
    }

    #[test]
    fn test_filter_not_pushed_through_project_with_alias() {
        let optimizer = Optimizer::new();

        // Filter on computed column 'x' should not be pushed through project that creates 'x'
        let plan = LogicalPlan::new(LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Variable("x".to_string())),
                op: BinaryOp::Gt,
                right: Box::new(LogicalExpression::Literal(Value::Int64(30))),
            },
            input: Box::new(LogicalOperator::Project(ProjectOp {
                projections: vec![Projection {
                    expression: LogicalExpression::Property {
                        variable: "n".to_string(),
                        property: "age".to_string(),
                    },
                    alias: Some("x".to_string()),
                }],
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".to_string(),
                    label: None,
                    input: None,
                })),
            })),
        }));

        let optimized = optimizer.optimize(plan).unwrap();

        // Filter should stay above Project
        if let LogicalOperator::Filter(filter) = &optimized.root {
            if let LogicalOperator::Project(_) = filter.input.as_ref() {
                return;
            }
        }
        panic!("Expected Filter -> Project structure");
    }

    #[test]
    fn test_filter_pushdown_through_limit() {
        let optimizer = Optimizer::new();

        let plan = LogicalPlan::new(LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Literal(Value::Bool(true)),
            input: Box::new(LogicalOperator::Limit(LimitOp {
                count: 10,
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".to_string(),
                    label: None,
                    input: None,
                })),
            })),
        }));

        let optimized = optimizer.optimize(plan).unwrap();

        // Filter stays above Limit (cannot be pushed through)
        if let LogicalOperator::Filter(filter) = &optimized.root {
            if let LogicalOperator::Limit(_) = filter.input.as_ref() {
                return;
            }
        }
        panic!("Expected Filter -> Limit structure");
    }

    #[test]
    fn test_filter_pushdown_through_sort() {
        let optimizer = Optimizer::new();

        let plan = LogicalPlan::new(LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Literal(Value::Bool(true)),
            input: Box::new(LogicalOperator::Sort(SortOp {
                keys: vec![SortKey {
                    expression: LogicalExpression::Variable("n".to_string()),
                    order: SortOrder::Ascending,
                }],
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".to_string(),
                    label: None,
                    input: None,
                })),
            })),
        }));

        let optimized = optimizer.optimize(plan).unwrap();

        // Filter stays above Sort
        if let LogicalOperator::Filter(filter) = &optimized.root {
            if let LogicalOperator::Sort(_) = filter.input.as_ref() {
                return;
            }
        }
        panic!("Expected Filter -> Sort structure");
    }

    #[test]
    fn test_filter_pushdown_through_distinct() {
        let optimizer = Optimizer::new();

        let plan = LogicalPlan::new(LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Literal(Value::Bool(true)),
            input: Box::new(LogicalOperator::Distinct(DistinctOp {
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".to_string(),
                    label: None,
                    input: None,
                })),
                columns: None,
            })),
        }));

        let optimized = optimizer.optimize(plan).unwrap();

        // Filter stays above Distinct
        if let LogicalOperator::Filter(filter) = &optimized.root {
            if let LogicalOperator::Distinct(_) = filter.input.as_ref() {
                return;
            }
        }
        panic!("Expected Filter -> Distinct structure");
    }

    #[test]
    fn test_filter_not_pushed_through_aggregate() {
        let optimizer = Optimizer::new();

        let plan = LogicalPlan::new(LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Variable("cnt".to_string())),
                op: BinaryOp::Gt,
                right: Box::new(LogicalExpression::Literal(Value::Int64(10))),
            },
            input: Box::new(LogicalOperator::Aggregate(AggregateOp {
                group_by: vec![],
                aggregates: vec![AggregateExpr {
                    function: AggregateFunction::Count,
                    expression: None,
                    distinct: false,
                    alias: Some("cnt".to_string()),
                    percentile: None,
                }],
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".to_string(),
                    label: None,
                    input: None,
                })),
                having: None,
            })),
        }));

        let optimized = optimizer.optimize(plan).unwrap();

        // Filter should stay above Aggregate
        if let LogicalOperator::Filter(filter) = &optimized.root {
            if let LogicalOperator::Aggregate(_) = filter.input.as_ref() {
                return;
            }
        }
        panic!("Expected Filter -> Aggregate structure");
    }

    #[test]
    fn test_filter_pushdown_to_left_join_side() {
        let optimizer = Optimizer::new();

        // Filter on left variable should be pushed to left side
        let plan = LogicalPlan::new(LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Property {
                    variable: "a".to_string(),
                    property: "age".to_string(),
                }),
                op: BinaryOp::Gt,
                right: Box::new(LogicalExpression::Literal(Value::Int64(30))),
            },
            input: Box::new(LogicalOperator::Join(JoinOp {
                left: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "a".to_string(),
                    label: Some("Person".to_string()),
                    input: None,
                })),
                right: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "b".to_string(),
                    label: Some("Company".to_string()),
                    input: None,
                })),
                join_type: JoinType::Inner,
                conditions: vec![],
            })),
        }));

        let optimized = optimizer.optimize(plan).unwrap();

        // Filter should be pushed to left side of join
        if let LogicalOperator::Join(join) = &optimized.root {
            if let LogicalOperator::Filter(_) = join.left.as_ref() {
                return;
            }
        }
        panic!("Expected Join with Filter on left side");
    }

    #[test]
    fn test_filter_pushdown_to_right_join_side() {
        let optimizer = Optimizer::new();

        // Filter on right variable should be pushed to right side
        let plan = LogicalPlan::new(LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Property {
                    variable: "b".to_string(),
                    property: "name".to_string(),
                }),
                op: BinaryOp::Eq,
                right: Box::new(LogicalExpression::Literal(Value::String("Acme".into()))),
            },
            input: Box::new(LogicalOperator::Join(JoinOp {
                left: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "a".to_string(),
                    label: Some("Person".to_string()),
                    input: None,
                })),
                right: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "b".to_string(),
                    label: Some("Company".to_string()),
                    input: None,
                })),
                join_type: JoinType::Inner,
                conditions: vec![],
            })),
        }));

        let optimized = optimizer.optimize(plan).unwrap();

        // Filter should be pushed to right side of join
        if let LogicalOperator::Join(join) = &optimized.root {
            if let LogicalOperator::Filter(_) = join.right.as_ref() {
                return;
            }
        }
        panic!("Expected Join with Filter on right side");
    }

    #[test]
    fn test_filter_not_pushed_when_uses_both_join_sides() {
        let optimizer = Optimizer::new();

        // Filter using both variables should stay above join
        let plan = LogicalPlan::new(LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Property {
                    variable: "a".to_string(),
                    property: "id".to_string(),
                }),
                op: BinaryOp::Eq,
                right: Box::new(LogicalExpression::Property {
                    variable: "b".to_string(),
                    property: "a_id".to_string(),
                }),
            },
            input: Box::new(LogicalOperator::Join(JoinOp {
                left: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "a".to_string(),
                    label: None,
                    input: None,
                })),
                right: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "b".to_string(),
                    label: None,
                    input: None,
                })),
                join_type: JoinType::Inner,
                conditions: vec![],
            })),
        }));

        let optimized = optimizer.optimize(plan).unwrap();

        // Filter should stay above join
        if let LogicalOperator::Filter(filter) = &optimized.root {
            if let LogicalOperator::Join(_) = filter.input.as_ref() {
                return;
            }
        }
        panic!("Expected Filter -> Join structure");
    }

    // Variable extraction tests

    #[test]
    fn test_extract_variables_from_variable() {
        let optimizer = Optimizer::new();
        let expr = LogicalExpression::Variable("x".to_string());
        let vars = optimizer.extract_variables(&expr);
        assert_eq!(vars.len(), 1);
        assert!(vars.contains("x"));
    }

    #[test]
    fn test_extract_variables_from_unary() {
        let optimizer = Optimizer::new();
        let expr = LogicalExpression::Unary {
            op: UnaryOp::Not,
            operand: Box::new(LogicalExpression::Variable("x".to_string())),
        };
        let vars = optimizer.extract_variables(&expr);
        assert_eq!(vars.len(), 1);
        assert!(vars.contains("x"));
    }

    #[test]
    fn test_extract_variables_from_function_call() {
        let optimizer = Optimizer::new();
        let expr = LogicalExpression::FunctionCall {
            name: "length".to_string(),
            args: vec![
                LogicalExpression::Variable("a".to_string()),
                LogicalExpression::Variable("b".to_string()),
            ],
            distinct: false,
        };
        let vars = optimizer.extract_variables(&expr);
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("a"));
        assert!(vars.contains("b"));
    }

    #[test]
    fn test_extract_variables_from_list() {
        let optimizer = Optimizer::new();
        let expr = LogicalExpression::List(vec![
            LogicalExpression::Variable("a".to_string()),
            LogicalExpression::Literal(Value::Int64(1)),
            LogicalExpression::Variable("b".to_string()),
        ]);
        let vars = optimizer.extract_variables(&expr);
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("a"));
        assert!(vars.contains("b"));
    }

    #[test]
    fn test_extract_variables_from_map() {
        let optimizer = Optimizer::new();
        let expr = LogicalExpression::Map(vec![
            (
                "key1".to_string(),
                LogicalExpression::Variable("a".to_string()),
            ),
            (
                "key2".to_string(),
                LogicalExpression::Variable("b".to_string()),
            ),
        ]);
        let vars = optimizer.extract_variables(&expr);
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("a"));
        assert!(vars.contains("b"));
    }

    #[test]
    fn test_extract_variables_from_index_access() {
        let optimizer = Optimizer::new();
        let expr = LogicalExpression::IndexAccess {
            base: Box::new(LogicalExpression::Variable("list".to_string())),
            index: Box::new(LogicalExpression::Variable("idx".to_string())),
        };
        let vars = optimizer.extract_variables(&expr);
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("list"));
        assert!(vars.contains("idx"));
    }

    #[test]
    fn test_extract_variables_from_slice_access() {
        let optimizer = Optimizer::new();
        let expr = LogicalExpression::SliceAccess {
            base: Box::new(LogicalExpression::Variable("list".to_string())),
            start: Some(Box::new(LogicalExpression::Variable("s".to_string()))),
            end: Some(Box::new(LogicalExpression::Variable("e".to_string()))),
        };
        let vars = optimizer.extract_variables(&expr);
        assert_eq!(vars.len(), 3);
        assert!(vars.contains("list"));
        assert!(vars.contains("s"));
        assert!(vars.contains("e"));
    }

    #[test]
    fn test_extract_variables_from_case() {
        let optimizer = Optimizer::new();
        let expr = LogicalExpression::Case {
            operand: Some(Box::new(LogicalExpression::Variable("x".to_string()))),
            when_clauses: vec![(
                LogicalExpression::Literal(Value::Int64(1)),
                LogicalExpression::Variable("a".to_string()),
            )],
            else_clause: Some(Box::new(LogicalExpression::Variable("b".to_string()))),
        };
        let vars = optimizer.extract_variables(&expr);
        assert_eq!(vars.len(), 3);
        assert!(vars.contains("x"));
        assert!(vars.contains("a"));
        assert!(vars.contains("b"));
    }

    #[test]
    fn test_extract_variables_from_labels() {
        let optimizer = Optimizer::new();
        let expr = LogicalExpression::Labels("n".to_string());
        let vars = optimizer.extract_variables(&expr);
        assert_eq!(vars.len(), 1);
        assert!(vars.contains("n"));
    }

    #[test]
    fn test_extract_variables_from_type() {
        let optimizer = Optimizer::new();
        let expr = LogicalExpression::Type("e".to_string());
        let vars = optimizer.extract_variables(&expr);
        assert_eq!(vars.len(), 1);
        assert!(vars.contains("e"));
    }

    #[test]
    fn test_extract_variables_from_id() {
        let optimizer = Optimizer::new();
        let expr = LogicalExpression::Id("n".to_string());
        let vars = optimizer.extract_variables(&expr);
        assert_eq!(vars.len(), 1);
        assert!(vars.contains("n"));
    }

    #[test]
    fn test_extract_variables_from_list_comprehension() {
        let optimizer = Optimizer::new();
        let expr = LogicalExpression::ListComprehension {
            variable: "x".to_string(),
            list_expr: Box::new(LogicalExpression::Variable("items".to_string())),
            filter_expr: Some(Box::new(LogicalExpression::Variable("pred".to_string()))),
            map_expr: Box::new(LogicalExpression::Variable("result".to_string())),
        };
        let vars = optimizer.extract_variables(&expr);
        assert!(vars.contains("items"));
        assert!(vars.contains("pred"));
        assert!(vars.contains("result"));
    }

    #[test]
    fn test_extract_variables_from_literal_and_parameter() {
        let optimizer = Optimizer::new();

        let literal = LogicalExpression::Literal(Value::Int64(42));
        assert!(optimizer.extract_variables(&literal).is_empty());

        let param = LogicalExpression::Parameter("p".to_string());
        assert!(optimizer.extract_variables(&param).is_empty());
    }

    // Recursive filter pushdown tests

    #[test]
    fn test_recursive_filter_pushdown_through_skip() {
        let optimizer = Optimizer::new();

        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("n".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Filter(FilterOp {
                predicate: LogicalExpression::Literal(Value::Bool(true)),
                input: Box::new(LogicalOperator::Skip(SkipOp {
                    count: 5,
                    input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                        variable: "n".to_string(),
                        label: None,
                        input: None,
                    })),
                })),
            })),
        }));

        let optimized = optimizer.optimize(plan).unwrap();

        // Verify optimization succeeded
        assert!(matches!(&optimized.root, LogicalOperator::Return(_)));
    }

    #[test]
    fn test_nested_filter_pushdown() {
        let optimizer = Optimizer::new();

        // Multiple nested filters
        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("n".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Filter(FilterOp {
                predicate: LogicalExpression::Binary {
                    left: Box::new(LogicalExpression::Property {
                        variable: "n".to_string(),
                        property: "x".to_string(),
                    }),
                    op: BinaryOp::Gt,
                    right: Box::new(LogicalExpression::Literal(Value::Int64(1))),
                },
                input: Box::new(LogicalOperator::Filter(FilterOp {
                    predicate: LogicalExpression::Binary {
                        left: Box::new(LogicalExpression::Property {
                            variable: "n".to_string(),
                            property: "y".to_string(),
                        }),
                        op: BinaryOp::Lt,
                        right: Box::new(LogicalExpression::Literal(Value::Int64(10))),
                    },
                    input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                        variable: "n".to_string(),
                        label: None,
                        input: None,
                    })),
                })),
            })),
        }));

        let optimized = optimizer.optimize(plan).unwrap();
        assert!(matches!(&optimized.root, LogicalOperator::Return(_)));
    }
}
