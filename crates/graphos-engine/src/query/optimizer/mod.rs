//! Query optimizer.
//!
//! Transforms logical plans for better performance.
//!
//! ## Optimization Rules
//!
//! - **Filter Pushdown**: Pushes filters closer to scans to reduce data early
//! - **Predicate Simplification**: Simplifies constant expressions
//! - **Join Reordering**: Optimizes join order using DPccp algorithm
//!
//! ## Submodules
//!
//! - [`cost`] - Cost model for estimating operator costs
//! - [`cardinality`] - Cardinality estimation for query operators
//! - [`join_order`] - DPccp join ordering algorithm

pub mod cardinality;
pub mod cost;
pub mod join_order;

pub use cardinality::{CardinalityEstimator, ColumnStats, TableStats};
pub use cost::{Cost, CostModel};
pub use join_order::{BitSet, DPccp, JoinGraph, JoinGraphBuilder, JoinPlan};

use crate::query::plan::{
    FilterOp, LogicalExpression, LogicalOperator, LogicalPlan,
};
use graphos_common::utils::error::Result;
use std::collections::HashSet;

/// Query optimizer that transforms logical plans for better performance.
pub struct Optimizer {
    /// Whether to enable filter pushdown.
    enable_filter_pushdown: bool,
    /// Whether to enable join reordering.
    enable_join_reorder: bool,
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

        Ok(LogicalPlan::new(root))
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
                    proj.input =
                        Box::new(self.try_push_filter_into(predicate, *proj.input));
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

            // Can push through Expand if predicate only uses source variable
            LogicalOperator::Expand(mut expand) => {
                let predicate_vars = self.extract_variables(&predicate);

                // Check if predicate only uses the source variable
                let uses_only_source =
                    predicate_vars.iter().all(|v| v == &expand.from_variable);

                if uses_only_source {
                    // Push the filter before the expand
                    expand.input =
                        Box::new(self.try_push_filter_into(predicate, *expand.input));
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
        self.collect_output_variables_recursive(op, &mut vars);
        vars
    }

    /// Recursively collects output variables from an operator.
    fn collect_output_variables_recursive(&self, op: &LogicalOperator, vars: &mut HashSet<String>) {
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
                self.collect_output_variables_recursive(&expand.input, vars);
            }
            LogicalOperator::Filter(filter) => {
                self.collect_output_variables_recursive(&filter.input, vars);
            }
            LogicalOperator::Project(proj) => {
                for p in &proj.projections {
                    if let Some(alias) = &p.alias {
                        vars.insert(alias.clone());
                    }
                }
                self.collect_output_variables_recursive(&proj.input, vars);
            }
            LogicalOperator::Join(join) => {
                self.collect_output_variables_recursive(&join.left, vars);
                self.collect_output_variables_recursive(&join.right, vars);
            }
            LogicalOperator::Aggregate(agg) => {
                for expr in &agg.group_by {
                    self.collect_variables(expr, vars);
                }
                for agg_expr in &agg.aggregates {
                    if let Some(alias) = &agg_expr.alias {
                        vars.insert(alias.clone());
                    }
                }
            }
            LogicalOperator::Return(ret) => {
                self.collect_output_variables_recursive(&ret.input, vars);
            }
            LogicalOperator::Limit(limit) => {
                self.collect_output_variables_recursive(&limit.input, vars);
            }
            LogicalOperator::Skip(skip) => {
                self.collect_output_variables_recursive(&skip.input, vars);
            }
            LogicalOperator::Sort(sort) => {
                self.collect_output_variables_recursive(&sort.input, vars);
            }
            LogicalOperator::Distinct(distinct) => {
                self.collect_output_variables_recursive(&distinct.input, vars);
            }
            _ => {}
        }
    }

    /// Extracts all variable names referenced in an expression.
    fn extract_variables(&self, expr: &LogicalExpression) -> HashSet<String> {
        let mut vars = HashSet::new();
        self.collect_variables(expr, &mut vars);
        vars
    }

    /// Recursively collects variable names from an expression.
    fn collect_variables(&self, expr: &LogicalExpression, vars: &mut HashSet<String>) {
        match expr {
            LogicalExpression::Variable(name) => {
                vars.insert(name.clone());
            }
            LogicalExpression::Property { variable, .. } => {
                vars.insert(variable.clone());
            }
            LogicalExpression::Binary { left, right, .. } => {
                self.collect_variables(left, vars);
                self.collect_variables(right, vars);
            }
            LogicalExpression::Unary { operand, .. } => {
                self.collect_variables(operand, vars);
            }
            LogicalExpression::FunctionCall { args, .. } => {
                for arg in args {
                    self.collect_variables(arg, vars);
                }
            }
            LogicalExpression::List(items) => {
                for item in items {
                    self.collect_variables(item, vars);
                }
            }
            LogicalExpression::Case {
                operand,
                when_clauses,
                else_clause,
            } => {
                if let Some(op) = operand {
                    self.collect_variables(op, vars);
                }
                for (cond, result) in when_clauses {
                    self.collect_variables(cond, vars);
                    self.collect_variables(result, vars);
                }
                if let Some(else_expr) = else_clause {
                    self.collect_variables(else_expr, vars);
                }
            }
            LogicalExpression::Labels(var)
            | LogicalExpression::Type(var)
            | LogicalExpression::Id(var) => {
                vars.insert(var.clone());
            }
            LogicalExpression::Literal(_) | LogicalExpression::Parameter(_) => {}
        }
    }

    /// Extracts aliases from projection expressions.
    fn extract_projection_aliases(
        &self,
        projections: &[crate::query::plan::Projection],
    ) -> HashSet<String> {
        projections
            .iter()
            .filter_map(|p| p.alias.clone())
            .collect()
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
        BinaryOp, ExpandDirection, ExpandOp, NodeScanOp, ReturnItem, ReturnOp,
    };
    use graphos_common::types::Value;

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
}
