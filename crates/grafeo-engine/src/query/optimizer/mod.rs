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
mod cycles;
pub(crate) use cycles::close_cycles;
pub mod join_order;

pub use cardinality::{
    CardinalityEstimator, ColumnStats, EstimationLog, SelectivityConfig, TableStats,
};
pub use cost::{Cost, CostModel};
pub use join_order::{BitSet, DPccp, JoinGraph, JoinGraphBuilder, JoinPlan};

use crate::query::plan::{
    BinaryOp, ExpandDirection, ExpandOp, FilterOp, JoinCondition, JoinType, LogicalExpression,
    LogicalOperator, LogicalPlan, MultiWayJoinOp, NodeScanOp,
};
use crate::query::planner::lpg::seek;
use grafeo_common::grafeo_debug_span;
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
    /// How cyclic joins of three or more relations are planned.
    cyclic_joins: CyclicJoins,
    /// The node properties with a property index: a filter on one of them
    /// pins a node like `id()` does (see `start_at_the_sought_end`).
    indexed_properties: HashSet<String>,
    /// Whether the store keeps the incoming edges of each node.
    incoming_edges: IncomingEdges,
}

/// Whether a store keeps the incoming edges of each node (its backward
/// adjacency), which an expand that follows edges backward reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IncomingEdges {
    /// Kept with each node, like the outgoing ones.
    Kept,
    /// Not kept: following edges backward is slow.
    NotKept,
}

impl IncomingEdges {
    /// The incoming edges of `store`.
    fn of(store: &dyn grafeo_core::graph::GraphStore) -> Self {
        if store.has_backward_adjacency() {
            Self::Kept
        } else {
            Self::NotKept
        }
    }
}

/// How the optimizer plans a cyclic join of three or more relations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CyclicJoins {
    /// Binary joins in cost-based order. LPG plans: the LPG leapfrog join
    /// returned wrong rows.
    Binary,
    /// One leapfrog `MultiWayJoin`, which the RDF planner executes.
    MultiWay,
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
            cyclic_joins: CyclicJoins::Binary,
            indexed_properties: HashSet::new(),
            incoming_edges: IncomingEdges::Kept,
        }
    }

    /// Creates an optimizer with cardinality estimates from the store's statistics.
    ///
    /// Pre-populates the cardinality estimator with per-label row counts and
    /// edge type fanout. Feeds per-edge-type degree stats, label cardinalities,
    /// and graph totals into the cost model for accurate estimation.
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn from_store(store: &grafeo_core::graph::lpg::LpgStore) -> Self {
        store.ensure_statistics_fresh();
        let stats = store.statistics();
        Self {
            indexed_properties: store.property_index_keys().into_iter().collect(),
            incoming_edges: IncomingEdges::of(store),
            ..Self::from_statistics(&stats)
        }
    }

    /// Creates an optimizer from any GraphStore implementation.
    ///
    /// Unlike [`from_store`](Self::from_store), this does not call
    /// `ensure_statistics_fresh()` since external stores manage their own
    /// statistics. The store's [`statistics()`](grafeo_core::graph::GraphStore::statistics) method
    /// is called directly.
    #[must_use]
    pub fn from_graph_store(store: &dyn grafeo_core::graph::GraphStore) -> Self {
        let stats = store.statistics();
        let indexed_properties = store
            .all_property_keys()
            .into_iter()
            .filter(|key| store.has_property_index(key))
            .collect();
        Self {
            indexed_properties,
            incoming_edges: IncomingEdges::of(store),
            ..Self::from_statistics(&stats)
        }
    }

    /// Creates an optimizer from RDF statistics.
    ///
    /// Uses triple pattern cardinality estimates for cost-based optimization
    /// of SPARQL queries. Maps total triples to graph totals for the cost model.
    #[cfg(feature = "triple-store")]
    #[must_use]
    pub fn from_rdf_statistics(rdf_stats: grafeo_core::statistics::RdfStatistics) -> Self {
        let total = rdf_stats.total_triples;
        let estimator = CardinalityEstimator::from_rdf_statistics(rdf_stats);
        Self {
            enable_filter_pushdown: true,
            enable_join_reorder: true,
            enable_projection_pushdown: true,
            cost_model: CostModel::new().with_graph_totals(total, total),
            card_estimator: estimator,
            cyclic_joins: CyclicJoins::MultiWay,
            indexed_properties: HashSet::new(),
            incoming_edges: IncomingEdges::NotKept,
        }
    }

    /// Creates an optimizer from a Statistics snapshot.
    ///
    /// Extracts label cardinalities, edge type degrees, and graph totals
    /// for both the cardinality estimator and cost model.
    #[must_use]
    fn from_statistics(stats: &grafeo_core::statistics::Statistics) -> Self {
        let estimator = CardinalityEstimator::from_statistics(stats);

        let avg_fanout = if stats.total_nodes > 0 {
            (stats.total_edges as f64 / stats.total_nodes as f64).max(1.0)
        } else {
            10.0
        };

        let edge_type_degrees: std::collections::HashMap<String, (f64, f64)> = stats
            .edge_types
            .iter()
            .map(|(name, et)| (name.clone(), (et.avg_out_degree, et.avg_in_degree)))
            .collect();

        let label_cardinalities: std::collections::HashMap<String, u64> = stats
            .labels
            .iter()
            .map(|(name, ls)| (name.clone(), ls.node_count))
            .collect();

        Self {
            enable_filter_pushdown: true,
            enable_join_reorder: true,
            enable_projection_pushdown: true,
            cost_model: CostModel::new()
                .with_avg_fanout(avg_fanout)
                .with_edge_type_degrees(edge_type_degrees)
                .with_label_cardinalities(label_cardinalities)
                .with_graph_totals(stats.total_nodes, stats.total_edges),
            card_estimator: estimator,
            cyclic_joins: CyclicJoins::Binary,
            indexed_properties: HashSet::new(),
            incoming_edges: IncomingEdges::Kept,
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

    /// Sets the selectivity configuration for the cardinality estimator.
    pub fn with_selectivity_config(mut self, config: SelectivityConfig) -> Self {
        self.card_estimator = CardinalityEstimator::with_selectivity_config(config);
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

    /// Estimates the total cost of a plan by recursively costing the entire tree.
    ///
    /// Walks every operator in the plan, computing cardinality at each level
    /// and summing the per-operator costs. Uses actual child cardinalities
    /// for join cost estimation rather than approximations.
    pub fn estimate_cost(&self, plan: &LogicalPlan) -> Cost {
        self.cost_model
            .estimate_tree(&plan.root, &self.card_estimator)
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
        let _span = grafeo_debug_span!("grafeo::query::optimize");
        // Correctness first: a pattern through a node or edge that was bound
        // before must be checked.
        let mut root = cycles::close_cycles(plan.root);

        // Apply optimization rules
        if self.enable_filter_pushdown {
            // Propagate filters across LeftJoin shared variables BEFORE
            // pushdown, so OPTIONAL MATCH right-side subtrees pick up
            // the same WHERE constraints the main MATCH applies.
            // Otherwise pushdown rewrites the left subtree and the
            // chance is gone.
            root = self.propagate_join_predicates(root);
            root = self.push_filters_down(root);
            // With each filter right above what it reads, an expand whose
            // target a filter pins starts there.
            root = self.start_at_the_sought_end(root, false);
        }

        if self.enable_join_reorder {
            root = self.reorder_joins(root);
        }

        if self.enable_projection_pushdown {
            root = self.push_projections_down(root);
        }

        Ok(LogicalPlan {
            root,
            explain: plan.explain,
            profile: plan.profile,
            default_params: plan.default_params,
        })
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
            LogicalOperator::MultiWayJoin(mwj) => {
                for cond in &mwj.conditions {
                    Self::collect_from_expression(&cond.left, required);
                    Self::collect_from_expression(&cond.right, required);
                }
                for input in &mwj.inputs {
                    Self::collect_required_recursive(input, required);
                }
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
            LogicalExpression::MapAccess { base, .. } => {
                Self::collect_from_expression(base, required);
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
            LogicalOperator::MapCollect(mut mc) => {
                mc.input = Box::new(self.push_projections_recursive(*mc.input, required));
                LogicalOperator::MapCollect(mc)
            }
            LogicalOperator::MultiWayJoin(mut mwj) => {
                mwj.inputs = mwj
                    .inputs
                    .into_iter()
                    .map(|input| self.push_projections_recursive(input, required))
                    .collect();
                LogicalOperator::MultiWayJoin(mwj)
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
        if let Some((relations, conditions)) = self.extract_join_tree(&op)
            && relations.len() >= 2
            && let Some(optimized) = self.optimize_join_order(&relations, &conditions)
        {
            return optimized;
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
            LogicalOperator::MapCollect(mut mc) => {
                mc.input = Box::new(self.reorder_joins(*mc.input));
                LogicalOperator::MapCollect(mc)
            }
            LogicalOperator::MultiWayJoin(mut mwj) => {
                mwj.inputs = mwj
                    .inputs
                    .into_iter()
                    .map(|input| self.reorder_joins(input))
                    .collect();
                LogicalOperator::MultiWayJoin(mwj)
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
            // A filter on a base relation is part of that relation: the
            // reordered plan is built from the relations, so the filter goes
            // with it. A filter over a join reads more than one relation and
            // no relation can take it: the tree is not reordered then.
            LogicalOperator::Filter(filter) => {
                let mut below = Vec::new();
                let mut below_conditions = Vec::new();
                if !self.collect_join_tree(&filter.input, &mut below, &mut below_conditions) {
                    return false;
                }
                match below.pop() {
                    Some((name, _)) if below.is_empty() && below_conditions.is_empty() => {
                        relations.push((name, op.clone()));
                        true
                    }
                    _ => false,
                }
            }
            LogicalOperator::Expand(expand) => {
                // Expand is a special case - it's like a join with the adjacency
                // For now, treat the whole Expand subtree as a single relation
                relations.push((expand.to_variable.clone(), op.clone()));
                true
            }
            #[cfg(feature = "triple-store")]
            LogicalOperator::TripleScan(scan) => {
                // Use the first variable found as the relation name.
                // For all-constant patterns, generate a unique fallback to
                // avoid HashMap key collisions in `add_relation`.
                let name = scan
                    .subject
                    .as_variable()
                    .or_else(|| scan.predicate.as_variable())
                    .or_else(|| scan.object.as_variable())
                    .map_or_else(|| format!("__tp_{}", relations.len()), String::from);
                relations.push((name, op.clone()));
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

    /// Optimizes the join order using DPccp, or produces a multi-way
    /// leapfrog join for cyclic patterns where the planner supports one (see
    /// [`CyclicJoins`]).
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

        // For cyclic graphs with 3+ relations, use leapfrog (WCOJ) join where
        // the planner has a correct one (RDF). The LPG leapfrog join
        // intersected only the first shared variable, dropped pushed-down
        // filters and ordered its variables nondeterministically, so it
        // returned wrong rows; LPG plans keep binary joins until it holds
        // every join condition.
        if self.cyclic_joins == CyclicJoins::MultiWay && graph.is_cyclic() && relations.len() >= 3 {
            // Collect shared variables (variables appearing in 2+ conditions)
            let mut var_counts: std::collections::HashMap<&str, usize> =
                std::collections::HashMap::new();
            for cond in conditions {
                *var_counts.entry(&cond.left_var).or_default() += 1;
                *var_counts.entry(&cond.right_var).or_default() += 1;
            }
            let shared_variables: Vec<String> = var_counts
                .into_iter()
                .filter(|(_, count)| *count >= 2)
                .map(|(var, _)| var.to_string())
                .collect();

            let join_conditions: Vec<JoinCondition> = conditions
                .iter()
                .map(|c| JoinCondition {
                    left: c.left_expr.clone(),
                    right: c.right_expr.clone(),
                })
                .collect();

            return Some(LogicalOperator::MultiWayJoin(MultiWayJoinOp {
                inputs: relations.iter().map(|(_, rel)| rel.clone()).collect(),
                conditions: join_conditions,
                shared_variables,
            }));
        }

        // Fall through to DPccp for binary join ordering
        let mut dpccp = DPccp::new(&graph, &self.cost_model, &self.card_estimator);
        let plan = dpccp.optimize()?;

        Some(plan.operator)
    }

    /// Propagates filter predicates across LeftJoin boundaries when the
    /// predicate references variables bound on both sides of the join.
    ///
    /// OPTIONAL MATCH compiles to `LeftJoin(left, right)` where the right
    /// subtree binds the same variable as the left (typically via its own
    /// NodeScan), so the right side independently scans the full table even
    /// when the left side is bound to a tiny set. After regular pushdown
    /// rewrites `Filter(P, MainMatch)` into something like
    /// `Expand(NodeList(P))`, the original filter is gone and we can no
    /// longer detect the constraint to mirror to the right. This pass runs
    /// BEFORE pushdown: for every `LeftJoin(Filter(P, X), Y)` where P's
    /// variables are bound in both X and Y, wrap Y with the same Filter.
    ///
    /// Safe by LeftJoin semantics: matched pairs satisfy left.v = right.v
    /// for shared variable v, so a right row that fails P also fails P
    /// when joined to its matching left row, which already failed P. And
    /// OPTIONAL left rows with no right match are unaffected, since we
    /// only constrain the right side's enumeration.
    fn propagate_join_predicates(&self, op: LogicalOperator) -> LogicalOperator {
        // Recurse into children first (post-order). After children are
        // processed, propagate at this level if it's a LeftJoin.
        let op = op.map_children(|child| self.propagate_join_predicates(child));

        let LogicalOperator::LeftJoin(mut left_join) = op else {
            return op;
        };

        // Variables bound on both sides act as the implicit join keys
        // (typically a single shared variable like `c`). Predicates that
        // reference only these variables are safe to mirror to the right.
        // The columns each side holds (see `bound_columns`): a variable a
        // `WITH` drops is not one, whatever its name; when a side's columns
        // are not known, nothing is mirrored.
        let shared_vars: HashSet<String> = match bound_columns(&left_join.left)
            .zip(bound_columns(&left_join.right))
        {
            Some((left_vars, right_vars)) => left_vars.intersection(&right_vars).cloned().collect(),
            None => HashSet::new(),
        };

        if shared_vars.is_empty() {
            return LogicalOperator::LeftJoin(left_join);
        }

        let mut shared_filters = Vec::new();
        self.collect_shared_var_filters(&left_join.left, &shared_vars, &mut shared_filters);
        for predicate in shared_filters {
            left_join.right = Box::new(LogicalOperator::Filter(FilterOp {
                predicate,
                pushdown_hint: None,
                input: left_join.right,
            }));
        }

        LogicalOperator::LeftJoin(left_join)
    }

    /// Walks a (sub)tree collecting filter predicates that reference only
    /// the given shared variables. These are safe to mirror across a
    /// LeftJoin to the right subtree.
    fn collect_shared_var_filters(
        &self,
        op: &LogicalOperator,
        shared_vars: &HashSet<String>,
        out: &mut Vec<LogicalExpression>,
    ) {
        match op {
            LogicalOperator::Filter(f) => {
                let predicate_vars = self.extract_variables(&f.predicate);
                if !predicate_vars.is_empty()
                    && predicate_vars.iter().all(|v| shared_vars.contains(v))
                {
                    out.push(f.predicate.clone());
                }
                self.collect_shared_var_filters(&f.input, shared_vars, out);
            }
            // Don't descend through operators that change row semantics
            // (Aggregate, Limit, Skip, Sort, Distinct): predicates above
            // them aren't safe to extract because they apply to a
            // post-aggregation/limit world. Project, Return, and Expand
            // are row-preserving so we can keep walking; below a
            // projection, a shared name is the same binding only when the
            // projection passes it on unchanged (`WITH c AS a` makes a new
            // `a`).
            LogicalOperator::Project(p) => {
                let items = p
                    .projections
                    .iter()
                    .map(|projection| (&projection.expression, projection.alias.as_deref()));
                let kept = passed_on(shared_vars, items, p.pass_through_input);
                if !kept.is_empty() {
                    self.collect_shared_var_filters(&p.input, &kept, out);
                }
            }
            LogicalOperator::Return(r) => {
                let items = r
                    .items
                    .iter()
                    .map(|item| (&item.expression, item.alias.as_deref()));
                let kept = passed_on(shared_vars, items, false);
                if !kept.is_empty() {
                    self.collect_shared_var_filters(&r.input, &kept, out);
                }
            }
            LogicalOperator::Expand(e) => {
                self.collect_shared_var_filters(&e.input, shared_vars, out);
            }
            // For nested LeftJoins (chained OPTIONAL MATCH), the LEFT side
            // is the row-driving subtree: its surviving rows feed the outer
            // join. Filters anywhere in that left subtree are implicitly
            // applied to every output row, so they're safe to mirror across
            // the outer join. The right side is optional and may contribute
            // NULL-padded rows that don't satisfy those filters, so we
            // deliberately don't walk into it.
            LogicalOperator::LeftJoin(j) => {
                self.collect_shared_var_filters(&j.left, shared_vars, out);
            }
            LogicalOperator::Join(j) => {
                self.collect_shared_var_filters(&j.left, shared_vars, out);
                self.collect_shared_var_filters(&j.right, shared_vars, out);
            }
            // Stop at Apply, Aggregate, Limit, Skip, etc.: too risky to
            // pull predicates across those boundaries.
            _ => {}
        }
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
            // The input of a later `MATCH`'s scan holds the earlier clauses,
            // with their filters.
            LogicalOperator::NodeScan(mut scan) => {
                scan.input = scan
                    .input
                    .map(|input| Box::new(self.push_filters_down(*input)));
                LogicalOperator::NodeScan(scan)
            }
            LogicalOperator::Join(mut join) => {
                join.left = Box::new(self.push_filters_down(*join.left));
                join.right = Box::new(self.push_filters_down(*join.right));
                LogicalOperator::Join(join)
            }
            LogicalOperator::LeftJoin(mut left_join) => {
                left_join.left = Box::new(self.push_filters_down(*left_join.left));
                left_join.right = Box::new(self.push_filters_down(*left_join.right));
                LogicalOperator::LeftJoin(left_join)
            }
            LogicalOperator::AntiJoin(mut anti_join) => {
                anti_join.left = Box::new(self.push_filters_down(*anti_join.left));
                anti_join.right = Box::new(self.push_filters_down(*anti_join.right));
                LogicalOperator::AntiJoin(anti_join)
            }
            LogicalOperator::Apply(mut apply) => {
                apply.input = Box::new(self.push_filters_down(*apply.input));
                apply.subplan = Box::new(self.push_filters_down(*apply.subplan));
                LogicalOperator::Apply(apply)
            }
            LogicalOperator::Union(mut union) => {
                union.inputs = union
                    .inputs
                    .into_iter()
                    .map(|input| self.push_filters_down(input))
                    .collect();
                LogicalOperator::Union(union)
            }
            LogicalOperator::Unwind(mut unwind) => {
                unwind.input = Box::new(self.push_filters_down(*unwind.input));
                LogicalOperator::Unwind(unwind)
            }
            LogicalOperator::Aggregate(mut agg) => {
                agg.input = Box::new(self.push_filters_down(*agg.input));
                LogicalOperator::Aggregate(agg)
            }
            LogicalOperator::MapCollect(mut mc) => {
                mc.input = Box::new(self.push_filters_down(*mc.input));
                LogicalOperator::MapCollect(mc)
            }
            LogicalOperator::MultiWayJoin(mut mwj) => {
                mwj.inputs = mwj
                    .inputs
                    .into_iter()
                    .map(|input| self.push_filters_down(input))
                    .collect();
                LogicalOperator::MultiWayJoin(mwj)
            }
            // The filters inside the input of any other operator move down
            // within it: below a write (`... WHERE id(s) = row.src AND
            // id(d) = row.dst MERGE (s)-[:T]->(d)`) as anywhere else.
            other => other.map_children(|child| self.push_filters_down(child)),
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
        // Subquery and volatile conjuncts stay above `op`, where they were
        // written; only the others move down.
        let (pinned, movable): (Vec<_>, Vec<_>) =
            conjuncts(predicate).into_iter().partition(stays_in_place);
        let predicate = match (conjunction(pinned), conjunction(movable)) {
            (Some(pinned), movable) => {
                let input = match movable {
                    Some(movable) => self.try_push_filter_into(movable, op),
                    None => op,
                };
                return LogicalOperator::Filter(FilterOp {
                    predicate: pinned,
                    pushdown_hint: None,
                    input: Box::new(input),
                });
            }
            (None, Some(movable)) => movable,
            (None, None) => return op,
        };
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
                        pushdown_hint: None,
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
                // - The path alias (if any), with the columns `length(p)`,
                //   `nodes(p)` and `edges(p)` read
                let path_columns: Vec<String> = expand
                    .path_alias
                    .iter()
                    .flat_map(|path| path_columns(path))
                    .collect();
                let mut introduced_vars = vec![&expand.to_variable];
                if let Some(ref edge_var) = expand.edge_variable {
                    introduced_vars.push(edge_var);
                }
                if let Some(ref path_alias) = expand.path_alias {
                    introduced_vars.push(path_alias);
                }
                introduced_vars.extend(&path_columns);

                // Check if predicate uses any variables introduced by this expand
                let uses_introduced_vars =
                    predicate_vars.iter().any(|v| introduced_vars.contains(&v));

                if !uses_introduced_vars {
                    // Predicate doesn't use vars from this expand, so push through
                    expand.input = Box::new(self.try_push_filter_into(predicate, *expand.input));
                    return LogicalOperator::Expand(expand);
                }

                // Push the conjuncts that don't use the expand's variables
                // below it and keep the others after it: in
                // `id(s) = $s AND id(d) = $d` the first pins `s` before the
                // expand walks its edges.
                let introduced: HashSet<String> = introduced_vars.into_iter().cloned().collect();
                let (below, after): (Vec<_>, Vec<_>) =
                    conjuncts(predicate).into_iter().partition(|conjunct| {
                        self.extract_variables(conjunct).is_disjoint(&introduced)
                    });
                if let Some(below) = conjunction(below) {
                    expand.input = Box::new(self.try_push_filter_into(below, *expand.input));
                }
                let expand = LogicalOperator::Expand(expand);
                match conjunction(after) {
                    Some(predicate) => LogicalOperator::Filter(FilterOp {
                        predicate,
                        pushdown_hint: None,
                        input: Box::new(expand),
                    }),
                    None => expand,
                }
            }

            // A side of a join gets a conjunct only when it binds every
            // variable the conjunct reads (see `bound_columns`; a side whose
            // columns are not known gets nothing). A variable both sides bind
            // must be one the join equates, or the side's value may not be
            // the row's. With every variable equated, an inner join filters
            // both sides. The other conjuncts stay above the join.
            LogicalOperator::Join(mut join) => {
                let sides = bound_columns(&join.left).zip(bound_columns(&join.right));
                let equated = |var: &String| {
                    join.join_type != JoinType::Cross && join.conditions.iter().any(|condition| {
                        matches!(
                            (&condition.left, &condition.right),
                            (LogicalExpression::Variable(left), LogicalExpression::Variable(right))
                                if left == var && right == var
                        )
                    })
                };
                let (mut to_left, mut to_right, mut above) = (Vec::new(), Vec::new(), Vec::new());
                for conjunct in conjuncts(predicate) {
                    let vars = self.extract_variables(&conjunct);
                    let (in_left, in_right) =
                        sides
                            .as_ref()
                            .map_or((false, false), |(left_vars, right_vars)| {
                                let shared_equated = vars
                                    .iter()
                                    .filter(|v| left_vars.contains(*v) && right_vars.contains(*v))
                                    .all(equated);
                                (
                                    shared_equated
                                        && vars.is_subset(left_vars)
                                        && matches!(
                                            join.join_type,
                                            JoinType::Inner
                                                | JoinType::Cross
                                                | JoinType::Left
                                                | JoinType::Semi
                                                | JoinType::Anti
                                        ),
                                    shared_equated
                                        && vars.is_subset(right_vars)
                                        && matches!(
                                            join.join_type,
                                            JoinType::Inner | JoinType::Cross
                                        ),
                                )
                            });
                    match (in_left, in_right) {
                        (true, true) if !vars.is_empty() => {
                            to_left.push(conjunct.clone());
                            to_right.push(conjunct);
                        }
                        (true, _) => to_left.push(conjunct),
                        (false, true) => to_right.push(conjunct),
                        (false, false) => above.push(conjunct),
                    }
                }
                if let Some(predicate) = conjunction(to_left) {
                    join.left = Box::new(self.try_push_filter_into(predicate, *join.left));
                }
                if let Some(predicate) = conjunction(to_right) {
                    join.right = Box::new(self.try_push_filter_into(predicate, *join.right));
                }
                let join = LogicalOperator::Join(join);
                match conjunction(above) {
                    Some(predicate) => LogicalOperator::Filter(FilterOp {
                        predicate,
                        pushdown_hint: None,
                        input: Box::new(join),
                    }),
                    None => join,
                }
            }

            // LeftJoin pushdown is semantics-preserving only on the LEFT
            // side: anything that filters out a left row also filters out
            // every (left, NULL) pair the join would have emitted. Pushing
            // to the right side is unsafe because OPTIONAL MATCH must keep
            // left rows that have no right match. Each conjunct is routed on
            // its own. One that reads only columns the left side binds (see
            // `bound_columns`) filters the left side: a row of the join holds
            // the left row's values for them, a name both sides bind being a
            // join key. The other conjuncts stay above the left join.
            LogicalOperator::LeftJoin(mut left_join) => {
                let sides = bound_columns(&left_join.left).zip(bound_columns(&left_join.right));
                let (mut to_left, mut to_right, mut above) = (Vec::new(), Vec::new(), Vec::new());
                for conjunct in conjuncts(predicate) {
                    let vars = self.extract_variables(&conjunct);
                    let (in_left, in_right) =
                        sides
                            .as_ref()
                            .map_or((false, false), |(left_vars, right_vars)| {
                                (vars.is_subset(left_vars), vars.is_subset(right_vars))
                            });
                    match (in_left, in_right) {
                        // A conjunct on join-key variables alone also
                        // filters the right side. The OPTIONAL MATCH
                        // compiles to a LeftJoin where the right subtree
                        // independently re-binds the shared variable
                        // (typically via its own NodeScan), so the right
                        // side can balloon to the full table even when the
                        // left side is bound to a tiny set. Duplicating the
                        // conjunct to both sides is safe: matched pairs
                        // satisfy left.x == right.x, so a right row that
                        // fails it either has no left match or pairs with a
                        // left row that also fails. Unmatched (OPTIONAL)
                        // left rows are unaffected. This is what collapses
                        // hydrate's three independent 30k-row scans into
                        // 6-id index lookups.
                        (true, true) if !vars.is_empty() => {
                            to_left.push(conjunct.clone());
                            to_right.push(conjunct);
                        }
                        (true, _) => to_left.push(conjunct),
                        (false, _) => above.push(conjunct),
                    }
                }
                if let Some(predicate) = conjunction(to_left) {
                    left_join.left =
                        Box::new(self.try_push_filter_into(predicate, *left_join.left));
                }
                if let Some(predicate) = conjunction(to_right) {
                    left_join.right =
                        Box::new(self.try_push_filter_into(predicate, *left_join.right));
                }
                let left_join = LogicalOperator::LeftJoin(left_join);
                match conjunction(above) {
                    Some(predicate) => LogicalOperator::Filter(FilterOp {
                        predicate,
                        pushdown_hint: None,
                        input: Box::new(left_join),
                    }),
                    None => left_join,
                }
            }

            // Apply (correlated subquery): the input is the outer plan, the
            // subplan re-evaluates per outer row. A predicate on variables
            // the input binds, none of which the subplan returns, can push
            // into the input; not when the subplan writes, which it does
            // once per input row.
            LogicalOperator::Apply(mut apply) => {
                let predicate_vars = self.extract_variables(&predicate);
                let into_input = !apply.subplan.has_mutations()
                    && bound_columns(&apply.input)
                        .zip(bound_columns(&apply.subplan))
                        .is_some_and(|(input_vars, returned)| {
                            predicate_vars.is_subset(&input_vars)
                                && predicate_vars.is_disjoint(&returned)
                        });

                if into_input {
                    apply.input = Box::new(self.try_push_filter_into(predicate, *apply.input));
                    LogicalOperator::Apply(apply)
                } else {
                    LogicalOperator::Filter(FilterOp {
                        predicate,
                        pushdown_hint: None,
                        input: Box::new(LogicalOperator::Apply(apply)),
                    })
                }
            }

            // Cannot push through Aggregate (predicate refers to aggregated values)
            LogicalOperator::Aggregate(agg) => LogicalOperator::Filter(FilterOp {
                predicate,
                pushdown_hint: None,
                input: Box::new(LogicalOperator::Aggregate(agg)),
            }),

            // A scan without input is the bottom: the filter stays on top. A
            // scan with input (a later `MATCH` without a shared variable)
            // scans once per input row: the conjuncts that read only what
            // the input binds (see `bound_columns`) filter its rows first, and
            // may go further down (onto an earlier scan, below an expand).
            // The others stay above the scan.
            LogicalOperator::NodeScan(mut scan) => {
                let (below, above) = match scan.input.as_deref().and_then(bound_columns) {
                    Some(input_vars) => conjuncts(predicate).into_iter().partition(|conjunct| {
                        let vars = self.extract_variables(conjunct);
                        !vars.contains(&scan.variable) && vars.is_subset(&input_vars)
                    }),
                    None => (Vec::new(), vec![predicate]),
                };
                if let Some(below) = conjunction(below)
                    && let Some(input) = scan.input.take()
                {
                    scan.input = Some(Box::new(self.try_push_filter_into(below, *input)));
                }
                let scan = LogicalOperator::NodeScan(scan);
                match conjunction(above) {
                    Some(predicate) => LogicalOperator::Filter(FilterOp {
                        predicate,
                        pushdown_hint: None,
                        input: Box::new(scan),
                    }),
                    None => scan,
                }
            }

            // Filters commute (boolean conjunction is associative/commutative),
            // so we can push the outer predicate past an existing inner
            // Filter and continue trying to push further down. Without this
            // case, a pattern like
            // `Filter(r.id IN [...], Filter(hasLabel(d), Expand))` (emitted
            // by Cypher when the WHERE clause is conjoined with automatic
            // label filters) gets stuck at the top, missing the chance to
            // anchor the filter on r's NodeScan and trigger the property-
            // index fast path.
            //
            // Only commute when the outer predicate references variables
            // that are bound at or below the inner filter's input. If the
            // inner filter is a final scope-narrowing step that introduces
            // synthetic columns the outer predicate depends on (e.g. path
            // variables produced by a variable-length expand wrapped in a
            // label filter), pushing past it would evaluate the outer
            // predicate against rows where those columns aren't bound and
            // every row would be filtered out.
            LogicalOperator::Filter(inner_filter) => {
                let predicate_vars = self.extract_variables(&predicate);
                let inner_input_vars = self.collect_output_variables(&inner_filter.input);
                let safe_to_commute = predicate_vars.iter().all(|v| inner_input_vars.contains(v));
                if safe_to_commute {
                    let mut inner_filter = inner_filter;
                    inner_filter.input =
                        Box::new(self.try_push_filter_into(predicate, *inner_filter.input));
                    LogicalOperator::Filter(inner_filter)
                } else {
                    LogicalOperator::Filter(FilterOp {
                        predicate,
                        pushdown_hint: None,
                        input: Box::new(LogicalOperator::Filter(inner_filter)),
                    })
                }
            }

            // For other operators, keep filter on top
            other => LogicalOperator::Filter(FilterOp {
                predicate,
                pushdown_hint: None,
                input: Box::new(other),
            }),
        }
    }

    /// Starts each single-hop expand whose target a filter pins by a key,
    /// from a source nothing pins, at that target (see
    /// [`Self::turn_at_the_sought_end`]): `MATCH (src)-[r]->(tgt) WHERE
    /// id(tgt) IN $ids` then seeks the few targets and follows their
    /// incoming edges, instead of expanding every edge of every node. The
    /// turned expand binds the same rows, with its columns in another order,
    /// so it is made only where the order of the columns is `free`: below an
    /// operator that names the columns it passes on (a `RETURN` without `*`,
    /// a `WITH`, an aggregation). A `RETURN *` reads them in their order, and
    /// so do the set operations that line their inputs up.
    fn start_at_the_sought_end(&self, op: LogicalOperator, free: bool) -> LogicalOperator {
        match op {
            LogicalOperator::Filter(_) => self.turn_at_the_sought_end(op, free),
            LogicalOperator::Return(mut ret) => {
                let star = ret.items.iter().any(
                    |item| matches!(&item.expression, LogicalExpression::Variable(name) if name == "*"),
                );
                ret.input = Box::new(self.start_at_the_sought_end(*ret.input, !star));
                LogicalOperator::Return(ret)
            }
            LogicalOperator::Project(mut project) => {
                let free = free || !project.pass_through_input;
                project.input = Box::new(self.start_at_the_sought_end(*project.input, free));
                LogicalOperator::Project(project)
            }
            LogicalOperator::Aggregate(mut aggregate) => {
                aggregate.input = Box::new(self.start_at_the_sought_end(*aggregate.input, true));
                LogicalOperator::Aggregate(aggregate)
            }
            op @ (LogicalOperator::Union(_)
            | LogicalOperator::Except(_)
            | LogicalOperator::Intersect(_)
            | LogicalOperator::Otherwise(_)) => {
                op.map_children(|child| self.start_at_the_sought_end(child, false))
            }
            other => other.map_children(|child| self.start_at_the_sought_end(child, free)),
        }
    }

    /// `Filter(P, Expand(src -> tgt, checks of src over NodeScan(src)))`
    /// turned around, when the order of the columns is `free` and
    /// [`Self::turned_direction`] allows it: the conjuncts of `P` on `tgt`
    /// alone (with the values of the scan's input rows) filter a scan of
    /// `tgt`, which the planner turns into a seek; the expand follows the
    /// edges from `tgt` back to `src`; the checks of `src`, then the other
    /// conjuncts of `P`, filter its rows. Each row has the same values as
    /// before: the expand matches the same edges from either end, a
    /// self-loop as often from each. Otherwise `op` stays as it is, and its
    /// input goes on to [`Self::start_at_the_sought_end`].
    fn turn_at_the_sought_end(&self, op: LogicalOperator, free: bool) -> LogicalOperator {
        let unturned = |op: LogicalOperator| {
            op.map_children(|child| self.start_at_the_sought_end(child, free))
        };
        if !free {
            return unturned(op);
        }
        let (filters, below) = peel_filters(op);
        let LogicalOperator::Expand(mut expand) = below else {
            return unturned(wrap_filters(filters, below));
        };
        let (checks, source) = peel_filters(std::mem::replace(
            expand.input.as_mut(),
            LogicalOperator::Empty,
        ));
        let LogicalOperator::NodeScan(scan) = source else {
            *expand.input = wrap_filters(checks, source);
            return unturned(wrap_filters(filters, LogicalOperator::Expand(expand)));
        };
        // The conjuncts of the filters and of the checks, each in the order
        // they run: from the bottom filter up.
        let in_order = |filters: &[FilterOp]| -> Vec<LogicalExpression> {
            filters
                .iter()
                .rev()
                .flat_map(|filter| conjuncts(filter.predicate.clone()))
                .collect()
        };
        let (written, checked) = (in_order(&filters), in_order(&checks));
        let Some((direction, input_vars)) =
            self.turned_direction(&written, &checked, &expand, &scan)
        else {
            *expand.input = wrap_filters(checks, LogicalOperator::NodeScan(scan));
            return unturned(wrap_filters(filters, LogicalOperator::Expand(expand)));
        };

        let (source, target) = (expand.from_variable, expand.to_variable);
        let (on_target, rest): (Vec<_>, Vec<_>) = written.into_iter().partition(|conjunct| {
            movable_variables(conjunct).is_some_and(|vars| {
                vars.contains(&target)
                    && vars
                        .iter()
                        .all(|var| *var == target || input_vars.contains(var))
            })
        });
        let sought = LogicalOperator::NodeScan(NodeScanOp {
            variable: target.clone(),
            label: None,
            input: scan
                .input
                .map(|input| Box::new(self.start_at_the_sought_end(*input, true))),
        });
        let sought = match conjunction(on_target) {
            Some(predicate) => LogicalOperator::Filter(FilterOp {
                predicate,
                pushdown_hint: None,
                input: Box::new(sought),
            }),
            None => sought,
        };
        let turned = LogicalOperator::Expand(ExpandOp {
            from_variable: target,
            to_variable: source,
            edge_variable: expand.edge_variable,
            direction,
            edge_types: expand.edge_types,
            min_hops: 1,
            max_hops: Some(1),
            input: Box::new(sought),
            path_alias: None,
            path_mode: expand.path_mode,
            quantified: false,
        });
        match conjunction(checked.into_iter().chain(rest).collect()) {
            Some(predicate) => LogicalOperator::Filter(FilterOp {
                predicate,
                pushdown_hint: None,
                input: Box::new(turned),
            }),
            None => turned,
        }
    }

    /// The direction of `expand` started at its target, and the variables
    /// of the rows its source's scan runs for, when turning it seeks the
    /// target and keeps every row: `written` (the conjuncts of the filters
    /// above it) pins the target by a key (see [`seek::pins`]) and
    /// `checked` (the conjuncts of the checks between it and the scan of
    /// its source) does not pin the source, which the planner would seek
    /// then.
    ///
    /// The expand is a single hop without a named path (a path, or the list
    /// of a quantified edge, would come out reversed). Its source is a scan
    /// without a label: a label scan finds the nodes that have the label
    /// now, while a check would read the labels of a past epoch (see
    /// `may_choose_scan_label`). The checks move above the turned expand,
    /// so they read only the source and the scan's input, and may move (see
    /// [`movable_variables`]). The scan's input, whose rows the turned plan
    /// scans the target for, binds none of the expand's variables and does
    /// not write: a seek would not see what it writes for later rows.
    /// Starting an outgoing expand at its target follows incoming edges,
    /// which needs the store's backward edges.
    fn turned_direction(
        &self,
        written: &[LogicalExpression],
        checked: &[LogicalExpression],
        expand: &ExpandOp,
        scan: &NodeScanOp,
    ) -> Option<(ExpandDirection, HashSet<String>)> {
        let (source, target) = (&expand.from_variable, &expand.to_variable);
        if expand.is_variable_length()
            || expand.path_alias.is_some()
            || source == target
            || scan.variable != *source
            || scan.label.is_some()
        {
            return None;
        }
        let direction = match expand.direction {
            ExpandDirection::Outgoing if self.incoming_edges == IncomingEdges::Kept => {
                ExpandDirection::Incoming
            }
            ExpandDirection::Outgoing => return None,
            ExpandDirection::Incoming => ExpandDirection::Outgoing,
            ExpandDirection::Both => ExpandDirection::Both,
        };
        let input_vars = match scan.input.as_deref() {
            Some(input) if input.has_mutations() => return None,
            Some(input) => bound_columns(input)?,
            None => HashSet::new(),
        };
        if [source, target]
            .into_iter()
            .chain(&expand.edge_variable)
            .any(|var| input_vars.contains(var))
        {
            return None;
        }
        let has_index = |property: &str| self.indexed_properties.contains(property);
        let checks_hold_after = checked.iter().all(|check| {
            movable_variables(check).is_some_and(|vars| {
                vars.iter()
                    .all(|var| var == source || input_vars.contains(var))
            }) && !seek::pins(check, source, has_index, &input_vars)
        });
        let sought = written
            .iter()
            .any(|conjunct| seek::pins(conjunct, target, has_index, &input_vars));
        (checks_hold_after && sought).then_some((direction, input_vars))
    }

    // NOTE: Filter-into-TripleScan pushdown is intentionally not implemented.
    // Binding a variable position in the scan removes it from output columns,
    // which breaks downstream operators (SELECT, ORDER BY) that reference it.
    // A correct implementation needs a separate `pushed_bindings` field on
    // TripleScanOp so the variable stays in output while the scan is constrained.

    /// Collects all output variable names from an operator.
    fn collect_output_variables(&self, op: &LogicalOperator) -> HashSet<String> {
        let mut vars = HashSet::new();
        Self::collect_output_variables_recursive(op, &mut vars);
        vars
    }

    /// Recursively collects output variables from an operator.
    fn collect_output_variables_recursive(op: &LogicalOperator, vars: &mut HashSet<String>) {
        match op {
            // A scan with input passes the input's columns on.
            LogicalOperator::NodeScan(scan) => {
                vars.insert(scan.variable.clone());
                if let Some(input) = &scan.input {
                    Self::collect_output_variables_recursive(input, vars);
                }
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
            #[cfg(feature = "triple-store")]
            LogicalOperator::PropertyPath(path) => {
                for component in [Some(&path.subject), Some(&path.object), path.graph.as_ref()]
                    .into_iter()
                    .flatten()
                {
                    if let Some(v) = component.as_variable() {
                        vars.insert(v.to_string());
                    }
                }
            }
            #[cfg(feature = "triple-store")]
            LogicalOperator::TripleScan(scan) => {
                if let Some(v) = scan.subject.as_variable() {
                    vars.insert(v.to_string());
                }
                if let Some(v) = scan.predicate.as_variable() {
                    vars.insert(v.to_string());
                }
                if let Some(v) = scan.object.as_variable() {
                    vars.insert(v.to_string());
                }
                if let Some(ref g) = scan.graph
                    && let Some(v) = g.as_variable()
                {
                    vars.insert(v.to_string());
                }
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
            LogicalExpression::MapAccess { base, .. } => Self::collect_variables(base, vars),
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
            LogicalExpression::ListPredicate {
                list_expr,
                predicate,
                ..
            } => {
                Self::collect_variables(list_expr, vars);
                Self::collect_variables(predicate, vars);
            }
            LogicalExpression::ExistsSubquery(_)
            | LogicalExpression::CountSubquery(_)
            | LogicalExpression::ValueSubquery(_) => {
                // Subqueries have their own variable scope
            }
            LogicalExpression::PatternComprehension { projection, .. } => {
                Self::collect_variables(projection, vars);
            }
            LogicalExpression::MapProjection { base, entries } => {
                vars.insert(base.clone());
                for entry in entries {
                    if let crate::query::plan::MapProjectionEntry::LiteralEntry(_, expr) = entry {
                        Self::collect_variables(expr, vars);
                    }
                }
            }
            LogicalExpression::Reduce {
                initial,
                list,
                expression,
                ..
            } => {
                Self::collect_variables(initial, vars);
                Self::collect_variables(list, vars);
                Self::collect_variables(expression, vars);
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

/// Whether a conjunct must stay where it is written: it holds a subquery,
/// whose outer references [`Optimizer::collect_variables`] does not see, or a
/// volatile function, which would run a different number of times lower down.
fn stays_in_place(expr: &LogicalExpression) -> bool {
    let any = |items: &[LogicalExpression]| items.iter().any(stays_in_place);
    let maybe = |expr: &Option<Box<LogicalExpression>>| expr.as_deref().is_some_and(stays_in_place);
    match expr {
        LogicalExpression::ExistsSubquery(_)
        | LogicalExpression::CountSubquery(_)
        | LogicalExpression::ValueSubquery(_)
        | LogicalExpression::PatternComprehension { .. } => true,
        LogicalExpression::FunctionCall { name, args, .. } => {
            name.eq_ignore_ascii_case("rand") || name.eq_ignore_ascii_case("random") || any(args)
        }
        LogicalExpression::Binary { left, right, .. } => {
            stays_in_place(left) || stays_in_place(right)
        }
        LogicalExpression::Unary { operand, .. } => stays_in_place(operand),
        LogicalExpression::List(items) => any(items),
        LogicalExpression::Map(pairs) => pairs.iter().any(|(_, value)| stays_in_place(value)),
        LogicalExpression::IndexAccess { base, index } => {
            stays_in_place(base) || stays_in_place(index)
        }
        LogicalExpression::MapAccess { base, .. } => stays_in_place(base),
        LogicalExpression::SliceAccess { base, start, end } => {
            stays_in_place(base) || maybe(start) || maybe(end)
        }
        LogicalExpression::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            maybe(operand)
                || when_clauses
                    .iter()
                    .any(|(condition, result)| stays_in_place(condition) || stays_in_place(result))
                || maybe(else_clause)
        }
        LogicalExpression::ListComprehension {
            list_expr,
            filter_expr,
            map_expr,
            ..
        } => stays_in_place(list_expr) || maybe(filter_expr) || stays_in_place(map_expr),
        LogicalExpression::ListPredicate {
            list_expr,
            predicate,
            ..
        } => stays_in_place(list_expr) || stays_in_place(predicate),
        LogicalExpression::MapProjection { entries, .. } => entries.iter().any(|entry| {
            matches!(entry, crate::query::plan::MapProjectionEntry::LiteralEntry(_, value) if stays_in_place(value))
        }),
        LogicalExpression::Reduce {
            initial,
            list,
            expression,
            ..
        } => stays_in_place(initial) || stays_in_place(list) || stays_in_place(expression),
        LogicalExpression::Variable(_)
        | LogicalExpression::Property { .. }
        | LogicalExpression::Literal(_)
        | LogicalExpression::Parameter(_)
        | LogicalExpression::Labels(_)
        | LogicalExpression::Type(_)
        | LogicalExpression::Id(_) => false,
    }
}

/// The names of `names` that a projection of `items` (expression and alias)
/// passes on unchanged: projected as the variable itself under its own name,
/// or, with `pass_through`, not redefined by an item.
fn passed_on<'a>(
    names: &HashSet<String>,
    items: impl Iterator<Item = (&'a LogicalExpression, Option<&'a str>)> + Clone,
    pass_through: bool,
) -> HashSet<String> {
    names
        .iter()
        .filter(|name| {
            let mut kept = pass_through;
            for (expression, alias) in items.clone() {
                let itself = matches!(expression, LogicalExpression::Variable(v) if v == *name);
                let output = match (alias, expression) {
                    (Some(alias), _) => Some(alias),
                    (None, LogicalExpression::Variable(v)) => Some(v.as_str()),
                    _ => None,
                };
                if output == Some(name.as_str()) {
                    if !itself {
                        return false;
                    }
                    kept = true;
                }
            }
            kept
        })
        .cloned()
        .collect()
}

/// The columns the rows of `op` hold: the variables it binds (see
/// [`LogicalOperator::bound_variables`]) and, for each named path among them,
/// the columns `length(p)`, `nodes(p)` and `edges(p)` read (`_path_length_p`
/// and so on). `None` when they are not known: no predicate moves into `op`
/// then.
fn bound_columns(op: &LogicalOperator) -> Option<HashSet<String>> {
    let mut bound = op.bound_variables(None)?;
    let mut paths = Vec::new();
    named_paths(op, &mut paths);
    for path in paths {
        if bound.contains(&path) {
            bound.extend(path_columns(&path));
        }
    }
    Some(bound)
}

/// The named paths of the expands and shortest-path searches of `op` and its
/// inputs.
fn named_paths(op: &LogicalOperator, out: &mut Vec<String>) {
    match op {
        LogicalOperator::Expand(expand) => out.extend(expand.path_alias.iter().cloned()),
        LogicalOperator::ShortestPath(path) => out.push(path.path_alias.clone()),
        _ => {}
    }
    for child in op.children() {
        named_paths(child, out);
    }
}

/// The columns the planner adds for the named path `path`: its length, its
/// nodes and its edges.
fn path_columns(path: &str) -> Vec<String> {
    vec![
        format!("_path_length_{path}"),
        format!("_path_nodes_{path}"),
        format!("_path_edges_{path}"),
    ]
}

/// The variables `expr` reads, when it may move away from where it is
/// written; `None` when it must stay there (see [`stays_in_place`]).
pub(crate) fn movable_variables(expr: &LogicalExpression) -> Option<HashSet<String>> {
    if stays_in_place(expr) {
        return None;
    }
    let mut vars = HashSet::new();
    Optimizer::collect_variables(expr, &mut vars);
    Some(vars)
}

/// The conjuncts of an `AND` chain, in order.
fn conjuncts(predicate: LogicalExpression) -> Vec<LogicalExpression> {
    match predicate {
        LogicalExpression::Binary {
            left,
            op: BinaryOp::And,
            right,
        } => {
            let mut all = conjuncts(*left);
            all.extend(conjuncts(*right));
            all
        }
        other => vec![other],
    }
}

/// The filters right above each other at the top of `op`, from the top down
/// (each without its input), and the operator below them.
fn peel_filters(op: LogicalOperator) -> (Vec<FilterOp>, LogicalOperator) {
    let mut filters = Vec::new();
    let mut below = op;
    while let LogicalOperator::Filter(mut filter) = below {
        below = std::mem::replace(filter.input.as_mut(), LogicalOperator::Empty);
        filters.push(filter);
    }
    (filters, below)
}

/// `op` below the `filters` [`peel_filters`] took off it.
fn wrap_filters(filters: Vec<FilterOp>, op: LogicalOperator) -> LogicalOperator {
    filters.into_iter().rev().fold(op, |input, mut filter| {
        *filter.input = input;
        LogicalOperator::Filter(filter)
    })
}

/// The `AND` of the conjuncts, or `None` when there are none.
fn conjunction(conjuncts: Vec<LogicalExpression>) -> Option<LogicalExpression> {
    LogicalExpression::conjunction(conjuncts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::plan::{
        AggregateExpr, AggregateFunction, AggregateOp, BinaryOp, DistinctOp, ExpandDirection,
        ExpandOp, JoinOp, JoinType, LimitOp, NodeScanOp, PathMode, ProjectOp, Projection,
        ReturnItem, ReturnOp, SkipOp, SortKey, SortOp, SortOrder, UnaryOp,
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
                pushdown_hint: None,
            })),
        }));

        let optimizer = Optimizer::new();
        let optimized = optimizer.optimize(plan).unwrap();

        // The structure should remain similar (filter stays near scan)
        if let LogicalOperator::Return(ret) = &optimized.root
            && let LogicalOperator::Filter(filter) = ret.input.as_ref()
            && let LogicalOperator::NodeScan(scan) = filter.input.as_ref()
        {
            assert_eq!(scan.variable, "n");
            return;
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
                pushdown_hint: None,
                input: Box::new(LogicalOperator::Expand(ExpandOp {
                    quantified: false,
                    from_variable: "a".to_string(),
                    to_variable: "b".to_string(),
                    edge_variable: None,
                    direction: ExpandDirection::Outgoing,
                    edge_types: vec!["KNOWS".to_string()],
                    min_hops: 1,
                    max_hops: Some(1),
                    input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                        variable: "a".to_string(),
                        label: Some("Person".to_string()),
                        input: None,
                    })),
                    path_alias: None,
                    path_mode: PathMode::Walk,
                })),
            })),
        }));

        let optimizer = Optimizer::new();
        let optimized = optimizer.optimize(plan).unwrap();

        // Filter on 'a' should be pushed before the expand
        // Expected: Return -> Expand -> Filter -> NodeScan
        if let LogicalOperator::Return(ret) = &optimized.root
            && let LogicalOperator::Expand(expand) = ret.input.as_ref()
            && let LogicalOperator::Filter(filter) = expand.input.as_ref()
            && let LogicalOperator::NodeScan(scan) = filter.input.as_ref()
        {
            assert_eq!(scan.variable, "a");
            assert_eq!(expand.from_variable, "a");
            assert_eq!(expand.to_variable, "b");
            return;
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
                pushdown_hint: None,
                input: Box::new(LogicalOperator::Expand(ExpandOp {
                    quantified: false,
                    from_variable: "a".to_string(),
                    to_variable: "b".to_string(),
                    edge_variable: None,
                    direction: ExpandDirection::Outgoing,
                    edge_types: vec!["KNOWS".to_string()],
                    min_hops: 1,
                    max_hops: Some(1),
                    input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                        variable: "a".to_string(),
                        label: Some("Person".to_string()),
                        input: None,
                    })),
                    path_alias: None,
                    path_mode: PathMode::Walk,
                })),
            })),
        }));

        let optimizer = Optimizer::new();
        let optimized = optimizer.optimize(plan).unwrap();

        // Filter on 'b' should stay after the expand
        // Expected: Return -> Filter -> Expand -> NodeScan
        if let LogicalOperator::Return(ret) = &optimized.root
            && let LogicalOperator::Filter(filter) = ret.input.as_ref()
        {
            // Check that the filter is on 'b'
            if let LogicalExpression::Binary { left, .. } = &filter.predicate
                && let LogicalExpression::Property { variable, .. } = left.as_ref()
            {
                assert_eq!(variable, "b");
            }

            if let LogicalOperator::Expand(expand) = filter.input.as_ref()
                && let LogicalOperator::NodeScan(_) = expand.input.as_ref()
            {
                return;
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
                pushdown_hint: None,
            })),
        }));

        let optimized = optimizer.optimize(plan).unwrap();
        // Structure should be unchanged
        if let LogicalOperator::Return(ret) = &optimized.root
            && let LogicalOperator::Filter(_) = ret.input.as_ref()
        {
            return;
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
            pushdown_hint: None,
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
                pass_through_input: false,
            })),
        }));

        let optimized = optimizer.optimize(plan).unwrap();

        // Filter should be pushed through Project
        if let LogicalOperator::Project(proj) = &optimized.root
            && let LogicalOperator::Filter(_) = proj.input.as_ref()
        {
            return;
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
            pushdown_hint: None,
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
                pass_through_input: false,
            })),
        }));

        let optimized = optimizer.optimize(plan).unwrap();

        // Filter should stay above Project
        if let LogicalOperator::Filter(filter) = &optimized.root
            && let LogicalOperator::Project(_) = filter.input.as_ref()
        {
            return;
        }
        panic!("Expected Filter -> Project structure");
    }

    #[test]
    fn test_filter_pushdown_through_limit() {
        let optimizer = Optimizer::new();

        let plan = LogicalPlan::new(LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Literal(Value::Bool(true)),
            pushdown_hint: None,
            input: Box::new(LogicalOperator::Limit(LimitOp {
                count: 10.into(),
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".to_string(),
                    label: None,
                    input: None,
                })),
            })),
        }));

        let optimized = optimizer.optimize(plan).unwrap();

        // Filter stays above Limit (cannot be pushed through)
        if let LogicalOperator::Filter(filter) = &optimized.root
            && let LogicalOperator::Limit(_) = filter.input.as_ref()
        {
            return;
        }
        panic!("Expected Filter -> Limit structure");
    }

    #[test]
    fn test_filter_pushdown_through_sort() {
        let optimizer = Optimizer::new();

        let plan = LogicalPlan::new(LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Literal(Value::Bool(true)),
            pushdown_hint: None,
            input: Box::new(LogicalOperator::Sort(SortOp {
                keys: vec![SortKey {
                    expression: LogicalExpression::Variable("n".to_string()),
                    order: SortOrder::Ascending,
                    nulls: None,
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
        if let LogicalOperator::Filter(filter) = &optimized.root
            && let LogicalOperator::Sort(_) = filter.input.as_ref()
        {
            return;
        }
        panic!("Expected Filter -> Sort structure");
    }

    #[test]
    fn test_filter_pushdown_through_distinct() {
        let optimizer = Optimizer::new();

        let plan = LogicalPlan::new(LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Literal(Value::Bool(true)),
            pushdown_hint: None,
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
        if let LogicalOperator::Filter(filter) = &optimized.root
            && let LogicalOperator::Distinct(_) = filter.input.as_ref()
        {
            return;
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
            pushdown_hint: None,
            input: Box::new(LogicalOperator::Aggregate(AggregateOp {
                group_by: vec![],
                aggregates: vec![AggregateExpr {
                    function: AggregateFunction::Count,
                    expression: None,
                    expression2: None,
                    distinct: false,
                    alias: Some("cnt".to_string()),
                    percentile: None,
                    separator: None,
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
        if let LogicalOperator::Filter(filter) = &optimized.root
            && let LogicalOperator::Aggregate(_) = filter.input.as_ref()
        {
            return;
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
            pushdown_hint: None,
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
        if let LogicalOperator::Join(join) = &optimized.root
            && let LogicalOperator::Filter(_) = join.left.as_ref()
        {
            return;
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
            pushdown_hint: None,
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
        if let LogicalOperator::Join(join) = &optimized.root
            && let LogicalOperator::Filter(_) = join.right.as_ref()
        {
            return;
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
            pushdown_hint: None,
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
        if let LogicalOperator::Filter(filter) = &optimized.root
            && let LogicalOperator::Join(_) = filter.input.as_ref()
        {
            return;
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
                pushdown_hint: None,
                input: Box::new(LogicalOperator::Skip(SkipOp {
                    count: 5.into(),
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
                pushdown_hint: None,
                input: Box::new(LogicalOperator::Filter(FilterOp {
                    predicate: LogicalExpression::Binary {
                        left: Box::new(LogicalExpression::Property {
                            variable: "n".to_string(),
                            property: "y".to_string(),
                        }),
                        op: BinaryOp::Lt,
                        right: Box::new(LogicalExpression::Literal(Value::Int64(10))),
                    },
                    pushdown_hint: None,
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

    /// `a JOIN b JOIN c` with the conditions a = b, b = c and c = a: a cyclic
    /// join graph of three relations.
    fn triangle_join_plan() -> LogicalPlan {
        use crate::query::plan::JoinCondition;

        // Triangle pattern: a ⋈ b ⋈ c ⋈ a (cyclic)
        let scan_a = LogicalOperator::NodeScan(NodeScanOp {
            variable: "a".to_string(),
            label: Some("Person".to_string()),
            input: None,
        });
        let scan_b = LogicalOperator::NodeScan(NodeScanOp {
            variable: "b".to_string(),
            label: Some("Person".to_string()),
            input: None,
        });
        let scan_c = LogicalOperator::NodeScan(NodeScanOp {
            variable: "c".to_string(),
            label: Some("Person".to_string()),
            input: None,
        });

        // Build: Join(Join(a, b, a=b), c, b=c) with extra condition c=a
        let join_ab = LogicalOperator::Join(JoinOp {
            left: Box::new(scan_a),
            right: Box::new(scan_b),
            join_type: JoinType::Inner,
            conditions: vec![JoinCondition {
                left: LogicalExpression::Variable("a".to_string()),
                right: LogicalExpression::Variable("b".to_string()),
            }],
        });

        let join_abc = LogicalOperator::Join(JoinOp {
            left: Box::new(join_ab),
            right: Box::new(scan_c),
            join_type: JoinType::Inner,
            conditions: vec![
                JoinCondition {
                    left: LogicalExpression::Variable("b".to_string()),
                    right: LogicalExpression::Variable("c".to_string()),
                },
                JoinCondition {
                    left: LogicalExpression::Variable("c".to_string()),
                    right: LogicalExpression::Variable("a".to_string()),
                },
            ],
        });

        LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("a".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(join_abc),
        }))
    }

    fn has_multi_way_join(op: &LogicalOperator) -> bool {
        match op {
            LogicalOperator::MultiWayJoin(_) => true,
            LogicalOperator::Return(ret) => has_multi_way_join(&ret.input),
            LogicalOperator::Filter(f) => has_multi_way_join(&f.input),
            LogicalOperator::Project(p) => has_multi_way_join(&p.input),
            _ => false,
        }
    }

    /// The LPG leapfrog join returned wrong rows (it intersected only the
    /// first shared variable), so LPG plans keep binary joins.
    #[test]
    fn test_cyclic_join_uses_binary_joins() {
        let mut optimizer = Optimizer::new();
        optimizer
            .card_estimator
            .add_table_stats("Person", cardinality::TableStats::new(1000));
        let optimized = optimizer.optimize(triangle_join_plan()).unwrap();
        assert!(!has_multi_way_join(&optimized.root));
    }

    /// The RDF planner's leapfrog join still gets cyclic joins.
    #[cfg(feature = "triple-store")]
    #[test]
    fn test_rdf_cyclic_join_produces_multi_way_join() {
        let optimizer =
            Optimizer::from_rdf_statistics(grafeo_core::statistics::RdfStatistics::default());
        let optimized = optimizer.optimize(triangle_join_plan()).unwrap();
        assert!(has_multi_way_join(&optimized.root));
    }

    #[test]
    fn test_acyclic_join_uses_binary_joins() {
        use crate::query::plan::JoinCondition;

        // Chain: a ⋈ b ⋈ c (acyclic)
        let scan_a = LogicalOperator::NodeScan(NodeScanOp {
            variable: "a".to_string(),
            label: Some("Person".to_string()),
            input: None,
        });
        let scan_b = LogicalOperator::NodeScan(NodeScanOp {
            variable: "b".to_string(),
            label: Some("Person".to_string()),
            input: None,
        });
        let scan_c = LogicalOperator::NodeScan(NodeScanOp {
            variable: "c".to_string(),
            label: Some("Company".to_string()),
            input: None,
        });

        let join_ab = LogicalOperator::Join(JoinOp {
            left: Box::new(scan_a),
            right: Box::new(scan_b),
            join_type: JoinType::Inner,
            conditions: vec![JoinCondition {
                left: LogicalExpression::Variable("a".to_string()),
                right: LogicalExpression::Variable("b".to_string()),
            }],
        });

        let join_abc = LogicalOperator::Join(JoinOp {
            left: Box::new(join_ab),
            right: Box::new(scan_c),
            join_type: JoinType::Inner,
            conditions: vec![JoinCondition {
                left: LogicalExpression::Variable("b".to_string()),
                right: LogicalExpression::Variable("c".to_string()),
            }],
        });

        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("a".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(join_abc),
        }));

        let mut optimizer = Optimizer::new();
        optimizer
            .card_estimator
            .add_table_stats("Person", cardinality::TableStats::new(1000));
        optimizer
            .card_estimator
            .add_table_stats("Company", cardinality::TableStats::new(100));

        let optimized = optimizer.optimize(plan).unwrap();

        // Should NOT contain MultiWayJoin for acyclic pattern
        fn has_multi_way_join(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::MultiWayJoin(_) => true,
                LogicalOperator::Return(ret) => has_multi_way_join(&ret.input),
                LogicalOperator::Filter(f) => has_multi_way_join(&f.input),
                LogicalOperator::Project(p) => has_multi_way_join(&p.input),
                LogicalOperator::Join(j) => {
                    has_multi_way_join(&j.left) || has_multi_way_join(&j.right)
                }
                _ => false,
            }
        }

        assert!(
            !has_multi_way_join(&optimized.root),
            "Acyclic join should NOT produce MultiWayJoin"
        );
    }

    /// A filter over a node scan with input: a later `MATCH` without a shared
    /// variable, scanned once per row of the earlier ones (#455).
    mod scan_with_input {
        use super::*;

        pub(super) fn property(variable: &str, name: &str) -> LogicalExpression {
            LogicalExpression::Property {
                variable: variable.to_string(),
                property: name.to_string(),
            }
        }

        pub(super) fn compare(
            left: LogicalExpression,
            op: BinaryOp,
            right: LogicalExpression,
        ) -> LogicalExpression {
            LogicalExpression::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            }
        }

        pub(super) fn equals(
            left: LogicalExpression,
            right: LogicalExpression,
        ) -> LogicalExpression {
            compare(left, BinaryOp::Eq, right)
        }

        pub(super) fn and(conjuncts: Vec<LogicalExpression>) -> LogicalExpression {
            conjunction(conjuncts).expect("at least one conjunct")
        }

        pub(super) fn active(variable: &str) -> LogicalExpression {
            equals(
                property(variable, "active"),
                LogicalExpression::Literal(Value::Bool(true)),
            )
        }

        pub(super) fn scan(
            variable: &str,
            label: Option<&str>,
            input: Option<LogicalOperator>,
        ) -> LogicalOperator {
            LogicalOperator::NodeScan(NodeScanOp {
                variable: variable.to_string(),
                label: label.map(str::to_string),
                input: input.map(Box::new),
            })
        }

        pub(super) fn filter(
            predicate: LogicalExpression,
            input: LogicalOperator,
        ) -> LogicalOperator {
            LogicalOperator::Filter(FilterOp {
                predicate,
                pushdown_hint: None,
                input: Box::new(input),
            })
        }

        /// `(graph_src)-[edge:CALLS]->(graph_tgt)`.
        fn calls(input: LogicalOperator) -> LogicalOperator {
            LogicalOperator::Expand(ExpandOp {
                quantified: false,
                from_variable: "graph_src".to_string(),
                to_variable: "graph_tgt".to_string(),
                edge_variable: Some("edge".to_string()),
                direction: ExpandDirection::Outgoing,
                edge_types: vec!["CALLS".to_string()],
                min_hops: 1,
                max_hops: Some(1),
                input: Box::new(input),
                path_alias: None,
                path_mode: PathMode::Walk,
            })
        }

        /// The plan text of `root` after optimization.
        fn optimized(root: LogicalOperator) -> String {
            Optimizer::new()
                .optimize(LogicalPlan::new(root))
                .unwrap()
                .root
                .explain_tree()
        }

        /// The shape of #455: each conjunct of the second `WHERE` filters the
        /// rows right above the scan of the node it reads, and one that reads
        /// only the first `MATCH` goes further down, below both scans. The
        /// first `MATCH`'s own filter is pushed down too.
        #[test]
        fn each_conjunct_moves_onto_the_scan_of_its_node() {
            let first_match = || {
                filter(
                    and(vec![active("graph_src"), active("graph_tgt")]),
                    calls(scan("graph_src", None, None)),
                )
            };
            let source_model = equals(
                property("model_src", "source_identifier"),
                property("graph_src", "id"),
            );
            let target_model = equals(
                property("model_tgt", "source_identifier"),
                property("graph_tgt", "id"),
            );
            let no_self_call = compare(
                property("graph_src", "id"),
                BinaryOp::Ne,
                property("graph_tgt", "id"),
            );
            let written = filter(
                and(vec![
                    source_model.clone(),
                    target_model.clone(),
                    no_self_call.clone(),
                ]),
                scan(
                    "model_tgt",
                    Some("Model"),
                    Some(scan("model_src", Some("Model"), Some(first_match()))),
                ),
            );
            let expected = filter(
                target_model,
                scan(
                    "model_tgt",
                    Some("Model"),
                    Some(filter(
                        source_model,
                        scan(
                            "model_src",
                            Some("Model"),
                            Some(filter(
                                active("graph_tgt"),
                                filter(
                                    no_self_call,
                                    calls(filter(
                                        active("graph_src"),
                                        scan("graph_src", None, None),
                                    )),
                                ),
                            )),
                        ),
                    )),
                ),
            );
            assert_eq!(optimized(written), expected.explain_tree());
        }

        /// A conjunct that reads the scanned node stays above the scan, and so
        /// does one that reads a variable the input does not bind; the others
        /// filter the input. The conjuncts left above keep their order.
        #[test]
        fn only_conjuncts_on_the_input_alone_move_below_the_scan() {
            let input_only = compare(
                property("f", "size"),
                BinaryOp::Gt,
                LogicalExpression::Literal(Value::Int64(3)),
            );
            let key = equals(property("t", "filePath"), property("f", "path"));
            let unbound = equals(property("f", "owner"), property("outer", "name"));
            let constant = equals(
                LogicalExpression::Parameter("flag".to_string()),
                LogicalExpression::Literal(Value::Bool(true)),
            );
            let written = filter(
                and(vec![
                    key.clone(),
                    input_only.clone(),
                    unbound.clone(),
                    constant.clone(),
                ]),
                scan(
                    "t",
                    Some("TypeDefinition"),
                    Some(scan("f", Some("File"), None)),
                ),
            );
            let expected = filter(
                and(vec![key, unbound]),
                scan(
                    "t",
                    Some("TypeDefinition"),
                    Some(filter(
                        and(vec![input_only, constant]),
                        scan("f", Some("File"), None),
                    )),
                ),
            );
            assert_eq!(optimized(written), expected.explain_tree());
        }

        /// When the input binds the scanned variable too (a correlated
        /// subquery), a conjunct that reads it still stays above the scan.
        #[test]
        fn a_conjunct_on_the_scanned_node_stays_when_the_input_binds_it() {
            let kind = equals(
                property("t", "kind"),
                LogicalExpression::Literal(Value::from("struct")),
            );
            let input_only = compare(
                property("f", "size"),
                BinaryOp::Gt,
                LogicalExpression::Literal(Value::Int64(3)),
            );
            let bound = |input: LogicalOperator| scan("t", None, Some(input));
            let written = filter(
                and(vec![kind.clone(), input_only.clone()]),
                scan(
                    "t",
                    Some("TypeDefinition"),
                    Some(bound(scan("f", Some("File"), None))),
                ),
            );
            let expected = filter(
                kind,
                scan(
                    "t",
                    Some("TypeDefinition"),
                    Some(bound(filter(input_only, scan("f", Some("File"), None)))),
                ),
            );
            assert_eq!(optimized(written), expected.explain_tree());
        }

        /// Subquery and volatile conjuncts stay where they are written, also
        /// when they read only the input: they would run a different number
        /// of times below the scan.
        #[test]
        fn subqueries_and_volatile_conjuncts_stay_above_the_scan() {
            let input_only = compare(
                property("f", "size"),
                BinaryOp::Gt,
                LogicalExpression::Literal(Value::Int64(3)),
            );
            let random = compare(
                LogicalExpression::FunctionCall {
                    name: "rand".to_string(),
                    args: vec![],
                    distinct: false,
                },
                BinaryOp::Lt,
                property("f", "share"),
            );
            let exists = LogicalExpression::ExistsSubquery(Box::new(filter(
                equals(property("m", "path"), property("f", "path")),
                scan("m", Some("Model"), None),
            )));
            let written = filter(
                and(vec![random.clone(), input_only.clone(), exists.clone()]),
                scan(
                    "t",
                    Some("TypeDefinition"),
                    Some(scan("f", Some("File"), None)),
                ),
            );
            let expected = filter(
                and(vec![random, exists]),
                scan(
                    "t",
                    Some("TypeDefinition"),
                    Some(filter(input_only, scan("f", Some("File"), None))),
                ),
            );
            assert_eq!(optimized(written), expected.explain_tree());
        }

        /// `UNWIND $rows AS item MATCH (s), (d) WHERE id(s) = item.src AND
        /// id(d) = item.dst` with the rows written to: the filter below the
        /// write is pushed down like any other, each conjunct onto the scan of
        /// its node, where the planner seeks it. Left above both scans, it
        /// pinned only `d` and scanned every `s` for each row.
        #[test]
        fn a_filter_below_a_write_moves_onto_the_scans_of_its_nodes() {
            let rows = || {
                LogicalOperator::Unwind(crate::query::plan::UnwindOp {
                    expression: LogicalExpression::Parameter("rows".to_string()),
                    variable: "item".to_string(),
                    ordinality_var: None,
                    offset_var: None,
                    input: Box::new(LogicalOperator::Empty),
                })
            };
            let pinned = |node: &str, key: &str| {
                equals(
                    LogicalExpression::Id(node.to_string()),
                    property("item", key),
                )
            };
            let merge = |input: LogicalOperator| {
                LogicalOperator::MergeRelationship(crate::query::plan::MergeRelationshipOp {
                    variable: "r".to_string(),
                    source_variable: "s".to_string(),
                    target_variable: "d".to_string(),
                    undirected: false,
                    edge_type: "LINK".to_string(),
                    match_properties: Vec::new(),
                    on_create: Vec::new(),
                    on_match: Vec::new(),
                    input: Box::new(input),
                })
            };
            let written = merge(filter(
                and(vec![pinned("s", "src"), pinned("d", "dst")]),
                scan("d", None, Some(scan("s", None, Some(rows())))),
            ));
            let expected = merge(filter(
                pinned("d", "dst"),
                scan(
                    "d",
                    None,
                    Some(filter(pinned("s", "src"), scan("s", None, Some(rows())))),
                ),
            ));
            assert_eq!(optimized(written), expected.explain_tree());
        }

        /// A scan without input has no rows to filter first: the filter stays
        /// above it, whatever its conjuncts read.
        #[test]
        fn a_scan_without_input_keeps_its_filter() {
            let written = filter(
                and(vec![
                    equals(
                        LogicalExpression::Parameter("flag".to_string()),
                        LogicalExpression::Literal(Value::Bool(true)),
                    ),
                    equals(
                        property("t", "filePath"),
                        LogicalExpression::Literal(Value::from("amsterdam.rs")),
                    ),
                ]),
                scan("t", Some("TypeDefinition"), None),
            );
            let expected = written.explain_tree();
            assert_eq!(optimized(written), expected);
        }
    }

    /// A filter over a join or a `CALL` subquery moves into a side only when
    /// that side binds every variable it reads (#455).
    mod join_sides {
        use super::scan_with_input::{and, compare, equals, filter, property, scan};
        use super::*;
        use crate::query::plan::{CreateNodeOp, JoinCondition, ParameterScanOp};

        fn variable(name: &str) -> LogicalExpression {
            LogicalExpression::Variable(name.to_string())
        }

        fn literal(value: i64) -> LogicalExpression {
            LogicalExpression::Literal(Value::Int64(value))
        }

        /// `(a)-[:R]->(c)` over `input`, named `p` when `path`.
        fn expand(input: LogicalOperator, path: bool) -> LogicalOperator {
            LogicalOperator::Expand(ExpandOp {
                quantified: false,
                from_variable: "a".to_string(),
                to_variable: "c".to_string(),
                edge_variable: None,
                direction: ExpandDirection::Outgoing,
                edge_types: vec!["R".to_string()],
                min_hops: 1,
                max_hops: Some(1),
                input: Box::new(input),
                path_alias: path.then(|| "p".to_string()),
                path_mode: PathMode::Walk,
            })
        }

        /// `MATCH (b:B), (a:A), (a)-[:R]->(c)`: a scan of `a` with input on
        /// the left, joined on `a` with the expand from another scan of `a`.
        fn joined(path: bool) -> LogicalOperator {
            LogicalOperator::Join(JoinOp {
                left: Box::new(scan("a", Some("A"), Some(scan("b", Some("B"), None)))),
                right: Box::new(expand(scan("a", None, None), path)),
                join_type: JoinType::Inner,
                conditions: vec![JoinCondition {
                    left: variable("a"),
                    right: variable("a"),
                }],
            })
        }

        /// `CALL { WITH b <subquery> RETURN <returned> }` after
        /// `MATCH (a:A) MATCH (b:B)`.
        fn call(subquery: LogicalOperator, returned: ReturnItem) -> LogicalOperator {
            LogicalOperator::Apply(crate::query::plan::ApplyOp {
                input: Box::new(scan("b", Some("B"), Some(scan("a", Some("A"), None)))),
                subplan: Box::new(LogicalOperator::Return(ReturnOp {
                    items: vec![returned],
                    distinct: false,
                    input: Box::new(subquery),
                })),
                shared_variables: vec!["b".to_string()],
                optional: false,
                unit: false,
            })
        }

        fn imported() -> LogicalOperator {
            LogicalOperator::ParameterScan(ParameterScanOp {
                columns: vec!["b".to_string()],
            })
        }

        /// The plan text after filter pushdown.
        fn pushed(root: LogicalOperator) -> String {
            Optimizer::new().push_filters_down(root).explain_tree()
        }

        fn join(
            join_type: JoinType,
            conditions: Vec<JoinCondition>,
        ) -> impl Fn(LogicalOperator, LogicalOperator) -> LogicalOperator {
            move |left, right| {
                LogicalOperator::Join(JoinOp {
                    left: Box::new(left),
                    right: Box::new(right),
                    join_type,
                    conditions: conditions.clone(),
                })
            }
        }

        fn on_a() -> Vec<JoinCondition> {
            vec![JoinCondition {
                left: variable("a"),
                right: variable("a"),
            }]
        }

        /// The left side of `joined`.
        fn left_side() -> LogicalOperator {
            scan("a", Some("A"), Some(scan("b", Some("B"), None)))
        }

        /// A predicate on the length of `p` and on `b` reads both sides: it
        /// stays above the join (the left side has no `_path_length_p`).
        #[test]
        fn a_predicate_on_both_sides_stays_above_the_join() {
            let written = filter(
                compare(variable("_path_length_p"), BinaryOp::Ge, property("b", "k")),
                joined(true),
            );
            let expected = written.explain_tree();
            assert_eq!(pushed(written), expected);
        }

        /// The length of `p` is a column of the expand that binds `p`: a
        /// predicate on it alone goes to that side, and stays above the
        /// expand.
        #[test]
        fn a_predicate_on_a_path_length_goes_above_the_expand_of_its_side() {
            let length = compare(variable("_path_length_p"), BinaryOp::Ge, literal(1));
            let written = filter(length.clone(), joined(true));
            let expected = join(JoinType::Inner, on_a())(
                left_side(),
                filter(length, expand(scan("a", None, None), true)),
            );
            assert_eq!(pushed(written), expected.explain_tree());
        }

        /// A variable an inner join equates has the same value on both sides:
        /// a predicate on it filters both, each right above its scan.
        #[test]
        fn a_predicate_on_an_equated_variable_filters_both_sides() {
            let key = equals(property("a", "k"), literal(1));
            let written = filter(key.clone(), joined(false));
            let expected = join(JoinType::Inner, on_a())(
                filter(key.clone(), left_side()),
                expand(filter(key, scan("a", None, None)), false),
            );
            assert_eq!(pushed(written), expected.explain_tree());
        }

        /// Each conjunct goes where its own variables are bound: one on `b`
        /// to the left side, one on the equated `a` to both.
        #[test]
        fn each_conjunct_goes_to_the_sides_that_bind_it() {
            let on_b = equals(property("b", "k"), literal(1));
            let on_key = equals(property("a", "k"), literal(2));
            let on_c = equals(property("c", "k"), literal(3));
            let written = filter(
                and(vec![on_b.clone(), on_key.clone(), on_c.clone()]),
                joined(false),
            );
            let expected = join(JoinType::Inner, on_a())(
                filter(
                    on_key.clone(),
                    scan(
                        "a",
                        Some("A"),
                        Some(filter(on_b, scan("b", Some("B"), None))),
                    ),
                ),
                filter(on_c, expand(filter(on_key, scan("a", None, None)), false)),
            );
            assert_eq!(pushed(written), expected.explain_tree());
        }

        /// A cross join does not equate the variables both sides bind: a
        /// predicate on one of them stays above it. One on a variable of one
        /// side goes to that side.
        #[test]
        fn a_cross_join_keeps_a_predicate_on_a_variable_of_both_sides() {
            let cross = join(JoinType::Cross, Vec::new());
            let written = filter(
                equals(property("a", "k"), literal(1)),
                cross(left_side(), expand(scan("a", None, None), false)),
            );
            let expected = written.explain_tree();
            assert_eq!(pushed(written), expected);

            let on_b = equals(property("b", "k"), literal(1));
            let written = filter(
                on_b.clone(),
                cross(left_side(), expand(scan("a", None, None), false)),
            );
            let expected = cross(
                scan(
                    "a",
                    Some("A"),
                    Some(filter(on_b, scan("b", Some("B"), None))),
                ),
                expand(scan("a", None, None), false),
            );
            assert_eq!(pushed(written), expected.explain_tree());
        }

        /// A predicate on a value the subquery returns stays above the call;
        /// one on the input alone filters the input.
        #[test]
        fn a_predicate_on_a_returned_value_stays_above_the_call() {
            let returned = || ReturnItem {
                expression: property("b", "k"),
                alias: Some("w".to_string()),
            };
            let written = filter(
                equals(property("a", "k"), variable("w")),
                call(imported(), returned()),
            );
            let expected = written.explain_tree();
            assert_eq!(pushed(written), expected);

            let on_a = equals(property("a", "k"), literal(1));
            let written = filter(on_a.clone(), call(imported(), returned()));
            let LogicalOperator::Apply(mut expected) = call(imported(), returned()) else {
                unreachable!("call builds an apply")
            };
            expected.input = Box::new(scan(
                "b",
                Some("B"),
                Some(filter(on_a, scan("a", Some("A"), None))),
            ));
            assert_eq!(
                pushed(written),
                LogicalOperator::Apply(expected).explain_tree()
            );
        }

        /// A subquery that writes runs once per input row: the filter stays
        /// above it, so that it writes for every row.
        #[test]
        fn a_filter_stays_above_a_call_that_writes() {
            let writes = LogicalOperator::CreateNode(CreateNodeOp {
                variable: "n".to_string(),
                labels: vec!["T".to_string()],
                properties: Vec::new(),
                input: Some(Box::new(imported())),
            });
            let written = filter(
                equals(property("a", "k"), literal(1)),
                call(
                    writes,
                    ReturnItem {
                        expression: literal(1),
                        alias: Some("one".to_string()),
                    },
                ),
            );
            let expected = written.explain_tree();
            assert_eq!(pushed(written), expected);
        }

        /// `OPTIONAL MATCH`: a predicate on the left side and on a column only
        /// the right side has stays above the left join.
        #[test]
        fn a_predicate_on_both_sides_stays_above_the_left_join() {
            let written = filter(
                compare(variable("_path_length_p"), BinaryOp::Ge, property("b", "k")),
                LogicalOperator::LeftJoin(crate::query::plan::LeftJoinOp {
                    left: Box::new(left_side()),
                    right: Box::new(expand(scan("a", None, None), true)),
                    condition: None,
                }),
            );
            let expected = written.explain_tree();
            assert_eq!(pushed(written), expected);
        }

        fn left_join(left: LogicalOperator, right: LogicalOperator) -> LogicalOperator {
            LogicalOperator::LeftJoin(crate::query::plan::LeftJoinOp {
                left: Box::new(left),
                right: Box::new(right),
                condition: None,
            })
        }

        /// A `WHERE` after `MATCH (b:B), (a:A) OPTIONAL MATCH (a)-[:R]->(c)
        /// WITH *`: each conjunct goes where its variables are bound. One on
        /// the left side alone (`b`, or `b` with the shared `a`) filters the
        /// left side, one on the shared `a` alone both sides (a pair matches
        /// on `a`, so the left row fails it too), and one on `c`, which only
        /// the optional side binds, stays above the left join: moved into it,
        /// it would turn a row the `WHERE` drops into one with nulls.
        #[test]
        fn each_conjunct_above_a_left_join_goes_to_the_sides_that_bind_it() {
            let on_b = equals(property("b", "k"), literal(1));
            let on_c = equals(property("c", "k"), literal(3));
            let on_a_and_b = equals(property("a", "k"), property("b", "k"));
            let on_key = equals(property("a", "k"), literal(2));
            let written = filter(
                and(vec![
                    on_b.clone(),
                    on_c.clone(),
                    on_a_and_b.clone(),
                    on_key.clone(),
                ]),
                left_join(left_side(), expand(scan("a", None, None), false)),
            );
            let expected = filter(
                on_c,
                left_join(
                    filter(
                        and(vec![on_a_and_b, on_key.clone()]),
                        scan(
                            "a",
                            Some("A"),
                            Some(filter(on_b, scan("b", Some("B"), None))),
                        ),
                    ),
                    expand(filter(on_key, scan("a", None, None)), false),
                ),
            );
            assert_eq!(pushed(written), expected.explain_tree());
        }

        /// A conjunct on the optional side alone stays above the left join,
        /// also when it is the only one.
        #[test]
        fn a_conjunct_on_the_optional_side_stays_above_the_left_join() {
            let written = filter(
                LogicalExpression::Unary {
                    op: UnaryOp::IsNull,
                    operand: Box::new(variable("c")),
                },
                left_join(left_side(), expand(scan("a", None, None), false)),
            );
            let expected = written.explain_tree();
            assert_eq!(pushed(written), expected);
        }

        /// The filter of an earlier `MATCH` inside the input of a later scan
        /// is pushed down too.
        #[test]
        fn a_filter_in_the_input_of_a_scan_is_pushed_down() {
            let on_a = equals(property("a", "k"), literal(1));
            let on_c = equals(property("c", "k"), literal(2));
            let written = scan(
                "m",
                Some("M"),
                Some(filter(
                    and(vec![on_a.clone(), on_c.clone()]),
                    expand(scan("a", None, None), false),
                )),
            );
            let expected = scan(
                "m",
                Some("M"),
                Some(filter(
                    on_c,
                    expand(filter(on_a, scan("a", None, None)), false),
                )),
            );
            assert_eq!(pushed(written), expected.explain_tree());
        }
    }

    /// Join reordering builds its plan from the relations it collects: a
    /// filter collected with them must come back with its relation.
    mod join_tree {
        use super::scan_with_input::{equals, filter, property, scan};
        use super::*;
        use crate::query::plan::JoinCondition;

        fn literal(value: i64) -> LogicalExpression {
            LogicalExpression::Literal(Value::Int64(value))
        }

        /// An inner join of `left` and `right` on `left_key = right_key`.
        fn join(
            left: LogicalOperator,
            right: LogicalOperator,
            left_key: LogicalExpression,
            right_key: LogicalExpression,
        ) -> LogicalOperator {
            LogicalOperator::Join(JoinOp {
                left: Box::new(left),
                right: Box::new(right),
                join_type: JoinType::Inner,
                conditions: vec![JoinCondition {
                    left: left_key,
                    right: right_key,
                }],
            })
        }

        /// The predicates of the filters of `op`, each with the operator
        /// right below it, as plan text.
        fn filters(op: &LogicalOperator, out: &mut Vec<String>) {
            if let LogicalOperator::Filter(filter) = op {
                out.push(format!(
                    "{:?} over {}",
                    filter.predicate,
                    filter.input.explain_tree().lines().next().unwrap_or("")
                ));
            }
            for child in op.children() {
                filters(child, out);
            }
        }

        fn filters_of(op: &LogicalOperator) -> Vec<String> {
            let mut out = Vec::new();
            filters(op, &mut out);
            out.sort();
            out
        }

        /// `MATCH (a), (b) WHERE a.k = 3 AND a.id = b.ref`, with the filter on
        /// `a` below the join: a reordered plan keeps it on the scan of `a`.
        #[test]
        fn a_filter_on_a_relation_stays_with_it_when_joins_are_reordered() {
            let on_a = equals(property("a", "k"), literal(3));
            let plan = join(
                filter(on_a, scan("a", None, None)),
                scan("b", None, None),
                property("a", "id"),
                property("b", "ref"),
            );
            let before = filters_of(&plan);
            let reordered = Optimizer::new().reorder_joins(plan);
            assert_eq!(
                filters_of(&reordered),
                before,
                "{}",
                reordered.explain_tree()
            );
        }

        /// A filter over a join of two relations, inside a join with a third:
        /// it reads both, so no relation can take it, and it stays.
        #[test]
        fn a_filter_over_a_join_in_a_join_tree_stays() {
            let across = equals(property("a", "k"), property("b", "k"));
            let plan = join(
                filter(
                    across,
                    join(
                        scan("a", None, None),
                        scan("b", None, None),
                        property("a", "id"),
                        property("b", "ref"),
                    ),
                ),
                scan("c", None, None),
                property("b", "id"),
                property("c", "ref"),
            );
            let before = filters_of(&plan);
            let reordered = Optimizer::new().reorder_joins(plan);
            assert_eq!(
                filters_of(&reordered),
                before,
                "{}",
                reordered.explain_tree()
            );
        }
    }

    /// A single-hop expand whose target a filter pins by ID or by an indexed
    /// property, from a source nothing pins, starts at the target and
    /// follows the edges the other way: the planner then seeks the few
    /// targets instead of expanding every edge of every node.
    mod sought_end {
        use super::scan_with_input::{and, compare, equals, filter, property, scan};
        use super::*;
        use crate::query::plan::UnwindOp;

        fn ids() -> LogicalExpression {
            LogicalExpression::List(vec![
                LogicalExpression::Literal(Value::Int64(3)),
                LogicalExpression::Literal(Value::Int64(19)),
            ])
        }

        /// `id(variable) IN [3, 19]`.
        fn in_ids(variable: &str) -> LogicalExpression {
            compare(
                LogicalExpression::Id(variable.to_string()),
                BinaryOp::In,
                ids(),
            )
        }

        fn has_label(variable: &str, label: &str) -> LogicalExpression {
            LogicalExpression::FunctionCall {
                name: "hasLabel".to_string(),
                args: vec![
                    LogicalExpression::Variable(variable.to_string()),
                    LogicalExpression::Literal(Value::from(label)),
                ],
                distinct: false,
            }
        }

        /// `(from)-[r:R]->(to)` in `direction`, over `input`.
        fn hop(
            from: &str,
            to: &str,
            direction: ExpandDirection,
            input: LogicalOperator,
        ) -> ExpandOp {
            ExpandOp {
                quantified: false,
                from_variable: from.to_string(),
                to_variable: to.to_string(),
                edge_variable: Some("r".to_string()),
                direction,
                edge_types: vec!["R".to_string()],
                min_hops: 1,
                max_hops: Some(1),
                input: Box::new(input),
                path_alias: None,
                path_mode: PathMode::Walk,
            }
        }

        fn expand(
            from: &str,
            to: &str,
            direction: ExpandDirection,
            input: LogicalOperator,
        ) -> LogicalOperator {
            LogicalOperator::Expand(hop(from, to, direction, input))
        }

        /// `RETURN r, src, tgt`: names its columns, so their order is free.
        fn returned(input: LogicalOperator) -> LogicalOperator {
            LogicalOperator::Return(ReturnOp {
                items: ["r", "src", "tgt"]
                    .into_iter()
                    .map(|name| ReturnItem {
                        expression: LogicalExpression::Variable(name.to_string()),
                        alias: None,
                    })
                    .collect(),
                distinct: false,
                input: Box::new(input),
            })
        }

        fn optimized_by(optimizer: &Optimizer, root: LogicalOperator) -> String {
            optimizer
                .optimize(LogicalPlan::new(root))
                .unwrap()
                .root
                .explain_tree()
        }

        fn optimized(root: LogicalOperator) -> String {
            optimized_by(&Optimizer::new(), root)
        }

        /// `MATCH (src)-[r:R]->(tgt) WHERE id(tgt) IN [3, 19]`, written each
        /// way round and undirected: the expand starts at the sought `tgt`.
        #[test]
        fn a_target_pinned_by_id_starts_the_expand() {
            for (written, turned) in [
                (ExpandDirection::Outgoing, ExpandDirection::Incoming),
                (ExpandDirection::Incoming, ExpandDirection::Outgoing),
                (ExpandDirection::Both, ExpandDirection::Both),
            ] {
                let plan = returned(filter(
                    in_ids("tgt"),
                    expand("src", "tgt", written, scan("src", None, None)),
                ));
                let expected = returned(expand(
                    "tgt",
                    "src",
                    turned,
                    filter(in_ids("tgt"), scan("tgt", None, None)),
                ));
                assert_eq!(optimized(plan), expected.explain_tree(), "{written:?}");
            }
        }

        /// The conjuncts on the target alone go with the seek; a check of
        /// the source and a conjunct that reads both filter the rows the
        /// turned expand makes, the source's first, as they ran before.
        #[test]
        fn the_other_conjuncts_filter_after_the_turned_expand() {
            let on_source = equals(
                property("src", "k"),
                LogicalExpression::Literal(Value::Int64(88)),
            );
            let across = compare(property("tgt", "k"), BinaryOp::Gt, property("src", "k"));
            let plan = returned(filter(
                and(vec![
                    in_ids("tgt"),
                    across.clone(),
                    on_source.clone(),
                    has_label("tgt", "Person"),
                ]),
                expand(
                    "src",
                    "tgt",
                    ExpandDirection::Outgoing,
                    scan("src", None, None),
                ),
            ));
            let expected = returned(filter(
                and(vec![on_source, across]),
                expand(
                    "tgt",
                    "src",
                    ExpandDirection::Incoming,
                    filter(
                        and(vec![in_ids("tgt"), has_label("tgt", "Person")]),
                        scan("tgt", None, None),
                    ),
                ),
            ));
            assert_eq!(optimized(plan), expected.explain_tree());
        }

        /// The plans that keep their expand as written, each with the reason.
        #[test]
        fn an_expand_keeps_its_start_when_turning_would_not_seek_or_not_hold() {
            let outgoing =
                |source: LogicalOperator| expand("src", "tgt", ExpandDirection::Outgoing, source);
            let pinned_source = equals(
                LogicalExpression::Id("src".to_string()),
                LogicalExpression::Literal(Value::Int64(88)),
            );
            let random = compare(
                LogicalExpression::FunctionCall {
                    name: "rand".to_string(),
                    args: Vec::new(),
                    distinct: false,
                },
                BinaryOp::Lt,
                LogicalExpression::Literal(Value::Float64(0.5)),
            );
            let mut named_path = hop(
                "src",
                "tgt",
                ExpandDirection::Outgoing,
                scan("src", None, None),
            );
            named_path.path_alias = Some("p".to_string());
            let mut quantified = hop(
                "src",
                "tgt",
                ExpandDirection::Outgoing,
                scan("src", None, None),
            );
            quantified.quantified = true;
            let mut longer = hop(
                "src",
                "tgt",
                ExpandDirection::Outgoing,
                scan("src", None, None),
            );
            longer.max_hops = Some(3);
            let star = |input: LogicalOperator| {
                LogicalOperator::Return(ReturnOp {
                    items: vec![ReturnItem {
                        expression: LogicalExpression::Variable("*".to_string()),
                        alias: None,
                    }],
                    distinct: false,
                    input: Box::new(input),
                })
            };
            let kept: Vec<(&str, LogicalOperator)> = vec![
                (
                    "the source is sought by ID",
                    returned(filter(
                        and(vec![pinned_source, in_ids("tgt")]),
                        outgoing(scan("src", None, None)),
                    )),
                ),
                (
                    "the source scans a label, which a check would read at the epoch",
                    returned(filter(
                        in_ids("tgt"),
                        outgoing(scan("src", Some("Person"), None)),
                    )),
                ),
                (
                    "a check of the source calls a volatile function",
                    returned(filter(
                        in_ids("tgt"),
                        outgoing(filter(random, scan("src", None, None))),
                    )),
                ),
                (
                    "the key reads the source",
                    returned(filter(
                        equals(
                            LogicalExpression::Id("tgt".to_string()),
                            property("src", "next"),
                        ),
                        outgoing(scan("src", None, None)),
                    )),
                ),
                (
                    "a property that is not indexed",
                    returned(filter(
                        equals(
                            property("tgt", "key"),
                            LogicalExpression::Literal(Value::from("Amsterdam")),
                        ),
                        outgoing(scan("src", None, None)),
                    )),
                ),
                (
                    "a named path",
                    returned(filter(in_ids("tgt"), LogicalOperator::Expand(named_path))),
                ),
                (
                    "a quantified edge",
                    returned(filter(in_ids("tgt"), LogicalOperator::Expand(quantified))),
                ),
                (
                    "more than one hop",
                    returned(filter(in_ids("tgt"), LogicalOperator::Expand(longer))),
                ),
                (
                    "the input binds the source (a correlated subquery)",
                    returned(filter(
                        in_ids("tgt"),
                        outgoing(scan("src", None, Some(scan("src", None, None)))),
                    )),
                ),
                (
                    "RETURN * reads the columns in their order",
                    star(filter(in_ids("tgt"), outgoing(scan("src", None, None)))),
                ),
            ];
            for (reason, plan) in kept {
                let pushed_only = Optimizer::new().push_filters_down(plan.clone());
                assert_eq!(optimized(plan), pushed_only.explain_tree(), "{reason}");
            }
        }

        /// An indexed property of the target pins it like an ID does.
        #[test]
        fn an_indexed_property_of_the_target_starts_the_expand() {
            let keyed = equals(
                property("tgt", "key"),
                LogicalExpression::Literal(Value::from("Amsterdam")),
            );
            let plan = returned(filter(
                keyed.clone(),
                expand(
                    "src",
                    "tgt",
                    ExpandDirection::Outgoing,
                    scan("src", None, None),
                ),
            ));
            let mut optimizer = Optimizer::new();
            optimizer.indexed_properties.insert("key".to_string());
            let expected = returned(expand(
                "tgt",
                "src",
                ExpandDirection::Incoming,
                filter(keyed, scan("tgt", None, None)),
            ));
            assert_eq!(optimized_by(&optimizer, plan), expected.explain_tree());
        }

        /// Without backward edges an incoming expand is slow: an outgoing
        /// expand keeps its start, an incoming one still turns.
        #[test]
        fn without_backward_edges_only_an_incoming_expand_turns() {
            let mut optimizer = Optimizer::new();
            optimizer.incoming_edges = IncomingEdges::NotKept;
            let written = |direction| {
                returned(filter(
                    in_ids("tgt"),
                    expand("src", "tgt", direction, scan("src", None, None)),
                ))
            };
            let outgoing = written(ExpandDirection::Outgoing);
            let pushed_only = Optimizer::new().push_filters_down(outgoing.clone());
            assert_eq!(
                optimized_by(&optimizer, outgoing),
                pushed_only.explain_tree()
            );
            let expected = returned(expand(
                "tgt",
                "src",
                ExpandDirection::Outgoing,
                filter(in_ids("tgt"), scan("tgt", None, None)),
            ));
            assert_eq!(
                optimized_by(&optimizer, written(ExpandDirection::Incoming)),
                expected.explain_tree()
            );
        }

        /// `UNWIND $keys AS k MATCH (src)-[r:R]->(tgt) WHERE id(tgt) = k`:
        /// the target is sought for each row of the scan's input.
        #[test]
        fn a_key_from_the_input_row_starts_the_expand() {
            let keys = || {
                LogicalOperator::Unwind(UnwindOp {
                    expression: LogicalExpression::Parameter("keys".to_string()),
                    variable: "k".to_string(),
                    ordinality_var: None,
                    offset_var: None,
                    input: Box::new(LogicalOperator::Empty),
                })
            };
            let pinned = equals(
                LogicalExpression::Id("tgt".to_string()),
                LogicalExpression::Variable("k".to_string()),
            );
            let plan = returned(filter(
                pinned.clone(),
                expand(
                    "src",
                    "tgt",
                    ExpandDirection::Outgoing,
                    scan("src", None, Some(keys())),
                ),
            ));
            let expected = returned(expand(
                "tgt",
                "src",
                ExpandDirection::Incoming,
                filter(pinned, scan("tgt", None, Some(keys()))),
            ));
            assert_eq!(optimized(plan), expected.explain_tree());
        }
    }
}
