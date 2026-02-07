//! Converts logical plans into physical execution trees.
//!
//! The optimizer produces a logical plan (what data you want), but the planner
//! converts it to a physical plan (how to actually get it). This means choosing
//! hash joins vs nested loops, picking index scans vs full scans, etc.

use crate::query::plan::{
    AddLabelOp, AggregateFunction as LogicalAggregateFunction, AggregateOp, AntiJoinOp, BinaryOp,
    CreateEdgeOp, CreateNodeOp, DeleteEdgeOp, DeleteNodeOp, DistinctOp, ExpandDirection, ExpandOp,
    FilterOp, JoinOp, JoinType, LeftJoinOp, LimitOp, LogicalExpression, LogicalOperator,
    LogicalPlan, MergeOp, NodeScanOp, RemoveLabelOp, ReturnOp, SetPropertyOp, ShortestPathOp,
    SkipOp, SortOp, SortOrder, UnaryOp, UnionOp, UnwindOp,
};
use grafeo_common::types::{EpochId, TxId};
use grafeo_common::types::{LogicalType, Value};
use grafeo_common::utils::error::{Error, Result};
use grafeo_core::execution::AdaptiveContext;
use grafeo_core::execution::operators::{
    AddLabelOperator, AggregateExpr as PhysicalAggregateExpr,
    AggregateFunction as PhysicalAggregateFunction, BinaryFilterOp, CreateEdgeOperator,
    CreateNodeOperator, DeleteEdgeOperator, DeleteNodeOperator, DistinctOperator, EmptyOperator,
    ExpandOperator, ExpandStep, ExpressionPredicate, FactorizedAggregate,
    FactorizedAggregateOperator, FilterExpression, FilterOperator, HashAggregateOperator,
    HashJoinOperator, JoinType as PhysicalJoinType, LazyFactorizedChainOperator,
    LeapfrogJoinOperator, LimitOperator, MergeOperator, NestedLoopJoinOperator, NodeListOperator,
    NullOrder, Operator, ProjectExpr, ProjectOperator, PropertySource, RemoveLabelOperator,
    ScanOperator, SetPropertyOperator, ShortestPathOperator, SimpleAggregateOperator, SkipOperator,
    SortDirection, SortKey as PhysicalSortKey, SortOperator, UnaryFilterOp, UnionOperator,
    UnwindOperator, VariableLengthExpandOperator,
};
use grafeo_core::graph::{Direction, lpg::LpgStore};
use std::collections::HashMap;
use std::sync::Arc;

use crate::transaction::TransactionManager;

/// Range bounds for property-based range queries.
struct RangeBounds<'a> {
    min: Option<&'a Value>,
    max: Option<&'a Value>,
    min_inclusive: bool,
    max_inclusive: bool,
}

/// Converts a logical plan to a physical operator tree.
pub struct Planner {
    /// The graph store to scan from.
    store: Arc<LpgStore>,
    /// Transaction manager for MVCC operations.
    tx_manager: Option<Arc<TransactionManager>>,
    /// Current transaction ID (if in a transaction).
    tx_id: Option<TxId>,
    /// Epoch to use for visibility checks.
    viewing_epoch: EpochId,
    /// Counter for generating unique anonymous edge column names.
    anon_edge_counter: std::cell::Cell<u32>,
    /// Whether to use factorized execution for multi-hop queries.
    factorized_execution: bool,
}

impl Planner {
    /// Creates a new planner with the given store.
    ///
    /// This creates a planner without transaction context, using the current
    /// epoch from the store for visibility.
    #[must_use]
    pub fn new(store: Arc<LpgStore>) -> Self {
        let epoch = store.current_epoch();
        Self {
            store,
            tx_manager: None,
            tx_id: None,
            viewing_epoch: epoch,
            anon_edge_counter: std::cell::Cell::new(0),
            factorized_execution: true,
        }
    }

    /// Creates a new planner with transaction context for MVCC-aware planning.
    ///
    /// # Arguments
    ///
    /// * `store` - The graph store
    /// * `tx_manager` - Transaction manager for recording reads/writes
    /// * `tx_id` - Current transaction ID (None for auto-commit)
    /// * `viewing_epoch` - Epoch to use for version visibility
    #[must_use]
    pub fn with_context(
        store: Arc<LpgStore>,
        tx_manager: Arc<TransactionManager>,
        tx_id: Option<TxId>,
        viewing_epoch: EpochId,
    ) -> Self {
        Self {
            store,
            tx_manager: Some(tx_manager),
            tx_id,
            viewing_epoch,
            anon_edge_counter: std::cell::Cell::new(0),
            factorized_execution: true,
        }
    }

    /// Returns the viewing epoch for this planner.
    #[must_use]
    pub fn viewing_epoch(&self) -> EpochId {
        self.viewing_epoch
    }

    /// Returns the transaction ID for this planner, if any.
    #[must_use]
    pub fn tx_id(&self) -> Option<TxId> {
        self.tx_id
    }

    /// Returns a reference to the transaction manager, if available.
    #[must_use]
    pub fn tx_manager(&self) -> Option<&Arc<TransactionManager>> {
        self.tx_manager.as_ref()
    }

    /// Enables or disables factorized execution for multi-hop queries.
    #[must_use]
    pub fn with_factorized_execution(mut self, enabled: bool) -> Self {
        self.factorized_execution = enabled;
        self
    }

    /// Counts consecutive single-hop expand operations.
    ///
    /// Returns the count and the deepest non-expand operator (the base of the chain).
    fn count_expand_chain(op: &LogicalOperator) -> (usize, &LogicalOperator) {
        match op {
            LogicalOperator::Expand(expand) => {
                // Only count single-hop expands (factorization doesn't apply to variable-length)
                let is_single_hop = expand.min_hops == 1 && expand.max_hops == Some(1);

                if is_single_hop {
                    let (inner_count, base) = Self::count_expand_chain(&expand.input);
                    (inner_count + 1, base)
                } else {
                    // Variable-length path breaks the chain
                    (0, op)
                }
            }
            _ => (0, op),
        }
    }

    /// Collects expand operations from the outermost down to the base.
    ///
    /// Returns expands in order from innermost (base) to outermost.
    fn collect_expand_chain(op: &LogicalOperator) -> Vec<&ExpandOp> {
        let mut chain = Vec::new();
        let mut current = op;

        while let LogicalOperator::Expand(expand) = current {
            // Only include single-hop expands
            let is_single_hop = expand.min_hops == 1 && expand.max_hops == Some(1);
            if !is_single_hop {
                break;
            }
            chain.push(expand);
            current = &expand.input;
        }

        // Reverse so we go from base to outer
        chain.reverse();
        chain
    }

    /// Plans a logical plan into a physical operator.
    ///
    /// # Errors
    ///
    /// Returns an error if planning fails.
    pub fn plan(&self, logical_plan: &LogicalPlan) -> Result<PhysicalPlan> {
        let (operator, columns) = self.plan_operator(&logical_plan.root)?;
        Ok(PhysicalPlan {
            operator,
            columns,
            adaptive_context: None,
        })
    }

    /// Plans a logical plan with adaptive execution support.
    ///
    /// Creates cardinality checkpoints at key points in the plan (scans, filters,
    /// joins) that can be monitored during execution to detect estimate errors.
    ///
    /// # Errors
    ///
    /// Returns an error if planning fails.
    pub fn plan_adaptive(&self, logical_plan: &LogicalPlan) -> Result<PhysicalPlan> {
        let (operator, columns) = self.plan_operator(&logical_plan.root)?;

        // Build adaptive context with cardinality estimates
        let mut adaptive_context = AdaptiveContext::new();
        self.collect_cardinality_estimates(&logical_plan.root, &mut adaptive_context, 0);

        Ok(PhysicalPlan {
            operator,
            columns,
            adaptive_context: Some(adaptive_context),
        })
    }

    /// Collects cardinality estimates from the logical plan into an adaptive context.
    fn collect_cardinality_estimates(
        &self,
        op: &LogicalOperator,
        ctx: &mut AdaptiveContext,
        depth: usize,
    ) {
        match op {
            LogicalOperator::NodeScan(scan) => {
                // Estimate based on label statistics
                let estimate = if let Some(label) = &scan.label {
                    self.store.nodes_by_label(label).len() as f64
                } else {
                    self.store.node_count() as f64
                };
                let id = format!("scan_{}", scan.variable);
                ctx.set_estimate(&id, estimate);

                // Recurse into input if present
                if let Some(input) = &scan.input {
                    self.collect_cardinality_estimates(input, ctx, depth + 1);
                }
            }
            LogicalOperator::Filter(filter) => {
                // Default selectivity estimate for filters (30%)
                let input_estimate = self.estimate_cardinality(&filter.input);
                let estimate = input_estimate * 0.3;
                let id = format!("filter_{depth}");
                ctx.set_estimate(&id, estimate);

                self.collect_cardinality_estimates(&filter.input, ctx, depth + 1);
            }
            LogicalOperator::Expand(expand) => {
                // Estimate based on average degree from store statistics
                let input_estimate = self.estimate_cardinality(&expand.input);
                let stats = self.store.statistics();
                let avg_degree = self.estimate_expand_degree(&stats, expand);
                let estimate = input_estimate * avg_degree;
                let id = format!("expand_{}", expand.to_variable);
                ctx.set_estimate(&id, estimate);

                self.collect_cardinality_estimates(&expand.input, ctx, depth + 1);
            }
            LogicalOperator::Join(join) => {
                // Estimate join output (product with selectivity)
                let left_est = self.estimate_cardinality(&join.left);
                let right_est = self.estimate_cardinality(&join.right);
                let estimate = (left_est * right_est).sqrt(); // Geometric mean as rough estimate
                let id = format!("join_{depth}");
                ctx.set_estimate(&id, estimate);

                self.collect_cardinality_estimates(&join.left, ctx, depth + 1);
                self.collect_cardinality_estimates(&join.right, ctx, depth + 1);
            }
            LogicalOperator::Aggregate(agg) => {
                // Aggregates typically reduce cardinality
                let input_estimate = self.estimate_cardinality(&agg.input);
                let estimate = if agg.group_by.is_empty() {
                    1.0 // Scalar aggregate
                } else {
                    (input_estimate * 0.1).max(1.0) // 10% of input as group estimate
                };
                let id = format!("aggregate_{depth}");
                ctx.set_estimate(&id, estimate);

                self.collect_cardinality_estimates(&agg.input, ctx, depth + 1);
            }
            LogicalOperator::Distinct(distinct) => {
                let input_estimate = self.estimate_cardinality(&distinct.input);
                let estimate = (input_estimate * 0.5).max(1.0);
                let id = format!("distinct_{depth}");
                ctx.set_estimate(&id, estimate);

                self.collect_cardinality_estimates(&distinct.input, ctx, depth + 1);
            }
            LogicalOperator::Return(ret) => {
                self.collect_cardinality_estimates(&ret.input, ctx, depth + 1);
            }
            LogicalOperator::Limit(limit) => {
                let input_estimate = self.estimate_cardinality(&limit.input);
                let estimate = (input_estimate).min(limit.count as f64);
                let id = format!("limit_{depth}");
                ctx.set_estimate(&id, estimate);

                self.collect_cardinality_estimates(&limit.input, ctx, depth + 1);
            }
            LogicalOperator::Skip(skip) => {
                let input_estimate = self.estimate_cardinality(&skip.input);
                let estimate = (input_estimate - skip.count as f64).max(0.0);
                let id = format!("skip_{depth}");
                ctx.set_estimate(&id, estimate);

                self.collect_cardinality_estimates(&skip.input, ctx, depth + 1);
            }
            LogicalOperator::Sort(sort) => {
                // Sort doesn't change cardinality
                self.collect_cardinality_estimates(&sort.input, ctx, depth + 1);
            }
            LogicalOperator::Union(union) => {
                let estimate: f64 = union
                    .inputs
                    .iter()
                    .map(|input| self.estimate_cardinality(input))
                    .sum();
                let id = format!("union_{depth}");
                ctx.set_estimate(&id, estimate);

                for input in &union.inputs {
                    self.collect_cardinality_estimates(input, ctx, depth + 1);
                }
            }
            _ => {
                // For other operators, try to recurse into known input patterns
            }
        }
    }

    /// Estimates cardinality for a logical operator subtree.
    fn estimate_cardinality(&self, op: &LogicalOperator) -> f64 {
        match op {
            LogicalOperator::NodeScan(scan) => {
                if let Some(label) = &scan.label {
                    self.store.nodes_by_label(label).len() as f64
                } else {
                    self.store.node_count() as f64
                }
            }
            LogicalOperator::Filter(filter) => self.estimate_cardinality(&filter.input) * 0.3,
            LogicalOperator::Expand(expand) => {
                let stats = self.store.statistics();
                let avg_degree = self.estimate_expand_degree(&stats, expand);
                self.estimate_cardinality(&expand.input) * avg_degree
            }
            LogicalOperator::Join(join) => {
                let left = self.estimate_cardinality(&join.left);
                let right = self.estimate_cardinality(&join.right);
                (left * right).sqrt()
            }
            LogicalOperator::Aggregate(agg) => {
                if agg.group_by.is_empty() {
                    1.0
                } else {
                    (self.estimate_cardinality(&agg.input) * 0.1).max(1.0)
                }
            }
            LogicalOperator::Distinct(distinct) => {
                (self.estimate_cardinality(&distinct.input) * 0.5).max(1.0)
            }
            LogicalOperator::Return(ret) => self.estimate_cardinality(&ret.input),
            LogicalOperator::Limit(limit) => self
                .estimate_cardinality(&limit.input)
                .min(limit.count as f64),
            LogicalOperator::Skip(skip) => {
                (self.estimate_cardinality(&skip.input) - skip.count as f64).max(0.0)
            }
            LogicalOperator::Sort(sort) => self.estimate_cardinality(&sort.input),
            LogicalOperator::Union(union) => union
                .inputs
                .iter()
                .map(|input| self.estimate_cardinality(input))
                .sum(),
            _ => 1000.0, // Default estimate for unknown operators
        }
    }

    /// Estimates the average edge degree for an expand operation using store statistics.
    fn estimate_expand_degree(
        &self,
        stats: &grafeo_core::statistics::Statistics,
        expand: &ExpandOp,
    ) -> f64 {
        let outgoing = !matches!(expand.direction, ExpandDirection::Incoming);
        if let Some(edge_type) = &expand.edge_type {
            stats.estimate_avg_degree(edge_type, outgoing)
        } else if stats.total_nodes > 0 {
            (stats.total_edges as f64 / stats.total_nodes as f64).max(1.0)
        } else {
            10.0 // fallback for empty graph
        }
    }

    /// Plans a single logical operator.
    fn plan_operator(&self, op: &LogicalOperator) -> Result<(Box<dyn Operator>, Vec<String>)> {
        match op {
            LogicalOperator::NodeScan(scan) => self.plan_node_scan(scan),
            LogicalOperator::Expand(expand) => {
                // Check for expand chains when factorized execution is enabled
                if self.factorized_execution {
                    let (chain_len, _base) = Self::count_expand_chain(op);
                    if chain_len >= 2 {
                        // Use factorized chain for 2+ consecutive single-hop expands
                        return self.plan_expand_chain(op);
                    }
                }
                self.plan_expand(expand)
            }
            LogicalOperator::Return(ret) => self.plan_return(ret),
            LogicalOperator::Filter(filter) => self.plan_filter(filter),
            LogicalOperator::Project(project) => self.plan_project(project),
            LogicalOperator::Limit(limit) => self.plan_limit(limit),
            LogicalOperator::Skip(skip) => self.plan_skip(skip),
            LogicalOperator::Sort(sort) => self.plan_sort(sort),
            LogicalOperator::Aggregate(agg) => self.plan_aggregate(agg),
            LogicalOperator::Join(join) => self.plan_join(join),
            LogicalOperator::Union(union) => self.plan_union(union),
            LogicalOperator::Distinct(distinct) => self.plan_distinct(distinct),
            LogicalOperator::CreateNode(create) => self.plan_create_node(create),
            LogicalOperator::CreateEdge(create) => self.plan_create_edge(create),
            LogicalOperator::DeleteNode(delete) => self.plan_delete_node(delete),
            LogicalOperator::DeleteEdge(delete) => self.plan_delete_edge(delete),
            LogicalOperator::LeftJoin(left_join) => self.plan_left_join(left_join),
            LogicalOperator::AntiJoin(anti_join) => self.plan_anti_join(anti_join),
            LogicalOperator::Unwind(unwind) => self.plan_unwind(unwind),
            LogicalOperator::Merge(merge) => self.plan_merge(merge),
            LogicalOperator::AddLabel(add_label) => self.plan_add_label(add_label),
            LogicalOperator::RemoveLabel(remove_label) => self.plan_remove_label(remove_label),
            LogicalOperator::SetProperty(set_prop) => self.plan_set_property(set_prop),
            LogicalOperator::ShortestPath(sp) => self.plan_shortest_path(sp),
            LogicalOperator::Empty => Err(Error::Internal("Empty plan".to_string())),
            LogicalOperator::VectorScan(_) => Err(Error::Internal(
                "VectorScan requires vector-index feature".to_string(),
            )),
            LogicalOperator::VectorJoin(_) => Err(Error::Internal(
                "VectorJoin requires vector-index feature".to_string(),
            )),
            _ => Err(Error::Internal(format!(
                "Unsupported operator: {:?}",
                std::mem::discriminant(op)
            ))),
        }
    }

    /// Plans a node scan operator.
    fn plan_node_scan(&self, scan: &NodeScanOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let scan_op = if let Some(label) = &scan.label {
            ScanOperator::with_label(Arc::clone(&self.store), label)
        } else {
            ScanOperator::new(Arc::clone(&self.store))
        };

        // Apply MVCC context if available
        let scan_operator: Box<dyn Operator> =
            Box::new(scan_op.with_tx_context(self.viewing_epoch, self.tx_id));

        // If there's an input, chain operators with a nested loop join (cross join)
        if let Some(input) = &scan.input {
            let (input_op, mut input_columns) = self.plan_operator(input)?;

            // Build output schema: input columns + scan column
            let mut output_schema: Vec<LogicalType> =
                input_columns.iter().map(|_| LogicalType::Any).collect();
            output_schema.push(LogicalType::Node);

            // Add scan column to input columns
            input_columns.push(scan.variable.clone());

            // Use nested loop join to combine input rows with scanned nodes
            let join_op = Box::new(NestedLoopJoinOperator::new(
                input_op,
                scan_operator,
                None, // No join condition (cross join)
                PhysicalJoinType::Cross,
                output_schema,
            ));

            Ok((join_op, input_columns))
        } else {
            let columns = vec![scan.variable.clone()];
            Ok((scan_operator, columns))
        }
    }

    /// Plans an expand operator.
    fn plan_expand(&self, expand: &ExpandOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Plan the input operator first
        let (input_op, input_columns) = self.plan_operator(&expand.input)?;

        // Find the source column index
        let source_column = input_columns
            .iter()
            .position(|c| c == &expand.from_variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Source variable '{}' not found in input columns",
                    expand.from_variable
                ))
            })?;

        // Convert expand direction
        let direction = match expand.direction {
            ExpandDirection::Outgoing => Direction::Outgoing,
            ExpandDirection::Incoming => Direction::Incoming,
            ExpandDirection::Both => Direction::Both,
        };

        // Check if this is a variable-length path
        let is_variable_length =
            expand.min_hops != 1 || expand.max_hops.is_none() || expand.max_hops != Some(1);

        let operator: Box<dyn Operator> = if is_variable_length {
            // Use VariableLengthExpandOperator for multi-hop paths
            let max_hops = expand.max_hops.unwrap_or(expand.min_hops + 10); // Default max if unlimited
            let mut expand_op = VariableLengthExpandOperator::new(
                Arc::clone(&self.store),
                input_op,
                source_column,
                direction,
                expand.edge_type.clone(),
                expand.min_hops,
                max_hops,
            )
            .with_tx_context(self.viewing_epoch, self.tx_id);

            // If a path alias is set, enable path length output
            if expand.path_alias.is_some() {
                expand_op = expand_op.with_path_length_output();
            }

            Box::new(expand_op)
        } else {
            // Use simple ExpandOperator for single-hop paths
            let expand_op = ExpandOperator::new(
                Arc::clone(&self.store),
                input_op,
                source_column,
                direction,
                expand.edge_type.clone(),
            )
            .with_tx_context(self.viewing_epoch, self.tx_id);
            Box::new(expand_op)
        };

        // Build output columns: [input_columns..., edge, target, (path_length)?]
        // Preserve all input columns and add edge + target to match ExpandOperator output
        let mut columns = input_columns;

        // Generate edge column name - use provided name or generate anonymous name
        let edge_col_name = expand.edge_variable.clone().unwrap_or_else(|| {
            let count = self.anon_edge_counter.get();
            self.anon_edge_counter.set(count + 1);
            format!("_anon_edge_{}", count)
        });
        columns.push(edge_col_name);

        columns.push(expand.to_variable.clone());

        // If a path alias is set, add a column for the path length
        if let Some(ref path_alias) = expand.path_alias {
            columns.push(format!("_path_length_{}", path_alias));
        }

        Ok((operator, columns))
    }

    /// Plans a chain of consecutive expand operations using factorized execution.
    ///
    /// This avoids the Cartesian product explosion that occurs with separate expands.
    /// For a 2-hop query with degree d, this uses O(d) memory instead of O(d^2).
    ///
    /// The chain is executed lazily at query time, not during planning. This ensures
    /// that any filters applied above the expand chain are properly respected.
    fn plan_expand_chain(&self, op: &LogicalOperator) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let expands = Self::collect_expand_chain(op);
        if expands.is_empty() {
            return Err(Error::Internal("Empty expand chain".to_string()));
        }

        // Get the base operator (before first expand)
        let first_expand = expands[0];
        let (base_op, base_columns) = self.plan_operator(&first_expand.input)?;

        let mut columns = base_columns.clone();
        let mut steps = Vec::new();

        // Track the level-local source column for each expand
        // For the first expand, it's the column in the input (base_columns)
        // For subsequent expands, the target from the previous level is always at index 1
        // (each level adds [edge, target], so target is at index 1)
        let mut is_first = true;

        for expand in &expands {
            // Find source column for this expand
            let source_column = if is_first {
                // For first expand, find in base columns
                base_columns
                    .iter()
                    .position(|c| c == &expand.from_variable)
                    .ok_or_else(|| {
                        Error::Internal(format!(
                            "Source variable '{}' not found in base columns",
                            expand.from_variable
                        ))
                    })?
            } else {
                // For subsequent expands, the target from the previous level is at index 1
                // (each level adds [edge, target], so target is the second column)
                1
            };

            // Convert direction
            let direction = match expand.direction {
                ExpandDirection::Outgoing => Direction::Outgoing,
                ExpandDirection::Incoming => Direction::Incoming,
                ExpandDirection::Both => Direction::Both,
            };

            // Add expand step configuration
            steps.push(ExpandStep {
                source_column,
                direction,
                edge_type: expand.edge_type.clone(),
            });

            // Add edge and target columns
            let edge_col_name = expand.edge_variable.clone().unwrap_or_else(|| {
                let count = self.anon_edge_counter.get();
                self.anon_edge_counter.set(count + 1);
                format!("_anon_edge_{}", count)
            });
            columns.push(edge_col_name);
            columns.push(expand.to_variable.clone());

            is_first = false;
        }

        // Create lazy operator that executes at query time, not planning time
        let mut lazy_op = LazyFactorizedChainOperator::new(Arc::clone(&self.store), base_op, steps);

        if let Some(tx_id) = self.tx_id {
            lazy_op = lazy_op.with_tx_context(self.viewing_epoch, Some(tx_id));
        } else {
            lazy_op = lazy_op.with_tx_context(self.viewing_epoch, None);
        }

        Ok((Box::new(lazy_op), columns))
    }

    /// Plans a RETURN clause.
    fn plan_return(&self, ret: &ReturnOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Plan the input operator
        let (input_op, input_columns) = self.plan_operator(&ret.input)?;

        // Build variable to column index mapping
        let variable_columns: HashMap<String, usize> = input_columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        // Extract column names from return items
        let columns: Vec<String> = ret
            .items
            .iter()
            .map(|item| {
                item.alias.clone().unwrap_or_else(|| {
                    // Generate a default name from the expression
                    expression_to_string(&item.expression)
                })
            })
            .collect();

        // Check if we need a project operator (for property access or expression evaluation)
        let needs_project = ret
            .items
            .iter()
            .any(|item| !matches!(&item.expression, LogicalExpression::Variable(_)));

        if needs_project {
            // Build project expressions
            let mut projections = Vec::with_capacity(ret.items.len());
            let mut output_types = Vec::with_capacity(ret.items.len());

            for item in &ret.items {
                match &item.expression {
                    LogicalExpression::Variable(name) => {
                        let col_idx = *variable_columns.get(name).ok_or_else(|| {
                            Error::Internal(format!("Variable '{}' not found in input", name))
                        })?;
                        projections.push(ProjectExpr::Column(col_idx));
                        // Use Node type for variables (they could be nodes, edges, or values)
                        output_types.push(LogicalType::Node);
                    }
                    LogicalExpression::Property { variable, property } => {
                        let col_idx = *variable_columns.get(variable).ok_or_else(|| {
                            Error::Internal(format!("Variable '{}' not found in input", variable))
                        })?;
                        projections.push(ProjectExpr::PropertyAccess {
                            column: col_idx,
                            property: property.clone(),
                        });
                        // Property could be any type - use Any/Generic to preserve type
                        output_types.push(LogicalType::Any);
                    }
                    LogicalExpression::Literal(value) => {
                        projections.push(ProjectExpr::Constant(value.clone()));
                        output_types.push(value_to_logical_type(value));
                    }
                    LogicalExpression::FunctionCall { name, args, .. } => {
                        // Handle built-in functions
                        match name.to_lowercase().as_str() {
                            "type" => {
                                // type(r) returns the edge type string
                                if args.len() != 1 {
                                    return Err(Error::Internal(
                                        "type() requires exactly one argument".to_string(),
                                    ));
                                }
                                if let LogicalExpression::Variable(var_name) = &args[0] {
                                    let col_idx =
                                        *variable_columns.get(var_name).ok_or_else(|| {
                                            Error::Internal(format!(
                                                "Variable '{}' not found in input",
                                                var_name
                                            ))
                                        })?;
                                    projections.push(ProjectExpr::EdgeType { column: col_idx });
                                    output_types.push(LogicalType::String);
                                } else {
                                    return Err(Error::Internal(
                                        "type() argument must be a variable".to_string(),
                                    ));
                                }
                            }
                            "length" => {
                                // length(p) returns the path length
                                // For shortestPath results, the path column already contains the length
                                if args.len() != 1 {
                                    return Err(Error::Internal(
                                        "length() requires exactly one argument".to_string(),
                                    ));
                                }
                                if let LogicalExpression::Variable(var_name) = &args[0] {
                                    let col_idx =
                                        *variable_columns.get(var_name).ok_or_else(|| {
                                            Error::Internal(format!(
                                                "Variable '{}' not found in input",
                                                var_name
                                            ))
                                        })?;
                                    // Pass through the column value directly
                                    projections.push(ProjectExpr::Column(col_idx));
                                    output_types.push(LogicalType::Int64);
                                } else {
                                    return Err(Error::Internal(
                                        "length() argument must be a variable".to_string(),
                                    ));
                                }
                            }
                            // For other functions (head, tail, size, etc.), use expression evaluation
                            _ => {
                                let filter_expr = self.convert_expression(&item.expression)?;
                                projections.push(ProjectExpr::Expression {
                                    expr: filter_expr,
                                    variable_columns: variable_columns.clone(),
                                });
                                output_types.push(LogicalType::Any);
                            }
                        }
                    }
                    LogicalExpression::Case { .. } => {
                        // Convert CASE expression to FilterExpression for evaluation
                        let filter_expr = self.convert_expression(&item.expression)?;
                        projections.push(ProjectExpr::Expression {
                            expr: filter_expr,
                            variable_columns: variable_columns.clone(),
                        });
                        // CASE can return any type - use Any
                        output_types.push(LogicalType::Any);
                    }
                    _ => {
                        return Err(Error::Internal(format!(
                            "Unsupported RETURN expression: {:?}",
                            item.expression
                        )));
                    }
                }
            }

            let operator = Box::new(ProjectOperator::with_store(
                input_op,
                projections,
                output_types,
                Arc::clone(&self.store),
            ));

            Ok((operator, columns))
        } else {
            // Simple case: just return variables
            // Re-order columns to match return items if needed
            let mut projections = Vec::with_capacity(ret.items.len());
            let mut output_types = Vec::with_capacity(ret.items.len());

            for item in &ret.items {
                if let LogicalExpression::Variable(name) = &item.expression {
                    let col_idx = *variable_columns.get(name).ok_or_else(|| {
                        Error::Internal(format!("Variable '{}' not found in input", name))
                    })?;
                    projections.push(ProjectExpr::Column(col_idx));
                    output_types.push(LogicalType::Node);
                }
            }

            // Only add ProjectOperator if reordering is needed
            if projections.len() == input_columns.len()
                && projections
                    .iter()
                    .enumerate()
                    .all(|(i, p)| matches!(p, ProjectExpr::Column(c) if *c == i))
            {
                // No reordering needed
                Ok((input_op, columns))
            } else {
                let operator = Box::new(ProjectOperator::new(input_op, projections, output_types));
                Ok((operator, columns))
            }
        }
    }

    /// Plans a project operator (for WITH clause).
    fn plan_project(
        &self,
        project: &crate::query::plan::ProjectOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Handle Empty input specially (standalone WITH like: WITH [1,2,3] AS nums)
        let (input_op, input_columns): (Box<dyn Operator>, Vec<String>) =
            if matches!(project.input.as_ref(), LogicalOperator::Empty) {
                // Create a single-row operator for projecting literals
                let single_row_op: Box<dyn Operator> = Box::new(
                    grafeo_core::execution::operators::single_row::SingleRowOperator::new(),
                );
                (single_row_op, Vec::new())
            } else {
                self.plan_operator(&project.input)?
            };

        // Build variable to column index mapping
        let variable_columns: HashMap<String, usize> = input_columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        // Build projections and new column names
        let mut projections = Vec::with_capacity(project.projections.len());
        let mut output_types = Vec::with_capacity(project.projections.len());
        let mut output_columns = Vec::with_capacity(project.projections.len());

        for projection in &project.projections {
            // Determine the output column name (alias or expression string)
            let col_name = projection
                .alias
                .clone()
                .unwrap_or_else(|| expression_to_string(&projection.expression));
            output_columns.push(col_name);

            match &projection.expression {
                LogicalExpression::Variable(name) => {
                    let col_idx = *variable_columns.get(name).ok_or_else(|| {
                        Error::Internal(format!("Variable '{}' not found in input", name))
                    })?;
                    projections.push(ProjectExpr::Column(col_idx));
                    output_types.push(LogicalType::Node);
                }
                LogicalExpression::Property { variable, property } => {
                    let col_idx = *variable_columns.get(variable).ok_or_else(|| {
                        Error::Internal(format!("Variable '{}' not found in input", variable))
                    })?;
                    projections.push(ProjectExpr::PropertyAccess {
                        column: col_idx,
                        property: property.clone(),
                    });
                    output_types.push(LogicalType::Any);
                }
                LogicalExpression::Literal(value) => {
                    projections.push(ProjectExpr::Constant(value.clone()));
                    output_types.push(value_to_logical_type(value));
                }
                _ => {
                    // For complex expressions, use full expression evaluation
                    let filter_expr = self.convert_expression(&projection.expression)?;
                    projections.push(ProjectExpr::Expression {
                        expr: filter_expr,
                        variable_columns: variable_columns.clone(),
                    });
                    output_types.push(LogicalType::Any);
                }
            }
        }

        let operator = Box::new(ProjectOperator::with_store(
            input_op,
            projections,
            output_types,
            Arc::clone(&self.store),
        ));

        Ok((operator, output_columns))
    }

    /// Plans a filter operator.
    ///
    /// Uses zone map pre-filtering to potentially skip scans when predicates
    /// definitely won't match any data. Also uses property indexes when available
    /// for O(1) lookups instead of full scans.
    fn plan_filter(&self, filter: &FilterOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Check zone maps for simple property predicates before scanning
        // If zone map says "definitely no matches", we can short-circuit
        if let Some(false) = self.check_zone_map_for_predicate(&filter.predicate) {
            // Zone map says no matches possible - return empty result
            let (_, columns) = self.plan_operator(&filter.input)?;
            let schema = self.derive_schema_from_columns(&columns);
            let empty_op = Box::new(EmptyOperator::new(schema));
            return Ok((empty_op, columns));
        }

        // Try to use property index for equality predicates on indexed properties
        if let Some(result) = self.try_plan_filter_with_property_index(filter)? {
            return Ok(result);
        }

        // Try to use range optimization for range predicates (>, <, >=, <=)
        if let Some(result) = self.try_plan_filter_with_range_index(filter)? {
            return Ok(result);
        }

        // Plan the input operator first
        let (input_op, columns) = self.plan_operator(&filter.input)?;

        // Build variable to column index mapping
        let variable_columns: HashMap<String, usize> = columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        // Convert logical expression to filter expression
        let filter_expr = self.convert_expression(&filter.predicate)?;

        // Create the predicate
        let predicate =
            ExpressionPredicate::new(filter_expr, variable_columns, Arc::clone(&self.store));

        // Create the filter operator
        let operator = Box::new(FilterOperator::new(input_op, Box::new(predicate)));

        Ok((operator, columns))
    }

    /// Checks zone maps for a predicate to see if we can skip the scan entirely.
    ///
    /// Returns:
    /// - `Some(false)` if zone map proves no matches possible (can skip)
    /// - `Some(true)` if zone map says matches might exist
    /// - `None` if zone map check not applicable
    fn check_zone_map_for_predicate(&self, predicate: &LogicalExpression) -> Option<bool> {
        use grafeo_core::graph::lpg::CompareOp;

        match predicate {
            LogicalExpression::Binary { left, op, right } => {
                // Check for AND/OR first (compound conditions)
                match op {
                    BinaryOp::And => {
                        let left_result = self.check_zone_map_for_predicate(left);
                        let right_result = self.check_zone_map_for_predicate(right);

                        return match (left_result, right_result) {
                            // If either side definitely won't match, the AND won't match
                            (Some(false), _) | (_, Some(false)) => Some(false),
                            // If both might match, might match overall
                            (Some(true), Some(true)) => Some(true),
                            // Otherwise, can't determine
                            _ => None,
                        };
                    }
                    BinaryOp::Or => {
                        let left_result = self.check_zone_map_for_predicate(left);
                        let right_result = self.check_zone_map_for_predicate(right);

                        return match (left_result, right_result) {
                            // Both sides definitely won't match
                            (Some(false), Some(false)) => Some(false),
                            // At least one side might match
                            (Some(true), _) | (_, Some(true)) => Some(true),
                            // Otherwise, can't determine
                            _ => None,
                        };
                    }
                    _ => {}
                }

                // Simple property comparison: n.property op value
                let (property, compare_op, value) = match (left.as_ref(), right.as_ref()) {
                    (
                        LogicalExpression::Property { property, .. },
                        LogicalExpression::Literal(val),
                    ) => {
                        let cmp = match op {
                            BinaryOp::Eq => CompareOp::Eq,
                            BinaryOp::Ne => CompareOp::Ne,
                            BinaryOp::Lt => CompareOp::Lt,
                            BinaryOp::Le => CompareOp::Le,
                            BinaryOp::Gt => CompareOp::Gt,
                            BinaryOp::Ge => CompareOp::Ge,
                            _ => return None,
                        };
                        (property.clone(), cmp, val.clone())
                    }
                    (
                        LogicalExpression::Literal(val),
                        LogicalExpression::Property { property, .. },
                    ) => {
                        // Flip comparison for reversed operands
                        let cmp = match op {
                            BinaryOp::Eq => CompareOp::Eq,
                            BinaryOp::Ne => CompareOp::Ne,
                            BinaryOp::Lt => CompareOp::Gt, // val < prop means prop > val
                            BinaryOp::Le => CompareOp::Ge,
                            BinaryOp::Gt => CompareOp::Lt,
                            BinaryOp::Ge => CompareOp::Le,
                            _ => return None,
                        };
                        (property.clone(), cmp, val.clone())
                    }
                    _ => return None,
                };

                // Check zone map for node properties
                let might_match =
                    self.store
                        .node_property_might_match(&property.into(), compare_op, &value);

                Some(might_match)
            }

            _ => None,
        }
    }

    /// Tries to use a property index for filter optimization.
    ///
    /// When a filter predicate is an equality check on an indexed property,
    /// and the input is a simple NodeScan, we can use the index to look up
    /// matching nodes directly instead of scanning all nodes.
    ///
    /// Returns `Ok(Some((operator, columns)))` if optimization was applied,
    /// `Ok(None)` if not applicable, or `Err` on error.
    fn try_plan_filter_with_property_index(
        &self,
        filter: &FilterOp,
    ) -> Result<Option<(Box<dyn Operator>, Vec<String>)>> {
        // Only optimize if input is a simple NodeScan (not nested)
        let (scan_variable, scan_label) = match filter.input.as_ref() {
            LogicalOperator::NodeScan(scan) if scan.input.is_none() => {
                (scan.variable.clone(), scan.label.clone())
            }
            _ => return Ok(None),
        };

        // Extract property equality conditions from the predicate
        // Handles both simple (n.prop = val) and compound (n.a = 1 AND n.b = 2)
        let conditions = self.extract_equality_conditions(&filter.predicate, &scan_variable);

        if conditions.is_empty() {
            return Ok(None);
        }

        // Check if at least one condition has an index (otherwise full scan is needed anyway)
        let has_indexed_condition = conditions
            .iter()
            .any(|(prop, _)| self.store.has_property_index(prop));

        if !has_indexed_condition {
            return Ok(None);
        }

        // Use the optimized batch lookup for multiple conditions
        let conditions_ref: Vec<(&str, Value)> = conditions
            .iter()
            .map(|(p, v)| (p.as_str(), v.clone()))
            .collect();
        let mut matching_nodes = self.store.find_nodes_by_properties(&conditions_ref);

        // If there's a label filter, also filter by label
        if let Some(label) = &scan_label {
            let label_nodes: std::collections::HashSet<_> =
                self.store.nodes_by_label(label).into_iter().collect();
            matching_nodes.retain(|n| label_nodes.contains(n));
        }

        // Create a NodeListOperator with the matching nodes
        let node_list_op = Box::new(NodeListOperator::new(matching_nodes, 2048));
        let columns = vec![scan_variable];

        Ok(Some((node_list_op, columns)))
    }

    /// Extracts equality conditions (property = literal) from a predicate.
    ///
    /// Handles both simple predicates and AND chains:
    /// - `n.name = "Alice"` → `[("name", "Alice")]`
    /// - `n.name = "Alice" AND n.age = 30` → `[("name", "Alice"), ("age", 30)]`
    fn extract_equality_conditions(
        &self,
        predicate: &LogicalExpression,
        target_variable: &str,
    ) -> Vec<(String, Value)> {
        let mut conditions = Vec::new();
        self.collect_equality_conditions(predicate, target_variable, &mut conditions);
        conditions
    }

    /// Recursively collects equality conditions from AND expressions.
    fn collect_equality_conditions(
        &self,
        expr: &LogicalExpression,
        target_variable: &str,
        conditions: &mut Vec<(String, Value)>,
    ) {
        match expr {
            // Handle AND: recurse into both sides
            LogicalExpression::Binary {
                left,
                op: BinaryOp::And,
                right,
            } => {
                self.collect_equality_conditions(left, target_variable, conditions);
                self.collect_equality_conditions(right, target_variable, conditions);
            }

            // Handle equality: extract property and value
            LogicalExpression::Binary {
                left,
                op: BinaryOp::Eq,
                right,
            } => {
                if let Some((var, prop, val)) = self.extract_property_equality(left, right)
                    && var == target_variable
                {
                    conditions.push((prop, val));
                }
            }

            _ => {}
        }
    }

    /// Extracts (variable, property, value) from a property equality expression.
    fn extract_property_equality(
        &self,
        left: &LogicalExpression,
        right: &LogicalExpression,
    ) -> Option<(String, String, Value)> {
        match (left, right) {
            (
                LogicalExpression::Property { variable, property },
                LogicalExpression::Literal(val),
            ) => Some((variable.clone(), property.clone(), val.clone())),
            (
                LogicalExpression::Literal(val),
                LogicalExpression::Property { variable, property },
            ) => Some((variable.clone(), property.clone(), val.clone())),
            _ => None,
        }
    }

    /// Tries to optimize a filter using range queries on properties.
    ///
    /// This optimization is applied when:
    /// - The input is a simple NodeScan (no nested operations)
    /// - The predicate contains range comparisons (>, <, >=, <=)
    /// - The same variable and property are being filtered
    ///
    /// Handles both simple range predicates (`n.age > 30`) and BETWEEN patterns
    /// (`n.age >= 30 AND n.age <= 50`).
    ///
    /// Returns `Ok(Some((operator, columns)))` if optimization was applied,
    /// `Ok(None)` if not applicable, or `Err` on error.
    fn try_plan_filter_with_range_index(
        &self,
        filter: &FilterOp,
    ) -> Result<Option<(Box<dyn Operator>, Vec<String>)>> {
        // Only optimize if input is a simple NodeScan (not nested)
        let (scan_variable, scan_label) = match filter.input.as_ref() {
            LogicalOperator::NodeScan(scan) if scan.input.is_none() => {
                (scan.variable.clone(), scan.label.clone())
            }
            _ => return Ok(None),
        };

        // Try to extract BETWEEN pattern first (more efficient)
        if let Some((variable, property, min, max, min_inc, max_inc)) =
            self.extract_between_predicate(&filter.predicate)
            && variable == scan_variable
        {
            return self.plan_range_filter(
                &scan_variable,
                &scan_label,
                &property,
                RangeBounds {
                    min: Some(&min),
                    max: Some(&max),
                    min_inclusive: min_inc,
                    max_inclusive: max_inc,
                },
            );
        }

        // Try to extract simple range predicate
        if let Some((variable, property, op, value)) =
            self.extract_range_predicate(&filter.predicate)
            && variable == scan_variable
        {
            let (min, max, min_inc, max_inc) = match op {
                BinaryOp::Lt => (None, Some(value), false, false),
                BinaryOp::Le => (None, Some(value), false, true),
                BinaryOp::Gt => (Some(value), None, false, false),
                BinaryOp::Ge => (Some(value), None, true, false),
                _ => return Ok(None),
            };
            return self.plan_range_filter(
                &scan_variable,
                &scan_label,
                &property,
                RangeBounds {
                    min: min.as_ref(),
                    max: max.as_ref(),
                    min_inclusive: min_inc,
                    max_inclusive: max_inc,
                },
            );
        }

        Ok(None)
    }

    /// Plans a range filter using `find_nodes_in_range`.
    fn plan_range_filter(
        &self,
        scan_variable: &str,
        scan_label: &Option<String>,
        property: &str,
        bounds: RangeBounds<'_>,
    ) -> Result<Option<(Box<dyn Operator>, Vec<String>)>> {
        // Use the store's range query method
        let mut matching_nodes = self.store.find_nodes_in_range(
            property,
            bounds.min,
            bounds.max,
            bounds.min_inclusive,
            bounds.max_inclusive,
        );

        // If there's a label filter, also filter by label
        if let Some(label) = scan_label {
            let label_nodes: std::collections::HashSet<_> =
                self.store.nodes_by_label(label).into_iter().collect();
            matching_nodes.retain(|n| label_nodes.contains(n));
        }

        // Create a NodeListOperator with the matching nodes
        let node_list_op = Box::new(NodeListOperator::new(matching_nodes, 2048));
        let columns = vec![scan_variable.to_string()];

        Ok(Some((node_list_op, columns)))
    }

    /// Extracts a simple range predicate (>, <, >=, <=) from an expression.
    ///
    /// Returns `(variable, property, operator, value)` if found.
    fn extract_range_predicate(
        &self,
        predicate: &LogicalExpression,
    ) -> Option<(String, String, BinaryOp, Value)> {
        match predicate {
            LogicalExpression::Binary { left, op, right } => {
                match op {
                    BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
                        // Try property on left: n.age > 30
                        if let (
                            LogicalExpression::Property { variable, property },
                            LogicalExpression::Literal(val),
                        ) = (left.as_ref(), right.as_ref())
                        {
                            return Some((variable.clone(), property.clone(), *op, val.clone()));
                        }

                        // Try property on right: 30 < n.age (flip operator)
                        if let (
                            LogicalExpression::Literal(val),
                            LogicalExpression::Property { variable, property },
                        ) = (left.as_ref(), right.as_ref())
                        {
                            let flipped_op = match op {
                                BinaryOp::Lt => BinaryOp::Gt,
                                BinaryOp::Le => BinaryOp::Ge,
                                BinaryOp::Gt => BinaryOp::Lt,
                                BinaryOp::Ge => BinaryOp::Le,
                                _ => return None,
                            };
                            return Some((
                                variable.clone(),
                                property.clone(),
                                flipped_op,
                                val.clone(),
                            ));
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        None
    }

    /// Extracts a BETWEEN pattern from compound predicates.
    ///
    /// Recognizes patterns like:
    /// - `n.age >= 30 AND n.age <= 50`
    /// - `n.age > 30 AND n.age < 50`
    ///
    /// Returns `(variable, property, min_value, max_value, min_inclusive, max_inclusive)`.
    fn extract_between_predicate(
        &self,
        predicate: &LogicalExpression,
    ) -> Option<(String, String, Value, Value, bool, bool)> {
        // Must be an AND expression
        let (left, right) = match predicate {
            LogicalExpression::Binary {
                left,
                op: BinaryOp::And,
                right,
            } => (left.as_ref(), right.as_ref()),
            _ => return None,
        };

        // Extract range predicates from both sides
        let left_range = self.extract_range_predicate(left);
        let right_range = self.extract_range_predicate(right);

        let (left_var, left_prop, left_op, left_val) = left_range?;
        let (right_var, right_prop, right_op, right_val) = right_range?;

        // Must be same variable and property
        if left_var != right_var || left_prop != right_prop {
            return None;
        }

        // Determine which is lower bound and which is upper bound
        let (min_val, max_val, min_inc, max_inc) = match (left_op, right_op) {
            // n.x >= min AND n.x <= max
            (BinaryOp::Ge, BinaryOp::Le) => (left_val, right_val, true, true),
            // n.x >= min AND n.x < max
            (BinaryOp::Ge, BinaryOp::Lt) => (left_val, right_val, true, false),
            // n.x > min AND n.x <= max
            (BinaryOp::Gt, BinaryOp::Le) => (left_val, right_val, false, true),
            // n.x > min AND n.x < max
            (BinaryOp::Gt, BinaryOp::Lt) => (left_val, right_val, false, false),
            // Reversed order: n.x <= max AND n.x >= min
            (BinaryOp::Le, BinaryOp::Ge) => (right_val, left_val, true, true),
            // n.x < max AND n.x >= min
            (BinaryOp::Lt, BinaryOp::Ge) => (right_val, left_val, true, false),
            // n.x <= max AND n.x > min
            (BinaryOp::Le, BinaryOp::Gt) => (right_val, left_val, false, true),
            // n.x < max AND n.x > min
            (BinaryOp::Lt, BinaryOp::Gt) => (right_val, left_val, false, false),
            _ => return None,
        };

        Some((left_var, left_prop, min_val, max_val, min_inc, max_inc))
    }

    /// Plans a LIMIT operator.
    fn plan_limit(&self, limit: &LimitOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, columns) = self.plan_operator(&limit.input)?;
        let output_schema = self.derive_schema_from_columns(&columns);
        let operator = Box::new(LimitOperator::new(input_op, limit.count, output_schema));
        Ok((operator, columns))
    }

    /// Plans a SKIP operator.
    fn plan_skip(&self, skip: &SkipOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, columns) = self.plan_operator(&skip.input)?;
        let output_schema = self.derive_schema_from_columns(&columns);
        let operator = Box::new(SkipOperator::new(input_op, skip.count, output_schema));
        Ok((operator, columns))
    }

    /// Plans a SORT (ORDER BY) operator.
    fn plan_sort(&self, sort: &SortOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (mut input_op, input_columns) = self.plan_operator(&sort.input)?;

        // Build variable to column index mapping
        let mut variable_columns: HashMap<String, usize> = input_columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        // Collect property expressions that need to be projected before sorting
        let mut property_projections: Vec<(String, String, String)> = Vec::new();
        let mut next_col_idx = input_columns.len();

        for key in &sort.keys {
            if let LogicalExpression::Property { variable, property } = &key.expression {
                let col_name = format!("{}_{}", variable, property);
                if !variable_columns.contains_key(&col_name) {
                    property_projections.push((
                        variable.clone(),
                        property.clone(),
                        col_name.clone(),
                    ));
                    variable_columns.insert(col_name, next_col_idx);
                    next_col_idx += 1;
                }
            }
        }

        // Track output columns
        let mut output_columns = input_columns.clone();

        // If we have property expressions, add a projection to materialize them
        if !property_projections.is_empty() {
            let mut projections = Vec::new();
            let mut output_types = Vec::new();

            // First, pass through all existing columns (use Node type to preserve node IDs
            // for subsequent property access - nodes need VectorData::NodeId for get_node_id())
            for (i, _) in input_columns.iter().enumerate() {
                projections.push(ProjectExpr::Column(i));
                output_types.push(LogicalType::Node);
            }

            // Then add property access projections
            for (variable, property, col_name) in &property_projections {
                let source_col = *variable_columns.get(variable).ok_or_else(|| {
                    Error::Internal(format!(
                        "Variable '{}' not found for ORDER BY property projection",
                        variable
                    ))
                })?;
                projections.push(ProjectExpr::PropertyAccess {
                    column: source_col,
                    property: property.clone(),
                });
                output_types.push(LogicalType::Any);
                output_columns.push(col_name.clone());
            }

            input_op = Box::new(ProjectOperator::with_store(
                input_op,
                projections,
                output_types,
                Arc::clone(&self.store),
            ));
        }

        // Convert logical sort keys to physical sort keys
        let physical_keys: Vec<PhysicalSortKey> = sort
            .keys
            .iter()
            .map(|key| {
                let col_idx = self
                    .resolve_sort_expression_with_properties(&key.expression, &variable_columns)?;
                Ok(PhysicalSortKey {
                    column: col_idx,
                    direction: match key.order {
                        SortOrder::Ascending => SortDirection::Ascending,
                        SortOrder::Descending => SortDirection::Descending,
                    },
                    null_order: NullOrder::NullsLast,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let output_schema = self.derive_schema_from_columns(&output_columns);
        let operator = Box::new(SortOperator::new(input_op, physical_keys, output_schema));
        Ok((operator, output_columns))
    }

    /// Resolves a sort expression to a column index, using projected property columns.
    fn resolve_sort_expression_with_properties(
        &self,
        expr: &LogicalExpression,
        variable_columns: &HashMap<String, usize>,
    ) -> Result<usize> {
        match expr {
            LogicalExpression::Variable(name) => {
                variable_columns.get(name).copied().ok_or_else(|| {
                    Error::Internal(format!("Variable '{}' not found for ORDER BY", name))
                })
            }
            LogicalExpression::Property { variable, property } => {
                // Look up the projected property column (e.g., "p_age" for p.age)
                let col_name = format!("{}_{}", variable, property);
                variable_columns.get(&col_name).copied().ok_or_else(|| {
                    Error::Internal(format!(
                        "Property column '{}' not found for ORDER BY (from {}.{})",
                        col_name, variable, property
                    ))
                })
            }
            _ => Err(Error::Internal(format!(
                "Unsupported ORDER BY expression: {:?}",
                expr
            ))),
        }
    }

    /// Derives a schema from column names (uses Any type to handle all value types).
    fn derive_schema_from_columns(&self, columns: &[String]) -> Vec<LogicalType> {
        columns.iter().map(|_| LogicalType::Any).collect()
    }

    /// Plans an AGGREGATE operator.
    fn plan_aggregate(&self, agg: &AggregateOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Check if we can use factorized aggregation for speedup
        // Conditions:
        // 1. Factorized execution is enabled
        // 2. Input is an expand chain (multi-hop)
        // 3. No GROUP BY
        // 4. All aggregates are simple (COUNT, SUM, AVG, MIN, MAX)
        if self.factorized_execution
            && agg.group_by.is_empty()
            && Self::count_expand_chain(&agg.input).0 >= 2
            && self.is_simple_aggregate(agg)
            && let Ok((op, cols)) = self.plan_factorized_aggregate(agg)
        {
            return Ok((op, cols));
        }
        // Fall through to regular aggregate if factorized planning fails

        let (mut input_op, input_columns) = self.plan_operator(&agg.input)?;

        // Build variable to column index mapping
        let mut variable_columns: HashMap<String, usize> = input_columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        // Collect all property expressions that need to be projected before aggregation
        let mut property_projections: Vec<(String, String, String)> = Vec::new(); // (variable, property, new_column_name)
        let mut next_col_idx = input_columns.len();

        // Check group-by expressions for properties
        for expr in &agg.group_by {
            if let LogicalExpression::Property { variable, property } = expr {
                let col_name = format!("{}_{}", variable, property);
                if !variable_columns.contains_key(&col_name) {
                    property_projections.push((
                        variable.clone(),
                        property.clone(),
                        col_name.clone(),
                    ));
                    variable_columns.insert(col_name, next_col_idx);
                    next_col_idx += 1;
                }
            }
        }

        // Check aggregate expressions for properties
        for agg_expr in &agg.aggregates {
            if let Some(LogicalExpression::Property { variable, property }) = &agg_expr.expression {
                let col_name = format!("{}_{}", variable, property);
                if !variable_columns.contains_key(&col_name) {
                    property_projections.push((
                        variable.clone(),
                        property.clone(),
                        col_name.clone(),
                    ));
                    variable_columns.insert(col_name, next_col_idx);
                    next_col_idx += 1;
                }
            }
        }

        // If we have property expressions, add a projection to materialize them
        if !property_projections.is_empty() {
            let mut projections = Vec::new();
            let mut output_types = Vec::new();

            // First, pass through all existing columns (use Node type to preserve node IDs
            // for subsequent property access - nodes need VectorData::NodeId for get_node_id())
            for (i, _) in input_columns.iter().enumerate() {
                projections.push(ProjectExpr::Column(i));
                output_types.push(LogicalType::Node);
            }

            // Then add property access projections
            for (variable, property, _col_name) in &property_projections {
                let source_col = *variable_columns.get(variable).ok_or_else(|| {
                    Error::Internal(format!(
                        "Variable '{}' not found for property projection",
                        variable
                    ))
                })?;
                projections.push(ProjectExpr::PropertyAccess {
                    column: source_col,
                    property: property.clone(),
                });
                output_types.push(LogicalType::Any); // Properties can be any type (string, int, etc.)
            }

            input_op = Box::new(ProjectOperator::with_store(
                input_op,
                projections,
                output_types,
                Arc::clone(&self.store),
            ));
        }

        // Convert group-by expressions to column indices
        let group_columns: Vec<usize> = agg
            .group_by
            .iter()
            .map(|expr| self.resolve_expression_to_column_with_properties(expr, &variable_columns))
            .collect::<Result<Vec<_>>>()?;

        // Convert aggregate expressions to physical form
        let physical_aggregates: Vec<PhysicalAggregateExpr> = agg
            .aggregates
            .iter()
            .map(|agg_expr| {
                let column = agg_expr
                    .expression
                    .as_ref()
                    .map(|e| {
                        self.resolve_expression_to_column_with_properties(e, &variable_columns)
                    })
                    .transpose()?;

                Ok(PhysicalAggregateExpr {
                    function: convert_aggregate_function(agg_expr.function),
                    column,
                    distinct: agg_expr.distinct,
                    alias: agg_expr.alias.clone(),
                    percentile: agg_expr.percentile,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // Build output schema and column names
        let mut output_schema = Vec::new();
        let mut output_columns = Vec::new();

        // Add group-by columns
        for expr in &agg.group_by {
            output_schema.push(LogicalType::Any); // Group-by values can be any type
            output_columns.push(expression_to_string(expr));
        }

        // Add aggregate result columns
        for agg_expr in &agg.aggregates {
            let result_type = match agg_expr.function {
                LogicalAggregateFunction::Count | LogicalAggregateFunction::CountNonNull => {
                    LogicalType::Int64
                }
                LogicalAggregateFunction::Sum => LogicalType::Int64,
                LogicalAggregateFunction::Avg => LogicalType::Float64,
                LogicalAggregateFunction::Min | LogicalAggregateFunction::Max => {
                    // MIN/MAX preserve input type; use Int64 as default for numeric comparisons
                    // since the aggregate can return any Value type, but the most common case
                    // is numeric values from property expressions
                    LogicalType::Int64
                }
                LogicalAggregateFunction::Collect => LogicalType::Any, // List type (using Any since List is a complex type)
                // Statistical functions return Float64
                LogicalAggregateFunction::StdDev
                | LogicalAggregateFunction::StdDevPop
                | LogicalAggregateFunction::PercentileDisc
                | LogicalAggregateFunction::PercentileCont => LogicalType::Float64,
            };
            output_schema.push(result_type);
            output_columns.push(
                agg_expr
                    .alias
                    .clone()
                    .unwrap_or_else(|| format!("{:?}(...)", agg_expr.function).to_lowercase()),
            );
        }

        // Choose operator based on whether there are group-by columns
        let mut operator: Box<dyn Operator> = if group_columns.is_empty() {
            Box::new(SimpleAggregateOperator::new(
                input_op,
                physical_aggregates,
                output_schema,
            ))
        } else {
            Box::new(HashAggregateOperator::new(
                input_op,
                group_columns,
                physical_aggregates,
                output_schema,
            ))
        };

        // Apply HAVING clause filter if present
        if let Some(having_expr) = &agg.having {
            // Build variable to column mapping for the aggregate output
            let having_var_columns: HashMap<String, usize> = output_columns
                .iter()
                .enumerate()
                .map(|(i, name)| (name.clone(), i))
                .collect();

            let filter_expr = self.convert_expression(having_expr)?;
            let predicate =
                ExpressionPredicate::new(filter_expr, having_var_columns, Arc::clone(&self.store));
            operator = Box::new(FilterOperator::new(operator, Box::new(predicate)));
        }

        Ok((operator, output_columns))
    }

    /// Checks if an aggregate is simple enough for factorized execution.
    ///
    /// Simple aggregates:
    /// - COUNT(*) or COUNT(variable)
    /// - SUM, AVG, MIN, MAX on variables (not properties for now)
    fn is_simple_aggregate(&self, agg: &AggregateOp) -> bool {
        agg.aggregates.iter().all(|agg_expr| {
            match agg_expr.function {
                LogicalAggregateFunction::Count | LogicalAggregateFunction::CountNonNull => {
                    // COUNT(*) is always OK, COUNT(var) is OK
                    agg_expr.expression.is_none()
                        || matches!(&agg_expr.expression, Some(LogicalExpression::Variable(_)))
                }
                LogicalAggregateFunction::Sum
                | LogicalAggregateFunction::Avg
                | LogicalAggregateFunction::Min
                | LogicalAggregateFunction::Max => {
                    // For now, only support when expression is a variable
                    // (property access would require flattening first)
                    matches!(&agg_expr.expression, Some(LogicalExpression::Variable(_)))
                }
                // Other aggregates (Collect, StdDev, Percentile) not supported in factorized form
                _ => false,
            }
        })
    }

    /// Plans a factorized aggregate that operates directly on factorized data.
    ///
    /// This avoids the O(n²) cost of flattening before aggregation.
    fn plan_factorized_aggregate(
        &self,
        agg: &AggregateOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Build the expand chain - this returns a LazyFactorizedChainOperator
        let expands = Self::collect_expand_chain(&agg.input);
        if expands.is_empty() {
            return Err(Error::Internal(
                "Expected expand chain for factorized aggregate".to_string(),
            ));
        }

        // Get the base operator (before first expand)
        let first_expand = expands[0];
        let (base_op, base_columns) = self.plan_operator(&first_expand.input)?;

        let mut columns = base_columns.clone();
        let mut steps = Vec::new();
        let mut is_first = true;

        for expand in &expands {
            // Find source column for this expand
            let source_column = if is_first {
                base_columns
                    .iter()
                    .position(|c| c == &expand.from_variable)
                    .ok_or_else(|| {
                        Error::Internal(format!(
                            "Source variable '{}' not found in base columns",
                            expand.from_variable
                        ))
                    })?
            } else {
                1 // Target from previous level
            };

            let direction = match expand.direction {
                ExpandDirection::Outgoing => Direction::Outgoing,
                ExpandDirection::Incoming => Direction::Incoming,
                ExpandDirection::Both => Direction::Both,
            };

            steps.push(ExpandStep {
                source_column,
                direction,
                edge_type: expand.edge_type.clone(),
            });

            let edge_col_name = expand.edge_variable.clone().unwrap_or_else(|| {
                let count = self.anon_edge_counter.get();
                self.anon_edge_counter.set(count + 1);
                format!("_anon_edge_{}", count)
            });
            columns.push(edge_col_name);
            columns.push(expand.to_variable.clone());

            is_first = false;
        }

        // Create the lazy factorized chain operator
        let mut lazy_op = LazyFactorizedChainOperator::new(Arc::clone(&self.store), base_op, steps);

        if let Some(tx_id) = self.tx_id {
            lazy_op = lazy_op.with_tx_context(self.viewing_epoch, Some(tx_id));
        } else {
            lazy_op = lazy_op.with_tx_context(self.viewing_epoch, None);
        }

        // Convert logical aggregates to factorized aggregates
        let factorized_aggs: Vec<FactorizedAggregate> = agg
            .aggregates
            .iter()
            .map(|agg_expr| {
                match agg_expr.function {
                    LogicalAggregateFunction::Count | LogicalAggregateFunction::CountNonNull => {
                        // COUNT(*) uses simple count, COUNT(col) uses column count
                        if agg_expr.expression.is_none() {
                            FactorizedAggregate::count()
                        } else {
                            // For COUNT(variable), we use the deepest level's target column
                            // which is the last column added to the schema
                            FactorizedAggregate::count_column(1) // Target is at index 1 in deepest level
                        }
                    }
                    LogicalAggregateFunction::Sum => {
                        // SUM on deepest level target
                        FactorizedAggregate::sum(1)
                    }
                    LogicalAggregateFunction::Avg => FactorizedAggregate::avg(1),
                    LogicalAggregateFunction::Min => FactorizedAggregate::min(1),
                    LogicalAggregateFunction::Max => FactorizedAggregate::max(1),
                    _ => {
                        // Shouldn't reach here due to is_simple_aggregate check
                        FactorizedAggregate::count()
                    }
                }
            })
            .collect();

        // Build output column names
        let output_columns: Vec<String> = agg
            .aggregates
            .iter()
            .map(|agg_expr| {
                agg_expr
                    .alias
                    .clone()
                    .unwrap_or_else(|| format!("{:?}(...)", agg_expr.function).to_lowercase())
            })
            .collect();

        // Create the factorized aggregate operator
        let factorized_agg_op = FactorizedAggregateOperator::new(lazy_op, factorized_aggs);

        Ok((Box::new(factorized_agg_op), output_columns))
    }

    /// Resolves a logical expression to a column index.
    #[allow(dead_code)]
    fn resolve_expression_to_column(
        &self,
        expr: &LogicalExpression,
        variable_columns: &HashMap<String, usize>,
    ) -> Result<usize> {
        match expr {
            LogicalExpression::Variable(name) => variable_columns
                .get(name)
                .copied()
                .ok_or_else(|| Error::Internal(format!("Variable '{}' not found", name))),
            LogicalExpression::Property { variable, .. } => variable_columns
                .get(variable)
                .copied()
                .ok_or_else(|| Error::Internal(format!("Variable '{}' not found", variable))),
            _ => Err(Error::Internal(format!(
                "Cannot resolve expression to column: {:?}",
                expr
            ))),
        }
    }

    /// Resolves a logical expression to a column index, using projected property columns.
    ///
    /// This is used for aggregations where properties have been projected into their own columns.
    fn resolve_expression_to_column_with_properties(
        &self,
        expr: &LogicalExpression,
        variable_columns: &HashMap<String, usize>,
    ) -> Result<usize> {
        match expr {
            LogicalExpression::Variable(name) => variable_columns
                .get(name)
                .copied()
                .ok_or_else(|| Error::Internal(format!("Variable '{}' not found", name))),
            LogicalExpression::Property { variable, property } => {
                // Look up the projected property column (e.g., "p_price" for p.price)
                let col_name = format!("{}_{}", variable, property);
                variable_columns.get(&col_name).copied().ok_or_else(|| {
                    Error::Internal(format!(
                        "Property column '{}' not found (from {}.{})",
                        col_name, variable, property
                    ))
                })
            }
            _ => Err(Error::Internal(format!(
                "Cannot resolve expression to column: {:?}",
                expr
            ))),
        }
    }

    /// Converts a logical expression to a filter expression.
    fn convert_expression(&self, expr: &LogicalExpression) -> Result<FilterExpression> {
        match expr {
            LogicalExpression::Literal(v) => Ok(FilterExpression::Literal(v.clone())),
            LogicalExpression::Variable(name) => Ok(FilterExpression::Variable(name.clone())),
            LogicalExpression::Property { variable, property } => Ok(FilterExpression::Property {
                variable: variable.clone(),
                property: property.clone(),
            }),
            LogicalExpression::Binary { left, op, right } => {
                let left_expr = self.convert_expression(left)?;
                let right_expr = self.convert_expression(right)?;
                let filter_op = convert_binary_op(*op)?;
                Ok(FilterExpression::Binary {
                    left: Box::new(left_expr),
                    op: filter_op,
                    right: Box::new(right_expr),
                })
            }
            LogicalExpression::Unary { op, operand } => {
                let operand_expr = self.convert_expression(operand)?;
                let filter_op = convert_unary_op(*op)?;
                Ok(FilterExpression::Unary {
                    op: filter_op,
                    operand: Box::new(operand_expr),
                })
            }
            LogicalExpression::FunctionCall { name, args, .. } => {
                let filter_args: Vec<FilterExpression> = args
                    .iter()
                    .map(|a| self.convert_expression(a))
                    .collect::<Result<Vec<_>>>()?;
                Ok(FilterExpression::FunctionCall {
                    name: name.clone(),
                    args: filter_args,
                })
            }
            LogicalExpression::Case {
                operand,
                when_clauses,
                else_clause,
            } => {
                let filter_operand = operand
                    .as_ref()
                    .map(|e| self.convert_expression(e))
                    .transpose()?
                    .map(Box::new);
                let filter_when_clauses: Vec<(FilterExpression, FilterExpression)> = when_clauses
                    .iter()
                    .map(|(cond, result)| {
                        Ok((
                            self.convert_expression(cond)?,
                            self.convert_expression(result)?,
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let filter_else = else_clause
                    .as_ref()
                    .map(|e| self.convert_expression(e))
                    .transpose()?
                    .map(Box::new);
                Ok(FilterExpression::Case {
                    operand: filter_operand,
                    when_clauses: filter_when_clauses,
                    else_clause: filter_else,
                })
            }
            LogicalExpression::List(items) => {
                let filter_items: Vec<FilterExpression> = items
                    .iter()
                    .map(|item| self.convert_expression(item))
                    .collect::<Result<Vec<_>>>()?;
                Ok(FilterExpression::List(filter_items))
            }
            LogicalExpression::Map(pairs) => {
                let filter_pairs: Vec<(String, FilterExpression)> = pairs
                    .iter()
                    .map(|(k, v)| Ok((k.clone(), self.convert_expression(v)?)))
                    .collect::<Result<Vec<_>>>()?;
                Ok(FilterExpression::Map(filter_pairs))
            }
            LogicalExpression::IndexAccess { base, index } => {
                let base_expr = self.convert_expression(base)?;
                let index_expr = self.convert_expression(index)?;
                Ok(FilterExpression::IndexAccess {
                    base: Box::new(base_expr),
                    index: Box::new(index_expr),
                })
            }
            LogicalExpression::SliceAccess { base, start, end } => {
                let base_expr = self.convert_expression(base)?;
                let start_expr = start
                    .as_ref()
                    .map(|s| self.convert_expression(s))
                    .transpose()?
                    .map(Box::new);
                let end_expr = end
                    .as_ref()
                    .map(|e| self.convert_expression(e))
                    .transpose()?
                    .map(Box::new);
                Ok(FilterExpression::SliceAccess {
                    base: Box::new(base_expr),
                    start: start_expr,
                    end: end_expr,
                })
            }
            LogicalExpression::Parameter(_) => Err(Error::Internal(
                "Parameters not yet supported in filters".to_string(),
            )),
            LogicalExpression::Labels(var) => Ok(FilterExpression::Labels(var.clone())),
            LogicalExpression::Type(var) => Ok(FilterExpression::Type(var.clone())),
            LogicalExpression::Id(var) => Ok(FilterExpression::Id(var.clone())),
            LogicalExpression::ListComprehension {
                variable,
                list_expr,
                filter_expr,
                map_expr,
            } => {
                let list = self.convert_expression(list_expr)?;
                let filter = filter_expr
                    .as_ref()
                    .map(|f| self.convert_expression(f))
                    .transpose()?
                    .map(Box::new);
                let map = self.convert_expression(map_expr)?;
                Ok(FilterExpression::ListComprehension {
                    variable: variable.clone(),
                    list_expr: Box::new(list),
                    filter_expr: filter,
                    map_expr: Box::new(map),
                })
            }
            LogicalExpression::ExistsSubquery(subplan) => {
                // Extract the pattern from the subplan
                // For EXISTS { MATCH (n)-[:TYPE]->() }, we extract start_var, direction, edge_type
                let (start_var, direction, edge_type, end_labels) =
                    self.extract_exists_pattern(subplan)?;

                Ok(FilterExpression::ExistsSubquery {
                    start_var,
                    direction,
                    edge_type,
                    end_labels,
                    min_hops: None,
                    max_hops: None,
                })
            }
            LogicalExpression::CountSubquery(_) => Err(Error::Internal(
                "COUNT subqueries not yet supported".to_string(),
            )),
        }
    }

    /// Extracts the pattern from an EXISTS subplan.
    /// Returns (start_variable, direction, edge_type, end_labels).
    fn extract_exists_pattern(
        &self,
        subplan: &LogicalOperator,
    ) -> Result<(String, Direction, Option<String>, Option<Vec<String>>)> {
        match subplan {
            LogicalOperator::Expand(expand) => {
                // Get end node labels from the to_variable if there's a node scan input
                let end_labels = self.extract_end_labels_from_expand(expand);
                let direction = match expand.direction {
                    ExpandDirection::Outgoing => Direction::Outgoing,
                    ExpandDirection::Incoming => Direction::Incoming,
                    ExpandDirection::Both => Direction::Both,
                };
                Ok((
                    expand.from_variable.clone(),
                    direction,
                    expand.edge_type.clone(),
                    end_labels,
                ))
            }
            LogicalOperator::NodeScan(scan) => {
                if let Some(input) = &scan.input {
                    self.extract_exists_pattern(input)
                } else {
                    Err(Error::Internal(
                        "EXISTS subquery must contain an edge pattern".to_string(),
                    ))
                }
            }
            LogicalOperator::Filter(filter) => self.extract_exists_pattern(&filter.input),
            _ => Err(Error::Internal(
                "Unsupported EXISTS subquery pattern".to_string(),
            )),
        }
    }

    /// Extracts end node labels from an Expand operator if present.
    fn extract_end_labels_from_expand(&self, expand: &ExpandOp) -> Option<Vec<String>> {
        // Check if the expand has a NodeScan input with a label filter
        match expand.input.as_ref() {
            LogicalOperator::NodeScan(scan) => scan.label.clone().map(|l| vec![l]),
            _ => None,
        }
    }

    /// Plans a JOIN operator.
    fn plan_join(&self, join: &JoinOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (left_op, left_columns) = self.plan_operator(&join.left)?;
        let (right_op, right_columns) = self.plan_operator(&join.right)?;

        // Build combined output columns
        let mut columns = left_columns.clone();
        columns.extend(right_columns.clone());

        // Convert join type
        let physical_join_type = match join.join_type {
            JoinType::Inner => PhysicalJoinType::Inner,
            JoinType::Left => PhysicalJoinType::Left,
            JoinType::Right => PhysicalJoinType::Right,
            JoinType::Full => PhysicalJoinType::Full,
            JoinType::Cross => PhysicalJoinType::Cross,
            JoinType::Semi => PhysicalJoinType::Semi,
            JoinType::Anti => PhysicalJoinType::Anti,
        };

        // Build key columns from join conditions
        let (probe_keys, build_keys): (Vec<usize>, Vec<usize>) = if join.conditions.is_empty() {
            // Cross join - no keys
            (vec![], vec![])
        } else {
            join.conditions
                .iter()
                .filter_map(|cond| {
                    // Try to extract column indices from expressions
                    let left_idx = self.expression_to_column(&cond.left, &left_columns).ok()?;
                    let right_idx = self
                        .expression_to_column(&cond.right, &right_columns)
                        .ok()?;
                    Some((left_idx, right_idx))
                })
                .unzip()
        };

        let output_schema = self.derive_schema_from_columns(&columns);

        // Check if we should use leapfrog join for cyclic patterns
        // Currently we use hash join by default; leapfrog is available but
        // requires explicit multi-way join detection which will be added
        // when we have proper cyclic pattern detection in the optimizer.
        // For now, LeapfrogJoinOperator is available for direct use.
        let _ = LeapfrogJoinOperator::new; // Suppress unused warning

        let operator: Box<dyn Operator> = Box::new(HashJoinOperator::new(
            left_op,
            right_op,
            probe_keys,
            build_keys,
            physical_join_type,
            output_schema,
        ));

        Ok((operator, columns))
    }

    /// Checks if a join pattern is cyclic (e.g., triangle, clique).
    ///
    /// A cyclic pattern occurs when join conditions reference variables
    /// that create a cycle in the join graph. For example, a triangle
    /// pattern (a)->(b)->(c)->(a) creates a cycle.
    ///
    /// Returns true if the join graph contains at least one cycle with 3+ nodes,
    /// indicating potential for worst-case optimal join (WCOJ) optimization.
    #[allow(dead_code)]
    fn is_cyclic_join_pattern(&self, join: &JoinOp) -> bool {
        // Build adjacency list for join variables
        let mut edges: HashMap<String, Vec<String>> = HashMap::new();
        let mut all_vars: std::collections::HashSet<String> = std::collections::HashSet::new();

        // Collect edges from join conditions
        Self::collect_join_edges(
            &LogicalOperator::Join(join.clone()),
            &mut edges,
            &mut all_vars,
        );

        // Need at least 3 variables to form a cycle
        if all_vars.len() < 3 {
            return false;
        }

        // Detect cycle using DFS with coloring
        Self::has_cycle(&edges, &all_vars)
    }

    /// Collects edges from join conditions into an adjacency list.
    fn collect_join_edges(
        op: &LogicalOperator,
        edges: &mut HashMap<String, Vec<String>>,
        vars: &mut std::collections::HashSet<String>,
    ) {
        match op {
            LogicalOperator::Join(join) => {
                // Process join conditions
                for cond in &join.conditions {
                    if let (Some(left_var), Some(right_var)) = (
                        Self::extract_join_variable(&cond.left),
                        Self::extract_join_variable(&cond.right),
                    ) && left_var != right_var
                    {
                        vars.insert(left_var.clone());
                        vars.insert(right_var.clone());

                        // Add bidirectional edge
                        edges
                            .entry(left_var.clone())
                            .or_default()
                            .push(right_var.clone());
                        edges.entry(right_var).or_default().push(left_var);
                    }
                }

                // Recurse into children
                Self::collect_join_edges(&join.left, edges, vars);
                Self::collect_join_edges(&join.right, edges, vars);
            }
            LogicalOperator::Expand(expand) => {
                // Expand creates implicit join between from_variable and to_variable
                vars.insert(expand.from_variable.clone());
                vars.insert(expand.to_variable.clone());

                edges
                    .entry(expand.from_variable.clone())
                    .or_default()
                    .push(expand.to_variable.clone());
                edges
                    .entry(expand.to_variable.clone())
                    .or_default()
                    .push(expand.from_variable.clone());

                Self::collect_join_edges(&expand.input, edges, vars);
            }
            LogicalOperator::Filter(filter) => {
                Self::collect_join_edges(&filter.input, edges, vars);
            }
            LogicalOperator::NodeScan(scan) => {
                vars.insert(scan.variable.clone());
            }
            _ => {}
        }
    }

    /// Extracts the variable name from a join expression.
    fn extract_join_variable(expr: &LogicalExpression) -> Option<String> {
        match expr {
            LogicalExpression::Variable(v) => Some(v.clone()),
            LogicalExpression::Property { variable, .. } => Some(variable.clone()),
            LogicalExpression::Id(v) => Some(v.clone()),
            _ => None,
        }
    }

    /// Detects if the graph has a cycle using DFS coloring.
    ///
    /// Colors: 0 = white (unvisited), 1 = gray (in progress), 2 = black (done)
    fn has_cycle(
        edges: &HashMap<String, Vec<String>>,
        vars: &std::collections::HashSet<String>,
    ) -> bool {
        let mut color: HashMap<&String, u8> = HashMap::new();

        for var in vars {
            color.insert(var, 0);
        }

        for start in vars {
            if color[start] == 0 && Self::dfs_cycle(start, None, edges, &mut color) {
                return true;
            }
        }

        false
    }

    /// DFS helper for cycle detection.
    fn dfs_cycle(
        node: &String,
        parent: Option<&String>,
        edges: &HashMap<String, Vec<String>>,
        color: &mut HashMap<&String, u8>,
    ) -> bool {
        *color.get_mut(node).unwrap() = 1; // Gray

        if let Some(neighbors) = edges.get(node) {
            for neighbor in neighbors {
                // Skip the edge back to parent (undirected graph)
                if parent == Some(neighbor) {
                    continue;
                }

                if let Some(&c) = color.get(neighbor) {
                    if c == 1 {
                        // Found a back edge - cycle detected
                        return true;
                    }
                    if c == 0 && Self::dfs_cycle(neighbor, Some(node), edges, color) {
                        return true;
                    }
                }
            }
        }

        *color.get_mut(node).unwrap() = 2; // Black
        false
    }

    /// Counts the number of base relations in a logical operator tree.
    #[allow(dead_code)]
    fn count_relations(op: &LogicalOperator) -> usize {
        match op {
            LogicalOperator::NodeScan(_) | LogicalOperator::EdgeScan(_) => 1,
            LogicalOperator::Expand(e) => Self::count_relations(&e.input),
            LogicalOperator::Filter(f) => Self::count_relations(&f.input),
            LogicalOperator::Join(j) => {
                Self::count_relations(&j.left) + Self::count_relations(&j.right)
            }
            _ => 0,
        }
    }

    /// Extracts a column index from an expression.
    fn expression_to_column(&self, expr: &LogicalExpression, columns: &[String]) -> Result<usize> {
        match expr {
            LogicalExpression::Variable(name) => columns
                .iter()
                .position(|c| c == name)
                .ok_or_else(|| Error::Internal(format!("Variable '{}' not found", name))),
            _ => Err(Error::Internal(
                "Only variables supported in join conditions".to_string(),
            )),
        }
    }

    /// Plans a UNION operator.
    fn plan_union(&self, union: &UnionOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        if union.inputs.is_empty() {
            return Err(Error::Internal(
                "Union requires at least one input".to_string(),
            ));
        }

        let mut inputs = Vec::with_capacity(union.inputs.len());
        let mut columns = Vec::new();

        for (i, input) in union.inputs.iter().enumerate() {
            let (op, cols) = self.plan_operator(input)?;
            if i == 0 {
                columns = cols;
            }
            inputs.push(op);
        }

        let output_schema = self.derive_schema_from_columns(&columns);
        let operator = Box::new(UnionOperator::new(inputs, output_schema));

        Ok((operator, columns))
    }

    /// Plans a DISTINCT operator.
    fn plan_distinct(&self, distinct: &DistinctOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, columns) = self.plan_operator(&distinct.input)?;
        let output_schema = self.derive_schema_from_columns(&columns);
        let operator = Box::new(DistinctOperator::new(input_op, output_schema));
        Ok((operator, columns))
    }

    /// Plans a CREATE NODE operator.
    fn plan_create_node(&self, create: &CreateNodeOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Plan input if present
        let (input_op, mut columns) = if let Some(ref input) = create.input {
            let (op, cols) = self.plan_operator(input)?;
            (Some(op), cols)
        } else {
            (None, vec![])
        };

        // Output column for the created node
        let output_column = columns.len();
        columns.push(create.variable.clone());

        // Convert properties (with constant-folding for lists and function calls)
        let properties: Vec<(String, PropertySource)> = create
            .properties
            .iter()
            .map(|(name, expr)| {
                let source = match Self::try_fold_expression(expr) {
                    Some(value) => PropertySource::Constant(value),
                    None => PropertySource::Constant(grafeo_common::types::Value::Null),
                };
                (name.clone(), source)
            })
            .collect();

        let output_schema = self.derive_schema_from_columns(&columns);

        let operator = Box::new(
            CreateNodeOperator::new(
                Arc::clone(&self.store),
                input_op,
                create.labels.clone(),
                properties,
                output_schema,
                output_column,
            )
            .with_tx_context(self.viewing_epoch, self.tx_id),
        );

        Ok((operator, columns))
    }

    /// Plans a CREATE EDGE operator.
    fn plan_create_edge(&self, create: &CreateEdgeOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, mut columns) = self.plan_operator(&create.input)?;

        // Find source and target columns
        let from_column = columns
            .iter()
            .position(|c| c == &create.from_variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Source variable '{}' not found",
                    create.from_variable
                ))
            })?;

        let to_column = columns
            .iter()
            .position(|c| c == &create.to_variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Target variable '{}' not found",
                    create.to_variable
                ))
            })?;

        // Output column for the created edge (if named)
        let output_column = create.variable.as_ref().map(|v| {
            let idx = columns.len();
            columns.push(v.clone());
            idx
        });

        // Convert properties (with constant-folding for function calls like vector())
        let properties: Vec<(String, PropertySource)> = create
            .properties
            .iter()
            .map(|(name, expr)| {
                let source = match Self::try_fold_expression(expr) {
                    Some(value) => PropertySource::Constant(value),
                    None => PropertySource::Constant(grafeo_common::types::Value::Null),
                };
                (name.clone(), source)
            })
            .collect();

        let output_schema = self.derive_schema_from_columns(&columns);

        let mut operator = CreateEdgeOperator::new(
            Arc::clone(&self.store),
            input_op,
            from_column,
            to_column,
            create.edge_type.clone(),
            output_schema,
        )
        .with_properties(properties)
        .with_tx_context(self.viewing_epoch, self.tx_id);

        if let Some(col) = output_column {
            operator = operator.with_output_column(col);
        }

        let operator = Box::new(operator);

        Ok((operator, columns))
    }

    /// Plans a DELETE NODE operator.
    fn plan_delete_node(&self, delete: &DeleteNodeOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, columns) = self.plan_operator(&delete.input)?;

        let node_column = columns
            .iter()
            .position(|c| c == &delete.variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Variable '{}' not found for delete",
                    delete.variable
                ))
            })?;

        // Output schema for delete count
        let output_schema = vec![LogicalType::Int64];
        let output_columns = vec!["deleted_count".to_string()];

        let operator = Box::new(
            DeleteNodeOperator::new(
                Arc::clone(&self.store),
                input_op,
                node_column,
                output_schema,
                delete.detach, // DETACH DELETE deletes connected edges first
            )
            .with_tx_context(self.viewing_epoch, self.tx_id),
        );

        Ok((operator, output_columns))
    }

    /// Plans a DELETE EDGE operator.
    fn plan_delete_edge(&self, delete: &DeleteEdgeOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, columns) = self.plan_operator(&delete.input)?;

        let edge_column = columns
            .iter()
            .position(|c| c == &delete.variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Variable '{}' not found for delete",
                    delete.variable
                ))
            })?;

        // Output schema for delete count
        let output_schema = vec![LogicalType::Int64];
        let output_columns = vec!["deleted_count".to_string()];

        let operator = Box::new(
            DeleteEdgeOperator::new(
                Arc::clone(&self.store),
                input_op,
                edge_column,
                output_schema,
            )
            .with_tx_context(self.viewing_epoch, self.tx_id),
        );

        Ok((operator, output_columns))
    }

    /// Plans a LEFT JOIN operator (for OPTIONAL MATCH).
    fn plan_left_join(&self, left_join: &LeftJoinOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (left_op, left_columns) = self.plan_operator(&left_join.left)?;
        let (right_op, right_columns) = self.plan_operator(&left_join.right)?;

        // Build combined output columns (left + right)
        let mut columns = left_columns.clone();
        columns.extend(right_columns.clone());

        // Find common variables between left and right for join keys
        let mut probe_keys = Vec::new();
        let mut build_keys = Vec::new();

        for (right_idx, right_col) in right_columns.iter().enumerate() {
            if let Some(left_idx) = left_columns.iter().position(|c| c == right_col) {
                probe_keys.push(left_idx);
                build_keys.push(right_idx);
            }
        }

        let output_schema = self.derive_schema_from_columns(&columns);

        let operator: Box<dyn Operator> = Box::new(HashJoinOperator::new(
            left_op,
            right_op,
            probe_keys,
            build_keys,
            PhysicalJoinType::Left,
            output_schema,
        ));

        Ok((operator, columns))
    }

    /// Plans an ANTI JOIN operator (for WHERE NOT EXISTS patterns).
    fn plan_anti_join(&self, anti_join: &AntiJoinOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (left_op, left_columns) = self.plan_operator(&anti_join.left)?;
        let (right_op, right_columns) = self.plan_operator(&anti_join.right)?;

        // Anti-join only keeps left columns (filters out matching rows)
        let columns = left_columns.clone();

        // Find common variables between left and right for join keys
        let mut probe_keys = Vec::new();
        let mut build_keys = Vec::new();

        for (right_idx, right_col) in right_columns.iter().enumerate() {
            if let Some(left_idx) = left_columns.iter().position(|c| c == right_col) {
                probe_keys.push(left_idx);
                build_keys.push(right_idx);
            }
        }

        let output_schema = self.derive_schema_from_columns(&columns);

        let operator: Box<dyn Operator> = Box::new(HashJoinOperator::new(
            left_op,
            right_op,
            probe_keys,
            build_keys,
            PhysicalJoinType::Anti,
            output_schema,
        ));

        Ok((operator, columns))
    }

    /// Plans an unwind operator.
    fn plan_unwind(&self, unwind: &UnwindOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Plan the input operator first
        // Handle Empty specially - use a single-row operator
        let (input_op, input_columns): (Box<dyn Operator>, Vec<String>) =
            if matches!(&*unwind.input, LogicalOperator::Empty) {
                // For UNWIND without prior MATCH, create a single-row input
                // We need an operator that produces one row with the list to unwind
                // For now, use EmptyScan which produces no rows - we'll handle the literal
                // list in the unwind operator itself
                let literal_list = self.convert_expression(&unwind.expression)?;

                // Create a project operator that produces a single row with the list
                let single_row_op: Box<dyn Operator> = Box::new(
                    grafeo_core::execution::operators::single_row::SingleRowOperator::new(),
                );
                let project_op: Box<dyn Operator> = Box::new(ProjectOperator::with_store(
                    single_row_op,
                    vec![ProjectExpr::Expression {
                        expr: literal_list,
                        variable_columns: HashMap::new(),
                    }],
                    vec![LogicalType::Any],
                    Arc::clone(&self.store),
                ));

                (project_op, vec!["__list__".to_string()])
            } else {
                self.plan_operator(&unwind.input)?
            };

        // The UNWIND expression should be a list - we need to find/evaluate it
        // For now, we handle the case where the expression references an existing column
        // or is a literal list

        // Find if the expression references an existing column (like a list property)
        let list_col_idx = match &unwind.expression {
            LogicalExpression::Variable(var) => input_columns.iter().position(|c| c == var),
            LogicalExpression::Property { variable, .. } => {
                // Property access needs to be evaluated - for now we'll need the filter predicate
                // to evaluate this. For simple cases, we treat it as a list column.
                input_columns.iter().position(|c| c == variable)
            }
            LogicalExpression::List(_) | LogicalExpression::Literal(_) => {
                // Literal list expression - we'll add it as a virtual column
                None
            }
            _ => None,
        };

        // Build output columns: all input columns plus the new variable
        let mut columns = input_columns.clone();
        columns.push(unwind.variable.clone());

        // Build output schema
        let mut output_schema = self.derive_schema_from_columns(&input_columns);
        output_schema.push(LogicalType::Any); // The unwound element type is dynamic

        // Use the list column index if found, otherwise default to 0
        // (in which case the first column should contain the list)
        let col_idx = list_col_idx.unwrap_or(0);

        let operator: Box<dyn Operator> = Box::new(UnwindOperator::new(
            input_op,
            col_idx,
            unwind.variable.clone(),
            output_schema,
        ));

        Ok((operator, columns))
    }

    /// Plans a MERGE operator.
    fn plan_merge(&self, merge: &MergeOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Plan the input operator if present (skip if Empty)
        let mut columns = if matches!(merge.input.as_ref(), LogicalOperator::Empty) {
            Vec::new()
        } else {
            let (_input_op, cols) = self.plan_operator(&merge.input)?;
            cols
        };

        // Convert match properties from LogicalExpression to Value
        let match_properties: Vec<(String, grafeo_common::types::Value)> = merge
            .match_properties
            .iter()
            .filter_map(|(name, expr)| {
                if let LogicalExpression::Literal(v) = expr {
                    Some((name.clone(), v.clone()))
                } else {
                    None // Skip non-literal expressions for now
                }
            })
            .collect();

        // Convert ON CREATE properties
        let on_create_properties: Vec<(String, grafeo_common::types::Value)> = merge
            .on_create
            .iter()
            .filter_map(|(name, expr)| {
                if let LogicalExpression::Literal(v) = expr {
                    Some((name.clone(), v.clone()))
                } else {
                    None
                }
            })
            .collect();

        // Convert ON MATCH properties
        let on_match_properties: Vec<(String, grafeo_common::types::Value)> = merge
            .on_match
            .iter()
            .filter_map(|(name, expr)| {
                if let LogicalExpression::Literal(v) = expr {
                    Some((name.clone(), v.clone()))
                } else {
                    None
                }
            })
            .collect();

        // Add the merged node variable to output columns
        columns.push(merge.variable.clone());

        let operator: Box<dyn Operator> = Box::new(MergeOperator::new(
            Arc::clone(&self.store),
            merge.variable.clone(),
            merge.labels.clone(),
            match_properties,
            on_create_properties,
            on_match_properties,
        ));

        Ok((operator, columns))
    }

    /// Plans a SHORTEST PATH operator.
    fn plan_shortest_path(&self, sp: &ShortestPathOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Plan the input operator
        let (input_op, mut columns) = self.plan_operator(&sp.input)?;

        // Find source and target node columns
        let source_column = columns
            .iter()
            .position(|c| c == &sp.source_var)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Source variable '{}' not found for shortestPath",
                    sp.source_var
                ))
            })?;

        let target_column = columns
            .iter()
            .position(|c| c == &sp.target_var)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Target variable '{}' not found for shortestPath",
                    sp.target_var
                ))
            })?;

        // Convert direction
        let direction = match sp.direction {
            ExpandDirection::Outgoing => Direction::Outgoing,
            ExpandDirection::Incoming => Direction::Incoming,
            ExpandDirection::Both => Direction::Both,
        };

        // Create the shortest path operator
        let operator: Box<dyn Operator> = Box::new(
            ShortestPathOperator::new(
                Arc::clone(&self.store),
                input_op,
                source_column,
                target_column,
                sp.edge_type.clone(),
                direction,
            )
            .with_all_paths(sp.all_paths),
        );

        // Add path length column with the expected naming convention
        // The translator expects _path_length_{alias} format for length(p) calls
        columns.push(format!("_path_length_{}", sp.path_alias));

        Ok((operator, columns))
    }

    /// Plans an ADD LABEL operator.
    fn plan_add_label(&self, add_label: &AddLabelOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, columns) = self.plan_operator(&add_label.input)?;

        // Find the node column
        let node_column = columns
            .iter()
            .position(|c| c == &add_label.variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Variable '{}' not found for ADD LABEL",
                    add_label.variable
                ))
            })?;

        // Output schema for update count
        let output_schema = vec![LogicalType::Int64];
        let output_columns = vec!["labels_added".to_string()];

        let operator = Box::new(AddLabelOperator::new(
            Arc::clone(&self.store),
            input_op,
            node_column,
            add_label.labels.clone(),
            output_schema,
        ));

        Ok((operator, output_columns))
    }

    /// Plans a REMOVE LABEL operator.
    fn plan_remove_label(
        &self,
        remove_label: &RemoveLabelOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, columns) = self.plan_operator(&remove_label.input)?;

        // Find the node column
        let node_column = columns
            .iter()
            .position(|c| c == &remove_label.variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Variable '{}' not found for REMOVE LABEL",
                    remove_label.variable
                ))
            })?;

        // Output schema for update count
        let output_schema = vec![LogicalType::Int64];
        let output_columns = vec!["labels_removed".to_string()];

        let operator = Box::new(RemoveLabelOperator::new(
            Arc::clone(&self.store),
            input_op,
            node_column,
            remove_label.labels.clone(),
            output_schema,
        ));

        Ok((operator, output_columns))
    }

    /// Plans a SET PROPERTY operator.
    fn plan_set_property(
        &self,
        set_prop: &SetPropertyOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, columns) = self.plan_operator(&set_prop.input)?;

        // Find the entity column (node or edge variable)
        let entity_column = columns
            .iter()
            .position(|c| c == &set_prop.variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Variable '{}' not found for SET",
                    set_prop.variable
                ))
            })?;

        // Convert properties to PropertySource
        let properties: Vec<(String, PropertySource)> = set_prop
            .properties
            .iter()
            .map(|(name, expr)| {
                let source = self.expression_to_property_source(expr, &columns)?;
                Ok((name.clone(), source))
            })
            .collect::<Result<Vec<_>>>()?;

        // Output schema preserves input schema (passes through)
        let output_schema: Vec<LogicalType> = columns.iter().map(|_| LogicalType::Node).collect();
        let output_columns = columns.clone();

        // Determine if this is a node or edge (for now assume node, edge detection can be added later)
        let operator = Box::new(SetPropertyOperator::new_for_node(
            Arc::clone(&self.store),
            input_op,
            entity_column,
            properties,
            output_schema,
        ));

        Ok((operator, output_columns))
    }

    /// Converts a logical expression to a PropertySource.
    fn expression_to_property_source(
        &self,
        expr: &LogicalExpression,
        columns: &[String],
    ) -> Result<PropertySource> {
        match expr {
            LogicalExpression::Literal(value) => Ok(PropertySource::Constant(value.clone())),
            LogicalExpression::Variable(name) => {
                let col_idx = columns.iter().position(|c| c == name).ok_or_else(|| {
                    Error::Internal(format!("Variable '{}' not found for property source", name))
                })?;
                Ok(PropertySource::Column(col_idx))
            }
            LogicalExpression::Parameter(name) => {
                // Parameters should be resolved before planning
                // For now, treat as a placeholder
                Ok(PropertySource::Constant(
                    grafeo_common::types::Value::String(format!("${}", name).into()),
                ))
            }
            _ => {
                if let Some(value) = Self::try_fold_expression(expr) {
                    Ok(PropertySource::Constant(value))
                } else {
                    Err(Error::Internal(format!(
                        "Unsupported expression type for property source: {:?}",
                        expr
                    )))
                }
            }
        }
    }

    /// Tries to evaluate a constant expression at plan time.
    ///
    /// Recursively folds literals, lists, and known function calls (like `vector()`)
    /// into concrete values. Returns `None` if the expression contains non-constant
    /// parts (variables, property accesses, etc.).
    fn try_fold_expression(expr: &LogicalExpression) -> Option<Value> {
        match expr {
            LogicalExpression::Literal(v) => Some(v.clone()),
            LogicalExpression::List(items) => {
                let values: Option<Vec<Value>> =
                    items.iter().map(Self::try_fold_expression).collect();
                let values = values?;
                // All-numeric lists become vectors (matches Python list[float] behavior)
                let all_numeric = !values.is_empty()
                    && values
                        .iter()
                        .all(|v| matches!(v, Value::Float64(_) | Value::Int64(_)));
                if all_numeric {
                    let floats: Vec<f32> = values
                        .iter()
                        .filter_map(|v| match v {
                            Value::Float64(f) => Some(*f as f32),
                            Value::Int64(i) => Some(*i as f32),
                            _ => None,
                        })
                        .collect();
                    Some(Value::Vector(floats.into()))
                } else {
                    Some(Value::List(values.into()))
                }
            }
            LogicalExpression::FunctionCall { name, args, .. } => {
                match name.to_lowercase().as_str() {
                    "vector" => {
                        if args.len() != 1 {
                            return None;
                        }
                        let val = Self::try_fold_expression(&args[0])?;
                        match val {
                            Value::List(items) => {
                                let floats: Vec<f32> = items
                                    .iter()
                                    .filter_map(|v| match v {
                                        Value::Float64(f) => Some(*f as f32),
                                        Value::Int64(i) => Some(*i as f32),
                                        _ => None,
                                    })
                                    .collect();
                                if floats.len() == items.len() {
                                    Some(Value::Vector(floats.into()))
                                } else {
                                    None
                                }
                            }
                            // Already a vector (from all-numeric list folding)
                            Value::Vector(v) => Some(Value::Vector(v)),
                            _ => None,
                        }
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

/// Converts a logical binary operator to a filter binary operator.
pub fn convert_binary_op(op: BinaryOp) -> Result<BinaryFilterOp> {
    match op {
        BinaryOp::Eq => Ok(BinaryFilterOp::Eq),
        BinaryOp::Ne => Ok(BinaryFilterOp::Ne),
        BinaryOp::Lt => Ok(BinaryFilterOp::Lt),
        BinaryOp::Le => Ok(BinaryFilterOp::Le),
        BinaryOp::Gt => Ok(BinaryFilterOp::Gt),
        BinaryOp::Ge => Ok(BinaryFilterOp::Ge),
        BinaryOp::And => Ok(BinaryFilterOp::And),
        BinaryOp::Or => Ok(BinaryFilterOp::Or),
        BinaryOp::Xor => Ok(BinaryFilterOp::Xor),
        BinaryOp::Add => Ok(BinaryFilterOp::Add),
        BinaryOp::Sub => Ok(BinaryFilterOp::Sub),
        BinaryOp::Mul => Ok(BinaryFilterOp::Mul),
        BinaryOp::Div => Ok(BinaryFilterOp::Div),
        BinaryOp::Mod => Ok(BinaryFilterOp::Mod),
        BinaryOp::StartsWith => Ok(BinaryFilterOp::StartsWith),
        BinaryOp::EndsWith => Ok(BinaryFilterOp::EndsWith),
        BinaryOp::Contains => Ok(BinaryFilterOp::Contains),
        BinaryOp::In => Ok(BinaryFilterOp::In),
        BinaryOp::Regex => Ok(BinaryFilterOp::Regex),
        BinaryOp::Pow => Ok(BinaryFilterOp::Pow),
        BinaryOp::Concat | BinaryOp::Like => Err(Error::Internal(format!(
            "Binary operator {:?} not yet supported in filters",
            op
        ))),
    }
}

/// Converts a logical unary operator to a filter unary operator.
pub fn convert_unary_op(op: UnaryOp) -> Result<UnaryFilterOp> {
    match op {
        UnaryOp::Not => Ok(UnaryFilterOp::Not),
        UnaryOp::IsNull => Ok(UnaryFilterOp::IsNull),
        UnaryOp::IsNotNull => Ok(UnaryFilterOp::IsNotNull),
        UnaryOp::Neg => Ok(UnaryFilterOp::Neg),
    }
}

/// Converts a logical aggregate function to a physical aggregate function.
pub fn convert_aggregate_function(func: LogicalAggregateFunction) -> PhysicalAggregateFunction {
    match func {
        LogicalAggregateFunction::Count => PhysicalAggregateFunction::Count,
        LogicalAggregateFunction::CountNonNull => PhysicalAggregateFunction::CountNonNull,
        LogicalAggregateFunction::Sum => PhysicalAggregateFunction::Sum,
        LogicalAggregateFunction::Avg => PhysicalAggregateFunction::Avg,
        LogicalAggregateFunction::Min => PhysicalAggregateFunction::Min,
        LogicalAggregateFunction::Max => PhysicalAggregateFunction::Max,
        LogicalAggregateFunction::Collect => PhysicalAggregateFunction::Collect,
        LogicalAggregateFunction::StdDev => PhysicalAggregateFunction::StdDev,
        LogicalAggregateFunction::StdDevPop => PhysicalAggregateFunction::StdDevPop,
        LogicalAggregateFunction::PercentileDisc => PhysicalAggregateFunction::PercentileDisc,
        LogicalAggregateFunction::PercentileCont => PhysicalAggregateFunction::PercentileCont,
    }
}

/// Converts a logical expression to a filter expression.
///
/// This is a standalone function that can be used by both LPG and RDF planners.
pub fn convert_filter_expression(expr: &LogicalExpression) -> Result<FilterExpression> {
    match expr {
        LogicalExpression::Literal(v) => Ok(FilterExpression::Literal(v.clone())),
        LogicalExpression::Variable(name) => Ok(FilterExpression::Variable(name.clone())),
        LogicalExpression::Property { variable, property } => Ok(FilterExpression::Property {
            variable: variable.clone(),
            property: property.clone(),
        }),
        LogicalExpression::Binary { left, op, right } => {
            let left_expr = convert_filter_expression(left)?;
            let right_expr = convert_filter_expression(right)?;
            let filter_op = convert_binary_op(*op)?;
            Ok(FilterExpression::Binary {
                left: Box::new(left_expr),
                op: filter_op,
                right: Box::new(right_expr),
            })
        }
        LogicalExpression::Unary { op, operand } => {
            let operand_expr = convert_filter_expression(operand)?;
            let filter_op = convert_unary_op(*op)?;
            Ok(FilterExpression::Unary {
                op: filter_op,
                operand: Box::new(operand_expr),
            })
        }
        LogicalExpression::FunctionCall { name, args, .. } => {
            let filter_args: Vec<FilterExpression> = args
                .iter()
                .map(convert_filter_expression)
                .collect::<Result<Vec<_>>>()?;
            Ok(FilterExpression::FunctionCall {
                name: name.clone(),
                args: filter_args,
            })
        }
        LogicalExpression::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            let filter_operand = operand
                .as_ref()
                .map(|e| convert_filter_expression(e))
                .transpose()?
                .map(Box::new);
            let filter_when_clauses: Vec<(FilterExpression, FilterExpression)> = when_clauses
                .iter()
                .map(|(cond, result)| {
                    Ok((
                        convert_filter_expression(cond)?,
                        convert_filter_expression(result)?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            let filter_else = else_clause
                .as_ref()
                .map(|e| convert_filter_expression(e))
                .transpose()?
                .map(Box::new);
            Ok(FilterExpression::Case {
                operand: filter_operand,
                when_clauses: filter_when_clauses,
                else_clause: filter_else,
            })
        }
        LogicalExpression::List(items) => {
            let filter_items: Vec<FilterExpression> = items
                .iter()
                .map(convert_filter_expression)
                .collect::<Result<Vec<_>>>()?;
            Ok(FilterExpression::List(filter_items))
        }
        LogicalExpression::Map(pairs) => {
            let filter_pairs: Vec<(String, FilterExpression)> = pairs
                .iter()
                .map(|(k, v)| Ok((k.clone(), convert_filter_expression(v)?)))
                .collect::<Result<Vec<_>>>()?;
            Ok(FilterExpression::Map(filter_pairs))
        }
        LogicalExpression::IndexAccess { base, index } => {
            let base_expr = convert_filter_expression(base)?;
            let index_expr = convert_filter_expression(index)?;
            Ok(FilterExpression::IndexAccess {
                base: Box::new(base_expr),
                index: Box::new(index_expr),
            })
        }
        LogicalExpression::SliceAccess { base, start, end } => {
            let base_expr = convert_filter_expression(base)?;
            let start_expr = start
                .as_ref()
                .map(|s| convert_filter_expression(s))
                .transpose()?
                .map(Box::new);
            let end_expr = end
                .as_ref()
                .map(|e| convert_filter_expression(e))
                .transpose()?
                .map(Box::new);
            Ok(FilterExpression::SliceAccess {
                base: Box::new(base_expr),
                start: start_expr,
                end: end_expr,
            })
        }
        LogicalExpression::Parameter(_) => Err(Error::Internal(
            "Parameters not yet supported in filters".to_string(),
        )),
        LogicalExpression::Labels(var) => Ok(FilterExpression::Labels(var.clone())),
        LogicalExpression::Type(var) => Ok(FilterExpression::Type(var.clone())),
        LogicalExpression::Id(var) => Ok(FilterExpression::Id(var.clone())),
        LogicalExpression::ListComprehension {
            variable,
            list_expr,
            filter_expr,
            map_expr,
        } => {
            let list = convert_filter_expression(list_expr)?;
            let filter = filter_expr
                .as_ref()
                .map(|f| convert_filter_expression(f))
                .transpose()?
                .map(Box::new);
            let map = convert_filter_expression(map_expr)?;
            Ok(FilterExpression::ListComprehension {
                variable: variable.clone(),
                list_expr: Box::new(list),
                filter_expr: filter,
                map_expr: Box::new(map),
            })
        }
        LogicalExpression::ExistsSubquery(_) | LogicalExpression::CountSubquery(_) => Err(
            Error::Internal("Subqueries not yet supported in filters".to_string()),
        ),
    }
}

/// Infers the logical type from a value.
fn value_to_logical_type(value: &grafeo_common::types::Value) -> LogicalType {
    use grafeo_common::types::Value;
    match value {
        Value::Null => LogicalType::String, // Default type for null
        Value::Bool(_) => LogicalType::Bool,
        Value::Int64(_) => LogicalType::Int64,
        Value::Float64(_) => LogicalType::Float64,
        Value::String(_) => LogicalType::String,
        Value::Bytes(_) => LogicalType::String, // No Bytes logical type, use String
        Value::Timestamp(_) => LogicalType::Timestamp,
        Value::List(_) => LogicalType::String, // Lists not yet supported as logical type
        Value::Map(_) => LogicalType::String,  // Maps not yet supported as logical type
        Value::Vector(v) => LogicalType::Vector(v.len()),
    }
}

/// Converts an expression to a string for column naming.
fn expression_to_string(expr: &LogicalExpression) -> String {
    match expr {
        LogicalExpression::Variable(name) => name.clone(),
        LogicalExpression::Property { variable, property } => {
            format!("{variable}.{property}")
        }
        LogicalExpression::Literal(value) => format!("{value:?}"),
        LogicalExpression::FunctionCall { name, .. } => format!("{name}(...)"),
        _ => "expr".to_string(),
    }
}

/// A physical plan ready for execution.
pub struct PhysicalPlan {
    /// The root physical operator.
    pub operator: Box<dyn Operator>,
    /// Column names for the result.
    pub columns: Vec<String>,
    /// Adaptive execution context with cardinality estimates.
    ///
    /// When adaptive execution is enabled, this context contains estimated
    /// cardinalities at various checkpoints in the plan. During execution,
    /// actual row counts are recorded and compared against estimates.
    pub adaptive_context: Option<AdaptiveContext>,
}

impl PhysicalPlan {
    /// Returns the column names.
    #[must_use]
    pub fn columns(&self) -> &[String] {
        &self.columns
    }

    /// Consumes the plan and returns the operator.
    pub fn into_operator(self) -> Box<dyn Operator> {
        self.operator
    }

    /// Returns the adaptive context, if adaptive execution is enabled.
    #[must_use]
    pub fn adaptive_context(&self) -> Option<&AdaptiveContext> {
        self.adaptive_context.as_ref()
    }

    /// Takes ownership of the adaptive context.
    pub fn take_adaptive_context(&mut self) -> Option<AdaptiveContext> {
        self.adaptive_context.take()
    }
}

/// Helper operator that returns a single result chunk once.
///
/// Used by the factorized expand chain to wrap the final result.
#[allow(dead_code)]
struct SingleResultOperator {
    result: Option<grafeo_core::execution::DataChunk>,
}

impl SingleResultOperator {
    #[allow(dead_code)]
    fn new(result: Option<grafeo_core::execution::DataChunk>) -> Self {
        Self { result }
    }
}

impl Operator for SingleResultOperator {
    fn next(&mut self) -> grafeo_core::execution::operators::OperatorResult {
        Ok(self.result.take())
    }

    fn reset(&mut self) {
        // Cannot reset - result is consumed
    }

    fn name(&self) -> &'static str {
        "SingleResult"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::plan::{
        AggregateExpr as LogicalAggregateExpr, CreateEdgeOp, CreateNodeOp, DeleteNodeOp,
        DistinctOp as LogicalDistinctOp, ExpandOp, FilterOp, JoinCondition, JoinOp,
        LimitOp as LogicalLimitOp, NodeScanOp, ReturnItem, ReturnOp, SkipOp as LogicalSkipOp,
        SortKey, SortOp,
    };
    use grafeo_common::types::Value;

    fn create_test_store() -> Arc<LpgStore> {
        let store = Arc::new(LpgStore::new());
        store.create_node(&["Person"]);
        store.create_node(&["Person"]);
        store.create_node(&["Company"]);
        store
    }

    // ==================== Simple Scan Tests ====================

    #[test]
    fn test_plan_simple_scan() {
        let store = create_test_store();
        let planner = Planner::new(store);

        // MATCH (n:Person) RETURN n
        let logical = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("n".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: Some("Person".to_string()),
                input: None,
            })),
        }));

        let physical = planner.plan(&logical).unwrap();
        assert_eq!(physical.columns(), &["n"]);
    }

    #[test]
    fn test_plan_scan_without_label() {
        let store = create_test_store();
        let planner = Planner::new(store);

        // MATCH (n) RETURN n
        let logical = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("n".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: None,
                input: None,
            })),
        }));

        let physical = planner.plan(&logical).unwrap();
        assert_eq!(physical.columns(), &["n"]);
    }

    #[test]
    fn test_plan_return_with_alias() {
        let store = create_test_store();
        let planner = Planner::new(store);

        // MATCH (n:Person) RETURN n AS person
        let logical = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("n".to_string()),
                alias: Some("person".to_string()),
            }],
            distinct: false,
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: Some("Person".to_string()),
                input: None,
            })),
        }));

        let physical = planner.plan(&logical).unwrap();
        assert_eq!(physical.columns(), &["person"]);
    }

    #[test]
    fn test_plan_return_property() {
        let store = create_test_store();
        let planner = Planner::new(store);

        // MATCH (n:Person) RETURN n.name
        let logical = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Property {
                    variable: "n".to_string(),
                    property: "name".to_string(),
                },
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: Some("Person".to_string()),
                input: None,
            })),
        }));

        let physical = planner.plan(&logical).unwrap();
        assert_eq!(physical.columns(), &["n.name"]);
    }

    #[test]
    fn test_plan_return_literal() {
        let store = create_test_store();
        let planner = Planner::new(store);

        // MATCH (n) RETURN 42 AS answer
        let logical = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Literal(Value::Int64(42)),
                alias: Some("answer".to_string()),
            }],
            distinct: false,
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: None,
                input: None,
            })),
        }));

        let physical = planner.plan(&logical).unwrap();
        assert_eq!(physical.columns(), &["answer"]);
    }

    // ==================== Filter Tests ====================

    #[test]
    fn test_plan_filter_equality() {
        let store = create_test_store();
        let planner = Planner::new(store);

        // MATCH (n:Person) WHERE n.age = 30 RETURN n
        let logical = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
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
                    op: BinaryOp::Eq,
                    right: Box::new(LogicalExpression::Literal(Value::Int64(30))),
                },
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".to_string(),
                    label: Some("Person".to_string()),
                    input: None,
                })),
            })),
        }));

        let physical = planner.plan(&logical).unwrap();
        assert_eq!(physical.columns(), &["n"]);
    }

    #[test]
    fn test_plan_filter_compound_and() {
        let store = create_test_store();
        let planner = Planner::new(store);

        // WHERE n.age > 20 AND n.age < 40
        let logical = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("n".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Filter(FilterOp {
                predicate: LogicalExpression::Binary {
                    left: Box::new(LogicalExpression::Binary {
                        left: Box::new(LogicalExpression::Property {
                            variable: "n".to_string(),
                            property: "age".to_string(),
                        }),
                        op: BinaryOp::Gt,
                        right: Box::new(LogicalExpression::Literal(Value::Int64(20))),
                    }),
                    op: BinaryOp::And,
                    right: Box::new(LogicalExpression::Binary {
                        left: Box::new(LogicalExpression::Property {
                            variable: "n".to_string(),
                            property: "age".to_string(),
                        }),
                        op: BinaryOp::Lt,
                        right: Box::new(LogicalExpression::Literal(Value::Int64(40))),
                    }),
                },
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".to_string(),
                    label: None,
                    input: None,
                })),
            })),
        }));

        let physical = planner.plan(&logical).unwrap();
        assert_eq!(physical.columns(), &["n"]);
    }

    #[test]
    fn test_plan_filter_unary_not() {
        let store = create_test_store();
        let planner = Planner::new(store);

        // WHERE NOT n.active
        let logical = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("n".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Filter(FilterOp {
                predicate: LogicalExpression::Unary {
                    op: UnaryOp::Not,
                    operand: Box::new(LogicalExpression::Property {
                        variable: "n".to_string(),
                        property: "active".to_string(),
                    }),
                },
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".to_string(),
                    label: None,
                    input: None,
                })),
            })),
        }));

        let physical = planner.plan(&logical).unwrap();
        assert_eq!(physical.columns(), &["n"]);
    }

    #[test]
    fn test_plan_filter_is_null() {
        let store = create_test_store();
        let planner = Planner::new(store);

        // WHERE n.email IS NULL
        let logical = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("n".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Filter(FilterOp {
                predicate: LogicalExpression::Unary {
                    op: UnaryOp::IsNull,
                    operand: Box::new(LogicalExpression::Property {
                        variable: "n".to_string(),
                        property: "email".to_string(),
                    }),
                },
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".to_string(),
                    label: None,
                    input: None,
                })),
            })),
        }));

        let physical = planner.plan(&logical).unwrap();
        assert_eq!(physical.columns(), &["n"]);
    }

    #[test]
    fn test_plan_filter_function_call() {
        let store = create_test_store();
        let planner = Planner::new(store);

        // WHERE size(n.friends) > 0
        let logical = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("n".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Filter(FilterOp {
                predicate: LogicalExpression::Binary {
                    left: Box::new(LogicalExpression::FunctionCall {
                        name: "size".to_string(),
                        args: vec![LogicalExpression::Property {
                            variable: "n".to_string(),
                            property: "friends".to_string(),
                        }],
                        distinct: false,
                    }),
                    op: BinaryOp::Gt,
                    right: Box::new(LogicalExpression::Literal(Value::Int64(0))),
                },
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".to_string(),
                    label: None,
                    input: None,
                })),
            })),
        }));

        let physical = planner.plan(&logical).unwrap();
        assert_eq!(physical.columns(), &["n"]);
    }

    // ==================== Expand Tests ====================

    #[test]
    fn test_plan_expand_outgoing() {
        let store = create_test_store();
        let planner = Planner::new(store);

        // MATCH (a:Person)-[:KNOWS]->(b) RETURN a, b
        let logical = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![
                ReturnItem {
                    expression: LogicalExpression::Variable("a".to_string()),
                    alias: None,
                },
                ReturnItem {
                    expression: LogicalExpression::Variable("b".to_string()),
                    alias: None,
                },
            ],
            distinct: false,
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
        }));

        let physical = planner.plan(&logical).unwrap();
        // The return should have columns [a, b]
        assert!(physical.columns().contains(&"a".to_string()));
        assert!(physical.columns().contains(&"b".to_string()));
    }

    #[test]
    fn test_plan_expand_with_edge_variable() {
        let store = create_test_store();
        let planner = Planner::new(store);

        // MATCH (a)-[r:KNOWS]->(b) RETURN a, r, b
        let logical = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![
                ReturnItem {
                    expression: LogicalExpression::Variable("a".to_string()),
                    alias: None,
                },
                ReturnItem {
                    expression: LogicalExpression::Variable("r".to_string()),
                    alias: None,
                },
                ReturnItem {
                    expression: LogicalExpression::Variable("b".to_string()),
                    alias: None,
                },
            ],
            distinct: false,
            input: Box::new(LogicalOperator::Expand(ExpandOp {
                from_variable: "a".to_string(),
                to_variable: "b".to_string(),
                edge_variable: Some("r".to_string()),
                direction: ExpandDirection::Outgoing,
                edge_type: Some("KNOWS".to_string()),
                min_hops: 1,
                max_hops: Some(1),
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "a".to_string(),
                    label: None,
                    input: None,
                })),
                path_alias: None,
            })),
        }));

        let physical = planner.plan(&logical).unwrap();
        assert!(physical.columns().contains(&"a".to_string()));
        assert!(physical.columns().contains(&"r".to_string()));
        assert!(physical.columns().contains(&"b".to_string()));
    }

    // ==================== Limit/Skip/Sort Tests ====================

    #[test]
    fn test_plan_limit() {
        let store = create_test_store();
        let planner = Planner::new(store);

        // MATCH (n) RETURN n LIMIT 10
        let logical = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("n".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Limit(LogicalLimitOp {
                count: 10,
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".to_string(),
                    label: None,
                    input: None,
                })),
            })),
        }));

        let physical = planner.plan(&logical).unwrap();
        assert_eq!(physical.columns(), &["n"]);
    }

    #[test]
    fn test_plan_skip() {
        let store = create_test_store();
        let planner = Planner::new(store);

        // MATCH (n) RETURN n SKIP 5
        let logical = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("n".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Skip(LogicalSkipOp {
                count: 5,
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".to_string(),
                    label: None,
                    input: None,
                })),
            })),
        }));

        let physical = planner.plan(&logical).unwrap();
        assert_eq!(physical.columns(), &["n"]);
    }

    #[test]
    fn test_plan_sort() {
        let store = create_test_store();
        let planner = Planner::new(store);

        // MATCH (n) RETURN n ORDER BY n.name ASC
        let logical = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("n".to_string()),
                alias: None,
            }],
            distinct: false,
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

        let physical = planner.plan(&logical).unwrap();
        assert_eq!(physical.columns(), &["n"]);
    }

    #[test]
    fn test_plan_sort_descending() {
        let store = create_test_store();
        let planner = Planner::new(store);

        // ORDER BY n DESC
        let logical = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("n".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Sort(SortOp {
                keys: vec![SortKey {
                    expression: LogicalExpression::Variable("n".to_string()),
                    order: SortOrder::Descending,
                }],
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".to_string(),
                    label: None,
                    input: None,
                })),
            })),
        }));

        let physical = planner.plan(&logical).unwrap();
        assert_eq!(physical.columns(), &["n"]);
    }

    #[test]
    fn test_plan_distinct() {
        let store = create_test_store();
        let planner = Planner::new(store);

        // MATCH (n) RETURN DISTINCT n
        let logical = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("n".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Distinct(LogicalDistinctOp {
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".to_string(),
                    label: None,
                    input: None,
                })),
                columns: None,
            })),
        }));

        let physical = planner.plan(&logical).unwrap();
        assert_eq!(physical.columns(), &["n"]);
    }

    // ==================== Aggregate Tests ====================

    #[test]
    fn test_plan_aggregate_count() {
        let store = create_test_store();
        let planner = Planner::new(store);

        // MATCH (n) RETURN count(n)
        let logical = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("cnt".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Aggregate(AggregateOp {
                group_by: vec![],
                aggregates: vec![LogicalAggregateExpr {
                    function: LogicalAggregateFunction::Count,
                    expression: Some(LogicalExpression::Variable("n".to_string())),
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

        let physical = planner.plan(&logical).unwrap();
        assert!(physical.columns().contains(&"cnt".to_string()));
    }

    #[test]
    fn test_plan_aggregate_with_group_by() {
        let store = create_test_store();
        let planner = Planner::new(store);

        // MATCH (n:Person) RETURN n.city, count(n) GROUP BY n.city
        let logical = LogicalPlan::new(LogicalOperator::Aggregate(AggregateOp {
            group_by: vec![LogicalExpression::Property {
                variable: "n".to_string(),
                property: "city".to_string(),
            }],
            aggregates: vec![LogicalAggregateExpr {
                function: LogicalAggregateFunction::Count,
                expression: Some(LogicalExpression::Variable("n".to_string())),
                distinct: false,
                alias: Some("cnt".to_string()),
                percentile: None,
            }],
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: Some("Person".to_string()),
                input: None,
            })),
            having: None,
        }));

        let physical = planner.plan(&logical).unwrap();
        assert_eq!(physical.columns().len(), 2);
    }

    #[test]
    fn test_plan_aggregate_sum() {
        let store = create_test_store();
        let planner = Planner::new(store);

        // SUM(n.value)
        let logical = LogicalPlan::new(LogicalOperator::Aggregate(AggregateOp {
            group_by: vec![],
            aggregates: vec![LogicalAggregateExpr {
                function: LogicalAggregateFunction::Sum,
                expression: Some(LogicalExpression::Property {
                    variable: "n".to_string(),
                    property: "value".to_string(),
                }),
                distinct: false,
                alias: Some("total".to_string()),
                percentile: None,
            }],
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: None,
                input: None,
            })),
            having: None,
        }));

        let physical = planner.plan(&logical).unwrap();
        assert!(physical.columns().contains(&"total".to_string()));
    }

    #[test]
    fn test_plan_aggregate_avg() {
        let store = create_test_store();
        let planner = Planner::new(store);

        // AVG(n.score)
        let logical = LogicalPlan::new(LogicalOperator::Aggregate(AggregateOp {
            group_by: vec![],
            aggregates: vec![LogicalAggregateExpr {
                function: LogicalAggregateFunction::Avg,
                expression: Some(LogicalExpression::Property {
                    variable: "n".to_string(),
                    property: "score".to_string(),
                }),
                distinct: false,
                alias: Some("average".to_string()),
                percentile: None,
            }],
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: None,
                input: None,
            })),
            having: None,
        }));

        let physical = planner.plan(&logical).unwrap();
        assert!(physical.columns().contains(&"average".to_string()));
    }

    #[test]
    fn test_plan_aggregate_min_max() {
        let store = create_test_store();
        let planner = Planner::new(store);

        // MIN(n.age), MAX(n.age)
        let logical = LogicalPlan::new(LogicalOperator::Aggregate(AggregateOp {
            group_by: vec![],
            aggregates: vec![
                LogicalAggregateExpr {
                    function: LogicalAggregateFunction::Min,
                    expression: Some(LogicalExpression::Property {
                        variable: "n".to_string(),
                        property: "age".to_string(),
                    }),
                    distinct: false,
                    alias: Some("youngest".to_string()),
                    percentile: None,
                },
                LogicalAggregateExpr {
                    function: LogicalAggregateFunction::Max,
                    expression: Some(LogicalExpression::Property {
                        variable: "n".to_string(),
                        property: "age".to_string(),
                    }),
                    distinct: false,
                    alias: Some("oldest".to_string()),
                    percentile: None,
                },
            ],
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: None,
                input: None,
            })),
            having: None,
        }));

        let physical = planner.plan(&logical).unwrap();
        assert!(physical.columns().contains(&"youngest".to_string()));
        assert!(physical.columns().contains(&"oldest".to_string()));
    }

    // ==================== Join Tests ====================

    #[test]
    fn test_plan_inner_join() {
        let store = create_test_store();
        let planner = Planner::new(store);

        // Inner join between two scans
        let logical = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![
                ReturnItem {
                    expression: LogicalExpression::Variable("a".to_string()),
                    alias: None,
                },
                ReturnItem {
                    expression: LogicalExpression::Variable("b".to_string()),
                    alias: None,
                },
            ],
            distinct: false,
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
                conditions: vec![JoinCondition {
                    left: LogicalExpression::Variable("a".to_string()),
                    right: LogicalExpression::Variable("b".to_string()),
                }],
            })),
        }));

        let physical = planner.plan(&logical).unwrap();
        assert!(physical.columns().contains(&"a".to_string()));
        assert!(physical.columns().contains(&"b".to_string()));
    }

    #[test]
    fn test_plan_cross_join() {
        let store = create_test_store();
        let planner = Planner::new(store);

        // Cross join (no conditions)
        let logical = LogicalPlan::new(LogicalOperator::Join(JoinOp {
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
            join_type: JoinType::Cross,
            conditions: vec![],
        }));

        let physical = planner.plan(&logical).unwrap();
        assert_eq!(physical.columns().len(), 2);
    }

    #[test]
    fn test_plan_left_join() {
        let store = create_test_store();
        let planner = Planner::new(store);

        let logical = LogicalPlan::new(LogicalOperator::Join(JoinOp {
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
            join_type: JoinType::Left,
            conditions: vec![],
        }));

        let physical = planner.plan(&logical).unwrap();
        assert_eq!(physical.columns().len(), 2);
    }

    // ==================== Mutation Tests ====================

    #[test]
    fn test_plan_create_node() {
        let store = create_test_store();
        let planner = Planner::new(store);

        // CREATE (n:Person {name: 'Alice'})
        let logical = LogicalPlan::new(LogicalOperator::CreateNode(CreateNodeOp {
            variable: "n".to_string(),
            labels: vec!["Person".to_string()],
            properties: vec![(
                "name".to_string(),
                LogicalExpression::Literal(Value::String("Alice".into())),
            )],
            input: None,
        }));

        let physical = planner.plan(&logical).unwrap();
        assert!(physical.columns().contains(&"n".to_string()));
    }

    #[test]
    fn test_plan_create_edge() {
        let store = create_test_store();
        let planner = Planner::new(store);

        // MATCH (a), (b) CREATE (a)-[:KNOWS]->(b)
        let logical = LogicalPlan::new(LogicalOperator::CreateEdge(CreateEdgeOp {
            variable: Some("r".to_string()),
            from_variable: "a".to_string(),
            to_variable: "b".to_string(),
            edge_type: "KNOWS".to_string(),
            properties: vec![],
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
                join_type: JoinType::Cross,
                conditions: vec![],
            })),
        }));

        let physical = planner.plan(&logical).unwrap();
        assert!(physical.columns().contains(&"r".to_string()));
    }

    #[test]
    fn test_plan_delete_node() {
        let store = create_test_store();
        let planner = Planner::new(store);

        // MATCH (n) DELETE n
        let logical = LogicalPlan::new(LogicalOperator::DeleteNode(DeleteNodeOp {
            variable: "n".to_string(),
            detach: false,
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: None,
                input: None,
            })),
        }));

        let physical = planner.plan(&logical).unwrap();
        assert!(physical.columns().contains(&"deleted_count".to_string()));
    }

    // ==================== Error Cases ====================

    #[test]
    fn test_plan_empty_errors() {
        let store = create_test_store();
        let planner = Planner::new(store);

        let logical = LogicalPlan::new(LogicalOperator::Empty);
        let result = planner.plan(&logical);
        assert!(result.is_err());
    }

    #[test]
    fn test_plan_missing_variable_in_return() {
        let store = create_test_store();
        let planner = Planner::new(store);

        // Return variable that doesn't exist in input
        let logical = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("missing".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: None,
                input: None,
            })),
        }));

        let result = planner.plan(&logical);
        assert!(result.is_err());
    }

    // ==================== Helper Function Tests ====================

    #[test]
    fn test_convert_binary_ops() {
        assert!(convert_binary_op(BinaryOp::Eq).is_ok());
        assert!(convert_binary_op(BinaryOp::Ne).is_ok());
        assert!(convert_binary_op(BinaryOp::Lt).is_ok());
        assert!(convert_binary_op(BinaryOp::Le).is_ok());
        assert!(convert_binary_op(BinaryOp::Gt).is_ok());
        assert!(convert_binary_op(BinaryOp::Ge).is_ok());
        assert!(convert_binary_op(BinaryOp::And).is_ok());
        assert!(convert_binary_op(BinaryOp::Or).is_ok());
        assert!(convert_binary_op(BinaryOp::Add).is_ok());
        assert!(convert_binary_op(BinaryOp::Sub).is_ok());
        assert!(convert_binary_op(BinaryOp::Mul).is_ok());
        assert!(convert_binary_op(BinaryOp::Div).is_ok());
    }

    #[test]
    fn test_convert_unary_ops() {
        assert!(convert_unary_op(UnaryOp::Not).is_ok());
        assert!(convert_unary_op(UnaryOp::IsNull).is_ok());
        assert!(convert_unary_op(UnaryOp::IsNotNull).is_ok());
        assert!(convert_unary_op(UnaryOp::Neg).is_ok());
    }

    #[test]
    fn test_convert_aggregate_functions() {
        assert!(matches!(
            convert_aggregate_function(LogicalAggregateFunction::Count),
            PhysicalAggregateFunction::Count
        ));
        assert!(matches!(
            convert_aggregate_function(LogicalAggregateFunction::Sum),
            PhysicalAggregateFunction::Sum
        ));
        assert!(matches!(
            convert_aggregate_function(LogicalAggregateFunction::Avg),
            PhysicalAggregateFunction::Avg
        ));
        assert!(matches!(
            convert_aggregate_function(LogicalAggregateFunction::Min),
            PhysicalAggregateFunction::Min
        ));
        assert!(matches!(
            convert_aggregate_function(LogicalAggregateFunction::Max),
            PhysicalAggregateFunction::Max
        ));
    }

    #[test]
    fn test_planner_accessors() {
        let store = create_test_store();
        let planner = Planner::new(Arc::clone(&store));

        assert!(planner.tx_id().is_none());
        assert!(planner.tx_manager().is_none());
        let _ = planner.viewing_epoch(); // Just ensure it's accessible
    }

    #[test]
    fn test_physical_plan_accessors() {
        let store = create_test_store();
        let planner = Planner::new(store);

        let logical = LogicalPlan::new(LogicalOperator::NodeScan(NodeScanOp {
            variable: "n".to_string(),
            label: None,
            input: None,
        }));

        let physical = planner.plan(&logical).unwrap();
        assert_eq!(physical.columns(), &["n"]);

        // Test into_operator
        let _ = physical.into_operator();
    }

    // ==================== Adaptive Planning Tests ====================

    #[test]
    fn test_plan_adaptive_with_scan() {
        let store = create_test_store();
        let planner = Planner::new(store);

        // MATCH (n:Person) RETURN n
        let logical = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("n".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: Some("Person".to_string()),
                input: None,
            })),
        }));

        let physical = planner.plan_adaptive(&logical).unwrap();
        assert_eq!(physical.columns(), &["n"]);
        // Should have adaptive context with estimates
        assert!(physical.adaptive_context.is_some());
    }

    #[test]
    fn test_plan_adaptive_with_filter() {
        let store = create_test_store();
        let planner = Planner::new(store);

        // MATCH (n) WHERE n.age > 30 RETURN n
        let logical = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
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
                    label: None,
                    input: None,
                })),
            })),
        }));

        let physical = planner.plan_adaptive(&logical).unwrap();
        assert!(physical.adaptive_context.is_some());
    }

    #[test]
    fn test_plan_adaptive_with_expand() {
        let store = create_test_store();
        let planner = Planner::new(Arc::clone(&store)).with_factorized_execution(false);

        // MATCH (a)-[:KNOWS]->(b) RETURN a, b
        let logical = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![
                ReturnItem {
                    expression: LogicalExpression::Variable("a".to_string()),
                    alias: None,
                },
                ReturnItem {
                    expression: LogicalExpression::Variable("b".to_string()),
                    alias: None,
                },
            ],
            distinct: false,
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
                    label: None,
                    input: None,
                })),
                path_alias: None,
            })),
        }));

        let physical = planner.plan_adaptive(&logical).unwrap();
        assert!(physical.adaptive_context.is_some());
    }

    #[test]
    fn test_plan_adaptive_with_join() {
        let store = create_test_store();
        let planner = Planner::new(store);

        let logical = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![
                ReturnItem {
                    expression: LogicalExpression::Variable("a".to_string()),
                    alias: None,
                },
                ReturnItem {
                    expression: LogicalExpression::Variable("b".to_string()),
                    alias: None,
                },
            ],
            distinct: false,
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
                join_type: JoinType::Cross,
                conditions: vec![],
            })),
        }));

        let physical = planner.plan_adaptive(&logical).unwrap();
        assert!(physical.adaptive_context.is_some());
    }

    #[test]
    fn test_plan_adaptive_with_aggregate() {
        let store = create_test_store();
        let planner = Planner::new(store);

        let logical = LogicalPlan::new(LogicalOperator::Aggregate(AggregateOp {
            group_by: vec![],
            aggregates: vec![LogicalAggregateExpr {
                function: LogicalAggregateFunction::Count,
                expression: Some(LogicalExpression::Variable("n".to_string())),
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
        }));

        let physical = planner.plan_adaptive(&logical).unwrap();
        assert!(physical.adaptive_context.is_some());
    }

    #[test]
    fn test_plan_adaptive_with_distinct() {
        let store = create_test_store();
        let planner = Planner::new(store);

        let logical = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("n".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Distinct(LogicalDistinctOp {
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".to_string(),
                    label: None,
                    input: None,
                })),
                columns: None,
            })),
        }));

        let physical = planner.plan_adaptive(&logical).unwrap();
        assert!(physical.adaptive_context.is_some());
    }

    #[test]
    fn test_plan_adaptive_with_limit() {
        let store = create_test_store();
        let planner = Planner::new(store);

        let logical = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("n".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Limit(LogicalLimitOp {
                count: 10,
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".to_string(),
                    label: None,
                    input: None,
                })),
            })),
        }));

        let physical = planner.plan_adaptive(&logical).unwrap();
        assert!(physical.adaptive_context.is_some());
    }

    #[test]
    fn test_plan_adaptive_with_skip() {
        let store = create_test_store();
        let planner = Planner::new(store);

        let logical = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("n".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Skip(LogicalSkipOp {
                count: 5,
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".to_string(),
                    label: None,
                    input: None,
                })),
            })),
        }));

        let physical = planner.plan_adaptive(&logical).unwrap();
        assert!(physical.adaptive_context.is_some());
    }

    #[test]
    fn test_plan_adaptive_with_sort() {
        let store = create_test_store();
        let planner = Planner::new(store);

        let logical = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("n".to_string()),
                alias: None,
            }],
            distinct: false,
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

        let physical = planner.plan_adaptive(&logical).unwrap();
        assert!(physical.adaptive_context.is_some());
    }

    #[test]
    fn test_plan_adaptive_with_union() {
        let store = create_test_store();
        let planner = Planner::new(store);

        let logical = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("n".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Union(UnionOp {
                inputs: vec![
                    LogicalOperator::NodeScan(NodeScanOp {
                        variable: "n".to_string(),
                        label: Some("Person".to_string()),
                        input: None,
                    }),
                    LogicalOperator::NodeScan(NodeScanOp {
                        variable: "n".to_string(),
                        label: Some("Company".to_string()),
                        input: None,
                    }),
                ],
            })),
        }));

        let physical = planner.plan_adaptive(&logical).unwrap();
        assert!(physical.adaptive_context.is_some());
    }

    // ==================== Variable Length Path Tests ====================

    #[test]
    fn test_plan_expand_variable_length() {
        let store = create_test_store();
        let planner = Planner::new(store);

        // MATCH (a)-[:KNOWS*1..3]->(b) RETURN a, b
        let logical = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![
                ReturnItem {
                    expression: LogicalExpression::Variable("a".to_string()),
                    alias: None,
                },
                ReturnItem {
                    expression: LogicalExpression::Variable("b".to_string()),
                    alias: None,
                },
            ],
            distinct: false,
            input: Box::new(LogicalOperator::Expand(ExpandOp {
                from_variable: "a".to_string(),
                to_variable: "b".to_string(),
                edge_variable: None,
                direction: ExpandDirection::Outgoing,
                edge_type: Some("KNOWS".to_string()),
                min_hops: 1,
                max_hops: Some(3),
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "a".to_string(),
                    label: None,
                    input: None,
                })),
                path_alias: None,
            })),
        }));

        let physical = planner.plan(&logical).unwrap();
        assert!(physical.columns().contains(&"a".to_string()));
        assert!(physical.columns().contains(&"b".to_string()));
    }

    #[test]
    fn test_plan_expand_with_path_alias() {
        let store = create_test_store();
        let planner = Planner::new(store);

        // MATCH p = (a)-[:KNOWS*1..3]->(b) RETURN a, b
        let logical = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![
                ReturnItem {
                    expression: LogicalExpression::Variable("a".to_string()),
                    alias: None,
                },
                ReturnItem {
                    expression: LogicalExpression::Variable("b".to_string()),
                    alias: None,
                },
            ],
            distinct: false,
            input: Box::new(LogicalOperator::Expand(ExpandOp {
                from_variable: "a".to_string(),
                to_variable: "b".to_string(),
                edge_variable: None,
                direction: ExpandDirection::Outgoing,
                edge_type: Some("KNOWS".to_string()),
                min_hops: 1,
                max_hops: Some(3),
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "a".to_string(),
                    label: None,
                    input: None,
                })),
                path_alias: Some("p".to_string()),
            })),
        }));

        let physical = planner.plan(&logical).unwrap();
        // Verify plan was created successfully with expected output columns
        assert!(physical.columns().contains(&"a".to_string()));
        assert!(physical.columns().contains(&"b".to_string()));
    }

    #[test]
    fn test_plan_expand_incoming() {
        let store = create_test_store();
        let planner = Planner::new(Arc::clone(&store)).with_factorized_execution(false);

        // MATCH (a)<-[:KNOWS]-(b) RETURN a, b
        let logical = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![
                ReturnItem {
                    expression: LogicalExpression::Variable("a".to_string()),
                    alias: None,
                },
                ReturnItem {
                    expression: LogicalExpression::Variable("b".to_string()),
                    alias: None,
                },
            ],
            distinct: false,
            input: Box::new(LogicalOperator::Expand(ExpandOp {
                from_variable: "a".to_string(),
                to_variable: "b".to_string(),
                edge_variable: None,
                direction: ExpandDirection::Incoming,
                edge_type: Some("KNOWS".to_string()),
                min_hops: 1,
                max_hops: Some(1),
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "a".to_string(),
                    label: None,
                    input: None,
                })),
                path_alias: None,
            })),
        }));

        let physical = planner.plan(&logical).unwrap();
        assert!(physical.columns().contains(&"a".to_string()));
        assert!(physical.columns().contains(&"b".to_string()));
    }

    #[test]
    fn test_plan_expand_both_directions() {
        let store = create_test_store();
        let planner = Planner::new(Arc::clone(&store)).with_factorized_execution(false);

        // MATCH (a)-[:KNOWS]-(b) RETURN a, b
        let logical = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![
                ReturnItem {
                    expression: LogicalExpression::Variable("a".to_string()),
                    alias: None,
                },
                ReturnItem {
                    expression: LogicalExpression::Variable("b".to_string()),
                    alias: None,
                },
            ],
            distinct: false,
            input: Box::new(LogicalOperator::Expand(ExpandOp {
                from_variable: "a".to_string(),
                to_variable: "b".to_string(),
                edge_variable: None,
                direction: ExpandDirection::Both,
                edge_type: Some("KNOWS".to_string()),
                min_hops: 1,
                max_hops: Some(1),
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "a".to_string(),
                    label: None,
                    input: None,
                })),
                path_alias: None,
            })),
        }));

        let physical = planner.plan(&logical).unwrap();
        assert!(physical.columns().contains(&"a".to_string()));
        assert!(physical.columns().contains(&"b".to_string()));
    }

    // ==================== With Context Tests ====================

    #[test]
    fn test_planner_with_context() {
        use crate::transaction::TransactionManager;

        let store = create_test_store();
        let tx_manager = Arc::new(TransactionManager::new());
        let tx_id = tx_manager.begin();
        let epoch = tx_manager.current_epoch();

        let planner = Planner::with_context(
            Arc::clone(&store),
            Arc::clone(&tx_manager),
            Some(tx_id),
            epoch,
        );

        assert_eq!(planner.tx_id(), Some(tx_id));
        assert!(planner.tx_manager().is_some());
        assert_eq!(planner.viewing_epoch(), epoch);
    }

    #[test]
    fn test_planner_with_factorized_execution_disabled() {
        let store = create_test_store();
        let planner = Planner::new(Arc::clone(&store)).with_factorized_execution(false);

        // Two consecutive expands - should NOT use factorized execution
        let logical = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![
                ReturnItem {
                    expression: LogicalExpression::Variable("a".to_string()),
                    alias: None,
                },
                ReturnItem {
                    expression: LogicalExpression::Variable("c".to_string()),
                    alias: None,
                },
            ],
            distinct: false,
            input: Box::new(LogicalOperator::Expand(ExpandOp {
                from_variable: "b".to_string(),
                to_variable: "c".to_string(),
                edge_variable: None,
                direction: ExpandDirection::Outgoing,
                edge_type: None,
                min_hops: 1,
                max_hops: Some(1),
                input: Box::new(LogicalOperator::Expand(ExpandOp {
                    from_variable: "a".to_string(),
                    to_variable: "b".to_string(),
                    edge_variable: None,
                    direction: ExpandDirection::Outgoing,
                    edge_type: None,
                    min_hops: 1,
                    max_hops: Some(1),
                    input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                        variable: "a".to_string(),
                        label: None,
                        input: None,
                    })),
                    path_alias: None,
                })),
                path_alias: None,
            })),
        }));

        let physical = planner.plan(&logical).unwrap();
        assert!(physical.columns().contains(&"a".to_string()));
        assert!(physical.columns().contains(&"c".to_string()));
    }

    // ==================== Sort with Property Tests ====================

    #[test]
    fn test_plan_sort_by_property() {
        let store = create_test_store();
        let planner = Planner::new(store);

        // MATCH (n) RETURN n ORDER BY n.name ASC
        let logical = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("n".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Sort(SortOp {
                keys: vec![SortKey {
                    expression: LogicalExpression::Property {
                        variable: "n".to_string(),
                        property: "name".to_string(),
                    },
                    order: SortOrder::Ascending,
                }],
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".to_string(),
                    label: None,
                    input: None,
                })),
            })),
        }));

        let physical = planner.plan(&logical).unwrap();
        // Should have the property column projected
        assert!(physical.columns().contains(&"n".to_string()));
    }

    // ==================== Scan with Input Tests ====================

    #[test]
    fn test_plan_scan_with_input() {
        let store = create_test_store();
        let planner = Planner::new(store);

        // A scan with another scan as input (for chained patterns)
        let logical = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![
                ReturnItem {
                    expression: LogicalExpression::Variable("a".to_string()),
                    alias: None,
                },
                ReturnItem {
                    expression: LogicalExpression::Variable("b".to_string()),
                    alias: None,
                },
            ],
            distinct: false,
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "b".to_string(),
                label: Some("Company".to_string()),
                input: Some(Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "a".to_string(),
                    label: Some("Person".to_string()),
                    input: None,
                }))),
            })),
        }));

        let physical = planner.plan(&logical).unwrap();
        assert!(physical.columns().contains(&"a".to_string()));
        assert!(physical.columns().contains(&"b".to_string()));
    }
}
