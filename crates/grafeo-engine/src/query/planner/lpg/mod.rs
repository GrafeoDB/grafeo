//! LPG (Labeled Property Graph) planner.
//!
//! Converts logical plans with LPG operators (NodeScan, Expand, etc.) to
//! physical operators that execute against an LPG store.
//!
//! # Structure
//!
//! Logical operators are consumed by `plan_operator`, dispatched to
//! per-operator submodules that share the crate-private `Planner`
//! struct:
//!
//! | Submodule    | Owns                                                         |
//! |--------------|--------------------------------------------------------------|
//! | `scan`       | `plan_node_scan`: label, property, and index-seek scans      |
//! | `expand`     | single-hop and multi-hop expansion (factorized chain)        |
//! | `filter`     | predicate rewriting, zone maps, index pushdown, EXISTS/COUNT |
//! | `expression` | `LogicalExpression` -> `FilterExpression` conversion         |
//! | `join`       | hash, nested-loop, leapfrog, multi-way, semi/anti            |
//! | `aggregate`, `project`, `mutation` | the rest of the pipeline               |
//!
//! # Rewrites applied during planning
//!
//! The planner is not a pure tree walker: it performs several shape
//! transforms up-front because they decide which *physical* operator is
//! legal, not just which is faster. Each rewrite lives in the submodule
//! that owns the corresponding logical operator.
//!
//! 1. **Factorized expand chains** (`expand::plan_expand_chain`):
//!    two or more consecutive `Expand`s are collapsed into a single
//!    `LazyFactorizedChainOperator`. A hand-rolled tree of `Expand`
//!    operators would materialise the Cartesian product between hops
//!    (O(d^k) rows for a k-hop query with average degree d); factorisation
//!    represents each hop as a set per row and evaluates upward filters
//!    lazily, which is asymptotically better *and* preserves filter
//!    semantics (filters apply to the flattened view). Disabled when
//!    `factorized_execution` is false on the planner, used as an escape
//!    hatch for operators that need materialised intermediate state.
//!
//! 2. **EXISTS / NOT EXISTS as semi/anti joins** (`filter::extract_complex_exists`):
//!    anything more than a trivial single-hop EXISTS is rewritten to a
//!    semi-join (EXISTS) or anti-join (NOT EXISTS) before the filter is
//!    planned. A literal nested re-execution would re-scan the subquery
//!    per outer row; the join form piggy-backs on the regular hash-join
//!    infrastructure. The fast path in `expression::convert_expression`
//!    keeps trivial single-hop EXISTS as inline predicates so small
//!    queries stay scan-local.
//!
//! 3. **EXISTS inside OR** (`filter::extract_exists_from_or`):
//!    semi-joins filter rows and therefore compose incorrectly with the
//!    disjunctive scalar side of an OR. The rewrite splits the OR into a
//!    semi-join branch and a regular filter branch, UNIONNs them, and
//!    deduplicates. Without the split, rows satisfied only by the scalar
//!    side would be dropped.
//!
//! 4. **COUNT subquery comparisons** (`filter::extract_count_comparison`):
//!    `COUNT { ... } op N` rewrites into Apply + Aggregate + Filter.
//!    Evaluating the COUNT inline per row would be a correlated
//!    subquery; Apply + Aggregate reuses the hash-aggregate physical
//!    operator.
//!
//! # Filter pushdown and index selection
//!
//! `filter::plan_filter` is the central point. It tries a sequence of
//! optimisations before falling back to a generic `FilterOperator`; the
//! ordering is load-bearing:
//!
//! 1. **Complex EXISTS / OR / COUNT rewrites** (above). These must run
//!    first because later steps assume a pure-scalar predicate.
//! 2. **Zone-map short-circuit**: if the column's zone summary proves the
//!    predicate cannot match any row group, plan an `EmptyOperator`.
//! 3. **Property index seek**: equality on an indexed property becomes an
//!    index scan (O(1) per lookup) instead of a filtered full scan.
//! 4. **Range index lookup**: `>`, `<`, `>=`, `<=` on an indexed
//!    property becomes a range scan.
//! 5. **Hybrid vector+text pushdown** (feature-gated): compound
//!    AND/OR predicates over `vector_match` / `text_match` are folded
//!    into a single fused scan.
//! 6. **Generic `FilterOperator`**: the residual predicate wraps the
//!    input scan.
//!
//! Whatever *remains* after a pushdown (e.g. the non-index-backed half
//! of an AND) is planned as a residual filter on top of the chosen scan,
//! so the returned tree is always semantically equivalent to the
//! input plan.
//!
//! # Transaction context
//!
//! Every operator inherits the planner's `viewing_epoch` and
//! `transaction_id`. Factorised chains and mutations carry them
//! through `with_transaction_context()` so MVCC visibility is applied
//! at every hop: see [`crate::transaction`] for the epoch/visibility
//! rules.

mod aggregate;
mod expand;
mod expression;
mod filter;
mod filter_hybrid;
mod join;
mod mutation;
mod project;
mod scan;

#[cfg(feature = "algos")]
use crate::query::plan::CallProcedureOp;
#[cfg(feature = "text-index")]
use crate::query::plan::TextScanOp;
use crate::query::plan::{
    AddLabelOp, AggregateFunction as LogicalAggregateFunction, AggregateOp, AntiJoinOp, ApplyOp,
    BinaryOp, CreateEdgeOp, CreateNodeOp, DeleteEdgeOp, DeleteNodeOp, DistinctOp,
    EntityKind as LogicalEntityKind, ExceptOp, ExpandDirection, ExpandOp, FilterOp,
    HorizontalAggregateOp, IntersectOp, JoinOp, JoinType, LeftJoinOp, LimitOp, LogicalExpression,
    LogicalOperator, LogicalPlan, MapCollectOp, MergeOp, MergeRelationshipOp, MultiWayJoinOp,
    NodeScanOp, OtherwiseOp, PathMode, RemoveLabelOp, ReturnOp, SetPropertyOp, ShortestPathOp,
    SkipOp, SortOp, SortOrder, UnaryOp, UnionOp, UnwindOp,
};
#[cfg(feature = "vector-index")]
use crate::query::plan::{VectorMetric, VectorScanOp};
use grafeo_common::grafeo_debug_span;
use grafeo_common::types::{EpochId, TransactionId};
use grafeo_common::types::{LogicalType, Value};
use grafeo_common::utils::error::{Error, Result};
use grafeo_core::execution::AdaptiveContext;
use grafeo_core::execution::operators::{
    AddLabelOperator, AggregateExpr as PhysicalAggregateExpr, ApplyOperator, ConstraintValidator,
    CreateEdgeOperator, CreateNodeOperator, DeleteEdgeOperator, DeleteNodeOperator,
    DistinctOperator, EmptyOperator, EntityKind, ExecutionPathMode, ExpandOperator, ExpandStep,
    ExpressionPredicate, FactorizedAggregate, FactorizedAggregateOperator, FilterExpression,
    FilterOperator, HashAggregateOperator, HashJoinOperator, HorizontalAggregateOperator,
    JoinType as PhysicalJoinType, LazyFactorizedChainOperator, LeapfrogJoinOperator,
    LoadDataOperator, MapCollectOperator, MergeConfig, MergeOperator, MergeRelationshipConfig,
    MergeRelationshipOperator, NestedLoopJoinOperator, NodeListOperator, NullOrder, Operator,
    ParameterScanOperator, ProjectExpr, ProjectOperator, PropertySource, RangeScanOperator,
    RemoveLabelOperator, ScanOperator, SetPropertyOperator, ShortestPathOperator,
    SimpleAggregateOperator, SortDirection, SortKey as PhysicalSortKey, SortOperator,
    UnionOperator, UnwindOperator, VariableLengthExpandOperator,
};
use grafeo_core::graph::{Direction, GraphStoreMut, GraphStoreSearch};
use std::collections::HashMap;
use std::sync::Arc;

use crate::query::planner::common;
use crate::query::planner::common::{expression_to_string, output_column_name, resolved_column_name};
use crate::query::planner::{
    PhysicalPlan, convert_aggregate_function, convert_binary_op, convert_filter_expression,
    convert_unary_op, value_to_logical_type,
};
use crate::transaction::TransactionManager;

/// Range bounds for property-based range queries.
struct RangeBounds<'a> {
    min: Option<&'a Value>,
    max: Option<&'a Value>,
    min_inclusive: bool,
    max_inclusive: bool,
}

/// Converts a logical plan to a physical operator tree for LPG stores.
pub struct Planner {
    /// The graph store (read-only operations).
    pub(super) store: Arc<dyn GraphStoreSearch>,
    /// Writable graph store (None for read-only databases).
    pub(super) write_store: Option<Arc<dyn GraphStoreMut>>,
    /// Transaction manager for MVCC operations.
    pub(super) transaction_manager: Option<Arc<TransactionManager>>,
    /// Current transaction ID (if in a transaction).
    pub(super) transaction_id: Option<TransactionId>,
    /// Epoch to use for visibility checks.
    pub(super) viewing_epoch: EpochId,
    /// Counter for generating unique anonymous edge column names.
    pub(super) anon_edge_counter: std::cell::Cell<u32>,
    /// Whether to use factorized execution for multi-hop queries.
    pub(super) factorized_execution: bool,
    /// Variables that hold scalar values (from UNWIND/FOR), not node/edge IDs.
    /// Used by plan_return to assign `LogicalType::Any` instead of `Node`.
    pub(super) scalar_columns: std::cell::RefCell<std::collections::HashSet<String>>,
    /// Variables that hold edge IDs (from MATCH edge patterns).
    /// Used by plan_return to emit `EdgeResolve` instead of `NodeResolve`.
    pub(super) edge_columns: std::cell::RefCell<std::collections::HashSet<String>>,
    /// Optional constraint validator for schema enforcement during mutations.
    pub(super) validator: Option<Arc<dyn ConstraintValidator>>,
    /// Catalog for user-defined procedure lookup.
    pub(super) catalog: Option<Arc<crate::catalog::Catalog>>,
    /// LPG store handle for procedures that need direct index access (vector
    /// and text search reach HNSW / BM25 indexes owned by the LPG store).
    #[cfg(feature = "lpg")]
    pub(super) lpg_store: Option<Arc<grafeo_core::graph::lpg::LpgStore>>,
    /// Shared parameter state for the currently planning correlated Apply.
    /// Set by `plan_apply` before planning the inner operator, consumed by
    /// `plan_operator` when encountering `ParameterScan`.
    pub(super) correlated_param_state:
        std::cell::RefCell<Option<Arc<grafeo_core::execution::operators::ParameterState>>>,
    /// Variables from variable-length expand patterns (group-list variables).
    /// Used by the aggregate planner to detect horizontal aggregation (GE09).
    pub(super) group_list_variables: std::cell::RefCell<std::collections::HashSet<String>>,
    /// When true, each physical operator is wrapped in `ProfiledOperator`.
    profiling: std::cell::Cell<bool>,
    /// Profile entries collected during planning (post-order).
    profile_entries: std::cell::RefCell<Vec<crate::query::profile::ProfileEntry>>,
    /// Optional write tracker for recording writes during mutations.
    write_tracker: Option<grafeo_core::execution::operators::SharedWriteTracker>,
    /// Session context for introspection functions (info, schema, current_schema, etc.).
    pub(super) session_context: grafeo_core::execution::operators::SessionContext,
    /// When true, expand operators use epoch-only visibility (no MVCC version
    /// chain walks).  Set when the plan contains no mutations, so PENDING
    /// writes are impossible to observe.
    pub(super) read_only: bool,
    /// LIMIT hint pushed down from `plan_limit` to leaf scan operators.
    ///
    /// Phase 4e: when `plan_limit`'s input is a Filter that plans to a
    /// `RangeScanOperator`, the inner operator gets `with_limit(n)` so
    /// its materialization step terminates after `n` matches. The outer
    /// `LimitOperator` still wraps the result for correctness.
    ///
    /// Save-and-restore pattern: callers set the hint via
    /// `Cell::replace`, recurse, then restore. This handles nested
    /// LIMITs (e.g. subqueries) without state leaking across scopes.
    pub(super) limit_hint: std::cell::Cell<Option<usize>>,
}

impl Planner {
    /// Creates a new planner with the given store.
    ///
    /// This creates a planner without transaction context, using the current
    /// epoch from the store for visibility.
    #[must_use]
    pub fn new(store: Arc<dyn GraphStoreSearch>) -> Self {
        let epoch = store.current_epoch();
        Self {
            store,
            write_store: None,
            transaction_manager: None,
            transaction_id: None,
            viewing_epoch: epoch,
            anon_edge_counter: std::cell::Cell::new(0),
            factorized_execution: true,
            scalar_columns: std::cell::RefCell::new(std::collections::HashSet::new()),
            edge_columns: std::cell::RefCell::new(std::collections::HashSet::new()),
            validator: None,
            catalog: None,
            #[cfg(feature = "lpg")]
            lpg_store: None,
            correlated_param_state: std::cell::RefCell::new(None),
            group_list_variables: std::cell::RefCell::new(std::collections::HashSet::new()),
            profiling: std::cell::Cell::new(false),
            profile_entries: std::cell::RefCell::new(Vec::new()),
            write_tracker: None,
            session_context: grafeo_core::execution::operators::SessionContext::default(),
            read_only: false,
            limit_hint: std::cell::Cell::new(None),
        }
    }

    /// Creates a new planner with transaction context for MVCC-aware planning.
    #[must_use]
    pub fn with_context(
        store: Arc<dyn GraphStoreSearch>,
        write_store: Option<Arc<dyn GraphStoreMut>>,
        transaction_manager: Arc<TransactionManager>,
        transaction_id: Option<TransactionId>,
        viewing_epoch: EpochId,
    ) -> Self {
        use crate::transaction::TransactionWriteTracker;

        // Create write tracker when there's an active transaction
        let write_tracker: Option<grafeo_core::execution::operators::SharedWriteTracker> =
            if transaction_id.is_some() {
                Some(Arc::new(TransactionWriteTracker::new(Arc::clone(
                    &transaction_manager,
                ))))
            } else {
                None
            };

        Self {
            store,
            write_store,
            transaction_manager: Some(transaction_manager),
            transaction_id,
            viewing_epoch,
            anon_edge_counter: std::cell::Cell::new(0),
            factorized_execution: true,
            scalar_columns: std::cell::RefCell::new(std::collections::HashSet::new()),
            edge_columns: std::cell::RefCell::new(std::collections::HashSet::new()),
            validator: None,
            catalog: None,
            #[cfg(feature = "lpg")]
            lpg_store: None,
            correlated_param_state: std::cell::RefCell::new(None),
            group_list_variables: std::cell::RefCell::new(std::collections::HashSet::new()),
            profiling: std::cell::Cell::new(false),
            profile_entries: std::cell::RefCell::new(Vec::new()),
            write_tracker,
            session_context: grafeo_core::execution::operators::SessionContext::default(),
            read_only: false,
            limit_hint: std::cell::Cell::new(None),
        }
    }

    /// Marks this planner as planning a read-only query (no mutations),
    /// enabling fast-path visibility checks in expand operators.
    #[must_use]
    pub fn with_read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }

    /// Returns the writable store, or `TransactionError::ReadOnly` if unavailable.
    fn write_store(&self) -> Result<Arc<dyn GraphStoreMut>> {
        self.write_store
            .as_ref()
            .map(Arc::clone)
            .ok_or(Error::Transaction(
                grafeo_common::utils::error::TransactionError::ReadOnly,
            ))
    }

    /// Returns the viewing epoch for this planner.
    #[must_use]
    pub fn viewing_epoch(&self) -> EpochId {
        self.viewing_epoch
    }

    /// Returns the transaction ID for this planner, if any.
    #[must_use]
    pub fn transaction_id(&self) -> Option<TransactionId> {
        self.transaction_id
    }

    /// Returns a reference to the transaction manager, if available.
    #[must_use]
    pub fn transaction_manager(&self) -> Option<&Arc<TransactionManager>> {
        self.transaction_manager.as_ref()
    }

    /// Enables or disables factorized execution for multi-hop queries.
    #[must_use]
    pub fn with_factorized_execution(mut self, enabled: bool) -> Self {
        self.factorized_execution = enabled;
        self
    }

    /// Sets the constraint validator for schema enforcement during mutations.
    #[must_use]
    pub fn with_validator(mut self, validator: Arc<dyn ConstraintValidator>) -> Self {
        self.validator = Some(validator);
        self
    }

    /// Sets the catalog for user-defined procedure lookup.
    #[must_use]
    pub fn with_catalog(mut self, catalog: Arc<crate::catalog::Catalog>) -> Self {
        self.catalog = Some(catalog);
        self
    }

    /// Attaches an LPG store handle so `CALL grafeo.search.*` procedures can
    /// reach the vector and text indexes.
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn with_lpg_store(mut self, lpg_store: Arc<grafeo_core::graph::lpg::LpgStore>) -> Self {
        self.lpg_store = Some(lpg_store);
        self
    }

    /// Sets the session context for introspection functions.
    #[must_use]
    pub fn with_session_context(
        mut self,
        context: grafeo_core::execution::operators::SessionContext,
    ) -> Self {
        self.session_context = context;
        self
    }

    /// Generates an edge column name from an expand's edge variable (or an
    /// anonymous fallback) and registers it in `edge_columns` so downstream
    /// RETURN emits `EdgeResolve` instead of `NodeResolve`.
    ///
    /// Returns the column name for the caller to push into its output columns.
    pub(super) fn register_edge_column(&self, edge_variable: &Option<String>) -> String {
        let name = edge_variable.clone().unwrap_or_else(|| {
            let count = self.anon_edge_counter.get();
            self.anon_edge_counter.set(count + 1);
            format!("_anon_edge_{}", count)
        });
        self.edge_columns.borrow_mut().insert(name.clone());
        name
    }

    /// Counts consecutive single-hop expand operations.
    ///
    /// Returns the count and the deepest non-expand operator (the base of the chain).
    fn count_expand_chain(op: &LogicalOperator) -> (usize, &LogicalOperator) {
        match op {
            LogicalOperator::Expand(expand) => {
                let is_single_hop = expand.min_hops == 1 && expand.max_hops == Some(1);

                if is_single_hop {
                    let (inner_count, base) = Self::count_expand_chain(&expand.input);
                    (inner_count + 1, base)
                } else {
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
            let is_single_hop = expand.min_hops == 1 && expand.max_hops == Some(1);
            if !is_single_hop {
                break;
            }
            chain.push(expand);
            current = &expand.input;
        }

        chain.reverse();
        chain
    }

    /// Plans a logical plan into a physical operator.
    ///
    /// # Errors
    ///
    /// Returns an error if the logical plan contains unsupported operators
    /// or invalid expressions.
    pub fn plan(&self, logical_plan: &LogicalPlan) -> Result<PhysicalPlan> {
        let _span = grafeo_debug_span!("grafeo::query::plan");
        let (operator, columns) = self.plan_operator(&logical_plan.root)?;
        Ok(PhysicalPlan {
            operator,
            columns,
            adaptive_context: None,
        })
    }

    /// Plans a logical plan with profiling: each physical operator is wrapped
    /// in [`ProfiledOperator`](grafeo_core::execution::ProfiledOperator) to
    /// collect row counts and timing. Returns the physical plan together with
    /// the collected [`ProfileEntry`](crate::query::profile::ProfileEntry)
    /// items in post-order (children before parents).
    ///
    /// # Errors
    ///
    /// Returns an error if the logical plan contains unsupported operators
    /// or invalid expressions.
    pub fn plan_profiled(
        &self,
        logical_plan: &LogicalPlan,
    ) -> Result<(PhysicalPlan, Vec<crate::query::profile::ProfileEntry>)> {
        self.profiling.set(true);
        self.profile_entries.borrow_mut().clear();

        let result = self.plan_operator(&logical_plan.root);

        self.profiling.set(false);
        let (operator, columns) = result?;
        let entries = self.profile_entries.borrow_mut().drain(..).collect();

        Ok((
            PhysicalPlan {
                operator,
                columns,
                adaptive_context: None,
            },
            entries,
        ))
    }

    /// Plans a logical plan with adaptive execution support.
    ///
    /// # Errors
    ///
    /// Returns an error if the logical plan contains unsupported operators
    /// or invalid expressions.
    pub fn plan_adaptive(&self, logical_plan: &LogicalPlan) -> Result<PhysicalPlan> {
        let (operator, columns) = self.plan_operator(&logical_plan.root)?;

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
                let estimate = if let Some(label) = &scan.label {
                    self.store.nodes_by_label(label).len() as f64
                } else {
                    self.store.node_count() as f64
                };
                let id = format!("scan_{}", scan.variable);
                ctx.set_estimate(&id, estimate);

                if let Some(input) = &scan.input {
                    self.collect_cardinality_estimates(input, ctx, depth + 1);
                }
            }
            LogicalOperator::Filter(filter) => {
                let input_estimate = self.estimate_cardinality(&filter.input);
                let estimate = input_estimate * 0.3;
                let id = format!("filter_{depth}");
                ctx.set_estimate(&id, estimate);

                self.collect_cardinality_estimates(&filter.input, ctx, depth + 1);
            }
            LogicalOperator::Expand(expand) => {
                let input_estimate = self.estimate_cardinality(&expand.input);
                let stats = self.store.statistics();
                let avg_degree = self.estimate_expand_degree(&stats, expand);
                let estimate = input_estimate * avg_degree;
                let id = format!("expand_{}", expand.to_variable);
                ctx.set_estimate(&id, estimate);

                self.collect_cardinality_estimates(&expand.input, ctx, depth + 1);
            }
            LogicalOperator::Join(join) => {
                let left_est = self.estimate_cardinality(&join.left);
                let right_est = self.estimate_cardinality(&join.right);
                let estimate = (left_est * right_est).sqrt();
                let id = format!("join_{depth}");
                ctx.set_estimate(&id, estimate);

                self.collect_cardinality_estimates(&join.left, ctx, depth + 1);
                self.collect_cardinality_estimates(&join.right, ctx, depth + 1);
            }
            LogicalOperator::Aggregate(agg) => {
                let input_estimate = self.estimate_cardinality(&agg.input);
                let estimate = if agg.group_by.is_empty() {
                    1.0
                } else {
                    (input_estimate * 0.1).max(1.0)
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
                let estimate = (input_estimate).min(limit.count.estimate());
                let id = format!("limit_{depth}");
                ctx.set_estimate(&id, estimate);

                self.collect_cardinality_estimates(&limit.input, ctx, depth + 1);
            }
            LogicalOperator::Skip(skip) => {
                let input_estimate = self.estimate_cardinality(&skip.input);
                let estimate = (input_estimate - skip.count.estimate()).max(0.0);
                let id = format!("skip_{depth}");
                ctx.set_estimate(&id, estimate);

                self.collect_cardinality_estimates(&skip.input, ctx, depth + 1);
            }
            LogicalOperator::Sort(sort) => {
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
                .min(limit.count.estimate()),
            LogicalOperator::Skip(skip) => {
                (self.estimate_cardinality(&skip.input) - skip.count.estimate()).max(0.0)
            }
            LogicalOperator::Sort(sort) => self.estimate_cardinality(&sort.input),
            LogicalOperator::Union(union) => union
                .inputs
                .iter()
                .map(|input| self.estimate_cardinality(input))
                .sum(),
            LogicalOperator::Except(except) => {
                let left = self.estimate_cardinality(&except.left);
                let right = self.estimate_cardinality(&except.right);
                (left - right).max(0.0)
            }
            LogicalOperator::Intersect(intersect) => {
                let left = self.estimate_cardinality(&intersect.left);
                let right = self.estimate_cardinality(&intersect.right);
                left.min(right)
            }
            LogicalOperator::Otherwise(otherwise) => self
                .estimate_cardinality(&otherwise.left)
                .max(self.estimate_cardinality(&otherwise.right)),
            _ => 1000.0,
        }
    }

    /// Estimates the average edge degree for an expand operation using store statistics.
    fn estimate_expand_degree(
        &self,
        stats: &grafeo_core::statistics::Statistics,
        expand: &ExpandOp,
    ) -> f64 {
        let outgoing = !matches!(expand.direction, ExpandDirection::Incoming);
        if expand.edge_types.len() == 1 {
            stats.estimate_avg_degree(&expand.edge_types[0], outgoing)
        } else if stats.total_nodes > 0 {
            (stats.total_edges as f64 / stats.total_nodes as f64).max(1.0)
        } else {
            10.0
        }
    }

    /// If profiling is enabled, wraps a planned result in `ProfiledOperator`
    /// and records a [`ProfileEntry`](crate::query::profile::ProfileEntry).
    fn maybe_profile(
        &self,
        result: Result<(Box<dyn Operator>, Vec<String>)>,
        op: &LogicalOperator,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        if self.profiling.get() {
            let (physical, columns) = result?;
            let (entry, stats) =
                crate::query::profile::ProfileEntry::new(physical.name(), op.display_label());
            let profiled = grafeo_core::execution::ProfiledOperator::new(physical, stats);
            self.profile_entries.borrow_mut().push(entry);
            Ok((Box::new(profiled), columns))
        } else {
            result
        }
    }

    /// Plans a single logical operator.
    fn plan_operator(&self, op: &LogicalOperator) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let result = match op {
            LogicalOperator::NodeScan(scan) => self.plan_node_scan(scan),
            LogicalOperator::Expand(expand) => {
                // Factorized chain fuses N Expand logical operators into a
                // single LazyFactorizedChainOperator. It kicks in only when
                // it actually helps: chain_len >= 2 means at least two
                // consecutive Expands, where separate plans would
                // Cartesian-product per hop. For a single hop, the plain
                // `plan_expand` path is cheaper and integrates more naturally
                // with adjacent operators.
                //
                // Disable under PROFILE for the same reason as
                // `try_topk_rewrite` in plan_limit: PROFILE's
                // `build_profile_tree` walks the logical plan expecting one
                // ProfileEntry per logical operator, and a fused chain only
                // emits one entry for the whole chain. The result is a count
                // mismatch and a panic at
                // `crates/grafeo-engine/src/query/profile.rs:75`. Run the
                // unfused per-Expand path under PROFILE so each Expand gets
                // its own entry.
                if self.factorized_execution && !self.profiling.get() {
                    let (chain_len, _base) = Self::count_expand_chain(op);
                    if chain_len >= 2 {
                        return self.maybe_profile(self.plan_expand_chain(op), op);
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
            LogicalOperator::Except(except) => self.plan_except(except),
            LogicalOperator::Intersect(intersect) => self.plan_intersect(intersect),
            LogicalOperator::Otherwise(otherwise) => self.plan_otherwise(otherwise),
            LogicalOperator::Apply(apply) => self.plan_apply(apply),
            LogicalOperator::Distinct(distinct) => self.plan_distinct(distinct),
            LogicalOperator::CreateNode(create) => self.plan_create_node(create),
            LogicalOperator::CreateEdge(create) => self.plan_create_edge(create),
            LogicalOperator::DeleteNode(delete) => self.plan_delete_node(delete),
            LogicalOperator::DeleteEdge(delete) => self.plan_delete_edge(delete),
            LogicalOperator::LeftJoin(left_join) => self.plan_left_join(left_join),
            LogicalOperator::AntiJoin(anti_join) => self.plan_anti_join(anti_join),
            LogicalOperator::Unwind(unwind) => self.plan_unwind(unwind),
            LogicalOperator::Merge(merge) => self.plan_merge(merge),
            LogicalOperator::MergeRelationship(merge_rel) => {
                self.plan_merge_relationship(merge_rel)
            }
            LogicalOperator::AddLabel(add_label) => self.plan_add_label(add_label),
            LogicalOperator::RemoveLabel(remove_label) => self.plan_remove_label(remove_label),
            LogicalOperator::SetProperty(set_prop) => self.plan_set_property(set_prop),
            LogicalOperator::ShortestPath(sp) => self.plan_shortest_path(sp),
            LogicalOperator::MapCollect(mc) => self.plan_map_collect(mc),
            #[cfg(feature = "algos")]
            LogicalOperator::CallProcedure(call) => self.plan_call_procedure(call),
            #[cfg(not(feature = "algos"))]
            LogicalOperator::CallProcedure(_) => Err(Error::Internal(
                "CALL procedures require the 'algos' feature".to_string(),
            )),
            LogicalOperator::ParameterScan(_param_scan) => {
                let state = self
                    .correlated_param_state
                    .borrow()
                    .clone()
                    .ok_or_else(|| {
                        Error::Internal(
                            "ParameterScan without correlated Apply context".to_string(),
                        )
                    })?;
                // Use the actual column names from the ParameterState (which may
                // have been expanded from "*" to real variable names in plan_apply)
                let columns = state.columns.clone();
                let operator: Box<dyn Operator> = Box::new(ParameterScanOperator::new(state));
                Ok((operator, columns))
            }
            LogicalOperator::MultiWayJoin(mwj) => self.plan_multi_way_join(mwj),
            LogicalOperator::HorizontalAggregate(ha) => self.plan_horizontal_aggregate(ha),
            LogicalOperator::LoadData(load) => {
                let operator: Box<dyn Operator> = Box::new(LoadDataOperator::new(
                    load.path.clone(),
                    load.format,
                    load.with_headers,
                    load.field_terminator,
                    load.variable.clone(),
                ));
                Ok((operator, vec![load.variable.clone()]))
            }
            LogicalOperator::Empty => Err(Error::Internal("Empty plan".to_string())),
            #[cfg(feature = "vector-index")]
            LogicalOperator::VectorScan(scan) => self.plan_vector_scan(scan),
            #[cfg(not(feature = "vector-index"))]
            LogicalOperator::VectorScan(_) => Err(Error::Internal(
                "VectorScan requires vector-index feature".to_string(),
            )),
            LogicalOperator::VectorJoin(_) => Err(Error::Internal(
                "VectorJoin requires vector-index feature".to_string(),
            )),
            #[cfg(feature = "text-index")]
            LogicalOperator::TextScan(scan) => self.plan_text_scan(scan),
            #[cfg(not(feature = "text-index"))]
            LogicalOperator::TextScan(_) => Err(Error::Internal(
                "TextScan requires text-index feature".to_string(),
            )),
            _ => Err(Error::Internal(format!(
                "Unsupported operator: {:?}",
                std::mem::discriminant(op)
            ))),
        };
        self.maybe_profile(result, op)
    }

    /// Plans a horizontal aggregate operator (per-row aggregation over a list column).
    fn plan_horizontal_aggregate(
        &self,
        ha: &HorizontalAggregateOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (child_op, child_columns) = self.plan_operator(&ha.input)?;

        let list_col_idx = child_columns
            .iter()
            .position(|c| c == &ha.list_column)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "HorizontalAggregate list column '{}' not found in {:?}",
                    ha.list_column, child_columns
                ))
            })?;

        let entity_kind = match ha.entity_kind {
            LogicalEntityKind::Edge => EntityKind::Edge,
            LogicalEntityKind::Node => EntityKind::Node,
        };

        let function = convert_aggregate_function(ha.function);
        let input_column_count = child_columns.len();

        let operator: Box<dyn Operator> = Box::new(HorizontalAggregateOperator::new(
            child_op,
            list_col_idx,
            entity_kind,
            function,
            ha.property.clone(),
            Arc::clone(&self.store) as Arc<dyn GraphStoreSearch>,
            input_column_count,
        ));

        let mut columns = child_columns;
        columns.push(ha.alias.clone());
        // Mark the result as a scalar column
        self.scalar_columns.borrow_mut().insert(ha.alias.clone());

        Ok((operator, columns))
    }

    /// Plans a `MapCollect` operator that collapses grouped rows into a single Map value.
    fn plan_map_collect(&self, mc: &MapCollectOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (child_op, child_columns) = self.plan_operator(&mc.input)?;
        let key_idx = child_columns
            .iter()
            .position(|c| c == &mc.key_var)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "MapCollect key '{}' not in columns {:?}",
                    mc.key_var, child_columns
                ))
            })?;
        let value_idx = child_columns
            .iter()
            .position(|c| c == &mc.value_var)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "MapCollect value '{}' not in columns {:?}",
                    mc.value_var, child_columns
                ))
            })?;
        let operator = Box::new(MapCollectOperator::new(child_op, key_idx, value_idx));
        self.scalar_columns.borrow_mut().insert(mc.alias.clone());
        Ok((operator, vec![mc.alias.clone()]))
    }

    /// Plans a text search scan operator using BM25 inverted index.
    #[cfg(feature = "text-index")]
    fn plan_text_scan(&self, scan: &TextScanOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        use grafeo_core::execution::operators::TextScanOperator;

        let query_string = match &scan.query {
            LogicalExpression::Literal(Value::String(s)) => s.to_string(),
            LogicalExpression::Parameter(name) => {
                return Err(Error::Internal(format!(
                    "TextScan query parameter ${} not resolved",
                    name
                )));
            }
            _ => {
                return Err(Error::Internal(
                    "TextScan query must be a string literal or parameter".to_string(),
                ));
            }
        };

        let operator: Box<dyn Operator> = if let Some(k) = scan.k {
            Box::new(TextScanOperator::top_k(
                Arc::clone(&self.store),
                &scan.label,
                &scan.property,
                &query_string,
                k,
            ))
        } else if let Some(threshold) = scan.threshold {
            Box::new(TextScanOperator::with_threshold(
                Arc::clone(&self.store),
                &scan.label,
                &scan.property,
                &query_string,
                threshold,
            ))
        } else {
            Box::new(TextScanOperator::top_k(
                Arc::clone(&self.store),
                &scan.label,
                &scan.property,
                &query_string,
                100,
            ))
        };

        let mut columns = vec![scan.variable.clone()];
        if let Some(ref score_col) = scan.score_column {
            columns.push(score_col.clone());
        }

        Ok((operator, columns))
    }

    /// Plans a VectorScan logical operator into a physical VectorScanOperator.
    #[cfg(feature = "vector-index")]
    pub(super) fn plan_vector_scan(
        &self,
        scan: &VectorScanOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        use grafeo_core::execution::operators::VectorScanOperator;
        use grafeo_core::index::vector::DistanceMetric;

        // Hybrid shape `VectorScan(input=graph_pattern)` is not supported by
        // the physical VectorScanOperator: it has no input slot and would
        // silently drop upstream bindings. Reject rather than plan it
        // incorrectly; callers should build a VectorJoin for this case.
        if scan.input.is_some() {
            return Err(Error::Internal(
                "VectorScan with an input subtree is not supported, use VectorJoin for hybrid graph+vector queries".to_string(),
            ));
        }

        let query_vec = self.resolve_vector_literal(&scan.query_vector)?;

        let requested_metric = scan.metric.map(|m| match m {
            VectorMetric::Cosine => DistanceMetric::Cosine,
            VectorMetric::Euclidean => DistanceMetric::Euclidean,
            VectorMetric::DotProduct => DistanceMetric::DotProduct,
            VectorMetric::Manhattan => DistanceMetric::Manhattan,
        });

        // Top-k mode uses HNSW when available; threshold/unbounded mode
        // bounds k to the label's node count so we don't feed usize::MAX
        // into HNSW (which degrades to full traversal and risks overflow
        // in quantized rescore paths even with saturating_mul).
        let k = scan.k.unwrap_or_else(|| {
            scan.label.as_ref().map_or_else(
                || self.store.node_count(),
                |l| self.store.nodes_by_label_count(l),
            )
        });

        // Pick the metric we'll execute under. When the user asked for a
        // specific one, honor it. Otherwise inherit the index's metric (so a
        // cosine-built index drives cosine scoring) or default to Cosine for
        // the unindexed brute-force path.
        let index_metric = scan
            .label
            .as_ref()
            .and_then(|label| self.store.vector_index_metric(label, &scan.property));
        let metric = requested_metric
            .or(index_metric)
            .unwrap_or(DistanceMetric::Cosine);

        // The store's vector_search routes HNSW when an index exists whose
        // metric matches `metric`, and brute-force scan otherwise. No handle
        // downcast; no Arc<dyn Any>.
        let mut operator = VectorScanOperator::new(
            Arc::clone(&self.store),
            scan.label.clone(),
            scan.property.clone(),
            query_vec,
            k,
            metric,
        );

        if let Some(sim) = scan.min_similarity {
            operator = operator.with_min_similarity(sim);
        }
        if let Some(dist) = scan.max_distance {
            operator = operator.with_max_distance(dist);
        }

        let mut columns = vec![scan.variable.clone()];
        // VectorScan always projects a score column keyed by the resolved
        // metric (after index-driven fallback) so downstream score reuse
        // matches the actual distance values written out.
        let metric_tag = match metric {
            DistanceMetric::Cosine => "cos",
            DistanceMetric::Euclidean => "euc",
            DistanceMetric::DotProduct => "dot",
            DistanceMetric::Manhattan => "man",
            // Future metrics (DistanceMetric is #[non_exhaustive]) fall back
            // to a generic tag so they still produce a stable column name.
            _ => "other",
        };
        columns.push(project::vector_score_column_name(
            metric_tag,
            &scan.property,
            &scan.variable,
            &scan.query_vector,
        ));

        Ok((Box::new(operator), columns))
    }

    /// Resolves a LogicalExpression to a Vec<f32> for vector operations.
    #[cfg(feature = "vector-index")]
    pub(super) fn resolve_vector_literal(&self, expr: &LogicalExpression) -> Result<Vec<f32>> {
        // f64→f32 precision loss throughout is intentional: vectors are stored and searched as f32.
        #[allow(clippy::cast_possible_truncation)]
        match expr {
            LogicalExpression::Literal(Value::Vector(v)) => Ok(v.to_vec()),
            LogicalExpression::Literal(Value::List(list)) => {
                let mut vec = Vec::with_capacity(list.len());
                for item in list.iter() {
                    match item {
                        Value::Float64(f) => vec.push(*f as f32),
                        Value::Int64(i) => vec.push(*i as f32),
                        _ => {
                            return Err(Error::Internal(
                                "Vector elements must be numeric".to_string(),
                            ));
                        }
                    }
                }
                Ok(vec)
            }
            // GQL/Cypher parser produces List([Literal(Float64), ...]) for inline vectors like
            // [0.9, 0.1, 0.0] — handle this form by recursively resolving each element.
            LogicalExpression::List(items) => {
                let mut vec = Vec::with_capacity(items.len());
                for item in items {
                    match item {
                        LogicalExpression::Literal(Value::Float64(f)) => vec.push(*f as f32),
                        LogicalExpression::Literal(Value::Int64(i)) => vec.push(*i as f32),
                        _ => {
                            return Err(Error::Internal(
                                "Vector elements must be numeric literals".to_string(),
                            ));
                        }
                    }
                }
                Ok(vec)
            }
            _ => Err(Error::Internal("Expected vector literal".to_string())),
        }
    }
}

/// An operator that yields a static set of rows (for `grafeo.procedures()` etc.).
#[cfg(feature = "algos")]
struct StaticResultOperator {
    rows: Vec<Vec<Value>>,
    column_indices: Vec<usize>,
    row_index: usize,
}

#[cfg(feature = "algos")]
impl Operator for StaticResultOperator {
    fn next(&mut self) -> grafeo_core::execution::operators::OperatorResult {
        use grafeo_core::execution::DataChunk;

        if self.row_index >= self.rows.len() {
            return Ok(None);
        }

        let remaining = self.rows.len() - self.row_index;
        let chunk_rows = remaining.min(1024);
        let col_count = self.column_indices.len();

        let col_types: Vec<LogicalType> = vec![LogicalType::Any; col_count];
        let mut chunk = DataChunk::with_capacity(&col_types, chunk_rows);

        for row_offset in 0..chunk_rows {
            let row = &self.rows[self.row_index + row_offset];
            for (col_idx, &src_idx) in self.column_indices.iter().enumerate() {
                let value = row.get(src_idx).cloned().unwrap_or(Value::Null);
                if let Some(col) = chunk.column_mut(col_idx) {
                    col.push_value(value);
                }
            }
        }
        chunk.set_count(chunk_rows);

        self.row_index += chunk_rows;
        Ok(Some(chunk))
    }

    fn reset(&mut self) {
        self.row_index = 0;
    }

    fn name(&self) -> &'static str {
        "StaticResult"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::plan::{
        AggregateExpr as LogicalAggregateExpr, CreateEdgeOp, CreateNodeOp, DeleteNodeOp,
        DistinctOp as LogicalDistinctOp, ExpandOp, FilterOp, JoinCondition, JoinOp,
        LimitOp as LogicalLimitOp, NodeScanOp, PathMode, ReturnItem, ReturnOp,
        SkipOp as LogicalSkipOp, SortKey, SortOp,
    };
    use grafeo_common::types::Value;
    use grafeo_core::execution::operators::AggregateFunction as PhysicalAggregateFunction;
    use grafeo_core::graph::GraphStoreMut;
    use grafeo_core::graph::lpg::LpgStore;

    fn create_test_store() -> Arc<LpgStore> {
        let store = Arc::new(LpgStore::new().unwrap());
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
                pushdown_hint: None,
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
                pushdown_hint: None,
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
                pushdown_hint: None,
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
                pushdown_hint: None,
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
                pushdown_hint: None,
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
                edge_types: vec!["KNOWS".to_string()],
                min_hops: 1,
                max_hops: Some(1),
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "a".to_string(),
                    label: None,
                    input: None,
                })),
                path_alias: None,
                path_mode: PathMode::Walk,
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
                count: 10.into(),
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
                count: 5.into(),
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

    // ==================== Phase 4e: LIMIT pushdown ====================

    #[test]
    fn test_limit_pushdown_into_range_scan_sets_inner_limit() {
        use grafeo_core::execution::operators::{LimitOperator, RangeScanOperator};
        let store = create_test_store();
        let planner = Planner::new(store);

        // LIMIT 5 over WHERE n.age > 30 over MATCH (n:Person)
        let logical = LogicalPlan::new(LogicalOperator::Limit(LogicalLimitOp {
            count: 5.into(),
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

        let physical = planner.plan(&logical).unwrap();
        let root = physical.into_operator();

        // Top: LimitOperator
        let limit_op = root
            .into_any()
            .downcast::<LimitOperator>()
            .expect("top operator is LimitOperator");
        let (inner, limit_count) = limit_op.into_parts();
        assert_eq!(limit_count, 5, "outer LimitOperator preserves the cap");

        // Inner: RangeScanOperator with limit pushed down
        let range_op = inner
            .into_any()
            .downcast::<RangeScanOperator>()
            .expect("inner operator is RangeScanOperator");
        assert_eq!(
            range_op.limit(),
            Some(5),
            "LIMIT 5 must be pushed into the inner range scan"
        );
    }

    #[test]
    fn test_limit_pushdown_blocked_by_sort_in_between() {
        use grafeo_core::execution::operators::{
            LimitOperator, ProjectOperator, RangeScanOperator, SortOperator,
        };
        let store = create_test_store();
        let planner = Planner::new(store);

        // LIMIT 5 over SORT BY n.name over WHERE n.age > 30 over Scan(n:Person)
        // Sort needs full materialization; the LIMIT must NOT be pushed
        // through it into the range scan.
        let logical = LogicalPlan::new(LogicalOperator::Limit(LogicalLimitOp {
            count: 5.into(),
            input: Box::new(LogicalOperator::Sort(SortOp {
                keys: vec![SortKey {
                    expression: LogicalExpression::Property {
                        variable: "n".to_string(),
                        property: "name".to_string(),
                    },
                    order: SortOrder::Ascending,
                    nulls: None,
                }],
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
            })),
        }));

        let physical = planner.plan(&logical).unwrap();
        let root = physical.into_operator();

        // Walk down: Limit → Sort → RangeScan, asserting at each step.
        let limit_op = root
            .into_any()
            .downcast::<LimitOperator>()
            .expect("top operator is LimitOperator");
        let (after_limit, _cap) = limit_op.into_parts();

        let sort_op = after_limit
            .into_any()
            .downcast::<SortOperator>()
            .expect("operator under Limit must be Sort (Sort blocks pushdown)");
        let (after_sort, _keys) = sort_op.into_parts();

        // Sort wraps its input in a Project that materializes the sort
        // keys. The fold from Filter+NodeScan to RangeScan happens
        // inside that Project.
        let project_op = after_sort
            .into_any()
            .downcast::<ProjectOperator>()
            .expect("operator under Sort must be Project (Sort materializes sort keys)");
        let (after_project, _projs, _types) = project_op.into_parts();

        let range_op = after_project
            .into_any()
            .downcast::<RangeScanOperator>()
            .expect("operator under Project must be RangeScan (Filter folded into scan)");

        // Core assertion of the test: with a Sort between the Limit and
        // the range scan, LIMIT must NOT be pushed into the scan —
        // doing so would let Sort observe a truncated input and lose
        // the globally-smallest rows.
        assert_eq!(
            range_op.limit(),
            None,
            "Sort between Limit and RangeScan must block LIMIT pushdown"
        );

        // Sanity: the planner cleared its scratch limit-hint after
        // planning, so subsequent plans aren't contaminated.
        assert_eq!(
            planner.limit_hint.get(),
            None,
            "limit_hint must be cleared after plan_limit returns"
        );

        // Fixture sanity: a Limit-over-Filter (no Sort) DOES push down,
        // proving the negative result above isn't a misconfigured fixture.
        let direct = LogicalPlan::new(LogicalOperator::Limit(LogicalLimitOp {
            count: 5.into(),
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
        let direct_phys = planner.plan(&direct).unwrap();
        let direct_root = direct_phys.into_operator();
        let direct_limit = direct_root
            .into_any()
            .downcast::<LimitOperator>()
            .expect("direct top is LimitOperator");
        let (direct_inner, _) = direct_limit.into_parts();
        let direct_range = direct_inner
            .into_any()
            .downcast::<RangeScanOperator>()
            .expect("direct inner is RangeScanOperator");
        assert_eq!(
            direct_range.limit(),
            Some(5),
            "fixture sanity: pushdown DOES fire when input is Filter"
        );
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
                    nulls: None,
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
                    nulls: None,
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

    #[test]
    fn test_plan_distinct_with_columns() {
        let store = create_test_store();
        let planner = Planner::new(store);

        // DISTINCT on specific columns (column-specific dedup)
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
                columns: Some(vec!["n".to_string()]),
            })),
        }));

        let physical = planner.plan(&logical).unwrap();
        assert_eq!(physical.columns(), &["n"]);
    }

    #[test]
    fn test_plan_distinct_with_nonexistent_columns() {
        let store = create_test_store();
        let planner = Planner::new(store);

        // When distinct columns don't match any output columns,
        // it falls back to full-row distinct.
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
                columns: Some(vec!["nonexistent".to_string()]),
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
                expression2: None,
                distinct: false,
                alias: Some("cnt".to_string()),
                percentile: None,
                separator: None,
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
                expression2: None,
                distinct: false,
                alias: Some("total".to_string()),
                percentile: None,
                separator: None,
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
                expression2: None,
                distinct: false,
                alias: Some("average".to_string()),
                percentile: None,
                separator: None,
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
                    expression2: None,
                    distinct: false,
                    alias: Some("youngest".to_string()),
                    percentile: None,
                    separator: None,
                },
                LogicalAggregateExpr {
                    function: LogicalAggregateFunction::Max,
                    expression: Some(LogicalExpression::Property {
                        variable: "n".to_string(),
                        property: "age".to_string(),
                    }),
                    expression2: None,
                    distinct: false,
                    alias: Some("oldest".to_string()),
                    percentile: None,
                    separator: None,
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

    fn create_writable_planner(store: &Arc<LpgStore>) -> Planner {
        let mut p = Planner::new(Arc::clone(store) as Arc<dyn GraphStoreSearch>);
        p.write_store = Some(Arc::clone(store) as Arc<dyn GraphStoreMut>);
        p
    }

    #[test]
    fn test_plan_create_node() {
        let store = create_test_store();
        let planner = create_writable_planner(&store);

        // CREATE (n:Person {name: 'Alix'})
        let logical = LogicalPlan::new(LogicalOperator::CreateNode(CreateNodeOp {
            variable: "n".to_string(),
            labels: vec!["Person".to_string()],
            properties: vec![(
                "name".to_string(),
                LogicalExpression::Literal(Value::String("Alix".into())),
            )],
            input: None,
        }));

        let physical = planner.plan(&logical).unwrap();
        assert!(physical.columns().contains(&"n".to_string()));
    }

    #[test]
    fn test_plan_create_edge() {
        let store = create_test_store();
        let planner = create_writable_planner(&store);

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
        let planner = create_writable_planner(&store);

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
        assert!(physical.columns().contains(&"n".to_string()));
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
        let planner = Planner::new(Arc::clone(&store) as Arc<dyn GraphStoreSearch>);

        assert!(planner.transaction_id().is_none());
        assert!(planner.transaction_manager().is_none());
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
                pushdown_hint: None,
            })),
        }));

        let physical = planner.plan_adaptive(&logical).unwrap();
        assert!(physical.adaptive_context.is_some());
    }

    #[test]
    fn test_plan_adaptive_with_expand() {
        let store = create_test_store();
        let planner = Planner::new(Arc::clone(&store) as Arc<dyn GraphStoreSearch>)
            .with_factorized_execution(false);

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
                edge_types: vec!["KNOWS".to_string()],
                min_hops: 1,
                max_hops: Some(1),
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "a".to_string(),
                    label: None,
                    input: None,
                })),
                path_alias: None,
                path_mode: PathMode::Walk,
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
                count: 10.into(),
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
                count: 5.into(),
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
                    nulls: None,
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
                edge_types: vec!["KNOWS".to_string()],
                min_hops: 1,
                max_hops: Some(3),
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "a".to_string(),
                    label: None,
                    input: None,
                })),
                path_alias: None,
                path_mode: PathMode::Walk,
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
                edge_types: vec!["KNOWS".to_string()],
                min_hops: 1,
                max_hops: Some(3),
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "a".to_string(),
                    label: None,
                    input: None,
                })),
                path_alias: Some("p".to_string()),
                path_mode: PathMode::Walk,
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
        let planner = Planner::new(Arc::clone(&store) as Arc<dyn GraphStoreSearch>)
            .with_factorized_execution(false);

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
                edge_types: vec!["KNOWS".to_string()],
                min_hops: 1,
                max_hops: Some(1),
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "a".to_string(),
                    label: None,
                    input: None,
                })),
                path_alias: None,
                path_mode: PathMode::Walk,
            })),
        }));

        let physical = planner.plan(&logical).unwrap();
        assert!(physical.columns().contains(&"a".to_string()));
        assert!(physical.columns().contains(&"b".to_string()));
    }

    #[test]
    fn test_plan_expand_both_directions() {
        let store = create_test_store();
        let planner = Planner::new(Arc::clone(&store) as Arc<dyn GraphStoreSearch>)
            .with_factorized_execution(false);

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
                edge_types: vec!["KNOWS".to_string()],
                min_hops: 1,
                max_hops: Some(1),
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "a".to_string(),
                    label: None,
                    input: None,
                })),
                path_alias: None,
                path_mode: PathMode::Walk,
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
        let transaction_manager = Arc::new(TransactionManager::new());
        let transaction_id = transaction_manager.begin();
        let epoch = transaction_manager.current_epoch();

        let planner = Planner::with_context(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            Some(Arc::clone(&store) as Arc<dyn GraphStoreMut>),
            Arc::clone(&transaction_manager),
            Some(transaction_id),
            epoch,
        );

        assert_eq!(planner.transaction_id(), Some(transaction_id));
        assert!(planner.transaction_manager().is_some());
        assert_eq!(planner.viewing_epoch(), epoch);
    }

    #[test]
    fn test_planner_with_factorized_execution_disabled() {
        let store = create_test_store();
        let planner = Planner::new(Arc::clone(&store) as Arc<dyn GraphStoreSearch>)
            .with_factorized_execution(false);

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
                edge_types: vec![],
                min_hops: 1,
                max_hops: Some(1),
                input: Box::new(LogicalOperator::Expand(ExpandOp {
                    from_variable: "a".to_string(),
                    to_variable: "b".to_string(),
                    edge_variable: None,
                    direction: ExpandDirection::Outgoing,
                    edge_types: vec![],
                    min_hops: 1,
                    max_hops: Some(1),
                    input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                        variable: "a".to_string(),
                        label: None,
                        input: None,
                    })),
                    path_alias: None,
                    path_mode: PathMode::Walk,
                })),
                path_alias: None,
                path_mode: PathMode::Walk,
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
                    nulls: None,
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

    // ==================== Additional Coverage Tests ====================
    //
    // These tests target branches that were not exercised by the original
    // planner tests: builder methods, read-only flag, profiled planning,
    // unsupported operator error paths, and the dispatch branches for every
    // plan_* function reachable through plan_operator.

    use crate::catalog::Catalog;
    use crate::query::plan::{
        AddLabelOp, AntiJoinOp, ApplyOp, BindOp, DeleteEdgeOp, EdgeScanOp, ExceptOp,
        HorizontalAggregateOp, IntersectOp, LeftJoinOp, LoadDataFormat, LoadDataOp, MapCollectOp,
        MergeOp, MergeRelationshipOp, MultiWayJoinOp, OtherwiseOp, ParameterScanOp, RemoveLabelOp,
        SetPropertyOp, ShortestPathOp, TripleComponent, TripleScanOp, UnionOp, UnwindOp,
    };
    use grafeo_core::execution::operators::{Operator, SessionContext};

    fn full_store() -> Arc<LpgStore> {
        // Richer store so expand and shortest path tests have real data.
        let store = Arc::new(LpgStore::new().unwrap());
        let vincent = store.create_node(&["Person"]);
        let jules = store.create_node(&["Person"]);
        let mia = store.create_node(&["Person"]);
        let _company = store.create_node(&["Company"]);
        store.create_edge(vincent, jules, "KNOWS");
        store.create_edge(jules, mia, "KNOWS");
        store
    }

    fn scan_person(var: &str) -> LogicalOperator {
        LogicalOperator::NodeScan(NodeScanOp {
            variable: var.to_string(),
            label: Some("Person".to_string()),
            input: None,
        })
    }

    fn scan_any(var: &str) -> LogicalOperator {
        LogicalOperator::NodeScan(NodeScanOp {
            variable: var.to_string(),
            label: None,
            input: None,
        })
    }

    // ==================== Builder Methods ====================

    #[test]
    fn test_with_read_only_flag() {
        let store = create_test_store();
        let planner =
            Planner::new(Arc::clone(&store) as Arc<dyn GraphStoreSearch>).with_read_only(true);
        assert!(planner.read_only);

        let planner_off =
            Planner::new(Arc::clone(&store) as Arc<dyn GraphStoreSearch>).with_read_only(false);
        assert!(!planner_off.read_only);
    }

    #[test]
    fn test_with_catalog() {
        let store = create_test_store();
        let catalog = Arc::new(Catalog::new());
        let planner = Planner::new(Arc::clone(&store) as Arc<dyn GraphStoreSearch>)
            .with_catalog(Arc::clone(&catalog));
        assert!(planner.catalog.is_some());
    }

    #[test]
    fn test_with_session_context() {
        let store = create_test_store();
        let context = SessionContext {
            current_schema: Some("public".to_string()),
            current_graph: Some("main".to_string()),
            ..SessionContext::default()
        };
        let planner = Planner::new(Arc::clone(&store) as Arc<dyn GraphStoreSearch>)
            .with_session_context(context);
        assert_eq!(
            planner.session_context.current_schema.as_deref(),
            Some("public")
        );
        assert_eq!(
            planner.session_context.current_graph.as_deref(),
            Some("main")
        );
    }

    // ==================== register_edge_column ====================

    #[test]
    fn test_register_edge_column_named() {
        let store = create_test_store();
        let planner = Planner::new(Arc::clone(&store) as Arc<dyn GraphStoreSearch>);
        let name = planner.register_edge_column(&Some("r".to_string()));
        assert_eq!(name, "r");
        assert!(planner.edge_columns.borrow().contains("r"));
    }

    #[test]
    fn test_register_edge_column_anonymous_counter_advances() {
        let store = create_test_store();
        let planner = Planner::new(Arc::clone(&store) as Arc<dyn GraphStoreSearch>);
        let a = planner.register_edge_column(&None);
        let b = planner.register_edge_column(&None);
        assert_eq!(a, "_anon_edge_0");
        assert_eq!(b, "_anon_edge_1");
        assert!(planner.edge_columns.borrow().contains("_anon_edge_0"));
        assert!(planner.edge_columns.borrow().contains("_anon_edge_1"));
    }

    // ==================== write_store() error path ====================

    #[test]
    fn test_create_node_without_write_store_errors() {
        // Read-only planner: CREATE should fail with ReadOnly transaction error.
        let store = create_test_store();
        let planner = Planner::new(store);

        let logical = LogicalPlan::new(LogicalOperator::CreateNode(CreateNodeOp {
            variable: "n".to_string(),
            labels: vec!["Person".to_string()],
            properties: vec![],
            input: None,
        }));

        let result = planner.plan(&logical);
        assert!(result.is_err());
    }

    // ==================== plan_profiled ====================

    #[test]
    fn test_plan_profiled_collects_entries() {
        let store = create_test_store();
        let planner = Planner::new(store);

        let logical = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("n".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(scan_person("n")),
        }));

        let (physical, entries) = planner.plan_profiled(&logical).unwrap();
        assert_eq!(physical.columns(), &["n"]);
        // Post-order: scan, then return (at least two entries).
        assert!(
            entries.len() >= 2,
            "expected entries, got {}",
            entries.len()
        );
        // After profiling, the internal flag is cleared.
        assert!(!planner.profiling.get());
    }

    #[test]
    fn test_plan_profiled_propagates_plan_errors() {
        let store = create_test_store();
        let planner = Planner::new(store);
        let logical = LogicalPlan::new(LogicalOperator::Empty);
        let result = planner.plan_profiled(&logical);
        assert!(result.is_err());
        // Profiling must still be reset to false even on error.
        assert!(!planner.profiling.get());
    }

    // ==================== Unsupported operator error paths ====================

    #[test]
    fn test_plan_edge_scan_is_unsupported() {
        // LPG planner does not handle bare EdgeScan; this hits the catch-all branch.
        let store = create_test_store();
        let planner = Planner::new(store);
        let logical = LogicalPlan::new(LogicalOperator::EdgeScan(EdgeScanOp {
            variable: "e".to_string(),
            edge_types: vec![],
            input: None,
        }));
        let err = planner.plan(&logical).err().expect("plan should fail");
        assert!(format!("{err}").contains("Unsupported operator"));
    }

    #[test]
    fn test_plan_triple_scan_is_unsupported() {
        let store = create_test_store();
        let planner = Planner::new(store);
        let logical = LogicalPlan::new(LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Variable("p".to_string()),
            object: TripleComponent::Variable("o".to_string()),
            graph: None,
            input: None,
            dataset: None,
        }));
        assert!(planner.plan(&logical).is_err());
    }

    #[test]
    fn test_plan_bind_is_unsupported() {
        let store = create_test_store();
        let planner = Planner::new(store);
        let logical = LogicalPlan::new(LogicalOperator::Bind(BindOp {
            expression: LogicalExpression::Literal(Value::Int64(1)),
            variable: "x".to_string(),
            input: Box::new(scan_any("n")),
        }));
        assert!(planner.plan(&logical).is_err());
    }

    #[test]
    fn test_plan_parameter_scan_without_apply_errors() {
        let store = create_test_store();
        let planner = Planner::new(store);
        let logical = LogicalPlan::new(LogicalOperator::ParameterScan(ParameterScanOp {
            columns: vec!["n".to_string()],
        }));
        let err = planner.plan(&logical).err().expect("plan should fail");
        assert!(format!("{err}").contains("ParameterScan"));
    }

    // ==================== plan_operator dispatch branches ====================

    #[test]
    fn test_plan_union_dispatch() {
        let store = create_test_store();
        let planner = Planner::new(store);
        let logical = LogicalPlan::new(LogicalOperator::Union(UnionOp {
            inputs: vec![scan_person("n"), scan_person("n")],
        }));
        let physical = planner.plan(&logical).unwrap();
        assert_eq!(physical.columns(), &["n"]);
    }

    #[test]
    fn test_plan_except_dispatch() {
        let store = create_test_store();
        let planner = Planner::new(store);
        let logical = LogicalPlan::new(LogicalOperator::Except(ExceptOp {
            left: Box::new(scan_person("n")),
            right: Box::new(scan_person("n")),
            all: false,
        }));
        let physical = planner.plan(&logical).unwrap();
        assert_eq!(physical.columns(), &["n"]);
    }

    #[test]
    fn test_plan_intersect_dispatch() {
        let store = create_test_store();
        let planner = Planner::new(store);
        let logical = LogicalPlan::new(LogicalOperator::Intersect(IntersectOp {
            left: Box::new(scan_person("n")),
            right: Box::new(scan_person("n")),
            all: false,
        }));
        let physical = planner.plan(&logical).unwrap();
        assert_eq!(physical.columns(), &["n"]);
    }

    #[test]
    fn test_plan_otherwise_dispatch() {
        let store = create_test_store();
        let planner = Planner::new(store);
        let logical = LogicalPlan::new(LogicalOperator::Otherwise(OtherwiseOp {
            left: Box::new(scan_person("n")),
            right: Box::new(scan_any("n")),
        }));
        let physical = planner.plan(&logical).unwrap();
        assert_eq!(physical.columns(), &["n"]);
    }

    #[test]
    fn test_plan_left_join_dispatch() {
        let store = create_test_store();
        let planner = Planner::new(store);
        let logical = LogicalPlan::new(LogicalOperator::LeftJoin(LeftJoinOp {
            left: Box::new(scan_any("a")),
            right: Box::new(scan_any("b")),
            condition: None,
        }));
        let physical = planner.plan(&logical).unwrap();
        assert!(physical.columns().contains(&"a".to_string()));
        assert!(physical.columns().contains(&"b".to_string()));
    }

    #[test]
    fn test_plan_anti_join_dispatch() {
        let store = create_test_store();
        let planner = Planner::new(store);
        let logical = LogicalPlan::new(LogicalOperator::AntiJoin(AntiJoinOp {
            left: Box::new(scan_any("a")),
            right: Box::new(scan_any("b")),
        }));
        let physical = planner.plan(&logical).unwrap();
        assert!(physical.columns().contains(&"a".to_string()));
    }

    #[test]
    fn test_plan_apply_uncorrelated_dispatch() {
        let store = create_test_store();
        let planner = Planner::new(store);
        let logical = LogicalPlan::new(LogicalOperator::Apply(ApplyOp {
            input: Box::new(scan_any("a")),
            subplan: Box::new(scan_any("b")),
            shared_variables: vec![],
            optional: false,
        }));
        let physical = planner.plan(&logical).unwrap();
        assert!(physical.columns().contains(&"a".to_string()));
        assert!(physical.columns().contains(&"b".to_string()));
    }

    #[test]
    fn test_plan_unwind_literal_list() {
        let store = create_test_store();
        let planner = Planner::new(store);

        // UNWIND [1,2,3] AS x
        let logical = LogicalPlan::new(LogicalOperator::Unwind(UnwindOp {
            expression: LogicalExpression::List(vec![
                LogicalExpression::Literal(Value::Int64(1)),
                LogicalExpression::Literal(Value::Int64(2)),
                LogicalExpression::Literal(Value::Int64(3)),
            ]),
            variable: "x".to_string(),
            ordinality_var: None,
            offset_var: None,
            input: Box::new(LogicalOperator::Empty),
        }));
        let physical = planner.plan(&logical).unwrap();
        assert!(physical.columns().contains(&"x".to_string()));
    }

    #[test]
    fn test_plan_merge_dispatch() {
        let store = create_test_store();
        let planner = create_writable_planner(&store);

        // MERGE (n:Person)
        let logical = LogicalPlan::new(LogicalOperator::Merge(MergeOp {
            variable: "n".to_string(),
            labels: vec!["Person".to_string()],
            match_properties: vec![],
            on_create: vec![],
            on_match: vec![],
            input: Box::new(LogicalOperator::Empty),
        }));
        let physical = planner.plan(&logical).unwrap();
        assert!(physical.columns().contains(&"n".to_string()));
    }

    #[test]
    fn test_plan_merge_relationship_dispatch() {
        let store = full_store();
        let planner = create_writable_planner(&store);

        // MATCH (a:Person),(b:Person) MERGE (a)-[r:KNOWS]->(b)
        let logical = LogicalPlan::new(LogicalOperator::MergeRelationship(MergeRelationshipOp {
            variable: "r".to_string(),
            source_variable: "a".to_string(),
            target_variable: "b".to_string(),
            edge_type: "KNOWS".to_string(),
            match_properties: vec![],
            on_create: vec![],
            on_match: vec![],
            input: Box::new(LogicalOperator::Join(JoinOp {
                left: Box::new(scan_person("a")),
                right: Box::new(scan_person("b")),
                join_type: JoinType::Cross,
                conditions: vec![],
            })),
        }));
        let physical = planner.plan(&logical).unwrap();
        assert!(physical.columns().contains(&"r".to_string()));
    }

    #[test]
    fn test_plan_add_label_dispatch() {
        let store = full_store();
        let planner = create_writable_planner(&store);
        let logical = LogicalPlan::new(LogicalOperator::AddLabel(AddLabelOp {
            variable: "n".to_string(),
            labels: vec!["VIP".to_string()],
            input: Box::new(scan_person("n")),
        }));
        let physical = planner.plan(&logical).unwrap();
        assert!(physical.columns().contains(&"labels_added".to_string()));
    }

    #[test]
    fn test_plan_remove_label_dispatch() {
        let store = full_store();
        let planner = create_writable_planner(&store);
        let logical = LogicalPlan::new(LogicalOperator::RemoveLabel(RemoveLabelOp {
            variable: "n".to_string(),
            labels: vec!["Person".to_string()],
            input: Box::new(scan_person("n")),
        }));
        let physical = planner.plan(&logical).unwrap();
        assert!(physical.columns().contains(&"labels_removed".to_string()));
    }

    #[test]
    fn test_plan_set_property_dispatch() {
        let store = full_store();
        let planner = create_writable_planner(&store);
        let logical = LogicalPlan::new(LogicalOperator::SetProperty(SetPropertyOp {
            variable: "n".to_string(),
            properties: vec![(
                "city".to_string(),
                LogicalExpression::Literal(Value::String("Amsterdam".into())),
            )],
            replace: false,
            is_edge: false,
            input: Box::new(scan_person("n")),
        }));
        let physical = planner.plan(&logical).unwrap();
        assert!(physical.columns().contains(&"n".to_string()));
    }

    #[test]
    fn test_plan_delete_edge_dispatch() {
        let store = full_store();
        let planner = create_writable_planner(&store);

        // Register the edge column first via an outgoing expand, then DELETE r.
        let expand_op = LogicalOperator::Expand(ExpandOp {
            from_variable: "a".to_string(),
            to_variable: "b".to_string(),
            edge_variable: Some("r".to_string()),
            direction: ExpandDirection::Outgoing,
            edge_types: vec!["KNOWS".to_string()],
            min_hops: 1,
            max_hops: Some(1),
            input: Box::new(scan_person("a")),
            path_alias: None,
            path_mode: PathMode::Walk,
        });
        let logical = LogicalPlan::new(LogicalOperator::DeleteEdge(DeleteEdgeOp {
            variable: "r".to_string(),
            input: Box::new(expand_op),
        }));
        let physical = planner.plan(&logical).unwrap();
        assert!(physical.columns().contains(&"r".to_string()));
    }

    #[test]
    fn test_plan_shortest_path_dispatch() {
        let store = full_store();
        let planner = Planner::new(Arc::clone(&store) as Arc<dyn GraphStoreSearch>);

        // SHORTEST PATH (a)-(b)
        let logical = LogicalPlan::new(LogicalOperator::ShortestPath(ShortestPathOp {
            input: Box::new(LogicalOperator::Join(JoinOp {
                left: Box::new(scan_person("a")),
                right: Box::new(scan_person("b")),
                join_type: JoinType::Cross,
                conditions: vec![],
            })),
            source_var: "a".to_string(),
            target_var: "b".to_string(),
            edge_types: vec!["KNOWS".to_string()],
            direction: ExpandDirection::Outgoing,
            path_alias: "p".to_string(),
            all_paths: false,
        }));
        let physical = planner.plan(&logical).unwrap();
        assert!(
            physical
                .columns()
                .iter()
                .any(|c| c.contains("_path_length_p"))
        );
    }

    #[test]
    fn test_plan_shortest_path_missing_source_errors() {
        let store = full_store();
        let planner = Planner::new(Arc::clone(&store) as Arc<dyn GraphStoreSearch>);
        let logical = LogicalPlan::new(LogicalOperator::ShortestPath(ShortestPathOp {
            input: Box::new(scan_person("a")),
            source_var: "missing".to_string(),
            target_var: "a".to_string(),
            edge_types: vec![],
            direction: ExpandDirection::Both,
            path_alias: "p".to_string(),
            all_paths: false,
        }));
        let err = planner.plan(&logical).err().expect("plan should fail");
        assert!(format!("{err}").contains("Source variable"));
    }

    #[test]
    fn test_plan_map_collect_dispatch() {
        // Build rows with two columns named 'k' and 'v', then collect k->v into a map.
        let store = create_test_store();
        let planner = Planner::new(store);
        let input_with_kv = LogicalOperator::Project(crate::query::plan::ProjectOp {
            projections: vec![
                crate::query::plan::Projection {
                    expression: LogicalExpression::Literal(Value::String("key".into())),
                    alias: Some("k".to_string()),
                },
                crate::query::plan::Projection {
                    expression: LogicalExpression::Literal(Value::Int64(1)),
                    alias: Some("v".to_string()),
                },
            ],
            input: Box::new(scan_person("n")),
            pass_through_input: false,
        });
        let logical = LogicalPlan::new(LogicalOperator::MapCollect(MapCollectOp {
            key_var: "k".to_string(),
            value_var: "v".to_string(),
            alias: "m".to_string(),
            input: Box::new(input_with_kv),
        }));
        let physical = planner.plan(&logical).unwrap();
        assert_eq!(physical.columns(), &["m"]);
    }

    #[test]
    fn test_plan_map_collect_missing_key_errors() {
        let store = create_test_store();
        let planner = Planner::new(store);
        let logical = LogicalPlan::new(LogicalOperator::MapCollect(MapCollectOp {
            key_var: "not_there".to_string(),
            value_var: "also_missing".to_string(),
            alias: "m".to_string(),
            input: Box::new(scan_any("n")),
        }));
        let err = planner.plan(&logical).err().expect("plan should fail");
        let msg = format!("{err}");
        assert!(msg.contains("MapCollect key"), "got: {msg}");
    }

    #[test]
    fn test_plan_map_collect_missing_value_errors() {
        let store = create_test_store();
        let planner = Planner::new(store);
        // Input has column "n" so key resolves but value does not.
        let logical = LogicalPlan::new(LogicalOperator::MapCollect(MapCollectOp {
            key_var: "n".to_string(),
            value_var: "missing_value".to_string(),
            alias: "m".to_string(),
            input: Box::new(scan_any("n")),
        }));
        let err = planner.plan(&logical).err().expect("plan should fail");
        let msg = format!("{err}");
        assert!(msg.contains("MapCollect value"), "got: {msg}");
    }

    #[test]
    fn test_plan_horizontal_aggregate_missing_column_errors() {
        let store = create_test_store();
        let planner = Planner::new(store);
        let logical = LogicalPlan::new(LogicalOperator::HorizontalAggregate(
            HorizontalAggregateOp {
                list_column: "not_a_column".to_string(),
                entity_kind: crate::query::plan::EntityKind::Edge,
                function: LogicalAggregateFunction::Count,
                property: "age".to_string(),
                alias: "total".to_string(),
                input: Box::new(scan_any("n")),
            },
        ));
        let err = planner.plan(&logical).err().expect("plan should fail");
        assert!(format!("{err}").contains("HorizontalAggregate"));
    }

    #[test]
    fn test_plan_load_data_dispatch() {
        let store = create_test_store();
        let planner = Planner::new(store);
        // Path does not need to exist: planning just builds the operator.
        let logical = LogicalPlan::new(LogicalOperator::LoadData(LoadDataOp {
            format: LoadDataFormat::Csv,
            with_headers: true,
            path: "/nonexistent/data.csv".to_string(),
            variable: "row".to_string(),
            field_terminator: Some(','),
        }));
        let physical = planner.plan(&logical).unwrap();
        assert_eq!(physical.columns(), &["row"]);
    }

    #[test]
    fn test_plan_multi_way_join_dispatch() {
        // Three-way join over three expand inputs. Some configurations may
        // error during planning; we just require no panic.
        let store = full_store();
        let planner = Planner::new(Arc::clone(&store) as Arc<dyn GraphStoreSearch>);
        let ab = LogicalOperator::Expand(ExpandOp {
            from_variable: "a".to_string(),
            to_variable: "b".to_string(),
            edge_variable: None,
            direction: ExpandDirection::Outgoing,
            edge_types: vec!["KNOWS".to_string()],
            min_hops: 1,
            max_hops: Some(1),
            input: Box::new(scan_person("a")),
            path_alias: None,
            path_mode: PathMode::Walk,
        });
        let bc = LogicalOperator::Expand(ExpandOp {
            from_variable: "b".to_string(),
            to_variable: "c".to_string(),
            edge_variable: None,
            direction: ExpandDirection::Outgoing,
            edge_types: vec!["KNOWS".to_string()],
            min_hops: 1,
            max_hops: Some(1),
            input: Box::new(scan_person("b")),
            path_alias: None,
            path_mode: PathMode::Walk,
        });
        let ca = LogicalOperator::Expand(ExpandOp {
            from_variable: "c".to_string(),
            to_variable: "a".to_string(),
            edge_variable: None,
            direction: ExpandDirection::Outgoing,
            edge_types: vec!["KNOWS".to_string()],
            min_hops: 1,
            max_hops: Some(1),
            input: Box::new(scan_person("c")),
            path_alias: None,
            path_mode: PathMode::Walk,
        });
        let logical = LogicalPlan::new(LogicalOperator::MultiWayJoin(MultiWayJoinOp {
            inputs: vec![ab, bc, ca],
            conditions: vec![],
            shared_variables: vec!["a".to_string(), "b".to_string(), "c".to_string()],
        }));
        let _ = planner.plan(&logical);
    }

    #[test]
    fn test_plan_horizontal_aggregate_dispatch() {
        // Variable-length expand produces a list column that the aggregate targets.
        let store = full_store();
        let planner = Planner::new(Arc::clone(&store) as Arc<dyn GraphStoreSearch>);

        let path = LogicalOperator::Expand(ExpandOp {
            from_variable: "a".to_string(),
            to_variable: "b".to_string(),
            edge_variable: Some("r".to_string()),
            direction: ExpandDirection::Outgoing,
            edge_types: vec!["KNOWS".to_string()],
            min_hops: 1,
            max_hops: Some(3),
            input: Box::new(scan_person("a")),
            path_alias: Some("p".to_string()),
            path_mode: PathMode::Walk,
        });
        // Variable-length expand emits a column named _path_edges_p.
        let logical = LogicalPlan::new(LogicalOperator::HorizontalAggregate(
            HorizontalAggregateOp {
                list_column: "_path_edges_p".to_string(),
                entity_kind: crate::query::plan::EntityKind::Edge,
                function: LogicalAggregateFunction::Count,
                property: "weight".to_string(),
                alias: "edge_count".to_string(),
                input: Box::new(path),
            },
        ));
        let physical = planner.plan(&logical).unwrap();
        assert!(physical.columns().contains(&"edge_count".to_string()));
    }

    // ==================== Cardinality estimation branches ====================

    #[test]
    fn test_plan_adaptive_with_except() {
        let store = create_test_store();
        let planner = Planner::new(store);
        let logical = LogicalPlan::new(LogicalOperator::Except(ExceptOp {
            left: Box::new(scan_person("n")),
            right: Box::new(scan_person("n")),
            all: false,
        }));
        let physical = planner.plan_adaptive(&logical).unwrap();
        assert!(physical.adaptive_context.is_some());
    }

    #[test]
    fn test_plan_adaptive_with_intersect() {
        let store = create_test_store();
        let planner = Planner::new(store);
        let logical = LogicalPlan::new(LogicalOperator::Intersect(IntersectOp {
            left: Box::new(scan_person("n")),
            right: Box::new(scan_any("n")),
            all: false,
        }));
        let physical = planner.plan_adaptive(&logical).unwrap();
        assert!(physical.adaptive_context.is_some());
    }

    #[test]
    fn test_plan_adaptive_with_otherwise() {
        let store = create_test_store();
        let planner = Planner::new(store);
        let logical = LogicalPlan::new(LogicalOperator::Otherwise(OtherwiseOp {
            left: Box::new(scan_person("n")),
            right: Box::new(scan_any("n")),
        }));
        let physical = planner.plan_adaptive(&logical).unwrap();
        assert!(physical.adaptive_context.is_some());
    }

    // ==================== count_expand_chain edge case ====================

    #[test]
    fn test_count_expand_chain_variable_length_breaks_chain() {
        // A variable-length expand (not single-hop) should NOT count in the chain.
        let var_expand = LogicalOperator::Expand(ExpandOp {
            from_variable: "a".to_string(),
            to_variable: "b".to_string(),
            edge_variable: None,
            direction: ExpandDirection::Outgoing,
            edge_types: vec!["KNOWS".to_string()],
            min_hops: 1,
            max_hops: Some(3),
            input: Box::new(scan_person("a")),
            path_alias: None,
            path_mode: PathMode::Walk,
        });
        let (count, _) = Planner::count_expand_chain(&var_expand);
        assert_eq!(count, 0);
    }

    // ==================== StaticResultOperator ====================

    #[cfg(feature = "algos")]
    #[test]
    fn test_static_result_operator_emits_rows_and_resets() {
        use grafeo_common::types::Value;
        let rows = vec![
            vec![Value::Int64(1), Value::String("Vincent".into())],
            vec![Value::Int64(2), Value::String("Jules".into())],
        ];
        let mut op = StaticResultOperator {
            rows,
            column_indices: vec![0, 1],
            row_index: 0,
        };
        assert_eq!(op.name(), "StaticResult");
        let chunk = op.next().unwrap().expect("first chunk");
        assert_eq!(chunk.row_count(), 2);
        // Exhausted.
        assert!(op.next().unwrap().is_none());
        // Reset allows re-emitting.
        op.reset();
        assert!(op.next().unwrap().is_some());
        // into_any round trip.
        let boxed: Box<dyn Operator> = Box::new(StaticResultOperator {
            rows: vec![vec![Value::Null]],
            column_indices: vec![0],
            row_index: 0,
        });
        let _any = boxed.into_any();
    }

    // ==================== VectorScan k-bounding regression ====================

    /// `VectorScanOp.k = None` means "return every match," but feeding
    /// `usize::MAX` into the store's vector_search degrades HNSW to a full
    /// traversal and (pre-saturating-mul) could overflow quantized rescore
    /// paths. The planner must bound k to the label's node count (or the
    /// global count when no label is set, or zero when the label is unknown).
    ///
    /// We can't inspect VectorScanOperator.k directly — it's private — so the
    /// assertion here is that planning succeeds across all three branches
    /// without panicking or surfacing an internal-state error. Paired with
    /// `LpgStore::nodes_by_label_count` tests, this guards the full path.
    #[cfg(feature = "vector-index")]
    #[test]
    fn test_plan_vector_scan_k_none_bounds_across_label_states() {
        use crate::query::plan::VectorScanOp;

        let store = create_test_store();
        // Sanity: the test store has 2 Person nodes, 1 Company, 0 Unknown.
        assert_eq!(store.nodes_by_label_count("Person"), 2);
        assert_eq!(store.nodes_by_label_count("Unknown"), 0);
        assert_eq!(store.node_count(), 3);

        let planner = Planner::new(store);
        let make_scan = |label: Option<&str>| VectorScanOp {
            variable: "n".to_string(),
            index_name: None,
            property: "embedding".to_string(),
            label: label.map(str::to_string),
            query_vector: LogicalExpression::Literal(Value::List(
                vec![
                    Value::Float64(1.0),
                    Value::Float64(0.0),
                    Value::Float64(0.0),
                ]
                .into(),
            )),
            k: None,
            metric: Some(VectorMetric::Cosine),
            min_similarity: Some(0.5),
            max_distance: None,
            input: None,
        };

        for label in [Some("Person"), Some("Unknown"), None] {
            let (_op, cols) = planner
                .plan_vector_scan(&make_scan(label))
                .unwrap_or_else(|e| panic!("plan_vector_scan failed for {label:?}: {e:?}"));
            assert_eq!(cols[0], "n", "variable column must be first for {label:?}");
        }
    }
}
