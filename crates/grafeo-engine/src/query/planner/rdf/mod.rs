//! RDF Query Planner.
//!
//! Converts logical plans with RDF operators (TripleScan, etc.) to physical
//! operators that execute against an RDF store.
//!
//! This planner follows the same push-based, vectorized execution model as
//! the LPG planner for consistent performance characteristics.

use std::collections::HashMap;
use std::sync::Arc;

use grafeo_common::change::RdfGraphOp;
use grafeo_common::storage::log_record::RdfGraphTarget;
use grafeo_common::types::{LogicalType, Value};
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result};
use grafeo_core::execution::DataChunk;
use grafeo_core::execution::operators::{
    FilterExpression, FilterOperator, HashAggregateOperator, Operator, OperatorError,
    SimpleAggregateOperator, SingleRowOperator, StatementClock,
};
use grafeo_core::execution::vector::ValueVector;
use grafeo_core::graph::rdf::{Literal, RdfStore, Term, Triple, TriplePattern};

use crate::query::plan::{
    AddGraphOp, AggregateFunction as LogicalAggregateFunction, AggregateOp, AntiJoinOp, BindOp,
    ClearGraphOp, ConstructOp, CopyGraphOp, CreateGraphOp, DatasetRestriction, DeleteTripleOp,
    DistinctOp, DropGraphOp, FilterOp, InsertTripleOp, LeftJoinOp, LimitOp, LogicalExpression,
    LogicalOperator, LogicalPlan, ModifyOp, MoveGraphOp, SkipOp, SortOp, TripleComponent,
    TripleScanOp, TripleTemplate,
};
use crate::query::planner::{PhysicalPlan, convert_aggregate_function, convert_filter_expression};
use crate::transaction::RdfWriter;

mod evaluator;
mod exists_pair;
mod exists_per_row;
mod exists_read;
mod named_graphs;
mod order;
mod path_per_row;
mod term_ids;
mod terms;
mod triple_lookup;

use evaluator::{ExpressionContext, RdfExpressionPredicate};
use exists_per_row::{ExistsPerRow, RowBinding, RowPlanParts};
use grafeo_common::types::TermId;
use term_ids::{QueryTermIds, id_of, id_value};

/// Default chunk size for morsel-driven execution.
const DEFAULT_CHUNK_SIZE: usize = 1024;

/// Groups the variable-substitution operands for pattern-based mutation operators.
///
/// Used to keep `RdfInsertPatternOperator::new` and `RdfDeletePatternOperator::new`
/// within the 7-argument clippy limit.
struct TripleOperands {
    subject: TripleComponent,
    predicate: TripleComponent,
    object: TripleComponent,
    column_map: HashMap<String, usize>,
    /// Named graph to mutate (`None` = default graph).
    graph: Option<String>,
    /// The query's term dictionary, which the variable columns index.
    terms: Arc<QueryTermIds>,
}

/// Fails for a graph an update cannot target: a `?`-prefixed name is an
/// unbound `GRAPH ?var` target, which is rejected instead of being written
/// to the wrong graph.
fn check_mutation_graph(graph_name: Option<&str>) -> std::result::Result<(), OperatorError> {
    match graph_name {
        Some(name) if name.starts_with('?') => Err(OperatorError::from(
            grafeo_common::utils::error::Error::from(grafeo_common::utils::error::QueryError::new(
                grafeo_common::utils::error::QueryErrorKind::Semantic,
                format!(
                    "SPARQL Update cannot target a variable graph ({name}); use a concrete \
                         graph IRI or a WITH clause"
                ),
            )),
        )),
        _ => Ok(()),
    }
}

/// The writer of the transaction an update runs in. A session runs every
/// update in a transaction and plans it with that transaction's writer; a
/// plan made without one (for `EXPLAIN`) does not run an update.
fn update_writer(writer: &Option<RdfWriter>) -> std::result::Result<&RdfWriter, OperatorError> {
    writer.as_ref().ok_or_else(|| {
        OperatorError::Internal(
            "an RDF update runs in a transaction, and this plan was made without one".to_string(),
        )
    })
}

/// Converts logical plans with RDF operators to physical operators.
///
/// This planner produces push-based operators that process data in chunks
/// (morsels) for cache efficiency and parallelism compatibility.
pub struct RdfPlanner {
    /// The RDF store to query.
    store: Arc<RdfStore>,
    /// Chunk size for vectorized execution.
    chunk_size: usize,
    /// The writer of the transaction the plan runs in: updates record their
    /// triples through it, and scans see them.
    writer: Option<RdfWriter>,
    /// When true, each physical operator is wrapped in `ProfiledOperator`.
    profiling: std::cell::Cell<bool>,
    /// Profile entries collected during planning (post-order).
    profile_entries: std::cell::RefCell<Vec<crate::query::profile::ProfileEntry>>,
    /// How many `EXISTS` marks it has named.
    exists_marks: std::cell::Cell<usize>,
    /// Set when the operator just planned is its input as it is (a
    /// projection of the input's columns in their order): it runs nothing,
    /// so the profile shows the input in its place.
    folded: std::cell::Cell<bool>,
    /// The term dictionary of the query: every variable column of the plan
    /// holds ids into it (see [`term_ids`]).
    terms: Arc<QueryTermIds>,
    /// Whether results hold the values GraphQL reads for terms
    /// ([`Term::query_value`]) instead of the terms themselves.
    query_values: bool,
    /// Whether a plan without `ORDER BY` returns its rows in random order.
    shuffle_unordered: bool,
    /// One traversal's maximum resident memory, narrowed by the caller's budget.
    path_search_budget: usize,
    /// The same execution deadline used by the caller's executor.
    deadline: Option<grafeo_common::clock::Instant>,
    /// Counts the triples the plan's scans read: the statement's
    /// `rows_scanned`.
    read_counter: Arc<grafeo_core::execution::operators::ReadCounter>,
    /// Where the statement raises its warnings: the translation's, which
    /// [`plan`](Self::plan) raises, and the run's.
    warnings: Arc<grafeo_common::utils::warning::WarningSink>,
    /// The instant the query started, which `NOW()` gives in every row:
    /// the planner is made for one execution of the query.
    clock: StatementClock,
    /// The row this plan is for, when it is the pattern of an `EXISTS` planned per row.
    row: Option<Arc<RowBinding>>,
    /// Whether an `EXISTS` that a run without the row answers runs once.
    exists_run_once: bool,
    /// The routes of the `EXISTS` tests of the operator being planned, which its PROFILE line names.
    exists_routes: std::cell::RefCell<Vec<&'static str>>,
    /// The statement's operator memory context, set as the ambient scope while
    /// this planner plans and converts one statement (patch 9, #751).
    #[cfg(feature = "spill")]
    operator_memory: Option<grafeo_core::execution::OperatorMemoryContext>,
}

impl RdfPlanner {
    /// Creates a new RDF planner with the given store.
    #[must_use]
    pub fn new(store: Arc<RdfStore>) -> Self {
        let terms = Arc::new(QueryTermIds::new(Arc::clone(store.dictionary())));
        Self {
            store,
            chunk_size: DEFAULT_CHUNK_SIZE,
            writer: None,
            profiling: std::cell::Cell::new(false),
            profile_entries: std::cell::RefCell::new(Vec::new()),
            exists_marks: std::cell::Cell::new(0),
            folded: std::cell::Cell::new(false),
            terms,
            query_values: false,
            shuffle_unordered: false,
            path_search_budget: grafeo_core::execution::operators::DEFAULT_PATH_SEARCH_BUDGET,
            deadline: None,
            read_counter: Arc::default(),
            warnings: Arc::default(),
            clock: StatementClock::now(),
            row: None,
            exists_run_once: true,
            exists_routes: std::cell::RefCell::new(Vec::new()),
            #[cfg(feature = "spill")]
            operator_memory: Some(grafeo_core::execution::OperatorMemoryContext::new(
                grafeo_common::memory::buffer::BufferManager::new(
                    grafeo_common::memory::buffer::BufferManagerConfig::default(),
                ),
                None,
            )),
        }
    }

    /// A planner of the pattern of an `EXISTS` for the row `row`.
    fn for_row(parts: &RowPlanParts, row: Arc<RowBinding>) -> Self {
        Self {
            store: Arc::clone(&parts.store),
            chunk_size: parts.chunk_size,
            writer: parts.writer.clone(),
            profiling: std::cell::Cell::new(false),
            profile_entries: std::cell::RefCell::new(Vec::new()),
            exists_marks: std::cell::Cell::new(0),
            folded: std::cell::Cell::new(false),
            terms: Arc::clone(&parts.terms),
            query_values: false,
            shuffle_unordered: false,
            path_search_budget: parts.path_search_budget,
            deadline: parts.deadline,
            read_counter: Arc::clone(&parts.read_counter),
            warnings: Arc::clone(&parts.warnings),
            clock: parts.clock,
            row: Some(row),
            exists_run_once: false,
            exists_routes: std::cell::RefCell::new(Vec::new()),
            #[cfg(feature = "spill")]
            operator_memory: Some(grafeo_core::execution::OperatorMemoryContext::new(
                grafeo_common::memory::buffer::BufferManager::new(
                    grafeo_common::memory::buffer::BufferManagerConfig::default(),
                ),
                None,
            )),
        }
    }

    /// Whether an `EXISTS` that a run without the row answers runs once; off, every one runs per
    /// row, which must give the same answers.
    #[must_use]
    #[cfg(feature = "sparql")]
    pub(crate) fn with_exists_run_once(mut self, run_once: bool) -> Self {
        self.exists_run_once = run_once;
        self
    }

    /// What planning an `EXISTS` per row needs.
    fn row_plan_parts(&self) -> RowPlanParts {
        RowPlanParts {
            store: Arc::clone(&self.store),
            chunk_size: self.chunk_size,
            writer: self.writer.clone(),
            terms: Arc::clone(&self.terms),
            path_search_budget: self.path_search_budget,
            deadline: self.deadline,
            read_counter: Arc::clone(&self.read_counter),
            warnings: Arc::clone(&self.warnings),
            clock: self.clock,
        }
    }

    /// What an expression over the row of the plan reads besides its row.
    fn expression_context(&self) -> ExpressionContext {
        ExpressionContext {
            clock: self.clock,
            row: self.row.clone(),
        }
    }

    /// [`Self::expression_context`] for an expression over the rows of `input`: above an
    /// aggregation the row is gone.
    fn expression_context_over(&self, input: &LogicalOperator) -> ExpressionContext {
        if exists_per_row::over_an_aggregate(input) {
            self.expression_context_with_no_row()
        } else {
            self.expression_context()
        }
    }

    fn expression_context_with_no_row(&self) -> ExpressionContext {
        ExpressionContext {
            clock: self.clock,
            row: None,
        }
    }

    /// How to run `subquery` once for rows of the columns `row`, `None` when it runs per row.
    fn run_once_read<'a>(
        &self,
        subquery: &'a LogicalOperator,
        row: &[String],
    ) -> Option<exists_read::ExistsRead<'a>> {
        if !self.exists_run_once {
            return None;
        }
        exists_read::read_of_exists(subquery, row)
    }

    /// Notes the route of an `EXISTS` for the PROFILE line of the operator that holds it.
    fn note_exists_route(&self, route: &'static str) {
        if self.profiling.get() {
            self.exists_routes.borrow_mut().push(route);
        }
    }

    /// Counts the triples the plan's scans read, before any filter: the
    /// statement's `rows_scanned`.
    #[must_use]
    pub fn read_counter(&self) -> Arc<grafeo_core::execution::operators::ReadCounter> {
        Arc::clone(&self.read_counter)
    }

    /// Where the statement's warnings go (see [`with_warnings`](Self::with_warnings)).
    #[must_use]
    pub fn warnings(&self) -> Arc<grafeo_common::utils::warning::WarningSink> {
        Arc::clone(&self.warnings)
    }

    /// Raises the statement's warnings into `sink`, which the session made
    /// for it.
    #[must_use]
    pub fn with_warnings(mut self, sink: Arc<grafeo_common::utils::warning::WarningSink>) -> Self {
        self.warnings = sink;
        self
    }

    /// Sets the chunk size for vectorized execution.
    #[must_use]
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Returns the rows of plans without `ORDER BY` in random order (the
    /// `shuffle_unordered` test option).
    #[must_use]
    pub fn with_shuffle_unordered(mut self, shuffle: bool) -> Self {
        self.shuffle_unordered = shuffle;
        self
    }

    /// Sets the statement's operator memory context, entered as the ambient
    /// scope while this planner plans one statement (patch 9, #751).
    #[cfg(feature = "spill")]
    #[must_use]
    pub fn with_operator_memory(
        mut self,
        ctx: Option<grafeo_core::execution::OperatorMemoryContext>,
    ) -> Self {
        self.operator_memory = ctx;
        self
    }

    /// Sets the memory one native property-path traversal may retain.
    #[must_use]
    pub fn with_path_search_budget(mut self, bytes: usize) -> Self {
        self.path_search_budget = bytes;
        self
    }

    /// Sets the deadline checked inside native property-path lookups.
    #[must_use]
    pub fn with_deadline(mut self, deadline: Option<grafeo_common::clock::Instant>) -> Self {
        self.deadline = deadline;
        self
    }

    /// Returns results as the values GraphQL reads for terms (an IRI as its
    /// text, a numeric literal as its number) instead of RDF terms.
    #[must_use]
    #[cfg(rdf_language)]
    pub(crate) fn with_query_values(mut self, query_values: bool) -> Self {
        self.query_values = query_values;
        self
    }

    /// Plans with the writer of the transaction the plan runs in: its
    /// updates record their triples through it, and its scans see them.
    /// Only an RDF query runs in a transaction that writes triples.
    #[must_use]
    #[cfg(any(rdf_language, test))]
    pub(crate) fn with_writer(mut self, writer: Option<RdfWriter>) -> Self {
        self.writer = writer;
        self
    }

    /// Whether the plan's transaction wrote triples it has not committed:
    /// its reads must then lay them over the store, which the counts from
    /// the store's indexes and the ring join do not.
    fn has_pending_writes(&self) -> bool {
        self.writer.as_ref().is_some_and(RdfWriter::has_pending)
    }

    /// Plans a logical plan into a physical operator tree.
    ///
    /// # Errors
    ///
    /// Returns an error if planning fails.
    pub fn plan(&self, logical_plan: &LogicalPlan) -> Result<PhysicalPlan> {
        #[cfg(feature = "spill")]
        let _memory_scope = self
            .operator_memory
            .clone()
            .map(grafeo_core::execution::enter_operator_memory);
        self.warnings.raise_all(logical_plan.warnings);
        let (operator, columns, types) =
            self.without_hidden_columns(self.plan_operator(&logical_plan.root)?);
        let operator = self.decode_results(operator, &types, logical_plan.result_kind);
        let operator: Box<dyn Operator> =
            if self.shuffle_unordered && !super::common::orders_rows(&logical_plan.root) {
                Box::new(grafeo_core::execution::operators::ShuffleOperator::new(
                    operator,
                ))
            } else {
                operator
            };
        Ok(PhysicalPlan { operator, columns })
    }

    /// Plans a logical plan with profiling: each physical operator is wrapped
    /// in [`ProfiledOperator`](grafeo_core::execution::ProfiledOperator) to
    /// collect row counts and timing.
    ///
    /// # Errors
    ///
    /// Returns an error if the logical plan contains unsupported SPARQL operators
    /// or invalid expressions.
    pub fn plan_profiled(
        &self,
        logical_plan: &LogicalPlan,
    ) -> Result<(PhysicalPlan, Vec<crate::query::profile::ProfileEntry>)> {
        self.profiling.set(true);
        self.profile_entries.borrow_mut().clear();
        #[cfg(feature = "spill")]
        let _memory_scope = self
            .operator_memory
            .clone()
            .map(grafeo_core::execution::enter_operator_memory);
        self.warnings.raise_all(logical_plan.warnings);

        let result = self.plan_operator(&logical_plan.root);

        self.profiling.set(false);
        let (operator, columns, types) = self.without_hidden_columns(result?);
        let operator = self.decode_results(operator, &types, logical_plan.result_kind);
        let entries = self.profile_entries.borrow_mut().drain(..).collect();

        Ok((PhysicalPlan { operator, columns }, entries))
    }

    /// `planned` without the columns of hidden names, which no solution shows (see
    /// [`is_hidden_variable`](crate::query::plan::is_hidden_variable)).
    fn without_hidden_columns(&self, planned: PlannedOperator) -> PlannedOperator {
        let keep = planned
            .1
            .iter()
            .enumerate()
            .filter(|(_, name)| !crate::query::plan::is_hidden_variable(name))
            .map(|(at, _)| at)
            .collect();
        self.keeping_columns(planned, keep)
    }

    /// Decodes the term id columns of a plan's result: each becomes the
    /// RDF terms it holds; for GraphQL the values it reads for them, and for
    /// ASK its boolean.
    fn decode_results(
        &self,
        operator: Box<dyn Operator>,
        types: &[LogicalType],
        kind: crate::database::ResultKind,
    ) -> Box<dyn Operator> {
        let columns: Vec<usize> = types
            .iter()
            .enumerate()
            .filter(|(_, ty)| **ty == LogicalType::RdfTermId)
            .map(|(i, _)| i)
            .collect();
        if columns.is_empty() {
            return operator;
        }
        let decode_as = if kind == crate::database::ResultKind::Boolean {
            DecodeAs::Boolean
        } else if self.query_values {
            DecodeAs::QueryValues
        } else {
            DecodeAs::Terms
        };
        Box::new(TermDecodeOperator {
            input: operator,
            terms: Arc::clone(&self.terms),
            columns,
            decode_as,
        })
    }

    /// Wraps `operator` so the columns `columns` hold term ids: a term, or a
    /// native value as its literal, is interned.
    fn intern(&self, operator: Box<dyn Operator>, columns: Vec<usize>) -> Box<dyn Operator> {
        Box::new(TermInternOperator {
            input: operator,
            terms: Arc::clone(&self.terms),
            columns,
        })
    }

    /// The id of `term` in the query's dictionary.
    fn term_id(&self, term: &Term) -> Result<Value> {
        let id = self
            .terms
            .intern(term)
            .map_err(|e| Error::Internal(e.to_string()))?;
        Ok(id_value(id))
    }

    /// If profiling is enabled, wraps a planned result in `ProfiledOperator`
    /// and records a [`ProfileEntry`](crate::query::profile::ProfileEntry).
    fn maybe_profile(
        &self,
        result: Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)>,
        op: &LogicalOperator,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        if self.folded.replace(false) {
            if self.profiling.get() && result.is_ok() {
                self.profile_entries.borrow_mut().push(
                    crate::query::profile::ProfileEntry::folded(op.display_label()),
                );
            }
            return result;
        }
        if self.profiling.get() {
            let (physical, columns, types) = result?;
            let mut label = op.display_label();
            for route in self.exists_routes.borrow_mut().drain(..) {
                label.push_str(" [");
                label.push_str(route);
                label.push(']');
            }
            let (entry, stats) = crate::query::profile::ProfileEntry::new(physical.name(), label);
            let profiled = grafeo_core::execution::ProfiledOperator::new(physical, stats);
            self.profile_entries.borrow_mut().push(entry);
            Ok((Box::new(profiled), columns, types))
        } else {
            result
        }
    }

    /// Plans `subquery`, the pattern of an `EXISTS` or `NOT EXISTS` in a
    /// filter or an `OPTIONAL`'s condition, with profiling off: it is part
    /// of an expression, with no node of its own in the logical tree PROFILE
    /// walks, so its rows and time count in the operator that holds it.
    /// Recording its entries would shift every later one onto the wrong
    /// operator.
    fn plan_exists_subquery(
        &self,
        subquery: &LogicalOperator,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        let profiling = self.profiling.replace(false);
        let planned = self.plan_operator(subquery);
        self.profiling.set(profiling);
        planned
    }

    /// Plans a single logical operator.
    fn plan_operator(
        &self,
        op: &LogicalOperator,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        let result = match op {
            LogicalOperator::TripleScan(scan) => self.plan_triple_scan(scan),
            LogicalOperator::PropertyPath(path) => self.plan_property_path(path),
            LogicalOperator::RowValues(values) => self.plan_row_values(values),
            LogicalOperator::NamedGraphs(graphs) => {
                let op: Box<dyn Operator> = Box::new(named_graphs::NamedGraphsOperator::new(
                    Arc::clone(&self.store),
                    self.writer.clone(),
                    graphs.dataset.clone(),
                    Arc::clone(&self.terms),
                ));
                Ok((
                    op,
                    vec![graphs.variable.clone()],
                    vec![LogicalType::RdfTermId],
                ))
            }
            LogicalOperator::InNamedGraph(graph) => self.plan_in_named_graph(graph),
            LogicalOperator::Filter(filter) => self.plan_filter(filter),
            LogicalOperator::Project(project) => self.plan_project(project),
            LogicalOperator::Limit(limit) => self.plan_limit(limit),
            LogicalOperator::Skip(skip) => self.plan_skip(skip),
            LogicalOperator::Sort(sort) => self.plan_sort(sort),
            LogicalOperator::Aggregate(agg) => self.plan_aggregate(agg),
            LogicalOperator::Return(ret) => self.plan_return(ret),
            LogicalOperator::Join(join) => self.plan_join(join),
            LogicalOperator::LeftJoin(join) => self.plan_left_join(join),
            LogicalOperator::AntiJoin(join) => self.plan_anti_join(join),
            LogicalOperator::Union(union) => self.plan_union(union),
            LogicalOperator::Distinct(distinct) => self.plan_distinct(distinct),
            LogicalOperator::InsertTriple(insert) => self.plan_insert_triple(insert),
            LogicalOperator::DeleteTriple(delete) => self.plan_delete_triple(delete),
            LogicalOperator::Modify(modify) => self.plan_modify(modify),
            LogicalOperator::ClearGraph(clear) => self.plan_clear_graph(clear),
            LogicalOperator::CreateGraph(create) => self.plan_create_graph(create),
            LogicalOperator::DropGraph(drop_op) => self.plan_drop_graph(drop_op),
            LogicalOperator::CopyGraph(copy) => self.plan_copy_graph(copy),
            LogicalOperator::MoveGraph(move_op) => self.plan_move_graph(move_op),
            LogicalOperator::AddGraph(add) => self.plan_add_graph(add),
            LogicalOperator::Bind(bind) => self.plan_bind(bind),
            LogicalOperator::Construct(construct) => self.plan_construct(construct),
            LogicalOperator::MultiWayJoin(mwj) => self.plan_multi_way_join(mwj),
            LogicalOperator::Empty => {
                let op: Box<dyn Operator> = Box::new(SingleRowOperator::new());
                Ok((op, vec![], vec![]))
            }
            // A stub must not report success: LOAD fails until it reads
            // files, also with SILENT.
            LogicalOperator::LoadGraph(_) => Err(Error::from(QueryError::new(
                QueryErrorKind::Unsupported,
                "SPARQL LOAD is not supported yet: load a Turtle or N-Triples file or text \
                 with load_rdf and load_rdf_text (Python), or load_rdf, load_rdf_into and \
                 load_rdf_text (Rust), or insert the data with INSERT DATA",
            ))),
            _ => Err(Error::Internal(format!(
                "Unsupported RDF operator: {:?}",
                std::mem::discriminant(op)
            ))),
        };
        self.maybe_profile(result, op)
    }

    /// The one row of the values of the row this plan is for.
    fn plan_row_values(
        &self,
        values: &crate::query::plan::RowValuesOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        let row = self.row.as_deref().ok_or_else(|| {
            Error::Internal("the values of a row in a plan for no row".to_string())
        })?;
        let held = values
            .names
            .iter()
            .map(|name| {
                row.value(name).cloned().ok_or_else(|| {
                    Error::Internal(format!("the row being tested does not bind ?{name}"))
                })
            })
            .collect::<Result<Vec<Value>>>()?;
        let types = vec![LogicalType::RdfTermId; held.len()];
        let operator = Box::new(exists_per_row::RowValuesOperator::new(held));
        Ok((operator, values.names.clone(), types))
    }

    fn plan_property_path(
        &self,
        path: &crate::query::plan::PropertyPathOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        let operator = self.path_source().operator(self.path_config(path)?);
        let columns = operator.columns();
        let types = vec![LogicalType::RdfTermId; columns.len()];
        let operator = self.intern(Box::new(operator), (0..columns.len()).collect());
        Ok((operator, columns, types))
    }

    /// What the query's path searches read with (see
    /// [`path_per_row::PathSource`]).
    fn path_source(&self) -> path_per_row::PathSource {
        path_per_row::PathSource {
            store: Arc::clone(&self.store),
            budget: self.path_search_budget,
            deadline: self.deadline,
            overlay: self.writer.as_ref().map(RdfWriter::path_overlay),
        }
    }

    /// The search of `path`: its ends, graph and step.
    fn path_config(
        &self,
        path: &crate::query::plan::PropertyPathOp,
    ) -> Result<grafeo_core::execution::operators::RdfPathConfig> {
        use grafeo_core::execution::operators::{RdfPathConfig, RdfPathGraph};
        let graph = match path.graph.as_ref() {
            Some(TripleComponent::Variable(_)) => RdfPathGraph::NamedGraphs(
                path.dataset
                    .as_ref()
                    .and_then(DatasetRestriction::listed_named_graphs)
                    .map(<[String]>::to_vec),
            ),
            Some(component) => {
                let iri = match component {
                    TripleComponent::Iri(iri) => iri.clone(),
                    TripleComponent::Literal(Value::String(iri)) => iri.to_string(),
                    _ => {
                        return Err(Error::from(QueryError::new(
                            QueryErrorKind::Semantic,
                            "A path graph must be an IRI or variable",
                        )));
                    }
                };
                match path
                    .dataset
                    .as_ref()
                    .and_then(DatasetRestriction::listed_named_graphs)
                {
                    Some(listed) if listed.contains(&iri) => {
                        RdfPathGraph::NamedGraphs(Some(vec![iri]))
                    }
                    Some(_) => RdfPathGraph::NamedGraphs(Some(Vec::new())),
                    None => RdfPathGraph::Named(iri),
                }
            }
            None => path.dataset.as_ref().map_or(RdfPathGraph::Default, |ds| {
                RdfPathGraph::Union(ds.default_graphs.clone())
            }),
        };
        Ok(RdfPathConfig {
            subject: component_to_term(&path.subject),
            object: component_to_term(&path.object),
            subject_var: path.subject.as_variable().map(str::to_string),
            object_var: path.object.as_variable().map(str::to_string),
            graph_var: path
                .graph
                .as_ref()
                .and_then(TripleComponent::as_variable)
                .map(str::to_string),
            min_hops: !path.zero_length,
            repeats: path.repeats,
            path: path.path.clone(),
            graph,
            transaction_id: None,
            chunk_capacity: self.chunk_size,
            zero_length_in_graph: false,
        })
    }

    /// The search per row of a join of a property path with `other`, the
    /// join's other side, when `other` always binds the path's subject or
    /// object (see [`path_per_row`]): the path's config with those ends left
    /// open, their columns in `other_columns`, and the path's own columns.
    /// `None` when `other` binds neither end, or holds a column of the path's
    /// free end or graph variable without always binding it (a join then
    /// checks the pairs).
    fn path_per_row(
        &self,
        path: &crate::query::plan::PropertyPathOp,
        other: &LogicalOperator,
        other_columns: &[String],
    ) -> Result<Option<path_per_row::PathPerRowPlan>> {
        let Some(bound) = crate::query::plan::always_bound(other) else {
            return Ok(None);
        };
        let column = |component: &TripleComponent| {
            component
                .as_variable()
                .filter(|variable| bound.contains(*variable))
                .and_then(|variable| other_columns.iter().position(|c| c == variable))
        };
        let (subject_column, object_column) = (column(&path.subject), column(&path.object));
        if subject_column.is_none() && object_column.is_none() {
            return Ok(None);
        }
        let mut config = self.path_config(path)?;
        // Between two variables the zero-length solutions are the graph's
        // terms: a row's term outside the graph has none.
        config.zero_length_in_graph =
            path.subject.as_variable().is_some() && path.object.as_variable().is_some();
        if subject_column.is_some() {
            config.subject_var = None;
        }
        if object_column.is_some() {
            config.object_var = None;
        }
        let columns = grafeo_core::execution::operators::RdfPathOperator::new(
            Arc::clone(&self.store),
            config.clone(),
        )
        .columns();
        if columns.iter().any(|name| other_columns.contains(name)) {
            return Ok(None);
        }
        Ok(Some(path_per_row::PathPerRowPlan {
            config,
            subject_column,
            object_column,
            columns,
        }))
    }

    /// Plans a triple scan operator.
    ///
    /// Creates a lazy scanning operator that reads triples in chunks
    /// for cache-efficient, vectorized processing.
    fn plan_triple_scan(
        &self,
        scan: &TripleScanOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        let (scan_op, columns) = self.triple_scan(scan, [false; 3]);
        let types = vec![LogicalType::RdfTermId; columns.len()];
        Ok((Box::new(scan_op), columns, types))
    }

    /// The scan of `scan`'s pattern and its columns, its variables in
    /// subject, predicate, object and graph order, without the positions
    /// `left_out` marks (which a row fills in, see [`triple_lookup`]).
    fn triple_scan(
        &self,
        scan: &TripleScanOp,
        left_out: [bool; 3],
    ) -> (RdfTripleScanOperator, Vec<String>) {
        // Build the triple pattern for querying the store
        let pattern = self.build_triple_pattern(scan);

        // Determine which columns are variables (and thus in output). A variable repeated in
        // the pattern is one column, and the scan keeps the rows whose positions hold the same
        // term there.
        let mut columns: Vec<String> = Vec::new();
        let mut output_mask = [false, false, false, false]; // s, p, o, g
        let mut same_as: [Option<usize>; 4] = [None; 4];
        let mut first_at: Vec<(&str, usize)> = Vec::with_capacity(4);
        let graph = scan.graph.as_ref();
        for (position, component) in [&scan.subject, &scan.predicate, &scan.object]
            .into_iter()
            .chain(graph)
            .enumerate()
        {
            if position < 3 && left_out[position] {
                continue;
            }
            let TripleComponent::Variable(name) = component else {
                continue;
            };
            match first_at.iter().find(|(seen, _)| *seen == name) {
                Some(&(_, first)) => same_as[position] = Some(first),
                None => {
                    first_at.push((name, position));
                    columns.push(name.clone());
                    output_mask[position] = true;
                }
            }
        }

        // Resolve graph context
        let (graph_iri, scan_all_graphs) = match &scan.graph {
            Some(TripleComponent::Iri(iri)) => (Some(iri.clone()), false),
            Some(TripleComponent::Literal(Value::String(iri))) => (Some(iri.to_string()), false),
            Some(TripleComponent::Variable(_)) => (None, true),
            _ => (None, false),
        };

        // Create the lazy scanning operator
        let scan_op = RdfTripleScanOperator::new(
            Arc::clone(&self.store),
            pattern,
            output_mask,
            self.chunk_size,
            GraphContext {
                graph: graph_iri,
                scan_all_graphs,
                dataset: scan.dataset.clone(),
            },
            Arc::clone(&self.terms),
        )
        .with_same_terms(same_as)
        .with_writer(self.writer.clone())
        .with_read_counter(self.read_counter());
        (scan_op, columns)
    }

    /// The lookup join of `scan` with `other`, the join's other side, when
    /// `other` always binds a variable of the pattern (see
    /// [`triple_lookup`]): the full scan, the scan a row fills in, the plan
    /// and the columns the join adds. `None` when `other` binds none, holds
    /// a free variable of the pattern without always binding it (a join
    /// then checks the pairs), the pattern repeats a variable, or has a
    /// graph variable or an input.
    fn triple_lookup(
        &self,
        scan: &TripleScanOp,
        other: &LogicalOperator,
        other_columns: &[String],
        scan_estimate: Option<f64>,
        other_estimate: Option<f64>,
    ) -> Option<triple_lookup::TripleLookupParts> {
        let limit = lookup_limit_against(other, scan_estimate, other_estimate)?;
        let full_columns: Vec<String> = pattern_variables(scan)
            .into_iter()
            .map(|(_, name)| name.to_string())
            .collect();
        let (lookups, added_names) =
            self.pattern_lookups(&[scan], &full_columns, other, other_columns)?;
        let (full, full_columns) = self.triple_scan(scan, [false; 3]);
        Some(triple_lookup::TripleLookupParts {
            full: Box::new(full),
            full_types: vec![LogicalType::RdfTermId; full_columns.len()],
            full_columns,
            lookups,
            added_names,
            limit,
        })
    }

    /// The lookup join of a UNION of triple patterns with `other`, the
    /// join's other side, when each branch is a pattern `other` would look up
    /// (see [`Self::pattern_lookups`]; an alternative path `?s (p|^p) ?o` is
    /// such a UNION): the patterns, the columns the join adds and the limit
    /// of rows. `union_columns` are the UNION's own. `None` when a branch is
    /// not a single pattern, or holds no variable `other` binds.
    fn union_lookup(
        &self,
        union: &crate::query::plan::UnionOp,
        union_columns: &[String],
        other: &LogicalOperator,
        other_columns: &[String],
        other_estimate: Option<f64>,
    ) -> Option<UnionLookupPlan> {
        let scans: Vec<&TripleScanOp> = union
            .inputs
            .iter()
            .map(|branch| match union_branch_pattern(branch) {
                Some(LogicalOperator::TripleScan(scan)) => Some(scan),
                _ => None,
            })
            .collect::<Option<_>>()?;
        // Each row looks up in every pattern: the limit is that of the
        // patterns' mean estimate.
        #[expect(
            clippy::cast_precision_loss,
            reason = "a count of UNION branches, which an f64 holds exactly"
        )]
        let branches = scans.len() as f64;
        let per_pattern = self
            .union_patterns_estimate(union)
            .map(|triples| triples / branches);
        let limit = lookup_limit_against(other, per_pattern, other_estimate)?;
        let (lookups, added_names) =
            self.pattern_lookups(&scans, union_columns, other, other_columns)?;
        Some((lookups, added_names, limit))
    }

    /// The lookup of a join of a UNION of patterns with its other side (see
    /// [`Self::union_lookup`]), and whether the UNION is on the right.
    fn union_lookup_of_join(
        &self,
        join: &crate::query::plan::JoinOp,
        left_columns: &[String],
        right_columns: &[String],
        left_estimate: Option<f64>,
        right_estimate: Option<f64>,
    ) -> Option<(UnionLookupPlan, bool)> {
        let on_right = match join.right.as_ref() {
            LogicalOperator::Union(union) => self.union_lookup(
                union,
                right_columns,
                &join.left,
                left_columns,
                left_estimate,
            ),
            _ => None,
        };
        if let Some(plan) = on_right {
            return Some((plan, true));
        }
        // A VALUES block of several rows is a UNION too.
        match join.left.as_ref() {
            LogicalOperator::Union(union) => self
                .union_lookup(
                    union,
                    left_columns,
                    &join.right,
                    right_columns,
                    right_estimate,
                )
                .map(|plan| (plan, false)),
            _ => None,
        }
    }

    /// How many triples the patterns of a UNION of patterns are estimated to
    /// match together (see [`union_branch_pattern`]), `None` for another
    /// UNION.
    fn union_patterns_estimate(&self, union: &crate::query::plan::UnionOp) -> Option<f64> {
        union
            .inputs
            .iter()
            .map(|branch| {
                union_branch_pattern(branch)
                    .and_then(|pattern| estimate_operator_cardinality(pattern, &self.store))
            })
            .sum()
    }

    /// The lookup of each of `scans` per row of `other`, whose columns are
    /// `other_columns`, and the names of the columns the join adds: those of
    /// `full_columns` (the columns of the scan, or of the UNION of the scans)
    /// `other` lacks. `None` when `other` binds no variable of a pattern,
    /// holds a column of `full_columns` a pattern lacks or that it does not
    /// always bind (a join then checks the pairs), or a pattern repeats a
    /// variable, has a graph variable or an input.
    fn pattern_lookups(
        &self,
        scans: &[&TripleScanOp],
        full_columns: &[String],
        other: &LogicalOperator,
        other_columns: &[String],
    ) -> Option<(Vec<triple_lookup::PatternLookup>, Vec<String>)> {
        let bound_names = crate::query::plan::always_bound(other)?;
        let (shared, added_names): (Vec<&String>, Vec<&String>) = full_columns
            .iter()
            .partition(|column| other_columns.contains(column));
        let added_names: Vec<String> = added_names.into_iter().cloned().collect();
        let mut lookups = Vec::with_capacity(scans.len());
        for scan in scans {
            if scan.input.is_some() || matches!(scan.graph, Some(TripleComponent::Variable(_))) {
                return None;
            }
            let variables = pattern_variables(scan);
            let mut seen = std::collections::HashSet::new();
            if !variables.iter().all(|(_, name)| seen.insert(*name)) {
                return None;
            }
            let mut left_out = [false; 3];
            let mut bound = Vec::new();
            let mut targets = Vec::new();
            for &(position, name) in &variables {
                match other_columns.iter().position(|column| column == name) {
                    Some(column) if bound_names.contains(name) => {
                        left_out[position] = true;
                        bound.push((position, column));
                    }
                    Some(_) => return None,
                    None => targets.push(added_names.iter().position(|added| added == name)?),
                }
            }
            // Every column the sides share is a key of each pattern's lookup.
            if bound.is_empty() || bound.len() != shared.len() {
                return None;
            }
            let (scan, _) = self.triple_scan(scan, left_out);
            lookups.push(triple_lookup::PatternLookup {
                scan,
                bound,
                targets,
            });
        }
        Some((lookups, added_names))
    }

    /// The lookup join of `parts` over `input` (see [`triple_lookup`]), its
    /// columns those of `input`, then the added ones. Out of line: the
    /// operator it builds would enlarge `plan_join`'s frame at every join of
    /// a chain.
    #[inline(never)]
    fn lookup_join(
        &self,
        parts: triple_lookup::TripleLookupParts,
        input: Box<dyn Operator>,
        columns: Vec<String>,
        types: Vec<LogicalType>,
        cardinalities: Option<(f64, f64)>,
        left: bool,
    ) -> (Box<dyn Operator>, Vec<String>, Vec<LogicalType>) {
        let mut out_columns = columns.clone();
        let mut out_types = types.clone();
        out_types.extend(std::iter::repeat_n(
            LogicalType::RdfTermId,
            parts.added_names.len(),
        ));
        out_columns.extend(parts.added_names.iter().cloned());
        let operator = triple_lookup::TripleLookupOperator::new(
            input,
            parts,
            triple_lookup::HashJoinInputs {
                columns,
                types,
                cardinalities,
                left,
            },
            Arc::clone(&self.terms),
            self.chunk_size,
        );
        (Box::new(operator), out_columns, out_types)
    }

    /// The lookup join of the rows of `side` in the patterns of a UNION (see
    /// [`Self::union_lookup`]), `union` the planned UNION, which the join
    /// hashes past the limit. The UNION's PROFILE lines, `union_entries`,
    /// join the lookup's.
    #[inline(never)]
    fn union_lookup_join(
        &self,
        (lookups, added_names, limit): UnionLookupPlan,
        side: PlannedOperator,
        union: PlannedOperator,
        union_entries: std::ops::Range<usize>,
        cardinalities: Option<(f64, f64)>,
        left: bool,
    ) -> PlannedOperator {
        if let Some(entries) = self.profile_entries.borrow_mut().get_mut(union_entries) {
            for entry in entries {
                entry.absorbed = true;
            }
        }
        let (full, full_columns, full_types) = union;
        let parts = triple_lookup::TripleLookupParts {
            full,
            full_columns,
            full_types,
            lookups,
            added_names,
            limit,
        };
        let (input, columns, types) = side;
        self.lookup_join(parts, input, columns, types, cardinalities, left)
    }

    /// Builds a TriplePattern from a TripleScanOp.
    fn build_triple_pattern(&self, scan: &TripleScanOp) -> TriplePattern {
        TriplePattern {
            subject: component_to_term(&scan.subject),
            predicate: component_to_term(&scan.predicate),
            object: component_to_term(&scan.object),
        }
    }

    /// Plans a RETURN clause.
    fn plan_return(
        &self,
        ret: &crate::query::plan::ReturnOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        let (input_op, _input_columns, input_types) = self.plan_operator(&ret.input)?;

        // Extract output column names
        let columns: Vec<String> = ret
            .items
            .iter()
            .map(|item| output_column_name(item.alias.as_deref(), &item.expression))
            .collect();

        Ok((input_op, columns, input_types))
    }

    /// Plans a filter operator.
    ///
    /// A whole predicate of `EXISTS` or `NOT EXISTS` that a run without the row answers becomes a
    /// semi-join or anti-join; any other `EXISTS` a column the predicate reads.
    fn plan_filter(
        &self,
        filter: &FilterOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        let planned = self.plan_operator(&filter.input)?;
        if let Some((subquery, is_negated)) = self.extract_exists_pattern(&filter.predicate) {
            // The join compares what both bind, so a key either may leave unbound, which matches
            // any value, is left to the mark.
            if let Some(read) = self.run_once_read(subquery, &planned.1) {
                let binds_the_keys = |plan: &LogicalOperator| {
                    crate::query::plan::always_bound(plan)
                        .is_none_or(|bound| read.shared.iter().all(|name| bound.contains(name)))
                };
                if read.row_conjuncts.is_empty()
                    && binds_the_keys(&filter.input)
                    && binds_the_keys(subquery)
                {
                    return self.plan_exists_as_join(planned, subquery, is_negated);
                }
            }
        }
        let width = planned.1.len();
        let (predicate, (input_op, columns, types)) = self.marked(&filter.predicate, planned)?;

        // Build variable to column index mapping
        let variable_columns: HashMap<String, usize> = columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        // Convert logical expression to filter expression
        let filter_expr = convert_filter_expression(&predicate)?;

        // Above an aggregation the row is gone, as for a projection or a sort.
        let predicate = RdfExpressionPredicate::new(
            filter_expr,
            variable_columns,
            Arc::clone(&self.terms),
            self.expression_context_over(&filter.input),
        );

        let operator: Box<dyn Operator> =
            Box::new(FilterOperator::new(input_op, Box::new(predicate)));
        Ok(self.keeping_columns((operator, columns, types), (0..width).collect()))
    }

    /// Extracts an EXISTS or NOT EXISTS pattern from a filter predicate.
    /// Returns the subquery operator and whether it's negated (NOT EXISTS).
    fn extract_exists_pattern<'a>(
        &self,
        predicate: &'a LogicalExpression,
    ) -> Option<(&'a LogicalOperator, bool)> {
        use crate::query::plan::UnaryOp;

        match predicate {
            // EXISTS { pattern }
            LogicalExpression::ExistsSubquery(subquery) => Some((subquery.as_ref(), false)),
            // NOT EXISTS { pattern }
            LogicalExpression::Unary {
                op: UnaryOp::Not,
                operand,
            } => {
                if let LogicalExpression::ExistsSubquery(subquery) = operand.as_ref() {
                    Some((subquery.as_ref(), true))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Plans an EXISTS/NOT EXISTS pattern as a semi-join or anti-join.
    fn plan_exists_as_join(
        &self,
        (left_op, left_columns, left_types): PlannedOperator,
        subquery: &LogicalOperator,
        is_negated: bool,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        use crate::query::planner::common;

        let (right_op, right_columns, _right_types) = self.plan_exists_subquery(subquery)?;

        // Use Anti for NOT EXISTS, Semi for EXISTS
        let (result_op, result_columns) = if is_negated {
            common::build_not_exists_join(
                left_op,
                right_op,
                left_columns,
                &right_columns,
                left_types.clone(),
            )
        } else {
            common::build_semi_join(
                left_op,
                right_op,
                left_columns,
                &right_columns,
                left_types.clone(),
            )
        };

        Ok((result_op, result_columns, left_types))
    }

    /// `expression` with each `EXISTS` in it reading a column, and `planned` with those columns:
    /// one evaluation for every position an expression holds. The operator that reads a column
    /// leaves it out of what it returns, so a `SELECT *` or a parent never sees one, and its name
    /// is none a variable can have.
    fn marked(
        &self,
        expression: &LogicalExpression,
        mut planned: PlannedOperator,
    ) -> Result<(LogicalExpression, PlannedOperator)> {
        let mut tests = Vec::new();
        let expression = exists_read::take_exists(expression.clone(), &mut |subquery| {
            let number = self.exists_marks.get();
            self.exists_marks.set(number + 1);
            let mark = format!("{}{number}", exists_per_row::MARK_PREFIX);
            tests.push((mark.clone(), subquery));
            LogicalExpression::Variable(mark)
        });
        for (mark, subquery) in tests {
            planned = self.plan_exists_mark(planned, mark, &subquery)?;
        }
        Ok((expression, planned))
    }

    /// `planned` with the columns `keep` alone, in that order: nothing to do when they are all
    /// of them, as they are unless an `EXISTS` was marked.
    #[inline(never)]
    fn keeping_columns(
        &self,
        (operator, columns, types): PlannedOperator,
        keep: Vec<usize>,
    ) -> PlannedOperator {
        if keep.len() == columns.len() {
            return (operator, columns, types);
        }
        let kept_columns = keep.iter().map(|&at| columns[at].clone()).collect();
        let kept_types: Vec<LogicalType> = keep.iter().map(|&at| types[at].clone()).collect();
        let operator = Box::new(RdfProjectOperator::new(
            operator,
            keep.into_iter().map(RdfProjectExpr::Column).collect(),
            kept_types.clone(),
            Arc::clone(&self.terms),
            self.expression_context(),
        ));
        (operator, kept_columns, kept_types)
    }

    /// As [`marked`](Self::marked), of an expression that may be absent.
    fn marked_if_any(
        &self,
        expression: Option<&LogicalExpression>,
        planned: PlannedOperator,
    ) -> Result<(Option<LogicalExpression>, PlannedOperator)> {
        match expression {
            Some(expression) => {
                let (expression, planned) = self.marked(expression, planned)?;
                Ok((Some(expression), planned))
            }
            None => Ok((None, planned)),
        }
    }

    /// `planned` with a column `mark`: whether `subquery` holds for the row.
    #[inline(never)]
    fn plan_exists_mark(
        &self,
        (child, mut columns, mut types): PlannedOperator,
        mark: String,
        subquery: &LogicalOperator,
    ) -> Result<PlannedOperator> {
        use exists_pair::{ExistsMarkOperator, ExistsTest, MarkTest, RowCheck};

        let Some(read) = self.run_once_read(subquery, &columns) else {
            self.note_exists_route("ExistsPerRow");
            let test = MarkTest::EachRow(Box::new(ExistsPerRow::new(
                self.row_plan_parts(),
                subquery.clone(),
                &columns,
                &[],
                self.row.clone(),
            )));
            columns.push(mark);
            types.push(LogicalType::RdfTermId);
            let operator = ExistsMarkOperator::new(child, test, Arc::clone(&self.terms));
            return Ok((Box::new(operator), columns, types));
        };
        let (operator, subquery_columns, _) = self.plan_exists_subquery(&read.subquery)?;
        let shared = || {
            subquery_columns
                .iter()
                .enumerate()
                .filter_map(|(at, name)| {
                    let in_row = columns.iter().position(|column| column == name)?;
                    Some((in_row, at))
                })
        };
        let test = match LogicalExpression::conjunction(read.row_conjuncts) {
            Some(conjuncts) => {
                self.note_exists_route("ExistsRowCheck");
                MarkTest::PerRow(RowCheck::new(
                    operator,
                    shared().collect(),
                    self.merged_row_condition(&conjuncts, &columns, &subquery_columns)?,
                ))
            }
            None => {
                self.note_exists_route("ExistsMark");
                let (keys, at) = shared()
                    .map(|(in_row, at)| ((Some(in_row), None), at))
                    .unzip();
                MarkTest::Lookup(ExistsTest::new(operator, false, keys, at))
            }
        };
        columns.push(mark);
        types.push(LogicalType::RdfTermId);
        let operator = ExistsMarkOperator::new(child, test, Arc::clone(&self.terms));
        Ok((Box::new(operator), columns, types))
    }

    /// The condition of an `OPTIONAL`'s left join, over the merged solution
    /// of a pair (SPARQL 1.1 sec 18.5): a variable both sides have reads the
    /// right side's value where the left leaves it unbound, and each
    /// conjunct that is an `EXISTS` or `NOT EXISTS` is decided per pair (see
    /// [`exists_pair`]).
    fn left_join_condition(
        &self,
        condition: &LogicalExpression,
        left_columns: &[String],
        right_columns: &[String],
    ) -> Result<Box<dyn grafeo_core::execution::operators::JoinCondition>> {
        let mut conjuncts = Vec::new();
        conjuncts_of(condition, &mut conjuncts);
        let mut tests = Vec::new();
        let mut rest = Vec::new();
        // An `EXISTS` deeper in a conjunct is the same test, its answer a column of the pair that
        // the conjunct reads: a mark.
        let mut marks = Vec::new();
        let mut marked_right = right_columns.to_vec();
        for conjunct in conjuncts {
            match self.extract_exists_pattern(conjunct) {
                Some((subquery, negated)) => {
                    tests.push(self.pair_test(subquery, negated, left_columns, right_columns)?);
                }
                None if crate::query::plan::has_subquery(conjunct) => {
                    let mut found = Vec::new();
                    let expression = exists_read::take_exists(conjunct.clone(), &mut |subquery| {
                        let number = self.exists_marks.get();
                        self.exists_marks.set(number + 1);
                        let mark = format!("{}{number}", exists_per_row::MARK_PREFIX);
                        found.push((mark.clone(), subquery));
                        LogicalExpression::Variable(mark)
                    });
                    for (mark, subquery) in found {
                        marks.push(self.pair_test(
                            &subquery,
                            false,
                            left_columns,
                            right_columns,
                        )?);
                        marked_right.push(mark);
                    }
                    rest.push(expression);
                }
                None => rest.push(conjunct.clone()),
            }
        }
        let rest = LogicalExpression::conjunction(rest)
            .map(|rest| self.merged_row_condition(&rest, left_columns, &marked_right))
            .transpose()?;
        if tests.is_empty() && marks.is_empty() {
            return rest.ok_or_else(|| {
                Error::Internal("a left join condition with no conjunct".to_string())
            });
        }
        Ok(Box::new(exists_pair::ExistsPairCondition::new(
            tests,
            marks,
            rest,
            Arc::clone(&self.terms),
        )))
    }

    /// The test of `subquery` for a joined pair: the row is the left columns, then the right ones.
    #[inline(never)]
    fn pair_test(
        &self,
        subquery: &LogicalOperator,
        negated: bool,
        left_columns: &[String],
        right_columns: &[String],
    ) -> Result<exists_pair::PairTest> {
        let mut row = left_columns.to_vec();
        row.extend_from_slice(right_columns);
        let read = self
            .run_once_read(subquery, &row)
            .filter(|read| read.row_conjuncts.is_empty());
        let Some(read) = read else {
            self.note_exists_route("ExistsPerRow");
            return Ok(exists_pair::PairTest::each_pair(
                ExistsPerRow::new(
                    self.row_plan_parts(),
                    subquery.clone(),
                    left_columns,
                    right_columns,
                    self.row.clone(),
                ),
                negated,
            ));
        };
        self.note_exists_route("ExistsMark");
        let (op, columns, _) = self.plan_exists_subquery(&read.subquery)?;
        let (keys, subquery_columns) = columns
            .iter()
            .enumerate()
            .filter_map(|(at, name)| {
                let left = left_columns.iter().position(|c| c == name);
                let right = right_columns.iter().position(|c| c == name);
                (left.is_some() || right.is_some()).then_some(((left, right), at))
            })
            .unzip();
        Ok(exists_pair::PairTest::Lookup(exists_pair::ExistsTest::new(
            op,
            negated,
            keys,
            subquery_columns,
        )))
    }

    /// `condition` over a row followed by another: a variable both have reads the second's value
    /// where the first leaves it unbound (sec 18.5, LeftJoin).
    fn merged_row_condition(
        &self,
        condition: &LogicalExpression,
        left_columns: &[String],
        right_columns: &[String],
    ) -> Result<Box<dyn grafeo_core::execution::operators::JoinCondition>> {
        let mut variable_columns: HashMap<String, usize> = HashMap::new();
        for (i, name) in left_columns.iter().chain(right_columns).enumerate() {
            variable_columns.entry(name.clone()).or_insert(i);
        }
        let fallback_columns: HashMap<String, usize> = right_columns
            .iter()
            .enumerate()
            .filter(|(_, name)| left_columns.contains(name))
            .map(|(i, name)| (name.clone(), left_columns.len() + i))
            .collect();
        let predicate = RdfExpressionPredicate::new(
            convert_filter_expression(condition)?,
            variable_columns,
            Arc::clone(&self.terms),
            self.expression_context(),
        )
        .with_fallback_columns(fallback_columns);
        Ok(Box::new(
            grafeo_core::execution::operators::JoinedRowCondition::new(Box::new(predicate)),
        ))
    }

    /// Plans a DISTINCT operator.
    fn plan_distinct(
        &self,
        distinct: &DistinctOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        use crate::query::planner::common;
        let (input_op, columns, types) = self.plan_operator(&distinct.input)?;
        let (op, cols) = common::build_distinct(
            input_op,
            columns,
            distinct.columns.as_deref(),
            grafeo_common::types::TemporalComparison::TypesApart,
        );
        Ok((op, cols, types))
    }

    /// Plans a LIMIT operator.
    fn plan_limit(
        &self,
        limit: &LimitOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        use crate::query::planner::common;
        let (input_op, columns, types) = self.plan_operator(&limit.input)?;
        let (op, cols) = common::build_limit(input_op, columns, limit.count.value());
        Ok((op, cols, types))
    }

    /// Plans a SKIP operator.
    fn plan_skip(
        &self,
        skip: &SkipOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        use crate::query::planner::common;
        let (input_op, columns, types) = self.plan_operator(&skip.input)?;
        let (op, cols) = common::build_skip(input_op, columns, skip.count.value());
        Ok((op, cols, types))
    }

    /// Plans a SORT operator: every key is pre-projected as the RDF term it
    /// is (a variable decoded, an expression evaluated), so the sort orders
    /// terms in SPARQL order (section 15.1), unbound first ascending.
    fn plan_sort(
        &self,
        sort: &SortOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        use crate::query::plan::SortOrder;

        let mut planned = self.plan_operator(&sort.input)?;
        let width = planned.1.len();
        let mut key_expressions = Vec::with_capacity(sort.keys.len());
        for key in &sort.keys {
            let (expression, next) = self.marked(&key.expression, planned)?;
            key_expressions.push(expression);
            planned = next;
        }
        let (input_op, mut columns, mut types) = planned;
        let variable_columns = column_map(&columns);
        let context = self.expression_context_over(&sort.input);

        let mut projections: Vec<RdfProjectExpr> =
            (0..columns.len()).map(RdfProjectExpr::Column).collect();
        let mut output_types = types.clone();
        let mut keys = Vec::with_capacity(sort.keys.len());
        for (key, expression) in sort.keys.iter().zip(&key_expressions) {
            // A variable's key is its own id column; an expression's value is
            // projected as a term after the returned columns. A variable the
            // pattern does not bind is unbound in every solution: a key that
            // orders nothing.
            let column = match expression {
                LogicalExpression::Variable(name) => match variable_columns.get(name) {
                    // The row's value where its own column is unbound, as a
                    // projection reads it.
                    Some(_) if context.row_value(name).is_some() => {
                        projections.push(RdfProjectExpr::Evaluated {
                            expr: convert_filter_expression(expression)?,
                            variable_columns: variable_columns.clone(),
                            output: EvaluatedOutput::Term,
                        });
                        output_types.push(LogicalType::Any);
                        projections.len() - 1
                    }
                    Some(&column) => column,
                    None => {
                        projections.push(RdfProjectExpr::Constant(
                            context.row_value(name).cloned().unwrap_or(Value::Null),
                        ));
                        output_types.push(LogicalType::Any);
                        projections.len() - 1
                    }
                },
                other => {
                    projections.push(RdfProjectExpr::Evaluated {
                        expr: convert_filter_expression(other)?,
                        variable_columns: variable_columns.clone(),
                        output: EvaluatedOutput::Term,
                    });
                    output_types.push(LogicalType::Any);
                    projections.len() - 1
                }
            };
            keys.push(order::OrderKey {
                column,
                descending: key.order == SortOrder::Descending,
                // Unbound is the least value (15.1): first ascending.
                nulls_first: matches!(
                    (key.nulls, key.order),
                    (Some(crate::query::plan::NullsOrdering::First), _)
                        | (None, SortOrder::Ascending)
                ),
            });
        }
        // Variables only: the input as it is.
        let projected: Box<dyn Operator> = if projections.len() == columns.len() {
            input_op
        } else {
            Box::new(RdfProjectOperator::new(
                input_op,
                projections,
                output_types,
                Arc::clone(&self.terms),
                context,
            ))
        };
        // The key columns come last and are not returned, the columns of an `EXISTS` key with them.
        columns.truncate(width);
        types.truncate(width);
        let sort_op =
            order::RdfSortOperator::new(projected, keys, types.clone(), Arc::clone(&self.terms));
        Ok((Box::new(sort_op), columns, types))
    }

    /// A pattern in a named graph: its plan, or with no such graph in the dataset now, nothing
    /// under the same columns (SPARQL 1.1 sec 18.6).
    fn plan_in_named_graph(
        &self,
        graph: &crate::query::plan::InNamedGraphOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        let first_entry = self.profile_entries.borrow().len();
        let (operator, mut columns, mut types) = self.plan_operator(&graph.input)?;
        let names = named_graphs::named_graph_names(
            &self.store,
            self.writer.as_ref(),
            graph.dataset.as_ref(),
        );
        let found = match &graph.graph {
            crate::query::plan::NamedGraph::Iri(iri) => names.binary_search(iri).is_ok(),
            crate::query::plan::NamedGraph::Variable(variable) => {
                // The branches below run in its place: their rows and time count in this line.
                let entries = first_entry..self.profile_entries.borrow().len();
                if let Some(entries) = self.profile_entries.borrow_mut().get_mut(entries) {
                    for entry in entries {
                        entry.absorbed = true;
                    }
                }
                if !names.is_empty() {
                    return self.plan_in_each_graph(graph, variable, &names);
                }
                if !columns.contains(variable) {
                    columns.push(variable.clone());
                    types.push(LogicalType::RdfTermId);
                }
                false
            }
        };
        if found {
            return Ok((operator, columns, types));
        }
        let empty: Box<dyn Operator> = Box::new(
            grafeo_core::execution::operators::EmptyOperator::new(types.clone()),
        );
        Ok((empty, columns, types))
    }

    /// `GRAPH ?g` over a pattern that must run once per graph: a branch per graph of `names`, each
    /// reading its graph alone and joined with that graph's row. The join binds `?g` where the
    /// pattern does not, and where the pattern lists it, fills an unbound and drops another value,
    /// as SPARQL's join of the pattern with the graph's binding does (18.6).
    #[inline(never)]
    fn plan_in_each_graph(
        &self,
        graph: &crate::query::plan::InNamedGraphOp,
        variable: &str,
        names: &[String],
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        use crate::query::plan::{JoinOp, JoinType, NamedGraphsOp, UnionOp};
        let inputs = names
            .iter()
            .map(|name| {
                let mut graphs = NamedGraphsOp {
                    variable: variable.to_string(),
                    dataset: graph.dataset.clone(),
                };
                named_graphs::read_only_graph(&mut graphs.dataset, name);
                LogicalOperator::Join(JoinOp {
                    left: Box::new(LogicalOperator::NamedGraphs(graphs)),
                    right: Box::new(named_graphs::in_one_graph(
                        (*graph.input).clone(),
                        variable,
                        graph.read_as.as_ref(),
                        name,
                    )),
                    join_type: JoinType::Inner,
                    conditions: Vec::new(),
                })
            })
            .collect();
        let profiling = self.profiling.replace(false);
        let planned = self.plan_union(&UnionOp { inputs });
        self.profiling.set(profiling);
        planned
    }

    /// Plans a PROJECT operator: a variable's ids are copied, a constant and
    /// an expression's value are interned.
    fn plan_project(
        &self,
        project: &crate::query::plan::ProjectOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        let mut planned = self.plan_operator(&project.input)?;
        let mut expressions = Vec::with_capacity(project.projections.len());
        for proj in &project.projections {
            let (expression, next) = self.marked(&proj.expression, planned)?;
            expressions.push(expression);
            planned = next;
        }
        let (input_op, input_columns, input_types) = planned;
        let variable_columns = column_map(&input_columns);
        let context = self.expression_context_over(&project.input);

        let mut projections = Vec::new();
        let mut output_columns = Vec::new();
        let mut output_types = Vec::new();

        for (proj, expression) in project.projections.iter().zip(&expressions) {
            match expression {
                LogicalExpression::Variable(name) => {
                    match variable_columns.get(name) {
                        // The row's value where its own column is unbound.
                        Some(_) if context.row_value(name).is_some() => {
                            projections.push(RdfProjectExpr::Expression {
                                expr: convert_filter_expression(expression)?,
                                variable_columns: variable_columns.clone(),
                            });
                            output_types.push(LogicalType::RdfTermId);
                        }
                        Some(&col_idx) => {
                            projections.push(RdfProjectExpr::Column(col_idx));
                            output_types.push(input_types[col_idx].clone());
                        }
                        // A variable the pattern does not bind is unbound in
                        // every solution (SPARQL 1.1, 18.2.5), or is the row's.
                        None => {
                            projections.push(RdfProjectExpr::Constant(
                                context.row_value(name).cloned().unwrap_or(Value::Null),
                            ));
                            output_types.push(LogicalType::RdfTermId);
                        }
                    }
                    output_columns.push(proj.alias.clone().unwrap_or_else(|| name.clone()));
                }
                LogicalExpression::Literal(value) => {
                    let id = match terms::to_term(value) {
                        Some(term) => self.term_id(&term)?,
                        None => Value::Null,
                    };
                    projections.push(RdfProjectExpr::Constant(id));
                    output_columns.push(proj.alias.clone().unwrap_or_else(|| format!("{value}")));
                    output_types.push(LogicalType::RdfTermId);
                }
                expr => {
                    projections.push(RdfProjectExpr::Expression {
                        expr: convert_filter_expression(expr)?,
                        variable_columns: variable_columns.clone(),
                    });
                    output_columns.push(proj.alias.clone().unwrap_or_else(|| format!("{expr:?}")));
                    output_types.push(LogicalType::RdfTermId);
                }
            }
        }

        if projections.is_empty() {
            return Ok((input_op, input_columns, input_types));
        }
        // The input's columns in their order: only the names change.
        let identity = projections.len() == input_columns.len()
            && projections
                .iter()
                .enumerate()
                .all(|(at, projection)| matches!(projection, RdfProjectExpr::Column(column) if *column == at));
        if identity {
            self.folded.set(true);
            return Ok((input_op, output_columns, output_types));
        }

        let operator: Box<dyn Operator> = Box::new(RdfProjectOperator::new(
            input_op,
            projections,
            output_types.clone(),
            Arc::clone(&self.terms),
            context,
        ));
        Ok((operator, output_columns, output_types))
    }

    /// Plans a BIND operator: the expression's value at each row, interned,
    /// in a new column. For example
    /// `BIND (CONCAT(?name, " (age ", STR(?age), ")") AS ?label)`.
    fn plan_bind(
        &self,
        bind: &BindOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        let planned = self.plan_operator(&bind.input)?;
        let width = planned.1.len();
        let (expression, (input_op, input_columns, mut input_types)) =
            self.marked(&bind.expression, planned)?;
        let variable_columns = column_map(&input_columns);
        let filter_expr = convert_filter_expression(&expression)?;

        // A BIND onto a name the input binds replaces its column: SPARQL refuses one in a query
        // (sec 18.2.1), and `GRAPH ?g` binds `?g` so after a pattern that names it.
        let replaced = input_columns
            .iter()
            .position(|column| *column == bind.variable);
        let mut output_columns = input_columns;
        output_columns.push(bind.variable.clone());
        input_types.push(LogicalType::RdfTermId);

        let operator: Box<dyn Operator> = Box::new(RdfBindOperator {
            child: input_op,
            evaluator: RdfExpressionPredicate::new(
                filter_expr,
                variable_columns,
                Arc::clone(&self.terms),
                self.expression_context(),
            ),
            terms: Arc::clone(&self.terms),
        });
        let bound = output_columns.len() - 1;
        let keep = match replaced {
            Some(at) if at < width => (0..width)
                .map(|column| if column == at { bound } else { column })
                .collect(),
            _ => (0..width).chain([bound]).collect(),
        };
        Ok(self.keeping_columns((operator, output_columns, input_types), keep))
    }

    /// Plans a CONSTRUCT operator.
    ///
    /// Evaluates the WHERE clause, then for each row substitutes variable
    /// bindings into the template to produce (subject, predicate, object) rows.
    fn plan_construct(
        &self,
        construct: &ConstructOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        let (input_op, input_columns, _input_types) = self.plan_operator(&construct.input)?;

        let variable_columns = column_map(&input_columns);
        // Each component once: a variable's column, a constant's id, a blank
        // node's label (fresh per solution).
        let slot = |component: &TripleComponent| -> Result<TemplateSlot> {
            Ok(match component {
                TripleComponent::Variable(name) => {
                    let name = name.strip_prefix('?').unwrap_or(name);
                    variable_columns
                        .get(name)
                        .map_or(TemplateSlot::Unbound, |&column| {
                            TemplateSlot::Column(column)
                        })
                }
                TripleComponent::BlankNode(label) => TemplateSlot::Blank(label.clone()),
                other => match component_to_term(other) {
                    Some(term) => TemplateSlot::Constant(
                        self.terms
                            .intern(&term)
                            .map_err(|e| Error::Internal(e.to_string()))?,
                    ),
                    None => TemplateSlot::Unbound,
                },
            })
        };
        let templates = construct
            .templates
            .iter()
            .map(|template| {
                Ok([
                    slot(&template.subject)?,
                    slot(&template.predicate)?,
                    slot(&template.object)?,
                ])
            })
            .collect::<Result<Vec<_>>>()?;

        let operator = Box::new(ConstructOperator {
            input: input_op,
            templates,
            terms: Arc::clone(&self.terms),
            seen: grafeo_common::utils::hash::FxHashSet::default(),
        });

        let columns = vec![
            "subject".to_string(),
            "predicate".to_string(),
            "object".to_string(),
        ];
        // Terms, not ids: the result boundary does not decode them.
        let types = vec![LogicalType::Any; 3];
        Ok((operator, columns, types))
    }

    /// Plans an AGGREGATE operator. Group keys are term ids (a computed key
    /// interned), so groups are by term identity. COUNT reads ids; SUM and
    /// AVG the numbers of their arguments; MIN, MAX and SAMPLE the terms,
    /// which MIN and MAX compare in SPARQL order; GROUP_CONCAT the lexical
    /// forms. The results are interned.
    fn plan_aggregate(
        &self,
        agg: &AggregateOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        if let Some(result) = self.try_count_fast_path(agg) {
            return Ok(result);
        }

        use grafeo_core::execution::operators::AggregateExpr as PhysicalAggregateExpr;

        let mut planned = self.plan_operator(&agg.input)?;
        let mut group_by = Vec::with_capacity(agg.group_by.len());
        for expr in &agg.group_by {
            let (expression, next) = self.marked(expr, planned)?;
            group_by.push(expression);
            planned = next;
        }
        let mut arguments = Vec::with_capacity(agg.aggregates.len());
        for agg_expr in &agg.aggregates {
            let (first, next) = self.marked_if_any(agg_expr.expression.as_ref(), planned)?;
            let (second, next) = self.marked_if_any(agg_expr.expression2.as_ref(), next)?;
            arguments.push((first, second));
            planned = next;
        }
        let (input_op, input_columns, input_types) = planned;
        let input_map = column_map(&input_columns);
        let mut variable_columns = input_map.clone();
        let mut projections: Vec<RdfProjectExpr> = (0..input_columns.len())
            .map(RdfProjectExpr::Column)
            .collect();
        let mut projected_types = input_types;

        let mut group_columns = Vec::with_capacity(agg.group_by.len());
        for expr in &group_by {
            let column = match expr {
                LogicalExpression::Variable(_) => resolve_expression(expr, &variable_columns)?,
                other => {
                    let name = resolved_column_name(other);
                    if let Some(&column) = variable_columns.get(&name) {
                        column
                    } else {
                        projections.push(RdfProjectExpr::Expression {
                            expr: convert_filter_expression(other)?,
                            variable_columns: input_map.clone(),
                        });
                        projected_types.push(LogicalType::RdfTermId);
                        variable_columns.insert(name, projections.len() - 1);
                        projections.len() - 1
                    }
                }
            };
            group_columns.push(column);
        }

        let mut argument = |expr: &LogicalExpression, output: Option<EvaluatedOutput>| {
            if output.is_none()
                && let LogicalExpression::Variable(_) = expr
            {
                return resolve_expression(expr, &variable_columns);
            }
            let expr = convert_filter_expression(expr)?;
            projections.push(match output {
                Some(output) => RdfProjectExpr::Evaluated {
                    expr,
                    variable_columns: input_map.clone(),
                    output,
                },
                None => RdfProjectExpr::Expression {
                    expr,
                    variable_columns: input_map.clone(),
                },
            });
            projected_types.push(if output.is_some() {
                LogicalType::Any
            } else {
                LogicalType::RdfTermId
            });
            Ok(projections.len() - 1)
        };

        let mut physical_aggregates = Vec::with_capacity(agg.aggregates.len());
        for (agg_expr, (first, second)) in agg.aggregates.iter().zip(&arguments) {
            let output = match agg_expr.function {
                LogicalAggregateFunction::Count | LogicalAggregateFunction::CountNonNull => None,
                LogicalAggregateFunction::Sum | LogicalAggregateFunction::Avg => {
                    Some(EvaluatedOutput::Number)
                }
                LogicalAggregateFunction::GroupConcat => Some(EvaluatedOutput::Lexical),
                _ => Some(EvaluatedOutput::Term),
            };
            let column = first
                .as_ref()
                .map(|expr| argument(expr, output))
                .transpose()?;
            let column2 = second
                .as_ref()
                .map(|expr| argument(expr, Some(EvaluatedOutput::Term)))
                .transpose()?;
            physical_aggregates.push(PhysicalAggregateExpr {
                function: convert_aggregate_function(agg_expr.function),
                column,
                column2,
                distinct: agg_expr.distinct,
                alias: agg_expr.alias.clone(),
                percentile: agg_expr.percentile,
                separator: agg_expr.separator.clone(),
                // SPARQL's SUM and AVG: XPath's division for AVG, and an
                // argument that is no number an error (the result unbound).
                rdf_literals: matches!(
                    agg_expr.function,
                    LogicalAggregateFunction::Sum | LogicalAggregateFunction::Avg
                ),
                // SPARQL compares its terms as they are.
                temporal: grafeo_common::types::TemporalComparison::TypesApart,
            });
        }

        let input_op: Box<dyn Operator> = if projections.len() > input_columns.len() {
            Box::new(RdfProjectOperator::new(
                input_op,
                projections,
                projected_types,
                Arc::clone(&self.terms),
                self.expression_context(),
            ))
        } else {
            input_op
        };

        let mut agg_schema = vec![LogicalType::RdfTermId; group_columns.len()];
        let mut output_columns: Vec<String> =
            agg.group_by.iter().map(expression_to_string).collect();
        for agg_expr in &agg.aggregates {
            agg_schema.push(match agg_expr.function {
                LogicalAggregateFunction::Count | LogicalAggregateFunction::CountNonNull => {
                    LogicalType::Int64
                }
                _ => LogicalType::Any,
            });
            output_columns.push(
                agg_expr.alias.clone().unwrap_or_else(|| {
                    crate::query::planner::common::aggregate_column_name(agg_expr)
                }),
            );
        }

        let group_count = group_columns.len();
        let operator: Box<dyn Operator> = if group_columns.is_empty() {
            Box::new(SimpleAggregateOperator::new(
                input_op,
                physical_aggregates,
                agg_schema,
            ))
        } else {
            Box::new(HashAggregateOperator::new(
                input_op,
                group_columns,
                physical_aggregates,
                agg_schema,
            ))
        };
        // The aggregates' values become terms of the dictionary.
        let operator = self.intern(operator, (group_count..output_columns.len()).collect());
        let output_schema = vec![LogicalType::RdfTermId; output_columns.len()];
        let width = output_columns.len();
        let planned = (operator, output_columns, output_schema);

        let Some(having_expr) = &agg.having else {
            return Ok(planned);
        };
        let (having, (operator, output_columns, output_schema)) =
            self.marked(having_expr, planned)?;
        let predicate = RdfExpressionPredicate::new(
            convert_filter_expression(&having)?,
            column_map(&output_columns),
            Arc::clone(&self.terms),
            self.expression_context_with_no_row(),
        );
        let operator: Box<dyn Operator> =
            Box::new(FilterOperator::new(operator, Box::new(predicate)));
        Ok(self.keeping_columns(
            (operator, output_columns, output_schema),
            (0..width).collect(),
        ))
    }

    /// Detects COUNT(*) over a TripleScan and returns the count directly when possible.
    ///
    /// Supported patterns (no GROUP BY, no DISTINCT, no HAVING):
    /// - `COUNT(*) WHERE { ?s ?p ?o }`: fully unbound, returns `store.len()`
    /// - `COUNT(*) WHERE { ?s <pred> ?o }`: predicate-bound, returns per-predicate count
    /// - `COUNT(*) WHERE { GRAPH <g> { ?s ?p ?o } }`: per-graph count
    fn try_count_fast_path(
        &self,
        agg: &AggregateOp,
    ) -> Option<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        // Must be: no GROUP BY, no HAVING, exactly one aggregate
        if !agg.group_by.is_empty() || agg.having.is_some() || agg.aggregates.len() != 1 {
            return None;
        }

        let agg_expr = &agg.aggregates[0];

        // Must be COUNT(*): Count function, no expression (not COUNT(?x)), no DISTINCT
        if agg_expr.function != LogicalAggregateFunction::Count
            || agg_expr.expression.is_some()
            || agg_expr.distinct
        {
            return None;
        }

        // Input must be a simple TripleScan (no chained input)
        let scan = Self::find_simple_triple_scan(&agg.input)?;

        // No dataset restriction (FROM / FROM NAMED)
        if scan.dataset.is_some() {
            return None;
        }

        let count = self.count_for_scan(scan)?;

        let alias = agg_expr
            .alias
            .clone()
            .unwrap_or_else(|| "count(*)".to_string());

        self.make_constant_integer(count, alias).ok()
    }

    /// Resolves the count for a simple triple scan pattern, or returns `None`
    /// if the pattern is too complex for a fast-path.
    fn count_for_scan(&self, scan: &TripleScanOp) -> Option<i64> {
        // The store's counts miss the transaction's own writes.
        if self.has_pending_writes() {
            return None;
        }
        let s_var = scan.subject.as_variable().is_some();
        let p_var = scan.predicate.as_variable().is_some();
        let o_var = scan.object.as_variable().is_some();
        // A variable the pattern repeats keeps fewer triples than the store counts.
        if repeats_a_variable(scan) {
            return None;
        }

        // Per-graph count: GRAPH <iri> { ?s ?p ?o }
        if let Some(graph) = &scan.graph {
            if s_var
                && p_var
                && o_var
                && let TripleComponent::Iri(graph_iri) = graph
            {
                let ng = self.store.graph(graph_iri)?;
                // reason: triple count will not exceed i64::MAX
                #[allow(clippy::cast_possible_wrap)]
                return Some(ng.len() as i64);
            }
            return None;
        }

        // Fully unbound: ?s ?p ?o
        if s_var && p_var && o_var {
            // reason: triple count will not exceed i64::MAX
            #[allow(clippy::cast_possible_wrap)]
            return Some(self.store.len() as i64);
        }

        // Ring Index: use ring.count() for any partially-bound pattern (O(log sigma))
        #[cfg(feature = "ring-index")]
        if let Some(ring) = self.store.ring() {
            let pattern = self.build_triple_pattern(scan);
            // reason: triple count will not exceed i64::MAX
            #[allow(clippy::cast_possible_wrap)]
            return Some(ring.count(&pattern) as i64);
        }

        // Predicate-bound: ?s <pred> ?o
        if s_var && !p_var && o_var {
            if let TripleComponent::Iri(pred_iri) = &scan.predicate {
                // From the index, not the statistics: they may be stale.
                let count = self
                    .store
                    .count_with_predicate(&Term::iri(pred_iri.clone()));
                return i64::try_from(count).ok();
            }
            return None;
        }

        // Not a fast-path pattern
        None
    }

    /// A single-row result of one `xsd:integer`.
    fn make_constant_integer(
        &self,
        value: i64,
        column_name: String,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        let mut chunk = DataChunk::with_capacity(&[LogicalType::RdfTermId], 1);
        chunk
            .column_mut(0)
            .expect("just created")
            .push_value(self.term_id(&Term::Literal(Literal::integer(value)))?);
        chunk.set_count(1);
        let operator = Box::new(ConstantOperator::new(chunk));
        Ok((operator, vec![column_name], vec![LogicalType::RdfTermId]))
    }

    /// Walks through the input operator to find a simple TripleScan (no chained input).
    ///
    /// Sees through Project operators.
    fn find_simple_triple_scan(op: &LogicalOperator) -> Option<&TripleScanOp> {
        match op {
            LogicalOperator::TripleScan(scan) if scan.input.is_none() => Some(scan),
            LogicalOperator::Project(proj) => Self::find_simple_triple_scan(&proj.input),
            _ => None,
        }
    }

    /// Plans a JOIN operator using HashJoin.
    ///
    /// For SPARQL, we join on shared variables (equi-join). When no shared
    /// variables exist, falls back to cross join.
    fn plan_join(
        &self,
        join: &crate::query::plan::JoinOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        use crate::query::planner::common;
        let first_entry = self.profile_entries.borrow().len();
        let (left_op, left_columns, left_types) = self.plan_operator(&join.left)?;
        let right_entry = self.profile_entries.borrow().len();
        let (right_op, right_columns, right_types) = self.plan_operator(&join.right)?;

        // A path with an end the other side binds: one search per row from
        // that end (the path's operator planned above reads nothing until it
        // runs, and is dropped).
        let per_row = match (join.left.as_ref(), join.right.as_ref()) {
            (_, LogicalOperator::PropertyPath(path)) => self
                .path_per_row(path, &join.left, &left_columns)?
                .map(|plan| (plan, true)),
            (LogicalOperator::PropertyPath(path), _) => self
                .path_per_row(path, &join.right, &right_columns)?
                .map(|plan| (plan, false)),
            _ => None,
        };
        if let Some((plan, path_on_right)) = per_row {
            let (input, mut columns, mut types, path_entry) = if path_on_right {
                (left_op, left_columns, left_types, right_entry)
            } else {
                (right_op, right_columns, right_types, first_entry)
            };
            // The path's PROFILE line joins the search's.
            if let Some(entry) = self.profile_entries.borrow_mut().get_mut(path_entry) {
                entry.absorbed = true;
            }
            types.extend(std::iter::repeat_n(
                LogicalType::RdfTermId,
                plan.columns.len(),
            ));
            let operator = path_per_row::PathPerRowOperator {
                input,
                source: self.path_source(),
                config: plan.config,
                subject_column: plan.subject_column,
                object_column: plan.object_column,
                path_columns: plan.columns.len(),
                terms: Arc::clone(&self.terms),
            };
            columns.extend(plan.columns);
            return Ok((Box::new(operator), columns, types));
        }

        // A triple pattern with a variable the other side binds: a lookup
        // per row of that side (the scan planned above reads nothing until
        // it runs, and is dropped). Of two patterns, the one estimated to
        // match more triples is looked up; a wrong guess costs at most
        // `LOOKUP_ROWS` lookups before the join hashes the scan.
        let left_estimate = estimate_operator_cardinality(&join.left, &self.store);
        let right_estimate = estimate_operator_cardinality(&join.right, &self.store);
        // A UNION of patterns with a variable the other side binds (an
        // alternative path is one): a lookup per row of that side in each
        // branch, the planned UNION what the join hashes past the limit. Of
        // a pattern and such a UNION, as of two patterns, the side estimated
        // to match more is looked up.
        let union_first = match (join.left.as_ref(), join.right.as_ref()) {
            (LogicalOperator::TripleScan(_), LogicalOperator::Union(union)) => {
                self.union_patterns_estimate(union) > left_estimate
            }
            (LogicalOperator::Union(union), LogicalOperator::TripleScan(_)) => {
                self.union_patterns_estimate(union) > right_estimate
            }
            _ => false,
        };
        let mut union_lookup = if union_first {
            self.union_lookup_of_join(
                join,
                &left_columns,
                &right_columns,
                left_estimate,
                right_estimate,
            )
        } else {
            None
        };
        let lookup = match (join.left.as_ref(), join.right.as_ref()) {
            _ if union_lookup.is_some() => None,
            (LogicalOperator::TripleScan(left), LogicalOperator::TripleScan(right)) => {
                if left_estimate > right_estimate {
                    self.triple_lookup(
                        left,
                        &join.right,
                        &right_columns,
                        left_estimate,
                        right_estimate,
                    )
                    .map(|plan| (plan, false))
                    .or_else(|| {
                        self.triple_lookup(
                            right,
                            &join.left,
                            &left_columns,
                            right_estimate,
                            left_estimate,
                        )
                        .map(|plan| (plan, true))
                    })
                } else {
                    self.triple_lookup(
                        right,
                        &join.left,
                        &left_columns,
                        right_estimate,
                        left_estimate,
                    )
                    .map(|plan| (plan, true))
                    .or_else(|| {
                        self.triple_lookup(
                            left,
                            &join.right,
                            &right_columns,
                            left_estimate,
                            right_estimate,
                        )
                        .map(|plan| (plan, false))
                    })
                }
            }
            (_, LogicalOperator::TripleScan(right)) => self
                .triple_lookup(
                    right,
                    &join.left,
                    &left_columns,
                    right_estimate,
                    left_estimate,
                )
                .map(|plan| (plan, true)),
            (LogicalOperator::TripleScan(left), _) => self
                .triple_lookup(
                    left,
                    &join.right,
                    &right_columns,
                    left_estimate,
                    right_estimate,
                )
                .map(|plan| (plan, false)),
            _ => None,
        };
        if let Some((parts, scan_on_right)) = lookup {
            let (input, columns, types, scan_entry, cardinalities) = if scan_on_right {
                let estimates = left_estimate.zip(right_estimate);
                (left_op, left_columns, left_types, right_entry, estimates)
            } else {
                let estimates = right_estimate.zip(left_estimate);
                (right_op, right_columns, right_types, first_entry, estimates)
            };
            // The scan's PROFILE line joins the lookup's.
            if let Some(entry) = self.profile_entries.borrow_mut().get_mut(scan_entry) {
                entry.absorbed = true;
            }
            return Ok(self.lookup_join(parts, input, columns, types, cardinalities, false));
        }

        if union_lookup.is_none() && !union_first {
            union_lookup = self.union_lookup_of_join(
                join,
                &left_columns,
                &right_columns,
                left_estimate,
                right_estimate,
            );
        }
        if let Some((plan, union_on_right)) = union_lookup {
            let left = (left_op, left_columns, left_types);
            let right = (right_op, right_columns, right_types);
            return Ok(if union_on_right {
                let entries = right_entry..self.profile_entries.borrow().len();
                let cardinalities = left_estimate.zip(right_estimate);
                self.union_lookup_join(plan, left, right, entries, cardinalities, false)
            } else {
                let cardinalities = right_estimate.zip(left_estimate);
                self.union_lookup_join(
                    plan,
                    right,
                    left,
                    first_entry..right_entry,
                    cardinalities,
                    false,
                )
            });
        }

        let loose = loosely_shared(
            crate::query::plan::always_bound(&join.left),
            crate::query::plan::always_bound(&join.right),
            &left_columns,
            &right_columns,
        );
        if !loose.is_empty() {
            return Ok(common::build_compatible_join(
                left_op,
                right_op,
                &left_columns,
                &right_columns,
                &left_types,
                &right_types,
                &loose,
                grafeo_core::execution::operators::JoinType::Inner,
                None,
            ));
        }

        // Estimate cardinalities for build-side selection
        let cardinalities = estimate_operator_cardinality(&join.left, &self.store)
            .zip(estimate_operator_cardinality(&join.right, &self.store));

        Ok(common::build_inner_join(
            left_op,
            right_op,
            &left_columns,
            &right_columns,
            &left_types,
            &right_types,
            cardinalities,
        ))
    }

    /// Plans a LEFT JOIN operator (for SPARQL OPTIONAL) using HashJoin.
    ///
    /// The condition (the FILTERs of `OPTIONAL { P FILTER(F) }`) decides which
    /// pairs are matches over the merged solution (LeftJoin, SPARQL 1.1 sec
    /// 18.5), so a left row none of whose pairs pass it is kept alone. It
    /// reads the joined row: the left columns, then the right ones (a name
    /// both sides have reads the left one, which the join keys make equal).
    fn plan_left_join(
        &self,
        join: &LeftJoinOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        use crate::query::planner::common;
        let (left_op, left_columns, left_types) = self.plan_operator(&join.left)?;
        let right_entry = self.profile_entries.borrow().len();
        let (right_op, right_columns, right_types) = self.plan_operator(&join.right)?;

        // `OPTIONAL { ?s p ?o }` whose subject or object the left side binds,
        // with no condition: a lookup per left row, which keeps a row with
        // no triple (see `triple_lookup`).
        if join.condition.is_none()
            && let LogicalOperator::TripleScan(scan) = join.right.as_ref()
        {
            let left_estimate = estimate_operator_cardinality(&join.left, &self.store);
            let right_estimate = estimate_operator_cardinality(&join.right, &self.store);
            if let Some(parts) = self.triple_lookup(
                scan,
                &join.left,
                &left_columns,
                right_estimate,
                left_estimate,
            ) {
                // The scan's PROFILE line joins the lookup's.
                if let Some(entry) = self.profile_entries.borrow_mut().get_mut(right_entry) {
                    entry.absorbed = true;
                }
                let cardinalities = left_estimate.zip(right_estimate);
                return Ok(self.lookup_join(
                    parts,
                    left_op,
                    left_columns,
                    left_types,
                    cardinalities,
                    true,
                ));
            }
        }
        // `OPTIONAL { ?s (p|^p) ?o }` the same, in each pattern of the UNION:
        // a left row none of them joins is kept.
        if join.condition.is_none()
            && let LogicalOperator::Union(union) = join.right.as_ref()
        {
            let left_estimate = estimate_operator_cardinality(&join.left, &self.store);
            if let Some(plan) = self.union_lookup(
                union,
                &right_columns,
                &join.left,
                &left_columns,
                left_estimate,
            ) {
                let entries = right_entry..self.profile_entries.borrow().len();
                let cardinalities =
                    left_estimate.zip(estimate_operator_cardinality(&join.right, &self.store));
                return Ok(self.union_lookup_join(
                    plan,
                    (left_op, left_columns, left_types),
                    (right_op, right_columns, right_types),
                    entries,
                    cardinalities,
                    true,
                ));
            }
        }
        let residual = match &join.condition {
            Some(condition) => {
                Some(self.left_join_condition(condition, &left_columns, &right_columns)?)
            }
            None => None,
        };
        let loose = loosely_shared(
            crate::query::plan::always_bound(&join.left),
            crate::query::plan::always_bound(&join.right),
            &left_columns,
            &right_columns,
        );
        if !loose.is_empty() {
            return Ok(common::build_compatible_join(
                left_op,
                right_op,
                &left_columns,
                &right_columns,
                &left_types,
                &right_types,
                &loose,
                grafeo_core::execution::operators::JoinType::Left,
                residual,
            ));
        }
        Ok(common::build_left_join(
            left_op,
            right_op,
            &left_columns,
            &right_columns,
            &left_types,
            &right_types,
            |join| match residual {
                Some(residual) => join.with_residual(residual),
                None => join,
            },
        ))
    }

    /// Plans an ANTI JOIN operator (for SPARQL MINUS).
    ///
    /// A shared variable a side may leave unbound still hashes here (see
    /// [`loosely_shared`] for the joins that check it per pair).
    fn plan_anti_join(
        &self,
        join: &AntiJoinOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        use crate::query::planner::common;
        let (left_op, left_columns, left_types) = self.plan_operator(&join.left)?;
        let (right_op, right_columns, _right_types) = self.plan_operator(&join.right)?;
        // Both sides start from a row that binds something, so their domains are never disjoint:
        // any right row compatible with a left row removes it.
        if self.row.as_deref().is_some_and(RowBinding::binds_anything) {
            let (op, cols) = common::build_not_exists_join(
                left_op,
                right_op,
                left_columns,
                &right_columns,
                left_types.clone(),
            );
            return Ok((op, cols, left_types));
        }
        let (op, cols) = common::build_anti_join(
            left_op,
            right_op,
            left_columns,
            &right_columns,
            left_types.clone(),
        );
        Ok((op, cols, left_types))
    }

    /// Plans a multi-way join. When the Ring Index is available and all inputs
    /// are TripleScans, uses LeapfrogRing (WCOJ) for worst-case optimal joins.
    /// Otherwise falls back to cascading pairwise hash joins.
    fn plan_multi_way_join(
        &self,
        mwj: &crate::query::plan::MultiWayJoinOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        use crate::query::planner::common;

        if mwj.inputs.is_empty() {
            return Err(Error::Internal(
                "MultiWayJoin requires at least one input".to_string(),
            ));
        }

        // Try Ring-backed LeapfrogRing (WCOJ) when all inputs are TripleScans.
        // Skip when companion columns are needed (LANG/DATATYPE functions)
        // because the leapfrog operator emits raw variable columns only.
        #[cfg(feature = "ring-index")]
        if let Some(result) = self.try_leapfrog_ring(mwj) {
            return result;
        }

        // Plan all inputs and estimate cardinalities
        type Planned = (
            Box<dyn Operator>,
            Vec<String>,
            Vec<LogicalType>,
            f64,
            Option<std::collections::HashSet<String>>,
        );
        let mut planned: Vec<Planned> = Vec::new();
        for input in &mwj.inputs {
            let (op, cols, types) = self.plan_operator(input)?;
            let card = estimate_operator_cardinality(input, &self.store).unwrap_or(1000.0);
            let bound = crate::query::plan::always_bound(input);
            planned.push((op, cols, types, card, bound));
        }

        // Sort by cardinality (smallest first) so we build on smaller inputs
        planned.sort_by(|a, b| a.3.total_cmp(&b.3));

        // Fold with pairwise hash joins, each time on the smallest input that
        // shares a variable with the rows so far (see `next_join_input`); a
        // shared variable a side may leave unbound is checked per pair (see
        // `loosely_shared`).
        let (mut current_op, mut current_cols, mut current_types, mut current_card, mut bound) =
            planned.remove(0);
        while !planned.is_empty() {
            let next = next_join_input(
                planned.iter().map(|input| input.1.as_slice()),
                &current_cols,
            );
            let (right_op, right_cols, right_types, right_card, right_bound) = planned.remove(next);
            let loose = loosely_shared(
                bound.clone(),
                right_bound.clone(),
                &current_cols,
                &right_cols,
            );
            let (joined_op, joined_cols, joined_types) = if loose.is_empty() {
                common::build_inner_join(
                    current_op,
                    right_op,
                    &current_cols,
                    &right_cols,
                    &current_types,
                    &right_types,
                    Some((current_card, right_card)),
                )
            } else {
                common::build_compatible_join(
                    current_op,
                    right_op,
                    &current_cols,
                    &right_cols,
                    &current_types,
                    &right_types,
                    &loose,
                    grafeo_core::execution::operators::JoinType::Inner,
                    None,
                )
            };
            bound = bound.zip(right_bound).map(|(mut left, right)| {
                left.extend(right);
                left
            });
            // Rough estimate for cascaded join output
            current_card = (current_card * right_card * 0.1).max(1.0);
            current_op = joined_op;
            current_cols = joined_cols;
            current_types = joined_types;
        }

        Ok((current_op, current_cols, current_types))
    }

    /// Attempts to plan a multi-way join using LeapfrogRing (WCOJ).
    ///
    /// Returns `Some(Ok(...))` when the Ring is available and all inputs are
    /// simple TripleScans, `None` to fall back to cascading hash joins.
    #[cfg(feature = "ring-index")]
    fn try_leapfrog_ring(
        &self,
        mwj: &crate::query::plan::MultiWayJoinOp,
    ) -> Option<Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)>> {
        use grafeo_core::index::ring::AnnotatedPattern;

        // The ring misses the transaction's own writes.
        if self.has_pending_writes() {
            return None;
        }
        let ring = self.store.ring()?;

        // All inputs must be simple TripleScans (no chained input, no graph context)
        let mut annotated = Vec::new();
        let mut all_vars: Vec<String> = Vec::new();
        for input in &mwj.inputs {
            let LogicalOperator::TripleScan(scan) = input else {
                return None;
            };
            if scan.input.is_some() || scan.graph.is_some() || scan.dataset.is_some() {
                return None;
            }
            let pattern = self.build_triple_pattern(scan);
            let ap = AnnotatedPattern {
                pattern,
                subject_var: scan.subject.as_variable().map(str::to_string),
                predicate_var: scan.predicate.as_variable().map(str::to_string),
                object_var: scan.object.as_variable().map(str::to_string),
            };
            // Collect output variables in order
            for v in [&ap.subject_var, &ap.predicate_var, &ap.object_var]
                .into_iter()
                .flatten()
            {
                if !all_vars.contains(v) {
                    all_vars.push(v.clone());
                }
            }
            annotated.push(ap);
        }

        let columns = all_vars.clone();
        let types = vec![LogicalType::RdfTermId; columns.len()];
        let operator = Box::new(RdfLeapfrogOperator::new(
            ring,
            annotated,
            columns.clone(),
            Arc::clone(&self.terms),
        ));
        Some(Ok((operator, columns, types)))
    }

    /// Plans a UNION operator.
    ///
    /// The output variables are the union of all branches' variables (SPARQL
    /// 1.1 sec 18.2.1); a variable missing from a branch is unbound (NULL) there.
    /// Each branch is aligned by name and padded to the full width (#365).
    fn plan_union(
        &self,
        union: &crate::query::plan::UnionOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        if union.inputs.is_empty() {
            return Err(Error::Internal("Empty UNION".to_string()));
        }

        // Plan every branch, keeping its own columns and types.
        let mut planned: Vec<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> =
            Vec::with_capacity(union.inputs.len());
        for input in &union.inputs {
            planned.push(self.plan_operator(input)?);
        }

        // A single branch needs no alignment.
        if planned.len() == 1 {
            let (op, cols, tys) = planned.into_iter().next().expect("single-element vector");
            return Ok((op, cols, tys));
        }

        // Union of branch variables in first-seen order; a variable bound with
        // different types in different branches widens to Any.
        let mut unified_columns: Vec<String> = Vec::new();
        let mut unified_types: Vec<LogicalType> = Vec::new();
        for (_, cols, tys) in &planned {
            for (col, ty) in cols.iter().zip(tys.iter()) {
                if let Some(existing) = unified_columns.iter().position(|c| c == col) {
                    if unified_types[existing] != *ty {
                        unified_types[existing] = LogicalType::Any;
                    }
                } else {
                    unified_columns.push(col.clone());
                    unified_types.push(ty.clone());
                }
            }
        }

        // Pad each branch to the full width with real NULL columns: DISTINCT and
        // ORDER BY copy rows at each chunk's own width.
        let operators: Vec<Box<dyn Operator>> = planned
            .into_iter()
            .map(|(op, cols, _)| {
                pad_union_branch(
                    op,
                    &cols,
                    &unified_columns,
                    &unified_types,
                    &self.terms,
                    self.expression_context(),
                )
            })
            .collect();

        let operator = Box::new(RdfUnionOperator::new(operators, unified_columns.len()));
        Ok((operator, unified_columns, unified_types))
    }

    /// Plans an INSERT TRIPLE operator.
    fn plan_insert_triple(
        &self,
        insert: &InsertTripleOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        // Check if this is a pattern-based insert (has variables in the template).
        // Blank nodes are concrete values, not variables, so they don't trigger
        // the pattern-based path.
        let has_variables = matches!(&insert.subject, TripleComponent::Variable(_))
            || matches!(&insert.predicate, TripleComponent::Variable(_))
            || matches!(&insert.object, TripleComponent::Variable(_));

        if has_variables {
            // Pattern-based insertion: need to query first, then insert each match
            if let Some(ref input) = insert.input {
                let (input_op, input_columns, _input_types) = self.plan_operator(input)?;

                // Build column index map for variable substitution
                let column_map: HashMap<String, usize> = input_columns
                    .iter()
                    .enumerate()
                    .map(|(i, name)| (name.clone(), i))
                    .collect();

                let operator = Box::new(RdfInsertPatternOperator::new(
                    self.writer.clone(),
                    input_op,
                    TripleOperands {
                        subject: insert.subject.clone(),
                        predicate: insert.predicate.clone(),
                        object: insert.object.clone(),
                        column_map,
                        graph: insert.graph.clone(),
                        terms: Arc::clone(&self.terms),
                    },
                ));

                return Ok((operator, Vec::new(), Vec::new()));
            }
        }

        // Direct insertion with concrete terms
        let subject = self.component_to_term(&insert.subject)?;
        let predicate = self.component_to_term(&insert.predicate)?;
        let object = self.component_to_term(&insert.object)?;
        // A literal subject or a predicate that is not an IRI is no RDF
        // triple: a template skips it (above), data written out is refused.
        if !is_valid_triple(&subject, &predicate) {
            return Err(Error::from(QueryError::new(
                QueryErrorKind::Semantic,
                format!(
                    "INSERT DATA: {subject} {predicate} is not a triple's subject and predicate: \
                     a subject is an IRI or a blank node, a predicate an IRI"
                ),
            )));
        }

        let triple = Triple::new(subject, predicate, object);
        let operator = Box::new(RdfInsertTripleOperator::new(
            self.writer.clone(),
            triple,
            insert.graph.clone(),
        ));

        // Insert operations don't output columns
        Ok((operator, Vec::new(), Vec::new()))
    }

    /// The term of a constant component of `INSERT DATA` or `DELETE DATA`.
    fn component_to_term(&self, component: &TripleComponent) -> Result<Term> {
        component_to_term(component).ok_or_else(|| match component {
            TripleComponent::Variable(name) => {
                Error::Internal(format!("Unbound variable '{name}' in INSERT DATA"))
            }
            other => Error::Internal(format!("{other:?} is no RDF term")),
        })
    }

    /// Plans a DELETE TRIPLE operator: with an input, the template's
    /// instance for each of its rows (none without a row, also for a
    /// triple without variables), otherwise the triple of `DELETE DATA`.
    fn plan_delete_triple(
        &self,
        delete: &DeleteTripleOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        if let Some(ref input) = delete.input {
            let (input_op, input_columns, _input_types) = self.plan_operator(input)?;
            let operator = Box::new(RdfDeletePatternOperator::new(
                self.writer.clone(),
                input_op,
                TripleOperands {
                    subject: delete.subject.clone(),
                    predicate: delete.predicate.clone(),
                    object: delete.object.clone(),
                    column_map: column_map(&input_columns),
                    graph: delete.graph.clone(),
                    terms: Arc::clone(&self.terms),
                },
            ));
            return Ok((operator, Vec::new(), Vec::new()));
        }

        let subject = self.component_to_term(&delete.subject)?;
        let predicate = self.component_to_term(&delete.predicate)?;
        let object = self.component_to_term(&delete.object)?;

        let triple = Triple::new(subject, predicate, object);
        let operator = Box::new(RdfDeleteTripleOperator::new(
            self.writer.clone(),
            triple,
            delete.graph.clone(),
        ));

        Ok((operator, Vec::new(), Vec::new()))
    }

    /// Plans a CLEAR GRAPH operator.
    fn plan_clear_graph(
        &self,
        clear: &ClearGraphOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        let op = RdfGraphOp::Clear {
            target: clear.target.clone(),
        };
        Ok(self.graph_operator("RdfClearGraph", op, GraphCheck::None))
    }

    /// Plans a CREATE GRAPH operator.
    fn plan_create_graph(
        &self,
        create: &CreateGraphOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        let op = RdfGraphOp::Create {
            name: create.graph.clone(),
        };
        let check = GraphCheck::Absent {
            graph: create.graph.clone(),
            silent: create.silent,
        };
        Ok(self.graph_operator("RdfCreateGraph", op, check))
    }

    /// Plans a DROP GRAPH operator.
    fn plan_drop_graph(
        &self,
        drop_op: &DropGraphOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        let target = drop_op.target.clone();
        let check = match &target {
            RdfGraphTarget::Named(name) => GraphCheck::Present {
                graph: name.clone(),
                silent: drop_op.silent,
                what: "Graph",
            },
            _ => GraphCheck::None,
        };
        Ok(self.graph_operator("RdfDropGraph", RdfGraphOp::Drop { target }, check))
    }

    /// Plans a COPY graph operator.
    fn plan_copy_graph(
        &self,
        copy: &CopyGraphOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        let op = RdfGraphOp::Copy {
            source: copy.source.clone(),
            target: copy.destination.clone(),
        };
        let check = GraphCheck::source(&copy.source, copy.silent);
        Ok(self.graph_operator("RdfCopyGraph", op, check))
    }

    /// Plans a MOVE graph operator.
    fn plan_move_graph(
        &self,
        move_op: &MoveGraphOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        let op = RdfGraphOp::Move {
            source: move_op.source.clone(),
            target: move_op.destination.clone(),
        };
        let check = GraphCheck::source(&move_op.source, move_op.silent);
        Ok(self.graph_operator("RdfMoveGraph", op, check))
    }

    /// Plans an ADD graph operator.
    fn plan_add_graph(
        &self,
        add: &AddGraphOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        let op = RdfGraphOp::Add {
            source: add.source.clone(),
            target: add.destination.clone(),
        };
        let check = GraphCheck::source(&add.source, add.silent);
        Ok(self.graph_operator("RdfAddGraph", op, check))
    }

    /// The operator of the whole-graph operation `op`, named `name`, which
    /// `check` checks before it applies.
    fn graph_operator(
        &self,
        name: &'static str,
        op: RdfGraphOp,
        check: GraphCheck,
    ) -> (Box<dyn Operator>, Vec<String>, Vec<LogicalType>) {
        let operator = Box::new(RdfGraphOperator {
            writer: self.writer.clone(),
            name,
            op,
            check,
            done: false,
        });
        (operator, Vec::new(), Vec::new())
    }

    /// Plans a SPARQL MODIFY operator (DELETE/INSERT WHERE).
    ///
    /// Per SPARQL 1.1 spec:
    /// 1. Evaluate WHERE clause once to get bindings
    /// 2. Apply DELETE templates using those bindings
    /// 3. Apply INSERT templates using the SAME bindings
    fn plan_modify(
        &self,
        modify: &ModifyOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        // Plan the WHERE clause
        let (where_op, where_columns, _where_types) = self.plan_operator(&modify.where_clause)?;

        // Build column index map for variable substitution
        let column_map: HashMap<String, usize> = where_columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        let operator = Box::new(RdfModifyOperator {
            writer: self.writer.clone(),
            input: where_op,
            delete_templates: modify.delete_templates.clone(),
            insert_templates: modify.insert_templates.clone(),
            column_map,
            terms: Arc::clone(&self.terms),
            done: false,
        });

        Ok((operator, Vec::new(), Vec::new()))
    }
}

// ============================================================================
// RDF Insert Triple Operator
// ============================================================================

/// Operator that inserts a triple: it records the insert in the
/// transaction's change set, which applies it at the commit.
struct RdfInsertTripleOperator {
    writer: Option<RdfWriter>,
    triple: Triple,
    graph_name: Option<String>,
    inserted: bool,
}

impl RdfInsertTripleOperator {
    fn new(writer: Option<RdfWriter>, triple: Triple, graph_name: Option<String>) -> Self {
        Self {
            writer,
            triple,
            graph_name,
            inserted: false,
        }
    }
}

impl Operator for RdfInsertTripleOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.inserted {
            return Ok(None);
        }
        check_mutation_graph(self.graph_name.as_deref())?;
        update_writer(&self.writer)?.insert(self.graph_name.as_deref(), self.triple.clone())?;
        self.inserted = true;

        // Return an empty result (INSERT doesn't produce rows)
        Ok(None)
    }

    fn reset(&mut self) {
        self.inserted = false;
    }

    fn name(&self) -> &'static str {
        "RdfInsertTriple"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

// ============================================================================
// RDF Insert Pattern Operator
// ============================================================================

/// Operator that inserts triples based on a pattern from the RDF store.
/// Used for INSERT { } WHERE { } operations where the triple template contains variables.
struct RdfInsertPatternOperator {
    writer: Option<RdfWriter>,
    input: Box<dyn Operator>,
    subject: TripleComponent,
    predicate: TripleComponent,
    object: TripleComponent,
    column_map: HashMap<String, usize>,
    /// Named graph to insert into (`None` = default graph).
    graph_name: Option<String>,
    terms: Arc<QueryTermIds>,
    done: bool,
}

impl RdfInsertPatternOperator {
    fn new(writer: Option<RdfWriter>, input: Box<dyn Operator>, operands: TripleOperands) -> Self {
        Self {
            writer,
            input,
            subject: operands.subject,
            predicate: operands.predicate,
            object: operands.object,
            column_map: operands.column_map,
            graph_name: operands.graph,
            terms: operands.terms,
            done: false,
        }
    }
}

impl Operator for RdfInsertPatternOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.done {
            return Ok(None);
        }

        // Check the target first so a variable graph errors even with no rows.
        check_mutation_graph(self.graph_name.as_deref())?;
        let writer = update_writer(&self.writer)?;

        // Collect all triples to insert
        let mut triples_to_insert = Vec::new();

        while let Some(chunk) = self.input.next()? {
            for row in chunk.selected_indices() {
                let mut blanks = FreshBlankNodes::default();
                let mut term = |component| {
                    template_term(
                        component,
                        &self.column_map,
                        &chunk,
                        row,
                        &self.terms,
                        &mut blanks,
                    )
                };
                if let (Some(s), Some(p), Some(o)) = (
                    term(&self.subject),
                    term(&self.predicate),
                    term(&self.object),
                ) && is_valid_triple(&s, &p)
                {
                    triples_to_insert.push(Triple::new(s, p, o));
                }
            }
        }

        // Only the triples that are new, as the transaction sees the graph,
        // are recorded.
        for triple in triples_to_insert {
            writer.insert(self.graph_name.as_deref(), triple)?;
        }

        self.done = true;
        Ok(None)
    }

    fn reset(&mut self) {
        self.done = false;
        self.input.reset();
    }

    fn name(&self) -> &'static str {
        "RdfInsertPattern"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

// ============================================================================
// RDF Delete Triple Operator
// ============================================================================

/// Operator that deletes a triple: it records the delete in the
/// transaction's change set, which applies it at the commit.
struct RdfDeleteTripleOperator {
    writer: Option<RdfWriter>,
    triple: Triple,
    graph_name: Option<String>,
    deleted: bool,
}

impl RdfDeleteTripleOperator {
    fn new(writer: Option<RdfWriter>, triple: Triple, graph_name: Option<String>) -> Self {
        Self {
            writer,
            triple,
            graph_name,
            deleted: false,
        }
    }
}

impl Operator for RdfDeleteTripleOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.deleted {
            return Ok(None);
        }
        check_mutation_graph(self.graph_name.as_deref())?;
        update_writer(&self.writer)?.delete(self.graph_name.as_deref(), &self.triple)?;
        self.deleted = true;

        // Return an empty result (DELETE doesn't produce rows)
        Ok(None)
    }

    fn reset(&mut self) {
        self.deleted = false;
    }

    fn name(&self) -> &'static str {
        "RdfDeleteTriple"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

// ============================================================================
// RDF Delete Pattern Operator
// ============================================================================

/// Operator that deletes triples matching a pattern from the RDF store.
/// Used for DELETE WHERE operations where the triple template contains variables.
struct RdfDeletePatternOperator {
    writer: Option<RdfWriter>,
    input: Box<dyn Operator>,
    subject: TripleComponent,
    predicate: TripleComponent,
    object: TripleComponent,
    column_map: HashMap<String, usize>,
    /// Named graph to delete from (`None` = default graph).
    graph_name: Option<String>,
    terms: Arc<QueryTermIds>,
    done: bool,
}

impl RdfDeletePatternOperator {
    fn new(writer: Option<RdfWriter>, input: Box<dyn Operator>, operands: TripleOperands) -> Self {
        Self {
            writer,
            input,
            subject: operands.subject,
            predicate: operands.predicate,
            object: operands.object,
            column_map: operands.column_map,
            graph_name: operands.graph,
            terms: operands.terms,
            done: false,
        }
    }
}

impl Operator for RdfDeletePatternOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.done {
            return Ok(None);
        }

        check_mutation_graph(self.graph_name.as_deref())?;
        let writer = update_writer(&self.writer)?;

        // Collect all triples to delete
        let mut triples_to_delete = Vec::new();

        while let Some(chunk) = self.input.next()? {
            for row in chunk.selected_indices() {
                let mut blanks = FreshBlankNodes::default();
                let mut term = |component| {
                    template_term(
                        component,
                        &self.column_map,
                        &chunk,
                        row,
                        &self.terms,
                        &mut blanks,
                    )
                };
                if let (Some(s), Some(p), Some(o)) = (
                    term(&self.subject),
                    term(&self.predicate),
                    term(&self.object),
                ) {
                    triples_to_delete.push(Triple::new(s, p, o));
                }
            }
        }

        // Only the triples the transaction sees in the graph are recorded.
        for triple in &triples_to_delete {
            writer.delete(self.graph_name.as_deref(), triple)?;
        }

        self.done = true;
        Ok(None)
    }

    fn reset(&mut self) {
        self.done = false;
        self.input.reset();
    }

    fn name(&self) -> &'static str {
        "RdfDeletePattern"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

// ============================================================================
// RDF Graph Operations (CREATE, DROP, CLEAR, COPY, MOVE, ADD)
// ============================================================================

/// What a whole-graph operation checks before it applies.
enum GraphCheck {
    /// Nothing: it always applies.
    None,
    /// The graph must not exist (`CREATE`); `SILENT` skips the operation
    /// when it does.
    Absent { graph: String, silent: bool },
    /// The named graph must exist (`DROP`, the source of `COPY`, `MOVE` and
    /// `ADD`, which `what` names in the error); `SILENT` skips a `DROP` of a
    /// graph that does not exist, and lets the others read it as empty.
    Present {
        graph: String,
        silent: bool,
        what: &'static str,
    },
}

impl GraphCheck {
    /// The check of the source of a `COPY`, `MOVE` or `ADD`: none for the
    /// default graph, which always exists.
    fn source(source: &Option<String>, silent: bool) -> Self {
        match source {
            None => Self::None,
            Some(graph) => Self::Present {
                graph: graph.clone(),
                silent,
                what: "Source graph",
            },
        }
    }
}

/// Operator that runs a whole-graph operation as a standalone change (see
/// [`RdfWriter::graph_op`]): it takes effect at once, also inside a
/// transaction, whose rollback keeps it.
struct RdfGraphOperator {
    writer: Option<RdfWriter>,
    /// The operator's name in a plan.
    name: &'static str,
    op: RdfGraphOp,
    check: GraphCheck,
    done: bool,
}

impl Operator for RdfGraphOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.done {
            return Ok(None);
        }
        self.done = true;
        let skips_missing = matches!(self.op, RdfGraphOp::Drop { .. });
        let check = &self.check;
        update_writer(&self.writer)?.graph_op(self.op.clone(), |store| match check {
            GraphCheck::None => Ok(true),
            GraphCheck::Absent { graph, silent } => match (store.graph(graph).is_some(), silent) {
                (false, _) => Ok(true),
                (true, true) => Ok(false),
                (true, false) => Err(OperatorError::Execution(format!(
                    "Graph <{graph}> already exists"
                ))),
            },
            GraphCheck::Present {
                graph,
                silent,
                what,
            } => match (store.graph(graph).is_some(), silent) {
                (true, _) => Ok(true),
                (false, true) => Ok(!skips_missing),
                (false, false) => Err(OperatorError::Execution(format!(
                    "{what} <{graph}> does not exist"
                ))),
            },
        })?;
        Ok(None)
    }

    fn reset(&mut self) {
        self.done = false;
    }

    fn name(&self) -> &'static str {
        self.name
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

// ============================================================================
// RDF Modify Operator (SPARQL DELETE/INSERT WHERE)
// ============================================================================

/// Operator that handles SPARQL MODIFY operations (DELETE/INSERT WHERE).
///
/// Per SPARQL 1.1 Update spec:
/// 1. Evaluate WHERE clause once to get all bindings
/// 2. Apply DELETE templates to each binding
/// 3. Apply INSERT templates to each binding (using SAME bindings)
struct RdfModifyOperator {
    writer: Option<RdfWriter>,
    input: Box<dyn Operator>,
    delete_templates: Vec<TripleTemplate>,
    insert_templates: Vec<TripleTemplate>,
    column_map: HashMap<String, usize>,
    terms: Arc<QueryTermIds>,
    done: bool,
}

impl Operator for RdfModifyOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.done {
            return Ok(None);
        }
        let writer = update_writer(&self.writer)?;

        // Step 1: every solution of the WHERE clause, before any change.
        let mut bindings: Vec<(DataChunk, usize)> = Vec::new();
        while let Some(chunk) = self.input.next()? {
            for row in chunk.selected_indices() {
                bindings.push((chunk.clone(), row));
            }
        }

        // Step 2: the DELETE templates, each variable its exact bound term.
        for template in &self.delete_templates {
            let graph = template.graph.as_deref();
            check_mutation_graph(graph)?;
            for (chunk, row) in &bindings {
                let mut blanks = FreshBlankNodes::default();
                let mut term = |component| {
                    template_term(
                        component,
                        &self.column_map,
                        chunk,
                        *row,
                        &self.terms,
                        &mut blanks,
                    )
                };
                if let (Some(s), Some(p), Some(o)) = (
                    term(&template.subject),
                    term(&template.predicate),
                    term(&template.object),
                ) {
                    writer.delete(graph, &Triple::new(s, p, o))?;
                }
            }
        }

        // Step 3: the INSERT templates over the same solutions, a template
        // blank node fresh for each solution.
        for (chunk, row) in &bindings {
            let mut blanks = FreshBlankNodes::default();
            for template in &self.insert_templates {
                let graph = template.graph.as_deref();
                check_mutation_graph(graph)?;
                let mut term = |component| {
                    template_term(
                        component,
                        &self.column_map,
                        chunk,
                        *row,
                        &self.terms,
                        &mut blanks,
                    )
                };
                if let (Some(s), Some(p), Some(o)) = (
                    term(&template.subject),
                    term(&template.predicate),
                    term(&template.object),
                ) && is_valid_triple(&s, &p)
                {
                    writer.insert(graph, Triple::new(s, p, o))?;
                }
            }
        }

        self.done = true;
        Ok(None)
    }

    fn reset(&mut self) {
        self.done = false;
        self.input.reset();
    }

    fn name(&self) -> &'static str {
        "RdfModify"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

// ============================================================================
// RDF Union Operator
// ============================================================================

/// Emits the chunks of each (already width-aligned) UNION branch in turn,
/// without de-duplication (SPARQL 1.1 sec 18.5).
struct RdfUnionOperator {
    operators: Vec<Box<dyn Operator>>,
    current_idx: usize,
    /// Unified output width (the number of columns in the union variable
    /// domain). Every branch is null-padded to this width by `plan_union`.
    expected_width: usize,
}

impl RdfUnionOperator {
    fn new(operators: Vec<Box<dyn Operator>>, expected_width: usize) -> Self {
        Self {
            operators,
            current_idx: 0,
            expected_width,
        }
    }
}

impl Operator for RdfUnionOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        // Execute all operators
        while self.current_idx < self.operators.len() {
            let op = &mut self.operators[self.current_idx];
            match op.next()? {
                Some(chunk) => {
                    debug_assert_eq!(
                        chunk.column_count(),
                        self.expected_width,
                        "RdfUnion branch {} emitted a {}-column chunk but the unified \
                         UNION schema is {} columns wide; branches must be null-padded \
                         to a uniform width for DISTINCT / ORDER BY correctness",
                        self.current_idx,
                        chunk.column_count(),
                        self.expected_width,
                    );
                    return Ok(Some(chunk));
                }
                None => self.current_idx += 1,
            }
        }
        Ok(None)
    }

    fn reset(&mut self) {
        self.current_idx = 0;
        for op in &mut self.operators {
            op.reset();
        }
    }

    fn name(&self) -> &'static str {
        "RdfUnion"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

// ============================================================================
// RDF Bind Operator
// ============================================================================

/// Operator that appends a computed column to each row: the term id of an
/// expression's value (SPARQL `BIND`), NULL when it errs.
struct RdfBindOperator {
    child: Box<dyn Operator>,
    evaluator: RdfExpressionPredicate,
    terms: Arc<QueryTermIds>,
}

impl Operator for RdfBindOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        let Some(input) = self.child.next()? else {
            return Ok(None);
        };
        let rows: Vec<usize> = input.selected_indices().collect();
        let ids = rows
            .iter()
            .map(|&row| intern_value(&self.terms, self.evaluator.eval(&input, row)))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(Some(chunk_with_column(&input, &rows, ids)))
    }

    fn reset(&mut self) {
        self.child.reset();
    }

    fn name(&self) -> &'static str {
        "RdfBind"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

// ============================================================================
// RDF Project Operator
// ============================================================================

/// What an evaluated projection gives for an expression's value.
#[derive(Debug, Clone, Copy)]
enum EvaluatedOutput {
    /// The value as an RDF term ([`Value::RdfTerm`]): a sort key, a MIN,
    /// MAX or SAMPLE argument.
    Term,
    /// The number of a numeric value, the term of any other (an error for
    /// SUM and AVG), NULL when unbound: a SUM or AVG argument.
    Number,
    /// The lexical form (`STR`): a GROUP_CONCAT argument.
    Lexical,
}

/// One column of an [`RdfProjectOperator`].
enum RdfProjectExpr {
    /// An input column, copied (term ids stay ids).
    Column(usize),
    /// A constant: a term id, or NULL.
    Constant(Value),
    /// An expression's value, interned: a term id, NULL when it errs.
    Expression {
        expr: FilterExpression,
        variable_columns: HashMap<String, usize>,
    },
    /// An expression's value as `output` says, not interned: the columns an
    /// ORDER BY or an aggregate reads, which go no further.
    Evaluated {
        expr: FilterExpression,
        variable_columns: HashMap<String, usize>,
        output: EvaluatedOutput,
    },
}

/// The project operator of SPARQL plans: evaluates expressions with the
/// SPARQL evaluator and interns their values.
struct RdfProjectOperator {
    child: Box<dyn Operator>,
    projections: Vec<(RdfProjectExpr, Option<RdfExpressionPredicate>)>,
    output_types: Vec<LogicalType>,
    terms: Arc<QueryTermIds>,
}

impl RdfProjectOperator {
    fn new(
        child: Box<dyn Operator>,
        projections: Vec<RdfProjectExpr>,
        output_types: Vec<LogicalType>,
        terms: Arc<QueryTermIds>,
        context: ExpressionContext,
    ) -> Self {
        assert_eq!(projections.len(), output_types.len());
        let projections = projections
            .into_iter()
            .map(|projection| {
                let evaluator = match &projection {
                    RdfProjectExpr::Expression {
                        expr,
                        variable_columns,
                    }
                    | RdfProjectExpr::Evaluated {
                        expr,
                        variable_columns,
                        ..
                    } => Some(RdfExpressionPredicate::new(
                        expr.clone(),
                        variable_columns.clone(),
                        Arc::clone(&terms),
                        context.clone(),
                    )),
                    _ => None,
                };
                (projection, evaluator)
            })
            .collect();
        Self {
            child,
            projections,
            output_types,
            terms,
        }
    }
}

impl Operator for RdfProjectOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        let Some(input) = self.child.next()? else {
            return Ok(None);
        };
        let rows: Vec<usize> = input.selected_indices().collect();
        let mut output = DataChunk::with_capacity(&self.output_types, rows.len());

        for (i, (projection, evaluator)) in self.projections.iter().enumerate() {
            match projection {
                RdfProjectExpr::Column(col_idx) => {
                    if input.column(*col_idx).is_none() {
                        return Err(OperatorError::ColumnNotFound(format!("Column {col_idx}")));
                    }
                    copy_column(&input, *col_idx, &rows, &mut output, i);
                }
                RdfProjectExpr::Constant(value) => {
                    let column = output.column_mut(i).expect("column exists");
                    for _ in &rows {
                        column.push_value(value.clone());
                    }
                }
                RdfProjectExpr::Expression { .. } => {
                    let evaluator = evaluator.as_ref().expect("expression evaluator");
                    let ids = rows
                        .iter()
                        .map(|&row| intern_value(&self.terms, evaluator.eval(&input, row)))
                        .collect::<std::result::Result<Vec<_>, _>>()?;
                    let column = output.column_mut(i).expect("column exists");
                    for id in ids {
                        column.push_value(id);
                    }
                }
                RdfProjectExpr::Evaluated { output: kind, .. } => {
                    let evaluator = evaluator.as_ref().expect("expression evaluator");
                    let values: Vec<Value> = rows
                        .iter()
                        .map(|&row| {
                            let value = evaluator.eval(&input, row);
                            value
                                .and_then(|value| evaluated(&value, *kind))
                                .unwrap_or(Value::Null)
                        })
                        .collect();
                    let column = output.column_mut(i).expect("column exists");
                    for value in values {
                        column.push_value(value);
                    }
                }
            }
        }

        output.set_count(rows.len());
        Ok(Some(output))
    }

    fn reset(&mut self) {
        self.child.reset();
    }

    fn name(&self) -> &'static str {
        "RdfProject"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// What an [`RdfProjectExpr::Evaluated`] column holds for `value`.
fn evaluated(value: &Value, output: EvaluatedOutput) -> Option<Value> {
    match output {
        EvaluatedOutput::Term => terms::to_term(value).map(|term| Value::RdfTerm(Arc::new(term))),
        // A term that is no number goes on as itself, which SUM and AVG take
        // as an error (their result is unbound); an unbound value is skipped.
        EvaluatedOutput::Number => match terms::native(value).map(std::borrow::Cow::into_owned) {
            Some(number @ (Value::Int64(_) | Value::Float64(_) | Value::Decimal(_))) => {
                Some(number)
            }
            _ => terms::to_term(value).map(|term| Value::RdfTerm(Arc::new(term))),
        },
        EvaluatedOutput::Lexical => terms::str_of(value).map(|text| Value::from(&*text)),
    }
}

/// The id of an expression's value: its term interned, NULL for an error.
fn intern_value(
    terms: &QueryTermIds,
    value: Option<Value>,
) -> std::result::Result<Value, OperatorError> {
    match value.as_ref().and_then(terms::to_term) {
        Some(term) => Ok(id_value(terms.intern(&term)?)),
        None => Ok(Value::Null),
    }
}

/// `rows` of `input` with a column more, of term ids, one per row.
fn id_at(chunk: &DataChunk, column: usize, row: usize) -> Option<i64> {
    match chunk.column(column)?.get_value(row)? {
        Value::Int64(id) => Some(id),
        _ => None,
    }
}

fn chunk_with_column(input: &DataChunk, rows: &[usize], ids: Vec<Value>) -> DataChunk {
    let mut output_types: Vec<LogicalType> = (0..input.column_count())
        .map(|i| {
            input
                .column(i)
                .map_or(LogicalType::Any, |c| c.data_type().clone())
        })
        .collect();
    output_types.push(LogicalType::RdfTermId);
    let mut output = DataChunk::with_capacity(&output_types, rows.len());
    for col_idx in 0..input.column_count() {
        copy_column(input, col_idx, rows, &mut output, col_idx);
    }
    let column = output
        .column_mut(input.column_count())
        .expect("column exists: the new column is last in the schema");
    for id in ids {
        column.push_value(id);
    }
    output.set_count(rows.len());
    output
}

/// Copies column `from` of `input` at `rows` into column `to` of `output`.
fn copy_column(input: &DataChunk, from: usize, rows: &[usize], output: &mut DataChunk, to: usize) {
    let Some(source) = input.column(from) else {
        return;
    };
    let target = output.column_mut(to).expect("column exists");
    for &row in rows {
        target.push_value(source.get_value(row).unwrap_or(Value::Null));
    }
}

// ============================================================================
// RDF Triple Scan Operator
// ============================================================================

/// Graph resolution context for a triple scan.
///
/// Groups the graph IRI, scan-all flag, and SPARQL dataset restriction
/// to keep the operator constructor argument count manageable.
#[derive(Clone)]
struct GraphContext {
    /// Named graph to query. `None` = default graph.
    graph: Option<String>,
    /// Whether to scan ALL graphs (when GRAPH ?var is used).
    scan_all_graphs: bool,
    /// SPARQL dataset restriction from FROM / FROM NAMED clauses.
    dataset: Option<DatasetRestriction>,
}

/// LeapfrogRing WCOJ operator for Ring-backed multi-way joins.
///
/// Wraps `LeapfrogRing::with_variables()` and converts each join result
/// (Vec<Triple>) into a `DataChunk` with the appropriate variable columns.
#[cfg(feature = "ring-index")]
struct RdfLeapfrogOperator {
    ring: Arc<grafeo_core::index::ring::TripleRing>,
    annotated_patterns: Vec<grafeo_core::index::ring::AnnotatedPattern>,
    /// Output column names (deduplicated variables).
    output_columns: Vec<String>,
    /// Buffered results from the LeapfrogRing iterator.
    results: Vec<Vec<Triple>>,
    /// Current position in results buffer.
    position: usize,
    /// Whether the leapfrog join has been executed.
    executed: bool,
    terms: Arc<QueryTermIds>,
}

#[cfg(feature = "ring-index")]
impl RdfLeapfrogOperator {
    fn new(
        ring: Arc<grafeo_core::index::ring::TripleRing>,
        annotated_patterns: Vec<grafeo_core::index::ring::AnnotatedPattern>,
        output_columns: Vec<String>,
        terms: Arc<QueryTermIds>,
    ) -> Self {
        Self {
            ring,
            annotated_patterns,
            output_columns,
            results: Vec::new(),
            position: 0,
            executed: false,
            terms,
        }
    }

    fn execute_join(&mut self) {
        if self.executed {
            return;
        }
        self.executed = true;

        let join = grafeo_core::index::ring::LeapfrogRing::with_variables(
            &self.ring,
            self.annotated_patterns.clone(),
        );
        self.results = join.collect();
    }

    /// The term a variable is bound to in a set of matched triples.
    /// The id a variable takes in a match: the store's id the ring triple
    /// carries, or its term interned (a ring of triples no store took);
    /// `None` when no pattern binds it.
    fn resolve_variable(
        &self,
        var: &str,
        matched_triples: &[Triple],
    ) -> Option<std::result::Result<TermId, OperatorError>> {
        for (ap, triple) in self.annotated_patterns.iter().zip(matched_triples) {
            let slot = [&ap.subject_var, &ap.predicate_var, &ap.object_var]
                .into_iter()
                .position(|bound| bound.as_deref() == Some(var));
            if let Some(slot) = slot {
                return Some(match triple.ids() {
                    Some(ids) => Ok(ids[slot]),
                    None => {
                        let term = [triple.subject(), triple.predicate(), triple.object()][slot];
                        self.terms.intern(term)
                    }
                });
            }
        }
        None
    }
}

#[cfg(feature = "ring-index")]
impl Operator for RdfLeapfrogOperator {
    fn next(&mut self) -> grafeo_core::execution::operators::OperatorResult {
        self.execute_join();

        if self.position >= self.results.len() {
            return Ok(None);
        }

        let end = (self.position + DEFAULT_CHUNK_SIZE).min(self.results.len());
        let batch_size = end - self.position;
        let col_count = self.output_columns.len();

        let schema = vec![LogicalType::RdfTermId; col_count];
        let mut chunk = DataChunk::with_capacity(&schema, batch_size);

        // The ids of each match; a variable no pattern binds is unbound.
        for matched in &self.results[self.position..end] {
            for (column, var_name) in self.output_columns.iter().enumerate() {
                let value = match self.resolve_variable(var_name, matched) {
                    Some(id) => id_value(id?),
                    None => Value::Null,
                };
                chunk
                    .column_mut(column)
                    .expect("column exists")
                    .push_value(value);
            }
        }

        chunk.set_count(batch_size);
        self.position = end;
        Ok(Some(chunk))
    }

    fn reset(&mut self) {
        self.position = 0;
        // Keep results cached
    }

    fn name(&self) -> &'static str {
        "RdfLeapfrog"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// CONSTRUCT operator: instantiates triple templates from variable bindings
/// (SPARQL 1.1 section 16.2).
///
/// Each solution gives each template's triple, with the variables' exact
/// terms and the template's blank nodes fresh for that solution. A triple
/// with an unbound variable, a literal subject or a predicate that is no IRI
/// is left out, and a triple given before is not repeated (a graph is a
/// set). The output columns `subject`, `predicate` and `object` hold
/// [`Value::RdfTerm`]s.
/// Where a CONSTRUCT template component comes from in each solution.
#[derive(Debug, Clone)]
enum TemplateSlot {
    /// A variable's id column.
    Column(usize),
    /// A constant term's id.
    Constant(TermId),
    /// A template blank node: a fresh one per solution (SPARQL 1.1, 16.2).
    Blank(String),
    /// A variable the pattern does not bind: no triple.
    Unbound,
}

/// Instantiates CONSTRUCT templates per solution, as term ids: a triple is
/// new when its three ids are (one id per term), and only new triples are
/// decoded.
struct ConstructOperator {
    input: Box<dyn Operator>,
    templates: Vec<[TemplateSlot; 3]>,
    terms: Arc<QueryTermIds>,
    seen: grafeo_common::utils::hash::FxHashSet<[TermId; 3]>,
}

impl ConstructOperator {
    /// Whether `ids` make an RDF triple: a subject that is no literal and an
    /// IRI predicate (a template can put anything there).
    fn is_triple(&self, ids: [TermId; 3]) -> bool {
        let is = |id: TermId, check: fn(&Term) -> bool| {
            self.terms
                .entry(id)
                .is_some_and(|entry| check(entry.term()))
        };
        is(ids[0], |term| !term.is_literal()) && is(ids[1], Term::is_iri)
    }
}

impl Operator for ConstructOperator {
    fn next(&mut self) -> grafeo_core::execution::operators::OperatorResult {
        loop {
            let Some(input_chunk) = self.input.next()? else {
                return Ok(None);
            };
            let schema = vec![LogicalType::Any; 3];
            let mut output =
                DataChunk::with_capacity(&schema, input_chunk.row_count() * self.templates.len());
            let mut count = 0;
            for row in input_chunk.selected_indices() {
                let mut blanks: HashMap<&str, TermId> = HashMap::new();
                for template in &self.templates {
                    let mut found = [None; 3];
                    for (slot, id) in template.iter().zip(&mut found) {
                        *id = match slot {
                            TemplateSlot::Column(column) => input_chunk
                                .column(*column)
                                .and_then(|c| c.get_value(row))
                                .as_ref()
                                .and_then(id_of),
                            TemplateSlot::Constant(constant) => Some(*constant),
                            TemplateSlot::Blank(label) => Some(match blanks.get(label.as_str()) {
                                Some(&blank) => blank,
                                None => {
                                    let blank = self.terms.intern(&fresh_blank_node())?;
                                    blanks.insert(label, blank);
                                    blank
                                }
                            }),
                            TemplateSlot::Unbound => None,
                        };
                    }
                    let [Some(subject), Some(predicate), Some(object)] = found else {
                        continue;
                    };
                    let ids = [subject, predicate, object];
                    if !self.is_triple(ids) || !self.seen.insert(ids) {
                        continue;
                    }
                    for (i, id) in ids.into_iter().enumerate() {
                        let value = self.terms.entry(id).map_or(Value::Null, |entry| {
                            Value::RdfTerm(Arc::clone(entry.term()))
                        });
                        output
                            .column_mut(i)
                            .expect("column exists")
                            .push_value(value);
                    }
                    count += 1;
                }
            }
            if count > 0 {
                output.set_count(count);
                return Ok(Some(output));
            }
        }
    }

    fn reset(&mut self) {
        self.input.reset();
        self.seen.clear();
    }

    fn name(&self) -> &'static str {
        "Construct"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// Operator that produces a single pre-computed `DataChunk` and then stops.
///
/// Used for fast-path results such as `COUNT(*)` short-circuits where the
/// answer is known without scanning the data.
struct ConstantOperator {
    chunk: Option<DataChunk>,
}

impl ConstantOperator {
    fn new(chunk: DataChunk) -> Self {
        Self { chunk: Some(chunk) }
    }
}

impl Operator for ConstantOperator {
    fn next(&mut self) -> grafeo_core::execution::operators::OperatorResult {
        Ok(self.chunk.take())
    }

    fn reset(&mut self) {
        // Cannot reset: the chunk was consumed. This is fine for single-use
        // fast-path results.
    }

    fn name(&self) -> &'static str {
        "Constant"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// What a result's term id columns become.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DecodeAs {
    /// The terms ([`Value::RdfTerm`]): SPARQL results.
    Terms,
    /// The values GraphQL reads for them ([`Term::query_value`]).
    QueryValues,
    /// A boolean ([`Value::Bool`]): the one cell of an ASK result.
    Boolean,
}

/// The result boundary of a SPARQL plan: every term id column becomes what
/// [`DecodeAs`] says.
struct TermDecodeOperator {
    input: Box<dyn Operator>,
    terms: Arc<QueryTermIds>,
    /// The term id columns.
    columns: Vec<usize>,
    decode_as: DecodeAs,
}

impl TermDecodeOperator {
    /// What the result holds for the id column value `value`.
    fn decode(&self, value: Option<Value>) -> Value {
        value
            .and_then(|id| self.terms.entry_of(&id))
            .map_or(Value::Null, |entry| match self.decode_as {
                DecodeAs::Terms => Value::RdfTerm(Arc::clone(entry.term())),
                DecodeAs::QueryValues => entry.term().query_value(),
                DecodeAs::Boolean => entry.native().cloned().unwrap_or(Value::Null),
            })
    }
}

impl Operator for TermDecodeOperator {
    fn next(&mut self) -> grafeo_core::execution::operators::OperatorResult {
        let Some(mut chunk) = self.input.next()? else {
            return Ok(None);
        };
        // Every row is kept: the id columns are decoded in place.
        if chunk.selection().is_none() {
            let rows = chunk.total_row_count();
            for &column in &self.columns {
                let Some(source) = chunk.column(column) else {
                    continue;
                };
                let mut decoded = ValueVector::with_capacity(LogicalType::Any, rows);
                for row in 0..rows {
                    decoded.push_value(self.decode(source.get_value(row)));
                }
                if let Some(target) = chunk.column_mut(column) {
                    *target = decoded;
                }
            }
            return Ok(Some(chunk));
        }
        let rows: Vec<usize> = chunk.selected_indices().collect();
        let schema: Vec<LogicalType> = (0..chunk.column_count())
            .map(|i| {
                if self.columns.contains(&i) {
                    LogicalType::Any
                } else {
                    chunk
                        .column(i)
                        .map_or(LogicalType::Any, |c| c.data_type().clone())
                }
            })
            .collect();
        let mut out = DataChunk::with_capacity(&schema, rows.len());
        for col_idx in 0..chunk.column_count() {
            if !self.columns.contains(&col_idx) {
                copy_column(&chunk, col_idx, &rows, &mut out, col_idx);
                continue;
            }
            let Some(source) = chunk.column(col_idx) else {
                continue;
            };
            let target = out.column_mut(col_idx).expect("column exists");
            for &row in &rows {
                target.push_value(self.decode(source.get_value(row)));
            }
        }
        out.set_count(rows.len());
        Ok(Some(out))
    }

    fn reset(&mut self) {
        self.input.reset();
    }

    fn name(&self) -> &'static str {
        "TermDecode"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// Turns value columns into term id columns: a [`Value::RdfTerm`] (a path's
/// endpoint), or a native value (an aggregate's result) as its literal, is
/// interned; NULL stays NULL.
struct TermInternOperator {
    input: Box<dyn Operator>,
    terms: Arc<QueryTermIds>,
    columns: Vec<usize>,
}

impl Operator for TermInternOperator {
    fn next(&mut self) -> grafeo_core::execution::operators::OperatorResult {
        let Some(chunk) = self.input.next()? else {
            return Ok(None);
        };
        let rows: Vec<usize> = chunk.selected_indices().collect();
        let schema: Vec<LogicalType> = (0..chunk.column_count())
            .map(|i| {
                if self.columns.contains(&i) {
                    LogicalType::RdfTermId
                } else {
                    chunk
                        .column(i)
                        .map_or(LogicalType::Any, |c| c.data_type().clone())
                }
            })
            .collect();
        let mut out = DataChunk::with_capacity(&schema, rows.len());
        for col_idx in 0..chunk.column_count() {
            if !self.columns.contains(&col_idx) {
                copy_column(&chunk, col_idx, &rows, &mut out, col_idx);
                continue;
            }
            let Some(source) = chunk.column(col_idx) else {
                continue;
            };
            let ids = rows
                .iter()
                .map(|&row| intern_value(&self.terms, source.get_value(row)))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let target = out.column_mut(col_idx).expect("column exists");
            for id in ids {
                target.push_value(id);
            }
        }
        out.set_count(rows.len());
        Ok(Some(out))
    }

    fn reset(&mut self) {
        self.input.reset();
    }

    /// The name of the operator whose output it interns: the interning is
    /// part of that operator in a profile.
    fn name(&self) -> &'static str {
        self.input.name()
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// Lazy triple scan operator that processes triples in chunks.
///
/// This operator queries the RDF store and emits results in DataChunks
/// for efficient vectorized processing.
struct RdfTripleScanOperator {
    /// The RDF store to scan.
    store: Arc<RdfStore>,
    /// The pattern to match.
    pattern: TriplePattern,
    /// Which components to include in output [s, p, o, g].
    output_mask: [bool; 4],
    /// Graph resolution context (graph IRI, scan-all flag, dataset restriction).
    graph_context: GraphContext,
    /// Chunk size for batching.
    chunk_size: usize,
    /// Cached matching triples with graph names (lazily populated).
    triples: Option<Vec<(Option<String>, Arc<Triple>)>>,
    /// Current position in the triples.
    position: usize,
    /// The query's term dictionary, which the output columns index.
    terms: Arc<QueryTermIds>,
    /// The writer of the scan's transaction, whose own writes it sees.
    writer: Option<RdfWriter>,
    /// Counts the triples the scan reads.
    reads: Option<Arc<grafeo_core::execution::operators::ReadCounter>>,
    /// For each position [s, p, o, g], the earlier position that must hold the same term: a
    /// variable repeated in the pattern (`?x :p ?x`).
    same_as: [Option<usize>; 4],
}

impl RdfTripleScanOperator {
    fn new(
        store: Arc<RdfStore>,
        pattern: TriplePattern,
        output_mask: [bool; 4],
        chunk_size: usize,
        graph_context: GraphContext,
        terms: Arc<QueryTermIds>,
    ) -> Self {
        Self {
            store,
            pattern,
            output_mask,
            graph_context,
            chunk_size,
            triples: None,
            position: 0,
            terms,
            writer: None,
            reads: None,
            same_as: [None; 4],
        }
    }

    /// The same scan of `pattern`, nothing read yet (a lookup per row, see
    /// [`triple_lookup`]).
    fn with_pattern(&self, pattern: TriplePattern) -> Self {
        Self {
            store: Arc::clone(&self.store),
            pattern,
            output_mask: self.output_mask,
            graph_context: self.graph_context.clone(),
            chunk_size: self.chunk_size,
            triples: None,
            position: 0,
            terms: Arc::clone(&self.terms),
            writer: self.writer.clone(),
            reads: self.reads.clone(),
            same_as: self.same_as,
        }
    }

    /// Keeps the rows whose position `i` holds the same term as position `same_as[i]` (see
    /// [`same_as`](Self::same_as)).
    fn with_same_terms(mut self, same_as: [Option<usize>; 4]) -> Self {
        self.same_as = same_as;
        self
    }

    /// Makes the scan see the writes of the transaction of `writer`
    /// (read-your-writes).
    fn with_writer(mut self, writer: Option<RdfWriter>) -> Self {
        self.writer = writer;
        self
    }

    /// Counts the triples the scan reads in `reads`, before any filter.
    fn with_read_counter(
        mut self,
        reads: Arc<grafeo_core::execution::operators::ReadCounter>,
    ) -> Self {
        self.reads = Some(reads);
        self
    }

    /// The triples matching the pattern in the graphs `graphs` names (as
    /// `RdfStore::find_in_graphs` reads them), as the scan's transaction sees
    /// them.
    fn find_in(&self, graphs: Option<&[&str]>) -> Vec<(Option<String>, Arc<Triple>)> {
        match &self.writer {
            Some(writer) => writer.find(&self.pattern, graphs),
            None => self.store.find_in_graphs(&self.pattern, graphs),
        }
    }

    /// Lazily load matching triples on first access.
    ///
    /// Respects SPARQL dataset clauses (FROM / FROM NAMED):
    /// - FROM: basic patterns (no graph context) scan the union of specified named graphs.
    /// - FROM NAMED: GRAPH patterns only iterate listed named graphs.
    fn ensure_triples(&mut self) {
        if self.triples.is_none() {
            let ctx = &self.graph_context;
            let listed = ctx
                .dataset
                .as_ref()
                .and_then(DatasetRestriction::listed_named_graphs);
            self.triples = Some(if ctx.scan_all_graphs {
                // GRAPH ?var: the graphs the dataset lists, or every graph of the store. An empty
                // list to `find_in` means every graph, so listing none is answered here.
                match listed {
                    Some([]) => Vec::new(),
                    Some(names) => {
                        let graph_refs: Vec<&str> = names.iter().map(String::as_str).collect();
                        self.find_in(Some(&graph_refs))
                    }
                    None => self.find_in(Some(&[])),
                }
            } else if let Some(ref graph_iri) = ctx.graph {
                // GRAPH <iri>: a graph the dataset does not list is empty.
                if listed.is_some_and(|names| !names.contains(graph_iri)) {
                    Vec::new()
                } else {
                    self.find_in(Some(&[graph_iri.as_str()]))
                }
            } else {
                // No graph context (basic triple pattern).
                // FROM clauses redefine the default graph as the union of specified graphs.
                if let Some(ref ds) = ctx.dataset {
                    if !ds.default_graphs.is_empty() {
                        // FROM: default graph = union of specified named graphs.
                        // Deduplicate graph IRIs first so listing the same IRI
                        // twice does not produce duplicate triples.
                        let mut unique_graphs: Vec<&str> =
                            ds.default_graphs.iter().map(String::as_str).collect();
                        unique_graphs.sort_unstable();
                        unique_graphs.dedup();
                        let mut results = self.find_in(Some(&unique_graphs));
                        // Clear graph names so results appear as default-graph triples
                        for item in &mut results {
                            item.0 = None;
                        }
                        results
                    } else {
                        // Dataset has FROM NAMED only, no FROM: default graph is empty
                        // per SPARQL spec sec 13.2
                        Vec::new()
                    }
                } else if self.writer.as_ref().is_some_and(RdfWriter::has_pending) {
                    // The transaction wrote triples: read them too.
                    self.find_in(None)
                } else {
                    // No dataset restriction: use actual default graph.
                    // Prefer Ring Index when available (O(log sigma) access).
                    #[cfg(feature = "ring-index")]
                    {
                        if let Some(ring) = self.store.ring() {
                            ring.find(&self.pattern)
                                .map(|t| (None, Arc::new(t)))
                                .collect()
                        } else {
                            self.store
                                .find(&self.pattern)
                                .into_iter()
                                .map(|t| (None, t))
                                .collect()
                        }
                    }
                    #[cfg(not(feature = "ring-index"))]
                    {
                        self.store
                            .find(&self.pattern)
                            .into_iter()
                            .map(|t| (None, t))
                            .collect()
                    }
                }
            });
            if let (Some(reads), Some(triples)) = (&self.reads, &self.triples) {
                reads.add(triples.len());
            }
        }
    }
}

impl Operator for RdfTripleScanOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        self.ensure_triples();

        let triples = self
            .triples
            .as_ref()
            .expect("triples populated by ensure_triples");

        if self.position >= triples.len() {
            return Ok(None);
        }

        let end = (self.position + self.chunk_size).min(triples.len());
        let batch_size = end - self.position;
        let col_count = self.output_mask.iter().filter(|&&b| b).count();
        let schema = vec![LogicalType::RdfTermId; col_count];
        let mut chunk = DataChunk::with_capacity(&schema, batch_size);

        // A stored triple carries its ids; a triple no store took yet (a
        // pending write, the ring's) gets them here. A graph name is an IRI
        // (the same for runs of rows), and a default-graph triple's graph is
        // unbound.
        let mut last_graph: Option<(&str, TermId)> = None;
        let constrained = self.same_as.iter().any(Option::is_some);
        let graph_read = self.output_mask[3] || self.same_as[3].is_some();
        let mut kept = 0;
        for (graph, triple) in &triples[self.position..end] {
            let ids = match triple.ids() {
                Some(ids) => ids,
                None => {
                    let ids = self.terms.intern_all([
                        triple.subject(),
                        triple.predicate(),
                        triple.object(),
                    ])?;
                    [ids[0], ids[1], ids[2]]
                }
            };
            // A graph name's id, the same for runs of rows; a default-graph triple's graph is
            // unbound, and equals no term.
            let graph_id = match (graph_read, graph.as_deref()) {
                (true, Some(name)) => Some(match last_graph {
                    Some((last, id)) if last == name => id,
                    _ => {
                        let id = self.terms.intern(&Term::iri(name))?;
                        last_graph = Some((name, id));
                        id
                    }
                }),
                _ => None,
            };
            if constrained {
                let id_at = |position: usize| match position {
                    3 => graph_id,
                    position => Some(ids[position]),
                };
                let holds = self.same_as.iter().enumerate().all(|(position, first)| {
                    first.is_none_or(|first| id_at(position) == id_at(first))
                });
                if !holds {
                    continue;
                }
            }
            kept += 1;
            let mut col_idx = 0;
            for (component, id) in ids.into_iter().enumerate() {
                if self.output_mask[component] {
                    chunk
                        .column_mut(col_idx)
                        .expect("column exists")
                        .push_value(id_value(id));
                    col_idx += 1;
                }
            }
            if self.output_mask[3] {
                let value = graph_id.map_or(Value::Null, id_value);
                chunk
                    .column_mut(col_idx)
                    .expect("column exists")
                    .push_value(value);
            }
        }

        chunk.set_count(kept);
        self.position = end;

        Ok(Some(chunk))
    }

    fn reset(&mut self) {
        self.position = 0;
        // Keep triples cached for re-execution
    }

    fn name(&self) -> &'static str {
        "RdfTripleScan"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Whether `scan`'s pattern names one variable in two positions (subject, predicate, object or
/// graph).
fn repeats_a_variable(scan: &TripleScanOp) -> bool {
    let names: Vec<&str> = [&scan.subject, &scan.predicate, &scan.object]
        .into_iter()
        .chain(scan.graph.as_ref())
        .filter_map(TripleComponent::as_variable)
        .collect();
    names
        .iter()
        .enumerate()
        .any(|(at, name)| names[..at].contains(name))
}

/// Aligns a UNION branch to the unified columns by name, filling columns the
/// branch does not bind with NULL.
///
/// A branch whose columns already equal the unified domain in order is returned
/// unwrapped (this also covers branch 0 when no later branch adds a variable).
fn pad_union_branch(
    branch: Box<dyn Operator>,
    branch_columns: &[String],
    unified_columns: &[String],
    unified_types: &[LogicalType],
    terms: &Arc<QueryTermIds>,
    context: ExpressionContext,
) -> Box<dyn Operator> {
    if branch_columns == unified_columns {
        return branch;
    }

    let local_index: HashMap<&str, usize> = branch_columns
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), i))
        .collect();

    let projections: Vec<RdfProjectExpr> = unified_columns
        .iter()
        .map(|name| match local_index.get(name.as_str()) {
            Some(&idx) => RdfProjectExpr::Column(idx),
            None => RdfProjectExpr::Constant(Value::Null),
        })
        .collect();

    Box::new(RdfProjectOperator::new(
        branch,
        projections,
        unified_types.to_vec(),
        Arc::clone(terms),
        context,
    ))
}

/// The term of a constant triple component: an IRI, a blank node, a literal
/// as written in the query ([`Value::RdfTerm`]), or a native value as its
/// literal. `None` for a variable.
fn component_to_term(component: &TripleComponent) -> Option<Term> {
    match component {
        TripleComponent::Variable(_) => None,
        TripleComponent::BlankNode(label) => Some(Term::blank(label.clone())),
        TripleComponent::Iri(iri) => Some(Term::iri(iri.clone())),
        TripleComponent::Literal(value) => terms::to_term(value)
            .or_else(|| Some(Term::Literal(Literal::simple(value.to_string())))),
        TripleComponent::LangLiteral { value, lang } => {
            Some(Term::lang_literal(value.clone(), lang.clone()))
        }
    }
}

/// The blank nodes a template's labels stand for in one solution: fresh for
/// each solution (SPARQL 1.1 sections 16.2 and 3.1.1 of Update), the same
/// within it.
#[derive(Default)]
struct FreshBlankNodes {
    labels: HashMap<String, Term>,
}

impl FreshBlankNodes {
    fn get(&mut self, label: &str) -> Term {
        self.labels
            .entry(label.to_string())
            .or_insert_with(fresh_blank_node)
            .clone()
    }
}

/// A blank node no other has: a label of this process's run and a counter.
fn fresh_blank_node() -> Term {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, Ordering};
    static RUN: OnceLock<u64> = OnceLock::new();
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let run = RUN.get_or_init(|| {
        use std::hash::{BuildHasher, Hasher};
        let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
        hasher.write_u64(0x6772_6166_656f);
        hasher.finish()
    });
    Term::blank(format!(
        "g{run:016x}{:x}",
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}

/// The term of a template component in a solution: a variable's bound term
/// (`None` when unbound), a blank node fresh for the solution, a constant as
/// written.
fn template_term(
    component: &TripleComponent,
    columns: &HashMap<String, usize>,
    chunk: &DataChunk,
    row: usize,
    terms: &QueryTermIds,
    blanks: &mut FreshBlankNodes,
) -> Option<Term> {
    match component {
        TripleComponent::Variable(name) => {
            let name = name.strip_prefix('?').unwrap_or(name);
            let id = chunk.column(*columns.get(name)?)?.get_value(row)?;
            terms
                .entry_of(&id)
                .map(|entry| entry.term().as_ref().clone())
        }
        TripleComponent::BlankNode(label) => Some(blanks.get(label)),
        other => component_to_term(other),
    }
}

/// Whether a template's subject and predicate make an RDF triple: no
/// literal subject, an IRI predicate (SPARQL 1.1 section 16.2).
fn is_valid_triple(subject: &Term, predicate: &Term) -> bool {
    !matches!(subject, Term::Literal(_)) && matches!(predicate, Term::Iri(_))
}

/// The column index of each name.
fn column_map(columns: &[String]) -> HashMap<String, usize> {
    columns
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), i))
        .collect()
}

/// Quick cardinality estimate for a logical operator subtree.
///
/// Uses store index sizes to estimate how many rows an operator produces,
/// without collecting full statistics. Returns `None` if estimation is
/// not possible for this operator type.
fn estimate_operator_cardinality(
    op: &crate::query::plan::LogicalOperator,
    store: &RdfStore,
) -> Option<f64> {
    use crate::query::plan::LogicalOperator;
    match op {
        LogicalOperator::RowValues(_) => Some(1.0),
        LogicalOperator::TripleScan(scan) => {
            let stats = store.stats();
            let total = stats.triple_count as f64;
            if total == 0.0 {
                return Some(0.0);
            }

            // Estimate based on which components are bound
            let s_bound = matches!(
                scan.subject,
                crate::query::plan::TripleComponent::Iri(_)
                    | crate::query::plan::TripleComponent::Literal(_)
                    | crate::query::plan::TripleComponent::LangLiteral { .. }
            );
            let p_bound = matches!(
                scan.predicate,
                crate::query::plan::TripleComponent::Iri(_)
                    | crate::query::plan::TripleComponent::Literal(_)
                    | crate::query::plan::TripleComponent::LangLiteral { .. }
            );
            let o_bound = matches!(
                scan.object,
                crate::query::plan::TripleComponent::Iri(_)
                    | crate::query::plan::TripleComponent::Literal(_)
                    | crate::query::plan::TripleComponent::LangLiteral { .. }
            );

            // With a predicate, its own counts from the optimizer's
            // statistics (cached, see `RdfStore::get_or_collect_statistics`).
            if let crate::query::plan::TripleComponent::Iri(predicate) = &scan.predicate
                && let Some(counted) = store
                    .get_or_collect_statistics()
                    .get_predicate(&format!("<{predicate}>"))
            {
                let triples = counted.triple_count as f64;
                let estimate = match (s_bound, o_bound) {
                    (true, true) => 1.0,
                    (true, false) => triples / counted.distinct_subjects.max(1) as f64,
                    (false, true) => triples / counted.distinct_objects.max(1) as f64,
                    (false, false) => triples,
                };
                return Some(estimate.max(1.0));
            }
            let estimate = match (s_bound, p_bound, o_bound) {
                (true, true, true) => 1.0,
                (true, true, false) => total / stats.subject_count.max(1) as f64,
                (true, false, true) => total / stats.subject_count.max(1) as f64,
                (false, true, true) => total / stats.predicate_count.max(1) as f64,
                (true, false, false) => total / stats.subject_count.max(1) as f64,
                (false, true, false) => total / stats.predicate_count.max(1) as f64,
                (false, false, true) => total / stats.object_count.max(1) as f64,
                (false, false, false) => total,
            };
            Some(estimate.max(1.0))
        }
        LogicalOperator::Filter(f) => {
            estimate_operator_cardinality(&f.input, store).map(|c| (c * 0.33).max(1.0))
        }
        LogicalOperator::Join(j) => {
            let left = estimate_operator_cardinality(&j.left, store)?;
            let right = estimate_operator_cardinality(&j.right, store)?;
            Some((left * right * 0.1).max(1.0))
        }
        LogicalOperator::Limit(l) => {
            if let crate::query::plan::CountExpr::Literal(n) = l.count {
                Some(n as f64)
            } else {
                estimate_operator_cardinality(&l.input, store)
            }
        }
        // A VALUES block is a union of one row each (see the translator).
        LogicalOperator::Empty => Some(1.0),
        LogicalOperator::Bind(bind) => estimate_operator_cardinality(&bind.input, store),
        LogicalOperator::Union(union) => union
            .inputs
            .iter()
            .map(|input| estimate_operator_cardinality(input, store))
            .sum::<Option<f64>>(),
        _ => None,
    }
}

/// How many rows of `other` look their triples up in patterns estimated to
/// match `triples` each (see [`triple_lookup::lookup_limit`]); `None` when
/// `other` is known while planning to hold more, which the hash join takes
/// without reading the first rows to find out. Only a scan's or a VALUES
/// block's estimate counts: a join's (left x right / 10) overstates a star of
/// patterns on one subject, which the switch at run time handles.
fn lookup_limit_against(
    other: &LogicalOperator,
    triples: Option<f64>,
    other_estimate: Option<f64>,
) -> Option<usize> {
    let other_estimate = other_estimate.filter(|_| {
        matches!(
            other,
            LogicalOperator::TripleScan(_)
                | LogicalOperator::Union(_)
                | LogicalOperator::Bind(_)
                | LogicalOperator::Empty
        )
    });
    let limit = triple_lookup::lookup_limit(triples);
    #[expect(
        clippy::cast_precision_loss,
        reason = "the limit is at most LOOKUP_ROWS, which an f64 holds exactly"
    )]
    let limit_rows = limit as f64;
    (!other_estimate.is_some_and(|rows| rows > limit_rows)).then_some(limit)
}

/// A planned operator: the operator, its columns and their types.
type PlannedOperator = (Box<dyn Operator>, Vec<String>, Vec<LogicalType>);

/// The lookup of a UNION's patterns (see `RdfPlanner::union_lookup`): the
/// patterns, the names of the columns the join adds, the limit of rows.
type UnionLookupPlan = (Vec<triple_lookup::PatternLookup>, Vec<String>, usize);

/// The variables of `scan`'s subject, predicate and object, with their
/// positions (0, 1 and 2), in that order: the columns of its scan, without a
/// graph variable.
fn pattern_variables(scan: &TripleScanOp) -> Vec<(usize, &str)> {
    [&scan.subject, &scan.predicate, &scan.object]
        .into_iter()
        .enumerate()
        .filter_map(|(position, component)| match component {
            TripleComponent::Variable(name) => Some((position, name.as_str())),
            _ => None,
        })
        .collect()
}

/// The triple pattern a UNION branch is: the branch itself, or the pattern
/// under a projection that keeps each of its variables as it is (the
/// translator projects an inverse branch of an alternative path, `^p`, on
/// its two ends). `None` for any other branch.
fn union_branch_pattern(branch: &LogicalOperator) -> Option<&LogicalOperator> {
    match branch {
        LogicalOperator::TripleScan(scan) if scan.input.is_none() => Some(branch),
        LogicalOperator::Project(project) if !project.pass_through_input => {
            let pattern = project.input.as_ref();
            let LogicalOperator::TripleScan(scan) = pattern else {
                return None;
            };
            let mut variables: Vec<&str> = pattern_variables(scan)
                .into_iter()
                .map(|(_, name)| name)
                .chain(scan.graph.as_ref().and_then(TripleComponent::as_variable))
                .collect();
            let mut kept = project
                .projections
                .iter()
                .map(
                    |projection| match (&projection.expression, &projection.alias) {
                        (LogicalExpression::Variable(name), None) => Some(name.as_str()),
                        (LogicalExpression::Variable(name), Some(alias)) if alias == name => {
                            Some(name.as_str())
                        }
                        _ => None,
                    },
                )
                .collect::<Option<Vec<&str>>>()?;
            variables.sort_unstable();
            kept.sort_unstable();
            (variables == kept).then_some(pattern)
        }
        _ => None,
    }
}

/// Resolves an expression to a column index.
fn resolve_expression(
    expr: &LogicalExpression,
    variable_columns: &HashMap<String, usize>,
) -> Result<usize> {
    crate::query::planner::common::resolve_expression_to_column(expr, variable_columns, "")
}

// expression_to_string is now in planner/common.rs
use crate::query::planner::common::{
    expression_to_string, output_column_name, resolved_column_name,
};

// ============================================================================
// Tests
// ============================================================================

/// The shared columns of a join that a side may leave unbound: those not in
/// both sides' [`always_bound`](crate::query::plan::always_bound) sets. An
/// operator the analysis does not know counts as binding all its columns.
fn loosely_shared(
    left: Option<std::collections::HashSet<String>>,
    right: Option<std::collections::HashSet<String>>,
    left_columns: &[String],
    right_columns: &[String],
) -> Vec<String> {
    let bound = |side: &Option<std::collections::HashSet<String>>, name: &String| {
        side.as_ref().is_none_or(|bound| bound.contains(name))
    };
    left_columns
        .iter()
        .filter(|name| right_columns.contains(name))
        .filter(|name| !(bound(&left, name) && bound(&right, name)))
        .cloned()
        .collect()
}

/// The conjuncts of `expression`, its `AND`s taken apart.
fn conjuncts_of<'a>(expression: &'a LogicalExpression, into: &mut Vec<&'a LogicalExpression>) {
    match expression {
        LogicalExpression::Binary {
            left,
            op: crate::query::plan::BinaryOp::And,
            right,
        } => {
            conjuncts_of(left, into);
            conjuncts_of(right, into);
        }
        other => into.push(other),
    }
}

/// The position in `inputs` (their columns, smallest estimate first) of the
/// next input of a multi-way join over the rows of `joined`: the smallest one
/// that shares a variable with them, or the smallest when none does. Taking
/// the smallest whatever it shares would cross small inputs that are not
/// connected yet: a 7-pattern join crossed its 20 tags, 1,000 posts and 1,001
/// people first.
fn next_join_input<'a>(mut inputs: impl Iterator<Item = &'a [String]>, joined: &[String]) -> usize {
    inputs
        .position(|columns| columns.iter().any(|column| joined.contains(column)))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A simple literal as a result holds it.
    fn literal(text: &str) -> Value {
        Value::RdfTerm(Arc::new(Term::literal(text)))
    }

    /// An `xsd:integer` as a result holds it.
    fn integer(n: i64) -> Value {
        Value::RdfTerm(Arc::new(Term::Literal(Literal::integer(n))))
    }
    use crate::query::plan::{CountExpr, JoinType, LogicalPlan};

    #[test]
    fn test_rdf_planner_simple_scan() {
        let store = Arc::new(RdfStore::new());

        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));

        let planner = RdfPlanner::new(store);

        let scan = TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Variable("p".to_string()),
            object: TripleComponent::Variable("o".to_string()),
            graph: None,
            input: None,
            dataset: None,
        };

        let plan = LogicalPlan::new(LogicalOperator::TripleScan(scan));
        let physical = planner.plan(&plan).unwrap();

        assert_eq!(physical.columns, vec!["s", "p", "o"]);
    }

    #[test]
    fn test_rdf_planner_with_pattern() {
        let store = Arc::new(RdfStore::new());

        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));
        store.insert(Triple::new(
            Term::iri("http://example.org/gus"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Gus"),
        ));
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/age"),
            Term::typed_literal("30", "http://www.w3.org/2001/XMLSchema#integer"),
        ));

        let planner = RdfPlanner::new(store);

        let scan = TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Variable("o".to_string()),
            graph: None,
            input: None,
            dataset: None,
        };

        let plan = LogicalPlan::new(LogicalOperator::TripleScan(scan));
        let physical = planner.plan(&plan).unwrap();

        // Only s and o are variables (predicate is fixed)
        assert_eq!(physical.columns, vec!["s", "o"]);
    }

    #[test]
    fn test_rdf_scan_operator_chunking() {
        let store = Arc::new(RdfStore::new());

        // Insert 100 triples
        for i in 0..100 {
            store.insert(Triple::new(
                Term::iri(format!("http://example.org/item{}", i)),
                Term::iri("http://example.org/value"),
                Term::literal(i.to_string()),
            ));
        }

        let pattern = TriplePattern {
            subject: None,
            predicate: None,
            object: None,
        };

        let mut operator = RdfTripleScanOperator::new(
            Arc::clone(&store),
            pattern,
            [true, true, true, false],
            30,
            GraphContext {
                graph: None,
                scan_all_graphs: false,
                dataset: None,
            },
            Arc::default(),
        );

        let mut total_rows = 0;
        while let Ok(Some(chunk)) = operator.next() {
            total_rows += chunk.row_count();
            assert!(chunk.row_count() <= 30); // Respects chunk size
        }

        assert_eq!(total_rows, 100);
    }

    #[test]
    fn test_copy_graph_operator() {
        let store = Arc::new(RdfStore::new());

        // Insert triples into a named graph
        store.create_graph("http://example.org/src");
        let src = store.graph("http://example.org/src").unwrap();
        src.insert(Triple::new(
            Term::iri("http://example.org/a"),
            Term::iri("http://example.org/p"),
            Term::literal("val"),
        ));
        assert_eq!(src.len(), 1);

        // Copy src -> dst via operator
        let plan = LogicalPlan::new(LogicalOperator::CopyGraph(CopyGraphOp {
            source: Some("http://example.org/src".to_string()),
            destination: Some("http://example.org/dst".to_string()),
            silent: false,
        }));
        execute_update(&store, &plan).unwrap();

        // Source still has its data
        assert_eq!(store.graph("http://example.org/src").unwrap().len(), 1);
        // Destination has a copy
        assert_eq!(store.graph("http://example.org/dst").unwrap().len(), 1);
    }

    #[test]
    fn test_move_graph_operator() {
        let store = Arc::new(RdfStore::new());

        store.create_graph("http://example.org/src");
        let src = store.graph("http://example.org/src").unwrap();
        src.insert(Triple::new(
            Term::iri("http://example.org/a"),
            Term::iri("http://example.org/p"),
            Term::literal("val"),
        ));

        // Move src -> dst via operator
        let plan = LogicalPlan::new(LogicalOperator::MoveGraph(MoveGraphOp {
            source: Some("http://example.org/src".to_string()),
            destination: Some("http://example.org/dst".to_string()),
            silent: false,
        }));
        execute_update(&store, &plan).unwrap();

        // Source is gone (move drops it)
        assert!(store.graph("http://example.org/src").is_none());
        // Destination has the data
        assert_eq!(store.graph("http://example.org/dst").unwrap().len(), 1);
    }

    #[test]
    fn test_add_graph_operator() {
        let store = Arc::new(RdfStore::new());

        // Create src with 1 triple
        store.create_graph("http://example.org/src");
        store
            .graph("http://example.org/src")
            .unwrap()
            .insert(Triple::new(
                Term::iri("http://example.org/a"),
                Term::iri("http://example.org/p"),
                Term::literal("from-src"),
            ));

        // Create dst with 1 different triple
        store.create_graph("http://example.org/dst");
        store
            .graph("http://example.org/dst")
            .unwrap()
            .insert(Triple::new(
                Term::iri("http://example.org/b"),
                Term::iri("http://example.org/q"),
                Term::literal("from-dst"),
            ));

        // Add src -> dst via operator (merges)
        let plan = LogicalPlan::new(LogicalOperator::AddGraph(AddGraphOp {
            source: Some("http://example.org/src".to_string()),
            destination: Some("http://example.org/dst".to_string()),
            silent: false,
        }));
        execute_update(&store, &plan).unwrap();

        // Source unchanged
        assert_eq!(store.graph("http://example.org/src").unwrap().len(), 1);
        // Destination has both triples (union)
        assert_eq!(store.graph("http://example.org/dst").unwrap().len(), 2);
    }

    #[test]
    fn test_copy_nonexistent_source_errors_without_silent() {
        let store = Arc::new(RdfStore::new());

        let plan = LogicalPlan::new(LogicalOperator::CopyGraph(CopyGraphOp {
            source: Some("http://example.org/nope".to_string()),
            destination: Some("http://example.org/dst".to_string()),
            silent: false,
        }));
        // Should error because source doesn't exist
        let error = execute_update(&store, &plan).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Source graph <http://example.org/nope> does not exist"),
            "{error}"
        );
        assert!(store.graph("http://example.org/dst").is_none());
    }

    #[test]
    fn test_copy_nonexistent_source_silent_ok() {
        let store = Arc::new(RdfStore::new());

        let plan = LogicalPlan::new(LogicalOperator::CopyGraph(CopyGraphOp {
            source: Some("http://example.org/nope".to_string()),
            destination: Some("http://example.org/dst".to_string()),
            silent: true,
        }));
        assert!(execute_update(&store, &plan).is_ok());
    }

    /// Triple scan propagates LogicalType::String for the subject, predicate
    /// and companion columns, and Any for the object, where a literal may be a
    /// number of its XSD type.
    #[test]
    fn test_type_propagation_triple_scan() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));

        let planner = RdfPlanner::new(store);
        let scan = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Variable("p".to_string()),
            object: TripleComponent::Variable("o".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });

        let (_op, columns, types) = planner.plan_operator(&scan).unwrap();
        // Every variable column holds term ids, with no companion columns.
        assert_eq!(columns, vec!["s", "p", "o"]);
        for (i, ty) in types.iter().enumerate() {
            assert_eq!(*ty, LogicalType::RdfTermId, "column {i} ({})", columns[i]);
        }
    }

    /// Join of two triple scans propagates the types of both sides.
    #[test]
    fn test_type_propagation_join() {
        use crate::query::plan::JoinOp;

        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/age"),
            Term::typed_literal("30", "http://www.w3.org/2001/XMLSchema#integer"),
        ));

        let planner = RdfPlanner::new(store);

        let left = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Variable("name".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let right = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/age".to_string()),
            object: TripleComponent::Variable("age".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });

        let join = LogicalOperator::Join(JoinOp {
            left: Box::new(left),
            right: Box::new(right),
            join_type: JoinType::Inner,
            conditions: vec![],
        });

        let (_op, columns, types) = planner.plan_operator(&join).unwrap();
        assert_eq!(columns.len(), types.len());
        for (i, ty) in types.iter().enumerate() {
            assert_eq!(
                *ty,
                LogicalType::RdfTermId,
                "join column {i} ({})",
                columns[i]
            );
        }
    }

    /// BIND appends an Any-typed column for the computed expression.
    #[test]
    fn test_type_propagation_bind() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));

        let planner = RdfPlanner::new(store);

        let scan = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Variable("name".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });

        let bind = LogicalOperator::Bind(BindOp {
            input: Box::new(scan),
            variable: "label".to_string(),
            expression: crate::query::plan::LogicalExpression::Variable("name".to_string()),
        });

        let (_op, columns, types) = planner.plan_operator(&bind).unwrap();
        // The BIND variable comes last, a term id column like the others.
        assert_eq!(columns.last().map(String::as_str), Some("label"));
        assert!(
            types.iter().all(|ty| *ty == LogicalType::RdfTermId),
            "{types:?}"
        );
    }

    /// Aggregate output has concrete types: group-by columns are String,
    /// COUNT is Int64, SUM/AVG are Float64.
    #[test]
    fn test_type_propagation_aggregate() {
        use crate::query::plan::{AggregateExpr, AggregateFunction, AggregateOp};

        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));

        let planner = RdfPlanner::new(store);

        let scan = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Variable("name".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });

        let agg = LogicalOperator::Aggregate(AggregateOp {
            input: Box::new(scan),
            group_by: vec![crate::query::plan::LogicalExpression::Variable(
                "name".to_string(),
            )],
            aggregates: vec![AggregateExpr {
                function: AggregateFunction::Count,
                expression: None,
                expression2: None,
                distinct: false,
                alias: Some("cnt".to_string()),
                percentile: None,
                separator: None,
            }],
            having: None,
        });

        let (_op, columns, types) = planner.plan_operator(&agg).unwrap();
        assert_eq!(columns, vec!["name", "cnt"]);
        assert_eq!(
            types[0],
            LogicalType::RdfTermId,
            "the group key is a term id"
        );
        assert_eq!(types[1], LogicalType::RdfTermId, "COUNT is interned");
    }

    /// Filter, Distinct, Limit, Skip all preserve input types.
    #[test]
    fn test_type_propagation_passthrough_operators() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));

        let planner = RdfPlanner::new(store);

        let scan = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Variable("name".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });

        // Get baseline types from the scan
        let (_op, _cols, scan_types) = planner.plan_operator(&scan).unwrap();

        // Wrap in Limit
        let limited = LogicalOperator::Limit(LimitOp {
            input: Box::new(scan.clone()),
            count: CountExpr::Literal(10),
        });
        let (_op, _cols, limit_types) = planner.plan_operator(&limited).unwrap();
        assert_eq!(scan_types, limit_types, "Limit preserves types");

        // Wrap in Distinct
        let distinct = LogicalOperator::Distinct(DistinctOp {
            input: Box::new(scan.clone()),
            columns: None,
        });
        let (_op, _cols, distinct_types) = planner.plan_operator(&distinct).unwrap();
        assert_eq!(scan_types, distinct_types, "Distinct preserves types");

        // Wrap in Skip
        let skipped = LogicalOperator::Skip(SkipOp {
            input: Box::new(scan),
            count: CountExpr::Literal(1),
        });
        let (_op, _cols, skip_types) = planner.plan_operator(&skipped).unwrap();
        assert_eq!(scan_types, skip_types, "Skip preserves types");
    }

    #[test]
    fn test_plan_filter_with_comparison() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/age"),
            Term::typed_literal("30", "http://www.w3.org/2001/XMLSchema#integer"),
        ));
        store.insert(Triple::new(
            Term::iri("http://example.org/gus"),
            Term::iri("http://xmlns.com/foaf/0.1/age"),
            Term::typed_literal("20", "http://www.w3.org/2001/XMLSchema#integer"),
        ));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let scan = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/age".to_string()),
            object: TripleComponent::Variable("age".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let filter = LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Variable("age".to_string())),
                op: crate::query::plan::BinaryOp::Gt,
                right: Box::new(LogicalExpression::Literal(Value::Int64(25))),
            },
            input: Box::new(scan),
            pushdown_hint: None,
        });
        let physical = planner.plan(&LogicalPlan::new(filter)).unwrap();
        assert_eq!(physical.columns, vec!["s", "age"]);
        let mut op = physical.operator;
        let mut rows = 0;
        while let Ok(Some(c)) = op.next() {
            rows += c.row_count();
        }
        assert_eq!(rows, 1);
    }

    #[test]
    fn test_plan_aggregate_count_star_fast_path() {
        use crate::query::plan::{AggregateExpr, AggregateFunction, AggregateOp};
        let store = Arc::new(RdfStore::new());
        for i in 0..10 {
            store.insert(Triple::new(
                Term::iri(format!("http://example.org/item{i}")),
                Term::iri("http://example.org/value"),
                Term::literal(i.to_string()),
            ));
        }
        let planner = RdfPlanner::new(Arc::clone(&store));
        let scan = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Variable("p".to_string()),
            object: TripleComponent::Variable("o".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let agg = LogicalOperator::Aggregate(AggregateOp {
            input: Box::new(scan),
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
            having: None,
        });
        let physical = planner.plan(&LogicalPlan::new(agg)).unwrap();
        let mut op = physical.operator;
        let chunk = op.next().unwrap().unwrap();
        assert_eq!(chunk.column(0).unwrap().get_value(0), Some(integer(10)));
    }

    #[test]
    fn test_plan_aggregate_with_having() {
        use crate::query::plan::{AggregateExpr, AggregateFunction, AggregateOp};
        let store = Arc::new(RdfStore::new());
        for i in 0..3 {
            store.insert(Triple::new(
                Term::iri("http://example.org/alix"),
                Term::iri(format!("http://example.org/p{i}")),
                Term::literal(format!("val{i}")),
            ));
        }
        store.insert(Triple::new(
            Term::iri("http://example.org/gus"),
            Term::iri("http://example.org/p0"),
            Term::literal("gus_val"),
        ));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let scan = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Variable("p".to_string()),
            object: TripleComponent::Variable("o".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let agg = LogicalOperator::Aggregate(AggregateOp {
            input: Box::new(scan),
            group_by: vec![LogicalExpression::Variable("s".to_string())],
            aggregates: vec![AggregateExpr {
                function: AggregateFunction::Count,
                expression: None,
                expression2: None,
                distinct: false,
                alias: Some("cnt".to_string()),
                percentile: None,
                separator: None,
            }],
            having: Some(LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Variable("cnt".to_string())),
                op: crate::query::plan::BinaryOp::Gt,
                right: Box::new(LogicalExpression::Literal(Value::Int64(1))),
            }),
        });
        let physical = planner.plan(&LogicalPlan::new(agg)).unwrap();
        let mut op = physical.operator;
        let mut rows = 0;
        while let Ok(Some(c)) = op.next() {
            rows += c.row_count();
        }
        assert_eq!(rows, 1);
    }

    #[test]
    fn test_plan_aggregate_sum_avg() {
        use crate::query::plan::{AggregateExpr, AggregateFunction, AggregateOp};
        let store = Arc::new(RdfStore::new());
        for i in 1..=4 {
            store.insert(Triple::new(
                Term::iri("http://example.org/a"),
                Term::iri("http://example.org/val"),
                Term::typed_literal(i.to_string(), "http://www.w3.org/2001/XMLSchema#integer"),
            ));
        }
        let planner = RdfPlanner::new(Arc::clone(&store));
        let scan = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://example.org/val".to_string()),
            object: TripleComponent::Variable("v".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let agg = LogicalOperator::Aggregate(AggregateOp {
            input: Box::new(scan),
            group_by: vec![],
            aggregates: vec![
                AggregateExpr {
                    function: AggregateFunction::Sum,
                    expression: Some(LogicalExpression::Variable("v".to_string())),
                    expression2: None,
                    distinct: false,
                    alias: Some("total".to_string()),
                    percentile: None,
                    separator: None,
                },
                AggregateExpr {
                    function: AggregateFunction::Avg,
                    expression: Some(LogicalExpression::Variable("v".to_string())),
                    expression2: None,
                    distinct: false,
                    alias: Some("average".to_string()),
                    percentile: None,
                    separator: None,
                },
            ],
            having: None,
        });
        let physical = planner.plan(&LogicalPlan::new(agg)).unwrap();
        assert_eq!(physical.columns, vec!["total", "average"]);
    }

    #[test]
    fn test_plan_join_execution() {
        use crate::query::plan::JoinOp;
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/age"),
            Term::typed_literal("30", "http://www.w3.org/2001/XMLSchema#integer"),
        ));
        store.insert(Triple::new(
            Term::iri("http://example.org/gus"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Gus"),
        ));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let left = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Variable("name".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let right = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/age".to_string()),
            object: TripleComponent::Variable("age".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let join = LogicalOperator::Join(JoinOp {
            left: Box::new(left),
            right: Box::new(right),
            join_type: JoinType::Inner,
            conditions: vec![],
        });
        let physical = planner.plan(&LogicalPlan::new(join)).unwrap();
        let mut op = physical.operator;
        let mut rows = 0;
        while let Ok(Some(c)) = op.next() {
            rows += c.row_count();
        }
        assert_eq!(rows, 1);
    }

    #[test]
    fn test_plan_left_join() {
        use crate::query::plan::LeftJoinOp;
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));
        store.insert(Triple::new(
            Term::iri("http://example.org/gus"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Gus"),
        ));
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/age"),
            Term::typed_literal("30", "http://www.w3.org/2001/XMLSchema#integer"),
        ));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let left = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Variable("name".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let right = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/age".to_string()),
            object: TripleComponent::Variable("age".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let join = LogicalOperator::LeftJoin(LeftJoinOp {
            left: Box::new(left),
            right: Box::new(right),
            condition: None,
        });
        let physical = planner.plan(&LogicalPlan::new(join)).unwrap();
        let mut op = physical.operator;
        let mut rows = 0;
        while let Ok(Some(c)) = op.next() {
            rows += c.row_count();
        }
        assert_eq!(rows, 2);
    }

    #[test]
    fn test_plan_anti_join() {
        use crate::query::plan::AntiJoinOp;
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));
        store.insert(Triple::new(
            Term::iri("http://example.org/gus"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Gus"),
        ));
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/age"),
            Term::typed_literal("30", "http://www.w3.org/2001/XMLSchema#integer"),
        ));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let left = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Variable("name".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let right = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/age".to_string()),
            object: TripleComponent::Variable("age".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let anti = LogicalOperator::AntiJoin(AntiJoinOp {
            left: Box::new(left),
            right: Box::new(right),
        });
        let physical = planner.plan(&LogicalPlan::new(anti)).unwrap();
        let mut op = physical.operator;
        let mut rows = 0;
        while let Ok(Some(c)) = op.next() {
            rows += c.row_count();
        }
        assert_eq!(rows, 1);
    }

    #[test]
    fn test_plan_sort() {
        use crate::query::plan::{SortKey, SortOp, SortOrder};
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/c"),
            Term::iri("http://example.org/val"),
            Term::literal("3"),
        ));
        store.insert(Triple::new(
            Term::iri("http://example.org/a"),
            Term::iri("http://example.org/val"),
            Term::literal("1"),
        ));
        store.insert(Triple::new(
            Term::iri("http://example.org/b"),
            Term::iri("http://example.org/val"),
            Term::literal("2"),
        ));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let scan = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://example.org/val".to_string()),
            object: TripleComponent::Variable("v".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let sort = LogicalOperator::Sort(SortOp {
            keys: vec![SortKey {
                expression: LogicalExpression::Variable("v".to_string()),
                order: SortOrder::Ascending,
                nulls: None,
            }],
            input: Box::new(scan),
        });
        let physical = planner.plan(&LogicalPlan::new(sort)).unwrap();
        assert_eq!(physical.columns, vec!["s", "v"]);
        let mut op = physical.operator;
        let mut vals = Vec::new();
        while let Ok(Some(chunk)) = op.next() {
            if let Some(col) = chunk.column(1) {
                for row in 0..chunk.row_count() {
                    if let Some(v) = col.get_value(row) {
                        vals.push(v);
                    }
                }
            }
        }
        assert_eq!(vals, vec![literal("1"), literal("2"), literal("3")]);
    }

    #[test]
    fn test_plan_sort_honors_explicit_null_order() {
        use crate::query::plan::{LeftJoinOp, NullsOrdering, SortKey, SortOp, SortOrder};
        let store = Arc::new(RdfStore::new());
        for (person, name, age) in [
            ("alix", "Alix", Some("30")),
            ("gus", "Gus", None),
            ("vincent", "Vincent", Some("25")),
        ] {
            let subject = Term::iri(format!("http://example.org/{person}"));
            store.insert(Triple::new(
                subject.clone(),
                Term::iri("http://xmlns.com/foaf/0.1/name"),
                Term::literal(name),
            ));
            if let Some(age) = age {
                store.insert(Triple::new(
                    subject,
                    Term::iri("http://xmlns.com/foaf/0.1/age"),
                    Term::literal(age),
                ));
            }
        }
        let scan = |predicate: &str, object: &str| {
            LogicalOperator::TripleScan(TripleScanOp {
                subject: TripleComponent::Variable("s".to_string()),
                predicate: TripleComponent::Iri(format!("http://xmlns.com/foaf/0.1/{predicate}")),
                object: TripleComponent::Variable(object.to_string()),
                graph: None,
                input: None,
                dataset: None,
            })
        };
        // The ages in sort order, `None` for the person without one.
        let ages = |order: SortOrder, nulls: NullsOrdering| {
            let sort = LogicalOperator::Sort(SortOp {
                keys: vec![SortKey {
                    expression: LogicalExpression::Variable("age".to_string()),
                    order,
                    nulls: Some(nulls),
                }],
                input: Box::new(LogicalOperator::LeftJoin(LeftJoinOp {
                    left: Box::new(scan("name", "name")),
                    right: Box::new(scan("age", "age")),
                    condition: None,
                })),
            });
            let physical = RdfPlanner::new(Arc::clone(&store))
                .plan(&LogicalPlan::new(sort))
                .unwrap();
            let column = physical.columns.iter().position(|c| c == "age").unwrap();
            let mut op = physical.operator;
            let mut ages = Vec::new();
            while let Some(chunk) = op.next().unwrap() {
                let values = chunk.column(column).unwrap();
                for row in chunk.selected_indices() {
                    ages.push(match values.get_value(row) {
                        Some(Value::RdfTerm(age)) => {
                            age.as_literal().map(|l| l.value().to_string())
                        }
                        _ => None,
                    });
                }
            }
            ages
        };
        let some = |age: &str| Some(age.to_string());

        assert_eq!(
            ages(SortOrder::Ascending, NullsOrdering::First),
            vec![None, some("25"), some("30")]
        );
        assert_eq!(
            ages(SortOrder::Descending, NullsOrdering::Last),
            vec![some("30"), some("25"), None]
        );
    }

    #[test]
    fn test_plan_insert_triple_concrete() {
        let store = Arc::new(RdfStore::new());
        let insert = LogicalOperator::InsertTriple(InsertTripleOp {
            subject: TripleComponent::Iri("http://example.org/alix".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Literal(Value::String("Alix".into())),
            graph: None,
            input: None,
        });
        execute_update(&store, &LogicalPlan::new(insert)).unwrap();
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_plan_insert_triple_into_named_graph() {
        let store = Arc::new(RdfStore::new());
        let insert = LogicalOperator::InsertTriple(InsertTripleOp {
            subject: TripleComponent::Iri("http://example.org/alix".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Literal(Value::String("Alix".into())),
            graph: Some("http://example.org/g1".to_string()),
            input: None,
        });
        execute_update(&store, &LogicalPlan::new(insert)).unwrap();
        assert_eq!(store.graph("http://example.org/g1").unwrap().len(), 1);
    }

    #[test]
    fn test_plan_delete_triple_concrete() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));
        let delete = LogicalOperator::DeleteTriple(DeleteTripleOp {
            subject: TripleComponent::Iri("http://example.org/alix".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Literal(Value::String("Alix".into())),
            graph: None,
            input: None,
        });
        execute_update(&store, &LogicalPlan::new(delete)).unwrap();
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_plan_insert_pattern_from_where() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));
        let where_scan = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Variable("name".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let insert = LogicalOperator::InsertTriple(InsertTripleOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://example.org/label".to_string()),
            object: TripleComponent::Variable("name".to_string()),
            graph: None,
            input: Some(Box::new(where_scan)),
        });
        execute_update(&store, &LogicalPlan::new(insert)).unwrap();
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_plan_delete_pattern_from_where() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));
        store.insert(Triple::new(
            Term::iri("http://example.org/gus"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Gus"),
        ));
        let where_scan = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Variable("p".to_string()),
            object: TripleComponent::Variable("o".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let delete = LogicalOperator::DeleteTriple(DeleteTripleOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Variable("p".to_string()),
            object: TripleComponent::Variable("o".to_string()),
            graph: None,
            input: Some(Box::new(where_scan)),
        });
        execute_update(&store, &LogicalPlan::new(delete)).unwrap();
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_plan_modify_delete_insert_where() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));
        let where_scan = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Variable("old".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let modify = LogicalOperator::Modify(ModifyOp {
            delete_templates: vec![TripleTemplate {
                subject: TripleComponent::Variable("s".to_string()),
                predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
                object: TripleComponent::Variable("old".to_string()),
                graph: None,
            }],
            insert_templates: vec![TripleTemplate {
                subject: TripleComponent::Variable("s".to_string()),
                predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
                object: TripleComponent::Literal(Value::String("Alix R.".into())),
                graph: None,
            }],
            where_clause: Box::new(where_scan),
            graph: None,
        });
        execute_update(&store, &LogicalPlan::new(modify)).unwrap();
        assert_eq!(store.len(), 1);
        let all = store.find(&TriplePattern {
            subject: None,
            predicate: None,
            object: None,
        });
        assert_eq!(all[0].object().to_string(), "\"Alix R.\"");
    }

    #[test]
    fn test_plan_union() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));
        store.insert(Triple::new(
            Term::iri("http://example.org/gus"),
            Term::iri("http://example.org/label"),
            Term::literal("Gus"),
        ));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let scan1 = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Variable("o".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let scan2 = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://example.org/label".to_string()),
            object: TripleComponent::Variable("o".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let union = LogicalOperator::Union(crate::query::plan::UnionOp {
            inputs: vec![scan1, scan2],
        });
        let physical = planner.plan(&LogicalPlan::new(union)).unwrap();
        let mut op = physical.operator;
        let mut rows = 0;
        while let Ok(Some(c)) = op.next() {
            rows += c.row_count();
        }
        assert_eq!(rows, 2);
    }

    /// White-box guard for the UNION output-arity fix: when a later branch
    /// binds a variable absent from branch 0, `plan_union` must advertise the
    /// UNION of the branch variable sets and every emitted chunk must be
    /// null-padded to that unified width (not the branch-0 width). This is the
    /// planner-level counterpart to the gtest oracles; a regression here is the
    /// branch-0-only seed returning, which surfaces downstream as GRAFEO-X001 /
    /// GRAFEO-V001 / silent SELECT-star corruption.
    #[test]
    fn test_plan_union_heterogeneous_pads_to_union_domain() {
        let store = Arc::new(RdfStore::new());
        // Branch 0 (?s <p> ?a) binds {s, a}; branch 1 (?s <q> ?b) binds {s, b}.
        store.insert(Triple::new(
            Term::iri("http://example.org/x"),
            Term::iri("http://example.org/p"),
            Term::literal("PVAL"),
        ));
        store.insert(Triple::new(
            Term::iri("http://example.org/y"),
            Term::iri("http://example.org/q"),
            Term::literal("QVAL"),
        ));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let scan1 = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://example.org/p".to_string()),
            object: TripleComponent::Variable("a".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let scan2 = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://example.org/q".to_string()),
            object: TripleComponent::Variable("b".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let union = LogicalOperator::Union(crate::query::plan::UnionOp {
            inputs: vec![scan1, scan2],
        });
        let physical = planner.plan(&LogicalPlan::new(union)).unwrap();

        // The advertised schema is the UNION of the branch variable sets.
        assert_eq!(physical.columns.len(), 3, "expected the union of {{s,a,b}}");
        for v in ["s", "a", "b"] {
            assert!(
                physical.columns.iter().any(|c| c == v),
                "unified UNION schema is missing variable '{v}': {:?}",
                physical.columns
            );
        }
        let a_idx = physical.columns.iter().position(|c| c == "a").unwrap();
        let b_idx = physical.columns.iter().position(|c| c == "b").unwrap();

        let mut op = physical.operator;
        let mut rows = 0;
        let mut saw_a_only = false;
        let mut saw_b_only = false;
        while let Ok(Some(chunk)) = op.next() {
            // Every branch chunk is genuinely padded to the unified width.
            assert_eq!(
                chunk.column_count(),
                3,
                "branch chunk was not null-padded to the unified width"
            );
            for row in chunk.selected_indices() {
                let a = chunk.column(a_idx).unwrap().get_value(row);
                let b = chunk.column(b_idx).unwrap().get_value(row);
                let a_null = matches!(a, None | Some(Value::Null));
                let b_null = matches!(b, None | Some(Value::Null));
                // Disjoint branches: exactly one of {a, b} is bound per row, and
                // the absent branch's variable is UNBOUND (Null), never dropped.
                assert!(
                    a_null ^ b_null,
                    "each row binds exactly one branch variable; got a={a:?}, b={b:?}"
                );
                saw_a_only |= b_null;
                saw_b_only |= a_null;
                rows += 1;
            }
        }
        assert_eq!(rows, 2, "expected one row from each branch");
        assert!(saw_a_only, "branch 0 row (a bound, b UNBOUND) missing");
        assert!(saw_b_only, "branch 1 row (b bound, a UNBOUND) missing");
    }

    #[test]
    fn test_plan_construct() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let scan = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Variable("name".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let construct = LogicalOperator::Construct(ConstructOp {
            templates: vec![TripleTemplate {
                subject: TripleComponent::Variable("s".to_string()),
                predicate: TripleComponent::Iri("http://example.org/label".to_string()),
                object: TripleComponent::Variable("name".to_string()),
                graph: None,
            }],
            input: Box::new(scan),
        });
        let physical = planner.plan(&LogicalPlan::new(construct)).unwrap();
        assert_eq!(physical.columns, vec!["subject", "predicate", "object"]);
        let mut op = physical.operator;
        let mut rows = 0;
        while let Ok(Some(c)) = op.next() {
            rows += c.row_count();
        }
        assert_eq!(rows, 1);
    }

    #[test]
    fn test_plan_bind_execution() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let scan = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Variable("name".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let bind = LogicalOperator::Bind(BindOp {
            input: Box::new(scan),
            variable: "upper".to_string(),
            expression: crate::query::plan::LogicalExpression::SparqlCall {
                name: "UCASE".to_string(),
                args: vec![crate::query::plan::LogicalExpression::Variable(
                    "name".to_string(),
                )],
            },
        });
        let physical = planner.plan(&LogicalPlan::new(bind)).unwrap();
        assert!(physical.columns.contains(&"upper".to_string()));
        let mut op = physical.operator;
        let chunk = op.next().unwrap().unwrap();
        let upper_idx = physical.columns.iter().position(|c| c == "upper").unwrap();
        assert_eq!(
            chunk.column(upper_idx).unwrap().get_value(0),
            Some(literal("ALIX"))
        );
    }

    #[test]
    fn test_plan_project_with_expression() {
        use crate::query::plan::{ProjectOp, Projection};
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let scan = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Variable("name".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let project = LogicalOperator::Project(ProjectOp {
            projections: vec![
                Projection {
                    expression: LogicalExpression::Variable("name".to_string()),
                    alias: Some("n".to_string()),
                },
                Projection {
                    expression: LogicalExpression::Literal(Value::String("const".into())),
                    alias: Some("c".to_string()),
                },
            ],
            input: Box::new(scan),
            pass_through_input: false,
        });
        let physical = planner.plan(&LogicalPlan::new(project)).unwrap();
        assert_eq!(physical.columns, vec!["n", "c"]);
        let mut op = physical.operator;
        let chunk = op.next().unwrap().unwrap();
        assert_eq!(chunk.column(0).unwrap().get_value(0), Some(literal("Alix")));
        assert_eq!(
            chunk.column(1).unwrap().get_value(0),
            Some(literal("const"))
        );
    }

    #[test]
    fn test_plan_create_graph() {
        let store = Arc::new(RdfStore::new());
        execute_update(
            &store,
            &LogicalPlan::new(LogicalOperator::CreateGraph(CreateGraphOp {
                graph: "http://example.org/g1".to_string(),
                silent: false,
            })),
        )
        .unwrap();
        assert!(store.graph("http://example.org/g1").is_some());
    }

    #[test]
    fn test_plan_create_graph_already_exists_errors() {
        let store = Arc::new(RdfStore::new());
        store.create_graph("http://example.org/g1");
        let plan = LogicalPlan::new(LogicalOperator::CreateGraph(CreateGraphOp {
            graph: "http://example.org/g1".to_string(),
            silent: false,
        }));
        let error = execute_update(&store, &plan).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Graph <http://example.org/g1> already exists"),
            "{error}"
        );
    }

    #[test]
    fn test_plan_create_graph_silent() {
        let store = Arc::new(RdfStore::new());
        store.create_graph("http://example.org/g1");
        assert!(
            execute_update(
                &store,
                &LogicalPlan::new(LogicalOperator::CreateGraph(CreateGraphOp {
                    graph: "http://example.org/g1".to_string(),
                    silent: true,
                },))
            )
            .is_ok()
        );
    }

    #[test]
    fn test_plan_drop_graph() {
        let store = Arc::new(RdfStore::new());
        store.create_graph("http://example.org/g1");
        execute_update(
            &store,
            &LogicalPlan::new(LogicalOperator::DropGraph(DropGraphOp {
                target: grafeo_common::storage::log_record::RdfGraphTarget::Named(
                    "http://example.org/g1".to_string(),
                ),
                silent: false,
            })),
        )
        .unwrap();
        assert!(store.graph("http://example.org/g1").is_none());
    }

    #[test]
    fn test_plan_drop_nonexistent_graph_errors() {
        let store = Arc::new(RdfStore::new());
        let plan = LogicalPlan::new(LogicalOperator::DropGraph(DropGraphOp {
            target: grafeo_common::storage::log_record::RdfGraphTarget::Named(
                "http://example.org/nope".to_string(),
            ),
            silent: false,
        }));
        let error = execute_update(&store, &plan).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Graph <http://example.org/nope> does not exist"),
            "{error}"
        );
    }

    #[test]
    fn test_plan_drop_default_graph() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/a"),
            Term::iri("http://example.org/b"),
            Term::literal("c"),
        ));
        execute_update(
            &store,
            &LogicalPlan::new(LogicalOperator::DropGraph(DropGraphOp {
                target: grafeo_common::storage::log_record::RdfGraphTarget::Default,
                silent: false,
            })),
        )
        .unwrap();
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_plan_clear_graph_default() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/a"),
            Term::iri("http://example.org/b"),
            Term::literal("c"),
        ));
        execute_update(
            &store,
            &LogicalPlan::new(LogicalOperator::ClearGraph(ClearGraphOp {
                target: grafeo_common::storage::log_record::RdfGraphTarget::Default,
                silent: false,
            })),
        )
        .unwrap();
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_plan_clear_all_graphs() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/a"),
            Term::iri("http://example.org/b"),
            Term::literal("c"),
        ));
        store.create_graph("http://example.org/g1");
        store
            .graph("http://example.org/g1")
            .unwrap()
            .insert(Triple::new(
                Term::iri("http://example.org/x"),
                Term::iri("http://example.org/y"),
                Term::literal("z"),
            ));
        execute_update(
            &store,
            &LogicalPlan::new(LogicalOperator::ClearGraph(ClearGraphOp {
                target: grafeo_common::storage::log_record::RdfGraphTarget::All,
                silent: false,
            })),
        )
        .unwrap();
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_plan_multi_way_join_hash_fallback() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/age"),
            Term::typed_literal("30", "http://www.w3.org/2001/XMLSchema#integer"),
        ));
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/knows"),
            Term::iri("http://example.org/gus"),
        ));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let scan1 = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Variable("name".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let scan2 = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/age".to_string()),
            object: TripleComponent::Variable("age".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let scan3 = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/knows".to_string()),
            object: TripleComponent::Variable("friend".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let mwj = LogicalOperator::MultiWayJoin(crate::query::plan::MultiWayJoinOp {
            inputs: vec![scan1, scan2, scan3],
            conditions: vec![],
            shared_variables: vec!["s".to_string()],
        });
        let physical = planner.plan(&LogicalPlan::new(mwj)).unwrap();
        let mut op = physical.operator;
        let mut rows = 0;
        while let Ok(Some(c)) = op.next() {
            rows += c.row_count();
        }
        assert_eq!(rows, 1);
    }

    #[test]
    fn a_multi_way_join_takes_the_smallest_connected_input_next() {
        let names = |list: &[&str]| list.iter().map(ToString::to_string).collect::<Vec<_>>();
        let (post, person, tagged, interest) = (
            names(&["m"]),
            names(&["p"]),
            names(&["m", "t"]),
            names(&["p", "t"]),
        );
        let inputs = [
            post.as_slice(),
            person.as_slice(),
            tagged.as_slice(),
            interest.as_slice(),
        ];
        // Over the tags the posts and people are smaller but unconnected.
        assert_eq!(next_join_input(inputs.into_iter(), &names(&["t"])), 2);
        // Once `m` is bound the posts connect, and they are the smallest.
        assert_eq!(next_join_input(inputs.into_iter(), &names(&["t", "m"])), 0);
        // Nothing connects: the smallest, a real cross product.
        assert_eq!(next_join_input(inputs.into_iter(), &names(&["x"])), 0);
    }

    /// Reads the tags of a graph where the smallest inputs share no variable:
    /// 60 tags, 1,500 posts and 1,501 people cross to 135 million rows, which
    /// runs into the timeout, when a join takes them by size alone.
    #[cfg(feature = "sparql")]
    #[test]
    fn a_multi_way_join_does_not_cross_its_smallest_inputs() {
        use std::fmt::Write as _;

        let db = crate::GrafeoDB::with_config(
            crate::config::Config::in_memory()
                .with_query_timeout(std::time::Duration::from_secs(10)),
        )
        .unwrap();
        let mut triples = String::new();
        for tag in 0..60 {
            writeln!(
                triples,
                "<urn:t{tag}> a <urn:Tag> ; <urn:name> \"tag{tag:02}\" ."
            )
            .unwrap();
        }
        for person in 0..1501 {
            let (first, second) = (person % 60, (person + 1) % 60);
            writeln!(
                triples,
                "<urn:p{person}> a <urn:Person> ; <urn:interest> <urn:t{first}> , <urn:t{second}> ."
            )
            .unwrap();
        }
        for post in 0..1500 {
            let tag = post % 60;
            writeln!(
                triples,
                "<urn:m{post}> a <urn:Post> ; <urn:tag> <urn:t{tag}> ; <urn:creator> <urn:p{post}> ."
            )
            .unwrap();
        }
        // Tagged and created like a post, but not one: these make the edges
        // outnumber the posts and people, as in a real graph.
        for comment in 0..2000 {
            let (tag, creator) = (comment % 60, comment % 1501);
            writeln!(
                triples,
                "<urn:c{comment}> <urn:tag> <urn:t{tag}> ; <urn:creator> <urn:p{creator}> ."
            )
            .unwrap();
        }
        db.execute_sparql(&format!(
            "INSERT DATA {{
{triples}}}"
        ))
        .unwrap();

        let result = db
            .execute_sparql(
                "SELECT ?name (COUNT(*) AS ?posts) WHERE {
                   ?p a <urn:Person> ; <urn:interest> ?t .
                   ?t a <urn:Tag> ; <urn:name> ?name .
                   ?m a <urn:Post> ; <urn:tag> ?t ; <urn:creator> ?p .
                 } GROUP BY ?name ORDER BY ?name",
            )
            .unwrap();

        // Post i has tag i % 60 and creator i, whose interests include that
        // tag: each of the 60 tags counts its 25 posts.
        let expected: Vec<Vec<Value>> = (0..60)
            .map(|tag| vec![literal(&format!("tag{tag:02}")), integer(25)])
            .collect();
        assert_eq!(result.rows(), expected.as_slice());
    }

    #[test]
    fn test_plan_profiled() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));
        let planner = RdfPlanner::new(store);
        let plan = LogicalPlan::new(LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Variable("p".to_string()),
            object: TripleComponent::Variable("o".to_string()),
            graph: None,
            input: None,
            dataset: None,
        }));
        let (physical, entries) = planner.plan_profiled(&plan).unwrap();
        assert_eq!(physical.columns, vec!["s", "p", "o"]);
        assert!(!entries.is_empty());
    }

    #[test]
    fn test_plan_unsupported_operator_returns_error() {
        let store = Arc::new(RdfStore::new());
        let planner = RdfPlanner::new(store);
        let map = LogicalOperator::MapCollect(crate::query::plan::MapCollectOp {
            input: Box::new(LogicalOperator::Empty),
            key_var: "k".to_string(),
            value_var: "v".to_string(),
            alias: "m".to_string(),
        });
        assert!(planner.plan(&LogicalPlan::new(map)).is_err());
    }

    #[test]
    fn test_plan_empty_operator() {
        let store = Arc::new(RdfStore::new());
        let planner = RdfPlanner::new(store);
        let physical = planner
            .plan(&LogicalPlan::new(LogicalOperator::Empty))
            .unwrap();
        assert!(physical.columns.is_empty(), "{:?}", physical.columns);
    }

    #[test]
    fn test_component_to_term_conversions() {
        let store = Arc::new(RdfStore::new());
        let planner = RdfPlanner::new(store);
        assert!(matches!(
            planner
                .component_to_term(&TripleComponent::Iri("http://example.org/x".to_string()))
                .unwrap(),
            Term::Iri(_)
        ));
        assert!(matches!(
            planner
                .component_to_term(&TripleComponent::Literal(Value::String("hello".into())))
                .unwrap(),
            Term::Literal(_)
        ));
        assert!(matches!(
            planner
                .component_to_term(&TripleComponent::Literal(Value::Int64(42)))
                .unwrap(),
            Term::Literal(_)
        ));
        assert!(matches!(
            planner
                .component_to_term(&TripleComponent::Literal(Value::Float64(2.72)))
                .unwrap(),
            Term::Literal(_)
        ));
        assert!(matches!(
            planner
                .component_to_term(&TripleComponent::Literal(Value::Bool(true)))
                .unwrap(),
            Term::Literal(_)
        ));
        assert!(matches!(
            planner
                .component_to_term(&TripleComponent::LangLiteral {
                    value: "hello".to_string(),
                    lang: "en".to_string()
                })
                .unwrap(),
            Term::Literal(_)
        ));
        assert!(matches!(
            planner
                .component_to_term(&TripleComponent::BlankNode("b0".to_string()))
                .unwrap(),
            Term::BlankNode(_)
        ));
        assert!(
            planner
                .component_to_term(&TripleComponent::Variable("x".to_string()))
                .is_err()
        );
    }

    #[test]
    fn test_count_fast_path_predicate_bound() {
        use crate::query::plan::{AggregateExpr, AggregateFunction, AggregateOp};
        let store = Arc::new(RdfStore::new());
        for i in 0..5 {
            store.insert(Triple::new(
                Term::iri(format!("http://example.org/item{i}")),
                Term::iri("http://example.org/type"),
                Term::literal(format!("val{i}")),
            ));
        }
        store.insert(Triple::new(
            Term::iri("http://example.org/other"),
            Term::iri("http://example.org/other_pred"),
            Term::literal("x"),
        ));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let scan = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://example.org/type".to_string()),
            object: TripleComponent::Variable("o".to_string()),
            graph: None,
            input: None,
            dataset: None,
        });
        let agg = LogicalOperator::Aggregate(AggregateOp {
            input: Box::new(scan),
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
            having: None,
        });
        let physical = planner.plan(&LogicalPlan::new(agg)).unwrap();
        let mut op = physical.operator;
        let chunk = op.next().unwrap().unwrap();
        assert_eq!(chunk.column(0).unwrap().get_value(0), Some(integer(5)));
    }

    /// Runs the update `plan` on `store` in a private transaction, as a
    /// session runs one outside a transaction: its triples apply once it
    /// ran.
    fn execute_update(store: &Arc<RdfStore>, plan: &LogicalPlan) -> Result<()> {
        let manager = Arc::new(crate::transaction::TransactionManager::new());
        crate::transaction::update_privately(store, &manager, |writer| {
            let mut physical = RdfPlanner::new(Arc::clone(store))
                .with_writer(Some(writer))
                .plan(plan)?;
            while physical
                .operator
                .next()
                .map_err(crate::query::executor::convert_operator_error)?
                .is_some()
            {}
            Ok(())
        })
    }

    // ---- into_any() coverage tests ----
    //
    // Each RDF operator implements `into_any()` for downcasting support in the
    // push pipeline. These tests construct minimal instances and verify the
    // method returns a valid `Box<dyn Any + Send>` that can be downcast back.

    #[test]
    fn test_into_any_rdf_insert_triple_operator() {
        let triple = Triple::new(
            Term::iri("http://example.org/s"),
            Term::iri("http://example.org/p"),
            Term::literal("o"),
        );
        let op: Box<dyn Operator> = Box::new(RdfInsertTripleOperator::new(None, triple, None));
        let any = op.into_any();
        assert!(any.downcast::<RdfInsertTripleOperator>().is_ok());
    }

    #[test]
    fn test_into_any_rdf_insert_pattern_operator() {
        let child: Box<dyn Operator> = Box::new(SingleRowOperator::new());
        let operands = TripleOperands {
            subject: TripleComponent::Iri("http://example.org/s".to_string()),
            predicate: TripleComponent::Iri("http://example.org/p".to_string()),
            object: TripleComponent::Literal(Value::String("o".into())),
            column_map: HashMap::new(),
            graph: None,
            terms: Arc::default(),
        };
        let op: Box<dyn Operator> = Box::new(RdfInsertPatternOperator::new(None, child, operands));
        let any = op.into_any();
        assert!(any.downcast::<RdfInsertPatternOperator>().is_ok());
    }

    #[test]
    fn test_into_any_rdf_delete_triple_operator() {
        let triple = Triple::new(
            Term::iri("http://example.org/s"),
            Term::iri("http://example.org/p"),
            Term::literal("o"),
        );
        let op: Box<dyn Operator> = Box::new(RdfDeleteTripleOperator::new(None, triple, None));
        let any = op.into_any();
        assert!(any.downcast::<RdfDeleteTripleOperator>().is_ok());
    }

    #[test]
    fn test_into_any_rdf_delete_pattern_operator() {
        let child: Box<dyn Operator> = Box::new(SingleRowOperator::new());
        let operands = TripleOperands {
            subject: TripleComponent::Iri("http://example.org/s".to_string()),
            predicate: TripleComponent::Iri("http://example.org/p".to_string()),
            object: TripleComponent::Literal(Value::String("o".into())),
            column_map: HashMap::new(),
            graph: None,
            terms: Arc::default(),
        };
        let op: Box<dyn Operator> = Box::new(RdfDeletePatternOperator::new(None, child, operands));
        let any = op.into_any();
        assert!(any.downcast::<RdfDeletePatternOperator>().is_ok());
    }

    #[test]
    fn test_into_any_rdf_graph_operator() {
        let op: Box<dyn Operator> = Box::new(RdfGraphOperator {
            writer: None,
            name: "RdfClearGraph",
            op: RdfGraphOp::Clear {
                target: RdfGraphTarget::Default,
            },
            check: GraphCheck::None,
            done: false,
        });
        assert_eq!(op.name(), "RdfClearGraph");
        let any = op.into_any();
        assert!(any.downcast::<RdfGraphOperator>().is_ok());
    }

    #[test]
    fn test_into_any_rdf_modify_operator() {
        let child: Box<dyn Operator> = Box::new(SingleRowOperator::new());
        let op: Box<dyn Operator> = Box::new(RdfModifyOperator {
            writer: None,
            input: child,
            delete_templates: vec![],
            insert_templates: vec![],
            column_map: HashMap::new(),
            terms: Arc::default(),
            done: false,
        });
        let any = op.into_any();
        assert!(any.downcast::<RdfModifyOperator>().is_ok());
    }

    #[test]
    fn test_into_any_rdf_union_operator() {
        let op: Box<dyn Operator> = Box::new(RdfUnionOperator::new(vec![], 0));
        let any = op.into_any();
        assert!(any.downcast::<RdfUnionOperator>().is_ok());
    }

    #[test]
    fn test_into_any_rdf_bind_operator() {
        let child: Box<dyn Operator> = Box::new(SingleRowOperator::new());
        let expr = FilterExpression::Literal(Value::String("test".into()));
        let terms: Arc<QueryTermIds> = Arc::default();
        let op: Box<dyn Operator> = Box::new(RdfBindOperator {
            child,
            evaluator: RdfExpressionPredicate::new(
                expr,
                HashMap::new(),
                Arc::clone(&terms),
                ExpressionContext {
                    clock: StatementClock::now(),
                    row: None,
                },
            ),
            terms,
        });
        let any = op.into_any();
        assert!(any.downcast::<RdfBindOperator>().is_ok());
    }

    #[test]
    fn test_into_any_rdf_project_operator() {
        let child: Box<dyn Operator> = Box::new(SingleRowOperator::new());
        let op: Box<dyn Operator> = Box::new(RdfProjectOperator::new(
            child,
            vec![RdfProjectExpr::Constant(Value::Null)],
            vec![LogicalType::RdfTermId],
            Arc::default(),
            ExpressionContext {
                clock: StatementClock::now(),
                row: None,
            },
        ));
        let any = op.into_any();
        assert!(any.downcast::<RdfProjectOperator>().is_ok());
    }

    #[test]
    fn test_into_any_construct_operator() {
        let child: Box<dyn Operator> = Box::new(SingleRowOperator::new());
        let op: Box<dyn Operator> = Box::new(ConstructOperator {
            input: child,
            templates: vec![],
            terms: Arc::default(),
            seen: grafeo_common::utils::hash::FxHashSet::default(),
        });
        let any = op.into_any();
        assert!(any.downcast::<ConstructOperator>().is_ok());
    }

    #[test]
    fn test_into_any_constant_operator() {
        let chunk = DataChunk::empty();
        let op: Box<dyn Operator> = Box::new(ConstantOperator::new(chunk));
        let any = op.into_any();
        assert!(any.downcast::<ConstantOperator>().is_ok());
    }

    #[test]
    fn test_into_any_term_decode_operator() {
        let child: Box<dyn Operator> = Box::new(SingleRowOperator::new());
        let op: Box<dyn Operator> = Box::new(TermDecodeOperator {
            input: child,
            terms: Arc::default(),
            columns: vec![],
            decode_as: DecodeAs::Terms,
        });
        let any = op.into_any();
        assert!(any.downcast::<TermDecodeOperator>().is_ok());
    }

    #[test]
    fn test_into_any_rdf_triple_scan_operator() {
        let store = Arc::new(RdfStore::new());
        let pattern = TriplePattern {
            subject: None,
            predicate: None,
            object: None,
        };
        let op: Box<dyn Operator> = Box::new(RdfTripleScanOperator::new(
            store,
            pattern,
            [true, true, true, false],
            1024,
            GraphContext {
                graph: None,
                scan_all_graphs: false,
                dataset: None,
            },
            Arc::default(),
        ));
        let any = op.into_any();
        assert!(any.downcast::<RdfTripleScanOperator>().is_ok());
    }
}
