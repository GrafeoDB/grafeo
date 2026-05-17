//! RDF Query Planner.
//!
//! Converts logical plans with RDF operators (TripleScan, etc.) to physical
//! operators that execute against an RDF store.
//!
//! This planner follows the same push-based, vectorized execution model as
//! the LPG planner for consistent performance characteristics.

use std::collections::HashMap;
use std::sync::Arc;

use grafeo_common::grafeo_warn;
use grafeo_common::types::{LogicalType, TransactionId, Value};
use grafeo_common::utils::error::{Error, Result};
use grafeo_core::execution::DataChunk;
use grafeo_core::execution::operators::{
    BinaryFilterOp, FilterExpression, FilterOperator, HashAggregateOperator, Operator,
    OperatorError, Predicate, ProjectExpr, ProjectOperator, SimpleAggregateOperator,
    SingleRowOperator, SortOperator, UnaryFilterOp,
};
use grafeo_core::graph::GraphStoreSearch;
use grafeo_core::graph::rdf::{Literal, RdfStore, Term, Triple, TriplePattern};

use crate::query::plan::{
    AddGraphOp, AggregateFunction as LogicalAggregateFunction, AggregateOp, AntiJoinOp, BindOp,
    ClearGraphOp, ConstructOp, CopyGraphOp, CreateGraphOp, DatasetRestriction, DeleteTripleOp,
    DistinctOp, DropGraphOp, FilterOp, InsertTripleOp, LeftJoinOp, LimitOp, LogicalExpression,
    LogicalOperator, LogicalPlan, ModifyOp, MoveGraphOp, SkipOp, SortOp, TripleComponent,
    TripleScanOp, TripleTemplate,
};
use crate::query::planner::{PhysicalPlan, convert_aggregate_function, convert_filter_expression};

#[cfg(feature = "regex")]
use regex::Regex;
#[cfg(all(feature = "regex-lite", not(feature = "regex")))]
use regex_lite::Regex;

/// Default chunk size for morsel-driven execution.
const DEFAULT_CHUNK_SIZE: usize = 1024;

/// Logs an RDF WAL record if a WAL reference is present.
#[cfg(feature = "wal")]
fn log_rdf_wal(wal: &Option<Arc<RdfWal>>, record: &grafeo_storage::wal::WalRecord) {
    if let Some(wal) = wal
        && let Err(err) = wal.log(record)
    {
        grafeo_warn!("RDF WAL log failed: {err}");
    }
}

/// Converts a Term to its N-Triples string for WAL serialization.
#[cfg(feature = "wal")]
fn term_to_wal(term: &Term) -> String {
    term.to_string()
}

/// Records a triple insertion to the CDC log if one is configured.
#[cfg(feature = "cdc")]
fn record_cdc_triple_insert(
    cdc_log: &Option<Arc<crate::cdc::CdcLog>>,
    subject: &Term,
    predicate: &Term,
    object: &Term,
    graph: Option<&str>,
    epoch: grafeo_common::types::EpochId,
) {
    if let Some(log) = cdc_log {
        log.record_triple_insert(
            &subject.to_string(),
            &predicate.to_string(),
            &object.to_string(),
            graph,
            epoch,
        );
    }
}

/// Records a triple deletion to the CDC log if one is configured.
#[cfg(feature = "cdc")]
fn record_cdc_triple_delete(
    cdc_log: &Option<Arc<crate::cdc::CdcLog>>,
    subject: &Term,
    predicate: &Term,
    object: &Term,
    graph: Option<&str>,
    epoch: grafeo_common::types::EpochId,
) {
    if let Some(log) = cdc_log {
        log.record_triple_delete(
            &subject.to_string(),
            &predicate.to_string(),
            &object.to_string(),
            graph,
            epoch,
        );
    }
}

/// Type alias for the WAL used by the RDF planner.
#[cfg(feature = "wal")]
type RdfWal = grafeo_storage::wal::LpgWal;

/// Groups the variable-substitution operands for pattern-based mutation operators.
///
/// Used to keep `RdfInsertPatternOperator::new` and `RdfDeletePatternOperator::new`
/// within the 7-argument clippy limit.
struct TripleOperands {
    subject: TripleComponent,
    predicate: TripleComponent,
    object: TripleComponent,
    column_map: HashMap<String, usize>,
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
    /// Optional transaction ID for transactional operations.
    transaction_id: Option<TransactionId>,
    /// When true, each physical operator is wrapped in `ProfiledOperator`.
    profiling: std::cell::Cell<bool>,
    /// Profile entries collected during planning (post-order).
    profile_entries: std::cell::RefCell<Vec<crate::query::profile::ProfileEntry>>,
    /// Optional WAL for logging RDF mutations.
    #[cfg(feature = "wal")]
    wal: Option<Arc<RdfWal>>,
    /// Optional CDC log for recording RDF triple mutations.
    #[cfg(feature = "cdc")]
    cdc_log: Option<Arc<crate::cdc::CdcLog>>,
    /// Epoch to stamp CDC events with (snapshot at plan time).
    #[cfg(feature = "cdc")]
    cdc_epoch: grafeo_common::types::EpochId,
    /// Whether the query uses LANG()/LANGMATCHES()/DATATYPE() functions.
    /// When false, companion columns are not emitted, saving ~66% scan overhead.
    needs_companion_columns: std::cell::Cell<bool>,
    /// Term dictionary for dictionary-encoded triple scans. When present,
    /// `plan_triple_scan()` emits Int64 term IDs instead of strings, and a
    /// `DictResolveOperator` at the result boundary converts them back.
    dictionary: Option<Arc<grafeo_core::graph::rdf::TermDictionary>>,
    /// Column names that carry dictionary-encoded Int64 term IDs.
    /// Populated by `plan_triple_scan()`, consumed by `plan()` for resolution.
    encoded_columns: std::cell::RefCell<std::collections::HashSet<String>>,
}

impl RdfPlanner {
    /// Creates a new RDF planner with the given store.
    #[must_use]
    pub fn new(store: Arc<RdfStore>) -> Self {
        Self {
            store,
            chunk_size: DEFAULT_CHUNK_SIZE,
            transaction_id: None,
            profiling: std::cell::Cell::new(false),
            profile_entries: std::cell::RefCell::new(Vec::new()),
            needs_companion_columns: std::cell::Cell::new(false),
            dictionary: None,
            encoded_columns: std::cell::RefCell::new(std::collections::HashSet::new()),
            #[cfg(feature = "wal")]
            wal: None,
            #[cfg(feature = "cdc")]
            cdc_log: None,
            #[cfg(feature = "cdc")]
            cdc_epoch: grafeo_common::types::EpochId(0),
        }
    }

    /// Sets the chunk size for vectorized execution.
    #[must_use]
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Sets the transaction ID for transactional operations.
    #[must_use]
    pub fn with_transaction_id(mut self, transaction_id: Option<TransactionId>) -> Self {
        self.transaction_id = transaction_id;
        self
    }

    /// Sets the WAL for logging RDF mutations.
    #[cfg(feature = "wal")]
    #[must_use]
    pub fn with_wal(mut self, wal: Option<Arc<RdfWal>>) -> Self {
        self.wal = wal;
        self
    }

    /// Sets the CDC log and epoch for recording RDF triple mutations.
    #[cfg(feature = "cdc")]
    #[must_use]
    pub fn with_cdc_log(
        mut self,
        cdc_log: Option<Arc<crate::cdc::CdcLog>>,
        epoch: grafeo_common::types::EpochId,
    ) -> Self {
        self.cdc_log = cdc_log;
        self.cdc_epoch = epoch;
        self
    }

    /// Plans a logical plan into a physical operator tree.
    ///
    /// # Errors
    ///
    /// Returns an error if planning fails.
    pub fn plan(&self, logical_plan: &LogicalPlan) -> Result<PhysicalPlan> {
        // Pre-analyze: only emit companion columns if the query uses LANG/DATATYPE
        self.needs_companion_columns
            .set(uses_lang_or_datatype(&logical_plan.root));

        let (mut operator, columns, _types) = self.plan_operator(&logical_plan.root)?;

        // Resolve dictionary-encoded columns back to strings at the result boundary.
        if let Some(ref dict) = self.dictionary {
            let encoded = self.encoded_columns.borrow();
            if !encoded.is_empty() {
                let encoded_indices: Vec<usize> = columns
                    .iter()
                    .enumerate()
                    .filter(|(_, name)| encoded.contains(*name))
                    .map(|(i, _)| i)
                    .collect();
                if !encoded_indices.is_empty() {
                    operator = Box::new(DictResolveOperator::new(
                        operator,
                        Arc::clone(dict),
                        encoded_indices,
                    ));
                }
            }
        }

        // Strip internal companion columns (__lang_<var>, __datatype_<var>)
        // from the output. They are used by LANG()/LANGMATCHES()/DATATYPE()
        // during evaluation but should never appear in query results.
        let (operator, columns) = strip_internal_columns(operator, columns);
        Ok(PhysicalPlan {
            operator,
            columns,
            adaptive_context: None,
        })
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

        let result = self.plan_operator(&logical_plan.root);

        self.profiling.set(false);
        let (operator, columns, _types) = result?;
        let (operator, columns) = strip_internal_columns(operator, columns);
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

    /// If profiling is enabled, wraps a planned result in `ProfiledOperator`
    /// and records a [`ProfileEntry`](crate::query::profile::ProfileEntry).
    fn maybe_profile(
        &self,
        result: Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)>,
        op: &LogicalOperator,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        if self.profiling.get() {
            let (physical, columns, types) = result?;
            let (entry, stats) =
                crate::query::profile::ProfileEntry::new(physical.name(), op.display_label());
            let profiled = grafeo_core::execution::ProfiledOperator::new(physical, stats);
            self.profile_entries.borrow_mut().push(entry);
            Ok((Box::new(profiled), columns, types))
        } else {
            result
        }
    }

    /// Plans a single logical operator.
    fn plan_operator(
        &self,
        op: &LogicalOperator,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        let result = match op {
            LogicalOperator::TripleScan(scan) => self.plan_triple_scan(scan),
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
            _ => Err(Error::Internal(format!(
                "Unsupported RDF operator: {:?}",
                std::mem::discriminant(op)
            ))),
        };
        self.maybe_profile(result, op)
    }

    /// Plans a triple scan operator.
    ///
    /// Creates a lazy scanning operator that reads triples in chunks
    /// for cache-efficient, vectorized processing.
    fn plan_triple_scan(
        &self,
        scan: &TripleScanOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        // Build the triple pattern for querying the store
        let pattern = self.build_triple_pattern(scan);

        // Determine which columns are variables (and thus in output)
        let mut columns = Vec::new();
        let mut output_mask = [false, false, false, false]; // s, p, o, g

        if let TripleComponent::Variable(name) = &scan.subject {
            columns.push(name.clone());
            output_mask[0] = true;
        }
        if let TripleComponent::Variable(name) = &scan.predicate {
            columns.push(name.clone());
            output_mask[1] = true;
        }
        // Track whether the object is a variable (for language-tag companion column)
        let mut object_var_name: Option<String> = None;
        if let TripleComponent::Variable(name) = &scan.object {
            columns.push(name.clone());
            output_mask[2] = true;
            object_var_name = Some(name.clone());
        }

        // When the object is a variable, add a hidden companion column for
        // language tags so that LANG() and LANGMATCHES() can access them.
        // This must be added BEFORE the graph column to match the DataChunk
        // layout (the lang column is emitted right after the object column).
        let emit_companion_columns = object_var_name.is_some();
        let emit_datatype_column = emit_companion_columns && self.needs_companion_columns.get();
        if let Some(ref obj_name) = object_var_name {
            columns.push(format!("__lang_{obj_name}"));
            if emit_datatype_column {
                columns.push(format!("__datatype_{obj_name}"));
            }
        }

        if let Some(TripleComponent::Variable(name)) = &scan.graph {
            columns.push(name.clone());
            output_mask[3] = true;
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
            emit_companion_columns,
            emit_datatype_column,
        );

        // Dictionary encoding is available but not yet automatically enabled for
        // all queries. The infrastructure (TermDictionary, DictResolveOperator) is
        // in place for use by the Ring Index planner (Phase 4) and WCOJ joins.
        // Automatic activation requires detecting whether downstream operators
        // (FILTER, BIND, ORDER BY) inspect string content, which is deferred.
        let _ = &self.dictionary; // suppress unused warning
        let _ = &self.encoded_columns;

        let types = vec![LogicalType::String; columns.len()];
        Ok((Box::new(scan_op), columns, types))
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
    /// Handles EXISTS/NOT EXISTS patterns by transforming them into semi-joins/anti-joins.
    fn plan_filter(
        &self,
        filter: &FilterOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        // Check for EXISTS/NOT EXISTS patterns and transform to semi/anti joins
        if let Some((subquery, is_negated)) = self.extract_exists_pattern(&filter.predicate) {
            return self.plan_exists_as_join(&filter.input, subquery, is_negated);
        }

        let (input_op, columns, types) = self.plan_operator(&filter.input)?;

        // Build variable to column index mapping
        let variable_columns: HashMap<String, usize> = columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        // Convert logical expression to filter expression
        let filter_expr = convert_filter_expression(&filter.predicate)?;

        // Create RDF-specific predicate (doesn't need LpgStore)
        let predicate = RdfExpressionPredicate::new(filter_expr, variable_columns);

        let operator = Box::new(FilterOperator::new(input_op, Box::new(predicate)));
        Ok((operator, columns, types))
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
        input: &LogicalOperator,
        subquery: &LogicalOperator,
        is_negated: bool,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        use crate::query::planner::common;

        let (left_op, left_columns, left_types) = self.plan_operator(input)?;
        let (right_op, right_columns, _right_types) = self.plan_operator(subquery)?;

        // Use Anti for NOT EXISTS, Semi for EXISTS
        let (result_op, result_columns) = if is_negated {
            common::build_anti_join(
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
            types.clone(),
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
        let (op, cols) = common::build_limit(input_op, columns, limit.count.value(), types.clone());
        Ok((op, cols, types))
    }

    /// Plans a SKIP operator.
    fn plan_skip(
        &self,
        skip: &SkipOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        use crate::query::planner::common;
        let (input_op, columns, types) = self.plan_operator(&skip.input)?;
        let (op, cols) = common::build_skip(input_op, columns, skip.count.value(), types.clone());
        Ok((op, cols, types))
    }

    /// Plans a SORT operator.
    fn plan_sort(
        &self,
        sort: &SortOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        use crate::query::plan::SortOrder;
        use grafeo_core::execution::operators::{
            FilterExpression, NullOrder, ProjectExpr, ProjectOperator, SortDirection, SortKey,
        };

        let (mut input_op, columns, types) = self.plan_operator(&sort.input)?;

        let mut variable_columns: HashMap<String, usize> = columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        // Pre-project complex sort key expressions
        let mut expression_projections: Vec<(FilterExpression, String)> = Vec::new();
        let mut next_col_idx = columns.len();
        for key in &sort.keys {
            match &key.expression {
                LogicalExpression::Variable(_) => {}
                _ => {
                    let col_name = resolved_column_name(&key.expression);
                    if !variable_columns.contains_key(&col_name) {
                        let filter_expr = convert_filter_expression(&key.expression)?;
                        expression_projections.push((filter_expr, col_name.clone()));
                        variable_columns.insert(col_name, next_col_idx);
                        next_col_idx += 1;
                    }
                }
            }
        }

        if !expression_projections.is_empty() {
            let mut projections: Vec<ProjectExpr> =
                (0..columns.len()).map(ProjectExpr::Column).collect();
            let mut output_types: Vec<LogicalType> = types.clone();

            for (filter_expr, _col_name) in &expression_projections {
                projections.push(ProjectExpr::Expression {
                    expr: filter_expr.clone(),
                    variable_columns: variable_columns.clone(),
                });
                output_types.push(LogicalType::Any); // computed expressions may produce non-string types
            }

            input_op = Box::new(ProjectOperator::with_store(
                input_op,
                projections,
                output_types,
                Arc::new(grafeo_core::graph::NullGraphStore) as Arc<dyn GraphStoreSearch>,
            ));
        }

        let physical_keys: Vec<SortKey> = sort
            .keys
            .iter()
            .map(|key| {
                let col_idx = resolve_expression(&key.expression, &variable_columns)?;
                Ok(SortKey {
                    column: col_idx,
                    direction: match key.order {
                        SortOrder::Ascending => SortDirection::Ascending,
                        SortOrder::Descending => SortDirection::Descending,
                    },
                    null_order: NullOrder::NullsLast,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let operator = Box::new(SortOperator::new(input_op, physical_keys, types.clone()));
        Ok((operator, columns, types))
    }

    /// Plans a PROJECT operator.
    ///
    /// Projects only the requested columns from the input.
    fn plan_project(
        &self,
        project: &crate::query::plan::ProjectOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        let (input_op, input_columns, input_types) = self.plan_operator(&project.input)?;

        // Build mapping from variable name to column index
        let variable_columns: HashMap<String, usize> = input_columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        let mut projections = Vec::new();
        let mut output_columns = Vec::new();
        let mut output_types = Vec::new();

        for proj in &project.projections {
            match &proj.expression {
                LogicalExpression::Variable(name) => {
                    if let Some(&col_idx) = variable_columns.get(name) {
                        projections.push(RdfProjectExpr::Column(col_idx));
                        output_columns.push(proj.alias.clone().unwrap_or_else(|| name.clone()));
                        output_types.push(input_types[col_idx].clone());
                    } else {
                        return Err(Error::Internal(format!(
                            "Variable '{}' not found in input columns",
                            name
                        )));
                    }
                }
                LogicalExpression::Literal(value) => {
                    projections.push(RdfProjectExpr::Constant(value.clone()));
                    output_columns.push(proj.alias.clone().unwrap_or_else(|| format!("{value}")));
                    output_types.push(LogicalType::Any);
                }
                expr => {
                    // Convert complex expressions (function calls, arithmetic, etc.)
                    // to physical filter expressions and evaluate them in the projection.
                    let filter_expr = convert_filter_expression(expr)?;
                    projections.push(RdfProjectExpr::Expression {
                        expr: filter_expr,
                        variable_columns: variable_columns.clone(),
                    });
                    output_columns.push(proj.alias.clone().unwrap_or_else(|| format!("{expr:?}")));
                    output_types.push(LogicalType::Any);
                }
            }
        }

        // If no projections were extracted, just return the input as-is
        if projections.is_empty() {
            return Ok((input_op, input_columns, input_types));
        }

        // Use RdfProjectOperator which delegates expression evaluation to
        // RdfExpressionPredicate, giving access to SPARQL functions (STRLEN,
        // UCASE, LCASE, etc.) that the generic ProjectOperator does not know.
        let operator: Box<dyn Operator> = Box::new(RdfProjectOperator::new(
            input_op,
            projections,
            output_types.clone(),
        ));
        Ok((operator, output_columns, output_types))
    }

    /// Plans a BIND operator.
    ///
    /// BIND adds a computed column to each row by evaluating an expression.
    /// For example: `BIND (CONCAT(?name, " (age ", STR(?age), ")") AS ?label)`
    fn plan_bind(
        &self,
        bind: &BindOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        let (input_op, input_columns, mut input_types) = self.plan_operator(&bind.input)?;

        // Build variable-to-column mapping for expression evaluation
        let variable_columns: HashMap<String, usize> = input_columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        // Convert the BIND expression to a FilterExpression
        let filter_expr = convert_filter_expression(&bind.expression)?;

        // Build output columns: all input columns + the new BIND variable
        let mut output_columns = input_columns;
        output_columns.push(bind.variable.clone());
        input_types.push(LogicalType::Any);

        let operator = Box::new(RdfBindOperator::new(
            input_op,
            filter_expr,
            variable_columns,
        ));
        Ok((operator, output_columns, input_types))
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

        let variable_columns: HashMap<String, usize> = input_columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        let operator = Box::new(ConstructOperator::new(
            input_op,
            construct.templates.clone(),
            variable_columns,
        ));

        let columns = vec![
            "subject".to_string(),
            "predicate".to_string(),
            "object".to_string(),
        ];
        let types = vec![LogicalType::String; 3];
        Ok((operator, columns, types))
    }

    /// Plans an AGGREGATE operator.
    fn plan_aggregate(
        &self,
        agg: &AggregateOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        // COUNT(*) fast-path: when the aggregate is a single COUNT(*) with no
        // GROUP BY, no DISTINCT, no HAVING, and the input is a fully-unbound
        // TripleScan, short-circuit to store.len() in O(1).
        if let Some(result) = self.try_count_fast_path(agg) {
            return Ok(result);
        }

        use grafeo_core::execution::operators::AggregateExpr as PhysicalAggregateExpr;

        let (mut input_op, input_columns, input_types) = self.plan_operator(&agg.input)?;

        let mut variable_columns: HashMap<String, usize> = input_columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        // Pre-project complex expressions in group-by keys and aggregate arguments
        let mut expression_projections: Vec<(FilterExpression, String)> = Vec::new();
        let mut next_col_idx = input_columns.len();

        // Group-by expressions (Labels, Type, FunctionCall, etc.)
        for expr in &agg.group_by {
            match expr {
                LogicalExpression::Variable(_) => {}
                _ => {
                    let col_name = resolved_column_name(expr);
                    if !variable_columns.contains_key(&col_name) {
                        let filter_expr = convert_filter_expression(expr)?;
                        expression_projections.push((filter_expr, col_name.clone()));
                        variable_columns.insert(col_name, next_col_idx);
                        next_col_idx += 1;
                    }
                }
            }
        }

        // Aggregate argument expressions
        for agg_expr in &agg.aggregates {
            for expr_opt in [&agg_expr.expression, &agg_expr.expression2] {
                let Some(expr) = expr_opt else { continue };
                match expr {
                    LogicalExpression::Variable(_) => {}
                    _ => {
                        let col_name = resolved_column_name(expr);
                        if !variable_columns.contains_key(&col_name) {
                            let filter_expr = convert_filter_expression(expr)?;
                            expression_projections.push((filter_expr, col_name.clone()));
                            variable_columns.insert(col_name, next_col_idx);
                            next_col_idx += 1;
                        }
                    }
                }
            }
        }

        if !expression_projections.is_empty() {
            let mut projections: Vec<ProjectExpr> =
                (0..input_columns.len()).map(ProjectExpr::Column).collect();
            let mut output_types: Vec<LogicalType> = input_types;

            for (filter_expr, _col_name) in &expression_projections {
                projections.push(ProjectExpr::Expression {
                    expr: filter_expr.clone(),
                    variable_columns: variable_columns.clone(),
                });
                output_types.push(LogicalType::Any); // computed expressions may produce non-string types
            }

            input_op = Box::new(ProjectOperator::with_store(
                input_op,
                projections,
                output_types,
                Arc::new(grafeo_core::graph::NullGraphStore) as Arc<dyn GraphStoreSearch>,
            ));
        }

        let group_columns: Vec<usize> = agg
            .group_by
            .iter()
            .map(|expr| resolve_expression(expr, &variable_columns))
            .collect::<Result<Vec<_>>>()?;

        let physical_aggregates: Vec<PhysicalAggregateExpr> = agg
            .aggregates
            .iter()
            .map(|agg_expr| {
                let column = agg_expr
                    .expression
                    .as_ref()
                    .map(|e| resolve_expression(e, &variable_columns))
                    .transpose()?;

                let column2 = agg_expr
                    .expression2
                    .as_ref()
                    .map(|e| resolve_expression(e, &variable_columns))
                    .transpose()?;

                Ok(PhysicalAggregateExpr {
                    function: convert_aggregate_function(agg_expr.function),
                    column,
                    column2,
                    distinct: agg_expr.distinct,
                    alias: agg_expr.alias.clone(),
                    percentile: agg_expr.percentile,
                    separator: agg_expr.separator.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let mut output_schema = Vec::new();
        let mut output_columns = Vec::new();

        for expr in &agg.group_by {
            output_schema.push(LogicalType::String);
            output_columns.push(expression_to_string(expr));
        }

        for agg_expr in &agg.aggregates {
            // For RDF, numeric values are strings that get converted to floats
            // So SUM should also output Float64 (since SumFloat returns Float64)
            let result_type = match agg_expr.function {
                LogicalAggregateFunction::Count | LogicalAggregateFunction::CountNonNull => {
                    LogicalType::Int64
                }
                LogicalAggregateFunction::Sum => LogicalType::Float64,
                LogicalAggregateFunction::Avg => LogicalType::Float64,
                _ => LogicalType::String,
            };
            output_schema.push(result_type);
            output_columns.push(
                agg_expr
                    .alias
                    .clone()
                    .unwrap_or_else(|| format!("{:?}(...)", agg_expr.function).to_lowercase()),
            );
        }

        let agg_schema = output_schema.clone();
        let mut operator: Box<dyn Operator> = if group_columns.is_empty() {
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

        // Apply HAVING clause filter if present
        if let Some(having_expr) = &agg.having {
            let having_var_columns: HashMap<String, usize> = output_columns
                .iter()
                .enumerate()
                .map(|(i, name)| (name.clone(), i))
                .collect();

            let filter_expr = convert_filter_expression(having_expr)?;
            let predicate = RdfExpressionPredicate::new(filter_expr, having_var_columns);
            operator = Box::new(FilterOperator::new(operator, Box::new(predicate)));
        }

        Ok((operator, output_columns, output_schema))
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

        Some(Self::make_constant_int64(count, alias))
    }

    /// Resolves the count for a simple triple scan pattern, or returns `None`
    /// if the pattern is too complex for a fast-path.
    fn count_for_scan(&self, scan: &TripleScanOp) -> Option<i64> {
        let s_var = scan.subject.as_variable().is_some();
        let p_var = scan.predicate.as_variable().is_some();
        let o_var = scan.object.as_variable().is_some();

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
                let stats = self.store.get_or_collect_statistics();
                if let Some(pred_stats) = stats.get_predicate(pred_iri) {
                    // reason: predicate triple count will not exceed i64::MAX
                    #[allow(clippy::cast_possible_wrap)]
                    return Some(pred_stats.triple_count as i64);
                }
            }
            return None;
        }

        // Not a fast-path pattern
        None
    }

    /// Creates a constant Int64 single-row result.
    fn make_constant_int64(
        value: i64,
        column_name: String,
    ) -> (Box<dyn Operator>, Vec<String>, Vec<LogicalType>) {
        let mut chunk = DataChunk::with_capacity(&[LogicalType::Int64], 1);
        chunk
            .column_mut(0)
            .expect("just created")
            .push_value(Value::Int64(value));
        chunk.set_count(1);
        let operator = Box::new(ConstantOperator::new(chunk));
        (operator, vec![column_name], vec![LogicalType::Int64])
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
        let (left_op, left_columns, left_types) = self.plan_operator(&join.left)?;
        let (right_op, right_columns, right_types) = self.plan_operator(&join.right)?;

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
    fn plan_left_join(
        &self,
        join: &LeftJoinOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        use crate::query::planner::common;
        let (left_op, left_columns, left_types) = self.plan_operator(&join.left)?;
        let (right_op, right_columns, right_types) = self.plan_operator(&join.right)?;
        Ok(common::build_left_join(
            left_op,
            right_op,
            &left_columns,
            &right_columns,
            &left_types,
            &right_types,
        ))
    }

    /// Plans an ANTI JOIN operator (for SPARQL MINUS).
    fn plan_anti_join(
        &self,
        join: &AntiJoinOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        use crate::query::planner::common;
        let (left_op, left_columns, left_types) = self.plan_operator(&join.left)?;
        let (right_op, right_columns, _right_types) = self.plan_operator(&join.right)?;
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
        if !self.needs_companion_columns.get()
            && let Some(result) = self.try_leapfrog_ring(mwj)
        {
            return result;
        }

        // Plan all inputs and estimate cardinalities
        let mut planned: Vec<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>, f64)> = Vec::new();
        for input in &mwj.inputs {
            let (op, cols, types) = self.plan_operator(input)?;
            let card = estimate_operator_cardinality(input, &self.store).unwrap_or(1000.0);
            planned.push((op, cols, types, card));
        }

        // Sort by cardinality (smallest first) so we build on smaller inputs
        planned.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal));

        // Fold left-to-right with pairwise hash joins
        let (mut current_op, mut current_cols, mut current_types, mut current_card) =
            planned.remove(0);
        for (right_op, right_cols, right_types, right_card) in planned {
            let cardinalities = Some((current_card, right_card));
            let (joined_op, joined_cols, joined_types) = common::build_inner_join(
                current_op,
                right_op,
                &current_cols,
                &right_cols,
                &current_types,
                &right_types,
                cardinalities,
            );
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

        // Build output columns and types
        let columns = all_vars.clone();
        let types = vec![LogicalType::String; columns.len()];

        let operator = Box::new(RdfLeapfrogOperator::new(ring, annotated, columns.clone()));
        Some(Ok((operator, columns, types)))
    }

    /// Plans a UNION operator.
    fn plan_union(
        &self,
        union: &crate::query::plan::UnionOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        if union.inputs.is_empty() {
            return Err(Error::Internal("Empty UNION".to_string()));
        }

        // For INSERT operations, we execute all operators in sequence
        let mut operators: Vec<Box<dyn Operator>> = Vec::new();
        let mut columns = Vec::new();
        let mut types = Vec::new();

        for (i, input) in union.inputs.iter().enumerate() {
            let (op, cols, tys) = self.plan_operator(input)?;
            operators.push(op);
            if i == 0 {
                columns = cols;
                types = tys;
            }
        }

        if operators.len() == 1 {
            return Ok((
                operators
                    .into_iter()
                    .next()
                    .expect("single-element iterator"),
                columns,
                types,
            ));
        }

        // Create a chain operator that executes all operators in sequence
        let operator = Box::new(RdfUnionOperator::new(operators));
        Ok((operator, columns, types))
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
                    Arc::clone(&self.store),
                    input_op,
                    TripleOperands {
                        subject: insert.subject.clone(),
                        predicate: insert.predicate.clone(),
                        object: insert.object.clone(),
                        column_map,
                    },
                    #[cfg(feature = "wal")]
                    self.wal.clone(),
                    #[cfg(feature = "cdc")]
                    self.cdc_log.clone(),
                    #[cfg(feature = "cdc")]
                    self.cdc_epoch,
                ));

                return Ok((operator, Vec::new(), Vec::new()));
            }
        }

        // Direct insertion with concrete terms
        let subject = self.component_to_term(&insert.subject)?;
        let predicate = self.component_to_term(&insert.predicate)?;
        let object = self.component_to_term(&insert.object)?;

        let triple = Triple::new(subject, predicate, object);
        let operator = Box::new(RdfInsertTripleOperator::new(
            Arc::clone(&self.store),
            triple,
            insert.graph.clone(),
            self.transaction_id,
            #[cfg(feature = "wal")]
            self.wal.clone(),
            #[cfg(feature = "cdc")]
            self.cdc_log.clone(),
            #[cfg(feature = "cdc")]
            self.cdc_epoch,
        ));

        // Insert operations don't output columns
        Ok((operator, Vec::new(), Vec::new()))
    }

    /// Converts a TripleComponent to an RDF Term.
    fn component_to_term(&self, component: &TripleComponent) -> Result<Term> {
        match component {
            TripleComponent::Iri(iri) => Ok(Term::Iri(iri.clone().into())),
            TripleComponent::Literal(value) => {
                let lit = match value {
                    Value::String(s) => Literal::simple(s.to_string()),
                    Value::Int64(n) => Literal::integer(*n),
                    Value::Float64(f) => {
                        Literal::typed(f.to_string(), "http://www.w3.org/2001/XMLSchema#double")
                    }
                    Value::Bool(b) => {
                        Literal::typed(b.to_string(), "http://www.w3.org/2001/XMLSchema#boolean")
                    }
                    Value::Date(d) => {
                        Literal::typed(d.to_string(), "http://www.w3.org/2001/XMLSchema#date")
                    }
                    Value::Time(t) => {
                        Literal::typed(t.to_string(), "http://www.w3.org/2001/XMLSchema#time")
                    }
                    Value::Timestamp(ts) => {
                        Literal::typed(ts.to_string(), "http://www.w3.org/2001/XMLSchema#dateTime")
                    }
                    Value::ZonedDatetime(zdt) => {
                        Literal::typed(zdt.to_string(), "http://www.w3.org/2001/XMLSchema#dateTime")
                    }
                    Value::Duration(dur) => {
                        Literal::typed(dur.to_string(), "http://www.w3.org/2001/XMLSchema#duration")
                    }
                    _ => Literal::simple(format!("{:?}", value)),
                };
                Ok(Term::Literal(lit))
            }
            TripleComponent::LangLiteral { value, lang } => {
                Ok(Term::lang_literal(value.clone(), lang.clone()))
            }
            TripleComponent::BlankNode(label) => Ok(Term::blank(label.clone())),
            TripleComponent::Variable(name) => {
                // Variables in INSERT DATA should have been bound
                Err(Error::Internal(format!(
                    "Unbound variable '{}' in INSERT DATA",
                    name
                )))
            }
        }
    }

    /// Plans a DELETE TRIPLE operator.
    fn plan_delete_triple(
        &self,
        delete: &DeleteTripleOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        // Check if this is a pattern-based delete (has variables in the template)
        let has_variables = matches!(&delete.subject, TripleComponent::Variable(_))
            || matches!(&delete.predicate, TripleComponent::Variable(_))
            || matches!(&delete.object, TripleComponent::Variable(_));

        if has_variables {
            // Pattern-based deletion: need to query first, then delete each match
            if let Some(ref input) = delete.input {
                let (input_op, input_columns, _input_types) = self.plan_operator(input)?;

                // Build column index map for variable substitution
                let column_map: HashMap<String, usize> = input_columns
                    .iter()
                    .enumerate()
                    .map(|(i, name)| (name.clone(), i))
                    .collect();

                let operator = Box::new(RdfDeletePatternOperator::new(
                    Arc::clone(&self.store),
                    input_op,
                    TripleOperands {
                        subject: delete.subject.clone(),
                        predicate: delete.predicate.clone(),
                        object: delete.object.clone(),
                        column_map,
                    },
                    #[cfg(feature = "wal")]
                    self.wal.clone(),
                    #[cfg(feature = "cdc")]
                    self.cdc_log.clone(),
                    #[cfg(feature = "cdc")]
                    self.cdc_epoch,
                ));

                return Ok((operator, Vec::new(), Vec::new()));
            }
        }

        // Direct deletion with concrete terms
        let subject = self.component_to_term(&delete.subject)?;
        let predicate = self.component_to_term(&delete.predicate)?;
        let object = self.component_to_term(&delete.object)?;

        let triple = Triple::new(subject, predicate, object);
        let operator = Box::new(RdfDeleteTripleOperator::new(
            Arc::clone(&self.store),
            triple,
            delete.graph.clone(),
            self.transaction_id,
            #[cfg(feature = "wal")]
            self.wal.clone(),
            #[cfg(feature = "cdc")]
            self.cdc_log.clone(),
            #[cfg(feature = "cdc")]
            self.cdc_epoch,
        ));

        Ok((operator, Vec::new(), Vec::new()))
    }

    /// Plans a CLEAR GRAPH operator.
    fn plan_clear_graph(
        &self,
        clear: &ClearGraphOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        let operator = Box::new(RdfClearGraphOperator::new(
            Arc::clone(&self.store),
            clear.graph.clone(),
            clear.silent,
            #[cfg(feature = "wal")]
            self.wal.clone(),
        ));
        Ok((operator, Vec::new(), Vec::new()))
    }

    /// Plans a CREATE GRAPH operator.
    fn plan_create_graph(
        &self,
        create: &CreateGraphOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        let operator = Box::new(RdfCreateGraphOperator::new(
            Arc::clone(&self.store),
            create.graph.clone(),
            create.silent,
            #[cfg(feature = "wal")]
            self.wal.clone(),
        ));
        Ok((operator, Vec::new(), Vec::new()))
    }

    /// Plans a DROP GRAPH operator.
    fn plan_drop_graph(
        &self,
        drop_op: &DropGraphOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        let operator = Box::new(RdfDropGraphOperator::new(
            Arc::clone(&self.store),
            drop_op.graph.clone(),
            drop_op.silent,
            #[cfg(feature = "wal")]
            self.wal.clone(),
        ));
        Ok((operator, Vec::new(), Vec::new()))
    }

    /// Plans a COPY graph operator.
    fn plan_copy_graph(
        &self,
        copy: &CopyGraphOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        let operator = Box::new(RdfCopyGraphOperator::new(
            Arc::clone(&self.store),
            copy.source.clone(),
            copy.destination.clone(),
            copy.silent,
        ));
        Ok((operator, Vec::new(), Vec::new()))
    }

    /// Plans a MOVE graph operator.
    fn plan_move_graph(
        &self,
        move_op: &MoveGraphOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        let operator = Box::new(RdfMoveGraphOperator::new(
            Arc::clone(&self.store),
            move_op.source.clone(),
            move_op.destination.clone(),
            move_op.silent,
        ));
        Ok((operator, Vec::new(), Vec::new()))
    }

    /// Plans an ADD graph operator.
    fn plan_add_graph(
        &self,
        add: &AddGraphOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>, Vec<LogicalType>)> {
        let operator = Box::new(RdfAddGraphOperator::new(
            Arc::clone(&self.store),
            add.source.clone(),
            add.destination.clone(),
            add.silent,
        ));
        Ok((operator, Vec::new(), Vec::new()))
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

        let operator = Box::new(RdfModifyOperator::new(
            Arc::clone(&self.store),
            where_op,
            modify.delete_templates.clone(),
            modify.insert_templates.clone(),
            column_map,
            #[cfg(feature = "cdc")]
            self.cdc_log.clone(),
            #[cfg(feature = "cdc")]
            self.cdc_epoch,
        ));

        Ok((operator, Vec::new(), Vec::new()))
    }
}

// ============================================================================
// RDF Insert Triple Operator
// ============================================================================

/// Operator that inserts a triple into the RDF store.
struct RdfInsertTripleOperator {
    store: Arc<RdfStore>,
    triple: Triple,
    graph_name: Option<String>,
    transaction_id: Option<TransactionId>,
    inserted: bool,
    #[cfg(feature = "wal")]
    wal: Option<Arc<RdfWal>>,
    #[cfg(feature = "cdc")]
    cdc_log: Option<Arc<crate::cdc::CdcLog>>,
    #[cfg(feature = "cdc")]
    cdc_epoch: grafeo_common::types::EpochId,
}

impl RdfInsertTripleOperator {
    fn new(
        store: Arc<RdfStore>,
        triple: Triple,
        graph_name: Option<String>,
        transaction_id: Option<TransactionId>,
        #[cfg(feature = "wal")] wal: Option<Arc<RdfWal>>,
        #[cfg(feature = "cdc")] cdc_log: Option<Arc<crate::cdc::CdcLog>>,
        #[cfg(feature = "cdc")] cdc_epoch: grafeo_common::types::EpochId,
    ) -> Self {
        Self {
            store,
            triple,
            graph_name,
            transaction_id,
            inserted: false,
            #[cfg(feature = "wal")]
            wal,
            #[cfg(feature = "cdc")]
            cdc_log,
            #[cfg(feature = "cdc")]
            cdc_epoch,
        }
    }
}

impl Operator for RdfInsertTripleOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.inserted {
            return Ok(None);
        }

        // Resolve target store: named graph or default
        let target = match &self.graph_name {
            Some(name) => self.store.graph_or_create(name),
            None => Arc::clone(&self.store),
        };

        // Insert the triple (buffered if in a transaction)
        if let Some(transaction_id) = self.transaction_id {
            target.insert_in_transaction(transaction_id, self.triple.clone());
        } else {
            target.insert(self.triple.clone());
        }

        #[cfg(feature = "wal")]
        log_rdf_wal(
            &self.wal,
            &grafeo_storage::wal::WalRecord::InsertRdfTriple {
                subject: term_to_wal(self.triple.subject()),
                predicate: term_to_wal(self.triple.predicate()),
                object: term_to_wal(self.triple.object()),
                graph: self.graph_name.clone(),
            },
        );

        #[cfg(feature = "cdc")]
        record_cdc_triple_insert(
            &self.cdc_log,
            self.triple.subject(),
            self.triple.predicate(),
            self.triple.object(),
            self.graph_name.as_deref(),
            self.cdc_epoch,
        );

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
    store: Arc<RdfStore>,
    input: Box<dyn Operator>,
    subject: TripleComponent,
    predicate: TripleComponent,
    object: TripleComponent,
    column_map: HashMap<String, usize>,
    done: bool,
    #[cfg(feature = "wal")]
    wal: Option<Arc<RdfWal>>,
    #[cfg(feature = "cdc")]
    cdc_log: Option<Arc<crate::cdc::CdcLog>>,
    #[cfg(feature = "cdc")]
    cdc_epoch: grafeo_common::types::EpochId,
}

impl RdfInsertPatternOperator {
    fn new(
        store: Arc<RdfStore>,
        input: Box<dyn Operator>,
        operands: TripleOperands,
        #[cfg(feature = "wal")] wal: Option<Arc<RdfWal>>,
        #[cfg(feature = "cdc")] cdc_log: Option<Arc<crate::cdc::CdcLog>>,
        #[cfg(feature = "cdc")] cdc_epoch: grafeo_common::types::EpochId,
    ) -> Self {
        Self {
            store,
            input,
            subject: operands.subject,
            predicate: operands.predicate,
            object: operands.object,
            column_map: operands.column_map,
            done: false,
            #[cfg(feature = "wal")]
            wal,
            #[cfg(feature = "cdc")]
            cdc_log,
            #[cfg(feature = "cdc")]
            cdc_epoch,
        }
    }

    fn resolve_component(
        &self,
        component: &TripleComponent,
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Term> {
        match component {
            TripleComponent::Iri(iri) => Some(Term::Iri(iri.clone().into())),
            TripleComponent::BlankNode(label) => Some(Term::blank(label.clone())),
            TripleComponent::Literal(value) => {
                let lit = match value {
                    Value::String(s) => Literal::simple(s.to_string()),
                    Value::Int64(n) => Literal::integer(*n),
                    Value::Float64(f) => {
                        Literal::typed(f.to_string(), "http://www.w3.org/2001/XMLSchema#double")
                    }
                    Value::Bool(b) => {
                        Literal::typed(b.to_string(), "http://www.w3.org/2001/XMLSchema#boolean")
                    }
                    _ => Literal::simple(format!("{:?}", value)),
                };
                Some(Term::Literal(lit))
            }
            TripleComponent::LangLiteral { value, lang } => {
                Some(Term::lang_literal(value.clone(), lang.clone()))
            }
            TripleComponent::Variable(name) => {
                // Remove the leading '?' if present
                let var_name = name.strip_prefix('?').unwrap_or(name);
                if let Some(&col_idx) = self.column_map.get(var_name)
                    && let Some(col) = chunk.column(col_idx)
                    && let Some(value) = col.get_value(row)
                {
                    return Self::value_to_term(&value);
                }
                None
            }
        }
    }

    fn value_to_term(value: &Value) -> Option<Term> {
        match value {
            Value::String(s) => {
                if s.starts_with("http://") || s.starts_with("https://") || s.starts_with("urn:") {
                    Some(Term::Iri(s.to_string().into()))
                } else {
                    Some(Term::Literal(Literal::simple(s.to_string())))
                }
            }
            Value::Int64(n) => Some(Term::Literal(Literal::integer(*n))),
            Value::Float64(f) => Some(Term::Literal(Literal::typed(
                f.to_string(),
                "http://www.w3.org/2001/XMLSchema#double",
            ))),
            Value::Bool(b) => Some(Term::Literal(Literal::typed(
                b.to_string(),
                "http://www.w3.org/2001/XMLSchema#boolean",
            ))),
            _ => None,
        }
    }
}

impl Operator for RdfInsertPatternOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.done {
            return Ok(None);
        }

        // Collect all triples to insert
        let mut triples_to_insert = Vec::new();

        while let Some(chunk) = self.input.next()? {
            for row in 0..chunk.row_count() {
                let subject = self.resolve_component(&self.subject, &chunk, row);
                let predicate = self.resolve_component(&self.predicate, &chunk, row);
                let object = self.resolve_component(&self.object, &chunk, row);

                if let (Some(s), Some(p), Some(o)) = (subject, predicate, object) {
                    triples_to_insert.push(Triple::new(s, p, o));
                }
            }
        }

        // Insert all collected triples
        for triple in &triples_to_insert {
            self.store.insert(triple.clone());
        }

        #[cfg(feature = "wal")]
        for triple in &triples_to_insert {
            log_rdf_wal(
                &self.wal,
                &grafeo_storage::wal::WalRecord::InsertRdfTriple {
                    subject: term_to_wal(triple.subject()),
                    predicate: term_to_wal(triple.predicate()),
                    object: term_to_wal(triple.object()),
                    graph: None,
                },
            );
        }

        #[cfg(feature = "cdc")]
        for triple in &triples_to_insert {
            record_cdc_triple_insert(
                &self.cdc_log,
                triple.subject(),
                triple.predicate(),
                triple.object(),
                None,
                self.cdc_epoch,
            );
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

/// Operator that deletes a triple from the RDF store.
struct RdfDeleteTripleOperator {
    store: Arc<RdfStore>,
    triple: Triple,
    graph_name: Option<String>,
    transaction_id: Option<TransactionId>,
    deleted: bool,
    #[cfg(feature = "wal")]
    wal: Option<Arc<RdfWal>>,
    #[cfg(feature = "cdc")]
    cdc_log: Option<Arc<crate::cdc::CdcLog>>,
    #[cfg(feature = "cdc")]
    cdc_epoch: grafeo_common::types::EpochId,
}

impl RdfDeleteTripleOperator {
    fn new(
        store: Arc<RdfStore>,
        triple: Triple,
        graph_name: Option<String>,
        transaction_id: Option<TransactionId>,
        #[cfg(feature = "wal")] wal: Option<Arc<RdfWal>>,
        #[cfg(feature = "cdc")] cdc_log: Option<Arc<crate::cdc::CdcLog>>,
        #[cfg(feature = "cdc")] cdc_epoch: grafeo_common::types::EpochId,
    ) -> Self {
        Self {
            store,
            triple,
            graph_name,
            transaction_id,
            deleted: false,
            #[cfg(feature = "wal")]
            wal,
            #[cfg(feature = "cdc")]
            cdc_log,
            #[cfg(feature = "cdc")]
            cdc_epoch,
        }
    }
}

impl Operator for RdfDeleteTripleOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.deleted {
            return Ok(None);
        }

        // Resolve target store: named graph or default
        let target = match &self.graph_name {
            Some(name) => self.store.graph_or_create(name),
            None => Arc::clone(&self.store),
        };

        // Delete the triple (buffered if in a transaction)
        if let Some(transaction_id) = self.transaction_id {
            target.remove_in_transaction(transaction_id, self.triple.clone());
        } else {
            target.remove(&self.triple);
        }

        #[cfg(feature = "wal")]
        log_rdf_wal(
            &self.wal,
            &grafeo_storage::wal::WalRecord::DeleteRdfTriple {
                subject: term_to_wal(self.triple.subject()),
                predicate: term_to_wal(self.triple.predicate()),
                object: term_to_wal(self.triple.object()),
                graph: self.graph_name.clone(),
            },
        );

        #[cfg(feature = "cdc")]
        record_cdc_triple_delete(
            &self.cdc_log,
            self.triple.subject(),
            self.triple.predicate(),
            self.triple.object(),
            self.graph_name.as_deref(),
            self.cdc_epoch,
        );

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
    store: Arc<RdfStore>,
    input: Box<dyn Operator>,
    subject: TripleComponent,
    predicate: TripleComponent,
    object: TripleComponent,
    column_map: HashMap<String, usize>,
    done: bool,
    #[cfg(feature = "wal")]
    wal: Option<Arc<RdfWal>>,
    #[cfg(feature = "cdc")]
    cdc_log: Option<Arc<crate::cdc::CdcLog>>,
    #[cfg(feature = "cdc")]
    cdc_epoch: grafeo_common::types::EpochId,
}

impl RdfDeletePatternOperator {
    fn new(
        store: Arc<RdfStore>,
        input: Box<dyn Operator>,
        operands: TripleOperands,
        #[cfg(feature = "wal")] wal: Option<Arc<RdfWal>>,
        #[cfg(feature = "cdc")] cdc_log: Option<Arc<crate::cdc::CdcLog>>,
        #[cfg(feature = "cdc")] cdc_epoch: grafeo_common::types::EpochId,
    ) -> Self {
        Self {
            store,
            input,
            subject: operands.subject,
            predicate: operands.predicate,
            object: operands.object,
            column_map: operands.column_map,
            done: false,
            #[cfg(feature = "wal")]
            wal,
            #[cfg(feature = "cdc")]
            cdc_log,
            #[cfg(feature = "cdc")]
            cdc_epoch,
        }
    }

    fn resolve_component(
        &self,
        component: &TripleComponent,
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Term> {
        match component {
            TripleComponent::Iri(iri) => Some(Term::Iri(iri.clone().into())),
            TripleComponent::BlankNode(label) => Some(Term::blank(label.clone())),
            TripleComponent::Literal(value) => {
                let lit = match value {
                    Value::String(s) => Literal::simple(s.to_string()),
                    Value::Int64(n) => Literal::integer(*n),
                    Value::Float64(f) => {
                        Literal::typed(f.to_string(), "http://www.w3.org/2001/XMLSchema#double")
                    }
                    Value::Bool(b) => {
                        Literal::typed(b.to_string(), "http://www.w3.org/2001/XMLSchema#boolean")
                    }
                    _ => Literal::simple(format!("{:?}", value)),
                };
                Some(Term::Literal(lit))
            }
            TripleComponent::LangLiteral { value, lang } => {
                Some(Term::lang_literal(value.clone(), lang.clone()))
            }
            TripleComponent::Variable(name) => {
                // Remove the leading '?' if present
                let var_name = name.strip_prefix('?').unwrap_or(name);
                if let Some(&col_idx) = self.column_map.get(var_name)
                    && let Some(col) = chunk.column(col_idx)
                    && let Some(value) = col.get_value(row)
                {
                    return Self::value_to_term(&value);
                }
                None
            }
        }
    }

    fn value_to_term(value: &Value) -> Option<Term> {
        match value {
            Value::String(s) => {
                if s.starts_with("http://") || s.starts_with("https://") || s.starts_with("urn:") {
                    Some(Term::Iri(s.to_string().into()))
                } else {
                    Some(Term::Literal(Literal::simple(s.to_string())))
                }
            }
            Value::Int64(n) => Some(Term::Literal(Literal::integer(*n))),
            Value::Float64(f) => Some(Term::Literal(Literal::typed(
                f.to_string(),
                "http://www.w3.org/2001/XMLSchema#double",
            ))),
            Value::Bool(b) => Some(Term::Literal(Literal::typed(
                b.to_string(),
                "http://www.w3.org/2001/XMLSchema#boolean",
            ))),
            _ => None,
        }
    }
}

impl Operator for RdfDeletePatternOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.done {
            return Ok(None);
        }

        // Collect all triples to delete
        let mut triples_to_delete = Vec::new();

        while let Some(chunk) = self.input.next()? {
            for row in 0..chunk.row_count() {
                let subject = self.resolve_component(&self.subject, &chunk, row);
                let predicate = self.resolve_component(&self.predicate, &chunk, row);
                let object = self.resolve_component(&self.object, &chunk, row);

                if let (Some(s), Some(p), Some(o)) = (subject, predicate, object) {
                    triples_to_delete.push(Triple::new(s, p, o));
                }
            }
        }

        // Delete all collected triples
        for triple in &triples_to_delete {
            self.store.remove(triple);
        }

        #[cfg(feature = "wal")]
        for triple in &triples_to_delete {
            log_rdf_wal(
                &self.wal,
                &grafeo_storage::wal::WalRecord::DeleteRdfTriple {
                    subject: term_to_wal(triple.subject()),
                    predicate: term_to_wal(triple.predicate()),
                    object: term_to_wal(triple.object()),
                    graph: None,
                },
            );
        }

        #[cfg(feature = "cdc")]
        for triple in &triples_to_delete {
            record_cdc_triple_delete(
                &self.cdc_log,
                triple.subject(),
                triple.predicate(),
                triple.object(),
                None,
                self.cdc_epoch,
            );
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
// RDF Clear Graph Operator
// ============================================================================

/// Operator that clears triples from a graph in the RDF store.
struct RdfClearGraphOperator {
    store: Arc<RdfStore>,
    graph: Option<String>,
    cleared: bool,
    #[cfg(feature = "wal")]
    wal: Option<Arc<RdfWal>>,
}

impl RdfClearGraphOperator {
    fn new(
        store: Arc<RdfStore>,
        graph: Option<String>,
        _silent: bool,
        #[cfg(feature = "wal")] wal: Option<Arc<RdfWal>>,
    ) -> Self {
        Self {
            store,
            graph,
            cleared: false,
            #[cfg(feature = "wal")]
            wal,
        }
    }
}

impl Operator for RdfClearGraphOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.cleared {
            return Ok(None);
        }

        // Empty string is the sentinel for CLEAR ALL (both default and all named graphs)
        if self.graph.as_deref() == Some("") {
            self.store.clear();
            self.store.clear_all_named();
        } else {
            self.store.clear_graph(self.graph.as_deref());
        }

        #[cfg(feature = "wal")]
        log_rdf_wal(
            &self.wal,
            &grafeo_storage::wal::WalRecord::ClearRdfGraph {
                graph: self.graph.clone(),
            },
        );

        self.cleared = true;

        Ok(None)
    }

    fn reset(&mut self) {
        self.cleared = false;
    }

    fn name(&self) -> &'static str {
        "RdfClearGraph"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

// ============================================================================
// RDF CREATE/DROP Graph Operators
// ============================================================================

/// Operator that creates a named graph.
struct RdfCreateGraphOperator {
    store: Arc<RdfStore>,
    graph: String,
    silent: bool,
    done: bool,
    #[cfg(feature = "wal")]
    wal: Option<Arc<RdfWal>>,
}

impl RdfCreateGraphOperator {
    fn new(
        store: Arc<RdfStore>,
        graph: String,
        silent: bool,
        #[cfg(feature = "wal")] wal: Option<Arc<RdfWal>>,
    ) -> Self {
        Self {
            store,
            graph,
            silent,
            done: false,
            #[cfg(feature = "wal")]
            wal,
        }
    }
}

impl Operator for RdfCreateGraphOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.done {
            return Ok(None);
        }
        self.done = true;
        let created = self.store.create_graph(&self.graph);
        if !created && !self.silent {
            return Err(OperatorError::Execution(format!(
                "Graph <{}> already exists",
                self.graph
            )));
        }
        #[cfg(feature = "wal")]
        if created {
            log_rdf_wal(
                &self.wal,
                &grafeo_storage::wal::WalRecord::CreateRdfGraph {
                    name: self.graph.clone(),
                },
            );
        }
        Ok(None)
    }

    fn reset(&mut self) {
        self.done = false;
    }

    fn name(&self) -> &'static str {
        "RdfCreateGraph"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// Operator that drops a named graph.
struct RdfDropGraphOperator {
    store: Arc<RdfStore>,
    graph: Option<String>,
    silent: bool,
    done: bool,
    #[cfg(feature = "wal")]
    wal: Option<Arc<RdfWal>>,
}

impl RdfDropGraphOperator {
    fn new(
        store: Arc<RdfStore>,
        graph: Option<String>,
        silent: bool,
        #[cfg(feature = "wal")] wal: Option<Arc<RdfWal>>,
    ) -> Self {
        Self {
            store,
            graph,
            silent,
            done: false,
            #[cfg(feature = "wal")]
            wal,
        }
    }
}

impl Operator for RdfDropGraphOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.done {
            return Ok(None);
        }
        self.done = true;
        match &self.graph {
            None => {
                // DROP DEFAULT: clear the default graph
                self.store.clear();
            }
            Some(name) => {
                let dropped = self.store.drop_graph(name);
                if !dropped && !self.silent {
                    return Err(OperatorError::Execution(format!(
                        "Graph <{name}> does not exist"
                    )));
                }
            }
        }
        #[cfg(feature = "wal")]
        log_rdf_wal(
            &self.wal,
            &grafeo_storage::wal::WalRecord::DropRdfGraph {
                name: self.graph.clone(),
            },
        );
        Ok(None)
    }

    fn reset(&mut self) {
        self.done = false;
    }

    fn name(&self) -> &'static str {
        "RdfDropGraph"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

// ============================================================================
// RDF COPY/MOVE/ADD Graph Operators
// ============================================================================

/// Operator that copies all triples from one graph to another.
struct RdfCopyGraphOperator {
    store: Arc<RdfStore>,
    source: Option<String>,
    destination: Option<String>,
    silent: bool,
    done: bool,
}

impl RdfCopyGraphOperator {
    fn new(
        store: Arc<RdfStore>,
        source: Option<String>,
        destination: Option<String>,
        silent: bool,
    ) -> Self {
        Self {
            store,
            source,
            destination,
            silent,
            done: false,
        }
    }
}

impl Operator for RdfCopyGraphOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.done {
            return Ok(None);
        }
        self.done = true;

        // Check source exists (unless silent)
        if !self.silent
            && let Some(ref name) = self.source
            && self.store.graph(name).is_none()
        {
            return Err(OperatorError::Execution(format!(
                "Source graph <{name}> does not exist"
            )));
        }

        self.store
            .copy_graph(self.source.as_deref(), self.destination.as_deref());
        Ok(None)
    }

    fn reset(&mut self) {
        self.done = false;
    }

    fn name(&self) -> &'static str {
        "RdfCopyGraph"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// Operator that moves all triples from one graph to another.
struct RdfMoveGraphOperator {
    store: Arc<RdfStore>,
    source: Option<String>,
    destination: Option<String>,
    silent: bool,
    done: bool,
}

impl RdfMoveGraphOperator {
    fn new(
        store: Arc<RdfStore>,
        source: Option<String>,
        destination: Option<String>,
        silent: bool,
    ) -> Self {
        Self {
            store,
            source,
            destination,
            silent,
            done: false,
        }
    }
}

impl Operator for RdfMoveGraphOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.done {
            return Ok(None);
        }
        self.done = true;

        // Check source exists (unless silent)
        if !self.silent
            && let Some(ref name) = self.source
            && self.store.graph(name).is_none()
        {
            return Err(OperatorError::Execution(format!(
                "Source graph <{name}> does not exist"
            )));
        }

        self.store
            .move_graph(self.source.as_deref(), self.destination.as_deref());
        Ok(None)
    }

    fn reset(&mut self) {
        self.done = false;
    }

    fn name(&self) -> &'static str {
        "RdfMoveGraph"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// Operator that adds (merges) all triples from one graph into another.
struct RdfAddGraphOperator {
    store: Arc<RdfStore>,
    source: Option<String>,
    destination: Option<String>,
    silent: bool,
    done: bool,
}

impl RdfAddGraphOperator {
    fn new(
        store: Arc<RdfStore>,
        source: Option<String>,
        destination: Option<String>,
        silent: bool,
    ) -> Self {
        Self {
            store,
            source,
            destination,
            silent,
            done: false,
        }
    }
}

impl Operator for RdfAddGraphOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.done {
            return Ok(None);
        }
        self.done = true;

        // Check source exists (unless silent)
        if !self.silent
            && let Some(ref name) = self.source
            && self.store.graph(name).is_none()
        {
            return Err(OperatorError::Execution(format!(
                "Source graph <{name}> does not exist"
            )));
        }

        self.store
            .add_graph(self.source.as_deref(), self.destination.as_deref());
        Ok(None)
    }

    fn reset(&mut self) {
        self.done = false;
    }

    fn name(&self) -> &'static str {
        "RdfAddGraph"
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
    store: Arc<RdfStore>,
    input: Box<dyn Operator>,
    delete_templates: Vec<TripleTemplate>,
    insert_templates: Vec<TripleTemplate>,
    column_map: HashMap<String, usize>,
    done: bool,
    #[cfg(feature = "cdc")]
    cdc_log: Option<Arc<crate::cdc::CdcLog>>,
    #[cfg(feature = "cdc")]
    cdc_epoch: grafeo_common::types::EpochId,
}

impl RdfModifyOperator {
    fn new(
        store: Arc<RdfStore>,
        input: Box<dyn Operator>,
        delete_templates: Vec<TripleTemplate>,
        insert_templates: Vec<TripleTemplate>,
        column_map: HashMap<String, usize>,
        #[cfg(feature = "cdc")] cdc_log: Option<Arc<crate::cdc::CdcLog>>,
        #[cfg(feature = "cdc")] cdc_epoch: grafeo_common::types::EpochId,
    ) -> Self {
        Self {
            store,
            input,
            delete_templates,
            insert_templates,
            column_map,
            done: false,
            #[cfg(feature = "cdc")]
            cdc_log,
            #[cfg(feature = "cdc")]
            cdc_epoch,
        }
    }

    fn resolve_component(
        &self,
        component: &TripleComponent,
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Term> {
        match component {
            TripleComponent::Iri(iri) => Some(Term::Iri(iri.clone().into())),
            TripleComponent::BlankNode(label) => Some(Term::blank(label.clone())),
            TripleComponent::Literal(value) => {
                let lit = match value {
                    Value::String(s) => Literal::simple(s.to_string()),
                    Value::Int64(n) => Literal::integer(*n),
                    Value::Float64(f) => {
                        Literal::typed(f.to_string(), "http://www.w3.org/2001/XMLSchema#double")
                    }
                    Value::Bool(b) => {
                        Literal::typed(b.to_string(), "http://www.w3.org/2001/XMLSchema#boolean")
                    }
                    _ => Literal::simple(format!("{:?}", value)),
                };
                Some(Term::Literal(lit))
            }
            TripleComponent::LangLiteral { value, lang } => {
                Some(Term::lang_literal(value.clone(), lang.clone()))
            }
            TripleComponent::Variable(name) => {
                let var_name = name.strip_prefix('?').unwrap_or(name);
                if let Some(&col_idx) = self.column_map.get(var_name)
                    && let Some(col) = chunk.column(col_idx)
                    && let Some(value) = col.get_value(row)
                {
                    return Self::value_to_term(&value);
                }
                None
            }
        }
    }

    fn value_to_term(value: &Value) -> Option<Term> {
        match value {
            Value::String(s) => {
                if s.starts_with("http://") || s.starts_with("https://") || s.starts_with("urn:") {
                    Some(Term::Iri(s.to_string().into()))
                } else {
                    Some(Term::Literal(Literal::simple(s.to_string())))
                }
            }
            Value::Int64(n) => Some(Term::Literal(Literal::integer(*n))),
            Value::Float64(f) => Some(Term::Literal(Literal::typed(
                f.to_string(),
                "http://www.w3.org/2001/XMLSchema#double",
            ))),
            Value::Bool(b) => Some(Term::Literal(Literal::typed(
                b.to_string(),
                "http://www.w3.org/2001/XMLSchema#boolean",
            ))),
            _ => None,
        }
    }
}

impl Operator for RdfModifyOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.done {
            return Ok(None);
        }

        // Step 1: Collect all bindings from WHERE clause (before any modifications)
        let mut bindings: Vec<(DataChunk, usize)> = Vec::new();
        while let Some(chunk) = self.input.next()? {
            for row in 0..chunk.row_count() {
                bindings.push((chunk.clone(), row));
            }
        }

        // Step 2: Apply DELETE templates using collected bindings.
        //
        // RDF typed literals (xsd:integer, xsd:boolean, ...) are stored with
        // their datatype, but the WHERE clause returns them as plain strings
        // (Value::String). Reconstructing the Term from the string loses the
        // type, so a direct `store.remove(&triple)` would fail to match.
        //
        // Fix: when the exact triple isn't found, query the store with the
        // subject+predicate pattern and remove any triple whose literal value
        // matches the bound string. This handles the type mismatch without
        // changing the column type system.
        for template in &self.delete_templates {
            for (chunk, row) in &bindings {
                let subject = self.resolve_component(&template.subject, chunk, *row);
                let predicate = self.resolve_component(&template.predicate, chunk, *row);
                let object = self.resolve_component(&template.object, chunk, *row);

                if let (Some(s), Some(p), Some(o)) = (subject, predicate, object) {
                    let triple = Triple::new(s.clone(), p.clone(), o.clone());
                    if !self.store.remove(&triple) {
                        // Exact match failed: the object may be a plain string
                        // whose stored form is a typed literal (e.g. "1" vs
                        // xsd:integer "1"). Query by subject+predicate and
                        // match on the literal's lexical value.
                        if let Term::Literal(target_lit) = &o {
                            let pattern = TriplePattern {
                                subject: Some(s.clone()),
                                predicate: Some(p.clone()),
                                object: None,
                            };
                            let matching: Vec<_> = self
                                .store
                                .find(&pattern)
                                .into_iter()
                                .filter(|t| {
                                    if let Term::Literal(lit) = t.object() {
                                        // Only match typed literals whose lexical
                                        // value equals the target. Plain strings
                                        // (xsd:string) should have matched exactly.
                                        lit.value() == target_lit.value()
                                            && lit.datatype() != Literal::XSD_STRING
                                    } else {
                                        false
                                    }
                                })
                                .collect();
                            for matched in matching {
                                #[cfg(feature = "cdc")]
                                record_cdc_triple_delete(
                                    &self.cdc_log,
                                    matched.subject(),
                                    matched.predicate(),
                                    matched.object(),
                                    None,
                                    self.cdc_epoch,
                                );
                                self.store.remove(&matched);
                            }
                            continue;
                        }
                    }
                    #[cfg(feature = "cdc")]
                    record_cdc_triple_delete(
                        &self.cdc_log,
                        triple.subject(),
                        triple.predicate(),
                        triple.object(),
                        None,
                        self.cdc_epoch,
                    );
                }
            }
        }

        // Step 3: Apply INSERT templates using the SAME bindings
        for template in &self.insert_templates {
            for (chunk, row) in &bindings {
                let subject = self.resolve_component(&template.subject, chunk, *row);
                let predicate = self.resolve_component(&template.predicate, chunk, *row);
                let object = self.resolve_component(&template.object, chunk, *row);

                if let (Some(s), Some(p), Some(o)) = (subject, predicate, object) {
                    let triple = Triple::new(s, p, o);
                    #[cfg(feature = "cdc")]
                    record_cdc_triple_insert(
                        &self.cdc_log,
                        triple.subject(),
                        triple.predicate(),
                        triple.object(),
                        None,
                        self.cdc_epoch,
                    );
                    self.store.insert(triple);
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

/// Operator that executes multiple operators in sequence.
/// Used for UNION of INSERT operations.
struct RdfUnionOperator {
    operators: Vec<Box<dyn Operator>>,
    current_idx: usize,
}

impl RdfUnionOperator {
    fn new(operators: Vec<Box<dyn Operator>>) -> Self {
        Self {
            operators,
            current_idx: 0,
        }
    }
}

impl Operator for RdfUnionOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        // Execute all operators
        while self.current_idx < self.operators.len() {
            let op = &mut self.operators[self.current_idx];
            match op.next()? {
                Some(chunk) => return Ok(Some(chunk)),
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

/// Operator that appends a computed column to each row.
///
/// Used for SPARQL BIND expressions (e.g., `BIND (CONCAT(?x, ?y) AS ?z)`).
/// Evaluates an expression using `RdfExpressionPredicate` and appends the
/// result as a new column in the output chunk.
struct RdfBindOperator {
    /// Child operator providing input rows.
    child: Box<dyn Operator>,
    /// Expression to evaluate per row.
    expression: FilterExpression,
    /// Variable name to column index mapping for expression evaluation.
    variable_columns: HashMap<String, usize>,
}

impl RdfBindOperator {
    fn new(
        child: Box<dyn Operator>,
        expression: FilterExpression,
        variable_columns: HashMap<String, usize>,
    ) -> Self {
        Self {
            child,
            expression,
            variable_columns,
        }
    }
}

impl Operator for RdfBindOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        let Some(input) = self.child.next()? else {
            return Ok(None);
        };

        let input_col_count = input.column_count();
        let row_count = input.row_count();

        // Build output schema: preserve input column types + one extra column for BIND result
        let mut output_types: Vec<LogicalType> = Vec::with_capacity(input_col_count + 1);
        for col_idx in 0..input_col_count {
            if let Some(col) = input.column(col_idx) {
                output_types.push(col.logical_type());
            } else {
                output_types.push(LogicalType::Any);
            }
        }
        output_types.push(LogicalType::Any);

        let mut output = DataChunk::with_capacity(&output_types, row_count);

        // Copy existing columns
        for col_idx in 0..input_col_count {
            let output_col = output
                .column_mut(col_idx)
                .expect("column exists: index within schema bounds");
            if let Some(input_col) = input.column(col_idx) {
                for row in input.selected_indices() {
                    if let Some(value) = input_col.get_value(row) {
                        output_col.push_value(value);
                    } else {
                        output_col.push_value(Value::Null);
                    }
                }
            }
        }

        // Evaluate expression for each row and append as new column
        let evaluator =
            RdfExpressionPredicate::new(self.expression.clone(), self.variable_columns.clone());
        let bind_col = output
            .column_mut(input_col_count)
            .expect("column exists: bind column is last in schema");
        for row in input.selected_indices() {
            let value = evaluator.eval(&input, row).unwrap_or(Value::Null);
            bind_col.push_value(value);
        }

        output.set_count(row_count);
        Ok(Some(output))
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

/// Projection variant for expression evaluation.
enum RdfProjectExpr {
    /// Reference to an input column.
    Column(usize),
    /// A constant value.
    Constant(Value),
    /// Full expression evaluation using `RdfExpressionPredicate`.
    Expression {
        /// The filter expression to evaluate.
        expr: FilterExpression,
        /// Variable name to column index mapping.
        variable_columns: HashMap<String, usize>,
    },
}

/// An RDF-specific project operator that uses `RdfExpressionPredicate` for
/// expression evaluation, giving access to SPARQL functions (STRLEN, UCASE,
/// LCASE, etc.) that the generic `ProjectOperator` does not support.
struct RdfProjectOperator {
    /// Child operator providing input rows.
    child: Box<dyn Operator>,
    /// Projection expressions.
    projections: Vec<RdfProjectExpr>,
    /// Output column types.
    output_types: Vec<LogicalType>,
}

impl RdfProjectOperator {
    fn new(
        child: Box<dyn Operator>,
        projections: Vec<RdfProjectExpr>,
        output_types: Vec<LogicalType>,
    ) -> Self {
        assert_eq!(projections.len(), output_types.len());
        Self {
            child,
            projections,
            output_types,
        }
    }
}

impl Operator for RdfProjectOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        let Some(input) = self.child.next()? else {
            return Ok(None);
        };

        let mut output = DataChunk::with_capacity(&self.output_types, input.row_count());

        for (i, proj) in self.projections.iter().enumerate() {
            let output_col = output
                .column_mut(i)
                .expect("column exists: index matches projection schema");

            match proj {
                RdfProjectExpr::Column(col_idx) => {
                    let input_col = input.column(*col_idx).ok_or_else(|| {
                        OperatorError::ColumnNotFound(format!("Column {col_idx}"))
                    })?;
                    for row in input.selected_indices() {
                        if let Some(value) = input_col.get_value(row) {
                            output_col.push_value(value);
                        } else {
                            output_col.push_value(Value::Null);
                        }
                    }
                }
                RdfProjectExpr::Constant(value) => {
                    for _ in input.selected_indices() {
                        output_col.push_value(value.clone());
                    }
                }
                RdfProjectExpr::Expression {
                    expr,
                    variable_columns,
                } => {
                    let evaluator =
                        RdfExpressionPredicate::new(expr.clone(), variable_columns.clone());
                    for row in input.selected_indices() {
                        let value = evaluator.eval(&input, row).unwrap_or(Value::Null);
                        output_col.push_value(value);
                    }
                }
            }
        }

        output.set_count(input.row_count());
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

// ============================================================================
// RDF Triple Scan Operator
// ============================================================================

/// Graph resolution context for a triple scan.
///
/// Groups the graph IRI, scan-all flag, and SPARQL dataset restriction
/// to keep the operator constructor argument count manageable.
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
}

#[cfg(feature = "ring-index")]
impl RdfLeapfrogOperator {
    fn new(
        ring: Arc<grafeo_core::index::ring::TripleRing>,
        annotated_patterns: Vec<grafeo_core::index::ring::AnnotatedPattern>,
        output_columns: Vec<String>,
    ) -> Self {
        Self {
            ring,
            annotated_patterns,
            output_columns,
            results: Vec::new(),
            position: 0,
            executed: false,
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

    /// Extracts the value for a variable from a set of matched triples.
    fn resolve_variable(&self, var: &str, matched_triples: &[Triple]) -> Option<String> {
        for (i, ap) in self.annotated_patterns.iter().enumerate() {
            if let Some(triple) = matched_triples.get(i) {
                if ap.subject_var.as_deref() == Some(var) {
                    return Some(term_to_string(triple.subject()));
                }
                if ap.predicate_var.as_deref() == Some(var) {
                    return Some(term_to_string(triple.predicate()));
                }
                if ap.object_var.as_deref() == Some(var) {
                    return Some(term_to_string(triple.object()));
                }
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

        let schema = vec![LogicalType::String; col_count];
        let mut chunk = DataChunk::with_capacity(&schema, batch_size);

        for i in self.position..end {
            let matched = &self.results[i];
            for (col_idx, var_name) in self.output_columns.iter().enumerate() {
                if let Some(col) = chunk.column_mut(col_idx) {
                    if let Some(val) = self.resolve_variable(var_name, matched) {
                        col.push_string(val);
                    } else {
                        col.push_value(Value::Null);
                    }
                }
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

/// CONSTRUCT operator: instantiates triple templates from variable bindings.
///
/// For each input row, substitutes variables in each template triple to produce
/// (subject, predicate, object) output rows. Skips template triples where a
/// variable has no binding (unbound variables produce no triple).
struct ConstructOperator {
    input: Box<dyn Operator>,
    templates: Vec<TripleTemplate>,
    variable_columns: HashMap<String, usize>,
}

impl ConstructOperator {
    fn new(
        input: Box<dyn Operator>,
        templates: Vec<TripleTemplate>,
        variable_columns: HashMap<String, usize>,
    ) -> Self {
        Self {
            input,
            templates,
            variable_columns,
        }
    }

    /// Resolves a `TripleComponent` to a string using the current row bindings.
    fn resolve_component(
        &self,
        component: &TripleComponent,
        chunk: &DataChunk,
        row: usize,
    ) -> Option<String> {
        match component {
            TripleComponent::Variable(name) => {
                let col_idx = *self.variable_columns.get(name)?;
                let col = chunk.column(col_idx)?;
                let val = col.get_value(row)?;
                if val.is_null() {
                    None
                } else {
                    Some(val.to_string())
                }
            }
            TripleComponent::Iri(iri) => Some(iri.clone()),
            TripleComponent::Literal(val) => Some(val.to_string()),
            TripleComponent::LangLiteral { value, lang } => Some(format!("\"{value}\"@{lang}")),
            TripleComponent::BlankNode(label) => Some(format!("_:{label}")),
        }
    }
}

impl Operator for ConstructOperator {
    fn next(&mut self) -> grafeo_core::execution::operators::OperatorResult {
        loop {
            let Some(input_chunk) = self.input.next()? else {
                return Ok(None);
            };

            let row_count = input_chunk.row_count();
            if row_count == 0 {
                continue;
            }

            // Each input row can produce up to templates.len() output rows
            let max_output = row_count * self.templates.len();
            let schema = vec![LogicalType::String; 3];
            let mut output = DataChunk::with_capacity(&schema, max_output);
            let mut actual_count = 0;

            for row in 0..row_count {
                for template in &self.templates {
                    let Some(subject) =
                        self.resolve_component(&template.subject, &input_chunk, row)
                    else {
                        continue;
                    };
                    let Some(predicate) =
                        self.resolve_component(&template.predicate, &input_chunk, row)
                    else {
                        continue;
                    };
                    let Some(object) = self.resolve_component(&template.object, &input_chunk, row)
                    else {
                        continue;
                    };

                    if let Some(col) = output.column_mut(0) {
                        col.push_string(subject);
                    }
                    if let Some(col) = output.column_mut(1) {
                        col.push_string(predicate);
                    }
                    if let Some(col) = output.column_mut(2) {
                        col.push_string(object);
                    }
                    actual_count += 1;
                }
            }

            if actual_count > 0 {
                output.set_count(actual_count);
                return Ok(Some(output));
            }
        }
    }

    fn reset(&mut self) {
        self.input.reset();
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

/// Resolves dictionary-encoded Int64 term IDs back to String values.
///
/// Placed at the result boundary (just before output) so that all intermediate
/// operators (joins, aggregates, sorts) work with compact Int64 keys.
struct DictResolveOperator {
    input: Box<dyn Operator>,
    dictionary: Arc<grafeo_core::graph::rdf::TermDictionary>,
    /// Column indices that carry encoded term IDs and need resolution.
    encoded_col_indices: Vec<usize>,
}

impl DictResolveOperator {
    fn new(
        input: Box<dyn Operator>,
        dictionary: Arc<grafeo_core::graph::rdf::TermDictionary>,
        encoded_col_indices: Vec<usize>,
    ) -> Self {
        Self {
            input,
            dictionary,
            encoded_col_indices,
        }
    }
}

impl Operator for DictResolveOperator {
    fn next(&mut self) -> grafeo_core::execution::operators::OperatorResult {
        let Some(chunk) = self.input.next()? else {
            return Ok(None);
        };

        if self.encoded_col_indices.is_empty() {
            return Ok(Some(chunk));
        }

        let row_count = chunk.row_count();
        let col_count = chunk.column_count();

        // Build output schema: replace Int64 encoded columns with String
        let mut schema = Vec::with_capacity(col_count);
        for col_idx in 0..col_count {
            if self.encoded_col_indices.contains(&col_idx) {
                schema.push(LogicalType::String);
            } else if let Some(col) = chunk.column(col_idx) {
                schema.push(col.data_type().clone());
            } else {
                schema.push(LogicalType::String);
            }
        }

        let mut out = DataChunk::with_capacity(&schema, row_count);

        for row in 0..row_count {
            for col_idx in 0..col_count {
                let Some(in_col) = chunk.column(col_idx) else {
                    continue;
                };
                let Some(out_col) = out.column_mut(col_idx) else {
                    continue;
                };

                if self.encoded_col_indices.contains(&col_idx) {
                    // Resolve term ID to string
                    if in_col.is_null(row) {
                        out_col.push_value(Value::Null);
                    } else if let Some(term_id) = in_col.get_int64(row) {
                        // reason: term IDs are assigned sequentially from 0 and fit u32
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        let tid = term_id as u32;
                        if let Some(term) = self.dictionary.get_term(tid) {
                            out_col.push_string(term_to_string(term));
                        } else {
                            out_col.push_value(Value::Null);
                        }
                    } else {
                        out_col.push_value(Value::Null);
                    }
                } else {
                    // Pass through non-encoded columns
                    if let Some(val) = in_col.get_value(row) {
                        out_col.push_value(val);
                    } else {
                        out_col.push_value(Value::Null);
                    }
                }
            }
        }

        out.set_count(row_count);
        Ok(Some(out))
    }

    fn reset(&mut self) {
        self.input.reset();
    }

    fn name(&self) -> &'static str {
        "DictResolve"
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
    /// Whether to emit a companion language-tag column after the object column.
    emit_companion_columns: bool,
    /// Whether to also emit a companion datatype column (only when DATATYPE() is used).
    emit_datatype_column: bool,
    /// Chunk size for batching.
    chunk_size: usize,
    /// Cached matching triples with graph names (lazily populated).
    triples: Option<Vec<(Option<String>, Arc<Triple>)>>,
    /// Current position in the triples.
    position: usize,
    /// Optional term dictionary for dictionary-encoded output. When present,
    /// S/P/O variable columns emit Int64 term IDs instead of String values.
    dictionary: Option<Arc<grafeo_core::graph::rdf::TermDictionary>>,
}

impl RdfTripleScanOperator {
    fn new(
        store: Arc<RdfStore>,
        pattern: TriplePattern,
        output_mask: [bool; 4],
        chunk_size: usize,
        graph_context: GraphContext,
        emit_companion_columns: bool,
        emit_datatype_column: bool,
    ) -> Self {
        Self {
            store,
            pattern,
            output_mask,
            graph_context,
            emit_companion_columns,
            emit_datatype_column,
            chunk_size,
            triples: None,
            position: 0,
            dictionary: None,
        }
    }

    /// Enables dictionary-encoded output for S/P/O columns.
    ///
    /// Infrastructure for Phase 4 (Ring Index) which will automatically enable
    /// dictionary encoding when the Ring provides native integer-keyed iteration.
    #[allow(dead_code)]
    fn with_dictionary(mut self, dict: Arc<grafeo_core::graph::rdf::TermDictionary>) -> Self {
        self.dictionary = Some(dict);
        self
    }

    /// Lazily load matching triples on first access.
    ///
    /// Respects SPARQL dataset clauses (FROM / FROM NAMED):
    /// - FROM: basic patterns (no graph context) scan the union of specified named graphs.
    /// - FROM NAMED: GRAPH patterns only iterate listed named graphs.
    fn ensure_triples(&mut self) {
        if self.triples.is_none() {
            let ctx = &self.graph_context;
            self.triples = Some(if ctx.scan_all_graphs {
                // GRAPH ?var: scan named graphs (restricted by FROM NAMED if present)
                if let Some(ref ds) = ctx.dataset {
                    if !ds.named_graphs.is_empty() {
                        // FROM NAMED restricts which named graphs are visible
                        let graph_refs: Vec<&str> =
                            ds.named_graphs.iter().map(String::as_str).collect();
                        self.store.find_in_graphs(&self.pattern, Some(&graph_refs))
                    } else {
                        // No FROM NAMED: all graphs visible
                        self.store.find_in_graphs(&self.pattern, Some(&[]))
                    }
                } else {
                    // No dataset restriction: scan all graphs
                    self.store.find_in_graphs(&self.pattern, Some(&[]))
                }
            } else if let Some(ref graph_iri) = ctx.graph {
                // GRAPH <iri>: scan specific named graph (restricted by FROM NAMED if present)
                if let Some(ref ds) = ctx.dataset {
                    if !ds.named_graphs.is_empty()
                        && !ds.named_graphs.iter().any(|g| g == graph_iri)
                    {
                        // The specified graph is not in the FROM NAMED list: empty result
                        Vec::new()
                    } else {
                        self.store
                            .graph(graph_iri)
                            .map(|g| {
                                g.find(&self.pattern)
                                    .into_iter()
                                    .map(|t| (Some(graph_iri.clone()), t))
                                    .collect()
                            })
                            .unwrap_or_default()
                    }
                } else {
                    self.store
                        .graph(graph_iri)
                        .map(|g| {
                            g.find(&self.pattern)
                                .into_iter()
                                .map(|t| (Some(graph_iri.clone()), t))
                                .collect()
                        })
                        .unwrap_or_default()
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
                        let mut results = self
                            .store
                            .find_in_graphs(&self.pattern, Some(&unique_graphs));
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
        }
    }

    /// Count how many output columns we have.
    fn output_column_count(&self) -> usize {
        let base = self.output_mask.iter().filter(|&&b| b).count();
        if self.emit_companion_columns {
            base + if self.emit_datatype_column { 2 } else { 1 }
        } else {
            base
        }
    }

    /// Builds the output schema. In dictionary mode, S/P/O variable columns
    /// are Int64 (term IDs), companion and graph columns remain String.
    fn build_output_schema(&self, col_count: usize, dict_mode: bool) -> Vec<LogicalType> {
        if !dict_mode {
            return vec![LogicalType::String; col_count];
        }
        let mut schema = Vec::with_capacity(col_count);
        // S, P, O variable columns get Int64
        for i in 0..3 {
            if self.output_mask[i] {
                schema.push(LogicalType::Int64);
            }
        }
        // Companion columns (lang, datatype) are always String
        if self.output_mask[2] && self.emit_companion_columns {
            schema.push(LogicalType::String); // lang
            if self.emit_datatype_column {
                schema.push(LogicalType::String); // datatype
            }
        }
        // Graph column is always String
        if self.output_mask[3] {
            schema.push(LogicalType::String);
        }
        schema
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
        let col_count = self.output_column_count();

        // Create output schema: Int64 for dictionary-encoded S/P/O, String otherwise
        let dict_mode = self.dictionary.is_some();
        let schema: Vec<LogicalType> = self.build_output_schema(col_count, dict_mode);
        let mut chunk = DataChunk::with_capacity(&schema, batch_size);

        // Fill the chunk
        for i in self.position..end {
            let (ref graph_name, ref triple) = triples[i];
            let mut col_idx = 0;

            if self.output_mask[0] {
                // Subject
                if let Some(col) = chunk.column_mut(col_idx) {
                    if let Some(ref dict) = self.dictionary {
                        if let Some(id) = dict.get_id(triple.subject()) {
                            col.push_value(Value::Int64(i64::from(id)));
                        } else {
                            col.push_value(Value::Null);
                        }
                    } else {
                        col.push_string(term_to_string(triple.subject()));
                    }
                }
                col_idx += 1;
            }
            if self.output_mask[1] {
                // Predicate
                if let Some(col) = chunk.column_mut(col_idx) {
                    if let Some(ref dict) = self.dictionary {
                        if let Some(id) = dict.get_id(triple.predicate()) {
                            col.push_value(Value::Int64(i64::from(id)));
                        } else {
                            col.push_value(Value::Null);
                        }
                    } else {
                        col.push_string(term_to_string(triple.predicate()));
                    }
                }
                col_idx += 1;
            }
            if self.output_mask[2] {
                // Object: dictionary encoding applies here too
                if let Some(col) = chunk.column_mut(col_idx) {
                    if let Some(ref dict) = self.dictionary {
                        if let Some(id) = dict.get_id(triple.object()) {
                            col.push_value(Value::Int64(i64::from(id)));
                        } else {
                            col.push_value(Value::Null);
                        }
                    } else {
                        push_term_value(col, triple.object());
                    }
                }
                col_idx += 1;

                // Companion language-tag and datatype columns (always String)
                if self.emit_companion_columns {
                    if let Some(col) = chunk.column_mut(col_idx) {
                        let lang_tag = match triple.object() {
                            Term::Literal(lit) => lit.language().unwrap_or("").to_string(),
                            _ => String::new(),
                        };
                        col.push_string(lang_tag);
                    }
                    col_idx += 1;

                    if self.emit_datatype_column {
                        if let Some(col) = chunk.column_mut(col_idx) {
                            let datatype = match triple.object() {
                                Term::Literal(lit) => lit.datatype().to_string(),
                                _ => String::new(),
                            };
                            col.push_string(datatype);
                        }
                        col_idx += 1;
                    }
                }
            }
            if self.output_mask[3] {
                // Graph (always String)
                if let Some(col) = chunk.column_mut(col_idx) {
                    match graph_name {
                        Some(name) => col.push_string(name.clone()),
                        None => col.push_value(Value::Null),
                    }
                }
            }
        }

        chunk.set_count(batch_size);
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
// RDF Expression Predicate
// ============================================================================

/// Expression predicate for RDF queries.
///
/// Unlike the LPG predicate, this doesn't need a store reference because
/// RDF values are already materialized in the DataChunk columns.
struct RdfExpressionPredicate {
    expression: FilterExpression,
    variable_columns: HashMap<String, usize>,
}

impl RdfExpressionPredicate {
    fn new(expression: FilterExpression, variable_columns: HashMap<String, usize>) -> Self {
        Self {
            expression,
            variable_columns,
        }
    }

    fn eval(&self, chunk: &DataChunk, row: usize) -> Option<Value> {
        self.eval_expr(&self.expression, chunk, row)
    }

    fn eval_expr(&self, expr: &FilterExpression, chunk: &DataChunk, row: usize) -> Option<Value> {
        match expr {
            FilterExpression::Literal(v) => Some(v.clone()),
            FilterExpression::Variable(name) => {
                let col_idx = *self.variable_columns.get(name)?;
                chunk.column(col_idx)?.get_value(row)
            }
            FilterExpression::Property { variable, .. } => {
                // For RDF, treat property access as variable access
                let col_idx = *self.variable_columns.get(variable)?;
                chunk.column(col_idx)?.get_value(row)
            }
            FilterExpression::Binary { left, op, right } => {
                // IN operator: evaluate right side as a list, then check membership
                if *op == BinaryFilterOp::In {
                    let left_val = self.eval_expr(left, chunk, row)?;
                    let right_val = self.eval_expr(right, chunk, row)?;
                    return match right_val {
                        Value::List(items) => {
                            if left_val.is_null() {
                                return Some(Value::Null);
                            }
                            let mut has_null = false;
                            for item in items.iter() {
                                if item.is_null() {
                                    has_null = true;
                                } else if rdf_values_equal(&left_val, item) {
                                    return Some(Value::Bool(true));
                                }
                            }
                            if has_null {
                                Some(Value::Null)
                            } else {
                                Some(Value::Bool(false))
                            }
                        }
                        _ => None,
                    };
                }
                let left_val = self.eval_expr(left, chunk, row)?;
                let right_val = self.eval_expr(right, chunk, row)?;
                self.eval_binary_op(&left_val, *op, &right_val)
            }
            FilterExpression::Unary { op, operand } => {
                let val = self.eval_expr(operand, chunk, row);
                self.eval_unary_op(*op, val)
            }
            FilterExpression::Id(var)
            | FilterExpression::Labels(var)
            | FilterExpression::Type(var) => {
                // Treat Id/Labels/Type access as variable lookup for RDF
                let col_idx = *self.variable_columns.get(var)?;
                chunk.column(col_idx)?.get_value(row)
            }
            FilterExpression::FunctionCall { name, args } => {
                self.eval_function_call(name, args, chunk, row)
            }
            FilterExpression::List(items) => {
                let values: Vec<Value> = items
                    .iter()
                    .filter_map(|item| self.eval_expr(item, chunk, row))
                    .collect();
                Some(Value::List(values.into()))
            }
            // These expression types are not commonly used in RDF FILTER clauses
            FilterExpression::Case { .. }
            | FilterExpression::Map(_)
            | FilterExpression::IndexAccess { .. }
            | FilterExpression::SliceAccess { .. }
            | FilterExpression::ListComprehension { .. }
            | FilterExpression::ListPredicate { .. }
            | FilterExpression::ExistsSubquery { .. }
            | FilterExpression::CountSubquery { .. }
            | FilterExpression::Reduce { .. } => None,
            _ => None,
        }
    }

    fn eval_binary_op(&self, left: &Value, op: BinaryFilterOp, right: &Value) -> Option<Value> {
        match op {
            BinaryFilterOp::And => Some(Value::Bool(left.as_bool()? && right.as_bool()?)),
            BinaryFilterOp::Or => Some(Value::Bool(left.as_bool()? || right.as_bool()?)),
            BinaryFilterOp::Xor => Some(Value::Bool(left.as_bool()? != right.as_bool()?)),
            BinaryFilterOp::Eq => compare_values(left, right, |o| o.is_eq()),
            BinaryFilterOp::Ne => compare_values(left, right, |o| o.is_ne()),
            BinaryFilterOp::Lt => compare_values(left, right, |o| o.is_lt()),
            BinaryFilterOp::Le => compare_values(left, right, |o| o.is_le()),
            BinaryFilterOp::Gt => compare_values(left, right, |o| o.is_gt()),
            BinaryFilterOp::Ge => compare_values(left, right, |o| o.is_ge()),
            BinaryFilterOp::Add => {
                // Helper to convert to f64 for arithmetic
                fn to_f64(v: &Value) -> Option<f64> {
                    match v {
                        Value::Int64(i) => Some(*i as f64),
                        Value::Float64(f) => Some(*f),
                        Value::String(s) => s.parse::<f64>().ok(),
                        _ => None,
                    }
                }
                match (left, right) {
                    (Value::Int64(l), Value::Int64(r)) => Some(Value::Int64(l + r)),
                    (Value::Float64(l), Value::Float64(r)) => Some(Value::Float64(l + r)),
                    (Value::Int64(l), Value::Float64(r)) => Some(Value::Float64(*l as f64 + r)),
                    (Value::Float64(l), Value::Int64(r)) => Some(Value::Float64(l + *r as f64)),
                    // Handle string-to-numeric conversion for RDF
                    _ => {
                        let l = to_f64(left)?;
                        let r = to_f64(right)?;
                        Some(Value::Float64(l + r))
                    }
                }
            }
            BinaryFilterOp::Sub => {
                fn to_f64(v: &Value) -> Option<f64> {
                    match v {
                        Value::Int64(i) => Some(*i as f64),
                        Value::Float64(f) => Some(*f),
                        Value::String(s) => s.parse::<f64>().ok(),
                        _ => None,
                    }
                }
                match (left, right) {
                    (Value::Int64(l), Value::Int64(r)) => Some(Value::Int64(l - r)),
                    (Value::Float64(l), Value::Float64(r)) => Some(Value::Float64(l - r)),
                    (Value::Int64(l), Value::Float64(r)) => Some(Value::Float64(*l as f64 - r)),
                    (Value::Float64(l), Value::Int64(r)) => Some(Value::Float64(l - *r as f64)),
                    _ => {
                        let l = to_f64(left)?;
                        let r = to_f64(right)?;
                        Some(Value::Float64(l - r))
                    }
                }
            }
            BinaryFilterOp::Mul => {
                fn to_f64(v: &Value) -> Option<f64> {
                    match v {
                        Value::Int64(i) => Some(*i as f64),
                        Value::Float64(f) => Some(*f),
                        Value::String(s) => s.parse::<f64>().ok(),
                        _ => None,
                    }
                }
                match (left, right) {
                    (Value::Int64(l), Value::Int64(r)) => Some(Value::Int64(l * r)),
                    (Value::Float64(l), Value::Float64(r)) => Some(Value::Float64(l * r)),
                    (Value::Int64(l), Value::Float64(r)) => Some(Value::Float64(*l as f64 * r)),
                    (Value::Float64(l), Value::Int64(r)) => Some(Value::Float64(l * *r as f64)),
                    _ => {
                        let l = to_f64(left)?;
                        let r = to_f64(right)?;
                        Some(Value::Float64(l * r))
                    }
                }
            }
            BinaryFilterOp::Div => {
                fn to_f64(v: &Value) -> Option<f64> {
                    match v {
                        Value::Int64(i) => Some(*i as f64),
                        Value::Float64(f) => Some(*f),
                        Value::String(s) => s.parse::<f64>().ok(),
                        _ => None,
                    }
                }
                match (left, right) {
                    (Value::Int64(l), Value::Int64(r)) if *r != 0 => Some(Value::Int64(l / r)),
                    (Value::Float64(l), Value::Float64(r)) if *r != 0.0 => {
                        Some(Value::Float64(l / r))
                    }
                    (Value::Int64(l), Value::Float64(r)) if *r != 0.0 => {
                        Some(Value::Float64(*l as f64 / r))
                    }
                    (Value::Float64(l), Value::Int64(r)) if *r != 0 => {
                        Some(Value::Float64(l / *r as f64))
                    }
                    _ => {
                        let l = to_f64(left)?;
                        let r = to_f64(right)?;
                        if r != 0.0 {
                            Some(Value::Float64(l / r))
                        } else {
                            None
                        }
                    }
                }
            }
            BinaryFilterOp::Mod => match (left, right) {
                (Value::Int64(l), Value::Int64(r)) if *r != 0 => Some(Value::Int64(l % r)),
                _ => None,
            },
            BinaryFilterOp::Contains => match (left, right) {
                (Value::String(l), Value::String(r)) => Some(Value::Bool(l.contains(&**r))),
                _ => None,
            },
            BinaryFilterOp::StartsWith => match (left, right) {
                (Value::String(l), Value::String(r)) => Some(Value::Bool(l.starts_with(&**r))),
                _ => None,
            },
            BinaryFilterOp::EndsWith => match (left, right) {
                (Value::String(l), Value::String(r)) => Some(Value::Bool(l.ends_with(&**r))),
                _ => None,
            },
            BinaryFilterOp::In => {
                // Not implemented for RDF filter evaluation
                None
            }
            BinaryFilterOp::Regex => {
                // SPARQL REGEX(string, pattern) - returns true if string matches pattern
                match (left, right) {
                    (Value::String(text), Value::String(pattern)) => {
                        // Compile the regex pattern
                        match Regex::new(pattern) {
                            Ok(re) => Some(Value::Bool(re.is_match(text))),
                            Err(_) => None, // Invalid regex pattern
                        }
                    }
                    _ => None,
                }
            }
            BinaryFilterOp::Pow => {
                // Power operation
                match (left, right) {
                    (Value::Int64(base), Value::Int64(exp)) => {
                        Some(Value::Float64((*base as f64).powf(*exp as f64)))
                    }
                    (Value::Float64(base), Value::Float64(exp)) => {
                        Some(Value::Float64(base.powf(*exp)))
                    }
                    (Value::Int64(base), Value::Float64(exp)) => {
                        Some(Value::Float64((*base as f64).powf(*exp)))
                    }
                    (Value::Float64(base), Value::Int64(exp)) => {
                        Some(Value::Float64(base.powf(*exp as f64)))
                    }
                    _ => None,
                }
            }
            BinaryFilterOp::Like => {
                match (left, right) {
                    (Value::String(s), Value::String(pattern)) => {
                        // Convert SQL LIKE pattern to regex
                        let mut re_pat = String::with_capacity(pattern.len() + 4);
                        re_pat.push('^');
                        let mut chars = pattern.chars().peekable();
                        while let Some(ch) = chars.next() {
                            match ch {
                                '%' => re_pat.push_str(".*"),
                                '_' => re_pat.push('.'),
                                '\\' => {
                                    if let Some(next) = chars.next() {
                                        if ".+*?^${}()|[]\\".contains(next) {
                                            re_pat.push('\\');
                                        }
                                        re_pat.push(next);
                                    }
                                }
                                _ => {
                                    if ".+*?^${}()|[]\\".contains(ch) {
                                        re_pat.push('\\');
                                    }
                                    re_pat.push(ch);
                                }
                            }
                        }
                        re_pat.push('$');
                        match Regex::new(&re_pat) {
                            Ok(re) => Some(Value::Bool(re.is_match(s))),
                            Err(_) => None,
                        }
                    }
                    _ => None,
                }
            }
            BinaryFilterOp::Concat => match (left, right) {
                (Value::String(a), Value::String(b)) => {
                    let mut s = String::with_capacity(a.len() + b.len());
                    s.push_str(a);
                    s.push_str(b);
                    Some(Value::String(s.into()))
                }
                (Value::String(a), other) => {
                    let b = match other {
                        Value::Int64(i) => i.to_string(),
                        Value::Float64(f) => f.to_string(),
                        Value::Bool(b) => b.to_string(),
                        _ => return None,
                    };
                    let mut s = String::with_capacity(a.len() + b.len());
                    s.push_str(a);
                    s.push_str(&b);
                    Some(Value::String(s.into()))
                }
                (other, Value::String(b)) => {
                    let a = match other {
                        Value::Int64(i) => i.to_string(),
                        Value::Float64(f) => f.to_string(),
                        Value::Bool(bo) => bo.to_string(),
                        _ => return None,
                    };
                    let mut s = String::with_capacity(a.len() + b.len());
                    s.push_str(&a);
                    s.push_str(b);
                    Some(Value::String(s.into()))
                }
                _ => None,
            },
            _ => None,
        }
    }

    fn eval_unary_op(&self, op: UnaryFilterOp, val: Option<Value>) -> Option<Value> {
        match op {
            UnaryFilterOp::Not => Some(Value::Bool(!val?.as_bool()?)),
            UnaryFilterOp::IsNull => Some(Value::Bool(val.is_none())),
            UnaryFilterOp::IsNotNull => Some(Value::Bool(val.is_some())),
            UnaryFilterOp::Neg => match val? {
                Value::Int64(v) => Some(Value::Int64(-v)),
                Value::Float64(v) => Some(Value::Float64(-v)),
                _ => None,
            },
            _ => None,
        }
    }

    /// Evaluates SPARQL function calls.
    fn eval_function_call(
        &self,
        name: &str,
        args: &[FilterExpression],
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Value> {
        // Normalize function name to uppercase for case-insensitive matching
        let func_name = name.to_uppercase();

        match func_name.as_str() {
            // CONCAT - concatenate multiple strings
            "CONCAT" => {
                let mut result = String::new();
                for arg in args {
                    if let Some(Value::String(s)) = self.eval_expr(arg, chunk, row) {
                        result.push_str(&s);
                    } else if let Some(val) = self.eval_expr(arg, chunk, row) {
                        // Convert non-string values to string
                        result.push_str(&value_to_string(&val));
                    }
                }
                Some(Value::String(result.into()))
            }

            // REPLACE - replace occurrences of pattern with replacement
            "REPLACE" => {
                if args.len() < 3 {
                    return None;
                }
                let text = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                let pattern = match self.eval_expr(&args[1], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                let replacement = match self.eval_expr(&args[2], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };

                // Check if the pattern should be treated as regex (4th argument with 'r' flag)
                if args.len() >= 4
                    && let Some(Value::String(flags)) = self.eval_expr(&args[3], chunk, row)
                    && (flags.contains('r') || flags.contains('i'))
                {
                    // Regex-based replace
                    let regex_pattern = if flags.contains('i') {
                        format!("(?i){}", &pattern)
                    } else {
                        pattern.clone()
                    };
                    if let Ok(re) = Regex::new(&regex_pattern) {
                        return Some(Value::String(re.replace_all(&text, &replacement).into()));
                    }
                }

                // Simple string replace
                Some(Value::String(text.replace(&pattern, &replacement).into()))
            }

            // STRLEN - string length
            "STRLEN" => {
                if args.is_empty() {
                    return None;
                }
                let text = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                // reason: string length will not exceed i64::MAX
                #[allow(clippy::cast_possible_wrap)]
                Some(Value::Int64(text.chars().count() as i64))
            }

            // UCASE - uppercase
            "UCASE" | "UPPER" => {
                if args.is_empty() {
                    return None;
                }
                let text = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                Some(Value::String(text.to_uppercase().into()))
            }

            // LCASE - lowercase
            "LCASE" | "LOWER" => {
                if args.is_empty() {
                    return None;
                }
                let text = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                Some(Value::String(text.to_lowercase().into()))
            }

            // SUBSTR - substring extraction
            "SUBSTR" | "SUBSTRING" => {
                if args.len() < 2 {
                    return None;
                }
                let text = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                let start = match self.eval_expr(&args[1], chunk, row)? {
                    // reason: clamped to >= 0 by .max(1) - 1
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    Value::Int64(i) => (i.max(1) - 1) as usize, // SPARQL uses 1-based indexing
                    _ => return None,
                };
                let len = if args.len() >= 3 {
                    match self.eval_expr(&args[2], chunk, row)? {
                        // reason: clamped to >= 0 by .max(0)
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        Value::Int64(i) => Some(i.max(0) as usize),
                        _ => return None,
                    }
                } else {
                    None
                };

                let chars: Vec<char> = text.chars().collect();
                let substr: String = if let Some(len) = len {
                    chars.iter().skip(start).take(len).collect()
                } else {
                    chars.iter().skip(start).collect()
                };
                Some(Value::String(substr.into()))
            }

            // STRSTARTS - check if string starts with prefix
            "STRSTARTS" => {
                if args.len() < 2 {
                    return None;
                }
                let text = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                let prefix = match self.eval_expr(&args[1], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                Some(Value::Bool(text.starts_with(&prefix)))
            }

            // STRENDS - check if string ends with suffix
            "STRENDS" => {
                if args.len() < 2 {
                    return None;
                }
                let text = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                let suffix = match self.eval_expr(&args[1], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                Some(Value::Bool(text.ends_with(&suffix)))
            }

            // CONTAINS - check if string contains substring
            "CONTAINS" => {
                if args.len() < 2 {
                    return None;
                }
                let text = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                let pattern = match self.eval_expr(&args[1], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                Some(Value::Bool(text.contains(&pattern)))
            }

            // STRBEFORE - substring before pattern
            "STRBEFORE" => {
                if args.len() < 2 {
                    return None;
                }
                let text = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                let pattern = match self.eval_expr(&args[1], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                if let Some(pos) = text.find(&pattern) {
                    Some(Value::String(text[..pos].to_string().into()))
                } else {
                    Some(Value::String("".into()))
                }
            }

            // STRAFTER - substring after pattern
            "STRAFTER" => {
                if args.len() < 2 {
                    return None;
                }
                let text = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                let pattern = match self.eval_expr(&args[1], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                if let Some(pos) = text.find(&pattern) {
                    Some(Value::String(
                        text[pos + pattern.len()..].to_string().into(),
                    ))
                } else {
                    Some(Value::String("".into()))
                }
            }

            // ENCODE_FOR_URI - URL encode
            "ENCODE_FOR_URI" => {
                if args.is_empty() {
                    return None;
                }
                let text = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                // Simple URL encoding for common characters
                let encoded: String = text
                    .chars()
                    .map(|c| match c {
                        'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | '.' | '~' => c.to_string(),
                        _ => format!("%{:02X}", c as u32),
                    })
                    .collect();
                Some(Value::String(encoded.into()))
            }

            // COALESCE - return first non-null value
            "COALESCE" => {
                for arg in args {
                    if let Some(val) = self.eval_expr(arg, chunk, row)
                        && !matches!(val, Value::Null)
                    {
                        return Some(val);
                    }
                }
                None
            }

            // IF - conditional expression
            "IF" => {
                if args.len() < 3 {
                    return None;
                }
                let condition = self.eval_expr(&args[0], chunk, row)?;
                if condition.as_bool()? {
                    self.eval_expr(&args[1], chunk, row)
                } else {
                    self.eval_expr(&args[2], chunk, row)
                }
            }

            // BOUND - check if variable is bound
            "BOUND" => {
                if args.is_empty() {
                    return None;
                }
                // For variable arguments, check the validity bitmap directly so
                // that NULL entries from LEFT JOIN (OPTIONAL) are recognized as
                // "unbound" rather than "bound to Null".
                if let FilterExpression::Variable(var_name) = &args[0] {
                    if let Some(&col_idx) = self.variable_columns.get(var_name)
                        && let Some(col) = chunk.column(col_idx)
                    {
                        return Some(Value::Bool(!col.is_null(row)));
                    }
                    // Variable not in column map: unbound
                    return Some(Value::Bool(false));
                }
                // Non-variable arguments: fall back to expression evaluation
                let is_bound = self.eval_expr(&args[0], chunk, row).is_some();
                Some(Value::Bool(is_bound))
            }

            // STR - convert to string
            "STR" => {
                if args.is_empty() {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                Some(Value::String(value_to_string(&val).into()))
            }

            // ISIRI / ISURI - check if value is an IRI
            "ISIRI" | "ISURI" => {
                if args.is_empty() {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                if let Value::String(s) = val {
                    // Check if it looks like an IRI (starts with a scheme)
                    let is_iri = s.contains("://") || s.starts_with("urn:");
                    Some(Value::Bool(is_iri))
                } else {
                    Some(Value::Bool(false))
                }
            }

            // ISBLANK - check if value is a blank node
            "ISBLANK" => {
                if args.is_empty() {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                if let Value::String(s) = val {
                    Some(Value::Bool(s.starts_with("_:")))
                } else {
                    Some(Value::Bool(false))
                }
            }

            // ISLITERAL - check if value is a literal
            "ISLITERAL" => {
                if args.is_empty() {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                // In our model, non-IRI strings and other values are literals
                match &val {
                    Value::String(s) => {
                        Some(Value::Bool(!s.contains("://") && !s.starts_with("_:")))
                    }
                    _ => Some(Value::Bool(true)),
                }
            }

            // ISNUMERIC - check if value is numeric
            "ISNUMERIC" => {
                if args.is_empty() {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                let is_numeric = matches!(val, Value::Int64(_) | Value::Float64(_))
                    || matches!(&val, Value::String(s) if s.parse::<f64>().is_ok());
                Some(Value::Bool(is_numeric))
            }

            // ABS - absolute value
            "ABS" => {
                if args.is_empty() {
                    return None;
                }
                match self.eval_expr(&args[0], chunk, row)? {
                    Value::Int64(v) => Some(Value::Int64(v.abs())),
                    Value::Float64(v) => Some(Value::Float64(v.abs())),
                    _ => None,
                }
            }

            // CEIL - ceiling
            "CEIL" => {
                if args.is_empty() {
                    return None;
                }
                match self.eval_expr(&args[0], chunk, row)? {
                    Value::Int64(v) => Some(Value::Int64(v)),
                    Value::Float64(v) => Some(Value::Float64(v.ceil())),
                    _ => None,
                }
            }

            // FLOOR - floor
            "FLOOR" => {
                if args.is_empty() {
                    return None;
                }
                match self.eval_expr(&args[0], chunk, row)? {
                    Value::Int64(v) => Some(Value::Int64(v)),
                    Value::Float64(v) => Some(Value::Float64(v.floor())),
                    _ => None,
                }
            }

            // ROUND - round to nearest integer
            "ROUND" => {
                if args.is_empty() {
                    return None;
                }
                match self.eval_expr(&args[0], chunk, row)? {
                    Value::Int64(v) => Some(Value::Int64(v)),
                    Value::Float64(v) => Some(Value::Float64(v.round())),
                    _ => None,
                }
            }

            // REGEX - regular expression matching
            "REGEX" => {
                if args.len() < 2 {
                    return None;
                }
                let text = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                let pattern = match self.eval_expr(&args[1], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    _ => return None,
                };
                // Optional flags argument (3rd arg): "i" for case-insensitive
                let regex_pattern = if args.len() >= 3
                    && let Some(Value::String(flags)) = self.eval_expr(&args[2], chunk, row)
                    && flags.contains('i')
                {
                    format!("(?i){pattern}")
                } else {
                    pattern
                };
                match Regex::new(&regex_pattern) {
                    Ok(re) => Some(Value::Bool(re.is_match(&text))),
                    Err(_) => None,
                }
            }

            // ================================================================
            // Date/Time Functions (SPARQL 1.1 Section 17.4.5)
            // ================================================================

            // NOW - current datetime
            "NOW" => {
                let ts = grafeo_common::types::Timestamp::now();
                Some(Value::Timestamp(ts))
            }

            // YEAR - extract year from date/datetime
            "YEAR" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                match val {
                    Value::Date(d) => Some(Value::Int64(i64::from(d.year()))),
                    Value::Timestamp(ts) => Some(Value::Int64(i64::from(ts.to_date().year()))),
                    Value::ZonedDatetime(zdt) => {
                        Some(Value::Int64(i64::from(zdt.to_local_date().year())))
                    }
                    Value::String(s) => {
                        // Try full dateTime parse first (handles "2024-06-15T10:30:45+02:00"),
                        // then fall back to date-only parse (handles "2024-06-15").
                        if let Some(zdt) = grafeo_common::types::ZonedDatetime::parse(&s) {
                            Some(Value::Int64(i64::from(zdt.to_local_date().year())))
                        } else {
                            parse_datetime_date(&s).map(|d| Value::Int64(i64::from(d.year())))
                        }
                    }
                    _ => None,
                }
            }

            // MONTH - extract month from date/datetime
            "MONTH" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                match val {
                    Value::Date(d) => Some(Value::Int64(i64::from(d.month()))),
                    Value::Timestamp(ts) => Some(Value::Int64(i64::from(ts.to_date().month()))),
                    Value::ZonedDatetime(zdt) => {
                        Some(Value::Int64(i64::from(zdt.to_local_date().month())))
                    }
                    Value::String(s) => {
                        if let Some(zdt) = grafeo_common::types::ZonedDatetime::parse(&s) {
                            Some(Value::Int64(i64::from(zdt.to_local_date().month())))
                        } else {
                            parse_datetime_date(&s).map(|d| Value::Int64(i64::from(d.month())))
                        }
                    }
                    _ => None,
                }
            }

            // DAY - extract day from date/datetime
            "DAY" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                match val {
                    Value::Date(d) => Some(Value::Int64(i64::from(d.day()))),
                    Value::Timestamp(ts) => Some(Value::Int64(i64::from(ts.to_date().day()))),
                    Value::ZonedDatetime(zdt) => {
                        Some(Value::Int64(i64::from(zdt.to_local_date().day())))
                    }
                    Value::String(s) => {
                        if let Some(zdt) = grafeo_common::types::ZonedDatetime::parse(&s) {
                            Some(Value::Int64(i64::from(zdt.to_local_date().day())))
                        } else {
                            parse_datetime_date(&s).map(|d| Value::Int64(i64::from(d.day())))
                        }
                    }
                    _ => None,
                }
            }

            // HOURS - extract hours from time/datetime
            "HOURS" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                match val {
                    Value::Time(t) => Some(Value::Int64(i64::from(t.hour()))),
                    Value::Timestamp(ts) => Some(Value::Int64(i64::from(ts.to_time().hour()))),
                    Value::ZonedDatetime(zdt) => {
                        Some(Value::Int64(i64::from(zdt.to_local_time().hour())))
                    }
                    Value::String(s) => {
                        if let Some(zdt) = grafeo_common::types::ZonedDatetime::parse(&s) {
                            Some(Value::Int64(i64::from(zdt.to_local_time().hour())))
                        } else {
                            parse_datetime_time(&s).map(|t| Value::Int64(i64::from(t.hour())))
                        }
                    }
                    _ => None,
                }
            }

            // MINUTES - extract minutes from time/datetime
            "MINUTES" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                match val {
                    Value::Time(t) => Some(Value::Int64(i64::from(t.minute()))),
                    Value::Timestamp(ts) => Some(Value::Int64(i64::from(ts.to_time().minute()))),
                    Value::ZonedDatetime(zdt) => {
                        Some(Value::Int64(i64::from(zdt.to_local_time().minute())))
                    }
                    Value::String(s) => {
                        if let Some(zdt) = grafeo_common::types::ZonedDatetime::parse(&s) {
                            Some(Value::Int64(i64::from(zdt.to_local_time().minute())))
                        } else {
                            parse_datetime_time(&s).map(|t| Value::Int64(i64::from(t.minute())))
                        }
                    }
                    _ => None,
                }
            }

            // SECONDS - extract seconds (with fractional) from time/datetime
            "SECONDS" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                let to_secs = |t: &grafeo_common::types::Time| {
                    f64::from(t.second()) + f64::from(t.nanosecond()) / 1_000_000_000.0
                };
                match val {
                    Value::Time(t) => Some(Value::Float64(to_secs(&t))),
                    Value::Timestamp(ts) => Some(Value::Float64(to_secs(&ts.to_time()))),
                    Value::ZonedDatetime(zdt) => {
                        Some(Value::Float64(to_secs(&zdt.to_local_time())))
                    }
                    Value::String(s) => {
                        if let Some(zdt) = grafeo_common::types::ZonedDatetime::parse(&s) {
                            Some(Value::Float64(to_secs(&zdt.to_local_time())))
                        } else {
                            grafeo_common::types::Time::parse(&s)
                                .map(|t| Value::Float64(to_secs(&t)))
                        }
                    }
                    _ => None,
                }
            }

            // TIMEZONE - extract timezone as xsd:dayTimeDuration
            "TIMEZONE" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                match val {
                    Value::Time(t) => t.offset_seconds().map(|offset| {
                        Value::Duration(grafeo_common::types::Duration::from_seconds(i64::from(
                            offset,
                        )))
                    }),
                    Value::ZonedDatetime(zdt) => Some(Value::Duration(
                        grafeo_common::types::Duration::from_seconds(i64::from(
                            zdt.offset_seconds(),
                        )),
                    )),
                    Value::String(s) => grafeo_common::types::ZonedDatetime::parse(&s).map(|zdt| {
                        Value::Duration(grafeo_common::types::Duration::from_seconds(i64::from(
                            zdt.offset_seconds(),
                        )))
                    }),
                    _ => None,
                }
            }

            // TZ - extract timezone as string ("+05:00", "Z", "")
            "TZ" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                match val {
                    Value::Time(t) => {
                        if let Some(offset) = t.offset_seconds() {
                            Some(Value::String(format_tz_offset(offset).into()))
                        } else {
                            Some(Value::String(String::new().into()))
                        }
                    }
                    Value::ZonedDatetime(zdt) => {
                        Some(Value::String(format_tz_offset(zdt.offset_seconds()).into()))
                    }
                    Value::String(s) => {
                        if let Some(zdt) = grafeo_common::types::ZonedDatetime::parse(&s) {
                            Some(Value::String(format_tz_offset(zdt.offset_seconds()).into()))
                        } else {
                            Some(Value::String(String::new().into()))
                        }
                    }
                    _ => Some(Value::String(String::new().into())),
                }
            }

            // ================================================================
            // Hash Functions (SPARQL 1.1 Section 17.4.4)
            // ================================================================
            "MD5" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                let s = value_to_string(&val);
                let digest = md5::compute(s.as_bytes());
                Some(Value::String(format!("{digest:x}").into()))
            }

            "SHA1" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                let s = value_to_string(&val);
                use sha1::Digest as _;
                let hash = sha1::Sha1::digest(s.as_bytes());
                let hex = hash.iter().fold(String::new(), |mut s, b| {
                    use std::fmt::Write as _;
                    let _ = write!(s, "{b:02x}");
                    s
                });
                Some(Value::String(hex.into()))
            }

            "SHA256" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                let s = value_to_string(&val);
                use sha2::Digest as _;
                let hash = sha2::Sha256::digest(s.as_bytes());
                let hex = hash.iter().fold(String::new(), |mut s, b| {
                    use std::fmt::Write as _;
                    let _ = write!(s, "{b:02x}");
                    s
                });
                Some(Value::String(hex.into()))
            }

            "SHA384" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                let s = value_to_string(&val);
                use sha2::Digest as _;
                let hash = sha2::Sha384::digest(s.as_bytes());
                let hex = hash.iter().fold(String::new(), |mut s, b| {
                    use std::fmt::Write as _;
                    let _ = write!(s, "{b:02x}");
                    s
                });
                Some(Value::String(hex.into()))
            }

            "SHA512" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                let s = value_to_string(&val);
                use sha2::Digest as _;
                let hash = sha2::Sha512::digest(s.as_bytes());
                let hex = hash.iter().fold(String::new(), |mut s, b| {
                    use std::fmt::Write as _;
                    let _ = write!(s, "{b:02x}");
                    s
                });
                Some(Value::String(hex.into()))
            }

            // ================================================================
            // RDF Term Functions (SPARQL 1.1 Section 17.4.2)
            // ================================================================

            // LANG - language tag of a literal
            "LANG" => {
                // Look up the companion language-tag column for the variable.
                // The triple scan emits a hidden __lang_<var> column alongside
                // each object variable.
                if let Some(FilterExpression::Variable(var_name)) = args.first() {
                    let lang_col_name = format!("__lang_{var_name}");
                    if let Some(&col_idx) = self.variable_columns.get(&lang_col_name)
                        && let Some(col) = chunk.column(col_idx)
                        && let Some(Value::String(tag)) = col.get_value(row)
                    {
                        return Some(Value::String(tag));
                    }
                }
                // No language tag found: return empty string per SPARQL spec
                Some(Value::String(String::new().into()))
            }

            // LANGMATCHES - BCP47 language range matching (Section 17.4.2.9)
            "LANGMATCHES" => {
                if args.len() < 2 {
                    return None;
                }
                let tag = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    _ => return None,
                };
                let range = match self.eval_expr(&args[1], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    _ => return None,
                };

                // SPARQL spec: LANGMATCHES(tag, "*") matches any non-empty tag
                if range == "*" {
                    return Some(Value::Bool(!tag.is_empty()));
                }

                // Case-insensitive prefix match per RFC 4647 basic filtering:
                // tag "en-US" matches range "en" because "en" is a prefix and
                // the next character in tag is '-'.
                let tag_lower = tag.to_lowercase();
                let range_lower = range.to_lowercase();
                let matches = tag_lower == range_lower
                    || (tag_lower.starts_with(&range_lower)
                        && tag_lower.as_bytes().get(range_lower.len()) == Some(&b'-'));
                Some(Value::Bool(matches))
            }

            // DATATYPE - datatype IRI of a literal
            "DATATYPE" => {
                if let Some(FilterExpression::Variable(var_name)) = args.first() {
                    let dt_col_name = format!("__datatype_{var_name}");
                    if let Some(&col_idx) = self.variable_columns.get(&dt_col_name)
                        && let Some(col) = chunk.column(col_idx)
                        && let Some(Value::String(dt)) = col.get_value(row)
                        && !dt.is_empty()
                    {
                        return Some(Value::String(dt));
                    }
                }
                let val = self.eval_expr(args.first()?, chunk, row)?;
                let dt = match &val {
                    Value::String(_) => "http://www.w3.org/2001/XMLSchema#string",
                    Value::Int64(_) => "http://www.w3.org/2001/XMLSchema#integer",
                    Value::Float64(_) => "http://www.w3.org/2001/XMLSchema#double",
                    Value::Bool(_) => "http://www.w3.org/2001/XMLSchema#boolean",
                    Value::Date(_) => "http://www.w3.org/2001/XMLSchema#date",
                    Value::Time(_) => "http://www.w3.org/2001/XMLSchema#time",
                    Value::Timestamp(_) => "http://www.w3.org/2001/XMLSchema#dateTime",
                    Value::Duration(_) => "http://www.w3.org/2001/XMLSchema#duration",
                    _ => return None,
                };
                Some(Value::String(dt.to_string().into()))
            }

            // IRI / URI - construct an IRI
            "IRI" | "URI" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                match val {
                    Value::String(s) => Some(Value::String(s)),
                    v => Some(Value::String(value_to_string(&v).into())),
                }
            }

            // BNODE - construct or retrieve a blank node
            "BNODE" => {
                if args.is_empty() {
                    // Generate a unique blank node ID
                    use std::sync::atomic::{AtomicU64, Ordering};
                    static BNODE_COUNTER: AtomicU64 = AtomicU64::new(0);
                    let id = BNODE_COUNTER.fetch_add(1, Ordering::Relaxed);
                    Some(Value::String(format!("_:b{id}").into()))
                } else {
                    let val = self.eval_expr(&args[0], chunk, row)?;
                    let label = value_to_string(&val);
                    Some(Value::String(format!("_:b{label}").into()))
                }
            }

            // STRDT - construct a typed literal
            "STRDT" => {
                if args.len() < 2 {
                    return None;
                }
                let lexical = self.eval_expr(&args[0], chunk, row)?;
                let datatype = self.eval_expr(&args[1], chunk, row)?;
                let lex_str = value_to_string(&lexical);
                let dt_str = value_to_string(&datatype);
                match dt_str.as_str() {
                    "http://www.w3.org/2001/XMLSchema#integer"
                    | "http://www.w3.org/2001/XMLSchema#int"
                    | "http://www.w3.org/2001/XMLSchema#long" => {
                        lex_str.parse::<i64>().ok().map(Value::Int64)
                    }
                    "http://www.w3.org/2001/XMLSchema#double"
                    | "http://www.w3.org/2001/XMLSchema#float"
                    | "http://www.w3.org/2001/XMLSchema#decimal" => {
                        lex_str.parse::<f64>().ok().map(Value::Float64)
                    }
                    "http://www.w3.org/2001/XMLSchema#boolean" => match lex_str.as_str() {
                        "true" | "1" => Some(Value::Bool(true)),
                        "false" | "0" => Some(Value::Bool(false)),
                        _ => None,
                    },
                    _ => Some(Value::String(lex_str.into())),
                }
            }

            // STRLANG - construct a language-tagged literal
            "STRLANG" => {
                if args.len() < 2 {
                    return None;
                }
                // Return the string value (language tag is metadata)
                self.eval_expr(&args[0], chunk, row)
            }

            // UUID - generate a UUID IRI
            "UUID" => {
                use std::sync::atomic::{AtomicU64, Ordering};
                static UUID_COUNTER: AtomicU64 = AtomicU64::new(0);
                let id = UUID_COUNTER.fetch_add(1, Ordering::Relaxed);
                let ts = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_nanos());
                Some(Value::String(format!("urn:uuid:{ts:032x}-{id:04x}").into()))
            }

            // STRUUID - generate a UUID string (no urn: prefix)
            "STRUUID" => {
                use std::sync::atomic::{AtomicU64, Ordering};
                static STRUUID_COUNTER: AtomicU64 = AtomicU64::new(0);
                let id = STRUUID_COUNTER.fetch_add(1, Ordering::Relaxed);
                let ts = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_nanos());
                Some(Value::String(format!("{ts:032x}-{id:04x}").into()))
            }

            // sameTerm - strict RDF term equality
            "SAMETERM" => {
                if args.len() < 2 {
                    return None;
                }
                let a = self.eval_expr(&args[0], chunk, row)?;
                let b = self.eval_expr(&args[1], chunk, row)?;
                Some(Value::Bool(a == b))
            }

            // ================================================================
            // Numeric Functions (SPARQL 1.1 Section 17.4.4)
            // ================================================================

            // RAND - random double in [0, 1)
            "RAND" => {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                use std::sync::atomic::{AtomicU64, Ordering};
                static RAND_STATE: AtomicU64 = AtomicU64::new(0);
                let state = RAND_STATE.fetch_add(1, Ordering::Relaxed);
                let mut hasher = DefaultHasher::new();
                state.hash(&mut hasher);
                // reason: truncation is intentional for hash seed entropy
                #[allow(clippy::cast_possible_truncation)]
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0u64, |d| d.as_nanos() as u64)
                    .hash(&mut hasher);
                let bits = hasher.finish();
                let value = (bits >> 11) as f64 / (1u64 << 53) as f64;
                Some(Value::Float64(value))
            }

            // Unknown function
            _ => None,
        }
    }
}

/// Parses the date component from a dateTime or date string.
///
/// Handles both "YYYY-MM-DD" and "YYYY-MM-DDTHH:MM:SS..." forms by splitting
/// at 'T' and parsing only the date portion, then falling back to a plain date
/// parse.
fn parse_datetime_date(s: &str) -> Option<grafeo_common::types::Date> {
    if let Some(pos) = s.find('T').or_else(|| s.find('t')) {
        grafeo_common::types::Date::parse(&s[..pos])
    } else {
        grafeo_common::types::Date::parse(s)
    }
}

/// Parses the time component from a dateTime or time string.
///
/// Handles both "HH:MM:SS..." and "YYYY-MM-DDTHH:MM:SS..." forms by splitting
/// at 'T' and parsing only the time portion, then falling back to a plain time
/// parse.
fn parse_datetime_time(s: &str) -> Option<grafeo_common::types::Time> {
    if let Some(pos) = s.find('T').or_else(|| s.find('t')) {
        grafeo_common::types::Time::parse(&s[pos + 1..])
    } else {
        grafeo_common::types::Time::parse(s)
    }
}

/// Formats a timezone offset in seconds as "+HH:MM" or "Z".
fn format_tz_offset(offset_secs: i32) -> String {
    if offset_secs == 0 {
        return "Z".to_string();
    }
    let sign = if offset_secs >= 0 { '+' } else { '-' };
    let abs = offset_secs.unsigned_abs();
    let hours = abs / 3600;
    let minutes = (abs % 3600) / 60;
    format!("{sign}{hours:02}:{minutes:02}")
}

impl Predicate for RdfExpressionPredicate {
    fn evaluate(&self, chunk: &DataChunk, row: usize) -> bool {
        matches!(self.eval(chunk, row), Some(Value::Bool(true)))
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Strips internal `__lang_*` companion columns from the final output.
///
/// If no internal columns are present, the operator and columns pass through
/// unchanged. Otherwise, a lightweight projection is inserted to remove them.
fn strip_internal_columns(
    operator: Box<dyn Operator>,
    columns: Vec<String>,
) -> (Box<dyn Operator>, Vec<String>) {
    let keep_indices: Vec<usize> = columns
        .iter()
        .enumerate()
        .filter(|(_, name)| !name.starts_with("__lang_") && !name.starts_with("__datatype_"))
        .map(|(i, _)| i)
        .collect();

    if keep_indices.len() == columns.len() {
        // Nothing to strip
        return (operator, columns);
    }

    let output_columns: Vec<String> = keep_indices.iter().map(|&i| columns[i].clone()).collect();
    let output_types: Vec<LogicalType> = keep_indices.iter().map(|_| LogicalType::Any).collect();

    let projections = keep_indices
        .into_iter()
        .map(RdfProjectExpr::Column)
        .collect();

    let stripped = Box::new(RdfProjectOperator::new(operator, projections, output_types));
    (stripped, output_columns)
}

/// Converts an RDF Term to a string for IRI/blank node representation.
fn term_to_string(term: &Term) -> String {
    match term {
        Term::Iri(iri) => iri.as_str().to_string(),
        Term::BlankNode(bnode) => format!("_:{}", bnode.id()),
        Term::Literal(lit) => lit.value().to_string(),
        _ => String::new(),
    }
}

/// Pushes an RDF term value to a column.
///
/// For RDF columns (which use String type), we always push as string to avoid
/// type mismatches. The typed literal's value is preserved as a string, and
/// numeric comparisons are handled at the filter level.
fn push_term_value(col: &mut grafeo_core::execution::ValueVector, term: &Term) {
    match term {
        Term::Iri(iri) => col.push_string(iri.as_str().to_string()),
        Term::BlankNode(bnode) => col.push_string(format!("_:{}", bnode.id())),
        Term::Literal(lit) => {
            // Always push as string since RDF columns are String type
            // Numeric operations are handled by the filter evaluation
            col.push_string(lit.value().to_string());
        }
        _ => col.push_value(Value::Null),
    }
}

/// Converts a TripleComponent to an Option<Term> for pattern matching.
fn component_to_term(component: &TripleComponent) -> Option<Term> {
    match component {
        TripleComponent::Variable(_) => None,
        TripleComponent::BlankNode(label) => Some(Term::blank(label.clone())),
        TripleComponent::Iri(iri) => Some(Term::iri(iri.clone())),
        TripleComponent::Literal(value) => match value {
            Value::String(s) => Some(Term::literal(s.clone())),
            Value::Int64(i) => Some(Term::typed_literal(i.to_string(), Literal::XSD_INTEGER)),
            Value::Float64(f) => Some(Term::typed_literal(f.to_string(), Literal::XSD_DOUBLE)),
            Value::Bool(b) => Some(Term::typed_literal(b.to_string(), Literal::XSD_BOOLEAN)),
            _ => Some(Term::literal(value.to_string())),
        },
        TripleComponent::LangLiteral { value, lang } => {
            Some(Term::lang_literal(value.clone(), lang.clone()))
        }
    }
}

/// Checks whether a logical plan tree references LANG, LANGMATCHES, or DATATYPE
/// functions. When true, the planner emits companion columns on triple scans.
fn uses_lang_or_datatype(op: &LogicalOperator) -> bool {
    use LogicalOperator::{
        Aggregate, AntiJoin, Bind, Distinct, Filter, Join, LeftJoin, Limit, Project, Return, Skip,
        Sort, TripleScan, Union, Unwind,
    };
    match op {
        Filter(f) => expr_uses_lang_or_datatype(&f.predicate) || uses_lang_or_datatype(&f.input),
        Project(p) => {
            p.projections
                .iter()
                .any(|proj| expr_uses_lang_or_datatype(&proj.expression))
                || uses_lang_or_datatype(&p.input)
        }
        Bind(b) => expr_uses_lang_or_datatype(&b.expression) || uses_lang_or_datatype(&b.input),
        Aggregate(a) => {
            a.aggregates.iter().any(|ae| {
                ae.expression
                    .as_ref()
                    .is_some_and(expr_uses_lang_or_datatype)
            }) || a.having.as_ref().is_some_and(expr_uses_lang_or_datatype)
                || uses_lang_or_datatype(&a.input)
        }
        Return(r) => {
            r.items
                .iter()
                .any(|item| expr_uses_lang_or_datatype(&item.expression))
                || uses_lang_or_datatype(&r.input)
        }
        Sort(s) => {
            s.keys
                .iter()
                .any(|k| expr_uses_lang_or_datatype(&k.expression))
                || uses_lang_or_datatype(&s.input)
        }
        Join(j) => uses_lang_or_datatype(&j.left) || uses_lang_or_datatype(&j.right),
        LeftJoin(j) => {
            uses_lang_or_datatype(&j.left)
                || uses_lang_or_datatype(&j.right)
                || j.condition.as_ref().is_some_and(expr_uses_lang_or_datatype)
        }
        AntiJoin(j) => uses_lang_or_datatype(&j.left) || uses_lang_or_datatype(&j.right),
        Union(u) => u.inputs.iter().any(uses_lang_or_datatype),
        Distinct(d) => uses_lang_or_datatype(&d.input),
        Limit(l) => uses_lang_or_datatype(&l.input),
        Skip(s) => uses_lang_or_datatype(&s.input),
        Unwind(u) => uses_lang_or_datatype(&u.input),
        TripleScan(t) => t.input.as_ref().is_some_and(|i| uses_lang_or_datatype(i)),
        LogicalOperator::MultiWayJoin(mwj) => mwj.inputs.iter().any(uses_lang_or_datatype),
        // For any other operator, conservatively assume companion columns are needed
        _ => true,
    }
}

/// Checks whether a logical expression references LANG, LANGMATCHES, or DATATYPE.
fn expr_uses_lang_or_datatype(expr: &LogicalExpression) -> bool {
    match expr {
        LogicalExpression::FunctionCall { name, args, .. } => {
            let upper = name.to_uppercase();
            if upper == "LANG" || upper == "LANGMATCHES" || upper == "DATATYPE" {
                return true;
            }
            args.iter().any(expr_uses_lang_or_datatype)
        }
        LogicalExpression::Binary { left, right, .. } => {
            expr_uses_lang_or_datatype(left) || expr_uses_lang_or_datatype(right)
        }
        LogicalExpression::Unary { operand, .. } => expr_uses_lang_or_datatype(operand),
        LogicalExpression::List(items) => items.iter().any(expr_uses_lang_or_datatype),
        LogicalExpression::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            operand.as_deref().is_some_and(expr_uses_lang_or_datatype)
                || when_clauses
                    .iter()
                    .any(|(w, t)| expr_uses_lang_or_datatype(w) || expr_uses_lang_or_datatype(t))
                || else_clause
                    .as_deref()
                    .is_some_and(expr_uses_lang_or_datatype)
        }
        _ => false,
    }
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
use crate::query::planner::common::{expression_to_string, output_column_name, resolved_column_name};

/// Converts a value to its string representation.
fn value_to_string(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(b) => b.to_string(),
        Value::Int64(i) => i.to_string(),
        Value::Float64(f) => f.to_string(),
        Value::String(s) => s.to_string(),
        Value::Bytes(b) => String::from_utf8_lossy(b).to_string(),
        Value::Timestamp(t) => t.to_string(),
        Value::Date(d) => d.to_string(),
        Value::Time(t) => t.to_string(),
        Value::Duration(d) => d.to_string(),
        Value::List(items) => {
            let parts: Vec<String> = items.iter().map(value_to_string).collect();
            format!("[{}]", parts.join(", "))
        }
        Value::Map(entries) => {
            let parts: Vec<String> = entries
                .iter()
                .map(|(k, v)| format!("{}: {}", k, value_to_string(v)))
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
        Value::Vector(v) => {
            let parts: Vec<String> = v.iter().map(|f| f.to_string()).collect();
            format!("vector([{}])", parts.join(", "))
        }
        Value::ZonedDatetime(zdt) => zdt.to_string(),
        Value::Path { nodes, edges } => {
            format!("<path: {} nodes, {} edges>", nodes.len(), edges.len())
        }
        Value::GCounter(counts) => {
            let total: u64 = counts.values().sum();
            format!("GCounter({total})")
        }
        Value::OnCounter { pos, neg } => {
            // reason: counter values are small increments, sum will not overflow i64
            #[allow(clippy::cast_possible_wrap)]
            let pos_sum: i64 = pos.values().copied().map(|v| v as i64).sum();
            // reason: value is a small counter, well within i64::MAX
            #[allow(clippy::cast_possible_wrap)]
            let neg_sum: i64 = neg.values().copied().map(|v| v as i64).sum();
            format!("OnCounter({})", pos_sum - neg_sum)
        }
        _ => value.to_string(),
    }
}

/// Compares two values and returns a boolean result.
fn compare_values<F>(left: &Value, right: &Value, cmp: F) -> Option<Value>
where
    F: Fn(std::cmp::Ordering) -> bool,
{
    let ordering = match (left, right) {
        (Value::Int64(l), Value::Int64(r)) => l.cmp(r),
        (Value::Float64(l), Value::Float64(r)) => l.partial_cmp(r)?,
        (Value::String(l), Value::String(r)) => {
            // Try numeric comparison first if both look like numbers
            if let (Ok(l_num), Ok(r_num)) = (l.parse::<f64>(), r.parse::<f64>()) {
                l_num.partial_cmp(&r_num)?
            } else {
                l.cmp(r)
            }
        }
        (Value::Int64(l), Value::Float64(r)) => (*l as f64).partial_cmp(r)?,
        (Value::Float64(l), Value::Int64(r)) => l.partial_cmp(&(*r as f64))?,
        // RDF values are often stored as strings - try numeric conversion
        (Value::String(s), Value::Int64(r)) => {
            let l_num = s.parse::<f64>().ok()?;
            l_num.partial_cmp(&(*r as f64))?
        }
        (Value::String(s), Value::Float64(r)) => {
            let l_num = s.parse::<f64>().ok()?;
            l_num.partial_cmp(r)?
        }
        (Value::Int64(l), Value::String(s)) => {
            let r_num = s.parse::<f64>().ok()?;
            (*l as f64).partial_cmp(&r_num)?
        }
        (Value::Float64(l), Value::String(s)) => {
            let r_num = s.parse::<f64>().ok()?;
            l.partial_cmp(&r_num)?
        }
        _ => return None,
    };
    Some(Value::Bool(cmp(ordering)))
}

/// Checks equality of two RDF values, with cross-type numeric coercion.
///
/// Used by the IN operator to compare the left-hand value against each element
/// of the right-hand list.
fn rdf_values_equal(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::Int64(a), Value::Int64(b)) => a == b,
        (Value::Float64(a), Value::Float64(b)) => (a - b).abs() < f64::EPSILON,
        (Value::String(a), Value::String(b)) => a == b,
        (Value::Int64(a), Value::Float64(b)) | (Value::Float64(b), Value::Int64(a)) => {
            (*a as f64 - b).abs() < f64::EPSILON
        }
        // RDF stores numeric literals as strings: allow cross-type equality
        (Value::String(s), Value::Int64(i)) | (Value::Int64(i), Value::String(s)) => {
            s.parse::<i64>().is_ok_and(|n| n == *i)
        }
        (Value::String(s), Value::Float64(f)) | (Value::Float64(f), Value::String(s)) => {
            s.parse::<f64>().is_ok_and(|n| (n - f).abs() < f64::EPSILON)
        }
        _ => false,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
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
            false,
            false,
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
        let planner = RdfPlanner::new(Arc::clone(&store));
        let physical = planner.plan(&plan).unwrap();
        let mut op = physical.operator;
        while op.next().unwrap().is_some() {}

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
        let planner = RdfPlanner::new(Arc::clone(&store));
        let physical = planner.plan(&plan).unwrap();
        let mut op = physical.operator;
        while op.next().unwrap().is_some() {}

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
        let planner = RdfPlanner::new(Arc::clone(&store));
        let physical = planner.plan(&plan).unwrap();
        let mut op = physical.operator;
        while op.next().unwrap().is_some() {}

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
        let planner = RdfPlanner::new(Arc::clone(&store));
        let physical = planner.plan(&plan).unwrap();
        let mut op = physical.operator;

        // Should error because source doesn't exist
        let result = op.next();
        assert!(result.is_err());
    }

    #[test]
    fn test_copy_nonexistent_source_silent_ok() {
        let store = Arc::new(RdfStore::new());

        let plan = LogicalPlan::new(LogicalOperator::CopyGraph(CopyGraphOp {
            source: Some("http://example.org/nope".to_string()),
            destination: Some("http://example.org/dst".to_string()),
            silent: true,
        }));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let physical = planner.plan(&plan).unwrap();
        let mut op = physical.operator;

        // Should succeed silently
        assert!(op.next().is_ok());
    }

    /// Triple scan propagates LogicalType::String for every output column.
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
        // All RDF columns are String: s, p, o, plus __lang_o companion
        assert_eq!(columns.len(), types.len());
        for (i, ty) in types.iter().enumerate() {
            assert_eq!(
                *ty,
                LogicalType::String,
                "column {i} ({}) should be String",
                columns[i]
            );
        }
    }

    /// Join of two triple scans propagates String types through to output.
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
        // All output columns should be String (join preserves types from both sides)
        for (i, ty) in types.iter().enumerate() {
            assert_eq!(
                *ty,
                LogicalType::String,
                "join column {i} ({}) should be String",
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
        assert_eq!(columns.last(), Some(&"label".to_string()));
        // Input columns are String, BIND column is Any
        let last_idx = types.len() - 1;
        assert_eq!(
            types[last_idx],
            LogicalType::Any,
            "BIND column should be Any"
        );
        for ty in &types[..last_idx] {
            assert_eq!(*ty, LogicalType::String, "input columns should be String");
        }
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
        assert_eq!(types[0], LogicalType::String, "group-by column is String");
        assert_eq!(types[1], LogicalType::Int64, "COUNT produces Int64");
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
                right: Box::new(LogicalExpression::Literal(Value::String("25".into()))),
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
        assert_eq!(
            chunk.column(0).unwrap().get_value(0),
            Some(Value::Int64(10))
        );
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
                right: Box::new(LogicalExpression::Literal(Value::String("1".into()))),
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
        assert_eq!(
            vals,
            vec![
                Value::String("1".into()),
                Value::String("2".into()),
                Value::String("3".into())
            ]
        );
    }

    #[test]
    fn test_plan_insert_triple_concrete() {
        let store = Arc::new(RdfStore::new());
        let planner = RdfPlanner::new(Arc::clone(&store));
        let insert = LogicalOperator::InsertTriple(InsertTripleOp {
            subject: TripleComponent::Iri("http://example.org/alix".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Literal(Value::String("Alix".into())),
            graph: None,
            input: None,
        });
        let physical = planner.plan(&LogicalPlan::new(insert)).unwrap();
        let mut op = physical.operator;
        while op.next().unwrap().is_some() {}
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_plan_insert_triple_into_named_graph() {
        let store = Arc::new(RdfStore::new());
        let planner = RdfPlanner::new(Arc::clone(&store));
        let insert = LogicalOperator::InsertTriple(InsertTripleOp {
            subject: TripleComponent::Iri("http://example.org/alix".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Literal(Value::String("Alix".into())),
            graph: Some("http://example.org/g1".to_string()),
            input: None,
        });
        let physical = planner.plan(&LogicalPlan::new(insert)).unwrap();
        let mut op = physical.operator;
        while op.next().unwrap().is_some() {}
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
        let planner = RdfPlanner::new(Arc::clone(&store));
        let delete = LogicalOperator::DeleteTriple(DeleteTripleOp {
            subject: TripleComponent::Iri("http://example.org/alix".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Literal(Value::String("Alix".into())),
            graph: None,
            input: None,
        });
        let physical = planner.plan(&LogicalPlan::new(delete)).unwrap();
        let mut op = physical.operator;
        while op.next().unwrap().is_some() {}
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
        let planner = RdfPlanner::new(Arc::clone(&store));
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
        let physical = planner.plan(&LogicalPlan::new(insert)).unwrap();
        let mut op = physical.operator;
        while op.next().unwrap().is_some() {}
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
        let planner = RdfPlanner::new(Arc::clone(&store));
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
        let physical = planner.plan(&LogicalPlan::new(delete)).unwrap();
        let mut op = physical.operator;
        while op.next().unwrap().is_some() {}
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
        let planner = RdfPlanner::new(Arc::clone(&store));
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
        let physical = planner.plan(&LogicalPlan::new(modify)).unwrap();
        let mut op = physical.operator;
        while op.next().unwrap().is_some() {}
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
            expression: crate::query::plan::LogicalExpression::FunctionCall {
                name: "UCASE".to_string(),
                args: vec![crate::query::plan::LogicalExpression::Variable(
                    "name".to_string(),
                )],
                distinct: false,
            },
        });
        let physical = planner.plan(&LogicalPlan::new(bind)).unwrap();
        assert!(physical.columns.contains(&"upper".to_string()));
        let mut op = physical.operator;
        let chunk = op.next().unwrap().unwrap();
        let upper_idx = physical.columns.iter().position(|c| c == "upper").unwrap();
        assert_eq!(
            chunk.column(upper_idx).unwrap().get_value(0),
            Some(Value::String("ALIX".into()))
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
        assert_eq!(
            chunk.column(0).unwrap().get_value(0),
            Some(Value::String("Alix".into()))
        );
        assert_eq!(
            chunk.column(1).unwrap().get_value(0),
            Some(Value::String("const".into()))
        );
    }

    #[test]
    fn test_plan_create_graph() {
        let store = Arc::new(RdfStore::new());
        let planner = RdfPlanner::new(Arc::clone(&store));
        let physical = planner
            .plan(&LogicalPlan::new(LogicalOperator::CreateGraph(
                CreateGraphOp {
                    graph: "http://example.org/g1".to_string(),
                    silent: false,
                },
            )))
            .unwrap();
        let mut op = physical.operator;
        while op.next().unwrap().is_some() {}
        assert!(store.graph("http://example.org/g1").is_some());
    }

    #[test]
    fn test_plan_create_graph_already_exists_errors() {
        let store = Arc::new(RdfStore::new());
        store.create_graph("http://example.org/g1");
        let planner = RdfPlanner::new(Arc::clone(&store));
        let mut physical = planner
            .plan(&LogicalPlan::new(LogicalOperator::CreateGraph(
                CreateGraphOp {
                    graph: "http://example.org/g1".to_string(),
                    silent: false,
                },
            )))
            .unwrap();
        assert!(physical.operator.next().is_err());
    }

    #[test]
    fn test_plan_create_graph_silent() {
        let store = Arc::new(RdfStore::new());
        store.create_graph("http://example.org/g1");
        let planner = RdfPlanner::new(Arc::clone(&store));
        let mut physical = planner
            .plan(&LogicalPlan::new(LogicalOperator::CreateGraph(
                CreateGraphOp {
                    graph: "http://example.org/g1".to_string(),
                    silent: true,
                },
            )))
            .unwrap();
        assert!(physical.operator.next().is_ok());
    }

    #[test]
    fn test_plan_drop_graph() {
        let store = Arc::new(RdfStore::new());
        store.create_graph("http://example.org/g1");
        let planner = RdfPlanner::new(Arc::clone(&store));
        let physical = planner
            .plan(&LogicalPlan::new(LogicalOperator::DropGraph(DropGraphOp {
                graph: Some("http://example.org/g1".to_string()),
                silent: false,
            })))
            .unwrap();
        let mut op = physical.operator;
        while op.next().unwrap().is_some() {}
        assert!(store.graph("http://example.org/g1").is_none());
    }

    #[test]
    fn test_plan_drop_nonexistent_graph_errors() {
        let store = Arc::new(RdfStore::new());
        let planner = RdfPlanner::new(Arc::clone(&store));
        let mut physical = planner
            .plan(&LogicalPlan::new(LogicalOperator::DropGraph(DropGraphOp {
                graph: Some("http://example.org/nope".to_string()),
                silent: false,
            })))
            .unwrap();
        assert!(physical.operator.next().is_err());
    }

    #[test]
    fn test_plan_drop_default_graph() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/a"),
            Term::iri("http://example.org/b"),
            Term::literal("c"),
        ));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let physical = planner
            .plan(&LogicalPlan::new(LogicalOperator::DropGraph(DropGraphOp {
                graph: None,
                silent: false,
            })))
            .unwrap();
        let mut op = physical.operator;
        while op.next().unwrap().is_some() {}
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
        let planner = RdfPlanner::new(Arc::clone(&store));
        let physical = planner
            .plan(&LogicalPlan::new(LogicalOperator::ClearGraph(
                ClearGraphOp {
                    graph: None,
                    silent: false,
                },
            )))
            .unwrap();
        let mut op = physical.operator;
        while op.next().unwrap().is_some() {}
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
        let planner = RdfPlanner::new(Arc::clone(&store));
        let physical = planner
            .plan(&LogicalPlan::new(LogicalOperator::ClearGraph(
                ClearGraphOp {
                    graph: Some(String::new()),
                    silent: false,
                },
            )))
            .unwrap();
        let mut op = physical.operator;
        while op.next().unwrap().is_some() {}
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
        assert!(physical.columns.is_empty());
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
        assert_eq!(chunk.column(0).unwrap().get_value(0), Some(Value::Int64(5)));
    }

    // ---- into_any() coverage tests ----
    //
    // Each RDF operator implements `into_any()` for downcasting support in the
    // push pipeline. These tests construct minimal instances and verify the
    // method returns a valid `Box<dyn Any + Send>` that can be downcast back.

    #[test]
    fn test_into_any_rdf_insert_triple_operator() {
        let store = Arc::new(RdfStore::new());
        let triple = Triple::new(
            Term::iri("http://example.org/s"),
            Term::iri("http://example.org/p"),
            Term::literal("o"),
        );
        let op: Box<dyn Operator> = Box::new(RdfInsertTripleOperator::new(
            store,
            triple,
            None,
            None,
            #[cfg(feature = "wal")]
            None,
            #[cfg(feature = "cdc")]
            None,
            #[cfg(feature = "cdc")]
            grafeo_common::types::EpochId(0),
        ));
        let any = op.into_any();
        assert!(any.downcast::<RdfInsertTripleOperator>().is_ok());
    }

    #[test]
    fn test_into_any_rdf_insert_pattern_operator() {
        let store = Arc::new(RdfStore::new());
        let child: Box<dyn Operator> = Box::new(SingleRowOperator::new());
        let operands = TripleOperands {
            subject: TripleComponent::Iri("http://example.org/s".to_string()),
            predicate: TripleComponent::Iri("http://example.org/p".to_string()),
            object: TripleComponent::Literal(Value::String("o".into())),
            column_map: HashMap::new(),
        };
        let op: Box<dyn Operator> = Box::new(RdfInsertPatternOperator::new(
            store,
            child,
            operands,
            #[cfg(feature = "wal")]
            None,
            #[cfg(feature = "cdc")]
            None,
            #[cfg(feature = "cdc")]
            grafeo_common::types::EpochId(0),
        ));
        let any = op.into_any();
        assert!(any.downcast::<RdfInsertPatternOperator>().is_ok());
    }

    #[test]
    fn test_into_any_rdf_delete_triple_operator() {
        let store = Arc::new(RdfStore::new());
        let triple = Triple::new(
            Term::iri("http://example.org/s"),
            Term::iri("http://example.org/p"),
            Term::literal("o"),
        );
        let op: Box<dyn Operator> = Box::new(RdfDeleteTripleOperator::new(
            store,
            triple,
            None,
            None,
            #[cfg(feature = "wal")]
            None,
            #[cfg(feature = "cdc")]
            None,
            #[cfg(feature = "cdc")]
            grafeo_common::types::EpochId(0),
        ));
        let any = op.into_any();
        assert!(any.downcast::<RdfDeleteTripleOperator>().is_ok());
    }

    #[test]
    fn test_into_any_rdf_delete_pattern_operator() {
        let store = Arc::new(RdfStore::new());
        let child: Box<dyn Operator> = Box::new(SingleRowOperator::new());
        let operands = TripleOperands {
            subject: TripleComponent::Iri("http://example.org/s".to_string()),
            predicate: TripleComponent::Iri("http://example.org/p".to_string()),
            object: TripleComponent::Literal(Value::String("o".into())),
            column_map: HashMap::new(),
        };
        let op: Box<dyn Operator> = Box::new(RdfDeletePatternOperator::new(
            store,
            child,
            operands,
            #[cfg(feature = "wal")]
            None,
            #[cfg(feature = "cdc")]
            None,
            #[cfg(feature = "cdc")]
            grafeo_common::types::EpochId(0),
        ));
        let any = op.into_any();
        assert!(any.downcast::<RdfDeletePatternOperator>().is_ok());
    }

    #[test]
    fn test_into_any_rdf_clear_graph_operator() {
        let store = Arc::new(RdfStore::new());
        let op: Box<dyn Operator> = Box::new(RdfClearGraphOperator::new(
            store,
            None,
            false,
            #[cfg(feature = "wal")]
            None,
        ));
        let any = op.into_any();
        assert!(any.downcast::<RdfClearGraphOperator>().is_ok());
    }

    #[test]
    fn test_into_any_rdf_create_graph_operator() {
        let store = Arc::new(RdfStore::new());
        let op: Box<dyn Operator> = Box::new(RdfCreateGraphOperator::new(
            store,
            "http://example.org/g".to_string(),
            true,
            #[cfg(feature = "wal")]
            None,
        ));
        let any = op.into_any();
        assert!(any.downcast::<RdfCreateGraphOperator>().is_ok());
    }

    #[test]
    fn test_into_any_rdf_drop_graph_operator() {
        let store = Arc::new(RdfStore::new());
        let op: Box<dyn Operator> = Box::new(RdfDropGraphOperator::new(
            store,
            Some("http://example.org/g".to_string()),
            true,
            #[cfg(feature = "wal")]
            None,
        ));
        let any = op.into_any();
        assert!(any.downcast::<RdfDropGraphOperator>().is_ok());
    }

    #[test]
    fn test_into_any_rdf_copy_graph_operator() {
        let store = Arc::new(RdfStore::new());
        let op: Box<dyn Operator> = Box::new(RdfCopyGraphOperator::new(
            store,
            Some("http://example.org/src".to_string()),
            Some("http://example.org/dst".to_string()),
            true,
        ));
        let any = op.into_any();
        assert!(any.downcast::<RdfCopyGraphOperator>().is_ok());
    }

    #[test]
    fn test_into_any_rdf_move_graph_operator() {
        let store = Arc::new(RdfStore::new());
        let op: Box<dyn Operator> = Box::new(RdfMoveGraphOperator::new(
            store,
            Some("http://example.org/src".to_string()),
            Some("http://example.org/dst".to_string()),
            true,
        ));
        let any = op.into_any();
        assert!(any.downcast::<RdfMoveGraphOperator>().is_ok());
    }

    #[test]
    fn test_into_any_rdf_add_graph_operator() {
        let store = Arc::new(RdfStore::new());
        let op: Box<dyn Operator> = Box::new(RdfAddGraphOperator::new(
            store,
            Some("http://example.org/src".to_string()),
            Some("http://example.org/dst".to_string()),
            true,
        ));
        let any = op.into_any();
        assert!(any.downcast::<RdfAddGraphOperator>().is_ok());
    }

    #[test]
    fn test_into_any_rdf_modify_operator() {
        let store = Arc::new(RdfStore::new());
        let child: Box<dyn Operator> = Box::new(SingleRowOperator::new());
        let op: Box<dyn Operator> = Box::new(RdfModifyOperator::new(
            store,
            child,
            vec![],
            vec![],
            HashMap::new(),
            #[cfg(feature = "cdc")]
            None,
            #[cfg(feature = "cdc")]
            grafeo_common::types::EpochId(0),
        ));
        let any = op.into_any();
        assert!(any.downcast::<RdfModifyOperator>().is_ok());
    }

    #[test]
    fn test_into_any_rdf_union_operator() {
        let op: Box<dyn Operator> = Box::new(RdfUnionOperator::new(vec![]));
        let any = op.into_any();
        assert!(any.downcast::<RdfUnionOperator>().is_ok());
    }

    #[test]
    fn test_into_any_rdf_bind_operator() {
        let child: Box<dyn Operator> = Box::new(SingleRowOperator::new());
        let expr = FilterExpression::Literal(Value::String("test".into()));
        let op: Box<dyn Operator> = Box::new(RdfBindOperator::new(child, expr, HashMap::new()));
        let any = op.into_any();
        assert!(any.downcast::<RdfBindOperator>().is_ok());
    }

    #[test]
    fn test_into_any_rdf_project_operator() {
        let child: Box<dyn Operator> = Box::new(SingleRowOperator::new());
        let op: Box<dyn Operator> = Box::new(RdfProjectOperator::new(
            child,
            vec![RdfProjectExpr::Constant(Value::String("x".into()))],
            vec![LogicalType::String],
        ));
        let any = op.into_any();
        assert!(any.downcast::<RdfProjectOperator>().is_ok());
    }

    #[test]
    fn test_into_any_construct_operator() {
        let child: Box<dyn Operator> = Box::new(SingleRowOperator::new());
        let op: Box<dyn Operator> = Box::new(ConstructOperator::new(child, vec![], HashMap::new()));
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
    fn test_into_any_dict_resolve_operator() {
        let child: Box<dyn Operator> = Box::new(SingleRowOperator::new());
        let dict = Arc::new(grafeo_core::graph::rdf::TermDictionary::new());
        let op: Box<dyn Operator> = Box::new(DictResolveOperator::new(child, dict, vec![]));
        let any = op.into_any();
        assert!(any.downcast::<DictResolveOperator>().is_ok());
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
            false,
            false,
        ));
        let any = op.into_any();
        assert!(any.downcast::<RdfTripleScanOperator>().is_ok());
    }
}
