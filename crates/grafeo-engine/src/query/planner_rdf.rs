//! RDF Query Planner.
//!
//! Converts logical plans with RDF operators (TripleScan, etc.) to physical
//! operators that execute against an RDF store.
//!
//! This planner follows the same push-based, vectorized execution model as
//! the LPG planner for consistent performance characteristics.

use std::collections::HashMap;
use std::sync::Arc;

use grafeo_common::types::{LogicalType, TxId, Value};
use grafeo_common::utils::error::{Error, Result};
use grafeo_core::execution::DataChunk;
use grafeo_core::execution::operators::JoinType;
use grafeo_core::execution::operators::{
    BinaryFilterOp, FilterExpression, FilterOperator, HashAggregateOperator, JoinCondition,
    LimitOperator, NestedLoopJoinOperator, Operator, OperatorError, Predicate, ProjectOperator,
    SimpleAggregateOperator, SkipOperator, SortOperator, UnaryFilterOp,
};
use grafeo_core::graph::rdf::{Literal, RdfStore, Term, Triple, TriplePattern};

use crate::query::plan::{
    AggregateFunction as LogicalAggregateFunction, AggregateOp, AntiJoinOp, ClearGraphOp,
    CreateGraphOp, DeleteTripleOp, DropGraphOp, FilterOp, InsertTripleOp, LeftJoinOp, LimitOp,
    LogicalExpression, LogicalOperator, LogicalPlan, ModifyOp, SkipOp, SortOp, TripleComponent,
    TripleScanOp, TripleTemplate,
};
use crate::query::planner::{PhysicalPlan, convert_aggregate_function, convert_filter_expression};

/// Default chunk size for morsel-driven execution.
const DEFAULT_CHUNK_SIZE: usize = 1024;

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
    tx_id: Option<TxId>,
}

impl RdfPlanner {
    /// Creates a new RDF planner with the given store.
    #[must_use]
    pub fn new(store: Arc<RdfStore>) -> Self {
        Self {
            store,
            chunk_size: DEFAULT_CHUNK_SIZE,
            tx_id: None,
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
    pub fn with_tx_id(mut self, tx_id: Option<TxId>) -> Self {
        self.tx_id = tx_id;
        self
    }

    /// Plans a logical plan into a physical operator tree.
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

    /// Plans a single logical operator.
    fn plan_operator(&self, op: &LogicalOperator) -> Result<(Box<dyn Operator>, Vec<String>)> {
        match op {
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
            LogicalOperator::Distinct(distinct) => self.plan_operator(&distinct.input),
            LogicalOperator::InsertTriple(insert) => self.plan_insert_triple(insert),
            LogicalOperator::DeleteTriple(delete) => self.plan_delete_triple(delete),
            LogicalOperator::Modify(modify) => self.plan_modify(modify),
            LogicalOperator::ClearGraph(clear) => self.plan_clear_graph(clear),
            LogicalOperator::CreateGraph(create) => self.plan_create_graph(create),
            LogicalOperator::DropGraph(drop_op) => self.plan_drop_graph(drop_op),
            LogicalOperator::Empty => Err(Error::Internal("Empty plan".to_string())),
            _ => Err(Error::Internal(format!(
                "Unsupported RDF operator: {:?}",
                std::mem::discriminant(op)
            ))),
        }
    }

    /// Plans a triple scan operator.
    ///
    /// Creates a lazy scanning operator that reads triples in chunks
    /// for cache-efficient, vectorized processing.
    fn plan_triple_scan(&self, scan: &TripleScanOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
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
        if let TripleComponent::Variable(name) = &scan.object {
            columns.push(name.clone());
            output_mask[2] = true;
        }
        if let Some(TripleComponent::Variable(name)) = &scan.graph {
            columns.push(name.clone());
            output_mask[3] = true;
        }

        // Create the lazy scanning operator
        let operator = Box::new(RdfTripleScanOperator::new(
            Arc::clone(&self.store),
            pattern,
            output_mask,
            self.chunk_size,
        ));

        Ok((operator, columns))
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
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, _input_columns) = self.plan_operator(&ret.input)?;

        // Extract output column names
        let columns: Vec<String> = ret
            .items
            .iter()
            .map(|item| {
                item.alias
                    .clone()
                    .unwrap_or_else(|| expression_to_string(&item.expression))
            })
            .collect();

        Ok((input_op, columns))
    }

    /// Plans a filter operator.
    fn plan_filter(&self, filter: &FilterOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, columns) = self.plan_operator(&filter.input)?;

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
        Ok((operator, columns))
    }

    /// Plans a LIMIT operator.
    fn plan_limit(&self, limit: &LimitOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, columns) = self.plan_operator(&limit.input)?;
        let output_schema = derive_rdf_schema(&columns);
        let operator = Box::new(LimitOperator::new(input_op, limit.count, output_schema));
        Ok((operator, columns))
    }

    /// Plans a SKIP operator.
    fn plan_skip(&self, skip: &SkipOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, columns) = self.plan_operator(&skip.input)?;
        let output_schema = derive_rdf_schema(&columns);
        let operator = Box::new(SkipOperator::new(input_op, skip.count, output_schema));
        Ok((operator, columns))
    }

    /// Plans a SORT operator.
    fn plan_sort(&self, sort: &SortOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        use crate::query::plan::SortOrder;
        use grafeo_core::execution::operators::{NullOrder, SortDirection, SortKey};

        let (input_op, columns) = self.plan_operator(&sort.input)?;

        let variable_columns: HashMap<String, usize> = columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

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

        let output_schema = derive_rdf_schema(&columns);
        let operator = Box::new(SortOperator::new(input_op, physical_keys, output_schema));
        Ok((operator, columns))
    }

    /// Plans a PROJECT operator.
    ///
    /// Projects only the requested columns from the input.
    fn plan_project(
        &self,
        project: &crate::query::plan::ProjectOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        use grafeo_core::execution::operators::{ProjectExpr, ProjectOperator};

        let (input_op, input_columns) = self.plan_operator(&project.input)?;

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
                        projections.push(ProjectExpr::Column(col_idx));
                        output_columns.push(proj.alias.clone().unwrap_or_else(|| name.clone()));
                        output_types.push(LogicalType::String); // RDF values are strings
                    } else {
                        return Err(Error::Internal(format!(
                            "Variable '{}' not found in input columns",
                            name
                        )));
                    }
                }
                _ => {
                    // For non-variable expressions, we need to evaluate them
                    // For now, skip complex expressions in projection
                    continue;
                }
            }
        }

        // If no projections were extracted, just return the input as-is
        if projections.is_empty() {
            return Ok((input_op, input_columns));
        }

        let operator = Box::new(ProjectOperator::new(input_op, projections, output_types));
        Ok((operator, output_columns))
    }

    /// Plans an AGGREGATE operator.
    fn plan_aggregate(&self, agg: &AggregateOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        use grafeo_core::execution::operators::AggregateExpr as PhysicalAggregateExpr;

        let (input_op, input_columns) = self.plan_operator(&agg.input)?;

        let variable_columns: HashMap<String, usize> = input_columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

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

                Ok(PhysicalAggregateExpr {
                    function: convert_aggregate_function(agg_expr.function),
                    column,
                    distinct: agg_expr.distinct,
                    alias: agg_expr.alias.clone(),
                    percentile: agg_expr.percentile,
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
                LogicalAggregateFunction::Count => LogicalType::Int64,
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

        let operator: Box<dyn Operator> = if group_columns.is_empty() {
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

        Ok((operator, output_columns))
    }

    /// Plans a JOIN operator.
    ///
    /// For SPARQL, we need to join on shared variables (equi-join).
    /// Variables that appear in both left and right should be matched.
    fn plan_join(
        &self,
        join: &crate::query::plan::JoinOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (left_op, left_columns) = self.plan_operator(&join.left)?;
        let (right_op, right_columns) = self.plan_operator(&join.right)?;

        let left_col_count = left_columns.len();

        // Find shared variables for equi-join
        let mut shared_vars: Vec<(usize, usize)> = Vec::new(); // (left_idx, right_idx)
        for (left_idx, left_col) in left_columns.iter().enumerate() {
            for (right_idx, right_col) in right_columns.iter().enumerate() {
                if left_col == right_col {
                    shared_vars.push((left_idx, right_idx));
                }
            }
        }

        // Build the full join output (all left + all right columns)
        let mut full_columns: Vec<String> = left_columns.clone();
        full_columns.extend(right_columns.clone());
        let full_schema = derive_rdf_schema(&full_columns);

        // Determine which columns to project (all left + non-duplicate right)
        let mut projection_indices: Vec<usize> = (0..left_col_count).collect();
        let mut output_columns = left_columns.clone();
        for (right_idx, right_col) in right_columns.iter().enumerate() {
            if !left_columns.contains(right_col) {
                projection_indices.push(left_col_count + right_idx);
                output_columns.push(right_col.clone());
            }
        }

        let join_type = if shared_vars.is_empty() {
            JoinType::Cross
        } else {
            JoinType::Inner
        };

        let join_condition: Option<Box<dyn JoinCondition>> = if shared_vars.is_empty() {
            None
        } else {
            Some(Box::new(RdfJoinCondition::new(shared_vars)))
        };

        let join_op = Box::new(NestedLoopJoinOperator::new(
            left_op,
            right_op,
            join_condition,
            join_type,
            full_schema,
        ));

        // If we have duplicate columns to remove, wrap with projection
        if projection_indices.len() < full_columns.len() {
            let output_schema = derive_rdf_schema(&output_columns);
            let project_op = Box::new(ProjectOperator::select_columns(
                join_op,
                projection_indices,
                output_schema,
            ));
            Ok((project_op, output_columns))
        } else {
            Ok((join_op, output_columns))
        }
    }

    /// Plans a LEFT JOIN operator (for SPARQL OPTIONAL).
    fn plan_left_join(&self, join: &LeftJoinOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (left_op, left_columns) = self.plan_operator(&join.left)?;
        let (right_op, right_columns) = self.plan_operator(&join.right)?;

        let left_col_count = left_columns.len();

        // Find shared variables for equi-join
        let mut shared_vars: Vec<(usize, usize)> = Vec::new();
        for (left_idx, left_col) in left_columns.iter().enumerate() {
            for (right_idx, right_col) in right_columns.iter().enumerate() {
                if left_col == right_col {
                    shared_vars.push((left_idx, right_idx));
                }
            }
        }

        // Build the full join output (all left + all right columns)
        let mut full_columns: Vec<String> = left_columns.clone();
        full_columns.extend(right_columns.clone());
        let full_schema = derive_rdf_schema(&full_columns);

        // Determine which columns to project (all left + non-duplicate right)
        let mut projection_indices: Vec<usize> = (0..left_col_count).collect();
        let mut output_columns = left_columns.clone();
        for (right_idx, right_col) in right_columns.iter().enumerate() {
            if !left_columns.contains(right_col) {
                projection_indices.push(left_col_count + right_idx);
                output_columns.push(right_col.clone());
            }
        }

        let join_condition: Option<Box<dyn JoinCondition>> = if shared_vars.is_empty() {
            None
        } else {
            Some(Box::new(RdfJoinCondition::new(shared_vars)))
        };

        let join_op = Box::new(NestedLoopJoinOperator::new(
            left_op,
            right_op,
            join_condition,
            JoinType::Left,
            full_schema,
        ));

        // If we have duplicate columns to remove, wrap with projection
        if projection_indices.len() < full_columns.len() {
            let output_schema = derive_rdf_schema(&output_columns);
            let project_op = Box::new(ProjectOperator::select_columns(
                join_op,
                projection_indices,
                output_schema,
            ));
            Ok((project_op, output_columns))
        } else {
            Ok((join_op, output_columns))
        }
    }

    /// Plans an ANTI JOIN operator (for SPARQL MINUS).
    ///
    /// Note: NestedLoopJoinOperator doesn't properly implement Anti join semantics.
    /// For now, we use HashJoinOperator for anti-joins which does support it.
    fn plan_anti_join(&self, join: &AntiJoinOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        use grafeo_core::execution::operators::HashJoinOperator;

        let (left_op, left_columns) = self.plan_operator(&join.left)?;
        let (right_op, right_columns) = self.plan_operator(&join.right)?;

        // Find shared variables for anti-join matching
        let mut left_keys: Vec<usize> = Vec::new();
        let mut right_keys: Vec<usize> = Vec::new();
        for (left_idx, left_col) in left_columns.iter().enumerate() {
            for (right_idx, right_col) in right_columns.iter().enumerate() {
                if left_col == right_col {
                    left_keys.push(left_idx);
                    right_keys.push(right_idx);
                }
            }
        }

        // Output is just left columns (anti-join filters out matching rows)
        let columns = left_columns.clone();
        let output_schema = derive_rdf_schema(&columns);

        // Use HashJoinOperator which properly implements Anti join
        let operator = Box::new(HashJoinOperator::new(
            left_op,
            right_op,
            left_keys,
            right_keys,
            JoinType::Anti,
            output_schema,
        ));
        Ok((operator, columns))
    }

    /// Plans a UNION operator.
    fn plan_union(
        &self,
        union: &crate::query::plan::UnionOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        if union.inputs.is_empty() {
            return Err(Error::Internal("Empty UNION".to_string()));
        }

        // For INSERT operations, we execute all operators in sequence
        let mut operators: Vec<Box<dyn Operator>> = Vec::new();
        let mut columns = Vec::new();

        for (i, input) in union.inputs.iter().enumerate() {
            let (op, cols) = self.plan_operator(input)?;
            operators.push(op);
            if i == 0 {
                columns = cols;
            }
        }

        if operators.len() == 1 {
            return Ok((operators.into_iter().next().unwrap(), columns));
        }

        // Create a chain operator that executes all operators in sequence
        let operator = Box::new(RdfUnionOperator::new(operators));
        Ok((operator, columns))
    }

    /// Plans an INSERT TRIPLE operator.
    fn plan_insert_triple(
        &self,
        insert: &InsertTripleOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Check if this is a pattern-based insert (has variables in the template)
        let has_variables = matches!(&insert.subject, TripleComponent::Variable(_))
            || matches!(&insert.predicate, TripleComponent::Variable(_))
            || matches!(&insert.object, TripleComponent::Variable(_));

        if has_variables {
            // Pattern-based insertion: need to query first, then insert each match
            if let Some(ref input) = insert.input {
                let (input_op, input_columns) = self.plan_operator(input)?;

                // Build column index map for variable substitution
                let column_map: HashMap<String, usize> = input_columns
                    .iter()
                    .enumerate()
                    .map(|(i, name)| (name.clone(), i))
                    .collect();

                let operator = Box::new(RdfInsertPatternOperator::new(
                    Arc::clone(&self.store),
                    input_op,
                    insert.subject.clone(),
                    insert.predicate.clone(),
                    insert.object.clone(),
                    column_map,
                ));

                return Ok((operator, Vec::new()));
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
            self.tx_id,
        ));

        // Insert operations don't output columns
        Ok((operator, Vec::new()))
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
                    _ => Literal::simple(format!("{:?}", value)),
                };
                Ok(Term::Literal(lit))
            }
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
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Check if this is a pattern-based delete (has variables in the template)
        let has_variables = matches!(&delete.subject, TripleComponent::Variable(_))
            || matches!(&delete.predicate, TripleComponent::Variable(_))
            || matches!(&delete.object, TripleComponent::Variable(_));

        if has_variables {
            // Pattern-based deletion: need to query first, then delete each match
            if let Some(ref input) = delete.input {
                let (input_op, input_columns) = self.plan_operator(input)?;

                // Build column index map for variable substitution
                let column_map: HashMap<String, usize> = input_columns
                    .iter()
                    .enumerate()
                    .map(|(i, name)| (name.clone(), i))
                    .collect();

                let operator = Box::new(RdfDeletePatternOperator::new(
                    Arc::clone(&self.store),
                    input_op,
                    delete.subject.clone(),
                    delete.predicate.clone(),
                    delete.object.clone(),
                    column_map,
                ));

                return Ok((operator, Vec::new()));
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
            self.tx_id,
        ));

        Ok((operator, Vec::new()))
    }

    /// Plans a CLEAR GRAPH operator.
    fn plan_clear_graph(&self, clear: &ClearGraphOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let operator = Box::new(RdfClearGraphOperator::new(
            Arc::clone(&self.store),
            clear.graph.clone(),
            clear.silent,
        ));
        Ok((operator, Vec::new()))
    }

    /// Plans a CREATE GRAPH operator.
    fn plan_create_graph(
        &self,
        create: &CreateGraphOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Named graphs are not yet fully supported in the RDF store
        // For now, CREATE GRAPH is a no-op (the graph is implicitly created when triples are added)
        let operator = Box::new(RdfNoOpOperator::new(create.silent));
        Ok((operator, Vec::new()))
    }

    /// Plans a DROP GRAPH operator.
    fn plan_drop_graph(&self, drop_op: &DropGraphOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // For default graph (None), clear all triples
        // For named graph, we would need named graph support
        if drop_op.graph.is_none() {
            let operator = Box::new(RdfClearGraphOperator::new(
                Arc::clone(&self.store),
                None,
                drop_op.silent,
            ));
            Ok((operator, Vec::new()))
        } else {
            // Named graphs not yet fully supported
            let operator = Box::new(RdfNoOpOperator::new(drop_op.silent));
            Ok((operator, Vec::new()))
        }
    }

    /// Plans a SPARQL MODIFY operator (DELETE/INSERT WHERE).
    ///
    /// Per SPARQL 1.1 spec:
    /// 1. Evaluate WHERE clause once to get bindings
    /// 2. Apply DELETE templates using those bindings
    /// 3. Apply INSERT templates using the SAME bindings
    fn plan_modify(&self, modify: &ModifyOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Plan the WHERE clause
        let (where_op, where_columns) = self.plan_operator(&modify.where_clause)?;

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
        ));

        Ok((operator, Vec::new()))
    }
}

// ============================================================================
// RDF Insert Triple Operator
// ============================================================================

/// Operator that inserts a triple into the RDF store.
struct RdfInsertTripleOperator {
    store: Arc<RdfStore>,
    triple: Triple,
    tx_id: Option<TxId>,
    inserted: bool,
}

impl RdfInsertTripleOperator {
    fn new(store: Arc<RdfStore>, triple: Triple, tx_id: Option<TxId>) -> Self {
        Self {
            store,
            triple,
            tx_id,
            inserted: false,
        }
    }
}

impl Operator for RdfInsertTripleOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.inserted {
            return Ok(None);
        }

        // Insert the triple (buffered if in a transaction)
        if let Some(tx_id) = self.tx_id {
            self.store.insert_in_tx(tx_id, self.triple.clone());
        } else {
            self.store.insert(self.triple.clone());
        }
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
}

impl RdfInsertPatternOperator {
    fn new(
        store: Arc<RdfStore>,
        input: Box<dyn Operator>,
        subject: TripleComponent,
        predicate: TripleComponent,
        object: TripleComponent,
        column_map: HashMap<String, usize>,
    ) -> Self {
        Self {
            store,
            input,
            subject,
            predicate,
            object,
            column_map,
            done: false,
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
            TripleComponent::Variable(name) => {
                // Remove the leading '?' if present
                let var_name = name.strip_prefix('?').unwrap_or(name);
                if let Some(&col_idx) = self.column_map.get(var_name) {
                    if let Some(col) = chunk.column(col_idx) {
                        if let Some(value) = col.get_value(row) {
                            return Self::value_to_term(&value);
                        }
                    }
                }
                None
            }
        }
    }

    fn value_to_term(value: &Value) -> Option<Term> {
        match value {
            Value::String(s) => {
                // Check if it looks like an IRI
                if s.starts_with("http://") || s.starts_with("https://") || s.starts_with("urn:") {
                    Some(Term::Iri(s.to_string().into()))
                } else if let Ok(n) = s.parse::<i64>() {
                    // Try to parse as integer
                    Some(Term::Literal(Literal::integer(n)))
                } else if let Ok(f) = s.parse::<f64>() {
                    // Try to parse as float
                    Some(Term::Literal(Literal::typed(
                        f.to_string(),
                        "http://www.w3.org/2001/XMLSchema#double",
                    )))
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
        for triple in triples_to_insert {
            self.store.insert(triple);
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
}

// ============================================================================
// RDF Delete Triple Operator
// ============================================================================

/// Operator that deletes a triple from the RDF store.
struct RdfDeleteTripleOperator {
    store: Arc<RdfStore>,
    triple: Triple,
    tx_id: Option<TxId>,
    deleted: bool,
}

impl RdfDeleteTripleOperator {
    fn new(store: Arc<RdfStore>, triple: Triple, tx_id: Option<TxId>) -> Self {
        Self {
            store,
            triple,
            tx_id,
            deleted: false,
        }
    }
}

impl Operator for RdfDeleteTripleOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.deleted {
            return Ok(None);
        }

        // Delete the triple (buffered if in a transaction)
        if let Some(tx_id) = self.tx_id {
            self.store.remove_in_tx(tx_id, self.triple.clone());
        } else {
            self.store.remove(&self.triple);
        }
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
}

impl RdfDeletePatternOperator {
    fn new(
        store: Arc<RdfStore>,
        input: Box<dyn Operator>,
        subject: TripleComponent,
        predicate: TripleComponent,
        object: TripleComponent,
        column_map: HashMap<String, usize>,
    ) -> Self {
        Self {
            store,
            input,
            subject,
            predicate,
            object,
            column_map,
            done: false,
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
            TripleComponent::Variable(name) => {
                // Remove the leading '?' if present
                let var_name = name.strip_prefix('?').unwrap_or(name);
                if let Some(&col_idx) = self.column_map.get(var_name) {
                    if let Some(col) = chunk.column(col_idx) {
                        if let Some(value) = col.get_value(row) {
                            return Self::value_to_term(&value);
                        }
                    }
                }
                None
            }
        }
    }

    fn value_to_term(value: &Value) -> Option<Term> {
        match value {
            Value::String(s) => {
                // Check if it looks like an IRI
                if s.starts_with("http://") || s.starts_with("https://") || s.starts_with("urn:") {
                    Some(Term::Iri(s.to_string().into()))
                } else if let Ok(n) = s.parse::<i64>() {
                    // Try to parse as integer
                    Some(Term::Literal(Literal::integer(n)))
                } else if let Ok(f) = s.parse::<f64>() {
                    // Try to parse as float
                    Some(Term::Literal(Literal::typed(
                        f.to_string(),
                        "http://www.w3.org/2001/XMLSchema#double",
                    )))
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
        for triple in triples_to_delete {
            self.store.remove(&triple);
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
}

// ============================================================================
// RDF Clear Graph Operator
// ============================================================================

/// Operator that clears triples from a graph in the RDF store.
struct RdfClearGraphOperator {
    store: Arc<RdfStore>,
    #[allow(dead_code)]
    graph: Option<String>,
    #[allow(dead_code)]
    silent: bool,
    cleared: bool,
}

impl RdfClearGraphOperator {
    fn new(store: Arc<RdfStore>, graph: Option<String>, silent: bool) -> Self {
        Self {
            store,
            graph,
            silent,
            cleared: false,
        }
    }
}

impl Operator for RdfClearGraphOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.cleared {
            return Ok(None);
        }

        // For now, clear all triples (named graph support would filter by graph)
        self.store.clear();
        self.cleared = true;

        Ok(None)
    }

    fn reset(&mut self) {
        self.cleared = false;
    }

    fn name(&self) -> &'static str {
        "RdfClearGraph"
    }
}

// ============================================================================
// RDF No-Op Operator
// ============================================================================

/// A no-op operator for operations that are not fully supported yet.
struct RdfNoOpOperator {
    #[allow(dead_code)]
    silent: bool,
    executed: bool,
}

impl RdfNoOpOperator {
    fn new(silent: bool) -> Self {
        Self {
            silent,
            executed: false,
        }
    }
}

impl Operator for RdfNoOpOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.executed {
            return Ok(None);
        }
        self.executed = true;
        Ok(None)
    }

    fn reset(&mut self) {
        self.executed = false;
    }

    fn name(&self) -> &'static str {
        "RdfNoOp"
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
}

impl RdfModifyOperator {
    fn new(
        store: Arc<RdfStore>,
        input: Box<dyn Operator>,
        delete_templates: Vec<TripleTemplate>,
        insert_templates: Vec<TripleTemplate>,
        column_map: HashMap<String, usize>,
    ) -> Self {
        Self {
            store,
            input,
            delete_templates,
            insert_templates,
            column_map,
            done: false,
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
            TripleComponent::Variable(name) => {
                let var_name = name.strip_prefix('?').unwrap_or(name);
                if let Some(&col_idx) = self.column_map.get(var_name) {
                    if let Some(col) = chunk.column(col_idx) {
                        if let Some(value) = col.get_value(row) {
                            return Self::value_to_term(&value);
                        }
                    }
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
                } else if let Ok(n) = s.parse::<i64>() {
                    Some(Term::Literal(Literal::integer(n)))
                } else if let Ok(f) = s.parse::<f64>() {
                    Some(Term::Literal(Literal::typed(
                        f.to_string(),
                        "http://www.w3.org/2001/XMLSchema#double",
                    )))
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

        // Step 2: Apply DELETE templates using collected bindings
        for template in &self.delete_templates {
            for (chunk, row) in &bindings {
                let subject = self.resolve_component(&template.subject, chunk, *row);
                let predicate = self.resolve_component(&template.predicate, chunk, *row);
                let object = self.resolve_component(&template.object, chunk, *row);

                if let (Some(s), Some(p), Some(o)) = (subject, predicate, object) {
                    let triple = Triple::new(s, p, o);
                    self.store.remove(&triple);
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
}

// ============================================================================
// RDF Triple Scan Operator
// ============================================================================

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
    /// Chunk size for batching.
    chunk_size: usize,
    /// Cached matching triples (lazily populated).
    triples: Option<Vec<Arc<Triple>>>,
    /// Current position in the triples.
    position: usize,
}

impl RdfTripleScanOperator {
    fn new(
        store: Arc<RdfStore>,
        pattern: TriplePattern,
        output_mask: [bool; 4],
        chunk_size: usize,
    ) -> Self {
        Self {
            store,
            pattern,
            output_mask,
            chunk_size,
            triples: None,
            position: 0,
        }
    }

    /// Lazily load matching triples on first access.
    fn ensure_triples(&mut self) {
        if self.triples.is_none() {
            self.triples = Some(self.store.find(&self.pattern));
        }
    }

    /// Count how many output columns we have.
    fn output_column_count(&self) -> usize {
        self.output_mask.iter().filter(|&&b| b).count()
    }
}

impl Operator for RdfTripleScanOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        self.ensure_triples();

        let triples = self.triples.as_ref().unwrap();

        if self.position >= triples.len() {
            return Ok(None);
        }

        let end = (self.position + self.chunk_size).min(triples.len());
        let batch_size = end - self.position;
        let col_count = self.output_column_count();

        // Create output schema (all String for RDF)
        let schema: Vec<LogicalType> = (0..col_count).map(|_| LogicalType::String).collect();
        let mut chunk = DataChunk::with_capacity(&schema, batch_size);

        // Fill the chunk
        for i in self.position..end {
            let triple = &triples[i];
            let mut col_idx = 0;

            if self.output_mask[0] {
                // Subject
                if let Some(col) = chunk.column_mut(col_idx) {
                    col.push_string(term_to_string(triple.subject()));
                }
                col_idx += 1;
            }
            if self.output_mask[1] {
                // Predicate
                if let Some(col) = chunk.column_mut(col_idx) {
                    col.push_string(term_to_string(triple.predicate()));
                }
                col_idx += 1;
            }
            if self.output_mask[2] {
                // Object
                if let Some(col) = chunk.column_mut(col_idx) {
                    push_term_value(col, triple.object());
                }
                col_idx += 1;
            }
            if self.output_mask[3] {
                // Graph (always null for now - named graphs not yet supported)
                if let Some(col) = chunk.column_mut(col_idx) {
                    col.push_value(Value::Null);
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
            // These expression types are not commonly used in RDF FILTER clauses
            FilterExpression::FunctionCall { .. }
            | FilterExpression::List(_)
            | FilterExpression::Case { .. }
            | FilterExpression::Map(_)
            | FilterExpression::IndexAccess { .. }
            | FilterExpression::SliceAccess { .. }
            | FilterExpression::ListComprehension { .. }
            | FilterExpression::ExistsSubquery { .. } => None,
        }
    }

    fn eval_binary_op(&self, left: &Value, op: BinaryFilterOp, right: &Value) -> Option<Value> {
        match op {
            BinaryFilterOp::And => Some(Value::Bool(left.as_bool()? && right.as_bool()?)),
            BinaryFilterOp::Or => Some(Value::Bool(left.as_bool()? || right.as_bool()?)),
            BinaryFilterOp::Xor => Some(Value::Bool(left.as_bool()? != right.as_bool()?)),
            BinaryFilterOp::Eq => Some(Value::Bool(left == right)),
            BinaryFilterOp::Ne => Some(Value::Bool(left != right)),
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
                // Regex matching - not yet implemented for RDF
                // Would need regex crate for full support
                None
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
        }
    }
}

impl Predicate for RdfExpressionPredicate {
    fn evaluate(&self, chunk: &DataChunk, row: usize) -> bool {
        matches!(self.eval(chunk, row), Some(Value::Bool(true)))
    }
}

// ============================================================================
// RDF Join Condition
// ============================================================================

/// Join condition for joining on shared variables in SPARQL.
///
/// This condition checks that shared variables have equal values between
/// left and right sides of a join.
struct RdfJoinCondition {
    /// Pairs of (left_col_idx, right_col_idx) for shared variables
    shared_vars: Vec<(usize, usize)>,
}

impl RdfJoinCondition {
    fn new(shared_vars: Vec<(usize, usize)>) -> Self {
        Self { shared_vars }
    }
}

impl JoinCondition for RdfJoinCondition {
    fn evaluate(
        &self,
        left_chunk: &DataChunk,
        left_row: usize,
        right_chunk: &DataChunk,
        right_row: usize,
    ) -> bool {
        // Check that all shared variables have equal values
        for (left_idx, right_idx) in &self.shared_vars {
            let left_val = left_chunk
                .column(*left_idx)
                .and_then(|c| c.get_value(left_row));
            let right_val = right_chunk
                .column(*right_idx)
                .and_then(|c| c.get_value(right_row));

            match (left_val, right_val) {
                (Some(l), Some(r)) => {
                    if l != r {
                        return false;
                    }
                }
                // If either is null/missing, they don't match
                _ => return false,
            }
        }
        true
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Converts an RDF Term to a string for IRI/blank node representation.
fn term_to_string(term: &Term) -> String {
    match term {
        Term::Iri(iri) => iri.as_str().to_string(),
        Term::BlankNode(bnode) => format!("_:{}", bnode.id()),
        Term::Literal(lit) => lit.value().to_string(),
    }
}

/// Pushes an RDF term value to a column, preserving type where possible.
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
    }
}

/// Converts a TripleComponent to an Option<Term> for pattern matching.
fn component_to_term(component: &TripleComponent) -> Option<Term> {
    match component {
        TripleComponent::Variable(_) => None,
        TripleComponent::Iri(iri) => Some(Term::iri(iri.clone())),
        TripleComponent::Literal(value) => match value {
            Value::String(s) => Some(Term::literal(Arc::clone(s))),
            Value::Int64(i) => Some(Term::typed_literal(i.to_string(), Literal::XSD_INTEGER)),
            Value::Float64(f) => Some(Term::typed_literal(f.to_string(), Literal::XSD_DOUBLE)),
            Value::Bool(b) => Some(Term::typed_literal(b.to_string(), Literal::XSD_BOOLEAN)),
            _ => Some(Term::literal(value.to_string())),
        },
    }
}

/// Derives RDF schema (all String type for simplicity).
fn derive_rdf_schema(columns: &[String]) -> Vec<LogicalType> {
    columns.iter().map(|_| LogicalType::String).collect()
}

/// Resolves an expression to a column index.
fn resolve_expression(
    expr: &LogicalExpression,
    variable_columns: &HashMap<String, usize>,
) -> Result<usize> {
    match expr {
        LogicalExpression::Variable(name) => variable_columns
            .get(name)
            .copied()
            .ok_or_else(|| Error::Internal(format!("Variable '{}' not found", name))),
        _ => Err(Error::Internal(format!(
            "Cannot resolve expression to column: {:?}",
            expr
        ))),
    }
}

/// Converts an expression to a string for column naming.
fn expression_to_string(expr: &LogicalExpression) -> String {
    match expr {
        LogicalExpression::Variable(name) => name.clone(),
        LogicalExpression::Property { variable, property } => format!("{variable}.{property}"),
        LogicalExpression::Literal(value) => format!("{value:?}"),
        _ => "expr".to_string(),
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::plan::LogicalPlan;

    #[test]
    fn test_rdf_planner_simple_scan() {
        let store = Arc::new(RdfStore::new());

        store.insert(Triple::new(
            Term::iri("http://example.org/alice"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alice"),
        ));

        let planner = RdfPlanner::new(store);

        let scan = TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Variable("p".to_string()),
            object: TripleComponent::Variable("o".to_string()),
            graph: None,
            input: None,
        };

        let plan = LogicalPlan::new(LogicalOperator::TripleScan(scan));
        let physical = planner.plan(&plan).unwrap();

        assert_eq!(physical.columns, vec!["s", "p", "o"]);
    }

    #[test]
    fn test_rdf_planner_with_pattern() {
        let store = Arc::new(RdfStore::new());

        store.insert(Triple::new(
            Term::iri("http://example.org/alice"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alice"),
        ));
        store.insert(Triple::new(
            Term::iri("http://example.org/bob"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Bob"),
        ));
        store.insert(Triple::new(
            Term::iri("http://example.org/alice"),
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

        let mut operator =
            RdfTripleScanOperator::new(Arc::clone(&store), pattern, [true, true, true, false], 30);

        let mut total_rows = 0;
        while let Ok(Some(chunk)) = operator.next() {
            total_rows += chunk.row_count();
            assert!(chunk.row_count() <= 30); // Respects chunk size
        }

        assert_eq!(total_rows, 100);
    }
}
