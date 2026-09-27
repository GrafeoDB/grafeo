//! Aggregate and factorized aggregate planning.

use super::{
    AggregateOp, Arc, Direction, Error, ExpandDirection, ExpandStep, ExpressionPredicate,
    FactorizedAggregate, FactorizedAggregateOperator, FilterExpression, FilterOperator,
    GraphStoreSearch, HashAggregateOperator, HashMap, LazyFactorizedChainOperator,
    LogicalAggregateFunction, LogicalExpression, LogicalType, Operator, PhysicalAggregateExpr,
    ProjectExpr, ProjectOperator, Result, SimpleAggregateOperator, convert_aggregate_function,
    expression_to_string, resolved_column_name,
};

impl super::Planner {
    /// Plans an AGGREGATE operator.
    pub(super) fn plan_aggregate(
        &self,
        agg: &AggregateOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
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

        // Collect all extra projections (property access and complex expressions)
        // in a single ordered list so that column index assignment matches the
        // order they are added to the ProjectOperator.
        enum ExtraProjection {
            Property { variable: String, property: String },
            Expression { filter_expr: FilterExpression },
        }
        let mut extra_projections: Vec<ExtraProjection> = Vec::new();
        let mut next_col_idx = input_columns.len();

        // Check group-by expressions for properties and complex expressions
        // (Labels, Type, FunctionCall, IndexAccess, etc.)
        for expr in &agg.group_by {
            match expr {
                LogicalExpression::Property { variable, property } => {
                    let col_name = resolved_column_name(expr);
                    if !variable_columns.contains_key(&col_name) {
                        extra_projections.push(ExtraProjection::Property {
                            variable: variable.clone(),
                            property: property.clone(),
                        });
                        variable_columns.insert(col_name, next_col_idx);
                        next_col_idx += 1;
                    }
                }
                LogicalExpression::Variable(_) => {
                    // Already in variable_columns, nothing to project
                }
                _ => {
                    // Complex expression (Labels, Type, FunctionCall, IndexAccess,
                    // CASE, Binary, etc.): project as computed column
                    let col_name = resolved_column_name(expr);
                    if !variable_columns.contains_key(&col_name) {
                        let filter_expr = self.convert_expression(expr)?;
                        extra_projections.push(ExtraProjection::Expression { filter_expr });
                        variable_columns.insert(col_name, next_col_idx);
                        next_col_idx += 1;
                    }
                }
            }
        }

        // Check aggregate expressions for properties and complex expressions
        // (both first and second arguments)
        for agg_expr in &agg.aggregates {
            for expr_opt in [&agg_expr.expression, &agg_expr.expression2] {
                let Some(expr) = expr_opt else { continue };
                match expr {
                    LogicalExpression::Property { variable, property } => {
                        let col_name = resolved_column_name(expr);
                        if !variable_columns.contains_key(&col_name) {
                            extra_projections.push(ExtraProjection::Property {
                                variable: variable.clone(),
                                property: property.clone(),
                            });
                            variable_columns.insert(col_name, next_col_idx);
                            next_col_idx += 1;
                        }
                    }
                    LogicalExpression::Variable(_) => {
                        // Already in variable_columns, nothing to project
                    }
                    _ => {
                        // Complex expression (CASE, Binary, etc.): project as computed column
                        let col_name = resolved_column_name(expr);
                        if !variable_columns.contains_key(&col_name) {
                            let filter_expr = self.convert_expression(expr)?;
                            extra_projections.push(ExtraProjection::Expression { filter_expr });
                            variable_columns.insert(col_name, next_col_idx);
                            next_col_idx += 1;
                        }
                    }
                }
            }
        }

        // If we have extra projections, add a projection to materialize them
        if !extra_projections.is_empty() {
            let mut projections = Vec::new();
            let mut output_types = Vec::new();

            // First, pass through all existing columns (use Node type to preserve node IDs
            // for subsequent property access - nodes need VectorData::NodeId for get_node_id())
            for (i, _) in input_columns.iter().enumerate() {
                projections.push(ProjectExpr::Column(i));
                output_types.push(LogicalType::Node);
            }

            // Add extra projections in the same order as index assignment
            for proj in &extra_projections {
                match proj {
                    ExtraProjection::Property {
                        variable, property, ..
                    } => {
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
                        output_types.push(LogicalType::Any);
                    }
                    ExtraProjection::Expression { filter_expr, .. } => {
                        projections.push(ProjectExpr::Expression {
                            expr: filter_expr.clone(),
                            variable_columns: variable_columns.clone(),
                        });
                        output_types.push(LogicalType::Any);
                    }
                }
            }

            input_op = Box::new(
                ProjectOperator::with_store(
                    input_op,
                    projections,
                    output_types,
                    Arc::clone(&self.store) as Arc<dyn GraphStoreSearch>,
                )
                .with_transaction_context(self.viewing_epoch, self.transaction_id)
                .with_session_context(self.session_context.clone()),
            );
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

                let column2 = agg_expr
                    .expression2
                    .as_ref()
                    .map(|e| {
                        self.resolve_expression_to_column_with_properties(e, &variable_columns)
                    })
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
                LogicalAggregateFunction::Sum => LogicalType::Any,
                LogicalAggregateFunction::Avg => LogicalType::Float64,
                LogicalAggregateFunction::Min | LogicalAggregateFunction::Max => {
                    // MIN/MAX preserve input type: the result can be any type
                    // (Int64, Float64, String, Date, etc.), so use Any/Generic
                    // to avoid type mismatch when pushing the finalized value.
                    LogicalType::Any
                }
                LogicalAggregateFunction::Collect => LogicalType::Any, // List type (using Any since List is a complex type)
                LogicalAggregateFunction::GroupConcat => LogicalType::String,
                LogicalAggregateFunction::Sample => LogicalType::Any,
                // Statistical functions return Float64
                LogicalAggregateFunction::StdDev
                | LogicalAggregateFunction::StdDevPop
                | LogicalAggregateFunction::Variance
                | LogicalAggregateFunction::VariancePop
                | LogicalAggregateFunction::PercentileDisc
                | LogicalAggregateFunction::PercentileCont
                | LogicalAggregateFunction::CovarSamp
                | LogicalAggregateFunction::CovarPop
                | LogicalAggregateFunction::Corr
                | LogicalAggregateFunction::RegrSlope
                | LogicalAggregateFunction::RegrIntercept
                | LogicalAggregateFunction::RegrR2
                | LogicalAggregateFunction::RegrSxx
                | LogicalAggregateFunction::RegrSyy
                | LogicalAggregateFunction::RegrSxy
                | LogicalAggregateFunction::RegrAvgx
                | LogicalAggregateFunction::RegrAvgy => LogicalType::Float64,
                // REGR_COUNT returns Int64
                LogicalAggregateFunction::RegrCount => LogicalType::Int64,
            };
            output_schema.push(result_type);
            output_columns.push(
                agg_expr
                    .alias
                    .clone()
                    .unwrap_or_else(|| format!("{:?}(...)", agg_expr.function).to_lowercase()),
            );
        }

        // Register all aggregate output columns as scalar (group-by values and
        // aggregate results are materialized scalar values, not entity references)
        for col in &output_columns {
            self.scalar_columns.borrow_mut().insert(col.clone());
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
            let predicate = ExpressionPredicate::new(
                filter_expr,
                having_var_columns,
                Arc::clone(&self.store) as Arc<dyn GraphStoreSearch>,
            )
            .with_transaction_context(self.viewing_epoch, self.transaction_id)
            .with_session_context(self.session_context.clone());
            operator = Box::new(FilterOperator::new(operator, Box::new(predicate)));
        }

        Ok((operator, output_columns))
    }

    /// Checks if an aggregate is simple enough for factorized execution.
    ///
    /// Simple aggregates:
    /// - COUNT(*) or COUNT(variable)
    /// - SUM, AVG, MIN, MAX on variables (not properties for now)
    pub(super) fn is_simple_aggregate(&self, agg: &AggregateOp) -> bool {
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
    pub(super) fn plan_factorized_aggregate(
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
                edge_types: expand.edge_types.clone(),
            });

            let edge_col_name = self.register_edge_column(&expand.edge_variable);
            columns.push(edge_col_name);
            columns.push(expand.to_variable.clone());

            is_first = false;
        }

        // Create the lazy factorized chain operator
        let mut lazy_op = LazyFactorizedChainOperator::new(
            Arc::clone(&self.store) as Arc<dyn GraphStoreSearch>,
            base_op,
            steps,
        );

        if let Some(transaction_id) = self.transaction_id {
            lazy_op = lazy_op.with_transaction_context(self.viewing_epoch, Some(transaction_id));
        } else {
            lazy_op = lazy_op.with_transaction_context(self.viewing_epoch, None);
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

        // Register output columns as scalar (aggregate results are materialized
        // scalar values, not entity references). Without this, a post-Return
        // projection would treat them as node IDs and attempt NodeResolve, which
        // corrupts the result on 3+ hop queries.
        for col in &output_columns {
            self.scalar_columns.borrow_mut().insert(col.clone());
        }

        // Create the factorized aggregate operator
        let factorized_agg_op = FactorizedAggregateOperator::new(lazy_op, factorized_aggs);

        Ok((Box::new(factorized_agg_op), output_columns))
    }

    /// Resolves a logical expression to a column index, using projected property columns.
    ///
    /// This is used for aggregations where properties have been projected into their own columns.
    pub(super) fn resolve_expression_to_column_with_properties(
        &self,
        expr: &LogicalExpression,
        variable_columns: &HashMap<String, usize>,
    ) -> Result<usize> {
        crate::query::planner::common::resolve_expression_to_column(expr, variable_columns, "")
    }
}
