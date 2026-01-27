//! Logical query planner.
//!
//! Converts a logical plan into a physical plan (tree of operators).

use crate::query::plan::{
    AggregateFunction as LogicalAggregateFunction,
    AggregateOp, BinaryOp, ExpandDirection, ExpandOp, FilterOp, LimitOp, LogicalExpression,
    LogicalOperator, LogicalPlan, NodeScanOp, ReturnOp, SkipOp, SortOp, SortOrder,
    UnaryOp,
};
use graphos_common::utils::error::{Error, Result};
use graphos_common::types::LogicalType;
use graphos_core::execution::operators::{
    AggregateExpr as PhysicalAggregateExpr, AggregateFunction as PhysicalAggregateFunction,
    BinaryFilterOp, ExpandOperator, ExpressionPredicate, FilterExpression, FilterOperator,
    HashAggregateOperator, LimitOperator, NullOrder, Operator, ProjectExpr, ProjectOperator,
    ScanOperator, SimpleAggregateOperator, SkipOperator, SortDirection,
    SortKey as PhysicalSortKey, SortOperator, UnaryFilterOp,
};
use graphos_core::graph::{lpg::LpgStore, Direction};
use std::collections::HashMap;
use std::sync::Arc;

/// Converts a logical plan to a physical operator tree.
pub struct Planner {
    /// The graph store to scan from.
    store: Arc<LpgStore>,
}

impl Planner {
    /// Creates a new planner with the given store.
    #[must_use]
    pub fn new(store: Arc<LpgStore>) -> Self {
        Self { store }
    }

    /// Plans a logical plan into a physical operator.
    ///
    /// # Errors
    ///
    /// Returns an error if planning fails.
    pub fn plan(&self, logical_plan: &LogicalPlan) -> Result<PhysicalPlan> {
        let (operator, columns) = self.plan_operator(&logical_plan.root)?;
        Ok(PhysicalPlan { operator, columns })
    }

    /// Plans a single logical operator.
    fn plan_operator(&self, op: &LogicalOperator) -> Result<(Box<dyn Operator>, Vec<String>)> {
        match op {
            LogicalOperator::NodeScan(scan) => self.plan_node_scan(scan),
            LogicalOperator::Expand(expand) => self.plan_expand(expand),
            LogicalOperator::Return(ret) => self.plan_return(ret),
            LogicalOperator::Filter(filter) => self.plan_filter(filter),
            LogicalOperator::Project(project) => {
                // For now, just plan the input
                self.plan_operator(&project.input)
            }
            LogicalOperator::Limit(limit) => self.plan_limit(limit),
            LogicalOperator::Skip(skip) => self.plan_skip(skip),
            LogicalOperator::Sort(sort) => self.plan_sort(sort),
            LogicalOperator::Aggregate(agg) => self.plan_aggregate(agg),
            LogicalOperator::Empty => Err(Error::Internal("Empty plan".to_string())),
            _ => Err(Error::Internal(format!(
                "Unsupported operator: {:?}",
                std::mem::discriminant(op)
            ))),
        }
    }

    /// Plans a node scan operator.
    fn plan_node_scan(&self, scan: &NodeScanOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let operator: Box<dyn Operator> = if let Some(label) = &scan.label {
            Box::new(ScanOperator::with_label(Arc::clone(&self.store), label))
        } else {
            Box::new(ScanOperator::new(Arc::clone(&self.store)))
        };

        let columns = vec![scan.variable.clone()];

        // If there's an input, we'd need to chain operators
        // For now, just return the scan
        Ok((operator, columns))
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

        // Create the expand operator
        let operator = Box::new(ExpandOperator::new(
            Arc::clone(&self.store),
            input_op,
            source_column,
            direction,
            expand.edge_type.clone(),
        ));

        // Build output columns: source, [edge], target
        let mut columns = vec![expand.from_variable.clone()];
        if let Some(ref edge_var) = expand.edge_variable {
            columns.push(edge_var.clone());
        }
        columns.push(expand.to_variable.clone());

        Ok((operator, columns))
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
        let needs_project = ret.items.iter().any(|item| {
            !matches!(&item.expression, LogicalExpression::Variable(_))
        });

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
                        // Property could be any type - use String as default
                        output_types.push(LogicalType::String);
                    }
                    LogicalExpression::Literal(value) => {
                        projections.push(ProjectExpr::Constant(value.clone()));
                        output_types.push(value_to_logical_type(value));
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
                && projections.iter().enumerate().all(|(i, p)| {
                    matches!(p, ProjectExpr::Column(c) if *c == i)
                })
            {
                // No reordering needed
                Ok((input_op, columns))
            } else {
                let operator = Box::new(ProjectOperator::new(input_op, projections, output_types));
                Ok((operator, columns))
            }
        }
    }

    /// Plans a filter operator.
    fn plan_filter(&self, filter: &FilterOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
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
        let predicate = ExpressionPredicate::new(
            filter_expr,
            variable_columns,
            Arc::clone(&self.store),
        );

        // Create the filter operator
        let operator = Box::new(FilterOperator::new(input_op, Box::new(predicate)));

        Ok((operator, columns))
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
        let (input_op, columns) = self.plan_operator(&sort.input)?;

        // Build variable to column index mapping
        let variable_columns: HashMap<String, usize> = columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        // Convert logical sort keys to physical sort keys
        let physical_keys: Vec<PhysicalSortKey> = sort
            .keys
            .iter()
            .map(|key| {
                let col_idx = self.resolve_sort_expression(&key.expression, &variable_columns)?;
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

        let output_schema = self.derive_schema_from_columns(&columns);
        let operator = Box::new(SortOperator::new(input_op, physical_keys, output_schema));
        Ok((operator, columns))
    }

    /// Resolves a sort expression to a column index.
    fn resolve_sort_expression(
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
            LogicalExpression::Property { variable, .. } => {
                // For property access, find the variable column
                variable_columns.get(variable).copied().ok_or_else(|| {
                    Error::Internal(format!("Variable '{}' not found for ORDER BY", variable))
                })
            }
            _ => Err(Error::Internal(format!(
                "Unsupported ORDER BY expression: {:?}",
                expr
            ))),
        }
    }

    /// Derives a schema from column names (assumes Node type as default).
    fn derive_schema_from_columns(&self, columns: &[String]) -> Vec<LogicalType> {
        columns.iter().map(|_| LogicalType::Node).collect()
    }

    /// Plans an AGGREGATE operator.
    fn plan_aggregate(&self, agg: &AggregateOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, input_columns) = self.plan_operator(&agg.input)?;

        // Build variable to column index mapping
        let variable_columns: HashMap<String, usize> = input_columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        // Convert group-by expressions to column indices
        let group_columns: Vec<usize> = agg
            .group_by
            .iter()
            .map(|expr| self.resolve_expression_to_column(expr, &variable_columns))
            .collect::<Result<Vec<_>>>()?;

        // Convert aggregate expressions to physical form
        let physical_aggregates: Vec<PhysicalAggregateExpr> = agg
            .aggregates
            .iter()
            .map(|agg_expr| {
                let column = agg_expr
                    .expression
                    .as_ref()
                    .map(|e| self.resolve_expression_to_column(e, &variable_columns))
                    .transpose()?;

                Ok(PhysicalAggregateExpr {
                    function: convert_aggregate_function(agg_expr.function),
                    column,
                    distinct: agg_expr.distinct,
                    alias: agg_expr.alias.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // Build output schema and column names
        let mut output_schema = Vec::new();
        let mut output_columns = Vec::new();

        // Add group-by columns
        for (idx, expr) in agg.group_by.iter().enumerate() {
            output_schema.push(LogicalType::Node); // Default type
            output_columns.push(expression_to_string(expr));
            // If there's a group column, we need to track it
            if idx < group_columns.len() {
                // Group column preserved
            }
        }

        // Add aggregate result columns
        for agg_expr in &agg.aggregates {
            let result_type = match agg_expr.function {
                LogicalAggregateFunction::Count => LogicalType::Int64,
                LogicalAggregateFunction::Sum => LogicalType::Int64,
                LogicalAggregateFunction::Avg => LogicalType::Float64,
                LogicalAggregateFunction::Min | LogicalAggregateFunction::Max => {
                    LogicalType::Node // Preserves input type
                }
                LogicalAggregateFunction::Collect => LogicalType::String, // List type
            };
            output_schema.push(result_type);
            output_columns.push(agg_expr.alias.clone().unwrap_or_else(|| {
                format!("{:?}(...)", agg_expr.function).to_lowercase()
            }));
        }

        // Choose operator based on whether there are group-by columns
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

    /// Resolves a logical expression to a column index.
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

    /// Converts a logical expression to a filter expression.
    fn convert_expression(&self, expr: &LogicalExpression) -> Result<FilterExpression> {
        match expr {
            LogicalExpression::Literal(v) => Ok(FilterExpression::Literal(v.clone())),
            LogicalExpression::Variable(name) => Ok(FilterExpression::Variable(name.clone())),
            LogicalExpression::Property { variable, property } => {
                Ok(FilterExpression::Property {
                    variable: variable.clone(),
                    property: property.clone(),
                })
            }
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
            LogicalExpression::FunctionCall { .. } => Err(Error::Internal(
                "Function calls not yet supported in filters".to_string(),
            )),
            LogicalExpression::Case { .. } => Err(Error::Internal(
                "CASE expressions not yet supported in filters".to_string(),
            )),
            LogicalExpression::List(_) => Err(Error::Internal(
                "List expressions not yet supported in filters".to_string(),
            )),
            LogicalExpression::Parameter(_) => Err(Error::Internal(
                "Parameters not yet supported in filters".to_string(),
            )),
            LogicalExpression::Labels(_) => Err(Error::Internal(
                "labels() function not yet supported in filters".to_string(),
            )),
            LogicalExpression::Type(_) => Err(Error::Internal(
                "type() function not yet supported in filters".to_string(),
            )),
            LogicalExpression::Id(_) => Err(Error::Internal(
                "id() function not yet supported in filters".to_string(),
            )),
        }
    }
}

/// Converts a logical binary operator to a filter binary operator.
fn convert_binary_op(op: BinaryOp) -> Result<BinaryFilterOp> {
    match op {
        BinaryOp::Eq => Ok(BinaryFilterOp::Eq),
        BinaryOp::Ne => Ok(BinaryFilterOp::Ne),
        BinaryOp::Lt => Ok(BinaryFilterOp::Lt),
        BinaryOp::Le => Ok(BinaryFilterOp::Le),
        BinaryOp::Gt => Ok(BinaryFilterOp::Gt),
        BinaryOp::Ge => Ok(BinaryFilterOp::Ge),
        BinaryOp::And => Ok(BinaryFilterOp::And),
        BinaryOp::Or => Ok(BinaryFilterOp::Or),
        BinaryOp::Xor | BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div
        | BinaryOp::Mod | BinaryOp::Contains | BinaryOp::StartsWith
        | BinaryOp::EndsWith | BinaryOp::Concat | BinaryOp::In | BinaryOp::Like => {
            Err(Error::Internal(format!(
                "Binary operator {:?} not yet supported in filters",
                op
            )))
        }
    }
}

/// Converts a logical unary operator to a filter unary operator.
fn convert_unary_op(op: UnaryOp) -> Result<UnaryFilterOp> {
    match op {
        UnaryOp::Not => Ok(UnaryFilterOp::Not),
        UnaryOp::IsNull => Ok(UnaryFilterOp::IsNull),
        UnaryOp::IsNotNull => Ok(UnaryFilterOp::IsNotNull),
        UnaryOp::Neg => Err(Error::Internal(
            "Negation not supported in filter predicates".to_string(),
        )),
    }
}

/// Converts a logical aggregate function to a physical aggregate function.
fn convert_aggregate_function(func: LogicalAggregateFunction) -> PhysicalAggregateFunction {
    match func {
        LogicalAggregateFunction::Count => PhysicalAggregateFunction::Count,
        LogicalAggregateFunction::Sum => PhysicalAggregateFunction::Sum,
        LogicalAggregateFunction::Avg => PhysicalAggregateFunction::Avg,
        LogicalAggregateFunction::Min => PhysicalAggregateFunction::Min,
        LogicalAggregateFunction::Max => PhysicalAggregateFunction::Max,
        LogicalAggregateFunction::Collect => PhysicalAggregateFunction::Collect,
    }
}

/// Infers the logical type from a value.
fn value_to_logical_type(value: &graphos_common::types::Value) -> LogicalType {
    use graphos_common::types::Value;
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::plan::{NodeScanOp, ReturnItem, ReturnOp};

    #[test]
    fn test_plan_simple_scan() {
        let store = Arc::new(LpgStore::new());
        store.create_node(&["Person"]);
        store.create_node(&["Person"]);

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
}
