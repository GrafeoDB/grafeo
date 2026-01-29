//! GQL to LogicalPlan translator.
//!
//! Translates GQL AST to the common logical plan representation.

use crate::query::plan::{
    AddLabelOp, AggregateExpr, AggregateFunction, AggregateOp, BinaryOp, DeleteNodeOp, DistinctOp,
    ExpandDirection, ExpandOp, FilterOp, JoinOp, JoinType, LeftJoinOp, LimitOp, LogicalExpression,
    LogicalOperator, LogicalPlan, NodeScanOp, ProjectOp, Projection, ReturnItem, ReturnOp,
    SetPropertyOp, SkipOp, SortKey, SortOp, SortOrder, UnaryOp, UnwindOp,
};
use grafeo_adapters::query::gql::{self, ast};
use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, Result};

/// Translates a GQL query string to a logical plan.
///
/// # Errors
///
/// Returns an error if the query cannot be parsed or translated.
pub fn translate(query: &str) -> Result<LogicalPlan> {
    let statement = gql::parse(query)?;
    let translator = GqlTranslator::new();
    translator.translate_statement(&statement)
}

/// Translator from GQL AST to LogicalPlan.
struct GqlTranslator;

impl GqlTranslator {
    fn new() -> Self {
        Self
    }

    fn translate_statement(&self, stmt: &ast::Statement) -> Result<LogicalPlan> {
        match stmt {
            ast::Statement::Query(query) => self.translate_query(query),
            ast::Statement::DataModification(dm) => self.translate_data_modification(dm),
            ast::Statement::Schema(_) => Err(Error::Internal(
                "Schema statements not yet supported".to_string(),
            )),
        }
    }

    fn translate_query(&self, query: &ast::QueryStatement) -> Result<LogicalPlan> {
        // Start with the pattern scan (MATCH clauses)
        let mut plan = LogicalOperator::Empty;

        for match_clause in &query.match_clauses {
            let match_plan = self.translate_match(match_clause)?;
            if matches!(plan, LogicalOperator::Empty) {
                plan = match_plan;
            } else if match_clause.optional {
                // OPTIONAL MATCH uses LEFT JOIN semantics
                plan = LogicalOperator::LeftJoin(LeftJoinOp {
                    left: Box::new(plan),
                    right: Box::new(match_plan),
                    condition: None,
                });
            } else {
                // Regular MATCH - combine with cross join (implicit join on shared variables)
                plan = LogicalOperator::Join(JoinOp {
                    left: Box::new(plan),
                    right: Box::new(match_plan),
                    join_type: JoinType::Cross,
                    conditions: vec![],
                });
            }
        }

        // Handle UNWIND clauses
        for unwind_clause in &query.unwind_clauses {
            let expression = self.translate_expression(&unwind_clause.expression)?;
            plan = LogicalOperator::Unwind(UnwindOp {
                expression,
                variable: unwind_clause.alias.clone(),
                input: Box::new(plan),
            });
        }

        // Apply WHERE filter
        if let Some(where_clause) = &query.where_clause {
            let predicate = self.translate_expression(&where_clause.expression)?;
            plan = LogicalOperator::Filter(FilterOp {
                predicate,
                input: Box::new(plan),
            });
        }

        // Handle SET clauses
        for set_clause in &query.set_clauses {
            // Handle property assignments
            for assignment in &set_clause.assignments {
                let value = self.translate_expression(&assignment.value)?;
                plan = LogicalOperator::SetProperty(SetPropertyOp {
                    variable: assignment.variable.clone(),
                    properties: vec![(assignment.property.clone(), value)],
                    replace: false,
                    input: Box::new(plan),
                });
            }
            // Handle label operations (SET n:Label)
            for label_op in &set_clause.label_operations {
                plan = LogicalOperator::AddLabel(AddLabelOp {
                    variable: label_op.variable.clone(),
                    labels: label_op.labels.clone(),
                    input: Box::new(plan),
                });
            }
        }

        // Handle WITH clauses (projection for query chaining)
        for with_clause in &query.with_clauses {
            let projections: Vec<Projection> = with_clause
                .items
                .iter()
                .map(|item| {
                    Ok(Projection {
                        expression: self.translate_expression(&item.expression)?,
                        alias: item.alias.clone(),
                    })
                })
                .collect::<Result<_>>()?;

            plan = LogicalOperator::Project(ProjectOp {
                projections,
                input: Box::new(plan),
            });

            // Apply WHERE filter if present in WITH clause
            if let Some(where_clause) = &with_clause.where_clause {
                let predicate = self.translate_expression(&where_clause.expression)?;
                plan = LogicalOperator::Filter(FilterOp {
                    predicate,
                    input: Box::new(plan),
                });
            }

            // Handle DISTINCT
            if with_clause.distinct {
                plan = LogicalOperator::Distinct(DistinctOp {
                    input: Box::new(plan),
                });
            }
        }

        // Apply SKIP
        if let Some(skip_expr) = &query.return_clause.skip {
            if let ast::Expression::Literal(ast::Literal::Integer(n)) = skip_expr {
                plan = LogicalOperator::Skip(SkipOp {
                    count: *n as usize,
                    input: Box::new(plan),
                });
            }
        }

        // Apply LIMIT
        if let Some(limit_expr) = &query.return_clause.limit {
            if let ast::Expression::Literal(ast::Literal::Integer(n)) = limit_expr {
                plan = LogicalOperator::Limit(LimitOp {
                    count: *n as usize,
                    input: Box::new(plan),
                });
            }
        }

        // Check if RETURN contains aggregate functions
        let has_aggregates = query
            .return_clause
            .items
            .iter()
            .any(|item| contains_aggregate(&item.expression));

        if has_aggregates {
            // Extract aggregate and group-by expressions
            let (aggregates, group_by) =
                self.extract_aggregates_and_groups(&query.return_clause.items)?;

            // Insert Aggregate operator - this is the final operator for aggregate queries
            // The aggregate operator produces the output columns directly
            plan = LogicalOperator::Aggregate(AggregateOp {
                group_by,
                aggregates,
                input: Box::new(plan),
            });

            // Note: For aggregate queries, we don't add a Return operator
            // because Aggregate already produces the final output
        } else {
            // Apply ORDER BY
            if let Some(order_by) = &query.return_clause.order_by {
                let keys = order_by
                    .items
                    .iter()
                    .map(|item| {
                        Ok(SortKey {
                            expression: self.translate_expression(&item.expression)?,
                            order: match item.order {
                                ast::SortOrder::Asc => SortOrder::Ascending,
                                ast::SortOrder::Desc => SortOrder::Descending,
                            },
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;

                plan = LogicalOperator::Sort(SortOp {
                    keys,
                    input: Box::new(plan),
                });
            }

            // Apply RETURN
            let return_items = query
                .return_clause
                .items
                .iter()
                .map(|item| {
                    Ok(ReturnItem {
                        expression: self.translate_expression(&item.expression)?,
                        alias: item.alias.clone(),
                    })
                })
                .collect::<Result<Vec<_>>>()?;

            plan = LogicalOperator::Return(ReturnOp {
                items: return_items,
                distinct: query.return_clause.distinct,
                input: Box::new(plan),
            });
        }

        Ok(LogicalPlan::new(plan))
    }

    /// Builds return items for an aggregate query.
    #[allow(dead_code)]
    fn build_aggregate_return_items(&self, items: &[ast::ReturnItem]) -> Result<Vec<ReturnItem>> {
        let mut return_items = Vec::new();
        let mut agg_idx = 0;

        for item in items {
            if contains_aggregate(&item.expression) {
                // For aggregate expressions, use a variable reference to the aggregate result
                let alias = item.alias.clone().unwrap_or_else(|| {
                    if let ast::Expression::FunctionCall { name, .. } = &item.expression {
                        format!("{}(...)", name.to_lowercase())
                    } else {
                        format!("agg_{}", agg_idx)
                    }
                });
                return_items.push(ReturnItem {
                    expression: LogicalExpression::Variable(format!("__agg_{}", agg_idx)),
                    alias: Some(alias),
                });
                agg_idx += 1;
            } else {
                // Non-aggregate expressions are group-by columns
                return_items.push(ReturnItem {
                    expression: self.translate_expression(&item.expression)?,
                    alias: item.alias.clone(),
                });
            }
        }

        Ok(return_items)
    }

    fn translate_match(&self, match_clause: &ast::MatchClause) -> Result<LogicalOperator> {
        let mut plan: Option<LogicalOperator> = None;

        for pattern in &match_clause.patterns {
            let pattern_plan = self.translate_pattern(pattern, plan.take())?;
            plan = Some(pattern_plan);
        }

        plan.ok_or_else(|| Error::Internal("Empty MATCH clause".to_string()))
    }

    fn translate_pattern(
        &self,
        pattern: &ast::Pattern,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        match pattern {
            ast::Pattern::Node(node) => self.translate_node_pattern(node, input),
            ast::Pattern::Path(path) => self.translate_path_pattern(path, input),
        }
    }

    fn translate_node_pattern(
        &self,
        node: &ast::NodePattern,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let variable = node
            .variable
            .clone()
            .unwrap_or_else(|| format!("_anon_{}", rand_id()));

        let label = node.labels.first().cloned();

        let mut plan = LogicalOperator::NodeScan(NodeScanOp {
            variable: variable.clone(),
            label,
            input: input.map(Box::new),
        });

        // Add filter for node pattern properties (e.g., {name: 'Alice'})
        if !node.properties.is_empty() {
            let predicate = self.build_property_predicate(&variable, &node.properties)?;
            plan = LogicalOperator::Filter(FilterOp {
                predicate,
                input: Box::new(plan),
            });
        }

        Ok(plan)
    }

    /// Builds a predicate expression for property filters like {name: 'Alice', age: 30}.
    fn build_property_predicate(
        &self,
        variable: &str,
        properties: &[(String, ast::Expression)],
    ) -> Result<LogicalExpression> {
        let mut predicates: Vec<LogicalExpression> = Vec::new();

        for (prop_name, prop_value) in properties {
            let left = LogicalExpression::Property {
                variable: variable.to_string(),
                property: prop_name.clone(),
            };
            let right = self.translate_expression(prop_value)?;

            predicates.push(LogicalExpression::Binary {
                left: Box::new(left),
                op: BinaryOp::Eq,
                right: Box::new(right),
            });
        }

        // Combine all predicates with AND
        let mut result = predicates.remove(0);
        for pred in predicates {
            result = LogicalExpression::Binary {
                left: Box::new(result),
                op: BinaryOp::And,
                right: Box::new(pred),
            };
        }

        Ok(result)
    }

    fn translate_path_pattern(
        &self,
        path: &ast::PathPattern,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        // Start with the source node
        let source_var = path
            .source
            .variable
            .clone()
            .unwrap_or_else(|| format!("_anon_{}", rand_id()));

        let source_label = path.source.labels.first().cloned();

        let mut plan = LogicalOperator::NodeScan(NodeScanOp {
            variable: source_var.clone(),
            label: source_label,
            input: input.map(Box::new),
        });

        // Add filter for source node properties (e.g., {id: 'a'})
        if !path.source.properties.is_empty() {
            let predicate = self.build_property_predicate(&source_var, &path.source.properties)?;
            plan = LogicalOperator::Filter(FilterOp {
                predicate,
                input: Box::new(plan),
            });
        }

        // Process each edge in the chain
        let mut current_source = source_var;

        for edge in &path.edges {
            let target_var = edge
                .target
                .variable
                .clone()
                .unwrap_or_else(|| format!("_anon_{}", rand_id()));

            let edge_var = edge.variable.clone();
            let edge_type = edge.types.first().cloned();

            let direction = match edge.direction {
                ast::EdgeDirection::Outgoing => ExpandDirection::Outgoing,
                ast::EdgeDirection::Incoming => ExpandDirection::Incoming,
                ast::EdgeDirection::Undirected => ExpandDirection::Both,
            };

            plan = LogicalOperator::Expand(ExpandOp {
                from_variable: current_source,
                to_variable: target_var.clone(),
                edge_variable: edge_var,
                direction,
                edge_type,
                min_hops: edge.min_hops.unwrap_or(1),
                max_hops: edge.max_hops.or(Some(1)),
                input: Box::new(plan),
            });

            // Add filter for target node properties
            if !edge.target.properties.is_empty() {
                let predicate = self.build_property_predicate(&target_var, &edge.target.properties)?;
                plan = LogicalOperator::Filter(FilterOp {
                    predicate,
                    input: Box::new(plan),
                });
            }

            current_source = target_var;
        }

        Ok(plan)
    }

    fn translate_data_modification(
        &self,
        dm: &ast::DataModificationStatement,
    ) -> Result<LogicalPlan> {
        match dm {
            ast::DataModificationStatement::Insert(insert) => self.translate_insert(insert),
            ast::DataModificationStatement::Delete(delete) => self.translate_delete(delete),
            ast::DataModificationStatement::Set(set) => self.translate_set(set),
        }
    }

    fn translate_delete(&self, delete: &ast::DeleteStatement) -> Result<LogicalPlan> {
        // DELETE requires a preceding MATCH clause to identify what to delete.
        // For standalone DELETE, we need to scan and delete the specified variables.
        // This is typically used as: MATCH (n:Label) DELETE n

        if delete.variables.is_empty() {
            return Err(Error::Internal(
                "DELETE requires at least one variable".to_string(),
            ));
        }

        // For now, we only support deleting nodes (not edges directly)
        // Build a chain of delete operators
        let first_var = &delete.variables[0];

        // Create a scan to find the entities to delete
        let scan = LogicalOperator::NodeScan(NodeScanOp {
            variable: first_var.clone(),
            label: None,
            input: None,
        });

        // Delete the first variable
        let mut plan = LogicalOperator::DeleteNode(DeleteNodeOp {
            variable: first_var.clone(),
            input: Box::new(scan),
        });

        // Chain additional deletes
        for var in delete.variables.iter().skip(1) {
            plan = LogicalOperator::DeleteNode(DeleteNodeOp {
                variable: var.clone(),
                input: Box::new(plan),
            });
        }

        Ok(LogicalPlan::new(plan))
    }

    fn translate_set(&self, set: &ast::SetStatement) -> Result<LogicalPlan> {
        // SET requires a preceding MATCH clause to identify what to update.
        // For standalone SET, we error - it should be part of a query.

        if set.assignments.is_empty() {
            return Err(Error::Internal(
                "SET requires at least one assignment".to_string(),
            ));
        }

        // Group assignments by variable
        let first_assignment = &set.assignments[0];
        let var = &first_assignment.variable;

        // Create a scan to find the entity to update
        let scan = LogicalOperator::NodeScan(NodeScanOp {
            variable: var.clone(),
            label: None,
            input: None,
        });

        // Build property assignments for this variable
        let properties: Vec<(String, LogicalExpression)> = set
            .assignments
            .iter()
            .filter(|a| &a.variable == var)
            .map(|a| Ok((a.property.clone(), self.translate_expression(&a.value)?)))
            .collect::<Result<_>>()?;

        let plan = LogicalOperator::SetProperty(SetPropertyOp {
            variable: var.clone(),
            properties,
            replace: false,
            input: Box::new(scan),
        });

        Ok(LogicalPlan::new(plan))
    }

    fn translate_insert(&self, insert: &ast::InsertStatement) -> Result<LogicalPlan> {
        // For now, just translate insert patterns as creates
        // A full implementation would handle multiple patterns

        if insert.patterns.is_empty() {
            return Err(Error::Internal("Empty INSERT statement".to_string()));
        }

        let pattern = &insert.patterns[0];

        match pattern {
            ast::Pattern::Node(node) => {
                let variable = node
                    .variable
                    .clone()
                    .unwrap_or_else(|| format!("_anon_{}", rand_id()));

                let properties = node
                    .properties
                    .iter()
                    .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
                    .collect::<Result<Vec<_>>>()?;

                let create = LogicalOperator::CreateNode(crate::query::plan::CreateNodeOp {
                    variable: variable.clone(),
                    labels: node.labels.clone(),
                    properties,
                    input: None,
                });

                // Return the created node
                let ret = LogicalOperator::Return(ReturnOp {
                    items: vec![ReturnItem {
                        expression: LogicalExpression::Variable(variable),
                        alias: None,
                    }],
                    distinct: false,
                    input: Box::new(create),
                });

                Ok(LogicalPlan::new(ret))
            }
            ast::Pattern::Path(_) => {
                Err(Error::Internal("Path INSERT not yet supported".to_string()))
            }
        }
    }

    fn translate_expression(&self, expr: &ast::Expression) -> Result<LogicalExpression> {
        match expr {
            ast::Expression::Literal(lit) => Ok(self.translate_literal(lit)),
            ast::Expression::Variable(name) => Ok(LogicalExpression::Variable(name.clone())),
            ast::Expression::Parameter(name) => Ok(LogicalExpression::Parameter(name.clone())),
            ast::Expression::PropertyAccess { variable, property } => {
                Ok(LogicalExpression::Property {
                    variable: variable.clone(),
                    property: property.clone(),
                })
            }
            ast::Expression::Binary { left, op, right } => {
                let left = self.translate_expression(left)?;
                let right = self.translate_expression(right)?;
                let op = self.translate_binary_op(*op);
                Ok(LogicalExpression::Binary {
                    left: Box::new(left),
                    op,
                    right: Box::new(right),
                })
            }
            ast::Expression::Unary { op, operand } => {
                let operand = self.translate_expression(operand)?;
                let op = self.translate_unary_op(*op);
                Ok(LogicalExpression::Unary {
                    op,
                    operand: Box::new(operand),
                })
            }
            ast::Expression::FunctionCall { name, args, distinct } => {
                let args = args
                    .iter()
                    .map(|a| self.translate_expression(a))
                    .collect::<Result<Vec<_>>>()?;
                // TODO: Pass distinct flag to aggregate expressions when needed
                let _ = distinct; // Currently handled separately in aggregate translation
                Ok(LogicalExpression::FunctionCall {
                    name: name.clone(),
                    args,
                })
            }
            ast::Expression::List(items) => {
                let items = items
                    .iter()
                    .map(|i| self.translate_expression(i))
                    .collect::<Result<Vec<_>>>()?;
                Ok(LogicalExpression::List(items))
            }
            ast::Expression::Case {
                input,
                whens,
                else_clause,
            } => {
                let operand = input
                    .as_ref()
                    .map(|e| self.translate_expression(e))
                    .transpose()?
                    .map(Box::new);

                let when_clauses = whens
                    .iter()
                    .map(|(cond, result)| {
                        Ok((
                            self.translate_expression(cond)?,
                            self.translate_expression(result)?,
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?;

                let else_clause = else_clause
                    .as_ref()
                    .map(|e| self.translate_expression(e))
                    .transpose()?
                    .map(Box::new);

                Ok(LogicalExpression::Case {
                    operand,
                    when_clauses,
                    else_clause,
                })
            }
            ast::Expression::ExistsSubquery { query } => {
                // Translate inner query to logical operator
                let inner_plan = self.translate_subquery_to_operator(query)?;
                Ok(LogicalExpression::ExistsSubquery(Box::new(inner_plan)))
            }
        }
    }

    fn translate_literal(&self, lit: &ast::Literal) -> LogicalExpression {
        let value = match lit {
            ast::Literal::Null => Value::Null,
            ast::Literal::Bool(b) => Value::Bool(*b),
            ast::Literal::Integer(i) => Value::Int64(*i),
            ast::Literal::Float(f) => Value::Float64(*f),
            ast::Literal::String(s) => Value::String(s.clone().into()),
        };
        LogicalExpression::Literal(value)
    }

    fn translate_binary_op(&self, op: ast::BinaryOp) -> BinaryOp {
        match op {
            ast::BinaryOp::Eq => BinaryOp::Eq,
            ast::BinaryOp::Ne => BinaryOp::Ne,
            ast::BinaryOp::Lt => BinaryOp::Lt,
            ast::BinaryOp::Le => BinaryOp::Le,
            ast::BinaryOp::Gt => BinaryOp::Gt,
            ast::BinaryOp::Ge => BinaryOp::Ge,
            ast::BinaryOp::And => BinaryOp::And,
            ast::BinaryOp::Or => BinaryOp::Or,
            ast::BinaryOp::Add => BinaryOp::Add,
            ast::BinaryOp::Sub => BinaryOp::Sub,
            ast::BinaryOp::Mul => BinaryOp::Mul,
            ast::BinaryOp::Div => BinaryOp::Div,
            ast::BinaryOp::Mod => BinaryOp::Mod,
            ast::BinaryOp::Concat => BinaryOp::Concat,
            ast::BinaryOp::Like => BinaryOp::Like,
            ast::BinaryOp::In => BinaryOp::In,
        }
    }

    fn translate_unary_op(&self, op: ast::UnaryOp) -> UnaryOp {
        match op {
            ast::UnaryOp::Not => UnaryOp::Not,
            ast::UnaryOp::Neg => UnaryOp::Neg,
            ast::UnaryOp::IsNull => UnaryOp::IsNull,
            ast::UnaryOp::IsNotNull => UnaryOp::IsNotNull,
        }
    }

    /// Translates a subquery to a logical operator (without Return).
    fn translate_subquery_to_operator(
        &self,
        query: &ast::QueryStatement,
    ) -> Result<LogicalOperator> {
        let mut plan = LogicalOperator::Empty;

        for match_clause in &query.match_clauses {
            let match_plan = self.translate_match(match_clause)?;
            plan = if matches!(plan, LogicalOperator::Empty) {
                match_plan
            } else {
                LogicalOperator::Join(JoinOp {
                    left: Box::new(plan),
                    right: Box::new(match_plan),
                    join_type: JoinType::Cross,
                    conditions: vec![],
                })
            };
        }

        if let Some(where_clause) = &query.where_clause {
            let predicate = self.translate_expression(&where_clause.expression)?;
            plan = LogicalOperator::Filter(FilterOp {
                predicate,
                input: Box::new(plan),
            });
        }

        Ok(plan)
    }

    /// Extracts aggregate expressions and group-by expressions from RETURN items.
    fn extract_aggregates_and_groups(
        &self,
        items: &[ast::ReturnItem],
    ) -> Result<(Vec<AggregateExpr>, Vec<LogicalExpression>)> {
        let mut aggregates = Vec::new();
        let mut group_by = Vec::new();

        for item in items {
            if let Some(agg_expr) = self.try_extract_aggregate(&item.expression, &item.alias)? {
                aggregates.push(agg_expr);
            } else {
                // Non-aggregate expressions become group-by keys
                let expr = self.translate_expression(&item.expression)?;
                group_by.push(expr);
            }
        }

        Ok((aggregates, group_by))
    }

    /// Tries to extract an aggregate expression from an AST expression.
    fn try_extract_aggregate(
        &self,
        expr: &ast::Expression,
        alias: &Option<String>,
    ) -> Result<Option<AggregateExpr>> {
        match expr {
            ast::Expression::FunctionCall { name, args, distinct } => {
                if let Some(func) = to_aggregate_function(name) {
                    let agg_expr = if args.is_empty() {
                        // COUNT(*) case
                        AggregateExpr {
                            function: func,
                            expression: None,
                            distinct: *distinct,
                            alias: alias.clone(),
                        }
                    } else {
                        // COUNT(x), SUM(x), etc.
                        AggregateExpr {
                            function: func,
                            expression: Some(self.translate_expression(&args[0])?),
                            distinct: *distinct,
                            alias: alias.clone(),
                        }
                    };
                    Ok(Some(agg_expr))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }
}

/// Generate a simple random-ish ID for anonymous variables.
fn rand_id() -> u32 {
    use std::sync::atomic::{AtomicU32, Ordering};
    static COUNTER: AtomicU32 = AtomicU32::new(0);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Returns true if the function name is an aggregate function.
fn is_aggregate_function(name: &str) -> bool {
    matches!(
        name.to_uppercase().as_str(),
        "COUNT" | "SUM" | "AVG" | "MIN" | "MAX" | "COLLECT"
    )
}

/// Converts a function name to an AggregateFunction enum.
fn to_aggregate_function(name: &str) -> Option<AggregateFunction> {
    match name.to_uppercase().as_str() {
        "COUNT" => Some(AggregateFunction::Count),
        "SUM" => Some(AggregateFunction::Sum),
        "AVG" => Some(AggregateFunction::Avg),
        "MIN" => Some(AggregateFunction::Min),
        "MAX" => Some(AggregateFunction::Max),
        "COLLECT" => Some(AggregateFunction::Collect),
        _ => None,
    }
}

/// Checks if an AST expression contains an aggregate function call.
fn contains_aggregate(expr: &ast::Expression) -> bool {
    match expr {
        ast::Expression::FunctionCall { name, .. } => is_aggregate_function(name),
        ast::Expression::Binary { left, right, .. } => {
            contains_aggregate(left) || contains_aggregate(right)
        }
        ast::Expression::Unary { operand, .. } => contains_aggregate(operand),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // === Basic MATCH Tests ===

    #[test]
    fn test_translate_simple_match() {
        let query = "MATCH (n:Person) RETURN n";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::Return(ret) = &plan.root {
            assert_eq!(ret.items.len(), 1);
            assert!(!ret.distinct);
        } else {
            panic!("Expected Return operator");
        }
    }

    #[test]
    fn test_translate_match_with_where() {
        let query = "MATCH (n:Person) WHERE n.age > 30 RETURN n.name";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::Return(ret) = &plan.root {
            // Should have Filter as input
            if let LogicalOperator::Filter(filter) = ret.input.as_ref() {
                if let LogicalExpression::Binary { op, .. } = &filter.predicate {
                    assert_eq!(*op, BinaryOp::Gt);
                } else {
                    panic!("Expected binary expression");
                }
            } else {
                panic!("Expected Filter operator");
            }
        } else {
            panic!("Expected Return operator");
        }
    }

    #[test]
    fn test_translate_match_without_label() {
        let query = "MATCH (n) RETURN n";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::Return(ret) = &plan.root {
            if let LogicalOperator::NodeScan(scan) = ret.input.as_ref() {
                assert!(scan.label.is_none());
            } else {
                panic!("Expected NodeScan operator");
            }
        } else {
            panic!("Expected Return operator");
        }
    }

    #[test]
    fn test_translate_match_distinct() {
        let query = "MATCH (n:Person) RETURN DISTINCT n.name";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::Return(ret) = &plan.root {
            assert!(ret.distinct);
        } else {
            panic!("Expected Return operator");
        }
    }

    // === Filter and Predicate Tests ===

    #[test]
    fn test_translate_filter_equality() {
        let query = "MATCH (n:Person) WHERE n.name = 'Alice' RETURN n";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        // Navigate to find Filter
        fn find_filter(op: &LogicalOperator) -> Option<&FilterOp> {
            match op {
                LogicalOperator::Filter(f) => Some(f),
                LogicalOperator::Return(r) => find_filter(&r.input),
                _ => None,
            }
        }

        let filter = find_filter(&plan.root).expect("Expected Filter");
        if let LogicalExpression::Binary { op, .. } = &filter.predicate {
            assert_eq!(*op, BinaryOp::Eq);
        }
    }

    #[test]
    fn test_translate_filter_and() {
        let query = "MATCH (n:Person) WHERE n.age > 20 AND n.age < 40 RETURN n";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn find_filter(op: &LogicalOperator) -> Option<&FilterOp> {
            match op {
                LogicalOperator::Filter(f) => Some(f),
                LogicalOperator::Return(r) => find_filter(&r.input),
                _ => None,
            }
        }

        let filter = find_filter(&plan.root).expect("Expected Filter");
        if let LogicalExpression::Binary { op, .. } = &filter.predicate {
            assert_eq!(*op, BinaryOp::And);
        }
    }

    #[test]
    fn test_translate_filter_or() {
        let query = "MATCH (n:Person) WHERE n.name = 'Alice' OR n.name = 'Bob' RETURN n";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn find_filter(op: &LogicalOperator) -> Option<&FilterOp> {
            match op {
                LogicalOperator::Filter(f) => Some(f),
                LogicalOperator::Return(r) => find_filter(&r.input),
                _ => None,
            }
        }

        let filter = find_filter(&plan.root).expect("Expected Filter");
        if let LogicalExpression::Binary { op, .. } = &filter.predicate {
            assert_eq!(*op, BinaryOp::Or);
        }
    }

    #[test]
    fn test_translate_filter_not() {
        let query = "MATCH (n:Person) WHERE NOT n.active RETURN n";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn find_filter(op: &LogicalOperator) -> Option<&FilterOp> {
            match op {
                LogicalOperator::Filter(f) => Some(f),
                LogicalOperator::Return(r) => find_filter(&r.input),
                _ => None,
            }
        }

        let filter = find_filter(&plan.root).expect("Expected Filter");
        if let LogicalExpression::Unary { op, .. } = &filter.predicate {
            assert_eq!(*op, UnaryOp::Not);
        }
    }

    // === Path Pattern / Join Tests ===

    #[test]
    fn test_translate_path_pattern() {
        let query = "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        // Find Expand operator
        fn find_expand(op: &LogicalOperator) -> Option<&ExpandOp> {
            match op {
                LogicalOperator::Expand(e) => Some(e),
                LogicalOperator::Return(r) => find_expand(&r.input),
                LogicalOperator::Filter(f) => find_expand(&f.input),
                _ => None,
            }
        }

        let expand = find_expand(&plan.root).expect("Expected Expand");
        assert_eq!(expand.direction, ExpandDirection::Outgoing);
        assert_eq!(expand.edge_type.as_deref(), Some("KNOWS"));
    }

    #[test]
    fn test_translate_incoming_path() {
        let query = "MATCH (a:Person)<-[:KNOWS]-(b:Person) RETURN a, b";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn find_expand(op: &LogicalOperator) -> Option<&ExpandOp> {
            match op {
                LogicalOperator::Expand(e) => Some(e),
                LogicalOperator::Return(r) => find_expand(&r.input),
                _ => None,
            }
        }

        let expand = find_expand(&plan.root).expect("Expected Expand");
        assert_eq!(expand.direction, ExpandDirection::Incoming);
    }

    #[test]
    fn test_translate_undirected_path() {
        let query = "MATCH (a:Person)-[:KNOWS]-(b:Person) RETURN a, b";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn find_expand(op: &LogicalOperator) -> Option<&ExpandOp> {
            match op {
                LogicalOperator::Expand(e) => Some(e),
                LogicalOperator::Return(r) => find_expand(&r.input),
                _ => None,
            }
        }

        let expand = find_expand(&plan.root).expect("Expected Expand");
        assert_eq!(expand.direction, ExpandDirection::Both);
    }

    // === Aggregation Tests ===

    #[test]
    fn test_translate_count_aggregate() {
        let query = "MATCH (n:Person) RETURN COUNT(n)";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::Aggregate(agg) = &plan.root {
            assert_eq!(agg.aggregates.len(), 1);
            assert_eq!(agg.aggregates[0].function, AggregateFunction::Count);
        } else {
            panic!("Expected Aggregate operator, got {:?}", plan.root);
        }
    }

    #[test]
    fn test_translate_sum_aggregate() {
        let query = "MATCH (n:Person) RETURN SUM(n.age)";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::Aggregate(agg) = &plan.root {
            assert_eq!(agg.aggregates.len(), 1);
            assert_eq!(agg.aggregates[0].function, AggregateFunction::Sum);
        } else {
            panic!("Expected Aggregate operator");
        }
    }

    #[test]
    fn test_translate_group_by_aggregate() {
        let query = "MATCH (n:Person) RETURN n.city, COUNT(n)";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::Aggregate(agg) = &plan.root {
            assert_eq!(agg.group_by.len(), 1); // n.city
            assert_eq!(agg.aggregates.len(), 1); // COUNT(n)
        } else {
            panic!("Expected Aggregate operator");
        }
    }

    // === Ordering and Pagination Tests ===

    #[test]
    fn test_translate_order_by() {
        let query = "MATCH (n:Person) RETURN n ORDER BY n.name";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::Return(ret) = &plan.root {
            if let LogicalOperator::Sort(sort) = ret.input.as_ref() {
                assert_eq!(sort.keys.len(), 1);
                assert_eq!(sort.keys[0].order, SortOrder::Ascending);
            } else {
                panic!("Expected Sort operator");
            }
        } else {
            panic!("Expected Return operator");
        }
    }

    #[test]
    fn test_translate_limit() {
        let query = "MATCH (n:Person) RETURN n LIMIT 10";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        // Find Limit
        fn find_limit(op: &LogicalOperator) -> Option<&LimitOp> {
            match op {
                LogicalOperator::Limit(l) => Some(l),
                LogicalOperator::Return(r) => find_limit(&r.input),
                LogicalOperator::Sort(s) => find_limit(&s.input),
                _ => None,
            }
        }

        let limit = find_limit(&plan.root).expect("Expected Limit");
        assert_eq!(limit.count, 10);
    }

    #[test]
    fn test_translate_skip() {
        let query = "MATCH (n:Person) RETURN n SKIP 5";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn find_skip(op: &LogicalOperator) -> Option<&SkipOp> {
            match op {
                LogicalOperator::Skip(s) => Some(s),
                LogicalOperator::Return(r) => find_skip(&r.input),
                LogicalOperator::Limit(l) => find_skip(&l.input),
                _ => None,
            }
        }

        let skip = find_skip(&plan.root).expect("Expected Skip");
        assert_eq!(skip.count, 5);
    }

    // === Mutation Tests ===

    #[test]
    fn test_translate_insert_node() {
        let query = "INSERT (n:Person {name: 'Alice', age: 30})";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        // Find CreateNode
        fn find_create(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::CreateNode(_) => true,
                LogicalOperator::Return(r) => find_create(&r.input),
                _ => false,
            }
        }

        assert!(find_create(&plan.root));
    }

    #[test]
    fn test_translate_delete() {
        let query = "DELETE n";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::DeleteNode(del) = &plan.root {
            assert_eq!(del.variable, "n");
        } else {
            panic!("Expected DeleteNode operator");
        }
    }

    #[test]
    fn test_translate_set() {
        // SET is not a standalone statement in GQL, test the translator method directly
        let translator = GqlTranslator::new();
        let set_stmt = ast::SetStatement {
            assignments: vec![ast::PropertyAssignment {
                variable: "n".to_string(),
                property: "name".to_string(),
                value: ast::Expression::Literal(ast::Literal::String("Bob".to_string())),
            }],
            span: None,
        };

        let result = translator.translate_set(&set_stmt);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::SetProperty(set) = &plan.root {
            assert_eq!(set.variable, "n");
            assert_eq!(set.properties.len(), 1);
            assert_eq!(set.properties[0].0, "name");
        } else {
            panic!("Expected SetProperty operator");
        }
    }

    // === Expression Translation Tests ===

    #[test]
    fn test_translate_literals() {
        let query = "MATCH (n) WHERE n.count = 42 AND n.active = true AND n.rate = 3.14 RETURN n";
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_parameter() {
        let query = "MATCH (n:Person) WHERE n.name = $name RETURN n";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn find_filter(op: &LogicalOperator) -> Option<&FilterOp> {
            match op {
                LogicalOperator::Filter(f) => Some(f),
                LogicalOperator::Return(r) => find_filter(&r.input),
                _ => None,
            }
        }

        let filter = find_filter(&plan.root).expect("Expected Filter");
        if let LogicalExpression::Binary { right, .. } = &filter.predicate {
            if let LogicalExpression::Parameter(name) = right.as_ref() {
                assert_eq!(name, "name");
            } else {
                panic!("Expected Parameter");
            }
        }
    }

    // === Error Handling Tests ===

    #[test]
    fn test_translate_empty_delete_error() {
        // Create translator directly to test empty delete
        let translator = GqlTranslator::new();
        let delete = ast::DeleteStatement {
            variables: vec![],
            detach: false,
            span: None,
        };
        let result = translator.translate_delete(&delete);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_empty_set_error() {
        let translator = GqlTranslator::new();
        let set = ast::SetStatement {
            assignments: vec![],
            span: None,
        };
        let result = translator.translate_set(&set);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_empty_insert_error() {
        let translator = GqlTranslator::new();
        let insert = ast::InsertStatement {
            patterns: vec![],
            span: None,
        };
        let result = translator.translate_insert(&insert);
        assert!(result.is_err());
    }

    // === Helper Function Tests ===

    #[test]
    fn test_is_aggregate_function() {
        assert!(is_aggregate_function("COUNT"));
        assert!(is_aggregate_function("count"));
        assert!(is_aggregate_function("SUM"));
        assert!(is_aggregate_function("AVG"));
        assert!(is_aggregate_function("MIN"));
        assert!(is_aggregate_function("MAX"));
        assert!(is_aggregate_function("COLLECT"));
        assert!(!is_aggregate_function("UPPER"));
        assert!(!is_aggregate_function("RANDOM"));
    }

    #[test]
    fn test_to_aggregate_function() {
        assert_eq!(
            to_aggregate_function("COUNT"),
            Some(AggregateFunction::Count)
        );
        assert_eq!(to_aggregate_function("sum"), Some(AggregateFunction::Sum));
        assert_eq!(to_aggregate_function("Avg"), Some(AggregateFunction::Avg));
        assert_eq!(to_aggregate_function("min"), Some(AggregateFunction::Min));
        assert_eq!(to_aggregate_function("MAX"), Some(AggregateFunction::Max));
        assert_eq!(
            to_aggregate_function("collect"),
            Some(AggregateFunction::Collect)
        );
        assert_eq!(to_aggregate_function("UNKNOWN"), None);
    }

    #[test]
    fn test_contains_aggregate() {
        let count_expr = ast::Expression::FunctionCall {
            name: "COUNT".to_string(),
            args: vec![],
            distinct: false,
        };
        assert!(contains_aggregate(&count_expr));

        let upper_expr = ast::Expression::FunctionCall {
            name: "UPPER".to_string(),
            args: vec![],
            distinct: false,
        };
        assert!(!contains_aggregate(&upper_expr));

        let var_expr = ast::Expression::Variable("n".to_string());
        assert!(!contains_aggregate(&var_expr));
    }

    #[test]
    fn test_binary_op_translation() {
        let translator = GqlTranslator::new();

        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Eq),
            BinaryOp::Eq
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Ne),
            BinaryOp::Ne
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Lt),
            BinaryOp::Lt
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Le),
            BinaryOp::Le
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Gt),
            BinaryOp::Gt
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Ge),
            BinaryOp::Ge
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::And),
            BinaryOp::And
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Or),
            BinaryOp::Or
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Add),
            BinaryOp::Add
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Sub),
            BinaryOp::Sub
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Mul),
            BinaryOp::Mul
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Div),
            BinaryOp::Div
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Mod),
            BinaryOp::Mod
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Like),
            BinaryOp::Like
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::In),
            BinaryOp::In
        );
    }

    #[test]
    fn test_unary_op_translation() {
        let translator = GqlTranslator::new();

        assert_eq!(
            translator.translate_unary_op(ast::UnaryOp::Not),
            UnaryOp::Not
        );
        assert_eq!(
            translator.translate_unary_op(ast::UnaryOp::Neg),
            UnaryOp::Neg
        );
        assert_eq!(
            translator.translate_unary_op(ast::UnaryOp::IsNull),
            UnaryOp::IsNull
        );
        assert_eq!(
            translator.translate_unary_op(ast::UnaryOp::IsNotNull),
            UnaryOp::IsNotNull
        );
    }
}
