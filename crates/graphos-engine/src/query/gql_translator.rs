//! GQL to LogicalPlan translator.
//!
//! Translates GQL AST to the common logical plan representation.

use crate::query::plan::{
    AggregateExpr, AggregateFunction, AggregateOp, BinaryOp, ExpandDirection, ExpandOp, FilterOp,
    LimitOp, LogicalExpression, LogicalOperator, LogicalPlan, NodeScanOp, ReturnItem, ReturnOp,
    SkipOp, SortKey, SortOp, SortOrder, UnaryOp,
};
use graphos_adapters::query::gql::{self, ast};
use graphos_common::types::Value;
use graphos_common::utils::error::{Error, Result};

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
            // FIXME(optional-match): Handle OPTIONAL MATCH with LEFT JOIN semantics
            // For now, treat all MATCH clauses the same way
            let match_plan = self.translate_match(match_clause)?;
            if matches!(plan, LogicalOperator::Empty) {
                plan = match_plan;
            } else {
                // Combine multiple MATCH clauses
                plan = match_plan;
            }
        }

        // Apply WHERE filter
        if let Some(where_clause) = &query.where_clause {
            let predicate = self.translate_expression(&where_clause.expression)?;
            plan = LogicalOperator::Filter(FilterOp {
                predicate,
                input: Box::new(plan),
            });
        }

        // Handle WITH clauses (projection for query chaining)
        // FIXME(with-clause): Handle each WITH as a subquery
        for _with_clause in &query.with_clauses {
            // WITH clause handling - for now we just continue
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
    fn build_aggregate_return_items(
        &self,
        items: &[ast::ReturnItem],
    ) -> Result<Vec<ReturnItem>> {
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

        Ok(LogicalOperator::NodeScan(NodeScanOp {
            variable,
            label,
            input: input.map(Box::new),
        }))
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
                min_hops: 1,
                max_hops: Some(1),
                input: Box::new(plan),
            });

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
            ast::DataModificationStatement::Delete(_) => Err(Error::Internal(
                "DELETE not yet supported".to_string(),
            )),
            ast::DataModificationStatement::Set(_) => Err(Error::Internal(
                "SET not yet supported".to_string(),
            )),
        }
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
            ast::Pattern::Path(_) => Err(Error::Internal(
                "Path INSERT not yet supported".to_string(),
            )),
        }
    }

    fn translate_expression(&self, expr: &ast::Expression) -> Result<LogicalExpression> {
        match expr {
            ast::Expression::Literal(lit) => Ok(self.translate_literal(lit)),
            ast::Expression::Variable(name) => Ok(LogicalExpression::Variable(name.clone())),
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
            ast::Expression::FunctionCall { name, args } => {
                let args = args
                    .iter()
                    .map(|a| self.translate_expression(a))
                    .collect::<Result<Vec<_>>>()?;
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
            ast::Expression::FunctionCall { name, args } => {
                if let Some(func) = to_aggregate_function(name) {
                    let agg_expr = if args.is_empty() {
                        // COUNT(*) case
                        AggregateExpr {
                            function: func,
                            expression: None,
                            distinct: false,
                            alias: alias.clone(),
                        }
                    } else {
                        // COUNT(x), SUM(x), etc.
                        AggregateExpr {
                            function: func,
                            expression: Some(self.translate_expression(&args[0])?),
                            distinct: false,
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
}
