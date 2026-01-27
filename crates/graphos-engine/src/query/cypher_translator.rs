//! Cypher AST to Logical Plan translator.
//!
//! Translates parsed Cypher queries into the common logical plan representation
//! that can be optimized and executed.

use crate::query::plan::*;
use graphos_adapters::query::cypher::{self, ast};
use graphos_common::types::Value;
use graphos_common::utils::error::{Error, Result};

/// Translates a Cypher query string to a logical plan.
pub fn translate(query: &str) -> Result<LogicalPlan> {
    let statement = cypher::parse(query)?;
    let translator = CypherTranslator::new();
    translator.translate_statement(&statement)
}

/// Cypher AST to logical plan translator.
struct CypherTranslator {
    /// Variable counter for generating unique variable names.
    #[allow(dead_code)]
    var_counter: u32,
}

impl CypherTranslator {
    fn new() -> Self {
        Self { var_counter: 0 }
    }

    fn translate_statement(&self, stmt: &ast::Statement) -> Result<LogicalPlan> {
        match stmt {
            ast::Statement::Query(query) => self.translate_query(query),
            ast::Statement::Create(create) => self.translate_create_statement(create),
            ast::Statement::Merge(_) => Err(Error::Internal("MERGE not yet supported".into())),
            ast::Statement::Delete(_) => Err(Error::Internal("DELETE not yet supported".into())),
            ast::Statement::Set(_) => Err(Error::Internal("SET not yet supported".into())),
            ast::Statement::Remove(_) => Err(Error::Internal("REMOVE not yet supported".into())),
        }
    }

    fn translate_query(&self, query: &ast::Query) -> Result<LogicalPlan> {
        let mut plan: Option<LogicalOperator> = None;

        for clause in &query.clauses {
            plan = Some(self.translate_clause(clause, plan)?);
        }

        let root = plan.ok_or_else(|| Error::Internal("Empty query".into()))?;
        Ok(LogicalPlan::new(root))
    }

    fn translate_clause(
        &self,
        clause: &ast::Clause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        match clause {
            ast::Clause::Match(match_clause) => self.translate_match(match_clause, input),
            ast::Clause::OptionalMatch(match_clause) => self.translate_match(match_clause, input),
            ast::Clause::Where(where_clause) => self.translate_where(where_clause, input),
            ast::Clause::With(with_clause) => self.translate_with(with_clause, input),
            ast::Clause::Return(return_clause) => self.translate_return(return_clause, input),
            ast::Clause::Unwind(_) => Err(Error::Internal("UNWIND not yet supported".into())),
            ast::Clause::OrderBy(order_by) => self.translate_order_by(order_by, input),
            ast::Clause::Skip(expr) => self.translate_skip(expr, input),
            ast::Clause::Limit(expr) => self.translate_limit(expr, input),
            ast::Clause::Create(create_clause) => self.translate_create_clause(create_clause, input),
            ast::Clause::Merge(_) => Err(Error::Internal("MERGE not yet supported".into())),
            ast::Clause::Delete(_) => Err(Error::Internal("DELETE not yet supported".into())),
            ast::Clause::Set(_) => Err(Error::Internal("SET not yet supported".into())),
            ast::Clause::Remove(_) => Err(Error::Internal("REMOVE not yet supported".into())),
        }
    }

    fn translate_match(
        &self,
        match_clause: &ast::MatchClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let mut plan = input;

        for pattern in &match_clause.patterns {
            plan = Some(self.translate_pattern(pattern, plan)?);
        }

        plan.ok_or_else(|| Error::Internal("Empty MATCH pattern".into()))
    }

    fn translate_pattern(
        &self,
        pattern: &ast::Pattern,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        match pattern {
            ast::Pattern::Node(node_pattern) => self.translate_node_pattern(node_pattern, input),
            ast::Pattern::Path(path_pattern) => self.translate_path_pattern(path_pattern, input),
            ast::Pattern::NamedPath { pattern, .. } => self.translate_pattern(pattern, input),
        }
    }

    fn translate_node_pattern(
        &self,
        node: &ast::NodePattern,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let variable = node.variable.clone().unwrap_or_else(|| "_anon".to_string());
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
        let mut plan = self.translate_node_pattern(&path.start, input)?;

        for rel in &path.chain {
            plan = self.translate_relationship_pattern(rel, plan)?;
        }

        Ok(plan)
    }

    fn translate_relationship_pattern(
        &self,
        rel: &ast::RelationshipPattern,
        input: LogicalOperator,
    ) -> Result<LogicalOperator> {
        let from_variable = self.get_last_variable(&input)?;
        let edge_variable = rel.variable.clone();
        let edge_type = rel.types.first().cloned();
        let to_variable = rel.target.variable.clone().unwrap_or_else(|| "_anon".to_string());
        let target_label = rel.target.labels.first().cloned();

        let direction = match rel.direction {
            ast::Direction::Outgoing => ExpandDirection::Outgoing,
            ast::Direction::Incoming => ExpandDirection::Incoming,
            ast::Direction::Undirected => ExpandDirection::Both,
        };

        let (min_hops, max_hops) = if let Some(range) = &rel.length {
            (range.min.unwrap_or(1), range.max)
        } else {
            (1, Some(1))
        };

        let expand = LogicalOperator::Expand(ExpandOp {
            from_variable,
            to_variable: to_variable.clone(),
            edge_variable,
            direction,
            edge_type,
            min_hops,
            max_hops,
            input: Box::new(input),
        });

        if let Some(label) = target_label {
            Ok(LogicalOperator::Filter(FilterOp {
                predicate: LogicalExpression::FunctionCall {
                    name: "hasLabel".into(),
                    args: vec![
                        LogicalExpression::Variable(to_variable),
                        LogicalExpression::Literal(Value::from(label)),
                    ],
                },
                input: Box::new(expand),
            }))
        } else {
            Ok(expand)
        }
    }

    fn translate_where(
        &self,
        where_clause: &ast::WhereClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let input = input.ok_or_else(|| Error::Internal("WHERE requires input".into()))?;
        let predicate = self.translate_expression(&where_clause.predicate)?;

        Ok(LogicalOperator::Filter(FilterOp {
            predicate,
            input: Box::new(input),
        }))
    }

    fn translate_with(
        &self,
        with_clause: &ast::WithClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let input = input.ok_or_else(|| Error::Internal("WITH requires input".into()))?;

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

        let mut plan = LogicalOperator::Project(ProjectOp {
            projections,
            input: Box::new(input),
        });

        if let Some(where_clause) = &with_clause.where_clause {
            let predicate = self.translate_expression(&where_clause.predicate)?;
            plan = LogicalOperator::Filter(FilterOp {
                predicate,
                input: Box::new(plan),
            });
        }

        if with_clause.distinct {
            plan = LogicalOperator::Distinct(DistinctOp {
                input: Box::new(plan),
            });
        }

        Ok(plan)
    }

    fn translate_return(
        &self,
        return_clause: &ast::ReturnClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let input = input.ok_or_else(|| Error::Internal("RETURN requires input".into()))?;

        let items = match &return_clause.items {
            ast::ReturnItems::All => {
                vec![ReturnItem {
                    expression: LogicalExpression::Variable("*".into()),
                    alias: None,
                }]
            }
            ast::ReturnItems::Explicit(items) => {
                items
                    .iter()
                    .map(|item| {
                        Ok(ReturnItem {
                            expression: self.translate_expression(&item.expression)?,
                            alias: item.alias.clone(),
                        })
                    })
                    .collect::<Result<_>>()?
            }
        };

        Ok(LogicalOperator::Return(ReturnOp {
            items,
            distinct: return_clause.distinct,
            input: Box::new(input),
        }))
    }

    fn translate_order_by(
        &self,
        order_by: &ast::OrderByClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let input = input.ok_or_else(|| Error::Internal("ORDER BY requires input".into()))?;

        let keys: Vec<SortKey> = order_by
            .items
            .iter()
            .map(|item| {
                Ok(SortKey {
                    expression: self.translate_expression(&item.expression)?,
                    order: match item.direction {
                        ast::SortDirection::Asc => SortOrder::Ascending,
                        ast::SortDirection::Desc => SortOrder::Descending,
                    },
                })
            })
            .collect::<Result<_>>()?;

        Ok(LogicalOperator::Sort(SortOp {
            keys,
            input: Box::new(input),
        }))
    }

    fn translate_skip(
        &self,
        expr: &ast::Expression,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let input = input.ok_or_else(|| Error::Internal("SKIP requires input".into()))?;
        let count = self.eval_as_usize(expr)?;

        Ok(LogicalOperator::Skip(SkipOp {
            count,
            input: Box::new(input),
        }))
    }

    fn translate_limit(
        &self,
        expr: &ast::Expression,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let input = input.ok_or_else(|| Error::Internal("LIMIT requires input".into()))?;
        let count = self.eval_as_usize(expr)?;

        Ok(LogicalOperator::Limit(LimitOp {
            count,
            input: Box::new(input),
        }))
    }

    fn translate_create_clause(
        &self,
        create_clause: &ast::CreateClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let mut plan = input;

        for pattern in &create_clause.patterns {
            plan = Some(self.translate_create_pattern(pattern, plan)?);
        }

        plan.ok_or_else(|| Error::Internal("Empty CREATE pattern".into()))
    }

    fn translate_create_pattern(
        &self,
        pattern: &ast::Pattern,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        match pattern {
            ast::Pattern::Node(node) => {
                let variable = node.variable.clone().unwrap_or_else(|| "_anon".to_string());
                let labels = node.labels.clone();
                let properties: Vec<(String, LogicalExpression)> = node
                    .properties
                    .iter()
                    .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
                    .collect::<Result<_>>()?;

                Ok(LogicalOperator::CreateNode(CreateNodeOp {
                    variable,
                    labels,
                    properties,
                    input: input.map(Box::new),
                }))
            }
            ast::Pattern::Path(path) => {
                let mut current = self.translate_create_pattern(
                    &ast::Pattern::Node(path.start.clone()),
                    input,
                )?;

                for rel in &path.chain {
                    let from_variable = self.get_last_node_variable(&Some(current.clone()))?;
                    let to_variable = rel.target.variable.clone().unwrap_or_else(|| "_anon".to_string());
                    let edge_type = rel.types.first().cloned().unwrap_or_else(|| "RELATED".to_string());

                    let target_labels = rel.target.labels.clone();
                    let target_props: Vec<(String, LogicalExpression)> = rel
                        .target
                        .properties
                        .iter()
                        .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
                        .collect::<Result<_>>()?;

                    current = LogicalOperator::CreateNode(CreateNodeOp {
                        variable: to_variable.clone(),
                        labels: target_labels,
                        properties: target_props,
                        input: Some(Box::new(current)),
                    });

                    let edge_props: Vec<(String, LogicalExpression)> = rel
                        .properties
                        .iter()
                        .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
                        .collect::<Result<_>>()?;

                    current = LogicalOperator::CreateEdge(CreateEdgeOp {
                        variable: rel.variable.clone(),
                        from_variable,
                        to_variable,
                        edge_type,
                        properties: edge_props,
                        input: Box::new(current),
                    });
                }

                Ok(current)
            }
            ast::Pattern::NamedPath { pattern, .. } => {
                self.translate_create_pattern(pattern, input)
            }
        }
    }

    fn translate_create_statement(&self, create: &ast::CreateClause) -> Result<LogicalPlan> {
        let mut plan: Option<LogicalOperator> = None;

        for pattern in &create.patterns {
            plan = Some(self.translate_create_pattern(pattern, plan)?);
        }

        let root = plan.ok_or_else(|| Error::Internal("Empty CREATE".into()))?;
        Ok(LogicalPlan::new(root))
    }

    fn translate_expression(&self, expr: &ast::Expression) -> Result<LogicalExpression> {
        match expr {
            ast::Expression::Literal(lit) => self.translate_literal(lit),
            ast::Expression::Variable(name) => Ok(LogicalExpression::Variable(name.clone())),
            ast::Expression::Parameter(name) => Ok(LogicalExpression::Parameter(name.clone())),
            ast::Expression::PropertyAccess { base, property } => {
                if let ast::Expression::Variable(var) = base.as_ref() {
                    Ok(LogicalExpression::Property {
                        variable: var.clone(),
                        property: property.clone(),
                    })
                } else {
                    Err(Error::Internal("Nested property access not supported".into()))
                }
            }
            ast::Expression::IndexAccess { .. } => {
                Err(Error::Internal("Index access not yet supported".into()))
            }
            ast::Expression::SliceAccess { .. } => {
                Err(Error::Internal("Slice access not yet supported".into()))
            }
            ast::Expression::Binary { left, op, right } => {
                let left_expr = self.translate_expression(left)?;
                let right_expr = self.translate_expression(right)?;
                let binary_op = self.translate_binary_op(op)?;

                Ok(LogicalExpression::Binary {
                    left: Box::new(left_expr),
                    op: binary_op,
                    right: Box::new(right_expr),
                })
            }
            ast::Expression::Unary { op, operand } => {
                let operand_expr = self.translate_expression(operand)?;
                let unary_op = self.translate_unary_op(op)?;

                Ok(LogicalExpression::Unary {
                    op: unary_op,
                    operand: Box::new(operand_expr),
                })
            }
            ast::Expression::FunctionCall { name, args, .. } => {
                let translated_args: Vec<LogicalExpression> = args
                    .iter()
                    .map(|a| self.translate_expression(a))
                    .collect::<Result<_>>()?;

                Ok(LogicalExpression::FunctionCall {
                    name: name.clone(),
                    args: translated_args,
                })
            }
            ast::Expression::List(items) => {
                let translated: Vec<LogicalExpression> = items
                    .iter()
                    .map(|i| self.translate_expression(i))
                    .collect::<Result<_>>()?;

                Ok(LogicalExpression::List(translated))
            }
            ast::Expression::Map(_) => {
                Err(Error::Internal("Map literals not yet supported".into()))
            }
            ast::Expression::Case { input, whens, else_clause } => {
                let translated_operand = if let Some(op) = input {
                    Some(Box::new(self.translate_expression(op)?))
                } else {
                    None
                };

                let translated_when: Vec<(LogicalExpression, LogicalExpression)> = whens
                    .iter()
                    .map(|(when, then)| {
                        Ok((
                            self.translate_expression(when)?,
                            self.translate_expression(then)?,
                        ))
                    })
                    .collect::<Result<_>>()?;

                let translated_else = if let Some(el) = else_clause {
                    Some(Box::new(self.translate_expression(el)?))
                } else {
                    None
                };

                Ok(LogicalExpression::Case {
                    operand: translated_operand,
                    when_clauses: translated_when,
                    else_clause: translated_else,
                })
            }
            ast::Expression::ListComprehension { .. } => {
                Err(Error::Internal("List comprehension not yet supported".into()))
            }
            ast::Expression::PatternComprehension { .. } => {
                Err(Error::Internal("Pattern comprehension not yet supported".into()))
            }
            ast::Expression::Exists(_) => {
                Err(Error::Internal("EXISTS not yet supported".into()))
            }
            ast::Expression::CountSubquery(_) => {
                Err(Error::Internal("COUNT subquery not yet supported".into()))
            }
        }
    }

    fn translate_literal(&self, lit: &ast::Literal) -> Result<LogicalExpression> {
        let value = match lit {
            ast::Literal::Null => Value::Null,
            ast::Literal::Bool(b) => Value::Bool(*b),
            ast::Literal::Integer(i) => Value::Int64(*i),
            ast::Literal::Float(f) => Value::Float64(*f),
            ast::Literal::String(s) => Value::from(s.as_str()),
        };
        Ok(LogicalExpression::Literal(value))
    }

    fn translate_binary_op(&self, op: &ast::BinaryOp) -> Result<BinaryOp> {
        Ok(match op {
            ast::BinaryOp::Eq => BinaryOp::Eq,
            ast::BinaryOp::Ne => BinaryOp::Ne,
            ast::BinaryOp::Lt => BinaryOp::Lt,
            ast::BinaryOp::Le => BinaryOp::Le,
            ast::BinaryOp::Gt => BinaryOp::Gt,
            ast::BinaryOp::Ge => BinaryOp::Ge,
            ast::BinaryOp::And => BinaryOp::And,
            ast::BinaryOp::Or => BinaryOp::Or,
            ast::BinaryOp::Xor => BinaryOp::Xor,
            ast::BinaryOp::Add => BinaryOp::Add,
            ast::BinaryOp::Sub => BinaryOp::Sub,
            ast::BinaryOp::Mul => BinaryOp::Mul,
            ast::BinaryOp::Div => BinaryOp::Div,
            ast::BinaryOp::Mod => BinaryOp::Mod,
            ast::BinaryOp::Pow => {
                return Err(Error::Internal("Power operator not yet supported".into()));
            }
            ast::BinaryOp::Concat => BinaryOp::Concat,
            ast::BinaryOp::StartsWith => BinaryOp::StartsWith,
            ast::BinaryOp::EndsWith => BinaryOp::EndsWith,
            ast::BinaryOp::Contains => BinaryOp::Contains,
            ast::BinaryOp::RegexMatch => {
                return Err(Error::Internal("Regex match not yet supported".into()));
            }
            ast::BinaryOp::In => BinaryOp::In,
        })
    }

    fn translate_unary_op(&self, op: &ast::UnaryOp) -> Result<UnaryOp> {
        Ok(match op {
            ast::UnaryOp::Not => UnaryOp::Not,
            ast::UnaryOp::Neg => UnaryOp::Neg,
            ast::UnaryOp::Pos => {
                return Err(Error::Internal("Unary positive not yet supported".into()));
            }
            ast::UnaryOp::IsNull => UnaryOp::IsNull,
            ast::UnaryOp::IsNotNull => UnaryOp::IsNotNull,
        })
    }

    fn eval_as_usize(&self, expr: &ast::Expression) -> Result<usize> {
        match expr {
            ast::Expression::Literal(ast::Literal::Integer(i)) => {
                if *i >= 0 {
                    Ok(*i as usize)
                } else {
                    Err(Error::Internal("Expected non-negative integer".into()))
                }
            }
            _ => Err(Error::Internal("Expected integer literal".into())),
        }
    }

    fn get_last_variable(&self, plan: &LogicalOperator) -> Result<String> {
        match plan {
            LogicalOperator::NodeScan(scan) => Ok(scan.variable.clone()),
            LogicalOperator::Expand(expand) => Ok(expand.to_variable.clone()),
            LogicalOperator::Filter(filter) => self.get_last_variable(&filter.input),
            LogicalOperator::Project(project) => self.get_last_variable(&project.input),
            _ => Err(Error::Internal("Cannot get variable from operator".into())),
        }
    }

    fn get_last_node_variable(&self, plan: &Option<LogicalOperator>) -> Result<String> {
        match plan {
            Some(LogicalOperator::CreateNode(node)) => Ok(node.variable.clone()),
            Some(LogicalOperator::NodeScan(scan)) => Ok(scan.variable.clone()),
            Some(LogicalOperator::CreateEdge(edge)) => Ok(edge.to_variable.clone()),
            Some(other) => self.get_last_node_variable(&self.extract_input(other)),
            None => Err(Error::Internal("No previous node variable".into())),
        }
    }

    fn extract_input(&self, plan: &LogicalOperator) -> Option<LogicalOperator> {
        match plan {
            LogicalOperator::CreateNode(n) => n.input.as_ref().map(|b| b.as_ref().clone()),
            LogicalOperator::CreateEdge(e) => Some(e.input.as_ref().clone()),
            LogicalOperator::Filter(f) => Some(f.input.as_ref().clone()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translate_simple_match() {
        let plan = translate("MATCH (n:Person) RETURN n").unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            assert_eq!(ret.items.len(), 1);
            if let LogicalOperator::NodeScan(scan) = ret.input.as_ref() {
                assert_eq!(scan.variable, "n");
                assert_eq!(scan.label, Some("Person".into()));
            } else {
                panic!("Expected NodeScan");
            }
        } else {
            panic!("Expected Return");
        }
    }

    #[test]
    fn test_translate_match_with_where() {
        let plan = translate("MATCH (n:Person) WHERE n.age > 30 RETURN n").unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            if let LogicalOperator::Filter(filter) = ret.input.as_ref() {
                if let LogicalExpression::Binary { op, .. } = &filter.predicate {
                    assert_eq!(*op, BinaryOp::Gt);
                }
            } else {
                panic!("Expected Filter");
            }
        } else {
            panic!("Expected Return");
        }
    }

    #[test]
    fn test_translate_create_node() {
        let plan = translate("CREATE (n:Person {name: 'Alice'})").unwrap();

        if let LogicalOperator::CreateNode(create) = &plan.root {
            assert_eq!(create.variable, "n");
            assert_eq!(create.labels, vec!["Person".to_string()]);
            assert_eq!(create.properties.len(), 1);
            assert_eq!(create.properties[0].0, "name");
        } else {
            panic!("Expected CreateNode, got {:?}", plan.root);
        }
    }
}
