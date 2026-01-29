//! Gremlin to LogicalPlan translator.
//!
//! Translates Gremlin AST to the common logical plan representation.

use crate::query::plan::{
    AggregateExpr, AggregateFunction, AggregateOp, BinaryOp, CreateNodeOp, DeleteNodeOp,
    DistinctOp, ExpandDirection, ExpandOp, FilterOp, LimitOp, LogicalExpression, LogicalOperator,
    LogicalPlan, NodeScanOp, ReturnItem, ReturnOp, SkipOp, SortKey, SortOp, SortOrder, UnaryOp,
};
use grafeo_adapters::query::gremlin::{self, ast};
use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, Result};
use std::sync::atomic::{AtomicU32, Ordering};

/// Translates a Gremlin query string to a logical plan.
///
/// # Errors
///
/// Returns an error if the query cannot be parsed or translated.
pub fn translate(query: &str) -> Result<LogicalPlan> {
    let statement = gremlin::parse(query)?;
    let translator = GremlinTranslator::new();
    translator.translate_statement(&statement)
}

/// Translator from Gremlin AST to LogicalPlan.
struct GremlinTranslator {
    /// Counter for generating anonymous variables.
    var_counter: AtomicU32,
}

impl GremlinTranslator {
    fn new() -> Self {
        Self {
            var_counter: AtomicU32::new(0),
        }
    }

    fn translate_statement(&self, stmt: &ast::Statement) -> Result<LogicalPlan> {
        // Start with the source
        let mut plan = self.translate_source(&stmt.source)?;

        // Track current variable for property access
        let mut current_var = self.get_current_var(&stmt.source);

        // Process each step
        for step in &stmt.steps {
            let (new_plan, new_var) = self.translate_step(step, plan, &current_var)?;
            plan = new_plan;
            if let Some(v) = new_var {
                current_var = v;
            }
        }

        // If the last step doesn't produce a Return, wrap with one
        if !matches!(plan, LogicalOperator::Return(_)) {
            plan = LogicalOperator::Return(ReturnOp {
                items: vec![ReturnItem {
                    expression: LogicalExpression::Variable(current_var),
                    alias: None,
                }],
                distinct: false,
                input: Box::new(plan),
            });
        }

        Ok(LogicalPlan::new(plan))
    }

    fn translate_source(&self, source: &ast::TraversalSource) -> Result<LogicalOperator> {
        match source {
            ast::TraversalSource::V(ids) => {
                let var = self.next_var();
                let mut plan = LogicalOperator::NodeScan(NodeScanOp {
                    variable: var.clone(),
                    label: None,
                    input: None,
                });

                // If specific IDs, add filter
                if let Some(ids) = ids {
                    if !ids.is_empty() {
                        let id_filter = self.build_id_filter(&var, ids);
                        plan = LogicalOperator::Filter(FilterOp {
                            predicate: id_filter,
                            input: Box::new(plan),
                        });
                    }
                }

                Ok(plan)
            }
            ast::TraversalSource::E(ids) => {
                // Edge scan - need to scan nodes and expand
                let var = self.next_var();
                let mut plan = LogicalOperator::NodeScan(NodeScanOp {
                    variable: var.clone(),
                    label: None,
                    input: None,
                });

                let edge_var = self.next_var();
                let target_var = self.next_var();

                plan = LogicalOperator::Expand(ExpandOp {
                    from_variable: var,
                    to_variable: target_var,
                    edge_variable: Some(edge_var.clone()),
                    direction: ExpandDirection::Both,
                    edge_type: None,
                    min_hops: 1,
                    max_hops: Some(1),
                    input: Box::new(plan),
                });

                // Filter by edge IDs if specified
                if let Some(ids) = ids {
                    if !ids.is_empty() {
                        let id_filter = self.build_id_filter(&edge_var, ids);
                        plan = LogicalOperator::Filter(FilterOp {
                            predicate: id_filter,
                            input: Box::new(plan),
                        });
                    }
                }

                Ok(plan)
            }
            ast::TraversalSource::AddV(label) => {
                let var = self.next_var();
                Ok(LogicalOperator::CreateNode(CreateNodeOp {
                    variable: var,
                    labels: label.iter().cloned().collect(),
                    properties: Vec::new(),
                    input: None,
                }))
            }
            ast::TraversalSource::AddE(_label) => {
                // AddE needs from/to steps to complete
                Err(Error::Internal(
                    "addE requires from() and to() steps".to_string(),
                ))
            }
        }
    }

    fn translate_step(
        &self,
        step: &ast::Step,
        input: LogicalOperator,
        current_var: &str,
    ) -> Result<(LogicalOperator, Option<String>)> {
        match step {
            // Navigation steps
            ast::Step::Out(labels) => {
                let target_var = self.next_var();
                let edge_type = labels.first().cloned();
                let plan = LogicalOperator::Expand(ExpandOp {
                    from_variable: current_var.to_string(),
                    to_variable: target_var.clone(),
                    edge_variable: None,
                    direction: ExpandDirection::Outgoing,
                    edge_type,
                    min_hops: 1,
                    max_hops: Some(1),
                    input: Box::new(input),
                });
                Ok((plan, Some(target_var)))
            }
            ast::Step::In(labels) => {
                let target_var = self.next_var();
                let edge_type = labels.first().cloned();
                let plan = LogicalOperator::Expand(ExpandOp {
                    from_variable: current_var.to_string(),
                    to_variable: target_var.clone(),
                    edge_variable: None,
                    direction: ExpandDirection::Incoming,
                    edge_type,
                    min_hops: 1,
                    max_hops: Some(1),
                    input: Box::new(input),
                });
                Ok((plan, Some(target_var)))
            }
            ast::Step::Both(labels) => {
                let target_var = self.next_var();
                let edge_type = labels.first().cloned();
                let plan = LogicalOperator::Expand(ExpandOp {
                    from_variable: current_var.to_string(),
                    to_variable: target_var.clone(),
                    edge_variable: None,
                    direction: ExpandDirection::Both,
                    edge_type,
                    min_hops: 1,
                    max_hops: Some(1),
                    input: Box::new(input),
                });
                Ok((plan, Some(target_var)))
            }
            ast::Step::OutE(labels) => {
                let edge_var = self.next_var();
                let target_var = self.next_var();
                let edge_type = labels.first().cloned();
                let plan = LogicalOperator::Expand(ExpandOp {
                    from_variable: current_var.to_string(),
                    to_variable: target_var,
                    edge_variable: Some(edge_var.clone()),
                    direction: ExpandDirection::Outgoing,
                    edge_type,
                    min_hops: 1,
                    max_hops: Some(1),
                    input: Box::new(input),
                });
                Ok((plan, Some(edge_var)))
            }
            ast::Step::InE(labels) => {
                let edge_var = self.next_var();
                let target_var = self.next_var();
                let edge_type = labels.first().cloned();
                let plan = LogicalOperator::Expand(ExpandOp {
                    from_variable: current_var.to_string(),
                    to_variable: target_var,
                    edge_variable: Some(edge_var.clone()),
                    direction: ExpandDirection::Incoming,
                    edge_type,
                    min_hops: 1,
                    max_hops: Some(1),
                    input: Box::new(input),
                });
                Ok((plan, Some(edge_var)))
            }
            ast::Step::BothE(labels) => {
                let edge_var = self.next_var();
                let target_var = self.next_var();
                let edge_type = labels.first().cloned();
                let plan = LogicalOperator::Expand(ExpandOp {
                    from_variable: current_var.to_string(),
                    to_variable: target_var,
                    edge_variable: Some(edge_var.clone()),
                    direction: ExpandDirection::Both,
                    edge_type,
                    min_hops: 1,
                    max_hops: Some(1),
                    input: Box::new(input),
                });
                Ok((plan, Some(edge_var)))
            }

            // Filter steps
            ast::Step::Has(has_step) => {
                let predicate = self.translate_has_step(has_step, current_var)?;
                let plan = LogicalOperator::Filter(FilterOp {
                    predicate,
                    input: Box::new(input),
                });
                Ok((plan, None))
            }
            ast::Step::HasLabel(labels) => {
                let predicate = if labels.len() == 1 {
                    LogicalExpression::Binary {
                        left: Box::new(LogicalExpression::Labels(current_var.to_string())),
                        op: BinaryOp::Eq,
                        right: Box::new(LogicalExpression::Literal(Value::String(
                            labels[0].clone().into(),
                        ))),
                    }
                } else {
                    LogicalExpression::Binary {
                        left: Box::new(LogicalExpression::Labels(current_var.to_string())),
                        op: BinaryOp::In,
                        right: Box::new(LogicalExpression::List(
                            labels
                                .iter()
                                .map(|l| {
                                    LogicalExpression::Literal(Value::String(l.clone().into()))
                                })
                                .collect(),
                        )),
                    }
                };
                let plan = LogicalOperator::Filter(FilterOp {
                    predicate,
                    input: Box::new(input),
                });
                Ok((plan, None))
            }
            ast::Step::HasId(ids) => {
                let predicate = self.build_id_filter(current_var, ids);
                let plan = LogicalOperator::Filter(FilterOp {
                    predicate,
                    input: Box::new(input),
                });
                Ok((plan, None))
            }
            ast::Step::HasNot(key) => {
                let predicate = LogicalExpression::Unary {
                    op: UnaryOp::IsNull,
                    operand: Box::new(LogicalExpression::Property {
                        variable: current_var.to_string(),
                        property: key.clone(),
                    }),
                };
                let plan = LogicalOperator::Filter(FilterOp {
                    predicate,
                    input: Box::new(input),
                });
                Ok((plan, None))
            }
            ast::Step::Dedup(_keys) => {
                // TODO: Use keys for column-specific dedup when supported
                let plan = LogicalOperator::Distinct(DistinctOp {
                    input: Box::new(input),
                });
                Ok((plan, None))
            }
            ast::Step::Limit(n) => {
                let plan = LogicalOperator::Limit(LimitOp {
                    count: *n,
                    input: Box::new(input),
                });
                Ok((plan, None))
            }
            ast::Step::Skip(n) => {
                let plan = LogicalOperator::Skip(SkipOp {
                    count: *n,
                    input: Box::new(input),
                });
                Ok((plan, None))
            }
            ast::Step::Range(start, end) => {
                let plan = LogicalOperator::Skip(SkipOp {
                    count: *start,
                    input: Box::new(input),
                });
                let plan = LogicalOperator::Limit(LimitOp {
                    count: end - start,
                    input: Box::new(plan),
                });
                Ok((plan, None))
            }

            // Map steps
            ast::Step::Values(keys) => {
                if keys.len() == 1 {
                    let plan = LogicalOperator::Return(ReturnOp {
                        items: vec![ReturnItem {
                            expression: LogicalExpression::Property {
                                variable: current_var.to_string(),
                                property: keys[0].clone(),
                            },
                            alias: Some(keys[0].clone()),
                        }],
                        distinct: false,
                        input: Box::new(input),
                    });
                    Ok((plan, None))
                } else {
                    let items: Vec<ReturnItem> = keys
                        .iter()
                        .map(|k| ReturnItem {
                            expression: LogicalExpression::Property {
                                variable: current_var.to_string(),
                                property: k.clone(),
                            },
                            alias: Some(k.clone()),
                        })
                        .collect();
                    let plan = LogicalOperator::Return(ReturnOp {
                        items,
                        distinct: false,
                        input: Box::new(input),
                    });
                    Ok((plan, None))
                }
            }
            ast::Step::Id => {
                let plan = LogicalOperator::Return(ReturnOp {
                    items: vec![ReturnItem {
                        expression: LogicalExpression::Id(current_var.to_string()),
                        alias: Some("id".to_string()),
                    }],
                    distinct: false,
                    input: Box::new(input),
                });
                Ok((plan, None))
            }
            ast::Step::Label => {
                let plan = LogicalOperator::Return(ReturnOp {
                    items: vec![ReturnItem {
                        expression: LogicalExpression::Labels(current_var.to_string()),
                        alias: Some("label".to_string()),
                    }],
                    distinct: false,
                    input: Box::new(input),
                });
                Ok((plan, None))
            }
            ast::Step::Count => {
                let plan = LogicalOperator::Aggregate(AggregateOp {
                    group_by: Vec::new(),
                    aggregates: vec![AggregateExpr {
                        function: AggregateFunction::Count,
                        expression: None,
                        distinct: false,
                        alias: Some("count".to_string()),
                    }],
                    input: Box::new(input),
                });
                Ok((plan, None))
            }
            ast::Step::Sum => {
                let plan = LogicalOperator::Aggregate(AggregateOp {
                    group_by: Vec::new(),
                    aggregates: vec![AggregateExpr {
                        function: AggregateFunction::Sum,
                        expression: Some(LogicalExpression::Variable(current_var.to_string())),
                        distinct: false,
                        alias: Some("sum".to_string()),
                    }],
                    input: Box::new(input),
                });
                Ok((plan, None))
            }
            ast::Step::Mean => {
                let plan = LogicalOperator::Aggregate(AggregateOp {
                    group_by: Vec::new(),
                    aggregates: vec![AggregateExpr {
                        function: AggregateFunction::Avg,
                        expression: Some(LogicalExpression::Variable(current_var.to_string())),
                        distinct: false,
                        alias: Some("mean".to_string()),
                    }],
                    input: Box::new(input),
                });
                Ok((plan, None))
            }
            ast::Step::Min => {
                let plan = LogicalOperator::Aggregate(AggregateOp {
                    group_by: Vec::new(),
                    aggregates: vec![AggregateExpr {
                        function: AggregateFunction::Min,
                        expression: Some(LogicalExpression::Variable(current_var.to_string())),
                        distinct: false,
                        alias: Some("min".to_string()),
                    }],
                    input: Box::new(input),
                });
                Ok((plan, None))
            }
            ast::Step::Max => {
                let plan = LogicalOperator::Aggregate(AggregateOp {
                    group_by: Vec::new(),
                    aggregates: vec![AggregateExpr {
                        function: AggregateFunction::Max,
                        expression: Some(LogicalExpression::Variable(current_var.to_string())),
                        distinct: false,
                        alias: Some("max".to_string()),
                    }],
                    input: Box::new(input),
                });
                Ok((plan, None))
            }
            ast::Step::Fold => {
                let plan = LogicalOperator::Aggregate(AggregateOp {
                    group_by: Vec::new(),
                    aggregates: vec![AggregateExpr {
                        function: AggregateFunction::Collect,
                        expression: Some(LogicalExpression::Variable(current_var.to_string())),
                        distinct: false,
                        alias: Some("fold".to_string()),
                    }],
                    input: Box::new(input),
                });
                Ok((plan, None))
            }
            ast::Step::Order(modifiers) => {
                let keys = if modifiers.is_empty() {
                    vec![SortKey {
                        expression: LogicalExpression::Variable(current_var.to_string()),
                        order: SortOrder::Ascending,
                    }]
                } else {
                    modifiers
                        .iter()
                        .map(|m| SortKey {
                            expression: self.translate_by_modifier(&m.by, current_var),
                            order: match m.order {
                                ast::SortOrder::Asc => SortOrder::Ascending,
                                ast::SortOrder::Desc => SortOrder::Descending,
                                ast::SortOrder::Shuffle => SortOrder::Ascending, // Not supported
                            },
                        })
                        .collect()
                };
                let plan = LogicalOperator::Sort(SortOp {
                    keys,
                    input: Box::new(input),
                });
                Ok((plan, None))
            }

            // Side effect steps
            ast::Step::As(label) => {
                // 'as' just adds a label, which we track via variables
                // In LogicalPlan, we use the label as an alias
                Ok((input, Some(label.clone())))
            }
            ast::Step::Property(_prop_step) => {
                // TODO: Translate property setting to SetPropertyOp
                Ok((input, None))
            }
            ast::Step::Drop => {
                // Delete the current element
                let plan = LogicalOperator::DeleteNode(DeleteNodeOp {
                    variable: current_var.to_string(),
                    input: Box::new(input),
                });
                Ok((plan, None))
            }
            ast::Step::AddV(label) => {
                let var = self.next_var();
                let plan = LogicalOperator::CreateNode(CreateNodeOp {
                    variable: var.clone(),
                    labels: label.iter().cloned().collect(),
                    properties: Vec::new(),
                    input: Some(Box::new(input)),
                });
                Ok((plan, Some(var)))
            }
            ast::Step::AddE(_label) => {
                // AddE needs from/to context
                Err(Error::Internal(
                    "addE requires from() and to() context".to_string(),
                ))
            }

            // Steps not fully supported
            _ => Ok((input, None)),
        }
    }

    fn translate_has_step(&self, has: &ast::HasStep, var: &str) -> Result<LogicalExpression> {
        match has {
            ast::HasStep::Key(key) => {
                // has(key) - check if property exists
                Ok(LogicalExpression::Unary {
                    op: UnaryOp::IsNotNull,
                    operand: Box::new(LogicalExpression::Property {
                        variable: var.to_string(),
                        property: key.clone(),
                    }),
                })
            }
            ast::HasStep::KeyValue(key, value) => {
                // has(key, value) - check property equals value
                Ok(LogicalExpression::Binary {
                    left: Box::new(LogicalExpression::Property {
                        variable: var.to_string(),
                        property: key.clone(),
                    }),
                    op: BinaryOp::Eq,
                    right: Box::new(LogicalExpression::Literal(value.clone())),
                })
            }
            ast::HasStep::KeyPredicate(key, pred) => {
                let prop = LogicalExpression::Property {
                    variable: var.to_string(),
                    property: key.clone(),
                };
                Self::translate_predicate(pred, prop)
            }
            ast::HasStep::LabelKeyValue(label, key, value) => {
                // has(label, key, value) - check label AND property
                let label_check = LogicalExpression::Binary {
                    left: Box::new(LogicalExpression::Labels(var.to_string())),
                    op: BinaryOp::Eq,
                    right: Box::new(LogicalExpression::Literal(Value::String(
                        label.clone().into(),
                    ))),
                };
                let prop_check = LogicalExpression::Binary {
                    left: Box::new(LogicalExpression::Property {
                        variable: var.to_string(),
                        property: key.clone(),
                    }),
                    op: BinaryOp::Eq,
                    right: Box::new(LogicalExpression::Literal(value.clone())),
                };
                Ok(LogicalExpression::Binary {
                    left: Box::new(label_check),
                    op: BinaryOp::And,
                    right: Box::new(prop_check),
                })
            }
        }
    }

    fn translate_predicate(
        pred: &ast::Predicate,
        expr: LogicalExpression,
    ) -> Result<LogicalExpression> {
        match pred {
            ast::Predicate::Eq(value) => Ok(LogicalExpression::Binary {
                left: Box::new(expr),
                op: BinaryOp::Eq,
                right: Box::new(LogicalExpression::Literal(value.clone())),
            }),
            ast::Predicate::Neq(value) => Ok(LogicalExpression::Binary {
                left: Box::new(expr),
                op: BinaryOp::Ne,
                right: Box::new(LogicalExpression::Literal(value.clone())),
            }),
            ast::Predicate::Lt(value) => Ok(LogicalExpression::Binary {
                left: Box::new(expr),
                op: BinaryOp::Lt,
                right: Box::new(LogicalExpression::Literal(value.clone())),
            }),
            ast::Predicate::Lte(value) => Ok(LogicalExpression::Binary {
                left: Box::new(expr),
                op: BinaryOp::Le,
                right: Box::new(LogicalExpression::Literal(value.clone())),
            }),
            ast::Predicate::Gt(value) => Ok(LogicalExpression::Binary {
                left: Box::new(expr),
                op: BinaryOp::Gt,
                right: Box::new(LogicalExpression::Literal(value.clone())),
            }),
            ast::Predicate::Gte(value) => Ok(LogicalExpression::Binary {
                left: Box::new(expr),
                op: BinaryOp::Ge,
                right: Box::new(LogicalExpression::Literal(value.clone())),
            }),
            ast::Predicate::Within(values) => Ok(LogicalExpression::Binary {
                left: Box::new(expr),
                op: BinaryOp::In,
                right: Box::new(LogicalExpression::List(
                    values
                        .iter()
                        .map(|v| LogicalExpression::Literal(v.clone()))
                        .collect(),
                )),
            }),
            ast::Predicate::Without(values) => Ok(LogicalExpression::Unary {
                op: UnaryOp::Not,
                operand: Box::new(LogicalExpression::Binary {
                    left: Box::new(expr),
                    op: BinaryOp::In,
                    right: Box::new(LogicalExpression::List(
                        values
                            .iter()
                            .map(|v| LogicalExpression::Literal(v.clone()))
                            .collect(),
                    )),
                }),
            }),
            ast::Predicate::Between(start, end) => Ok(LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Binary {
                    left: Box::new(expr.clone()),
                    op: BinaryOp::Ge,
                    right: Box::new(LogicalExpression::Literal(start.clone())),
                }),
                op: BinaryOp::And,
                right: Box::new(LogicalExpression::Binary {
                    left: Box::new(expr),
                    op: BinaryOp::Lt,
                    right: Box::new(LogicalExpression::Literal(end.clone())),
                }),
            }),
            ast::Predicate::Containing(s) => Ok(LogicalExpression::Binary {
                left: Box::new(expr),
                op: BinaryOp::Contains,
                right: Box::new(LogicalExpression::Literal(Value::String(s.clone().into()))),
            }),
            ast::Predicate::StartingWith(s) => Ok(LogicalExpression::Binary {
                left: Box::new(expr),
                op: BinaryOp::StartsWith,
                right: Box::new(LogicalExpression::Literal(Value::String(s.clone().into()))),
            }),
            ast::Predicate::EndingWith(s) => Ok(LogicalExpression::Binary {
                left: Box::new(expr),
                op: BinaryOp::EndsWith,
                right: Box::new(LogicalExpression::Literal(Value::String(s.clone().into()))),
            }),
            ast::Predicate::And(preds) => {
                let mut result = Self::translate_predicate(&preds[0], expr.clone())?;
                for pred in &preds[1..] {
                    let right = Self::translate_predicate(pred, expr.clone())?;
                    result = LogicalExpression::Binary {
                        left: Box::new(result),
                        op: BinaryOp::And,
                        right: Box::new(right),
                    };
                }
                Ok(result)
            }
            ast::Predicate::Or(preds) => {
                let mut result = Self::translate_predicate(&preds[0], expr.clone())?;
                for pred in &preds[1..] {
                    let right = Self::translate_predicate(pred, expr.clone())?;
                    result = LogicalExpression::Binary {
                        left: Box::new(result),
                        op: BinaryOp::Or,
                        right: Box::new(right),
                    };
                }
                Ok(result)
            }
            ast::Predicate::Not(pred) => Ok(LogicalExpression::Unary {
                op: UnaryOp::Not,
                operand: Box::new(Self::translate_predicate(pred, expr)?),
            }),
            _ => Err(Error::Internal("Unsupported predicate".to_string())),
        }
    }

    fn translate_by_modifier(&self, by: &ast::ByModifier, current_var: &str) -> LogicalExpression {
        match by {
            ast::ByModifier::Identity => LogicalExpression::Variable(current_var.to_string()),
            ast::ByModifier::Key(key) => LogicalExpression::Property {
                variable: current_var.to_string(),
                property: key.clone(),
            },
            ast::ByModifier::Token(token) => match token {
                ast::TokenType::Id => LogicalExpression::Id(current_var.to_string()),
                ast::TokenType::Label => LogicalExpression::Labels(current_var.to_string()),
                _ => LogicalExpression::Variable(current_var.to_string()),
            },
            _ => LogicalExpression::Variable(current_var.to_string()),
        }
    }

    fn build_id_filter(&self, var: &str, ids: &[Value]) -> LogicalExpression {
        if ids.len() == 1 {
            LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Id(var.to_string())),
                op: BinaryOp::Eq,
                right: Box::new(LogicalExpression::Literal(ids[0].clone())),
            }
        } else {
            LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Id(var.to_string())),
                op: BinaryOp::In,
                right: Box::new(LogicalExpression::List(
                    ids.iter()
                        .map(|id| LogicalExpression::Literal(id.clone()))
                        .collect(),
                )),
            }
        }
    }

    fn get_current_var(&self, _source: &ast::TraversalSource) -> String {
        format!("_v{}", self.var_counter.load(Ordering::Relaxed))
    }

    fn next_var(&self) -> String {
        let n = self.var_counter.fetch_add(1, Ordering::Relaxed);
        format!("_v{}", n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // === Basic Traversal Tests ===

    #[test]
    fn test_translate_simple_traversal() {
        let result = translate("g.V()");
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_with_filter() {
        let result = translate("g.V().hasLabel('Person')");
        assert!(result.is_ok());
    }

    // === Navigation Tests ===

    #[test]
    fn test_translate_navigation() {
        let result = translate("g.V().out('knows')");
        assert!(result.is_ok());
        let plan = result.unwrap();
        // Should have NodeScan -> Expand -> Return
        if let LogicalOperator::Return(ret) = &plan.root {
            if let LogicalOperator::Expand(expand) = ret.input.as_ref() {
                assert_eq!(expand.edge_type, Some("knows".to_string()));
                assert_eq!(expand.direction, ExpandDirection::Outgoing);
            } else {
                panic!("Expected Expand operator");
            }
        } else {
            panic!("Expected Return operator");
        }
    }

    #[test]
    fn test_translate_in_navigation() {
        let result = translate("g.V().in('knows')");
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
    fn test_translate_both_navigation() {
        let result = translate("g.V().both('knows')");
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

    #[test]
    fn test_translate_out_e() {
        let result = translate("g.V().outE('knows')");
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
        assert!(expand.edge_variable.is_some());
    }

    // === Filter Tests ===

    #[test]
    fn test_translate_has_key_value() {
        let result = translate("g.V().has('name', 'Alice')");
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
            assert_eq!(*op, BinaryOp::Eq);
        }
    }

    #[test]
    fn test_translate_has_not() {
        let result = translate("g.V().hasNot('deleted')");
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
            assert_eq!(*op, UnaryOp::IsNull);
        }
    }

    #[test]
    fn test_translate_dedup() {
        let result = translate("g.V().dedup()");
        assert!(result.is_ok());
        let plan = result.unwrap();

        fn find_distinct(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Distinct(_) => true,
                LogicalOperator::Return(r) => find_distinct(&r.input),
                _ => false,
            }
        }

        assert!(find_distinct(&plan.root));
    }

    // === Pagination Tests ===

    #[test]
    fn test_translate_limit() {
        let result = translate("g.V().limit(10)");
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_skip() {
        let result = translate("g.V().skip(5)");
        assert!(result.is_ok());
        let plan = result.unwrap();

        fn find_skip(op: &LogicalOperator) -> Option<&SkipOp> {
            match op {
                LogicalOperator::Skip(s) => Some(s),
                LogicalOperator::Return(r) => find_skip(&r.input),
                _ => None,
            }
        }

        let skip = find_skip(&plan.root).expect("Expected Skip");
        assert_eq!(skip.count, 5);
    }

    #[test]
    fn test_translate_range() {
        let result = translate("g.V().range(5, 15)");
        assert!(result.is_ok());
        let plan = result.unwrap();

        fn find_limit(op: &LogicalOperator) -> Option<&LimitOp> {
            match op {
                LogicalOperator::Limit(l) => Some(l),
                LogicalOperator::Return(r) => find_limit(&r.input),
                _ => None,
            }
        }

        let limit = find_limit(&plan.root).expect("Expected Limit");
        assert_eq!(limit.count, 10); // 15 - 5
    }

    // === Aggregation Tests ===

    #[test]
    fn test_translate_count() {
        let result = translate("g.V().count()");
        assert!(result.is_ok());
        let plan = result.unwrap();
        // The result is wrapped in Return(Aggregate(...))
        if let LogicalOperator::Return(ret) = &plan.root {
            if let LogicalOperator::Aggregate(agg) = ret.input.as_ref() {
                assert_eq!(agg.aggregates.len(), 1);
                assert_eq!(agg.aggregates[0].function, AggregateFunction::Count);
            } else {
                panic!("Expected Aggregate operator inside Return");
            }
        } else {
            panic!("Expected Return operator");
        }
    }

    #[test]
    fn test_translate_sum() {
        let result = translate("g.V().values('age').sum()");
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_mean() {
        let result = translate("g.V().values('age').mean()");
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_min() {
        let result = translate("g.V().values('age').min()");
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_max() {
        let result = translate("g.V().values('age').max()");
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_fold() {
        let result = translate("g.V().fold()");
        assert!(result.is_ok());
        let plan = result.unwrap();

        fn find_aggregate(op: &LogicalOperator) -> Option<&AggregateOp> {
            match op {
                LogicalOperator::Aggregate(a) => Some(a),
                LogicalOperator::Return(r) => find_aggregate(&r.input),
                _ => None,
            }
        }

        let agg = find_aggregate(&plan.root).expect("Expected Aggregate");
        assert_eq!(agg.aggregates[0].function, AggregateFunction::Collect);
    }

    // === Map Steps ===

    #[test]
    fn test_translate_values() {
        let result = translate("g.V().values('name')");
        assert!(result.is_ok());
        let plan = result.unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            assert_eq!(ret.items.len(), 1);
            if let LogicalExpression::Property { property, .. } = &ret.items[0].expression {
                assert_eq!(property, "name");
            }
        }
    }

    #[test]
    fn test_translate_id() {
        let result = translate("g.V().id()");
        assert!(result.is_ok());
        let plan = result.unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            if let LogicalExpression::Id(_) = &ret.items[0].expression {
                // OK
            } else {
                panic!("Expected Id expression");
            }
        }
    }

    #[test]
    fn test_translate_label() {
        let result = translate("g.V().label()");
        assert!(result.is_ok());
        let plan = result.unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            if let LogicalExpression::Labels(_) = &ret.items[0].expression {
                // OK
            } else {
                panic!("Expected Labels expression");
            }
        }
    }

    // === Mutation Tests ===

    #[test]
    fn test_translate_add_v() {
        let result = translate("g.addV('Person')");
        assert!(result.is_ok());
        let plan = result.unwrap();

        fn find_create(op: &LogicalOperator) -> Option<&CreateNodeOp> {
            match op {
                LogicalOperator::CreateNode(c) => Some(c),
                LogicalOperator::Return(r) => find_create(&r.input),
                _ => None,
            }
        }

        let create = find_create(&plan.root).expect("Expected CreateNode");
        assert_eq!(create.labels, vec!["Person".to_string()]);
    }

    #[test]
    fn test_translate_drop() {
        let result = translate("g.V().drop()");
        assert!(result.is_ok());
        let plan = result.unwrap();

        fn find_delete(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::DeleteNode(_) => true,
                LogicalOperator::Return(r) => find_delete(&r.input),
                _ => false,
            }
        }

        assert!(find_delete(&plan.root));
    }

    // === Order Tests ===

    #[test]
    fn test_translate_order() {
        let result = translate("g.V().order()");
        assert!(result.is_ok());
        let plan = result.unwrap();

        fn find_sort(op: &LogicalOperator) -> Option<&SortOp> {
            match op {
                LogicalOperator::Sort(s) => Some(s),
                LogicalOperator::Return(r) => find_sort(&r.input),
                _ => None,
            }
        }

        assert!(find_sort(&plan.root).is_some());
    }

    // === Predicate Tests ===

    #[test]
    fn test_predicate_gt() {
        let expr = LogicalExpression::Variable("x".to_string());
        let pred = ast::Predicate::Gt(Value::Int64(10));
        let result = GremlinTranslator::translate_predicate(&pred, expr).unwrap();

        if let LogicalExpression::Binary { op, .. } = result {
            assert_eq!(op, BinaryOp::Gt);
        } else {
            panic!("Expected Binary expression");
        }
    }

    #[test]
    fn test_predicate_within() {
        let expr = LogicalExpression::Variable("x".to_string());
        let pred = ast::Predicate::Within(vec![Value::Int64(1), Value::Int64(2)]);
        let result = GremlinTranslator::translate_predicate(&pred, expr).unwrap();

        if let LogicalExpression::Binary { op, .. } = result {
            assert_eq!(op, BinaryOp::In);
        } else {
            panic!("Expected Binary expression");
        }
    }

    #[test]
    fn test_predicate_containing() {
        let expr = LogicalExpression::Variable("x".to_string());
        let pred = ast::Predicate::Containing("test".to_string());
        let result = GremlinTranslator::translate_predicate(&pred, expr).unwrap();

        if let LogicalExpression::Binary { op, .. } = result {
            assert_eq!(op, BinaryOp::Contains);
        } else {
            panic!("Expected Binary expression");
        }
    }
}
