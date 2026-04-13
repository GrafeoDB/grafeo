//! Expression conversion from logical to physical representations.

use super::{
    Direction, Error, ExpandDirection, ExpandOp, FilterExpression, LogicalExpression,
    LogicalOperator, Result, Value, convert_binary_op, convert_unary_op,
};

impl super::Planner {
    /// Converts a logical expression to a filter expression.
    pub(super) fn convert_expression(&self, expr: &LogicalExpression) -> Result<FilterExpression> {
        match expr {
            LogicalExpression::Literal(v) => Ok(FilterExpression::Literal(v.clone())),
            LogicalExpression::Variable(name) => Ok(FilterExpression::Variable(name.clone())),
            LogicalExpression::Property { variable, property } => Ok(FilterExpression::Property {
                variable: variable.clone(),
                property: property.clone(),
            }),
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
            LogicalExpression::FunctionCall { name, args, .. } => {
                let filter_args: Vec<FilterExpression> = args
                    .iter()
                    .map(|a| self.convert_expression(a))
                    .collect::<Result<Vec<_>>>()?;
                Ok(FilterExpression::FunctionCall {
                    name: name.clone(),
                    args: filter_args,
                })
            }
            LogicalExpression::Case {
                operand,
                when_clauses,
                else_clause,
            } => {
                let filter_operand = operand
                    .as_ref()
                    .map(|e| self.convert_expression(e))
                    .transpose()?
                    .map(Box::new);
                let filter_when_clauses: Vec<(FilterExpression, FilterExpression)> = when_clauses
                    .iter()
                    .map(|(cond, result)| {
                        Ok((
                            self.convert_expression(cond)?,
                            self.convert_expression(result)?,
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let filter_else = else_clause
                    .as_ref()
                    .map(|e| self.convert_expression(e))
                    .transpose()?
                    .map(Box::new);
                Ok(FilterExpression::Case {
                    operand: filter_operand,
                    when_clauses: filter_when_clauses,
                    else_clause: filter_else,
                })
            }
            LogicalExpression::List(items) => {
                let filter_items: Vec<FilterExpression> = items
                    .iter()
                    .map(|item| self.convert_expression(item))
                    .collect::<Result<Vec<_>>>()?;
                Ok(FilterExpression::List(filter_items))
            }
            LogicalExpression::Map(pairs) => {
                let filter_pairs: Vec<(String, FilterExpression)> = pairs
                    .iter()
                    .map(|(k, v)| Ok((k.clone(), self.convert_expression(v)?)))
                    .collect::<Result<Vec<_>>>()?;
                Ok(FilterExpression::Map(filter_pairs))
            }
            LogicalExpression::IndexAccess { base, index } => {
                let base_expr = self.convert_expression(base)?;
                let index_expr = self.convert_expression(index)?;
                Ok(FilterExpression::IndexAccess {
                    base: Box::new(base_expr),
                    index: Box::new(index_expr),
                })
            }
            LogicalExpression::SliceAccess { base, start, end } => {
                let base_expr = self.convert_expression(base)?;
                let start_expr = start
                    .as_ref()
                    .map(|s| self.convert_expression(s))
                    .transpose()?
                    .map(Box::new);
                let end_expr = end
                    .as_ref()
                    .map(|e| self.convert_expression(e))
                    .transpose()?
                    .map(Box::new);
                Ok(FilterExpression::SliceAccess {
                    base: Box::new(base_expr),
                    start: start_expr,
                    end: end_expr,
                })
            }
            LogicalExpression::Parameter(name) => Err(Error::Internal(
                format!("Unresolved parameter ${name} in filter — substitution should have replaced this before planning"),
            )),
            LogicalExpression::Labels(var) => Ok(FilterExpression::Labels(var.clone())),
            LogicalExpression::Type(var) => Ok(FilterExpression::Type(var.clone())),
            LogicalExpression::Id(var) => Ok(FilterExpression::Id(var.clone())),
            LogicalExpression::ListComprehension {
                variable,
                list_expr,
                filter_expr,
                map_expr,
            } => {
                let list = self.convert_expression(list_expr)?;
                let filter = filter_expr
                    .as_ref()
                    .map(|f| self.convert_expression(f))
                    .transpose()?
                    .map(Box::new);
                let map = self.convert_expression(map_expr)?;
                Ok(FilterExpression::ListComprehension {
                    variable: variable.clone(),
                    list_expr: Box::new(list),
                    filter_expr: filter,
                    map_expr: Box::new(map),
                })
            }
            LogicalExpression::ListPredicate {
                kind,
                variable,
                list_expr,
                predicate,
            } => {
                let filter_kind = match kind {
                    crate::query::plan::ListPredicateKind::All => {
                        grafeo_core::execution::operators::ListPredicateKind::All
                    }
                    crate::query::plan::ListPredicateKind::Any => {
                        grafeo_core::execution::operators::ListPredicateKind::Any
                    }
                    crate::query::plan::ListPredicateKind::None => {
                        grafeo_core::execution::operators::ListPredicateKind::None
                    }
                    crate::query::plan::ListPredicateKind::Single => {
                        grafeo_core::execution::operators::ListPredicateKind::Single
                    }
                };
                let list = self.convert_expression(list_expr)?;
                let pred = self.convert_expression(predicate)?;
                Ok(FilterExpression::ListPredicate {
                    kind: filter_kind,
                    variable: variable.clone(),
                    list_expr: Box::new(list),
                    predicate: Box::new(pred),
                })
            }
            LogicalExpression::ExistsSubquery(subplan) => {
                // Extract the pattern from the subplan
                // For EXISTS { MATCH (n)-[:TYPE]->() }, we extract start_var, direction, edge_type
                let (start_var, direction, edge_types, end_labels) =
                    self.extract_exists_pattern(subplan)?;

                Ok(FilterExpression::ExistsSubquery {
                    start_var,
                    direction,
                    edge_types,
                    end_labels,
                    min_hops: None,
                    max_hops: None,
                })
            }
            LogicalExpression::CountSubquery(subplan) => {
                // Reuse the same pattern extraction as EXISTS (fast path for simple edges)
                let (start_var, direction, edge_types, end_labels) =
                    self.extract_exists_pattern(subplan)?;

                Ok(FilterExpression::CountSubquery {
                    start_var,
                    direction,
                    edge_types,
                    end_labels,
                })
            }
            LogicalExpression::ValueSubquery(_) => {
                // VALUE subqueries should be lifted into Apply at the translator level
                // before reaching the expression converter. If we get here, it was not lifted.
                Err(Error::Internal(
                    "VALUE subquery should have been lifted into Apply by the translator".into(),
                ))
            }
            LogicalExpression::MapProjection { base, entries } => {
                let physical_entries: Vec<(String, FilterExpression)> = entries
                    .iter()
                    .map(|entry| match entry {
                        crate::query::plan::MapProjectionEntry::PropertySelector(name) => Ok((
                            name.clone(),
                            FilterExpression::Property {
                                variable: base.clone(),
                                property: name.clone(),
                            },
                        )),
                        crate::query::plan::MapProjectionEntry::LiteralEntry(key, expr) => {
                            Ok((key.clone(), self.convert_expression(expr)?))
                        }
                        crate::query::plan::MapProjectionEntry::AllProperties => {
                            // AllProperties is handled at runtime as a special marker
                            Ok((
                                "*".to_string(),
                                FilterExpression::FunctionCall {
                                    name: "properties".to_string(),
                                    args: vec![FilterExpression::Variable(base.clone())],
                                },
                            ))
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(FilterExpression::Map(physical_entries))
            }
            LogicalExpression::Reduce {
                accumulator,
                initial,
                variable,
                list,
                expression,
            } => {
                let init = self.convert_expression(initial)?;
                let list_expr = self.convert_expression(list)?;
                let body = self.convert_expression(expression)?;
                Ok(FilterExpression::Reduce {
                    accumulator: accumulator.clone(),
                    initial: Box::new(init),
                    variable: variable.clone(),
                    list: Box::new(list_expr),
                    expression: Box::new(body),
                })
            }
            LogicalExpression::PatternComprehension { .. } => {
                // Pattern comprehensions should be rewritten by the Cypher translator
                // into Apply + Aggregate(Collect) + ParameterScan before reaching the
                // planner. If we get here, the rewrite was skipped.
                Err(Error::Internal(
                    "PatternComprehension reached the planner without being rewritten; \
                     this is a bug in the Cypher translator"
                        .to_string(),
                ))
            }
        }
    }

    /// Extracts the pattern from an EXISTS subplan for the simple single-hop fast path.
    ///
    /// Returns `(start_variable, direction, edge_type, end_labels)` only for bare
    /// single-hop patterns like `(n)-[:TYPE]->()` or `()-[:TYPE]->(n)`. Rejects
    /// multi-hop patterns, inner WHERE filters, label constraints on target nodes,
    /// and non-correlated patterns (both endpoints anonymous), all of which are
    /// handled correctly by the semi-join rewrite in `plan_filter`.
    ///
    /// When the correlated variable appears on the target side of the pattern
    /// (e.g., `()-[:CALLS]->(m)` where `m` is from the outer scope), the direction
    /// is flipped so the runtime can evaluate from the correlated node.
    pub(super) fn extract_exists_pattern(
        &self,
        subplan: &LogicalOperator,
    ) -> Result<(String, Direction, Vec<String>, Option<Vec<String>>)> {
        match subplan {
            LogicalOperator::Expand(expand) => {
                // Only accept single-hop: the Expand's input (source plan) must be
                // a plain NodeScan. Another Expand means multi-hop; a Filter means
                // inner WHERE or label constraint. Both require the semi-join path.
                if !matches!(expand.input.as_ref(), LogicalOperator::NodeScan(_)) {
                    return Err(Error::Internal(
                        "Unsupported EXISTS subquery pattern".to_string(),
                    ));
                }

                let from_is_anon = expand.from_variable.starts_with("_anon_");
                let to_is_anon = expand.to_variable.starts_with("_anon_");

                if from_is_anon && to_is_anon {
                    // Both endpoints anonymous: non-correlated subquery.
                    // Must go through the semi-join path.
                    return Err(Error::Internal(
                        "Non-correlated EXISTS subquery requires semi-join".to_string(),
                    ));
                }

                if from_is_anon {
                    // Correlated variable is on the target side, e.g. ()-[:CALLS]->(m).
                    // Flip direction: "does m have an incoming CALLS edge?"
                    let direction = match expand.direction {
                        ExpandDirection::Outgoing => Direction::Incoming,
                        ExpandDirection::Incoming => Direction::Outgoing,
                        ExpandDirection::Both => Direction::Both,
                    };
                    let end_labels = self.extract_source_labels_from_expand(expand);
                    Ok((
                        expand.to_variable.clone(),
                        direction,
                        expand.edge_types.clone(),
                        end_labels,
                    ))
                } else {
                    // Normal case: correlated variable on the source side, e.g. (m)-[:CALLS]->()
                    //
                    // No end_labels: the Expand's input is the source NodeScan, whose labels
                    // belong to the correlated variable (already filtered by the outer scope).
                    // Target labels would create a Filter wrapping the Expand, which is
                    // rejected above and correctly routed to the semi-join path.
                    let direction = match expand.direction {
                        ExpandDirection::Outgoing => Direction::Outgoing,
                        ExpandDirection::Incoming => Direction::Incoming,
                        ExpandDirection::Both => Direction::Both,
                    };
                    Ok((
                        expand.from_variable.clone(),
                        direction,
                        expand.edge_types.clone(),
                        None,
                    ))
                }
            }
            LogicalOperator::NodeScan(scan) => {
                if let Some(input) = &scan.input {
                    self.extract_exists_pattern(input)
                } else {
                    Err(Error::Internal(
                        "EXISTS subquery must contain an edge pattern".to_string(),
                    ))
                }
            }
            // A Filter wrapping an Expand typically arises from a label constraint
            // on the anonymous endpoint, e.g. EXISTS { (u)<-[:AUTH]-(:Identity) }.
            // Extract the inner Expand pattern and fold the label filter into end_labels.
            // Only pure hasLabel filters are supported; property filters (WHERE m.age > 30)
            // must go through the semi-join path for correct evaluation.
            LogicalOperator::Filter(filter_op) => {
                let end_labels = self.extract_labels_from_filter_predicate(&filter_op.predicate);
                match end_labels {
                    Some(labels) => {
                        let (start_var, direction, edge_types, _) =
                            self.extract_exists_pattern(&filter_op.input)?;
                        Ok((start_var, direction, edge_types, Some(labels)))
                    }
                    None => {
                        // Non-label filter (property checks, etc.): reject so the
                        // planner uses the semi-join rewrite instead.
                        Err(Error::Internal(
                            "Unsupported EXISTS subquery pattern".to_string(),
                        ))
                    }
                }
            }
            _ => Err(Error::Internal(
                "Unsupported EXISTS subquery pattern".to_string(),
            )),
        }
    }

    /// Extracts label names from a hasLabel filter predicate.
    ///
    /// Given `FunctionCall { name: "hasLabel", args: [Variable(x), Literal("Label")] }`,
    /// returns `Some(vec!["Label"])`.
    fn extract_labels_from_filter_predicate(
        &self,
        predicate: &LogicalExpression,
    ) -> Option<Vec<String>> {
        match predicate {
            LogicalExpression::FunctionCall { name, args, .. } if name == "hasLabel" => {
                if let Some(LogicalExpression::Literal(Value::String(label))) = args.get(1) {
                    Some(vec![label.to_string()])
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Extracts source (input) node labels from an Expand for the flipped EXISTS case.
    ///
    /// When the pattern is `(:Label)-[:TYPE]->(m)` and we flip to start from `m`,
    /// the source node's labels become the "end" labels for the reversed traversal.
    fn extract_source_labels_from_expand(&self, expand: &ExpandOp) -> Option<Vec<String>> {
        match expand.input.as_ref() {
            LogicalOperator::NodeScan(scan) => scan.label.clone().map(|l| vec![l]),
            _ => None,
        }
    }
}
