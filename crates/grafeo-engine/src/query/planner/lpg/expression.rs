//! Expression conversion from logical to physical representations.
//!
//! `convert_expression` is a recursive walk that mirrors
//! `LogicalExpression` to `FilterExpression`. The walk is *not* a pure
//! translation: a few match arms recognise patterns that deserve a
//! specialised physical shape rather than the straight equivalent.
//!
//! The most load-bearing case is `LogicalExpression::Exists`: the fast
//! path invokes `extract_exists_pattern` to turn a single-hop EXISTS
//! into an inline predicate that can evaluate during the scan, avoiding
//! a semi-join for the common `EXISTS { (n)-[:R]->() }` shape. Multi-hop
//! or otherwise complex EXISTS instead falls through and is rewritten
//! at the filter layer via `extract_complex_exists` (see
//! [`super::filter`]). The two entry points are coupled: `plan_filter`
//! checks for the complex form *before* calling `convert_expression`
//! so the fast path here only ever sees shapes it can actually handle
//! inline.

use super::{
    Direction, Error, ExpandDirection, FilterExpression, LogicalExpression, LogicalOperator,
    PathMode, Result, Value, convert_binary_op, convert_unary_op,
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
            LogicalExpression::MapAccess { base, key } => Ok(FilterExpression::IndexAccess {
                base: Box::new(self.convert_expression(base)?),
                index: Box::new(FilterExpression::Literal(Value::from(key.as_str()))),
            }),
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
            LogicalExpression::Parameter(name) => Err(Error::Internal(format!(
                "Unresolved parameter ${name} in filter: substitution should have replaced this before planning"
            ))),
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
                // For EXISTS { MATCH (n)-[:TYPE]->() }: a check on n's own edges.
                let check = self.extract_exists_pattern(subplan)?;
                Ok(FilterExpression::ExistsSubquery {
                    start_var: check.start_var,
                    end_var: check.end_var,
                    edge_var: check.edge_var,
                    direction: check.direction,
                    edge_types: check.edge_types,
                    end_labels: check.end_labels,
                    min_hops: (!check.one_edge).then_some(1),
                    max_hops: check.max_hops,
                })
            }
            LogicalExpression::CountSubquery(subplan) => {
                // The same edge check, counted; only a single edge counts the same.
                let check = self.extract_exists_pattern(subplan)?;
                if !check.one_edge {
                    return Err(Error::Internal(
                        "Unsupported COUNT subquery pattern".to_string(),
                    ));
                }
                Ok(FilterExpression::CountSubquery {
                    start_var: check.start_var,
                    end_var: check.end_var,
                    edge_var: check.edge_var,
                    direction: check.direction,
                    edge_types: check.edge_types,
                    end_labels: check.end_labels,
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

    /// Extracts the edge check that an `EXISTS` subplan reduces to, for the
    /// fast path that looks only at the start node's own edges.
    ///
    /// Accepts a single edge, like `(n)-[:TYPE]->()`, `(n)-[:TYPE]->(:Label)`
    /// or `()-[:TYPE]->(n)`, and a path of at least one edge with no
    /// condition on its end, like `(n)-[:TYPE*]->()`, in the WALK path mode.
    /// Everything else goes to the semi-join rewrite in `plan_filter`: a path
    /// with a minimum other than one hop, a labeled end or another path mode,
    /// a label on the start, an inner `WHERE` other than a label on the end,
    /// a node pattern apart from the edge, and patterns with no named node.
    ///
    /// The start is the pattern's named source, or its target when the source
    /// is anonymous (e.g. `()-[:CALLS]->(m)`, with the direction flipped).
    /// Which of the pattern's variables the outer row binds is known only per
    /// row: the evaluation matches its end and edge to the row's when it binds
    /// them, and `plan_filter` takes the fast path only for a start the
    /// outer row binds.
    pub(super) fn extract_exists_pattern(&self, subplan: &LogicalOperator) -> Result<EdgeCheck> {
        let unsupported = || Error::Internal("Unsupported EXISTS subquery pattern".to_string());
        match subplan {
            LogicalOperator::Expand(expand) => {
                // The Expand's input must be the plain scan of its source: another
                // Expand means more edges, a Filter an inner WHERE or a second label,
                // and a scan with an input a pattern before this one.
                let LogicalOperator::NodeScan(source) = expand.input.as_ref() else {
                    return Err(unsupported());
                };
                if expand.min_hops != 1 || source.input.is_some() {
                    return Err(unsupported());
                }
                let one_edge = expand.max_hops == Some(1) && !expand.quantified;
                // A path mode other than WALK (TRAIL, SIMPLE, ACYCLIC) limits the
                // paths of a longer pattern, which the check does not; a single
                // edge is expanded the same way in every mode.
                if !one_edge && expand.path_mode != PathMode::Walk {
                    return Err(unsupported());
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
                    // Outer variable on the target side, e.g. ()-[:CALLS]->(m).
                    // Flip direction: "does m have an incoming CALLS edge?" The
                    // source's label becomes a label of the far end.
                    let direction = match expand.direction {
                        ExpandDirection::Outgoing => Direction::Incoming,
                        ExpandDirection::Incoming => Direction::Outgoing,
                        ExpandDirection::Both => Direction::Both,
                    };
                    let end_labels = source.label.clone().map(|label| vec![label]);
                    if end_labels.is_some() && !one_edge {
                        return Err(unsupported());
                    }
                    Ok(EdgeCheck {
                        start_var: expand.to_variable.clone(),
                        end_var: expand.from_variable.clone(),
                        edge_var: expand.edge_variable.clone(),
                        direction,
                        edge_types: expand.edge_types.clone(),
                        end_labels,
                        one_edge,
                        max_hops: expand.max_hops,
                    })
                } else {
                    // Outer variable on the source side, e.g. (m)-[:CALLS]->(). A
                    // label on it here (`(m:Label)-[:CALLS]->()`) is a condition
                    // the edge check cannot make.
                    if source.label.is_some() {
                        return Err(unsupported());
                    }
                    let direction = match expand.direction {
                        ExpandDirection::Outgoing => Direction::Outgoing,
                        ExpandDirection::Incoming => Direction::Incoming,
                        ExpandDirection::Both => Direction::Both,
                    };
                    Ok(EdgeCheck {
                        start_var: expand.from_variable.clone(),
                        end_var: expand.to_variable.clone(),
                        edge_var: expand.edge_variable.clone(),
                        direction,
                        edge_types: expand.edge_types.clone(),
                        end_labels: None,
                        one_edge,
                        max_hops: expand.max_hops,
                    })
                }
            }
            // A node pattern after the edge, like the second MATCH of
            // `MATCH (n)-[:R]->(m) MATCH (m:Label)`, reuses the node of the edge it
            // names: a label on the end joins the end labels. Any other node is a
            // pattern of its own, which the edge check cannot make.
            LogicalOperator::NodeScan(scan) => {
                let Some(input) = &scan.input else {
                    return Err(Error::Internal(
                        "EXISTS subquery must contain an edge pattern".to_string(),
                    ));
                };
                let mut check = self.extract_exists_pattern(input)?;
                if scan.variable == check.end_var && (scan.label.is_none() || check.one_edge) {
                    if let Some(label) = &scan.label {
                        check
                            .end_labels
                            .get_or_insert_with(Vec::new)
                            .push(label.clone());
                    }
                    Ok(check)
                } else if scan.variable == check.start_var && scan.label.is_none() {
                    Ok(check)
                } else {
                    Err(unsupported())
                }
            }
            // A label on the far end of a single edge, e.g.
            // EXISTS { (u)<-[:AUTH]-(:Identity) }, joins the end labels. Any other
            // filter (a property, a label on another variable, a label on the end
            // of a longer path) goes to the semi-join path.
            LogicalOperator::Filter(filter_op) => {
                let (variable, label) = self
                    .label_condition(&filter_op.predicate)
                    .ok_or_else(unsupported)?;
                let mut check = self.extract_exists_pattern(&filter_op.input)?;
                if variable != check.end_var || !check.one_edge {
                    return Err(unsupported());
                }
                check.end_labels.get_or_insert_with(Vec::new).push(label);
                Ok(check)
            }
            _ => Err(unsupported()),
        }
    }

    /// The variable and label of a `hasLabel(variable, 'Label')` predicate.
    fn label_condition<'a>(&self, predicate: &'a LogicalExpression) -> Option<(&'a str, String)> {
        match predicate {
            LogicalExpression::FunctionCall { name, args, .. } if name == "hasLabel" => {
                match (args.first(), args.get(1)) {
                    (
                        Some(LogicalExpression::Variable(variable)),
                        Some(LogicalExpression::Literal(Value::String(label))),
                    ) => Some((variable, label.to_string())),
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

/// The check on a node's own edges that an `EXISTS` or `COUNT` subquery
/// reduces to (see `extract_exists_pattern`).
pub(super) struct EdgeCheck {
    /// The variable the check starts from.
    pub start_var: String,
    /// The pattern's other end.
    pub end_var: String,
    /// The pattern's edge variable, if it has one.
    pub edge_var: Option<String>,
    /// The direction of the edge, seen from the start.
    pub direction: Direction,
    /// The edge types to match (empty matches any).
    pub edge_types: Vec<String>,
    /// The labels the other end must all have.
    pub end_labels: Option<Vec<String>>,
    /// Whether the pattern is a single edge. A longer path is accepted only for
    /// `EXISTS`, where it exists when its first edge does (unless the row
    /// binds its other end or edge), but counts differently.
    pub one_edge: bool,
    /// The maximum number of hops of a longer path (`None`: unbounded).
    pub max_hops: Option<u32>,
}
