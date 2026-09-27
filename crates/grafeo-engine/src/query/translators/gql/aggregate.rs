//! Aggregate extraction from RETURN items.

#[allow(clippy::wildcard_imports)]
use super::*;

impl GqlTranslator {
    /// Extracts aggregate and group-by expressions from RETURN items.
    ///
    /// Returns `(aggregates, group_by, post_return)`. The Aggregate operator
    /// outputs its grouping keys first, then its aggregates, named by alias or
    /// [`aggregate_column_name`](crate::query::planner::common::aggregate_column_name).
    /// `post_return` is `Some(...)` whenever that output differs from what the
    /// items ask for: an item wraps an aggregate (`count(n) > 0 AS exists`),
    /// an item has an alias, a grouping key follows an aggregate (so columns
    /// must be reordered), or `explicit_group_by` is set (the GROUP BY keys may
    /// not all be returned).
    pub(super) fn extract_aggregates_and_groups(
        &self,
        items: &[ast::ReturnItem],
        explicit_group_by: bool,
    ) -> Result<(
        Vec<AggregateExpr>,
        Vec<LogicalExpression>,
        Option<Vec<ReturnItem>>,
    )> {
        let mut aggregates = Vec::new();
        let mut group_by = Vec::new();
        let mut needs_post_return = false;
        let mut post_return_items = Vec::new();
        let mut agg_counter: u32 = 0;
        let mut seen_aggregate = false;

        for item in items {
            if let Some(mut agg_expr) = self.try_extract_aggregate(&item.expression, &item.alias)? {
                // Direct aggregate (e.g. `count(n) AS cnt`). An unaliased one is
                // named after its source text (`count(n)`); the name doubles as
                // the alias so the binder resolves the post-Return reference.
                let column = agg_expr.alias.clone().unwrap_or_else(|| {
                    crate::query::planner::common::aggregate_column_name(&agg_expr)
                });
                agg_expr.alias = Some(column.clone());
                aggregates.push(agg_expr);
                post_return_items.push(ReturnItem {
                    expression: LogicalExpression::Variable(column),
                    alias: item.alias.clone(),
                });
                seen_aggregate = true;
            } else if contains_aggregate(&item.expression) {
                // Wrapped aggregate (e.g. `count(n) > 0 AS exists`,
                // or `sum(a) + count(b)` with multiple aggregates).
                needs_post_return = true;
                seen_aggregate = true;

                let substitute = self.extract_wrapped_aggregates(
                    &item.expression,
                    &mut agg_counter,
                    &mut aggregates,
                )?;
                // Unaliased, the column is named after the written expression
                // (`sum(n.v) + 1`), not the substitute (`_agg_0 + 1`).
                let alias = item.alias.clone().or_else(|| {
                    self.translate_expression(&item.expression)
                        .ok()
                        .map(|expr| crate::query::planner::common::expression_to_string(&expr))
                });
                post_return_items.push(ReturnItem {
                    expression: substitute,
                    alias,
                });
            } else {
                // Non-aggregate expression: group-by key.
                // The Aggregate operator names its output columns using
                // expression_to_string, so the post-Return must reference
                // those column names (not the raw property expression).
                // Keys come first in the Aggregate's output, so a key after an
                // aggregate needs the post-Return to restore the item order.
                needs_post_return |= seen_aggregate;
                let expr = self.translate_expression(&item.expression)?;
                group_by.push(expr.clone());
                let col_name = crate::query::planner::common::expression_to_string(&expr);
                post_return_items.push(ReturnItem {
                    expression: LogicalExpression::Variable(col_name),
                    alias: item.alias.clone(),
                });
            }
        }

        // Always produce a post-Return when any item has an alias, so that
        // output column names reflect the aliases and are visible to ORDER BY.
        let has_aliases = items.iter().any(|item| item.alias.is_some());
        if needs_post_return || has_aliases || explicit_group_by {
            Ok((aggregates, group_by, Some(post_return_items)))
        } else {
            Ok((aggregates, group_by, None))
        }
    }

    /// Extracts all aggregates from a wrapping expression, assigning each a
    /// unique synthetic alias via `agg_counter`. Extracted aggregates are
    /// pushed to `aggregates_out`. Returns the substituted expression with
    /// aggregate positions replaced by variable references.
    pub(super) fn extract_wrapped_aggregates(
        &self,
        expr: &ast::Expression,
        agg_counter: &mut u32,
        aggregates_out: &mut Vec<AggregateExpr>,
    ) -> Result<LogicalExpression> {
        match expr {
            ast::Expression::FunctionCall {
                name,
                args,
                distinct,
            } => {
                // If this function IS an aggregate, extract it directly.
                let alias = format!("_agg_{agg_counter}");
                if let Some(agg) = self.try_extract_aggregate(expr, &Some(alias.clone()))? {
                    *agg_counter += 1;
                    aggregates_out.push(agg);
                    return Ok(LogicalExpression::Variable(alias));
                }
                // Non-aggregate function wrapping aggregate arguments.
                // Process ALL args, extracting aggregates from each.
                let mut translated_args = Vec::with_capacity(args.len());
                for arg in args {
                    if contains_aggregate(arg) {
                        let sub =
                            self.extract_wrapped_aggregates(arg, agg_counter, aggregates_out)?;
                        translated_args.push(sub);
                    } else {
                        translated_args.push(self.translate_expression(arg)?);
                    }
                }
                Ok(LogicalExpression::FunctionCall {
                    name: name.clone(),
                    args: translated_args,
                    distinct: *distinct,
                })
            }
            ast::Expression::Binary { left, op, right } => {
                let binary_op = self.translate_binary_op(*op);
                let left_sub = if contains_aggregate(left) {
                    self.extract_wrapped_aggregates(left, agg_counter, aggregates_out)?
                } else {
                    self.translate_expression(left)?
                };
                let right_sub = if contains_aggregate(right) {
                    self.extract_wrapped_aggregates(right, agg_counter, aggregates_out)?
                } else {
                    self.translate_expression(right)?
                };
                Ok(LogicalExpression::Binary {
                    left: Box::new(left_sub),
                    op: binary_op,
                    right: Box::new(right_sub),
                })
            }
            ast::Expression::Unary { op, operand } => {
                let sub = self.extract_wrapped_aggregates(operand, agg_counter, aggregates_out)?;
                if *op == ast::UnaryOp::Pos {
                    return Ok(sub);
                }
                let unary_op = self.translate_unary_op(*op);
                Ok(LogicalExpression::Unary {
                    op: unary_op,
                    operand: Box::new(sub),
                })
            }
            ast::Expression::Case {
                input,
                whens,
                else_clause,
            } => {
                // For CASE, extract all aggregates from branches.
                // We translate the full CASE and replace aggregate positions.
                let operand = match input {
                    Some(inp) if contains_aggregate(inp) => Some(Box::new(
                        self.extract_wrapped_aggregates(inp, agg_counter, aggregates_out)?,
                    )),
                    Some(inp) => Some(Box::new(self.translate_expression(inp)?)),
                    None => None,
                };
                let mut when_clauses = Vec::with_capacity(whens.len());
                for (cond, then) in whens {
                    let cond_expr = if contains_aggregate(cond) {
                        self.extract_wrapped_aggregates(cond, agg_counter, aggregates_out)?
                    } else {
                        self.translate_expression(cond)?
                    };
                    let then_expr = if contains_aggregate(then) {
                        self.extract_wrapped_aggregates(then, agg_counter, aggregates_out)?
                    } else {
                        self.translate_expression(then)?
                    };
                    when_clauses.push((cond_expr, then_expr));
                }
                let else_expr = match else_clause {
                    Some(el) if contains_aggregate(el) => Some(Box::new(
                        self.extract_wrapped_aggregates(el, agg_counter, aggregates_out)?,
                    )),
                    Some(el) => Some(Box::new(self.translate_expression(el)?)),
                    None => None,
                };
                Ok(LogicalExpression::Case {
                    operand,
                    when_clauses,
                    else_clause: else_expr,
                })
            }
            _ => Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Unsupported expression wrapping an aggregate",
            ))),
        }
    }

    /// Tries to extract an aggregate expression from an AST expression.
    pub(super) fn try_extract_aggregate(
        &self,
        expr: &ast::Expression,
        alias: &Option<String>,
    ) -> Result<Option<AggregateExpr>> {
        match expr {
            ast::Expression::FunctionCall {
                name,
                args,
                distinct,
            } => {
                if let Some(func) = to_aggregate_function(name) {
                    let agg_expr = if args.is_empty() {
                        // COUNT(*) case
                        AggregateExpr {
                            function: func,
                            expression: None,
                            expression2: None,
                            distinct: *distinct,
                            alias: alias.clone(),
                            percentile: None,
                            separator: None,
                        }
                    } else {
                        // COUNT(x), SUM(x), etc.
                        // For COUNT with an expression, use CountNonNull to ensure we fetch values
                        let actual_func = if func == AggregateFunction::Count {
                            AggregateFunction::CountNonNull
                        } else {
                            func
                        };
                        // Extract percentile parameter for percentile functions
                        let percentile = if matches!(
                            actual_func,
                            AggregateFunction::PercentileDisc | AggregateFunction::PercentileCont
                        ) && args.len() >= 2
                        {
                            // Second argument is the percentile value
                            if let ast::Expression::Literal(ast::Literal::Float(p)) = &args[1] {
                                Some((*p).clamp(0.0, 1.0))
                            } else if let ast::Expression::Literal(ast::Literal::Integer(p)) =
                                &args[1]
                            {
                                Some((*p as f64).clamp(0.0, 1.0))
                            } else {
                                Some(0.5) // Default to median
                            }
                        } else {
                            None
                        };
                        // Extract second argument for binary set functions
                        let expression2 = if is_binary_set_function(actual_func) && args.len() >= 2
                        {
                            Some(self.translate_expression(&args[1])?)
                        } else {
                            None
                        };
                        // Extract separator for LISTAGG / GROUP_CONCAT
                        let upper_name = name.to_uppercase();
                        let separator = if actual_func == AggregateFunction::GroupConcat {
                            if args.len() >= 2 {
                                // Second argument is the separator string
                                if let ast::Expression::Literal(ast::Literal::String(s)) = &args[1]
                                {
                                    Some(s.clone())
                                } else if upper_name == "LISTAGG" {
                                    Some(",".to_string())
                                } else {
                                    None // GROUP_CONCAT default (space) handled in AggregateState
                                }
                            } else if upper_name == "LISTAGG" {
                                Some(",".to_string()) // ISO GQL default for LISTAGG
                            } else {
                                None // GROUP_CONCAT default (space) handled in AggregateState
                            }
                        } else {
                            None
                        };
                        AggregateExpr {
                            function: actual_func,
                            expression: Some(self.translate_expression(&args[0])?),
                            expression2,
                            distinct: *distinct,
                            alias: alias.clone(),
                            percentile,
                            separator,
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

/// Checks if an AST expression contains an aggregate function call.
pub(super) fn contains_aggregate(expr: &ast::Expression) -> bool {
    match expr {
        ast::Expression::FunctionCall { name, args, .. } => {
            is_aggregate_function(name) || args.iter().any(contains_aggregate)
        }
        ast::Expression::Binary { left, right, .. } => {
            contains_aggregate(left) || contains_aggregate(right)
        }
        ast::Expression::Unary { operand, .. } => contains_aggregate(operand),
        ast::Expression::Case {
            input,
            whens,
            else_clause,
        } => {
            input.as_deref().is_some_and(contains_aggregate)
                || whens
                    .iter()
                    .any(|(w, t)| contains_aggregate(w) || contains_aggregate(t))
                || else_clause.as_deref().is_some_and(contains_aggregate)
        }
        ast::Expression::List(items) => items.iter().any(contains_aggregate),
        ast::Expression::ListComprehension {
            filter_expr,
            map_expr,
            ..
        } => filter_expr.as_deref().is_some_and(contains_aggregate) || contains_aggregate(map_expr),
        _ => false,
    }
}
