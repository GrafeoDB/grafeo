//! Aggregate extraction from RETURN items.

#[allow(clippy::wildcard_imports)]
use super::*;

impl GqlTranslator {
    /// Takes the horizontal aggregates out of `items` (a RETURN or WITH list),
    /// the keys of its `order_by` and its `having` condition.
    ///
    /// An aggregate over a property of a group variable, as `sum(e.w)` for
    /// the edges `e` of a quantified edge pattern, is computed for each row
    /// over the variable's list (horizontal aggregation, ISO/IEC 39075:2024
    /// 20.9 <aggregate function>; a group variable, 16.7): not over the rows.
    /// Each one becomes a [`HorizontalAggregateOp`] on `plan`, into a column
    /// of its own that the returned items read instead, so the items keep a
    /// regular aggregate only where they have one beside it, and group by
    /// the horizontal ones then. An item that changes keeps the name it had:
    /// its alias, or its text (`sum(e.w)`). The keys of `order_by` and the
    /// `having` condition read the same columns (`ORDER BY sum(e.w)`,
    /// `HAVING max(sum(e.w)) > 100`).
    ///
    /// # Errors
    ///
    /// Returns an error for a binary set function (`covar_samp(e.w, e.v)`)
    /// over a group variable, which is not supported.
    pub(super) fn take_horizontal_aggregates(
        &self,
        items: &[ast::ReturnItem],
        order_by: Option<&ast::OrderByClause>,
        having: Option<&ast::Expression>,
        mut plan: LogicalOperator,
    ) -> Result<WithoutHorizontalAggregates> {
        if self.group_list_variables.borrow().is_empty() {
            return Ok(WithoutHorizontalAggregates {
                items: items.to_vec(),
                order_by: order_by.cloned(),
                having: having.cloned(),
                plan,
            });
        }
        // The column of each horizontal aggregate, by its text, so the same
        // aggregate twice is computed once
        let mut columns: HashMap<String, String> = HashMap::new();
        let mut taken = Vec::with_capacity(items.len());
        for item in items {
            let mut expression = item.expression.clone();
            if !self.substitute_horizontal(&mut expression, &mut plan, &mut columns)? {
                taken.push(item.clone());
                continue;
            }
            let alias = match &item.alias {
                Some(alias) => alias.clone(),
                None => match self.try_extract_aggregate(&item.expression, &None)? {
                    Some(aggregate) => {
                        crate::query::planner::common::aggregate_column_name(&aggregate)
                    }
                    None => crate::query::planner::common::expression_to_string(
                        &self.translate_expression(&item.expression)?,
                    ),
                },
            };
            taken.push(ast::ReturnItem {
                expression,
                alias: Some(alias),
                span: item.span,
            });
        }
        // A key that is a returned aggregate reads the item's column (the
        // result names it after the item), any other one the aggregate's own
        let returned: HashMap<String, String> = taken
            .iter()
            .filter_map(|item| match (&item.expression, &item.alias) {
                (ast::Expression::Variable(column), Some(alias)) => {
                    Some((column.clone(), alias.clone()))
                }
                _ => None,
            })
            .collect();
        let order_by = order_by
            .map(|clause| {
                let mut clause = clause.clone();
                for key in &mut clause.items {
                    self.substitute_horizontal(&mut key.expression, &mut plan, &mut columns)?;
                    if let ast::Expression::Variable(column) = &key.expression
                        && let Some(alias) = returned.get(column)
                    {
                        key.expression = ast::Expression::Variable(alias.clone());
                    }
                }
                Ok::<_, Error>(clause)
            })
            .transpose()?;
        let having = having
            .map(|condition| {
                let mut condition = condition.clone();
                self.substitute_horizontal(&mut condition, &mut plan, &mut columns)?;
                Ok::<_, Error>(condition)
            })
            .transpose()?;
        Ok(WithoutHorizontalAggregates {
            items: taken,
            order_by,
            having,
            plan,
        })
    }

    /// Replaces each horizontal aggregate in `expr` (see
    /// [`take_horizontal_aggregates`](Self::take_horizontal_aggregates)) by
    /// the column a [`HorizontalAggregateOp`] on `plan` computes it into.
    /// Returns whether `expr` changed.
    fn substitute_horizontal(
        &self,
        expr: &mut ast::Expression,
        plan: &mut LogicalOperator,
        columns: &mut HashMap<String, String>,
    ) -> Result<bool> {
        let group_variable = match &*expr {
            ast::Expression::FunctionCall { name, args, .. } if is_aggregate_function(name) => {
                match args.first() {
                    Some(ast::Expression::PropertyAccess { variable, property })
                        if self.group_list_variables.borrow().contains_key(variable)
                            && binds_edge_list(plan, variable) =>
                    {
                        Some((variable.clone(), property.clone()))
                    }
                    _ => None,
                }
            }
            _ => None,
        };
        if let Some((list_column, property)) = group_variable {
            let Some(aggregate) = self.try_extract_aggregate(expr, &None)? else {
                return Ok(false);
            };
            if aggregate.expression2.is_some() {
                return Err(Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    format!(
                        "a binary set function over the group variable '{list_column}' is not \
                         supported"
                    ),
                )));
            }
            let text = crate::query::planner::common::aggregate_column_name(&aggregate);
            let column = match columns.get(&text) {
                Some(column) => column.clone(),
                None => {
                    let column = format!("_horizontal_{}", rand_id());
                    let input = std::mem::replace(plan, LogicalOperator::Empty);
                    *plan = LogicalOperator::HorizontalAggregate(HorizontalAggregateOp {
                        list_column,
                        entity_kind: EntityKind::Edge,
                        function: aggregate.function,
                        distinct: aggregate.distinct,
                        percentile: aggregate.percentile,
                        separator: aggregate.separator,
                        property,
                        alias: column.clone(),
                        input: Box::new(input),
                    });
                    columns.insert(text, column.clone());
                    column
                }
            };
            *expr = ast::Expression::Variable(column);
            return Ok(true);
        }
        let mut changed = false;
        match expr {
            ast::Expression::FunctionCall { args, .. } | ast::Expression::List(args) => {
                for arg in args {
                    changed |= self.substitute_horizontal(arg, plan, columns)?;
                }
            }
            ast::Expression::Binary { left, right, .. } => {
                changed |= self.substitute_horizontal(left, plan, columns)?;
                changed |= self.substitute_horizontal(right, plan, columns)?;
            }
            ast::Expression::Unary { operand, .. } => {
                changed |= self.substitute_horizontal(operand, plan, columns)?;
            }
            ast::Expression::Case {
                input,
                whens,
                else_clause,
            } => {
                if let Some(input) = input {
                    changed |= self.substitute_horizontal(input, plan, columns)?;
                }
                for (condition, result) in whens {
                    changed |= self.substitute_horizontal(condition, plan, columns)?;
                    changed |= self.substitute_horizontal(result, plan, columns)?;
                }
                if let Some(else_clause) = else_clause {
                    changed |= self.substitute_horizontal(else_clause, plan, columns)?;
                }
            }
            _ => {}
        }
        Ok(changed)
    }

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

/// Whether the rows of `plan` hold `variable` as the edge list of a quantified
/// edge pattern (a group variable): the nearest clause that binds the name is
/// such a pattern, or passes its list on unchanged (`WITH b, e`). A later
/// pattern, UNWIND or WITH that binds the name to something else hides it.
fn binds_edge_list(plan: &LogicalOperator, variable: &str) -> bool {
    let named = |name: &Option<String>| name.as_deref() == Some(variable);
    match plan {
        LogicalOperator::Expand(expand) if named(&expand.edge_variable) => {
            expand.is_variable_length()
        }
        LogicalOperator::ShortestPath(search) if named(&search.edge_variable) => {
            search.quantified
        }
        LogicalOperator::Project(project) => {
            match project
                .projections
                .iter()
                .find(|projection| match (&projection.alias, &projection.expression) {
                    (Some(alias), _) => alias == variable,
                    (None, LogicalExpression::Variable(name)) => name == variable,
                    _ => false,
                }) {
                Some(projection) => {
                    matches!(&projection.expression, LogicalExpression::Variable(name) if name == variable)
                        && binds_edge_list(&project.input, variable)
                }
                None => project.pass_through_input && binds_edge_list(&project.input, variable),
            }
        }
        LogicalOperator::Return(ret) => ret.items.iter().any(|item| {
            matches!(&item.expression, LogicalExpression::Variable(name) if name == variable || name == "*")
                && item.alias.as_deref().is_none_or(|alias| alias == variable)
        }) && binds_edge_list(&ret.input, variable),
        LogicalOperator::Aggregate(aggregate) => {
            aggregate.group_by.iter().any(
                |key| matches!(key, LogicalExpression::Variable(name) if name == variable),
            ) && binds_edge_list(&aggregate.input, variable)
        }
        LogicalOperator::NodeScan(scan) if scan.variable == variable => false,
        LogicalOperator::Unwind(unwind) if unwind.variable == variable => false,
        LogicalOperator::Bind(bind) if bind.variable == variable => false,
        other => other
            .children()
            .into_iter()
            .any(|child| binds_edge_list(child, variable)),
    }
}

/// A RETURN or WITH clause whose horizontal aggregates are computed by its
/// input `plan` (see `GqlTranslator::take_horizontal_aggregates`): its
/// items, ORDER BY and HAVING read their columns instead.
pub(super) struct WithoutHorizontalAggregates {
    /// The items of the clause.
    pub(super) items: Vec<ast::ReturnItem>,
    /// The ORDER BY of the clause.
    pub(super) order_by: Option<ast::OrderByClause>,
    /// The HAVING condition of the clause.
    pub(super) having: Option<ast::Expression>,
    /// The input of the clause, with the horizontal aggregates.
    pub(super) plan: LogicalOperator,
}

/// What a HAVING condition reads: the output of the Aggregate, whose columns
/// are the grouping keys (named by their text, `m.name`) and the aggregates.
/// A grouping key written in the condition reads its key column, and an
/// alias of the RETURN list reads what the result computes for it (`k` for
/// `m.name AS k`, `c` for `count(*) + 1 AS c`).
pub(super) struct HavingScope {
    keys: HashSet<String>,
    aliases: HashMap<String, LogicalExpression>,
}

impl HavingScope {
    /// The scope of an Aggregate grouped by `group_by`, whose result
    /// `post_return` projects (if it projects one).
    pub(super) fn new(group_by: &[LogicalExpression], post_return: Option<&[ReturnItem]>) -> Self {
        let keys = group_by
            .iter()
            .map(crate::query::planner::common::expression_to_string)
            .collect();
        let aliases = post_return
            .unwrap_or_default()
            .iter()
            .filter_map(|item| Some((item.alias.clone()?, item.expression.clone())))
            .collect();
        Self { keys, aliases }
    }

    /// `expr` reading the Aggregate's output (see [`HavingScope`]).
    pub(super) fn resolve(&self, expr: LogicalExpression) -> LogicalExpression {
        if let LogicalExpression::Variable(name) = &expr {
            return self.aliases.get(name).cloned().unwrap_or(expr);
        }
        let text = crate::query::planner::common::expression_to_string(&expr);
        if self.keys.contains(&text) {
            return LogicalExpression::Variable(text);
        }
        match expr {
            LogicalExpression::Binary { left, op, right } => LogicalExpression::Binary {
                left: Box::new(self.resolve(*left)),
                op,
                right: Box::new(self.resolve(*right)),
            },
            LogicalExpression::Unary { op, operand } => LogicalExpression::Unary {
                op,
                operand: Box::new(self.resolve(*operand)),
            },
            LogicalExpression::FunctionCall {
                name,
                args,
                distinct,
            } => LogicalExpression::FunctionCall {
                name,
                args: args.into_iter().map(|arg| self.resolve(arg)).collect(),
                distinct,
            },
            LogicalExpression::List(items) => {
                LogicalExpression::List(items.into_iter().map(|item| self.resolve(item)).collect())
            }
            LogicalExpression::Case {
                operand,
                when_clauses,
                else_clause,
            } => LogicalExpression::Case {
                operand: operand.map(|operand| Box::new(self.resolve(*operand))),
                when_clauses: when_clauses
                    .into_iter()
                    .map(|(condition, result)| (self.resolve(condition), self.resolve(result)))
                    .collect(),
                else_clause: else_clause.map(|clause| Box::new(self.resolve(*clause))),
            },
            other => other,
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
