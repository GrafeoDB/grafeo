//! Aggregate extraction from RETURN items.

#[allow(clippy::wildcard_imports)]
use super::*;
pub(super) use crate::query::translators::common::HavingScope;

// Shared liveness belongs to the lexical binding, including nested scopes.
#[derive(Clone)]
struct AggregateLetBinding {
    name: String,
    value: ast::Expression,
    // Fully resolved in the binding's original input scope, independently
    // of the lexical value retained for the scalar result.
    input_value: ast::Expression,
    local_origin: Option<String>,
    used: std::rc::Rc<std::cell::Cell<bool>>,
}

#[derive(Clone, Copy)]
enum AggregateSubstitutionMode {
    Scalar,
    Argument { crosses_local_binding: bool },
}

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
        self.substitute_horizontal_in_scope(expr, plan, columns, &[])
    }

    /// Horizontal aggregates read the group variable in their lexical scope.
    /// Local binders shadow it; LET aliases may name the same group variable.
    fn substitute_horizontal_in_scope(
        &self,
        expr: &mut ast::Expression,
        plan: &mut LogicalOperator,
        columns: &mut HashMap<String, String>,
        bindings: &[(String, Option<String>)],
    ) -> Result<bool> {
        let group_variable = match &*expr {
            ast::Expression::FunctionCall { name, args, .. } if is_aggregate_function(name) => {
                match args.first() {
                    Some(ast::Expression::PropertyAccess { variable, property }) => {
                        let variable = bindings
                            .iter()
                            .rev()
                            .find(|(name, _)| name == variable)
                            .map_or_else(|| Some(variable.clone()), |(_, target)| target.clone());
                        variable
                            .filter(|variable| {
                                self.group_list_variables.borrow().contains_key(variable)
                                    && binds_edge_list(plan, variable)
                            })
                            .map(|variable| (variable, property.clone()))
                    }
                    _ => None,
                }
            }
            _ => None,
        };
        if let Some((list_column, property)) = group_variable {
            // Canonicalize a LET alias before deduplicating the aggregate, so
            // two local aliases with the same name cannot share a wrong column.
            let mut canonical = expr.clone();
            if let ast::Expression::FunctionCall { args, .. } = &mut canonical
                && let Some(ast::Expression::PropertyAccess { variable, .. }) = args.first_mut()
            {
                variable.clone_from(&list_column);
            }
            let Some(aggregate) = self.try_extract_aggregate(&canonical, &None)? else {
                return Ok(false);
            };
            if aggregate.expression2.is_some() {
                return Err(Error::from(QueryError::new(
                    QueryErrorKind::Unsupported,
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
                    let column = self.names.next("_horizontal_");
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
        let mut substitute = |child: &mut ast::Expression| {
            self.substitute_horizontal_in_scope(child, plan, columns, bindings)
        };
        match expr {
            ast::Expression::FunctionCall { args, .. } | ast::Expression::List(args) => {
                for arg in args {
                    changed |= substitute(arg)?;
                }
            }
            ast::Expression::Binary { left, right, .. }
            | ast::Expression::IndexAccess {
                base: left,
                index: right,
            } => {
                changed |= substitute(left)?;
                changed |= substitute(right)?;
            }
            ast::Expression::Unary { operand, .. } => {
                changed |= substitute(operand)?;
            }
            ast::Expression::MapAccess { base, .. } => {
                changed |= substitute(base)?;
            }
            ast::Expression::SliceAccess { base, start, end } => {
                changed |= substitute(base)?;
                for bound in [start, end].into_iter().flatten() {
                    changed |= substitute(bound)?;
                }
            }
            ast::Expression::Map(entries) => {
                for (_, value) in entries {
                    changed |= substitute(value)?;
                }
            }
            ast::Expression::Case {
                input,
                whens,
                else_clause,
            } => {
                if let Some(input) = input {
                    changed |= substitute(input)?;
                }
                for (condition, result) in whens {
                    changed |= substitute(condition)?;
                    changed |= substitute(result)?;
                }
                if let Some(else_clause) = else_clause {
                    changed |= substitute(else_clause)?;
                }
            }
            ast::Expression::ListComprehension {
                variable,
                list_expr,
                filter_expr,
                map_expr,
            } => {
                changed |= substitute(list_expr)?;
                let mut bindings = bindings.to_vec();
                bindings.push((variable.clone(), None));
                if let Some(filter) = filter_expr {
                    changed |=
                        self.substitute_horizontal_in_scope(filter, plan, columns, &bindings)?;
                }
                changed |=
                    self.substitute_horizontal_in_scope(map_expr, plan, columns, &bindings)?;
            }
            ast::Expression::ListPredicate {
                variable,
                list_expr,
                predicate,
                ..
            } => {
                changed |= substitute(list_expr)?;
                let mut bindings = bindings.to_vec();
                bindings.push((variable.clone(), None));
                changed |=
                    self.substitute_horizontal_in_scope(predicate, plan, columns, &bindings)?;
            }
            ast::Expression::Reduce {
                accumulator,
                initial,
                variable,
                list,
                expression,
            } => {
                changed |= substitute(initial)?;
                changed |= substitute(list)?;
                let mut bindings = bindings.to_vec();
                bindings.push((accumulator.clone(), None));
                bindings.push((variable.clone(), None));
                changed |=
                    self.substitute_horizontal_in_scope(expression, plan, columns, &bindings)?;
            }
            ast::Expression::LetIn {
                bindings: let_bindings,
                body,
            } => {
                for (_, value) in let_bindings.iter_mut() {
                    changed |= substitute(value)?;
                }
                // Each binding reads the enclosing scope, just as the normal
                // expression translator does, rather than preceding bindings.
                let mut body_bindings = bindings.to_vec();
                for (name, value) in let_bindings {
                    let target = match value {
                        ast::Expression::Variable(variable) => bindings
                            .iter()
                            .rev()
                            .find(|(name, _)| name.as_str() == variable.as_str())
                            .map_or_else(|| Some(variable.clone()), |(_, target)| target.clone()),
                        _ => None,
                    };
                    body_bindings.push((name.clone(), target));
                }
                changed |=
                    self.substitute_horizontal_in_scope(body, plan, columns, &body_bindings)?;
            }
            ast::Expression::Literal(_)
            | ast::Expression::Variable(_)
            | ast::Expression::Parameter(_)
            | ast::Expression::PropertyAccess { .. }
            | ast::Expression::ExistsSubquery { .. }
            | ast::Expression::CountSubquery { .. }
            | ast::Expression::ValueSubquery { .. } => {}
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
                post_return_items.push(ReturnItem {
                    expression: crate::query::translators::common::group_key_reference(&expr),
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

    /// The grouping keys of the explicit `GROUP BY` `group_by` of the RETURN
    /// list `items`, whose rows come from `plan`.
    ///
    /// In ISO GQL a grouping element is a name: `<grouping element> ::=
    /// <binding variable reference>` (ISO/IEC 39075:2024 <group by clause>).
    /// Grafeo also takes an expression over the incoming variables (`GROUP BY
    /// c.name`), as Microsoft Fabric does. A name that is the alias of a
    /// RETURN item groups by that item's value (`RETURN c.id AS cityId ...
    /// GROUP BY cityId`, Fabric's documented example); any other name, and
    /// every expression, reads the incoming variables. A key that repeats an
    /// earlier one is left out.
    ///
    /// # Errors
    ///
    /// Returns a semantic error for the alias of an aggregate (it is computed
    /// per group, so it is no key), for an alias that also names an incoming
    /// variable of another value (the two make different groups), and for
    /// an alias inside an expression (an alias stands for its item's value
    /// only on its own).
    pub(super) fn translate_group_by(
        &self,
        group_by: &[ast::Expression],
        items: &[ast::ReturnItem],
        plan: &LogicalOperator,
    ) -> Result<Vec<LogicalExpression>> {
        let incoming = plan.bound_variables(None);
        let is_incoming = |name: &str| incoming.as_ref().is_some_and(|names| names.contains(name));
        // The item an alias names, unless the item returns the variable of
        // that name (`RETURN c AS c`), which reads the same either way.
        let aliased = |name: &str| {
            items.iter().find(|item| {
                item.alias.as_deref() == Some(name)
                    && !matches!(&item.expression, ast::Expression::Variable(v) if v == name)
            })
        };
        let mut keys = Vec::new();
        let mut key_texts = HashSet::new();
        for element in group_by {
            let key = match element {
                ast::Expression::Variable(name) => match aliased(name) {
                    Some(item) if contains_aggregate(&item.expression) => {
                        return Err(grouping_error(format!(
                            "GROUP BY {name}: '{name}' is the alias of an aggregate, which is \
                             computed per group, so it cannot be a grouping key"
                        )));
                    }
                    Some(item) if is_incoming(name) => {
                        let value = crate::query::planner::common::expression_to_string(
                            &self.translate_expression(&item.expression)?,
                        );
                        return Err(grouping_error(format!(
                            "GROUP BY {name} is ambiguous: '{name}' is a variable of the grouped \
                             rows and the alias of the RETURN item {value} AS {name}, and the two \
                             make different groups; give that RETURN item another alias"
                        )));
                    }
                    Some(item) => self.translate_expression(&item.expression)?,
                    None => self.translate_expression(element)?,
                },
                _ => {
                    let key = self.translate_expression(element)?;
                    let mut names = HashSet::new();
                    collect_expression_variables(&key, &mut names);
                    if let Some(alias) = names
                        .iter()
                        .filter(|name| aliased(name).is_some() && !is_incoming(name))
                        .min()
                    {
                        return Err(grouping_error(format!(
                            "GROUP BY {}: '{alias}' is an alias of the RETURN list, which a \
                             grouping key names only on its own (GROUP BY {alias})",
                            crate::query::planner::common::expression_to_string(&key)
                        )));
                    }
                    key
                }
            };
            if key_texts.insert(crate::query::planner::common::expression_to_string(&key)) {
                keys.push(key);
            }
        }
        Ok(keys)
    }

    /// Makes the items of the RETURN list `items`, grouped by the explicit
    /// `keys`, read the output of the Aggregate. `post_return` holds what
    /// [`extract_aggregates_and_groups`](Self::extract_aggregates_and_groups)
    /// made of them (one item each, in order), and `aggregates` are their
    /// aggregates.
    ///
    /// The result has one row per group, so an item may read the grouping
    /// keys, the aggregates and constants, which have one value per group:
    /// a key itself (`p.name` for `GROUP BY p.name`, or through its alias),
    /// an expression over the keys (`p.name` or `upper(p.name)` for `GROUP BY
    /// p`, `count(*) + p.age`), as Neo4j's Cypher 25 `GROUP BY` and SQL's
    /// functionally dependent columns do, or a constant (`3 AS three`). Each
    /// such item is rewritten to read the key columns. A `CALL` body's import
    /// has one value per call, so an item may read it too: this returns the
    /// imports the items read, which the Aggregate's output must have back
    /// (see `with_imports`) before the items read it.
    ///
    /// # Errors
    ///
    /// Returns a semantic error naming the first item that reads another
    /// variable of the grouped rows outside an aggregate: it has a value per
    /// grouped row, not per group.
    pub(super) fn resolve_grouped_items(
        &self,
        items: &[ast::ReturnItem],
        post_return: &mut [ReturnItem],
        keys: &[LogicalExpression],
        aggregates: &[AggregateExpr],
    ) -> Result<Vec<String>> {
        let scope = HavingScope::new(keys, None);
        let key_texts: Vec<String> = keys
            .iter()
            .map(crate::query::planner::common::expression_to_string)
            .collect();
        let aggregate_columns: HashSet<&str> = aggregates
            .iter()
            .filter_map(|aggregate| aggregate.alias.as_deref())
            .collect();
        let mut imports_read = Vec::new();
        for (item, returned) in items.iter().zip(post_return.iter_mut()) {
            // A key or a direct aggregate already reads its column.
            if let LogicalExpression::Variable(column) = &returned.expression
                && (key_texts.contains(column) || aggregate_columns.contains(column.as_str()))
            {
                continue;
            }
            // A wrapped aggregate reads its aggregates' columns already.
            let expression = if contains_aggregate(&item.expression) {
                returned.expression.clone()
            } else {
                let expression = self.translate_expression(&item.expression)?;
                returned.alias.get_or_insert_with(|| {
                    crate::query::planner::common::expression_to_string(&expression)
                });
                expression
            };
            let resolved = scope.resolve(expression);
            let mut names = HashSet::new();
            collect_expression_variables(&resolved, &mut names);
            let mut read: Vec<&String> = names
                .iter()
                .filter(|name| self.call_imports.contains(name) && !imports_read.contains(*name))
                .collect();
            read.sort();
            imports_read.extend(read.into_iter().cloned());
            if let Some(name) = names
                .iter()
                .filter(|name| {
                    !key_texts.contains(*name)
                        && !aggregate_columns.contains(name.as_str())
                        && !self.call_imports.contains(name)
                })
                .min()
            {
                let written = self.translate_expression(&item.expression).map_or_else(
                    |_| returned.alias.clone().unwrap_or_default(),
                    |expression| crate::query::planner::common::expression_to_string(&expression),
                );
                // An aggregate's text drops its `*` (`count()`): name such an
                // item by its alias.
                let item_text = match &item.alias {
                    Some(alias) if contains_aggregate(&item.expression) => alias.clone(),
                    Some(alias) => format!("{written} AS {alias}"),
                    None => written,
                };
                return Err(grouping_error(format!(
                    "RETURN item {item_text} reads {name}, which is not a grouping key: with \
                     GROUP BY {}, a RETURN item may read the grouping keys, aggregates and \
                     constants, which have one value per group; add {name} to GROUP BY or \
                     aggregate it",
                    key_texts.join(", ")
                )));
            }
            returned.expression = resolved;
        }
        Ok(imports_read)
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
        let mut rewritten = expr.clone();
        self.substitute_aggregate_calls(
            &mut rewritten,
            agg_counter,
            aggregates_out,
            &[],
            &[],
            AggregateSubstitutionMode::Scalar,
        )?;
        self.translate_expression(&rewritten)
    }

    /// Replaces aggregate calls before the normal expression translator handles
    /// their wrappers. Nested queries keep their own aggregation scope.
    fn substitute_aggregate_calls(
        &self,
        expr: &mut ast::Expression,
        agg_counter: &mut u32,
        aggregates_out: &mut Vec<AggregateExpr>,
        bindings: &[AggregateLetBinding],
        locals: &[(String, String)],
        mode: AggregateSubstitutionMode,
    ) -> Result<()> {
        let take_aggregates = matches!(mode, AggregateSubstitutionMode::Scalar);
        if take_aggregates
            && matches!(expr, ast::Expression::FunctionCall { name, .. } if is_aggregate_function(name))
        {
            if let ast::Expression::FunctionCall { args, .. } = expr {
                for arg in args {
                    if let Some(name) = self.aggregate_local_origin(arg, bindings, locals)? {
                        return Err(Error::from(QueryError::new(
                            QueryErrorKind::Semantic,
                            format!("Aggregate cannot read locally bound variable '{name}'"),
                        )));
                    }
                    // Resolve aliases only for the input expression. Scalar
                    // LET references keep an ordinary lexical binding below.
                    self.substitute_aggregate_calls(
                        arg,
                        agg_counter,
                        aggregates_out,
                        bindings,
                        locals,
                        AggregateSubstitutionMode::Argument {
                            crosses_local_binding: !locals.is_empty() || !bindings.is_empty(),
                        },
                    )?;
                }
            }
            while self.names.is_written(&format!("_agg_{agg_counter}")) {
                *agg_counter += 1;
            }
            let alias = format!("_agg_{agg_counter}");
            if let Some(aggregate) = self.try_extract_aggregate(expr, &Some(alias.clone()))? {
                *agg_counter += 1;
                aggregates_out.push(aggregate);
                *expr = ast::Expression::Variable(alias);
                return Ok(());
            }
        }
        let mut substitute = |child: &mut ast::Expression| {
            self.substitute_aggregate_calls(
                child,
                agg_counter,
                aggregates_out,
                bindings,
                locals,
                mode,
            )
        };
        match expr {
            ast::Expression::Variable(name) => {
                if let Some(binding) = bindings.iter().rev().find(|binding| binding.name == *name) {
                    if take_aggregates {
                        binding.used.set(true);
                    } else {
                        let mut input = binding.input_value.clone();
                        self.substitute_aggregate_calls(
                            &mut input,
                            agg_counter,
                            aggregates_out,
                            &[],
                            locals,
                            mode,
                        )?;
                        *expr = input;
                    }
                }
            }
            ast::Expression::PropertyAccess { variable, property } => {
                if let Some(binding) = bindings
                    .iter()
                    .rev()
                    .find(|binding| binding.name == *variable)
                {
                    if take_aggregates {
                        binding.used.set(true);
                    } else {
                        let mut input = binding.input_value.clone();
                        self.substitute_aggregate_calls(
                            &mut input,
                            agg_counter,
                            aggregates_out,
                            &[],
                            locals,
                            mode,
                        )?;
                        *expr = ast::Expression::MapAccess {
                            base: Box::new(input),
                            key: property.clone(),
                        };
                    }
                }
            }
            ast::Expression::MapAccess { base, .. } => {
                substitute(base)?;
            }
            ast::Expression::FunctionCall { args, .. } | ast::Expression::List(args) => {
                for arg in args {
                    substitute(arg)?;
                }
            }
            ast::Expression::Binary { left, right, .. }
            | ast::Expression::IndexAccess {
                base: left,
                index: right,
            } => {
                substitute(left)?;
                substitute(right)?;
            }
            ast::Expression::Unary { operand, .. } => {
                substitute(operand)?;
            }
            ast::Expression::SliceAccess { base, start, end } => {
                substitute(base)?;
                for bound in [start, end].into_iter().flatten() {
                    substitute(bound)?;
                }
            }
            ast::Expression::Map(entries) => {
                for (_, value) in entries {
                    substitute(value)?;
                }
            }
            ast::Expression::Case {
                input,
                whens,
                else_clause,
            } => {
                if let Some(input) = input {
                    substitute(input)?;
                }
                for (condition, value) in whens {
                    substitute(condition)?;
                    substitute(value)?;
                }
                if let Some(value) = else_clause {
                    substitute(value)?;
                }
            }
            ast::Expression::ListComprehension {
                variable,
                list_expr,
                filter_expr,
                map_expr,
            } => {
                substitute(list_expr)?;
                let mut bindings = bindings.to_vec();
                let mut locals = locals.to_vec();
                *variable = self.aggregate_local_binding(variable, &mut bindings, &mut locals);
                if let Some(filter) = filter_expr {
                    self.substitute_aggregate_calls(
                        filter,
                        agg_counter,
                        aggregates_out,
                        &bindings,
                        &locals,
                        mode,
                    )?;
                }
                self.substitute_aggregate_calls(
                    map_expr,
                    agg_counter,
                    aggregates_out,
                    &bindings,
                    &locals,
                    mode,
                )?;
            }
            ast::Expression::ListPredicate {
                variable,
                list_expr,
                predicate,
                ..
            } => {
                substitute(list_expr)?;
                let mut bindings = bindings.to_vec();
                let mut locals = locals.to_vec();
                *variable = self.aggregate_local_binding(variable, &mut bindings, &mut locals);
                self.substitute_aggregate_calls(
                    predicate,
                    agg_counter,
                    aggregates_out,
                    &bindings,
                    &locals,
                    mode,
                )?;
            }
            ast::Expression::Reduce {
                accumulator,
                initial,
                variable,
                list,
                expression,
            } => {
                substitute(initial)?;
                substitute(list)?;
                let mut bindings = bindings.to_vec();
                let mut locals = locals.to_vec();
                *accumulator =
                    self.aggregate_local_binding(accumulator, &mut bindings, &mut locals);
                *variable = self.aggregate_local_binding(variable, &mut bindings, &mut locals);
                self.substitute_aggregate_calls(
                    expression,
                    agg_counter,
                    aggregates_out,
                    &bindings,
                    &locals,
                    mode,
                )?;
            }
            ast::Expression::LetIn {
                bindings: let_bindings,
                body,
            } => {
                // Evaluate every value in the enclosing scope, before any of
                // this LET's names bind. Aggregate input aliases additionally
                // record whether that value originated in a local scope.
                let used_before: Vec<_> =
                    bindings.iter().map(|binding| binding.used.get()).collect();
                let mut lexical = Vec::with_capacity(let_bindings.len());
                for (name, value) in let_bindings.iter_mut() {
                    self.substitute_aggregate_calls(
                        value,
                        agg_counter,
                        aggregates_out,
                        bindings,
                        locals,
                        mode,
                    )?;
                    let mut input_value = value.clone();
                    self.substitute_aggregate_calls(
                        &mut input_value,
                        agg_counter,
                        aggregates_out,
                        bindings,
                        locals,
                        AggregateSubstitutionMode::Argument {
                            crosses_local_binding: false,
                        },
                    )?;
                    lexical.push(AggregateLetBinding {
                        name: name.clone(),
                        value: value.clone(),
                        input_value,
                        local_origin: self.aggregate_local_origin(value, bindings, locals)?,
                        used: std::rc::Rc::new(std::cell::Cell::new(false)),
                    });
                }
                // A value of a binding used only by an aggregate does not
                // itself become a scalar read of an enclosing LET binding.
                for (binding, used) in bindings.iter().zip(used_before) {
                    binding.used.set(used);
                }
                let mut enclosing = bindings.to_vec();
                enclosing.extend(lexical.iter().cloned());
                self.substitute_aggregate_calls(
                    body,
                    agg_counter,
                    aggregates_out,
                    &enclosing,
                    locals,
                    mode,
                )?;
                // A binding read only by an extracted aggregate is no longer
                // a scalar input: retaining it could read an ungrouped column.
                lexical.retain(|binding| binding.used.get());
                for binding in &mut lexical {
                    self.substitute_aggregate_calls(
                        &mut binding.value,
                        agg_counter,
                        aggregates_out,
                        bindings,
                        locals,
                        mode,
                    )?;
                }
                *let_bindings = lexical
                    .into_iter()
                    .map(|binding| (binding.name, binding.value))
                    .collect();
            }
            ast::Expression::ExistsSubquery { .. }
            | ast::Expression::CountSubquery { .. }
            | ast::Expression::ValueSubquery { .. } => {
                if matches!(
                    mode,
                    AggregateSubstitutionMode::Argument {
                        crosses_local_binding: true
                    }
                ) {
                    return Err(Error::from(QueryError::new(
                        QueryErrorKind::Unsupported,
                        "Aggregate arguments containing a subquery cannot be lifted across a local binding",
                    )));
                }
                // The nested query owns its scope. Keep enclosing scalar LET
                // values available without attempting to rename its query AST.
                if take_aggregates {
                    for binding in bindings {
                        binding.used.set(true);
                    }
                }
            }
            ast::Expression::Literal(_) | ast::Expression::Parameter(_) => {}
        }
        Ok(())
    }

    /// Resolve an argument's original lexical origin before expanding aliases.
    /// The normal variable collector removes names bound inside the argument.
    fn aggregate_local_origin(
        &self,
        expr: &ast::Expression,
        bindings: &[AggregateLetBinding],
        locals: &[(String, String)],
    ) -> Result<Option<String>> {
        let mut reads = HashSet::new();
        collect_expression_variables(&self.translate_expression(expr)?, &mut reads);
        let mut reads: Vec<_> = reads.into_iter().collect();
        reads.sort();
        for name in reads {
            if let Some(binding) = bindings.iter().rev().find(|binding| binding.name == name) {
                if let Some(origin) = &binding.local_origin {
                    return Ok(Some(origin.clone()));
                }
            } else if let Some((original, _)) = locals.iter().find(|(_, local)| *local == name) {
                return Ok(Some(original.clone()));
            }
        }
        Ok(None)
    }

    fn aggregate_local_binding(
        &self,
        name: &str,
        bindings: &mut Vec<AggregateLetBinding>,
        locals: &mut Vec<(String, String)>,
    ) -> String {
        let name = name.to_string();
        bindings.push(AggregateLetBinding {
            name: name.clone(),
            value: ast::Expression::Variable(name.clone()),
            input_value: ast::Expression::Variable(name.clone()),
            local_origin: Some(name.clone()),
            used: std::rc::Rc::new(std::cell::Cell::new(false)),
        });
        locals.push((name.clone(), name.clone()));
        name
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
                if let Some(func) = AggregateFunction::from_name(name) {
                    crate::query::functions::check_aggregate_call(name, func, args.len())?;
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
                            } else if let ast::Expression::Literal(ast::Literal::Decimal(p)) =
                                &args[1]
                            {
                                Some(p.to_f64().clamp(0.0, 1.0))
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

/// A semantic error of a grouping query (see `translate_group_by`).
fn grouping_error(message: String) -> Error {
    Error::from(QueryError::new(QueryErrorKind::Semantic, message))
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

/// Checks if an AST expression contains an aggregate function call.
pub(super) fn contains_aggregate(expr: &ast::Expression) -> bool {
    match expr {
        ast::Expression::FunctionCall { name, args, .. } => {
            is_aggregate_function(name) || args.iter().any(contains_aggregate)
        }
        ast::Expression::Binary { left, right, .. }
        | ast::Expression::IndexAccess {
            base: left,
            index: right,
        } => contains_aggregate(left) || contains_aggregate(right),
        ast::Expression::Unary { operand, .. } => contains_aggregate(operand),
        ast::Expression::MapAccess { base, .. } => contains_aggregate(base),
        ast::Expression::SliceAccess { base, start, end } => {
            contains_aggregate(base)
                || start.as_deref().is_some_and(contains_aggregate)
                || end.as_deref().is_some_and(contains_aggregate)
        }
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
        ast::Expression::Map(entries) => entries.iter().any(|(_, value)| contains_aggregate(value)),
        ast::Expression::ListComprehension {
            list_expr,
            filter_expr,
            map_expr,
            ..
        } => {
            contains_aggregate(list_expr)
                || filter_expr.as_deref().is_some_and(contains_aggregate)
                || contains_aggregate(map_expr)
        }
        ast::Expression::ListPredicate {
            list_expr,
            predicate,
            ..
        } => contains_aggregate(list_expr) || contains_aggregate(predicate),
        ast::Expression::Reduce {
            initial,
            list,
            expression,
            ..
        } => {
            contains_aggregate(initial)
                || contains_aggregate(list)
                || contains_aggregate(expression)
        }
        ast::Expression::LetIn { bindings, body } => {
            bindings.iter().any(|(_, value)| contains_aggregate(value)) || contains_aggregate(body)
        }
        ast::Expression::Literal(_)
        | ast::Expression::Variable(_)
        | ast::Expression::Parameter(_)
        | ast::Expression::PropertyAccess { .. }
        | ast::Expression::ExistsSubquery { .. }
        | ast::Expression::CountSubquery { .. }
        | ast::Expression::ValueSubquery { .. } => false,
    }
}
