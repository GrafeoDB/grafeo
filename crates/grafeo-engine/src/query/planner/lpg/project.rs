//! Projection, RETURN, sort, limit, and skip planning.

use super::{
    Arc, Error, FilterExpression, GraphStoreSearch, HashMap, LimitOp, LogicalExpression,
    LogicalOperator, LogicalType, NullOrder, Operator, PhysicalSortKey, ProjectExpr,
    ProjectOperator, Result, ReturnOp, SkipOp, SortDirection, SortOp, SortOperator, SortOrder,
    common, expression_to_string, value_to_logical_type,
};

impl super::Planner {
    /// Plans a RETURN clause.
    pub(super) fn plan_return(&self, ret: &ReturnOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Handle Empty input (standalone RETURN like: RETURN 2 * 3 AS product)
        let (input_op, input_columns): (Box<dyn Operator>, Vec<String>) =
            if matches!(ret.input.as_ref(), LogicalOperator::Empty) {
                let single_row_op: Box<dyn Operator> = Box::new(
                    grafeo_core::execution::operators::single_row::SingleRowOperator::new(),
                );
                (single_row_op, Vec::new())
            } else {
                self.plan_operator(&ret.input)?
            };

        self.plan_return_with_input(ret, input_op, input_columns)
    }

    /// Plans a RETURN operator with an already-planned input operator.
    /// This is used by `plan_sort` when ORDER BY needs pre-Return property projections.
    pub(super) fn plan_return_with_input(
        &self,
        ret: &ReturnOp,
        input_op: Box<dyn Operator>,
        input_columns: Vec<String>,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (operator, columns) = self.plan_return_projection(ret, input_op, input_columns)?;

        // Apply DISTINCT if requested
        if ret.distinct {
            let schema = vec![LogicalType::Any; columns.len()];
            Ok(common::build_distinct(operator, columns, None, schema))
        } else {
            Ok((operator, columns))
        }
    }

    /// Plans the projection part of a RETURN clause (without DISTINCT).
    fn plan_return_projection(
        &self,
        ret: &ReturnOp,
        input_op: Box<dyn Operator>,
        input_columns: Vec<String>,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Expand RETURN * wildcard: replace with all user-visible input columns
        let expanded_items;
        let items = if ret.items.len() == 1
            && matches!(&ret.items[0].expression, LogicalExpression::Variable(n) if n == "*")
        {
            expanded_items = input_columns
                .iter()
                .filter(|col| !col.starts_with('_')) // Skip internal columns
                .map(|col| crate::query::plan::ReturnItem {
                    expression: LogicalExpression::Variable(col.clone()),
                    alias: None,
                })
                .collect::<Vec<_>>();
            &expanded_items
        } else {
            &ret.items
        };

        // Build variable to column index mapping
        let variable_columns: HashMap<String, usize> = input_columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        // Extract column names from return items
        let columns: Vec<String> = items
            .iter()
            .map(|item| {
                item.alias.clone().unwrap_or_else(|| {
                    // Generate a default name from the expression
                    expression_to_string(&item.expression)
                })
            })
            .collect();

        // Check if we need a project operator (for property access or expression evaluation)
        let needs_project = items
            .iter()
            .any(|item| !matches!(&item.expression, LogicalExpression::Variable(_)));

        if needs_project {
            // Build project expressions
            let mut projections = Vec::with_capacity(items.len());
            let mut output_types = Vec::with_capacity(items.len());

            for item in items {
                match &item.expression {
                    LogicalExpression::Variable(name) => {
                        let col_idx = *variable_columns.get(name).ok_or_else(|| {
                            Error::Internal(format!("Variable '{}' not found in input", name))
                        })?;
                        // Path detail variables and UNWIND/FOR scalar variables pass through as-is
                        if name.starts_with("_path_nodes_")
                            || name.starts_with("_path_edges_")
                            || name.starts_with("_path_length_")
                            || self.scalar_columns.borrow().contains(name)
                        {
                            projections.push(ProjectExpr::Column(col_idx));
                            output_types.push(LogicalType::Any);
                        } else if self.edge_columns.borrow().contains(name) {
                            projections.push(ProjectExpr::EdgeResolve { column: col_idx });
                            output_types.push(LogicalType::Any);
                        } else {
                            projections.push(ProjectExpr::NodeResolve { column: col_idx });
                            output_types.push(LogicalType::Any);
                        }
                    }
                    LogicalExpression::Property { variable, property } => {
                        let col_idx = *variable_columns.get(variable).ok_or_else(|| {
                            Error::Internal(format!("Variable '{}' not found in input", variable))
                        })?;
                        projections.push(ProjectExpr::PropertyAccess {
                            column: col_idx,
                            property: property.clone(),
                        });
                        // Property could be any type - use Any/Generic to preserve type
                        output_types.push(LogicalType::Any);
                    }
                    LogicalExpression::Literal(value) => {
                        projections.push(ProjectExpr::Constant(value.clone()));
                        output_types.push(value_to_logical_type(value));
                    }
                    LogicalExpression::FunctionCall { name, args, .. } => {
                        // Handle built-in functions
                        match name.to_lowercase().as_str() {
                            "type" => {
                                // type(r) returns the edge type string
                                if args.len() != 1 {
                                    return Err(Error::Internal(
                                        "type() requires exactly one argument".to_string(),
                                    ));
                                }
                                if let LogicalExpression::Variable(var_name) = &args[0] {
                                    let col_idx =
                                        *variable_columns.get(var_name).ok_or_else(|| {
                                            Error::Internal(format!(
                                                "Variable '{}' not found in input",
                                                var_name
                                            ))
                                        })?;
                                    projections.push(ProjectExpr::EdgeType { column: col_idx });
                                    output_types.push(LogicalType::String);
                                } else {
                                    return Err(Error::Internal(
                                        "type() argument must be a variable".to_string(),
                                    ));
                                }
                            }
                            "length" => {
                                // length(p) returns the path length for path variables,
                                // or delegates to the expression evaluator for other
                                // arguments (e.g. length(a.name) on strings/lists).
                                if args.len() != 1 {
                                    return Err(Error::Internal(
                                        "length() requires exactly one argument".to_string(),
                                    ));
                                }
                                if let LogicalExpression::Variable(var_name) = &args[0] {
                                    // Try direct column first, then path detail column
                                    let path_col = format!("_path_length_{var_name}");
                                    let col_idx = variable_columns
                                        .get(&path_col)
                                        .or_else(|| variable_columns.get(var_name))
                                        .ok_or_else(|| {
                                            Error::Internal(format!(
                                                "Variable '{}' not found in input",
                                                var_name
                                            ))
                                        })?;
                                    projections.push(ProjectExpr::Column(*col_idx));
                                    output_types.push(LogicalType::Int64);
                                } else {
                                    // Non-variable argument (e.g. property access):
                                    // fall through to expression evaluation
                                    let filter_expr = self.convert_expression(&item.expression)?;
                                    projections.push(ProjectExpr::Expression {
                                        expr: filter_expr,
                                        variable_columns: variable_columns.clone(),
                                    });
                                    output_types.push(LogicalType::Any);
                                }
                            }
                            "nodes" | "edges" | "relationships" => {
                                // nodes(p) / edges(p) / relationships(p) returns path components
                                let func_name = name.to_lowercase();
                                if args.len() != 1 {
                                    return Err(Error::Internal(format!(
                                        "{}() requires exactly one argument",
                                        name
                                    )));
                                }
                                if let LogicalExpression::Variable(var_name) = &args[0] {
                                    // Map to internal column name
                                    let suffix = if func_name == "nodes" {
                                        "nodes"
                                    } else {
                                        "edges"
                                    };
                                    let path_col = format!("_path_{suffix}_{var_name}");
                                    let col_idx = variable_columns
                                        .get(&path_col)
                                        .or_else(|| variable_columns.get(var_name))
                                        .ok_or_else(|| {
                                            Error::Internal(format!(
                                                "Variable '{var_name}' not found in input",
                                            ))
                                        })?;
                                    projections.push(ProjectExpr::Column(*col_idx));
                                    output_types.push(LogicalType::Any);
                                } else {
                                    return Err(Error::Internal(format!(
                                        "{}() argument must be a variable",
                                        name
                                    )));
                                }
                            }
                            // For other functions (head, tail, size, etc.), use expression evaluation.
                            // As a special case, if this is a vector/text score function and the
                            // scan already projected a score column, reference that column directly
                            // instead of recomputing the distance/score.
                            _ => {
                                if let Some(score_col) =
                                    self.find_projected_score(&item.expression, &input_columns)
                                {
                                    let col_idx =
                                        *variable_columns.get(&score_col).ok_or_else(|| {
                                            Error::Internal(format!(
                                                "Score column '{}' not found in input",
                                                score_col
                                            ))
                                        })?;
                                    projections.push(ProjectExpr::Column(col_idx));
                                    output_types.push(LogicalType::Any);
                                } else {
                                    let filter_expr = self.convert_expression(&item.expression)?;
                                    projections.push(ProjectExpr::Expression {
                                        expr: filter_expr,
                                        variable_columns: variable_columns.clone(),
                                    });
                                    output_types.push(LogicalType::Any);
                                }
                            }
                        }
                    }
                    LogicalExpression::Case { .. } => {
                        // Convert CASE expression to FilterExpression for evaluation
                        let filter_expr = self.convert_expression(&item.expression)?;
                        projections.push(ProjectExpr::Expression {
                            expr: filter_expr,
                            variable_columns: variable_columns.clone(),
                        });
                        // CASE can return any type - use Any
                        output_types.push(LogicalType::Any);
                    }
                    LogicalExpression::Binary { .. }
                    | LogicalExpression::Unary { .. }
                    | LogicalExpression::List(_)
                    | LogicalExpression::Map(_)
                    | LogicalExpression::IndexAccess { .. }
                    | LogicalExpression::SliceAccess { .. }
                    | LogicalExpression::CountSubquery(_)
                    | LogicalExpression::ValueSubquery(_)
                    | LogicalExpression::MapProjection { .. }
                    | LogicalExpression::Reduce { .. }
                    | LogicalExpression::PatternComprehension { .. }
                    | LogicalExpression::ListComprehension { .. }
                    | LogicalExpression::ListPredicate { .. }
                    | LogicalExpression::ExistsSubquery(_) => {
                        // Convert complex expressions to FilterExpression for evaluation
                        let filter_expr = self.convert_expression(&item.expression)?;
                        projections.push(ProjectExpr::Expression {
                            expr: filter_expr,
                            variable_columns: variable_columns.clone(),
                        });
                        output_types.push(LogicalType::Any);
                    }
                    _ => {
                        return Err(Error::Internal(format!(
                            "Unsupported RETURN expression: {:?}",
                            item.expression
                        )));
                    }
                }
            }

            let operator = Box::new(
                ProjectOperator::with_store(
                    input_op,
                    projections,
                    output_types,
                    Arc::clone(&self.store) as Arc<dyn GraphStoreSearch>,
                )
                .with_transaction_context(self.viewing_epoch, self.transaction_id)
                .with_session_context(self.session_context.clone()),
            );

            // RETURN materializes all outputs (PropertyAccess, NodeResolve,
            // expressions, etc.). Register them as scalar so enclosing Apply
            // operators do not misinterpret them as raw node/edge IDs.
            for col in &columns {
                self.scalar_columns.borrow_mut().insert(col.clone());
            }

            Ok((operator, columns))
        } else {
            // Simple case: all return items are bare variables
            // Emit resolve variants for entity variables
            let mut projections = Vec::with_capacity(items.len());
            let mut output_types = Vec::with_capacity(items.len());

            for item in items {
                if let LogicalExpression::Variable(name) = &item.expression {
                    let col_idx = *variable_columns.get(name).ok_or_else(|| {
                        Error::Internal(format!("Variable '{}' not found in input", name))
                    })?;
                    if self.scalar_columns.borrow().contains(name) {
                        projections.push(ProjectExpr::Column(col_idx));
                        output_types.push(LogicalType::Any);
                    } else if self.edge_columns.borrow().contains(name) {
                        projections.push(ProjectExpr::EdgeResolve { column: col_idx });
                        output_types.push(LogicalType::Any);
                    } else {
                        projections.push(ProjectExpr::NodeResolve { column: col_idx });
                        output_types.push(LogicalType::Any);
                    }
                }
            }

            // RETURN materializes all outputs; register as scalar.
            for col in &columns {
                self.scalar_columns.borrow_mut().insert(col.clone());
            }

            // Skip ProjectOperator only when all projections are plain Column pass-throughs
            // (i.e., only scalar variables with no reordering). NodeResolve/EdgeResolve
            // always require a ProjectOperator with store access.
            if projections.len() == input_columns.len()
                && projections
                    .iter()
                    .enumerate()
                    .all(|(i, p)| matches!(p, ProjectExpr::Column(c) if *c == i))
            {
                // No reordering or resolution needed
                Ok((input_op, columns))
            } else {
                let operator = Box::new(
                    ProjectOperator::with_store(
                        input_op,
                        projections,
                        output_types,
                        Arc::clone(&self.store) as Arc<dyn GraphStoreSearch>,
                    )
                    .with_transaction_context(self.viewing_epoch, self.transaction_id)
                    .with_session_context(self.session_context.clone()),
                );
                Ok((operator, columns))
            }
        }
    }

    /// Plans a project operator (for WITH clause).
    pub(super) fn plan_project(
        &self,
        project: &crate::query::plan::ProjectOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Handle Empty input specially (standalone WITH like: WITH [1,2,3] AS nums)
        let (input_op, input_columns): (Box<dyn Operator>, Vec<String>) =
            if matches!(project.input.as_ref(), LogicalOperator::Empty) {
                // Create a single-row operator for projecting literals
                let single_row_op: Box<dyn Operator> = Box::new(
                    grafeo_core::execution::operators::single_row::SingleRowOperator::new(),
                );
                (single_row_op, Vec::new())
            } else {
                self.plan_operator(&project.input)?
            };

        // Build variable to column index mapping
        let variable_columns: HashMap<String, usize> = input_columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        // Build projections and new column names
        let capacity = if project.pass_through_input {
            input_columns.len() + project.projections.len()
        } else {
            project.projections.len()
        };
        let mut projections = Vec::with_capacity(capacity);
        let mut output_types = Vec::with_capacity(capacity);
        let mut output_columns = Vec::with_capacity(capacity);

        // When pass_through_input is set (e.g. LET clause), first pass through
        // all existing input columns so they remain accessible to downstream
        // operators. The explicit projections are then appended as new columns.
        if project.pass_through_input {
            for (idx, col_name) in input_columns.iter().enumerate() {
                projections.push(ProjectExpr::Column(idx));
                output_types.push(LogicalType::Any);
                output_columns.push(col_name.clone());
            }
        }

        for projection in &project.projections {
            // Determine the output column name (alias or expression string)
            let col_name = projection
                .alias
                .clone()
                .unwrap_or_else(|| expression_to_string(&projection.expression));

            match &projection.expression {
                LogicalExpression::Variable(name) => {
                    let col_idx = *variable_columns.get(name).ok_or_else(|| {
                        Error::Internal(format!("Variable '{}' not found in input", name))
                    })?;
                    projections.push(ProjectExpr::Column(col_idx));
                    // Use Any for scalar variables so string/numeric values
                    // are not coerced to NodeId by the typed vector push.
                    if self.scalar_columns.borrow().contains(name) {
                        output_types.push(LogicalType::Any);
                        self.scalar_columns.borrow_mut().insert(col_name.clone());
                    } else if self.edge_columns.borrow().contains(name) {
                        output_types.push(LogicalType::Edge);
                        self.edge_columns.borrow_mut().insert(col_name.clone());
                    } else {
                        output_types.push(LogicalType::Node);
                    }
                }
                LogicalExpression::Property { variable, property } => {
                    let col_idx = *variable_columns.get(variable).ok_or_else(|| {
                        Error::Internal(format!("Variable '{}' not found in input", variable))
                    })?;
                    projections.push(ProjectExpr::PropertyAccess {
                        column: col_idx,
                        property: property.clone(),
                    });
                    output_types.push(LogicalType::Any);
                    // Property access produces a scalar value
                    self.scalar_columns.borrow_mut().insert(col_name.clone());
                }
                LogicalExpression::Literal(value) => {
                    projections.push(ProjectExpr::Constant(value.clone()));
                    output_types.push(value_to_logical_type(value));
                    // Literals are scalar values
                    self.scalar_columns.borrow_mut().insert(col_name.clone());
                }
                _ => {
                    // For complex expressions, use full expression evaluation
                    let filter_expr = self.convert_expression(&projection.expression)?;
                    projections.push(ProjectExpr::Expression {
                        expr: filter_expr,
                        variable_columns: variable_columns.clone(),
                    });
                    output_types.push(LogicalType::Any);
                    // Expression results are scalar values
                    self.scalar_columns.borrow_mut().insert(col_name.clone());
                }
            }

            output_columns.push(col_name);
        }

        let operator = Box::new(
            ProjectOperator::with_store(
                input_op,
                projections,
                output_types,
                Arc::clone(&self.store) as Arc<dyn GraphStoreSearch>,
            )
            .with_transaction_context(self.viewing_epoch, self.transaction_id)
            .with_session_context(self.session_context.clone()),
        );

        Ok((operator, output_columns))
    }

    /// Plans a LIMIT operator.
    pub(super) fn plan_limit(&self, limit: &LimitOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Top-K optimization: Limit(k) -> Sort(score_fn) -> NodeScan
        // can be rewritten to VectorScan(k) or TextScan(k). GQL emits the plan as
        // Limit-above-Sort, so the check belongs here rather than in plan_sort.
        //
        // The rewrite is disabled under PROFILE because it fuses three logical
        // operators into one physical operator without updating the logical tree.
        // PROFILE walks the logical tree to build its output and would panic on
        // the entry-count mismatch (see `try_topk_rewrite` docstring).
        if !self.profiling.get()
            && let LogicalOperator::Sort(sort) = limit.input.as_ref()
            && let Some(result) = self.try_topk_rewrite(sort, &limit.count)?
        {
            return Ok(result);
        }

        // Phase 4e LIMIT pushdown: when this LIMIT sits directly above a
        // Filter, push the count as a hint into leaf scan operators
        // (currently `RangeScanOperator`). The outer `LimitOperator`
        // wrapper still enforces correctness; the hint is purely an
        // optimization that bounds the inner materialization step.
        //
        // Only fires when the input is `Filter` directly: any
        // intermediate operator (Sort, Skip, Project, Distinct) needs
        // full materialization and would produce wrong results if its
        // child were truncated. Save-and-restore via `Cell::replace` so
        // nested LIMITs don't leak hints across scopes.
        let saved_hint = if matches!(limit.input.as_ref(), LogicalOperator::Filter(_)) {
            Some(self.limit_hint.replace(Some(limit.count.value())))
        } else {
            None
        };

        let plan_result = self.plan_operator(&limit.input);

        if let Some(prev) = saved_hint {
            self.limit_hint.set(prev);
        }

        let (input_op, columns) = plan_result?;
        let schema = self.derive_schema_from_columns(&columns);
        Ok(crate::query::planner::common::build_limit(
            input_op,
            columns,
            limit.count.value(),
            schema,
        ))
    }

    /// Plans a SKIP operator.
    pub(super) fn plan_skip(&self, skip: &SkipOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, columns) = self.plan_operator(&skip.input)?;
        let schema = self.derive_schema_from_columns(&columns);
        Ok(crate::query::planner::common::build_skip(
            input_op,
            columns,
            skip.count.value(),
            schema,
        ))
    }

    /// Plans a SORT (ORDER BY) operator.
    ///
    /// When Sort wraps a Return (e.g. `RETURN p.name ORDER BY p.age`), ORDER BY
    /// may reference entity variables that Return has already projected away. In
    /// that case we inject a property projection BEFORE the Return so the sort
    /// key is available in the output columns.
    pub(super) fn plan_sort(&self, sort: &SortOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Top-K rewrite is wired from plan_limit (the actual plan shape is
        // Limit-above-Sort), so plan_sort only handles the standard path.

        // Collect variable references from an expression tree (e.g., `n` in `labels(n)[0]`).
        fn collect_vars(expr: &LogicalExpression, out: &mut Vec<String>) {
            match expr {
                LogicalExpression::Variable(v)
                | LogicalExpression::Property { variable: v, .. }
                | LogicalExpression::Labels(v)
                | LogicalExpression::Type(v)
                | LogicalExpression::Id(v) => out.push(v.clone()),
                LogicalExpression::FunctionCall { args, .. } => {
                    for a in args {
                        collect_vars(a, out);
                    }
                }
                LogicalExpression::IndexAccess { base, .. } => collect_vars(base, out),
                LogicalExpression::Binary { left, right, .. } => {
                    collect_vars(left, out);
                    collect_vars(right, out);
                }
                LogicalExpression::Unary { operand, .. } => collect_vars(operand, out),
                _ => {}
            }
        }

        // Check if we need pre-Return property/entity projections. This is
        // necessary when ORDER BY references a variable (e.g. p.age, labels(n))
        // that is not included in the RETURN clause.
        let needs_pre_return = if let LogicalOperator::Return(ret) = sort.input.as_ref() {
            sort.keys.iter().any(|key| {
                let mut vars = Vec::new();
                collect_vars(&key.expression, &mut vars);
                vars.iter().any(|variable| {
                    // Check if the Return items produce a column matching this variable
                    !ret.items.iter().any(|item| {
                        item.alias.as_deref() == Some(variable)
                            || matches!(
                                &item.expression,
                                LogicalExpression::Variable(v) if v == variable
                            )
                    })
                })
            })
        } else {
            false
        };

        // Number of extra sort-key columns appended after Return items
        let mut sort_extra_count: usize = 0;

        let (mut input_op, input_columns) = if needs_pre_return {
            // Plan the Return's input, then build a combined projection that
            // outputs both RETURN items and ORDER BY sort-key properties.
            let LogicalOperator::Return(ret) = sort.input.as_ref() else {
                unreachable!()
            };
            let (inner_op, inner_columns) = self.plan_operator(&ret.input)?;
            let inner_vars: HashMap<String, usize> = inner_columns
                .iter()
                .enumerate()
                .map(|(i, n)| (n.clone(), i))
                .collect();

            // Build augmented Return items: original items plus ORDER BY
            // expressions that reference variables available in the Match but
            // not in the Return. This includes both property accesses and
            // complex expressions (labels(n)[0], type(r), etc.).
            let mut augmented_items = ret.items.clone();
            let mut extra_columns = Vec::new();
            let mut seen = std::collections::HashSet::new();
            for key in &sort.keys {
                match &key.expression {
                    LogicalExpression::Variable(variable) => {
                        // ORDER BY referencing a WITH-clause alias that the
                        // outer RETURN drops: e.g. `WITH x AS s ... RETURN x
                        // ORDER BY s`. The Variable is in inner_vars (Return's
                        // input columns) but not in ret.items, so without a
                        // passthrough Sort sees a missing column at runtime.
                        if !inner_vars.contains_key(variable) {
                            continue;
                        }
                        let already_in_return = ret.items.iter().any(|item| {
                            item.alias.as_deref() == Some(variable.as_str())
                                || matches!(
                                    &item.expression,
                                    LogicalExpression::Variable(v) if v == variable
                                )
                        });
                        if already_in_return {
                            continue;
                        }
                        if seen.insert(variable.clone()) {
                            augmented_items.push(crate::query::plan::ReturnItem {
                                expression: key.expression.clone(),
                                alias: Some(variable.clone()),
                            });
                            extra_columns.push(variable.clone());
                        }
                    }
                    LogicalExpression::Property { variable, property } => {
                        if !inner_vars.contains_key(variable) {
                            continue;
                        }
                        // Skip if the Return already materializes this exact
                        // property access (possibly under an alias). E.g.
                        // RETURN caller.name AS caller ORDER BY caller.name
                        // already has caller.name in the Return items.
                        let already_in_return = ret.items.iter().any(|item| {
                            matches!(
                                &item.expression,
                                LogicalExpression::Property {
                                    variable: v,
                                    property: p,
                                } if v == variable && p == property
                            )
                        });
                        if already_in_return {
                            continue;
                        }
                        let col_name = format!("{}_{}", variable, property);
                        if seen.insert(col_name.clone()) {
                            augmented_items.push(crate::query::plan::ReturnItem {
                                expression: key.expression.clone(),
                                alias: Some(col_name.clone()),
                            });
                            extra_columns.push(col_name);
                        }
                    }
                    expr => {
                        let col_name = format!("__expr_{expr:?}");
                        if seen.insert(col_name.clone()) {
                            augmented_items.push(crate::query::plan::ReturnItem {
                                expression: expr.clone(),
                                alias: Some(col_name.clone()),
                            });
                            extra_columns.push(col_name);
                        }
                    }
                }
            }

            let augmented_ret = crate::query::plan::ReturnOp {
                items: augmented_items,
                distinct: ret.distinct,
                input: ret.input.clone(),
            };

            // Plan the augmented Return with the original inner operator.
            // Route through `maybe_profile` so that under PROFILE, this fused
            // physical op contributes a ProfileEntry attributed to the
            // logical `Return` node — `build_profile_tree` walks the logical
            // tree and otherwise panics with a count mismatch at the
            // ancestor (typically Limit) once entries run out.
            let (op, columns) = self.maybe_profile(
                self.plan_return_with_input(&augmented_ret, inner_op, inner_columns),
                sort.input.as_ref(),
            )?;
            sort_extra_count = extra_columns.len();
            (op, columns)
        } else {
            self.plan_operator(&sort.input)?
        };

        // Build variable to column index mapping
        let mut variable_columns: HashMap<String, usize> = input_columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        // When the sort input is a Return, some sort key expressions may
        // already be computed by the Return under an alias. For example,
        // RETURN caller.name AS caller ORDER BY caller.name: the property
        // access is already materialized in column "caller". Register the
        // sort-style name ("caller_name") so the loop below resolves to the
        // existing column instead of adding a broken extra PropertyAccess
        // on a non-entity column.
        if let LogicalOperator::Return(ret) = sort.input.as_ref() {
            for item in &ret.items {
                if let LogicalExpression::Property { variable, property } = &item.expression {
                    let sort_col_name = format!("{variable}_{property}");
                    if !variable_columns.contains_key(&sort_col_name) {
                        let output_name = item
                            .alias
                            .clone()
                            .unwrap_or_else(|| expression_to_string(&item.expression));
                        if let Some(&col_idx) = variable_columns.get(&output_name) {
                            variable_columns.insert(sort_col_name, col_idx);
                        }
                    }
                }
            }
        }

        // Collect extra projections in a single ordered list so that column
        // index assignment matches the order they are added to the ProjectOperator.
        enum SortExtraProjection {
            Property {
                variable: String,
                property: String,
                col_name: String,
            },
            Expression {
                filter_expr: FilterExpression,
                col_name: String,
            },
        }
        let mut extra_projections: Vec<SortExtraProjection> = Vec::new();
        let mut next_col_idx = input_columns.len();
        let mut expr_extra_count: usize = 0;

        for key in &sort.keys {
            match &key.expression {
                LogicalExpression::Property { variable, property } => {
                    let col_name = format!("{}_{}", variable, property);
                    if !variable_columns.contains_key(&col_name) {
                        extra_projections.push(SortExtraProjection::Property {
                            variable: variable.clone(),
                            property: property.clone(),
                            col_name: col_name.clone(),
                        });
                        variable_columns.insert(col_name, next_col_idx);
                        next_col_idx += 1;
                    }
                }
                LogicalExpression::Variable(_) => {
                    // Already in variable_columns
                }
                _ => {
                    // Complex expression (Labels, Type, FunctionCall, IndexAccess, etc.)
                    // If this is a vector/text score function and the scan already projected
                    // a score column, register a direct alias so resolve_sort_expression
                    // picks it up without injecting a new projection.
                    if let Some(score_col) =
                        self.find_projected_score(&key.expression, &input_columns)
                    {
                        let col_name = format!("__expr_{:?}", key.expression);
                        if !variable_columns.contains_key(&col_name)
                            && let Some(&existing_idx) = variable_columns.get(&score_col)
                        {
                            variable_columns.insert(col_name, existing_idx);
                        }
                    } else {
                        let col_name = format!("__expr_{:?}", key.expression);
                        if !variable_columns.contains_key(&col_name) {
                            let filter_expr = self.convert_expression(&key.expression)?;
                            extra_projections.push(SortExtraProjection::Expression {
                                filter_expr,
                                col_name: col_name.clone(),
                            });
                            variable_columns.insert(col_name, next_col_idx);
                            next_col_idx += 1;
                            expr_extra_count += 1;
                        }
                    }
                }
            }
        }

        // Track output columns
        let mut output_columns = input_columns.clone();

        // If we have extra projections, add a projection to materialize them
        if !extra_projections.is_empty() {
            let mut projections = Vec::new();
            let mut output_types = Vec::new();

            // Pass through existing columns with correct types. Using Any
            // (Generic vectors) preserves all value kinds: strings, maps,
            // node IDs, edge IDs. PropertyAccess handles Generic vectors
            // via get_node_id()/get_edge_id() fallback paths.
            let pass_through_types = self.derive_schema_from_columns(&input_columns);
            for (i, _) in input_columns.iter().enumerate() {
                projections.push(ProjectExpr::Column(i));
                output_types.push(pass_through_types[i].clone());
            }

            // Add extra projections in the same order as index assignment
            for proj in &extra_projections {
                match proj {
                    SortExtraProjection::Property {
                        variable,
                        property,
                        col_name,
                    } => {
                        let source_col = *variable_columns.get(variable).ok_or_else(|| {
                            Error::Internal(format!(
                                "Variable '{}' not found for ORDER BY property projection",
                                variable
                            ))
                        })?;
                        projections.push(ProjectExpr::PropertyAccess {
                            column: source_col,
                            property: property.clone(),
                        });
                        output_types.push(LogicalType::Any);
                        output_columns.push(col_name.clone());
                    }
                    SortExtraProjection::Expression {
                        filter_expr,
                        col_name,
                    } => {
                        projections.push(ProjectExpr::Expression {
                            expr: filter_expr.clone(),
                            variable_columns: variable_columns.clone(),
                        });
                        output_types.push(LogicalType::Any);
                        output_columns.push(col_name.clone());
                    }
                }
            }

            input_op = Box::new(
                ProjectOperator::with_store(
                    input_op,
                    projections,
                    output_types,
                    Arc::clone(&self.store) as Arc<dyn GraphStoreSearch>,
                )
                .with_transaction_context(self.viewing_epoch, self.transaction_id)
                .with_session_context(self.session_context.clone()),
            );
        }

        // Convert logical sort keys to physical sort keys
        let physical_keys: Vec<PhysicalSortKey> = sort
            .keys
            .iter()
            .map(|key| {
                let col_idx = self
                    .resolve_sort_expression_with_properties(&key.expression, &variable_columns)?;
                Ok(PhysicalSortKey {
                    column: col_idx,
                    direction: match key.order {
                        SortOrder::Ascending => SortDirection::Ascending,
                        SortOrder::Descending => SortDirection::Descending,
                    },
                    null_order: match key.nulls {
                        Some(crate::query::plan::NullsOrdering::First) => NullOrder::NullsFirst,
                        Some(crate::query::plan::NullsOrdering::Last) => NullOrder::NullsLast,
                        None => NullOrder::NullsLast, // default
                    },
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let output_schema = self.derive_schema_from_columns(&output_columns);
        let mut operator: Box<dyn Operator> =
            Box::new(SortOperator::new(input_op, physical_keys, output_schema));

        // Strip extra columns injected for ORDER BY resolution: both pre-Return
        // property projections (sort_extra_count) and synthetic __expr_ columns
        // for complex expressions like labels(n)[0] or type(r).
        let total_extra = sort_extra_count + expr_extra_count;
        if total_extra > 0 {
            let keep_count = output_columns.len() - total_extra;
            let strip_projections: Vec<ProjectExpr> =
                (0..keep_count).map(ProjectExpr::Column).collect();
            let strip_types: Vec<LogicalType> = (0..keep_count).map(|_| LogicalType::Any).collect();
            operator = Box::new(ProjectOperator::new(
                operator,
                strip_projections,
                strip_types,
            ));
            output_columns.truncate(keep_count);
        }

        Ok((operator, output_columns))
    }

    /// Resolves a sort expression to a column index, using projected property columns.
    pub(super) fn resolve_sort_expression_with_properties(
        &self,
        expr: &LogicalExpression,
        variable_columns: &HashMap<String, usize>,
    ) -> Result<usize> {
        crate::query::planner::common::resolve_expression_to_column(
            expr,
            variable_columns,
            " for ORDER BY",
        )
    }

    /// Derives a schema from column names using the planner's type tracking.
    ///
    /// Defaults to `Any` (safe for all value types: scalars, maps, property
    /// projections, etc.). Columns explicitly tracked in `edge_columns` get
    /// `Edge` for compact `Vec<EdgeId>` storage. Mutation operators that add
    /// new entity-ID columns (CREATE, MERGE) should append `Node`/`Edge`
    /// explicitly after calling this for pass-through columns.
    pub(super) fn derive_schema_from_columns(&self, columns: &[String]) -> Vec<LogicalType> {
        let edges = self.edge_columns.borrow();
        columns
            .iter()
            .map(|name| {
                if edges.contains(name) {
                    LogicalType::Edge
                } else {
                    LogicalType::Any
                }
            })
            .collect()
    }

    /// Attempts to rewrite Limit-above-Sort on a vector/text scoring function
    /// into a direct index scan that returns results in order.
    ///
    /// Called from `plan_limit` when its input is a `Sort`. The expected plan
    /// shape is `Limit(k) -> Sort(score_fn DESC) -> NodeScan(label)`, optionally
    /// with a `Return(var)` between Sort and NodeScan (`find_node_scan` walks
    /// through it). The sort key must be the score function call directly: an
    /// alias like `... AS rank` followed by `ORDER BY rank` is not currently
    /// resolved, so the rewrite is skipped in that case.
    ///
    /// Caveats:
    ///
    /// - **Physical-only**. The logical plan tree is unchanged, so EXPLAIN still
    ///   prints `Limit/Sort/Return/NodeScan`. PROFILE walks the logical tree and
    ///   would mismatch the (smaller) physical entry count; the caller in
    ///   `plan_limit` skips this rewrite when `self.profiling` is set.
    /// - **Bare scan only**. The rewrite produces a physical operator whose
    ///   columns are `[NodeId, score]`. If the original tree had a `Return` or
    ///   `Project` between Sort and NodeScan (which GQL emits for any RETURN
    ///   clause), substituting the rewrite drops that projection — output rows
    ///   would contain raw `Int64` NodeIds instead of resolved nodes / mapped
    ///   columns. To stay correct we therefore require `sort.input` to be a bare
    ///   NodeScan with no wrapping. In practice that means the rewrite never
    ///   fires from GQL today; it remains in place for future planner shapes
    ///   (e.g. an internal rewrite that hoists the projection above Limit).
    ///
    /// Both caveats disappear if the rewrite is lifted into the logical
    /// optimization phase so the two trees stay in sync.
    fn try_topk_rewrite(
        &self,
        sort: &SortOp,
        count: &crate::query::plan::CountExpr,
    ) -> Result<Option<(Box<dyn Operator>, Vec<String>)>> {
        if sort.keys.len() != 1 {
            return Ok(None);
        }
        let sort_key = &sort.keys[0];

        // All remaining work is index-specific; skip when no index feature is compiled in.
        #[cfg(any(feature = "vector-index", feature = "text-index"))]
        {
            let crate::query::plan::CountExpr::Literal(k) = count else {
                return Ok(None);
            };
            let k = *k;

            // Bare-scan-only guard (see docstring): refuse to fire when there
            // is any wrapping between Sort and NodeScan, since the rewrite
            // would silently drop it.
            let LogicalOperator::NodeScan(scan) = sort.input.as_ref() else {
                return Ok(None);
            };
            let scan_var = scan.variable.clone();
            let Some(ref label) = scan.label else {
                return Ok(None);
            };

            match &sort_key.expression {
                LogicalExpression::FunctionCall { name, args, .. } => match name.as_str() {
                    #[cfg(feature = "vector-index")]
                    "cosine_similarity" | "euclidean_distance" | "dot_product"
                    | "manhattan_distance" => {
                        self.try_vector_topk(name, args, k, &scan_var, label, sort_key)
                    }
                    #[cfg(feature = "text-index")]
                    "text_score" => self.try_text_topk(args, k, &scan_var, label, sort_key),
                    _ => Ok(None),
                },
                _ => Ok(None),
            }
        }

        #[cfg(not(any(feature = "vector-index", feature = "text-index")))]
        {
            let _ = (sort_key, count);
            Ok(None)
        }
    }

    #[cfg(feature = "vector-index")]
    fn try_vector_topk(
        &self,
        fn_name: &str,
        args: &[LogicalExpression],
        k: usize,
        scan_var: &str,
        label: &str,
        sort_key: &crate::query::plan::SortKey,
    ) -> Result<Option<(Box<dyn Operator>, Vec<String>)>> {
        if args.len() != 2 {
            return Ok(None);
        }

        // Determine metric and expected sort direction
        let (metric, is_similarity) = match fn_name {
            "cosine_similarity" => (super::VectorMetric::Cosine, true),
            "dot_product" => (super::VectorMetric::DotProduct, true),
            "euclidean_distance" => (super::VectorMetric::Euclidean, false),
            "manhattan_distance" => (super::VectorMetric::Manhattan, false),
            _ => return Ok(None),
        };

        // Similarity needs DESC, distance needs ASC
        let expected_order = if is_similarity {
            SortOrder::Descending
        } else {
            SortOrder::Ascending
        };
        if sort_key.order != expected_order {
            return Ok(None); // Wrong direction — can't use index
        }

        // Extract property and query vector
        let (property, query_vector) =
            if let LogicalExpression::Property { variable, property } = &args[0] {
                if variable != scan_var {
                    return Ok(None);
                }
                (property.clone(), args[1].clone())
            } else if let LogicalExpression::Property { variable, property } = &args[1] {
                if variable != scan_var {
                    return Ok(None);
                }
                (property.clone(), args[0].clone())
            } else {
                return Ok(None);
            };

        // Check for vector index
        if !self.store.has_vector_index(label, &property) {
            return Ok(None);
        }

        // Top-K rewrite needs a resolvable query vector. Fall through
        // otherwise so the standard Sort + Filter path evaluates per-row.
        if self.resolve_vector_literal(&query_vector).is_err() {
            return Ok(None);
        }

        let vector_scan = super::VectorScanOp {
            variable: scan_var.to_string(),
            index_name: Some(format!("{}:{}", label, property)),
            property,
            label: Some(label.to_string()),
            query_vector,
            k: Some(k),
            metric: Some(metric),
            min_similarity: None,
            max_distance: None,
            input: None,
        };

        self.plan_operator(&LogicalOperator::VectorScan(vector_scan))
            .map(Some)
    }

    #[cfg(feature = "text-index")]
    fn try_text_topk(
        &self,
        args: &[LogicalExpression],
        k: usize,
        scan_var: &str,
        label: &str,
        sort_key: &crate::query::plan::SortKey,
    ) -> Result<Option<(Box<dyn Operator>, Vec<String>)>> {
        if args.len() != 2 {
            return Ok(None);
        }

        // text_score needs DESC order
        if sort_key.order != SortOrder::Descending {
            return Ok(None);
        }

        // First arg must be property access on the scan variable
        let LogicalExpression::Property { variable, property } = &args[0] else {
            return Ok(None);
        };
        if variable != scan_var {
            return Ok(None);
        }

        // Check for text index (required)
        if !self.store.has_text_index(label, property) {
            return Ok(None); // Silently skip — error will come from text pushdown if used
        }

        let text_scan = super::TextScanOp {
            variable: scan_var.to_string(),
            property: property.clone(),
            label: label.to_string(),
            query: args[1].clone(),
            k: Some(k),
            threshold: None,
            score_column: Some(text_score_column_name(scan_var, property, &args[1])),
        };

        self.plan_operator(&LogicalOperator::TextScan(text_scan))
            .map(Some)
    }

    /// Checks if a logical expression matches a projected score column.
    ///
    /// When `VectorScan` or `TextScan` already computed a score and projected it
    /// as `_vscore_{var}` / `_tscore_{var}`, downstream expressions that call the
    /// same function can reference the projected column instead of recomputing the
    /// distance or BM25 score.
    ///
    /// Returns the column name to reference, or `None` if no reuse is possible.
    pub(super) fn find_projected_score(
        &self,
        expr: &LogicalExpression,
        columns: &[String],
    ) -> Option<String> {
        #[cfg(not(any(feature = "vector-index", feature = "text-index")))]
        {
            let _ = (expr, columns);
            None
        }

        #[cfg(any(feature = "vector-index", feature = "text-index"))]
        {
            let LogicalExpression::FunctionCall { name, args, .. } = expr else {
                return None;
            };

            #[cfg(feature = "vector-index")]
            let is_vector_fn = matches!(
                name.as_str(),
                "cosine_similarity" | "euclidean_distance" | "dot_product" | "manhattan_distance"
            );
            #[cfg(not(feature = "vector-index"))]
            let is_vector_fn = false;

            #[cfg(feature = "text-index")]
            let is_text_fn = name == "text_score";
            #[cfg(not(feature = "text-index"))]
            let is_text_fn = false;

            if !(is_vector_fn || is_text_fn) {
                return None;
            }

            // Either arg may hold the property access (the producer
            // `try_vector_topk` accepts both orderings). Identify the
            // property and treat the *other* arg as the query expression,
            // which is embedded in the column name so different queries
            // against the same property never reuse each other's score.
            if args.len() != 2 {
                return None;
            }
            let (variable, property, query) = match (&args[0], &args[1]) {
                (LogicalExpression::Property { variable, property }, query) => {
                    (variable, property, query)
                }
                (query, LogicalExpression::Property { variable, property }) => {
                    (variable, property, query)
                }
                _ => return None,
            };

            #[cfg(feature = "vector-index")]
            if is_vector_fn {
                let metric_tag = match name.as_str() {
                    "cosine_similarity" => "cos",
                    "euclidean_distance" => "euc",
                    "dot_product" => "dot",
                    "manhattan_distance" => "man",
                    _ => "cos",
                };
                let score_col = vector_score_column_name(metric_tag, property, variable, query);
                if columns.contains(&score_col) {
                    return Some(score_col);
                }
            }

            #[cfg(feature = "text-index")]
            if is_text_fn {
                let score_col = text_score_column_name(variable, property, query);
                if columns.contains(&score_col) {
                    return Some(score_col);
                }
            }

            None
        }
    }
}

// Score-column naming. The hash of the query expression is part of the name so
// that two scoring calls with different query arguments (e.g. $q1 vs $q2) on
// the same (variable, property) never collide. Producer and consumer sites
// must go through these helpers, otherwise `find_projected_score` could hand
// back a score that was computed against a different query.
#[cfg(any(feature = "vector-index", feature = "text-index"))]
pub(super) fn score_query_hash(query: &LogicalExpression) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    format!("{:?}", query).hash(&mut h);
    h.finish()
}

#[cfg(feature = "vector-index")]
pub(super) fn vector_score_column_name(
    metric_tag: &str,
    property: &str,
    variable: &str,
    query: &LogicalExpression,
) -> String {
    format!(
        "_vscore_{}_{}_{}_{:x}",
        metric_tag,
        property,
        variable,
        score_query_hash(query)
    )
}

#[cfg(feature = "text-index")]
pub(super) fn text_score_column_name(
    variable: &str,
    property: &str,
    query: &LogicalExpression,
) -> String {
    format!(
        "_tscore_{}_{}_{:x}",
        property,
        variable,
        score_query_hash(query)
    )
}
