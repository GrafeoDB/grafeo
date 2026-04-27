//! Filter planning with zone map pre-filtering and index lookups.
//!
//! The order of the rewrite/pushdown attempts in [`Planner::plan_filter`][pf]
//! is load-bearing, not a performance tweak:
//!
//! 1. Subquery rewrites (`extract_complex_exists`, `extract_exists_from_or`,
//!    `extract_count_comparison`) must run first. The later steps assume
//!    a scalar predicate with no subquery shape; once a rewrite fires,
//!    the rest of the method is bypassed.
//! 2. Zone-map short-circuit fires before any index lookup so we can
//!    skip opening an index file at all when summary statistics prove
//!    emptiness.
//! 3. Property-index and range-index attempts come before hybrid
//!    pushdown because they are cheaper and strictly more specific.
//! 4. Compound hybrid (`try_plan_filter_compound_hybrid`) is tried
//!    before single-sided vector/text pushdown so an `AND` over both
//!    kinds doesn't get torn apart into a scan + filter.
//! 5. The generic `FilterOperator` is the last resort; whatever fell
//!    through still runs correctly, just without an index.
//!
//! Changing this order needs a correctness argument, not just a
//! benchmark.
//!
//! [pf]: super::Planner::plan_filter

use super::{
    ApplyOperator, Arc, BinaryOp, DistinctOperator, EmptyOperator, ExpressionPredicate,
    FilterExpression, FilterOp, FilterOperator, GraphStoreSearch, HashAggregateOperator,
    HashJoinOperator, HashMap, LogicalExpression, LogicalOperator, NodeListOperator, Operator,
    PhysicalAggregateExpr, PhysicalJoinType, RangeBounds, RangeScanOperator, Result, TransactionId,
    UnaryOp, UnionOperator, Value, convert_binary_op, convert_filter_expression,
};

/// Cross-type equality comparison with Int64/Float64 coercion.
fn values_equal_coerced(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int64(a), Value::Float64(b)) => (*a as f64 - b).abs() < f64::EPSILON,
        (Value::Float64(a), Value::Int64(b)) => (a - *b as f64).abs() < f64::EPSILON,
        _ => a == b,
    }
}

impl super::Planner {
    /// Plans a filter operator.
    ///
    /// Uses zone map pre-filtering to potentially skip scans when predicates
    /// definitely won't match any data. Also uses property indexes when available
    /// for O(1) lookups instead of full scans. Complex EXISTS/NOT EXISTS subqueries
    /// are rewritten as semi-joins or anti-joins.
    pub(super) fn plan_filter(
        &self,
        filter: &FilterOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Check for complex EXISTS/NOT EXISTS patterns and rewrite as semi/anti join.
        // Simple single-hop EXISTS patterns are handled by the fast path in
        // convert_expression() -> extract_exists_pattern().
        if let Some((subquery, is_negated, remaining)) =
            self.extract_complex_exists(&filter.predicate)
        {
            return self.plan_exists_as_semi_join(&filter.input, subquery, is_negated, remaining);
        }

        // Complex EXISTS inside OR predicates can't use semi-join (which filters).
        // Instead, split the OR into two branches (EXISTS via semi-join, scalar
        // via filter), union the results, and deduplicate.
        if let Some((subquery, is_negated, other_pred)) =
            self.extract_exists_from_or(&filter.predicate)
        {
            return self.plan_exists_or_as_union(&filter.input, subquery, is_negated, other_pred);
        }

        // Check for COUNT subquery comparisons and rewrite as Apply + Aggregate + Filter.
        // Handles patterns like: COUNT { MATCH ... } > 5, COUNT { ... } = 0, etc.
        if let Some((subquery, op, threshold, remaining)) =
            Self::extract_count_comparison(&filter.predicate)
        {
            return self.plan_count_as_apply(&filter.input, subquery, op, threshold, remaining);
        }

        // Check zone maps for simple property predicates before scanning
        // If zone map says "definitely no matches", we can short-circuit
        if let Some(false) = self.check_zone_map_for_predicate(&filter.predicate) {
            // Zone map says no matches possible - return empty result
            let (_, columns) = self.plan_operator(&filter.input)?;
            let schema = self.derive_schema_from_columns(&columns);
            let empty_op = Box::new(EmptyOperator::new(schema));
            return Ok((empty_op, columns));
        }

        // Try to use property index for equality predicates on indexed properties
        if let Some(result) = self.try_plan_filter_with_property_index(filter)? {
            return Ok(result);
        }

        // Try to use property index for `var.prop IN [literals]` predicates,
        // unioning per-value index lookups instead of falling back to a full
        // scan + filter. Empirically this is the difference between ~10ms
        // and multiple seconds on a few thousand reviews when an `id` index
        // exists.
        if let Some(result) = self.try_plan_filter_with_in_list_index(filter)? {
            return Ok(result);
        }

        // Try to use range optimization for range predicates (>, <, >=, <=)
        if let Some(result) = self.try_plan_filter_with_range_index(filter)? {
            return Ok(result);
        }

        // Try compound hybrid first (both vector AND/OR text in same predicate)
        #[cfg(all(feature = "vector-index", feature = "text-index"))]
        if let Some(result) = self.try_plan_filter_compound_hybrid(filter)? {
            return Ok(result);
        }

        // Try to use vector index for similarity/distance predicates
        #[cfg(feature = "vector-index")]
        if let Some(result) = self.try_plan_filter_with_vector_index(filter)? {
            return Ok(result);
        }

        // Try to use text index for text_score/text_match predicates
        #[cfg(feature = "text-index")]
        if let Some(result) = self.try_plan_filter_with_text_index(filter)? {
            return Ok(result);
        }

        // Plan the input operator first
        let (input_op, columns) = self.plan_operator(&filter.input)?;

        // Build variable to column index mapping
        let variable_columns: HashMap<String, usize> = columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        // Convert logical expression to filter expression
        let filter_expr = self.convert_expression(&filter.predicate)?;

        // Create the predicate
        let predicate = ExpressionPredicate::new(
            filter_expr,
            variable_columns,
            Arc::clone(&self.store) as Arc<dyn GraphStoreSearch>,
        )
        .with_transaction_context(self.viewing_epoch, self.transaction_id)
        .with_session_context(self.session_context.clone());

        // Create the filter operator
        let operator = Box::new(FilterOperator::new(input_op, Box::new(predicate)));

        Ok((operator, columns))
    }

    /// Extracts an EXISTS or NOT EXISTS subquery from a filter predicate for
    /// semi-join rewriting.
    ///
    /// Returns `(subquery, is_negated, remaining_predicate)`.
    ///
    /// At the top level (standalone EXISTS / NOT EXISTS), only complex patterns
    /// that cannot use the inline fast path in `extract_exists_pattern()` are
    /// extracted. Within AND trees, ALL EXISTS patterns are extracted regardless
    /// of complexity: when multiple EXISTS subqueries share a WHERE clause, every
    /// one must go through the semi-join path so the recursive handler in
    /// `plan_exists_as_semi_join` can chain them.
    ///
    /// The Cypher translator builds left-leaning AND trees from sequential WHERE
    /// predicates, so EXISTS nodes sit at the right child of AND nodes at various
    /// depths. The recursive semi-join handler (`plan_exists_as_semi_join`) calls
    /// this function again on each remaining predicate, peeling off one EXISTS
    /// per level until only scalar predicates remain.
    fn extract_complex_exists<'a>(
        &self,
        predicate: &'a LogicalExpression,
    ) -> Option<(&'a LogicalOperator, bool, Option<LogicalExpression>)> {
        match predicate {
            LogicalExpression::ExistsSubquery(subplan) => {
                // Top-level EXISTS: only use semi-join for complex patterns.
                // Simple single-hop patterns use the fast path in convert_expression().
                if self.extract_exists_pattern(subplan).is_err() {
                    Some((subplan.as_ref(), false, None))
                } else {
                    None
                }
            }
            LogicalExpression::Unary {
                op: UnaryOp::Not,
                operand,
            } => {
                if let LogicalExpression::ExistsSubquery(subplan) = operand.as_ref() {
                    if self.extract_exists_pattern(subplan).is_err() {
                        Some((subplan.as_ref(), true, None))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            LogicalExpression::Binary {
                op: BinaryOp::And,
                left,
                right,
            } => {
                // In AND context, extract ANY EXISTS (simple or complex).
                // When multiple EXISTS appear in the same WHERE, extracting the
                // first one (even if simple) lets the recursive semi-join handler
                // find and extract the remaining complex ones from the rest.
                if let Some((subplan, negated)) = Self::extract_exists_from_expr(left) {
                    return Some((subplan, negated, Some(right.as_ref().clone())));
                }
                if let Some((subplan, negated)) = Self::extract_exists_from_expr(right) {
                    return Some((subplan, negated, Some(left.as_ref().clone())));
                }
                // Recurse into left subtree (handles left-leaning AND trees where
                // EXISTS nodes are buried deeper than immediate children)
                if let Some((subplan, negated, inner_remaining)) = self.extract_complex_exists(left)
                {
                    let remaining = match inner_remaining {
                        Some(inner) => LogicalExpression::Binary {
                            left: Box::new(inner),
                            op: BinaryOp::And,
                            right: right.clone(),
                        },
                        None => right.as_ref().clone(),
                    };
                    return Some((subplan, negated, Some(remaining)));
                }
                // Recurse into right subtree
                if let Some((subplan, negated, inner_remaining)) =
                    self.extract_complex_exists(right)
                {
                    let remaining = match inner_remaining {
                        Some(inner) => LogicalExpression::Binary {
                            left: left.clone(),
                            op: BinaryOp::And,
                            right: Box::new(inner),
                        },
                        None => left.as_ref().clone(),
                    };
                    return Some((subplan, negated, Some(remaining)));
                }
                None
            }
            _ => None,
        }
    }

    /// Helper: extracts EXISTS or NOT EXISTS from a single expression node.
    /// Returns `(subplan, is_negated)`.
    fn extract_exists_from_expr(expr: &LogicalExpression) -> Option<(&LogicalOperator, bool)> {
        match expr {
            LogicalExpression::ExistsSubquery(subplan) => Some((subplan.as_ref(), false)),
            LogicalExpression::Unary {
                op: UnaryOp::Not,
                operand,
            } => {
                if let LogicalExpression::ExistsSubquery(subplan) = operand.as_ref() {
                    Some((subplan.as_ref(), true))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Checks if a top-level OR predicate contains a complex EXISTS subquery that
    /// cannot use the inline fast path. Returns the EXISTS subplan, its negation
    /// flag, and the other side of the OR.
    fn extract_exists_from_or<'a>(
        &self,
        predicate: &'a LogicalExpression,
    ) -> Option<(&'a LogicalOperator, bool, &'a LogicalExpression)> {
        let LogicalExpression::Binary {
            op: BinaryOp::Or,
            left,
            right,
        } = predicate
        else {
            return None;
        };

        // Check left side for complex EXISTS
        if let Some((subplan, negated)) = Self::extract_exists_from_expr(left)
            && self.extract_exists_pattern(subplan).is_err()
        {
            return Some((subplan, negated, right));
        }
        // Check right side for complex EXISTS
        if let Some((subplan, negated)) = Self::extract_exists_from_expr(right)
            && self.extract_exists_pattern(subplan).is_err()
        {
            return Some((subplan, negated, left));
        }
        None
    }

    /// Plans a filter with `complex_EXISTS OR other_pred` using Union + Distinct.
    ///
    /// The OR is split into two branches:
    /// 1. Semi-join (or anti-join for NOT EXISTS) for the EXISTS branch
    /// 2. Normal filter for the scalar predicate branch
    ///
    /// The results are combined with UNION ALL and deduplicated.
    fn plan_exists_or_as_union(
        &self,
        input: &LogicalOperator,
        subquery: &LogicalOperator,
        is_negated: bool,
        other_predicate: &LogicalExpression,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Branch 1: rows matching EXISTS (via semi-join / anti-join)
        let (exists_op, exists_cols) =
            self.plan_exists_as_semi_join(input, subquery, is_negated, None)?;

        // Branch 2: rows matching the scalar predicate (via normal filter)
        let other_filter = super::FilterOp {
            predicate: other_predicate.clone(),
            input: Box::new(input.clone()),
            pushdown_hint: None,
        };
        let (other_op, _other_cols) = self.plan_filter(&other_filter)?;

        // Union both branches
        let schema = self.derive_schema_from_columns(&exists_cols);
        let union_op = UnionOperator::new(vec![exists_op, other_op], schema.clone());

        // Deduplicate (OR semantics: each row appears at most once)
        let distinct_op = DistinctOperator::new(Box::new(union_op), schema);

        Ok((Box::new(distinct_op), exists_cols))
    }

    /// Plans a complex EXISTS/NOT EXISTS as a hash-based semi-join or anti-join.
    ///
    /// The inner subquery is planned as a full operator tree via `plan_operator()`.
    /// Variables shared between the outer input and inner subquery become equi-join
    /// keys. Uses `HashJoinOperator` for efficient O(N + M) evaluation.
    ///
    /// When the inner subquery contains a `ParameterScan` (indicating correlated
    /// variable references from the outer scope), a correlated `ApplyOperator`
    /// with EXISTS mode is used instead of a hash-based semi-join.
    fn plan_exists_as_semi_join(
        &self,
        outer_input: &LogicalOperator,
        subquery: &LogicalOperator,
        is_negated: bool,
        remaining_predicate: Option<LogicalExpression>,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Detect correlated subquery (contains ParameterScan from translator)
        if let Some(param_vars) = Self::extract_parameter_scan_vars(subquery) {
            return self.plan_correlated_exists(
                outer_input,
                subquery,
                &param_vars,
                is_negated,
                remaining_predicate,
            );
        }

        let (left_op, left_columns) = self.plan_operator(outer_input)?;
        let (right_op, right_columns) = self.plan_operator(subquery)?;

        // Semi/anti join only outputs left (outer) columns
        let output_columns = left_columns.clone();

        // Find shared variables for equi-join keys
        let mut probe_keys = Vec::new();
        let mut build_keys = Vec::new();
        for (right_idx, right_col) in right_columns.iter().enumerate() {
            if let Some(left_idx) = left_columns.iter().position(|c| c == right_col) {
                probe_keys.push(left_idx);
                build_keys.push(right_idx);
            }
        }

        let output_schema = self.derive_schema_from_columns(&output_columns);

        let join_type = if is_negated {
            PhysicalJoinType::Anti
        } else {
            PhysicalJoinType::Semi
        };

        let join_op: Box<dyn Operator> = Box::new(HashJoinOperator::new(
            left_op,
            right_op,
            probe_keys,
            build_keys,
            join_type,
            output_schema,
        ));

        // If there's a remaining predicate (from AND splitting), check if it
        // contains more EXISTS subqueries that need semi-join rewriting.
        if let Some(ref remaining) = remaining_predicate {
            // Recursively handle nested EXISTS in the remaining predicate
            if let Some((nested_sub, nested_neg, nested_rest)) =
                self.extract_complex_exists(remaining)
            {
                return self.plan_exists_as_semi_join_with_input(
                    join_op,
                    &output_columns,
                    nested_sub,
                    nested_neg,
                    nested_rest,
                );
            }

            let variable_columns: HashMap<String, usize> = output_columns
                .iter()
                .enumerate()
                .map(|(i, name)| (name.clone(), i))
                .collect();
            let filter_expr = self.convert_expression(remaining)?;
            let predicate = ExpressionPredicate::new(
                filter_expr,
                variable_columns,
                Arc::clone(&self.store) as Arc<dyn GraphStoreSearch>,
            )
            .with_transaction_context(self.viewing_epoch, self.transaction_id)
            .with_session_context(self.session_context.clone());
            let filter_op = Box::new(FilterOperator::new(join_op, Box::new(predicate)));
            return Ok((filter_op, output_columns));
        }

        Ok((join_op, output_columns))
    }

    /// Extracts `ParameterScan` variable names from a logical plan, if present.
    ///
    /// Returns `Some(vars)` when the plan contains a `ParameterScan` node
    /// (indicating a correlated subquery from the translator).
    fn extract_parameter_scan_vars(plan: &LogicalOperator) -> Option<Vec<String>> {
        match plan {
            LogicalOperator::ParameterScan(ps) => Some(ps.columns.clone()),
            LogicalOperator::Filter(f) => Self::extract_parameter_scan_vars(&f.input),
            LogicalOperator::Join(j) => Self::extract_parameter_scan_vars(&j.left)
                .or_else(|| Self::extract_parameter_scan_vars(&j.right)),
            LogicalOperator::NodeScan(s) => s
                .input
                .as_ref()
                .and_then(|i| Self::extract_parameter_scan_vars(i)),
            LogicalOperator::Expand(e) => Self::extract_parameter_scan_vars(&e.input),
            _ => None,
        }
    }

    /// Plans a correlated EXISTS/NOT EXISTS using `ApplyOperator` with EXISTS mode.
    ///
    /// The inner subquery contains a `ParameterScan` for outer variable references.
    /// For each outer row, the inner subquery is executed with the outer values
    /// injected via `ParameterState`. Semi-join keeps rows where inner has results,
    /// anti-join keeps rows where inner has no results.
    fn plan_correlated_exists(
        &self,
        outer_input: &LogicalOperator,
        subquery: &LogicalOperator,
        param_vars: &[String],
        is_negated: bool,
        remaining_predicate: Option<LogicalExpression>,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (left_op, left_columns) = self.plan_operator(outer_input)?;

        // Set up ParameterState for correlated variables
        let param_state = std::sync::Arc::new(
            grafeo_core::execution::operators::ParameterState::new(param_vars.to_vec()),
        );
        let param_col_indices: Vec<usize> = param_vars
            .iter()
            .map(|var| left_columns.iter().position(|c| c == var).unwrap_or(0))
            .collect();

        // Plan inner subquery with correlated context
        *self.correlated_param_state.borrow_mut() = Some(std::sync::Arc::clone(&param_state));
        let (inner_op, _inner_columns) = self.plan_operator(subquery)?;
        *self.correlated_param_state.borrow_mut() = None;

        // Create Apply with EXISTS mode (semi or anti join)
        let op = ApplyOperator::new_correlated(left_op, inner_op, param_state, param_col_indices)
            .with_exists_mode(!is_negated);

        let output_columns = left_columns;
        let mut result: Box<dyn Operator> = Box::new(op);

        // Handle remaining predicate (from AND splitting)
        if let Some(ref remaining) = remaining_predicate {
            if let Some((nested_sub, nested_neg, nested_rest)) =
                self.extract_complex_exists(remaining)
            {
                return self.plan_exists_as_semi_join_with_input(
                    result,
                    &output_columns,
                    nested_sub,
                    nested_neg,
                    nested_rest,
                );
            }

            let variable_columns: HashMap<String, usize> = output_columns
                .iter()
                .enumerate()
                .map(|(i, name)| (name.clone(), i))
                .collect();
            let filter_expr = self.convert_expression(remaining)?;
            let predicate = ExpressionPredicate::new(
                filter_expr,
                variable_columns,
                Arc::clone(&self.store) as Arc<dyn GraphStoreSearch>,
            )
            .with_transaction_context(self.viewing_epoch, self.transaction_id)
            .with_session_context(self.session_context.clone());
            result = Box::new(FilterOperator::new(result, Box::new(predicate)));
        }

        Ok((result, output_columns))
    }

    /// Plans an EXISTS/NOT EXISTS semi-join with an already-planned outer input.
    ///
    /// Used when the remaining predicate from a prior semi-join contains additional
    /// EXISTS subqueries that also need semi-join rewriting (nested NOT EXISTS).
    fn plan_exists_as_semi_join_with_input(
        &self,
        left_op: Box<dyn Operator>,
        left_columns: &[String],
        subquery: &LogicalOperator,
        is_negated: bool,
        remaining_predicate: Option<LogicalExpression>,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (right_op, right_columns) = self.plan_operator(subquery)?;

        let output_columns = left_columns.to_vec();

        let mut probe_keys = Vec::new();
        let mut build_keys = Vec::new();
        for (right_idx, right_col) in right_columns.iter().enumerate() {
            if let Some(left_idx) = left_columns.iter().position(|c| c == right_col) {
                probe_keys.push(left_idx);
                build_keys.push(right_idx);
            }
        }

        let output_schema = self.derive_schema_from_columns(&output_columns);
        let join_type = if is_negated {
            PhysicalJoinType::Anti
        } else {
            PhysicalJoinType::Semi
        };

        let join_op: Box<dyn Operator> = Box::new(HashJoinOperator::new(
            left_op,
            right_op,
            probe_keys,
            build_keys,
            join_type,
            output_schema,
        ));

        // Recursively handle any further EXISTS in the remaining predicate
        if let Some(ref remaining) = remaining_predicate {
            if let Some((nested_sub, nested_neg, nested_rest)) =
                self.extract_complex_exists(remaining)
            {
                return self.plan_exists_as_semi_join_with_input(
                    join_op,
                    &output_columns,
                    nested_sub,
                    nested_neg,
                    nested_rest,
                );
            }

            let variable_columns: HashMap<String, usize> = output_columns
                .iter()
                .enumerate()
                .map(|(i, name)| (name.clone(), i))
                .collect();
            let filter_expr = self.convert_expression(remaining)?;
            let predicate = ExpressionPredicate::new(
                filter_expr,
                variable_columns,
                Arc::clone(&self.store) as Arc<dyn GraphStoreSearch>,
            )
            .with_transaction_context(self.viewing_epoch, self.transaction_id)
            .with_session_context(self.session_context.clone());
            let filter_op = Box::new(FilterOperator::new(join_op, Box::new(predicate)));
            return Ok((filter_op, output_columns));
        }

        Ok((join_op, output_columns))
    }

    /// Extracts a COUNT subquery comparison from a filter predicate.
    ///
    /// Recognizes patterns like:
    /// - `COUNT { MATCH ... } > 5`
    /// - `COUNT { MATCH ... } = 0`
    /// - `5 < COUNT { MATCH ... }` (reversed operands)
    ///
    /// Returns `(subquery, comparison_op, threshold_value, remaining_predicate)`.
    fn extract_count_comparison(
        predicate: &LogicalExpression,
    ) -> Option<(
        &LogicalOperator,
        BinaryOp,
        &LogicalExpression,
        Option<&LogicalExpression>,
    )> {
        match predicate {
            LogicalExpression::Binary { left, op, right } => {
                // Check for AND-combined: extract COUNT comparison from either side
                if *op == BinaryOp::And {
                    if let Some(result) = Self::extract_count_from_binary(left) {
                        return Some((result.0, result.1, result.2, Some(right)));
                    }
                    if let Some(result) = Self::extract_count_from_binary(right) {
                        return Some((result.0, result.1, result.2, Some(left)));
                    }
                    return None;
                }

                // Direct comparison: COUNT { ... } op value
                Self::extract_count_from_binary(predicate)
                    .map(|(sub, op, threshold)| (sub, op, threshold, None))
            }
            _ => None,
        }
    }

    /// Helper: extracts COUNT subquery comparison from a binary expression.
    fn extract_count_from_binary(
        expr: &LogicalExpression,
    ) -> Option<(&LogicalOperator, BinaryOp, &LogicalExpression)> {
        if let LogicalExpression::Binary { left, op, right } = expr {
            match op {
                BinaryOp::Eq
                | BinaryOp::Ne
                | BinaryOp::Gt
                | BinaryOp::Ge
                | BinaryOp::Lt
                | BinaryOp::Le => {
                    // COUNT { ... } op literal
                    if let LogicalExpression::CountSubquery(subplan) = left.as_ref() {
                        return Some((subplan.as_ref(), *op, right.as_ref()));
                    }
                    // literal op COUNT { ... } (flip the operator)
                    if let LogicalExpression::CountSubquery(subplan) = right.as_ref() {
                        let flipped = match op {
                            BinaryOp::Gt => BinaryOp::Lt,
                            BinaryOp::Ge => BinaryOp::Le,
                            BinaryOp::Lt => BinaryOp::Gt,
                            BinaryOp::Le => BinaryOp::Ge,
                            other => *other, // Eq/Ne are symmetric
                        };
                        return Some((subplan.as_ref(), flipped, left.as_ref()));
                    }
                }
                _ => {}
            }
        }
        None
    }

    /// Plans a COUNT subquery comparison as Join + Aggregate + Filter.
    ///
    /// Rewrites `WHERE COUNT { MATCH pattern } > N` into:
    /// 1. Inner join on shared variables to get all matches per outer row
    /// 2. Aggregate(COUNT) grouped by outer columns
    /// 3. Filter(count > N) on the aggregated result
    fn plan_count_as_apply(
        &self,
        outer_input: &LogicalOperator,
        subquery: &LogicalOperator,
        op: BinaryOp,
        threshold: &LogicalExpression,
        remaining_predicate: Option<&LogicalExpression>,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (left_op, left_columns) = self.plan_operator(outer_input)?;
        let (right_op, right_columns) = self.plan_operator(subquery)?;

        let output_columns = left_columns.clone();

        // Find shared variables for equi-join keys
        let mut probe_keys = Vec::new();
        let mut build_keys = Vec::new();
        for (right_idx, right_col) in right_columns.iter().enumerate() {
            if let Some(left_idx) = left_columns.iter().position(|c| c == right_col) {
                probe_keys.push(left_idx);
                build_keys.push(right_idx);
            }
        }

        // Left join to preserve outer rows with no matches (needed for COUNT = 0).
        // The join physically outputs both left and right columns.
        let mut join_columns = left_columns.clone();
        join_columns.extend(right_columns.iter().cloned());
        let join_schema = self.derive_schema_from_columns(&join_columns);
        let join_op: Box<dyn Operator> = Box::new(HashJoinOperator::new(
            left_op,
            right_op,
            probe_keys,
            build_keys,
            PhysicalJoinType::Left,
            join_schema,
        ));

        // Aggregate: COUNT(right_col) grouped by all outer columns.
        // Using COUNT on a right-side column so nulls (no match) produce 0 instead of 1.
        let count_alias = "_count_subquery_".to_string();
        let mut agg_columns = output_columns.clone();
        agg_columns.push(count_alias.clone());

        let group_keys: Vec<usize> = (0..output_columns.len()).collect();
        // Pick the first right-side column for COUNT (index = left_columns.len())
        let right_col_idx = left_columns.len();
        let agg_exprs = vec![PhysicalAggregateExpr::count(right_col_idx)];
        let agg_schema = self.derive_schema_from_columns(&agg_columns);

        let agg_op: Box<dyn Operator> = Box::new(HashAggregateOperator::new(
            join_op, group_keys, agg_exprs, agg_schema,
        ));

        // Filter: _count_ op threshold
        let threshold_expr = convert_filter_expression(threshold)?;
        let count_var_columns: HashMap<String, usize> = agg_columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();
        let filter_op_code = convert_binary_op(op)?;
        let count_filter = FilterExpression::Binary {
            left: Box::new(FilterExpression::Variable(count_alias)),
            op: filter_op_code,
            right: Box::new(threshold_expr),
        };

        let predicate = ExpressionPredicate::new(
            count_filter,
            count_var_columns.clone(),
            Arc::clone(&self.store) as Arc<dyn GraphStoreSearch>,
        )
        .with_transaction_context(self.viewing_epoch, self.transaction_id)
        .with_session_context(self.session_context.clone());
        let mut result_op: Box<dyn Operator> =
            Box::new(FilterOperator::new(agg_op, Box::new(predicate)));

        // If there's a remaining predicate, apply it too
        if let Some(remaining) = remaining_predicate {
            let remaining_expr = self.convert_expression(remaining)?;
            let remaining_predicate = ExpressionPredicate::new(
                remaining_expr,
                count_var_columns,
                Arc::clone(&self.store) as Arc<dyn GraphStoreSearch>,
            )
            .with_transaction_context(self.viewing_epoch, self.transaction_id)
            .with_session_context(self.session_context.clone());
            result_op = Box::new(FilterOperator::new(
                result_op,
                Box::new(remaining_predicate),
            ));
        }

        Ok((result_op, output_columns))
    }

    /// Checks zone maps for a predicate to see if we can skip the scan entirely.
    ///
    /// Returns:
    /// - `Some(false)` if zone map proves no matches possible (can skip)
    /// - `Some(true)` if zone map says matches might exist
    /// - `None` if zone map check not applicable
    pub(super) fn check_zone_map_for_predicate(
        &self,
        predicate: &LogicalExpression,
    ) -> Option<bool> {
        use grafeo_core::graph::lpg::CompareOp;

        match predicate {
            LogicalExpression::Binary { left, op, right } => {
                // Check for AND/OR first (compound conditions)
                match op {
                    BinaryOp::And => {
                        let left_result = self.check_zone_map_for_predicate(left);
                        let right_result = self.check_zone_map_for_predicate(right);

                        return match (left_result, right_result) {
                            // If either side definitely won't match, the AND won't match
                            (Some(false), _) | (_, Some(false)) => Some(false),
                            // If both might match, might match overall
                            (Some(true), Some(true)) => Some(true),
                            // Otherwise, can't determine
                            _ => None,
                        };
                    }
                    BinaryOp::Or => {
                        let left_result = self.check_zone_map_for_predicate(left);
                        let right_result = self.check_zone_map_for_predicate(right);

                        return match (left_result, right_result) {
                            // Both sides definitely won't match
                            (Some(false), Some(false)) => Some(false),
                            // At least one side might match
                            (Some(true), _) | (_, Some(true)) => Some(true),
                            // Otherwise, can't determine
                            _ => None,
                        };
                    }
                    _ => {}
                }

                // Simple property comparison: n.property op value
                let (property, compare_op, value) = match (left.as_ref(), right.as_ref()) {
                    (
                        LogicalExpression::Property { property, .. },
                        LogicalExpression::Literal(val),
                    ) => {
                        let cmp = match op {
                            BinaryOp::Eq => CompareOp::Eq,
                            BinaryOp::Ne => CompareOp::Ne,
                            BinaryOp::Lt => CompareOp::Lt,
                            BinaryOp::Le => CompareOp::Le,
                            BinaryOp::Gt => CompareOp::Gt,
                            BinaryOp::Ge => CompareOp::Ge,
                            _ => return None,
                        };
                        (property.clone(), cmp, val.clone())
                    }
                    (
                        LogicalExpression::Literal(val),
                        LogicalExpression::Property { property, .. },
                    ) => {
                        // Flip comparison for reversed operands
                        let cmp = match op {
                            BinaryOp::Eq => CompareOp::Eq,
                            BinaryOp::Ne => CompareOp::Ne,
                            BinaryOp::Lt => CompareOp::Gt, // val < prop means prop > val
                            BinaryOp::Le => CompareOp::Ge,
                            BinaryOp::Gt => CompareOp::Lt,
                            BinaryOp::Ge => CompareOp::Le,
                            _ => return None,
                        };
                        (property.clone(), cmp, val.clone())
                    }
                    _ => return None,
                };

                // Check zone map for node properties
                let might_match =
                    self.store
                        .node_property_might_match(&property.into(), compare_op, &value);

                Some(might_match)
            }

            _ => None,
        }
    }

    /// Tries to use a property index for filter optimization.
    ///
    /// When a filter predicate is an equality check on an indexed property,
    /// and the input is a simple NodeScan, we can use the index to look up
    /// matching nodes directly instead of scanning all nodes.
    ///
    /// Returns `Ok(Some((operator, columns)))` if optimization was applied,
    /// `Ok(None)` if not applicable, or `Err` on error.
    pub(super) fn try_plan_filter_with_property_index(
        &self,
        filter: &FilterOp,
    ) -> Result<Option<(Box<dyn Operator>, Vec<String>)>> {
        // Only optimize if input is a simple NodeScan (not nested)
        let (scan_variable, scan_label) = match filter.input.as_ref() {
            LogicalOperator::NodeScan(scan) if scan.input.is_none() => {
                (scan.variable.clone(), scan.label.clone())
            }
            _ => return Ok(None),
        };

        // Extract property equality conditions from the predicate
        // Handles both simple (n.prop = val) and compound (n.a = 1 AND n.b = 2)
        let conditions = self.extract_equality_conditions(&filter.predicate, &scan_variable);

        if conditions.is_empty() {
            return Ok(None);
        }

        // Check if at least one condition has an index
        let has_indexed_condition = conditions
            .iter()
            .any(|(prop, _)| self.store.has_property_index(prop));

        // Without an index we can still optimize when there's a label constraint:
        // label-first scan + property check avoids DataChunk/expression overhead.
        if !has_indexed_condition && scan_label.is_none() {
            return Ok(None);
        }

        let mut matching_nodes = if has_indexed_condition {
            // Use index-based batch lookup
            let conditions_ref: Vec<(&str, Value)> = conditions
                .iter()
                .map(|(p, v)| (p.as_str(), v.clone()))
                .collect();
            let mut nodes = self.store.find_nodes_by_properties(&conditions_ref);

            // Intersect with label if present
            if let Some(label) = &scan_label {
                let label_nodes: std::collections::HashSet<_> =
                    self.store.nodes_by_label(label).into_iter().collect();
                nodes.retain(|n| label_nodes.contains(n));
            }
            nodes
        } else {
            // No index but we have a label: scan label first, then check properties.
            // This is more efficient than ScanOperator → DataChunk → FilterOperator
            // because it avoids DataChunk materialization and expression evaluation.
            let label = scan_label.as_ref().expect("label checked above");
            let label_nodes = self.store.nodes_by_label(label);
            let epoch = self.viewing_epoch;
            let tx_id = self.transaction_id;
            label_nodes
                .into_iter()
                .filter(|&node_id| {
                    // Use versioned/epoch-aware node access for correct properties
                    let node = if let Some(tx) = tx_id {
                        self.store.get_node_versioned(node_id, epoch, tx)
                    } else {
                        self.store.get_node_at_epoch(node_id, epoch)
                    };
                    node.is_some_and(|n| {
                        conditions.iter().all(|(prop, val)| {
                            n.get_property(prop)
                                .is_some_and(|v| values_equal_coerced(v, val))
                        })
                    })
                })
                .collect()
        };

        // MVCC visibility: filter out nodes not visible at the current epoch/tx.
        // Without this, rolled-back or uncommitted nodes could leak through.
        let epoch = self.viewing_epoch;
        if let Some(tx) = self.transaction_id {
            matching_nodes.retain(|id| self.store.get_node_versioned(*id, epoch, tx).is_some());
        } else {
            matching_nodes.retain(|id| self.store.get_node_at_epoch(*id, epoch).is_some());
        }

        let columns = vec![scan_variable.clone()];
        let node_list_op: Box<dyn Operator> = Box::new(NodeListOperator::new(matching_nodes, 2048));

        // Index optimization absorbs the child NodeScan — emit a
        // synthetic profile entry so build_profile_tree's strict
        // logical-op:entry 1:1 mapping holds. See
        // `record_absorbed_scan_entry` for full reasoning.
        self.record_absorbed_scan_entry(&filter.input);

        // Check for remaining predicate parts that weren't pushed down
        // (e.g., range conditions in a compound predicate like `n.name = 'Alix' AND n.age > 30`)
        if let Some(remaining) =
            self.extract_remaining_predicate(&filter.predicate, &scan_variable, &conditions)
        {
            let variable_columns: HashMap<String, usize> = columns
                .iter()
                .enumerate()
                .map(|(i, name)| (name.clone(), i))
                .collect();
            let filter_expr = self.convert_expression(&remaining)?;
            let predicate = ExpressionPredicate::new(
                filter_expr,
                variable_columns,
                Arc::clone(&self.store) as Arc<dyn GraphStoreSearch>,
            )
            .with_transaction_context(self.viewing_epoch, self.transaction_id)
            .with_session_context(self.session_context.clone());
            let filtered = Box::new(FilterOperator::new(node_list_op, Box::new(predicate)));
            Ok(Some((filtered, columns)))
        } else {
            Ok(Some((node_list_op, columns)))
        }
    }

    /// Tries to optimize a filter shaped as `var.prop IN [literals]` using
    /// the property index. For each literal, calls `find_nodes_by_property`
    /// and unions the results. Returns `Ok(None)` when not applicable —
    /// e.g. the input isn't a simple NodeScan, the right side isn't a list
    /// of literals, the property is unindexed, or the predicate is part of
    /// a more complex expression we don't yet decompose.
    ///
    /// Compound `prop IN [...] AND label-or-other` is currently left to
    /// the slow path. Pure `IN` is the common shape (`WHERE r.id IN $ids`).
    pub(super) fn try_plan_filter_with_in_list_index(
        &self,
        filter: &FilterOp,
    ) -> Result<Option<(Box<dyn Operator>, Vec<String>)>> {
        // Only optimize if input is a simple NodeScan (not nested).
        let (scan_variable, scan_label) = match filter.input.as_ref() {
            LogicalOperator::NodeScan(scan) if scan.input.is_none() => {
                (scan.variable.clone(), scan.label.clone())
            }
            _ => return Ok(None),
        };

        // Predicate must be a single `var.prop IN [literal, ...]`.
        let LogicalExpression::Binary {
            left,
            op: BinaryOp::In,
            right,
        } = &filter.predicate
        else {
            return Ok(None);
        };
        let LogicalExpression::Property { variable, property } = left.as_ref() else {
            return Ok(None);
        };
        if variable != &scan_variable {
            return Ok(None);
        }
        if !self.store.has_property_index(property) {
            return Ok(None);
        }

        // Right side: List of literal expressions, or a Literal containing a Value::List.
        let values: Vec<Value> = match right.as_ref() {
            LogicalExpression::List(items) => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    match item {
                        LogicalExpression::Literal(v) if !v.is_null() => out.push(v.clone()),
                        // A non-literal element (parameter, expression) means
                        // we can't enumerate at plan time — bail to slow path.
                        _ => return Ok(None),
                    }
                }
                out
            }
            LogicalExpression::Literal(Value::List(items)) => items
                .iter()
                .filter(|v| !v.is_null())
                .cloned()
                .collect(),
            _ => return Ok(None),
        };
        if values.is_empty() {
            // `prop IN []` is always false — return an empty result without
            // touching the store.
            self.record_absorbed_scan_entry(&filter.input);
            let columns = vec![scan_variable];
            let empty = Box::new(NodeListOperator::new(Vec::new(), 2048));
            return Ok(Some((empty, columns)));
        }

        // Per-value index lookup, deduplicate (the same node could match
        // duplicate values, and we don't want to emit it twice).
        let mut seen = std::collections::HashSet::new();
        let mut matching_nodes = Vec::new();
        for value in &values {
            for node_id in self.store.find_nodes_by_property(property, value) {
                if seen.insert(node_id) {
                    matching_nodes.push(node_id);
                }
            }
        }

        // Intersect with the label constraint, if any.
        if let Some(label) = &scan_label {
            let label_nodes: std::collections::HashSet<_> =
                self.store.nodes_by_label(label).into_iter().collect();
            matching_nodes.retain(|n| label_nodes.contains(n));
        }

        // MVCC visibility: drop nodes that aren't visible at the current
        // epoch/tx, matching the equality fast path's semantics.
        let epoch = self.viewing_epoch;
        if let Some(tx) = self.transaction_id {
            matching_nodes.retain(|id| self.store.get_node_versioned(*id, epoch, tx).is_some());
        } else {
            matching_nodes.retain(|id| self.store.get_node_at_epoch(*id, epoch).is_some());
        }

        // Index optimization absorbs the child NodeScan — emit a synthetic
        // profile entry so build_profile_tree's strict logical-op:entry 1:1
        // mapping holds. See `record_absorbed_scan_entry` for full reasoning.
        self.record_absorbed_scan_entry(&filter.input);

        let columns = vec![scan_variable];
        let node_list_op: Box<dyn Operator> = Box::new(NodeListOperator::new(matching_nodes, 2048));
        Ok(Some((node_list_op, columns)))
    }

    /// Extracts the remaining predicate after removing pushed-down equality conditions.
    ///
    /// Given `n.name = 'Alix' AND n.age > 30` with pushed conditions `[("name", "Alix")]`,
    /// returns `Some(n.age > 30)`. Returns `None` when all conditions were pushed down.
    pub(super) fn extract_remaining_predicate(
        &self,
        predicate: &LogicalExpression,
        target_variable: &str,
        pushed_conditions: &[(String, Value)],
    ) -> Option<LogicalExpression> {
        match predicate {
            LogicalExpression::Binary {
                left,
                op: BinaryOp::And,
                right,
            } => {
                let left_remaining =
                    self.extract_remaining_predicate(left, target_variable, pushed_conditions);
                let right_remaining =
                    self.extract_remaining_predicate(right, target_variable, pushed_conditions);

                match (left_remaining, right_remaining) {
                    (Some(l), Some(r)) => Some(LogicalExpression::Binary {
                        left: Box::new(l),
                        op: BinaryOp::And,
                        right: Box::new(r),
                    }),
                    (Some(l), None) => Some(l),
                    (None, Some(r)) => Some(r),
                    (None, None) => None,
                }
            }
            LogicalExpression::Binary {
                left,
                op: BinaryOp::Eq,
                right,
            } => {
                // Check if this equality was pushed down
                if let Some((var, prop, val)) = self.extract_property_equality(left, right)
                    && var == target_variable
                    && pushed_conditions
                        .iter()
                        .any(|(p, v)| *p == prop && *v == val)
                {
                    None // Already handled at the store level
                } else {
                    Some(predicate.clone())
                }
            }
            _ => Some(predicate.clone()),
        }
    }

    /// Extracts equality conditions (property = literal) from a predicate.
    ///
    /// Handles both simple predicates and AND chains:
    /// - `n.name = "Alix"` → `[("name", "Alix")]`
    /// - `n.name = "Alix" AND n.age = 30` → `[("name", "Alix"), ("age", 30)]`
    pub(super) fn extract_equality_conditions(
        &self,
        predicate: &LogicalExpression,
        target_variable: &str,
    ) -> Vec<(String, Value)> {
        let mut conditions = Vec::new();
        self.collect_equality_conditions(predicate, target_variable, &mut conditions);
        conditions
    }

    /// Recursively collects equality conditions from AND expressions.
    pub(super) fn collect_equality_conditions(
        &self,
        expr: &LogicalExpression,
        target_variable: &str,
        conditions: &mut Vec<(String, Value)>,
    ) {
        match expr {
            // Handle AND: recurse into both sides
            LogicalExpression::Binary {
                left,
                op: BinaryOp::And,
                right,
            } => {
                self.collect_equality_conditions(left, target_variable, conditions);
                self.collect_equality_conditions(right, target_variable, conditions);
            }

            // Handle equality: extract property and value
            LogicalExpression::Binary {
                left,
                op: BinaryOp::Eq,
                right,
            } => {
                if let Some((var, prop, val)) = self.extract_property_equality(left, right)
                    && var == target_variable
                {
                    conditions.push((prop, val));
                }
            }

            _ => {}
        }
    }

    /// Extracts (variable, property, value) from a property equality expression.
    ///
    /// Returns `None` for NULL literals: `property = NULL` is UNKNOWN in
    /// three-valued logic and must never be optimized into an index lookup.
    pub(super) fn extract_property_equality(
        &self,
        left: &LogicalExpression,
        right: &LogicalExpression,
    ) -> Option<(String, String, Value)> {
        match (left, right) {
            (
                LogicalExpression::Property { variable, property },
                LogicalExpression::Literal(val),
            ) if !val.is_null() => Some((variable.clone(), property.clone(), val.clone())),
            (
                LogicalExpression::Literal(val),
                LogicalExpression::Property { variable, property },
            ) if !val.is_null() => Some((variable.clone(), property.clone(), val.clone())),
            _ => None,
        }
    }

    /// Tries to optimize a filter using range queries on properties.
    ///
    /// This optimization is applied when:
    /// - The input is a simple NodeScan (no nested operations)
    /// - The predicate contains range comparisons (>, <, >=, <=)
    /// - The same variable and property are being filtered
    ///
    /// Handles both simple range predicates (`n.age > 30`) and BETWEEN patterns
    /// (`n.age >= 30 AND n.age <= 50`).
    ///
    /// Returns `Ok(Some((operator, columns)))` if optimization was applied,
    /// `Ok(None)` if not applicable, or `Err` on error.
    /// When a `Filter` plan_* fast path absorbs its child NodeScan into
    /// a single physical operator (NodeListOperator), the NodeScan
    /// logical operator never gets planned via `plan_operator` — so no
    /// `ProfileEntry` is recorded for it. `build_profile_tree` then
    /// panics with "profile entry count must match logical operator
    /// count" because it walks the logical plan in post-order and
    /// expects 1 entry per logical operator. This helper pushes a
    /// synthetic entry for the absorbed scan so the mapping holds.
    ///
    /// Stats stay empty — the actual scan-and-filter work is
    /// attributed to the wrapping operator's entry, recorded by the
    /// outer `maybe_profile` once the Filter logical op finishes
    /// planning. Slight cosmetic quirk in PROFILE output (NodeScan
    /// shows 0 rows / 0 ns) but no panic and no perf impact.
    pub(super) fn record_absorbed_scan_entry(&self, absorbed: &LogicalOperator) {
        if !self.profiling.get() {
            return;
        }
        let label = match absorbed {
            LogicalOperator::NodeScan(_) => absorbed.display_label(),
            _ => String::from("NodeScan(absorbed)"),
        };
        let (entry, _stats) = crate::query::profile::ProfileEntry::new("NodeScan", label);
        self.profile_entries.borrow_mut().push(entry);
    }

    pub(super) fn try_plan_filter_with_range_index(
        &self,
        filter: &FilterOp,
    ) -> Result<Option<(Box<dyn Operator>, Vec<String>)>> {
        // Only optimize if input is a simple NodeScan (not nested)
        let (scan_variable, scan_label) = match filter.input.as_ref() {
            LogicalOperator::NodeScan(scan) if scan.input.is_none() => {
                (scan.variable.clone(), scan.label.clone())
            }
            _ => return Ok(None),
        };

        // Try to extract BETWEEN pattern first (more efficient)
        if let Some((variable, property, min, max, min_inc, max_inc)) =
            self.extract_between_predicate(&filter.predicate)
            && variable == scan_variable
        {
            self.record_absorbed_scan_entry(&filter.input);
            return self.plan_range_filter(
                &scan_variable,
                &scan_label,
                &property,
                RangeBounds {
                    min: Some(&min),
                    max: Some(&max),
                    min_inclusive: min_inc,
                    max_inclusive: max_inc,
                },
            );
        }

        // Try to extract simple range predicate
        if let Some((variable, property, op, value)) =
            self.extract_range_predicate(&filter.predicate)
            && variable == scan_variable
        {
            let (min, max, min_inc, max_inc) = match op {
                BinaryOp::Lt => (None, Some(value), false, false),
                BinaryOp::Le => (None, Some(value), false, true),
                BinaryOp::Gt => (Some(value), None, false, false),
                BinaryOp::Ge => (Some(value), None, true, false),
                _ => return Ok(None),
            };
            self.record_absorbed_scan_entry(&filter.input);
            return self.plan_range_filter(
                &scan_variable,
                &scan_label,
                &property,
                RangeBounds {
                    min: min.as_ref(),
                    max: max.as_ref(),
                    min_inclusive: min_inc,
                    max_inclusive: max_inc,
                },
            );
        }

        Ok(None)
    }

    /// Plans a range filter using [`RangeScanOperator`].
    ///
    /// Phase 4d emits a dedicated range-scan operator that consumes
    /// [`GraphStoreSearch::find_nodes_in_range_iter`] (lazy, with
    /// per-block zone-map skip pruning when the store is a CompactStore;
    /// eager fallback otherwise). The operator absorbs the label and
    /// MVCC post-filters that the eager `NodeListOperator` path used to
    /// apply manually, preserving identical query semantics.
    ///
    /// Phase 4e: when an outer `plan_limit` set `self.limit_hint`, the
    /// hint is forwarded to the operator via `with_limit(n)` so the
    /// materialization step terminates after `n` matches. The outer
    /// `LimitOperator` still wraps the result; the hint is a pure
    /// optimization.
    pub(super) fn plan_range_filter(
        &self,
        scan_variable: &str,
        scan_label: &Option<String>,
        property: &str,
        bounds: RangeBounds<'_>,
    ) -> Result<Option<(Box<dyn Operator>, Vec<String>)>> {
        let epoch = self.viewing_epoch;
        let tx = self.transaction_id.unwrap_or(TransactionId::SYSTEM);

        let mut op = RangeScanOperator::new(
            Arc::clone(&self.store),
            property.to_string(),
            bounds.min.cloned(),
            bounds.max.cloned(),
            bounds.min_inclusive,
            bounds.max_inclusive,
            2048,
        )
        .with_transaction_context(epoch, tx);

        if let Some(label) = scan_label {
            op = op.with_label_filter(label.clone());
        }

        if let Some(hint) = self.limit_hint.get() {
            op = op.with_limit(hint);
        }

        let columns = vec![scan_variable.to_string()];
        Ok(Some((Box::new(op), columns)))
    }

    /// Extracts a simple range predicate (>, <, >=, <=) from an expression.
    ///
    /// Returns `(variable, property, operator, value)` if found.
    pub(super) fn extract_range_predicate(
        &self,
        predicate: &LogicalExpression,
    ) -> Option<(String, String, BinaryOp, Value)> {
        match predicate {
            LogicalExpression::Binary { left, op, right } => {
                match op {
                    BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
                        // Try property on left: n.age > 30
                        if let (
                            LogicalExpression::Property { variable, property },
                            LogicalExpression::Literal(val),
                        ) = (left.as_ref(), right.as_ref())
                        {
                            return Some((variable.clone(), property.clone(), *op, val.clone()));
                        }

                        // Try property on right: 30 < n.age (flip operator)
                        if let (
                            LogicalExpression::Literal(val),
                            LogicalExpression::Property { variable, property },
                        ) = (left.as_ref(), right.as_ref())
                        {
                            let flipped_op = match op {
                                BinaryOp::Lt => BinaryOp::Gt,
                                BinaryOp::Le => BinaryOp::Ge,
                                BinaryOp::Gt => BinaryOp::Lt,
                                BinaryOp::Ge => BinaryOp::Le,
                                _ => return None,
                            };
                            return Some((
                                variable.clone(),
                                property.clone(),
                                flipped_op,
                                val.clone(),
                            ));
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        None
    }

    /// Extracts a BETWEEN pattern from compound predicates.
    ///
    /// Recognizes patterns like:
    /// - `n.age >= 30 AND n.age <= 50`
    /// - `n.age > 30 AND n.age < 50`
    ///
    /// Returns `(variable, property, min_value, max_value, min_inclusive, max_inclusive)`.
    pub(super) fn extract_between_predicate(
        &self,
        predicate: &LogicalExpression,
    ) -> Option<(String, String, Value, Value, bool, bool)> {
        // Must be an AND expression
        let (left, right) = match predicate {
            LogicalExpression::Binary {
                left,
                op: BinaryOp::And,
                right,
            } => (left.as_ref(), right.as_ref()),
            _ => return None,
        };

        // Extract range predicates from both sides
        let left_range = self.extract_range_predicate(left);
        let right_range = self.extract_range_predicate(right);

        let (left_var, left_prop, left_op, left_val) = left_range?;
        let (right_var, right_prop, right_op, right_val) = right_range?;

        // Must be same variable and property
        if left_var != right_var || left_prop != right_prop {
            return None;
        }

        // Determine which is lower bound and which is upper bound
        let (min_val, max_val, min_inc, max_inc) = match (left_op, right_op) {
            // n.x >= min AND n.x <= max
            (BinaryOp::Ge, BinaryOp::Le) => (left_val, right_val, true, true),
            // n.x >= min AND n.x < max
            (BinaryOp::Ge, BinaryOp::Lt) => (left_val, right_val, true, false),
            // n.x > min AND n.x <= max
            (BinaryOp::Gt, BinaryOp::Le) => (left_val, right_val, false, true),
            // n.x > min AND n.x < max
            (BinaryOp::Gt, BinaryOp::Lt) => (left_val, right_val, false, false),
            // Reversed order: n.x <= max AND n.x >= min
            (BinaryOp::Le, BinaryOp::Ge) => (right_val, left_val, true, true),
            // n.x < max AND n.x >= min
            (BinaryOp::Lt, BinaryOp::Ge) => (right_val, left_val, true, false),
            // n.x <= max AND n.x > min
            (BinaryOp::Le, BinaryOp::Gt) => (right_val, left_val, false, true),
            // n.x < max AND n.x > min
            (BinaryOp::Lt, BinaryOp::Gt) => (right_val, left_val, false, false),
            _ => return None,
        };

        Some((left_var, left_prop, min_val, max_val, min_inc, max_inc))
    }
}

/// Extracted vector predicate from a filter expression.
///
/// `remaining` carries the residual predicate that must run in a
/// `FilterOperator` above the `VectorScanOperator`. It is required for
/// strict operators (`>` / `<`): the scan compares inclusively
/// (`>=` / `<=`) when applying `min_similarity` / `max_distance`, so without
/// the residual the boundary row at exactly `threshold` leaks through.
/// Construct via [`Self::inclusive`] or [`Self::strict`] to make that
/// invariant explicit at every call site.
#[cfg(feature = "vector-index")]
pub(super) struct ExtractedVectorPredicate {
    pub(super) property: String,
    /// Variable bound in the property access (e.g. `n` in `n.embedding`).
    /// Validated against the enclosing NodeScan variable before pushdown.
    pub(super) variable: String,
    pub(super) query_vector: LogicalExpression,
    pub(super) metric: super::VectorMetric,
    pub(super) min_similarity: Option<f32>,
    pub(super) max_distance: Option<f32>,
    /// Residual predicate to apply above the scan. `Some` for strict
    /// operators; `None` for inclusive operators (the scan's own
    /// comparison is sufficient).
    pub(super) remaining: Option<LogicalExpression>,
}

#[cfg(feature = "vector-index")]
impl ExtractedVectorPredicate {
    /// Builds a predicate for an inclusive operator (`>=` / `<=`). No
    /// residual is needed because the scan's internal comparison uses the
    /// same inclusive bound.
    fn inclusive(
        property: String,
        variable: String,
        query_vector: LogicalExpression,
        metric: super::VectorMetric,
        min_similarity: Option<f32>,
        max_distance: Option<f32>,
    ) -> Self {
        Self {
            property,
            variable,
            query_vector,
            metric,
            min_similarity,
            max_distance,
            remaining: None,
        }
    }

    /// Builds a predicate for a strict operator (`>` / `<`). The caller
    /// must supply the residual predicate; the planner threads it into a
    /// `FilterOperator` above the `VectorScanOperator` so rows exactly at
    /// the threshold are rejected.
    fn strict(
        property: String,
        variable: String,
        query_vector: LogicalExpression,
        metric: super::VectorMetric,
        min_similarity: Option<f32>,
        max_distance: Option<f32>,
        residual: LogicalExpression,
    ) -> Self {
        Self {
            property,
            variable,
            query_vector,
            metric,
            min_similarity,
            max_distance,
            remaining: Some(residual),
        }
    }
}

#[cfg(feature = "vector-index")]
impl super::Planner {
    /// Tries to push a vector similarity/distance predicate down into a VectorScan operator.
    ///
    /// Recognizes patterns like:
    /// - `cosine_similarity(n.embedding, $qv) > 0.8`
    /// - `euclidean_distance(n.embedding, $qv) < 0.5`
    ///
    /// Returns `Ok(Some((operator, columns)))` if rewritten, `Ok(None)` otherwise.
    pub(super) fn try_plan_filter_with_vector_index(
        &self,
        filter: &super::FilterOp,
    ) -> Result<Option<(Box<dyn Operator>, Vec<String>)>> {
        // Only push down when input is a full label scan (no nested input)
        let LogicalOperator::NodeScan(scan) = filter.input.as_ref() else {
            return Ok(None);
        };
        let Some(ref label) = scan.label else {
            return Ok(None);
        };

        // Extract a vector predicate from the filter expression
        let Some(extracted) = self.extract_vector_predicate(&filter.predicate) else {
            return Ok(None);
        };

        // Ensure the predicate references the same variable as the scan being planned
        if extracted.variable != scan.variable {
            return Ok(None);
        }

        // Check that a vector index exists (or fall through to brute-force eval)
        if !self.store.has_vector_index(label, &extracted.property) {
            return Ok(None);
        }

        // The pushdown requires a query vector resolvable at plan time. If the
        // expression is something like a property access or non-literal, fall
        // through so per-row Filter evaluation handles it.
        if self
            .resolve_vector_literal(&extracted.query_vector)
            .is_err()
        {
            return Ok(None);
        }

        // Build VectorScanOp
        let vector_scan = super::VectorScanOp {
            variable: scan.variable.clone(),
            index_name: Some(format!("{}:{}", label, extracted.property)),
            property: extracted.property.clone(),
            label: Some(label.clone()),
            query_vector: extracted.query_vector.clone(),
            k: None, // threshold mode — filter by similarity/distance, no top-k limit
            metric: Some(extracted.metric),
            min_similarity: extracted.min_similarity,
            max_distance: extracted.max_distance,
            input: None,
        };

        // Plan through the VectorScan path
        let (scan_op, scan_columns) =
            self.plan_operator(&LogicalOperator::VectorScan(vector_scan))?;

        // If there are remaining predicates (AND with non-vector conditions), wrap in filter
        if let Some(remaining) = &extracted.remaining {
            let variable_columns: HashMap<String, usize> = scan_columns
                .iter()
                .enumerate()
                .map(|(i, name)| (name.clone(), i))
                .collect();
            let filter_expr = self.convert_expression(remaining)?;
            let predicate = ExpressionPredicate::new(
                filter_expr,
                variable_columns,
                Arc::clone(&self.store) as Arc<dyn GraphStoreSearch>,
            )
            .with_transaction_context(self.viewing_epoch, self.transaction_id)
            .with_session_context(self.session_context.clone());
            let filter_op = Box::new(FilterOperator::new(scan_op, Box::new(predicate)));
            Ok(Some((filter_op, scan_columns)))
        } else {
            Ok(Some((scan_op, scan_columns)))
        }
    }

    /// Recursively extracts a vector predicate from a (potentially compound) expression.
    pub(super) fn extract_vector_predicate(
        &self,
        expr: &LogicalExpression,
    ) -> Option<ExtractedVectorPredicate> {
        match expr {
            LogicalExpression::Binary { left, op, right } => {
                // Try the left side as a vector function call
                if let LogicalExpression::FunctionCall { name, args, .. } = left.as_ref()
                    && let Some(extracted) = self.try_extract_vector_fn(name, args, *op, right)
                {
                    return Some(extracted);
                }
                // AND: recurse into both sides, accumulating remaining predicates
                if *op == BinaryOp::And {
                    if let Some(mut extracted) = self.extract_vector_predicate(left) {
                        extracted.remaining = Some(match extracted.remaining {
                            Some(prev) => LogicalExpression::Binary {
                                left: Box::new(prev),
                                op: BinaryOp::And,
                                right: right.clone(),
                            },
                            None => *right.clone(),
                        });
                        return Some(extracted);
                    }
                    if let Some(mut extracted) = self.extract_vector_predicate(right) {
                        extracted.remaining = Some(match extracted.remaining {
                            Some(prev) => LogicalExpression::Binary {
                                left: left.clone(),
                                op: BinaryOp::And,
                                right: Box::new(prev),
                            },
                            None => *left.clone(),
                        });
                        return Some(extracted);
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Tries to extract a vector function call predicate.
    ///
    /// Recognizes: `cosine_similarity(n.prop, vec) > threshold`
    /// and the distance variants: `euclidean_distance(n.prop, vec) < threshold`
    fn try_extract_vector_fn(
        &self,
        name: &str,
        args: &[LogicalExpression],
        op: BinaryOp,
        threshold: &LogicalExpression,
    ) -> Option<ExtractedVectorPredicate> {
        if args.len() != 2 {
            return None;
        }

        let (metric, is_similarity) = match name {
            "cosine_similarity" => (super::VectorMetric::Cosine, true),
            "euclidean_distance" => (super::VectorMetric::Euclidean, false),
            "manhattan_distance" => (super::VectorMetric::Manhattan, false),
            // dot_product is intentionally not pushed down: the physical
            // VectorScanOperator's apply_filters only honors min_similarity
            // when metric == Cosine, so a pushed DotProduct threshold would
            // be silently dropped, returning unfiltered results. Fall
            // through to per-row evaluation instead.
            _ => return None,
        };

        // Both strict (>, <) and inclusive (>=, <=) operators push down. The
        // index-level threshold is applied inclusively (apply_filters uses
        // >= / <=), so for strict operators we attach a residual filter that
        // re-applies the strict comparison to exclude boundary rows (scores
        // exactly at the threshold).
        let (valid_op, is_strict) = if is_similarity {
            match op {
                BinaryOp::Gt => (true, true),
                BinaryOp::Ge => (true, false),
                _ => (false, false),
            }
        } else {
            match op {
                BinaryOp::Lt => (true, true),
                BinaryOp::Le => (true, false),
                _ => (false, false),
            }
        };
        if !valid_op {
            return None;
        }

        // One arg must be a property access, the other a vector literal/parameter
        let (variable, property, query_vector) =
            if let LogicalExpression::Property { variable, property } = &args[0] {
                (variable.clone(), property.clone(), args[1].clone())
            } else if let LogicalExpression::Property { variable, property } = &args[1] {
                (variable.clone(), property.clone(), args[0].clone())
            } else {
                return None;
            };

        // Extract threshold value; f64→f32 precision loss is intentional (vectors are f32).
        #[allow(clippy::cast_possible_truncation)]
        let threshold_val = match threshold {
            LogicalExpression::Literal(Value::Float64(v)) => *v as f32,
            LogicalExpression::Literal(Value::Int64(v)) => *v as f32,
            _ => return None,
        };

        let (min_similarity, max_distance) = if is_similarity {
            (Some(threshold_val), None)
        } else {
            (None, Some(threshold_val))
        };

        // For strict ops, reconstruct the original predicate as the residual
        // filter. The planner threads it into a FilterOperator above the
        // VectorScan (see `plan_filter_with_vector_pushdown`) to strip
        // boundary rows that the scan's inclusive comparison would let through.
        if is_strict {
            let residual = LogicalExpression::Binary {
                left: Box::new(LogicalExpression::FunctionCall {
                    name: name.to_string(),
                    args: args.to_vec(),
                    distinct: false,
                }),
                op,
                right: Box::new(threshold.clone()),
            };
            Some(ExtractedVectorPredicate::strict(
                property,
                variable,
                query_vector,
                metric,
                min_similarity,
                max_distance,
                residual,
            ))
        } else {
            Some(ExtractedVectorPredicate::inclusive(
                property,
                variable,
                query_vector,
                metric,
                min_similarity,
                max_distance,
            ))
        }
    }
}
