//! Hybrid (text + vector) predicate pushdown into index scan operators.
//!
//! Extracted from filter.rs to keep that file focused on general filter planning.
//! All functions here are `impl super::Planner` methods; Rust merges impl blocks
//! within the same module, so they share visibility with the rest of the planner.

#[cfg(feature = "text-index")]
use super::{
    Arc, BinaryOp, ExpressionPredicate, FilterOperator, GraphStoreSearch, HashMap,
    LogicalExpression, LogicalOperator, Operator, Result, Value,
};

#[cfg(all(feature = "vector-index", feature = "text-index"))]
use super::{FilterOp, HashJoinOperator, PhysicalJoinType};

// ============================================================================
// Text predicate extraction and pushdown
// ============================================================================

/// Extracted text predicate from a filter expression.
#[cfg(feature = "text-index")]
pub(super) struct ExtractedTextPredicate {
    pub(super) property: String,
    /// Variable bound in the property access (e.g. `n` in `n.body`).
    /// Validated against the enclosing NodeScan variable before pushdown.
    pub(super) variable: String,
    pub(super) query_expr: LogicalExpression,
    pub(super) threshold: f64,
    pub(super) remaining: Option<LogicalExpression>,
}

#[cfg(feature = "text-index")]
impl super::Planner {
    /// Tries to push a text search predicate down into a `TextScan` operator.
    ///
    /// Recognizes patterns like:
    /// - `text_score(n.body, "search terms") > 0.5`
    /// - `text_match(n.body, "search terms")`  (standalone boolean)
    ///
    /// Falls through to per-row evaluation when the text index is absent (D1).
    ///
    /// Returns `Ok(Some(...))` if rewritten, `Ok(None)` to fall through.
    pub(super) fn try_plan_filter_with_text_index(
        &self,
        filter: &super::FilterOp,
    ) -> Result<Option<(Box<dyn Operator>, Vec<String>)>> {
        // Only push down when input is a full label scan (no nested input):
        // the search finds the nodes alone, so a scan per input row (a later
        // MATCH, an UNWIND) checks the predicate per row instead.
        let LogicalOperator::NodeScan(scan) = filter.input.as_ref() else {
            return Ok(None);
        };
        if scan.input.is_some() {
            return Ok(None);
        }
        let Some(ref label) = scan.label else {
            return Ok(None);
        };

        // Extract a text predicate from the filter expression
        let Some(extracted) = self.extract_text_predicate(&filter.predicate) else {
            return Ok(None);
        };

        // Ensure the predicate references the same variable as the scan being planned
        if extracted.variable != scan.variable {
            return Ok(None);
        }

        // No text index: fall through to per-row evaluation, same as vector behavior.
        // text_score returns 0.0 and text_match returns false for every row.
        if !self.store.has_text_index(label, &extracted.property) {
            return Ok(None);
        }

        // Build TextScanOp. Always project the score column so downstream
        // RETURN/ORDER BY expressions can reference it instead of recomputing.
        let text_scan = super::TextScanOp {
            variable: scan.variable.clone(),
            property: extracted.property.clone(),
            label: label.clone(),
            query: extracted.query_expr.clone(),
            k: None,
            threshold: Some(extracted.threshold),
            score_column: Some(super::project::text_score_column_name(
                &scan.variable,
                &extracted.property,
                &extracted.query_expr,
            )),
        };

        // Plan through the TextScan path
        let (scan_op, scan_columns) = self.plan_operator(&LogicalOperator::TextScan(text_scan))?;

        // If there are remaining predicates (AND with non-text conditions), wrap in filter
        if let Some(remaining) = &extracted.remaining {
            let filter_op = self.wrap_with_remaining_filter(scan_op, &scan_columns, remaining)?;
            Ok(Some((filter_op, scan_columns)))
        } else {
            Ok(Some((scan_op, scan_columns)))
        }
    }

    /// Recursively extracts a text predicate from a (potentially compound) expression.
    pub(super) fn extract_text_predicate(
        &self,
        expr: &LogicalExpression,
    ) -> Option<ExtractedTextPredicate> {
        match expr {
            LogicalExpression::Binary { left, op, right } => {
                // text_score(n.prop, "query") > threshold
                if let LogicalExpression::FunctionCall { name, args, .. } = left.as_ref()
                    && name == "text_score"
                    && matches!(op, BinaryOp::Gt | BinaryOp::Ge)
                    && let Some(extracted) = self.try_extract_text_fn(args, right)
                {
                    return Some(extracted);
                }
                // AND: recurse into both sides, accumulating remaining predicates
                if *op == BinaryOp::And {
                    // Try left as text, right as remaining
                    if let Some(mut extracted) = self.extract_text_predicate(left) {
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
                    // Try right as text, left as remaining
                    if let Some(mut extracted) = self.extract_text_predicate(right) {
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
            // text_match(n.prop, "query") — standalone boolean (score > 0.0)
            LogicalExpression::FunctionCall { name, args, .. } if name == "text_match" => {
                self.try_extract_text_fn(args, &LogicalExpression::Literal(Value::Float64(0.0)))
            }
            _ => None,
        }
    }

    /// Tries to extract the property, variable, query expression, and threshold
    /// from a `text_score` or `text_match` argument list.
    fn try_extract_text_fn(
        &self,
        args: &[LogicalExpression],
        threshold_expr: &LogicalExpression,
    ) -> Option<ExtractedTextPredicate> {
        if args.len() != 2 {
            return None;
        }

        let LogicalExpression::Property { variable, property } = &args[0] else {
            return None;
        };

        let threshold = match threshold_expr {
            LogicalExpression::Literal(Value::Float64(v)) => *v,
            LogicalExpression::Literal(Value::Int64(v)) => *v as f64,
            _ => return None,
        };

        Some(ExtractedTextPredicate {
            property: property.clone(),
            variable: variable.clone(),
            query_expr: args[1].clone(),
            threshold,
            remaining: None,
        })
    }

    /// Wraps `op` in a `FilterOperator` that evaluates `remaining` over `columns`.
    ///
    /// Shared across every hybrid pushdown path where an index scan consumes the
    /// bulk of the predicate and a residual scalar condition has to be re-applied
    /// above it. Centralizes column indexing, transaction/session context
    /// attachment, and the `FilterOperator` boxing so the call sites stay terse
    /// and any future change to `ExpressionPredicate` wiring lands in one place.
    fn wrap_with_remaining_filter(
        &self,
        op: Box<dyn Operator>,
        columns: &[String],
        remaining: &LogicalExpression,
    ) -> Result<Box<dyn Operator>> {
        let variable_columns: HashMap<String, usize> = columns
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
        Ok(Box::new(FilterOperator::new(op, Box::new(predicate))))
    }
}

// ============================================================================
// Compound hybrid (vector AND/OR text) pushdown
// ============================================================================

#[cfg(all(feature = "vector-index", feature = "text-index"))]
impl super::Planner {
    /// Tries to plan a filter that contains BOTH a vector predicate and a text predicate.
    ///
    /// When both predicates are present, runs both index scans independently and
    /// hash-joins the results: intersect (Inner join) for AND, union (Full join) for OR.
    /// Both scores are projected as columns for downstream use.
    ///
    /// For AND: extractors recurse into the AND tree (finding vector/text anywhere).
    /// For OR: extracts from each OR operand independently, requiring each side
    /// to be a pure vector or text predicate (no mixed scalar conditions).
    ///
    /// Falls through when only one predicate is present or an index is missing (D1).
    pub(super) fn try_plan_filter_compound_hybrid(
        &self,
        filter: &FilterOp,
    ) -> Result<Option<(Box<dyn Operator>, Vec<String>)>> {
        // A full label scan only: a scan per input row checks the predicate
        // per row (see `try_plan_filter_with_text_index`).
        let LogicalOperator::NodeScan(scan) = filter.input.as_ref() else {
            return Ok(None);
        };
        if scan.input.is_some() {
            return Ok(None);
        }
        let Some(ref label) = scan.label else {
            return Ok(None);
        };

        // Extract vector and text predicates. Strategy depends on AND vs OR.
        let (vector_pred, text_pred, is_or) = if let LogicalExpression::Binary {
            left,
            op: BinaryOp::Or,
            right,
        } = &filter.predicate
        {
            // OR: extract from each operand independently.
            // Try vector-left + text-right, then vector-right + text-left.
            let result = self
                .extract_vector_predicate(left)
                .and_then(|v| self.extract_text_predicate(right).map(|t| (v, t)))
                .or_else(|| {
                    self.extract_vector_predicate(right)
                        .and_then(|v| self.extract_text_predicate(left).map(|t| (v, t)))
                });
            let Some((v, t)) = result else {
                return Ok(None);
            };
            (v, t, true)
        } else {
            // AND: extractors recurse into the AND tree (existing behavior).
            let vector = self.extract_vector_predicate(&filter.predicate);
            let text = self.extract_text_predicate(&filter.predicate);
            match (vector, text) {
                (Some(v), Some(t)) => (v, t, false),
                _ => return Ok(None),
            }
        };

        // Validate both reference the scan variable
        if vector_pred.variable != scan.variable || text_pred.variable != scan.variable {
            return Ok(None);
        }

        // If either index is missing, fall through to per-row evaluation (D1).
        if !self.store.has_vector_index(label, &vector_pred.property) {
            return Ok(None);
        }
        if !self.store.has_text_index(label, &text_pred.property) {
            return Ok(None);
        }

        // Pushdown needs a resolvable query vector. Fall through otherwise.
        if self
            .resolve_vector_literal(&vector_pred.query_vector)
            .is_err()
        {
            return Ok(None);
        }

        // Build VectorScanOp (threshold mode, return all candidates above threshold).
        let vector_scan_op = LogicalOperator::VectorScan(super::VectorScanOp {
            variable: scan.variable.clone(),
            index_name: Some(format!("{}:{}", label, vector_pred.property)),
            property: vector_pred.property.clone(),
            label: Some(label.clone()),
            query_vector: vector_pred.query_vector.clone(),
            k: None, // threshold mode
            metric: Some(vector_pred.metric),
            min_similarity: vector_pred.min_similarity,
            max_distance: vector_pred.max_distance,
            input: None,
        });

        // Build TextScanOp (threshold mode, always project score column).
        let text_scan_op = LogicalOperator::TextScan(super::TextScanOp {
            variable: scan.variable.clone(),
            property: text_pred.property.clone(),
            label: label.clone(),
            query: text_pred.query_expr.clone(),
            k: None,
            threshold: Some(text_pred.threshold),
            score_column: Some(super::project::text_score_column_name(
                &scan.variable,
                &text_pred.property,
                &text_pred.query_expr,
            )),
        });

        let (left_op, left_cols) = self.plan_operator(&vector_scan_op)?;
        let (right_op, right_cols) = self.plan_operator(&text_scan_op)?;

        // For OR with scalar remainders on either branch, wrap that branch
        // in a filter before the join. E.g., for
        //   cosine_similarity(...) > 0.8 OR (text_match(...) AND published = true)
        // the text branch gets a Filter(published = true) around its TextScan.
        let left_op = if let Some(remaining) = &vector_pred.remaining {
            self.wrap_with_remaining_filter(left_op, &left_cols, remaining)?
        } else {
            left_op
        };

        let right_op = if let Some(remaining) = &text_pred.remaining {
            self.wrap_with_remaining_filter(right_op, &right_cols, remaining)?
        } else {
            right_op
        };

        // Determine join type: AND → Inner (intersect), OR → Full (union)
        let join_type = if is_or {
            PhysicalJoinType::Full
        } else {
            PhysicalJoinType::Inner
        };

        // HashJoin outputs all left columns + all right columns.
        // Right side: [variable, _tscore_variable]. The variable column is a duplicate
        // of left column 0 and must be projected out.
        let mut all_cols = left_cols.clone();
        all_cols.extend(right_cols.iter().cloned());
        let all_schema = self.derive_schema_from_columns(&all_cols);

        let join_op: Box<dyn Operator> = Box::new(HashJoinOperator::new(
            left_op,
            right_op,
            vec![0], // probe key: column 0 (NodeId / variable)
            vec![0], // build key: column 0 (NodeId / variable)
            join_type,
            all_schema,
        ));

        // Project out the duplicate node-variable column from the right side.
        // left_cols = [variable, _vscore_variable]  (indices 0, 1)
        // right_cols = [variable, _tscore_variable]  (indices 2, 3 in all_cols)
        // Output: [variable, _vscore_variable, _tscore_variable]
        //
        // For OR (Full join): right-only rows have NULL in left column 0.
        // Use COALESCE(left_var, right_var) so the variable is always non-NULL.
        let left_count = left_cols.len();
        let mut proj_exprs: Vec<super::ProjectExpr> = Vec::new();
        let mut output_cols: Vec<String> = Vec::new();

        for (i, col) in left_cols.iter().enumerate() {
            if i == 0 && is_or {
                // For OR joins, coalesce the variable from both sides
                proj_exprs.push(super::ProjectExpr::Coalesce {
                    first: 0,
                    second: left_count,
                });
            } else {
                proj_exprs.push(super::ProjectExpr::Column(i));
            }
            output_cols.push(col.clone());
        }
        for (i, col) in right_cols.iter().enumerate() {
            if i == 0 {
                continue; // Duplicate variable column (merged via Coalesce above)
            }
            proj_exprs.push(super::ProjectExpr::Column(left_count + i));
            output_cols.push(col.clone());
        }

        let proj_schema = self.derive_schema_from_columns(&output_cols);
        let proj_op: Box<dyn Operator> = Box::new(super::ProjectOperator::new(
            join_op,
            proj_exprs,
            proj_schema,
        ));

        // Apply any remaining scalar predicates (parts of the expression that are
        // neither vector nor text, e.g. an extra AND condition)
        let scalar_remaining = self.extract_scalar_remaining(&filter.predicate);
        if let Some(remaining) = scalar_remaining {
            let filter_op = self.wrap_with_remaining_filter(proj_op, &output_cols, &remaining)?;
            Ok(Some((filter_op, output_cols)))
        } else {
            Ok(Some((proj_op, output_cols)))
        }
    }

    /// Extracts the parts of a predicate that are neither a vector nor a text sub-predicate.
    ///
    /// Recursively walks AND trees, keeping only scalar (non-index) conditions.
    /// Used to find conditions that must be applied after the hash join.
    pub(super) fn extract_scalar_remaining(
        &self,
        expr: &LogicalExpression,
    ) -> Option<LogicalExpression> {
        match expr {
            LogicalExpression::Binary {
                left,
                op: BinaryOp::And,
                right,
            } => {
                let left_scalar = self.extract_scalar_remaining(left);
                let right_scalar = self.extract_scalar_remaining(right);

                match (left_scalar, right_scalar) {
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
                op: BinaryOp::Or,
                right,
            } => {
                // If both sides of OR are index predicates, the full-join
                // already computes the union — no scalar filter needed.
                let left_remaining = self.extract_scalar_remaining(left);
                let right_remaining = self.extract_scalar_remaining(right);
                match (left_remaining, right_remaining) {
                    (None, None) => None,
                    _ => Some(expr.clone()),
                }
            }
            // Leaf node: check if it's a vector or text predicate
            other => {
                let is_vector = self.extract_vector_predicate(other).is_some();
                let is_text = self.extract_text_predicate(other).is_some();
                if is_vector || is_text {
                    None // Handled by index scan, drop it
                } else {
                    Some(other.clone()) // Scalar predicate, keep it
                }
            }
        }
    }
}

// ============================================================================
// White-box unit tests
//
// These tests exercise branches in the text and hybrid predicate extractors
// that are difficult to reach through GQL queries alone: arity and type guards
// inside `try_extract_text_fn`, the multi-AND accumulator path inside
// `extract_text_predicate`, the scan-variable and label early returns of
// `try_plan_filter_with_text_index`, and the leaf/combine arms of
// `extract_scalar_remaining`. End-to-end pushdown is covered separately by
// `tests/hybrid_pushdown_coverage.rs` and `tests/hybrid_query.rs`.
// ============================================================================

#[cfg(all(test, feature = "text-index"))]
mod text_extract_tests {
    use super::super::{
        Arc, BinaryOp, FilterOp, GraphStoreSearch, LogicalExpression, LogicalOperator, NodeScanOp,
        Planner, Value,
    };
    use grafeo_core::graph::lpg::LpgStore;

    fn test_planner() -> Planner {
        let store = Arc::new(LpgStore::new().unwrap());
        let article = store.create_node(&["Article"]);
        store.set_node_property(article, "body", Value::String("rust database".into()));
        Planner::new(store as Arc<dyn GraphStoreSearch>)
    }

    fn property(var: &str, name: &str) -> LogicalExpression {
        LogicalExpression::Property {
            variable: var.to_string(),
            property: name.to_string(),
        }
    }

    fn literal_string(s: &str) -> LogicalExpression {
        LogicalExpression::Literal(Value::String(s.into()))
    }

    fn text_score_call(var: &str, prop: &str, query: &str) -> LogicalExpression {
        LogicalExpression::FunctionCall {
            name: "text_score".to_string(),
            args: vec![property(var, prop), literal_string(query)],
            distinct: false,
        }
    }

    fn text_match_call(var: &str, prop: &str, query: &str) -> LogicalExpression {
        LogicalExpression::FunctionCall {
            name: "text_match".to_string(),
            args: vec![property(var, prop), literal_string(query)],
            distinct: false,
        }
    }

    fn binary(
        left: LogicalExpression,
        op: BinaryOp,
        right: LogicalExpression,
    ) -> LogicalExpression {
        LogicalExpression::Binary {
            left: Box::new(left),
            op,
            right: Box::new(right),
        }
    }

    // ------------------------------------------------------------------
    // try_extract_text_fn guards
    // ------------------------------------------------------------------

    #[test]
    fn extract_text_predicate_rejects_wrong_arg_count() {
        // A text_score call with a single argument must not match; the arity
        // guard in try_extract_text_fn returns None.
        let planner = test_planner();
        let single_arg_call = LogicalExpression::FunctionCall {
            name: "text_score".to_string(),
            args: vec![property("doc", "body")],
            distinct: false,
        };
        let expr = binary(
            single_arg_call,
            BinaryOp::Gt,
            LogicalExpression::Literal(Value::Float64(0.5)),
        );
        assert!(planner.extract_text_predicate(&expr).is_none());
    }

    #[test]
    fn extract_text_predicate_rejects_non_property_first_arg() {
        // First argument must be a Property; a literal here must fail the
        // let-else in try_extract_text_fn.
        let planner = test_planner();
        let call = LogicalExpression::FunctionCall {
            name: "text_score".to_string(),
            args: vec![literal_string("not a property"), literal_string("query")],
            distinct: false,
        };
        let expr = binary(
            call,
            BinaryOp::Gt,
            LogicalExpression::Literal(Value::Float64(0.5)),
        );
        assert!(planner.extract_text_predicate(&expr).is_none());
    }

    #[test]
    fn extract_text_predicate_accepts_ge_operator() {
        // text_score(...) >= threshold must match; covers the Ge arm of the
        // `matches!(op, BinaryOp::Gt | BinaryOp::Ge)` guard.
        let planner = test_planner();
        let expr = binary(
            text_score_call("doc", "body", "vincent"),
            BinaryOp::Ge,
            LogicalExpression::Literal(Value::Float64(0.25)),
        );
        let extracted = planner.extract_text_predicate(&expr).unwrap();
        assert_eq!(extracted.property, "body");
        assert_eq!(extracted.variable, "doc");
        assert!((extracted.threshold - 0.25).abs() < f64::EPSILON);
        assert!(extracted.remaining.is_none());
    }

    #[test]
    fn extract_text_predicate_accepts_int64_threshold() {
        // Integer threshold must be accepted and coerced to f64.
        let planner = test_planner();
        let expr = binary(
            text_score_call("doc", "body", "jules"),
            BinaryOp::Gt,
            LogicalExpression::Literal(Value::Int64(1)),
        );
        let extracted = planner.extract_text_predicate(&expr).unwrap();
        assert!((extracted.threshold - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn extract_text_predicate_rejects_non_literal_threshold() {
        // Non-literal thresholds (e.g., another property) must not push down.
        let planner = test_planner();
        let expr = binary(
            text_score_call("doc", "body", "mia"),
            BinaryOp::Gt,
            property("doc", "score"),
        );
        assert!(planner.extract_text_predicate(&expr).is_none());
    }

    #[test]
    fn extract_text_predicate_rejects_lt_operator() {
        // Only Gt and Ge match; a Lt must fall through all arms and return
        // None (not an AND, not a standalone text_match, and not > / >=).
        let planner = test_planner();
        let expr = binary(
            text_score_call("doc", "body", "butch"),
            BinaryOp::Lt,
            LogicalExpression::Literal(Value::Float64(0.9)),
        );
        assert!(planner.extract_text_predicate(&expr).is_none());
    }

    // ------------------------------------------------------------------
    // extract_text_predicate match arms and recursion
    // ------------------------------------------------------------------

    #[test]
    fn extract_text_predicate_text_match_uses_zero_threshold() {
        // Standalone text_match(...) must produce threshold = 0.0.
        let planner = test_planner();
        let expr = text_match_call("doc", "body", "django");
        let extracted = planner.extract_text_predicate(&expr).unwrap();
        assert!((extracted.threshold - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn extract_text_predicate_fallthrough_on_plain_variable() {
        // A bare variable reference matches neither the Binary nor the
        // FunctionCall-text_match arms. The _ => None fallthrough fires.
        let planner = test_planner();
        let expr = LogicalExpression::Variable("doc".to_string());
        assert!(planner.extract_text_predicate(&expr).is_none());
    }

    #[test]
    fn extract_text_predicate_and_accumulates_prev_remaining() {
        // (text_score(doc.body, 'x') > 0.0 AND published = true) AND live = true
        // The inner AND produces an Extracted with remaining = published=true.
        // The outer AND then wraps that "prev" remainder alongside live=true,
        // hitting the `Some(prev)` arm of the accumulator.
        let planner = test_planner();
        let published_true = binary(
            property("doc", "published"),
            BinaryOp::Eq,
            LogicalExpression::Literal(Value::Bool(true)),
        );
        let live_true = binary(
            property("doc", "live"),
            BinaryOp::Eq,
            LogicalExpression::Literal(Value::Bool(true)),
        );
        let text_gt = binary(
            text_score_call("doc", "body", "shosanna"),
            BinaryOp::Gt,
            LogicalExpression::Literal(Value::Float64(0.0)),
        );
        let inner_and = binary(text_gt, BinaryOp::And, published_true);
        let outer_and = binary(inner_and, BinaryOp::And, live_true);
        let extracted = planner.extract_text_predicate(&outer_and).unwrap();
        // The accumulated remainder must be an AND of both scalar predicates.
        match extracted.remaining.expect("remaining must be set") {
            LogicalExpression::Binary { op, .. } => assert_eq!(op, BinaryOp::And),
            other => panic!("expected AND remainder, got {other:?}"),
        }
    }

    #[test]
    fn extract_text_predicate_and_accumulates_prev_remaining_right_branch() {
        // Mirror of the above, but the text predicate lives on the right-hand
        // side of both ANDs, forcing the right-recursive branch to fire the
        // `Some(prev)` arm for its accumulator.
        let planner = test_planner();
        let scalar_left = binary(
            property("doc", "active"),
            BinaryOp::Eq,
            LogicalExpression::Literal(Value::Bool(true)),
        );
        let scalar_outer_left = binary(
            property("doc", "live"),
            BinaryOp::Eq,
            LogicalExpression::Literal(Value::Bool(true)),
        );
        let text_gt = binary(
            text_score_call("doc", "body", "hans"),
            BinaryOp::Gt,
            LogicalExpression::Literal(Value::Float64(0.0)),
        );
        let inner_and = binary(scalar_left, BinaryOp::And, text_gt);
        let outer_and = binary(scalar_outer_left, BinaryOp::And, inner_and);
        let extracted = planner.extract_text_predicate(&outer_and).unwrap();
        match extracted.remaining.expect("remaining must be set") {
            LogicalExpression::Binary { op, .. } => assert_eq!(op, BinaryOp::And),
            other => panic!("expected AND remainder, got {other:?}"),
        }
    }

    #[test]
    fn extract_text_predicate_and_with_no_text_on_either_side() {
        // An AND tree that does not contain text_score or text_match anywhere
        // must return None after both recursive attempts.
        let planner = test_planner();
        let left = binary(
            property("doc", "age"),
            BinaryOp::Gt,
            LogicalExpression::Literal(Value::Int64(18)),
        );
        let right = binary(
            property("doc", "city"),
            BinaryOp::Eq,
            literal_string("Amsterdam"),
        );
        let expr = binary(left, BinaryOp::And, right);
        assert!(planner.extract_text_predicate(&expr).is_none());
    }

    // ------------------------------------------------------------------
    // try_plan_filter_with_text_index early returns
    // ------------------------------------------------------------------

    #[test]
    fn try_plan_filter_with_text_index_rejects_non_nodescan_input() {
        // Filter input is a nested NodeScan wrapped in a Filter rather than a
        // plain NodeScan. The outer try_plan_filter_with_text_index sees a
        // non-NodeScan input and returns Ok(None) without extracting anything.
        let planner = test_planner();
        let inner_filter = FilterOp {
            predicate: LogicalExpression::Literal(Value::Bool(true)),
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "doc".to_string(),
                label: Some("Article".to_string()),
                input: None,
            })),
            pushdown_hint: None,
        };
        let outer_filter = FilterOp {
            predicate: binary(
                text_score_call("doc", "body", "beatrix"),
                BinaryOp::Gt,
                LogicalExpression::Literal(Value::Float64(0.1)),
            ),
            input: Box::new(LogicalOperator::Filter(inner_filter)),
            pushdown_hint: None,
        };
        let planned = planner
            .try_plan_filter_with_text_index(&outer_filter)
            .unwrap();
        assert!(planned.is_none());
    }

    #[test]
    fn try_plan_filter_with_text_index_rejects_label_less_scan() {
        // A NodeScan without a label cannot pushdown: the early return on
        // `scan.label` being None fires.
        let planner = test_planner();
        let filter = FilterOp {
            predicate: binary(
                text_score_call("doc", "body", "vincent"),
                BinaryOp::Gt,
                LogicalExpression::Literal(Value::Float64(0.1)),
            ),
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "doc".to_string(),
                label: None,
                input: None,
            })),
            pushdown_hint: None,
        };
        let planned = planner.try_plan_filter_with_text_index(&filter).unwrap();
        assert!(planned.is_none());
    }

    #[test]
    fn try_plan_filter_with_text_index_rejects_variable_mismatch() {
        // Predicate references a variable `m` while the NodeScan binds `doc`.
        // The variable guard rejects the pushdown.
        let planner = test_planner();
        let filter = FilterOp {
            predicate: binary(
                text_score_call("m", "body", "prague"),
                BinaryOp::Gt,
                LogicalExpression::Literal(Value::Float64(0.1)),
            ),
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "doc".to_string(),
                label: Some("Article".to_string()),
                input: None,
            })),
            pushdown_hint: None,
        };
        let planned = planner.try_plan_filter_with_text_index(&filter).unwrap();
        assert!(planned.is_none());
    }

    #[test]
    fn try_plan_filter_with_text_index_rejects_missing_index() {
        // An Article store without a text index on `body`: the planner must
        // fall through (per-row evaluation handles text_score/text_match).
        let planner = test_planner();
        let filter = FilterOp {
            predicate: binary(
                text_score_call("doc", "body", "django"),
                BinaryOp::Gt,
                LogicalExpression::Literal(Value::Float64(0.1)),
            ),
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "doc".to_string(),
                label: Some("Article".to_string()),
                input: None,
            })),
            pushdown_hint: None,
        };
        let planned = planner.try_plan_filter_with_text_index(&filter).unwrap();
        assert!(planned.is_none());
    }

    #[test]
    fn try_plan_filter_with_text_index_rejects_no_text_predicate() {
        // Filter that is a pure scalar (no text_score / text_match) returns
        // None after extract_text_predicate yields nothing.
        let planner = test_planner();
        let filter = FilterOp {
            predicate: binary(
                property("doc", "age"),
                BinaryOp::Gt,
                LogicalExpression::Literal(Value::Int64(25)),
            ),
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "doc".to_string(),
                label: Some("Article".to_string()),
                input: None,
            })),
            pushdown_hint: None,
        };
        let planned = planner.try_plan_filter_with_text_index(&filter).unwrap();
        assert!(planned.is_none());
    }
}

// ============================================================================
// White-box tests for the compound (text AND/OR vector) hybrid planner
// ============================================================================

#[cfg(all(test, feature = "vector-index", feature = "text-index"))]
mod compound_hybrid_tests {
    use super::super::{
        Arc, BinaryOp, FilterOp, GraphStoreSearch, LogicalExpression, LogicalOperator, NodeScanOp,
        Planner, Value,
    };
    use grafeo_core::graph::lpg::LpgStore;

    fn planner_with_article() -> Planner {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["Article"]);
        store.set_node_property(a, "body", Value::String("vincent and jules".into()));
        store.set_node_property(a, "embedding", Value::Vector(vec![0.1f32, 0.9, 0.0].into()));
        Planner::new(store as Arc<dyn GraphStoreSearch>)
    }

    fn property(var: &str, name: &str) -> LogicalExpression {
        LogicalExpression::Property {
            variable: var.to_string(),
            property: name.to_string(),
        }
    }

    fn literal_vec(values: &[f32]) -> LogicalExpression {
        LogicalExpression::List(
            values
                .iter()
                .map(|v| LogicalExpression::Literal(Value::Float64(f64::from(*v))))
                .collect(),
        )
    }

    fn literal_string(s: &str) -> LogicalExpression {
        LogicalExpression::Literal(Value::String(s.into()))
    }

    fn text_score_call(var: &str, prop: &str, query: &str) -> LogicalExpression {
        LogicalExpression::FunctionCall {
            name: "text_score".to_string(),
            args: vec![property(var, prop), literal_string(query)],
            distinct: false,
        }
    }

    fn cosine_call(var: &str, prop: &str, query_vec: &[f32]) -> LogicalExpression {
        LogicalExpression::FunctionCall {
            name: "cosine_similarity".to_string(),
            args: vec![property(var, prop), literal_vec(query_vec)],
            distinct: false,
        }
    }

    fn binary(
        left: LogicalExpression,
        op: BinaryOp,
        right: LogicalExpression,
    ) -> LogicalExpression {
        LogicalExpression::Binary {
            left: Box::new(left),
            op,
            right: Box::new(right),
        }
    }

    fn compound_and() -> LogicalExpression {
        let vec_pred = binary(
            cosine_call("doc", "embedding", &[0.1, 0.9, 0.0]),
            BinaryOp::Ge,
            LogicalExpression::Literal(Value::Float64(0.5)),
        );
        let text_pred = binary(
            text_score_call("doc", "body", "vincent"),
            BinaryOp::Gt,
            LogicalExpression::Literal(Value::Float64(0.0)),
        );
        binary(vec_pred, BinaryOp::And, text_pred)
    }

    // ------------------------------------------------------------------
    // try_plan_filter_compound_hybrid early returns
    // ------------------------------------------------------------------

    #[test]
    fn compound_hybrid_rejects_non_nodescan_input() {
        // Compound hybrid predicate over a nested Filter input: the outer
        // planner must reject because the immediate child is not a NodeScan.
        let planner = planner_with_article();
        let filter = FilterOp {
            predicate: compound_and(),
            input: Box::new(LogicalOperator::Filter(FilterOp {
                predicate: LogicalExpression::Literal(Value::Bool(true)),
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "doc".to_string(),
                    label: Some("Article".to_string()),
                    input: None,
                })),
                pushdown_hint: None,
            })),
            pushdown_hint: None,
        };
        let planned = planner.try_plan_filter_compound_hybrid(&filter).unwrap();
        assert!(planned.is_none());
    }

    #[test]
    fn compound_hybrid_rejects_label_less_scan() {
        // A NodeScan without a label cannot be a hybrid pushdown target.
        let planner = planner_with_article();
        let filter = FilterOp {
            predicate: compound_and(),
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "doc".to_string(),
                label: None,
                input: None,
            })),
            pushdown_hint: None,
        };
        let planned = planner.try_plan_filter_compound_hybrid(&filter).unwrap();
        assert!(planned.is_none());
    }

    #[test]
    fn compound_hybrid_rejects_and_without_both_predicates() {
        // An AND tree with only a text predicate must not be rewritten as a
        // compound hybrid (the AND arm returns None when vector is missing).
        let planner = planner_with_article();
        let predicate = binary(
            binary(
                text_score_call("doc", "body", "shosanna"),
                BinaryOp::Gt,
                LogicalExpression::Literal(Value::Float64(0.0)),
            ),
            BinaryOp::And,
            binary(
                property("doc", "age"),
                BinaryOp::Gt,
                LogicalExpression::Literal(Value::Int64(18)),
            ),
        );
        let filter = FilterOp {
            predicate,
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "doc".to_string(),
                label: Some("Article".to_string()),
                input: None,
            })),
            pushdown_hint: None,
        };
        let planned = planner.try_plan_filter_compound_hybrid(&filter).unwrap();
        assert!(planned.is_none());
    }

    #[test]
    fn compound_hybrid_rejects_variable_mismatch() {
        // Text predicate binds `other`, vector binds `doc`: the variable-mismatch
        // guard on text_pred.variable fires (covers the `text_pred.variable !=
        // scan.variable` branch).
        let planner = planner_with_article();
        let predicate = binary(
            binary(
                cosine_call("doc", "embedding", &[0.1, 0.9, 0.0]),
                BinaryOp::Ge,
                LogicalExpression::Literal(Value::Float64(0.5)),
            ),
            BinaryOp::And,
            binary(
                text_score_call("other", "body", "mia"),
                BinaryOp::Gt,
                LogicalExpression::Literal(Value::Float64(0.0)),
            ),
        );
        let filter = FilterOp {
            predicate,
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "doc".to_string(),
                label: Some("Article".to_string()),
                input: None,
            })),
            pushdown_hint: None,
        };
        let planned = planner.try_plan_filter_compound_hybrid(&filter).unwrap();
        assert!(planned.is_none());
    }

    #[test]
    fn compound_hybrid_rejects_or_without_both_predicates() {
        // Pure OR of two scalar predicates cannot be a hybrid OR pushdown;
        // the OR branch's extractor returns None for both alternatives.
        let planner = planner_with_article();
        let left = binary(
            property("doc", "age"),
            BinaryOp::Gt,
            LogicalExpression::Literal(Value::Int64(18)),
        );
        let right = binary(
            property("doc", "city"),
            BinaryOp::Eq,
            literal_string("Berlin"),
        );
        let filter = FilterOp {
            predicate: binary(left, BinaryOp::Or, right),
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "doc".to_string(),
                label: Some("Article".to_string()),
                input: None,
            })),
            pushdown_hint: None,
        };
        let planned = planner.try_plan_filter_compound_hybrid(&filter).unwrap();
        assert!(planned.is_none());
    }

    #[test]
    fn compound_hybrid_rejects_missing_vector_index() {
        // No vector index is created on the store, so the has_vector_index
        // guard must cause the planner to fall through.
        let planner = planner_with_article();
        let filter = FilterOp {
            predicate: compound_and(),
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "doc".to_string(),
                label: Some("Article".to_string()),
                input: None,
            })),
            pushdown_hint: None,
        };
        let planned = planner.try_plan_filter_compound_hybrid(&filter).unwrap();
        assert!(planned.is_none());
    }

    // ------------------------------------------------------------------
    // extract_scalar_remaining: systematic coverage of all match arms
    // ------------------------------------------------------------------

    #[test]
    fn extract_scalar_remaining_returns_none_for_text_leaf() {
        // Leaf text predicate must be dropped (None) because an index scan
        // handles it.
        let planner = planner_with_article();
        let expr = binary(
            text_score_call("doc", "body", "beatrix"),
            BinaryOp::Gt,
            LogicalExpression::Literal(Value::Float64(0.0)),
        );
        assert!(planner.extract_scalar_remaining(&expr).is_none());
    }

    #[test]
    fn extract_scalar_remaining_returns_none_for_vector_leaf() {
        // Leaf vector predicate must also be dropped.
        let planner = planner_with_article();
        let expr = binary(
            cosine_call("doc", "embedding", &[0.1, 0.9, 0.0]),
            BinaryOp::Ge,
            LogicalExpression::Literal(Value::Float64(0.5)),
        );
        assert!(planner.extract_scalar_remaining(&expr).is_none());
    }

    #[test]
    fn extract_scalar_remaining_keeps_plain_scalar_leaf() {
        // A scalar leaf (age > 18) must be preserved.
        let planner = planner_with_article();
        let expr = binary(
            property("doc", "age"),
            BinaryOp::Gt,
            LogicalExpression::Literal(Value::Int64(18)),
        );
        let remaining = planner.extract_scalar_remaining(&expr);
        assert!(remaining.is_some());
    }

    #[test]
    fn extract_scalar_remaining_and_drops_both_index_predicates() {
        // AND of vector + text: both drop, so the recursion returns None.
        let planner = planner_with_article();
        let expr = compound_and();
        assert!(planner.extract_scalar_remaining(&expr).is_none());
    }

    #[test]
    fn extract_scalar_remaining_and_keeps_left_scalar_only() {
        // AND(scalar, text): right drops, left survives. Covers (Some, None).
        let planner = planner_with_article();
        let scalar = binary(
            property("doc", "age"),
            BinaryOp::Gt,
            LogicalExpression::Literal(Value::Int64(18)),
        );
        let text = binary(
            text_score_call("doc", "body", "hans"),
            BinaryOp::Gt,
            LogicalExpression::Literal(Value::Float64(0.0)),
        );
        let and = binary(scalar, BinaryOp::And, text);
        let remaining = planner.extract_scalar_remaining(&and).unwrap();
        // Must be exactly the scalar, unwrapped from the AND.
        match remaining {
            LogicalExpression::Binary { op, .. } => assert_eq!(op, BinaryOp::Gt),
            other => panic!("expected scalar Binary(Gt), got {other:?}"),
        }
    }

    #[test]
    fn extract_scalar_remaining_and_keeps_right_scalar_only() {
        // Symmetric to above: AND(text, scalar) -> keep scalar. Covers
        // the (None, Some) arm.
        let planner = planner_with_article();
        let text = binary(
            text_score_call("doc", "body", "beatrix"),
            BinaryOp::Gt,
            LogicalExpression::Literal(Value::Float64(0.0)),
        );
        let scalar = binary(
            property("doc", "age"),
            BinaryOp::Gt,
            LogicalExpression::Literal(Value::Int64(18)),
        );
        let and = binary(text, BinaryOp::And, scalar);
        let remaining = planner.extract_scalar_remaining(&and).unwrap();
        match remaining {
            LogicalExpression::Binary { op, .. } => assert_eq!(op, BinaryOp::Gt),
            other => panic!("expected scalar Binary(Gt), got {other:?}"),
        }
    }

    #[test]
    fn extract_scalar_remaining_and_combines_two_scalars() {
        // AND(scalar, scalar) -> combined AND. Covers (Some, Some) arm.
        let planner = planner_with_article();
        let left = binary(
            property("doc", "age"),
            BinaryOp::Gt,
            LogicalExpression::Literal(Value::Int64(18)),
        );
        let right = binary(
            property("doc", "city"),
            BinaryOp::Eq,
            literal_string("Prague"),
        );
        let and = binary(left, BinaryOp::And, right);
        let remaining = planner.extract_scalar_remaining(&and).unwrap();
        match remaining {
            LogicalExpression::Binary { op, .. } => assert_eq!(op, BinaryOp::And),
            other => panic!("expected combined AND, got {other:?}"),
        }
    }

    #[test]
    fn extract_scalar_remaining_or_pure_index_drops_whole_tree() {
        // OR(vector, text): both sides are index predicates with no scalar
        // remainder. extract_scalar_remaining returns None (None, None arm of
        // the OR match).
        let planner = planner_with_article();
        let vec_pred = binary(
            cosine_call("doc", "embedding", &[0.1, 0.9, 0.0]),
            BinaryOp::Ge,
            LogicalExpression::Literal(Value::Float64(0.5)),
        );
        let text_pred = binary(
            text_score_call("doc", "body", "mia"),
            BinaryOp::Gt,
            LogicalExpression::Literal(Value::Float64(0.0)),
        );
        let or = binary(vec_pred, BinaryOp::Or, text_pred);
        assert!(planner.extract_scalar_remaining(&or).is_none());
    }

    #[test]
    fn extract_scalar_remaining_or_with_scalar_keeps_whole_expr() {
        // OR(vector AND scalar, text): the left side has a scalar remainder
        // ("published = true"), so the whole OR must be preserved post-join
        // (match arm that returns Some(expr.clone())).
        let planner = planner_with_article();
        let vec_pred = binary(
            cosine_call("doc", "embedding", &[0.1, 0.9, 0.0]),
            BinaryOp::Ge,
            LogicalExpression::Literal(Value::Float64(0.5)),
        );
        let scalar = binary(
            property("doc", "published"),
            BinaryOp::Eq,
            LogicalExpression::Literal(Value::Bool(true)),
        );
        let left_and = binary(vec_pred, BinaryOp::And, scalar);
        let text_pred = binary(
            text_score_call("doc", "body", "vincent"),
            BinaryOp::Gt,
            LogicalExpression::Literal(Value::Float64(0.0)),
        );
        let or = binary(left_and, BinaryOp::Or, text_pred);
        let remaining = planner.extract_scalar_remaining(&or);
        assert!(remaining.is_some());
        match remaining.unwrap() {
            LogicalExpression::Binary { op, .. } => assert_eq!(op, BinaryOp::Or),
            other => panic!("expected Or preserved whole, got {other:?}"),
        }
    }
}
