//! Expression, literal, and operator translation.

#[allow(clippy::wildcard_imports)]
use super::*;

impl GqlTranslator {
    pub(super) fn translate_expression(&self, expr: &ast::Expression) -> Result<LogicalExpression> {
        match expr {
            ast::Expression::Literal(lit) => Ok(self.translate_literal(lit)),
            ast::Expression::Variable(name) => Ok(LogicalExpression::Variable(name.clone())),
            ast::Expression::Parameter(name) => Ok(LogicalExpression::Parameter(name.clone())),
            ast::Expression::PropertyAccess { variable, property } => {
                Ok(LogicalExpression::Property {
                    variable: variable.clone(),
                    property: property.clone(),
                })
            }
            ast::Expression::Binary { left, op, right } => {
                let left = self.translate_expression(left)?;
                let right = self.translate_expression(right)?;
                let op = self.translate_binary_op(*op);
                Ok(LogicalExpression::Binary {
                    left: Box::new(left),
                    op,
                    right: Box::new(right),
                })
            }
            ast::Expression::Unary { op, operand } => {
                let operand = self.translate_expression(operand)?;
                // Unary positive is identity: just return the operand
                if *op == ast::UnaryOp::Pos {
                    return Ok(operand);
                }
                let op = self.translate_unary_op(*op);
                Ok(LogicalExpression::Unary {
                    op,
                    operand: Box::new(operand),
                })
            }
            ast::Expression::FunctionCall {
                name,
                args,
                distinct,
            } => {
                // Special handling for length() on path variables
                // When length(p) is called where p is a path alias, we convert it
                // to a variable reference to the path length column
                if name.to_lowercase() == "length"
                    && args.len() == 1
                    && let ast::Expression::Variable(var_name) = &args[0]
                {
                    // Check if this looks like a path variable
                    // Path lengths are stored in columns named _path_length_{alias}
                    return Ok(LogicalExpression::Variable(format!(
                        "_path_length_{}",
                        var_name
                    )));
                }

                // NULLIF(a, b) desugars to CASE WHEN a = b THEN NULL ELSE a END
                if name.eq_ignore_ascii_case("nullif") {
                    if args.len() != 2 {
                        return Err(Error::Query(QueryError::new(
                            QueryErrorKind::Semantic,
                            "NULLIF requires exactly 2 arguments",
                        )));
                    }
                    let a = self.translate_expression(&args[0])?;
                    let b = self.translate_expression(&args[1])?;
                    return Ok(LogicalExpression::Case {
                        operand: None,
                        when_clauses: vec![(
                            LogicalExpression::Binary {
                                left: Box::new(a.clone()),
                                op: BinaryOp::Eq,
                                right: Box::new(b),
                            },
                            LogicalExpression::Literal(Value::Null),
                        )],
                        else_clause: Some(Box::new(a)),
                    });
                }

                let args = args
                    .iter()
                    .map(|a| self.translate_expression(a))
                    .collect::<Result<Vec<_>>>()?;
                Ok(LogicalExpression::FunctionCall {
                    name: name.clone(),
                    args,
                    distinct: *distinct,
                })
            }
            ast::Expression::List(items) => {
                let items = items
                    .iter()
                    .map(|i| self.translate_expression(i))
                    .collect::<Result<Vec<_>>>()?;
                Ok(LogicalExpression::List(items))
            }
            ast::Expression::Case {
                input,
                whens,
                else_clause,
            } => {
                let operand = input
                    .as_ref()
                    .map(|e| self.translate_expression(e))
                    .transpose()?
                    .map(Box::new);

                let when_clauses = whens
                    .iter()
                    .map(|(cond, result)| {
                        Ok((
                            self.translate_expression(cond)?,
                            self.translate_expression(result)?,
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?;

                let else_clause = else_clause
                    .as_ref()
                    .map(|e| self.translate_expression(e))
                    .transpose()?
                    .map(Box::new);

                Ok(LogicalExpression::Case {
                    operand,
                    when_clauses,
                    else_clause,
                })
            }
            ast::Expression::Map(entries) => {
                let entries = entries
                    .iter()
                    .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
                    .collect::<Result<Vec<_>>>()?;
                Ok(LogicalExpression::Map(entries))
            }
            ast::Expression::ExistsSubquery { query } => {
                // Translate inner query to logical operator
                let inner_plan = self.translate_subquery_to_operator(query)?;
                Ok(LogicalExpression::ExistsSubquery(Box::new(inner_plan)))
            }
            ast::Expression::CountSubquery { query } => {
                let inner_plan = self.translate_subquery_to_operator(query)?;
                Ok(LogicalExpression::CountSubquery(Box::new(inner_plan)))
            }
            ast::Expression::IndexAccess { base, index } => {
                let base_expr = self.translate_expression(base)?;
                let index_expr = self.translate_expression(index)?;
                Ok(LogicalExpression::IndexAccess {
                    base: Box::new(base_expr),
                    index: Box::new(index_expr),
                })
            }
            ast::Expression::SliceAccess { base, start, end } => {
                let base_expr = self.translate_expression(base)?;
                let start_expr = start
                    .as_ref()
                    .map(|s| self.translate_expression(s))
                    .transpose()?
                    .map(Box::new);
                let end_expr = end
                    .as_ref()
                    .map(|e| self.translate_expression(e))
                    .transpose()?
                    .map(Box::new);
                Ok(LogicalExpression::SliceAccess {
                    base: Box::new(base_expr),
                    start: start_expr,
                    end: end_expr,
                })
            }
            ast::Expression::ValueSubquery { query } => {
                // VALUE { subquery } returns a scalar from the inner query.
                // If the inner RETURN is a count() aggregate over an edge pattern,
                // use CountSubquery (optimized path that handles correlation).
                // Otherwise, translate the full query and use ValueSubquery + Apply.
                if Self::is_count_aggregate_return(&query.return_clause) {
                    let inner_plan = self.translate_subquery_to_operator(query)?;
                    Ok(LogicalExpression::CountSubquery(Box::new(inner_plan)))
                } else {
                    let inner_logical_plan = self.translate_query(query)?;
                    Ok(LogicalExpression::ValueSubquery(Box::new(
                        inner_logical_plan.root,
                    )))
                }
            }
            ast::Expression::ListComprehension {
                variable,
                list_expr,
                filter_expr,
                map_expr,
            } => {
                let list = self.translate_expression(list_expr)?;
                let filter = filter_expr
                    .as_ref()
                    .map(|f| self.translate_expression(f))
                    .transpose()?
                    .map(Box::new);
                let map = self.translate_expression(map_expr)?;
                Ok(LogicalExpression::ListComprehension {
                    variable: variable.clone(),
                    list_expr: Box::new(list),
                    filter_expr: filter,
                    map_expr: Box::new(map),
                })
            }
            ast::Expression::ListPredicate {
                kind,
                variable,
                list_expr,
                predicate,
            } => {
                let list = self.translate_expression(list_expr)?;
                let pred = self.translate_expression(predicate)?;
                let logical_kind = match kind {
                    ast::ListPredicateKind::All => plan::ListPredicateKind::All,
                    ast::ListPredicateKind::Any => plan::ListPredicateKind::Any,
                    ast::ListPredicateKind::None => plan::ListPredicateKind::None,
                    ast::ListPredicateKind::Single => plan::ListPredicateKind::Single,
                };
                Ok(LogicalExpression::ListPredicate {
                    kind: logical_kind,
                    variable: variable.clone(),
                    list_expr: Box::new(list),
                    predicate: Box::new(pred),
                })
            }
            ast::Expression::Reduce {
                accumulator,
                initial,
                variable,
                list,
                expression,
            } => {
                let init = self.translate_expression(initial)?;
                let list_expr = self.translate_expression(list)?;
                let body = self.translate_expression(expression)?;
                Ok(LogicalExpression::Reduce {
                    accumulator: accumulator.clone(),
                    initial: Box::new(init),
                    variable: variable.clone(),
                    list: Box::new(list_expr),
                    expression: Box::new(body),
                })
            }
            ast::Expression::LetIn { bindings, body } => {
                // LET x = expr1, y = expr2 IN body END
                // Translate each binding, then inline-substitute into the body.
                let binding_exprs: Vec<(String, LogicalExpression)> = bindings
                    .iter()
                    .map(|(name, expr)| Ok((name.clone(), self.translate_expression(expr)?)))
                    .collect::<Result<_>>()?;
                let body_expr = self.translate_expression(body)?;
                Ok(Self::substitute_let_bindings(body_expr, &binding_exprs))
            }
        }
    }

    pub(super) fn translate_literal(&self, lit: &ast::Literal) -> LogicalExpression {
        let value = match lit {
            ast::Literal::Null => Value::Null,
            ast::Literal::Bool(b) => Value::Bool(*b),
            ast::Literal::Integer(i) => Value::Int64(*i),
            ast::Literal::Float(f) => Value::Float64(*f),
            ast::Literal::String(s) => Value::String(s.clone().into()),
            ast::Literal::Date(s) => grafeo_common::types::Date::parse(s)
                .map_or_else(|| Value::String(s.clone().into()), Value::Date),
            ast::Literal::Time(s) => grafeo_common::types::Time::parse(s)
                .map_or_else(|| Value::String(s.clone().into()), Value::Time),
            ast::Literal::Duration(s) => grafeo_common::types::Duration::parse(s)
                .map_or_else(|| Value::String(s.clone().into()), Value::Duration),
            ast::Literal::Datetime(s) => {
                // Try full ISO datetime: YYYY-MM-DDTHH:MM:SS[.fff][Z|+HH:MM]
                if let Some(pos) = s.find('T') {
                    if let (Some(d), Some(t)) = (
                        grafeo_common::types::Date::parse(&s[..pos]),
                        grafeo_common::types::Time::parse(&s[pos + 1..]),
                    ) {
                        Value::Timestamp(grafeo_common::types::Timestamp::from_date_time(d, t))
                    } else {
                        Value::String(s.clone().into())
                    }
                } else if let Some(d) = grafeo_common::types::Date::parse(s) {
                    Value::Timestamp(d.to_timestamp())
                } else {
                    Value::String(s.clone().into())
                }
            }
            ast::Literal::ZonedDatetime(s) => grafeo_common::types::ZonedDatetime::parse(s)
                .map_or_else(|| Value::String(s.clone().into()), Value::ZonedDatetime),
            ast::Literal::ZonedTime(s) => {
                // Parse as Time with required offset
                if let Some(t) = grafeo_common::types::Time::parse(s)
                    && t.offset_seconds().is_some()
                {
                    Value::Time(t)
                } else {
                    Value::String(s.clone().into())
                }
            }
        };
        LogicalExpression::Literal(value)
    }

    pub(super) fn translate_binary_op(&self, op: ast::BinaryOp) -> BinaryOp {
        match op {
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
            ast::BinaryOp::Pow => BinaryOp::Pow,
            ast::BinaryOp::Concat => BinaryOp::Concat,
            ast::BinaryOp::Like => BinaryOp::Like,
            ast::BinaryOp::In => BinaryOp::In,
            ast::BinaryOp::StartsWith => BinaryOp::StartsWith,
            ast::BinaryOp::EndsWith => BinaryOp::EndsWith,
            ast::BinaryOp::Contains => BinaryOp::Contains,
        }
    }

    pub(super) fn translate_unary_op(&self, op: ast::UnaryOp) -> UnaryOp {
        match op {
            ast::UnaryOp::Not => UnaryOp::Not,
            ast::UnaryOp::Neg => UnaryOp::Neg,
            // Pos is handled as a no-op at the call site; this arm is unreachable.
            ast::UnaryOp::Pos => UnaryOp::Not,
            ast::UnaryOp::IsNull => UnaryOp::IsNull,
            ast::UnaryOp::IsNotNull => UnaryOp::IsNotNull,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn translator() -> GqlTranslator {
        GqlTranslator::new()
    }

    // --- NULLIF desugaring ---

    #[test]
    fn test_nullif_desugars_to_case() {
        // NULLIF(a, b) should produce:
        //   CASE WHEN a = b THEN NULL ELSE a END
        let t = translator();
        let expr = ast::Expression::FunctionCall {
            name: "NULLIF".to_string(),
            args: vec![
                ast::Expression::Variable("a".to_string()),
                ast::Expression::Variable("b".to_string()),
            ],
            distinct: false,
        };

        let result = t.translate_expression(&expr).unwrap();

        // Verify the Case structure
        if let LogicalExpression::Case {
            operand,
            when_clauses,
            else_clause,
        } = &result
        {
            // No operand (searched CASE, not simple CASE)
            assert!(
                operand.is_none(),
                "NULLIF should produce a searched CASE (no operand)"
            );

            // Exactly one WHEN clause
            assert_eq!(
                when_clauses.len(),
                1,
                "NULLIF should produce exactly one WHEN clause"
            );

            // WHEN condition: a = b
            let (condition, then_expr) = &when_clauses[0];
            if let LogicalExpression::Binary { left, op, right } = condition {
                assert!(
                    matches!(op, BinaryOp::Eq),
                    "WHEN condition should use Eq operator"
                );
                assert!(
                    matches!(left.as_ref(), LogicalExpression::Variable(v) if v == "a"),
                    "left side of condition should be variable 'a'"
                );
                assert!(
                    matches!(right.as_ref(), LogicalExpression::Variable(v) if v == "b"),
                    "right side of condition should be variable 'b'"
                );
            } else {
                panic!("WHEN condition should be a Binary expression, got: {condition:?}");
            }

            // THEN NULL
            assert!(
                matches!(then_expr, LogicalExpression::Literal(Value::Null)),
                "THEN clause should be NULL"
            );

            // ELSE a
            let else_expr = else_clause
                .as_ref()
                .expect("NULLIF should have an ELSE clause");
            assert!(
                matches!(else_expr.as_ref(), LogicalExpression::Variable(v) if v == "a"),
                "ELSE clause should be variable 'a'"
            );
        } else {
            panic!("NULLIF should translate to a Case expression, got: {result:?}");
        }
    }

    #[test]
    fn test_nullif_case_insensitive() {
        // Verify that "nullif" (lowercase) also works
        let t = translator();
        let expr = ast::Expression::FunctionCall {
            name: "nullif".to_string(),
            args: vec![
                ast::Expression::Literal(ast::Literal::Integer(1)),
                ast::Expression::Literal(ast::Literal::Integer(2)),
            ],
            distinct: false,
        };

        let result = t.translate_expression(&expr).unwrap();
        assert!(
            matches!(result, LogicalExpression::Case { .. }),
            "lowercase 'nullif' should also desugar to Case"
        );
    }

    #[test]
    fn test_nullif_wrong_arity_rejected() {
        let t = translator();

        // Too few arguments
        let expr_one = ast::Expression::FunctionCall {
            name: "NULLIF".to_string(),
            args: vec![ast::Expression::Variable("a".to_string())],
            distinct: false,
        };
        assert!(
            t.translate_expression(&expr_one).is_err(),
            "NULLIF with 1 argument should fail"
        );

        // Too many arguments
        let expr_three = ast::Expression::FunctionCall {
            name: "NULLIF".to_string(),
            args: vec![
                ast::Expression::Variable("a".to_string()),
                ast::Expression::Variable("b".to_string()),
                ast::Expression::Variable("c".to_string()),
            ],
            distinct: false,
        };
        assert!(
            t.translate_expression(&expr_three).is_err(),
            "NULLIF with 3 arguments should fail"
        );
    }

    // --- length() on path variables ---

    #[test]
    fn test_length_of_path_variable() {
        // length(p) where p is a path variable should translate to
        // Variable("_path_length_p")
        let t = translator();
        let expr = ast::Expression::FunctionCall {
            name: "length".to_string(),
            args: vec![ast::Expression::Variable("p".to_string())],
            distinct: false,
        };

        let result = t.translate_expression(&expr).unwrap();
        assert!(
            matches!(&result, LogicalExpression::Variable(v) if v == "_path_length_p"),
            "length(p) should produce Variable(\"_path_length_p\"), got: {result:?}"
        );
    }

    #[test]
    fn test_length_case_insensitive() {
        // LENGTH(myPath) should also work (case-insensitive function name)
        let t = translator();
        let expr = ast::Expression::FunctionCall {
            name: "LENGTH".to_string(),
            args: vec![ast::Expression::Variable("myPath".to_string())],
            distinct: false,
        };

        let result = t.translate_expression(&expr).unwrap();
        assert!(
            matches!(&result, LogicalExpression::Variable(v) if v == "_path_length_myPath"),
            "LENGTH(myPath) should produce _path_length_myPath, got: {result:?}"
        );
    }

    #[test]
    fn test_length_with_non_variable_arg_is_regular_function() {
        // length(expr) where expr is NOT a plain variable should remain a regular function call
        let t = translator();
        let expr = ast::Expression::FunctionCall {
            name: "length".to_string(),
            args: vec![ast::Expression::PropertyAccess {
                variable: "n".to_string(),
                property: "name".to_string(),
            }],
            distinct: false,
        };

        let result = t.translate_expression(&expr).unwrap();
        assert!(
            matches!(&result, LogicalExpression::FunctionCall { name, .. } if name == "length"),
            "length(n.name) should stay as FunctionCall, got: {result:?}"
        );
    }

    #[test]
    fn test_length_with_multiple_args_is_regular_function() {
        // length(a, b) should remain a regular function call (path optimization is single-arg only)
        let t = translator();
        let expr = ast::Expression::FunctionCall {
            name: "length".to_string(),
            args: vec![
                ast::Expression::Variable("a".to_string()),
                ast::Expression::Variable("b".to_string()),
            ],
            distinct: false,
        };

        let result = t.translate_expression(&expr).unwrap();
        assert!(
            matches!(&result, LogicalExpression::FunctionCall { name, .. } if name == "length"),
            "length(a, b) should stay as FunctionCall, got: {result:?}"
        );
    }
}
