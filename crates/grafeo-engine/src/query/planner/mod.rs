//! Converts logical plans into physical execution trees.
//!
//! The optimizer produces a logical plan (what data you want), but the planner
//! converts it to a physical plan (how to actually get it). This means choosing
//! hash joins vs nested loops, picking index scans vs full scans, etc.
//!
//! This module contains shared infrastructure used by both the LPG and RDF planners:
//! - [`PhysicalPlan`] - the output of planning
//! - Expression and operator conversion functions
//! - Reusable operator builders (in the common submodule)
//!
//! Model-specific planning lives in [`lpg`] and [`rdf`].

pub(crate) mod common;
pub mod lpg;

#[cfg(feature = "triple-store")]
pub mod rdf;

// Re-export the LPG planner as the default `Planner` for backwards compatibility.
pub use lpg::Planner;

use crate::query::plan::{
    AggregateFunction as LogicalAggregateFunction, BinaryOp, LogicalExpression, UnaryOp,
};
use grafeo_common::types::LogicalType;
use grafeo_common::utils::error::{Error, Result};
use grafeo_core::execution::AdaptiveContext;
use grafeo_core::execution::operators::{
    AggregateFunction as PhysicalAggregateFunction, BinaryFilterOp, FilterExpression, Operator,
    UnaryFilterOp,
};

/// A physical plan ready for execution.
pub struct PhysicalPlan {
    /// The root physical operator.
    pub operator: Box<dyn Operator>,
    /// Column names for the result.
    pub columns: Vec<String>,
    /// Adaptive execution context with cardinality estimates.
    ///
    /// When adaptive execution is enabled, this context contains estimated
    /// cardinalities at various checkpoints in the plan. During execution,
    /// actual row counts are recorded and compared against estimates.
    pub adaptive_context: Option<AdaptiveContext>,
}

impl PhysicalPlan {
    /// Returns the column names.
    #[must_use]
    pub fn columns(&self) -> &[String] {
        &self.columns
    }

    /// Consumes the plan and returns the operator.
    pub fn into_operator(self) -> Box<dyn Operator> {
        self.operator
    }

    /// Returns the adaptive context, if adaptive execution is enabled.
    #[must_use]
    pub fn adaptive_context(&self) -> Option<&AdaptiveContext> {
        self.adaptive_context.as_ref()
    }

    /// Takes ownership of the adaptive context.
    pub fn take_adaptive_context(&mut self) -> Option<AdaptiveContext> {
        self.adaptive_context.take()
    }
}

// ---------------------------------------------------------------------------
// Shared conversion functions (used by both LPG and RDF planners)
// ---------------------------------------------------------------------------

/// Converts a logical binary operator to a filter binary operator.
///
/// # Errors
///
/// Currently infallible for all `BinaryOp` variants, but returns `Result`
/// for forward compatibility.
pub fn convert_binary_op(op: BinaryOp) -> Result<BinaryFilterOp> {
    match op {
        BinaryOp::Eq => Ok(BinaryFilterOp::Eq),
        BinaryOp::Ne => Ok(BinaryFilterOp::Ne),
        BinaryOp::Lt => Ok(BinaryFilterOp::Lt),
        BinaryOp::Le => Ok(BinaryFilterOp::Le),
        BinaryOp::Gt => Ok(BinaryFilterOp::Gt),
        BinaryOp::Ge => Ok(BinaryFilterOp::Ge),
        BinaryOp::And => Ok(BinaryFilterOp::And),
        BinaryOp::Or => Ok(BinaryFilterOp::Or),
        BinaryOp::Xor => Ok(BinaryFilterOp::Xor),
        BinaryOp::Add => Ok(BinaryFilterOp::Add),
        BinaryOp::Sub => Ok(BinaryFilterOp::Sub),
        BinaryOp::Mul => Ok(BinaryFilterOp::Mul),
        BinaryOp::Div => Ok(BinaryFilterOp::Div),
        BinaryOp::Mod => Ok(BinaryFilterOp::Mod),
        BinaryOp::StartsWith => Ok(BinaryFilterOp::StartsWith),
        BinaryOp::EndsWith => Ok(BinaryFilterOp::EndsWith),
        BinaryOp::Contains => Ok(BinaryFilterOp::Contains),
        BinaryOp::In => Ok(BinaryFilterOp::In),
        BinaryOp::Regex => Ok(BinaryFilterOp::Regex),
        BinaryOp::Pow => Ok(BinaryFilterOp::Pow),
        BinaryOp::Concat => Ok(BinaryFilterOp::Concat),
        BinaryOp::Like => Ok(BinaryFilterOp::Like),
    }
}

/// Converts a logical unary operator to a filter unary operator.
///
/// # Errors
///
/// Currently infallible for all `UnaryOp` variants, but returns `Result`
/// for forward compatibility.
pub fn convert_unary_op(op: UnaryOp) -> Result<UnaryFilterOp> {
    match op {
        UnaryOp::Not => Ok(UnaryFilterOp::Not),
        UnaryOp::IsNull => Ok(UnaryFilterOp::IsNull),
        UnaryOp::IsNotNull => Ok(UnaryFilterOp::IsNotNull),
        UnaryOp::Neg => Ok(UnaryFilterOp::Neg),
    }
}

/// Converts a logical aggregate function to a physical aggregate function.
pub fn convert_aggregate_function(func: LogicalAggregateFunction) -> PhysicalAggregateFunction {
    match func {
        LogicalAggregateFunction::Count => PhysicalAggregateFunction::Count,
        LogicalAggregateFunction::CountNonNull => PhysicalAggregateFunction::CountNonNull,
        LogicalAggregateFunction::Sum => PhysicalAggregateFunction::Sum,
        LogicalAggregateFunction::Avg => PhysicalAggregateFunction::Avg,
        LogicalAggregateFunction::Min => PhysicalAggregateFunction::Min,
        LogicalAggregateFunction::Max => PhysicalAggregateFunction::Max,
        LogicalAggregateFunction::Collect => PhysicalAggregateFunction::Collect,
        LogicalAggregateFunction::StdDev => PhysicalAggregateFunction::StdDev,
        LogicalAggregateFunction::StdDevPop => PhysicalAggregateFunction::StdDevPop,
        LogicalAggregateFunction::Variance => PhysicalAggregateFunction::Variance,
        LogicalAggregateFunction::VariancePop => PhysicalAggregateFunction::VariancePop,
        LogicalAggregateFunction::PercentileDisc => PhysicalAggregateFunction::PercentileDisc,
        LogicalAggregateFunction::PercentileCont => PhysicalAggregateFunction::PercentileCont,
        LogicalAggregateFunction::GroupConcat => PhysicalAggregateFunction::GroupConcat,
        LogicalAggregateFunction::Sample => PhysicalAggregateFunction::Sample,
        LogicalAggregateFunction::CovarSamp => PhysicalAggregateFunction::CovarSamp,
        LogicalAggregateFunction::CovarPop => PhysicalAggregateFunction::CovarPop,
        LogicalAggregateFunction::Corr => PhysicalAggregateFunction::Corr,
        LogicalAggregateFunction::RegrSlope => PhysicalAggregateFunction::RegrSlope,
        LogicalAggregateFunction::RegrIntercept => PhysicalAggregateFunction::RegrIntercept,
        LogicalAggregateFunction::RegrR2 => PhysicalAggregateFunction::RegrR2,
        LogicalAggregateFunction::RegrCount => PhysicalAggregateFunction::RegrCount,
        LogicalAggregateFunction::RegrSxx => PhysicalAggregateFunction::RegrSxx,
        LogicalAggregateFunction::RegrSyy => PhysicalAggregateFunction::RegrSyy,
        LogicalAggregateFunction::RegrSxy => PhysicalAggregateFunction::RegrSxy,
        LogicalAggregateFunction::RegrAvgx => PhysicalAggregateFunction::RegrAvgx,
        LogicalAggregateFunction::RegrAvgy => PhysicalAggregateFunction::RegrAvgy,
    }
}

/// Converts a logical expression to a filter expression.
///
/// This is a standalone function used by both LPG and RDF planners.
///
/// # Errors
///
/// Returns an error if the expression contains an unsupported operator or
/// function that cannot be translated to a filter expression.
pub fn convert_filter_expression(expr: &LogicalExpression) -> Result<FilterExpression> {
    match expr {
        LogicalExpression::Literal(v) => Ok(FilterExpression::Literal(v.clone())),
        LogicalExpression::Variable(name) => Ok(FilterExpression::Variable(name.clone())),
        LogicalExpression::Property { variable, property } => Ok(FilterExpression::Property {
            variable: variable.clone(),
            property: property.clone(),
        }),
        LogicalExpression::Binary { left, op, right } => {
            let left_expr = convert_filter_expression(left)?;
            let right_expr = convert_filter_expression(right)?;
            let filter_op = convert_binary_op(*op)?;
            Ok(FilterExpression::Binary {
                left: Box::new(left_expr),
                op: filter_op,
                right: Box::new(right_expr),
            })
        }
        LogicalExpression::Unary { op, operand } => {
            let operand_expr = convert_filter_expression(operand)?;
            let filter_op = convert_unary_op(*op)?;
            Ok(FilterExpression::Unary {
                op: filter_op,
                operand: Box::new(operand_expr),
            })
        }
        LogicalExpression::FunctionCall { name, args, .. } => {
            let filter_args: Vec<FilterExpression> = args
                .iter()
                .map(convert_filter_expression)
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
                .map(|e| convert_filter_expression(e))
                .transpose()?
                .map(Box::new);
            let filter_when_clauses: Vec<(FilterExpression, FilterExpression)> = when_clauses
                .iter()
                .map(|(cond, result)| {
                    Ok((
                        convert_filter_expression(cond)?,
                        convert_filter_expression(result)?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            let filter_else = else_clause
                .as_ref()
                .map(|e| convert_filter_expression(e))
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
                .map(convert_filter_expression)
                .collect::<Result<Vec<_>>>()?;
            Ok(FilterExpression::List(filter_items))
        }
        LogicalExpression::Map(pairs) => {
            let filter_pairs: Vec<(String, FilterExpression)> = pairs
                .iter()
                .map(|(k, v)| Ok((k.clone(), convert_filter_expression(v)?)))
                .collect::<Result<Vec<_>>>()?;
            Ok(FilterExpression::Map(filter_pairs))
        }
        LogicalExpression::IndexAccess { base, index } => {
            let base_expr = convert_filter_expression(base)?;
            let index_expr = convert_filter_expression(index)?;
            Ok(FilterExpression::IndexAccess {
                base: Box::new(base_expr),
                index: Box::new(index_expr),
            })
        }
        LogicalExpression::MapAccess { base, key } => Ok(FilterExpression::IndexAccess {
            base: Box::new(convert_filter_expression(base)?),
            index: Box::new(FilterExpression::Literal(
                grafeo_common::types::Value::from(key.as_str()),
            )),
        }),
        LogicalExpression::SliceAccess { base, start, end } => {
            let base_expr = convert_filter_expression(base)?;
            let start_expr = start
                .as_ref()
                .map(|s| convert_filter_expression(s))
                .transpose()?
                .map(Box::new);
            let end_expr = end
                .as_ref()
                .map(|e| convert_filter_expression(e))
                .transpose()?
                .map(Box::new);
            Ok(FilterExpression::SliceAccess {
                base: Box::new(base_expr),
                start: start_expr,
                end: end_expr,
            })
        }
        LogicalExpression::Parameter(_) => Err(Error::Internal(
            "Parameters not yet supported in filters".to_string(),
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
            let list = convert_filter_expression(list_expr)?;
            let filter = filter_expr
                .as_ref()
                .map(|f| convert_filter_expression(f))
                .transpose()?
                .map(Box::new);
            let map = convert_filter_expression(map_expr)?;
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
            use crate::query::plan::ListPredicateKind as LPK;
            let filter_kind = match kind {
                LPK::All => grafeo_core::execution::operators::ListPredicateKind::All,
                LPK::Any => grafeo_core::execution::operators::ListPredicateKind::Any,
                LPK::None => grafeo_core::execution::operators::ListPredicateKind::None,
                LPK::Single => grafeo_core::execution::operators::ListPredicateKind::Single,
            };
            let list = convert_filter_expression(list_expr)?;
            let pred = convert_filter_expression(predicate)?;
            Ok(FilterExpression::ListPredicate {
                kind: filter_kind,
                variable: variable.clone(),
                list_expr: Box::new(list),
                predicate: Box::new(pred),
            })
        }
        LogicalExpression::ExistsSubquery(_)
        | LogicalExpression::CountSubquery(_)
        | LogicalExpression::ValueSubquery(_) => {
            // Complex subqueries are handled at the plan_filter level via semi-join
            // or Apply rewrites. If we reach here, the subquery is in a position that
            // cannot be rewritten (e.g., nested inside a CASE expression). Return a
            // literal false/zero as a safe fallback.
            Err(Error::Internal(
                "Subquery expressions in this position require the semi-join or Apply rewrite; \
                 move the EXISTS/COUNT/VALUE subquery to a top-level WHERE predicate or RETURN"
                    .to_string(),
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
                        Ok((key.clone(), convert_filter_expression(expr)?))
                    }
                    crate::query::plan::MapProjectionEntry::AllProperties => Ok((
                        "*".to_string(),
                        FilterExpression::FunctionCall {
                            name: "properties".to_string(),
                            args: vec![FilterExpression::Variable(base.clone())],
                        },
                    )),
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
        } => Ok(FilterExpression::Reduce {
            accumulator: accumulator.clone(),
            initial: Box::new(convert_filter_expression(initial)?),
            variable: variable.clone(),
            list: Box::new(convert_filter_expression(list)?),
            expression: Box::new(convert_filter_expression(expression)?),
        }),
        LogicalExpression::PatternComprehension { projection, .. } => {
            let proj = convert_filter_expression(projection)?;
            Ok(FilterExpression::FunctionCall {
                name: "collect".to_string(),
                args: vec![proj],
            })
        }
    }
}

/// Infers the logical type from a value.
pub(crate) fn value_to_logical_type(value: &grafeo_common::types::Value) -> LogicalType {
    use grafeo_common::types::Value;
    match value {
        Value::Null => LogicalType::String,
        Value::Bool(_) => LogicalType::Bool,
        Value::Int64(_) => LogicalType::Int64,
        Value::Float64(_) => LogicalType::Float64,
        Value::String(_) => LogicalType::String,
        Value::Bytes(_) => LogicalType::String,
        Value::Timestamp(_) => LogicalType::Timestamp,
        Value::Date(_) => LogicalType::Date,
        Value::Time(_) => LogicalType::Time,
        Value::Duration(_) => LogicalType::Duration,
        Value::ZonedDatetime(_) => LogicalType::ZonedDatetime,
        Value::List(_) => LogicalType::String,
        Value::Map(_) => LogicalType::String,
        Value::Vector(v) => LogicalType::Vector(v.len()),
        Value::Path { .. } => LogicalType::Any,
        Value::GCounter(_) | Value::OnCounter { .. } => LogicalType::Any,
        _ => LogicalType::Any,
    }
}

/// Evaluates a constant logical expression to a Value.
///
/// Only handles literals, unary minus on numeric literals, and simple expressions.
/// Returns an error for runtime-dependent expressions (variables, property accesses, etc.).
#[cfg(feature = "algos")]
pub(crate) fn eval_constant_expression(
    expr: &crate::query::plan::LogicalExpression,
) -> Result<grafeo_common::types::Value> {
    use crate::query::plan::LogicalExpression;
    use grafeo_common::types::Value;

    match expr {
        LogicalExpression::Literal(val) => Ok(val.clone()),
        LogicalExpression::Unary {
            op: crate::query::plan::UnaryOp::Neg,
            operand,
        } => {
            let val = eval_constant_expression(operand)?;
            match val {
                Value::Int64(n) => Ok(Value::Int64(-n)),
                Value::Float64(f) => Ok(Value::Float64(-f)),
                _ => Err(Error::Internal("Cannot negate non-numeric value".into())),
            }
        }
        _ => Err(Error::Internal(
            "Procedure argument must be a constant value".into(),
        )),
    }
}
