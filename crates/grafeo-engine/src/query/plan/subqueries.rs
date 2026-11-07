//! The plans an expression's subqueries hold, which the SPARQL translator and the RDF planner
//! read.

use grafeo_common::types::Value;

use super::{LogicalExpression, LogicalOperator};

/// The plans of the `EXISTS` tests in `expression`, at any depth.
pub(crate) fn exists_plans(expression: &LogicalExpression) -> Vec<LogicalOperator> {
    fn found(expression: LogicalExpression, plans: &mut Vec<LogicalOperator>) -> LogicalExpression {
        match expression {
            LogicalExpression::ExistsSubquery(plan) => {
                plans.push(*plan);
                LogicalExpression::Literal(Value::Null)
            }
            other => other.map_children(|child| found(child, plans)),
        }
    }
    let mut plans = Vec::new();
    found(expression.clone(), &mut plans);
    plans
}
