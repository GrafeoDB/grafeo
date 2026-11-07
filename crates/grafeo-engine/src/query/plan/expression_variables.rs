//! The variables an expression reads.

use std::collections::HashSet;

use super::LogicalExpression;

pub(crate) fn collect_expression_variables(expr: &LogicalExpression, vars: &mut HashSet<String>) {
    match expr {
        LogicalExpression::Variable(name) => {
            vars.insert(name.clone());
        }
        LogicalExpression::Property { variable, .. }
        | LogicalExpression::Labels(variable)
        | LogicalExpression::Type(variable)
        | LogicalExpression::Id(variable) => {
            vars.insert(variable.clone());
        }
        LogicalExpression::Binary { left, right, .. } => {
            collect_expression_variables(left, vars);
            collect_expression_variables(right, vars);
        }
        LogicalExpression::Unary { operand, .. } => {
            collect_expression_variables(operand, vars);
        }
        LogicalExpression::FunctionCall { args, .. }
        | LogicalExpression::SparqlCall { args, .. } => {
            for arg in args {
                collect_expression_variables(arg, vars);
            }
        }
        LogicalExpression::List(items) => {
            for item in items {
                collect_expression_variables(item, vars);
            }
        }
        LogicalExpression::Map(pairs) => {
            for (_, value) in pairs {
                collect_expression_variables(value, vars);
            }
        }
        LogicalExpression::IndexAccess { base, index } => {
            collect_expression_variables(base, vars);
            collect_expression_variables(index, vars);
        }
        LogicalExpression::MapAccess { base, .. } => collect_expression_variables(base, vars),
        LogicalExpression::SliceAccess { base, start, end } => {
            collect_expression_variables(base, vars);
            if let Some(s) = start {
                collect_expression_variables(s, vars);
            }
            if let Some(e) = end {
                collect_expression_variables(e, vars);
            }
        }
        LogicalExpression::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            if let Some(op) = operand {
                collect_expression_variables(op, vars);
            }
            for (cond, result) in when_clauses {
                collect_expression_variables(cond, vars);
                collect_expression_variables(result, vars);
            }
            if let Some(else_expr) = else_clause {
                collect_expression_variables(else_expr, vars);
            }
        }
        // The variable a comprehension, list predicate or `reduce` binds is
        // its own: only the other names its body uses come from outside.
        LogicalExpression::ListComprehension {
            variable,
            list_expr,
            filter_expr,
            map_expr,
        } => {
            collect_expression_variables(list_expr, vars);
            let mut body = HashSet::new();
            if let Some(filter) = filter_expr {
                collect_expression_variables(filter, &mut body);
            }
            collect_expression_variables(map_expr, &mut body);
            body.remove(variable);
            vars.extend(body);
        }
        LogicalExpression::ListPredicate {
            variable,
            list_expr,
            predicate,
            ..
        } => {
            collect_expression_variables(list_expr, vars);
            let mut body = HashSet::new();
            collect_expression_variables(predicate, &mut body);
            body.remove(variable);
            vars.extend(body);
        }
        LogicalExpression::MapProjection { base, entries } => {
            vars.insert(base.clone());
            for entry in entries {
                if let crate::query::plan::MapProjectionEntry::LiteralEntry(_, expr) = entry {
                    collect_expression_variables(expr, vars);
                }
            }
        }
        LogicalExpression::Reduce {
            accumulator,
            initial,
            variable,
            list,
            expression,
        } => {
            collect_expression_variables(initial, vars);
            collect_expression_variables(list, vars);
            let mut body = HashSet::new();
            collect_expression_variables(expression, &mut body);
            body.remove(accumulator);
            body.remove(variable);
            vars.extend(body);
        }
        LogicalExpression::PatternComprehension { projection, .. } => {
            collect_expression_variables(projection, vars);
        }
        LogicalExpression::Literal(_)
        | LogicalExpression::Parameter(_)
        | LogicalExpression::ExistsSubquery(_)
        | LogicalExpression::CountSubquery(_)
        | LogicalExpression::ValueSubquery(_) => {}
    }
}
