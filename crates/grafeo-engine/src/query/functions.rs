//! The functions a query calls and the patterns it matches, checked when the
//! query is planned.
//!
//! A call is to a scalar function of the expression evaluator (see
//! [`function_support`]) or to an aggregate, which groups rows. The LPG
//! planner checks every call and every `=~` and LIKE it plans, so a query
//! that calls a function that does not exist (a misspelled `upperr`), a
//! function this build lacks a feature for, or that matches a pattern this
//! build cannot, fails before any row is read. The evaluator would answer
//! each of these with a null for every row, and a WHERE would keep no rows.

use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result};
use grafeo_common::utils::strings::{find_similar, format_suggestion};
use grafeo_core::execution::operators::{
    FunctionSupport, REGEX_SUPPORT, function_names, function_support, regex_pattern_error,
};

use crate::query::plan::{BinaryOp, LogicalExpression};

/// The aggregate functions: every spelling a query may use (case does not
/// matter).
const AGGREGATE_FUNCTIONS: &[&str] = &[
    "count",
    "sum",
    "avg",
    "min",
    "max",
    "collect",
    "stdev",
    "stddev",
    "stddev_samp",
    "stdevp",
    "stddevp",
    "stddev_pop",
    "variance",
    "var_samp",
    "var_pop",
    "percentile_disc",
    "percentileDisc",
    "percentile_cont",
    "percentileCont",
    "group_concat",
    "groupConcat",
    "listagg",
    "sample",
    "covar_samp",
    "covar_pop",
    "corr",
    "regr_slope",
    "regr_intercept",
    "regr_r2",
    "regr_count",
    "regr_sxx",
    "regr_syy",
    "regr_sxy",
    "regr_avgx",
    "regr_avgy",
];

/// Whether `name` is an aggregate function (case does not matter).
pub(crate) fn is_aggregate_function(name: &str) -> bool {
    AGGREGATE_FUNCTIONS
        .iter()
        .any(|aggregate| aggregate.eq_ignore_ascii_case(name))
}

/// A semantic error with `message`.
fn semantic(message: String) -> Error {
    Error::Query(QueryError::new(QueryErrorKind::Semantic, message))
}

/// Checks a call to the function `name`: an aggregate, or a scalar function
/// the evaluator computes in this build.
///
/// # Errors
///
/// `Unknown function '<name>'`, with the closest function name as a hint when
/// one is close, or, for a function that needs a feature this build does not
/// have, an error that names the feature.
pub(crate) fn check_function_call(name: &str) -> Result<()> {
    if is_aggregate_function(name) {
        return Ok(());
    }
    match function_support(name) {
        FunctionSupport::Available => Ok(()),
        FunctionSupport::NeedsFeature(feature) => Err(semantic(format!(
            "Function '{name}' is not available in this build: it needs the '{feature}' feature"
        ))),
        _ => {
            let known: Vec<&str> = function_names()
                .chain(AGGREGATE_FUNCTIONS.iter().copied())
                .collect();
            let error = QueryError::new(
                QueryErrorKind::Semantic,
                format!("Unknown function '{name}'"),
            );
            Err(Error::Query(match find_similar(name, &known) {
                Some(similar) => error.with_hint(format_suggestion(similar)),
                None => error,
            }))
        }
    }
}

/// Checks a `=~` or LIKE whose pattern is `pattern`: the build must have
/// regular expressions, and a `=~` pattern the query gives as a string (a
/// literal, or a parameter, which the planner sees as its value) must be one.
/// A pattern read from the rows is matched as it comes; one that is not a
/// regular expression matches nothing.
///
/// # Errors
///
/// An error that says the build has no regular expressions, or that names the
/// pattern and what is wrong with it.
pub(crate) fn check_pattern_match(op: BinaryOp, pattern: &LogicalExpression) -> Result<()> {
    let operator = match op {
        BinaryOp::Regex => "=~",
        BinaryOp::Like => "LIKE",
        _ => return Ok(()),
    };
    if !REGEX_SUPPORT {
        return Err(semantic(format!(
            "{operator} needs regular expressions, and this build has no regular expressions: \
             it needs the 'regex' or 'regex-lite' feature"
        )));
    }
    if op == BinaryOp::Regex
        && let LogicalExpression::Literal(Value::String(pattern)) = pattern
        && let Some(reason) = regex_pattern_error(pattern)
    {
        return Err(semantic(format!(
            "Invalid regular expression '{pattern}': {reason}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The message of `result`'s error, with its hint.
    fn message(result: Result<()>) -> String {
        result.expect_err("expected an error").to_string()
    }

    #[test]
    fn aggregates_and_scalar_functions_are_known_in_any_case() {
        for name in [
            "count",
            "COUNT",
            "percentileDisc",
            "upper",
            "toUpper",
            "ELEMENTID",
        ] {
            assert!(check_function_call(name).is_ok(), "{name}");
        }
    }

    #[test]
    fn an_unknown_function_names_the_closest_known_one() {
        let error = message(check_function_call("upperr"));
        assert!(error.contains("Unknown function 'upperr'"), "{error}");
        assert!(error.contains("Did you mean 'upper'?"), "{error}");
        let error = message(check_function_call("coutn"));
        assert!(error.contains("Did you mean 'count'?"), "{error}");
        let error = message(check_function_call("regexp_matches"));
        assert!(
            error.contains("Unknown function 'regexp_matches'") && !error.contains("Did you mean"),
            "{error}"
        );
    }

    #[test]
    fn only_regex_and_like_patterns_are_checked() {
        let pattern = LogicalExpression::Literal(Value::from("(src"));
        assert!(check_pattern_match(BinaryOp::Eq, &pattern).is_ok());
        assert!(check_pattern_match(BinaryOp::Contains, &pattern).is_ok());
        if REGEX_SUPPORT {
            // A LIKE pattern is escaped into a regular expression, so any
            // string is one.
            assert!(check_pattern_match(BinaryOp::Like, &pattern).is_ok());
            let error = message(check_pattern_match(BinaryOp::Regex, &pattern));
            assert!(
                error.contains("Invalid regular expression '(src': "),
                "{error}"
            );
            // A pattern read from a row is matched when the row comes.
            let read = LogicalExpression::Property {
                variable: "n".to_string(),
                property: "pattern".to_string(),
            };
            assert!(check_pattern_match(BinaryOp::Regex, &read).is_ok());
        } else {
            let error = message(check_pattern_match(BinaryOp::Like, &pattern));
            assert!(
                error.contains("this build has no regular expressions"),
                "{error}"
            );
        }
    }
}
