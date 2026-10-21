//! The functions a query calls and the patterns it matches, checked before
//! any row is read.
//!
//! A call is to a built-in function of the expression evaluator or to an
//! aggregate, which groups rows. The GQL, Cypher and SQL/PGQ translators
//! resolve every call they translate ([`resolve_call`]), and the LPG planner
//! checks every `=~` and LIKE it plans, so a query that calls a function that
//! does not exist (a misspelled `upperr`), passes a function a number of
//! arguments it does not take (`toUpper('a', 'b')`), calls a function this
//! build lacks a feature for, or matches a pattern this build cannot, fails
//! before any row is read. The evaluator would answer each of these with a
//! null for every row, and a WHERE would keep no rows.

use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result};
#[cfg(any(test, feature = "gql", feature = "cypher", feature = "sql-pgq"))]
use grafeo_common::utils::strings::{find_similar, format_suggestion};
#[cfg(any(test, feature = "gql", feature = "cypher", feature = "sql-pgq"))]
use grafeo_core::execution::operators::{FunctionSupport, function_names, function_support};
use grafeo_core::execution::operators::{REGEX_SUPPORT, regex_pattern_error};

#[cfg(any(test, feature = "gql", feature = "cypher", feature = "sql-pgq"))]
use crate::query::plan::AggregateFunction;
use crate::query::plan::{BinaryOp, LogicalExpression};
#[cfg(any(feature = "gql", feature = "cypher", feature = "sql-pgq"))]
use crate::query::plan::{BuiltinFunction, Callee};

/// Whether `name` is an aggregate function (case does not matter).
#[cfg(any(test, feature = "gql", feature = "cypher", feature = "sql-pgq"))]
pub(crate) fn is_aggregate_function(name: &str) -> bool {
    AggregateFunction::from_name(name).is_some()
}

/// Checks the number of arguments of a call to the aggregate `function`
/// (named `name` in the query): `count` takes 0 (`count(*)`) or 1,
/// `group_concat` and `listagg` 1 or 2 (the separator), the percentiles, the
/// covariances, `corr` and the regressions 2, every other aggregate 1. A call
/// with another number used to build from what it found: `count(a, b)`
/// counted `a`, and `sum()` counted the rows.
///
/// # Errors
///
/// `Function '<name>' takes <numbers>, got <arguments>`, as for a built-in
/// function.
#[cfg(any(test, feature = "gql", feature = "cypher", feature = "sql-pgq"))]
pub(crate) fn check_aggregate_call(
    name: &str,
    function: AggregateFunction,
    arguments: usize,
) -> Result<()> {
    let (low, high, takes) = match function {
        AggregateFunction::Count | AggregateFunction::CountNonNull => (0, 1, "0 or 1 arguments"),
        AggregateFunction::GroupConcat => (1, 2, "1 or 2 arguments"),
        AggregateFunction::PercentileDisc
        | AggregateFunction::PercentileCont
        | AggregateFunction::CovarSamp
        | AggregateFunction::CovarPop
        | AggregateFunction::Corr
        | AggregateFunction::RegrSlope
        | AggregateFunction::RegrIntercept
        | AggregateFunction::RegrR2
        | AggregateFunction::RegrCount
        | AggregateFunction::RegrSxx
        | AggregateFunction::RegrSyy
        | AggregateFunction::RegrSxy
        | AggregateFunction::RegrAvgx
        | AggregateFunction::RegrAvgy => (2, 2, "2 arguments"),
        AggregateFunction::Sum
        | AggregateFunction::Avg
        | AggregateFunction::Min
        | AggregateFunction::Max
        | AggregateFunction::Collect
        | AggregateFunction::StdDev
        | AggregateFunction::StdDevPop
        | AggregateFunction::Variance
        | AggregateFunction::VariancePop
        | AggregateFunction::Sample => (1, 1, "1 argument"),
    };
    if (low..=high).contains(&arguments) {
        Ok(())
    } else {
        Err(semantic(format!(
            "Function '{name}' takes {takes}, got {arguments}"
        )))
    }
}

/// A semantic error with `message`.
fn semantic(message: String) -> Error {
    Error::Query(QueryError::new(QueryErrorKind::Semantic, message))
}

/// Checks a call to the function `name` with `arguments` arguments: an
/// aggregate, or a scalar function the evaluator computes in this build with
/// that many arguments.
///
/// # Errors
///
/// `Unknown function '<name>'`, with the closest function name as a hint when
/// one is close; `Function '<name>' takes <numbers>, got <arguments>` for a
/// number of arguments the function does not take; or, for a function that
/// needs a feature this build does not have, an error that names the feature.
#[cfg(any(test, feature = "gql", feature = "cypher", feature = "sql-pgq"))]
pub(crate) fn check_function_call(name: &str, arguments: usize) -> Result<()> {
    if is_aggregate_function(name) {
        return Ok(());
    }
    match function_support(name) {
        FunctionSupport::Available(arity) if arity.accepts(arguments) => Ok(()),
        FunctionSupport::Available(arity) => Err(semantic(format!(
            "Function '{name}' takes {arity}, got {arguments}"
        ))),
        FunctionSupport::NeedsFeature(feature) => {
            Err(Error::Query(QueryError::unsupported(format!(
                "Function '{name}' is not available in this build: it needs the '{feature}' \
                 feature"
            ))))
        }
        _ => {
            let known: Vec<&str> = function_names()
                .chain(AggregateFunction::ALIASES.iter().map(|&(alias, _)| alias))
                .collect();
            let error = QueryError::invalid_reference(format!("Unknown function '{name}'"));
            Err(Error::Query(match find_similar(name, &known) {
                Some(similar) => error.with_hint(format_suggestion(similar)),
                None => error,
            }))
        }
    }
}

/// The call a translator builds for `name(args)`: an aggregate, or a built-in
/// this build computes with that many arguments.
///
/// # Errors
///
/// The errors of [`check_function_call`], with the same texts: an unknown
/// function, a number of arguments it does not take, or a feature this build
/// lacks.
#[cfg(any(feature = "gql", feature = "cypher", feature = "sql-pgq"))]
pub(crate) fn resolve_call(
    name: &str,
    args: Vec<LogicalExpression>,
    distinct: bool,
) -> Result<LogicalExpression> {
    let callee = match AggregateFunction::from_name(name) {
        Some(aggregate) => Callee::Aggregate(aggregate),
        None => {
            check_function_call(name, args.len())?;
            let function = BuiltinFunction::from_name(name).ok_or_else(|| {
                Error::Query(QueryError::invalid_reference(format!(
                    "Unknown function '{name}'"
                )))
            })?;
            Callee::Builtin(function)
        }
    };
    Ok(LogicalExpression::FunctionCall {
        callee,
        name: name.to_string(),
        args,
        distinct,
    })
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
        return Err(Error::Query(QueryError::unsupported(format!(
            "{operator} needs regular expressions, and this build has no regular expressions: \
             it needs the 'regex' or 'regex-lite' feature"
        ))));
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
            assert!(check_function_call(name, 1).is_ok(), "{name}");
        }
    }

    #[test]
    fn an_unknown_function_names_the_closest_known_one() {
        let error = message(check_function_call("upperr", 1));
        assert!(error.contains("Unknown function 'upperr'"), "{error}");
        assert!(error.contains("Did you mean 'upper'?"), "{error}");
        let error = message(check_function_call("coutn", 1));
        assert!(error.contains("Did you mean 'count'?"), "{error}");
        let error = message(check_function_call("regexp_matches", 2));
        assert!(
            error.contains("Unknown function 'regexp_matches'") && !error.contains("Did you mean"),
            "{error}"
        );
    }

    #[test]
    fn a_wrong_number_of_arguments_names_the_numbers_the_function_takes() {
        let cases = [
            ("toUpper", 2, "Function 'toUpper' takes 1 argument, got 2"),
            (
                "substring",
                1,
                "Function 'substring' takes 2 or 3 arguments, got 1",
            ),
            ("TRIM", 2, "Function 'TRIM' takes 1 or 3 arguments, got 2"),
            (
                "coalesce",
                0,
                "Function 'coalesce' takes at least 1 argument, got 0",
            ),
            ("pi", 1, "Function 'pi' takes 0 arguments, got 1"),
        ];
        for (name, arguments, expected) in cases {
            assert_eq!(
                message(check_function_call(name, arguments)),
                format!("GRAFEO-Q002: semantic error: {expected}"),
                "{name} with {arguments}"
            );
        }
        for (name, arguments) in [
            ("substring", 2),
            ("substring", 3),
            ("trim", 3),
            ("coalesce", 19),
        ] {
            assert!(
                check_function_call(name, arguments).is_ok(),
                "{name} with {arguments}"
            );
        }
        // The numbers of arguments of an aggregate are not checked here.
        assert!(check_function_call("count", 0).is_ok());
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
