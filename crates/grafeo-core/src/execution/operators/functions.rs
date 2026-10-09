//! What the expression evaluator ([`ExpressionPredicate`](super::ExpressionPredicate))
//! can compute: the scalar functions it knows, and whether this build has
//! regular expressions for `=~` and LIKE.
//!
//! The evaluator answers a call it cannot compute with no value, a null, for
//! every row. The planner asks here first, so a query that calls an unknown
//! function, or matches a pattern without regular expressions, fails before
//! any row is read instead.

#[cfg(feature = "regex")]
use regex::Regex;
#[cfg(all(feature = "regex-lite", not(feature = "regex")))]
use regex_lite::Regex;

/// The scalar functions of the evaluator: every spelling it accepts, in the
/// case the documentation writes it (a call's name may use any case). One
/// string rather than a list of strings keeps the browser build small.
///
/// Keep it in step with the `eval_*_fn` methods of `filter.rs`: a test reads
/// their `match name` arms and compares.
const SCALAR_FUNCTIONS: &str = "\
    abs acos all_different asin atan atan2 byte_length cardinality ceil ceiling \
    char_length charLength coalesce cos cosine_distance cosine_similarity current_date \
    currentDate current_graph current_schema current_time currentTime current_timestamp \
    currentTimestamp date date_trunc datetime day degrees dot_product duration e edges \
    element_id elementId end_node endNode euclidean_distance exists exp floor hasLabel head \
    home_graph home_schema hour id info isAcyclic isDestination isDirected isNormalized \
    isSimple isSource isTrail isTyped keys labels last left length ln local_datetime \
    local_time localdatetime log log10 log2 lower ltrim manhattan_distance minute month \
    nodes normalize now nullif octet_length path pi pow power properties property_exists \
    property_values radians rand random range relationships replace reverse right round \
    rtrim same schema second session_user sign sin size split sqrt start_node startNode \
    string_join substring tail tan time timestamp toBool toBoolean toDate toDatetime \
    toDuration toFloat toInt toInteger toList toLower toString toTime toTypedList toUpper \
    toZonedDatetime toZonedTime trim truncate type upper vector year zoned_datetime \
    zonedDatetime zonedTime";

/// The functions the evaluator computes only in a build with a feature: the
/// name, the feature, and whether this build has it.
const FEATURE_FUNCTIONS: &[(&str, &str, bool)] = &[
    ("text_match", "text-index", cfg!(feature = "text-index")),
    ("text_score", "text-index", cfg!(feature = "text-index")),
];

/// Whether the evaluator can compute a call to a function (see
/// [`function_support`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FunctionSupport {
    /// The evaluator computes it.
    Available,
    /// The evaluator computes it in a build with this feature, which this
    /// build does not have.
    NeedsFeature(&'static str),
    /// No function has this name.
    Unknown,
}

/// Whether the evaluator can compute a call to the function `name`. Names
/// are matched without regard to case, as the evaluator matches them.
#[must_use]
pub fn function_support(name: &str) -> FunctionSupport {
    if let Some(&(_, feature, enabled)) = FEATURE_FUNCTIONS
        .iter()
        .find(|(function, _, _)| function.eq_ignore_ascii_case(name))
    {
        return if enabled {
            FunctionSupport::Available
        } else {
            FunctionSupport::NeedsFeature(feature)
        };
    }
    if SCALAR_FUNCTIONS
        .split_ascii_whitespace()
        .any(|function| function.eq_ignore_ascii_case(name))
    {
        FunctionSupport::Available
    } else {
        FunctionSupport::Unknown
    }
}

/// The names of the evaluator's scalar functions, including those this build
/// lacks a feature for, for "did you mean" hints.
pub fn function_names() -> impl Iterator<Item = &'static str> {
    SCALAR_FUNCTIONS
        .split_ascii_whitespace()
        .chain(FEATURE_FUNCTIONS.iter().map(|&(function, _, _)| function))
}

/// Whether this build has regular expressions, which `=~` and LIKE need: the
/// `regex` feature, or `regex-lite` in the browser build.
pub const REGEX_SUPPORT: bool = cfg!(any(feature = "regex", feature = "regex-lite"));

/// Why `pattern` is not a regular expression the evaluator takes for `=~`, or
/// `None` when it is one. The pattern is checked on its own, as written: the
/// evaluator puts it in a group anchored at both ends, which would accept an
/// unbalanced pattern such as `a)|(b`. Always `None` without
/// [`REGEX_SUPPORT`].
#[must_use]
pub fn regex_pattern_error(pattern: &str) -> Option<String> {
    #[cfg(any(feature = "regex", feature = "regex-lite"))]
    {
        // `regex` explains a syntax error over several lines (the pattern,
        // a caret, then `error: <reason>`); the reason is the last line.
        Regex::new(pattern).err().map(|error| {
            let text = error.to_string();
            let reason = text.lines().rev().find(|line| !line.trim().is_empty());
            let reason = reason.unwrap_or(&text).trim();
            reason.strip_prefix("error: ").unwrap_or(reason).to_string()
        })
    }
    #[cfg(not(any(feature = "regex", feature = "regex-lite")))]
    {
        let _ = pattern;
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The names in the `match name` arms of the evaluator's `eval_*_fn`
    /// methods in `filter.rs`, lowercase: the functions it computes.
    fn evaluator_function_names() -> BTreeSet<String> {
        let source = include_str!("filter.rs");
        let lines: Vec<&str> = source.lines().collect();
        let mut names = BTreeSet::new();
        let mut index = 0;
        while index < lines.len() {
            let line = lines[index];
            index += 1;
            if !(line.starts_with("    fn eval_") && line.contains("_fn(")) {
                continue;
            }
            // The method's body ends at the first line that closes it.
            let mut arm_indent = None;
            while index < lines.len() && lines[index] != "    }" {
                let body_line = lines[index];
                index += 1;
                let indent = body_line.len() - body_line.trim_start().len();
                let text = body_line.trim_start();
                if arm_indent.is_none() && text.starts_with("match name {") {
                    arm_indent = Some(indent + 4);
                } else if Some(indent) == arm_indent {
                    let pattern = text.strip_prefix("| ").unwrap_or(text);
                    if pattern.starts_with('"') {
                        let before_arrow = pattern.split("=>").next().unwrap_or(pattern);
                        for quoted in before_arrow.split('"').skip(1).step_by(2) {
                            names.insert(quoted.to_string());
                        }
                    }
                }
            }
        }
        names
    }

    #[test]
    fn the_table_lists_exactly_the_functions_the_evaluator_computes() {
        let evaluator = evaluator_function_names();
        assert!(
            evaluator.len() > 100,
            "the scan of filter.rs found only {} names: has the layout of its \
             `eval_*_fn` methods changed?",
            evaluator.len()
        );
        let table: BTreeSet<String> = function_names().map(str::to_lowercase).collect();
        let missing: Vec<&String> = evaluator.difference(&table).collect();
        let extra: Vec<&String> = table.difference(&evaluator).collect();
        assert!(
            missing.is_empty() && extra.is_empty(),
            "SCALAR_FUNCTIONS (or FEATURE_FUNCTIONS) lacks {missing:?} and has {extra:?}, \
             which the evaluator does not compute"
        );
    }

    #[test]
    fn a_function_is_found_in_any_case() {
        for name in [
            "upper",
            "UPPER",
            "toUpper",
            "TOUPPER",
            "startNode",
            "start_node",
        ] {
            assert_eq!(function_support(name), FunctionSupport::Available, "{name}");
        }
        for name in ["upperr", "regexp_matches", "", "upper lower"] {
            assert_eq!(function_support(name), FunctionSupport::Unknown, "{name:?}");
        }
    }

    #[test]
    fn a_text_function_needs_text_index() {
        let expected = if cfg!(feature = "text-index") {
            FunctionSupport::Available
        } else {
            FunctionSupport::NeedsFeature("text-index")
        };
        assert_eq!(function_support("text_score"), expected);
        assert_eq!(function_support("TEXT_MATCH"), expected);
    }

    #[cfg(any(feature = "regex", feature = "regex-lite"))]
    #[test]
    fn a_pattern_is_checked_as_written() {
        assert_eq!(regex_pattern_error(r".*(test|spec|\.git).*"), None);
        let reason = regex_pattern_error("(src").expect("an unclosed group is no pattern");
        assert!(!reason.is_empty() && !reason.contains('\n'), "{reason:?}");
        assert!(
            regex_pattern_error("a)|(b").is_some(),
            "the anchoring group must not balance the pattern"
        );
    }

    #[cfg(not(any(feature = "regex", feature = "regex-lite")))]
    #[test]
    fn without_regex_no_pattern_is_checked() {
        // The planner refuses `=~` itself in such a build (`REGEX_SUPPORT`).
        assert_eq!(regex_pattern_error("(src"), None);
    }
}
