//! What the expression evaluator ([`ExpressionPredicate`](super::ExpressionPredicate))
//! can compute: the scalar functions it knows with the numbers of arguments
//! each takes, and whether this build has regular expressions for `=~` and
//! LIKE.
//!
//! The evaluator answers a call it cannot compute with no value, a null, for
//! every row. The planner asks here first, so a query that calls an unknown
//! function, passes a function a number of arguments it does not take, or
//! matches a pattern without regular expressions, fails before any row is
//! read instead.

#[cfg(feature = "regex")]
use regex::Regex;
#[cfg(all(feature = "regex-lite", not(feature = "regex")))]
use regex_lite::Regex;

/// The scalar functions of the evaluator: every spelling it accepts, in the
/// case the documentation writes it (a call's name may use any case), each
/// followed by a slash and the numbers of arguments it takes (see [`Arity`]).
/// One string rather than a list of strings keeps the browser build small.
///
/// Keep it in step with the `eval_*_fn` methods of `filter.rs`: a test reads
/// their `match name` arms and compares.
const SCALAR_FUNCTIONS: &str = "\
    abs/1 acos/1 all_different/1+ asin/1 atan/1 atan2/2 byte_length/1 cardinality/1 ceil/1 \
    ceiling/1 char_length/1 charLength/1 coalesce/1+ cos/1 cosine_distance/2 \
    cosine_similarity/2 current_date/0 currentDate/0 current_graph/0 current_schema/0 \
    current_time/0 currentTime/0 current_timestamp/0 currentTimestamp/0 date/0-1 \
    date_trunc/2 datetime/0-1 day/1 degrees/1 dot_product/2 duration/1 e/0 edges/1 \
    element_id/1 elementId/1 end_node/1 endNode/1 euclidean_distance/2 exists/1 exp/1 \
    floor/1 hasLabel/2 head/1 home_graph/0 home_schema/0 hour/1 id/1 info/0 isAcyclic/1 \
    isDestination/2 isDirected/1 isNormalized/1-2 isSimple/1 isSource/2 isTrail/1 isTyped/2 \
    keys/1 labels/1 last/1 left/2 length/1 ln/1 local_datetime/0-1 local_time/0-1 \
    localdatetime/0-1 log/1 log10/1 log2/1 lower/1 ltrim/1 manhattan_distance/2 minute/1 \
    month/1 nodes/1 normalize/1 now/0 nullif/2 octet_length/1 path/1+ path_length/1 pi/0 \
    pow/2 power/2 properties/1 property_exists/2 property_values/1 radians/1 rand/0 \
    random/0 range/2-3 relationships/1 replace/3 reverse/1 right/2 round/1 rtrim/1 same/1+ \
    schema/0 second/1 session_user/0 sign/1 sin/1 size/1 split/2 sqrt/1 start_node/1 \
    startNode/1 string_join/2 substring/2-3 tail/1 tan/1 time/0-1 timestamp/0 toBool/1 \
    toBoolean/1 toDate/0-1 toDatetime/0-1 toDuration/1 toFloat/1 toInt/1 toInteger/1 \
    toList/1 toLower/1 toString/1 toTime/0-1 toTypedList/2 toUpper/1 toZonedDatetime/0-1 \
    toZonedTime/1 trim/1,3 truncate/2 type/1 upper/1 vector/1 year/1 zoned_datetime/0-1 \
    zonedDatetime/0-1 zonedTime/1";

/// The functions the evaluator computes only in a build with a feature: the
/// name with its numbers of arguments (as in [`SCALAR_FUNCTIONS`]), the
/// feature, and whether this build has it.
const FEATURE_FUNCTIONS: &[(&str, &str, bool)] = &[
    ("text_match/2", "text-index", cfg!(feature = "text-index")),
    ("text_score/2", "text-index", cfg!(feature = "text-index")),
];

/// The numbers of arguments a function takes, as the function table writes
/// them after the name: `2` (exactly 2), `0-1` (from 0 to 1), `1+` (1 or
/// more) or `1,3` (1 or 3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Arity(&'static str);

impl Arity {
    /// Whether a call with `count` arguments fits.
    #[must_use]
    pub fn accepts(self, count: usize) -> bool {
        let number = |text: &str| text.parse::<usize>().ok();
        if let Some(low) = self.0.strip_suffix('+') {
            number(low).is_some_and(|low| count >= low)
        } else if let Some((low, high)) = self.0.split_once('-') {
            matches!(
                (number(low), number(high)),
                (Some(low), Some(high)) if (low..=high).contains(&count)
            )
        } else {
            self.0
                .split(',')
                .any(|accepted| number(accepted) == Some(count))
        }
    }
}

impl std::fmt::Display for Arity {
    /// "1 argument", "2 or 3 arguments", "0 to 2 arguments", "1 or 3
    /// arguments", "at least 1 argument".
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let noun = |count: &str| {
            if count == "1" {
                "argument"
            } else {
                "arguments"
            }
        };
        if let Some(low) = self.0.strip_suffix('+') {
            write!(f, "at least {low} {}", noun(low))
        } else if let Some((low, high)) = self.0.split_once('-') {
            let next = matches!(
                (low.parse::<usize>(), high.parse::<usize>()),
                (Ok(low), Ok(high)) if high == low + 1
            );
            let joint = if next { "or" } else { "to" };
            write!(f, "{low} {joint} {high} arguments")
        } else if let Some((rest, last)) = self.0.rsplit_once(',') {
            write!(f, "{} or {last} arguments", rest.replace(',', ", "))
        } else {
            write!(f, "{} {}", self.0, noun(self.0))
        }
    }
}

/// The name and the numbers of arguments of an entry of the function table.
fn split_entry(entry: &'static str) -> (&'static str, Arity) {
    let (name, arity) = entry.split_once('/').unwrap_or((entry, ""));
    (name, Arity(arity))
}

/// Whether the evaluator can compute a call to a function (see
/// [`function_support`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FunctionSupport {
    /// The evaluator computes it, called with one of these numbers of
    /// arguments.
    Available(Arity),
    /// The evaluator computes it in a build with this feature, which this
    /// build does not have.
    NeedsFeature(&'static str),
    /// No function has this name.
    Unknown,
}

/// Whether the evaluator can compute a call to the function `name`, and with
/// how many arguments. Names are matched without regard to case, as the
/// evaluator matches them.
#[must_use]
pub fn function_support(name: &str) -> FunctionSupport {
    if let Some(&(entry, feature, enabled)) = FEATURE_FUNCTIONS
        .iter()
        .find(|(entry, _, _)| split_entry(entry).0.eq_ignore_ascii_case(name))
    {
        return if enabled {
            FunctionSupport::Available(split_entry(entry).1)
        } else {
            FunctionSupport::NeedsFeature(feature)
        };
    }
    SCALAR_FUNCTIONS
        .split_ascii_whitespace()
        .map(split_entry)
        .find(|(function, _)| function.eq_ignore_ascii_case(name))
        .map_or(FunctionSupport::Unknown, |(_, arity)| {
            FunctionSupport::Available(arity)
        })
}

/// The names of the evaluator's scalar functions, including those this build
/// lacks a feature for, for "did you mean" hints.
pub fn function_names() -> impl Iterator<Item = &'static str> {
    SCALAR_FUNCTIONS
        .split_ascii_whitespace()
        .chain(FEATURE_FUNCTIONS.iter().map(|&(entry, _, _)| entry))
        .map(|entry| split_entry(entry).0)
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
            assert!(
                matches!(function_support(name), FunctionSupport::Available(_)),
                "{name}"
            );
        }
        for name in [
            "upperr",
            "regexp_matches",
            "",
            "upper lower",
            "upper/1",
            "1",
        ] {
            assert_eq!(function_support(name), FunctionSupport::Unknown, "{name:?}");
        }
    }

    #[test]
    fn a_text_function_needs_text_index() {
        let expected = if cfg!(feature = "text-index") {
            FunctionSupport::Available(Arity("2"))
        } else {
            FunctionSupport::NeedsFeature("text-index")
        };
        assert_eq!(function_support("text_score"), expected);
        assert_eq!(function_support("TEXT_MATCH"), expected);
    }

    /// The numbers of arguments a call of `name` may pass, out of 0 to 4.
    fn accepted_counts(name: &str) -> Vec<usize> {
        let FunctionSupport::Available(arity) = function_support(name) else {
            panic!("{name} is not available");
        };
        (0..=4).filter(|&count| arity.accepts(count)).collect()
    }

    #[test]
    fn each_function_takes_the_numbers_of_arguments_the_evaluator_reads() {
        assert_eq!(accepted_counts("toUpper"), [1]);
        assert_eq!(accepted_counts("UPPER"), [1]);
        assert_eq!(accepted_counts("substring"), [2, 3]);
        assert_eq!(accepted_counts("trim"), [1, 3]);
        assert_eq!(accepted_counts("coalesce"), [1, 2, 3, 4]);
        assert_eq!(accepted_counts("pi"), [0]);
        assert_eq!(accepted_counts("date"), [0, 1]);
        assert_eq!(accepted_counts("path_length"), [1]);
        assert_eq!(accepted_counts("vector"), [1]);
    }

    #[test]
    fn every_entry_of_the_table_has_a_name_and_numbers_of_arguments() {
        let entries = SCALAR_FUNCTIONS
            .split_ascii_whitespace()
            .chain(FEATURE_FUNCTIONS.iter().map(|&(entry, _, _)| entry));
        for entry in entries {
            let (name, arity) = split_entry(entry);
            assert!(
                !name.is_empty() && entry.contains('/'),
                "{entry:?} has no name or no numbers of arguments"
            );
            assert!(
                (0..=8).any(|count| arity.accepts(count)),
                "{entry:?}: no number of arguments fits"
            );
        }
    }

    #[test]
    fn numbers_of_arguments_describe_themselves() {
        assert_eq!(Arity("1").to_string(), "1 argument");
        assert_eq!(Arity("0").to_string(), "0 arguments");
        assert_eq!(Arity("0-1").to_string(), "0 or 1 arguments");
        assert_eq!(Arity("1-3").to_string(), "1 to 3 arguments");
        assert_eq!(Arity("1,3").to_string(), "1 or 3 arguments");
        assert_eq!(Arity("1+").to_string(), "at least 1 argument");
        assert_eq!(Arity("2+").to_string(), "at least 2 arguments");
        assert!(!Arity("2-3").accepts(4));
        assert!(!Arity("2-3").accepts(1));
        assert!(!Arity("1,3").accepts(2));
        assert!(Arity("1+").accepts(19));
        assert!(!Arity("1+").accepts(0));
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
