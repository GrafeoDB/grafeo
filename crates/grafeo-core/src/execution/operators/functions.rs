//! What the expression evaluator ([`ExpressionPredicate`](super::ExpressionPredicate))
//! can compute: the built-in functions ([`BuiltinFunction`]), each declared
//! once with its spellings, the numbers of arguments it takes, its evaluator
//! group, its [`Volatility`] and the feature it needs, and whether this build
//! has regular expressions for `=~` and LIKE.
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

/// How the result of a function depends on more than its arguments, as
/// PostgreSQL and DataFusion classify functions (DuckDB: CONSISTENT,
/// CONSISTENT_WITHIN_QUERY, VOLATILE).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Volatility {
    /// The same arguments give the same result. An entity argument counts as
    /// its state when the call runs: `labels(n)` is Immutable, but a rule that
    /// kept a result across a write in the same statement would have to check
    /// that write (see the planner's `after_write`).
    Immutable,
    /// Fixed for a statement: the clock (`now()`, `date()` without arguments)
    /// and the session (`current_graph()`, `session_user()`).
    Stable,
    /// May differ on every call: `rand()`.
    Volatile,
}

/// The evaluator method that computes a function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FunctionGroup {
    /// Nodes and edges: `id`, `labels`, `type`, `startNode`, ...
    GraphElement,
    /// Conversions and type checks: `toString`, `toInteger`, `isTyped`, ...
    Type,
    /// Lists and maps: `size`, `coalesce`, `keys`, `head`, `range`, ...
    Collection,
    /// Strings: `trim`, `substring`, `toUpper`, ...
    String,
    /// Numbers: `abs`, `round`, `sqrt`, `rand`, ...
    Numeric,
    /// Dates, times and durations: `date`, `now`, `year`, ...
    Temporal,
    /// Paths: `nodes`, `edges`, `path_length`, ...
    Path,
    /// Vectors: `cosine_similarity`, `dot_product`, ...
    Vector,
    /// Full-text search: `text_score`, `text_match`.
    Text,
    /// The session and the database: `session_user`, `current_graph`, ...
    Session,
}

fn immutable(_: usize) -> Volatility {
    Volatility::Immutable
}

fn stable(_: usize) -> Volatility {
    Volatility::Stable
}

fn volatile(_: usize) -> Volatility {
    Volatility::Volatile
}

/// A temporal constructor: from its argument (`date('2026-10-17')`), or from
/// the clock without one.
fn clock(arguments: usize) -> Volatility {
    if arguments == 0 {
        Volatility::Stable
    } else {
        Volatility::Immutable
    }
}

/// Declares every built-in function once: its variant, its spellings (the
/// first is the canonical name, in the case the documentation writes it; a
/// call may use any case), the numbers of arguments it takes (see
/// [`Arity`]), its evaluator group, its volatility for a number of arguments,
/// and the feature it needs with whether this build has it. An entry without
/// a volatility does not match, so a new function cannot compile without one.
macro_rules! builtin_functions {
    ($(
        $variant:ident => [$($alias:literal),+ $(,)?], $arity:literal, $group:ident, $volatility:ident
        $(, feature = $feature:literal, $enabled:expr)?;
    )+) => {
        /// A built-in function of the LPG query languages.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum BuiltinFunction {
            $(
                #[doc = concat!($("`", $alias, "` "),+)]
                $variant,
            )+
        }

        impl BuiltinFunction {
            /// Every built-in function.
            pub const ALL: &'static [BuiltinFunction] = &[$(BuiltinFunction::$variant,)+];

            /// Every spelling of the function; the first is its name.
            #[must_use]
            pub fn aliases(self) -> &'static [&'static str] {
                match self {
                    $(Self::$variant => &[$($alias,)+],)+
                }
            }

            /// The numbers of arguments it takes.
            #[must_use]
            pub fn arity(self) -> Arity {
                match self {
                    $(Self::$variant => Arity($arity),)+
                }
            }

            /// The evaluator method that computes it.
            #[must_use]
            pub fn group(self) -> FunctionGroup {
                match self {
                    $(Self::$variant => FunctionGroup::$group,)+
                }
            }

            /// How its result depends on more than its `arguments` arguments.
            #[must_use]
            pub fn volatility(self, arguments: usize) -> Volatility {
                match self {
                    $(Self::$variant => $volatility(arguments),)+
                }
            }

            /// The feature it needs and whether this build has it; `None` for
            /// a function every build computes.
            #[must_use]
            pub fn required_feature(self) -> Option<(&'static str, bool)> {
                match self {
                    $(Self::$variant => builtin_functions!(@feature $($feature, $enabled)?),)+
                }
            }
        }
    };
    (@feature) => { None };
    (@feature $feature:literal, $enabled:expr) => { Some(($feature, $enabled)) };
}

builtin_functions! {
    Id => ["id"], "1", GraphElement, immutable;
    ElementId => ["element_id", "elementId"], "1", GraphElement, immutable;
    Labels => ["labels"], "1", GraphElement, immutable;
    Type => ["type"], "1", GraphElement, immutable;
    StartNode => ["startNode", "start_node"], "1", GraphElement, immutable;
    EndNode => ["endNode", "end_node"], "1", GraphElement, immutable;
    PropertyExists => ["property_exists"], "2", GraphElement, immutable;
    HasLabel => ["hasLabel"], "2", GraphElement, immutable;
    IsSource => ["isSource"], "2", GraphElement, immutable;
    IsDestination => ["isDestination"], "2", GraphElement, immutable;
    IsDirected => ["isDirected"], "1", GraphElement, immutable;
    ToString => ["toString"], "1", Type, immutable;
    ToInteger => ["toInteger", "toInt"], "1", Type, immutable;
    ToFloat => ["toFloat"], "1", Type, immutable;
    ToDecimal => ["toDecimal"], "1,3", Type, immutable;
    ToBoolean => ["toBoolean", "toBool"], "1", Type, immutable;
    ToList => ["toList"], "1", Type, immutable;
    ToTypedList => ["toTypedList"], "2", Type, immutable;
    IsTyped => ["isTyped"], "2", Type, immutable;
    Size => ["size", "length", "cardinality"], "1", Collection, immutable;
    Coalesce => ["coalesce"], "1+", Collection, immutable;
    Exists => ["exists"], "1", Collection, immutable;
    Keys => ["keys"], "1", Collection, immutable;
    Properties => ["properties"], "1", Collection, immutable;
    PropertyValues => ["property_values"], "1", Collection, immutable;
    Head => ["head"], "1", Collection, immutable;
    Tail => ["tail"], "1", Collection, immutable;
    Last => ["last"], "1", Collection, immutable;
    Reverse => ["reverse"], "1", Collection, immutable;
    Vector => ["vector"], "1", Collection, immutable;
    AllDifferent => ["all_different"], "1+", Collection, immutable;
    Same => ["same"], "1+", Collection, immutable;
    Range => ["range"], "2-3", Collection, immutable;
    StringJoin => ["string_join"], "2", Collection, immutable;
    Trim => ["trim"], "1,3", String, immutable;
    Ltrim => ["ltrim"], "1", String, immutable;
    Rtrim => ["rtrim"], "1", String, immutable;
    Replace => ["replace"], "3", String, immutable;
    Substring => ["substring"], "2-3", String, immutable;
    Split => ["split"], "2", String, immutable;
    Upper => ["toUpper", "upper"], "1", String, immutable;
    Lower => ["toLower", "lower"], "1", String, immutable;
    CharLength => ["char_length", "charLength"], "1", String, immutable;
    Left => ["left"], "2", String, immutable;
    Right => ["right"], "2", String, immutable;
    OctetLength => ["octet_length", "byte_length"], "1", String, immutable;
    Normalize => ["normalize"], "1-2", String, immutable;
    IsNormalized => ["isNormalized"], "1-2", String, immutable;
    Abs => ["abs"], "1", Numeric, immutable;
    Ceil => ["ceil", "ceiling"], "1", Numeric, immutable;
    Floor => ["floor"], "1", Numeric, immutable;
    Round => ["round"], "1", Numeric, immutable;
    Sqrt => ["sqrt"], "1", Numeric, immutable;
    Sign => ["sign"], "1", Numeric, immutable;
    Power => ["power", "pow"], "2", Numeric, immutable;
    Rand => ["rand", "random"], "0", Numeric, volatile;
    Ln => ["log", "ln"], "1", Numeric, immutable;
    Log10 => ["log10"], "1", Numeric, immutable;
    Log2 => ["log2"], "1", Numeric, immutable;
    Exp => ["exp"], "1", Numeric, immutable;
    E => ["e"], "0", Numeric, immutable;
    Pi => ["pi"], "0", Numeric, immutable;
    Sin => ["sin"], "1", Numeric, immutable;
    Cos => ["cos"], "1", Numeric, immutable;
    Tan => ["tan"], "1", Numeric, immutable;
    Asin => ["asin"], "1", Numeric, immutable;
    Acos => ["acos"], "1", Numeric, immutable;
    Atan => ["atan"], "1", Numeric, immutable;
    Atan2 => ["atan2"], "2", Numeric, immutable;
    Degrees => ["degrees"], "1", Numeric, immutable;
    Radians => ["radians"], "1", Numeric, immutable;
    Date => ["date", "toDate"], "0-1", Temporal, clock;
    Time => ["time", "toTime", "local_time", "localtime"], "0-1", Temporal, clock;
    Datetime => ["datetime", "localdatetime", "local_datetime", "toDatetime"], "0-1", Temporal, clock;
    Duration => ["duration", "toDuration"], "1", Temporal, immutable;
    ZonedDatetime => ["toZonedDatetime", "zonedDatetime", "zoned_datetime"], "0-1", Temporal, clock;
    ZonedTime => ["toZonedTime", "zonedTime", "zoned_time"], "0-1", Temporal, clock;
    CurrentDate => ["current_date", "currentDate"], "0", Temporal, stable;
    CurrentTime => ["current_time", "currentTime"], "0", Temporal, stable;
    Now => ["now", "current_timestamp", "currentTimestamp"], "0", Temporal, stable;
    Timestamp => ["timestamp"], "0", Temporal, stable;
    Year => ["year"], "1", Temporal, immutable;
    Month => ["month"], "1", Temporal, immutable;
    Day => ["day"], "1", Temporal, immutable;
    Hour => ["hour"], "1", Temporal, immutable;
    Minute => ["minute"], "1", Temporal, immutable;
    Second => ["second"], "1", Temporal, immutable;
    DateTrunc => ["date_trunc", "truncate"], "2", Temporal, immutable;
    Path => ["path"], "1+", Path, immutable;
    PathLength => ["path_length"], "1", Path, immutable;
    Nodes => ["nodes"], "1", Path, immutable;
    Edges => ["edges", "relationships"], "1", Path, immutable;
    IsAcyclic => ["isAcyclic"], "1", Path, immutable;
    IsSimple => ["isSimple"], "1", Path, immutable;
    IsTrail => ["isTrail"], "1", Path, immutable;
    CosineSimilarity => ["cosine_similarity"], "2", Vector, immutable;
    CosineDistance => ["cosine_distance"], "2", Vector, immutable;
    EuclideanDistance => ["euclidean_distance"], "2", Vector, immutable;
    DotProduct => ["dot_product"], "2", Vector, immutable;
    ManhattanDistance => ["manhattan_distance"], "2", Vector, immutable;
    TextScore => ["text_score"], "2", Text, immutable, feature = "text-index", cfg!(feature = "text-index");
    TextMatch => ["text_match"], "2", Text, immutable, feature = "text-index", cfg!(feature = "text-index");
    SessionUser => ["session_user"], "0", Session, stable;
    CurrentSchema => ["current_schema"], "0", Session, stable;
    CurrentGraph => ["current_graph"], "0", Session, stable;
    HomeSchema => ["home_schema"], "0", Session, stable;
    HomeGraph => ["home_graph"], "0", Session, stable;
    Info => ["info"], "0", Session, stable;
    Schema => ["schema"], "0", Session, stable;
    Nullif => ["nullif"], "2", Session, immutable;
}

impl BuiltinFunction {
    /// The function a call names, in any case.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|function| {
            function
                .aliases()
                .iter()
                .any(|alias| alias.eq_ignore_ascii_case(name))
        })
    }

    /// The canonical name.
    #[must_use]
    pub fn name(self) -> &'static str {
        self.aliases()[0]
    }
}

/// The numbers of arguments a function takes, as the declaration of
/// [`BuiltinFunction`] writes them: `2` (exactly 2), `0-1` (from 0 to 1),
/// `1+` (1 or more) or `1,3` (1 or 3).
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
    match BuiltinFunction::from_name(name) {
        None => FunctionSupport::Unknown,
        Some(function) => match function.required_feature() {
            Some((feature, false)) => FunctionSupport::NeedsFeature(feature),
            _ => FunctionSupport::Available(function.arity()),
        },
    }
}

/// The names of the evaluator's scalar functions, including those this build
/// lacks a feature for, for "did you mean" hints.
pub fn function_names() -> impl Iterator<Item = &'static str> {
    BuiltinFunction::ALL
        .iter()
        .flat_map(|function| function.aliases().iter().copied())
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
    use std::collections::{BTreeMap, BTreeSet};

    /// The variants in the `match function` arms of each `eval_*_fn` method
    /// of `filter.rs`, by the method's group (`eval_graph_element_fn` is
    /// `GraphElement`).
    fn evaluator_arms() -> BTreeMap<String, BTreeSet<String>> {
        let source = include_str!("filter.rs");
        let lines: Vec<&str> = source.lines().collect();
        let mut arms: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let mut index = 0;
        while index < lines.len() {
            let line = lines[index];
            index += 1;
            let Some(method) = line
                .strip_prefix("    fn eval_")
                .and_then(|rest| rest.split_once("_fn("))
                .map(|(method, _)| method)
            else {
                continue;
            };
            let group: String = method
                .split('_')
                .map(|part| {
                    let mut chars = part.chars();
                    chars
                        .next()
                        .map(|first| first.to_uppercase().chain(chars).collect::<String>())
                        .unwrap_or_default()
                })
                .collect();
            let entry = arms.entry(group).or_default();
            // The method's body ends at the first line that closes it.
            let mut arm_indent = None;
            while index < lines.len() && lines[index] != "    }" {
                let body_line = lines[index];
                index += 1;
                let indent = body_line.len() - body_line.trim_start().len();
                let text = body_line.trim_start();
                if arm_indent.is_none() && text.starts_with("match function {") {
                    arm_indent = Some(indent + 4);
                } else if Some(indent) == arm_indent {
                    let pattern = text.strip_prefix("| ").unwrap_or(text);
                    let before_arrow = pattern.split("=>").next().unwrap_or(pattern);
                    for part in before_arrow.split('|') {
                        if let Some(variant) = part.trim().strip_prefix("BuiltinFunction::") {
                            entry.insert(variant.trim().to_string());
                        }
                    }
                }
            }
        }
        arms
    }

    /// Each group method answers every function of its group: a function
    /// without an arm would fall to the method's `_ => None` and be null for
    /// every row.
    #[test]
    fn every_function_has_an_arm_in_its_group_method() {
        let arms = evaluator_arms();
        let found: usize = arms.values().map(BTreeSet::len).sum();
        assert!(
            found >= BuiltinFunction::ALL.len(),
            "the scan of filter.rs found {found} arms: has the layout of its \
             `eval_*_fn` methods changed?"
        );
        for &function in BuiltinFunction::ALL {
            let group = format!("{:?}", function.group());
            let variant = format!("{function:?}");
            assert!(
                arms.get(&group)
                    .is_some_and(|names| names.contains(&variant)),
                "{variant} belongs to {group}, whose evaluator method has no arm for it"
            );
        }
        for (group, names) in &arms {
            for name in names {
                let function = BuiltinFunction::ALL
                    .iter()
                    .find(|function| format!("{function:?}") == *name)
                    .unwrap_or_else(|| panic!("{name} in {group} is no BuiltinFunction"));
                assert_eq!(&format!("{:?}", function.group()), group, "{name}");
            }
        }
    }

    #[test]
    fn every_alias_resolves_in_any_case() {
        for &function in BuiltinFunction::ALL {
            for alias in function.aliases() {
                for spelling in [
                    alias.to_string(),
                    alias.to_uppercase(),
                    alias.to_lowercase(),
                ] {
                    assert_eq!(
                        BuiltinFunction::from_name(&spelling),
                        Some(function),
                        "{spelling}"
                    );
                }
            }
            assert_eq!(BuiltinFunction::from_name(function.name()), Some(function));
        }
        assert_eq!(BuiltinFunction::from_name("upperr"), None);
        assert_eq!(BuiltinFunction::from_name("upper/1"), None);
        assert_eq!(BuiltinFunction::from_name(""), None);
    }

    #[test]
    fn no_alias_names_two_functions() {
        let mut seen = std::collections::HashMap::new();
        for &function in BuiltinFunction::ALL {
            for alias in function.aliases() {
                let previous = seen.insert(alias.to_lowercase(), function);
                assert!(previous.is_none_or(|p| p == function), "{alias}");
            }
        }
        assert_eq!(BuiltinFunction::ALL.len(), 110);
    }

    #[test]
    fn volatility_of_the_clock_random_and_entity_functions() {
        use Volatility::{Immutable, Stable, Volatile};
        let of = |name: &str, arguments| {
            BuiltinFunction::from_name(name)
                .unwrap()
                .volatility(arguments)
        };
        assert_eq!(of("rand", 0), Volatile);
        assert_eq!(of("random", 0), Volatile);
        assert_eq!(of("now", 0), Stable);
        assert_eq!(of("current_date", 0), Stable);
        assert_eq!(of("session_user", 0), Stable);
        assert_eq!(of("date", 0), Stable);
        assert_eq!(of("date", 1), Immutable);
        assert_eq!(of("datetime", 0), Stable);
        assert_eq!(of("datetime", 1), Immutable);
        assert_eq!(of("upper", 1), Immutable);
        assert_eq!(of("labels", 1), Immutable);
        assert_eq!(of("properties", 1), Immutable);
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
    fn every_function_has_a_name_and_numbers_of_arguments() {
        for &function in BuiltinFunction::ALL {
            assert!(!function.name().is_empty(), "{function:?} has no name");
            assert!(
                (0..=8).any(|count| function.arity().accepts(count)),
                "{function:?}: no number of arguments fits"
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
