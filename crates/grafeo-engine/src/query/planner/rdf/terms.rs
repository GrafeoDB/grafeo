//! RDF terms as SPARQL expressions read them: the value of a literal, the
//! operator mapping of `=`, `!=`, `<` and the others (SPARQL 1.1 section
//! 17.3), the effective boolean value (17.2.2), and the string arguments of
//! the string functions (17.4.3).
//!
//! An expression computes with [`Value`]s. A variable and a constant are a
//! [`Value::RdfTerm`]; a computed result is a native value (a boolean, a
//! number, a `String` for a simple literal, a date, time, datetime or
//! duration) or a term when it carries more than a native value holds (a
//! tagged literal, a literal of another datatype, an IRI, a blank node).

use std::cmp::Ordering;
use std::sync::Arc;

use grafeo_common::types::rdf::{Literal, Term};
use grafeo_common::types::{Duration, Value};
use grafeo_core::execution::operators::value_utils;

const XSD: &str = "http://www.w3.org/2001/XMLSchema#";

/// Whether `term` is of a kind the SPARQL engine handles: the query's term
/// dictionary refuses any other with this error, so a query that meets one
/// fails instead of reading it as something else.
///
/// # Errors
///
/// [`UnsupportedTermKind`](grafeo_common::types::rdf::UnsupportedTermKind)
/// for a term of another kind.
pub(crate) fn supported(term: &Term) -> Result<(), grafeo_common::types::rdf::UnsupportedTermKind> {
    match term {
        Term::Iri(_) | Term::BlankNode(_) | Term::Literal(_) => Ok(()),
        other => Err(grafeo_common::types::rdf::UnsupportedTermKind::of(other)),
    }
}

/// The literal of a native value, with the XSD type of the value: a string
/// as a simple literal, a number as an `xsd:integer`, `xsd:decimal` or
/// `xsd:double` in its canonical form (see [`decimal_lexical`] and
/// [`double_lexical`]), a boolean as an `xsd:boolean`, a date, time,
/// datetime or duration as its XSD type. `None` for a value with no literal
/// form (a list, a map, NULL).
pub(crate) fn native_literal(value: &Value) -> Option<Literal> {
    Some(match value {
        Value::String(s) => Literal::simple(s.as_str()),
        Value::Int64(n) => Literal::integer(*n),
        Value::Float64(f) => Literal::typed(double_lexical(*f), Literal::XSD_DOUBLE),
        Value::Decimal(d) => Literal::typed(decimal_lexical(d), Literal::XSD_DECIMAL),
        Value::Bool(b) => Literal::typed(b.to_string(), Literal::XSD_BOOLEAN),
        Value::Date(d) => Literal::typed(d.to_string(), format!("{XSD}date")),
        Value::Time(t) => Literal::typed(t.to_string(), format!("{XSD}time")),
        Value::Timestamp(ts) => Literal::typed(ts.to_string(), Literal::XSD_DATETIME),
        Value::ZonedDatetime(zdt) => Literal::typed(zdt.to_string(), Literal::XSD_DATETIME),
        Value::Duration(dur) => Literal::typed(dur.to_string(), format!("{XSD}duration")),
        _ => return None,
    })
}

/// A computed decimal as the canonical lexical form of `xsd:decimal` (XML
/// Schema 1.1, 3.3.3.2, as RDF 1.1 and the W3C results write it): no
/// trailing zeros after the point, and no point in a whole number, so
/// `1.50 + 0` is `1.5` and `4 / 2` is `2`. A value has no lexical form of its
/// own; a stored literal keeps its text.
fn decimal_lexical(value: &grafeo_common::types::Decimal) -> String {
    value.normalize().to_string()
}

/// A computed double as the canonical lexical form of `xsd:double` (XML
/// Schema 1.1, 3.3.5.2): a mantissa with one digit before the point and at
/// least one after it, then `E` and the exponent, with the fewest digits that
/// read back as the same double (`1000` is `1.0E3`, `0.25` is `2.5E-1`, zero
/// is `0.0E0`), and `INF`, `-INF` and `NaN` for the special values. A stored
/// literal keeps its text.
fn double_lexical(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 { "INF" } else { "-INF" }.to_string();
    }
    // `{:e}` writes the shortest round-trip digits: `1e3`, `2.5e-1`, `0e0`.
    exponent_lexical(&format!("{value:e}"))
}

/// A computed float as the canonical lexical form of `xsd:float` (XML Schema
/// 1.1, 3.3.4.2): as [`double_lexical`], with the fewest digits that read
/// back as the same float (`0.5` is `5.0E-1`).
pub(super) fn float_lexical(value: f32) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 { "INF" } else { "-INF" }.to_string();
    }
    exponent_lexical(&format!("{value:e}"))
}

/// Rust's shortest `{:e}` text (`1e3`, `2.5e-1`) in the XSD form: a point
/// in the mantissa and a capital `E` (`1.0E3`, `2.5E-1`).
fn exponent_lexical(shortest: &str) -> String {
    let (mantissa, exponent) = shortest.split_once('e').unwrap_or((shortest, "0"));
    if mantissa.contains('.') {
        format!("{mantissa}E{exponent}")
    } else {
        format!("{mantissa}.0E{exponent}")
    }
}

/// The term of an expression's result: a term as itself, a native value as
/// its literal (see [`native_literal`]). `None` for NULL and a value with no
/// literal form.
pub(crate) fn to_term(value: &Value) -> Option<Term> {
    match value {
        Value::RdfTerm(term) => Some(term.as_ref().clone()),
        other => native_literal(other).map(Term::Literal),
    }
}

/// A value as the operators read it: the native value of a literal term
/// (see [`Literal::native_value`]), a native value as itself. `None` for a term
/// without one (an IRI, a blank node, a tagged or other literal).
pub(crate) fn native(value: &Value) -> Option<std::borrow::Cow<'_, Value>> {
    match value {
        Value::RdfTerm(term) => match term.as_ref() {
            Term::Literal(literal) => literal.native_value().map(std::borrow::Cow::Owned),
            _ => None,
        },
        Value::Null => None,
        other => Some(std::borrow::Cow::Borrowed(other)),
    }
}

/// What an operand of `=`, `<` and the other comparisons is, by SPARQL 1.1's
/// operator mapping (17.3).
#[derive(Debug)]
pub(crate) enum Operand<'a> {
    /// A number of any XSD numeric type.
    Number(Value),
    /// A simple literal or an `xsd:string`.
    Str(&'a str),
    /// A language-tagged string: its text and tag.
    Tagged(&'a str, &'a str),
    /// An `xsd:boolean`.
    Bool(bool),
    /// An `xsd:dateTime` (zoned or local), `xsd:date`, `xsd:time` or
    /// `xsd:duration` value.
    Temporal(Value),
    /// An IRI.
    Iri(&'a str),
    /// A blank node.
    Blank(&'a str),
    /// A literal of another datatype, or one whose text is not of its type.
    OtherLiteral(&'a Literal),
    /// A term of a kind this evaluator does not know: every operator on it
    /// is an error. Not reached (see [`supported`]).
    Unsupported,
}

impl<'a> Operand<'a> {
    /// The operand `value` is; `None` for NULL (unbound, or an error) and a
    /// value no SPARQL expression has (a list, a map).
    pub(crate) fn of(value: &'a Value) -> Option<Self> {
        match value {
            Value::RdfTerm(term) => Some(Self::of_term(term, None)),
            other => Self::of_native(other),
        }
    }

    /// The operand `term` is, with its native value when the caller has it
    /// (the per-query cache of [`super::term_ids::TermEntry`]).
    pub(crate) fn of_term(term: &'a Term, native: Option<&Value>) -> Self {
        match term {
            // Not reached: the query's dictionary refuses a term of another
            // kind (see [`supported`]), the only way a term enters a plan.
            other @ (Term::Iri(_) | Term::BlankNode(_) | Term::Literal(_)) => {
                Self::known(other, native)
            }
            _ => Self::Unsupported,
        }
    }

    fn known(term: &'a Term, native: Option<&Value>) -> Self {
        match term {
            Term::Iri(iri) => Self::Iri(iri.as_str()),
            Term::BlankNode(node) => Self::Blank(node.id()),
            Term::Literal(literal) => {
                if let Some(tag) = literal.language() {
                    return Self::Tagged(literal.value(), tag);
                }
                if literal.datatype() == Literal::XSD_STRING {
                    return Self::Str(literal.value());
                }
                let kind = match native {
                    Some(native) => native_kind(native),
                    None => literal.native_value().as_ref().and_then(native_kind),
                };
                kind.unwrap_or(Self::OtherLiteral(literal))
            }
            _ => Self::Unsupported,
        }
    }

    fn of_native(value: &'a Value) -> Option<Self> {
        Some(match value {
            Value::Int64(_) | Value::Decimal(_) | Value::Float64(_) => Self::Number(value.clone()),
            Value::String(s) => Self::Str(s.as_str()),
            Value::Bool(b) => Self::Bool(*b),
            Value::Date(_)
            | Value::Time(_)
            | Value::Timestamp(_)
            | Value::ZonedDatetime(_)
            | Value::Duration(_) => Self::Temporal(value.clone()),
            _ => return None,
        })
    }

    fn is_literal(&self) -> bool {
        !matches!(self, Self::Iri(_) | Self::Blank(_))
    }
}

/// The operand kind of a literal's native value: a number, a boolean or a
/// temporal value (a string-valued literal is an `xsd:string`, classified
/// before this), owning what it holds.
fn native_kind(native: &Value) -> Option<Operand<'static>> {
    Some(match native {
        Value::Int64(_) | Value::Decimal(_) | Value::Float64(_) => Operand::Number(native.clone()),
        Value::Bool(b) => Operand::Bool(*b),
        Value::Date(_)
        | Value::Time(_)
        | Value::Timestamp(_)
        | Value::ZonedDatetime(_)
        | Value::Duration(_) => Operand::Temporal(native.clone()),
        _ => return None,
    })
}

/// The order of two numbers, exactly across integers, decimals and doubles;
/// `None` when a NaN takes part.
fn compare_numbers(a: &Value, b: &Value) -> Option<Ordering> {
    match (a, b) {
        (Value::Decimal(_), _) | (_, Value::Decimal(_)) => value_utils::compare_with_decimal(a, b),
        (Value::Int64(x), Value::Int64(y)) => Some(x.cmp(y)),
        (Value::Float64(x), Value::Float64(y)) => x.partial_cmp(y),
        (Value::Int64(x), Value::Float64(y)) => value_utils::partial_cmp_int_float(*x, *y),
        (Value::Float64(x), Value::Int64(y)) => {
            value_utils::partial_cmp_int_float(*y, *x).map(Ordering::reverse)
        }
        _ => None,
    }
}

/// The order of two temporal values of the same kind: dates, times (both
/// zoned or both local), datetimes (both zoned, by instant, or both local)
/// and durations of one kind (months only or days and time only). `None`
/// for two of different kinds, which SPARQL cannot order.
fn compare_temporal(a: &Value, b: &Value) -> Option<Ordering> {
    match (a, b) {
        (Value::Date(x), Value::Date(y)) => Some(x.cmp(y)),
        (Value::ZonedDatetime(x), Value::ZonedDatetime(y)) => Some(x.cmp(y)),
        (Value::Timestamp(x), Value::Timestamp(y)) => Some(x.cmp(y)),
        (Value::Time(x), Value::Time(y))
            if x.offset_seconds().is_some() == y.offset_seconds().is_some() =>
        {
            Some(x.cmp(y))
        }
        (Value::Duration(x), Value::Duration(y)) => {
            let months_only = |d: &Duration| d.days() == 0 && d.nanos() == 0;
            let time_only = |d: &Duration| d.months() == 0;
            if months_only(x) && months_only(y) {
                Some(x.months().cmp(&y.months()))
            } else if time_only(x) && time_only(y) {
                Some((x.days(), x.nanos()).cmp(&(y.days(), y.nanos())))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// `RDFterm-equal` (SPARQL 1.1 17.4.1.7) of two operands no value
/// comparison covers: true for the same term; a type error (`None`) for two
/// different literals one of which has a datatype the engine does not know
/// (or a text that is not of its type), as their values may still be equal;
/// false otherwise, also for literals of two different known types
/// (`"42" = 42`, `"Alix" = "Alix"@en`), whose values differ, as Oxigraph
/// reads it.
fn term_equal(a: &Operand<'_>, b: &Operand<'_>) -> Option<bool> {
    let same = match (a, b) {
        (Operand::Iri(x), Operand::Iri(y)) | (Operand::Blank(x), Operand::Blank(y)) => x == y,
        (Operand::OtherLiteral(x), Operand::OtherLiteral(y)) => x == y,
        _ => false,
    };
    let unknown = |operand: &Operand<'_>| matches!(operand, Operand::OtherLiteral(_));
    if same {
        Some(true)
    } else if a.is_literal() && b.is_literal() && (unknown(a) || unknown(b)) {
        None
    } else {
        Some(false)
    }
}

/// SPARQL `=` (17.3): numbers, strings, booleans and temporal values by
/// value; tagged strings by text and tag (the tag without case); anything
/// else by `RDFterm-equal`. `None` is a type error.
pub(crate) fn sparql_equal(a: &Operand<'_>, b: &Operand<'_>) -> Option<bool> {
    match (a, b) {
        (Operand::Number(x), Operand::Number(y)) => {
            // IEEE: NaN equals nothing, not even NaN.
            Some(compare_numbers(x, y) == Some(Ordering::Equal))
        }
        (Operand::Str(x), Operand::Str(y)) => Some(x == y),
        (Operand::Tagged(x, tag_x), Operand::Tagged(y, tag_y)) => {
            Some(x == y && tag_x.eq_ignore_ascii_case(tag_y))
        }
        (Operand::Bool(x), Operand::Bool(y)) => Some(x == y),
        (Operand::Temporal(x), Operand::Temporal(y)) => match compare_temporal(x, y) {
            Some(order) => Some(order == Ordering::Equal),
            None => term_equal(a, b),
        },
        _ => term_equal(a, b),
    }
}

/// Whether `a` and `b` are two numbers of which one is NaN: XPath's
/// `op:numeric-less-than` and `op:numeric-greater-than` are false for them,
/// not an error.
pub(crate) fn numbers_with_a_nan(a: &Operand<'_>, b: &Operand<'_>) -> bool {
    let nan =
        |operand: &Operand<'_>| matches!(operand, Operand::Number(Value::Float64(f)) if f.is_nan());
    matches!((a, b), (Operand::Number(_), Operand::Number(_))) && (nan(a) || nan(b))
}

/// The order SPARQL's `<`, `<=`, `>` and `>=` see (17.3): numbers, strings,
/// booleans (`false` before `true`) and temporal values of one kind. `None`
/// for another pair, and for two numbers with a NaN, which have no order
/// (see [`numbers_with_a_nan`]).
pub(crate) fn sparql_compare(a: &Operand<'_>, b: &Operand<'_>) -> Option<Ordering> {
    match (a, b) {
        (Operand::Number(x), Operand::Number(y)) => compare_numbers(x, y),
        (Operand::Str(x), Operand::Str(y)) => Some(x.cmp(y)),
        (Operand::Bool(x), Operand::Bool(y)) => Some(x.cmp(y)),
        (Operand::Temporal(x), Operand::Temporal(y)) => compare_temporal(x, y),
        _ => None,
    }
}

/// The effective boolean value (SPARQL 1.1 17.2.2): a boolean as itself, a
/// string as whether it is non-empty, a number as whether it is neither zero
/// nor NaN, and a boolean or numeric literal whose text is not of its type
/// (`"abc"^^xsd:integer`) as false. `None` (a type error) for anything else.
pub(crate) fn effective_boolean(value: &Value) -> Option<bool> {
    match Operand::of(value)? {
        Operand::Bool(b) => Some(b),
        Operand::Str(text) | Operand::Tagged(text, _) => Some(!text.is_empty()),
        Operand::Number(Value::Int64(n)) => Some(n != 0),
        Operand::Number(Value::Decimal(d)) => Some(d.signum() != 0),
        Operand::Number(Value::Float64(f)) => Some(f != 0.0 && !f.is_nan()),
        // A boolean or a number whose text is not of its type: false, not
        // an error (17.2.2).
        Operand::OtherLiteral(literal)
            if literal.datatype() == Literal::XSD_BOOLEAN || literal.has_numeric_datatype() =>
        {
            Some(false)
        }
        _ => None,
    }
}

/// A string argument of a string function (17.4.3): a simple literal, an
/// `xsd:string` or a tagged literal, with its tag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StringArg {
    /// The text.
    pub(crate) text: Arc<str>,
    /// The language tag of a tagged literal.
    pub(crate) language: Option<Arc<str>>,
}

impl StringArg {
    /// The string argument `value` is; `None` (a type error) for a number,
    /// an IRI, a blank node, a literal of another datatype and NULL.
    pub(crate) fn of(value: &Value) -> Option<Self> {
        match value {
            Value::String(s) => Some(Self {
                text: Arc::from(s.as_str()),
                language: None,
            }),
            Value::RdfTerm(term) => match term.as_ref() {
                Term::Literal(literal) => match literal.language() {
                    Some(tag) => Some(Self {
                        text: Arc::from(literal.value()),
                        language: Some(Arc::from(tag)),
                    }),
                    None if literal.datatype() == Literal::XSD_STRING => Some(Self {
                        text: Arc::from(literal.value()),
                        language: None,
                    }),
                    None => None,
                },
                _ => None,
            },
            _ => None,
        }
    }

    /// Whether `self` and `other` are argument-compatible (17.4.3.1.1):
    /// both untagged, both with the same tag, or `other` untagged.
    pub(crate) fn compatible_with(&self, other: &Self) -> bool {
        compatible_languages(self.language.as_deref(), other.language.as_deref())
    }

    /// A result string with this argument's tag (as `UCASE` and the other
    /// functions that keep it return): a tagged literal, or a simple one.
    pub(crate) fn with_text(&self, text: impl Into<Arc<str>>) -> Value {
        let text: Arc<str> = text.into();
        match &self.language {
            Some(tag) => Value::RdfTerm(Arc::new(Term::lang_literal(text, Arc::clone(tag)))),
            None => Value::from(&*text),
        }
    }
}

/// Whether string arguments of tags `a` and `b` are argument-compatible
/// (17.4.3.1.1): both untagged, both with the same tag, or `b` untagged.
pub(crate) fn compatible_languages(a: Option<&str>, b: Option<&str>) -> bool {
    match (a, b) {
        (_, None) => true,
        (Some(a), Some(b)) => a.eq_ignore_ascii_case(b),
        (None, Some(_)) => false,
    }
}

/// The lexical form `STR` gives (17.4.2.5): a literal's text, an IRI's text;
/// `None` for a blank node, which has no string form, and NULL.
pub(crate) fn str_of(value: &Value) -> Option<Arc<str>> {
    match value {
        Value::RdfTerm(term) => match term.as_ref() {
            Term::Literal(literal) => Some(Arc::from(literal.value())),
            Term::Iri(iri) => Some(Arc::from(iri.as_str())),
            // A blank node has no string form; another kind is not reached
            // (see [`supported`]).
            _ => None,
        },
        Value::Null => None,
        other => native_literal(other).map(|literal| Arc::from(literal.value())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::types::Decimal;

    fn term(term: Term) -> Value {
        Value::RdfTerm(Arc::new(term))
    }

    fn typed(text: &str, datatype: &str) -> Value {
        term(Term::typed_literal(text, format!("{XSD}{datatype}")))
    }

    fn equal(a: &Value, b: &Value) -> Option<bool> {
        sparql_equal(&Operand::of(a)?, &Operand::of(b)?)
    }

    fn less(a: &Value, b: &Value) -> Option<bool> {
        sparql_compare(&Operand::of(a)?, &Operand::of(b)?).map(Ordering::is_lt)
    }

    #[test]
    fn literals_read_as_values_of_their_type() {
        assert_eq!(
            native(&typed("019", "integer")).unwrap().as_ref(),
            &Value::Int64(19)
        );
        assert_eq!(
            native(&typed("+3", "int")).unwrap().as_ref(),
            &Value::Int64(3)
        );
        assert_eq!(
            native(&typed("99999999999999999999", "integer"))
                .unwrap()
                .as_ref(),
            &Value::Decimal(Decimal::parse("99999999999999999999").unwrap())
        );
        assert_eq!(
            native(&typed("1.50", "decimal")).unwrap().as_ref(),
            &Value::Decimal(Decimal::parse("1.50").unwrap())
        );
        assert_eq!(
            native(&typed("-INF", "double")).unwrap().as_ref(),
            &Value::Float64(f64::NEG_INFINITY)
        );
        assert_eq!(
            native(&typed("1", "boolean")).unwrap().as_ref(),
            &Value::Bool(true)
        );
        assert!(matches!(
            native(&typed("2026-01-15T10:00:00+01:00", "dateTime"))
                .unwrap()
                .as_ref(),
            Value::ZonedDatetime(_)
        ));
        for ill_typed in [
            typed("1.5", "integer"),
            typed("1e3", "integer"),
            typed("abc", "decimal"),
            typed("1e3", "decimal"),
            typed("infinity", "double"),
            typed("yes", "boolean"),
            typed("x", "http://example.org/custom"),
            term(Term::lang_literal("x", "en")),
            term(Term::iri("http://example.org/a")),
        ] {
            assert!(native(&ill_typed).is_none(), "{ill_typed}");
        }
    }

    #[test]
    fn numbers_compare_by_value_across_types_and_strings_as_strings() {
        assert_eq!(
            equal(&typed("1", "integer"), &typed("1.0", "decimal")),
            Some(true)
        );
        assert_eq!(
            equal(&typed("01", "integer"), &typed("1E0", "double")),
            Some(true)
        );
        assert_eq!(
            equal(&typed("1", "integer"), &Value::Float64(1.0)),
            Some(true)
        );
        assert_eq!(
            equal(&typed("NaN", "double"), &typed("NaN", "double")),
            Some(false)
        );
        assert_eq!(
            less(&typed("9", "integer"), &typed("10", "integer")),
            Some(true)
        );
        // Two simple literals compare as strings: "9" is after "10".
        assert_eq!(
            less(&term(Term::literal("9")), &term(Term::literal("10"))),
            Some(false)
        );
        assert_eq!(
            equal(&term(Term::literal("a")), &Value::from("a")),
            Some(true)
        );
    }

    #[test]
    fn a_string_against_a_number_is_unequal_and_unordered() {
        assert_eq!(
            equal(&term(Term::literal("42")), &typed("42", "integer")),
            Some(false)
        );
        assert_eq!(less(&term(Term::literal("9")), &Value::Int64(10)), None);
        assert_eq!(less(&Value::Bool(true), &Value::Int64(1)), None);
    }

    #[test]
    fn tagged_strings_are_equal_only_with_the_same_tag() {
        let en = term(Term::lang_literal("Fruit", "en"));
        assert_eq!(
            equal(&en, &term(Term::lang_literal("Fruit", "EN"))),
            Some(true)
        );
        assert_eq!(
            equal(&en, &term(Term::lang_literal("Fruit", "fr"))),
            Some(false)
        );
        // A tagged and a simple literal are of two known types: unequal.
        assert_eq!(equal(&en, &term(Term::literal("Fruit"))), Some(false));
        assert_eq!(less(&en, &term(Term::lang_literal("Gus", "en"))), None);
    }

    #[test]
    fn other_terms_are_equal_when_they_are_the_same_term() {
        let iri = term(Term::iri("http://example.org/a"));
        assert_eq!(equal(&iri, &iri.clone()), Some(true));
        assert_eq!(
            equal(&iri, &term(Term::iri("http://example.org/b"))),
            Some(false)
        );
        assert_eq!(
            equal(&iri, &term(Term::literal("http://example.org/a"))),
            Some(false)
        );
        assert_eq!(equal(&iri, &term(Term::blank("a"))), Some(false));
        let custom = typed("x", "http://example.org/custom");
        assert_eq!(equal(&custom, &custom.clone()), Some(true));
        assert_eq!(
            equal(&custom, &typed("y", "http://example.org/custom")),
            None
        );
        // A custom or ill-typed literal against another may be equal by value: an error.
        assert_eq!(equal(&custom, &term(Term::literal("x"))), None);
        assert_eq!(
            equal(&typed("abc", "integer"), &typed("1", "integer")),
            None
        );
        assert_eq!(less(&iri, &term(Term::iri("http://example.org/b"))), None);
    }

    #[test]
    fn booleans_and_datetimes_compare_by_value() {
        assert_eq!(
            equal(&typed("true", "boolean"), &typed("1", "boolean")),
            Some(true)
        );
        assert_eq!(
            less(&typed("false", "boolean"), &Value::Bool(true)),
            Some(true)
        );
        let earlier = typed("2026-01-15T10:00:00Z", "dateTime");
        let later = typed("2026-01-15T09:00:00-02:00", "dateTime");
        assert_eq!(less(&earlier, &later), Some(true));
        assert_eq!(
            equal(
                &typed("2026-01-15T10:00:00Z", "dateTime"),
                &typed("2026-01-15T11:00:00+01:00", "dateTime")
            ),
            Some(true)
        );
        // A zoned and a local datetime cannot be ordered.
        assert_eq!(
            less(&earlier, &typed("2026-01-15T10:00:00", "dateTime")),
            None
        );
    }

    #[test]
    fn the_effective_boolean_value() {
        assert_eq!(effective_boolean(&typed("true", "boolean")), Some(true));
        assert_eq!(effective_boolean(&term(Term::literal(""))), Some(false));
        assert_eq!(
            effective_boolean(&term(Term::lang_literal("x", "en"))),
            Some(true)
        );
        assert_eq!(effective_boolean(&typed("0.0", "decimal")), Some(false));
        assert_eq!(effective_boolean(&typed("NaN", "double")), Some(false));
        assert_eq!(effective_boolean(&typed("3", "integer")), Some(true));
        assert_eq!(
            effective_boolean(&term(Term::iri("http://example.org/a"))),
            None
        );
        // A number or boolean of an invalid form is false (17.2.2); a literal
        // of an unknown datatype is an error.
        assert_eq!(effective_boolean(&typed("abc", "integer")), Some(false));
        assert_eq!(effective_boolean(&typed("maybe", "boolean")), Some(false));
        assert_eq!(
            effective_boolean(&term(Term::typed_literal("x", "http://example.org/dt"))),
            None
        );
        assert_eq!(effective_boolean(&Value::Null), None);
    }

    #[test]
    fn string_arguments_keep_their_tag() {
        let en = StringArg::of(&term(Term::lang_literal("Fruit", "en"))).unwrap();
        assert_eq!(en.language.as_deref(), Some("en"));
        assert_eq!(
            en.with_text("FRUIT"),
            term(Term::lang_literal("FRUIT", "en"))
        );
        let plain = StringArg::of(&Value::from("x")).unwrap();
        assert_eq!(plain.with_text("y"), Value::from("y"));
        assert!(en.compatible_with(&plain));
        assert!(!plain.compatible_with(&en));
        assert!(StringArg::of(&typed("1", "integer")).is_none());
        assert!(StringArg::of(&term(Term::iri("http://example.org/a"))).is_none());
    }

    #[test]
    fn results_become_terms_of_their_type() {
        assert_eq!(
            to_term(&Value::Int64(3)),
            Some(Term::typed_literal("3", Literal::XSD_INTEGER))
        );
        assert_eq!(to_term(&Value::from("x")), Some(Term::literal("x")));
        assert_eq!(
            to_term(&Value::Float64(f64::NEG_INFINITY)),
            Some(Term::typed_literal("-INF", Literal::XSD_DOUBLE))
        );
        let iri = Term::iri("http://example.org/a");
        assert_eq!(to_term(&term(iri.clone())), Some(iri));
        assert_eq!(to_term(&Value::Null), None);
        assert_eq!(
            str_of(&term(Term::iri("http://example.org/a"))).as_deref(),
            Some("http://example.org/a")
        );
        assert_eq!(str_of(&typed("01", "integer")).as_deref(), Some("01"));
        assert_eq!(str_of(&term(Term::blank("b"))), None);
    }
}
