//! The order of RDF terms in SPARQL `ORDER BY` (SPARQL 1.1 section 15.1).

use std::cmp::Ordering;
use std::sync::Arc;

use super::rdf::{Literal, Term};
use super::{Date, Decimal, TermId, Value, ZonedDatetime};

const XSD_FLOAT: &str = "http://www.w3.org/2001/XMLSchema#float";
const XSD_DATE: &str = "http://www.w3.org/2001/XMLSchema#date";

/// `xsd:integer` and the XSD types derived from it.
pub(crate) const XSD_INTEGER_TYPES: [&str; 13] = [
    "http://www.w3.org/2001/XMLSchema#integer",
    "http://www.w3.org/2001/XMLSchema#nonPositiveInteger",
    "http://www.w3.org/2001/XMLSchema#negativeInteger",
    "http://www.w3.org/2001/XMLSchema#long",
    "http://www.w3.org/2001/XMLSchema#int",
    "http://www.w3.org/2001/XMLSchema#short",
    "http://www.w3.org/2001/XMLSchema#byte",
    "http://www.w3.org/2001/XMLSchema#nonNegativeInteger",
    "http://www.w3.org/2001/XMLSchema#unsignedLong",
    "http://www.w3.org/2001/XMLSchema#unsignedInt",
    "http://www.w3.org/2001/XMLSchema#unsignedShort",
    "http://www.w3.org/2001/XMLSchema#unsignedByte",
    "http://www.w3.org/2001/XMLSchema#positiveInteger",
];

/// Orders two RDF terms for SPARQL `ORDER BY` (SPARQL 1.1 section 15.1): blank
/// nodes, then IRIs, then literals.
///
/// Literals fall in four groups, in this order: numbers (every XSD numeric
/// type, by value: integers and decimals exactly, a double by the decimal it
/// prints as), `xsd:dateTime` values with a time zone (by instant),
/// `xsd:date` values (by day), then every other literal. A literal whose text
/// does not parse as its type (`"abc"^^xsd:integer`) is in the last group.
/// Within a group, equal values order by lexical form, then datatype IRI,
/// then language tag, so the order is total and a sort never sees two
/// different terms as equal.
#[must_use]
pub fn sparql_order(a: &Term, b: &Term) -> Ordering {
    term_rank(a).cmp(&term_rank(b)).then_with(|| match (a, b) {
        (Term::BlankNode(x), Term::BlankNode(y)) => x.id().cmp(y.id()),
        (Term::Iri(x), Term::Iri(y)) => x.as_str().cmp(y.as_str()),
        (Term::Literal(x), Term::Literal(y)) => compare_literals(x, y),
        _ => Ordering::Equal,
    })
}

fn term_rank(term: &Term) -> u8 {
    match term {
        Term::BlankNode(_) => 0,
        Term::Iri(_) => 1,
        Term::Literal(_) => 2,
    }
}

/// A number, kept exact: a decimal where it fits one, otherwise an integer
/// of 39 digits or a float (an infinity, NaN, or a float whose value no
/// decimal of 38 digits holds).
#[derive(Debug, Clone, Copy)]
enum Number {
    Decimal(Decimal),
    Integer(i128),
    Float(f64),
}

impl Number {
    const fn of_integer(integer: i128) -> Self {
        Self::Integer(integer)
    }

    fn of_float(float: f64) -> Self {
        match Decimal::from_f64(float) {
            // Only when the decimal is the float's value: 1e-300 rounds to 0
            Ok(decimal) if decimal.to_f64().total_cmp(&float).is_eq() => Self::Decimal(decimal),
            _ => Self::Float(float),
        }
    }

    fn order(self, other: Self) -> Ordering {
        match (self, other) {
            (Self::Decimal(x), Self::Decimal(y)) => x.cmp(&y),
            (Self::Integer(x), Self::Integer(y)) => x.cmp(&y),
            (Self::Float(x), Self::Float(y)) => compare_floats(x, y),
            (Self::Decimal(x), Self::Float(y)) => x.cmp_f64(y),
            (Self::Float(x), Self::Decimal(y)) => y.cmp_f64(x).reverse(),
            (Self::Integer(x), Self::Float(y)) => compare_integer_float(x, y),
            (Self::Float(x), Self::Integer(y)) => compare_integer_float(y, x).reverse(),
            // An integer that is no decimal has 39 digits: beyond every decimal
            (Self::Decimal(x), Self::Integer(y)) => {
                Decimal::new(y, 0).map_or_else(|_| 0.cmp(&y), |y| x.cmp(&y))
            }
            (Self::Integer(x), Self::Decimal(y)) => {
                Decimal::new(x, 0).map_or_else(|_| x.cmp(&0), |x| x.cmp(&y))
            }
        }
    }
}

/// A literal's value, for the literals whose value orders them.
#[derive(Debug, Clone)]
enum LiteralKey {
    Number(Number),
    DateTime(ZonedDatetime),
    Date(Date),
    Other,
}

impl LiteralKey {
    fn of(literal: &Literal) -> Self {
        let text = literal.value().trim();
        let datatype = literal.datatype();
        if XSD_INTEGER_TYPES.contains(&datatype) {
            return text.parse::<i128>().map_or(Self::Other, |integer| {
                Self::Number(Number::of_integer(integer))
            });
        }
        if datatype == Literal::XSD_DECIMAL {
            return match Decimal::parse(text) {
                Ok(decimal) => Self::Number(Number::Decimal(decimal)),
                // More than 38 digits: still a number, by its nearest float
                Err(_) => float_key(text),
            };
        }
        if datatype == Literal::XSD_DOUBLE || datatype == XSD_FLOAT {
            return float_key(text);
        }
        if datatype == Literal::XSD_DATETIME {
            return ZonedDatetime::parse(text).map_or(Self::Other, Self::DateTime);
        }
        if datatype == XSD_DATE {
            return Date::parse(text).map_or(Self::Other, Self::Date);
        }
        Self::Other
    }

    /// The key of a literal whose value [`Literal::native_value`] gives as
    /// `native`: the same key [`of`](Self::of) parses from its text.
    fn of_native(native: &Value) -> Self {
        match native {
            Value::Int64(integer) => Self::Number(Number::of_integer(i128::from(*integer))),
            Value::Decimal(decimal) => Self::Number(Number::Decimal(*decimal)),
            Value::Float64(float) => Self::Number(Number::of_float(*float)),
            Value::ZonedDatetime(datetime) => Self::DateTime(*datetime),
            Value::Date(date) => Self::Date(*date),
            // A boolean, a string, a local dateTime, a time or a duration
            _ => Self::Other,
        }
    }

    fn group(&self) -> u8 {
        match self {
            Self::Number(_) => 0,
            Self::DateTime(_) => 1,
            Self::Date(_) => 2,
            Self::Other => 3,
        }
    }
}

fn float_key(text: &str) -> LiteralKey {
    text.parse::<f64>().map_or(LiteralKey::Other, |float| {
        LiteralKey::Number(Number::of_float(float))
    })
}

fn compare_literals(a: &Literal, b: &Literal) -> Ordering {
    compare_literal_keys(a, &LiteralKey::of(a), b, &LiteralKey::of(b))
}

fn compare_literal_keys(
    a: &Literal,
    key_a: &LiteralKey,
    b: &Literal,
    key_b: &LiteralKey,
) -> Ordering {
    compare_values(key_a, key_b).then_with(|| compare_lexical(a, b))
}

/// The group and the value of two literals' keys.
fn compare_values(key_a: &LiteralKey, key_b: &LiteralKey) -> Ordering {
    key_a
        .group()
        .cmp(&key_b.group())
        .then_with(|| match (key_a, key_b) {
            (LiteralKey::Number(x), LiteralKey::Number(y)) => x.order(*y),
            (LiteralKey::DateTime(x), LiteralKey::DateTime(y)) => x.cmp(y),
            (LiteralKey::Date(x), LiteralKey::Date(y)) => x.cmp(y),
            _ => Ordering::Equal,
        })
}

/// The tie of two literals of equal values: lexical form, datatype, tag.
fn compare_lexical(a: &Literal, b: &Literal) -> Ordering {
    a.value()
        .cmp(b.value())
        .then_with(|| a.datatype().cmp(b.datatype()))
        .then_with(|| a.language().cmp(&b.language()))
}

/// A term's place in the SPARQL order ([`sparql_order`]), with a literal's
/// value parsed once: a sort of many terms computes each term's key once,
/// where comparing terms parses both literals at every comparison.
#[derive(Debug, Clone)]
pub struct SparqlOrderKey {
    source: KeySource,
    literal: Option<LiteralKey>,
}

/// What a key orders: a term, or a literal an id inlines (whose text is made
/// only to break a tie of equal values).
#[derive(Debug, Clone)]
enum KeySource {
    Term(Arc<Term>),
    Inline(TermId),
}

impl SparqlOrderKey {
    /// The key of `term`.
    #[must_use]
    pub fn new(term: Arc<Term>) -> Self {
        let literal = term.as_literal().map(LiteralKey::of);
        Self {
            source: KeySource::Term(term),
            literal,
        }
    }

    /// The key of `term`, whose value [`Literal::native_value`] gives as
    /// `native` (computed once, as a dictionary keeps it): a literal with a
    /// value is keyed without parsing its text again.
    #[must_use]
    pub fn with_native(term: Arc<Term>, native: Option<&Value>) -> Self {
        let literal = term.as_literal().map(|literal| match native {
            Some(native) => LiteralKey::of_native(native),
            None => LiteralKey::of(literal),
        });
        Self {
            source: KeySource::Term(term),
            literal,
        }
    }

    /// The key of the literal an inlined id holds, without making the term;
    /// `None` for a dictionary or query id.
    #[must_use]
    pub fn of_inline(id: TermId) -> Option<Self> {
        let native = id.inline_value()?;
        Some(Self {
            source: KeySource::Inline(id),
            literal: Some(LiteralKey::of_native(&native)),
        })
    }

    /// Runs `f` with the term, made for an inlined id.
    fn with_term<R>(&self, f: impl FnOnce(&Term) -> R) -> R {
        match &self.source {
            KeySource::Term(term) => f(term),
            KeySource::Inline(id) => f(&id
                .inline_term()
                .expect("a key of an inlined id has its term")),
        }
    }

    fn rank(&self) -> u8 {
        match &self.source {
            KeySource::Term(term) => term_rank(term),
            KeySource::Inline(_) => 2,
        }
    }
}

impl Ord for SparqlOrderKey {
    /// The order [`sparql_order`] gives the two terms.
    fn cmp(&self, other: &Self) -> Ordering {
        self.rank().cmp(&other.rank()).then_with(|| {
            if let (Some(key_a), Some(key_b)) = (&self.literal, &other.literal) {
                return compare_values(key_a, key_b).then_with(|| {
                    self.with_term(|a| {
                        other.with_term(|b| match (a, b) {
                            (Term::Literal(x), Term::Literal(y)) => compare_lexical(x, y),
                            _ => Ordering::Equal,
                        })
                    })
                });
            }
            self.with_term(|a| {
                other.with_term(|b| match (a, b) {
                    (Term::BlankNode(x), Term::BlankNode(y)) => x.id().cmp(y.id()),
                    (Term::Iri(x), Term::Iri(y)) => x.as_str().cmp(y.as_str()),
                    (Term::Literal(x), Term::Literal(y)) => compare_literals(x, y),
                    _ => Ordering::Equal,
                })
            })
        })
    }
}

impl PartialOrd for SparqlOrderKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for SparqlOrderKey {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for SparqlOrderKey {}

/// Floats numerically, NaN after every number; `-0.0` equals `0.0`.
fn compare_floats(a: f64, b: f64) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
    }
}

/// An integer against a float, exactly (no rounding of the integer to f64);
/// NaN after every integer.
fn compare_integer_float(integer: i128, float: f64) -> Ordering {
    /// 2^127: every float at or above it is larger than every i128.
    const LIMIT: f64 = 170_141_183_460_469_231_731_687_303_715_884_105_728.0;
    if float.is_nan() || float >= LIMIT {
        return Ordering::Less;
    }
    if float < -LIMIT {
        return Ordering::Greater;
    }
    let whole = float.trunc();
    #[allow(
        clippy::cast_possible_truncation,
        reason = "-2^127 <= whole < 2^127 (checked above) and whole has no fraction, so the cast is exact"
    )]
    let whole_integer = whole as i128;
    integer.cmp(&whole_integer).then_with(|| {
        0.0_f64
            .partial_cmp(&(float - whole))
            .unwrap_or(Ordering::Equal)
    })
}

#[cfg(test)]
mod tests {
    use super::{SparqlOrderKey, sparql_order};
    use crate::types::rdf::{Literal, Term};
    use std::cmp::Ordering;

    const XSD: &str = "http://www.w3.org/2001/XMLSchema#";

    fn typed(value: &str, datatype: &str) -> Term {
        Term::typed_literal(value, format!("{XSD}{datatype}"))
    }

    fn sorted(mut terms: Vec<Term>) -> Vec<Term> {
        terms.sort_by(sparql_order);
        terms
    }

    fn lexical(terms: &[Term]) -> Vec<&str> {
        terms
            .iter()
            .map(|term| term.as_literal().unwrap().value())
            .collect()
    }

    /// Every pair and triple of a sorted list must agree with the sort: a
    /// check that the order is consistent and transitive (a sort with an
    /// inconsistent order may panic).
    fn assert_consistent(terms: &[Term]) {
        for (i, a) in terms.iter().enumerate() {
            for b in &terms[i..] {
                assert_ne!(sparql_order(a, b), Ordering::Greater, "{a} after {b}");
                assert_eq!(
                    sparql_order(a, b),
                    sparql_order(b, a).reverse(),
                    "{a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn blank_nodes_then_iris_then_literals() {
        let terms = sorted(vec![
            Term::literal("a"),
            Term::iri("http://example.org/b"),
            Term::blank("c"),
        ]);
        assert_eq!(
            terms,
            vec![
                Term::blank("c"),
                Term::iri("http://example.org/b"),
                Term::literal("a")
            ]
        );
    }

    #[test]
    fn numbers_sort_by_value_across_numeric_types() {
        let terms = sorted(vec![
            typed("10", "integer"),
            typed("9", "int"),
            typed("2.5", "decimal"),
            typed("-1E0", "double"),
            typed("+3", "integer"),
            typed("2.25", "float"),
        ]);
        assert_eq!(
            lexical(&terms),
            vec!["-1E0", "2.25", "2.5", "+3", "9", "10"]
        );
    }

    #[test]
    fn decimals_compare_exactly() {
        // 0.1 + 0.2 as doubles is 0.30000000000000004; as decimals, these
        // three are three different numbers in this order.
        let terms = sorted(vec![
            typed("0.30000000000000000000000000000000000001", "decimal"),
            typed("0.3", "decimal"),
            typed("0.29999999999999999999999999999999999999", "decimal"),
        ]);
        assert_eq!(
            lexical(&terms),
            vec![
                "0.29999999999999999999999999999999999999",
                "0.3",
                "0.30000000000000000000000000000000000001"
            ]
        );
        // 1.50 and 1.5 are one number: the lexical form breaks the tie
        assert_eq!(
            sparql_order(&typed("1.50", "decimal"), &typed("1.5", "decimal")),
            Ordering::Greater
        );
    }

    #[test]
    fn big_integers_compare_exactly_with_doubles_and_decimals() {
        let terms = sorted(vec![
            typed("9007199254740993", "integer"),
            typed("9007199254740992", "double"),
            typed("1e30", "double"),
            typed("-0.0", "double"),
            typed("0", "integer"),
            typed("170141183460469231731687303715884105727", "integer"),
            typed("99999999999999999999999999999999999999", "decimal"),
            typed("-170141183460469231731687303715884105727", "integer"),
        ]);
        assert_consistent(&terms);
        assert_eq!(
            lexical(&terms),
            vec![
                "-170141183460469231731687303715884105727",
                "-0.0",
                "0",
                "9007199254740992",
                "9007199254740993",
                "1e30",
                "99999999999999999999999999999999999999",
                "170141183460469231731687303715884105727",
            ]
        );
    }

    #[test]
    fn nan_after_every_number_and_infinity_after_every_finite_one() {
        let terms = sorted(vec![
            typed("NaN", "double"),
            typed("INF", "double"),
            typed("1e308", "double"),
            typed("-INF", "double"),
            typed("170141183460469231731687303715884105727", "integer"),
            typed("3", "decimal"),
            typed("1e-300", "double"),
        ]);
        assert_consistent(&terms);
        assert_eq!(
            lexical(&terms),
            vec![
                "-INF",
                "1e-300",
                "3",
                "170141183460469231731687303715884105727",
                "1e308",
                "INF",
                "NaN"
            ]
        );
    }

    #[test]
    fn ill_typed_numbers_sort_as_text_after_the_numbers() {
        let terms = sorted(vec![
            typed("abc", "integer"),
            typed("1e3", "integer"),
            typed("7", "integer"),
            typed("1.2.3", "decimal"),
            Term::literal("5"),
        ]);
        assert_consistent(&terms);
        assert_eq!(terms[0], typed("7", "integer"));
    }

    #[test]
    fn date_times_sort_by_instant() {
        let later_but_smaller_text = typed("2026-01-15T09:00:00-02:00", "dateTime");
        let earlier = typed("2026-01-15T10:00:00Z", "dateTime");
        assert_eq!(
            sparql_order(&earlier, &later_but_smaller_text),
            Ordering::Less
        );
        assert_eq!(
            sparql_order(&typed("2026-01-02", "date"), &typed("2026-01-10", "date")),
            Ordering::Less
        );
    }

    #[test]
    fn mixed_literals_form_a_total_order() {
        let mut terms = vec![
            typed("10", "integer"),
            typed("9", "integer"),
            typed("9.0", "decimal"),
            typed("9E0", "double"),
            typed("NaN", "float"),
            Term::literal("5"),
            Term::literal("10"),
            Term::lang_literal("Fruit", "en"),
            Term::lang_literal("Fruit", "fr"),
            Term::literal("Fruit"),
            typed("2026-01-15", "date"),
            typed("2026-01-15T10:00:00Z", "dateTime"),
            typed("x", "http://example.org/custom"),
            Term::literal(""),
            Term::blank("b0"),
            Term::iri("http://example.org/a"),
        ];
        terms.extend(crate::types::rdf::testing::unusual_terms());
        let terms = sorted(terms);
        assert_consistent(&terms);
    }

    #[test]
    fn same_value_breaks_ties_by_text_then_datatype_then_language() {
        let english = Term::lang_literal("Fruit", "en");
        let french = Term::lang_literal("Fruit", "fr");
        assert_eq!(sparql_order(&english, &french), Ordering::Less);
        assert_eq!(sparql_order(&english, &english.clone()), Ordering::Equal);
        assert_ne!(
            sparql_order(
                &Term::literal("1"),
                &Term::typed_literal("1", Literal::XSD_INTEGER)
            ),
            Ordering::Equal
        );
        // 9, 9.0 and 9E0 are one number, told apart by their text
        assert_eq!(
            lexical(&sorted(vec![
                typed("9E0", "double"),
                typed("9.0", "decimal"),
                typed("9", "integer"),
            ])),
            vec!["9", "9.0", "9E0"]
        );
    }

    /// Keys order every pair of terms as `sparql_order` does.
    #[test]
    fn order_keys_agree_with_sparql_order() {
        let terms = vec![
            Term::blank("b1"),
            Term::blank("a0"),
            Term::iri("http://example.org/b"),
            Term::iri("http://example.org/a"),
            typed("10", "integer"),
            typed("9", "integer"),
            typed("09", "integer"),
            typed("2.50", "decimal"),
            typed("2.5", "decimal"),
            typed("1e1", "double"),
            typed("NaN", "double"),
            typed("abc", "integer"),
            typed("2026-01-15", "date"),
            typed("2026-01-15T10:30:00Z", "dateTime"),
            Term::literal("Alix"),
            Term::lang_literal("Alix", "en"),
            typed("x", "http://example.org/dt"),
            typed("INF", "double"),
            typed("-INF", "double"),
            typed("+5", "integer"),
            typed(".5", "decimal"),
            typed("1.5e0", "decimal"),
            typed("123456789012345678901234567890", "integer"),
            typed("1234567890123456789012345678901234567890", "integer"),
            typed("true", "boolean"),
            typed("2026-01-15T10:30:00", "dateTime"),
            typed("10:30:00", "time"),
            typed("5", "int"),
        ];
        let parsed: Vec<SparqlOrderKey> = terms
            .iter()
            .map(|term| SparqlOrderKey::new(std::sync::Arc::new(term.clone())))
            .collect();
        let native: Vec<SparqlOrderKey> = terms
            .iter()
            .map(|term| {
                let value = term.as_literal().and_then(Literal::native_value);
                SparqlOrderKey::with_native(std::sync::Arc::new(term.clone()), value.as_ref())
            })
            .collect();
        let inline: Vec<SparqlOrderKey> = terms
            .iter()
            .map(|term| {
                let id = super::super::TermId::inline(term);
                id.and_then(SparqlOrderKey::of_inline)
                    .unwrap_or_else(|| SparqlOrderKey::new(std::sync::Arc::new(term.clone())))
            })
            .collect();
        assert!(
            terms
                .iter()
                .any(|term| super::super::TermId::inline(term).is_some()),
            "some keys are of inlined ids"
        );
        for keys in [&parsed, &native, &inline] {
            for (a, key_a) in terms.iter().zip(keys) {
                for (b, key_b) in terms.iter().zip(keys) {
                    assert_eq!(key_a.cmp(key_b), sparql_order(a, b), "{a} vs {b}");
                }
            }
        }
    }
}
