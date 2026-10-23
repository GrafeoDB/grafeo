//! The id of an RDF term in a dataset: 64 bits, a 4-bit tag and a 60-bit
//! payload.
//!
//! A term has exactly one id, so two ids are equal exactly when their terms
//! are the same term. Small literals are inlined in the id: an
//! `xsd:integer`, an `xsd:decimal`, an `xsd:boolean`, an `xsd:date`, an
//! `xsd:dateTime` and a short simple literal, each only in the canonical
//! form of its value (so the id gives the term back exactly: `"01"^^xsd:integer`
//! is not inlined, `"1"^^xsd:integer` is). Every other term has an index in
//! the dataset's dictionary, or, when only one query made it, in that query's
//! own list.
//!
//! | Tag | Term | Payload |
//! | --- | --- | --- |
//! | 0 | a dictionary term | its index |
//! | 1 | `xsd:integer` within ±2^59 | the value, two's complement |
//! | 2 | `xsd:decimal` with a mantissa within ±2^51 and a scale of at most 15 | mantissa (52 bits) and scale (4 bits) |
//! | 3 | `xsd:boolean` | 0 or 1 |
//! | 4 | `xsd:date` of a year from 1 to 9999, without an offset | days from 1970-01-01 |
//! | 5 | `xsd:dateTime` of a year from 1 to 9999, without an offset or with `Z`, in whole microseconds | a `Z` bit and microseconds from 1970-01-01T00:00:00 |
//! | 6 | a simple literal of at most 7 UTF-8 bytes | the length (3 bits) and the bytes |
//! | 15 | a term one query made | its index in the query's list |
//!
//! The tags are part of the file format (the RDF section stores ids): a tag
//! never changes meaning.

use std::fmt::Write as _;
use std::sync::{Arc, OnceLock};

use super::rdf::{Literal, Term};
use super::{Date, Decimal, Timestamp, Value, ZonedDatetime};

const TAG_SHIFT: u32 = 60;
const PAYLOAD_MASK: u64 = (1 << TAG_SHIFT) - 1;
const XSD: &str = "http://www.w3.org/2001/XMLSchema#";

/// What an id holds, by its tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TermIdKind {
    /// A term of the dataset's dictionary (tag 0).
    Dictionary,
    /// An inlined `xsd:integer` (tag 1).
    Integer,
    /// An inlined `xsd:decimal` (tag 2).
    Decimal,
    /// An inlined `xsd:boolean` (tag 3).
    Boolean,
    /// An inlined `xsd:date` (tag 4).
    Date,
    /// An inlined `xsd:dateTime` (tag 5).
    DateTime,
    /// An inlined short simple literal (tag 6).
    ShortString,
    /// A term only one query made (tag 15).
    Query,
}

impl TermIdKind {
    const fn tag(self) -> u64 {
        match self {
            Self::Dictionary => 0,
            Self::Integer => 1,
            Self::Decimal => 2,
            Self::Boolean => 3,
            Self::Date => 4,
            Self::DateTime => 5,
            Self::ShortString => 6,
            Self::Query => 15,
        }
    }

    const fn of_tag(tag: u64) -> Option<Self> {
        Some(match tag {
            0 => Self::Dictionary,
            1 => Self::Integer,
            2 => Self::Decimal,
            3 => Self::Boolean,
            4 => Self::Date,
            5 => Self::DateTime,
            6 => Self::ShortString,
            15 => Self::Query,
            _ => return None,
        })
    }
}

/// The id of an RDF term (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TermId(u64);

impl TermId {
    /// The largest index of a dictionary or query term.
    pub const MAX_INDEX: u64 = PAYLOAD_MASK;

    const fn with(kind: TermIdKind, payload: u64) -> Self {
        Self((kind.tag() << TAG_SHIFT) | (payload & PAYLOAD_MASK))
    }

    /// The id of the dictionary term at `index`; `None` past
    /// [`MAX_INDEX`](Self::MAX_INDEX).
    #[must_use]
    pub const fn dictionary(index: u64) -> Option<Self> {
        if index > PAYLOAD_MASK {
            None
        } else {
            Some(Self::with(TermIdKind::Dictionary, index))
        }
    }

    /// The id of the query's own term at `index`; `None` past
    /// [`MAX_INDEX`](Self::MAX_INDEX).
    #[must_use]
    pub const fn query(index: u64) -> Option<Self> {
        if index > PAYLOAD_MASK {
            None
        } else {
            Some(Self::with(TermIdKind::Query, index))
        }
    }

    /// The id `raw` holds, as [`raw`](Self::raw) gives it; `None` for a tag
    /// that has no meaning (a damaged file).
    #[must_use]
    pub const fn from_raw(raw: u64) -> Option<Self> {
        match TermIdKind::of_tag(raw >> TAG_SHIFT) {
            Some(_) => Some(Self(raw)),
            None => None,
        }
    }

    /// The 64 bits of the id.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }

    /// What the id holds.
    ///
    /// # Panics
    ///
    /// Never: every way to make an id ([`from_raw`](Self::from_raw) and the
    /// constructors) refuses a tag no kind has.
    #[must_use]
    pub fn kind(self) -> TermIdKind {
        TermIdKind::of_tag(self.0 >> TAG_SHIFT).expect("an id is built with a known tag")
    }

    const fn payload(self) -> u64 {
        self.0 & PAYLOAD_MASK
    }

    /// The payload as a signed number of 60 bits.
    #[allow(
        clippy::cast_possible_wrap,
        reason = "the payload has 60 bits, so the shift by 4 and back sign-extends it exactly"
    )]
    const fn signed_payload(self) -> i64 {
        ((self.payload() << 4) as i64) >> 4
    }

    /// The index of a dictionary or query term; `None` for an inlined term.
    #[must_use]
    pub fn index(self) -> Option<u64> {
        matches!(self.kind(), TermIdKind::Dictionary | TermIdKind::Query).then(|| self.payload())
    }

    /// The id that holds `term` in itself, when the term is a literal this
    /// scheme inlines in its canonical form; `None` for any other term,
    /// which a dictionary gives its id.
    #[must_use]
    pub fn inline(term: &Term) -> Option<Self> {
        let Term::Literal(literal) = term else {
            return None;
        };
        if literal.language().is_some() {
            return None;
        }
        let text = literal.value();
        let datatype = literal.datatype();
        if datatype == Literal::XSD_STRING {
            return short_string(text);
        }
        match datatype.strip_prefix(XSD)? {
            "integer" => inline_integer(text),
            "decimal" => inline_decimal(text),
            "boolean" => match text {
                "false" => Some(Self::with(TermIdKind::Boolean, 0)),
                "true" => Some(Self::with(TermIdKind::Boolean, 1)),
                _ => None,
            },
            "date" => inline_date(text),
            "dateTime" => inline_datetime(text),
            _ => None,
        }
    }

    /// The term an inlined id holds; `None` for a dictionary or query id.
    #[must_use]
    pub fn inline_term(self) -> Option<Term> {
        let (text, datatype) = match self.kind() {
            TermIdKind::Dictionary | TermIdKind::Query => return None,
            TermIdKind::ShortString => {
                let mut bytes = [0u8; 8];
                return Some(Term::literal(Arc::<str>::from(self.short_str(&mut bytes)?)));
            }
            TermIdKind::Integer => (shared_text(self.signed_payload()), 0),
            TermIdKind::Decimal => (shared_text(self.decimal()?), 1),
            TermIdKind::Boolean => (shared_text(self.payload() == 1), 2),
            TermIdKind::Date => (shared_text(Date::from_days(self.days())), 3),
            TermIdKind::DateTime => {
                let (micros, zulu) = self.datetime();
                (Arc::from(datetime_text(micros, zulu)), 4)
            }
        };
        Some(Term::typed_literal(text, shared_datatype(datatype)))
    }

    /// The value an expression computes with for an inlined id, as
    /// [`Literal::native_value`] gives it for the term; `None` for a
    /// dictionary or query id.
    #[must_use]
    pub fn inline_value(self) -> Option<Value> {
        Some(match self.kind() {
            TermIdKind::Dictionary | TermIdKind::Query => return None,
            TermIdKind::Integer => Value::Int64(self.signed_payload()),
            TermIdKind::Decimal => Value::Decimal(self.decimal()?),
            TermIdKind::Boolean => Value::Bool(self.payload() == 1),
            TermIdKind::Date => Value::Date(Date::from_days(self.days())),
            TermIdKind::DateTime => {
                let (micros, zulu) = self.datetime();
                let local = Timestamp::from_micros(micros);
                if zulu {
                    Value::ZonedDatetime(ZonedDatetime::from_timestamp_offset(local, 0))
                } else {
                    Value::Timestamp(local)
                }
            }
            TermIdKind::ShortString => {
                let mut bytes = [0u8; 8];
                Value::from(self.short_str(&mut bytes)?)
            }
        })
    }

    fn decimal(self) -> Option<Decimal> {
        let scale = u8::try_from(self.payload() & 0xF).ok()?;
        let mantissa = self.signed_payload() >> 4;
        Decimal::new(i128::from(mantissa), scale).ok()
    }

    #[allow(
        clippy::cast_possible_truncation,
        reason = "a date id is built from an i32 day count, so its payload fits"
    )]
    const fn days(self) -> i32 {
        self.signed_payload() as i32
    }

    /// Microseconds of the wall time, and whether the text ends in `Z`.
    const fn datetime(self) -> (i64, bool) {
        (self.signed_payload() >> 1, self.payload() & 1 == 1)
    }

    /// The text of an inlined short simple literal, written to `bytes` (a
    /// string test reads it without making the term); `None` for any other
    /// id.
    #[must_use]
    pub fn short_string_text(self, bytes: &mut [u8; 8]) -> Option<&str> {
        if self.kind() != TermIdKind::ShortString {
            return None;
        }
        self.short_str(bytes)
    }

    /// The text of a short string id, in `bytes`.
    fn short_str(self, bytes: &mut [u8; 8]) -> Option<&str> {
        let payload = self.payload();
        let len = usize::try_from(payload >> 56).ok()?;
        *bytes = payload.to_le_bytes();
        std::str::from_utf8(bytes.get(..len)?).ok()
    }
}

/// `value`'s text in one allocation: written to the stack, then copied into
/// the `Arc` (an inlined literal's text is short).
fn shared_text(value: impl std::fmt::Display) -> Arc<str> {
    struct Stack {
        bytes: [u8; 64],
        len: usize,
    }
    impl std::fmt::Write for Stack {
        fn write_str(&mut self, text: &str) -> std::fmt::Result {
            let end = self.len + text.len();
            let target = self.bytes.get_mut(self.len..end).ok_or(std::fmt::Error)?;
            target.copy_from_slice(text.as_bytes());
            self.len = end;
            Ok(())
        }
    }
    let mut stack = Stack {
        bytes: [0; 64],
        len: 0,
    };
    match write!(stack, "{value}") {
        Ok(()) => Arc::from(std::str::from_utf8(&stack.bytes[..stack.len]).unwrap_or_default()),
        Err(_) => Arc::from(value.to_string()),
    }
}

/// The local names of the datatypes of the inlined literals.
const DATATYPE_NAMES: [&str; 5] = ["integer", "decimal", "boolean", "date", "dateTime"];

/// Their IRIs, made once and shared by every term an id gives back.
static DATATYPES: [OnceLock<Arc<str>>; 5] = [const { OnceLock::new() }; 5];

/// The shared IRI of datatype `index` of [`DATATYPE_NAMES`].
fn shared_datatype(index: usize) -> Arc<str> {
    Arc::clone(
        DATATYPES[index].get_or_init(|| Arc::from(format!("{XSD}{}", DATATYPE_NAMES[index]))),
    )
}

const INLINE_MIN: i64 = -(1 << 59);
const INLINE_MAX: i64 = (1 << 59) - 1;

#[allow(
    clippy::cast_sign_loss,
    reason = "the payload is the value's two's complement, masked to 60 bits"
)]
fn signed(kind: TermIdKind, value: i64) -> Option<TermId> {
    (INLINE_MIN..=INLINE_MAX)
        .contains(&value)
        .then(|| TermId::with(kind, value as u64))
}

fn inline_integer(text: &str) -> Option<TermId> {
    let value: i64 = text.parse().ok()?;
    if value.to_string() != text {
        return None;
    }
    signed(TermIdKind::Integer, value)
}

fn inline_decimal(text: &str) -> Option<TermId> {
    let decimal = Decimal::parse(text).ok()?;
    if decimal.to_string() != text || decimal.scale() > 15 {
        return None;
    }
    let mantissa = i64::try_from(decimal.mantissa()).ok()?;
    if !(-(1 << 51)..(1 << 51)).contains(&mantissa) {
        return None;
    }
    signed(
        TermIdKind::Decimal,
        (mantissa << 4) | i64::from(decimal.scale()),
    )
}

fn inline_date(text: &str) -> Option<TermId> {
    let date = Date::parse(text)?;
    if !(1..=9999).contains(&date.year()) || date.to_string() != text {
        return None;
    }
    signed(TermIdKind::Date, i64::from(date.as_days()))
}

fn inline_datetime(text: &str) -> Option<TermId> {
    let (date, rest) = text.split_once('T')?;
    let date = Date::parse(date)?;
    if !(1..=9999).contains(&date.year()) {
        return None;
    }
    let (clock, zulu) = match rest.strip_suffix('Z') {
        Some(clock) => (clock, true),
        None => (rest, false),
    };
    let mut parts = clock.splitn(3, ':');
    let hours: i64 = parts.next()?.parse().ok()?;
    let minutes: i64 = parts.next()?.parse().ok()?;
    let seconds = parts.next()?;
    let (seconds, fraction) = seconds.split_once('.').unwrap_or((seconds, ""));
    let seconds: i64 = seconds.parse().ok()?;
    if hours > 23 || minutes > 59 || seconds > 59 || fraction.len() > 6 {
        return None;
    }
    let micros_of_second: i64 = if fraction.is_empty() {
        0
    } else {
        format!("{fraction:0<6}").parse().ok()?
    };
    let micros = i64::from(date.as_days()) * 86_400_000_000
        + ((hours * 60 + minutes) * 60 + seconds) * 1_000_000
        + micros_of_second;
    if datetime_text(micros, zulu) != text {
        return None;
    }
    signed(TermIdKind::DateTime, (micros << 1) | i64::from(zulu))
}

/// The canonical text of a dateTime: `YYYY-MM-DDTHH:MM:SS`, a fraction
/// without trailing zeros when it is not zero, and `Z` when `zulu`.
fn datetime_text(micros: i64, zulu: bool) -> String {
    let days = micros.div_euclid(86_400_000_000);
    let of_day = micros.rem_euclid(86_400_000_000);
    let seconds = of_day / 1_000_000;
    let fraction = of_day % 1_000_000;
    let date = Date::from_days(i32::try_from(days).unwrap_or(i32::MAX));
    let mut text = format!(
        "{date}T{:02}:{:02}:{:02}",
        seconds / 3600,
        (seconds / 60) % 60,
        seconds % 60
    );
    if fraction != 0 {
        let digits = format!("{fraction:06}");
        let _ = write!(text, ".{}", digits.trim_end_matches('0'));
    }
    if zulu {
        text.push('Z');
    }
    text
}

fn short_string(text: &str) -> Option<TermId> {
    let bytes = text.as_bytes();
    if bytes.len() > 7 {
        return None;
    }
    let mut payload = [0u8; 8];
    payload[..bytes.len()].copy_from_slice(bytes);
    let len = u64::try_from(bytes.len()).ok()?;
    Some(TermId::with(
        TermIdKind::ShortString,
        u64::from_le_bytes(payload) | (len << 56),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn typed(text: &str, datatype: &str) -> Term {
        Term::typed_literal(text, format!("{XSD}{datatype}"))
    }

    /// Every literal the scheme inlines comes back as the same term, with
    /// the value `Literal::native_value` gives it.
    #[test]
    fn an_inlined_term_round_trips_exactly() {
        let terms = [
            typed("0", "integer"),
            typed("-42", "integer"),
            typed(&INLINE_MAX.to_string(), "integer"),
            typed(&INLINE_MIN.to_string(), "integer"),
            typed("1.50", "decimal"),
            typed("-0.5", "decimal"),
            typed("12", "decimal"),
            typed("0.000000000000001", "decimal"),
            typed("true", "boolean"),
            typed("false", "boolean"),
            typed("2026-01-15", "date"),
            typed("0001-01-01", "date"),
            typed("9999-12-31", "date"),
            typed("2026-01-15T10:30:00", "dateTime"),
            typed("2026-01-15T10:30:00Z", "dateTime"),
            typed("1969-12-31T23:59:59.5Z", "dateTime"),
            typed("2026-01-15T10:30:00.000001", "dateTime"),
            Term::literal(""),
            Term::literal("Alix"),
            Term::literal("Kraków"),
            Term::literal("1234567"),
        ];
        for term in terms {
            let id = TermId::inline(&term).unwrap_or_else(|| panic!("{term} is inlined"));
            assert_eq!(id.inline_term().as_ref(), Some(&term), "{term}");
            assert_eq!(TermId::from_raw(id.raw()), Some(id));
            assert_eq!(id.index(), None);
            let literal = term.as_literal().expect("a literal");
            assert_eq!(id.inline_value(), literal.native_value(), "{term}");
        }
    }

    /// A form other than the canonical one, a value past the payload, and
    /// every other kind of term are left to the dictionary.
    #[test]
    fn other_forms_and_terms_are_not_inlined() {
        let terms = [
            typed("01", "integer"),
            typed("+1", "integer"),
            typed(" 1", "integer"),
            typed(&(INLINE_MAX + 1).to_string(), "integer"),
            typed("1.5e0", "decimal"),
            typed(".5", "decimal"),
            typed("+1.5", "decimal"),
            typed("1.0000000000000000", "decimal"),
            typed("1", "boolean"),
            typed("TRUE", "boolean"),
            typed("2026-1-15", "date"),
            typed("10000-01-01", "date"),
            typed("-0001-01-01", "date"),
            typed("2026-01-15Z", "date"),
            typed("2026-01-15T10:30:00+01:00", "dateTime"),
            typed("2026-01-15T10:30:00.50", "dateTime"),
            typed("2026-01-15T10:30:00.0000001", "dateTime"),
            typed("2026-01-15T24:00:00", "dateTime"),
            typed("2026-01-15T10:30", "dateTime"),
            typed("1.5", "double"),
            typed("1", "int"),
            typed("x", "http://example.org/dt"),
            Term::literal("Eight by"),
            Term::lang_literal("Fruit", "en"),
            Term::iri("http://example.org/a"),
            Term::blank("b0"),
        ];
        for term in terms {
            assert_eq!(TermId::inline(&term), None, "{term}");
        }
    }

    /// Indexes of dictionary and query terms, and the tags no id has.
    #[test]
    fn indexes_and_tags() {
        let id = TermId::dictionary(7).unwrap();
        assert_eq!((id.kind(), id.index()), (TermIdKind::Dictionary, Some(7)));
        assert_eq!(id.inline_term(), None);
        assert_eq!(id.inline_value(), None);
        let query = TermId::query(TermId::MAX_INDEX).unwrap();
        assert_eq!(query.index(), Some(TermId::MAX_INDEX));
        assert_ne!(query, TermId::dictionary(TermId::MAX_INDEX).unwrap());
        assert_eq!(TermId::dictionary(TermId::MAX_INDEX + 1), None);
        for tag in 7..15u64 {
            assert_eq!(TermId::from_raw(tag << 60), None, "tag {tag}");
        }
    }

    /// Ids are the file's: a tag and its payload never change meaning.
    #[test]
    fn ids_are_pinned() {
        let pinned = [
            (typed("1", "integer"), 0x1000_0000_0000_0001_u64),
            (typed("-1", "integer"), 0x1FFF_FFFF_FFFF_FFFF),
            (typed("1.50", "decimal"), 0x2000_0000_0000_0962),
            (typed("true", "boolean"), 0x3000_0000_0000_0001),
            (typed("1970-01-02", "date"), 0x4000_0000_0000_0001),
            (
                typed("1970-01-01T00:00:01Z", "dateTime"),
                0x5000_0000_001E_8481,
            ),
            (Term::literal("ab"), 0x6200_0000_0000_6261),
        ];
        for (term, raw) in pinned {
            assert_eq!(TermId::inline(&term).map(TermId::raw), Some(raw), "{term}");
        }
    }
}
