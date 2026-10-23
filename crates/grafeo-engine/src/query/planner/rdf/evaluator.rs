//! SPARQL expressions on RDF terms (SPARQL 1.1 section 17).
//!
//! The evaluator reads a variable as the term its id column holds and
//! computes with [`Value`]s (see [`super::terms`]). An error, such as a type
//! error or an unbound variable, is `None`: SPARQL turns it into an unbound
//! value, and a FILTER drops the row.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::Arc;

use grafeo_common::types::rdf::{Literal, Term};
use grafeo_common::types::{Decimal, Value};
use grafeo_core::execution::DataChunk;
use grafeo_core::execution::operators::{
    BinaryFilterOp, FilterExpression, OperatorError, Predicate, StatementClock, UnaryFilterOp,
};

use super::term_ids::{QueryTermIds, TermEntry, id_of};
use super::terms::{
    self, Operand, StringArg, compatible_languages, effective_boolean, native, numbers_with_a_nan,
    sparql_compare, sparql_equal, str_of,
};

#[cfg(feature = "regex")]
use regex::Regex;
#[cfg(all(feature = "regex-lite", not(feature = "regex")))]
use regex_lite::Regex;

const XSD: &str = "http://www.w3.org/2001/XMLSchema#";

/// Evaluates one SPARQL expression over the rows of a chunk whose variable
/// columns hold term ids.
pub(super) struct RdfExpressionPredicate {
    expression: FilterExpression,
    variable_columns: HashMap<String, usize>,
    terms: Arc<QueryTermIds>,
    /// The instant the query started, which `NOW()` gives.
    clock: StatementClock,
}

/// An operand of a comparison: a variable's dictionary entry (whose value is
/// parsed once while the dataset is open), or a computed value (an inlined
/// literal's too, which is the value of its canonical form).
enum Arg<'a> {
    Entry(TermEntry<'a>),
    Value(Value),
}

impl Arg<'_> {
    fn operand(&self) -> Option<Operand<'_>> {
        match self {
            Self::Entry(entry) => Some(Operand::of_term(entry.term(), entry.native())),
            Self::Value(value) => Operand::of(value),
        }
    }
}

/// A string argument read where it is: a stored literal's text and tag, an
/// inlined short string's bytes, a constant of the query, or else a computed
/// string. The string functions that give a boolean or a number read their
/// arguments this way, so a row copies no text.
enum StringView<'a> {
    Borrowed {
        text: &'a str,
        language: Option<&'a str>,
    },
    Short {
        bytes: [u8; 8],
        len: usize,
    },
    Computed(StringArg),
}

impl StringView<'_> {
    fn text(&self) -> &str {
        match self {
            Self::Borrowed { text, .. } => text,
            // UTF-8, checked when the view was made.
            Self::Short { bytes, len } => {
                std::str::from_utf8(bytes.get(..*len).unwrap_or_default()).unwrap_or_default()
            }
            Self::Computed(arg) => &arg.text,
        }
    }

    fn language(&self) -> Option<&str> {
        match self {
            Self::Borrowed { language, .. } => *language,
            Self::Short { .. } => None,
            Self::Computed(arg) => arg.language.as_deref(),
        }
    }

    /// As [`StringArg::compatible_with`].
    fn compatible_with(&self, other: &StringView<'_>) -> bool {
        compatible_languages(self.language(), other.language())
    }

    /// The view of a stored term: a simple, `xsd:string` or tagged literal;
    /// `None` (a type error) for any other term.
    fn of_term(term: &Term) -> Option<StringView<'_>> {
        let Term::Literal(literal) = term else {
            return None;
        };
        match literal.language() {
            Some(tag) => Some(StringView::Borrowed {
                text: literal.value(),
                language: Some(tag),
            }),
            None if literal.datatype() == Literal::XSD_STRING => Some(StringView::Borrowed {
                text: literal.value(),
                language: None,
            }),
            None => None,
        }
    }
}

impl RdfExpressionPredicate {
    pub(super) fn new(
        expression: FilterExpression,
        variable_columns: HashMap<String, usize>,
        terms: Arc<QueryTermIds>,
        clock: StatementClock,
    ) -> Self {
        Self {
            expression: fold_compared_constants(expression),
            variable_columns,
            terms,
            clock,
        }
    }

    /// The expression's value at `row`; `None` for an error or an unbound
    /// result.
    pub(super) fn eval(&self, chunk: &DataChunk, row: usize) -> Option<Value> {
        self.eval_expr(&self.expression, chunk, row)
    }

    /// The dictionary entry of a variable at `row`; `None` when it is
    /// unbound.
    fn entry(&self, name: &str, chunk: &DataChunk, row: usize) -> Option<TermEntry<'_>> {
        let column = chunk.column(*self.variable_columns.get(name)?)?;
        self.terms.entry_of(&column.get_value(row)?)
    }

    /// A variable as an operand: an inlined literal as its value, without
    /// building its term; any other term as its entry.
    fn variable_arg(&self, name: &str, chunk: &DataChunk, row: usize) -> Option<Arg<'_>> {
        let column = chunk.column(*self.variable_columns.get(name)?)?;
        let value = column.get_value(row)?;
        match id_of(&value)?.inline_value() {
            Some(native) => Some(Arg::Value(native)),
            None => self.terms.entry_of(&value).map(Arg::Entry),
        }
    }

    fn variable(&self, name: &str, chunk: &DataChunk, row: usize) -> Option<Value> {
        self.entry(name, chunk, row)
            .map(|entry| Value::RdfTerm(Arc::clone(entry.term())))
    }

    fn arg(&self, expr: &FilterExpression, chunk: &DataChunk, row: usize) -> Option<Arg<'_>> {
        match expr {
            FilterExpression::Variable(name) => self.variable_arg(name, chunk, row),
            other => self.eval_expr(other, chunk, row).map(Arg::Value),
        }
    }

    fn eval_expr(&self, expr: &FilterExpression, chunk: &DataChunk, row: usize) -> Option<Value> {
        match expr {
            FilterExpression::Literal(Value::Null) => None,
            FilterExpression::Literal(value) => Some(value.clone()),
            FilterExpression::Variable(name)
            | FilterExpression::Property { variable: name, .. }
            | FilterExpression::Id(name)
            | FilterExpression::Labels(name)
            | FilterExpression::Type(name) => self.variable(name, chunk, row),
            FilterExpression::Binary { left, op, right } => {
                self.eval_binary(left, *op, right, chunk, row)
            }
            FilterExpression::Unary { op, operand } => self.eval_unary(*op, operand, chunk, row),
            FilterExpression::SparqlCall { name, args } => {
                self.eval_function_call(name, args, chunk, row)
            }
            FilterExpression::List(items) => Some(Value::List(
                items
                    .iter()
                    .map(|item| self.eval_expr(item, chunk, row).unwrap_or(Value::Null))
                    .collect::<Vec<_>>()
                    .into(),
            )),
            _ => None,
        }
    }

    fn ebv(&self, expr: &FilterExpression, chunk: &DataChunk, row: usize) -> Option<bool> {
        effective_boolean(&self.eval_expr(expr, chunk, row)?)
    }

    fn eval_binary(
        &self,
        left: &FilterExpression,
        op: BinaryFilterOp,
        right: &FilterExpression,
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Value> {
        match op {
            // `||` and `&&` with errors (17.2): an error is false's or true's
            // neighbour, decided by the other side when it can be.
            BinaryFilterOp::And => {
                let (a, b) = (self.ebv(left, chunk, row), self.ebv(right, chunk, row));
                match (a, b) {
                    (Some(false), _) | (_, Some(false)) => Some(Value::Bool(false)),
                    (Some(true), Some(true)) => Some(Value::Bool(true)),
                    _ => None,
                }
            }
            BinaryFilterOp::Or => {
                let (a, b) = (self.ebv(left, chunk, row), self.ebv(right, chunk, row));
                match (a, b) {
                    (Some(true), _) | (_, Some(true)) => Some(Value::Bool(true)),
                    (Some(false), Some(false)) => Some(Value::Bool(false)),
                    _ => None,
                }
            }
            BinaryFilterOp::Eq | BinaryFilterOp::Ne => {
                let (a, b) = (self.arg(left, chunk, row)?, self.arg(right, chunk, row)?);
                let equal = sparql_equal(&a.operand()?, &b.operand()?)?;
                Some(Value::Bool(equal == (op == BinaryFilterOp::Eq)))
            }
            BinaryFilterOp::Lt | BinaryFilterOp::Le | BinaryFilterOp::Gt | BinaryFilterOp::Ge => {
                let (a, b) = (self.arg(left, chunk, row)?, self.arg(right, chunk, row)?);
                let (a, b) = (a.operand()?, b.operand()?);
                // A NaN is neither less nor greater than a number (XPath).
                if numbers_with_a_nan(&a, &b) {
                    return Some(Value::Bool(false));
                }
                let order = sparql_compare(&a, &b)?;
                Some(Value::Bool(match op {
                    BinaryFilterOp::Lt => order.is_lt(),
                    BinaryFilterOp::Le => order.is_le(),
                    BinaryFilterOp::Gt => order.is_gt(),
                    _ => order.is_ge(),
                }))
            }
            // `IN` is `=` with each member, or'ed (17.4.1.9): true on a match,
            // otherwise an error if a comparison erred, otherwise false.
            BinaryFilterOp::In => {
                let needle = self.arg(left, chunk, row)?;
                let needle = needle.operand()?;
                let Value::List(items) = self.eval_expr(right, chunk, row)? else {
                    return None;
                };
                let mut erred = false;
                for item in items.iter() {
                    match Operand::of(item).and_then(|item| sparql_equal(&needle, &item)) {
                        Some(true) => return Some(Value::Bool(true)),
                        Some(false) => {}
                        None => erred = true,
                    }
                }
                if erred {
                    None
                } else {
                    Some(Value::Bool(false))
                }
            }
            BinaryFilterOp::Add
            | BinaryFilterOp::Sub
            | BinaryFilterOp::Mul
            | BinaryFilterOp::Div => {
                let a = self.eval_expr(left, chunk, row)?;
                let b = self.eval_expr(right, chunk, row)?;
                numeric_op(native(&a)?.as_ref(), op, native(&b)?.as_ref())
            }
            _ => None,
        }
    }

    fn eval_unary(
        &self,
        op: UnaryFilterOp,
        operand: &FilterExpression,
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Value> {
        match op {
            UnaryFilterOp::Not => Some(Value::Bool(!self.ebv(operand, chunk, row)?)),
            UnaryFilterOp::IsNull => {
                Some(Value::Bool(self.eval_expr(operand, chunk, row).is_none()))
            }
            UnaryFilterOp::IsNotNull => {
                Some(Value::Bool(self.eval_expr(operand, chunk, row).is_some()))
            }
            UnaryFilterOp::Neg => match native(&self.eval_expr(operand, chunk, row)?)?.as_ref() {
                // -i64::MIN is past the i64 range: a decimal, as XPath integers have no limit.
                Value::Int64(v) => Some(
                    v.checked_neg()
                        .map_or_else(|| Value::Decimal(Decimal::from_i64(*v).neg()), Value::Int64),
                ),
                Value::Float64(v) => Some(Value::Float64(-v)),
                Value::Decimal(v) => Some(Value::Decimal(v.neg())),
                _ => None,
            },
            UnaryFilterOp::Pos => match native(&self.eval_expr(operand, chunk, row)?)?.into_owned()
            {
                number @ (Value::Int64(_) | Value::Float64(_) | Value::Decimal(_)) => Some(number),
                _ => None,
            },
            _ => None,
        }
    }

    fn string_arg(
        &self,
        expr: Option<&FilterExpression>,
        chunk: &DataChunk,
        row: usize,
    ) -> Option<StringArg> {
        StringArg::of(&self.eval_expr(expr?, chunk, row)?)
    }

    /// A string argument read in place (see [`StringView`]).
    fn string_view<'s>(
        &'s self,
        expr: Option<&'s FilterExpression>,
        chunk: &DataChunk,
        row: usize,
    ) -> Option<StringView<'s>> {
        match expr? {
            FilterExpression::Variable(name) => {
                let column = chunk.column(*self.variable_columns.get(name)?)?;
                let id = id_of(&column.get_value(row)?)?;
                let mut bytes = [0u8; 8];
                if let Some(text) = id.short_string_text(&mut bytes) {
                    let len = text.len();
                    return Some(StringView::Short { bytes, len });
                }
                match self.terms.entry(id)? {
                    TermEntry::Stored(entry) => StringView::of_term(entry.term()),
                    // A number, a boolean, a date or a dateTime.
                    TermEntry::Inline { .. } => None,
                }
            }
            FilterExpression::Literal(Value::String(text)) => Some(StringView::Borrowed {
                text: text.as_str(),
                language: None,
            }),
            other => self
                .string_arg(Some(other), chunk, row)
                .map(StringView::Computed),
        }
    }

    /// A simple literal or `xsd:string` argument (a pattern, a flag, a tag).
    fn simple_string(
        &self,
        expr: Option<&FilterExpression>,
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Arc<str>> {
        let arg = self.string_arg(expr, chunk, row)?;
        arg.language.is_none().then_some(arg.text)
    }

    fn number(
        &self,
        expr: Option<&FilterExpression>,
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Value> {
        let value = self.eval_expr(expr?, chunk, row)?;
        match native(&value)?.into_owned() {
            number @ (Value::Int64(_) | Value::Float64(_) | Value::Decimal(_)) => Some(number),
            _ => None,
        }
    }

    /// The term an argument evaluates to (a computed value as its literal).
    fn term(&self, expr: Option<&FilterExpression>, chunk: &DataChunk, row: usize) -> Option<Term> {
        terms::to_term(&self.eval_expr(expr?, chunk, row)?)
    }

    /// Evaluates a SPARQL function call (17.4).
    #[allow(
        clippy::too_many_lines,
        reason = "one arm per SPARQL function, as section 17.4 lists them"
    )]
    fn eval_function_call(
        &self,
        name: &str,
        args: &[FilterExpression],
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Value> {
        let func_name = name.to_uppercase();
        let arg = |i: usize| args.get(i);
        match func_name.as_str() {
            // ---- Functional forms (17.4.1) ----
            "BOUND" => match arg(0)? {
                FilterExpression::Variable(name) => {
                    Some(Value::Bool(self.entry(name, chunk, row).is_some()))
                }
                other => Some(Value::Bool(self.eval_expr(other, chunk, row).is_some())),
            },
            "IF" => {
                if self.ebv(arg(0)?, chunk, row)? {
                    self.eval_expr(arg(1)?, chunk, row)
                } else {
                    self.eval_expr(arg(2)?, chunk, row)
                }
            }
            "COALESCE" => args.iter().find_map(|a| self.eval_expr(a, chunk, row)),
            "SAMETERM" => {
                let (a, b) = (
                    self.term(arg(0), chunk, row)?,
                    self.term(arg(1), chunk, row)?,
                );
                Some(Value::Bool(a == b))
            }

            // ---- Functions on terms (17.4.2) ----
            "ISIRI" | "ISURI" => {
                let term = self.term(arg(0), chunk, row)?;
                Some(Value::Bool(matches!(term, Term::Iri(_))))
            }
            "ISBLANK" => {
                let term = self.term(arg(0), chunk, row)?;
                Some(Value::Bool(matches!(term, Term::BlankNode(_))))
            }
            "ISLITERAL" => {
                let term = self.term(arg(0), chunk, row)?;
                Some(Value::Bool(matches!(term, Term::Literal(_))))
            }
            "ISNUMERIC" => {
                let value = self.eval_expr(arg(0)?, chunk, row)?;
                Some(Value::Bool(matches!(
                    native(&value).as_deref(),
                    Some(Value::Int64(_) | Value::Float64(_) | Value::Decimal(_))
                )))
            }
            "STR" => str_of(&self.eval_expr(arg(0)?, chunk, row)?).map(|text| Value::from(&*text)),
            "LANG" => match self.term(arg(0), chunk, row)? {
                Term::Literal(literal) => Some(Value::from(literal.language().unwrap_or(""))),
                _ => None,
            },
            "DATATYPE" => match self.term(arg(0), chunk, row)? {
                Term::Literal(literal) => Some(iri_value(literal.datatype())),
                _ => None,
            },
            "IRI" | "URI" => match self.eval_expr(arg(0)?, chunk, row)? {
                iri @ Value::RdfTerm(_)
                    if matches!(self.term(arg(0), chunk, row)?, Term::Iri(_)) =>
                {
                    Some(iri)
                }
                other => {
                    let text = StringArg::of(&other)?;
                    text.language.is_none().then(|| iri_value(&text.text))
                }
            },
            "BNODE" => {
                use std::sync::atomic::{AtomicU64, Ordering};
                static BNODE_COUNTER: AtomicU64 = AtomicU64::new(0);
                let label = match arg(0) {
                    None => format!(
                        "grafeo-bnode-{}",
                        BNODE_COUNTER.fetch_add(1, Ordering::Relaxed)
                    ),
                    Some(_) => format!("grafeo-bnode-{}", self.simple_string(arg(0), chunk, row)?),
                };
                Some(Value::RdfTerm(Arc::new(Term::blank(label))))
            }
            "STRDT" => {
                let text = self.simple_string(arg(0), chunk, row)?;
                let Term::Iri(datatype) = self.term(arg(1), chunk, row)? else {
                    return None;
                };
                Some(Value::RdfTerm(Arc::new(Term::typed_literal(
                    text,
                    datatype.as_str(),
                ))))
            }
            "STRLANG" => {
                let text = self.simple_string(arg(0), chunk, row)?;
                let tag = self.simple_string(arg(1), chunk, row)?;
                if tag.is_empty() {
                    return None;
                }
                Some(Value::RdfTerm(Arc::new(Term::lang_literal(text, tag))))
            }
            "UUID" => Some(iri_value(&format!("urn:uuid:{}", uuid_v4()))),
            "STRUUID" => Some(Value::from(uuid_v4())),

            // ---- Functions on strings (17.4.3) ----
            "STRLEN" => {
                let text = self.string_view(arg(0), chunk, row)?;
                Some(Value::Int64(
                    i64::try_from(text.text().chars().count()).ok()?,
                ))
            }
            "SUBSTR" | "SUBSTRING" => {
                let source = self.string_arg(arg(0), chunk, row)?;
                let start = as_f64(&self.number(arg(1), chunk, row)?);
                let length = match arg(2) {
                    Some(_) => Some(as_f64(&self.number(arg(2), chunk, row)?)),
                    None => None,
                };
                Some(source.with_text(substring(&source.text, start, length)))
            }
            "UCASE" | "UPPER" => {
                let source = self.string_arg(arg(0), chunk, row)?;
                Some(source.with_text(source.text.to_uppercase()))
            }
            "LCASE" | "LOWER" => {
                let source = self.string_arg(arg(0), chunk, row)?;
                Some(source.with_text(source.text.to_lowercase()))
            }
            "STRSTARTS" | "STRENDS" | "CONTAINS" => {
                let a = self.string_view(arg(0), chunk, row)?;
                let b = self.string_view(arg(1), chunk, row)?;
                if !a.compatible_with(&b) {
                    return None;
                }
                let (a, b) = (a.text(), b.text());
                Some(Value::Bool(match func_name.as_str() {
                    "STRSTARTS" => a.starts_with(b),
                    "STRENDS" => a.ends_with(b),
                    _ => a.contains(b),
                }))
            }
            "STRBEFORE" | "STRAFTER" => {
                let (a, b) = self.compatible_pair(args, chunk, row)?;
                Some(match a.text.find(&*b.text) {
                    Some(pos) if func_name == "STRBEFORE" => a.with_text(&a.text[..pos]),
                    Some(pos) => a.with_text(&a.text[pos + b.text.len()..]),
                    // No match: the empty simple literal (17.4.3.7).
                    None => Value::from(""),
                })
            }
            "ENCODE_FOR_URI" => {
                let text = self.string_arg(arg(0), chunk, row)?.text;
                let mut encoded = String::with_capacity(text.len());
                for byte in text.bytes() {
                    if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
                        encoded.push(char::from(byte));
                    } else {
                        let _ = write!(encoded, "%{byte:02X}");
                    }
                }
                Some(Value::from(encoded))
            }
            "CONCAT" => {
                let parts = args
                    .iter()
                    .map(|a| self.string_arg(Some(a), chunk, row))
                    .collect::<Option<Vec<_>>>()?;
                let text: String = parts.iter().map(|part| &*part.text).collect();
                // The tag all arguments share, or none (17.4.3.12).
                let shared = parts.first().and_then(|first| {
                    let tag = first.language.as_ref()?;
                    parts
                        .iter()
                        .all(|part| {
                            part.language
                                .as_deref()
                                .is_some_and(|t| t.eq_ignore_ascii_case(tag))
                        })
                        .then(|| Arc::clone(tag))
                });
                Some(match shared {
                    Some(tag) => Value::RdfTerm(Arc::new(Term::lang_literal(text, tag))),
                    None => Value::from(text),
                })
            }
            "LANGMATCHES" => {
                let tag = self.simple_string(arg(0), chunk, row)?;
                let range = self.simple_string(arg(1), chunk, row)?;
                if &*range == "*" {
                    return Some(Value::Bool(!tag.is_empty()));
                }
                // RFC 4647 basic filtering: "en-US" matches "en".
                let (tag, range) = (tag.to_lowercase(), range.to_lowercase());
                Some(Value::Bool(
                    tag == range
                        || (tag.starts_with(&range)
                            && tag.as_bytes().get(range.len()) == Some(&b'-')),
                ))
            }
            "REGEX" => {
                let text = self.string_view(arg(0), chunk, row)?;
                let regex = self.regex(arg(1), arg(2), chunk, row)?;
                Some(Value::Bool(regex.is_match(text.text())))
            }
            "REPLACE" => {
                let source = self.string_arg(arg(0), chunk, row)?;
                let regex = self.regex(arg(1), arg(3), chunk, row)?;
                let replacement = self.simple_string(arg(2), chunk, row)?;
                // A pattern that matches the empty string is an error (17.4.3.15).
                if regex.is_match("") {
                    return None;
                }
                Some(source.with_text(regex.replace_all(&source.text, &*replacement).into_owned()))
            }

            // ---- Functions on numerics (17.4.4) ----
            "ABS" => match self.number(arg(0), chunk, row)? {
                Value::Int64(v) => Some(
                    v.checked_abs()
                        .map_or_else(|| Value::Decimal(Decimal::from_i64(v).abs()), Value::Int64),
                ),
                Value::Float64(v) => Some(Value::Float64(v.abs())),
                Value::Decimal(v) => Some(Value::Decimal(v.abs())),
                _ => None,
            },
            "CEIL" => match self.number(arg(0), chunk, row)? {
                Value::Float64(v) => Some(Value::Float64(v.ceil())),
                Value::Decimal(v) => Some(Value::Decimal(v.ceil())),
                integer => Some(integer),
            },
            "FLOOR" => match self.number(arg(0), chunk, row)? {
                Value::Float64(v) => Some(Value::Float64(v.floor())),
                Value::Decimal(v) => Some(Value::Decimal(v.floor())),
                integer => Some(integer),
            },
            // fn:round: the nearest integer, of two equally near ones the one
            // nearer positive infinity (ROUND(-2.5) is -2).
            "ROUND" => match self.number(arg(0), chunk, row)? {
                Value::Float64(v) => Some(Value::Float64(round_half_up(v))),
                Value::Decimal(v) => Some(Value::Decimal(
                    v.checked_add(&Decimal::new(5, 1).ok()?)
                        .map_or_else(|_| v.round(), |half_up| half_up.floor()),
                )),
                integer => Some(integer),
            },
            "RAND" => Some(Value::Float64(random_unit())),

            // ---- Functions on dates and times (17.4.5) ----
            // One instant for the whole query (17.4.5.1).
            "NOW" => Some(Value::Timestamp(self.clock.instant())),
            "YEAR" | "MONTH" | "DAY" => {
                let value = self.eval_expr(arg(0)?, chunk, row)?;
                let date = match native(&value)?.into_owned() {
                    Value::Date(d) => d,
                    Value::Timestamp(ts) => ts.to_date(),
                    Value::ZonedDatetime(zdt) => zdt.to_local_date(),
                    _ => return None,
                };
                Some(Value::Int64(match func_name.as_str() {
                    "YEAR" => i64::from(date.year()),
                    "MONTH" => i64::from(date.month()),
                    _ => i64::from(date.day()),
                }))
            }
            "HOURS" | "MINUTES" | "SECONDS" => {
                let value = self.eval_expr(arg(0)?, chunk, row)?;
                let time = match native(&value)?.into_owned() {
                    Value::Time(t) => t,
                    Value::Timestamp(ts) => ts.to_time(),
                    Value::ZonedDatetime(zdt) => zdt.to_local_time(),
                    _ => return None,
                };
                Some(match func_name.as_str() {
                    "HOURS" => Value::Int64(i64::from(time.hour())),
                    "MINUTES" => Value::Int64(i64::from(time.minute())),
                    // xsd:decimal seconds, the fraction exact to the nanosecond.
                    _ => Value::Decimal(
                        Decimal::new(
                            i128::from(time.second()) * 1_000_000_000
                                + i128::from(time.nanosecond()),
                            9,
                        )
                        .ok()?
                        .normalize(),
                    ),
                })
            }
            "TIMEZONE" => {
                let value = self.eval_expr(arg(0)?, chunk, row)?;
                let offset = match native(&value)?.into_owned() {
                    Value::ZonedDatetime(zdt) => zdt.offset_seconds(),
                    Value::Time(t) => t.offset_seconds()?,
                    _ => return None,
                };
                Some(Value::RdfTerm(Arc::new(Term::typed_literal(
                    day_time_duration(offset),
                    format!("{XSD}dayTimeDuration"),
                ))))
            }
            "TZ" => {
                let value = self.eval_expr(arg(0)?, chunk, row)?;
                Some(Value::from(match native(&value)?.into_owned() {
                    Value::ZonedDatetime(zdt) => format_tz_offset(zdt.offset_seconds()),
                    Value::Time(t) => t.offset_seconds().map(format_tz_offset).unwrap_or_default(),
                    Value::Timestamp(_) => String::new(),
                    _ => return None,
                }))
            }

            // ---- Hash functions (17.4.6): a simple literal or xsd:string ----
            "MD5" | "SHA1" | "SHA256" | "SHA384" | "SHA512" => {
                let text = self.simple_string(arg(0), chunk, row)?;
                Some(Value::from(hash_hex(&func_name, text.as_bytes())?))
            }

            // ---- XPath constructor functions (17.5), called by their IRI ----
            iri if iri.starts_with("HTTP://WWW.W3.ORG/2001/XMLSCHEMA#") => {
                let value = self.eval_expr(arg(0)?, chunk, row)?;
                let target = &iri["HTTP://WWW.W3.ORG/2001/XMLSCHEMA#".len()..];
                xsd_cast(target, &cast_source(&value)?)
            }
            _ => None,
        }
    }

    /// Two string arguments that are argument-compatible (17.4.3.1.1).
    fn compatible_pair(
        &self,
        args: &[FilterExpression],
        chunk: &DataChunk,
        row: usize,
    ) -> Option<(StringArg, StringArg)> {
        let a = self.string_arg(args.first(), chunk, row)?;
        let b = self.string_arg(args.get(1), chunk, row)?;
        a.compatible_with(&b).then_some((a, b))
    }

    /// The regular expression of a pattern and its flags (17.4.3.14,
    /// XPath's `s`, `m`, `i`, `x` and `q`).
    fn regex(
        &self,
        pattern: Option<&FilterExpression>,
        flags: Option<&FilterExpression>,
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Regex> {
        let pattern = self.simple_string(pattern, chunk, row)?;
        let flags = match flags {
            Some(_) => self.simple_string(flags, chunk, row)?,
            None => Arc::from(""),
        };
        let mut inline = String::new();
        let mut quoted = false;
        for flag in flags.chars() {
            match flag {
                'i' | 's' | 'm' | 'x' => inline.push(flag),
                'q' => quoted = true,
                _ => return None,
            }
        }
        let body = if quoted {
            regex_escape(&pattern)
        } else {
            pattern.to_string()
        };
        let source = if inline.is_empty() {
            body
        } else {
            format!("(?{inline}){body}")
        };
        Regex::new(&source).ok()
    }
}

impl Predicate for RdfExpressionPredicate {
    /// A FILTER keeps a row whose expression's effective boolean value is
    /// true (17.2.2); an error eliminates it, it never fails the query.
    fn evaluate(&self, chunk: &DataChunk, row: usize) -> Result<bool, OperatorError> {
        Ok(self
            .eval(chunk, row)
            .and_then(|value| effective_boolean(&value))
            == Some(true))
    }
}

/// `expression` with each constant a comparison reads (`=`, `!=`, `<`, `<=`,
/// `>`, `>=`, `IN`) that is a literal the id scheme inlines replaced by its
/// value, so a row does not parse it again: for an operand, an inlined
/// literal's value is the literal (its canonical form). Constants elsewhere
/// keep their terms (`STR` reads their lexical form).
fn fold_compared_constants(expression: FilterExpression) -> FilterExpression {
    use grafeo_common::types::TermId;
    let fold_operand = |operand: FilterExpression| match operand {
        FilterExpression::Literal(Value::RdfTerm(term)) => {
            match TermId::inline(&term).and_then(TermId::inline_value) {
                Some(value) => FilterExpression::Literal(value),
                None => FilterExpression::Literal(Value::RdfTerm(term)),
            }
        }
        other => fold_compared_constants(other),
    };
    match expression {
        FilterExpression::Binary { left, op, right } => {
            let compares = matches!(
                op,
                BinaryFilterOp::Eq
                    | BinaryFilterOp::Ne
                    | BinaryFilterOp::Lt
                    | BinaryFilterOp::Le
                    | BinaryFilterOp::Gt
                    | BinaryFilterOp::Ge
            );
            let (left, right) = if compares {
                (fold_operand(*left), fold_operand(*right))
            } else if op == BinaryFilterOp::In {
                let right = match *right {
                    FilterExpression::List(items) => {
                        FilterExpression::List(items.into_iter().map(fold_operand).collect())
                    }
                    other => fold_compared_constants(other),
                };
                (fold_operand(*left), right)
            } else {
                (
                    fold_compared_constants(*left),
                    fold_compared_constants(*right),
                )
            };
            FilterExpression::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            }
        }
        FilterExpression::Unary { op, operand } => FilterExpression::Unary {
            op,
            operand: Box::new(fold_compared_constants(*operand)),
        },
        FilterExpression::SparqlCall { name, args } => FilterExpression::SparqlCall {
            name,
            args: args.into_iter().map(fold_compared_constants).collect(),
        },
        other => other,
    }
}

/// An IRI term as a value.
fn iri_value(iri: &str) -> Value {
    Value::RdfTerm(Arc::new(Term::iri(iri)))
}

#[allow(
    clippy::cast_precision_loss,
    reason = "fn:substring computes positions as doubles (XPath 7.4.3)"
)]
fn as_f64(number: &Value) -> f64 {
    match number {
        Value::Int64(i) => *i as f64,
        Value::Decimal(d) => d.to_f64(),
        Value::Float64(f) => *f,
        _ => f64::NAN,
    }
}

/// `fn:substring` (XPath 7.4.3): the characters at 1-based positions `p`
/// with `round(start) <= p < round(start) + round(length)`.
fn substring(text: &str, start: f64, length: Option<f64>) -> String {
    let first = round_half_up(start);
    let end = length.map_or(f64::INFINITY, |length| first + round_half_up(length));
    text.chars()
        .zip(1u32..)
        .filter(|&(_, position)| {
            let position = f64::from(position);
            position >= first && position < end
        })
        .map(|(c, _)| c)
        .collect()
}

/// fn:round of a double: the nearest integer, of two equally near ones the
/// one nearer positive infinity, keeping the sign of a negative value that
/// rounds to zero. `v - v.floor()` is exact for every finite double.
pub(super) fn round_half_up(v: f64) -> f64 {
    let floor = v.floor();
    let rounded = if v - floor >= 0.5 { floor + 1.0 } else { floor };
    if rounded == 0.0 {
        0.0_f64.copysign(v)
    } else {
        rounded
    }
}

/// A SPARQL numeric operand by its XPath type.
enum Numeric {
    Integer(i64),
    Decimal(Decimal),
    Double(f64),
}

fn numeric(value: &Value) -> Option<Numeric> {
    match value {
        Value::Int64(i) => Some(Numeric::Integer(*i)),
        Value::Decimal(d) => Some(Numeric::Decimal(*d)),
        Value::Float64(f) => Some(Numeric::Double(*f)),
        _ => None,
    }
}

/// `+`, `-`, `*` and `/` by the XPath numeric promotion of SPARQL 1.1
/// (`op:numeric-add` and the others): a double makes a double; otherwise a
/// decimal makes a decimal; two integers give an integer, except a division,
/// which gives a decimal, and a result past the `i64` range, computed as
/// decimals. Division by zero of integers or decimals and decimal overflow
/// are errors (`None`).
#[allow(
    clippy::cast_precision_loss,
    reason = "an integer meeting a double is promoted to that double, as XPath promotes it"
)]
pub(super) fn numeric_op(left: &Value, op: BinaryFilterOp, right: &Value) -> Option<Value> {
    let (l, r) = (numeric(left)?, numeric(right)?);
    let as_double = |n: &Numeric| match n {
        Numeric::Integer(i) => *i as f64,
        Numeric::Decimal(d) => d.to_f64(),
        Numeric::Double(f) => *f,
    };
    if matches!(l, Numeric::Double(_)) || matches!(r, Numeric::Double(_)) {
        let (a, b) = (as_double(&l), as_double(&r));
        return match op {
            BinaryFilterOp::Add => Some(Value::Float64(a + b)),
            BinaryFilterOp::Sub => Some(Value::Float64(a - b)),
            BinaryFilterOp::Mul => Some(Value::Float64(a * b)),
            // IEEE 754: a double divided by zero is INF, -INF or NaN.
            BinaryFilterOp::Div => Some(Value::Float64(a / b)),
            _ => None,
        };
    }
    if let (Numeric::Integer(a), Numeric::Integer(b)) = (&l, &r) {
        let integer = match op {
            BinaryFilterOp::Add => a.checked_add(*b),
            BinaryFilterOp::Sub => a.checked_sub(*b),
            BinaryFilterOp::Mul => a.checked_mul(*b),
            _ => None,
        };
        if let Some(integer) = integer {
            return Some(Value::Int64(integer));
        }
    }
    let as_decimal = |n: &Numeric| match n {
        Numeric::Integer(i) => Some(Decimal::from_i64(*i)),
        Numeric::Decimal(d) => Some(*d),
        Numeric::Double(_) => None,
    };
    let (a, b) = (as_decimal(&l)?, as_decimal(&r)?);
    match op {
        BinaryFilterOp::Add => a.checked_add(&b),
        BinaryFilterOp::Sub => a.checked_sub(&b),
        BinaryFilterOp::Mul => a.checked_mul(&b),
        BinaryFilterOp::Div => a.checked_div(&b),
        _ => return None,
    }
    .ok()
    .map(Value::Decimal)
}

/// What an XPath cast reads from its argument: a literal's value, or its text
/// when it has none of its own (a simple literal, an IRI for `xsd:string`).
fn cast_source(value: &Value) -> Option<Value> {
    if let Some(native) = native(value) {
        return Some(native.into_owned());
    }
    match value {
        Value::RdfTerm(term) => match term.as_ref() {
            // A literal whose text is not of its type casts from that text.
            Term::Literal(literal) if literal.language().is_none() => {
                Some(Value::from(literal.value()))
            }
            Term::Iri(iri) => Some(Value::RdfTerm(Arc::new(Term::iri(iri.as_str())))),
            _ => None,
        },
        _ => None,
    }
}

/// The XPath cast of `value` to the XSD type `name` (in capitals): `DECIMAL`,
/// `DOUBLE`, `FLOAT`, `INTEGER` (and `INT`, `LONG`), `STRING`, `BOOLEAN`,
/// `DATE`, `TIME` or `DATETIME`.
/// A number casts to a number (a double by its shortest text to a decimal, a
/// decimal or a double to an integer by truncation); a string casts as the
/// lexical form of the type; only `xsd:string` takes an IRI. `None` for
/// another type, a value that does not cast, NaN or an infinity to a decimal
/// or an integer.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "XPath casts an integer to the nearest double, and a double to an integer by \
              truncation after the range check"
)]
fn xsd_cast(name: &str, value: &Value) -> Option<Value> {
    if let Value::RdfTerm(term) = value {
        return match (name, term.as_ref()) {
            ("STRING", Term::Iri(iri)) => Some(Value::from(iri.as_str())),
            _ => None,
        };
    }
    match name {
        "DECIMAL" => match value {
            Value::Decimal(d) => Some(Value::Decimal(*d)),
            Value::Int64(i) => Some(Value::Decimal(Decimal::from_i64(*i))),
            Value::Float64(f) => Decimal::from_f64(*f).ok().map(Value::Decimal),
            Value::Bool(b) => Some(Value::Decimal(Decimal::from_i64(i64::from(*b)))),
            Value::String(s) => Decimal::parse(s.trim()).ok().map(Value::Decimal),
            _ => None,
        },
        "DOUBLE" | "FLOAT" => match value {
            Value::Float64(f) => Some(Value::Float64(*f)),
            Value::Int64(i) => Some(Value::Float64(*i as f64)),
            Value::Decimal(d) => Some(Value::Float64(d.to_f64())),
            Value::Bool(b) => Some(Value::Float64(f64::from(u8::from(*b)))),
            Value::String(s) => Literal::typed(s.as_str(), Literal::XSD_DOUBLE).native_value(),
            _ => None,
        },
        "INTEGER" | "INT" | "LONG" => match value {
            Value::Int64(i) => Some(Value::Int64(*i)),
            Value::Decimal(d) => d.trunc_to_i64().map(Value::Int64),
            Value::Float64(f) if f.is_finite() && f.abs() < 9.223_372_036_854_775e18 => {
                Some(Value::Int64(f.trunc() as i64))
            }
            Value::Bool(b) => Some(Value::Int64(i64::from(*b))),
            Value::String(s) => s.trim().parse::<i64>().ok().map(Value::Int64),
            _ => None,
        },
        "STRING" => terms::native_literal(value).map(|literal| Value::from(literal.value())),
        "BOOLEAN" => match value {
            Value::Bool(b) => Some(Value::Bool(*b)),
            Value::Int64(i) => Some(Value::Bool(*i != 0)),
            Value::Decimal(d) => Some(Value::Bool(d.signum() != 0)),
            Value::Float64(f) => Some(Value::Bool(*f != 0.0 && !f.is_nan())),
            Value::String(s) => match s.trim() {
                "true" | "1" => Some(Value::Bool(true)),
                "false" | "0" => Some(Value::Bool(false)),
                _ => None,
            },
            _ => None,
        },
        // A date from its text or from a dateTime (its local date).
        "DATE" => match value {
            Value::Date(d) => Some(Value::Date(*d)),
            Value::String(s) => grafeo_common::types::Date::parse(s.trim()).map(Value::Date),
            Value::Timestamp(ts) => Some(Value::Date(ts.to_date())),
            Value::ZonedDatetime(zdt) => Some(Value::Date(zdt.to_local_date())),
            _ => None,
        },
        // A time from its text or from a dateTime (its local time).
        "TIME" => match value {
            Value::Time(t) => Some(Value::Time(*t)),
            Value::String(s) => grafeo_common::types::Time::parse(s.trim()).map(Value::Time),
            Value::Timestamp(ts) => Some(Value::Time(ts.to_time())),
            Value::ZonedDatetime(zdt) => Some(Value::Time(zdt.to_local_time())),
            _ => None,
        },
        // A dateTime from its text (zoned when it has an offset) or from a
        // date (its midnight).
        "DATETIME" => match value {
            Value::Timestamp(_) | Value::ZonedDatetime(_) => Some(value.clone()),
            Value::String(s) => Literal::typed(s.as_str(), Literal::XSD_DATETIME).native_value(),
            Value::Date(d) => grafeo_common::types::Time::parse("00:00:00").map(|midnight| {
                Value::Timestamp(grafeo_common::types::Timestamp::from_date_time(
                    *d, midnight,
                ))
            }),
            _ => None,
        },
        _ => None,
    }
}

/// An offset as an `xsd:dayTimeDuration` lexical form (`PT1H`, `-PT5H30M`,
/// `PT0S`).
fn day_time_duration(offset_seconds: i32) -> String {
    let sign = if offset_seconds < 0 { "-" } else { "" };
    let total = offset_seconds.unsigned_abs();
    let (hours, minutes, seconds) = (total / 3600, (total % 3600) / 60, total % 60);
    if total == 0 {
        return "PT0S".to_string();
    }
    let mut text = format!("{sign}PT");
    if hours > 0 {
        let _ = write!(text, "{hours}H");
    }
    if minutes > 0 {
        let _ = write!(text, "{minutes}M");
    }
    if seconds > 0 {
        let _ = write!(text, "{seconds}S");
    }
    text
}

/// Formats a timezone offset in seconds as "+HH:MM" or "Z".
fn format_tz_offset(offset_secs: i32) -> String {
    if offset_secs == 0 {
        return "Z".to_string();
    }
    let sign = if offset_secs >= 0 { '+' } else { '-' };
    let abs = offset_secs.unsigned_abs();
    format!("{sign}{:02}:{:02}", abs / 3600, (abs % 3600) / 60)
}

/// The hex digest of `bytes` by the SPARQL hash function `name`.
fn hash_hex(name: &str, bytes: &[u8]) -> Option<String> {
    let hex = |digest: &[u8]| {
        digest.iter().fold(String::new(), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
    };
    Some(match name {
        "MD5" => format!("{:x}", md5::compute(bytes)),
        "SHA1" => {
            use sha1::Digest as _;
            hex(&sha1::Sha1::digest(bytes))
        }
        "SHA256" => {
            use sha2::Digest as _;
            hex(&sha2::Sha256::digest(bytes))
        }
        "SHA384" => {
            use sha2::Digest as _;
            hex(&sha2::Sha384::digest(bytes))
        }
        "SHA512" => {
            use sha2::Digest as _;
            hex(&sha2::Sha512::digest(bytes))
        }
        _ => return None,
    })
}

/// 64 bits that differ on every call (see [`grafeo_common::utils::random`]).
fn random_bits() -> u64 {
    grafeo_common::utils::random::next_u64()
}

/// A double in `[0, 1)`.
fn random_unit() -> f64 {
    grafeo_common::utils::random::next_f64()
}

/// A version 4 UUID (RFC 9562) in its hyphenated form.
fn uuid_v4() -> String {
    let high = random_bits();
    let low = random_bits();
    let bytes: Vec<u8> = high
        .to_be_bytes()
        .into_iter()
        .chain(low.to_be_bytes())
        .collect();
    let mut bytes: [u8; 16] = bytes.try_into().expect("16 bytes");
    bytes[6] = (bytes[6] & 0x0F) | 0x40;
    bytes[8] = (bytes[8] & 0x3F) | 0x80;
    let hex = bytes.iter().fold(String::with_capacity(32), |mut hex, b| {
        let _ = write!(hex, "{b:02x}");
        hex
    });
    format!(
        "{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..]
    )
}

/// `text` with every regex metacharacter escaped (the `q` flag).
fn regex_escape(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for c in text.chars() {
        if "\\.+*?()|[]{}^$#&-~".contains(c) {
            escaped.push('\\');
        }
        escaped.push(c);
    }
    escaped
}

#[cfg(test)]
mod tests {
    use super::*;

    /// fn:round takes a half towards positive infinity, keeps the sign of a
    /// negative value that rounds to zero, and passes infinities and NaN on.
    #[test]
    fn round_half_up_follows_fn_round() {
        assert_eq!(round_half_up(2.5), 3.0);
        assert_eq!(round_half_up(-2.5), -2.0);
        assert_eq!(round_half_up(-2.6), -3.0);
        assert_eq!(round_half_up(0.49999999999999994), 0.0);
        assert!(round_half_up(-0.4).is_sign_negative());
        assert_eq!(round_half_up(-0.4), 0.0);
        assert_eq!(round_half_up(-0.5), 0.0);
        assert!(round_half_up(-0.5).is_sign_negative());
        assert_eq!(round_half_up(f64::INFINITY), f64::INFINITY);
        assert_eq!(round_half_up(f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert!(round_half_up(f64::NAN).is_nan());
        assert_eq!(
            round_half_up(4_503_599_627_370_495.5),
            4_503_599_627_370_496.0
        );
    }

    /// A double divided by zero follows IEEE 754; integer and decimal
    /// division by zero leave the result unbound.
    #[test]
    fn numeric_division_by_zero() {
        let divide = |left: Value, right: Value| numeric_op(&left, BinaryFilterOp::Div, &right);
        assert_eq!(
            divide(Value::Float64(1.0), Value::Int64(0)),
            Some(Value::Float64(f64::INFINITY))
        );
        assert_eq!(
            divide(Value::Float64(-1.0), Value::Int64(0)),
            Some(Value::Float64(f64::NEG_INFINITY))
        );
        assert!(matches!(
            divide(Value::Float64(0.0), Value::Float64(0.0)),
            Some(Value::Float64(v)) if v.is_nan()
        ));
        assert_eq!(divide(Value::Int64(1), Value::Int64(0)), None);
        assert_eq!(
            divide(
                Value::Decimal(Decimal::new(15, 1).unwrap()),
                Value::Int64(0)
            ),
            None
        );
    }

    #[test]
    fn substring_counts_from_one_and_rounds() {
        assert_eq!(substring("motorcar", 6.0, None), "car");
        assert_eq!(substring("metadata", 4.0, Some(3.0)), "ada");
        assert_eq!(substring("12345", 1.5, Some(2.6)), "234");
        assert_eq!(substring("12345", 0.0, Some(3.0)), "12");
        assert_eq!(substring("Kraków", 5.0, None), "ów");
    }

    #[test]
    fn offsets_as_day_time_durations() {
        assert_eq!(day_time_duration(0), "PT0S");
        assert_eq!(day_time_duration(3600), "PT1H");
        assert_eq!(day_time_duration(-19_800), "-PT5H30M");
        assert_eq!(format_tz_offset(-18_000), "-05:00");
    }

    #[test]
    fn a_uuid_has_the_version_and_variant_bits() {
        let uuid = uuid_v4();
        assert_eq!(uuid.len(), 36);
        assert_eq!(&uuid[14..15], "4");
        assert!("89ab".contains(&uuid[19..20]), "{uuid}");
        assert_ne!(uuid, uuid_v4());
    }
}
