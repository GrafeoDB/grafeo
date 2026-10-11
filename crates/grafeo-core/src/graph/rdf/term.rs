//! RDF Terms.
//!
//! RDF terms are the building blocks of triples. There are four types:
//! - IRIs (Internationalized Resource Identifiers)
//! - Blank nodes (anonymous nodes)
//! - Literals (data values)
//! - Variables (for query patterns, not stored)

use grafeo_common::storage::log_record::TermRecord;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;

/// An RDF term.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Term {
    /// An IRI (Internationalized Resource Identifier).
    Iri(Iri),
    /// A blank node (anonymous node).
    BlankNode(BlankNode),
    /// A literal value.
    Literal(Literal),
}

impl Term {
    /// Creates an IRI term.
    #[inline]
    pub fn iri(value: impl Into<Arc<str>>) -> Self {
        Term::Iri(Iri::new(value))
    }

    /// Creates a blank node term.
    #[inline]
    pub fn blank(id: impl Into<Arc<str>>) -> Self {
        Term::BlankNode(BlankNode::new(id))
    }

    /// Creates a simple literal (xsd:string).
    #[inline]
    pub fn literal(value: impl Into<Arc<str>>) -> Self {
        Term::Literal(Literal::simple(value))
    }

    /// Creates a typed literal.
    #[inline]
    pub fn typed_literal(value: impl Into<Arc<str>>, datatype: impl Into<Arc<str>>) -> Self {
        Term::Literal(Literal::typed(value, datatype))
    }

    /// Creates a language-tagged literal.
    #[inline]
    pub fn lang_literal(value: impl Into<Arc<str>>, lang: impl Into<Arc<str>>) -> Self {
        Term::Literal(Literal::with_language(value, lang))
    }

    /// Returns true if this term is an IRI.
    #[inline]
    #[must_use]
    pub fn is_iri(&self) -> bool {
        matches!(self, Term::Iri(_))
    }

    /// Returns true if this term is a blank node.
    #[inline]
    #[must_use]
    pub fn is_blank_node(&self) -> bool {
        matches!(self, Term::BlankNode(_))
    }

    /// Returns true if this term is a literal.
    #[inline]
    #[must_use]
    pub fn is_literal(&self) -> bool {
        matches!(self, Term::Literal(_))
    }

    /// Returns the IRI if this term is an IRI.
    #[inline]
    #[must_use]
    pub fn as_iri(&self) -> Option<&Iri> {
        match self {
            Term::Iri(iri) => Some(iri),
            _ => None,
        }
    }

    /// Returns the blank node if this term is a blank node.
    #[inline]
    #[must_use]
    pub fn as_blank_node(&self) -> Option<&BlankNode> {
        match self {
            Term::BlankNode(bn) => Some(bn),
            _ => None,
        }
    }

    /// Returns the literal if this term is a literal.
    #[inline]
    #[must_use]
    pub fn as_literal(&self) -> Option<&Literal> {
        match self {
            Term::Literal(lit) => Some(lit),
            _ => None,
        }
    }
}

impl Term {
    /// Reads a term from its N-Triples string, the inverse of the term's
    /// [`Display`](fmt::Display): `Term::from_ntriples(&term.to_string())`
    /// gives `term` back for every term.
    ///
    /// The forms:
    /// - `<iri>` for IRIs: everything between the first `<` and the last `>`
    /// - `_:id` for blank nodes: everything after `_:`
    /// - `"value"` for simple literals
    /// - `"value"^^<type>` for typed literals
    /// - `"value"@lang` for language-tagged literals: everything after `@`
    ///
    /// The string is read as it is: whitespace before or after a term is
    /// refused, and whitespace inside an id, a tag or an IRI belongs to it. A
    /// literal's value is UTF-8 text with the N-Triples escapes: a backslash
    /// before `t`, `b`, `n`, `r`, `f`, `"`, `'` or a backslash stands for
    /// that character, and before `u` with 4 or `U` with 8 hexadecimal digits
    /// for the Unicode scalar value they spell. IRIs, ids, tags and datatypes
    /// have no escapes, as [`Display`](fmt::Display) writes none there.
    ///
    /// # Errors
    ///
    /// Returns a [`TermParseError`] naming what is wrong and where: an empty
    /// string, a string that starts as no term, an IRI without its closing
    /// `>`, a literal without its closing quote, an unknown or unfinished
    /// escape, an escape that spells no Unicode scalar value, or anything
    /// after a literal other than a language tag or a datatype.
    pub fn from_ntriples(s: &str) -> Result<Self, TermParseError> {
        if s.is_empty() {
            return Err(TermParseError::new(s, "an empty string"));
        }
        if let Some(rest) = s.strip_prefix('<') {
            let inner = rest
                .strip_suffix('>')
                .ok_or_else(|| TermParseError::new(s, "an IRI without its closing '>'"))?;
            return Ok(Term::Iri(Iri::new(inner)));
        }
        if let Some(id) = s.strip_prefix("_:") {
            return Ok(Term::BlankNode(BlankNode::new(id)));
        }
        if s.starts_with('"') {
            let (value, end) =
                read_literal_value(s).map_err(|reason| TermParseError::new(s, reason))?;
            let rest = &s[end..];
            return if rest.is_empty() {
                Ok(Term::Literal(Literal::simple(value)))
            } else if let Some(language) = rest.strip_prefix('@') {
                Ok(Term::Literal(Literal::with_language(value, language)))
            } else if let Some(datatype) = rest
                .strip_prefix("^^<")
                .and_then(|datatype| datatype.strip_suffix('>'))
            {
                Ok(Term::Literal(Literal::typed(value, datatype)))
            } else {
                Err(TermParseError::new(
                    s,
                    format!(
                        "text after the literal's closing quote at byte {end}, neither a \
                         language tag ('@') nor a datatype ('^^<...>')"
                    ),
                ))
            };
        }
        Err(TermParseError::new(
            s,
            "neither an IRI ('<'), a blank node ('_:') nor a literal ('\"')",
        ))
    }

    /// Converts this term to its N-Triples string representation.
    ///
    /// Round-trips with [`from_ntriples`](Self::from_ntriples).
    pub fn to_ntriples(&self) -> String {
        self.to_string()
    }
}

/// Reads the value of the literal that starts `s` (with its opening quote):
/// the value with its escapes decoded, and the byte offset after the closing
/// quote. The error is the reason, with the byte offset where it applies.
fn read_literal_value(s: &str) -> Result<(String, usize), String> {
    let mut value = String::new();
    let mut chars = s.char_indices().skip(1);
    while let Some((offset, ch)) = chars.next() {
        match ch {
            '"' => return Ok((value, offset + 1)),
            '\\' => {
                let Some((_, escape)) = chars.next() else {
                    return Err(format!("an escape without its character at byte {offset}"));
                };
                let decoded = match escape {
                    't' => '\t',
                    'b' => '\x08',
                    'n' => '\n',
                    'r' => '\r',
                    'f' => '\x0c',
                    '"' => '"',
                    '\'' => '\'',
                    '\\' => '\\',
                    'u' => read_unicode_escape(&mut chars, 4, offset)?,
                    'U' => read_unicode_escape(&mut chars, 8, offset)?,
                    other => {
                        return Err(format!("an unknown escape '\\{other}' at byte {offset}"));
                    }
                };
                value.push(decoded);
            }
            other => value.push(other),
        }
    }
    Err("a literal without its closing quote".to_string())
}

/// Reads the `digits` hexadecimal digits of the escape at byte `offset` and
/// returns the character they spell.
fn read_unicode_escape(
    chars: &mut impl Iterator<Item = (usize, char)>,
    digits: usize,
    offset: usize,
) -> Result<char, String> {
    let mut code = 0u32;
    for _ in 0..digits {
        let digit = chars
            .next()
            .and_then(|(_, ch)| ch.to_digit(16))
            .ok_or_else(|| {
                format!("the escape at byte {offset} needs {digits} hexadecimal digits")
            })?;
        code = code * 16 + digit;
    }
    char::from_u32(code).ok_or_else(|| {
        format!(
            "the escape at byte {offset} spells U+{code:X}, which is not a Unicode scalar value"
        )
    })
}

/// How many characters of a refused string a [`TermParseError`] shows.
const EXCERPT_CHARS: usize = 64;

/// A string that [`Term::from_ntriples`] does not read as a term: what is
/// wrong, and the start of the string.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("not an N-Triples term ({reason}): {excerpt:?}")]
pub struct TermParseError {
    /// What is wrong, with the byte offset where it applies.
    reason: String,
    /// The string, cut after [`EXCERPT_CHARS`] characters.
    excerpt: String,
}

impl TermParseError {
    /// The error for `text`, which is not a term for `reason`.
    fn new(text: &str, reason: impl Into<String>) -> Self {
        let mut excerpt: String = text.chars().take(EXCERPT_CHARS).collect();
        if excerpt.len() < text.len() {
            excerpt.push_str("...");
        }
        Self {
            reason: reason.into(),
            excerpt,
        }
    }
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Term::Iri(iri) => write!(f, "{}", iri),
            Term::BlankNode(bn) => write!(f, "{}", bn),
            Term::Literal(lit) => write!(f, "{}", lit),
        }
    }
}

/// An IRI (Internationalized Resource Identifier).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Iri {
    /// The IRI string.
    value: Arc<str>,
}

impl Iri {
    /// Creates a new IRI.
    #[inline]
    pub fn new(value: impl Into<Arc<str>>) -> Self {
        Self {
            value: value.into(),
        }
    }

    /// Returns the IRI string.
    #[inline]
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.value
    }

    /// Returns the local name (part after last # or /).
    #[must_use]
    pub fn local_name(&self) -> &str {
        if let Some(pos) = self.value.rfind('#') {
            &self.value[pos + 1..]
        } else if let Some(pos) = self.value.rfind('/') {
            &self.value[pos + 1..]
        } else {
            &self.value
        }
    }

    /// Returns the namespace (part before local name).
    #[must_use]
    pub fn namespace(&self) -> &str {
        if let Some(pos) = self.value.rfind('#') {
            &self.value[..=pos]
        } else if let Some(pos) = self.value.rfind('/') {
            &self.value[..=pos]
        } else {
            ""
        }
    }
}

impl fmt::Display for Iri {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<{}>", self.value)
    }
}

impl From<&str> for Iri {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for Iri {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

/// A blank node (anonymous node).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlankNode {
    /// The blank node identifier.
    id: Arc<str>,
}

impl BlankNode {
    /// Creates a new blank node with the given identifier.
    #[inline]
    pub fn new(id: impl Into<Arc<str>>) -> Self {
        Self { id: id.into() }
    }

    /// Returns the blank node identifier.
    #[inline]
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }
}

impl fmt::Display for BlankNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "_:{}", self.id)
    }
}

/// An RDF literal (data value).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Literal {
    /// The lexical form (string value).
    value: Arc<str>,
    /// The datatype IRI (e.g., xsd:string, xsd:integer).
    datatype: Arc<str>,
    /// Optional language tag (e.g., "en", "de").
    language: Option<Arc<str>>,
}

impl Literal {
    /// XSD namespace.
    pub const XSD: &'static str = "http://www.w3.org/2001/XMLSchema#";

    /// RDF namespace.
    pub const RDF: &'static str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";

    /// xsd:string datatype IRI.
    pub const XSD_STRING: &'static str = "http://www.w3.org/2001/XMLSchema#string";

    /// xsd:integer datatype IRI.
    pub const XSD_INTEGER: &'static str = "http://www.w3.org/2001/XMLSchema#integer";

    /// xsd:decimal datatype IRI.
    pub const XSD_DECIMAL: &'static str = "http://www.w3.org/2001/XMLSchema#decimal";

    /// xsd:double datatype IRI.
    pub const XSD_DOUBLE: &'static str = "http://www.w3.org/2001/XMLSchema#double";

    /// xsd:boolean datatype IRI.
    pub const XSD_BOOLEAN: &'static str = "http://www.w3.org/2001/XMLSchema#boolean";

    /// xsd:dateTime datatype IRI.
    pub const XSD_DATETIME: &'static str = "http://www.w3.org/2001/XMLSchema#dateTime";

    /// rdf:langString datatype IRI.
    pub const RDF_LANG_STRING: &'static str =
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString";

    /// Creates a simple string literal.
    #[inline]
    pub fn simple(value: impl Into<Arc<str>>) -> Self {
        Self {
            value: value.into(),
            datatype: Self::XSD_STRING.into(),
            language: None,
        }
    }

    /// Creates a typed literal.
    #[inline]
    pub fn typed(value: impl Into<Arc<str>>, datatype: impl Into<Arc<str>>) -> Self {
        Self {
            value: value.into(),
            datatype: datatype.into(),
            language: None,
        }
    }

    /// Creates a language-tagged literal.
    #[inline]
    pub fn with_language(value: impl Into<Arc<str>>, language: impl Into<Arc<str>>) -> Self {
        Self {
            value: value.into(),
            datatype: Self::RDF_LANG_STRING.into(),
            language: Some(language.into()),
        }
    }

    /// Creates an integer literal.
    #[inline]
    pub fn integer(value: i64) -> Self {
        Self::typed(value.to_string(), Self::XSD_INTEGER)
    }

    /// Creates a double literal.
    #[inline]
    pub fn double(value: f64) -> Self {
        Self::typed(value.to_string(), Self::XSD_DOUBLE)
    }

    /// Creates a boolean literal.
    #[inline]
    pub fn boolean(value: bool) -> Self {
        Self::typed(if value { "true" } else { "false" }, Self::XSD_BOOLEAN)
    }

    /// Returns the lexical form.
    #[inline]
    #[must_use]
    pub fn value(&self) -> &str {
        &self.value
    }

    /// Returns the datatype IRI.
    #[inline]
    #[must_use]
    pub fn datatype(&self) -> &str {
        &self.datatype
    }

    /// Returns the language tag, if any.
    #[inline]
    #[must_use]
    pub fn language(&self) -> Option<&str> {
        self.language.as_deref()
    }

    /// Returns true if this is a simple string literal.
    #[inline]
    #[must_use]
    pub fn is_simple(&self) -> bool {
        self.datatype.as_ref() == Self::XSD_STRING && self.language.is_none()
    }

    /// Returns true if this literal has a language tag.
    #[inline]
    #[must_use]
    pub fn is_lang_string(&self) -> bool {
        self.language.is_some()
    }

    /// Attempts to parse the literal as an integer.
    #[must_use]
    pub fn as_integer(&self) -> Option<i64> {
        self.value.parse().ok()
    }

    /// Attempts to parse the literal as a double.
    #[must_use]
    pub fn as_double(&self) -> Option<f64> {
        self.value.parse().ok()
    }

    /// Attempts to parse the literal as a boolean.
    #[must_use]
    pub fn as_boolean(&self) -> Option<bool> {
        match self.value.as_ref() {
            "true" | "1" => Some(true),
            "false" | "0" => Some(false),
            _ => None,
        }
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Escape special characters in the value
        write!(f, "\"")?;
        for ch in self.value.chars() {
            match ch {
                '"' => write!(f, "\\\"")?,
                '\\' => write!(f, "\\\\")?,
                '\n' => write!(f, "\\n")?,
                '\r' => write!(f, "\\r")?,
                '\t' => write!(f, "\\t")?,
                _ => write!(f, "{}", ch)?,
            }
        }
        write!(f, "\"")?;

        if let Some(ref lang) = self.language {
            write!(f, "@{}", lang)
        } else if self.datatype.as_ref() != Self::XSD_STRING {
            write!(f, "^^<{}>", self.datatype)
        } else {
            Ok(())
        }
    }
}

/// A term as a log record or a change set stores it: every part kept as
/// it is, so [`Term::from`] the record gives the term back.
impl From<&Term> for TermRecord {
    fn from(term: &Term) -> Self {
        match term {
            Term::Iri(iri) => Self::Iri(iri.as_str().to_string()),
            Term::BlankNode(blank) => Self::Blank(blank.id().to_string()),
            Term::Literal(literal) => Self::Literal {
                value: literal.value.to_string(),
                datatype: literal.datatype.to_string(),
                language: literal.language.as_deref().map(ToString::to_string),
            },
        }
    }
}

/// The term a log record or a change set stores.
impl From<&TermRecord> for Term {
    fn from(record: &TermRecord) -> Self {
        match record {
            TermRecord::Iri(iri) => Self::iri(iri.as_str()),
            TermRecord::Blank(id) => Self::blank(id.as_str()),
            TermRecord::Literal {
                value,
                datatype,
                language,
            } => Self::Literal(Literal {
                value: value.as_str().into(),
                datatype: datatype.as_str().into(),
                language: language.as_deref().map(Into::into),
            }),
        }
    }
}

/// `~` stands for a backslash in the N-Triples strings of these tests.
#[cfg(test)]
pub(crate) fn escaped(text: &str) -> String {
    text.replace('~', "\\")
}

/// Terms whose N-Triples strings hold non-ASCII text, every character
/// `Display` escapes, and whitespace that belongs to the term.
#[cfg(test)]
pub(crate) fn unusual_terms() -> Vec<Term> {
    vec![
        Term::literal("Kraków"),
        Term::literal("Ámsterdam"),
        Term::literal("🚲 naar Amsterdam"),
        Term::literal("阿姆斯特丹"),
        Term::lang_literal("アムステルダム", "ja"),
        Term::lang_literal("Praha", "cs"),
        Term::lang_literal("Berlin", "de-DE"),
        Term::lang_literal(" Paris ", "fr "),
        Term::lang_literal("", ""),
        Term::typed_literal("1030", Literal::XSD_INTEGER),
        Term::typed_literal("Kraków", "http://example.org/types/miasto"),
        Term::typed_literal("Gus", "http://example.org/a>b"),
        Term::literal("Gus \"de bus\"\nAmsterdam\r\t\\"),
        Term::literal("\x08\x0c'"),
        Term::literal(escaped("~u00F3 stays six characters")),
        Term::literal(""),
        Term::literal("  Mia  "),
        Term::blank("b0"),
        Term::blank("b0 "),
        Term::blank(" Vincent"),
        Term::blank("jules@paris"),
        Term::blank("Kraków"),
        Term::blank(""),
        Term::iri("http://example.org/Kraków"),
        Term::iri("http://example.org/a b"),
        Term::iri("http://example.org/<a>"),
        Term::iri(""),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iri_creation() {
        let iri = Iri::new("http://example.org/resource");
        assert_eq!(iri.as_str(), "http://example.org/resource");
        assert_eq!(iri.local_name(), "resource");
        assert_eq!(iri.namespace(), "http://example.org/");
    }

    #[test]
    fn test_iri_with_fragment() {
        let iri = Iri::new("http://xmlns.com/foaf/0.1#Person");
        assert_eq!(iri.local_name(), "Person");
        assert_eq!(iri.namespace(), "http://xmlns.com/foaf/0.1#");
    }

    #[test]
    fn test_blank_node() {
        let bn = BlankNode::new("b0");
        assert_eq!(bn.id(), "b0");
        assert_eq!(bn.to_string(), "_:b0");
    }

    #[test]
    fn test_simple_literal() {
        let lit = Literal::simple("Hello");
        assert_eq!(lit.value(), "Hello");
        assert_eq!(lit.datatype(), Literal::XSD_STRING);
        assert!(lit.is_simple());
        assert!(!lit.is_lang_string());
    }

    #[test]
    fn test_typed_literal() {
        let lit = Literal::integer(42);
        assert_eq!(lit.value(), "42");
        assert_eq!(lit.datatype(), Literal::XSD_INTEGER);
        assert_eq!(lit.as_integer(), Some(42));
    }

    #[test]
    fn test_lang_literal() {
        let lit = Literal::with_language("Bonjour", "fr");
        assert_eq!(lit.value(), "Bonjour");
        assert_eq!(lit.language(), Some("fr"));
        assert!(lit.is_lang_string());
    }

    #[test]
    fn every_term_prints_and_parses_back_as_itself() {
        for term in unusual_terms() {
            let text = term.to_string();
            assert_eq!(Term::from_ntriples(&text), Ok(term.clone()), "{text}");
        }
    }

    #[test]
    fn ntriples_escapes_decode_to_their_characters() {
        let cases = [
            (
                r#""~t~b~n~r~f~"~'~~""#,
                Term::literal("\t\x08\n\r\x0c\"'\\"),
            ),
            (r#""Krak~u00F3w""#, Term::literal("Kraków")),
            (r#""Krak~u00f3w""#, Term::literal("Kraków")),
            (
                r#""~U0001F6B2 naar Berlin""#,
                Term::literal("🚲 naar Berlin"),
            ),
            (r#""Krak~u00F3w"@pl"#, Term::lang_literal("Kraków", "pl")),
            (
                r#""~u0031030"^^<http://www.w3.org/2001/XMLSchema#integer>"#,
                Term::typed_literal("1030", Literal::XSD_INTEGER),
            ),
        ];
        for (text, term) in cases {
            assert_eq!(Term::from_ntriples(&escaped(text)), Ok(term), "{text}");
        }
    }

    #[test]
    fn whitespace_belongs_to_the_term() {
        assert_eq!(Term::from_ntriples("_:b0 "), Ok(Term::blank("b0 ")));
        assert_ne!(Term::from_ntriples("_:b0 "), Term::from_ntriples("_:b0"));
        assert_eq!(
            Term::from_ntriples("\"Praha\"@cs "),
            Ok(Term::lang_literal("Praha", "cs "))
        );
        for text in [
            " <http://example.org/alix>",
            "<http://example.org/alix> ",
            "\"Alix\" ",
            " _:b0",
        ] {
            assert!(Term::from_ntriples(text).is_err(), "{text:?}");
        }
    }

    #[test]
    fn strings_that_are_no_term_are_refused_with_the_reason() {
        let neither = r#"neither an IRI ('<'), a blank node ('_:') nor a literal ('"')"#;
        let after = "text after the literal's closing quote at byte 6, neither a language tag \
                     ('@') nor a datatype ('^^<...>')";
        let four = "the escape at byte 1 needs 4 hexadecimal digits";
        let cases = [
            ("", "an empty string"),
            ("Alix", neither),
            ("_", neither),
            ("<<not a term", "an IRI without its closing '>'"),
            ("<http://example.org/alix", "an IRI without its closing '>'"),
            (r#""Alix"#, "a literal without its closing quote"),
            (r#""Alix~"#, "an escape without its character at byte 5"),
            (r#""Alix"x"#, after),
            (r#""Alix"^^<http://www.w3.org/2001/XMLSchema#string"#, after),
            (r#""~x""#, "an unknown escape '~x' at byte 1"),
            (r#""~u00F""#, four),
            (r#""~u00G3""#, four),
            (r#""~u+0F3""#, four),
            (
                r#""~U0001F6B""#,
                "the escape at byte 1 needs 8 hexadecimal digits",
            ),
            (
                r#""~uD800""#,
                "the escape at byte 1 spells U+D800, which is not a Unicode scalar value",
            ),
            (
                r#""~U00110000""#,
                "the escape at byte 1 spells U+110000, which is not a Unicode scalar value",
            ),
        ];
        for (text, reason) in cases {
            let text = escaped(text);
            let error = Term::from_ntriples(&text).unwrap_err().to_string();
            assert_eq!(
                error,
                format!("not an N-Triples term ({}): {text:?}", escaped(reason)),
                "{text}"
            );
        }
        // A long string is shown by its start only.
        let long = format!("\"{}", "Gus ".repeat(1000));
        let error = Term::from_ntriples(&long).unwrap_err().to_string();
        assert!(
            error.contains("closing quote") && error.len() < 160,
            "{error}"
        );
        assert!(error.ends_with("Gus...\""), "{error}");
    }

    #[test]
    fn test_term_display() {
        assert_eq!(
            Term::iri("http://example.org").to_string(),
            "<http://example.org>"
        );
        assert_eq!(Term::blank("b0").to_string(), "_:b0");
        assert_eq!(Term::literal("Hello").to_string(), "\"Hello\"");
        assert_eq!(
            Term::lang_literal("Bonjour", "fr").to_string(),
            "\"Bonjour\"@fr"
        );
        assert_eq!(
            Term::typed_literal("42", Literal::XSD_INTEGER).to_string(),
            "\"42\"^^<http://www.w3.org/2001/XMLSchema#integer>"
        );
    }
}
