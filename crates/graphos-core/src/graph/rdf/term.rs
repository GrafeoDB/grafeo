//! RDF Terms.
//!
//! RDF terms are the building blocks of triples. There are four types:
//! - IRIs (Internationalized Resource Identifiers)
//! - Blank nodes (anonymous nodes)
//! - Literals (data values)
//! - Variables (for query patterns, not stored)

use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;

/// An RDF term.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
