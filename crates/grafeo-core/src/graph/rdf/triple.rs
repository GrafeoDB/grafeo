//! RDF Triples.
//!
//! A triple is the fundamental unit of RDF data, consisting of:
//! - Subject: the resource being described (IRI or blank node)
//! - Predicate: the property or relationship (IRI)
//! - Object: the value (IRI, blank node, or literal)

use super::term::Term;
use grafeo_common::storage::log_record::{TermRecord, TripleRecord};
use serde::{Deserialize, Serialize};
use std::fmt;

/// An RDF triple (subject, predicate, object).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Triple {
    /// The subject (IRI or blank node).
    subject: Term,
    /// The predicate (always an IRI).
    predicate: Term,
    /// The object (IRI, blank node, or literal).
    object: Term,
}

impl Triple {
    /// Creates a new triple.
    ///
    /// # Panics
    ///
    /// Panics if the subject is a literal or the predicate is not an IRI.
    pub fn new(subject: Term, predicate: Term, object: Term) -> Self {
        debug_assert!(
            subject.is_iri() || subject.is_blank_node(),
            "Subject must be an IRI or blank node"
        );
        debug_assert!(predicate.is_iri(), "Predicate must be an IRI");

        Self {
            subject,
            predicate,
            object,
        }
    }

    /// Creates a new triple without validation.
    ///
    /// Use this only when you're sure the terms are valid.
    #[inline]
    pub fn new_unchecked(subject: Term, predicate: Term, object: Term) -> Self {
        Self {
            subject,
            predicate,
            object,
        }
    }

    /// Returns the subject of the triple.
    #[inline]
    #[must_use]
    pub fn subject(&self) -> &Term {
        &self.subject
    }

    /// Returns the predicate of the triple.
    #[inline]
    #[must_use]
    pub fn predicate(&self) -> &Term {
        &self.predicate
    }

    /// Returns the object of the triple.
    #[inline]
    #[must_use]
    pub fn object(&self) -> &Term {
        &self.object
    }

    /// Deconstructs the triple into its components.
    #[inline]
    #[must_use]
    pub fn into_parts(self) -> (Term, Term, Term) {
        (self.subject, self.predicate, self.object)
    }

    /// Returns the triple as a tuple reference.
    #[inline]
    #[must_use]
    pub fn as_tuple(&self) -> (&Term, &Term, &Term) {
        (&self.subject, &self.predicate, &self.object)
    }
}

/// A triple as a log record or a change set stores it.
impl From<&Triple> for TripleRecord {
    fn from(triple: &Triple) -> Self {
        Self {
            subject: TermRecord::from(&triple.subject),
            predicate: TermRecord::from(&triple.predicate),
            object: TermRecord::from(&triple.object),
        }
    }
}

/// The triple a log record or a change set stores. A record holds what a
/// triple held, so it is not checked again.
impl From<&TripleRecord> for Triple {
    fn from(record: &TripleRecord) -> Self {
        Self::new_unchecked(
            Term::from(&record.subject),
            Term::from(&record.predicate),
            Term::from(&record.object),
        )
    }
}

impl fmt::Display for Triple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {} .", self.subject, self.predicate, self.object)
    }
}

/// A triple pattern for matching.
#[derive(Debug, Clone)]
pub struct TriplePattern {
    /// Subject pattern (None = variable).
    pub subject: Option<Term>,
    /// Predicate pattern (None = variable).
    pub predicate: Option<Term>,
    /// Object pattern (None = variable).
    pub object: Option<Term>,
}

impl TriplePattern {
    /// Creates a pattern that matches any triple.
    pub fn any() -> Self {
        Self {
            subject: None,
            predicate: None,
            object: None,
        }
    }

    /// Creates a pattern with a specific subject.
    pub fn with_subject(subject: Term) -> Self {
        Self {
            subject: Some(subject),
            predicate: None,
            object: None,
        }
    }

    /// Creates a pattern with a specific predicate.
    pub fn with_predicate(predicate: Term) -> Self {
        Self {
            subject: None,
            predicate: Some(predicate),
            object: None,
        }
    }

    /// Creates a pattern with a specific object.
    pub fn with_object(object: Term) -> Self {
        Self {
            subject: None,
            predicate: None,
            object: Some(object),
        }
    }

    /// Checks if a triple matches this pattern.
    pub fn matches(&self, triple: &Triple) -> bool {
        if let Some(ref s) = self.subject
            && s != triple.subject()
        {
            return false;
        }
        if let Some(ref p) = self.predicate
            && p != triple.predicate()
        {
            return false;
        }
        if let Some(ref o) = self.object
            && o != triple.object()
        {
            return false;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triple_creation() {
        let triple = Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        );

        assert!(triple.subject().is_iri());
        assert!(triple.predicate().is_iri());
        assert!(triple.object().is_literal());
    }

    #[test]
    fn test_triple_display() {
        let triple = Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        );

        let display = triple.to_string();
        assert!(display.contains("<http://example.org/alix>"));
        assert!(display.contains("<http://xmlns.com/foaf/0.1/name>"));
        assert!(display.contains("\"Alix\""));
        assert!(display.ends_with('.'));
    }

    /// A triple a change set or a log record stores comes back equal, with
    /// every part of its terms: a literal's datatype and language tag, text
    /// that N-Triples escapes, whitespace and empty strings.
    #[test]
    fn a_triple_record_gives_the_triple_back() {
        let knows = Term::iri("http://example.org/knows");
        for term in super::super::term::unusual_terms() {
            let subject = if term.is_literal() {
                Term::blank("b1")
            } else {
                term.clone()
            };
            let triple = Triple::new_unchecked(subject, knows.clone(), term);
            let record = TripleRecord::from(&triple);
            assert_eq!(Triple::from(&record), triple, "{record:?}");
        }
        let tagged = Term::lang_literal("Amsterdam", "nl");
        let TermRecord::Literal {
            datatype, language, ..
        } = TermRecord::from(&tagged)
        else {
            panic!("a literal's record is a literal");
        };
        assert_eq!(
            (datatype.as_str(), language.as_deref()),
            (super::super::term::Literal::RDF_LANG_STRING, Some("nl"))
        );
    }

    #[test]
    fn test_triple_pattern_matching() {
        let triple = Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        );

        // Match any
        assert!(TriplePattern::any().matches(&triple));

        // Match by subject
        assert!(TriplePattern::with_subject(Term::iri("http://example.org/alix")).matches(&triple));
        assert!(!TriplePattern::with_subject(Term::iri("http://example.org/gus")).matches(&triple));

        // Match by predicate
        assert!(
            TriplePattern::with_predicate(Term::iri("http://xmlns.com/foaf/0.1/name"))
                .matches(&triple)
        );
    }
}
