//! RDF Triples.
//!
//! A triple is the fundamental unit of RDF data, consisting of:
//! - Subject: the resource being described (IRI or blank node)
//! - Predicate: the property or relationship (IRI)
//! - Object: the value (IRI, blank node, or literal)

use super::term::Term;
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

impl fmt::Display for Triple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {} .", self.subject, self.predicate, self.object)
    }
}

/// A quad extends a triple with a graph name.
///
/// Note: This type is provided for future named graph support.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct Quad {
    /// The subject.
    subject: Term,
    /// The predicate.
    predicate: Term,
    /// The object.
    object: Term,
    /// The graph name (IRI or blank node, None = default graph).
    graph: Option<Term>,
}

#[allow(dead_code)]
impl Quad {
    /// Creates a new quad in the default graph.
    pub fn new(subject: Term, predicate: Term, object: Term) -> Self {
        Self {
            subject,
            predicate,
            object,
            graph: None,
        }
    }

    /// Creates a new quad in a named graph.
    pub fn in_graph(subject: Term, predicate: Term, object: Term, graph: Term) -> Self {
        debug_assert!(
            graph.is_iri() || graph.is_blank_node(),
            "Graph name must be an IRI or blank node"
        );
        Self {
            subject,
            predicate,
            object,
            graph: Some(graph),
        }
    }

    /// Returns the subject.
    #[inline]
    #[must_use]
    pub fn subject(&self) -> &Term {
        &self.subject
    }

    /// Returns the predicate.
    #[inline]
    #[must_use]
    pub fn predicate(&self) -> &Term {
        &self.predicate
    }

    /// Returns the object.
    #[inline]
    #[must_use]
    pub fn object(&self) -> &Term {
        &self.object
    }

    /// Returns the graph name (None for default graph).
    #[inline]
    #[must_use]
    pub fn graph(&self) -> Option<&Term> {
        self.graph.as_ref()
    }

    /// Returns true if this quad is in the default graph.
    #[inline]
    #[must_use]
    pub fn is_default_graph(&self) -> bool {
        self.graph.is_none()
    }

    /// Converts this quad to a triple (discarding the graph name).
    #[inline]
    #[must_use]
    pub fn into_triple(self) -> Triple {
        Triple::new_unchecked(self.subject, self.predicate, self.object)
    }
}

impl fmt::Display for Quad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {}", self.subject, self.predicate, self.object)?;
        if let Some(ref graph) = self.graph {
            write!(f, " {}", graph)?;
        }
        write!(f, " .")
    }
}

impl From<Triple> for Quad {
    fn from(triple: Triple) -> Self {
        Quad::new(triple.subject, triple.predicate, triple.object)
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
            Term::iri("http://example.org/alice"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alice"),
        );

        assert!(triple.subject().is_iri());
        assert!(triple.predicate().is_iri());
        assert!(triple.object().is_literal());
    }

    #[test]
    fn test_triple_display() {
        let triple = Triple::new(
            Term::iri("http://example.org/alice"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alice"),
        );

        let display = triple.to_string();
        assert!(display.contains("<http://example.org/alice>"));
        assert!(display.contains("<http://xmlns.com/foaf/0.1/name>"));
        assert!(display.contains("\"Alice\""));
        assert!(display.ends_with('.'));
    }

    #[test]
    fn test_quad_default_graph() {
        let quad = Quad::new(
            Term::iri("http://example.org/alice"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alice"),
        );

        assert!(quad.is_default_graph());
        assert!(quad.graph().is_none());
    }

    #[test]
    fn test_quad_named_graph() {
        let quad = Quad::in_graph(
            Term::iri("http://example.org/alice"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alice"),
            Term::iri("http://example.org/graph1"),
        );

        assert!(!quad.is_default_graph());
        assert!(quad.graph().is_some());
    }

    #[test]
    fn test_triple_pattern_matching() {
        let triple = Triple::new(
            Term::iri("http://example.org/alice"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alice"),
        );

        // Match any
        assert!(TriplePattern::any().matches(&triple));

        // Match by subject
        assert!(
            TriplePattern::with_subject(Term::iri("http://example.org/alice")).matches(&triple)
        );
        assert!(!TriplePattern::with_subject(Term::iri("http://example.org/bob")).matches(&triple));

        // Match by predicate
        assert!(
            TriplePattern::with_predicate(Term::iri("http://xmlns.com/foaf/0.1/name"))
                .matches(&triple)
        );
    }
}
