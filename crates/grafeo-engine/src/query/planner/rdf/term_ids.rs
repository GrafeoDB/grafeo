//! The term ids one SPARQL query reads: the dataset's, and the query's own.
//!
//! Every variable column of a SPARQL plan holds term ids
//! ([`LogicalType::RdfTermId`](grafeo_common::types::LogicalType::RdfTermId)),
//! the 64 bits of a [`TermId`] in an `Int64`, so joins, DISTINCT and grouping
//! compare integers, and a term is decoded only where its content matters:
//! an expression, ORDER BY, an aggregate, a template, and the result. Two ids
//! are equal exactly when their terms are the same term, which is SPARQL's
//! solution compatibility.
//!
//! A scan emits the ids the dataset's dictionary gave the stored triples. A
//! term the query makes (a BIND or aggregate result, a VALUES row, a CONSTRUCT
//! blank node, a pending write) takes the dataset's id when it has one, its
//! inlined id when it is a small literal, and otherwise an id of the query's
//! own (tag 15), which never leaves the query.

use std::sync::Arc;

use grafeo_common::types::rdf::Term;
use grafeo_common::types::{TermId, TermIdKind, Value};
use grafeo_core::execution::operators::OperatorError;
use grafeo_core::graph::rdf::{DictionaryEntry, TermDictionary};

/// One term as a query reads it: the term, and the value an expression
/// computes with ([`Literal::native_value`](grafeo_common::types::rdf::Literal::native_value)),
/// borrowed from the dictionary that holds it, or made from an inlined id.
#[derive(Debug)]
pub(crate) enum TermEntry<'a> {
    /// A term of the dataset's dictionary or of the query's own.
    Stored(&'a DictionaryEntry),
    /// A literal an id inlines: its term and value made when first read.
    Inline {
        id: TermId,
        term: std::cell::OnceCell<Arc<Term>>,
        native: std::cell::OnceCell<Option<Value>>,
    },
}

impl TermEntry<'_> {
    /// The term.
    pub(crate) fn term(&self) -> &Arc<Term> {
        match self {
            Self::Stored(entry) => entry.term(),
            Self::Inline { id, term, .. } => term
                .get_or_init(|| Arc::new(id.inline_term().expect("an inlined id holds its term"))),
        }
    }

    /// The value of a literal the evaluator computes with; `None` for an
    /// IRI, a blank node, a tagged literal, a literal of an unknown datatype
    /// or one whose text is not of its type.
    pub(crate) fn native(&self) -> Option<&Value> {
        match self {
            Self::Stored(entry) => entry.native(),
            Self::Inline { id, native, .. } => native.get_or_init(|| id.inline_value()).as_ref(),
        }
    }
}

/// The value a column holds for `id`.
#[allow(
    clippy::cast_possible_wrap,
    reason = "a column holds the id's 64 bits in an i64, read back by id_of"
)]
pub(crate) const fn id_value(id: TermId) -> Value {
    Value::Int64(id.raw() as i64)
}

/// The id a column value holds; `None` for NULL (unbound) and any other
/// value.
#[allow(
    clippy::cast_sign_loss,
    reason = "a column holds the id's 64 bits in an i64, as id_value writes them"
)]
pub(crate) fn id_of(value: &Value) -> Option<TermId> {
    match value {
        Value::Int64(raw) => TermId::from_raw(*raw as u64),
        _ => None,
    }
}

/// The term ids of one SPARQL query (see the module docs).
pub(crate) struct QueryTermIds {
    dataset: Arc<TermDictionary>,
    /// The query's own terms, made the first time it has one.
    own: std::sync::OnceLock<TermDictionary>,
}

impl Default for QueryTermIds {
    /// A query over an empty dataset.
    fn default() -> Self {
        Self::new(Arc::new(TermDictionary::new()))
    }
}

impl QueryTermIds {
    /// The ids of a query over the dataset of `dataset`.
    pub(crate) fn new(dataset: Arc<TermDictionary>) -> Self {
        Self {
            dataset,
            own: std::sync::OnceLock::new(),
        }
    }

    /// The id of `term`: the dataset's (or its inlined id), or one of the
    /// query's own, added when it is new.
    ///
    /// # Errors
    ///
    /// For a kind of term the SPARQL engine does not handle, and when the
    /// query has made `2^60` terms of its own.
    pub(crate) fn intern(&self, term: &Term) -> Result<TermId, OperatorError> {
        super::terms::supported(term)?;
        if let Some(id) = self.dataset.lookup(term) {
            return Ok(id);
        }
        let own = self
            .own
            .get_or_init(TermDictionary::new)
            .intern(term)
            .map_err(|error| OperatorError::Execution(error.to_string()))?;
        own.index().and_then(TermId::query).ok_or_else(|| {
            OperatorError::Execution("an inlined term reached the query's own terms".into())
        })
    }

    /// The ids of `terms`, in order.
    ///
    /// # Errors
    ///
    /// As [`intern`](Self::intern).
    pub(crate) fn intern_all<'t>(
        &self,
        terms: impl IntoIterator<Item = &'t Term>,
    ) -> Result<Vec<TermId>, OperatorError> {
        terms.into_iter().map(|term| self.intern(term)).collect()
    }

    /// The term of `id`, or `None` for an id neither the dataset nor this
    /// query gave.
    pub(crate) fn entry(&self, id: TermId) -> Option<TermEntry<'_>> {
        match id.kind() {
            TermIdKind::Dictionary => self.dataset.get(id.index()?).map(TermEntry::Stored),
            TermIdKind::Query => self.own.get()?.get(id.index()?).map(TermEntry::Stored),
            _ => Some(TermEntry::Inline {
                id,
                term: std::cell::OnceCell::new(),
                native: std::cell::OnceCell::new(),
            }),
        }
    }

    /// The term an id column holds at a row's value: `None` for NULL
    /// (unbound) and for a value that is no id.
    pub(crate) fn entry_of(&self, value: &Value) -> Option<TermEntry<'_>> {
        self.entry(id_of(value)?)
    }

    /// How many terms the query made of its own.
    #[cfg(test)]
    pub(crate) fn len(&self) -> u64 {
        self.own.get().map_or(0, TermDictionary::len)
    }
}

impl std::fmt::Debug for QueryTermIds {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryTermIds")
            .field("dataset", &self.dataset)
            .field("own", &self.own.get().map_or(0, TermDictionary::len))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::types::rdf::Literal;

    #[test]
    fn the_same_term_gets_the_same_id_and_different_terms_different_ones() {
        let ids = QueryTermIds::default();
        let english = Term::lang_literal("Fruit", "en");
        let french = Term::lang_literal("Fruit", "fr");
        let a = ids.intern(&english).unwrap();
        assert_eq!(ids.intern(&english).unwrap(), a);
        let b = ids.intern(&french).unwrap();
        assert_ne!(a, b);
        let all = ids
            .intern_all([
                &Term::literal("1"),
                &Term::typed_literal("1", Literal::XSD_INTEGER),
                &english,
                &Term::iri("http://example.org/1"),
                &Term::blank("1"),
            ])
            .unwrap();
        assert_eq!(all[2], a);
        let distinct: std::collections::HashSet<TermId> = all.iter().copied().collect();
        assert_eq!(distinct.len(), 5, "five different terms: {all:?}");
        assert_eq!(ids.len(), 4, "the two inlined literals take no entry");
        assert_eq!(all[1].kind(), TermIdKind::Integer);
    }

    /// A stored term takes the dataset's id; a term the query makes, an id
    /// of its own, which decodes back.
    #[test]
    fn the_dataset_ids_come_first() {
        let dataset = Arc::new(TermDictionary::new());
        let alix = Term::iri("http://example.org/alix");
        let stored = dataset.intern(&alix).unwrap();
        let ids = QueryTermIds::new(Arc::clone(&dataset));
        assert_eq!(ids.intern(&alix).unwrap(), stored);
        let made = ids.intern(&Term::blank("g01")).unwrap();
        assert_eq!(made.kind(), TermIdKind::Query);
        assert_eq!(
            ids.entry(made)
                .map(|entry| Arc::clone(entry.term()))
                .as_deref(),
            Some(&Term::blank("g01"))
        );
        assert_eq!(dataset.len(), 1, "the query's own terms stay its own");
    }

    #[test]
    fn an_id_decodes_to_its_term_and_its_value() {
        let ids = QueryTermIds::default();
        for (text, native) in [("019", 19), ("19", 19)] {
            let term = Term::typed_literal(text, Literal::XSD_INTEGER);
            let id = ids.intern(&term).unwrap();
            let entry = ids.entry_of(&id_value(id)).unwrap();
            assert_eq!(entry.term().as_ref(), &term);
            assert_eq!(entry.native(), Some(&Value::Int64(native)));
        }
        assert!(ids.entry_of(&Value::Null).is_none());
        assert!(
            ids.entry_of(&id_value(TermId::dictionary(7).unwrap()))
                .is_none()
        );
        assert!(
            ids.entry_of(&Value::Int64(-1)).is_none(),
            "tag 15, index 2^60 - 1"
        );
    }
}
