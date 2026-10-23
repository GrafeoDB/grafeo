//! The term dictionary of an RDF dataset: one id per term, for the default
//! graph and every named graph.
//!
//! A literal the id scheme inlines ([`TermId::inline`]) needs no entry; every
//! other term gets the next index the first time a write brings it, and keeps
//! it while the dataset is open. The dictionary only grows: a term no triple
//! uses any more keeps its entry until the next checkpoint writes the terms
//! the triples use and a load reads them back.
//!
//! Reads do not lock: the entries live in segments of doubling size, each
//! allocated once and never moved, and an entry is set once. Writes (the
//! commit path, a load, an import) take one lock to find or add a term.

use std::hash::BuildHasher;
use std::sync::{Arc, OnceLock};

use grafeo_common::types::{TermId, Value};
use grafeo_common::utils::hash::FxBuildHasher;
use hashbrown::HashTable;
use parking_lot::RwLock;

use super::Term;

/// The entries of the first segment; segment `k` holds `FIRST << k`.
const FIRST: usize = 1024;
/// Enough segments for every index an id can hold.
const SEGMENTS: usize = 51;

/// A term of the dictionary, with the value an expression computes with,
/// parsed once while the dataset is open.
#[derive(Debug)]
pub struct DictionaryEntry {
    term: Arc<Term>,
    native: OnceLock<Option<Value>>,
}

impl DictionaryEntry {
    /// The term.
    #[must_use]
    pub fn term(&self) -> &Arc<Term> {
        &self.term
    }

    /// The value of a literal as an expression computes with it
    /// ([`Literal::native_value`](grafeo_common::types::rdf::Literal::native_value));
    /// `None` for an IRI, a blank node, a tagged literal, or a literal of
    /// another datatype or whose text is not of its type.
    #[must_use]
    pub fn native(&self) -> Option<&Value> {
        self.native
            .get_or_init(|| self.term.as_literal().and_then(|l| l.native_value()))
            .as_ref()
    }
}

/// The index from term to dictionary index.
#[derive(Default)]
struct Index {
    hasher: FxBuildHasher,
    indexes: HashTable<u64>,
}

/// The term dictionary of an RDF dataset (see the module docs).
pub struct TermDictionary {
    segments: Box<[OnceLock<Box<[OnceLock<DictionaryEntry>]>>]>,
    len: std::sync::atomic::AtomicU64,
    index: RwLock<Index>,
}

impl Default for TermDictionary {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for TermDictionary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TermDictionary")
            .field("terms", &self.len())
            .finish()
    }
}

/// The segment and the slot in it of `index`.
fn slot_of(index: u64) -> (usize, usize) {
    // Segment k starts at FIRST * (2^k - 1).
    let scaled = index / FIRST as u64 + 1;
    let segment = (u64::BITS - 1 - scaled.leading_zeros()) as usize;
    let start = FIRST as u64 * ((1u64 << segment) - 1);
    (
        segment,
        usize::try_from(index - start).unwrap_or(usize::MAX),
    )
}

impl TermDictionary {
    /// An empty dictionary.
    #[must_use]
    pub fn new() -> Self {
        Self {
            segments: (0..SEGMENTS).map(|_| OnceLock::new()).collect(),
            len: std::sync::atomic::AtomicU64::new(0),
            index: RwLock::new(Index::default()),
        }
    }

    /// How many terms have an index.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.len.load(std::sync::atomic::Ordering::Acquire)
    }

    /// An estimate of the heap the dictionary holds beyond the terms'
    /// strings (which the triples share): per entry its term, the slot and
    /// the writers' index.
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        let per_entry = std::mem::size_of::<Term>()
            + 2 * std::mem::size_of::<usize>()
            + std::mem::size_of::<OnceLock<DictionaryEntry>>()
            + std::mem::size_of::<u64>()
            + 1;
        usize::try_from(self.len())
            .unwrap_or(usize::MAX)
            .saturating_mul(per_entry)
    }

    /// Whether no term has an index.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The entry at a dictionary index, without a lock; `None` for an index
    /// no term has.
    #[must_use]
    pub fn get(&self, index: u64) -> Option<&DictionaryEntry> {
        let (segment, slot) = slot_of(index);
        self.segments.get(segment)?.get()?.get(slot)?.get()
    }

    /// The term of `id`: the inlined term, or the dictionary's; `None` for a
    /// query's own id or an index no term has.
    #[must_use]
    pub fn term(&self, id: TermId) -> Option<Arc<Term>> {
        match id.kind() {
            grafeo_common::types::TermIdKind::Dictionary => {
                self.get(id.index()?).map(|entry| Arc::clone(entry.term()))
            }
            grafeo_common::types::TermIdKind::Query => None,
            _ => id.inline_term().map(Arc::new),
        }
    }

    /// The id `term` has, without adding it: the inlined id, or its index
    /// when a write gave it one.
    #[must_use]
    pub fn lookup(&self, term: &Term) -> Option<TermId> {
        if let Some(id) = TermId::inline(term) {
            return Some(id);
        }
        let index = self.index.read();
        let hash = index.hasher.hash_one(term);
        self.find(&index, term, hash).and_then(TermId::dictionary)
    }

    /// The id of `term`, adding it when it is new.
    ///
    /// # Errors
    ///
    /// When the dictionary already holds [`TermId::MAX_INDEX`] terms.
    pub fn intern(&self, term: &Term) -> Result<TermId, DictionaryFull> {
        if let Some(id) = TermId::inline(term) {
            return Ok(id);
        }
        let mut index = self.index.write();
        self.add(&mut index, term)
    }

    /// The ids of `terms` in order, under one lock.
    ///
    /// # Errors
    ///
    /// As [`intern`](Self::intern).
    pub fn intern_all<'t>(
        &self,
        terms: impl IntoIterator<Item = &'t Term>,
    ) -> Result<Vec<TermId>, DictionaryFull> {
        let mut index = self.index.write();
        terms
            .into_iter()
            .map(|term| match TermId::inline(term) {
                Some(id) => Ok(id),
                None => self.add(&mut index, term),
            })
            .collect()
    }

    fn find(&self, index: &Index, term: &Term, hash: u64) -> Option<u64> {
        index
            .indexes
            .find(hash, |&at| {
                self.get(at).is_some_and(|known| **known.term() == *term)
            })
            .copied()
    }

    fn add(&self, index: &mut Index, term: &Term) -> Result<TermId, DictionaryFull> {
        let hash = index.hasher.hash_one(term);
        if let Some(at) = self.find(index, term, hash) {
            return TermId::dictionary(at).ok_or(DictionaryFull);
        }
        let at = self.len();
        let id = TermId::dictionary(at).ok_or(DictionaryFull)?;
        let (segment, slot) = slot_of(at);
        let entries = self
            .segments
            .get(segment)
            .ok_or(DictionaryFull)?
            .get_or_init(|| (0..FIRST << segment).map(|_| OnceLock::new()).collect());
        let _ = entries[slot].set(DictionaryEntry {
            term: Arc::new(term.clone()),
            native: OnceLock::new(),
        });
        self.len.store(at + 1, std::sync::atomic::Ordering::Release);
        let Index { hasher, indexes } = index;
        indexes.insert_unique(hash, at, |&known| {
            self.get(known)
                .map_or(0, |entry| hasher.hash_one(&**entry.term()))
        });
        Ok(id)
    }
}

/// The dictionary holds every index an id can hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("the RDF dataset holds more than {} distinct terms", TermId::MAX_INDEX)]
pub struct DictionaryFull;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_term_gets_one_id_and_reads_back() {
        let dictionary = TermDictionary::new();
        let alix = Term::iri("http://example.org/alix");
        let label = Term::lang_literal("Alix", "nl");
        let id = dictionary.intern(&alix).unwrap();
        assert_eq!(id, TermId::dictionary(0).unwrap());
        assert_eq!(
            dictionary.intern(&label).unwrap(),
            TermId::dictionary(1).unwrap()
        );
        assert_eq!(
            dictionary.intern(&alix).unwrap(),
            id,
            "the same term, the same id"
        );
        assert_eq!(dictionary.lookup(&alix), Some(id));
        assert_eq!(dictionary.term(id).as_deref(), Some(&alix));
        assert_eq!(
            dictionary.lookup(&Term::iri("http://example.org/gus")),
            None
        );
        assert_eq!(dictionary.len(), 2);
    }

    /// An inlined literal takes no entry.
    #[test]
    fn an_inlined_literal_takes_no_entry() {
        let dictionary = TermDictionary::new();
        let one = Term::typed_literal("1", "http://www.w3.org/2001/XMLSchema#integer");
        let id = dictionary.intern(&one).unwrap();
        assert_eq!(id, TermId::inline(&one).unwrap());
        assert_eq!(dictionary.term(id).as_deref(), Some(&one));
        assert!(dictionary.is_empty());
        let zero_one = Term::typed_literal("01", "http://www.w3.org/2001/XMLSchema#integer");
        assert_eq!(
            dictionary.intern(&zero_one).unwrap(),
            TermId::dictionary(0).unwrap()
        );
    }

    /// Indexes cross segment boundaries in order, and every term reads back.
    #[test]
    fn indexes_cross_segments() {
        let dictionary = TermDictionary::new();
        let terms: Vec<Term> = (0..FIRST * 7 + 3)
            .map(|i| Term::iri(format!("http://example.org/{i}")))
            .collect();
        let ids = dictionary.intern_all(&terms).unwrap();
        for (i, (term, id)) in terms.iter().zip(&ids).enumerate() {
            assert_eq!(id.index(), Some(i as u64));
            assert_eq!(dictionary.get(i as u64).map(|e| &**e.term()), Some(term));
        }
        assert_eq!(slot_of(0), (0, 0));
        assert_eq!(slot_of(FIRST as u64 - 1), (0, FIRST - 1));
        assert_eq!(slot_of(FIRST as u64), (1, 0));
        assert_eq!(slot_of(3 * FIRST as u64), (2, 0));
        assert!(dictionary.get(terms.len() as u64).is_none());
    }

    /// Readers on other threads see every term a writer added.
    #[test]
    fn readers_see_what_writers_added() {
        let dictionary = Arc::new(TermDictionary::new());
        let writer = {
            let dictionary = Arc::clone(&dictionary);
            std::thread::spawn(move || {
                for i in 0..20_000 {
                    dictionary
                        .intern(&Term::iri(format!("http://example.org/{i}")))
                        .unwrap();
                }
            })
        };
        let mut seen = 0;
        while seen < 20_000 {
            let len = dictionary.len();
            for index in seen..len {
                let entry = dictionary.get(index).expect("an index below len is set");
                assert_eq!(
                    **entry.term(),
                    Term::iri(format!("http://example.org/{index}"))
                );
            }
            seen = len;
        }
        writer.join().unwrap();
    }
}
