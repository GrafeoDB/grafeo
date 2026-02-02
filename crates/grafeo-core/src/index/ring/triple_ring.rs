//! Triple Ring - compact RDF triple index.
//!
//! The TripleRing stores RDF triples in a compact representation using
//! wavelet trees and succinct permutations, achieving ~3x space reduction
//! compared to hash-based triple indexing.

use super::permutation::SuccinctPermutation;
use crate::graph::rdf::{Term, Triple, TriplePattern};
use crate::storage::succinct::WaveletTree;
use hashbrown::HashMap;
use std::sync::Arc;

/// Term dictionary mapping terms to compact integer IDs.
#[derive(Debug, Clone, Default)]
pub struct TermDictionary {
    /// Term to ID mapping.
    term_to_id: HashMap<Arc<Term>, u32, ahash::RandomState>,
    /// ID to term mapping.
    id_to_term: Vec<Arc<Term>>,
}

impl TermDictionary {
    /// Creates a new empty term dictionary.
    #[must_use]
    pub fn new() -> Self {
        Self {
            term_to_id: HashMap::with_hasher(ahash::RandomState::new()),
            id_to_term: Vec::new(),
        }
    }

    /// Creates a term dictionary with specified capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            term_to_id: HashMap::with_capacity_and_hasher(capacity, ahash::RandomState::new()),
            id_to_term: Vec::with_capacity(capacity),
        }
    }

    /// Returns the number of terms.
    #[must_use]
    pub fn len(&self) -> usize {
        self.id_to_term.len()
    }

    /// Returns whether the dictionary is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.id_to_term.is_empty()
    }

    /// Gets or inserts a term, returning its ID.
    pub fn get_or_insert(&mut self, term: Term) -> u32 {
        let term = Arc::new(term);
        if let Some(&id) = self.term_to_id.get(&term) {
            return id;
        }

        let id = self.id_to_term.len() as u32;
        self.id_to_term.push(Arc::clone(&term));
        self.term_to_id.insert(term, id);
        id
    }

    /// Looks up a term by ID.
    #[must_use]
    pub fn get_term(&self, id: u32) -> Option<&Term> {
        self.id_to_term.get(id as usize).map(Arc::as_ref)
    }

    /// Looks up an ID by term.
    #[must_use]
    pub fn get_id(&self, term: &Term) -> Option<u32> {
        self.term_to_id.get(term).copied()
    }

    /// Returns size in bytes.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        let base = std::mem::size_of::<Self>();
        let terms: usize = self
            .id_to_term
            .iter()
            .map(|t| std::mem::size_of_val(t.as_ref()) + std::mem::size_of::<Arc<Term>>())
            .sum();
        let map_overhead = self.term_to_id.capacity()
            * (std::mem::size_of::<Arc<Term>>() + std::mem::size_of::<u32>());
        base + terms + map_overhead
    }
}

/// Compact triple representation using term IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CompactTriple {
    subject: u32,
    predicate: u32,
    object: u32,
}

/// The Ring Index for RDF triples.
///
/// Stores triples compactly using:
/// - Term dictionary for string → ID mapping
/// - Wavelet trees for each triple component
/// - Succinct permutations for navigating between orderings
#[derive(Debug)]
pub struct TripleRing {
    /// Term dictionary.
    dict: TermDictionary,

    /// Number of triples.
    num_triples: usize,

    /// Subjects in SPO order (wavelet tree over subject IDs).
    subjects: WaveletTree,

    /// Predicates in SPO order.
    predicates: WaveletTree,

    /// Objects in SPO order.
    objects: WaveletTree,

    /// Permutation from SPO position to POS position.
    spo_to_pos: SuccinctPermutation,

    /// Permutation from SPO position to OSP position.
    spo_to_osp: SuccinctPermutation,
}

impl TripleRing {
    /// Creates a Ring Index from an iterator of triples.
    ///
    /// # Arguments
    ///
    /// * `triples` - Iterator over RDF triples
    #[must_use]
    pub fn from_triples(triples: impl Iterator<Item = Triple>) -> Self {
        // Collect all triples and build dictionary
        let mut dict = TermDictionary::new();
        let mut compact_triples: Vec<CompactTriple> = Vec::new();

        for triple in triples {
            let (s, p, o) = triple.into_parts();
            let compact = CompactTriple {
                subject: dict.get_or_insert(s),
                predicate: dict.get_or_insert(p),
                object: dict.get_or_insert(o),
            };
            compact_triples.push(compact);
        }

        if compact_triples.is_empty() {
            return Self {
                dict,
                num_triples: 0,
                subjects: WaveletTree::new(&[]),
                predicates: WaveletTree::new(&[]),
                objects: WaveletTree::new(&[]),
                spo_to_pos: SuccinctPermutation::default(),
                spo_to_osp: SuccinctPermutation::default(),
            };
        }

        // Sort by SPO (primary order)
        compact_triples.sort_by_key(|t| (t.subject, t.predicate, t.object));

        // Remove duplicates
        compact_triples.dedup();
        let n = compact_triples.len();

        // Build sequences for wavelet trees
        let subjects: Vec<u64> = compact_triples.iter().map(|t| t.subject as u64).collect();
        let predicates: Vec<u64> = compact_triples.iter().map(|t| t.predicate as u64).collect();
        let objects: Vec<u64> = compact_triples.iter().map(|t| t.object as u64).collect();

        // Build wavelet trees
        let subjects_wt = WaveletTree::new(&subjects);
        let predicates_wt = WaveletTree::new(&predicates);
        let objects_wt = WaveletTree::new(&objects);

        // Build permutations to POS and OSP orderings

        // For SPO → POS: sort by (predicate, object, subject)
        let mut pos_order: Vec<usize> = (0..n).collect();
        pos_order.sort_by_key(|&i| {
            let t = &compact_triples[i];
            (t.predicate, t.object, t.subject)
        });

        // spo_to_pos[spo_idx] = pos_idx means: triple at SPO position spo_idx
        // is at POS position pos_idx
        let mut spo_to_pos_arr = vec![0usize; n];
        for (pos_idx, &spo_idx) in pos_order.iter().enumerate() {
            spo_to_pos_arr[spo_idx] = pos_idx;
        }

        // For SPO → OSP: sort by (object, subject, predicate)
        let mut osp_order: Vec<usize> = (0..n).collect();
        osp_order.sort_by_key(|&i| {
            let t = &compact_triples[i];
            (t.object, t.subject, t.predicate)
        });

        let mut spo_to_osp_arr = vec![0usize; n];
        for (osp_idx, &spo_idx) in osp_order.iter().enumerate() {
            spo_to_osp_arr[spo_idx] = osp_idx;
        }

        Self {
            dict,
            num_triples: n,
            subjects: subjects_wt,
            predicates: predicates_wt,
            objects: objects_wt,
            spo_to_pos: SuccinctPermutation::new(&spo_to_pos_arr),
            spo_to_osp: SuccinctPermutation::new(&spo_to_osp_arr),
        }
    }

    /// Returns the number of triples.
    #[must_use]
    pub fn len(&self) -> usize {
        self.num_triples
    }

    /// Returns whether the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.num_triples == 0
    }

    /// Returns the number of distinct terms.
    #[must_use]
    pub fn num_terms(&self) -> usize {
        self.dict.len()
    }

    /// Returns the triple at position i in SPO order.
    #[must_use]
    pub fn get_spo(&self, index: usize) -> Option<Triple> {
        if index >= self.num_triples {
            return None;
        }

        let s_id = self.subjects.access(index) as u32;
        let p_id = self.predicates.access(index) as u32;
        let o_id = self.objects.access(index) as u32;

        let s = self.dict.get_term(s_id)?.clone();
        let p = self.dict.get_term(p_id)?.clone();
        let o = self.dict.get_term(o_id)?.clone();

        Some(Triple::new_unchecked(s, p, o))
    }

    /// Returns the subjects wavelet tree.
    #[must_use]
    pub fn subjects_wt(&self) -> &WaveletTree {
        &self.subjects
    }

    /// Returns the predicates wavelet tree.
    #[must_use]
    pub fn predicates_wt(&self) -> &WaveletTree {
        &self.predicates
    }

    /// Returns the objects wavelet tree.
    #[must_use]
    pub fn objects_wt(&self) -> &WaveletTree {
        &self.objects
    }

    /// Returns the position in SPO order for a given POS position.
    #[must_use]
    pub fn pos_to_spo(&self, pos_index: usize) -> Option<usize> {
        self.spo_to_pos.apply_inverse(pos_index)
    }

    /// Returns the position in SPO order for a given OSP position.
    #[must_use]
    pub fn osp_to_spo(&self, osp_index: usize) -> Option<usize> {
        self.spo_to_osp.apply_inverse(osp_index)
    }

    /// Returns the position in POS order for a given SPO position.
    #[must_use]
    pub fn spo_to_pos(&self, spo_index: usize) -> Option<usize> {
        self.spo_to_pos.apply(spo_index)
    }

    /// Returns the position in OSP order for a given SPO position.
    #[must_use]
    pub fn spo_to_osp(&self, spo_index: usize) -> Option<usize> {
        self.spo_to_osp.apply(spo_index)
    }

    /// Returns an iterator over all triples matching a pattern.
    pub fn find<'a>(&'a self, pattern: &'a TriplePattern) -> impl Iterator<Item = Triple> + 'a {
        RingPatternIterator {
            ring: self,
            pattern,
            current: 0,
        }
    }

    /// Returns the count of triples matching a pattern.
    ///
    /// Uses wavelet tree rank operations for efficient counting.
    #[must_use]
    pub fn count(&self, pattern: &TriplePattern) -> usize {
        // If all components are bound, check for exact match
        if let (Some(s), Some(p), Some(o)) = (&pattern.subject, &pattern.predicate, &pattern.object)
        {
            // Get IDs
            let s_id = match self.dict.get_id(s) {
                Some(id) => id,
                None => return 0,
            };
            let p_id = match self.dict.get_id(p) {
                Some(id) => id,
                None => return 0,
            };
            let o_id = match self.dict.get_id(o) {
                Some(id) => id,
                None => return 0,
            };

            // Check if this exact triple exists
            return usize::from(self.contains_ids(s_id, p_id, o_id));
        }

        // For partial patterns, use wavelet tree counting
        match (&pattern.subject, &pattern.predicate, &pattern.object) {
            (Some(s), None, None) => {
                // Count triples with this subject
                if let Some(s_id) = self.dict.get_id(s) {
                    self.subjects.count(s_id as u64)
                } else {
                    0
                }
            }
            (None, Some(p), None) => {
                // Count triples with this predicate
                if let Some(p_id) = self.dict.get_id(p) {
                    self.predicates.count(p_id as u64)
                } else {
                    0
                }
            }
            (None, None, Some(o)) => {
                // Count triples with this object
                if let Some(o_id) = self.dict.get_id(o) {
                    self.objects.count(o_id as u64)
                } else {
                    0
                }
            }
            (None, None, None) => self.num_triples,
            _ => {
                // For other patterns, fall back to iteration
                self.find(pattern).count()
            }
        }
    }

    /// Checks if a triple with the given IDs exists.
    fn contains_ids(&self, s_id: u32, p_id: u32, o_id: u32) -> bool {
        // Find positions where subject matches
        let s_count = self.subjects.count(s_id as u64);
        if s_count == 0 {
            return false;
        }

        // Check each position with matching subject
        for rank in 0..s_count {
            if let Some(pos) = self.subjects.select(s_id as u64, rank) {
                // Check if predicate and object also match at this position
                let p = self.predicates.access(pos);
                let o = self.objects.access(pos);
                if p as u32 == p_id && o as u32 == o_id {
                    return true;
                }
            }
        }

        false
    }

    /// Returns the term dictionary.
    #[must_use]
    pub fn dictionary(&self) -> &TermDictionary {
        &self.dict
    }

    /// Returns size in bytes.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        let base = std::mem::size_of::<Self>();
        let dict = self.dict.size_bytes();
        let subjects = self.subjects.size_bytes();
        let predicates = self.predicates.size_bytes();
        let objects = self.objects.size_bytes();
        let spo_to_pos = self.spo_to_pos.size_bytes();
        let spo_to_osp = self.spo_to_osp.size_bytes();

        base + dict + subjects + predicates + objects + spo_to_pos + spo_to_osp
    }
}

/// Iterator over triples matching a pattern.
struct RingPatternIterator<'a> {
    ring: &'a TripleRing,
    pattern: &'a TriplePattern,
    current: usize,
}

impl Iterator for RingPatternIterator<'_> {
    type Item = Triple;

    fn next(&mut self) -> Option<Self::Item> {
        while self.current < self.ring.num_triples {
            let idx = self.current;
            self.current += 1;

            if let Some(triple) = self.ring.get_spo(idx) {
                if self.pattern.matches(&triple) {
                    return Some(triple);
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_triple(s: &str, p: &str, o: &str) -> Triple {
        Triple::new(Term::iri(s), Term::iri(p), Term::iri(o))
    }

    #[test]
    fn test_empty() {
        let ring = TripleRing::from_triples(std::iter::empty());
        assert!(ring.is_empty());
        assert_eq!(ring.len(), 0);
        assert_eq!(ring.num_terms(), 0);
    }

    #[test]
    fn test_single_triple() {
        let triples = vec![make_triple("s1", "p1", "o1")];
        let ring = TripleRing::from_triples(triples.into_iter());

        assert_eq!(ring.len(), 1);
        assert_eq!(ring.num_terms(), 3);

        let retrieved = ring.get_spo(0).unwrap();
        assert_eq!(retrieved.subject(), &Term::iri("s1"));
        assert_eq!(retrieved.predicate(), &Term::iri("p1"));
        assert_eq!(retrieved.object(), &Term::iri("o1"));
    }

    #[test]
    fn test_multiple_triples() {
        let triples = vec![
            make_triple("s1", "p1", "o1"),
            make_triple("s1", "p2", "o2"),
            make_triple("s2", "p1", "o1"),
            make_triple("s2", "p1", "o3"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        assert_eq!(ring.len(), 4);
        // Terms: s1, s2, p1, p2, o1, o2, o3 = 7
        assert_eq!(ring.num_terms(), 7);
    }

    #[test]
    fn test_deduplication() {
        let triples = vec![
            make_triple("s1", "p1", "o1"),
            make_triple("s1", "p1", "o1"), // duplicate
            make_triple("s2", "p1", "o1"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        // Should have 2 unique triples
        assert_eq!(ring.len(), 2);
    }

    #[test]
    fn test_find_by_subject() {
        let triples = vec![
            make_triple("alice", "knows", "bob"),
            make_triple("alice", "knows", "carol"),
            make_triple("bob", "knows", "carol"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        let pattern = TriplePattern::with_subject(Term::iri("alice"));
        let results: Vec<Triple> = ring.find(&pattern).collect();

        assert_eq!(results.len(), 2);
        for triple in &results {
            assert_eq!(triple.subject(), &Term::iri("alice"));
        }
    }

    #[test]
    fn test_find_by_predicate() {
        let triples = vec![
            make_triple("s1", "type", "Person"),
            make_triple("s2", "type", "Place"),
            make_triple("s1", "name", "Alice"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        let pattern = TriplePattern::with_predicate(Term::iri("type"));
        let results: Vec<Triple> = ring.find(&pattern).collect();

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_find_by_object() {
        let triples = vec![
            make_triple("s1", "p1", "shared"),
            make_triple("s2", "p2", "shared"),
            make_triple("s3", "p3", "other"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        let pattern = TriplePattern::with_object(Term::iri("shared"));
        let results: Vec<Triple> = ring.find(&pattern).collect();

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_count() {
        let triples = vec![
            make_triple("s1", "p1", "o1"),
            make_triple("s1", "p2", "o2"),
            make_triple("s2", "p1", "o1"),
            make_triple("s2", "p1", "o3"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        // Count by subject
        assert_eq!(
            ring.count(&TriplePattern::with_subject(Term::iri("s1"))),
            2
        );
        assert_eq!(
            ring.count(&TriplePattern::with_subject(Term::iri("s2"))),
            2
        );

        // Count by predicate
        assert_eq!(
            ring.count(&TriplePattern::with_predicate(Term::iri("p1"))),
            3
        );
        assert_eq!(
            ring.count(&TriplePattern::with_predicate(Term::iri("p2"))),
            1
        );

        // Count by object
        assert_eq!(
            ring.count(&TriplePattern::with_object(Term::iri("o1"))),
            2
        );

        // Count all
        assert_eq!(ring.count(&TriplePattern::any()), 4);
    }

    #[test]
    fn test_permutation_consistency() {
        let triples = vec![
            make_triple("a", "x", "1"),
            make_triple("a", "y", "2"),
            make_triple("b", "x", "1"),
            make_triple("b", "y", "3"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        // Check that permutations are consistent
        for spo_idx in 0..ring.len() {
            // SPO → POS → SPO should round-trip
            if let Some(pos_idx) = ring.spo_to_pos(spo_idx) {
                let back = ring.pos_to_spo(pos_idx);
                assert_eq!(back, Some(spo_idx), "POS roundtrip failed for {}", spo_idx);
            }

            // SPO → OSP → SPO should round-trip
            if let Some(osp_idx) = ring.spo_to_osp(spo_idx) {
                let back = ring.osp_to_spo(osp_idx);
                assert_eq!(back, Some(spo_idx), "OSP roundtrip failed for {}", spo_idx);
            }
        }
    }

    #[test]
    fn test_size_bytes() {
        let triples: Vec<Triple> = (0..100)
            .map(|i| make_triple(&format!("s{}", i % 10), "knows", &format!("o{}", i % 20)))
            .collect();
        let ring = TripleRing::from_triples(triples.into_iter());

        let size = ring.size_bytes();
        // Should be reasonable (not huge)
        assert!(size > 0);
        assert!(size < 100_000, "Size {} seems too large", size);
    }
}
