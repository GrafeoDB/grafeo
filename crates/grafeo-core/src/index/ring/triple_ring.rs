//! Triple Ring - compact RDF triple index.
//!
//! The TripleRing stores RDF triples in a compact representation using
//! wavelet trees and succinct permutations, achieving ~3x space reduction
//! compared to hash-based triple indexing.

use super::permutation::SuccinctPermutation;
use crate::codec::succinct::WaveletTree;
use crate::graph::rdf::{Term, Triple, TriplePattern};
use hashbrown::HashMap;
use std::sync::Arc;

/// Structural-invariant violation surfaced by
/// [`TripleRing::from_packed_parts`] when malformed packed metadata is
/// detected during reconstruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TripleRingInvariantError {
    /// Per-component sequence length disagrees with `num_triples`.
    ComponentLengthMismatch {
        /// Which component disagreed: "subjects", "predicates",
        /// "objects", "spo_to_pos", or "spo_to_osp".
        component: &'static str,
        /// `num_triples` value declared in the header.
        expected: usize,
        /// Length actually carried by the component.
        actual: usize,
    },
    /// Packed dictionary's count exceeds `u32::MAX`, so a term id
    /// cannot fit into the `u32` term-id space the heap
    /// [`TermDictionary`] uses.
    DictionaryOverflow {
        /// Reported dictionary length.
        len: usize,
    },
    /// Packed dictionary reported a length but `get_term(id)` returned
    /// `None` for an id within `0..len`. Indicates corrupted dict
    /// payload — would otherwise have panicked in the original code.
    DictionaryMissingTerm {
        /// Id where the lookup failed.
        id: u32,
    },
}

impl std::fmt::Display for TripleRingInvariantError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ComponentLengthMismatch {
                component,
                expected,
                actual,
            } => write!(
                f,
                "triple ring {component} length ({actual}) does not match num_triples ({expected})"
            ),
            Self::DictionaryOverflow { len } => write!(
                f,
                "triple ring packed dictionary length ({len}) exceeds u32::MAX"
            ),
            Self::DictionaryMissingTerm { id } => write!(
                f,
                "triple ring packed dictionary missing term for id {id} (corrupt payload)"
            ),
        }
    }
}

impl std::error::Error for TripleRingInvariantError {}

/// Term dictionary mapping terms to compact integer IDs.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TermDictionary {
    /// Term to ID mapping.
    term_to_id: HashMap<Arc<Term>, u32, foldhash::fast::RandomState>,
    /// ID to term mapping.
    id_to_term: Vec<Arc<Term>>,
}

impl TermDictionary {
    /// Creates a new empty term dictionary.
    #[must_use]
    pub fn new() -> Self {
        Self {
            term_to_id: HashMap::with_hasher(foldhash::fast::RandomState::default()),
            id_to_term: Vec::new(),
        }
    }

    /// Creates a term dictionary with specified capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            term_to_id: HashMap::with_capacity_and_hasher(
                capacity,
                foldhash::fast::RandomState::default(),
            ),
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

        // reason: term dictionary size fits u32
        #[allow(clippy::cast_possible_truncation)]
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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

    /// Phase 6e packed-format access — returns the SPO→POS permutation.
    /// Distinct name from the existing query method `spo_to_pos(usize)`.
    #[must_use]
    pub fn spo_to_pos_perm(&self) -> &SuccinctPermutation {
        &self.spo_to_pos
    }

    /// Phase 6e packed-format access — returns the SPO→OSP permutation.
    /// Distinct name from the existing query method `spo_to_osp(usize)`.
    #[must_use]
    pub fn spo_to_osp_perm(&self) -> &SuccinctPermutation {
        &self.spo_to_osp
    }

    /// Phase 6e: reconstruction entry point used by
    /// [`crate::index::ring::packed_format::deserialize_triple_ring`]
    /// after parsing the v2 packed format. Skips the build path because
    /// the sub-components are already authoritative.
    ///
    /// `packed_dict` is the parsed packed dictionary; we materialize it
    /// back to a heap [`TermDictionary`] so the rest of the Ring's
    /// query path (which uses the heap dict) keeps working unchanged.
    /// A future pass can teach the query path to read directly from
    /// the packed dict for true zero-copy.
    ///
    /// # Errors
    ///
    /// Returns [`TripleRingInvariantError`] if any of these structural
    /// invariants is broken: per-component lengths must equal
    /// `num_triples`, the packed dictionary's count must fit in `u32`,
    /// and every id `< len` must resolve to a term. Without these
    /// checks, a corrupt or hand-crafted payload could cause `get_spo`
    /// to panic on out-of-bounds wavelet/permutation access.
    pub fn from_packed_parts(
        packed_dict: super::PackedTermDictionary,
        num_triples: usize,
        subjects: WaveletTree,
        predicates: WaveletTree,
        objects: WaveletTree,
        spo_to_pos: SuccinctPermutation,
        spo_to_osp: SuccinctPermutation,
    ) -> Result<Self, TripleRingInvariantError> {
        // Per-component length must equal num_triples; otherwise
        // `get_spo` would index out of bounds against the wavelet trees
        // or permutations.
        let lengths: [(&'static str, usize); 5] = [
            ("subjects", subjects.len()),
            ("predicates", predicates.len()),
            ("objects", objects.len()),
            ("spo_to_pos", spo_to_pos.len()),
            ("spo_to_osp", spo_to_osp.len()),
        ];
        for (component, actual) in lengths {
            if actual != num_triples {
                return Err(TripleRingInvariantError::ComponentLengthMismatch {
                    component,
                    expected: num_triples,
                    actual,
                });
            }
        }

        // Materialize the packed dictionary into a heap TermDictionary.
        // get_or_insert preserves insertion order, so id N in the
        // packed dict ends up as id N in the heap dict.
        let dict_len = packed_dict.len();
        if u32::try_from(dict_len).is_err() {
            return Err(TripleRingInvariantError::DictionaryOverflow { len: dict_len });
        }
        let mut dict = TermDictionary::with_capacity(dict_len);
        for id in 0..dict_len {
            // Cast is bounds-checked above.
            #[allow(clippy::cast_possible_truncation)]
            let id_u32 = id as u32;
            let term = packed_dict
                .get_term(id_u32)
                .ok_or(TripleRingInvariantError::DictionaryMissingTerm { id: id_u32 })?;
            dict.get_or_insert(term);
        }

        Ok(Self {
            dict,
            num_triples,
            subjects,
            predicates,
            objects,
            spo_to_pos,
            spo_to_osp,
        })
    }

    /// Returns the triple at position i in SPO order.
    #[must_use]
    pub fn get_spo(&self, index: usize) -> Option<Triple> {
        if index >= self.num_triples {
            return None;
        }

        // reason: dictionary IDs fit u32
        #[allow(clippy::cast_possible_truncation)]
        let s_id = self.subjects.access(index) as u32;
        // reason: dictionary IDs fit u32
        #[allow(clippy::cast_possible_truncation)]
        let p_id = self.predicates.access(index) as u32;
        // reason: dictionary IDs fit u32
        #[allow(clippy::cast_possible_truncation)]
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
            let Some(s_id) = self.dict.get_id(s) else {
                return 0;
            };
            let Some(p_id) = self.dict.get_id(p) else {
                return 0;
            };
            let Some(o_id) = self.dict.get_id(o) else {
                return 0;
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
                // reason: wavelet tree values are dictionary IDs, fit u32
                #[allow(clippy::cast_possible_truncation)]
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

    /// Serializes the Ring to a writer using bincode.
    ///
    /// The output contains the complete state: term dictionary, wavelet trees,
    /// and permutations. Use [`TripleRing::load`] to restore.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if writing fails or bincode encoding fails.
    pub fn save(&self, mut writer: impl std::io::Write) -> std::io::Result<()> {
        bincode::serde::encode_into_std_write(self, &mut writer, bincode::config::standard())
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        Ok(())
    }

    /// Validates structural invariants after deserialization.
    ///
    /// Ensures that all internal arrays are consistent with `num_triples`
    /// and the term dictionary, preventing panics from corrupted data.
    fn validate(&self) -> std::io::Result<()> {
        let n = self.num_triples;

        // --- Term dictionary consistency ---
        if self.dict.term_to_id.len() != self.dict.id_to_term.len() {
            return Err(std::io::Error::other(format!(
                "term dictionary inconsistent: term_to_id has {} entries, id_to_term has {}",
                self.dict.term_to_id.len(),
                self.dict.id_to_term.len()
            )));
        }

        // --- Wavelet tree lengths must match num_triples ---
        if self.subjects.len() != n {
            return Err(std::io::Error::other(format!(
                "subjects wavelet tree length {} != num_triples {n}",
                self.subjects.len()
            )));
        }
        if self.predicates.len() != n {
            return Err(std::io::Error::other(format!(
                "predicates wavelet tree length {} != num_triples {n}",
                self.predicates.len()
            )));
        }
        if self.objects.len() != n {
            return Err(std::io::Error::other(format!(
                "objects wavelet tree length {} != num_triples {n}",
                self.objects.len()
            )));
        }

        // --- Wavelet tree internal consistency ---
        self.subjects
            .validate()
            .map_err(|e| std::io::Error::other(format!("subjects wavelet tree: {e}")))?;
        self.predicates
            .validate()
            .map_err(|e| std::io::Error::other(format!("predicates wavelet tree: {e}")))?;
        self.objects
            .validate()
            .map_err(|e| std::io::Error::other(format!("objects wavelet tree: {e}")))?;

        // --- Wavelet tree symbols must reference valid dictionary IDs ---
        let dict_len = self.dict.len() as u64;
        for sym in self.subjects.alphabet() {
            if sym >= dict_len {
                return Err(std::io::Error::other(format!(
                    "subjects wavelet tree contains symbol {sym} >= dict size {dict_len}"
                )));
            }
        }
        for sym in self.predicates.alphabet() {
            if sym >= dict_len {
                return Err(std::io::Error::other(format!(
                    "predicates wavelet tree contains symbol {sym} >= dict size {dict_len}"
                )));
            }
        }
        for sym in self.objects.alphabet() {
            if sym >= dict_len {
                return Err(std::io::Error::other(format!(
                    "objects wavelet tree contains symbol {sym} >= dict size {dict_len}"
                )));
            }
        }

        // --- Permutation lengths must match num_triples ---
        if self.spo_to_pos.len() != n {
            return Err(std::io::Error::other(format!(
                "spo_to_pos permutation length {} != num_triples {n}",
                self.spo_to_pos.len()
            )));
        }
        if self.spo_to_osp.len() != n {
            return Err(std::io::Error::other(format!(
                "spo_to_osp permutation length {} != num_triples {n}",
                self.spo_to_osp.len()
            )));
        }

        Ok(())
    }

    /// Deserializes a Ring from a reader.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if reading fails or the data is malformed.
    pub fn load(mut reader: impl std::io::Read) -> std::io::Result<Self> {
        let ring: Self =
            bincode::serde::decode_from_std_read(&mut reader, bincode::config::standard())
                .map_err(|e| std::io::Error::other(e.to_string()))?;
        ring.validate()?;
        Ok(ring)
    }

    /// Saves the Ring to a file path.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the file cannot be created or writing fails.
    pub fn save_to_file(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        let writer = std::io::BufWriter::new(file);
        self.save(writer)
    }

    /// Loads a Ring from a file path.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the file cannot be opened or the data is malformed.
    pub fn load_from_file(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        Self::load(reader)
    }

    /// Serializes the Ring to a byte vector.
    ///
    /// Used by the Section trait for container persistence.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if encoding fails.
    pub fn save_to_bytes(&self) -> std::io::Result<Vec<u8>> {
        bincode::serde::encode_to_vec(self, bincode::config::standard())
            .map_err(|e| std::io::Error::other(e.to_string()))
    }

    /// Deserializes a Ring from a byte slice.
    ///
    /// Used by the Section trait when loading from the container.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the data is malformed.
    pub fn load_from_bytes(data: &[u8]) -> std::io::Result<Self> {
        let (ring, _bytes_read): (Self, _) =
            bincode::serde::decode_from_slice(data, bincode::config::standard())
                .map_err(|e| std::io::Error::other(e.to_string()))?;
        ring.validate()?;
        Ok(ring)
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

            if let Some(triple) = self.ring.get_spo(idx)
                && self.pattern.matches(&triple)
            {
                return Some(triple);
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
            make_triple("alix", "knows", "gus"),
            make_triple("alix", "knows", "harm"),
            make_triple("gus", "knows", "harm"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        let pattern = TriplePattern::with_subject(Term::iri("alix"));
        let results: Vec<Triple> = ring.find(&pattern).collect();

        assert_eq!(results.len(), 2);
        for triple in &results {
            assert_eq!(triple.subject(), &Term::iri("alix"));
        }
    }

    #[test]
    fn test_find_by_predicate() {
        let triples = vec![
            make_triple("s1", "type", "Person"),
            make_triple("s2", "type", "Place"),
            make_triple("s1", "name", "Alix"),
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
        assert_eq!(ring.count(&TriplePattern::with_subject(Term::iri("s1"))), 2);
        assert_eq!(ring.count(&TriplePattern::with_subject(Term::iri("s2"))), 2);

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
        assert_eq!(ring.count(&TriplePattern::with_object(Term::iri("o1"))), 2);

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

    #[test]
    fn test_term_dictionary_with_capacity() {
        let mut dict = TermDictionary::with_capacity(100);
        assert!(dict.is_empty());
        assert_eq!(dict.len(), 0);

        // Add some terms
        let id1 = dict.get_or_insert(Term::iri("test1"));
        let id2 = dict.get_or_insert(Term::iri("test2"));

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(dict.len(), 2);
    }

    #[test]
    fn test_term_dictionary_size_bytes() {
        let mut dict = TermDictionary::new();
        let empty_size = dict.size_bytes();
        assert!(empty_size > 0);

        // Add terms and verify size increases
        dict.get_or_insert(Term::iri("some_long_term_name"));
        let size_with_term = dict.size_bytes();
        assert!(size_with_term > empty_size);
    }

    #[test]
    fn test_term_dictionary_get_existing() {
        let mut dict = TermDictionary::new();
        let term = Term::iri("test");

        let id1 = dict.get_or_insert(term.clone());
        let id2 = dict.get_or_insert(term.clone());

        // Should return same ID for duplicate term
        assert_eq!(id1, id2);
        assert_eq!(dict.len(), 1);
    }

    #[test]
    fn test_term_dictionary_get_term_not_found() {
        let dict = TermDictionary::new();
        assert!(dict.get_term(999).is_none());
    }

    #[test]
    fn test_term_dictionary_get_id_not_found() {
        let dict = TermDictionary::new();
        assert!(dict.get_id(&Term::iri("nonexistent")).is_none());
    }

    #[test]
    fn test_get_spo_out_of_bounds() {
        let triples = vec![make_triple("s", "p", "o")];
        let ring = TripleRing::from_triples(triples.into_iter());

        assert!(ring.get_spo(0).is_some());
        assert!(ring.get_spo(1).is_none());
        assert!(ring.get_spo(100).is_none());
    }

    #[test]
    fn test_count_exact_match() {
        let triples = vec![
            make_triple("s1", "p1", "o1"),
            make_triple("s1", "p1", "o2"),
            make_triple("s2", "p1", "o1"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        // Exact match should return 1
        let pattern = TriplePattern {
            subject: Some(Term::iri("s1")),
            predicate: Some(Term::iri("p1")),
            object: Some(Term::iri("o1")),
        };
        assert_eq!(ring.count(&pattern), 1);

        // Non-existent exact match should return 0
        let pattern_missing = TriplePattern {
            subject: Some(Term::iri("s1")),
            predicate: Some(Term::iri("p1")),
            object: Some(Term::iri("o3")),
        };
        assert_eq!(ring.count(&pattern_missing), 0);
    }

    #[test]
    fn test_count_two_components_bound() {
        let triples = vec![
            make_triple("s1", "p1", "o1"),
            make_triple("s1", "p1", "o2"),
            make_triple("s1", "p2", "o1"),
            make_triple("s2", "p1", "o1"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        // Subject and predicate bound
        let pattern_sp = TriplePattern {
            subject: Some(Term::iri("s1")),
            predicate: Some(Term::iri("p1")),
            object: None,
        };
        assert_eq!(ring.count(&pattern_sp), 2);

        // Subject and object bound
        let pattern_so = TriplePattern {
            subject: Some(Term::iri("s1")),
            predicate: None,
            object: Some(Term::iri("o1")),
        };
        assert_eq!(ring.count(&pattern_so), 2);

        // Predicate and object bound
        let pattern_po = TriplePattern {
            subject: None,
            predicate: Some(Term::iri("p1")),
            object: Some(Term::iri("o1")),
        };
        assert_eq!(ring.count(&pattern_po), 2);
    }

    #[test]
    fn test_count_nonexistent_term() {
        let triples = vec![make_triple("s1", "p1", "o1")];
        let ring = TripleRing::from_triples(triples.into_iter());

        assert_eq!(
            ring.count(&TriplePattern::with_subject(Term::iri("nonexistent"))),
            0
        );
        assert_eq!(
            ring.count(&TriplePattern::with_predicate(Term::iri("nonexistent"))),
            0
        );
        assert_eq!(
            ring.count(&TriplePattern::with_object(Term::iri("nonexistent"))),
            0
        );
    }

    #[test]
    fn test_count_exact_match_nonexistent_subject() {
        let triples = vec![make_triple("s1", "p1", "o1")];
        let ring = TripleRing::from_triples(triples.into_iter());

        let pattern = TriplePattern {
            subject: Some(Term::iri("nonexistent")),
            predicate: Some(Term::iri("p1")),
            object: Some(Term::iri("o1")),
        };
        assert_eq!(ring.count(&pattern), 0);
    }

    #[test]
    fn test_count_exact_match_nonexistent_predicate() {
        let triples = vec![make_triple("s1", "p1", "o1")];
        let ring = TripleRing::from_triples(triples.into_iter());

        let pattern = TriplePattern {
            subject: Some(Term::iri("s1")),
            predicate: Some(Term::iri("nonexistent")),
            object: Some(Term::iri("o1")),
        };
        assert_eq!(ring.count(&pattern), 0);
    }

    #[test]
    fn test_count_exact_match_nonexistent_object() {
        let triples = vec![make_triple("s1", "p1", "o1")];
        let ring = TripleRing::from_triples(triples.into_iter());

        let pattern = TriplePattern {
            subject: Some(Term::iri("s1")),
            predicate: Some(Term::iri("p1")),
            object: Some(Term::iri("nonexistent")),
        };
        assert_eq!(ring.count(&pattern), 0);
    }

    #[test]
    fn test_dictionary_accessor() {
        let triples = vec![
            make_triple("alix", "knows", "gus"),
            make_triple("alix", "likes", "vincent"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        let dict = ring.dictionary();
        assert!(!dict.is_empty());
        // Should have 5 unique terms: alix, knows, gus, likes, vincent
        assert_eq!(dict.len(), 5);

        // Verify we can look up terms
        assert!(dict.get_id(&Term::iri("alix")).is_some());
        assert!(dict.get_id(&Term::iri("knows")).is_some());
        assert!(dict.get_id(&Term::iri("gus")).is_some());
    }

    #[test]
    fn test_same_term_multiple_positions() {
        // Same term appears as subject, predicate, and object
        let triples = vec![
            make_triple("same", "same", "same"),
            make_triple("same", "other", "different"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        // Should only have 3 unique terms: same, other, different
        assert_eq!(ring.num_terms(), 3);
        assert_eq!(ring.len(), 2);

        // Verify we can find triples with this term
        let pattern_s = TriplePattern::with_subject(Term::iri("same"));
        assert_eq!(ring.count(&pattern_s), 2);

        let pattern_p = TriplePattern::with_predicate(Term::iri("same"));
        assert_eq!(ring.count(&pattern_p), 1);

        let pattern_o = TriplePattern::with_object(Term::iri("same"));
        assert_eq!(ring.count(&pattern_o), 1);
    }

    #[test]
    fn test_find_no_matches() {
        let triples = vec![make_triple("s1", "p1", "o1")];
        let ring = TripleRing::from_triples(triples.into_iter());

        let pattern = TriplePattern::with_subject(Term::iri("nonexistent"));
        let results: Vec<Triple> = ring.find(&pattern).collect();
        assert!(results.is_empty(), "{results:?}");
    }

    #[test]
    fn test_find_all_triples() {
        let triples = vec![
            make_triple("s1", "p1", "o1"),
            make_triple("s2", "p2", "o2"),
            make_triple("s3", "p3", "o3"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        let pattern = TriplePattern::any();
        let results: Vec<Triple> = ring.find(&pattern).collect();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_wavelet_tree_accessors() {
        let triples = vec![make_triple("s", "p", "o")];
        let ring = TripleRing::from_triples(triples.into_iter());

        // Verify wavelet tree accessors work
        let subjects_wt = ring.subjects_wt();
        let predicates_wt = ring.predicates_wt();
        let objects_wt = ring.objects_wt();

        // Each should have exactly one entry
        assert_eq!(subjects_wt.len(), 1);
        assert_eq!(predicates_wt.len(), 1);
        assert_eq!(objects_wt.len(), 1);
    }

    #[test]
    fn test_permutation_out_of_bounds() {
        let triples = vec![make_triple("s", "p", "o")];
        let ring = TripleRing::from_triples(triples.into_iter());

        // Index 0 should work
        assert!(ring.spo_to_pos(0).is_some());
        assert!(ring.spo_to_osp(0).is_some());

        // Out of bounds should return None
        assert!(ring.spo_to_pos(100).is_none());
        assert!(ring.spo_to_osp(100).is_none());
        assert!(ring.pos_to_spo(100).is_none());
        assert!(ring.osp_to_spo(100).is_none());
    }

    #[test]
    fn test_contains_ids_no_match() {
        let triples = vec![make_triple("s1", "p1", "o1"), make_triple("s1", "p2", "o2")];
        let ring = TripleRing::from_triples(triples.into_iter());

        // Exact match that doesn't exist (s1, p1, o2)
        let pattern = TriplePattern {
            subject: Some(Term::iri("s1")),
            predicate: Some(Term::iri("p1")),
            object: Some(Term::iri("o2")),
        };
        assert_eq!(ring.count(&pattern), 0);
    }

    #[test]
    fn test_empty_ring_operations() {
        let ring = TripleRing::from_triples(std::iter::empty());

        assert!(ring.is_empty());
        assert_eq!(ring.len(), 0);
        assert!(ring.get_spo(0).is_none());
        assert_eq!(ring.count(&TriplePattern::any()), 0);
        assert_eq!(ring.count(&TriplePattern::with_subject(Term::iri("s"))), 0);
        assert!(ring.spo_to_pos(0).is_none());
        assert!(ring.osp_to_spo(0).is_none());

        // Find on empty ring
        let results: Vec<Triple> = ring.find(&TriplePattern::any()).collect();
        assert!(results.is_empty(), "{results:?}");
    }

    #[test]
    fn test_serialization_roundtrip() {
        let triples = vec![
            Triple::new(
                Term::iri("http://ex.org/alix"),
                Term::iri("http://xmlns.com/foaf/0.1/name"),
                Term::literal("Alix"),
            ),
            Triple::new(
                Term::iri("http://ex.org/gus"),
                Term::iri("http://xmlns.com/foaf/0.1/name"),
                Term::literal("Gus"),
            ),
            Triple::new(
                Term::iri("http://ex.org/alix"),
                Term::iri("http://xmlns.com/foaf/0.1/knows"),
                Term::iri("http://ex.org/gus"),
            ),
        ];

        let ring = TripleRing::from_triples(triples.into_iter());
        assert_eq!(ring.len(), 3);

        // Save to buffer
        let mut buf = Vec::new();
        ring.save(&mut buf).expect("save should succeed");
        assert!(!buf.is_empty(), "buf is empty");

        // Load from buffer
        let loaded = TripleRing::load(&buf[..]).expect("load should succeed");
        assert_eq!(loaded.len(), ring.len());
        assert_eq!(loaded.num_terms(), ring.num_terms());

        // Verify all triples round-trip
        let original: Vec<Triple> = ring.find(&TriplePattern::any()).collect();
        let restored: Vec<Triple> = loaded.find(&TriplePattern::any()).collect();
        assert_eq!(original.len(), restored.len());

        // Verify count operations work on loaded ring
        let name_pattern = TriplePattern {
            subject: None,
            predicate: Some(Term::iri("http://xmlns.com/foaf/0.1/name")),
            object: None,
        };
        assert_eq!(loaded.count(&name_pattern), 2);
    }

    #[test]
    fn test_save_load_file() {
        let triples = vec![
            Triple::new(
                Term::iri("http://ex.org/a"),
                Term::iri("http://ex.org/p"),
                Term::iri("http://ex.org/b"),
            ),
            Triple::new(
                Term::iri("http://ex.org/b"),
                Term::iri("http://ex.org/p"),
                Term::iri("http://ex.org/c"),
            ),
        ];

        let ring = TripleRing::from_triples(triples.into_iter());

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("test.ring");

        ring.save_to_file(&path).expect("save_to_file");
        let loaded = TripleRing::load_from_file(&path).expect("load_from_file");

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.count(&TriplePattern::any()), 2);
    }

    #[test]
    fn test_load_rejects_truncated_data() {
        let triples = vec![make_triple("s1", "p1", "o1")];
        let ring = TripleRing::from_triples(triples.into_iter());

        let bytes = ring.save_to_bytes().expect("save_to_bytes");
        // Truncate to half the data
        let truncated = &bytes[..bytes.len() / 2];
        let result = TripleRing::load_from_bytes(truncated);
        assert!(result.is_err(), "truncated data should fail to load");
    }

    #[test]
    fn test_load_rejects_empty_bytes() {
        let result = TripleRing::load_from_bytes(&[]);
        assert!(result.is_err(), "empty bytes should fail to load");
    }

    #[test]
    fn test_load_rejects_garbage_bytes() {
        let garbage = vec![0xFF; 256];
        let result = TripleRing::load_from_bytes(&garbage);
        assert!(result.is_err(), "garbage bytes should fail to load");
    }

    #[test]
    fn test_load_from_bytes_roundtrip() {
        let triples = vec![make_triple("s1", "p1", "o1"), make_triple("s2", "p2", "o2")];
        let ring = TripleRing::from_triples(triples.into_iter());

        let bytes = ring.save_to_bytes().expect("save_to_bytes");
        let loaded = TripleRing::load_from_bytes(&bytes).expect("load_from_bytes");

        assert_eq!(loaded.len(), ring.len());
        assert_eq!(loaded.num_terms(), ring.num_terms());
    }

    #[test]
    fn test_validate_passes_for_valid_ring() {
        let triples = vec![
            make_triple("alix", "knows", "gus"),
            make_triple("alix", "likes", "vincent"),
            make_triple("gus", "knows", "vincent"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        // Validation should pass for a freshly constructed ring
        assert!(ring.validate().is_ok());
    }

    #[test]
    fn test_validate_passes_for_empty_ring() {
        let ring = TripleRing::from_triples(std::iter::empty());
        assert!(ring.validate().is_ok());
    }

    #[test]
    fn test_save_load_bytes_roundtrip() {
        let triples = vec![
            make_triple("alix", "knows", "gus"),
            make_triple("alix", "likes", "vincent"),
            make_triple("gus", "knows", "vincent"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());
        let bytes = ring.save_to_bytes().unwrap();
        let loaded = TripleRing::load_from_bytes(&bytes).unwrap();
        assert_eq!(loaded.len(), ring.len());
        // Verify query results match
        let pattern = TriplePattern {
            subject: None,
            predicate: None,
            object: None,
        };
        assert_eq!(loaded.count(&pattern), ring.count(&pattern));
    }

    #[test]
    fn test_save_load_writer_reader_roundtrip() {
        let triples = vec![
            make_triple("alix", "knows", "gus"),
            make_triple("gus", "likes", "vincent"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());
        let mut buf = Vec::new();
        ring.save(&mut buf).unwrap();
        let loaded = TripleRing::load(&buf[..]).unwrap();
        assert_eq!(loaded.len(), ring.len());
    }

    #[test]
    fn test_load_from_bytes_corrupt_data_fails() {
        let result = TripleRing::load_from_bytes(&[0xFF, 0xFE, 0xFD, 0xFC]);
        assert!(result.is_err());
    }

    #[test]
    fn test_save_load_file_roundtrip() {
        let triples = vec![
            make_triple("alix", "knows", "gus"),
            make_triple("gus", "knows", "vincent"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("ring_roundtrip.bin");
        ring.save_to_file(&path).unwrap();
        let loaded = TripleRing::load_from_file(&path).unwrap();
        assert_eq!(loaded.len(), ring.len());
        // dir dropped here: automatic cleanup even on panic
    }

    #[test]
    fn test_save_load_empty_ring() {
        let ring = TripleRing::from_triples(std::iter::empty());
        let bytes = ring.save_to_bytes().unwrap();
        let loaded = TripleRing::load_from_bytes(&bytes).unwrap();
        assert_eq!(loaded.len(), 0);
    }
}
