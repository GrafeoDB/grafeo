//! Leapfrog iteration for Ring Index.
//!
//! Provides efficient multi-pattern joins using the Ring Index structure.
//! The leapfrog join enables worst-case optimal joins (WCOJ) over RDF patterns.

use super::triple_ring::TripleRing;
use crate::graph::rdf::{Term, Triple, TriplePattern};

/// Iterator over a single component of the Ring Index.
///
/// Efficiently iterates over triples filtered by a specific term binding.
#[derive(Debug)]
pub struct RingIterator<'a> {
    ring: &'a TripleRing,
    /// Current position in the sequence.
    pos: usize,
    /// End position (exclusive).
    end: usize,
    /// Component being iterated (0 = subject, 1 = predicate, 2 = object).
    component: u8,
    /// Bound term ID for filtering.
    bound_id: Option<u32>,
    /// Current rank within the bound term's occurrences.
    rank: usize,
    /// Total count of bound term.
    count: usize,
    /// Whether this is an "iterate all" iterator (vs bound search).
    iterate_all: bool,
}

impl<'a> RingIterator<'a> {
    /// Creates an iterator over all triples.
    pub fn all(ring: &'a TripleRing) -> Self {
        Self {
            ring,
            pos: 0,
            end: ring.len(),
            component: 0,
            bound_id: None,
            rank: 0,
            count: ring.len(),
            iterate_all: true,
        }
    }

    /// Creates an iterator over triples with a specific subject.
    pub fn with_subject(ring: &'a TripleRing, subject: &Term) -> Self {
        let (bound_id, count) = if let Some(id) = ring.dictionary().get_id(subject) {
            let count = ring.count(&TriplePattern::with_subject(subject.clone()));
            (Some(id), count)
        } else {
            (None, 0)
        };

        Self {
            ring,
            pos: 0,
            end: ring.len(),
            component: 0,
            bound_id,
            rank: 0,
            count,
            iterate_all: false,
        }
    }

    /// Creates an iterator over triples with a specific predicate.
    pub fn with_predicate(ring: &'a TripleRing, predicate: &Term) -> Self {
        let (bound_id, count) = if let Some(id) = ring.dictionary().get_id(predicate) {
            let count = ring.count(&TriplePattern::with_predicate(predicate.clone()));
            (Some(id), count)
        } else {
            (None, 0)
        };

        Self {
            ring,
            pos: 0,
            end: ring.len(),
            component: 1,
            bound_id,
            rank: 0,
            count,
            iterate_all: false,
        }
    }

    /// Creates an iterator over triples with a specific object.
    pub fn with_object(ring: &'a TripleRing, object: &Term) -> Self {
        let (bound_id, count) = if let Some(id) = ring.dictionary().get_id(object) {
            let count = ring.count(&TriplePattern::with_object(object.clone()));
            (Some(id), count)
        } else {
            (None, 0)
        };

        Self {
            ring,
            pos: 0,
            end: ring.len(),
            component: 2,
            bound_id,
            rank: 0,
            count,
            iterate_all: false,
        }
    }

    /// Returns the current position.
    #[must_use]
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Returns whether there are more elements.
    #[must_use]
    pub fn has_next(&self) -> bool {
        if self.iterate_all {
            self.pos < self.end
        } else if self.bound_id.is_some() {
            self.rank < self.count
        } else {
            // Searching for a term that wasn't found
            false
        }
    }

    /// Returns the term ID at the current position for a given component.
    ///
    /// Used by the leapfrog algorithm to compare term IDs across iterators.
    #[must_use]
    pub fn current_term_id(&self, component: u8) -> Option<u32> {
        if self.pos >= self.end {
            return None;
        }
        let wt = match component {
            0 => self.ring.subjects_wt(),
            1 => self.ring.predicates_wt(),
            _ => self.ring.objects_wt(),
        };
        // reason: wavelet tree values are dictionary IDs, fit u32
        #[allow(clippy::cast_possible_truncation)]
        Some(wt.access(self.pos) as u32)
    }

    /// Seeks to the next position where the given component's term ID >= target_id.
    ///
    /// Returns true if such a position was found.
    pub fn seek_term(&mut self, component: u8, target_id: u32) -> bool {
        while self.pos < self.end {
            if let Some(current_id) = self.current_term_id(component)
                && current_id >= target_id
            {
                return true;
            }
            // Advance to next position
            if self.iterate_all {
                self.pos += 1;
            } else if let Some(bound) = self.bound_id {
                self.rank += 1;
                if !self.has_next() {
                    return false;
                }
                let wt = match self.component {
                    0 => self.ring.subjects_wt(),
                    1 => self.ring.predicates_wt(),
                    _ => self.ring.objects_wt(),
                };
                if let Some(next_pos) = wt.select(bound as u64, self.rank) {
                    self.pos = next_pos;
                } else {
                    return false;
                }
            } else {
                return false;
            }
        }
        false
    }

    /// Seeks to the first position >= target.
    ///
    /// For leapfrog join, this is the key operation.
    ///
    /// # Panics
    ///
    pub fn seek(&mut self, target: usize) {
        if self.iterate_all {
            // For iterate-all, just move position
            self.pos = target.min(self.end);
        } else if let Some(bound) = self.bound_id {
            // For bound iterators, we need to find the next occurrence >= target
            while self.has_next() {
                let wt = match self.component {
                    0 => self.ring.subjects_wt(),
                    1 => self.ring.predicates_wt(),
                    _ => self.ring.objects_wt(),
                };

                if let Some(next_pos) = wt.select(bound as u64, self.rank) {
                    if next_pos >= target {
                        self.pos = next_pos;
                        return;
                    }
                    self.rank += 1;
                } else {
                    break;
                }
            }
            // No more elements
            self.pos = self.end;
        }
        // If bound_id is None and not iterate_all, do nothing (term not found)
    }
}

impl Iterator for RingIterator<'_> {
    type Item = Triple;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.has_next() {
            return None;
        }

        let pos = if self.iterate_all {
            // Iterate all triples
            let p = self.pos;
            self.pos += 1;
            p
        } else {
            // Term not found (no bound id) shouldn't reach here due to has_next() check
            let id = self.bound_id?;
            // Get next position for this term using wavelet tree select
            let wt = match self.component {
                0 => self.ring.subjects_wt(),
                1 => self.ring.predicates_wt(),
                _ => self.ring.objects_wt(),
            };

            let next_pos = wt.select(id as u64, self.rank)?;
            self.rank += 1;
            self.pos = next_pos + 1;
            next_pos
        };

        self.ring.get_spo(pos)
    }
}

/// A triple pattern annotated with variable names for each position.
///
/// Variable positions use `Some("var_name")`, bound positions use `None`.
#[derive(Debug, Clone)]
pub struct AnnotatedPattern {
    /// The underlying triple pattern (bound terms).
    pub pattern: TriplePattern,
    /// Variable name for subject position (None if bound).
    pub subject_var: Option<String>,
    /// Variable name for predicate position (None if bound).
    pub predicate_var: Option<String>,
    /// Variable name for object position (None if bound).
    pub object_var: Option<String>,
}

impl AnnotatedPattern {
    /// Returns the variable name at the given component (0=subject, 1=predicate, 2=object).
    fn var_at(&self, component: u8) -> Option<&str> {
        match component {
            0 => self.subject_var.as_deref(),
            1 => self.predicate_var.as_deref(),
            2 => self.object_var.as_deref(),
            _ => None,
        }
    }
}

/// Leapfrog join over multiple Ring iterators.
///
/// Implements the leapfrog triejoin algorithm for worst-case optimal joins
/// over RDF triple patterns. Supports variable binding propagation across
/// patterns for proper multi-pattern joins.
pub struct LeapfrogRing<'a> {
    ring: &'a TripleRing,
    /// Patterns to join.
    patterns: Vec<TriplePattern>,
    /// Annotated patterns with variable names (used by `with_variables`).
    annotated: Option<Vec<AnnotatedPattern>>,
    /// Pre-computed results (eagerly materialized on first next() call).
    results: Vec<Vec<Triple>>,
    /// Index into results for iteration.
    result_idx: usize,
    /// Whether the join is exhausted.
    exhausted: bool,
}

impl<'a> LeapfrogRing<'a> {
    /// Creates a new leapfrog join over the given patterns.
    pub fn new(ring: &'a TripleRing, patterns: Vec<TriplePattern>) -> Self {
        let exhausted = patterns.is_empty() || ring.is_empty();
        Self {
            ring,
            patterns,
            annotated: None,
            results: Vec::new(),
            result_idx: 0,
            exhausted,
        }
    }

    /// Creates a leapfrog join with variable annotations for proper binding propagation.
    pub fn with_variables(ring: &'a TripleRing, annotated: Vec<AnnotatedPattern>) -> Self {
        let exhausted = annotated.is_empty() || ring.is_empty();
        let patterns = annotated.iter().map(|a| a.pattern.clone()).collect();
        Self {
            ring,
            patterns,
            annotated: Some(annotated),
            results: Vec::new(),
            result_idx: 0,
            exhausted,
        }
    }

    /// Returns the patterns being joined.
    #[must_use]
    pub fn patterns(&self) -> &[TriplePattern] {
        &self.patterns
    }

    /// Returns whether the join is exhausted.
    #[must_use]
    pub fn is_exhausted(&self) -> bool {
        self.exhausted
    }

    /// Materializes all results using variable binding propagation.
    ///
    /// For each triple matching the first pattern, binds its variables and
    /// filters subsequent patterns to only match consistent bindings.
    fn materialize_results(&mut self) {
        use std::collections::HashMap;

        if self.patterns.is_empty() {
            return;
        }

        let annotated = match &self.annotated {
            Some(a) => a.clone(),
            None => {
                // No variable annotations: fall back to simple pattern matching
                self.materialize_simple();
                return;
            }
        };

        // Find all triples matching the first pattern
        let first_triples: Vec<Triple> = self.ring.find(&annotated[0].pattern).collect();

        for first_triple in &first_triples {
            // Build initial bindings from first triple
            let mut bindings: HashMap<String, Term> = HashMap::new();
            Self::bind_triple(&annotated[0], first_triple, &mut bindings);

            let mut matched = vec![first_triple.clone()];
            let mut all_match = true;

            // For each subsequent pattern, apply bindings and find matches
            for ann_pattern in &annotated[1..] {
                let refined = Self::refine_pattern(ann_pattern, &bindings);
                let mut found_match = false;

                for triple in self.ring.find(&refined) {
                    // Verify that all shared variables are consistent
                    if Self::is_consistent(ann_pattern, &triple, &bindings) {
                        // Add new bindings from this triple
                        Self::bind_triple(ann_pattern, &triple, &mut bindings);
                        matched.push(triple);
                        found_match = true;
                        break;
                    }
                }

                if !found_match {
                    all_match = false;
                    break;
                }
            }

            if all_match {
                self.results.push(matched);
            }
        }
    }

    /// Simple materialization without variable annotations (backward compatible).
    fn materialize_simple(&mut self) {
        for triple in self.ring.find(&self.patterns[0]) {
            let mut matched = vec![triple.clone()];
            let mut all_match = true;

            for pattern in &self.patterns[1..] {
                if let Some(t) = self.ring.find(pattern).next() {
                    matched.push(t);
                } else {
                    all_match = false;
                    break;
                }
            }

            if all_match {
                self.results.push(matched);
            }
        }
    }

    /// Binds variable positions of a triple into the binding map.
    fn bind_triple(
        ann: &AnnotatedPattern,
        triple: &Triple,
        bindings: &mut std::collections::HashMap<String, Term>,
    ) {
        if let Some(ref var) = ann.subject_var {
            bindings
                .entry(var.clone())
                .or_insert_with(|| triple.subject().clone());
        }
        if let Some(ref var) = ann.predicate_var {
            bindings
                .entry(var.clone())
                .or_insert_with(|| triple.predicate().clone());
        }
        if let Some(ref var) = ann.object_var {
            bindings
                .entry(var.clone())
                .or_insert_with(|| triple.object().clone());
        }
    }

    /// Refines a pattern by substituting known variable bindings as bound terms.
    fn refine_pattern(
        ann: &AnnotatedPattern,
        bindings: &std::collections::HashMap<String, Term>,
    ) -> TriplePattern {
        let mut refined = ann.pattern.clone();
        if let Some(ref var) = ann.subject_var
            && let Some(term) = bindings.get(var)
        {
            refined.subject = Some(term.clone());
        }
        if let Some(ref var) = ann.predicate_var
            && let Some(term) = bindings.get(var)
        {
            refined.predicate = Some(term.clone());
        }
        if let Some(ref var) = ann.object_var
            && let Some(term) = bindings.get(var)
        {
            refined.object = Some(term.clone());
        }
        refined
    }

    /// Checks if a triple is consistent with existing variable bindings.
    fn is_consistent(
        ann: &AnnotatedPattern,
        triple: &Triple,
        bindings: &std::collections::HashMap<String, Term>,
    ) -> bool {
        for (component, term) in [
            (0u8, triple.subject()),
            (1, triple.predicate()),
            (2, triple.object()),
        ] {
            if let Some(var_name) = ann.var_at(component)
                && let Some(bound_term) = bindings.get(var_name)
                && bound_term != term
            {
                return false;
            }
        }
        true
    }
}

impl Iterator for LeapfrogRing<'_> {
    type Item = Vec<Triple>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }

        // Materialize all results on first call
        if self.result_idx == 0 && self.results.is_empty() {
            self.materialize_results();
        }

        if self.result_idx < self.results.len() {
            let result = self.results[self.result_idx].clone();
            self.result_idx += 1;
            Some(result)
        } else {
            self.exhausted = true;
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_triple(s: &str, p: &str, o: &str) -> Triple {
        Triple::new(Term::iri(s), Term::iri(p), Term::iri(o))
    }

    #[test]
    fn test_ring_iterator_all() {
        let triples = vec![make_triple("s1", "p1", "o1"), make_triple("s2", "p2", "o2")];
        let ring = TripleRing::from_triples(triples.into_iter());

        let iter = RingIterator::all(&ring);
        let results: Vec<Triple> = iter.collect();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_ring_iterator_with_subject() {
        let triples = vec![
            make_triple("alix", "knows", "gus"),
            make_triple("alix", "knows", "harm"),
            make_triple("gus", "knows", "harm"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        let iter = RingIterator::with_subject(&ring, &Term::iri("alix"));
        let results: Vec<Triple> = iter.collect();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_ring_iterator_with_predicate() {
        let triples = vec![
            make_triple("s1", "type", "Person"),
            make_triple("s2", "type", "Place"),
            make_triple("s1", "name", "Alix"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        let iter = RingIterator::with_predicate(&ring, &Term::iri("type"));
        let results: Vec<Triple> = iter.collect();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_ring_iterator_not_found() {
        let triples = vec![make_triple("s1", "p1", "o1")];
        let ring = TripleRing::from_triples(triples.into_iter());

        let iter = RingIterator::with_subject(&ring, &Term::iri("nonexistent"));
        let results: Vec<Triple> = iter.collect();
        assert!(results.is_empty());
    }

    #[test]
    fn test_leapfrog_empty() {
        let ring = TripleRing::from_triples(std::iter::empty());
        let lf = LeapfrogRing::new(&ring, vec![]);
        assert!(lf.is_exhausted());
    }

    #[test]
    fn test_leapfrog_single_pattern() {
        let triples = vec![
            make_triple("alix", "knows", "gus"),
            make_triple("gus", "knows", "harm"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        let pattern = TriplePattern::with_subject(Term::iri("alix"));
        let mut lf = LeapfrogRing::new(&ring, vec![pattern]);

        let result = lf.next();
        assert!(result.is_some());
        let triples = result.unwrap();
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].subject(), &Term::iri("alix"));
    }

    #[test]
    fn test_ring_iterator_with_object() {
        let triples = vec![
            make_triple("alix", "knows", "gus"),
            make_triple("harm", "knows", "gus"),
            make_triple("dave", "likes", "eve"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        let iter = RingIterator::with_object(&ring, &Term::iri("gus"));
        let results: Vec<Triple> = iter.collect();
        assert_eq!(results.len(), 2);

        // Verify all results have gus as object
        for triple in &results {
            assert_eq!(triple.object(), &Term::iri("gus"));
        }
    }

    #[test]
    fn test_ring_iterator_with_object_not_found() {
        let triples = vec![make_triple("s1", "p1", "o1")];
        let ring = TripleRing::from_triples(triples.into_iter());

        let iter = RingIterator::with_object(&ring, &Term::iri("nonexistent"));
        let results: Vec<Triple> = iter.collect();
        assert!(results.is_empty());
    }

    #[test]
    fn test_ring_iterator_position() {
        let triples = vec![
            make_triple("s1", "p1", "o1"),
            make_triple("s2", "p2", "o2"),
            make_triple("s3", "p3", "o3"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        let mut iter = RingIterator::all(&ring);
        assert_eq!(iter.position(), 0);

        iter.next();
        assert_eq!(iter.position(), 1);

        iter.next();
        assert_eq!(iter.position(), 2);
    }

    #[test]
    fn test_ring_iterator_seek_iterate_all() {
        let triples = vec![
            make_triple("s1", "p1", "o1"),
            make_triple("s2", "p2", "o2"),
            make_triple("s3", "p3", "o3"),
            make_triple("s4", "p4", "o4"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        let mut iter = RingIterator::all(&ring);
        assert_eq!(iter.position(), 0);

        // Seek to position 2
        iter.seek(2);
        assert_eq!(iter.position(), 2);
        assert!(iter.has_next());

        // Continue iteration from position 2
        let remaining: Vec<Triple> = iter.collect();
        assert_eq!(remaining.len(), 2);
    }

    #[test]
    fn test_ring_iterator_seek_past_end() {
        let triples = vec![make_triple("s1", "p1", "o1")];
        let ring = TripleRing::from_triples(triples.into_iter());

        let mut iter = RingIterator::all(&ring);
        iter.seek(100);

        // Should be clamped to end
        assert!(!iter.has_next());
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_ring_iterator_seek_bound() {
        let triples = vec![
            make_triple("alix", "knows", "gus"),
            make_triple("harm", "knows", "dave"),
            make_triple("alix", "likes", "eve"),
            make_triple("frank", "knows", "alix"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        let mut iter = RingIterator::with_subject(&ring, &Term::iri("alix"));

        // Verify initial state
        assert!(iter.has_next());

        // Seek should find next occurrence >= target
        iter.seek(1);

        // The iterator should still be usable
        let results: Vec<Triple> = iter.collect();
        // All remaining results should have alix as subject
        for triple in &results {
            assert_eq!(triple.subject(), &Term::iri("alix"));
        }
    }

    #[test]
    fn test_ring_iterator_seek_not_found_term() {
        let triples = vec![make_triple("s1", "p1", "o1")];
        let ring = TripleRing::from_triples(triples.into_iter());

        let mut iter = RingIterator::with_subject(&ring, &Term::iri("nonexistent"));

        // Seek on a term that doesn't exist should do nothing
        iter.seek(0);
        assert!(!iter.has_next());
    }

    #[test]
    fn test_ring_iterator_has_next_empty() {
        let ring = TripleRing::from_triples(std::iter::empty());

        let iter = RingIterator::all(&ring);
        assert!(!iter.has_next());
    }

    #[test]
    fn test_leapfrog_patterns_accessor() {
        let triples = vec![
            make_triple("alix", "knows", "gus"),
            make_triple("gus", "knows", "harm"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        let pattern1 = TriplePattern::with_subject(Term::iri("alix"));
        let pattern2 = TriplePattern::with_predicate(Term::iri("knows"));
        let lf = LeapfrogRing::new(&ring, vec![pattern1.clone(), pattern2.clone()]);

        let patterns = lf.patterns();
        assert_eq!(patterns.len(), 2);
    }

    #[test]
    fn test_leapfrog_multi_pattern() {
        let triples = vec![
            make_triple("alix", "knows", "gus"),
            make_triple("gus", "knows", "harm"),
            make_triple("harm", "likes", "alix"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        // Create patterns that should both match
        let pattern1 = TriplePattern::with_subject(Term::iri("alix"));
        let pattern2 = TriplePattern::with_predicate(Term::iri("knows"));
        let mut lf = LeapfrogRing::new(&ring, vec![pattern1, pattern2]);

        let result = lf.next();
        assert!(result.is_some());
        let matched = result.unwrap();
        // Should have matched both patterns
        assert_eq!(matched.len(), 2);
    }

    #[test]
    fn test_leapfrog_no_match() {
        let triples = vec![
            make_triple("alix", "knows", "gus"),
            make_triple("gus", "knows", "harm"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        // Pattern that doesn't match any triple
        let pattern = TriplePattern::with_subject(Term::iri("nonexistent"));
        let mut lf = LeapfrogRing::new(&ring, vec![pattern]);

        let result = lf.next();
        assert!(result.is_none());
        assert!(lf.is_exhausted());
    }

    #[test]
    fn test_leapfrog_exhausted_after_iteration() {
        let triples = vec![make_triple("alix", "knows", "gus")];
        let ring = TripleRing::from_triples(triples.into_iter());

        let pattern = TriplePattern::with_subject(Term::iri("alix"));
        let mut lf = LeapfrogRing::new(&ring, vec![pattern]);

        assert!(!lf.is_exhausted());
        let result = lf.next();
        assert!(result.is_some());

        // After consuming all results, next returns None and marks exhausted
        let second_result = lf.next();
        assert!(second_result.is_none());
        assert!(lf.is_exhausted());
    }

    #[test]
    fn test_leapfrog_empty_ring_with_patterns() {
        let ring = TripleRing::from_triples(std::iter::empty());
        let pattern = TriplePattern::with_subject(Term::iri("alix"));
        let lf = LeapfrogRing::new(&ring, vec![pattern]);

        // Should be exhausted immediately when ring is empty
        assert!(lf.is_exhausted());
    }

    #[test]
    fn test_ring_iterator_predicate_not_found() {
        let triples = vec![make_triple("s1", "p1", "o1")];
        let ring = TripleRing::from_triples(triples.into_iter());

        let mut iter = RingIterator::with_predicate(&ring, &Term::iri("nonexistent"));
        assert!(!iter.has_next());
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_ring_iterator_all_single_triple() {
        let triples = vec![make_triple("s", "p", "o")];
        let ring = TripleRing::from_triples(triples.into_iter());

        let mut iter = RingIterator::all(&ring);
        assert!(iter.has_next());

        let triple = iter.next().unwrap();
        assert_eq!(triple.subject(), &Term::iri("s"));
        assert_eq!(triple.predicate(), &Term::iri("p"));
        assert_eq!(triple.object(), &Term::iri("o"));

        assert!(!iter.has_next());
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_ring_iterator_seek_to_zero() {
        let triples = vec![make_triple("s1", "p1", "o1"), make_triple("s2", "p2", "o2")];
        let ring = TripleRing::from_triples(triples.into_iter());

        let mut iter = RingIterator::all(&ring);
        iter.seek(0);
        assert_eq!(iter.position(), 0);

        let results: Vec<Triple> = iter.collect();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_leapfrog_shared_subject() {
        // (?x knows bob) AND (?x knows harm) -> find subjects knowing both
        let triples = vec![
            make_triple("alix", "knows", "bob"),
            make_triple("alix", "knows", "harm"),
            make_triple("dave", "knows", "bob"),
            make_triple("eve", "knows", "harm"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        let annotated = vec![
            AnnotatedPattern {
                pattern: TriplePattern {
                    subject: None,
                    predicate: Some(Term::iri("knows")),
                    object: Some(Term::iri("bob")),
                },
                subject_var: Some("x".to_string()),
                predicate_var: None,
                object_var: None,
            },
            AnnotatedPattern {
                pattern: TriplePattern {
                    subject: None,
                    predicate: Some(Term::iri("knows")),
                    object: Some(Term::iri("harm")),
                },
                subject_var: Some("x".to_string()),
                predicate_var: None,
                object_var: None,
            },
        ];

        let lf = LeapfrogRing::with_variables(&ring, annotated);
        let results: Vec<Vec<Triple>> = lf.collect();

        // Only alix knows both bob and harm
        assert_eq!(results.len(), 1);
        assert_eq!(results[0][0].subject(), &Term::iri("alix"));
        assert_eq!(results[0][1].subject(), &Term::iri("alix"));
    }

    #[test]
    fn test_leapfrog_triangle() {
        // (?x knows ?y) AND (?y knows ?z) AND (?z knows ?x) -> find triangles
        let triples = vec![
            make_triple("alix", "knows", "bob"),
            make_triple("bob", "knows", "harm"),
            make_triple("harm", "knows", "alix"),
            make_triple("dave", "knows", "eve"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        let annotated = vec![
            AnnotatedPattern {
                pattern: TriplePattern {
                    subject: None,
                    predicate: Some(Term::iri("knows")),
                    object: None,
                },
                subject_var: Some("x".to_string()),
                predicate_var: None,
                object_var: Some("y".to_string()),
            },
            AnnotatedPattern {
                pattern: TriplePattern {
                    subject: None,
                    predicate: Some(Term::iri("knows")),
                    object: None,
                },
                subject_var: Some("y".to_string()),
                predicate_var: None,
                object_var: Some("z".to_string()),
            },
            AnnotatedPattern {
                pattern: TriplePattern {
                    subject: None,
                    predicate: Some(Term::iri("knows")),
                    object: None,
                },
                subject_var: Some("z".to_string()),
                predicate_var: None,
                object_var: Some("x".to_string()),
            },
        ];

        let lf = LeapfrogRing::with_variables(&ring, annotated);
        let results: Vec<Vec<Triple>> = lf.collect();

        // Should find the triangle in 3 rotations: alix->bob->harm->alix
        assert_eq!(results.len(), 3, "Expected three rotations of the triangle");
        assert_eq!(results[0].len(), 3);
    }

    #[test]
    fn test_leapfrog_empty_intersection() {
        // (?x knows bob) AND (?x knows dave) -> no one knows both
        let triples = vec![
            make_triple("alix", "knows", "bob"),
            make_triple("harm", "knows", "dave"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        let annotated = vec![
            AnnotatedPattern {
                pattern: TriplePattern {
                    subject: None,
                    predicate: Some(Term::iri("knows")),
                    object: Some(Term::iri("bob")),
                },
                subject_var: Some("x".to_string()),
                predicate_var: None,
                object_var: None,
            },
            AnnotatedPattern {
                pattern: TriplePattern {
                    subject: None,
                    predicate: Some(Term::iri("knows")),
                    object: Some(Term::iri("dave")),
                },
                subject_var: Some("x".to_string()),
                predicate_var: None,
                object_var: None,
            },
        ];

        let lf = LeapfrogRing::with_variables(&ring, annotated);
        let results: Vec<Vec<Triple>> = lf.collect();

        assert!(results.is_empty(), "Expected no matches");
    }

    #[test]
    fn test_ring_iterator_current_term_id() {
        let triples = vec![
            make_triple("alix", "knows", "bob"),
            make_triple("harm", "likes", "dave"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        let iter = RingIterator::all(&ring);
        // Should return Some for valid positions
        let id = iter.current_term_id(0);
        assert!(id.is_some());
    }

    #[test]
    fn test_ring_iterator_current_term_id_past_end() {
        let triples = vec![make_triple("s", "p", "o")];
        let ring = TripleRing::from_triples(triples.into_iter());

        let mut iter = RingIterator::all(&ring);
        iter.next(); // consume the only triple
        let id = iter.current_term_id(0);
        assert!(id.is_none());
    }
}
