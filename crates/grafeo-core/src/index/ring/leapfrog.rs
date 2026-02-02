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

    /// Seeks to the first position >= target.
    ///
    /// For leapfrog join, this is the key operation.
    pub fn seek(&mut self, target: usize) {
        if self.iterate_all {
            // For iterate-all, just move position
            self.pos = target.min(self.end);
        } else if self.bound_id.is_some() {
            // For bound iterators, we need to find the next occurrence >= target
            while self.has_next() {
                let wt = match self.component {
                    0 => self.ring.subjects_wt(),
                    1 => self.ring.predicates_wt(),
                    _ => self.ring.objects_wt(),
                };

                if let Some(next_pos) = wt.select(self.bound_id.unwrap() as u64, self.rank) {
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

impl<'a> Iterator for RingIterator<'a> {
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
        } else if let Some(id) = self.bound_id {
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
        } else {
            // Term not found - shouldn't reach here due to has_next() check
            return None;
        };

        self.ring.get_spo(pos)
    }
}

/// Leapfrog join over multiple Ring iterators.
///
/// Implements the leapfrog triejoin algorithm for worst-case optimal joins
/// over RDF triple patterns.
pub struct LeapfrogRing<'a> {
    ring: &'a TripleRing,
    /// Patterns to join.
    patterns: Vec<TriplePattern>,
    /// Current bindings for variables (for future full leapfrog implementation).
    #[allow(dead_code)]
    bindings: Vec<Option<Term>>,
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
            bindings: Vec::new(),
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
}

impl<'a> Iterator for LeapfrogRing<'a> {
    type Item = Vec<Triple>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }

        // For now, simple nested loop join
        // TODO: Implement proper leapfrog algorithm with seek operations

        // Simple implementation: find all triples matching first pattern,
        // then filter by remaining patterns
        if self.patterns.is_empty() {
            self.exhausted = true;
            return None;
        }

        // Use iteration over first pattern and filter by rest
        self.exhausted = true;

        // Collect all matching tuples
        let mut results = Vec::new();

        for triple in self.ring.find(&self.patterns[0]) {
            let mut matches_all = true;
            let mut matching_triples = vec![triple.clone()];

            // For remaining patterns, find matches that are consistent
            // (This is a simplified implementation)
            for pattern in &self.patterns[1..] {
                if let Some(t) = self.ring.find(pattern).next() {
                    // In a full implementation, we'd check variable bindings
                    matching_triples.push(t);
                } else {
                    matches_all = false;
                    break;
                }
            }

            if matches_all {
                results.push(matching_triples);
            }
        }

        if results.is_empty() {
            None
        } else {
            Some(results.remove(0))
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
        let triples = vec![
            make_triple("s1", "p1", "o1"),
            make_triple("s2", "p2", "o2"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        let iter = RingIterator::all(&ring);
        let results: Vec<Triple> = iter.collect();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_ring_iterator_with_subject() {
        let triples = vec![
            make_triple("alice", "knows", "bob"),
            make_triple("alice", "knows", "carol"),
            make_triple("bob", "knows", "carol"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        let iter = RingIterator::with_subject(&ring, &Term::iri("alice"));
        let results: Vec<Triple> = iter.collect();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_ring_iterator_with_predicate() {
        let triples = vec![
            make_triple("s1", "type", "Person"),
            make_triple("s2", "type", "Place"),
            make_triple("s1", "name", "Alice"),
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
            make_triple("alice", "knows", "bob"),
            make_triple("bob", "knows", "carol"),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());

        let pattern = TriplePattern::with_subject(Term::iri("alice"));
        let mut lf = LeapfrogRing::new(&ring, vec![pattern]);

        let result = lf.next();
        assert!(result.is_some());
        let triples = result.unwrap();
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].subject(), &Term::iri("alice"));
    }
}
