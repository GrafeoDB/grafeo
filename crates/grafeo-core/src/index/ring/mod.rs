//! Ring Index for RDF triple storage.
//!
//! The Ring Index is a compact representation for RDF triples that achieves
//! approximately 3x space reduction compared to traditional triple indexing
//! using HashMaps.
//!
//! # Design
//!
//! Instead of maintaining separate SPO, POS, and OSP indexes with redundant
//! storage, the Ring Index uses:
//!
//! 1. **Term Dictionary**: Maps RDF terms to compact integer IDs
//! 2. **Wavelet Trees**: One for each component (subject, predicate, object)
//! 3. **Succinct Permutations**: Navigate between orderings without duplicating data
//!
//! # Space Comparison (1M triples)
//!
//! | Approach | Size |
//! |----------|------|
//! | 3 HashMaps | ~120 MB |
//! | Ring Index | ~40 MB |
//! | **Savings** | **3x** |
//!
//! # Example
//!
//! ```ignore
//! use grafeo_core::index::ring::TripleRing;
//! use grafeo_core::graph::rdf::{Triple, Term, TriplePattern};
//!
//! // Build from triples
//! let triples = vec![
//!     Triple::new(Term::iri("s1"), Term::iri("p1"), Term::iri("o1")),
//!     Triple::new(Term::iri("s1"), Term::iri("p2"), Term::iri("o2")),
//! ];
//! let ring = TripleRing::from_triples(triples.into_iter());
//!
//! // Query by pattern
//! let pattern = TriplePattern::with_subject(Term::iri("s1"));
//! for triple in ring.find(&pattern) {
//!     println!("{}", triple);
//! }
//! ```
//!
//! # References
//!
//! - Álvarez-García et al., "Compressed Vertical Partitioning for Efficient RDF Management"
//! - MillenniumDB Ring implementation

mod leapfrog;
mod permutation;
mod triple_ring;

pub use leapfrog::{LeapfrogRing, RingIterator};
pub use permutation::SuccinctPermutation;
pub use triple_ring::TripleRing;
