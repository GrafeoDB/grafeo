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
//! ```no_run
//! # use grafeo_core::index::ring::TripleRing;
//! # use grafeo_core::graph::rdf::{Triple, Term, TriplePattern};
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
//!     println!("{:?}", triple);
//! }
//! ```
//!
//! # References
//!
//! - Álvarez-García et al., "Compressed Vertical Partitioning for Efficient RDF Management"
//! - MillenniumDB Ring implementation

mod leapfrog;
pub mod packed_dict;
pub mod packed_format;
pub mod packed_permutation;
pub mod packed_wavelet;
mod permutation;
pub mod section;
pub mod triple_ring;

pub use leapfrog::{AnnotatedPattern, LeapfrogRing, RingIterator};
pub use packed_dict::{PackedDictError, PackedTermDictionary};
pub use packed_format::{
    PackedRingError, assemble_triple_ring, deserialize_triple_ring, serialize_triple_ring,
};
pub use packed_permutation::{
    PackedPermutationError, deserialize_permutation, serialize_permutation, write_permutation,
};
pub use packed_wavelet::{
    PackedWaveletError, deserialize_wavelet_tree, serialize_wavelet_tree, write_wavelet_tree,
};
pub use permutation::SuccinctPermutation;
pub use section::RdfRingSection;
pub use triple_ring::{TripleRing, TripleRingInvariantError};
