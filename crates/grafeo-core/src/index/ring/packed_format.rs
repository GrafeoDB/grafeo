//! Ring index v2 packed on-disk format (Phase 6e).
//!
//! Composes the four packed sub-formats from Phase 6a-d into a single
//! mmap-friendly byte buffer:
//!
//! - [`PackedTermDictionary`] (Phase 6b)
//! - `PackedWaveletTree` (see [`super::packed_wavelet`]) for subjects,
//!   predicates, objects (Phase 6c)
//! - `PackedPermutation` (see [`super::packed_permutation`]) for spo→pos
//!   and spo→osp (Phase 6d)
//!
//! ## Layout
//!
//! ```text
//! Header (64 bytes):
//!     0..4    magic "GRFR"
//!     4       version u8 = 2
//!     5..8    reserved (3 bytes, zero)
//!     8..16   num_triples u64 LE
//!     16..24  dict_offset u64 LE
//!     24..32  subjects_offset u64 LE
//!     32..40  predicates_offset u64 LE
//!     40..48  objects_offset u64 LE
//!     48..56  spo_to_pos_offset u64 LE
//!     56..64  spo_to_osp_offset u64 LE
//!
//! sub-sections (laid out at their declared offsets, each carries its own
//! magic + header):
//!     PackedTermDictionary (PDCT)
//!     PackedWaveletTree subjects (WTRE)
//!     PackedWaveletTree predicates (WTRE)
//!     PackedWaveletTree objects (WTRE)
//!     PackedPermutation spo_to_pos (PERM)
//!     PackedPermutation spo_to_osp (PERM)
//!
//! trailer (4 bytes):
//!     CRC32 LE of bytes [0..end-4]
//! ```
//!
//! Explicit offsets (rather than sequential parsing) let an mmap reader
//! `Bytes::slice` directly to any sub-section without first walking the
//! preceding ones.
//!
//! The six sub-sections are also the parts of the ring section of version 3
//! (see [`super::section`]), which stores each in its own stream without the
//! envelope; [`assemble_triple_ring`] builds the ring from the six parts
//! however they were read.

use bytes::Bytes;

use crate::index::ring::{
    PackedDictError, PackedPermutationError, PackedTermDictionary, PackedWaveletError,
    SuccinctPermutation, TripleRing, deserialize_permutation, deserialize_wavelet_tree,
    serialize_permutation, serialize_wavelet_tree,
};

const MAGIC: &[u8; 4] = b"GRFR";
const VERSION: u8 = 2;
const HEADER_SIZE: usize = 64;
const TRAILER_SIZE: usize = 4;

/// Errors returned when parsing a Ring v2 packed file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackedRingError {
    /// Buffer is too small to even read the header.
    TruncatedHeader,
    /// First 4 bytes don't match "GRFR".
    BadMagic,
    /// Version byte not recognized (only `2` is accepted).
    UnsupportedVersion(u8),
    /// A declared sub-section offset points outside the buffer.
    OffsetOutOfBounds {
        /// Sub-section that had the bad offset.
        section: &'static str,
        /// The declared offset.
        offset: u64,
    },
    /// The file's CRC32 trailer doesn't match the computed CRC.
    ChecksumMismatch {
        /// CRC32 the trailer claims.
        expected: u32,
        /// CRC32 computed from the bytes.
        actual: u32,
    },
    /// Embedded `PackedTermDictionary` failed to parse.
    Dict(PackedDictError),
    /// Embedded `PackedWaveletTree` failed to parse.
    Wavelet {
        /// Which tree: "subjects", "predicates" or "objects".
        part: &'static str,
        /// Why it does not parse.
        error: PackedWaveletError,
    },
    /// Embedded `PackedPermutation` failed to parse.
    Permutation {
        /// Which permutation: "spo_to_pos" or "spo_to_osp".
        part: &'static str,
        /// Why it does not parse.
        error: PackedPermutationError,
    },
    /// `num_triples` declared in the header doesn't match the rebuilt
    /// permutations / wavelets.
    NumTriplesMismatch {
        /// Value declared in the header.
        declared: usize,
        /// Value implied by the parsed sub-sections.
        observed: usize,
    },
    /// Reconstructed sub-components violated a [`TripleRing`]
    /// structural invariant — caught here rather than letting the ring
    /// panic on a later query.
    RingInvariantViolation(super::triple_ring::TripleRingInvariantError),
}

impl std::fmt::Display for PackedRingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TruncatedHeader => write!(f, "ring v2 header truncated"),
            Self::BadMagic => write!(f, "ring v2 bad magic (expected 'GRFR')"),
            Self::UnsupportedVersion(v) => write!(f, "ring v2 unsupported version {v}"),
            Self::OffsetOutOfBounds { section, offset } => write!(
                f,
                "ring v2 offset out of bounds: section '{section}' at {offset}"
            ),
            Self::ChecksumMismatch { expected, actual } => write!(
                f,
                "ring v2 CRC mismatch: expected {expected:#010X}, got {actual:#010X}"
            ),
            Self::Dict(e) => write!(f, "ring v2 dictionary parse error: {e}"),
            Self::Wavelet { part, error } => {
                write!(f, "ring v2 {part} wavelet parse error: {error}")
            }
            Self::Permutation { part, error } => {
                write!(f, "ring v2 {part} permutation parse error: {error}")
            }
            Self::NumTriplesMismatch { declared, observed } => write!(
                f,
                "ring v2 num_triples mismatch: declared {declared}, observed {observed}"
            ),
            Self::RingInvariantViolation(e) => {
                write!(f, "ring v2 ring invariant violation: {e}")
            }
        }
    }
}

impl std::error::Error for PackedRingError {}

impl From<PackedDictError> for PackedRingError {
    fn from(e: PackedDictError) -> Self {
        Self::Dict(e)
    }
}

impl From<super::triple_ring::TripleRingInvariantError> for PackedRingError {
    fn from(e: super::triple_ring::TripleRingInvariantError) -> Self {
        Self::RingInvariantViolation(e)
    }
}

/// Serializes a [`TripleRing`] to the v2 packed format.
///
/// The output buffer is laid out per the module-top documentation: a
/// 64-byte header with explicit sub-section offsets, the six packed
/// sub-sections in order, then a 4-byte CRC32 trailer.
#[must_use]
pub fn serialize_triple_ring(ring: &TripleRing) -> Vec<u8> {
    // Serialize each sub-section first so we know their sizes for the
    // offset table.
    let dict_bytes = PackedTermDictionary::from_term_dict(ring.dictionary()).to_bytes();
    let subj_bytes = serialize_wavelet_tree(ring.subjects_wt());
    let pred_bytes = serialize_wavelet_tree(ring.predicates_wt());
    let obj_bytes = serialize_wavelet_tree(ring.objects_wt());
    let pos_bytes = serialize_permutation(ring.spo_to_pos_perm());
    let osp_bytes = serialize_permutation(ring.spo_to_osp_perm());

    let dict_offset = HEADER_SIZE as u64;
    let subj_offset = dict_offset + dict_bytes.len() as u64;
    let pred_offset = subj_offset + subj_bytes.len() as u64;
    let obj_offset = pred_offset + pred_bytes.len() as u64;
    let pos_offset = obj_offset + obj_bytes.len() as u64;
    let osp_offset = pos_offset + pos_bytes.len() as u64;
    let body_end = osp_offset + osp_bytes.len() as u64;

    let total = usize::try_from(body_end).unwrap_or(usize::MAX) + TRAILER_SIZE;
    let mut buf = Vec::with_capacity(total);

    // Header.
    buf.extend_from_slice(MAGIC); // 0..4
    buf.push(VERSION); // 4
    buf.extend_from_slice(&[0u8; 3]); // 5..8 reserved
    buf.extend_from_slice(&(ring.len() as u64).to_le_bytes()); // 8..16 num_triples
    buf.extend_from_slice(&dict_offset.to_le_bytes()); // 16..24
    buf.extend_from_slice(&subj_offset.to_le_bytes()); // 24..32
    buf.extend_from_slice(&pred_offset.to_le_bytes()); // 32..40
    buf.extend_from_slice(&obj_offset.to_le_bytes()); // 40..48
    buf.extend_from_slice(&pos_offset.to_le_bytes()); // 48..56
    buf.extend_from_slice(&osp_offset.to_le_bytes()); // 56..64

    // Sub-sections at their declared offsets.
    buf.extend_from_slice(&dict_bytes);
    buf.extend_from_slice(&subj_bytes);
    buf.extend_from_slice(&pred_bytes);
    buf.extend_from_slice(&obj_bytes);
    buf.extend_from_slice(&pos_bytes);
    buf.extend_from_slice(&osp_bytes);

    // Trailer: CRC32 of everything above.
    let crc = crc32fast::hash(&buf);
    buf.extend_from_slice(&crc.to_le_bytes());

    buf
}

/// Parses a [`TripleRing`] from the v2 packed format.
///
/// `data` is consumed via [`Bytes::slice`] so each sub-section's
/// allocation is shared with the caller. Mmap-backed buffers stay
/// zero-copy through term-dict + per-level-bitvector reconstruction.
///
/// # Errors
///
/// Returns a [`PackedRingError`] on any of: truncation, magic/version
/// mismatch, out-of-bounds sub-section offsets, CRC trailer mismatch,
/// or any of the embedded sub-format errors propagating up.
///
/// # Panics
///
/// Internal `expect` calls describe invariants the bounds checks above
/// already guarantee — every indexed read is preceded by a length
/// check. Does not panic in normal operation.
pub fn deserialize_triple_ring(data: Bytes) -> Result<TripleRing, PackedRingError> {
    if data.len() < HEADER_SIZE + TRAILER_SIZE {
        return Err(PackedRingError::TruncatedHeader);
    }
    if &data[0..4] != MAGIC {
        return Err(PackedRingError::BadMagic);
    }
    let version = data[4];
    if version != VERSION {
        return Err(PackedRingError::UnsupportedVersion(version));
    }

    // CRC trailer first — fail fast on corruption.
    let body_end = data.len() - TRAILER_SIZE;
    let trailer: [u8; 4] = data[body_end..].try_into().expect("4-byte trailer slice");
    let expected_crc = u32::from_le_bytes(trailer);
    let actual_crc = crc32fast::hash(&data[..body_end]);
    if actual_crc != expected_crc {
        return Err(PackedRingError::ChecksumMismatch {
            expected: expected_crc,
            actual: actual_crc,
        });
    }

    let num_triples_raw = u64::from_le_bytes(data[8..16].try_into().expect("8-byte slice"));
    let num_triples =
        usize::try_from(num_triples_raw).map_err(|_| PackedRingError::OffsetOutOfBounds {
            section: "num_triples",
            offset: num_triples_raw,
        })?;
    let dict_offset = u64::from_le_bytes(data[16..24].try_into().expect("8-byte slice"));
    let subj_offset = u64::from_le_bytes(data[24..32].try_into().expect("8-byte slice"));
    let pred_offset = u64::from_le_bytes(data[32..40].try_into().expect("8-byte slice"));
    let obj_offset = u64::from_le_bytes(data[40..48].try_into().expect("8-byte slice"));
    let pos_offset = u64::from_le_bytes(data[48..56].try_into().expect("8-byte slice"));
    let osp_offset = u64::from_le_bytes(data[56..64].try_into().expect("8-byte slice"));

    let body_end_u64 = body_end as u64;
    for (section, offset) in [
        ("dict", dict_offset),
        ("subjects", subj_offset),
        ("predicates", pred_offset),
        ("objects", obj_offset),
        ("spo_to_pos", pos_offset),
        ("spo_to_osp", osp_offset),
    ] {
        if offset > body_end_u64 {
            return Err(PackedRingError::OffsetOutOfBounds { section, offset });
        }
    }
    // Offsets must be strictly increasing.
    let offsets = [
        dict_offset,
        subj_offset,
        pred_offset,
        obj_offset,
        pos_offset,
        osp_offset,
    ];
    for window in offsets.windows(2) {
        if window[0] >= window[1] {
            return Err(PackedRingError::OffsetOutOfBounds {
                section: "(non-monotonic offsets)",
                offset: window[1],
            });
        }
    }

    // Slice each region. Lengths are: end-of-this to start-of-next, with
    // the last one going to body_end.
    let to_usize = |o: u64| -> Result<usize, PackedRingError> {
        usize::try_from(o).map_err(|_| PackedRingError::OffsetOutOfBounds {
            section: "(offset overflow)",
            offset: o,
        })
    };
    let dict_slice = data.slice(to_usize(dict_offset)?..to_usize(subj_offset)?);
    let subj_slice = data.slice(to_usize(subj_offset)?..to_usize(pred_offset)?);
    let pred_slice = data.slice(to_usize(pred_offset)?..to_usize(obj_offset)?);
    let obj_slice = data.slice(to_usize(obj_offset)?..to_usize(pos_offset)?);
    let pos_slice = data.slice(to_usize(pos_offset)?..to_usize(osp_offset)?);
    let osp_slice = data.slice(to_usize(osp_offset)?..body_end);

    assemble_triple_ring(
        num_triples,
        dict_slice,
        subj_slice,
        pred_slice,
        obj_slice,
        pos_slice,
        osp_slice,
    )
}

/// The ring from its six parts, each in its packed format.
///
/// The parts are those the v2 envelope holds at its offsets: the
/// [`PackedTermDictionary`], the wavelet trees of the subjects, predicates
/// and objects, and the permutations from SPO to POS and to OSP order. Each
/// part's buffer becomes its storage where the part keeps its bytes (the
/// dictionary until the ring has read its terms, the level bits of the
/// wavelet trees).
///
/// # Errors
///
/// Returns a [`PackedRingError`] when a part does not parse, when the
/// subjects do not hold `num_triples` entries, or when the parts break an
/// invariant of the ring (see [`TripleRing::from_packed_parts`]).
pub fn assemble_triple_ring(
    num_triples: usize,
    dictionary: Bytes,
    subjects: Bytes,
    predicates: Bytes,
    objects: Bytes,
    spo_to_pos: Bytes,
    spo_to_osp: Bytes,
) -> Result<TripleRing, PackedRingError> {
    let tree = |part: &'static str, bytes: Bytes| {
        deserialize_wavelet_tree(bytes).map_err(|error| PackedRingError::Wavelet { part, error })
    };
    let permutation = |part: &'static str, bytes: Bytes| {
        deserialize_permutation(bytes).map_err(|error| PackedRingError::Permutation { part, error })
    };
    let dict = PackedTermDictionary::from_bytes(dictionary)?;
    let subjects = tree("subjects", subjects)?;
    let predicates = tree("predicates", predicates)?;
    let objects = tree("objects", objects)?;
    let spo_to_pos: SuccinctPermutation = permutation("spo_to_pos", spo_to_pos)?;
    let spo_to_osp: SuccinctPermutation = permutation("spo_to_osp", spo_to_osp)?;

    if subjects.len() != num_triples {
        return Err(PackedRingError::NumTriplesMismatch {
            declared: num_triples,
            observed: subjects.len(),
        });
    }

    TripleRing::from_packed_parts(
        dict,
        num_triples,
        subjects,
        predicates,
        objects,
        spo_to_pos,
        spo_to_osp,
    )
    .map_err(PackedRingError::RingInvariantViolation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::rdf::{Term, Triple, TriplePattern};

    fn build_test_ring() -> TripleRing {
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
        TripleRing::from_triples(triples.into_iter())
    }

    #[test]
    fn alix_packed_ring_roundtrip() {
        let ring = build_test_ring();
        let bytes = serialize_triple_ring(&ring);
        let restored = deserialize_triple_ring(Bytes::from(bytes)).expect("deserialize");

        assert_eq!(restored.len(), ring.len());
        assert_eq!(restored.num_terms(), ring.num_terms());

        // Query equivalence: foaf:name predicate should match 2 triples.
        let pattern = TriplePattern {
            subject: None,
            predicate: Some(Term::iri("http://xmlns.com/foaf/0.1/name")),
            object: None,
        };
        assert_eq!(restored.count(&pattern), ring.count(&pattern));
        assert_eq!(restored.count(&pattern), 2);
    }

    /// The bytes of the version 2 envelope, whose parts the streams of
    /// version 3 hold, do not change: the length and the CRC32 of the body
    /// (the trailer) of two envelopes. The format is that of 0.5.x; since
    /// 0.6.0 the term ids follow the byte order of the terms, which changed
    /// the bytes of the three-triple ring (the empty one kept its 0.5.x
    /// bytes; `a_pinned_0_5_envelope_reads_through_read_from` in the
    /// section's tests holds an envelope as 0.5.x wrote it). (The CRC32 of
    /// a whole envelope is the same for every envelope: a CRC over bytes and
    /// their own CRC is a constant.)
    #[test]
    fn the_packed_ring_bytes_are_unchanged() {
        let body_crc = |bytes: &[u8]| crc32fast::hash(&bytes[..bytes.len() - TRAILER_SIZE]);
        let bytes = serialize_triple_ring(&build_test_ring());
        let empty = serialize_triple_ring(&TripleRing::from_triples(std::iter::empty()));
        assert_eq!(
            (bytes.len(), body_crc(&bytes)),
            (615, 0xDD83_1D12),
            "three triples"
        );
        assert_eq!((empty.len(), body_crc(&empty)), (252, 0x292E_F3EF), "empty");
    }

    /// The six parts of an envelope, given apart, assemble into the ring the
    /// envelope holds; another triple count is refused.
    #[test]
    fn the_parts_of_an_envelope_assemble_into_its_ring() {
        let ring = build_test_ring();
        let bytes = Bytes::from(serialize_triple_ring(&ring));
        let offset = |at: usize| {
            usize::try_from(u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())).unwrap()
        };
        let bounds = [
            offset(16),
            offset(24),
            offset(32),
            offset(40),
            offset(48),
            offset(56),
            bytes.len() - TRAILER_SIZE,
        ];
        let part = |index: usize| bytes.slice(bounds[index]..bounds[index + 1]);
        let assemble = |num_triples| {
            assemble_triple_ring(
                num_triples,
                part(0),
                part(1),
                part(2),
                part(3),
                part(4),
                part(5),
            )
        };
        let restored = assemble(3).expect("the parts assemble");
        let pattern = TriplePattern {
            subject: Some(Term::iri("http://ex.org/alix")),
            predicate: None,
            object: None,
        };
        assert_eq!(restored.len(), 3);
        assert_eq!(restored.count(&pattern), 2);
        for index in 0..3 {
            assert_eq!(
                restored.get_spo(index),
                ring.get_spo(index),
                "triple {index}"
            );
        }
        assert_eq!(
            assemble(19).unwrap_err(),
            PackedRingError::NumTriplesMismatch {
                declared: 19,
                observed: 3
            }
        );
        assert!(
            matches!(
                assemble_triple_ring(3, part(1), part(1), part(2), part(3), part(4), part(5)),
                Err(PackedRingError::Dict(PackedDictError::BadMagic))
            ),
            "a wavelet tree where the dictionary belongs"
        );
    }

    #[test]
    fn gus_packed_ring_empty() {
        let ring = TripleRing::from_triples(std::iter::empty());
        let bytes = serialize_triple_ring(&ring);
        let restored = deserialize_triple_ring(Bytes::from(bytes)).expect("deserialize empty");
        assert_eq!(restored.len(), 0);
        assert!(restored.is_empty());
    }

    #[test]
    fn vincent_packed_ring_bad_magic_rejected() {
        let bad = Bytes::from(vec![0u8; HEADER_SIZE + TRAILER_SIZE]);
        assert_eq!(
            deserialize_triple_ring(bad).unwrap_err(),
            PackedRingError::BadMagic
        );
    }

    #[test]
    fn jules_packed_ring_truncated_header_rejected() {
        let short = Bytes::from(vec![b'G', b'R', b'F', b'R']);
        assert_eq!(
            deserialize_triple_ring(short).unwrap_err(),
            PackedRingError::TruncatedHeader
        );
    }

    #[test]
    fn mia_packed_ring_unsupported_version_rejected() {
        // Build a header that has the right magic but a wrong version
        // byte. The trailer doesn't matter for this assertion path
        // because version is checked before CRC.
        let mut buf = vec![0u8; HEADER_SIZE + TRAILER_SIZE];
        buf[..4].copy_from_slice(MAGIC);
        buf[4] = 99;
        let crc = crc32fast::hash(&buf[..buf.len() - TRAILER_SIZE]);
        let len = buf.len();
        buf[len - 4..].copy_from_slice(&crc.to_le_bytes());
        assert_eq!(
            deserialize_triple_ring(Bytes::from(buf)).unwrap_err(),
            PackedRingError::UnsupportedVersion(99)
        );
    }

    #[test]
    fn shosanna_packed_ring_corrupted_byte_caught_by_crc() {
        let ring = build_test_ring();
        let mut bytes = serialize_triple_ring(&ring);
        // Flip a byte deep inside the body.
        let len = bytes.len();
        bytes[len / 2] ^= 0x01;
        let result = deserialize_triple_ring(Bytes::from(bytes));
        assert!(matches!(
            result.unwrap_err(),
            PackedRingError::ChecksumMismatch { .. }
        ));
    }

    #[test]
    fn beatrix_packed_ring_query_correctness() {
        // Build a richer graph and verify several query patterns
        // round-trip correctly.
        let triples = vec![
            Triple::new(
                Term::iri("http://ex.org/a"),
                Term::iri("http://ex.org/p"),
                Term::iri("http://ex.org/x"),
            ),
            Triple::new(
                Term::iri("http://ex.org/a"),
                Term::iri("http://ex.org/p"),
                Term::iri("http://ex.org/y"),
            ),
            Triple::new(
                Term::iri("http://ex.org/b"),
                Term::iri("http://ex.org/p"),
                Term::iri("http://ex.org/x"),
            ),
            Triple::new(
                Term::iri("http://ex.org/b"),
                Term::iri("http://ex.org/q"),
                Term::iri("http://ex.org/y"),
            ),
        ];
        let ring = TripleRing::from_triples(triples.into_iter());
        let bytes = serialize_triple_ring(&ring);
        let restored = deserialize_triple_ring(Bytes::from(bytes)).expect("deserialize");

        // All-? pattern: count every triple.
        let all = TriplePattern {
            subject: None,
            predicate: None,
            object: None,
        };
        assert_eq!(restored.count(&all), 4);

        // ?, p, ?: 3 matches.
        let p_only = TriplePattern {
            subject: None,
            predicate: Some(Term::iri("http://ex.org/p")),
            object: None,
        };
        assert_eq!(restored.count(&p_only), 3);
        assert_eq!(restored.count(&p_only), ring.count(&p_only));

        // ?, ?, x: 2 matches.
        let x_only = TriplePattern {
            subject: None,
            predicate: None,
            object: Some(Term::iri("http://ex.org/x")),
        };
        assert_eq!(restored.count(&x_only), 2);

        // a, p, ?: 2 matches.
        let a_p = TriplePattern {
            subject: Some(Term::iri("http://ex.org/a")),
            predicate: Some(Term::iri("http://ex.org/p")),
            object: None,
        };
        assert_eq!(restored.count(&a_p), 2);
    }

    #[test]
    fn hans_packed_ring_size_smaller_than_bincode() {
        // Build a ring big enough for v2 to clearly win.
        let triples: Vec<Triple> = (0..100u32)
            .flat_map(|i| {
                let s = format!("http://ex.org/s-{i}");
                (0..5u32).map(move |j| {
                    Triple::new(
                        Term::iri(s.clone()),
                        Term::iri(format!("http://ex.org/p-{}", j)),
                        Term::literal(format!("value-{}-{}", i, j)),
                    )
                })
            })
            .collect();
        let ring = TripleRing::from_triples(triples.into_iter());
        let v2_bytes = serialize_triple_ring(&ring);
        let v1_bytes = ring.save_to_bytes().expect("v1 save");
        eprintln!(
            "v1 bincode: {} bytes, v2 packed: {} bytes",
            v1_bytes.len(),
            v2_bytes.len()
        );
        assert!(
            v2_bytes.len() < v1_bytes.len(),
            "v2 packed must be smaller than v1 bincode (v1={}, v2={})",
            v1_bytes.len(),
            v2_bytes.len()
        );
    }
}
