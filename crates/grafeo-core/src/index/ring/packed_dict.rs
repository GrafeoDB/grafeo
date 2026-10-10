//! Packed term dictionary for the v2 Ring on-disk format (Phase 6b).
//!
//! Stores RDF terms as concatenated UTF-8 N-Triples strings in a refcounted
//! [`Bytes`] buffer, with an offset index for O(1) `id -> term` lookups and
//! a sorted-id permutation for O(log n) `term -> id` lookups via binary
//! search.
//!
//! The packed format is designed to mmap directly: the entire structure
//! is three contiguous byte slices (`string_table`, `offsets`, `sorted_ids`)
//! all stored as little-endian primitives. No allocation on load — the
//! [`from_bytes`](PackedTermDictionary::from_bytes) entry point adopts
//! pre-mapped memory via `Bytes::slice`.
//!
//! ## Layout
//!
//! ```text
//! Header (16 bytes):
//!     magic: 4 bytes, "PDCT"
//!     version: u8
//!     reserved: 3 bytes, zero
//!     count: u64 LE
//!
//! string_table_size: u64 LE (one record header per region)
//! string_table: variable bytes (UTF-8, in insertion order)
//! offsets: (count + 1) * 8 bytes (u64 LE, sentinel at end)
//! sorted_ids: count * 4 bytes (u32 LE, lex-sorted permutation)
//! ```
//!
//! ## Why insertion order + sort permutation
//!
//! The wavelet trees and permutations elsewhere in the Ring index reference
//! terms by their original insertion-order IDs. Sorting the dictionary
//! would require rebuilding those structures — expensive at migration
//! time. Instead, we keep IDs stable and store an auxiliary `sorted_ids`
//! array so query-time `get_id` lookups remain O(log n).

use std::io::Write;

use bytes::Bytes;

use crate::graph::rdf::Term;
use crate::index::ring::triple_ring::TermDictionary;

const MAGIC: &[u8; 4] = b"PDCT";
const VERSION: u8 = 1;
const HEADER_SIZE: usize = 16;

/// Packed dictionary in the v2 Ring on-disk format (Phase 6b).
#[derive(Debug, Clone)]
pub struct PackedTermDictionary {
    /// Concatenated UTF-8 N-Triples encodings in insertion order.
    string_table: Bytes,
    /// Byte offsets into `string_table`, length `count + 1`. Sentinel at
    /// `offsets[count] == string_table.len()`. Each entry is a u64 LE.
    offsets: Bytes,
    /// Ids sorted by their term's lexicographic order. Each entry u32 LE.
    /// `sorted_ids.len() == count`.
    sorted_ids: Bytes,
    /// Number of terms.
    count: usize,
}

/// Errors returned when parsing a packed term dictionary from bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackedDictError {
    /// Buffer is too short to contain even the fixed-size header.
    TruncatedHeader,
    /// First 4 bytes don't match "PDCT".
    BadMagic,
    /// Version byte not recognized.
    UnsupportedVersion(u8),
    /// Recorded sizes overflow the input buffer.
    InconsistentSizes {
        /// Total bytes the header claims the dictionary occupies.
        expected_total: usize,
        /// Total bytes actually available in the input buffer.
        actual_total: usize,
    },
    /// Offsets array is not strictly non-decreasing or sentinel mismatches
    /// `string_table` length.
    InvalidOffsets,
    /// Sorted-id entry references an id outside `[0, count)`.
    InvalidSortedId(u32),
}

impl std::fmt::Display for PackedDictError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TruncatedHeader => write!(f, "packed dict header truncated"),
            Self::BadMagic => write!(f, "packed dict bad magic (expected 'PDCT')"),
            Self::UnsupportedVersion(v) => write!(f, "packed dict unsupported version {v}"),
            Self::InconsistentSizes {
                expected_total,
                actual_total,
            } => write!(
                f,
                "packed dict size mismatch: expected {expected_total} bytes, got {actual_total}"
            ),
            Self::InvalidOffsets => write!(
                f,
                "packed dict offsets invalid (non-monotonic or sentinel mismatch)"
            ),
            Self::InvalidSortedId(id) => write!(f, "packed dict sorted-id {id} out of range"),
        }
    }
}

impl std::error::Error for PackedDictError {}

impl PackedTermDictionary {
    /// Number of terms.
    #[must_use]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns whether the dictionary is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Returns the N-Triples-encoded string for the given id, or `None`
    /// if the id is out of range.
    ///
    /// O(1): two `u64` reads from the offsets array, then a `Bytes::get`
    /// on the slab.
    #[must_use]
    pub fn get_term_str(&self, id: u32) -> Option<&str> {
        let i = id as usize;
        if i >= self.count {
            return None;
        }
        let start = usize::try_from(read_u64_at(&self.offsets, i)?).ok()?;
        let end = usize::try_from(read_u64_at(&self.offsets, i + 1)?).ok()?;
        let bytes = self.string_table.get(start..end)?;
        std::str::from_utf8(bytes).ok()
    }

    /// Returns the parsed [`Term`] for the given id, or `None` if the id is
    /// out of range or its string is not an N-Triples term.
    ///
    /// Combines [`get_term_str`](Self::get_term_str) with N-Triples
    /// parsing. Allocates one `Term`.
    #[must_use]
    pub fn get_term(&self, id: u32) -> Option<Term> {
        let s = self.get_term_str(id)?;
        Term::from_ntriples(s).ok()
    }

    /// Returns the id of the term whose N-Triples encoding equals `s`,
    /// or `None` if no such term exists.
    ///
    /// O(log n) via binary search on `sorted_ids`. Each comparison costs
    /// one offsets-array lookup + one string slice, no allocation.
    #[must_use]
    pub fn get_id_by_str(&self, s: &str) -> Option<u32> {
        if self.count == 0 {
            return None;
        }
        let target = s.as_bytes();
        let mut lo = 0usize;
        let mut hi = self.count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let id = read_u32_at(&self.sorted_ids, mid)?;
            let candidate = self.term_bytes_for(id)?;
            match candidate.cmp(target) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(id),
            }
        }
        None
    }

    /// Returns the id of the given term, or `None` if not present.
    #[must_use]
    pub fn get_id(&self, term: &Term) -> Option<u32> {
        let s = term.to_string();
        self.get_id_by_str(&s)
    }

    /// Builds a [`PackedTermDictionary`] from an in-memory
    /// [`TermDictionary`] (Phase 6e ring serialization).
    ///
    /// Walks the dictionary in insertion order, accumulating the string
    /// table and offsets, then computes the lex-sorted permutation
    /// once at the end.
    ///
    /// # Panics
    ///
    /// Panics if the source dictionary contains more than `u32::MAX`
    /// terms. This matches the rest of the Ring index, which uses `u32`
    /// term ids throughout.
    #[must_use]
    pub fn from_term_dict(dict: &TermDictionary) -> Self {
        let count = dict.len();
        assert!(
            u32::try_from(count).is_ok(),
            "PackedTermDictionary supports up to u32::MAX terms; got {count}"
        );
        // Insertion-order string table.
        let mut strings: Vec<String> = Vec::with_capacity(count);
        let mut total_bytes = 0usize;
        for id in 0..count {
            let term = dict
                .get_term(u32::try_from(id).expect("count <= u32::MAX checked above"))
                .expect("id < len");
            let s = term.to_string();
            total_bytes += s.len();
            strings.push(s);
        }

        let mut string_table_buf = Vec::with_capacity(total_bytes);
        let mut offsets_buf = Vec::with_capacity((count + 1) * 8);
        let mut current = 0u64;
        for s in &strings {
            offsets_buf.extend_from_slice(&current.to_le_bytes());
            string_table_buf.extend_from_slice(s.as_bytes());
            current = current.saturating_add(s.len() as u64);
        }
        // Sentinel.
        offsets_buf.extend_from_slice(&current.to_le_bytes());

        // Lex-sorted permutation.
        let mut sorted: Vec<u32> = (0..u32::try_from(count).expect("count fits u32")).collect();
        sorted.sort_unstable_by(|&a, &b| {
            strings[a as usize]
                .as_bytes()
                .cmp(strings[b as usize].as_bytes())
        });
        let mut sorted_buf = Vec::with_capacity(count * 4);
        for id in &sorted {
            sorted_buf.extend_from_slice(&id.to_le_bytes());
        }

        Self {
            string_table: Bytes::from(string_table_buf),
            offsets: Bytes::from(offsets_buf),
            sorted_ids: Bytes::from(sorted_buf),
            count,
        }
    }

    /// Serializes this dictionary to a flat byte buffer per the layout
    /// documented at the module top (see [`write_to`](Self::write_to)).
    ///
    /// # Panics
    ///
    /// Does not panic: writing into a `Vec` does not fail.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let total =
            HEADER_SIZE + 8 + self.string_table.len() + self.offsets.len() + self.sorted_ids.len();
        let mut buf = Vec::with_capacity(total);
        self.write_to(&mut buf)
            .expect("writing into a Vec does not fail");
        buf
    }

    /// Writes this dictionary to `out` per the layout documented at the
    /// module top: the header, then its three regions as they are stored,
    /// without a copy.
    ///
    /// # Errors
    ///
    /// Returns the first error of `out`.
    pub fn write_to(&self, out: &mut dyn Write) -> std::io::Result<()> {
        // Header.
        out.write_all(MAGIC)?;
        out.write_all(&[VERSION, 0, 0, 0])?; // version, then 3 reserved bytes
        out.write_all(&(self.count as u64).to_le_bytes())?;
        // String table size + table.
        out.write_all(&(self.string_table.len() as u64).to_le_bytes())?;
        out.write_all(&self.string_table)?;
        // Offsets (count + 1 u64 LE).
        out.write_all(&self.offsets)?;
        // Sorted ids (count u32 LE).
        out.write_all(&self.sorted_ids)
    }

    fn read_count_from_header(data: &[u8]) -> Result<usize, PackedDictError> {
        let raw = u64::from_le_bytes(data[8..16].try_into().expect("16-byte slice"));
        usize::try_from(raw).map_err(|_| PackedDictError::InconsistentSizes {
            expected_total: usize::MAX,
            actual_total: data.len(),
        })
    }

    /// Parses a packed dictionary from a refcounted [`Bytes`] buffer.
    ///
    /// Zero-copy where the runtime can subslice; the resulting
    /// `PackedTermDictionary` shares the underlying allocation with `data`.
    ///
    /// # Errors
    ///
    /// Returns a [`PackedDictError`] on truncation, bytes after the sorted
    /// ids, magic/version mismatch, inconsistent sizes (sizes that overflow
    /// included), or invalid offsets/sorted-ids.
    ///
    /// # Panics
    ///
    /// Does not panic in normal operation: header-size checks happen
    /// before any indexed read. Internal `expect`s describe invariants
    /// that the bounds checks above already guarantee.
    pub fn from_bytes(data: Bytes) -> Result<Self, PackedDictError> {
        if data.len() < HEADER_SIZE {
            return Err(PackedDictError::TruncatedHeader);
        }
        if &data[0..4] != MAGIC {
            return Err(PackedDictError::BadMagic);
        }
        let version = data[4];
        if version != VERSION {
            return Err(PackedDictError::UnsupportedVersion(version));
        }
        let count = Self::read_count_from_header(&data)?;

        // String table size header.
        let mut cursor = HEADER_SIZE;
        if cursor + 8 > data.len() {
            return Err(PackedDictError::TruncatedHeader);
        }
        let st_size_raw =
            u64::from_le_bytes(data[cursor..cursor + 8].try_into().expect("8-byte slice"));
        let st_size =
            usize::try_from(st_size_raw).map_err(|_| PackedDictError::InconsistentSizes {
                expected_total: usize::MAX,
                actual_total: data.len(),
            })?;
        cursor += 8;

        // String table region.
        let st_end = cursor
            .checked_add(st_size)
            .ok_or(PackedDictError::InconsistentSizes {
                expected_total: 0,
                actual_total: data.len(),
            })?;
        if st_end > data.len() {
            return Err(PackedDictError::InconsistentSizes {
                expected_total: st_end,
                actual_total: data.len(),
            });
        }
        let string_table = data.slice(cursor..st_end);
        cursor = st_end;

        // Offsets region: (count + 1) * 8 bytes.
        let offsets_size = count
            .checked_add(1)
            .and_then(|entries| entries.checked_mul(8))
            .ok_or(PackedDictError::InconsistentSizes {
                expected_total: usize::MAX,
                actual_total: data.len(),
            })?;
        let offsets_end =
            cursor
                .checked_add(offsets_size)
                .ok_or(PackedDictError::InconsistentSizes {
                    expected_total: 0,
                    actual_total: data.len(),
                })?;
        if offsets_end > data.len() {
            return Err(PackedDictError::InconsistentSizes {
                expected_total: offsets_end,
                actual_total: data.len(),
            });
        }
        let offsets = data.slice(cursor..offsets_end);
        cursor = offsets_end;

        // Sorted ids region: count * 4 bytes.
        let sorted_size = count
            .checked_mul(4)
            .ok_or(PackedDictError::InconsistentSizes {
                expected_total: usize::MAX,
                actual_total: data.len(),
            })?;
        let sorted_end =
            cursor
                .checked_add(sorted_size)
                .ok_or(PackedDictError::InconsistentSizes {
                    expected_total: 0,
                    actual_total: data.len(),
                })?;
        // The sorted ids end the encoding: a part holds exactly its
        // encoding (parts are laid out without padding, in the envelope as
        // in the streams).
        if sorted_end != data.len() {
            return Err(PackedDictError::InconsistentSizes {
                expected_total: sorted_end,
                actual_total: data.len(),
            });
        }
        let sorted_ids = data.slice(cursor..sorted_end);

        let dict = Self {
            string_table,
            offsets,
            sorted_ids,
            count,
        };

        dict.validate_offsets()?;
        dict.validate_sorted_ids()?;
        Ok(dict)
    }

    fn term_bytes_for(&self, id: u32) -> Option<&[u8]> {
        let i = id as usize;
        if i >= self.count {
            return None;
        }
        let start = usize::try_from(read_u64_at(&self.offsets, i)?).ok()?;
        let end = usize::try_from(read_u64_at(&self.offsets, i + 1)?).ok()?;
        self.string_table.get(start..end)
    }

    fn validate_offsets(&self) -> Result<(), PackedDictError> {
        // Sentinel must equal string_table length; offsets must be
        // non-decreasing.
        let mut prev: u64 = 0;
        for i in 0..=self.count {
            let v = read_u64_at(&self.offsets, i).ok_or(PackedDictError::InvalidOffsets)?;
            if v < prev {
                return Err(PackedDictError::InvalidOffsets);
            }
            prev = v;
        }
        let prev_usize = usize::try_from(prev).map_err(|_| PackedDictError::InvalidOffsets)?;
        if prev_usize != self.string_table.len() {
            return Err(PackedDictError::InvalidOffsets);
        }
        Ok(())
    }

    fn validate_sorted_ids(&self) -> Result<(), PackedDictError> {
        for i in 0..self.count {
            let id = read_u32_at(&self.sorted_ids, i).ok_or(PackedDictError::InvalidOffsets)?;
            if id as usize >= self.count {
                return Err(PackedDictError::InvalidSortedId(id));
            }
        }
        Ok(())
    }

    /// Returns the heap-bytes footprint approximation (the underlying
    /// allocation may be shared via `Bytes` refcounting).
    #[must_use]
    pub fn approximate_bytes(&self) -> usize {
        self.string_table.len() + self.offsets.len() + self.sorted_ids.len()
    }
}

fn read_u64_at(bytes: &Bytes, idx: usize) -> Option<u64> {
    let start = idx.checked_mul(8)?;
    let end = start.checked_add(8)?;
    let chunk: [u8; 8] = bytes.get(start..end)?.try_into().ok()?;
    Some(u64::from_le_bytes(chunk))
}

fn read_u32_at(bytes: &Bytes, idx: usize) -> Option<u32> {
    let start = idx.checked_mul(4)?;
    let end = start.checked_add(4)?;
    let chunk: [u8; 4] = bytes.get(start..end)?.try_into().ok()?;
    Some(u32::from_le_bytes(chunk))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_dict_with(terms: &[Term]) -> TermDictionary {
        let mut dict = TermDictionary::new();
        for t in terms {
            dict.get_or_insert(t.clone());
        }
        dict
    }

    #[test]
    fn alix_packed_dict_roundtrip_through_bytes() {
        let terms = [
            Term::iri("http://ex.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
            Term::iri("http://ex.org/gus"),
            Term::literal("Gus"),
        ];
        let dict = build_dict_with(&terms);
        let packed = PackedTermDictionary::from_term_dict(&dict);
        let bytes = packed.to_bytes();
        let restored = PackedTermDictionary::from_bytes(Bytes::from(bytes)).expect("from_bytes");

        assert_eq!(restored.len(), terms.len());
        for (i, term) in terms.iter().enumerate() {
            let id = u32::try_from(i).expect("test fixture id fits u32");
            let restored_term = restored.get_term(id).expect("get_term");
            assert_eq!(&restored_term, term, "term for id {id}");
        }
    }

    #[test]
    fn gus_get_id_by_str_finds_via_binary_search() {
        // Use enough terms that binary search exercises multiple steps.
        let terms: Vec<Term> = (0..32)
            .map(|i| Term::iri(format!("http://ex.org/term-{i:03}")))
            .collect();
        let dict = build_dict_with(&terms);
        let packed = PackedTermDictionary::from_term_dict(&dict);

        for (i, term) in terms.iter().enumerate() {
            let id = packed
                .get_id(term)
                .unwrap_or_else(|| panic!("id for {term}"));
            let expected = u32::try_from(i).expect("test fixture id fits u32");
            assert_eq!(id, expected, "mismatched id for {term}");
        }
    }

    #[test]
    fn vincent_get_id_returns_none_for_absent_term() {
        let dict = build_dict_with(&[Term::iri("http://a"), Term::iri("http://b")]);
        let packed = PackedTermDictionary::from_term_dict(&dict);
        assert!(packed.get_id(&Term::iri("http://c")).is_none());
        assert!(packed.get_id(&Term::literal("missing")).is_none());
    }

    #[test]
    fn jules_empty_dict_round_trip() {
        let dict = TermDictionary::new();
        let packed = PackedTermDictionary::from_term_dict(&dict);
        assert!(packed.is_empty());
        let bytes = packed.to_bytes();
        let restored = PackedTermDictionary::from_bytes(Bytes::from(bytes)).expect("empty");
        assert!(restored.is_empty());
        assert!(restored.get_term(0).is_none());
        assert!(restored.get_id_by_str("anything").is_none());
    }

    #[test]
    fn mia_get_term_str_out_of_range_returns_none() {
        let dict = build_dict_with(&[Term::iri("http://a")]);
        let packed = PackedTermDictionary::from_term_dict(&dict);
        assert!(packed.get_term_str(0).is_some());
        assert!(packed.get_term_str(1).is_none());
        assert!(packed.get_term_str(u32::MAX).is_none());
    }

    #[test]
    fn shosanna_bad_magic_rejected() {
        let bad = Bytes::from(vec![0u8; 32]);
        let result = PackedTermDictionary::from_bytes(bad);
        assert_eq!(result.unwrap_err(), PackedDictError::BadMagic);
    }

    #[test]
    fn beatrix_truncated_header_rejected() {
        let short = Bytes::from(vec![b'P', b'D', b'C', b'T']);
        let result = PackedTermDictionary::from_bytes(short);
        assert_eq!(result.unwrap_err(), PackedDictError::TruncatedHeader);
    }

    #[test]
    fn hans_unsupported_version_rejected() {
        let mut buf = Vec::with_capacity(HEADER_SIZE);
        buf.extend_from_slice(MAGIC);
        buf.push(99); // bad version
        buf.extend_from_slice(&[0u8; 3]);
        buf.extend_from_slice(&0u64.to_le_bytes());
        let result = PackedTermDictionary::from_bytes(Bytes::from(buf));
        assert_eq!(result.unwrap_err(), PackedDictError::UnsupportedVersion(99));
    }

    #[test]
    fn django_inconsistent_size_rejected() {
        // Build a valid header that claims count=5 but supply no body.
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.push(VERSION);
        buf.extend_from_slice(&[0u8; 3]);
        buf.extend_from_slice(&5u64.to_le_bytes());
        // string_table_size header...
        buf.extend_from_slice(&100u64.to_le_bytes()); // claims 100 bytes, but doesn't provide them.
        let result = PackedTermDictionary::from_bytes(Bytes::from(buf));
        assert!(matches!(
            result.unwrap_err(),
            PackedDictError::InconsistentSizes { .. }
        ));
    }

    #[test]
    fn tarantino_zero_copy_via_bytes_refcount() {
        // Build packed dict, serialize, capture pointer, parse back, ensure
        // string_table sub-slice points into the original allocation.
        let dict = build_dict_with(&[
            Term::iri("http://example.org/alix"),
            Term::iri("http://example.org/gus"),
        ]);
        let packed = PackedTermDictionary::from_term_dict(&dict);
        let serialized = packed.to_bytes();
        let source = Bytes::from(serialized);
        let source_ptr = source.as_ptr();

        let restored = PackedTermDictionary::from_bytes(source).expect("from_bytes");
        // Verify the string_table slice's pointer falls within the source
        // allocation — proves Bytes::slice did not copy.
        let st_ptr = restored.string_table.as_ptr();
        // st_ptr must be >= source_ptr (and within source_len), confirming
        // it's a sub-slice of the original allocation.
        let offset = st_ptr as usize - source_ptr as usize;
        assert!(
            offset < (HEADER_SIZE + 8 + restored.string_table.len() + 256),
            "string_table should be inside source allocation; offset={offset}"
        );
    }
    /// A packed dictionary is exactly its encoding: bytes after the sorted
    /// ids are refused.
    #[test]
    fn bytes_after_the_sorted_ids_are_refused() {
        let dict = build_dict_with(&[Term::iri("http://example.org/alix"), Term::literal("Gus")]);
        for packed in [
            PackedTermDictionary::from_term_dict(&dict),
            PackedTermDictionary::from_term_dict(&TermDictionary::new()),
        ] {
            let mut bytes = packed.to_bytes();
            let expected = bytes.len();
            bytes.extend_from_slice(b"Jules");
            assert_eq!(
                PackedTermDictionary::from_bytes(Bytes::from(bytes)).unwrap_err(),
                PackedDictError::InconsistentSizes {
                    expected_total: expected,
                    actual_total: expected + 5
                }
            );
        }
    }

    /// A term count whose regions would not fit any buffer is refused
    /// without overflowing.
    #[test]
    fn a_count_that_overflows_is_refused() {
        let dict = build_dict_with(&[Term::iri("http://example.org/vincent")]);
        let bytes = PackedTermDictionary::from_term_dict(&dict).to_bytes();
        for count in [1u64 << 61, (1u64 << 62) + 3, u64::MAX] {
            let mut crafted = bytes.clone();
            crafted[8..16].copy_from_slice(&count.to_le_bytes());
            assert!(
                matches!(
                    PackedTermDictionary::from_bytes(Bytes::from(crafted)),
                    Err(PackedDictError::InconsistentSizes { .. })
                ),
                "count {count:#x}"
            );
        }
    }
}
