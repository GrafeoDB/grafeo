//! Packed wavelet tree for the v2 Ring on-disk format (Phase 6c).
//!
//! The in-memory [`WaveletTree`] stores `height` `SuccinctBitVector`
//! levels alongside rank/select sampling caches. The bincode'd v1 format
//! serializes both the bit data AND the caches, even though the caches
//! are O(n) rebuildable from the bits alone (per
//! [`SuccinctBitVector::from_bitvec`]).
//!
//! v2 keeps only the bit data, packed as little-endian `u64` words, and
//! rebuilds caches on `to_wavelet_tree`. This shrinks the on-disk size
//! ~30-40% and removes schema overhead, while the reload cost is
//! unchanged (cache rebuild is the dominant term either way).
//!
//! ## Layout
//!
//! ```text
//! Header (40 bytes):
//!     0..4    magic "WTRE"
//!     4       version u8 = 1
//!     5..8    reserved (3 bytes, zero)
//!     8..12   height u32 LE
//!     12..16  padding (4 bytes, zero) — aligns u64 fields to 8-byte boundary
//!     16..24  sigma u64 LE                 // alphabet size
//!     24..32  len u64 LE                   // sequence length (== bits per level)
//!     32..40  symbol_count u64 LE          // length of the symbols region in elements
//!
//! symbols region: symbol_count * 8 bytes (u64 LE, sorted)
//! per-level region (height entries):
//!     bit_count: u64 LE
//!     word_count: u64 LE
//!     word_count * 8 bytes of LE u64 BitVector data
//! ```

use std::io::Write;

use bytes::Bytes;

use crate::codec::BitVector;
use crate::codec::succinct::{SuccinctBitVector, WaveletTree};

const MAGIC: &[u8; 4] = b"WTRE";
const VERSION: u8 = 1;
const HEADER_SIZE: usize = 40;

/// Errors returned when parsing a packed wavelet tree from bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackedWaveletError {
    /// Buffer is too short to contain even the fixed-size header.
    TruncatedHeader,
    /// First 4 bytes don't match "WTRE".
    BadMagic,
    /// Version byte not recognized.
    UnsupportedVersion(u8),
    /// Recorded sizes overflow the input buffer.
    Truncated {
        /// Region we were trying to read.
        region: &'static str,
    },
    /// A field overflows the platform-native usize.
    SizeOverflow,
    /// The header claims more levels than a tree over `u64` symbols has
    /// (64), refused before anything is allocated for them.
    TooHigh {
        /// The height the header claims.
        height: u32,
    },
    /// Bytes follow the last level: a packed tree is exactly its
    /// encoding.
    TrailingBytes {
        /// The length the header and the levels declare.
        expected: usize,
        /// The length of the buffer.
        actual: usize,
    },
    /// Per-level bit count doesn't match the declared `len` field.
    BitCountMismatch {
        /// Level index where the mismatch was observed.
        level: usize,
        /// Bit count declared in the header.
        expected: u64,
        /// Bit count observed in the level.
        actual: u64,
    },
    /// Reconstructed parts violated a structural [`WaveletTree`]
    /// invariant — caught here rather than letting the tree return
    /// inconsistent answers from `access`/`rank`.
    InvariantViolation(crate::codec::succinct::WaveletInvariantError),
}

impl std::fmt::Display for PackedWaveletError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TruncatedHeader => write!(f, "packed wavelet header truncated"),
            Self::BadMagic => write!(f, "packed wavelet bad magic (expected 'WTRE')"),
            Self::UnsupportedVersion(v) => write!(f, "packed wavelet unsupported version {v}"),
            Self::Truncated { region } => write!(f, "packed wavelet truncated in {region}"),
            Self::SizeOverflow => write!(f, "packed wavelet size field overflows usize"),
            Self::TooHigh { height } => {
                write!(f, "packed wavelet height {height} exceeds 64 levels")
            }
            Self::TrailingBytes { expected, actual } => write!(
                f,
                "packed wavelet holds {actual} bytes, its encoding {expected}"
            ),
            Self::BitCountMismatch {
                level,
                expected,
                actual,
            } => write!(
                f,
                "packed wavelet bit count mismatch at level {level}: expected {expected}, got {actual}"
            ),
            Self::InvariantViolation(e) => write!(f, "packed wavelet invariant violation: {e}"),
        }
    }
}

impl std::error::Error for PackedWaveletError {}

/// Serializes a [`WaveletTree`] to the v2 packed format.
///
/// # Panics
///
/// Panics when the tree is higher than `u32::MAX` levels, which a tree over
/// `u64` symbols never is (see [`write_wavelet_tree`]).
#[must_use]
pub fn serialize_wavelet_tree(tree: &WaveletTree) -> Vec<u8> {
    let mut buf = Vec::with_capacity(packed_len(tree));
    write_wavelet_tree(tree, &mut buf).expect("writing into a Vec fails only for a tree too high");
    buf
}

/// The length of the packed format of `tree`.
fn packed_len(tree: &WaveletTree) -> usize {
    let symbols_bytes = tree.symbols_slice().len() * 8;
    let level_bytes: usize = tree
        .levels_slice()
        .iter()
        .map(|sbv| 16 /* bit_count + word_count */ + sbv.inner().data_bytes().len())
        .sum();
    HEADER_SIZE + symbols_bytes + level_bytes
}

/// Writes `tree` to `out` in the v2 packed format, piece by piece: the
/// header, the symbols, then each level's bits as they are stored. No copy
/// of the tree is made.
///
/// # Errors
///
/// Returns the first error of `out`, or [`std::io::ErrorKind::InvalidInput`]
/// when the tree is higher than `u32::MAX` levels, which the header cannot
/// record.
pub fn write_wavelet_tree(tree: &WaveletTree, out: &mut dyn Write) -> std::io::Result<()> {
    let symbols = tree.symbols_slice();
    let height = u32::try_from(tree.height()).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "a packed wavelet tree records at most u32::MAX levels, this one has {}",
                tree.height()
            ),
        )
    })?;

    // Header (40 bytes total, see the module-top layout doc):
    out.write_all(MAGIC)?; // 0..4
    out.write_all(&[VERSION, 0, 0, 0])?; // 4, then 5..8 reserved
    out.write_all(&height.to_le_bytes())?; // 8..12
    out.write_all(&[0u8; 4])?; // 12..16 padding to align sigma
    out.write_all(&tree.sigma().to_le_bytes())?; // 16..24
    out.write_all(&(tree.len() as u64).to_le_bytes())?; // 24..32
    out.write_all(&(symbols.len() as u64).to_le_bytes())?; // 32..40 symbol_count

    // Symbols.
    for &sym in symbols {
        out.write_all(&sym.to_le_bytes())?;
    }

    // Levels.
    for sbv in tree.levels_slice() {
        let bv = sbv.inner();
        let word_data = bv.data_bytes();
        out.write_all(&(bv.len() as u64).to_le_bytes())?; // bit_count
        out.write_all(&((word_data.len() / 8) as u64).to_le_bytes())?; // word_count
        out.write_all(word_data)?;
    }
    Ok(())
}

/// Parses a [`WaveletTree`] from the v2 packed format. Rebuilds rank/select
/// caches per level via [`SuccinctBitVector::from_bitvec`].
///
/// `data` is consumed via `Bytes::slice` so the underlying allocation is
/// shared with the caller. Per-level `BitVector`s adopt their slices via
/// [`BitVector::from_mmap`], so a mmap-backed buffer never copies.
///
/// # Errors
///
/// Returns a [`PackedWaveletError`] on truncation, magic/version
/// mismatch, a height above 64 levels, bytes after the last level, a
/// per-level bit-count inconsistency, or parts that break an invariant of
/// the tree (see [`WaveletTree::from_packed_parts`]).
///
/// # Panics
///
/// Internal `expect` calls describe invariants that the bounds checks
/// above already guarantee — every indexed read is preceded by an
/// explicit length check. Does not panic in normal operation.
pub fn deserialize_wavelet_tree(data: Bytes) -> Result<WaveletTree, PackedWaveletError> {
    if data.len() < HEADER_SIZE {
        return Err(PackedWaveletError::TruncatedHeader);
    }
    if &data[0..4] != MAGIC {
        return Err(PackedWaveletError::BadMagic);
    }
    let version = data[4];
    if version != VERSION {
        return Err(PackedWaveletError::UnsupportedVersion(version));
    }
    // Header offsets (per module-top layout doc):
    let height_raw = u32::from_le_bytes(data[8..12].try_into().expect("4-byte slice"));
    // 12..16 is padding.
    let sigma = u64::from_le_bytes(data[16..24].try_into().expect("8-byte slice"));
    let len_raw = u64::from_le_bytes(data[24..32].try_into().expect("8-byte slice"));
    let symbol_count_raw = u64::from_le_bytes(data[32..40].try_into().expect("8-byte slice"));

    // A tree over u64 symbols has at most 64 levels. Refused before the
    // levels are allocated: a crafted height must not size an allocation.
    if height_raw > 64 {
        return Err(PackedWaveletError::TooHigh { height: height_raw });
    }
    let height = usize::try_from(height_raw).map_err(|_| PackedWaveletError::SizeOverflow)?;
    let len_usize = usize::try_from(len_raw).map_err(|_| PackedWaveletError::SizeOverflow)?;
    let symbol_count =
        usize::try_from(symbol_count_raw).map_err(|_| PackedWaveletError::SizeOverflow)?;

    let mut cursor = HEADER_SIZE;

    // Symbols region.
    let symbols_bytes = symbol_count
        .checked_mul(8)
        .ok_or(PackedWaveletError::SizeOverflow)?;
    let symbols_end = cursor
        .checked_add(symbols_bytes)
        .ok_or(PackedWaveletError::SizeOverflow)?;
    if symbols_end > data.len() {
        return Err(PackedWaveletError::Truncated { region: "symbols" });
    }
    let mut symbols: Vec<u64> = Vec::with_capacity(symbol_count);
    for i in 0..symbol_count {
        // Inner offsets are safe: `symbols_end = cursor + symbols_bytes`
        // is bounds-checked above, and i < symbol_count implies
        // `cursor + i*8 + 8 <= symbols_end`.
        let off = cursor + i * 8;
        let chunk: [u8; 8] = data[off..off + 8].try_into().expect("8-byte slice");
        symbols.push(u64::from_le_bytes(chunk));
    }
    cursor = symbols_end;

    // Levels region.
    let mut levels: Vec<SuccinctBitVector> = Vec::with_capacity(height);
    for level_idx in 0..height {
        let level_header_end = cursor
            .checked_add(16)
            .ok_or(PackedWaveletError::SizeOverflow)?;
        if level_header_end > data.len() {
            return Err(PackedWaveletError::Truncated {
                region: "level header",
            });
        }
        // Header bounds verified above: `level_header_end = cursor + 16`
        // doesn't overflow and is in range.
        let bit_count =
            u64::from_le_bytes(data[cursor..cursor + 8].try_into().expect("8-byte slice"));
        let word_count = u64::from_le_bytes(
            data[cursor + 8..cursor + 16]
                .try_into()
                .expect("8-byte slice"),
        );
        cursor = level_header_end;

        if bit_count != len_raw {
            return Err(PackedWaveletError::BitCountMismatch {
                level: level_idx,
                expected: len_raw,
                actual: bit_count,
            });
        }

        let level_bytes = usize::try_from(
            word_count
                .checked_mul(8)
                .ok_or(PackedWaveletError::SizeOverflow)?,
        )
        .map_err(|_| PackedWaveletError::SizeOverflow)?;
        let level_data_end = cursor
            .checked_add(level_bytes)
            .ok_or(PackedWaveletError::SizeOverflow)?;
        if level_data_end > data.len() {
            return Err(PackedWaveletError::Truncated {
                region: "level data",
            });
        }
        let level_slice = data.slice(cursor..level_data_end);
        cursor = level_data_end;

        let bv = BitVector::from_mmap(level_slice, len_usize).map_err(|_| {
            PackedWaveletError::Truncated {
                region: "level bits",
            }
        })?;
        levels.push(SuccinctBitVector::from_bitvec(bv));
    }
    // A part holds exactly its encoding (parts are laid out without
    // padding, in the envelope as in the streams).
    if cursor != data.len() {
        return Err(PackedWaveletError::TrailingBytes {
            expected: cursor,
            actual: data.len(),
        });
    }

    WaveletTree::from_packed_parts(levels, height, sigma, len_usize, symbols)
        .map_err(PackedWaveletError::InvariantViolation)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_tree(seq: &[u64]) -> WaveletTree {
        WaveletTree::new(seq)
    }

    fn assert_trees_equal(orig: &WaveletTree, restored: &WaveletTree) {
        assert_eq!(orig.len(), restored.len());
        assert_eq!(orig.sigma(), restored.sigma());
        for i in 0..orig.len() {
            assert_eq!(
                orig.access(i),
                restored.access(i),
                "access mismatch at position {i}"
            );
        }
    }

    #[test]
    fn alix_packed_wavelet_roundtrip_small() {
        let seq = vec![1u64, 3, 2, 1, 2, 3, 1, 2];
        let tree = build_tree(&seq);
        let bytes = serialize_wavelet_tree(&tree);
        let restored = deserialize_wavelet_tree(Bytes::from(bytes)).expect("deserialize");
        assert_trees_equal(&tree, &restored);
    }

    #[test]
    fn gus_packed_wavelet_roundtrip_large() {
        // 1024 symbols drawn from an alphabet of 16 — exercises multi-level
        // wavelet structure at non-trivial size.
        let seq: Vec<u64> = (0..1024u64).map(|i| (i * 7) % 16).collect();
        let tree = build_tree(&seq);
        let bytes = serialize_wavelet_tree(&tree);
        let restored = deserialize_wavelet_tree(Bytes::from(bytes)).expect("deserialize");
        assert_trees_equal(&tree, &restored);
    }

    #[test]
    fn vincent_packed_wavelet_empty() {
        let tree = WaveletTree::new(&[]);
        let bytes = serialize_wavelet_tree(&tree);
        let restored = deserialize_wavelet_tree(Bytes::from(bytes)).expect("deserialize");
        assert_eq!(restored.len(), 0);
        assert!(restored.is_empty());
    }

    #[test]
    fn jules_packed_wavelet_single_symbol() {
        // sigma = 1; height ends up = 1 per `WaveletTree::new`.
        let seq = vec![42u64; 16];
        let tree = build_tree(&seq);
        let bytes = serialize_wavelet_tree(&tree);
        let restored = deserialize_wavelet_tree(Bytes::from(bytes)).expect("deserialize");
        assert_trees_equal(&tree, &restored);
    }

    #[test]
    fn mia_packed_wavelet_bad_magic_rejected() {
        let bad = Bytes::from(vec![0u8; HEADER_SIZE]);
        assert_eq!(
            deserialize_wavelet_tree(bad).unwrap_err(),
            PackedWaveletError::BadMagic
        );
    }

    #[test]
    fn shosanna_packed_wavelet_truncated_header_rejected() {
        let short = Bytes::from(vec![b'W', b'T', b'R', b'E']);
        assert_eq!(
            deserialize_wavelet_tree(short).unwrap_err(),
            PackedWaveletError::TruncatedHeader
        );
    }

    #[test]
    fn beatrix_packed_wavelet_unsupported_version_rejected() {
        let mut buf = vec![0u8; HEADER_SIZE];
        buf[..4].copy_from_slice(MAGIC);
        buf[4] = 99;
        assert_eq!(
            deserialize_wavelet_tree(Bytes::from(buf)).unwrap_err(),
            PackedWaveletError::UnsupportedVersion(99)
        );
    }

    #[test]
    fn hans_packed_wavelet_size_smaller_than_bincode() {
        // The whole point of v2: smaller than bincode by removing
        // redundant rank/select caches and schema overhead.
        let seq: Vec<u64> = (0..2048u64).map(|i| (i * 11) % 64).collect();
        let tree = build_tree(&seq);
        let v2_bytes = serialize_wavelet_tree(&tree);
        let v1_bytes = bincode::serde::encode_to_vec(&tree, bincode::config::standard())
            .expect("bincode encode");
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

    #[test]
    fn django_packed_wavelet_zero_copy_per_level() {
        // Round-trip and confirm restored levels' inner BitVector data
        // shares the underlying source allocation (zero-copy mmap path).
        let seq = vec![1u64, 2, 3, 4, 5, 6, 7, 8];
        let tree = build_tree(&seq);
        let bytes = serialize_wavelet_tree(&tree);
        let source = Bytes::from(bytes);
        let source_ptr = source.as_ptr();
        let source_len = source.len();

        let restored = deserialize_wavelet_tree(source).expect("deserialize");
        for (idx, level) in restored.levels_slice().iter().enumerate() {
            let inner_ptr = level.inner().data_bytes().as_ptr();
            let offset = inner_ptr as usize - source_ptr as usize;
            assert!(
                offset < source_len,
                "level {idx}: inner BitVector should be inside source allocation; offset={offset}"
            );
        }
    }
    use crate::codec::succinct::WaveletInvariantError;

    /// A packed tree with `height` levels of `len` zero bits each over
    /// `symbols`, whose header claims `sigma`.
    fn crafted_tree(height: u32, sigma: u64, len: u64, symbols: &[u64]) -> Bytes {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&[VERSION, 0, 0, 0]);
        out.extend_from_slice(&height.to_le_bytes());
        out.extend_from_slice(&[0; 4]);
        out.extend_from_slice(&sigma.to_le_bytes());
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(symbols.len() as u64).to_le_bytes());
        for symbol in symbols {
            out.extend_from_slice(&symbol.to_le_bytes());
        }
        let words = len.div_ceil(64);
        for _ in 0..height {
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(&words.to_le_bytes());
            out.extend(std::iter::repeat_n(
                0u8,
                usize::try_from(words * 8).unwrap(),
            ));
        }
        Bytes::from(out)
    }

    /// A header claiming more levels than a tree over `u64` symbols has is
    /// refused before anything is allocated for its levels (a claim of
    /// `u32::MAX` levels once aborted the process).
    #[test]
    fn a_height_above_64_levels_is_refused() {
        for height in [65, 70, u32::MAX] {
            let mut bytes = serialize_wavelet_tree(&build_tree(&[3, 19]));
            bytes[8..12].copy_from_slice(&height.to_le_bytes());
            assert_eq!(
                deserialize_wavelet_tree(Bytes::from(bytes)).unwrap_err(),
                PackedWaveletError::TooHigh { height },
                "height {height}"
            );
        }
        assert_eq!(
            deserialize_wavelet_tree(crafted_tree(70, 2, 3, &[3, 19])).unwrap_err(),
            PackedWaveletError::TooHigh { height: 70 },
            "70 levels, all present"
        );
    }

    /// A packed tree is exactly its encoding: bytes after the last level
    /// are refused.
    #[test]
    fn bytes_after_the_last_level_are_refused() {
        for sequence in [&[][..], &[3, 19, 88, 3][..]] {
            let mut bytes = serialize_wavelet_tree(&build_tree(sequence));
            let expected = bytes.len();
            bytes.extend_from_slice(b"Mia");
            assert_eq!(
                deserialize_wavelet_tree(Bytes::from(bytes)).unwrap_err(),
                PackedWaveletError::TrailingBytes {
                    expected,
                    actual: expected + 3
                },
                "{sequence:?}"
            );
        }
    }

    /// A tree whose alphabet, symbols and height disagree is refused: the
    /// alphabet size is the number of symbols, a sequence has symbols when
    /// it is not empty, and the height is the bits the codes take.
    #[test]
    fn an_inconsistent_tree_is_refused() {
        let invariant = |bytes: Bytes| match deserialize_wavelet_tree(bytes) {
            Err(PackedWaveletError::InvariantViolation(error)) => error,
            other => panic!("expected an invariant violation, got {other:?}"),
        };
        assert_eq!(
            invariant(crafted_tree(1, 3, 3, &[3, 19])),
            WaveletInvariantError::SigmaMismatch {
                symbols_len: 2,
                sigma: 3
            }
        );
        assert_eq!(
            invariant(crafted_tree(1, 0, 3, &[])),
            WaveletInvariantError::AlphabetMismatch {
                len: 3,
                symbols_len: 0
            }
        );
        assert_eq!(
            invariant(crafted_tree(0, 1, 0, &[88])),
            WaveletInvariantError::AlphabetMismatch {
                len: 0,
                symbols_len: 1
            }
        );
        for (height, symbols, expected) in [
            (2, &[3, 19][..], 1),
            (64, &[3, 19][..], 1),
            (1, &[3, 19, 88][..], 2),
            (2, &[88][..], 1),
        ] {
            assert_eq!(
                invariant(crafted_tree(height, symbols.len() as u64, 3, symbols)),
                WaveletInvariantError::HeightMismatch {
                    height: height as usize,
                    expected
                },
                "{height} levels for {symbols:?}"
            );
        }
        for sequence in [&[][..], &[88][..], &[3, 19][..], &[3, 19, 88][..]] {
            let tree = build_tree(sequence);
            let restored = deserialize_wavelet_tree(Bytes::from(serialize_wavelet_tree(&tree)))
                .expect("a consistent tree");
            assert_trees_equal(&tree, &restored);
        }
    }
}
