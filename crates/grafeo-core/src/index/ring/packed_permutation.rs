//! Packed succinct permutation for the v2 Ring on-disk format (Phase 6d).
//!
//! [`SuccinctPermutation`] stores both the forward and inverse mappings on
//! the heap (O(2n) space). The bincode'd v1 format serializes both, even
//! though the inverse is O(n)-rebuildable from the forward mapping alone.
//!
//! v2 stores only the forward mapping as a packed `u32` LE array.
//! Deserialization rebuilds the inverse in a single linear pass. This
//! halves the on-disk size compared to v1.
//!
//! ## Layout
//!
//! ```text
//! Header (16 bytes):
//!     0..4    magic "PERM"
//!     4       version u8 = 1
//!     5..8    reserved (3 bytes, zero)
//!     8..16   n u64 LE
//!
//! forward region: n * 4 bytes (u32 LE)
//! ```

use std::io::Write;

use bytes::Bytes;

use crate::index::ring::SuccinctPermutation;

const MAGIC: &[u8; 4] = b"PERM";
const VERSION: u8 = 1;
const HEADER_SIZE: usize = 16;

/// Errors returned when parsing a packed permutation from bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackedPermutationError {
    /// Buffer is too short to contain even the fixed-size header.
    TruncatedHeader,
    /// First 4 bytes don't match "PERM".
    BadMagic,
    /// Version byte not recognized.
    UnsupportedVersion(u8),
    /// Forward array is shorter than `n` declares.
    TruncatedForward {
        /// Bytes the forward array should contain.
        expected: usize,
        /// Bytes available in the input.
        actual: usize,
    },
    /// `n` field overflows the platform-native usize.
    SizeOverflow,
    /// Bytes follow the forward array: a packed permutation is exactly
    /// its encoding.
    TrailingBytes {
        /// The length the header declares.
        expected: usize,
        /// The length of the buffer.
        actual: usize,
    },
    /// A `forward\[i\]` entry references an index >= n (not a valid
    /// permutation).
    InvalidPermutation {
        /// Position that contained the bad value.
        index: usize,
        /// The bad value.
        value: u32,
    },
    /// A target index appears twice in the forward mapping (not a
    /// bijection).
    DuplicateTarget {
        /// Position whose target collided with an earlier one.
        index: usize,
        /// The duplicated target value.
        value: u32,
    },
}

impl std::fmt::Display for PackedPermutationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TruncatedHeader => write!(f, "packed permutation header truncated"),
            Self::BadMagic => write!(f, "packed permutation bad magic (expected 'PERM')"),
            Self::UnsupportedVersion(v) => {
                write!(f, "packed permutation unsupported version {v}")
            }
            Self::TruncatedForward { expected, actual } => write!(
                f,
                "packed permutation forward truncated: expected {expected} bytes, got {actual}"
            ),
            Self::SizeOverflow => write!(f, "packed permutation size field overflows usize"),
            Self::TrailingBytes { expected, actual } => write!(
                f,
                "packed permutation holds {actual} bytes, its encoding {expected}"
            ),
            Self::InvalidPermutation { index, value } => write!(
                f,
                "packed permutation forward[{index}] = {value} is out of range"
            ),
            Self::DuplicateTarget { index, value } => write!(
                f,
                "packed permutation forward[{index}] = {value} duplicates an earlier entry"
            ),
        }
    }
}

impl std::error::Error for PackedPermutationError {}

/// Serializes a [`SuccinctPermutation`] to the v2 packed format.
///
/// # Panics
///
/// Panics when the permutation has more than `u32::MAX` entries, which the
/// packed format cannot hold (see [`write_permutation`]).
#[must_use]
pub fn serialize_permutation(perm: &SuccinctPermutation) -> Vec<u8> {
    let mut buf = Vec::with_capacity(HEADER_SIZE + perm.len() * 4);
    write_permutation(perm, &mut buf)
        .expect("writing into a Vec fails only for a permutation too long");
    buf
}

/// Writes `perm` to `out` in the v2 packed format: the header, then the
/// forward mapping one entry at a time, without a copy of it.
///
/// # Errors
///
/// Returns the first error of `out`, or [`std::io::ErrorKind::InvalidInput`]
/// when the permutation has more than `u32::MAX` entries: the packed format
/// stores each target as a `u32`, and [`deserialize_permutation`] refuses
/// more.
///
/// # Panics
///
/// The internal `apply(i)` `expect` describes an invariant the `0..n` loop
/// bounds already guarantee: `apply(i)` returns `None` only when `i >= n`.
/// Does not panic in normal operation.
pub fn write_permutation(perm: &SuccinctPermutation, out: &mut dyn Write) -> std::io::Result<()> {
    let n = perm.len();
    if u32::try_from(n).is_err() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("a packed permutation holds at most u32::MAX entries, this one has {n}"),
        ));
    }
    out.write_all(MAGIC)?; // 0..4
    out.write_all(&[VERSION, 0, 0, 0])?; // 4, then 5..8 reserved
    out.write_all(&(n as u64).to_le_bytes())?; // 8..16
    for i in 0..n {
        // `apply` is O(1) on the heap representation; safe because i < n.
        let target = perm.apply(i).expect("i < n");
        // Every target is below n, which fits a u32 (checked above).
        let target = u32::try_from(target).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("permutation target {target} at {i} does not fit a u32"),
            )
        })?;
        out.write_all(&target.to_le_bytes())?;
    }
    Ok(())
}

/// Parses a [`SuccinctPermutation`] from the v2 packed format. Rebuilds
/// the inverse mapping in a single linear pass.
///
/// # Errors
///
/// Returns a [`PackedPermutationError`] on truncation, bytes after the
/// forward array, magic/version mismatch, out-of-range entries, or
/// duplicate targets (the input is not a valid permutation).
///
/// # Panics
///
/// Internal `expect` calls describe invariants the bounds checks above
/// already guarantee — every indexed read is preceded by a length
/// check. Does not panic in normal operation.
pub fn deserialize_permutation(data: Bytes) -> Result<SuccinctPermutation, PackedPermutationError> {
    if data.len() < HEADER_SIZE {
        return Err(PackedPermutationError::TruncatedHeader);
    }
    if &data[0..4] != MAGIC {
        return Err(PackedPermutationError::BadMagic);
    }
    let version = data[4];
    if version != VERSION {
        return Err(PackedPermutationError::UnsupportedVersion(version));
    }
    let n_raw = u64::from_le_bytes(data[8..16].try_into().expect("8-byte slice"));
    let n = usize::try_from(n_raw).map_err(|_| PackedPermutationError::SizeOverflow)?;

    let forward_bytes = n
        .checked_mul(4)
        .ok_or(PackedPermutationError::SizeOverflow)?;
    let total = HEADER_SIZE
        .checked_add(forward_bytes)
        .ok_or(PackedPermutationError::SizeOverflow)?;
    if total > data.len() {
        return Err(PackedPermutationError::TruncatedForward {
            expected: forward_bytes,
            actual: data.len() - HEADER_SIZE,
        });
    }
    // A part holds exactly its encoding (parts are laid out without
    // padding, in the envelope as in the streams).
    if total < data.len() {
        return Err(PackedPermutationError::TrailingBytes {
            expected: total,
            actual: data.len(),
        });
    }

    // Validate + collect forward array as usize for SuccinctPermutation::new.
    let n_u32 = u32::try_from(n).map_err(|_| PackedPermutationError::SizeOverflow)?;
    let mut forward: Vec<usize> = Vec::with_capacity(n);
    let mut seen: Vec<bool> = vec![false; n];
    for i in 0..n {
        let off = HEADER_SIZE + i * 4;
        let chunk: [u8; 4] = data[off..off + 4].try_into().expect("4-byte slice");
        let value = u32::from_le_bytes(chunk);
        if value >= n_u32 {
            return Err(PackedPermutationError::InvalidPermutation { index: i, value });
        }
        if seen[value as usize] {
            return Err(PackedPermutationError::DuplicateTarget { index: i, value });
        }
        seen[value as usize] = true;
        forward.push(value as usize);
    }

    Ok(SuccinctPermutation::new(&forward))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_perm(forward: &[usize]) -> SuccinctPermutation {
        SuccinctPermutation::new(forward)
    }

    #[test]
    fn alix_packed_perm_roundtrip_small() {
        let forward = vec![3usize, 0, 4, 1, 2];
        let perm = build_perm(&forward);
        let bytes = serialize_permutation(&perm);
        let restored = deserialize_permutation(Bytes::from(bytes)).expect("deserialize");
        assert_eq!(restored.len(), perm.len());
        for i in 0..perm.len() {
            assert_eq!(restored.apply(i), perm.apply(i));
            assert_eq!(restored.apply_inverse(i), perm.apply_inverse(i));
        }
    }

    #[test]
    fn gus_packed_perm_roundtrip_identity() {
        let forward: Vec<usize> = (0..256).collect();
        let perm = build_perm(&forward);
        let bytes = serialize_permutation(&perm);
        let restored = deserialize_permutation(Bytes::from(bytes)).expect("deserialize");
        for i in 0..256 {
            assert_eq!(restored.apply(i), Some(i));
        }
    }

    #[test]
    fn vincent_packed_perm_roundtrip_reverse() {
        let forward: Vec<usize> = (0..128).rev().collect();
        let perm = build_perm(&forward);
        let bytes = serialize_permutation(&perm);
        let restored = deserialize_permutation(Bytes::from(bytes)).expect("deserialize");
        for i in 0..128 {
            assert_eq!(restored.apply(i), Some(127 - i));
            // Inverse of reverse is also reverse.
            assert_eq!(restored.apply_inverse(i), Some(127 - i));
        }
    }

    #[test]
    fn jules_packed_perm_empty() {
        let perm = build_perm(&[]);
        let bytes = serialize_permutation(&perm);
        assert_eq!(bytes.len(), HEADER_SIZE);
        let restored = deserialize_permutation(Bytes::from(bytes)).expect("empty");
        assert_eq!(restored.len(), 0);
        assert!(restored.is_empty());
    }

    #[test]
    fn mia_packed_perm_bad_magic_rejected() {
        let bad = Bytes::from(vec![0u8; HEADER_SIZE]);
        assert_eq!(
            deserialize_permutation(bad).unwrap_err(),
            PackedPermutationError::BadMagic
        );
    }

    #[test]
    fn shosanna_packed_perm_truncated_header_rejected() {
        let short = Bytes::from(vec![b'P', b'E', b'R', b'M']);
        assert_eq!(
            deserialize_permutation(short).unwrap_err(),
            PackedPermutationError::TruncatedHeader
        );
    }

    #[test]
    fn beatrix_packed_perm_unsupported_version_rejected() {
        let mut buf = vec![0u8; HEADER_SIZE];
        buf[..4].copy_from_slice(MAGIC);
        buf[4] = 99;
        assert_eq!(
            deserialize_permutation(Bytes::from(buf)).unwrap_err(),
            PackedPermutationError::UnsupportedVersion(99)
        );
    }

    #[test]
    fn hans_packed_perm_invalid_target_rejected() {
        // n=2 but forward[0] = 5 (out of range)
        let mut buf = Vec::with_capacity(HEADER_SIZE + 8);
        buf.extend_from_slice(MAGIC);
        buf.push(VERSION);
        buf.extend_from_slice(&[0u8; 3]);
        buf.extend_from_slice(&2u64.to_le_bytes());
        buf.extend_from_slice(&5u32.to_le_bytes()); // bad
        buf.extend_from_slice(&0u32.to_le_bytes());
        let result = deserialize_permutation(Bytes::from(buf));
        assert!(matches!(
            result.unwrap_err(),
            PackedPermutationError::InvalidPermutation { index: 0, value: 5 }
        ));
    }

    #[test]
    fn django_packed_perm_duplicate_target_rejected() {
        // n=3, forward = [0, 0, 2] — 0 appears twice.
        let mut buf = Vec::with_capacity(HEADER_SIZE + 12);
        buf.extend_from_slice(MAGIC);
        buf.push(VERSION);
        buf.extend_from_slice(&[0u8; 3]);
        buf.extend_from_slice(&3u64.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // dup
        buf.extend_from_slice(&2u32.to_le_bytes());
        let result = deserialize_permutation(Bytes::from(buf));
        assert!(matches!(
            result.unwrap_err(),
            PackedPermutationError::DuplicateTarget { index: 1, value: 0 }
        ));
    }

    #[test]
    fn tarantino_packed_perm_size_competitive_with_bincode() {
        // Bincode varint encoding is surprisingly compact for small
        // permutations: indices < 16384 take only 2-3 bytes per
        // element, vs v2's fixed 4-byte u32. v2 stores only the forward
        // mapping (half the elements), so the total comparison ends
        // up close-to-even at small n. v2's win materialises at
        // n > ~16k where varint widens to 3-4 bytes per element.
        //
        // This test pins the small-n behaviour to "v2 not larger than
        // v1" so format regressions still get caught.
        let forward: Vec<usize> = (0..512).map(|i| (i * 17) % 512).collect();
        let perm = build_perm(&forward);
        let v2_bytes = serialize_permutation(&perm);
        let v1_bytes = bincode::serde::encode_to_vec(&perm, bincode::config::standard())
            .expect("bincode encode");
        eprintln!(
            "v1 bincode: {} bytes, v2 packed: {} bytes",
            v1_bytes.len(),
            v2_bytes.len()
        );
        assert!(
            v2_bytes.len() <= v1_bytes.len(),
            "v2 packed must not be larger than v1 bincode (v1={}, v2={})",
            v1_bytes.len(),
            v2_bytes.len()
        );
    }
    /// A packed permutation is exactly its encoding: bytes after the
    /// forward array are refused.
    #[test]
    fn bytes_after_the_forward_array_are_refused() {
        for forward in [&[][..], &[3usize, 0, 4, 1, 2][..]] {
            let mut bytes = serialize_permutation(&build_perm(forward));
            let expected = bytes.len();
            bytes.extend_from_slice(b"Gus");
            assert_eq!(
                deserialize_permutation(Bytes::from(bytes)).unwrap_err(),
                PackedPermutationError::TrailingBytes {
                    expected,
                    actual: expected + 3
                },
                "{forward:?}"
            );
        }
    }
}
