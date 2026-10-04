//! Bit-packing for small integers.
//!
//! If your largest value is 15, why use 64 bits per number? Bit-packing uses
//! only the bits you need - 4 bits for values up to 15, giving you 16x compression.
//!
//! This works especially well after delta encoding sorted data, where the deltas
//! are often tiny even when the original values are huge.
//!
//! # Example
//!
//! ```no_run
//! # use grafeo_core::codec::bitpack::BitPackedInts;
//! // Values [5, 2, 3, 5, 5, 8, 2] - max is 8, needs 4 bits
//! // Without packing: 7 * 64 = 448 bits
//! // With packing:    7 * 4  = 28 bits (16x smaller!)
//!
//! let values = vec![5u64, 2, 3, 5, 5, 8, 2];
//! let packed = BitPackedInts::pack(&values);
//! let unpacked = packed.unpack();
//! assert_eq!(values, unpacked);
//! ```

use std::io;

use bytes::{Bytes, BytesMut};

/// Converts a slice of `u64` words to a refcounted `Bytes` buffer of
/// little-endian bytes (8 bytes per word). Used by builders that produce
/// `Vec<u64>` and need to expose it as `Bytes` storage.
fn words_to_bytes(words: &[u64]) -> Bytes {
    let mut buf = BytesMut::with_capacity(words.len() * 8);
    for &w in words {
        buf.extend_from_slice(&w.to_le_bytes());
    }
    buf.freeze()
}

/// Two-variant backing for the packed `u64` words.
///
/// `Inline` is used when the column is built in RAM (the common path for
/// `pack`, `from_raw_parts`, and any in-memory constructor): scans can
/// iterate the `&[u64]` slice directly with no per-element decode.
///
/// `Mapped` is used when the column is constructed from a slice of an
/// mmap-backed file (`from_bytes_storage`). Per-element access pays an
/// 8-byte little-endian decode but no copy occurs.
#[derive(Debug, Clone)]
enum WordStore {
    Inline(Vec<u64>),
    Mapped(Bytes),
}

impl WordStore {
    #[inline]
    fn byte_len(&self) -> usize {
        match self {
            Self::Inline(v) => v.len() * 8,
            Self::Mapped(b) => b.len(),
        }
    }

    #[inline]
    fn word_count(&self) -> usize {
        match self {
            Self::Inline(v) => v.len(),
            Self::Mapped(b) => b.len() / 8,
        }
    }

    #[inline]
    fn as_slice(&self) -> Option<&[u64]> {
        match self {
            Self::Inline(v) => Some(v.as_slice()),
            Self::Mapped(_) => None,
        }
    }

    #[inline]
    fn word_at(&self, idx: usize) -> Option<u64> {
        match self {
            Self::Inline(v) => v.get(idx).copied(),
            Self::Mapped(b) => {
                let start = idx.checked_mul(8)?;
                let end = start.checked_add(8)?;
                let chunk: [u8; 8] = b.get(start..end)?.try_into().ok()?;
                Some(u64::from_le_bytes(chunk))
            }
        }
    }

    /// Returns a `Bytes` view for serialization. `Inline` materialises a
    /// fresh LE-encoded buffer; `Mapped` returns the existing refcount.
    fn to_bytes(&self) -> Bytes {
        match self {
            Self::Inline(v) => words_to_bytes(v),
            Self::Mapped(b) => b.clone(),
        }
    }
}

/// Stores integers using only as many bits as the largest value needs.
///
/// Pass your values to [`pack()`](Self::pack) and we'll figure out the optimal
/// bit width automatically. Random access via [`get()`](Self::get) is O(1).
///
/// # Storage
///
/// Phase 3c: word storage has two shapes.
/// `Inline(Vec<u64>)` is used for in-RAM builds (`pack`, `from_raw_parts`)
/// so scans can iterate the `&[u64]` slice directly with no per-element
/// decode. `Mapped(Bytes)` is used for mmap-backed loads
/// (`from_bytes_storage`) so the column shares the underlying allocation
/// without copying. Per-element access via [`word_at`](Self::word_at)
/// works for both; callers that want zero-decode iteration should use
/// [`as_words_slice`](Self::as_words_slice) and fall back to `word_at`
/// when it returns `None`.
#[derive(Debug, Clone)]
pub struct BitPackedInts {
    /// Packed `u64` words: `Inline(Vec<u64>)` for RAM builds, `Mapped(Bytes)`
    /// for mmap-backed loads. Scans branch once at the call site to use the
    /// slice fast path when available.
    data: WordStore,
    /// Number of bits per value.
    bits_per_value: u8,
    /// Number of values.
    count: usize,
}

impl BitPackedInts {
    /// Reconstructs from pre-packed raw parts.
    ///
    /// Used by section deserialization. The caller is responsible for ensuring
    /// the data is consistent (correct word count for the given bits and count).
    #[must_use]
    pub fn from_raw_parts(data: Vec<u64>, bits_per_value: u8, count: usize) -> Self {
        Self {
            data: WordStore::Inline(data),
            bits_per_value,
            count,
        }
    }

    /// Reconstructs from pre-encoded bytes (Phase 3c entry point).
    ///
    /// The byte slice must be `word_count * 8` bytes of little-endian
    /// `u64` words. Used by the mmap path so a column can hold a slice
    /// of mapped memory without copying.
    #[must_use]
    pub fn from_bytes_storage(data: Bytes, bits_per_value: u8, count: usize) -> Self {
        Self {
            data: WordStore::Mapped(data),
            bits_per_value,
            count,
        }
    }

    /// Packs a slice of u64 values using the minimum bits needed.
    #[must_use]
    pub fn pack(values: &[u64]) -> Self {
        if values.is_empty() {
            return Self {
                data: WordStore::Inline(Vec::new()),
                bits_per_value: 0,
                count: 0,
            };
        }

        let max_value = values.iter().copied().max().unwrap_or(0);
        let bits = Self::bits_needed(max_value);
        Self::pack_with_bits(values, bits)
    }

    /// Packs values using a specified bit width.
    ///
    /// # Panics
    ///
    /// Panics if any value doesn't fit in the specified bit width.
    #[must_use]
    pub fn pack_with_bits(values: &[u64], bits_per_value: u8) -> Self {
        if values.is_empty() {
            return Self {
                data: WordStore::Inline(Vec::new()),
                bits_per_value,
                count: 0,
            };
        }

        if bits_per_value == 0 {
            // All values must be 0
            debug_assert!(values.iter().all(|&v| v == 0));
            return Self {
                data: WordStore::Inline(Vec::new()),
                bits_per_value: 0,
                count: values.len(),
            };
        }

        let bits = bits_per_value as usize;
        let values_per_word = 64 / bits;
        let num_words = values.len().div_ceil(values_per_word);

        let mut words = vec![0u64; num_words];
        let mask = if bits >= 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };

        for (i, &value) in values.iter().enumerate() {
            debug_assert!(
                value <= mask,
                "Value {} doesn't fit in {} bits",
                value,
                bits_per_value
            );

            let word_idx = i / values_per_word;
            let bit_offset = (i % values_per_word) * bits;
            words[word_idx] |= (value & mask) << bit_offset;
        }

        Self {
            data: WordStore::Inline(words),
            bits_per_value,
            count: values.len(),
        }
    }

    /// Unpacks all values back to u64.
    #[must_use]
    pub fn unpack(&self) -> Vec<u64> {
        if self.count == 0 {
            return Vec::new();
        }

        if self.bits_per_value == 0 {
            return vec![0u64; self.count];
        }

        let bits = self.bits_per_value as usize;
        let values_per_word = 64 / bits;
        let mask = if bits >= 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };

        let mut result = Vec::with_capacity(self.count);

        for i in 0..self.count {
            let word_idx = i / values_per_word;
            let bit_offset = (i % values_per_word) * bits;
            let word = self.word_at(word_idx).unwrap_or(0);
            let value = (word >> bit_offset) & mask;
            result.push(value);
        }

        result
    }

    /// Gets a single value at the given index.
    #[must_use]
    #[inline]
    pub fn get(&self, index: usize) -> Option<u64> {
        if index >= self.count {
            return None;
        }

        if self.bits_per_value == 0 {
            return Some(0);
        }

        let bits = self.bits_per_value as usize;
        let values_per_word = 64 / bits;
        let word_idx = index / values_per_word;
        let bit_offset = (index % values_per_word) * bits;
        let mask = if bits >= 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };

        let word = match &self.data {
            WordStore::Inline(v) => *v.get(word_idx)?,
            // Duplicates WordStore::word_at's Mapped arm intentionally so LLVM sees
            // the full hot path without a call boundary. Keep in sync.
            WordStore::Mapped(b) => {
                let start = word_idx.checked_mul(8)?;
                let end = start.checked_add(8)?;
                let chunk: [u8; 8] = b.get(start..end)?.try_into().ok()?;
                u64::from_le_bytes(chunk)
            }
        };
        Some((word >> bit_offset) & mask)
    }

    /// Returns the row offsets where the unpacked value equals `target`.
    ///
    /// Hoists the `WordStore` discriminant check, the `bits_per_value`
    /// constants, and the `values_per_word` / `mask` computation out of the
    /// per-row loop. CodSpeed callgrind counts every retired instruction, so
    /// re-deriving these per row (as `(0..len).filter(|i| get(*i) == ...)` did)
    /// inflated the instruction count materially even though wall-time on
    /// real CPUs was unaffected.
    #[must_use]
    pub fn scan_eq(&self, target: u64) -> Vec<usize> {
        if self.count == 0 {
            return Vec::new();
        }

        if self.bits_per_value == 0 {
            return if target == 0 {
                (0..self.count).collect()
            } else {
                Vec::new()
            };
        }

        let bits = self.bits_per_value as usize;
        // bits is in 1..=64 here (constructor guarantees), so values_per_word is 1..=64.
        let values_per_word = 64 / bits;
        let mask = if bits >= 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };

        // Bail if the target can't be represented in `bits` bits — no row
        // can possibly match. Saves the entire scan.
        if mask != u64::MAX && target & !mask != 0 {
            return Vec::new();
        }

        match &self.data {
            WordStore::Inline(words) => {
                let mut out = Vec::new();
                for i in 0..self.count {
                    let word_idx = i / values_per_word;
                    let bit_offset = (i % values_per_word) * bits;
                    // `word_idx < words.len()` is guaranteed by the
                    // count/bits invariant set up at construction time;
                    // use `get` defensively rather than panic-on-malformed.
                    let Some(&word) = words.get(word_idx) else {
                        break;
                    };
                    if (word >> bit_offset) & mask == target {
                        out.push(i);
                    }
                }
                out
            }
            WordStore::Mapped(b) => {
                let mut out = Vec::new();
                let bytes = b.as_ref();
                for i in 0..self.count {
                    let word_idx = i / values_per_word;
                    let bit_offset = (i % values_per_word) * bits;
                    let start = word_idx * 8;
                    let Some(slice) = bytes.get(start..start + 8) else {
                        break;
                    };
                    let Ok(chunk) = <[u8; 8]>::try_from(slice) else {
                        break;
                    };
                    let word = u64::from_le_bytes(chunk);
                    if (word >> bit_offset) & mask == target {
                        out.push(i);
                    }
                }
                out
            }
        }
    }

    /// Returns the number of values.
    #[must_use]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns whether the encoding is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Returns the number of bits per value.
    #[must_use]
    pub fn bits_per_value(&self) -> u8 {
        self.bits_per_value
    }

    /// Returns the raw packed bytes.
    ///
    /// Phase 3b/3c: when storage is `Inline`, we materialise an LE-encoded
    /// view; when storage is `Mapped`, we return the existing refcount.
    /// Callers that hold the result must accept either ownership shape.
    #[must_use]
    pub fn data_bytes(&self) -> Bytes {
        self.data.to_bytes()
    }

    /// Returns the number of `u64` words backing this column.
    #[must_use]
    pub fn word_count(&self) -> usize {
        self.data.word_count()
    }

    /// Returns a direct `&[u64]` view when the column lives in RAM.
    ///
    /// `Some(slice)` means the caller can iterate words without per-element
    /// decode. `None` means the column is mmap-backed; callers should fall
    /// back to [`Self::word_at`] (one safe LE decode per access).
    #[must_use]
    #[inline]
    pub fn as_words_slice(&self) -> Option<&[u64]> {
        self.data.as_slice()
    }

    /// Returns the word at `idx`, or `None` if out of range.
    #[must_use]
    #[inline]
    pub fn word_at(&self, idx: usize) -> Option<u64> {
        self.data.word_at(idx)
    }

    /// Returns the compression ratio compared to storing full u64s.
    #[must_use]
    pub fn compression_ratio(&self) -> f64 {
        if self.count == 0 {
            return 1.0;
        }

        let original_size = self.count * 8; // 8 bytes per u64
        let packed_size = self.data.byte_len();

        if packed_size == 0 {
            return f64::INFINITY; // All zeros, perfect compression
        }

        original_size as f64 / packed_size as f64
    }

    /// Returns the number of bits needed to represent a value.
    ///
    /// The result is always in `1..=64`.
    ///
    /// # Panics
    ///
    /// Cannot panic: the result of `64 - leading_zeros()` is always in
    /// `1..=64`, which fits `u8`.
    #[must_use]
    pub fn bits_needed(value: u64) -> u8 {
        if value == 0 {
            1 // Need at least 1 bit to represent 0
        } else {
            // leading_zeros() returns u32 in [0, 63] for non-zero u64;
            // (64 - n) is in [1, 64], which always fits u8.
            u8::try_from(64u32 - value.leading_zeros()).expect("bits_needed result is in 1..=64")
        }
    }

    /// Serializes to bytes.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the value count exceeds `u32::MAX`.
    pub fn to_bytes(&self) -> io::Result<Vec<u8>> {
        let count_u32 = u32::try_from(self.count).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "BitPackedInts count {} exceeds u32::MAX, cannot serialize",
                    self.count
                ),
            )
        })?;
        let mut buf = Vec::with_capacity(1 + 4 + self.data.byte_len());
        buf.push(self.bits_per_value);
        buf.extend_from_slice(&count_u32.to_le_bytes());
        // Storage is little-endian words: borrow from `Mapped`, or build a
        // fresh LE buffer for `Inline`.
        let body = self.data.to_bytes();
        buf.extend_from_slice(&body);
        Ok(buf)
    }

    /// Deserializes from bytes.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the byte slice is too short or contains invalid data.
    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < 5 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "BitPackedInts too short",
            ));
        }

        let bits_per_value = bytes[0];
        if bits_per_value > 64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("BitPackedInts bits_per_value {bits_per_value} exceeds 64"),
            ));
        }
        let count = u32::from_le_bytes(
            bytes[1..5]
                .try_into()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?,
        ) as usize;

        let num_words = if bits_per_value == 0 || count == 0 {
            0
        } else {
            let values_per_word = 64 / bits_per_value as usize;
            (count + values_per_word - 1) / values_per_word
        };

        let needed = 5 + num_words * 8;
        if bytes.len() < needed {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "BitPackedInts truncated",
            ));
        }

        // Note: this always produces Mapped storage. Callers performing
        // in-memory deserialisation that have a Vec<u64> in hand should
        // prefer from_raw_parts() to keep the slice fast path.
        let data = WordStore::Mapped(Bytes::copy_from_slice(&bytes[5..needed]));

        Ok(Self {
            data,
            bits_per_value,
            count,
        })
    }
}

/// The best compression for sorted integers - delta encoding plus bit-packing.
///
/// Stores the first value, then packs the differences between consecutive values.
/// For sequential IDs like [1000, 1001, 1002, ...], deltas are all 1, needing just
/// 1 bit each - that's up to 64x compression!
#[derive(Debug, Clone)]
pub struct DeltaBitPacked {
    /// Base value (first value in sequence).
    base: u64,
    /// Bit-packed deltas.
    deltas: BitPackedInts,
}

impl DeltaBitPacked {
    /// Encodes sorted values using delta + bit-packing.
    #[must_use]
    pub fn encode(values: &[u64]) -> Self {
        if values.is_empty() {
            return Self {
                base: 0,
                deltas: BitPackedInts::pack(&[]),
            };
        }

        let base = values[0];
        let delta_values: Vec<u64> = values
            .windows(2)
            .map(|w| w[1].saturating_sub(w[0]))
            .collect();

        let deltas = BitPackedInts::pack(&delta_values);

        Self { base, deltas }
    }

    /// Decodes back to the original values.
    #[must_use]
    pub fn decode(&self) -> Vec<u64> {
        if self.deltas.is_empty() && self.base == 0 {
            return Vec::new();
        }

        let delta_values = self.deltas.unpack();
        let mut result = Vec::with_capacity(delta_values.len() + 1);
        let mut current = self.base;
        result.push(current);

        for delta in delta_values {
            current = current.wrapping_add(delta);
            result.push(current);
        }

        result
    }

    /// Returns the number of values.
    #[must_use]
    pub fn len(&self) -> usize {
        if self.deltas.is_empty() && self.base == 0 {
            0
        } else {
            self.deltas.len() + 1
        }
    }

    /// Returns whether the encoding is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.deltas.is_empty() && self.base == 0
    }

    /// Returns the base value.
    #[must_use]
    pub fn base(&self) -> u64 {
        self.base
    }

    /// Returns the bits used per delta.
    #[must_use]
    pub fn bits_per_delta(&self) -> u8 {
        self.deltas.bits_per_value()
    }

    /// Returns the compression ratio.
    #[must_use]
    pub fn compression_ratio(&self) -> f64 {
        let count = self.len();
        if count == 0 {
            return 1.0;
        }

        let original_size = count * 8;
        let packed_size = 8 + self.deltas.data_bytes().len(); // base + packed deltas

        original_size as f64 / packed_size as f64
    }

    /// Serializes to bytes.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the delta count exceeds `u32::MAX`.
    pub fn to_bytes(&self) -> io::Result<Vec<u8>> {
        let delta_bytes = self.deltas.to_bytes()?;
        let mut buf = Vec::with_capacity(8 + delta_bytes.len());
        buf.extend_from_slice(&self.base.to_le_bytes());
        buf.extend_from_slice(&delta_bytes);
        Ok(buf)
    }

    /// Deserializes from bytes.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the byte slice is too short or contains invalid data.
    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < 8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "DeltaBitPacked too short",
            ));
        }

        let base = u64::from_le_bytes(
            bytes[0..8]
                .try_into()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?,
        );
        let deltas = BitPackedInts::from_bytes(&bytes[8..])?;

        Ok(Self { base, deltas })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitpack_basic() {
        let values = vec![5u64, 2, 3, 5, 5, 8, 2];
        let packed = BitPackedInts::pack(&values);
        let unpacked = packed.unpack();
        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_bitpack_empty() {
        let values: Vec<u64> = vec![];
        let packed = BitPackedInts::pack(&values);
        assert!(packed.is_empty());
        assert_eq!(packed.unpack(), values);
    }

    #[test]
    fn test_bitpack_single() {
        let values = vec![42u64];
        let packed = BitPackedInts::pack(&values);
        assert_eq!(packed.len(), 1);
        assert_eq!(packed.unpack(), values);
    }

    #[test]
    fn test_bitpack_all_zeros() {
        let values = vec![0u64; 100];
        let packed = BitPackedInts::pack(&values);
        assert_eq!(packed.bits_per_value(), 1);
        assert_eq!(packed.unpack(), values);
    }

    #[test]
    fn test_bitpack_powers_of_two() {
        for bits in 1..=64u8 {
            let max_val = if bits == 64 {
                u64::MAX
            } else {
                (1u64 << bits) - 1
            };
            let values = vec![0, max_val / 2, max_val];
            let packed = BitPackedInts::pack(&values);
            assert_eq!(packed.bits_per_value(), bits);
            assert_eq!(packed.unpack(), values);
        }
    }

    #[test]
    fn test_bitpack_get() {
        let values = vec![1u64, 2, 3, 4, 5];
        let packed = BitPackedInts::pack(&values);

        for (i, &expected) in values.iter().enumerate() {
            assert_eq!(packed.get(i), Some(expected));
        }
        assert_eq!(packed.get(100), None);
    }

    #[test]
    fn test_bitpack_compression() {
        // 100 values all <= 15 (4 bits each)
        let values: Vec<u64> = (0..100).map(|i| i % 16).collect();
        let packed = BitPackedInts::pack(&values);
        assert_eq!(packed.bits_per_value(), 4);
        // 100 * 64 bits -> 100 * 4 bits = 16x compression
        let ratio = packed.compression_ratio();
        assert!(ratio > 10.0, "Expected ratio > 10, got {}", ratio);
    }

    #[test]
    fn test_bitpack_serialization() {
        let values = vec![1u64, 3, 7, 15, 31];
        let packed = BitPackedInts::pack(&values);
        let bytes = packed.to_bytes().unwrap();
        let restored = BitPackedInts::from_bytes(&bytes).unwrap();
        assert_eq!(packed.unpack(), restored.unpack());
    }

    #[test]
    fn test_delta_bitpacked_basic() {
        let values = vec![100u64, 105, 107, 110, 115, 120, 128, 130];
        let encoded = DeltaBitPacked::encode(&values);
        let decoded = encoded.decode();
        assert_eq!(values, decoded);
    }

    #[test]
    fn test_delta_bitpacked_sequential() {
        // Sequential values: deltas are all 1, needs only 1 bit each
        let values: Vec<u64> = (1000..1100).collect();
        let encoded = DeltaBitPacked::encode(&values);
        assert_eq!(encoded.bits_per_delta(), 1);
        assert_eq!(encoded.decode(), values);

        // Great compression: 100 * 64 bits -> 8 (base) + ~100 bits
        let ratio = encoded.compression_ratio();
        assert!(ratio > 5.0, "Expected ratio > 5, got {}", ratio);
    }

    #[test]
    fn test_delta_bitpacked_empty() {
        let values: Vec<u64> = vec![];
        let encoded = DeltaBitPacked::encode(&values);
        assert!(encoded.is_empty());
        assert_eq!(encoded.decode(), values);
    }

    #[test]
    fn test_delta_bitpacked_single() {
        let values = vec![42u64];
        let encoded = DeltaBitPacked::encode(&values);
        assert_eq!(encoded.len(), 1);
        assert_eq!(encoded.decode(), values);
    }

    #[test]
    fn test_delta_bitpacked_serialization() {
        let values = vec![100u64, 105, 107, 110, 115];
        let encoded = DeltaBitPacked::encode(&values);
        let bytes = encoded.to_bytes().unwrap();
        let restored = DeltaBitPacked::from_bytes(&bytes).unwrap();
        assert_eq!(encoded.decode(), restored.decode());
    }

    #[test]
    fn test_bits_needed() {
        assert_eq!(BitPackedInts::bits_needed(0), 1);
        assert_eq!(BitPackedInts::bits_needed(1), 1);
        assert_eq!(BitPackedInts::bits_needed(2), 2);
        assert_eq!(BitPackedInts::bits_needed(3), 2);
        assert_eq!(BitPackedInts::bits_needed(4), 3);
        assert_eq!(BitPackedInts::bits_needed(7), 3);
        assert_eq!(BitPackedInts::bits_needed(8), 4);
        assert_eq!(BitPackedInts::bits_needed(255), 8);
        assert_eq!(BitPackedInts::bits_needed(256), 9);
        assert_eq!(BitPackedInts::bits_needed(u64::MAX), 64);
    }

    // ── Phase 3b: Bytes-backed storage ────────────────────────────────

    #[test]
    fn test_bitpack_word_at_returns_words_from_bytes() {
        // Pack 5 values at 4 bits each → fits in one u64 word.
        let packed = BitPackedInts::pack(&[1u64, 3, 7, 15, 4]);
        assert_eq!(packed.word_count(), 1);
        let word = packed.word_at(0).unwrap();
        // 4-bit values packed LE: bit 0..3 = 1, 4..7 = 3, 8..11 = 7, 12..15 = 15, 16..19 = 4.
        assert_eq!(word & 0xF, 1);
        assert_eq!((word >> 4) & 0xF, 3);
        assert_eq!((word >> 8) & 0xF, 7);
        assert_eq!((word >> 12) & 0xF, 15);
        assert_eq!((word >> 16) & 0xF, 4);
    }

    #[test]
    fn test_bitpack_word_at_out_of_range_returns_none() {
        let packed = BitPackedInts::pack(&[1u64, 2, 3]);
        assert!(packed.word_at(packed.word_count()).is_none());
        assert!(packed.word_at(usize::MAX).is_none());
    }

    #[test]
    fn test_bitpack_data_bytes_length_matches_word_count() {
        let packed = BitPackedInts::pack_with_bits(
            &(0u64..200).collect::<Vec<_>>(),
            8, // 200 values × 8 bits = 1600 bits = 25 u64 words = 200 bytes
        );
        assert_eq!(packed.data_bytes().len(), packed.word_count() * 8);
        // Round-trip: read each word via word_at, recombine, unpack still works.
        let unpacked = packed.unpack();
        assert_eq!(unpacked, (0u64..200).collect::<Vec<_>>());
    }

    #[test]
    fn test_bitpack_round_trip_through_to_bytes_and_from_bytes() {
        let values: Vec<u64> = (0u64..50).map(|i| i * 7 % 1024).collect();
        let packed = BitPackedInts::pack(&values);
        let serialized = packed.to_bytes().unwrap();
        let restored = BitPackedInts::from_bytes(&serialized).unwrap();
        assert_eq!(packed.unpack(), restored.unpack());
        assert_eq!(packed.bits_per_value(), restored.bits_per_value());
        assert_eq!(packed.len(), restored.len());
    }

    // ── Inline/Mapped storage variants (Task 1: failing tests for as_words_slice) ──

    #[test]
    fn test_bitpack_inline_storage_word_slice_via_pack() {
        let values: Vec<u64> = (0..1000).map(|i| i % 16).collect();
        let packed = BitPackedInts::pack(&values);

        let slice = packed.as_words_slice();
        assert!(
            slice.is_some(),
            "BitPackedInts::pack must produce Inline storage so scans skip per-element decode"
        );
    }

    #[test]
    fn test_bitpack_inline_storage_word_slice_matches_input() {
        let words = vec![0xDEAD_BEEFu64, 0xCAFE_BABE];
        let packed = BitPackedInts::from_raw_parts(words.clone(), 8, 16);
        let slice = packed.as_words_slice().expect("Vec input is Inline");
        assert_eq!(slice, words.as_slice());
    }

    #[test]
    fn test_bitpack_mapped_storage_no_word_slice() {
        let words: Vec<u64> = vec![1, 2, 3];
        let mut buf = bytes::BytesMut::with_capacity(words.len() * 8);
        for &w in &words {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        let packed = BitPackedInts::from_bytes_storage(buf.freeze(), 8, 24);

        assert!(
            packed.as_words_slice().is_none(),
            "Bytes-backed (mmap) storage cannot expose a u64 slice"
        );
        assert_eq!(packed.word_at(0), Some(1));
        assert_eq!(packed.word_at(2), Some(3));
    }

    #[test]
    fn test_bitpack_get_matches_between_inline_and_mapped_storage() {
        let values: Vec<u64> = (0..2_000).map(|i| (i * 7) % 31).collect();
        let inline = BitPackedInts::pack(&values);

        // Build a Mapped equivalent by re-encoding through bytes.
        let bytes = inline.data_bytes();
        let mapped =
            BitPackedInts::from_bytes_storage(bytes, inline.bits_per_value(), values.len());

        for i in 0..values.len() {
            assert_eq!(inline.get(i), Some(values[i]), "inline mismatch at {i}");
            assert_eq!(mapped.get(i), Some(values[i]), "mapped mismatch at {i}");
        }
        assert_eq!(inline.get(values.len()), None);
        assert_eq!(mapped.get(values.len()), None);
    }

    #[test]
    fn test_scan_eq_matches_get_filter_inline() {
        let values: Vec<u64> = (0..5_000).map(|i| (i * 13) % 97).collect();
        let packed = BitPackedInts::pack(&values);

        for target in [0u64, 1, 50, 96, 97, u64::MAX] {
            let scan: Vec<usize> = packed.scan_eq(target);
            let getf: Vec<usize> = (0..values.len())
                .filter(|&i| packed.get(i) == Some(target))
                .collect();
            assert_eq!(scan, getf, "scan_eq vs get-filter for target {target}");
        }
    }

    #[test]
    fn test_scan_eq_matches_get_filter_mapped() {
        let values: Vec<u64> = (0..5_000).map(|i| (i * 13) % 97).collect();
        let inline = BitPackedInts::pack(&values);
        let bytes = inline.data_bytes();
        let mapped =
            BitPackedInts::from_bytes_storage(bytes, inline.bits_per_value(), values.len());

        for target in [0u64, 1, 50, 96, 97, u64::MAX] {
            let scan: Vec<usize> = mapped.scan_eq(target);
            let getf: Vec<usize> = (0..values.len())
                .filter(|&i| mapped.get(i) == Some(target))
                .collect();
            assert_eq!(
                scan, getf,
                "mapped scan_eq vs get-filter for target {target}"
            );
        }
    }

    #[test]
    fn test_scan_eq_all_zeros_short_circuit() {
        // bits_per_value == 0 path: every row stores 0.
        let packed = BitPackedInts::pack(&vec![0u64; 64]);
        assert_eq!(packed.bits_per_value(), 1);
        // Force the zero-bits path by constructing manually.
        let zero_bits = BitPackedInts::from_raw_parts(Vec::new(), 0, 64);
        assert_eq!(zero_bits.scan_eq(0), (0..64).collect::<Vec<_>>());
        assert!(zero_bits.scan_eq(1).is_empty(), "expected empty");
    }

    #[test]
    fn test_scan_eq_target_exceeds_mask_returns_empty() {
        // 4-bit packing can hold 0..=15. A target of 100 must be rejected
        // up-front without scanning.
        let values: Vec<u64> = (0..10).map(|i| i % 16).collect();
        let packed = BitPackedInts::pack(&values);
        assert!(packed.bits_per_value() <= 4);
        assert!(packed.scan_eq(100).is_empty(), "expected empty");
    }

    #[test]
    fn test_scan_eq_empty_column() {
        let packed = BitPackedInts::pack(&[]);
        assert!(packed.scan_eq(0).is_empty(), "expected empty");
        assert!(packed.scan_eq(42).is_empty(), "expected empty");
    }
}
