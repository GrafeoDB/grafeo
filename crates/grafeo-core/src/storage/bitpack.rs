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
//! ```ignore
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

/// Stores integers using only as many bits as the largest value needs.
///
/// Pass your values to [`pack()`](Self::pack) and we'll figure out the optimal
/// bit width automatically. Random access via [`get()`](Self::get) is O(1).
#[derive(Debug, Clone)]
pub struct BitPackedInts {
    /// Packed data.
    data: Vec<u64>,
    /// Number of bits per value.
    bits_per_value: u8,
    /// Number of values.
    count: usize,
}

impl BitPackedInts {
    /// Packs a slice of u64 values using the minimum bits needed.
    #[must_use]
    pub fn pack(values: &[u64]) -> Self {
        if values.is_empty() {
            return Self {
                data: Vec::new(),
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
                data: Vec::new(),
                bits_per_value,
                count: 0,
            };
        }

        if bits_per_value == 0 {
            // All values must be 0
            debug_assert!(values.iter().all(|&v| v == 0));
            return Self {
                data: Vec::new(),
                bits_per_value: 0,
                count: values.len(),
            };
        }

        let bits = bits_per_value as usize;
        let values_per_word = 64 / bits;
        let num_words = (values.len() + values_per_word - 1) / values_per_word;

        let mut data = vec![0u64; num_words];
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
            data[word_idx] |= (value & mask) << bit_offset;
        }

        Self {
            data,
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
            let value = (self.data[word_idx] >> bit_offset) & mask;
            result.push(value);
        }

        result
    }

    /// Gets a single value at the given index.
    #[must_use]
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

        Some((self.data[word_idx] >> bit_offset) & mask)
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

    /// Returns the raw packed data.
    #[must_use]
    pub fn data(&self) -> &[u64] {
        &self.data
    }

    /// Returns the compression ratio compared to storing full u64s.
    #[must_use]
    pub fn compression_ratio(&self) -> f64 {
        if self.count == 0 {
            return 1.0;
        }

        let original_size = self.count * 8; // 8 bytes per u64
        let packed_size = self.data.len() * 8;

        if packed_size == 0 {
            return f64::INFINITY; // All zeros, perfect compression
        }

        original_size as f64 / packed_size as f64
    }

    /// Returns the number of bits needed to represent a value.
    #[must_use]
    pub fn bits_needed(value: u64) -> u8 {
        if value == 0 {
            1 // Need at least 1 bit to represent 0
        } else {
            64 - value.leading_zeros() as u8
        }
    }

    /// Serializes to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(1 + 4 + self.data.len() * 8);
        buf.push(self.bits_per_value);
        buf.extend_from_slice(&(self.count as u32).to_le_bytes());
        for &word in &self.data {
            buf.extend_from_slice(&word.to_le_bytes());
        }
        buf
    }

    /// Deserializes from bytes.
    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < 5 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "BitPackedInts too short",
            ));
        }

        let bits_per_value = bytes[0];
        let count = u32::from_le_bytes(bytes[1..5].try_into().unwrap()) as usize;

        let num_words = if bits_per_value == 0 || count == 0 {
            0
        } else {
            let values_per_word = 64 / bits_per_value as usize;
            (count + values_per_word - 1) / values_per_word
        };

        if bytes.len() < 5 + num_words * 8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "BitPackedInts truncated",
            ));
        }

        let mut data = Vec::with_capacity(num_words);
        for i in 0..num_words {
            let offset = 5 + i * 8;
            let word = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
            data.push(word);
        }

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
        let packed_size = 8 + self.deltas.data().len() * 8; // base + packed deltas

        original_size as f64 / packed_size as f64
    }

    /// Serializes to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let delta_bytes = self.deltas.to_bytes();
        let mut buf = Vec::with_capacity(8 + delta_bytes.len());
        buf.extend_from_slice(&self.base.to_le_bytes());
        buf.extend_from_slice(&delta_bytes);
        buf
    }

    /// Deserializes from bytes.
    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < 8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "DeltaBitPacked too short",
            ));
        }

        let base = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
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
        let bytes = packed.to_bytes();
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
        let bytes = encoded.to_bytes();
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
}
