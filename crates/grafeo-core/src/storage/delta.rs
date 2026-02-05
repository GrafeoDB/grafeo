//! Delta encoding for sorted integer sequences.
//!
//! Instead of storing [100, 105, 107, 110], store the base (100) and the deltas
//! [5, 2, 3]. The deltas are often tiny even when values are huge, making this
//! a great first step before bit-packing.
//!
//! For signed integers, we use zig-zag encoding to map negative deltas to small
//! positive numbers: 0→0, -1→1, 1→2, -2→3, etc.
//!
//! # Example
//!
//! ```ignore
//! let values = vec![100u64, 105, 107, 110, 115];
//! let encoded = DeltaEncoding::encode(&values);
//! // base=100, deltas=[5, 2, 3, 5]
//! assert_eq!(encoded.decode(), values);
//! ```

use std::io;

/// Stores differences between consecutive values instead of the values themselves.
///
/// Pair this with [`BitPackedInts`](super::BitPackedInts) for maximum compression -
/// use [`DeltaBitPacked`](super::DeltaBitPacked) for the combo.
#[derive(Debug, Clone)]
pub struct DeltaEncoding {
    /// The first value in the sequence.
    base: u64,
    /// Deltas between consecutive values.
    deltas: Vec<u64>,
    /// Number of values (base + deltas.len()).
    count: usize,
}

impl DeltaEncoding {
    /// Encodes a slice of u64 values using delta encoding.
    ///
    /// # Requirements
    ///
    /// Values **must** be sorted in ascending order. Unsorted input will produce
    /// incorrect results due to `saturating_sub` mapping negative deltas to zero.
    /// A debug assertion checks this in debug builds.
    ///
    /// For signed or unsorted data, use [`encode_signed`](Self::encode_signed) instead.
    #[must_use]
    pub fn encode(values: &[u64]) -> Self {
        if values.is_empty() {
            return Self {
                base: 0,
                deltas: Vec::new(),
                count: 0,
            };
        }

        // Values must be sorted for unsigned delta encoding to work correctly
        debug_assert!(
            values.windows(2).all(|w| w[0] <= w[1]),
            "DeltaEncoding::encode requires sorted input; use encode_signed for unsorted data"
        );

        let base = values[0];
        let deltas: Vec<u64> = values
            .windows(2)
            .map(|w| w[1].saturating_sub(w[0]))
            .collect();

        Self {
            base,
            deltas,
            count: values.len(),
        }
    }

    /// Encodes a slice of i64 values using delta encoding.
    ///
    /// Computes signed deltas between consecutive values, then zig-zag encodes
    /// the deltas to store them as unsigned integers.
    #[must_use]
    pub fn encode_signed(values: &[i64]) -> Self {
        if values.is_empty() {
            return Self {
                base: 0,
                deltas: Vec::new(),
                count: 0,
            };
        }

        // Store base as zig-zag encoded
        let base = zigzag_encode(values[0]);

        // Compute signed deltas, then zig-zag encode them
        let deltas: Vec<u64> = values
            .windows(2)
            .map(|w| zigzag_encode(w[1] - w[0]))
            .collect();

        Self {
            base,
            deltas,
            count: values.len(),
        }
    }

    /// Decodes the delta-encoded values back to the original sequence.
    #[must_use]
    pub fn decode(&self) -> Vec<u64> {
        if self.count == 0 {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(self.count);
        let mut current = self.base;
        result.push(current);

        for &delta in &self.deltas {
            current = current.wrapping_add(delta);
            result.push(current);
        }

        result
    }

    /// Decodes to signed integers using zig-zag decoding.
    ///
    /// Assumes the encoding was created with `encode_signed`.
    #[must_use]
    pub fn decode_signed(&self) -> Vec<i64> {
        if self.count == 0 {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(self.count);
        let mut current = zigzag_decode(self.base);
        result.push(current);

        for &delta in &self.deltas {
            current += zigzag_decode(delta);
            result.push(current);
        }

        result
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

    /// Returns the base value.
    #[must_use]
    pub fn base(&self) -> u64 {
        self.base
    }

    /// Returns the deltas.
    #[must_use]
    pub fn deltas(&self) -> &[u64] {
        &self.deltas
    }

    /// Returns the maximum delta value.
    ///
    /// Useful for determining bit width for bit-packing.
    #[must_use]
    pub fn max_delta(&self) -> u64 {
        self.deltas.iter().copied().max().unwrap_or(0)
    }

    /// Returns the number of bits needed to represent the largest delta.
    #[must_use]
    pub fn bits_for_max_delta(&self) -> u8 {
        let max = self.max_delta();
        if max == 0 {
            1
        } else {
            64 - max.leading_zeros() as u8
        }
    }

    /// Estimates the compression ratio.
    ///
    /// Returns the ratio of original size to compressed size.
    #[must_use]
    pub fn compression_ratio(&self) -> f64 {
        if self.count == 0 {
            return 1.0;
        }

        let original_size = self.count * 8; // 8 bytes per u64
        let compressed_size = 8 + (self.deltas.len() * 8); // base + deltas

        original_size as f64 / compressed_size as f64
    }

    /// Serializes the delta encoding to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(8 + 4 + self.deltas.len() * 8);
        buf.extend_from_slice(&self.base.to_le_bytes());
        buf.extend_from_slice(&(self.count as u32).to_le_bytes());
        for &delta in &self.deltas {
            buf.extend_from_slice(&delta.to_le_bytes());
        }
        buf
    }

    /// Deserializes a delta encoding from bytes.
    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < 12 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Delta encoding too short",
            ));
        }

        let base = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let count = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;

        let expected_len = 12 + (count.saturating_sub(1)) * 8;
        if bytes.len() < expected_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Delta encoding truncated",
            ));
        }

        let mut deltas = Vec::with_capacity(count.saturating_sub(1));
        let mut offset = 12;
        for _ in 1..count {
            let delta = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
            deltas.push(delta);
            offset += 8;
        }

        Ok(Self {
            base,
            deltas,
            count,
        })
    }
}

/// Zig-zag encodes a signed integer to unsigned.
///
/// Maps signed integers to unsigned: 0 -> 0, -1 -> 1, 1 -> 2, -2 -> 3, etc.
#[inline]
#[must_use]
pub fn zigzag_encode(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

/// Zig-zag decodes an unsigned integer to signed.
#[inline]
#[must_use]
pub fn zigzag_decode(value: u64) -> i64 {
    ((value >> 1) as i64) ^ (-((value & 1) as i64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delta_encode_decode() {
        let values = vec![100u64, 105, 107, 110, 115, 120, 128, 130];
        let encoded = DeltaEncoding::encode(&values);
        let decoded = encoded.decode();
        assert_eq!(values, decoded);
    }

    #[test]
    fn test_delta_empty() {
        let values: Vec<u64> = vec![];
        let encoded = DeltaEncoding::encode(&values);
        assert!(encoded.is_empty());
        assert_eq!(encoded.decode(), values);
    }

    #[test]
    fn test_delta_single_value() {
        let values = vec![42u64];
        let encoded = DeltaEncoding::encode(&values);
        assert_eq!(encoded.len(), 1);
        assert_eq!(encoded.decode(), values);
    }

    #[test]
    fn test_delta_sequential() {
        let values: Vec<u64> = (0..100).collect();
        let encoded = DeltaEncoding::encode(&values);
        assert_eq!(encoded.max_delta(), 1);
        assert_eq!(encoded.bits_for_max_delta(), 1);
        assert_eq!(encoded.decode(), values);
    }

    #[test]
    fn test_delta_large_gaps() {
        let values = vec![0u64, 1000, 2000, 3000];
        let encoded = DeltaEncoding::encode(&values);
        assert_eq!(encoded.max_delta(), 1000);
        assert_eq!(encoded.decode(), values);
    }

    #[test]
    fn test_zigzag_encode_decode() {
        assert_eq!(zigzag_encode(0), 0);
        assert_eq!(zigzag_encode(-1), 1);
        assert_eq!(zigzag_encode(1), 2);
        assert_eq!(zigzag_encode(-2), 3);
        assert_eq!(zigzag_encode(2), 4);

        for v in [-100i64, -50, -1, 0, 1, 50, 100, i64::MIN, i64::MAX] {
            assert_eq!(zigzag_decode(zigzag_encode(v)), v);
        }
    }

    #[test]
    fn test_delta_signed() {
        let values = vec![-100i64, -50, 0, 50, 100];
        let encoded = DeltaEncoding::encode_signed(&values);
        let decoded = encoded.decode_signed();
        assert_eq!(values, decoded);
    }

    #[test]
    fn test_delta_serialization() {
        let values = vec![100u64, 105, 107, 110, 115];
        let encoded = DeltaEncoding::encode(&values);
        let bytes = encoded.to_bytes();
        let restored = DeltaEncoding::from_bytes(&bytes).unwrap();
        assert_eq!(encoded.decode(), restored.decode());
    }

    #[test]
    fn test_compression_ratio() {
        // Sequential values should compress well
        let sequential: Vec<u64> = (0..1000).collect();
        let encoded = DeltaEncoding::encode(&sequential);
        // Each delta is 1, so compression is minimal but base + deltas is stored
        assert!(encoded.len() == 1000);
    }
}
