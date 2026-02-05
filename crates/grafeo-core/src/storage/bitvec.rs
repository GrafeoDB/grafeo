//! Stores booleans as individual bits - 8x smaller than `Vec<bool>`.
//!
//! Use this when you're tracking lots of boolean flags (like "visited" markers
//! in graph traversals, or null bitmaps). Backed by `Vec<u64>` so bitwise
//! operations like AND/OR/XOR stay cache-friendly.
//!
//! # Example
//!
//! ```ignore
//! let bools = vec![true, false, true, true, false, false, true, false];
//! let bitvec = BitVector::from_bools(&bools);
//! // Stored as: 0b01001101 (1 byte instead of 8)
//!
//! assert_eq!(bitvec.get(0), Some(true));
//! assert_eq!(bitvec.get(1), Some(false));
//! assert_eq!(bitvec.count_ones(), 4);
//! ```

use std::io;

/// Stores booleans as individual bits - 8x smaller than `Vec<bool>`.
///
/// Supports bitwise operations ([`and`](Self::and), [`or`](Self::or),
/// [`not`](Self::not)) for combining filter results efficiently.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitVector {
    /// Packed bits (little-endian within each word).
    data: Vec<u64>,
    /// Number of bits stored.
    len: usize,
}

impl BitVector {
    /// Creates an empty bit vector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            len: 0,
        }
    }

    /// Creates a bit vector with the specified capacity (in bits).
    #[must_use]
    pub fn with_capacity(bits: usize) -> Self {
        let words = (bits + 63) / 64;
        Self {
            data: Vec::with_capacity(words),
            len: 0,
        }
    }

    /// Creates a bit vector from a slice of booleans.
    #[must_use]
    pub fn from_bools(bools: &[bool]) -> Self {
        let num_words = (bools.len() + 63) / 64;
        let mut data = vec![0u64; num_words];

        for (i, &b) in bools.iter().enumerate() {
            if b {
                let word_idx = i / 64;
                let bit_idx = i % 64;
                data[word_idx] |= 1 << bit_idx;
            }
        }

        Self {
            data,
            len: bools.len(),
        }
    }

    /// Creates a bit vector with all bits set to the same value.
    #[must_use]
    pub fn filled(len: usize, value: bool) -> Self {
        let num_words = (len + 63) / 64;
        let fill = if value { u64::MAX } else { 0 };
        let data = vec![fill; num_words];

        Self { data, len }
    }

    /// Creates a bit vector with all bits set to false (0).
    #[must_use]
    pub fn zeros(len: usize) -> Self {
        Self::filled(len, false)
    }

    /// Creates a bit vector with all bits set to true (1).
    #[must_use]
    pub fn ones(len: usize) -> Self {
        Self::filled(len, true)
    }

    /// Returns the number of bits.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the bit vector is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Gets the bit at the given index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<bool> {
        if index >= self.len {
            return None;
        }

        let word_idx = index / 64;
        let bit_idx = index % 64;
        Some((self.data[word_idx] & (1 << bit_idx)) != 0)
    }

    /// Sets the bit at the given index.
    ///
    /// # Panics
    ///
    /// Panics if index >= len.
    pub fn set(&mut self, index: usize, value: bool) {
        assert!(index < self.len, "Index out of bounds");

        let word_idx = index / 64;
        let bit_idx = index % 64;

        if value {
            self.data[word_idx] |= 1 << bit_idx;
        } else {
            self.data[word_idx] &= !(1 << bit_idx);
        }
    }

    /// Appends a bit to the end.
    pub fn push(&mut self, value: bool) {
        let word_idx = self.len / 64;
        let bit_idx = self.len % 64;

        if word_idx >= self.data.len() {
            self.data.push(0);
        }

        if value {
            self.data[word_idx] |= 1 << bit_idx;
        }

        self.len += 1;
    }

    /// Returns the number of bits set to true.
    #[must_use]
    pub fn count_ones(&self) -> usize {
        if self.is_empty() {
            return 0;
        }

        let full_words = self.len / 64;
        let remaining_bits = self.len % 64;

        let mut count: usize = self.data[..full_words]
            .iter()
            .map(|&w| w.count_ones() as usize)
            .sum();

        if remaining_bits > 0 && full_words < self.data.len() {
            let mask = (1u64 << remaining_bits) - 1;
            count += (self.data[full_words] & mask).count_ones() as usize;
        }

        count
    }

    /// Returns the number of bits set to false.
    #[must_use]
    pub fn count_zeros(&self) -> usize {
        self.len - self.count_ones()
    }

    /// Converts back to a `Vec<bool>`.
    #[must_use]
    pub fn to_bools(&self) -> Vec<bool> {
        (0..self.len).map(|i| self.get(i).unwrap()).collect()
    }

    /// Returns an iterator over the bits.
    pub fn iter(&self) -> impl Iterator<Item = bool> + '_ {
        (0..self.len).map(move |i| self.get(i).unwrap())
    }

    /// Returns an iterator over indices where bits are true.
    pub fn ones_iter(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.len).filter(move |&i| self.get(i).unwrap())
    }

    /// Returns an iterator over indices where bits are false.
    pub fn zeros_iter(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.len).filter(move |&i| !self.get(i).unwrap())
    }

    /// Returns the raw data.
    #[must_use]
    pub fn data(&self) -> &[u64] {
        &self.data
    }

    /// Returns the compression ratio (original bytes / compressed bytes).
    #[must_use]
    pub fn compression_ratio(&self) -> f64 {
        if self.is_empty() {
            return 1.0;
        }

        // Original: 1 byte per bool
        let original_size = self.len;
        // Compressed: ceil(len / 8) bytes
        let compressed_size = self.data.len() * 8;

        if compressed_size == 0 {
            return 1.0;
        }

        original_size as f64 / compressed_size as f64
    }

    /// Performs bitwise AND with another bit vector.
    ///
    /// The result has the length of the shorter vector.
    #[must_use]
    pub fn and(&self, other: &Self) -> Self {
        let len = self.len.min(other.len);
        let num_words = (len + 63) / 64;

        let data: Vec<u64> = self
            .data
            .iter()
            .zip(&other.data)
            .take(num_words)
            .map(|(&a, &b)| a & b)
            .collect();

        Self { data, len }
    }

    /// Performs bitwise OR with another bit vector.
    ///
    /// The result has the length of the shorter vector.
    #[must_use]
    pub fn or(&self, other: &Self) -> Self {
        let len = self.len.min(other.len);
        let num_words = (len + 63) / 64;

        let data: Vec<u64> = self
            .data
            .iter()
            .zip(&other.data)
            .take(num_words)
            .map(|(&a, &b)| a | b)
            .collect();

        Self { data, len }
    }

    /// Performs bitwise NOT.
    #[must_use]
    pub fn not(&self) -> Self {
        let data: Vec<u64> = self.data.iter().map(|&w| !w).collect();
        Self {
            data,
            len: self.len,
        }
    }

    /// Performs bitwise XOR with another bit vector.
    #[must_use]
    pub fn xor(&self, other: &Self) -> Self {
        let len = self.len.min(other.len);
        let num_words = (len + 63) / 64;

        let data: Vec<u64> = self
            .data
            .iter()
            .zip(&other.data)
            .take(num_words)
            .map(|(&a, &b)| a ^ b)
            .collect();

        Self { data, len }
    }

    /// Serializes to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(4 + self.data.len() * 8);
        buf.extend_from_slice(&(self.len as u32).to_le_bytes());
        for &word in &self.data {
            buf.extend_from_slice(&word.to_le_bytes());
        }
        buf
    }

    /// Deserializes from bytes.
    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < 4 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "BitVector too short",
            ));
        }

        let len = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
        let num_words = (len + 63) / 64;

        if bytes.len() < 4 + num_words * 8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "BitVector truncated",
            ));
        }

        let mut data = Vec::with_capacity(num_words);
        for i in 0..num_words {
            let offset = 4 + i * 8;
            let word = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
            data.push(word);
        }

        Ok(Self { data, len })
    }
}

impl Default for BitVector {
    fn default() -> Self {
        Self::new()
    }
}

impl FromIterator<bool> for BitVector {
    fn from_iter<T: IntoIterator<Item = bool>>(iter: T) -> Self {
        let mut bitvec = BitVector::new();
        for b in iter {
            bitvec.push(b);
        }
        bitvec
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitvec_basic() {
        let bools = vec![true, false, true, true, false, false, true, false];
        let bitvec = BitVector::from_bools(&bools);

        assert_eq!(bitvec.len(), 8);
        for (i, &expected) in bools.iter().enumerate() {
            assert_eq!(bitvec.get(i), Some(expected));
        }
    }

    #[test]
    fn test_bitvec_empty() {
        let bitvec = BitVector::new();
        assert!(bitvec.is_empty());
        assert_eq!(bitvec.get(0), None);
    }

    #[test]
    fn test_bitvec_push() {
        let mut bitvec = BitVector::new();
        bitvec.push(true);
        bitvec.push(false);
        bitvec.push(true);

        assert_eq!(bitvec.len(), 3);
        assert_eq!(bitvec.get(0), Some(true));
        assert_eq!(bitvec.get(1), Some(false));
        assert_eq!(bitvec.get(2), Some(true));
    }

    #[test]
    fn test_bitvec_set() {
        let mut bitvec = BitVector::zeros(8);

        bitvec.set(0, true);
        bitvec.set(3, true);
        bitvec.set(7, true);

        assert_eq!(bitvec.get(0), Some(true));
        assert_eq!(bitvec.get(1), Some(false));
        assert_eq!(bitvec.get(3), Some(true));
        assert_eq!(bitvec.get(7), Some(true));
    }

    #[test]
    fn test_bitvec_count() {
        let bools = vec![true, false, true, true, false, false, true, false];
        let bitvec = BitVector::from_bools(&bools);

        assert_eq!(bitvec.count_ones(), 4);
        assert_eq!(bitvec.count_zeros(), 4);
    }

    #[test]
    fn test_bitvec_filled() {
        let zeros = BitVector::zeros(100);
        assert_eq!(zeros.count_ones(), 0);
        assert_eq!(zeros.count_zeros(), 100);

        let ones = BitVector::ones(100);
        assert_eq!(ones.count_ones(), 100);
        assert_eq!(ones.count_zeros(), 0);
    }

    #[test]
    fn test_bitvec_to_bools() {
        let original = vec![true, false, true, true, false];
        let bitvec = BitVector::from_bools(&original);
        let restored = bitvec.to_bools();
        assert_eq!(original, restored);
    }

    #[test]
    fn test_bitvec_large() {
        // Test with more than 64 bits
        let bools: Vec<bool> = (0..200).map(|i| i % 3 == 0).collect();
        let bitvec = BitVector::from_bools(&bools);

        assert_eq!(bitvec.len(), 200);
        for (i, &expected) in bools.iter().enumerate() {
            assert_eq!(bitvec.get(i), Some(expected), "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_bitvec_and() {
        let a = BitVector::from_bools(&[true, true, false, false]);
        let b = BitVector::from_bools(&[true, false, true, false]);
        let result = a.and(&b);

        assert_eq!(result.to_bools(), vec![true, false, false, false]);
    }

    #[test]
    fn test_bitvec_or() {
        let a = BitVector::from_bools(&[true, true, false, false]);
        let b = BitVector::from_bools(&[true, false, true, false]);
        let result = a.or(&b);

        assert_eq!(result.to_bools(), vec![true, true, true, false]);
    }

    #[test]
    fn test_bitvec_not() {
        let a = BitVector::from_bools(&[true, false, true, false]);
        let result = a.not();

        // Note: NOT inverts all bits in the word, so we check the relevant bits
        assert_eq!(result.get(0), Some(false));
        assert_eq!(result.get(1), Some(true));
        assert_eq!(result.get(2), Some(false));
        assert_eq!(result.get(3), Some(true));
    }

    #[test]
    fn test_bitvec_xor() {
        let a = BitVector::from_bools(&[true, true, false, false]);
        let b = BitVector::from_bools(&[true, false, true, false]);
        let result = a.xor(&b);

        assert_eq!(result.to_bools(), vec![false, true, true, false]);
    }

    #[test]
    fn test_bitvec_serialization() {
        let bools = vec![true, false, true, true, false, false, true, false];
        let bitvec = BitVector::from_bools(&bools);
        let bytes = bitvec.to_bytes();
        let restored = BitVector::from_bytes(&bytes).unwrap();
        assert_eq!(bitvec, restored);
    }

    #[test]
    fn test_bitvec_compression_ratio() {
        let bitvec = BitVector::zeros(64);
        let ratio = bitvec.compression_ratio();
        // 64 bools = 64 bytes original, 8 bytes compressed = 8x
        assert!((ratio - 8.0).abs() < 0.1);
    }

    #[test]
    fn test_bitvec_ones_iter() {
        let bools = vec![true, false, true, true, false];
        let bitvec = BitVector::from_bools(&bools);
        let ones: Vec<usize> = bitvec.ones_iter().collect();
        assert_eq!(ones, vec![0, 2, 3]);
    }

    #[test]
    fn test_bitvec_zeros_iter() {
        let bools = vec![true, false, true, true, false];
        let bitvec = BitVector::from_bools(&bools);
        let zeros: Vec<usize> = bitvec.zeros_iter().collect();
        assert_eq!(zeros, vec![1, 4]);
    }

    #[test]
    fn test_bitvec_from_iter() {
        let bitvec: BitVector = vec![true, false, true].into_iter().collect();
        assert_eq!(bitvec.len(), 3);
        assert_eq!(bitvec.get(0), Some(true));
        assert_eq!(bitvec.get(1), Some(false));
        assert_eq!(bitvec.get(2), Some(true));
    }
}
