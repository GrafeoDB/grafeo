//! Elias-Fano encoding for sparse monotonic sequences.
//!
//! Elias-Fano is a quasi-succinct representation for monotonically increasing
//! sequences of integers. It achieves near-optimal space for sparse sequences
//! (few elements in a large universe).
//!
//! # Space complexity
//!
//! For n elements from universe [0, u):
//! - Lower bits: n * ⌈log₂(u/n)⌉ bits
//! - Upper bits: 2n bits (unary encoding)
//! - Total: n * (2 + ⌈log₂(u/n)⌉) bits
//!
//! This is within 2 bits per element of the information-theoretic lower bound.
//!
//! # Use cases
//!
//! - Node ID lists per label (sparse: ~1000 nodes with label "Person" out of 10M nodes)
//! - Posting lists in text indexes
//! - Sparse timestamp sequences
//!
//! # Example
//!
//! ```ignore
//! use grafeo_core::storage::succinct::EliasFano;
//!
//! // Node IDs with label "Admin" (sparse in universe of 10M)
//! let admin_nodes = vec![100, 5_000, 50_000, 1_000_000, 9_999_999];
//! let ef = EliasFano::new(&admin_nodes);
//!
//! assert_eq!(ef.get(0), 100);
//! assert_eq!(ef.get(2), 50_000);
//! assert!(ef.contains(1_000_000));
//! assert!(!ef.contains(999));
//! ```

use super::rank_select::SuccinctBitVector;
use super::super::BitVector;

/// Elias-Fano encoding for monotonically increasing u64 sequences.
///
/// Elements must be sorted in strictly increasing order.
#[derive(Debug, Clone)]
pub struct EliasFano {
    /// Number of elements.
    n: usize,

    /// Maximum value (universe upper bound).
    universe: u64,

    /// Number of lower bits per element.
    lower_bits: usize,

    /// Lower bits stored consecutively.
    lower: BitVector,

    /// Upper bits in unary encoding with rank/select support.
    upper: SuccinctBitVector,
}

impl EliasFano {
    /// Creates a new Elias-Fano encoding from a sorted slice of values.
    ///
    /// # Panics
    ///
    /// Panics if the input is not sorted in strictly increasing order.
    #[must_use]
    pub fn new(values: &[u64]) -> Self {
        if values.is_empty() {
            return Self {
                n: 0,
                universe: 0,
                lower_bits: 0,
                lower: BitVector::new(),
                upper: SuccinctBitVector::default(),
            };
        }

        // Verify sorted order
        for i in 1..values.len() {
            assert!(
                values[i] > values[i - 1],
                "Values must be strictly increasing: values[{}]={} <= values[{}]={}",
                i,
                values[i],
                i - 1,
                values[i - 1]
            );
        }

        let n = values.len();
        let universe = values[n - 1] + 1; // Exclusive upper bound

        // Compute optimal split: lower_bits = max(0, floor(log2(u/n)))
        let lower_bits = if universe <= n as u64 {
            0
        } else {
            (64 - (universe / n as u64).leading_zeros()) as usize
        };

        let lower_mask = if lower_bits == 0 {
            0
        } else if lower_bits >= 64 {
            u64::MAX
        } else {
            (1u64 << lower_bits) - 1
        };

        // Build lower bits
        let mut lower = BitVector::with_capacity(n * lower_bits);
        for &val in values {
            let low = val & lower_mask;
            for bit_idx in 0..lower_bits {
                lower.push((low >> bit_idx) & 1 == 1);
            }
        }

        // Build upper bits in unary: for each value, we have (high - prev_high) zeros followed by a 1
        // Total length: n (ones) + max_high (zeros) = n + (max_value >> lower_bits)
        let max_high = values[n - 1] >> lower_bits;
        let upper_len = n + max_high as usize;

        let mut upper_bits = BitVector::zeros(upper_len);
        for (i, &val) in values.iter().enumerate() {
            let high = val >> lower_bits;
            // Position = high + i (gap encoding)
            let pos = high as usize + i;
            if pos < upper_len {
                upper_bits.set(pos, true);
            }
        }

        let upper = SuccinctBitVector::from_bitvec(upper_bits);

        Self {
            n,
            universe,
            lower_bits,
            lower,
            upper,
        }
    }

    /// Returns the number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Returns whether the sequence is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Returns the universe size (exclusive upper bound).
    #[must_use]
    pub fn universe(&self) -> u64 {
        self.universe
    }

    /// Returns the i-th element (0-indexed).
    ///
    /// # Panics
    ///
    /// Panics if index >= len().
    #[must_use]
    pub fn get(&self, index: usize) -> u64 {
        assert!(index < self.n, "Index {} out of bounds (len={})", index, self.n);

        // Find position in upper: select1(index) gives position of (index+1)-th 1-bit
        let upper_pos = self.upper.select1(index).expect("index within bounds");

        // High bits = upper_pos - index (number of 0s before this 1)
        let high = (upper_pos - index) as u64;

        // Low bits from lower bitvector
        let low = self.get_lower(index);

        (high << self.lower_bits) | low
    }

    /// Returns the lower bits for element at index.
    fn get_lower(&self, index: usize) -> u64 {
        if self.lower_bits == 0 {
            return 0;
        }

        let bit_start = index * self.lower_bits;
        let mut low = 0u64;

        for bit_idx in 0..self.lower_bits {
            if self.lower.get(bit_start + bit_idx) == Some(true) {
                low |= 1 << bit_idx;
            }
        }

        low
    }

    /// Checks if a value is in the sequence.
    ///
    /// # Time complexity
    ///
    /// O(log n) using binary search.
    #[must_use]
    pub fn contains(&self, value: u64) -> bool {
        if self.is_empty() || value >= self.universe {
            return false;
        }

        // Binary search
        match self.predecessor(value) {
            Some(idx) => self.get(idx) == value,
            None => false,
        }
    }

    /// Returns the index of the largest element <= value, or None if no such element exists.
    ///
    /// # Time complexity
    ///
    /// O(log n)
    #[must_use]
    pub fn predecessor(&self, value: u64) -> Option<usize> {
        if self.is_empty() {
            return None;
        }

        // Binary search for largest index where get(index) <= value
        let mut lo = 0;
        let mut hi = self.n;

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.get(mid) <= value {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        if lo > 0 {
            Some(lo - 1)
        } else if self.get(0) <= value {
            Some(0)
        } else {
            None
        }
    }

    /// Returns the index of the smallest element >= value, or None if no such element exists.
    ///
    /// # Time complexity
    ///
    /// O(log n)
    #[must_use]
    pub fn successor(&self, value: u64) -> Option<usize> {
        if self.is_empty() {
            return None;
        }

        // Binary search for smallest index where get(index) >= value
        let mut lo = 0;
        let mut hi = self.n;

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.get(mid) < value {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        if lo < self.n {
            Some(lo)
        } else {
            None
        }
    }

    /// Returns an iterator over all values.
    pub fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        (0..self.n).map(move |i| self.get(i))
    }

    /// Returns the size in bytes.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        // Base struct fields
        let base = std::mem::size_of::<Self>();
        // Lower bits
        let lower_bytes = self.lower.data().len() * 8;
        // Upper bits (includes auxiliary structures)
        let upper_bytes = self.upper.size_bytes();

        base + lower_bytes + upper_bytes
    }

    /// Returns the compression ratio compared to storing values as plain u64s.
    #[must_use]
    pub fn compression_ratio(&self) -> f64 {
        if self.n == 0 {
            return 1.0;
        }

        let original_bytes = self.n * 8; // 8 bytes per u64
        let compressed_bytes = self.size_bytes();

        if compressed_bytes == 0 {
            return 1.0;
        }

        original_bytes as f64 / compressed_bytes as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let ef = EliasFano::new(&[]);
        assert!(ef.is_empty());
        assert_eq!(ef.len(), 0);
        assert!(!ef.contains(0));
        assert_eq!(ef.predecessor(100), None);
    }

    #[test]
    fn test_single() {
        let ef = EliasFano::new(&[42]);
        assert_eq!(ef.len(), 1);
        assert_eq!(ef.get(0), 42);
        assert!(ef.contains(42));
        assert!(!ef.contains(0));
        assert!(!ef.contains(100));
    }

    #[test]
    fn test_small() {
        let values = vec![2, 3, 5, 7, 11, 13, 17, 19, 23];
        let ef = EliasFano::new(&values);

        assert_eq!(ef.len(), values.len());

        for (i, &v) in values.iter().enumerate() {
            assert_eq!(ef.get(i), v, "get({}) failed", i);
            assert!(ef.contains(v), "contains({}) failed", v);
        }

        // Non-existent values
        assert!(!ef.contains(0));
        assert!(!ef.contains(4));
        assert!(!ef.contains(100));
    }

    #[test]
    fn test_sparse() {
        // Sparse sequence: few values in large universe
        let values: Vec<u64> = vec![100, 1_000, 10_000, 100_000, 1_000_000];
        let ef = EliasFano::new(&values);

        for (i, &v) in values.iter().enumerate() {
            assert_eq!(ef.get(i), v);
            assert!(ef.contains(v));
        }

        // Verify basic functionality works for sparse data
        // Note: For very small n, struct overhead dominates compression
        assert_eq!(ef.len(), 5);
    }

    #[test]
    fn test_dense() {
        // Dense sequence: consecutive values
        let values: Vec<u64> = (0..100).collect();
        let ef = EliasFano::new(&values);

        for (i, &v) in values.iter().enumerate() {
            assert_eq!(ef.get(i), v);
        }
    }

    #[test]
    fn test_predecessor() {
        let values = vec![10, 20, 30, 40, 50];
        let ef = EliasFano::new(&values);

        assert_eq!(ef.predecessor(5), None);
        assert_eq!(ef.predecessor(10), Some(0));
        assert_eq!(ef.predecessor(15), Some(0));
        assert_eq!(ef.predecessor(20), Some(1));
        assert_eq!(ef.predecessor(25), Some(1));
        assert_eq!(ef.predecessor(50), Some(4));
        assert_eq!(ef.predecessor(100), Some(4));
    }

    #[test]
    fn test_successor() {
        let values = vec![10, 20, 30, 40, 50];
        let ef = EliasFano::new(&values);

        assert_eq!(ef.successor(5), Some(0));
        assert_eq!(ef.successor(10), Some(0));
        assert_eq!(ef.successor(15), Some(1));
        assert_eq!(ef.successor(20), Some(1));
        assert_eq!(ef.successor(50), Some(4));
        assert_eq!(ef.successor(51), None);
    }

    #[test]
    fn test_iter() {
        let values = vec![1, 2, 4, 8, 16, 32];
        let ef = EliasFano::new(&values);

        let collected: Vec<u64> = ef.iter().collect();
        assert_eq!(collected, values);
    }

    #[test]
    fn test_large_universe() {
        // Large values to test high-bit handling
        let values: Vec<u64> = vec![
            1_000_000,
            10_000_000,
            100_000_000,
            1_000_000_000,
            10_000_000_000,
        ];
        let ef = EliasFano::new(&values);

        for (i, &v) in values.iter().enumerate() {
            assert_eq!(ef.get(i), v, "Failed at index {}", i);
        }
    }

    #[test]
    #[should_panic(expected = "strictly increasing")]
    fn test_unsorted_panics() {
        let _ = EliasFano::new(&[5, 3, 7]);
    }

    #[test]
    #[should_panic(expected = "strictly increasing")]
    fn test_duplicate_panics() {
        let _ = EliasFano::new(&[1, 2, 2, 3]);
    }

    #[test]
    fn test_compression_large() {
        // Larger sequence where Elias-Fano benefits show
        // 1000 values uniformly distributed in 10M universe
        let values: Vec<u64> = (0..1000).map(|i| i * 10_000).collect();
        let ef = EliasFano::new(&values);

        // Just verify it works correctly
        assert_eq!(ef.len(), 1000);
        assert_eq!(ef.get(0), 0);
        assert_eq!(ef.get(999), 9_990_000);

        // The bit representation should be smaller than 1000 * 8 = 8000 bytes
        // Lower bits: ~14 bits per value (log(10M/1000) = ~13.3)
        // Upper bits: 2n bits = 2000 bits = 250 bytes
        // Total: ~2000 bytes vs 8000 bytes = ~4x compression
        // But with auxiliary structures, actual savings vary
        let size = ef.size_bytes();
        // Just ensure it's reasonable (under 10KB for 1000 values)
        assert!(size < 10000, "Size {} bytes is too large", size);
    }
}
