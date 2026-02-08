//! Wavelet tree for sequence indexing with rank/select/access.
//!
//! A wavelet tree is a space-efficient data structure for sequences over
//! a finite alphabet, supporting:
//!
//! - `access(i)`: Symbol at position i in O(log σ)
//! - `rank(c, i)`: Count of symbol c in [0, i) in O(log σ)
//! - `select(c, k)`: Position of k-th occurrence of c in O(log σ)
//!
//! where σ is the alphabet size.
//!
//! # Space complexity
//!
//! n log σ + o(n log σ) bits, where n is sequence length and σ is alphabet size.
//!
//! # Use cases
//!
//! - Ring Index for RDF triples (Phase 17)
//! - Compressed suffix arrays
//! - Burrows-Wheeler Transform indexing
//!
//! # Example
//!
//! ```ignore
//! use grafeo_core::storage::succinct::WaveletTree;
//!
//! // Index a sequence of predicate IDs
//! let predicates = vec![0, 1, 0, 2, 1, 0, 2, 2];
//! let wt = WaveletTree::new(&predicates);
//!
//! assert_eq!(wt.access(3), 2);
//! assert_eq!(wt.rank(0, 6), 3);  // Three 0s in [0, 6)
//! assert_eq!(wt.select(1, 1), Some(4));  // Second 1 at position 4
//! ```

use super::super::BitVector;
use super::rank_select::SuccinctBitVector;

/// Wavelet tree for sequence rank/select/access operations.
///
/// Supports sequences of u64 symbols. Internally builds a binary tree
/// of bitvectors, one level per bit of the alphabet encoding.
#[derive(Debug, Clone)]
pub struct WaveletTree {
    /// Bitvectors at each level of the tree.
    /// Level 0 is the root, level h-1 is the leaves.
    levels: Vec<SuccinctBitVector>,

    /// Number of bits needed to represent the alphabet (ceil(log2(sigma))).
    height: usize,

    /// Alphabet size (number of distinct symbols + 1 for 0).
    sigma: u64,

    /// Length of the original sequence.
    len: usize,

    /// Sorted unique symbols (for mapping back).
    symbols: Vec<u64>,

    /// Symbol to code mapping.
    symbol_to_code: hashbrown::HashMap<u64, u64>,
}

impl WaveletTree {
    /// Creates a new wavelet tree from a sequence of symbols.
    ///
    /// The symbols are internally remapped to a compact alphabet [0, sigma).
    #[must_use]
    pub fn new(sequence: &[u64]) -> Self {
        if sequence.is_empty() {
            return Self {
                levels: Vec::new(),
                height: 0,
                sigma: 0,
                len: 0,
                symbols: Vec::new(),
                symbol_to_code: hashbrown::HashMap::default(),
            };
        }

        // Build symbol mapping
        let mut symbols: Vec<u64> = sequence.to_vec();
        symbols.sort_unstable();
        symbols.dedup();

        let sigma = symbols.len() as u64;
        let height = if sigma <= 1 {
            1
        } else {
            64 - (sigma - 1).leading_zeros() as usize
        };

        let mut symbol_to_code = hashbrown::HashMap::with_capacity(symbols.len());
        for (code, &sym) in symbols.iter().enumerate() {
            symbol_to_code.insert(sym, code as u64);
        }

        // Remap sequence to codes
        let codes: Vec<u64> = sequence
            .iter()
            .map(|&s| *symbol_to_code.get(&s).unwrap())
            .collect();

        // Build wavelet tree levels
        let levels = Self::build_levels(&codes, height);

        Self {
            levels,
            height,
            sigma,
            len: sequence.len(),
            symbols,
            symbol_to_code,
        }
    }

    /// Build the bitvector levels of the wavelet tree.
    fn build_levels(codes: &[u64], height: usize) -> Vec<SuccinctBitVector> {
        if codes.is_empty() || height == 0 {
            return Vec::new();
        }

        let mut levels = Vec::with_capacity(height);
        let mut current_sequence: Vec<(u64, usize)> = codes
            .iter()
            .copied()
            .enumerate()
            .map(|(i, c)| (c, i))
            .collect();

        for level in 0..height {
            let bit_pos = height - 1 - level;
            let mut bits = BitVector::with_capacity(current_sequence.len());

            // Build bitvector for this level
            for &(code, _) in &current_sequence {
                let bit = (code >> bit_pos) & 1;
                bits.push(bit == 1);
            }

            levels.push(SuccinctBitVector::from_bitvec(bits));

            // Partition for next level: 0-bits go left, 1-bits go right
            let mut left = Vec::new();
            let mut right = Vec::new();

            for &(code, orig_idx) in &current_sequence {
                let bit = (code >> bit_pos) & 1;
                if bit == 0 {
                    left.push((code, orig_idx));
                } else {
                    right.push((code, orig_idx));
                }
            }

            // Concatenate for next level (left then right)
            current_sequence = left;
            current_sequence.extend(right);
        }

        levels
    }

    /// Returns the length of the sequence.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the sequence is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the alphabet size.
    #[must_use]
    pub fn sigma(&self) -> u64 {
        self.sigma
    }

    /// Returns the symbol at position i.
    ///
    /// # Time complexity
    ///
    /// O(log σ)
    ///
    /// # Panics
    ///
    /// Panics if i >= len().
    #[must_use]
    pub fn access(&self, i: usize) -> u64 {
        assert!(i < self.len, "Index {} out of bounds (len={})", i, self.len);

        let mut pos = i;
        let mut code = 0u64;

        for level in 0..self.height {
            let bv = &self.levels[level];
            let bit = bv.get(pos).unwrap_or(false);
            let bit_pos = self.height - 1 - level;

            if bit {
                code |= 1 << bit_pos;
                // Move to right subtree: position becomes rank1 + offset
                let zeros_total = bv.count_zeros();
                pos = zeros_total + bv.rank1(pos);
            } else {
                // Move to left subtree: position becomes rank0
                pos = bv.rank0(pos);
            }
        }

        // Map code back to original symbol
        self.symbols.get(code as usize).copied().unwrap_or(0)
    }

    /// Returns the number of occurrences of symbol in [0, i).
    ///
    /// # Time complexity
    ///
    /// O(log σ)
    #[must_use]
    pub fn rank(&self, symbol: u64, i: usize) -> usize {
        if i == 0 || self.is_empty() {
            return 0;
        }

        let Some(&code) = self.symbol_to_code.get(&symbol) else {
            return 0; // Symbol not in alphabet
        };

        let i = i.min(self.len);
        let mut lo = 0;
        let mut hi = i;

        for level in 0..self.height {
            let bv = &self.levels[level];
            let bit_pos = self.height - 1 - level;
            let bit = (code >> bit_pos) & 1;

            if bit == 0 {
                // Follow left (0-bits)
                lo = bv.rank0(lo);
                hi = bv.rank0(hi);
            } else {
                // Follow right (1-bits)
                let zeros_total = bv.count_zeros();
                lo = zeros_total + bv.rank1(lo);
                hi = zeros_total + bv.rank1(hi);
            }
        }

        hi - lo
    }

    /// Returns the position of the k-th occurrence of symbol (0-indexed).
    ///
    /// Returns `None` if there are fewer than k+1 occurrences.
    ///
    /// # Time complexity
    ///
    /// O(log σ)
    #[must_use]
    pub fn select(&self, symbol: u64, k: usize) -> Option<usize> {
        if self.is_empty() {
            return None;
        }

        let Some(&code) = self.symbol_to_code.get(&symbol) else {
            return None;
        };

        // Find the range for this symbol at the deepest level
        let mut lo = 0usize;
        let mut hi = self.len;

        // Navigate down to find the range
        for level in 0..self.height {
            let bv = &self.levels[level];
            let bit_pos = self.height - 1 - level;
            let bit = (code >> bit_pos) & 1;

            if bit == 0 {
                lo = bv.rank0(lo);
                hi = bv.rank0(hi);
            } else {
                let zeros_total = bv.count_zeros();
                lo = zeros_total + bv.rank1(lo);
                hi = zeros_total + bv.rank1(hi);
            }
        }

        // Check if k-th occurrence exists
        if k >= hi - lo {
            return None;
        }

        // Navigate back up using select
        let mut pos = lo + k;

        for level in (0..self.height).rev() {
            let bv = &self.levels[level];
            let bit_pos = self.height - 1 - level;
            let bit = (code >> bit_pos) & 1;

            if bit == 0 {
                // We need to find position where rank0 gives us 'pos'
                pos = bv.select0(pos)?;
            } else {
                // Adjust for the offset in right subtree
                let zeros_total = bv.count_zeros();
                let rank_in_right = pos - zeros_total;
                pos = bv.select1(rank_in_right)?;
            }
        }

        Some(pos)
    }

    /// Returns the count of symbol in the entire sequence.
    #[must_use]
    pub fn count(&self, symbol: u64) -> usize {
        self.rank(symbol, self.len)
    }

    /// Returns an iterator over all distinct symbols in the sequence.
    pub fn alphabet(&self) -> impl Iterator<Item = u64> + '_ {
        self.symbols.iter().copied()
    }

    /// Returns the size in bytes.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        let base = std::mem::size_of::<Self>();
        let levels_bytes: usize = self.levels.iter().map(|bv| bv.size_bytes()).sum();
        let symbols_bytes = self.symbols.len() * 8;
        let map_bytes = self.symbol_to_code.len() * 16; // Approximate

        base + levels_bytes + symbols_bytes + map_bytes
    }

    /// Returns an iterator over all (position, symbol) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (usize, u64)> + '_ {
        (0..self.len).map(move |i| (i, self.access(i)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let wt = WaveletTree::new(&[]);
        assert!(wt.is_empty());
        assert_eq!(wt.len(), 0);
        assert_eq!(wt.sigma(), 0);
        assert_eq!(wt.rank(0, 0), 0);
        assert_eq!(wt.select(0, 0), None);
    }

    #[test]
    fn test_single() {
        let wt = WaveletTree::new(&[42]);
        assert_eq!(wt.len(), 1);
        assert_eq!(wt.access(0), 42);
        assert_eq!(wt.rank(42, 1), 1);
        assert_eq!(wt.select(42, 0), Some(0));
    }

    #[test]
    fn test_small() {
        let seq = vec![0, 1, 0, 2, 1, 0, 2, 2];
        let wt = WaveletTree::new(&seq);

        // Test access
        for (i, &expected) in seq.iter().enumerate() {
            assert_eq!(wt.access(i), expected, "access({}) failed", i);
        }

        // Test rank
        assert_eq!(wt.rank(0, 0), 0);
        assert_eq!(wt.rank(0, 1), 1);
        assert_eq!(wt.rank(0, 3), 2);
        assert_eq!(wt.rank(0, 8), 3);

        assert_eq!(wt.rank(1, 2), 1);
        assert_eq!(wt.rank(1, 5), 2);
        assert_eq!(wt.rank(1, 8), 2);

        assert_eq!(wt.rank(2, 4), 1);
        assert_eq!(wt.rank(2, 8), 3);

        // Test select
        assert_eq!(wt.select(0, 0), Some(0));
        assert_eq!(wt.select(0, 1), Some(2));
        assert_eq!(wt.select(0, 2), Some(5));
        assert_eq!(wt.select(0, 3), None);

        assert_eq!(wt.select(1, 0), Some(1));
        assert_eq!(wt.select(1, 1), Some(4));
        assert_eq!(wt.select(1, 2), None);

        assert_eq!(wt.select(2, 0), Some(3));
        assert_eq!(wt.select(2, 1), Some(6));
        assert_eq!(wt.select(2, 2), Some(7));
    }

    #[test]
    fn test_rank_select_consistency() {
        let seq: Vec<u64> = (0..1000).map(|i| (i % 10) as u64).collect();
        let wt = WaveletTree::new(&seq);

        // For each symbol, verify rank/select consistency
        for sym in 0..10u64 {
            let count = wt.count(sym);
            for k in 0..count {
                let pos = wt.select(sym, k).expect("select should succeed");
                assert_eq!(
                    wt.rank(sym, pos),
                    k,
                    "rank(select({})) mismatch for symbol {}",
                    k,
                    sym
                );
                assert_eq!(wt.access(pos), sym, "access mismatch at position {}", pos);
            }
        }
    }

    #[test]
    fn test_access_all() {
        let seq: Vec<u64> = vec![5, 3, 8, 1, 3, 5, 1, 8, 3];
        let wt = WaveletTree::new(&seq);

        for (i, &expected) in seq.iter().enumerate() {
            assert_eq!(wt.access(i), expected, "access({}) failed", i);
        }
    }

    #[test]
    fn test_large_alphabet() {
        // Test with larger alphabet
        let seq: Vec<u64> = (0..100).map(|i| i * 7 % 50).collect();
        let wt = WaveletTree::new(&seq);

        assert_eq!(wt.len(), 100);

        for (i, &expected) in seq.iter().enumerate() {
            assert_eq!(wt.access(i), expected, "access({}) failed", i);
        }
    }

    #[test]
    fn test_count() {
        let seq = vec![0, 1, 0, 2, 1, 0, 2, 2];
        let wt = WaveletTree::new(&seq);

        assert_eq!(wt.count(0), 3);
        assert_eq!(wt.count(1), 2);
        assert_eq!(wt.count(2), 3);
        assert_eq!(wt.count(99), 0); // Non-existent symbol
    }

    #[test]
    fn test_nonexistent_symbol() {
        let wt = WaveletTree::new(&[1, 2, 3]);

        assert_eq!(wt.rank(99, 3), 0);
        assert_eq!(wt.select(99, 0), None);
        assert_eq!(wt.count(99), 0);
    }

    #[test]
    fn test_alphabet() {
        let seq = vec![5, 3, 8, 1];
        let wt = WaveletTree::new(&seq);

        let mut alpha: Vec<u64> = wt.alphabet().collect();
        alpha.sort();
        assert_eq!(alpha, vec![1, 3, 5, 8]);
    }

    #[test]
    fn test_iter() {
        let seq = vec![2, 0, 1];
        let wt = WaveletTree::new(&seq);

        let collected: Vec<(usize, u64)> = wt.iter().collect();
        assert_eq!(collected, vec![(0, 2), (1, 0), (2, 1)]);
    }

    #[test]
    fn test_single_symbol_repeated() {
        // All same symbol
        let seq = vec![7, 7, 7, 7, 7];
        let wt = WaveletTree::new(&seq);

        assert_eq!(wt.sigma(), 1);
        for i in 0..5 {
            assert_eq!(wt.access(i), 7);
        }
        assert_eq!(wt.rank(7, 3), 3);
        assert_eq!(wt.select(7, 2), Some(2));
    }

    #[test]
    fn test_large_values() {
        let seq: Vec<u64> = vec![1_000_000, 5_000_000, 1_000_000, 10_000_000];
        let wt = WaveletTree::new(&seq);

        for (i, &expected) in seq.iter().enumerate() {
            assert_eq!(wt.access(i), expected, "access({}) failed", i);
        }

        assert_eq!(wt.count(1_000_000), 2);
        assert_eq!(wt.rank(1_000_000, 3), 2);
    }
}
