//! Succinct bitvector with O(1) rank and select operations.
//!
//! Wraps the existing [`BitVector`] with auxiliary structures that enable
//! constant-time rank (count of 1s up to position) and near-constant-time
//! select (position of k-th 1-bit) operations.
//!
//! # Space overhead
//!
//! The auxiliary structures add approximately 3-5% space overhead:
//! - Superblock ranks: 1 u32 per 512 bits = 0.78% overhead
//! - Block ranks: 1 u8 per 64 bits = 1.56% overhead
//! - Select samples: 1 u32 per 4096 1-bits ≈ 0.1% for dense vectors
//!
//! # Performance
//!
//! - `rank1(pos)`: O(1) - superblock lookup + block lookup + popcount
//! - `select1(k)`: O(1) amortized - sampling + binary search within block
//! - Construction: O(n) time, single pass over data

use super::super::BitVector;

/// Number of bits per superblock (512 = 8 words of 64 bits).
const SUPERBLOCK_BITS: usize = 512;

/// Number of bits per block (64 = 1 word).
const BLOCK_BITS: usize = 64;

/// Blocks per superblock.
const BLOCKS_PER_SUPERBLOCK: usize = SUPERBLOCK_BITS / BLOCK_BITS; // 8

/// Sampling interval for select (every 4096 1-bits).
const SELECT_SAMPLE_RATE: usize = 4096;

/// Succinct bitvector with O(1) rank and near-O(1) select support.
///
/// Wraps [`BitVector`] with auxiliary index structures. The original
/// bitvector is preserved and can be accessed via [`inner()`](Self::inner).
///
/// # Example
///
/// ```ignore
/// use grafeo_core::storage::succinct::SuccinctBitVector;
///
/// let bits: Vec<bool> = (0..1000).map(|i| i % 5 == 0).collect();
/// let sbv = SuccinctBitVector::from_bools(&bits);
///
/// // Count 1-bits in [0, 500)
/// assert_eq!(sbv.rank1(500), 100);
///
/// // Find position of 50th 1-bit (0-indexed)
/// assert_eq!(sbv.select1(49), Some(245));
/// ```
#[derive(Debug, Clone)]
pub struct SuccinctBitVector {
    /// Underlying bit storage.
    inner: BitVector,

    /// Cumulative rank at the start of each superblock.
    /// superblock_ranks[i] = number of 1-bits in [0, i * SUPERBLOCK_BITS).
    superblock_ranks: Vec<u32>,

    /// Relative rank within superblock for each block.
    /// block_ranks[i] = number of 1-bits from superblock start to block i start.
    /// Uses u8 since max value is SUPERBLOCK_BITS - BLOCK_BITS = 448.
    block_ranks: Vec<u8>,

    /// Sample positions for select1.
    /// select1_samples[i] = position of (i * SELECT_SAMPLE_RATE)-th 1-bit.
    select1_samples: Vec<u32>,

    /// Sample positions for select0.
    select0_samples: Vec<u32>,

    /// Total number of 1-bits (cached for efficiency).
    ones_count: usize,
}

impl SuccinctBitVector {
    /// Creates a new succinct bitvector from a slice of booleans.
    #[must_use]
    pub fn from_bools(bools: &[bool]) -> Self {
        let inner = BitVector::from_bools(bools);
        Self::from_bitvec(inner)
    }

    /// Creates a new succinct bitvector from an existing [`BitVector`].
    ///
    /// This is the preferred constructor when you already have a `BitVector`.
    #[must_use]
    pub fn from_bitvec(inner: BitVector) -> Self {
        let len = inner.len();
        let num_superblocks = (len + SUPERBLOCK_BITS - 1) / SUPERBLOCK_BITS + 1;
        let num_blocks = (len + BLOCK_BITS - 1) / BLOCK_BITS;

        let mut superblock_ranks = Vec::with_capacity(num_superblocks);
        let mut block_ranks = Vec::with_capacity(num_blocks);
        let mut select1_samples = Vec::new();
        let mut select0_samples = Vec::new();

        let mut cumulative_ones: u32 = 0;
        let mut cumulative_zeros: u32 = 0;
        let mut superblock_start_ones: u32 = 0;

        let data = inner.data();

        for (block_idx, word) in data.iter().enumerate() {
            let bit_pos = block_idx * BLOCK_BITS;

            // Start of new superblock?
            if block_idx % BLOCKS_PER_SUPERBLOCK == 0 {
                superblock_ranks.push(cumulative_ones);
                superblock_start_ones = cumulative_ones;
            }

            // Store relative rank within superblock
            let relative_rank = cumulative_ones - superblock_start_ones;
            block_ranks.push(relative_rank as u8);

            // Count bits in this word
            let bits_in_word = if bit_pos + BLOCK_BITS <= len {
                BLOCK_BITS
            } else {
                len - bit_pos
            };

            let word_ones = if bits_in_word == BLOCK_BITS {
                word.count_ones()
            } else {
                // Mask out bits beyond len
                let mask = (1u64 << bits_in_word) - 1;
                (word & mask).count_ones()
            };

            // Sample for select1
            let next_ones = cumulative_ones + word_ones;
            while select1_samples.len() * SELECT_SAMPLE_RATE < next_ones as usize {
                select1_samples.push(bit_pos as u32);
            }

            // Sample for select0
            let word_zeros = bits_in_word as u32 - word_ones;
            let next_zeros = cumulative_zeros + word_zeros;
            while select0_samples.len() * SELECT_SAMPLE_RATE < next_zeros as usize {
                select0_samples.push(bit_pos as u32);
            }

            cumulative_ones = next_ones;
            cumulative_zeros = next_zeros;
        }

        // Final superblock entry
        if superblock_ranks.len() * SUPERBLOCK_BITS <= len || superblock_ranks.is_empty() {
            superblock_ranks.push(cumulative_ones);
        }

        Self {
            inner,
            superblock_ranks,
            block_ranks,
            select1_samples,
            select0_samples,
            ones_count: cumulative_ones as usize,
        }
    }

    /// Returns the number of bits.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns whether the bitvector is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the total number of 1-bits.
    #[must_use]
    pub fn count_ones(&self) -> usize {
        self.ones_count
    }

    /// Returns the total number of 0-bits.
    #[must_use]
    pub fn count_zeros(&self) -> usize {
        self.len() - self.ones_count
    }

    /// Gets the bit at the given index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<bool> {
        self.inner.get(index)
    }

    /// Returns a reference to the underlying [`BitVector`].
    #[must_use]
    pub fn inner(&self) -> &BitVector {
        &self.inner
    }

    /// Consumes self and returns the underlying [`BitVector`].
    #[must_use]
    pub fn into_inner(self) -> BitVector {
        self.inner
    }

    /// Returns the number of 1-bits in the range [0, pos).
    ///
    /// # Time complexity
    ///
    /// O(1) - uses superblock ranks, block ranks, and popcount.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let sbv = SuccinctBitVector::from_bools(&[true, false, true, true, false]);
    /// assert_eq!(sbv.rank1(0), 0);  // No bits before position 0
    /// assert_eq!(sbv.rank1(1), 1);  // One 1-bit in [0, 1)
    /// assert_eq!(sbv.rank1(4), 3);  // Three 1-bits in [0, 4)
    /// assert_eq!(sbv.rank1(5), 3);  // Three 1-bits in [0, 5)
    /// ```
    #[must_use]
    pub fn rank1(&self, pos: usize) -> usize {
        if pos == 0 {
            return 0;
        }

        // If pos >= len, return total count
        if pos >= self.len() {
            return self.ones_count;
        }

        let superblock_idx = pos / SUPERBLOCK_BITS;
        let block_idx = pos / BLOCK_BITS;
        let bit_offset = pos % BLOCK_BITS;

        // Start with superblock cumulative count
        let mut rank = self.superblock_ranks[superblock_idx] as usize;

        // Add block relative count
        if block_idx < self.block_ranks.len() {
            rank += self.block_ranks[block_idx] as usize;
        }

        // Add popcount within the current word
        if bit_offset > 0 && block_idx < self.inner.data().len() {
            let word = self.inner.data()[block_idx];
            let mask = (1u64 << bit_offset) - 1;
            rank += (word & mask).count_ones() as usize;
        }

        rank
    }

    /// Returns the number of 0-bits in the range [0, pos).
    ///
    /// # Time complexity
    ///
    /// O(1) - computed as pos - rank1(pos).
    #[must_use]
    pub fn rank0(&self, pos: usize) -> usize {
        let pos = pos.min(self.len());
        pos - self.rank1(pos)
    }

    /// Returns the position of the k-th 1-bit (0-indexed).
    ///
    /// Returns `None` if there are fewer than k+1 1-bits.
    ///
    /// # Time complexity
    ///
    /// O(log(n/sample_rate)) amortized - uses sampling and binary search.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let sbv = SuccinctBitVector::from_bools(&[true, false, true, true, false]);
    /// assert_eq!(sbv.select1(0), Some(0));  // First 1-bit at position 0
    /// assert_eq!(sbv.select1(1), Some(2));  // Second 1-bit at position 2
    /// assert_eq!(sbv.select1(2), Some(3));  // Third 1-bit at position 3
    /// assert_eq!(sbv.select1(3), None);     // No fourth 1-bit
    /// ```
    #[must_use]
    pub fn select1(&self, k: usize) -> Option<usize> {
        if k >= self.ones_count {
            return None;
        }

        // Use sampling to find approximate position
        let sample_idx = k / SELECT_SAMPLE_RATE;
        let start_pos = if sample_idx < self.select1_samples.len() {
            self.select1_samples[sample_idx] as usize
        } else {
            0
        };

        // Binary search in superblocks from start_pos
        let start_superblock = start_pos / SUPERBLOCK_BITS;
        let target_rank = k + 1; // We want rank1(result) = k, so find first pos where rank1 >= k+1

        // Find the superblock containing the k-th 1-bit
        let superblock_idx = self.binary_search_superblock(target_rank, start_superblock);

        // Find the block within the superblock
        let block_start = superblock_idx * BLOCKS_PER_SUPERBLOCK;
        let block_end = ((superblock_idx + 1) * BLOCKS_PER_SUPERBLOCK).min(self.block_ranks.len());
        let superblock_base_rank = self.superblock_ranks[superblock_idx] as usize;

        let mut block_idx = block_start;
        for i in block_start..block_end {
            let block_rank = superblock_base_rank + self.block_ranks[i] as usize;
            if block_rank >= target_rank {
                break;
            }
            block_idx = i;
        }

        // Linear scan within the block
        let block_base_rank = superblock_base_rank + self.block_ranks[block_idx] as usize;
        let remaining = k - block_base_rank;

        if block_idx >= self.inner.data().len() {
            return None;
        }

        let word = self.inner.data()[block_idx];
        let bit_pos = Self::select_in_word(word, remaining)?;

        let result = block_idx * BLOCK_BITS + bit_pos;
        if result < self.len() {
            Some(result)
        } else {
            None
        }
    }

    /// Returns the position of the k-th 0-bit (0-indexed).
    ///
    /// Returns `None` if there are fewer than k+1 0-bits.
    #[must_use]
    pub fn select0(&self, k: usize) -> Option<usize> {
        let zeros = self.count_zeros();
        if k >= zeros {
            return None;
        }

        // Use sampling for approximate position
        let sample_idx = k / SELECT_SAMPLE_RATE;
        let start_pos = if sample_idx < self.select0_samples.len() {
            self.select0_samples[sample_idx] as usize
        } else {
            0
        };

        // Binary search from start position
        let mut lo = start_pos;
        let mut hi = self.len();

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.rank0(mid + 1) <= k {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        if lo < self.len() && self.rank0(lo + 1) == k + 1 {
            Some(lo)
        } else {
            None
        }
    }

    /// Binary search for the superblock containing the target rank.
    fn binary_search_superblock(&self, target_rank: usize, start: usize) -> usize {
        let mut lo = start;
        let mut hi = self.superblock_ranks.len();

        while lo + 1 < hi {
            let mid = lo + (hi - lo) / 2;
            if (self.superblock_ranks[mid] as usize) < target_rank {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        lo
    }

    /// Select the k-th 1-bit within a single 64-bit word.
    fn select_in_word(word: u64, k: usize) -> Option<usize> {
        let ones = word.count_ones() as usize;
        if k >= ones {
            return None;
        }

        // Use broadword selection algorithm
        let mut remaining = k;
        let mut pos = 0;

        // Process 8 bits at a time
        for byte_idx in 0..8 {
            let byte = ((word >> (byte_idx * 8)) & 0xFF) as u8;
            let byte_ones = byte.count_ones() as usize;

            if remaining < byte_ones {
                // Target is in this byte
                for bit in 0..8 {
                    if (byte >> bit) & 1 == 1 {
                        if remaining == 0 {
                            return Some(pos + bit);
                        }
                        remaining -= 1;
                    }
                }
            }

            remaining -= byte_ones;
            pos += 8;
        }

        None
    }

    /// Returns the approximate size in bytes of the auxiliary structures.
    #[must_use]
    pub fn auxiliary_size_bytes(&self) -> usize {
        self.superblock_ranks.len() * 4
            + self.block_ranks.len()
            + self.select1_samples.len() * 4
            + self.select0_samples.len() * 4
    }

    /// Returns the total size in bytes (data + auxiliary).
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.inner.data().len() * 8 + self.auxiliary_size_bytes()
    }

    /// Returns the space overhead as a fraction (auxiliary / data).
    #[must_use]
    pub fn space_overhead(&self) -> f64 {
        let data_size = self.inner.data().len() * 8;
        if data_size == 0 {
            return 0.0;
        }
        self.auxiliary_size_bytes() as f64 / data_size as f64
    }
}

impl Default for SuccinctBitVector {
    fn default() -> Self {
        Self::from_bitvec(BitVector::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let sbv = SuccinctBitVector::from_bools(&[]);
        assert!(sbv.is_empty());
        assert_eq!(sbv.rank1(0), 0);
        assert_eq!(sbv.rank0(0), 0);
        assert_eq!(sbv.select1(0), None);
        assert_eq!(sbv.select0(0), None);
    }

    #[test]
    fn test_all_zeros() {
        let sbv = SuccinctBitVector::from_bools(&[false; 100]);
        assert_eq!(sbv.count_ones(), 0);
        assert_eq!(sbv.count_zeros(), 100);
        assert_eq!(sbv.rank1(50), 0);
        assert_eq!(sbv.rank0(50), 50);
        assert_eq!(sbv.select1(0), None);
        assert_eq!(sbv.select0(0), Some(0));
        assert_eq!(sbv.select0(99), Some(99));
    }

    #[test]
    fn test_all_ones() {
        let sbv = SuccinctBitVector::from_bools(&[true; 100]);
        assert_eq!(sbv.count_ones(), 100);
        assert_eq!(sbv.count_zeros(), 0);
        assert_eq!(sbv.rank1(50), 50);
        assert_eq!(sbv.rank0(50), 0);
        assert_eq!(sbv.select1(0), Some(0));
        assert_eq!(sbv.select1(99), Some(99));
        assert_eq!(sbv.select0(0), None);
    }

    #[test]
    fn test_small() {
        let sbv = SuccinctBitVector::from_bools(&[true, false, true, true, false]);

        assert_eq!(sbv.len(), 5);
        assert_eq!(sbv.count_ones(), 3);
        assert_eq!(sbv.count_zeros(), 2);

        // Test rank1
        assert_eq!(sbv.rank1(0), 0);
        assert_eq!(sbv.rank1(1), 1);
        assert_eq!(sbv.rank1(2), 1);
        assert_eq!(sbv.rank1(3), 2);
        assert_eq!(sbv.rank1(4), 3);
        assert_eq!(sbv.rank1(5), 3);

        // Test rank0
        assert_eq!(sbv.rank0(0), 0);
        assert_eq!(sbv.rank0(1), 0);
        assert_eq!(sbv.rank0(2), 1);
        assert_eq!(sbv.rank0(3), 1);
        assert_eq!(sbv.rank0(4), 1);
        assert_eq!(sbv.rank0(5), 2);

        // Test select1
        assert_eq!(sbv.select1(0), Some(0));
        assert_eq!(sbv.select1(1), Some(2));
        assert_eq!(sbv.select1(2), Some(3));
        assert_eq!(sbv.select1(3), None);

        // Test select0
        assert_eq!(sbv.select0(0), Some(1));
        assert_eq!(sbv.select0(1), Some(4));
        assert_eq!(sbv.select0(2), None);
    }

    #[test]
    fn test_rank_select_consistency() {
        // For all k in [0, count_ones), rank1(select1(k)) == k
        let bits: Vec<bool> = (0..1000).map(|i| i % 3 == 0).collect();
        let sbv = SuccinctBitVector::from_bools(&bits);

        let ones_count = sbv.count_ones();
        for k in 0..ones_count {
            let pos = sbv.select1(k).expect("select1 should succeed");
            assert_eq!(sbv.rank1(pos), k, "rank1(select1({})) != {}", k, k);
            assert!(sbv.get(pos) == Some(true), "bit at select1({}) should be 1", k);
        }

        let zeros_count = sbv.count_zeros();
        for k in 0..zeros_count {
            let pos = sbv.select0(k).expect("select0 should succeed");
            assert_eq!(sbv.rank0(pos), k, "rank0(select0({})) != {}", k, k);
            assert!(
                sbv.get(pos) == Some(false),
                "bit at select0({}) should be 0",
                k
            );
        }
    }

    #[test]
    fn test_large() {
        // Test with more than one superblock (512 bits)
        let bits: Vec<bool> = (0..10000).map(|i| i % 7 == 0).collect();
        let sbv = SuccinctBitVector::from_bools(&bits);

        let expected_ones = bits.iter().filter(|&&b| b).count();
        assert_eq!(sbv.count_ones(), expected_ones);

        // Verify some random positions
        for pos in [0, 100, 500, 1000, 5000, 9999] {
            let expected_rank = bits[..pos].iter().filter(|&&b| b).count();
            assert_eq!(sbv.rank1(pos), expected_rank, "rank1({}) mismatch", pos);
        }
    }

    #[test]
    fn test_boundary_conditions() {
        // Test at word boundaries (64 bits)
        let bits: Vec<bool> = (0..128).map(|i| i < 64).collect();
        let sbv = SuccinctBitVector::from_bools(&bits);

        assert_eq!(sbv.rank1(64), 64);
        assert_eq!(sbv.rank1(128), 64);
        assert_eq!(sbv.select1(63), Some(63));
        assert_eq!(sbv.select1(64), None);
    }

    #[test]
    fn test_superblock_boundary() {
        // Test at superblock boundaries (512 bits)
        let bits: Vec<bool> = (0..1024).map(|i| i < 512).collect();
        let sbv = SuccinctBitVector::from_bools(&bits);

        assert_eq!(sbv.rank1(512), 512);
        assert_eq!(sbv.rank1(1024), 512);
    }

    #[test]
    fn test_space_overhead() {
        // Large vector to amortize fixed overhead
        let bits: Vec<bool> = (0..1_000_000).map(|i| i % 2 == 0).collect();
        let sbv = SuccinctBitVector::from_bools(&bits);

        let overhead = sbv.space_overhead();
        // Should be less than 25% (theoretical is ~3.5% but we have select samples)
        // For 1M bits: data = 125KB, auxiliary ≈ superblocks(1.5KB) + blocks(15KB) + select(~500 samples each)
        assert!(
            overhead < 0.25,
            "Space overhead {} is too high (expected < 25%)",
            overhead
        );
    }

    #[test]
    fn test_from_bitvec() {
        let bv = BitVector::from_bools(&[true, false, true]);
        let sbv = SuccinctBitVector::from_bitvec(bv.clone());

        assert_eq!(sbv.inner().to_bools(), bv.to_bools());
        assert_eq!(sbv.count_ones(), 2);
    }

    #[test]
    fn test_into_inner() {
        let original = BitVector::from_bools(&[true, false, true]);
        let sbv = SuccinctBitVector::from_bitvec(original.clone());
        let recovered = sbv.into_inner();

        assert_eq!(recovered.to_bools(), original.to_bools());
    }
}
