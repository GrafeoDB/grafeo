//! Succinct permutations for Ring Index.
//!
//! A permutation represents a mapping π: [0,n) → [0,n).
//!
//! In the Ring Index, permutations map between triple orderings:
//! - SPO → POS: Given position in SPO order, find position in POS order
//! - SPO → OSP: Given position in SPO order, find position in OSP order
//!
//! This allows navigating all three orderings with a single copy of the data.

/// A representation of a permutation.
///
/// Stores both forward and inverse mappings for O(1) access.
/// Uses 2n * 32 bits = 8n bytes for n elements.
#[derive(Debug, Clone)]
pub struct SuccinctPermutation {
    /// The number of elements in the permutation.
    n: usize,

    /// Forward mapping: forward[i] = π(i)
    forward: Vec<u32>,

    /// Inverse mapping: inverse[j] = π⁻¹(j)
    inverse: Vec<u32>,
}

impl SuccinctPermutation {
    /// Creates a permutation from an array where `perm[i] = j` means
    /// position i maps to position j.
    ///
    /// # Arguments
    ///
    /// * `permutation` - Array where `permutation[i]` gives the target of position i
    #[must_use]
    pub fn new(permutation: &[usize]) -> Self {
        let n = permutation.len();

        if n == 0 {
            return Self {
                n: 0,
                forward: Vec::new(),
                inverse: Vec::new(),
            };
        }

        // Build forward mapping
        let forward: Vec<u32> = permutation.iter().map(|&x| x as u32).collect();

        // Build inverse mapping
        let mut inverse = vec![0u32; n];
        for (i, &target) in permutation.iter().enumerate() {
            inverse[target] = i as u32;
        }

        Self {
            n,
            forward,
            inverse,
        }
    }

    /// Returns the number of elements in the permutation.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Returns whether the permutation is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Applies the permutation: returns π(index), the target of position index.
    ///
    /// # Time complexity
    ///
    /// O(1)
    #[must_use]
    pub fn apply(&self, index: usize) -> Option<usize> {
        if index >= self.n {
            return None;
        }
        Some(self.forward[index] as usize)
    }

    /// Applies the inverse permutation: returns π⁻¹(target), the position that
    /// maps to target.
    ///
    /// # Time complexity
    ///
    /// O(1)
    #[must_use]
    pub fn apply_inverse(&self, target: usize) -> Option<usize> {
        if target >= self.n {
            return None;
        }
        Some(self.inverse[target] as usize)
    }

    /// Returns the size in bytes.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        let base = std::mem::size_of::<Self>();
        let forward_bytes = self.forward.capacity() * std::mem::size_of::<u32>();
        let inverse_bytes = self.inverse.capacity() * std::mem::size_of::<u32>();
        base + forward_bytes + inverse_bytes
    }
}

/// Identity permutation (for when no reordering is needed).
impl Default for SuccinctPermutation {
    fn default() -> Self {
        Self {
            n: 0,
            forward: Vec::new(),
            inverse: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let perm = SuccinctPermutation::new(&[]);
        assert!(perm.is_empty());
        assert_eq!(perm.apply(0), None);
        assert_eq!(perm.apply_inverse(0), None);
    }

    #[test]
    fn test_single() {
        let perm = SuccinctPermutation::new(&[0]);
        assert_eq!(perm.len(), 1);
        assert_eq!(perm.apply(0), Some(0));
        assert_eq!(perm.apply_inverse(0), Some(0));
    }

    #[test]
    fn test_identity() {
        let perm = SuccinctPermutation::new(&[0, 1, 2, 3, 4]);
        for i in 0..5 {
            assert_eq!(perm.apply(i), Some(i));
            assert_eq!(perm.apply_inverse(i), Some(i));
        }
    }

    #[test]
    fn test_reverse() {
        // Permutation that reverses: 0→4, 1→3, 2→2, 3→1, 4→0
        let perm = SuccinctPermutation::new(&[4, 3, 2, 1, 0]);

        assert_eq!(perm.apply(0), Some(4));
        assert_eq!(perm.apply(1), Some(3));
        assert_eq!(perm.apply(2), Some(2));
        assert_eq!(perm.apply(3), Some(1));
        assert_eq!(perm.apply(4), Some(0));

        // Inverse
        assert_eq!(perm.apply_inverse(4), Some(0));
        assert_eq!(perm.apply_inverse(3), Some(1));
        assert_eq!(perm.apply_inverse(2), Some(2));
        assert_eq!(perm.apply_inverse(1), Some(3));
        assert_eq!(perm.apply_inverse(0), Some(4));
    }

    #[test]
    fn test_cyclic() {
        // Cyclic permutation: 0→1, 1→2, 2→3, 3→0
        let perm = SuccinctPermutation::new(&[1, 2, 3, 0]);

        assert_eq!(perm.apply(0), Some(1));
        assert_eq!(perm.apply(1), Some(2));
        assert_eq!(perm.apply(2), Some(3));
        assert_eq!(perm.apply(3), Some(0));

        // Inverse: where does target i come from?
        assert_eq!(perm.apply_inverse(1), Some(0));
        assert_eq!(perm.apply_inverse(2), Some(1));
        assert_eq!(perm.apply_inverse(3), Some(2));
        assert_eq!(perm.apply_inverse(0), Some(3));
    }

    #[test]
    fn test_random_permutation() {
        // A more complex permutation
        let perm_array = [3, 0, 5, 2, 7, 1, 4, 6];
        let perm = SuccinctPermutation::new(&perm_array);

        // Verify apply
        for (i, &expected) in perm_array.iter().enumerate() {
            assert_eq!(perm.apply(i), Some(expected), "apply({}) failed", i);
        }

        // Verify inverse: for each target, find which position maps to it
        for target in 0..8 {
            let source = perm.apply_inverse(target);
            assert!(source.is_some(), "apply_inverse({}) failed", target);
            assert_eq!(
                perm.apply(source.unwrap()),
                Some(target),
                "roundtrip failed for target {}",
                target
            );
        }
    }

    #[test]
    fn test_large_permutation() {
        // Test with a larger permutation
        let n = 1000;
        let mut perm_array: Vec<usize> = (0..n).collect();

        // Shuffle deterministically (reverse in chunks)
        for chunk in perm_array.chunks_mut(10) {
            chunk.reverse();
        }

        let perm = SuccinctPermutation::new(&perm_array);

        // Verify all mappings
        for (i, &expected) in perm_array.iter().enumerate() {
            assert_eq!(perm.apply(i), Some(expected));
        }

        // Verify inverse for a sample
        for target in (0..n).step_by(37) {
            let source = perm.apply_inverse(target).unwrap();
            assert_eq!(perm.apply(source), Some(target));
        }
    }

    #[test]
    fn test_out_of_bounds() {
        let perm = SuccinctPermutation::new(&[2, 0, 1]);
        assert_eq!(perm.apply(3), None);
        assert_eq!(perm.apply(100), None);
        assert_eq!(perm.apply_inverse(3), None);
        assert_eq!(perm.apply_inverse(100), None);
    }

    #[test]
    fn test_size_bytes() {
        let perm = SuccinctPermutation::new(&[0, 1, 2, 3, 4, 5, 6, 7]);
        let size = perm.size_bytes();
        // Should be reasonable (not huge)
        assert!(size < 1000);
    }
}
