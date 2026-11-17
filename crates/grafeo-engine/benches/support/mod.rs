//! What more than one bench needs: the seeded generator their data comes from.

/// The house generator: a seeded xorshift, the same sequence on every platform and run, with no
/// `rand` dependency.
pub struct Xorshift(u64);

impl Xorshift {
    /// A generator from `seed`, which must not be 0 (xorshift stays at 0).
    pub const fn new(seed: u64) -> Self {
        assert!(seed != 0, "a xorshift seed is not 0");
        Self(seed)
    }

    /// The next value of the sequence.
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// The next value below `bound` (above 0).
    pub fn below(&mut self, bound: usize) -> usize {
        usize::try_from(self.next_u64() % bound as u64).expect("below the bound")
    }
}
