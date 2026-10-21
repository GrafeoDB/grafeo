//! A growable set of rows, one bit per row of a row group.

/// The rows of a row group that have something: a label, a value, a true
/// boolean. Grows to the highest row set; rows past its end are unset.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct Bitset {
    words: Vec<u64>,
}

impl Bitset {
    /// Whether `row` is set.
    pub(super) fn get(&self, row: usize) -> bool {
        self.words
            .get(row / 64)
            .is_some_and(|word| word & (1 << (row % 64)) != 0)
    }

    /// Sets `row`.
    pub(super) fn set(&mut self, row: usize) {
        let word = row / 64;
        if word >= self.words.len() {
            self.words.resize(word + 1, 0);
        }
        self.words[word] |= 1 << (row % 64);
    }

    /// Unsets `row`.
    pub(super) fn clear(&mut self, row: usize) {
        if let Some(word) = self.words.get_mut(row / 64) {
            *word &= !(1 << (row % 64));
        }
    }

    /// Sets the rows of `bits` in word `index`, rows `[64 * index, 64 *
    /// (index + 1))`: what [`set`](Self::set) of each does, the bitset
    /// growing alike (not at all for no rows).
    pub(super) fn set_word(&mut self, index: usize, bits: u64) {
        if bits == 0 {
            return;
        }
        if index >= self.words.len() {
            self.words.resize(index + 1, 0);
        }
        self.words[index] |= bits;
    }

    /// Sets `row` to `value`.
    pub(super) fn put(&mut self, row: usize, value: bool) {
        if value {
            self.set(row);
        } else {
            self.clear(row);
        }
    }

    /// Whether no row is set.
    pub(super) fn is_empty(&self) -> bool {
        self.words.iter().all(|word| *word == 0)
    }

    /// The word of rows `[64 * index, 64 * (index + 1))`, one bit per row.
    pub(super) fn word(&self, index: usize) -> u64 {
        self.words.get(index).copied().unwrap_or(0)
    }

    /// The words, rows `[64 * i, 64 * (i + 1))` in word `i`.
    pub(super) fn words(&self) -> &[u64] {
        &self.words
    }

    /// How many rows of `rows` are set.
    pub(super) fn count_in(&self, rows: std::ops::Range<usize>) -> usize {
        if rows.is_empty() {
            return 0;
        }
        let (first, last) = (rows.start / 64, (rows.end - 1) / 64);
        let mut count = 0;
        for (index, word) in self.words.iter().enumerate().take(last + 1).skip(first) {
            let mut bits = *word;
            if index == first {
                bits &= u64::MAX << (rows.start % 64);
            }
            if index == last {
                bits &= u64::MAX >> (63 - (rows.end - 1) % 64);
            }
            count += bits.count_ones() as usize;
        }
        count
    }

    /// The rows set, ascending.
    pub(super) fn rows(&self) -> impl Iterator<Item = usize> + '_ {
        self.words.iter().enumerate().flat_map(|(index, word)| {
            let mut bits = *word;
            std::iter::from_fn(move || {
                if bits == 0 {
                    return None;
                }
                let bit = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                Some(index * 64 + bit)
            })
        })
    }

    /// The heap bytes it holds.
    pub(super) fn heap_bytes(&self) -> usize {
        self.words.capacity() * std::mem::size_of::<u64>()
    }
}

#[cfg(test)]
mod tests {
    use super::Bitset;

    #[test]
    fn rows_are_set_cleared_and_listed_in_order() {
        let mut bits = Bitset::default();
        assert!(bits.is_empty());
        for row in [65_535, 0, 63, 64, 130] {
            bits.set(row);
        }
        assert!(bits.get(64) && bits.get(65_535) && !bits.get(65));
        bits.clear(63);
        bits.clear(1_000_000);
        assert_eq!(bits.rows().collect::<Vec<_>>(), [0, 64, 130, 65_535]);
        bits.put(130, false);
        bits.put(3, true);
        assert_eq!(bits.rows().collect::<Vec<_>>(), [0, 3, 64, 65_535]);
    }

    #[test]
    fn rows_in_a_range_are_counted_across_words() {
        let mut bits = Bitset::default();
        for row in [0, 3, 63, 64, 127, 128, 200] {
            bits.set(row);
        }
        for (rows, expected) in [
            (0..0, 0),
            (0..1, 1),
            (1..3, 0),
            (3..4, 1),
            (0..64, 3),
            (63..65, 2),
            (64..128, 2),
            (0..201, 7),
            (129..200, 0),
            (200..100_000, 1),
            (1_000..2_000, 0),
        ] {
            assert_eq!(bits.count_in(rows.clone()), expected, "{rows:?}");
        }
    }
}
