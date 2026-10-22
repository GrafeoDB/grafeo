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
}
