//! Values by row of one row group, held from the vector of its first row on.
//!
//! A row group's dense per-row structures (hot property values, edge ends,
//! adjacency offsets) are indexed by row. Held from row 0, the slots of the
//! rows below the group's first live row stay allocated after those rows are
//! deleted, and a rewrite that takes new ids allocates them again.
//! [`GroupRows`] holds a structure from the vector of its first row instead,
//! so those rows cost nothing, and [`GroupRows::trim_front`] gives the slots
//! of a dead head back once the rows below a vector are gone.

use super::versions::VECTOR_ROWS;

/// The first row of the vector that holds `row`.
pub(super) const fn vector_start(row: usize) -> usize {
    row - row % VECTOR_ROWS
}

/// Whether a structure that holds rows from `first_row` on gives back the
/// rows below `head`, the group's first vector with a row in any version:
/// once at least one whole vector below it is dead (both are multiples of
/// [`VECTOR_ROWS`]), so the dead slots it keeps stay under one vector. The
/// trim copies the rows it keeps, at most a group's.
pub(super) const fn trims(first_row: usize, head: usize) -> bool {
    head > first_row
}

/// Values by row of one row group, held from the vector of its first row on:
/// the rows below `first_row` hold none and cost nothing. `first_row` is a
/// multiple of [`VECTOR_ROWS`].
#[derive(Debug, Clone)]
pub(super) struct GroupRows<T> {
    first_row: usize,
    values: Vec<T>,
}

impl<T> Default for GroupRows<T> {
    fn default() -> Self {
        Self {
            first_row: 0,
            values: Vec::new(),
        }
    }
}

impl<T: Clone> GroupRows<T> {
    /// The value at `row`, `None` below the first row held or past the end.
    #[inline]
    pub(super) fn get(&self, row: usize) -> Option<&T> {
        self.values.get(row.checked_sub(self.first_row)?)
    }

    /// The value at `row`, which it holds.
    #[inline]
    pub(super) fn at(&self, row: usize) -> &T {
        &self.values[row - self.first_row]
    }

    /// The value at `row` to change, `None` below the first row held or past
    /// the end.
    #[inline]
    pub(super) fn get_mut(&mut self, row: usize) -> Option<&mut T> {
        self.values.get_mut(row.checked_sub(self.first_row)?)
    }

    /// Sets the value at `row`, filling the rows between the held ones and
    /// `row` with `filler`. An empty holder starts at the vector of `row`; a
    /// row below the start moves the start down to its vector, with fillers in
    /// front (one copy of the held values).
    pub(super) fn put(&mut self, row: usize, value: T, filler: T) {
        if self.values.is_empty() {
            self.first_row = vector_start(row);
        } else if row < self.first_row {
            let start = vector_start(row);
            let mut moved = Vec::with_capacity(self.first_row - start + self.values.len());
            moved.resize(self.first_row - start, filler.clone());
            moved.append(&mut self.values);
            self.values = moved;
            self.first_row = start;
        }
        let at = row - self.first_row;
        if at >= self.values.len() {
            self.values.resize(at, filler);
            self.values.push(value);
        } else {
            self.values[at] = value;
        }
    }

    /// The first row held: a multiple of [`VECTOR_ROWS`].
    #[inline]
    pub(super) fn first_row(&self) -> usize {
        self.first_row
    }

    /// One past the last row held (the first row held when it holds none).
    #[cfg(test)]
    fn end(&self) -> usize {
        self.first_row + self.values.len()
    }

    /// The values of the rows `first_row()..end()`, in row order.
    #[inline]
    pub(super) fn as_slice(&self) -> &[T] {
        &self.values
    }

    /// Drops the rows below `row` (a multiple of [`VECTOR_ROWS`]; at or past
    /// the end, every row) into a new allocation of exactly the rest, so their
    /// memory leaves the heap; returns the heap bytes freed.
    pub(super) fn trim_front(&mut self, row: usize) -> usize {
        debug_assert_eq!(row % VECTOR_ROWS, 0, "a trim keeps whole vectors");
        if row <= self.first_row {
            return 0;
        }
        let before = self.heap_bytes();
        let cut = (row - self.first_row).min(self.values.len());
        let mut kept = Vec::with_capacity(self.values.len() - cut);
        kept.extend(self.values.drain(cut..));
        self.values = kept;
        self.first_row = row;
        before - self.heap_bytes()
    }

    /// Gives back the rows below `head` when they are due (see [`trims`]);
    /// returns the heap bytes freed.
    pub(super) fn trim_dead_head(&mut self, head: usize) -> usize {
        if trims(self.first_row, head) {
            self.trim_front(head)
        } else {
            0
        }
    }

    /// The heap bytes of the values: the allocation's capacity, which is what
    /// the process holds.
    #[inline]
    pub(super) fn heap_bytes(&self) -> usize {
        self.values.capacity() * std::mem::size_of::<T>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_holder_has_nothing_and_costs_nothing() {
        let rows = GroupRows::<u64>::default();
        assert_eq!(rows.get(0), None);
        assert_eq!(rows.get(VECTOR_ROWS * 3), None);
        assert_eq!(rows.as_slice(), &[] as &[u64]);
        assert_eq!(rows.heap_bytes(), 0);
    }

    #[test]
    fn the_first_put_starts_at_the_vector_of_its_row() {
        let mut rows = GroupRows::default();
        let row = 3 * VECTOR_ROWS + 17;
        rows.put(row, 7_u64, 0);
        assert_eq!(rows.first_row(), 3 * VECTOR_ROWS);
        assert_eq!(rows.end(), row + 1);
        assert_eq!(rows.get(row), Some(&7));
        assert_eq!(
            rows.get(3 * VECTOR_ROWS),
            Some(&0),
            "a filler in the vector"
        );
        assert_eq!(rows.get(3 * VECTOR_ROWS - 1), None, "below the start");
        assert_eq!(rows.get(row + 1), None, "past the end");
        assert_eq!(rows.as_slice().len(), 18);
    }

    #[test]
    fn a_put_below_the_start_moves_it_down_with_fillers() {
        let mut rows = GroupRows::default();
        rows.put(3 * VECTOR_ROWS + 1, 30_u32, 9);
        rows.put(VECTOR_ROWS + 5, 10, 9);
        assert_eq!(rows.first_row(), VECTOR_ROWS);
        assert_eq!(rows.get(VECTOR_ROWS + 5), Some(&10));
        assert_eq!(rows.get(3 * VECTOR_ROWS + 1), Some(&30), "kept at its row");
        assert_eq!(rows.get(VECTOR_ROWS), Some(&9));
        assert_eq!(rows.get(2 * VECTOR_ROWS + 100), Some(&9));
        assert_eq!(rows.end(), 3 * VECTOR_ROWS + 2);
    }

    #[test]
    fn a_put_past_the_end_fills_and_a_put_inside_replaces() {
        let mut rows = GroupRows::default();
        rows.put(4, String::from("Alix"), String::new());
        rows.put(9, String::from("Gus"), String::new());
        rows.put(4, String::from("Vincent"), String::new());
        assert_eq!(rows.first_row(), 0);
        assert_eq!(rows.get(4).map(String::as_str), Some("Vincent"));
        assert_eq!(rows.get(7).map(String::as_str), Some(""));
        assert_eq!(rows.get(9).map(String::as_str), Some("Gus"));
        assert_eq!(rows.end(), 10);
    }

    #[test]
    fn get_mut_changes_the_row_it_names() {
        let mut rows = GroupRows::default();
        rows.put(VECTOR_ROWS + 2, 1_i64, 0);
        *rows.get_mut(VECTOR_ROWS + 2).unwrap() = 5;
        assert_eq!(rows.get(VECTOR_ROWS + 2), Some(&5));
        assert!(rows.get_mut(VECTOR_ROWS - 1).is_none());
    }

    #[test]
    fn a_trim_frees_the_head_and_keeps_every_value_above_it() {
        let mut rows = GroupRows::default();
        for row in 0..4 * VECTOR_ROWS {
            rows.put(row, row as u64, 0);
        }
        let before = rows.heap_bytes();
        let freed = rows.trim_front(3 * VECTOR_ROWS);
        assert_eq!(rows.first_row(), 3 * VECTOR_ROWS);
        assert_eq!(
            rows.heap_bytes(),
            VECTOR_ROWS * std::mem::size_of::<u64>(),
            "exactly the rest stays allocated"
        );
        assert_eq!(freed, before - rows.heap_bytes());
        for row in 3 * VECTOR_ROWS..4 * VECTOR_ROWS {
            assert_eq!(rows.get(row), Some(&(row as u64)));
        }
        assert_eq!(rows.get(3 * VECTOR_ROWS - 1), None);
    }

    #[test]
    fn a_trim_at_the_start_frees_nothing_and_a_trim_at_the_end_empties() {
        let mut rows = GroupRows::default();
        rows.put(VECTOR_ROWS + 3, 1_u8, 0);
        assert_eq!(rows.trim_front(VECTOR_ROWS), 0);
        assert_eq!(rows.first_row(), VECTOR_ROWS);

        let mut all = GroupRows::default();
        for row in 0..VECTOR_ROWS {
            all.put(row, 1_u16, 0);
        }
        assert!(all.trim_front(VECTOR_ROWS) > 0);
        assert_eq!(all.as_slice(), &[] as &[u16]);
        assert_eq!(all.heap_bytes(), 0);
        assert_eq!(all.first_row(), VECTOR_ROWS);
        all.put(5, 2, 0);
        assert_eq!(all.first_row(), 0, "an empty holder starts at its new row");
        assert_eq!(all.get(5), Some(&2));
    }

    #[test]
    fn a_trim_past_the_end_empties_and_starts_at_its_row() {
        let mut rows = GroupRows::default();
        rows.put(10, 1_u64, 0);
        assert!(rows.trim_front(3 * VECTOR_ROWS) > 0);
        assert_eq!(rows.as_slice(), &[] as &[u64]);
        assert_eq!(rows.first_row(), 3 * VECTOR_ROWS);
        assert_eq!(rows.get(10), None);
    }

    #[test]
    fn a_dead_head_trims_once_one_whole_vector_is_dead() {
        assert!(!trims(0, 0), "no dead head");
        assert!(
            trims(0, VECTOR_ROWS),
            "one dead vector, whatever lives above"
        );
        assert!(trims(VECTOR_ROWS, 3 * VECTOR_ROWS));
        assert!(
            !trims(2 * VECTOR_ROWS, 2 * VECTOR_ROWS),
            "the start is the head"
        );

        let mut rows = GroupRows::default();
        for row in 0..4 * VECTOR_ROWS {
            rows.put(row, row, 0);
        }
        assert_eq!(rows.trim_dead_head(0), 0);
        assert!(rows.trim_dead_head(VECTOR_ROWS) > 0);
        assert_eq!(rows.first_row(), VECTOR_ROWS);
        assert_eq!(rows.get(VECTOR_ROWS), Some(&VECTOR_ROWS));
        assert_eq!(rows.get(4 * VECTOR_ROWS - 1), Some(&(4 * VECTOR_ROWS - 1)));
    }

    #[test]
    fn heap_bytes_follow_the_capacity() {
        let mut rows = GroupRows::default();
        rows.put(VECTOR_ROWS + 99, 1_u64, 0);
        assert!(rows.heap_bytes() >= 100 * std::mem::size_of::<u64>());
        assert_eq!(
            rows.heap_bytes(),
            rows.values.capacity() * std::mem::size_of::<u64>()
        );
    }
}
