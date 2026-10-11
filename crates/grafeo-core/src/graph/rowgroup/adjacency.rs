//! A node row group's adjacency in one direction, in FD6's shape: per
//! node, a list sorted by (edge type, other node), so an expand of one edge
//! type reads one slice of it, and a hot delta per node for the edges added
//! since the lists were built.
//!
//! The delta merges into the sorted lists once it holds more than half as
//! many entries as they do (at least [`MERGE_MIN`]): a linear merge per node,
//! each node's delta sorted first. So an edge is copied about twice over its
//! life, and a checkpoint (H1c) merges what is left. An edge taken out of a sorted list (an undone
//! create) is skipped by readers until the next merge drops it. A deleted
//! edge stays listed for readers at earlier epochs; readers check each
//! edge's visibility.

use grafeo_common::utils::hash::{FxHashMap, FxHashSet};

use super::bitset::Bitset;

/// The fewest delta entries a merge waits for.
const MERGE_MIN: usize = 1_024;

/// An edge in a node's adjacency. The field order is the sort order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Adjacent {
    /// The edge's type id.
    pub edge_type: u32,
    /// The node at the edge's other end.
    pub other: u64,
    /// The edge.
    pub edge: u64,
}

/// One direction of a node row group's adjacency.
#[derive(Debug, Default)]
pub(super) struct Adjacency {
    /// Row `r` (below `offsets.len() - 1`) has the sorted list
    /// `sorted[offsets[r]..offsets[r + 1]]`.
    offsets: Vec<usize>,
    sorted: Vec<Adjacent>,
    /// Edges taken out of the sorted lists since they were built.
    removed: FxHashSet<u64>,
    /// Edges added since, per row, in the order they came.
    delta: FxHashMap<usize, Vec<Adjacent>>,
    /// The rows with edges in `delta`, so a read of another row skips the
    /// map.
    delta_rows: Bitset,
    /// The entries in `delta`.
    delta_len: usize,
}

impl Adjacency {
    /// The row's sorted list, removed edges included.
    fn sorted_of(&self, row: usize) -> &[Adjacent] {
        match (self.offsets.get(row), self.offsets.get(row + 1)) {
            (Some(start), Some(end)) => &self.sorted[*start..*end],
            _ => &[],
        }
    }

    /// Whether `adjacent` was taken out of a sorted list.
    fn is_removed(&self, adjacent: &Adjacent) -> bool {
        !self.removed.is_empty() && self.removed.contains(&adjacent.edge)
    }

    /// The row's delta.
    fn delta_of(&self, row: usize) -> &[Adjacent] {
        if self.delta_rows.get(row) {
            self.delta.get(&row).map_or(&[], Vec::as_slice)
        } else {
            &[]
        }
    }

    /// The row's edges: the sorted list, then the delta in the order it
    /// came.
    pub(super) fn of(&self, row: usize) -> impl Iterator<Item = Adjacent> + '_ {
        self.sorted_of(row)
            .iter()
            .filter(|adjacent| !self.is_removed(adjacent))
            .chain(self.delta_of(row))
            .copied()
    }

    /// The row's edges of one type: a slice of the sorted list, then the
    /// delta's of that type.
    pub(super) fn of_type(
        &self,
        row: usize,
        edge_type: u32,
    ) -> impl Iterator<Item = Adjacent> + '_ {
        let sorted = self.sorted_of(row);
        let start = sorted.partition_point(|adjacent| adjacent.edge_type < edge_type);
        let end = sorted.partition_point(|adjacent| adjacent.edge_type <= edge_type);
        sorted[start..end]
            .iter()
            .filter(|adjacent| !self.is_removed(adjacent))
            .chain(
                self.delta_of(row)
                    .iter()
                    .filter(move |adjacent| adjacent.edge_type == edge_type),
            )
            .copied()
    }

    /// Sets the sorted list of each row of `lists` (row, list), rows
    /// ascending and each list sorted, as a load hands them over; nothing is
    /// in the delta.
    pub(super) fn from_sorted(lists: impl IntoIterator<Item = (usize, Vec<Adjacent>)>) -> Self {
        let mut adjacency = Self::default();
        adjacency.offsets.push(0);
        for (row, list) in lists {
            while adjacency.offsets.len() <= row {
                adjacency.offsets.push(adjacency.sorted.len());
            }
            adjacency.sorted.extend(list);
            adjacency.offsets.push(adjacency.sorted.len());
        }
        adjacency
    }

    /// Adds an edge to the row's delta.
    pub(super) fn push(&mut self, row: usize, adjacent: Adjacent) {
        self.delta.entry(row).or_default().push(adjacent);
        self.delta_rows.set(row);
        self.delta_len += 1;
        if self.delta_len > MERGE_MIN.max(self.sorted.len() / 2) {
            self.merge();
        }
    }

    /// Takes edge `edge` out of the row's adjacency.
    pub(super) fn remove(&mut self, row: usize, edge: u64) {
        if let Some(list) = self.delta.get_mut(&row)
            && let Some(index) = list.iter().position(|adjacent| adjacent.edge == edge)
        {
            list.remove(index);
            self.delta_len -= 1;
            if list.is_empty() {
                self.delta.remove(&row);
                self.delta_rows.clear(row);
            }
            return;
        }
        if self
            .sorted_of(row)
            .iter()
            .any(|adjacent| adjacent.edge == edge)
        {
            self.removed.insert(edge);
        }
    }

    /// Takes every edge out of the row's adjacency.
    pub(super) fn clear(&mut self, row: usize) {
        if let Some(list) = self.delta.remove(&row) {
            self.delta_len -= list.len();
            self.delta_rows.clear(row);
        }
        let sorted: Vec<u64> = self
            .sorted_of(row)
            .iter()
            .map(|adjacent| adjacent.edge)
            .collect();
        self.removed.extend(sorted);
    }

    /// Merges the delta into the sorted lists and drops the removed edges.
    pub(super) fn merge(&mut self) {
        let rows = self
            .offsets
            .len()
            .saturating_sub(1)
            .max(self.delta.keys().max().map_or(0, |row| row + 1));
        let mut offsets = Vec::with_capacity(rows + 1);
        let mut sorted = Vec::with_capacity(self.sorted.len() + self.delta_len);
        offsets.push(0);
        for row in 0..rows {
            let mut added = self.delta.remove(&row).unwrap_or_default();
            added.sort_unstable();
            let mut kept = self
                .sorted_of(row)
                .iter()
                .filter(|adjacent| !self.removed.contains(&adjacent.edge))
                .copied()
                .peekable();
            let mut added = added.into_iter().peekable();
            loop {
                let next = match (kept.peek(), added.peek()) {
                    (Some(a), Some(b)) if a <= b => kept.next(),
                    (Some(_), Some(_)) | (None, Some(_)) => added.next(),
                    (Some(_), None) => kept.next(),
                    (None, None) => break,
                };
                sorted.extend(next);
            }
            offsets.push(sorted.len());
        }
        self.offsets = offsets;
        self.sorted = sorted;
        self.removed.clear();
        self.delta.clear();
        self.delta_rows = Bitset::default();
        self.delta_len = 0;
    }

    pub(super) fn heap_bytes(&self) -> usize {
        self.offsets.capacity() * std::mem::size_of::<usize>()
            + (self.sorted.capacity() + self.delta_len) * std::mem::size_of::<Adjacent>()
            + self.removed.capacity() * std::mem::size_of::<u64>()
            + self.delta.capacity()
                * (std::mem::size_of::<usize>() + std::mem::size_of::<Vec<Adjacent>>())
    }
}

#[cfg(test)]
mod tests {
    use super::{Adjacency, Adjacent, MERGE_MIN};

    fn edge(edge_type: u32, other: u64, edge: u64) -> Adjacent {
        Adjacent {
            edge_type,
            other,
            edge,
        }
    }

    fn edges(adjacency: &Adjacency, row: usize) -> Vec<u64> {
        adjacency.of(row).map(|adjacent| adjacent.edge).collect()
    }

    #[test]
    fn a_merge_sorts_each_list_by_type_and_other_node() {
        let mut adjacency = Adjacency::default();
        adjacency.push(3, edge(2, 19, 10));
        adjacency.push(3, edge(1, 88, 11));
        adjacency.push(3, edge(1, 7, 12));
        adjacency.push(0, edge(5, 3, 13));
        assert_eq!(
            edges(&adjacency, 3),
            [10, 11, 12],
            "the delta keeps the order edges came in"
        );
        adjacency.merge();
        assert_eq!(edges(&adjacency, 3), [12, 11, 10]);
        assert_eq!(edges(&adjacency, 0), [13]);
        assert!(edges(&adjacency, 1).is_empty() && edges(&adjacency, 4).is_empty());

        adjacency.push(3, edge(1, 50, 14));
        assert_eq!(
            edges(&adjacency, 3),
            [12, 11, 10, 14],
            "new edges come after the sorted ones"
        );
        adjacency.merge();
        assert_eq!(edges(&adjacency, 3), [12, 14, 11, 10], "merged into place");
    }

    #[test]
    fn a_typed_expand_reads_one_slice_and_the_delta() {
        let mut adjacency = Adjacency::default();
        for (edge_type, other, id) in [(2, 1, 1), (1, 2, 2), (3, 3, 3), (1, 4, 4), (2, 5, 5)] {
            adjacency.push(0, edge(edge_type, other, id));
        }
        adjacency.merge();
        adjacency.push(0, edge(1, 0, 6));
        let typed = |edge_type| -> Vec<u64> {
            adjacency
                .of_type(0, edge_type)
                .map(|adjacent| adjacent.edge)
                .collect()
        };
        assert_eq!(typed(1), [2, 4, 6]);
        assert_eq!(typed(2), [1, 5]);
        assert_eq!(typed(3), [3]);
        assert_eq!(typed(4), Vec::<u64>::new());
    }

    #[test]
    fn removed_edges_are_skipped_and_dropped_by_the_next_merge() {
        let mut adjacency = Adjacency::default();
        for id in 0..4 {
            adjacency.push(1, edge(0, id, id));
        }
        adjacency.merge();
        adjacency.push(1, edge(0, 9, 9));
        adjacency.remove(1, 2);
        adjacency.remove(1, 9);
        adjacency.remove(1, 1_000);
        assert_eq!(edges(&adjacency, 1), [0, 1, 3]);
        adjacency.merge();
        assert_eq!(edges(&adjacency, 1), [0, 1, 3]);
        assert!(adjacency.removed.is_empty());
        adjacency.clear(1);
        assert_eq!(edges(&adjacency, 1), Vec::<u64>::new());
    }

    #[test]
    fn the_delta_merges_once_it_outgrows_half_the_lists() {
        let mut adjacency = Adjacency::default();
        for id in 0..=MERGE_MIN as u64 {
            adjacency.push(usize::try_from(id % 7).unwrap(), edge(0, id, id));
        }
        assert_eq!(adjacency.delta_len, 0, "the push past the minimum merged");
        assert_eq!(adjacency.sorted.len(), MERGE_MIN + 1);
    }
}
