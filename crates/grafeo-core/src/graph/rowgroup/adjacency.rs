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
use super::slots::vector_start;

/// The longest run compared entry by entry along a delta, whose comparisons take far
/// less than a search of the run.
const SHORT_RUN: usize = 16;

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

/// Calls `found` with each pair of an entry of `near` and of `far`, two runs sorted by
/// other end, that reach one node, galloping through the longer from the shorter. The
/// probes it took. The one place that knows how a run is sorted.
pub(super) fn intersect_runs(
    near: &[Adjacent],
    far: &[Adjacent],
    found: &mut dyn FnMut(&Adjacent, &Adjacent),
) -> usize {
    let swapped = near.len() > far.len();
    let (short, long) = if swapped { (far, near) } else { (near, far) };
    let mut probes = 0;
    let mut from = 0;
    let mut index = 0;
    while index < short.len() {
        let other = short[index].other;
        let run = short[index..].partition_point(|entry| entry.other == other);
        let (start, reads) = gallop(long, from, other);
        probes += reads + 1;
        from = start;
        let count = long[start..].partition_point(|entry| entry.other == other);
        for short_entry in &short[index..index + run] {
            for long_entry in &long[start..start + count] {
                if swapped {
                    found(long_entry, short_entry);
                } else {
                    found(short_entry, long_entry);
                }
            }
        }
        index += run;
    }
    probes
}

/// Calls `found` with each pair of an entry of `run`, sorted by other end, and of
/// `delta`, in no order, that reach one node. The probes it took. A short run compares
/// its entries along the delta, which a hub's holds many of; a longer one is searched
/// for each entry of the delta.
pub(super) fn intersect_run_with_delta(
    run: &[Adjacent],
    delta: &[Adjacent],
    found: &mut dyn FnMut(&Adjacent, &Adjacent),
) -> usize {
    if run.len() < SHORT_RUN {
        for entry in run {
            for other in delta.iter().filter(|other| other.other == entry.other) {
                found(entry, other);
            }
        }
        return run.len() * delta.len();
    }
    for other in delta {
        for entry in entries_to(run, other.other) {
            found(entry, other);
        }
    }
    delta.len()
}

/// The entries of `run`, sorted by other end, that reach `other`.
fn entries_to(run: &[Adjacent], other: u64) -> &[Adjacent] {
    let start = run.partition_point(|entry| entry.other < other);
    let count = run[start..].partition_point(|entry| entry.other == other);
    &run[start..start + count]
}

/// The first position from `from` of `sorted` whose other end is at least `other`,
/// by doubling steps and then a search of the last step, and the probes it took.
fn gallop(sorted: &[Adjacent], from: usize, other: u64) -> (usize, usize) {
    let mut step = 1;
    let mut low = from;
    let mut probes = 0;
    while low + step < sorted.len() && sorted[low + step].other < other {
        low += step;
        step *= 2;
        probes += 1;
    }
    let high = (low + step + 1).min(sorted.len());
    let found = sorted[low..high].partition_point(|entry| entry.other < other);
    (low + found, probes + step.ilog2() as usize + 1)
}

/// One direction of a node row group's adjacency.
#[derive(Debug, Default)]
pub(super) struct Adjacency {
    /// The row `offsets` starts at: the first row of the vector of the
    /// lowest row with a sorted list, so the rows below cost nothing.
    first_row: usize,
    /// Row `first_row + i` (for `i` below `offsets.len() - 1`) has the
    /// sorted list `sorted[offsets[i]..offsets[i + 1]]`.
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
    /// The heap bytes of the lists in `delta`, kept as they change.
    delta_bytes: usize,
}

/// The heap bytes of a list of adjacency entries.
fn list_bytes(list: &Vec<Adjacent>) -> usize {
    list.capacity() * std::mem::size_of::<Adjacent>()
}

impl Adjacency {
    /// The row's sorted list, removed edges included.
    fn sorted_of(&self, row: usize) -> &[Adjacent] {
        let Some(at) = row.checked_sub(self.first_row) else {
            return &[];
        };
        match (self.offsets.get(at), self.offsets.get(at + 1)) {
            (Some(start), Some(end)) => &self.sorted[*start..*end],
            _ => &[],
        }
    }

    /// Whether `adjacent` was taken out of a sorted list.
    pub(super) fn is_removed(&self, adjacent: &Adjacent) -> bool {
        !self.removed.is_empty() && self.removed.contains(&adjacent.edge)
    }

    /// The row's delta, in the order it came.
    pub(super) fn delta_of(&self, row: usize) -> &[Adjacent] {
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

    /// How many entries a read of the row may have to go through: its sorted list
    /// and its delta, removed edges and other types among them.
    pub(super) fn entries_of(&self, row: usize) -> usize {
        self.sorted_of(row).len() + self.delta_of(row).len()
    }

    /// The row's sorted list as its runs of one type each, each sorted by other end,
    /// with what `is_removed` drops still in them.
    pub(super) fn runs_of(&self, row: usize) -> impl Iterator<Item = &[Adjacent]> + '_ {
        let mut rest = self.sorted_of(row);
        std::iter::from_fn(move || {
            let edge_type = rest.first()?.edge_type;
            let end = rest.partition_point(|adjacent| adjacent.edge_type <= edge_type);
            let (run, after) = rest.split_at(end);
            rest = after;
            Some(run)
        })
    }

    /// The row's edges to `other`, in the order [`of`](Self::of) gives them:
    /// a binary search in the sorted list per edge type, then the delta's.
    pub(super) fn to_other(&self, row: usize, other: u64) -> impl Iterator<Item = Adjacent> + '_ {
        self.runs_of(row)
            .flat_map(move |run| entries_to(run, other))
            .filter(|adjacent| !self.is_removed(adjacent))
            .chain(
                self.delta_of(row)
                    .iter()
                    .filter(move |adjacent| adjacent.other == other),
            )
            .copied()
    }

    /// Sets the sorted list of `row`, after the rows set before it, as a
    /// load reads them chunk by chunk: rows ascending, each list sorted,
    /// nothing in the delta.
    /// The heap bytes of the rows' offsets into the sorted lists: the one
    /// figure of the offsets' size the walk and the account read.
    fn offsets_bytes(&self) -> usize {
        self.offsets.capacity() * std::mem::size_of::<usize>()
    }

    pub(super) fn push_sorted(&mut self, row: usize, list: &[Adjacent]) {
        if self.offsets.is_empty() {
            self.first_row = vector_start(row);
            self.offsets.push(0);
        }
        debug_assert!(row >= self.first_row, "a load sets rows ascending");
        while self.first_row + self.offsets.len() <= row {
            self.offsets.push(self.sorted.len());
        }
        self.sorted.extend_from_slice(list);
        self.offsets.push(self.sorted.len());
    }

    /// Adds `list` at the end of the last row's sorted list: the next piece
    /// of a list a load reads over several chunks.
    pub(super) fn extend_last_sorted(&mut self, list: &[Adjacent]) {
        self.sorted.extend_from_slice(list);
        if let Some(end) = self.offsets.last_mut() {
            *end = self.sorted.len();
        }
    }

    /// The last entry of the sorted lists: the end of the last row's list.
    pub(super) fn last_sorted(&self) -> Option<&Adjacent> {
        self.sorted.last()
    }

    /// The rows with a sorted list, ascending, each with it.
    pub(super) fn sorted_rows(&self) -> impl Iterator<Item = (usize, &[Adjacent])> + '_ {
        self.offsets
            .windows(2)
            .enumerate()
            .filter(|(_, ends)| ends[0] < ends[1])
            .map(|(at, ends)| (self.first_row + at, &self.sorted[ends[0]..ends[1]]))
    }

    /// The entries of the sorted lists.
    pub(super) fn sorted_len(&self) -> usize {
        self.sorted.len()
    }

    /// Gives back the room the sorted lists grew past their entries, once a
    /// load set every row.
    pub(super) fn shrink_to_fit(&mut self) {
        self.offsets.shrink_to_fit();
        self.sorted.shrink_to_fit();
    }

    /// Adds an edge to the row's delta.
    pub(super) fn push(&mut self, row: usize, adjacent: Adjacent) {
        let list = self.delta.entry(row).or_default();
        let before = list_bytes(list);
        list.push(adjacent);
        self.delta_bytes += list_bytes(list) - before;
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
                self.delta_bytes = super::sub_count(self.delta_bytes, list_bytes(list));
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

    /// Merges once the removed edges are more than half the sorted lists (at
    /// least [`MERGE_MIN`]), as a push merges a delta that large: a deleted
    /// edge's entry leaves the lists then, so a clear gives its entries back
    /// at the collect that unlinks them, not at some later push. The collect
    /// calls it, never a commit or a rollback.
    pub(super) fn merge_if_removed_dominate(&mut self) {
        if self.removed.len() > MERGE_MIN.max(self.sorted.len() / 2) {
            self.merge();
        }
    }

    /// Takes every edge out of the row's adjacency.
    pub(super) fn clear(&mut self, row: usize) {
        if let Some(list) = self.delta.remove(&row) {
            self.delta_len -= list.len();
            self.delta_bytes = super::sub_count(self.delta_bytes, list_bytes(&list));
            self.delta_rows.clear(row);
        }
        let sorted: Vec<u64> = self
            .sorted_of(row)
            .iter()
            .map(|adjacent| adjacent.edge)
            .collect();
        self.removed.extend(sorted);
    }

    /// Whether the delta is empty and no edge waits to be dropped: a merge
    /// would change nothing.
    pub(super) fn is_merged(&self) -> bool {
        self.delta_len == 0 && self.removed.is_empty()
    }

    /// Merges the delta into the sorted lists and drops the removed edges.
    /// The offsets start again at the vector of the lowest row left with a
    /// list, so the rows below a dead head give their offsets back.
    pub(super) fn merge(&mut self) {
        let held = (!self.offsets.is_empty())
            .then(|| (self.first_row, self.first_row + self.offsets.len() - 1));
        let added = self
            .delta
            .keys()
            .fold(None, |range: Option<(usize, usize)>, &row| {
                Some(range.map_or((row, row + 1), |(low, high)| {
                    (low.min(row), high.max(row + 1))
                }))
            });
        let (low, high) = match (held, added) {
            (Some((a, b)), Some((c, d))) => (a.min(c), b.max(d)),
            (Some(range), None) | (None, Some(range)) => range,
            (None, None) => (0, 0),
        };
        let mut offsets = Vec::new();
        let mut first_row = 0;
        // The removed edges are entries of the sorted lists (`remove` and `clear` take only
        // those), so the merged lists hold exactly the rest and the delta.
        let mut sorted = Vec::with_capacity(
            self.sorted.len().saturating_sub(self.removed.len()) + self.delta_len,
        );
        for row in low..high {
            let before = sorted.len();
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
            if offsets.is_empty() {
                if sorted.len() == before {
                    // Below the lowest row with a list: no offset yet.
                    continue;
                }
                first_row = vector_start(row);
                // The rows of its vector below it have empty lists.
                offsets.resize(row - first_row + 1, 0);
            }
            offsets.push(sorted.len());
        }
        offsets.shrink_to_fit();
        self.first_row = first_row;
        self.offsets = offsets;
        self.sorted = sorted;
        // New, empty: a clear would keep the room the removed edges and the delta took.
        self.removed = FxHashSet::default();
        self.delta = FxHashMap::default();
        self.delta_rows = Bitset::default();
        self.delta_len = 0;
        self.delta_bytes = 0;
    }

    /// Gives back the offsets of the rows below `row` (a multiple of
    /// [`VECTOR_ROWS`](super::versions::VECTOR_ROWS)), whose lists are empty
    /// or removed: their sorted entries go at the next merge, as a removed
    /// edge's do. Returns the heap bytes freed.
    pub(super) fn trim_front(&mut self, row: usize) -> usize {
        let cut = row
            .saturating_sub(self.first_row)
            .min(self.offsets.len().saturating_sub(1));
        if cut == 0 {
            // Only the end is left: the rows start at `row`, so the same
            // dead head is not due again.
            if self.offsets.len() == 1 && row > self.first_row {
                self.first_row = row;
            }
            return 0;
        }
        let before = self.offsets.capacity();
        let mut kept = Vec::with_capacity(self.offsets.len() - cut);
        kept.extend(self.offsets.drain(cut..));
        self.offsets = kept;
        // With only the end left, no row below `row` has a list: the rows
        // start there (a dead head is due only past it).
        self.first_row = if self.offsets.len() == 1 {
            row
        } else {
            self.first_row + cut
        };
        (before - self.offsets.capacity()) * std::mem::size_of::<usize>()
    }

    /// Gives back the offsets of the rows below `head` when they are due
    /// (see [`trims`](super::slots::trims)); returns the heap bytes freed.
    pub(super) fn trim_dead_head(&mut self, head: usize) -> usize {
        if super::slots::trims(self.first_row, head) {
            self.trim_front(head)
        } else {
            0
        }
    }

    /// The heap bytes it holds: its offsets, sorted lists, removed edges,
    /// the delta's lists, map and rows. A walk of the delta's lists, for
    /// the oracle.
    pub(super) fn heap_bytes(&self) -> usize {
        self.fixed_bytes() + self.delta.values().map(list_bytes).sum::<usize>()
    }

    /// [`heap_bytes`](Self::heap_bytes) from what it keeps as the delta
    /// changes: a few loads.
    pub(super) fn counted_bytes(&self) -> usize {
        self.fixed_bytes() + self.delta_bytes
    }

    /// Sets its count to what the delta's lists take: after a load that
    /// built it without counting.
    pub(super) fn retally(&mut self) {
        self.delta_bytes = self.delta.values().map(list_bytes).sum();
    }

    /// Its heap bytes but the delta's lists.
    fn fixed_bytes(&self) -> usize {
        self.offsets_bytes()
            + self.sorted.capacity() * std::mem::size_of::<Adjacent>()
            + self.removed.capacity() * std::mem::size_of::<u64>()
            // The table's allocation, not its capacity: a removed row's slot
            // may stay a tombstone, which takes capacity depending on where
            // the map's random hash put the row, so two adjacencies that
            // took the same pushes and removes would report different sizes.
            + self.delta.allocation_size()
            + self.delta_rows.heap_bytes()
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

    /// A trim past every row with a list leaves only the end, and the rows
    /// start at the trimmed row: the same dead head is not due again.
    #[test]
    fn a_trim_past_every_list_starts_the_rows_at_the_trimmed_row() {
        use super::super::versions::VECTOR_ROWS;

        let mut adjacency = Adjacency::default();
        for row in 0..4 {
            adjacency.push(row, edge(0, row as u64, row as u64));
        }
        adjacency.merge();
        let head = 2 * VECTOR_ROWS;
        assert!(adjacency.trim_dead_head(head) > 0, "the offsets went");
        assert_eq!(adjacency.offsets.len(), 1, "only the end is left");
        assert_eq!(adjacency.first_row, head);
        assert!(!super::super::slots::trims(adjacency.first_row, head));
        assert_eq!(adjacency.trim_dead_head(head), 0);
        assert_eq!(edges(&adjacency, 3), Vec::<u64>::new());
    }

    /// Two adjacencies that took the same pushes and removes report the same
    /// heap size, whatever the random hash state of their maps: a removed
    /// row's tombstone does not count.
    #[test]
    fn the_same_writes_report_the_same_heap_size() {
        let build = || {
            let mut adjacency = Adjacency::default();
            for row in 0..200 {
                adjacency.push(row, edge(1, row as u64, row as u64));
            }
            for row in (0..200).step_by(3) {
                adjacency.remove(row, row as u64);
            }
            adjacency
        };
        let sizes: std::collections::BTreeSet<usize> =
            (0..64).map(|_| build().heap_bytes()).collect();
        assert_eq!(sizes.len(), 1, "sizes {sizes:?}");
    }
}
