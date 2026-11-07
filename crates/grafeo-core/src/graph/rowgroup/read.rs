//! The row-group store's read traits.
//!
//! The reads at a snapshot (`*_versioned`, `*_at_epoch`, the reads at a view
//! `*_at`, the visibility checks) read each row's version and the update
//! chains of its values and labels: the store answers exactly at any view,
//! point reads, batches, label checks, scans, lookups and adjacency
//! included. A vector without a chain costs a value or label read one test
//! more than a read in place.
//!
//! The reads without a view read the committed state at the store's epoch,
//! but for the label scans and adjacency, which list what any reader may
//! see: a label scan lists the nodes with the label in any version (open
//! transactions' labels and labels a later write took off included), and
//! adjacency lists every edge not yet collected, deleted ones included, for
//! readers at earlier epochs; a reader checks each one at its view. So do
//! the property index's maybe-equal lookups without a view: every node the
//! index holds that a reader may find (see `index`).

use std::sync::Arc;
use std::sync::atomic::Ordering;

use grafeo_common::change::Labels;
use grafeo_common::types::{ArcStr, EdgeId, EpochId, NodeId, PropertyKey, ReadView, Value};
use grafeo_common::utils::error::Result;
use grafeo_common::utils::hash::FxHashMap;
use smallvec::SmallVec;

use super::adjacency::Adjacent;
use super::adjacency::{Adjacency, intersect_run_with_delta, intersect_runs};
use super::column::Columns;
#[cfg(feature = "text-index")]
use super::index::TextLimit;
use super::{Inner, ROWS_PER_GROUP, Read, RowGroupStore};
use crate::graph::Direction;
use crate::graph::lpg::{CompareOp, Edge, Node};
use crate::graph::traits::GraphStore;
use crate::graph::traits::{CommonEdge, HopFrom, KeyColumnGroup};
#[cfg(feature = "vector-index")]
use crate::index::vector::{DistanceMetric, HnswConfig};
use crate::statistics::{EdgeTypeStatistics, LabelStatistics, Statistics};

pub(super) use crate::graph::traits::compare_for_range;

/// A count kept as a signed total, as a size.
fn size(count: i64) -> usize {
    usize::try_from(count.max(0)).unwrap_or(usize::MAX)
}

/// One way of a hop an intersection reads.
struct HopSide<'a> {
    adjacency: &'a Adjacency,
    row: usize,
    /// A `Both` reads a self-loop once, going out: the incoming side drops it.
    skips_self_loops: bool,
}

/// The ways a hop goes, and what they keep.
struct HopSides<'a> {
    sides: SmallVec<[HopSide<'a>; 2]>,
    /// The edge types it follows, `None` for every type.
    ids: Option<SmallVec<[u32; 4]>>,
    node: u64,
}

/// A list this short asks its far ends rather than walking a long delta.
const FEW_ENTRIES: usize = 4;

/// The delta that makes a walk of it cost more than asking those far ends.
const LONG_DELTA: usize = 64;

impl<'a> HopSides<'a> {
    /// The entries of the hop's lists, other types and removed edges among them.
    fn entries(&self) -> usize {
        self.sides
            .iter()
            .map(|side| side.adjacency.entries_of(side.row))
            .sum()
    }

    /// The entries of the hop's deltas.
    fn delta_entries(&self) -> usize {
        self.sides
            .iter()
            .map(|side| side.adjacency.delta_of(side.row).len())
            .sum()
    }

    /// The sorted runs of `side` of the types the hop follows.
    fn runs(&self, side: &HopSide<'a>) -> impl Iterator<Item = &'a [Adjacent]> + use<'a, '_> {
        side.adjacency.runs_of(side.row).filter(|run| {
            self.ids.as_ref().is_none_or(|ids| {
                run.first()
                    .is_some_and(|entry| ids.contains(&entry.edge_type))
            })
        })
    }

    /// Whether the hop follows `entry` of `side`.
    fn keeps(&self, entry: &Adjacent, side: &HopSide<'_>) -> bool {
        let dropped_self_loop = side.skips_self_loops && entry.other == self.node;
        !dropped_self_loop
            && !side.adjacency.is_removed(entry)
            && self
                .ids
                .as_ref()
                .is_none_or(|ids| ids.contains(&entry.edge_type))
    }
}

impl Inner {
    /// The value of `key` at a row of `columns` as `read` sees it.
    fn value_at(&self, columns: &Columns, row: usize, key: &str, read: Read) -> Option<Value> {
        columns.get_at(self.keys.get_id(key)?, row, read)
    }

    /// A row's values for `keys` (all of them for `None`) as `read` sees
    /// them.
    fn values_of(
        &self,
        columns: &Columns,
        row: usize,
        keys: Option<&[PropertyKey]>,
        read: Read,
    ) -> FxHashMap<PropertyKey, Value> {
        match keys {
            None => self.values_at(columns, row, read).into_iter().collect(),
            Some(keys) => keys
                .iter()
                .filter_map(|key| {
                    Some((
                        key.clone(),
                        self.value_at(columns, row, key.as_str(), read)?,
                    ))
                })
                .collect(),
        }
    }

    /// The value of `key` of node `id` as `read` sees the node.
    pub(super) fn node_value_at(&self, id: u64, key: &str, read: Read) -> Option<Value> {
        let (group, row) = self.node_seen(id, read)?;
        self.value_at(&group.columns, row, key, read)
    }

    /// The value of `key` of edge `id` as `read` sees the edge.
    pub(super) fn edge_value_at(&self, id: u64, key: &str, read: Read) -> Option<Value> {
        let (group, row) = self.edge_seen(id, read)?;
        self.value_at(&group.columns, row, key, read)
    }

    /// Node `id`'s values for `keys` (all of them for `None`) as `read` sees
    /// the node: none when it does not see it.
    fn node_values_at(
        &self,
        id: NodeId,
        keys: Option<&[PropertyKey]>,
        read: Read,
    ) -> FxHashMap<PropertyKey, Value> {
        match self.node(id.as_u64()) {
            Some((group, row)) if group.version(row).visible(read) => {
                self.values_of(&group.columns, row, keys, read)
            }
            _ => FxHashMap::default(),
        }
    }

    /// Edge `id`'s values for `keys` as `read` sees the edge.
    fn edge_values_at(
        &self,
        id: EdgeId,
        keys: &[PropertyKey],
        read: Read,
    ) -> FxHashMap<PropertyKey, Value> {
        match self.edge(id.as_u64()) {
            Some((group, row)) if group.version(row).visible(read) => {
                self.values_of(&group.columns, row, Some(keys), read)
            }
            _ => FxHashMap::default(),
        }
    }

    /// The nodes `read` sees with at least one of the label ids, ascending
    /// and each once: the label bitmaps give the newest labels, and the rows
    /// a label chain holds are read at the view instead. A row group without
    /// label chains reads its bitmaps alone.
    fn nodes_with_labels_at(&self, label_ids: &[u32], read: Read) -> Vec<NodeId> {
        let mut ids = Vec::new();
        for (index, group) in &self.nodes {
            let base = index * ROWS_PER_GROUP;
            let changed = group.label_updates.rows();
            // One label and no label chain, the common case: the rows of
            // its bitmap `read` sees are the answer, ascending and once.
            if changed.is_none()
                && let [label] = label_ids
            {
                if let Some(rows) = group.labels.get(label) {
                    group.versions.for_each_visible(rows, read, |row| {
                        ids.push(NodeId::new(base + row as u64));
                    });
                }
                continue;
            }
            // The rows the newest label bitmaps hold that `read` sees, but
            // those a label chain holds, which are read at the view.
            let mut rows = Vec::new();
            for label_rows in label_ids.iter().filter_map(|label| group.labels.get(label)) {
                group.versions.for_each_visible(label_rows, read, |row| {
                    if changed.as_ref().is_none_or(|changed| !changed.get(row)) {
                        rows.push(row);
                    }
                });
            }
            if let Some(changed) = &changed {
                rows.extend(changed.rows().filter(|row| {
                    group.version(*row).visible(read)
                        && label_ids
                            .iter()
                            .any(|label| group.has_label_at(*row, *label, read))
                }));
            }
            rows.sort_unstable();
            rows.dedup();
            ids.extend(rows.into_iter().map(|row| NodeId::new(base + row as u64)));
        }
        ids
    }

    /// About `size` nodes `read` sees with label `label`, spread over the row groups, and whether
    /// they are all of the label's nodes. Each group gives its share of `size` by its rows in the
    /// newest label bitmap, at ranks in a golden-ratio sequence that starts at a point the label
    /// sets, and a picked row counts when `read` sees it and its labels at `read` hold the label. It
    /// reads the groups' counts and the picked rows, never the label's list. A label of at most
    /// `size` rows is read whole. The same nodes for the same data, whoever asks first.
    pub(super) fn sampled_nodes_with_label(
        &self,
        label: u32,
        size: usize,
        read: Read,
    ) -> (Vec<NodeId>, bool) {
        let counts: Vec<(u64, usize)> = self
            .nodes
            .iter()
            .map(|(index, group)| {
                let rows = group.labels.get(&label);
                (
                    *index,
                    rows.map_or(0, |rows| rows.count_in(0..rows.words().len() * 64)),
                )
            })
            .collect();
        let total: usize = counts.iter().map(|(_, count)| count).sum();
        if total <= size {
            return (self.nodes_with_labels_at(&[label], read), true);
        }
        const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
        let start = u64::from(label).wrapping_mul(GOLDEN);
        let mut picked = Vec::with_capacity(size + counts.len());
        for (index, count) in counts {
            let (Some(group), Some(rows)) = (
                self.nodes.get(&index),
                self.nodes
                    .get(&index)
                    .and_then(|group| group.labels.get(&label)),
            ) else {
                continue;
            };
            if count == 0 {
                continue;
            }
            let share = (size * count).div_ceil(total) as u64;
            let count = count as u64;
            let mut ranks: Vec<u64> = (1..=share)
                .map(|i| {
                    let fraction = u128::from(start.wrapping_add(i.wrapping_mul(GOLDEN)));
                    u64::try_from((fraction * u128::from(count)) >> 64).unwrap_or(count - 1)
                })
                .collect();
            ranks.sort_unstable();
            ranks.dedup();
            let base = index * ROWS_PER_GROUP;
            for row in rows.rows_at_ranks(&ranks) {
                if group.version(row).visible(read) && group.has_label_at(row, label, read) {
                    picked.push(NodeId::new(base + row as u64));
                }
            }
        }
        (picked, false)
    }

    /// Every node `read` sees, ascending, a word of rows at a time.
    fn visible_node_ids(&self, read: Read) -> Vec<NodeId> {
        let mut ids = Vec::new();
        for (index, group) in &self.nodes {
            let base = index * ROWS_PER_GROUP;
            group.versions.for_each_visible_row(read, |row| {
                ids.push(NodeId::new(base + row as u64));
            });
        }
        ids
    }

    /// Every node row `read` sees whose values meet `conditions`, ascending.
    fn nodes_matching_at(&self, conditions: &[(&str, Value)], read: Read) -> Vec<NodeId> {
        self.node_rows(|_, group, row| {
            group.version(row).visible(read)
                && conditions.iter().all(|(key, value)| {
                    self.value_at(&group.columns, row, key, read).as_ref() == Some(value)
                })
        })
    }

    /// Every node row `read` sees whose `property` lies within the bounds.
    fn nodes_in_range_at(
        &self,
        property: &str,
        range: &crate::graph::ValueRange,
        read: Read,
    ) -> Vec<NodeId> {
        // The key's id once, not per row: no node has a key the store has
        // no id for.
        let Some(key) = self.keys.get_id(property) else {
            return Vec::new();
        };
        self.node_rows(|_, group, row| {
            group.version(row).visible(read)
                && group
                    .columns
                    .get_at(key, row, read)
                    .is_some_and(|value| range.contains(&value))
        })
    }

    /// Every node row, ascending, that `keep` accepts.
    fn node_rows(
        &self,
        mut keep: impl FnMut(u64, &super::NodeGroup, usize) -> bool,
    ) -> Vec<NodeId> {
        let mut ids = Vec::new();
        for (index, group) in &self.nodes {
            for (row, _) in group.versions.rows() {
                let id = index * ROWS_PER_GROUP + row as u64;
                if keep(id, group, row) {
                    ids.push(NodeId::new(id));
                }
            }
        }
        ids
    }

    /// A node's entries in `direction`, of one of the edge type ids or of
    /// all (`None`), as (other node, edge), outgoing ones first: each kept
    /// direction's sorted list, then its delta; without the incoming edges
    /// kept, the incoming ones from the edge table, by edge id. One type
    /// reads one slice of each sorted list.
    fn entries_in(
        &self,
        node: NodeId,
        direction: Direction,
        edge_types: Option<&[u32]>,
    ) -> Vec<(NodeId, EdgeId)> {
        let Some((group, row)) = self.node(node.as_u64()) else {
            return Vec::new();
        };
        let directions: &[bool] = match direction {
            Direction::Outgoing => &[true],
            Direction::Incoming => &[false],
            Direction::Both => &[true, false],
        };
        let pair = |adjacent: Adjacent| (NodeId::new(adjacent.other), EdgeId::new(adjacent.edge));
        let wanted = |adjacent: &Adjacent| {
            edge_types.is_none_or(|types| types.contains(&adjacent.edge_type))
        };
        let mut entries = Vec::new();
        for &outgoing in directions {
            if outgoing || self.keeps_incoming() {
                let adjacency = group.adjacency(outgoing);
                match edge_types {
                    None => entries.extend(adjacency.of(row).map(pair)),
                    Some(&[edge_type]) => {
                        entries.extend(adjacency.of_type(row, edge_type).map(pair));
                    }
                    Some(_) => entries.extend(adjacency.of(row).filter(wanted).map(pair)),
                }
            } else {
                entries.extend(
                    self.incoming_by_scan(node.as_u64())
                        .filter(wanted)
                        .map(pair),
                );
            }
        }
        entries
    }

    /// A node's adjacency entries, as (other node, edge): each direction's
    /// sorted list, then its delta (see [`entries_in`](Self::entries_in)).
    pub(super) fn adjacency(&self, node: NodeId, direction: Direction) -> Vec<(NodeId, EdgeId)> {
        self.entries_in(node, direction, None)
    }

    /// Calls `seen` with each edge of `node` in `direction`, of one of the
    /// edge type ids or of all (`None`), that `read` sees, as (other node,
    /// edge), in the order of [`entries_in`](Self::entries_in); `Both` lists
    /// a self-loop once, as outgoing (its incoming entry is the same edge).
    /// Nothing for a node `read` does not see. No list is built: a count or
    /// a collect reads the adjacency once.
    ///
    /// The other end of an edge `read` sees is one it sees too, by
    /// construction: an edge's create needs both ends where its writer reads
    /// them, the engine's claims let no transaction commit an edge to a node
    /// a concurrent one deletes (first writer wins, and at commit), a delete
    /// of a node with an edge its writer sees is refused, and a load leaves
    /// out an edge without its ends. So no end is checked here.
    fn each_seen(
        &self,
        node: NodeId,
        direction: Direction,
        edge_types: Option<&[u32]>,
        read: Read,
        mut seen: impl FnMut(NodeId, EdgeId),
    ) {
        let raw = node.as_u64();
        let Some((group, row)) = self.node_seen(raw, read) else {
            return;
        };
        let both = direction == Direction::Both;
        let mut visit = |adjacent: Adjacent, outgoing: bool| {
            let self_loop = adjacent.other == raw;
            if (both && !outgoing && self_loop)
                || edge_types.is_some_and(|types| !types.contains(&adjacent.edge_type))
            {
                return;
            }
            if self.edge_visible(adjacent.edge, read) {
                seen(NodeId::new(adjacent.other), EdgeId::new(adjacent.edge));
            }
        };
        let directions: &[bool] = match direction {
            Direction::Outgoing => &[true],
            Direction::Incoming => &[false],
            Direction::Both => &[true, false],
        };
        for &outgoing in directions {
            if outgoing || self.keeps_incoming() {
                let adjacency = group.adjacency(outgoing);
                match edge_types {
                    Some(&[edge_type]) => adjacency
                        .of_type(row, edge_type)
                        .for_each(|adjacent| visit(adjacent, outgoing)),
                    _ => adjacency
                        .of(row)
                        .for_each(|adjacent| visit(adjacent, outgoing)),
                }
            } else {
                self.incoming_by_scan(raw)
                    .for_each(|adjacent| visit(adjacent, outgoing));
            }
        }
    }

    /// Whether `read` sees an edge [`each_seen`](Self::each_seen) would
    /// give, stopping at the first.
    fn any_seen(
        &self,
        node: NodeId,
        direction: Direction,
        edge_types: Option<&[u32]>,
        read: Read,
    ) -> bool {
        self.seen_until(node, direction, edge_types, read, |_| true)
    }

    /// Gives the other end of each edge [`each_seen`](Self::each_seen) would give to `stop`,
    /// until it answers true: whether it did.
    fn seen_until(
        &self,
        node: NodeId,
        direction: Direction,
        edge_types: Option<&[u32]>,
        read: Read,
        mut stop: impl FnMut(NodeId) -> bool,
    ) -> bool {
        let raw = node.as_u64();
        let Some((group, row)) = self.node_seen(raw, read) else {
            return false;
        };
        let both = direction == Direction::Both;
        let counts = |adjacent: Adjacent, outgoing: bool| {
            let self_loop = adjacent.other == raw;
            !((both && !outgoing && self_loop)
                || edge_types.is_some_and(|types| !types.contains(&adjacent.edge_type)))
                && self.edge_visible(adjacent.edge, read)
        };
        let directions: &[bool] = match direction {
            Direction::Outgoing => &[true],
            Direction::Incoming => &[false],
            Direction::Both => &[true, false],
        };
        directions.iter().any(|&outgoing| {
            if outgoing || self.keeps_incoming() {
                let adjacency = group.adjacency(outgoing);
                let mut stops = |adjacent: Adjacent| {
                    counts(adjacent, outgoing) && stop(NodeId::new(adjacent.other))
                };
                match edge_types {
                    Some(&[edge_type]) => adjacency.of_type(row, edge_type).any(&mut stops),
                    _ => adjacency.of(row).any(&mut stops),
                }
            } else {
                self.incoming_by_scan(raw)
                    .any(|adjacent| counts(adjacent, outgoing) && stop(NodeId::new(adjacent.other)))
            }
        })
    }

    /// The most edges [`each_seen`](Self::each_seen) can give, without going through them:
    /// the entries of the lists it reads, removed and unseen ones among them, and `Both` counts
    /// a self-loop twice. `None` when it would scan for the incoming edges this store does not
    /// keep.
    fn entries_at_most(
        &self,
        node: NodeId,
        direction: Direction,
        edge_types: Option<&[u32]>,
        read: Read,
    ) -> Option<usize> {
        let Some((group, row)) = self.node_seen(node.as_u64(), read) else {
            return Some(0);
        };
        let directions: &[bool] = match direction {
            Direction::Outgoing => &[true],
            Direction::Incoming => &[false],
            Direction::Both => &[true, false],
        };
        let mut entries = 0;
        for &outgoing in directions {
            if !(outgoing || self.keeps_incoming()) {
                return None;
            }
            let adjacency = group.adjacency(outgoing);
            entries += match edge_types {
                None => adjacency.entries_of(row),
                Some(types) => {
                    let sorted: usize = adjacency
                        .runs_of(row)
                        .filter(|run| types.contains(&run[0].edge_type))
                        .map(<[Adjacent]>::len)
                        .sum();
                    let delta = adjacency.delta_of(row).iter();
                    sorted
                        + delta
                            .filter(|adjacent| types.contains(&adjacent.edge_type))
                            .count()
                }
            };
        }
        Some(entries)
    }

    /// The type ids of `types`, the names the store has an edge type of;
    /// `None` for every type (no names). Empty when it has none of them.
    fn type_ids(&self, types: &[ArcStr]) -> Option<SmallVec<[u32; 4]>> {
        (!types.is_empty()).then(|| {
            types
                .iter()
                .filter_map(|name| self.edge_types.get_id(name))
                .collect()
        })
    }

    /// The other ends of the first `limit` of the edges [`edges_seen`](Self::edges_seen)
    /// lists, reading no further.
    pub(super) fn first_ends_seen(
        &self,
        node: NodeId,
        direction: Direction,
        types: &[ArcStr],
        limit: usize,
        read: Read,
    ) -> Vec<NodeId> {
        let ids = self.type_ids(types);
        let mut ends = Vec::new();
        if limit == 0 || ids.as_ref().is_some_and(SmallVec::is_empty) {
            return ends;
        }
        self.seen_until(node, direction, ids.as_deref(), read, |end| {
            ends.push(end);
            ends.len() >= limit
        });
        ends
    }

    /// [`entries_at_most`](Self::entries_at_most) of `types` (every type when empty).
    pub(super) fn entries_of_types_at_most(
        &self,
        node: NodeId,
        direction: Direction,
        types: &[ArcStr],
        read: Read,
    ) -> Option<usize> {
        let ids = self.type_ids(types);
        if ids.as_ref().is_some_and(SmallVec::is_empty) {
            return Some(0);
        }
        self.entries_at_most(node, direction, ids.as_deref(), read)
    }

    /// The edges of `node` in `direction` of one of `types` (every type
    /// when empty) that `read` sees, as (other node, edge); `Both` lists a
    /// self-loop once. Empty for a node `read` does not see, and for types
    /// the store has no edge of.
    fn edges_seen(
        &self,
        node: NodeId,
        direction: Direction,
        types: &[ArcStr],
        read: Read,
    ) -> Vec<(NodeId, EdgeId)> {
        let ids = self.type_ids(types);
        if ids.as_ref().is_some_and(SmallVec::is_empty) {
            return Vec::new();
        }
        if ids.is_none()
            && let Some((group, row, outgoing)) = self.kept_direction(node, direction, read)
        {
            let adjacency = group.adjacency(outgoing);
            let mut edges = Vec::new();
            for adjacent in adjacency.of(row) {
                if self.edge_visible(adjacent.edge, read) {
                    edges.push((NodeId::new(adjacent.other), EdgeId::new(adjacent.edge)));
                }
            }
            return edges;
        }
        if ids.is_none() && direction == Direction::Incoming && !self.keeps_incoming() {
            let mut edges = Vec::new();
            self.each_incoming_by_scan(node, read, |other, edge| edges.push((other, edge)));
            return edges;
        }
        let mut edges = Vec::new();
        self.each_seen(node, direction, ids.as_deref(), read, |other, edge| {
            edges.push((other, edge));
        });
        edges
    }

    /// The adjacency a hop reads for an intersection, one per way it goes; a `Both`
    /// lists a self-loop once, going out. `None` when `read` does not see the node.
    fn hop_sides<'a>(&'a self, hop: &HopFrom<'_>, read: Read) -> Option<HopSides<'a>> {
        let (group, row) = self.node_seen(hop.node.as_u64(), read)?;
        let ways: &[bool] = match hop.direction {
            Direction::Outgoing => &[true],
            Direction::Incoming => &[false],
            Direction::Both => &[true, false],
        };
        Some(HopSides {
            sides: ways
                .iter()
                .map(|&outgoing| HopSide {
                    adjacency: group.adjacency(outgoing),
                    row,
                    skips_self_loops: hop.direction == Direction::Both && !outgoing,
                })
                .collect(),
            ids: self.type_ids(hop.edge_types),
            node: hop.node.as_u64(),
        })
    }

    /// [`GraphStore::common_edges`] on the lists where they lie: each pair of sorted
    /// runs gallops from the shorter, and a delta is searched entry by entry. `None`
    /// when the store keeps no list of a way a hop goes. A node `read` does not see
    /// has no edges.
    fn common_edges_seen(
        &self,
        near: &HopFrom<'_>,
        far: &HopFrom<'_>,
        read: Read,
    ) -> Option<(Vec<CommonEdge>, usize)> {
        let kept =
            |hop: &HopFrom<'_>| hop.direction == Direction::Outgoing || self.keeps_incoming();
        if !kept(near) || !kept(far) {
            return None;
        }
        let (Some(near_sides), Some(far_sides)) =
            (self.hop_sides(near, read), self.hop_sides(far, read))
        else {
            return Some((Vec::new(), 0));
        };
        // A very short list against a hub's delta, which only a walk can search: the far
        // ends' own lists, usually short, say it sooner. Near 4 entries the two are level.
        if near_sides.entries() < FEW_ENTRIES && far_sides.delta_entries() >= LONG_DELTA {
            return Some(self.common_edges_by_far_ends(&near_sides, far, read, false));
        }
        if far_sides.entries() < FEW_ENTRIES && near_sides.delta_entries() >= LONG_DELTA {
            return Some(self.common_edges_by_far_ends(&far_sides, near, read, true));
        }
        let mut common = Vec::new();
        let mut probes = 0;
        let mut pair = |near_entry: &Adjacent,
                        far_entry: &Adjacent,
                        near_side: &HopSide<'_>,
                        far_side: &HopSide<'_>| {
            if near_sides.keeps(near_entry, near_side)
                && far_sides.keeps(far_entry, far_side)
                && self.edge_visible(near_entry.edge, read)
                && self.edge_visible(far_entry.edge, read)
            {
                common.push(CommonEdge {
                    node: NodeId::new(near_entry.other),
                    near_edge: EdgeId::new(near_entry.edge),
                    far_edge: EdgeId::new(far_entry.edge),
                });
            }
        };
        for near_side in &near_sides.sides {
            for far_side in &far_sides.sides {
                for near_run in near_sides.runs(near_side) {
                    for far_run in far_sides.runs(far_side) {
                        probes += intersect_runs(near_run, far_run, &mut |a, b| {
                            pair(a, b, near_side, far_side);
                        });
                    }
                }
                // A delta against the other side's runs and delta.
                let near_delta = near_side.adjacency.delta_of(near_side.row);
                let far_delta = far_side.adjacency.delta_of(far_side.row);
                for far_run in far_sides.runs(far_side) {
                    probes += intersect_run_with_delta(far_run, near_delta, &mut |far, near| {
                        pair(near, far, near_side, far_side);
                    });
                }
                for near_run in near_sides.runs(near_side) {
                    probes += intersect_run_with_delta(near_run, far_delta, &mut |near, far| {
                        pair(near, far, near_side, far_side);
                    });
                }
                probes += near_delta.len() * far_delta.len();
                for near_entry in near_delta {
                    for far_entry in far_delta
                        .iter()
                        .filter(|entry| entry.other == near_entry.other)
                    {
                        pair(near_entry, far_entry, near_side, far_side);
                    }
                }
            }
        }
        common.sort_unstable();
        Some((common, probes))
    }

    /// [`common_edges_seen`](Self::common_edges_seen) from the entries of `short`, each
    /// far end asked for its edges to `other`'s node: `flipped` when `short` is the far
    /// hop of the two.
    fn common_edges_by_far_ends(
        &self,
        short: &HopSides<'_>,
        other: &HopFrom<'_>,
        read: Read,
        flipped: bool,
    ) -> (Vec<CommonEdge>, usize) {
        let back = match other.direction {
            Direction::Outgoing => Direction::Incoming,
            Direction::Incoming => Direction::Outgoing,
            Direction::Both => Direction::Both,
        };
        let mut common = Vec::new();
        let mut probes = 0;
        for side in &short.sides {
            for entry in side.adjacency.of(side.row) {
                probes += 1;
                if !short.keeps(&entry, side) || !self.edge_visible(entry.edge, read) {
                    continue;
                }
                let node = NodeId::new(entry.other);
                let edges = self.edges_between_seen(node, other.node, back, other.edge_types, read);
                probes += edges.len();
                for edge in edges {
                    let own = EdgeId::new(entry.edge);
                    let (near_edge, far_edge) = if flipped { (edge, own) } else { (own, edge) };
                    common.push(CommonEdge {
                        node,
                        near_edge,
                        far_edge,
                    });
                }
            }
        }
        common.sort_unstable();
        (common, probes)
    }

    /// The edges [`edges_seen`](Self::edges_seen) lists for `node` whose other
    /// end is `other`, in its order, from a search of each adjacency it keeps.
    fn edges_between_seen(
        &self,
        node: NodeId,
        other: NodeId,
        direction: Direction,
        types: &[ArcStr],
        read: Read,
    ) -> Vec<EdgeId> {
        if direction != Direction::Outgoing && !self.keeps_incoming() {
            let mut edges = self.edges_seen(node, direction, types, read);
            edges.retain(|(end, _)| *end == other);
            return edges.into_iter().map(|(_, edge)| edge).collect();
        }
        let ids = self.type_ids(types);
        if ids.as_ref().is_some_and(SmallVec::is_empty) {
            return Vec::new();
        }
        let (raw, target) = (node.as_u64(), other.as_u64());
        let Some((group, row)) = self.node_seen(raw, read) else {
            return Vec::new();
        };
        // One way and one type, the far end's list holds the same edges in the same
        // order (by edge, the type's other end being fixed): the shorter is read,
        // as a hub's delta is searched entry by entry.
        if direction != Direction::Both
            && self.keeps_incoming()
            && ids.as_ref().is_some_and(|ids| ids.len() == 1)
            && let Some((far_group, far_row)) = self.node_seen(target, read)
        {
            let outgoing = direction == Direction::Outgoing;
            let near_entries = group.adjacency(outgoing).entries_of(row);
            let far_entries = far_group.adjacency(!outgoing).entries_of(far_row);
            if far_entries < near_entries {
                let mut edges: Vec<EdgeId> = far_group
                    .adjacency(!outgoing)
                    .to_other(far_row, raw)
                    .filter(|adjacent| {
                        ids.as_ref()
                            .is_none_or(|ids| ids.contains(&adjacent.edge_type))
                            && self.edge_visible(adjacent.edge, read)
                    })
                    .map(|adjacent| EdgeId::new(adjacent.edge))
                    .collect();
                edges.sort_unstable();
                return edges;
            }
        }
        let directions: &[bool] = match direction {
            Direction::Outgoing => &[true],
            Direction::Incoming => &[false],
            Direction::Both => &[true, false],
        };
        let mut edges = Vec::new();
        for &outgoing in directions {
            // A self-loop is listed once, going out, when both ways are read.
            if direction == Direction::Both && !outgoing && raw == target {
                continue;
            }
            edges.extend(
                group
                    .adjacency(outgoing)
                    .to_other(row, target)
                    .filter(|adjacent| {
                        ids.as_ref()
                            .is_none_or(|ids| ids.contains(&adjacent.edge_type))
                            && self.edge_visible(adjacent.edge, read)
                    })
                    .map(|adjacent| EdgeId::new(adjacent.edge)),
            );
        }
        edges
    }

    /// How many edges [`edges_seen`](Self::edges_seen) lists for `node` in
    /// `direction` of every type, counted without the list.
    fn degree_seen(&self, node: NodeId, direction: Direction, read: Read) -> usize {
        if let Some((group, row, outgoing)) = self.kept_direction(node, direction, read) {
            return group
                .adjacency(outgoing)
                .of(row)
                .filter(|adjacent| self.edge_visible(adjacent.edge, read))
                .count();
        }
        if direction == Direction::Incoming && !self.keeps_incoming() {
            let mut count = 0;
            self.each_incoming_by_scan(node, read, |_, _| count += 1);
            return count;
        }
        let mut count = 0;
        self.each_seen(node, direction, None, read, |_, _| count += 1);
        count
    }

    /// How many edges [`each_seen`](Self::each_seen) gives for `node` in
    /// `direction` of the edge type ids (all for `None`) whose other end has
    /// every label id of `end_labels` as `read` sees it, counted without the
    /// list.
    fn count_seen(
        &self,
        node: NodeId,
        direction: Direction,
        edge_types: Option<&[u32]>,
        end_labels: &[u32],
        read: Read,
    ) -> usize {
        if edge_types.is_none() && end_labels.is_empty() {
            return self.degree_seen(node, direction, read);
        }
        let mut count = 0;
        self.each_seen(node, direction, edge_types, read, |other, _| {
            let labelled = end_labels.is_empty()
                || self.node(other.as_u64()).is_some_and(|(group, row)| {
                    group.version(row).visible(read)
                        && end_labels
                            .iter()
                            .all(|&label| group.has_label_at(row, label, read))
                });
            count += usize::from(labelled);
        });
        count
    }

    /// The other ends of the edges [`edges_seen`](Self::edges_seen) lists
    /// for `node` in `direction` of every type, in its order.
    fn neighbors_seen(&self, node: NodeId, direction: Direction, read: Read) -> Vec<NodeId> {
        let mut nodes = Vec::new();
        self.each_seen(node, direction, None, read, |other, _| nodes.push(other));
        nodes
    }

    /// How many edges a committed reader or transaction `read` sees: what
    /// the edges of every node it sees add up to, from their sources (their
    /// ends are seen too, see [`each_seen`](Self::each_seen)).
    fn edges_seen_count(&self, read: Read) -> usize {
        self.edges
            .values()
            .map(|group| group.versions.count_visible(read))
            .sum()
    }

    /// Calls `seen` with each edge into `node` that `read` sees, as (source,
    /// edge), by edge id, for a store that keeps no incoming adjacency: a
    /// plain loop over each edge group's target column, the version read in
    /// the edge's own group. Nothing for a node `read` does not see.
    fn each_incoming_by_scan(
        &self,
        node: NodeId,
        read: Read,
        mut seen: impl FnMut(NodeId, EdgeId),
    ) {
        let raw = node.as_u64();
        if self.node_seen(raw, read).is_none() {
            return;
        }
        for (index, group) in &self.edges {
            let base = index * ROWS_PER_GROUP;
            let first = group.dst.first_row();
            for (offset, &dst) in group.dst.as_slice().iter().enumerate() {
                let row = first + offset;
                if dst == raw && group.versions.visible(row, read) {
                    seen(
                        NodeId::new(*group.src.at(row)),
                        EdgeId::new(base + row as u64),
                    );
                }
            }
        }
    }

    /// For a read of one direction of `node` whose adjacency the store keeps
    /// (outgoing, or incoming with the incoming edges kept): the node's group
    /// and row, if `read` sees it, and whether the direction is outgoing.
    /// `None` for both directions and for incoming edges read from the edge
    /// table, which take the general path. A read of the one direction is a
    /// plain loop over the node's adjacency.
    fn kept_direction(
        &self,
        node: NodeId,
        direction: Direction,
        read: Read,
    ) -> Option<(&super::NodeGroup, usize, bool)> {
        let outgoing = match direction {
            Direction::Outgoing => true,
            Direction::Incoming if self.keeps_incoming() => false,
            Direction::Incoming | Direction::Both => return None,
        };
        let (group, row) = self.node_seen(node.as_u64(), read)?;
        Some((group, row, outgoing))
    }

    /// Whether some node's or edge's (`nodes` false) column of `property`
    /// may hold a value that meets `op value`: by the zone maps of the hot
    /// columns (every value they held, so every version a reader may see),
    /// and always for a column still cold or a key no row group has.
    fn might_match(
        &self,
        nodes: bool,
        property: &PropertyKey,
        op: CompareOp,
        value: &Value,
    ) -> bool {
        // A key no row group has a column of might match, as for a store
        // without zone maps (the planner keeps the filter).
        let Some(key) = self.keys.get_id(property.as_str()) else {
            return true;
        };
        let mut held = false;
        let mut decide = |found: Option<bool>| match found {
            Some(true) => Some(true),
            Some(false) => {
                held = true;
                None
            }
            None => None,
        };
        let matched = if nodes {
            self.nodes
                .values()
                .any(|group| decide(group.columns.might_match(key, op, value)).is_some())
        } else {
            self.edges
                .values()
                .any(|group| decide(group.columns.might_match(key, op, value)).is_some())
        };
        matched || !held
    }

    fn statistics(&self) -> Statistics {
        let mut statistics = Statistics::new();
        statistics.total_nodes = size(self.counts.nodes) as u64;
        statistics.total_edges = size(self.counts.edges) as u64;
        #[expect(
            clippy::cast_precision_loss,
            reason = "an estimate of the average degree, as LpgStore computes it"
        )]
        let average_degree = if statistics.total_nodes > 0 {
            statistics.total_edges as f64 / statistics.total_nodes as f64
        } else {
            0.0
        };
        for (id, name) in self.labels.iter() {
            let count = self.counts.labels.get(id as usize).copied().unwrap_or(0);
            if count > 0 {
                statistics.update_label(
                    name.as_str(),
                    LabelStatistics::new(size(count) as u64)
                        .with_degrees(average_degree, average_degree),
                );
            }
        }
        for (id, name) in self.edge_types.iter() {
            let count = self
                .counts
                .edge_types
                .get(id as usize)
                .copied()
                .unwrap_or(0);
            if count > 0 {
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "an estimate of the average degree, as LpgStore computes it"
                )]
                let degree = if statistics.total_nodes > 0 {
                    count as f64 / statistics.total_nodes as f64
                } else {
                    0.0
                };
                statistics.update_edge_type(
                    name.as_str(),
                    EdgeTypeStatistics::new(size(count) as u64, degree, degree),
                );
            }
        }
        statistics
    }
}

/// The reads at a view: existence from the rows' versions, values and
/// labels from the columns and bitmaps and their update chains, lookups
/// through a property index from its entries of every version, each one
/// read again at the view (`index`). The row-group store reads nothing from
/// a file it could fail on, so every read is `Ok`.
impl GraphStore for RowGroupStore {
    fn node(&self, id: NodeId, view: ReadView) -> Result<Option<Node>> {
        Ok(self.inner.read().read_node(id.as_u64(), Read::view(view)))
    }

    fn edge(&self, id: EdgeId, view: ReadView) -> Result<Option<Edge>> {
        Ok(self.inner.read().read_edge(id.as_u64(), Read::view(view)))
    }

    fn is_node_visible(&self, id: NodeId, view: ReadView) -> Result<bool> {
        Ok(self
            .inner
            .read()
            .node_visible(id.as_u64(), Read::view(view)))
    }

    fn is_edge_visible(&self, id: EdgeId, view: ReadView) -> Result<bool> {
        Ok(self
            .inner
            .read()
            .edge_visible(id.as_u64(), Read::view(view)))
    }

    fn filter_visible_node_ids(&self, ids: &[NodeId], view: ReadView) -> Result<Vec<NodeId>> {
        let (inner, read) = (self.inner.read(), Read::view(view));
        Ok(ids
            .iter()
            .copied()
            .filter(|id| inner.node_visible(id.as_u64(), read))
            .collect())
    }

    fn node_property(&self, id: NodeId, key: &str, view: ReadView) -> Result<Option<Value>> {
        Ok(self
            .inner
            .read()
            .node_value_at(id.as_u64(), key, Read::view(view)))
    }

    fn edge_property(&self, id: EdgeId, key: &str, view: ReadView) -> Result<Option<Value>> {
        Ok(self
            .inner
            .read()
            .edge_value_at(id.as_u64(), key, Read::view(view)))
    }

    fn node_labels(&self, id: NodeId, view: ReadView) -> Result<Option<Labels>> {
        let (inner, read) = (self.inner.read(), Read::view(view));
        let Some((group, row)) = inner.node(id.as_u64()) else {
            return Ok(None);
        };
        if !group.version(row).visible(read) {
            return Ok(None);
        }
        Ok(Some(inner.label_names(&group.labels_at(row, read))))
    }

    fn node_label_mask(&self, id: NodeId, view: ReadView) -> Result<Option<u64>> {
        let (inner, read) = (self.inner.read(), Read::view(view));
        let Some((group, row)) = inner.node(id.as_u64()) else {
            return Ok(None);
        };
        if !group.version(row).visible(read) {
            return Ok(None);
        }
        Ok(group.label_mask_at(row, read))
    }

    fn label_mask_names(&self, mask: u64) -> Result<Labels> {
        let ids: Vec<u32> = (0..64).filter(|bit| mask & (1 << bit) != 0).collect();
        let mut names = self.inner.read().label_names(&ids);
        names.sort();
        Ok(names)
    }

    fn node_has_label(&self, id: NodeId, label: &str, view: ReadView) -> Result<Option<bool>> {
        let (inner, read) = (self.inner.read(), Read::view(view));
        let Some((group, row)) = inner.node(id.as_u64()) else {
            return Ok(None);
        };
        if !group.version(row).visible(read) {
            return Ok(None);
        }
        Ok(Some(
            inner
                .label_id(label)
                .is_some_and(|label| group.has_label_at(row, label, read)),
        ))
    }

    fn edge_type(&self, id: EdgeId, view: ReadView) -> Result<Option<ArcStr>> {
        let inner = self.inner.read();
        let Some((group, row)) = inner.edge_seen(id.as_u64(), Read::view(view)) else {
            return Ok(None);
        };
        let (_, _, edge_type) = group.ends(row);
        Ok(inner.edge_types.get_name(edge_type).cloned())
    }

    fn node_property_batch(
        &self,
        ids: &[NodeId],
        key: &PropertyKey,
        view: ReadView,
    ) -> Result<Vec<Option<Value>>> {
        let (inner, read) = (self.inner.read(), Read::view(view));
        // The key's id once for the batch: no node has a key the store has
        // no id for.
        let Some(key) = inner.keys.get_id(key.as_str()) else {
            return Ok(vec![None; ids.len()]);
        };
        Ok(ids
            .iter()
            .map(|id| {
                let (group, row) = inner.node_seen(id.as_u64(), read)?;
                group.columns.get_at(key, row, read)
            })
            .collect())
    }

    fn node_properties_batch(
        &self,
        ids: &[NodeId],
        view: ReadView,
    ) -> Result<Vec<FxHashMap<PropertyKey, Value>>> {
        let (inner, read) = (self.inner.read(), Read::view(view));
        Ok(ids
            .iter()
            .map(|id| inner.node_values_at(*id, None, read))
            .collect())
    }

    fn node_properties_selective_batch(
        &self,
        ids: &[NodeId],
        keys: &[PropertyKey],
        view: ReadView,
    ) -> Result<Vec<FxHashMap<PropertyKey, Value>>> {
        let (inner, read) = (self.inner.read(), Read::view(view));
        Ok(ids
            .iter()
            .map(|id| inner.node_values_at(*id, Some(keys), read))
            .collect())
    }

    fn edge_properties_selective_batch(
        &self,
        ids: &[EdgeId],
        keys: &[PropertyKey],
        view: ReadView,
    ) -> Result<Vec<FxHashMap<PropertyKey, Value>>> {
        let (inner, read) = (self.inner.read(), Read::view(view));
        Ok(ids
            .iter()
            .map(|id| inner.edge_values_at(*id, keys, read))
            .collect())
    }

    fn edges(
        &self,
        node: NodeId,
        direction: Direction,
        types: &[ArcStr],
        view: ReadView,
    ) -> Result<Vec<(NodeId, EdgeId)>> {
        Ok(self
            .inner
            .read()
            .edges_seen(node, direction, types, Read::view(view)))
    }

    fn common_edges(
        &self,
        near: &HopFrom<'_>,
        far: &HopFrom<'_>,
        view: ReadView,
    ) -> Result<(Vec<CommonEdge>, usize)> {
        if let Some(found) = self
            .inner
            .read()
            .common_edges_seen(near, far, Read::view(view))
        {
            return Ok(found);
        }
        crate::graph::traits::common_edges_by_lists(
            self.edges(near.node, near.direction, near.edge_types, view)?,
            self.edges(far.node, far.direction, far.edge_types, view)?,
        )
    }

    fn edges_between(
        &self,
        node: NodeId,
        other: NodeId,
        direction: Direction,
        types: &[ArcStr],
        view: ReadView,
    ) -> Result<Vec<EdgeId>> {
        Ok(self
            .inner
            .read()
            .edges_between_seen(node, other, direction, types, Read::view(view)))
    }

    fn has_edge(
        &self,
        node: NodeId,
        direction: Direction,
        types: &[ArcStr],
        view: ReadView,
    ) -> Result<bool> {
        let inner = self.inner.read();
        let ids = inner.type_ids(types);
        if ids.as_ref().is_some_and(SmallVec::is_empty) {
            return Ok(false);
        }
        Ok(inner.any_seen(node, direction, ids.as_deref(), Read::view(view)))
    }

    fn count_edges(
        &self,
        node: NodeId,
        direction: Direction,
        types: &[ArcStr],
        end_labels: &[ArcStr],
        view: ReadView,
    ) -> Result<usize> {
        let inner = self.inner.read();
        let ids = inner.type_ids(types);
        if ids.as_ref().is_some_and(SmallVec::is_empty) {
            return Ok(0);
        }
        // A label the store has no id for is a label no node has.
        let mut labels: SmallVec<[u32; 2]> = SmallVec::with_capacity(end_labels.len());
        for label in end_labels {
            match inner.label_id(label) {
                Some(id) => labels.push(id),
                None => return Ok(0),
            }
        }
        Ok(inner.count_seen(node, direction, ids.as_deref(), &labels, Read::view(view)))
    }

    fn degree(&self, node: NodeId, direction: Direction, view: ReadView) -> Result<usize> {
        Ok(self
            .inner
            .read()
            .degree_seen(node, direction, Read::view(view)))
    }

    fn neighbors(&self, node: NodeId, direction: Direction, view: ReadView) -> Result<Vec<NodeId>> {
        Ok(self
            .inner
            .read()
            .neighbors_seen(node, direction, Read::view(view)))
    }

    /// Whether it keeps each node's incoming edges: without them an
    /// incoming read scans the edge table (see
    /// [`RowGroupStore::with_backward_edges`]).
    fn has_backward_adjacency(&self) -> bool {
        self.keeps_backward_edges()
    }

    fn node_ids(&self, view: ReadView) -> Result<Vec<NodeId>> {
        Ok(self.inner.read().visible_node_ids(Read::view(view)))
    }

    fn nodes_by_label(&self, label: &str, view: ReadView) -> Result<Vec<NodeId>> {
        let inner = self.inner.read();
        Ok(inner.label_id(label).map_or_else(Vec::new, |label| {
            inner.nodes_with_labels_at(&[label], Read::view(view))
        }))
    }

    fn nodes_with_any_label(&self, labels: &[&str], view: ReadView) -> Result<Vec<NodeId>> {
        let inner = self.inner.read();
        let mut label_ids: Vec<u32> = labels
            .iter()
            .filter_map(|label| inner.label_id(label))
            .collect();
        label_ids.sort_unstable();
        label_ids.dedup();
        Ok(inner.nodes_with_labels_at(&label_ids, Read::view(view)))
    }

    /// From the count the store keeps when it answers for `view` (see
    /// `counts_answer`), by a scan otherwise.
    fn nodes_by_label_count(&self, label: &str, view: ReadView) -> Result<usize> {
        let inner = self.inner.read();
        let Some(label) = inner.label_id(label) else {
            return Ok(0);
        };
        if self.counts_answer(view) {
            return Ok(inner
                .counts
                .labels
                .get(label as usize)
                .copied()
                .map_or(0, size));
        }
        Ok(inner.nodes_with_labels_at(&[label], Read::view(view)).len())
    }

    fn label_count_is_kept(&self, _label: &str, view: ReadView) -> bool {
        self.counts_answer(view)
    }

    /// From the count the store keeps when it answers for `view`, by a scan
    /// otherwise.
    fn node_count(&self, view: ReadView) -> Result<usize> {
        let inner = self.inner.read();
        if self.counts_answer(view) {
            return Ok(size(inner.counts.nodes));
        }
        let read = Read::view(view);
        Ok(inner
            .nodes
            .values()
            .map(|group| group.versions.count_visible(read))
            .sum())
    }

    /// From the count the store keeps when it answers for `view`, by a scan
    /// otherwise.
    fn edge_count(&self, view: ReadView) -> Result<usize> {
        let inner = self.inner.read();
        if self.counts_answer(view) {
            return Ok(size(inner.counts.edges));
        }
        Ok(inner.edges_seen_count(Read::view(view)))
    }

    fn all_labels(&self) -> Result<Vec<String>> {
        Ok(self
            .inner
            .read()
            .labels
            .iter()
            .map(|(_, name)| name.to_string())
            .collect())
    }

    fn all_edge_types(&self) -> Result<Vec<String>> {
        Ok(self
            .inner
            .read()
            .edge_types
            .iter()
            .map(|(_, name)| name.to_string())
            .collect())
    }

    /// From the counts the store keeps when they answer for `view` (see
    /// `counts_answer`), as the default reads them otherwise.
    fn labels_in_use(&self, view: ReadView) -> Result<Vec<String>> {
        let inner = self.inner.read();
        if !self.counts_answer(view) {
            let read = Read::view(view);
            return Ok(inner
                .labels
                .iter()
                .filter(|&(id, _)| !inner.nodes_with_labels_at(&[id], read).is_empty())
                .map(|(_, name)| name.to_string())
                .collect());
        }
        Ok(inner
            .labels
            .iter()
            .filter(|&(id, _)| inner.counts.labels.get(id as usize).is_some_and(|&n| n > 0))
            .map(|(_, name)| name.to_string())
            .collect())
    }

    /// From the counts the store keeps when they answer for `view`, by
    /// reading the edges otherwise.
    fn edge_types_in_use(&self, view: ReadView) -> Result<Vec<String>> {
        if !self.counts_answer(view) {
            let mut in_use = std::collections::BTreeSet::new();
            for node in self.node_ids(view)? {
                for (_, edge) in self.edges(node, Direction::Outgoing, &[], view)? {
                    if let Some(edge_type) = self.edge_type(edge, view)? {
                        in_use.insert(edge_type.to_string());
                    }
                }
            }
            return Ok(in_use.into_iter().collect());
        }
        let inner = self.inner.read();
        Ok(inner
            .edge_types
            .iter()
            .filter(|&(id, _)| {
                inner
                    .counts
                    .edge_types
                    .get(id as usize)
                    .is_some_and(|&n| n > 0)
            })
            .map(|(_, name)| name.to_string())
            .collect())
    }

    fn all_property_keys(&self) -> Result<Vec<String>> {
        Ok(self
            .inner
            .read()
            .keys
            .iter()
            .map(|(_, name)| name.to_string())
            .collect())
    }

    fn has_property_index(&self, property: &str) -> bool {
        RowGroupStore::has_property_index(self, property)
    }

    fn index_serves(&self, labels: &[&str], property: &str) -> bool {
        RowGroupStore::index_serves(self, labels, property)
    }

    fn has_label_index(&self, label: &str, property: &str) -> bool {
        RowGroupStore::has_label_index(self, label, property)
    }

    fn reads_key_columns(&self) -> bool {
        true
    }

    fn key_column_group(
        &self,
        query: &mut super::KeyColumnQuery,
        from: u64,
        view: ReadView,
    ) -> Result<KeyColumnGroup> {
        let ids_given = self.next_node_id();
        let mut group = self
            .inner
            .read()
            .key_column_group(query, from, Read::view(view))?;
        group.ids_given = ids_given;
        Ok(group)
    }

    fn far_walk_back_within(
        &self,
        far_label: &str,
        types: &[&str],
        back: Direction,
        budget: &mut u64,
    ) -> Option<f64> {
        self.sample_far_walk_back(far_label, types, back, budget)
    }

    fn indexed_distinct_values(&self, labels: &[&str], property: &str) -> Option<u64> {
        RowGroupStore::indexed_distinct_values(self, labels, property)
    }

    fn edge_entries_at_most(
        &self,
        node: NodeId,
        direction: Direction,
        types: &[ArcStr],
        view: ReadView,
    ) -> Option<usize> {
        self.inner
            .read()
            .entries_of_types_at_most(node, direction, types, Read::view(view))
    }

    fn label_index_entries(&self) -> Vec<(String, String)> {
        RowGroupStore::label_index_entries(self)
    }

    fn sampled_degree(&self, label: &str, edge_type: &str, direction: Direction) -> Option<f64> {
        self.sample_degree(label, edge_type, direction)
    }

    fn sampled_max_degree(
        &self,
        label: &str,
        edge_type: &str,
        direction: Direction,
    ) -> Option<f64> {
        self.sample_max_degree(label, edge_type, direction)
    }

    fn sampled_reached_degree_within(
        &self,
        start_label: Option<&str>,
        via_types: &[&str],
        via_direction: Direction,
        on_types: &[&str],
        on_direction: Direction,
        budget: &mut u64,
    ) -> Option<f64> {
        self.sample_reached_degree(
            start_label,
            via_types,
            via_direction,
            on_types,
            on_direction,
            budget,
        )
    }

    fn find_nodes_by_property(
        &self,
        property: &str,
        value: &Value,
        view: ReadView,
    ) -> Result<Vec<NodeId>> {
        let (inner, read) = (self.inner.read(), Read::view(view));
        Ok(inner
            .indexed_equal_at(property, value, read)
            .unwrap_or_else(|| inner.nodes_matching_at(&[(property, value.clone())], read)))
    }

    fn find_nodes_by_properties(
        &self,
        conditions: &[(&str, Value)],
        view: ReadView,
    ) -> Result<Vec<NodeId>> {
        self.find_indexed_by_properties(&[], conditions, view)
    }

    fn find_indexed_by_properties(
        &self,
        labels: &[&str],
        conditions: &[(&str, Value)],
        view: ReadView,
    ) -> Result<Vec<NodeId>> {
        let (inner, read) = (self.inner.read(), Read::view(view));
        Ok(inner
            .indexed_matching_at(labels, conditions, read)
            .unwrap_or_else(|| inner.nodes_matching_at(conditions, read)))
    }

    fn find_few_nodes_maybe_equal(
        &self,
        property: &str,
        value: &Value,
        limit: usize,
        view: ReadView,
    ) -> Result<Option<Vec<NodeId>>> {
        self.find_few_indexed_maybe_equal(&[], property, value, limit, view)
    }

    fn find_few_indexed_maybe_equal(
        &self,
        labels: &[&str],
        property: &str,
        value: &Value,
        limit: usize,
        view: ReadView,
    ) -> Result<Option<Vec<NodeId>>> {
        let inner = self.inner.read();
        let few = inner.property_candidates_within(labels, property, value, limit);
        Ok(few
            .then(|| inner.maybe_equal_at(labels, property, value, Read::view(view)))
            .flatten())
    }

    fn find_nodes_maybe_equal(
        &self,
        property: &str,
        value: &Value,
        view: ReadView,
    ) -> Result<Option<Vec<NodeId>>> {
        self.find_indexed_maybe_equal(&[], property, value, view)
    }

    fn find_indexed_maybe_equal(
        &self,
        labels: &[&str],
        property: &str,
        value: &Value,
        view: ReadView,
    ) -> Result<Option<Vec<NodeId>>> {
        Ok(self
            .inner
            .read()
            .maybe_equal_at(labels, property, value, Read::view(view)))
    }

    fn index_kinds(
        &self,
        labels: &[&str],
        property: &str,
    ) -> Option<crate::execution::operators::value_utils::ValueKinds> {
        self.inner.read().index_kinds(labels, property)
    }

    fn find_nodes_in_range(
        &self,
        property: &str,
        range: &crate::graph::ValueRange,
        view: ReadView,
    ) -> Result<Vec<NodeId>> {
        Ok(self
            .inner
            .read()
            .nodes_in_range_at(property, range, Read::view(view)))
    }

    fn node_property_might_match(
        &self,
        property: &PropertyKey,
        op: CompareOp,
        value: &Value,
    ) -> bool {
        self.inner.read().might_match(true, property, op, value)
    }

    fn edge_property_might_match(
        &self,
        property: &PropertyKey,
        op: CompareOp,
        value: &Value,
    ) -> bool {
        self.inner.read().might_match(false, property, op, value)
    }

    fn statistics(&self) -> Arc<Statistics> {
        Arc::new(self.inner.read().statistics())
    }

    /// From the count the statistics are built from, without building them.
    fn label_node_estimate(&self, label: &str) -> u64 {
        let inner = self.inner.read();
        inner.label_id(label).map_or(0, |label| {
            inner
                .counts
                .labels
                .get(label as usize)
                .copied()
                .map_or(0, |count| size(count) as u64)
        })
    }

    fn node_estimate(&self) -> u64 {
        size(self.inner.read().counts.nodes) as u64
    }

    fn committed_epoch(&self) -> EpochId {
        self.current_epoch()
    }

    /// The node's versions as the store holds them, each with its own
    /// values and labels (see `history`).
    fn node_history(&self, id: NodeId) -> Result<Vec<(EpochId, Option<EpochId>, Node)>> {
        Ok(self.node_versions(id))
    }

    /// The edge's versions as the store holds them, each with its own
    /// values (see `history`).
    fn edge_history(&self, id: EdgeId) -> Result<Vec<(EpochId, Option<EpochId>, Edge)>> {
        Ok(self.edge_versions(id))
    }

    #[cfg(feature = "text-index")]
    fn has_text_index(&self, label: &str, property: &str) -> bool {
        self.inner.read().has_text_index(label, property)
    }

    #[cfg(feature = "text-index")]
    fn score_text(
        &self,
        id: NodeId,
        label: &str,
        property: &str,
        query: &str,
        view: ReadView,
    ) -> Result<Option<f64>> {
        Ok(self
            .inner
            .read()
            .score_text_at(id.as_u64(), label, property, query, Read::view(view)))
    }

    #[cfg(feature = "text-index")]
    fn text_search(
        &self,
        label: &str,
        property: &str,
        query: &str,
        k: usize,
        view: ReadView,
    ) -> Result<Vec<(NodeId, f64)>> {
        self.inner.read().text_search_at(
            label,
            property,
            query,
            TextLimit::Top(k),
            None,
            Read::view(view),
        )
    }

    #[cfg(feature = "text-index")]
    fn text_search_with_threshold(
        &self,
        label: &str,
        property: &str,
        query: &str,
        threshold: f64,
        view: ReadView,
    ) -> Result<Vec<(NodeId, f64)>> {
        self.inner.read().text_search_at(
            label,
            property,
            query,
            TextLimit::AtLeast(threshold),
            None,
            Read::view(view),
        )
    }

    #[cfg(feature = "vector-index")]
    fn has_vector_index_on(&self, property: &str) -> bool {
        self.inner.read().has_vector_index_on(property)
    }

    #[cfg(feature = "vector-index")]
    fn vector_index_config(&self, label: &str, property: &str) -> Option<HnswConfig> {
        self.inner.read().vector_index_config(label, property)
    }

    #[cfg(feature = "vector-index")]
    fn vector_search(
        &self,
        label: Option<&str>,
        property: &str,
        query: &[f32],
        k: usize,
        metric: DistanceMetric,
        view: ReadView,
    ) -> Result<Vec<(NodeId, f64)>> {
        self.inner
            .read()
            .vector_search_at(label, property, query, k, metric, Read::view(view))
    }

    #[cfg(feature = "vector-index")]
    fn vector_search_with_threshold(
        &self,
        label: Option<&str>,
        property: &str,
        query: &[f32],
        threshold: f64,
        metric: DistanceMetric,
        view: ReadView,
    ) -> Result<Vec<(NodeId, f64)>> {
        Ok(self.inner.read().vector_search_within_at(
            label,
            property,
            query,
            threshold,
            metric,
            Read::view(view),
        ))
    }
}

impl RowGroupStore {
    /// Whether the counts the store keeps answer for `view`: a committed
    /// reader at or after the store's epoch sees every commit they count,
    /// and nothing they do not. Called under the store's lock, under which
    /// the writes move the counts and the epoch together.
    fn counts_answer(&self, view: ReadView) -> bool {
        view.transaction.is_none() && view.epoch.as_u64() >= self.epoch.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use grafeo_common::types::ReadView;
    use grafeo_common::types::{ArcStr, PropertyKey, Value};

    use crate::execution::operators::value_utils::AcrossKinds;
    use crate::graph::Direction;
    use crate::graph::ValueRange;
    use crate::graph::conformance::{Tx, people};
    use crate::graph::rowgroup::RowGroupStore;
    use crate::graph::traits::GraphStore;

    /// The zone maps of hot columns prune by every value a column held: a
    /// value replaced since (a reader may still see it) still might match,
    /// one never held does not; a key without a column always might.
    #[test]
    fn hot_zone_maps_cover_every_value_a_column_held() {
        use crate::graph::lpg::CompareOp;

        let store = RowGroupStore::new();
        let age = PropertyKey::new("age");
        let alix = store.create_node_with_props(&["Person"], &[(age.clone(), Value::Int64(19))]);
        store.set_node_property(alix, "age", Value::Int64(88));
        for (op, value, might) in [
            (CompareOp::Eq, 19, true),
            (CompareOp::Eq, 88, true),
            (CompareOp::Eq, 50, true),
            (CompareOp::Eq, 89, false),
            (CompareOp::Gt, 88, false),
            (CompareOp::Ge, 88, true),
            (CompareOp::Lt, 19, false),
            (CompareOp::Le, 19, true),
        ] {
            assert_eq!(
                store.node_property_might_match(&age, op, &Value::Int64(value)),
                might,
                "{op:?} {value}"
            );
        }
        assert!(
            store.node_property_might_match(&age, CompareOp::Eq, &Value::from("Prague")),
            "a value that does not order against the column might"
        );
        assert!(
            store.node_property_might_match(
                &PropertyKey::new("city"),
                CompareOp::Eq,
                &Value::from("Prague")
            ),
            "a key without a column might, as without zone maps"
        );
        assert!(store.edge_property_might_match(&age, CompareOp::Eq, &Value::Int64(19)));
    }

    /// A decimal is found by an equal decimal of another scale, and a range
    /// of decimals compares them by value, with and without a property
    /// index (#610 on the row-group store).
    #[test]
    fn decimals_are_found_by_value_and_in_a_range() {
        use grafeo_common::types::Decimal;

        let decimal = |text: &str| Value::Decimal(Decimal::parse(text).unwrap());
        for indexed in [false, true] {
            let store = RowGroupStore::new();
            if indexed {
                store.create_property_index("price");
            }
            let mia = store
                .create_node_with_props(&["Item"], &[(PropertyKey::new("price"), decimal("1.50"))]);
            let gus = store.create_node_with_props(
                &["Item"],
                &[(PropertyKey::new("price"), decimal("197.97"))],
            );
            assert_eq!(
                store
                    .find_nodes_by_property("price", &decimal("1.5"), ReadView::latest())
                    .unwrap(),
                [mia],
                "indexed: {indexed}"
            );
            assert_eq!(
                store
                    .find_nodes_in_range(
                        "price",
                        &crate::graph::ValueRange::new(
                            Some(&decimal("197.50")),
                            None,
                            false,
                            true,
                            crate::execution::operators::value_utils::AcrossKinds::Apart
                        ),
                        ReadView::latest()
                    )
                    .unwrap(),
                [gus],
                "indexed: {indexed}"
            );
        }
    }

    /// The edges between two nodes are the ones the list holds, in its order,
    /// for each direction and set of types, through sorted lists, deltas,
    /// merges and removed edges, with and without the incoming edges kept.
    #[test]
    fn edges_between_are_the_edges_the_list_holds() {
        let types = ["KNOWS", "LIKES", "HATES"];
        for backward in [true, false] {
            let store = RowGroupStore::with_backward_edges(backward);
            let nodes: Vec<_> = (0..24).map(|_| store.create_node(&["Person"])).collect();
            // Enough edges for the adjacency to merge its delta on the way,
            // several between one pair and self-loops among them.
            let mut state = 7_u64;
            let mut next = |bound: usize| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                usize::try_from(state >> 33).unwrap() % bound
            };
            let mut edges = Vec::new();
            for _ in 0..3_000 {
                let (from, to) = (nodes[next(24)], nodes[next(24)]);
                edges.push(store.create_edge(from, to, types[next(3)]));
            }
            for edge in edges.iter().step_by(7) {
                assert!(store.delete_edge(*edge));
            }
            for _ in 0..40 {
                store.create_edge(nodes[next(24)], nodes[next(24)], types[next(3)]);
            }

            let view = ReadView::latest();
            let knows = [ArcStr::from("KNOWS")];
            let two = [ArcStr::from("HATES"), ArcStr::from("KNOWS")];
            let none = [ArcStr::from("NO_SUCH_TYPE")];
            let filters: [&[ArcStr]; 4] = [&[], &knows, &two, &none];
            let mut found = 0;
            for &node in &nodes {
                for &other in &nodes {
                    for direction in [Direction::Outgoing, Direction::Incoming, Direction::Both] {
                        for filter in filters {
                            let listed: Vec<_> = store
                                .edges(node, direction, filter, view)
                                .unwrap()
                                .into_iter()
                                .filter(|(end, _)| *end == other)
                                .map(|(_, edge)| edge)
                                .collect();
                            found += listed.len();
                            assert_eq!(
                                store
                                    .edges_between(node, other, direction, filter, view)
                                    .unwrap(),
                                listed,
                                "{node:?} {other:?} {direction:?} {filter:?} backward {backward}"
                            );
                        }
                    }
                }
            }
            assert!(found > 10_000, "only {found} edges compared");
        }
    }

    /// A read without a view reads the committed state at the store's
    /// epoch, through stamped and flat vectors alike: an open transaction's
    /// create, its value writes and its deletes stay unseen until the
    /// commit, and an edge or node it deleted is gone once it commits.
    #[test]
    fn a_read_without_a_view_sees_no_open_transactions_writes() {
        let store = RowGroupStore::new();
        let people = people(&store).unwrap();
        let (age, name, since) = (
            PropertyKey::new("age"),
            PropertyKey::new("name"),
            PropertyKey::new("since"),
        );
        let mut tx = Tx::begin(&store, 10);
        let vincent = tx
            .create_node(&["Person"], &[("name", Value::from("Vincent"))])
            .unwrap();
        tx.set_node(people.alix, "age", Value::Int64(20)).unwrap();
        tx.set_edge(people.alix_gus, "since", Value::Int64(1990))
            .unwrap();
        tx.delete_edge(people.gus_mia).unwrap();
        tx.delete_node(people.mia).unwrap();
        assert_eq!(
            store
                .node_property(vincent, name.as_str(), ReadView::latest())
                .unwrap(),
            None,
            "a create"
        );
        assert_eq!(
            store
                .node_property(people.alix, age.as_str(), ReadView::latest())
                .unwrap(),
            Some(Value::Int64(19))
        );
        assert_eq!(
            store
                .edge_property(people.alix_gus, since.as_str(), ReadView::latest())
                .unwrap(),
            Some(Value::Int64(1988))
        );
        assert_eq!(
            store
                .edge_property(people.gus_mia, since.as_str(), ReadView::latest())
                .unwrap(),
            Some(Value::Int64(319)),
            "a delete"
        );
        assert_eq!(
            store
                .node_property(people.mia, name.as_str(), ReadView::latest())
                .unwrap(),
            Some(Value::from("Mia"))
        );

        tx.commit().unwrap();
        assert_eq!(
            store
                .node_property(vincent, name.as_str(), ReadView::latest())
                .unwrap(),
            Some(Value::from("Vincent"))
        );
        assert_eq!(
            store
                .node_property(people.alix, age.as_str(), ReadView::latest())
                .unwrap(),
            Some(Value::Int64(20))
        );
        assert_eq!(
            store
                .edge_property(people.alix_gus, since.as_str(), ReadView::latest())
                .unwrap(),
            Some(Value::Int64(1990))
        );
        assert_eq!(
            store
                .edge_property(people.gus_mia, since.as_str(), ReadView::latest())
                .unwrap(),
            None
        );
        assert_eq!(
            store
                .node_property(people.mia, name.as_str(), ReadView::latest())
                .unwrap(),
            None
        );
    }

    #[test]
    fn range_bounds_include_or_exclude_their_ends() {
        let (three, nineteen, eighty_eight) =
            (Value::Int64(3), Value::Int64(19), Value::Float64(88.0));
        assert!(
            ValueRange::new(
                Some(&three),
                Some(&eighty_eight),
                false,
                false,
                AcrossKinds::Apart
            )
            .contains(&nineteen)
        );
        assert!(
            ValueRange::new(Some(&three), None, true, false, AcrossKinds::Apart).contains(&three)
        );
        assert!(
            !ValueRange::new(Some(&three), None, false, false, AcrossKinds::Apart).contains(&three)
        );
        assert!(
            !ValueRange::new(None, Some(&eighty_eight), true, false, AcrossKinds::Apart)
                .contains(&Value::Int64(88))
        );
        assert!(
            !ValueRange::new(Some(&three), None, true, true, AcrossKinds::Apart)
                .contains(&Value::from("Gus")),
            "no order between them"
        );
    }
}
