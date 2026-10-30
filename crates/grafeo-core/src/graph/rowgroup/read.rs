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
use super::column::Columns;
#[cfg(feature = "text-index")]
use super::index::TextLimit;
use super::{Inner, ROWS_PER_GROUP, Read, RowGroupStore};
use crate::graph::Direction;
use crate::graph::lpg::{CompareOp, Edge, Node};
use crate::graph::traits::GraphStore;
#[cfg(feature = "vector-index")]
use crate::index::vector::{DistanceMetric, HnswConfig};
use crate::statistics::{EdgeTypeStatistics, LabelStatistics, Statistics};

pub(super) use crate::graph::traits::compare_for_range;
use crate::graph::traits::value_in_range as in_range;

/// A count kept as a signed total, as a size.
fn size(count: i64) -> usize {
    usize::try_from(count.max(0)).unwrap_or(usize::MAX)
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
        min: Option<&Value>,
        max: Option<&Value>,
        min_inclusive: bool,
        max_inclusive: bool,
        read: Read,
    ) -> Vec<NodeId> {
        self.node_rows(|_, group, row| {
            group.version(row).visible(read)
                && self
                    .value_at(&group.columns, row, property, read)
                    .is_some_and(|value| in_range(&value, min, max, min_inclusive, max_inclusive))
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
                match edge_types {
                    Some(&[edge_type]) => adjacency
                        .of_type(row, edge_type)
                        .any(|adjacent| counts(adjacent, outgoing)),
                    _ => adjacency.of(row).any(|adjacent| counts(adjacent, outgoing)),
                }
            } else {
                self.incoming_by_scan(raw)
                    .any(|adjacent| counts(adjacent, outgoing))
            }
        })
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
            for (row, &dst) in group.dst.iter().enumerate() {
                if dst == raw && group.versions.visible(row, read) {
                    seen(NodeId::new(group.src[row]), EdgeId::new(base + row as u64));
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
        Ok(ids
            .iter()
            .map(|id| inner.node_value_at(id.as_u64(), key.as_str(), read))
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
        let (inner, read) = (self.inner.read(), Read::view(view));
        Ok(inner
            .indexed_matching_at(conditions, read)
            .unwrap_or_else(|| inner.nodes_matching_at(conditions, read)))
    }

    fn find_nodes_maybe_equal(
        &self,
        property: &str,
        value: &Value,
        view: ReadView,
    ) -> Result<Option<Vec<NodeId>>> {
        Ok(self
            .inner
            .read()
            .maybe_equal_at(property, value, Read::view(view)))
    }

    fn find_nodes_in_range(
        &self,
        property: &str,
        min: Option<&Value>,
        max: Option<&Value>,
        min_inclusive: bool,
        max_inclusive: bool,
        view: ReadView,
    ) -> Result<Vec<NodeId>> {
        Ok(self.inner.read().nodes_in_range_at(
            property,
            min,
            max,
            min_inclusive,
            max_inclusive,
            Read::view(view),
        ))
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
        Ok(self.inner.read().text_search_at(
            label,
            property,
            query,
            TextLimit::Top(k),
            None,
            Read::view(view),
        ))
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
        Ok(self.inner.read().text_search_at(
            label,
            property,
            query,
            TextLimit::AtLeast(threshold),
            None,
            Read::view(view),
        ))
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
        Ok(self
            .inner
            .read()
            .vector_search_at(label, property, query, k, metric, Read::view(view)))
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
    use grafeo_common::types::{PropertyKey, Value};

    use super::in_range;
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
                        Some(&decimal("197.50")),
                        None,
                        false,
                        true,
                        ReadView::latest()
                    )
                    .unwrap(),
                [gus],
                "indexed: {indexed}"
            );
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
        assert!(in_range(
            &nineteen,
            Some(&three),
            Some(&eighty_eight),
            false,
            false
        ));
        assert!(in_range(&three, Some(&three), None, true, false));
        assert!(!in_range(&three, Some(&three), None, false, false));
        assert!(!in_range(
            &Value::Int64(88),
            None,
            Some(&eighty_eight),
            true,
            false
        ));
        assert!(
            !in_range(&Value::from("Gus"), Some(&three), None, true, true),
            "no order between them"
        );
    }
}
