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

use std::cmp::Ordering as CmpOrdering;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use grafeo_common::change::Labels;
use grafeo_common::types::{
    ArcStr, EdgeId, EpochId, NodeId, PropertyKey, ReadView, TransactionId, Value,
};
use grafeo_common::utils::error::Result;
use grafeo_common::utils::hash::FxHashMap;

use super::adjacency::Adjacent;
use super::column::Columns;
#[cfg(feature = "text-index")]
use super::index::TextLimit;
use super::{Inner, ROWS_PER_GROUP, Read, RowGroupStore};
use crate::graph::Direction;
use crate::graph::lpg::{CompareOp, Edge, Node};
use crate::graph::traits::{GraphStore, GraphStoreSearch};
#[cfg(feature = "vector-index")]
use crate::index::vector::{DistanceMetric, HnswConfig};
use crate::statistics::{EdgeTypeStatistics, LabelStatistics, Statistics};

/// Compares two values for a range check (as `LpgStore` does).
pub(super) fn compare_for_range(a: &Value, b: &Value) -> Option<CmpOrdering> {
    match (a, b) {
        // Decimals by value, also against integers and floats, as a filter
        // compares them (#610).
        (Value::Decimal(_), _) | (_, Value::Decimal(_)) => a.compare_decimal(b),
        (Value::Int64(a), Value::Int64(b)) => Some(a.cmp(b)),
        (Value::Float64(a), Value::Float64(b)) => a.partial_cmp(b),
        // An integer and a float compare exactly, as a filter does.
        (Value::Int64(a), Value::Float64(b)) => {
            crate::execution::operators::value_utils::partial_cmp_int_float(*a, *b)
        }
        (Value::Float64(a), Value::Int64(b)) => {
            crate::execution::operators::value_utils::partial_cmp_int_float(*b, *a)
                .map(CmpOrdering::reverse)
        }
        (Value::String(a), Value::String(b)) => Some(a.cmp(b)),
        (Value::Bool(a), Value::Bool(b)) => Some(a.cmp(b)),
        (Value::Timestamp(a), Value::Timestamp(b)) => Some(a.cmp(b)),
        (Value::Date(a), Value::Date(b)) => Some(a.cmp(b)),
        (Value::Time(a), Value::Time(b)) => Some(a.cmp(b)),
        _ => a.compare_instants(b),
    }
}

/// Whether `value` lies between the bounds.
fn in_range(
    value: &Value,
    min: Option<&Value>,
    max: Option<&Value>,
    min_inclusive: bool,
    max_inclusive: bool,
) -> bool {
    let above = min.is_none_or(|min| match compare_for_range(value, min) {
        Some(CmpOrdering::Greater) => true,
        Some(CmpOrdering::Equal) => min_inclusive,
        _ => false,
    });
    let below = max.is_none_or(|max| match compare_for_range(value, max) {
        Some(CmpOrdering::Less) => true,
        Some(CmpOrdering::Equal) => max_inclusive,
        _ => false,
    });
    above && below
}

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
    fn edge_value_at(&self, id: u64, key: &str, read: Read) -> Option<Value> {
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

    /// The edges of `node` in `direction` that `read` sees.
    fn adjacency_at(
        &self,
        node: NodeId,
        direction: Direction,
        read: Read,
    ) -> Vec<(NodeId, EdgeId)> {
        let mut edges = self.adjacency(node, direction);
        edges.retain(|(_, edge)| self.edge_visible(edge.as_u64(), read));
        edges
    }

    /// The nodes `read` sees with at least one of the label ids, ascending
    /// and each once: the label bitmaps give the newest labels, and the rows
    /// a label chain holds are read at the view instead. A row group without
    /// label chains reads its bitmaps alone.
    fn nodes_with_labels_at(&self, label_ids: &[u32], read: Read) -> Vec<NodeId> {
        let mut ids = Vec::new();
        for (index, group) in &self.nodes {
            let newest = label_ids
                .iter()
                .filter_map(|label| group.labels.get(label))
                .flat_map(|rows| group.versions.existing(rows));
            let changed = group.label_updates.rows();
            let mut rows: Vec<usize> = match &changed {
                None => newest
                    .filter(|row| group.version(*row).visible(read))
                    .collect(),
                Some(changed) => {
                    let mut rows: Vec<usize> = newest
                        .filter(|row| !changed.get(*row) && group.version(*row).visible(read))
                        .collect();
                    rows.extend(changed.rows().filter(|row| {
                        group.version(*row).visible(read)
                            && label_ids
                                .iter()
                                .any(|label| group.has_label_at(*row, *label, read))
                    }));
                    rows
                }
            };
            if label_ids.len() > 1 || changed.is_some() {
                rows.sort_unstable();
                rows.dedup();
            }
            ids.extend(
                rows.into_iter()
                    .map(|row| NodeId::new(index * ROWS_PER_GROUP + row as u64)),
            );
        }
        ids
    }

    /// The nodes with the label id in any version, ascending: the newest
    /// labels and what the label chains kept.
    fn nodes_with_label_in_any_version(&self, label: u32) -> Vec<NodeId> {
        let mut ids = Vec::new();
        for (index, group) in &self.nodes {
            let mut rows: Vec<usize> = group
                .labels
                .get(&label)
                .map(|rows| group.versions.existing(rows).collect())
                .unwrap_or_default();
            if let Some(changed) = group.label_updates.rows() {
                rows.extend(changed.rows().filter(|row| {
                    group.versions.exists(*row)
                        && group
                            .label_updates
                            .kept(*row)
                            .any(|kept| kept.binary_search(&label).is_ok())
                }));
                rows.sort_unstable();
                rows.dedup();
            }
            ids.extend(
                rows.into_iter()
                    .map(|row| NodeId::new(index * ROWS_PER_GROUP + row as u64)),
            );
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

    /// A node's entries in `direction`, of one edge type id or of all, as
    /// (other node, edge), outgoing ones first: each kept direction's sorted
    /// list, then its delta; without the incoming edges kept, the incoming
    /// ones from the edge table, by edge id.
    fn entries_in(
        &self,
        node: NodeId,
        direction: Direction,
        edge_type: Option<u32>,
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
        let mut entries = Vec::new();
        for &outgoing in directions {
            if outgoing || self.keeps_incoming() {
                let adjacency = group.adjacency(outgoing);
                match edge_type {
                    None => entries.extend(adjacency.of(row).map(pair)),
                    Some(edge_type) => entries.extend(adjacency.of_type(row, edge_type).map(pair)),
                }
            } else {
                entries.extend(
                    self.incoming_by_scan(node.as_u64())
                        .filter(|adjacent| {
                            edge_type.is_none_or(|wanted| adjacent.edge_type == wanted)
                        })
                        .map(pair),
                );
            }
        }
        entries
    }

    /// A node's adjacency entries, as (other node, edge): each direction's
    /// sorted list, then its delta (see [`entries`](Self::entries)).
    pub(super) fn adjacency(&self, node: NodeId, direction: Direction) -> Vec<(NodeId, EdgeId)> {
        self.entries_in(node, direction, None)
    }

    /// A node's adjacency entries of one edge type, as (other node, edge).
    fn adjacency_of_type(
        &self,
        node: NodeId,
        direction: Direction,
        edge_type: &str,
    ) -> Vec<(NodeId, EdgeId)> {
        match self.edge_types.get_id(edge_type) {
            Some(edge_type) => self.entries_in(node, direction, Some(edge_type)),
            None => Vec::new(),
        }
    }

    /// How many of a node's edges in one direction `read` sees.
    fn degree(&self, node: NodeId, outgoing: bool, read: Read) -> usize {
        let seen = |adjacent: &Adjacent| self.edge_visible(adjacent.edge, read);
        self.node(node.as_u64()).map_or(0, |(group, row)| {
            if outgoing || self.keeps_incoming() {
                group.adjacency(outgoing).of(row).filter(seen).count()
            } else {
                self.incoming_by_scan(node.as_u64()).filter(seen).count()
            }
        })
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

impl GraphStore for RowGroupStore {
    fn get_node(&self, id: NodeId) -> Option<Node> {
        self.inner.read().read_node(id.as_u64(), self.now())
    }

    fn get_edge(&self, id: EdgeId) -> Option<Edge> {
        self.inner.read().read_edge(id.as_u64(), self.now())
    }

    fn get_node_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Node> {
        self.inner
            .read()
            .read_node(id.as_u64(), Read::transaction(transaction_id, epoch))
    }

    fn get_edge_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Edge> {
        self.inner
            .read()
            .read_edge(id.as_u64(), Read::transaction(transaction_id, epoch))
    }

    fn get_node_at_epoch(&self, id: NodeId, epoch: EpochId) -> Option<Node> {
        self.inner.read().read_node(id.as_u64(), Read::at(epoch))
    }

    fn get_edge_at_epoch(&self, id: EdgeId, epoch: EpochId) -> Option<Edge> {
        self.inner.read().read_edge(id.as_u64(), Read::at(epoch))
    }

    /// The node's versions as the store holds them, each with its own
    /// values and labels (see `history`).
    fn get_node_history(&self, id: NodeId) -> Vec<(EpochId, Option<EpochId>, Node)> {
        self.node_history(id)
    }

    /// The edge's versions as the store holds them, each with its own
    /// values (see `history`).
    fn get_edge_history(&self, id: EdgeId) -> Vec<(EpochId, Option<EpochId>, Edge)> {
        self.edge_history(id)
    }

    fn get_node_property(&self, id: NodeId, key: &PropertyKey) -> Option<Value> {
        self.inner
            .read()
            .node_value_at(id.as_u64(), key.as_str(), self.now())
    }

    fn get_edge_property(&self, id: EdgeId, key: &PropertyKey) -> Option<Value> {
        self.inner
            .read()
            .edge_value_at(id.as_u64(), key.as_str(), self.now())
    }

    fn get_node_property_batch(&self, ids: &[NodeId], key: &PropertyKey) -> Vec<Option<Value>> {
        let (inner, now) = (self.inner.read(), self.now());
        ids.iter()
            .map(|id| inner.node_value_at(id.as_u64(), key.as_str(), now))
            .collect()
    }

    fn try_get_node_property_batch(
        &self,
        ids: &[NodeId],
        key: &PropertyKey,
    ) -> Result<Vec<Option<Value>>> {
        Ok(self.get_node_property_batch(ids, key))
    }

    fn get_nodes_properties_batch(&self, ids: &[NodeId]) -> Vec<FxHashMap<PropertyKey, Value>> {
        self.get_nodes_properties_batch_at(ids, self.now_view())
    }

    fn get_nodes_properties_selective_batch(
        &self,
        ids: &[NodeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        self.get_nodes_properties_selective_batch_at(ids, keys, self.now_view())
    }

    fn get_edges_properties_selective_batch(
        &self,
        ids: &[EdgeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        self.get_edges_properties_selective_batch_at(ids, keys, self.now_view())
    }

    fn neighbors(&self, node: NodeId, direction: Direction) -> Vec<NodeId> {
        self.inner
            .read()
            .adjacency(node, direction)
            .into_iter()
            .map(|(other, _)| other)
            .collect()
    }

    fn edges_from(&self, node: NodeId, direction: Direction) -> Vec<(NodeId, EdgeId)> {
        self.inner.read().adjacency(node, direction)
    }

    fn out_degree(&self, node: NodeId) -> usize {
        self.inner.read().degree(node, true, self.now())
    }

    fn in_degree(&self, node: NodeId) -> usize {
        self.inner.read().degree(node, false, self.now())
    }

    /// Whether it keeps each node's incoming edges: without them an
    /// incoming read scans the edge table (see
    /// [`RowGroupStore::with_backward_edges`]).
    fn has_backward_adjacency(&self) -> bool {
        self.keeps_backward_edges()
    }

    fn node_ids(&self) -> Vec<NodeId> {
        let now = self.now();
        self.inner
            .read()
            .node_rows(|_, group, row| group.version(row).visible(now))
    }

    fn all_node_ids(&self) -> Vec<NodeId> {
        self.inner.read().node_rows(|_, _, _| true)
    }

    /// The nodes with `label` in any version a reader may see, open
    /// transactions' included: a reader filters them at its view (see the
    /// module docs).
    fn nodes_by_label(&self, label: &str) -> Vec<NodeId> {
        let inner = self.inner.read();
        inner.label_id(label).map_or_else(Vec::new, |label| {
            inner.nodes_with_label_in_any_version(label)
        })
    }

    fn nodes_by_label_count(&self, label: &str) -> usize {
        let inner = self.inner.read();
        inner
            .label_id(label)
            .and_then(|id| inner.counts.labels.get(id as usize).copied())
            .map_or(0, size)
    }

    fn nodes_by_label_count_at_epoch(&self, label: &str, epoch: EpochId) -> usize {
        self.nodes_by_label_at(label, ReadView::committed(epoch))
            .len()
    }

    fn node_count(&self) -> usize {
        size(self.inner.read().counts.nodes)
    }

    fn edge_count(&self) -> usize {
        size(self.inner.read().counts.edges)
    }

    fn edge_type(&self, id: EdgeId) -> Option<ArcStr> {
        let inner = self.inner.read();
        let (group, row) = inner.edge(id.as_u64())?;
        let (_, _, edge_type) = group.ends(row);
        inner.edge_types.get_name(edge_type).cloned()
    }

    fn has_property_index(&self, property: &str) -> bool {
        RowGroupStore::has_property_index(self, property)
    }

    fn find_nodes_by_property(&self, property: &str, value: &Value) -> Vec<NodeId> {
        self.find_nodes_by_property_at(property, value, self.now_view())
    }

    fn find_nodes_by_properties(&self, conditions: &[(&str, Value)]) -> Vec<NodeId> {
        self.find_nodes_by_properties_at(conditions, self.now_view())
    }

    fn find_nodes_maybe_equal(&self, property: &str, value: &Value) -> Option<Vec<NodeId>> {
        RowGroupStore::find_nodes_maybe_equal(self, property, value)
    }

    fn find_nodes_in_range(
        &self,
        property: &str,
        min: Option<&Value>,
        max: Option<&Value>,
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> Vec<NodeId> {
        self.inner.read().nodes_in_range_at(
            property,
            min,
            max,
            min_inclusive,
            max_inclusive,
            self.now(),
        )
    }

    /// By the zone maps of the hot columns (every value they held, so every
    /// version a reader may see), and always for a column still cold.
    fn node_property_might_match(
        &self,
        property: &PropertyKey,
        op: CompareOp,
        value: &Value,
    ) -> bool {
        let inner = self.inner.read();
        // A key no row group has a column of might match, as for a store
        // without zone maps (the planner keeps the filter).
        let Some(key) = inner.keys.get_id(property.as_str()) else {
            return true;
        };
        let mut held = false;
        for group in inner.nodes.values() {
            match group.columns.might_match(key, op, value) {
                Some(true) => return true,
                Some(false) => held = true,
                None => {}
            }
        }
        !held
    }

    /// As [`node_property_might_match`](Self::node_property_might_match).
    fn edge_property_might_match(
        &self,
        property: &PropertyKey,
        op: CompareOp,
        value: &Value,
    ) -> bool {
        let inner = self.inner.read();
        // A key no row group has a column of might match, as for a store
        // without zone maps (the planner keeps the filter).
        let Some(key) = inner.keys.get_id(property.as_str()) else {
            return true;
        };
        let mut held = false;
        for group in inner.edges.values() {
            match group.columns.might_match(key, op, value) {
                Some(true) => return true,
                Some(false) => held = true,
                None => {}
            }
        }
        !held
    }

    fn statistics(&self) -> Arc<Statistics> {
        Arc::new(self.inner.read().statistics())
    }

    fn estimate_label_cardinality(&self, label: &str) -> f64 {
        self.inner
            .read()
            .statistics()
            .estimate_label_cardinality(label)
    }

    fn estimate_avg_degree(&self, edge_type: &str, outgoing: bool) -> f64 {
        self.inner
            .read()
            .statistics()
            .estimate_avg_degree(edge_type, outgoing)
    }

    fn current_epoch(&self) -> EpochId {
        EpochId::new(self.epoch.load(Ordering::Acquire))
    }

    fn all_labels(&self) -> Vec<String> {
        self.inner
            .read()
            .labels
            .iter()
            .map(|(_, name)| name.to_string())
            .collect()
    }

    fn all_edge_types(&self) -> Vec<String> {
        self.inner
            .read()
            .edge_types
            .iter()
            .map(|(_, name)| name.to_string())
            .collect()
    }

    fn all_property_keys(&self) -> Vec<String> {
        self.inner
            .read()
            .keys
            .iter()
            .map(|(_, name)| name.to_string())
            .collect()
    }

    fn is_node_visible_at_epoch(&self, id: NodeId, epoch: EpochId) -> bool {
        self.inner.read().node_visible(id.as_u64(), Read::at(epoch))
    }

    fn is_node_visible_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        self.inner
            .read()
            .node_visible(id.as_u64(), Read::transaction(transaction_id, epoch))
    }

    fn is_edge_visible_at_epoch(&self, id: EdgeId, epoch: EpochId) -> bool {
        self.inner.read().edge_visible(id.as_u64(), Read::at(epoch))
    }

    fn is_edge_visible_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        self.inner
            .read()
            .edge_visible(id.as_u64(), Read::transaction(transaction_id, epoch))
    }

    fn node_property_versioned(
        &self,
        id: NodeId,
        key: &str,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Value> {
        self.inner
            .read()
            .node_value_at(id.as_u64(), key, Read::transaction(transaction_id, epoch))
    }

    fn node_property_at_epoch(&self, id: NodeId, key: &str, epoch: EpochId) -> Option<Value> {
        self.inner
            .read()
            .node_value_at(id.as_u64(), key, Read::at(epoch))
    }

    fn edge_property_versioned(
        &self,
        id: EdgeId,
        key: &str,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Value> {
        self.inner
            .read()
            .edge_value_at(id.as_u64(), key, Read::transaction(transaction_id, epoch))
    }

    fn edge_property_at_epoch(&self, id: EdgeId, key: &str, epoch: EpochId) -> Option<Value> {
        self.inner
            .read()
            .edge_value_at(id.as_u64(), key, Read::at(epoch))
    }

    fn filter_visible_node_ids(&self, ids: &[NodeId], epoch: EpochId) -> Vec<NodeId> {
        self.filter_visible_node_ids_at(ids, ReadView::committed(epoch))
    }

    fn filter_visible_node_ids_versioned(
        &self,
        ids: &[NodeId],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Vec<NodeId> {
        self.filter_visible_node_ids_at(ids, ReadView::transaction(transaction_id, epoch))
    }

    // Reads at a view: existence from the rows' versions, values and labels
    // from the columns and bitmaps and their update chains.

    fn answers_at_view(&self) -> bool {
        true
    }

    fn is_node_visible_at(&self, id: NodeId, view: ReadView) -> bool {
        self.inner
            .read()
            .node_visible(id.as_u64(), Read::view(view))
    }

    fn is_edge_visible_at(&self, id: EdgeId, view: ReadView) -> bool {
        self.inner
            .read()
            .edge_visible(id.as_u64(), Read::view(view))
    }

    fn filter_visible_node_ids_at(&self, ids: &[NodeId], view: ReadView) -> Vec<NodeId> {
        let (inner, read) = (self.inner.read(), Read::view(view));
        ids.iter()
            .copied()
            .filter(|id| inner.node_visible(id.as_u64(), read))
            .collect()
    }

    fn get_node_at(&self, id: NodeId, view: ReadView) -> Option<Node> {
        self.inner.read().read_node(id.as_u64(), Read::view(view))
    }

    fn get_edge_at(&self, id: EdgeId, view: ReadView) -> Option<Edge> {
        self.inner.read().read_edge(id.as_u64(), Read::view(view))
    }

    fn node_property_at(&self, id: NodeId, key: &str, view: ReadView) -> Option<Value> {
        self.inner
            .read()
            .node_value_at(id.as_u64(), key, Read::view(view))
    }

    fn edge_property_at(&self, id: EdgeId, key: &str, view: ReadView) -> Option<Value> {
        self.inner
            .read()
            .edge_value_at(id.as_u64(), key, Read::view(view))
    }

    fn node_labels_at(&self, id: NodeId, view: ReadView) -> Option<Labels> {
        let (inner, read) = (self.inner.read(), Read::view(view));
        let (group, row) = inner.node(id.as_u64())?;
        if !group.version(row).visible(read) {
            return None;
        }
        Some(inner.label_names(&group.labels_at(row, read)))
    }

    fn node_has_label_at(&self, id: NodeId, label: &str, view: ReadView) -> Option<bool> {
        let (inner, read) = (self.inner.read(), Read::view(view));
        let (group, row) = inner.node(id.as_u64())?;
        if !group.version(row).visible(read) {
            return None;
        }
        Some(
            inner
                .label_id(label)
                .is_some_and(|label| group.has_label_at(row, label, read)),
        )
    }

    fn get_node_property_batch_at(
        &self,
        ids: &[NodeId],
        key: &PropertyKey,
        view: ReadView,
    ) -> Vec<Option<Value>> {
        let (inner, read) = (self.inner.read(), Read::view(view));
        ids.iter()
            .map(|id| inner.node_value_at(id.as_u64(), key.as_str(), read))
            .collect()
    }

    fn get_nodes_properties_batch_at(
        &self,
        ids: &[NodeId],
        view: ReadView,
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        let (inner, read) = (self.inner.read(), Read::view(view));
        ids.iter()
            .map(|id| inner.node_values_at(*id, None, read))
            .collect()
    }

    fn get_nodes_properties_selective_batch_at(
        &self,
        ids: &[NodeId],
        keys: &[PropertyKey],
        view: ReadView,
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        let (inner, read) = (self.inner.read(), Read::view(view));
        ids.iter()
            .map(|id| inner.node_values_at(*id, Some(keys), read))
            .collect()
    }

    fn get_edges_properties_selective_batch_at(
        &self,
        ids: &[EdgeId],
        keys: &[PropertyKey],
        view: ReadView,
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        let (inner, read) = (self.inner.read(), Read::view(view));
        ids.iter()
            .map(|id| inner.edge_values_at(*id, keys, read))
            .collect()
    }

    fn edges_from_at(
        &self,
        node: NodeId,
        direction: Direction,
        view: ReadView,
    ) -> Vec<(NodeId, EdgeId)> {
        self.inner
            .read()
            .adjacency_at(node, direction, Read::view(view))
    }

    fn neighbors_at(&self, node: NodeId, direction: Direction, view: ReadView) -> Vec<NodeId> {
        self.inner
            .read()
            .adjacency_at(node, direction, Read::view(view))
            .into_iter()
            .map(|(other, _)| other)
            .collect()
    }

    fn nodes_by_label_at(&self, label: &str, view: ReadView) -> Vec<NodeId> {
        let inner = self.inner.read();
        inner.label_id(label).map_or_else(Vec::new, |label| {
            inner.nodes_with_labels_at(&[label], Read::view(view))
        })
    }

    fn nodes_with_any_label_at(&self, labels: &[&str], view: ReadView) -> Vec<NodeId> {
        let inner = self.inner.read();
        let mut label_ids: Vec<u32> = labels
            .iter()
            .filter_map(|label| inner.label_id(label))
            .collect();
        label_ids.sort_unstable();
        label_ids.dedup();
        inner.nodes_with_labels_at(&label_ids, Read::view(view))
    }

    // Lookups through a property index take its entries, of every version,
    // as candidates and read each one's value at the view (`index`).

    fn find_nodes_by_property_at(
        &self,
        property: &str,
        value: &Value,
        view: ReadView,
    ) -> Vec<NodeId> {
        let (inner, read) = (self.inner.read(), Read::view(view));
        inner
            .indexed_equal_at(property, value, read)
            .unwrap_or_else(|| inner.nodes_matching_at(&[(property, value.clone())], read))
    }

    fn find_nodes_by_properties_at(
        &self,
        conditions: &[(&str, Value)],
        view: ReadView,
    ) -> Vec<NodeId> {
        let (inner, read) = (self.inner.read(), Read::view(view));
        inner
            .indexed_matching_at(conditions, read)
            .unwrap_or_else(|| inner.nodes_matching_at(conditions, read))
    }

    fn find_nodes_maybe_equal_at(
        &self,
        property: &str,
        value: &Value,
        view: ReadView,
    ) -> Option<Vec<NodeId>> {
        self.inner
            .read()
            .maybe_equal_at(property, value, Read::view(view))
    }

    fn find_nodes_in_range_at(
        &self,
        property: &str,
        min: Option<&Value>,
        max: Option<&Value>,
        min_inclusive: bool,
        max_inclusive: bool,
        view: ReadView,
    ) -> Vec<NodeId> {
        self.inner.read().nodes_in_range_at(
            property,
            min,
            max,
            min_inclusive,
            max_inclusive,
            Read::view(view),
        )
    }
}

/// The searches through the text and vector indexes at a view (`index`); the
/// searches without a view search at the store's epoch.
impl GraphStoreSearch for RowGroupStore {
    #[cfg(feature = "text-index")]
    fn has_text_index(&self, label: &str, property: &str) -> bool {
        self.inner.read().has_text_index(label, property)
    }

    #[cfg(feature = "text-index")]
    fn score_text(&self, node_id: NodeId, label: &str, property: &str, query: &str) -> Option<f64> {
        self.score_text_at(node_id, label, property, query, self.now_view())
    }

    #[cfg(feature = "text-index")]
    fn score_text_at(
        &self,
        node_id: NodeId,
        label: &str,
        property: &str,
        query: &str,
        view: ReadView,
    ) -> Option<f64> {
        self.inner
            .read()
            .score_text_at(node_id.as_u64(), label, property, query, Read::view(view))
    }

    #[cfg(feature = "text-index")]
    fn text_search(
        &self,
        label: &str,
        property: &str,
        query: &str,
        k: usize,
    ) -> Vec<(NodeId, f64)> {
        self.text_search_at(label, property, query, k, self.now_view())
    }

    #[cfg(feature = "text-index")]
    fn text_search_with_threshold(
        &self,
        label: &str,
        property: &str,
        query: &str,
        threshold: f64,
    ) -> Vec<(NodeId, f64)> {
        self.text_search_with_threshold_at(label, property, query, threshold, self.now_view())
    }

    #[cfg(feature = "text-index")]
    fn text_search_at(
        &self,
        label: &str,
        property: &str,
        query: &str,
        k: usize,
        view: ReadView,
    ) -> Vec<(NodeId, f64)> {
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
    fn text_search_with_threshold_at(
        &self,
        label: &str,
        property: &str,
        query: &str,
        threshold: f64,
        view: ReadView,
    ) -> Vec<(NodeId, f64)> {
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
    fn has_vector_index(&self, label: &str, property: &str) -> bool {
        self.vector_index_config(label, property).is_some()
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
    ) -> Vec<(NodeId, f64)> {
        self.vector_search_at(label, property, query, k, metric, self.now_view())
    }

    #[cfg(feature = "vector-index")]
    fn vector_search_with_threshold(
        &self,
        label: Option<&str>,
        property: &str,
        query: &[f32],
        threshold: f64,
        metric: DistanceMetric,
    ) -> Vec<(NodeId, f64)> {
        self.vector_search_with_threshold_at(
            label,
            property,
            query,
            threshold,
            metric,
            self.now_view(),
        )
    }

    #[cfg(feature = "vector-index")]
    fn vector_search_at(
        &self,
        label: Option<&str>,
        property: &str,
        query: &[f32],
        k: usize,
        metric: DistanceMetric,
        view: ReadView,
    ) -> Vec<(NodeId, f64)> {
        self.inner
            .read()
            .vector_search_at(label, property, query, k, metric, Read::view(view))
    }

    #[cfg(feature = "vector-index")]
    fn vector_search_with_threshold_at(
        &self,
        label: Option<&str>,
        property: &str,
        query: &[f32],
        threshold: f64,
        metric: DistanceMetric,
        view: ReadView,
    ) -> Vec<(NodeId, f64)> {
        self.inner.read().vector_search_within_at(
            label,
            property,
            query,
            threshold,
            metric,
            Read::view(view),
        )
    }
}

impl RowGroupStore {
    /// A node's edges of one type, as (other node, edge): one slice of each
    /// direction's sorted list, then its delta. As `edges_from`, it lists
    /// deleted edges too; a reader checks each edge's visibility.
    #[must_use]
    pub fn edges_of_type(
        &self,
        node: NodeId,
        direction: Direction,
        edge_type: &str,
    ) -> Vec<(NodeId, EdgeId)> {
        self.inner
            .read()
            .adjacency_of_type(node, direction, edge_type)
    }
}

#[cfg(test)]
mod tests {
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
        use crate::graph::GraphStoreMut;
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
        use crate::graph::GraphStoreMut;
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
                store.find_nodes_by_property("price", &decimal("1.5")),
                [mia],
                "indexed: {indexed}"
            );
            assert_eq!(
                store.find_nodes_in_range("price", Some(&decimal("197.50")), None, false, true),
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
        assert_eq!(store.get_node_property(vincent, &name), None, "a create");
        assert_eq!(
            store.get_node_property(people.alix, &age),
            Some(Value::Int64(19))
        );
        assert_eq!(
            store.get_edge_property(people.alix_gus, &since),
            Some(Value::Int64(1988))
        );
        assert_eq!(
            store.get_edge_property(people.gus_mia, &since),
            Some(Value::Int64(319)),
            "a delete"
        );
        assert_eq!(
            store.get_node_property(people.mia, &name),
            Some(Value::from("Mia"))
        );

        tx.commit().unwrap();
        assert_eq!(
            store.get_node_property(vincent, &name),
            Some(Value::from("Vincent"))
        );
        assert_eq!(
            store.get_node_property(people.alix, &age),
            Some(Value::Int64(20))
        );
        assert_eq!(
            store.get_edge_property(people.alix_gus, &since),
            Some(Value::Int64(1990))
        );
        assert_eq!(store.get_edge_property(people.gus_mia, &since), None);
        assert_eq!(store.get_node_property(people.mia, &name), None);
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
