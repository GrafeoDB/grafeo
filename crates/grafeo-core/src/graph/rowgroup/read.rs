//! The row-group store's read traits.
//!
//! The reads at a snapshot (`*_versioned`, `*_at_epoch`, the visibility
//! checks) read each row's version. The reads without one read the
//! committed state now, but for values, labels and adjacency, which are
//! written in place (H1a): they also list what open transactions wrote, and
//! adjacency lists every edge not yet collected, deleted ones included, for
//! readers at earlier epochs; a reader checks each edge's visibility.

use std::cmp::Ordering as CmpOrdering;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use grafeo_common::types::{ArcStr, EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};
use grafeo_common::utils::error::Result;
use grafeo_common::utils::hash::FxHashMap;

use super::column::Columns;
use super::{ABSENT, Inner, ROWS_PER_GROUP, Read, RowGroupStore};
use crate::graph::Direction;
use crate::graph::lpg::{CompareOp, Edge, Node};
use crate::graph::traits::{GraphStore, GraphStoreSearch};
use crate::statistics::{EdgeTypeStatistics, LabelStatistics, Statistics};

/// Compares two values for a range check (as `LpgStore` does).
fn compare_for_range(a: &Value, b: &Value) -> Option<CmpOrdering> {
    match (a, b) {
        (Value::Int64(a), Value::Int64(b)) => Some(a.cmp(b)),
        (Value::Float64(a), Value::Float64(b)) => a.partial_cmp(b),
        #[expect(
            clippy::cast_precision_loss,
            reason = "a range check compares an integer with a float as a float, as a filter does"
        )]
        (Value::Int64(a), Value::Float64(b)) => (*a as f64).partial_cmp(b),
        #[expect(
            clippy::cast_precision_loss,
            reason = "a range check compares an integer with a float as a float, as a filter does"
        )]
        (Value::Float64(a), Value::Int64(b)) => a.partial_cmp(&(*b as f64)),
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
    /// The value of `key` at a row of `columns`.
    fn value_at(&self, columns: &Columns, row: usize, key: &str) -> Option<Value> {
        columns.get(self.keys.get_id(key)?, row)
    }

    /// A row's values for `keys` (all of them for `None`), unless the row is
    /// deleted.
    fn values_of(
        &self,
        columns: &Columns,
        row: usize,
        keys: Option<&[PropertyKey]>,
    ) -> FxHashMap<PropertyKey, Value> {
        match keys {
            None => self.values(columns, row).into_iter().collect(),
            Some(keys) => keys
                .iter()
                .filter_map(|key| Some((key.clone(), self.value_at(columns, row, key.as_str())?)))
                .collect(),
        }
    }

    /// A node's value now: `None` for a deleted node.
    fn node_value(&self, id: NodeId, key: &PropertyKey) -> Option<Value> {
        let (group, row) = self.node(id.as_u64())?;
        if group.version(row).deleted != ABSENT {
            return None;
        }
        self.value_at(&group.columns, row, key.as_str())
    }

    fn node_values(
        &self,
        id: NodeId,
        keys: Option<&[PropertyKey]>,
    ) -> FxHashMap<PropertyKey, Value> {
        match self.node(id.as_u64()) {
            Some((group, row)) if group.version(row).deleted == ABSENT => {
                self.values_of(&group.columns, row, keys)
            }
            _ => FxHashMap::default(),
        }
    }

    /// Every node row, ascending, that `keep` accepts.
    fn node_rows(
        &self,
        mut keep: impl FnMut(u64, &super::NodeGroup, usize) -> bool,
    ) -> Vec<NodeId> {
        let mut ids = Vec::new();
        for (index, group) in &self.nodes {
            for (row, version) in group.versions.iter().enumerate() {
                let id = index * ROWS_PER_GROUP + row as u64;
                if version.exists() && keep(id, group, row) {
                    ids.push(NodeId::new(id));
                }
            }
        }
        ids
    }

    /// A node's adjacency entries, as (other node, edge): each direction's
    /// sorted list, then its delta.
    pub(super) fn adjacency(&self, node: NodeId, direction: Direction) -> Vec<(NodeId, EdgeId)> {
        let Some((group, row)) = self.node(node.as_u64()) else {
            return Vec::new();
        };
        let directions: &[bool] = match direction {
            Direction::Outgoing => &[true],
            Direction::Incoming => &[false],
            Direction::Both => &[true, false],
        };
        directions
            .iter()
            .flat_map(|outgoing| group.adjacency(*outgoing).of(row))
            .map(|adjacent| (NodeId::new(adjacent.other), EdgeId::new(adjacent.edge)))
            .collect()
    }

    /// A node's adjacency entries of one edge type, as (other node, edge).
    fn adjacency_of_type(
        &self,
        node: NodeId,
        direction: Direction,
        edge_type: &str,
    ) -> Vec<(NodeId, EdgeId)> {
        let (Some((group, row)), Some(edge_type)) =
            (self.node(node.as_u64()), self.edge_types.get_id(edge_type))
        else {
            return Vec::new();
        };
        let directions: &[bool] = match direction {
            Direction::Outgoing => &[true],
            Direction::Incoming => &[false],
            Direction::Both => &[true, false],
        };
        directions
            .iter()
            .flat_map(|outgoing| group.adjacency(*outgoing).of_type(row, edge_type))
            .map(|adjacent| (NodeId::new(adjacent.other), EdgeId::new(adjacent.edge)))
            .collect()
    }

    /// How many of a node's edges in one direction `read` sees.
    fn degree(&self, node: NodeId, outgoing: bool, read: Read) -> usize {
        self.node(node.as_u64()).map_or(0, |(group, row)| {
            group
                .adjacency(outgoing)
                .of(row)
                .filter(|adjacent| self.edge_visible(adjacent.edge, read))
                .count()
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

    fn get_node_property(&self, id: NodeId, key: &PropertyKey) -> Option<Value> {
        self.inner.read().node_value(id, key)
    }

    fn get_edge_property(&self, id: EdgeId, key: &PropertyKey) -> Option<Value> {
        let inner = self.inner.read();
        let (group, row) = inner.edge(id.as_u64())?;
        if group.version(row).deleted != ABSENT {
            return None;
        }
        inner.value_at(&group.columns, row, key.as_str())
    }

    fn get_node_property_batch(&self, ids: &[NodeId], key: &PropertyKey) -> Vec<Option<Value>> {
        let inner = self.inner.read();
        ids.iter().map(|id| inner.node_value(*id, key)).collect()
    }

    fn try_get_node_property_batch(
        &self,
        ids: &[NodeId],
        key: &PropertyKey,
    ) -> Result<Vec<Option<Value>>> {
        Ok(self.get_node_property_batch(ids, key))
    }

    fn get_nodes_properties_batch(&self, ids: &[NodeId]) -> Vec<FxHashMap<PropertyKey, Value>> {
        let inner = self.inner.read();
        ids.iter().map(|id| inner.node_values(*id, None)).collect()
    }

    fn get_nodes_properties_selective_batch(
        &self,
        ids: &[NodeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        let inner = self.inner.read();
        ids.iter()
            .map(|id| inner.node_values(*id, Some(keys)))
            .collect()
    }

    fn get_edges_properties_selective_batch(
        &self,
        ids: &[EdgeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        let inner = self.inner.read();
        ids.iter()
            .map(|id| match inner.edge(id.as_u64()) {
                Some((group, row)) if group.version(row).deleted == ABSENT => {
                    inner.values_of(&group.columns, row, Some(keys))
                }
                _ => FxHashMap::default(),
            })
            .collect()
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

    fn has_backward_adjacency(&self) -> bool {
        true
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

    fn nodes_by_label(&self, label: &str) -> Vec<NodeId> {
        let inner = self.inner.read();
        let Some(label) = inner.label_id(label) else {
            return Vec::new();
        };
        let mut ids = Vec::new();
        for (index, group) in &inner.nodes {
            if let Some(rows) = group.labels.get(&label) {
                ids.extend(
                    rows.rows()
                        .filter(|row| group.version(*row).exists())
                        .map(|row| NodeId::new(index * ROWS_PER_GROUP + row as u64)),
                );
            }
        }
        ids
    }

    fn nodes_by_label_count(&self, label: &str) -> usize {
        let inner = self.inner.read();
        inner
            .label_id(label)
            .and_then(|id| inner.counts.labels.get(id as usize).copied())
            .map_or(0, size)
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

    fn find_nodes_by_property(&self, property: &str, value: &Value) -> Vec<NodeId> {
        self.find_nodes_by_properties(&[(property, value.clone())])
    }

    fn find_nodes_by_properties(&self, conditions: &[(&str, Value)]) -> Vec<NodeId> {
        let now = self.now();
        let inner = self.inner.read();
        inner.node_rows(|_, group, row| {
            group.version(row).visible(now)
                && conditions.iter().all(|(key, value)| {
                    inner.value_at(&group.columns, row, key).as_ref() == Some(value)
                })
        })
    }

    fn find_nodes_in_range(
        &self,
        property: &str,
        min: Option<&Value>,
        max: Option<&Value>,
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> Vec<NodeId> {
        let now = self.now();
        let inner = self.inner.read();
        inner.node_rows(|_, group, row| {
            group.version(row).visible(now)
                && inner
                    .value_at(&group.columns, row, property)
                    .is_some_and(|value| in_range(&value, min, max, min_inclusive, max_inclusive))
        })
    }

    fn node_property_might_match(
        &self,
        _property: &PropertyKey,
        _op: CompareOp,
        _value: &Value,
    ) -> bool {
        // No zone maps on hot row groups yet (H1c): every row group might.
        true
    }

    fn edge_property_might_match(
        &self,
        _property: &PropertyKey,
        _op: CompareOp,
        _value: &Value,
    ) -> bool {
        true
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
}

impl GraphStoreSearch for RowGroupStore {}

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
    use grafeo_common::types::Value;

    use super::in_range;

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
