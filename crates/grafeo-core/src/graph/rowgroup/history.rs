//! History on the row-group store: what time travel reads (H2c, the
//! time-travel design).
//!
//! History is the versions snapshot isolation keeps (`versions`, `updates`),
//! kept longer: the engine collects at the older of its read horizon and the
//! start of the database's retention window, so a reader at any epoch from
//! there on reads what was committed then, through the same reads at a view
//! as a transaction's snapshot. Nothing here forks a read path: this module
//! holds what the store needs besides.
//!
//! - Whether the database keeps history ([`set_keeps_history`]): replay then
//!   keeps what a replayed value or label write replaces, stamped at its
//!   group's epoch, as an immediate write keeps it, so a crash keeps the
//!   history the log holds. Without it replay keeps nothing, as before.
//! - The floor ([`history_floor`]): the oldest epoch the store answers reads
//!   at, the newest horizon it was collected at (a collect drops what a
//!   reader before its horizon would read), or the one a load restored its
//!   history from.
//! - The history calls `LpgStore` answers with `temporal`, with its names
//!   (`node_property_history`, `node_property_history_for_key`,
//!   `edge_property_history`, `get_node_property_at_epoch`, and the
//!   [`GraphStore`](crate::graph::GraphStore) calls `get_node_history` and
//!   `get_edge_history`), answered from the versions the store holds.
//!
//! A version's epoch is the commit that wrote it. The oldest version of a
//! row starts at the epoch its create committed when the store still knows
//! it, else at the floor: what the row held before is not retained.
//!
//! [`set_keeps_history`]: RowGroupStore::set_keeps_history
//! [`history_floor`]: RowGroupStore::history_floor

use std::sync::atomic::Ordering;

use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, Value};

use super::column::Columns;
use super::{ABSENT, Inner, PENDING, Read, RowGroupStore, RowVersion};
use crate::graph::lpg::{Edge, Node};

/// Whether `stamp` is a commit epoch (not absent, not a transaction's).
fn committed(stamp: u64) -> bool {
    stamp != ABSENT && stamp & PENDING == 0
}

/// The epochs a row's versions span as the store holds them: the first
/// epoch it vouches for (its create's, or the floor), the last epoch a
/// reader sees the row at (`None`: it is not deleted by a commit), and the
/// delete's epoch. `None` for a row no commit created.
fn span(version: RowVersion, floor: u64) -> Option<(u64, Option<u64>)> {
    if !committed(version.created) {
        return None;
    }
    let start = version.created.max(floor);
    let deleted = committed(version.deleted).then_some(version.deleted);
    if deleted.is_some_and(|deleted| deleted <= start) {
        return None;
    }
    Some((start, deleted))
}

/// The commit epochs of the writes a chain keeps a row's earlier state of,
/// from `stamps`, that lie after `start` and before `end` (the delete):
/// the epochs a new version of the row began at.
fn boundaries(start: u64, end: Option<u64>, stamps: impl Iterator<Item = u64>) -> Vec<u64> {
    let mut epochs: Vec<u64> = std::iter::once(start)
        .chain(stamps.filter(|stamp| {
            committed(*stamp) && *stamp > start && end.is_none_or(|end| *stamp < end)
        }))
        .collect();
    epochs.sort_unstable();
    epochs.dedup();
    epochs
}

impl Inner {
    /// The versions of key `key` of a row of `columns` whose versions span
    /// `start` to `end` (see [`span`]): each epoch a commit wrote the key at
    /// with the value it wrote, a removal as `Value::Null`, oldest first.
    /// The value the row held at `start` comes first unless it had none.
    fn key_history(
        columns: &Columns,
        key: u32,
        row: usize,
        start: u64,
        end: Option<u64>,
    ) -> Vec<(EpochId, Value)> {
        let epochs = boundaries(start, end, columns.kept(key, row).map(|(stamp, _)| stamp));
        let mut versions = Vec::with_capacity(epochs.len());
        for epoch in epochs {
            match columns.get_at(key, row, Read::At(epoch)) {
                Some(value) => versions.push((EpochId::new(epoch), value)),
                None if epoch == start => {}
                None => versions.push((EpochId::new(epoch), Value::Null)),
            }
        }
        versions
    }

    /// Every key's versions of a row of `columns` (see
    /// [`key_history`](Self::key_history)), keys by name ascending; a key
    /// with no version is left out.
    fn row_history(
        &self,
        columns: &Columns,
        row: usize,
        start: u64,
        end: Option<u64>,
    ) -> Vec<(PropertyKey, Vec<(EpochId, Value)>)> {
        let mut history: Vec<(PropertyKey, Vec<(EpochId, Value)>)> = columns
            .keys()
            .filter_map(|key| {
                let name = self.keys.get_name(key)?;
                let versions = Self::key_history(columns, key, row, start, end);
                (!versions.is_empty()).then(|| (PropertyKey::new(name.as_str()), versions))
            })
            .collect();
        history.sort_by(|(a, _), (b, _)| a.cmp(b));
        history
    }
}

impl RowGroupStore {
    /// Sets whether the database keeps history (a retention window is set),
    /// for this store and its named graphs, and for the graphs created in it
    /// from now on: with it, replay keeps what a value or label write
    /// replaces, stamped at its group's epoch, so the versions a crash
    /// leaves in the log stay readable after the reopen; without, replay
    /// keeps nothing (no reader runs during recovery).
    pub fn set_keeps_history(&self, keeps: bool) {
        self.history.store(keeps, Ordering::Release);
        for name in self.graph_names() {
            if let Some(graph) = self.graph(&name) {
                graph.set_keeps_history(keeps);
            }
        }
    }

    /// Whether the store keeps history (see
    /// [`set_keeps_history`](Self::set_keeps_history)).
    #[must_use]
    pub fn keeps_history(&self) -> bool {
        self.history.load(Ordering::Acquire)
    }

    /// The oldest epoch the store answers reads at: the newest horizon it
    /// was collected at, or the one its history was loaded from. A read at
    /// an earlier epoch may miss what a collect dropped, so the engine
    /// refuses it.
    #[must_use]
    pub fn history_floor(&self) -> EpochId {
        EpochId::new(self.floor.load(Ordering::Acquire))
    }

    /// Moves the floor of this store and its named graphs to `floor` when it
    /// is later (see [`history_floor`](Self::history_floor)): what a load
    /// does with the epoch the file's history starts at.
    pub fn set_history_floor(&self, floor: EpochId) {
        self.floor.fetch_max(floor.as_u64(), Ordering::AcqRel);
        for name in self.graph_names() {
            if let Some(graph) = self.graph(&name) {
                graph.set_history_floor(floor);
            }
        }
    }

    /// Notes a collect at `horizon`: the floor moves to it.
    pub(super) fn note_floor(&self, horizon: EpochId) {
        self.floor.fetch_max(horizon.as_u64(), Ordering::AcqRel);
    }

    /// The value of `key` of node `id` as a reader at `epoch` sees it, as
    /// `LpgStore::get_node_property_at_epoch` returns it: `None` when the
    /// node had no such value then, or no reader at `epoch` sees the node.
    #[must_use]
    pub fn get_node_property_at_epoch(
        &self,
        id: NodeId,
        key: &PropertyKey,
        epoch: EpochId,
    ) -> Option<Value> {
        self.inner
            .read()
            .node_value_at(id.as_u64(), key.as_str(), Read::at(epoch))
    }

    /// The versions of every property of node `id`, as
    /// `LpgStore::node_property_history` lists them: per key (by name,
    /// ascending) each epoch a commit wrote it at with the value it wrote,
    /// a removal as `Value::Null`, oldest first; the first version at the
    /// epoch the node was created, or at the floor when that is later (see
    /// the module docs). Empty for a node no commit created.
    #[must_use]
    pub fn node_property_history(&self, id: NodeId) -> Vec<(PropertyKey, Vec<(EpochId, Value)>)> {
        let floor = self.floor.load(Ordering::Acquire);
        let inner = self.inner.read();
        let Some((group, row)) = inner.node(id.as_u64()) else {
            return Vec::new();
        };
        let Some((start, end)) = span(group.version(row), floor) else {
            return Vec::new();
        };
        inner.row_history(&group.columns, row, start, end)
    }

    /// The versions of property `key` of node `id` (see
    /// [`node_property_history`](Self::node_property_history)).
    #[must_use]
    pub fn node_property_history_for_key(&self, id: NodeId, key: &str) -> Vec<(EpochId, Value)> {
        let floor = self.floor.load(Ordering::Acquire);
        let inner = self.inner.read();
        let (Some((group, row)), Some(key)) = (inner.node(id.as_u64()), inner.keys.get_id(key))
        else {
            return Vec::new();
        };
        let Some((start, end)) = span(group.version(row), floor) else {
            return Vec::new();
        };
        Inner::key_history(&group.columns, key, row, start, end)
    }

    /// The versions of every property of edge `id` (see
    /// [`node_property_history`](Self::node_property_history)).
    #[must_use]
    pub fn edge_property_history(&self, id: EdgeId) -> Vec<(PropertyKey, Vec<(EpochId, Value)>)> {
        let floor = self.floor.load(Ordering::Acquire);
        let inner = self.inner.read();
        let Some((group, row)) = inner.edge(id.as_u64()) else {
            return Vec::new();
        };
        let Some((start, end)) = span(group.version(row), floor) else {
            return Vec::new();
        };
        inner.row_history(&group.columns, row, start, end)
    }

    /// The versions of node `id`, newest first, as
    /// [`GraphStore::get_node_history`](crate::graph::GraphStore::get_node_history)
    /// lists them: each the epoch a commit made it at (the create, or a
    /// write of a value or a label), the epoch the next one or the delete
    /// ended it at (`None` for the version a reader sees now), and the node
    /// as a reader at its first epoch sees it.
    pub(super) fn node_history(&self, id: NodeId) -> Vec<(EpochId, Option<EpochId>, Node)> {
        let floor = self.floor.load(Ordering::Acquire);
        let inner = self.inner.read();
        let Some((group, row)) = inner.node(id.as_u64()) else {
            return Vec::new();
        };
        let Some((start, end)) = span(group.version(row), floor) else {
            return Vec::new();
        };
        let stamps = group
            .columns
            .keys()
            .flat_map(|key| group.columns.kept(key, row).map(|(stamp, _)| stamp))
            .chain(
                group
                    .label_updates
                    .kept_stamped(row)
                    .map(|(stamp, _)| stamp),
            );
        let epochs = boundaries(start, end, stamps);
        versions(&epochs, end, |epoch| {
            inner.read_node(id.as_u64(), Read::At(epoch))
        })
    }

    /// The versions of edge `id`, newest first (see
    /// [`node_history`](Self::node_history)): a write of a value begins a
    /// version.
    pub(super) fn edge_history(&self, id: EdgeId) -> Vec<(EpochId, Option<EpochId>, Edge)> {
        let floor = self.floor.load(Ordering::Acquire);
        let inner = self.inner.read();
        let Some((group, row)) = inner.edge(id.as_u64()) else {
            return Vec::new();
        };
        let Some((start, end)) = span(group.version(row), floor) else {
            return Vec::new();
        };
        let stamps = group
            .columns
            .keys()
            .flat_map(|key| group.columns.kept(key, row).map(|(stamp, _)| stamp));
        let epochs = boundaries(start, end, stamps);
        versions(&epochs, end, |epoch| {
            inner.read_edge(id.as_u64(), Read::At(epoch))
        })
    }
}

/// The versions that begin at `epochs` (ascending) and the last of which
/// ends at `end`, newest first, each read at its first epoch by `read`.
fn versions<T>(
    epochs: &[u64],
    end: Option<u64>,
    read: impl Fn(u64) -> Option<T>,
) -> Vec<(EpochId, Option<EpochId>, T)> {
    let mut versions = Vec::with_capacity(epochs.len());
    for (index, epoch) in epochs.iter().enumerate() {
        let until = epochs.get(index + 1).copied().or(end).map(EpochId::new);
        if let Some(entity) = read(*epoch) {
            versions.push((EpochId::new(*epoch), until, entity));
        }
    }
    versions.reverse();
    versions
}

#[cfg(test)]
mod tests;
