//! [`LpgStore`] as a [`ChangeTarget`]: the store side of a change set.
//!
//! `apply` is the store's versioned writes, returning what a write replaced
//! instead of logging it per transaction: a create inserts at the op's id, a
//! transaction's create is a version at `EpochId::PENDING` named after the
//! transaction, a delete marks the version deleted by it. Without `temporal`
//! values and labels change in place; with it, each write of a value or a
//! label set appends a version at `EpochId::PENDING`. So every write reports
//! [`PendingVersion::Created`], but a delete of a node or edge the
//! transaction created, which changes the transaction's own pending version
//! ([`PendingVersion::Replaced`]).
//!
//! `stamp` gives the versions the entries name the commit epoch and moves
//! the statistics counters (a transaction's writes count at commit), and
//! `undo` drops or writes back what the entries changed, last to first.
//! Both walk only the entries they are given. `undo` re-inserts into maps,
//! so the store cannot promise that it never fails: an error is a broken
//! invariant, and the engine poisons the database.
//!
//! Replay applies with [`Writer::Replay`]: the same writes at the group's
//! epoch, by `TransactionId::SYSTEM`, counted and stamped at once, without
//! before-images. An immediate write ([`Writer::Immediate`]) goes the same
//! way, but lenient and with before-images: what it returns is recorded
//! for the log and change data capture, and never stamped or undone.

use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};

use grafeo_common::change::{
    Before, BulkRange, Change, DataOp, EdgeImage, Entity, Labels, NodeImage, PendingVersion,
    Properties, Table,
};
#[cfg(not(feature = "tiered-storage"))]
use grafeo_common::mvcc::VersionChain;
#[cfg(feature = "tiered-storage")]
use grafeo_common::mvcc::{HotVersionRef, VersionIndex};
use grafeo_common::types::{ArcStr, EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use parking_lot::RwLock;

use super::LpgStore;
use crate::graph::apply::{
    Applied, ApplyError, ChangeTarget, Writer, check_bulk_row, refuse_triple, refused,
};
use crate::graph::lpg::{EdgeRecord, NodeRecord};

/// The versions of a node: a chain of records, or with tiered storage an
/// index of arena and cold references. Both name the transaction that
/// created and deleted each version, through methods of the same names.
#[cfg(not(feature = "tiered-storage"))]
type NodeVersions = VersionChain<NodeRecord>;
#[cfg(feature = "tiered-storage")]
type NodeVersions = VersionIndex;

/// The versions of an edge (see [`NodeVersions`]).
#[cfg(not(feature = "tiered-storage"))]
type EdgeVersions = VersionChain<EdgeRecord>;
#[cfg(feature = "tiered-storage")]
type EdgeVersions = VersionIndex;

/// How a [`Writer`] writes this store.
#[derive(Debug, Clone, Copy)]
struct Mode {
    /// The writer.
    writer: Writer,
    /// Whose versions it writes: the transaction's, `TransactionId::SYSTEM`
    /// for replay.
    by: TransactionId,
    /// The epoch it reads at (a transaction's snapshot, replay's group
    /// epoch). Records and arenas are allocated at it, and deletes marked
    /// deleted at it, as the store's versioned writes do.
    at: EpochId,
    /// The epoch of the versions it writes: `EpochId::PENDING` for a
    /// transaction (stamped at commit), the group's for replay.
    version: EpochId,
}

impl Mode {
    /// The mode of `writer`.
    ///
    /// # Errors
    ///
    /// Refuses a transaction writer with the system transaction's id: its
    /// pending versions would be no transaction's.
    fn of(writer: Writer) -> Result<Self, ApplyError> {
        match writer {
            Writer::Transaction { id, snapshot } => {
                if id == TransactionId::SYSTEM {
                    return Err(ApplyError::Refused(
                        "a transaction's write needs the transaction's id, not the system one"
                            .to_string(),
                    ));
                }
                Ok(Self {
                    writer,
                    by: id,
                    at: snapshot,
                    version: EpochId::PENDING,
                })
            }
            Writer::Replay { epoch } | Writer::Immediate { epoch, .. } => Ok(Self {
                writer,
                by: TransactionId::SYSTEM,
                at: epoch,
                version: epoch,
            }),
        }
    }

    /// Whether the write is stamped at once (replay, an immediate write):
    /// written by the system at its epoch, which it reads at, and counted
    /// as it is applied.
    fn stamped(self) -> bool {
        self.writer.stamps_at_once()
    }

    /// Whether the before-image is built (a transaction, an immediate
    /// write; not replay).
    fn images(self) -> bool {
        self.writer.builds_images()
    }

    /// What `apply` returns once it applied a write: the entry to record
    /// with its before-image (built only then), or `Committed` for replay.
    fn applied(self, before: impl FnOnce() -> Before, version: PendingVersion) -> Applied {
        if self.images() {
            Applied::Changed {
                before: before(),
                version,
            }
        } else {
            Applied::Committed
        }
    }

    /// Whether the writer is the transaction that created what it sees, as
    /// `created_by_writer` tells (asked only for a transaction): a stamped
    /// write has no pending version of its own.
    fn owns(self, created_by_writer: impl FnOnce() -> bool) -> bool {
        !self.stamped() && created_by_writer()
    }
}

/// Whether an edge create looks up its endpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Endpoints {
    /// It does: the writer must see both.
    Check,
    /// It does not: a bulk write checked them for every row (see
    /// [`ChangeTarget::apply_bulk_row`]).
    Checked,
}

/// The version an entry created, or the transaction's own it replaced.
fn version_of(own: bool) -> PendingVersion {
    if own {
        PendingVersion::Replaced
    } else {
        PendingVersion::Created
    }
}

/// The error of a record that cannot be read (tiered storage: a cold one
/// that does not decode).
#[cfg(feature = "tiered-storage")]
fn unreadable(entity: Entity) -> ApplyError {
    ApplyError::Refused(format!(
        "the record of {} cannot be read",
        crate::graph::apply::describe(entity)
    ))
}

/// The error of an entry whose before-image is not the one its op has: the
/// change set refuses one, so it is a broken invariant.
fn misfit(op: &DataOp, before: &Before) -> ApplyError {
    ApplyError::Refused(format!(
        "an entry of kind {} with a before-image that does not fit it: {before:?}",
        op.kind()
    ))
}

/// Reserves `count` ids of `next`, all below the invalid id (`u64::MAX`).
fn reserve(next: &AtomicU64, count: u64, what: &str) -> Result<Range<u64>, ApplyError> {
    let mut start = next.load(Ordering::Acquire);
    loop {
        let end = start
            .checked_add(count)
            .filter(|end| *end < u64::MAX)
            .ok_or_else(|| {
                ApplyError::Refused(format!(
                    "{count} {what} ids from {start}: the ids are exhausted"
                ))
            })?;
        match next.compare_exchange_weak(start, end, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return Ok(start..end),
            Err(current) => start = current,
        }
    }
}

impl ChangeTarget for LpgStore {
    fn reserve_node_ids(&self, count: u64) -> Result<Range<u64>, ApplyError> {
        reserve(&self.next_node_id, count, "node")
    }

    fn reserve_edge_ids(&self, count: u64) -> Result<Range<u64>, ApplyError> {
        reserve(&self.next_edge_id, count, "edge")
    }

    fn apply(&self, op: &DataOp, writer: Writer) -> Result<Applied, ApplyError> {
        let mode = Mode::of(writer)?;
        self.refuse_if_dropped()?;
        let applied = match op {
            DataOp::CreateNode {
                id,
                labels,
                properties,
            } => self.apply_create_node(*id, labels, properties, mode),
            DataOp::DeleteNode { id } => self.apply_delete_node(*id, mode),
            DataOp::CreateEdge {
                id,
                src,
                dst,
                edge_type,
                properties,
            } => self.apply_create_edge(
                *id,
                (*src, *dst),
                edge_type,
                properties,
                mode,
                Endpoints::Check,
            ),
            DataOp::DeleteEdge { id } => self.apply_delete_edge(*id, mode),
            DataOp::SetNodeProperty { id, key, value } => {
                self.apply_set_node_value(*id, key, value, mode)
            }
            DataOp::RemoveNodeProperty { id, key } => self.apply_remove_node_value(*id, key, mode),
            DataOp::SetEdgeProperty { id, key, value } => {
                self.apply_set_edge_value(*id, key, value, mode)
            }
            DataOp::RemoveEdgeProperty { id, key } => self.apply_remove_edge_value(*id, key, mode),
            DataOp::AddNodeLabel { id, label } => self.apply_label(*id, label, true, mode),
            DataOp::RemoveNodeLabel { id, label } => self.apply_label(*id, label, false, mode),
            DataOp::InsertTriple { .. } | DataOp::DeleteTriple { .. } => Err(refuse_triple(op)),
        }?;
        if mode.stamped() {
            // A stamped write is committed as it is applied: the store's
            // epoch follows.
            self.sync_epoch(mode.at);
        }
        Ok(applied)
    }

    /// A row of a bulk write: the create `apply` makes, without looking up
    /// an edge's endpoints again (see [`ChangeTarget::apply_bulk_row`]).
    fn apply_bulk_row(&self, op: &DataOp, writer: Writer) -> Result<(), ApplyError> {
        check_bulk_row(op, writer)?;
        let mode = Mode::of(writer)?;
        self.refuse_if_dropped()?;
        match op {
            DataOp::CreateNode {
                id,
                labels,
                properties,
            } => self.apply_create_node(*id, labels, properties, mode),
            DataOp::CreateEdge {
                id,
                src,
                dst,
                edge_type,
                properties,
            } => self.apply_create_edge(
                *id,
                (*src, *dst),
                edge_type,
                properties,
                mode,
                Endpoints::Checked,
            ),
            // `check_bulk_row` lets creates through only.
            other => Err(ApplyError::Refused(format!(
                "a bulk write's row of kind {}",
                other.kind()
            ))),
        }
        .map(drop)
    }

    fn stamp(
        &self,
        transaction: TransactionId,
        entries: &mut dyn Iterator<Item = &Change>,
        epoch: EpochId,
    ) -> Result<(), ApplyError> {
        let mut stamp = Stamp::default();
        for change in entries {
            stamp.add(self, change)?;
        }
        self.stamp_versions(transaction, &stamp, epoch)?;
        for range in &stamp.bulk {
            self.stamp_bulk(transaction, range, epoch, &mut stamp.counts)?;
        }
        stamp.counts.apply(self);
        self.sync_epoch(epoch);
        Ok(())
    }

    fn undo(
        &self,
        transaction: TransactionId,
        entries: &mut dyn DoubleEndedIterator<Item = &Change>,
    ) -> Result<(), ApplyError> {
        while let Some(change) = entries.next_back() {
            match change {
                Change::Data {
                    op,
                    before,
                    version,
                    ..
                } => self.undo_one(transaction, op, before, *version)?,
                Change::Bulk(range) => self.undo_bulk(transaction, range),
            }
        }
        Ok(())
    }
}

// ── Versions ────────────────────────────────────────────────────────

impl LpgStore {
    /// Refuses every write once the graph is dropped: a writer that resolved
    /// the graph before would log its write under the graph's name, and
    /// replay would create the graph again for it.
    fn refuse_if_dropped(&self) -> Result<(), ApplyError> {
        if self.is_dropped() {
            return Err(ApplyError::Refused(
                "the graph was dropped: it takes no more writes".to_string(),
            ));
        }
        Ok(())
    }

    /// The nodes' versions, by id.
    fn node_version_map(&self) -> &RwLock<FxHashMap<NodeId, NodeVersions>> {
        #[cfg(not(feature = "tiered-storage"))]
        return &self.nodes;
        #[cfg(feature = "tiered-storage")]
        return &self.node_versions;
    }

    /// The edges' versions, by id.
    fn edge_version_map(&self) -> &RwLock<FxHashMap<EdgeId, EdgeVersions>> {
        #[cfg(not(feature = "tiered-storage"))]
        return &self.edges;
        #[cfg(feature = "tiered-storage")]
        return &self.edge_versions;
    }

    /// Whether the writer sees node `id` (`None` when it sees none, or sees
    /// it deleted), and if so whether its own transaction created it.
    ///
    /// # Errors
    ///
    /// Refuses when the version it sees cannot be read (tiered storage).
    fn seen_node(&self, id: NodeId, mode: Mode) -> Result<Option<bool>, ApplyError> {
        self.seen_node_in(&self.node_version_map().read(), id, mode)
    }

    /// [`seen_node`](Self::seen_node) in `map`, the nodes' versions the
    /// caller holds locked (to look at several nodes under one lock).
    fn seen_node_in(
        &self,
        map: &FxHashMap<NodeId, NodeVersions>,
        id: NodeId,
        mode: Mode,
    ) -> Result<Option<bool>, ApplyError> {
        let Some(versions) = map.get(&id) else {
            return Ok(None);
        };
        let visible = if mode.stamped() {
            versions.visible_at(mode.at)
        } else {
            versions.visible_to(mode.at, mode.by)
        };
        #[cfg(not(feature = "tiered-storage"))]
        let record = visible.copied();
        #[cfg(feature = "tiered-storage")]
        let record = match visible {
            Some(version) => Some(
                self.read_node_record(&version)
                    .ok_or_else(|| unreadable(Entity::Node(id)))?,
            ),
            None => None,
        };
        Ok(record
            .filter(|record| !record.is_deleted())
            .map(|_| mode.owns(|| versions.modified_by(mode.by))))
    }

    /// The record of edge `id` the writer sees, and whether its own
    /// transaction created the edge.
    ///
    /// # Errors
    ///
    /// Refuses when the version it sees cannot be read (tiered storage).
    fn seen_edge(&self, id: EdgeId, mode: Mode) -> Result<Option<(EdgeRecord, bool)>, ApplyError> {
        let map = self.edge_version_map().read();
        let Some(versions) = map.get(&id) else {
            return Ok(None);
        };
        let visible = if mode.stamped() {
            versions.visible_at(mode.at)
        } else {
            versions.visible_to(mode.at, mode.by)
        };
        #[cfg(not(feature = "tiered-storage"))]
        let record = visible.copied();
        #[cfg(feature = "tiered-storage")]
        let record = match visible {
            Some(version) => Some(
                self.read_edge_record(&version)
                    .ok_or_else(|| unreadable(Entity::Edge(id)))?,
            ),
            None => None,
        };
        Ok(record
            .filter(|record| !record.is_deleted())
            .map(|record| (record, mode.owns(|| versions.modified_by(mode.by)))))
    }

    /// Whether the writer sees an edge of node `id`. Both ways with backward
    /// adjacency; without it outgoing edges only, and incoming ones too by a
    /// scan of the edges in debug builds (in release builds a scan per
    /// delete would make delete-heavy statements quadratic, and the writer
    /// checked incoming edges before the delete).
    fn sees_an_edge_of(&self, id: NodeId, mode: Mode) -> Result<bool, ApplyError> {
        for (_, edge) in self.forward_adj.edges_from(id) {
            if self.seen_edge(edge, mode)?.is_some() {
                return Ok(true);
            }
        }
        if let Some(backward) = &self.backward_adj {
            for (_, edge) in backward.edges_from(id) {
                if self.seen_edge(edge, mode)?.is_some() {
                    return Ok(true);
                }
            }
        } else if cfg!(debug_assertions) {
            let held: Vec<EdgeId> = self.edge_version_map().read().keys().copied().collect();
            for edge in held {
                if self
                    .seen_edge(edge, mode)?
                    .is_some_and(|(record, _)| record.dst == id)
                {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// Inserts the first version of node `id`, `record`, as the writer's.
    ///
    /// # Errors
    ///
    /// Returns [`ApplyError::Exists`] when the store holds the id, and
    /// refuses when the arena cannot allocate the record (tiered storage);
    /// nothing changed then.
    fn insert_node_version(
        &self,
        id: NodeId,
        record: NodeRecord,
        mode: Mode,
    ) -> Result<(), ApplyError> {
        // A tiered store checks first, so a refused create allocates no
        // arena record; the check under the write lock below decides.
        #[cfg(feature = "tiered-storage")]
        if self.node_version_map().read().contains_key(&id) {
            return Err(ApplyError::Exists(Entity::Node(id)));
        }
        #[cfg(not(feature = "tiered-storage"))]
        let versions = VersionChain::with_initial(record, mode.version, mode.by);
        #[cfg(feature = "tiered-storage")]
        let versions = {
            // The arena's lock is released before the version lock is taken
            // (see the lock order on `arena_allocator`).
            let offset = self
                .arena_allocator
                .arena_or_create(mode.at)
                .and_then(|arena| {
                    arena
                        .alloc_value_with_offset(record)
                        .map(|(offset, _)| offset)
                })
                .map_err(|error| refused(Entity::Node(id), &error))?;
            VersionIndex::with_initial(HotVersionRef::new(mode.version, mode.at, offset, mode.by))
        };
        let mut map = self.node_version_map().write();
        if map.contains_key(&id) {
            return Err(ApplyError::Exists(Entity::Node(id)));
        }
        map.insert(id, versions);
        Ok(())
    }

    /// Inserts the first version of edge `id`, as
    /// [`insert_node_version`](Self::insert_node_version) does for a node.
    fn insert_edge_version(
        &self,
        id: EdgeId,
        record: EdgeRecord,
        mode: Mode,
    ) -> Result<(), ApplyError> {
        #[cfg(feature = "tiered-storage")]
        if self.edge_version_map().read().contains_key(&id) {
            return Err(ApplyError::Exists(Entity::Edge(id)));
        }
        #[cfg(not(feature = "tiered-storage"))]
        let versions = VersionChain::with_initial(record, mode.version, mode.by);
        #[cfg(feature = "tiered-storage")]
        let versions = {
            let offset = self
                .arena_allocator
                .arena_or_create(mode.at)
                .and_then(|arena| {
                    arena
                        .alloc_value_with_offset(record)
                        .map(|(offset, _)| offset)
                })
                .map_err(|error| refused(Entity::Edge(id), &error))?;
            VersionIndex::with_initial(HotVersionRef::new(mode.version, mode.at, offset, mode.by))
        };
        let mut map = self.edge_version_map().write();
        if map.contains_key(&id) {
            return Err(ApplyError::Exists(Entity::Edge(id)));
        }
        map.insert(id, versions);
        Ok(())
    }

    /// Whether `transaction` created node `id` (it holds a version by it).
    fn node_created_by(&self, id: NodeId, transaction: TransactionId) -> bool {
        self.node_version_map()
            .read()
            .get(&id)
            .is_some_and(|versions| versions.modified_by(transaction))
    }

    /// Whether `transaction` created edge `id`.
    fn edge_created_by(&self, id: EdgeId, transaction: TransactionId) -> bool {
        self.edge_version_map()
            .read()
            .get(&id)
            .is_some_and(|versions| versions.modified_by(transaction))
    }

    /// Keeps the node id allocator above `id`, a valid id (below
    /// `u64::MAX`, so one more fits).
    fn keep_node_ids_above(&self, id: NodeId) {
        self.next_node_id
            .fetch_max(id.as_u64() + 1, Ordering::AcqRel);
    }

    /// Keeps the edge id allocator above `id`, a valid id.
    fn keep_edge_ids_above(&self, id: EdgeId) {
        self.next_edge_id
            .fetch_max(id.as_u64() + 1, Ordering::AcqRel);
    }
}

// ── Values and labels, with their indexes ───────────────────────────

impl LpgStore {
    /// Sets a node's value with the index updates a set makes; with
    /// `temporal`, as a version at `at`.
    fn put_node_value(&self, id: NodeId, key: &PropertyKey, value: Value, at: EpochId) {
        self.update_property_index_on_set(id, key, &value);
        #[cfg(feature = "text-index")]
        self.update_text_index_on_set(id, key.as_str(), &value);
        #[cfg(not(feature = "temporal"))]
        {
            let _ = at;
            self.node_properties.set(id, key.clone(), value);
        }
        #[cfg(feature = "temporal")]
        self.node_properties.set(id, key.clone(), value, at);
        #[cfg(feature = "vector-index")]
        self.sync_vector_indexes_for_property(id, key.as_str());
    }

    /// Removes a node's value with the index updates a removal makes (with
    /// `temporal`, as a null version at `at`), returning the value removed:
    /// read and hidden in one step.
    ///
    /// # Errors
    ///
    /// Refuses, changing nothing, when the value cannot be read (a spilled
    /// value whose file cannot be read): the entry would miss it.
    fn take_node_value(
        &self,
        id: NodeId,
        key: &PropertyKey,
        at: EpochId,
    ) -> Result<Option<Value>, ApplyError> {
        #[cfg(not(feature = "temporal"))]
        {
            let _ = at;
            self.remove_node_property(id, key.as_str())
                .map_err(|error| refused(Entity::Node(id), &error))
        }
        #[cfg(feature = "temporal")]
        {
            let removed = self.node_properties.remove(id, key, at);
            self.update_indexes_on_remove(id, key.as_str(), removed.as_ref());
            Ok(removed)
        }
    }

    /// Sets an edge's value; with `temporal`, as a version at `at`.
    fn put_edge_value(&self, id: EdgeId, key: &PropertyKey, value: Value, at: EpochId) {
        #[cfg(not(feature = "temporal"))]
        {
            let _ = at;
            self.edge_properties.set(id, key.clone(), value);
        }
        #[cfg(feature = "temporal")]
        self.edge_properties.set(id, key.clone(), value, at);
    }

    /// Removes an edge's value, as [`take_node_value`](Self::take_node_value)
    /// does a node's.
    fn take_edge_value(
        &self,
        id: EdgeId,
        key: &PropertyKey,
        at: EpochId,
    ) -> Result<Option<Value>, ApplyError> {
        #[cfg(not(feature = "temporal"))]
        {
            let _ = at;
            self.edge_properties
                .remove(id, key)
                .map_err(|error| refused(Entity::Edge(id), &error))
        }
        #[cfg(feature = "temporal")]
        Ok(self.edge_properties.remove(id, key, at))
    }

    /// The label ids node `id` has now (with `temporal`, its latest label
    /// set: the writer's own).
    fn label_ids_of(&self, id: NodeId) -> FxHashSet<u32> {
        let node_labels = self.node_labels.read();
        #[cfg(not(feature = "temporal"))]
        let labels = node_labels.get(&id);
        #[cfg(feature = "temporal")]
        let labels = node_labels.get(&id).and_then(|log| log.latest());
        labels.cloned().unwrap_or_default()
    }

    /// The labels of node `id` now, in label id order.
    fn label_names_of(&self, id: NodeId) -> Labels {
        let mut ids: Vec<u32> = self.label_ids_of(id).into_iter().collect();
        ids.sort_unstable();
        let registry = self.label_registry.read();
        ids.into_iter()
            .filter_map(|label_id| registry.get_name(label_id).cloned())
            .collect()
    }

    /// The ids of `labels`, which the label dictionary holds: it keeps every
    /// name it was given.
    #[cfg(not(feature = "temporal"))]
    fn label_ids(&self, id: NodeId, labels: &Labels) -> Result<FxHashSet<u32>, ApplyError> {
        let registry = self.label_registry.read();
        labels
            .iter()
            .map(|label| registry.get_id(label))
            .collect::<Option<FxHashSet<u32>>>()
            .ok_or(ApplyError::Missing(Entity::Node(id)))
    }

    /// Takes node `id`'s labels away for its delete: out of the label index
    /// and, without `temporal`, out of the node's label map; with it, an
    /// empty label set is the node's new version at `at` (its history
    /// stays, and undo drops the version).
    fn drop_node_labels(&self, id: NodeId, at: EpochId) {
        let mut index = self.label_index.write();
        let mut node_labels = self.node_labels.write();
        #[cfg(not(feature = "temporal"))]
        let dropped = {
            let _ = at;
            node_labels.remove(&id).unwrap_or_default()
        };
        #[cfg(feature = "temporal")]
        let dropped = {
            let current = node_labels
                .get(&id)
                .and_then(|log| log.latest())
                .cloned()
                .unwrap_or_default();
            self.append_labels(&mut node_labels, id, at, FxHashSet::default());
            current
        };
        for label_id in dropped {
            if let Some(members) = index.get_mut(label_id as usize) {
                members.remove(&id);
            }
        }
    }

    /// Brings the label index, and the text and vector indexes of each
    /// label, in line with node `id` going from labels `from` to `to`.
    fn reindex_labels(&self, id: NodeId, from: &FxHashSet<u32>, to: &FxHashSet<u32>) {
        let lost: Vec<u32> = from.difference(to).copied().collect();
        let gained: Vec<u32> = to.difference(from).copied().collect();
        {
            let mut index = self.label_index.write();
            for &label_id in &lost {
                if let Some(members) = index.get_mut(label_id as usize) {
                    members.remove(&id);
                }
            }
            for &label_id in &gained {
                let at = label_id as usize;
                if index.len() <= at {
                    index.resize_with(at + 1, FxHashMap::default);
                }
                index[at].insert(id, ());
            }
        }
        let names = |ids: &[u32]| -> Vec<ArcStr> {
            let registry = self.label_registry.read();
            ids.iter()
                .filter_map(|label_id| registry.get_name(*label_id).cloned())
                .collect()
        };
        for label in names(&lost) {
            self.unindex_node_under_label(id, &label);
        }
        for label in names(&gained) {
            self.index_node_under_label(id, &label);
        }
    }
}

// ── apply ───────────────────────────────────────────────────────────

impl LpgStore {
    fn apply_create_node(
        &self,
        id: NodeId,
        labels: &Labels,
        properties: &Properties,
        mode: Mode,
    ) -> Result<Applied, ApplyError> {
        if !id.is_valid() {
            return Err(ApplyError::Refused(
                "a node create names the invalid node id".to_string(),
            ));
        }
        // The record's count is a hint that stops at its maximum, as the
        // label writes keep it (`update_label_count`): the node's labels are
        // in the label map, however many it has.
        let mut record = NodeRecord::new(id, mode.at);
        record.set_label_count(u16::try_from(labels.len()).unwrap_or(u16::MAX));
        self.insert_node_version(id, record, mode)?;
        self.keep_node_ids_above(id);

        let names: Vec<&str> = labels.iter().map(ArcStr::as_str).collect();
        #[cfg(not(feature = "temporal"))]
        self.register_node_labels(id, &names);
        #[cfg(feature = "temporal")]
        self.register_node_labels(id, &names, mode.version);
        for (key, value) in properties {
            self.put_node_value(id, key, value.clone(), mode.version);
        }
        if mode.stamped() {
            self.live_node_count.fetch_add(1, Ordering::Relaxed);
        }
        Ok(mode.applied(|| Before::Absent, PendingVersion::Created))
    }

    fn apply_delete_node(&self, id: NodeId, mode: Mode) -> Result<Applied, ApplyError> {
        let Some(own) = self.seen_node(id, mode)? else {
            return mode.writer.unseen(Entity::Node(id));
        };
        if self.sees_an_edge_of(id, mode)? {
            return Err(ApplyError::HasEdges(id));
        }
        // The image is read before anything changes: a value that cannot be
        // read refuses the delete, as an undo could not restore it.
        let image = if mode.images() {
            let mut properties: Properties = self
                .node_properties
                .try_get_all(id)
                .map_err(|error| refused(Entity::Node(id), &error))?
                .into_iter()
                .collect();
            properties.sort_by(|(a, _), (b, _)| a.cmp(b));
            Some(NodeImage {
                labels: self.label_names_of(id),
                properties,
            })
        } else {
            None
        };
        let marked = self
            .node_version_map()
            .write()
            .get_mut(&id)
            .is_some_and(|versions| versions.mark_deleted(mode.at, mode.by));
        if !marked {
            return Err(ApplyError::Missing(Entity::Node(id)));
        }
        self.drop_node_labels(id, mode.version);
        #[cfg(feature = "text-index")]
        self.remove_from_all_text_indexes(id);
        // Out of the property indexes while the node's values are known.
        self.remove_from_all_property_indexes(id);
        #[cfg(feature = "vector-index")]
        self.remove_from_all_vector_indexes(id);
        #[cfg(not(feature = "temporal"))]
        self.node_properties.remove_all(id);
        #[cfg(feature = "temporal")]
        self.node_properties.remove_all(id, mode.version);
        if mode.stamped() {
            self.live_node_count.fetch_sub(1, Ordering::Relaxed);
        }
        Ok(match image {
            Some(image) => Applied::Changed {
                before: Before::Node(Box::new(image)),
                version: version_of(own),
            },
            None => Applied::Committed,
        })
    }

    fn apply_create_edge(
        &self,
        id: EdgeId,
        (src, dst): (NodeId, NodeId),
        edge_type: &ArcStr,
        properties: &Properties,
        mode: Mode,
        endpoints: Endpoints,
    ) -> Result<Applied, ApplyError> {
        if !id.is_valid() {
            return Err(ApplyError::Refused(
                "an edge create names the invalid edge id".to_string(),
            ));
        }
        // A tiered store checks the id first, so a refused create allocates
        // no arena record; the insert checks it under the write lock.
        #[cfg(feature = "tiered-storage")]
        if self.edge_version_map().read().contains_key(&id) {
            return Err(ApplyError::Exists(Entity::Edge(id)));
        }
        if endpoints == Endpoints::Check {
            let nodes = self.node_version_map().read();
            for end in [src, dst] {
                if self.seen_node_in(&nodes, end, mode)?.is_none() {
                    return Err(ApplyError::Missing(Entity::Node(end)));
                }
            }
        }
        let type_id = self.get_or_create_edge_type_id(edge_type);
        let record = EdgeRecord::new(id, src, dst, type_id, mode.at);
        self.insert_edge_version(id, record, mode)?;
        self.keep_edge_ids_above(id);
        self.forward_adj.add_edge(src, dst, id);
        if let Some(backward) = &self.backward_adj {
            backward.add_edge(dst, src, id);
        }
        for (key, value) in properties {
            self.put_edge_value(id, key, value.clone(), mode.version);
        }
        if mode.stamped() {
            self.live_edge_count.fetch_add(1, Ordering::Relaxed);
            self.increment_edge_type_count(type_id);
        }
        Ok(mode.applied(|| Before::Absent, PendingVersion::Created))
    }

    fn apply_delete_edge(&self, id: EdgeId, mode: Mode) -> Result<Applied, ApplyError> {
        let Some((record, own)) = self.seen_edge(id, mode)? else {
            return mode.writer.unseen(Entity::Edge(id));
        };
        let image = if !mode.images() {
            None
        } else {
            let edge_type = self
                .edge_types
                .read()
                .get_name(record.type_id)
                .cloned()
                .ok_or(ApplyError::Missing(Entity::Edge(id)))?;
            let mut properties: Properties = self
                .edge_properties
                .try_get_all(id)
                .map_err(|error| refused(Entity::Edge(id), &error))?
                .into_iter()
                .collect();
            properties.sort_by(|(a, _), (b, _)| a.cmp(b));
            Some(EdgeImage {
                src: record.src,
                dst: record.dst,
                edge_type,
                properties,
            })
        };
        let marked = self
            .edge_version_map()
            .write()
            .get_mut(&id)
            .is_some_and(|versions| versions.mark_deleted(mode.at, mode.by));
        if !marked {
            return Err(ApplyError::Missing(Entity::Edge(id)));
        }
        self.forward_adj.mark_deleted(record.src, id);
        if let Some(backward) = &self.backward_adj {
            backward.mark_deleted(record.dst, id);
        }
        #[cfg(not(feature = "temporal"))]
        self.edge_properties.remove_all(id);
        #[cfg(feature = "temporal")]
        self.edge_properties.remove_all(id, mode.version);
        if mode.stamped() {
            self.live_edge_count.fetch_sub(1, Ordering::Relaxed);
            self.decrement_edge_type_count(record.type_id);
        }
        Ok(match image {
            Some(image) => Applied::Changed {
                before: Before::Edge(Box::new(image)),
                version: version_of(own),
            },
            None => Applied::Committed,
        })
    }

    fn apply_set_node_value(
        &self,
        id: NodeId,
        key: &PropertyKey,
        value: &Value,
        mode: Mode,
    ) -> Result<Applied, ApplyError> {
        if self.seen_node(id, mode)?.is_none() {
            return mode.writer.unseen(Entity::Node(id));
        }
        let old = if !mode.images() {
            None
        } else {
            self.node_properties
                .try_get(id, key)
                .map_err(|error| refused(Entity::Node(id), &error))?
        };
        self.put_node_value(id, key, value.clone(), mode.version);
        Ok(mode.applied(|| Before::Value(old), PendingVersion::Created))
    }

    fn apply_remove_node_value(
        &self,
        id: NodeId,
        key: &PropertyKey,
        mode: Mode,
    ) -> Result<Applied, ApplyError> {
        if self.seen_node(id, mode)?.is_none() {
            return mode.writer.unseen(Entity::Node(id));
        }
        match self.take_node_value(id, key, mode.version)? {
            Some(old) => Ok(mode.applied(|| Before::Value(Some(old)), PendingVersion::Created)),
            None => mode.writer.no_op(ApplyError::Missing(Entity::Node(id))),
        }
    }

    fn apply_set_edge_value(
        &self,
        id: EdgeId,
        key: &PropertyKey,
        value: &Value,
        mode: Mode,
    ) -> Result<Applied, ApplyError> {
        if self.seen_edge(id, mode)?.is_none() {
            return mode.writer.unseen(Entity::Edge(id));
        }
        let old = if !mode.images() {
            None
        } else {
            self.edge_properties
                .try_get(id, key)
                .map_err(|error| refused(Entity::Edge(id), &error))?
        };
        self.put_edge_value(id, key, value.clone(), mode.version);
        Ok(mode.applied(|| Before::Value(old), PendingVersion::Created))
    }

    fn apply_remove_edge_value(
        &self,
        id: EdgeId,
        key: &PropertyKey,
        mode: Mode,
    ) -> Result<Applied, ApplyError> {
        if self.seen_edge(id, mode)?.is_none() {
            return mode.writer.unseen(Entity::Edge(id));
        }
        match self.take_edge_value(id, key, mode.version)? {
            Some(old) => Ok(mode.applied(|| Before::Value(Some(old)), PendingVersion::Created)),
            None => mode.writer.no_op(ApplyError::Missing(Entity::Edge(id))),
        }
    }

    fn apply_label(
        &self,
        id: NodeId,
        label: &ArcStr,
        add: bool,
        mode: Mode,
    ) -> Result<Applied, ApplyError> {
        if self.seen_node(id, mode)?.is_none() {
            return mode.writer.unseen(Entity::Node(id));
        }
        let has = self
            .label_id(label)
            .is_some_and(|label_id| self.label_ids_of(id).contains(&label_id));
        if has == add {
            return mode.writer.no_op(if add {
                ApplyError::Exists(Entity::Node(id))
            } else {
                ApplyError::Missing(Entity::Node(id))
            });
        }
        let before = if !mode.images() {
            None
        } else {
            Some(self.label_names_of(id))
        };
        let changed = if add {
            self.add_label_at(id, label, mode.version)
        } else {
            self.remove_label_at(id, label, mode.version)
        };
        if !changed {
            // The check above holds under the transaction's claim on the
            // node: a broken invariant, reported before anything changed.
            return Err(ApplyError::Missing(Entity::Node(id)));
        }
        Ok(match before {
            Some(before) => Applied::Changed {
                before: Before::Labels(before),
                version: PendingVersion::Created,
            },
            None => Applied::Committed,
        })
    }
}

// ── stamp ───────────────────────────────────────────────────────────

/// The statistics counters a commit moves.
#[derive(Debug, Default)]
struct Counts {
    nodes: i64,
    edges: i64,
    /// Per edge type id.
    edge_types: FxHashMap<u32, i64>,
}

impl Counts {
    /// Adds the deltas to the store's counters.
    fn apply(&self, store: &LpgStore) {
        if self.nodes != 0 {
            store
                .live_node_count
                .fetch_add(self.nodes, Ordering::Relaxed);
        }
        if self.edges != 0 {
            store
                .live_edge_count
                .fetch_add(self.edges, Ordering::Relaxed);
        }
        if self.edge_types.is_empty() {
            return;
        }
        let mut counts = store.edge_type_live_counts.write();
        for (&type_id, &delta) in &self.edge_types {
            let at = type_id as usize;
            if counts.len() <= at {
                counts.resize(at + 1, 0);
            }
            counts[at] += delta;
        }
    }
}

/// What a commit stamps, gathered from its entries.
#[derive(Default)]
struct Stamp<'c> {
    /// The nodes and edges the transaction created: their versions get the
    /// epoch.
    nodes: Vec<NodeId>,
    edges: Vec<EdgeId>,
    /// The values whose pending versions get the epoch (`temporal`).
    #[cfg(feature = "temporal")]
    node_values: Vec<(NodeId, &'c PropertyKey)>,
    #[cfg(feature = "temporal")]
    edge_values: Vec<(EdgeId, &'c PropertyKey)>,
    /// The nodes whose pending label sets get the epoch (`temporal`).
    #[cfg(feature = "temporal")]
    label_nodes: Vec<NodeId>,
    /// The bulk ranges, stamped id by id.
    bulk: Vec<&'c BulkRange>,
    counts: Counts,
}

impl<'c> Stamp<'c> {
    /// Gathers what `change` stamps.
    ///
    /// # Errors
    ///
    /// A triple, a delete without its image or an edge type the store does
    /// not know: broken invariants.
    fn add(&mut self, store: &LpgStore, change: &'c Change) -> Result<(), ApplyError> {
        let (op, before) = match change {
            Change::Data { op, before, .. } => (op, before),
            Change::Bulk(range) => {
                self.bulk.push(range);
                return Ok(());
            }
        };
        match op {
            DataOp::CreateNode { id, properties, .. } => {
                self.nodes.push(*id);
                self.counts.nodes += 1;
                #[cfg(feature = "temporal")]
                {
                    self.label_nodes.push(*id);
                    self.node_values
                        .extend(properties.iter().map(|(key, _)| (*id, key)));
                }
                #[cfg(not(feature = "temporal"))]
                let _ = properties;
            }
            DataOp::DeleteNode { id } => {
                let Before::Node(image) = before else {
                    return Err(misfit(op, before));
                };
                self.counts.nodes -= 1;
                // The delete's empty label set and its null versions.
                #[cfg(feature = "temporal")]
                {
                    self.label_nodes.push(*id);
                    self.node_values
                        .extend(image.properties.iter().map(|(key, _)| (*id, key)));
                }
                #[cfg(not(feature = "temporal"))]
                let _ = (id, image);
            }
            DataOp::CreateEdge {
                id,
                edge_type,
                properties,
                ..
            } => {
                self.edges.push(*id);
                self.counts.edges += 1;
                let type_id = store
                    .edge_type_id(edge_type)
                    .ok_or(ApplyError::Missing(Entity::Edge(*id)))?;
                *self.counts.edge_types.entry(type_id).or_default() += 1;
                #[cfg(feature = "temporal")]
                self.edge_values
                    .extend(properties.iter().map(|(key, _)| (*id, key)));
                #[cfg(not(feature = "temporal"))]
                let _ = properties;
            }
            DataOp::DeleteEdge { id } => {
                let Before::Edge(image) = before else {
                    return Err(misfit(op, before));
                };
                self.counts.edges -= 1;
                let type_id = store
                    .edge_type_id(&image.edge_type)
                    .ok_or(ApplyError::Missing(Entity::Edge(*id)))?;
                *self.counts.edge_types.entry(type_id).or_default() -= 1;
                #[cfg(feature = "temporal")]
                self.edge_values
                    .extend(image.properties.iter().map(|(key, _)| (*id, key)));
            }
            // Without `temporal` values and labels change in place: nothing
            // of them is pending.
            DataOp::SetNodeProperty { id, key, .. } | DataOp::RemoveNodeProperty { id, key } => {
                #[cfg(feature = "temporal")]
                self.node_values.push((*id, key));
                #[cfg(not(feature = "temporal"))]
                let _ = (id, key);
            }
            DataOp::SetEdgeProperty { id, key, .. } | DataOp::RemoveEdgeProperty { id, key } => {
                #[cfg(feature = "temporal")]
                self.edge_values.push((*id, key));
                #[cfg(not(feature = "temporal"))]
                let _ = (id, key);
            }
            DataOp::AddNodeLabel { id, .. } | DataOp::RemoveNodeLabel { id, .. } => {
                #[cfg(feature = "temporal")]
                self.label_nodes.push(*id);
                #[cfg(not(feature = "temporal"))]
                let _ = id;
            }
            DataOp::InsertTriple { .. } | DataOp::DeleteTriple { .. } => {
                return Err(refuse_triple(op));
            }
        }
        Ok(())
    }
}

impl LpgStore {
    /// Gives the versions `stamp` gathered the commit epoch.
    fn stamp_versions(
        &self,
        transaction: TransactionId,
        stamp: &Stamp<'_>,
        epoch: EpochId,
    ) -> Result<(), ApplyError> {
        if !stamp.nodes.is_empty() {
            let mut map = self.node_version_map().write();
            for &id in &stamp.nodes {
                map.get_mut(&id)
                    .ok_or(ApplyError::Missing(Entity::Node(id)))?
                    .finalize_epochs(transaction, epoch);
            }
        }
        if !stamp.edges.is_empty() {
            let mut map = self.edge_version_map().write();
            for &id in &stamp.edges {
                map.get_mut(&id)
                    .ok_or(ApplyError::Missing(Entity::Edge(id)))?
                    .finalize_epochs(transaction, epoch);
            }
        }
        #[cfg(feature = "temporal")]
        self.stamp_pending_values(stamp, epoch)?;
        Ok(())
    }

    /// Gives the pending value and label set versions `stamp` gathered the
    /// commit epoch.
    #[cfg(feature = "temporal")]
    fn stamp_pending_values(&self, stamp: &Stamp<'_>, epoch: EpochId) -> Result<(), ApplyError> {
        if !stamp.node_values.is_empty() {
            let mut columns = self.node_properties.columns_write();
            for &(id, key) in &stamp.node_values {
                columns
                    .get_mut(key)
                    .ok_or(ApplyError::Missing(Entity::Node(id)))?
                    .finalize_pending_for(id, epoch);
            }
        }
        if !stamp.edge_values.is_empty() {
            let mut columns = self.edge_properties.columns_write();
            for &(id, key) in &stamp.edge_values {
                columns
                    .get_mut(key)
                    .ok_or(ApplyError::Missing(Entity::Edge(id)))?
                    .finalize_pending_for(id, epoch);
            }
        }
        if !stamp.label_nodes.is_empty() {
            let mut labels = self.node_labels.write();
            for &id in &stamp.label_nodes {
                labels
                    .get_mut(&id)
                    .ok_or(ApplyError::Missing(Entity::Node(id)))?
                    .finalize_pending(epoch);
            }
        }
        Ok(())
    }

    /// Stamps the rows of a bulk range the transaction created: an id the
    /// store does not hold, or holds from another writer, is skipped. Costs
    /// O(ids) and, with `temporal`, O(ids × columns): the range does not say
    /// which keys its rows have.
    fn stamp_bulk(
        &self,
        transaction: TransactionId,
        range: &BulkRange,
        epoch: EpochId,
        counts: &mut Counts,
    ) -> Result<(), ApplyError> {
        match range.table {
            Table::Nodes => {
                let mut stamped = Vec::new();
                {
                    let mut map = self.node_version_map().write();
                    for raw in range.ids.clone() {
                        let id = NodeId::new(raw);
                        if let Some(versions) = map.get_mut(&id)
                            && versions.modified_by(transaction)
                        {
                            versions.finalize_epochs(transaction, epoch);
                            stamped.push(id);
                        }
                    }
                }
                counts.nodes += i64::try_from(stamped.len()).map_err(|_| {
                    ApplyError::Refused("a bulk range of more nodes than a count holds".into())
                })?;
                #[cfg(feature = "temporal")]
                {
                    let mut labels = self.node_labels.write();
                    for id in &stamped {
                        if let Some(log) = labels.get_mut(id) {
                            log.finalize_pending(epoch);
                        }
                    }
                    drop(labels);
                    let mut columns = self.node_properties.columns_write();
                    for column in columns.values_mut() {
                        for id in &stamped {
                            column.finalize_pending_for(*id, epoch);
                        }
                    }
                }
            }
            Table::Edges => {
                let mut stamped = Vec::new();
                {
                    let mut map = self.edge_version_map().write();
                    for raw in range.ids.clone() {
                        let id = EdgeId::new(raw);
                        if let Some(versions) = map.get_mut(&id)
                            && versions.modified_by(transaction)
                        {
                            versions.finalize_epochs(transaction, epoch);
                            #[cfg(not(feature = "tiered-storage"))]
                            let record = versions.latest().copied();
                            #[cfg(feature = "tiered-storage")]
                            let record = versions
                                .latest()
                                .and_then(|version| self.read_edge_record(&version));
                            let record = record.ok_or(ApplyError::Missing(Entity::Edge(id)))?;
                            stamped.push((id, record.type_id));
                        }
                    }
                }
                for &(_, type_id) in &stamped {
                    counts.edges += 1;
                    *counts.edge_types.entry(type_id).or_default() += 1;
                }
                #[cfg(feature = "temporal")]
                {
                    let mut columns = self.edge_properties.columns_write();
                    for column in columns.values_mut() {
                        for (id, _) in &stamped {
                            column.finalize_pending_for(*id, epoch);
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

// ── undo ────────────────────────────────────────────────────────────

impl LpgStore {
    /// Undoes one entry of `transaction`.
    fn undo_one(
        &self,
        transaction: TransactionId,
        op: &DataOp,
        before: &Before,
        version: PendingVersion,
    ) -> Result<(), ApplyError> {
        // The store reports `Replaced` for a delete of what the transaction
        // created, and only then: anything else is a broken invariant.
        let own = match op {
            DataOp::DeleteNode { id } => self.node_created_by(*id, transaction),
            DataOp::DeleteEdge { id } => self.edge_created_by(*id, transaction),
            _ => false,
        };
        if version != version_of(own) {
            return Err(ApplyError::Refused(format!(
                "an entry of kind {} reports a {version:?} pending version, which this store \
                 did not report for it",
                op.kind()
            )));
        }
        match (op, before) {
            (DataOp::CreateNode { id, .. }, Before::Absent) => {
                if self.remove_created_node(*id, transaction) {
                    Ok(())
                } else {
                    Err(ApplyError::Missing(Entity::Node(*id)))
                }
            }
            (DataOp::CreateEdge { id, .. }, Before::Absent) => {
                match self.remove_created_edge(*id, transaction) {
                    Some(_) => Ok(()),
                    None => Err(ApplyError::Missing(Entity::Edge(*id))),
                }
            }
            (DataOp::DeleteNode { id }, Before::Node(image)) => {
                self.restore_node(*id, transaction, image)
            }
            (DataOp::DeleteEdge { id }, Before::Edge(image)) => {
                self.restore_edge(*id, transaction, image)
            }
            (
                DataOp::SetNodeProperty { id, key, .. } | DataOp::RemoveNodeProperty { id, key },
                Before::Value(old),
            ) => self.restore_node_value(*id, key, old.as_ref()),
            (
                DataOp::SetEdgeProperty { id, key, .. } | DataOp::RemoveEdgeProperty { id, key },
                Before::Value(old),
            ) => self.restore_edge_value(*id, key, old.as_ref()),
            (
                DataOp::AddNodeLabel { id, .. } | DataOp::RemoveNodeLabel { id, .. },
                Before::Labels(labels),
            ) => self.restore_labels(*id, labels),
            (DataOp::InsertTriple { .. } | DataOp::DeleteTriple { .. }, _) => {
                Err(refuse_triple(op))
            }
            _ => Err(misfit(op, before)),
        }
    }

    /// Takes back a delete of node `id`: its version, values and labels, and
    /// its entries in the label, property, text and vector indexes.
    fn restore_node(
        &self,
        id: NodeId,
        transaction: TransactionId,
        image: &NodeImage,
    ) -> Result<(), ApplyError> {
        let unmarked = self
            .node_version_map()
            .write()
            .get_mut(&id)
            .is_some_and(|versions| versions.unmark_deleted_by(transaction));
        if !unmarked {
            return Err(ApplyError::Missing(Entity::Node(id)));
        }
        // The values first, so the labels' text and vector indexes find
        // them; then the labels.
        #[cfg(not(feature = "temporal"))]
        let labels = {
            for (key, value) in &image.properties {
                self.node_properties.set(id, key.clone(), value.clone());
            }
            let labels = self.label_ids(id, &image.labels)?;
            self.node_labels.write().insert(id, labels.clone());
            labels
        };
        #[cfg(feature = "temporal")]
        let labels = {
            // The delete wrote a null version per value and an empty label
            // set: dropping them brings the values and labels back.
            {
                let mut columns = self.node_properties.columns_write();
                for (key, _) in &image.properties {
                    columns
                        .get_mut(key)
                        .ok_or(ApplyError::Missing(Entity::Node(id)))?
                        .pop_n_pending_for(id, 1);
                }
            }
            let mut node_labels = self.node_labels.write();
            let log = node_labels
                .get_mut(&id)
                .ok_or(ApplyError::Missing(Entity::Node(id)))?;
            log.pop_n_pending(1);
            log.latest().cloned().unwrap_or_default()
        };
        self.reindex_labels(id, &FxHashSet::default(), &labels);
        for (key, value) in &image.properties {
            self.update_property_index_on_set(id, key, value);
        }
        Ok(())
    }

    /// Takes back a delete of edge `id`. Its nodes are not deleted by the
    /// transaction: a node delete comes after the deletes of its edges, so
    /// its undo came first.
    fn restore_edge(
        &self,
        id: EdgeId,
        transaction: TransactionId,
        image: &EdgeImage,
    ) -> Result<(), ApplyError> {
        {
            let nodes = self.node_version_map().read();
            for end in [image.src, image.dst] {
                if nodes
                    .get(&end)
                    .is_some_and(|versions| versions.deleted_by(transaction))
                {
                    return Err(ApplyError::Missing(Entity::Node(end)));
                }
            }
        }
        let unmarked = self
            .edge_version_map()
            .write()
            .get_mut(&id)
            .is_some_and(|versions| versions.unmark_deleted_by(transaction));
        if !unmarked {
            return Err(ApplyError::Missing(Entity::Edge(id)));
        }
        self.forward_adj.unmark_deleted(image.src, id);
        if let Some(backward) = &self.backward_adj {
            backward.unmark_deleted(image.dst, id);
        }
        #[cfg(not(feature = "temporal"))]
        for (key, value) in &image.properties {
            self.edge_properties.set(id, key.clone(), value.clone());
        }
        #[cfg(feature = "temporal")]
        {
            let mut columns = self.edge_properties.columns_write();
            for (key, _) in &image.properties {
                columns
                    .get_mut(key)
                    .ok_or(ApplyError::Missing(Entity::Edge(id)))?
                    .pop_n_pending_for(id, 1);
            }
        }
        Ok(())
    }

    /// Takes back a set or a removal of a node's value: without `temporal`
    /// writes `old` back, with it drops the version the write appended. The
    /// property, text and vector indexes follow the value.
    fn restore_node_value(
        &self,
        id: NodeId,
        key: &PropertyKey,
        old: Option<&Value>,
    ) -> Result<(), ApplyError> {
        // A later delete of the node was undone first: the store holds it.
        if !self.node_version_map().read().contains_key(&id) {
            return Err(ApplyError::Missing(Entity::Node(id)));
        }
        #[cfg(not(feature = "temporal"))]
        match old {
            Some(value) => self.put_node_value(id, key, value.clone(), self.current_epoch()),
            None => self.undo_node_property_set(id, key),
        }
        #[cfg(feature = "temporal")]
        {
            let _ = old;
            let indexed = self.indexed_values(std::iter::once((id, key)));
            self.node_properties
                .columns_write()
                .get_mut(key)
                .ok_or(ApplyError::Missing(Entity::Node(id)))?
                .pop_n_pending_for(id, 1);
            self.reconcile_property_indexes(indexed);
            #[cfg(feature = "text-index")]
            match self.node_properties.get(id, key) {
                Some(value) => self.update_text_index_on_set(id, key.as_str(), &value),
                None => self.update_text_index_on_remove(id, key.as_str()),
            }
            #[cfg(feature = "vector-index")]
            self.sync_vector_indexes_for_property(id, key.as_str());
        }
        Ok(())
    }

    /// Takes back a set or a removal of an edge's value, as
    /// [`restore_node_value`](Self::restore_node_value) does a node's.
    fn restore_edge_value(
        &self,
        id: EdgeId,
        key: &PropertyKey,
        old: Option<&Value>,
    ) -> Result<(), ApplyError> {
        if !self.edge_version_map().read().contains_key(&id) {
            return Err(ApplyError::Missing(Entity::Edge(id)));
        }
        #[cfg(not(feature = "temporal"))]
        match old {
            Some(value) => self.edge_properties.set(id, key.clone(), value.clone()),
            None => self.undo_edge_property_set(id, key),
        }
        #[cfg(feature = "temporal")]
        {
            let _ = old;
            self.edge_properties
                .columns_write()
                .get_mut(key)
                .ok_or(ApplyError::Missing(Entity::Edge(id)))?
                .pop_n_pending_for(id, 1);
        }
        Ok(())
    }

    /// Takes back a label op on node `id`: without `temporal` writes the
    /// labels `before` back, with it drops the label set the op appended.
    /// The label, text and vector indexes follow.
    fn restore_labels(&self, id: NodeId, before: &Labels) -> Result<(), ApplyError> {
        #[cfg(not(feature = "temporal"))]
        let (from, to) = {
            let to = self.label_ids(id, before)?;
            let from = self
                .node_labels
                .write()
                .insert(id, to.clone())
                .unwrap_or_default();
            (from, to)
        };
        #[cfg(feature = "temporal")]
        let (from, to) = {
            let _ = before;
            let mut node_labels = self.node_labels.write();
            let log = node_labels
                .get_mut(&id)
                .ok_or(ApplyError::Missing(Entity::Node(id)))?;
            let from = log.latest().cloned().unwrap_or_default();
            log.pop_n_pending(1);
            (from, log.latest().cloned().unwrap_or_default())
        };
        self.reindex_labels(id, &from, &to);
        #[cfg(not(any(feature = "temporal", feature = "tiered-storage")))]
        self.update_label_count(id);
        Ok(())
    }

    /// Undoes a bulk range: removes each row the transaction created in it.
    /// An id the store does not hold, or holds from another writer, is
    /// skipped.
    fn undo_bulk(&self, transaction: TransactionId, range: &BulkRange) {
        for raw in range.ids.clone().rev() {
            match range.table {
                Table::Nodes => {
                    let id = NodeId::new(raw);
                    if self.node_created_by(id, transaction) {
                        self.remove_created_node(id, transaction);
                    }
                }
                Table::Edges => {
                    let id = EdgeId::new(raw);
                    if self.edge_created_by(id, transaction) {
                        self.remove_created_edge(id, transaction);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod testing;

#[cfg(test)]
mod tests;
