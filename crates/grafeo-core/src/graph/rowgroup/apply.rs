//! The row-group store's side of the change set: `apply`, `stamp` and
//! `undo` (change target, sections 3 and 6).
//!
//! A transaction's create writes a row stamped with the transaction; its
//! delete marks the row the same way; `stamp` overwrites those marks with
//! the commit epoch and moves the counts, `undo` takes the rows out or the
//! marks off. Values and labels are written in place, and undo writes the
//! before-image back (H2b replaces this with update chains). Replay and an
//! immediate write stamp as they apply, through the same functions.

use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};

use grafeo_common::change::{
    Before, BulkRange, Change, DataOp, Entity, Labels, PendingVersion, Properties, Table,
};
use grafeo_common::types::{ArcStr, EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};

use super::{ABSENT, Adjacent, Inner, PENDING, Read, RowGroupStore, RowVersion, locate};
use crate::graph::apply::{Applied, ApplyError, ChangeTarget, Writer};

/// How a write is applied.
#[derive(Debug, Clone, Copy)]
struct Mode {
    writer: Writer,
    /// What the writer sees.
    read: Read,
    /// The stamp of what it writes: its transaction's, or the epoch of a
    /// write stamped at once.
    stamp: u64,
    /// The epoch of a write stamped at once (replay, an immediate write).
    epoch: Option<EpochId>,
    /// Whether it builds before-images.
    images: bool,
}

impl Mode {
    fn of(writer: Writer) -> Result<Self, ApplyError> {
        match writer {
            Writer::Transaction { id, snapshot } => Ok(Self {
                writer,
                read: Read::transaction(id, snapshot),
                stamp: pending_stamp(id)?,
                epoch: None,
                images: true,
            }),
            Writer::Replay { epoch } | Writer::Immediate { epoch, .. } => {
                if epoch.as_u64() >= PENDING {
                    return Err(ApplyError::Refused(format!(
                        "epoch {} is past the epochs a row group can stamp",
                        epoch.as_u64()
                    )));
                }
                Ok(Self {
                    writer,
                    read: Read::at(epoch),
                    stamp: epoch.as_u64(),
                    epoch: Some(epoch),
                    images: writer.builds_images(),
                })
            }
        }
    }

    fn stamped(self) -> bool {
        self.epoch.is_some()
    }

    /// What `apply` returns for a write that changed something.
    fn applied(self, before: impl FnOnce() -> Before, version: PendingVersion) -> Applied {
        if self.images {
            Applied::Changed {
                before: before(),
                version,
            }
        } else {
            Applied::Committed
        }
    }

    /// The version a delete reports: `Replaced` when the transaction deletes
    /// what it created itself (undo restores its pending create).
    fn version_of(self, created: u64) -> PendingVersion {
        if !self.stamped() && created == self.stamp {
            PendingVersion::Replaced
        } else {
            PendingVersion::Created
        }
    }
}

/// The stamp of transaction `id`'s pending versions.
fn pending_stamp(id: TransactionId) -> Result<u64, ApplyError> {
    if id == TransactionId::SYSTEM {
        return Err(ApplyError::Refused(
            "a transaction's write needs the transaction's id, not the system one".to_string(),
        ));
    }
    if id.as_u64() >= PENDING - 1 {
        return Err(ApplyError::Refused(format!(
            "transaction {} is past the ids a row group can stamp",
            id.as_u64()
        )));
    }
    Ok(PENDING | id.as_u64())
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

/// The error of a broken invariant: an entry the store has no pending
/// version for.
fn not_pending(entity: Entity, what: &str) -> ApplyError {
    ApplyError::Refused(format!(
        "{} has no pending {what} of this transaction",
        crate::graph::apply::describe(entity)
    ))
}

/// The error of an entry whose before-image does not fit its op.
fn misfit(op: &DataOp, before: &Before) -> ApplyError {
    ApplyError::Refused(format!(
        "an entry of kind {} with a before-image that does not fit it: {before:?}",
        op.kind()
    ))
}

fn refuse_triple(op: &DataOp) -> ApplyError {
    ApplyError::Refused(format!(
        "{} is an RDF op; a labeled property graph's store does not take it",
        op.kind()
    ))
}

impl Inner {
    /// Label ids for `labels`, ascending without repeats, created as needed.
    fn label_ids_for(&mut self, labels: &Labels) -> Vec<u32> {
        let mut ids: Vec<u32> = labels
            .iter()
            .map(|label| self.labels.get_or_create(label))
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Key ids for `properties`, created as needed.
    fn key_ids_for<'a>(&mut self, properties: &'a Properties) -> Vec<(u32, &'a Value)> {
        properties
            .iter()
            .map(|(key, value)| (self.keys.get_or_create(key.as_str()), value))
            .collect()
    }

    /// The ids of label names the dictionary holds (it forgets none).
    fn known_label_ids(&self, labels: &Labels) -> Result<Vec<u32>, ApplyError> {
        let mut ids = labels
            .iter()
            .map(|label| {
                self.label_id(label).ok_or_else(|| {
                    ApplyError::Refused(format!("label {label} is not in the dictionary"))
                })
            })
            .collect::<Result<Vec<u32>, ApplyError>>()?;
        ids.sort_unstable();
        ids.dedup();
        Ok(ids)
    }

    fn create_node(
        &mut self,
        id: NodeId,
        labels: &Labels,
        properties: &Properties,
        mode: Mode,
    ) -> Result<Applied, ApplyError> {
        let raw = id.as_u64();
        if self.node(raw).is_some() {
            return Err(ApplyError::Exists(Entity::Node(id)));
        }
        let label_ids = self.label_ids_for(labels);
        let values = self.key_ids_for(properties);
        let (group, row) = locate(raw);
        let group = self.nodes.entry(group).or_default();
        *group.version_mut(row) = RowVersion {
            created: mode.stamp,
            deleted: ABSENT,
        };
        group.set_labels(row, &label_ids);
        for (key, value) in values {
            group.columns.set(key, row, value);
        }
        if mode.stamped() {
            self.counts.nodes += 1;
            for label in &label_ids {
                self.counts.label(*label, 1);
            }
        }
        Ok(mode.applied(|| Before::Absent, PendingVersion::Created))
    }

    fn create_edge(
        &mut self,
        id: EdgeId,
        src: NodeId,
        dst: NodeId,
        edge_type: &ArcStr,
        properties: &Properties,
        mode: Mode,
    ) -> Result<Applied, ApplyError> {
        let raw = id.as_u64();
        if self.edge(raw).is_some() {
            return Err(ApplyError::Exists(Entity::Edge(id)));
        }
        for end in [src, dst] {
            if !self.node_visible(end.as_u64(), mode.read) {
                return Err(ApplyError::Missing(Entity::Node(end)));
            }
        }
        let type_id = self.edge_types.get_or_create(edge_type);
        let values = self.key_ids_for(properties);
        let (group, row) = locate(raw);
        let group = self.edges.entry(group).or_default();
        *group.version_mut(row) = RowVersion {
            created: mode.stamp,
            deleted: ABSENT,
        };
        group.set_ends(row, src.as_u64(), dst.as_u64(), type_id);
        for (key, value) in values {
            group.columns.set(key, row, value);
        }
        for (node, other, outgoing) in [(src, dst, true), (dst, src, false)] {
            let (group, row) = locate(node.as_u64());
            self.nodes
                .get_mut(&group)
                .ok_or(ApplyError::Missing(Entity::Node(node)))?
                .adjacency_mut(outgoing)
                .push(
                    row,
                    Adjacent {
                        edge_type: type_id,
                        other: other.as_u64(),
                        edge: raw,
                    },
                );
        }
        if mode.stamped() {
            self.counts.edges += 1;
            self.counts.edge_type(type_id, 1);
        }
        Ok(mode.applied(|| Before::Absent, PendingVersion::Created))
    }

    fn delete_node(&mut self, id: NodeId, mode: Mode) -> Result<Applied, ApplyError> {
        let raw = id.as_u64();
        let Some((group, row)) = self.node(raw) else {
            return mode.writer.unseen(Entity::Node(id));
        };
        let version = group.version(row);
        if !version.visible(mode.read) {
            return mode.writer.unseen(Entity::Node(id));
        }
        if version.deleted != ABSENT {
            // Another transaction deletes it: the claims never let two.
            return Err(ApplyError::Missing(Entity::Node(id)));
        }
        if self.sees_an_edge_of(raw, mode.read) {
            return Err(ApplyError::HasEdges(id));
        }
        let labels = group.labels_of(row);
        let image = if mode.images {
            self.node_image(raw)
        } else {
            None
        };
        let (group, row) = self
            .node_mut(raw)
            .ok_or(ApplyError::Missing(Entity::Node(id)))?;
        group.version_mut(row).deleted = mode.stamp;
        if mode.stamped() {
            self.counts.nodes -= 1;
            for label in labels {
                self.counts.label(label, -1);
            }
        }
        Ok(mode.applied(
            || Before::Node(Box::new(image.expect("the image of a node it read"))),
            mode.version_of(version.created),
        ))
    }

    fn delete_edge(&mut self, id: EdgeId, mode: Mode) -> Result<Applied, ApplyError> {
        let raw = id.as_u64();
        let Some((group, row)) = self.edge(raw) else {
            return mode.writer.unseen(Entity::Edge(id));
        };
        let version = group.version(row);
        if !version.visible(mode.read) {
            return mode.writer.unseen(Entity::Edge(id));
        }
        if version.deleted != ABSENT {
            return Err(ApplyError::Missing(Entity::Edge(id)));
        }
        let (_, _, type_id) = group.ends(row);
        let image = if mode.images {
            self.edge_image(raw)
        } else {
            None
        };
        let (group, row) = self
            .edge_mut(raw)
            .ok_or(ApplyError::Missing(Entity::Edge(id)))?;
        group.version_mut(row).deleted = mode.stamp;
        if mode.stamped() {
            self.counts.edges -= 1;
            self.counts.edge_type(type_id, -1);
        }
        Ok(mode.applied(
            || Before::Edge(Box::new(image.expect("the image of an edge it read"))),
            mode.version_of(version.created),
        ))
    }
}

impl ChangeTarget for RowGroupStore {
    fn reserve_node_ids(&self, count: u64) -> Result<Range<u64>, ApplyError> {
        reserve(&self.next_node_id, count, "node")
    }

    fn reserve_edge_ids(&self, count: u64) -> Result<Range<u64>, ApplyError> {
        reserve(&self.next_edge_id, count, "edge")
    }

    fn apply(&self, op: &DataOp, writer: Writer) -> Result<Applied, ApplyError> {
        let mode = Mode::of(writer)?;
        let applied = {
            let mut inner = self.inner.write();
            match op {
                DataOp::CreateNode {
                    id,
                    labels,
                    properties,
                } => {
                    let applied = inner.create_node(*id, labels, properties, mode)?;
                    self.next_node_id
                        .fetch_max(id.as_u64() + 1, Ordering::AcqRel);
                    applied
                }
                DataOp::CreateEdge {
                    id,
                    src,
                    dst,
                    edge_type,
                    properties,
                } => {
                    let applied =
                        inner.create_edge(*id, *src, *dst, edge_type, properties, mode)?;
                    self.next_edge_id
                        .fetch_max(id.as_u64() + 1, Ordering::AcqRel);
                    applied
                }
                DataOp::DeleteNode { id } => inner.delete_node(*id, mode)?,
                DataOp::DeleteEdge { id } => inner.delete_edge(*id, mode)?,
                DataOp::SetNodeProperty { id, key, value } => {
                    inner.write_value(Entity::Node(*id), key, Some(value), mode)?
                }
                DataOp::RemoveNodeProperty { id, key } => {
                    inner.write_value(Entity::Node(*id), key, None, mode)?
                }
                DataOp::SetEdgeProperty { id, key, value } => {
                    inner.write_value(Entity::Edge(*id), key, Some(value), mode)?
                }
                DataOp::RemoveEdgeProperty { id, key } => {
                    inner.write_value(Entity::Edge(*id), key, None, mode)?
                }
                DataOp::AddNodeLabel { id, label } => inner.write_label(*id, label, true, mode)?,
                DataOp::RemoveNodeLabel { id, label } => {
                    inner.write_label(*id, label, false, mode)?
                }
                DataOp::InsertTriple { .. } | DataOp::DeleteTriple { .. } => {
                    return Err(refuse_triple(op));
                }
            }
        };
        if let Some(epoch) = mode.epoch {
            self.sync_epoch(epoch);
        }
        Ok(applied)
    }

    fn stamp(
        &self,
        transaction: TransactionId,
        entries: &mut dyn Iterator<Item = &Change>,
        epoch: EpochId,
    ) -> Result<(), ApplyError> {
        let pending = pending_stamp(transaction)?;
        let at = epoch.as_u64();
        {
            let mut inner = self.inner.write();
            for change in entries {
                match change {
                    Change::Data { op, before, .. } => inner.stamp_op(op, before, pending, at)?,
                    Change::Bulk(range) => inner.stamp_bulk(range, pending, at),
                }
            }
        }
        self.sync_epoch(epoch);
        Ok(())
    }

    fn undo(
        &self,
        transaction: TransactionId,
        entries: &mut dyn DoubleEndedIterator<Item = &Change>,
    ) -> Result<(), ApplyError> {
        let pending = pending_stamp(transaction)?;
        let mut inner = self.inner.write();
        while let Some(change) = entries.next_back() {
            match change {
                Change::Data { op, before, .. } => inner.undo_op(op, before, pending)?,
                Change::Bulk(range) => inner.undo_bulk(range, pending),
            }
        }
        Ok(())
    }
}

impl Inner {
    /// Sets (`Some`) or removes (`None`) a value of a node or an edge.
    fn write_value(
        &mut self,
        entity: Entity,
        key: &PropertyKey,
        value: Option<&Value>,
        mode: Mode,
    ) -> Result<Applied, ApplyError> {
        let seen = match entity {
            Entity::Node(id) => self.node_visible(id.as_u64(), mode.read),
            Entity::Edge(id) => self.edge_visible(id.as_u64(), mode.read),
        };
        if !seen {
            return mode.writer.unseen(entity);
        }
        let key = match value {
            Some(_) => self.keys.get_or_create(key.as_str()),
            None => match self.keys.get_id(key.as_str()) {
                Some(key) => key,
                None => return mode.writer.no_op(ApplyError::Missing(entity)),
            },
        };
        let (columns, row) = match entity {
            Entity::Node(id) => {
                let (group, row) = self
                    .node_mut(id.as_u64())
                    .ok_or(ApplyError::Missing(entity))?;
                (&mut group.columns, row)
            }
            Entity::Edge(id) => {
                let (group, row) = self
                    .edge_mut(id.as_u64())
                    .ok_or(ApplyError::Missing(entity))?;
                (&mut group.columns, row)
            }
        };
        match value {
            Some(value) => {
                let old = if mode.images {
                    columns.get(key, row)
                } else {
                    None
                };
                columns.set(key, row, value);
                Ok(mode.applied(|| Before::Value(old), PendingVersion::Created))
            }
            None => match columns.take(key, row) {
                Some(old) => Ok(mode.applied(|| Before::Value(Some(old)), PendingVersion::Created)),
                None => mode.writer.no_op(ApplyError::Missing(entity)),
            },
        }
    }

    /// Adds (`add`) or removes a node's label.
    fn write_label(
        &mut self,
        id: NodeId,
        label: &ArcStr,
        add: bool,
        mode: Mode,
    ) -> Result<Applied, ApplyError> {
        let raw = id.as_u64();
        if !self.node_visible(raw, mode.read) {
            return mode.writer.unseen(Entity::Node(id));
        }
        let label_id = if add {
            self.labels.get_or_create(label)
        } else {
            match self.label_id(label) {
                Some(label_id) => label_id,
                None => return mode.writer.no_op(ApplyError::Missing(Entity::Node(id))),
            }
        };
        let (group, row) = self
            .node(raw)
            .ok_or(ApplyError::Missing(Entity::Node(id)))?;
        let has = group
            .labels
            .get(&label_id)
            .is_some_and(|rows| rows.get(row));
        if has == add {
            let error = if add {
                ApplyError::Exists(Entity::Node(id))
            } else {
                ApplyError::Missing(Entity::Node(id))
            };
            return mode.writer.no_op(error);
        }
        let before = if mode.images {
            self.label_names(&group.labels_of(row))
        } else {
            Labels::new()
        };
        let (group, row) = self
            .node_mut(raw)
            .ok_or(ApplyError::Missing(Entity::Node(id)))?;
        group.labels.entry(label_id).or_default().put(row, add);
        if mode.stamped() {
            self.counts.label(label_id, if add { 1 } else { -1 });
        }
        Ok(mode.applied(|| Before::Labels(before), PendingVersion::Created))
    }

    /// Commits one entry of transaction `pending` at epoch `at`.
    fn stamp_op(
        &mut self,
        op: &DataOp,
        before: &Before,
        pending: u64,
        at: u64,
    ) -> Result<(), ApplyError> {
        match op {
            DataOp::CreateNode { id, labels, .. } => {
                let label_ids = self.known_label_ids(labels)?;
                let (group, row) = self
                    .node_mut(id.as_u64())
                    .ok_or_else(|| not_pending(Entity::Node(*id), "create"))?;
                let version = group.version_mut(row);
                if version.created != pending {
                    return Err(not_pending(Entity::Node(*id), "create"));
                }
                version.created = at;
                self.counts.nodes += 1;
                for label in label_ids {
                    self.counts.label(label, 1);
                }
            }
            DataOp::DeleteNode { id } => {
                let Before::Node(image) = before else {
                    return Err(misfit(op, before));
                };
                let label_ids = self.known_label_ids(&image.labels)?;
                let (group, row) = self
                    .node_mut(id.as_u64())
                    .ok_or_else(|| not_pending(Entity::Node(*id), "delete"))?;
                let version = group.version_mut(row);
                if version.deleted != pending {
                    return Err(not_pending(Entity::Node(*id), "delete"));
                }
                version.deleted = at;
                self.counts.nodes -= 1;
                for label in label_ids {
                    self.counts.label(label, -1);
                }
            }
            DataOp::CreateEdge { id, .. } => {
                let (group, row) = self
                    .edge_mut(id.as_u64())
                    .ok_or_else(|| not_pending(Entity::Edge(*id), "create"))?;
                let (_, _, type_id) = group.ends(row);
                let version = group.version_mut(row);
                if version.created != pending {
                    return Err(not_pending(Entity::Edge(*id), "create"));
                }
                version.created = at;
                self.counts.edges += 1;
                self.counts.edge_type(type_id, 1);
            }
            DataOp::DeleteEdge { id } => {
                let (group, row) = self
                    .edge_mut(id.as_u64())
                    .ok_or_else(|| not_pending(Entity::Edge(*id), "delete"))?;
                let (_, _, type_id) = group.ends(row);
                let version = group.version_mut(row);
                if version.deleted != pending {
                    return Err(not_pending(Entity::Edge(*id), "delete"));
                }
                version.deleted = at;
                self.counts.edges -= 1;
                self.counts.edge_type(type_id, -1);
            }
            DataOp::AddNodeLabel { label, .. } | DataOp::RemoveNodeLabel { label, .. } => {
                let label_id = self.label_id(label).ok_or_else(|| {
                    ApplyError::Refused(format!("label {label} is not in the dictionary"))
                })?;
                let delta = if matches!(op, DataOp::AddNodeLabel { .. }) {
                    1
                } else {
                    -1
                };
                self.counts.label(label_id, delta);
            }
            // Values are written in place: nothing to stamp (H1a).
            DataOp::SetNodeProperty { .. }
            | DataOp::RemoveNodeProperty { .. }
            | DataOp::SetEdgeProperty { .. }
            | DataOp::RemoveEdgeProperty { .. } => {}
            DataOp::InsertTriple { .. } | DataOp::DeleteTriple { .. } => {
                return Err(refuse_triple(op));
            }
        }
        Ok(())
    }

    /// Commits the rows transaction `pending` created in a bulk range;
    /// absent ids are skipped.
    fn stamp_bulk(&mut self, range: &BulkRange, pending: u64, at: u64) {
        for raw in range.ids.clone() {
            match range.table {
                Table::Nodes => {
                    let Some((group, row)) = self.node_mut(raw) else {
                        continue;
                    };
                    let version = group.version_mut(row);
                    if version.created != pending {
                        continue;
                    }
                    version.created = at;
                    let labels = group.labels_of(row);
                    self.counts.nodes += 1;
                    for label in labels {
                        self.counts.label(label, 1);
                    }
                }
                Table::Edges => {
                    let Some((group, row)) = self.edge_mut(raw) else {
                        continue;
                    };
                    let (_, _, type_id) = group.ends(row);
                    let version = group.version_mut(row);
                    if version.created != pending {
                        continue;
                    }
                    version.created = at;
                    self.counts.edges += 1;
                    self.counts.edge_type(type_id, 1);
                }
            }
        }
    }

    /// Takes an edge row out with its two adjacency entries.
    pub(super) fn remove_edge_row(&mut self, raw: u64) {
        let Some((group, row)) = self.edge_mut(raw) else {
            return;
        };
        let (src, dst, _) = group.ends(row);
        group.remove(row);
        for (node, outgoing) in [(src, true), (dst, false)] {
            let (group, row) = locate(node);
            if let Some(group) = self.nodes.get_mut(&group) {
                group.adjacency_mut(outgoing).remove(row, raw);
            }
        }
    }

    /// Undoes one entry of transaction `pending`.
    fn undo_op(&mut self, op: &DataOp, before: &Before, pending: u64) -> Result<(), ApplyError> {
        match (op, before) {
            (DataOp::CreateNode { id, .. }, Before::Absent) => {
                let (group, row) = self
                    .node_mut(id.as_u64())
                    .ok_or_else(|| not_pending(Entity::Node(*id), "create"))?;
                if group.version(row).created != pending {
                    return Err(not_pending(Entity::Node(*id), "create"));
                }
                group.remove(row);
            }
            (DataOp::CreateEdge { id, .. }, Before::Absent) => {
                let created = self
                    .edge(id.as_u64())
                    .map(|(group, row)| group.version(row).created);
                if created != Some(pending) {
                    return Err(not_pending(Entity::Edge(*id), "create"));
                }
                self.remove_edge_row(id.as_u64());
            }
            (DataOp::DeleteNode { id }, Before::Node(_)) => {
                let (group, row) = self
                    .node_mut(id.as_u64())
                    .ok_or_else(|| not_pending(Entity::Node(*id), "delete"))?;
                let version = group.version_mut(row);
                if version.deleted != pending {
                    return Err(not_pending(Entity::Node(*id), "delete"));
                }
                version.deleted = ABSENT;
            }
            (DataOp::DeleteEdge { id }, Before::Edge(_)) => {
                let (group, row) = self
                    .edge_mut(id.as_u64())
                    .ok_or_else(|| not_pending(Entity::Edge(*id), "delete"))?;
                let version = group.version_mut(row);
                if version.deleted != pending {
                    return Err(not_pending(Entity::Edge(*id), "delete"));
                }
                version.deleted = ABSENT;
            }
            (
                DataOp::SetNodeProperty { id, key, .. } | DataOp::RemoveNodeProperty { id, key },
                Before::Value(old),
            ) => self.restore_value(Entity::Node(*id), key, old.as_ref())?,
            (
                DataOp::SetEdgeProperty { id, key, .. } | DataOp::RemoveEdgeProperty { id, key },
                Before::Value(old),
            ) => self.restore_value(Entity::Edge(*id), key, old.as_ref())?,
            (
                DataOp::AddNodeLabel { id, .. } | DataOp::RemoveNodeLabel { id, .. },
                Before::Labels(labels),
            ) => {
                let label_ids = self.known_label_ids(labels)?;
                let (group, row) = self
                    .node_mut(id.as_u64())
                    .ok_or(ApplyError::Missing(Entity::Node(*id)))?;
                group.set_labels(row, &label_ids);
            }
            (DataOp::InsertTriple { .. } | DataOp::DeleteTriple { .. }, _) => {
                return Err(refuse_triple(op));
            }
            _ => return Err(misfit(op, before)),
        }
        Ok(())
    }

    /// Writes a before-image's value back: `Some` sets it, `None` removes it.
    fn restore_value(
        &mut self,
        entity: Entity,
        key: &PropertyKey,
        old: Option<&Value>,
    ) -> Result<(), ApplyError> {
        let key_id = self.keys.get_id(key.as_str()).ok_or_else(|| {
            ApplyError::Refused(format!("property key {key} is not in the dictionary"))
        })?;
        let (columns, row) = match entity {
            Entity::Node(id) => {
                let (group, row) = self
                    .node_mut(id.as_u64())
                    .ok_or(ApplyError::Missing(entity))?;
                (&mut group.columns, row)
            }
            Entity::Edge(id) => {
                let (group, row) = self
                    .edge_mut(id.as_u64())
                    .ok_or(ApplyError::Missing(entity))?;
                (&mut group.columns, row)
            }
        };
        match old {
            Some(value) => columns.set(key_id, row, value),
            None => {
                columns.take(key_id, row);
            }
        }
        Ok(())
    }

    /// Takes out the rows transaction `pending` created in a bulk range,
    /// last first; absent ids are skipped.
    fn undo_bulk(&mut self, range: &BulkRange, pending: u64) {
        for raw in range.ids.clone().rev() {
            match range.table {
                Table::Nodes => {
                    if let Some((group, row)) = self.node_mut(raw)
                        && group.version(row).created == pending
                    {
                        group.remove(row);
                    }
                }
                Table::Edges => {
                    if self
                        .edge(raw)
                        .is_some_and(|(group, row)| group.version(row).created == pending)
                    {
                        self.remove_edge_row(raw);
                    }
                }
            }
        }
    }
}
