//! The row-group store's side of the change set: `apply`, `stamp` and
//! `undo` (change target, sections 3 and 6).
//!
//! A transaction's create writes a row stamped with the transaction; its
//! delete marks the row the same way; `stamp` overwrites those marks with
//! the commit epoch and moves the counts, `undo` takes the rows out or the
//! marks off. A value or label write puts the newest state in place and
//! first keeps what it replaces in the transaction's undo node of the
//! vector (`updates`), unless the transaction created the row itself (no
//! one else sees it) or a node of the transaction holds it already (the
//! write reports [`PendingVersion::Replaced`]); `stamp` gives the node the
//! commit epoch, `undo` writes the kept state back and takes it out of the
//! node, or writes the before-image back for a replaced version. Replay and
//! an immediate write stamp as they apply, through the same functions:
//! replay keeps nothing (no reader runs during recovery) unless the store
//! keeps history (`history`), an immediate write keeps what it replaces in a
//! node stamped with its epoch, for the readers at earlier epochs, until
//! `collect`, which the engine asks for after each write at its read
//! horizon. Every write that stamps a vector or may keep
//! something lists its vector for `collect`, and every write stamped at once
//! and every entry a commit stamps counts toward the next collect after a
//! write (`collect`). A property set to
//! null removes the property: the store holds no null (#412 decision Q8).

use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};

use grafeo_common::change::{
    Before, BulkRange, Change, DataOp, Entity, Labels, PendingVersion, Properties, Table,
};
use grafeo_common::types::{ArcStr, EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};

use super::column::Columns;
use super::versions::VECTOR_ROWS;
use super::{
    ABSENT, Adjacent, Inner, PENDING, ROWS_PER_GROUP, Read, RowGroupStore, label_ids, locate,
};
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
    /// Whether the store keeps history: replay then keeps what it replaces
    /// (`history`).
    history: bool,
}

impl Mode {
    /// How `writer` writes, in a store that keeps history (`history`) or
    /// not.
    fn of(writer: Writer, history: bool) -> Result<Self, ApplyError> {
        match writer {
            Writer::Transaction { id, snapshot } => Ok(Self {
                writer,
                read: Read::transaction(id, snapshot),
                stamp: pending_stamp(id)?,
                epoch: None,
                images: true,
                history,
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
                    history,
                })
            }
        }
    }

    fn stamped(self) -> bool {
        self.epoch.is_some()
    }

    /// Whether a value or label write of a row created by `created` keeps
    /// what it replaces, for the readers that must not see it, and under
    /// which stamp and snapshot: a transaction's write of a row it did not
    /// create; an immediate write of a row an earlier epoch created, for
    /// the readers before the write's epoch, whether or not one runs (a
    /// collect after the write drops it once one is due, when none does;
    /// #412 S6, decision B); replay only in a store that keeps history, as
    /// an immediate write keeps it, for the readers of the history the log
    /// holds (no reader runs during recovery otherwise).
    fn keeps(self, created: u64) -> Option<(u64, u64)> {
        if created == self.stamp {
            return None;
        }
        match self.writer {
            Writer::Transaction { snapshot, .. } => Some((self.stamp, snapshot.as_u64())),
            Writer::Immediate { epoch, .. } => Some((self.stamp, epoch.as_u64())),
            Writer::Replay { epoch } => self.history.then_some((self.stamp, epoch.as_u64())),
        }
    }

    /// The vector `op` makes versions in, which the store lists for
    /// `collect`: a create's or a delete's (it stamps the vector), a value
    /// or label write's but in replay without history (which keeps
    /// nothing). `None` for an RDF op.
    fn lists(self, op: &DataOp) -> Option<(Table, u64)> {
        let replay = matches!(self.writer, Writer::Replay { .. }) && !self.history;
        match op {
            DataOp::CreateNode { id, .. } | DataOp::DeleteNode { id } => {
                Some((Table::Nodes, id.as_u64()))
            }
            DataOp::CreateEdge { id, .. } | DataOp::DeleteEdge { id } => {
                Some((Table::Edges, id.as_u64()))
            }
            DataOp::SetNodeProperty { id, .. }
            | DataOp::RemoveNodeProperty { id, .. }
            | DataOp::AddNodeLabel { id, .. }
            | DataOp::RemoveNodeLabel { id, .. } => {
                (!replay).then_some((Table::Nodes, id.as_u64()))
            }
            DataOp::SetEdgeProperty { id, .. } | DataOp::RemoveEdgeProperty { id, .. } => {
                (!replay).then_some((Table::Edges, id.as_u64()))
            }
            DataOp::InsertTriple { .. } | DataOp::DeleteTriple { .. } => None,
        }
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

/// The vectors a bulk range spans, at most: what its commit may stamp.
fn bulk_vectors(range: &BulkRange) -> usize {
    let rows = range.ids.end.saturating_sub(range.ids.start);
    usize::try_from(rows / VECTOR_ROWS as u64 + 1).unwrap_or(usize::MAX)
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

/// Checks what `stamp` found of a value or label write of transaction
/// `pending`: the newest undo node of the row is the transaction's (or was
/// stamped `at` by an earlier entry of the commit), or the transaction
/// created the row and kept nothing.
fn check_kept(
    entity: Entity,
    kept: Option<u64>,
    created: u64,
    pending: u64,
    at: u64,
    what: &str,
) -> Result<(), ApplyError> {
    let own_node = kept.is_some_and(|stamp| stamp == pending || stamp == at);
    if own_node || created == pending || created == at {
        Ok(())
    } else {
        Err(not_pending(entity, what))
    }
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

    /// Key ids for `properties`, created as needed; a null is no value
    /// (#412 decision Q8), so it is left out.
    fn key_ids_for<'a>(&mut self, properties: &'a Properties) -> Vec<(u32, &'a Value)> {
        properties
            .iter()
            .filter(|(_, value)| !value.is_null())
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
        group.versions.set_created(row, mode.stamp);
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
        if !self.indexes.is_empty() {
            self.index_created(raw);
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
        group.versions.set_created(row, mode.stamp);
        group.set_ends(row, src.as_u64(), dst.as_u64(), type_id);
        for (key, value) in values {
            group.columns.set(key, row, value);
        }
        for (node, other, outgoing) in self.kept_ends(src.as_u64(), dst.as_u64()) {
            let (group, row) = locate(node);
            self.nodes
                .get_mut(&group)
                .ok_or(ApplyError::Missing(Entity::Node(NodeId::new(node))))?
                .adjacency_mut(outgoing)
                .push(
                    row,
                    Adjacent {
                        edge_type: type_id,
                        other,
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
        group.versions.set_deleted(row, mode.stamp);
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
        group.versions.set_deleted(row, mode.stamp);
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

impl Inner {
    /// Applies `op` as `mode` writes.
    fn apply_op(&mut self, op: &DataOp, mode: Mode) -> Result<Applied, ApplyError> {
        match op {
            DataOp::CreateNode {
                id,
                labels,
                properties,
            } => self.create_node(*id, labels, properties, mode),
            DataOp::CreateEdge {
                id,
                src,
                dst,
                edge_type,
                properties,
            } => self.create_edge(*id, *src, *dst, edge_type, properties, mode),
            DataOp::DeleteNode { id } => self.delete_node(*id, mode),
            DataOp::DeleteEdge { id } => self.delete_edge(*id, mode),
            DataOp::SetNodeProperty { id, key, value } => {
                self.write_value(Entity::Node(*id), key, Some(value), mode)
            }
            DataOp::RemoveNodeProperty { id, key } => {
                self.write_value(Entity::Node(*id), key, None, mode)
            }
            DataOp::SetEdgeProperty { id, key, value } => {
                self.write_value(Entity::Edge(*id), key, Some(value), mode)
            }
            DataOp::RemoveEdgeProperty { id, key } => {
                self.write_value(Entity::Edge(*id), key, None, mode)
            }
            DataOp::AddNodeLabel { id, label } => self.write_label(*id, label, true, mode),
            DataOp::RemoveNodeLabel { id, label } => self.write_label(*id, label, false, mode),
            DataOp::InsertTriple { .. } | DataOp::DeleteTriple { .. } => Err(refuse_triple(op)),
        }
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
        let mode = Mode::of(writer, self.keeps_history())?;
        self.refuse_if_dropped()?;
        let applied = {
            let mut inner = self.inner.write();
            let applied = inner.apply_op(op, mode);
            // Listed also when the op failed: one that failed halfway may
            // have stamped its vector, and a vector listed with nothing in
            // it leaves the list at the next collect.
            if !matches!(applied, Ok(Applied::Unchanged))
                && let Some((table, raw)) = mode.lists(op)
            {
                if inner.list(table, raw) {
                    self.note_listed();
                }
                // Stamped at once: what it kept or stamped waits for a
                // reader before its epoch alone.
                if mode.stamped() {
                    self.note_settled(1);
                }
            }
            applied?
        };
        match op {
            DataOp::CreateNode { id, .. } => {
                self.next_node_id
                    .fetch_max(id.as_u64() + 1, Ordering::AcqRel);
            }
            DataOp::CreateEdge { id, .. } => {
                self.next_edge_id
                    .fetch_max(id.as_u64() + 1, Ordering::AcqRel);
            }
            _ => {}
        }
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
            let mut settled = 0;
            let mut stamped = Ok(());
            for change in entries {
                match change {
                    Change::Data { op, before, .. } => {
                        settled += 1;
                        stamped = inner.stamp_op(op, before, pending, at);
                    }
                    Change::Bulk(range) => {
                        settled += bulk_vectors(range);
                        inner.stamp_bulk(range, pending, at);
                    }
                }
                if stamped.is_err() {
                    break;
                }
            }
            // Counted also when a stamp failed halfway: what it stamped
            // waits for the collect like the rest.
            self.note_settled(settled);
            stamped?;
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
                Change::Data {
                    op,
                    before,
                    version,
                    ..
                } => inner.undo_op(op, before, *version, pending)?,
                Change::Bulk(range) => inner.undo_bulk(range, pending),
            }
        }
        Ok(())
    }
}

impl Inner {
    /// The columns, the row and the creator's stamp of a node or an edge
    /// the store holds.
    fn columns_mut(&mut self, entity: Entity) -> Result<(&mut Columns, usize, u64), ApplyError> {
        match entity {
            Entity::Node(id) => {
                let (group, row) = self
                    .node_mut(id.as_u64())
                    .ok_or(ApplyError::Missing(entity))?;
                let created = group.version(row).created;
                Ok((&mut group.columns, row, created))
            }
            Entity::Edge(id) => {
                let (group, row) = self
                    .edge_mut(id.as_u64())
                    .ok_or(ApplyError::Missing(entity))?;
                let created = group.version(row).created;
                Ok((&mut group.columns, row, created))
            }
        }
    }

    /// Sets (`Some`) or removes (`None`) a value of a node or an edge. A set
    /// to null removes the value (#412 decision Q8); one of a value the
    /// entity lacks changes nothing, also in replay, where a property set
    /// always applies.
    fn write_value(
        &mut self,
        entity: Entity,
        key: &PropertyKey,
        value: Option<&Value>,
        mode: Mode,
    ) -> Result<Applied, ApplyError> {
        let set = value.is_some();
        let value = value.filter(|value| !value.is_null());
        let seen = match entity {
            Entity::Node(id) => self.node_visible(id.as_u64(), mode.read),
            Entity::Edge(id) => self.edge_visible(id.as_u64(), mode.read),
        };
        if !seen {
            return mode.writer.unseen(entity);
        }
        let absent = || match mode.writer {
            Writer::Replay { .. } if set => Ok(Applied::Committed),
            _ => mode.writer.no_op(ApplyError::Missing(entity)),
        };
        let key = match value {
            Some(_) => self.keys.get_or_create(key.as_str()),
            None => match self.keys.get_id(key.as_str()) {
                Some(key) => key,
                None => return absent(),
            },
        };
        // A node's indexes need the value the write replaces.
        let indexed = matches!(entity, Entity::Node(_)) && !self.indexes.is_empty();
        let (columns, row, created) = self.columns_mut(entity)?;
        let old = if mode.images || value.is_none() || indexed {
            columns.get(key, row)
        } else {
            None
        };
        if value.is_none() && old.is_none() {
            return absent();
        }
        let version = columns.write(key, row, value, mode.keeps(created));
        if let (true, Entity::Node(id)) = (indexed, entity) {
            self.index_value(id.as_u64(), key, old.as_ref());
        }
        Ok(mode.applied(|| Before::Value(old), version))
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
        let created = group.version(row).created;
        let before = if mode.images {
            self.label_names(&group.labels_of(row))
        } else {
            Labels::new()
        };
        let (group, row) = self
            .node_mut(raw)
            .ok_or(ApplyError::Missing(Entity::Node(id)))?;
        let version = match mode.keeps(created) {
            Some((stamp, snapshot)) => {
                let labels = &group.labels;
                if group.label_updates.keep(row, stamp, snapshot, || {
                    label_ids(labels, row).into_boxed_slice()
                }) {
                    PendingVersion::Created
                } else {
                    PendingVersion::Replaced
                }
            }
            None => PendingVersion::Created,
        };
        group.labels.entry(label_id).or_default().put(row, add);
        if mode.stamped() {
            self.counts.label(label_id, if add { 1 } else { -1 });
        }
        if !self.indexes.is_empty() {
            self.index_labels(raw, Some(label.as_str()));
        }
        Ok(mode.applied(|| Before::Labels(before), version))
    }

    /// Commits the kept value of `key` of a node or an edge that
    /// transaction `pending` wrote: its undo node gets epoch `at`. A row the
    /// transaction created has no node (no one else saw it).
    fn stamp_value(
        &mut self,
        entity: Entity,
        key: &PropertyKey,
        pending: u64,
        at: u64,
    ) -> Result<(), ApplyError> {
        let key = self.keys.get_id(key.as_str()).ok_or_else(|| {
            ApplyError::Refused(format!("property key {key} is not in the dictionary"))
        })?;
        let (columns, row, created) = self
            .columns_mut(entity)
            .map_err(|_| not_pending(entity, "value"))?;
        let kept = columns.stamp(key, row, pending, at);
        check_kept(entity, kept, created, pending, at, "value")
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
                if group
                    .versions
                    .commit_created(row..row + 1, pending, at)
                    .is_empty()
                {
                    return Err(not_pending(Entity::Node(*id), "create"));
                }
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
                if group.versions.deleted(row) != pending {
                    return Err(not_pending(Entity::Node(*id), "delete"));
                }
                group.versions.set_deleted(row, at);
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
                if group
                    .versions
                    .commit_created(row..row + 1, pending, at)
                    .is_empty()
                {
                    return Err(not_pending(Entity::Edge(*id), "create"));
                }
                self.counts.edges += 1;
                self.counts.edge_type(type_id, 1);
            }
            DataOp::DeleteEdge { id } => {
                let (group, row) = self
                    .edge_mut(id.as_u64())
                    .ok_or_else(|| not_pending(Entity::Edge(*id), "delete"))?;
                let (_, _, type_id) = group.ends(row);
                if group.versions.deleted(row) != pending {
                    return Err(not_pending(Entity::Edge(*id), "delete"));
                }
                group.versions.set_deleted(row, at);
                self.counts.edges -= 1;
                self.counts.edge_type(type_id, -1);
            }
            DataOp::AddNodeLabel { id, label } | DataOp::RemoveNodeLabel { id, label } => {
                let label_id = self.label_id(label).ok_or_else(|| {
                    ApplyError::Refused(format!("label {label} is not in the dictionary"))
                })?;
                let entity = Entity::Node(*id);
                let (group, row) = self
                    .node_mut(id.as_u64())
                    .ok_or_else(|| not_pending(entity, "label"))?;
                let created = group.version(row).created;
                let kept = group.label_updates.stamp(row, pending, at);
                check_kept(entity, kept, created, pending, at, "label")?;
                let delta = if matches!(op, DataOp::AddNodeLabel { .. }) {
                    1
                } else {
                    -1
                };
                self.counts.label(label_id, delta);
            }
            DataOp::SetNodeProperty { id, key, .. } | DataOp::RemoveNodeProperty { id, key } => {
                self.stamp_value(Entity::Node(*id), key, pending, at)?;
            }
            DataOp::SetEdgeProperty { id, key, .. } | DataOp::RemoveEdgeProperty { id, key } => {
                self.stamp_value(Entity::Edge(*id), key, pending, at)?;
            }
            DataOp::InsertTriple { .. } | DataOp::DeleteTriple { .. } => {
                return Err(refuse_triple(op));
            }
        }
        Ok(())
    }

    /// Commits the rows transaction `pending` created in a bulk range, a
    /// vector at a time (change target R7; one constant stamp for a vector
    /// the transaction filled): the row groups the range has no rows in
    /// cost nothing, absent ids are skipped.
    fn stamp_bulk(&mut self, range: &BulkRange, pending: u64, at: u64) {
        if range.ids.is_empty() {
            return;
        }
        let (first, _) = locate(range.ids.start);
        let (last, _) = locate(range.ids.end - 1);
        // The rows of the range in group `index`.
        let rows_in = |index: u64| {
            let base = index * ROWS_PER_GROUP;
            let start = range.ids.start.max(base) - base;
            let end = range.ids.end.min(base + ROWS_PER_GROUP) - base;
            usize::try_from(start).expect("a row below 65,536")
                ..usize::try_from(end).expect("a row count up to 65,536")
        };
        match range.table {
            Table::Nodes => {
                for (index, group) in self.nodes.range_mut(first..=last) {
                    let runs = group.versions.commit_created(rows_in(*index), pending, at);
                    let rows: usize = runs.iter().map(ExactSizeIterator::len).sum();
                    self.counts.nodes += i64::try_from(rows).expect("rows of a row group");
                    for (label, members) in &group.labels {
                        let labeled: usize =
                            runs.iter().map(|run| members.count_in(run.clone())).sum();
                        if labeled > 0 {
                            self.counts.label(
                                *label,
                                i64::try_from(labeled).expect("rows of a row group"),
                            );
                        }
                    }
                }
            }
            Table::Edges => {
                for (index, group) in self.edges.range_mut(first..=last) {
                    for run in group.versions.commit_created(rows_in(*index), pending, at) {
                        for row in run {
                            self.counts.edges += 1;
                            self.counts.edge_type(group.ends(row).2, 1);
                        }
                    }
                }
            }
        }
    }

    /// Takes a node row out whole, and first out of every index.
    fn remove_node_row(&mut self, raw: u64) {
        if !self.indexes.is_empty() {
            self.unindex(raw);
        }
        if let Some((group, row)) = self.node_mut(raw) {
            group.remove(row);
        }
    }

    /// Takes an edge row out with its adjacency entries.
    pub(super) fn remove_edge_row(&mut self, raw: u64) {
        let Some((group, row)) = self.edge_mut(raw) else {
            return;
        };
        let (src, dst, _) = group.ends(row);
        group.remove(row);
        for (node, _, outgoing) in self.kept_ends(src, dst) {
            let (group, row) = locate(node);
            if let Some(group) = self.nodes.get_mut(&group) {
                group.adjacency_mut(outgoing).remove(row, raw);
            }
        }
    }

    /// Undoes one entry of transaction `pending`, which did `version` to the
    /// transaction's version of what it wrote.
    fn undo_op(
        &mut self,
        op: &DataOp,
        before: &Before,
        version: PendingVersion,
        pending: u64,
    ) -> Result<(), ApplyError> {
        match (op, before) {
            (DataOp::CreateNode { id, .. }, Before::Absent) => {
                let created = self
                    .node(id.as_u64())
                    .map(|(group, row)| group.version(row).created);
                if created != Some(pending) {
                    return Err(not_pending(Entity::Node(*id), "create"));
                }
                self.remove_node_row(id.as_u64());
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
                if group.versions.deleted(row) != pending {
                    return Err(not_pending(Entity::Node(*id), "delete"));
                }
                group.versions.set_deleted(row, ABSENT);
            }
            (DataOp::DeleteEdge { id }, Before::Edge(_)) => {
                let (group, row) = self
                    .edge_mut(id.as_u64())
                    .ok_or_else(|| not_pending(Entity::Edge(*id), "delete"))?;
                if group.versions.deleted(row) != pending {
                    return Err(not_pending(Entity::Edge(*id), "delete"));
                }
                group.versions.set_deleted(row, ABSENT);
            }
            (
                DataOp::SetNodeProperty { id, key, .. } | DataOp::RemoveNodeProperty { id, key },
                Before::Value(old),
            ) => self.restore_value(Entity::Node(*id), key, old.as_ref(), version, pending)?,
            (
                DataOp::SetEdgeProperty { id, key, .. } | DataOp::RemoveEdgeProperty { id, key },
                Before::Value(old),
            ) => self.restore_value(Entity::Edge(*id), key, old.as_ref(), version, pending)?,
            (
                DataOp::AddNodeLabel { id, .. } | DataOp::RemoveNodeLabel { id, .. },
                Before::Labels(labels),
            ) => {
                let entity = Entity::Node(*id);
                let label_ids = self.known_label_ids(labels)?;
                let (group, row) = self
                    .node_mut(id.as_u64())
                    .ok_or(ApplyError::Missing(entity))?;
                // A replaced version keeps its node: the before-image is
                // the transaction's own labels. A created one gives back
                // what its node kept, unless the transaction created the
                // row and kept nothing.
                let restored = match version {
                    PendingVersion::Replaced => label_ids,
                    PendingVersion::Created => match group.label_updates.undo(row, pending) {
                        Some(kept) => kept.into_vec(),
                        None if group.version(row).created == pending => label_ids,
                        None => return Err(not_pending(entity, "label")),
                    },
                };
                group.set_labels(row, &restored);
                if !self.indexes.is_empty() {
                    self.index_labels(id.as_u64(), None);
                }
            }
            (DataOp::InsertTriple { .. } | DataOp::DeleteTriple { .. }, _) => {
                return Err(refuse_triple(op));
            }
            _ => return Err(misfit(op, before)),
        }
        Ok(())
    }

    /// Undoes a value write of transaction `pending`: a replaced version
    /// gets its before-image back (the transaction's own earlier value) and
    /// keeps its undo node; a created one gets what its node kept, and
    /// leaves the node, or its before-image when the transaction created
    /// the row and kept nothing. `Some` sets the value, `None` removes it.
    fn restore_value(
        &mut self,
        entity: Entity,
        key: &PropertyKey,
        old: Option<&Value>,
        version: PendingVersion,
        pending: u64,
    ) -> Result<(), ApplyError> {
        let key_id = self.keys.get_id(key.as_str()).ok_or_else(|| {
            ApplyError::Refused(format!("property key {key} is not in the dictionary"))
        })?;
        let indexed = matches!(entity, Entity::Node(_)) && !self.indexes.is_empty();
        let (columns, row, created) = self.columns_mut(entity)?;
        // The value undone, which the indexes may lose.
        let undone = if indexed {
            columns.get(key_id, row)
        } else {
            None
        };
        match version {
            PendingVersion::Replaced => columns.restore(key_id, row, old),
            PendingVersion::Created => {
                if !columns.undo(key_id, row, pending) {
                    if created != pending {
                        return Err(not_pending(entity, "value"));
                    }
                    columns.restore(key_id, row, old);
                }
            }
        }
        if let (true, Entity::Node(id)) = (indexed, entity) {
            self.index_value(id.as_u64(), key_id, undone.as_ref());
        }
        Ok(())
    }

    /// Takes out the rows transaction `pending` created in a bulk range,
    /// last first; absent ids are skipped.
    fn undo_bulk(&mut self, range: &BulkRange, pending: u64) {
        for raw in range.ids.clone().rev() {
            match range.table {
                Table::Nodes => {
                    if self
                        .node(raw)
                        .is_some_and(|(group, row)| group.version(row).created == pending)
                    {
                        self.remove_node_row(raw);
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
