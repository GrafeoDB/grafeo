//! The store side of a change set: how a graph's store applies a
//! transaction's writes, stamps them at commit and undoes them at rollback.
//!
//! A transaction records each write it applies in its
//! [`ChangeSet`](grafeo_common::change::ChangeSet): the op, with its
//! after-image, and what the write replaced. A store keeps no log of its own
//! per transaction: it applies an op and reports what it replaced
//! ([`ChangeTarget::apply`]), and reads the transaction's entries back to
//! stamp them with the commit epoch ([`ChangeTarget::stamp`]) or to undo
//! them ([`ChangeTarget::undo`]). Replay of the log calls the same `apply`,
//! as a [`Writer::Replay`], so what replay builds cannot drift from what the
//! live writes built.
//!
//! One target per graph: each named graph is a store of its own. The
//! targets are [`LpgStore`](crate::graph::lpg::LpgStore) and
//! [`ExternalTarget`], which bridges a store a database was built on
//! ([`GraphStoreMut`]).

use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use grafeo_common::change::{
    Before, Change, DataOp, EdgeImage, Entity, Labels, NodeImage, PendingVersion, Properties,
};
use grafeo_common::types::{ArcStr, EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};

use super::Direction;
use super::lpg::{Edge, Node};
use super::traits::GraphStoreMut;

/// How an op is applied: who writes, and so what it becomes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Writer {
    /// A live transaction (also a direct call's private one, and a bulk
    /// write): a pending version only it sees until `stamp` or `undo`.
    /// Lenient: a write that changes nothing returns [`Applied::Unchanged`].
    Transaction {
        /// The transaction.
        id: TransactionId,
        /// The epoch it reads at.
        snapshot: EpochId,
    },
    /// Replay of a logged group: applied and stamped at its epoch, with all
    /// [`ChangeTarget::stamp`] does, and no before-image built. Strict: a
    /// create at an id in use, a missing entity, a node delete with edges,
    /// removing an absent property or label, or adding a present one is an
    /// error. A property set always applies.
    Replay {
        /// The group's epoch.
        epoch: EpochId,
    },
    /// A write outside any transaction that commits at once (a direct call
    /// while no transaction is open): applied and stamped at its epoch as
    /// [`Writer::Replay`] does, through the same internal function, but
    /// lenient like a transaction (a write that changes nothing is
    /// [`Applied::Unchanged`]). With `before_images`, a write that changed
    /// something returns its before-image ([`Applied::Changed`]), which the
    /// log and change data capture read; without, it returns
    /// [`Applied::Committed`]. The caller runs it while nothing else can
    /// write what it writes, and never stamps or undoes its entries.
    Immediate {
        /// The epoch it commits at, which it also reads at.
        epoch: EpochId,
        /// Whether to build the before-images.
        before_images: bool,
    },
}

impl Writer {
    /// Whether the write is applied and stamped at once
    /// ([`Writer::Replay`], [`Writer::Immediate`]), by the system, instead of
    /// as a transaction's pending version: a target stamps such a write as
    /// it applies it.
    #[must_use]
    pub const fn stamps_at_once(self) -> bool {
        matches!(self, Self::Replay { .. } | Self::Immediate { .. })
    }

    /// Whether `apply` builds the before-image: for a transaction and an
    /// immediate write that asks for them, not for replay.
    #[must_use]
    pub(crate) const fn builds_images(self) -> bool {
        matches!(
            self,
            Self::Transaction { .. }
                | Self::Immediate {
                    before_images: true,
                    ..
                }
        )
    }

    /// What `apply` returns for an op whose entity this writer does not
    /// see: nothing to record for a transaction, [`ApplyError::Missing`]
    /// for replay.
    ///
    /// # Errors
    ///
    /// Returns [`ApplyError::Missing`] under [`Writer::Replay`].
    pub(crate) fn unseen(self, entity: Entity) -> Result<Applied, ApplyError> {
        self.no_op(ApplyError::Missing(entity))
    }

    /// What `apply` returns for a write that would change nothing (a label
    /// the node has already, a property it lacks): nothing to record for a
    /// transaction, `error` for replay.
    ///
    /// # Errors
    ///
    /// Returns `error` under [`Writer::Replay`].
    pub(crate) fn no_op(self, error: ApplyError) -> Result<Applied, ApplyError> {
        match self {
            Self::Transaction { .. } | Self::Immediate { .. } => Ok(Applied::Unchanged),
            Self::Replay { .. } => Err(error),
        }
    }
}

/// What [`ChangeTarget::apply`] did.
#[derive(Debug, Clone, PartialEq)]
pub enum Applied {
    /// Nothing changed ([`Writer::Transaction`], [`Writer::Immediate`]):
    /// nothing to record.
    Unchanged,
    /// A pending version holds the change ([`Writer::Transaction`]), or the
    /// change is committed already ([`Writer::Immediate`]): the entry to
    /// record.
    Changed {
        /// What the write replaced.
        before: Before,
        /// Whether it created the transaction's pending version of what it
        /// wrote or changed one an earlier write of the transaction created.
        version: PendingVersion,
    },
    /// Applied and stamped ([`Writer::Replay`], [`Writer::Immediate`]
    /// without before-images); no before-image was built.
    Committed,
}

/// Why [`ChangeTarget::apply`], `stamp` or `undo` refused. On an `apply`
/// error nothing changed; a `stamp` or `undo` error is a broken invariant.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ApplyError {
    /// A create names an id the store already holds (also: replay adds a
    /// label the node has).
    Exists(Entity),
    /// An op names an entity the store does not hold, or the writer does not
    /// see (also: replay removes a property or a label the entity lacks).
    Missing(Entity),
    /// A node delete while the writer sees an edge of the node.
    HasEdges(NodeId),
    /// The store refuses the write, checked before any change: a name past
    /// a store limit (length, label count), an op or a writer the store
    /// does not take, or a value the before-image needs that the store
    /// cannot read (a spilled value whose file cannot be read). No value is
    /// refused for its type.
    Refused(String),
}

impl fmt::Display for ApplyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exists(entity) => write!(f, "{} already exists", describe(*entity)),
            Self::Missing(entity) => write!(
                f,
                "{} does not exist, or the writer does not see it",
                describe(*entity)
            ),
            Self::HasEdges(id) => write!(
                f,
                "node {} still has edges: delete them first (DETACH DELETE)",
                id.as_u64()
            ),
            Self::Refused(reason) => write!(f, "the store refused the write: {reason}"),
        }
    }
}

impl std::error::Error for ApplyError {}

/// How an error names an entity.
pub(crate) fn describe(entity: Entity) -> String {
    match entity {
        Entity::Node(id) => format!("node {}", id.as_u64()),
        Entity::Edge(id) => format!("edge {}", id.as_u64()),
    }
}

/// What a target's [`ChangeTarget::undo`] promises; fixed for the target's
/// life.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum UndoSupport {
    /// Undo restores every entry it is given and fails only on a broken
    /// invariant.
    Exact,
    /// No undo: the store keeps what `apply` wrote ([`ExternalTarget`]).
    /// The engine never calls `undo`.
    None,
}

/// A graph's store, as the change set writes it. One per graph. Every method
/// takes `&self`; the store synchronizes internally and never calls back
/// into the engine.
///
/// # The stamp and undo contract
///
/// `apply` allocates everything a version needs, so `stamp` only gives the
/// versions the commit epoch and `undo` only drops or rewrites what `apply`
/// made. An `Err` from `stamp` or `undo` is a broken invariant: the engine
/// poisons the database rather than retry, as a commit of two graphs whose
/// second `stamp` fails is half stamped. The pending version of what an
/// entry wrote is the newest of its entity, key or label set: the
/// transaction manager's claims let one transaction at a time write an
/// entity, so `stamp` and `undo` find it without a handle in the entry.
pub trait ChangeTarget: Send + Sync {
    /// Reserves `count` consecutive node ids, never handed out again (also
    /// not after a rollback).
    ///
    /// # Errors
    ///
    /// Returns [`ApplyError::Refused`] when the ids are exhausted, or the
    /// store gives its own ids ([`ExternalTarget`]).
    fn reserve_node_ids(&self, count: u64) -> Result<Range<u64>, ApplyError>;

    /// Reserves `count` consecutive edge ids.
    ///
    /// # Errors
    ///
    /// As [`reserve_node_ids`](Self::reserve_node_ids).
    fn reserve_edge_ids(&self, count: u64) -> Result<Range<u64>, ApplyError>;

    /// Applies `op` as `writer`; on `Err` nothing changed. A create inserts
    /// at the op's id and keeps the allocator above it. Replay calls this
    /// same function (skipping only the before-images): no drift.
    ///
    /// # Errors
    ///
    /// See [`ApplyError`] and [`Writer`]: a transaction's write that changes
    /// nothing is [`Applied::Unchanged`], replay's is an error.
    fn apply(&self, op: &DataOp, writer: Writer) -> Result<Applied, ApplyError>;

    /// Applies `op`, one row of a bulk write: a create by a transaction
    /// ([`Writer::Transaction`]) at an id of a range it reserved and
    /// recorded as one entry ([`BulkRange`](grafeo_common::change::BulkRange)),
    /// which its commit stamps and its rollback undoes as a range. Builds no
    /// before-image (the range is the entry), and leaves to the bulk write
    /// what it checked itself for every row: an edge's endpoints, which its
    /// writer sees. On `Err` nothing changed.
    ///
    /// The default is [`apply`](Self::apply), which checks everything again.
    ///
    /// # Errors
    ///
    /// [`ApplyError::Refused`] for an op that is no create or a writer that
    /// is no transaction; otherwise as [`apply`](Self::apply).
    fn apply_bulk_row(&self, op: &DataOp, writer: Writer) -> Result<(), ApplyError> {
        check_bulk_row(op, writer)?;
        self.apply(op, writer).map(drop)
    }

    /// Commits `transaction`'s entries of this graph at `epoch`: the pending
    /// versions they name get the epoch, and the statistics counters and
    /// dirty marks take the entries. A bulk range costs O(row groups) where
    /// a run's version is one record, O(ids) at most. Recorded order,
    /// O(entries).
    ///
    /// # Errors
    ///
    /// A broken invariant (see the trait docs): the database is poisoned.
    fn stamp(
        &self,
        transaction: TransactionId,
        entries: &mut dyn Iterator<Item = &Change>,
        epoch: EpochId,
    ) -> Result<(), ApplyError>;

    /// Undoes `transaction`'s entries, last to first (`next_back`): drops
    /// the version an entry [created](PendingVersion::Created) (or, in a
    /// store that writes in place, writes its before-image back), writes the
    /// before-image back into one it [replaced](PendingVersion::Replaced) and
    /// keeps it; a bulk range drops its pending ids. O(entries).
    ///
    /// # Errors
    ///
    /// A broken invariant (see the trait docs): the database is poisoned.
    fn undo(
        &self,
        transaction: TransactionId,
        entries: &mut dyn DoubleEndedIterator<Item = &Change>,
    ) -> Result<(), ApplyError>;

    /// Whether `undo` works.
    fn undo_support(&self) -> UndoSupport {
        UndoSupport::Exact
    }
}

// ── External stores ─────────────────────────────────────────────────

/// A store a database was built on ([`GraphStoreMut`], through
/// `GrafeoDB::with_store`) as a change target: `apply` writes through the
/// store's own versioned methods and reads before it writes, so the entry
/// has its before-image (change data capture reports it); `stamp` does
/// nothing (the store commits its writes itself); there is no undo
/// ([`UndoSupport::None`]): the store keeps what `apply` wrote, and the
/// engine never calls [`undo`](ChangeTarget::undo).
///
/// A [`GraphStoreMut`] gives the ids of what it creates and cannot create at
/// an id given to it, so this target reserves no ids and `apply` refuses
/// creates: a writer creates through [`create_node`](Self::create_node) and
/// [`create_edge`](Self::create_edge), which use the store's own ids and
/// return the op with the id it gave.
pub struct ExternalTarget {
    store: Arc<dyn GraphStoreMut>,
}

impl ExternalTarget {
    /// A change target writing `store`.
    #[must_use]
    pub fn new(store: Arc<dyn GraphStoreMut>) -> Self {
        Self { store }
    }

    /// The store this target writes.
    #[must_use]
    pub fn store(&self) -> &Arc<dyn GraphStoreMut> {
        &self.store
    }

    /// Creates a node with `labels` and `properties` as `writer`, at the id
    /// the store gives: returns the op with that id, and what to record (a
    /// create replaced nothing).
    ///
    /// # Errors
    ///
    /// Returns [`ApplyError::Refused`] when the store refuses a value. The
    /// store then holds the node with the values set before it: an external
    /// store keeps what it applied.
    pub fn create_node(
        &self,
        labels: Labels,
        properties: Properties,
        writer: Writer,
    ) -> Result<(DataOp, Applied), ApplyError> {
        let names: Vec<&str> = labels.iter().map(ArcStr::as_str).collect();
        let id = match writer {
            Writer::Transaction {
                id: transaction,
                snapshot,
            } => self
                .store
                .create_node_versioned(&names, snapshot, transaction),
            Writer::Replay { .. } | Writer::Immediate { .. } => self.store.create_node(&names),
        };
        for (key, value) in &properties {
            match writer {
                Writer::Transaction {
                    id: transaction, ..
                } => self
                    .store
                    .set_node_property_versioned(id, key.as_str(), value.clone(), transaction)
                    .map_err(|error| refused(Entity::Node(id), &error))?,
                Writer::Replay { .. } | Writer::Immediate { .. } => {
                    self.store
                        .set_node_property(id, key.as_str(), value.clone());
                }
            }
        }
        let op = DataOp::CreateNode {
            id,
            labels,
            properties,
        };
        Ok((op, Self::done(writer, Before::Absent)))
    }

    /// Creates an edge from `src` to `dst` with `properties` as `writer`, at
    /// the id the store gives, as [`create_node`](Self::create_node) does a
    /// node.
    ///
    /// # Errors
    ///
    /// Returns [`ApplyError::Missing`], creating nothing, when the writer
    /// does not see an endpoint, and [`ApplyError::Refused`] when the store
    /// refuses the edge or a value (the values set before it stay).
    pub fn create_edge(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: ArcStr,
        properties: Properties,
        writer: Writer,
    ) -> Result<(DataOp, Applied), ApplyError> {
        for end in [src, dst] {
            if self.node(end, writer).is_none() {
                return Err(ApplyError::Missing(Entity::Node(end)));
            }
        }
        let id = match writer {
            Writer::Transaction {
                id: transaction,
                snapshot,
            } => self
                .store
                .create_edge_versioned(src, dst, &edge_type, snapshot, transaction)
                .map_err(|error| refused(Entity::Node(src), &error))?,
            Writer::Replay { .. } | Writer::Immediate { .. } => {
                self.store.create_edge(src, dst, &edge_type)
            }
        };
        for (key, value) in &properties {
            match writer {
                Writer::Transaction {
                    id: transaction, ..
                } => {
                    self.store.set_edge_property_versioned(
                        id,
                        key.as_str(),
                        value.clone(),
                        transaction,
                    );
                }
                Writer::Replay { .. } | Writer::Immediate { .. } => {
                    self.store
                        .set_edge_property(id, key.as_str(), value.clone());
                }
            }
        }
        let op = DataOp::CreateEdge {
            id,
            src,
            dst,
            edge_type,
            properties,
        };
        Ok((op, Self::done(writer, Before::Absent)))
    }

    /// Node `id` as `writer` sees it.
    fn node(&self, id: NodeId, writer: Writer) -> Option<Node> {
        match writer {
            Writer::Transaction {
                id: transaction,
                snapshot,
            } => self.store.get_node_versioned(id, snapshot, transaction),
            Writer::Replay { .. } | Writer::Immediate { .. } => self.store.get_node(id),
        }
    }

    /// Edge `id` as `writer` sees it.
    fn edge(&self, id: EdgeId, writer: Writer) -> Option<Edge> {
        match writer {
            Writer::Transaction {
                id: transaction,
                snapshot,
            } => self.store.get_edge_versioned(id, snapshot, transaction),
            Writer::Replay { .. } | Writer::Immediate { .. } => self.store.get_edge(id),
        }
    }

    /// Whether `writer` sees an edge of node `id`, either way.
    fn sees_an_edge_of(&self, id: NodeId, writer: Writer) -> bool {
        self.store
            .edges_from(id, Direction::Both)
            .into_iter()
            .any(|(_, edge)| self.edge(edge, writer).is_some())
    }

    /// The applied result of a write that replaced `before`.
    fn done(writer: Writer, before: Before) -> Applied {
        if writer.builds_images() {
            Applied::Changed {
                before,
                version: PendingVersion::Created,
            }
        } else {
            Applied::Committed
        }
    }

    /// Refuses a create: see the type's docs.
    fn refuse_create(entity: Entity) -> ApplyError {
        ApplyError::Refused(format!(
            "{}: an external store gives the ids of what it creates, so it cannot create at \
             an id reserved for it",
            describe(entity)
        ))
    }

    fn delete_node(&self, id: NodeId, writer: Writer) -> Result<Applied, ApplyError> {
        let Some(node) = self.node(id, writer) else {
            return writer.unseen(Entity::Node(id));
        };
        if self.sees_an_edge_of(id, writer) {
            return Err(ApplyError::HasEdges(id));
        }
        let image = NodeImage {
            labels: node.labels.clone(),
            properties: sorted(node.properties.iter()),
        };
        let deleted = match writer {
            Writer::Transaction {
                id: transaction,
                snapshot,
            } => self
                .store
                .delete_node_versioned(id, snapshot, transaction)
                .map_err(|error| refused(Entity::Node(id), &error))?,
            Writer::Replay { .. } | Writer::Immediate { .. } => self.store.delete_node(id),
        };
        if !deleted {
            return writer.unseen(Entity::Node(id));
        }
        Ok(Self::done(writer, Before::Node(Box::new(image))))
    }

    fn delete_edge(&self, id: EdgeId, writer: Writer) -> Result<Applied, ApplyError> {
        let Some(edge) = self.edge(id, writer) else {
            return writer.unseen(Entity::Edge(id));
        };
        let image = EdgeImage {
            src: edge.src,
            dst: edge.dst,
            edge_type: edge.edge_type.clone(),
            properties: sorted(edge.properties.iter()),
        };
        let deleted = match writer {
            Writer::Transaction {
                id: transaction,
                snapshot,
            } => self.store.delete_edge_versioned(id, snapshot, transaction),
            Writer::Replay { .. } | Writer::Immediate { .. } => self.store.delete_edge(id),
        };
        if !deleted {
            return writer.unseen(Entity::Edge(id));
        }
        Ok(Self::done(writer, Before::Edge(Box::new(image))))
    }

    fn set_node_value(
        &self,
        id: NodeId,
        key: &PropertyKey,
        value: &Value,
        writer: Writer,
    ) -> Result<Applied, ApplyError> {
        let Some(node) = self.node(id, writer) else {
            return writer.unseen(Entity::Node(id));
        };
        let old = node.properties.get(key).cloned();
        match writer {
            Writer::Transaction {
                id: transaction, ..
            } => self
                .store
                .set_node_property_versioned(id, key.as_str(), value.clone(), transaction)
                .map_err(|error| refused(Entity::Node(id), &error))?,
            Writer::Replay { .. } | Writer::Immediate { .. } => {
                self.store
                    .set_node_property(id, key.as_str(), value.clone());
            }
        }
        Ok(Self::done(writer, Before::Value(old)))
    }

    fn remove_node_value(
        &self,
        id: NodeId,
        key: &PropertyKey,
        writer: Writer,
    ) -> Result<Applied, ApplyError> {
        let Some(node) = self.node(id, writer) else {
            return writer.unseen(Entity::Node(id));
        };
        if node.properties.get(key).is_none() {
            return writer.no_op(ApplyError::Missing(Entity::Node(id)));
        }
        let removed = match writer {
            Writer::Transaction {
                id: transaction, ..
            } => self
                .store
                .remove_node_property_versioned(id, key.as_str(), transaction),
            Writer::Replay { .. } | Writer::Immediate { .. } => {
                self.store.remove_node_property(id, key.as_str())
            }
        }
        .map_err(|error| refused(Entity::Node(id), &error))?;
        match removed {
            Some(old) => Ok(Self::done(writer, Before::Value(Some(old)))),
            None => writer.no_op(ApplyError::Missing(Entity::Node(id))),
        }
    }

    fn set_edge_value(
        &self,
        id: EdgeId,
        key: &PropertyKey,
        value: &Value,
        writer: Writer,
    ) -> Result<Applied, ApplyError> {
        let Some(edge) = self.edge(id, writer) else {
            return writer.unseen(Entity::Edge(id));
        };
        let old = edge.properties.get(key).cloned();
        match writer {
            Writer::Transaction {
                id: transaction, ..
            } => {
                self.store.set_edge_property_versioned(
                    id,
                    key.as_str(),
                    value.clone(),
                    transaction,
                );
            }
            Writer::Replay { .. } | Writer::Immediate { .. } => {
                self.store
                    .set_edge_property(id, key.as_str(), value.clone());
            }
        }
        Ok(Self::done(writer, Before::Value(old)))
    }

    fn remove_edge_value(
        &self,
        id: EdgeId,
        key: &PropertyKey,
        writer: Writer,
    ) -> Result<Applied, ApplyError> {
        let Some(edge) = self.edge(id, writer) else {
            return writer.unseen(Entity::Edge(id));
        };
        if edge.properties.get(key).is_none() {
            return writer.no_op(ApplyError::Missing(Entity::Edge(id)));
        }
        let removed = match writer {
            Writer::Transaction {
                id: transaction, ..
            } => self
                .store
                .remove_edge_property_versioned(id, key.as_str(), transaction),
            Writer::Replay { .. } | Writer::Immediate { .. } => {
                self.store.remove_edge_property(id, key.as_str())
            }
        }
        .map_err(|error| refused(Entity::Edge(id), &error))?;
        match removed {
            Some(old) => Ok(Self::done(writer, Before::Value(Some(old)))),
            None => writer.no_op(ApplyError::Missing(Entity::Edge(id))),
        }
    }

    fn change_label(
        &self,
        id: NodeId,
        label: &str,
        add: bool,
        writer: Writer,
    ) -> Result<Applied, ApplyError> {
        let Some(node) = self.node(id, writer) else {
            return writer.unseen(Entity::Node(id));
        };
        if node.has_label(label) == add {
            return writer.no_op(if add {
                ApplyError::Exists(Entity::Node(id))
            } else {
                ApplyError::Missing(Entity::Node(id))
            });
        }
        let before = node.labels.clone();
        let changed = match (writer, add) {
            (
                Writer::Transaction {
                    id: transaction, ..
                },
                true,
            ) => self.store.add_label_versioned(id, label, transaction),
            (
                Writer::Transaction {
                    id: transaction, ..
                },
                false,
            ) => self.store.remove_label_versioned(id, label, transaction),
            (Writer::Replay { .. } | Writer::Immediate { .. }, true) => {
                self.store.add_label(id, label)
            }
            (Writer::Replay { .. } | Writer::Immediate { .. }, false) => {
                self.store.remove_label(id, label)
            }
        };
        if !changed {
            return writer.unseen(Entity::Node(id));
        }
        Ok(Self::done(writer, Before::Labels(before)))
    }
}

impl fmt::Debug for ExternalTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExternalTarget").finish_non_exhaustive()
    }
}

impl ChangeTarget for ExternalTarget {
    fn reserve_node_ids(&self, count: u64) -> Result<Range<u64>, ApplyError> {
        Err(ApplyError::Refused(format!(
            "{count} node ids: an external store gives the ids of what it creates"
        )))
    }

    fn reserve_edge_ids(&self, count: u64) -> Result<Range<u64>, ApplyError> {
        Err(ApplyError::Refused(format!(
            "{count} edge ids: an external store gives the ids of what it creates"
        )))
    }

    fn apply(&self, op: &DataOp, writer: Writer) -> Result<Applied, ApplyError> {
        match op {
            DataOp::CreateNode { id, .. } => Err(Self::refuse_create(Entity::Node(*id))),
            DataOp::CreateEdge { id, .. } => Err(Self::refuse_create(Entity::Edge(*id))),
            DataOp::DeleteNode { id } => self.delete_node(*id, writer),
            DataOp::DeleteEdge { id } => self.delete_edge(*id, writer),
            DataOp::SetNodeProperty { id, key, value } => {
                self.set_node_value(*id, key, value, writer)
            }
            DataOp::RemoveNodeProperty { id, key } => self.remove_node_value(*id, key, writer),
            DataOp::SetEdgeProperty { id, key, value } => {
                self.set_edge_value(*id, key, value, writer)
            }
            DataOp::RemoveEdgeProperty { id, key } => self.remove_edge_value(*id, key, writer),
            DataOp::AddNodeLabel { id, label } => self.change_label(*id, label, true, writer),
            DataOp::RemoveNodeLabel { id, label } => self.change_label(*id, label, false, writer),
            DataOp::InsertTriple { .. } | DataOp::DeleteTriple { .. } => Err(refuse_triple(op)),
        }
    }

    /// Does nothing: the store commits what `apply` wrote itself.
    fn stamp(
        &self,
        _transaction: TransactionId,
        _entries: &mut dyn Iterator<Item = &Change>,
        _epoch: EpochId,
    ) -> Result<(), ApplyError> {
        Ok(())
    }

    /// Never called ([`UndoSupport::None`]): refuses, so a call that should
    /// not happen does not pass for an undo.
    fn undo(
        &self,
        _transaction: TransactionId,
        _entries: &mut dyn DoubleEndedIterator<Item = &Change>,
    ) -> Result<(), ApplyError> {
        Err(ApplyError::Refused(
            "an external store has no undo: it keeps what was applied".to_string(),
        ))
    }

    fn undo_support(&self) -> UndoSupport {
        UndoSupport::None
    }
}

/// Refuses what no bulk write applies as a row (see
/// [`ChangeTarget::apply_bulk_row`]): an op that is no create, or a writer
/// that is no transaction.
///
/// # Errors
///
/// [`ApplyError::Refused`], naming the op's kind or the writer.
pub(crate) fn check_bulk_row(op: &DataOp, writer: Writer) -> Result<(), ApplyError> {
    if !matches!(op, DataOp::CreateNode { .. } | DataOp::CreateEdge { .. }) {
        return Err(ApplyError::Refused(format!(
            "a bulk write's row is a create, not an op of kind {}",
            op.kind()
        )));
    }
    if !matches!(writer, Writer::Transaction { .. }) {
        return Err(ApplyError::Refused(format!(
            "a bulk write's row is a transaction's pending create, not a write by {writer:?}"
        )));
    }
    Ok(())
}

/// The error of a write a store refused with `error`.
pub(crate) fn refused(entity: Entity, error: &impl fmt::Display) -> ApplyError {
    ApplyError::Refused(format!("{}: {error}", describe(entity)))
}

/// The error of a triple op given to a labeled property graph's target.
pub(crate) fn refuse_triple(op: &DataOp) -> ApplyError {
    ApplyError::Refused(format!(
        "an RDF op (kind {}) given to a labeled property graph's store",
        op.kind()
    ))
}

/// Properties in key order, as an image holds them.
fn sorted<'p>(properties: impl Iterator<Item = (&'p PropertyKey, &'p Value)>) -> Properties {
    let mut properties: Properties = properties
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    properties.sort_by(|(a, _), (b, _)| a.cmp(b));
    properties
}

#[cfg(all(test, feature = "lpg"))]
mod tests {
    use grafeo_common::change::Labels;
    use grafeo_common::types::ArcStr;

    use super::*;
    use crate::graph::lpg::LpgStore;

    fn labels(names: &[&str]) -> Labels {
        names.iter().map(|name| ArcStr::from(*name)).collect()
    }

    /// An external store keeps what `apply` wrote, so its target has no
    /// undo and refuses a call to it; `apply` reads what each write replaces
    /// before it writes, so the entries have their before-images (change
    /// data capture reports them); creates are refused, with nothing
    /// created, as the store gives its own ids.
    #[test]
    fn an_external_target_builds_images_and_has_no_undo() {
        let store = Arc::new(LpgStore::new().unwrap());
        let alix = store.create_node_with_props(
            &["Person"],
            [
                ("name", Value::from("Alix")),
                ("city", Value::from("Amsterdam")),
            ],
        );
        let gus = store.create_node_with_props(&["Person"], [("name", Value::from("Gus"))]);
        let knows = store.create_edge_with_props(alix, gus, "KNOWS", [("since", Value::Int64(3))]);
        let target = ExternalTarget::new(Arc::clone(&store) as Arc<dyn GraphStoreMut>);
        assert_eq!(target.undo_support(), UndoSupport::None);
        let writer = Writer::Transaction {
            id: TransactionId::new(19),
            snapshot: store.current_epoch(),
        };
        let key = PropertyKey::new;
        let changed = |before: Before| {
            Ok(Applied::Changed {
                before,
                version: PendingVersion::Created,
            })
        };

        assert_eq!(
            target.apply(
                &DataOp::SetNodeProperty {
                    id: alix,
                    key: key("city"),
                    value: Value::from("Paris")
                },
                writer
            ),
            changed(Before::Value(Some(Value::from("Amsterdam"))))
        );
        assert_eq!(
            target.apply(
                &DataOp::RemoveNodeProperty {
                    id: alix,
                    key: key("name")
                },
                writer
            ),
            changed(Before::Value(Some(Value::from("Alix"))))
        );
        assert_eq!(
            target.apply(
                &DataOp::RemoveNodeProperty {
                    id: alix,
                    key: key("name")
                },
                writer
            ),
            Ok(Applied::Unchanged),
            "a removal of an absent value changes nothing"
        );
        assert_eq!(
            target.apply(
                &DataOp::AddNodeLabel {
                    id: gus,
                    label: ArcStr::from("Traveller")
                },
                writer
            ),
            changed(Before::Labels(labels(&["Person"])))
        );
        assert_eq!(
            target.apply(&DataOp::DeleteNode { id: gus }, writer),
            Err(ApplyError::HasEdges(gus))
        );
        assert_eq!(
            target.apply(&DataOp::DeleteEdge { id: knows }, writer),
            changed(Before::Edge(Box::new(EdgeImage {
                src: alix,
                dst: gus,
                edge_type: ArcStr::from("KNOWS"),
                properties: vec![(key("since"), Value::Int64(3))],
            })))
        );
        let Ok(Applied::Changed {
            before: Before::Node(image),
            ..
        }) = target.apply(&DataOp::DeleteNode { id: gus }, writer)
        else {
            panic!("a node delete has its image");
        };
        let mut names: Vec<&str> = image.labels.iter().map(ArcStr::as_str).collect();
        names.sort_unstable();
        assert_eq!(names, ["Person", "Traveller"]);
        assert_eq!(image.properties, vec![(key("name"), Value::from("Gus"))]);

        assert_eq!(
            target.stamp(TransactionId::new(19), &mut [].iter(), EpochId::new(3)),
            Ok(())
        );
        assert!(matches!(
            target.undo(TransactionId::new(19), &mut [].iter()),
            Err(ApplyError::Refused(_))
        ));
        let nodes = store.next_node_id();
        assert!(matches!(
            target.reserve_node_ids(1),
            Err(ApplyError::Refused(_))
        ));
        assert!(matches!(
            target.apply(
                &DataOp::CreateNode {
                    id: NodeId::new(nodes),
                    labels: labels(&["Person"]),
                    properties: Vec::new(),
                },
                writer
            ),
            Err(ApplyError::Refused(_))
        ));
        assert_eq!(store.next_node_id(), nodes, "nothing was created");
        assert!(store.get_node(NodeId::new(nodes)).is_none());
    }
}
