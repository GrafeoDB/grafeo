//! The store contract's write side: how a graph's backend applies a
//! transaction's writes, commits them and rolls them back
//! ([`GraphBackend`]).
//!
//! A transaction records each write it applies in its
//! [`ChangeSet`](grafeo_common::change::ChangeSet): the op, with its
//! after-image, and what the write replaced. That is the one log: a backend
//! applies an op and reports what it replaced ([`GraphBackend::apply`]), and
//! is handed the transaction's changes of its graph, read-only, to commit
//! them at an epoch ([`GraphBackend::commit`]) or to roll them back
//! ([`GraphBackend::rollback`], [`GraphBackend::rollback_to`]). Replay of
//! the log calls the same `apply`, as a [`Writer::Replay`], so what replay
//! builds cannot drift from what the live writes built.
//!
//! One backend per graph: each named graph is a store of its own. The
//! built-in backend is [`RowGroupStore`](crate::graph::rowgroup::RowGroupStore).

use std::fmt;
use std::ops::Range;

use grafeo_common::change::{Before, Change, ChangeMark, DataOp, Entity, PendingVersion};
use grafeo_common::types::{EpochId, NodeId, ReadView, TransactionId};

use super::traits::GraphStore;

/// How an op is applied: who writes, and so what it becomes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Writer {
    /// A live transaction (also a direct call's private one, and a bulk
    /// write): a pending version only it sees until its commit or rollback.
    /// Lenient: a write that changes nothing returns [`Applied::Unchanged`].
    Transaction {
        /// The transaction.
        id: TransactionId,
        /// The epoch it reads at.
        snapshot: EpochId,
    },
    /// Replay of a logged group: applied and stamped at its epoch, with all
    /// [`GraphBackend::commit`] does, and no before-image built. Strict: a
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
    /// write what it writes, and never commits or rolls back its entries.
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
    /// as a transaction's pending version: a backend stamps such a write as
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

/// What [`GraphBackend::apply`] did.
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

/// Why [`GraphBackend::apply`], `commit` or a rollback refused. On an `apply`
/// error nothing changed; a commit or rollback error is a broken invariant.
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

/// A backend's transaction: the handle [`GraphBackend::begin`] returns,
/// which names the transaction its writes apply in
/// ([`writer`](Self::writer)), its reads see ([`view`](Self::view)), and its
/// commit and rollback end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendTransaction {
    id: TransactionId,
    snapshot: EpochId,
}

impl BackendTransaction {
    /// The handle of transaction `id`, which reads at `snapshot`.
    #[must_use]
    pub const fn new(id: TransactionId, snapshot: EpochId) -> Self {
        Self { id, snapshot }
    }

    /// The transaction.
    #[must_use]
    pub const fn id(self) -> TransactionId {
        self.id
    }

    /// The epoch it reads at.
    #[must_use]
    pub const fn snapshot(self) -> EpochId {
        self.snapshot
    }

    /// The writer its writes apply as.
    #[must_use]
    pub const fn writer(self) -> Writer {
        Writer::Transaction {
            id: self.id,
            snapshot: self.snapshot,
        }
    }

    /// The view its reads take: its snapshot, its own writes included.
    #[must_use]
    pub const fn view(self) -> ReadView {
        ReadView::transaction(self.id, self.snapshot)
    }
}

/// The store contract's write side: a graph's store as the engine writes,
/// commits and rolls back a transaction in it. One per graph. Every method
/// takes `&self`; the backend synchronizes internally and never calls back
/// into the engine.
///
/// # Stability: Experimental
///
/// Implementing this trait (and so building a database on a backend of
/// one's own, `GrafeoDB::with_store`) is experimental through 0.6.x: any
/// release may add or change a required method. Calling it is stable.
///
/// # The transaction
///
/// The backend owns its transactions. The engine opens one with
/// [`begin`](Self::begin) before the transaction's first write to the graph;
/// every write of the transaction then [`apply`](Self::apply)s in it (as the
/// handle's [`writer`](BackendTransaction::writer)), and its reads at the
/// handle's [`view`](BackendTransaction::view) see them, no other reader
/// does. [`commit`](Self::commit) makes them visible at the commit's epoch,
/// [`rollback`](Self::rollback) drops them all, and
/// [`rollback_to`](Self::rollback_to) drops those after a savepoint and keeps
/// the transaction open.
///
/// # One log
///
/// The engine records every write it applies in the transaction's change
/// set: the op, with its after-image, and what the write replaced (its
/// [`Before`]). Commit and the rollbacks hand the backend the recorded
/// changes of its graph, read-only, in the order they were applied: a
/// backend with versions of its own (the row-group store) stamps or drops
/// them, one that writes in place undoes from the before-images, one with
/// native transactions ends them and may ignore the changes. The engine
/// writes the same changes to its write-ahead log and to change data
/// capture, and replays a logged group through [`apply`](Self::apply) as a
/// [`Writer::Replay`], so what recovery builds cannot drift from what the
/// live writes built.
///
/// # Ids
///
/// The engine reserves the ids of what a write creates
/// ([`reserve_node_ids`](Self::reserve_node_ids)) and a create names its id:
/// a backend creates at the id it is given, so a replay of the log creates
/// at the logged ids.
///
/// # Durability
///
/// A database on a backend with a path logs every commit to its own
/// write-ahead log first. At a checkpoint the engine asks the backend to
/// make everything up to an epoch durable ([`flush`](Self::flush)), then
/// drops the logged commits up to the epoch the backend reports durable
/// ([`durable_epoch`](Self::durable_epoch)); at an open it replays the
/// logged commits after that epoch. A backend that keeps nothing across a
/// restart reports [`EpochId::INITIAL`]: the engine then replays the whole
/// log at every open and never drops any of it, so the log is what persists.
///
/// # Errors and invariants
///
/// `apply` allocates everything a version needs, so `commit` only makes the
/// writes visible and the rollbacks only drop or restore what `apply` made.
/// An `Err` from `commit` or a rollback is a broken invariant: the engine
/// poisons the database rather than retry, as a commit of two graphs whose
/// second commit fails is half done. The transaction manager's claims let
/// one transaction at a time write an entity, so the newest pending version
/// of an entity, key or label set is the one a change wrote.
pub trait GraphBackend: GraphStore {
    /// Reserves `count` consecutive node ids, never handed out again (also
    /// not after a rollback).
    ///
    /// # Errors
    ///
    /// Returns [`ApplyError::Refused`] when the ids are exhausted.
    fn reserve_node_ids(&self, count: u64) -> Result<Range<u64>, ApplyError>;

    /// Reserves `count` consecutive edge ids.
    ///
    /// # Errors
    ///
    /// As [`reserve_node_ids`](Self::reserve_node_ids).
    fn reserve_edge_ids(&self, count: u64) -> Result<Range<u64>, ApplyError>;

    /// Opens transaction `transaction`, which reads at `snapshot`, before
    /// its first write to this graph; its writes apply in the returned
    /// handle. A backend whose versions need no opening (the row-group
    /// store) returns the handle at once.
    ///
    /// # Errors
    ///
    /// Returns [`ApplyError::Refused`] when the backend cannot open one;
    /// the engine then fails the write.
    fn begin(
        &self,
        transaction: TransactionId,
        snapshot: EpochId,
    ) -> Result<BackendTransaction, ApplyError>;

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
    /// recorded as one change ([`BulkRange`](grafeo_common::change::BulkRange)),
    /// which its commit commits and its rollback drops as a range. Builds no
    /// before-image (the range is the change), and leaves to the bulk write
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

    /// Commits `transaction` at `epoch`: its writes in this graph, whose
    /// recorded `changes` come in the order they were applied, become
    /// visible to the readers at `epoch` and after, and the handle ends.
    /// O(changes); a bulk range costs O(row groups) in the row-group store,
    /// where a run's version is one record.
    ///
    /// # Errors
    ///
    /// A broken invariant (see the trait docs): the database is poisoned.
    fn commit(
        &self,
        transaction: BackendTransaction,
        changes: &mut dyn Iterator<Item = &Change>,
        epoch: EpochId,
    ) -> Result<(), ApplyError>;

    /// Rolls `transaction` back: none of its writes in this graph stays, and
    /// the handle ends. Its recorded `changes` come in the order they were
    /// applied; a backend that undoes walks them last to first (`next_back`):
    /// it drops the version a change [created](PendingVersion::Created) (or,
    /// writing in place, writes its before-image back) and writes the
    /// before-image back into one it [replaced](PendingVersion::Replaced); a
    /// bulk range drops its pending ids. O(changes).
    ///
    /// # Errors
    ///
    /// A broken invariant (see the trait docs): the database is poisoned.
    fn rollback(
        &self,
        transaction: BackendTransaction,
        changes: &mut dyn DoubleEndedIterator<Item = &Change>,
    ) -> Result<(), ApplyError>;

    /// Rolls `transaction` back to `savepoint`, a position in its change
    /// set: its writes in this graph after the savepoint, whose recorded
    /// `changes_since` come in the order they were applied, are undone as
    /// [`rollback`](Self::rollback) undoes them; the writes before it stay
    /// and the transaction stays open. A savepoint taken before the
    /// transaction's first write to this graph undoes all of them, and a
    /// failed statement inside a transaction rolls back to the position
    /// before it the same way. O(changes since).
    ///
    /// # Errors
    ///
    /// A broken invariant (see the trait docs): the database is poisoned.
    fn rollback_to(
        &self,
        transaction: BackendTransaction,
        savepoint: ChangeMark,
        changes_since: &mut dyn DoubleEndedIterator<Item = &Change>,
    ) -> Result<(), ApplyError>;

    /// Drops what no reader at `horizon` or later needs: the versions every
    /// reader sees past. The engine calls it after commits, now and then,
    /// with the oldest epoch an open reader or the history retention still
    /// reads. A backend without versions of its own has nothing to drop.
    fn collect(&self, horizon: EpochId);

    /// The latest epoch whose commits the backend holds durably: the commits
    /// at it and before it survive a restart of the process, and an open
    /// replays the logged commits after it. [`EpochId::INITIAL`] for a
    /// backend that keeps nothing across a restart: an open replays the
    /// whole log, and the engine never drops any of it.
    fn durable_epoch(&self) -> EpochId;

    /// Makes every commit up to `epoch` (the database's committed epoch at a
    /// checkpoint) durable, so [`durable_epoch`](Self::durable_epoch)
    /// reports `epoch` or later afterwards; then the engine drops the
    /// logged commits up to it. A backend that keeps nothing across a
    /// restart returns `Ok` and keeps reporting [`EpochId::INITIAL`]: the
    /// log keeps every commit.
    ///
    /// # Errors
    ///
    /// When the backend cannot make the commits durable: the checkpoint
    /// fails, and the log keeps every commit.
    fn flush(&self, epoch: EpochId) -> Result<(), ApplyError>;
}

/// Refuses what no bulk write applies as a row (see
/// [`GraphBackend::apply_bulk_row`]): an op that is no create, or a writer
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
