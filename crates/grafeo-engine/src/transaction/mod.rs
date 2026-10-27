//! Transaction management with MVCC and configurable isolation levels.
//!
//! # Isolation Levels
//!
//! Sessions accept `ReadCommitted` and the default `SnapshotIsolation`.
//! They refuse `Serializable` with `GRAFEO-Q004` until query read tracking
//! is implemented (#400).
//!
//! | Level | Session behavior |
//! |-------|------------------|
//! | [`ReadCommitted`](IsolationLevel::ReadCommitted) | Accepted; queries use the transaction's snapshot |
//! | [`SnapshotIsolation`](IsolationLevel::SnapshotIsolation) | Default; a consistent snapshot, with write skew possible |
//! | [`Serializable`](IsolationLevel::Serializable) | Refused until query read tracking is available |
//!
//! The default is **Snapshot Isolation (SI)**, which offers strong consistency
//! guarantees while maintaining high concurrency. Each transaction sees a consistent
//! snapshot of the database as of its start time.
//!
//! ## Guarantees
//!
//! - **Read Consistency**: A transaction always reads the same values for the same
//!   entities throughout its lifetime (repeatable reads).
//! - **Write-Write Conflict Detection**: A conflicting write is refused before
//!   mutation; commit also checks writes committed after the transaction began.
//! - **No Dirty Reads**: A transaction never sees uncommitted changes from other
//!   transactions.
//! - **No Lost Updates**: Write-write conflicts prevent the lost update anomaly.
//!
//! ## Limitations (Write Skew Anomaly)
//!
//! Snapshot Isolation does **not** provide full Serializable isolation. The "write skew"
//! anomaly is possible when two transactions read overlapping data but write to
//! disjoint sets:
//!
//! ```text
//! Initial state: A = 50, B = 50, constraint: A + B >= 0
//!
//! T1: Read A, B → both 50
//! T2: Read A, B → both 50
//! T1: Write A = A - 100 = -50 (valid: -50 + 50 = 0)
//! T2: Write B = B - 100 = -50 (valid: 50 + -50 = 0)
//! T1: Commit ✓
//! T2: Commit ✓ (no write-write conflict since T1 wrote A, T2 wrote B)
//!
//! Final state: A = -50, B = -50, constraint violated: A + B = -100 < 0
//! ```
//!
//! ## Workarounds for Write Skew
//!
//! If your application has invariants that span multiple entities, consider:
//!
//! 1. **Promote reads to writes**: Add a dummy write to entities you read if you
//!    need them to remain unchanged.
//! 2. **Application-level locking**: Use external locks for critical sections.
//! 3. **Constraint checking**: Validate invariants at commit time and retry if
//!    violated.
//!
//! ## Epoch-Based Versioning
//!
//! Grafeo uses epoch-based MVCC where:
//! - A commit reserves an epoch, published only when the commit completes
//! - Transactions read data visible at their start epoch and their own writes
//! - Version chains store multiple versions for concurrent access
//! - Garbage collection retains versions needed by transactions and registered read pins
//!
//! # Implementation guide for contributors
//!
//! This section ties together the MVCC pieces split across
//! `grafeo-common` (metadata types) and `grafeo-engine` (lifecycle +
//! enforcement). It is intentionally prose, not API reference; use the
//! item links to jump into the code.
//!
//! ## Visibility invariants
//!
//! The common [`VersionInfo`] helper records `created_epoch` and an
//! optional `deleted_epoch`. Its `is_visible_at` checks committed epoch
//! visibility; `is_visible_to` also checks transaction ownership for reading
//! one's own writes and deletes. These helpers serve the common version
//! chains; they are not the built-in store's row format or its hot-path reads.
//!
//! ## Uncommitted writes and `EpochId::PENDING`
//!
//! Common version chains use [`grafeo_common::types::EpochId::PENDING`]
//! (`u64::MAX`) for uncommitted creation epochs. Their owner-aware helper
//! admits the writer's own versions while the committed-epoch check excludes
//! pending versions.
//!
//! The built-in [`RowGroupStore`](grafeo_core::graph::rowgroup::RowGroupStore)
//! instead encodes a pending transaction id in a stamp with its high bit set
//! (`1 << 63`). Its transaction reader sees its own pending stamps; a committed
//! reader sees only committed stamps at or before its selected epoch. The
//! store's `u64::MAX` stamp means absent, not a pending creation epoch.
//!
//! Every write of a transaction is recorded in its change set (see
//! `TransactionChanges`): the op with its after-image, what it replaced,
//! and what it did to the transaction's pending version. On commit, each
//! graph's store stamps the entries recorded for it with the commit epoch;
//! a rollback, or a rollback to a savepoint, undoes the entries after its
//! position, last to first, through the same stores. Neither scans a
//! store: both cost what the transaction changed. So neither state leaks
//! into a concurrent reader's snapshot.
//!
//! ## Write-write conflict detection
//!
//! A [`GraphWriter`](grafeo_core::execution::operators::GraphWriter) claims
//! what each write changes through its change recorder *before* it touches
//! the store; the engine's recorder forwards to
//! [`TransactionManager::record_write`] (and `record_delete`,
//! `record_endpoints`), which enforces **first-writer-wins**: if another
//! active transaction already has the entity in its write set, the current
//! transaction fails with `OperatorError::WriteConflict` before the store
//! mutates. A direct database call uses a private transaction while other
//! transactions are open, so their claims conflict in either direction.
//! With none open it can hold the idle gate and write at once; no transaction
//! can start until that call ends. A node a write creates is not
//! claimed: no other transaction knows its id before the commit.
//!
//! Write claims refuse a conflicting mutation synchronously; commit-time
//! validation also checks transactions that committed since the snapshot.
//! The low-level manager can validate entity reads supplied explicitly through
//! [`TransactionManager::record_read`]. Queries do not supply those reads yet,
//! so sessions refuse `Serializable`; manual read-set validation is not a
//! query serializability guarantee.
//!
//! ## Epoch lifecycle
//!
//! The manager keeps separate assigned and published epochs. A reserved
//! epoch does not become a read snapshot until its commit completes:
//!
//! 1. **Begin**: the transaction snapshots the latest published epoch. Its
//!    snapshot is read and counted under the read-pins lock, so collection
//!    cannot pass a reader that registration has not counted yet.
//! 2. **Write**: the store records pending versions owned by the transaction,
//!    while the change recorder keeps its operations and undo information.
//! 3. **Commit decision**: conflict validation reserves the next assigned
//!    epoch under the commit lock and leaves the transaction `Committing`.
//!    New transactions may still begin at the published epoch.
//! 4. **Log and publication**: when logging, a session writes its WAL group
//!    before stamping or reporting the changes. Once written, the commit lock may be released
//!    while publication waits for lower epochs. In its publication turn the
//!    engine stamps the recorded changes and reports events, then publishes
//!    the completed epoch and releases the transaction's snapshot. A failed
//!    incomplete commit prevents publication at it or any higher epoch.
//! 5. **GC**: after writes and in [`GrafeoDB::gc()`](crate::GrafeoDB::gc),
//!    `collect_horizon()` chooses the older of the read horizon and the history
//!    retention window. The read horizon counts open/committing transaction
//!    snapshots and all registered read pins, including historical overrides
//!    inside transactions, and never passes the published epoch. Collection
//!    keeps versions needed at that horizon or later and advances the history
//!    floor. CDC retention uses the same epoch (see [`crate::cdc`]).
//!
//! Readers use the completed published epoch, not an assigned epoch or a
//! store's local stamp. Reserving an epoch and publishing it are separate
//! steps; an atomic update of the assigned counter alone publishes nothing.
//!
//! ## Where to look next
//!
//! - `transaction::manager` (private module): public API, begin/commit
//!   paths, conflict detector. Re-exports: [`TransactionManager`],
//!   [`TransactionInfo`], [`TransactionState`], [`IsolationLevel`].
//! - `transaction::mvcc` (private module): `VersionChain` container,
//!   per-entity version history. Re-exports: [`VersionChain`],
//!   [`VersionInfo`].
//! - [`grafeo_common::mvcc`]: the visibility functions.
//! - [`grafeo_common::types::EpochId`]: `PENDING`, the sentinel that
//!   makes pending-epoch dirty reads impossible.
//! - [`crate::cdc`]: the commit path that emits change events at the
//!   finalised epoch.
//!
//! # Example
//!
//! ```no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use grafeo_engine::GrafeoDB;
//!
//! let db = GrafeoDB::new_in_memory();
//! let mut session = db.session();
//!
//! session.begin_transaction()?;
//!
//! // All reads see a consistent snapshot
//! let result = session.execute("MATCH (n:Person) RETURN n")?;
//!
//! // Writes are isolated until commit
//! session.execute("INSERT (:Person {name: 'Alix'})")?;
//!
//! // Commit may fail if write-write conflict detected
//! session.commit()?;
//! # Ok(())
//! # }
//! ```

mod changes;
mod history;
mod manager;
mod mvcc;
mod prepared;
#[cfg(feature = "triple-store")]
mod rdf;

pub(crate) use changes::{BuiltIndex, StandaloneChange};
pub(crate) use changes::{StreamingChanges, TransactionChanges};
pub(crate) use history::HistoryImage;
#[cfg(feature = "wal")]
pub(crate) use manager::CommitGuard;
pub(crate) use manager::ReadPin;
pub(crate) use manager::{ClaimMark, CommitsHeld};
pub use manager::{
    EntityId, GraphEntity, IsolationLevel, KeyClaim, TransactionInfo, TransactionManager,
    TransactionState,
};
#[doc(hidden)]
pub use mvcc::{VersionChain, VersionInfo};
pub use prepared::{CommitInfo, PreparedCommit};
#[cfg(all(feature = "triple-store", feature = "cdc"))]
pub(crate) use rdf::ntriples_terms;
#[cfg(feature = "triple-store")]
pub(crate) use rdf::{RdfWriter, update_privately};
