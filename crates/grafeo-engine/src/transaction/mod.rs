//! Transaction management with MVCC and configurable isolation levels.
//!
//! # Isolation Levels
//!
//! Grafeo supports multiple isolation levels to balance consistency vs. performance:
//!
//! | Level | Anomalies Prevented | Use Case |
//! |-------|---------------------|----------|
//! | [`ReadCommitted`](IsolationLevel::ReadCommitted) | Dirty reads | High throughput, relaxed consistency |
//! | [`SnapshotIsolation`](IsolationLevel::SnapshotIsolation) | + Non-repeatable reads, phantom reads | Default - good balance |
//! | [`Serializable`](IsolationLevel::Serializable) | + Write skew | Full ACID, financial/critical workloads |
//!
//! The default is **Snapshot Isolation (SI)**, which offers strong consistency
//! guarantees while maintaining high concurrency. Each transaction sees a consistent
//! snapshot of the database as of its start time.
//!
//! ## Guarantees
//!
//! - **Read Consistency**: A transaction always reads the same values for the same
//!   entities throughout its lifetime (repeatable reads).
//! - **Write-Write Conflict Detection**: If two concurrent transactions write to the
//!   same entity, the second to commit will be aborted.
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
//! - Each commit advances the global epoch
//! - Transactions read data visible at their start epoch
//! - Version chains store multiple versions for concurrent access
//! - Garbage collection removes versions no longer needed by active transactions
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
//! Every version carries a [`VersionInfo`] with two epoch slots:
//! `created_epoch` and `deleted_epoch`
//! (`Option<EpochId>`). Two free functions decide visibility, and they
//! have different callers for a reason:
//!
//! - **`VersionInfo::is_visible_at(viewing_epoch)`** is the *snapshot*
//!   check. It ignores transaction identity entirely: a version is
//!   visible if `created_epoch <= viewing_epoch` and it is not deleted
//!   at or before `viewing_epoch`. Callers: GC ("is any active
//!   transaction still able to see this?"), epoch-scoped scans, and
//!   post-commit reads from the layered store.
//!
//! - **`VersionInfo::is_visible_to(viewing_epoch, viewing_tx)`** layers
//!   "read your own writes" on top. It first rules out versions the
//!   current transaction has deleted (even before commit), then treats
//!   own-writes as visible regardless of epoch, and otherwise falls
//!   through to `is_visible_at`. Callers: every read on the hot path
//!   inside an active transaction.
//!
//! The two functions share the epoch check to keep the semantics in
//! sync; when you need to change one, inspect both.
//!
//! ## Uncommitted writes and `EpochId::PENDING`
//!
//! An active transaction stamps its own versions with
//! [`grafeo_common::types::EpochId::PENDING`] (`= u64::MAX`) as their
//! `created_epoch`. This is the dirty-read
//! guard: any external reader (different transaction, or a snapshot at
//! a real epoch) performs `created_epoch <= viewing_epoch`, and since
//! `u64::MAX` is always greater than any real epoch, the version is
//! filtered out. The owning transaction still sees it because
//! `is_visible_to` short-circuits on `created_by == viewing_tx`.
//!
//! On commit, the session walks the transaction's write set and
//! replaces every `PENDING` epoch with the real commit epoch
//! (see [`crate::transaction::TransactionWriteTracker`] and the
//! `finalize_*` paths in `TransactionManager`). A rollback drops the
//! pending versions instead of finalising them, so neither state
//! leaks into a concurrent reader's snapshot.
//!
//! ## Write-write conflict detection
//!
//! [`TransactionWriteTracker`] is the bridge between execution-time
//! mutations and the manager's conflict detector. Every mutation
//! operator (node set/delete, edge set/delete) in
//! [`grafeo_core::execution::operators`] goes through a `WriteTracker`
//! trait call *before* touching the store. The engine's implementation
//! forwards to [`TransactionManager::record_write`], which enforces
//! **first-writer-wins**: if another active transaction already has
//! the entity in its write set, the current transaction aborts with
//! `OperatorError::WriteConflict` before the store mutates.
//!
//! This is a commit-time contract, not a best-effort check: the
//! mutation is refused synchronously, so the overlay store never
//! contains state from a doomed transaction. Snapshot isolation
//! (the default) validates only the write-set; `Serializable` also
//! validates the read-set at commit via the same machinery.
//!
//! ## Epoch lifecycle
//!
//! The manager owns a single `AtomicU64` current epoch, observed by
//! every reader via `TransactionManager::current_epoch()`. The
//! lifecycle for a transaction touching epoch `E`:
//!
//! 1. **Allocation**: `begin()` stamps the new transaction with the
//!    current epoch as its `start_epoch`. Reads during the transaction
//!    use this epoch for `is_visible_at`.
//! 2. **Write**: new versions get `EpochId::PENDING`. Writes are
//!    recorded via `record_write` for conflict detection.
//! 3. **Commit**: a successful commit fetches a new epoch via
//!    `fetch_add(1)`, finalises every pending version to that epoch,
//!    and records `(transaction_id -> commit_epoch)` for post-hoc
//!    conflict checks from transactions whose `start_epoch` preceded
//!    the commit.
//! 4. **GC**: [`GrafeoDB::gc()`](crate::GrafeoDB::gc) polls
//!    `min_active_epoch()` and prunes version chains whose last
//!    version is older than that minimum. The same epoch drives CDC
//!    retention (see [`crate::cdc`]) so both subsystems agree on
//!    "how old is safely-forgotten."
//!
//! Every epoch the manager advances is visible to every reader once
//! the atomic store retires; there is no separate publication step.
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

mod manager;
mod mvcc;
#[cfg(feature = "parallel")]
pub mod parallel;
#[cfg(feature = "lpg")]
mod prepared;

pub use manager::{
    EntityId, IsolationLevel, TransactionInfo, TransactionManager, TransactionState,
};
#[doc(hidden)]
pub use mvcc::{VersionChain, VersionInfo};
#[cfg(feature = "lpg")]
pub use prepared::{CommitInfo, PreparedCommit};
pub use write_tracker::TransactionWriteTracker;

mod write_tracker;

#[cfg(feature = "wal")]
pub(crate) mod wal_buffer;

#[cfg(feature = "parallel")]
pub use parallel::{BatchRequest, BatchResult, ExecutionStatus, ParallelExecutor};
