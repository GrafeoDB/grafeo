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
//! # Example
//!
//! ```ignore
//! let tx = session.begin_transaction()?;
//!
//! // All reads see a consistent snapshot
//! let result = session.execute("MATCH (n:Person) RETURN n")?;
//!
//! // Writes are isolated until commit
//! session.execute("CREATE (n:Person {name: 'Alice'})")?;
//!
//! // Commit may fail if write-write conflict detected
//! session.commit()?;
//! ```

mod manager;
mod mvcc;
#[cfg(feature = "block-stm")]
pub mod parallel;

pub use manager::{EntityId, IsolationLevel, TransactionManager, TxInfo, TxState};
pub use mvcc::{Version, VersionChain, VersionInfo};

#[cfg(feature = "block-stm")]
pub use parallel::{BatchRequest, BatchResult, ExecutionStatus, ParallelExecutor};
