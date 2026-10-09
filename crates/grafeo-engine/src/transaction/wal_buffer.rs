//! Per-session buffer of WAL records.
//!
//! Records carry no transaction id, and recovery keeps a single buffer of
//! pending records that the next commit or abort marker closes. If sessions
//! wrote their records as they happen, records of concurrent transactions
//! would interleave and one session's marker would commit or discard another
//! session's records (#411).
//!
//! A [`WalBuffer`] collects a session's records instead, and writes them as
//! one contiguous group at commit, followed by the commit marker. A rollback
//! clears the buffer and writes nothing; a rollback to a savepoint truncates
//! it. Writes outside a transaction are written as an implicit group with its
//! own commit marker.
//!
//! Each group carries its own named-graph context: it emits `SwitchGraph`
//! before records of another graph and switches back to the default graph
//! before its markers, so replay of every group starts and ends in the
//! default graph.

use std::sync::Arc;

use grafeo_common::types::TransactionId;
#[cfg(feature = "lpg")]
use grafeo_common::utils::error::Error;
use grafeo_common::utils::error::Result;
#[cfg(feature = "lpg")]
use grafeo_storage::wal::GroupError;
use grafeo_storage::wal::{LpgWal, WalRecord};
use parking_lot::Mutex;

/// A record waiting for its group, with the named graph it applies to
/// (`None` = default graph).
type PendingRecord = (Option<String>, WalRecord);

/// Whether the caller can undo and continue, or must fence the database.
#[cfg(feature = "lpg")]
#[derive(Debug)]
pub(crate) enum WalCommitFailure {
    /// No commit marker can close this group, and the writer remains usable.
    NotWritten(Error),
    /// A marker may exist: recovery must determine the durable outcome.
    OutcomeUnknown(Error),
    /// The writer cannot safely accept another group before recovery.
    Unavailable(Error),
}

#[cfg(feature = "lpg")]
impl WalCommitFailure {
    fn from_error(error: Error) -> Self {
        if let Error::Io(source) = &error {
            return match source
                .get_ref()
                .and_then(|source| source.downcast_ref::<GroupError>())
            {
                Some(GroupError::NotWritten(_)) => Self::NotWritten(error),
                Some(GroupError::OutcomeUnknown(_)) => Self::OutcomeUnknown(error),
                // Unknown future dispositions and unclassified I/O errors
                // must not permit an unsafe subsequent commit.
                _ => Self::Unavailable(error),
            };
        }
        // Serialization and frame-size validation fail before the writer
        // starts. Errors after writing have a typed I/O disposition.
        Self::NotWritten(error)
    }
}

/// Buffers one session's WAL records until they are written as a group.
pub(crate) struct WalBuffer {
    wal: Arc<LpgWal>,
    pending: Mutex<Vec<PendingRecord>>,
}

impl WalBuffer {
    /// Creates an empty buffer writing to `wal`.
    pub(crate) fn new(wal: Arc<LpgWal>) -> Self {
        Self {
            wal,
            pending: Mutex::new(Vec::new()),
        }
    }

    /// The WAL this buffer writes to.
    pub(crate) fn wal(&self) -> &Arc<LpgWal> {
        &self.wal
    }

    /// Adds a record for `graph` (`None` = default graph).
    pub(crate) fn push(&self, graph: Option<String>, record: WalRecord) {
        self.pending.lock().push((graph, record));
    }

    /// Number of buffered records, used as a savepoint position.
    pub(crate) fn len(&self) -> usize {
        self.pending.lock().len()
    }

    /// Drops the records added after position `len` (savepoint rollback).
    pub(crate) fn truncate(&self, len: usize) {
        self.pending.lock().truncate(len);
    }

    /// Drops every buffered record (rollback).
    pub(crate) fn clear(&self) {
        self.pending.lock().clear();
    }

    /// Writes the buffered records as one group, closed by `markers`.
    ///
    /// The markers are written even when no record is buffered.
    ///
    /// # Errors
    ///
    /// Returns an error if the WAL write fails. The buffered records are
    /// dropped either way.
    #[cfg(test)]
    pub(crate) fn flush(&self, markers: &[WalRecord]) -> Result<()> {
        self.write_group(&mut self.pending.lock(), markers)
    }

    /// Flushes a commit without losing its WAL failure disposition.
    #[cfg(feature = "lpg")]
    pub(crate) fn flush_commit(
        &self,
        markers: &[WalRecord],
    ) -> std::result::Result<(), WalCommitFailure> {
        self.write_group(&mut self.pending.lock(), markers)
            .map_err(WalCommitFailure::from_error)
    }

    /// Writes buffered records from outside a transaction as an implicit
    /// group with its own commit marker. Does nothing when the buffer is empty.
    ///
    /// # Errors
    ///
    /// Returns an error if the WAL write fails.
    pub(crate) fn flush_implicit(&self) -> Result<()> {
        let mut pending = self.pending.lock();
        if pending.is_empty() {
            return Ok(());
        }
        self.write_group(
            &mut pending,
            &[WalRecord::TransactionCommit {
                transaction_id: TransactionId::SYSTEM,
            }],
        )
    }

    /// Writes `pending` as a group closed by `markers` and empties it. The
    /// caller holds the buffer lock through the write, so a flush from
    /// another thread cannot write later records first.
    fn write_group(&self, pending: &mut Vec<PendingRecord>, markers: &[WalRecord]) -> Result<()> {
        let group = build_group(std::mem::take(pending), markers);
        if group.is_empty() {
            return Ok(());
        }
        self.wal.log_batch(&group)
    }
}

/// Builds a group: the records with `SwitchGraph` wherever the graph changes,
/// a switch back to the default graph if needed, then the markers.
fn build_group(pending: Vec<PendingRecord>, markers: &[WalRecord]) -> Vec<WalRecord> {
    let mut group = Vec::with_capacity(pending.len() + markers.len() + 2);
    let mut context: Option<String> = None;
    for (graph, record) in pending {
        if graph != context {
            group.push(WalRecord::SwitchGraph {
                name: graph.clone(),
            });
            context = graph;
        }
        group.push(record);
    }
    if context.is_some() {
        group.push(WalRecord::SwitchGraph { name: None });
    }
    group.extend(markers.iter().cloned());
    group
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::types::{EpochId, NodeId};

    fn create(id: u64) -> WalRecord {
        WalRecord::CreateNode {
            id: NodeId::new(id),
            labels: vec!["N".to_string()],
        }
    }

    fn commit() -> WalRecord {
        WalRecord::TransactionCommit {
            transaction_id: TransactionId::new(7),
        }
    }

    /// Short form of a group for assertions.
    fn shape(group: &[WalRecord]) -> Vec<String> {
        group
            .iter()
            .map(|record| match record {
                WalRecord::CreateNode { id, .. } => format!("node {}", id.as_u64()),
                WalRecord::SwitchGraph { name } => format!("switch {name:?}"),
                WalRecord::TransactionCommit { .. } => "commit".to_string(),
                WalRecord::EpochAdvance { epoch } => format!("epoch {}", epoch.as_u64()),
                other => format!("{other:?}"),
            })
            .collect()
    }

    #[test]
    fn default_graph_group_has_no_switches() {
        let group = build_group(vec![(None, create(1)), (None, create(2))], &[commit()]);
        assert_eq!(shape(&group), ["node 1", "node 2", "commit"]);
    }

    #[test]
    fn group_switches_graphs_and_returns_to_default() {
        let pending = vec![
            (Some("g".to_string()), create(1)),
            (Some("g".to_string()), create(2)),
            (None, create(3)),
            (Some("h".to_string()), create(4)),
        ];
        let markers = [
            commit(),
            WalRecord::EpochAdvance {
                epoch: EpochId::new(5),
            },
        ];
        assert_eq!(
            shape(&build_group(pending, &markers)),
            [
                "switch Some(\"g\")",
                "node 1",
                "node 2",
                "switch None",
                "node 3",
                "switch Some(\"h\")",
                "node 4",
                "switch None",
                "commit",
                "epoch 5",
            ]
        );
    }

    #[test]
    fn empty_group_is_only_markers() {
        assert_eq!(shape(&build_group(Vec::new(), &[commit()])), ["commit"]);
        assert!(build_group(Vec::new(), &[]).is_empty());
    }

    #[test]
    fn truncate_and_clear() {
        let dir = tempfile::tempdir().unwrap();
        let buffer = WalBuffer::new(Arc::new(LpgWal::open(dir.path()).unwrap()));
        buffer.push(None, create(1));
        let savepoint = buffer.len();
        buffer.push(None, create(2));
        buffer.push(None, create(3));
        buffer.truncate(savepoint);
        assert_eq!(buffer.len(), 1);
        buffer.clear();
        assert_eq!(buffer.len(), 0);
        // Nothing buffered: an implicit flush writes nothing.
        buffer.flush_implicit().unwrap();
        assert_eq!(buffer.wal().record_count(), 0);
    }

    /// Threads writing through one buffer: a flush can take another thread's
    /// records, but every record still reaches the WAL in the order it was
    /// pushed. A stress test: with the lock released before the write, about
    /// two runs in five failed.
    #[test]
    fn concurrent_flushes_keep_the_push_order() {
        const PER_THREAD: u64 = 20_000;

        let dir = tempfile::tempdir().unwrap();
        let buffer = Arc::new(WalBuffer::new(Arc::new(LpgWal::open(dir.path()).unwrap())));
        let writers: Vec<_> = (0..8)
            .map(|thread| {
                let buffer = Arc::clone(&buffer);
                std::thread::spawn(move || {
                    for i in 0..PER_THREAD {
                        buffer.push(None, create(thread * PER_THREAD + i));
                        buffer.flush_implicit().unwrap();
                    }
                })
            })
            .collect();
        for writer in writers {
            writer.join().unwrap();
        }
        buffer.wal().flush().unwrap();

        let ids: Vec<u64> = grafeo_storage::wal::WalRecovery::new(dir.path())
            .recover()
            .unwrap()
            .into_iter()
            .filter_map(|record| match record {
                WalRecord::CreateNode { id, .. } => Some(id.as_u64()),
                _ => None,
            })
            .collect();
        assert_eq!(ids.len() as u64, 8 * PER_THREAD);
        for thread in 0..8 {
            let own: Vec<u64> = ids
                .iter()
                .copied()
                .filter(|id| id / PER_THREAD == thread)
                .collect();
            let out_of_order = own.windows(2).find(|pair| pair[0] > pair[1]);
            assert_eq!(out_of_order, None, "thread {thread}");
        }
    }
}
