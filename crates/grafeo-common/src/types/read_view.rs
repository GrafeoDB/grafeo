//! Who reads a store, and at what epoch.

use super::{EpochId, TransactionId};

/// Who reads: a transaction at its snapshot, its own writes included, or a
/// committed reader at an epoch (a read outside a transaction, a read of a
/// past epoch).
///
/// The read traits of a store take a view on every read that must answer
/// as of a snapshot (scans, label scans, index lookups, batch reads,
/// adjacency, search): the store decides what the reader sees.
///
/// # Examples
///
/// ```
/// use grafeo_common::types::{EpochId, ReadView, TransactionId};
///
/// let snapshot = ReadView::transaction(TransactionId::new(7), EpochId::new(3));
/// assert_eq!(snapshot.transaction, Some(TransactionId::new(7)));
/// assert_eq!(snapshot.epoch, EpochId::new(3));
///
/// let committed = ReadView::committed(EpochId::new(3));
/// assert_eq!(committed.transaction, None);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReadView {
    /// The epoch the reader reads at: a transaction's snapshot, or the epoch
    /// of a committed reader.
    pub epoch: EpochId,
    /// The transaction whose own writes the reader sees, if any.
    pub transaction: Option<TransactionId>,
}

impl ReadView {
    /// A view at `epoch`, of `transaction` when there is one: the pair the
    /// engine's operators carry.
    #[inline]
    #[must_use]
    pub const fn new(epoch: EpochId, transaction: Option<TransactionId>) -> Self {
        Self { epoch, transaction }
    }

    /// Transaction `id` at its snapshot `snapshot`: it sees what committed at
    /// or before the snapshot, and its own writes.
    #[inline]
    #[must_use]
    pub const fn transaction(id: TransactionId, snapshot: EpochId) -> Self {
        Self {
            epoch: snapshot,
            transaction: Some(id),
        }
    }

    /// A committed reader at `epoch`: it sees what committed at or before
    /// `epoch`, and no transaction's pending writes.
    #[inline]
    #[must_use]
    pub const fn committed(epoch: EpochId) -> Self {
        Self {
            epoch,
            transaction: None,
        }
    }

    /// A committed reader after every commit a store holds: what a write's
    /// conflict checks read (an edge committed after a transaction's
    /// snapshot still keeps its node from a delete). A query never reads at
    /// it: its view is pinned, and this one is not.
    #[inline]
    #[must_use]
    pub const fn latest() -> Self {
        Self::committed(EpochId::LATEST)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_keeps_the_pair_the_engine_carries() {
        let epoch = EpochId::new(19);
        assert_eq!(
            ReadView::new(epoch, Some(TransactionId::new(88))),
            ReadView::transaction(TransactionId::new(88), epoch)
        );
        assert_eq!(ReadView::new(epoch, None), ReadView::committed(epoch));
    }

    #[test]
    fn the_latest_reader_reads_after_every_epoch_and_is_committed() {
        let latest = ReadView::latest();
        assert_eq!(latest.transaction, None);
        assert!(latest.epoch > EpochId::new(u64::from(u32::MAX)));
        assert!(latest.epoch < EpochId::PENDING);
    }
}
