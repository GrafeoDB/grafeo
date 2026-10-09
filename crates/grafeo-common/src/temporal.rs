//! Append-only versioned property storage.
//!
//! [`VersionLog<T>`] is a sorted, append-only log of `(EpochId, T)` entries.
//! It is the core data structure behind the `temporal` feature flag, enabling
//! point-in-time property queries without modifying existing MVCC machinery.
//!
//! # Design
//!
//! - Entries are sorted ascending by epoch. Latest value is at the back.
//! - `SmallVec<[_; 1]>` inlines the single-version case (zero heap allocation).
//! - No PENDING/visibility logic inside VersionLog: PENDING epoch semantics
//!   are handled at the PropertyColumn level.

use crate::types::EpochId;
use smallvec::SmallVec;

/// Sorted append-only log of versioned values.
///
/// Entries are sorted by epoch (ascending). The latest value is at the back.
/// Uses `SmallVec` to inline the single-version case (zero heap allocation).
#[derive(Debug, Clone)]
pub struct VersionLog<T> {
    entries: SmallVec<[(EpochId, T); 1]>,
}

impl<T> Default for VersionLog<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> VersionLog<T> {
    /// Creates an empty version log.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: SmallVec::new(),
        }
    }

    /// Creates a version log with a single entry.
    ///
    /// This is the common creation path: most entities have one version.
    /// The entry is stored inline (no heap allocation).
    #[must_use]
    pub fn with_value(epoch: EpochId, value: T) -> Self {
        let mut entries = SmallVec::new();
        entries.push((epoch, value));
        Self { entries }
    }

    /// Returns the latest (most recent) value, or `None` if empty.
    ///
    /// O(1): peeks at the last entry.
    #[must_use]
    pub fn latest(&self) -> Option<&T> {
        self.entries.last().map(|(_, v)| v)
    }

    /// Returns the latest entry as `(epoch, value)`, or `None` if empty.
    ///
    /// O(1): peeks at the last entry.
    #[must_use]
    pub fn latest_entry(&self) -> Option<&(EpochId, T)> {
        self.entries.last()
    }

    /// Returns the epoch of the latest entry, or `None` if empty.
    ///
    /// O(1): peeks at the last entry's epoch.
    #[must_use]
    pub fn latest_epoch(&self) -> Option<EpochId> {
        self.entries.last().map(|(e, _)| *e)
    }

    /// Returns the value at the given epoch.
    ///
    /// Finds the latest entry whose epoch is <= the target epoch.
    /// Returns `None` if the log is empty or all entries are after the target.
    ///
    /// O(1) when the target epoch is at or after the latest entry (the common
    /// case for current-epoch reads). Falls back to O(log N) binary search
    /// for historical queries.
    #[must_use]
    pub fn at(&self, epoch: EpochId) -> Option<&T> {
        let last = self.entries.last()?;
        // Fast path: target epoch is at or after the latest entry.
        // This covers the common case of reading at the current epoch,
        // while correctly skipping PENDING entries (epoch = u64::MAX)
        // when the target is a real epoch.
        if last.0 <= epoch {
            return Some(&last.1);
        }
        // Slow path: historical read, binary search
        let idx = self.entries.partition_point(|(e, _)| *e <= epoch);
        if idx == 0 {
            None
        } else {
            Some(&self.entries[idx - 1].1)
        }
    }

    /// Appends a new version to the log.
    ///
    /// The epoch must be >= the last entry's epoch (ascending order).
    /// O(1) amortized (SmallVec push).
    pub fn append(&mut self, epoch: EpochId, value: T) {
        debug_assert!(
            self.entries.last().map_or(true, |(e, _)| epoch >= *e),
            "VersionLog::append: epoch {epoch:?} is before last entry {:?}",
            self.entries.last().map(|(e, _)| e)
        );
        self.entries.push((epoch, value));
    }

    /// Removes all entries with `EpochId::PENDING` from the back of the log.
    ///
    /// PENDING entries are always at the tail (appended during an uncommitted
    /// transaction). This pops them off in O(1) per entry.
    pub fn remove_pending(&mut self) {
        while self
            .entries
            .last()
            .is_some_and(|(e, _)| *e == EpochId::PENDING)
        {
            self.entries.pop();
        }
    }

    /// Removes up to `n` PENDING entries from the back of the log.
    ///
    /// Used by savepoint rollback to pop only the entries added after the
    /// savepoint, leaving earlier PENDING entries (from before the savepoint)
    /// intact.
    pub fn pop_n_pending(&mut self, n: usize) {
        for _ in 0..n {
            if self
                .entries
                .last()
                .is_some_and(|(e, _)| *e == EpochId::PENDING)
            {
                self.entries.pop();
            } else {
                break;
            }
        }
    }

    /// Replaces `EpochId::PENDING` entries with the real commit epoch.
    ///
    /// Iterates backwards from the end, replacing PENDING epochs with
    /// `real_epoch`, stopping at the first non-PENDING entry.
    pub fn finalize_pending(&mut self, real_epoch: EpochId) {
        for entry in self.entries.iter_mut().rev() {
            if entry.0 == EpochId::PENDING {
                entry.0 = real_epoch;
            } else {
                break;
            }
        }
    }

    /// Discards all entries with epoch > the target.
    ///
    /// Used for savepoint rollback: removes mutations after the savepoint.
    /// O(log N) for the binary search + O(1) amortized for truncation.
    pub fn truncate_after(&mut self, epoch: EpochId) {
        let keep = self.entries.partition_point(|(e, _)| *e <= epoch);
        self.entries.truncate(keep);
    }

    /// Garbage-collects old versions, keeping one baseline before `min_epoch`.
    ///
    /// Retains the latest entry with epoch < `min_epoch` (the baseline visible
    /// to readers at `min_epoch`) plus all entries at or after `min_epoch`.
    /// If all entries are before `min_epoch`, keeps only the last one. Returns
    /// how many entries it dropped.
    pub fn gc(&mut self, min_epoch: EpochId) -> usize {
        if self.entries.len() <= 1 {
            return 0;
        }
        // Find the first entry at or after min_epoch
        let first_recent = self.entries.partition_point(|(e, _)| *e < min_epoch);
        if first_recent == 0 {
            // All entries are at or after min_epoch: nothing to GC
            return 0;
        }
        // Keep one baseline (the entry just before first_recent) plus all recent entries.
        // The baseline is at index first_recent - 1.
        let baseline = first_recent - 1;
        if baseline > 0 {
            self.entries.drain(..baseline);
        }
        baseline
    }

    /// Returns the number of versions in the log.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the log has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns all `(epoch, value)` pairs as a slice.
    ///
    /// Entries are in ascending epoch order.
    #[must_use]
    pub fn history(&self) -> &[(EpochId, T)] {
        self.entries.as_slice()
    }

    /// Returns an iterator over `(epoch, value)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = &(EpochId, T)> {
        self.entries.iter()
    }

    /// Returns whether the internal `SmallVec` has spilled to the heap.
    ///
    /// For testing: a single-version log should not spill.
    #[cfg(test)]
    #[must_use]
    pub fn spilled(&self) -> bool {
        self.entries.spilled()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn epoch(n: u64) -> EpochId {
        EpochId::new(n)
    }

    #[test]
    fn test_empty_log() {
        let log: VersionLog<i32> = VersionLog::new();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
        assert!(log.latest().is_none());
        assert!(log.latest_entry().is_none());
        assert!(log.latest_epoch().is_none());
        assert!(log.at(epoch(0)).is_none());
        assert!(log.history().is_empty(), "{:?}", log.history());
    }

    #[test]
    fn test_single_entry() {
        let log = VersionLog::with_value(epoch(5), 42);
        assert!(!log.is_empty());
        assert_eq!(log.len(), 1);
        assert_eq!(log.latest(), Some(&42));
        assert_eq!(log.latest_epoch(), Some(epoch(5)));
        assert!(!log.spilled());
    }

    #[test]
    fn test_multiple_entries() {
        let mut log = VersionLog::new();
        log.append(epoch(1), 10);
        log.append(epoch(3), 30);
        log.append(epoch(5), 50);

        assert_eq!(log.len(), 3);
        assert_eq!(log.latest(), Some(&50));
        assert_eq!(log.latest_epoch(), Some(epoch(5)));
    }

    #[test]
    fn test_at_exact_match() {
        let mut log = VersionLog::new();
        log.append(epoch(1), 10);
        log.append(epoch(3), 30);
        log.append(epoch(5), 50);

        assert_eq!(log.at(epoch(1)), Some(&10));
        assert_eq!(log.at(epoch(3)), Some(&30));
        assert_eq!(log.at(epoch(5)), Some(&50));
    }

    #[test]
    fn test_at_between_versions() {
        let mut log = VersionLog::new();
        log.append(epoch(1), 10);
        log.append(epoch(3), 30);
        log.append(epoch(5), 50);

        // Between epoch 1 and 3: returns value at epoch 1
        assert_eq!(log.at(epoch(2)), Some(&10));
        // Between epoch 3 and 5: returns value at epoch 3
        assert_eq!(log.at(epoch(4)), Some(&30));
        // After last: returns latest
        assert_eq!(log.at(epoch(100)), Some(&50));
    }

    #[test]
    fn test_at_before_first() {
        let mut log = VersionLog::new();
        log.append(epoch(5), 50);

        assert!(log.at(epoch(0)).is_none());
        assert!(log.at(epoch(4)).is_none());
        assert_eq!(log.at(epoch(5)), Some(&50));
    }

    #[test]
    fn test_append_same_epoch() {
        let mut log = VersionLog::new();
        log.append(epoch(1), 10);
        log.append(epoch(1), 20);

        assert_eq!(log.len(), 2);
        assert_eq!(log.latest(), Some(&20));
        // at(epoch 1) finds the latest entry with epoch <= 1
        // partition_point finds first entry > 1, which is index 2
        // so we return entries[1] = 20
        assert_eq!(log.at(epoch(1)), Some(&20));
    }

    #[test]
    fn test_remove_pending() {
        let mut log = VersionLog::new();
        log.append(epoch(1), 10);
        log.append(EpochId::PENDING, 20);
        log.append(EpochId::PENDING, 30);

        assert_eq!(log.len(), 3);
        log.remove_pending();
        assert_eq!(log.len(), 1);
        assert_eq!(log.latest(), Some(&10));
    }

    #[test]
    fn test_remove_pending_no_pending() {
        let mut log = VersionLog::new();
        log.append(epoch(1), 10);
        log.append(epoch(2), 20);

        log.remove_pending();
        assert_eq!(log.len(), 2);
        assert_eq!(log.latest(), Some(&20));
    }

    #[test]
    fn test_remove_pending_all_pending() {
        let mut log = VersionLog::new();
        log.append(EpochId::PENDING, 10);

        log.remove_pending();
        assert!(log.is_empty());
    }

    #[test]
    fn test_finalize_pending() {
        let mut log = VersionLog::new();
        log.append(epoch(1), 10);
        log.append(EpochId::PENDING, 20);
        log.append(EpochId::PENDING, 30);

        log.finalize_pending(epoch(5));

        assert_eq!(log.len(), 3);
        let history = log.history();
        assert_eq!(history[0].0, epoch(1));
        assert_eq!(history[1].0, epoch(5));
        assert_eq!(history[2].0, epoch(5));
        assert_eq!(log.at(epoch(3)), Some(&10));
        assert_eq!(log.at(epoch(5)), Some(&30));
    }

    #[test]
    fn test_truncate_after() {
        let mut log = VersionLog::new();
        log.append(epoch(1), 10);
        log.append(epoch(3), 30);
        log.append(epoch(5), 50);
        log.append(epoch(7), 70);

        log.truncate_after(epoch(4));

        assert_eq!(log.len(), 2);
        assert_eq!(log.latest(), Some(&30));
        assert_eq!(log.latest_epoch(), Some(epoch(3)));
    }

    #[test]
    fn test_truncate_after_exact_epoch() {
        let mut log = VersionLog::new();
        log.append(epoch(1), 10);
        log.append(epoch(3), 30);
        log.append(epoch(5), 50);

        log.truncate_after(epoch(3));

        assert_eq!(log.len(), 2);
        assert_eq!(log.latest(), Some(&30));
    }

    #[test]
    fn test_truncate_after_before_all() {
        let mut log = VersionLog::new();
        log.append(epoch(5), 50);

        log.truncate_after(epoch(1));

        assert!(log.is_empty());
    }

    #[test]
    fn test_gc_keeps_baseline_and_recent() {
        let mut log = VersionLog::new();
        log.append(epoch(1), 10);
        log.append(epoch(3), 30);
        log.append(epoch(5), 50);
        log.append(epoch(7), 70);

        // GC with min_epoch=5: keep baseline (epoch 3) + entries at/after 5
        log.gc(epoch(5));

        assert_eq!(log.len(), 3);
        let history = log.history();
        assert_eq!(history[0], (epoch(3), 30));
        assert_eq!(history[1], (epoch(5), 50));
        assert_eq!(history[2], (epoch(7), 70));
    }

    #[test]
    fn test_gc_all_old() {
        let mut log = VersionLog::new();
        log.append(epoch(1), 10);
        log.append(epoch(3), 30);

        log.gc(epoch(100));

        // All entries are before min_epoch. baseline = last entry (epoch 3).
        // drain(..1) removes epoch(1), keeping only epoch(3).
        assert_eq!(log.len(), 1);
        assert_eq!(log.latest(), Some(&30));
    }

    #[test]
    fn test_gc_all_recent() {
        let mut log = VersionLog::new();
        log.append(epoch(5), 50);
        log.append(epoch(7), 70);

        // GC with min_epoch=1: all entries are at/after min, nothing to GC
        log.gc(epoch(1));

        assert_eq!(log.len(), 2);
    }

    #[test]
    fn test_gc_single_entry() {
        let mut log = VersionLog::new();
        log.append(epoch(1), 10);

        log.gc(epoch(100));

        // Single entry: never GC'd
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn test_default() {
        let log: VersionLog<i32> = VersionLog::default();
        assert!(log.is_empty());
    }

    #[test]
    fn test_smallvec_inline_single_entry() {
        let log = VersionLog::with_value(epoch(1), 42);
        assert!(!log.spilled());
    }

    #[test]
    fn test_smallvec_spills_on_second_entry() {
        let mut log = VersionLog::with_value(epoch(1), 42);
        log.append(epoch(2), 43);
        assert!(log.spilled());
    }

    #[test]
    fn test_iter() {
        let mut log = VersionLog::new();
        log.append(epoch(1), 10);
        log.append(epoch(3), 30);

        let collected: Vec<_> = log.iter().collect();
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0], &(epoch(1), 10));
        assert_eq!(collected[1], &(epoch(3), 30));
    }

    #[test]
    fn test_history_slice() {
        let mut log = VersionLog::new();
        log.append(epoch(1), "first");
        log.append(epoch(5), "second");

        let history = log.history();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].1, "first");
        assert_eq!(history[1].1, "second");
    }

    #[test]
    fn test_latest_entry() {
        let mut log = VersionLog::new();
        log.append(epoch(3), 30);
        log.append(epoch(7), 70);

        let entry = log.latest_entry().unwrap();
        assert_eq!(entry.0, epoch(7));
        assert_eq!(entry.1, 70);
    }

    #[test]
    fn test_pop_n_pending_partial() {
        let mut log = VersionLog::new();
        log.append(epoch(1), 10);
        log.append(EpochId::PENDING, 20);
        log.append(EpochId::PENDING, 30);
        log.append(EpochId::PENDING, 40);

        // Pop 2 of 3 PENDING entries
        log.pop_n_pending(2);
        assert_eq!(log.len(), 2);
        assert_eq!(log.latest(), Some(&20));
        // The remaining PENDING entry is still there
        assert_eq!(log.latest_epoch(), Some(EpochId::PENDING));
    }

    #[test]
    fn test_pop_n_pending_stops_at_committed() {
        let mut log = VersionLog::new();
        log.append(epoch(1), 10);
        log.append(EpochId::PENDING, 20);

        // Request more pops than PENDING entries: stops at committed
        log.pop_n_pending(5);
        assert_eq!(log.len(), 1);
        assert_eq!(log.latest(), Some(&10));
    }

    #[test]
    fn test_pop_n_pending_zero() {
        let mut log = VersionLog::new();
        log.append(EpochId::PENDING, 10);

        log.pop_n_pending(0);
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn test_clone() {
        let mut log = VersionLog::new();
        log.append(epoch(1), 10);
        log.append(epoch(3), 30);

        let cloned = log.clone();
        assert_eq!(cloned.len(), 2);
        assert_eq!(cloned.latest(), Some(&30));
    }
}
