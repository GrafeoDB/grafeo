//! MVCC (Multi-Version Concurrency Control) primitives.
//!
//! This is how Grafeo handles concurrent reads and writes without blocking.
//! Each entity has a [`VersionChain`] that tracks all versions. Readers see
//! consistent snapshots, writers create new versions, and old versions get
//! garbage collected when no one needs them anymore.

use std::collections::VecDeque;

use crate::types::{EpochId, TransactionId};

/// Tracks when a version was created and deleted for visibility checks.
#[derive(Debug, Clone, Copy)]
pub struct VersionInfo {
    /// The epoch this version was created in.
    pub created_epoch: EpochId,
    /// The epoch this version was deleted in (if any).
    pub deleted_epoch: Option<EpochId>,
    /// The transaction that created this version.
    pub created_by: TransactionId,
    /// The transaction that deleted this version (if any).
    /// Used for rollback: only the deleting transaction's rollback can undelete.
    pub deleted_by: Option<TransactionId>,
}

impl VersionInfo {
    /// Creates a new version info.
    #[must_use]
    pub fn new(created_epoch: EpochId, created_by: TransactionId) -> Self {
        Self {
            created_epoch,
            deleted_epoch: None,
            created_by,
            deleted_by: None,
        }
    }

    /// Marks this version as deleted by a specific transaction.
    pub fn mark_deleted(&mut self, epoch: EpochId, deleted_by: TransactionId) {
        self.deleted_epoch = Some(epoch);
        self.deleted_by = Some(deleted_by);
    }

    /// Unmarks deletion if deleted by the given transaction. Returns `true` if undeleted.
    ///
    /// Used during rollback to restore versions deleted within the rolled-back transaction.
    pub fn unmark_deleted_by(&mut self, transaction_id: TransactionId) -> bool {
        if self.deleted_by == Some(transaction_id) {
            self.deleted_epoch = None;
            self.deleted_by = None;
            true
        } else {
            false
        }
    }

    /// Checks if this version is visible at the given epoch.
    #[inline]
    #[must_use]
    pub fn is_visible_at(&self, epoch: EpochId) -> bool {
        // Visible if created before or at the viewing epoch
        // and not deleted before the viewing epoch
        if !self.created_epoch.is_visible_at(epoch) {
            return false;
        }

        if let Some(deleted) = self.deleted_epoch {
            // Not visible if deleted at or before the viewing epoch
            deleted.as_u64() > epoch.as_u64()
        } else {
            true
        }
    }

    /// Checks if this version is visible to a specific transaction.
    ///
    /// A version is visible to a transaction if:
    /// 1. It was created by the same transaction and not deleted by self, OR
    /// 2. It was created in an epoch before the transaction's start epoch
    ///    and not deleted before that epoch, and not deleted by this transaction
    #[inline]
    #[must_use]
    pub fn is_visible_to(&self, viewing_epoch: EpochId, viewing_tx: TransactionId) -> bool {
        // If this version was deleted by the viewing transaction, it is not visible
        if self.deleted_by == Some(viewing_tx) {
            return false;
        }

        // Own modifications are always visible (if not deleted)
        if self.created_by == viewing_tx {
            return self.deleted_epoch.is_none();
        }

        // Otherwise, use epoch-based visibility
        self.is_visible_at(viewing_epoch)
    }
}

/// A single version of data.
#[derive(Debug, Clone)]
pub struct Version<T> {
    /// Visibility metadata.
    pub info: VersionInfo,
    /// The actual data.
    pub data: T,
}

impl<T> Version<T> {
    /// Creates a new version.
    #[must_use]
    pub fn new(data: T, created_epoch: EpochId, created_by: TransactionId) -> Self {
        Self {
            info: VersionInfo::new(created_epoch, created_by),
            data,
        }
    }
}

/// All versions of a single entity, newest first.
///
/// Each node/edge has one of these tracking its version history. Use
/// [`visible_at()`](Self::visible_at) to get the version at a specific epoch,
/// or [`visible_to()`](Self::visible_to) for transaction-aware visibility.
#[derive(Debug, Clone)]
pub struct VersionChain<T> {
    /// Versions ordered newest-first.
    versions: VecDeque<Version<T>>,
}

impl<T> VersionChain<T> {
    /// Creates a new empty version chain.
    #[must_use]
    pub fn new() -> Self {
        Self {
            versions: VecDeque::new(),
        }
    }

    /// Creates a version chain with an initial version.
    #[must_use]
    pub fn with_initial(data: T, created_epoch: EpochId, created_by: TransactionId) -> Self {
        Self {
            versions: VecDeque::from(vec![Version::new(data, created_epoch, created_by)]),
        }
    }

    /// Adds a new version to the chain.
    ///
    /// The new version becomes the head of the chain.
    pub fn add_version(&mut self, data: T, created_epoch: EpochId, created_by: TransactionId) {
        let version = Version::new(data, created_epoch, created_by);
        self.versions.push_front(version);
    }

    /// Finds the version visible at the given epoch.
    ///
    /// Returns a reference to the visible version's data, or `None` if no version
    /// is visible at that epoch.
    #[inline]
    #[must_use]
    pub fn visible_at(&self, epoch: EpochId) -> Option<&T> {
        self.versions
            .iter()
            .find(|v| v.info.is_visible_at(epoch))
            .map(|v| &v.data)
    }

    /// Finds the version visible to a specific transaction.
    ///
    /// This considers both the transaction's epoch and its own uncommitted changes.
    #[inline]
    #[must_use]
    pub fn visible_to(&self, epoch: EpochId, tx: TransactionId) -> Option<&T> {
        self.versions
            .iter()
            .find(|v| v.info.is_visible_to(epoch, tx))
            .map(|v| &v.data)
    }

    /// Marks the current visible version as deleted by a specific transaction.
    ///
    /// Returns `true` if a version was marked, `false` if no visible version exists.
    pub fn mark_deleted(&mut self, delete_epoch: EpochId, deleted_by: TransactionId) -> bool {
        for version in &mut self.versions {
            if version.info.deleted_epoch.is_none() {
                version.info.mark_deleted(delete_epoch, deleted_by);
                return true;
            }
        }
        false
    }

    /// Unmarks deletion for versions deleted by the given transaction.
    ///
    /// Returns `true` if any version was undeleted. Used during rollback to
    /// restore versions that were deleted within the rolled-back transaction.
    pub fn unmark_deleted_by(&mut self, tx: TransactionId) -> bool {
        let mut any_undeleted = false;
        for version in &mut self.versions {
            if version.info.unmark_deleted_by(tx) {
                any_undeleted = true;
            }
        }
        any_undeleted
    }

    /// Checks if any version was modified by the given transaction.
    #[must_use]
    pub fn modified_by(&self, tx: TransactionId) -> bool {
        self.versions.iter().any(|v| v.info.created_by == tx)
    }

    /// Checks if any version was deleted by the given transaction.
    #[must_use]
    pub fn deleted_by(&self, tx: TransactionId) -> bool {
        self.versions.iter().any(|v| v.info.deleted_by == Some(tx))
    }

    /// Removes all versions created by the given transaction.
    ///
    /// Used for rollback to discard uncommitted changes.
    pub fn remove_versions_by(&mut self, tx: TransactionId) {
        self.versions.retain(|v| v.info.created_by != tx);
    }

    /// Finalizes PENDING epochs for versions created by the given transaction.
    ///
    /// Called at commit time to make uncommitted versions visible at the
    /// real commit epoch instead of `EpochId::PENDING`.
    pub fn finalize_epochs(&mut self, transaction_id: TransactionId, commit_epoch: EpochId) {
        for version in &mut self.versions {
            if version.info.created_by == transaction_id
                && version.info.created_epoch == EpochId::PENDING
            {
                version.info.created_epoch = commit_epoch;
            }
        }
    }

    /// Checks if there's a concurrent modification conflict.
    ///
    /// A conflict exists if another transaction modified this entity
    /// after our start epoch.
    #[must_use]
    pub fn has_conflict(&self, start_epoch: EpochId, our_tx: TransactionId) -> bool {
        self.versions.iter().any(|v| {
            v.info.created_by != our_tx && v.info.created_epoch.as_u64() > start_epoch.as_u64()
        })
    }

    /// Returns the number of versions in the chain.
    #[must_use]
    pub fn version_count(&self) -> usize {
        self.versions.len()
    }

    /// Returns true if the chain has no versions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.versions.is_empty()
    }

    /// Garbage collects old versions that are no longer visible to any transaction.
    ///
    /// Keeps versions that might still be visible to transactions at or after `min_epoch`.
    /// Returns how many versions it dropped.
    pub fn gc(&mut self, min_epoch: EpochId) -> usize {
        if self.versions.is_empty() {
            return 0;
        }

        let mut keep_count = 0;
        let mut found_old_visible = false;

        for (i, version) in self.versions.iter().enumerate() {
            if version.info.created_epoch.as_u64() >= min_epoch.as_u64() {
                keep_count = i + 1;
            } else if !found_old_visible {
                // Keep the first (most recent) old version
                found_old_visible = true;
                keep_count = i + 1;
            }
        }

        let before = self.versions.len();
        self.versions.truncate(keep_count);
        before - self.versions.len()
    }

    /// Returns an iterator over all versions with their metadata, newest first.
    ///
    /// Each item is `(&VersionInfo, &T)` giving both the visibility metadata
    /// and a reference to the version data.
    pub fn history(&self) -> impl Iterator<Item = (&VersionInfo, &T)> {
        self.versions.iter().map(|v| (&v.info, &v.data))
    }

    /// Returns a reference to the latest version's data regardless of visibility.
    #[must_use]
    pub fn latest(&self) -> Option<&T> {
        self.versions.front().map(|v| &v.data)
    }

    /// Returns a mutable reference to the latest version's data.
    #[must_use]
    pub fn latest_mut(&mut self) -> Option<&mut T> {
        self.versions.front_mut().map(|v| &mut v.data)
    }

    /// Returns estimated heap memory in bytes for this version chain.
    ///
    /// Counts the `Vec` capacity overhead. Does not include the
    /// size of `T` payloads (the caller accounts for those).
    #[must_use]
    pub fn heap_memory_bytes(&self) -> usize {
        self.versions.capacity() * std::mem::size_of::<Version<T>>()
    }
}

impl<T> Default for VersionChain<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> VersionChain<T> {
    /// Gets a mutable reference to the visible version's data for modification.
    ///
    /// If the version is not owned by this transaction, creates a new version
    /// with a copy of the data.
    pub fn get_mut(
        &mut self,
        epoch: EpochId,
        tx: TransactionId,
        modify_epoch: EpochId,
    ) -> Option<&mut T> {
        // Find the visible version
        let visible_idx = self
            .versions
            .iter()
            .position(|v| v.info.is_visible_to(epoch, tx))?;

        let visible = &self.versions[visible_idx];

        if visible.info.created_by == tx {
            // Already our version, modify in place
            Some(&mut self.versions[visible_idx].data)
        } else {
            // Create a new version with copied data
            let new_data = visible.data.clone();
            self.add_version(new_data, modify_epoch, tx);
            Some(&mut self.versions[0].data)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_visibility() {
        let v = VersionInfo::new(EpochId::new(5), TransactionId::new(1));

        // Not visible before creation
        assert!(!v.is_visible_at(EpochId::new(4)));

        // Visible at creation epoch and after
        assert!(v.is_visible_at(EpochId::new(5)));
        assert!(v.is_visible_at(EpochId::new(10)));
    }

    #[test]
    fn test_deleted_version_visibility() {
        let mut v = VersionInfo::new(EpochId::new(5), TransactionId::new(1));
        v.mark_deleted(EpochId::new(10), TransactionId::new(99));

        // Visible between creation and deletion
        assert!(v.is_visible_at(EpochId::new(5)));
        assert!(v.is_visible_at(EpochId::new(9)));

        // Not visible at or after deletion
        assert!(!v.is_visible_at(EpochId::new(10)));
        assert!(!v.is_visible_at(EpochId::new(15)));
    }

    #[test]
    fn test_version_visibility_to_transaction() {
        let v = VersionInfo::new(EpochId::new(5), TransactionId::new(1));

        // Creator can see it even if viewing at earlier epoch
        assert!(v.is_visible_to(EpochId::new(3), TransactionId::new(1)));

        // Other transactions can only see it at or after creation epoch
        assert!(!v.is_visible_to(EpochId::new(3), TransactionId::new(2)));
        assert!(v.is_visible_to(EpochId::new(5), TransactionId::new(2)));
    }

    #[test]
    fn test_version_chain_basic() {
        let mut chain = VersionChain::with_initial("v1", EpochId::new(1), TransactionId::new(1));

        // Should see v1 at epoch 1+
        assert_eq!(chain.visible_at(EpochId::new(1)), Some(&"v1"));
        assert_eq!(chain.visible_at(EpochId::new(0)), None);

        // Add v2
        chain.add_version("v2", EpochId::new(5), TransactionId::new(2));

        // Should see v1 at epoch < 5, v2 at epoch >= 5
        assert_eq!(chain.visible_at(EpochId::new(1)), Some(&"v1"));
        assert_eq!(chain.visible_at(EpochId::new(4)), Some(&"v1"));
        assert_eq!(chain.visible_at(EpochId::new(5)), Some(&"v2"));
        assert_eq!(chain.visible_at(EpochId::new(10)), Some(&"v2"));
    }

    #[test]
    fn test_version_chain_rollback() {
        let mut chain = VersionChain::with_initial("v1", EpochId::new(1), TransactionId::new(1));
        chain.add_version("v2", EpochId::new(5), TransactionId::new(2));
        chain.add_version("v3", EpochId::new(6), TransactionId::new(2));

        assert_eq!(chain.version_count(), 3);

        // Rollback tx 2's changes
        chain.remove_versions_by(TransactionId::new(2));

        assert_eq!(chain.version_count(), 1);
        assert_eq!(chain.visible_at(EpochId::new(10)), Some(&"v1"));
    }

    #[test]
    fn test_version_chain_deletion() {
        let mut chain = VersionChain::with_initial("v1", EpochId::new(1), TransactionId::new(1));

        // Mark as deleted at epoch 5 by transaction 99 (system)
        assert!(chain.mark_deleted(EpochId::new(5), TransactionId::new(99)));

        // Should see v1 before deletion, nothing after
        assert_eq!(chain.visible_at(EpochId::new(4)), Some(&"v1"));
        assert_eq!(chain.visible_at(EpochId::new(5)), None);
        assert_eq!(chain.visible_at(EpochId::new(10)), None);
    }
}
