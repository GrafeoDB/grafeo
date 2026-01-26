//! MVCC (Multi-Version Concurrency Control) implementation.
//!
//! This module provides the version chain and visibility logic
//! for concurrent read/write access.

use graphos_common::types::{EpochId, TxId};

/// Visibility information for a version.
#[derive(Debug, Clone, Copy)]
pub struct VersionInfo {
    /// The epoch this version was created in.
    pub created_epoch: EpochId,
    /// The epoch this version was deleted in (if any).
    pub deleted_epoch: Option<EpochId>,
    /// The transaction that created this version.
    pub created_by: TxId,
}

impl VersionInfo {
    /// Creates a new version info.
    #[must_use]
    pub fn new(created_epoch: EpochId, created_by: TxId) -> Self {
        Self {
            created_epoch,
            deleted_epoch: None,
            created_by,
        }
    }

    /// Marks this version as deleted.
    pub fn mark_deleted(&mut self, epoch: EpochId) {
        self.deleted_epoch = Some(epoch);
    }

    /// Checks if this version is visible at the given epoch.
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_visibility() {
        let v = VersionInfo::new(EpochId::new(5), TxId::new(1));

        // Not visible before creation
        assert!(!v.is_visible_at(EpochId::new(4)));

        // Visible at creation epoch and after
        assert!(v.is_visible_at(EpochId::new(5)));
        assert!(v.is_visible_at(EpochId::new(10)));
    }

    #[test]
    fn test_deleted_version_visibility() {
        let mut v = VersionInfo::new(EpochId::new(5), TxId::new(1));
        v.mark_deleted(EpochId::new(10));

        // Visible between creation and deletion
        assert!(v.is_visible_at(EpochId::new(5)));
        assert!(v.is_visible_at(EpochId::new(9)));

        // Not visible at or after deletion
        assert!(!v.is_visible_at(EpochId::new(10)));
        assert!(!v.is_visible_at(EpochId::new(15)));
    }
}
