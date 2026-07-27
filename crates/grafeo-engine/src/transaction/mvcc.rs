//! MVCC (Multi-Version Concurrency Control) re-exports.
//!
//! The core MVCC types are defined in `grafeo-common` and re-exported here
//! for convenience within the engine crate.

pub use grafeo_common::mvcc::{VersionChain, VersionInfo};

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::types::{EpochId, TransactionId};

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
    fn test_version_chain_transaction_visibility() {
        let mut chain = VersionChain::with_initial("v1", EpochId::new(1), TransactionId::new(1));
        chain.add_version("v2", EpochId::new(5), TransactionId::new(2));

        // Transaction 2 can see its own uncommitted changes
        assert_eq!(
            chain.visible_to(EpochId::new(3), TransactionId::new(2)),
            Some(&"v2")
        );

        // Transaction 3 at epoch 3 cannot see v2 (created at epoch 5)
        assert_eq!(
            chain.visible_to(EpochId::new(3), TransactionId::new(3)),
            Some(&"v1")
        );
    }

    #[test]
    fn test_version_chain_deletion() {
        let mut chain = VersionChain::with_initial("v1", EpochId::new(1), TransactionId::new(1));

        // Mark as deleted at epoch 5
        assert!(chain.mark_deleted(EpochId::new(5), TransactionId::new(99)));

        // Should see v1 before deletion, nothing after
        assert_eq!(chain.visible_at(EpochId::new(4)), Some(&"v1"));
        assert_eq!(chain.visible_at(EpochId::new(5)), None);
        assert_eq!(chain.visible_at(EpochId::new(10)), None);
    }

    #[test]
    fn test_version_chain_rollback() {
        let mut chain = VersionChain::with_initial("v1", EpochId::new(1), TransactionId::new(1));
        chain.add_version("v2", EpochId::PENDING, TransactionId::new(2));
        chain.add_version("v3", EpochId::PENDING, TransactionId::new(2));

        assert_eq!(chain.version_count(), 3);

        // Rollback tx 2's PENDING changes
        chain.remove_versions_by(TransactionId::new(2));

        assert_eq!(chain.version_count(), 1);
        assert_eq!(chain.visible_at(EpochId::new(10)), Some(&"v1"));
    }

    #[test]
    fn test_version_chain_conflict_detection() {
        let mut chain = VersionChain::with_initial("v1", EpochId::new(1), TransactionId::new(1));

        // Transaction starting at epoch 1 sees v1 as the baseline - no conflict
        assert!(!chain.has_conflict(EpochId::new(1), TransactionId::new(2)));

        // Transaction starting at epoch 0 sees v1 (created at epoch 1) as concurrent - conflict
        assert!(chain.has_conflict(EpochId::new(0), TransactionId::new(2)));

        // Add a version from TransactionId(2)
        chain.add_version("v2", EpochId::new(5), TransactionId::new(2));

        // Transaction 3 starting at epoch 0 would conflict (v1 at epoch 1, v2 at epoch 5)
        assert!(chain.has_conflict(EpochId::new(0), TransactionId::new(3)));

        // Transaction 3 starting at epoch 5 would not conflict (v2 is at epoch 5, not after)
        assert!(!chain.has_conflict(EpochId::new(5), TransactionId::new(3)));

        // Transaction 2's own writes don't count as conflicts
        assert!(!chain.has_conflict(EpochId::new(4), TransactionId::new(2)));
    }

    #[test]
    fn test_version_chain_gc() {
        let mut chain = VersionChain::new();
        chain.add_version("v1", EpochId::new(1), TransactionId::new(1));
        chain.add_version("v2", EpochId::new(3), TransactionId::new(2));
        chain.add_version("v3", EpochId::new(5), TransactionId::new(3));
        chain.add_version("v4", EpochId::new(7), TransactionId::new(4));

        assert_eq!(chain.version_count(), 4);

        // GC with min_epoch = 6 should keep v3 and v4
        chain.gc(EpochId::new(6));

        // Should still see v4 at current epochs
        assert_eq!(chain.visible_at(EpochId::new(7)), Some(&"v4"));

        // Version count reduced
        assert!(chain.version_count() <= 4);
    }

    #[test]
    fn test_version_chain_get_mut() {
        let mut chain =
            VersionChain::with_initial(String::from("v1"), EpochId::new(1), TransactionId::new(1));

        // Transaction 1 modifying its own version
        {
            let data = chain
                .get_mut(EpochId::new(1), TransactionId::new(1), EpochId::new(2))
                .unwrap();
            data.push_str("_modified");
        }
        assert_eq!(chain.version_count(), 1); // Modified in place
        assert_eq!(chain.visible_at(EpochId::new(1)).unwrap(), "v1_modified");

        // Transaction 2 modifying creates a new version
        {
            let data = chain
                .get_mut(EpochId::new(3), TransactionId::new(2), EpochId::new(3))
                .unwrap();
            data.push_str("_by_tx2");
        }
        assert_eq!(chain.version_count(), 2); // New version created
        assert_eq!(
            chain.visible_at(EpochId::new(3)).unwrap(),
            "v1_modified_by_tx2"
        );
        // Old version still visible at earlier epoch
        assert_eq!(chain.visible_at(EpochId::new(2)).unwrap(), "v1_modified");
    }
}
