//! Storage tiers: whether a data structure's data is in RAM or on disk.
//!
//! A [`MemoryConsumer`](super::MemoryConsumer) reports its [`StorageTier`].
//! The [`BufferManager`](super::BufferManager) decides *when* data moves
//! between tiers (based on memory pressure); a section's `swap_to_mmap` and
//! `reload_to_ram` ([`Section`](crate::storage::section::Section)) move it.
//!
//! # Storage states
//!
//! ```text
//!       ┌──────────────┐    spill()     ┌──────────────┐
//!       │   InMemory   │ ────────────> │    OnDisk    │
//!       │ (RAM, fast)  │               │ (mmap, warm) │
//!       └──────────────┘               └──────┬───────┘
//!              ▲                               │
//!              └───────── reload() ────────────┘
//! ```
//!
//! All states expose the same read interface. Callers never need to know
//! which tier is active.

/// The current storage tier of a data structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum StorageTier {
    /// Fully in RAM. Fastest access for both reads and writes.
    InMemory,
    /// On disk, accessed via mmap. The OS page cache provides warm reads.
    /// Mutations go through a WAL overlay.
    OnDisk,
    /// Not yet initialized (structure exists but has no data).
    Uninitialized,
}

impl StorageTier {
    /// Returns `true` if data is fully in RAM.
    #[must_use]
    pub fn is_in_memory(self) -> bool {
        self == Self::InMemory
    }

    /// Returns `true` if data is served from disk (mmap).
    #[must_use]
    pub fn is_on_disk(self) -> bool {
        self == Self::OnDisk
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_tier_predicates() {
        assert!(StorageTier::InMemory.is_in_memory());
        assert!(!StorageTier::InMemory.is_on_disk());

        assert!(StorageTier::OnDisk.is_on_disk());
        assert!(!StorageTier::OnDisk.is_in_memory());

        assert!(!StorageTier::Uninitialized.is_in_memory());
        assert!(!StorageTier::Uninitialized.is_on_disk());
    }

    #[test]
    fn test_storage_tier_equality() {
        assert_eq!(StorageTier::InMemory, StorageTier::InMemory);
        assert_ne!(StorageTier::InMemory, StorageTier::OnDisk);
        assert_ne!(StorageTier::OnDisk, StorageTier::Uninitialized);
    }
}
