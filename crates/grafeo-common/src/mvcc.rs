//! MVCC (Multi-Version Concurrency Control) primitives.
//!
//! This is how Grafeo handles concurrent reads and writes without blocking.
//! Each entity has a [`VersionChain`] that tracks all versions. Readers see
//! consistent snapshots, writers create new versions, and old versions get
//! garbage collected when no one needs them anymore.

use std::collections::VecDeque;

#[cfg(feature = "tiered-storage")]
use smallvec::SmallVec;

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

    /// Unmarks deletion if deleted by the given transaction while still pending.
    ///
    /// Returns `true` if undeleted. Committed deletions (non-`PENDING`
    /// `deleted_epoch`) are left untouched even when `deleted_by` coincides
    /// with a later recycled transaction id.
    pub fn unmark_deleted_by(&mut self, transaction_id: TransactionId) -> bool {
        if self.deleted_by == Some(transaction_id) && self.deleted_epoch == Some(EpochId::PENDING) {
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

    /// Returns `true` when every version is an uncommitted create by `tx`.
    ///
    /// Used by rollback to detect entities created inside the active
    /// transaction. Matching requires `created_epoch == PENDING` so a recycled
    /// transaction id cannot treat already-committed versions as disposable.
    #[must_use]
    pub fn solely_created_by(&self, tx: TransactionId) -> bool {
        !self.versions.is_empty()
            && self
                .versions
                .iter()
                .all(|v| v.info.created_by == tx && v.info.created_epoch == EpochId::PENDING)
    }

    /// Removes uncommitted versions created by the given transaction.
    ///
    /// Only `PENDING` creates are discarded. Committed versions that happen to
    /// share `created_by` after transaction-manager recreation are preserved.
    pub fn remove_versions_by(&mut self, tx: TransactionId) {
        self.versions
            .retain(|v| !(v.info.created_by == tx && v.info.created_epoch == EpochId::PENDING));
    }

    /// Finalizes PENDING epochs for versions created or deleted by the given transaction.
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
            if version.info.deleted_by == Some(transaction_id)
                && version.info.deleted_epoch == Some(EpochId::PENDING)
            {
                version.info.deleted_epoch = Some(commit_epoch);
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
    pub fn gc(&mut self, min_epoch: EpochId) {
        if self.versions.is_empty() {
            return;
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

        self.versions.truncate(keep_count);
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

// ============================================================================
// Tiered Storage Types (Phase 13)
// ============================================================================
//
// These types support the tiered hot/cold storage architecture where version
// metadata is stored separately from version data. Data lives in arenas (hot)
// or compressed epoch blocks (cold), while VersionIndex holds lightweight refs.

/// Compact representation of an optional epoch ID.
///
/// Uses `u32::MAX` as sentinel for `None`, allowing epochs up to ~4 billion.
/// This saves 4 bytes compared to `Option<EpochId>` due to niche optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(transparent)]
#[cfg(feature = "tiered-storage")]
pub struct OptionalEpochId(u32);

#[cfg(feature = "tiered-storage")]
impl OptionalEpochId {
    /// Represents no epoch (deleted_epoch = None).
    pub const NONE: Self = Self(u32::MAX);

    /// Compact encoding of [`EpochId::PENDING`] for uncommitted deletions.
    ///
    /// Real epochs use `0..u32::MAX-1`; `u32::MAX` is [`NONE`].
    const PENDING_CODE: u32 = u32::MAX - 1;

    /// Creates an `OptionalEpochId` from an epoch.
    ///
    /// # Panics
    /// Panics if a non-PENDING epoch exceeds `u32::MAX - 2` (4,294,967,293).
    #[must_use]
    pub fn some(epoch: EpochId) -> Self {
        if epoch == EpochId::PENDING {
            return Self(Self::PENDING_CODE);
        }
        assert!(
            epoch.as_u64() < u64::from(Self::PENDING_CODE),
            "epoch {} exceeds OptionalEpochId capacity (max {})",
            epoch.as_u64(),
            u64::from(Self::PENDING_CODE) - 1
        );
        // reason: the assert above guarantees epoch < PENDING_CODE
        #[allow(clippy::cast_possible_truncation)]
        Self(epoch.as_u64() as u32)
    }

    /// Returns the contained epoch, or `None` if this is `NONE`.
    #[inline]
    #[must_use]
    pub fn get(self) -> Option<EpochId> {
        if self.0 == u32::MAX {
            None
        } else if self.0 == Self::PENDING_CODE {
            Some(EpochId::PENDING)
        } else {
            Some(EpochId::new(u64::from(self.0)))
        }
    }

    /// Returns `true` if this contains an epoch.
    #[must_use]
    pub fn is_some(self) -> bool {
        self.0 != u32::MAX
    }

    /// Returns `true` if this is `NONE`.
    #[inline]
    #[must_use]
    pub fn is_none(self) -> bool {
        self.0 == u32::MAX
    }

    /// Returns `true` when this encodes an uncommitted (`PENDING`) deletion.
    #[inline]
    #[must_use]
    pub fn is_pending(self) -> bool {
        self.0 == Self::PENDING_CODE
    }
}

/// Reference to a hot (arena-allocated) version.
///
/// Hot versions are stored in the epoch's arena and can be accessed directly.
/// This struct only holds metadata; the actual data lives in the arena.
///
/// # Memory Layout
/// - `epoch`: 8 bytes
/// - `arena_epoch`: 8 bytes
/// - `arena_offset`: 4 bytes
/// - `created_by`: 8 bytes
/// - `deleted_epoch`: 4 bytes
/// - Total: 32 bytes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(feature = "tiered-storage")]
pub struct HotVersionRef {
    /// Epoch used for MVCC visibility checks.
    ///
    /// Set to `EpochId::PENDING` for uncommitted transactional versions,
    /// then finalized to the real commit epoch on commit.
    pub epoch: EpochId,
    /// Epoch whose arena holds the actual data.
    ///
    /// Always the real epoch (never PENDING), used for arena lookups.
    pub arena_epoch: EpochId,
    /// Offset within the epoch's arena where the data is stored.
    pub arena_offset: u32,
    /// Transaction that created this version.
    pub created_by: TransactionId,
    /// Epoch when this version was deleted (NONE if still alive).
    pub deleted_epoch: OptionalEpochId,
    /// Transaction that deleted this version (for rollback support).
    pub deleted_by: Option<TransactionId>,
}

#[cfg(feature = "tiered-storage")]
impl HotVersionRef {
    /// Creates a new hot version reference.
    ///
    /// `epoch` is used for MVCC visibility (may be `PENDING` for uncommitted versions).
    /// `arena_epoch` is the real epoch whose arena holds the data.
    #[must_use]
    pub fn new(
        epoch: EpochId,
        arena_epoch: EpochId,
        arena_offset: u32,
        created_by: TransactionId,
    ) -> Self {
        Self {
            epoch,
            arena_epoch,
            arena_offset,
            created_by,
            deleted_epoch: OptionalEpochId::NONE,
            deleted_by: None,
        }
    }

    /// Marks this version as deleted by a specific transaction.
    pub fn mark_deleted(&mut self, epoch: EpochId, deleted_by: TransactionId) {
        self.deleted_epoch = OptionalEpochId::some(epoch);
        self.deleted_by = Some(deleted_by);
    }

    /// Unmarks a pending deletion by the given transaction. Returns `true` if undeleted.
    pub fn unmark_deleted_by(&mut self, transaction_id: TransactionId) -> bool {
        if self.deleted_by == Some(transaction_id) && self.deleted_epoch.is_pending() {
            self.deleted_epoch = OptionalEpochId::NONE;
            self.deleted_by = None;
            true
        } else {
            false
        }
    }

    /// Checks if this version is visible at the given epoch.
    #[inline]
    #[must_use]
    pub fn is_visible_at(&self, viewing_epoch: EpochId) -> bool {
        // Must be created at or before the viewing epoch
        if !self.epoch.is_visible_at(viewing_epoch) {
            return false;
        }
        // Must not be deleted at or before the viewing epoch
        match self.deleted_epoch.get() {
            Some(deleted) => deleted.as_u64() > viewing_epoch.as_u64(),
            None => true,
        }
    }

    /// Checks if this version is visible to a specific transaction.
    #[inline]
    #[must_use]
    pub fn is_visible_to(&self, viewing_epoch: EpochId, viewing_tx: TransactionId) -> bool {
        // If deleted by the viewing transaction, not visible
        if self.deleted_by == Some(viewing_tx) {
            return false;
        }
        // Own modifications are always visible (if not deleted)
        if self.created_by == viewing_tx {
            return self.deleted_epoch.is_none();
        }
        // Otherwise use epoch-based visibility
        self.is_visible_at(viewing_epoch)
    }
}

/// Reference to a cold (compressed) version.
///
/// Cold versions are stored in compressed epoch blocks. This is a placeholder
/// for Phase 14 - the actual compression logic will be implemented there.
///
/// # Memory Layout
/// - `epoch`: 8 bytes
/// - `block_offset`: 4 bytes
/// - `length`: 2 bytes
/// - `created_by`: 8 bytes
/// - `deleted_epoch`: 4 bytes
/// - Total: 26 bytes (+ 6 padding = 32 bytes aligned)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(feature = "tiered-storage")]
pub struct ColdVersionRef {
    /// Epoch when this version was created.
    pub epoch: EpochId,
    /// Offset within the compressed epoch block.
    pub block_offset: u32,
    /// Compressed length in bytes.
    pub length: u16,
    /// Transaction that created this version.
    pub created_by: TransactionId,
    /// Epoch when this version was deleted.
    pub deleted_epoch: OptionalEpochId,
    /// Transaction that deleted this version (for rollback support).
    pub deleted_by: Option<TransactionId>,
}

#[cfg(feature = "tiered-storage")]
impl ColdVersionRef {
    /// Checks if this version is visible at the given epoch.
    #[inline]
    #[must_use]
    pub fn is_visible_at(&self, viewing_epoch: EpochId) -> bool {
        if !self.epoch.is_visible_at(viewing_epoch) {
            return false;
        }
        match self.deleted_epoch.get() {
            Some(deleted) => deleted.as_u64() > viewing_epoch.as_u64(),
            None => true,
        }
    }

    /// Checks if this version is visible to a specific transaction.
    #[inline]
    #[must_use]
    pub fn is_visible_to(&self, viewing_epoch: EpochId, viewing_tx: TransactionId) -> bool {
        // If deleted by the viewing transaction, not visible
        if self.deleted_by == Some(viewing_tx) {
            return false;
        }
        if self.created_by == viewing_tx {
            return self.deleted_epoch.is_none();
        }
        self.is_visible_at(viewing_epoch)
    }
}

/// Unified reference to either a hot or cold version.
#[derive(Debug, Clone, Copy)]
#[cfg(feature = "tiered-storage")]
#[non_exhaustive]
pub enum VersionRef {
    /// Version data is in arena (hot tier).
    Hot(HotVersionRef),
    /// Version data is in compressed storage (cold tier).
    Cold(ColdVersionRef),
}

#[cfg(feature = "tiered-storage")]
impl VersionRef {
    /// Returns the epoch when this version was created.
    #[must_use]
    pub fn epoch(&self) -> EpochId {
        match self {
            Self::Hot(h) => h.epoch,
            Self::Cold(c) => c.epoch,
        }
    }

    /// Returns the transaction that created this version.
    #[must_use]
    pub fn created_by(&self) -> TransactionId {
        match self {
            Self::Hot(h) => h.created_by,
            Self::Cold(c) => c.created_by,
        }
    }

    /// Returns `true` if this is a hot version.
    #[must_use]
    pub fn is_hot(&self) -> bool {
        matches!(self, Self::Hot(_))
    }

    /// Returns `true` if this is a cold version.
    #[must_use]
    pub fn is_cold(&self) -> bool {
        matches!(self, Self::Cold(_))
    }

    /// Returns the epoch when this version was deleted, if any.
    #[must_use]
    pub fn deleted_epoch(&self) -> Option<EpochId> {
        match self {
            Self::Hot(h) => h.deleted_epoch.get(),
            Self::Cold(c) => c.deleted_epoch.get(),
        }
    }
}

/// Tiered version index - replaces `VersionChain<T>` for hot/cold storage.
///
/// Instead of storing data inline, `VersionIndex` holds lightweight references
/// to data stored in arenas (hot) or compressed blocks (cold). This enables:
///
/// - **No heap allocation** for typical 1-2 version case (SmallVec inline)
/// - **Separation of metadata and data** for compression
/// - **Fast visibility checks** via cached `latest_epoch`
/// - **O(1) epoch drops** instead of per-version deallocation
///
/// # Memory Layout
/// - `hot`: SmallVec<[HotVersionRef; 2]> ≈ 56 bytes inline
/// - `cold`: SmallVec<[ColdVersionRef; 4]> ≈ 136 bytes inline
/// - `latest_epoch`: 8 bytes
/// - Total: ~200 bytes, no heap for typical case
#[derive(Debug, Clone)]
#[cfg(feature = "tiered-storage")]
pub struct VersionIndex {
    /// Hot versions in arena storage (most recent first).
    hot: SmallVec<[HotVersionRef; 2]>,
    /// Cold versions in compressed storage (most recent first).
    cold: SmallVec<[ColdVersionRef; 4]>,
    /// Cached epoch of the latest version for fast staleness checks.
    latest_epoch: EpochId,
}

#[cfg(feature = "tiered-storage")]
impl VersionIndex {
    /// Creates a new empty version index.
    #[must_use]
    pub fn new() -> Self {
        Self {
            hot: SmallVec::new(),
            cold: SmallVec::new(),
            latest_epoch: EpochId::INITIAL,
        }
    }

    /// Creates a version index with an initial hot version.
    #[must_use]
    pub fn with_initial(hot_ref: HotVersionRef) -> Self {
        let mut index = Self::new();
        index.add_hot(hot_ref);
        index
    }

    /// Adds a new hot version (becomes the latest).
    pub fn add_hot(&mut self, hot_ref: HotVersionRef) {
        // Insert at front (most recent first)
        self.hot.insert(0, hot_ref);
        self.latest_epoch = hot_ref.epoch;
    }

    /// Returns the latest epoch for quick staleness checks.
    #[must_use]
    pub fn latest_epoch(&self) -> EpochId {
        self.latest_epoch
    }

    /// Returns `true` if this entity has no versions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.hot.is_empty() && self.cold.is_empty()
    }

    /// Returns the total version count (hot + cold).
    #[must_use]
    pub fn version_count(&self) -> usize {
        self.hot.len() + self.cold.len()
    }

    /// Returns the number of hot versions.
    #[must_use]
    pub fn hot_count(&self) -> usize {
        self.hot.len()
    }

    /// Returns the number of cold versions.
    #[must_use]
    pub fn cold_count(&self) -> usize {
        self.cold.len()
    }

    /// Finds the version visible at the given epoch.
    #[inline]
    #[must_use]
    pub fn visible_at(&self, epoch: EpochId) -> Option<VersionRef> {
        // Check hot versions first (most recent first, likely case)
        for v in &self.hot {
            if v.is_visible_at(epoch) {
                return Some(VersionRef::Hot(*v));
            }
        }
        // Fall back to cold versions
        for v in &self.cold {
            if v.is_visible_at(epoch) {
                return Some(VersionRef::Cold(*v));
            }
        }
        None
    }

    /// Finds the version visible to a specific transaction.
    #[inline]
    #[must_use]
    pub fn visible_to(&self, epoch: EpochId, tx: TransactionId) -> Option<VersionRef> {
        // Check hot versions first
        for v in &self.hot {
            if v.is_visible_to(epoch, tx) {
                return Some(VersionRef::Hot(*v));
            }
        }
        // Fall back to cold versions
        for v in &self.cold {
            if v.is_visible_to(epoch, tx) {
                return Some(VersionRef::Cold(*v));
            }
        }
        None
    }

    /// Marks the currently visible version as deleted by a specific transaction.
    ///
    /// Returns `true` if a version was marked, `false` if no visible version exists.
    pub fn mark_deleted(&mut self, delete_epoch: EpochId, deleted_by: TransactionId) -> bool {
        // Find the first non-deleted hot version and mark it
        for v in &mut self.hot {
            if v.deleted_epoch.is_none() {
                v.mark_deleted(delete_epoch, deleted_by);
                return true;
            }
        }
        // Check cold versions (rare case)
        for v in &mut self.cold {
            if v.deleted_epoch.is_none() {
                v.deleted_epoch = OptionalEpochId::some(delete_epoch);
                v.deleted_by = Some(deleted_by);
                return true;
            }
        }
        false
    }

    /// Unmarks pending deletion for versions deleted by the given transaction.
    ///
    /// Returns `true` if any version was undeleted. Used during rollback.
    pub fn unmark_deleted_by(&mut self, tx: TransactionId) -> bool {
        let mut any_undeleted = false;
        for v in &mut self.hot {
            if v.unmark_deleted_by(tx) {
                any_undeleted = true;
            }
        }
        for v in &mut self.cold {
            if v.deleted_by == Some(tx) && v.deleted_epoch.is_pending() {
                v.deleted_epoch = OptionalEpochId::NONE;
                v.deleted_by = None;
                any_undeleted = true;
            }
        }
        any_undeleted
    }

    /// Checks if any version was created by the given transaction.
    #[must_use]
    pub fn modified_by(&self, tx: TransactionId) -> bool {
        self.hot.iter().any(|v| v.created_by == tx) || self.cold.iter().any(|v| v.created_by == tx)
    }

    /// Checks if any version was deleted by the given transaction.
    #[must_use]
    pub fn deleted_by(&self, tx: TransactionId) -> bool {
        self.hot.iter().any(|v| v.deleted_by == Some(tx))
            || self.cold.iter().any(|v| v.deleted_by == Some(tx))
    }

    /// Returns `true` when every version is an uncommitted create by `tx`.
    ///
    /// See [`VersionChain::solely_created_by`] for the rollback-cleanup contract.
    #[must_use]
    pub fn solely_created_by(&self, tx: TransactionId) -> bool {
        !self.is_empty()
            && self
                .hot
                .iter()
                .all(|v| v.created_by == tx && v.epoch == EpochId::PENDING)
            && self
                .cold
                .iter()
                .all(|v| v.created_by == tx && v.epoch == EpochId::PENDING)
    }

    /// Removes uncommitted (`PENDING`) versions created by the given transaction.
    pub fn remove_versions_by(&mut self, tx: TransactionId) {
        self.hot
            .retain(|v| !(v.created_by == tx && v.epoch == EpochId::PENDING));
        self.cold
            .retain(|v| !(v.created_by == tx && v.epoch == EpochId::PENDING));
        self.recalculate_latest_epoch();
    }

    /// Finalizes PENDING create/delete epochs for versions owned by the transaction.
    ///
    /// Called at commit time to make uncommitted versions visible at the
    /// real commit epoch instead of `EpochId::PENDING`.
    pub fn finalize_epochs(&mut self, transaction_id: TransactionId, commit_epoch: EpochId) {
        for v in &mut self.hot {
            if v.created_by == transaction_id && v.epoch == EpochId::PENDING {
                v.epoch = commit_epoch;
            }
            if v.deleted_by == Some(transaction_id) && v.deleted_epoch.is_pending() {
                v.deleted_epoch = OptionalEpochId::some(commit_epoch);
            }
        }
        for v in &mut self.cold {
            if v.created_by == transaction_id && v.epoch == EpochId::PENDING {
                v.epoch = commit_epoch;
            }
            if v.deleted_by == Some(transaction_id) && v.deleted_epoch.is_pending() {
                v.deleted_epoch = OptionalEpochId::some(commit_epoch);
            }
        }
        self.recalculate_latest_epoch();
    }

    /// Checks for write conflict with concurrent transaction.
    ///
    /// A conflict exists if another transaction modified this entity
    /// after our start epoch.
    #[must_use]
    pub fn has_conflict(&self, start_epoch: EpochId, our_tx: TransactionId) -> bool {
        self.hot
            .iter()
            .any(|v| v.created_by != our_tx && v.epoch.as_u64() > start_epoch.as_u64())
            || self
                .cold
                .iter()
                .any(|v| v.created_by != our_tx && v.epoch.as_u64() > start_epoch.as_u64())
    }

    /// Garbage collects old versions not needed by any active transaction.
    ///
    /// Keeps versions that might still be visible to transactions at or after `min_epoch`.
    pub fn gc(&mut self, min_epoch: EpochId) {
        if self.is_empty() {
            return;
        }

        // Keep versions that:
        // 1. Were created at or after min_epoch
        // 2. The first (most recent) version created before min_epoch
        let mut found_old_visible = false;

        self.hot.retain(|v| {
            if v.epoch.as_u64() >= min_epoch.as_u64() {
                true
            } else if !found_old_visible {
                found_old_visible = true;
                true
            } else {
                false
            }
        });

        // Same for cold, but only if we haven't found an old visible in hot
        if !found_old_visible {
            self.cold.retain(|v| {
                if v.epoch.as_u64() >= min_epoch.as_u64() {
                    true
                } else if !found_old_visible {
                    found_old_visible = true;
                    true
                } else {
                    false
                }
            });
        } else {
            // All cold versions are older, only keep those >= min_epoch
            self.cold.retain(|v| v.epoch.as_u64() >= min_epoch.as_u64());
        }
    }

    /// Returns epoch IDs of all versions, newest first.
    ///
    /// Combines hot and cold versions, sorted by epoch descending.
    #[must_use]
    pub fn version_epochs(&self) -> Vec<EpochId> {
        let mut epochs: Vec<EpochId> = self
            .hot
            .iter()
            .map(|v| v.epoch)
            .chain(self.cold.iter().map(|v| v.epoch))
            .collect();
        epochs.sort_by_key(|e| std::cmp::Reverse(e.as_u64()));
        epochs
    }

    /// Returns all version references with their metadata, newest first.
    ///
    /// Each item is `(EpochId, Option<EpochId>, VersionRef)` giving
    /// the created epoch, deleted epoch, and a reference to read the data.
    #[must_use]
    pub fn version_history(&self) -> Vec<(EpochId, Option<EpochId>, VersionRef)> {
        let mut versions: Vec<(EpochId, Option<EpochId>, VersionRef)> = self
            .hot
            .iter()
            .map(|v| (v.epoch, v.deleted_epoch.get(), VersionRef::Hot(*v)))
            .chain(
                self.cold
                    .iter()
                    .map(|v| (v.epoch, v.deleted_epoch.get(), VersionRef::Cold(*v))),
            )
            .collect();
        versions.sort_by_key(|v| std::cmp::Reverse(v.0.as_u64()));
        versions
    }

    /// Returns a reference to the latest version regardless of visibility.
    #[must_use]
    pub fn latest(&self) -> Option<VersionRef> {
        self.hot
            .first()
            .map(|v| VersionRef::Hot(*v))
            .or_else(|| self.cold.first().map(|v| VersionRef::Cold(*v)))
    }

    /// Freezes hot versions for a given epoch into cold storage.
    ///
    /// This is called when an epoch is no longer needed by any active transaction.
    /// The actual compression happens in Phase 14; for now this just moves refs.
    pub fn freeze_epoch(
        &mut self,
        epoch: EpochId,
        cold_refs: impl Iterator<Item = ColdVersionRef>,
    ) {
        // Remove hot refs for this epoch
        self.hot.retain(|v| v.epoch != epoch);

        // Add cold refs
        self.cold.extend(cold_refs);

        // Keep cold sorted by epoch (descending = most recent first)
        self.cold
            .sort_by_key(|v| std::cmp::Reverse(v.epoch.as_u64()));

        self.recalculate_latest_epoch();
    }

    /// Returns hot version refs for a specific epoch (for freeze operation).
    pub fn hot_refs_for_epoch(&self, epoch: EpochId) -> impl Iterator<Item = &HotVersionRef> {
        self.hot.iter().filter(move |v| v.epoch == epoch)
    }

    /// Returns `true` if the hot SmallVec has spilled to the heap.
    #[must_use]
    pub fn hot_spilled(&self) -> bool {
        self.hot.spilled()
    }

    /// Returns `true` if the cold SmallVec has spilled to the heap.
    #[must_use]
    pub fn cold_spilled(&self) -> bool {
        self.cold.spilled()
    }

    fn recalculate_latest_epoch(&mut self) {
        self.latest_epoch = self
            .hot
            .first()
            .map(|v| v.epoch)
            .or_else(|| self.cold.first().map(|v| v.epoch))
            .unwrap_or(EpochId::INITIAL);
    }
}

#[cfg(feature = "tiered-storage")]
impl Default for VersionIndex {
    fn default() -> Self {
        Self::new()
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
        chain.add_version("v2", EpochId::PENDING, TransactionId::new(2));
        chain.add_version("v3", EpochId::PENDING, TransactionId::new(2));

        assert_eq!(chain.version_count(), 3);

        // Rollback tx 2's PENDING changes
        chain.remove_versions_by(TransactionId::new(2));

        assert_eq!(chain.version_count(), 1);
        assert_eq!(chain.visible_at(EpochId::new(10)), Some(&"v1"));
    }

    #[test]
    fn test_version_chain_rollback_ignores_committed_same_tx_id() {
        let tx = TransactionId::new(7);
        let mut chain = VersionChain::with_initial("committed", EpochId::new(3), tx);
        // Simulate transaction-manager recreation reusing the same id while a
        // prior committed version still carries that created_by.
        assert!(!chain.solely_created_by(tx));
        chain.remove_versions_by(tx);
        assert_eq!(chain.version_count(), 1);
        assert_eq!(chain.visible_at(EpochId::new(3)), Some(&"committed"));

        chain.add_version("pending", EpochId::PENDING, tx);
        assert!(chain.solely_created_by(tx) == false); // mixed committed + pending
        // Only the PENDING create is discarded.
        chain.remove_versions_by(tx);
        assert_eq!(chain.version_count(), 1);
        assert_eq!(chain.visible_at(EpochId::new(3)), Some(&"committed"));
    }

    #[test]
    fn test_version_chain_unmark_requires_pending_delete() {
        let tx = TransactionId::new(9);
        let mut chain = VersionChain::with_initial("v1", EpochId::new(1), TransactionId::new(1));
        chain.mark_deleted(EpochId::new(5), tx);
        // Committed-style delete epoch must not roll back via tx-id coincidence.
        assert!(!chain.unmark_deleted_by(tx));
        assert!(chain.deleted_by(tx));

        let mut pending = VersionChain::with_initial("v1", EpochId::new(1), TransactionId::new(1));
        pending.mark_deleted(EpochId::PENDING, tx);
        assert!(pending.unmark_deleted_by(tx));
        assert!(!pending.deleted_by(tx));
        assert_eq!(pending.visible_at(EpochId::new(10)), Some(&"v1"));
    }

    #[test]
    fn test_version_chain_solely_created_requires_pending() {
        let tx = TransactionId::new(3);
        let pending = VersionChain::with_initial("p", EpochId::PENDING, tx);
        assert!(pending.solely_created_by(tx));

        let committed = VersionChain::with_initial("c", EpochId::new(2), tx);
        assert!(!committed.solely_created_by(tx));
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

// ============================================================================
// Tiered Storage Tests
// ============================================================================

#[cfg(all(test, feature = "tiered-storage"))]
mod tiered_storage_tests {
    use super::*;

    #[test]
    fn test_optional_epoch_id() {
        // Test NONE
        let none = OptionalEpochId::NONE;
        assert!(none.is_none());
        assert!(!none.is_some());
        assert_eq!(none.get(), None);

        // Test Some
        let some = OptionalEpochId::some(EpochId::new(42));
        assert!(some.is_some());
        assert!(!some.is_none());
        assert_eq!(some.get(), Some(EpochId::new(42)));

        // Test zero epoch
        let zero = OptionalEpochId::some(EpochId::new(0));
        assert!(zero.is_some());
        assert_eq!(zero.get(), Some(EpochId::new(0)));
    }

    #[test]
    fn test_hot_version_ref_visibility() {
        let hot = HotVersionRef::new(EpochId::new(5), EpochId::new(5), 100, TransactionId::new(1));

        // Not visible before creation
        assert!(!hot.is_visible_at(EpochId::new(4)));

        // Visible at creation and after
        assert!(hot.is_visible_at(EpochId::new(5)));
        assert!(hot.is_visible_at(EpochId::new(10)));
    }

    #[test]
    fn test_hot_version_ref_deleted_visibility() {
        let mut hot =
            HotVersionRef::new(EpochId::new(5), EpochId::new(5), 100, TransactionId::new(1));
        hot.deleted_epoch = OptionalEpochId::some(EpochId::new(10));

        // Visible between creation and deletion
        assert!(hot.is_visible_at(EpochId::new(5)));
        assert!(hot.is_visible_at(EpochId::new(9)));

        // Not visible at or after deletion
        assert!(!hot.is_visible_at(EpochId::new(10)));
        assert!(!hot.is_visible_at(EpochId::new(15)));
    }

    #[test]
    fn test_hot_version_ref_transaction_visibility() {
        let hot = HotVersionRef::new(EpochId::new(5), EpochId::new(5), 100, TransactionId::new(1));

        // Creator can see it even at earlier epoch
        assert!(hot.is_visible_to(EpochId::new(3), TransactionId::new(1)));

        // Other transactions can only see it at or after creation
        assert!(!hot.is_visible_to(EpochId::new(3), TransactionId::new(2)));
        assert!(hot.is_visible_to(EpochId::new(5), TransactionId::new(2)));
    }

    #[test]
    fn test_version_index_basic() {
        let hot = HotVersionRef::new(EpochId::new(1), EpochId::new(1), 0, TransactionId::new(1));
        let mut index = VersionIndex::with_initial(hot);

        // Should see version at epoch 1+
        assert!(index.visible_at(EpochId::new(1)).is_some());
        assert!(index.visible_at(EpochId::new(0)).is_none());

        // Add another version
        let hot2 = HotVersionRef::new(EpochId::new(5), EpochId::new(5), 100, TransactionId::new(2));
        index.add_hot(hot2);

        // Should see v1 at epoch < 5, v2 at epoch >= 5
        let v1 = index.visible_at(EpochId::new(4)).unwrap();
        assert!(matches!(v1, VersionRef::Hot(h) if h.arena_offset == 0));

        let v2 = index.visible_at(EpochId::new(5)).unwrap();
        assert!(matches!(v2, VersionRef::Hot(h) if h.arena_offset == 100));
    }

    #[test]
    fn test_version_index_deletion() {
        let hot = HotVersionRef::new(EpochId::new(1), EpochId::new(1), 0, TransactionId::new(1));
        let mut index = VersionIndex::with_initial(hot);

        // Mark as deleted at epoch 5
        assert!(index.mark_deleted(EpochId::new(5), TransactionId::new(99)));

        // Should see version before deletion, nothing after
        assert!(index.visible_at(EpochId::new(4)).is_some());
        assert!(index.visible_at(EpochId::new(5)).is_none());
        assert!(index.visible_at(EpochId::new(10)).is_none());
    }

    #[test]
    fn test_version_index_transaction_visibility() {
        let tx = TransactionId::new(10);
        let hot = HotVersionRef::new(EpochId::new(5), EpochId::new(5), 0, tx);
        let index = VersionIndex::with_initial(hot);

        // Creator can see it even at earlier epoch
        assert!(index.visible_to(EpochId::new(3), tx).is_some());

        // Other transactions cannot see it before creation
        assert!(
            index
                .visible_to(EpochId::new(3), TransactionId::new(20))
                .is_none()
        );
        assert!(
            index
                .visible_to(EpochId::new(5), TransactionId::new(20))
                .is_some()
        );
    }

    #[test]
    fn test_version_index_rollback() {
        let tx1 = TransactionId::new(10);
        let tx2 = TransactionId::new(20);

        let mut index = VersionIndex::new();
        index.add_hot(HotVersionRef::new(EpochId::new(1), EpochId::new(1), 0, tx1));
        index.add_hot(HotVersionRef::new(
            EpochId::PENDING,
            EpochId::new(2),
            100,
            tx2,
        ));
        index.add_hot(HotVersionRef::new(
            EpochId::PENDING,
            EpochId::new(2),
            200,
            tx2,
        ));

        assert_eq!(index.version_count(), 3);
        assert!(index.modified_by(tx1));
        assert!(index.modified_by(tx2));

        // Rollback tx2's PENDING changes
        index.remove_versions_by(tx2);

        assert_eq!(index.version_count(), 1);
        assert!(index.modified_by(tx1));
        assert!(!index.modified_by(tx2));

        // Should only see tx1's version
        let v = index.visible_at(EpochId::new(10)).unwrap();
        assert!(matches!(v, VersionRef::Hot(h) if h.created_by == tx1));
    }

    #[test]
    fn test_version_index_rollback_ignores_committed_same_tx_id() {
        let tx = TransactionId::new(42);
        let mut index = VersionIndex::new();
        index.add_hot(HotVersionRef::new(EpochId::new(3), EpochId::new(3), 0, tx));
        assert!(!index.solely_created_by(tx));
        index.remove_versions_by(tx);
        assert_eq!(index.version_count(), 1);

        index.add_hot(HotVersionRef::new(
            EpochId::PENDING,
            EpochId::new(4),
            50,
            tx,
        ));
        index.remove_versions_by(tx);
        assert_eq!(index.version_count(), 1);
        let v = index.visible_at(EpochId::new(3)).unwrap();
        assert!(matches!(v, VersionRef::Hot(h) if h.epoch == EpochId::new(3)));
    }

    #[test]
    fn test_version_index_unmark_requires_pending_delete() {
        let tx = TransactionId::new(9);
        let mut index = VersionIndex::new();
        index.add_hot(HotVersionRef::new(
            EpochId::new(1),
            EpochId::new(1),
            0,
            TransactionId::new(1),
        ));
        assert!(index.mark_deleted(EpochId::new(5), tx));
        assert!(!index.unmark_deleted_by(tx));

        let mut pending = VersionIndex::new();
        pending.add_hot(HotVersionRef::new(
            EpochId::new(1),
            EpochId::new(1),
            0,
            TransactionId::new(1),
        ));
        assert!(pending.mark_deleted(EpochId::PENDING, tx));
        assert!(pending.unmark_deleted_by(tx));
        assert!(!pending.deleted_by(tx));
    }

    #[test]
    // reason: test epoch values are small, fit u32
    #[allow(clippy::cast_possible_truncation)]
    fn test_version_index_gc() {
        let mut index = VersionIndex::new();

        // Add versions at epochs 1, 3, 5
        for epoch in [1, 3, 5] {
            index.add_hot(HotVersionRef::new(
                EpochId::new(epoch),
                EpochId::new(epoch),
                epoch as u32 * 100,
                TransactionId::new(epoch),
            ));
        }

        assert_eq!(index.version_count(), 3);

        // GC with min_epoch = 4
        // Should keep: epoch 5 (>= 4) and epoch 3 (first old visible)
        index.gc(EpochId::new(4));

        assert_eq!(index.version_count(), 2);

        // Verify we kept epochs 5 and 3
        assert!(index.visible_at(EpochId::new(5)).is_some());
        assert!(index.visible_at(EpochId::new(3)).is_some());
    }

    #[test]
    fn test_version_index_conflict_detection() {
        let tx1 = TransactionId::new(10);
        let tx2 = TransactionId::new(20);

        let mut index = VersionIndex::new();
        index.add_hot(HotVersionRef::new(EpochId::new(1), EpochId::new(1), 0, tx1));
        index.add_hot(HotVersionRef::new(
            EpochId::new(5),
            EpochId::new(5),
            100,
            tx2,
        ));

        // tx1 started at epoch 0, tx2 modified at epoch 5 -> conflict for tx1
        assert!(index.has_conflict(EpochId::new(0), tx1));

        // tx2 started at epoch 0, tx1 modified at epoch 1 -> also conflict for tx2
        assert!(index.has_conflict(EpochId::new(0), tx2));

        // tx1 started after tx2's modification -> no conflict
        assert!(!index.has_conflict(EpochId::new(5), tx1));

        // tx2 started after tx1's modification -> no conflict
        assert!(!index.has_conflict(EpochId::new(1), tx2));

        // If only tx1's version exists, tx1 doesn't conflict with itself
        let mut index2 = VersionIndex::new();
        index2.add_hot(HotVersionRef::new(EpochId::new(5), EpochId::new(5), 0, tx1));
        assert!(!index2.has_conflict(EpochId::new(0), tx1));
    }

    #[test]
    // reason: test loop index 0..2 fits u32
    #[allow(clippy::cast_possible_truncation)]
    fn test_version_index_smallvec_no_heap() {
        let mut index = VersionIndex::new();

        // Add 2 hot versions (within inline capacity)
        for i in 0..2 {
            index.add_hot(HotVersionRef::new(
                EpochId::new(i),
                EpochId::new(i),
                i as u32,
                TransactionId::new(i),
            ));
        }

        // SmallVec should not have spilled to heap
        assert!(!index.hot_spilled());
        assert!(!index.cold_spilled());
    }

    #[test]
    fn test_version_index_freeze_epoch() {
        let mut index = VersionIndex::new();
        index.add_hot(HotVersionRef::new(
            EpochId::new(1),
            EpochId::new(1),
            0,
            TransactionId::new(1),
        ));
        index.add_hot(HotVersionRef::new(
            EpochId::new(2),
            EpochId::new(2),
            100,
            TransactionId::new(2),
        ));

        assert_eq!(index.hot_count(), 2);
        assert_eq!(index.cold_count(), 0);

        // Freeze epoch 1
        let cold_ref = ColdVersionRef {
            epoch: EpochId::new(1),
            block_offset: 0,
            length: 32,
            created_by: TransactionId::new(1),
            deleted_epoch: OptionalEpochId::NONE,
            deleted_by: None,
        };
        index.freeze_epoch(EpochId::new(1), std::iter::once(cold_ref));

        // Hot should have 1, cold should have 1
        assert_eq!(index.hot_count(), 1);
        assert_eq!(index.cold_count(), 1);

        // Visibility should still work
        assert!(index.visible_at(EpochId::new(1)).is_some());
        assert!(index.visible_at(EpochId::new(2)).is_some());

        // Check that cold version is actually cold
        let v1 = index.visible_at(EpochId::new(1)).unwrap();
        assert!(v1.is_cold());

        let v2 = index.visible_at(EpochId::new(2)).unwrap();
        assert!(v2.is_hot());
    }

    #[test]
    fn test_version_ref_accessors() {
        let hot = HotVersionRef::new(
            EpochId::new(5),
            EpochId::new(5),
            100,
            TransactionId::new(10),
        );
        let vr = VersionRef::Hot(hot);

        assert_eq!(vr.epoch(), EpochId::new(5));
        assert_eq!(vr.created_by(), TransactionId::new(10));
        assert!(vr.is_hot());
        assert!(!vr.is_cold());
    }

    #[test]
    fn test_version_index_latest_epoch() {
        let mut index = VersionIndex::new();
        assert_eq!(index.latest_epoch(), EpochId::INITIAL);

        index.add_hot(HotVersionRef::new(
            EpochId::new(5),
            EpochId::new(5),
            0,
            TransactionId::new(1),
        ));
        assert_eq!(index.latest_epoch(), EpochId::new(5));

        index.add_hot(HotVersionRef::new(
            EpochId::PENDING,
            EpochId::new(10),
            100,
            TransactionId::new(2),
        ));
        assert_eq!(index.latest_epoch(), EpochId::PENDING);

        // After rollback of the PENDING create, should recalculate
        index.remove_versions_by(TransactionId::new(2));
        assert_eq!(index.latest_epoch(), EpochId::new(5));
    }

    #[test]
    fn test_version_index_default() {
        let index = VersionIndex::default();
        assert!(index.is_empty());
        assert_eq!(index.version_count(), 0);
    }

    #[test]
    fn test_version_index_latest() {
        let mut index = VersionIndex::new();
        assert!(index.latest().is_none());

        index.add_hot(HotVersionRef::new(
            EpochId::new(1),
            EpochId::new(1),
            0,
            TransactionId::new(1),
        ));
        let latest = index.latest().unwrap();
        assert!(matches!(latest, VersionRef::Hot(h) if h.epoch == EpochId::new(1)));

        index.add_hot(HotVersionRef::new(
            EpochId::new(5),
            EpochId::new(5),
            100,
            TransactionId::new(2),
        ));
        let latest = index.latest().unwrap();
        assert!(matches!(latest, VersionRef::Hot(h) if h.epoch == EpochId::new(5)));
    }
}
