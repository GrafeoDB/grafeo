//! Unified flush: one code path for checkpoint, eviction, and explicit CHECKPOINT.
//!
//! The [`FlushManager`] replaces separate checkpoint/snapshot/close paths with
//! a single `flush()` method. Three triggers, one implementation:
//!
//! | Trigger | What gets written | RAM after flush |
//! |---------|-------------------|-----------------|
//! | Periodic checkpoint | All dirty sections | Kept |
//! | Memory pressure | Lowest-priority section | Mmap back, release RAM |
//! | Explicit CHECKPOINT | All sections | Kept |
//!
//! Future phases will add memory pressure integration with BufferManager.

use grafeo_common::storage::{Section, SectionType};
use grafeo_common::utils::error::Result;

#[cfg(feature = "grafeo-file")]
use grafeo_storage::file::GrafeoFileManager;

/// Reason for triggering a flush.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FlushReason {
    /// Periodic checkpoint (timer-driven) or database close.
    #[allow(dead_code)] // Used by async_ops (async-storage feature)
    Checkpoint,
    /// User-initiated `CHECKPOINT` command or `wal_checkpoint()` API.
    Explicit,
}

/// Context needed by each section during serialization.
pub(super) struct FlushContext {
    pub epoch: u64,
    pub transaction_id: u64,
    pub node_count: u64,
    pub edge_count: u64,
}

/// Result of a flush operation.
pub(super) struct FlushResult {
    /// Number of sections written to the container.
    pub sections_written: usize,
}

/// Executes the unified flush: serialize dirty sections, write to container, truncate WAL.
///
/// This is the single write path for all persistence operations. With a WAL,
/// the order is what makes a crash at any point safe (#417):
///
/// 1. start a new WAL file, so every record logged so far is in an earlier file,
/// 2. serialize the sections (the snapshot then contains all those records),
/// 3. write and sync the container,
/// 4. only then mark the WAL: recovery starts at the new file, and the earlier
///    files are deleted unless an incremental backup still needs them.
///
/// A crash before step 4 leaves the previous mark in place, so recovery
/// replays more than needed, which is harmless because replay is idempotent.
///
/// # Errors
///
/// Returns an error if serialization or I/O fails.
#[cfg(feature = "grafeo-file")]
pub(super) fn flush(
    fm: &GrafeoFileManager,
    sections: &[&dyn Section],
    context: &FlushContext,
    reason: FlushReason,
    #[cfg(feature = "wal")] wal: Option<&grafeo_storage::wal::LpgWal>,
) -> Result<FlushResult> {
    use grafeo_common::testing::crash::maybe_crash;

    // One checkpoint at a time: another one interleaving these steps could
    // delete WAL files this one relies on.
    let _checkpoint = fm.checkpoint_guard();

    maybe_crash("flush:before_serialize");

    // Write all sections for Explicit, only dirty ones for Checkpoint. If
    // nothing is dirty on a periodic checkpoint, skip the write entirely:
    // previous sections remain intact in the container.
    let selected: Vec<&dyn Section> = sections
        .iter()
        .copied()
        .filter(|section| reason == FlushReason::Explicit || section.is_dirty())
        .collect();
    if selected.is_empty() {
        return Ok(FlushResult {
            sections_written: 0,
        });
    }

    // Step 1: records logged before this point land in files below
    // `covered_sequence`, and their effects are in the snapshot below.
    #[cfg(feature = "wal")]
    let covered_sequence = match wal {
        Some(wal) => {
            wal.rotate()?;
            Some(wal.current_sequence())
        }
        None => None,
    };

    maybe_crash("flush:after_rotate");

    // Step 2: serialize.
    let mut targets: Vec<(SectionType, Vec<u8>)> = Vec::with_capacity(selected.len());
    for section in &selected {
        targets.push((section.section_type(), section.serialize()?));
    }

    let sections_written = targets.len();

    maybe_crash("flush:after_serialize");

    // Write sections to container
    let section_refs: Vec<(SectionType, &[u8])> =
        targets.iter().map(|(t, d)| (*t, d.as_slice())).collect();

    fm.write_sections(
        &section_refs,
        context.epoch,
        context.transaction_id,
        context.node_count,
        context.edge_count,
    )?;

    // Mark all written sections as clean
    for section in sections {
        if targets.iter().any(|(t, _)| *t == section.section_type()) {
            section.mark_clean();
        }
    }

    maybe_crash("flush:after_write");

    // Step 4: the container is durable (`write_sections` syncs it).
    #[cfg(feature = "wal")]
    if let (Some(wal), Some(sequence)) = (wal, covered_sequence) {
        use grafeo_common::types::{EpochId, TransactionId};

        wal.mark_checkpoint(
            sequence,
            EpochId::new(context.epoch),
            TransactionId::new(context.transaction_id),
        )?;

        maybe_crash("flush:after_mark_checkpoint");

        // Incremental backups read the files after their cursor: keep those.
        let keep_from = match super::backup::read_backup_cursor(wal.dir()) {
            Ok(Some(cursor)) => sequence.min(cursor.log_sequence + 1),
            Ok(None) => sequence,
            Err(e) => {
                grafeo_common::grafeo_warn!(
                    "keeping WAL files after checkpoint: cannot read backup cursor: {}",
                    e
                );
                0
            }
        };
        wal.remove_files_before(keep_from)?;
        wal.sync()?;
    }

    Ok(FlushResult { sections_written })
}

/// Builds the flush context from the current database state.
#[cfg(feature = "lpg")]
pub(super) fn build_context(
    store: &grafeo_core::graph::lpg::LpgStore,
    transaction_manager: &crate::transaction::TransactionManager,
) -> FlushContext {
    FlushContext {
        epoch: store.current_epoch().0,
        transaction_id: transaction_manager
            .last_assigned_transaction_id()
            .map_or(0, |t| t.0),
        node_count: store.node_count() as u64,
        edge_count: store.edge_count() as u64,
    }
}

/// Builds a minimal flush context when no LPG store is available.
#[cfg(not(feature = "lpg"))]
pub(super) fn build_context_minimal(
    transaction_manager: &crate::transaction::TransactionManager,
) -> FlushContext {
    FlushContext {
        epoch: 0,
        transaction_id: transaction_manager
            .last_assigned_transaction_id()
            .map_or(0, |t| t.0),
        node_count: 0,
        edge_count: 0,
    }
}
