//! Unified flush: one code path for every checkpoint.
//!
//! Periodic checkpoints, `wal_checkpoint()`, `close()` and the async snapshot
//! all write every section. A checkpoint writes a complete new container that
//! holds only the sections it was given, so leaving an unchanged section out
//! would drop it from the file. Writing only what changed needs a container
//! that keeps the other sections (incremental checkpoints, #430).

use grafeo_common::storage::{Section, SectionType};
use grafeo_common::utils::error::Result;

#[cfg(feature = "grafeo-file")]
use grafeo_storage::file::GrafeoFileManager;

use super::sections::CheckpointSources;

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

/// Executes the unified flush: serialize every section, write the container,
/// truncate the WAL.
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
    #[cfg(feature = "wal")] wal: Option<&grafeo_storage::wal::LpgWal>,
) -> Result<FlushResult> {
    use grafeo_common::testing::crash::maybe_crash;

    // One checkpoint at a time: another one interleaving these steps could
    // delete WAL files this one relies on.
    let _checkpoint = fm.checkpoint_guard();

    maybe_crash("flush:before_serialize");

    if sections.is_empty() {
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
    let mut targets: Vec<(SectionType, Vec<u8>)> = Vec::with_capacity(sections.len());
    for section in sections {
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

    for section in sections {
        section.mark_clean();
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

impl CheckpointSources {
    /// The header values of the checkpoint.
    pub fn context(&self) -> FlushContext {
        let transaction_id = self
            .transaction_manager
            .last_assigned_transaction_id()
            .map_or(0, |t| t.0);
        #[cfg(feature = "lpg")]
        if let Some(store) = &self.store {
            // After `compact()` the store is the overlay: count the base too.
            #[cfg(feature = "compact-store")]
            if let Some(layered) = &self.layered {
                use grafeo_core::graph::GraphStore;
                return FlushContext {
                    epoch: store.current_epoch().0,
                    transaction_id,
                    node_count: layered.node_count() as u64,
                    edge_count: layered.edge_count() as u64,
                };
            }
            return FlushContext {
                epoch: store.current_epoch().0,
                transaction_id,
                node_count: store.node_count() as u64,
                edge_count: store.edge_count() as u64,
            };
        }
        FlushContext {
            epoch: 0,
            transaction_id,
            node_count: 0,
            edge_count: 0,
        }
    }
}

#[cfg(all(test, feature = "grafeo-file"))]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// A section holding fixed bytes, with its own dirty flag.
    struct FixedSection {
        section_type: SectionType,
        data: Vec<u8>,
        dirty: AtomicBool,
    }

    impl FixedSection {
        fn new(section_type: SectionType, data: &[u8], dirty: bool) -> Self {
            Self {
                section_type,
                data: data.to_vec(),
                dirty: AtomicBool::new(dirty),
            }
        }
    }

    impl Section for FixedSection {
        fn section_type(&self) -> SectionType {
            self.section_type
        }

        fn serialize(&self) -> Result<Vec<u8>> {
            Ok(self.data.clone())
        }

        fn deserialize(&mut self, data: &[u8]) -> Result<()> {
            self.data = data.to_vec();
            Ok(())
        }

        fn is_dirty(&self) -> bool {
            self.dirty.load(Ordering::Acquire)
        }

        fn mark_clean(&self) {
            self.dirty.store(false, Ordering::Release);
        }

        fn memory_usage(&self) -> usize {
            self.data.len()
        }
    }

    fn run(fm: &GrafeoFileManager, sections: &[&dyn Section]) -> usize {
        let context = FlushContext {
            epoch: 1,
            transaction_id: 1,
            node_count: 0,
            edge_count: 0,
        };
        #[cfg(feature = "wal")]
        let result = flush(fm, sections, &context, None);
        #[cfg(not(feature = "wal"))]
        let result = flush(fm, sections, &context);
        result.unwrap().sections_written
    }

    fn stored(fm: &GrafeoFileManager, section_type: SectionType) -> Option<Vec<u8>> {
        let directory = fm.read_section_directory().unwrap()?;
        let entry = directory.find(section_type)?;
        Some(fm.read_section_data(entry).unwrap())
    }

    /// Each checkpoint writes a complete file holding only the sections it
    /// was given. The async snapshot used to write only the changed sections:
    /// that dropped the others from the file (and then deleted the WAL files
    /// that held them), and with nothing marked changed it wrote nothing.
    #[test]
    fn a_checkpoint_keeps_every_section_in_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let fm = GrafeoFileManager::create(dir.path().join("db.grafeo")).unwrap();
        let catalog = FixedSection::new(SectionType::Catalog, b"catalog", false);
        let store = FixedSection::new(SectionType::LpgStore, b"store", false);

        // Twice: the second checkpoint has nothing marked changed.
        for checkpoint in 0..2 {
            assert_eq!(run(&fm, &[&catalog, &store]), 2, "checkpoint {checkpoint}");
            assert_eq!(
                stored(&fm, SectionType::Catalog).as_deref(),
                Some(&b"catalog"[..])
            );
            assert_eq!(
                stored(&fm, SectionType::LpgStore).as_deref(),
                Some(&b"store"[..])
            );
        }

        let changed = FixedSection::new(SectionType::LpgStore, b"changed", true);
        assert_eq!(run(&fm, &[&catalog, &changed]), 2);
        assert_eq!(
            stored(&fm, SectionType::Catalog).as_deref(),
            Some(&b"catalog"[..]),
            "the unchanged section is still in the file"
        );
        assert_eq!(
            stored(&fm, SectionType::LpgStore).as_deref(),
            Some(&b"changed"[..])
        );
    }
}
