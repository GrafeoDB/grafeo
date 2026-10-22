//! Unified flush: one code path for every checkpoint.
//!
//! Periodic checkpoints, `wal_checkpoint()`, `close()` and the async snapshot
//! all write every section. A checkpoint writes a complete new image that
//! holds only the sections it was given, so leaving an unchanged section out
//! would drop it from the file. Writing only what changed needs an image that
//! keeps the other sections (incremental checkpoints, #430).

use grafeo_common::storage::Section;
use grafeo_common::utils::error::Result;

#[cfg(feature = "grafeo-file")]
use grafeo_storage::file::{CheckpointHeader, GrafeoFileManager};

use super::sections::CheckpointSources;
#[cfg(feature = "grafeo-file")]
use crate::transaction::CommitsHeld;

/// Context needed by each section during serialization.
pub(super) struct FlushContext {
    pub epoch: u64,
    pub transaction_id: u64,
    pub node_count: u64,
    pub edge_count: u64,
}

#[cfg(feature = "grafeo-file")]
impl FlushContext {
    /// The values the checkpoint records in its database header.
    /// `checkpoint_lsn` stays 0 until the WAL records sequence numbers: the
    /// WAL's own checkpoint marker tells recovery where to start.
    pub fn checkpoint_header(&self) -> CheckpointHeader {
        CheckpointHeader {
            checkpoint_lsn: 0,
            epoch: self.epoch,
            last_transaction_id: self.transaction_id,
            node_count: self.node_count,
            edge_count: self.edge_count,
        }
    }
}

/// Result of a flush operation.
pub(super) struct FlushResult {
    /// Number of sections written to the container.
    pub sections_written: usize,
}

/// Executes the unified flush of `sources`: write every section as a new
/// image, then truncate the WAL.
///
/// This is the single write path for all persistence operations. Commits are
/// held off (see
/// [`TransactionManager::hold_commits`](crate::transaction::TransactionManager))
/// from before the sections are built until the image is written: the image
/// holds every commit whole or not at all, never one that did not complete,
/// and its header's epoch and counts are those of its sections. With a WAL,
/// the order of the steps is what makes a crash at any point safe (#417):
///
/// 1. start a new WAL file, so every record logged so far is in an earlier file,
/// 2. serialize the sections and write them as a new image (the snapshot then
///    contains all those records); the file manager syncs the image, then
///    switches the database header to it,
/// 3. only then mark the WAL: recovery starts at the new file, and the earlier
///    files are deleted unless an incremental backup still needs them.
///
/// A crash before step 3 leaves the previous mark in place, so recovery
/// replays more than needed, which is harmless because replay is idempotent.
/// A failed image write keeps the WAL for the same reason: the new header may
/// have reached the disk, or not.
///
/// Lock order: the file's checkpoint guard, then the commit lock.
///
/// # Errors
///
/// Returns an error if serialization or I/O fails, or after a commit that
/// did not complete.
#[cfg(feature = "grafeo-file")]
pub(super) fn flush(
    fm: &GrafeoFileManager,
    sources: &CheckpointSources,
    #[cfg(feature = "wal")] wal: Option<&grafeo_storage::wal::LpgWal>,
) -> Result<FlushResult> {
    #[cfg(feature = "testing-statement-injection")]
    grafeo_common::testing::commit_hook::count_checkpoint(fm.path());
    // One checkpoint at a time: another one interleaving these steps could
    // delete WAL files this one relies on.
    let _checkpoint = fm.checkpoint_guard();
    let commits = sources.transaction_manager.hold_commits()?;
    let sections = sources.sections(&commits);
    let section_refs: Vec<&dyn Section> = sections.iter().map(AsRef::as_ref).collect();
    let context = sources.context();

    #[cfg(feature = "testing-statement-injection")]
    grafeo_common::testing::commit_hook::run_during_checkpoint();

    write_sections(
        fm,
        &section_refs,
        &context,
        #[cfg(feature = "wal")]
        wal,
        commits,
    )
}

/// Steps 1 to 3 of [`flush`] for `sections`: the caller holds the file's
/// checkpoint guard, and `commits` until the image is written, when this
/// releases them.
#[cfg(feature = "grafeo-file")]
fn write_sections(
    fm: &GrafeoFileManager,
    sections: &[&dyn Section],
    context: &FlushContext,
    #[cfg(feature = "wal")] wal: Option<&grafeo_storage::wal::LpgWal>,
    commits: CommitsHeld<'_>,
) -> Result<FlushResult> {
    use grafeo_common::testing::crash::maybe_crash;

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

    // Step 2: serialize each section into the new image.
    fm.write_checkpoint(sections, &context.checkpoint_header())?;
    let sections_written = sections.len();

    for section in sections {
        section.mark_clean();
    }
    // The image is written: commits may go on while the WAL is marked.
    drop(commits);

    maybe_crash("flush:after_write");

    // Step 3: the image is durable and active (`write_checkpoint` synced the
    // image and its header).
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
        let epoch = self.epoch();
        #[cfg(feature = "lpg")]
        if let Some(store) = &self.root_store() {
            return FlushContext {
                epoch,
                transaction_id,
                node_count: store.node_count() as u64,
                edge_count: store.edge_count() as u64,
            };
        }
        FlushContext {
            epoch,
            transaction_id,
            node_count: 0,
            edge_count: 0,
        }
    }
}

#[cfg(all(test, feature = "grafeo-file"))]
mod tests {
    use super::*;
    use grafeo_common::storage::{
        SectionSink, SectionSource, SectionType, legacy_bytes, read_raw, write_raw,
    };
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

        fn write_to(&self, sink: &mut dyn SectionSink) -> Result<()> {
            write_raw(self, sink)
        }

        fn read_from(&mut self, source: &dyn SectionSource) -> Result<()> {
            read_raw(self, source)
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

    fn run_with(
        fm: &GrafeoFileManager,
        sections: &[&dyn Section],
        context: &FlushContext,
    ) -> usize {
        let manager = crate::transaction::TransactionManager::new();
        let _checkpoint = fm.checkpoint_guard();
        let commits = manager.hold_commits().unwrap();
        #[cfg(feature = "wal")]
        let result = write_sections(fm, sections, context, None, commits);
        #[cfg(not(feature = "wal"))]
        let result = write_sections(fm, sections, context, commits);
        result.unwrap().sections_written
    }

    fn run(fm: &GrafeoFileManager, sections: &[&dyn Section]) -> usize {
        let context = FlushContext {
            epoch: 1,
            transaction_id: 1,
            node_count: 0,
            edge_count: 0,
        };
        run_with(fm, sections, &context)
    }

    /// The bytes of a section of the active image, stored as one raw chunk.
    fn stored(fm: &GrafeoFileManager, section_type: SectionType) -> Option<Vec<u8>> {
        fm.read_image(|image| match image.section_source(section_type) {
            Some(section) => Ok(legacy_bytes(&*section)?.map(Vec::from)),
            None => Ok(None),
        })
        .unwrap()
    }

    /// The new image's database header records the flush context; it has
    /// no WAL sequence number until the WAL records them.
    #[test]
    fn the_checkpoint_header_carries_the_flush_context() {
        let dir = tempfile::tempdir().unwrap();
        let fm = GrafeoFileManager::create(dir.path().join("db.grafeo"), None).unwrap();
        let catalog = FixedSection::new(SectionType::Catalog, b"Amsterdam", true);
        let context = FlushContext {
            epoch: 19,
            transaction_id: 88,
            node_count: 3,
            edge_count: 319,
        };
        assert_eq!(run_with(&fm, &[&catalog], &context), 1);

        let header = fm.active_header();
        assert_eq!(
            (
                header.iteration,
                header.checkpoint_lsn,
                header.epoch,
                header.last_transaction_id,
                header.node_count,
                header.edge_count
            ),
            (1, 0, 19, 88, 3, 319)
        );
        assert!(!catalog.is_dirty(), "a written section is clean");
    }

    /// Each checkpoint writes a complete file holding only the sections it
    /// was given. The async snapshot used to write only the changed sections:
    /// that dropped the others from the file (and then deleted the WAL files
    /// that held them), and with nothing marked changed it wrote nothing.
    #[test]
    fn a_checkpoint_keeps_every_section_in_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let fm = GrafeoFileManager::create(dir.path().join("db.grafeo"), None).unwrap();
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
