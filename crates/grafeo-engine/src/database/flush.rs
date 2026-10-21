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

#[cfg(all(feature = "wal", feature = "grafeo-file"))]
use std::sync::atomic::{AtomicU64, Ordering};

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
    /// The values the checkpoint records in its database header, with
    /// `checkpoint_lsn`, the log position the image covers: a reopen
    /// replays the WAL from there.
    pub fn checkpoint_header(&self, checkpoint_lsn: u64) -> CheckpointHeader {
        CheckpointHeader {
            checkpoint_lsn,
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
/// 1. start a new WAL segment: every group logged so far is in the sealed,
///    synced segments before its first log position, the checkpoint LSN,
/// 2. serialize the sections and write them as a new image (the snapshot then
///    contains all those groups), whose database header records the
///    checkpoint LSN; the file manager syncs the image, then switches the
///    database header to it, and from then on a reopen replays the WAL from
///    that LSN,
/// 3. only then delete the segments before it, except those an incremental
///    backup still needs.
///
/// A crash before the header switch leaves the previous header, whose
/// checkpoint LSN is older, and the segments after it: a reopen replays them
/// onto the previous image. A failed image write keeps every segment for the
/// same reason: the new header may have reached the disk, or not.
///
/// Lock order: the file's checkpoint guard, then the commit lock.
///
/// # Errors
///
/// Returns an error if serialization or I/O fails, or after a commit that
/// did not complete.
#[cfg(feature = "grafeo-file")]
pub(super) fn flush(fm: &GrafeoFileManager, sources: &CheckpointSources) -> Result<FlushResult> {
    flush_covering(fm, sources, None)
}

/// [`flush`], recording `covered_lsn` as the image's checkpoint LSN when the
/// database has no WAL: the end of a log the open replayed and retires, so
/// a crash before it is removed does not replay it onto the image again.
/// Without it a WAL-less checkpoint keeps the previous image's.
///
/// # Errors
///
/// As [`flush`].
#[cfg(feature = "grafeo-file")]
pub(super) fn flush_covering(
    fm: &GrafeoFileManager,
    sources: &CheckpointSources,
    covered_lsn: Option<u64>,
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
        sources
            .wal
            .as_deref()
            .map(|wal| (wal, &*sources.wal_records_at_checkpoint)),
        covered_lsn,
        commits,
    )
}

/// Steps 1 to 3 of [`flush`] for `sections`: the caller holds the file's
/// checkpoint guard, and `commits` until the image is written, when this
/// releases them. `log` is the WAL, with the count of its records the
/// checkpoint covers, which this sets.
#[cfg(feature = "grafeo-file")]
fn write_sections(
    fm: &GrafeoFileManager,
    sections: &[&dyn Section],
    context: &FlushContext,
    #[cfg(feature = "wal")] log: Option<(&grafeo_storage::wal::Wal, &AtomicU64)>,
    covered_lsn: Option<u64>,
    commits: CommitsHeld<'_>,
) -> Result<FlushResult> {
    use grafeo_common::testing::crash::maybe_crash;

    maybe_crash("flush:before_serialize");

    if sections.is_empty() {
        return Ok(FlushResult {
            sections_written: 0,
        });
    }

    // Step 1: the groups logged before the checkpoint LSN are in sealed
    // segments, and their effects are in the snapshot below (commits are
    // held, so no group follows them until the image is written).
    #[cfg(feature = "wal")]
    let covered = match log {
        Some((wal, _)) => Some((wal.rotate()?, wal.records_written())),
        None => None,
    };
    // Without a WAL the image covers the log the previous one did, or the
    // one the caller names.
    let without_log = covered_lsn.unwrap_or_else(|| fm.active_header().checkpoint_lsn);
    #[cfg(feature = "wal")]
    let checkpoint_lsn = covered.map_or(without_log, |(lsn, _)| lsn);
    #[cfg(not(feature = "wal"))]
    let checkpoint_lsn = without_log;

    maybe_crash("flush:after_rotate");

    // Step 2: serialize each section into the new image.
    fm.write_checkpoint(sections, &context.checkpoint_header(checkpoint_lsn))?;
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
    if let (Some((wal, records)), Some((lsn, written))) = (log, covered) {
        records.store(written, Ordering::Release);

        maybe_crash("flush:before_remove_segments");

        // An incremental backup reads the segments after its cursor: keep
        // those.
        let keep_from = match super::backup::read_backup_cursor(wal.dir()) {
            Ok(Some(cursor)) => cursor.wal_lsn.map_or(lsn, |backed_up| lsn.min(backed_up)),
            Ok(None) => lsn,
            Err(e) => {
                grafeo_common::grafeo_warn!(
                    "keeping WAL segments after checkpoint: cannot read backup cursor: {}",
                    e
                );
                0
            }
        };
        wal.remove_segments_before(keep_from)?;
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
        if let Some((nodes, edges)) = self.counts() {
            return FlushContext {
                epoch,
                transaction_id,
                node_count: nodes as u64,
                edge_count: edges as u64,
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
        let result = write_sections(fm, sections, context, None, None, commits);
        #[cfg(not(feature = "wal"))]
        let result = write_sections(fm, sections, context, None, commits);
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

    /// The new image's database header records the flush context; without
    /// a WAL it keeps the previous image's log position (0 for the first).
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
