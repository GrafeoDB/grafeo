//! Column spill of the row-group store (P8): the files its spilled columns
//! read, and the memory consumers that spill them when the buffer manager
//! asks.
//!
//! A spilled column is cold encoded chunks whose bytes are a spill file's
//! mapping (`ColumnSpillFiles`, in a build with `wal`, whose container code
//! writes and maps them): one new file per row group's column, in the open's
//! vector cache directory, named so that no other spill and no other process
//! ever has the name. A file lives as long as a chunk reads it: a write or a
//! reload that brings the column back into memory, or the store dropped,
//! unmaps and removes it. A file is never data: a checkpoint writes the
//! column's values into the database file, and what a crash leaves in the
//! directory the next read-write open removes
//! (`spill_directory::remove_stale_cache`). Nothing spills once the database
//! is closed (`SpillDirectory::close_for_writes`).
//!
//! `RowGroupVectorConsumer` reports the vector indexes of the root and every
//! named graph: under memory pressure it spills each vector index's
//! embedding column, and its reload brings every spilled column back.
//! `RowGroupTextConsumer` reports the text indexes; `RowGroupStoreConsumer`
//! the store's own memory, as the grants of its account hold it
//! ([`register_row_group_consumers`]). Each holds the root weakly: the
//! store's grants hold the buffer manager, so a consumer that held the
//! store would keep both alive past the database.

#[cfg(all(feature = "wal", feature = "vector-index"))]
use std::io;
#[cfg(all(feature = "wal", feature = "vector-index"))]
use std::path::Path;
#[cfg(all(feature = "wal", feature = "vector-index"))]
use std::path::PathBuf;
#[cfg(all(feature = "wal", feature = "vector-index"))]
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

#[cfg(all(feature = "wal", feature = "vector-index"))]
use bytes::Bytes;
use grafeo_common::memory::buffer::BufferManager;
use grafeo_common::memory::buffer::{MemoryConsumer, MemoryRegion, priorities};
#[cfg(feature = "vector-index")]
use grafeo_common::memory::buffer::{SpillCause, SpillError};
#[cfg(all(feature = "wal", feature = "vector-index"))]
use grafeo_common::storage::SectionType;
#[cfg(feature = "vector-index")]
use grafeo_common::types::PropertyKey;
#[cfg(all(feature = "wal", feature = "vector-index"))]
use grafeo_core::execution::spill::{SpillAccount, SpillOwner, SpillReservation};
#[cfg(all(feature = "wal", feature = "vector-index"))]
use grafeo_core::graph::rowgroup::ColumnSpill;
use grafeo_core::graph::rowgroup::RowGroupStore;
#[cfg(all(feature = "wal", feature = "vector-index"))]
use grafeo_storage::container::{MmapSection, write_and_mmap_spill_file};

#[cfg(all(feature = "wal", feature = "vector-index"))]
use super::spill_directory::SpillDirectory;

/// Writes the row-group store's spilled columns into new files in the
/// open's vector cache directory, each mapped (see the module docs). It
/// keeps the directory alive while it spills into it, and refuses to spill
/// once the database is closed.
#[cfg(all(feature = "wal", feature = "vector-index"))]
pub(super) struct ColumnSpillFiles {
    directory: Arc<SpillDirectory>,
    /// The open's spill account: each file reserves its bytes there before
    /// it is written, and gives them back when it is removed.
    account: Option<Arc<SpillAccount>>,
}

#[cfg(all(feature = "wal", feature = "vector-index"))]
impl ColumnSpillFiles {
    /// Spill files in `directory`, created when the first one is written,
    /// reserving their bytes in `account`.
    pub(super) fn new(directory: Arc<SpillDirectory>, account: Option<Arc<SpillAccount>>) -> Self {
        Self { directory, account }
    }

    /// Spill files in a directory of their own at `path`, unbounded (tests).
    #[cfg(test)]
    pub(super) fn in_dir(path: impl Into<PathBuf>) -> Self {
        Self::new(Arc::new(SpillDirectory::at(path.into())), None)
    }

    /// The directory the files go to.
    pub(super) fn directory(&self) -> &Path {
        self.directory.path()
    }
}

#[cfg(all(feature = "wal", feature = "vector-index"))]
impl ColumnSpill for ColumnSpillFiles {
    fn write(&self, bytes: &[u8]) -> io::Result<Bytes> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        // Reserved before anything is written: a refusal writes nothing.
        let reservation = match &self.account {
            Some(account) => Some(
                account
                    .reserve(SpillOwner::VectorColumns, bytes.len() as u64, true)
                    .map_err(grafeo_core::execution::spill::refused)?,
            ),
            None => None,
        };
        // Created on the first spill; refused once the database is closed.
        self.directory.create()?;
        // As a vector cache file's name: the process id keeps processes
        // apart, the counter this one's spills, the random part the rest.
        let path = self.directory().join(format!(
            "columns_{}_{}_{:016x}.bin",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed),
            super::spill_directory::random_u64()
        ));
        match write_and_mmap_spill_file(&path, bytes, SectionType::LpgStore) {
            Ok(map) => Ok(Bytes::from_owner(SpillMapping {
                map: Some(map),
                path,
                _reservation: reservation,
            })),
            Err(error) => {
                remove_written(&path);
                Err(io::Error::other(error.to_string()))
            }
        }
    }
}

/// The spill account's refusal a column spill's error carries.
#[cfg(all(feature = "wal", feature = "vector-index"))]
fn spill_refusal(
    error: &grafeo_common::utils::error::Error,
) -> Option<grafeo_common::utils::error::LimitRefusal> {
    match error {
        grafeo_common::utils::error::Error::Io(error) => {
            grafeo_core::execution::spill::refusal_of(error).cloned()
        }
        _ => None,
    }
}

/// Removes what a failed write left at `path`: the file, or the temporary
/// one the write renames into it.
#[cfg(all(feature = "wal", feature = "vector-index"))]
fn remove_written(path: &Path) {
    for leftover in [path.to_path_buf(), path.with_extension("spill.tmp")] {
        if leftover.exists() {
            // Best effort: the caller reports the write's own error.
            let _ = std::fs::remove_file(&leftover);
        }
    }
}

/// A spill file's mapping, which the chunks of a spilled column slice: the
/// last slice dropped unmaps the file and removes it.
#[cfg(all(feature = "wal", feature = "vector-index"))]
struct SpillMapping {
    /// `None` only while dropping (the mapping goes before the file).
    map: Option<MmapSection>,
    path: PathBuf,
    /// The file's bytes in the spill account, given back after the file is
    /// removed (the field drops after `drop` ran).
    _reservation: Option<SpillReservation>,
}

#[cfg(all(feature = "wal", feature = "vector-index"))]
impl AsRef<[u8]> for SpillMapping {
    fn as_ref(&self) -> &[u8] {
        self.map.as_ref().map_or(&[], MmapSection::as_bytes)
    }
}

#[cfg(all(feature = "wal", feature = "vector-index"))]
impl Drop for SpillMapping {
    fn drop(&mut self) {
        // Unmap first: a mapped file cannot be removed everywhere.
        self.map = None;
        if let Err(error) = std::fs::remove_file(&self.path) {
            grafeo_common::grafeo_warn!(
                "could not remove the column spill file {}: {error}",
                self.path.display()
            );
        }
    }
}

/// The root and its named graphs, each a store.
#[cfg(any(feature = "vector-index", feature = "text-index"))]
fn graphs(root: &Arc<RowGroupStore>) -> Vec<Arc<RowGroupStore>> {
    std::iter::once(Arc::clone(root))
        .chain(
            root.graph_names()
                .iter()
                .filter_map(|name| root.graph(name)),
        )
        .collect()
}

/// The memory consumer of the row-group store's vector indexes (see the
/// module docs): their memory, and the spill of their embedding columns.
#[cfg(feature = "vector-index")]
pub(super) struct RowGroupVectorConsumer {
    root: Weak<RowGroupStore>,
    /// Where spilled columns go; `None` disables spilling.
    #[cfg(all(feature = "wal", feature = "vector-index"))]
    files: Option<ColumnSpillFiles>,
    /// The database's collect horizon: a spill collects each graph at it
    /// first, since a column whose rows still keep versions does not spill.
    #[cfg(all(feature = "wal", feature = "vector-index"))]
    horizon: Option<CollectHorizon>,
}

/// The collect horizon of a database (see
/// `TransactionManager::collect_horizon`), read when a spill needs it.
#[cfg(all(feature = "wal", feature = "vector-index"))]
pub(super) type CollectHorizon = Arc<dyn Fn() -> grafeo_common::types::EpochId + Send + Sync>;

#[cfg(feature = "vector-index")]
impl RowGroupVectorConsumer {
    /// The consumer of `root`'s vector indexes and those of its named
    /// graphs, spilling into `files`.
    #[cfg(all(feature = "wal", feature = "vector-index"))]
    pub(super) fn new(root: &Arc<RowGroupStore>, files: Option<ColumnSpillFiles>) -> Self {
        Self {
            root: Arc::downgrade(root),
            files,
            horizon: None,
        }
    }

    /// Collects each graph at `horizon` before it spills.
    #[cfg(all(feature = "wal", feature = "vector-index"))]
    #[must_use]
    pub(super) fn collecting_at(mut self, horizon: Option<CollectHorizon>) -> Self {
        self.horizon = horizon;
        self
    }

    /// The consumer of `root`'s vector indexes and those of its named
    /// graphs, in a build that writes no spill files.
    #[cfg(not(all(feature = "wal", feature = "vector-index")))]
    pub(super) fn new(root: &Arc<RowGroupStore>) -> Self {
        Self {
            root: Arc::downgrade(root),
        }
    }

    /// The embedding properties of a graph's vector indexes, each once, in
    /// order (indexes on two labels can share one property column).
    fn indexed_properties(store: &RowGroupStore) -> Vec<PropertyKey> {
        let mut properties: Vec<PropertyKey> = store
            .vector_index_entries()
            .iter()
            .map(|(_, property, _)| PropertyKey::new(property))
            .collect();
        properties.sort_unstable();
        properties.dedup();
        properties
    }

    fn root(&self) -> Result<Arc<RowGroupStore>, SpillError> {
        self.root
            .upgrade()
            .ok_or_else(|| SpillError::IoError("store dropped".to_string()))
    }
}

#[cfg(feature = "vector-index")]
impl MemoryConsumer for RowGroupVectorConsumer {
    fn name(&self) -> &str {
        "section:VectorStore"
    }

    /// What an explicit spill frees: the heap bytes of the indexed
    /// embedding columns that are not spilled yet (see
    /// [`spillable_bytes`](MemoryConsumer::spillable_bytes) for a spill
    /// under pressure). The indexes' graphs (the HNSW topology) stay in
    /// memory through a spill, so they are not reported here; the store's
    /// account counts them. Walks the row groups of each indexed property:
    /// read on the eviction path, not by statements.
    fn memory_usage(&self) -> usize {
        self.spillable_bytes(SpillCause::Explicit)
    }

    /// What a spill for `cause` frees: under pressure, each graph's newest
    /// row group keeps its column (its next creates would read it back).
    fn spillable_bytes(&self, cause: SpillCause) -> usize {
        self.root.upgrade().map_or(0, |root| {
            graphs(&root)
                .iter()
                .map(|graph| {
                    Self::indexed_properties(graph)
                        .iter()
                        .map(|property| match cause {
                            SpillCause::Explicit => {
                                graph.node_property_column_spillable_bytes(property)
                            }
                            SpillCause::Pressure => {
                                graph.node_property_column_spillable_bytes_but_newest(property)
                            }
                        })
                        .sum::<usize>()
                })
                .sum()
        })
    }

    fn eviction_priority(&self) -> u8 {
        priorities::INDEX_BUFFERS
    }

    fn region(&self) -> MemoryRegion {
        MemoryRegion::IndexBuffers
    }

    fn evict(&self, _target_bytes: usize) -> usize {
        0
    }

    #[cfg(all(feature = "wal", feature = "vector-index"))]
    fn can_spill(&self) -> bool {
        self.files.is_some()
    }

    #[cfg(all(feature = "wal", feature = "vector-index"))]
    fn spill(&self, target_bytes: usize) -> Result<usize, SpillError> {
        self.spill_for(target_bytes, SpillCause::Explicit)
    }

    /// Spills the indexed embedding columns of every graph: all their row
    /// groups for an explicit spill, all but each graph's newest under
    /// pressure (see [`SpillCause`]).
    #[cfg(all(feature = "wal", feature = "vector-index"))]
    fn spill_for(&self, _target_bytes: usize, cause: SpillCause) -> Result<usize, SpillError> {
        let files = self.files.as_ref().ok_or(SpillError::NoSpillDirectory)?;
        let root = self.root()?;
        let mut freed = 0;
        let mut refused = None;
        if let Some(horizon) = &self.horizon {
            // The root's collect covers every named graph's.
            root.gc_versions(horizon());
        }
        for graph in graphs(&root) {
            // A row group already spilled is left as it is by the store, so
            // a column some group spilled still spills its other groups.
            for property in Self::indexed_properties(&graph) {
                let (report, result) = match cause {
                    SpillCause::Explicit => graph.spill_node_property_column(&property, files),
                    // Each graph's newest row group stays: its next creates
                    // would read a spilled column back.
                    SpillCause::Pressure => {
                        graph.spill_node_property_column_but_newest(&property, files)
                    }
                };
                // What spilled before an error counts too: the store returns
                // its partial report with the error.
                freed += report.freed_bytes;
                match result {
                    Ok(()) => {}
                    // The spill account refused: this column stays in memory,
                    // and a smaller one after it may still fit.
                    Err(error) if let Some(refusal) = spill_refusal(&error) => {
                        grafeo_common::grafeo_warn!(
                            "the vector column {} stays in memory: {refusal}",
                            property.as_str()
                        );
                        refused.get_or_insert(refusal);
                    }
                    // Continue: what spilled so far counts, and this column
                    // keeps what did not spill on the heap.
                    Err(error) => grafeo_common::grafeo_warn!(
                        "failed to spill the vector column {}: {error}",
                        property.as_str()
                    ),
                }
            }
        }
        match refused {
            // Nothing freed, and the account refused: the refusal is the answer.
            Some(refusal) if freed == 0 => Err(SpillError::Refused(Box::new(refusal))),
            _ => Ok(freed),
        }
    }

    fn reload(&self) -> Result<(), SpillError> {
        let root = self.root()?;
        // Every spilled column comes back, an index's or not (one dropped
        // since its column spilled): a column that cannot be read stays
        // spilled (nothing is lost), the others still reload.
        let mut first_error = None;
        for graph in graphs(&root) {
            for property in graph.spilled_node_property_columns() {
                if let Err(error) = graph.reload_node_property_column(&property) {
                    grafeo_common::grafeo_warn!(
                        "the vector column {} stays spilled: {error}",
                        property.as_str()
                    );
                    first_error.get_or_insert(SpillError::IoError(error.to_string()));
                }
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    fn current_tier(&self) -> grafeo_common::memory::StorageTier {
        use grafeo_common::memory::StorageTier;
        if self.root.upgrade().is_some_and(|root| {
            graphs(&root)
                .iter()
                .any(|graph| !graph.spilled_node_property_columns().is_empty())
        }) {
            return StorageTier::OnDisk;
        }
        if self.memory_usage() == 0 {
            StorageTier::Uninitialized
        } else {
            StorageTier::InMemory
        }
    }
}

/// The memory consumer of the row-group store's text indexes: it reports
/// the bytes of their inverted indexes (postings, terms and document
/// lengths, as each index counts them). Those are built in memory from the
/// indexed property and have no file form to read from, so the consumer
/// can neither spill nor evict: it reports what they hold, for the buffer
/// manager's figures.
#[cfg(feature = "text-index")]
pub(super) struct RowGroupTextConsumer {
    root: Weak<RowGroupStore>,
}

#[cfg(feature = "text-index")]
impl RowGroupTextConsumer {
    /// The consumer of `root`'s text indexes and those of its named graphs.
    pub(super) fn new(root: &Arc<RowGroupStore>) -> Self {
        Self {
            root: Arc::downgrade(root),
        }
    }
}

#[cfg(feature = "text-index")]
impl MemoryConsumer for RowGroupTextConsumer {
    fn name(&self) -> &str {
        "section:TextIndex"
    }

    fn memory_usage(&self) -> usize {
        self.root.upgrade().map_or(0, |root| {
            graphs(&root)
                .iter()
                .flat_map(|graph| graph.text_index_entries())
                .map(|(_, _, index)| index.read().counted_bytes())
                .sum()
        })
    }

    fn eviction_priority(&self) -> u8 {
        priorities::INDEX_BUFFERS
    }

    fn region(&self) -> MemoryRegion {
        MemoryRegion::IndexBuffers
    }

    fn evict(&self, _target_bytes: usize) -> usize {
        0
    }

    fn current_tier(&self) -> grafeo_common::memory::StorageTier {
        if self.memory_usage() == 0 {
            grafeo_common::memory::StorageTier::Uninitialized
        } else {
            grafeo_common::memory::StorageTier::InMemory
        }
    }
}

/// The memory consumer of the row-group store itself, its named graphs
/// with it: the bytes the grants of their accounts hold, read without a
/// walk. It cannot evict or spill (the store reports so until 0.6.3).
pub(super) struct RowGroupStoreConsumer {
    root: Weak<RowGroupStore>,
}

impl RowGroupStoreConsumer {
    /// The consumer of `root` and its named graphs.
    pub(super) fn new(root: &Arc<RowGroupStore>) -> Self {
        Self {
            root: Arc::downgrade(root),
        }
    }
}

impl MemoryConsumer for RowGroupStoreConsumer {
    fn name(&self) -> &str {
        "section:LpgStore"
    }

    fn memory_usage(&self) -> usize {
        self.root.upgrade().map_or(0, |root| root.granted_bytes())
    }

    fn eviction_priority(&self) -> u8 {
        priorities::GRAPH_STORAGE
    }

    fn region(&self) -> MemoryRegion {
        MemoryRegion::GraphStorage
    }

    fn evict(&self, _target_bytes: usize) -> usize {
        0
    }

    fn current_tier(&self) -> grafeo_common::memory::StorageTier {
        if self.memory_usage() == 0 {
            grafeo_common::memory::StorageTier::Uninitialized
        } else {
            grafeo_common::memory::StorageTier::InMemory
        }
    }
}

/// Registers the row-group store's memory consumers with `buffer_manager`:
/// the store itself, its vector indexes, spilling their columns into
/// `vector_cache` (`None`: no spill), and its text indexes.
pub(super) fn register_row_group_consumers(
    buffer_manager: &Arc<BufferManager>,
    root: &Arc<RowGroupStore>,
    #[cfg(all(feature = "wal", feature = "vector-index"))] vector_cache: Option<
        Arc<SpillDirectory>,
    >,
    #[cfg(all(feature = "wal", feature = "vector-index"))] spill_account: Option<Arc<SpillAccount>>,
    #[cfg(all(feature = "wal", feature = "vector-index"))] horizon: Option<CollectHorizon>,
) {
    buffer_manager.register_consumer(Arc::new(RowGroupStoreConsumer::new(root)));
    root.unregister_on_drop(buffer_manager, "section:LpgStore");
    #[cfg(all(feature = "wal", feature = "vector-index"))]
    buffer_manager.register_consumer(Arc::new(
        RowGroupVectorConsumer::new(
            root,
            vector_cache.map(|directory| ColumnSpillFiles::new(directory, spill_account)),
        )
        .collecting_at(horizon),
    ));
    #[cfg(all(feature = "vector-index", not(feature = "wal")))]
    buffer_manager.register_consumer(Arc::new(RowGroupVectorConsumer::new(root)));
    #[cfg(feature = "text-index")]
    buffer_manager.register_consumer(Arc::new(RowGroupTextConsumer::new(root)));
}

#[cfg(all(test, feature = "wal"))]
mod tests;
