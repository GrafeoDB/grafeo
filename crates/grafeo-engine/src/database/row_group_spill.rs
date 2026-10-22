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
//! `RowGroupTextConsumer` reports the text indexes; the store's own memory
//! is reported by a [`SectionConsumer`](super::section_consumer::SectionConsumer)
//! of its section ([`register_row_group_consumers`]).

#[cfg(all(feature = "wal", feature = "vector-index"))]
use std::io;
#[cfg(all(feature = "wal", feature = "vector-index"))]
use std::path::Path;
#[cfg(all(feature = "wal", feature = "vector-index"))]
use std::path::PathBuf;
use std::sync::Arc;
#[cfg(any(feature = "vector-index", feature = "text-index"))]
use std::sync::Weak;
#[cfg(all(feature = "wal", feature = "vector-index"))]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(all(feature = "wal", feature = "vector-index"))]
use bytes::Bytes;
use grafeo_common::memory::buffer::BufferManager;
#[cfg(feature = "vector-index")]
use grafeo_common::memory::buffer::SpillError;
#[cfg(any(feature = "vector-index", feature = "text-index"))]
use grafeo_common::memory::buffer::{MemoryConsumer, MemoryRegion, priorities};
#[cfg(all(feature = "wal", feature = "vector-index"))]
use grafeo_common::storage::SectionType;
#[cfg(all(feature = "wal", feature = "vector-index"))]
use grafeo_common::types::PropertyKey;
#[cfg(all(feature = "wal", feature = "vector-index"))]
use grafeo_core::graph::rowgroup::ColumnSpill;
use grafeo_core::graph::rowgroup::RowGroupStore;
use grafeo_core::graph::rowgroup::section::RowGroupStoreSection;
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
}

#[cfg(all(feature = "wal", feature = "vector-index"))]
impl ColumnSpillFiles {
    /// Spill files in `directory`, created when the first one is written.
    pub(super) fn new(directory: Arc<SpillDirectory>) -> Self {
        Self { directory }
    }

    /// Spill files in a directory of their own at `path` (tests).
    #[cfg(test)]
    pub(super) fn in_dir(path: impl Into<PathBuf>) -> Self {
        Self::new(Arc::new(SpillDirectory::at(path.into())))
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
            })),
            Err(error) => {
                remove_written(&path);
                Err(io::Error::other(error.to_string()))
            }
        }
    }
}

/// Removes what a failed write left at `path`: the file, or the temporary
/// one the write renames into it.
#[cfg(all(feature = "wal", feature = "vector-index"))]
fn remove_written(path: &Path) {
    for leftover in [path.to_path_buf(), path.with_extension("spill.tmp")] {
        if leftover.exists() {
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
    #[cfg(all(feature = "wal", feature = "vector-index"))]
    fn indexed_properties(store: &RowGroupStore) -> Vec<PropertyKey> {
        let mut properties: Vec<PropertyKey> = store
            .vector_index_entries()
            .iter()
            .filter_map(|(key, _)| {
                key.split_once(':')
                    .map(|(_, property)| PropertyKey::new(property))
            })
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

    fn memory_usage(&self) -> usize {
        self.root.upgrade().map_or(0, |root| {
            graphs(&root)
                .iter()
                .flat_map(|graph| graph.vector_index_entries())
                .map(|(_, index)| index.heap_memory_bytes())
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
    fn spill(&self, _target_bytes: usize) -> Result<usize, SpillError> {
        let files = self.files.as_ref().ok_or(SpillError::NoSpillDirectory)?;
        let root = self.root()?;
        let mut freed = 0;
        if let Some(horizon) = &self.horizon {
            // The root's collect covers every named graph's.
            root.gc_versions(horizon());
        }
        for graph in graphs(&root) {
            let spilled = graph.spilled_node_property_columns();
            for property in Self::indexed_properties(&graph) {
                if spilled.contains(&property) {
                    continue;
                }
                match graph.spill_node_property_column(&property, files) {
                    Ok(report) => freed += report.freed_bytes,
                    // Continue: what spilled so far counts, and this column
                    // keeps what did not spill on the heap.
                    Err(error) => grafeo_common::grafeo_warn!(
                        "failed to spill the vector column {}: {error}",
                        property.as_str()
                    ),
                }
            }
        }
        Ok(freed)
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

/// The memory consumer of the row-group store's text indexes: their memory
/// (they do not spill).
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
                .map(|(_, index)| index.read().heap_memory_bytes())
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

/// Registers the row-group store's memory consumers with `buffer_manager`:
/// its section (the store's heap), its vector indexes, spilling their
/// columns into `vector_cache` (`None`: no spill), and its text indexes.
pub(super) fn register_row_group_consumers(
    buffer_manager: &BufferManager,
    root: &Arc<RowGroupStore>,
    #[cfg(all(feature = "wal", feature = "vector-index"))] vector_cache: Option<
        Arc<SpillDirectory>,
    >,
    #[cfg(all(feature = "wal", feature = "vector-index"))] horizon: Option<CollectHorizon>,
) {
    buffer_manager.register_consumer(Arc::new(super::section_consumer::SectionConsumer::new(
        Arc::new(RowGroupStoreSection::new(Arc::clone(root))),
    )));
    #[cfg(all(feature = "wal", feature = "vector-index"))]
    buffer_manager.register_consumer(Arc::new(
        RowGroupVectorConsumer::new(root, vector_cache.map(ColumnSpillFiles::new))
            .collecting_at(horizon),
    ));
    #[cfg(all(feature = "vector-index", not(feature = "wal")))]
    buffer_manager.register_consumer(Arc::new(RowGroupVectorConsumer::new(root)));
    #[cfg(feature = "text-index")]
    buffer_manager.register_consumer(Arc::new(RowGroupTextConsumer::new(root)));
}

#[cfg(all(test, feature = "wal"))]
mod tests;
