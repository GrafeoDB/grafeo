//! Adapts storage sections into [`MemoryConsumer`]s for BufferManager integration.
//!
//! Each section (LPG, RDF, Vector, Text, Catalog) is registered with the
//! [`BufferManager`] so that memory tracking and pressure awareness include
//! section memory. This enables accurate `memory_usage()` reporting and
//! lays the groundwork for automatic spilling when tiered storage is added.

use std::path::PathBuf;
use std::sync::Arc;
#[cfg(any(
    all(
        feature = "lpg",
        feature = "vector-index",
        feature = "mmap",
        not(feature = "temporal")
    ),
    all(feature = "lpg", feature = "text-index"),
    // OverlayConsumer (lpg + compact-store, no mmap requirement) and
    // CompactStoreConsumer (lpg + compact-store + mmap) both hold a Weak
    // back to the layered store. The broader gate covers both.
    all(feature = "compact-store", feature = "lpg")
))]
use std::sync::Weak;

use grafeo_common::memory::buffer::{MemoryConsumer, MemoryRegion, SpillError, priorities};
use grafeo_common::storage::Section;
#[cfg(all(
    feature = "lpg",
    feature = "vector-index",
    feature = "mmap",
    not(feature = "temporal")
))]
use grafeo_common::types::{PropertyKey, Value};

/// Wraps a [`Section`] as a [`MemoryConsumer`] for the BufferManager.
///
/// Data sections (Catalog, LPG, RDF) use [`GRAPH_STORAGE`](priorities::GRAPH_STORAGE)
/// priority (evict last). Index sections (Vector, Text, RdfRing, PropertyIndex)
/// use [`INDEX_BUFFERS`](priorities::INDEX_BUFFERS) priority (evict before data).
///
/// Currently, `evict()` returns 0 because sections cannot release memory
/// without a full checkpoint + mmap cycle. The [`can_spill`](MemoryConsumer::can_spill)
/// method returns `true` for mmap-able index sections, signaling that future
/// tiered storage support will enable actual spilling.
///
/// The consumer is generic over the section's type: a `dyn Section` keeps
/// every method of the trait linked through its vtable, so a build that never
/// serializes a section (no `wal`, such as the WASM `edge` profile) would
/// carry the section's whole encoder and decoder.
pub struct SectionConsumer<S: Section + ?Sized = dyn Section> {
    name: String,
    section: Arc<S>,
    priority: u8,
    region: MemoryRegion,
    mmap_able: bool,
    /// Directory where this consumer writes spill files. `None` disables spilling.
    spill_path: Option<PathBuf>,
    /// Counter for unique spill file names within `spill_path`.
    #[cfg(feature = "wal")]
    file_counter: std::sync::atomic::AtomicUsize,
    /// `true` after a successful `spill_to_dir`, cleared on reload. Drives
    /// `current_tier()` so introspection reports the actual state of
    /// sections that opted into the `swap_to_mmap` path.
    is_spilled: std::sync::atomic::AtomicBool,
}

impl<S: Section + ?Sized> SectionConsumer<S> {
    /// Creates a consumer for the given section without spill support.
    ///
    /// Priority and region are assigned based on the section type:
    /// - Data sections (types 1-9): `GRAPH_STORAGE` priority, `GraphStorage` region
    /// - Index sections (types 10+): `INDEX_BUFFERS` priority, `IndexBuffers` region
    ///
    /// Calling `spill()` on a consumer constructed via `new` returns
    /// [`SpillError::NoSpillDirectory`]. Use [`with_spill`](Self::with_spill)
    /// to enable disk-backed eviction.
    pub fn new(section: Arc<S>) -> Self {
        Self::build(section, None)
    }

    /// Creates a consumer that spills the section's serialized bytes to a
    /// file under `spill_path` when memory pressure triggers eviction.
    ///
    /// On `spill()` the section is serialized, the bytes are written to
    /// `<spill_path>/<SectionType>_<n>.spill`, the file is mmapped, and
    /// the resulting [`PageFetcher`](grafeo_common::storage::PageFetcher)
    /// is handed to [`Section::swap_to_mmap`] for the section to consume.
    // Only consumed by the `ring-index` registration path today; other
    // section consumer types use specialized constructors (CompactStore,
    // VectorIndex, TextIndex). Allow dead_code under feature combinations
    // that don't include ring-index.
    #[cfg_attr(not(feature = "ring-index"), allow(dead_code))]
    pub fn with_spill(section: Arc<S>, spill_path: PathBuf) -> Self {
        Self::build(section, Some(spill_path))
    }

    fn build(section: Arc<S>, spill_path: Option<PathBuf>) -> Self {
        let section_type = section.section_type();
        let is_data = section_type.is_data_section();
        let flags = section_type.default_flags();

        Self {
            name: format!("section:{section_type:?}"),
            section,
            priority: if is_data {
                priorities::GRAPH_STORAGE
            } else {
                priorities::INDEX_BUFFERS
            },
            region: if is_data {
                MemoryRegion::GraphStorage
            } else {
                MemoryRegion::IndexBuffers
            },
            mmap_able: flags.mmap_able,
            spill_path,
            #[cfg(feature = "wal")]
            file_counter: std::sync::atomic::AtomicUsize::new(0),
            is_spilled: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Internal: perform the spill once preconditions have been checked.
    ///
    /// Behind the `wal` feature this serializes the section, writes a
    /// standalone spill file, mmaps it, and hands a fetcher to the
    /// section via [`Section::swap_to_mmap`]. Without `wal`, returns
    /// [`SpillError::NotSupported`] (no I/O dependencies available).
    #[cfg(feature = "wal")]
    fn spill_to_dir(&self, spill_dir: &std::path::Path) -> Result<usize, SpillError> {
        use grafeo_common::storage::PageFetcher;
        use grafeo_storage::container::{MmapPageFetcher, write_and_mmap_spill_file};

        let before = self.section.memory_usage();
        let bytes = self
            .section
            .serialize()
            .map_err(|e| SpillError::IoError(e.to_string()))?;

        let id = self
            .file_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let filename = format!("{:?}_{id}.spill", self.section.section_type());
        let path = spill_dir.join(filename);

        let mmap_section = write_and_mmap_spill_file(&path, &bytes, self.section.section_type())
            .map_err(|e| SpillError::IoError(e.to_string()))?;

        let fetcher: Arc<dyn PageFetcher> = Arc::new(MmapPageFetcher::new(Arc::new(mmap_section)));
        if let Err(e) = self.section.swap_to_mmap(fetcher) {
            // Section refused the swap. Best-effort cleanup of the spill
            // file so we don't leak it on a failed eviction. Errors here
            // are non-fatal: the file lives in spill_dir which is
            // user-managed.
            let _ = std::fs::remove_file(&path);
            return Err(e);
        }

        // Mark spilled so `current_tier()` reports OnDisk for
        // introspection, even when the section's `memory_usage()`
        // remains nonzero (the v2 Bytes-backed ring still occupies
        // heap, but its bulk data is paged from the spill mmap).
        self.is_spilled
            .store(true, std::sync::atomic::Ordering::Release);

        let after = self.section.memory_usage();
        Ok(before.saturating_sub(after))
    }

    #[cfg(not(feature = "wal"))]
    fn spill_to_dir(&self, _spill_dir: &std::path::Path) -> Result<usize, SpillError> {
        Err(SpillError::NotSupported)
    }
}

impl<S: Section + ?Sized> MemoryConsumer for SectionConsumer<S> {
    fn name(&self) -> &str {
        &self.name
    }

    fn memory_usage(&self) -> usize {
        self.section.memory_usage()
    }

    fn eviction_priority(&self) -> u8 {
        self.priority
    }

    fn region(&self) -> MemoryRegion {
        self.region
    }

    fn evict(&self, _target_bytes: usize) -> usize {
        // Sections cannot evict in-place. Freeing section memory requires
        // a checkpoint (serialize + write to container) followed by mmap.
        // The engine handles this at a higher level when pressure is detected.
        0
    }

    fn can_spill(&self) -> bool {
        // Index sections with mmap support can be spilled to the container
        // and served via memory-mapped I/O. Data sections require full
        // deserialization and cannot be mmap'd (yet).
        self.mmap_able
    }

    fn spill(&self, _target_bytes: usize) -> Result<usize, SpillError> {
        if !self.mmap_able {
            return Err(SpillError::NotSupported);
        }
        let spill_dir = self
            .spill_path
            .as_ref()
            .ok_or(SpillError::NoSpillDirectory)?;
        self.spill_to_dir(spill_dir)
    }

    fn reload(&self) -> Result<(), SpillError> {
        self.section.reload_to_ram()?;
        self.is_spilled
            .store(false, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    fn current_tier(&self) -> grafeo_common::memory::StorageTier {
        use grafeo_common::memory::StorageTier;
        if self.is_spilled.load(std::sync::atomic::Ordering::Acquire) {
            StorageTier::OnDisk
        } else if self.section.memory_usage() == 0 {
            StorageTier::Uninitialized
        } else {
            StorageTier::InMemory
        }
    }
}

/// Dynamic memory consumer for vector indexes.
///
/// Holds a `Weak<LpgStore>` and re-queries the live index map on each
/// `memory_usage()` call. On `spill()`, each embedding property column of a
/// vector index moves its vectors into a cache file it reads through
/// ([`VectorSpillFile`](super::vector_spill::VectorSpillFile)): the values
/// stay part of the column (queries, checkpoints and copies see them, search
/// reads them in place), only off the heap. `reload()` moves them back.
#[cfg(all(
    feature = "lpg",
    feature = "vector-index",
    feature = "mmap",
    not(feature = "temporal")
))]
pub struct VectorIndexConsumer {
    store: Weak<grafeo_core::graph::lpg::LpgStore>,
    /// Where the cache files go. `None` disables spilling.
    cache: Option<Arc<super::spill_directory::SpillDirectory>>,
}

#[cfg(all(
    feature = "lpg",
    feature = "vector-index",
    feature = "mmap",
    not(feature = "temporal")
))]
impl VectorIndexConsumer {
    /// Creates a consumer that dynamically queries the store for current
    /// vector indexes and spills their columns into `cache`.
    pub(crate) fn new(
        store: &Arc<grafeo_core::graph::lpg::LpgStore>,
        cache: Option<Arc<super::spill_directory::SpillDirectory>>,
    ) -> Self {
        Self {
            store: Arc::downgrade(store),
            cache,
        }
    }

    /// The embedding properties of the vector indexes, each once, in order
    /// (indexes on two labels can share one property column).
    fn indexed_properties(store: &grafeo_core::graph::lpg::LpgStore) -> Vec<PropertyKey> {
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

    /// The indexed properties whose columns are spilled.
    fn spilled_properties(store: &grafeo_core::graph::lpg::LpgStore) -> Vec<PropertyKey> {
        let spilled = store.spilled_node_property_columns();
        Self::indexed_properties(store)
            .into_iter()
            .filter(|property| spilled.contains(property))
            .collect()
    }

    /// Spills one embedding column: its vectors go to a cache file the
    /// column reads through. Returns the bytes of vectors moved off the heap.
    fn spill_column(
        &self,
        store: &grafeo_core::graph::lpg::LpgStore,
        cache: &Arc<super::spill_directory::SpillDirectory>,
        property: &PropertyKey,
    ) -> Result<usize, SpillError> {
        // Snapshot under the read lock; the file is written without it, and
        // what changes meanwhile wins when the backing is installed.
        let snapshot = store
            .node_property_column_entries(property)
            .map_err(|e| SpillError::IoError(e.to_string()))?;
        let vectors: Vec<(grafeo_common::types::NodeId, Arc<[f32]>)> = snapshot
            .iter()
            .filter_map(|(id, value)| match value {
                Value::Vector(vector) => Some((*id, Arc::clone(vector))),
                _ => None,
            })
            .collect();
        if vectors.is_empty() {
            return Ok(0);
        }
        let file = super::vector_spill::VectorSpillFile::write(cache, &vectors)
            .map_err(|e| SpillError::IoError(e.to_string()))?;
        let bytes = vectors
            .iter()
            .map(|(_, vector)| vector.len() * 4 + std::mem::size_of::<Arc<[f32]>>())
            .sum();
        // Refused when another spill got there first; the file then goes
        // with the refused backing.
        if !store.spill_node_property_column(property, Arc::new(file), &snapshot) {
            return Ok(0);
        }
        // An index dropped since the column was chosen reloaded nothing (the
        // column was not spilled yet), and no index, so no reload, would
        // bring the column back: it comes back now. The drop removes the
        // index before it reloads, and this checks after the install, so one
        // of the two sees the other.
        if !Self::indexed_properties(store).contains(property) {
            store
                .reload_node_property_column(property)
                .map_err(|e| SpillError::IoError(e.to_string()))?;
            return Ok(0);
        }
        Ok(bytes)
    }
}

#[cfg(all(
    feature = "lpg",
    feature = "vector-index",
    feature = "mmap",
    not(feature = "temporal")
))]
impl MemoryConsumer for VectorIndexConsumer {
    fn name(&self) -> &str {
        "section:VectorStore"
    }

    fn memory_usage(&self) -> usize {
        self.store.upgrade().map_or(0, |store| {
            store
                .vector_index_entries()
                .iter()
                .map(|(_, idx)| idx.heap_memory_bytes())
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

    fn can_spill(&self) -> bool {
        // Not after `close()` (see `SpillDirectory::close_for_writes`).
        self.cache.as_ref().is_some_and(|cache| !cache.is_closed())
    }

    fn current_tier(&self) -> grafeo_common::memory::StorageTier {
        use grafeo_common::memory::StorageTier;
        // A spilled embedding column means at least one index reads its
        // vectors from disk: OnDisk. Otherwise InMemory if any index has
        // data, else Uninitialized.
        if self
            .store
            .upgrade()
            .is_some_and(|store| !Self::spilled_properties(&store).is_empty())
        {
            return StorageTier::OnDisk;
        }
        if self.memory_usage() == 0 {
            StorageTier::Uninitialized
        } else {
            StorageTier::InMemory
        }
    }

    fn spill(&self, _target_bytes: usize) -> Result<usize, SpillError> {
        let cache = self.cache.as_ref().ok_or(SpillError::NoSpillDirectory)?;
        // A closed database spills no more; a reload stays allowed.
        if cache.is_closed() {
            return Ok(0);
        }
        let store = self
            .store
            .upgrade()
            .ok_or(SpillError::IoError("store dropped".to_string()))?;

        let spilled = store.spilled_node_property_columns();
        let mut total_freed = 0;
        for property in Self::indexed_properties(&store) {
            if spilled.contains(&property) {
                continue;
            }
            match self.spill_column(&store, cache, &property) {
                Ok(freed) => total_freed += freed,
                Err(e) => {
                    // Continue: the columns spilled so far count, and this
                    // one keeps its values on the heap.
                    grafeo_common::grafeo_warn!(
                        "failed to spill the vector column {}: {e}",
                        property.as_str()
                    );
                }
            }
        }

        Ok(total_freed)
    }

    fn reload(&self) -> Result<(), SpillError> {
        let store = self
            .store
            .upgrade()
            .ok_or(SpillError::IoError("store dropped".to_string()))?;
        // Each reload lets go of the column's file, which is then deleted. A
        // column whose file cannot be read stays spilled (nothing is lost);
        // the others still reload.
        let mut first_error = None;
        for property in Self::spilled_properties(&store) {
            if let Err(e) = store.reload_node_property_column(&property) {
                grafeo_common::grafeo_warn!(
                    "the vector column {} stays spilled: {e}",
                    property.as_str()
                );
                first_error.get_or_insert(SpillError::IoError(e.to_string()));
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}

/// Dynamic memory consumer for text indexes.
///
/// Same rationale as [`VectorIndexConsumer`]: avoids holding stale `Arc` refs
/// to indexes that may have been dropped, and automatically picks up new ones.
#[cfg(all(feature = "lpg", feature = "text-index"))]
pub struct TextIndexConsumer {
    store: Weak<grafeo_core::graph::lpg::LpgStore>,
}

#[cfg(all(feature = "lpg", feature = "text-index"))]
impl TextIndexConsumer {
    /// Creates a consumer that dynamically queries the store for current text indexes.
    pub fn new(store: &Arc<grafeo_core::graph::lpg::LpgStore>) -> Self {
        Self {
            store: Arc::downgrade(store),
        }
    }
}

#[cfg(all(feature = "lpg", feature = "text-index"))]
impl MemoryConsumer for TextIndexConsumer {
    fn name(&self) -> &str {
        "section:TextIndex"
    }

    fn memory_usage(&self) -> usize {
        self.store.upgrade().map_or(0, |store| {
            store
                .text_index_entries()
                .iter()
                .map(|(_, idx)| idx.read().heap_memory_bytes())
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

    fn can_spill(&self) -> bool {
        true
    }

    fn spill(&self, _target_bytes: usize) -> Result<usize, SpillError> {
        Err(SpillError::NotSupported)
    }

    fn current_tier(&self) -> grafeo_common::memory::StorageTier {
        // Text indexes never actually move to disk today: `spill` returns
        // `NotSupported`, so the consumer is always in-memory while alive.
        if self.memory_usage() == 0 {
            grafeo_common::memory::StorageTier::Uninitialized
        } else {
            grafeo_common::memory::StorageTier::InMemory
        }
    }
}

/// Memory consumer for the CompactStore base under a `LayeredStore`.
///
/// Delegates spill/reload to a [`CompactStoreTiered`] wrapper and atomically
/// swaps the `LayeredStore`'s base `Arc<CompactStore>` when tier state
/// changes, so the old in-memory allocation actually drops after a spill.
///
/// Priority is [`GRAPH_STORAGE`](priorities::GRAPH_STORAGE) (evict-last):
/// the compact base is persistent data, spilling it is the last resort
/// before query failure.
#[cfg(all(feature = "compact-store", feature = "mmap", feature = "lpg"))]
pub struct CompactStoreConsumer {
    tiered: Weak<super::compact_tiered::CompactStoreTiered>,
    layered: Weak<grafeo_core::graph::compact::layered::LayeredStore>,
    spill_path: Option<PathBuf>,
}

#[cfg(all(feature = "compact-store", feature = "mmap", feature = "lpg"))]
impl CompactStoreConsumer {
    /// Creates a consumer that spills the base to `<spill_path>/compact_base.grafeo`.
    ///
    /// `spill_path = None` disables spilling.
    pub fn new(
        tiered: &Arc<super::compact_tiered::CompactStoreTiered>,
        layered: &Arc<grafeo_core::graph::compact::layered::LayeredStore>,
        spill_path: Option<PathBuf>,
    ) -> Self {
        Self {
            tiered: Arc::downgrade(tiered),
            layered: Arc::downgrade(layered),
            spill_path,
        }
    }

    fn spill_file(&self) -> Option<PathBuf> {
        self.spill_path
            .as_ref()
            .map(|dir| dir.join("compact_base.grafeo"))
    }
}

#[cfg(all(feature = "compact-store", feature = "mmap", feature = "lpg"))]
impl MemoryConsumer for CompactStoreConsumer {
    fn name(&self) -> &str {
        "section:CompactStore"
    }

    fn memory_usage(&self) -> usize {
        // When OnDisk, the heap copy of CompactStore is still alive (we
        // deserialized from mmap eagerly). Report its heap bytes in both
        // states; the OS page cache that backs mmap lives outside the heap.
        self.tiered.upgrade().map_or(0, |t| t.memory_bytes())
    }

    fn eviction_priority(&self) -> u8 {
        priorities::GRAPH_STORAGE
    }

    fn region(&self) -> MemoryRegion {
        MemoryRegion::GraphStorage
    }

    fn evict(&self, _target_bytes: usize) -> usize {
        // CompactStore cannot evict in-place: use spill() to tier to disk.
        0
    }

    fn can_spill(&self) -> bool {
        let Some(tiered) = self.tiered.upgrade() else {
            return false;
        };
        self.spill_path.is_some() && !tiered.is_on_disk()
    }

    fn current_tier(&self) -> grafeo_common::memory::StorageTier {
        use grafeo_common::memory::StorageTier;
        let Some(tiered) = self.tiered.upgrade() else {
            return StorageTier::Uninitialized;
        };
        if tiered.is_on_disk() {
            StorageTier::OnDisk
        } else if self.memory_usage() == 0 {
            StorageTier::Uninitialized
        } else {
            StorageTier::InMemory
        }
    }

    fn spill(&self, _target_bytes: usize) -> Result<usize, SpillError> {
        let tiered = self
            .tiered
            .upgrade()
            .ok_or_else(|| SpillError::IoError("compact-store tiered dropped".to_string()))?;

        if tiered.is_on_disk() {
            return Ok(0);
        }

        let path = self.spill_file().ok_or(SpillError::NoSpillDirectory)?;

        let before = tiered.memory_bytes();
        tiered
            .persist_to_mmap(&path)
            .map_err(|e| SpillError::IoError(e.to_string()))?;

        // Publish the fresh (mmap-backed) base to the LayeredStore so readers
        // switch over and the old allocation can drop. If the LayeredStore has
        // been reconstructed (e.g. recompact() between registration and this
        // call), the weak ref returns None: the new LayeredStore already owns
        // a matching base from the new tiered wrapper, so there's nothing to
        // swap here.
        if let Some(layered) = self.layered.upgrade() {
            layered.swap_base(tiered.store());
        }

        let after = tiered.memory_bytes();
        Ok(before.saturating_sub(after))
    }

    fn reload(&self) -> Result<(), SpillError> {
        let tiered = self
            .tiered
            .upgrade()
            .ok_or_else(|| SpillError::IoError("compact-store tiered dropped".to_string()))?;

        if !tiered.is_on_disk() {
            return Ok(());
        }

        tiered
            .reload_to_ram()
            .map_err(|e| SpillError::IoError(e.to_string()))?;
        if let Some(layered) = self.layered.upgrade() {
            layered.swap_base(tiered.store());
        }
        Ok(())
    }
}

// ── Phase 5c: OverlayConsumer ─────────────────────────────────────────
//
// Tracks the LpgStore overlay portion of a `LayeredStore`. When memory
// pressure rises and the consumer is asked to spill, it calls
// `LayeredStore::merge_overlay_in_place()` which rebuilds the base from
// the combined view and clears the overlay, freeing all overlay heap.
//
// The new base is in-memory; if total memory pressure persists, the
// `CompactStoreConsumer` will spill that base to mmap on its own. Two
// independent consumers, one BufferManager — chains naturally.

/// Tracks the mutable overlay (LpgStore) of a `LayeredStore`.
///
/// Priority is [`GRAPH_STORAGE`](priorities::GRAPH_STORAGE) (evict-last):
/// the overlay holds unflushed mutations and merging it requires
/// rebuilding the base, so this is the last-resort spill before query
/// failure under sustained mutation pressure.
///
/// The merge runs with commits held off and the store frozen (see
/// [`TransactionManager::hold_commits`](crate::transaction::TransactionManager)):
/// the base has no versions, so a commit in the middle of being written, or
/// one that did not complete, would become visible in it, and a write of an
/// open transaction must not change the overlay while it is merged. It does
/// not wait: while a commit or such a write is in progress (possibly on the
/// thread that asks for memory), nothing is merged, and after a commit that
/// did not complete, the spill fails.
#[cfg(all(feature = "compact-store", feature = "lpg"))]
pub struct OverlayConsumer {
    layered: Weak<grafeo_core::graph::compact::layered::LayeredStore>,
    transaction_manager: Arc<crate::transaction::TransactionManager>,
}

#[cfg(all(feature = "compact-store", feature = "lpg"))]
impl OverlayConsumer {
    /// Creates a consumer that monitors the overlay of `layered`, whose
    /// commits `transaction_manager` runs.
    pub fn new(
        layered: &Arc<grafeo_core::graph::compact::layered::LayeredStore>,
        transaction_manager: &Arc<crate::transaction::TransactionManager>,
    ) -> Self {
        Self {
            layered: Arc::downgrade(layered),
            transaction_manager: Arc::clone(transaction_manager),
        }
    }
}

#[cfg(all(feature = "compact-store", feature = "lpg"))]
impl MemoryConsumer for OverlayConsumer {
    fn name(&self) -> &str {
        "overlay:LpgStore"
    }

    fn memory_usage(&self) -> usize {
        self.layered
            .upgrade()
            .map_or(0, |layered| layered.overlay_memory_bytes())
    }

    fn eviction_priority(&self) -> u8 {
        priorities::GRAPH_STORAGE
    }

    fn region(&self) -> MemoryRegion {
        MemoryRegion::GraphStorage
    }

    fn evict(&self, _target_bytes: usize) -> usize {
        // Cannot evict in place; spill via merge.
        0
    }

    fn can_spill(&self) -> bool {
        let Some(layered) = self.layered.upgrade() else {
            return false;
        };
        // Only worth spilling if the overlay actually has mutations.
        layered.overlay_mutation_count() > 0
    }

    fn spill(&self, _target_bytes: usize) -> Result<usize, SpillError> {
        let Some(layered) = self.layered.upgrade() else {
            return Err(SpillError::IoError("layered store dropped".to_string()));
        };

        if layered.overlay_mutation_count() == 0 {
            return Ok(0);
        }
        let Some(_commits) = self
            .transaction_manager
            .try_hold_commits()
            .map_err(|error| SpillError::IoError(error.to_string()))?
        else {
            // A commit or a write of an open transaction is in progress:
            // merge later.
            return Ok(0);
        };

        let before = layered.overlay_memory_bytes();
        layered
            .merge_overlay_in_place()
            .map_err(SpillError::IoError)?;
        let after = layered.overlay_memory_bytes();
        Ok(before.saturating_sub(after))
    }

    fn current_tier(&self) -> grafeo_common::memory::StorageTier {
        // Overlay "spills" by merging into the base store; it never moves
        // to disk on its own (the base may, separately).
        if self.memory_usage() == 0 {
            grafeo_common::memory::StorageTier::Uninitialized
        } else {
            grafeo_common::memory::StorageTier::InMemory
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::storage::page_fetcher::PageFetcher;
    use grafeo_common::storage::section::{
        SectionSink, SectionSource, SectionType, read_raw, write_raw,
    };
    use grafeo_common::utils::error::Result;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    /// Test section that records `swap_to_mmap` and `reload_to_ram`.
    ///
    /// Mimics the eager-deserialize spill model: after `swap_to_mmap`,
    /// `memory_usage()` drops to zero (representing the section having
    /// released its heap copy in favour of paging from the mmap), and
    /// `reload_to_ram` restores it.
    struct SwappableSection {
        section_type: SectionType,
        serialize_size: usize,
        in_memory: AtomicBool,
        swap_calls: AtomicUsize,
        reload_calls: AtomicUsize,
        captured_bytes: parking_lot::Mutex<Option<Vec<u8>>>,
    }

    impl SwappableSection {
        fn new(section_type: SectionType, serialize_size: usize) -> Self {
            Self {
                section_type,
                serialize_size,
                in_memory: AtomicBool::new(true),
                swap_calls: AtomicUsize::new(0),
                reload_calls: AtomicUsize::new(0),
                captured_bytes: parking_lot::Mutex::new(None),
            }
        }
        fn swap_count(&self) -> usize {
            self.swap_calls.load(Ordering::Relaxed)
        }
        fn reload_count(&self) -> usize {
            self.reload_calls.load(Ordering::Relaxed)
        }
    }

    impl Section for SwappableSection {
        fn section_type(&self) -> SectionType {
            self.section_type
        }
        fn serialize(&self) -> Result<Vec<u8>> {
            // Deterministic non-zero pattern so we can assert the spill
            // file actually contains what serialize produced.
            Ok(vec![0xAB; self.serialize_size])
        }
        fn deserialize(&mut self, _data: &[u8]) -> Result<()> {
            Ok(())
        }
        fn write_to(&self, sink: &mut dyn SectionSink) -> Result<()> {
            write_raw(self, sink)
        }
        fn read_from(&mut self, source: &dyn SectionSource) -> Result<()> {
            read_raw(self, source)
        }
        fn is_dirty(&self) -> bool {
            false
        }
        fn mark_clean(&self) {}
        fn memory_usage(&self) -> usize {
            if self.in_memory.load(Ordering::Relaxed) {
                self.serialize_size
            } else {
                0
            }
        }
        fn swap_to_mmap(
            &self,
            fetcher: Arc<dyn PageFetcher>,
        ) -> std::result::Result<(), SpillError> {
            let bytes = fetcher
                .fetch(0, fetcher.len())
                .map_err(|e| SpillError::IoError(e.to_string()))?
                .to_vec();
            *self.captured_bytes.lock() = Some(bytes);
            self.swap_calls.fetch_add(1, Ordering::Relaxed);
            self.in_memory.store(false, Ordering::Relaxed);
            Ok(())
        }
        fn reload_to_ram(&self) -> std::result::Result<(), SpillError> {
            self.reload_calls.fetch_add(1, Ordering::Relaxed);
            self.in_memory.store(true, Ordering::Relaxed);
            Ok(())
        }
    }

    // Spilling writes a file, which needs the `wal` feature's I/O.
    #[cfg(feature = "wal")]
    /// A column whose vector index is dropped while the column spills (the
    /// spill chose it, then the drop's reload found it not spilled yet) does
    /// not stay spilled: no index, and so no reload, would bring it back
    /// (#594).
    #[cfg(all(
        feature = "lpg",
        feature = "vector-index",
        feature = "mmap",
        not(feature = "temporal")
    ))]
    #[test]
    fn a_column_unindexed_while_it_spills_comes_back() {
        use grafeo_common::types::Value;
        use grafeo_core::graph::lpg::LpgStore;
        use grafeo_core::index::vector::{DistanceMetric, HnswConfig, HnswIndex, VectorIndexKind};

        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(LpgStore::new().unwrap());
        let alix = store.create_node_with_props(
            &["Item"],
            [("embedding", Value::Vector(vec![3.0, 19.0].into()))],
        );
        store.add_vector_index(
            "Item",
            "embedding",
            Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
                2,
                DistanceMetric::Euclidean,
            )))),
        );
        let cache =
            super::super::spill_directory::SpillLayout::for_open(Some(dir.path()), None, false)
                .vector_cache
                .unwrap();
        let consumer = VectorIndexConsumer::new(&store, Some(Arc::clone(&cache)));
        let property = PropertyKey::new("embedding");

        assert!(store.remove_vector_index("Item", "embedding"));
        consumer.spill_column(&store, &cache, &property).unwrap();
        assert_eq!(
            store.spilled_node_property_columns(),
            Vec::<PropertyKey>::new(),
            "a spilled column no index reads"
        );
        assert_eq!(
            store.get_node_property(alix, &property),
            Some(Value::Vector(vec![3.0, 19.0].into()))
        );
    }

    #[test]
    fn alix_spill_writes_serialized_bytes_through_swap_to_mmap() {
        let dir = tempfile::tempdir().expect("tempdir");
        let section = Arc::new(SwappableSection::new(SectionType::PropertyIndex, 4096));
        let consumer = SectionConsumer::with_spill(
            Arc::clone(&section) as Arc<dyn Section>,
            dir.path().to_path_buf(),
        );

        let freed = consumer.spill(0).expect("spill should succeed");
        assert_eq!(freed, 4096, "freed bytes equal section memory_usage");
        assert_eq!(section.swap_count(), 1, "swap_to_mmap called once");

        let captured = section
            .captured_bytes
            .lock()
            .clone()
            .expect("bytes captured");
        assert_eq!(
            captured,
            vec![0xAB; 4096],
            "mmap bytes equal serialize output"
        );
    }

    #[test]
    fn gus_spill_fails_with_no_spill_dir_when_path_missing() {
        let section = Arc::new(SwappableSection::new(SectionType::PropertyIndex, 1024));
        // SectionConsumer::new() = no spill_path
        let consumer = SectionConsumer::new(Arc::clone(&section) as Arc<dyn Section>);

        match consumer.spill(0) {
            Err(SpillError::NoSpillDirectory) => {}
            other => panic!("expected NoSpillDirectory, got {other:?}"),
        }
        assert_eq!(section.swap_count(), 0, "swap not called when path missing");
    }

    #[test]
    fn vincent_spill_returns_not_supported_when_section_does_not_override_swap() {
        let dir = tempfile::tempdir().expect("tempdir");
        // FakeSection is mmap-able by type (VectorStore) but uses the
        // default `swap_to_mmap`, which returns `NotSupported`.
        let section = Arc::new(FakeSection::new(SectionType::VectorStore, 1024));
        let consumer = SectionConsumer::with_spill(
            Arc::clone(&section) as Arc<dyn Section>,
            dir.path().to_path_buf(),
        );

        match consumer.spill(0) {
            Err(SpillError::NotSupported) => {}
            other => panic!("expected NotSupported from default swap_to_mmap, got {other:?}"),
        }
    }

    // Spilling writes a file, which needs the `wal` feature's I/O.
    #[cfg(feature = "wal")]
    #[test]
    fn jules_reload_calls_section_reload_to_ram() {
        let dir = tempfile::tempdir().expect("tempdir");
        let section = Arc::new(SwappableSection::new(SectionType::PropertyIndex, 1024));
        let consumer = SectionConsumer::with_spill(
            Arc::clone(&section) as Arc<dyn Section>,
            dir.path().to_path_buf(),
        );

        consumer.spill(0).expect("spill ok");
        consumer.reload().expect("reload ok");

        assert_eq!(section.reload_count(), 1, "reload_to_ram called once");
    }

    #[test]
    fn mia_reload_without_spill_is_noop() {
        // Reload before any spill should not error and should still call
        // reload_to_ram (which is a no-op by default for InMemory tier).
        let section = Arc::new(SwappableSection::new(SectionType::PropertyIndex, 1024));
        let consumer = SectionConsumer::new(Arc::clone(&section) as Arc<dyn Section>);

        consumer.reload().expect("reload before spill ok");
        assert_eq!(
            section.reload_count(),
            1,
            "reload_to_ram called even when not on disk"
        );
    }

    /// Minimal Section implementation for testing.
    struct FakeSection {
        section_type: SectionType,
        usage: usize,
        dirty: AtomicBool,
    }

    impl FakeSection {
        fn new(section_type: SectionType, usage: usize) -> Self {
            Self {
                section_type,
                usage,
                dirty: AtomicBool::new(false),
            }
        }
    }

    impl Section for FakeSection {
        fn section_type(&self) -> SectionType {
            self.section_type
        }
        fn serialize(&self) -> Result<Vec<u8>> {
            Ok(vec![0; self.usage])
        }
        fn deserialize(&mut self, _data: &[u8]) -> Result<()> {
            Ok(())
        }
        fn write_to(&self, sink: &mut dyn SectionSink) -> Result<()> {
            write_raw(self, sink)
        }
        fn read_from(&mut self, source: &dyn SectionSource) -> Result<()> {
            read_raw(self, source)
        }
        fn is_dirty(&self) -> bool {
            self.dirty.load(Ordering::Relaxed)
        }
        fn mark_clean(&self) {
            self.dirty.store(false, Ordering::Relaxed);
        }
        fn memory_usage(&self) -> usize {
            self.usage
        }
    }

    #[test]
    fn data_section_consumer_properties() {
        let section = Arc::new(FakeSection::new(SectionType::LpgStore, 1024));
        let consumer = SectionConsumer::new(section);

        assert_eq!(consumer.name(), "section:LpgStore");
        assert_eq!(consumer.memory_usage(), 1024);
        assert_eq!(consumer.eviction_priority(), priorities::GRAPH_STORAGE);
        assert_eq!(consumer.region(), MemoryRegion::GraphStorage);
        assert!(!consumer.can_spill());
    }

    #[test]
    fn index_section_consumer_properties() {
        let section = Arc::new(FakeSection::new(SectionType::VectorStore, 4096));
        let consumer = SectionConsumer::new(section);

        assert_eq!(consumer.name(), "section:VectorStore");
        assert_eq!(consumer.memory_usage(), 4096);
        assert_eq!(consumer.eviction_priority(), priorities::INDEX_BUFFERS);
        assert_eq!(consumer.region(), MemoryRegion::IndexBuffers);
        assert!(consumer.can_spill());
    }

    #[test]
    fn evict_returns_zero() {
        let section = Arc::new(FakeSection::new(SectionType::TextIndex, 8192));
        let consumer = SectionConsumer::new(section);

        // Sections can't evict in-place
        assert_eq!(consumer.evict(4096), 0);
        // Memory is unchanged
        assert_eq!(consumer.memory_usage(), 8192);
    }

    #[test]
    fn spill_returns_not_supported() {
        let section = Arc::new(FakeSection::new(SectionType::VectorStore, 4096));
        let consumer = SectionConsumer::new(section);

        let result = consumer.spill(2048);
        assert!(result.is_err());
    }

    #[test]
    fn catalog_section_is_data() {
        let section = Arc::new(FakeSection::new(SectionType::Catalog, 256));
        let consumer = SectionConsumer::new(section);

        assert_eq!(consumer.eviction_priority(), priorities::GRAPH_STORAGE);
        assert!(!consumer.can_spill());
    }

    #[test]
    fn rdf_ring_section_is_index() {
        let section = Arc::new(FakeSection::new(SectionType::RdfRing, 2048));
        let consumer = SectionConsumer::new(section);

        assert_eq!(consumer.eviction_priority(), priorities::INDEX_BUFFERS);
        assert!(consumer.can_spill());
    }

    #[test]
    fn property_index_section_is_index() {
        let section = Arc::new(FakeSection::new(SectionType::PropertyIndex, 512));
        let consumer = SectionConsumer::new(section);

        assert_eq!(consumer.name(), "section:PropertyIndex");
        assert_eq!(consumer.eviction_priority(), priorities::INDEX_BUFFERS);
        assert_eq!(consumer.region(), MemoryRegion::IndexBuffers);
        assert!(consumer.can_spill());
    }

    #[test]
    fn rdf_store_section_is_data() {
        let section = Arc::new(FakeSection::new(SectionType::RdfStore, 1024));
        let consumer = SectionConsumer::new(section);

        assert_eq!(consumer.name(), "section:RdfStore");
        assert_eq!(consumer.eviction_priority(), priorities::GRAPH_STORAGE);
        assert_eq!(consumer.region(), MemoryRegion::GraphStorage);
        assert!(!consumer.can_spill(), "data sections cannot spill");
    }

    #[test]
    fn spill_non_mmap_section_returns_not_supported() {
        // LpgStore is a data section (mmap_able=false), spill should fail
        let section = Arc::new(FakeSection::new(SectionType::LpgStore, 4096));
        let consumer = SectionConsumer::new(section);

        assert!(!consumer.can_spill());
        let result = consumer.spill(2048);
        match result {
            Err(SpillError::NotSupported) => {}
            other => panic!("expected NotSupported, got {other:?}"),
        }
    }

    #[test]
    fn zero_memory_section() {
        let section = Arc::new(FakeSection::new(SectionType::Catalog, 0));
        let consumer = SectionConsumer::new(section);

        assert_eq!(consumer.memory_usage(), 0);
        assert_eq!(consumer.evict(1024), 0);
    }

    #[test]
    fn section_consumer_name_format() {
        // Verify all section types produce "section:<Type>" names
        for section_type in [
            SectionType::Catalog,
            SectionType::LpgStore,
            SectionType::RdfStore,
            SectionType::VectorStore,
            SectionType::TextIndex,
            SectionType::RdfRing,
            SectionType::PropertyIndex,
        ] {
            let section = Arc::new(FakeSection::new(section_type, 100));
            let consumer = SectionConsumer::new(section);
            assert!(
                consumer.name().starts_with("section:"),
                "name should start with 'section:' for {section_type:?}"
            );
        }
    }
}
