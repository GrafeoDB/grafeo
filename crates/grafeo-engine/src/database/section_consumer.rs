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
use std::sync::atomic::{AtomicUsize, Ordering};

use grafeo_common::memory::buffer::{MemoryConsumer, MemoryRegion, SpillError, priorities};
use grafeo_common::storage::Section;
#[cfg(all(
    feature = "lpg",
    feature = "vector-index",
    feature = "mmap",
    not(feature = "temporal")
))]
use grafeo_common::types::{NodeId, PropertyKey, Value};
#[cfg(all(
    feature = "lpg",
    feature = "vector-index",
    feature = "mmap",
    not(feature = "temporal")
))]
use grafeo_core::index::vector::VectorStorage;
#[cfg(all(
    feature = "lpg",
    feature = "vector-index",
    feature = "mmap",
    not(feature = "temporal")
))]
use parking_lot::RwLock;
#[cfg(all(
    feature = "lpg",
    feature = "vector-index",
    feature = "mmap",
    not(feature = "temporal")
))]
use std::collections::{HashMap, HashSet};

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
pub struct SectionConsumer {
    name: String,
    section: Arc<dyn Section>,
    priority: u8,
    region: MemoryRegion,
    mmap_able: bool,
    /// Directory where this consumer writes spill files. `None` disables spilling.
    spill_path: Option<PathBuf>,
    /// Counter for unique spill file names within `spill_path`.
    file_counter: AtomicUsize,
    /// `true` after a successful `spill_to_dir`, cleared on reload. Drives
    /// `current_tier()` so introspection reports the actual state of
    /// sections that opted into the `swap_to_mmap` path.
    is_spilled: std::sync::atomic::AtomicBool,
}

impl SectionConsumer {
    /// Creates a consumer for the given section without spill support.
    ///
    /// Priority and region are assigned based on the section type:
    /// - Data sections (types 1-9): `GRAPH_STORAGE` priority, `GraphStorage` region
    /// - Index sections (types 10+): `INDEX_BUFFERS` priority, `IndexBuffers` region
    ///
    /// Calling `spill()` on a consumer constructed via `new` returns
    /// [`SpillError::NoSpillDirectory`]. Use [`with_spill`](Self::with_spill)
    /// to enable disk-backed eviction.
    pub fn new(section: Arc<dyn Section>) -> Self {
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
    pub fn with_spill(section: Arc<dyn Section>, spill_path: PathBuf) -> Self {
        Self::build(section, Some(spill_path))
    }

    fn build(section: Arc<dyn Section>, spill_path: Option<PathBuf>) -> Self {
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
            file_counter: AtomicUsize::new(0),
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

        let id = self.file_counter.fetch_add(1, Ordering::Relaxed);
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

impl MemoryConsumer for SectionConsumer {
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
/// `memory_usage()` call. On `spill()`, vector embedding property columns
/// are drained to `MmapStorage` files, freeing heap memory. Search uses
/// [`SpillableVectorAccessor`](grafeo_core::index::vector::SpillableVectorAccessor)
/// which checks the spill storage first, then falls back to property storage.
#[cfg(all(
    feature = "lpg",
    feature = "vector-index",
    feature = "mmap",
    not(feature = "temporal")
))]
pub struct VectorIndexConsumer {
    store: Weak<grafeo_core::graph::lpg::LpgStore>,
    /// Directory for spill files. `None` disables spilling.
    spill_path: Option<PathBuf>,
    /// Map of "label:property" -> MmapStorage for spilled indexes.
    /// Shared with the search path so `SpillableVectorAccessor` can read.
    pub(crate) spilled: Arc<RwLock<HashMap<String, Arc<grafeo_core::index::vector::MmapStorage>>>>,
}

#[cfg(all(
    feature = "lpg",
    feature = "vector-index",
    feature = "mmap",
    not(feature = "temporal")
))]
impl VectorIndexConsumer {
    /// Creates a consumer that dynamically queries the store for current vector indexes.
    pub fn new(
        store: &Arc<grafeo_core::graph::lpg::LpgStore>,
        spill_path: Option<PathBuf>,
    ) -> Self {
        Self {
            store: Arc::downgrade(store),
            spill_path,
            spilled: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Returns the shared spill registry for the search path.
    #[must_use]
    pub fn spilled_storages(
        &self,
    ) -> &Arc<RwLock<HashMap<String, Arc<grafeo_core::index::vector::MmapStorage>>>> {
        &self.spilled
    }

    /// Writes one label-scoped vector index payload to disk.
    fn spill_index(
        &self,
        key: &str,
        dimensions: usize,
        vectors: &[(NodeId, Arc<[f32]>)],
    ) -> Result<(), SpillError> {
        let spill_dir = self
            .spill_path
            .as_ref()
            .ok_or(SpillError::NoSpillDirectory)?;
        std::fs::create_dir_all(spill_dir).map_err(|e| SpillError::IoError(e.to_string()))?;

        // Sanitize key for filename ("Label:property" -> "Label%3Aproperty")
        // while preserving label case and underscores.
        let safe_key = key.replace('%', "%25").replace(':', "%3A");
        let spill_file = spill_dir.join(format!("vectors_{safe_key}.bin"));
        let mmap_storage = grafeo_core::index::vector::MmapStorage::create(&spill_file, dimensions)
            .map_err(|e| SpillError::IoError(e.to_string()))?;

        for (id, vector) in vectors {
            mmap_storage
                .insert(*id, vector)
                .map_err(|e| SpillError::IoError(e.to_string()))?;
        }
        mmap_storage
            .flush()
            .map_err(|e| SpillError::IoError(e.to_string()))?;
        self.spilled
            .write()
            .insert(key.to_string(), Arc::new(mmap_storage));
        Ok(())
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
        self.spill_path.is_some()
    }

    fn current_tier(&self) -> grafeo_common::memory::StorageTier {
        use grafeo_common::memory::StorageTier;
        // Any spilled per-index storage means at least one index's
        // embeddings live on disk via MmapStorage. Report OnDisk in
        // that case; otherwise InMemory if any index has data, else
        // Uninitialized.
        if !self.spilled.read().is_empty() {
            return StorageTier::OnDisk;
        }
        if self.memory_usage() == 0 {
            StorageTier::Uninitialized
        } else {
            StorageTier::InMemory
        }
    }

    fn spill(&self, _target_bytes: usize) -> Result<usize, SpillError> {
        let store = self
            .store
            .upgrade()
            .ok_or(SpillError::IoError("store dropped".to_string()))?;

        // Property columns are global by property name while vector indexes are
        // label-scoped. Drain each shared property exactly once, then partition
        // its vectors by the labels registered for each index. The previous
        // per-index drain made the first HashMap entry consume every label's
        // vectors and left later indexes with empty spill stores.
        let already_spilled: HashSet<String> = self.spilled.read().keys().cloned().collect();
        let mut indexes_by_property: HashMap<String, Vec<(String, usize, HashSet<NodeId>)>> =
            HashMap::new();
        for (key, index) in store.vector_index_entries() {
            if already_spilled.contains(&key) {
                continue;
            }
            let Some((label, property)) = key.split_once(':') else {
                eprintln!("failed to spill vector index {key}: invalid index key");
                continue;
            };
            let label = label.to_string();
            let property = property.to_string();
            indexes_by_property.entry(property).or_default().push((
                key,
                index.config().dimensions,
                store.nodes_by_label(&label).into_iter().collect(),
            ));
        }

        let mut total_freed = 0;
        for (property, indexes) in indexes_by_property {
            let prop_key = PropertyKey::new(&property);
            let drained = store.drain_node_property_column(&prop_key);
            if drained.is_empty() {
                continue;
            }

            let mut consumed = HashSet::new();
            for (key, dimensions, label_nodes) in indexes {
                let vectors: Vec<(NodeId, Arc<[f32]>)> = drained
                    .iter()
                    .filter_map(|(id, value)| match value {
                        Value::Vector(vector)
                            if label_nodes.contains(id) && vector.len() == dimensions =>
                        {
                            Some((*id, Arc::clone(vector)))
                        }
                        _ => None,
                    })
                    .collect();
                if vectors.is_empty() {
                    continue;
                }
                match self.spill_index(&key, dimensions, &vectors) {
                    Ok(()) => consumed.extend(vectors.iter().map(|(id, _)| *id)),
                    Err(error) => eprintln!("failed to spill vector index {key}: {error}"),
                }
            }

            for (id, value) in &drained {
                if consumed.contains(id) {
                    if let Value::Vector(vector) = value {
                        total_freed += vector.len() * std::mem::size_of::<f32>()
                            + std::mem::size_of::<Arc<[f32]>>();
                    }
                }
            }
            let unconsumed = drained.into_iter().filter(|(id, _)| !consumed.contains(id));
            store.restore_node_property_column(&prop_key, unconsumed);
        }

        Ok(total_freed)
    }

    fn reload(&self) -> Result<(), SpillError> {
        let store = self
            .store
            .upgrade()
            .ok_or(SpillError::IoError("store dropped".to_string()))?;

        let mut spilled = self.spilled.write();
        for (key, mmap_storage) in spilled.drain() {
            let property = key
                .split(':')
                .nth(1)
                .ok_or_else(|| SpillError::IoError(format!("invalid index key: {key}")))?;
            let prop_key = PropertyKey::new(property);

            // Export vectors from mmap, restore to property store
            let vectors = mmap_storage.export_all();
            store.restore_node_property_column(
                &prop_key,
                vectors
                    .into_iter()
                    .map(|(id, vec_data)| (id, Value::Vector(vec_data))),
            );

            // Delete spill file
            if let Ok(path) = std::fs::canonicalize(mmap_storage.path()) {
                let _ = std::fs::remove_file(path);
            }
        }

        Ok(())
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
#[cfg(all(feature = "compact-store", feature = "lpg"))]
pub struct OverlayConsumer {
    layered: Weak<grafeo_core::graph::compact::layered::LayeredStore>,
}

#[cfg(all(feature = "compact-store", feature = "lpg"))]
impl OverlayConsumer {
    /// Creates a consumer that monitors the overlay of `layered`.
    pub fn new(layered: &Arc<grafeo_core::graph::compact::layered::LayeredStore>) -> Self {
        Self {
            layered: Arc::downgrade(layered),
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
    use grafeo_common::storage::section::SectionType;
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
