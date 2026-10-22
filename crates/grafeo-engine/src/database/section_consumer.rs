//! Adapts storage sections into [`MemoryConsumer`]s for BufferManager integration.
//!
//! Each section (LPG, RDF, Vector, Text, Catalog) is registered with the
//! [`BufferManager`] so that memory tracking and pressure awareness include
//! section memory. This enables accurate `memory_usage()` reporting and
//! lays the groundwork for automatic spilling when tiered storage is added.

use std::path::PathBuf;
use std::sync::Arc;

use grafeo_common::memory::buffer::{MemoryConsumer, MemoryRegion, SpillError, priorities};
use grafeo_common::storage::Section;

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
    // section consumer types use specialized constructors (VectorIndex,
    // TextIndex). Allow dead_code under feature combinations
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
