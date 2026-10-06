//! Memory-mapped section access for the `.grafeo` container.
//!
//! After a section is flushed to the container file, it can be memory-mapped
//! for zero-copy read access. The OS page cache manages eviction, providing
//! graceful degradation when data exceeds available RAM.
//!
//! Only sections with `flags.mmap_able = true` can be mapped (index sections:
//! VectorStore, TextIndex, RdfRing, PropertyIndex). Data sections (Catalog,
//! LpgStore, RdfStore) must be deserialized into RAM.

use grafeo_common::storage::SectionType;

use super::page_fetcher::AccessHint;

/// A read-only memory-mapped view of a section in the `.grafeo` container.
///
/// Created by [`write_and_mmap_spill_file`](super::spill::write_and_mmap_spill_file).
/// The mapping remains valid as long as this struct is alive. The OS page
/// cache serves reads: warm data is zero-copy, cold pages fault in
/// transparently from disk.
///
/// # Lifecycle
///
/// 1. The engine writes a section it evicts to its own spill file
/// 2. The spill file is mapped back as an `MmapSection`
/// 3. The engine drops the in-memory copy of the section data
/// 4. Reads go through the `MmapSection` (zero-copy from page cache)
/// 5. Before the spill file is rewritten, the engine **drops its mmaps first**
///
/// # Platform note
///
/// On Windows, the OS rejects writes to a file with active memory mappings
/// (error 1224: `ERROR_USER_MAPPED_FILE`), so every `MmapSection` of a file
/// must be dropped before that file is written. On Linux/macOS, writes
/// succeed with active mappings (old mappings see stale data), but the
/// drop-before-write lifecycle is used on all platforms for consistency.
pub struct MmapSection {
    mmap: memmap2::Mmap,
    section_type: SectionType,
    checksum: u32,
}

impl MmapSection {
    /// Creates a new `MmapSection`.
    ///
    /// Called internally by the spill path, which records the CRC-32 of the
    /// bytes it wrote.
    pub(crate) fn new(mmap: memmap2::Mmap, section_type: SectionType, checksum: u32) -> Self {
        Self {
            mmap,
            section_type,
            checksum,
        }
    }

    /// Returns the section data as a byte slice (zero-copy).
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }

    /// The section type this mapping covers.
    #[must_use]
    pub fn section_type(&self) -> SectionType {
        self.section_type
    }

    /// The CRC-32 checksum of the section data (verified on creation).
    #[must_use]
    pub fn checksum(&self) -> u32 {
        self.checksum
    }

    /// The byte length of the mapped section.
    #[must_use]
    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    /// Whether the mapping is zero-length.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }

    /// Advise the OS about the expected access pattern for a range.
    ///
    /// On Unix this delegates to `madvise` via `memmap2`. On Windows
    /// (and other platforms without a portable equivalent) it is a no-op.
    /// Out-of-range arguments and underlying errors are silently ignored:
    /// advice is a hint, not a contract.
    pub fn advise(&self, offset: usize, len: usize, hint: AccessHint) {
        // On Windows there is no portable madvise without `unsafe` FFI,
        // so this is a no-op. The args are intentionally unused there.
        let _ = (offset, len, hint);
        #[cfg(unix)]
        {
            use memmap2::Advice;
            // memmap2's safe `Advice` enum exposes only the read-side
            // hints; `MADV_DONTNEED` lives on `UncheckedAdvice` because
            // it can zero-fill subsequent reads, so it requires `unsafe`.
            // Treating `DontNeed` as a no-op here keeps the call safe;
            // if we ever need real eviction, plumb it through an
            // `unsafe` path in a dedicated helper.
            let advice = match hint {
                AccessHint::Sequential => Some(Advice::Sequential),
                AccessHint::Random => Some(Advice::Random),
                AccessHint::WillNeed => Some(Advice::WillNeed),
                AccessHint::DontNeed => None,
            };
            if let Some(advice) = advice {
                // Out-of-range or otherwise failing advise is best-effort.
                let _ = self.mmap.advise_range(advice, offset, len);
            }
        }
    }
}

impl AsRef<[u8]> for MmapSection {
    fn as_ref(&self) -> &[u8] {
        &self.mmap
    }
}

impl std::fmt::Debug for MmapSection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MmapSection")
            .field("section_type", &self.section_type)
            .field("len", &self.mmap.len())
            .field("checksum", &format_args!("{:#010X}", self.checksum))
            .finish()
    }
}
