//! Standalone spill files for individual section eviction.
//!
//! A `.grafeo` checkpoint writes a whole new image and switches the database
//! header to it, which is too heavyweight for a single-section eviction
//! triggered by memory pressure. This module
//! provides a lightweight alternative: write one section's bytes to its
//! own file, mmap it back, and return an [`MmapSection`] ready for use
//! with [`MmapPageFetcher`](super::page_fetcher::MmapPageFetcher).
//!
//! Spill files are written atomically (write + rename) and live in a
//! caller-supplied directory. Cleanup is the caller's responsibility.

use std::fs;
use std::path::Path;

use grafeo_common::storage::SectionType;
use grafeo_common::utils::error::{Error, Result};

use super::mmap::MmapSection;

/// Atomically write `bytes` to `path`, mmap the file, and return an
/// [`MmapSection`] over the mapped region.
///
/// The CRC-32 of `bytes` is computed up-front and recorded on the
/// returned [`MmapSection`] so callers can verify integrity later.
///
/// # Errors
///
/// Returns [`Error::Internal`] if any of the following fails: creating
/// the parent directory, writing the temp file, renaming into place,
/// opening the file, or memory-mapping it.
pub fn write_and_mmap_spill_file(
    path: &Path,
    bytes: &[u8],
    section_type: SectionType,
) -> Result<MmapSection> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| Error::Internal(format!("create dir for {}: {e}", parent.display())))?;
    }

    // Write atomically: write to a sibling temp file, then rename.
    let tmp = path.with_extension("spill.tmp");
    fs::write(&tmp, bytes).map_err(|e| Error::Internal(format!("write {}: {e}", tmp.display())))?;
    fs::rename(&tmp, path).map_err(|e| {
        Error::Internal(format!(
            "rename {} -> {}: {e}",
            tmp.display(),
            path.display()
        ))
    })?;

    let checksum = crc32fast::hash(bytes);

    let file = fs::File::open(path)
        .map_err(|e| Error::Internal(format!("open {}: {e}", path.display())))?;

    // SAFETY: the file was just written by this process and is held
    // open for the lifetime of the mapping. External truncation while
    // the mmap is alive is undefined per memmap2 docs (same caveat as
    // every other mmap call site in the project). The mapping is
    // read-only.
    #[allow(unsafe_code)]
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }
        .map_err(|e| Error::Internal(format!("mmap {}: {e}", path.display())))?;

    Ok(MmapSection::new(mmap, section_type, checksum))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_and_mmap_round_trip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("test.spill");
        let payload: Vec<u8> = (0u32..1024)
            .map(|i| u8::try_from(i % 251).expect("fits in u8"))
            .collect();

        let mmap = write_and_mmap_spill_file(&path, &payload, SectionType::PropertyIndex)
            .expect("write+mmap");

        assert_eq!(mmap.section_type(), SectionType::PropertyIndex);
        assert_eq!(mmap.len(), payload.len());
        assert_eq!(mmap.as_bytes(), payload.as_slice());
        assert_eq!(mmap.checksum(), crc32fast::hash(&payload));
    }

    #[test]
    fn write_creates_missing_parent_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("nested").join("subdir").join("test.spill");
        let payload = vec![0xAB; 256];

        let mmap = write_and_mmap_spill_file(&path, &payload, SectionType::TextIndex)
            .expect("write+mmap with nested parent");
        assert_eq!(mmap.as_bytes(), payload.as_slice());
        assert!(path.exists(), "spill file must exist after write");
    }

    #[test]
    fn write_overwrites_existing_file_atomically() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("test.spill");

        // First write
        let first_payload = vec![0x11; 128];
        let _first = write_and_mmap_spill_file(&path, &first_payload, SectionType::VectorStore)
            .expect("first write");

        // Drop must happen so Windows allows write. memmap2::Mmap drops
        // on scope exit so an explicit drop is unnecessary, but we
        // bind to `_first` to make scoping clear above.

        // Second write replaces atomically
        let second_payload = vec![0x22; 256];
        let second = write_and_mmap_spill_file(&path, &second_payload, SectionType::VectorStore)
            .expect("second write");
        assert_eq!(second.as_bytes(), second_payload.as_slice());
        assert_eq!(second.len(), 256);
    }
}
