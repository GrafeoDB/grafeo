//! Image reader of the v3 container.
//!
//! The reader borrows the file for its lifetime and reads through a shared
//! handle with positional reads (`seek_read` on Windows, `read_exact_at` on
//! Unix), so every [`SectionChunks`] it hands out can fetch concurrently
//! without a lock.

use std::fs::File;

use bytes::Bytes;
use grafeo_common::storage::{ChunkMeta, SectionSource, SectionType};
use grafeo_common::utils::error::{Error, Result};

use super::alloc::{PageAllocator, PageRun};
use super::cipher::{ChunkCipher, ENCRYPTION_OVERHEAD};
use super::directory::{DirectoryEntry, decode_chain};
use super::header::{BlockRef, DATA_START_PAGE, PAGE_SIZE};

/// Reads exactly `length` bytes at `offset`.
fn read_at(file: &File, offset: u64, length: usize) -> Result<Vec<u8>> {
    let mut buffer = vec![0u8; length];
    #[cfg(unix)]
    {
        std::os::unix::fs::FileExt::read_exact_at(file, &mut buffer, offset).map_err(|error| {
            Error::Serialization(format!(
                "cannot read {length} bytes at offset {offset}: {error}"
            ))
        })?;
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (file, offset, &mut buffer);
        return Err(Error::Internal(
            "positional file I/O is not supported on this platform".to_string(),
        ));
    }
    #[cfg(windows)]
    {
        let mut done = 0usize;
        while done < length {
            let at = offset + done as u64;
            let read = std::os::windows::fs::FileExt::seek_read(file, &mut buffer[done..], at)
                .map_err(|error| {
                    Error::Serialization(format!(
                        "cannot read {length} bytes at offset {offset}: {error}"
                    ))
                })?;
            if read == 0 {
                return Err(Error::Serialization(format!(
                    "cannot read {length} bytes at offset {offset}: unexpected end of file"
                )));
            }
            done += read;
        }
    }
    Ok(buffer)
}

/// The chunks of one image, read through a shared file handle.
pub struct ImageReader<'f> {
    file: &'f File,
    cipher: Option<&'f ChunkCipher>,
    entries: Vec<DirectoryEntry>,
    runs: Vec<PageRun>,
}

impl<'f> ImageReader<'f> {
    /// Reads the directory of the image rooted at `root`.
    ///
    /// The reader keeps a borrow of `file` for as long as it lives.
    ///
    /// # Errors
    ///
    /// Returns an error when a directory block cannot be read, fails its
    /// checks or (for an encrypted image) cannot be decrypted with `cipher`,
    /// when a chunk is misplaced or lies beyond the end of the file, or when
    /// two chunks or directory blocks share a page.
    pub fn open(
        file: &'f mut File,
        root: BlockRef,
        cipher: Option<&'f ChunkCipher>,
    ) -> Result<Self> {
        let file: &'f File = file;
        // An encrypted block is stored with its nonce and tag around it.
        let extra = if cipher.is_some() {
            ENCRYPTION_OVERHEAD
        } else {
            0
        };
        let overhead = u32::try_from(extra).map_err(|_| {
            Error::Internal(format!("encryption overhead {extra} does not fit a u32"))
        })?;
        let (entries, runs) = decode_chain(root, overhead, |offset, length| {
            let stored = read_at(file, offset, length as usize + extra).map_err(|error| {
                Error::Serialization(format!("directory block at offset {offset}: {error}"))
            })?;
            #[cfg(feature = "encryption")]
            if let Some(cipher) = cipher {
                return cipher
                    .decrypt(&stored, &super::cipher::directory_aad(offset))
                    .map_err(|error| {
                        Error::Serialization(format!("directory block at offset {offset}: {error}"))
                    });
            }
            #[cfg(not(feature = "encryption"))]
            if let Some(cipher) = cipher {
                match *cipher {}
            }
            Ok(stored)
        })?;
        let file_length = file.metadata()?.len();
        for entry in entries.iter().filter(|entry| entry.length > 0) {
            let end = entry.offset.checked_add(entry.length);
            if !entry.offset.is_multiple_of(PAGE_SIZE)
                || entry.offset < DATA_START_PAGE * PAGE_SIZE
                || end.is_none_or(|end| end > file_length)
            {
                return Err(Error::Serialization(format!(
                    "chunk of section {:?} at offset {} (length {}) is misplaced or lies \
                     beyond the end of the file ({file_length} bytes)",
                    entry.section_type, entry.offset, entry.length
                )));
            }
        }
        let reader = Self {
            file,
            cipher,
            entries,
            runs,
        };
        // The next checkpoint builds its allocator from these runs; refuse an
        // image whose pages overlap now rather than when it is replaced.
        PageAllocator::from_used(reader.used_runs()).map_err(|error| {
            Error::Serialization(format!(
                "image with its directory at offset {}: {error}",
                root.offset
            ))
        })?;
        Ok(reader)
    }

    /// Every chunk of the image, in directory order.
    #[must_use]
    pub fn entries(&self) -> &[DirectoryEntry] {
        &self.entries
    }

    /// Pages the image uses: its chunks (zero-length chunks left out) and its
    /// directory blocks.
    #[must_use]
    pub fn used_runs(&self) -> Vec<PageRun> {
        let mut runs: Vec<PageRun> = self
            .entries
            .iter()
            .map(DirectoryEntry::run)
            .filter(|run| run.count > 0)
            .collect();
        runs.extend(self.runs.iter().copied());
        runs
    }

    /// The chunks of one section, or `None` when the image has none of it.
    #[must_use]
    pub fn section(&self, section_type: SectionType) -> Option<SectionChunks<'_>> {
        let entries: Vec<DirectoryEntry> = self
            .entries
            .iter()
            .filter(|entry| entry.section_type == section_type)
            .copied()
            .collect();
        if entries.is_empty() {
            return None;
        }
        let metas = entries.iter().map(|entry| entry.meta).collect();
        Some(SectionChunks {
            file: self.file,
            cipher: self.cipher,
            entries,
            metas,
        })
    }
}

/// The chunks of one section of an image.
pub struct SectionChunks<'r> {
    file: &'r File,
    cipher: Option<&'r ChunkCipher>,
    entries: Vec<DirectoryEntry>,
    metas: Vec<ChunkMeta>,
}

impl SectionSource for SectionChunks<'_> {
    fn chunks(&self) -> &[ChunkMeta] {
        &self.metas
    }

    fn fetch(&self, index: usize) -> Result<Bytes> {
        let entry = self.entries.get(index).ok_or_else(|| {
            Error::Internal(format!(
                "chunk {index} out of range: section has {} chunks",
                self.entries.len()
            ))
        })?;
        if entry.length == 0 {
            return Ok(Bytes::new());
        }
        let section_type = entry.section_type;
        let offset = entry.offset;
        let length = usize::try_from(entry.length).map_err(|_| {
            Error::Serialization(format!(
                "chunk of section {section_type:?} at offset {offset} is too large"
            ))
        })?;
        let stored = read_at(self.file, offset, length)?;
        if crc32fast::hash(&stored) != entry.crc {
            return Err(Error::Serialization(format!(
                "chunk of section {section_type:?} at offset {offset} fails its checksum"
            )));
        }
        #[cfg(feature = "encryption")]
        if let Some(cipher) = self.cipher {
            let plain = cipher
                .decrypt(&stored, &super::cipher::chunk_aad(entry))
                .map_err(|error| {
                    Error::Serialization(format!(
                        "chunk of section {section_type:?} at offset {offset}: {error}"
                    ))
                })?;
            return Ok(Bytes::from(plain));
        }
        #[cfg(not(feature = "encryption"))]
        if let Some(cipher) = self.cipher {
            match *cipher {}
        }
        Ok(Bytes::from(stored))
    }
}

#[cfg(test)]
mod tests {
    use super::read_at;

    #[test]
    fn a_failed_read_names_its_offset() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x");
        std::fs::write(&path, [3u8; 8192]).unwrap();
        let write_only = std::fs::File::options().write(true).open(&path).unwrap();
        let error = read_at(&write_only, 4096, 19).unwrap_err().to_string();
        assert!(error.contains("offset 4096"), "{error}");
    }
}
