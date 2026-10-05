//! Checkpoint writer of the v3 container.
//!
//! Streams the chunks of each section into free pages (copy-on-write: the
//! allocator never hands out a page the active image uses), then writes the
//! chained directory.

use std::fs::File;

use grafeo_common::storage::{ChunkMeta, SectionSink, SectionType};
use grafeo_common::utils::error::{Error, Result};

use super::alloc::{PageAllocator, PageRun};
use super::cipher::{ChunkCipher, ENCRYPTION_OVERHEAD};
use super::directory::{DirectoryEntry, encode_blocks};
use super::header::{BlockRef, PAGE_SIZE};

/// An I/O error that names the write it belongs to.
#[cfg(any(unix, windows))]
fn write_error(error: &std::io::Error, length: usize, offset: u64) -> Error {
    Error::Io(std::io::Error::new(
        error.kind(),
        format!("cannot write {length} bytes at offset {offset}: {error}"),
    ))
}

/// Refuses bytes whose length differs from the length their pages were
/// allocated for, so a write never leaves its run.
fn check_stored_length(
    what: impl std::fmt::Display,
    offset: u64,
    stored: usize,
    expected: usize,
) -> Result<()> {
    if stored == expected {
        return Ok(());
    }
    Err(Error::Internal(format!(
        "{what} at offset {offset} is {stored} bytes, expected {expected}: \
         refusing to write outside its pages"
    )))
}

/// Writes `bytes` at `offset`, extending the file when needed.
fn write_at(file: &File, offset: u64, bytes: &[u8]) -> Result<()> {
    #[cfg(unix)]
    {
        std::os::unix::fs::FileExt::write_all_at(file, bytes, offset)
            .map_err(|error| write_error(&error, bytes.len(), offset))?;
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (file, offset, bytes);
        return Err(Error::Internal(
            "positional file I/O is not supported on this platform".to_string(),
        ));
    }
    #[cfg(windows)]
    {
        let mut at = offset;
        let mut rest = bytes;
        while !rest.is_empty() {
            let written = std::os::windows::fs::FileExt::seek_write(file, rest, at)
                .map_err(|error| write_error(&error, bytes.len(), offset))?;
            if written == 0 {
                return Err(Error::Internal(format!(
                    "write of {} bytes at offset {offset} made no progress at offset {at}",
                    bytes.len()
                )));
            }
            at += written as u64;
            rest = &rest[written..];
        }
    }
    Ok(())
}

/// Writes the chunks of one checkpoint into pages the active header does not reach.
pub struct CheckpointWriter<'a> {
    file: &'a mut File,
    pages: PageAllocator,
    cipher: Option<&'a ChunkCipher>,
    entries: Vec<DirectoryEntry>,
    runs: Vec<PageRun>,
    current: Option<(SectionType, u8)>,
    /// Section types begun so far, in order.
    begun: Vec<SectionType>,
}

impl<'a> CheckpointWriter<'a> {
    /// Creates a writer allocating from `pages`; `cipher` is `None` for a
    /// plain image. Encrypted blocks get a random nonce each, because a
    /// retried checkpoint reuses the same offsets under the same key.
    pub fn new(file: &'a mut File, pages: PageAllocator, cipher: Option<&'a ChunkCipher>) -> Self {
        Self {
            file,
            pages,
            cipher,
            entries: Vec::new(),
            runs: Vec::new(),
            current: None,
            begun: Vec::new(),
        }
    }

    /// Starts a section; the chunks written next belong to it.
    ///
    /// # Errors
    ///
    /// Returns an error when `section_type` was already begun in this
    /// writer: the reader gathers every chunk of a type into one section, so
    /// a repeated type would mix two streams, possibly of different versions.
    /// After the error no section is current, so chunks are refused until the
    /// next successful `begin_section`.
    pub fn begin_section(&mut self, section_type: SectionType, version: u8) -> Result<()> {
        if self.begun.contains(&section_type) {
            self.current = None;
            return Err(Error::Internal(format!(
                "section {section_type:?} was already written in this checkpoint"
            )));
        }
        self.begun.push(section_type);
        self.current = Some((section_type, version));
        Ok(())
    }

    /// Writes the directory and returns its root and every run the new image uses.
    ///
    /// The directory has at least one block, also when no chunk was written,
    /// so the root always names a block. Zero-length runs are left out. The
    /// file is not padded to a page boundary.
    ///
    /// # Errors
    ///
    /// Returns an error when a write or an encryption fails, when the pages
    /// of a block would lie beyond the file offset range, or when the bytes
    /// of a block would not fit the pages allocated for it.
    pub fn finish(mut self) -> Result<(BlockRef, Vec<PageRun>)> {
        // An encrypted block is stored as nonce || ciphertext || tag.
        let stored_extra = if self.cipher.is_some() {
            ENCRYPTION_OVERHEAD
        } else {
            0
        };
        let pages = &mut self.pages;
        let runs = &mut self.runs;
        let (root, blocks) = encode_blocks(&self.entries, |length| {
            let stored = u64::try_from(length + stored_extra)
                .map_err(|_| Error::Internal("directory block is too large".to_string()))?;
            let run = pages.allocate(PageRun::for_bytes(stored))?;
            runs.push(run);
            run.offset()
        })?;
        for (offset, block) in blocks {
            let expected = block.len() + stored_extra;
            let stored = match self.cipher {
                #[cfg(feature = "encryption")]
                Some(cipher) => {
                    let nonce = grafeo_common::encryption::random_nonce();
                    cipher.encrypt(&block, &nonce, &super::cipher::directory_aad(offset))?
                }
                #[cfg(not(feature = "encryption"))]
                Some(cipher) => match *cipher {},
                None => block,
            };
            check_stored_length("directory block", offset, stored.len(), expected)?;
            write_at(self.file, offset, &stored)?;
        }
        Ok((root, self.runs))
    }
}

impl SectionSink for CheckpointWriter<'_> {
    fn write_chunk(&mut self, meta: ChunkMeta, bytes: &[u8]) -> Result<()> {
        let (section_type, section_version) = self
            .current
            .ok_or_else(|| Error::Internal("chunk written before a section began".to_string()))?;
        let mut entry = DirectoryEntry {
            section_type,
            section_version,
            meta,
            offset: 0,
            length: 0,
            crc: crc32fast::hash(&[]),
        };
        if !bytes.is_empty() {
            let stored_extra = if self.cipher.is_some() {
                ENCRYPTION_OVERHEAD
            } else {
                0
            };
            let expected = bytes.len() + stored_extra;
            let stored_length = u64::try_from(expected)
                .map_err(|_| Error::Internal("chunk is too large".to_string()))?;
            let run = self
                .pages
                .allocate(PageRun::for_bytes(stored_length))
                .map_err(|error| {
                    Error::Internal(format!("chunk of section {section_type:?}: {error}"))
                })?;
            entry.offset = run.offset()?;
            debug_assert!(entry.offset.is_multiple_of(PAGE_SIZE));
            entry.length = stored_length;
            #[cfg(feature = "encryption")]
            let encrypted;
            let stored: &[u8] = match self.cipher {
                #[cfg(feature = "encryption")]
                Some(cipher) => {
                    let nonce = grafeo_common::encryption::random_nonce();
                    encrypted = cipher
                        .encrypt(bytes, &nonce, &super::cipher::chunk_aad(&entry))
                        .map_err(|error| {
                            Error::Internal(format!("section {section_type:?}: {error}"))
                        })?;
                    &encrypted
                }
                #[cfg(not(feature = "encryption"))]
                Some(cipher) => match *cipher {},
                None => bytes,
            };
            check_stored_length(
                format_args!("chunk of section {section_type:?}"),
                entry.offset,
                stored.len(),
                expected,
            )?;
            entry.crc = crc32fast::hash(stored);
            write_at(self.file, entry.offset, stored)?;
            self.runs.push(run);
        }
        self.entries.push(entry);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use super::super::ImageReader;
    use super::super::alloc::{PageAllocator, PageRun};
    use super::super::header::{DATA_START_PAGE, PAGE_SIZE};
    use super::{CheckpointWriter, check_stored_length, write_at};

    fn open(dir: &tempfile::TempDir) -> File {
        File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(dir.path().join("x"))
            .unwrap()
    }

    /// A chunk whose pages would lie beyond the file offset range fails with
    /// its section named, before anything is written.
    #[test]
    fn a_chunk_beyond_the_offset_range_fails_before_it_is_written() {
        use grafeo_common::storage::{ChunkMeta, SectionSink, SectionType};

        let dir = tempfile::tempdir().unwrap();
        let mut file = open(&dir);
        // Every page up to the last whole page of the offset range is in use.
        let last = u64::MAX / PAGE_SIZE;
        let pages = PageAllocator::from_used([PageRun {
            first: DATA_START_PAGE,
            count: last - DATA_START_PAGE,
        }])
        .unwrap();
        assert!(pages.free_runs().is_empty() && pages.end_page() == last);
        let mut writer = CheckpointWriter::new(&mut file, pages, None);
        writer.begin_section(SectionType::LpgStore, 1).unwrap();
        let error = writer
            .write_chunk(ChunkMeta::raw(), b"Gus")
            .expect_err("the chunk's page would end past the last byte offset")
            .to_string();
        assert!(
            error.contains("LpgStore") && error.contains("offset range"),
            "the error names the section and says why: {error}"
        );
        drop(writer);
        assert_eq!(file.metadata().unwrap().len(), 0, "nothing was written");
    }

    #[test]
    fn an_image_without_chunks_still_has_one_directory_block() {
        let dir = tempfile::tempdir().unwrap();
        let mut file = open(&dir);
        let writer = CheckpointWriter::new(&mut file, PageAllocator::from_used([]).unwrap(), None);
        let (root, runs) = writer.finish().unwrap();
        assert_eq!(
            runs,
            [PageRun {
                first: DATA_START_PAGE,
                count: 1
            }],
            "one page for the directory block"
        );
        assert_eq!(root.offset, DATA_START_PAGE * PAGE_SIZE);
        assert_ne!(root.length, 0, "the root names a real block");
        assert_eq!(
            file.metadata().unwrap().len(),
            root.offset + u64::from(root.length),
            "the block is on disk"
        );
        let reader = ImageReader::open(&mut file, root, None).unwrap();
        assert_eq!(reader.entries(), [], "no chunks");
        assert_eq!(reader.used_runs(), runs);
    }

    #[cfg(feature = "encryption")]
    #[test]
    fn an_encrypted_image_without_chunks_still_needs_its_key() {
        use grafeo_common::encryption::PageEncryptor;
        let alix = PageEncryptor::new(&[3u8; 32]);
        let gus = PageEncryptor::new(&[19u8; 32]);
        let dir = tempfile::tempdir().unwrap();
        let mut file = open(&dir);
        let writer = CheckpointWriter::new(
            &mut file,
            PageAllocator::from_used([]).unwrap(),
            Some(&alix),
        );
        let (root, _) = writer.finish().unwrap();
        assert_eq!(
            ImageReader::open(&mut file, root, Some(&alix))
                .unwrap()
                .entries(),
            [],
            "the right key reads the empty directory"
        );
        let wrong = ImageReader::open(&mut file, root, Some(&gus))
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(wrong.contains("decrypt"), "{wrong}");
    }

    #[test]
    fn bytes_of_another_length_than_their_run_was_sized_for_are_refused() {
        check_stored_length("chunk of section Catalog", 12_288, 4 + 28, 32).unwrap();
        let error = check_stored_length("chunk of section Catalog", 12_288, 4097, 32)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("Catalog")
                && error.contains("offset 12288")
                && error.contains("4097 bytes, expected 32"),
            "{error}"
        );
    }

    #[test]
    fn a_failed_write_names_its_offset() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x");
        std::fs::write(&path, [3u8; 8192]).unwrap();
        let read_only = std::fs::File::open(&path).unwrap();
        let error = write_at(&read_only, 4096, b"Paris")
            .unwrap_err()
            .to_string();
        assert!(error.contains("offset 4096"), "{error}");
    }
}
