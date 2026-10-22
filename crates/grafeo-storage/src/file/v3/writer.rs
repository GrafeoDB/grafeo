//! Checkpoint writer of the v3 container.
//!
//! Streams the chunks of each section into free pages (copy-on-write: the
//! allocator never hands out a page the active image uses), then writes the
//! chained directory.
//!
//! A writer that follows an image (see [`CheckpointWriter::following`]) lets
//! each section put chunks of that image into the new one as they are
//! ([`SectionSink::reuse_chunk`]): the new directory points at the pages the
//! image stores them in, so a checkpoint writes only the chunks that changed
//! (#430). Those pages are the active image's, which the allocator never
//! hands out, and they become the new image's too.

use std::fs::File;

use grafeo_common::storage::{
    ChunkIdentities, ChunkMeta, ImageId, SectionSink, SectionSource, SectionType,
};
use grafeo_common::utils::error::{Error, Result};

use super::alloc::{PageAllocator, PageRun};
use super::cipher::{ChunkCipher, ENCRYPTION_OVERHEAD, chunk_aad_for};
use super::directory::{DirectoryEntry, ENTRY_SIZE, encode_blocks_raw};
use super::header::{BlockRef, PAGE_SIZE};
use super::reader::{ImageReader, SectionChunks};

/// What a checkpoint wrote: the chunks it stored and the bytes they take,
/// and the chunks it took over from the image it follows, which cost no
/// write.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CheckpointStats {
    /// Chunks whose bytes the checkpoint wrote (chunks without bytes
    /// included).
    pub chunks_written: usize,
    /// The bytes those chunks are stored in (an encrypted chunk with its
    /// nonce and tag); directory blocks are not counted.
    pub bytes_written: u64,
    /// Chunks of the image it follows that the new image points at as they
    /// are.
    pub chunks_reused: usize,
    /// The bytes those chunks are stored in.
    pub bytes_reused: u64,
    /// Of the reused chunks, those copied as they are stored into free pages
    /// nearer the start of the file, so the file can shrink (see
    /// [`CheckpointWriter::following`]).
    pub chunks_moved: usize,
    /// The bytes those chunks are stored in.
    pub bytes_moved: u64,
}

/// Where a stored chunk lies: offset, stored length and CRC of the stored
/// bytes.
struct Stored {
    offset: u64,
    length: u64,
    crc: u32,
}

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
    file: &'a File,
    pages: PageAllocator,
    cipher: Option<&'a ChunkCipher>,
    entries: Vec<DirectoryEntry>,
    runs: Vec<PageRun>,
    current: Option<(SectionType, u8)>,
    /// The image the checkpoint follows, whose chunks a section may reuse.
    following: Option<(ImageId, &'a ImageReader<'a>)>,
    /// That image's chunks of the current section.
    previous: Option<SectionChunks<'a>>,
    /// Whether a reused chunk moves into a free run before it.
    relocate: bool,
    stats: CheckpointStats,
    /// Section types begun so far, in order.
    begun: Vec<SectionType>,
    /// The identities the current section has used.
    identities: ChunkIdentities,
    /// Encoded entries of a newer version's chunks, which `finish` appends
    /// after the known entries.
    #[cfg(test)]
    foreign: Vec<[u8; ENTRY_SIZE]>,
}

impl<'a> CheckpointWriter<'a> {
    /// Creates a writer allocating from `pages`; `cipher` is `None` for a
    /// plain image. Encrypted blocks get a random nonce each, because a
    /// retried checkpoint reuses the same offsets under the same key.
    pub fn new(file: &'a File, pages: PageAllocator, cipher: Option<&'a ChunkCipher>) -> Self {
        Self {
            file,
            pages,
            cipher,
            entries: Vec::new(),
            runs: Vec::new(),
            current: None,
            following: None,
            previous: None,
            relocate: false,
            stats: CheckpointStats::default(),
            begun: Vec::new(),
            identities: ChunkIdentities::default(),
            #[cfg(test)]
            foreign: Vec::new(),
        }
    }

    /// The same writer following `image`, the image `reader` reads: each
    /// section is offered that image's chunks of it
    /// ([`SectionSink::previous`]) and may reuse them. Every page of
    /// `reader`'s image must be among those the allocator spares (the active
    /// image's are), so no chunk is written over a reused one.
    ///
    /// A reused chunk stays where it is, unless at least half of the pages
    /// the image spans are free: then a reused chunk that fits a free run
    /// before it is copied there as it is stored (its bytes, CRC and, for an
    /// encrypted file, nonce stay valid: a chunk's associated data does not
    /// name its offset), so the new image packs towards the start of the file
    /// and the file is cut after it. A file whose data shrank then shrinks
    /// too, instead of keeping its reused chunks where its end was.
    #[must_use]
    pub fn following(mut self, image: ImageId, reader: &'a ImageReader<'a>) -> Self {
        let runs = reader.used_runs();
        let used: u64 = runs.iter().map(|run| run.count).sum();
        let end = runs
            .iter()
            .map(|run| run.first + run.count)
            .max()
            .unwrap_or(super::header::DATA_START_PAGE);
        let span = end.saturating_sub(super::header::DATA_START_PAGE);
        self.relocate = span.saturating_sub(used) >= used;
        self.following = Some((image, reader));
        self
    }

    /// What the writer wrote and reused so far.
    #[must_use]
    pub fn stats(&self) -> CheckpointStats {
        self.stats
    }

    /// Starts a section; the chunks written next belong to it, and their
    /// identities are checked against each other only.
    ///
    /// # Errors
    ///
    /// Returns an error when `section_type` was already begun in this
    /// writer: the reader gathers every chunk of a type into one section, so
    /// a repeated type would mix two streams, possibly of different versions,
    /// and the identities of its second stream would be checked from
    /// scratch. After the error no section is current, so chunks are refused
    /// until the next successful `begin_section`.
    pub fn begin_section(&mut self, section_type: SectionType, version: u8) -> Result<()> {
        if self.begun.contains(&section_type) {
            self.current = None;
            return Err(Error::Internal(format!(
                "section {section_type:?} was already written in this checkpoint"
            )));
        }
        self.begun.push(section_type);
        self.current = Some((section_type, version));
        self.identities = ChunkIdentities::default();
        self.previous = self
            .following
            .and_then(|(_, reader)| reader.section(section_type));
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
        let encoded: Vec<[u8; ENTRY_SIZE]> = self
            .entries
            .iter()
            .map(|entry| {
                let mut bytes = [0u8; ENTRY_SIZE];
                entry.encode(&mut bytes);
                bytes
            })
            .collect();
        #[cfg(test)]
        let encoded = [encoded, std::mem::take(&mut self.foreign)].concat();
        let pages = &mut self.pages;
        let runs = &mut self.runs;
        let (root, blocks) = encode_blocks_raw(&encoded, |length| {
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

    /// Stores the bytes of one chunk in new pages, encrypted with the
    /// associated data `aad` builds when the writer has a cipher (it is not
    /// called otherwise), and records the pages among the image's runs. Empty
    /// bytes take no page: offset 0, length 0 and the CRC of nothing. `what`
    /// names the chunk in errors.
    fn store(&mut self, what: &str, aad: impl FnOnce() -> Vec<u8>, bytes: &[u8]) -> Result<Stored> {
        if bytes.is_empty() {
            return Ok(Stored {
                offset: 0,
                length: 0,
                crc: crc32fast::hash(&[]),
            });
        }
        let stored_extra = if self.cipher.is_some() {
            ENCRYPTION_OVERHEAD
        } else {
            0
        };
        let expected = bytes.len() + stored_extra;
        let stored_length =
            u64::try_from(expected).map_err(|_| Error::Internal(format!("{what} is too large")))?;
        let run = self
            .pages
            .allocate(PageRun::for_bytes(stored_length))
            .map_err(|error| Error::Internal(format!("{what}: {error}")))?;
        let offset = run.offset()?;
        debug_assert!(offset.is_multiple_of(PAGE_SIZE));
        #[cfg(feature = "encryption")]
        let encrypted;
        let stored: &[u8] = match self.cipher {
            #[cfg(feature = "encryption")]
            Some(cipher) => {
                let nonce = grafeo_common::encryption::random_nonce();
                encrypted = cipher
                    .encrypt(bytes, &nonce, &aad())
                    .map_err(|error| Error::Internal(format!("{what}: {error}")))?;
                &encrypted
            }
            #[cfg(not(feature = "encryption"))]
            Some(cipher) => match *cipher {},
            None => bytes,
        };
        #[cfg(not(feature = "encryption"))]
        let _ = aad;
        check_stored_length(what, offset, stored.len(), expected)?;
        let crc = crc32fast::hash(stored);
        write_at(self.file, offset, stored)?;
        self.runs.push(run);
        self.stats.bytes_written += stored_length;
        Ok(Stored {
            offset,
            length: stored_length,
            crc,
        })
    }

    /// Stands in for a writer of a newer version in tests: writes `bytes` as
    /// a chunk of section type byte `section_type` and chunk kind byte
    /// `kind`, which this version need not know, with the entry flags
    /// `flags`.
    ///
    /// The bytes go into new pages, encrypted with the associated data of
    /// those numbers (graph, column and first row 0) when the writer has a
    /// cipher. The entry carries the current section's version when
    /// `section_type` is its byte, else version 1, and zero in every other
    /// field; `finish` appends it after the known entries. No section needs
    /// to be current, and no identity is checked.
    ///
    /// # Errors
    ///
    /// Returns an error when allocating, encrypting or writing the chunk
    /// fails.
    #[cfg(test)]
    pub(crate) fn write_foreign_chunk(
        &mut self,
        section_type: u8,
        kind: u8,
        flags: u8,
        bytes: &[u8],
    ) -> Result<()> {
        let what = format!("chunk of section type {section_type}, kind {kind}");
        let stored = self.store(
            &what,
            || super::cipher::chunk_aad_parts(section_type, kind, 0, 0, 0, 0),
            bytes,
        )?;
        let version = match self.current {
            Some((current, version)) if current.to_u8() == section_type => version,
            _ => 1,
        };
        let mut entry = [0u8; ENTRY_SIZE];
        entry[0] = section_type;
        entry[1] = version;
        entry[2] = kind;
        entry[24..32].copy_from_slice(&stored.offset.to_le_bytes());
        entry[32..40].copy_from_slice(&stored.length.to_le_bytes());
        entry[40..44].copy_from_slice(&stored.crc.to_le_bytes());
        entry[44] = flags;
        self.foreign.push(entry);
        Ok(())
    }
}

impl SectionSink for CheckpointWriter<'_> {
    /// Writes the chunk into free pages of the current section.
    ///
    /// # Errors
    ///
    /// Returns an error when no section is current, when the section wrote
    /// a chunk with the same identity ([`ChunkMeta::identity`]) already, or
    /// when allocating, encrypting or writing the chunk fails. A chunk
    /// refused for its identity takes no page and writes nothing.
    fn write_chunk(&mut self, meta: ChunkMeta, bytes: &[u8]) -> Result<()> {
        let (section_type, section_version) = self
            .current
            .ok_or_else(|| Error::Internal("chunk written before a section began".to_string()))?;
        // Before a page is allocated or a byte encrypted, so a refused chunk
        // changes nothing.
        self.identities.insert(section_type, &meta)?;
        // The reader rebuilds the same associated data from the entry.
        let stored = self.store(
            &format!("chunk of section {section_type:?}"),
            || chunk_aad_for(section_type, &meta),
            bytes,
        )?;
        self.entries.push(DirectoryEntry {
            section_type,
            section_version,
            meta,
            offset: stored.offset,
            length: stored.length,
            crc: stored.crc,
            flags: DirectoryEntry::flags_for(section_type, meta.kind),
        });
        self.stats.chunks_written += 1;
        Ok(())
    }

    fn previous(&self) -> Option<(ImageId, &dyn SectionSource)> {
        let (image, _) = self.following?;
        let previous = self.previous.as_ref()?;
        Some((image, previous as &dyn SectionSource))
    }

    /// Points the new image at the pages chunk `index` of the followed
    /// image's section is stored in, with its directory entry as it was.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Internal`] when no section is current, the writer
    /// follows no image or the image has no chunk `index` of the section, or
    /// the chunk carries another section version than the current section;
    /// the error of a repeated identity. A refused chunk changes nothing.
    fn reuse_chunk(&mut self, index: usize) -> Result<()> {
        let (section_type, section_version) = self
            .current
            .ok_or_else(|| Error::Internal("chunk reused before a section began".to_string()))?;
        let entry = self
            .previous
            .as_ref()
            .and_then(|previous| previous.entry(index))
            .copied()
            .ok_or_else(|| {
                Error::Internal(format!(
                    "section {section_type:?}: the image this checkpoint follows has no chunk \
                     {index} to reuse"
                ))
            })?;
        if entry.section_version != section_version {
            return Err(Error::Internal(format!(
                "section {section_type:?}: chunk {index} of the image this checkpoint follows is \
                 of version {}, the section is written as version {section_version}",
                entry.section_version
            )));
        }
        self.identities.insert(section_type, &entry.meta)?;
        let mut offset = entry.offset;
        if entry.length > 0 {
            let run = entry.run();
            match self
                .relocate
                .then(|| self.pages.allocate_before(run.count, run.first))
                .flatten()
            {
                Some(target) => {
                    let stored = self
                        .previous
                        .as_ref()
                        .expect("the entry came from it")
                        .stored_bytes(index)?;
                    offset = target.offset()?;
                    write_at(self.file, offset, &stored)?;
                    self.runs.push(target);
                    self.stats.chunks_moved += 1;
                    self.stats.bytes_moved += entry.length;
                }
                None => self.runs.push(run),
            }
        }
        self.stats.chunks_reused += 1;
        self.stats.bytes_reused += entry.length;
        self.entries.push(DirectoryEntry {
            flags: DirectoryEntry::flags_for(section_type, entry.meta.kind),
            offset,
            ..entry
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use super::super::ImageReader;
    use super::super::alloc::{PageAllocator, PageRun};
    use super::super::header::{DATA_START_PAGE, PAGE_SIZE};
    use super::{CheckpointWriter, Error, check_stored_length, write_at};

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
        let file = open(&dir);
        // Every page up to the last whole page of the offset range is in use.
        let last = u64::MAX / PAGE_SIZE;
        let pages = PageAllocator::from_used([PageRun {
            first: DATA_START_PAGE,
            count: last - DATA_START_PAGE,
        }])
        .unwrap();
        assert!(pages.free_runs().is_empty() && pages.end_page() == last);
        let mut writer = CheckpointWriter::new(&file, pages, None);
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

    /// A chunk whose identity its section used already is refused before it
    /// takes a page or writes a byte; another section may use the identity.
    #[test]
    fn a_repeated_chunk_identity_is_refused_before_it_is_written() {
        use grafeo_common::storage::{ChunkMeta, SectionSink, SectionSource, SectionType};

        let dir = tempfile::tempdir().unwrap();
        let file = open(&dir);
        let mut writer = CheckpointWriter::new(&file, PageAllocator::from_used([]).unwrap(), None);
        writer.begin_section(SectionType::LpgStore, 3).unwrap();
        writer
            .write_chunk(ChunkMeta::column(0, 16, 0, 3, 7), b"Alix")
            .unwrap();
        let error = writer
            .write_chunk(ChunkMeta::column(0, 16, 0, 3, 7), b"Gus")
            .unwrap_err();
        assert!(matches!(error, Error::Serialization(_)), "{error:?}");
        let error = error.to_string();
        assert!(
            error.contains("LpgStore") && error.contains("two chunks"),
            "{error}"
        );
        writer.begin_section(SectionType::RdfStore, 3).unwrap();
        // Another section: allowed.
        writer
            .write_chunk(ChunkMeta::column(0, 16, 0, 3, 7), b"Mia")
            .unwrap();
        let (root, runs) = writer.finish().unwrap();
        assert_eq!(
            runs.iter().map(|run| run.count).sum::<u64>(),
            1 + 1 + 1,
            "two chunks and one directory block: the refused chunk took no page"
        );
        assert_eq!(
            runs.iter().map(|run| run.end().unwrap()).max(),
            Some(DATA_START_PAGE + 3),
            "the three pages follow each other: the refused chunk left no gap"
        );
        let bytes = std::fs::read(dir.path().join("x")).unwrap();
        assert!(
            !bytes.windows(3).any(|window| window == b"Gus"),
            "the refused chunk's bytes were not written"
        );
        let reader = ImageReader::open(&file, root, None).unwrap();
        let store = reader.section(SectionType::LpgStore).unwrap();
        assert_eq!(store.chunks(), [ChunkMeta::column(0, 16, 0, 3, 7)]);
        assert_eq!(
            &store.fetch(0).unwrap()[..],
            b"Alix",
            "the first chunk stays"
        );
        let rdf = reader.section(SectionType::RdfStore).unwrap();
        assert_eq!(&rdf.fetch(0).unwrap()[..], b"Mia");
    }

    /// A chunk without bytes is held to the identity rule too, so every
    /// image the writer finishes opens.
    #[test]
    fn a_repeated_identity_of_chunks_without_bytes_is_refused() {
        use grafeo_common::storage::{ChunkMeta, SectionSink, SectionType};

        let dir = tempfile::tempdir().unwrap();
        let file = open(&dir);
        let mut writer = CheckpointWriter::new(&file, PageAllocator::from_used([]).unwrap(), None);
        writer.begin_section(SectionType::LpgStore, 3).unwrap();
        writer.write_chunk(ChunkMeta::meta(), b"").unwrap();
        let error = writer
            .write_chunk(ChunkMeta::meta(), b"")
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("LpgStore") && error.contains("two chunks"),
            "{error}"
        );
        let (root, runs) = writer.finish().unwrap();
        assert_eq!(
            runs.iter().map(|run| run.count).sum::<u64>(),
            1,
            "the directory block only: chunks without bytes take no page"
        );
        let reader =
            ImageReader::open(&file, root, None).expect("every image the writer finishes opens");
        assert_eq!(
            reader.entries().len(),
            1,
            "the refused chunk left no entry in the directory"
        );
    }

    #[test]
    fn an_image_without_chunks_still_has_one_directory_block() {
        let dir = tempfile::tempdir().unwrap();
        let file = open(&dir);
        let writer = CheckpointWriter::new(&file, PageAllocator::from_used([]).unwrap(), None);
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
        let reader = ImageReader::open(&file, root, None).unwrap();
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
        let file = open(&dir);
        let writer =
            CheckpointWriter::new(&file, PageAllocator::from_used([]).unwrap(), Some(&alix));
        let (root, _) = writer.finish().unwrap();
        assert_eq!(
            ImageReader::open(&file, root, Some(&alix))
                .unwrap()
                .entries(),
            [],
            "the right key reads the empty directory"
        );
        let wrong = ImageReader::open(&file, root, Some(&gus))
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
