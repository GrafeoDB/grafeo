//! Image reader of the v3 container.
//!
//! The reader borrows the file for its lifetime and reads through a shared
//! handle with positional reads (`seek_read` on Windows, `read_exact_at` on
//! Unix), so every [`SectionChunks`] it hands out can fetch concurrently
//! without a lock.

use std::fs::File;

use bytes::Bytes;
use grafeo_common::storage::{ChunkIdentities, ChunkMeta, ImageSource, SectionSource, SectionType};
use grafeo_common::utils::error::{Error, Result};
use grafeo_common::utils::hash::FxHashMap;

use super::alloc::{PageAllocator, PageRun};
use super::cipher::{ChunkCipher, ENCRYPTION_OVERHEAD};
use super::directory::{DirectoryEntry, SkippedEntry, decode_chain};
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

/// Where a chunk is stored, for an error: "at offset N", or "without bytes"
/// for a chunk that has no pages (and so no offset).
fn stored_at(entry: &DirectoryEntry) -> String {
    if entry.length == 0 {
        "without bytes".to_string()
    } else {
        format!("at offset {}", entry.offset)
    }
}

/// Every chunk of one section carries one version, and no identity repeats
/// within a section.
///
/// The entries are grouped by section type: each section keeps the version
/// of its first chunk and the identities ([`ChunkMeta::identity`]) of its
/// chunks so far. Two sections may use the same identity and different
/// versions.
///
/// # Errors
///
/// Returns [`Error::Serialization`] naming the section and the chunk that
/// breaks a rule (its offset, or "without bytes"): "versions {first} and
/// {other}" with the kind, graph, column and first row of a chunk of
/// another version than the section's first, "two chunks of kind ..." for
/// an identity the section used already.
pub(super) fn check_sections(entries: &[DirectoryEntry]) -> Result<()> {
    let mut sections: FxHashMap<SectionType, (u8, ChunkIdentities)> = FxHashMap::default();
    for entry in entries {
        let section_type = entry.section_type;
        let (version, identities) = sections
            .entry(section_type)
            .or_insert_with(|| (entry.section_version, ChunkIdentities::default()));
        if entry.section_version != *version {
            let meta = &entry.meta;
            return Err(Error::Serialization(format!(
                "section {section_type:?} has chunks of versions {} and {}: the chunk of kind \
                 {:?} for graph {}, column {}, first row {} ({}) has version {}, but every \
                 chunk of a section carries one version",
                *version,
                entry.section_version,
                meta.kind,
                meta.graph_id,
                meta.column_id,
                meta.row_start,
                stored_at(entry),
                entry.section_version
            )));
        }
        identities
            .insert(section_type, &entry.meta)
            .map_err(|error| match error {
                Error::Serialization(message) => {
                    Error::Serialization(format!("{message}, the second {}", stored_at(entry)))
                }
                other => other,
            })?;
    }
    Ok(())
}

/// Refuses a chunk with bytes that is not page aligned, lies before the data
/// area or ends beyond the end of the file. `what` names the chunk in the
/// error. A chunk without bytes has no place to check.
fn check_placement(
    what: impl FnOnce() -> String,
    offset: u64,
    length: u64,
    file_length: u64,
) -> Result<()> {
    if length == 0 {
        return Ok(());
    }
    let end = offset.checked_add(length);
    if !offset.is_multiple_of(PAGE_SIZE)
        || offset < DATA_START_PAGE * PAGE_SIZE
        || end.is_none_or(|end| end > file_length)
    {
        return Err(Error::Serialization(format!(
            "{} at offset {offset} (length {length}) is misplaced or lies beyond the end of the \
             file ({file_length} bytes)",
            what()
        )));
    }
    Ok(())
}

/// The chunks of one image, read through a shared file handle.
pub struct ImageReader<'f> {
    file: &'f File,
    cipher: Option<&'f ChunkCipher>,
    entries: Vec<DirectoryEntry>,
    /// Entries of a newer version that the directory marks optional: only
    /// their pages are kept.
    skipped: Vec<SkippedEntry>,
    runs: Vec<PageRun>,
}

impl<'f> ImageReader<'f> {
    /// Reads the directory of the image rooted at `root`.
    ///
    /// The reader keeps a borrow of `file` for as long as it lives.
    ///
    /// An entry of a section type or chunk kind this version does not know,
    /// which the directory marks optional, is set apart: it is listed by
    /// [`skipped`](Self::skipped) and never handed to a section, and its
    /// chunk is never read. Its placement is checked as a known chunk's, and
    /// [`used_runs`](Self::used_runs) counts its pages. The reader logs
    /// nothing about them: the file manager warns once per section type and
    /// chunk kind when it opens the file.
    ///
    /// # Errors
    ///
    /// Returns an error when a directory block cannot be read, fails its
    /// checks or (for an encrypted image) cannot be decrypted with `cipher`,
    /// when an entry has an unknown section type or chunk kind that it does
    /// not mark optional, or an unknown flag among bits 0 to 3, when a chunk
    /// (a skipped one included) is misplaced or lies beyond the end of the
    /// file, when the chunks of a section carry two versions or two of them
    /// share an identity ([`ChunkMeta::identity`]), or when two chunks or
    /// directory blocks share a page.
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
        let (entries, skipped, runs) = decode_chain(root, overhead, |offset, length| {
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
        // `decode_chain` set the skipped entries apart: every entry here is of
        // a section type and chunk kind this build knows, so every one is
        // held to the rules of its section.
        check_sections(&entries)?;
        let file_length = file.metadata()?.len();
        for entry in &entries {
            check_placement(
                || format!("chunk of section {:?}", entry.section_type),
                entry.offset,
                entry.length,
                file_length,
            )?;
        }
        for entry in &skipped {
            check_placement(
                || {
                    format!(
                        "skipped chunk of section type {}, kind {}",
                        entry.section_type, entry.kind
                    )
                },
                entry.offset,
                entry.length,
                file_length,
            )?;
        }
        let reader = Self {
            file,
            cipher,
            entries,
            skipped,
            runs,
        };
        // The next checkpoint builds its allocator from these runs (the
        // skipped chunks' included); refuse an image whose pages overlap now
        // rather than when it is replaced.
        PageAllocator::from_used(reader.used_runs()).map_err(|error| {
            Error::Serialization(format!(
                "image with its directory at offset {}: {error}",
                root.offset
            ))
        })?;
        Ok(reader)
    }

    /// Every chunk of the image this version knows, in directory order.
    #[must_use]
    pub fn entries(&self) -> &[DirectoryEntry] {
        &self.entries
    }

    /// The entries skipped as optional; `used_runs()` includes their pages.
    ///
    /// They are in directory order. A checkpoint does not write them again,
    /// so they are gone once its image is active.
    #[must_use]
    pub fn skipped(&self) -> &[SkippedEntry] {
        &self.skipped
    }

    /// Pages of the directory blocks, one run per block, in chain order.
    #[must_use]
    pub fn directory_runs(&self) -> &[PageRun] {
        &self.runs
    }

    /// Pages the image uses: its chunks, the skipped ones included
    /// (zero-length chunks left out), and its directory blocks. The file
    /// header and the database headers are not part of an image.
    ///
    /// A checkpoint must not write these pages while the image is active, so
    /// a skipped chunk stays intact until the image is replaced.
    #[must_use]
    pub fn used_runs(&self) -> Vec<PageRun> {
        let mut runs: Vec<PageRun> = self
            .entries
            .iter()
            .map(DirectoryEntry::run)
            .chain(self.skipped.iter().map(SkippedEntry::run))
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
        let version = entries.first()?.section_version;
        let metas = entries.iter().map(|entry| entry.meta).collect();
        Some(SectionChunks {
            file: self.file,
            cipher: self.cipher,
            entries,
            metas,
            version,
        })
    }
}

impl ImageSource for ImageReader<'_> {
    /// The chunks of `section_type` (see [`ImageReader::section`]), or `None`
    /// when the image has none of it.
    fn section_source(&self, section_type: SectionType) -> Option<Box<dyn SectionSource + '_>> {
        self.section(section_type)
            .map(|section| Box::new(section) as Box<dyn SectionSource + '_>)
    }
}

/// The chunks of one section of an image.
///
/// [`ImageReader::open`] checked the section's chunks: they carry one
/// section version, and no two share an identity.
pub struct SectionChunks<'r> {
    file: &'r File,
    cipher: Option<&'r ChunkCipher>,
    entries: Vec<DirectoryEntry>,
    metas: Vec<ChunkMeta>,
    /// The section version, which every chunk carries.
    version: u8,
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

    /// The length the directory entry gives, which [`ImageReader::open`]
    /// checked lies inside the file.
    fn stored_length(&self, index: usize) -> Result<u64> {
        self.entries
            .get(index)
            .map(|entry| entry.length)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "chunk {index} out of range: section has {} chunks",
                    self.entries.len()
                ))
            })
    }

    fn section_version(&self) -> u8 {
        self.version
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use grafeo_common::storage::{ChunkMeta, SectionSink, SectionType};
    use grafeo_common::utils::error::Error;

    use super::super::alloc::PageAllocator;
    use super::super::directory::{DirectoryEntry, encode_blocks};
    use super::super::header::PAGE_SIZE;
    use super::super::writer::CheckpointWriter;
    use super::{ImageReader, check_sections, read_at};

    #[test]
    fn entries_with_a_repeated_identity_or_mixed_versions_are_refused() {
        let entry = |version, meta| DirectoryEntry {
            section_type: SectionType::LpgStore,
            section_version: version,
            meta,
            offset: 12_288,
            length: 0,
            crc: crc32fast::hash(&[]),
            flags: 0,
        };
        check_sections(&[
            entry(3, ChunkMeta::meta()),
            entry(3, ChunkMeta::column(0, 0, 0, 3, 2)),
        ])
        .unwrap();
        let repeated = check_sections(&[entry(3, ChunkMeta::meta()), entry(3, ChunkMeta::meta())])
            .unwrap_err();
        assert!(matches!(repeated, Error::Serialization(_)), "{repeated:?}");
        assert!(
            repeated.to_string().contains("two chunks")
                && repeated.to_string().contains("the second without bytes"),
            "{repeated}"
        );
        let mixed = check_sections(&[
            entry(3, ChunkMeta::meta()),
            entry(2, ChunkMeta::column(0, 19, 88, 3, 2)),
        ])
        .unwrap_err();
        assert!(matches!(mixed, Error::Serialization(_)), "{mixed:?}");
        assert!(mixed.to_string().contains("versions 3 and 2"), "{mixed}");
        assert!(
            mixed.to_string().contains(
                "the chunk of kind Column for graph 0, column 19, first row 88 (without bytes) \
                 has version 2"
            ),
            "a chunk without bytes is named by its identity: {mixed}"
        );
        check_sections(&[
            entry(3, ChunkMeta::meta()),
            DirectoryEntry {
                section_type: SectionType::RdfStore,
                ..entry(2, ChunkMeta::meta())
            },
        ])
        .expect("each section has its own version and its own identities");
    }

    /// `open` holds every image to the section rules: a directory written
    /// past the writer's checks is refused, naming the section and the
    /// offset of the chunk that breaks the rule.
    #[test]
    fn an_image_whose_directory_breaks_a_section_rule_does_not_open() {
        use std::io::{Seek, SeekFrom, Write};

        let dir = tempfile::tempdir().unwrap();
        let mut file = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(dir.path().join("x"))
            .unwrap();
        let mut writer =
            CheckpointWriter::new(&mut file, PageAllocator::from_used([]).unwrap(), None);
        writer.begin_section(SectionType::LpgStore, 3).unwrap();
        writer.write_chunk(ChunkMeta::meta(), b"Vincent").unwrap();
        writer
            .write_chunk(ChunkMeta::column(0, 3, 0, 19, 2), b"Jules")
            .unwrap();
        let (root, _) = writer.finish().unwrap();
        let good = ImageReader::open(&mut file, root, None)
            .unwrap()
            .entries()
            .to_vec();
        let repeated = [
            good[0],
            DirectoryEntry {
                meta: good[0].meta,
                ..good[1]
            },
        ];
        let mixed = [
            good[0],
            DirectoryEntry {
                section_version: 2,
                ..good[1]
            },
        ];
        for (entries, expected) in [(repeated, "two chunks"), (mixed, "versions 3 and 2")] {
            let at = file.metadata().unwrap().len().next_multiple_of(PAGE_SIZE);
            let (root, blocks) = encode_blocks(&entries, |_| Ok(at)).unwrap();
            file.seek(SeekFrom::Start(at)).unwrap();
            file.write_all(&blocks[0].1).unwrap();
            let error = ImageReader::open(&mut file, root, None)
                .map(|_| ())
                .unwrap_err()
                .to_string();
            assert!(
                error.contains(expected)
                    && error.contains("LpgStore")
                    && error.contains(&format!("offset {}", good[1].offset)),
                "{error}"
            );
        }
    }

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
