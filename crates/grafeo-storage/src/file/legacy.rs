//! Reading `.grafeo` files written by 0.5.x (container v1 and v2).
//!
//! 0.5.x wrote either one snapshot blob (v1) or a section directory at
//! `DIRECTORY_OFFSET` followed by the section data (v2). 0.6.x writes
//! container v3 and keeps these readers only to migrate old files.
//!
//! [`LegacyFile`] opens such a file read-only.
//! [`GrafeoFileManager`](super::GrafeoFileManager) opens only container v3
//! and refuses these files: they must be migrated first.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use grafeo_common::storage::{SectionDirectoryEntry, SectionType};
use grafeo_common::testing::child_process;
use grafeo_common::utils::error::{Error, Result};

use super::format::{DATA_OFFSET, DbHeader};
use super::header;
use super::v3::cipher::ChunkCipher;
use crate::container::SectionDirectory;
use crate::container::directory::DIRECTORY_OFFSET;

/// What a 0.5.x database file holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LegacyContents {
    /// The file was never checkpointed, or holds no data.
    Empty,
    /// Container v1: one snapshot blob.
    Snapshot(Vec<u8>),
    /// Container v2: the sections of the directory, decrypted.
    Sections(Vec<(SectionType, Vec<u8>)>),
}

/// A database file written by 0.5.x (container v1 or v2), opened to read it once.
///
/// Holds a shared lock for as long as it lives and never writes.
pub struct LegacyFile<'cipher> {
    /// Path to the database file (not the pending image).
    path: PathBuf,
    /// The file that is read: the database file, or a pending checkpoint image.
    file: File,
    /// The active database header of `file`.
    header: DbHeader,
    /// The database file, when `file` is a pending image instead. It carries
    /// the shared lock.
    lock_holder: Option<File>,
    /// Decrypts section data (`None` = unencrypted).
    cipher: Option<&'cipher ChunkCipher>,
}

impl<'cipher> LegacyFile<'cipher> {
    /// Opens the 0.5.x database file at `path` for reading.
    ///
    /// Takes a shared lock. If a checkpoint was interrupted while installing
    /// its new image, the complete image `<path>.checkpoint` is read instead
    /// of the database file and left in place. Never writes anything.
    ///
    /// # Errors
    ///
    /// Returns an error if the file does not exist, cannot be locked, has
    /// invalid magic or an unsupported format version.
    pub fn open(path: &Path, cipher: Option<&'cipher ChunkCipher>) -> Result<Self> {
        let (mut file, lock_holder) = open_shared(path)?;
        let (_, header) = read_active_header(&mut file)?;
        Ok(Self {
            path: path.to_path_buf(),
            file,
            header,
            lock_holder,
            cipher,
        })
    }

    /// The path of the database file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The active database header.
    #[must_use]
    pub fn header(&self) -> &DbHeader {
        &self.header
    }

    /// The sidecar WAL directory of 0.5.x: `<path>.wal`.
    #[must_use]
    pub fn sidecar_wal_path(&self) -> PathBuf {
        super::detect::sidecar_wal_path(&self.path)
    }

    /// Reads everything the file holds: the sections of a v2 file (decrypted
    /// with the cipher given to [`open`](Self::open)), the snapshot blob of a
    /// v1 file, or nothing for a file that was never checkpointed.
    ///
    /// # Errors
    ///
    /// Returns an error if a checksum does not match, the directory is
    /// corrupt, or a section cannot be decrypted.
    pub fn contents(&self) -> Result<LegacyContents> {
        let mut file = self.file.try_clone()?;
        if let Some(directory) = read_directory(&mut file, &self.header)? {
            let mut sections = Vec::with_capacity(directory.len());
            for entry in directory.entries() {
                let data = read_section(&mut file, entry, self.cipher)?;
                sections.push((entry.section_type, data));
            }
            return Ok(LegacyContents::Sections(sections));
        }
        let blob = read_snapshot_blob(&mut file, &self.header)?;
        if blob.is_empty() {
            Ok(LegacyContents::Empty)
        } else {
            Ok(LegacyContents::Snapshot(blob))
        }
    }
}

impl Drop for LegacyFile<'_> {
    fn drop(&mut self) {
        let _ = self.lock_holder.as_ref().unwrap_or(&self.file).unlock();
    }
}

/// Opens `path` read-only under a shared lock. Returns the file to read and,
/// when that is a pending checkpoint image instead of the database file, the
/// database file that carries the lock.
fn open_shared(path: &Path) -> Result<(File, Option<File>)> {
    let database_file = OpenOptions::new().read(true).open(path)?;

    // A shared lock coexists with other shared locks but blocks if an
    // exclusive lock cannot be shared (platform-dependent).
    child_process::take_lock(
        || database_file.try_lock_shared(),
        |e| matches!(e, std::fs::TryLockError::WouldBlock),
    )
    .map_err(|_| {
        Error::Internal(format!(
            "database file cannot be locked for reading: {}",
            path.display()
        ))
    })?;

    let pending_image = checkpoint_image_path(path);
    Ok(if pending_image.exists() {
        (File::open(&pending_image)?, Some(database_file))
    } else {
        (database_file, None)
    })
}

/// A complete new image that a 0.5.x checkpoint was installing over the
/// database file: `<path>.checkpoint`. Its presence means the install was
/// cut off, and the image holds the database.
fn checkpoint_image_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".checkpoint");
    PathBuf::from(name)
}

/// Reads and validates the file header, then selects the active database
/// header. Returns its slot and the header.
pub(super) fn read_active_header(file: &mut File) -> Result<(u8, DbHeader)> {
    let file_header = header::read_file_header(file)?;
    header::validate_file_header(&file_header)?;
    let (h0, h1) = header::read_db_headers(file)?;
    Ok(header::active_db_header(&h0, &h1))
}

/// Reads the v1 snapshot blob using the active database header.
///
/// Returns an empty `Vec` if the database has never been checkpointed (the
/// header is empty) or the file is v2 (`snapshot_length == 0`).
///
/// # Errors
///
/// Returns an error if the snapshot lies beyond the end of the file (checked
/// before its buffer is allocated: 0.5.x headers have no checksum, so a
/// damaged length never allocates), the read fails or the CRC checksum does
/// not match.
pub(super) fn read_snapshot_blob(file: &mut File, active_header: &DbHeader) -> Result<Vec<u8>> {
    if active_header.is_empty() {
        return Ok(Vec::new());
    }

    // v2 files store sections rather than a v1 snapshot blob. They set
    // snapshot_length == 0 and put the directory CRC in the checksum field.
    // Reading 0 bytes here would CRC to 0 and mismatch the directory CRC.
    if active_header.snapshot_length == 0 {
        return Ok(Vec::new());
    }

    let file_length = file.metadata()?.len();
    if DATA_OFFSET
        .checked_add(active_header.snapshot_length)
        .is_none_or(|end| end > file_length)
    {
        return Err(Error::Internal(format!(
            "snapshot at offset {DATA_OFFSET} (length {}) lies beyond the end of the file \
             ({file_length} bytes)",
            active_header.snapshot_length
        )));
    }
    let length = usize::try_from(active_header.snapshot_length).map_err(|_| {
        Error::Internal(format!(
            "snapshot length {} does not fit in memory on this platform",
            active_header.snapshot_length
        ))
    })?;
    let expected_checksum = active_header.checksum;

    file.seek(SeekFrom::Start(DATA_OFFSET))?;

    let mut data = vec![0u8; length];
    file.read_exact(&mut data)?;

    let actual_checksum = crc32fast::hash(&data);
    if actual_checksum != expected_checksum {
        return Err(Error::Internal(format!(
            "snapshot checksum mismatch: expected {expected_checksum:#010X}, got {actual_checksum:#010X}"
        )));
    }

    Ok(data)
}

/// Reads the v2 section directory from the file.
///
/// Detects v2 format by checking the `snapshot_length` field in the active
/// database header: v2 writes set `snapshot_length = 0`, while v1 always has
/// a non-zero snapshot length when data exists.
///
/// Returns `None` only when the file is unambiguously v1 (`snapshot_length`
/// non-zero), uninitialized (header iteration is 0) or has an empty
/// directory. Once the header asserts v2, any failure to locate or parse the
/// directory is surfaced as an error: misreporting v2 corruption as a v1
/// file would cause callers to fall back to v1 read paths and mask the
/// underlying problem.
///
/// # Errors
///
/// Returns an error if:
/// - I/O fails
/// - The header asserts v2 but the file is too short to hold a directory page
/// - The directory page bytes fail to parse as a `SectionDirectory`
/// - The directory page CRC does not match the value recorded in the active header
pub(super) fn read_directory(
    file: &mut File,
    active_header: &DbHeader,
) -> Result<Option<SectionDirectory>> {
    // v1 files have snapshot_length > 0; v2 files set it to 0 and put the
    // directory CRC in the checksum field. An uninitialized header (iteration
    // == 0) means no data has been written yet.
    if active_header.is_empty() || active_header.snapshot_length > 0 {
        return Ok(None);
    }
    let expected_checksum = active_header.checksum;

    // Past this point the header asserts v2: any failure to read or parse
    // the directory is real corruption, not a v1/v2 misdetection. Surface
    // it instead of silently falling through to the snapshot read, where v1
    // CRC logic would mask the underlying cause.
    let file_size = file.metadata()?.len();
    if file_size < DIRECTORY_OFFSET + 4096 {
        return Err(Error::Internal(format!(
            "v2 header indicates section directory at offset {DIRECTORY_OFFSET:#X}, \
             but file is only {file_size} bytes",
        )));
    }

    file.seek(SeekFrom::Start(DIRECTORY_OFFSET))?;

    let mut buf = vec![0u8; 4096];
    file.read_exact(&mut buf)?;

    let dir = SectionDirectory::from_bytes(&buf).map_err(|e| {
        Error::Internal(format!(
            "v2 section directory at offset {DIRECTORY_OFFSET:#X} failed to parse: {e}",
        ))
    })?;

    // Cross-check the directory bytes against the CRC the writer recorded
    // in the active header. A mismatch means the directory page is torn or
    // corrupted (e.g. a partial write from a crashed checkpoint), not a
    // format ambiguity.
    let actual_checksum = crc32fast::hash(&buf);
    if actual_checksum != expected_checksum {
        return Err(Error::Internal(format!(
            "v2 section directory checksum mismatch: \
             header recorded {expected_checksum:#010X}, computed {actual_checksum:#010X}",
        )));
    }

    if dir.is_empty() {
        return Ok(None);
    }
    Ok(Some(dir))
}

/// Reads one v2 section's data using its directory entry: verifies the CRC
/// of the stored bytes, then decrypts them when a cipher is given (AAD
/// `grafeo-section:{section type}`).
///
/// # Errors
///
/// Returns an error if the section lies beyond the end of the file (checked
/// before its buffer is allocated, so a corrupt length never allocates), the
/// read fails, the CRC checksum does not match, or the decryption fails.
pub(super) fn read_section(
    file: &mut File,
    entry: &SectionDirectoryEntry,
    cipher: Option<&ChunkCipher>,
) -> Result<Vec<u8>> {
    let file_length = file.metadata()?.len();
    if entry
        .offset
        .checked_add(entry.length)
        .is_none_or(|end| end > file_length)
    {
        return Err(Error::Internal(format!(
            "section {:?} at offset {} (length {}) lies beyond the end of the file \
             ({file_length} bytes)",
            entry.section_type, entry.offset, entry.length
        )));
    }
    file.seek(SeekFrom::Start(entry.offset))?;

    let length = usize::try_from(entry.length).map_err(|_| {
        Error::Internal(format!(
            "section {:?} length {} does not fit in memory on this platform",
            entry.section_type, entry.length
        ))
    })?;
    let mut data = vec![0u8; length];
    file.read_exact(&mut data)?;

    // Verify CRC on the raw bytes (encrypted or plaintext)
    let actual_crc = crc32fast::hash(&data);
    if actual_crc != entry.checksum {
        return Err(Error::Internal(format!(
            "section {:?} CRC mismatch: expected {:#010X}, got {actual_crc:#010X}",
            entry.section_type, entry.checksum
        )));
    }

    #[cfg(feature = "encryption")]
    if let Some(cipher) = cipher {
        let aad = format!("grafeo-section:{}", entry.section_type as u32);
        return cipher.decrypt(&data, aad.as_bytes()).map_err(|_| {
            Error::Internal(format!(
                "section {:?} decryption failed: wrong key or corrupted data",
                entry.section_type
            ))
        });
    }
    #[cfg(not(feature = "encryption"))]
    if let Some(cipher) = cipher {
        match *cipher {}
    }

    Ok(data)
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::{Seek, SeekFrom, Write};
    use std::path::{Path, PathBuf};

    use grafeo_common::storage::{SectionDirectoryEntry, SectionType};
    use tempfile::TempDir;

    use super::{LegacyContents, LegacyFile};
    use crate::container::SectionDirectory;
    use crate::container::directory::{DIRECTORY_OFFSET, SECTION_DATA_OFFSET};
    use crate::file::format::{DATA_OFFSET, DbHeader, FileHeader};
    use crate::file::header::{write_db_header, write_file_header};

    // 0.5.x wrote these layouts; 0.6.x only reads them, so the tests build them.

    /// Writes a 0.5.x file header, an empty slot 0 and `active` in slot 1.
    fn write_headers(file: &mut File, active: &DbHeader) {
        write_file_header(file, &FileHeader::new()).unwrap();
        write_db_header(file, 0, &DbHeader::EMPTY).unwrap();
        write_db_header(file, 1, active).unwrap();
    }

    /// A 0.5.x file that was never checkpointed.
    fn write_empty(path: &Path) {
        write_headers(&mut File::create(path).unwrap(), &DbHeader::EMPTY);
    }

    /// A 0.5.x container v1 file: one snapshot blob.
    fn write_v1(path: &Path, snapshot: &[u8]) {
        let mut file = File::create(path).unwrap();
        write_headers(
            &mut file,
            &DbHeader {
                iteration: 1,
                checksum: crc32fast::hash(snapshot),
                snapshot_length: u64::try_from(snapshot.len()).unwrap(),
                epoch: 1,
                ..DbHeader::EMPTY
            },
        );
        file.seek(SeekFrom::Start(DATA_OFFSET)).unwrap();
        file.write_all(snapshot).unwrap();
    }

    /// A 0.5.x container v2 file: a section directory and the sections.
    fn write_v2(path: &Path, sections: &[(SectionType, &[u8])]) {
        let mut file = File::create(path).unwrap();
        let mut directory = SectionDirectory::new();
        let mut offset = SECTION_DATA_OFFSET;
        for (section_type, data) in sections {
            let length = u64::try_from(data.len()).unwrap();
            file.seek(SeekFrom::Start(offset)).unwrap();
            file.write_all(data).unwrap();
            directory
                .upsert(SectionDirectoryEntry {
                    section_type: *section_type,
                    version: 1,
                    flags: section_type.default_flags(),
                    offset,
                    length,
                    checksum: crc32fast::hash(data),
                })
                .unwrap();
            offset = (offset + length).next_multiple_of(4096);
        }
        file.seek(SeekFrom::Start(DIRECTORY_OFFSET)).unwrap();
        file.write_all(&directory.to_bytes()).unwrap();
        write_headers(
            &mut file,
            &DbHeader {
                iteration: 1,
                checksum: directory.checksum(),
                epoch: 1,
                ..DbHeader::EMPTY
            },
        );
    }

    fn released(version: &str) -> PathBuf {
        PathBuf::from(format!(
            "{}/../grafeo-engine/tests/fixtures/released/{version}/closed.grafeo",
            env!("CARGO_MANIFEST_DIR")
        ))
    }

    /// Copies a released fixture into a temp dir.
    fn copy_fixture(version: &str) -> (TempDir, PathBuf) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("closed.grafeo");
        std::fs::copy(released(version), &path).unwrap();
        (dir, path)
    }

    fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
        let mut name = path.as_os_str().to_owned();
        name.push(suffix);
        PathBuf::from(name)
    }

    fn assert_catalog_and_lpg(contents: LegacyContents) {
        let LegacyContents::Sections(sections) = contents else {
            panic!("a 0.5.x closed database holds sections, got {contents:?}");
        };
        for wanted in [SectionType::Catalog, SectionType::LpgStore] {
            assert!(
                sections.iter().any(|(kind, _)| *kind == wanted),
                "missing section {wanted:?}"
            );
        }
    }

    #[test]
    fn the_released_0_5_44_file_reads_as_sections() {
        let (_dir, path) = copy_fixture("0.5.44");
        let file = LegacyFile::open(&path, None).unwrap();
        assert_catalog_and_lpg(file.contents().unwrap());
        assert_eq!(file.sidecar_wal_path(), with_suffix(&path, ".wal"));
    }

    #[test]
    fn the_released_0_5_43_file_reads_as_sections() {
        let (_dir, path) = copy_fixture("0.5.43");
        let file = LegacyFile::open(&path, None).unwrap();
        assert_catalog_and_lpg(file.contents().unwrap());
    }

    #[test]
    fn opening_and_reading_never_change_the_file() {
        let (_dir, path) = copy_fixture("0.5.44");
        let before = std::fs::read(&path).unwrap();
        {
            let file = LegacyFile::open(&path, None).unwrap();
            file.contents().unwrap();
        }
        assert_eq!(std::fs::read(&path).unwrap(), before);
        assert!(!with_suffix(&path, ".checkpoint").exists());
    }

    #[test]
    fn a_pending_checkpoint_image_is_read_instead_of_the_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("db.grafeo");
        let other = dir.path().join("other.grafeo");
        write_v2(&path, &[(SectionType::LpgStore, b"Amsterdam")]);
        write_v2(&other, &[(SectionType::LpgStore, b"Berlin")]);
        let pending = with_suffix(&path, ".checkpoint");
        std::fs::rename(&other, &pending).unwrap();
        let before = std::fs::read(&path).unwrap();

        {
            let file = LegacyFile::open(&path, None).unwrap();
            assert_eq!(
                file.contents().unwrap(),
                LegacyContents::Sections(vec![(SectionType::LpgStore, b"Berlin".to_vec())])
            );
        }
        assert!(pending.exists(), "a reader leaves the image in place");
        assert_eq!(std::fs::read(&path).unwrap(), before);
    }

    #[test]
    fn a_never_checkpointed_file_is_empty() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("db.grafeo");
        write_empty(&path);
        let file = LegacyFile::open(&path, None).unwrap();
        assert_eq!(file.contents().unwrap(), LegacyContents::Empty);
    }

    /// A corrupt v2 directory entry can claim a section far longer than the
    /// file: the read fails with the section and its offset before it
    /// allocates a buffer of that length.
    #[test]
    fn a_section_longer_than_the_file_is_refused_before_it_is_read() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("db.grafeo");
        write_v2(&path, &[(SectionType::LpgStore, b"Amsterdam")]);
        let file_length = std::fs::metadata(&path).unwrap().len();
        let remaining = file_length - SECTION_DATA_OFFSET;
        // Past the end by one byte, a terabyte, and an end beyond `u64`.
        for length in [remaining + 1, 1 << 40, u64::MAX - 88] {
            let mut directory = SectionDirectory::new();
            directory
                .upsert(SectionDirectoryEntry {
                    section_type: SectionType::LpgStore,
                    version: 1,
                    flags: SectionType::LpgStore.default_flags(),
                    offset: SECTION_DATA_OFFSET,
                    length,
                    checksum: crc32fast::hash(b"Amsterdam"),
                })
                .unwrap();
            let mut file = File::options().write(true).open(&path).unwrap();
            file.seek(SeekFrom::Start(DIRECTORY_OFFSET)).unwrap();
            file.write_all(&directory.to_bytes()).unwrap();
            write_headers(
                &mut file,
                &DbHeader {
                    iteration: 1,
                    checksum: directory.checksum(),
                    epoch: 1,
                    ..DbHeader::EMPTY
                },
            );
            drop(file);

            let legacy = LegacyFile::open(&path, None).unwrap();
            let error = legacy
                .contents()
                .expect_err("a section beyond the end of the file")
                .to_string();
            assert!(
                error.contains("LpgStore")
                    && error.contains(&SECTION_DATA_OFFSET.to_string())
                    && error.contains(&file_length.to_string()),
                "length {length}: the error names the section, its offset and the file \
                 length: {error}"
            );
        }
    }

    /// A damaged v1 header (0.5.x headers have no checksum) can claim a
    /// snapshot far longer than the file: the read fails with its offset,
    /// length and the file size before it allocates a buffer of that length.
    #[test]
    fn a_snapshot_longer_than_the_file_is_refused_before_it_is_read() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("db.grafeo");
        write_v1(&path, b"Vincent");
        let file_length = std::fs::metadata(&path).unwrap().len();
        let remaining = file_length - DATA_OFFSET;
        // Past the end by one byte, a terabyte, and an end beyond `u64`.
        for length in [remaining + 1, 1 << 40, u64::MAX - 88] {
            let mut file = File::options().write(true).open(&path).unwrap();
            write_headers(
                &mut file,
                &DbHeader {
                    iteration: 1,
                    checksum: crc32fast::hash(b"Vincent"),
                    snapshot_length: length,
                    epoch: 1,
                    ..DbHeader::EMPTY
                },
            );
            drop(file);

            let legacy = LegacyFile::open(&path, None).unwrap();
            let error = legacy
                .contents()
                .expect_err("a snapshot beyond the end of the file")
                .to_string();
            assert!(
                error.contains(&DATA_OFFSET.to_string())
                    && error.contains(&length.to_string())
                    && error.contains(&file_length.to_string()),
                "length {length}: the error names the offset, the length and the file \
                 length: {error}"
            );
        }
    }

    #[test]
    fn a_v1_snapshot_reads_as_a_snapshot() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("db.grafeo");
        write_v1(&path, b"Vincent");
        let file = LegacyFile::open(&path, None).unwrap();
        assert_eq!(
            file.contents().unwrap(),
            LegacyContents::Snapshot(b"Vincent".to_vec())
        );
    }
}
