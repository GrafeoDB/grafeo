//! High-level manager for `.grafeo` database files.
//!
//! [`GrafeoFileManager`] owns the file handle and provides create, open,
//! snapshot write/read, and sidecar WAL lifecycle management.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use fs2::FileExt;
use grafeo_common::testing::child_process;
use grafeo_common::utils::error::{Error, Result};
use parking_lot::{Mutex, MutexGuard};

use super::format::{DATA_OFFSET, DbHeader, FileHeader};
use super::header;

/// Manages a single `.grafeo` database file.
///
/// # Lifecycle
///
/// 1. [`create`](Self::create) or [`open`](Self::open)
/// 2. Mutations flow through a sidecar WAL (managed externally by the engine)
/// 3. [`write_snapshot`](Self::write_snapshot) checkpoints memory to the file;
///    periodic checkpoints retain the sidecar WAL
/// 4. On a clean shutdown, after the final checkpoint, call
///    [`remove_sidecar_wal`](Self::remove_sidecar_wal) to discard the sidecar
/// 5. [`close`](Self::close) (or drop) releases the file handle
pub struct GrafeoFileManager {
    /// Path to the `.grafeo` file.
    path: PathBuf,
    /// Open file handle (read/write or read-only).
    file: Mutex<File>,
    /// File header (read once on open, immutable afterwards).
    file_header: FileHeader,
    /// Currently active database header.
    active_header: Mutex<DbHeader>,
    /// Slot index (0 or 1) of the active header.
    active_slot: Mutex<u8>,
    /// Whether this manager was opened in read-only mode.
    read_only: bool,
    /// Held for a whole checkpoint, see [`checkpoint_guard`](Self::checkpoint_guard).
    checkpoint_lock: Mutex<()>,
    /// A checkpoint image a failed step of this process left on disk.
    leftover_image: Mutex<LeftoverImage>,
    /// The database file, when `file` is a pending checkpoint image instead
    /// (read-only open, see [`open_read_only`](Self::open_read_only)). It
    /// carries the shared lock.
    lock_holder: Option<File>,
    /// Encryptor for section data (None = unencrypted).
    #[cfg(feature = "encryption")]
    section_encryptor: Option<grafeo_common::encryption::PageEncryptor>,
}

/// A checkpoint image that [`GrafeoFileManager::write_sections`] could not
/// finish with. An open finishes any image it finds, so in a read-write
/// manager only a failed step of this process leaves one behind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LeftoverImage {
    /// No image on disk.
    None,
    /// Installing it failed, so the database file may be half-written:
    /// every read and write installs it first.
    NotInstalled,
    /// Installed, but removing it failed. The next open would install it
    /// again, over anything written since: every write removes it first.
    NotRemoved,
}

impl GrafeoFileManager {
    /// Creates a new `.grafeo` file at `path`.
    ///
    /// Writes the file header and two empty database headers. The file must
    /// not already exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the file already exists or cannot be created.
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        if path.exists() {
            return Err(Error::Internal(format!(
                "file already exists: {}",
                path.display()
            )));
        }

        // Ensure parent directory exists
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }

        // Checkpoint side files next to a missing database file belong to a
        // deleted database: a pending image must not be installed over the
        // new one later.
        remove_image(&checkpoint_image_path(&path))?;
        remove_if_exists(&checkpoint_tmp_path(&path))?;

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::AlreadyExists || e.raw_os_error() == Some(183) {
                    Error::Io(std::io::Error::new(
                        std::io::ErrorKind::AlreadyExists,
                        format!(
                            "database file already exists (may be open by another process): {}",
                            path.display()
                        ),
                    ))
                } else {
                    Error::Io(e)
                }
            })?;

        // Acquire an exclusive lock: prevents other processes from opening the same file
        child_process::take_lock(|| file.try_lock_exclusive(), is_lock_contended).map_err(
            |_| {
                Error::Internal(format!(
                    "database file is locked by another process: {}",
                    path.display()
                ))
            },
        )?;

        let file_header = FileHeader::new();
        header::write_file_header(&mut file, &file_header)?;
        header::write_db_header(&mut file, 0, &DbHeader::EMPTY)?;
        header::write_db_header(&mut file, 1, &DbHeader::EMPTY)?;
        file.sync_all()?;

        Ok(Self {
            path,
            file: Mutex::new(file),
            file_header,
            active_header: Mutex::new(DbHeader::EMPTY),
            active_slot: Mutex::new(0),
            read_only: false,
            checkpoint_lock: Mutex::new(()),
            leftover_image: Mutex::new(LeftoverImage::None),
            lock_holder: None,
            #[cfg(feature = "encryption")]
            section_encryptor: None,
        })
    }

    /// Opens an existing `.grafeo` file.
    ///
    /// Validates the magic bytes and format version, then selects the
    /// active database header.
    ///
    /// # Errors
    ///
    /// Returns an error if the file does not exist, has invalid magic, or
    /// an unsupported format version.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        let mut file = OpenOptions::new().read(true).write(true).open(&path)?;

        // Acquire an exclusive lock: prevents other processes from opening the same file
        child_process::take_lock(|| file.try_lock_exclusive(), is_lock_contended).map_err(
            |_| {
                Error::Internal(format!(
                    "database file is locked by another process: {}",
                    path.display()
                ))
            },
        )?;

        finish_interrupted_checkpoint(&path, &mut file)?;

        let file_header = header::read_file_header(&mut file)?;
        header::validate_file_header(&file_header)?;

        let (h0, h1) = header::read_db_headers(&mut file)?;
        let (active_slot, active_header) = header::active_db_header(&h0, &h1);

        Ok(Self {
            path,
            file: Mutex::new(file),
            file_header,
            active_header: Mutex::new(active_header),
            active_slot: Mutex::new(active_slot),
            read_only: false,
            checkpoint_lock: Mutex::new(()),
            leftover_image: Mutex::new(LeftoverImage::None),
            lock_holder: None,
            #[cfg(feature = "encryption")]
            section_encryptor: None,
        })
    }

    /// Opens an existing `.grafeo` file in read-only mode.
    ///
    /// Uses a **shared** file lock (`try_lock_shared`), allowing multiple
    /// readers to open the same file concurrently, even while a writer holds
    /// an exclusive lock (on platforms with advisory locking).
    ///
    /// The returned manager only supports [`read_snapshot`](Self::read_snapshot)
    /// and other read-only operations. Calling [`write_snapshot`](Self::write_snapshot)
    /// will return an error.
    ///
    /// If a checkpoint was interrupted while installing its new image, a
    /// reader cannot finish it: it reads that complete image instead of the
    /// database file and leaves it for the next read-write open.
    ///
    /// # Errors
    ///
    /// Returns an error if the file does not exist, has invalid magic, or
    /// an unsupported format version.
    pub fn open_read_only(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        let database_file = OpenOptions::new().read(true).open(&path)?;

        // Acquire a shared lock: coexists with other shared locks but
        // blocks if an exclusive lock cannot be shared (platform-dependent).
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

        let pending_image = checkpoint_image_path(&path);
        let (mut file, lock_holder) = if pending_image.exists() {
            (File::open(&pending_image)?, Some(database_file))
        } else {
            (database_file, None)
        };

        let file_header = header::read_file_header(&mut file)?;
        header::validate_file_header(&file_header)?;

        let (h0, h1) = header::read_db_headers(&mut file)?;
        let (active_slot, active_header) = header::active_db_header(&h0, &h1);

        Ok(Self {
            path,
            file: Mutex::new(file),
            file_header,
            active_header: Mutex::new(active_header),
            active_slot: Mutex::new(active_slot),
            read_only: true,
            checkpoint_lock: Mutex::new(()),
            leftover_image: Mutex::new(LeftoverImage::None),
            lock_holder,
            #[cfg(feature = "encryption")]
            section_encryptor: None,
        })
    }

    /// Sets the encryptor for section-level encryption.
    ///
    /// When set, all section data is encrypted on write and decrypted on read.
    /// The GCM authentication tag provides integrity verification, replacing
    /// the CRC-32 checksum for encrypted sections.
    #[cfg(feature = "encryption")]
    pub fn set_section_encryptor(&mut self, encryptor: grafeo_common::encryption::PageEncryptor) {
        self.section_encryptor = Some(encryptor);
    }

    /// Returns `true` if this manager was opened in read-only mode.
    #[must_use]
    pub fn is_read_only(&self) -> bool {
        self.read_only
    }

    /// Serializes checkpoints of this file.
    ///
    /// A checkpoint rotates the WAL, takes its snapshot, writes the image and
    /// then marks and truncates the WAL. Two checkpoints interleaving those
    /// steps (for example the periodic timer and an explicit checkpoint) could
    /// delete WAL files the other one still relies on, so the caller holds
    /// this guard for the whole sequence.
    pub fn checkpoint_guard(&self) -> parking_lot::MutexGuard<'_, ()> {
        self.checkpoint_lock.lock()
    }

    /// Writes snapshot data into the file and updates the inactive DB header.
    ///
    /// Steps:
    /// 1. Write `data` at [`DATA_OFFSET`]
    /// 2. Compute CRC-32 checksum
    /// 3. Build a new [`DbHeader`] and write it to the inactive slot
    /// 4. `fsync` the file
    /// 5. Update internal active header/slot state
    ///
    /// # Errors
    ///
    /// Returns an error if any I/O operation fails.
    pub fn write_snapshot(
        &self,
        data: &[u8],
        epoch: u64,
        transaction_id: u64,
        node_count: u64,
        edge_count: u64,
    ) -> Result<()> {
        if self.read_only {
            return Err(Error::Internal(
                "cannot write snapshot: database is open in read-only mode".to_string(),
            ));
        }

        use grafeo_common::testing::crash::maybe_crash;

        let checksum = crc32fast::hash(data);
        // reason: millis since UNIX epoch fits in u64 for ~585 million years
        #[allow(clippy::cast_possible_truncation)]
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let mut file = self.file.lock();
        self.clear_leftover_image(&mut file)?;
        let active_header = self.active_header.lock();
        let mut active_slot = self.active_slot.lock();

        let new_iteration = active_header.iteration + 1;
        let target_slot = u8::from(*active_slot == 0);

        maybe_crash("write_snapshot:before_data_write");

        // Write snapshot data
        file.seek(SeekFrom::Start(DATA_OFFSET))?;
        file.write_all(data)?;

        maybe_crash("write_snapshot:after_data_write");

        // Truncate file to exact size (remove stale trailing data)
        let file_end = DATA_OFFSET + data.len() as u64;
        file.set_len(file_end)?;

        maybe_crash("write_snapshot:after_truncate");

        // Build and write new header to inactive slot
        let new_header = DbHeader {
            iteration: new_iteration,
            checksum,
            snapshot_length: data.len() as u64,
            epoch,
            transaction_id,
            node_count,
            edge_count,
            timestamp_ms,
        };
        header::write_db_header(&mut file, target_slot, &new_header)?;

        maybe_crash("write_snapshot:after_header_write");

        // Ensure everything is on disk before we consider this committed
        file.sync_all()?;

        maybe_crash("write_snapshot:after_fsync");

        // Update internal state: drop the old lock, reacquire to update
        drop(active_header);
        *self.active_header.lock() = new_header;
        *active_slot = target_slot;

        Ok(())
    }

    /// Reads snapshot data from the file using the active database header.
    ///
    /// Returns an empty `Vec` if the database has never been checkpointed
    /// (both headers are empty).
    ///
    /// # Errors
    ///
    /// Returns an error if the read fails or the CRC checksum does not match.
    pub fn read_snapshot(&self) -> Result<Vec<u8>> {
        let mut file = self.lock_for_read()?;
        let active_header = self.active_header.lock();

        if active_header.is_empty() {
            return Ok(Vec::new());
        }

        // v2 files store sections rather than a v1 snapshot blob. They set
        // snapshot_length == 0 and put the directory CRC in the checksum field.
        // Reading 0 bytes here would CRC to 0 and mismatch the directory CRC.
        if active_header.snapshot_length == 0 {
            return Ok(Vec::new());
        }

        // reason: snapshot_length is the size of serialized in-memory data, fits in usize on 64-bit targets;
        // on 32-bit targets the database would OOM long before reaching 4 GiB
        // reason: value bounded by collection size, fits usize
        #[allow(clippy::cast_possible_truncation)]
        let length = active_header.snapshot_length as usize;
        let expected_checksum = active_header.checksum;
        drop(active_header);

        file.seek(SeekFrom::Start(DATA_OFFSET))?;

        let mut data = vec![0u8; length];
        std::io::Read::read_exact(&mut *file, &mut data)?;

        // Verify CRC
        let actual_checksum = crc32fast::hash(&data);
        if actual_checksum != expected_checksum {
            return Err(Error::Internal(format!(
                "snapshot checksum mismatch: expected {expected_checksum:#010X}, got {actual_checksum:#010X}"
            )));
        }

        Ok(data)
    }

    /// Returns the path for the sidecar WAL directory.
    ///
    /// For a database at `mydb.grafeo`, the sidecar is `mydb.grafeo.wal/`.
    #[must_use]
    pub fn sidecar_wal_path(&self) -> PathBuf {
        let mut wal_path = self.path.as_os_str().to_owned();
        wal_path.push(".wal");
        PathBuf::from(wal_path)
    }

    /// Returns `true` if a sidecar WAL directory exists.
    #[must_use]
    pub fn has_sidecar_wal(&self) -> bool {
        self.sidecar_wal_path().exists()
    }

    /// Removes the sidecar WAL directory after the final checkpoint on a clean
    /// shutdown. Periodic checkpoints retain the sidecar so an unexpected
    /// restart can still recover in-flight mutations; only an orderly `close()`
    /// discards it.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory exists but cannot be removed.
    pub fn remove_sidecar_wal(&self) -> Result<()> {
        let wal_path = self.sidecar_wal_path();
        if wal_path.exists() {
            fs::remove_dir_all(&wal_path)?;
        }
        Ok(())
    }

    /// Returns the file path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns a clone of the currently active database header.
    #[must_use]
    pub fn active_header(&self) -> DbHeader {
        self.active_header.lock().clone()
    }

    /// Returns the file header (written at creation, immutable).
    #[must_use]
    pub fn file_header(&self) -> &FileHeader {
        &self.file_header
    }

    /// Returns the total file size on disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the file metadata cannot be read.
    pub fn file_size(&self) -> Result<u64> {
        let file = self.file.lock();
        let metadata = file.metadata()?;
        Ok(metadata.len())
    }

    /// Flushes and syncs the file.
    ///
    /// # Errors
    ///
    /// Returns an error if sync fails.
    pub fn sync(&self) -> Result<()> {
        if !self.read_only {
            let file = self.file.lock();
            file.sync_all()?;
        }
        Ok(())
    }

    // ── Section-based I/O (v2 container format) ─────────────────────

    /// Writes multiple sections to the file using the v2 container format.
    ///
    /// Each section is written at a page-aligned offset. A section directory
    /// is written at `DIRECTORY_OFFSET`, and a new DbHeader is committed to
    /// the inactive slot.
    ///
    /// The database file is never overwritten before a complete copy of the
    /// new image exists (#418):
    ///
    /// 1. the new image is written to `<file>.checkpoint.tmp` and synced,
    /// 2. it is renamed to `<file>.checkpoint`: from now on an open finishes
    ///    installing it (see [`open`](Self::open)),
    /// 3. it is copied over the database file, which is synced,
    /// 4. `<file>.checkpoint` is removed.
    ///
    /// A failure or crash before step 2 leaves the database file untouched;
    /// after it, the next open installs the image, and until then this
    /// manager installs it before its next read or write. Once step 3 is
    /// done the checkpoint has succeeded: if step 4 fails, the next write
    /// removes the image. This needs room for a second copy of the file
    /// while it runs.
    ///
    /// # Errors
    ///
    /// Returns an error if write or sync fails.
    pub fn write_sections(
        &self,
        sections: &[(grafeo_common::storage::SectionType, &[u8])],
        epoch: u64,
        transaction_id: u64,
        node_count: u64,
        edge_count: u64,
    ) -> Result<()> {
        use grafeo_common::testing::crash::maybe_crash;

        if self.read_only {
            return Err(Error::Internal(
                "cannot write sections: database is open in read-only mode".to_string(),
            ));
        }

        let mut file = self.file.lock();
        let tmp_path = checkpoint_tmp_path(&self.path);
        let image_path = checkpoint_image_path(&self.path);

        // The new image starts from the current file, which an earlier
        // failed checkpoint may have left half-written.
        self.clear_leftover_image(&mut file)?;

        let active_header = self.active_header.lock();
        let mut active_slot = self.active_slot.lock();

        // Step 1.
        let (new_header, target_slot) = match self.write_image(
            &mut file,
            &tmp_path,
            sections,
            &active_header,
            *active_slot,
            (epoch, transaction_id, node_count, edge_count),
        ) {
            Ok(written) => written,
            Err(e) => {
                // Do not leave a partial image taking up space (full disk).
                let _ = fs::remove_file(&tmp_path);
                return Err(e);
            }
        };

        maybe_crash("checkpoint:after_image");

        // Step 2.
        fs::rename(&tmp_path, &image_path)?;
        *self.leftover_image.lock() = LeftoverImage::NotInstalled;
        sync_parent_dir(&image_path)?;

        maybe_crash("checkpoint:after_rename");

        // Step 3.
        install_image(&mut file, &image_path)?;
        drop(active_header);
        *self.active_header.lock() = new_header;
        *active_slot = target_slot;
        drop(active_slot);
        *self.leftover_image.lock() = LeftoverImage::NotRemoved;

        maybe_crash("checkpoint:after_install");

        // Step 4.
        self.remove_installed_image();
        Ok(())
    }

    /// Locks the file for a read, first installing an image whose install
    /// failed, so no read or copy sees a half-written file.
    fn lock_for_read(&self) -> Result<MutexGuard<'_, File>> {
        let mut file = self.file.lock();
        self.finish_install(&mut file)?;
        Ok(file)
    }

    /// Installs an image whose install failed earlier and takes its active
    /// header. Called with the file lock held.
    fn finish_install(&self, file: &mut File) -> Result<()> {
        if *self.leftover_image.lock() != LeftoverImage::NotInstalled {
            return Ok(());
        }
        install_image(file, &checkpoint_image_path(&self.path))?;
        let (h0, h1) = header::read_db_headers(file)?;
        let (slot, header) = header::active_db_header(&h0, &h1);
        *self.active_header.lock() = header;
        *self.active_slot.lock() = slot;
        *self.leftover_image.lock() = LeftoverImage::NotRemoved;
        self.remove_installed_image();
        Ok(())
    }

    /// Before a write: installs a leftover image and removes it, since the
    /// next open would install it over the new data. Called with the file
    /// lock held.
    fn clear_leftover_image(&self, file: &mut File) -> Result<()> {
        self.finish_install(file)?;
        let mut leftover = self.leftover_image.lock();
        if *leftover == LeftoverImage::NotRemoved {
            remove_image(&checkpoint_image_path(&self.path))?;
            *leftover = LeftoverImage::None;
        }
        Ok(())
    }

    /// Removes an installed image. If that fails the file is still correct,
    /// so the caller carries on and the next write tries again.
    fn remove_installed_image(&self) {
        let image_path = checkpoint_image_path(&self.path);
        let mut leftover = self.leftover_image.lock();
        match remove_image(&image_path) {
            Ok(()) => *leftover = LeftoverImage::None,
            Err(e) => grafeo_common::grafeo_warn!(
                "could not remove the installed checkpoint image {}, the next write retries: {e}",
                image_path.display()
            ),
        }
    }

    /// Writes a complete new image of the database file to `image_path`: the
    /// current file and database headers, the sections, the directory and a
    /// new header in the inactive slot. Returns that header and its slot.
    fn write_image(
        &self,
        main: &mut File,
        image_path: &Path,
        sections: &[(grafeo_common::storage::SectionType, &[u8])],
        active_header: &DbHeader,
        active_slot: u8,
        (epoch, transaction_id, node_count, edge_count): (u64, u64, u64, u64),
    ) -> Result<(DbHeader, u8)> {
        use crate::container::SectionDirectory;
        use crate::container::directory::{DIRECTORY_OFFSET, SECTION_DATA_OFFSET};
        use grafeo_common::storage::SectionDirectoryEntry;
        use grafeo_common::testing::crash::maybe_crash;

        let mut options = OpenOptions::new();
        options.read(true).write(true).create(true).truncate(true);
        // The image holds the whole database: give it the database file's
        // permissions, from creation on, and also to an old file it reuses.
        #[cfg(unix)]
        let permissions = main.metadata()?.permissions();
        #[cfg(unix)]
        {
            use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
            options.mode(permissions.mode() & 0o7777);
        }
        let mut image = options.open(image_path)?;
        #[cfg(unix)]
        image.set_permissions(permissions)?;

        // Start from the current file header and both database headers.
        // reason: DIRECTORY_OFFSET is 12 KiB
        #[allow(clippy::cast_possible_truncation)]
        let mut headers = vec![0u8; DIRECTORY_OFFSET as usize];
        main.seek(SeekFrom::Start(0))?;
        main.read_exact(&mut headers)?;
        image.write_all(&headers)?;

        let mut dir = SectionDirectory::new();

        maybe_crash("write_sections:before_data");

        // Write each section at page-aligned offsets
        let page_size = 4096u64;
        let mut current_offset = SECTION_DATA_OFFSET;
        // Next checkpoint iteration, used as the high part of the nonce so that
        // the same (section_type, offset) pair produces a different nonce across
        // checkpoints. Without this, identical section layouts would reuse nonces.
        #[cfg(feature = "encryption")]
        // reason: iteration wraps at u32::MAX which takes billions of checkpoints (~100+ years at 1/s)
        #[allow(clippy::cast_possible_truncation)]
        let nonce_iteration = (active_header.iteration + 1) as u32;

        for (section_type, data) in sections {
            // Encrypt section data if encryption is enabled.
            // Nonce high word: iteration in bits [31:8], section type in bits [7:0].
            // Bit-packing (not XOR) ensures unique high words: XOR is commutative
            // so `iter ^ type` can collide across different (iter, type) pairs,
            // but packing into disjoint bit lanes is injective for type < 256.
            // Nonce low word: page-aligned write offset (unique within a checkpoint).
            // AAD binds the ciphertext to the section type, preventing relocation.
            // Encrypt section data if an encryptor is configured, otherwise
            // write the plaintext bytes directly (no allocation).
            #[cfg(feature = "encryption")]
            let encrypted_buf: Option<Vec<u8>> = if let Some(ref enc) = self.section_encryptor {
                let nonce_high = (nonce_iteration << 8) | (*section_type as u32 & 0xFF);
                let nonce = grafeo_common::encryption::build_nonce(nonce_high, current_offset);
                let aad = format!("grafeo-section:{}", *section_type as u32);
                Some(
                    enc.encrypt(data, &nonce, aad.as_bytes())
                        .map_err(|e| Error::Internal(format!("section encryption failed: {e}")))?,
                )
            } else {
                None
            };

            #[cfg(feature = "encryption")]
            let write_data: &[u8] = encrypted_buf.as_deref().unwrap_or(data);
            #[cfg(not(feature = "encryption"))]
            let write_data: &[u8] = data;

            let checksum = crc32fast::hash(write_data);
            let length = write_data.len() as u64;

            image.seek(SeekFrom::Start(current_offset))?;
            image.write_all(write_data)?;

            dir.upsert(SectionDirectoryEntry {
                section_type: *section_type,
                version: 1,
                flags: section_type.default_flags(),
                offset: current_offset,
                length,
                checksum,
            })?;

            // Align next section to page boundary
            let section_end = current_offset + length;
            current_offset = (section_end + page_size - 1) / page_size * page_size;
        }

        maybe_crash("write_sections:after_data");

        // Cut the image right after the last section
        image.set_len(current_offset)?;

        // Write section directory
        let dir_bytes = dir.to_bytes();
        image.seek(SeekFrom::Start(DIRECTORY_OFFSET))?;
        image.write_all(&dir_bytes)?;

        maybe_crash("write_sections:after_directory");

        // Build and write new DbHeader to inactive slot
        let new_iteration = active_header.iteration + 1;
        let target_slot = u8::from(active_slot == 0);
        // reason: millis since UNIX epoch fits in u64 for ~585 million years
        #[allow(clippy::cast_possible_truncation)]
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let new_header = DbHeader {
            iteration: new_iteration,
            checksum: dir.checksum(),
            snapshot_length: 0, // Not used in v2; directory CRC is in checksum field
            epoch,
            transaction_id,
            node_count,
            edge_count,
            timestamp_ms,
        };
        header::write_db_header(&mut image, target_slot, &new_header)?;

        // Ensure everything is on disk
        image.sync_all()?;

        maybe_crash("write_sections:after_fsync");

        Ok((new_header, target_slot))
    }

    /// Reads the section directory from the file.
    ///
    /// Detects v2 format by checking the `snapshot_length` field in the active
    /// DbHeader: v2 writes set `snapshot_length = 0`, while v1 always has a
    /// non-zero snapshot length when data exists.
    ///
    /// Returns `None` only when the file is unambiguously v1
    /// (`snapshot_length` non-zero) or uninitialized (header iteration is 0).
    /// Once the header asserts v2, any failure to locate or parse the
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
    pub fn read_section_directory(&self) -> Result<Option<crate::container::SectionDirectory>> {
        use crate::container::SectionDirectory;
        use crate::container::directory::DIRECTORY_OFFSET;

        let mut file = self.lock_for_read()?;
        let active_header = self.active_header.lock();

        // v1 files have snapshot_length > 0; v2 files set it to 0 and put the
        // directory CRC in the checksum field. An uninitialized header (iteration
        // == 0) means no data has been written yet.
        if active_header.is_empty() || active_header.snapshot_length > 0 {
            return Ok(None);
        }
        let expected_checksum = active_header.checksum;
        drop(active_header);

        // Past this point the header asserts v2: any failure to read or parse
        // the directory is real corruption, not a v1/v2 misdetection. Surface
        // it instead of silently falling through to read_snapshot, where v1 CRC
        // logic would mask the underlying cause.
        let file_size = file.metadata()?.len();
        if file_size < DIRECTORY_OFFSET + 4096 {
            return Err(Error::Internal(format!(
                "v2 header indicates section directory at offset {DIRECTORY_OFFSET:#X}, \
                 but file is only {file_size} bytes",
            )));
        }

        file.seek(SeekFrom::Start(DIRECTORY_OFFSET))?;

        let mut buf = vec![0u8; 4096];
        std::io::Read::read_exact(&mut *file, &mut buf)?;

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

    /// Reads a single section's data from the file.
    ///
    /// Uses the section directory entry to locate and verify the data.
    ///
    /// # Errors
    ///
    /// Returns an error if read fails or CRC checksum doesn't match.
    pub fn read_section_data(
        &self,
        entry: &grafeo_common::storage::SectionDirectoryEntry,
    ) -> Result<Vec<u8>> {
        let mut file = self.lock_for_read()?;
        file.seek(SeekFrom::Start(entry.offset))?;

        // reason: section length is bounded by file size, which fits in usize on 64-bit targets;
        // on 32-bit targets sections would OOM long before reaching 4 GiB
        // reason: value bounded by collection size, fits usize
        #[allow(clippy::cast_possible_truncation)]
        let mut data = vec![0u8; entry.length as usize];
        std::io::Read::read_exact(&mut *file, &mut data)?;

        // Verify CRC on the raw bytes (encrypted or plaintext)
        let actual_crc = crc32fast::hash(&data);
        if actual_crc != entry.checksum {
            return Err(Error::Internal(format!(
                "section {:?} CRC mismatch: expected {:#010X}, got {actual_crc:#010X}",
                entry.section_type, entry.checksum
            )));
        }

        // Decrypt if encryption is enabled
        #[cfg(feature = "encryption")]
        if let Some(ref enc) = self.section_encryptor {
            let aad = format!("grafeo-section:{}", entry.section_type as u32);
            return enc.decrypt(&data, aad.as_bytes()).map_err(|_| {
                Error::Internal(format!(
                    "section {:?} decryption failed: wrong key or corrupted data",
                    entry.section_type
                ))
            });
        }

        Ok(data)
    }

    /// Memory-maps a single section for zero-copy read access.
    ///
    /// The section's CRC-32 is verified against the mmap'd bytes before
    /// returning, which also warms the OS page cache. Only sections with
    /// `flags.mmap_able = true` can be mapped (index sections).
    ///
    /// The returned [`MmapSection`](crate::container::MmapSection) is
    /// independent of the file mutex: multiple mmaps can coexist. However,
    /// all `MmapSection` handles **must be dropped before writing** (via
    /// `write_sections()` or `write_snapshot()`). On Windows the OS rejects
    /// writes to a file with active mappings; on Linux/macOS stale mappings
    /// would read outdated data. See [`MmapSection`](crate::container::MmapSection)
    /// for the full lifecycle.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The section is not mmap-able (data section)
    /// - The mmap system call fails
    /// - The CRC-32 checksum does not match (corrupt data)
    #[allow(unsafe_code)]
    pub fn mmap_section(
        &self,
        entry: &grafeo_common::storage::SectionDirectoryEntry,
    ) -> Result<crate::container::MmapSection> {
        if !entry.flags.mmap_able {
            return Err(Error::Internal(format!(
                "section {:?} is not mmap-able (data sections must be deserialized)",
                entry.section_type
            )));
        }

        if entry.length == 0 {
            return Err(Error::Internal(format!(
                "section {:?} has zero length, cannot mmap",
                entry.section_type
            )));
        }

        let file = self.lock_for_read()?;

        // SAFETY: We hold an exclusive lock on the `.grafeo` file, preventing
        // concurrent modification by other processes. The mapping is read-only.
        // The section region [offset .. offset+length] was written by
        // write_sections() and its CRC is verified below before the mmap
        // is exposed to callers.
        // reason: section length is bounded by file size, fits in usize on 64-bit targets
        #[allow(clippy::cast_possible_truncation)]
        let section_len = entry.length as usize;
        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .offset(entry.offset)
                .len(section_len)
                .map(&*file)
        }
        .map_err(Error::Io)?;

        drop(file);

        // Verify CRC on the mmap'd bytes. This reads through the mapping,
        // which triggers page faults and warms the OS page cache: a free
        // prefetch disguised as an integrity check.
        let actual_crc = crc32fast::hash(&mmap);
        if actual_crc != entry.checksum {
            return Err(Error::Internal(format!(
                "section {:?} CRC mismatch: expected {:#010X}, got {actual_crc:#010X}",
                entry.section_type, entry.checksum
            )));
        }

        Ok(crate::container::MmapSection::new(
            mmap,
            entry.section_type,
            entry.checksum,
        ))
    }

    /// Copies the database file to `dest` using the already-locked file handle.
    ///
    /// `std::fs::copy()` opens the source with a new handle, which fails on
    /// Windows when an exclusive lock is held. This method reads through the
    /// existing handle, avoiding lock conflicts.
    ///
    /// # Errors
    ///
    /// Returns an error if the read or write fails.
    pub fn copy_to(&self, dest: &Path) -> Result<u64> {
        let mut file = self.lock_for_read()?;
        file.seek(SeekFrom::Start(0))?;

        let mut dest_file = fs::File::create(dest)?;
        let bytes = std::io::copy(&mut *file, &mut dest_file).map_err(Error::Io)?;
        dest_file.sync_all()?;
        Ok(bytes)
    }

    /// Releases the file lock and syncs.
    ///
    /// # Errors
    ///
    /// Returns an error if sync or unlock fails.
    pub fn close(&self) -> Result<()> {
        let file = self.file.lock();
        if !self.read_only {
            file.sync_all()?;
        }
        self.lock_carrier(&file)
            .unlock()
            .map_err(|e| Error::Internal(format!("failed to unlock database file: {e}")))?;
        Ok(())
    }

    /// The handle that holds this manager's file lock.
    fn lock_carrier<'a>(&'a self, file: &'a File) -> &'a File {
        self.lock_holder.as_ref().unwrap_or(file)
    }
}

impl Drop for GrafeoFileManager {
    fn drop(&mut self) {
        let file = self.file.lock();
        let _ = self.lock_carrier(&file).unlock();
    }
}

/// Where a checkpoint writes its new image until it is complete.
fn checkpoint_tmp_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".checkpoint.tmp");
    PathBuf::from(name)
}

/// A complete new image that a checkpoint is installing over the database
/// file. Its presence means the install has to be finished.
fn checkpoint_image_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".checkpoint");
    PathBuf::from(name)
}

fn remove_if_exists(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.into()),
    }
}

/// Removes a checkpoint image and makes the removal durable before the
/// caller writes anything newer: an image that came back after a power loss
/// would be installed over those writes at the next open.
fn remove_image(image: &Path) -> Result<()> {
    remove_if_exists(image)?;
    sync_parent_dir(image)
}

/// Copies the complete image at `image` over the database file and syncs it.
fn install_image(file: &mut File, image: &Path) -> Result<()> {
    use grafeo_common::testing::crash::maybe_crash;

    let mut source = File::open(image)?;
    let length = source.metadata()?.len();
    file.seek(SeekFrom::Start(0))?;
    std::io::copy(&mut source, file)?;

    maybe_crash("checkpoint:after_install_write");

    file.set_len(length)?;
    file.sync_all()?;
    Ok(())
}

/// Whether a failed `fs2` lock attempt failed because another handle holds
/// the lock (as opposed to an I/O error).
fn is_lock_contended(error: &std::io::Error) -> bool {
    error.raw_os_error() == fs2::lock_contended_error().raw_os_error()
}

/// Opening a database file for writing: an image that was still being
/// written is discarded (the database file was not touched yet), and a
/// complete image whose install was cut off is installed.
fn finish_interrupted_checkpoint(path: &Path, file: &mut File) -> Result<()> {
    remove_if_exists(&checkpoint_tmp_path(path))?;
    let image = checkpoint_image_path(path);
    if image.exists() {
        grafeo_common::grafeo_warn!(
            "finishing a checkpoint that was interrupted while installing {}",
            path.display()
        );
        install_image(file, &image)?;
        remove_image(&image)?;
    }
    Ok(())
}

/// Makes a rename or removal in the directory holding `path` durable.
/// Windows has no directory handles to sync; its directory changes are
/// metadata-journaled.
fn sync_parent_dir(path: &Path) -> Result<()> {
    #[cfg(unix)]
    if let Some(parent) = path.parent() {
        let parent = if parent.as_os_str().is_empty() {
            Path::new(".")
        } else {
            parent
        };
        File::open(parent)?.sync_all()?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_dir() -> TempDir {
        TempDir::new().expect("create temp dir")
    }

    #[test]
    fn create_and_open() {
        let dir = test_dir();
        let path = dir.path().join("test.grafeo");

        // Create
        let manager = GrafeoFileManager::create(&path).unwrap();
        assert!(path.exists());
        assert!(manager.active_header().is_empty());
        drop(manager);

        // Open
        let manager = GrafeoFileManager::open(&path).unwrap();
        assert!(manager.active_header().is_empty());
    }

    #[test]
    fn create_fails_if_exists() {
        let dir = test_dir();
        let path = dir.path().join("test.grafeo");

        GrafeoFileManager::create(&path).unwrap();
        let result = GrafeoFileManager::create(&path);
        assert!(result.is_err());
    }

    #[test]
    fn open_fails_if_not_exists() {
        let dir = test_dir();
        let path = dir.path().join("nonexistent.grafeo");

        let result = GrafeoFileManager::open(&path);
        assert!(result.is_err());
    }

    #[test]
    fn write_and_read_snapshot() {
        let dir = test_dir();
        let path = dir.path().join("test.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();

        let snapshot_data = b"hello grafeo snapshot data";
        manager.write_snapshot(snapshot_data, 1, 1, 10, 20).unwrap();

        let loaded = manager.read_snapshot().unwrap();
        assert_eq!(loaded, snapshot_data);

        // Verify header was updated
        let header = manager.active_header();
        assert_eq!(header.iteration, 1);
        assert_eq!(header.snapshot_length, snapshot_data.len() as u64);
        assert_eq!(header.epoch, 1);
        assert_eq!(header.node_count, 10);
        assert_eq!(header.edge_count, 20);
    }

    #[test]
    fn snapshot_persists_across_reopen() {
        let dir = test_dir();
        let path = dir.path().join("test.grafeo");

        let snapshot_data = b"persistent data across reopen";

        // Write
        {
            let manager = GrafeoFileManager::create(&path).unwrap();
            manager
                .write_snapshot(snapshot_data, 5, 3, 100, 200)
                .unwrap();
        }

        // Reopen and read
        {
            let manager = GrafeoFileManager::open(&path).unwrap();
            let loaded = manager.read_snapshot().unwrap();
            assert_eq!(loaded, snapshot_data);

            let header = manager.active_header();
            assert_eq!(header.iteration, 1);
            assert_eq!(header.epoch, 5);
            assert_eq!(header.node_count, 100);
        }
    }

    #[test]
    fn alternating_snapshots() {
        let dir = test_dir();
        let path = dir.path().join("test.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();

        // First checkpoint
        let data1 = b"snapshot version 1";
        manager.write_snapshot(data1, 1, 1, 10, 5).unwrap();
        assert_eq!(manager.active_header().iteration, 1);

        // Second checkpoint (alternates to other slot)
        let data2 = b"snapshot version 2 with more data";
        manager.write_snapshot(data2, 2, 2, 20, 10).unwrap();
        assert_eq!(manager.active_header().iteration, 2);

        let loaded = manager.read_snapshot().unwrap();
        assert_eq!(loaded, data2);
    }

    #[test]
    fn read_empty_snapshot() {
        let dir = test_dir();
        let path = dir.path().join("test.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();
        let data = manager.read_snapshot().unwrap();
        assert!(data.is_empty(), "{data:?}");
    }

    #[test]
    fn read_snapshot_returns_empty_on_v2_header() {
        // After write_sections, snapshot_length == 0 in the active header and the
        // checksum field holds the section-directory CRC. The pre-fix v1 reader
        // would read 0 bytes, CRC empty data to 0, and mismatch the directory CRC.
        // The fix early-returns Ok(Vec::new()) when snapshot_length == 0.
        use grafeo_common::storage::SectionType;

        let dir = test_dir();
        let path = dir.path().join("v2.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();
        manager
            .write_sections(&[(SectionType::LpgStore, b"section payload")], 1, 1, 0, 0)
            .unwrap();

        // Pre-fix: this returned Err("snapshot checksum mismatch").
        // Post-fix: returns Ok(Vec::new()), letting engine fall through to v2 dispatch.
        let data = manager.read_snapshot().unwrap();
        assert!(
            data.is_empty(),
            "v2 file should produce empty snapshot vec, not an error"
        );

        // Sanity: header confirms this is a v2 file (snapshot_length == 0 with non-zero checksum).
        let header = manager.active_header();
        assert_eq!(header.snapshot_length, 0);
        assert!(!header.is_empty());
    }

    #[test]
    fn read_section_directory_surfaces_parse_error_on_v2_header() {
        // A v2 header with a corrupted directory page must not silently
        // degrade to "this is a v1 file" — that masking is what made the
        // GRAFEO-X001 in #323 surface as a misleading snapshot CRC error
        // instead of pointing at the real directory corruption.
        use crate::container::directory::DIRECTORY_OFFSET;
        use grafeo_common::storage::SectionType;

        let dir = test_dir();
        let path = dir.path().join("corrupt_dir.grafeo");

        {
            let manager = GrafeoFileManager::create(&path).unwrap();
            manager
                .write_sections(&[(SectionType::LpgStore, b"section payload")], 1, 1, 0, 0)
                .unwrap();
        }

        // Overwrite the directory page count field with a value above MAX_SECTIONS
        // so SectionDirectory::from_bytes rejects it as malformed.
        {
            let mut file = OpenOptions::new().write(true).open(&path).unwrap();
            file.seek(SeekFrom::Start(DIRECTORY_OFFSET)).unwrap();
            file.write_all(&u32::MAX.to_le_bytes()).unwrap();
        }

        let manager = GrafeoFileManager::open(&path).unwrap();
        let err = manager
            .read_section_directory()
            .expect_err("corrupt v2 directory must surface as Err, not Ok(None)");
        let msg = err.to_string();
        assert!(
            msg.contains("v2 section directory") && msg.contains("failed to parse"),
            "error should name the v2 directory and the parse failure, got: {msg}"
        );
    }

    #[test]
    fn read_section_directory_surfaces_checksum_mismatch_on_v2_header() {
        // A torn write (e.g. a crashed checkpoint) can leave the directory page
        // bytes inconsistent with the CRC the writer recorded in the active
        // header. The pre-fix wildcard match swallowed this, falling through to
        // v1 read logic that reported a misleading snapshot checksum mismatch.
        use crate::container::directory::DIRECTORY_OFFSET;
        use grafeo_common::storage::SectionType;

        let dir = test_dir();
        let path = dir.path().join("torn_dir.grafeo");

        {
            let manager = GrafeoFileManager::create(&path).unwrap();
            manager
                .write_sections(&[(SectionType::LpgStore, b"section payload")], 1, 1, 0, 0)
                .unwrap();
        }

        // Flip a byte in the reserved area of the directory page (bytes 4-7).
        // The page still parses (count is intact, no entries change) but the
        // CRC over the page no longer matches the value in the active header.
        {
            let mut file = OpenOptions::new().write(true).open(&path).unwrap();
            file.seek(SeekFrom::Start(DIRECTORY_OFFSET + 4)).unwrap();
            file.write_all(&[0xAA]).unwrap();
        }

        let manager = GrafeoFileManager::open(&path).unwrap();
        let err = manager
            .read_section_directory()
            .expect_err("torn v2 directory must surface as Err, not Ok(None)");
        let msg = err.to_string();
        assert!(
            msg.contains("v2 section directory checksum mismatch"),
            "error should identify the directory CRC mismatch, got: {msg}"
        );
    }

    #[test]
    fn sidecar_wal_path_computation() {
        let dir = test_dir();
        let path = dir.path().join("mydb.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();
        let wal_path = manager.sidecar_wal_path();

        assert_eq!(
            wal_path.file_name().unwrap().to_str().unwrap(),
            "mydb.grafeo.wal"
        );
        assert!(!manager.has_sidecar_wal());
    }

    #[test]
    fn sidecar_wal_detect_and_remove() {
        let dir = test_dir();
        let path = dir.path().join("test.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();
        assert!(!manager.has_sidecar_wal());

        // Create sidecar directory manually (simulating engine behavior)
        fs::create_dir_all(manager.sidecar_wal_path()).unwrap();
        assert!(manager.has_sidecar_wal());

        // Remove it
        manager.remove_sidecar_wal().unwrap();
        assert!(!manager.has_sidecar_wal());
    }

    #[test]
    fn file_size_grows_with_data() {
        let dir = test_dir();
        let path = dir.path().join("test.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();
        let empty_size = manager.file_size().unwrap();

        // Empty file should be at least 12 KiB (3 headers)
        assert!(empty_size >= DATA_OFFSET, "empty size: {empty_size}");

        let big_data = vec![0xAB; 100_000];
        manager.write_snapshot(&big_data, 1, 1, 0, 0).unwrap();

        let full_size = manager.file_size().unwrap();
        assert!(full_size > empty_size);
        assert_eq!(full_size, DATA_OFFSET + big_data.len() as u64);
    }

    #[test]
    fn exclusive_lock_prevents_second_open() {
        let dir = test_dir();
        let path = dir.path().join("locked.grafeo");

        let _manager1 = GrafeoFileManager::create(&path).unwrap();

        // Second open should fail
        let result = GrafeoFileManager::open(&path);
        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("locked"));
    }

    #[test]
    fn lock_released_after_close() {
        let dir = test_dir();
        let path = dir.path().join("lockclose.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();
        manager.write_snapshot(b"data", 1, 1, 0, 0).unwrap();
        manager.close().unwrap();

        // Should succeed after close
        let manager2 = GrafeoFileManager::open(&path).unwrap();
        let data = manager2.read_snapshot().unwrap();
        assert_eq!(data, b"data");
    }

    #[test]
    fn lock_released_on_drop() {
        let dir = test_dir();
        let path = dir.path().join("lockdrop.grafeo");

        {
            let _manager = GrafeoFileManager::create(&path).unwrap();
            // Drop without explicit close
        }

        // Should succeed after drop
        let _manager2 = GrafeoFileManager::open(&path).unwrap();
    }

    #[test]
    fn checksum_mismatch_detected() {
        let dir = test_dir();
        let path = dir.path().join("test.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();
        manager.write_snapshot(b"valid data", 1, 1, 0, 0).unwrap();
        drop(manager);

        // Corrupt the snapshot data in the file
        {
            let mut file = OpenOptions::new().write(true).open(&path).unwrap();
            file.seek(SeekFrom::Start(DATA_OFFSET)).unwrap();
            file.write_all(b"CORRUPT!!!").unwrap();
        }

        let manager = GrafeoFileManager::open(&path).unwrap();
        let result = manager.read_snapshot();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("checksum"));
    }

    #[test]
    fn open_read_only_reads_snapshot() {
        let dir = test_dir();
        let path = dir.path().join("ro.grafeo");

        // Create and write snapshot, then close
        {
            let manager = GrafeoFileManager::create(&path).unwrap();
            manager
                .write_snapshot(b"read-only test data", 3, 2, 5, 10)
                .unwrap();
            manager.close().unwrap();
        }

        // Open read-only
        let ro = GrafeoFileManager::open_read_only(&path).unwrap();
        assert!(ro.is_read_only());
        let data = ro.read_snapshot().unwrap();
        assert_eq!(data, b"read-only test data");

        let header = ro.active_header();
        assert_eq!(header.epoch, 3);
        assert_eq!(header.node_count, 5);
        assert_eq!(header.edge_count, 10);
    }

    #[test]
    fn read_only_rejects_write_snapshot() {
        let dir = test_dir();
        let path = dir.path().join("ro_write.grafeo");

        {
            let manager = GrafeoFileManager::create(&path).unwrap();
            manager.close().unwrap();
        }

        let ro = GrafeoFileManager::open_read_only(&path).unwrap();
        let result = ro.write_snapshot(b"nope", 1, 1, 0, 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("read-only"));
    }

    #[test]
    fn read_only_coexists_with_exclusive_after_close() {
        let dir = test_dir();
        let path = dir.path().join("coexist.grafeo");

        // Create, write, close
        {
            let manager = GrafeoFileManager::create(&path).unwrap();
            manager.write_snapshot(b"coexist data", 1, 1, 1, 1).unwrap();
            manager.close().unwrap();
        }

        // Two read-only opens should coexist
        let ro1 = GrafeoFileManager::open_read_only(&path).unwrap();
        let ro2 = GrafeoFileManager::open_read_only(&path).unwrap();

        assert_eq!(ro1.read_snapshot().unwrap(), b"coexist data");
        assert_eq!(ro2.read_snapshot().unwrap(), b"coexist data");
    }

    // ── Mmap section tests ─────────────────────────────────────────

    #[test]
    fn mmap_section_roundtrip() {
        use grafeo_common::storage::SectionType;

        let dir = test_dir();
        let path = dir.path().join("mmap.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();

        // Write two sections: one data (LPG), one index (VectorStore)
        let lpg_data = b"lpg node data here";
        let vector_data = vec![0x42u8; 8192]; // 8 KiB of vector embeddings

        manager
            .write_sections(
                &[
                    (SectionType::LpgStore, lpg_data.as_slice()),
                    (SectionType::VectorStore, &vector_data),
                ],
                1,
                1,
                10,
                5,
            )
            .unwrap();

        // Read the directory to get entries
        let section_dir = manager.read_section_directory().unwrap().unwrap();

        // Mmap the VectorStore section (mmap-able)
        let vector_entry = section_dir.find(SectionType::VectorStore).unwrap();
        let mmap = manager.mmap_section(vector_entry).unwrap();

        assert_eq!(mmap.section_type(), SectionType::VectorStore);
        assert_eq!(mmap.len(), vector_data.len());
        assert_eq!(mmap.as_bytes(), &vector_data);
        assert!(!mmap.is_empty());
    }

    #[test]
    fn mmap_rejects_data_sections() {
        use grafeo_common::storage::SectionType;

        let dir = test_dir();
        let path = dir.path().join("mmap_reject.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();
        manager
            .write_sections(&[(SectionType::LpgStore, b"data")], 1, 1, 1, 0)
            .unwrap();

        let section_dir = manager.read_section_directory().unwrap().unwrap();
        let lpg_entry = section_dir.find(SectionType::LpgStore).unwrap();

        let result = manager.mmap_section(lpg_entry);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not mmap-able"));
    }

    #[test]
    fn mmap_detects_corruption() {
        use grafeo_common::storage::SectionType;
        use std::io::Write as IoWrite;

        let dir = test_dir();
        let path = dir.path().join("mmap_corrupt.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();
        let vector_data = vec![0xAB; 4096];
        manager
            .write_sections(&[(SectionType::VectorStore, &vector_data)], 1, 1, 0, 0)
            .unwrap();

        let section_dir = manager.read_section_directory().unwrap().unwrap();
        let entry = section_dir.find(SectionType::VectorStore).unwrap().clone();

        // Corrupt the section data by writing directly to the file
        drop(manager);
        {
            let mut file = OpenOptions::new().write(true).open(&path).unwrap();
            file.seek(SeekFrom::Start(entry.offset)).unwrap();
            file.write_all(b"CORRUPTED!").unwrap();
        }

        let manager = GrafeoFileManager::open(&path).unwrap();
        let result = manager.mmap_section(&entry);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("CRC mismatch"));
    }

    #[test]
    fn mmap_multiple_sections_coexist() {
        use grafeo_common::storage::SectionType;

        let dir = test_dir();
        let path = dir.path().join("mmap_multi.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();

        let vector_data = vec![0x11; 4096];
        let text_data = vec![0x22; 2048];

        manager
            .write_sections(
                &[
                    (SectionType::VectorStore, &vector_data),
                    (SectionType::TextIndex, &text_data),
                ],
                1,
                1,
                0,
                0,
            )
            .unwrap();

        let section_dir = manager.read_section_directory().unwrap().unwrap();

        // Mmap both index sections simultaneously
        let vec_entry = section_dir.find(SectionType::VectorStore).unwrap();
        let text_entry = section_dir.find(SectionType::TextIndex).unwrap();

        let vec_mmap = manager.mmap_section(vec_entry).unwrap();
        let text_mmap = manager.mmap_section(text_entry).unwrap();

        // Both are valid and independent
        assert_eq!(vec_mmap.as_bytes(), &vector_data);
        assert_eq!(text_mmap.as_bytes(), &text_data);
        assert_eq!(vec_mmap.section_type(), SectionType::VectorStore);
        assert_eq!(text_mmap.section_type(), SectionType::TextIndex);
    }

    #[test]
    fn mmap_drop_then_checkpoint_lifecycle() {
        use grafeo_common::storage::SectionType;

        let dir = test_dir();
        let path = dir.path().join("mmap_lifecycle.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();

        // Checkpoint 1: write vector section
        let vector_v1 = vec![0x11; 4096];
        manager
            .write_sections(&[(SectionType::VectorStore, &vector_v1)], 1, 1, 0, 0)
            .unwrap();

        // Mmap the section, read it
        let section_dir = manager.read_section_directory().unwrap().unwrap();
        let entry = section_dir.find(SectionType::VectorStore).unwrap();
        let mmap = manager.mmap_section(entry).unwrap();
        assert_eq!(mmap.as_bytes(), &vector_v1);

        // Drop the mmap before next checkpoint.
        // On Windows, writes fail if mmaps are still active (error 1224).
        // On all platforms, the intended lifecycle is: drop mmaps, checkpoint,
        // re-mmap. This keeps the flow simple and cross-platform.
        drop(mmap);

        // Checkpoint 2: write updated vector section
        let vector_v2 = vec![0x22; 8192];
        manager
            .write_sections(&[(SectionType::VectorStore, &vector_v2)], 2, 2, 0, 0)
            .unwrap();

        // Re-mmap the new section
        let section_dir = manager.read_section_directory().unwrap().unwrap();
        let entry = section_dir.find(SectionType::VectorStore).unwrap();
        let mmap = manager.mmap_section(entry).unwrap();
        assert_eq!(mmap.as_bytes(), &vector_v2);
        assert_eq!(mmap.len(), 8192);
    }

    #[test]
    fn mmap_section_debug_format() {
        use grafeo_common::storage::SectionType;

        let dir = test_dir();
        let path = dir.path().join("mmap_debug.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();
        manager
            .write_sections(&[(SectionType::VectorStore, &[1, 2, 3, 4])], 1, 1, 0, 0)
            .unwrap();

        let section_dir = manager.read_section_directory().unwrap().unwrap();
        let entry = section_dir.find(SectionType::VectorStore).unwrap();
        let mmap = manager.mmap_section(entry).unwrap();

        let debug = format!("{mmap:?}");
        assert!(debug.contains("MmapSection"));
        assert!(debug.contains("VectorStore"));
    }

    #[test]
    fn path_returns_database_file_path() {
        let dir = test_dir();
        let path = dir.path().join("alix.grafeo");
        let manager = GrafeoFileManager::create(&path).unwrap();
        assert_eq!(manager.path(), path);
    }

    #[test]
    fn file_header_returns_valid_header() {
        use crate::file::format;
        let dir = test_dir();
        let path = dir.path().join("gus.grafeo");
        let manager = GrafeoFileManager::create(&path).unwrap();
        let header = manager.file_header();
        assert_eq!(header.magic, format::MAGIC);
        assert_eq!(header.format_version, format::FORMAT_VERSION);
    }

    #[test]
    fn sync_succeeds_for_writable_manager() {
        let dir = test_dir();
        let path = dir.path().join("vincent.grafeo");
        let manager = GrafeoFileManager::create(&path).unwrap();
        manager.write_snapshot(b"sync test", 1, 1, 5, 3).unwrap();
        manager.sync().unwrap();
    }

    #[test]
    fn sync_skips_for_read_only_manager() {
        let dir = test_dir();
        let path = dir.path().join("jules.grafeo");
        {
            let manager = GrafeoFileManager::create(&path).unwrap();
            manager.write_snapshot(b"ro sync", 1, 1, 0, 0).unwrap();
            manager.close().unwrap();
        }
        let ro = GrafeoFileManager::open_read_only(&path).unwrap();
        ro.sync().unwrap();
    }

    #[test]
    fn close_succeeds_for_read_only_manager() {
        let dir = test_dir();
        let path = dir.path().join("mia.grafeo");
        {
            let manager = GrafeoFileManager::create(&path).unwrap();
            manager.close().unwrap();
        }
        let ro = GrafeoFileManager::open_read_only(&path).unwrap();
        ro.close().unwrap();
    }

    #[test]
    fn remove_sidecar_wal_no_op_when_absent() {
        let dir = test_dir();
        let path = dir.path().join("django.grafeo");
        let manager = GrafeoFileManager::create(&path).unwrap();
        assert!(!manager.has_sidecar_wal());
        manager.remove_sidecar_wal().unwrap();
        assert!(!manager.has_sidecar_wal());
    }

    #[test]
    fn multiple_snapshots_alternate_slots() {
        let dir = test_dir();
        let path = dir.path().join("shosanna.grafeo");
        let manager = GrafeoFileManager::create(&path).unwrap();

        manager.write_snapshot(b"epoch one", 1, 1, 1, 0).unwrap();
        assert_eq!(manager.active_header().iteration, 1);

        manager.write_snapshot(b"epoch two", 2, 2, 2, 1).unwrap();
        assert_eq!(manager.active_header().iteration, 2);

        manager
            .write_snapshot(b"epoch three, longer data", 3, 3, 3, 2)
            .unwrap();
        assert_eq!(manager.active_header().iteration, 3);

        let loaded = manager.read_snapshot().unwrap();
        assert_eq!(loaded, b"epoch three, longer data");

        let header = manager.active_header();
        assert_eq!(header.epoch, 3);
        assert_eq!(header.node_count, 3);
        assert!(header.timestamp_ms > 0);
    }

    #[test]
    fn snapshot_truncates_stale_trailing_data() {
        let dir = test_dir();
        let path = dir.path().join("hans.grafeo");
        let manager = GrafeoFileManager::create(&path).unwrap();

        let large_data = vec![0xAA; 50_000];
        manager.write_snapshot(&large_data, 1, 1, 0, 0).unwrap();
        let size_after_large = manager.file_size().unwrap();

        let small_data = b"tiny";
        manager.write_snapshot(small_data, 2, 2, 0, 0).unwrap();
        let size_after_small = manager.file_size().unwrap();

        assert!(
            size_after_small < size_after_large,
            "file should shrink: {size_after_small} >= {size_after_large}"
        );
        assert_eq!(manager.read_snapshot().unwrap(), small_data);
    }

    #[test]
    fn open_read_only_fails_for_nonexistent_file() {
        let dir = test_dir();
        let path = dir.path().join("beatrix_missing.grafeo");
        assert!(GrafeoFileManager::open_read_only(&path).is_err());
    }

    #[test]
    fn copy_to_produces_identical_file() {
        let dir = test_dir();
        let src = dir.path().join("copy_src.grafeo");
        let dest = dir.path().join("copy_dest.grafeo");

        let manager = GrafeoFileManager::create(&src).unwrap();
        manager
            .write_snapshot(b"copy test payload", 5, 3, 10, 20)
            .unwrap();

        // copy_to reads through the locked handle (no new open)
        let bytes = manager.copy_to(&dest).unwrap();
        assert!(bytes > 0);

        // The original is still usable
        let snap = manager.read_snapshot().unwrap();
        assert_eq!(snap, b"copy test payload");
        manager.close().unwrap();

        // The copy is a valid .grafeo file
        let copy = GrafeoFileManager::open(&dest).unwrap();
        let snap = copy.read_snapshot().unwrap();
        assert_eq!(snap, b"copy test payload");

        let header = copy.active_header();
        assert_eq!(header.epoch, 5);
        assert_eq!(header.node_count, 10);
        assert_eq!(header.edge_count, 20);
        copy.close().unwrap();
    }

    #[test]
    fn copy_to_from_read_only_manager() {
        let dir = test_dir();
        let src = dir.path().join("ro_copy_src.grafeo");
        let dest = dir.path().join("ro_copy_dest.grafeo");

        {
            let manager = GrafeoFileManager::create(&src).unwrap();
            manager
                .write_snapshot(b"read-only copy data", 7, 4, 3, 1)
                .unwrap();
            manager.close().unwrap();
        }

        let ro = GrafeoFileManager::open_read_only(&src).unwrap();
        let bytes = ro.copy_to(&dest).unwrap();
        assert!(bytes > 0);

        let copy = GrafeoFileManager::open(&dest).unwrap();
        assert_eq!(copy.read_snapshot().unwrap(), b"read-only copy data");
        copy.close().unwrap();
    }

    #[test]
    #[cfg(all(feature = "encryption", not(miri)))]
    fn encrypted_section_roundtrip() {
        use grafeo_common::encryption::KeyChain;
        use grafeo_common::storage::SectionType;

        let dir = test_dir();
        let path = dir.path().join("encrypted.grafeo");

        let kc = KeyChain::new([0xAB; 32]);

        let section_data = b"sensitive graph data that must be encrypted";

        // Write with encryption
        {
            let mut manager = GrafeoFileManager::create(&path).unwrap();
            manager.set_section_encryptor(kc.encryptor_for("section", b"test"));
            manager
                .write_sections(&[(SectionType::LpgStore, &section_data[..])], 1, 0, 0, 0)
                .unwrap();
            manager.close().unwrap();
        }

        // Read back with same key
        {
            let mut manager = GrafeoFileManager::open(&path).unwrap();
            manager.set_section_encryptor(kc.encryptor_for("section", b"test"));
            let dir_opt = manager.read_section_directory().unwrap();
            let section_dir = dir_opt.expect("directory should exist");
            let entry = section_dir
                .entries()
                .iter()
                .find(|e| e.section_type == SectionType::LpgStore)
                .expect("LpgStore section should exist");
            let decrypted = manager.read_section_data(entry).unwrap();
            assert_eq!(decrypted, section_data);
        }
    }

    #[test]
    #[cfg(all(feature = "encryption", not(miri)))]
    fn encrypted_section_wrong_key_fails() {
        use grafeo_common::encryption::KeyChain;
        use grafeo_common::storage::SectionType;

        let dir = test_dir();
        let path = dir.path().join("wrong_key.grafeo");

        let kc_a = KeyChain::new([0xAA; 32]);
        let kc_b = KeyChain::new([0xBB; 32]);

        // Write with key A
        {
            let mut manager = GrafeoFileManager::create(&path).unwrap();
            manager.set_section_encryptor(kc_a.encryptor_for("section", b"test"));
            manager
                .write_sections(&[(SectionType::LpgStore, b"secret data")], 1, 0, 0, 0)
                .unwrap();
            manager.close().unwrap();
        }

        // Read with key B: CRC passes (computed on encrypted bytes), but decryption fails
        {
            let mut manager = GrafeoFileManager::open(&path).unwrap();
            manager.set_section_encryptor(kc_b.encryptor_for("section", b"test"));
            let dir_opt = manager.read_section_directory().unwrap();
            let section_dir = dir_opt.expect("directory should exist");
            let entry = section_dir
                .entries()
                .iter()
                .find(|e| e.section_type == SectionType::LpgStore)
                .expect("section should exist");
            let result = manager.read_section_data(entry);
            assert!(result.is_err(), "decryption with wrong key should fail");
        }
    }

    // ── Checkpoints never overwrite the file before a complete copy exists (#418) ──

    fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
        let mut name = path.as_os_str().to_owned();
        name.push(suffix);
        PathBuf::from(name)
    }

    fn lpg_payload(manager: &GrafeoFileManager) -> Vec<u8> {
        use grafeo_common::storage::SectionType;

        let dir = manager.read_section_directory().unwrap().unwrap();
        let entry = dir.find(SectionType::LpgStore).unwrap().clone();
        manager.read_section_data(&entry).unwrap()
    }

    fn write_payload(manager: &GrafeoFileManager, payload: &[u8], epoch: u64) -> Result<()> {
        use grafeo_common::storage::SectionType;

        manager.write_sections(&[(SectionType::LpgStore, payload)], epoch, 1, 0, 0)
    }

    /// The bytes of the database file, read through the locked handle
    /// (reading the path directly fails on Windows while it is locked).
    fn file_bytes(manager: &GrafeoFileManager, dir: &TempDir) -> Vec<u8> {
        let copy = dir.path().join("copy.bin");
        manager.copy_to(&copy).unwrap();
        fs::read(&copy).unwrap()
    }

    /// A complete new image of `path` with `payload`, built in another file.
    fn image_with_payload(dir: &TempDir, payload: &[u8]) -> Vec<u8> {
        let other = dir.path().join("other.grafeo");
        {
            let manager = GrafeoFileManager::create(&other).unwrap();
            write_payload(&manager, payload, 7).unwrap();
        }
        let bytes = fs::read(&other).unwrap();
        fs::remove_file(&other).unwrap();
        bytes
    }

    #[test]
    fn checkpoint_leaves_no_side_files() {
        let dir = test_dir();
        let path = dir.path().join("db.grafeo");
        let manager = GrafeoFileManager::create(&path).unwrap();
        write_payload(&manager, b"first", 1).unwrap();
        write_payload(&manager, b"second", 2).unwrap();

        assert_eq!(lpg_payload(&manager), b"second");
        assert!(!with_suffix(&path, ".checkpoint").exists());
        assert!(!with_suffix(&path, ".checkpoint.tmp").exists());
    }

    /// A checkpoint that fails before its new image is complete (here: the
    /// image file cannot be created, like on a full disk) leaves the database
    /// file exactly as it was.
    #[test]
    fn failed_checkpoint_leaves_the_file_untouched() {
        let dir = test_dir();
        let path = dir.path().join("db.grafeo");
        let manager = GrafeoFileManager::create(&path).unwrap();
        write_payload(&manager, b"good state", 1).unwrap();
        let before = file_bytes(&manager, &dir);

        let blocker = with_suffix(&path, ".checkpoint.tmp");
        fs::create_dir(&blocker).unwrap();
        assert!(write_payload(&manager, b"never written", 2).is_err());
        fs::remove_dir(&blocker).unwrap();

        assert_eq!(file_bytes(&manager, &dir), before, "file bytes unchanged");
        assert_eq!(lpg_payload(&manager), b"good state");
        drop(manager);
        let reopened = GrafeoFileManager::open(&path).unwrap();
        assert_eq!(lpg_payload(&reopened), b"good state");
    }

    /// A complete image left by a checkpoint interrupted while installing it
    /// is installed when the database is opened.
    #[test]
    fn open_finishes_an_interrupted_checkpoint() {
        let dir = test_dir();
        let path = dir.path().join("db.grafeo");
        {
            let manager = GrafeoFileManager::create(&path).unwrap();
            write_payload(&manager, b"old", 1).unwrap();
        }
        let pending = with_suffix(&path, ".checkpoint");
        fs::write(&pending, image_with_payload(&dir, b"new")).unwrap();
        // The install was cut off halfway through the database file.
        let mut torn = fs::read(&path).unwrap();
        torn.truncate(torn.len() / 2);
        fs::write(&path, torn).unwrap();

        let manager = GrafeoFileManager::open(&path).unwrap();
        assert_eq!(lpg_payload(&manager), b"new");
        assert_eq!(manager.active_header().epoch, 7);
        assert!(
            !pending.exists(),
            "the pending image is removed once installed"
        );
    }

    /// An image that was still being written is discarded: the database file
    /// was never touched, so it is still valid.
    #[test]
    fn open_discards_an_unfinished_image() {
        let dir = test_dir();
        let path = dir.path().join("db.grafeo");
        {
            let manager = GrafeoFileManager::create(&path).unwrap();
            write_payload(&manager, b"old", 1).unwrap();
        }
        let unfinished = with_suffix(&path, ".checkpoint.tmp");
        fs::write(&unfinished, b"half an image").unwrap();

        let manager = GrafeoFileManager::open(&path).unwrap();
        assert_eq!(lpg_payload(&manager), b"old");
        assert!(!unfinished.exists());
    }

    /// A read-only open cannot install a pending image, so it reads it
    /// instead of the database file, and leaves it for the next writer.
    #[test]
    fn read_only_open_reads_a_pending_image() {
        let dir = test_dir();
        let path = dir.path().join("db.grafeo");
        {
            let manager = GrafeoFileManager::create(&path).unwrap();
            write_payload(&manager, b"old", 1).unwrap();
        }
        let pending = with_suffix(&path, ".checkpoint");
        fs::write(&pending, image_with_payload(&dir, b"new")).unwrap();

        {
            let reader = GrafeoFileManager::open_read_only(&path).unwrap();
            assert_eq!(lpg_payload(&reader), b"new");
            reader.close().unwrap();
        }
        assert!(pending.exists(), "a reader does not install the image");

        let writer = GrafeoFileManager::open(&path).unwrap();
        assert_eq!(lpg_payload(&writer), b"new");
        assert!(!pending.exists());
    }

    /// Side files next to a path whose database file is gone belong to a
    /// deleted database: creating a new one there removes them.
    #[test]
    fn create_removes_stale_checkpoint_files() {
        let dir = test_dir();
        let path = dir.path().join("db.grafeo");
        let pending = with_suffix(&path, ".checkpoint");
        fs::write(&pending, image_with_payload(&dir, b"stale")).unwrap();
        fs::write(with_suffix(&path, ".checkpoint.tmp"), b"stale").unwrap();

        {
            let manager = GrafeoFileManager::create(&path).unwrap();
            assert!(manager.active_header().is_empty());
        }
        assert!(!pending.exists());
        assert!(!with_suffix(&path, ".checkpoint.tmp").exists());
        let reopened = GrafeoFileManager::open(&path).unwrap();
        assert!(reopened.active_header().is_empty());
    }

    /// The image holds the whole database, so it is never readable by more
    /// users than the database file, also when an old image file is reused.
    #[cfg(unix)]
    #[test]
    fn checkpoint_image_is_as_private_as_the_database() {
        use grafeo_common::storage::SectionType;
        use std::os::unix::fs::PermissionsExt;

        let dir = test_dir();
        let path = dir.path().join("db.grafeo");
        let manager = GrafeoFileManager::create(&path).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        let image = with_suffix(&path, ".checkpoint.tmp");

        for reused in [false, true] {
            if reused {
                fs::write(&image, b"stale").unwrap();
                fs::set_permissions(&image, fs::Permissions::from_mode(0o644)).unwrap();
            }
            let header = manager.active_header();
            let mut file = manager.file.lock();
            manager
                .write_image(
                    &mut file,
                    &image,
                    &[(SectionType::LpgStore, b"data".as_slice())],
                    &header,
                    0,
                    (1, 1, 0, 0),
                )
                .unwrap();
            let mode = fs::metadata(&image).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600, "reused: {reused}");
            fs::remove_file(&image).unwrap();
        }
    }

    /// Checkpoints `new` over a database holding `old` and makes it fail at
    /// injection point `point`: the injected panic is caught, like an error
    /// return, and the manager stays in use. Returns whether it completed.
    #[cfg(feature = "testing-crash-injection")]
    fn fail_checkpoint_at(point: u64, path: &Path) -> (GrafeoFileManager, bool) {
        use grafeo_common::testing::crash::{CrashResult, with_crash_at};

        let manager = GrafeoFileManager::create(path).unwrap();
        write_payload(&manager, b"old", 1).unwrap();
        let target = std::panic::AssertUnwindSafe(&manager);
        let result = with_crash_at(point, move || write_payload(*target, b"new", 2));
        let completed = matches!(result, CrashResult::Completed(Ok(())));
        (manager, completed)
    }

    #[cfg(feature = "testing-crash-injection")]
    fn try_lpg_payload(manager: &GrafeoFileManager) -> Result<Vec<u8>> {
        use grafeo_common::storage::SectionType;

        let dir = manager
            .read_section_directory()?
            .ok_or_else(|| Error::Internal("no section directory".to_string()))?;
        let entry = dir.find(SectionType::LpgStore).unwrap().clone();
        manager.read_section_data(&entry)
    }

    /// After a checkpoint fails at any step, this process reads either the
    /// old or the new state, never a half-written file, a copy (a backup)
    /// holds the same, and so does the next open.
    #[cfg(feature = "testing-crash-injection")]
    #[test]
    fn reads_after_a_failed_checkpoint_agree_with_the_next_open() {
        // More points than a checkpoint has, so the last runs complete.
        for point in 1..=10 {
            let dir = test_dir();
            let path = dir.path().join("db.grafeo");
            let (manager, completed) = fail_checkpoint_at(point, &path);

            let seen = try_lpg_payload(&manager)
                .unwrap_or_else(|e| panic!("point {point}: read failed: {e}"));
            assert!(
                seen == b"old" || seen == b"new",
                "point {point}: read {seen:?}"
            );
            if completed {
                assert_eq!(seen, b"new", "point {point}");
            }

            let copy = dir.path().join("copy.grafeo");
            manager.copy_to(&copy).unwrap();
            let copied = GrafeoFileManager::open_read_only(&copy).unwrap();
            assert_eq!(
                try_lpg_payload(&copied).unwrap(),
                seen,
                "point {point}: copy"
            );
            drop(copied);

            drop(manager);
            let reopened = GrafeoFileManager::open(&path).unwrap();
            assert_eq!(lpg_payload(&reopened), seen, "point {point}: reopen");
        }
    }

    /// A write after a failed checkpoint is not undone at the next open by
    /// an image that the failed checkpoint left behind.
    #[cfg(feature = "testing-crash-injection")]
    #[test]
    fn writes_after_a_failed_checkpoint_survive_the_next_open() {
        for point in 1..=10 {
            let dir = test_dir();
            let path = dir.path().join("db.grafeo");

            let (manager, _) = fail_checkpoint_at(point, &path);
            write_payload(&manager, b"newer", 3).unwrap();
            assert_eq!(lpg_payload(&manager), b"newer", "point {point}");
            drop(manager);
            let reopened = GrafeoFileManager::open(&path).unwrap();
            assert_eq!(lpg_payload(&reopened), b"newer", "point {point}: sections");
            drop(reopened);
            fs::remove_file(&path).unwrap();

            let (manager, _) = fail_checkpoint_at(point, &path);
            manager.write_snapshot(b"snapshot", 3, 1, 0, 0).unwrap();
            drop(manager);
            let reopened = GrafeoFileManager::open(&path).unwrap();
            assert_eq!(
                reopened.read_snapshot().unwrap(),
                b"snapshot",
                "point {point}: snapshot"
            );
        }
    }
}
