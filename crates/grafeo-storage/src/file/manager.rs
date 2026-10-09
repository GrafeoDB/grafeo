//! High-level manager for `.grafeo` database files (container format v3).
//!
//! [`GrafeoFileManager`] owns the file handle and its lock. It creates and
//! opens v3 files, writes copy-on-write checkpoints, reads the sections of the
//! active image and manages the sidecar WAL directory. Files written by 0.5.x
//! are read by [`LegacyFile`](super::legacy::LegacyFile) and must be migrated
//! before this manager opens them.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use fs2::FileExt;
use grafeo_common::grafeo_warn;
use grafeo_common::storage::{ImageSource, Section};
use grafeo_common::testing::child_process;
use grafeo_common::testing::crash::{maybe_crash, maybe_fail};
use grafeo_common::utils::error::{Error, Result};
use grafeo_common::utils::hash::FxHashSet;
use parking_lot::Mutex;

use super::format::MAGIC;
use super::v3::alloc::{PageAllocator, PageRun};
use super::v3::directory::SkippedEntry;
use super::v3::header::{
    DATA_START_PAGE, DbHeaderV3, FORMAT_REVISION, FileHeaderV3, PAGE_SIZE, active_header,
    check_format_revision, new_database_id,
};
use super::v3::{CheckpointWriter, ChunkCipher, ImageReader};

/// Size of a page (and of each header) in bytes, as a `usize`.
const PAGE_BYTES: usize = 4096;

/// The values a checkpoint records in its database header. The manager adds
/// the iteration, the root of the new image and the time.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CheckpointHeader {
    /// WAL sequence number the checkpoint covers.
    pub checkpoint_lsn: u64,
    /// MVCC epoch at the checkpoint.
    pub epoch: u64,
    /// Last transaction id at the checkpoint.
    pub last_transaction_id: u64,
    /// Number of nodes at the checkpoint.
    pub node_count: u64,
    /// Number of edges at the checkpoint.
    pub edge_count: u64,
}

/// What the active image holds, counted from its directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageStats {
    /// Chunks of all sections this version knows, zero-length chunks
    /// included. Chunks skipped as optional (written by a newer version) are
    /// not counted.
    pub chunks: usize,
    /// Blocks of the chained directory (at least one).
    pub directory_blocks: usize,
    /// Pages the chunks (skipped ones included) and the directory blocks
    /// use. The file header and the two database headers are not counted:
    /// they belong to the file, not to an image.
    pub pages: u64,
}

/// The image the active database header points at.
struct ActiveImage {
    /// The active database header.
    header: DbHeaderV3,
    /// Slot (0 or 1) that holds it.
    slot: u8,
    /// Pages the next checkpoint must not write: those of the active image,
    /// plus those of any failed checkpoint whose header may have reached the
    /// disk (a reopen could find that image active).
    runs: Vec<PageRun>,
}

#[cfg(test)]
thread_local! {
    /// Every skipped-entry warning logged on this thread, in order, so a test
    /// can count them.
    static WARNINGS_LOGGED: std::cell::RefCell<Vec<(u8, u8)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Logs one warning per distinct (section type, chunk kind) of the entries
/// the active image skips.
///
/// The manager calls it once, when it opens the file. Every reader it opens
/// later reads that image or one its own checkpoint wrote, which holds no
/// skipped entries (a checkpoint writes only the section types and chunk
/// kinds it knows), so nothing would be warned about again.
fn warn_skipped(skipped: &[SkippedEntry]) {
    for (section_type, kind) in skipped_kinds(skipped) {
        grafeo_warn!(
            "skipping chunks of section type {section_type}, kind {kind}, written by a newer \
             version: they are optional, and the next checkpoint drops them"
        );
        #[cfg(test)]
        WARNINGS_LOGGED.with_borrow_mut(|logged| logged.push((section_type, kind)));
    }
}

/// The distinct (section type, chunk kind) pairs of `skipped`, in the order
/// they first appear.
fn skipped_kinds(skipped: &[SkippedEntry]) -> Vec<(u8, u8)> {
    let mut seen = FxHashSet::default();
    skipped
        .iter()
        .map(|entry| (entry.section_type, entry.kind))
        .filter(|pair| seen.insert(*pair))
        .collect()
}

/// Manages a single `.grafeo` database file in container format v3.
///
/// # Lifecycle
///
/// 1. [`create`](Self::create) or [`open`](Self::open)
/// 2. Mutations flow through a sidecar WAL (managed externally by the engine)
/// 3. [`write_checkpoint`](Self::write_checkpoint) writes the sections as a
///    new image next to the active one and then switches the inactive
///    database header to it; periodic checkpoints retain the sidecar WAL
/// 4. On a clean shutdown, after the final checkpoint, call
///    [`remove_sidecar_wal`](Self::remove_sidecar_wal) to discard the sidecar
/// 5. [`close`](Self::close) (or drop) releases the file handle
///
/// The manager owns the cipher of an encrypted file from construction on:
/// every image it writes is encrypted with it, and every read decrypts.
pub struct GrafeoFileManager {
    /// Path to the `.grafeo` file.
    path: PathBuf,
    /// Open file handle (read/write or read-only), which holds the lock.
    file: Mutex<File>,
    /// File header (read once on open, immutable afterwards).
    file_header: FileHeaderV3,
    /// The active image. Changed only while `file` is locked.
    active: Mutex<ActiveImage>,
    /// Whether this manager was opened in read-only mode.
    read_only: bool,
    /// Held for a whole checkpoint, see [`checkpoint_guard`](Self::checkpoint_guard).
    checkpoint_lock: Mutex<()>,
    /// Encrypts and decrypts the images (`None` for a plain file).
    cipher: Option<ChunkCipher>,
}

impl GrafeoFileManager {
    /// Creates a new `.grafeo` file at `path`.
    ///
    /// Writes the file header (encrypted when `cipher` is given, with a new
    /// database id), an image without sections (one directory block) and a
    /// valid iteration-0 database header in slot 0 pointing at it; slot 1
    /// stays all zero. Nothing may exist at `path`: no file, directory or
    /// symbolic link (a dangling one included).
    ///
    /// The file is built and synced under `<path>.creating`, then renamed to
    /// `path` (and the directory synced), so `path` never holds a partial
    /// file. A `<path>.creating` that a failed or crashed create left behind
    /// is replaced; one that another create still holds locked is left alone
    /// and this create fails.
    ///
    /// # Errors
    ///
    /// Returns an error if anything exists at `path` (or its metadata cannot
    /// be read), another create of the same path is in progress, or a write,
    /// sync or the rename fails.
    pub fn create(path: impl AsRef<Path>, cipher: Option<ChunkCipher>) -> Result<Self> {
        Self::create_with_id(path, new_database_id(), cipher)
    }

    /// Creates a new `.grafeo` file at `path` with the database id
    /// `database_id`, as [`create`](Self::create) does with a new one.
    ///
    /// For a cipher that depends on the database, such as a key derived from
    /// the id: the caller picks the id with
    /// [`new_database_id`], derives the
    /// cipher from it and passes both.
    ///
    /// # Errors
    ///
    /// The same as [`create`](Self::create).
    pub fn create_with_id(
        path: impl AsRef<Path>,
        database_id: u128,
        cipher: Option<ChunkCipher>,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        refuse_existing(&path)?;

        // Ensure parent directory exists
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }

        // The lock on `<path>.creating` makes concurrent creates of one path
        // take turns; it moves with the file to `path` and stays the
        // manager's exclusive lock.
        let creating = creating_path(&path);
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&creating)?;
        lock_exclusive(&file, &creating)?;
        if let Err(error) = refuse_existing(&path) {
            // Another create finished between the check above and the lock.
            drop(file);
            remove_if_exists(&creating)?;
            return Err(error);
        }
        // Whatever a failed create left in the file goes.
        file.set_len(0)?;

        let file_header = FileHeaderV3 {
            database_id,
            ..FileHeaderV3::new(cipher.is_some())
        };
        let (root, runs) =
            CheckpointWriter::new(&mut file, PageAllocator::from_used([])?, cipher.as_ref())
                .finish()?;
        let header = DbHeaderV3 {
            format_revision: FORMAT_REVISION,
            root,
            timestamp_ms: now_ms(),
            ..DbHeaderV3::default()
        };
        write_page(&mut file, 0, &file_header.encode())?;
        write_page(&mut file, slot_offset(0), &header.encode())?;
        write_page(&mut file, slot_offset(1), &[0u8; PAGE_BYTES])?;
        file.set_len(image_end(&runs)?)?;
        file.sync_all()?;
        maybe_crash("create:after_write");
        fs::rename(&creating, &path)?;
        sync_parent_dir(&path)?;

        Ok(Self {
            path,
            file: Mutex::new(file),
            file_header,
            active: Mutex::new(ActiveImage {
                header,
                slot: 0,
                runs,
            }),
            read_only: false,
            checkpoint_lock: Mutex::new(()),
            cipher,
        })
    }

    /// Opens an existing v3 `.grafeo` file for reading and writing, under an
    /// exclusive lock.
    ///
    /// Selects the active database header and reads the directory of its
    /// image once, which checks it (and, for an encrypted file, the key).
    ///
    /// # Errors
    ///
    /// Returns an error if the file does not exist or is locked, was written
    /// by 0.5.x (it must be migrated first), is encrypted and `cipher` is
    /// `None`, is not encrypted and `cipher` is given, has no valid database
    /// header, or its active image cannot be read (for example with the wrong
    /// key).
    pub fn open(path: impl AsRef<Path>, cipher: Option<ChunkCipher>) -> Result<Self> {
        Self::open_existing(path.as_ref(), |_| cipher, false)
    }

    /// Opens an existing v3 `.grafeo` file for reading and writing, as
    /// [`open`](Self::open) does, with the cipher chosen for the database:
    /// `cipher_for` is called once with the database id of the file header,
    /// read under the lock before anything is decrypted, and returns the
    /// cipher, or `None` to open without one.
    ///
    /// For keys derived per database, such as a key chain's
    /// `encryptor_for(context, &database_id.to_le_bytes())`. A file written
    /// by 0.5.x is refused before `cipher_for` is called.
    ///
    /// # Errors
    ///
    /// The same as [`open`](Self::open), with the cipher `cipher_for` chose.
    pub fn open_with_cipher_for(
        path: impl AsRef<Path>,
        cipher_for: impl FnOnce(u128) -> Option<ChunkCipher>,
    ) -> Result<Self> {
        Self::open_existing(path.as_ref(), cipher_for, false)
    }

    /// Opens an existing v3 `.grafeo` file in read-only mode.
    ///
    /// Uses a **shared** file lock (`try_lock_shared`), so several read-only
    /// managers can open the same file at once. A read-write manager's
    /// exclusive lock and these shared locks exclude each other (`flock` on
    /// Unix, `LockFileEx` on Windows): a reader cannot open a file a writer
    /// holds, and a writer cannot open a file a reader holds. The returned
    /// manager refuses [`write_checkpoint`](Self::write_checkpoint).
    ///
    /// # Errors
    ///
    /// The same as [`open`](Self::open); the lock fails while a read-write
    /// manager holds the file.
    pub fn open_read_only(path: impl AsRef<Path>, cipher: Option<ChunkCipher>) -> Result<Self> {
        Self::open_existing(path.as_ref(), |_| cipher, true)
    }

    /// Opens an existing v3 `.grafeo` file in read-only mode, as
    /// [`open_read_only`](Self::open_read_only) does, with the cipher chosen
    /// for the database as [`open_with_cipher_for`](Self::open_with_cipher_for)
    /// chooses it.
    ///
    /// # Errors
    ///
    /// The same as [`open_read_only`](Self::open_read_only), with the cipher
    /// `cipher_for` chose.
    pub fn open_read_only_with_cipher_for(
        path: impl AsRef<Path>,
        cipher_for: impl FnOnce(u128) -> Option<ChunkCipher>,
    ) -> Result<Self> {
        Self::open_existing(path.as_ref(), cipher_for, true)
    }

    fn open_existing(
        path: &Path,
        cipher_for: impl FnOnce(u128) -> Option<ChunkCipher>,
        read_only: bool,
    ) -> Result<Self> {
        let path = path.to_path_buf();
        let mut file = if read_only {
            let file = OpenOptions::new().read(true).open(&path)?;
            lock_shared(&file, &path)?;
            file
        } else {
            let file = OpenOptions::new().read(true).write(true).open(&path)?;
            lock_exclusive(&file, &path)?;
            file
        };
        let (file_header, active, cipher) = read_active_image(&mut file, &path, cipher_for)?;
        Ok(Self {
            path,
            file: Mutex::new(file),
            file_header,
            active: Mutex::new(active),
            read_only,
            checkpoint_lock: Mutex::new(()),
            cipher,
        })
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

    /// Writes `sections` as a new image and makes it the active one.
    ///
    /// Copy-on-write: the chunks and the directory go into pages the active
    /// image does not use, and the file is synced; then the next database
    /// header (iteration + 1, the values of `header`, the new root and the
    /// active header's format revision) is written
    /// into the inactive slot and synced; then the file is cut after the last
    /// page of the new image. Until the header is on disk the active image is
    /// untouched, so a crash at any point opens either the old or the new
    /// image.
    ///
    /// # Errors
    ///
    /// Returns an error if the manager is read-only, a section fails to
    /// serialize, repeats a section type or writes two chunks with one
    /// identity ([`ChunkMeta::identity`](grafeo_common::storage::ChunkMeta::identity))
    /// in one section, or a write or sync fails.
    ///
    /// After a failure before the header write, the file holds no image newer
    /// than before the call. After a failure at or after the header write,
    /// the new header may be on disk: [`active_header`](Self::active_header)
    /// and [`read_image`](Self::read_image) keep serving the previous image,
    /// while a reopen may find the new one. The next checkpoint spares
    /// the pages of both. The caller must therefore keep the WAL from the
    /// previous image's `checkpoint_lsn` until a later checkpoint succeeds.
    pub fn write_checkpoint(
        &self,
        sections: &[&dyn Section],
        header: &CheckpointHeader,
    ) -> Result<()> {
        if self.read_only {
            return Err(Error::Internal(
                "cannot write a checkpoint: the database is open in read-only mode".to_string(),
            ));
        }

        let mut file = self.file.lock();
        let (iteration, revision, slot, spared) = {
            let active = self.active.lock();
            (
                active.header.iteration,
                active.header.format_revision,
                active.slot,
                active.runs.clone(),
            )
        };
        let next_iteration = iteration.checked_add(1).ok_or_else(|| {
            Error::Internal(format!(
                "database header iteration {iteration} cannot be incremented"
            ))
        })?;

        let mut writer = CheckpointWriter::new(
            &mut file,
            PageAllocator::from_used(spared)?,
            self.cipher.as_ref(),
        );
        for section in sections {
            writer.begin_section(section.section_type(), section.version())?;
            section.write_to(&mut writer)?;
        }
        let (root, runs) = writer.finish()?;
        maybe_crash("checkpoint:after_chunks");
        maybe_fail("checkpoint:after_chunks")?;
        file.sync_all()?;
        maybe_crash("checkpoint:after_data_sync");
        maybe_fail("checkpoint:after_data_sync")?;

        // From the header write on, a reopen may find the new image active:
        // a later checkpoint of this manager must spare its pages as well.
        self.active.lock().runs.extend(runs.iter().copied());
        let next_slot = 1 - slot;
        let next = DbHeaderV3 {
            // A file keeps its revision: an upgrade is an explicit step.
            format_revision: revision,
            iteration: next_iteration,
            checkpoint_lsn: header.checkpoint_lsn,
            epoch: header.epoch,
            last_transaction_id: header.last_transaction_id,
            root,
            node_count: header.node_count,
            edge_count: header.edge_count,
            timestamp_ms: now_ms(),
        };
        write_page(&mut file, slot_offset(next_slot), &next.encode())?;
        maybe_crash("checkpoint:after_header");
        maybe_fail("checkpoint:after_header")?;
        file.sync_all()?;

        // The previous image is no longer reachable: drop the pages after
        // the new image. A failure of the trim leaves them: the next
        // checkpoint cuts the file.
        maybe_crash("checkpoint:before_trim");
        maybe_fail("checkpoint:before_trim")?;
        file.set_len(image_end(&runs)?)?;

        *self.active.lock() = ActiveImage {
            header: next,
            slot: next_slot,
            runs,
        };
        Ok(())
    }

    /// Runs `read` on the active image under the file lock.
    ///
    /// The image is the one the active header names when the call takes the
    /// lock (decrypted with the file's cipher when it is encrypted), and it
    /// stays that image until `read` returns: a checkpoint takes the same
    /// lock, so it waits. `read` must not call a method of this manager that
    /// takes the file lock (`read_image`, `image_stats`, `write_checkpoint`,
    /// `file_size`, `sync`, `copy_to`, `close`): the lock is held, and that
    /// call would wait for it forever. Nor may it take a lock that a
    /// checkpoint holds while it waits for the file lock
    /// ([`checkpoint_guard`](Self::checkpoint_guard), or a hold the caller
    /// takes around its checkpoints, such as the engine's commit hold): that
    /// inverts the lock order and deadlocks against a concurrent checkpoint.
    ///
    /// # Errors
    ///
    /// Returns an error when the directory of the active image cannot be
    /// read or fails its checks, or the error `read` returns.
    pub fn read_image<T>(&self, read: impl FnOnce(&dyn ImageSource) -> Result<T>) -> Result<T> {
        self.with_active_reader(|reader| read(reader))
    }

    /// Chunk, directory block and page counts of the active image.
    ///
    /// The pages are those of its chunks and directory blocks (see
    /// [`ImageStats::pages`]).
    ///
    /// # Errors
    ///
    /// Returns an error when the directory of the active image cannot be
    /// read or fails its checks.
    pub fn image_stats(&self) -> Result<ImageStats> {
        self.with_active_reader(|reader| {
            Ok(ImageStats {
                chunks: reader.entries().len(),
                directory_blocks: reader.directory_runs().len(),
                pages: reader.used_runs().iter().map(|run| run.count).sum(),
            })
        })
    }

    /// Opens the reader of the active image under the file lock and runs
    /// `read` on it. The lock order (the file, then the active image for as
    /// long as it takes to read its root) is the order of `write_checkpoint`.
    fn with_active_reader<T>(&self, read: impl FnOnce(&ImageReader<'_>) -> Result<T>) -> Result<T> {
        let mut file = self.file.lock();
        let root = self.active.lock().header.root;
        let reader = ImageReader::open(&mut file, root, self.cipher.as_ref())?;
        read(&reader)
    }

    /// Returns the path for the sidecar WAL directory.
    ///
    /// For a database at `mydb.grafeo`, the sidecar is `mydb.grafeo.wal/`.
    #[must_use]
    pub fn sidecar_wal_path(&self) -> PathBuf {
        super::detect::sidecar_wal_path(&self.path)
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

    /// Returns a clone of the currently active database header (iteration 0
    /// before the first checkpoint).
    #[must_use]
    pub fn active_header(&self) -> DbHeaderV3 {
        self.active.lock().header.clone()
    }

    /// Returns the random id of this database, set when it was created.
    #[must_use]
    pub fn database_id(&self) -> u128 {
        self.file_header.database_id
    }

    /// Returns the file header (written at creation, immutable).
    #[must_use]
    pub fn file_header(&self) -> &FileHeaderV3 {
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
        let mut file = self.file.lock();
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
        file.unlock()
            .map_err(|e| Error::Internal(format!("failed to unlock database file: {e}")))?;
        Ok(())
    }
}

impl Drop for GrafeoFileManager {
    fn drop(&mut self) {
        let _ = self.file.lock().unlock();
    }
}

/// Takes the exclusive lock of a read-write manager.
fn lock_exclusive(file: &File, path: &Path) -> Result<()> {
    child_process::take_lock(|| file.try_lock_exclusive(), is_lock_contended).map_err(|_| {
        Error::Internal(format!(
            "database file is locked by another process: {}",
            path.display()
        ))
    })
}

/// Takes the shared lock of a read-only manager. It coexists with other
/// shared locks and fails while a read-write manager holds the exclusive one.
fn lock_shared(file: &File, path: &Path) -> Result<()> {
    child_process::take_lock(
        || file.try_lock_shared(),
        |e| matches!(e, std::fs::TryLockError::WouldBlock),
    )
    .map_err(|_| {
        Error::Internal(format!(
            "database file cannot be locked for reading: {}",
            path.display()
        ))
    })
}

/// Whether a failed `fs2` lock attempt failed because another handle holds
/// the lock (as opposed to an I/O error).
fn is_lock_contended(error: &std::io::Error) -> bool {
    error.raw_os_error() == fs2::lock_contended_error().raw_os_error()
}

/// Reads the headers of an existing file, chooses its cipher and checks its
/// active image. Returns the file header, the active image and the cipher.
///
/// Refuses a 0.5.x file (byte 4 is the varint `0x01` of its bincode format
/// version, see [`detect`](super::detect::detect)) before `cipher_for` is
/// called, then a file whose encryption does not match the cipher
/// `cipher_for` returns for its database id, a file without a valid database
/// header, an active header of a format revision this build does not read
/// (see [`check_format_revision`]), an active header without a directory
/// block, and an active image whose directory cannot be read. Nothing is
/// written: a refused file stays as it is.
fn read_active_image(
    file: &mut File,
    path: &Path,
    cipher_for: impl FnOnce(u128) -> Option<ChunkCipher>,
) -> Result<(FileHeaderV3, ActiveImage, Option<ChunkCipher>)> {
    let prefix = read_prefix(file, DATA_START_PAGE * PAGE_SIZE)?;
    if prefix.len() > 4 && prefix[..4] == MAGIC && prefix[4] == 1 {
        return Err(Error::InvalidValue(format!(
            "{} was written by Grafeo 0.5.x and must be migrated to the 0.6 file format \
             before it can be opened",
            path.display()
        )));
    }
    let context =
        |error: Error| Error::Serialization(format!("cannot open {}: {error}", path.display()));
    let file_header = FileHeaderV3::decode(page_of(&prefix, 0)).map_err(context)?;
    let cipher = cipher_for(file_header.database_id);
    match (file_header.encrypted, cipher.is_some()) {
        (true, false) => {
            return Err(Error::InvalidValue(format!(
                "the database is encrypted and needs its key: {}",
                path.display()
            )));
        }
        (false, true) => {
            return Err(Error::InvalidValue(format!(
                "the database is not encrypted, open it without a key: {}",
                path.display()
            )));
        }
        _ => {}
    }
    let slots = [
        DbHeaderV3::decode(page_of(&prefix, 1)),
        DbHeaderV3::decode(page_of(&prefix, 2)),
    ];
    let (slot, header) = active_header(slots).map_err(context)?.ok_or_else(|| {
        Error::Serialization(format!(
            "cannot open {}: both database header slots are empty",
            path.display()
        ))
    })?;
    // Before the directory: a revision this build does not know may use a
    // directory or chunks it cannot read.
    check_format_revision(header.format_revision).map_err(context)?;
    // Every image this format writes has a directory block. Without one an
    // encrypted file would open with any key (nothing to decrypt), and the
    // next checkpoint would write with that key.
    if header.root.length == 0 {
        return Err(Error::Serialization(format!(
            "cannot open {}: the active database header (iteration {}, slot {slot}) \
             has no directory block",
            path.display(),
            header.iteration
        )));
    }
    let reader = ImageReader::open(file, header.root, cipher.as_ref()).map_err(|error| {
        Error::Serialization(format!(
            "cannot read the active image of {} (iteration {}): {error}",
            path.display(),
            header.iteration
        ))
    })?;
    // The only warning: later readers see this image or one without
    // skipped entries.
    warn_skipped(reader.skipped());
    let runs = reader.used_runs();
    Ok((file_header, ActiveImage { header, slot, runs }, cipher))
}

/// Reads the first `length` bytes of the file, or fewer if it is shorter.
fn read_prefix(file: &mut File, length: u64) -> Result<Vec<u8>> {
    file.seek(SeekFrom::Start(0))?;
    let mut prefix = Vec::new();
    Read::by_ref(file).take(length).read_to_end(&mut prefix)?;
    Ok(prefix)
}

/// The bytes of page `index` within `prefix`, shorter (or empty) when the
/// file ends inside the page.
fn page_of(prefix: &[u8], index: usize) -> &[u8] {
    prefix
        .get(index * PAGE_BYTES..)
        .map_or(&[], |rest| &rest[..rest.len().min(PAGE_BYTES)])
}

/// Where [`GrafeoFileManager::create`] builds a new file before renaming it
/// into place: `<path>.creating`.
fn creating_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".creating");
    PathBuf::from(name)
}

/// Refuses a create at `path` while anything is there: a file, a directory,
/// a symbolic link (also a dangling one, which `Path::exists` does not see)
/// or an entry whose metadata cannot be read. The rename at the end of a
/// create would replace it.
fn refuse_existing(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Ok(_) => Err(Error::Internal(format!(
            "file already exists: {}",
            path.display()
        ))),
        Err(error) => Err(Error::Internal(format!(
            "cannot create {}: the path cannot be inspected, so it may exist: {error}",
            path.display()
        ))),
    }
}

fn remove_if_exists(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.into()),
    }
}

/// Makes a rename in the directory holding `path` durable. Windows has no
/// directory handles to sync; its directory changes are metadata-journaled.
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

/// Byte offset of database header slot 0 or 1.
fn slot_offset(slot: u8) -> u64 {
    PAGE_SIZE * (1 + u64::from(slot))
}

/// Writes `bytes` at `offset`.
fn write_page(file: &mut File, offset: u64, bytes: &[u8]) -> Result<()> {
    file.seek(SeekFrom::Start(offset))?;
    file.write_all(bytes)?;
    Ok(())
}

/// The end of the last page an image uses, where the file is cut.
fn image_end(runs: &[PageRun]) -> Result<u64> {
    let mut end_page = DATA_START_PAGE;
    for run in runs {
        end_page = end_page.max(run.end()?);
    }
    end_page.checked_mul(PAGE_SIZE).ok_or_else(|| {
        Error::Internal(format!(
            "image ends at page {end_page}, beyond the file offset range"
        ))
    })
}

/// The current time in milliseconds since the Unix epoch.
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| {
            u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX)
        })
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "testing-crash-injection")]
    use std::panic::AssertUnwindSafe;

    use grafeo_common::storage::{
        ChunkMeta, Section, SectionSink, SectionSource, SectionType, legacy_bytes, read_raw,
        write_raw,
    };
    use tempfile::TempDir;

    use super::super::v3::directory::{ENTRY_CHUNK_OPTIONAL, ENTRY_SECTION_OPTIONAL};
    use super::super::v3::header::{HeaderSlot, MAX_FORMAT_REVISION};
    use super::*;

    /// A section holding fixed bytes, written as one raw chunk.
    struct Fixed {
        section_type: SectionType,
        data: Vec<u8>,
    }

    impl Section for Fixed {
        fn section_type(&self) -> SectionType {
            self.section_type
        }

        fn serialize(&self) -> Result<Vec<u8>> {
            Ok(self.data.clone())
        }

        fn deserialize(&mut self, data: &[u8]) -> Result<()> {
            self.data = data.to_vec();
            Ok(())
        }

        fn write_to(&self, sink: &mut dyn SectionSink) -> Result<()> {
            write_raw(self, sink)
        }

        fn read_from(&mut self, source: &dyn SectionSource) -> Result<()> {
            read_raw(self, source)
        }

        fn is_dirty(&self) -> bool {
            true
        }

        fn mark_clean(&self) {}

        fn memory_usage(&self) -> usize {
            self.data.len()
        }
    }

    /// A section that streams two raw chunks, which are not 0.5.x section
    /// bytes.
    struct TwoChunks;

    impl Section for TwoChunks {
        fn section_type(&self) -> SectionType {
            SectionType::LpgStore
        }

        fn serialize(&self) -> Result<Vec<u8>> {
            Ok(b"VincentJules".to_vec())
        }

        fn deserialize(&mut self, _data: &[u8]) -> Result<()> {
            Ok(())
        }

        fn write_to(&self, sink: &mut dyn SectionSink) -> Result<()> {
            sink.write_chunk(ChunkMeta::raw(), b"Vincent")?;
            sink.write_chunk(
                ChunkMeta {
                    row_start: 1,
                    ..ChunkMeta::raw()
                },
                b"Jules",
            )
        }

        fn read_from(&mut self, source: &dyn SectionSource) -> Result<()> {
            read_raw(self, source)
        }

        fn is_dirty(&self) -> bool {
            true
        }

        fn mark_clean(&self) {}

        fn memory_usage(&self) -> usize {
            0
        }
    }

    /// A section writing `n` one-byte stream pieces, each a chunk of its own.
    struct Pieces(usize);

    impl Section for Pieces {
        fn section_type(&self) -> SectionType {
            SectionType::LpgStore
        }

        fn serialize(&self) -> Result<Vec<u8>> {
            Ok(vec![19u8; self.0])
        }

        fn deserialize(&mut self, _data: &[u8]) -> Result<()> {
            Ok(())
        }

        fn write_to(&self, sink: &mut dyn SectionSink) -> Result<()> {
            for offset in 0..u64::try_from(self.0).unwrap() {
                sink.write_chunk(ChunkMeta::stream_piece(0, 3, offset), &[19])?;
            }
            Ok(())
        }

        fn read_from(&mut self, source: &dyn SectionSource) -> Result<()> {
            read_raw(self, source)
        }

        fn is_dirty(&self) -> bool {
            true
        }

        fn mark_clean(&self) {}

        fn memory_usage(&self) -> usize {
            self.0
        }
    }

    fn fixed(section_type: SectionType, data: impl Into<Vec<u8>>) -> Fixed {
        Fixed {
            section_type,
            data: data.into(),
        }
    }

    fn test_dir() -> TempDir {
        TempDir::new().expect("create temp dir")
    }

    fn header(epoch: u64) -> CheckpointHeader {
        CheckpointHeader {
            checkpoint_lsn: 19,
            epoch,
            last_transaction_id: 88,
            node_count: 3,
            edge_count: 319,
        }
    }

    fn checkpoint(manager: &GrafeoFileManager, sections: &[Fixed], epoch: u64) -> Result<()> {
        let sections: Vec<&dyn Section> = sections.iter().map(|s| s as &dyn Section).collect();
        manager.write_checkpoint(&sections, &header(epoch))
    }

    /// The bytes of a section of the active image stored as one raw chunk,
    /// `None` when the image has no such section.
    fn raw_section(
        manager: &GrafeoFileManager,
        section_type: SectionType,
    ) -> Result<Option<Vec<u8>>> {
        manager.read_image(|image| match image.section_source(section_type) {
            Some(section) => Ok(legacy_bytes(&*section)?.map(Vec::from)),
            None => Ok(None),
        })
    }

    fn read(manager: &GrafeoFileManager, section_type: SectionType) -> Option<Vec<u8>> {
        raw_section(manager, section_type).unwrap()
    }

    /// The bytes of the database file, read through the locked handle
    /// (reading the path directly fails on Windows while it is locked).
    #[cfg(feature = "testing-crash-injection")]
    fn file_bytes(manager: &GrafeoFileManager, dir: &TempDir) -> Vec<u8> {
        let copy = dir.path().join("copy.bin");
        manager.copy_to(&copy).unwrap();
        fs::read(&copy).unwrap()
    }

    /// The pages the image rooted at `root` uses, read from a copy of the file.
    #[cfg(feature = "testing-crash-injection")]
    fn image_runs(
        bytes: &[u8],
        root: super::super::v3::header::BlockRef,
        dir: &TempDir,
    ) -> Vec<PageRun> {
        let copy = dir.path().join("runs.bin");
        fs::write(&copy, bytes).unwrap();
        let mut file = File::open(&copy).unwrap();
        ImageReader::open(&mut file, root, None)
            .unwrap()
            .used_runs()
    }

    fn slot_bytes(bytes: &[u8], slot: u8) -> &[u8] {
        let start = usize::try_from(slot_offset(slot)).unwrap();
        &bytes[start..start + 4096]
    }

    /// The active slot and header of a file, decided from its bytes alone
    /// (not from the manager's state).
    fn active_slot_of(bytes: &[u8]) -> (u8, DbHeaderV3) {
        active_header([
            DbHeaderV3::decode(slot_bytes(bytes, 0)),
            DbHeaderV3::decode(slot_bytes(bytes, 1)),
        ])
        .unwrap()
        .expect("a valid database header")
    }

    /// Asserts that every page of `runs` holds the same bytes in `now` as in
    /// `before` (a file cut inside a run fails too).
    #[cfg(feature = "testing-crash-injection")]
    fn assert_runs_unchanged(now: &[u8], before: &[u8], runs: &[PageRun], what: &str) {
        for run in runs {
            let start = usize::try_from(run.offset().unwrap()).unwrap();
            let end = usize::try_from(run.end().unwrap() * PAGE_SIZE).unwrap();
            assert!(before.get(start..end).is_some(), "{what}: run {run:?}");
            assert!(
                now.get(start..end) == before.get(start..end),
                "{what}: image A's run {run:?} was overwritten or cut"
            );
        }
    }

    /// Asserts that `manager` reads back every section of `image`.
    #[cfg(feature = "testing-crash-injection")]
    fn assert_holds(manager: &GrafeoFileManager, image: &[Fixed], what: &str) {
        for section in image {
            assert_eq!(
                read(manager, section.section_type).as_deref(),
                Some(section.data.as_slice()),
                "{what}: {:?}",
                section.section_type
            );
        }
    }

    fn overwrite(path: &Path, offset: u64, bytes: &[u8]) {
        let mut file = OpenOptions::new().write(true).open(path).unwrap();
        file.seek(SeekFrom::Start(offset)).unwrap();
        file.write_all(bytes).unwrap();
    }

    #[test]
    fn a_new_database_reopens_at_iteration_0_without_sections() {
        let dir = test_dir();
        let path = dir.path().join("alix.grafeo");
        let manager = GrafeoFileManager::create(&path, None).unwrap();
        let database_id = manager.database_id();
        assert_eq!(manager.active_header().iteration, 0);
        assert_eq!(read(&manager, SectionType::Catalog), None);
        assert_eq!(
            manager.file_size().unwrap(),
            4 * 4096,
            "three header pages and one directory page"
        );
        drop(manager);

        let manager = GrafeoFileManager::open(&path, None).unwrap();
        assert_eq!(manager.active_header().iteration, 0);
        assert_eq!(manager.database_id(), database_id, "the id survives reopen");
        for section_type in [
            SectionType::Catalog,
            SectionType::LpgStore,
            SectionType::RdfStore,
            SectionType::VectorStore,
        ] {
            assert_eq!(read(&manager, section_type), None, "{section_type:?}");
        }
    }

    #[test]
    fn create_writes_a_valid_iteration_0_header_and_leaves_slot_1_empty() {
        let dir = test_dir();
        let path = dir.path().join("gus.grafeo");
        drop(GrafeoFileManager::create(&path, None).unwrap());
        let bytes = fs::read(&path).unwrap();
        let HeaderSlot::Valid(first) = DbHeaderV3::decode(slot_bytes(&bytes, 0)) else {
            panic!("slot 0 holds a valid header");
        };
        assert_eq!(first.iteration, 0);
        assert_ne!(
            first.root.length, 0,
            "the empty image has a directory block"
        );
        assert_eq!(DbHeaderV3::decode(slot_bytes(&bytes, 1)), HeaderSlot::Empty);
    }

    #[test]
    fn database_ids_differ_between_databases() {
        let dir = test_dir();
        let first = GrafeoFileManager::create(dir.path().join("a.grafeo"), None).unwrap();
        let second = GrafeoFileManager::create(dir.path().join("b.grafeo"), None).unwrap();
        assert_ne!(first.database_id(), second.database_id());
    }

    #[test]
    fn a_second_checkpoint_replaces_the_first_after_reopen() {
        let dir = test_dir();
        let path = dir.path().join("vincent.grafeo");
        let manager = GrafeoFileManager::create(&path, None).unwrap();
        checkpoint(
            &manager,
            &[
                fixed(SectionType::Catalog, "Alix"),
                fixed(SectionType::LpgStore, "Berlin"),
            ],
            3,
        )
        .unwrap();
        let first = manager.active_header();
        assert_eq!(first.iteration, 1);
        assert_eq!(
            (
                first.checkpoint_lsn,
                first.epoch,
                first.last_transaction_id,
                first.node_count,
                first.edge_count
            ),
            (19, 3, 88, 3, 319),
            "the header carries the checkpoint values"
        );
        assert!(first.timestamp_ms > 0);
        drop(manager);

        let manager = GrafeoFileManager::open(&path, None).unwrap();
        assert_eq!(manager.active_header(), first);
        assert_eq!(read(&manager, SectionType::Catalog), Some(b"Alix".to_vec()));
        assert_eq!(
            read(&manager, SectionType::LpgStore),
            Some(b"Berlin".to_vec())
        );

        checkpoint(&manager, &[fixed(SectionType::Catalog, "Gus")], 19).unwrap();
        assert_eq!(manager.active_header().iteration, 2);
        assert_eq!(read(&manager, SectionType::Catalog), Some(b"Gus".to_vec()));
        drop(manager);

        let manager = GrafeoFileManager::open(&path, None).unwrap();
        let second = manager.active_header();
        assert_eq!((second.iteration, second.epoch), (2, 19));
        assert_eq!(read(&manager, SectionType::Catalog), Some(b"Gus".to_vec()));
        assert_eq!(
            read(&manager, SectionType::LpgStore),
            None,
            "a section the new image does not hold is gone"
        );
    }

    #[test]
    fn repeated_checkpoints_reuse_space() {
        let dir = test_dir();
        let path = dir.path().join("mia.grafeo");
        let manager = GrafeoFileManager::create(&path, None).unwrap();
        let sections = [
            fixed(SectionType::Catalog, vec![3u8; 1024]),
            fixed(SectionType::LpgStore, vec![19u8; 9 * 1024]),
            fixed(SectionType::VectorStore, vec![88u8; 100 * 1024]),
        ];
        let mut after_second = 0;
        for round in 1..=50u64 {
            checkpoint(&manager, &sections, round).unwrap();
            if round == 2 {
                after_second = manager.file_size().unwrap();
            }
        }
        let after_fiftieth = manager.file_size().unwrap();
        assert!(
            after_fiftieth <= 2 * after_second,
            "the file grew from {after_second} to {after_fiftieth} bytes: pages of replaced images are not reused"
        );
        assert_eq!(manager.active_header().iteration, 50);
        drop(manager);
        let manager = GrafeoFileManager::open(&path, None).unwrap();
        assert_eq!(
            read(&manager, SectionType::VectorStore),
            Some(vec![88u8; 100 * 1024])
        );
    }

    #[test]
    fn a_torn_inactive_header_opens_the_previous_image() {
        let dir = test_dir();
        let path = dir.path().join("jules.grafeo");
        let manager = GrafeoFileManager::create(&path, None).unwrap();
        checkpoint(&manager, &[fixed(SectionType::Catalog, "Alix")], 1).unwrap();
        drop(manager);
        let before = fs::read(&path).unwrap();
        let (active_slot, active) = active_slot_of(&before);
        assert_eq!(active.iteration, 1, "the checkpoint's header is active");
        let next_slot = 1 - active_slot;

        // A torn write of the next header: half a header, then garbage.
        let mut torn = DbHeaderV3 {
            iteration: 2,
            ..DbHeaderV3::default()
        }
        .encode();
        torn[40..].fill(88);
        overwrite(&path, slot_offset(next_slot), &torn);

        let manager = GrafeoFileManager::open(&path, None).unwrap();
        assert_eq!(manager.active_header().iteration, 1);
        assert_eq!(read(&manager, SectionType::Catalog), Some(b"Alix".to_vec()));

        // The next checkpoint writes over the torn slot, not the active one.
        checkpoint(&manager, &[fixed(SectionType::Catalog, "Gus")], 2).unwrap();
        drop(manager);
        let after = fs::read(&path).unwrap();
        let (slot, header) = active_slot_of(&after);
        assert_eq!((slot, header.iteration), (next_slot, 2), "the new header");
        assert_eq!(
            slot_bytes(&after, active_slot),
            slot_bytes(&before, active_slot),
            "the previous header is not overwritten"
        );
        let manager = GrafeoFileManager::open(&path, None).unwrap();
        assert_eq!(manager.active_header().iteration, 2);
        assert_eq!(read(&manager, SectionType::Catalog), Some(b"Gus".to_vec()));
    }

    #[test]
    fn a_file_without_a_valid_header_does_not_open() {
        let dir = test_dir();
        for (name, slot_0, expected) in [
            ("empty", [0u8; 4096], "both database header slots are empty"),
            (
                "damaged",
                [3u8; 4096],
                "slot 0 is damaged and slot 1 was never written",
            ),
        ] {
            let path = dir.path().join(format!("{name}.grafeo"));
            drop(GrafeoFileManager::create(&path, None).unwrap());
            overwrite(&path, slot_offset(0), &slot_0);
            let error = GrafeoFileManager::open(&path, None)
                .map(|_| ())
                .unwrap_err()
                .to_string();
            assert!(
                error.contains(&path.display().to_string()) && error.contains(expected),
                "{name}: the error names the file and what is wrong with its headers: {error}"
            );
        }
    }

    #[test]
    fn a_header_without_a_directory_block_is_refused() {
        let dir = test_dir();
        let path = dir.path().join("mia.grafeo");
        drop(GrafeoFileManager::create(&path, None).unwrap());
        let empty_root = DbHeaderV3::default();
        assert_eq!(empty_root.root.length, 0);
        overwrite(&path, slot_offset(0), &empty_root.encode());
        for read_only in [false, true] {
            let result = if read_only {
                GrafeoFileManager::open_read_only(&path, None)
            } else {
                GrafeoFileManager::open(&path, None)
            };
            let error = result.map(|_| ()).unwrap_err().to_string();
            assert!(
                error.contains("no directory block"),
                "read-only {read_only}: {error}"
            );
        }

        // Without a block to decrypt, any key would open an encrypted file.
        #[cfg(feature = "encryption")]
        {
            let encrypted = dir.path().join("vincent.grafeo");
            drop(GrafeoFileManager::create(&encrypted, key(3)).unwrap());
            overwrite(&encrypted, slot_offset(0), &empty_root.encode());
            let error = GrafeoFileManager::open(&encrypted, key(19))
                .map(|_| ())
                .unwrap_err()
                .to_string();
            assert!(error.contains("no directory block"), "{error}");
        }
    }

    /// A file whose active header has a format revision this build does not
    /// read (0 from a 0.6.0 development build, or one a newer Grafeo wrote)
    /// is refused by every open, before its directory is read, and the file
    /// keeps every byte.
    #[test]
    fn a_format_revision_this_build_does_not_read_is_refused_and_the_file_kept() {
        let dir = test_dir();
        for revision in [0, MAX_FORMAT_REVISION + 1, u32::MAX] {
            let path = dir.path().join(format!("revision-{revision}.grafeo"));
            let manager = GrafeoFileManager::create(&path, None).unwrap();
            checkpoint(&manager, &[fixed(SectionType::Catalog, "Alix")], 3).unwrap();
            drop(manager);
            let bytes = fs::read(&path).unwrap();
            let (slot, active) = active_slot_of(&bytes);
            let revised = DbHeaderV3 {
                format_revision: revision,
                ..active
            };
            overwrite(&path, slot_offset(slot), &revised.encode());
            let before = fs::read(&path).unwrap();
            for read_only in [false, true] {
                let result = if read_only {
                    GrafeoFileManager::open_read_only(&path, None)
                } else {
                    GrafeoFileManager::open(&path, None)
                };
                let error = result.map(|_| ()).unwrap_err().to_string();
                assert!(
                    error.contains(&path.display().to_string())
                        && error.contains(&format!("format revision {revision}")),
                    "revision {revision}, read-only {read_only}: {error}"
                );
            }
            assert_eq!(
                fs::read(&path).unwrap(),
                before,
                "revision {revision}: a refused file is left as it is"
            );
        }
    }

    /// Every header a manager writes carries a format revision: a new file
    /// the current one, a checkpoint the revision of the header it replaces
    /// (a file is upgraded only by an explicit step, never by a checkpoint).
    #[test]
    fn a_checkpoint_keeps_the_format_revision_of_the_file() {
        let dir = test_dir();
        let path = dir.path().join("jules.grafeo");
        drop(GrafeoFileManager::create(&path, None).unwrap());
        let created = active_slot_of(&fs::read(&path).unwrap()).1;
        assert_eq!(created.format_revision, FORMAT_REVISION, "a new file");
        let manager = GrafeoFileManager::open(&path, None).unwrap();
        checkpoint(&manager, &[fixed(SectionType::Catalog, "Alix")], 3).unwrap();
        assert_eq!(manager.active_header().format_revision, FORMAT_REVISION);
        drop(manager);
        let manager = GrafeoFileManager::open(&path, None).unwrap();
        // A revision the build reads but does not give new files: the
        // checkpoint writes it again.
        manager.active.lock().header.format_revision = 7;
        checkpoint(&manager, &[fixed(SectionType::Catalog, "Gus")], 19).unwrap();
        assert_eq!(manager.active_header().format_revision, 7);
        drop(manager);
        let bytes = fs::read(&path).unwrap();
        let (slot, written) = active_slot_of(&bytes);
        assert_eq!(
            (written.iteration, written.format_revision),
            (2, 7),
            "slot {slot}: the header on disk keeps the revision"
        );
    }

    /// A section of two chunks is served as both, in order; it is not the
    /// one raw chunk of 0.5.x section bytes.
    #[test]
    fn a_section_of_two_chunks_is_served_as_two_chunks() {
        let dir = test_dir();
        let manager = GrafeoFileManager::create(dir.path().join("hans.grafeo"), None).unwrap();
        manager.write_checkpoint(&[&TwoChunks], &header(1)).unwrap();
        let chunks = manager
            .read_image(|image| {
                let section = image.section_source(SectionType::LpgStore).unwrap();
                (0..section.chunks().len())
                    .map(|index| Ok(section.fetch(index)?.to_vec()))
                    .collect::<Result<Vec<_>>>()
            })
            .unwrap();
        assert_eq!(chunks, [b"Vincent".to_vec(), b"Jules".to_vec()]);
        let error = raw_section(&manager, SectionType::LpgStore)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("2 raw chunks"),
            "two raw chunks are not 0.5.x section bytes: {error}"
        );
    }

    #[test]
    fn read_image_serves_the_sections_of_the_active_image() {
        let dir = test_dir();
        let manager = GrafeoFileManager::create(dir.path().join("db.grafeo"), None).unwrap();
        assert_eq!(
            manager.image_stats().unwrap(),
            ImageStats {
                chunks: 0,
                directory_blocks: 1,
                pages: 1
            },
            "a new database: one directory block, no header pages counted"
        );
        checkpoint(
            &manager,
            &[
                fixed(SectionType::Catalog, "Alix"),
                fixed(SectionType::LpgStore, "Gus"),
            ],
            3,
        )
        .unwrap();
        manager
            .read_image(|image| {
                let catalog = image.section_source(SectionType::Catalog).unwrap();
                assert_eq!(
                    (catalog.section_version(), &catalog.fetch(0)?[..]),
                    (1, &b"Alix"[..])
                );
                assert!(image.section_source(SectionType::RdfStore).is_none());
                Ok(())
            })
            .unwrap();
        assert_eq!(
            manager.image_stats().unwrap(),
            ImageStats {
                chunks: 2,
                directory_blocks: 1,
                pages: 3
            }
        );
    }

    #[test]
    fn image_stats_count_every_directory_block() {
        use super::super::v3::directory::ENTRIES_PER_BLOCK;

        let dir = test_dir();
        let manager = GrafeoFileManager::create(dir.path().join("db.grafeo"), None).unwrap();
        let pieces = Pieces(ENTRIES_PER_BLOCK + 1);
        manager.write_checkpoint(&[&pieces], &header(3)).unwrap();
        let stats = manager.image_stats().unwrap();
        assert_eq!(
            (stats.chunks, stats.directory_blocks),
            (ENTRIES_PER_BLOCK + 1, 2)
        );
    }

    /// An error the closure returns comes back from `read_image`.
    #[test]
    fn read_image_returns_the_error_of_its_closure() {
        let dir = test_dir();
        let manager = GrafeoFileManager::create(dir.path().join("db.grafeo"), None).unwrap();
        let error = manager
            .read_image(|_| -> Result<()> { Err(Error::Internal("Vincent".to_string())) })
            .unwrap_err()
            .to_string();
        assert!(error.contains("Vincent"), "{error}");
    }

    #[test]
    fn image_stats_count_a_chunk_without_bytes_and_no_page_for_it() {
        let dir = test_dir();
        let manager = GrafeoFileManager::create(dir.path().join("db.grafeo"), None).unwrap();
        checkpoint(&manager, &[fixed(SectionType::Catalog, "")], 3).unwrap();
        assert_eq!(
            manager.image_stats().unwrap(),
            ImageStats {
                chunks: 1,
                directory_blocks: 1,
                pages: 1
            },
            "an empty chunk counts as a chunk and takes no page"
        );
    }

    /// Writes a newer version's checkpoint through the file itself, as
    /// iteration 1 in slot 1 over the image `create` wrote: the catalog chunk
    /// "Alix" and, after it, the `foreign` chunks as (section type byte, chunk
    /// kind byte, flags, bytes). Returns the entries a reader skips.
    fn checkpoint_as_a_newer_version(
        path: &Path,
        foreign: &[(u8, u8, u8, &[u8])],
    ) -> Vec<SkippedEntry> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .unwrap();
        let prefix = read_prefix(&mut file, DATA_START_PAGE * PAGE_SIZE).unwrap();
        let (active_slot, active) = active_slot_of(&prefix);
        assert_eq!((active_slot, active.iteration), (0, 0));
        let active_runs = ImageReader::open(&mut file, active.root, None)
            .unwrap()
            .used_runs();
        let mut writer = CheckpointWriter::new(
            &mut file,
            PageAllocator::from_used(active_runs).unwrap(),
            None,
        );
        writer.begin_section(SectionType::Catalog, 1).unwrap();
        writer.write_chunk(ChunkMeta::raw(), b"Alix").unwrap();
        for (section_type, kind, flags, bytes) in foreign {
            writer
                .write_foreign_chunk(*section_type, *kind, *flags, bytes)
                .unwrap();
        }
        let (root, _) = writer.finish().unwrap();
        let next = DbHeaderV3 {
            iteration: 1,
            root,
            ..DbHeaderV3::default()
        };
        write_page(&mut file, slot_offset(1), &next.encode()).unwrap();
        file.sync_all().unwrap();
        ImageReader::open(&mut file, root, None)
            .unwrap()
            .skipped()
            .to_vec()
    }

    /// A file a newer version checkpointed with an optional section this
    /// version does not know opens; the section's pages stay spared while its
    /// image is active, and the next checkpoint, which cannot write the
    /// section, drops it.
    #[test]
    fn a_file_with_an_unknown_optional_section_opens_and_its_next_checkpoint_drops_it() {
        let dir = test_dir();
        let path = dir.path().join("amsterdam.grafeo");
        drop(GrafeoFileManager::create(&path, None).unwrap());
        let skipped = checkpoint_as_a_newer_version(
            &path,
            &[(250, 0, ENTRY_SECTION_OPTIONAL, &[19u8; 9000])],
        );
        assert_eq!(skipped.len(), 1);
        let foreign = skipped[0];

        let manager = GrafeoFileManager::open(&path, None).unwrap();
        assert_eq!(manager.active_header().iteration, 1);
        manager
            .read_image(|image| {
                let catalog = image.section_source(SectionType::Catalog).unwrap();
                assert_eq!(&catalog.fetch(0)?[..], b"Alix");
                Ok(())
            })
            .unwrap();
        assert_eq!(read(&manager, SectionType::Catalog), Some(b"Alix".to_vec()));
        let with_foreign = manager.image_stats().unwrap();
        assert_eq!(
            with_foreign,
            ImageStats {
                chunks: 1,
                directory_blocks: 1,
                pages: 1 + 3 + 1
            },
            "the chunks count the known chunk; the pages count the skipped section's three"
        );

        checkpoint(&manager, &[fixed(SectionType::Catalog, "Alix")], 2).unwrap();
        drop(manager);
        let bytes = fs::read(&path).unwrap();
        let start = usize::try_from(foreign.offset).unwrap();
        assert_eq!(
            bytes.get(start..start + 9000),
            Some(&[19u8; 9000][..]),
            "the checkpoint wrote no page of the skipped section while its image was active"
        );

        let manager = GrafeoFileManager::open(&path, None).unwrap();
        assert_eq!(manager.active_header().iteration, 2);
        assert_eq!(read(&manager, SectionType::Catalog), Some(b"Alix".to_vec()));
        assert_eq!(
            manager.image_stats().unwrap().pages,
            with_foreign.pages - 3,
            "the skipped section was not written again"
        );
    }

    /// The open warns about each skipped section type and chunk kind once;
    /// reads of the image after it, and a checkpoint, do not warn again.
    #[test]
    fn each_skipped_section_type_and_kind_is_warned_about_once_at_open() {
        let dir = test_dir();
        let path = dir.path().join("berlin.grafeo");
        drop(GrafeoFileManager::create(&path, None).unwrap());
        let skipped = checkpoint_as_a_newer_version(
            &path,
            &[
                (250, 0, ENTRY_SECTION_OPTIONAL, b"Vincent"),
                (250, 0, ENTRY_SECTION_OPTIONAL, b"Jules"),
                (
                    SectionType::Catalog.to_u8(),
                    250,
                    ENTRY_CHUNK_OPTIONAL,
                    b"Mia",
                ),
            ],
        );
        assert_eq!(skipped.len(), 3, "three skipped entries, two pairs");
        // Every warning logged on this thread, whichever record logged it.
        WARNINGS_LOGGED.with_borrow_mut(Vec::clear);
        let logged = || WARNINGS_LOGGED.with_borrow(Clone::clone);
        let manager = GrafeoFileManager::open(&path, None).unwrap();
        let warned = vec![(250, 0), (1, 250)];
        assert_eq!(logged(), warned, "the open warns about each pair once");
        for _ in 0..2 {
            manager
                .read_image(|image| {
                    let catalog = image.section_source(SectionType::Catalog).unwrap();
                    assert_eq!(&catalog.fetch(0)?[..], b"Alix");
                    Ok(())
                })
                .unwrap();
        }
        manager.image_stats().unwrap();
        assert_eq!(read(&manager, SectionType::Catalog), Some(b"Alix".to_vec()));
        assert_eq!(
            logged(),
            warned,
            "three read_image calls and image_stats warn about nothing again"
        );
        checkpoint(&manager, &[fixed(SectionType::Catalog, "Alix")], 2).unwrap();
        manager.image_stats().unwrap();
        assert_eq!(logged(), warned, "nor does a checkpoint or a read after it");
    }

    /// One warning per distinct section type and kind, in the order the
    /// directory lists them.
    #[test]
    fn each_skipped_section_type_and_kind_is_named_once() {
        let entry = |section_type, kind| SkippedEntry {
            section_type,
            kind,
            flags: ENTRY_SECTION_OPTIONAL,
            offset: 12_288,
            length: 19,
        };
        let image = [
            entry(250, 0),
            entry(250, 0),
            entry(2, 250),
            entry(250, 3),
            entry(2, 250),
        ];
        assert_eq!(
            skipped_kinds(&image),
            [(250, 0), (2, 250), (250, 3)],
            "a repeated pair is named once, a new kind of the same type again"
        );
        assert_eq!(
            skipped_kinds(&[]),
            Vec::<(u8, u8)>::new(),
            "no skipped entry, no warning"
        );
    }

    /// `read_image` on one thread and checkpoints on another take turns, and
    /// every read is served the image the active header names, whole. A
    /// deadlock between them fails the test after 60 seconds instead of
    /// hanging it.
    #[test]
    fn read_image_serves_the_active_image_while_checkpoints_run() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::mpsc::{self, RecvTimeoutError};
        use std::time::Duration;

        /// Sets its flag when dropped, also when its thread panics.
        struct SetOnDrop(Arc<AtomicBool>);

        impl Drop for SetOnDrop {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }

        fn round(n: u64) -> [Fixed; 2] {
            let name = format!("Amsterdam {n}");
            [
                fixed(SectionType::Catalog, name.clone()),
                fixed(SectionType::LpgStore, name),
            ]
        }

        let dir = test_dir();
        let manager =
            Arc::new(GrafeoFileManager::create(dir.path().join("db.grafeo"), None).unwrap());
        checkpoint(&manager, &round(0), 0).unwrap();
        let checkpoints_done = Arc::new(AtomicBool::new(false));
        let (finished, finishes) = mpsc::channel();
        // Plain threads, not scoped ones: a scope joins its threads even while
        // it unwinds, so a deadlock would hang the test.
        let checkpoints = {
            let manager = Arc::clone(&manager);
            let done = SetOnDrop(Arc::clone(&checkpoints_done));
            let finished = finished.clone();
            std::thread::spawn(move || {
                let _done = done;
                for n in 1..=19 {
                    checkpoint(&manager, &round(n), n).unwrap();
                }
                finished.send(()).unwrap();
            })
        };
        let reads = {
            let manager = Arc::clone(&manager);
            std::thread::spawn(move || {
                let mut count = 0u32;
                // The reads go on for as long as the checkpoints do, so the
                // two overlap.
                while count < 88 || !checkpoints_done.load(Ordering::Acquire) {
                    manager
                        .read_image(|image| {
                            let catalog = image
                                .section_source(SectionType::Catalog)
                                .unwrap()
                                .fetch(0)?;
                            let store = image
                                .section_source(SectionType::LpgStore)
                                .unwrap()
                                .fetch(0)?;
                            let active = format!("Amsterdam {}", manager.active_header().epoch);
                            assert_eq!(
                                String::from_utf8_lossy(&catalog),
                                active,
                                "the image served is the active one"
                            );
                            assert_eq!(catalog, store, "both sections come from one image");
                            Ok(())
                        })
                        .unwrap();
                    count += 1;
                }
                finished.send(()).unwrap();
            })
        };
        for _ in 0..2 {
            match finishes.recv_timeout(Duration::from_secs(60)) {
                Ok(()) => {}
                // A thread panicked: its join below reports why.
                Err(RecvTimeoutError::Disconnected) => break,
                Err(RecvTimeoutError::Timeout) => panic!(
                    "read_image and write_checkpoint did not finish within 60 seconds: they \
                     deadlock"
                ),
            }
        }
        for thread in [checkpoints, reads] {
            if let Err(panic) = thread.join() {
                std::panic::resume_unwind(panic);
            }
        }
        assert_eq!(manager.active_header().epoch, 19);
    }

    #[cfg(feature = "encryption")]
    #[test]
    fn read_image_on_an_encrypted_file_serves_the_plaintext() {
        let dir = test_dir();
        let path = dir.path().join("prague.grafeo");
        let manager = GrafeoFileManager::create(&path, key(3)).unwrap();
        checkpoint(
            &manager,
            &[fixed(SectionType::Catalog, "Mia lives in Prague")],
            1,
        )
        .unwrap();
        let served = |manager: &GrafeoFileManager| {
            manager
                .read_image(|image| image.section_source(SectionType::Catalog).unwrap().fetch(0))
                .unwrap()
        };
        assert_eq!(&served(&manager)[..], b"Mia lives in Prague");
        drop(manager);
        assert!(
            !fs::read(&path)
                .unwrap()
                .windows(6)
                .any(|window| window == b"Prague"),
            "the file holds no plaintext"
        );
        let reader = GrafeoFileManager::open_read_only(&path, key(3)).unwrap();
        assert_eq!(
            &served(&reader)[..],
            b"Mia lives in Prague",
            "a reopen decrypts with the key"
        );
    }

    #[test]
    fn a_0_5_file_must_be_migrated_first() {
        let dir = test_dir();
        let path = dir.path().join("closed.grafeo");
        fs::copy(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../grafeo-engine/tests/fixtures/released/0.5.44/closed.grafeo"
            ),
            &path,
        )
        .unwrap();
        let before = fs::read(&path).unwrap();
        for read_only in [false, true] {
            let result = if read_only {
                GrafeoFileManager::open_read_only(&path, None)
            } else {
                GrafeoFileManager::open(&path, None)
            };
            let error = result.map(|_| ()).unwrap_err().to_string();
            assert!(
                error.contains("0.5") && error.contains("migrat"),
                "read-only {read_only}: {error}"
            );
        }
        assert_eq!(fs::read(&path).unwrap(), before, "the file is untouched");
    }

    /// Runs checkpoint B over image A once per crash point (in the order
    /// `write_checkpoint` reaches them), plus once to completion, and checks
    /// the file's bytes after each run: A's header slot is never written; B's
    /// header lands in the other slot, and only from the header write on;
    /// A's pages are intact until the trim (through `checkpoint:before_trim`;
    /// A stays the durable image through `checkpoint:after_header`, whose
    /// header is not yet synced); a reopen finds A before B's header write and
    /// B after it.
    ///
    /// `before_a` are checkpoints taken before A. Returns the file size after
    /// A and after the completed checkpoint B.
    ///
    /// In-process injection keeps every write in the page cache, so this
    /// checks the order of the writes, not their durability: a missing
    /// `sync_all` cannot be caught this way.
    #[cfg(feature = "testing-crash-injection")]
    fn crash_checkpoint_b_over_a(
        label: &str,
        before_a: &[&[Fixed]],
        image_a: &[Fixed],
        image_b: &[Fixed],
    ) -> (usize, usize) {
        use grafeo_common::testing::crash::{CrashResult, with_crash_at};

        const POINTS: [&str; 4] = [
            "checkpoint:after_chunks",
            "checkpoint:after_data_sync",
            "checkpoint:after_header",
            "checkpoint:before_trim",
        ];
        let mut sizes = (0, 0);
        for crash_after in 1..=POINTS.len() + 1 {
            let point = POINTS.get(crash_after - 1).copied().unwrap_or("no crash");
            let what = format!("{label}, {point}");
            let dir = test_dir();
            let path = dir.path().join("db.grafeo");
            let manager = GrafeoFileManager::create(&path, None).unwrap();
            for (epoch, image) in (1..).zip(before_a) {
                checkpoint(&manager, image, epoch).unwrap();
            }
            checkpoint(&manager, image_a, 19).unwrap();
            let a_bytes = file_bytes(&manager, &dir);
            let (a_slot, a_header) = active_slot_of(&a_bytes);
            assert_eq!(a_header, manager.active_header(), "{what}: A is active");
            let a_runs = image_runs(&a_bytes, a_header.root, &dir);

            let target = AssertUnwindSafe(&manager);
            let count = u64::try_from(crash_after).unwrap();
            let result = with_crash_at(count, move || checkpoint(*target, image_b, 88));
            match result {
                CrashResult::Crashed => assert!(crash_after <= POINTS.len(), "{what}"),
                CrashResult::Completed(outcome) => {
                    outcome.unwrap();
                    assert_eq!(
                        crash_after,
                        POINTS.len() + 1,
                        "{what}: a checkpoint has four points"
                    );
                }
                _ => unreachable!(),
            }
            drop(manager);

            let now = fs::read(&path).unwrap();
            assert!(
                slot_bytes(&now, a_slot) == slot_bytes(&a_bytes, a_slot),
                "{what}: A's header slot {a_slot} was written"
            );
            let b_slot = 1 - a_slot;
            let b_header_written = crash_after > 2;
            if b_header_written {
                let HeaderSlot::Valid(b_header) = DbHeaderV3::decode(slot_bytes(&now, b_slot))
                else {
                    panic!("{what}: B's header is not in slot {b_slot}");
                };
                assert_eq!(b_header.iteration, a_header.iteration + 1, "{what}");
            } else {
                assert!(
                    slot_bytes(&now, b_slot) == slot_bytes(&a_bytes, b_slot),
                    "{what}: slot {b_slot} was written before the header write"
                );
            }
            if crash_after <= 4 {
                assert_runs_unchanged(&now, &a_bytes, &a_runs, &what);
            }

            let reopened = GrafeoFileManager::open(&path, None).unwrap();
            if b_header_written {
                assert_eq!(
                    reopened.active_header().iteration,
                    a_header.iteration + 1,
                    "{what}"
                );
                assert_holds(&reopened, image_b, &what);
            } else {
                assert_eq!(reopened.active_header(), a_header, "{what}");
                assert_holds(&reopened, image_a, &what);
            }
            sizes = (a_bytes.len(), now.len());
        }
        sizes
    }

    #[cfg(feature = "testing-crash-injection")]
    #[test]
    fn a_checkpoint_never_overwrites_pages_the_active_header_reaches() {
        let image_a = [
            fixed(SectionType::Catalog, "Alix"),
            fixed(SectionType::LpgStore, vec![3u8; 9000]),
        ];
        let image_b = [
            fixed(SectionType::Catalog, "Gus"),
            fixed(SectionType::LpgStore, vec![19u8; 20_000]),
        ];
        crash_checkpoint_b_over_a("B larger than A", &[], &image_a, &image_b);
    }

    /// B fits in the gaps below A (the pages of the image before A), so the
    /// trim after B cuts A's pages: it must come after B's header.
    #[cfg(feature = "testing-crash-injection")]
    #[test]
    fn a_smaller_checkpoint_cuts_the_file_only_after_its_header() {
        let before_a = [fixed(SectionType::LpgStore, vec![88u8; 100 * 1024])];
        let image_a = [
            fixed(SectionType::Catalog, "Alix"),
            fixed(SectionType::LpgStore, vec![3u8; 9000]),
        ];
        let image_b = [
            fixed(SectionType::Catalog, "Gus"),
            fixed(SectionType::LpgStore, "Paris"),
        ];
        let (after_a, after_b) =
            crash_checkpoint_b_over_a("B smaller than A", &[&before_a], &image_a, &image_b);
        assert!(
            after_b < after_a,
            "B's trim cuts the file from {after_a} to {after_b} bytes, so A's last pages go"
        );
    }

    /// A checkpoint whose header may have reached the disk before it failed
    /// leaves two images that can be active: the next checkpoint of the same
    /// manager overwrites neither. A stays a possible durable image because
    /// B's header was never synced, and B because its header was written.
    #[cfg(feature = "testing-crash-injection")]
    #[test]
    fn a_checkpoint_after_a_failed_one_spares_both_images() {
        use grafeo_common::testing::crash::{CrashResult, with_crash_at};

        let dir = test_dir();
        let path = dir.path().join("db.grafeo");
        let manager = GrafeoFileManager::create(&path, None).unwrap();
        checkpoint(
            &manager,
            &[fixed(SectionType::LpgStore, vec![3u8; 9000])],
            1,
        )
        .unwrap();
        let a_bytes = file_bytes(&manager, &dir);
        let (a_slot, a_header) = active_slot_of(&a_bytes);
        let a_runs = image_runs(&a_bytes, a_header.root, &dir);

        // B fails right after its header write (count 3: checkpoint:after_header).
        let target = AssertUnwindSafe(&manager);
        let image_b = [fixed(SectionType::LpgStore, vec![19u8; 9000])];
        let result = with_crash_at(3, move || checkpoint(*target, &image_b, 2));
        assert!(matches!(result, CrashResult::Crashed));

        // C fails after writing its chunks (count 1), before its header.
        let target = AssertUnwindSafe(&manager);
        let image_c = [fixed(SectionType::LpgStore, vec![88u8; 9000])];
        let result = with_crash_at(1, move || checkpoint(*target, &image_c, 3));
        assert!(matches!(result, CrashResult::Crashed));
        let copy = dir.path().join("after_c.grafeo");
        manager.copy_to(&copy).unwrap();
        let after_c = fs::read(&copy).unwrap();
        assert!(
            slot_bytes(&after_c, a_slot) == slot_bytes(&a_bytes, a_slot),
            "A's header slot was written"
        );
        assert_runs_unchanged(&after_c, &a_bytes, &a_runs, "after C");
        let reopened = GrafeoFileManager::open(&copy, None).unwrap();
        assert_eq!(
            read(&reopened, SectionType::LpgStore),
            Some(vec![19u8; 9000]),
            "B's header is on disk, so B is active and C did not write over it"
        );
        drop(reopened);

        // D completes, and is what the file holds from then on.
        checkpoint(&manager, &[fixed(SectionType::LpgStore, "Paris")], 4).unwrap();
        assert_eq!(
            read(&manager, SectionType::LpgStore),
            Some(b"Paris".to_vec())
        );
        drop(manager);
        let reopened = GrafeoFileManager::open(&path, None).unwrap();
        assert_eq!(
            read(&reopened, SectionType::LpgStore),
            Some(b"Paris".to_vec())
        );
    }

    /// An injected failure at each point of `write_checkpoint` returns an
    /// error naming it, as a real I/O error there would. The manager keeps
    /// serving the previous image (its header and its sections); a reopen
    /// finds the previous image before the header write and the new one from
    /// it on; the next checkpoint of the same manager succeeds.
    ///
    /// The previous image (3 pages of data) ends after the pages the failing
    /// checkpoint's smaller image uses, so a trim before the failure would cut
    /// it off.
    #[cfg(feature = "testing-crash-injection")]
    #[test]
    fn a_failed_checkpoint_returns_an_error_and_the_next_one_succeeds() {
        use grafeo_common::testing::crash::with_failure_at;

        const POINTS: [&str; 4] = [
            "checkpoint:after_chunks",
            "checkpoint:after_data_sync",
            "checkpoint:after_header",
            "checkpoint:before_trim",
        ];
        let previous = vec![3u8; 9000];
        for (count, point) in (1..).zip(POINTS) {
            let dir = test_dir();
            let path = dir.path().join("db.grafeo");
            let manager = GrafeoFileManager::create(&path, None).unwrap();
            checkpoint(&manager, &[fixed(SectionType::Catalog, "Alix")], 1).unwrap();
            checkpoint(
                &manager,
                &[fixed(SectionType::Catalog, previous.clone())],
                2,
            )
            .unwrap();

            let error = with_failure_at(count, || {
                checkpoint(&manager, &[fixed(SectionType::Catalog, "Gus")], 3)
            })
            .unwrap_err()
            .to_string();
            assert!(error.contains(point), "{point}: {error}");
            assert_eq!(
                manager.active_header().epoch,
                2,
                "{point}: the manager serves the previous image"
            );
            match raw_section(&manager, SectionType::Catalog) {
                Ok(Some(bytes)) => assert!(
                    bytes == previous,
                    "{point}: the manager reads {} bytes, not the previous image's section",
                    bytes.len()
                ),
                other => panic!(
                    "{point}: the manager no longer reads the previous image's section: {other:?}"
                ),
            }

            let copy = dir.path().join("copy.grafeo");
            manager.copy_to(&copy).unwrap();
            let header_written = count >= 3;
            let found = read(
                &GrafeoFileManager::open(&copy, None).unwrap(),
                SectionType::Catalog,
            );
            let expected = if header_written {
                b"Gus".to_vec()
            } else {
                previous.clone()
            };
            assert!(
                found.as_ref() == Some(&expected),
                "{point}: a reopen finds the new image once its header is written, found {:?} bytes",
                found.as_ref().map(Vec::len)
            );

            checkpoint(&manager, &[fixed(SectionType::Catalog, "Vincent")], 4).unwrap();
            drop(manager);
            let reopened = GrafeoFileManager::open(&path, None).unwrap();
            assert_eq!(
                read(&reopened, SectionType::Catalog),
                Some(b"Vincent".to_vec()),
                "{point}: the checkpoint after the failed one"
            );
        }
    }

    /// `create_with_id` writes the given id into the file header, and both
    /// kinds of open hand that id to the cipher chooser.
    #[test]
    fn create_with_id_keeps_the_id_and_opens_pass_it_to_the_cipher_chooser() {
        let dir = test_dir();
        let path = dir.path().join("gus.grafeo");
        let id = 0x0003_0019_0088_u128;
        let manager = GrafeoFileManager::create_with_id(&path, id, None).unwrap();
        assert_eq!(manager.database_id(), id);
        drop(manager);
        assert_eq!(
            FileHeaderV3::decode(&fs::read(&path).unwrap()[..PAGE_BYTES])
                .unwrap()
                .database_id,
            id,
            "the id is in the file header"
        );

        for read_only in [false, true] {
            let mut seen = Vec::new();
            let chooser = |database_id| {
                seen.push(database_id);
                None
            };
            let manager = if read_only {
                GrafeoFileManager::open_read_only_with_cipher_for(&path, chooser)
            } else {
                GrafeoFileManager::open_with_cipher_for(&path, chooser)
            }
            .unwrap();
            assert_eq!(manager.database_id(), id);
            drop(manager);
            assert_eq!(
                seen,
                vec![id],
                "read-only {read_only}: the chooser runs once, with the file's id"
            );
        }
    }

    /// A 0.5.x file is refused before a cipher is chosen for it.
    #[test]
    fn a_0_5_file_is_refused_before_a_cipher_is_chosen() {
        let dir = test_dir();
        let path = dir.path().join("old.grafeo");
        let mut old = vec![0u8; 3 * PAGE_BYTES];
        old[..4].copy_from_slice(&MAGIC);
        old[4] = 1;
        fs::write(&path, &old).unwrap();

        let mut chosen = false;
        let error = GrafeoFileManager::open_with_cipher_for(&path, |_| {
            chosen = true;
            None
        })
        .map(|_| ())
        .unwrap_err()
        .to_string();
        assert!(error.contains("must be migrated"), "{error}");
        assert!(!chosen, "no cipher is chosen for a 0.5.x file");
    }

    #[cfg(feature = "encryption")]
    fn key(byte: u8) -> Option<ChunkCipher> {
        Some(grafeo_common::encryption::PageEncryptor::new(&[byte; 32]))
    }

    /// The cipher derived from the database id reads the file it was created
    /// with; one derived from another id, no cipher, or a cipher for a plain
    /// file are refused as `open` refuses them.
    #[cfg(feature = "encryption")]
    #[test]
    fn the_cipher_chosen_for_the_database_id_reads_its_file() {
        let chain = grafeo_common::encryption::KeyChain::new([3; 32]);
        let derive = |id: u128| chain.encryptor_for("grafeo-container", &id.to_le_bytes());
        let dir = test_dir();
        let path = dir.path().join("amsterdam.grafeo");
        let id = 88_u128;
        let manager = GrafeoFileManager::create_with_id(&path, id, Some(derive(id))).unwrap();
        assert!(manager.file_header().encrypted);
        checkpoint(&manager, &[fixed(SectionType::Catalog, "Mia")], 1).unwrap();
        drop(manager);

        for read_only in [false, true] {
            let open = |chooser: &dyn Fn(u128) -> Option<ChunkCipher>| {
                if read_only {
                    GrafeoFileManager::open_read_only_with_cipher_for(&path, chooser)
                } else {
                    GrafeoFileManager::open_with_cipher_for(&path, chooser)
                }
            };
            let manager = open(&|id| Some(derive(id))).unwrap();
            assert_eq!(read(&manager, SectionType::Catalog), Some(b"Mia".to_vec()));
            drop(manager);
            let error =
                |result: Result<GrafeoFileManager>| result.map(|_| ()).unwrap_err().to_string();
            let wrong = error(open(&|id| Some(derive(id + 1))));
            assert!(
                wrong.contains("wrong key"),
                "read-only {read_only}: {wrong}"
            );
            let missing = error(open(&|_| None));
            assert!(
                missing.contains("the database is encrypted and needs its key"),
                "read-only {read_only}: {missing}"
            );
        }

        let plain = dir.path().join("plain.grafeo");
        drop(GrafeoFileManager::create_with_id(&plain, id, None).unwrap());
        let error = GrafeoFileManager::open_with_cipher_for(&plain, |id| Some(derive(id)))
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(error.contains("the database is not encrypted"), "{error}");
    }

    #[cfg(feature = "encryption")]
    #[test]
    fn an_encrypted_database_round_trips() {
        let dir = test_dir();
        let path = dir.path().join("amsterdam.grafeo");
        let manager = GrafeoFileManager::create(&path, key(3)).unwrap();
        assert!(manager.file_header().encrypted);
        checkpoint(
            &manager,
            &[fixed(SectionType::Catalog, "Alix lives in Amsterdam")],
            1,
        )
        .unwrap();
        drop(manager);

        let bytes = fs::read(&path).unwrap();
        assert!(
            !bytes.windows(9).any(|window| window == b"Amsterdam"),
            "the plaintext is not in the file"
        );
        let manager = GrafeoFileManager::open(&path, key(3)).unwrap();
        assert_eq!(
            read(&manager, SectionType::Catalog),
            Some(b"Alix lives in Amsterdam".to_vec())
        );
        drop(manager);
        let reader = GrafeoFileManager::open_read_only(&path, key(3)).unwrap();
        assert_eq!(
            read(&reader, SectionType::Catalog),
            Some(b"Alix lives in Amsterdam".to_vec())
        );
    }

    #[cfg(feature = "encryption")]
    #[test]
    fn a_missing_or_wrong_key_and_a_key_for_a_plain_database_are_refused() {
        let dir = test_dir();
        let checkpointed = dir.path().join("berlin.grafeo");
        let manager = GrafeoFileManager::create(&checkpointed, key(3)).unwrap();
        checkpoint(&manager, &[fixed(SectionType::Catalog, "Gus")], 1).unwrap();
        drop(manager);
        let never_checkpointed = dir.path().join("paris.grafeo");
        drop(GrafeoFileManager::create(&never_checkpointed, key(3)).unwrap());
        let plain = dir.path().join("prague.grafeo");
        drop(GrafeoFileManager::create(&plain, None).unwrap());

        let cases: [(&Path, Option<u8>, &str); 5] = [
            (
                &checkpointed,
                None,
                "the database is encrypted and needs its key",
            ),
            (&checkpointed, Some(19), "wrong key"),
            (
                &never_checkpointed,
                None,
                "the database is encrypted and needs its key",
            ),
            (&never_checkpointed, Some(19), "wrong key"),
            (&plain, Some(3), "the database is not encrypted"),
        ];
        for (path, key_byte, expected) in cases {
            let before = fs::read(path).unwrap();
            for read_only in [false, true] {
                let cipher = key_byte.and_then(key);
                let result = if read_only {
                    GrafeoFileManager::open_read_only(path, cipher)
                } else {
                    GrafeoFileManager::open(path, cipher)
                };
                let error = result.map(|_| ()).unwrap_err().to_string();
                assert!(
                    error.contains(expected),
                    "{} with key {key_byte:?}, read-only {read_only}: {error}",
                    path.display()
                );
            }
            assert_eq!(
                fs::read(path).unwrap(),
                before,
                "{} is untouched",
                path.display()
            );
        }
    }

    #[cfg(feature = "testing-crash-injection")]
    #[test]
    fn a_crash_during_create_leaves_no_database_and_a_later_create_succeeds() {
        use grafeo_common::testing::crash::{CrashResult, with_crash_at};

        let dir = test_dir();
        let path = dir.path().join("gus.grafeo");
        let creating = creating_path(&path);
        let target = path.clone();
        let result = with_crash_at(1, move || {
            GrafeoFileManager::create(&target, None).map(|_| ())
        });
        assert!(
            matches!(result, CrashResult::Crashed),
            "create reaches create:after_write"
        );
        assert!(
            !path.exists(),
            "the database path never holds a partial file"
        );
        assert!(
            creating.exists(),
            "the partial file is left under its own name"
        );

        let manager = GrafeoFileManager::create(&path, None).unwrap();
        assert!(!creating.exists(), "the leftover is gone");
        assert_eq!(manager.active_header().iteration, 0);
        drop(manager);
        assert_eq!(
            GrafeoFileManager::open(&path, None)
                .unwrap()
                .active_header()
                .iteration,
            0
        );
    }

    #[test]
    fn create_replaces_a_leftover_partial_file() {
        let dir = test_dir();
        let path = dir.path().join("jules.grafeo");
        let creating = creating_path(&path);
        fs::write(&creating, vec![3u8; 3 * 4096 + 19]).unwrap();
        let manager = GrafeoFileManager::create(&path, None).unwrap();
        assert!(!creating.exists());
        assert_eq!(
            manager.file_size().unwrap(),
            4 * 4096,
            "nothing of the leftover remains"
        );
        drop(manager);
        let reopened = GrafeoFileManager::open(&path, None).unwrap();
        assert_eq!(read(&reopened, SectionType::Catalog), None);
    }

    #[test]
    fn create_leaves_a_create_in_progress_elsewhere_alone() {
        let dir = test_dir();
        let path = dir.path().join("vincent.grafeo");
        let creating = creating_path(&path);
        fs::write(&creating, b"Vincent is creating this").unwrap();
        let holder = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&creating)
            .unwrap();
        holder.try_lock_exclusive().unwrap();
        let error = GrafeoFileManager::create(&path, None)
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(error.contains("locked"), "{error}");
        drop(holder);
        assert!(!path.exists());
        assert_eq!(fs::read(&creating).unwrap(), b"Vincent is creating this");
    }

    #[test]
    fn create_fails_if_the_file_exists() {
        let dir = test_dir();
        let path = dir.path().join("test.grafeo");
        drop(GrafeoFileManager::create(&path, None).unwrap());
        assert!(GrafeoFileManager::create(&path, None).is_err());
    }

    /// A create refuses any entry at the path, also one `Path::exists` does
    /// not see: a dangling symbolic link is left as it is, and nothing is
    /// created at its target.
    #[test]
    fn create_refuses_a_dangling_symlink_at_the_path() {
        let dir = test_dir();
        let path = dir.path().join("mia.grafeo");
        let target = dir.path().join("nowhere.grafeo");
        #[cfg(unix)]
        let linked = std::os::unix::fs::symlink(&target, &path);
        #[cfg(windows)]
        let linked = std::os::windows::fs::symlink_file(&target, &path);
        #[cfg(not(any(unix, windows)))]
        let linked: std::io::Result<()> = Err(std::io::Error::other("no symbolic links"));
        if let Err(error) = linked {
            // Windows creates symbolic links only with the privilege to (or in
            // developer mode).
            eprintln!("skipped: this platform or user cannot create a symbolic link: {error}");
            return;
        }
        assert!(!path.exists(), "Path::exists does not see a dangling link");

        let error = GrafeoFileManager::create(&path, None)
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(error.contains("already exists"), "{error}");
        assert!(
            fs::symlink_metadata(&path)
                .unwrap()
                .file_type()
                .is_symlink(),
            "the link is left as it was"
        );
        assert!(!target.exists(), "nothing is created at the link's target");
        assert!(
            !creating_path(&path).exists(),
            "the refused create leaves no partial file"
        );
    }

    #[test]
    fn open_fails_if_the_file_does_not_exist() {
        let dir = test_dir();
        let path = dir.path().join("beatrix_missing.grafeo");
        assert!(GrafeoFileManager::open(&path, None).is_err());
        assert!(GrafeoFileManager::open_read_only(&path, None).is_err());
    }

    #[test]
    fn the_exclusive_lock_prevents_a_second_read_write_open() {
        let dir = test_dir();
        let path = dir.path().join("locked.grafeo");
        let _manager = GrafeoFileManager::create(&path, None).unwrap();
        let error = GrafeoFileManager::open(&path, None)
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(error.contains("locked"), "{error}");
    }

    #[test]
    fn read_only_opens_coexist() {
        let dir = test_dir();
        let path = dir.path().join("coexist.grafeo");
        {
            let manager = GrafeoFileManager::create(&path, None).unwrap();
            checkpoint(&manager, &[fixed(SectionType::Catalog, "Mia")], 1).unwrap();
            manager.close().unwrap();
        }
        let first = GrafeoFileManager::open_read_only(&path, None).unwrap();
        let second = GrafeoFileManager::open_read_only(&path, None).unwrap();
        assert!(first.is_read_only() && second.is_read_only());
        assert_eq!(read(&first, SectionType::Catalog), Some(b"Mia".to_vec()));
        assert_eq!(read(&second, SectionType::Catalog), Some(b"Mia".to_vec()));
        assert_eq!(first.active_header().iteration, 1);
    }

    #[test]
    fn a_read_only_manager_refuses_a_checkpoint() {
        let dir = test_dir();
        let path = dir.path().join("ro_write.grafeo");
        GrafeoFileManager::create(&path, None)
            .unwrap()
            .close()
            .unwrap();
        let reader = GrafeoFileManager::open_read_only(&path, None).unwrap();
        let before = fs::read(&path).unwrap();
        let error = checkpoint(&reader, &[fixed(SectionType::Catalog, "nope")], 1)
            .unwrap_err()
            .to_string();
        assert!(error.contains("read-only"), "{error}");
        assert_eq!(fs::read(&path).unwrap(), before);
    }

    #[test]
    fn lock_released_after_close() {
        let dir = test_dir();
        let path = dir.path().join("lockclose.grafeo");
        let manager = GrafeoFileManager::create(&path, None).unwrap();
        checkpoint(&manager, &[fixed(SectionType::Catalog, "data")], 1).unwrap();
        manager.close().unwrap();
        let reopened = GrafeoFileManager::open(&path, None).unwrap();
        assert_eq!(
            read(&reopened, SectionType::Catalog),
            Some(b"data".to_vec())
        );
    }

    #[test]
    fn lock_released_on_drop() {
        let dir = test_dir();
        let path = dir.path().join("lockdrop.grafeo");
        drop(GrafeoFileManager::create(&path, None).unwrap());
        let _reopened = GrafeoFileManager::open(&path, None).unwrap();
    }

    #[test]
    fn sidecar_wal_path_computation() {
        let dir = test_dir();
        let path = dir.path().join("mydb.grafeo");
        let manager = GrafeoFileManager::create(&path, None).unwrap();
        assert_eq!(
            manager
                .sidecar_wal_path()
                .file_name()
                .unwrap()
                .to_str()
                .unwrap(),
            "mydb.grafeo.wal"
        );
        assert!(!manager.has_sidecar_wal());
    }

    #[test]
    fn sidecar_wal_detect_and_remove() {
        let dir = test_dir();
        let path = dir.path().join("test.grafeo");
        let manager = GrafeoFileManager::create(&path, None).unwrap();
        assert!(!manager.has_sidecar_wal());
        fs::create_dir_all(manager.sidecar_wal_path()).unwrap();
        assert!(manager.has_sidecar_wal());
        manager.remove_sidecar_wal().unwrap();
        assert!(!manager.has_sidecar_wal());
        manager.remove_sidecar_wal().unwrap();
        assert!(
            !manager.has_sidecar_wal(),
            "removing an absent sidecar is a no-op"
        );
    }

    #[test]
    fn file_size_grows_with_data() {
        let dir = test_dir();
        let manager = GrafeoFileManager::create(dir.path().join("test.grafeo"), None).unwrap();
        let empty_size = manager.file_size().unwrap();
        checkpoint(
            &manager,
            &[fixed(SectionType::LpgStore, vec![0xAB; 100_000])],
            1,
        )
        .unwrap();
        let full_size = manager.file_size().unwrap();
        assert!(
            full_size >= empty_size + 100_000,
            "{empty_size} then {full_size}"
        );
        assert!(
            full_size.is_multiple_of(PAGE_SIZE),
            "the file ends on a page"
        );
    }

    #[test]
    fn path_returns_the_database_file_path() {
        let dir = test_dir();
        let path = dir.path().join("alix.grafeo");
        let manager = GrafeoFileManager::create(&path, None).unwrap();
        assert_eq!(manager.path(), path);
    }

    #[test]
    fn sync_succeeds_for_writable_and_read_only_managers() {
        let dir = test_dir();
        let path = dir.path().join("vincent.grafeo");
        {
            let manager = GrafeoFileManager::create(&path, None).unwrap();
            checkpoint(&manager, &[fixed(SectionType::Catalog, "sync")], 1).unwrap();
            manager.sync().unwrap();
            manager.close().unwrap();
        }
        let reader = GrafeoFileManager::open_read_only(&path, None).unwrap();
        reader.sync().unwrap();
        reader.close().unwrap();
    }

    #[test]
    fn copy_to_produces_an_identical_database() {
        let dir = test_dir();
        let src = dir.path().join("copy_src.grafeo");
        let dest = dir.path().join("copy_dest.grafeo");
        let manager = GrafeoFileManager::create(&src, None).unwrap();
        checkpoint(
            &manager,
            &[fixed(SectionType::Catalog, "copy test payload")],
            5,
        )
        .unwrap();
        let bytes = manager.copy_to(&dest).unwrap();
        assert_eq!(bytes, manager.file_size().unwrap());
        assert_eq!(
            read(&manager, SectionType::Catalog),
            Some(b"copy test payload".to_vec()),
            "the original is still usable"
        );
        let database_id = manager.database_id();
        manager.close().unwrap();

        let copy = GrafeoFileManager::open(&dest, None).unwrap();
        assert_eq!(
            read(&copy, SectionType::Catalog),
            Some(b"copy test payload".to_vec())
        );
        assert_eq!(copy.active_header().epoch, 5);
        assert_eq!(copy.database_id(), database_id);
    }

    #[test]
    fn copy_to_from_a_read_only_manager() {
        let dir = test_dir();
        let src = dir.path().join("ro_copy_src.grafeo");
        let dest = dir.path().join("ro_copy_dest.grafeo");
        {
            let manager = GrafeoFileManager::create(&src, None).unwrap();
            checkpoint(
                &manager,
                &[fixed(SectionType::Catalog, "read-only copy")],
                7,
            )
            .unwrap();
            manager.close().unwrap();
        }
        let reader = GrafeoFileManager::open_read_only(&src, None).unwrap();
        assert!(reader.copy_to(&dest).unwrap() > 0);
        let copy = GrafeoFileManager::open(&dest, None).unwrap();
        assert_eq!(
            read(&copy, SectionType::Catalog),
            Some(b"read-only copy".to_vec())
        );
    }
}
