//! Incremental backup and point-in-time recovery.
//!
//! Provides `backup_full()`, `backup_incremental()`, and `restore_to_epoch()`
//! APIs on [`GrafeoDB`](super::GrafeoDB). Full backups capture the entire
//! database state; incremental backups export only the WAL records since the
//! last backup. Recovery replays a chain of full + incremental backups to
//! restore the database to any committed epoch.
//!
//! # Backup chain model
//!
//! ```text
//! [Full Snapshot] -> [Incr 1] -> [Incr 2] -> ... -> [Incr N]
//!   epoch 0-100      101-200     201-300              901-1000
//! ```
//!
//! To restore to epoch 750: load full snapshot (epoch 100), replay
//! incrementals 1-7, stop at epoch 750.

use std::path::Path;

use grafeo_common::types::EpochId;
use grafeo_common::utils::error::{Error, Result};
use serde::{Deserialize, Serialize};

// ── Backup types ───────────────────────────────────────────────────

/// The type of a backup segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum BackupKind {
    /// A full snapshot of the entire database.
    Full,
    /// WAL records since the last backup checkpoint.
    Incremental,
}

/// Metadata for a single backup segment (full or incremental).
///
/// Read, not built, outside this crate: later releases may add fields. It is
/// stored with bincode, which has no field names, so a field added later
/// needs a reader for the layout without it (`#[serde(default)]` alone does
/// not read an older file).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct BackupSegment {
    /// Segment type.
    pub kind: BackupKind,
    /// File name (relative to backup directory).
    pub filename: String,
    /// Start epoch (inclusive).
    pub start_epoch: EpochId,
    /// End epoch (inclusive).
    pub end_epoch: EpochId,
    /// CRC-32 checksum of the segment file.
    pub checksum: u32,
    /// Size in bytes.
    pub size_bytes: u64,
    /// Timestamp when this backup was created (ms since UNIX epoch).
    pub created_at_ms: u64,
}

/// Tracks the full backup chain for a database.
///
/// Read, not built, outside this crate: later releases may add fields. It is
/// stored with bincode, which has no field names, so a field added later
/// needs a reader for the layout without it (`#[serde(default)]` alone does
/// not read an older file).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct BackupManifest {
    /// Manifest format version.
    pub version: u32,
    /// Ordered list of backup segments (full first, then incrementals).
    pub segments: Vec<BackupSegment>,
}

impl BackupManifest {
    /// Creates a new empty manifest.
    #[must_use]
    pub fn new() -> Self {
        Self {
            version: 1,
            segments: Vec::new(),
        }
    }

    /// Returns the most recent full backup segment, if any.
    #[must_use]
    pub fn latest_full(&self) -> Option<&BackupSegment> {
        self.segments
            .iter()
            .rev()
            .find(|s| s.kind == BackupKind::Full)
    }

    /// Returns incremental segments after the given epoch, in order.
    pub fn incrementals_after(&self, epoch: EpochId) -> Vec<&BackupSegment> {
        self.segments
            .iter()
            .filter(|s| s.kind == BackupKind::Incremental && s.start_epoch > epoch)
            .collect()
    }

    /// Returns the epoch range covered by this manifest.
    #[must_use]
    pub fn epoch_range(&self) -> Option<(EpochId, EpochId)> {
        let first = self.segments.first()?;
        let last = self.segments.last()?;
        Some((first.start_epoch, last.end_epoch))
    }
}

impl Default for BackupManifest {
    fn default() -> Self {
        Self::new()
    }
}

/// Tracks the WAL position of the last completed backup.
///
/// Persisted as `backup_cursor.meta` in the WAL directory.
///
/// Read, not built, outside this crate: later releases may add fields. It is
/// stored with bincode, which has no field names, so a field added later
/// needs a reader for the layout without it (`#[serde(default)]` alone does
/// not read an older file).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct BackupCursor {
    /// The epoch up to which WAL records have been backed up.
    pub backed_up_epoch: EpochId,
    /// The WAL log sequence number at the time of the last backup.
    pub log_sequence: u64,
    /// Timestamp of the last backup.
    pub timestamp_ms: u64,
}

// ── Manifest I/O ───────────────────────────────────────────────────

const MANIFEST_FILENAME: &str = "backup_manifest.json";
const BACKUP_CURSOR_FILENAME: &str = "backup_cursor.meta";

/// Reads the backup manifest from a backup directory.
///
/// Returns `None` if no manifest exists.
///
/// # Errors
///
/// Returns an error if the manifest file exists but cannot be read or parsed.
pub fn read_manifest(backup_dir: &Path) -> Result<Option<BackupManifest>> {
    let path = backup_dir.join(MANIFEST_FILENAME);
    if !path.exists() {
        return Ok(None);
    }
    let data = std::fs::read(&path)
        .map_err(|e| Error::Internal(format!("failed to read backup manifest: {e}")))?;
    let (manifest, _): (BackupManifest, _) =
        bincode::serde::decode_from_slice(&data, bincode::config::standard())
            .map_err(|e| Error::Internal(format!("failed to parse backup manifest: {e}")))?;
    Ok(Some(manifest))
}

/// Writes the backup manifest to a backup directory.
///
/// Uses write-to-temp-then-rename for atomicity.
///
/// # Errors
///
/// Returns an error if the manifest cannot be written.
pub fn write_manifest(backup_dir: &Path, manifest: &BackupManifest) -> Result<()> {
    std::fs::create_dir_all(backup_dir)
        .map_err(|e| Error::Internal(format!("failed to create backup directory: {e}")))?;

    let path = backup_dir.join(MANIFEST_FILENAME);
    let temp_path = backup_dir.join(format!("{MANIFEST_FILENAME}.tmp"));

    let data = bincode::serde::encode_to_vec(manifest, bincode::config::standard())
        .map_err(|e| Error::Internal(format!("failed to serialize backup manifest: {e}")))?;

    std::fs::write(&temp_path, data)
        .map_err(|e| Error::Internal(format!("failed to write backup manifest: {e}")))?;
    std::fs::rename(&temp_path, &path)
        .map_err(|e| Error::Internal(format!("failed to finalize backup manifest: {e}")))?;

    Ok(())
}

// ── Backup cursor I/O ──────────────────────────────────────────────

/// Reads the backup cursor from a WAL directory.
///
/// Returns `None` if no cursor exists (no backup has been taken).
///
/// # Errors
///
/// Returns an error if the cursor file exists but cannot be read.
pub fn read_backup_cursor(wal_dir: &Path) -> Result<Option<BackupCursor>> {
    let path = wal_dir.join(BACKUP_CURSOR_FILENAME);
    if !path.exists() {
        return Ok(None);
    }
    let data = std::fs::read(&path)
        .map_err(|e| Error::Internal(format!("failed to read backup cursor: {e}")))?;
    let cursor: BackupCursor =
        bincode::serde::decode_from_slice(&data, bincode::config::standard())
            .map(|(c, _)| c)
            .map_err(|e| Error::Internal(format!("failed to parse backup cursor: {e}")))?;
    Ok(Some(cursor))
}

/// Writes the backup cursor to a WAL directory.
///
/// Uses write-to-temp-then-rename for atomicity.
///
/// # Errors
///
/// Returns an error if the cursor cannot be written.
pub fn write_backup_cursor(wal_dir: &Path, cursor: &BackupCursor) -> Result<()> {
    let path = wal_dir.join(BACKUP_CURSOR_FILENAME);
    let temp_path = wal_dir.join(format!("{BACKUP_CURSOR_FILENAME}.tmp"));

    let data = bincode::serde::encode_to_vec(cursor, bincode::config::standard())
        .map_err(|e| Error::Internal(format!("failed to serialize backup cursor: {e}")))?;

    std::fs::write(&temp_path, &data)
        .map_err(|e| Error::Internal(format!("failed to write backup cursor: {e}")))?;
    std::fs::rename(&temp_path, &path)
        .map_err(|e| Error::Internal(format!("failed to finalize backup cursor: {e}")))?;

    Ok(())
}

// ── Incremental backup file format ─────────────────────────────────

/// Magic bytes for incremental backup files.
pub const BACKUP_MAGIC: [u8; 4] = *b"GBAK";
/// Current backup file version.
pub const BACKUP_VERSION: u32 = 1;

/// Header for an incremental backup file.
///
/// ```text
/// [magic: 4 bytes "GBAK"]
/// [version: u32 LE]
/// [start_epoch: u64 LE]
/// [end_epoch: u64 LE]
/// [record_count: u64 LE]
/// ... WAL frames ...
/// ```
pub const BACKUP_HEADER_SIZE: usize = 32;

/// Writes the incremental backup file header.
pub fn write_backup_header(
    buf: &mut Vec<u8>,
    start_epoch: EpochId,
    end_epoch: EpochId,
    record_count: u64,
) {
    buf.extend_from_slice(&BACKUP_MAGIC);
    buf.extend_from_slice(&BACKUP_VERSION.to_le_bytes());
    buf.extend_from_slice(&start_epoch.as_u64().to_le_bytes());
    buf.extend_from_slice(&end_epoch.as_u64().to_le_bytes());
    buf.extend_from_slice(&record_count.to_le_bytes());
}

/// Reads and validates the incremental backup file header.
///
/// Returns `(start_epoch, end_epoch, record_count)` on success.
///
/// # Errors
///
/// Returns an error if the header is invalid.
///
/// # Panics
///
/// Cannot panic: all slice indexing is bounds-checked by the length guard.
pub fn read_backup_header(data: &[u8]) -> Result<(EpochId, EpochId, u64)> {
    if data.len() < BACKUP_HEADER_SIZE {
        return Err(Error::Internal(
            "incremental backup file too short".to_string(),
        ));
    }
    if data[0..4] != BACKUP_MAGIC {
        return Err(Error::Internal(
            "invalid backup file magic bytes".to_string(),
        ));
    }
    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    if version > BACKUP_VERSION {
        return Err(Error::Internal(format!(
            "unsupported backup version {version}, max supported is {BACKUP_VERSION}"
        )));
    }
    let start_epoch = EpochId::new(u64::from_le_bytes(data[8..16].try_into().unwrap()));
    let end_epoch = EpochId::new(u64::from_le_bytes(data[16..24].try_into().unwrap()));
    let record_count = u64::from_le_bytes(data[24..32].try_into().unwrap());
    Ok((start_epoch, end_epoch, record_count))
}

/// Returns the timestamp in milliseconds since UNIX epoch.
// reason: millis since UNIX epoch fits u64 for centuries
#[allow(clippy::cast_possible_truncation)]
pub(super) fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

/// The sequence number of a WAL log file (`wal_<sequence>.log`), 0 when the
/// name has another form.
fn wal_file_sequence(path: &Path) -> u64 {
    path.file_stem()
        .and_then(|s| s.to_str())
        .and_then(|s| s.strip_prefix("wal_"))
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0)
}

// ── Backup operations (called from GrafeoDB) ───────────────────────

use std::io::Read;

use grafeo_storage::file::GrafeoFileManager;
use grafeo_storage::file::detect::{OnDisk, detect};
use grafeo_storage::file::v3::header::{DbHeaderV3, FileHeaderV3, active_header};
use grafeo_storage::wal::{LpgWal, WalCipher};

use super::encryption::DatabaseKeys;

/// Size of a page (and of each header) of a `.grafeo` file, in bytes.
const PAGE_BYTES: usize = 4096;

/// Creates a full backup by copying the .grafeo container file.
///
/// 1. Copies the container file to the backup directory via the locked handle.
/// 2. Updates the manifest and backup cursor.
///
/// Uses [`GrafeoFileManager::copy_to`] instead of `std::fs::copy()` so the
/// copy reads through the already-locked file handle. `std::fs::copy()` opens
/// a new handle, which fails on Windows when an exclusive lock is held.
///
/// The backup covers exactly what the copied file holds: its epoch, and the
/// WAL files before the WAL's checkpoint marker. The files from the marker on
/// are left to the next incremental backup.
///
/// # Errors
///
/// Returns an error if the database has no file manager, or if I/O fails.
pub(super) fn do_backup_full(
    backup_dir: &Path,
    fm: &GrafeoFileManager,
    wal: Option<&LpgWal>,
) -> Result<BackupSegment> {
    std::fs::create_dir_all(backup_dir)
        .map_err(|e| Error::Internal(format!("failed to create backup directory: {e}")))?;

    // Determine backup filename
    let mut manifest = read_manifest(backup_dir)?.unwrap_or_default();
    let segment_idx = manifest.segments.len();
    let filename = format!("backup_full_{segment_idx:04}.grafeo");
    let dest_path = backup_dir.join(&filename);

    // No checkpoint may change the file or the WAL's checkpoint marker while
    // they are read, so the two agree on what the copy contains.
    let _checkpoint = fm.checkpoint_guard();

    // Copy the .grafeo file to the backup directory through the locked handle
    fm.copy_to(&dest_path)?;

    let file_size = std::fs::metadata(&dest_path).map_or(0, |m| m.len());
    let file_data = std::fs::read(&dest_path)
        .map_err(|e| Error::Internal(format!("failed to read backup file for checksum: {e}")))?;
    let checksum = crc32fast::hash(&file_data);
    let copied_epoch = active_epoch(&file_data, &dest_path)?;

    let segment = BackupSegment {
        kind: BackupKind::Full,
        filename,
        start_epoch: EpochId::new(0),
        end_epoch: copied_epoch,
        checksum,
        size_bytes: file_size,
        created_at_ms: now_ms(),
    };

    manifest.segments.push(segment.clone());
    write_manifest(backup_dir, &manifest)?;

    // The checkpoint that wrote the copied file started a new WAL file (the
    // marker's sequence), so every record before it is in the copy. The next
    // incremental backup starts at that file: taking the file active at this
    // point instead left the records written since the checkpoint in neither.
    if let Some(wal) = wal {
        let covered = wal
            .read_checkpoint_metadata()?
            .ok_or_else(|| {
                Error::Internal("full backup: the WAL has no checkpoint marker".to_string())
            })?
            .log_sequence;
        let cursor = BackupCursor {
            backed_up_epoch: copied_epoch,
            log_sequence: covered.saturating_sub(1),
            timestamp_ms: now_ms(),
        };
        write_backup_cursor(wal.dir(), &cursor)?;
    }

    Ok(segment)
}

/// The epoch of the image a copied `.grafeo` file opens at, from its own
/// active database header (headers are never encrypted).
///
/// The manager's in-memory header can be older: after a checkpoint that
/// failed once its header was written, the manager keeps serving the
/// previous image while the file opens at the new one.
fn active_epoch(file_data: &[u8], path: &Path) -> Result<EpochId> {
    let page = |index: usize| {
        let start = index * PAGE_BYTES;
        file_data.get(start..start + PAGE_BYTES).unwrap_or(&[])
    };
    let (_, header) = active_header([DbHeaderV3::decode(page(1)), DbHeaderV3::decode(page(2))])
        .and_then(|active| {
            active.ok_or_else(|| {
                Error::Serialization("both database header slots are empty".to_string())
            })
        })
        .map_err(|e| {
            Error::Internal(format!(
                "full backup {}: cannot read the copied file's database header: {e}",
                path.display()
            ))
        })?;
    Ok(EpochId::new(header.epoch))
}

/// Creates an incremental backup containing WAL records since the last backup.
///
/// Reads WAL log files from the backup cursor's position forward, copies
/// the raw frames into a backup segment file.
///
/// # Errors
///
/// Returns an error if no full backup exists, or if the WAL files have been
/// truncated past the cursor.
pub(super) fn do_backup_incremental(
    backup_dir: &Path,
    wal: &LpgWal,
    current_epoch: EpochId,
) -> Result<BackupSegment> {
    let manifest = read_manifest(backup_dir)?.ok_or_else(|| {
        Error::Internal("no backup manifest found; run a full backup first".to_string())
    })?;

    if manifest.latest_full().is_none() {
        return Err(Error::Internal(
            "no full backup in manifest; run a full backup first".to_string(),
        ));
    }

    let cursor = read_backup_cursor(wal.dir())?.ok_or_else(|| {
        Error::Internal("no backup cursor found; run a full backup first".to_string())
    })?;

    // Nothing logged since the last backup: fail before sealing, or every
    // such call (a poll for new data) would leave another empty log file.
    let has_new_records = wal.log_files()?.iter().any(|path| {
        wal_file_sequence(path) > cursor.log_sequence
            && std::fs::metadata(path).is_ok_and(|meta| meta.len() > 0)
    });
    if !has_new_records {
        return Err(Error::Internal(
            "no new WAL records since last backup".to_string(),
        ));
    }

    // Seal the active file first: every record logged so far is then in a
    // file up to `sealed`, and records logged from now on go to newer files,
    // which the next incremental backup reads. Reading the active file and
    // rotating afterwards lost the records appended in between.
    let sealed = wal.current_sequence();
    wal.rotate().map_err(|e| {
        Error::Internal(format!("failed to rotate WAL for incremental backup: {e}"))
    })?;

    let log_files = wal.log_files()?;
    if log_files.is_empty() {
        return Err(Error::Internal("no WAL log files to backup".to_string()));
    }

    // Read WAL files from cursor position onward
    let mut wal_data = Vec::new();
    let mut record_count = 0u64;
    // cursor.backed_up_epoch used for start_epoch calculation below

    for file_path in &log_files {
        let seq = wal_file_sequence(file_path);

        // Files up to the cursor are in earlier backups; files after
        // `sealed` were started after the rotation above and belong to the
        // next backup.
        if seq <= cursor.log_sequence || seq > sealed {
            continue;
        }

        let file_bytes = std::fs::read(file_path).map_err(|e| {
            Error::Internal(format!(
                "failed to read WAL file {}: {e}",
                file_path.display()
            ))
        })?;

        if !file_bytes.is_empty() {
            wal_data.extend_from_slice(&file_bytes);
            // Count records by scanning for frame markers (rough count)
            // Exact count would require parsing, but we record approximate
            record_count += 1; // Per-file approximation
        }
    }

    if wal_data.is_empty() {
        return Err(Error::Internal(
            "no new WAL records since last backup".to_string(),
        ));
    }

    let start_epoch = EpochId::new(cursor.backed_up_epoch.as_u64() + 1);
    let end_epoch = current_epoch;

    // Write incremental backup file
    let segment_idx = manifest.segments.len();
    let filename = format!("backup_incr_{segment_idx:04}.wal");
    let dest_path = backup_dir.join(&filename);

    let mut output = Vec::new();
    write_backup_header(&mut output, start_epoch, end_epoch, record_count);
    output.extend_from_slice(&wal_data);

    std::fs::write(&dest_path, &output)
        .map_err(|e| Error::Internal(format!("failed to write incremental backup: {e}")))?;

    let checksum = crc32fast::hash(&output);
    let segment = BackupSegment {
        kind: BackupKind::Incremental,
        filename,
        start_epoch,
        end_epoch,
        checksum,
        size_bytes: output.len() as u64,
        created_at_ms: now_ms(),
    };

    // Update manifest
    let mut manifest = manifest;
    manifest.segments.push(segment.clone());
    write_manifest(backup_dir, &manifest)?;

    // Update backup cursor
    let new_cursor = BackupCursor {
        backed_up_epoch: current_epoch,
        log_sequence: sealed,
        timestamp_ms: now_ms(),
    };
    write_backup_cursor(wal.dir(), &new_cursor)?;

    Ok(segment)
}

// ── Restore ────────────────────────────────────────────────────────

/// Restores a database to a specific epoch from a backup chain.
///
/// 1. Finds the most recent full backup with `end_epoch <= target_epoch`
///    and the incremental segments after it, and checks `keys` against the
///    full backup ([`restore_ciphers`]).
/// 2. Copies the full backup to `output_path`.
/// 3. Replays incremental segments up to `target_epoch` using epoch-bounded
///    WAL recovery, and writes the records as the sidecar WAL of the
///    restored file. For an encrypted backup the segments are decrypted, and
///    that WAL encrypted, with the WAL key of the backup's database.
///
/// # Errors
///
/// Returns an error if the backup chain does not cover the target epoch,
/// if `keys` do not fit the backup, if segment checksums fail, or if I/O
/// fails.
pub(super) fn do_restore_to_epoch(
    backup_dir: &Path,
    target_epoch: EpochId,
    output_path: &Path,
    keys: &DatabaseKeys,
) -> Result<()> {
    // Restore must target a fresh path: the copy below overwrites
    // `output_path` and the sidecar handling deletes `<output_path>.wal/`,
    // which would destroy a live database and its unflushed WAL.
    let sidecar_dir = format!("{}.wal", output_path.display());
    if output_path.exists() || Path::new(&sidecar_dir).exists() {
        return Err(Error::InvalidValue(format!(
            "restore output path already exists: {} (restore must target a fresh path; \
             move or remove the existing database and its {sidecar_dir} sidecar first)",
            output_path.display(),
        )));
    }

    let manifest = read_manifest(backup_dir)?
        .ok_or_else(|| Error::Internal("no backup manifest found".to_string()))?;

    // Find the best full backup (latest one that doesn't exceed target)
    let full = manifest
        .segments
        .iter()
        .rfind(|s| s.kind == BackupKind::Full && s.end_epoch <= target_epoch)
        .ok_or_else(|| {
            Error::Internal(format!(
                "no full backup covers epoch {}",
                target_epoch.as_u64()
            ))
        })?;

    // Find incremental segments that cover (full.end_epoch, target_epoch]
    let incrementals: Vec<&BackupSegment> = manifest
        .segments
        .iter()
        .filter(|s| {
            s.kind == BackupKind::Incremental
                && s.start_epoch > full.end_epoch
                && s.start_epoch <= target_epoch
        })
        .collect();

    // The key is checked before anything is written.
    let full_path = backup_dir.join(&full.filename);
    let ciphers = restore_ciphers(&full_path, keys, !incrementals.is_empty())?;

    // Copy full backup to output path
    std::fs::copy(&full_path, output_path)
        .map_err(|e| Error::Internal(format!("failed to copy full backup to output: {e}")))?;

    if incrementals.is_empty() {
        // Full backup already covers the target epoch
        return Ok(());
    }

    // Create a temporary WAL directory for replay
    let wal_dir = output_path.parent().unwrap_or(Path::new(".")).join(format!(
        "{}.restore_wal",
        output_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("db")
    ));
    std::fs::create_dir_all(&wal_dir)
        .map_err(|e| Error::Internal(format!("failed to create restore WAL directory: {e}")))?;

    // Write incremental WAL data to temp WAL files for recovery
    for (i, incr) in incrementals.iter().enumerate() {
        let incr_path = backup_dir.join(&incr.filename);
        let incr_data = std::fs::read(&incr_path).map_err(|e| {
            Error::Internal(format!(
                "failed to read incremental backup {}: {e}",
                incr.filename
            ))
        })?;

        // Validate checksum
        let actual_crc = crc32fast::hash(&incr_data);
        if actual_crc != incr.checksum {
            return Err(Error::Internal(format!(
                "incremental backup {} CRC mismatch: expected {:08x}, got {actual_crc:08x}",
                incr.filename, incr.checksum,
            )));
        }

        // Skip the backup header, write the raw WAL frames to a temp log file
        if incr_data.len() > BACKUP_HEADER_SIZE {
            let wal_frames = &incr_data[BACKUP_HEADER_SIZE..];
            let wal_file = wal_dir.join(format!("wal_{i:08}.log"));
            std::fs::write(&wal_file, wal_frames).map_err(|e| {
                Error::Internal(format!("failed to write WAL file for restore: {e}"))
            })?;
        }
    }

    // Recover WAL records up to target epoch, then write a trimmed WAL
    // that contains only records within the epoch boundary. This ensures
    // that when GrafeoDB::open() replays the sidecar WAL, it does not
    // advance beyond the target epoch.
    let recovery = grafeo_storage::wal::WalRecovery::with_cipher(&wal_dir, ciphers.decrypt);
    let records = recovery.recover_until_epoch(target_epoch)?;

    // Write a single trimmed WAL file containing only the bounded records
    let trimmed_dir = wal_dir.parent().unwrap_or(Path::new(".")).join(format!(
        "{}.trimmed_wal",
        wal_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("wal")
    ));
    std::fs::create_dir_all(&trimmed_dir)
        .map_err(|e| Error::Internal(format!("failed to create trimmed WAL directory: {e}")))?;

    if !records.is_empty() {
        use grafeo_storage::wal::{LpgWal, WalConfig};
        let trimmed_wal =
            LpgWal::with_config_and_cipher(&trimmed_dir, WalConfig::default(), ciphers.encrypt)?;
        for record in &records {
            trimmed_wal.log(record)?;
        }
        trimmed_wal.flush()?;
        drop(trimmed_wal);
    }

    // Remove the original (untrimmed) restore WAL directory
    std::fs::remove_dir_all(&wal_dir)
        .map_err(|e| Error::Internal(format!("failed to remove restore WAL directory: {e}")))?;

    // Move the trimmed WAL to the sidecar location
    let sidecar_dir = format!("{}.wal", output_path.display());
    let sidecar_path = std::path::Path::new(&sidecar_dir);
    if sidecar_path.exists() {
        std::fs::remove_dir_all(sidecar_path)
            .map_err(|e| Error::Internal(format!("failed to remove existing sidecar WAL: {e}")))?;
    }
    std::fs::rename(&trimmed_dir, sidecar_path)
        .map_err(|e| Error::Internal(format!("failed to move WAL to sidecar location: {e}")))?;

    Ok(())
}

/// The WAL ciphers of a restore: one decrypts the segments, the other
/// encrypts the WAL written next to the restored file (the same key, but a
/// cipher cannot be shared between the two).
struct RestoreCiphers {
    decrypt: Option<WalCipher>,
    encrypt: Option<WalCipher>,
}

/// Checks `keys` against the full backup at `path` before a restore writes
/// anything, and returns the WAL ciphers of the backup's database (`None`
/// for an unencrypted backup).
///
/// An encrypted backup restored without a key is refused when
/// `has_segments` (its segments cannot be read); without segments the
/// restore only copies the encrypted file. A key is checked by opening the
/// backup with it, which decrypts its directory, and is refused for a backup
/// that is not encrypted.
fn restore_ciphers(path: &Path, keys: &DatabaseKeys, has_segments: bool) -> Result<RestoreCiphers> {
    let plain = RestoreCiphers {
        decrypt: None,
        encrypt: None,
    };
    // A full backup taken by 0.5.x is a 0.5.x file, which is never encrypted.
    let encrypted = match detect(path)? {
        OnDisk::Current => {
            let mut page = Vec::with_capacity(4096);
            std::fs::File::open(path)
                .and_then(|file| file.take(4096).read_to_end(&mut page))
                .map_err(|e| {
                    Error::Internal(format!(
                        "cannot read the full backup {}: {e}",
                        path.display()
                    ))
                })?;
            let header = FileHeaderV3::decode(&page).map_err(|e| {
                Error::Serialization(format!("full backup {}: {e}", path.display()))
            })?;
            header.encrypted.then_some(header.database_id)
        }
        _ => None,
    };
    match (encrypted, keys.is_encrypted()) {
        (None, false) => Ok(plain),
        (None, true) => Err(Error::InvalidValue(format!(
            "the backup {} is not encrypted: restore it with `GrafeoDB::restore_to_epoch`, \
             without a key",
            path.display()
        ))),
        (Some(_), false) if has_segments => Err(Error::InvalidValue(format!(
            "the backup {} is of an encrypted database, and its incremental segments can only \
             be replayed with its key: restore it with `GrafeoDB::restore_to_epoch_with`",
            path.display()
        ))),
        (Some(_), false) => Ok(plain),
        (Some(database_id), true) => {
            GrafeoFileManager::open_read_only_with_cipher_for(path, |id| keys.container_cipher(id))
                .and_then(|backup| backup.close())
                .map_err(|e| {
                    Error::InvalidValue(format!(
                        "the key does not open the backup {}: {e}",
                        path.display()
                    ))
                })?;
            Ok(RestoreCiphers {
                decrypt: keys.wal_cipher(database_id),
                encrypt: keys.wal_cipher(database_id),
            })
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_manifest_new() {
        let manifest = BackupManifest::new();
        assert_eq!(manifest.version, 1);
        assert!(manifest.segments.is_empty());
        assert!(manifest.latest_full().is_none());
        assert!(manifest.epoch_range().is_none());
    }

    #[test]
    fn test_manifest_with_segments() {
        let mut manifest = BackupManifest::new();
        manifest.segments.push(BackupSegment {
            kind: BackupKind::Full,
            filename: "backup_full_0000.grafeo".to_string(),
            start_epoch: EpochId::new(0),
            end_epoch: EpochId::new(100),
            checksum: 12345,
            size_bytes: 1024,
            created_at_ms: 1000,
        });
        manifest.segments.push(BackupSegment {
            kind: BackupKind::Incremental,
            filename: "backup_incr_0001.wal".to_string(),
            start_epoch: EpochId::new(101),
            end_epoch: EpochId::new(200),
            checksum: 67890,
            size_bytes: 256,
            created_at_ms: 2000,
        });

        let full = manifest.latest_full().unwrap();
        assert_eq!(full.end_epoch, EpochId::new(100));

        let incrs = manifest.incrementals_after(EpochId::new(100));
        assert_eq!(incrs.len(), 1);
        assert_eq!(incrs[0].start_epoch, EpochId::new(101));

        let (start, end) = manifest.epoch_range().unwrap();
        assert_eq!(start, EpochId::new(0));
        assert_eq!(end, EpochId::new(200));
    }

    #[test]
    fn test_manifest_round_trip() {
        let dir = TempDir::new().unwrap();
        let mut manifest = BackupManifest::new();
        manifest.segments.push(BackupSegment {
            kind: BackupKind::Full,
            filename: "test.grafeo".to_string(),
            start_epoch: EpochId::new(0),
            end_epoch: EpochId::new(50),
            checksum: 0,
            size_bytes: 512,
            created_at_ms: 0,
        });

        write_manifest(dir.path(), &manifest).unwrap();
        let loaded = read_manifest(dir.path()).unwrap().unwrap();
        assert_eq!(loaded.segments.len(), 1);
        assert_eq!(loaded.segments[0].filename, "test.grafeo");
    }

    #[test]
    fn test_manifest_not_found() {
        let dir = TempDir::new().unwrap();
        assert!(read_manifest(dir.path()).unwrap().is_none());
    }

    #[test]
    fn test_backup_cursor_round_trip() {
        let dir = TempDir::new().unwrap();
        let cursor = BackupCursor {
            backed_up_epoch: EpochId::new(42),
            log_sequence: 7,
            timestamp_ms: 12345,
        };

        write_backup_cursor(dir.path(), &cursor).unwrap();
        let loaded = read_backup_cursor(dir.path()).unwrap().unwrap();
        assert_eq!(loaded.backed_up_epoch, EpochId::new(42));
        assert_eq!(loaded.log_sequence, 7);
        assert_eq!(loaded.timestamp_ms, 12345);
    }

    /// A backup manifest as 0.6.0 writes it (bincode, standard config): a
    /// full segment of epochs 3 to 19 and an incremental one of 20 to 88.
    const MANIFEST_0_6_0: [u8; 79] = [
        1, 2, 0, 23, 98, 97, 99, 107, 117, 112, 95, 102, 117, 108, 108, 95, 48, 48, 48, 51, 46,
        103, 114, 97, 102, 101, 111, 3, 19, 88, 251, 0, 16, 253, 0, 236, 168, 19, 161, 1, 0, 0, 1,
        20, 98, 97, 99, 107, 117, 112, 95, 105, 110, 99, 114, 95, 48, 48, 50, 48, 46, 119, 97, 108,
        20, 88, 3, 251, 63, 1, 253, 192, 67, 170, 19, 161, 1, 0, 0,
    ];

    /// A backup cursor as 0.6.0 writes it: epoch 88, log sequence 3.
    const CURSOR_0_6_0: [u8; 11] = [88, 3, 253, 192, 67, 170, 19, 161, 1, 0, 0];

    /// The manifest and cursor that 0.6.0 writes still read. They are
    /// bincode, which has no field names: a field added to `BackupManifest`,
    /// `BackupSegment` or `BackupCursor` makes these bytes unreadable, even
    /// with `#[serde(default)]`, so such a change needs a reader for the old
    /// layout.
    #[test]
    fn a_manifest_and_cursor_written_by_0_6_0_still_read() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join(MANIFEST_FILENAME), MANIFEST_0_6_0).unwrap();
        std::fs::write(dir.path().join(BACKUP_CURSOR_FILENAME), CURSOR_0_6_0).unwrap();

        let manifest = read_manifest(dir.path()).unwrap().unwrap();
        assert_eq!(manifest.version, 1);
        let segments: Vec<_> = manifest
            .segments
            .iter()
            .map(|segment| {
                (
                    segment.kind,
                    segment.filename.as_str(),
                    segment.start_epoch.as_u64(),
                    segment.end_epoch.as_u64(),
                    segment.checksum,
                    segment.size_bytes,
                    segment.created_at_ms,
                )
            })
            .collect();
        assert_eq!(
            segments,
            [
                (
                    BackupKind::Full,
                    "backup_full_0003.grafeo",
                    3,
                    19,
                    88,
                    4096,
                    1_791_331_200_000
                ),
                (
                    BackupKind::Incremental,
                    "backup_incr_0020.wal",
                    20,
                    88,
                    3,
                    319,
                    1_791_331_288_000
                ),
            ]
        );

        let cursor = read_backup_cursor(dir.path()).unwrap().unwrap();
        assert_eq!(
            (
                cursor.backed_up_epoch.as_u64(),
                cursor.log_sequence,
                cursor.timestamp_ms
            ),
            (88, 3, 1_791_331_288_000)
        );
    }

    #[test]
    fn test_backup_cursor_not_found() {
        let dir = TempDir::new().unwrap();
        assert!(read_backup_cursor(dir.path()).unwrap().is_none());
    }

    #[test]
    fn test_backup_header_round_trip() {
        let mut buf = Vec::new();
        write_backup_header(&mut buf, EpochId::new(101), EpochId::new(200), 500);
        assert_eq!(buf.len(), BACKUP_HEADER_SIZE);

        let (start, end, count) = read_backup_header(&buf).unwrap();
        assert_eq!(start, EpochId::new(101));
        assert_eq!(end, EpochId::new(200));
        assert_eq!(count, 500);
    }

    #[test]
    fn test_backup_header_invalid_magic() {
        let data = vec![
            0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
        ];
        assert!(read_backup_header(&data).is_err());
    }

    #[test]
    fn test_backup_header_too_short() {
        let data = vec![0, 0, 0, 0];
        assert!(read_backup_header(&data).is_err());
    }

    #[test]
    fn test_backup_kind_serialization() {
        let config = bincode::config::standard();
        let encoded = bincode::serde::encode_to_vec(BackupKind::Full, config).unwrap();
        let (parsed, _): (BackupKind, _) =
            bincode::serde::decode_from_slice(&encoded, config).unwrap();
        assert_eq!(parsed, BackupKind::Full);
    }

    /// A full backup records the epoch of the image it copied. After a
    /// checkpoint failed once its database header was written, the manager
    /// still serves the previous image (epoch 3), but the file opens at the
    /// new one (epoch 19): the backup must say 19, or a restore to an epoch
    /// in between would copy it and return newer data.
    #[cfg(feature = "testing-crash-injection")]
    #[test]
    fn a_full_backup_records_the_epoch_of_the_image_it_copied() {
        use grafeo_common::storage::{
            Section, SectionSink, SectionSource, SectionType, read_raw, write_raw,
        };
        use grafeo_common::testing::crash::with_failure_at;
        use grafeo_storage::file::CheckpointHeader;

        /// A catalog section holding fixed bytes.
        struct Fixed(&'static [u8]);

        impl Section for Fixed {
            fn section_type(&self) -> SectionType {
                SectionType::Catalog
            }

            fn serialize(&self) -> Result<Vec<u8>> {
                Ok(self.0.to_vec())
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
                true
            }

            fn mark_clean(&self) {}

            fn memory_usage(&self) -> usize {
                self.0.len()
            }
        }

        let dir = TempDir::new().unwrap();
        let backup_dir = dir.path().join("backups");
        let fm = GrafeoFileManager::create(dir.path().join("vincent.grafeo"), None).unwrap();
        let checkpoint = |data: &'static [u8], epoch: u64| {
            fm.write_checkpoint(
                &[&Fixed(data)],
                &CheckpointHeader {
                    epoch,
                    ..CheckpointHeader::default()
                },
            )
        };
        checkpoint(b"Alix", 3).unwrap();
        // The third failure point of a checkpoint is checkpoint:after_header.
        let error = with_failure_at(3, || checkpoint(b"Gus", 19))
            .unwrap_err()
            .to_string();
        assert!(error.contains("checkpoint:after_header"), "{error}");
        assert_eq!(
            fm.active_header().epoch,
            3,
            "the manager serves the previous image"
        );

        let segment = do_backup_full(&backup_dir, &fm, None).unwrap();
        let copy = GrafeoFileManager::open(backup_dir.join(&segment.filename), None).unwrap();
        assert_eq!(
            copy.active_header().epoch,
            19,
            "the copy opens at the new image"
        );
        assert_eq!(
            segment.end_epoch,
            EpochId::new(19),
            "the backup records the epoch of the image it copied"
        );
    }

    #[test]
    fn test_do_restore_to_epoch_refuses_existing_output() {
        let dir = TempDir::new().unwrap();
        let backup_dir = dir.path().join("backup");
        std::fs::create_dir_all(&backup_dir).unwrap();

        // A pre-existing database file at the restore target must be refused
        // before any copy/overwrite happens (the guard fires ahead of manifest
        // reading, so no real backup chain is needed for this case).
        let output_path = dir.path().join("live.grafeo");
        std::fs::write(&output_path, b"existing database, must not be clobbered").unwrap();

        let err = do_restore_to_epoch(
            &backup_dir,
            EpochId::new(0),
            &output_path,
            &DatabaseKeys::none(),
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("already exists"),
            "expected refuse-if-exists error, got: {msg}"
        );

        // The existing file must be left untouched by the refused restore.
        assert_eq!(
            std::fs::read(&output_path).unwrap(),
            b"existing database, must not be clobbered"
        );
    }

    #[test]
    fn test_do_restore_to_epoch_refuses_existing_sidecar() {
        let dir = TempDir::new().unwrap();
        let backup_dir = dir.path().join("backup");
        std::fs::create_dir_all(&backup_dir).unwrap();

        // No `.grafeo` file, but a leftover sidecar WAL at the target is still a
        // non-fresh path: restoring would delete it.
        let output_path = dir.path().join("live.grafeo");
        let sidecar = dir.path().join("live.grafeo.wal");
        std::fs::create_dir_all(&sidecar).unwrap();

        let err = do_restore_to_epoch(
            &backup_dir,
            EpochId::new(0),
            &output_path,
            &DatabaseKeys::none(),
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("already exists"),
            "expected refuse-if-exists error for leftover sidecar, got: {err}"
        );
        assert!(
            sidecar.exists(),
            "refused restore must leave the sidecar in place"
        );
    }
}
