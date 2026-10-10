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
/// stored as JSON in the backup manifest: a release ignores the fields it
/// does not know, and a field added later needs `#[serde(default)]`, so the
/// manifests written before it still read.
///
/// ```compile_fail,E0639
/// use grafeo_common::types::EpochId;
/// use grafeo_engine::database::backup::{BackupKind, BackupSegment};
///
/// let segment = BackupSegment {
///     kind: BackupKind::Full,
///     filename: "backup_full_0000.grafeo".to_string(),
///     start_epoch: EpochId::new(0),
///     end_epoch: EpochId::new(19),
///     checksum: 88,
///     size_bytes: 4096,
///     created_at_ms: 0,
/// };
/// ```
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
/// stored as JSON in `backup_manifest.json`: a release ignores the fields it
/// does not know, and a field added later needs `#[serde(default)]`, so the
/// manifests written before it still read. Up to 0.5.x the file held
/// bincode; [`read_manifest`] still reads it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct BackupManifest {
    /// Manifest format version: 2, the JSON manifest. A 0.5.x manifest
    /// (version 1, bincode) reads as version 2, the format it is written in
    /// next. The version changes only for a change that older releases
    /// cannot read, and [`read_manifest`] refuses a newer one.
    pub version: u32,
    /// Ordered list of backup segments (full first, then incrementals).
    pub segments: Vec<BackupSegment>,
}

impl BackupManifest {
    /// Creates a new empty manifest.
    #[must_use]
    pub fn new() -> Self {
        Self {
            version: MANIFEST_VERSION,
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
/// stored as JSON: a release ignores the fields it does not know, and a field
/// added later needs `#[serde(default)]`, so the cursors written before it
/// still read. Up to 0.5.x the file held bincode; [`read_backup_cursor`]
/// still reads it.
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

/// The format version of the JSON manifest (0.5.x wrote version 1, in
/// bincode).
const MANIFEST_VERSION: u32 = 2;

/// Reads the backup manifest from a backup directory.
///
/// Returns `None` if no manifest exists. Reads the JSON manifest and the
/// bincode manifest of 0.5.x.
///
/// # Errors
///
/// Returns an error if the manifest file exists but cannot be read, or if a
/// newer release wrote it in a format version this one cannot read, and
/// [`Error::Corruption`] naming it if it does not parse.
pub fn read_manifest(backup_dir: &Path) -> Result<Option<BackupManifest>> {
    let path = backup_dir.join(MANIFEST_FILENAME);
    if !path.exists() {
        return Ok(None);
    }
    let data = std::fs::read(&path)
        .map_err(|e| Error::Internal(format!("failed to read backup manifest: {e}")))?;
    let manifest = decode_manifest(&data).map_err(|e| match e {
        ManifestDecodeError::Newer(reason) => Error::Internal(format!(
            "cannot read backup manifest {}: {reason}",
            path.display()
        )),
        ManifestDecodeError::Invalid(reason) => {
            Error::corruption(format!("the backup manifest does not parse: {reason}"))
                .in_file(&path)
        }
    })?;
    Ok(Some(manifest))
}

/// Why a manifest did not decode: a newer release wrote it (not damage), or it
/// is neither JSON nor a 0.5.x bincode manifest.
#[derive(Debug)]
enum ManifestDecodeError {
    /// A newer release wrote it in a format version this one cannot read.
    Newer(String),
    /// It does not parse.
    Invalid(String),
}

/// Decodes a manifest: JSON, or else the bincode of 0.5.x.
fn decode_manifest(data: &[u8]) -> std::result::Result<BackupManifest, ManifestDecodeError> {
    let json_error = match serde_json::from_slice::<BackupManifest>(data) {
        Ok(manifest) if manifest.version > MANIFEST_VERSION => {
            return Err(ManifestDecodeError::Newer(format!(
                "format version {} was written by a newer release; this one reads up to \
                 version {MANIFEST_VERSION}",
                manifest.version
            )));
        }
        Ok(manifest) => return Ok(manifest),
        Err(e) => e,
    };
    bincode_0_5::manifest(data).map_err(|bincode_error| {
        ManifestDecodeError::Invalid(format!(
            "neither JSON ({json_error}) nor a 0.5.x bincode manifest ({bincode_error})"
        ))
    })
}

/// Writes the backup manifest to a backup directory, as JSON.
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

    let data = serde_json::to_vec_pretty(manifest)
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
/// Returns `None` if no cursor exists (no backup has been taken). Reads the
/// JSON cursor and the bincode cursor of 0.5.x.
///
/// # Errors
///
/// Returns an error if the cursor file exists but cannot be read, and
/// [`Error::Corruption`] naming it if it does not parse.
pub fn read_backup_cursor(wal_dir: &Path) -> Result<Option<BackupCursor>> {
    let path = wal_dir.join(BACKUP_CURSOR_FILENAME);
    if !path.exists() {
        return Ok(None);
    }
    let data = std::fs::read(&path)
        .map_err(|e| Error::Internal(format!("failed to read backup cursor: {e}")))?;
    let cursor = decode_cursor(&data).map_err(|reason| {
        Error::corruption(format!("the backup cursor does not parse: {reason}")).in_file(&path)
    })?;
    Ok(Some(cursor))
}

/// Decodes a cursor: JSON, or else the bincode of 0.5.x. The order matters:
/// a bincode cursor can start with `{` (epoch 123), but it is never JSON.
fn decode_cursor(data: &[u8]) -> std::result::Result<BackupCursor, String> {
    serde_json::from_slice::<BackupCursor>(data).or_else(|json_error| {
        bincode_0_5::cursor(data).map_err(|bincode_error| {
            format!("neither JSON ({json_error}) nor a 0.5.x bincode cursor ({bincode_error})")
        })
    })
}

/// Writes the backup cursor to a WAL directory, as JSON.
///
/// Uses write-to-temp-then-rename for atomicity.
///
/// # Errors
///
/// Returns an error if the cursor cannot be written.
pub fn write_backup_cursor(wal_dir: &Path, cursor: &BackupCursor) -> Result<()> {
    let path = wal_dir.join(BACKUP_CURSOR_FILENAME);
    let temp_path = wal_dir.join(format!("{BACKUP_CURSOR_FILENAME}.tmp"));

    let data = serde_json::to_vec_pretty(cursor)
        .map_err(|e| Error::Internal(format!("failed to serialize backup cursor: {e}")))?;

    std::fs::write(&temp_path, &data)
        .map_err(|e| Error::Internal(format!("failed to write backup cursor: {e}")))?;
    std::fs::rename(&temp_path, &path)
        .map_err(|e| Error::Internal(format!("failed to finalize backup cursor: {e}")))?;

    Ok(())
}

/// The manifest and cursor as 0.5.x wrote them: bincode 2 with the standard
/// configuration, which has no field names, so these frozen copies of the
/// 0.5.x layout read them whatever fields the current types gain. Removed in
/// 0.7.0 with the other 0.5.x readers.
mod bincode_0_5 {
    use serde::Deserialize;

    use super::{BackupCursor, BackupKind, BackupManifest, BackupSegment, MANIFEST_VERSION};
    use grafeo_common::types::EpochId;

    /// The only version 0.5.x wrote.
    const VERSION: u32 = 1;

    /// A cap on what a decode claims, so a damaged file fails to decode
    /// instead of asking for a huge allocation (a 0.5.x manifest of 64 MiB
    /// would hold about a million segments).
    const LIMIT: usize = 64 << 20;

    #[derive(Deserialize)]
    enum Kind {
        Full,
        Incremental,
    }

    #[derive(Deserialize)]
    struct Segment {
        kind: Kind,
        filename: String,
        start_epoch: u64,
        end_epoch: u64,
        checksum: u32,
        size_bytes: u64,
        created_at_ms: u64,
    }

    #[derive(Deserialize)]
    struct Manifest {
        version: u32,
        segments: Vec<Segment>,
    }

    #[derive(Deserialize)]
    struct Cursor {
        backed_up_epoch: u64,
        log_sequence: u64,
        timestamp_ms: u64,
    }

    /// Decodes `data` as a whole: a 0.5.x file is exactly one value, so
    /// bytes left over mean it is something else.
    fn decode<T: serde::de::DeserializeOwned>(data: &[u8]) -> Result<T, String> {
        let config = bincode::config::standard().with_limit::<LIMIT>();
        let (value, read) =
            bincode::serde::decode_from_slice(data, config).map_err(|e| e.to_string())?;
        if read != data.len() {
            return Err(format!(
                "{} bytes left over after the first {read}",
                data.len() - read
            ));
        }
        Ok(value)
    }

    /// A 0.5.x manifest, as the current manifest (format version 2, the
    /// format it is written in next).
    pub(super) fn manifest(data: &[u8]) -> Result<BackupManifest, String> {
        let manifest: Manifest = decode(data)?;
        if manifest.version != VERSION {
            return Err(format!(
                "version {} (0.5.x wrote version {VERSION})",
                manifest.version
            ));
        }
        Ok(BackupManifest {
            version: MANIFEST_VERSION,
            segments: manifest
                .segments
                .into_iter()
                .map(|segment| BackupSegment {
                    kind: match segment.kind {
                        Kind::Full => BackupKind::Full,
                        Kind::Incremental => BackupKind::Incremental,
                    },
                    filename: segment.filename,
                    start_epoch: EpochId::new(segment.start_epoch),
                    end_epoch: EpochId::new(segment.end_epoch),
                    checksum: segment.checksum,
                    size_bytes: segment.size_bytes,
                    created_at_ms: segment.created_at_ms,
                })
                .collect(),
        })
    }

    /// A 0.5.x backup cursor.
    pub(super) fn cursor(data: &[u8]) -> Result<BackupCursor, String> {
        let cursor: Cursor = decode(data)?;
        Ok(BackupCursor {
            backed_up_epoch: EpochId::new(cursor.backed_up_epoch),
            log_sequence: cursor.log_sequence,
            timestamp_ms: cursor.timestamp_ms,
        })
    }
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
            active.ok_or_else(|| Error::corruption("both database header slots are empty"))
        })
        .map_err(|e| match e {
            Error::Corruption(_) => e
                .wrapped("full backup: the copied file's database header")
                .in_file(path),
            other => Error::Internal(format!(
                "full backup {}: cannot read the copied file's database header: {other}",
                path.display()
            )),
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
            let header = FileHeaderV3::decode(&page).map_err(|e| match e {
                Error::Corruption(_) => e.wrapped("full backup").in_file(path),
                other => other.wrapped(format_args!("full backup {}", path.display())),
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
        assert_eq!(manifest.version, 2, "a new manifest is a JSON manifest");
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

    /// A chain of a full segment of epochs 0 to 19 and an incremental one of
    /// 20 to 88.
    fn chain() -> BackupManifest {
        let mut manifest = BackupManifest::new();
        manifest.segments.push(BackupSegment {
            kind: BackupKind::Full,
            filename: "backup_full_0000.grafeo".to_string(),
            start_epoch: EpochId::new(0),
            end_epoch: EpochId::new(19),
            checksum: 88,
            size_bytes: 4096,
            created_at_ms: 1_791_331_200_000,
        });
        manifest.segments.push(BackupSegment {
            kind: BackupKind::Incremental,
            filename: "backup_incr_0001.wal".to_string(),
            start_epoch: EpochId::new(20),
            end_epoch: EpochId::new(88),
            checksum: 3,
            size_bytes: 319,
            created_at_ms: 1_791_331_288_000,
        });
        manifest
    }

    /// Every field of every segment.
    fn fields(manifest: &BackupManifest) -> Vec<(BackupKind, String, u64, u64, u32, u64, u64)> {
        manifest
            .segments
            .iter()
            .map(|segment| {
                (
                    segment.kind,
                    segment.filename.clone(),
                    segment.start_epoch.as_u64(),
                    segment.end_epoch.as_u64(),
                    segment.checksum,
                    segment.size_bytes,
                    segment.created_at_ms,
                )
            })
            .collect()
    }

    /// `chain()` as `write_manifest` writes it.
    const MANIFEST_JSON: &str = r#"{
  "version": 2,
  "segments": [
    {
      "kind": "Full",
      "filename": "backup_full_0000.grafeo",
      "start_epoch": 0,
      "end_epoch": 19,
      "checksum": 88,
      "size_bytes": 4096,
      "created_at_ms": 1791331200000
    },
    {
      "kind": "Incremental",
      "filename": "backup_incr_0001.wal",
      "start_epoch": 20,
      "end_epoch": 88,
      "checksum": 3,
      "size_bytes": 319,
      "created_at_ms": 1791331288000
    }
  ]
}"#;

    /// A cursor at epoch 88 and log sequence 3, as `write_backup_cursor`
    /// writes it.
    const CURSOR_JSON: &str = r#"{
  "backed_up_epoch": 88,
  "log_sequence": 3,
  "timestamp_ms": 1791331288000
}"#;

    fn cursor_fields(cursor: &BackupCursor) -> (u64, u64, u64) {
        (
            cursor.backed_up_epoch.as_u64(),
            cursor.log_sequence,
            cursor.timestamp_ms,
        )
    }

    /// The manifest is written as JSON, the file its name promises, and
    /// reads back with every field.
    #[test]
    fn a_manifest_is_written_as_json_and_reads_back() {
        let dir = TempDir::new().unwrap();
        write_manifest(dir.path(), &chain()).unwrap();
        let written = std::fs::read_to_string(dir.path().join(MANIFEST_FILENAME)).unwrap();
        assert_eq!(written, MANIFEST_JSON);

        let loaded = read_manifest(dir.path()).unwrap().unwrap();
        assert_eq!(loaded.version, 2);
        assert_eq!(fields(&loaded), fields(&chain()));
    }

    /// A manifest from a later 0.6 release, with fields this one does not
    /// know at the top and in a segment, reads: the unknown fields are
    /// ignored.
    #[test]
    fn a_json_manifest_with_fields_this_release_does_not_know_reads() {
        let dir = TempDir::new().unwrap();
        let later = MANIFEST_JSON
            .replacen(
                r#""version": 2,"#,
                r#""version": 2, "compression": "zstd", "database": {"name": "prague"},"#,
                1,
            )
            .replacen(
                r#""created_at_ms": 1791331200000"#,
                r#""created_at_ms": 1791331200000, "database_id": 19, "parent": null"#,
                1,
            );
        assert_ne!(later, MANIFEST_JSON, "the extra fields are in");
        std::fs::write(dir.path().join(MANIFEST_FILENAME), later).unwrap();

        let loaded = read_manifest(dir.path()).unwrap().unwrap();
        assert_eq!(loaded.version, 2);
        assert_eq!(fields(&loaded), fields(&chain()));
    }

    /// A manifest of a format version above 2 is refused, not read as far as
    /// its fields happen to match.
    #[test]
    fn a_manifest_of_a_newer_format_version_is_refused() {
        let dir = TempDir::new().unwrap();
        let newer = MANIFEST_JSON.replacen(r#""version": 2"#, r#""version": 3"#, 1);
        std::fs::write(dir.path().join(MANIFEST_FILENAME), newer).unwrap();

        let error = read_manifest(dir.path()).unwrap_err().to_string();
        assert!(
            error.contains("format version 3") && error.contains("newer release"),
            "{error}"
        );
    }

    #[test]
    fn test_manifest_not_found() {
        let dir = TempDir::new().unwrap();
        assert!(read_manifest(dir.path()).unwrap().is_none());
    }

    /// The cursor is written as JSON and reads back with every field.
    #[test]
    fn a_cursor_is_written_as_json_and_reads_back() {
        let dir = TempDir::new().unwrap();
        let cursor = BackupCursor {
            backed_up_epoch: EpochId::new(88),
            log_sequence: 3,
            timestamp_ms: 1_791_331_288_000,
        };
        write_backup_cursor(dir.path(), &cursor).unwrap();
        let written = std::fs::read_to_string(dir.path().join(BACKUP_CURSOR_FILENAME)).unwrap();
        assert_eq!(written, CURSOR_JSON);

        let loaded = read_backup_cursor(dir.path()).unwrap().unwrap();
        assert_eq!(cursor_fields(&loaded), (88, 3, 1_791_331_288_000));
    }

    /// A cursor with a field this release does not know reads.
    #[test]
    fn a_json_cursor_with_fields_this_release_does_not_know_reads() {
        let dir = TempDir::new().unwrap();
        let later = CURSOR_JSON.replacen(
            r#""log_sequence": 3,"#,
            r#""log_sequence": 3, "wal_format": 2, "segments": [19, 88],"#,
            1,
        );
        assert_ne!(later, CURSOR_JSON, "the extra fields are in");
        std::fs::write(dir.path().join(BACKUP_CURSOR_FILENAME), later).unwrap();

        let loaded = read_backup_cursor(dir.path()).unwrap().unwrap();
        assert_eq!(cursor_fields(&loaded), (88, 3, 1_791_331_288_000));
    }

    /// A backup manifest as 0.5.x wrote it (bincode, standard config): a
    /// full segment of epochs 3 to 19 and an incremental one of 20 to 88.
    const MANIFEST_BINCODE_0_5: [u8; 79] = [
        1, 2, 0, 23, 98, 97, 99, 107, 117, 112, 95, 102, 117, 108, 108, 95, 48, 48, 48, 51, 46,
        103, 114, 97, 102, 101, 111, 3, 19, 88, 251, 0, 16, 253, 0, 236, 168, 19, 161, 1, 0, 0, 1,
        20, 98, 97, 99, 107, 117, 112, 95, 105, 110, 99, 114, 95, 48, 48, 50, 48, 46, 119, 97, 108,
        20, 88, 3, 251, 63, 1, 253, 192, 67, 170, 19, 161, 1, 0, 0,
    ];

    /// A backup cursor as 0.5.x wrote it: epoch 88, log sequence 3.
    const CURSOR_BINCODE_0_5: [u8; 11] = [88, 3, 253, 192, 67, 170, 19, 161, 1, 0, 0];

    /// The bincode manifest and cursor that 0.5.x wrote (under the same file
    /// names) still read, the manifest as format version 2, which it is
    /// written in next. `tests/backup_restore.rs` restores a whole chain
    /// that 0.5.44 wrote.
    #[test]
    fn a_bincode_manifest_and_cursor_as_0_5_x_wrote_them_still_read() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join(MANIFEST_FILENAME), MANIFEST_BINCODE_0_5).unwrap();
        std::fs::write(dir.path().join(BACKUP_CURSOR_FILENAME), CURSOR_BINCODE_0_5).unwrap();

        let manifest = read_manifest(dir.path()).unwrap().unwrap();
        assert_eq!(manifest.version, 2, "read as the format it is written in");
        assert_eq!(
            fields(&manifest),
            [
                (
                    BackupKind::Full,
                    "backup_full_0003.grafeo".to_string(),
                    3,
                    19,
                    88,
                    4096,
                    1_791_331_200_000
                ),
                (
                    BackupKind::Incremental,
                    "backup_incr_0020.wal".to_string(),
                    20,
                    88,
                    3,
                    319,
                    1_791_331_288_000
                ),
            ]
        );

        let cursor = read_backup_cursor(dir.path()).unwrap().unwrap();
        assert_eq!(cursor_fields(&cursor), (88, 3, 1_791_331_288_000));
    }

    /// A bincode cursor of epoch 123 starts with `{`, as JSON does: it is
    /// still read as bincode.
    #[test]
    fn a_bincode_cursor_that_starts_like_json_still_reads() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join(BACKUP_CURSOR_FILENAME), [b'{', 3, 19]).unwrap();

        let cursor = read_backup_cursor(dir.path()).unwrap().unwrap();
        assert_eq!(cursor_fields(&cursor), (123, 3, 19));
    }

    /// A damaged manifest or cursor is refused: a torn JSON file, and bincode
    /// that is not exactly one 0.5.x value.
    #[test]
    fn a_manifest_or_cursor_that_is_neither_json_nor_0_5_x_bincode_is_refused() {
        let dir = TempDir::new().unwrap();
        let manifest = |bytes: &[u8]| {
            std::fs::write(dir.path().join(MANIFEST_FILENAME), bytes).unwrap();
            read_manifest(dir.path())
                .map(|_| ())
                .unwrap_err()
                .to_string()
        };

        let torn = manifest(&MANIFEST_JSON.as_bytes()[..88]);
        assert!(
            torn.contains("neither JSON") && torn.contains("0.5.x bincode manifest"),
            "{torn}"
        );
        let mut trailing = MANIFEST_BINCODE_0_5.to_vec();
        trailing.push(19);
        let trailing = manifest(&trailing);
        assert!(trailing.contains("1 bytes left over"), "{trailing}");
        let mut version_2 = MANIFEST_BINCODE_0_5;
        version_2[0] = 2;
        let version_2 = manifest(&version_2);
        assert!(version_2.contains("0.5.x wrote version 1"), "{version_2}");

        std::fs::write(
            dir.path().join(BACKUP_CURSOR_FILENAME),
            &CURSOR_JSON.as_bytes()[..19],
        )
        .unwrap();
        let cursor = read_backup_cursor(dir.path())
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(
            cursor.contains("neither JSON") && cursor.contains("0.5.x bincode cursor"),
            "{cursor}"
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

    /// The kinds are stored by name: a manifest names them as written here.
    #[test]
    fn backup_kinds_are_stored_by_name() {
        for (kind, name) in [
            (BackupKind::Full, r#""Full""#),
            (BackupKind::Incremental, r#""Incremental""#),
        ] {
            assert_eq!(serde_json::to_string(&kind).unwrap(), name);
            assert_eq!(serde_json::from_str::<BackupKind>(name).unwrap(), kind);
        }
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
