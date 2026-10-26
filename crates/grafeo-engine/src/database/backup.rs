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

use std::io::Write;
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
    /// The log position the segment ends at: every WAL group before it is in
    /// the chain up to this segment, and the next incremental backup starts
    /// there. `None` in a manifest written before 0.6.1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_wal_lsn: Option<u64>,
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
    /// The log file sequence of the last backup, in the WAL format of 0.5.x
    /// and 0.6.0-beta.1; 0 in the cursors this release writes (see
    /// [`wal_lsn`](Self::wal_lsn)). Removed in 0.7.0.
    pub log_sequence: u64,
    /// The log position up to which the WAL has been backed up: the next
    /// incremental backup copies the segments from there. `None` in a cursor
    /// of the earlier WAL format, which only a full backup replaces.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wal_lsn: Option<u64>,
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
    prepare_backup_directory(backup_dir)?;
    write_prepared_manifest(backup_dir, manifest)
}

/// Writes a manifest after this operation has prepared the backup directory.
fn write_prepared_manifest(backup_dir: &Path, manifest: &BackupManifest) -> Result<()> {
    let path = backup_dir.join(MANIFEST_FILENAME);
    let data = serde_json::to_vec_pretty(manifest)
        .map_err(|e| Error::Internal(format!("failed to serialize backup manifest: {e}")))?;

    write_metadata(
        &path,
        &data,
        "manifest",
        "backup:manifest_sync",
        "backup:manifest_directory_sync",
    )
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
    let data = serde_json::to_vec_pretty(cursor)
        .map_err(|e| Error::Internal(format!("failed to serialize backup cursor: {e}")))?;

    write_metadata(
        &path,
        &data,
        "cursor",
        "backup:cursor_sync",
        "backup:cursor_directory_sync",
    )
}

/// Publishes synced metadata, then makes its renamed directory entry durable.
fn write_metadata(
    path: &Path,
    data: &[u8],
    kind: &str,
    sync_point: &str,
    directory_sync_point: &str,
) -> Result<()> {
    let mut temp_path = path.as_os_str().to_owned();
    temp_path.push(".tmp");
    let temp_path = std::path::PathBuf::from(temp_path);
    let mut file = std::fs::File::create(&temp_path).map_err(|e| {
        Error::Internal(format!(
            "failed to create backup {kind} {}: {e}",
            temp_path.display()
        ))
    })?;
    file.write_all(data).map_err(|e| {
        Error::Internal(format!(
            "failed to write backup {kind} {}: {e}",
            temp_path.display()
        ))
    })?;
    grafeo_common::testing::crash::maybe_fail(sync_point)?;
    file.sync_all().map_err(|e| {
        Error::Internal(format!(
            "failed to sync backup {kind} {}: {e}",
            temp_path.display()
        ))
    })?;
    drop(file);
    std::fs::rename(&temp_path, path).map_err(|e| {
        Error::Internal(format!(
            "failed to finalize backup {kind} {}: {e}",
            path.display()
        ))
    })?;
    sync_backup_entry(path, directory_sync_point)
}

/// Persists the directory entry of a backup file or directory. This retains
/// the platform floor of `flush::sync_parent_dir`: no directory fsync off Unix.
fn sync_backup_entry(path: &Path, sync_point: &str) -> Result<()> {
    #[cfg(unix)]
    grafeo_common::testing::crash::maybe_fail(sync_point)?;
    #[cfg(not(unix))]
    let _ = sync_point;
    super::flush::sync_parent_dir(path).map_err(|e| {
        Error::Internal(format!(
            "failed to sync directory containing backup {}: {e}",
            path.display()
        ))
    })
}

/// Prepares the directory once per backup. Sync the existing directory's
/// parent once, also repairing an entry left by a failed earlier attempt;
/// then persist each new directory entry before creating its children.
/// Keep lexical paths so symlinks and `..` resolve as they do in create_dir_all.
fn prepare_backup_directory(directory: &Path) -> Result<()> {
    let mut existing = if directory.as_os_str().is_empty() {
        Path::new(".")
    } else {
        directory
    };
    let mut missing = Vec::new();
    loop {
        match std::fs::metadata(existing) {
            Ok(metadata) if metadata.is_dir() => break,
            Ok(_) => {
                return Err(Error::Internal(format!(
                    "backup directory {} is not a directory",
                    existing.display()
                )));
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                missing.push(existing);
                existing = existing
                    .parent()
                    .filter(|parent| !parent.as_os_str().is_empty())
                    .unwrap_or_else(|| Path::new("."));
            }
            Err(e) => {
                return Err(Error::Internal(format!(
                    "failed to inspect backup directory {}: {e}",
                    existing.display()
                )));
            }
        }
    }
    sync_backup_entry(existing, "backup:directory_sync")?;
    for directory in missing.into_iter().rev() {
        match std::fs::create_dir(directory) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists && directory.is_dir() => {}
            Err(e) => {
                return Err(Error::Internal(format!(
                    "failed to create backup directory {}: {e}",
                    directory.display()
                )));
            }
        }
        sync_backup_entry(directory, "backup:new_directory_sync")?;
    }
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
                    end_wal_lsn: None,
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
            wal_lsn: None,
            timestamp_ms: cursor.timestamp_ms,
        })
    }
}

// ── Incremental backup file format ─────────────────────────────────

/// Magic bytes for incremental backup files.
pub const BACKUP_MAGIC: [u8; 4] = *b"GBAK";
/// Current backup file version: 2 holds WAL segments; 1, written by 0.5.x
/// and 0.6.0-beta.1, holds the frames of the earlier WAL format.
pub const BACKUP_VERSION: u32 = 2;

/// The backup file version of the earlier WAL format.
const FIRST_FORMAT_VERSION: u32 = 1;

/// Header for an incremental backup file.
///
/// ```text
/// [magic: 4 bytes "GBAK"]
/// [version: u32 LE]
/// [start_epoch: u64 LE]
/// [end_epoch: u64 LE]
/// [record_count: u64 LE]   (version 2: the groups it holds)
/// version 2: per WAL segment [length: u64 LE][the segment file]
/// version 1: the frames of the earlier WAL format
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

/// The version of a valid incremental backup file (see [`BACKUP_VERSION`]).
fn backup_version(data: &[u8]) -> Result<u32> {
    read_backup_header(data)?;
    Ok(u32::from_le_bytes([data[4], data[5], data[6], data[7]]))
}

/// Returns the timestamp in milliseconds since UNIX epoch.
// reason: millis since UNIX epoch fits u64 for centuries
#[allow(clippy::cast_possible_truncation)]
pub(super) fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

// ── Backup operations (called from GrafeoDB) ───────────────────────

use std::io::Read;

use grafeo_storage::file::GrafeoFileManager;
use grafeo_storage::file::detect::{OnDisk, detect};
use grafeo_storage::file::v3::header::{DbHeaderV3, FileHeaderV3, active_header};
use grafeo_storage::wal::{
    CipherForSalt, DurabilityMode, ScanOptions, SegmentHeader, Wal, WalCipher, WalOptions, WalScan,
    list_wal_directory, segment_file_name,
};

use super::encryption::DatabaseKeys;

/// Size of a page (and of each header) of a `.grafeo` file, in bytes.
const PAGE_BYTES: usize = 4096;

impl super::GrafeoDB {
    /// The log position a full backup of this database covers when it has no
    /// WAL to start a segment in: for a read-only database, the end of the
    /// sidecar WAL its open replayed, whose commits the image holds (the
    /// shared lock keeps writers out, so the WAL still ends there); otherwise
    /// the position the file's last checkpoint recorded. A WAL of the format
    /// before 0.6.0 has no positions of this format: the file's, as its
    /// commits cannot chain to incremental backups anyway.
    pub(super) fn log_end_without_wal(
        &self,
        fm: &GrafeoFileManager,
        keys: &DatabaseKeys,
    ) -> Result<u64> {
        let checkpoint_lsn = fm.active_header().checkpoint_lsn;
        if !self.read_only
            || !fm.has_sidecar_wal()
            || Self::holds_first_format_wal(&fm.sidecar_wal_path())?
        {
            return Ok(checkpoint_lsn);
        }
        let options = Self::scan_options(fm, keys, self.config.wal_salvage());
        let mut scan = WalScan::open(fm.sidecar_wal_path(), options).map_err(Error::from)?;
        while scan.next_group().map_err(Error::from)?.is_some() {}
        let end = scan.finish().map_err(Error::from)?;
        Ok(end.end_lsn.max(checkpoint_lsn))
    }
}

/// Creates a full backup: a complete image of the database at an epoch E
/// (#362).
///
/// The image is written from a snapshot at E, as a checkpoint takes it (see
/// [`write_copy`](super::flush::write_copy)): commits are held off only while
/// the snapshot is taken and the sections without versions are written, and
/// go on while the graph is written. The image has the database's id and
/// keys, and holds every commit up to E and nothing after it. With `wal`, a
/// new WAL segment starts while commits are held: the groups after E are in
/// the segments from that one on, which the next incremental backup copies,
/// so a restore is the image and those segments. Without, `log_end` gives
/// the log position the image covers (a read-only database: the end of the
/// sidecar WAL it replayed, whose commits the image holds). The database's
/// own file is not touched.
///
/// The image is written under a temporary name and renamed once complete, so
/// the backup directory never holds a full backup that opens as less than E;
/// the manifest and the cursor are written after it.
///
/// # Errors
///
/// Returns an error if commits cannot be held, a section fails to write, or
/// I/O fails.
pub(super) fn do_backup_full(
    backup_dir: &Path,
    fm: &GrafeoFileManager,
    sources: &super::sections::CheckpointSources,
    keys: &DatabaseKeys,
    wal: Option<&Wal>,
    log_end: impl FnOnce() -> Result<u64>,
) -> Result<BackupSegment> {
    prepare_backup_directory(backup_dir)?;

    // Determine backup filename
    let mut manifest = read_manifest(backup_dir)?.unwrap_or_default();
    let segment_idx = manifest.segments.len();
    let filename = format!("backup_full_{segment_idx:04}.grafeo");
    let dest_path = backup_dir.join(&filename);
    // The name follows the manifest's segments, so a file under it is none
    // the manifest lists: the image of a backup a crash or an error cut off
    // before its manifest entry, which no restore reads.
    match std::fs::remove_file(&dest_path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => {
            return Err(Error::Internal(format!(
                "failed to remove the unlisted backup file {}: {e}",
                dest_path.display()
            )));
        }
    }

    // No checkpoint may remove the WAL segments from the image's log
    // position on before the cursor below names them.
    let _checkpoint = fm.checkpoint_guard();

    let written =
        super::flush::write_copy(sources, &dest_path, fm.database_id(), keys, || match wal {
            Some(wal) => Ok(wal.rotate()?),
            None => log_end(),
        })?;
    let (end_epoch, wal_lsn) = (written.epoch, written.checkpoint_lsn);
    grafeo_common::testing::crash::maybe_crash("backup:after_image");
    let (checksum, file_size) = file_checksum(&dest_path)?;

    let segment = BackupSegment {
        kind: BackupKind::Full,
        filename,
        start_epoch: EpochId::new(0),
        end_epoch: EpochId::new(end_epoch),
        checksum,
        size_bytes: file_size,
        created_at_ms: now_ms(),
        end_wal_lsn: Some(wal_lsn),
    };

    manifest.segments.push(segment.clone());
    write_prepared_manifest(backup_dir, &manifest)?;
    grafeo_common::testing::crash::maybe_crash("backup:after_manifest");

    // Every WAL group before the image's log position is in the image, and
    // none after it: the next incremental backup starts at that segment.
    if let Some(wal) = wal {
        let cursor = BackupCursor {
            backed_up_epoch: EpochId::new(end_epoch),
            log_sequence: 0,
            wal_lsn: Some(wal_lsn),
            timestamp_ms: now_ms(),
        };
        write_backup_cursor(wal.dir(), &cursor)?;
    }

    Ok(segment)
}

/// The CRC-32 and size of the file at `path`, read in pieces.
fn file_checksum(path: &Path) -> Result<(u32, u64)> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)
        .map_err(|e| Error::Internal(format!("failed to read backup file for checksum: {e}")))?;
    let mut hasher = crc32fast::Hasher::new();
    let mut buffer = vec![0u8; 1 << 20];
    let mut size = 0u64;
    loop {
        let read = file.read(&mut buffer).map_err(|e| {
            Error::Internal(format!("failed to read backup file for checksum: {e}"))
        })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        size += read as u64;
    }
    Ok((hasher.finalize(), size))
}

/// The active database header of a copied `.grafeo` file: the image it opens
/// at, with its epoch and log position (headers are never encrypted).
///
/// The manager's in-memory header can be older: after a checkpoint that
/// failed once its header was written, the manager keeps serving the
/// previous image while the file opens at the new one.
fn copied_header(file_data: &[u8], path: &Path) -> Result<DbHeaderV3> {
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
    Ok(header)
}

/// Creates an incremental backup containing the WAL groups since the last
/// backup: it seals the active segment, then copies the segments from the
/// cursor's log position up to it, whole.
///
/// # Errors
///
/// Returns an error if no full backup exists, the cursor is of the earlier
/// WAL format, the cursor is ahead of the chain's end epoch, nothing was
/// logged since the last backup, or the segments after the cursor were removed.
pub(super) fn do_backup_incremental(
    backup_dir: &Path,
    wal: &Wal,
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
    let from = cursor.wal_lsn.ok_or_else(|| {
        Error::Internal(
            "the backup cursor is of the WAL format before 0.6.0; run a full backup first"
                .to_string(),
        )
    })?;

    // An incremental backup copies the log from the cursor on, so the chain
    // must reach the cursor: one that ends before it would miss what lies
    // between. A read-only full backup can lead the writer's cursor:
    // overlapping WAL groups are valid. The log position decides; the epoch
    // alone misses a change logged without an epoch of its own (a graph
    // command, a schema statement, the index API), and is all a manifest
    // written before 0.6.1 has.
    if let Some(last) = manifest.segments.last() {
        let behind = match last.end_wal_lsn {
            Some(chain_end) => chain_end < from,
            None => last.end_epoch < cursor.backed_up_epoch,
        };
        if behind {
            return Err(Error::InvalidValue(format!(
                "backup chain {} ends at epoch {}, but the database backup cursor is at epoch {} \
                 and further along the log; run a full backup in this directory before taking \
                 an incremental backup",
                backup_dir.display(),
                last.end_epoch.as_u64(),
                cursor.backed_up_epoch.as_u64(),
            )));
        }
    }

    sync_backup_entry(backup_dir, "backup:directory_sync")?;

    // Seal the active segment first: every group logged so far is then in a
    // segment before `sealed`, and groups logged from now on go to later
    // ones, which the next incremental backup reads. Reading the active
    // segment instead lost the groups appended while it was copied. A
    // segment without frames is not sealed, so a poll with nothing new
    // leaves no empty segment.
    let sealed = wal.rotate().map_err(|e| {
        Error::Internal(format!("failed to rotate WAL for incremental backup: {e}"))
    })?;
    let groups = groups_between(wal, from, sealed)?;
    if groups == 0 {
        return Err(Error::Internal(
            "no new WAL records since last backup".to_string(),
        ));
    }

    let listing = list_wal_directory(wal.dir()).map_err(Error::from)?;
    let segments: Vec<&std::path::PathBuf> = listing
        .segments
        .iter()
        .filter(|(first_lsn, _)| (from..sealed).contains(first_lsn))
        .map(|(_, path)| path)
        .collect();
    if listing
        .segments
        .iter()
        .all(|(first_lsn, _)| *first_lsn != from)
    {
        return Err(Error::Internal(format!(
            "the WAL segment at log position {from}, where the last backup ended, was removed"
        )));
    }

    let start_epoch = EpochId::new(cursor.backed_up_epoch.as_u64() + 1);
    let end_epoch = current_epoch;

    let mut output = Vec::new();
    write_backup_header(&mut output, start_epoch, end_epoch, groups);
    for path in segments {
        let bytes = std::fs::read(path).map_err(|e| {
            Error::Internal(format!(
                "failed to read WAL segment {}: {e}",
                path.display()
            ))
        })?;
        output.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        output.extend_from_slice(&bytes);
    }

    let segment_idx = manifest.segments.len();
    let filename = format!("backup_incr_{segment_idx:04}.wal");
    let path = backup_dir.join(&filename);
    let mut file = std::fs::File::create(&path).map_err(|e| {
        Error::Internal(format!(
            "failed to create incremental backup {}: {e}",
            path.display()
        ))
    })?;
    file.write_all(&output).map_err(|e| {
        Error::Internal(format!(
            "failed to write incremental backup {}: {e}",
            path.display()
        ))
    })?;
    grafeo_common::testing::crash::maybe_fail("backup:incremental_sync")?;
    file.sync_all().map_err(|e| {
        Error::Internal(format!(
            "failed to sync incremental backup {}: {e}",
            path.display()
        ))
    })?;
    drop(file);
    sync_backup_entry(&path, "backup:incremental_directory_sync")?;
    grafeo_common::testing::crash::maybe_crash("backup:after_incremental_file");

    let segment = BackupSegment {
        kind: BackupKind::Incremental,
        filename,
        start_epoch,
        end_epoch,
        checksum: crc32fast::hash(&output),
        size_bytes: output.len() as u64,
        created_at_ms: now_ms(),
        end_wal_lsn: Some(sealed),
    };

    let mut manifest = manifest;
    manifest.segments.push(segment.clone());
    write_prepared_manifest(backup_dir, &manifest)?;
    grafeo_common::testing::crash::maybe_crash("backup:after_incremental_manifest");

    write_backup_cursor(
        wal.dir(),
        &BackupCursor {
            backed_up_epoch: current_epoch,
            log_sequence: 0,
            wal_lsn: Some(sealed),
            timestamp_ms: now_ms(),
        },
    )?;
    grafeo_common::testing::crash::maybe_crash("backup:after_incremental_cursor");

    Ok(segment)
}

/// How many complete groups `wal` holds from `from` up to `until`, both
/// segment boundaries.
fn groups_between(wal: &Wal, from: u64, until: u64) -> Result<u64> {
    let mut scan = WalScan::open(wal.dir(), wal.scan_options(from)).map_err(Error::from)?;
    let mut groups = 0;
    while let Some(group) = scan.next_group().map_err(Error::from)? {
        if group.start_lsn() >= until {
            break;
        }
        groups += 1;
    }
    Ok(groups)
}

// ── Restore ────────────────────────────────────────────────────────

/// Restores a database to a specific epoch from a backup chain.
///
/// 1. Finds the most recent full backup with `end_epoch <= target_epoch`
///    and the incremental segments after it, and checks `keys` against the
///    full backup ([`restore_ciphers`]).
/// 2. Copies the full backup to `output_path`.
/// 3. Writes the WAL groups of the incremental segments up to `target_epoch`
///    as the sidecar WAL of the restored file, which its next open replays.
///    For an encrypted backup the segments are read, and that WAL written,
///    with the WAL key of the backup's database. A chain of the earlier WAL
///    format (0.5.x, 0.6.0-beta.1) is restored as that format, which the
///    open replays once and retires.
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
    let full_index = manifest
        .segments
        .iter()
        .rposition(|s| s.kind == BackupKind::Full && s.end_epoch <= target_epoch)
        .ok_or_else(|| {
            Error::Internal(format!(
                "no full backup covers epoch {}",
                target_epoch.as_u64()
            ))
        })?;
    let full = &manifest.segments[full_index];

    // The incremental segments taken after the full backup, up to the first
    // that starts past the target: the WAL from where the image's log
    // position is on. One can start at or before the full backup's epoch: a
    // read-only database's full backup cannot move the WAL's backup cursor,
    // so the writer's next incremental starts at its own last full backup.
    // Its groups up to the image's log position are in the image too, and
    // the replay starts at that position, past them. An incremental's epochs
    // are read while commits go on, so they select by order, not by range.
    let incrementals: Vec<&BackupSegment> = manifest.segments[full_index + 1..]
        .iter()
        .filter(|s| s.kind == BackupKind::Incremental && s.start_epoch <= target_epoch)
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

    let mut segments = Vec::with_capacity(incrementals.len());
    for incr in &incrementals {
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
        let version = backup_version(&incr_data)?;
        segments.push((incr.filename.as_str(), version, incr_data));
    }

    match segments.iter().map(|(_, version, _)| *version).max() {
        // That format's records replay by epoch, from the first: they would
        // apply again what the image holds.
        Some(FIRST_FORMAT_VERSION)
            if incrementals
                .iter()
                .any(|incremental| incremental.start_epoch <= full.end_epoch) =>
        {
            Err(Error::Internal(format!(
                "the incremental backups after the full backup {} start before its epoch {} in \
                 the WAL format before 0.6.0, which cannot replay from it",
                full.filename,
                full.end_epoch.as_u64()
            )))
        }
        Some(FIRST_FORMAT_VERSION) => {
            restore_first_format(&segments, target_epoch, output_path, ciphers)
        }
        _ if segments
            .iter()
            .any(|(_, version, _)| *version == FIRST_FORMAT_VERSION) =>
        {
            Err(Error::Internal(
                "the incremental backups after the full backup mix two WAL formats".to_string(),
            ))
        }
        _ => restore_segments(&segments, target_epoch, output_path, ciphers.segments),
    }
}

/// Writes the WAL groups of the version 2 incremental backups `segments`, up
/// to `target_epoch`, as the sidecar WAL of the restored file at
/// `output_path`, starting at its image's log position.
fn restore_segments(
    segments: &[(&str, u32, Vec<u8>)],
    target_epoch: EpochId,
    output_path: &Path,
    cipher_for_salt: Option<CipherForSalt>,
) -> Result<()> {
    let image = restored_header(output_path)?;
    let staging = sibling(output_path, "restore_wal");
    std::fs::create_dir_all(&staging)
        .map_err(|e| Error::Internal(format!("failed to create restore WAL directory: {e}")))?;
    for (filename, _, data) in segments {
        let mut rest = &data[BACKUP_HEADER_SIZE..];
        while !rest.is_empty() {
            let truncated =
                || Error::Internal(format!("incremental backup {filename} is truncated"));
            let (length, after) = rest.split_first_chunk::<8>().ok_or_else(truncated)?;
            let length = usize::try_from(u64::from_le_bytes(*length)).map_err(|_| truncated())?;
            let (bytes, after) = after.split_at_checked(length).ok_or_else(truncated)?;
            let header = SegmentHeader::decode(bytes, Path::new(filename)).map_err(Error::from)?;
            std::fs::write(staging.join(segment_file_name(header.first_lsn)), bytes).map_err(
                |e| Error::Internal(format!("failed to write WAL segment for restore: {e}")),
            )?;
            rest = after;
        }
    }

    let sidecar = std::path::PathBuf::from(format!("{}.wal", output_path.display()));
    copy_groups_until(
        &staging,
        &sidecar,
        ScanOptions {
            cipher_for_salt: cipher_for_salt.clone(),
            ..ScanOptions::new(image.database_id, image.checkpoint_lsn)
        },
        target_epoch,
        cipher_for_salt,
    )?;
    std::fs::remove_dir_all(&staging)
        .map_err(|e| Error::Internal(format!("failed to remove restore WAL directory: {e}")))
}

/// Copies the complete groups of the WAL in `from` (read with `options`) up
/// to the first one past `target_epoch` into a new WAL in `to`, which starts
/// at `options.from_lsn`.
fn copy_groups_until(
    from: &Path,
    to: &Path,
    options: ScanOptions,
    target_epoch: EpochId,
    cipher_for_salt: Option<CipherForSalt>,
) -> Result<()> {
    use grafeo_common::storage::{LogRecord, read_log_records};

    let write_failed = |e: grafeo_storage::wal::GroupError| {
        Error::Internal(format!("failed to write the restored WAL: {e}"))
    };
    let (database_id, start_lsn) = (options.database_id, options.from_lsn);
    let mut scan = WalScan::open(from, options).map_err(Error::from)?;
    std::fs::create_dir_all(to)
        .map_err(|e| Error::Internal(format!("failed to create the restored WAL: {e}")))?;
    let wal = Wal::open(
        to,
        WalOptions {
            durability: DurabilityMode::NoSync,
            cipher_for_salt,
            ..WalOptions::new(database_id, start_lsn)
        },
    )
    .map_err(Error::from)?;
    let mut copied_all = true;
    while let Some(mut group) = scan.next_group().map_err(Error::from)? {
        let mut payloads = Vec::new();
        while let Some(payload) = group.next_payload().map_err(Error::from)? {
            payloads.push(payload.to_vec());
        }
        let mut first = None;
        if let Some(payload) = payloads.first() {
            read_log_records(payload, &mut |record| {
                first.get_or_insert(record);
                Ok(())
            })?;
        }
        let Some(LogRecord::GroupBegin(begin)) = first else {
            return Err(Error::Internal(format!(
                "the backed-up group at log position {} does not start with its GroupBegin",
                group.start_lsn()
            )));
        };
        if begin.epoch > target_epoch {
            copied_all = false;
            break;
        }
        let mut writer = wal
            .begin_group(group.transaction_id())
            .map_err(write_failed)?;
        for payload in &payloads {
            writer.push(payload).map_err(write_failed)?;
        }
        writer.finish().map_err(write_failed)?;
    }
    if copied_all {
        let end = scan.finish().map_err(Error::from)?;
        if let Some(tail) = end.tail {
            return Err(Error::Internal(format!(
                "the backed-up WAL ends in a damaged group at log position {}: {}",
                tail.lsn, tail.reason
            )));
        }
    }
    wal.sync().map_err(Error::from)?;
    wal.close();
    Ok(())
}

/// The active database header of the restored file at `path`, with the id
/// of its database (from its file header).
struct RestoredImage {
    database_id: u128,
    checkpoint_lsn: u64,
}

/// Reads the [`RestoredImage`] of the file at `path`.
fn restored_header(path: &Path) -> Result<RestoredImage> {
    let mut pages = Vec::with_capacity(3 * PAGE_BYTES);
    std::fs::File::open(path)
        .and_then(|file| file.take(3 * PAGE_BYTES as u64).read_to_end(&mut pages))
        .map_err(|e| {
            Error::Internal(format!(
                "cannot read the restored file {}: {e}",
                path.display()
            ))
        })?;
    let file_header = FileHeaderV3::decode(&pages).map_err(|e| e.wrapped("restore"))?;
    Ok(RestoredImage {
        database_id: file_header.database_id,
        checkpoint_lsn: copied_header(&pages, path)?.checkpoint_lsn,
    })
}

/// The path `<output_path file name>.<suffix>` next to `output_path`.
fn sibling(output_path: &Path, suffix: &str) -> std::path::PathBuf {
    output_path.parent().unwrap_or(Path::new(".")).join(format!(
        "{}.{suffix}",
        output_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("db")
    ))
}

/// Restores the incremental backups `segments` of the WAL format of 0.5.x and
/// 0.6.0-beta.1: replays them up to `target_epoch` with that format's
/// recovery and writes the records as a sidecar WAL of that format, which
/// the restored file's open replays once and retires. Removed in 0.7.0.
fn restore_first_format(
    segments: &[(&str, u32, Vec<u8>)],
    target_epoch: EpochId,
    output_path: &Path,
    ciphers: RestoreCiphers,
) -> Result<()> {
    let wal_dir = sibling(output_path, "restore_wal");
    std::fs::create_dir_all(&wal_dir)
        .map_err(|e| Error::Internal(format!("failed to create restore WAL directory: {e}")))?;

    // Write the frames of each segment to a temp log file for recovery
    for (i, (_, _, incr_data)) in segments.iter().enumerate() {
        if incr_data.len() > BACKUP_HEADER_SIZE {
            let wal_frames = &incr_data[BACKUP_HEADER_SIZE..];
            let wal_file = wal_dir.join(format!("wal_{i:08}.log"));
            std::fs::write(&wal_file, wal_frames).map_err(|e| {
                Error::Internal(format!("failed to write WAL file for restore: {e}"))
            })?;
        }
    }

    // Recover WAL records up to target epoch, then write a trimmed WAL    // Recover WAL records up to target epoch, then write a trimmed WAL
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
    /// The ciphers of the WAL segments, for a chain of the current format.
    segments: Option<CipherForSalt>,
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
        segments: None,
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
                segments: keys.wal_segment_ciphers(database_id),
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
            end_wal_lsn: None,
        });
        manifest.segments.push(BackupSegment {
            kind: BackupKind::Incremental,
            filename: "backup_incr_0001.wal".to_string(),
            start_epoch: EpochId::new(101),
            end_epoch: EpochId::new(200),
            checksum: 67890,
            size_bytes: 256,
            created_at_ms: 2000,
            end_wal_lsn: None,
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
            end_wal_lsn: None,
        });
        manifest.segments.push(BackupSegment {
            kind: BackupKind::Incremental,
            filename: "backup_incr_0001.wal".to_string(),
            start_epoch: EpochId::new(20),
            end_epoch: EpochId::new(88),
            checksum: 3,
            size_bytes: 319,
            created_at_ms: 1_791_331_288_000,
            end_wal_lsn: None,
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
            wal_lsn: None,
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
