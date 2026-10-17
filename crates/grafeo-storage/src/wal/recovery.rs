//! WAL recovery.

use super::record::WalEntry;
use super::{CheckpointMetadata, WalCipher, WalManager, WalRecord};
use grafeo_common::utils::error::{Error, Result};
use grafeo_common::{grafeo_debug, grafeo_info, grafeo_warn};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
#[cfg(feature = "encryption")]
use std::sync::Arc;

/// Name of the checkpoint metadata file.
const CHECKPOINT_METADATA_FILE: &str = "checkpoint.meta";

/// Committed records from a recovery pass, and whether the log ends torn.
#[derive(Debug)]
pub struct RecoveredWal<R = WalRecord> {
    /// Records of committed transactions, plus checkpoint and metadata
    /// records, in log order.
    pub records: Vec<R>,
    /// Whether the log ends with records that no commit or abort marker
    /// closes, or with bytes that do not form a complete record. Such a tail
    /// must be sealed (see `LpgWal::seal_torn_tail`) before anything new is
    /// logged, otherwise the next commit marker would commit the torn records.
    pub torn_tail: bool,
    /// The damage a salvaging recovery ([`WalRecovery::with_salvage`])
    /// stopped at, a [`Error::Corruption`] naming the file and offset: the
    /// records are the commits before it, and the WAL holds more than they
    /// do. `None` when the recovery read the whole log.
    pub damage: Option<Error>,
}

/// Handles WAL recovery after a crash.
pub struct WalRecovery {
    /// Directory containing WAL files.
    dir: std::path::PathBuf,
    /// Encryptor for decrypting WAL records (None = unencrypted).
    #[cfg(feature = "encryption")]
    encryptor: Option<Arc<grafeo_common::encryption::PageEncryptor>>,
    /// Stop at damage, with the commits before it, instead of failing.
    salvage: bool,
}

impl WalRecovery {
    /// Creates a new recovery handler for the given WAL directory.
    pub fn new(dir: impl AsRef<Path>) -> Self {
        Self {
            dir: dir.as_ref().to_path_buf(),
            #[cfg(feature = "encryption")]
            encryptor: None,
            salvage: false,
        }
    }

    /// Creates a recovery handler for the given WAL directory that decrypts
    /// its records with `cipher`; `None` reads plaintext records.
    #[must_use]
    pub fn with_cipher(dir: impl AsRef<Path>, cipher: Option<WalCipher>) -> Self {
        #[cfg(not(feature = "encryption"))]
        if let Some(cipher) = cipher {
            match cipher {}
        }
        Self {
            dir: dir.as_ref().to_path_buf(),
            #[cfg(feature = "encryption")]
            encryptor: cipher.map(Arc::new),
            salvage: false,
        }
    }

    /// This recovery, salvaging when `salvage` is true: a damaged record
    /// ends the recovery with the commits before it (reported in
    /// [`RecoveredWal::damage`]) instead of failing it.
    #[must_use]
    pub fn with_salvage(mut self, salvage: bool) -> Self {
        self.salvage = salvage;
        self
    }

    /// Sets the encryptor for decrypting WAL records during recovery.
    #[cfg(feature = "encryption")]
    pub fn set_encryptor(&mut self, encryptor: grafeo_common::encryption::PageEncryptor) {
        self.encryptor = Some(Arc::new(encryptor));
    }

    /// Creates a recovery handler from a WAL manager, which decrypts with
    /// the manager's encryptor, if it has one.
    #[must_use]
    pub fn from_wal(wal: &WalManager) -> Self {
        Self {
            dir: wal.dir().to_path_buf(),
            #[cfg(feature = "encryption")]
            encryptor: wal.encryptor().cloned(),
            salvage: false,
        }
    }

    /// Reads checkpoint metadata if it exists.
    ///
    /// Returns `None` if no checkpoint metadata is found.
    ///
    /// # Errors
    ///
    /// Returns an error if the metadata file cannot be read or deserialized.
    pub fn read_checkpoint_metadata(&self) -> Result<Option<CheckpointMetadata>> {
        let metadata_path = self.dir.join(CHECKPOINT_METADATA_FILE);

        if !metadata_path.exists() {
            return Ok(None);
        }

        let file = File::open(&metadata_path)?;
        let mut reader = BufReader::new(file);
        let mut data = Vec::new();
        reader.read_to_end(&mut data)?;

        let (metadata, _): (CheckpointMetadata, _) =
            bincode::serde::decode_from_slice(&data, bincode::config::standard())
                .map_err(|e| Error::Serialization(e.to_string()))?;

        Ok(Some(metadata))
    }

    /// Returns the checkpoint metadata, if any.
    ///
    /// This is useful for determining whether to perform a full or
    /// incremental recovery.
    #[must_use]
    pub fn checkpoint(&self) -> Option<CheckpointMetadata> {
        self.read_checkpoint_metadata().ok().flatten()
    }

    /// Recovers committed records from all WAL files.
    ///
    /// Returns only records that were part of committed transactions.
    /// If checkpoint metadata exists, only replays files from the
    /// checkpoint sequence onwards.
    ///
    /// # Errors
    ///
    /// Returns an error if recovery fails.
    pub fn recover(&self) -> Result<Vec<WalRecord>> {
        self.recover_as::<WalRecord>()
    }

    /// Recovers committed records of a specific type from all WAL files.
    ///
    /// This is the generic version of [`recover`](Self::recover). Use it
    /// when recovering a WAL that stores a custom record type.
    ///
    /// # Errors
    ///
    /// Returns an error if recovery fails.
    pub fn recover_as<R: WalEntry>(&self) -> Result<Vec<R>> {
        let checkpoint = self.read_checkpoint_metadata()?;
        Ok(self.recover_internal_as::<R>(checkpoint)?.records)
    }

    /// Recovers committed records from every WAL file, ignoring checkpoint
    /// metadata.
    ///
    /// For WAL-directory databases, where the WAL is the only copy of the data
    /// and no snapshot covers the files a checkpoint would skip. An existing
    /// `checkpoint.meta` (written by older versions) is reported with a warning.
    ///
    /// # Errors
    ///
    /// Returns an error if recovery fails.
    pub fn recover_all(&self) -> Result<Vec<WalRecord>> {
        Ok(self.recover_all_with_tail()?.records)
    }

    /// Like [`recover_all`](Self::recover_all), and also reports whether the
    /// log ends with a torn tail.
    ///
    /// # Errors
    ///
    /// Returns an error if recovery fails.
    pub fn recover_all_with_tail(&self) -> Result<RecoveredWal> {
        if self.dir.join(CHECKPOINT_METADATA_FILE).exists() {
            grafeo_warn!(
                "Ignoring {:?}: the WAL is the only copy of the data, replaying every WAL file",
                self.dir.join(CHECKPOINT_METADATA_FILE)
            );
        }
        self.recover_internal_as::<WalRecord>(None)
    }

    /// Like [`recover`](Self::recover), and also reports whether the log
    /// ends with a torn tail.
    ///
    /// # Errors
    ///
    /// Returns an error if recovery fails.
    pub fn recover_with_tail(&self) -> Result<RecoveredWal> {
        let checkpoint = self.read_checkpoint_metadata()?;
        self.recover_internal_as::<WalRecord>(checkpoint)
    }

    /// Recovers committed records up to and including the given epoch.
    ///
    /// Returns only records belonging to transactions committed at or before
    /// `max_epoch`. Records from the first transaction committed after
    /// `max_epoch` are excluded.
    ///
    /// The WAL commit sequence is: `[data records] [TxCommit] [EpochAdvance]`.
    /// When an `EpochAdvance { epoch }` where `epoch > max_epoch` is seen, we
    /// discard the preceding transaction's records (everything since the last
    /// `EpochAdvance` that was within range).
    ///
    /// Used by point-in-time recovery to restore a database to a specific epoch.
    ///
    /// # Errors
    ///
    /// Returns an error if recovery fails.
    pub fn recover_until_epoch(
        &self,
        max_epoch: grafeo_common::types::EpochId,
    ) -> Result<Vec<WalRecord>> {
        let all_records = self.recover()?;
        let mut committed = Vec::new();
        let mut pending = Vec::new();

        for record in all_records {
            if let WalRecord::EpochAdvance { epoch } = &record {
                if *epoch > max_epoch {
                    // This epoch's transaction is beyond our target:
                    // discard the pending records and stop.
                    break;
                }
                // Epoch is within range: flush pending into committed
                committed.append(&mut pending);
                committed.push(record);
            } else {
                pending.push(record);
            }
        }

        // Any remaining pending records lack a confirming EpochAdvance
        // within range, so they belong to a transaction beyond max_epoch
        // (or an incomplete transaction) and are intentionally dropped.

        Ok(committed)
    }

    fn recover_internal_as<R: WalEntry>(
        &self,
        checkpoint: Option<CheckpointMetadata>,
    ) -> Result<RecoveredWal<R>> {
        let mut current_tx_records = Vec::new();
        let mut committed_records = Vec::new();
        // Whether the last file read ended exactly at a record boundary.
        let mut last_file_clean = true;

        // Get all log files in order
        let log_files = self.get_log_files()?;
        // Lets a test move the directory between the listing and the reads.
        grafeo_common::testing::pause::maybe_pause("wal_recovery:after_listing");

        // Determine the minimum sequence number to process
        let min_sequence = checkpoint.as_ref().map_or(0, |cp| cp.log_sequence);

        if checkpoint.is_some() {
            grafeo_info!(
                "Recovering from checkpoint at epoch {:?}, starting from log sequence {}",
                checkpoint.as_ref().map(|c| c.epoch),
                min_sequence
            );
        }

        // Read log files in sequence, skipping those before checkpoint
        let newest = log_files.len().saturating_sub(1);
        let mut damage = None;
        'files: for (index, log_file) in log_files.iter().enumerate() {
            // Extract sequence number from filename
            let sequence = Self::sequence_from_path(log_file).unwrap_or(0);

            // Skip files that are completely before the checkpoint
            // We include the checkpoint sequence file because it may contain
            // records after the checkpoint record itself
            if sequence < min_sequence {
                grafeo_debug!(
                    "Skipping log file {:?} (sequence {} < checkpoint {})",
                    log_file,
                    sequence,
                    min_sequence
                );
                continue;
            }

            let file = match File::open(log_file) {
                Ok(f) => f,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => return Err(e.into()),
            };
            let file_len = file.metadata()?.len();
            let mut reader = BufReader::new(file);
            let mut position = 0u64;
            last_file_clean = true;

            // Read all records from this file
            loop {
                match self.read_frame_as::<R>(&mut reader, file_len - position)? {
                    Frame::Record(record, frame_len) => {
                        position += frame_len;
                        if record.is_commit() {
                            committed_records.append(&mut current_tx_records);
                            committed_records.push(record);
                        } else if record.is_abort() {
                            current_tx_records.clear();
                        } else if record.is_checkpoint() || record.is_metadata() {
                            // Checkpoint and metadata records (e.g. EpochAdvance)
                            // are not part of any transaction: always include them.
                            committed_records.push(record);
                        } else {
                            current_tx_records.push(record);
                        }
                    }
                    Frame::End => break,
                    Frame::Bad { reason, kind } => {
                        // Only the end of the newest file is written while
                        // a crash can cut it: there, a frame that did not
                        // finish with no frame behind it is a torn tail. An
                        // older file ends torn only when the next one starts
                        // with the abort marker that sealed it (see
                        // `LpgWal::seal_torn_tail`, which 0.5.x and
                        // 0.6.0-beta.1 called at open). Anywhere else the log
                        // lost records that later commits build on.
                        if kind != BadFrame::Undecodable
                            && (index == newest
                                || self.starts_with_abort::<R>(&log_files[index + 1])?)
                            && !self.frames_follow::<R>(
                                log_file,
                                position,
                                kind == BadFrame::Unchecked,
                            )?
                        {
                            last_file_clean = false;
                            break;
                        }
                        let error =
                            Error::corruption_at(format!("damaged WAL record: {reason}"), position)
                                .in_file(log_file);
                        if !self.salvage {
                            return Err(error);
                        }
                        grafeo_warn!(
                            "salvaging the WAL: {error}; the commits after it are left out"
                        );
                        current_tx_records.clear();
                        damage = Some(error);
                        break 'files;
                    }
                }
            }
        }

        // Uncommitted records in current_tx_records are discarded
        let torn_tail = !current_tx_records.is_empty() || !last_file_clean;
        if torn_tail {
            grafeo_warn!(
                "WAL in {:?} ends with an incomplete transaction ({} records without a commit marker{})",
                self.dir,
                current_tx_records.len(),
                if last_file_clean {
                    ""
                } else {
                    ", followed by an incomplete record"
                }
            );
        }

        Ok(RecoveredWal {
            records: committed_records,
            torn_tail,
            damage,
        })
    }

    /// Extracts the sequence number from a WAL log file path.
    fn sequence_from_path(path: &Path) -> Option<u64> {
        path.file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.strip_prefix("wal_"))
            .and_then(|s| s.parse().ok())
    }

    /// The WAL files of the directory, sorted by sequence. A missing
    /// directory holds none; one that cannot be listed is an error, never
    /// "no files": recovering nothing would lose the records it holds.
    fn get_log_files(&self) -> Result<Vec<std::path::PathBuf>> {
        let cannot_list = |error: std::io::Error| {
            Error::Io(std::io::Error::new(
                error.kind(),
                format!(
                    "cannot list the WAL directory {}: {error}",
                    self.dir.display()
                ),
            ))
        };
        let entries = match std::fs::read_dir(&self.dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(cannot_list(error)),
        };

        let mut files = Vec::new();
        for entry in entries {
            let path = entry.map_err(cannot_list)?.path();
            if path.extension().is_some_and(|ext| ext == "log") {
                files.push(path);
            }
        }

        // Sort by filename (which includes sequence number)
        files.sort();

        Ok(files)
    }

    /// The bytes a frame carries after its payload: the CRC32 of a
    /// plaintext frame (an encrypted payload carries its own tag).
    fn checksum_len(&self) -> u64 {
        #[cfg(feature = "encryption")]
        if self.encryptor.is_some() {
            return 0;
        }
        4
    }

    /// Reads the frame at the reader's position, with `remaining` bytes left
    /// in the file. A length is checked against them before anything is
    /// allocated for it.
    ///
    /// # Errors
    ///
    /// An I/O error of the read.
    fn read_frame_as<R: WalEntry>(
        &self,
        reader: &mut BufReader<File>,
        remaining: u64,
    ) -> Result<Frame<R>> {
        if remaining == 0 {
            return Ok(Frame::End);
        }
        if remaining < 4 {
            return Ok(Frame::Bad {
                reason: "a length prefix cut off by the end of the file".to_string(),
                kind: BadFrame::CutOff,
            });
        }
        let mut len_buf = [0u8; 4];
        reader.read_exact(&mut len_buf)?;
        let len = u64::from(u32::from_le_bytes(len_buf));
        let frame_len = 4 + len + self.checksum_len();
        if frame_len > remaining {
            return Ok(Frame::Bad {
                reason: format!(
                    "a record of {len} bytes runs past the end of the file ({} bytes left)",
                    remaining - 4
                ),
                kind: BadFrame::CutOff,
            });
        }
        let payload_len = usize::try_from(frame_len - 4).map_err(|_| {
            Error::corruption(format!(
                "a WAL record of {len} bytes does not fit this platform"
            ))
        })?;
        let mut payload = vec![0u8; payload_len];
        reader.read_exact(&mut payload)?;
        Ok(match self.decode_frame::<R>(&payload) {
            Ok(record) => Frame::Record(record, frame_len),
            Err((reason, kind)) => Frame::Bad { reason, kind },
        })
    }

    /// Whether the first record of `file` is an abort marker.
    ///
    /// # Errors
    ///
    /// An I/O error of the read.
    fn starts_with_abort<R: WalEntry>(&self, file: &Path) -> Result<bool> {
        let file = File::open(file)?;
        let length = file.metadata()?.len();
        Ok(matches!(
            self.read_frame_as::<R>(&mut BufReader::new(file), length)?,
            Frame::Record(record, _) if record.is_abort()
        ))
    }

    /// The record of a frame's payload (the bytes after its length prefix).
    fn decode_frame<R: WalEntry>(
        &self,
        payload: &[u8],
    ) -> std::result::Result<R, (String, BadFrame)> {
        #[cfg(feature = "encryption")]
        if let Some(ref enc) = self.encryptor {
            let data = enc.decrypt(payload, b"grafeo-wal").map_err(|_| {
                (
                    "it does not decrypt: a wrong key, or damage".to_string(),
                    BadFrame::Unchecked,
                )
            })?;
            return decode_record(&data);
        }
        let Some((data, checksum)) = payload.split_last_chunk::<4>() else {
            return Err((
                "a frame shorter than its checksum".to_string(),
                BadFrame::Unchecked,
            ));
        };
        if crc32fast::hash(data) != u32::from_le_bytes(*checksum) {
            return Err(("checksum mismatch".to_string(), BadFrame::Unchecked));
        }
        decode_record(data)
    }

    /// Whether whole frames lie behind the bad frame at `position` of `file`,
    /// so that it is damage and not a torn tail: a frame whose record checks
    /// starts anywhere after its first byte (also inside it, when its length
    /// is what is damaged), or, for a bad frame that is `whole`, the lengths
    /// after it chain to the end of the file (frames that do not check, as
    /// with a wrong key). Zeros to the end of the file are a torn tail: a
    /// crash can leave a file longer than what was written to it.
    ///
    /// # Errors
    ///
    /// An I/O error of the read.
    fn frames_follow<R: WalEntry>(&self, file: &Path, position: u64, whole: bool) -> Result<bool> {
        let mut rest = Vec::new();
        let mut reader = File::open(file)?;
        reader.seek(SeekFrom::Start(position))?;
        reader.read_to_end(&mut rest)?;
        if rest.iter().all(|byte| *byte == 0) {
            return Ok(false);
        }
        let checksum_len = usize::try_from(self.checksum_len()).unwrap_or(usize::MAX);
        let frame_end = |at: usize| -> Option<usize> {
            let len = u32::from_le_bytes(rest.get(at..at.checked_add(4)?)?.try_into().ok()?);
            let end = at
                .checked_add(4)?
                .checked_add(usize::try_from(len).ok()?)?
                .checked_add(checksum_len)?;
            (end <= rest.len()).then_some(end)
        };
        // Checking a candidate hashes its payload: past this many bytes the
        // search stops and the frame counts as damage, which a salvaging
        // recovery can still stop at.
        let mut budget: usize = 256 * 1024 * 1024;
        for at in 1..rest.len() {
            let Some(end) = frame_end(at) else {
                continue;
            };
            let Some(left) = budget.checked_sub(end - at) else {
                return Ok(true);
            };
            budget = left;
            if self.decode_frame::<R>(&rest[at + 4..end]).is_ok() {
                return Ok(true);
            }
        }
        if whole && let Some(mut at) = frame_end(0) {
            let mut frames = 0_usize;
            while at < rest.len() {
                let Some(end) = frame_end(at) else {
                    return Ok(false);
                };
                at = end;
                frames += 1;
            }
            return Ok(frames > 0);
        }
        Ok(false)
    }
}

/// What reading one frame found.
enum Frame<R> {
    /// A record, with the length of its frame in bytes.
    Record(R, u64),
    /// The end of the file, at a frame boundary.
    End,
    /// A frame that does not hold a record.
    Bad {
        /// What is wrong with it.
        reason: String,
        /// How it is wrong.
        kind: BadFrame,
    },
}

/// How a frame fails to hold a record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BadFrame {
    /// The end of the file cuts it off: a write that did not finish, or a
    /// damaged length.
    CutOff,
    /// Whole, but its checksum or authentication fails, or it is zeros: a
    /// write that did not finish, or damage.
    Unchecked,
    /// Whole and checked, but no record of this format: written as it is,
    /// never torn.
    Undecodable,
}

/// The record `data` encodes. Zeros are a frame that was never written
/// (a crash can leave a file longer than its writes), not a record.
fn decode_record<R: WalEntry>(data: &[u8]) -> std::result::Result<R, (String, BadFrame)> {
    bincode::serde::decode_from_slice(data, bincode::config::standard())
        .map(|(record, _)| record)
        .map_err(|error| {
            if data.iter().all(|byte| *byte == 0) {
                ("zeros, not a record".to_string(), BadFrame::Unchecked)
            } else {
                (
                    format!("the record does not decode: {error}"),
                    BadFrame::Undecodable,
                )
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::types::{EpochId, NodeId, TransactionId};
    use tempfile::tempdir;

    #[test]
    fn test_recovery_committed() {
        let dir = tempdir().unwrap();

        // Write some records
        {
            let wal = WalManager::open(dir.path()).unwrap();

            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Person".to_string()],
            })
            .unwrap();

            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();

            wal.sync().unwrap();
        }

        // Recover
        let recovery = WalRecovery::new(dir.path());
        let records = recovery.recover().unwrap();

        assert_eq!(records.len(), 2);
    }

    #[test]
    fn test_recovery_uncommitted() {
        let dir = tempdir().unwrap();

        // Write some records without commit
        {
            let wal = WalManager::open(dir.path()).unwrap();

            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Person".to_string()],
            })
            .unwrap();

            // No commit!
            wal.sync().unwrap();
        }

        // Recover
        let recovery = WalRecovery::new(dir.path());
        let records = recovery.recover().unwrap();

        // Uncommitted records should be discarded
        assert_eq!(records.len(), 0);
    }

    /// Denies the listing of a directory to everyone until dropped, which
    /// allows it again, also when the test fails in between, so the temporary
    /// directory can be removed.
    struct Unlisted<'a>(&'a Path);

    impl<'a> Unlisted<'a> {
        fn deny(dir: &'a Path) -> Self {
            assert!(
                set_listing(dir, false),
                "cannot deny listing {}",
                dir.display()
            );
            Self(dir)
        }
    }

    impl Drop for Unlisted<'_> {
        /// Reports a failure instead of panicking: a panic in a drop while a
        /// failed test unwinds would abort the test binary.
        fn drop(&mut self) {
            if !set_listing(self.0, true) {
                eprintln!("cannot allow listing {} again", self.0.display());
            }
        }
    }

    /// Denies (`allowed` false) or allows again the listing of `dir` to
    /// everyone: `icacls` on Windows, the mode on Unix. Returns whether it
    /// worked.
    fn set_listing(dir: &Path, allowed: bool) -> bool {
        #[cfg(windows)]
        {
            let mut command = std::process::Command::new("icacls");
            command.arg(dir);
            if allowed {
                command.args(["/remove:d", "*S-1-1-0"]);
            } else {
                command.args(["/deny", "*S-1-1-0:(RD)"]);
            }
            command.status().is_ok_and(|status| status.success())
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = if allowed { 0o700 } else { 0o300 };
            std::fs::set_permissions(dir, std::fs::Permissions::from_mode(mode)).is_ok()
        }
    }

    /// A WAL directory that exists but cannot be listed fails the recovery,
    /// naming the directory, instead of recovering nothing: the records it
    /// holds would be lost to whatever the caller does next (a migration
    /// writes an empty image, an open replays nothing). A missing directory
    /// still recovers nothing.
    #[test]
    fn a_wal_directory_that_cannot_be_listed_fails_the_recovery() {
        let dir = tempdir().unwrap();
        let wal_dir = dir.path().join("wal");
        {
            let wal = WalManager::open(&wal_dir).unwrap();
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(3),
                labels: vec!["Person".to_string()],
            })
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(19),
            })
            .unwrap();
            wal.sync().unwrap();
        }

        let (listed, recovered) = {
            let _denied = Unlisted::deny(&wal_dir);
            (
                std::fs::read_dir(&wal_dir).map(|_| ()),
                WalRecovery::new(&wal_dir).recover_all_with_tail(),
            )
        };
        if listed.is_ok() {
            eprintln!("skipped: the WAL directory can still be listed (permissions do not apply)");
            return;
        }

        let error = recovered
            .expect_err("a WAL directory that cannot be listed is an error")
            .to_string();
        assert!(
            error.contains(&wal_dir.display().to_string()),
            "the error names the WAL directory: {error}"
        );
        assert_eq!(
            WalRecovery::new(&wal_dir).recover().unwrap().len(),
            2,
            "listed again, the WAL recovers its records"
        );
        assert!(
            WalRecovery::new(dir.path().join("missing"))
                .recover()
                .unwrap()
                .is_empty(),
            "a missing WAL directory recovers nothing"
        );
    }

    #[test]
    fn test_recovery_multiple_files() {
        let dir = tempdir().unwrap();

        // Write records across multiple files
        {
            let config = super::super::WalConfig {
                max_log_size: 100, // Force rotation
                ..Default::default()
            };
            let wal = WalManager::with_config(dir.path(), config).unwrap();

            // First transaction
            for i in 0..5 {
                wal.log(&WalRecord::CreateNode {
                    id: NodeId::new(i),
                    labels: vec!["Test".to_string()],
                })
                .unwrap();
            }
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();

            // Second transaction
            for i in 5..10 {
                wal.log(&WalRecord::CreateNode {
                    id: NodeId::new(i),
                    labels: vec!["Test".to_string()],
                })
                .unwrap();
            }
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(2),
            })
            .unwrap();

            wal.sync().unwrap();
        }

        // Recover
        let recovery = WalRecovery::new(dir.path());
        let records = recovery.recover().unwrap();

        // Should have 10 CreateNode + 2 TransactionCommit
        assert_eq!(records.len(), 12);
    }

    #[test]
    fn test_checkpoint_metadata() {
        use grafeo_common::types::EpochId;

        let dir = tempdir().unwrap();

        // Write records and create a checkpoint
        {
            let wal = WalManager::open(dir.path()).unwrap();

            // First transaction
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Test".to_string()],
            })
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();

            // Create checkpoint
            wal.checkpoint(TransactionId::new(1), EpochId::new(10))
                .unwrap();

            // Second transaction after checkpoint
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(2),
                labels: vec!["Test".to_string()],
            })
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(2),
            })
            .unwrap();

            wal.sync().unwrap();
        }

        // Verify checkpoint metadata was written
        let recovery = WalRecovery::new(dir.path());
        let checkpoint = recovery.checkpoint();
        assert!(checkpoint.is_some(), "Checkpoint metadata should exist");

        let cp = checkpoint.unwrap();
        assert_eq!(cp.epoch.as_u64(), 10);
        assert_eq!(cp.transaction_id.as_u64(), 1);
    }

    #[test]
    fn test_recovery_from_checkpoint() {
        use super::super::WalConfig;
        use grafeo_common::types::EpochId;

        let dir = tempdir().unwrap();

        // Write records across multiple log files with checkpoint
        {
            let config = WalConfig {
                max_log_size: 100, // Force rotation
                ..Default::default()
            };
            let wal = WalManager::with_config(dir.path(), config).unwrap();

            // First batch of records (should end up in early log files)
            for i in 0..5 {
                wal.log(&WalRecord::CreateNode {
                    id: NodeId::new(i),
                    labels: vec!["Before".to_string()],
                })
                .unwrap();
            }
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();

            // Create checkpoint
            wal.checkpoint(TransactionId::new(1), EpochId::new(100))
                .unwrap();

            // Second batch after checkpoint
            for i in 100..103 {
                wal.log(&WalRecord::CreateNode {
                    id: NodeId::new(i),
                    labels: vec!["After".to_string()],
                })
                .unwrap();
            }
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(2),
            })
            .unwrap();

            wal.sync().unwrap();
        }

        // Recovery should use checkpoint metadata to skip old files
        let recovery = WalRecovery::new(dir.path());
        let records = recovery.recover().unwrap();

        // We should get all committed records (checkpoint metadata is used for optimization)
        // The number depends on how many log files were skipped
        assert!(!records.is_empty(), "Should recover some records");
    }

    #[test]
    fn test_recover_all_ignores_checkpoint_metadata() {
        use super::super::WalConfig;
        use grafeo_common::types::EpochId;

        let dir = tempdir().unwrap();
        {
            let config = WalConfig {
                max_log_size: 100, // Force rotation
                ..Default::default()
            };
            let wal = WalManager::with_config(dir.path(), config).unwrap();
            for i in 0..5 {
                wal.log(&WalRecord::CreateNode {
                    id: NodeId::new(i),
                    labels: vec!["Before".to_string()],
                })
                .unwrap();
                wal.log(&WalRecord::TransactionCommit {
                    transaction_id: TransactionId::new(i + 1),
                })
                .unwrap();
            }
            // Epoch 0 keeps all files on disk; the metadata still points at
            // the latest sequence.
            wal.checkpoint(TransactionId::new(5), EpochId::new(0))
                .unwrap();
        }

        let recovery = WalRecovery::new(dir.path());
        let node_ids = |records: &[WalRecord]| -> Vec<u64> {
            records
                .iter()
                .filter_map(|r| match r {
                    WalRecord::CreateNode { id, .. } => Some(id.as_u64()),
                    _ => None,
                })
                .collect()
        };
        assert!(
            node_ids(&recovery.recover().unwrap()).len() < 5,
            "checkpoint-aware recovery skips the rotated files"
        );
        assert_eq!(
            node_ids(&recovery.recover_all().unwrap()),
            vec![0, 1, 2, 3, 4]
        );
    }

    fn create(id: u64) -> WalRecord {
        WalRecord::CreateNode {
            id: NodeId::new(id),
            labels: vec!["N".to_string()],
        }
    }

    fn commit(tx: u64) -> WalRecord {
        WalRecord::TransactionCommit {
            transaction_id: TransactionId::new(tx),
        }
    }

    fn created_ids(records: &[WalRecord]) -> Vec<u64> {
        records
            .iter()
            .filter_map(|r| match r {
                WalRecord::CreateNode { id, .. } => Some(id.as_u64()),
                _ => None,
            })
            .collect()
    }

    fn only_log_file(dir: &Path) -> std::path::PathBuf {
        let mut files: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "log"))
            .collect();
        assert_eq!(files.len(), 1);
        files.pop().unwrap()
    }

    #[test]
    fn test_torn_tail_detection() {
        // Every committed: clean.
        let dir = tempdir().unwrap();
        {
            let wal = WalManager::open(dir.path()).unwrap();
            wal.log_batch(&[create(1), commit(1)]).unwrap();
        }
        let recovered = WalRecovery::new(dir.path()).recover_with_tail().unwrap();
        assert!(!recovered.torn_tail);
        assert_eq!(created_ids(&recovered.records), vec![1]);

        // Records without a closing marker at the end: torn.
        {
            let wal = WalManager::open(dir.path()).unwrap();
            wal.log(&create(2)).unwrap();
        }
        let recovered = WalRecovery::new(dir.path()).recover_with_tail().unwrap();
        assert!(recovered.torn_tail);
        assert_eq!(created_ids(&recovered.records), vec![1]);

        // An abort closes them: clean again.
        {
            let wal = WalManager::open(dir.path()).unwrap();
            wal.log(&WalRecord::TransactionAbort {
                transaction_id: TransactionId::new(2),
            })
            .unwrap();
        }
        let recovered = WalRecovery::new(dir.path())
            .recover_all_with_tail()
            .unwrap();
        assert!(!recovered.torn_tail);
        assert_eq!(created_ids(&recovered.records), vec![1]);
    }

    #[test]
    fn test_partial_frame_is_a_torn_tail() {
        // A crash mid-write leaves bytes that are not a complete record, even
        // when every complete record is committed.
        for garbage in [&[7u8, 0][..], &[40, 0, 0, 0, 1, 2, 3][..]] {
            let dir = tempdir().unwrap();
            {
                let wal = WalManager::open(dir.path()).unwrap();
                wal.log_batch(&[create(1), commit(1)]).unwrap();
            }
            use std::io::Write;
            std::fs::OpenOptions::new()
                .append(true)
                .open(only_log_file(dir.path()))
                .unwrap()
                .write_all(garbage)
                .unwrap();

            let recovered = WalRecovery::new(dir.path()).recover_with_tail().unwrap();
            assert!(recovered.torn_tail, "garbage {garbage:?}");
            assert_eq!(created_ids(&recovered.records), vec![1]);
        }
    }

    #[test]
    fn test_sealed_torn_tail_is_not_committed_later() {
        use super::super::LpgWal;

        let dir = tempdir().unwrap();
        {
            let wal = LpgWal::open(dir.path()).unwrap();
            wal.log_batch(&[create(1), commit(1)]).unwrap();
            wal.log(&create(2)).unwrap(); // crash before the commit marker
        }

        let recovered = WalRecovery::new(dir.path()).recover_with_tail().unwrap();
        assert!(recovered.torn_tail);
        {
            let wal = LpgWal::open(dir.path()).unwrap();
            wal.seal_torn_tail().unwrap();
            wal.log_batch(&[create(3), commit(3)]).unwrap();
        }

        let recovered = WalRecovery::new(dir.path()).recover_with_tail().unwrap();
        assert!(!recovered.torn_tail);
        assert_eq!(created_ids(&recovered.records), vec![1, 3]);
    }

    #[test]
    fn test_seal_after_partial_frame_keeps_new_records_readable() {
        use super::super::LpgWal;
        use std::io::Write;

        let dir = tempdir().unwrap();
        {
            let wal = LpgWal::open(dir.path()).unwrap();
            wal.log_batch(&[create(1), commit(1)]).unwrap();
        }
        std::fs::OpenOptions::new()
            .append(true)
            .open(only_log_file(dir.path()))
            .unwrap()
            .write_all(&[40, 0, 0, 0, 1, 2, 3])
            .unwrap();

        assert!(
            WalRecovery::new(dir.path())
                .recover_with_tail()
                .unwrap()
                .torn_tail
        );
        {
            // Records appended after the partial frame in the same file would
            // be unreadable: sealing moves on to a new file.
            let wal = LpgWal::open(dir.path()).unwrap();
            wal.seal_torn_tail().unwrap();
            wal.log_batch(&[create(2), commit(2)]).unwrap();
        }

        let recovered = WalRecovery::new(dir.path()).recover_with_tail().unwrap();
        assert_eq!(created_ids(&recovered.records), vec![1, 2]);
    }

    #[test]
    fn test_recover_as_generic() {
        let dir = tempdir().unwrap();

        // Write records using WalManager
        {
            let wal = WalManager::open(dir.path()).unwrap();

            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Person".to_string()],
            })
            .unwrap();

            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();

            wal.sync().unwrap();
        }

        // Recover using the generic method
        let recovery = WalRecovery::new(dir.path());
        let records: Vec<WalRecord> = recovery.recover_as().unwrap();

        assert_eq!(records.len(), 2);

        // Verify the records are correct via WalEntry trait methods
        assert!(!records[0].is_commit());
        assert!(records[1].is_commit());
    }

    #[test]
    fn test_recovery_truncated_wal_mid_record() {
        let dir = tempdir().unwrap();

        // Write valid records first
        {
            let wal = WalManager::open(dir.path()).unwrap();
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Person".to_string()],
            })
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();
            wal.sync().unwrap();
        }

        // Find the WAL file and append a truncated record (length prefix only, no data)
        let wal_files: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| {
                let e = e.ok()?;
                if e.path().extension().is_some_and(|ext| ext == "log") {
                    Some(e.path())
                } else {
                    None
                }
            })
            .collect();
        assert!(!wal_files.is_empty(), "wal_files is empty");

        // Append a partial record: just a length prefix, then truncate
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&wal_files[0])
            .unwrap();
        f.write_all(&100u32.to_le_bytes()).unwrap(); // length=100 but no data follows

        // Recovery should still return the committed records (best-effort)
        let recovery = WalRecovery::new(dir.path());
        let records = recovery.recover().unwrap();
        assert_eq!(
            records.len(),
            2,
            "committed records before truncation should be recovered"
        );
    }

    #[test]
    fn test_recovery_corrupted_checksum() {
        let dir = tempdir().unwrap();

        // Write valid records
        {
            let wal = WalManager::open(dir.path()).unwrap();
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(1),
                labels: vec!["First".to_string()],
            })
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();
            wal.sync().unwrap();
        }

        // Find the WAL file and corrupt a byte in the data section
        let wal_files: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| {
                let e = e.ok()?;
                if e.path().extension().is_some_and(|ext| ext == "log") {
                    Some(e.path())
                } else {
                    None
                }
            })
            .collect();
        assert!(!wal_files.is_empty(), "wal_files is empty");

        let mut data = std::fs::read(&wal_files[0]).unwrap();
        // Flip a byte in the middle of the data (after the 4-byte length prefix)
        if data.len() > 8 {
            data[6] ^= 0xFF;
        }
        std::fs::write(&wal_files[0], &data).unwrap();

        // The commit marker after it is whole: the record is damage, not a
        // torn tail, and the recovery fails (#425).
        let recovery = WalRecovery::new(dir.path());
        let result = recovery.recover();
        assert!(
            matches!(result, Err(Error::Corruption(_))),
            "a damaged record fails the recovery: {result:?}"
        );
    }

    #[test]
    fn test_recovery_empty_wal_file() {
        let dir = tempdir().unwrap();

        // Create an empty WAL file
        std::fs::write(dir.path().join("wal_00000000.log"), []).unwrap();

        let recovery = WalRecovery::new(dir.path());
        let records = recovery.recover().unwrap();
        assert_eq!(records.len(), 0, "empty WAL should produce no records");
    }

    // ── recover_until_epoch tests ─────────────────────────────────────

    /// Helper: writes a committed transaction with an EpochAdvance marker,
    /// matching the pattern used by the engine session commit path.
    fn write_tx_with_epoch(wal: &WalManager, node_id: u64, tx_id: u64, epoch: u64) {
        wal.log(&WalRecord::CreateNode {
            id: NodeId::new(node_id),
            labels: vec![format!("N{node_id}")],
        })
        .unwrap();
        wal.log(&WalRecord::TransactionCommit {
            transaction_id: TransactionId::new(tx_id),
        })
        .unwrap();
        wal.log(&WalRecord::EpochAdvance {
            epoch: EpochId::new(epoch),
        })
        .unwrap();
    }

    #[test]
    fn test_recover_until_epoch_includes_target() {
        let dir = tempdir().unwrap();
        {
            let wal = WalManager::open(dir.path()).unwrap();
            write_tx_with_epoch(&wal, 1, 1, 5);
            write_tx_with_epoch(&wal, 2, 2, 10);
            write_tx_with_epoch(&wal, 3, 3, 15);
            wal.sync().unwrap();
        }

        let recovery = WalRecovery::new(dir.path());
        let records = recovery.recover_until_epoch(EpochId::new(10)).unwrap();

        // Should include tx1 (epoch 5) and tx2 (epoch 10), each producing
        // CreateNode + TxCommit + EpochAdvance = 3 records per tx.
        assert_eq!(records.len(), 6, "should include epochs 5 and 10");

        // Verify no records from epoch 15
        assert!(
            !records.iter().any(
                |r| matches!(r, WalRecord::EpochAdvance { epoch } if *epoch > EpochId::new(10))
            ),
            "should not include EpochAdvance beyond target"
        );
    }

    #[test]
    fn test_recover_until_epoch_excludes_next_epoch() {
        let dir = tempdir().unwrap();
        {
            let wal = WalManager::open(dir.path()).unwrap();
            write_tx_with_epoch(&wal, 1, 1, 5);
            write_tx_with_epoch(&wal, 2, 2, 6);
            wal.sync().unwrap();
        }

        let recovery = WalRecovery::new(dir.path());
        let records = recovery.recover_until_epoch(EpochId::new(5)).unwrap();

        // Only tx1 (epoch 5): CreateNode + TxCommit + EpochAdvance = 3
        assert_eq!(records.len(), 3, "should exclude epoch 6 transaction");

        // Verify the CreateNode is for node 1 only
        let nodes: Vec<_> = records
            .iter()
            .filter_map(|r| match r {
                WalRecord::CreateNode { id, .. } => Some(id.as_u64()),
                _ => None,
            })
            .collect();
        assert_eq!(nodes, vec![1], "only node from epoch 5 should be present");
    }

    #[test]
    fn test_recover_until_epoch_zero_returns_empty() {
        let dir = tempdir().unwrap();
        {
            let wal = WalManager::open(dir.path()).unwrap();
            write_tx_with_epoch(&wal, 1, 1, 5);
            wal.sync().unwrap();
        }

        let recovery = WalRecovery::new(dir.path());
        let records = recovery.recover_until_epoch(EpochId::new(0)).unwrap();

        // Epoch 5 > 0, so all records should be excluded
        assert!(
            records.is_empty(),
            "no records should be at or below epoch 0"
        );
    }

    #[test]
    fn test_recover_until_epoch_beyond_max_returns_all() {
        let dir = tempdir().unwrap();
        {
            let wal = WalManager::open(dir.path()).unwrap();
            write_tx_with_epoch(&wal, 1, 1, 5);
            write_tx_with_epoch(&wal, 2, 2, 10);
            wal.sync().unwrap();
        }

        let recovery = WalRecovery::new(dir.path());
        let records = recovery.recover_until_epoch(EpochId::new(999)).unwrap();

        // Both transactions are within range: 6 records total
        assert_eq!(records.len(), 6, "all records should be included");
    }

    #[test]
    fn test_recover_until_epoch_empty_wal() {
        let dir = tempdir().unwrap();
        std::fs::create_dir_all(dir.path()).unwrap();

        let recovery = WalRecovery::new(dir.path());
        let records = recovery.recover_until_epoch(EpochId::new(100)).unwrap();
        assert!(records.is_empty());
    }

    // ── Damage and torn tails (#425) ───────────────────────────────

    /// Writes `transactions` committed transactions of one `:Person` node
    /// each (node and transaction ids from 1), starting a new log file after
    /// each when `rotate` is set, and returns the log files in order.
    fn write_transactions(dir: &Path, transactions: u64, rotate: bool) -> Vec<std::path::PathBuf> {
        let wal = WalManager::open(dir).unwrap();
        for id in 1..=transactions {
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(id),
                labels: vec!["Person".to_string()],
            })
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(id),
            })
            .unwrap();
            if rotate && id < transactions {
                wal.rotate().unwrap();
            }
        }
        wal.sync().unwrap();
        drop(wal);
        let mut files: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "log"))
            .collect();
        files.sort();
        files
    }

    /// The offset of every plaintext frame of `bytes`.
    fn frame_offsets(bytes: &[u8]) -> Vec<usize> {
        let mut offsets = Vec::new();
        let mut at = 0;
        while at < bytes.len() {
            offsets.push(at);
            let len = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
            at += 4 + usize::try_from(len).unwrap() + 4;
        }
        offsets
    }

    /// The ids of the nodes `records` create.
    fn created(records: &[WalRecord]) -> Vec<u64> {
        records
            .iter()
            .filter_map(|record| match record {
                WalRecord::CreateNode { id, .. } => Some(id.as_u64()),
                _ => None,
            })
            .collect()
    }

    /// The corruption `result` fails with: its file and offset.
    fn corruption_of<T: std::fmt::Debug>(result: Result<T>) -> (std::path::PathBuf, u64) {
        match result {
            Err(Error::Corruption(corruption)) => {
                (corruption.file.unwrap(), corruption.offset.unwrap())
            }
            other => panic!("expected a corruption, got {other:?}"),
        }
    }

    /// The bug of #425: a damaged record stopped the reading of its file
    /// with a warning, so the commits after it were lost, and the recovery
    /// went on with the next file, on top of the missing ones.
    #[test]
    fn a_damaged_record_with_commits_after_it_fails_the_recovery() {
        let dir = tempdir().unwrap();
        let files = write_transactions(dir.path(), 3, false);
        let mut bytes = std::fs::read(&files[0]).unwrap();
        let second = frame_offsets(&bytes)[2];
        bytes[second + 5] ^= 0x01;
        std::fs::write(&files[0], &bytes).unwrap();

        let (file, offset) = corruption_of(WalRecovery::new(dir.path()).recover());
        assert_eq!((file, offset), (files[0].clone(), second as u64));
    }

    /// A damaged length makes a record run past the end of the file, as a
    /// torn write does: the whole frames behind it make it damage.
    #[test]
    fn a_damaged_length_with_commits_after_it_fails_the_recovery() {
        let dir = tempdir().unwrap();
        let files = write_transactions(dir.path(), 2, false);
        let mut bytes = std::fs::read(&files[0]).unwrap();
        let commit = frame_offsets(&bytes)[1];
        bytes[commit..commit + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        std::fs::write(&files[0], &bytes).unwrap();

        let (file, offset) = corruption_of(WalRecovery::new(dir.path()).recover());
        assert_eq!((file, offset), (files[0].clone(), commit as u64));
    }

    /// Only the newest file can end torn: the end of an older one is
    /// damage, also when nothing follows it in that file.
    #[test]
    fn a_damaged_end_of_an_older_file_fails_the_recovery() {
        let dir = tempdir().unwrap();
        let files = write_transactions(dir.path(), 2, true);
        assert_eq!(files.len(), 2);
        let mut bytes = std::fs::read(&files[0]).unwrap();
        let commit = frame_offsets(&bytes)[1];
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;
        std::fs::write(&files[0], &bytes).unwrap();

        let (file, offset) = corruption_of(WalRecovery::new(dir.path()).recover());
        assert_eq!((file, offset), (files[0].clone(), commit as u64));
    }

    /// What a crash can leave at the end of the newest file is a torn tail:
    /// the recovery keeps every commit before it and reports it.
    #[test]
    fn a_torn_tail_keeps_every_commit_before_it() {
        type Tear = fn(&mut Vec<u8>);
        let cases: [(&str, Tear, Vec<u64>); 5] = [
            (
                "zeros past the last write",
                |bytes| bytes.extend([0; 64]),
                vec![1, 2],
            ),
            (
                "a length prefix cut off",
                |bytes| bytes.extend(&7u32.to_le_bytes()[..3]),
                vec![1, 2],
            ),
            (
                "a huge length with a few bytes",
                |bytes| {
                    bytes.extend(u32::MAX.to_le_bytes());
                    bytes.extend([3, 19, 88]);
                },
                vec![1, 2],
            ),
            (
                "the last frame cut in half",
                |bytes| {
                    let last = *frame_offsets(bytes).last().unwrap();
                    bytes.truncate(last + 6);
                },
                vec![1],
            ),
            (
                "the last frame's checksum wrong",
                |bytes| {
                    let last = bytes.len() - 1;
                    bytes[last] ^= 0x01;
                },
                vec![1],
            ),
        ];
        for (case, tear, kept) in cases {
            let dir = tempdir().unwrap();
            let files = write_transactions(dir.path(), 2, false);
            let mut bytes = std::fs::read(&files[0]).unwrap();
            tear(&mut bytes);
            std::fs::write(&files[0], &bytes).unwrap();

            let recovered = WalRecovery::new(dir.path())
                .recover_with_tail()
                .unwrap_or_else(|error| panic!("{case}: {error}"));
            assert_eq!(created(&recovered.records), kept, "{case}");
            assert!(recovered.torn_tail, "{case}");
            assert!(recovered.damage.is_none(), "{case}");
        }
    }

    /// A whole frame whose checksum is right but whose bytes are no record
    /// was written as it is: damage, also at the end of the newest file.
    #[test]
    fn an_undecodable_record_is_damage_also_at_the_end() {
        let dir = tempdir().unwrap();
        let files = write_transactions(dir.path(), 1, false);
        let mut bytes = std::fs::read(&files[0]).unwrap();
        let end = bytes.len() as u64;
        let garbage = [0xFF_u8; 3];
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(garbage);
        bytes.extend(crc32fast::hash(&garbage).to_le_bytes());
        std::fs::write(&files[0], &bytes).unwrap();

        let (file, offset) = corruption_of(WalRecovery::new(dir.path()).recover());
        assert_eq!((file, offset), (files[0].clone(), end));
    }

    /// A salvaging recovery stops at the damage with the commits before it
    /// and names the damage, instead of failing.
    #[test]
    fn a_salvaging_recovery_stops_at_the_last_commit_before_the_damage() {
        let dir = tempdir().unwrap();
        let files = write_transactions(dir.path(), 3, true);
        let mut bytes = std::fs::read(&files[1]).unwrap();
        bytes[5] ^= 0x01;
        std::fs::write(&files[1], &bytes).unwrap();

        let recovered = WalRecovery::new(dir.path())
            .with_salvage(true)
            .recover_with_tail()
            .unwrap();
        assert_eq!(created(&recovered.records), vec![1]);
        let (file, offset) = corruption_of::<()>(Err(recovered.damage.unwrap()));
        assert_eq!((file, offset), (files[1].clone(), 0));
        // Without salvage, the same log fails.
        assert!(WalRecovery::new(dir.path()).recover().is_err());
    }
}

/// Crash injection tests for WAL recovery.
///
/// These tests verify that WAL recovery produces a consistent state after
/// simulated crashes at every crash point in the write path. The three crash
/// points are:
/// - `wal_before_write`: before writing length prefix + data + checksum
/// - `wal_after_write`: after writing data but before durability handling
/// - `wal_before_flush`: before fsync on TransactionCommit in Sync mode
///
/// Run with:
/// ```bash
/// cargo test -p grafeo-adapters --features "wal,testing-crash-injection" -- crash
/// ```
#[cfg(all(test, feature = "testing-crash-injection"))]
mod crash_tests {
    use super::*;
    use grafeo_common::testing::crash::{CrashResult, with_crash_at};
    use grafeo_common::types::{EpochId, NodeId, TransactionId, Value};
    use tempfile::tempdir;

    /// Helper: Sync durability config so all three crash points are reachable.
    fn sync_config() -> super::super::WalConfig {
        super::super::WalConfig {
            durability: super::super::DurabilityMode::Sync,
            ..Default::default()
        }
    }

    /// Crash at `wal_before_write`: no record bytes reach disk.
    /// Recovery should only return previously committed data.
    #[test]
    fn test_crash_before_write_discards_record() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_path_buf();

        // Seed one committed transaction
        {
            let wal = WalManager::with_config(&path, sync_config()).unwrap();
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Committed".into()],
            })
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();
        }

        // Crash at the first crash point (wal_before_write)
        let p = path.clone();
        let result = with_crash_at(1, move || {
            let wal = WalManager::with_config(&p, sync_config()).unwrap();
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(2),
                labels: vec!["Lost".into()],
            })
            .unwrap();
        });
        assert!(matches!(result, CrashResult::Crashed));

        // Only the first committed tx should survive
        let recovery = WalRecovery::new(&path);
        let records = recovery.recover().unwrap();
        assert_eq!(records.len(), 2, "CreateNode(1) + TransactionCommit(1)");
    }

    /// Crash at `wal_after_write`: data may be in BufWriter but no commit
    /// marker. Recovery should discard the uncommitted record.
    #[test]
    fn test_crash_after_write_uncommitted_discarded() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_path_buf();

        // For a non-commit record the crash points are:
        //   1 = wal_before_write, 2 = wal_after_write
        let p = path.clone();
        let result = with_crash_at(2, move || {
            let wal = WalManager::with_config(&p, sync_config()).unwrap();
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Partial".into()],
            })
            .unwrap();
        });
        assert!(matches!(result, CrashResult::Crashed));

        // No committed tx ⇒ recovery returns nothing
        let recovery = WalRecovery::new(&path);
        let records = recovery.recover().unwrap();
        assert_eq!(records.len(), 0, "Uncommitted records must be discarded");
    }

    /// Two committed transactions, then crash during the third.
    /// Recovery should preserve exactly the first two.
    #[test]
    fn test_crash_preserves_prior_committed_transactions() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_path_buf();

        // Commit two transactions
        {
            let wal = WalManager::with_config(&path, sync_config()).unwrap();
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(1),
                labels: vec!["T1".into()],
            })
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(2),
                labels: vec!["T2".into()],
            })
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(2),
            })
            .unwrap();
        }

        // Third transaction crashes immediately
        let p = path.clone();
        let result = with_crash_at(1, move || {
            let wal = WalManager::with_config(&p, sync_config()).unwrap();
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(3),
                labels: vec!["T3".into()],
            })
            .unwrap();
        });
        assert!(matches!(result, CrashResult::Crashed));

        // Both committed txs intact, third discarded
        let recovery = WalRecovery::new(&path);
        let records = recovery.recover().unwrap();
        assert_eq!(records.len(), 4, "2 CreateNode + 2 TransactionCommit");
    }

    /// Crash during checkpoint: committed data must still be recoverable.
    #[test]
    fn test_crash_during_checkpoint_preserves_data() {
        for crash_at in 1..15 {
            let dir = tempdir().unwrap();
            let path = dir.path().to_path_buf();

            // Seed committed data
            {
                let wal = WalManager::with_config(&path, sync_config()).unwrap();
                wal.log(&WalRecord::CreateNode {
                    id: NodeId::new(1),
                    labels: vec!["A".into()],
                })
                .unwrap();
                wal.log(&WalRecord::TransactionCommit {
                    transaction_id: TransactionId::new(1),
                })
                .unwrap();
            }

            // Crash during checkpoint
            let p = path.clone();
            let _result = with_crash_at(crash_at, move || {
                let wal = WalManager::with_config(&p, sync_config()).unwrap();
                wal.checkpoint(TransactionId::new(1), EpochId::new(10))
                    .unwrap();
            });

            // Committed data must survive regardless of checkpoint outcome
            let recovery = WalRecovery::new(&path);
            let records = recovery.recover().unwrap();
            assert!(
                !records.is_empty(),
                "crash_at={crash_at}: committed data must survive checkpoint crash"
            );
        }
    }

    /// Crash with rotated log files: recovery should span all files.
    #[test]
    fn test_crash_with_log_rotation() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_path_buf();

        // Write enough to trigger rotation
        {
            let config = super::super::WalConfig {
                durability: super::super::DurabilityMode::Sync,
                max_log_size: 100, // force rotation
                ..Default::default()
            };
            let wal = WalManager::with_config(&path, config).unwrap();
            for i in 0..5 {
                wal.log(&WalRecord::CreateNode {
                    id: NodeId::new(i),
                    labels: vec!["Rotated".into()],
                })
                .unwrap();
            }
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();
        }

        // Crash during additional write
        let p = path.clone();
        let result = with_crash_at(1, move || {
            let config = super::super::WalConfig {
                durability: super::super::DurabilityMode::Sync,
                max_log_size: 100,
                ..Default::default()
            };
            let wal = WalManager::with_config(&p, config).unwrap();
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(99),
                labels: vec!["Crash".into()],
            })
            .unwrap();
        });
        assert!(matches!(result, CrashResult::Crashed));

        // All committed data across rotated files should survive
        let recovery = WalRecovery::new(&path);
        let records = recovery.recover().unwrap();
        assert_eq!(records.len(), 6, "5 CreateNode + 1 TransactionCommit");
    }

    /// Exhaustive sweep: crash at every possible point during a multi-record
    /// transaction and verify recovery invariants.
    ///
    /// Invariants checked:
    /// 1. Previously committed transactions always survive
    /// 2. Recovery output never contains partial (uncommitted) transactions
    #[test]
    fn test_crash_sweep_all_points() {
        for crash_at in 1..20 {
            let dir = tempdir().unwrap();
            let path = dir.path().to_path_buf();

            // Seed one committed transaction
            {
                let wal = WalManager::with_config(&path, sync_config()).unwrap();
                wal.log(&WalRecord::CreateNode {
                    id: NodeId::new(1),
                    labels: vec!["Base".into()],
                })
                .unwrap();
                wal.log(&WalRecord::TransactionCommit {
                    transaction_id: TransactionId::new(1),
                })
                .unwrap();
            }

            // Attempt a second transaction with crash injection
            let p = path.clone();
            let result = with_crash_at(crash_at, move || {
                let wal = WalManager::with_config(&p, sync_config()).unwrap();
                wal.log(&WalRecord::CreateNode {
                    id: NodeId::new(100),
                    labels: vec!["New".into()],
                })
                .unwrap();
                wal.log(&WalRecord::SetNodeProperty {
                    id: NodeId::new(100),
                    key: "name".into(),
                    value: Value::String("test".into()),
                })
                .unwrap();
                wal.log(&WalRecord::TransactionCommit {
                    transaction_id: TransactionId::new(2),
                })
                .unwrap();
            });

            // Verify recovery invariants
            let recovery = WalRecovery::new(&path);
            let records = recovery.recover().unwrap();

            // Invariant 1: base committed tx always survives
            assert!(
                records.len() >= 2,
                "crash_at={crash_at}: base tx must survive, got {} records",
                records.len()
            );

            // Invariant 2: no partial transactions in output
            let mut pending = 0usize;
            for record in &records {
                match record {
                    WalRecord::TransactionCommit { .. }
                    | WalRecord::TransactionAbort { .. }
                    | WalRecord::Checkpoint { .. } => pending = 0,
                    _ => pending += 1,
                }
            }
            assert_eq!(
                pending, 0,
                "crash_at={crash_at}: recovery must not output partial transactions"
            );

            // If the operation completed, the second tx should also be present
            if matches!(result, CrashResult::Completed(())) {
                assert!(
                    records.len() >= 5,
                    "crash_at={crash_at}: completed run should include second tx"
                );
            }
        }
    }

    /// Aborted transactions are not recovered even without a crash.
    /// Verifies that TransactionAbort correctly discards pending records.
    #[test]
    fn test_abort_then_crash_discards_aborted_tx() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_path_buf();

        {
            let wal = WalManager::with_config(&path, sync_config()).unwrap();
            // Committed tx
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Keep".into()],
            })
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();
            // Aborted tx
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(2),
                labels: vec!["Discard".into()],
            })
            .unwrap();
            wal.log(&WalRecord::TransactionAbort {
                transaction_id: TransactionId::new(2),
            })
            .unwrap();
        }

        // Crash during a third transaction
        let p = path.clone();
        let result = with_crash_at(1, move || {
            let wal = WalManager::with_config(&p, sync_config()).unwrap();
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(3),
                labels: vec!["Also lost".into()],
            })
            .unwrap();
        });
        assert!(matches!(result, CrashResult::Crashed));

        let recovery = WalRecovery::new(&path);
        let records = recovery.recover().unwrap();
        // Only the committed tx (2 records)
        assert_eq!(
            records.len(),
            2,
            "Aborted + crashed records should both be discarded"
        );
    }

    /// A recovery made from an encrypted WAL manager decrypts with its key.
    #[cfg(all(feature = "encryption", not(miri)))]
    #[test]
    fn recovery_from_an_encrypted_wal_keeps_its_cipher() {
        use grafeo_common::encryption::KeyChain;

        let dir = tempdir().unwrap();
        let chain = KeyChain::new([3; 32]);
        let wal = WalManager::with_config_and_cipher(
            dir.path(),
            super::super::WalConfig::default(),
            Some(chain.encryptor_for("grafeo-wal", &19u128.to_le_bytes())),
        )
        .unwrap();
        wal.log(&WalRecord::CreateNode {
            id: NodeId::new(88),
            labels: vec!["Person".to_string()],
        })
        .unwrap();
        wal.log(&WalRecord::TransactionCommit {
            transaction_id: TransactionId::new(3),
        })
        .unwrap();
        wal.flush().unwrap();

        let records = WalRecovery::from_wal(&wal).recover().unwrap();
        assert_eq!(records.len(), 2, "both records decrypt: {records:?}");
        assert!(
            WalRecovery::new(dir.path()).recover().unwrap().is_empty(),
            "without the cipher nothing reads"
        );
    }

    #[cfg(all(feature = "encryption", not(miri)))]
    #[test]
    fn test_encrypted_wal_roundtrip() {
        use grafeo_common::encryption::{KEY_SIZE, KeyChain};

        let dir = tempdir().unwrap();
        let key = [42u8; KEY_SIZE];
        let chain = KeyChain::new(key);

        // Write encrypted records
        {
            let mut wal = WalManager::open(dir.path()).unwrap();
            wal.set_encryptor(chain.encryptor_for("grafeo-wal", &0u64.to_be_bytes()));

            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Person".to_string()],
            })
            .unwrap();

            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();

            wal.sync().unwrap();
        }

        // Recover with same key
        let mut recovery = WalRecovery::new(dir.path());
        recovery.set_encryptor(chain.encryptor_for("grafeo-wal", &0u64.to_be_bytes()));
        let records = recovery.recover().unwrap();
        assert_eq!(records.len(), 2, "should recover both encrypted records");
    }

    #[cfg(all(feature = "encryption", not(miri)))]
    #[test]
    fn test_encrypted_wal_wrong_key_fails() {
        use grafeo_common::encryption::{KEY_SIZE, KeyChain};

        let dir = tempdir().unwrap();
        let key = [42u8; KEY_SIZE];
        let chain = KeyChain::new(key);

        // Write with key A
        {
            let mut wal = WalManager::open(dir.path()).unwrap();
            wal.set_encryptor(chain.encryptor_for("grafeo-wal", &0u64.to_be_bytes()));

            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Test".to_string()],
            })
            .unwrap();

            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();

            wal.sync().unwrap();
        }

        // Recovery with key B: no frame decrypts, and as whole frames follow
        // the first one, that is damage (or the wrong key), never a torn
        // tail that would lose the commits without a word (#425).
        let wrong_key = [99u8; KEY_SIZE];
        let wrong_chain = KeyChain::new(wrong_key);
        let mut recovery = WalRecovery::new(dir.path());
        recovery.set_encryptor(wrong_chain.encryptor_for("grafeo-wal", &0u64.to_be_bytes()));

        let result = recovery.recover();
        let Err(Error::Corruption(corruption)) = result else {
            panic!("a wrong key fails the recovery: {result:?}");
        };
        assert_eq!(corruption.offset, Some(0));
        assert!(
            corruption.what.contains("does not decrypt"),
            "{corruption:?}"
        );
    }
}
