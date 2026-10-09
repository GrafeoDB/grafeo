//! Errors of the WAL v2 writer and scanner.
//!
//! [`WalError`] says what is wrong and where: the file, the byte offset in
//! it and the log position. [`GroupError`] says what a failed group means for
//! the transaction that wrote it. Both convert into the crate-wide
//! [`Error`](grafeo_common::utils::error::Error) for callers that propagate
//! with `?`.

#![deny(clippy::let_underscore_must_use)]

use std::path::PathBuf;

use grafeo_common::utils::error::{Error, StorageError};

/// What went wrong while writing or scanning a WAL.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum WalError {
    /// An I/O operation on a WAL file or directory failed.
    #[error("I/O error on the WAL at {}: {source}", .path.display())]
    Io {
        /// The file or directory.
        path: PathBuf,
        /// The error of the operation.
        #[source]
        source: std::io::Error,
    },

    /// A segment header is damaged, or of a version or with a feature this
    /// build cannot read.
    #[error("the WAL segment {} has an invalid header: {reason}", .path.display())]
    SegmentHeader {
        /// The segment file.
        path: PathBuf,
        /// What is wrong with the header.
        reason: String,
    },

    /// A segment belongs to another database.
    #[error(
        "the WAL at {} belongs to database {found:032x}, not {expected:032x}",
        .path.display()
    )]
    ForeignDatabase {
        /// The segment file.
        path: PathBuf,
        /// The database id the segment header holds.
        found: u128,
        /// The database id of the database being opened.
        expected: u128,
    },

    /// The segments do not form one unbroken chain of log positions from the
    /// checkpoint on: a segment is missing, or one starts where none can.
    #[error("the WAL has a gap: {reason}")]
    Gap {
        /// Which segments and log positions disagree.
        reason: String,
    },

    /// The key does not open the segment: its key check fails.
    #[error(
        "wrong key for the WAL of database {database_id:032x} (segment {})",
        .path.display()
    )]
    WrongKey {
        /// The segment file.
        path: PathBuf,
        /// The database the segment belongs to.
        database_id: u128,
    },

    /// A segment is encrypted, and no key was given (or this build has no
    /// `encryption` feature).
    #[error(
        "the WAL segment {} is encrypted: open the database with its key, in a build with \
         the encryption feature",
        .path.display()
    )]
    MissingKey {
        /// The segment file.
        path: PathBuf,
    },

    /// A segment is plaintext, while the database is encrypted.
    #[error("the WAL segment {} is not encrypted, but the database is", .path.display())]
    NotEncrypted {
        /// The segment file.
        path: PathBuf,
    },

    /// Bytes of a frame are wrong where the log must be intact: in a sealed
    /// segment, before a group that was synced after them, or a frame whose
    /// checksum is right but whose authentication fails.
    #[error(
        "the WAL segment {} is damaged at offset {offset} (LSN {lsn}): {reason}",
        .path.display()
    )]
    Damaged {
        /// The segment file.
        path: PathBuf,
        /// The byte offset of the damaged frame in the file.
        offset: u64,
        /// The log position of the damaged frame.
        lsn: u64,
        /// What is wrong with the frame.
        reason: String,
    },

    /// A whole frame (its checksum right) that this build cannot read: it
    /// sets a flag of a later release. Refused, also when salvaging: it is
    /// not damage, and a newer version of Grafeo reads it.
    #[error(
        "the WAL segment {} holds a frame at offset {offset} (LSN {lsn}) that this version \
         cannot read: {reason}; the WAL needs a newer version of Grafeo",
        .path.display()
    )]
    UnsupportedFrame {
        /// The segment file.
        path: PathBuf,
        /// The byte offset of the frame in the file.
        offset: u64,
        /// The log position of the frame.
        lsn: u64,
        /// What this build does not know.
        reason: String,
    },

    /// A sync was asked for up to a log position past the end of the log:
    /// nothing written reaches it, so no sync can make it durable.
    #[error("cannot sync the WAL up to LSN {lsn}: the log ends at LSN {end_lsn}")]
    BeyondEnd {
        /// The log position asked for.
        lsn: u64,
        /// The end of the last complete group.
        end_lsn: u64,
    },

    /// A record does not fit in a frame.
    #[error("a WAL record of {length} bytes is over the limit of {limit} bytes")]
    RecordTooLarge {
        /// The length of the record.
        length: usize,
        /// The largest record a frame holds.
        limit: usize,
    },

    /// The writer stopped taking writes after an earlier failure.
    #[error(
        "the WAL takes no more writes after an earlier failure ({reason}): reopen the database"
    )]
    Unavailable {
        /// The earlier failure.
        reason: String,
    },

    /// The WAL directory does not end where the writer is asked to continue.
    #[error("{reason}")]
    Misplaced {
        /// Where the WAL ends and where the writer was asked to start.
        reason: String,
    },

    /// A log position would pass the largest one.
    #[error("the WAL log position would pass the largest LSN ({})", u64::MAX)]
    LsnOverflow,

    /// Encrypting a frame or a key check failed.
    #[error("WAL encryption failed: {reason}")]
    Encryption {
        /// The error of the cipher.
        reason: String,
    },
}

impl WalError {
    /// An I/O error on `path`.
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    /// The I/O error of an injected failure (`maybe_fail`) on `path`.
    pub(crate) fn injected(path: impl Into<PathBuf>, error: Error) -> Self {
        let source = match error {
            Error::Io(source) => source,
            other => std::io::Error::other(other.to_string()),
        };
        Self::io(path, source)
    }
}

/// Why a group of frames did not complete, and what that means for the
/// transaction that wrote it.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum GroupError {
    /// The group has no LAST frame in the log, so it is never replayed: the
    /// transaction can be rolled back. The writer cut the log back to where
    /// the group started; when that cut failed too, the writer is poisoned
    /// (see [`Wal::is_poisoned`](super::Wal::is_poisoned)) and the partial
    /// group stays as a torn tail that the next open cuts.
    #[error("the log group of the transaction was not written: {0}")]
    NotWritten(#[source] WalError),

    /// The LAST frame was written but syncing it failed, so whether the
    /// group is durable is unknown. The writer is poisoned; the next open
    /// decides from what the log holds.
    #[error(
        "the log group of the transaction was written, but syncing it failed, so whether it \
         is durable is unknown: {0}"
    )]
    OutcomeUnknown(#[source] WalError),

    /// The writer was poisoned by an earlier failure, so this group was not
    /// started.
    #[error(
        "the WAL takes no more writes after an earlier failure ({reason}): reopen the database"
    )]
    Unavailable {
        /// The earlier failure.
        reason: String,
    },
}

impl From<WalError> for Error {
    fn from(error: WalError) -> Self {
        match error {
            WalError::Io { path, source } => Error::Io(std::io::Error::new(
                source.kind(),
                format!("WAL {}: {source}", path.display()),
            )),
            WalError::SegmentHeader { .. } | WalError::Gap { .. } | WalError::Damaged { .. } => {
                Error::Storage(StorageError::Corruption(error.to_string()))
            }
            WalError::ForeignDatabase { .. }
            | WalError::WrongKey { .. }
            | WalError::MissingKey { .. }
            | WalError::NotEncrypted { .. }
            | WalError::UnsupportedFrame { .. }
            | WalError::Misplaced { .. } => {
                Error::Storage(StorageError::RecoveryFailed(error.to_string()))
            }
            WalError::RecordTooLarge { .. } => Error::InvalidValue(error.to_string()),
            WalError::Unavailable { .. } => Error::Io(std::io::Error::other(error.to_string())),
            WalError::LsnOverflow | WalError::BeyondEnd { .. } | WalError::Encryption { .. } => {
                Error::Internal(error.to_string())
            }
        }
    }
}

impl From<GroupError> for Error {
    fn from(error: GroupError) -> Self {
        match error {
            GroupError::NotWritten(inner) => {
                let message = format!("the log group of the transaction was not written: {inner}");
                match Error::from(inner) {
                    Error::Io(source) => Error::Io(std::io::Error::new(source.kind(), message)),
                    Error::InvalidValue(_) => Error::InvalidValue(message),
                    _ => Error::Internal(message),
                }
            }
            GroupError::OutcomeUnknown(_) | GroupError::Unavailable { .. } => {
                Error::Io(std::io::Error::other(error.to_string()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_damage_error_names_the_file_the_offset_and_the_lsn() {
        let error = WalError::Damaged {
            path: PathBuf::from("Amsterdam").join("wal_00000000000000000319.log"),
            offset: 1988,
            lsn: 3,
            reason: "checksum mismatch".to_string(),
        };
        let message = error.to_string();
        assert!(
            message.contains("wal_00000000000000000319.log")
                && message.contains("offset 1988")
                && message.contains("LSN 3")
                && message.contains("checksum mismatch"),
            "{message}"
        );
        let converted = Error::from(error);
        assert!(
            matches!(converted, Error::Storage(StorageError::Corruption(_))),
            "damage is corruption: {converted:?}"
        );
    }

    #[test]
    fn a_foreign_wal_names_both_databases() {
        let message = WalError::ForeignDatabase {
            path: PathBuf::from("wal_00000000000000000000.log"),
            found: 0x19,
            expected: 0x88,
        }
        .to_string();
        assert!(
            message.contains("00000000000000000000000000000019")
                && message.contains("00000000000000000000000000000088"),
            "{message}"
        );
    }

    #[test]
    fn a_group_that_was_not_written_keeps_the_io_kind() {
        let error = GroupError::NotWritten(WalError::io(
            "wal_00000000000000000088.log",
            std::io::Error::new(std::io::ErrorKind::StorageFull, "disk full"),
        ));
        match Error::from(error) {
            Error::Io(source) => {
                assert_eq!(source.kind(), std::io::ErrorKind::StorageFull);
                assert!(source.to_string().contains("not written"), "{source}");
            }
            other => panic!("expected an I/O error, got {other:?}"),
        }
    }
}
