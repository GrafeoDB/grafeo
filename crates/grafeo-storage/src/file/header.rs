//! Read and write file headers and database headers.
//!
//! All I/O targets a [`File`] handle. Headers are serialized with bincode and
//! zero-padded to their full region size so that the file layout is always
//! page-aligned.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};

use grafeo_common::utils::error::{Error, Result};

use super::format::{DATA_OFFSET, DB_HEADER_SIZE, DbHeader, FILE_HEADER_SIZE, FileHeader, MAGIC};

// ---------------------------------------------------------------------------
// File header (offset 0, 4 KiB)
// ---------------------------------------------------------------------------

/// Writes a [`FileHeader`] at offset 0, padded to [`FILE_HEADER_SIZE`] bytes.
///
/// # Errors
///
/// Returns an error if serialization fails or the I/O write fails.
pub fn write_file_header(file: &mut File, header: &FileHeader) -> Result<()> {
    let encoded = bincode::serde::encode_to_vec(header, bincode::config::standard())
        .map_err(|e| Error::Serialization(e.to_string()))?;

    // reason: FILE_HEADER_SIZE is 4096, a constant that fits in usize on all targets
    #[allow(clippy::cast_possible_truncation)]
    let mut buf = vec![0u8; FILE_HEADER_SIZE as usize];
    if encoded.len() > buf.len() {
        return Err(Error::Internal(
            "FileHeader serialization exceeds page size".into(),
        ));
    }
    buf[..encoded.len()].copy_from_slice(&encoded);

    file.seek(SeekFrom::Start(0))?;
    file.write_all(&buf)?;
    Ok(())
}

/// Reads and deserializes the [`FileHeader`] from offset 0.
///
/// # Errors
///
/// Returns an error if the I/O read or deserialization fails.
pub fn read_file_header(file: &mut File) -> Result<FileHeader> {
    // reason: FILE_HEADER_SIZE is 4096, a constant that fits in usize on all targets
    #[allow(clippy::cast_possible_truncation)]
    let mut buf = vec![0u8; FILE_HEADER_SIZE as usize];
    file.seek(SeekFrom::Start(0))?;
    file.read_exact(&mut buf)?;

    let (header, _): (FileHeader, _) =
        bincode::serde::decode_from_slice(&buf, bincode::config::standard())
            .map_err(|e| Error::Serialization(e.to_string()))?;
    Ok(header)
}

/// Validates the file header: checks magic bytes and format version.
///
/// # Errors
///
/// Returns an error if the magic bytes are invalid or the format version is unsupported.
pub fn validate_file_header(header: &FileHeader) -> Result<()> {
    if header.magic != MAGIC {
        return Err(Error::Internal(format!(
            "invalid magic bytes: expected {:?}, got {:?}",
            MAGIC, header.magic
        )));
    }
    if header.format_version > super::format::FORMAT_VERSION {
        return Err(Error::Internal(format!(
            "unsupported format version {} (max supported: {})",
            header.format_version,
            super::format::FORMAT_VERSION
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Database headers (two slots at offsets 4 KiB and 8 KiB)
// ---------------------------------------------------------------------------

/// Returns the byte offset of database header slot 0 or 1.
fn db_header_offset(slot: u8) -> u64 {
    FILE_HEADER_SIZE + u64::from(slot) * DB_HEADER_SIZE
}

/// Writes a [`DbHeader`] to the given slot (0 or 1), padded to
/// [`DB_HEADER_SIZE`] bytes.
///
/// # Errors
///
/// Returns an error if serialization fails or the I/O write fails.
pub fn write_db_header(file: &mut File, slot: u8, header: &DbHeader) -> Result<()> {
    debug_assert!(slot < 2, "db header slot must be 0 or 1");

    let encoded = bincode::serde::encode_to_vec(header, bincode::config::standard())
        .map_err(|e| Error::Serialization(e.to_string()))?;

    // reason: DB_HEADER_SIZE is 4096, a constant that fits in usize on all targets
    #[allow(clippy::cast_possible_truncation)]
    let mut buf = vec![0u8; DB_HEADER_SIZE as usize];
    if encoded.len() > buf.len() {
        return Err(Error::Internal(
            "DbHeader serialization exceeds page size".into(),
        ));
    }
    buf[..encoded.len()].copy_from_slice(&encoded);

    file.seek(SeekFrom::Start(db_header_offset(slot)))?;
    file.write_all(&buf)?;
    Ok(())
}

/// Reads a single [`DbHeader`] from the given slot.
fn read_db_header(file: &mut File, slot: u8) -> Result<DbHeader> {
    debug_assert!(slot < 2, "db header slot must be 0 or 1");

    // reason: DB_HEADER_SIZE is 4096, a constant that fits in usize on all targets
    #[allow(clippy::cast_possible_truncation)]
    let mut buf = vec![0u8; DB_HEADER_SIZE as usize];
    file.seek(SeekFrom::Start(db_header_offset(slot)))?;
    file.read_exact(&mut buf)?;

    let (header, _): (DbHeader, _) =
        bincode::serde::decode_from_slice(&buf, bincode::config::standard())
            .map_err(|e| Error::Serialization(e.to_string()))?;
    Ok(header)
}

/// Reads both database headers from slots 0 and 1.
///
/// # Errors
///
/// Returns an error if the I/O read or deserialization of either slot fails.
pub fn read_db_headers(file: &mut File) -> Result<(DbHeader, DbHeader)> {
    let h0 = read_db_header(file, 0)?;
    let h1 = read_db_header(file, 1)?;
    Ok((h0, h1))
}

/// Returns the active (authoritative) database header.
///
/// The header with the higher `iteration` counter wins; ties go to slot 0.
/// Both headers are returned by-value alongside their slot index. Checksum
/// validation against the data payload is the caller's responsibility (see
/// [`LegacyFile::contents`](crate::file::legacy::LegacyFile::contents)); this
/// function does not fall back to the other slot on a payload mismatch.
///
/// Returns `(active_slot, header)`.
#[must_use]
pub fn active_db_header(h0: &DbHeader, h1: &DbHeader) -> (u8, DbHeader) {
    if h1.iteration > h0.iteration {
        (1, h1.clone())
    } else {
        (0, h0.clone())
    }
}

/// Returns the slot index (0 or 1) that should be written next.
///
/// This is always the *inactive* (stale) slot, i.e., the one with the
/// lower iteration counter.
#[must_use]
pub fn inactive_slot(h0: &DbHeader, h1: &DbHeader) -> u8 {
    u8::from(h1.iteration <= h0.iteration)
}

/// Returns the byte offset where snapshot data should be written/read.
///
/// Currently always [`DATA_OFFSET`] (12 KiB), but exposed as a function
/// so callers don't depend on the constant directly.
#[must_use]
pub const fn snapshot_data_offset() -> u64 {
    DATA_OFFSET
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn create_test_file() -> (File, tempfile::TempPath) {
        let tmp = NamedTempFile::new().expect("create temp file");
        let (file, path) = tmp.into_parts();
        (file, path)
    }

    #[test]
    fn file_header_roundtrip() {
        let (mut file, _path) = create_test_file();
        let original = FileHeader::new();

        write_file_header(&mut file, &original).unwrap();
        let loaded = read_file_header(&mut file).unwrap();

        assert_eq!(original, loaded);
    }

    #[test]
    fn file_header_validation_rejects_bad_magic() {
        let mut header = FileHeader::new();
        header.magic = *b"NOPE";

        let result = validate_file_header(&header);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("invalid magic"));
    }

    #[test]
    fn file_header_validation_rejects_future_version() {
        let mut header = FileHeader::new();
        header.format_version = 999;

        let result = validate_file_header(&header);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("unsupported"));
    }

    #[test]
    fn db_header_roundtrip_slot0() {
        let (mut file, _path) = create_test_file();

        // Write file header + both DB header slots to establish full layout
        write_file_header(&mut file, &FileHeader::new()).unwrap();
        write_db_header(&mut file, 1, &DbHeader::EMPTY).unwrap();

        let original = DbHeader {
            iteration: 1,
            checksum: 0xDEAD_BEEF,
            snapshot_length: 1024,
            epoch: 42,
            transaction_id: 7,
            node_count: 100,
            edge_count: 200,
            timestamp_ms: 1_700_000_000_000,
        };

        write_db_header(&mut file, 0, &original).unwrap();
        let (h0, _h1) = read_db_headers(&mut file).unwrap();

        assert_eq!(original, h0);
    }

    #[test]
    fn db_header_roundtrip_slot1() {
        let (mut file, _path) = create_test_file();
        write_file_header(&mut file, &FileHeader::new()).unwrap();
        write_db_header(&mut file, 0, &DbHeader::EMPTY).unwrap();

        let original = DbHeader {
            iteration: 5,
            checksum: 0x1234,
            snapshot_length: 2048,
            epoch: 10,
            transaction_id: 3,
            node_count: 50,
            edge_count: 75,
            timestamp_ms: 1_700_000_001_000,
        };

        write_db_header(&mut file, 1, &original).unwrap();
        let (_h0, h1) = read_db_headers(&mut file).unwrap();

        assert_eq!(original, h1);
    }

    #[test]
    fn active_header_picks_higher_iteration() {
        let h0 = DbHeader {
            iteration: 3,
            ..DbHeader::EMPTY
        };
        let h1 = DbHeader {
            iteration: 5,
            ..DbHeader::EMPTY
        };

        let (slot, active) = active_db_header(&h0, &h1);
        assert_eq!(slot, 1);
        assert_eq!(active.iteration, 5);
    }

    #[test]
    fn active_header_defaults_to_slot0_when_equal() {
        let h0 = DbHeader {
            iteration: 2,
            ..DbHeader::EMPTY
        };
        let h1 = DbHeader {
            iteration: 2,
            ..DbHeader::EMPTY
        };

        let (slot, _) = active_db_header(&h0, &h1);
        assert_eq!(slot, 0);
    }

    #[test]
    fn active_header_handles_both_empty() {
        let (slot, header) = active_db_header(&DbHeader::EMPTY, &DbHeader::EMPTY);
        assert_eq!(slot, 0);
        assert!(header.is_empty());
    }

    #[test]
    fn inactive_slot_alternates() {
        let h0 = DbHeader {
            iteration: 3,
            ..DbHeader::EMPTY
        };
        let h1 = DbHeader {
            iteration: 5,
            ..DbHeader::EMPTY
        };

        // h1 is active (higher), so inactive is slot 0
        assert_eq!(inactive_slot(&h0, &h1), 0);

        // h0 is active (higher), so inactive is slot 1
        assert_eq!(inactive_slot(&h1, &h0), 1);
    }

    #[test]
    fn dual_header_alternation() {
        let (mut file, _path) = create_test_file();
        write_file_header(&mut file, &FileHeader::new()).unwrap();
        write_db_header(&mut file, 0, &DbHeader::EMPTY).unwrap();
        write_db_header(&mut file, 1, &DbHeader::EMPTY).unwrap();

        // First checkpoint: write to inactive slot
        let (h0, h1) = read_db_headers(&mut file).unwrap();
        let target_slot = inactive_slot(&h0, &h1);

        let checkpoint1 = DbHeader {
            iteration: 1,
            checksum: 0xAAAA,
            snapshot_length: 100,
            epoch: 1,
            ..DbHeader::EMPTY
        };
        write_db_header(&mut file, target_slot, &checkpoint1).unwrap();

        // Verify checkpoint 1 is active
        let (h0, h1) = read_db_headers(&mut file).unwrap();
        let (active_slot, active) = active_db_header(&h0, &h1);
        assert_eq!(active.iteration, 1);

        // Second checkpoint: write to the other slot
        let target_slot = inactive_slot(&h0, &h1);
        assert_ne!(target_slot, active_slot);

        let checkpoint2 = DbHeader {
            iteration: 2,
            checksum: 0xBBBB,
            snapshot_length: 200,
            epoch: 2,
            ..DbHeader::EMPTY
        };
        write_db_header(&mut file, target_slot, &checkpoint2).unwrap();

        // Verify checkpoint 2 is active
        let (h0, h1) = read_db_headers(&mut file).unwrap();
        let (_, active) = active_db_header(&h0, &h1);
        assert_eq!(active.iteration, 2);
        assert_eq!(active.snapshot_length, 200);
    }
}
