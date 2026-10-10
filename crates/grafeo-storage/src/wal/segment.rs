//! WAL v2 segment files: their names and their 128-byte header.
//!
//! A WAL is a directory of segment files named after the log position (LSN)
//! of their first frame, `wal_{first_lsn:020}.log`. Each starts with a
//! 128-byte header (little endian):
//!
//! | Offset | Field |
//! | --- | --- |
//! | 0 | magic `GRAFWAL\0` |
//! | 8 | version u16 = 2 |
//! | 10 | header length u16 = 128 |
//! | 12 | flags u32: bit 0 encrypted; bits 0 to 15 incompatible, 16 to 31 compatible |
//! | 16 | database id u128 |
//! | 32 | first LSN u64 (equals the file name) |
//! | 40 | creation time in milliseconds since the Unix epoch, u64 |
//! | 48 | salt `[u8; 32]` (zero when plaintext) |
//! | 80 | key check `[u8; 28]` (zero when plaintext) |
//! | 108 | reserved `[u8; 16]`, written 0, ignored |
//! | 124 | CRC-32 of bytes 0..124 |
//!
//! A reader refuses an incompatible flag it does not know and ignores a
//! compatible one, the split the file header uses.

#![deny(clippy::let_underscore_must_use)]

use std::path::{Path, PathBuf};

use super::error::WalError;

/// Size of a segment header in bytes, as the header stores it.
const SEGMENT_HEADER_LENGTH: u16 = 128;

/// Size of a segment header in bytes; the first frame starts right after it.
pub const SEGMENT_HEADER_BYTES: usize = SEGMENT_HEADER_LENGTH as usize;

/// The magic bytes every segment starts with.
pub const SEGMENT_MAGIC: [u8; 8] = *b"GRAFWAL\0";

/// The segment format version this build writes and reads.
pub const SEGMENT_VERSION: u16 = 2;

/// Size of a segment's salt, from which its key is derived.
pub const SALT_BYTES: usize = 32;

/// Size of a segment's key check: the nonce and tag of an empty message.
pub const KEY_CHECK_BYTES: usize = 28;

/// The header bytes the key check authenticates: everything before it.
pub const KEY_CHECK_AAD_BYTES: usize = 80;

/// Incompatible feature bit 0: the frames are encrypted.
const FLAG_ENCRYPTED: u32 = 1;
/// Bits 0 to 15 of the flags word: incompatible features.
const INCOMPATIBLE_FLAGS: u32 = 0x0000_FFFF;
/// The incompatible features this build can read.
const KNOWN_INCOMPATIBLE_FLAGS: u32 = FLAG_ENCRYPTED;
/// Offset of the CRC-32 of the bytes before it.
const CRC_OFFSET: usize = 124;

/// The header at the start of every segment file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentHeader {
    /// Whether the frames of the segment are encrypted.
    pub encrypted: bool,
    /// The database the segment belongs to (from the file header).
    pub database_id: u128,
    /// The log position of the segment's first frame; equals its file name.
    pub first_lsn: u64,
    /// When the segment was created, in milliseconds since the Unix epoch.
    pub creation_time_ms: u64,
    /// The random salt the segment key is derived from (zero when plaintext).
    pub salt: [u8; SALT_BYTES],
    /// The nonce and tag of an empty message under the segment key, with the
    /// header bytes before it as associated data (zero when plaintext).
    pub key_check: [u8; KEY_CHECK_BYTES],
}

impl SegmentHeader {
    /// Encodes the header into its 128 bytes, the checksum included.
    #[must_use]
    pub fn encode(&self) -> [u8; SEGMENT_HEADER_BYTES] {
        let mut bytes = [0u8; SEGMENT_HEADER_BYTES];
        bytes[..KEY_CHECK_AAD_BYTES].copy_from_slice(&self.key_check_aad());
        bytes[80..108].copy_from_slice(&self.key_check);
        let crc = crc32fast::hash(&bytes[..CRC_OFFSET]);
        bytes[CRC_OFFSET..].copy_from_slice(&crc.to_le_bytes());
        bytes
    }

    /// The header bytes 0..80 that the key check authenticates: magic,
    /// version, length, flags, database id, first LSN, creation time and salt.
    #[must_use]
    pub fn key_check_aad(&self) -> [u8; KEY_CHECK_AAD_BYTES] {
        let mut bytes = [0u8; KEY_CHECK_AAD_BYTES];
        bytes[0..8].copy_from_slice(&SEGMENT_MAGIC);
        bytes[8..10].copy_from_slice(&SEGMENT_VERSION.to_le_bytes());
        bytes[10..12].copy_from_slice(&SEGMENT_HEADER_LENGTH.to_le_bytes());
        let flags = if self.encrypted { FLAG_ENCRYPTED } else { 0 };
        bytes[12..16].copy_from_slice(&flags.to_le_bytes());
        bytes[16..32].copy_from_slice(&self.database_id.to_le_bytes());
        bytes[32..40].copy_from_slice(&self.first_lsn.to_le_bytes());
        bytes[40..48].copy_from_slice(&self.creation_time_ms.to_le_bytes());
        bytes[48..80].copy_from_slice(&self.salt);
        bytes
    }

    /// Decodes the header of the segment at `path` from its first bytes,
    /// checking the magic, then the version, then the checksum, the header
    /// length and the incompatible feature flags. The version comes before
    /// the checksum: a later version may lay its header out otherwise, so
    /// its checksum need not sit at byte 124, and the error says that the
    /// WAL needs a newer version of Grafeo, not that it is damaged.
    ///
    /// # Errors
    ///
    /// Returns [`WalError::UnsupportedSegment`] naming `path` when the
    /// version is above 2 or an incompatible feature flag this build does
    /// not know is set, and [`WalError::SegmentHeader`] (damage) when the
    /// bytes are shorter than a header, the magic is wrong, the version is
    /// below 2 (no release writes one with this magic), the checksum is
    /// wrong or the header length is not 128.
    pub fn decode(bytes: &[u8], path: &Path) -> Result<Self, WalError> {
        let invalid = |reason: String| WalError::SegmentHeader {
            path: path.to_path_buf(),
            reason,
        };
        let unsupported = |reason: String| WalError::UnsupportedSegment {
            path: path.to_path_buf(),
            reason,
        };
        if bytes.len() < SEGMENT_HEADER_BYTES {
            return Err(invalid(format!(
                "the header is {} bytes, expected {SEGMENT_HEADER_BYTES}",
                bytes.len()
            )));
        }
        if bytes[0..8] != SEGMENT_MAGIC {
            return Err(invalid("not a WAL v2 segment (wrong magic)".to_string()));
        }
        let version = read_u16(bytes, 8);
        if version > SEGMENT_VERSION {
            return Err(unsupported(format!(
                "segment version {version}, this version reads {SEGMENT_VERSION}"
            )));
        }
        if version != SEGMENT_VERSION {
            return Err(invalid(format!(
                "segment version {version}, expected {SEGMENT_VERSION}"
            )));
        }
        let stored = read_u32(bytes, CRC_OFFSET);
        let computed = crc32fast::hash(&bytes[..CRC_OFFSET]);
        if stored != computed {
            return Err(invalid(format!(
                "checksum mismatch: stored {stored:#010x}, computed {computed:#010x}"
            )));
        }
        let header_length = read_u16(bytes, 10);
        if usize::from(header_length) != SEGMENT_HEADER_BYTES {
            return Err(invalid(format!(
                "header length {header_length}, expected {SEGMENT_HEADER_BYTES}"
            )));
        }
        let flags = read_u32(bytes, 12);
        let unknown = flags & INCOMPATIBLE_FLAGS & !KNOWN_INCOMPATIBLE_FLAGS;
        if unknown != 0 {
            return Err(unsupported(format!(
                "unknown incompatible feature flags {unknown:#06x} ({})",
                describe_bits(unknown)
            )));
        }
        let mut database_id = [0u8; 16];
        database_id.copy_from_slice(&bytes[16..32]);
        let mut salt = [0u8; SALT_BYTES];
        salt.copy_from_slice(&bytes[48..80]);
        let mut key_check = [0u8; KEY_CHECK_BYTES];
        key_check.copy_from_slice(&bytes[80..108]);
        Ok(Self {
            encrypted: flags & FLAG_ENCRYPTED != 0,
            database_id: u128::from_le_bytes(database_id),
            first_lsn: read_u64(bytes, 32),
            creation_time_ms: read_u64(bytes, 40),
            salt,
            key_check,
        })
    }
}

/// The header bytes 0..80 as `bytes` (a decoded header) stores them: the
/// associated data of its key check. They are taken as stored, not rebuilt
/// from the decoded fields, so a compatible flag of a later release, which
/// the decoded header leaves out, is authenticated as it was written.
///
/// # Panics
///
/// Panics when `bytes` is shorter than a header; [`SegmentHeader::decode`]
/// refuses such bytes first.
#[must_use]
pub fn stored_key_check_aad(bytes: &[u8]) -> [u8; KEY_CHECK_AAD_BYTES] {
    let mut aad = [0u8; KEY_CHECK_AAD_BYTES];
    aad.copy_from_slice(&bytes[..KEY_CHECK_AAD_BYTES]);
    aad
}

/// Whether the first bytes of a segment, read up to a header, hold no
/// header at all: fewer bytes than a header, or only zeros.
pub(crate) fn header_is_blank(bytes: &[u8]) -> bool {
    bytes.len() < SEGMENT_HEADER_BYTES || bytes.iter().all(|&byte| byte == 0)
}

/// Whether the segment file at `path` was cut off while it was created:
/// shorter than a header, or zeros from its first byte to its last.
///
/// The writer syncs a new segment's header before it writes a frame, so a
/// header of zeros with any other byte behind it is never a segment cut off
/// during creation: it is a damaged header, and the frames behind it may
/// have been synced. Reads up to the first byte that is not zero.
///
/// # Errors
///
/// Returns [`WalError::Io`] when the file cannot be read.
pub fn is_unfinished_segment(path: &Path) -> Result<bool, WalError> {
    let io = |source| WalError::io(path, source);
    let mut file = std::fs::File::open(path).map_err(io)?;
    let length = file.metadata().map_err(io)?.len();
    if length < SEGMENT_HEADER_BYTES as u64 {
        return Ok(true);
    }
    let mut buffer = vec![0u8; 64 * 1024];
    loop {
        let read = match std::io::Read::read(&mut file, &mut buffer) {
            Ok(read) => read,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(source) => return Err(io(source)),
        };
        if read == 0 {
            return Ok(true);
        }
        if buffer[..read].iter().any(|&byte| byte != 0) {
            return Ok(false);
        }
    }
}

/// The file name of the segment whose first frame is at `first_lsn`.
#[must_use]
pub fn segment_file_name(first_lsn: u64) -> String {
    format!("wal_{first_lsn:020}.log")
}

/// The first LSN a segment file name stands for, or `None` for a name that
/// is not exactly `wal_` followed by 20 digits and `.log` (a 0.5.x segment
/// name has 8 digits).
#[must_use]
pub fn parse_segment_file_name(name: &str) -> Option<u64> {
    let digits = name.strip_prefix("wal_")?.strip_suffix(".log")?;
    if digits.len() != 20 || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

/// The entries of a WAL directory: its segments in LSN order, and every
/// other entry, which is never deleted.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WalDirectory {
    /// The segments, as (first LSN, path), in LSN order.
    pub segments: Vec<(u64, PathBuf)>,
    /// Entries that are not segments, in name order.
    pub others: Vec<PathBuf>,
}

/// Lists the WAL directory `dir`.
///
/// # Errors
///
/// Returns [`WalError::Io`] when the directory cannot be read.
pub fn list_wal_directory(dir: &Path) -> Result<WalDirectory, WalError> {
    let entries = std::fs::read_dir(dir).map_err(|source| WalError::io(dir, source))?;
    let mut listing = WalDirectory::default();
    for entry in entries {
        let entry = entry.map_err(|source| WalError::io(dir, source))?;
        let path = entry.path();
        let lsn = entry.file_name().to_str().and_then(parse_segment_file_name);
        let is_file = entry
            .file_type()
            .map_err(|source| WalError::io(&path, source))?
            .is_file();
        match lsn {
            Some(lsn) if is_file => listing.segments.push((lsn, path)),
            _ => listing.others.push(path),
        }
    }
    listing.segments.sort_unstable_by_key(|(lsn, _)| *lsn);
    listing.others.sort();
    Ok(listing)
}

/// Names the set bits of `mask`, as `bit 2` or `bits 2, 15`.
fn describe_bits(mask: u32) -> String {
    let bits: Vec<String> = (0..32)
        .filter(|bit| mask & (1 << bit) != 0)
        .map(|bit: u32| bit.to_string())
        .collect();
    let noun = if bits.len() == 1 { "bit" } else { "bits" };
    format!("{noun} {}", bits.join(", "))
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    let mut buffer = [0u8; 2];
    buffer.copy_from_slice(&bytes[offset..offset + 2]);
    u16::from_le_bytes(buffer)
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    let mut buffer = [0u8; 4];
    buffer.copy_from_slice(&bytes[offset..offset + 4]);
    u32::from_le_bytes(buffer)
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    let mut buffer = [0u8; 8];
    buffer.copy_from_slice(&bytes[offset..offset + 8]);
    u64::from_le_bytes(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn known_header() -> SegmentHeader {
        SegmentHeader {
            encrypted: true,
            database_id: 0x0102_0304_0506_0708_090A_0B0C_0D0E_0F10,
            first_lsn: 1988,
            creation_time_ms: 319,
            salt: [0x19; SALT_BYTES],
            key_check: [0x88; KEY_CHECK_BYTES],
        }
    }

    fn path() -> PathBuf {
        PathBuf::from(segment_file_name(1988))
    }

    /// The exact bytes of a known header, as every segment written so far
    /// carries them: a change here makes those segments unreadable.
    #[test]
    fn the_segment_header_bytes_are_pinned() {
        let bytes = known_header().encode();
        let mut expected = Vec::new();
        expected.extend_from_slice(b"GRAFWAL\0");
        expected.extend_from_slice(&[0x02, 0x00]); // version 2
        expected.extend_from_slice(&[0x80, 0x00]); // header length 128
        expected.extend_from_slice(&[0x01, 0x00, 0x00, 0x00]); // flags: encrypted
        expected.extend_from_slice(&[
            0x10, 0x0F, 0x0E, 0x0D, 0x0C, 0x0B, 0x0A, 0x09, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03,
            0x02, 0x01,
        ]); // database id, little endian
        expected.extend_from_slice(&[0xC4, 0x07, 0, 0, 0, 0, 0, 0]); // first LSN 1988
        expected.extend_from_slice(&[0x3F, 0x01, 0, 0, 0, 0, 0, 0]); // created at 319 ms
        expected.extend_from_slice(&[0x19; 32]); // salt
        expected.extend_from_slice(&[0x88; 28]); // key check
        expected.extend_from_slice(&[0; 16]); // reserved
        let crc = crc32fast::hash(&expected);
        expected.extend_from_slice(&crc.to_le_bytes());
        assert_eq!(bytes.to_vec(), expected);
        assert_eq!(
            crc, 0x150C_8AD8,
            "the checksum of the known header is pinned: {crc:#010x}"
        );
    }

    #[test]
    fn a_header_round_trips() {
        let header = known_header();
        assert_eq!(
            SegmentHeader::decode(&header.encode(), &path()).unwrap(),
            header
        );
        let plain = SegmentHeader {
            encrypted: false,
            salt: [0; SALT_BYTES],
            key_check: [0; KEY_CHECK_BYTES],
            ..known_header()
        };
        assert_eq!(
            SegmentHeader::decode(&plain.encode(), &path()).unwrap(),
            plain
        );
    }

    /// Sets `flags` in an encoded header and recomputes its checksum.
    fn with_flags(flags: u32) -> [u8; SEGMENT_HEADER_BYTES] {
        let mut bytes = known_header().encode();
        bytes[12..16].copy_from_slice(&flags.to_le_bytes());
        let crc = crc32fast::hash(&bytes[..CRC_OFFSET]);
        bytes[CRC_OFFSET..].copy_from_slice(&crc.to_le_bytes());
        bytes
    }

    #[test]
    fn unknown_incompatible_flags_are_refused() {
        let error =
            SegmentHeader::decode(&with_flags(FLAG_ENCRYPTED | 1 << 5), &path()).unwrap_err();
        assert!(
            matches!(error, WalError::UnsupportedSegment { .. }),
            "not damage: {error}"
        );
        let error = error.to_string();
        assert!(
            error.contains("bit 5") && error.contains("newer version"),
            "names the bit: {error}"
        );
        assert!(error.contains("wal_00000000000000001988.log"), "{error}");
        let error = SegmentHeader::decode(&with_flags(1 << 15), &path())
            .unwrap_err()
            .to_string();
        assert!(error.contains("bit 15"), "{error}");
        // A compatible flag this build does not know is ignored.
        let decoded =
            SegmentHeader::decode(&with_flags(FLAG_ENCRYPTED | 1 << 16 | 1 << 31), &path())
                .unwrap();
        assert!(decoded.encrypted, "the known flag still reads");
        let decoded = SegmentHeader::decode(&with_flags(1 << 20), &path()).unwrap();
        assert!(!decoded.encrypted);
    }

    #[test]
    fn a_damaged_header_is_refused_before_its_fields_are_read() {
        for index in 0..CRC_OFFSET + 4 {
            let mut bytes = known_header().encode();
            bytes[index] ^= 0x40;
            let error = SegmentHeader::decode(&bytes, &path()).unwrap_err();
            let expected = match index {
                0..8 => "magic",
                8..10 => "version",
                _ => "checksum",
            };
            // A flipped version byte reads as a later version: the version
            // is read before the checksum (see the next test).
            let damaged = !(8..10).contains(&index);
            assert_eq!(
                matches!(error, WalError::SegmentHeader { .. }),
                damaged,
                "byte {index}: {error}"
            );
            let error = error.to_string();
            assert!(error.contains(expected), "byte {index}: {error}");
        }
        // The reserved bytes are covered by the checksum but carry nothing.
        let error = SegmentHeader::decode(&known_header().encode()[..127], &path())
            .unwrap_err()
            .to_string();
        assert!(error.contains("127 bytes"), "{error}");
    }

    #[test]
    fn another_version_or_header_length_is_refused() {
        for (offset, value, expected, damaged) in [
            (8, 3u16, "newer version", false),
            (8, 1u16, "version 1", true),
            (10, 256u16, "header length 256", true),
        ] {
            let mut bytes = known_header().encode();
            bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
            let crc = crc32fast::hash(&bytes[..CRC_OFFSET]);
            bytes[CRC_OFFSET..].copy_from_slice(&crc.to_le_bytes());
            let error = SegmentHeader::decode(&bytes, &path()).unwrap_err();
            assert_eq!(
                matches!(error, WalError::SegmentHeader { .. }),
                damaged,
                "{expected}: {error}"
            );
            let error = error.to_string();
            assert!(error.contains(expected), "{expected}: {error}");
        }
    }

    /// A later version may lay its header out otherwise, its checksum
    /// elsewhere: the version is read first, so such a segment is reported
    /// as needing a newer Grafeo, not as damaged.
    #[test]
    fn a_newer_version_is_named_before_its_checksum_is_checked() {
        let mut bytes = known_header().encode();
        bytes[8..10].copy_from_slice(&3u16.to_le_bytes());
        bytes[CRC_OFFSET..].copy_from_slice(&[0x19, 0x88, 0x03, 0x19]);
        let error = SegmentHeader::decode(&bytes, &path())
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("version 3") && error.contains("newer version of Grafeo"),
            "{error}"
        );
        assert!(!error.contains("checksum"), "{error}");
    }

    #[test]
    fn the_key_check_data_is_taken_as_stored() {
        let mut bytes = known_header().encode();
        bytes[14] = 0x01; // compatible flag bit 16
        let aad = stored_key_check_aad(&bytes);
        assert_eq!(aad[..], bytes[..KEY_CHECK_AAD_BYTES]);
        assert_ne!(
            aad,
            known_header().key_check_aad(),
            "the decoded fields leave the compatible flag out"
        );
    }

    #[test]
    fn segment_names_carry_twenty_digits() {
        assert_eq!(segment_file_name(0), "wal_00000000000000000000.log");
        assert_eq!(segment_file_name(u64::MAX), "wal_18446744073709551615.log");
        assert_eq!(
            parse_segment_file_name("wal_00000000000000000319.log"),
            Some(319)
        );
        assert_eq!(
            parse_segment_file_name(&segment_file_name(u64::MAX)),
            Some(u64::MAX)
        );
        for name in [
            "wal_00000001.log",             // a 0.5.x segment
            "wal_0000000000000000031x.log", // not digits
            "wal_99999999999999999999.log", // past u64::MAX
            "wal_00000000000000000319.tmp",
            "checkpoint.meta",
        ] {
            assert_eq!(parse_segment_file_name(name), None, "{name}");
        }
    }

    #[test]
    fn a_segment_cut_off_during_creation_is_recognized() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join(segment_file_name(88));
        let mut zeros_then_a_frame_byte = vec![0u8; 3 * 64 * 1024 + 19];
        *zeros_then_a_frame_byte.last_mut().unwrap() = 0x03;
        for (contents, unfinished, what) in [
            (Vec::new(), true, "no byte"),
            (vec![0x47; 127], true, "part of a header"),
            (vec![0; 128], true, "a header of zeros"),
            (vec![0; 3 * 64 * 1024], true, "zeros past several reads"),
            (known_header().encode().to_vec(), false, "a header"),
            (
                zeros_then_a_frame_byte,
                false,
                "zeros with a byte behind them, three reads in",
            ),
        ] {
            std::fs::write(&file, &contents).unwrap();
            assert_eq!(is_unfinished_segment(&file).unwrap(), unfinished, "{what}");
        }
        assert!(header_is_blank(&[0; 128]) && header_is_blank(&[0x47; 127]));
        assert!(!header_is_blank(&known_header().encode()));
    }

    #[test]
    fn a_directory_listing_orders_segments_and_keeps_other_entries() {
        let dir = tempfile::tempdir().unwrap();
        for lsn in [88u64, 3, 1988] {
            std::fs::write(dir.path().join(segment_file_name(lsn)), b"").unwrap();
        }
        std::fs::write(dir.path().join("backup.cursor"), b"").unwrap();
        std::fs::write(dir.path().join("wal_00000001.log"), b"").unwrap();
        std::fs::create_dir(dir.path().join(segment_file_name(19))).unwrap();
        let listing = list_wal_directory(dir.path()).unwrap();
        let lsns: Vec<u64> = listing.segments.iter().map(|(lsn, _)| *lsn).collect();
        assert_eq!(lsns, [3, 88, 1988]);
        let others: Vec<String> = listing
            .others
            .iter()
            .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            others,
            [
                "backup.cursor",
                "wal_00000000000000000019.log",
                "wal_00000001.log"
            ],
            "a directory with a segment's name is not a segment"
        );
    }
}
