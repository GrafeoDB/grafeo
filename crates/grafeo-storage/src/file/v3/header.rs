//! File header and database headers of container format v3.
//!
//! Both are fixed little-endian encodings in a 4 KiB page, protected by a
//! CRC-32 over the used prefix. Byte 4 tells the formats apart: a 0.5.x file
//! header is bincode with varints, so byte 4 is the varint `0x01` of its
//! format version 1, while a v3 header has `03 00 00 00` there.

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use grafeo_common::utils::error::{Error, Result};

/// Container format version written by this module.
pub const FORMAT_V3: u32 = 3;
/// The format revision a new file gets. A file keeps the revision it was
/// created with: every checkpoint writes the revision of the header it
/// replaces.
pub const FORMAT_REVISION: u32 = 1;
/// The highest format revision this build reads.
pub const MAX_FORMAT_REVISION: u32 = 1;
// A build reads the revision it gives new files.
const _: () = assert!(FORMAT_REVISION >= 1 && FORMAT_REVISION <= MAX_FORMAT_REVISION);
/// Size of a page, and of each header, in bytes.
pub const PAGE_SIZE: u64 = 4096;
/// First page after the file header and the two database headers.
pub const DATA_START_PAGE: u64 = 3;

/// Byte offset of database header slot 0 or 1.
#[must_use]
pub fn slot_offset(slot: u8) -> u64 {
    PAGE_SIZE * (1 + u64::from(slot))
}

const PAGE_BYTES: usize = 4096;
const FILE_MAGIC: [u8; 4] = *b"GRAF";
const DB_MAGIC: [u8; 4] = *b"GDBH";
const FILE_CRC_OFFSET: usize = 72;
const DB_CRC_OFFSET: usize = 80;
/// Incompatible feature bit 0: the sections are encrypted.
const FLAG_ENCRYPTED: u32 = 1;
/// Bits 0 to 15 of the flags word: incompatible features.
const INCOMPATIBLE_FLAGS: u32 = 0x0000_FFFF;
/// The incompatible features this build can read.
const KNOWN_INCOMPATIBLE_FLAGS: u32 = FLAG_ENCRYPTED;

/// Names the set bits of `mask`, as `bit 2` or `bits 2, 15`.
fn describe_bits(mask: u32) -> String {
    let bits: Vec<String> = (0..32)
        .filter(|bit| mask & (1 << bit) != 0)
        .map(|bit: u32| bit.to_string())
        .collect();
    let noun = if bits.len() == 1 { "bit" } else { "bits" };
    format!("{noun} {}", bits.join(", "))
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

/// The 4 KiB file header at offset 0.
///
/// Layout (little-endian): `0 magic "GRAF"`, `4 format version u32` (3),
/// `8 page size u32` (4096), `12 flags u32`, `16 database id u128`,
/// `32 creation timestamp u64`, `40 creator version [u8; 32]`, `72 CRC-32 u32`
/// of bytes 0..72. The rest of the page is written as zero and ignored by
/// readers.
///
/// Flags: bits 0 to 15 are incompatible features, which change how the file
/// must be read, so a reader refuses a header that sets a bit there it does
/// not know. Bits 16 to 31 are compatible features, which a reader that does
/// not know them ignores. The only feature today is incompatible bit 0: the
/// sections are encrypted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileHeaderV3 {
    /// Page size in bytes; always 4096.
    pub page_size: u32,
    /// Whether the file's sections are encrypted.
    pub encrypted: bool,
    /// Random identifier of this database.
    pub database_id: u128,
    /// Creation time in milliseconds since the Unix epoch.
    pub creation_timestamp_ms: u64,
    /// Version of the library that created the file, zero padded.
    pub creator_version: [u8; 32],
}

impl FileHeaderV3 {
    /// Creates a header for a new database.
    #[must_use]
    pub fn new(encrypted: bool) -> Self {
        let mut creator_version = [0u8; 32];
        let version_bytes = env!("CARGO_PKG_VERSION").as_bytes();
        let copy_len = version_bytes.len().min(32);
        creator_version[..copy_len].copy_from_slice(&version_bytes[..copy_len]);
        Self {
            page_size: u32::try_from(PAGE_SIZE).unwrap_or(4096),
            encrypted,
            database_id: new_database_id(),
            creation_timestamp_ms: now_ms(),
            creator_version,
        }
    }

    /// Encodes the header into a 4 KiB page.
    #[must_use]
    pub fn encode(&self) -> [u8; PAGE_BYTES] {
        let mut page = [0u8; PAGE_BYTES];
        page[0..4].copy_from_slice(&FILE_MAGIC);
        page[4..8].copy_from_slice(&FORMAT_V3.to_le_bytes());
        page[8..12].copy_from_slice(&self.page_size.to_le_bytes());
        let flags = if self.encrypted { FLAG_ENCRYPTED } else { 0 };
        page[12..16].copy_from_slice(&flags.to_le_bytes());
        page[16..32].copy_from_slice(&self.database_id.to_le_bytes());
        page[32..40].copy_from_slice(&self.creation_timestamp_ms.to_le_bytes());
        page[40..72].copy_from_slice(&self.creator_version);
        let crc = crc32fast::hash(&page[..FILE_CRC_OFFSET]);
        page[FILE_CRC_OFFSET..FILE_CRC_OFFSET + 4].copy_from_slice(&crc.to_le_bytes());
        page
    }

    /// Decodes a file header, checking the magic, then the checksum, then
    /// the format version, the page size and the incompatible feature flags.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Serialization`] when the magic is wrong (not a Grafeo
    /// database), the format version is not 3, the page size is not 4096, or
    /// an incompatible feature flag this build does not know is set, and
    /// [`Error::Corruption`] at byte 0 when the page is too short for a
    /// header or its checksum is wrong.
    pub fn decode(page: &[u8]) -> Result<Self> {
        if page.get(0..4) != Some(FILE_MAGIC.as_slice()) {
            return Err(Error::Serialization(
                "invalid file header magic, not a Grafeo database".to_string(),
            ));
        }
        if page.len() < FILE_CRC_OFFSET + 4 {
            return Err(Error::corruption_at(
                format!(
                    "file header is {} bytes, expected at least {}",
                    page.len(),
                    FILE_CRC_OFFSET + 4
                ),
                0,
            ));
        }
        let stored = read_u32(page, FILE_CRC_OFFSET);
        let computed = crc32fast::hash(&page[..FILE_CRC_OFFSET]);
        if stored != computed {
            return Err(Error::corruption_at(
                format!(
                    "file header checksum mismatch: stored {stored:#010x}, computed \
                     {computed:#010x}"
                ),
                0,
            ));
        }
        let format = read_u32(page, 4);
        if format != FORMAT_V3 {
            return Err(Error::Serialization(format!(
                "unsupported file format version {format}, expected {FORMAT_V3}"
            )));
        }
        let page_size = read_u32(page, 8);
        if u64::from(page_size) != PAGE_SIZE {
            return Err(Error::Serialization(format!(
                "file header page size {page_size} is not supported, expected {PAGE_SIZE}"
            )));
        }
        let flags = read_u32(page, 12);
        let unknown = flags & INCOMPATIBLE_FLAGS & !KNOWN_INCOMPATIBLE_FLAGS;
        if unknown != 0 {
            return Err(Error::Serialization(format!(
                "file header sets unknown incompatible feature flags {unknown:#06x} ({}): \
                 the file needs a newer version of Grafeo",
                describe_bits(unknown)
            )));
        }
        let mut id = [0u8; 16];
        id.copy_from_slice(&page[16..32]);
        let mut creator_version = [0u8; 32];
        creator_version.copy_from_slice(&page[40..72]);
        Ok(Self {
            page_size,
            encrypted: flags & FLAG_ENCRYPTED != 0,
            database_id: u128::from_le_bytes(id),
            creation_timestamp_ms: read_u64(page, 32),
            creator_version,
        })
    }
}

/// A reference to a block in the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BlockRef {
    /// Byte offset of the block.
    pub offset: u64,
    /// Length of the block in bytes.
    pub length: u32,
    /// CRC-32 of the block.
    pub crc: u32,
}

/// One of the two alternating database headers, in the 4 KiB pages at
/// offsets 4096 (slot 0) and 8192 (slot 1).
///
/// Layout (little-endian): `0 magic "GDBH"`, `4 format revision u32`,
/// `8 iteration u64`, `16 checkpoint lsn u64`, `24 epoch u64`,
/// `32 last transaction id u64`, `40 root.offset u64`, `48 root.length u32`,
/// `52 root.crc u32`, `56 node count u64`, `64 edge count u64`,
/// `72 timestamp u64`, `80 CRC-32 u32` of bytes 0..80. The rest of the page
/// is written as zero and ignored by readers.
///
/// The format revision says which additions to container v3 the image may
/// use; see [`check_format_revision`] for what a reader accepts. The default
/// header has [`FORMAT_REVISION`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DbHeaderV3 {
    /// The format revision of the image (see [`check_format_revision`]).
    pub format_revision: u32,
    /// Monotonic write counter; the higher one is the active header.
    pub iteration: u64,
    /// WAL sequence number the checkpoint covers.
    pub checkpoint_lsn: u64,
    /// MVCC epoch at the checkpoint.
    pub epoch: u64,
    /// Last transaction id at the checkpoint.
    pub last_transaction_id: u64,
    /// The root (directory) block.
    pub root: BlockRef,
    /// Number of nodes at the checkpoint.
    pub node_count: u64,
    /// Number of edges at the checkpoint.
    pub edge_count: u64,
    /// Time of the checkpoint in milliseconds since the Unix epoch.
    pub timestamp_ms: u64,
}

impl Default for DbHeaderV3 {
    fn default() -> Self {
        Self {
            format_revision: FORMAT_REVISION,
            iteration: 0,
            checkpoint_lsn: 0,
            epoch: 0,
            last_transaction_id: 0,
            root: BlockRef::default(),
            node_count: 0,
            edge_count: 0,
            timestamp_ms: 0,
        }
    }
}

impl DbHeaderV3 {
    /// Encodes the header into a 4 KiB page.
    #[must_use]
    pub fn encode(&self) -> [u8; PAGE_BYTES] {
        let mut page = [0u8; PAGE_BYTES];
        page[0..4].copy_from_slice(&DB_MAGIC);
        page[4..8].copy_from_slice(&self.format_revision.to_le_bytes());
        page[8..16].copy_from_slice(&self.iteration.to_le_bytes());
        page[16..24].copy_from_slice(&self.checkpoint_lsn.to_le_bytes());
        page[24..32].copy_from_slice(&self.epoch.to_le_bytes());
        page[32..40].copy_from_slice(&self.last_transaction_id.to_le_bytes());
        page[40..48].copy_from_slice(&self.root.offset.to_le_bytes());
        page[48..52].copy_from_slice(&self.root.length.to_le_bytes());
        page[52..56].copy_from_slice(&self.root.crc.to_le_bytes());
        page[56..64].copy_from_slice(&self.node_count.to_le_bytes());
        page[64..72].copy_from_slice(&self.edge_count.to_le_bytes());
        page[72..80].copy_from_slice(&self.timestamp_ms.to_le_bytes());
        let crc = crc32fast::hash(&page[..DB_CRC_OFFSET]);
        page[DB_CRC_OFFSET..DB_CRC_OFFSET + 4].copy_from_slice(&crc.to_le_bytes());
        page
    }

    /// Decodes a database header slot.
    ///
    /// A slot whose bytes are all zero was never written
    /// ([`HeaderSlot::Empty`]). Any other slot without the magic and a
    /// matching checksum is [`HeaderSlot::Damaged`] (a torn or corrupted
    /// write), and so is a slot shorter than a header. The format revision
    /// comes back as stored: whether this build reads it is
    /// [`check_format_revision`]'s call, for the active header.
    #[must_use]
    pub fn decode(slot: &[u8]) -> HeaderSlot {
        if slot.len() < DB_CRC_OFFSET + 4 {
            return HeaderSlot::Damaged;
        }
        if slot.iter().all(|&byte| byte == 0) {
            return HeaderSlot::Empty;
        }
        if slot[0..4] != DB_MAGIC
            || read_u32(slot, DB_CRC_OFFSET) != crc32fast::hash(&slot[..DB_CRC_OFFSET])
        {
            return HeaderSlot::Damaged;
        }
        HeaderSlot::Valid(Self {
            format_revision: read_u32(slot, 4),
            iteration: read_u64(slot, 8),
            checkpoint_lsn: read_u64(slot, 16),
            epoch: read_u64(slot, 24),
            last_transaction_id: read_u64(slot, 32),
            root: BlockRef {
                offset: read_u64(slot, 40),
                length: read_u32(slot, 48),
                crc: read_u32(slot, 52),
            },
            node_count: read_u64(slot, 56),
            edge_count: read_u64(slot, 64),
            timestamp_ms: read_u64(slot, 72),
        })
    }
}

/// What a database header slot holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeaderSlot {
    /// Every byte is zero: the slot was never written.
    Empty,
    /// The slot holds bytes but no valid header: a torn or corrupted write.
    Damaged,
    /// A header whose magic and checksum are correct.
    Valid(DbHeaderV3),
}

/// Picks the active database header: the valid slot with the higher
/// iteration, slot 0 on a tie.
///
/// Without a valid slot, two empty slots are a file without a database
/// (`Ok(None)`). A damaged slot without a valid one is an error, not an empty
/// database: the image it pointed at may still be in the file, and the next
/// checkpoint would overwrite it. The file manager's `create` writes a valid
/// iteration-0 header, so an existing database file never has two empty
/// slots.
///
/// # Errors
///
/// Returns [`Error::Corruption`] at the first damaged slot when no slot is
/// valid and at least one is damaged.
pub fn active_header(slots: [HeaderSlot; 2]) -> Result<Option<(u8, DbHeaderV3)>> {
    use HeaderSlot::{Damaged, Empty, Valid};
    match slots {
        [Valid(first), Valid(second)] => Ok(Some(if second.iteration > first.iteration {
            (1, second)
        } else {
            (0, first)
        })),
        [Valid(first), _] => Ok(Some((0, first))),
        [_, Valid(second)] => Ok(Some((1, second))),
        [Empty, Empty] => Ok(None),
        [Damaged, Damaged] => Err(Error::corruption_at(
            "both database headers are damaged",
            slot_offset(0),
        )),
        [Damaged, Empty] => Err(Error::corruption_at(
            "database header slot 0 is damaged and slot 1 was never written",
            slot_offset(0),
        )),
        [Empty, Damaged] => Err(Error::corruption_at(
            "database header slot 1 is damaged and slot 0 was never written",
            slot_offset(1),
        )),
    }
}

/// Checks that this build reads an image of format `revision` (that of the
/// active database header): revisions 1 to [`MAX_FORMAT_REVISION`].
///
/// Revision 0 is the reserved field of a 0.6.0 development build from before
/// format revisions: no release wrote container v3 without one, so no
/// release reads it. A revision above the highest known was written by a
/// newer Grafeo; the file is left as it is.
///
/// # Errors
///
/// Returns [`Error::Serialization`] naming the revision for 0 and for one
/// above [`MAX_FORMAT_REVISION`].
pub fn check_format_revision(revision: u32) -> Result<()> {
    match revision {
        0 => Err(Error::Serialization(
            "the file has format revision 0: it was written by a 0.6.0 development build \
             before format revisions, which no release reads; recreate the database"
                .to_string(),
        )),
        1..=MAX_FORMAT_REVISION => Ok(()),
        _ => Err(Error::Serialization(format!(
            "the file has format revision {revision}, written by a newer version of Grafeo: \
             this version reads format revisions 1 to {MAX_FORMAT_REVISION}"
        ))),
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| {
            u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX)
        })
}

/// Generates a random 128-bit database id without extra dependencies.
///
/// Two `u64` come from `RandomState` (seeded by the OS), fed the current
/// time, the process id and a counter, so two calls in one process differ.
#[must_use]
pub fn new_database_id() -> u128 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let state = RandomState::new();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos());
    let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut parts = [0u64; 2];
    for (index, part) in parts.iter_mut().enumerate() {
        let mut hasher = state.build_hasher();
        hasher.write_u128(nanos);
        hasher.write_u32(std::process::id());
        hasher.write_u64(counter);
        hasher.write_usize(index);
        *part = hasher.finish();
    }
    (u128::from(parts[0]) << 64) | u128::from(parts[1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headers_round_trip() {
        let file = FileHeaderV3::new(true);
        assert_eq!(
            FileHeaderV3::decode(&file.encode()).unwrap().database_id,
            file.database_id
        );
        let db = DbHeaderV3 {
            format_revision: 3,
            iteration: 19,
            checkpoint_lsn: 88,
            epoch: 3,
            last_transaction_id: 319,
            root: BlockRef {
                offset: 12288,
                length: 4096,
                crc: 0xC0FFEE,
            },
            node_count: 1988,
            edge_count: 3,
            timestamp_ms: 1,
        };
        assert_eq!(DbHeaderV3::decode(&db.encode()), HeaderSlot::Valid(db));
    }

    #[test]
    fn a_torn_database_header_is_never_active() {
        let older = DbHeaderV3 {
            iteration: 3,
            ..Default::default()
        };
        let newer = DbHeaderV3 {
            iteration: 4,
            ..Default::default()
        };
        let mut torn = newer.encode();
        torn[20] ^= 0xFF; // a byte of the header changed after its CRC was taken
        let slots = [
            DbHeaderV3::decode(&older.encode()),
            DbHeaderV3::decode(&torn),
        ];
        assert_eq!(slots[1], HeaderSlot::Damaged, "the torn slot");
        assert_eq!(
            active_header(slots).unwrap(),
            Some((0, older)),
            "the older valid slot stays active"
        );
        assert_eq!(
            DbHeaderV3::decode(&[0u8; 4096]),
            HeaderSlot::Empty,
            "an empty slot"
        );
    }

    fn valid(iteration: u64, epoch: u64) -> DbHeaderV3 {
        DbHeaderV3 {
            iteration,
            epoch,
            ..Default::default()
        }
    }

    #[test]
    fn a_database_header_slot_is_empty_damaged_or_valid() {
        let header = valid(19, 88);
        assert_eq!(
            DbHeaderV3::decode(&header.encode()),
            HeaderSlot::Valid(header.clone())
        );
        assert_eq!(DbHeaderV3::decode(&[0u8; 4096]), HeaderSlot::Empty);
        let mut bad_magic = header.encode();
        bad_magic[0] = b'X';
        assert_eq!(
            DbHeaderV3::decode(&bad_magic),
            HeaderSlot::Damaged,
            "bad magic"
        );
        let mut stray = [0u8; 4096];
        stray[3000] = 3;
        assert_eq!(
            DbHeaderV3::decode(&stray),
            HeaderSlot::Damaged,
            "any non-zero byte without a header"
        );
        assert_eq!(
            DbHeaderV3::decode(&header.encode()[..40]),
            HeaderSlot::Damaged,
            "a slot shorter than a header"
        );
        let mut revised = header.encode();
        revised[4..8].copy_from_slice(&[3, 19, 88, 3]);
        let crc = crc32fast::hash(&revised[..80]);
        revised[80..84].copy_from_slice(&crc.to_le_bytes());
        assert_eq!(
            DbHeaderV3::decode(&revised),
            HeaderSlot::Valid(DbHeaderV3 {
                format_revision: 0x0358_1303,
                ..header
            }),
            "a slot keeps any format revision as stored: refusing one is not damage"
        );
    }

    #[test]
    fn the_valid_slot_with_the_higher_iteration_is_active_and_slot_0_wins_a_tie() {
        use HeaderSlot::{Empty, Valid};
        let cases = [
            ([Empty, Valid(valid(3, 0))], (1, valid(3, 0))),
            ([Valid(valid(3, 0)), Empty], (0, valid(3, 0))),
            ([Valid(valid(3, 0)), Valid(valid(4, 0))], (1, valid(4, 0))),
            ([Valid(valid(4, 0)), Valid(valid(3, 0))], (0, valid(4, 0))),
            (
                [Valid(valid(19, 3)), Valid(valid(19, 88))],
                (0, valid(19, 3)),
            ),
        ];
        for (slots, expected) in cases {
            let label = format!("{slots:?}");
            assert_eq!(active_header(slots).unwrap(), Some(expected), "{label}");
        }
    }

    #[test]
    fn two_empty_slots_are_a_file_without_a_database() {
        assert_eq!(
            active_header([HeaderSlot::Empty, HeaderSlot::Empty]).unwrap(),
            None
        );
    }

    #[test]
    fn damaged_slots_without_a_valid_one_are_an_error() {
        use HeaderSlot::{Damaged, Empty};
        let cases = [
            (
                [Damaged, Empty],
                "database header slot 0 is damaged and slot 1 was never written",
            ),
            (
                [Empty, Damaged],
                "database header slot 1 is damaged and slot 0 was never written",
            ),
            ([Damaged, Damaged], "both database headers are damaged"),
        ];
        for (slots, expected) in cases {
            let label = format!("{slots:?}");
            let error = active_header(slots).unwrap_err().to_string();
            assert!(error.contains(expected), "{label}: {error}");
        }
    }

    #[test]
    fn a_file_header_with_a_bad_crc_or_magic_is_refused() {
        let mut page = FileHeaderV3::new(false).encode();
        page[9] ^= 1;
        assert!(
            FileHeaderV3::decode(&page)
                .unwrap_err()
                .to_string()
                .contains("checksum")
        );
        let mut page = FileHeaderV3::new(false).encode();
        page[0] = b'X';
        assert!(
            FileHeaderV3::decode(&page)
                .unwrap_err()
                .to_string()
                .contains("magic")
        );
    }

    #[test]
    fn database_ids_differ() {
        assert_ne!(new_database_id(), new_database_id());
    }

    fn fixed_file_header(encrypted: bool) -> FileHeaderV3 {
        let mut creator_version = [0u8; 32];
        creator_version[..5].copy_from_slice(b"0.6.0");
        FileHeaderV3 {
            page_size: 4096,
            encrypted,
            database_id: 0x0102_0304_0506_0708_090A_0B0C_0D0E_0F10,
            creation_timestamp_ms: 0x0000_0188_0319_0388,
            creator_version,
        }
    }

    /// Recomputes the file header CRC after a test changed a field.
    fn reseal_file_header(page: &mut [u8; PAGE_BYTES]) {
        let crc = crc32fast::hash(&page[..72]);
        page[72..76].copy_from_slice(&crc.to_le_bytes());
    }

    #[test]
    fn the_file_header_has_its_documented_byte_layout() {
        let header = fixed_file_header(true);
        let page = header.encode();
        assert_eq!(&page[0..4], b"GRAF", "magic");
        assert_eq!(page[4..8], [3, 0, 0, 0], "format version");
        assert_eq!(page[8..12], [0x00, 0x10, 0x00, 0x00], "page size 4096");
        assert_eq!(page[12..16], [1, 0, 0, 0], "flags: bit 0 is encrypted");
        assert_eq!(
            page[16..32],
            [
                0x10, 0x0F, 0x0E, 0x0D, 0x0C, 0x0B, 0x0A, 0x09, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03,
                0x02, 0x01
            ],
            "database id, little-endian"
        );
        assert_eq!(
            page[32..40],
            [0x88, 0x03, 0x19, 0x03, 0x88, 0x01, 0x00, 0x00],
            "creation timestamp"
        );
        assert_eq!(&page[40..45], b"0.6.0", "creator version");
        assert!(page[45..72].iter().all(|&byte| byte == 0), "zero padded");
        assert_eq!(
            page[72..76],
            crc32fast::hash(&page[..72]).to_le_bytes(),
            "CRC over bytes 0..72"
        );
        assert!(page[76..].iter().all(|&byte| byte == 0), "rest is zero");
        assert_eq!(FileHeaderV3::decode(&page).unwrap(), header);
        let plain = fixed_file_header(false).encode();
        assert_eq!(plain[12..16], [0, 0, 0, 0], "no flags without encryption");
    }

    #[test]
    fn the_database_header_has_its_documented_byte_layout() {
        let header = DbHeaderV3 {
            format_revision: 1,
            iteration: 3,
            checkpoint_lsn: 19,
            epoch: 88,
            last_transaction_id: 319,
            root: BlockRef {
                offset: 0x0000_0003_1988_3000,
                length: 0x0001_0000,
                crc: 0x1988_0319,
            },
            node_count: 1988,
            edge_count: 8819,
            timestamp_ms: 0x0000_0188_0319_0388,
        };
        let page = header.encode();
        assert_eq!(&page[0..4], b"GDBH", "magic");
        assert_eq!(page[4..8], [1, 0, 0, 0], "format revision");
        assert_eq!(page[8..16], [3, 0, 0, 0, 0, 0, 0, 0], "iteration");
        assert_eq!(page[16..24], [19, 0, 0, 0, 0, 0, 0, 0], "checkpoint lsn");
        assert_eq!(page[24..32], [88, 0, 0, 0, 0, 0, 0, 0], "epoch");
        assert_eq!(
            page[32..40],
            [0x3F, 0x01, 0, 0, 0, 0, 0, 0],
            "last transaction id"
        );
        assert_eq!(
            page[40..48],
            [0x00, 0x30, 0x88, 0x19, 0x03, 0x00, 0x00, 0x00],
            "root offset"
        );
        assert_eq!(page[48..52], [0x00, 0x00, 0x01, 0x00], "root length");
        assert_eq!(page[52..56], [0x19, 0x03, 0x88, 0x19], "root crc");
        assert_eq!(page[56..64], [0xC4, 0x07, 0, 0, 0, 0, 0, 0], "node count");
        assert_eq!(page[64..72], [0x73, 0x22, 0, 0, 0, 0, 0, 0], "edge count");
        assert_eq!(
            page[72..80],
            [0x88, 0x03, 0x19, 0x03, 0x88, 0x01, 0x00, 0x00],
            "timestamp"
        );
        assert_eq!(
            page[80..84],
            crc32fast::hash(&page[..80]).to_le_bytes(),
            "CRC over bytes 0..80"
        );
        assert!(page[84..].iter().all(|&byte| byte == 0), "rest is zero");
    }

    #[test]
    fn a_new_header_has_the_current_format_revision() {
        assert_eq!(DbHeaderV3::default().format_revision, FORMAT_REVISION);
        assert_eq!(FORMAT_REVISION, 1, "0.6.0 writes revision 1");
    }

    #[test]
    fn this_build_reads_format_revisions_1_to_the_highest_it_knows() {
        check_format_revision(1).unwrap();
        check_format_revision(MAX_FORMAT_REVISION).unwrap();
        let error = check_format_revision(0).unwrap_err().to_string();
        assert!(
            error.contains("format revision 0") && error.contains("development build"),
            "{error}"
        );
        for newer in [MAX_FORMAT_REVISION + 1, u32::MAX] {
            let error = check_format_revision(newer).unwrap_err().to_string();
            assert!(
                error.contains(&format!("format revision {newer}"))
                    && error.contains("newer version")
                    && error.contains(&format!("1 to {MAX_FORMAT_REVISION}")),
                "{error}"
            );
        }
    }

    #[test]
    fn an_unknown_incompatible_feature_flag_is_refused() {
        for (bit, byte, mask) in [(2u32, 12usize, 0b100u8), (15, 13, 0x80)] {
            let mut page = FileHeaderV3::new(true).encode();
            page[byte] |= mask;
            reseal_file_header(&mut page);
            let error = FileHeaderV3::decode(&page).unwrap_err().to_string();
            let named = format!("{:#06x}", 1u32 << bit);
            assert!(
                error.contains(&named) && error.contains(&format!("bit {bit}")),
                "the error names bit {bit}: {error}"
            );
        }
    }

    #[test]
    fn an_unknown_compatible_feature_flag_is_ignored() {
        let header = fixed_file_header(true);
        let mut page = header.encode();
        page[14] |= 1; // bit 16
        page[15] |= 0x80; // bit 31
        reseal_file_header(&mut page);
        assert_eq!(FileHeaderV3::decode(&page).unwrap(), header);
    }

    #[test]
    fn a_page_size_other_than_4096_is_refused() {
        for page_size in [0u32, 8192] {
            let mut page = FileHeaderV3::new(false).encode();
            page[8..12].copy_from_slice(&page_size.to_le_bytes());
            reseal_file_header(&mut page);
            let error = FileHeaderV3::decode(&page).unwrap_err().to_string();
            assert!(
                error.contains(&format!("page size {page_size}")),
                "the error names the page size: {error}"
            );
        }
    }
}
