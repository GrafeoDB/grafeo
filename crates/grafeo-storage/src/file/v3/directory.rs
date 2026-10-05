//! Chained directory of the v3 container.
//!
//! Every chunk of a file is listed in a directory of fixed 48-byte entries.
//! The directory is stored in blocks of at most 64 KiB; each block names the
//! next one, so the number of chunks is unbounded. The database header points
//! at the first block. The functions here are pure: the caller decides where
//! blocks go and does the I/O.
//!
//! Block layout (little-endian): `0 magic "GDIR"`, `4 entry_count u32`,
//! `8 next.offset u64`, `16 next.length u32`, `20 next.crc u32`,
//! `24 reserved u64`, then the entries. `next.length == 0` ends the chain. The
//! reserved field is written as zero and ignored by readers. The CRC of a
//! block is kept by whoever points at it (the database header or the previous
//! block).

use std::collections::HashSet;

use grafeo_common::storage::{ChunkKind, ChunkMeta, SectionType};
use grafeo_common::utils::error::{Error, Result};

use super::alloc::PageRun;
use super::header::{BlockRef, DATA_START_PAGE, PAGE_SIZE};

/// Size of one encoded directory entry in bytes.
pub const ENTRY_SIZE: usize = 48;
/// Size of the header of a directory block in bytes.
pub const BLOCK_HEADER_SIZE: usize = 32;
/// Largest size of a directory block in bytes.
pub const MAX_BLOCK_SIZE: usize = 64 * 1024;
/// Largest number of entries in one directory block.
pub const ENTRIES_PER_BLOCK: usize = (MAX_BLOCK_SIZE - BLOCK_HEADER_SIZE) / ENTRY_SIZE;

/// Magic bytes at the start of every directory block.
const BLOCK_MAGIC: [u8; 4] = *b"GDIR";

/// One chunk of the file: what it holds and where it is stored.
///
/// Layout (48 bytes, little-endian): `0 section type u8`,
/// `1 section version u8`, `2 chunk kind u8`, `3 codec u8`, `4 graph id u32`,
/// `8 column id u32`, `12 row count u32`, `16 row start u64`, `24 offset u64`,
/// `32 length u64`, `40 crc u32`, `44 reserved u32`. The reserved field is
/// written as zero and ignored by readers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectoryEntry {
    /// Section the chunk belongs to.
    pub section_type: SectionType,
    /// Format version of the section's bytes.
    pub section_version: u8,
    /// What the chunk holds, as its section describes it.
    pub meta: ChunkMeta,
    /// Byte offset of the chunk in the file (page aligned).
    pub offset: u64,
    /// Length of the stored chunk in bytes.
    pub length: u64,
    /// CRC-32 of the stored chunk.
    pub crc: u32,
}

impl DirectoryEntry {
    /// Encodes the entry into its 48-byte form.
    pub fn encode(&self, out: &mut [u8; ENTRY_SIZE]) {
        out.fill(0);
        out[0] = self.section_type.to_u8();
        out[1] = self.section_version;
        out[2] = self.meta.kind.to_byte();
        out[3] = self.meta.codec;
        out[4..8].copy_from_slice(&self.meta.graph_id.to_le_bytes());
        out[8..12].copy_from_slice(&self.meta.column_id.to_le_bytes());
        out[12..16].copy_from_slice(&self.meta.row_count.to_le_bytes());
        out[16..24].copy_from_slice(&self.meta.row_start.to_le_bytes());
        out[24..32].copy_from_slice(&self.offset.to_le_bytes());
        out[32..40].copy_from_slice(&self.length.to_le_bytes());
        out[40..44].copy_from_slice(&self.crc.to_le_bytes());
    }

    /// Decodes an entry, refusing an unknown section type or chunk kind.
    ///
    /// # Errors
    ///
    /// Returns an error when the section type byte or the chunk kind byte is
    /// not known to this build.
    pub fn decode(bytes: &[u8; ENTRY_SIZE]) -> Result<Self> {
        let section_type = SectionType::from_u8(bytes[0]).ok_or_else(|| {
            Error::Serialization(format!(
                "directory entry has unknown section type {}",
                bytes[0]
            ))
        })?;
        let kind = ChunkKind::from_byte(bytes[2]).ok_or_else(|| {
            Error::Serialization(format!(
                "directory entry has unknown chunk kind {}",
                bytes[2]
            ))
        })?;
        Ok(Self {
            section_type,
            section_version: bytes[1],
            meta: ChunkMeta {
                kind,
                codec: bytes[3],
                graph_id: u32_at(bytes, 4),
                column_id: u32_at(bytes, 8),
                row_count: u32_at(bytes, 12),
                row_start: u64_at(bytes, 16),
            },
            offset: u64_at(bytes, 24),
            length: u64_at(bytes, 32),
            crc: u32_at(bytes, 40),
        })
    }

    /// Pages occupied by the chunk (a zero-length chunk has an empty run).
    #[must_use]
    pub fn run(&self) -> PageRun {
        PageRun {
            first: self.offset / PAGE_SIZE,
            count: PageRun::for_bytes(self.length),
        }
    }
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    let mut array = [0u8; 4];
    array.copy_from_slice(&bytes[at..at + 4]);
    u32::from_le_bytes(array)
}

fn u64_at(bytes: &[u8], at: usize) -> u64 {
    let mut array = [0u8; 8];
    array.copy_from_slice(&bytes[at..at + 8]);
    u64::from_le_bytes(array)
}

/// Encodes one block holding `entries` and naming `next`.
fn encode_block(entries: &[DirectoryEntry], next: BlockRef) -> Result<Vec<u8>> {
    let count = u32::try_from(entries.len())
        .map_err(|_| Error::Internal("directory block has too many entries".to_string()))?;
    let mut bytes = vec![0u8; BLOCK_HEADER_SIZE + entries.len() * ENTRY_SIZE];
    bytes[0..4].copy_from_slice(&BLOCK_MAGIC);
    bytes[4..8].copy_from_slice(&count.to_le_bytes());
    bytes[8..16].copy_from_slice(&next.offset.to_le_bytes());
    bytes[16..20].copy_from_slice(&next.length.to_le_bytes());
    bytes[20..24].copy_from_slice(&next.crc.to_le_bytes());
    let (slots, _) = bytes[BLOCK_HEADER_SIZE..].as_chunks_mut::<ENTRY_SIZE>();
    for (entry, slot) in entries.iter().zip(slots) {
        entry.encode(slot);
    }
    Ok(bytes)
}

/// Encodes `entries` into blocks of at most [`ENTRIES_PER_BLOCK`], each block
/// naming the next one.
///
/// `place(len)` returns where a block of `len` bytes goes; blocks are placed
/// last to first. Returns the reference to the first block (the root) and the
/// blocks as `(offset, bytes)` in chain order. The bytes are not padded to
/// pages.
///
/// An empty `entries` still produces one block, holding zero entries, so
/// every image has a directory block to read (and, when encrypted, to
/// decrypt) when it is opened.
///
/// # Errors
///
/// Returns an error when `place` fails.
pub fn encode_blocks(
    entries: &[DirectoryEntry],
    mut place: impl FnMut(usize) -> Result<u64>,
) -> Result<(BlockRef, Vec<(u64, Vec<u8>)>)> {
    let groups: Vec<&[DirectoryEntry]> = if entries.is_empty() {
        vec![&[]]
    } else {
        entries.chunks(ENTRIES_PER_BLOCK).collect()
    };
    let mut next = BlockRef::default();
    let mut blocks = Vec::new();
    for group in groups.into_iter().rev() {
        let bytes = encode_block(group, next)?;
        let offset = place(bytes.len())?;
        next = BlockRef {
            offset,
            length: u32::try_from(bytes.len())
                .map_err(|_| Error::Internal("directory block is too large".to_string()))?,
            crc: crc32fast::hash(&bytes),
        };
        blocks.push((offset, bytes));
    }
    blocks.reverse();
    Ok((next, blocks))
}

/// Checks a block pointer before it is read, so a corrupt pointer never
/// allocates a huge buffer.
fn check_pointer(pointer: BlockRef) -> Result<()> {
    let offset = pointer.offset;
    if !offset.is_multiple_of(PAGE_SIZE) {
        return Err(Error::Serialization(format!(
            "directory block at offset {offset} is not page aligned"
        )));
    }
    if offset < DATA_START_PAGE * PAGE_SIZE {
        return Err(Error::Serialization(format!(
            "directory block at offset {offset} lies before the data area"
        )));
    }
    let length = pointer.length as usize;
    if !(BLOCK_HEADER_SIZE..=MAX_BLOCK_SIZE).contains(&length)
        || !(length - BLOCK_HEADER_SIZE).is_multiple_of(ENTRY_SIZE)
    {
        return Err(Error::Serialization(format!(
            "directory block at offset {offset} has invalid length {length}"
        )));
    }
    Ok(())
}

/// Follows the chain from `root`, `read(offset, len)` returning the stored
/// bytes of a block.
///
/// Returns the entries in order and the pages of the blocks themselves. A
/// root with length 0 is accepted as an empty directory, although
/// [`encode_blocks`] never produces one.
///
/// # Errors
///
/// Returns an error naming the block offset when a pointer is misaligned, out
/// of range or revisited, or when a block fails its CRC, magic or length
/// checks. Errors from `read` are passed through.
pub fn decode_chain(
    root: BlockRef,
    mut read: impl FnMut(u64, u32) -> Result<Vec<u8>>,
) -> Result<(Vec<DirectoryEntry>, Vec<PageRun>)> {
    let mut entries = Vec::new();
    let mut runs = Vec::new();
    if root.length == 0 {
        return Ok((entries, runs));
    }
    let mut visited = HashSet::new();
    let mut pointer = root;
    loop {
        let offset = pointer.offset;
        if !visited.insert(offset) {
            return Err(Error::Serialization(format!(
                "directory chain revisits the block at offset {offset}"
            )));
        }
        check_pointer(pointer)?;
        let bytes = read(offset, pointer.length)?;
        if bytes.len() != pointer.length as usize {
            return Err(Error::Serialization(format!(
                "directory block at offset {offset} read {} bytes, expected {}",
                bytes.len(),
                pointer.length
            )));
        }
        if crc32fast::hash(&bytes) != pointer.crc {
            return Err(Error::Serialization(format!(
                "directory block at offset {offset} fails its checksum"
            )));
        }
        if bytes[0..4] != BLOCK_MAGIC {
            return Err(Error::Serialization(format!(
                "directory block at offset {offset} has a bad magic"
            )));
        }
        let count = u32_at(&bytes, 4) as usize;
        if count > ENTRIES_PER_BLOCK || BLOCK_HEADER_SIZE + count * ENTRY_SIZE != bytes.len() {
            return Err(Error::Serialization(format!(
                "directory block at offset {offset} declares {count} entries but holds {} bytes",
                bytes.len()
            )));
        }
        let (slots, _) = bytes[BLOCK_HEADER_SIZE..].as_chunks::<ENTRY_SIZE>();
        for slot in slots {
            entries.push(DirectoryEntry::decode(slot).map_err(|error| {
                Error::Serialization(format!("directory block at offset {offset}: {error}"))
            })?);
        }
        runs.push(PageRun {
            first: offset / PAGE_SIZE,
            count: PageRun::for_bytes(u64::from(pointer.length)),
        });
        pointer = BlockRef {
            offset: u64_at(&bytes, 8),
            length: u32_at(&bytes, 16),
            crc: u32_at(&bytes, 20),
        };
        if pointer.length == 0 {
            return Ok((entries, runs));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn len32(length: usize) -> u32 {
        u32::try_from(length).unwrap()
    }

    fn entry(i: u64) -> DirectoryEntry {
        DirectoryEntry {
            section_type: SectionType::LpgStore,
            section_version: 1,
            meta: ChunkMeta {
                row_start: i * 65_536,
                row_count: 65_536,
                ..ChunkMeta::raw()
            },
            offset: (3 + i) * 4096,
            length: 4096,
            crc: u32::try_from(i).unwrap(),
        }
    }

    #[test]
    fn a_directory_longer_than_one_block_round_trips_through_the_chain() {
        let entries: Vec<_> = (0..(ENTRIES_PER_BLOCK as u64 * 2 + 19))
            .map(entry)
            .collect();
        let mut next = 245 * PAGE_SIZE;
        let (root, blocks) = encode_blocks(&entries, |len| {
            let at = next;
            next += PageRun::for_bytes(len as u64) * PAGE_SIZE;
            Ok(at)
        })
        .unwrap();
        assert_eq!(blocks.len(), 3);
        let stored: std::collections::HashMap<u64, Vec<u8>> = blocks.into_iter().collect();
        let (back, runs) = decode_chain(root, |offset, len| {
            let b = &stored[&offset];
            assert_eq!(b.len(), len as usize);
            Ok(b.clone())
        })
        .unwrap();
        assert_eq!(back, entries);
        assert_eq!(runs.len(), 3);
    }

    #[test]
    fn a_damaged_block_fails_with_its_offset() {
        let entries = vec![entry(0)];
        let (root, mut blocks) = encode_blocks(&entries, |_| Ok(12_288)).unwrap();
        blocks[0].1[40] ^= 1;
        let error = decode_chain(root, |_, _| Ok(blocks[0].1.clone()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("12288"), "{error}");
    }

    #[test]
    fn an_unknown_section_type_or_chunk_kind_is_refused() {
        let mut bytes = [0u8; ENTRY_SIZE];
        entry(0).encode(&mut bytes);
        bytes[0] = 250;
        assert!(DirectoryEntry::decode(&bytes).is_err());
        entry(0).encode(&mut bytes);
        bytes[2] = 250;
        assert!(DirectoryEntry::decode(&bytes).is_err());
    }

    #[test]
    fn an_entry_round_trips_and_names_its_pages() {
        let original = entry(19);
        let mut bytes = [0u8; ENTRY_SIZE];
        original.encode(&mut bytes);
        assert_eq!(DirectoryEntry::decode(&bytes).unwrap(), original);
        assert_eq!(
            original.run(),
            PageRun {
                first: 22,
                count: 1
            }
        );
        let empty = DirectoryEntry {
            length: 0,
            ..original
        };
        assert_eq!(empty.run().count, 0);
    }

    #[test]
    fn an_unaligned_or_early_pointer_is_refused_before_reading() {
        for offset in [12_289u64, 4096] {
            let root = BlockRef {
                offset,
                length: 80,
                crc: 0,
            };
            let error = decode_chain(root, |_, _| panic!("must not read"))
                .unwrap_err()
                .to_string();
            assert!(error.contains(&offset.to_string()), "{error}");
        }
    }

    #[test]
    fn an_oversized_or_misshapen_block_length_is_refused_before_reading() {
        for length in [
            u32::MAX,
            len32(MAX_BLOCK_SIZE + 1),
            len32(BLOCK_HEADER_SIZE - 1),
            len32(BLOCK_HEADER_SIZE + 5),
        ] {
            let root = BlockRef {
                offset: 12_288,
                length,
                crc: 0,
            };
            let error = decode_chain(root, |_, _| panic!("must not read"))
                .unwrap_err()
                .to_string();
            assert!(error.contains("12288"), "{error}");
        }
    }

    #[test]
    fn a_chain_that_revisits_a_block_is_corruption() {
        let (first, second) = (12_288u64, 16_384u64);
        let tail = encode_block(
            &[],
            BlockRef {
                offset: first,
                length: 80,
                crc: 0,
            },
        )
        .unwrap();
        let head = encode_block(
            &[entry(0)],
            BlockRef {
                offset: second,
                length: len32(tail.len()),
                crc: crc32fast::hash(&tail),
            },
        )
        .unwrap();
        let root = BlockRef {
            offset: first,
            length: len32(head.len()),
            crc: crc32fast::hash(&head),
        };
        let error = decode_chain(root, |offset, _| {
            Ok(if offset == first {
                head.clone()
            } else {
                tail.clone()
            })
        })
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("12288") && error.contains("revisits"),
            "{error}"
        );
    }

    fn fixed_entry() -> DirectoryEntry {
        DirectoryEntry {
            section_type: SectionType::RdfStore,
            section_version: 19,
            meta: ChunkMeta {
                kind: ChunkKind::Raw,
                codec: 88,
                graph_id: 0x0102_0304,
                column_id: 0x0506_0708,
                row_count: 0x090A_0B0C,
                row_start: 0x1112_1314_1516_1718,
            },
            offset: 0x0000_0003_1988_3000,
            length: 0x0000_0001_0000_2328,
            crc: 0x1988_0319,
        }
    }

    #[test]
    fn a_directory_entry_has_its_documented_byte_layout() {
        let entry = fixed_entry();
        let mut bytes = [0xAA; ENTRY_SIZE];
        entry.encode(&mut bytes);
        assert_eq!(bytes[0], 3, "section type byte");
        assert_eq!(bytes[1], 19, "section version");
        assert_eq!(bytes[2], 0, "chunk kind byte");
        assert_eq!(bytes[3], 88, "codec");
        assert_eq!(bytes[4..8], [0x04, 0x03, 0x02, 0x01], "graph id");
        assert_eq!(bytes[8..12], [0x08, 0x07, 0x06, 0x05], "column id");
        assert_eq!(bytes[12..16], [0x0C, 0x0B, 0x0A, 0x09], "row count");
        assert_eq!(
            bytes[16..24],
            [0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11],
            "row start"
        );
        assert_eq!(
            bytes[24..32],
            [0x00, 0x30, 0x88, 0x19, 0x03, 0x00, 0x00, 0x00],
            "offset"
        );
        assert_eq!(
            bytes[32..40],
            [0x28, 0x23, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00],
            "length"
        );
        assert_eq!(bytes[40..44], [0x19, 0x03, 0x88, 0x19], "crc");
        assert_eq!(bytes[44..48], [0, 0, 0, 0], "reserved, written as zero");
        bytes[44..48].copy_from_slice(&[3, 19, 88, 1]);
        assert_eq!(
            DirectoryEntry::decode(&bytes).unwrap(),
            entry,
            "readers ignore the reserved bytes"
        );
    }

    #[test]
    fn a_directory_block_header_has_its_documented_byte_layout() {
        let next = BlockRef {
            offset: 0x0000_0003_1988_3000,
            length: 0x0000_0050,
            crc: 0xC0FF_EE19,
        };
        let mut block = encode_block(&[fixed_entry()], next).unwrap();
        assert_eq!(block.len(), 80, "header and one entry");
        assert_eq!(&block[0..4], b"GDIR", "magic");
        assert_eq!(block[4..8], [1, 0, 0, 0], "entry count");
        assert_eq!(
            block[8..16],
            [0x00, 0x30, 0x88, 0x19, 0x03, 0x00, 0x00, 0x00],
            "next offset"
        );
        assert_eq!(block[16..20], [0x50, 0, 0, 0], "next length");
        assert_eq!(block[20..24], [0x19, 0xEE, 0xFF, 0xC0], "next crc");
        assert_eq!(block[24..32], [0; 8], "reserved, written as zero");
        let mut entry = [0u8; ENTRY_SIZE];
        fixed_entry().encode(&mut entry);
        assert_eq!(block[32..80], entry, "the entries follow the header");

        let mut tail = encode_block(&[], BlockRef::default()).unwrap();
        tail[24..32].copy_from_slice(&[3, 19, 88, 3, 19, 88, 3, 19]);
        block[8..16].copy_from_slice(&16_384u64.to_le_bytes());
        block[16..20].copy_from_slice(&len32(tail.len()).to_le_bytes());
        block[20..24].copy_from_slice(&crc32fast::hash(&tail).to_le_bytes());
        block[24..32].copy_from_slice(&[88; 8]);
        let root = BlockRef {
            offset: 12_288,
            length: len32(block.len()),
            crc: crc32fast::hash(&block),
        };
        let (entries, _) = decode_chain(root, |offset, _| {
            Ok(if offset == 12_288 {
                block.clone()
            } else {
                tail.clone()
            })
        })
        .unwrap();
        assert_eq!(
            entries,
            [fixed_entry()],
            "readers ignore the reserved bytes"
        );
    }

    #[test]
    fn an_empty_entry_list_still_encodes_one_block() {
        let mut placed = Vec::new();
        let (root, blocks) = encode_blocks(&[], |length| {
            placed.push(length);
            Ok(12_288)
        })
        .unwrap();
        assert_eq!(
            placed,
            [BLOCK_HEADER_SIZE],
            "one block holding only its header"
        );
        assert_eq!(blocks.len(), 1);
        assert_eq!(
            root,
            BlockRef {
                offset: 12_288,
                length: len32(BLOCK_HEADER_SIZE),
                crc: crc32fast::hash(&blocks[0].1),
            }
        );
        let (entries, runs) = decode_chain(root, |offset, _| {
            assert_eq!(offset, 12_288);
            Ok(blocks[0].1.clone())
        })
        .unwrap();
        assert!(entries.is_empty(), "a block without entries");
        assert_eq!(runs, [PageRun { first: 3, count: 1 }], "the block's page");
    }

    #[test]
    fn a_zero_length_root_still_decodes_as_a_directory_without_chunks() {
        let (entries, runs) =
            decode_chain(BlockRef::default(), |_, _| panic!("must not read")).unwrap();
        assert!(entries.is_empty() && runs.is_empty());
    }
}
