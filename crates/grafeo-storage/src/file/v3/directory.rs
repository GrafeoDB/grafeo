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
//!
//! An entry's flags byte says whether a reader that does not know its section
//! type ([`ENTRY_SECTION_OPTIONAL`]) or its chunk kind
//! ([`ENTRY_CHUNK_OPTIONAL`]) may skip it. Such an entry decodes as a
//! [`SkippedEntry`]; an unknown entry without its bit is refused.

use std::collections::HashSet;

use grafeo_common::storage::{ChunkKind, ChunkMeta, ChunkNamespace, SectionType};
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

/// Flag bit 0: a reader that does not know the entry's section type skips the entry.
pub const ENTRY_SECTION_OPTIONAL: u8 = 0x01;
/// Flag bit 1: a reader that knows the section type but not the chunk kind skips the entry.
pub const ENTRY_CHUNK_OPTIONAL: u8 = 0x02;
/// Bits 0 to 3 change how an entry is read, so a reader refuses one it does not know; bits 4 to
/// 7 do not, and a reader ignores them (the same split as the file header's feature flags).
pub const ENTRY_INCOMPATIBLE_FLAGS: u8 = 0x0F;
/// The flag bits among bits 0 to 3 this version knows.
const KNOWN_INCOMPATIBLE_FLAGS: u8 = ENTRY_SECTION_OPTIONAL | ENTRY_CHUNK_OPTIONAL;

/// Byte of an entry that holds its flags.
const FLAGS_AT: usize = 44;
/// Byte of an entry that holds its chunk's namespace.
const NAMESPACE_AT: usize = 45;

/// Magic bytes at the start of every directory block.
const BLOCK_MAGIC: [u8; 4] = *b"GDIR";

/// One chunk of the file: what it holds and where it is stored.
///
/// Layout (48 bytes, little-endian): `0 section type u8`,
/// `1 section version u8`, `2 chunk kind u8`, `3 codec u8`, `4 graph id u32`,
/// `8 column id u32`, `12 row count u32`, `16 row start u64`, `24 offset u64`,
/// `32 length u64`, `40 crc u32`, `44 flags u8`, `45 namespace u8`,
/// `46 reserved [u8; 2]`. The reserved bytes are written as zero and ignored
/// by readers. The identity of a chunk, unique within its section, is (chunk
/// kind, namespace, graph id, column id, row start).
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
    /// [`ENTRY_SECTION_OPTIONAL`] and [`ENTRY_CHUNK_OPTIONAL`], which tell a
    /// reader that does not know the section type or the chunk kind whether
    /// it may skip the entry; see [`flags_for`](Self::flags_for). A decoded
    /// entry keeps the byte as read, bits 4 to 7 included.
    pub flags: u8,
}

impl DirectoryEntry {
    /// The flags a writer sets for a chunk: bit 0 from `section_type.is_optional()`, bit 1 from
    /// `kind.is_optional()`.
    #[must_use]
    pub const fn flags_for(section_type: SectionType, kind: ChunkKind) -> u8 {
        let mut flags = 0;
        if section_type.is_optional() {
            flags |= ENTRY_SECTION_OPTIONAL;
        }
        if kind.is_optional() {
            flags |= ENTRY_CHUNK_OPTIONAL;
        }
        flags
    }

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
        out[FLAGS_AT] = self.flags;
        out[NAMESPACE_AT] = self.meta.namespace.to_byte();
    }

    /// Decodes an entry. An unknown section type or chunk kind is `Skipped` when its bit is
    /// set, else an error naming it with the word "required"; the section bit never covers an
    /// unknown kind of a known section. An unknown flag among bits 0 to 3 is an error.
    ///
    /// The flags are checked first, so an entry with an unknown flag among
    /// bits 0 to 3 is refused also when it would be skipped. Bits 4 to 7 are
    /// ignored.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Serialization`] when the flags hold a bit among 0 to
    /// 3 this version does not know, or when the section type byte, or the
    /// chunk kind byte of a known section, is not known to this version and
    /// the entry's optional bit for it is not set. The message names the byte
    /// as stored.
    pub fn decode(bytes: &[u8; ENTRY_SIZE]) -> Result<DecodedEntry> {
        let flags = bytes[FLAGS_AT];
        let unknown = flags & ENTRY_INCOMPATIBLE_FLAGS & !KNOWN_INCOMPATIBLE_FLAGS;
        if unknown != 0 {
            return Err(Error::Serialization(format!(
                "directory entry has flags {flags:#04x}, with the unknown flag bits \
                 {unknown:#04x}: they change how the entry is read, and this version does \
                 not know them"
            )));
        }
        let (type_byte, kind_byte) = (bytes[0], bytes[2]);
        let skipped = || {
            DecodedEntry::Skipped(SkippedEntry {
                section_type: type_byte,
                kind: kind_byte,
                flags,
                offset: u64_at(bytes, 24),
                length: u64_at(bytes, 32),
            })
        };
        let Some(section_type) = SectionType::from_u8(type_byte) else {
            if flags & ENTRY_SECTION_OPTIONAL != 0 {
                return Ok(skipped());
            }
            return Err(Error::Serialization(format!(
                "directory entry has section type {type_byte}, which this version does not \
                 know, and the section is required (its optional bit is not set): the file \
                 was written by a newer version"
            )));
        };
        let Some(kind) = ChunkKind::from_byte(kind_byte) else {
            if flags & ENTRY_CHUNK_OPTIONAL != 0 {
                return Ok(skipped());
            }
            return Err(Error::Serialization(format!(
                "directory entry of section {section_type:?} has chunk kind {kind_byte}, which \
                 this version does not know, and the chunk is required (its optional bit is \
                 not set): the file was written by a newer version"
            )));
        };
        // Every namespace a file of a known revision can hold is known (a
        // newer one comes with a newer revision, refused before the
        // directory is read), so an unknown byte is damage, not news.
        let namespace_byte = bytes[NAMESPACE_AT];
        let Some(namespace) = ChunkNamespace::from_byte(namespace_byte) else {
            return Err(Error::corruption(format!(
                "directory entry of section {section_type:?}, chunk kind {kind:?}, has \
                 namespace {namespace_byte}, which no Grafeo writes"
            )));
        };
        Ok(DecodedEntry::Known(Self {
            section_type,
            section_version: bytes[1],
            meta: ChunkMeta {
                kind,
                namespace,
                codec: bytes[3],
                graph_id: u32_at(bytes, 4),
                column_id: u32_at(bytes, 8),
                row_count: u32_at(bytes, 12),
                row_start: u64_at(bytes, 16),
            },
            offset: u64_at(bytes, 24),
            length: u64_at(bytes, 32),
            crc: u32_at(bytes, 40),
            flags,
        }))
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

/// An entry this reader does not know and may skip. Its pages stay in use until a checkpoint,
/// which does not write it again.
///
/// Only where the chunk lies is kept: it is never fetched, decrypted or
/// handed to a section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SkippedEntry {
    /// The section type byte, as stored.
    pub section_type: u8,
    /// The chunk kind byte, as stored.
    pub kind: u8,
    /// The flags byte, as stored.
    pub flags: u8,
    /// Byte offset of the chunk in the file.
    pub offset: u64,
    /// Length of the stored chunk in bytes.
    pub length: u64,
}

impl SkippedEntry {
    /// Pages occupied by the chunk (a zero-length chunk has an empty run).
    #[must_use]
    pub fn run(&self) -> PageRun {
        PageRun {
            first: self.offset / PAGE_SIZE,
            count: PageRun::for_bytes(self.length),
        }
    }
}

/// A decoded directory entry: one this version knows, or one it may skip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodedEntry {
    /// An entry of a section type and chunk kind this version knows.
    Known(DirectoryEntry),
    /// An entry of an unknown section type or chunk kind whose optional bit
    /// is set.
    Skipped(SkippedEntry),
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

/// Encodes one block holding the encoded `entries` and naming `next`.
fn encode_block(entries: &[[u8; ENTRY_SIZE]], next: BlockRef) -> Result<Vec<u8>> {
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
        slot.copy_from_slice(entry);
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
    place: impl FnMut(usize) -> Result<u64>,
) -> Result<(BlockRef, Vec<(u64, Vec<u8>)>)> {
    let encoded: Vec<[u8; ENTRY_SIZE]> = entries
        .iter()
        .map(|entry| {
            let mut bytes = [0u8; ENTRY_SIZE];
            entry.encode(&mut bytes);
            bytes
        })
        .collect();
    encode_blocks_raw(&encoded, place)
}

/// `encode_blocks` over entries already encoded; `encode_blocks` encodes and calls it.
///
/// The bytes of each entry are written as given, so entries this version
/// cannot describe as a [`DirectoryEntry`] (those of a newer version, in
/// tests) go into the directory too.
///
/// # Errors
///
/// Returns an error when `place` fails.
pub fn encode_blocks_raw(
    entries: &[[u8; ENTRY_SIZE]],
    mut place: impl FnMut(usize) -> Result<u64>,
) -> Result<(BlockRef, Vec<(u64, Vec<u8>)>)> {
    let groups: Vec<&[[u8; ENTRY_SIZE]]> = if entries.is_empty() {
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

/// `error`, raised while the directory block at `offset` was read or
/// decoded, naming the block: damage at the block's offset (unless it has
/// one), any other error with the block's offset in its message.
pub(super) fn in_block(error: Error, offset: u64) -> Error {
    match error {
        Error::Corruption(_) => error.wrapped("directory block").at(offset),
        other => other.wrapped(format_args!("directory block at offset {offset}")),
    }
}

/// Checks a block pointer before it is read, so a corrupt pointer never
/// allocates a huge buffer.
fn check_pointer(pointer: BlockRef) -> Result<()> {
    let offset = pointer.offset;
    if !offset.is_multiple_of(PAGE_SIZE) {
        return Err(Error::corruption_at(
            format!("directory block at offset {offset} is not page aligned"),
            offset,
        ));
    }
    if offset < DATA_START_PAGE * PAGE_SIZE {
        return Err(Error::corruption_at(
            format!("directory block at offset {offset} lies before the data area"),
            offset,
        ));
    }
    let length = pointer.length as usize;
    if !(BLOCK_HEADER_SIZE..=MAX_BLOCK_SIZE).contains(&length)
        || !(length - BLOCK_HEADER_SIZE).is_multiple_of(ENTRY_SIZE)
    {
        return Err(Error::corruption_at(
            format!("directory block at offset {offset} has invalid length {length}"),
            offset,
        ));
    }
    Ok(())
}

/// Follows the chain from `root`, `read(offset, len)` returning the bytes of
/// the block at `offset` whose pointer names `len` bytes (for an encrypted
/// image, the decrypted bytes).
///
/// Every block is stored `overhead` bytes longer than its pointer's length
/// (the nonce and tag of an encrypted image, 0 otherwise). Returns the known
/// entries in order, the skipped entries (see [`DirectoryEntry::decode`]) in
/// order, and the pages the stored blocks occupy. A root with length 0 is
/// accepted as an empty directory, although [`encode_blocks`] never produces
/// one.
///
/// # Errors
///
/// Returns [`Error::Corruption`] at the block offset when a pointer is
/// misaligned, out of range or revisited, or when a block fails its CRC,
/// magic or length checks. An entry that is refused gives the error of
/// [`DirectoryEntry::decode`], a corruption at the block offset or an
/// error naming the block. Errors from `read` are passed through.
pub fn decode_chain(
    root: BlockRef,
    overhead: u32,
    mut read: impl FnMut(u64, u32) -> Result<Vec<u8>>,
) -> Result<(Vec<DirectoryEntry>, Vec<SkippedEntry>, Vec<PageRun>)> {
    let mut entries = Vec::new();
    let mut skipped = Vec::new();
    let mut runs = Vec::new();
    if root.length == 0 {
        return Ok((entries, skipped, runs));
    }
    let mut visited = HashSet::new();
    let mut pointer = root;
    loop {
        let offset = pointer.offset;
        if !visited.insert(offset) {
            return Err(Error::corruption_at(
                format!("directory chain revisits the block at offset {offset}"),
                offset,
            ));
        }
        check_pointer(pointer)?;
        let bytes = read(offset, pointer.length)?;
        if bytes.len() != pointer.length as usize {
            return Err(Error::corruption_at(
                format!(
                    "directory block at offset {offset} read {} bytes, expected {}",
                    bytes.len(),
                    pointer.length
                ),
                offset,
            ));
        }
        if crc32fast::hash(&bytes) != pointer.crc {
            return Err(Error::corruption_at(
                format!("directory block at offset {offset} fails its checksum"),
                offset,
            ));
        }
        if bytes[0..4] != BLOCK_MAGIC {
            return Err(Error::corruption_at(
                format!("directory block at offset {offset} has a bad magic"),
                offset,
            ));
        }
        let count = u32_at(&bytes, 4) as usize;
        if count > ENTRIES_PER_BLOCK || BLOCK_HEADER_SIZE + count * ENTRY_SIZE != bytes.len() {
            return Err(Error::corruption_at(
                format!(
                    "directory block at offset {offset} declares {count} entries but holds {} \
                     bytes",
                    bytes.len()
                ),
                offset,
            ));
        }
        let (slots, _) = bytes[BLOCK_HEADER_SIZE..].as_chunks::<ENTRY_SIZE>();
        for slot in slots {
            let decoded = DirectoryEntry::decode(slot).map_err(|error| in_block(error, offset))?;
            match decoded {
                DecodedEntry::Known(entry) => entries.push(entry),
                DecodedEntry::Skipped(entry) => skipped.push(entry),
            }
        }
        // Two `u32` values: the sum cannot overflow `u64`.
        runs.push(PageRun {
            first: offset / PAGE_SIZE,
            count: PageRun::for_bytes(u64::from(pointer.length) + u64::from(overhead)),
        });
        pointer = BlockRef {
            offset: u64_at(&bytes, 8),
            length: u32_at(&bytes, 16),
            crc: u32_at(&bytes, 20),
        };
        if pointer.length == 0 {
            return Ok((entries, skipped, runs));
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
            flags: 0,
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
        let (back, skipped, runs) = decode_chain(root, 0, |offset, len| {
            let b = &stored[&offset];
            assert_eq!(b.len(), len as usize);
            Ok(b.clone())
        })
        .unwrap();
        assert_eq!(back, entries);
        assert!(skipped.is_empty(), "every entry is known");
        assert_eq!(runs.len(), 3);
    }

    /// The encoded form of `entry`.
    fn encoded(entry: &DirectoryEntry) -> [u8; ENTRY_SIZE] {
        let mut bytes = [0u8; ENTRY_SIZE];
        entry.encode(&mut bytes);
        bytes
    }

    /// Entries a newer version wrote come back apart from the known ones, in
    /// directory order, across blocks.
    #[test]
    fn a_chain_returns_the_skipped_entries_apart_from_the_known_ones() {
        let mut foreign_section = encoded(&entry(1));
        foreign_section[0] = 250;
        foreign_section[44] = ENTRY_SECTION_OPTIONAL;
        let mut foreign_kind = encoded(&entry(2));
        foreign_kind[2] = 88;
        foreign_kind[44] = ENTRY_CHUNK_OPTIONAL;
        let mut raw: Vec<[u8; ENTRY_SIZE]> = (3..3 + ENTRIES_PER_BLOCK as u64)
            .map(|i| encoded(&entry(i)))
            .collect();
        raw.insert(0, encoded(&entry(0)));
        raw.insert(1, foreign_section);
        raw.push(foreign_kind);
        let mut next = 245 * PAGE_SIZE;
        let (root, blocks) = encode_blocks_raw(&raw, |len| {
            let at = next;
            next += PageRun::for_bytes(len as u64) * PAGE_SIZE;
            Ok(at)
        })
        .unwrap();
        assert_eq!(
            blocks.len(),
            2,
            "the last foreign entry is in the second block"
        );
        let stored: std::collections::HashMap<u64, Vec<u8>> = blocks.into_iter().collect();
        let (known, skipped, runs) =
            decode_chain(root, 0, |offset, _| Ok(stored[&offset].clone())).unwrap();
        assert_eq!(known.len(), ENTRIES_PER_BLOCK + 1);
        assert_eq!(known[0], entry(0));
        assert_eq!(
            known[1],
            entry(3),
            "the skipped entry is not among the known"
        );
        assert_eq!(
            skipped,
            [
                SkippedEntry {
                    section_type: 250,
                    kind: 0,
                    flags: ENTRY_SECTION_OPTIONAL,
                    offset: entry(1).offset,
                    length: entry(1).length,
                },
                SkippedEntry {
                    section_type: SectionType::LpgStore.to_u8(),
                    kind: 88,
                    flags: ENTRY_CHUNK_OPTIONAL,
                    offset: entry(2).offset,
                    length: entry(2).length,
                },
            ]
        );
        assert_eq!(runs.len(), 2);
        assert_eq!(
            skipped[0].run(),
            entry(1).run(),
            "a skipped entry names its pages as a known one does"
        );
    }

    /// An encrypted block is stored with a nonce and a tag around its
    /// plaintext: the runs cover the stored bytes. A block whose plaintext
    /// fills two pages exactly (170 entries) takes a third page encrypted.
    #[test]
    fn block_runs_cover_the_stored_bytes_with_their_overhead() {
        let entries: Vec<_> = (0..170).map(entry).collect();
        let (root, blocks) = encode_blocks(&entries, |_| Ok(12_288)).unwrap();
        assert_eq!(blocks[0].1.len(), 8192, "the plaintext fills two pages");
        for (overhead, pages) in [(0, 2), (28, 3)] {
            let (back, _, runs) = decode_chain(root, overhead, |offset, length| {
                assert_eq!(
                    (offset, length),
                    (12_288, root.length),
                    "the read is asked for the plaintext length"
                );
                Ok(blocks[0].1.clone())
            })
            .unwrap();
            assert_eq!(back, entries);
            assert_eq!(
                runs,
                [PageRun {
                    first: 3,
                    count: pages
                }],
                "overhead {overhead}: the stored block's pages"
            );
        }
    }

    #[test]
    fn a_damaged_block_fails_with_its_offset() {
        let entries = vec![entry(0)];
        let (root, mut blocks) = encode_blocks(&entries, |_| Ok(12_288)).unwrap();
        blocks[0].1[40] ^= 1;
        let error = decode_chain(root, 0, |_, _| Ok(blocks[0].1.clone()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("12288"), "{error}");
    }

    #[test]
    fn unknown_entries_are_skipped_when_optional_and_refused_when_required() {
        let mut bytes = [0u8; ENTRY_SIZE];
        entry(0).encode(&mut bytes); // an LpgStore entry
        assert_eq!(bytes[44], 0, "every entry of this release is required");
        let mut foreign_section = bytes;
        foreign_section[0] = 250;
        let error = DirectoryEntry::decode(&foreign_section)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("section type 250") && error.contains("required"),
            "{error}"
        );
        foreign_section[44] = ENTRY_SECTION_OPTIONAL;
        assert!(matches!(
            DirectoryEntry::decode(&foreign_section).unwrap(),
            DecodedEntry::Skipped(SkippedEntry {
                section_type: 250,
                ..
            })
        ));
        let mut foreign_kind = bytes;
        foreign_kind[2] = 250;
        let error = DirectoryEntry::decode(&foreign_kind)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("chunk kind 250")
                && error.contains("LpgStore")
                && error.contains("required"),
            "{error}"
        );
        foreign_kind[44] = ENTRY_SECTION_OPTIONAL;
        assert!(
            DirectoryEntry::decode(&foreign_kind).is_err(),
            "the section bit does not cover a kind of a known section"
        );
        foreign_kind[44] = ENTRY_CHUNK_OPTIONAL;
        assert!(matches!(
            DirectoryEntry::decode(&foreign_kind).unwrap(),
            DecodedEntry::Skipped(SkippedEntry { kind: 250, .. })
        ));
    }

    #[test]
    fn entry_flags_round_trip_and_unknown_incompatible_bits_are_refused() {
        let flagged = DirectoryEntry {
            flags: ENTRY_SECTION_OPTIONAL | ENTRY_CHUNK_OPTIONAL,
            ..entry(3)
        };
        let mut bytes = [0u8; ENTRY_SIZE];
        flagged.encode(&mut bytes);
        assert_eq!(
            DirectoryEntry::decode(&bytes).unwrap(),
            DecodedEntry::Known(flagged),
            "a known entry keeps its flags"
        );
        bytes[44] = 0x04;
        assert!(
            DirectoryEntry::decode(&bytes)
                .unwrap_err()
                .to_string()
                .contains("flag"),
            "bit 2 is not known"
        );
        bytes[44] = 0x10;
        assert!(
            matches!(
                DirectoryEntry::decode(&bytes).unwrap(),
                DecodedEntry::Known(_)
            ),
            "bits 4 to 7 are ignored"
        );
    }

    /// An unknown bit among 0 to 3 is refused also on an entry that would
    /// be skipped, and every one of bits 4 to 7 is ignored.
    #[test]
    fn every_incompatible_bit_is_refused_and_every_other_bit_ignored() {
        let mut foreign_section = encoded(&entry(0));
        foreign_section[0] = 250;
        for bit in 2..4 {
            foreign_section[44] = ENTRY_SECTION_OPTIONAL | (1 << bit);
            let error = DirectoryEntry::decode(&foreign_section)
                .unwrap_err()
                .to_string();
            assert!(error.contains("flag"), "bit {bit}: {error}");
        }
        for bit in 4..8 {
            let mut bytes = encoded(&entry(0));
            bytes[44] = 1 << bit;
            assert!(
                matches!(
                    DirectoryEntry::decode(&bytes).unwrap(),
                    DecodedEntry::Known(_)
                ),
                "bit {bit} is ignored"
            );
            foreign_section[44] = ENTRY_SECTION_OPTIONAL | (1 << bit);
            assert!(
                matches!(
                    DirectoryEntry::decode(&foreign_section).unwrap(),
                    DecodedEntry::Skipped(_)
                ),
                "bit {bit} is ignored on a skipped entry"
            );
        }
        assert_eq!(ENTRY_INCOMPATIBLE_FLAGS, 0x0F);
    }

    /// For every flags byte: only bit 0 lets an unknown section type be
    /// skipped, only bit 1 an unknown kind of a known section; bits 2 and 3
    /// are refused, and bits 4 to 7 never make an entry skippable.
    #[test]
    fn only_its_own_bit_lets_an_unknown_entry_be_skipped() {
        let known = encoded(&entry(0));
        for flags in 0..=u8::MAX {
            let readable = flags & 0x0C == 0;
            for (at, bit, what) in [
                (0, ENTRY_SECTION_OPTIONAL, "section type"),
                (2, ENTRY_CHUNK_OPTIONAL, "chunk kind"),
            ] {
                let mut foreign = known;
                foreign[at] = 250;
                foreign[44] = flags;
                let decoded = DirectoryEntry::decode(&foreign);
                if readable && flags & bit != 0 {
                    assert!(
                        matches!(decoded, Ok(DecodedEntry::Skipped(_))),
                        "unknown {what}, flags {flags:#04x}: skipped, not {decoded:?}"
                    );
                } else {
                    assert!(
                        decoded.is_err(),
                        "unknown {what}, flags {flags:#04x}: refused, not {decoded:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_writer_sets_no_optional_bit_for_a_section_of_this_release() {
        for section_type in [
            SectionType::Catalog,
            SectionType::LpgStore,
            SectionType::PropertyIndex,
        ] {
            for kind in [ChunkKind::Raw, ChunkKind::Meta, ChunkKind::Stream] {
                assert_eq!(
                    DirectoryEntry::flags_for(section_type, kind),
                    0,
                    "{section_type:?}, {kind:?}"
                );
            }
        }
    }

    #[test]
    fn an_entry_round_trips_and_names_its_pages() {
        let original = entry(19);
        let mut bytes = [0u8; ENTRY_SIZE];
        original.encode(&mut bytes);
        assert_eq!(
            DirectoryEntry::decode(&bytes).unwrap(),
            DecodedEntry::Known(original)
        );
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
            let error = decode_chain(root, 0, |_, _| panic!("must not read"))
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
            let error = decode_chain(root, 0, |_, _| panic!("must not read"))
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
            &[encoded(&entry(0))],
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
        let error = decode_chain(root, 0, |offset, _| {
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
                namespace: ChunkNamespace::EdgeStructure,
                codec: 88,
                graph_id: 0x0102_0304,
                column_id: 0x0506_0708,
                row_count: 0x090A_0B0C,
                row_start: 0x1112_1314_1516_1718,
            },
            offset: 0x0000_0003_1988_3000,
            length: 0x0000_0001_0000_2328,
            crc: 0x1988_0319,
            flags: 0,
        }
    }

    #[test]
    fn a_directory_entry_has_its_documented_byte_layout() {
        let entry = DirectoryEntry {
            flags: 0x33,
            ..fixed_entry()
        };
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
        assert_eq!(bytes[44], 0x33, "flags");
        assert_eq!(bytes[45], 32, "namespace byte (edge structure)");
        assert_eq!(bytes[46..48], [0, 0], "reserved, written as zero");
        // Bits 2, 3, 6 and 7: a reserved byte read into the flags would set
        // an incompatible bit and the entry would be refused.
        bytes[46..48].copy_from_slice(&[0xCC, 0xCC]);
        assert_eq!(
            DirectoryEntry::decode(&bytes).unwrap(),
            DecodedEntry::Known(entry),
            "readers ignore the reserved bytes"
        );
        assert_eq!(
            encoded(&fixed_entry())[44],
            0,
            "an entry without flags writes 0"
        );
    }

    /// A namespace byte no Grafeo writes (a reserved one included) is damage:
    /// the entry is refused, also when its optional bits are set, since only
    /// an unknown section type or chunk kind may be skipped.
    #[test]
    fn an_entry_with_an_unknown_namespace_is_refused() {
        for (namespace, flags) in [
            (20, 0),
            (36, 0),
            (255, ENTRY_SECTION_OPTIONAL | ENTRY_CHUNK_OPTIONAL),
        ] {
            let mut bytes = encoded(&DirectoryEntry {
                flags,
                ..fixed_entry()
            });
            bytes[45] = namespace;
            let error = DirectoryEntry::decode(&bytes).unwrap_err().to_string();
            assert!(
                error.contains(&format!("namespace {namespace}")) && error.contains("damaged"),
                "namespace {namespace}: {error}"
            );
        }
    }

    #[test]
    fn a_directory_block_header_has_its_documented_byte_layout() {
        let next = BlockRef {
            offset: 0x0000_0003_1988_3000,
            length: 0x0000_0050,
            crc: 0xC0FF_EE19,
        };
        let mut block = encode_block(&[encoded(&fixed_entry())], next).unwrap();
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
        let (entries, _, _) = decode_chain(root, 0, |offset, _| {
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
        let (entries, skipped, runs) = decode_chain(root, 0, |offset, _| {
            assert_eq!(offset, 12_288);
            Ok(blocks[0].1.clone())
        })
        .unwrap();
        assert!(
            entries.is_empty() && skipped.is_empty(),
            "a block without entries"
        );
        assert_eq!(runs, [PageRun { first: 3, count: 1 }], "the block's page");
    }

    #[test]
    fn a_zero_length_root_still_decodes_as_a_directory_without_chunks() {
        let (entries, skipped, runs) =
            decode_chain(BlockRef::default(), 0, |_, _| panic!("must not read")).unwrap();
        assert!(entries.is_empty() && skipped.is_empty() && runs.is_empty());
    }
}
