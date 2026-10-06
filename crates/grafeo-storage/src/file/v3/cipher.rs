//! The encryptor type the v3 writer and reader take.
//!
//! The writer and reader take `Option<&ChunkCipher>` in every feature set so
//! callers keep one signature. Without the `encryption` feature the type has
//! no values, so only `None` can be passed.

/// Encrypts and decrypts chunks and directory blocks.
#[cfg(feature = "encryption")]
pub type ChunkCipher = grafeo_common::encryption::PageEncryptor;

/// Placeholder without the `encryption` feature: it has no values, so a
/// cipher can never be supplied.
#[cfg(not(feature = "encryption"))]
#[derive(Debug)]
pub enum ChunkCipher {}

/// Bytes an encrypted block adds to its plaintext (nonce and tag).
#[cfg(feature = "encryption")]
pub const ENCRYPTION_OVERHEAD: usize = grafeo_common::encryption::ENCRYPTION_OVERHEAD;

/// Bytes an encrypted block adds to its plaintext (none without the feature).
#[cfg(not(feature = "encryption"))]
pub const ENCRYPTION_OVERHEAD: usize = 0;

/// Associated data of the chunk an entry describes, for the reader: see
/// [`chunk_aad_for`].
#[cfg(feature = "encryption")]
pub(super) fn chunk_aad(entry: &super::directory::DirectoryEntry) -> Vec<u8> {
    chunk_aad_for(entry.section_type, &entry.meta)
}

/// Associated data binding a chunk to its section and position:
/// `grafeo-chunk:{section type byte}:{chunk kind byte}:{graph}:{column}:{first row}`.
///
/// The one place that puts a chunk's fields in that order: the writer
/// encrypts with it and the reader ([`chunk_aad`]) decrypts with it. Every
/// part is an on-disk number, never a Rust name, so renaming a variant cannot
/// make an encrypted file undecryptable.
pub(super) fn chunk_aad_for(
    section_type: grafeo_common::storage::SectionType,
    meta: &grafeo_common::storage::ChunkMeta,
) -> Vec<u8> {
    chunk_aad_parts(
        section_type.to_u8(),
        meta.kind.to_byte(),
        meta.graph_id,
        meta.column_id,
        meta.row_start,
    )
}

/// Associated data of a chunk from the numbers its directory entry stores,
/// each in decimal (see [`chunk_aad_for`] for the order).
///
/// It takes the bytes as stored, so it also serves a chunk whose section type
/// or kind this version does not know (a newer version's, written in tests).
pub(super) fn chunk_aad_parts(
    section_type: u8,
    kind: u8,
    graph_id: u32,
    column_id: u32,
    row_start: u64,
) -> Vec<u8> {
    format!("grafeo-chunk:{section_type}:{kind}:{graph_id}:{column_id}:{row_start}").into_bytes()
}

/// Associated data binding a directory block to its offset:
/// `grafeo-directory:{offset}`.
#[cfg(feature = "encryption")]
pub(super) fn directory_aad(offset: u64) -> Vec<u8> {
    format!("grafeo-directory:{offset}").into_bytes()
}

#[cfg(all(test, feature = "encryption"))]
mod tests {
    use grafeo_common::storage::{ChunkKind, ChunkMeta, SectionType};

    use super::super::directory::DirectoryEntry;
    use super::{chunk_aad, chunk_aad_parts, directory_aad};

    fn text(aad: Vec<u8>) -> String {
        String::from_utf8(aad).unwrap()
    }

    /// An entry with a value of several bytes in every field the associated
    /// data holds, so a field written in another byte order or another
    /// order shows.
    fn wide_entry() -> DirectoryEntry {
        DirectoryEntry {
            section_type: SectionType::PropertyIndex,
            section_version: 3,
            meta: ChunkMeta {
                kind: ChunkKind::Stream,
                codec: 19,
                graph_id: 0x0102_0304,
                column_id: 0x0506_0708,
                row_start: 0x1112_1314_1516_1718,
                row_count: 88,
            },
            offset: 12_288,
            length: 32,
            crc: 0,
            flags: 0,
        }
    }

    /// The exact bytes of a known entry's associated data, as every encrypted
    /// image written so far carries them: a change here makes those files
    /// undecryptable.
    #[test]
    fn the_associated_data_of_a_known_entry_keeps_its_bytes() {
        let aad = chunk_aad(&wide_entry());
        assert_eq!(
            aad,
            b"grafeo-chunk:20:4:16909060:84281096:1230066625199609624",
            "as text: {}",
            String::from_utf8_lossy(&aad)
        );
    }

    #[test]
    fn the_associated_data_of_raw_numbers_is_that_of_the_entry_holding_them() {
        assert_eq!(
            chunk_aad_parts(20, 4, 0x0102_0304, 0x0506_0708, 0x1112_1314_1516_1718),
            chunk_aad(&wide_entry()),
            "a known entry's associated data is built from its on-disk numbers"
        );
        assert_eq!(
            text(chunk_aad_parts(250, 88, 3, 19, 88)),
            "grafeo-chunk:250:88:3:19:88",
            "numbers this version does not know are written the same way"
        );
    }

    #[test]
    fn the_associated_data_is_built_from_on_disk_numbers() {
        let lpg = DirectoryEntry {
            section_type: SectionType::LpgStore,
            section_version: 1,
            meta: ChunkMeta::raw(),
            offset: 12_288,
            length: 32,
            crc: 0,
            flags: 0,
        };
        assert_eq!(text(chunk_aad(&lpg)), "grafeo-chunk:2:0:0:0:0");
        let placed = DirectoryEntry {
            section_type: SectionType::PropertyIndex,
            meta: ChunkMeta {
                graph_id: 3,
                column_id: 19,
                row_start: 88,
                row_count: 3,
                ..ChunkMeta::raw()
            },
            ..lpg
        };
        assert_eq!(
            text(chunk_aad(&placed)),
            "grafeo-chunk:20:0:3:19:88",
            "section type byte, kind byte, graph, column, first row"
        );
        assert_eq!(text(directory_aad(12_288)), "grafeo-directory:12288");
    }
}
