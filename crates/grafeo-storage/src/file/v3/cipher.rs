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

/// Associated data binding a chunk to its section and position:
/// `grafeo-chunk:{section type byte}:{chunk kind byte}:{graph}:{column}:{first row}`.
///
/// Every part is an on-disk number, never a Rust name, so renaming a variant
/// cannot make an encrypted file undecryptable.
#[cfg(feature = "encryption")]
pub(super) fn chunk_aad(entry: &super::directory::DirectoryEntry) -> Vec<u8> {
    let meta = &entry.meta;
    format!(
        "grafeo-chunk:{}:{}:{}:{}:{}",
        entry.section_type.to_u8(),
        meta.kind.to_byte(),
        meta.graph_id,
        meta.column_id,
        meta.row_start
    )
    .into_bytes()
}

/// Associated data binding a directory block to its offset:
/// `grafeo-directory:{offset}`.
#[cfg(feature = "encryption")]
pub(super) fn directory_aad(offset: u64) -> Vec<u8> {
    format!("grafeo-directory:{offset}").into_bytes()
}

#[cfg(all(test, feature = "encryption"))]
mod tests {
    use grafeo_common::storage::{ChunkMeta, SectionType};

    use super::super::directory::DirectoryEntry;
    use super::{chunk_aad, directory_aad};

    fn text(aad: Vec<u8>) -> String {
        String::from_utf8(aad).unwrap()
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
