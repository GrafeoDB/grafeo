//! A damaged database file is an `Error::Corruption` naming the file, at
//! every layer an open reads: the file header, the directory, a chunk and a
//! section that does not decode. A read-only open and a read-write open
//! report it alike, and neither writes to the file.

#![cfg(all(feature = "grafeo-file", feature = "lpg", feature = "gql"))]

use std::path::Path;

use grafeo_common::storage::{ChunkMeta, Section, SectionSink, SectionSource, SectionType};
use grafeo_common::utils::error::{Error, ErrorCode, Result};
use grafeo_engine::{Config, GrafeoDB};
use grafeo_storage::file::{CheckpointHeader, GrafeoFileManager};

/// Writes a database holding Shosanna and Hans, and closes it.
fn write_database(path: &Path) {
    let db = GrafeoDB::with_config(Config::persistent(path)).unwrap();
    db.execute("INSERT (:Person {name: 'Shosanna', city: 'Paris'})")
        .unwrap();
    db.execute("INSERT (:Person {name: 'Hans', city: 'Berlin'})")
        .unwrap();
    db.close().unwrap();
}

/// Flips the byte at `offset` of the file.
fn flip(path: &Path, offset: u64) {
    let mut bytes = std::fs::read(path).unwrap();
    let at = usize::try_from(offset).unwrap();
    bytes[at] ^= 0x5A;
    std::fs::write(path, &bytes).unwrap();
}

/// The offset of the first page that holds `needle`.
fn page_holding(path: &Path, needle: &[u8]) -> u64 {
    let bytes = std::fs::read(path).unwrap();
    let at = bytes
        .windows(needle.len())
        .position(|window| window == needle)
        .expect("the bytes are in the file");
    let at = u64::try_from(at).unwrap();
    at - at % 4096
}

/// Opens `path` read-only and read-write, and asserts that each open fails
/// with a corruption of that file, at `offset` when one is given, whose
/// description holds `what`, and that neither open changes the file.
fn assert_both_opens_report(path: &Path, offset: Option<u64>, what: &str) {
    let before = std::fs::read(path).unwrap();
    for (open, result) in [
        ("read-only", GrafeoDB::open_read_only(path).map(|_| ())),
        (
            "read-write",
            GrafeoDB::with_config(Config::persistent(path)).map(|_| ()),
        ),
    ] {
        let error = result.expect_err(open);
        let Error::Corruption(corruption) = &error else {
            panic!("{open}: a damaged file is a corruption: {error:?}");
        };
        assert_eq!(corruption.file.as_deref(), Some(path), "{open}: {error}");
        if offset.is_some() {
            assert_eq!(corruption.offset, offset, "{open}: {error}");
        }
        assert!(corruption.what.contains(what), "{open}: {error}");
        assert_eq!(error.error_code(), ErrorCode::StorageCorrupted, "{open}");
        assert!(!error.error_code().is_retryable(), "{open}");
        let message = error.to_string();
        assert!(
            message.starts_with("GRAFEO-S002") && message.contains(&path.display().to_string()),
            "{open}: the message names the code and the file: {message}"
        );
    }
    assert!(
        std::fs::read(path).unwrap() == before,
        "a refused open writes nothing"
    );
}

#[test]
fn a_damaged_file_header_is_a_corruption_at_byte_0() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("paris.grafeo");
    write_database(&path);
    // Inside the database id, which the header checksum covers.
    flip(&path, 20);
    assert_both_opens_report(&path, Some(0), "file header checksum mismatch");
}

#[test]
fn a_damaged_directory_block_is_a_corruption_at_the_block() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("berlin.grafeo");
    write_database(&path);
    let root = GrafeoFileManager::open(&path, None)
        .unwrap()
        .active_header()
        .root;
    flip(&path, root.offset + 30);
    assert_both_opens_report(&path, Some(root.offset), "fails its checksum");
}

#[test]
fn a_damaged_chunk_is_a_corruption_at_the_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prague.grafeo");
    write_database(&path);
    let chunk = page_holding(&path, b"Shosanna");
    let at = std::fs::read(&path)
        .unwrap()
        .windows(8)
        .position(|window| window == b"Shosanna")
        .unwrap();
    flip(&path, u64::try_from(at).unwrap());
    assert_both_opens_report(&path, Some(chunk), "fails its checksum");
}

/// A section whose chunks pass their checksums but do not decode: written
/// again chunk for chunk by a checkpoint, with one byte of the LPG metadata
/// chunk changed.
#[test]
fn a_section_that_does_not_decode_is_a_corruption_naming_the_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("amsterdam.grafeo");
    write_database(&path);
    rewrite(&path, |section_type, chunks| {
        if section_type == SectionType::LpgStore {
            // The metadata chunk comes last; its first byte is the layout.
            let (_, meta) = chunks.last_mut().expect("a metadata chunk");
            meta[0] = 19;
        }
    });
    assert_both_opens_report(&path, None, "LPG metadata chunk, byte 0");
}

/// A section written back chunk for chunk, as it was read.
struct Replayed {
    section_type: SectionType,
    version: u8,
    chunks: Vec<(ChunkMeta, Vec<u8>)>,
}

impl Section for Replayed {
    fn section_type(&self) -> SectionType {
        self.section_type
    }

    fn version(&self) -> u8 {
        self.version
    }

    fn serialize(&self) -> Result<Vec<u8>> {
        Err(Error::Internal("a replayed section writes chunks".into()))
    }

    fn deserialize(&mut self, _data: &[u8]) -> Result<()> {
        Err(Error::Internal("a replayed section is not read".into()))
    }

    fn write_to(&self, sink: &mut dyn SectionSink) -> Result<()> {
        for (meta, bytes) in &self.chunks {
            sink.write_chunk(*meta, bytes)?;
        }
        Ok(())
    }

    fn read_from(&mut self, _source: &dyn SectionSource) -> Result<()> {
        Err(Error::Internal("a replayed section is not read".into()))
    }

    fn is_dirty(&self) -> bool {
        true
    }

    fn mark_clean(&self) {}

    fn memory_usage(&self) -> usize {
        0
    }
}

/// Checkpoints the database at `path` again with every section of its
/// active image, chunk for chunk, after `alter` changed what it wants.
fn rewrite(path: &Path, alter: impl Fn(SectionType, &mut Vec<(ChunkMeta, Vec<u8>)>)) {
    let manager = GrafeoFileManager::open(path, None).unwrap();
    let mut sections: Vec<Replayed> = manager
        .read_image(|image| {
            let mut sections = Vec::new();
            for byte in 0..=u8::MAX {
                let Some(section_type) = SectionType::from_u8(byte) else {
                    continue;
                };
                let Some(source) = image.section_source(section_type) else {
                    continue;
                };
                let chunks = (0..source.chunks().len())
                    .map(|index| Ok((source.chunks()[index], source.fetch(index)?.to_vec())))
                    .collect::<Result<Vec<_>>>()?;
                sections.push(Replayed {
                    section_type,
                    version: source.section_version(),
                    chunks,
                });
            }
            Ok(sections)
        })
        .unwrap();
    for section in &mut sections {
        alter(section.section_type, &mut section.chunks);
    }
    let active = manager.active_header();
    let sections: Vec<&dyn Section> = sections.iter().map(|s| s as &dyn Section).collect();
    manager
        .write_checkpoint(
            &sections,
            &CheckpointHeader {
                checkpoint_lsn: active.checkpoint_lsn,
                epoch: active.epoch,
                last_transaction_id: active.last_transaction_id,
                node_count: active.node_count,
                edge_count: active.edge_count,
            },
        )
        .unwrap();
    manager.close().unwrap();
}
