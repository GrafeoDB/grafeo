//! A `.grafeo` file whose image lies past 4 GiB reads back (#392).
//!
//! 0.5.x stored block offsets in a `u32`: past 4 GiB they wrapped, `close()`
//! still reported success, and the next open failed with a CRC mismatch.
//! Container v3 stores every chunk and directory offset as a `u64`. The full
//! scenario, a database over 4 GiB, is `grafeo-engine/tests/large_database.rs`,
//! which is ignored in normal runs for its size; this test runs in CI on a
//! sparse file instead.
//!
//! The file is created as usual; an image is then written with an allocator
//! whose used pages reach up to 4 GiB, so its first chunk straddles the
//! 4 GiB boundary and every other chunk and its directory lie past it. A
//! header pointing at that image makes it the active one. Reopening the file
//! reads every chunk back against its CRC; a checkpoint over that image goes
//! into the free pages below it, reads back after a second reopen, and cuts
//! the file back below 4 GiB.
//!
//! The pages below the image are never written, so the file is sparse: on
//! Linux and macOS a file is sparse by default; on Windows the test marks it
//! sparse with `fsutil` before writing past 4 GiB, so NTFS does not write the
//! gap as zeros.
//!
//! ```bash
//! cargo test -p grafeo-storage --features grafeo-file,encryption --test offsets_past_4_gib
//! ```

#![cfg(feature = "grafeo-file")]

use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

use grafeo_common::storage::{
    ChunkMeta, Section, SectionSink, SectionSource, SectionType, read_raw, write_raw,
};
use grafeo_common::utils::error::Result;
use grafeo_storage::file::v3::alloc::{PageAllocator, PageRun};
use grafeo_storage::file::v3::header::{DATA_START_PAGE, DbHeaderV3, PAGE_SIZE, slot_offset};
use grafeo_storage::file::v3::{CheckpointWriter, ChunkCipher, ImageReader};
use grafeo_storage::file::{CheckpointHeader, GrafeoFileManager};

const FOUR_GIB: u64 = 1 << 32;

/// Deterministic bytes that differ from page to page, so a chunk read from
/// the wrong offset fails its CRC or the comparison.
fn noise(seed: u64, len: usize) -> Vec<u8> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state.to_le_bytes()[0]
        })
        .collect()
}

/// The chunks of the image past 4 GiB: (section, chunk, bytes).
fn chunks() -> Vec<(SectionType, ChunkMeta, Vec<u8>)> {
    vec![
        // Three pages from one page below 4 GiB: straddles the boundary.
        (
            SectionType::Catalog,
            ChunkMeta::raw(),
            noise(1, 3 * 4096 - 100),
        ),
        // A column chunk whose first row is past `u32::MAX` too.
        (
            SectionType::LpgStore,
            ChunkMeta::column(0, 1, 5_000_000_000, 3, 0),
            b"Alix,Gus,Vincent".to_vec(),
        ),
        (
            SectionType::LpgStore,
            ChunkMeta::column(0, 2, 0, 1, 0),
            noise(2, 10_000),
        ),
        (
            SectionType::LpgStore,
            ChunkMeta::column(7, 3, 0, 0, 0),
            Vec::new(),
        ),
    ]
}

/// Marks `path` sparse, so the pages below the image take no disk space.
fn make_sparse(path: &Path) {
    #[cfg(windows)]
    {
        let status = std::process::Command::new("fsutil")
            .args(["sparse", "setflag"])
            .arg(path)
            .stdout(std::process::Stdio::null())
            .status()
            .expect("run fsutil");
        assert!(status.success(), "fsutil sparse setflag {}", path.display());
    }
    #[cfg(not(windows))]
    let _ = path;
}

/// Writes `chunks` as an image whose pages start one page below 4 GiB, and
/// makes it the active image with a header in slot 1 (`create` wrote slot 0).
fn write_image_past_4_gib(path: &Path, cipher: Option<&ChunkCipher>) -> DbHeaderV3 {
    let mut file: File = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    let below = PageRun {
        first: DATA_START_PAGE,
        count: FOUR_GIB / PAGE_SIZE - 1 - DATA_START_PAGE,
    };
    let mut writer =
        CheckpointWriter::new(&file, PageAllocator::from_used([below]).unwrap(), cipher);
    let mut current = None;
    for (section, meta, bytes) in chunks() {
        if current != Some(section) {
            writer.begin_section(section, 1).unwrap();
            current = Some(section);
        }
        writer.write_chunk(meta, &bytes).unwrap();
    }
    let (root, _) = writer.finish().unwrap();
    assert!(
        root.offset > FOUR_GIB,
        "the directory lies past 4 GiB: {root:?}"
    );

    let header = DbHeaderV3 {
        iteration: 1,
        checkpoint_lsn: 11,
        epoch: 12,
        last_transaction_id: 13,
        root,
        node_count: 3,
        edge_count: 2,
        ..DbHeaderV3::default()
    };
    file.seek(SeekFrom::Start(slot_offset(1))).unwrap();
    file.write_all(&header.encode()).unwrap();
    file.sync_all().unwrap();
    header
}

/// Every chunk of the active image, as (section, chunk, bytes) in the order
/// of [`chunks`].
fn read_back(manager: &GrafeoFileManager) -> Vec<(SectionType, ChunkMeta, Vec<u8>)> {
    manager
        .read_image(|image| {
            let mut read = Vec::new();
            for section in [SectionType::Catalog, SectionType::LpgStore] {
                let source = image.section_source(section).expect("the section is there");
                for (index, meta) in source.chunks().iter().enumerate() {
                    read.push((section, *meta, source.fetch(index)?.to_vec()));
                }
            }
            Ok(read)
        })
        .unwrap()
}

/// The byte range of each chunk of the image `header` points at.
fn active_entries(
    path: &Path,
    header: &DbHeaderV3,
    cipher: Option<&ChunkCipher>,
) -> Vec<(u64, u64)> {
    let file = File::open(path).unwrap();
    ImageReader::open(&file, header.root, cipher)
        .unwrap()
        .entries()
        .iter()
        .map(|entry| (entry.offset, entry.offset + entry.length))
        .collect()
}

/// A section written by the checkpoint over the image past 4 GiB.
struct Fixed(Vec<u8>);

impl Section for Fixed {
    fn section_type(&self) -> SectionType {
        SectionType::Catalog
    }

    fn serialize(&self) -> Result<Vec<u8>> {
        Ok(self.0.clone())
    }

    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        self.0 = data.to_vec();
        Ok(())
    }

    fn write_to(&self, sink: &mut dyn SectionSink) -> Result<()> {
        write_raw(self, sink)
    }

    fn read_from(&mut self, source: &dyn SectionSource) -> Result<()> {
        read_raw(self, source)
    }

    fn is_dirty(&self) -> bool {
        true
    }

    fn mark_clean(&self) {}

    fn memory_usage(&self) -> usize {
        self.0.len()
    }
}

fn image_past_4_gib_reads_back(cipher: impl Fn() -> Option<ChunkCipher>) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("large.grafeo");
    drop(GrafeoFileManager::create(&path, cipher()).unwrap());
    make_sparse(&path);
    let written = write_image_past_4_gib(&path, cipher().as_ref());
    assert!(
        std::fs::metadata(&path).unwrap().len() > FOUR_GIB,
        "the file is longer than 4 GiB"
    );

    // Every chunk lies past 4 GiB, except the first, which straddles it.
    let entries = active_entries(&path, &written, cipher().as_ref());
    assert_eq!(entries.len(), 4);
    assert!(
        entries[0].0 < FOUR_GIB && entries[0].1 > FOUR_GIB,
        "the first chunk straddles 4 GiB: {entries:?}"
    );
    assert!(
        entries[1..3].iter().all(|&(start, _)| start >= FOUR_GIB),
        "the other chunks start past 4 GiB: {entries:?}"
    );
    assert_eq!(entries[3], (0, 0), "the empty chunk takes no pages");

    // The reopened file serves the image: the header as written, every chunk
    // with its metadata and bytes.
    let manager = GrafeoFileManager::open(&path, cipher()).unwrap();
    assert_eq!(manager.active_header(), written);
    assert_eq!(read_back(&manager), chunks());

    // A checkpoint over it goes into the free pages below 4 GiB.
    let catalog = Fixed(noise(3, 20_000));
    manager
        .write_checkpoint(
            &[&catalog],
            &CheckpointHeader {
                checkpoint_lsn: 21,
                epoch: 22,
                last_transaction_id: 23,
                node_count: 4,
                edge_count: 2,
            },
        )
        .unwrap();
    drop(manager);

    let manager = GrafeoFileManager::open(&path, cipher()).unwrap();
    let header = manager.active_header();
    assert_eq!(
        (header.iteration, header.checkpoint_lsn, header.node_count),
        (2, 21, 4)
    );
    assert!(
        header.root.offset < FOUR_GIB,
        "the new image fills the pages below 4 GiB: {:?}",
        header.root
    );
    let read = manager
        .read_image(|image| {
            assert!(image.section_source(SectionType::LpgStore).is_none());
            image
                .section_source(SectionType::Catalog)
                .expect("the catalog is there")
                .fetch(0)
        })
        .unwrap();
    assert_eq!(&read[..], &catalog.0[..]);
    let len = std::fs::metadata(&path).unwrap().len();
    assert_eq!(
        len,
        (header.root.offset + u64::from(header.root.length)).next_multiple_of(PAGE_SIZE),
        "the file is cut at the last page of the new image, below 4 GiB"
    );
}

#[test]
fn an_image_past_4_gib_reads_back_after_a_reopen() {
    image_past_4_gib_reads_back(|| None);
}

#[cfg(feature = "encryption")]
#[test]
fn an_encrypted_image_past_4_gib_reads_back_after_a_reopen() {
    image_past_4_gib_reads_back(|| Some(grafeo_common::encryption::PageEncryptor::new(&[7u8; 32])));
}
