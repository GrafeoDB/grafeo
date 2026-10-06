//! Golden fixture tests for the `.grafeo` file format.
//!
//! These tests open a **committed** binary fixture (`golden_v1.grafeo`) and
//! verify the current code can still parse the file header, database headers,
//! and snapshot payload. If a code change alters the header encoding, magic
//! bytes, or page layout, these tests fail immediately.
//!
//! ## When these tests fail
//!
//! - **Accidental breakage** (no `FORMAT_VERSION` bump): fix the regression.
// Test constants are small known values, casts are safe
#![allow(clippy::cast_possible_truncation)]
//! - **Intentional format change** (version bumped): regenerate the fixture:
//!   ```
//!   cargo test --all-features -p grafeo-adapters --test golden_grafeo_file -- regenerate_grafeo_fixture --ignored
//!   ```
//!   Then commit the new fixture and update the version constant below.

#![cfg(feature = "grafeo-file")]

use grafeo_storage::file::format::{DATA_OFFSET, DB_HEADER_SIZE, FILE_HEADER_SIZE, MAGIC};
use grafeo_storage::file::legacy::{LegacyContents, LegacyFile};

/// Must match `FORMAT_VERSION` in `format.rs`.
const EXPECTED_FORMAT_VERSION: u32 = 1;

/// Known snapshot payload embedded in the golden fixture.
const GOLDEN_SNAPSHOT: &[u8] = b"golden-grafeo-file-test-payload-v1";

/// Copy the committed fixture to a temp directory so `LegacyFile::open`
/// can acquire a file lock.
fn open_golden_fixture() -> (tempfile::TempDir, LegacyFile<'static>) {
    let fixture = include_bytes!("fixtures/golden_v1.grafeo");
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("golden.grafeo");
    std::fs::write(&path, fixture).unwrap();
    let file = LegacyFile::open(&path, None).unwrap();
    (dir, file)
}

// ---------------------------------------------------------------------------
// Structural validation: can today's code parse yesterday's file?
// ---------------------------------------------------------------------------

#[test]
fn golden_file_opens() {
    let (_dir, _mgr) = open_golden_fixture();
}

#[test]
fn golden_magic_bytes() {
    let fixture = include_bytes!("fixtures/golden_v1.grafeo");
    assert_eq!(&fixture[..4], &MAGIC, "magic bytes mismatch");
}

#[test]
fn golden_format_version() {
    let fixture = include_bytes!("fixtures/golden_v1.grafeo");
    // bincode standard config: magic is 4 bytes, format_version follows as u32
    // Read the FileHeader via the official deserialization path
    let (header, _): (grafeo_storage::file::format::FileHeader, _) =
        bincode::serde::decode_from_slice(
            &fixture[..FILE_HEADER_SIZE as usize],
            bincode::config::standard(),
        )
        .unwrap();

    assert_eq!(header.magic, MAGIC);
    assert_eq!(
        header.format_version, EXPECTED_FORMAT_VERSION,
        "fixture format version ({}) does not match expected ({})",
        header.format_version, EXPECTED_FORMAT_VERSION,
    );
    assert_eq!(header.page_size, FILE_HEADER_SIZE as u32);
}

#[test]
fn golden_file_size() {
    let fixture = include_bytes!("fixtures/golden_v1.grafeo");
    let expected_size = DATA_OFFSET as usize + GOLDEN_SNAPSHOT.len();
    assert_eq!(
        fixture.len(),
        expected_size,
        "file size mismatch: expected {expected_size}, got {}",
        fixture.len(),
    );
}

#[test]
fn golden_db_header_fields() {
    let (_dir, mgr) = open_golden_fixture();
    let header = mgr.header();

    assert_eq!(header.iteration, 1, "expected one checkpoint");
    assert_eq!(
        header.snapshot_length,
        GOLDEN_SNAPSHOT.len() as u64,
        "snapshot length mismatch"
    );
    assert_eq!(header.epoch, 42);
    assert_eq!(header.transaction_id, 7);
    assert_eq!(header.node_count, 3);
    assert_eq!(header.edge_count, 2);

    // CRC-32 of the known payload
    let expected_crc = crc32fast::hash(GOLDEN_SNAPSHOT);
    assert_eq!(header.checksum, expected_crc, "CRC-32 checksum mismatch");
}

#[test]
fn golden_snapshot_payload() {
    let (_dir, mgr) = open_golden_fixture();
    let contents = mgr.contents().unwrap();
    assert_eq!(
        contents,
        LegacyContents::Snapshot(GOLDEN_SNAPSHOT.to_vec()),
        "snapshot payload mismatch"
    );
}

#[test]
fn golden_dual_header_layout() {
    // Verify the file has two DB header slots at the expected offsets.
    // Slot 0 starts at FILE_HEADER_SIZE, slot 1 at FILE_HEADER_SIZE + DB_HEADER_SIZE.
    let fixture = include_bytes!("fixtures/golden_v1.grafeo");

    let h0_offset = FILE_HEADER_SIZE as usize;
    let h1_offset = (FILE_HEADER_SIZE + DB_HEADER_SIZE) as usize;

    // Deserialize both slots
    let (h0, _): (grafeo_storage::file::format::DbHeader, _) = bincode::serde::decode_from_slice(
        &fixture[h0_offset..h0_offset + DB_HEADER_SIZE as usize],
        bincode::config::standard(),
    )
    .unwrap();

    let (h1, _): (grafeo_storage::file::format::DbHeader, _) = bincode::serde::decode_from_slice(
        &fixture[h1_offset..h1_offset + DB_HEADER_SIZE as usize],
        bincode::config::standard(),
    )
    .unwrap();

    // The fixture was written once, so one slot has iteration=1, the other is empty.
    // GrafeoFileManager writes to the *inactive* slot (slot 1 first time).
    let (active, inactive) = if h0.iteration > h1.iteration {
        (&h0, &h1)
    } else {
        (&h1, &h0)
    };

    assert_eq!(active.iteration, 1);
    assert_eq!(inactive.iteration, 0, "inactive slot should be empty");
}

// ---------------------------------------------------------------------------
// Generator: regenerate the golden fixture
// ---------------------------------------------------------------------------

#[test]
#[ignore = "one-shot fixture generator, not a regular test"]
fn regenerate_grafeo_fixture() {
    use std::io::{Seek, SeekFrom, Write};

    use grafeo_storage::file::format::{DbHeader, FileHeader};
    use grafeo_storage::file::header::{write_db_header, write_file_header};

    // 0.6.x no longer writes container v1, so the fixture is built from its
    // layout: the header of one checkpoint in slot 1, the snapshot after the
    // headers.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("golden.grafeo");
    {
        let mut file = std::fs::File::create(&path).unwrap();
        write_file_header(&mut file, &FileHeader::new()).unwrap();
        write_db_header(&mut file, 0, &DbHeader::EMPTY).unwrap();
        write_db_header(
            &mut file,
            1,
            &DbHeader {
                iteration: 1,
                checksum: crc32fast::hash(GOLDEN_SNAPSHOT),
                snapshot_length: GOLDEN_SNAPSHOT.len() as u64,
                epoch: 42,
                transaction_id: 7,
                node_count: 3,
                edge_count: 2,
                timestamp_ms: u64::try_from(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_millis(),
                )
                .unwrap(),
            },
        )
        .unwrap();
        file.seek(SeekFrom::Start(DATA_OFFSET)).unwrap();
        file.write_all(GOLDEN_SNAPSHOT).unwrap();
    }

    let bytes = std::fs::read(&path).unwrap();
    let dest = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/golden_v1.grafeo"
    );
    std::fs::write(dest, &bytes).unwrap();
    println!("Wrote {} bytes to {dest}", bytes.len());
}
