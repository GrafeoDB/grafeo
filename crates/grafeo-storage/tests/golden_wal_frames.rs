//! Golden fixture tests for WAL frame format stability.
//!
//! These tests deserialize a committed binary fixture (`golden_wal_v1.bin`)
//! containing hand-crafted WAL frames and verify the current code can still
//! parse them. If a code change alters the frame layout (length prefix, CRC
//! position), bincode encoding of `WalRecord`, or enum variant ordering,
//! these tests fail immediately.
//!
//! Frame format: `[length: u32 LE][data: bytes][crc32: u32 LE]`
//!
//! ## When these tests fail
//!
//! - **Accidental breakage**: fix the regression.
//! - **Intentional format change**: regenerate the fixture:
//!   ```
//!   cargo test --all-features -p grafeo-storage --test main -- golden_wal_frames::regenerate_wal_fixture --ignored
//!   ```

#![cfg(feature = "wal")]

use grafeo_common::types::{EdgeId, NodeId, TransactionId, Value};
use grafeo_storage::wal::WalRecord;

/// The set of WAL records embedded in the golden fixture.
/// Order matters: it must match the order in the fixture file.
fn golden_records() -> Vec<WalRecord> {
    vec![
        WalRecord::CreateNode {
            id: NodeId::new(1),
            labels: vec!["Person".to_string()],
        },
        WalRecord::SetNodeProperty {
            id: NodeId::new(1),
            key: "name".to_string(),
            value: Value::String("Alix".into()),
        },
        WalRecord::CreateNode {
            id: NodeId::new(2),
            labels: vec!["Person".to_string()],
        },
        WalRecord::CreateEdge {
            id: EdgeId::new(1),
            src: NodeId::new(1),
            dst: NodeId::new(2),
            edge_type: "KNOWS".to_string(),
        },
        WalRecord::SetEdgeProperty {
            id: EdgeId::new(1),
            key: "since".to_string(),
            value: Value::Int64(2020),
        },
        WalRecord::DeleteEdge { id: EdgeId::new(1) },
        WalRecord::DeleteNode { id: NodeId::new(2) },
        WalRecord::TransactionCommit {
            transaction_id: TransactionId::new(1),
        },
        WalRecord::Checkpoint {
            transaction_id: TransactionId::new(1),
        },
    ]
}

/// Encode a single WAL record as a frame: `[length: u32 LE][data][crc32: u32 LE]`.
fn encode_frame(record: &WalRecord) -> Vec<u8> {
    let data = bincode::serde::encode_to_vec(record, bincode::config::standard()).unwrap();
    let mut frame = Vec::with_capacity(4 + data.len() + 4);
    // reason: WAL frames are small (bytes to KBs), well within u32::MAX
    #[allow(clippy::cast_possible_truncation)]
    let data_len = data.len() as u32;
    frame.extend_from_slice(&data_len.to_le_bytes());
    frame.extend_from_slice(&data);
    frame.extend_from_slice(&crc32fast::hash(&data).to_le_bytes());
    frame
}

/// Parse all frames from raw bytes, returning `(record, data_bytes)` pairs.
fn parse_frames(mut bytes: &[u8]) -> Vec<(WalRecord, Vec<u8>)> {
    let mut results = Vec::new();
    while bytes.len() >= 8 {
        let len = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
        if bytes.len() < 4 + len + 4 {
            break;
        }
        let data = &bytes[4..4 + len];
        let stored_crc = u32::from_le_bytes(bytes[4 + len..4 + len + 4].try_into().unwrap());
        let actual_crc = crc32fast::hash(data);
        assert_eq!(stored_crc, actual_crc, "CRC mismatch in WAL frame");

        let (record, _): (WalRecord, _) =
            bincode::serde::decode_from_slice(data, bincode::config::standard()).unwrap();
        results.push((record, data.to_vec()));
        bytes = &bytes[4 + len + 4..];
    }
    results
}

fn golden_bytes() -> &'static [u8] {
    include_bytes!("fixtures/golden_wal_v1.bin")
}

// ---------------------------------------------------------------------------
// Backward-read tests
// ---------------------------------------------------------------------------

#[test]
fn golden_wal_frame_count() {
    let frames = parse_frames(golden_bytes());
    assert_eq!(
        frames.len(),
        golden_records().len(),
        "expected {} frames, got {}",
        golden_records().len(),
        frames.len(),
    );
}

#[test]
fn golden_wal_crc_integrity() {
    // parse_frames asserts CRC per frame, so if this succeeds, all CRCs match.
    let _ = parse_frames(golden_bytes());
}

#[test]
fn golden_wal_create_node() {
    let frames = parse_frames(golden_bytes());
    match &frames[0].0 {
        WalRecord::CreateNode { id, labels } => {
            assert_eq!(*id, NodeId::new(1));
            assert_eq!(labels, &["Person"]);
        }
        other => panic!("expected CreateNode, got {other:?}"),
    }
}

#[test]
fn golden_wal_set_node_property() {
    let frames = parse_frames(golden_bytes());
    match &frames[1].0 {
        WalRecord::SetNodeProperty { id, key, value } => {
            assert_eq!(*id, NodeId::new(1));
            assert_eq!(key, "name");
            assert_eq!(*value, Value::String("Alix".into()));
        }
        other => panic!("expected SetNodeProperty, got {other:?}"),
    }
}

#[test]
fn golden_wal_create_edge() {
    let frames = parse_frames(golden_bytes());
    match &frames[3].0 {
        WalRecord::CreateEdge {
            id,
            src,
            dst,
            edge_type,
        } => {
            assert_eq!(*id, EdgeId::new(1));
            assert_eq!(*src, NodeId::new(1));
            assert_eq!(*dst, NodeId::new(2));
            assert_eq!(edge_type, "KNOWS");
        }
        other => panic!("expected CreateEdge, got {other:?}"),
    }
}

#[test]
fn golden_wal_commit_and_checkpoint() {
    let frames = parse_frames(golden_bytes());
    match &frames[7].0 {
        WalRecord::TransactionCommit { transaction_id } => {
            assert_eq!(*transaction_id, TransactionId::new(1));
        }
        other => panic!("expected TransactionCommit, got {other:?}"),
    }
    match &frames[8].0 {
        WalRecord::Checkpoint { transaction_id } => {
            assert_eq!(*transaction_id, TransactionId::new(1));
        }
        other => panic!("expected Checkpoint, got {other:?}"),
    }
}

#[test]
fn golden_wal_byte_equality() {
    // Re-encode all golden records and verify byte-for-byte match.
    let mut fresh = Vec::new();
    for record in &golden_records() {
        fresh.extend_from_slice(&encode_frame(record));
    }
    assert_eq!(
        fresh.as_slice(),
        golden_bytes(),
        "WAL frame bytes differ, encoding may have changed",
    );
}

// ---------------------------------------------------------------------------
// Generator
// ---------------------------------------------------------------------------

#[test]
#[ignore = "one-shot fixture generator, not a regular test"]
fn regenerate_wal_fixture() {
    let mut bytes = Vec::new();
    for record in &golden_records() {
        bytes.extend_from_slice(&encode_frame(record));
    }

    let dest = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/golden_wal_v1.bin"
    );
    std::fs::write(dest, &bytes).unwrap();
    println!(
        "Wrote {} bytes ({} frames) to {dest}",
        bytes.len(),
        golden_records().len()
    );
}
