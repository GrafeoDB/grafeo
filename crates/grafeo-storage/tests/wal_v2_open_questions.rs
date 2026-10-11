//! The `synced_lsn` evidence in a group's prologue with a pipelined group
//! commit, where one fsync acknowledges several groups that all began before
//! it: the writer appends a sync marker (an empty group whose prologue
//! records the sync) after such an fsync, so damage in a group that was
//! synced is damage, never a torn tail.

#![cfg(feature = "wal")]

use std::path::{Path, PathBuf};

use grafeo_common::types::TransactionId;
use grafeo_storage::wal::{
    DurabilityMode, ScanEnd, ScanOptions, Wal, WalError, WalOptions, WalScan, segment_file_name,
};

const DATABASE: u128 = 0x0003_0019_0088_1988_0319_0003_0019_0088;

fn record(text: &str) -> Vec<u8> {
    let mut bytes = u32::try_from(text.len()).unwrap().to_le_bytes().to_vec();
    bytes.extend_from_slice(text.as_bytes());
    bytes
}

fn records_of(mut payload: &[u8]) -> Vec<String> {
    let mut records = Vec::new();
    while !payload.is_empty() {
        let length = usize::try_from(u32::from_le_bytes(payload[..4].try_into().unwrap())).unwrap();
        records.push(String::from_utf8(payload[4..4 + length].to_vec()).unwrap());
        payload = &payload[4 + length..];
    }
    records
}

type Group = (u64, Vec<String>);

fn scan_all(dir: &Path, options: ScanOptions) -> Result<(Vec<Group>, ScanEnd), WalError> {
    let mut scan = WalScan::open(dir, options)?;
    let mut groups = Vec::new();
    while let Some(mut frames) = scan.next_group()? {
        let mut records = Vec::new();
        while let Some(payload) = frames.next_payload()? {
            records.extend(records_of(payload));
        }
        groups.push((frames.transaction_id().as_u64(), records));
    }
    Ok((groups, scan.finish()?))
}

fn options(start_lsn: u64, durability: DurabilityMode) -> WalOptions {
    WalOptions {
        durability,
        frame_target_bytes: 16,
        ..WalOptions::new(DATABASE, start_lsn)
    }
}

fn write_group(wal: &Wal, id: u64, texts: &[&str]) {
    let mut group = wal.begin_group(TransactionId::new(id)).unwrap();
    for text in texts {
        group.push(&record(text)).unwrap();
    }
    group.finish().unwrap();
}

fn segment_path(dir: &Path, first_lsn: u64) -> PathBuf {
    dir.join(segment_file_name(first_lsn))
}

/// With a pipelined commit, two groups are written, then one fsync makes both
/// durable and both are acknowledged. Neither prologue can show that sync
/// (both groups began before it): the sync marker after it does, so damage
/// in the first one is refused instead of cut as a torn tail, which would
/// drop both acknowledged, synced groups without an error.
#[test]
fn damage_in_a_synced_group_followed_only_by_groups_of_the_same_sync_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let wal = Wal::open(dir.path(), options(0, DurabilityMode::Sync)).unwrap();
    write_group(&wal, 1, &["Alix"]);
    let mut gus = wal.begin_group(TransactionId::new(2)).unwrap();
    gus.push(&record("Gus in Prague")).unwrap();
    let second = gus.finish_unsynced().unwrap();
    let mut mia = wal.begin_group(TransactionId::new(3)).unwrap();
    mia.push(&record("Mia in Paris")).unwrap();
    let third = mia.finish_unsynced().unwrap();
    wal.sync_until(second.end_lsn).unwrap();
    wal.sync_until(third.end_lsn).unwrap();
    assert_eq!(
        wal.synced_lsn(),
        third.end_lsn,
        "both acknowledged as durable"
    );
    drop(wal);
    let path = segment_path(dir.path(), 0);
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[128 + usize::try_from(second.start_lsn).unwrap() + 30] ^= 0x5A;
    std::fs::write(&path, &bytes).unwrap();
    match scan_all(dir.path(), ScanOptions::new(DATABASE, 0)) {
        Err(error) => assert!(matches!(error, WalError::Damaged { .. }), "{error}"),
        Ok((groups, end)) => panic!(
            "two synced, acknowledged groups were cut as {:?}: the scan kept {groups:?} and \
             ends at LSN {} instead of {}",
            end.tail.map(|tail| tail.kind),
            end.end_lsn,
            third.end_lsn
        ),
    }
}
