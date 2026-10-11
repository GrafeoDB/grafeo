//! Guarantees of the WAL v2 writer and scanner, end to end through the files
//! they leave. Each test states the guarantee it checks: what is synced is
//! never cut, what a later release wrote is refused rather than cut, what
//! salvage sets aside is never overwritten, and a committer is never told a
//! position is durable when it is not.

#![cfg(feature = "wal")]

use std::path::{Path, PathBuf};
use std::sync::Barrier;

use grafeo_common::types::TransactionId;
use grafeo_storage::wal::{
    DurabilityMode, FrameFlags, FrameHeader, GroupEnd, ScanEnd, ScanOptions, SegmentHeader,
    TailKind, Wal, WalError, WalOptions, WalScan, segment_file_name,
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

fn write_group(wal: &Wal, id: u64, texts: &[&str]) -> GroupEnd {
    let mut group = wal.begin_group(TransactionId::new(id)).unwrap();
    for text in texts {
        group.push(&record(text)).unwrap();
    }
    group.finish().unwrap()
}

fn segment_path(dir: &Path, first_lsn: u64) -> PathBuf {
    dir.join(segment_file_name(first_lsn))
}

/// The writer syncs a segment's header before it writes a frame, so a header
/// of zeros with frames behind it is damage, never a segment cut off during
/// creation: the scan refuses it (salvage too: a damaged header is never set
/// aside), the writer does not append to it, and nothing deletes the synced
/// groups behind it.
#[test]
fn a_zeroed_header_before_synced_frames_in_the_newest_segment_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let wal = Wal::open(dir.path(), options(0, DurabilityMode::Sync)).unwrap();
    write_group(&wal, 1, &["Alix"]);
    let second = wal.rotate().unwrap();
    write_group(&wal, 2, &["Gus", "Mia"]);
    let end = write_group(&wal, 3, &["Vincent"]);
    assert_eq!(
        wal.synced_lsn(),
        end.end_lsn,
        "Sync mode: every group is durable"
    );
    drop(wal);
    let newest = segment_path(dir.path(), second);
    let mut bytes = std::fs::read(&newest).unwrap();
    bytes[..128].fill(0);
    std::fs::write(&newest, &bytes).unwrap();

    let salvage = ScanOptions {
        salvage: true,
        ..ScanOptions::new(DATABASE, 0)
    };
    for options in [ScanOptions::new(DATABASE, 0), salvage] {
        let error = scan_all(dir.path(), options.clone()).unwrap_err();
        assert!(
            matches!(&error, WalError::SegmentHeader { path, reason }
                if path == &newest && reason.contains("damaged")),
            "salvage {}: refused as a damaged header: {error}",
            options.salvage
        );
    }
    let error = Wal::open(dir.path(), options(end.end_lsn, DurabilityMode::Sync)).unwrap_err();
    assert!(
        matches!(error, WalError::SegmentHeader { .. }),
        "the writer does not append to it: {error}"
    );
    assert_eq!(
        std::fs::read(&newest).unwrap(),
        bytes,
        "the segment and its synced frames are untouched"
    );
}

/// Writes a plaintext segment by hand: frames with the given transaction,
/// flags byte and payload, each at its LSN, with a matching checksum.
fn put_segment(dir: &Path, first_lsn: u64, frames: &[(u64, u8, Vec<u8>)]) -> PathBuf {
    let header = SegmentHeader {
        encrypted: false,
        database_id: DATABASE,
        first_lsn,
        creation_time_ms: 1988,
        salt: [0; 32],
        key_check: [0; 28],
    };
    let mut bytes = header.encode().to_vec();
    let mut lsn = first_lsn;
    for (transaction, flags, payload) in frames {
        let length = u32::try_from(payload.len()).unwrap();
        // Encode with known flags, then set the stored flags byte and
        // recompute the checksum over it, as a newer writer would.
        let mut frame = FrameHeader::new(length, lsn, *transaction, FrameFlags::MIDDLE).encode();
        frame[24] = *flags;
        let mut covered = frame[..4].to_vec();
        covered.extend_from_slice(&frame[8..]);
        covered.extend_from_slice(payload);
        let crc = crc32fast::hash(&covered);
        frame[4..8].copy_from_slice(&crc.to_le_bytes());
        bytes.extend_from_slice(&frame);
        bytes.extend_from_slice(payload);
        lsn += 25 + u64::from(length);
    }
    let path = segment_path(dir, first_lsn);
    std::fs::write(&path, bytes).unwrap();
    path
}

fn first_payload(synced_lsn: u64, text: &str) -> Vec<u8> {
    let mut payload = synced_lsn.to_le_bytes().to_vec();
    payload.extend_from_slice(&record(text));
    payload
}

/// A whole frame (its checksum right) with a flag bit this release does not
/// know is refused (design 3.4), wherever it sits and also with salvage: it
/// is a later release's frame, not damage, and cutting it as a torn tail
/// would drop that release's groups.
#[test]
fn a_whole_frame_with_an_unknown_flag_in_the_last_segment_is_refused_not_cut() {
    let salvage = ScanOptions {
        salvage: true,
        ..ScanOptions::new(DATABASE, 0)
    };
    for (what, frames, unsupported_lsn) in [
        (
            "FIRST|LAST with bit 2",
            vec![
                (1, 0x03, first_payload(0, "Alix")),
                (2, 0x07, first_payload(41, "Gus")),
                (3, 0x03, first_payload(41, "Mia")),
            ],
            41,
        ),
        (
            "a middle frame with bit 7",
            vec![
                (1, 0x03, first_payload(0, "Alix")),
                (2, 0x01, first_payload(41, "Gus")),
                (2, 0x82, record("Mia")),
            ],
            41 + 40,
        ),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = put_segment(dir.path(), 0, &frames);
        let length = std::fs::metadata(&path).unwrap().len();
        for options in [ScanOptions::new(DATABASE, 0), salvage.clone()] {
            let error = scan_all(dir.path(), options).unwrap_err();
            match &error {
                WalError::UnsupportedFrame { lsn, reason, .. } => {
                    assert_eq!(*lsn, unsupported_lsn, "{what}: {error}");
                    assert!(reason.contains("flags"), "{what}: {reason}");
                    assert!(error.to_string().contains("newer version"), "{error}");
                }
                other => panic!("{what}: refused as a later release's frame, got {other}"),
            }
        }
        assert_eq!(std::fs::metadata(&path).unwrap().len(), length, "{what}");
    }
}

/// The same flags byte in a frame whose checksum does not match is a torn or
/// damaged frame, classified as any: in the last segment, with nothing
/// synced after it, a torn tail.
#[test]
fn an_unknown_flag_in_a_frame_with_a_wrong_checksum_is_a_torn_tail() {
    let dir = tempfile::tempdir().unwrap();
    let path = put_segment(
        dir.path(),
        0,
        &[
            (1, 0x03, first_payload(0, "Alix")),
            (2, 0x07, first_payload(0, "Gus")),
        ],
    );
    let mut bytes = std::fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0x5A;
    std::fs::write(&path, &bytes).unwrap();
    let (groups, end) = scan_all(dir.path(), ScanOptions::new(DATABASE, 0)).unwrap();
    assert_eq!(groups, [(1, vec!["Alix".to_string()])]);
    assert_eq!(end.tail.map(|tail| tail.kind), Some(TailKind::Torn));
}

/// Past a bad frame in the last segment, the scan looks for a later group
/// that was synced after it. A whole frame of a later release there is
/// refused: what it says about the sync cannot be read, so the bad frame can
/// be neither cut as torn nor called damage.
#[test]
fn a_later_release_frame_past_a_torn_frame_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = put_segment(
        dir.path(),
        0,
        &[
            (1, 0x03, first_payload(0, "Alix")),
            (2, 0x03, first_payload(0, "Gus 0000")),
            (3, 0x07, first_payload(41, "Mia")),
        ],
    );
    // Damage the second frame's payload: a bad frame with a whole frame of a
    // later release after it.
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[128 + 41 + 30] ^= 0x5A;
    std::fs::write(&path, &bytes).unwrap();
    let error = scan_all(dir.path(), ScanOptions::new(DATABASE, 0)).unwrap_err();
    assert!(
        matches!(&error, WalError::UnsupportedFrame { lsn, .. } if *lsn == 41 + 45),
        "{error}"
    );
}

/// Salvage moves the rest to a directory of its own, never deleting it: a
/// second salvage at the same LSN (a bad sector damages the same file offset
/// again after the log was rewritten there) sets its bytes aside next to the
/// first salvage's, which stay as they were.
#[test]
fn a_second_salvage_at_the_same_lsn_keeps_what_the_first_set_aside() {
    let dir = tempfile::tempdir().unwrap();
    let salvage = ScanOptions {
        salvage: true,
        ..ScanOptions::new(DATABASE, 0)
    };
    let wal = Wal::open(dir.path(), options(0, DurabilityMode::NoSync)).unwrap();
    let first = write_group(&wal, 1, &["Alix"]);
    write_group(&wal, 2, &["Butch in Prague"]);
    wal.rotate().unwrap();
    write_group(&wal, 3, &["Berlin"]);
    drop(wal);
    let sealed = segment_path(dir.path(), 0);
    let mut bytes = std::fs::read(&sealed).unwrap();
    let at = 128 + usize::try_from(first.end_lsn).unwrap() + 30;
    bytes[at] ^= 0x5A;
    std::fs::write(&sealed, &bytes).unwrap();
    let (_, end) = scan_all(dir.path(), salvage.clone()).unwrap();
    assert_eq!(end.tail.as_ref().unwrap().kind, TailKind::Damaged);
    end.cut_torn_tail().unwrap();
    let aside = dir
        .path()
        .join(format!("damaged-{:020}", first.end_lsn))
        .join(segment_file_name(0));
    let first_copy = std::fs::read(&aside).unwrap();

    // The log is written again at the same LSN, sealed, and damaged there
    // again.
    let wal = Wal::open(dir.path(), options(first.end_lsn, DurabilityMode::NoSync)).unwrap();
    write_group(&wal, 4, &["Jules in Barcelona, a longer record"]);
    wal.rotate().unwrap();
    write_group(&wal, 5, &["Paris"]);
    drop(wal);
    let mut bytes = std::fs::read(&sealed).unwrap();
    bytes[at] ^= 0x5A;
    std::fs::write(&sealed, &bytes).unwrap();
    let (_, end) = scan_all(dir.path(), salvage).unwrap();
    assert_eq!(end.tail.as_ref().unwrap().kind, TailKind::Damaged);
    assert_eq!(end.end_lsn, first.end_lsn, "the same LSN again");
    end.cut_torn_tail().unwrap();

    let kept: Vec<Vec<u8>> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("damaged-"))
        .flat_map(|entry| std::fs::read_dir(entry.path()).unwrap())
        .filter_map(Result::ok)
        .map(|entry| std::fs::read(entry.path()).unwrap())
        .collect();
    assert!(
        kept.contains(&first_copy),
        "the bytes the first salvage set aside are gone: {} files are kept, none holds the \
         first copy ({} bytes)",
        kept.len(),
        first_copy.len()
    );
    assert_eq!(
        std::fs::read(&aside).unwrap(),
        first_copy,
        "the first copy stays where it was put"
    );
    let second = dir
        .path()
        .join(format!("damaged-{:020}-1", first.end_lsn))
        .join(segment_file_name(0));
    let second_copy = std::fs::read(&second).unwrap();
    assert_ne!(second_copy, first_copy, "the second salvage's own bytes");
    assert_eq!(kept.len(), 4, "two copies and the two later segments");
}

/// The group-commit guarantee: `sync_until` returns `Ok` only once its LSN
/// is covered by a sync, under many concurrent committers that write with
/// `finish_unsynced` and sync after releasing the writer's lock.
#[test]
fn sync_until_returns_only_once_its_lsn_is_durable_under_eight_committers() {
    let dir = tempfile::tempdir().unwrap();
    let wal = Wal::open(dir.path(), options(0, DurabilityMode::Sync)).unwrap();
    let barrier = Barrier::new(8);
    std::thread::scope(|scope| {
        for writer in 0..8u64 {
            let (wal, barrier) = (&wal, &barrier);
            scope.spawn(move || {
                barrier.wait();
                for group in 0..40u64 {
                    let mut frames = wal
                        .begin_group(TransactionId::new(writer * 1000 + group))
                        .unwrap();
                    frames.push(&record("Mia in Amsterdam")).unwrap();
                    let end = frames.finish_unsynced().unwrap();
                    wal.sync_until(end.end_lsn).unwrap();
                    assert!(
                        wal.synced_lsn() >= end.end_lsn,
                        "sync_until returned before LSN {} was synced ({})",
                        end.end_lsn,
                        wal.synced_lsn()
                    );
                }
            });
        }
    });
    // Only the marker of the last sync can follow what is durable.
    wal.sync().unwrap();
    assert_eq!(wal.synced_lsn(), wal.end_lsn());
}

/// After a failed fsync the page cache can no longer be trusted: a committer
/// whose group was written before the failure (pipelined, not yet synced)
/// must not be told it is durable by a later sync that happens to succeed.
#[cfg(feature = "testing-crash-injection")]
#[test]
fn a_committer_waiting_to_sync_after_a_failed_sync_is_never_told_durable() {
    use grafeo_common::testing::crash::with_failure_at;

    let dir = tempfile::tempdir().unwrap();
    let wal = Wal::open(dir.path(), options(0, DurabilityMode::Sync)).unwrap();
    let mut gus = wal.begin_group(TransactionId::new(3)).unwrap();
    gus.push(&record("Gus in Berlin")).unwrap();
    let pending = gus.finish_unsynced().unwrap();
    let mut mia = wal.begin_group(TransactionId::new(19)).unwrap();
    mia.push(&record("Mia in Paris")).unwrap();
    let leader = mia.finish_unsynced().unwrap();
    let failed = with_failure_at(1, || wal.sync_until(leader.end_lsn));
    assert!(failed.is_err(), "the leader's sync failed");
    assert!(wal.synced_lsn() < pending.end_lsn);
    let follower = std::thread::scope(|scope| {
        scope
            .spawn(|| wal.sync_until(pending.end_lsn))
            .join()
            .unwrap()
    });
    assert!(
        follower.is_err(),
        "the follower was told its group is durable after the failed fsync"
    );
}

/// `sync_until` returns `Ok` only when every group up to its LSN is durable.
/// An LSN past the end of the log is reached by nothing written, so it is
/// refused, never answered `Ok` while `synced_lsn` stays below it; the
/// writer stays usable and syncs nothing for it.
#[test]
fn sync_until_past_the_end_of_the_log_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let wal = Wal::open(dir.path(), options(0, DurabilityMode::NoSync)).unwrap();
    let end = write_group(&wal, 1, &["Prague"]);
    let asked = end.end_lsn + 1988;
    let error = wal.sync_until(asked).unwrap_err();
    assert!(
        matches!(error, WalError::BeyondEnd { lsn, end_lsn } if lsn == asked && end_lsn == end.end_lsn),
        "{error}"
    );
    assert_eq!(wal.synced_lsn(), 0, "nothing was synced for it");
    assert!(
        !wal.is_poisoned(),
        "a caller's mistake does not stop the log"
    );
    wal.sync_until(end.end_lsn + 1).unwrap_err();
    wal.sync_until(end.end_lsn).unwrap();
    assert_eq!(wal.synced_lsn(), end.end_lsn, "the end itself is synced");
    // The sync's marker now ends the log.
    wal.sync_until(wal.end_lsn() + 1).unwrap_err();
}

/// A stale segment below the checkpoint is skipped by the scan, so only its
/// header checksum and identity are checked, not its key: a plaintext
/// segment left from before encryption was turned on (a crash after the
/// checkpoint that switched, before the segments were removed) does not fail
/// the open.
#[cfg(feature = "encryption")]
#[test]
fn a_stale_plaintext_segment_below_the_checkpoint_does_not_fail_an_encrypted_open() {
    use std::sync::Arc;

    use grafeo_common::encryption::KeyChain;
    use grafeo_storage::wal::{CipherForSalt, list_wal_directory};

    let chain = Arc::new(KeyChain::new([19; 32]));
    let cipher_for: CipherForSalt = Arc::new(move |salt: &[u8; 32]| {
        let mut id = DATABASE.to_le_bytes().to_vec();
        id.extend_from_slice(salt);
        chain.encryptor_for("grafeo-wal", &id)
    });
    let dir = tempfile::tempdir().unwrap();
    let wal = Wal::open(dir.path(), options(0, DurabilityMode::NoSync)).unwrap();
    let end = write_group(&wal, 1, &["Alix"]);
    drop(wal);
    let checkpoint = end.end_lsn + 88;
    let wal = Wal::open(
        dir.path(),
        WalOptions {
            cipher_for_salt: Some(Arc::clone(&cipher_for)),
            ..options(checkpoint, DurabilityMode::NoSync)
        },
    )
    .unwrap();
    write_group(&wal, 2, &["Gus"]);
    drop(wal);
    let outcome = scan_all(
        dir.path(),
        ScanOptions {
            cipher_for_salt: Some(cipher_for),
            ..ScanOptions::new(DATABASE, checkpoint)
        },
    );
    match outcome {
        Ok((groups, _)) => assert_eq!(groups, [(2, vec!["Gus".to_string()])]),
        Err(error) => panic!("a segment the scan skips failed the open: {error}"),
    }
    assert_eq!(list_wal_directory(dir.path()).unwrap().segments.len(), 2);
}

/// The segment header splits its flags into incompatible (refused when
/// unknown) and compatible (ignored when unknown) bits, and the key check
/// authenticates header bytes 0..80, the flags among them. The key check is
/// verified over those bytes as stored, so an encrypted segment of a later
/// release that sets a compatible flag reads (and takes more groups), never
/// "wrong key".
#[cfg(feature = "encryption")]
#[test]
fn an_unknown_compatible_segment_flag_is_ignored_in_an_encrypted_segment() {
    use std::sync::Arc;

    use grafeo_common::encryption::{KeyChain, random_nonce};
    use grafeo_storage::wal::CipherForSalt;

    let chain = Arc::new(KeyChain::new([3; 32]));
    let cipher_for: CipherForSalt = Arc::new(move |salt: &[u8; 32]| {
        let mut id = DATABASE.to_le_bytes().to_vec();
        id.extend_from_slice(salt);
        chain.encryptor_for("grafeo-wal", &id)
    });
    let dir = tempfile::tempdir().unwrap();
    let wal = Wal::open(
        dir.path(),
        WalOptions {
            cipher_for_salt: Some(Arc::clone(&cipher_for)),
            ..options(0, DurabilityMode::NoSync)
        },
    )
    .unwrap();
    write_group(&wal, 1, &["Vincent in Amsterdam"]);
    drop(wal);

    // What a later release writes: compatible flag bit 16 set, the key
    // check over the header bytes as stored, the checksum over them.
    let path = segment_path(dir.path(), 0);
    let mut bytes = std::fs::read(&path).unwrap();
    let flags = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) | 1 << 16;
    bytes[12..16].copy_from_slice(&flags.to_le_bytes());
    let mut salt = [0u8; 32];
    salt.copy_from_slice(&bytes[48..80]);
    let check = cipher_for(&salt)
        .encrypt(&[], &random_nonce(), &bytes[..80])
        .unwrap();
    bytes[80..108].copy_from_slice(&check);
    let crc = crc32fast::hash(&bytes[..124]);
    bytes[124..128].copy_from_slice(&crc.to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();

    let decoded = SegmentHeader::decode(&bytes, &path).unwrap();
    assert!(
        decoded.encrypted,
        "the header decodes: bit 16 is compatible"
    );
    let scan = ScanOptions {
        cipher_for_salt: Some(Arc::clone(&cipher_for)),
        ..ScanOptions::new(DATABASE, 0)
    };
    let (groups, end) = match scan_all(dir.path(), scan.clone()) {
        Ok(found) => found,
        Err(error) => panic!("a compatible flag made the segment unreadable: {error}"),
    };
    assert_eq!(groups, [(1, vec!["Vincent in Amsterdam".to_string()])]);

    // The writer appends to the segment under the same key check.
    let wal = Wal::open(
        dir.path(),
        WalOptions {
            cipher_for_salt: Some(cipher_for),
            ..options(end.end_lsn, DurabilityMode::NoSync)
        },
    )
    .unwrap();
    write_group(&wal, 2, &["Jules in Berlin"]);
    drop(wal);
    let (groups, _) = scan_all(dir.path(), scan).unwrap();
    assert_eq!(groups.len(), 2, "{groups:?}");
    assert_eq!(
        std::fs::read(&path).unwrap()[..128],
        bytes[..128],
        "the later release's header stays as it was written"
    );
}

#[cfg(feature = "encryption")]
fn encrypted_options(start_lsn: u64) -> WalOptions {
    use std::sync::Arc;

    use grafeo_common::encryption::KeyChain;

    let chain = Arc::new(KeyChain::new([88; 32]));
    WalOptions {
        cipher_for_salt: Some(Arc::new(move |salt: &[u8; 32]| {
            let mut id = DATABASE.to_le_bytes().to_vec();
            id.extend_from_slice(salt);
            chain.encryptor_for("grafeo-wal", &id)
        })),
        ..options(start_lsn, DurabilityMode::NoSync)
    }
}

/// The scanner reads a later group's prologue through the cipher when it
/// classifies a bad frame in an encrypted segment. A hole before a group
/// that began after a sync is damage, encrypted or not; without that sync
/// the same hole is a torn tail.
#[cfg(feature = "encryption")]
#[test]
fn an_encrypted_hole_before_a_group_synced_after_it_is_damage() {
    let dir = tempfile::tempdir().unwrap();
    let writer = encrypted_options(0);
    let scan = ScanOptions {
        cipher_for_salt: writer.cipher_for_salt.clone(),
        ..ScanOptions::new(DATABASE, 0)
    };
    let wal = Wal::open(dir.path(), writer).unwrap();
    write_group(&wal, 1, &["Alix"]);
    let hole = write_group(&wal, 2, &["Gus 00000", "Mia 00000", "Jules 000"]);
    wal.sync().unwrap();
    write_group(&wal, 3, &["Butch"]);
    drop(wal);
    let path = segment_path(dir.path(), 0);
    let mut bytes = std::fs::read(&path).unwrap();
    let at = 128 + usize::try_from(hole.start_lsn).unwrap() + 40;
    bytes[at] ^= 0x5A;
    std::fs::write(&path, &bytes).unwrap();
    match scan_all(dir.path(), scan.clone()) {
        Err(WalError::Damaged { reason, .. }) => {
            assert!(reason.contains("synced"), "{reason}");
        }
        Err(other) => panic!("expected damage, got {other}"),
        Ok((groups, end)) => panic!(
            "a hole before a synced group was cut as {:?}; groups {groups:?}",
            end.tail.map(|tail| tail.kind)
        ),
    }
    // Without the sync, the same hole is a torn tail.
    let dir = tempfile::tempdir().unwrap();
    let wal = Wal::open(dir.path(), encrypted_options(0)).unwrap();
    let scan = ScanOptions {
        cipher_for_salt: encrypted_options(0).cipher_for_salt,
        ..ScanOptions::new(DATABASE, 0)
    };
    write_group(&wal, 1, &["Alix"]);
    let hole = write_group(&wal, 2, &["Gus 00000", "Mia 00000", "Jules 000"]);
    write_group(&wal, 3, &["Butch"]);
    drop(wal);
    let path = segment_path(dir.path(), 0);
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[128 + usize::try_from(hole.start_lsn).unwrap() + 40] ^= 0x5A;
    std::fs::write(&path, &bytes).unwrap();
    let (groups, end) = scan_all(dir.path(), scan).unwrap();
    assert_eq!(groups, [(1, vec!["Alix".to_string()])]);
    assert_eq!(end.tail.map(|tail| tail.kind), Some(TailKind::Torn));
}

/// An encrypted group larger than the scan's buffer is read again frame by
/// frame through the cipher, and returns the records it holds.
#[cfg(feature = "encryption")]
#[test]
fn an_encrypted_large_group_is_read_again_frame_by_frame() {
    let dir = tempfile::tempdir().unwrap();
    let writer = encrypted_options(0);
    let scan = ScanOptions {
        cipher_for_salt: writer.cipher_for_salt.clone(),
        group_buffer_bytes: 256,
        ..ScanOptions::new(DATABASE, 0)
    };
    let wal = Wal::open(dir.path(), writer).unwrap();
    let texts: Vec<String> = (0..88).map(|n| format!("Barcelona {n:03}")).collect();
    let texts: Vec<&str> = texts.iter().map(String::as_str).collect();
    write_group(&wal, 1, &["Alix"]);
    write_group(&wal, 2, &texts);
    write_group(&wal, 3, &["Gus"]);
    drop(wal);
    let (groups, end) = scan_all(dir.path(), scan).unwrap();
    assert_eq!(groups.len(), 3);
    assert_eq!(groups[1].1, texts);
    assert!(end.is_clean());
}

/// The search for a later synced group reads the segment a 1 MiB window at a
/// time: a synced group more than a window past the hole still turns it into
/// damage.
#[test]
fn a_synced_group_more_than_a_window_past_a_hole_makes_it_damage() {
    let dir = tempfile::tempdir().unwrap();
    let wal = Wal::open(
        dir.path(),
        WalOptions {
            durability: DurabilityMode::NoSync,
            ..WalOptions::new(DATABASE, 0)
        },
    )
    .unwrap();
    write_group(&wal, 1, &["Alix"]);
    let hole = write_group(&wal, 2, &["Gus"]);
    // About 1.5 MiB of groups that began before any sync.
    let filler = "Vincent ".repeat(1024);
    for id in 3..200 {
        write_group(&wal, id, &[filler.as_str()]);
    }
    wal.sync().unwrap();
    let last = write_group(&wal, 200, &["Mia"]);
    drop(wal);
    assert!(last.start_lsn - hole.end_lsn > 1 << 20, "past one window");
    let path = segment_path(dir.path(), 0);
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[128 + usize::try_from(hole.start_lsn).unwrap() + 30] ^= 0x5A;
    std::fs::write(&path, &bytes).unwrap();
    match scan_all(dir.path(), ScanOptions::new(DATABASE, 0)) {
        Err(WalError::Damaged { reason, .. }) => assert!(reason.contains("synced"), "{reason}"),
        Err(other) => panic!("expected damage, got {other}"),
        Ok((groups, end)) => panic!(
            "cut as {:?} after {} groups",
            end.tail.map(|tail| tail.kind),
            groups.len()
        ),
    }
}

/// Damage reaches the crate-wide error as a corruption naming the segment
/// file and the byte offset of the damaged frame, so a caller that
/// propagates it with `?` reports where the WAL is damaged.
#[test]
fn damage_becomes_a_corruption_naming_the_segment_and_the_frame() {
    let dir = tempfile::tempdir().unwrap();
    let wal = Wal::open(dir.path(), options(0, DurabilityMode::Sync)).unwrap();
    let damaged = write_group(&wal, 1, &["Alix"]);
    write_group(&wal, 2, &["Gus"]);
    drop(wal);
    let path = segment_path(dir.path(), 0);
    let frame = 128 + damaged.start_lsn;
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[usize::try_from(frame).unwrap() + 30] ^= 0x5A;
    std::fs::write(&path, &bytes).unwrap();

    let error = scan_all(dir.path(), ScanOptions::new(DATABASE, 0)).expect_err("damage");
    assert!(matches!(error, WalError::Damaged { .. }), "{error}");
    let error = grafeo_common::utils::error::Error::from(error);
    let grafeo_common::utils::error::Error::Corruption(corruption) = &error else {
        panic!("damage is a corruption: {error:?}");
    };
    assert_eq!(corruption.file.as_deref(), Some(path.as_path()), "{error}");
    assert_eq!(corruption.offset, Some(frame), "the damaged frame: {error}");
    assert!(corruption.what.contains("LSN"), "{error}");
    assert!(error.to_string().starts_with("GRAFEO-S002"), "{error}");
}

/// In a plaintext segment the checksum covers the frame's LSN, so a whole
/// frame copied to another position (a duplicated block) passes its
/// checksum; the position check keeps it from being replayed a second time.
#[test]
fn a_plaintext_frame_copied_to_another_position_is_not_replayed_again() {
    let dir = tempfile::tempdir().unwrap();
    let wal = Wal::open(dir.path(), options(0, DurabilityMode::NoSync)).unwrap();
    let first = write_group(&wal, 1, &["Alix"]);
    drop(wal);
    let path = segment_path(dir.path(), 0);
    let mut bytes = std::fs::read(&path).unwrap();
    let frame = bytes[128..].to_vec();
    bytes.extend_from_slice(&frame);
    std::fs::write(&path, &bytes).unwrap();
    let (groups, end) = scan_all(dir.path(), ScanOptions::new(DATABASE, 0)).unwrap();
    assert_eq!(groups, [(1, vec!["Alix".to_string()])], "replayed once");
    assert_eq!(end.end_lsn, first.end_lsn);
    assert_eq!(end.tail.map(|tail| tail.kind), Some(TailKind::Torn));
}

/// Reopening appends to the newest segment and reports everything in it as
/// durable, so the open syncs it first: a failure injected at that sync
/// fails the open.
#[cfg(feature = "testing-crash-injection")]
#[test]
fn reopening_syncs_the_segment_it_appends_to_before_reporting_it_durable() {
    use grafeo_common::testing::crash::with_failure_at;

    let dir = tempfile::tempdir().unwrap();
    let wal = Wal::open(dir.path(), options(0, DurabilityMode::NoSync)).unwrap();
    let end = write_group(&wal, 1, &["Jules"]);
    assert_eq!(wal.synced_lsn(), 0);
    drop(wal);
    let outcome = with_failure_at(1, || {
        Wal::open(dir.path(), options(end.end_lsn, DurabilityMode::NoSync))
    });
    let error = outcome.map(|_| ()).unwrap_err();
    assert!(error.to_string().contains("wal:sync"), "{error}");
}

/// Flips a byte inside the payload of the frame at `lsn` of the segment that
/// starts at LSN 0.
fn damage_frame_at(dir: &Path, lsn: u64) {
    let path = segment_path(dir, 0);
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[128 + usize::try_from(lsn).unwrap() + 30] ^= 0x5A;
    std::fs::write(&path, &bytes).unwrap();
}

/// One fsync makes every group written before it durable: committers that
/// wrote with `finish_unsynced` and then ask for their own group share it.
#[test]
fn one_fsync_covers_every_group_written_before_it() {
    let dir = tempfile::tempdir().unwrap();
    let wal = Wal::open(dir.path(), options(0, DurabilityMode::Sync)).unwrap();
    let before = wal.sync_count();
    let ends: Vec<GroupEnd> = (1..=8u64)
        .map(|id| {
            let mut group = wal.begin_group(TransactionId::new(id)).unwrap();
            group.push(&record("Vincent in Berlin")).unwrap();
            group.finish_unsynced().unwrap()
        })
        .collect();
    for end in &ends {
        wal.sync_until(end.end_lsn).unwrap();
    }
    assert_eq!(wal.sync_count() - before, 1, "eight groups, one fsync");
    assert!(wal.synced_lsn() >= ends[7].end_lsn);
}

/// The last synced group of the log has no later group to show that it was
/// durable: the sync marker written after its fsync does, so damage in it
/// is refused, where it used to be cut as a torn tail (losing a synced,
/// acknowledged commit without an error). A scan returns the groups and
/// never the markers.
#[test]
fn damage_in_the_last_synced_group_is_refused_not_cut() {
    let dir = tempfile::tempdir().unwrap();
    let wal = Wal::open(dir.path(), options(0, DurabilityMode::Sync)).unwrap();
    write_group(&wal, 1, &["Alix"]);
    let last = write_group(&wal, 2, &["Gus in Prague"]);
    drop(wal);

    let (groups, end) = scan_all(dir.path(), ScanOptions::new(DATABASE, 0)).unwrap();
    let ids: Vec<u64> = groups.iter().map(|group| group.0).collect();
    assert_eq!(ids, [1, 2], "the markers are no groups of the scan");
    assert!(end.tail.is_none(), "{:?}", end.tail);
    assert!(
        end.end_lsn > last.end_lsn,
        "the log ends after the last marker"
    );

    damage_frame_at(dir.path(), last.start_lsn);
    let error = scan_all(dir.path(), ScanOptions::new(DATABASE, 0)).unwrap_err();
    assert!(matches!(error, WalError::Damaged { .. }), "{error}");
}

/// A sync that covers nothing but the last marker writes no marker of its
/// own: an idle log stops growing.
#[test]
fn an_idle_sync_writes_no_further_marker() {
    let dir = tempfile::tempdir().unwrap();
    let wal = Wal::open(dir.path(), options(0, DurabilityMode::Sync)).unwrap();
    write_group(&wal, 1, &["Mia in Paris"]);
    wal.sync().unwrap();
    let end = wal.end_lsn();
    assert_eq!(wal.synced_lsn(), end, "the marker itself is synced");
    for _ in 0..3 {
        wal.sync().unwrap();
    }
    assert_eq!(wal.end_lsn(), end);
}

/// A group that began before a sync and finishes after it cannot carry the
/// sync in its prologue, and the marker cannot be written while the group
/// holds the writer: it follows the group's LAST frame.
#[test]
fn a_marker_follows_a_group_that_was_being_written_during_the_sync() {
    let dir = tempfile::tempdir().unwrap();
    let wal = Wal::open(dir.path(), options(0, DurabilityMode::NoSync)).unwrap();
    let mut first = wal.begin_group(TransactionId::new(1)).unwrap();
    first.push(&record("Jules in Barcelona")).unwrap();
    let first = first.finish_unsynced().unwrap();
    let mut second = wal.begin_group(TransactionId::new(2)).unwrap();
    second.push(&record("Butch in Amsterdam")).unwrap();
    wal.sync_until(first.end_lsn).unwrap();
    let second = second.finish_unsynced().unwrap();
    assert!(
        wal.end_lsn() > second.end_lsn,
        "a marker follows the second group"
    );
    drop(wal);

    damage_frame_at(dir.path(), first.start_lsn);
    let error = scan_all(dir.path(), ScanOptions::new(DATABASE, 0)).unwrap_err();
    assert!(matches!(error, WalError::Damaged { .. }), "{error}");
}

/// A writer reopened at the end of a log that ends with a marker appends
/// after it, and the scan reads the groups on both sides.
#[test]
fn a_reopened_writer_appends_after_the_last_marker() {
    let dir = tempfile::tempdir().unwrap();
    let wal = Wal::open(dir.path(), options(0, DurabilityMode::Sync)).unwrap();
    write_group(&wal, 1, &["Django"]);
    drop(wal);
    let (_, end) = scan_all(dir.path(), ScanOptions::new(DATABASE, 0)).unwrap();
    let wal = Wal::open(dir.path(), options(end.end_lsn, DurabilityMode::Sync)).unwrap();
    write_group(&wal, 2, &["Shosanna"]);
    drop(wal);
    let (groups, _) = scan_all(dir.path(), ScanOptions::new(DATABASE, 0)).unwrap();
    let ids: Vec<u64> = groups.iter().map(|group| group.0).collect();
    assert_eq!(ids, [1, 2]);
}

/// A segment key seals a bounded number of frames: once a segment holds
/// that many, the next group starts a new segment (and so a new key),
/// whatever the segment size.
#[test]
fn a_group_starts_a_new_segment_once_the_key_sealed_its_share_of_frames() {
    let dir = tempfile::tempdir().unwrap();
    let wal = Wal::open(
        dir.path(),
        WalOptions {
            frames_per_segment_key: 4,
            ..options(0, DurabilityMode::NoSync)
        },
    )
    .unwrap();
    // Two frames per group (16-byte frames, one record each).
    let texts = ["Hans in Berlin", "Beatrix in Paris"];
    write_group(&wal, 1, &texts);
    let second = write_group(&wal, 2, &texts);
    let third = write_group(&wal, 3, &texts);
    assert!(segment_path(dir.path(), 0).exists());
    assert!(
        !segment_path(dir.path(), second.start_lsn).exists(),
        "two frames so far: the second group stays in the first segment"
    );
    assert!(
        segment_path(dir.path(), third.start_lsn).exists(),
        "four frames: the third group starts a segment"
    );
    drop(wal);
    let (groups, _) = scan_all(dir.path(), ScanOptions::new(DATABASE, 0)).unwrap();
    assert_eq!(groups.len(), 3);
}

/// A writer reopened on a segment counts the frames it may already hold (at
/// most one per smallest frame), so the bound holds across reopens.
#[test]
fn a_reopened_writer_counts_the_frames_its_segment_may_hold() {
    let dir = tempfile::tempdir().unwrap();
    let limited = |start_lsn| WalOptions {
        frames_per_segment_key: 2,
        ..options(start_lsn, DurabilityMode::NoSync)
    };
    let wal = Wal::open(dir.path(), limited(0)).unwrap();
    let first = write_group(&wal, 1, &["Lucas in Prague", "Marcus in Paris"]);
    drop(wal);
    let wal = Wal::open(dir.path(), limited(first.end_lsn)).unwrap();
    let second = write_group(&wal, 2, &["Harm"]);
    assert!(
        segment_path(dir.path(), second.start_lsn).exists(),
        "the reopened segment already held its share of frames"
    );
}
