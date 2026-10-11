//! Crash tests for the WAL v2 writer and scanner.
//!
//! Each crash runs in a child process that exits inside a panic hook at the
//! injected crash point, so nothing unwinds and no destructor runs (a
//! `GroupWriter` dropped by an unwinding panic would cut its partial group
//! back, which a real crash never does). The parent then scans what the
//! child left, cuts the torn tail, reopens the writer and writes on.
//!
//! The guarantee: after a crash at any point, the log holds exactly the
//! groups whose LAST frame was written, in order and whole; the next open
//! cuts everything after them, and the writer continues right there. The
//! sweeps cover writing (without syncs, encrypted, and syncing every group),
//! a failed group's cut back, the removal of segments below a checkpoint,
//! the cut of a torn tail, the removal of a segment cut off while it was
//! created, and the salvage that sets damaged bytes aside (which must keep
//! every byte it moves, whatever step the crash cuts short).
//!
//! The crash model is a process crash: what was written reaches the files.
//!
//! Requires the `testing-crash-injection` feature.

#![cfg(all(feature = "wal", feature = "testing-crash-injection"))]

use std::collections::BTreeSet;
use std::io::Write;
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::process::Command;

use grafeo_common::testing::child_process;
use grafeo_common::testing::crash::{CrashResult, with_crash_at, with_failure_at};
use grafeo_common::types::TransactionId;
use grafeo_storage::wal::{
    DurabilityMode, ScanEnd, ScanOptions, Wal, WalOptions, WalScan, list_wal_directory,
    segment_file_name,
};

const SCENARIO_VAR: &str = "GRAFEO_WAL_V2_CRASH_SCENARIO";
const DIR_VAR: &str = "GRAFEO_WAL_V2_CRASH_DIR";
const POINT_VAR: &str = "GRAFEO_WAL_V2_CRASH_POINT";
/// Exit code of a child that crashed at its crash point.
const CRASHED: i32 = 3;
/// Exit code of a child that panicked anywhere else.
const UNEXPECTED: i32 = 19;
const DATABASE: u128 = 0x0088_0019_0003_1988_0319_0088_0019_0003;
/// More crash points than any scenario has: a sweep that reaches it never
/// completed.
const MAX_POINTS: u64 = 1_000;

/// A planned group: transaction id and records.
type Group = (u64, Vec<String>);

/// The groups the child writes: one to six records each, so most groups
/// span several frames and the small segments rotate between them.
fn planned_groups() -> Vec<Group> {
    (1..=9u64)
        .map(|id| {
            let records = (0..=(id % 6))
                .map(|index| format!("Vincent {id:02} {index}"))
                .collect();
            (id, records)
        })
        .collect()
}

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

#[cfg(feature = "encryption")]
fn cipher_for() -> grafeo_storage::wal::CipherForSalt {
    use grafeo_common::encryption::KeyChain;
    let chain = std::sync::Arc::new(KeyChain::new([19; 32]));
    std::sync::Arc::new(move |salt: &[u8; 32]| {
        let mut id = DATABASE.to_le_bytes().to_vec();
        id.extend_from_slice(salt);
        chain.encryptor_for("grafeo-wal", &id)
    })
}

/// Whether `scenario` writes an encrypted log.
fn encrypted(scenario: &str) -> bool {
    let encrypted = scenario.ends_with("encrypted");
    assert!(
        !encrypted || cfg!(feature = "encryption"),
        "{scenario} needs the encryption feature"
    );
    encrypted
}

fn writer_options(scenario: &str, start_lsn: u64) -> WalOptions {
    let durability = if scenario.ends_with("sync") {
        DurabilityMode::Sync
    } else {
        DurabilityMode::NoSync
    };
    let options = WalOptions {
        durability,
        frame_target_bytes: 24,
        segment_bytes: 320,
        ..WalOptions::new(DATABASE, start_lsn)
    };
    if encrypted(scenario) {
        #[cfg(feature = "encryption")]
        return WalOptions {
            cipher_for_salt: Some(cipher_for()),
            ..options
        };
    }
    options
}

fn scan_options(scenario: &str) -> ScanOptions {
    let options = ScanOptions {
        salvage: scenario == "salvage",
        ..ScanOptions::new(DATABASE, 0)
    };
    if encrypted(scenario) {
        #[cfg(feature = "encryption")]
        return ScanOptions {
            cipher_for_salt: Some(cipher_for()),
            ..options
        };
    }
    options
}

fn scan(dir: &Path, scenario: &str) -> (Vec<Group>, ScanEnd) {
    scan_from(dir, scenario, 0)
}

/// Scans the log in `dir` from the checkpoint at `from_lsn`.
fn scan_from(dir: &Path, scenario: &str, from_lsn: u64) -> (Vec<Group>, ScanEnd) {
    let options = ScanOptions {
        from_lsn,
        ..scan_options(scenario)
    };
    let mut scan = WalScan::open(dir, options).unwrap();
    let mut groups = Vec::new();
    while let Some(mut frames) = scan.next_group().unwrap() {
        let mut records = Vec::new();
        while let Some(payload) = frames.next_payload().unwrap() {
            records.extend(records_of(payload));
        }
        groups.push((frames.transaction_id().as_u64(), records));
    }
    (groups, scan.finish().unwrap())
}

fn write_group(wal: &Wal, (id, records): &Group) {
    let mut group = wal.begin_group(TransactionId::new(*id)).unwrap();
    for text in records {
        group.push(&record(text)).unwrap();
    }
    group.finish().unwrap();
}

/// The message of an injected crash, followed by the point's name.
const CRASH_MESSAGE: &str = "crash injection at: ";

/// Runs `scenario` in a child process that crashes at `point`; returns the
/// name of the crash point it crashed at, or `None` when it ran to
/// completion instead. The child's output is captured, and shown when it
/// fails some other way (any other panic included).
fn run_child(scenario: &str, base: &Path, point: u64) -> Option<String> {
    let output = child_process::output(
        Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "crash_child", "--nocapture"])
            .env(SCENARIO_VAR, scenario)
            .env(DIR_VAR, base)
            .env(POINT_VAR, point.to_string()),
    )
    .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    match output.status.code() {
        Some(0) => None,
        Some(CRASHED) => {
            let at = stderr
                .find(CRASH_MESSAGE)
                .expect("the crash names its point");
            let name = stderr[at + CRASH_MESSAGE.len()..]
                .lines()
                .next()
                .unwrap_or_default();
            Some(name.trim().to_string())
        }
        other => {
            panic!("{scenario} at crash point {point}: the child failed with {other:?}:\n{stderr}")
        }
    }
}

/// Asserts that a sweep crashed at every one of `points`.
fn assert_crashed_at(scenario: &str, seen: &BTreeSet<String>, points: &[&str]) {
    for point in points {
        assert!(
            seen.contains(*point),
            "{scenario}: the sweep never crashed at {point}; it crashed at {seen:?}"
        );
    }
}

/// Child-process entry; a no-op when run directly.
#[test]
fn crash_child() {
    let (Ok(scenario), Some(base), Ok(point)) = (
        std::env::var(SCENARIO_VAR),
        std::env::var_os(DIR_VAR),
        std::env::var(POINT_VAR),
    ) else {
        return;
    };
    // Exit at the crash point, before anything unwinds; any other panic is a
    // failure of the child, not a crash.
    std::panic::set_hook(Box::new(|info| {
        let message = info.to_string();
        eprintln!("{message}");
        std::process::exit(if message.contains(CRASH_MESSAGE) {
            CRASHED
        } else {
            UNEXPECTED
        });
    }));
    let base = PathBuf::from(base);
    let point: u64 = point.parse().unwrap();
    let wal_dir = base.join("wal");
    let progress = base.join("progress.txt");
    let result = match scenario.as_str() {
        // The parent left a log to open; the child cuts what follows it.
        "cut" | "unfinished" | "salvage" => with_crash_at(point, || {
            let (_, end) = scan(&wal_dir, &scenario);
            end.cut_torn_tail().unwrap();
        }),
        // The log is written without crash points; the removal has them.
        "remove" => {
            let (wal, checkpoint) = log_with_a_checkpoint(&wal_dir, &scenario);
            let wal = AssertUnwindSafe(wal);
            with_crash_at(point, move || {
                wal.remove_segments_before(checkpoint).unwrap();
            })
        }
        "cut-back" => {
            let scenario = AssertUnwindSafe(scenario);
            with_crash_at(point, move || {
                let wal = Wal::open(&wal_dir, writer_options(&scenario, 0)).unwrap();
                for group in planned_groups() {
                    if group.0 == FAILED_GROUP {
                        write_a_failing_group(&wal, &group);
                    } else {
                        write_group(&wal, &group);
                        report_finished(&progress, group.0);
                    }
                }
            })
        }
        _ => {
            let scenario = AssertUnwindSafe(scenario);
            with_crash_at(point, move || {
                let wal = Wal::open(&wal_dir, writer_options(&scenario, 0)).unwrap();
                for group in planned_groups() {
                    write_group(&wal, &group);
                    report_finished(&progress, group.0);
                }
            })
        }
    };
    match result {
        CrashResult::Completed(()) => std::process::exit(0),
        _ => unreachable!("an injected crash exits in the panic hook"),
    }
}

/// Notes in `progress` that the group of `id` finished.
fn report_finished(progress: &Path, id: u64) {
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(progress)
        .unwrap();
    writeln!(file, "{id}").unwrap();
}

/// The group of the cut-back scenario whose write fails.
const FAILED_GROUP: u64 = 4;

/// Writes the first frames of `group`, then fails the write of the next one:
/// the writer cuts the group back to where it started.
fn write_a_failing_group(wal: &Wal, (id, records): &Group) {
    assert!(records.len() >= 3, "the group spans several frames");
    let mut group = wal.begin_group(TransactionId::new(*id)).unwrap();
    group.push(&record(&records[0])).unwrap();
    group.push(&record(&records[1])).unwrap();
    // The next push writes the second record's frame: that write fails.
    let failed = with_failure_at(1, || group.push(&record(&records[2])));
    assert!(failed.is_err(), "the injected write failure");
    drop(group);
    assert!(!wal.is_poisoned(), "the cut back worked");
}

/// A log of the planned groups over several segments, rotated once more so
/// its end is a segment start, as a checkpoint leaves it: the writer, and
/// that checkpoint LSN.
fn log_with_a_checkpoint(dir: &Path, scenario: &str) -> (Wal, u64) {
    let wal = Wal::open(dir, writer_options(scenario, 0)).unwrap();
    let planned = planned_groups();
    for group in &planned[..6] {
        write_group(&wal, group);
    }
    let checkpoint = wal.rotate().unwrap();
    for group in &planned[6..] {
        write_group(&wal, group);
    }
    (wal, checkpoint)
}

/// The groups the child reported as finished.
fn progress(base: &Path) -> Vec<u64> {
    std::fs::read_to_string(base.join("progress.txt"))
        .unwrap_or_default()
        .lines()
        .map(|line| line.parse().unwrap())
        .collect()
}

/// Cuts the torn tail of the log in `dir`, writes one more group, and checks
/// that it lands right after `groups`.
fn reopen_and_write_on(dir: &Path, scenario: &str, groups: &[Group], end: &ScanEnd, at: &str) {
    end.cut_torn_tail().unwrap();
    let wal = Wal::open(dir, writer_options(scenario, end.end_lsn)).unwrap();
    let more: Group = (319, vec!["Mia in Amsterdam".to_string()]);
    write_group(&wal, &more);
    drop(wal);
    let (after, end) = scan(dir, scenario);
    let mut expected = groups.to_vec();
    expected.push(more);
    assert_eq!(after, expected, "{at}: the log continues after the crash");
    assert!(end.is_clean(), "{at}: {end:?}");
}

/// Crashes `scenario` at every point in turn, until it completes, and checks
/// the log each crash leaves against `expected`, the groups it writes in
/// order: whole groups, never one that was not written, at most one more
/// than the child saw finish. Returns the crash points it crashed at.
fn sweep_writes(scenario: &str, expected: &[Group]) -> BTreeSet<String> {
    let mut crashed_at = BTreeSet::new();
    let mut crashes = 0;
    let mut cut_tails = 0;
    for point in 1..=MAX_POINTS {
        let base = tempfile::tempdir().unwrap();
        let crash = run_child(scenario, base.path(), point);
        let completed = crash.is_none();
        crashed_at.extend(crash);
        let at = format!("{scenario} at crash point {point}");
        let wal_dir = base.path().join("wal");
        if !wal_dir.exists() {
            assert!(!completed, "{at}");
            crashes += 1;
            continue;
        }
        let finished = progress(base.path());
        let (groups, end) = scan(&wal_dir, scenario);
        assert!(
            groups.len() <= expected.len() && groups[..] == expected[..groups.len()],
            "{at}: whole groups in order: {groups:?}"
        );
        assert!(
            groups.len() == finished.len() || groups.len() == finished.len() + 1,
            "{at}: the log holds the {} finished groups and at most the one whose LAST frame \
             was written when it crashed, not {}",
            finished.len(),
            groups.len()
        );
        if completed {
            assert_eq!(groups, expected, "{at}");
            assert!(end.is_clean(), "{at}: {end:?}");
            assert!(crashes > 25, "the sweep crashed at only {crashes} points");
            assert!(cut_tails > 8, "only {cut_tails} crashes left a torn tail");
            let segments = list_wal_directory(&wal_dir).unwrap().segments.len();
            assert!(segments > 3, "the scenario rotates: {segments} segments");
            return crashed_at;
        }
        crashes += 1;
        if !end.is_clean() {
            cut_tails += 1;
        }
        reopen_and_write_on(&wal_dir, scenario, &groups, &end, &at);
    }
    panic!("{scenario}: the child never completed");
}

/// The crash points of writing and rotating.
const WRITE_POINTS: [&str; 4] = [
    "wal:after_frame",
    "wal:after_segment_create",
    "wal:after_segment_header",
    "wal:before_dir_sync",
];

#[test]
fn a_crash_at_any_point_of_writing_leaves_exactly_the_groups_whose_last_frame_was_written() {
    let seen = sweep_writes("write", &planned_groups());
    assert_crashed_at("write", &seen, &WRITE_POINTS);
}

#[cfg(feature = "encryption")]
#[test]
fn a_crash_at_any_point_of_writing_an_encrypted_log_leaves_exactly_its_complete_groups() {
    let seen = sweep_writes("write-encrypted", &planned_groups());
    assert_crashed_at("write-encrypted", &seen, &WRITE_POINTS);
}

/// Sync mode syncs every group before `finish` returns: the sweep crashes
/// before and after each of those syncs as well.
#[test]
fn a_crash_at_any_point_of_writing_in_sync_mode_leaves_exactly_the_written_groups() {
    let seen = sweep_writes("write-sync", &planned_groups());
    assert_crashed_at("write-sync", &seen, &WRITE_POINTS);
    assert_crashed_at("write-sync", &seen, &["wal:before_sync", "wal:after_sync"]);
}

/// A group whose write fails is cut back to where it started: a crash
/// before, during or after that cut never leaves any of it in the log (it
/// has no LAST frame), and the later groups follow the earlier ones.
#[test]
fn a_crash_around_a_failed_groups_cut_back_never_leaves_part_of_it() {
    let expected: Vec<Group> = planned_groups()
        .into_iter()
        .filter(|(id, _)| *id != FAILED_GROUP)
        .collect();
    let seen = sweep_writes("cut-back", &expected);
    assert_crashed_at("cut-back", &seen, &["wal:after_cut_back"]);
}

/// A log with three complete groups and a torn fourth, in `base/wal`.
fn torn_log(base: &Path, scenario: &str) -> Vec<Group> {
    let dir = base.join("wal");
    let wal = Wal::open(&dir, writer_options(scenario, 0)).unwrap();
    let planned = planned_groups();
    for group in &planned[..3] {
        write_group(&wal, group);
    }
    let complete_end = wal.end_lsn();
    write_group(&wal, &planned[4]);
    drop(wal);
    let (_, path) = list_wal_directory(&dir).unwrap().segments.pop().unwrap();
    let length = std::fs::metadata(&path).unwrap().len();
    let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    file.set_len(length - 7).unwrap();
    let (groups, end) = scan(&dir, scenario);
    assert_eq!(groups, planned[..3]);
    assert_eq!(end.end_lsn, complete_end);
    assert!(end.tail.is_some(), "a torn tail: {end:?}");
    groups
}

#[test]
fn a_crash_while_cutting_the_torn_tail_leaves_a_log_that_scans_the_same() {
    let scenario = "cut";
    for point in 1..=MAX_POINTS {
        let base = tempfile::tempdir().unwrap();
        let groups = torn_log(base.path(), scenario);
        let completed = run_child(scenario, base.path(), point).is_none();
        let at = format!("cut at crash point {point}");
        let dir = base.path().join("wal");
        let (after, end) = scan(&dir, scenario);
        assert_eq!(after, groups, "{at}: the second open finds the same groups");
        if completed {
            assert!(end.is_clean(), "{at}: the cut completed: {end:?}");
            assert_eq!(
                point, 3,
                "a crash before and after the truncation, then none"
            );
            reopen_and_write_on(&dir, scenario, &groups, &end, &at);
            return;
        }
        reopen_and_write_on(&dir, scenario, &groups, &end, &at);
    }
    panic!("the cut never completed");
}

/// A log with three complete groups and, after them, a newest segment whose
/// header was cut off while it was created (all zero), in `base/wal`.
fn log_with_an_unfinished_segment(base: &Path, scenario: &str) -> (Vec<Group>, PathBuf) {
    let dir = base.join("wal");
    let wal = Wal::open(&dir, writer_options(scenario, 0)).unwrap();
    let planned = planned_groups();
    for group in &planned[..3] {
        write_group(&wal, group);
    }
    let end = wal.end_lsn();
    drop(wal);
    let unfinished = dir.join(segment_file_name(end));
    std::fs::write(&unfinished, [0u8; 128]).unwrap();
    let (groups, scanned) = scan(&dir, scenario);
    assert_eq!(groups, planned[..3]);
    assert_eq!(
        scanned.unfinished_segment.as_deref(),
        Some(unfinished.as_path())
    );
    (groups, unfinished)
}

#[test]
fn a_crash_while_removing_an_unfinished_segment_leaves_a_log_that_scans_the_same() {
    let scenario = "unfinished";
    let mut seen = BTreeSet::new();
    for point in 1..=MAX_POINTS {
        let base = tempfile::tempdir().unwrap();
        let (groups, unfinished) = log_with_an_unfinished_segment(base.path(), scenario);
        let crash = run_child(scenario, base.path(), point);
        let completed = crash.is_none();
        seen.extend(crash);
        let at = format!("unfinished at crash point {point}");
        let dir = base.path().join("wal");
        let (after, end) = scan(&dir, scenario);
        assert_eq!(after, groups, "{at}: the second open finds the same groups");
        assert_eq!(
            end.unfinished_segment.is_some(),
            unfinished.exists(),
            "{at}: an unfinished segment that is left is found again"
        );
        if completed {
            assert!(end.is_clean() && !unfinished.exists(), "{at}: {end:?}");
            assert_eq!(point, 3, "a crash before and after the removal, then none");
            assert_crashed_at(
                scenario,
                &seen,
                &[
                    "wal:before_remove_unfinished",
                    "wal:after_remove_unfinished",
                ],
            );
            reopen_and_write_on(&dir, scenario, &groups, &end, &at);
            return;
        }
        reopen_and_write_on(&dir, scenario, &groups, &end, &at);
    }
    panic!("the removal never completed");
}

/// Removing the segments below a checkpoint goes oldest first: a crash at
/// any point leaves the log from the checkpoint on whole, the segments left
/// below it are the newest of them, and the next removal finishes the job.
#[test]
fn a_crash_while_removing_segments_below_a_checkpoint_keeps_the_log_after_it() {
    let scenario = "remove";
    let reference = tempfile::tempdir().unwrap();
    let reference_dir = reference.path().join("wal");
    let (wal, checkpoint) = log_with_a_checkpoint(&reference_dir, scenario);
    let end_lsn = wal.end_lsn();
    drop(wal);
    let below: Vec<u64> = list_wal_directory(&reference_dir)
        .unwrap()
        .segments
        .iter()
        .map(|(lsn, _)| *lsn)
        .filter(|lsn| *lsn < checkpoint)
        .collect();
    assert!(below.len() > 2, "several segments to remove: {below:?}");
    let after_checkpoint = planned_groups()[6..].to_vec();
    let mut seen = BTreeSet::new();
    for point in 1..=MAX_POINTS {
        let base = tempfile::tempdir().unwrap();
        let crash = run_child(scenario, base.path(), point);
        let completed = crash.is_none();
        seen.extend(crash);
        let at = format!("remove at crash point {point}");
        let dir = base.path().join("wal");
        let left: Vec<u64> = list_wal_directory(&dir)
            .unwrap()
            .segments
            .iter()
            .map(|(lsn, _)| *lsn)
            .filter(|lsn| *lsn < checkpoint)
            .collect();
        assert_eq!(
            left[..],
            below[below.len() - left.len()..],
            "{at}: the oldest went first"
        );
        let (groups, end) = scan_from(&dir, scenario, checkpoint);
        assert_eq!(
            groups, after_checkpoint,
            "{at}: the log after the checkpoint"
        );
        assert!(end.is_clean() && end.end_lsn == end_lsn, "{at}: {end:?}");
        if completed {
            assert!(left.is_empty(), "{at}: {left:?}");
            assert_eq!(point, 2 * u64::try_from(below.len()).unwrap() + 1);
            assert_crashed_at(
                scenario,
                &seen,
                &["wal:before_remove_segment", "wal:after_remove_segment"],
            );
            return;
        }
        let wal = Wal::open(&dir, writer_options(scenario, end_lsn)).unwrap();
        wal.remove_segments_before(checkpoint).unwrap();
        let more: Group = (319, vec!["Mia in Amsterdam".to_string()]);
        write_group(&wal, &more);
        drop(wal);
        let remaining = list_wal_directory(&dir).unwrap().segments;
        assert_eq!(
            remaining[0].0, checkpoint,
            "{at}: the next removal finishes"
        );
        let (groups, _) = scan_from(&dir, scenario, checkpoint);
        let mut expected = after_checkpoint.clone();
        expected.push(more);
        assert_eq!(groups, expected, "{at}: the log continues");
    }
    panic!("the removal never completed");
}

/// A log over several segments with a damaged byte in the second group of
/// its first, sealed segment, in `base/wal`: the intact groups before the
/// damage, the damaged bytes from the cut on, and the later segments' bytes.
fn damaged_log(base: &Path, scenario: &str) -> (Vec<Group>, Vec<u8>, Vec<Vec<u8>>) {
    let dir = base.join("wal");
    let wal = Wal::open(&dir, writer_options(scenario, 0)).unwrap();
    let planned = planned_groups();
    write_group(&wal, &planned[0]);
    let cut = wal.end_lsn();
    for group in &planned[1..] {
        write_group(&wal, group);
    }
    drop(wal);
    let mut segments = list_wal_directory(&dir).unwrap().segments.into_iter();
    let (_, sealed) = segments.next().unwrap();
    let later: Vec<Vec<u8>> = segments
        .map(|(_, path)| std::fs::read(path).unwrap())
        .collect();
    assert!(later.len() > 2, "several later segments: {}", later.len());
    let mut bytes = std::fs::read(&sealed).unwrap();
    let offset = 128 + usize::try_from(cut).unwrap();
    assert!(
        bytes.len() > offset + 30,
        "the second group is in the sealed segment"
    );
    bytes[offset + 30] ^= 0x5A;
    std::fs::write(&sealed, &bytes).unwrap();
    (planned[..1].to_vec(), bytes[offset..].to_vec(), later)
}

/// Every file in the `damaged-*` directories of `dir`.
fn set_aside_files(dir: &Path) -> Vec<Vec<u8>> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        if entry.file_name().to_string_lossy().starts_with("damaged-") {
            for file in std::fs::read_dir(entry.path()).unwrap() {
                files.push(std::fs::read(file.unwrap().path()).unwrap());
            }
        }
    }
    files
}

/// Salvage copies the damaged bytes aside, moves the later segments after
/// them, then cuts the damaged segment. A crash at any step, followed by
/// the next open's salvage, keeps every byte it set aside: the damaged
/// bytes and each later segment are found whole in a `damaged-*` directory.
#[test]
fn a_crash_while_salvaging_keeps_every_byte_it_sets_aside() {
    let scenario = "salvage";
    let mut seen = BTreeSet::new();
    for point in 1..=MAX_POINTS {
        let base = tempfile::tempdir().unwrap();
        let (groups, damaged, later) = damaged_log(base.path(), scenario);
        let crash = run_child(scenario, base.path(), point);
        let completed = crash.is_none();
        seen.extend(crash);
        let at = format!("salvage at crash point {point}");
        let dir = base.path().join("wal");
        // The next open salvages again, without crashing.
        let (after, end) = scan(&dir, scenario);
        assert_eq!(after, groups, "{at}: the intact groups");
        end.cut_torn_tail().unwrap();
        let kept = set_aside_files(&dir);
        assert!(kept.contains(&damaged), "{at}: the damaged bytes are kept");
        for (index, segment) in later.iter().enumerate() {
            assert!(
                kept.contains(segment),
                "{at}: later segment {index} is kept"
            );
        }
        let (after, end) = scan(&dir, scenario);
        assert_eq!(after, groups, "{at}");
        assert!(end.is_clean(), "{at}: {end:?}");
        reopen_and_write_on(&dir, scenario, &groups, &end, &at);
        if completed {
            assert_crashed_at(
                scenario,
                &seen,
                &[
                    "wal:salvage_dir",
                    "wal:salvage_copy",
                    "wal:salvage_move",
                    "wal:before_cut",
                    "wal:after_cut",
                ],
            );
            return;
        }
    }
    panic!("the salvage never completed");
}
